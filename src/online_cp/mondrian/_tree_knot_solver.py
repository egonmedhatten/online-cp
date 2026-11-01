r"""Numba-accelerated exact knot solver for a single Mondrian tree conformal predictor.

This is the single-tree analogue of :mod:`online_cp.mondrian._forest_solver`.
A Mondrian-tree regressor's smoothed conformal p-value

    p(y) = (#{i : alpha_i(y) > alpha_n(y)} + tau * #{ties} + tau) / (n + 1)

is a piecewise-linear function of y whose slope changes only at a finite set
of "knots".  The interval {y : p(y) > epsilon} is recovered by an exact
left-to-right sweep over those knots.

The pure-Python reference implementation lives in
:meth:`online_cp.regressors.ConformalMondrianTreeRegressor._build_knot_events`
and :meth:`online_cp.regressors.ConformalMondrianTreeRegressor._pvalue_from_sweep`.
On large datasets (a single leaf holding tens of thousands of points) that
reference implementation is dominated by a per-knot O(A) re-evaluation of the
inside-point scores, which is the bottleneck behind the "tree slower than
forest" symptom.  This module ports the same exact algebra to Numba so the
sweep runs in compiled code.

Mathematical correctness is preserved because:

* Ancestor selection is label-blind (unchanged from the engine).
* The knot construction and the sweep are bit-identical to the Python
  reference; a property test in ``tests/test_mondrian_tree.py`` verifies
  ``tree_knot_solver`` agrees with the Python reference on many random cases.
* The ``njit`` decorator falls back to a no-op when Numba is unavailable, so
  the package still works (just slower) without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    def njit(*args, **kwargs):
        """Identity fallback when Numba is unavailable."""
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


__all__ = ["tree_knot_solver"]

_TIE_TOL = 1e-9


@njit(cache=True)
def _build_knot_events_compiled(
    A: int, B: float, ext_sorted: NDArray[np.float64], y_star: NDArray[np.float64]
):
    """Build sorted (value, flag) knot events for a single tree.

    Mirrors :meth:`ConformalMondrianTreeRegressor._build_knot_events`.

    Outside-leaf points (constant scores) generate two knots each:
    ``B/A - ((A+1)/A) * alpha_i`` (flag +1, entering) and
    ``B/A + ((A+1)/A) * alpha_i`` (flag -1, exiting).

    Inside-leaf points (V-shaped scores) generate two knots each (when A > 1):
    ``y_j`` and ``(2B - (A+1)*y_j)/(A - 1)``; the smaller carries flag +1
    (enter G_j), the larger flag -1 (exit G_j).  When A == 1 the single
    inside point is permanently tied with the test point, contributing
    ``permanent_ties = 1`` and no knots.
    """
    max_knots = 2 * ext_sorted.shape[0] + 2 * y_star.shape[0]
    knots_val = np.empty(max_knots, dtype=np.float64)
    knots_flag = np.empty(max_knots, dtype=np.int64)
    nk = 0

    permanent_ties = 0

    # Outside-leaf points: constant scores, two knots each.
    if A > 0:
        base = B / A
        fac = (A + 1.0) / A
        for k in range(ext_sorted.shape[0]):
            a = ext_sorted[k]
            knots_val[nk] = base - fac * a
            knots_flag[nk] = 1
            nk += 1
            knots_val[nk] = base + fac * a
            knots_flag[nk] = -1
            nk += 1

    # Inside-leaf points.
    if A == 0:
        pass
    elif A == 1:
        permanent_ties = 1
    else:
        for j in range(y_star.shape[0]):
            y_j = y_star[j]
            refl = (2.0 * B - (A + 1.0) * y_j) / (A - 1.0)
            left = y_j if y_j < refl else refl
            right = refl if y_j < refl else y_j
            knots_val[nk] = left
            knots_flag[nk] = 1
            nk += 1
            knots_val[nk] = right
            knots_flag[nk] = -1
            nk += 1

    # Sort by knot value.  Order among coincident knots is immaterial: the
    # sweep evaluates knot validity at the knot *value* (value-independent)
    # and aggregates the flags as it passes the index, so any ordering of
    # equal values yields the same result.
    if nk > 1:
        order = np.argsort(knots_val[:nk])
        sv = np.empty(nk, dtype=np.float64)
        sf = np.empty(nk, dtype=np.int64)
        for t in range(nk):
            sv[t] = knots_val[order[t]]
            sf[t] = knots_flag[order[t]]
    else:
        sv = knots_val[:nk].copy()
        sf = knots_flag[:nk].copy()
    return sv, sf, permanent_ties


@njit(cache=True)
def _evaluate_pvalue_at_point_compiled(
    y: float,
    A: int,
    B: float,
    ext_sorted: NDArray[np.float64],
    y_star: NDArray[np.float64],
    n_train: int,
    tau: float,
    permanent_ties: int,
) -> float:
    """Smoothed conformal p-value at an arbitrary y.

    Mirrors :meth:`ConformalMondrianTreeRegressor._evaluate_pvalue_at_point`.
    """
    ncm_test = abs(A * y - B) / (A + 1.0)

    # Mutually-exclusive classification (matches the Python reference,
    # _pvalues_at, and the forest solver's state machine):
    #   gt: score - ncm_test >  _TIE_TOL   (strictly greater by > tol)
    #   eq: |score - ncm_test| <= _TIE_TOL (within tol, a "tie")
    #   lt: ncm_test - score  >  _TIE_TOL   (strictly less by > tol)
    # A point is in exactly one category.  This prevents double-counting
    # of "near-greater" points (score - ncm_test in (0, _TIE_TOL]) that
    # would otherwise be counted in BOTH gt and eq.
    gt = 0
    eq = 0
    for k in range(ext_sorted.shape[0]):
        a = ext_sorted[k]
        if a - ncm_test > _TIE_TOL:
            gt += 1
        if abs(a - ncm_test) <= _TIE_TOL:
            eq += 1

    if A > 0 and y_star.shape[0] > 0:
        ap1 = A + 1.0
        for j in range(y_star.shape[0]):
            inside_ncm = abs(ap1 * y_star[j] - B - y) / ap1
            if inside_ncm - ncm_test > _TIE_TOL:
                gt += 1
            if abs(inside_ncm - ncm_test) <= _TIE_TOL:
                eq += 1

    return (gt + tau * (eq + 1 + permanent_ties)) / (n_train + 1.0)


@njit(cache=True)
def _pvalue_from_sweep_compiled(
    knots_val: NDArray[np.float64],
    knots_flag: NDArray[np.int64],
    A: int,
    B: float,
    ext_sorted: NDArray[np.float64],
    y_star: NDArray[np.float64],
    n_train: int,
    permanent_ties: int,
    epsilon: float,
    tau: float,
):
    """Left-to-right sweep to find the p(y) > epsilon interval.

    Mirrors :meth:`ConformalMondrianTreeRegressor._pvalue_from_sweep`.

    Returns (lo, hi, valid_knots) where lo/hi are the interval bounds
    (possibly +/-inf, or NaN when no region is valid) and valid_knots is a
    list of knot values that are themselves valid singletons.
    """
    nk = knots_val.shape[0]

    # Edge case: no knots → p(y) is constant everywhere.
    if nk == 0:
        p_const = tau * (1 + permanent_ties) / (n_train + 1.0)
        if p_const > epsilon:
            return -np.inf, np.inf, np.empty(0, dtype=np.float64)
        else:
            peak = B / A if A > 0 else 0.0
            one = np.array([peak], dtype=np.float64)
            return peak, peak, one

    # Baseline tally at y_start (far left of all knots).
    y_start = knots_val[0] - 1.0
    ncm_test_start = abs(A * y_start - B) / (A + 1.0)
    greater_count = 0
    for k in range(ext_sorted.shape[0]):
        if ext_sorted[k] > ncm_test_start:
            greater_count += 1
    if A > 0 and y_star.shape[0] > 0:
        ap1 = A + 1.0
        for j in range(y_star.shape[0]):
            # Division (not reciprocal-multiplication) to match the Python
            # reference bit-for-bit under fastmath.
            ncm_j = abs(ap1 * y_star[j] - B - y_start) / ap1
            if ncm_j > ncm_test_start:
                greater_count += 1

    first_valid_idx = -1
    last_valid_idx = -1
    valid_knots = np.empty(nk + 1, dtype=np.float64)
    nvk = 0
    has_prev = False  # True once we have passed at least one knot

    for i in range(nk + 1):
        if i < nk:
            knot_val = knots_val[i]
            flag = knots_flag[i]
            has_knot = True
        else:
            has_knot = False  # rightmost end-cell

        # p-value for this cell (open interval), using the running count.
        p_val = (greater_count + tau * (1 + permanent_ties)) / (n_train + 1.0)

        if p_val > epsilon:
            if first_valid_idx == -1:
                first_valid_idx = i
            last_valid_idx = i

        # Check validity of the knot itself (only interior knots, i.e. when
        # both a previous knot and the current knot exist).
        if has_knot and has_prev:
            knot_p_val = _evaluate_pvalue_at_point_compiled(
                knot_val, A, B, ext_sorted, y_star, n_train, tau, permanent_ties
            )
            if knot_p_val > epsilon:
                valid_knots[nvk] = knot_val
                nvk += 1

        # Update for the next iteration.
        if has_knot:
            greater_count += flag
            has_prev = True
        else:
            break

    # No valid region at all.
    if first_valid_idx == -1 and nvk == 0:
        return np.nan, np.nan, np.empty(0, dtype=np.float64)

    # All cells invalid, only knots valid.
    if first_valid_idx == -1:
        if nvk == 1:
            return valid_knots[0], valid_knots[0], valid_knots[:nvk]
        else:
            return np.min(valid_knots[:nvk]), np.max(valid_knots[:nvk]), valid_knots[:nvk]

    # Determine bounds from first and last valid cells.
    if first_valid_idx == 0:
        lo = -np.inf
    else:
        lo = knots_val[first_valid_idx - 1]

    if last_valid_idx == nk:
        hi = np.inf
    else:
        hi = knots_val[last_valid_idx]

    return lo, hi, valid_knots[:nvk]


def tree_knot_solver(
    A: int,
    B: float,
    ext_sorted: NDArray[np.floating[Any]],
    y_star: NDArray[np.floating[Any]],
    n_train: int,
    epsilon: float,
    tau: float,
) -> tuple[float, float, list[float]]:
    """Exact O(k log k) knot solver for a single Mondrian tree.

    Computes the interval {y : p(y) > epsilon} for the tree's smoothed
    conformal p-value by building the knot events and sweeping them, using
    the Numba-compiled kernels.

    Parameters
    ----------
    A : int
        Number of training points in the test leaf.
    B : float
        Sum of y-values in the test leaf.
    ext_sorted : ndarray
        Sorted nonconformity scores of training points outside the test leaf.
    y_star : ndarray
        y-values of training points inside the test leaf.
    n_train : int
        Total number of training points.
    epsilon : float
        Threshold; the valid set is {y : p(y) > epsilon}.
    tau : float
        Randomization variable in [0, 1] for the smoothed p-value.

    Returns
    -------
    (lo, hi, valid_knots) :
        ``lo``/``hi`` are the interval bounds (possibly ``-inf``/``+inf``, or
        ``nan`` when no region is valid) and ``valid_knots`` is a list of knot
        values that are themselves valid singletons.
    """
    ext = np.ascontiguousarray(ext_sorted, dtype=np.float64)
    ys = np.ascontiguousarray(y_star, dtype=np.float64)
    knots_val, knots_flag, permanent_ties = _build_knot_events_compiled(
        int(A), float(B), ext, ys
    )
    lo, hi, vk = _pvalue_from_sweep_compiled(
        knots_val,
        knots_flag,
        int(A),
        float(B),
        ext,
        ys,
        int(n_train),
        int(permanent_ties),
        float(epsilon),
        float(tau),
    )
    valid_knots = [float(v) for v in vk] if vk.shape[0] > 0 else []
    return float(lo), float(hi), valid_knots
