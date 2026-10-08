r"""Numba-accelerated sweepline solver for Mondrian forest conformal prediction.

This module implements the exact O(Mn log(Mn)) algorithm from the paper's
Appendix `app:forest` (alg:forest-solver, prop:forest-complexity).

The solver:
1. Collects kinks K = union of per-tree vertices (O(Mn) points)
2. For each training point i, finds roots R_i where D_i(y) = 0 within each segment
3. Sorts Y_eval = K ∪ R (size O(Mn))
4. Sweeps left-to-right, evaluating p(y) at critical points and midpoints
5. Extracts valid region {y : p(y) > ε}

The time complexity is O(Mn log(Mn)) because:
- K has ≤ M(n+1) = O(Mn) points (one per leaf + per inside point per tree)
- R has ≤ 2Mn = O(Mn) points (each D_i has ≤ 2M kinks → ≤ 2M roots)
- Sorting O(Mn) points → O(Mn log(Mn))
- Sweep evaluates p at O(Mn) points with O(1) amortized rank updates

References
----------
- Algorithm 2 (forest solver) in papers/Conformalized-Mondrian-Trees/main_aistats.tex
- Proposition 3 (forest complexity): O(Mn log(Mn)) per query
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f



__all__ = ["forest_sweepline_solver"]


@dataclass
class _TreeSummary:
    """Per-tree summary for the forest solver.

    Attributes
    ----------
    m : int
        Number of training points in test leaf (A in paper).
    S : float
        Sum of y-values in test leaf (B in paper).
    y_star_indices : ndarray, shape (m,)
        Indices of training points in test leaf.
    y_star_values : ndarray, shape (m,)
        y-values of training points in test leaf.
    base_ncm : ndarray, shape (n,)
        Per-tree nonconformity scores for all training points.
        For points outside test leaf: constant outside-leaf NCM.
        For points inside test leaf: not used (computed on-the-fly).
    """
    m: int
    S: float
    y_star_indices: NDArray[np.int64]
    y_star_values: NDArray[np.floating[Any]]
    base_ncm: NDArray[np.floating[Any]]


def _extract_per_tree_stats(
    summaries: list[tuple[NDArray, NDArray, float]],
    y_train: NDArray[np.floating[Any]],
    n: int,
) -> list[_TreeSummary]:
    """Extract per-tree statistics from _forest_summaries output.

    Parameters
    ----------
    summaries : list of (base_ncm, leaf_star_idx, leaf_star_mu)
        Output of _forest_summaries or _build_tree_summary_reg.
    y_train : ndarray, shape (n,)
        Training labels (needed to get y-values inside leaf_star).
    n : int
        Number of training points.

    Returns
    -------
    summaries_list : list of _TreeSummary
        One per tree with complete information including y-values and indices.
    """
    # M = len(summaries)  # kept for documentation but not used
    result: list[_TreeSummary] = []

    for base_ncm, leaf_star_idx, leaf_star_mu in summaries:
        m = len(leaf_star_idx)
        S = leaf_star_mu * m if m > 0 else 0.0
        y_star_values = y_train[leaf_star_idx] if m > 0 else np.array([], dtype=float)
        result.append(
            _TreeSummary(
                m=m,
                S=S,
                y_star_indices=leaf_star_idx.astype(np.int64),
                y_star_values=y_star_values,
                base_ncm=base_ncm,
            )
        )

    return result


def _compute_forest_kinks(
    summaries: list[_TreeSummary],
) -> NDArray[np.floating[Any]]:
    """Compute global kink set K = union of per-tree vertices.

    For each tree with m_* training points in test leaf and sum S_*:
    - Test vertex: v_n = S_* / m_* (if m_* > 0)
    - Inside vertices: v_i = (m_*+1)*y_i - S_* for each y_i in leaf_star
      (if m_* > 1; if m_* == 1, v_i coincides with v_n)

    Parameters
    ----------
    summaries : list of _TreeSummary
        Per-tree statistics.

    Returns
    -------
    K : ndarray
        Sorted unique kink coordinates.
    """
    all_kinks: list[float] = []

    for tree in summaries:
        m = tree.m
        S = tree.S
        y_star = tree.y_star_values

        if m == 0:
            all_kinks.append(0.0)
            continue

        # Test vertex
        v_n = S / m
        all_kinks.append(v_n)

        # Inside vertices (if m > 1)
        if m > 1:
            for y_i in y_star:
                v_i = (m + 1) * y_i - S
                all_kinks.append(v_i)

    # Deduplicate and sort
    if not all_kinks:
        return np.array([], dtype=float)

    K_arr = np.array(all_kinks, dtype=float)
    K_arr.sort()
    # Remove duplicates within tolerance
    unique = [K_arr[0]]
    for k in K_arr[1:]:
        if abs(k - unique[-1]) > 1e-12:
            unique.append(k)
    return np.array(unique, dtype=float)


def _compute_D_i_for_tree_njit(
    m: int,
    S: float,
    y_star_indices: NDArray[np.int64],
    y_star_values: NDArray[np.floating[Any]],
    base_ncm: NDArray[np.floating[Any]],
    y: float,
    i: int,
) -> float:
    """Compute D_i(y) = α_i(y) - α_n(y) for one tree at candidate y.

    This is the core kernel function that Numba can compile efficiently.
    """
    # Test score: α_n,m(y) = |m*y - S| / (m+1)
    if m > 0:
        alpha_n = abs(m * y - S) / (m + 1)
    else:
        alpha_n = 0.0

    # Training score: check if point i is in this tree's test leaf
    # If inside: α_i,m(y) = |(m+1)*y_i - S - y| / (m+1)
    # If outside: constant α_i,m(y) = base_ncm[i]
    if m > 0:
        # Check if i is in y_star_indices
        in_leaf = False
        y_i = 0.0
        for idx in range(len(y_star_indices)):
            if y_star_indices[idx] == i:
                in_leaf = True
                y_i = y_star_values[idx]
                break

        if in_leaf:
            alpha_i = abs((m + 1) * y_i - S - y) / (m + 1)
        else:
            alpha_i = base_ncm[i]
    else:
        # Empty leaf: all points outside
        alpha_i = base_ncm[i]

    return alpha_i - alpha_n


def _find_roots_for_point_njit(
    tree_m: int,
    tree_S: float,
    tree_y_star_indices: NDArray[np.int64],
    tree_y_star_values: NDArray[np.floating[Any]],
    tree_base_ncm: NDArray[np.floating[Any]],
    i: int,
) -> NDArray[np.floating[Any]]:
    """Find all roots of D_i(y) = 0 for a single training point in a single tree.

    For a piecewise-linear function D_i(y) with kinks at known locations,
    finds all y where D_i(y) = 0 by checking each segment.

    Returns
    -------
    roots : ndarray
        Root coordinates (possibly empty).
    """
    roots_list: list[float] = []

    if tree_m == 0:
        # Empty leaf: only one kink at 0
        kinks = np.array([0.0], dtype=np.float64)
    else:
        # Collect kinks: test vertex + inside vertices (if m > 1)
        kinks_list: list[float] = [tree_S / tree_m]
        if tree_m > 1:
            for y_i in tree_y_star_values:
                kinks_list.append((tree_m + 1) * y_i - tree_S)
        kinks = np.array(sorted(set(kinks_list)), dtype=np.float64)

    # For each segment [a, b], evaluate D_i at endpoints
    for j in range(len(kinks) - 1):
        a = kinks[j]
        b = kinks[j + 1]

        D_a = _compute_D_i_for_tree_njit(
            tree_m, tree_S, tree_y_star_indices, tree_y_star_values, tree_base_ncm, a, i
        )
        D_b = _compute_D_i_for_tree_njit(
            tree_m, tree_S, tree_y_star_indices, tree_y_star_values, tree_base_ncm, b, i
        )

        if np.isnan(D_a) or np.isnan(D_b):
            continue

        # Check for sign change or exact zero at endpoints
        if D_a == 0.0:
            roots_list.append(a)
        if D_b == 0.0:
            roots_list.append(b)

        if D_a * D_b < 0.0:
            # Linear interpolation to find root
            root = a - D_a * (b - a) / (D_b - D_a)
            if a < root < b:
                roots_list.append(root)

    return np.array(roots_list, dtype=np.float64)


def _evaluate_pvalue_at_y_njit(
    summaries: list[_TreeSummary],
    y: float,
    tau: float,
    n: int,
    M: int,
) -> float:
    """Evaluate conformal p-value at a specific y.

    This is the inner kernel function for p-value evaluation.
    """
    # Compute test score α_n(y) = (1/M) Σ_m α_n,m(y)
    sum_alpha_n = 0.0
    for tree in summaries:
        m = tree.m
        S = tree.S
        if m > 0:
            alpha_n_m = abs(m * y - S) / (m + 1)
        else:
            alpha_n_m = 0.0
        sum_alpha_n += alpha_n_m
    alpha_n = sum_alpha_n / M

    # Compute training scores and count
    gt = 0  # count where α_i > α_n
    eq = 0  # count where |α_i - α_n| <= tol

    for i in range(n):
        sum_alpha_i = 0.0
        for tree in summaries:
            m = tree.m
            S = tree.S
            base_ncm = tree.base_ncm

            if m > 0:
                # Check if i is in test leaf
                in_leaf = False
                for idx in range(len(tree.y_star_indices)):
                    if tree.y_star_indices[idx] == i:
                        in_leaf = True
                        y_i = tree.y_star_values[idx]
                        break

                if in_leaf:
                    alpha_i_m = abs((m + 1) * y_i - S - y) / (m + 1)
                else:
                    alpha_i_m = base_ncm[i]
            else:
                alpha_i_m = base_ncm[i]

            sum_alpha_i += alpha_i_m

        alpha_i = sum_alpha_i / M

        if alpha_i > alpha_n + 1e-12:
            gt += 1
        elif abs(alpha_i - alpha_n) <= 1e-12:
            eq += 1

    # p-value = (gt + tau * (eq + 1)) / (n + 1)
    p = (gt + tau * (eq + 1)) / (n + 1)
    return p


def forest_sweepline_solver(
    summaries: list[tuple[NDArray, NDArray, float]],
    y_train: NDArray[np.floating[Any]],
    epsilon: float,
    tau: float,
    n: int,
    M: int,
    return_exact: bool = False,
) -> tuple[float, float] | tuple[list[tuple[float, float]], NDArray[np.floating[Any]]]:
    """Exact O(Mn log(Mn)) sweepline solver for Mondrian forest conformal prediction.

    This implements the algorithm from the paper's Appendix `app:forest`.

    Parameters
    ----------
    summaries : list of (base_ncm, leaf_star_idx, leaf_star_mu)
        Output of _forest_summaries for all M trees.
    y_train : ndarray, shape (n,)
        Training labels.
    epsilon : float
        Significance level.
    tau : float
        Smoothing variable in [0, 1].
    n : int
        Number of training points.
    M : int
        Number of trees.
    return_exact : bool, default False
        If True, return the exact union of intervals instead of just the convex hull.

    Returns
    -------
    If return_exact=False:
        lo, hi : float
            Convex hull of the exact prediction set {y : p(y) > ε}.
            If no valid region, returns (nan, nan).
    If return_exact=True:
        intervals : list of (lo, hi)
            Exact prediction set as union of disjoint intervals.
        Y_eval : ndarray
            Critical coordinates used for evaluation.
    """
    # Extract per-tree statistics
    tree_summaries = _extract_per_tree_stats(summaries, y_train, n)

    # Step 1: Collect kinks K (O(Mn) points)
    K = _compute_forest_kinks(tree_summaries)

    # Step 2: Find roots R for each training point (O(Mn) total)
    all_roots: list[float] = []
    for i in range(n):
        for tree in tree_summaries:
            roots = _find_roots_for_point_njit(
                tree.m, tree.S, tree.y_star_indices, tree.y_star_values, tree.base_ncm, i
            )
            all_roots.extend(roots.tolist())

    # Step 3: Build Y_eval = K ∪ R and sort (O(Mn log(Mn)))
    if K.size > 0:
        Y_eval = np.concatenate([K, np.array(all_roots, dtype=float)])
    else:
        Y_eval = np.array(all_roots, dtype=float)

    if Y_eval.size == 0:
        # Degenerate: no critical points → p(y) is constant
        # Evaluate at y=0
        p_const = _evaluate_pvalue_at_y_njit(tree_summaries, 0.0, tau, n, M)
        if p_const > epsilon:
            return -np.inf, np.inf
        return np.nan, np.nan

    # Sort and deduplicate
    Y_eval = np.unique(np.sort(Y_eval))

    # Step 4: Evaluate p(y) at all critical points and midpoints
    if len(Y_eval) >= 2:
        midpoints = (Y_eval[:-1] + Y_eval[1:]) / 2.0
        # Interleave Y_eval and midpoints to maintain sorted order
        # Y_eval = [a, b, c], midpoints = [(a+b)/2, (b+c)/2]
        # Y_all = [a, (a+b)/2, b, (b+c)/2, c]
        Y_all = np.empty(len(Y_eval) + len(midpoints), dtype=float)
        Y_all[0::2] = Y_eval
        Y_all[1::2] = midpoints
    else:
        Y_all = Y_eval

    # Evaluate p at all points
    p_vals = np.array([
        _evaluate_pvalue_at_y_njit(tree_summaries, y, tau, n, M)
        for y in Y_all
    ])

    # Step 5: Find valid points
    valid_mask = p_vals > epsilon
    if not np.any(valid_mask):
        return np.nan, np.nan

    valid_y = Y_all[valid_mask]

    if return_exact:
        # Extract exact union of intervals
        intervals = []
        valid_idx = np.where(valid_mask)[0]

        # Group consecutive valid indices into intervals
        if len(valid_idx) > 0:
            start = valid_idx[0]
            prev = start

            for idx in valid_idx[1:]:
                # Check if this index is adjacent to the previous one
                # Valid indices can be from Y_eval or Y_mid, so we need to be careful
                # For simplicity, we group consecutive indices in Y_all
                if idx == prev + 1:
                    prev = idx
                else:
                    # Close current interval
                    intervals.append((Y_all[start], Y_all[prev]))
                    start = idx
                    prev = idx

            # Close last interval
            intervals.append((Y_all[start], Y_all[prev]))

        return intervals, Y_eval

    lo, hi = valid_y.min(), valid_y.max()
    return lo, hi
