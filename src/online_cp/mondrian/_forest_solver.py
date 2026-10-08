r"""Numba-accelerated sweepline solver for Mondrian forest conformal prediction.

The solver collects the score kinks and per-training-point score crossings,
evaluates conformal p-values at those points and their midpoints, and returns
the region where the p-value exceeds ``epsilon``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


__all__ = ["forest_sweepline_solver"]


@njit(fastmath=True)
def _eval_test_score_numba(m_arr, S_arr, y, M):
    """Evaluate the forest's test nonconformity score at ``y``."""
    score = 0.0
    for tree_idx in range(M):
        m = m_arr[tree_idx]
        if m > 0.0:
            score += abs(m * y - S_arr[tree_idx]) / (m + 1.0)
    return score / M


@njit(fastmath=True)
def _eval_train_score_numba(
    m_arr, S_arr, in_leaf_mask, base_ncm_matrix, y_train, y, i, M
):
    """Evaluate training point ``i``'s forest nonconformity score at ``y``."""
    score = 0.0
    for tree_idx in range(M):
        m = m_arr[tree_idx]
        if in_leaf_mask[tree_idx, i]:
            score += abs((m + 1.0) * y_train[i] - S_arr[tree_idx] - y) / (m + 1.0)
        else:
            score += base_ncm_matrix[tree_idx, i]
    return score / M


@njit(fastmath=True)
def _find_point_roots_numba(
    m_arr,
    S_arr,
    in_leaf_mask,
    base_ncm_matrix,
    y_train,
    i,
    test_vertices,
    M,
):
    """Find crossings of the test score and training point ``i``'s score."""
    capacity = len(test_vertices) + M
    kinks = np.empty(capacity, dtype=np.float64)
    num_kinks = 0

    for kink_idx in range(len(test_vertices)):
        kinks[num_kinks] = test_vertices[kink_idx]
        num_kinks += 1

    for tree_idx in range(M):
        if in_leaf_mask[tree_idx, i]:
            kinks[num_kinks] = (
                (m_arr[tree_idx] + 1.0) * y_train[i] - S_arr[tree_idx]
            )
            num_kinks += 1

    kinks = np.sort(kinks[:num_kinks])
    if num_kinks < 2:
        return np.empty(0, dtype=np.float64)

    roots = np.empty(2 * (num_kinks - 1), dtype=np.float64)
    num_roots = 0
    for kink_idx in range(num_kinks - 1):
        a = kinks[kink_idx]
        b = kinks[kink_idx + 1]
        if b <= a:
            continue

        D_a = _eval_train_score_numba(
            m_arr, S_arr, in_leaf_mask, base_ncm_matrix, y_train, a, i, M
        ) - _eval_test_score_numba(m_arr, S_arr, a, M)
        D_b = _eval_train_score_numba(
            m_arr, S_arr, in_leaf_mask, base_ncm_matrix, y_train, b, i, M
        ) - _eval_test_score_numba(m_arr, S_arr, b, M)

        if D_a == 0.0:
            roots[num_roots] = a
            num_roots += 1
        if D_b == 0.0:
            roots[num_roots] = b
            num_roots += 1
        if D_a * D_b < 0.0:
            root = a - D_a * (b - a) / (D_b - D_a)
            if a < root < b:
                roots[num_roots] = root
                num_roots += 1

    return roots[:num_roots]


@njit(fastmath=True)
def _evaluate_pvalues_compiled(
    Y_all,
    m_arr,
    S_arr,
    in_leaf_mask,
    base_ncm_matrix,
    y_train,
    tau,
    n,
    M,
):
    """Evaluate smoothed conformal p-values across candidate labels."""
    p_vals = np.empty(len(Y_all), dtype=np.float64)
    for idx in range(len(Y_all)):
        y = Y_all[idx]
        alpha_n = _eval_test_score_numba(m_arr, S_arr, y, M)
        gt = 0
        eq = 0
        for i in range(n):
            alpha_i = _eval_train_score_numba(
                m_arr,
                S_arr,
                in_leaf_mask,
                base_ncm_matrix,
                y_train,
                y,
                i,
                M,
            )
            if alpha_i > alpha_n + 1e-12:
                gt += 1
            elif abs(alpha_i - alpha_n) <= 1e-12:
                eq += 1
        p_vals[idx] = (gt + tau * (eq + 1.0)) / (n + 1.0)
    return p_vals


def _build_solver_arrays(
    summaries: list[tuple[NDArray, NDArray, float]],
    y_train: NDArray[np.floating[Any]],
    n: int,
    M: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Flatten tree summaries into contiguous arrays and collect score kinks."""
    m_arr = np.zeros(M, dtype=np.float64)
    S_arr = np.zeros(M, dtype=np.float64)
    in_leaf_mask = np.zeros((M, n), dtype=np.bool_)
    base_ncm_matrix = np.zeros((M, n), dtype=np.float64)
    vertices = np.zeros(M + sum(len(summary[1]) for summary in summaries))
    num_vertices = 0

    if len(summaries) != M:
        raise ValueError(f"Expected {M} tree summaries, got {len(summaries)}")

    for tree_idx, (base_ncm, leaf_star_idx, leaf_star_mu) in enumerate(summaries):
        leaf_indices = np.asarray(leaf_star_idx, dtype=np.int64)
        m = len(leaf_indices)
        m_arr[tree_idx] = m
        S_arr[tree_idx] = leaf_star_mu * m if m > 0 else 0.0
        base_ncm_matrix[tree_idx] = np.asarray(base_ncm, dtype=np.float64)

        if m == 0:
            vertices[num_vertices] = 0.0
            num_vertices += 1
            continue

        vertices[num_vertices] = S_arr[tree_idx] / m
        num_vertices += 1
        for i in leaf_indices:
            in_leaf_mask[tree_idx, i] = True
            vertices[num_vertices] = (m + 1.0) * y_train[i] - S_arr[tree_idx]
            num_vertices += 1

    test_vertices = np.unique(np.sort(vertices[:num_vertices]))
    return (
        np.ascontiguousarray(m_arr),
        np.ascontiguousarray(S_arr),
        np.ascontiguousarray(in_leaf_mask),
        np.ascontiguousarray(base_ncm_matrix),
        np.ascontiguousarray(test_vertices),
    )


def forest_sweepline_solver(
    summaries: list[tuple[NDArray, NDArray, float]],
    y_train: NDArray[np.floating[Any]],
    epsilon: float,
    tau: float,
    n: int,
    M: int,
    return_exact: bool = False,
) -> tuple[float, float] | tuple[list[tuple[float, float]], NDArray[np.floating[Any]]]:
    """Solve the Mondrian forest conformal prediction set by a sweepline.

    Parameters
    ----------
    summaries : list of (base_ncm, leaf_star_idx, leaf_star_mu)
        Output of ``_forest_summaries`` for all ``M`` trees.
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
        If True, return disjoint intervals instead of their convex hull.

    Returns
    -------
    If ``return_exact`` is False, return the convex hull ``(lo, hi)`` of the
    region where ``p(y) > epsilon``; an empty region is ``(nan, nan)``.
    Otherwise return ``(intervals, Y_eval)``, where ``Y_eval`` contains the
    critical coordinates used in the sweep.
    """
    (
        m_arr,
        S_arr,
        in_leaf_mask,
        base_ncm_matrix,
        test_vertices,
    ) = _build_solver_arrays(summaries, y_train, n, M)

    all_roots: list[float] = []
    for i in range(n):
        point_roots = _find_point_roots_numba(
            m_arr,
            S_arr,
            in_leaf_mask,
            base_ncm_matrix,
            y_train,
            i,
            test_vertices,
            M,
        )
        all_roots.extend(point_roots.tolist())

    if test_vertices.size:
        Y_eval = np.concatenate((test_vertices, np.asarray(all_roots, dtype=float)))
    else:
        Y_eval = np.asarray(all_roots, dtype=float)
    Y_eval = np.unique(np.sort(Y_eval))

    if Y_eval.size == 0:
        p_const = _evaluate_pvalues_compiled(
            np.array([0.0]),
            m_arr,
            S_arr,
            in_leaf_mask,
            base_ncm_matrix,
            y_train,
            tau,
            n,
            M,
        )[0]
        if p_const > epsilon:
            return -np.inf, np.inf
        return np.nan, np.nan

    if len(Y_eval) >= 2:
        midpoints = (Y_eval[:-1] + Y_eval[1:]) / 2.0
        Y_all = np.empty(len(Y_eval) + len(midpoints), dtype=float)
        Y_all[0::2] = Y_eval
        Y_all[1::2] = midpoints
    else:
        Y_all = Y_eval

    p_vals = _evaluate_pvalues_compiled(
        Y_all,
        m_arr,
        S_arr,
        in_leaf_mask,
        base_ncm_matrix,
        y_train,
        tau,
        n,
        M,
    )
    valid_mask = p_vals > epsilon
    if not np.any(valid_mask):
        if return_exact:
            return [], Y_eval
        return np.nan, np.nan

    valid_y = Y_all[valid_mask]
    if not return_exact:
        return float(valid_y.min()), float(valid_y.max())

    valid_idx = np.flatnonzero(valid_mask)
    intervals: list[tuple[float, float]] = []
    start = previous = valid_idx[0]
    for idx in valid_idx[1:]:
        if idx != previous + 1:
            intervals.append((float(Y_all[start]), float(Y_all[previous])))
            start = idx
        previous = idx
    intervals.append((float(Y_all[start]), float(Y_all[previous])))
    return intervals, Y_eval
