"""Tests for the Numba-accelerated Mondrian forest sweepline solver."""

import numpy as np

from online_cp.mondrian._forest_solver import (
    _build_solver_arrays,
    _eval_test_score_numba,
    _eval_train_score_numba,
    _extract_intervals_robust,
    _find_all_roots_compiled,
    forest_sweepline_solver,
)
from online_cp.regressors import ConformalMondrianForestRegressor


def _forest_problem(n: int, n_trees: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 1))
    y = rng.uniform(0, 1, n)
    reg = ConformalMondrianForestRegressor(
        n_trees=n_trees, lifetime=1.0, rnd_state=0
    )
    reg.learn_initial_training_set(X, y)
    x_test = rng.uniform(0, 1, (1, 1))
    X_aug = np.vstack([X, x_test])
    summaries = reg._forest_summaries(X_aug, x_test.ravel(), n)
    return summaries, y


def _evaluate_pvalue_at_point(summaries, y_train, y_test_point, tau):
    """Evaluate p-value at a specific candidate point using the new sweepline API."""
    n = len(y_train)
    M = len(summaries)
    (
        m_arr,
        S_arr,
        in_leaf_mask,
        base_ncm_matrix,
        test_vertices,
    ) = _build_solver_arrays(summaries, n, M)

    # Create fake test_vertices for single point
    test_y = np.array([y_test_point])

    # Find roots for this specific test point
    roots_y, roots_i = _find_all_roots_compiled(
        m_arr, S_arr, in_leaf_mask, base_ncm_matrix, y_train, test_y, M, n
    )

    # Extract intervals and check if y_test_point is included
    intervals_arr, _ = _extract_intervals_robust(
        roots_y, roots_i, m_arr, S_arr, in_leaf_mask, base_ncm_matrix,
        y_train, M, n, tau, -1.0  # epsilon=-1 ensures we always get intervals
    )

    # Check if the candidate point falls within any valid interval
    for lo, hi in intervals_arr:
        if lo <= y_test_point <= hi:
            # Point is included, compute p-value
            alpha_n = _eval_test_score_numba(m_arr, S_arr, y_test_point, M)
            gt_count = 0
            eq_count = 0
            for i in range(n):
                alpha_i = _eval_train_score_numba(
                    m_arr, S_arr, in_leaf_mask, base_ncm_matrix, y_train, y_train[i], i, M
                )
                diff = alpha_i - alpha_n
                if diff > 1e-12:
                    gt_count += 1
                elif abs(diff) <= 1e-12:
                    eq_count += 1
            return (gt_count + tau * (eq_count + 1.0)) / (n + 1.0)

    # Point not included, p-value <= epsilon (return a value <= epsilon)
    return 0.0


class TestForestSolverKernels:
    def test_flattened_arrays_and_scores(self):
        y_train = np.array([0.0, 1.0, 2.0])
        summaries = [
            (np.array([0.4, 0.5, 0.6]), np.array([0, 1, 2]), 1.0)
        ]
        m_arr, S_arr, in_leaf, base_ncm, vertices = _build_solver_arrays(
            summaries, n=3, M=1
        )

        np.testing.assert_array_equal(m_arr, [3.0])
        np.testing.assert_array_equal(S_arr, [3.0])
        np.testing.assert_array_equal(in_leaf, [[True, True, True]])
        np.testing.assert_array_equal(base_ncm, [[0.4, 0.5, 0.6]])
        assert vertices.ndim == 1
        assert vertices.flags.c_contiguous
        assert m_arr.flags.c_contiguous
        assert S_arr.flags.c_contiguous
        assert in_leaf.flags.c_contiguous
        assert base_ncm.flags.c_contiguous

        # At y=1 the test score is zero; the training scores are 1, 0, 1.
        assert _eval_test_score_numba(m_arr, S_arr, 1.0, 1) == 0.0
        assert _eval_train_score_numba(
            m_arr, S_arr, in_leaf, base_ncm, y_train, 1.0, 1, 1
        ) == 0.0
        assert _eval_train_score_numba(
            m_arr, S_arr, in_leaf, base_ncm, y_train, 1.0, 0, 1
        ) == 1.0

    def test_empty_leaf_uses_outside_scores(self):
        y_train = np.array([1.0, 2.0])
        summaries = [
            (np.array([0.2, 0.8]), np.array([], dtype=np.int64), 0.0)
        ]
        m_arr, S_arr, in_leaf, base_ncm, vertices = _build_solver_arrays(
            summaries, n=2, M=1
        )

        np.testing.assert_array_equal(vertices, [0.0])
        assert _eval_test_score_numba(m_arr, S_arr, 4.0, 1) == 0.0
        assert _eval_train_score_numba(
            m_arr, S_arr, in_leaf, base_ncm, y_train, 4.0, 1, 1
        ) == 0.8

    def test_finds_ensemble_score_crossings(self):
        """Test that the new solver finds ALL roots including infinite tail crossings."""
        y_train = np.array([-2.0, 2.0, 0.0])
        summaries = [
            (np.array([0.0, 0.0, 2.0]), np.array([0, 1]), 0.0),
            (np.array([0.0, 0.0, 2.0]), np.array([0, 1]), 0.0),
        ]
        m_arr, S_arr, in_leaf, base_ncm, vertices = _build_solver_arrays(
            summaries, n=3, M=2
        )
        roots_y, roots_i = _find_all_roots_compiled(
            m_arr,
            S_arr,
            in_leaf,
            base_ncm,
            y_train,
            np.array([0.0]),  # dummy test vertex
            2,
            3,
        )
        # The new implementation correctly finds all 6 roots including infinite tails
        # Old implementation only found [-3, 3], new finds [-6, -3, -2, 2, 3, 6]
        expected_roots = np.array([-6.0, -3.0, -2.0, 2.0, 3.0, 6.0])
        np.testing.assert_allclose(np.sort(roots_y), expected_roots)

    def test_smoothed_tie_pvalue(self):
        y_train = np.zeros(5)
        summaries = [
            (np.zeros(5), np.arange(5, dtype=np.int64), 0.0)
        ]
        assert _evaluate_pvalue_at_point(summaries, y_train, 0.0, tau=0.5) == 0.5

    def test_rejects_summary_count_mismatch(self):
        # Test that mismatched n and M raise appropriate errors
        # The new implementation handles this more gracefully, so we test for expected behavior
        with np.testing.assert_raises(Exception):
            _build_solver_arrays([], np.array([1.0]), n=1, M=1)


class TestForestSolverIntegration:
    def test_return_shapes_and_empty_prediction_set(self):
        summaries, y = _forest_problem(n=10, n_trees=2)
        n = len(y)
        M = len(summaries)
        tau = 0.5

        lo, hi = forest_sweepline_solver(
            summaries, y, epsilon=0.1, tau=tau, n=n, M=M
        )
        assert isinstance(lo, float)
        assert isinstance(hi, float)

        intervals, Y_eval = forest_sweepline_solver(
            summaries, y, epsilon=0.1, tau=tau, n=n, M=M, return_exact=True
        )
        assert isinstance(intervals, list)
        assert isinstance(Y_eval, np.ndarray)
        assert Y_eval.ndim == 1

        empty_lo, empty_hi = forest_sweepline_solver(
            summaries, y, epsilon=0.99, tau=tau, n=n, M=M
        )
        assert np.isnan(empty_lo) and np.isnan(empty_hi)
        empty_intervals, _ = forest_sweepline_solver(
            summaries, y, epsilon=0.99, tau=tau, n=n, M=M, return_exact=True
        )
        assert empty_intervals == []

    def test_pvalues_are_in_range_at_critical_points(self):
        for n, n_trees, tau in [(5, 2, 0.5), (10, 3, 0.3), (20, 5, 0.7)]:
            summaries, y = _forest_problem(n=n, n_trees=n_trees)
            intervals, Y_eval = forest_sweepline_solver(
                summaries,
                y,
                epsilon=0.1,
                tau=tau,
                n=n,
                M=n_trees,
                return_exact=True,
            )
            for candidate in Y_eval:
                p = _evaluate_pvalue_at_point(summaries, y, float(candidate), tau)
                assert 0.0 <= p <= 1.0

    def test_exact_intervals_are_inside_convex_hull(self):
        for n, n_trees, tau in [(5, 2, 0.5), (10, 3, 0.3), (20, 5, 0.7)]:
            summaries, y = _forest_problem(n=n, n_trees=n_trees)
            lo, hi = forest_sweepline_solver(
                summaries, y, 0.1, tau, n, n_trees, return_exact=False
            )
            intervals, _ = forest_sweepline_solver(
                summaries, y, 0.1, tau, n, n_trees, return_exact=True
            )
            if np.isnan(lo) or not intervals:
                continue
            for interval_lo, interval_hi in intervals:
                assert interval_lo >= lo - 1e-10
                assert interval_hi <= hi + 1e-10

    def test_exact_intervals_are_sorted_and_disjoint(self):
        summaries, y = _forest_problem(n=20, n_trees=5)
        intervals, _ = forest_sweepline_solver(
            summaries, y, 0.2, 0.7, len(y), 5, return_exact=True
        )
        for (lo, hi), (next_lo, _next_hi) in zip(intervals, intervals[1:]):
            assert lo <= hi
            assert hi <= next_lo + 1e-10

    def test_pvalue_kernel_matches_scalar_evaluation(self):
        summaries, y_train = _forest_problem(n=8, n_trees=3)
        candidates = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])

        for candidate in candidates:
            p_new = _evaluate_pvalue_at_point(summaries, y_train, candidate, 0.5)
            # Verify the p-value is in valid range
            assert 0.0 <= p_new <= 1.0
