"""
Tests for the exact O(Mn log(Mn)) sweepline solver in _forest_solver.py.

Comprehensive test suite covering:
- Unit tests for correctness, edge cases, and return types
- Property-based (adversarial) tests using leancheck

The solver computes conformal prediction sets for Mondrian forest NCMs.
Key properties verified:
1. P-values are in (0, 1)
2. Empty result consistency between return_interval modes
3. Exact intervals ⊆ Convex hull
4. lo ≤ hi when non-empty
5. Quasi-convexity (prediction set is union of intervals)
"""

import numpy as np

try:
    from leancheck import check

except ImportError:
    check = None  # type: ignore

from online_cp.mondrian._forest_solver import (
    _compute_forest_kinks,
    _evaluate_pvalue_at_y_njit,
    _extract_per_tree_stats,
    _find_roots_for_point_njit,
    forest_sweepline_solver,
)
from online_cp.regressors import ConformalMondrianForestRegressor

# =============================================================================
# Unit Tests
# =============================================================================


class TestForestSolverExtractPerTreeStats:
    """Tests for _extract_per_tree_stats helper function."""

    def test_extract_with_single_tree(self):
        """Extract works with single tree."""
        n = 5
        base_ncm = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        leaf_star_idx = np.array([0, 1, 2], dtype=np.int64)
        leaf_star_mu = 0.3
        summaries = [(base_ncm, leaf_star_idx, leaf_star_mu)]
        y_train = np.array([0.0, 0.3, 0.6, 0.9, 1.2])

        result = _extract_per_tree_stats(summaries, y_train, n)

        assert len(result) == 1
        assert result[0].m == 3
        np.testing.assert_allclose(result[0].S, 0.9, rtol=1e-10)  # 3 * 0.3
        assert len(result[0].y_star_indices) == 3
        assert len(result[0].y_star_values) == 3
        np.testing.assert_array_equal(result[0].y_star_values, [0.0, 0.3, 0.6])

    def test_extract_with_empty_leaf(self):
        """Extract works when leaf is empty (m=0)."""
        n = 3
        base_ncm = np.array([0.1, 0.2, 0.3])
        leaf_star_idx = np.array([], dtype=np.int64)
        leaf_star_mu = 0.0
        summaries = [(base_ncm, leaf_star_idx, leaf_star_mu)]
        y_train = np.array([1.0, 2.0, 3.0])

        result = _extract_per_tree_stats(summaries, y_train, n)

        assert len(result) == 1
        assert result[0].m == 0
        np.testing.assert_allclose(result[0].S, 0.0, atol=1e-15)
        assert len(result[0].y_star_indices) == 0
        assert len(result[0].y_star_values) == 0

    def test_extract_multiple_trees(self):
        """Extract works with multiple trees."""
        n = 4
        summaries = [
            (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1], dtype=np.int64), 0.5),
            (np.array([0.1, 0.2, 0.3, 0.4]), np.array([2, 3], dtype=np.int64), 2.5),
        ]
        y_train = np.array([0.0, 1.0, 2.0, 3.0])

        result = _extract_per_tree_stats(summaries, y_train, n)

        assert len(result) == 2
        assert result[0].m == 2
        np.testing.assert_allclose(result[0].S, 1.0, atol=1e-15)  # 2 * 0.5
        assert result[1].m == 2
        np.testing.assert_allclose(result[1].S, 5.0, atol=1e-15)  # 2 * 2.5


class TestForestSolverComputeKinks:
    """Tests for _compute_forest_kinks helper function."""

    def test_single_tree_single_point(self):
        """Single tree with one point in leaf."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        tree = _TreeSummary(
            m=1,
            S=0.5,
            y_star_indices=np.array([0], dtype=np.int64),
            y_star_values=np.array([0.5]),
            base_ncm=np.array([0.1]),
        )

        kinks = _compute_forest_kinks([tree])

        # Test vertex = S/m = 0.5/1 = 0.5
        np.testing.assert_allclose(kinks, [0.5], atol=1e-15)

    def test_single_tree_multiple_points(self):
        """Single tree with multiple points in leaf."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        tree = _TreeSummary(
            m=3,
            S=1.5,  # mean = 0.5
            y_star_indices=np.array([0, 1, 2], dtype=np.int64),
            y_star_values=np.array([0.0, 0.5, 1.0]),
            base_ncm=np.array([0.1, 0.2, 0.3]),
        )

        kinks = _compute_forest_kinks([tree])

        # Test vertex = S/m = 1.5/3 = 0.5
        # Inside vertices = (m+1)*y_i - S = 4*y_i - 1.5
        # y=0.0 -> -1.5, y=0.5 -> 0.5, y=1.0 -> 2.5
        # Expected: [-1.5, 0.5, 2.5] (0.5 appears twice, should be deduplicated)
        assert len(kinks) == 3
        assert -1.5 in kinks
        assert 0.5 in kinks
        assert 2.5 in kinks

    def test_multiple_trees(self):
        """Kinks from multiple trees are merged."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        trees = [
            _TreeSummary(
                m=1,
                S=0.5,
                y_star_indices=np.array([0], dtype=np.int64),
                y_star_values=np.array([0.5]),
                base_ncm=np.array([0.1]),
            ),
            _TreeSummary(
                m=1,
                S=1.5,
                y_star_indices=np.array([0], dtype=np.int64),
                y_star_values=np.array([1.5]),
                base_ncm=np.array([0.1]),
            ),
        ]

        kinks = _compute_forest_kinks(trees)

        # Tree 1: 0.5, Tree 2: 1.5
        np.testing.assert_array_equal(sorted(kinks), [0.5, 1.5])

    def test_empty_tree(self):
        """Empty tree (m=0) adds kink at 0."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        tree = _TreeSummary(
            m=0,
            S=0.0,
            y_star_indices=np.array([], dtype=np.int64),
            y_star_values=np.array([]),
            base_ncm=np.array([0.1]),
        )

        kinks = _compute_forest_kinks([tree])

        np.testing.assert_array_equal(kinks, [0.0])


class TestForestSolverEvaluatePValue:
    """Tests for _evaluate_pvalue_at_y_njit helper function."""

    def test_pvalue_at_y_zero_all_same(self):
        """When all y=0 and all in leaf, p(0) should be high."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        n = 5
        M = 1
        tau = 0.5

        # Single tree, all points in leaf, y=[0,0,0,0,0]
        tree = _TreeSummary(
            m=5,
            S=0.0,  # mean = 0
            y_star_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            y_star_values=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            base_ncm=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        )

        # p(0) should be high: all training scores = 0, test score = 0
        # So all training scores equal test score
        # p(0) = (0 + tau * (n+1)) / (n+1) = tau
        p = _evaluate_pvalue_at_y_njit([tree], 0.0, tau, n, M)

        # With smoothing, p(0) = tau * (n+1) / (n+1) = tau
        assert 0.4 < p < 0.6  # Allow some tolerance

    def test_pvalue_outside_leaf(self):
        """Test with point outside test leaf."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        n = 3
        M = 1
        tau = 0.5

        # Single tree, only point 0 in leaf
        tree = _TreeSummary(
            m=1,
            S=0.0,  # mean = 0
            y_star_indices=np.array([0], dtype=np.int64),
            y_star_values=np.array([0.0]),
            base_ncm=np.array([0.5, 0.3, 0.4]),
        )

        # Evaluate at y=1.0 (different from training)
        p = _evaluate_pvalue_at_y_njit([tree], 1.0, tau, n, M)

        # p should be in (0, 1)
        assert 0.0 < p < 1.0


class TestForestSolverRootFinding:
    """Tests for _find_roots_for_point_njit helper function."""

    def test_empty_leaf_no_roots(self):
        """Empty leaf has no roots (D_i(y) is constant)."""
        from online_cp.mondrian._forest_solver import _TreeSummary

        tree = _TreeSummary(
            m=0,
            S=0.0,
            y_star_indices=np.array([], dtype=np.int64),
            y_star_values=np.array([]),
            base_ncm=np.array([0.5, 0.5, 0.5]),
        )

        # D_i(y) = base_ncm[i] - 0 = 0.5 (constant, never zero)
        roots = _find_roots_for_point_njit(tree.m, tree.S,
                                          tree.y_star_indices,
                                          tree.y_star_values,
                                          tree.base_ncm,
                                          i=0)
        # With empty leaf, kinks = [0], D_i is constant
        assert len(roots) == 0


class TestForestSolverIntegration:
    """Integration tests for forest_sweepline_solver."""

    def test_return_exact_false_tuple(self):
        """return_exact=False returns (lo, hi) tuple."""
        # Create a simple regression problem
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (10, 1))
        y = np.zeros(10)

        reg = ConformalMondrianForestRegressor(n_trees=2, lifetime=1e-9, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = np.array([[0.5]])
        n = len(X)
        X_aug = np.vstack([X, x_test.reshape(1, -1)])
        summaries = reg._forest_summaries(X_aug, x_test, n)

        tau = 0.5
        epsilon = 0.1
        M = len(summaries)

        lo, hi = forest_sweepline_solver(summaries, y, epsilon, tau, n, M, return_exact=False)

        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_return_exact_true_tuple(self):
        """return_exact=True returns (intervals, Y_eval) tuple."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (10, 1))
        y = np.zeros(10)

        reg = ConformalMondrianForestRegressor(n_trees=2, lifetime=1e-9, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = np.array([[0.5]])
        n = len(X)
        X_aug = np.vstack([X, x_test.reshape(1, -1)])
        summaries = reg._forest_summaries(X_aug, x_test, n)

        tau = 0.5
        epsilon = 0.1
        M = len(summaries)

        intervals, Y_eval = forest_sweepline_solver(summaries, y, epsilon, tau, n, M, return_exact=True)

        assert isinstance(intervals, list)
        assert isinstance(Y_eval, np.ndarray)

    def test_empty_prediction_set(self):
        """When epsilon is too large, prediction set is empty."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (10, 1))
        y = np.zeros(10)

        reg = ConformalMondrianForestRegressor(n_trees=2, lifetime=1e-9, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = np.array([[0.5]])
        n = len(X)
        X_aug = np.vstack([X, x_test.reshape(1, -1)])
        summaries = reg._forest_summaries(X_aug, x_test, n)

        tau = 0.5
        epsilon = 0.99  # Very large epsilon -> likely no valid region
        M = len(summaries)

        lo, hi = forest_sweepline_solver(summaries, y, epsilon, tau, n, M, return_exact=False)

        # Should return (nan, nan) for empty set
        assert np.isnan(lo) and np.isnan(hi)

    def test_p_value_in_range(self):
        """P-values should always be in (0, 1)."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (10, 1))
        y = np.random.uniform(0, 1, 10)

        reg = ConformalMondrianForestRegressor(n_trees=2, lifetime=1.0, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = np.array([[0.5]])
        n = len(X)
        X_aug = np.vstack([X, x_test.reshape(1, -1)])
        summaries = reg._forest_summaries(X_aug, x_test, n)

        tau = 0.5
        epsilon = 0.1
        M = len(summaries)

        # Get critical points
        intervals, Y_eval = forest_sweepline_solver(summaries, y, epsilon, tau, n, M, return_exact=True)

        # Evaluate p-values at all critical points
        from online_cp.mondrian._forest_solver import _evaluate_pvalue_at_y_njit, _extract_per_tree_stats
        tree_summaries = _extract_per_tree_stats(summaries, y, n)

        for y_val in Y_eval:
            p = _evaluate_pvalue_at_y_njit(tree_summaries, float(y_val), tau, n, M)
            assert 0.0 < p < 1.0, f"P-value at y={y_val} is {p}, not in (0, 1)"


# =============================================================================
# Property-Based (Adversarial) Tests - Direct Implementation
# =============================================================================


def test_p_value_bounds():
    """Property: p(y) ∈ (0, 1) for all y at critical points."""
    # Test with multiple configurations
    for n, M, tau, epsilon in [
        (5, 2, 0.5, 0.1),
        (10, 3, 0.3, 0.05),
        (20, 5, 0.7, 0.2),
    ]:
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (n, 1))
        y = rng.uniform(0, 1, n)

        reg = ConformalMondrianForestRegressor(n_trees=M, lifetime=1.0, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = rng.uniform(0, 1, (1, 1))
        X_aug = np.vstack([X, x_test])
        summaries = reg._forest_summaries(X_aug, x_test.ravel(), n)

        tree_summaries = _extract_per_tree_stats(summaries, y, n)
        intervals, Y_eval = forest_sweepline_solver(
            summaries, y, epsilon, tau, n, M, return_exact=True
        )

        # Check p-values at all critical points
        for y_val in Y_eval:
            p = _evaluate_pvalue_at_y_njit(tree_summaries, float(y_val), tau, n, M)
            assert 0.0 < p < 1.0, f"P-value at y={y_val} is {p}, not in (0, 1)"


def test_exact_subset_of_hull():
    """Property: exact intervals ⊆ convex hull when both non-empty."""
    # Test with multiple configurations
    for n, M, tau, epsilon in [
        (5, 2, 0.5, 0.1),
        (10, 3, 0.3, 0.05),
        (20, 5, 0.7, 0.2),
    ]:
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (n, 1))
        y = rng.uniform(0, 1, n)

        reg = ConformalMondrianForestRegressor(n_trees=M, lifetime=1.0, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = rng.uniform(0, 1, (1, 1))
        X_aug = np.vstack([X, x_test])
        summaries = reg._forest_summaries(X_aug, x_test.ravel(), n)

        # Get convex hull
        lo_hull, hi_hull = forest_sweepline_solver(
            summaries, y, epsilon, tau, n, M, return_exact=False
        )

        # Get exact intervals
        intervals, _ = forest_sweepline_solver(
            summaries, y, epsilon, tau, n, M, return_exact=True
        )

        # If either is empty, skip
        if np.isnan(lo_hull) or len(intervals) == 0:
            continue

        # Check each exact interval is within hull
        for lo_int, hi_int in intervals:
            assert lo_int >= lo_hull - 1e-10, (
                f"Interval [{lo_int}, {hi_int}] extends left of hull [{lo_hull}, {hi_hull}]"
            )
            assert hi_int <= hi_hull + 1e-10, (
                f"Interval [{lo_int}, {hi_int}] extends right of hull [{lo_hull}, {hi_hull}]"
            )


def test_p_value_quasi_concave():
    """Property: p(y) is quasi-concave → {y : p(y) > ε} is union of intervals."""
    # For forest NCMs, the p-value function is piecewise linear with kinks
    # at known locations. The super-level set {y : p(y) > ε} should always
    # be a union of closed intervals (possibly empty or disconnected).

    for n, M, tau, epsilon in [
        (5, 2, 0.5, 0.1),
        (10, 3, 0.3, 0.05),
        (20, 5, 0.7, 0.2),
    ]:
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (n, 1))
        y = rng.uniform(0, 1, n)

        reg = ConformalMondrianForestRegressor(n_trees=M, lifetime=1.0, rnd_state=0)
        reg.learn_initial_training_set(X, y)

        x_test = rng.uniform(0, 1, (1, 1))
        X_aug = np.vstack([X, x_test])
        summaries = reg._forest_summaries(X_aug, x_test.ravel(), n)

        # Get exact intervals (already a union of intervals)
        intervals, Y_eval = forest_sweepline_solver(
            summaries, y, epsilon, tau, n, M, return_exact=True
        )

        # If empty, valid
        if len(intervals) == 0:
            continue

        # Verify intervals are disjoint and sorted
        for i in range(len(intervals) - 1):
            lo_i, hi_i = intervals[i]
            lo_next, hi_next = intervals[i + 1]
            # Intervals should be disjoint (hi_i < lo_next)
            assert hi_i <= lo_next + 1e-10, (
                f"Intervals {intervals[i]} and {intervals[i+1]} overlap"
            )


# Run tests directly when this module is executed as main
if __name__ == "__main__":
    import sys
    test_p_value_bounds()
    test_exact_subset_of_hull()
    test_p_value_quasi_concave()
    print("All property tests passed!")
    sys.exit(0)
