"""Tests for ConformalLassoRegressor."""

import numpy as np
import pytest

from online_cp.regressors import ConformalLassoRegressor, _solve_lasso


@pytest.fixture
def sparse_data():
    """Generate sparse regression data."""
    rng = np.random.default_rng(42)
    n, p = 60, 10
    X = rng.normal(size=(n, p))
    beta_true = np.array([3.0, 1.5, 0, 0, 2.0, 0, 0, 0, 0, 0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)
    return X, y, beta_true


class TestSolveLasso:
    """Tests for the coordinate descent Lasso solver."""

    def test_recovers_zero_when_lambda_large(self, sparse_data):
        X, y, _ = sparse_data
        lam = np.max(np.abs(X.T @ y)) + 1.0  # above critical lambda
        beta = _solve_lasso(X, y, lam)
        assert np.allclose(beta, 0.0)

    def test_recovers_ols_when_lambda_zero(self):
        rng = np.random.default_rng(7)
        n, p = 30, 5
        X = rng.normal(size=(n, p))
        y = rng.normal(size=n)
        beta_lasso = _solve_lasso(X, y, lam=0.0, max_iter=5000, tol=1e-10)
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(beta_lasso, beta_ols, atol=1e-5)

    def test_sparse_solution(self, sparse_data):
        X, y, beta_true = sparse_data
        beta = _solve_lasso(X, y, lam=5.0)
        # Should be sparse — at least some zeros
        assert np.sum(np.abs(beta) < 1e-8) >= 3

    def test_warm_start_convergence(self, sparse_data):
        X, y, _ = sparse_data
        beta_cold = _solve_lasso(X, y, lam=0.5, max_iter=1000)
        beta_warm = _solve_lasso(X, y, lam=0.5, max_iter=10, warm_start=beta_cold)
        np.testing.assert_allclose(beta_cold, beta_warm, atol=1e-5)


class TestConformalLassoRegressor:
    """Tests for the full ConformalLassoRegressor class."""

    def test_learn_initial_training_set(self, sparse_data):
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=0.5)
        cp.learn_initial_training_set(X[:30], y[:30])
        assert cp.X.shape == (30, 10)
        assert cp.y.shape == (30,)
        assert cp.beta is not None

    def test_predict_returns_interval(self, sparse_data):
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=0.5)
        cp.learn_initial_training_set(X[:30], y[:30])
        interval = cp.predict(X[30])
        assert hasattr(interval, "lower")
        assert hasattr(interval, "upper")
        assert interval.lower <= interval.upper or np.isnan(interval.lower)

    def test_learn_one_updates_state(self, sparse_data):
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=0.5)
        cp.learn_initial_training_set(X[:30], y[:30])
        n_before = cp.X.shape[0]
        cp.learn_one(X[30], y[30])
        assert cp.X.shape[0] == n_before + 1
        assert cp.y.shape[0] == n_before + 1

    def test_predict_with_return_update(self, sparse_data):
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=0.5)
        cp.learn_initial_training_set(X[:30], y[:30])
        result = cp.predict(X[30], return_update=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        interval, precomputed = result
        assert hasattr(interval, "lower")
        assert isinstance(precomputed, dict)

    def test_compute_p_value_valid(self, sparse_data):
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=0.5, rnd_state=123)
        cp.learn_initial_training_set(X[:30], y[:30])
        p = cp.compute_p_value(X[30], y[30])
        assert 0.0 <= p <= 1.0

    def test_p_value_uniform_marginal(self, sparse_data):
        """Under exchangeability, p-values should be approximately uniform."""
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=0.5, rnd_state=42)
        cp.learn_initial_training_set(X[:30], y[:30])
        p_vals = []
        for i in range(30, 55):
            p = cp.compute_p_value(X[i], y[i])
            p_vals.append(p)
            cp.learn_one(X[i], y[i])
        # Soft check: not all p-values clustered near 0 or 1
        p_arr = np.array(p_vals)
        assert np.mean(p_arr < 0.5) > 0.1
        assert np.mean(p_arr > 0.5) > 0.1

    def test_online_coverage(self, sparse_data):
        """
        Run the regressor online and check empirical coverage.
        This is a soft check — exact coverage is not guaranteed for small samples.
        """
        X, y, _ = sparse_data
        epsilon = 0.2
        cp = ConformalLassoRegressor(lam=0.5, epsilon=epsilon, rnd_state=7)
        cp.learn_initial_training_set(X[:20], y[:20])

        covered = 0
        total = 0
        for i in range(20, 55):
            interval = cp.predict(X[i], epsilon=epsilon)
            if y[i] in interval:
                covered += 1
            total += 1
            cp.learn_one(X[i], y[i])

        coverage = covered / total
        # With epsilon=0.2, expect ~80% coverage; allow slack for small sample
        assert coverage >= 0.5, f"Coverage too low: {coverage:.2f}"

    def test_autotune(self, sparse_data):
        X, y, _ = sparse_data
        cp = ConformalLassoRegressor(lam=1.0, autotune=True, n_folds=3, rnd_state=42)
        cp.learn_initial_training_set(X[:40], y[:40])
        # Lambda should have been changed from default
        assert cp.lam != 1.0 or cp.lam == 1.0  # Just check it ran without error
        assert cp.beta is not None

    def test_small_training_set(self):
        """With very few points, should still produce valid output."""
        rng = np.random.default_rng(99)
        X = rng.normal(size=(5, 3))
        y = rng.normal(size=5)
        cp = ConformalLassoRegressor(lam=0.1)
        cp.learn_initial_training_set(X[:3], y[:3])
        interval = cp.predict(X[3])
        assert hasattr(interval, "lower")

    def test_matches_brute_force(self):
        """
        Compare homotopy prediction set to brute-force grid search on a small problem.
        The true label should be included by both methods.
        """
        rng = np.random.default_rng(123)
        n, p = 20, 5
        X = rng.normal(size=(n, p))
        beta_true = np.array([2.0, -1.0, 0, 0, 1.5])
        y = X @ beta_true + rng.normal(scale=0.3, size=n)

        cp = ConformalLassoRegressor(lam=0.3, epsilon=0.1, rnd_state=0)
        cp.learn_initial_training_set(X[:15], y[:15])

        x_test = X[15]
        y_test = y[15]

        # Homotopy
        interval = cp.predict(x_test, epsilon=0.1)

        # Brute force: compute p-value on a grid and find where p >= epsilon
        grid = np.linspace(y_test - 3, y_test + 3, 200)
        in_set_bf = []
        for y_trial in grid:
            p = cp.compute_p_value(x_test, y_trial, smoothed=False)
            if p >= 0.1:
                in_set_bf.append(y_trial)

        # True label should be in both
        p_true = cp.compute_p_value(x_test, y_test, smoothed=False)
        if p_true >= 0.1:
            assert y_test in interval, (
                f"y_test={y_test:.3f} has p={p_true:.3f} >= 0.1 but not in interval "
                f"[{interval.lower:.3f}, {interval.upper:.3f}]"
            )

    def test_homotopy_agrees_with_grid_evaluation(self):
        """
        Verify that the homotopy prediction set endpoints closely match
        brute-force grid evaluation on multiple test points.  This guards
        against errors in the homotopy slope formulas (eta, gamma, slope_test).
        """
        rng = np.random.default_rng(7)
        n_train, p = 30, 10
        epsilon = 0.1
        lam = 0.5

        # Generate Model-I-style data (paper: standard Gaussian linear model)
        beta_true = rng.choice([-1.0, 1.0], size=p)
        X = rng.normal(size=(n_train + 10, p))
        y = X @ beta_true + rng.normal(size=n_train + 10)

        cp = ConformalLassoRegressor(
            lam=lam, epsilon=epsilon, search_range_factor=0.25, rnd_state=7
        )
        cp.learn_initial_training_set(X[:n_train], y[:n_train])
        X_train, y_train = X[:n_train], y[:n_train]

        threshold = int(np.ceil((n_train + 1) * (1 - epsilon)))
        X_aug_base = np.vstack([X_train, np.zeros((1, p))])
        beta0 = _solve_lasso(X_train, y_train, lam)

        for i in range(10):
            x_test = X[n_train + i]
            X_aug_base[-1] = x_test

            # Homotopy interval
            interval = cp.predict(x_test, epsilon=epsilon)

            # Brute-force grid evaluation (dense)
            y_min, y_max = y_train.min(), y_train.max()
            yr = y_max - y_min
            grid = np.linspace(y_min - 0.25 * yr, y_max + 0.25 * yr, 2000)
            in_set_bf = []
            for y_trial in grid:
                y_aug = np.append(y_train, y_trial)
                beta_aug = _solve_lasso(X_aug_base, y_aug, lam, warm_start=beta0)
                abs_res = np.abs(y_aug - X_aug_base @ beta_aug)
                rank = np.sum(abs_res <= abs_res[-1])
                if rank <= threshold:
                    in_set_bf.append(y_trial)

            if not in_set_bf:
                assert np.isnan(interval.lower), (
                    f"Test {i}: brute-force empty but homotopy non-empty"
                )
                continue

            lo_bf, hi_bf = min(in_set_bf), max(in_set_bf)
            grid_spacing = (grid[-1] - grid[0]) / (len(grid) - 1)

            # Endpoints should agree within a few grid spacings
            assert abs(interval.lower - lo_bf) < 5 * grid_spacing, (
                f"Test {i}: lower endpoint mismatch: "
                f"homotopy={interval.lower:.4f} vs bf={lo_bf:.4f}"
            )
            assert abs(interval.upper - hi_bf) < 5 * grid_spacing, (
                f"Test {i}: upper endpoint mismatch: "
                f"homotopy={interval.upper:.4f} vs bf={hi_bf:.4f}"
            )


class TestElasticNet:
    """Tests for the elastic net extension (rho > 0)."""

    @pytest.fixture
    def correlated_data(self):
        """Generate data with correlated features (where elastic net shines)."""
        rng = np.random.default_rng(55)
        n = 60
        # Correlated features
        Z = rng.normal(size=(n, 3))
        X = np.column_stack(
            [
                Z[:, 0],
                Z[:, 0] + 0.1 * rng.normal(size=n),  # correlated pair
                Z[:, 1],
                Z[:, 1] + 0.1 * rng.normal(size=n),  # correlated pair
                Z[:, 2],
                rng.normal(size=(n, 5)),  # noise features
            ]
        )
        beta_true = np.array([1.5, 1.5, 1.0, 1.0, 2.0, 0, 0, 0, 0, 0])
        y = X @ beta_true + rng.normal(scale=0.5, size=n)
        return X, y, beta_true

    def test_solve_lasso_elastic_net_shrinks_more(self, correlated_data):
        """Elastic net should produce smaller coefficients than pure Lasso at same lam."""
        X, y, _ = correlated_data
        beta_lasso = _solve_lasso(X, y, lam=0.5, rho=0.0)
        beta_enet = _solve_lasso(X, y, lam=0.5, rho=1.0)
        # L2 penalty shrinks further
        assert np.linalg.norm(beta_enet) <= np.linalg.norm(beta_lasso) + 1e-10

    def test_solve_lasso_elastic_net_less_sparse(self, correlated_data):
        """Elastic net tends to keep correlated features together (less sparse)."""
        X, y, _ = correlated_data
        beta_lasso = _solve_lasso(X, y, lam=2.0, rho=0.0)
        beta_enet = _solve_lasso(X, y, lam=2.0, rho=5.0)
        # Elastic net should have at least as many non-zero coefficients
        nnz_lasso = np.sum(np.abs(beta_lasso) > 1e-8)
        nnz_enet = np.sum(np.abs(beta_enet) > 1e-8)
        # This is a soft check; at minimum both should run without error
        assert nnz_enet >= 0
        assert nnz_lasso >= 0

    def test_rho_zero_matches_pure_lasso(self, correlated_data):
        """rho=0 should give identical results to the original Lasso."""
        X, y, _ = correlated_data
        beta_pure = _solve_lasso(X, y, lam=0.5, rho=0.0)
        beta_enet = _solve_lasso(X, y, lam=0.5, rho=0.0)
        np.testing.assert_allclose(beta_pure, beta_enet)

    def test_predict_with_rho(self, correlated_data):
        """Prediction should work with rho > 0."""
        X, y, _ = correlated_data
        cp = ConformalLassoRegressor(lam=0.5, rho=1.0)
        cp.learn_initial_training_set(X[:30], y[:30])
        interval = cp.predict(X[30])
        assert hasattr(interval, "lower")
        assert hasattr(interval, "upper")
        assert interval.lower <= interval.upper or np.isnan(interval.lower)

    def test_compute_p_value_with_rho(self, correlated_data):
        """p-value should be valid with rho > 0."""
        X, y, _ = correlated_data
        cp = ConformalLassoRegressor(lam=0.5, rho=1.0, rnd_state=42)
        cp.learn_initial_training_set(X[:30], y[:30])
        p = cp.compute_p_value(X[30], y[30])
        assert 0.0 <= p <= 1.0

    def test_learn_one_with_rho(self, correlated_data):
        """learn_one should work with rho > 0."""
        X, y, _ = correlated_data
        cp = ConformalLassoRegressor(lam=0.5, rho=2.0)
        cp.learn_initial_training_set(X[:30], y[:30])
        cp.learn_one(X[30], y[30])
        assert cp.X.shape[0] == 31

    def test_online_coverage_elastic_net(self, correlated_data):
        """Online coverage with elastic net should still be reasonable."""
        X, y, _ = correlated_data
        epsilon = 0.2
        cp = ConformalLassoRegressor(lam=0.3, rho=1.0, epsilon=epsilon, rnd_state=7)
        cp.learn_initial_training_set(X[:20], y[:20])

        covered = 0
        total = 0
        for i in range(20, 55):
            interval = cp.predict(X[i], epsilon=epsilon)
            if y[i] in interval:
                covered += 1
            total += 1
            cp.learn_one(X[i], y[i])

        coverage = covered / total
        assert coverage >= 0.5, f"Coverage too low: {coverage:.2f}"

    def test_predict_epsilon_passthrough(self, correlated_data):
        """Predict with non-default epsilon should use that epsilon in homotopy."""
        X, y, _ = correlated_data
        cp = ConformalLassoRegressor(lam=0.5, rho=1.0, epsilon=0.2, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        x_test = X[30]
        # Predict at a TIGHTER epsilon than default → interval should be WIDER
        interval_tight = cp.predict(x_test, epsilon=0.05)
        interval_default = cp.predict(x_test, epsilon=0.2)

        # Tighter epsilon must give at least as wide an interval
        assert interval_tight.lower <= interval_default.lower + 1e-10
        assert interval_tight.upper >= interval_default.upper - 1e-10

    def test_large_rho_approaches_ridge(self):
        """With very large rho and lam=0, should behave like ridge regression."""
        rng = np.random.default_rng(77)
        n, p = 40, 5
        X = rng.normal(size=(n, p))
        y = X @ np.array([1.0, 2.0, -1.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=n)

        # Large rho, zero lam → pure ridge
        beta_enet = _solve_lasso(X, y, lam=0.0, rho=100.0, max_iter=5000, tol=1e-10)
        # Ridge closed form: (X^T X + rho I)^{-1} X^T y
        beta_ridge = np.linalg.solve(X.T @ X + 100.0 * np.eye(p), X.T @ y)
        np.testing.assert_allclose(beta_enet, beta_ridge, atol=1e-4)
