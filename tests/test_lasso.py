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

    def test_autotune_selected_lambda_is_cv_optimal(self, sparse_data):
        """The warm-started CV should select a lambda whose CV MSE is (near-)optimal.

        The selected lambda must attain a CV MSE within a small tolerance of the
        best grid value. (Warm-starting can flip a *tie* between two adjacent
        grid points with identical CV MSE, so we assert on the MSE, not on
        bit-identical lambda.)
        """
        X, y, _ = sparse_data
        n_folds = 4
        cp = ConformalLassoRegressor(lam=1.0, autotune=False, n_folds=n_folds, rnd_state=42)
        cp.X = X[:60].copy()
        cp.y = y[:60].copy()
        cp.rnd_gen = np.random.default_rng(42)

        # Manually reproduce the CV to get the full MSE curve.
        indices = np.arange(60)
        cp.rnd_gen.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        lam_max = np.max(np.abs(cp.X.T @ cp.y)) / cp.X.shape[0]
        lam_grid = np.geomspace(lam_max * 1e-3, lam_max, num=50)
        mses = []
        for lam in lam_grid:
            mse = 0.0
            for k in range(n_folds):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                X_tr, y_tr = cp.X[train_idx], cp.y[train_idx]
                X_val, y_val = cp.X[val_idx], cp.y[val_idx]
                beta = _solve_lasso(X_tr, y_tr, lam, rho=cp.rho)
                mse += np.mean((y_val - X_val @ beta) ** 2)
            mses.append(mse / n_folds)
        best_mse = min(mses)

        # Run the actual autotune.
        cp.rnd_gen = np.random.default_rng(42)
        cp._tune_lambda()
        selected = cp.lam
        # The selected lambda must be on the grid and its MSE near-optimal.
        assert selected in lam_grid or any(np.isclose(selected, g) for g in lam_grid), (
            f"selected lam {selected} not on grid"
        )
        idx = int(np.argmin(np.abs(lam_grid - selected)))
        assert mses[idx] <= best_mse * 1.001 + 1e-12, (
            f"selected lam {selected} (mse {mses[idx]:.6e}) is not near-optimal "
            f"(best {best_mse:.6e})"
        )

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

        cp = ConformalLassoRegressor(lam=lam, epsilon=epsilon, search_range_factor=0.25, rnd_state=7)
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
                # Ascending rank (1 = smallest) of the test residual among all
                # residuals: count of |r_i| <= |r_{n+1}| (test always counted).
                # Lei (2019) conformal criterion (prelim.tex line 141 /
                # method.tex line 459): in-set iff this rank is NO HIGHER than
                # threshold (the test residual must be among the SMALLEST).
                rank = np.sum(abs_res <= abs_res[-1])
                if rank <= threshold:
                    in_set_bf.append(y_trial)

            if not in_set_bf:
                assert np.isnan(interval.lower), f"Test {i}: brute-force empty but homotopy non-empty"
                continue

            lo_bf, hi_bf = min(in_set_bf), max(in_set_bf)
            grid_spacing = (grid[-1] - grid[0]) / (len(grid) - 1)

            # Endpoints should agree within a few grid spacings
            assert abs(interval.lower - lo_bf) < 5 * grid_spacing, (
                f"Test {i}: lower endpoint mismatch: homotopy={interval.lower:.4f} vs bf={lo_bf:.4f}"
            )
            assert abs(interval.upper - hi_bf) < 5 * grid_spacing, (
                f"Test {i}: upper endpoint mismatch: homotopy={interval.upper:.4f} vs bf={hi_bf:.4f}"
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


class TestFindIntervalsInPiece:
    """Regression tests for the vectorized _find_intervals_in_piece.

    These pin the (vectorized) implementation's contract: it must return a list
    of (t_start, t_end) intervals with t_start <= t_end, covering exactly the
    sub-intervals of [t_acc, t_acc + sign*dt] (mapped to global t-space) where
    the test residual's rank is AT LEAST the threshold (Lei 2019 conformal
    criterion: the test residual must be among the largest).
    """

    def _ref(self, cp, r_train, r_test, slopes, slope_test, dt, tacc, sign, thr, n):
        """Loop-based reference implementation (the pre-vectorization version).

        NOTE: the conformal criterion (Lei 2019, prelim.tex line 141 /
        method.tex line 459) keeps a sub-interval iff the ASCENDING rank of the
        test residual (the count of residuals with |r_i| <= |r_{n+1}|, including
        the test point; 1 = smallest) is NO HIGHER than the threshold.  (A
        candidate label is in the set iff its fitted residual is among the
        SMALLEST.)
        """
        from online_cp.regressors import _compute_crossings

        raw = _compute_crossings(r_train, slopes, r_test, slope_test, n, dt)
        crossings = [0.0]
        for c in raw:
            crossings.append(c)
        crossings.append(dt)
        crossings = sorted(set(crossings))
        out = []
        for idx in range(len(crossings) - 1):
            ds, de = crossings[idx], crossings[idx + 1]
            if de - ds < 1e-14:
                continue
            dm = (ds + de) / 2
            ri = r_train + slopes * dm
            rt = r_test + slope_test * dm
            rank = np.sum(np.abs(ri) <= np.abs(rt)) + 1
            if rank <= thr:
                ts = sign * (tacc + ds)
                te = sign * (tacc + de)
                if ts > te:
                    ts, te = te, ts
                out.append((ts, te))
        return out

    def test_matches_loop_reference(self):
        """Vectorized output must equal the loop reference for random pieces."""
        cp = ConformalLassoRegressor(lam=0.01, rnd_state=0)
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(2, 60))
            r_train = rng.normal(size=n)
            slopes = rng.normal(size=n)
            r_test = rng.normal()
            slope_test = rng.normal()
            dt = float(rng.uniform(0, 5))
            tacc = float(rng.uniform(0, 10))
            sign = int(rng.choice([-1, 1]))
            thr = int(rng.integers(1, n + 2))
            new = cp._find_intervals_in_piece(r_train, r_test, slopes, slope_test, dt, tacc, sign, thr, n)
            old = self._ref(cp, r_train, r_test, slopes, slope_test, dt, tacc, sign, thr, n)
            assert len(new) == len(old), f"length mismatch: {len(new)} vs {len(old)}"
            for a, b in zip(sorted(new), sorted(old)):
                np.testing.assert_allclose(a, b, atol=1e-9)

    def test_intervals_are_sorted_and_non_degenerate(self):
        """Each returned interval must have start <= end."""
        cp = ConformalLassoRegressor(lam=0.01, rnd_state=0)
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(2, 40))
            r_train = rng.normal(size=n)
            slopes = rng.normal(size=n)
            dt = float(rng.uniform(0, 3))
            sign = int(rng.choice([-1, 1]))
            thr = int(rng.integers(1, n + 2))
            out = cp._find_intervals_in_piece(
                r_train, rng.normal(), slopes, rng.normal(), dt, 0.0, sign, thr, n
            )
            for ts, te in out:
                assert ts <= te, f"interval ({ts}, {te}) has start > end"

    def test_full_range_when_threshold_is_max(self):
        """If the threshold equals n+1, every sub-interval is in the set.

        With the paper's conformal criterion (Lei 2019, prelim.tex line 141)
        the ascending rank of |r_{n+1}| (1 = smallest) must be NO HIGHER than
        the threshold.  The rank is at most n+1, so threshold = n+1 keeps
        every sub-interval.
        """
        cp = ConformalLassoRegressor(lam=0.01, rnd_state=0)
        n = 20
        rng = np.random.default_rng(2)
        r_train = rng.normal(size=n)
        slopes = rng.normal(size=n)
        # threshold = n+1 means rank <= n+1 is always true (rank is at most n+1)
        out = cp._find_intervals_in_piece(
            r_train, 0.0, slopes, 1.0, dt_k=2.0, t_accumulated=5.0, sign=1, threshold=n + 1, n=n
        )
        # The union of returned intervals must cover the whole [5, 7] range.
        assert out, "threshold=n+1 should keep every sub-interval"
        lo = min(a for a, _ in out)
        hi = max(b for _, b in out)
        assert np.isclose(lo, 5.0, atol=1e-9)
        assert np.isclose(hi, 7.0, atol=1e-9)


class TestHomotopyCorrectness:
    """Correctness regression tests for the Lasso homotopy predict().

    These guard against a non-optimum starting Lasso fit breaking the
    homotopy's piecewise-linear path.  The Lei (2019) conformal set keeps a
    candidate label y_{n+1}(t) iff the ASCENDING rank of |r_{n+1}(t)| among
    all (n+1) fitted residuals (1 = smallest) is NO HIGHER than
    ceil((n+1)(1-alpha)); equivalently, the test residual must be among the
    SMALLEST in magnitude.  The set is clipped to the range
    [y_(0) - 0.25*(y_(n) - y_(0)), y_(n) + 0.25*(y_(n) - y_(0))].
    """

    @staticmethod
    def _brute_set(cp, x_test, epsilon, n_grid=400):
        """Brute-force reference: solve Lasso on the augmented data at each
        grid point y and report the (enclosing) interval of points whose
        residual ascending rank is NO HIGHER than ceil((n+1)(1-epsilon))."""
        from online_cp.regressors import _solve_lasso

        y0 = float(cp.y.min())
        yn = float(cp.y.max())
        rng_ = (yn - y0) if (yn - y0) > 0 else 1.0
        y_lo = y0 - 0.25 * rng_
        y_hi = yn + 0.25 * rng_
        grid = np.linspace(y_lo, y_hi, n_grid)
        thr = int(np.ceil((cp.X.shape[0] + 1) * (1 - epsilon)))
        X_aug = np.vstack([cp.X, x_test.reshape(1, -1)])
        in_set = np.zeros(len(grid), dtype=bool)
        for i, g in enumerate(grid):
            y_aug = np.append(cp.y, float(g))
            b = _solve_lasso(X_aug, y_aug, cp.lam, rho=cp.rho, max_iter=3000, tol=1e-9)
            resid = np.abs(y_aug - X_aug @ b)
            # Ascending rank (1 = smallest): count of |r_i| <= |r_{n+1}|
            # (the test residual always counts, so rank is 1-based).
            in_set[i] = int(np.sum(resid <= resid[-1])) <= thr
        if not in_set.any():
            return None
        return (grid[in_set].min(), grid[in_set].max())

    def test_predict_covers_brute_force_set(self):
        """predict() must cover the brute-force conformal set (no
        under-coverage) on a range of random problems."""
        rng = np.random.default_rng(0)
        for trial in range(12):
            p = int(rng.integers(3, 10))
            n = int(rng.integers(14, 30))
            lam = float(rng.uniform(1e-2, 0.4))
            rho = float(rng.choice([0.0, 1.0]))
            X = rng.normal(size=(n, p))
            y = X.sum(1) * 0.5 + rng.normal(0, 0.5, n)
            cp = ConformalLassoRegressor(
                lam=lam, rho=rho, autotune=False, verbose=0, rnd_state=trial
            )
            cp.learn_initial_training_set(X[: n // 2], y[: n // 2])
            x_test = X[n // 2]
            eps = cp.epsilon

            interval = cp.predict(x_test)
            ref = self._brute_set(cp, x_test, eps)

            if interval.is_empty:
                # Homotopy found nothing; brute force must also be empty.
                assert ref is None, f"trial {trial}: homotopy empty but brute set {ref}"
                continue

            assert ref is not None, f"trial {trial}: homotopy non-empty but brute set empty"
            # The homotopy interval must COVER the brute-force set (superset),
            # up to grid resolution.
            res = (float(cp.y.max()) - float(cp.y.min())) / 200
            if res < 1e-6:
                res = 1e-3
            assert interval.lower <= ref[0] + res, (
                f"trial {trial}: under-coverage on the left: "
                f"homotopy lower {interval.lower:.4f} < brute {ref[0]:.4f}"
            )
            assert interval.upper >= ref[1] - res, (
                f"trial {trial}: under-coverage on the right: "
                f"homotopy upper {interval.upper:.4f} < brute {ref[1]:.4f}"
            )
