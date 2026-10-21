import numpy as np

from online_cp.kernels import GaussianKernel
from online_cp.regressors import (
    ConformalRegressor,
    ConformalRidgeRegressor,
    KernelConformalRidgeRegressor,
    MultiLevelPredictionSet,
)


class TestConformalRidgeRegressor:
    def test_validity(self, linear_dataset):
        """Coverage should be >= 1 - epsilon - margin on iid data."""
        X, y = linear_dataset
        epsilon = 0.1
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0, epsilon=epsilon)

        n_init = cp.minimum_training_set(epsilon)
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_finite_intervals_after_min_training(self, linear_dataset):
        X, y = linear_dataset
        epsilon = 0.1
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)

        n_init = cp.minimum_training_set(epsilon)
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        Gamma = cp.predict(X[n_init], epsilon=epsilon)
        assert Gamma.lower > -np.inf
        assert Gamma.upper < np.inf

    def test_infinite_interval_before_min_training(self):
        cp = ConformalRidgeRegressor(a=1.0, warnings=False)
        cp.learn_initial_training_set(np.array([[1, 0]]), np.array([1.0]))
        Gamma = cp.predict(np.array([0, 1]), epsilon=0.1)
        assert Gamma.lower == -np.inf or Gamma.upper == np.inf

    def test_learn_one_no_initial_training(self):
        cp = ConformalRidgeRegressor(a=1.0, warnings=False)
        cp.learn_one(np.array([1.0, 2.0]), 3.0)
        assert cp.X.shape == (1, 2)
        assert cp.y[0] == 3.0

    def test_sherman_morrison_matches_recompute(self, linear_dataset):
        """Sherman-Morrison update should give same result as full recomputation."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False)
        cp.learn_initial_training_set(X[:20], y[:20])

        # Learn one more via incremental update
        x_new = X[20]
        cp.learn_one(x_new, y[20])
        XTXinv_incremental = cp.XTXinv

        # Recompute from scratch
        XTXinv_full = np.linalg.inv(cp.X.T @ cp.X + cp.a * np.identity(cp.p))

        assert np.allclose(XTXinv_incremental, XTXinv_full, atol=1e-10)

    def test_minimum_training_set(self):
        cp = ConformalRidgeRegressor()
        assert cp.minimum_training_set(0.1) == 20
        assert cp.minimum_training_set(0.1, "upper") == 10
        assert cp.minimum_training_set(0.05) == 40

    def test_x_txinv_stays_symmetric_over_long_stream(self):
        """Repeated rank-1 inverse updates must preserve symmetry.

        Regression test for the symmetrized `_rank1_inverse_update`: after many
        online `learn_one` calls the incremental (X'X + aI)^-1 must remain
        symmetric to machine precision (the unsymmetrized update accumulated
        ~1e-7 asymmetry over a few hundred near-collinear steps).
        """
        cp = ConformalRidgeRegressor(a=1e-4, warnings=False, rnd_state=0)
        rng = np.random.default_rng(0)
        d = 5
        X = rng.normal(size=(300, d)) * np.array([5.0, 1.0, 0.3, 0.1, 1.0])
        X[:, 1] = X[:, 0] + 0.01 * rng.normal(size=300)  # near-collinear
        y = X.sum(axis=1)
        cp.learn_initial_training_set(X[:40], y[:40])
        for i in range(40, 300):
            cp.learn_one(X[i], y[i])
        asym = np.abs(cp.XTXinv - cp.XTXinv.T).max()
        assert asym == 0.0, f"XTXinv lost symmetry after online updates (max asym={asym})"

    def test_ridge_inverse_update_matches_reference(self):
        """The symmetrized rank-1 update must match a direct re-inversion."""
        cp = ConformalRidgeRegressor(a=0.5, warnings=False, rnd_state=0)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 4))
        y = X.sum(axis=1)
        cp.learn_initial_training_set(X[:10], y[:10])
        for i in range(10, 60):
            cp.learn_one(X[i], y[i])
            expected = np.linalg.inv(X[: i + 1].T @ X[: i + 1] + cp.a * np.identity(cp.p))
            assert np.allclose(cp.XTXinv, expected, atol=1e-8), f"divergence at n={i + 1}"

    def test_order_statistic_clamping_at_small_n(self):
        """_get_lower/__get_upper clamp the rank to [1, n-1] and return rays.

        Pins the edge-case behavior: when the required 1-based rank exceeds the
        number of available training cut-points (n - 1), the selector returns the
        unbounded ray rather than wrapping to the opposite tail.
        """
        # 3 training cut-points => ranks 1..3 (n = 4 total).
        l_dic = {1: 1.0, 2: 2.0, 3: 3.0}
        u_dic = {1: 1.0, 2: 2.0, 3: 3.0}
        n = 4
        # Upper: required rank ceil((1 - eps)*n). eps -> 0 gives rank 4 > n-1 => +inf.
        assert np.isposinf(ConformalRegressor._get_upper(u_dic, 1e-6, n))
        # eps = 1.0 gives rank 0 (out of range) => +inf (KeyError fallback).
        assert np.isposinf(ConformalRegressor._get_upper(u_dic, 1.0, n))
        # eps = 0.5 => rank ceil(0.5*4) = 2 => u_dic[2] = 2.0
        assert ConformalRegressor._get_upper(u_dic, 0.5, n) == 2.0
        # Lower: required rank floor(eps*n). eps = 1 gives rank 4 > n-1 => -inf.
        assert np.isneginf(ConformalRegressor._get_lower(l_dic, 1.0, n))
        # eps = 0.5 => rank floor(0.5*4) = 2 => l_dic[2] = 2.0
        assert ConformalRegressor._get_lower(l_dic, 0.5, n) == 2.0
        # eps = 0 => rank 0 (out of range) => -inf (KeyError fallback).
        assert np.isneginf(ConformalRegressor._get_lower(l_dic, 0.0, n))

    def test_vectorised_l_and_u_drops_test_point(self):
        """l_dic/u_dic hold exactly n - 1 ranked cut-points (test point excluded)."""
        cp = ConformalRidgeRegressor(a=1.0, warnings=False)
        rng = np.random.default_rng(2)
        n = 12
        A = rng.normal(size=n)
        B = np.abs(rng.normal(size=n)) + 0.5  # keep B[-1] - B[i] well-defined
        B[-1] = B[-1] + 3.0  # make test point have the largest B
        l_dic, u_dic = cp._vectorised_l_and_u(A, B)
        assert len(l_dic) == n - 1
        assert len(u_dic) == n - 1
        assert set(l_dic) == set(range(1, n))
        assert set(u_dic) == set(range(1, n))

    def test_compute_p_value_in_unit_interval(self, linear_dataset):
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])

        for i in range(20, 30):
            p = cp.compute_p_value(X[i], y[i])
            assert 0 <= p <= 1

    def test_change_ridge_parameter(self, linear_dataset):
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=0.0, warnings=False)
        cp.learn_initial_training_set(X[:20], y[:20])

        cp.change_ridge_parameter(5.0)
        assert cp.a == 5.0
        expected = np.linalg.inv(cp.X.T @ cp.X + 5.0 * np.identity(cp.p))
        assert np.allclose(cp.XTXinv, expected)

    def test_ridge_gcv_eigenbasis_matches_inverse_gcv(self):
        """The eigenbasis GCV must minimise the *same* GCV as the inverse form.

        We evaluate the original (inverse) GCV at both the eigenbasis-tuned ``a``
        and the original-tuned ``a``; the eigenbasis result must attain a GCV no
        worse than the original's, and both must be near the true minimum of the
        GCV curve over a grid (the GCV is flat, so this is a robust check that
        does not depend on the exact optimiser).
        """
        rng = np.random.default_rng(0)
        n, p = 120, 6
        X = rng.normal(size=(n, p))
        y = X.sum(axis=1) + rng.normal(scale=0.2, size=n)

        cp = ConformalRidgeRegressor(a=1.0, warnings=False, verbose=0)
        cp.learn_initial_training_set(X, y)

        def gcv_inverse(a):
            XTX = cp.X.T @ cp.X
            Id = cp.Id
            In = np.identity(n)
            A = cp.X @ np.linalg.inv(XTX + a * Id) @ cp.X.T
            if np.max(np.diag(A)) > 1:
                return np.inf
            return (1 / n) * np.linalg.norm((In - A) @ cp.y) ** 2 / ((1 / n) * np.trace(In - A)) ** 2

        # True GCV minimum over a log grid (robust reference).
        grid = np.geomspace(1e-4, 1.0, num=200)
        ref_min = min(gcv_inverse(a) for a in grid)

        cp._tune_ridge_parameter()
        a_tuned = cp.a
        assert a_tuned >= 1e-6, "tuned ridge parameter must respect the lower bound"
        gcv_at_tuned = gcv_inverse(a_tuned)
        # The eigenbasis-tuned a must attain a GCV essentially at the minimum.
        assert gcv_at_tuned <= ref_min * 1.001 + 1e-12, (
            f"tuned a={a_tuned:.3e} attains GCV {gcv_at_tuned:.6e}, "
            f"but the grid minimum is {ref_min:.6e}"
        )

        # The resulting fit should be close to the (near-)OLS solution, since
        # the GCV is near-minimal (a small ridge parameter).
        beta_tuned = cp.XTXinv @ cp.X.T @ cp.y
        beta_ols = np.linalg.lstsq(cp.X, cp.y, rcond=None)[0]
        rel = np.max(np.abs(beta_tuned - beta_ols)) / (np.max(np.abs(beta_ols)) + 1e-12)
        assert rel < 0.05, f"tuned fit deviates too much from OLS (rel {rel:.2e})"


class TestKernelConformalRidgeRegressor:
    def test_validity(self, linear_dataset):
        X, y = linear_dataset
        epsilon = 0.1
        ker = GaussianKernel(sigma=1.0)
        cp = KernelConformalRidgeRegressor(a=1.0, kernel=ker, warnings=False, rnd_state=0, epsilon=epsilon)

        n_init = cp.minimum_training_set(epsilon)
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = min(80, len(y) - n_init)  # limit for speed
        for obj, lab in zip(X[n_init : n_init + n_test], y[n_init : n_init + n_test]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.08, f"Coverage {coverage:.3f} too low"

    def test_finite_intervals(self, linear_dataset):
        X, y = linear_dataset
        epsilon = 0.1
        ker = GaussianKernel(sigma=1.0)
        cp = KernelConformalRidgeRegressor(a=1.0, kernel=ker, warnings=False)

        n_init = cp.minimum_training_set(epsilon)
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        Gamma = cp.predict(X[n_init], epsilon=epsilon)
        assert Gamma.lower > -np.inf
        assert Gamma.upper < np.inf


class TestMultiLevelPredict:
    def test_ridge_multi_level(self, linear_dataset):
        """Ridge regressor with list of epsilons returns MultiLevelPredictionSet."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        epsilons = [0.01, 0.05, 0.1, 0.2]
        result = cp.predict(X[30], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert result.levels == sorted(epsilons)
        assert len(result) == 4

        # Smaller epsilon => wider interval
        for i in range(len(epsilons) - 1):
            assert result[epsilons[i]].width() >= result[epsilons[i + 1]].width()

        # __contains__ returns bool (True if covered at all levels)
        assert isinstance(y[30] in result, bool)

        # .coverage() returns dict
        coverage = result.coverage(y[30])
        assert isinstance(coverage, dict)
        assert set(coverage.keys()) == set(epsilons)

    def test_kernel_ridge_multi_level(self, linear_dataset):
        """Kernel ridge regressor with list of epsilons."""
        X, y = linear_dataset
        ker = GaussianKernel(sigma=1.0)
        cp = KernelConformalRidgeRegressor(a=1.0, kernel=ker, warnings=False)
        cp.learn_initial_training_set(X[:30], y[:30])

        epsilons = [0.05, 0.1, 0.2]
        result = cp.predict(X[30], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert len(result) == 3
        # Smaller epsilon => wider interval
        assert result[0.05].width() >= result[0.1].width()
        assert result[0.1].width() >= result[0.2].width()

    def test_scalar_epsilon_unchanged(self, linear_dataset):
        """Scalar epsilon should still return a single interval."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        from online_cp.regressors import ContinuousPredictionSet

        result = cp.predict(X[30], epsilon=0.1)
        assert isinstance(result, ContinuousPredictionSet)


class TestRecomputeInverse:
    """Tests for Sherman-Morrison numerical stability features."""

    def test_ridge_recompute_inverse_matches_fresh(self, linear_dataset):
        """recompute_inverse() should produce same result as computing from scratch."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])

        # Do many SM updates
        for i in range(20, 50):
            cp.learn_one(X[i], y[i])

        # Recompute and compare to ground truth
        cp.recompute_inverse()
        ground_truth = np.linalg.inv(cp.X.T @ cp.X + cp.a * cp.Id)
        assert np.allclose(cp.XTXinv, ground_truth, atol=1e-12)
        assert cp._n_sm_updates == 0

    def test_ridge_recompute_every_triggers(self, linear_dataset):
        """recompute_every should trigger periodic full recomputation."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0, recompute_every=5)
        cp.learn_initial_training_set(X[:20], y[:20])

        # After 5 SM updates, counter should reset
        for i in range(20, 25):
            cp.learn_one(X[i], y[i])
        assert cp._n_sm_updates == 0  # reset after recompute at step 5

        # After 3 more, should be at 3
        for i in range(25, 28):
            cp.learn_one(X[i], y[i])
        assert cp._n_sm_updates == 3

    def test_ridge_recompute_every_accuracy(self, linear_dataset):
        """With recompute_every, drift should be bounded vs ground truth."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0, recompute_every=10)
        cp.learn_initial_training_set(X[:20], y[:20])

        for i in range(20, 80):
            cp.learn_one(X[i], y[i])

        ground_truth = np.linalg.inv(cp.X.T @ cp.X + cp.a * cp.Id)
        assert np.allclose(cp.XTXinv, ground_truth, atol=1e-10)

    def test_ridge_recompute_inverse_idempotent(self, linear_dataset):
        """Calling recompute_inverse twice gives same result."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        for i in range(30, 50):
            cp.learn_one(X[i], y[i])

        cp.recompute_inverse()
        first = cp.XTXinv.copy()
        cp.recompute_inverse()
        assert np.array_equal(first, cp.XTXinv)

    def test_kernel_ridge_recompute_inverse_matches_fresh(self, linear_dataset):
        """KernelRidge recompute_inverse() matches ground truth."""
        X, y = linear_dataset
        kernel = GaussianKernel(sigma=1.0)
        cp = KernelConformalRidgeRegressor(kernel=kernel, a=0.1, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])

        for i in range(20, 35):
            cp.learn_one(X[i], y[i])

        cp.recompute_inverse()
        n = cp.K.shape[0]
        ground_truth = np.linalg.inv(cp.K + cp.a * np.identity(n))
        assert np.allclose(cp.Kinv, ground_truth, atol=1e-12)
        assert cp._n_sm_updates == 0

    def test_kernel_ridge_recompute_every_triggers(self, linear_dataset):
        """KernelRidge recompute_every triggers periodic recomputation."""
        X, y = linear_dataset
        kernel = GaussianKernel(sigma=1.0)
        cp = KernelConformalRidgeRegressor(kernel=kernel, a=0.1, warnings=False, rnd_state=0, recompute_every=5)
        cp.learn_initial_training_set(X[:20], y[:20])

        for i in range(20, 25):
            cp.learn_one(X[i], y[i])
        assert cp._n_sm_updates == 0  # reset after recompute

    def test_ridge_change_parameter_resets_counter(self, linear_dataset):
        """change_ridge_parameter should reset the SM update counter."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])

        for i in range(20, 30):
            cp.learn_one(X[i], y[i])
        assert cp._n_sm_updates > 0

        cp.change_ridge_parameter(2.0)
        assert cp._n_sm_updates == 0


class TestStudentised:
    """Tests for studentised conformal ridge regressors."""

    def test_ridge_studentised_coverage(self, linear_dataset):
        """Studentised Ridge should achieve valid coverage."""
        X, y = linear_dataset
        epsilon = 0.1
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0, studentised=True, epsilon=epsilon)

        n_init = 30
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Studentised coverage {coverage:.3f} too low"

    def test_kernel_ridge_studentised_coverage(self, linear_dataset):
        """Studentised KernelRidge should achieve valid coverage."""
        X, y = linear_dataset
        epsilon = 0.1
        kernel = GaussianKernel(sigma=1.0)
        cp = KernelConformalRidgeRegressor(
            kernel=kernel,
            a=0.1,
            warnings=False,
            rnd_state=0,
            studentised=True,
            epsilon=epsilon,
        )

        n_init = 30
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = min(20, len(y) - n_init)  # limit to keep test fast
        for obj, lab in zip(X[n_init : n_init + n_test], y[n_init : n_init + n_test]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.1, f"KernelRidge studentised coverage {coverage:.3f} too low"

    def test_ridge_studentised_guard_degenerate(self):
        """Studentised path should handle near-degenerate leverage gracefully."""
        cp = ConformalRidgeRegressor(a=1e-8, warnings=False, rnd_state=0, studentised=True)
        # Near-collinear features → high leverage → B[n] close to B[i]
        X = np.array([[1.0, 1.0001], [2.0, 2.0001], [3.0, 3.0001]])
        y = np.array([1.0, 2.0, 3.0])
        cp.learn_initial_training_set(X, y)
        # Should not crash even with near-degenerate data
        interval = cp.predict(np.array([4.0, 4.0001]))
        assert hasattr(interval, "lower")
        assert hasattr(interval, "upper")


class TestLassoCrashGuard:
    """Test that Lasso handles no-training-data gracefully."""

    def test_predict_before_training(self):
        """Lasso predict should return (-inf, inf) before training."""
        from online_cp.regressors import ConformalLassoRegressor

        cp = ConformalLassoRegressor(lam=0.5)
        result = cp.predict(np.array([1.0, 2.0, 3.0]))
        assert result.lower == -np.inf
        assert result.upper == np.inf

    def test_predict_multi_epsilon_before_training(self):
        """Lasso predict with multiple epsilons before training."""
        from online_cp.regressors import ConformalLassoRegressor

        cp = ConformalLassoRegressor(lam=0.5)
        result = cp.predict(np.array([1.0, 2.0]), epsilon=[0.05, 0.1])
        assert isinstance(result, MultiLevelPredictionSet)
        assert result[0.05].lower == -np.inf
