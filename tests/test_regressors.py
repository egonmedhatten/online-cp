import numpy as np

from online_cp.kernels import GaussianKernel
from online_cp.regressors import (
    ConformalRidgeRegressor,
    KernelConformalRidgeRegressor,
    MultiLevelPredictionInterval,
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
        """Ridge regressor with list of epsilons returns MultiLevelPredictionInterval."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        epsilons = [0.01, 0.05, 0.1, 0.2]
        result = cp.predict(X[30], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionInterval)
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

        assert isinstance(result, MultiLevelPredictionInterval)
        assert len(result) == 3
        # Smaller epsilon => wider interval
        assert result[0.05].width() >= result[0.1].width()
        assert result[0.1].width() >= result[0.2].width()

    def test_scalar_epsilon_unchanged(self, linear_dataset):
        """Scalar epsilon should still return a single interval."""
        X, y = linear_dataset
        cp = ConformalRidgeRegressor(a=1.0, warnings=False, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        from online_cp.regressors import ConformalPredictionInterval
        result = cp.predict(X[30], epsilon=0.1)
        assert isinstance(result, ConformalPredictionInterval)
