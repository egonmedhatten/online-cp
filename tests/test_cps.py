import numpy as np
import pytest

from online_cp.CPS import (
    DempsterHillConformalPredictiveSystem,
    KernelRidgePredictionMachine,
    NearestNeighboursPredictionMachine,
    RidgePredictionMachine,
)
from online_cp.kernels import GaussianKernel


@pytest.fixture
def ridge_cps_dataset():
    rng = np.random.default_rng(2024)
    N = 100
    X = rng.normal(size=(N, 4))
    beta = np.array([2, 1, 0, 0])
    Y = X @ beta + rng.normal(scale=1, size=N)
    return X, Y


class TestRidgePredictionMachine:
    def test_cpd_monotone_in_tau(self, ridge_cps_dataset):
        """CPD(y, tau=0) <= CPD(y, tau=1) for all y."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1, warnings=False)
        cps.learn_initial_training_set(X[:20], Y[:20])

        cpd = cps.predict_cpd(X[20])
        y_test = np.linspace(-5, 5, 50)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_cpd_bounded_zero_one(self, ridge_cps_dataset):
        """CPD values should be in [0, 1]."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1, warnings=False)
        cps.learn_initial_training_set(X[:20], Y[:20])

        cpd = cps.predict_cpd(X[20])
        y_test = np.linspace(-10, 10, 100)
        for y in y_test:
            val = cpd(y, tau=0.5)
            assert -1e-10 <= val <= 1 + 1e-10

    def test_coverage(self, ridge_cps_dataset):
        """Prediction sets should have valid coverage."""
        X, Y = ridge_cps_dataset
        epsilon = 0.1
        cps = RidgePredictionMachine(a=1, warnings=False, epsilon=epsilon)
        cps.learn_initial_training_set(X[:20], Y[:20])

        rng = np.random.default_rng(42)
        covered = 0
        n_test = 60
        for i in range(20, 20 + n_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(X[i], Y[i], precomputed=precomputed)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.10

    def test_learn_one_with_precomputed(self, ridge_cps_dataset):
        """Learning with precomputed should give same state as without."""
        X, Y = ridge_cps_dataset
        # With precomputed
        cps1 = RidgePredictionMachine(a=1, warnings=False)
        cps1.learn_initial_training_set(X[:20], Y[:20])
        _, precomputed = cps1.predict_cpd(X[20], return_update=True)
        cps1.learn_one(X[20], Y[20], precomputed=precomputed)

        # Without precomputed
        cps2 = RidgePredictionMachine(a=1, warnings=False)
        cps2.learn_initial_training_set(X[:20], Y[:20])
        cps2.learn_one(X[20], Y[20])

        assert np.allclose(cps1.XTXinv, cps2.XTXinv, atol=1e-10)
        assert np.allclose(cps1.y, cps2.y)


class TestNearestNeighboursPredictionMachine:
    def test_cpd_monotone_in_tau(self, ridge_cps_dataset):
        X, Y = ridge_cps_dataset
        cps = NearestNeighboursPredictionMachine(k=5)
        cps.learn_initial_training_set(X[:30], Y[:30])

        cpd = cps.predict_cpd(X[30])
        y_test = np.linspace(-5, 5, 30)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_coverage(self, ridge_cps_dataset):
        X, Y = ridge_cps_dataset
        epsilon = 0.1
        cps = NearestNeighboursPredictionMachine(k=5, epsilon=epsilon)
        cps.learn_initial_training_set(X[:30], Y[:30])

        rng = np.random.default_rng(42)
        covered = 0
        n_test = 40
        for i in range(30, 30 + n_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(X[i], Y[i], precomputed=precomputed)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.10

    def test_repeated_labels(self):
        cps = NearestNeighboursPredictionMachine(k=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 1.0, 2.0, 2.0, 3.0])  # Duplicate labels
        cps.learn_initial_training_set(X, y)
        cpd = cps.predict_cpd(np.array([3.5]))
        # Should produce a valid CPD (monotone, bounded in [0, 1])
        pi0, pi1 = cpd(2.0)
        assert 0 <= pi0 <= pi1 <= 1


class TestDempsterHillConformalPredictiveSystem:
    def test_cpd_monotone_in_tau(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=50)
        cps = DempsterHillConformalPredictiveSystem()
        cps.learn_initial_training_set(Y)

        cpd = cps.predict_cpd()
        y_test = np.linspace(-3, 3, 30)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_coverage(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(size=100)
        epsilon = 0.1
        cps = DempsterHillConformalPredictiveSystem(epsilon=epsilon)
        cps.learn_initial_training_set(Y[:20])

        covered = 0
        n_test = 60
        for i in range(20, 20 + n_test):
            tau = rng.uniform()
            cpd = cps.predict_cpd()
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(Y[i])

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.10

    def test_learn_one(self):
        cps = DempsterHillConformalPredictiveSystem()
        cps.learn_initial_training_set(np.array([1.0, 2.0, 3.0]))
        cps.learn_one(4.0)
        assert len(cps.y) == 4
        assert cps.y[-1] == 4.0


@pytest.fixture
def kernel_cps_dataset():
    rng = np.random.default_rng(2024)
    N = 80
    X = rng.normal(size=(N, 3))
    beta = np.array([2, 1, 0.5])
    Y = X @ beta + rng.normal(scale=1, size=N)
    return X, Y


class TestKernelRidgePredictionMachine:
    def test_cpd_monotone_in_tau(self, kernel_cps_dataset):
        """CPD(y, tau=0) <= CPD(y, tau=1) for all y."""
        X, Y = kernel_cps_dataset
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False)
        cps.learn_initial_training_set(X[:15], Y[:15])

        cpd = cps.predict_cpd(X[15])
        y_test = np.linspace(-5, 5, 50)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_cpd_bounded_zero_one(self, kernel_cps_dataset):
        """CPD values should be in [0, 1]."""
        X, Y = kernel_cps_dataset
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False)
        cps.learn_initial_training_set(X[:15], Y[:15])

        cpd = cps.predict_cpd(X[15])
        y_test = np.linspace(-10, 10, 100)
        for y in y_test:
            val = cpd(y, tau=0.5)
            assert -1e-10 <= val <= 1 + 1e-10

    def test_coverage(self, kernel_cps_dataset):
        """Prediction sets should have valid coverage."""
        X, Y = kernel_cps_dataset
        epsilon = 0.1
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False, epsilon=epsilon)
        cps.learn_initial_training_set(X[:15], Y[:15])

        rng = np.random.default_rng(42)
        covered = 0
        n_test = 40
        for i in range(15, 15 + n_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(X[i], Y[i], precomputed=precomputed)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.15

    def test_learn_one_with_precomputed(self, kernel_cps_dataset):
        """Learning with precomputed should give same state as without."""
        X, Y = kernel_cps_dataset
        kernel = GaussianKernel(sigma=1.0)

        # With precomputed
        cps1 = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False)
        cps1.learn_initial_training_set(X[:15], Y[:15])
        _, precomputed = cps1.predict_cpd(X[15], return_update=True)
        cps1.learn_one(X[15], Y[15], precomputed=precomputed)

        # Without precomputed
        cps2 = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False)
        cps2.learn_initial_training_set(X[:15], Y[:15])
        cps2.learn_one(X[15], Y[15])

        assert np.allclose(cps1.h_diag, cps2.h_diag, atol=1e-10)
        assert np.allclose(cps1.Hy, cps2.Hy, atol=1e-10)
        assert np.allclose(cps1.Kinv, cps2.Kinv, atol=1e-10)
        assert np.allclose(cps1.y, cps2.y)

    def test_learn_one_from_empty(self):
        """Can learn points one by one starting from empty."""
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False)

        rng = np.random.default_rng(0)
        for _i in range(5):
            x = rng.normal(size=(1, 2))
            y = float(rng.normal())
            cps.learn_one(x, y)

        assert cps.X.shape == (5, 2)
        assert len(cps.y) == 5
        assert len(cps.h_diag) == 5
        assert len(cps.Hy) == 5

    def test_predict_after_learn_from_empty(self):
        """Can predict after learning from empty."""
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1, warnings=False)

        rng = np.random.default_rng(1)
        for _i in range(10):
            x = rng.normal(size=(1, 2))
            y = float(rng.normal())
            cps.learn_one(x, y)

        x_test = rng.normal(size=(1, 2))
        cpd = cps.predict_cpd(x_test)
        # CPD should be valid
        assert 0 <= cpd(0.0, tau=0.5) <= 1


class TestMultiLevelCPD:
    def test_predict_set_multi_level(self):
        """CPD.predict_set with list of epsilons returns MultiLevelPredictionInterval."""
        from online_cp.regressors import MultiLevelPredictionInterval

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        y = X @ np.array([1.0, -0.5]) + rng.normal(0, 0.3, 50)

        cps = RidgePredictionMachine(a=1.0)
        cps.learn_initial_training_set(X[:40], y[:40])

        cpd = cps.predict_cpd(X[40])
        tau = 0.5

        epsilons = [0.01, 0.05, 0.1, 0.2]
        result = cpd.predict_set(tau, epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionInterval)
        assert result.levels == sorted(epsilons)
        assert len(result) == 4

        # Smaller epsilon => wider interval
        for i in range(len(epsilons) - 1):
            assert result[epsilons[i]].width() >= result[epsilons[i + 1]].width()


class TestHPDExactSweep:
    """Test predict_set(minimise_width=True) with exact O(n) sweep."""

    def test_hpd_narrower_or_equal(self, ridge_cps_dataset):
        """HPD interval should be ≤ equal-tailed interval in width."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5
        epsilon = 0.1

        equal_tailed = cpd.predict_set(tau, epsilon=epsilon, minimise_width=False)
        hpd = cpd.predict_set(tau, epsilon=epsilon, minimise_width=True)

        assert hpd.width() <= equal_tailed.width() + 1e-10

    def test_hpd_contains_median(self, ridge_cps_dataset):
        """HPD interval should generally contain the median (for unimodal CPDs)."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5
        epsilon = 0.1

        hpd = cpd.predict_set(tau, epsilon=epsilon, minimise_width=True)
        median = cpd.median(tau)
        assert hpd.lower <= median <= hpd.upper

    def test_hpd_width_monotone_in_epsilon(self, ridge_cps_dataset):
        """Smaller epsilon (higher coverage) => wider HPD."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5

        widths = []
        for eps in [0.05, 0.1, 0.2, 0.3]:
            hpd = cpd.predict_set(tau, epsilon=eps, minimise_width=True)
            widths.append(hpd.width())
        # widths should be non-increasing as epsilon increases
        for i in range(len(widths) - 1):
            assert widths[i] >= widths[i + 1] - 1e-10


class TestMedian:
    def test_median_equals_quantile_half(self, ridge_cps_dataset):
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5
        assert cpd.median(tau) == cpd.quantile(0.5, tau)
