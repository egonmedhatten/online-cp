import numpy as np
import pytest
from online_cp.CPS import (
    RidgePredictionMachine,
    NearestNeighboursPredictionMachine,
    DempsterHillConformalPredictiveSystem,
)


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

    def test_requires_distinct_labels(self):
        cps = NearestNeighboursPredictionMachine(k=3)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 1.0, 2.0])  # Duplicate labels
        with pytest.raises(Exception):
            cps.learn_initial_training_set(X, y)


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
