import numpy as np
import pytest

from online_cp.regressors import (
    ConformalNearestNeighboursRegressor,
    MultiLevelPredictionInterval,
)


@pytest.fixture
def nonlinear_dataset():
    """y = sin(2*pi*x) + noise. N=200, d=1."""
    rng = np.random.default_rng(42)
    N = 200
    X = rng.uniform(0, 1, (N, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(scale=0.1, size=N)
    return X, y


@pytest.fixture
def linear_dataset_2d():
    """y = x1 + x2 + noise. N=200, d=2."""
    rng = np.random.default_rng(123)
    N = 200
    X = rng.uniform(0, 1, (N, 2))
    y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=N)
    return X, y


class TestConformalNearestNeighboursRegressor:
    def test_validity_k1(self, nonlinear_dataset):
        """Coverage should be >= 1 - epsilon on iid data (k=1)."""
        X, y = nonlinear_dataset
        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0, epsilon=epsilon)

        n_init = 20
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_validity_k5(self, nonlinear_dataset):
        """Coverage should be >= 1 - epsilon on iid data (k=5)."""
        X, y = nonlinear_dataset
        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=5, rnd_state=0, epsilon=epsilon)

        n_init = 20
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_validity_median_aggregation(self, nonlinear_dataset):
        """Coverage with median aggregation."""
        X, y = nonlinear_dataset
        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=5, aggregation="median", rnd_state=0, epsilon=epsilon)

        n_init = 20
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_finite_intervals(self, nonlinear_dataset):
        """After sufficient training, intervals should be finite."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        Gamma = cp.predict(X[30], epsilon=0.1)
        assert Gamma.lower > -np.inf
        assert Gamma.upper < np.inf
        assert Gamma.width() > 0

    def test_infinite_interval_with_no_data(self):
        """With no training data, interval should be infinite."""
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        Gamma = cp.predict(np.array([0.5]), epsilon=0.1)
        assert Gamma.lower == -np.inf or Gamma.upper == np.inf

    def test_learn_one_no_initial(self):
        """learn_one from empty state."""
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        cp.learn_one(np.array([1.0, 2.0]), 3.0)
        assert cp.X.shape == (1, 2)
        assert cp.y[0] == 3.0

    def test_learn_one_incremental(self, nonlinear_dataset):
        """learn_one should grow the distance matrix correctly."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        cp.learn_initial_training_set(X[:5], y[:5])

        assert cp.D.shape == (5, 5)
        cp.learn_one(X[5], y[5])
        assert cp.D.shape == (6, 6)
        assert cp.X.shape[0] == 6
        assert len(cp.y) == 6

    def test_distance_matrix_symmetric(self, nonlinear_dataset):
        """Distance matrix should remain symmetric after updates."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        cp.learn_initial_training_set(X[:10], y[:10])

        for i in range(10, 15):
            cp.learn_one(X[i], y[i])

        assert np.allclose(cp.D, cp.D.T)
        assert np.allclose(np.diag(cp.D), 0)

    def test_return_update(self, nonlinear_dataset):
        """return_update should provide D for efficient learn_one."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])

        Gamma, precomputed = cp.predict(X[20], epsilon=0.1, return_update=True)
        assert "D" in precomputed
        assert precomputed["D"].shape == (21, 21)

        # Using precomputed should give same result as not using it
        cp2 = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp2.learn_initial_training_set(X[:20], y[:20])

        cp.learn_one(X[20], y[20], precomputed=precomputed)
        cp2.learn_one(X[20], y[20])

        assert np.allclose(cp.D, cp2.D)

    def test_multi_level_epsilon(self, nonlinear_dataset):
        """Array-valued epsilon should return MultiLevelPredictionInterval."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        epsilons = np.array([0.05, 0.1, 0.2])
        result = cp.predict(X[30], epsilon=epsilons)
        assert isinstance(result, MultiLevelPredictionInterval)
        assert len(result) == 3
        # Wider intervals at lower epsilon
        assert result[0.05].width() >= result[0.1].width()
        assert result[0.1].width() >= result[0.2].width()

    def test_custom_distance_func(self, linear_dataset_2d):
        """Custom distance function should work."""
        X, y = linear_dataset_2d

        def manhattan(X, y=None):
            from scipy.spatial.distance import cdist, pdist, squareform
            X = np.atleast_2d(X)
            if y is None:
                return squareform(pdist(X, metric="cityblock"))
            else:
                y = np.atleast_2d(y)
                return cdist(X, y, metric="cityblock")

        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=3, distance_func=manhattan, rnd_state=0, epsilon=epsilon)
        cp.learn_initial_training_set(X[:30], y[:30])

        covered = 0
        n_test = 50
        for obj, lab in zip(X[30:30 + n_test], y[30:30 + n_test]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.1, f"Coverage {coverage:.3f} too low"

    def test_invalid_aggregation(self):
        """Should raise ValueError for invalid aggregation."""
        with pytest.raises(ValueError, match="aggregation"):
            ConformalNearestNeighboursRegressor(aggregation="sum")

    def test_k_larger_than_n(self):
        """Should handle k > n gracefully."""
        cp = ConformalNearestNeighboursRegressor(k=100, rnd_state=0)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])
        cp.learn_initial_training_set(X, y)

        # Should not crash — uses all available neighbours
        Gamma = cp.predict(np.array([1.5]), epsilon=0.5)
        assert Gamma.lower <= 1.5 <= Gamma.upper

    def test_bounds_lower(self, nonlinear_dataset):
        """Lower-only bound should have upper = inf."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        Gamma = cp.predict(X[30], epsilon=0.1, bounds="lower")
        assert Gamma.lower > -np.inf
        assert Gamma.upper == np.inf

    def test_bounds_upper(self, nonlinear_dataset):
        """Upper-only bound should have lower = -inf."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        Gamma = cp.predict(X[30], epsilon=0.1, bounds="upper")
        assert Gamma.lower == -np.inf
        assert Gamma.upper < np.inf
