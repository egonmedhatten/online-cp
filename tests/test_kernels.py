import numpy as np
import pytest
from scipy.spatial.distance import cdist, squareform, pdist

from online_cp.kernels import (
    GaussianKernel,
    LinearCombinationKernel,
    LinearKernel,
    PeriodicKernel,
    PolynomialKernel,
    kernel_induced_distance,
    kernel_matrix_to_distance_matrix,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(7)
    return rng.normal(size=(20, 3))


class TestGaussianKernel:
    def test_gram_matrix_symmetric(self, X):
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        assert np.allclose(K, K.T)

    def test_gram_matrix_diagonal_is_one(self, X):
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        assert np.allclose(np.diag(K), 1.0)

    def test_gram_matrix_psd(self, X):
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)

    def test_single_vector_evaluation(self, X):
        k = GaussianKernel(sigma=1.0)
        y = X[0]
        result = k(X, y)
        assert result.shape == (X.shape[0],)
        # Self-similarity should be 1
        assert np.isclose(result[0], 1.0)

    def test_values_in_zero_one(self, X):
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        assert np.all(K >= 0)
        assert np.all(K <= 1.0 + 1e-10)


class TestLinearKernel:
    def test_matches_dot_product(self, X):
        k = LinearKernel()
        K = k(X)
        expected = X @ X.T
        assert np.allclose(K, expected)

    def test_single_vector(self, X):
        k = LinearKernel()
        y = X[0]
        result = k(X, y)
        expected = X @ y
        assert np.allclose(result, expected)


class TestPolynomialKernel:
    def test_formula(self, X):
        k = PolynomialKernel(d=2, c=1)
        K = k(X)
        expected = (X @ X.T + 1) ** 2
        assert np.allclose(K, expected)

    def test_single_vector(self, X):
        k = PolynomialKernel(d=3, c=0.5)
        y = X[0]
        result = k(X, y)
        expected = (X @ y + 0.5) ** 3
        assert np.allclose(result, expected)


class TestPeriodicKernel:
    def test_gram_matrix_symmetric(self, X):
        k = PeriodicKernel(p=1.0, s=1.0)
        K = k(X)
        assert np.allclose(K, K.T)

    def test_diagonal_is_one(self, X):
        k = PeriodicKernel(p=1.0, s=1.0)
        K = k(X)
        assert np.allclose(np.diag(K), 1.0)


class TestLinearCombinationKernel:
    def test_weighted_sum(self, X):
        k1 = LinearKernel()
        k2 = LinearKernel()
        k = LinearCombinationKernel([k1, k2], weights=[0.5, 0.5])
        K = k(X)
        expected = (k1(X) + k2(X)) / 2
        assert np.allclose(K, expected)

    def test_mismatched_lengths_raises(self):
        k1 = LinearKernel()
        with pytest.raises(ValueError):
            LinearCombinationKernel([k1], weights=[0.5, 0.5])


class TestKernelInducedDistance:
    """Tests for kernel_induced_distance utility."""

    def test_linear_kernel_equals_euclidean(self, X):
        """Linear kernel distance should equal Euclidean distance."""
        dist_fn = kernel_induced_distance(LinearKernel())
        D_kernel = dist_fn(X)
        D_euclidean = squareform(pdist(X, metric="euclidean"))
        assert np.allclose(D_kernel, D_euclidean, atol=1e-10)

    def test_linear_kernel_point_equals_euclidean(self, X):
        """Point-wise linear kernel distance should equal Euclidean cdist."""
        dist_fn = kernel_induced_distance(LinearKernel())
        y = X[3]
        d_kernel = dist_fn(X, y)
        d_euclidean = cdist(X, y.reshape(1, -1), metric="euclidean")
        assert np.allclose(d_kernel, d_euclidean, atol=1e-10)

    def test_symmetric(self, X):
        """Distance matrix should be symmetric."""
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        D = dist_fn(X)
        assert np.allclose(D, D.T)

    def test_zero_diagonal(self, X):
        """Distance from a point to itself should be zero."""
        dist_fn = kernel_induced_distance(PolynomialKernel(d=2, c=1.0))
        D = dist_fn(X)
        assert np.allclose(np.diag(D), 0.0)

    def test_non_negative(self, X):
        """All distances should be non-negative."""
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=0.5))
        D = dist_fn(X)
        assert np.all(D >= 0.0)

    def test_no_nan(self, X):
        """No NaN values from floating-point issues."""
        for kernel in [
            GaussianKernel(sigma=0.1),
            LinearKernel(),
            PolynomialKernel(d=3, c=2.0),
            PeriodicKernel(p=1.0, s=1.0),
        ]:
            dist_fn = kernel_induced_distance(kernel)
            D = dist_fn(X)
            assert not np.any(np.isnan(D))
            d = dist_fn(X, X[0])
            assert not np.any(np.isnan(d))

    def test_point_output_shape(self, X):
        """Point-wise distance should return column vector."""
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        d = dist_fn(X, X[5])
        assert d.shape == (X.shape[0], 1)

    def test_matrix_output_shape(self, X):
        """Full matrix should be square with correct size."""
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        D = dist_fn(X)
        assert D.shape == (X.shape[0], X.shape[0])

    def test_gaussian_same_rankings_as_euclidean(self, X):
        """Gaussian kernel distance preserves nearest-neighbour rankings."""
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        D_kernel = dist_fn(X)
        D_euclidean = squareform(pdist(X, metric="euclidean"))
        # For each row, check that the sorted order is the same
        for i in range(len(X)):
            rank_kernel = np.argsort(D_kernel[i])
            rank_euclidean = np.argsort(D_euclidean[i])
            assert np.array_equal(rank_kernel, rank_euclidean)

    def test_self_distance_zero_point(self, X):
        """Distance from a point to itself via point-wise call should be zero."""
        dist_fn = kernel_induced_distance(LinearKernel())
        for i in range(5):
            d = dist_fn(X, X[i])
            assert np.isclose(d[i, 0], 0.0, atol=1e-7)

    def test_triangle_inequality(self, X):
        """Kernel-induced distance should satisfy triangle inequality."""
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        D = dist_fn(X[:5])
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    assert D[i, j] <= D[i, k] + D[k, j] + 1e-10


class TestKernelMatrixToDistanceMatrix:
    """Tests for kernel_matrix_to_distance_matrix utility."""

    def test_linear_matches_euclidean(self, X):
        K = LinearKernel()(X)
        D = kernel_matrix_to_distance_matrix(K)
        D_expected = squareform(pdist(X, metric="euclidean"))
        assert np.allclose(D, D_expected, atol=1e-10)

    def test_symmetric(self, X):
        K = GaussianKernel(sigma=1.0)(X)
        D = kernel_matrix_to_distance_matrix(K)
        assert np.allclose(D, D.T)

    def test_zero_diagonal(self, X):
        K = PolynomialKernel(d=2, c=1.0)(X)
        D = kernel_matrix_to_distance_matrix(K)
        assert np.allclose(np.diag(D), 0.0)

    def test_consistent_with_distance_func(self, X):
        """Should match kernel_induced_distance for the full-matrix case."""
        kernel = GaussianKernel(sigma=2.0)
        dist_fn = kernel_induced_distance(kernel)
        D1 = dist_fn(X)
        K = kernel(X)
        D2 = kernel_matrix_to_distance_matrix(K)
        assert np.allclose(D1, D2)


class TestKernelDistanceIntegration:
    """Integration tests: kernel distance with conformal classifiers."""

    def test_classifier_with_kernel_distance(self):
        """Conformal k-NN classifier works with kernel-induced distance."""
        from online_cp import ConformalNearestNeighboursClassifier

        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(loc=[2, 0], size=(30, 2)),
                       rng.normal(loc=[-2, 0], size=(30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        idx = rng.permutation(60)
        X, y = X[idx], y[idx]

        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        clf = ConformalNearestNeighboursClassifier(
            k=5, label_space=np.array([0, 1]), distance_func=dist_fn
        )
        clf.learn_initial_training_set(X[:40], y[:40])

        errors = 0
        for i in range(40, 60):
            pred = clf.predict(X[i], epsilon=0.1)
            if y[i] not in pred:
                errors += 1
            clf.learn_one(X[i], y[i])

        # Validity: error rate should be near epsilon=0.1
        assert errors <= 6  # generous bound for 20 predictions

    def test_cps_with_kernel_distance(self):
        """Nearest neighbours CPS works with kernel-induced distance."""
        from online_cp import NearestNeighboursPredictionMachine

        rng = np.random.default_rng(99)
        X = rng.normal(size=(40, 2))
        y = (X[:, 0] > 0).astype(int)

        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.5))
        cps = NearestNeighboursPredictionMachine(
            k=3, distance_func=dist_fn
        )
        cps.learn_initial_training_set(X[:25], y[:25])

        for i in range(25, 40):
            cpd = cps.predict_cpd(X[i])
            # Should return a predictive distribution; verify it's callable
            assert cpd is not None
            cps.learn_one(X[i], y[i])

    def test_regressor_with_kernel_distance(self):
        """Conformal k-NN regressor works with kernel-induced distance."""
        from online_cp import ConformalNearestNeighboursRegressor

        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (60, 2))
        y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.normal(size=60)

        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        cp = ConformalNearestNeighboursRegressor(
            k=5, distance_func=dist_fn, rnd_state=0, epsilon=0.1
        )
        cp.learn_initial_training_set(X[:40], y[:40])

        covered = 0
        for i in range(40, 60):
            interval = cp.predict(X[i])
            covered += int(y[i] in interval)
            cp.learn_one(X[i], y[i])

        # Validity: coverage should be near 1 - epsilon = 0.9
        assert covered >= 15  # at least 75% of 20 (generous for small sample)
