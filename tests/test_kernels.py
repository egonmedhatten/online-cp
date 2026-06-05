import numpy as np
import pytest
from scipy.spatial.distance import cdist, squareform, pdist

from online_cp.kernels import (
    CustomKernel,
    GaussianKernel,
    LinearCombinationKernel,
    LinearKernel,
    PeriodicKernel,
    PolynomialKernel,
    ProductKernel,
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

    def test_period_affects_result(self):
        """Verify that changing p actually changes the kernel output."""
        X = np.array([[0.0], [0.5], [1.0], [1.5]])
        k1 = PeriodicKernel(p=1.0, s=1.0)
        k2 = PeriodicKernel(p=2.0, s=1.0)
        K1 = k1(X)
        K2 = k2(X)
        assert not np.allclose(K1, K2)

    def test_period_symmetry(self):
        """Points separated by exactly one period should have k=1."""
        X = np.array([[0.0], [2.0]])
        k = PeriodicKernel(p=2.0, s=1.0)
        K = k(X)
        # sin(pi * |0 - 2| / 2) = sin(pi) = 0 => exp(0) = 1
        assert np.isclose(K[0, 1], 1.0)

    def test_single_vector(self, X):
        k = PeriodicKernel(p=1.0, s=1.0)
        y = X[0]
        result = k(X, y)
        assert result.shape == (X.shape[0],)
        assert np.isclose(result[0], 1.0)


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

    def test_negative_weight_raises(self):
        k1 = LinearKernel()
        k2 = LinearKernel()
        with pytest.raises(ValueError, match="non-negative"):
            LinearCombinationKernel([k1, k2], weights=[1.0, -0.5])


class TestProductKernel:
    def test_element_wise_product(self, X):
        k1 = LinearKernel()
        k2 = GaussianKernel(sigma=1.0)
        k = ProductKernel([k1, k2])
        K = k(X)
        expected = k1(X) * k2(X)
        assert np.allclose(K, expected)

    def test_single_vector(self, X):
        k1 = LinearKernel()
        k2 = GaussianKernel(sigma=1.0)
        k = ProductKernel([k1, k2])
        y = X[0]
        result = k(X, y)
        expected = k1(X, y) * k2(X, y)
        assert np.allclose(result, expected)

    def test_requires_at_least_two(self):
        with pytest.raises(ValueError):
            ProductKernel([LinearKernel()])

    def test_psd(self, X):
        k = ProductKernel([GaussianKernel(sigma=1.0), GaussianKernel(sigma=2.0)])
        K = k(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)


class TestCustomKernel:
    def test_wraps_function(self, X):
        def my_linear(X, y=None):
            X = np.atleast_2d(X)
            if y is None:
                return X @ X.T
            return (X @ np.atleast_1d(y)).ravel()

        k = CustomKernel(my_linear, name="MyLinear")
        K = k(X)
        expected = X @ X.T
        assert np.allclose(K, expected)
        assert k.name == "MyLinear"

    def test_single_vector(self, X):
        k = CustomKernel(lambda X, y=None: LinearKernel()(X, y))
        y = X[2]
        result = k(X, y)
        expected = X @ y
        assert np.allclose(result, expected)

    def test_not_callable_raises(self):
        with pytest.raises(TypeError):
            CustomKernel("not a function")


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


class TestEdgeCases:
    """Edge cases that could reveal subtle bugs."""

    # --- Single-point dataset (n=1) ---

    def test_gaussian_single_point(self):
        """Gram matrix of a single point should be [[1]]."""
        X = np.array([[3.0, 4.0]])
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 1.0)

    def test_linear_single_point(self):
        X = np.array([[2.0, 3.0]])
        k = LinearKernel()
        K = k(X)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 13.0)  # 2^2 + 3^2

    def test_periodic_single_point(self):
        X = np.array([[1.5]])
        k = PeriodicKernel(p=2.0, s=1.0)
        K = k(X)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 1.0)

    def test_kernel_induced_distance_single_point(self):
        """Distance matrix of a single point should be [[0]]."""
        X = np.array([[1.0, 2.0]])
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        D = dist_fn(X)
        assert D.shape == (1, 1)
        assert np.isclose(D[0, 0], 0.0)

    # --- Identical points ---

    def test_gaussian_identical_points(self):
        """All-identical points → Gram matrix of all ones."""
        X = np.array([[1.0, 2.0]] * 5)
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        assert np.allclose(K, np.ones((5, 5)))

    def test_kernel_induced_distance_identical_points(self):
        """Identical points → all distances zero, no NaN."""
        X = np.array([[1.0, 2.0]] * 5)
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        D = dist_fn(X)
        assert not np.any(np.isnan(D))
        assert np.allclose(D, 0.0)

    def test_kernel_induced_distance_identical_point_wise(self):
        """Point-wise call with identical point → zero distance, no NaN."""
        X = np.array([[1.0, 2.0]] * 5)
        y = np.array([1.0, 2.0])
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1.0))
        d = dist_fn(X, y)
        assert not np.any(np.isnan(d))
        assert np.allclose(d, 0.0)

    # --- Extreme sigma (very small → near-zero off-diag) ---

    def test_gaussian_tiny_sigma_no_nan(self):
        """Very small sigma shouldn't produce NaN — just near-identity Gram."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 2))
        k = GaussianKernel(sigma=1e-10)
        K = k(X)
        assert not np.any(np.isnan(K))
        assert np.allclose(np.diag(K), 1.0)
        # Off-diagonal should be ~0
        np.fill_diagonal(K, 0)
        assert np.allclose(K, 0.0, atol=1e-10)

    def test_gaussian_tiny_sigma_distance_no_nan(self):
        """kernel_induced_distance with tiny sigma: no NaN."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 2))
        dist_fn = kernel_induced_distance(GaussianKernel(sigma=1e-10))
        D = dist_fn(X)
        assert not np.any(np.isnan(D))
        # d(x,x) = 0
        assert np.allclose(np.diag(D), 0.0)

    # --- Polynomial d=1, c=0 should equal LinearKernel ---

    def test_polynomial_d1_c0_equals_linear(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 3))
        k_poly = PolynomialKernel(d=1, c=0)
        k_lin = LinearKernel()
        assert np.allclose(k_poly(X), k_lin(X))
        y = X[0]
        assert np.allclose(k_poly(X, y), k_lin(X, y))

    # --- PeriodicKernel half-period → minimum similarity ---

    def test_periodic_half_period_minimum(self):
        """Points separated by p/2 give minimum similarity."""
        p = 3.0
        X = np.array([[0.0], [p / 2]])
        k = PeriodicKernel(p=p, s=1.0)
        K = k(X)
        # sin(pi * (p/2) / p) = sin(pi/2) = 1 → exp(-2/s) is minimum
        expected = np.exp(-2.0 / 1.0)
        assert np.isclose(K[0, 1], expected)

    # --- LinearCombinationKernel / ProductKernel with single-vector call ---

    def test_linear_combination_single_vector(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(10, 3))
        y = X[2]
        k1 = GaussianKernel(sigma=1.0)
        k2 = GaussianKernel(sigma=2.0)
        k = LinearCombinationKernel([k1, k2], weights=[0.3, 0.7])
        result = k(X, y)
        expected = 0.3 * k1(X, y) + 0.7 * k2(X, y)
        assert result.shape == (10,)
        assert np.allclose(result, expected)

    def test_product_kernel_single_vector(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(10, 3))
        y = X[4]
        k1 = GaussianKernel(sigma=1.0)
        k2 = GaussianKernel(sigma=2.0)
        k = ProductKernel([k1, k2])
        result = k(X, y)
        expected = k1(X, y) * k2(X, y)
        assert result.shape == (10,)
        assert np.allclose(result, expected)

    # --- ProductKernel with 3+ kernels ---

    def test_product_kernel_three_kernels(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(8, 2))
        k1 = GaussianKernel(sigma=1.0)
        k2 = GaussianKernel(sigma=2.0)
        k3 = LinearKernel()
        k = ProductKernel([k1, k2, k3])
        K = k(X)
        expected = k1(X) * k2(X) * k3(X)
        assert np.allclose(K, expected)

    # --- 1D scalar features ---

    def test_gaussian_1d_features(self):
        """Scalar features (1D array per point) should work."""
        X = np.array([[0.0], [1.0], [2.0]])
        k = GaussianKernel(sigma=1.0)
        K = k(X)
        assert K.shape == (3, 3)
        assert np.allclose(np.diag(K), 1.0)
        # k(0, 1) = exp(-1/2)
        assert np.isclose(K[0, 1], np.exp(-0.5))

    def test_gaussian_1d_single_vector(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.5])
        k = GaussianKernel(sigma=1.0)
        result = k(X, y)
        assert result.shape == (3,)
        # k(0, 0.5) = exp(-0.25/2) = exp(-0.125)
        assert np.isclose(result[0], np.exp(-0.25 / 2))

    # --- kernel_induced_distance self-distance via point-wise for non-linear kernels ---

    def test_self_distance_zero_all_kernels(self):
        """d(x, x) = 0 via point-wise call for various kernels."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 3))
        kernels = [
            GaussianKernel(sigma=1.0),
            GaussianKernel(sigma=0.1),
            PolynomialKernel(d=2, c=1.0),
            PeriodicKernel(p=2.0, s=0.5),
        ]
        for kernel in kernels:
            dist_fn = kernel_induced_distance(kernel)
            for i in range(5):
                d = dist_fn(X, X[i])
                assert np.isclose(d[i, 0], 0.0, atol=1e-7), (
                    f"Self-distance not zero for {kernel.name}, point {i}: {d[i, 0]}"
                )
