import numpy as np
import pytest
from online_cp.kernels import GaussianKernel, LinearKernel, PolynomialKernel, PeriodicKernel, LinearCombinationKernel


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
