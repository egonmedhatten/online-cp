import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_dataset(rng):
    """y = X @ beta + noise, N=200, d=4."""
    N, d = 200, 4
    X = rng.normal(size=(N, d))
    beta = np.array([2.0, 1.0, -0.5, 0.0])
    y = X @ beta + rng.normal(scale=0.5, size=N)
    return X, y


@pytest.fixture
def classification_dataset(rng):
    """Two-class data: class 0 centered at -1, class 1 at +1. N=200, d=4."""
    N = 200
    y = np.array([0, 1] * (N // 2))
    X = rng.normal(size=(N, 4))
    X[y == 0] -= 1.5
    X[y == 1] += 1.5
    return X, y


@pytest.fixture
def uniform_p_values(rng):
    """500 iid U(0,1) p-values (H0)."""
    return rng.uniform(0, 1, size=500)


@pytest.fixture
def skewed_p_values(rng):
    """500 Beta(0.5, 2) p-values (H1: skewed towards 0)."""
    return rng.beta(0.5, 2, size=500)
