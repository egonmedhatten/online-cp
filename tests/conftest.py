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


# --------------------------------------------------------------------------- #
# Data generators
# --------------------------------------------------------------------------- #

def generate_linear_data(rng, N, d, noise_scale=0.5):
    """Generate linear regression data with exchangeable observations."""
    X = rng.normal(size=(N, d))
    beta = rng.normal(size=d)
    y = X @ beta + rng.normal(scale=noise_scale, size=N)
    return X, y


def generate_binary_classification_data(rng, N, d):
    """Generate binary classification data with clear separation."""
    y = np.array([0, 1] * (N // 2))
    X = rng.normal(size=(N, d))
    X[y == 0] -= 1.5
    X[y == 1] += 1.5
    return X, y


def generate_multiclass_classification_data(rng, N, d, n_classes=4):
    """Generate multiclass classification data with clear separation."""
    y = np.tile(np.arange(n_classes), N // n_classes)
    X = rng.normal(size=(N, d))
    for i in range(n_classes):
        X[y == i] += i * 2.0
    return X, y


# --------------------------------------------------------------------------- #
# Helper functions for statistical testing
# --------------------------------------------------------------------------- #

def run_binomial_test(errors, trials, epsilon, alternative="greater"):
    """Run exact binomial test for coverage validity.

    Args:
        errors: Number of coverage violations (errors)
        trials: Total number of predictions
        epsilon: Target significance level (max allowed error rate)
        alternative: One of 'greater', 'less', 'two-sided'

    Returns:
        scipy.stats.binomtest result
    """
    from scipy.stats import binomtest
    return binomtest(errors, trials, epsilon, alternative=alternative)


def run_calibration_test(pred_probs, actual, alpha=0.01):
    """Run calibration-in-the-large test for Venn predictors.

    Args:
        pred_probs: Array of predicted probabilities (mean should equal empirical freq)
        actual: Array of binary outcomes
        alpha: Significance level

    Returns:
        (passed: bool, pvalue: float)
    """
    from scipy.stats import binomtest
    n_positives = int(actual.sum())
    n_trials = len(actual)
    mean_pred = float(pred_probs.mean())
    result = binomtest(n_positives, n_trials, mean_pred, alternative="two-sided")
    return result.pvalue > alpha, result.pvalue


# --------------------------------------------------------------------------- #
# Fixtures for common test configurations
# --------------------------------------------------------------------------- #

@pytest.fixture
def regressor_rng():
    """Random generator for regressor tests."""
    return np.random.default_rng(0)


@pytest.fixture
def classifier_rng():
    """Random generator for classifier tests."""
    return np.random.default_rng(1)


@pytest.fixture
def cps_rng():
    """Random generator for CPS tests."""
    return np.random.default_rng(2)


@pytest.fixture
def venn_rng():
    """Random generator for Venn tests."""
    return np.random.default_rng(2)


@pytest.fixture
def martingale_rng():
    """Random generator for martingale tests."""
    return np.random.default_rng(3)
