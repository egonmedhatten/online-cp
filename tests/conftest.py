import pathlib

import numpy as np
import pytest


def _numba_cache_dirs() -> list[pathlib.Path]:
    """Return the ``__pycache__`` directories that hold Numba's on-disk
    ``.nbc``/``.nbi`` cache files for the ``online_cp`` package.

    The package is laid out under ``<repo>/src/online_cp`` (see
    ``[tool.pytest.ini_options] pythonpath`` in ``pyproject.toml``).
    """
    # This file lives at <repo>/tests/conftest.py, so the repo root is two
    # levels up.
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    src_online_cp = repo_root / "src" / "online_cp"
    if not src_online_cp.is_dir():
        return []
    dirs = [p.parent for p in src_online_cp.rglob("*.nbc")]
    dirs = [p.parent for p in src_online_cp.rglob("*.nbi")]
    # Deduplicate while preserving order.
    seen: set[pathlib.Path] = set()
    out: list[pathlib.Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


@pytest.fixture(scope="session", autouse=True)
def _fresh_numba_cache():
    """Clear Numba's on-disk ``.nbc``/``.nbi`` cache before the test session.

    Numba's ``@njit(cache=True)`` invalidates entries by comparing the source
    file's ``(mtime, size)`` stamp. That stamp has **second** granularity, so
    rapid successive edits that land within the same wall-clock second can
    produce a *stale* cache entry that Numba believes is fresh, loading a
    ``.nbc`` that no longer matches the source. On this platform that
    manifests as a hard segfault mid-test rather than a clean recompile.

    Deleting the cache files once per session guarantees every run recompiles
    from the current source. The one-time recompile cost (~1 s for the two
    Mondrian kernels) is negligible against the ~15 s test runtime, and it
    removes the need to manually purge ``__pycache__`` between edits.

    The fixture is ``session``-scoped and ``autouse`` so it runs exactly once
    before the first test, regardless of which test files are selected.
    """
    cleared: list[str] = []
    for d in _numba_cache_dirs():
        for suffix in (".nbc", ".nbi"):
            for f in d.glob(f"*{suffix}"):
                try:
                    f.unlink()
                    cleared.append(str(f))
                except OSError:
                    pass
    # Expose the result for debugging (not asserted on).
    if cleared:
        # A single short line is enough; the full list is noisy.
        print(f"[conftest] cleared {len(cleared)} stale Numba cache file(s)")
    yield


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
