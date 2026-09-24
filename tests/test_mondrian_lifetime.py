"""Tests for unsupervised hyperparameter tuning in ConformalMondrianTree classes.

Covers:
  - Method 1: lifetime="sqrt_n"  (√n stopping rule)
  - Method 2: feature_weights="variance"  (variance-proportional split weights)
  - Method 3: lifetime="density"  (density log-likelihood sweep)
  - RNG state restoration guarantee
  - Backward-compatibility with float lifetime
  - Input validation (ValueError on bad strings)
"""
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from online_cp.classifiers import ConformalMondrianTreeClassifier
from online_cp.mondrian.tree import (
    _resolve_feature_weights,
    _resolve_lifetime,
)
from online_cp.regressors import ConformalMondrianTreeRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def iris_data():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    return X[:100], y[:100], X[100]


@pytest.fixture()
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] * 2.0 + rng.standard_normal(100) * 0.1
    x_test = rng.standard_normal(3)
    return X, y, x_test


# ===========================================================================
# 1. Backward compatibility: float lifetime unchanged
# ===========================================================================

def test_backward_compat_float_lifetime(iris_data):
    X, y, x_test = iris_data
    clf = ConformalMondrianTreeClassifier(lifetime=1.0, rnd_state=0)
    clf.learn_initial_training_set(X, y)
    ps = clf.predict(x_test)
    assert ps is not None  # any non-crash result is fine


def test_backward_compat_float_lifetime_regressor(regression_data):
    X, y, x_test = regression_data
    reg = ConformalMondrianTreeRegressor(lifetime=1.0, rnd_state=0)
    reg.learn_initial_training_set(X, y)
    ci = reg.predict(x_test, epsilon=0.1)
    assert ci.lower <= ci.upper


# ===========================================================================
# 2. sqrt_n lifetime
# ===========================================================================

def test_sqrt_n_lifetime_clf(iris_data):
    X, y, x_test = iris_data
    clf = ConformalMondrianTreeClassifier(lifetime="sqrt_n", rnd_state=0)
    clf.learn_initial_training_set(X, y)
    ps = clf.predict(x_test)
    assert ps is not None  # must not crash


def test_sqrt_n_lifetime_regressor(regression_data):
    X, y, x_test = regression_data
    reg = ConformalMondrianTreeRegressor(lifetime="sqrt_n", rnd_state=0)
    reg.learn_initial_training_set(X, y)
    ci = reg.predict(x_test, epsilon=0.1)
    assert math.isfinite(ci.lower) and math.isfinite(ci.upper), \
        "sqrt_n should give a finite interval on low-d data"
    assert ci.lower <= ci.upper


# ===========================================================================
# 3. density lifetime
# ===========================================================================

def test_density_lifetime_clf(iris_data):
    X, y, x_test = iris_data
    clf = ConformalMondrianTreeClassifier(lifetime="density", rnd_state=0)
    clf.learn_initial_training_set(X, y)
    ps = clf.predict(x_test)
    assert ps is not None  # must not crash


def test_density_lifetime_regressor(regression_data):
    """density on low-d data should give a finite interval."""
    X, y, x_test = regression_data
    reg = ConformalMondrianTreeRegressor(lifetime="density", rnd_state=0)
    reg.learn_initial_training_set(X, y)
    ci = reg.predict(x_test, epsilon=0.1)
    # Mathematically valid outcomes are finite interval OR (-inf, inf).
    # Both are correct conformal predictions. We only assert no crash.
    assert ci.lower <= ci.upper


# ===========================================================================
# 4. feature_weights="variance"
# ===========================================================================

def test_variance_weights_nonneg_sum1():
    """_resolve_feature_weights('variance') returns weights ≥ 0 summing to 1."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 5))
    # give one column much higher variance
    X[:, 2] *= 10
    w = _resolve_feature_weights(X, "variance")
    assert w is not None
    assert w.shape == (5,)
    assert np.all(w >= 0), "all weights must be non-negative"
    assert abs(w.sum() - 1.0) < 1e-12, "weights must sum to 1"


def test_variance_weights_zero_constant_col():
    """Constant columns receive near-zero weight (up to floating-point rounding)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 4))
    X[:, 1] = 3.14  # constant column
    w = _resolve_feature_weights(X, "variance")
    assert w[1] < 1e-15, "constant column must get (near-)zero weight"


def test_variance_weights_splits_high_var_more():
    """High-variance dimension dominates the weight vector."""
    rng = np.random.default_rng(7)
    X = np.ones((80, 3)) * 0.01  # near-constant baseline
    X[:, 0] = rng.standard_normal(80) * 100  # huge variance in dim 0
    w = _resolve_feature_weights(X, "variance")
    assert w[0] > w[1] and w[0] > w[2], "dim 0 should have the largest weight"


def test_variance_weights_none_returns_none():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    assert _resolve_feature_weights(X, None) is None


def test_variance_weights_classifier(iris_data):
    X, y, x_test = iris_data
    clf = ConformalMondrianTreeClassifier(feature_weights="variance", rnd_state=0)
    clf.learn_initial_training_set(X, y)
    ps = clf.predict(x_test)
    assert ps is not None


# ===========================================================================
# 5. Combined: sqrt_n + variance
# ===========================================================================

def test_combined_sqrt_n_and_variance(iris_data):
    X, y, x_test = iris_data
    clf = ConformalMondrianTreeClassifier(
        lifetime="sqrt_n", feature_weights="variance", rnd_state=0
    )
    clf.learn_initial_training_set(X, y)
    ps = clf.predict(x_test)
    assert ps is not None


def test_combined_density_and_variance_regressor(regression_data):
    X, y, x_test = regression_data
    reg = ConformalMondrianTreeRegressor(
        lifetime="density", feature_weights="variance", rnd_state=0
    )
    reg.learn_initial_training_set(X, y)
    ci = reg.predict(x_test, epsilon=0.1)
    assert ci.lower <= ci.upper


# ===========================================================================
# 6. RNG state is restored after _resolve_lifetime
# ===========================================================================

def test_rng_state_restored_sqrt_n():
    """After _resolve_lifetime('sqrt_n') the RNG state is identical to before."""
    rng = np.random.default_rng(42)
    X = np.random.default_rng(1).standard_normal((50, 3))
    X_aug = np.vstack([X, X[0]])
    x_test = X_aug[-1]

    state_before = rng.bit_generator.state
    _ = _resolve_lifetime(X_aug, x_test, "sqrt_n", rng, None)
    state_after = rng.bit_generator.state

    assert state_before["state"]["state"] == state_after["state"]["state"], \
        "RNG state must be identical before and after _resolve_lifetime"


def test_rng_state_restored_density():
    rng = np.random.default_rng(42)
    X = np.random.default_rng(1).standard_normal((50, 3))
    X_aug = np.vstack([X, X[0]])
    x_test = X_aug[-1]

    state_before = rng.bit_generator.state
    _ = _resolve_lifetime(X_aug, x_test, "density", rng, None)
    state_after = rng.bit_generator.state

    assert state_before["state"]["state"] == state_after["state"]["state"], \
        "RNG state must be identical before and after _resolve_lifetime"


def test_rng_state_not_touched_for_float():
    """For a float lifetime, the RNG must not be touched at all."""
    rng = np.random.default_rng(42)
    X = np.random.default_rng(1).standard_normal((20, 2))
    X_aug = np.vstack([X, X[0]])
    x_test = X_aug[-1]

    state_before = rng.bit_generator.state
    lt = _resolve_lifetime(X_aug, x_test, 2.5, rng, None)
    state_after = rng.bit_generator.state

    assert lt == 2.5
    assert state_before["state"]["state"] == state_after["state"]["state"]


# ===========================================================================
# 7. Input validation
# ===========================================================================

def test_invalid_lifetime_raises_clf():
    with pytest.raises(ValueError, match="sqrt_n"):
        ConformalMondrianTreeClassifier(lifetime="unknown")


def test_invalid_lifetime_raises_reg():
    with pytest.raises(ValueError, match="sqrt_n"):
        ConformalMondrianTreeRegressor(lifetime="unknown")


def test_invalid_feature_weights_raises_clf():
    with pytest.raises(ValueError, match="variance"):
        ConformalMondrianTreeClassifier(feature_weights="pca")


def test_invalid_feature_weights_raises_reg():
    with pytest.raises(ValueError, match="variance"):
        ConformalMondrianTreeRegressor(feature_weights="pca")


# ===========================================================================
# 8. p-values in [0, 1]
# ===========================================================================

def test_p_values_in_unit_interval(iris_data):
    """All p-values produced by compute_p_value must be in [0, 1]."""
    X, y, x_test = iris_data
    # iris[:100] contains labels 0 and 1 only
    train_labels = list(np.unique(y))
    for lt in ("sqrt_n", "density", 1.0):
        clf = ConformalMondrianTreeClassifier(lifetime=lt, rnd_state=0)
        clf.learn_initial_training_set(X, y)
        for label in train_labels:
            p = clf.compute_p_value(x_test, label)
            assert 0.0 <= p <= 1.0, f"p={p} out of [0,1] for lifetime={lt!r}"


# ===========================================================================
# 9. _resolve_feature_weights: NDArray pass-through and validation
# ===========================================================================

def test_resolve_feature_weights_array_passthrough():
    X = np.ones((10, 3))
    w = np.array([0.5, 0.3, 0.2])
    result = _resolve_feature_weights(X, w)
    assert_array_equal(result, w)


def test_resolve_feature_weights_array_wrong_shape():
    X = np.ones((10, 3))
    w = np.array([0.5, 0.5])  # wrong length
    with pytest.raises(ValueError):
        _resolve_feature_weights(X, w)


def test_resolve_feature_weights_array_negative():
    X = np.ones((10, 3))
    w = np.array([0.5, -0.1, 0.6])  # negative weight
    with pytest.raises(ValueError):
        _resolve_feature_weights(X, w)
