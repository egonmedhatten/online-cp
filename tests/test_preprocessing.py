"""Tests for online_cp.preprocessing (StandardScaler, MinMaxScaler)."""

import numpy as np
import pytest

from online_cp import (
    ConformalRidgeRegressor,
    ErrorRate,
    MinMaxScaler,
    Pipeline,
    StandardScaler,
    progressive_val,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def small_batch(rng):
    """20×3 matrix with very different per-feature scales."""
    X = rng.normal(size=(20, 3))
    X[:, 0] *= 100.0   # large scale
    X[:, 1] *= 0.001   # tiny scale
    # X[:,2] is ~ N(0,1)
    return X


# ===========================================================================
# StandardScaler
# ===========================================================================


class TestStandardScaler:
    def test_fit_stores_mean_and_scale(self, small_batch):
        sc = StandardScaler()
        sc.fit(small_batch)
        np.testing.assert_allclose(sc.mean_, small_batch.mean(axis=0))
        np.testing.assert_allclose(sc.scale_, small_batch.std(axis=0))

    def test_transform_zero_mean_unit_std(self, small_batch):
        sc = StandardScaler()
        sc.fit(small_batch)
        Xt = sc.transform(small_batch)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-10)

    def test_transform_one_matches_transform(self, small_batch):
        sc = StandardScaler()
        sc.fit(small_batch)
        for x in small_batch:
            np.testing.assert_allclose(sc.transform_one(x), sc.transform(x.reshape(1, -1))[0])

    def test_with_mean_false(self, small_batch):
        sc = StandardScaler(with_mean=False)
        sc.fit(small_batch)
        Xt = sc.transform(small_batch)
        # mean should NOT be zero (original mean preserved)
        assert not np.allclose(Xt.mean(axis=0), 0.0)
        # std should be ~1
        np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-10)

    def test_with_std_false(self, small_batch):
        sc = StandardScaler(with_std=False)
        sc.fit(small_batch)
        Xt = sc.transform(small_batch)
        # mean should be zero
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-10)
        # std should be the original std (not normalised)
        np.testing.assert_allclose(Xt.std(axis=0), small_batch.std(axis=0), atol=1e-10)

    def test_zero_variance_column_not_nan(self, rng):
        X = rng.normal(size=(10, 3))
        X[:, 1] = 5.0  # constant column
        sc = StandardScaler()
        sc.fit(X)
        Xt = sc.transform(X)
        assert np.all(np.isfinite(Xt))
        # Constant column → all zeros after mean-subtraction, scale=1
        np.testing.assert_allclose(Xt[:, 1], 0.0, atol=1e-10)

    def test_raises_before_fit(self):
        sc = StandardScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            sc.transform(np.ones((3, 2)))

    def test_raises_before_fit_one(self):
        sc = StandardScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            sc.transform_one(np.ones(2))

    def test_mode_is_frozen(self):
        assert StandardScaler().mode == "frozen"

    def test_fit_called_by_pipeline(self, small_batch, rng):
        """Pipeline.learn_initial_training_set must call fit on the scaler."""
        X = small_batch
        y = X[:, 0] + rng.normal(scale=0.1, size=len(X))
        sc = StandardScaler()
        pipe = Pipeline(sc, ConformalRidgeRegressor(a=1.0))
        pipe.learn_initial_training_set(X, y)
        # After learn_initial, scaler must be fitted
        assert sc.mean_ is not None
        assert sc.scale_ is not None

    def test_params_permutation_invariant(self, rng):
        """mean_ and scale_ must not depend on row order (symmetry check)."""
        X = rng.normal(size=(30, 4))
        perm = rng.permutation(len(X))
        sc1 = StandardScaler()
        sc1.fit(X)
        sc2 = StandardScaler()
        sc2.fit(X[perm])
        np.testing.assert_allclose(sc1.mean_, sc2.mean_)
        np.testing.assert_allclose(sc1.scale_, sc2.scale_)


# ===========================================================================
# MinMaxScaler
# ===========================================================================


class TestMinMaxScaler:
    def test_fit_stores_min_and_range(self, small_batch):
        sc = MinMaxScaler()
        sc.fit(small_batch)
        np.testing.assert_allclose(sc.data_min_, small_batch.min(axis=0))
        expected_range = small_batch.max(axis=0) - small_batch.min(axis=0)
        np.testing.assert_allclose(sc.data_range_, expected_range)

    def test_transform_in_unit_range(self, small_batch):
        sc = MinMaxScaler()
        sc.fit(small_batch)
        Xt = sc.transform(small_batch)
        assert Xt.min() >= -1e-12
        assert Xt.max() <= 1.0 + 1e-12

    def test_transform_min_is_0_max_is_1(self, small_batch):
        sc = MinMaxScaler()
        sc.fit(small_batch)
        Xt = sc.transform(small_batch)
        np.testing.assert_allclose(Xt.min(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xt.max(axis=0), 1.0, atol=1e-10)

    def test_custom_feature_range(self, small_batch):
        sc = MinMaxScaler(feature_range=(-1.0, 1.0))
        sc.fit(small_batch)
        Xt = sc.transform(small_batch)
        assert Xt.min() >= -1.0 - 1e-12
        assert Xt.max() <= 1.0 + 1e-12

    def test_transform_one_matches_transform(self, small_batch):
        sc = MinMaxScaler()
        sc.fit(small_batch)
        for x in small_batch:
            np.testing.assert_allclose(sc.transform_one(x), sc.transform(x.reshape(1, -1))[0])

    def test_constant_column_no_nan(self, rng):
        X = rng.normal(size=(10, 3))
        X[:, 2] = 3.0  # constant column
        sc = MinMaxScaler()
        sc.fit(X)
        Xt = sc.transform(X)
        assert np.all(np.isfinite(Xt))

    def test_invalid_feature_range_raises(self):
        with pytest.raises(ValueError, match="lo < hi"):
            MinMaxScaler(feature_range=(1.0, 0.0))

    def test_raises_before_fit(self):
        sc = MinMaxScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            sc.transform(np.ones((3, 2)))

    def test_raises_before_fit_one(self):
        sc = MinMaxScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            sc.transform_one(np.ones(2))

    def test_mode_is_frozen(self):
        assert MinMaxScaler().mode == "frozen"

    def test_params_permutation_invariant(self, rng):
        """data_min_ and data_range_ must not depend on row order."""
        X = rng.normal(size=(30, 4))
        perm = rng.permutation(len(X))
        sc1 = MinMaxScaler()
        sc1.fit(X)
        sc2 = MinMaxScaler()
        sc2.fit(X[perm])
        np.testing.assert_allclose(sc1.data_min_, sc2.data_min_)
        np.testing.assert_allclose(sc1.data_range_, sc2.data_range_)


# ===========================================================================
# Integration: coverage with frozen scalers on poorly-scaled features
# ===========================================================================


def test_standard_scaler_pipeline_nominal_coverage(rng):
    """Frozen StandardScaler in a Pipeline keeps nominal conformal coverage."""
    N, d = 300, 2
    scale = np.array([1000.0, 0.001])
    X_raw = rng.normal(size=(N, d))
    X = X_raw * scale
    beta = np.array([0.001, 1000.0])
    y = X @ beta + rng.normal(scale=0.5, size=N)

    n_train = 100
    epsilon = 0.1
    margin = 0.15

    pipe = StandardScaler() | ConformalRidgeRegressor(a=1.0, epsilon=epsilon)
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])

    metric = ErrorRate()
    progressive_val(pipe, X[n_train:], y[n_train:], metric=metric, epsilon=epsilon)
    assert metric.get() <= epsilon + margin


def test_minmax_scaler_pipeline_nominal_coverage(rng):
    """Frozen MinMaxScaler in a Pipeline keeps nominal conformal coverage."""
    N, d = 300, 2
    X = rng.uniform(low=-500.0, high=500.0, size=(N, d))
    y = X[:, 0] * 2.0 + rng.normal(scale=1.0, size=N)

    n_train = 100
    epsilon = 0.1
    margin = 0.15

    pipe = MinMaxScaler() | ConformalRidgeRegressor(a=1.0, epsilon=epsilon)
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])

    metric = ErrorRate()
    progressive_val(pipe, X[n_train:], y[n_train:], metric=metric, epsilon=epsilon)
    assert metric.get() <= epsilon + margin


def test_pipeline_predict_equals_manual_scale_then_fit(rng):
    """Pipeline with StandardScaler produces identical results to manual scaling."""
    N = 50
    X = rng.normal(loc=10.0, scale=5.0, size=(N, 3))
    y = X[:, 0] + rng.normal(scale=0.2, size=N)

    pipe = StandardScaler() | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    pipe.learn_initial_training_set(X, y)

    # Manual reference
    sc = StandardScaler()
    sc.fit(X)
    ref = ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    ref.learn_initial_training_set(sc.transform(X), y)

    x_test = rng.normal(loc=10.0, scale=5.0, size=3)
    iv_pipe = pipe.predict(x_test, epsilon=0.1)
    iv_ref = ref.predict(sc.transform_one(x_test), epsilon=0.1)

    np.testing.assert_allclose(iv_pipe.lower, iv_ref.lower)
    np.testing.assert_allclose(iv_pipe.upper, iv_ref.upper)
