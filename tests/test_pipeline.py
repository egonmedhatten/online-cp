"""Tests for online_cp.pipeline (Phase P1 + P2a composition utilities)."""

import numpy as np
import pytest

from online_cp import (
    ConformalNearestNeighboursClassifier,
    ConformalRidgeRegressor,
    Discard,
    FuncTransformer,
    ObservedFuzziness,
    IntervalWidth,
    Pipeline,
    Select,
    Transformer,
    TransformerUnion,
    iter_progressive_val,
    progressive_val,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def linear_dataset(rng):
    N, d = 200, 4
    X = rng.normal(size=(N, d))
    beta = np.array([2.0, 1.0, -0.5, 0.0])
    y = X @ beta + rng.normal(scale=0.5, size=N)
    return X, y


@pytest.fixture
def classification_dataset(rng):
    N = 200
    y = np.array([0, 1] * (N // 2))
    X = rng.normal(size=(N, 4))
    X[y == 0] -= 1.5
    X[y == 1] += 1.5
    return X, y


# ---------------------------------------------------------------------------
# 1. FuncTransformer correctness
# ---------------------------------------------------------------------------


def test_func_transformer_transform_one_log():
    ft = FuncTransformer(np.log1p)
    x = np.array([0.0, 1.0, 2.0])
    result = ft.transform_one(x)
    np.testing.assert_allclose(result, np.log1p(x))


def test_func_transformer_transform_batch():
    ft = FuncTransformer(lambda X: 2 * X + 1)
    X = np.arange(12, dtype=float).reshape(4, 3)
    result = ft.transform(X)
    np.testing.assert_allclose(result, 2 * X + 1)


def test_func_transformer_transform_one_affine():
    ft = FuncTransformer(lambda x: 2 * x + 1)
    x = np.array([1.0, 2.0])
    np.testing.assert_allclose(ft.transform_one(x), np.array([3.0, 5.0]))


def test_func_transformer_mode_is_fixed():
    ft = FuncTransformer(np.sqrt)
    assert ft.mode == "fixed"


def test_func_transformer_is_transformer_subclass():
    ft = FuncTransformer(np.sqrt)
    assert isinstance(ft, Transformer)


# ---------------------------------------------------------------------------
# 2. Pipeline.learn_initial_training_set feeds transformed X
# ---------------------------------------------------------------------------


def test_pipeline_learn_initial_feeds_transformed_X(linear_dataset):
    X, y = linear_dataset
    fn = lambda arr: arr * 2.0

    pipe = Pipeline(FuncTransformer(fn), ConformalRidgeRegressor(a=1.0, epsilon=0.1))
    pipe.learn_initial_training_set(X, y)

    # Build reference estimator trained on already-transformed data
    ref = ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    ref.learn_initial_training_set(fn(X), y)

    x = X[0]
    interval_pipe = pipe.predict(x, epsilon=0.1)
    interval_ref = ref.predict(fn(x), epsilon=0.1)

    assert interval_pipe.lower == pytest.approx(interval_ref.lower)
    assert interval_pipe.upper == pytest.approx(interval_ref.upper)


# ---------------------------------------------------------------------------
# 3. Pipeline.predict == estimator-on-transformed
# ---------------------------------------------------------------------------


def test_pipeline_predict_equals_estimator_on_transformed(linear_dataset):
    X, y = linear_dataset
    fn = np.log1p

    # Pipeline
    pipe = Pipeline(FuncTransformer(fn), ConformalRidgeRegressor(a=1.0, epsilon=0.1))
    pipe.learn_initial_training_set(np.abs(X) + 1, y)

    # Manual
    ref = ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    Xabs = np.abs(X) + 1
    ref.learn_initial_training_set(fn(Xabs), y)

    x = np.abs(X[5]) + 1
    ip = pipe.predict(x, epsilon=0.1)
    ir = ref.predict(fn(x), epsilon=0.1)
    assert ip.lower == pytest.approx(ir.lower)
    assert ip.upper == pytest.approx(ir.upper)


def test_pipeline_predict_classifier_equals_estimator_on_transformed(
    classification_dataset,
):
    X, y = classification_dataset
    fn = lambda arr: arr - arr.mean(axis=0)

    pipe = Pipeline(
        FuncTransformer(fn),
        ConformalNearestNeighboursClassifier(k=3, label_space=[0, 1], epsilon=0.1),
    )
    pipe.learn_initial_training_set(X, y)

    ref = ConformalNearestNeighboursClassifier(k=3, label_space=[0, 1], epsilon=0.1)
    ref.learn_initial_training_set(fn(X), y)

    x = X[10]
    result_pipe = pipe.predict(x, epsilon=0.1)
    result_ref = ref.predict(fn(x), epsilon=0.1)
    # Both should return a ConformalPredictionSet; compare contained labels
    assert set(result_pipe.elements) == set(result_ref.elements)


# ---------------------------------------------------------------------------
# 4. | operator builds equivalent Pipeline
# ---------------------------------------------------------------------------


def test_pipe_operator_builds_pipeline(linear_dataset):
    X, y = linear_dataset
    fn = lambda arr: arr * 3.0

    pipe_via_or = FuncTransformer(fn) | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    pipe_direct = Pipeline(FuncTransformer(fn), ConformalRidgeRegressor(a=1.0, epsilon=0.1))

    assert isinstance(pipe_via_or, Pipeline)

    pipe_via_or.learn_initial_training_set(X, y)
    pipe_direct.learn_initial_training_set(X, y)

    x = X[0]
    i1 = pipe_via_or.predict(x, epsilon=0.1)
    i2 = pipe_direct.predict(x, epsilon=0.1)
    assert i1.lower == pytest.approx(i2.lower)
    assert i1.upper == pytest.approx(i2.upper)


def test_transformer_pipe_operator_returns_pipeline():
    ft = FuncTransformer(np.abs)
    est = ConformalRidgeRegressor(a=1.0)
    result = ft | est
    assert isinstance(result, Pipeline)
    assert result.estimator is est
    assert result.transformers[0] is ft


# ---------------------------------------------------------------------------
# 5. Integration: regressor through progressive_val
# ---------------------------------------------------------------------------


def test_progressive_val_regressor_pipeline_coverage(linear_dataset):
    X, y = linear_dataset
    epsilon = 0.1
    n_train = 50

    pipe = Pipeline(
        FuncTransformer(lambda arr: arr - arr.mean(axis=0) if arr.ndim == 2 else arr),
        ConformalRidgeRegressor(a=1.0, epsilon=epsilon),
    )
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])

    from online_cp import IntervalWidth

    metric = IntervalWidth()
    progressive_val(pipe, X[n_train:], y[n_train:], metric=metric)
    # Just check it runs and metric is positive
    assert metric.get() > 0


def test_progressive_val_regressor_pipeline_nominal_coverage(linear_dataset):
    X, y = linear_dataset
    epsilon = 0.1
    n_train = 50
    margin = 0.15

    from online_cp import ErrorRate

    pipe = Pipeline(
        FuncTransformer(np.abs),
        ConformalRidgeRegressor(a=1.0, epsilon=epsilon),
    )
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])
    metric = ErrorRate()
    progressive_val(pipe, X[n_train:], y[n_train:], metric=metric, epsilon=epsilon)
    error_rate = metric.get()
    assert error_rate <= epsilon + margin


# ---------------------------------------------------------------------------
# 6. Integration: classifier through progressive_val with ObservedFuzziness
# ---------------------------------------------------------------------------


def test_progressive_val_classifier_pipeline_with_p_values(classification_dataset):
    X, y = classification_dataset
    epsilon = 0.1
    n_train = 50

    pipe = Pipeline(
        FuncTransformer(lambda arr: arr * 2.0),
        ConformalNearestNeighboursClassifier(
            k=3, label_space=[0, 1], epsilon=epsilon
        ),
    )
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])

    metric = ObservedFuzziness()
    progressive_val(pipe, X[n_train:], y[n_train:], metric=metric, epsilon=epsilon)
    # Fuzziness should be a finite non-negative number
    assert np.isfinite(metric.get())
    assert metric.get() >= 0.0


# ---------------------------------------------------------------------------
# 7. iter_progressive_val yields dicts with a Pipeline
# ---------------------------------------------------------------------------


def test_iter_progressive_val_yields_dicts(linear_dataset):
    X, y = linear_dataset
    n_train = 50
    epsilon = 0.1

    from online_cp import ErrorRate

    pipe = Pipeline(
        FuncTransformer(np.abs),
        ConformalRidgeRegressor(a=1.0, epsilon=epsilon),
    )
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])

    metric = ErrorRate()
    results = list(
        iter_progressive_val(pipe, X[n_train : n_train + 10], y[n_train : n_train + 10], metric=metric, epsilon=epsilon)
    )
    assert len(results) == 10
    assert all(isinstance(r, dict) for r in results)


# ---------------------------------------------------------------------------
# 8. Attribute forwarding: epsilon and label_space
# ---------------------------------------------------------------------------


def test_attribute_forwarding_epsilon():
    est = ConformalRidgeRegressor(a=1.0, epsilon=0.05)
    pipe = Pipeline(FuncTransformer(np.abs), est)
    assert pipe.epsilon == pytest.approx(0.05)


def test_attribute_forwarding_label_space():
    est = ConformalNearestNeighboursClassifier(k=1, label_space=[0, 1, 2])
    pipe = Pipeline(FuncTransformer(np.abs), est)
    np.testing.assert_array_equal(pipe.label_space, np.array([0, 1, 2]))


def test_getattr_raises_for_missing_attr():
    pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0))
    with pytest.raises(AttributeError):
        _ = pipe.nonexistent_attribute_xyz


# ---------------------------------------------------------------------------
# 9. Fixed-transform soundness: coverage preserved under rescaling
# ---------------------------------------------------------------------------


def test_soundness_fixed_rescale_preserves_coverage(rng):
    """A pipeline with a fixed affine rescale must maintain nominal coverage."""
    N, d = 300, 2
    X_raw = rng.normal(size=(N, d))
    # Poorly scaled: one feature is 1000x larger
    scale = np.array([1000.0, 1.0])
    X = X_raw * scale
    beta = np.array([0.001, 1.0])
    y = X @ beta + rng.normal(scale=0.5, size=N)

    n_train = 100
    epsilon = 0.1
    margin = 0.15

    from online_cp import ErrorRate

    # Normalize the badly-scaled features with a fixed map
    pipe = Pipeline(
        FuncTransformer(lambda arr: arr / scale),
        ConformalRidgeRegressor(a=1.0, epsilon=epsilon),
    )
    pipe.learn_initial_training_set(X[:n_train], y[:n_train])
    metric = ErrorRate()
    progressive_val(pipe, X[n_train:], y[n_train:], metric=metric, epsilon=epsilon)
    assert metric.get() <= epsilon + margin


# ---------------------------------------------------------------------------
# 10. learn_one updates the estimator
# ---------------------------------------------------------------------------


def test_learn_one_updates_estimator(linear_dataset):
    X, y = linear_dataset
    fn = lambda arr: arr * 2.0

    pipe = Pipeline(FuncTransformer(fn), ConformalRidgeRegressor(a=1.0, epsilon=0.1))
    pipe.learn_initial_training_set(X[:50], y[:50])

    x_test = X[51]
    interval_before = pipe.predict(x_test, epsilon=0.1)

    # Learn one more point
    pipe.learn_one(X[50], y[50])

    interval_after = pipe.predict(x_test, epsilon=0.1)
    # After learning a new point the interval should change (model has more data)
    # We can't guarantee direction, just that the estimator was updated
    assert interval_before.lower != pytest.approx(interval_after.lower) or \
           interval_before.upper != pytest.approx(interval_after.upper)


def test_learn_one_precomputed_dropped_gracefully(linear_dataset):
    """precomputed kwarg is accepted and silently discarded."""
    X, y = linear_dataset
    pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0, epsilon=0.1))
    pipe.learn_initial_training_set(X[:50], y[:50])
    # Should not raise even when precomputed is passed
    pipe.learn_one(X[50], y[50], precomputed=object())


# ---------------------------------------------------------------------------
# 11. Pipeline construction validation
# ---------------------------------------------------------------------------


def test_pipeline_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="at least two steps"):
        Pipeline(FuncTransformer(np.abs))


def test_pipeline_repr():
    ft = FuncTransformer(np.abs)
    est = ConformalRidgeRegressor(a=1.0)
    pipe = Pipeline(ft, est)
    r = repr(pipe)
    assert "Pipeline(" in r


# ---------------------------------------------------------------------------
# Phase P1 remainder — validity guard
# ---------------------------------------------------------------------------


class _IncrementalTransformer(Transformer):
    """Stub transformer with a non-sound mode for guard tests."""
    mode = "incremental"

    def transform(self, X):
        return X

    def transform_one(self, x):
        return x


class TestValidityGuard:
    def test_incremental_mode_raises(self):
        with pytest.raises(ValueError, match="mode="):
            Pipeline(_IncrementalTransformer(), ConformalRidgeRegressor(a=1.0))

    def test_unsafe_incremental_bypasses_guard(self):
        pipe = Pipeline(
            _IncrementalTransformer(),
            ConformalRidgeRegressor(a=1.0),
            unsafe_incremental=True,
        )
        assert pipe._unsafe_incremental is True

    def test_fixed_mode_is_always_allowed(self):
        pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0))
        assert not pipe._unsafe_incremental

    def test_frozen_mode_is_always_allowed(self):
        from online_cp import StandardScaler
        pipe = Pipeline(StandardScaler(), ConformalRidgeRegressor(a=1.0))
        assert not pipe._unsafe_incremental

    def test_mixed_sound_modes_allowed(self):
        from online_cp import StandardScaler
        pipe = Pipeline(
            FuncTransformer(np.abs),
            StandardScaler(),
            ConformalRidgeRegressor(a=1.0),
        )
        assert len(pipe.transformers) == 2

    def test_unsound_mode_in_chain_raises(self):
        from online_cp import StandardScaler
        with pytest.raises(ValueError, match="mode="):
            Pipeline(
                StandardScaler(),
                _IncrementalTransformer(),
                ConformalRidgeRegressor(a=1.0),
            )

    def test_summary_records_unsafe_flag(self):
        pipe = Pipeline(
            _IncrementalTransformer(),
            ConformalRidgeRegressor(a=1.0),
            unsafe_incremental=True,
        )
        assert pipe.summary()["unsafe_incremental"] is True


# ---------------------------------------------------------------------------
# Phase P3 — bare-callable auto-wrap
# ---------------------------------------------------------------------------


class TestAutoWrap:
    def test_bare_callable_wrapped_as_func_transformer(self):
        pipe = Pipeline(np.abs, ConformalRidgeRegressor(a=1.0))
        assert isinstance(pipe.transformers[0], FuncTransformer)
        assert pipe.transformers[0].fn is np.abs

    def test_lambda_wrapped(self):
        fn = lambda x: x * 2.0
        pipe = Pipeline(fn, ConformalRidgeRegressor(a=1.0))
        assert isinstance(pipe.transformers[0], FuncTransformer)

    def test_non_callable_non_transformer_raises(self):
        with pytest.raises(TypeError, match="neither a Transformer nor a callable"):
            Pipeline(42, ConformalRidgeRegressor(a=1.0))

    def test_string_raises(self):
        with pytest.raises(TypeError, match="neither a Transformer nor a callable"):
            Pipeline("log", ConformalRidgeRegressor(a=1.0))

    def test_wrapped_pipeline_runs_correctly(self, linear_dataset):
        X, y = linear_dataset
        pipe = Pipeline(np.abs, ConformalRidgeRegressor(a=1.0, epsilon=0.1))
        pipe.learn_initial_training_set(X, y)
        iv = pipe.predict(X[0], epsilon=0.1)
        assert iv.lower <= iv.upper

    def test_transformer_instance_not_double_wrapped(self):
        ft = FuncTransformer(np.abs)
        pipe = Pipeline(ft, ConformalRidgeRegressor(a=1.0))
        assert pipe.transformers[0] is ft


# ---------------------------------------------------------------------------
# Phase P3 — Pipeline.summary()
# ---------------------------------------------------------------------------


class TestPipelineSummary:
    def test_structure(self):
        pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0))
        s = pipe.summary()
        assert set(s.keys()) == {"n_steps", "transformers", "estimator", "unsafe_incremental", "bag_mode"}

    def test_n_steps(self):
        from online_cp import StandardScaler
        pipe = Pipeline(
            FuncTransformer(np.abs), StandardScaler(), ConformalRidgeRegressor(a=1.0)
        )
        assert pipe.summary()["n_steps"] == 3

    def test_estimator_type(self):
        pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0))
        assert pipe.summary()["estimator"]["type"] == "ConformalRidgeRegressor"

    def test_transformer_mode_reported(self):
        from online_cp import StandardScaler
        pipe = Pipeline(FuncTransformer(np.abs), StandardScaler(), ConformalRidgeRegressor(a=1.0))
        s = pipe.summary()
        assert s["transformers"][0]["mode"] == "fixed"
        assert s["transformers"][1]["mode"] == "frozen"

    def test_fitted_false_before_fit(self):
        from online_cp import StandardScaler
        pipe = Pipeline(StandardScaler(), ConformalRidgeRegressor(a=1.0))
        assert pipe.summary()["transformers"][0]["fitted"] is False

    def test_fitted_true_after_fit(self, linear_dataset):
        from online_cp import StandardScaler
        X, y = linear_dataset
        pipe = Pipeline(StandardScaler(), ConformalRidgeRegressor(a=1.0))
        pipe.learn_initial_training_set(X, y)
        assert pipe.summary()["transformers"][0]["fitted"] is True

    def test_fixed_transformer_fitted_after_fit(self, linear_dataset):
        X, y = linear_dataset
        pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0))
        pipe.learn_initial_training_set(X, y)
        # FuncTransformer's fit() sets _fitted via base Transformer.fit()
        assert pipe.summary()["transformers"][0]["fitted"] is True


# ---------------------------------------------------------------------------
# P2b — Bag-mode pipeline (mode="bag")
# ---------------------------------------------------------------------------


from online_cp import MinMaxScaler, StandardScaler  # noqa: E402


class TestBagMode:
    """Functional (L1) tests for bag-mode transformers in Pipeline."""

    @pytest.fixture
    def poorly_scaled_dataset(self, rng):
        N, d = 60, 3
        scale = np.array([1000.0, 0.001, 1.0])
        X = rng.normal(size=(N, d)) * scale
        beta = np.array([0.001, 1000.0, 1.0])
        y = X @ beta + rng.normal(scale=0.5, size=N)
        return X, y

    # --- Construction / validation ---

    def test_bag_pipeline_constructs(self):
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        assert pipe._bag_mode is True

    def test_non_bag_pipeline_bag_mode_false(self):
        pipe = StandardScaler() | ConformalRidgeRegressor(a=1.0)
        assert pipe._bag_mode is False

    def test_frozen_bag_mixing_raises(self):
        from online_cp import MinMaxScaler as MMS
        with pytest.raises(ValueError, match="not supported"):
            Pipeline(
                StandardScaler(),           # frozen
                MMS(mode="bag"),            # bag
                ConformalRidgeRegressor(a=1.0),
            )

    def test_summary_reports_bag_mode(self):
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        s = pipe.summary()
        assert s["bag_mode"] is True
        assert s["transformers"][0]["mode"] == "bag"

    def test_summary_bag_mode_false_for_frozen(self):
        pipe = StandardScaler() | ConformalRidgeRegressor(a=1.0)
        assert pipe.summary()["bag_mode"] is False

    # --- Finite intervals after data ---

    def test_finite_intervals_after_training(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        n = 30
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        pipe.learn_initial_training_set(X[:n], y[:n])
        iv = pipe.predict(X[n], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)
        assert iv.lower < iv.upper

    # --- Equivalence to manual augmented-bag scaling ---

    def test_equivalence_to_manual_augmented_bag(self, poorly_scaled_dataset):
        """Bag pipeline interval == manually scaling [X_train, x] then fitting ridge."""
        X, y = poorly_scaled_dataset
        n = 40
        X_tr, y_tr, x_test = X[:n], y[:n], X[n]
        epsilon = 0.1

        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        pipe.learn_initial_training_set(X_tr, y_tr)
        iv_pipe = pipe.predict(x_test, epsilon=epsilon)

        # Manual reference: fit scaler on augmented bag (label-free)
        X_aug = np.vstack([X_tr, x_test.reshape(1, -1)])
        sc = StandardScaler()
        sc.fit(X_aug)
        X_aug_scaled = sc.transform(X_aug)
        ref = ConformalRidgeRegressor(a=1.0)
        ref.learn_initial_training_set(X_aug_scaled[:-1], y_tr)
        iv_ref = ref.predict(X_aug_scaled[-1], epsilon=epsilon)

        np.testing.assert_allclose(iv_pipe.lower, iv_ref.lower, rtol=1e-10)
        np.testing.assert_allclose(iv_pipe.upper, iv_ref.upper, rtol=1e-10)

    # --- Permutation invariance (key exchangeability proof) ---

    def test_compute_p_value_permutation_invariant(self, poorly_scaled_dataset, rng):
        """p-value must not change under permutation of the training bag."""
        X, y = poorly_scaled_dataset
        n = 30
        X_tr, y_tr = X[:n], y[:n]
        x_test, y_test = X[n], float(y[n])

        pipe_ref = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0, warnings=False)
        pipe_ref.learn_initial_training_set(X_tr, y_tr)
        p_ref = pipe_ref.compute_p_value(x_test, y_test, tau=0.5)

        perm = rng.permutation(n)
        pipe_perm = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0, warnings=False)
        pipe_perm.learn_initial_training_set(X_tr[perm], y_tr[perm])
        p_perm = pipe_perm.compute_p_value(x_test, y_test, tau=0.5)

        assert np.isclose(p_ref, p_perm), (
            f"p-value changed under permutation: {p_ref} vs {p_perm}"
        )

    # --- Optional learn_initial_training_set ---

    def test_learn_one_only_then_predict(self, poorly_scaled_dataset):
        """learn_initial_training_set is optional; learn_one accumulates the bag."""
        X, y = poorly_scaled_dataset
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        n = 30
        for i in range(n):
            pipe.learn_one(X[i], float(y[i]))
        iv = pipe.predict(X[n], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)

    def test_learn_initial_then_learn_one(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        pipe.learn_initial_training_set(X[:20], y[:20])
        for i in range(20, 30):
            pipe.learn_one(X[i], float(y[i]))
        iv = pipe.predict(X[30], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)

    # --- Empty-bag validity ---

    def test_empty_bag_returns_trivial_interval(self):
        """No training data → default CP full-label-space prediction; must not raise."""
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        iv = pipe.predict(np.array([1.0, 2.0, 3.0]), epsilon=0.1)
        assert iv.lower == -np.inf
        assert iv.upper == np.inf

    # --- Read-only predict ---

    def test_predict_twice_identical(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        pipe.learn_initial_training_set(X[:30], y[:30])
        x_test = X[30]
        iv1 = pipe.predict(x_test, epsilon=0.1)
        iv2 = pipe.predict(x_test, epsilon=0.1)
        np.testing.assert_allclose(iv1.lower, iv2.lower)
        np.testing.assert_allclose(iv1.upper, iv2.upper)

    def test_predict_does_not_mutate_raw_bag(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        pipe.learn_initial_training_set(X[:30], y[:30])
        n_before = len(pipe._X_raw)
        pipe.predict(X[30], epsilon=0.1)
        assert len(pipe._X_raw) == n_before

    def test_bag_pipeline_in_progressive_val(self, poorly_scaled_dataset):
        from online_cp import ErrorRate
        X, y = poorly_scaled_dataset
        pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0, warnings=False)
        pipe.learn_initial_training_set(X[:30], y[:30])
        metric = ErrorRate()
        progressive_val(pipe, X[30:50], y[30:50], epsilon=0.1, metric=metric)
        assert 0.0 <= metric.get() <= 1.0

    # --- Chains with fixed transformers ---

    def test_select_then_bag_scaler(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        pipe = Pipeline(
            Select([0, 2]),
            StandardScaler(mode="bag"),
            ConformalRidgeRegressor(a=1.0),
        )
        pipe.learn_initial_training_set(X[:30], y[:30])
        # Pass the full original x; Pipeline applies Select internally.
        iv = pipe.predict(X[30], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)

    def test_func_transformer_then_bag_scaler(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        pipe = Pipeline(
            FuncTransformer(np.abs),
            MinMaxScaler(mode="bag"),
            ConformalRidgeRegressor(a=1.0),
        )
        pipe.learn_initial_training_set(X[:30], y[:30])
        iv = pipe.predict(X[30], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)

    # --- MinMaxScaler bag mode ---

    def test_minmax_bag_equivalence(self, poorly_scaled_dataset):
        X, y = poorly_scaled_dataset
        n = 40
        X_tr, y_tr, x_test = X[:n], y[:n], X[n]
        epsilon = 0.1

        pipe = MinMaxScaler(mode="bag") | ConformalRidgeRegressor(a=1.0)
        pipe.learn_initial_training_set(X_tr, y_tr)
        iv_pipe = pipe.predict(x_test, epsilon=epsilon)

        X_aug = np.vstack([X_tr, x_test.reshape(1, -1)])
        sc = MinMaxScaler()
        sc.fit(X_aug)
        X_aug_scaled = sc.transform(X_aug)
        ref = ConformalRidgeRegressor(a=1.0)
        ref.learn_initial_training_set(X_aug_scaled[:-1], y_tr)
        iv_ref = ref.predict(X_aug_scaled[-1], epsilon=epsilon)

        np.testing.assert_allclose(iv_pipe.lower, iv_ref.lower, rtol=1e-10)
        np.testing.assert_allclose(iv_pipe.upper, iv_ref.upper, rtol=1e-10)

    # --- Smoke test: estimator-agnostic bag path ---

    def test_bag_with_knn_regressor(self, poorly_scaled_dataset):
        from online_cp import ConformalNearestNeighboursRegressor
        X, y = poorly_scaled_dataset
        pipe = StandardScaler(mode="bag") | ConformalNearestNeighboursRegressor()
        for i in range(30):
            pipe.learn_one(X[i], float(y[i]))
        iv = pipe.predict(X[30], epsilon=0.1)
        assert iv.lower <= iv.upper
