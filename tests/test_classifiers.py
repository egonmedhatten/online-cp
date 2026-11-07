import warnings

import numpy as np
import pytest

from online_cp.classifiers import (
    ConformalClassifierWrapper,
    ConformalNearestNeighboursClassifier,
    MultiLevelPredictionSet,
)


class TestConformalNearestNeighboursClassifier:
    def test_p_values_in_unit_interval(self, classification_dataset):
        X, y = classification_dataset
        label_space = np.unique(y)
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=0)
        cp.learn_initial_training_set(X[:10], y[:10])

        for obj, lab in zip(X[10:30], y[10:30]):
            _, p_values = cp.predict(obj, return_p_values=True)
            cp.learn_one(obj, lab)
            for p in p_values.values():
                assert 0 <= p <= 1

    def test_validity(self, classification_dataset):
        """Error rate should be approximately <= epsilon on iid data."""
        X, y = classification_dataset
        label_space = np.unique(y)
        epsilon = 0.2
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=1, epsilon=epsilon)

        n_init = 10
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        errors = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            errors += int(lab not in Gamma)
            cp.learn_one(obj, lab)

        error_rate = errors / n_test
        # Allow generous margin for small sample
        assert error_rate <= epsilon + 0.10, f"Error rate {error_rate:.3f} exceeds epsilon={epsilon} + margin"

    def test_learn_one_grows_state(self):
        cp = ConformalNearestNeighboursClassifier(k=1, rnd_state=0)
        cp.learn_one(np.array([1.0, 2.0]), 1)
        assert cp.X.shape[0] == 1
        assert cp.y.shape[0] == 1

        cp.learn_one(np.array([3.0, 4.0]), -1)
        assert cp.X.shape[0] == 2
        assert cp.y.shape[0] == 2

    def test_prediction_set_nonempty_after_training(self, classification_dataset):
        """After sufficient training, prediction sets should not be empty at low epsilon."""
        X, y = classification_dataset
        label_space = np.unique(y)
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=2, epsilon=0.01)
        cp.learn_initial_training_set(X[:50], y[:50])

        for obj in X[50:60]:
            Gamma = cp.predict(obj)
            assert len(Gamma) >= 1

    def test_first_prediction_includes_all_labels(self):
        """With no training data, all labels should be predicted."""
        cp = ConformalNearestNeighboursClassifier(k=1, label_space=np.array([0, 1, 2]), rnd_state=0)
        Gamma = cp.predict(np.array([0.0, 0.0]))
        assert len(Gamma) == 3

    def test_multi_level_epsilon(self, classification_dataset):
        """predict with a list of epsilons should return MultiLevelPredictionSet."""
        X, y = classification_dataset
        label_space = np.unique(y)
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=0)
        cp.learn_initial_training_set(X[:50], y[:50])

        epsilons = [0.01, 0.05, 0.1, 0.2]
        result = cp.predict(X[50], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert result.levels == sorted(epsilons)
        assert len(result) == 4

        # Smaller epsilon => larger (or equal) prediction set
        for i in range(len(epsilons) - 1):
            assert len(result[epsilons[i]]) >= len(result[epsilons[i + 1]])

        # __contains__ returns bool (True if covered at all levels)
        assert isinstance(y[50] in result, bool)

        # .coverage() returns dict
        coverage = result.coverage(y[50])
        assert isinstance(coverage, dict)
        assert set(coverage.keys()) == set(epsilons)

    def test_multi_level_with_p_values(self, classification_dataset):
        """Multi-level predict with return_p_values=True."""
        X, y = classification_dataset
        label_space = np.unique(y)
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=0)
        cp.learn_initial_training_set(X[:50], y[:50])

        result, p_values = cp.predict(X[50], epsilon=[0.05, 0.1], return_p_values=True)
        assert isinstance(result, MultiLevelPredictionSet)
        assert isinstance(p_values, dict)

    def test_compute_p_value_matches_predict_per_label(self, classification_dataset):
        """compute_p_value should match predict() p-values for the same hypothesis."""
        X, y = classification_dataset
        label_space = np.unique(y)
        X_train, y_train = X[:40], y[:40]
        x_test = X[40]

        for label in label_space:
            cp_predict = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=123)
            cp_predict.learn_initial_training_set(X_train, y_train)
            _, p_values = cp_predict.predict(x_test, return_p_values=True)

            cp_single = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=123)
            cp_single.learn_initial_training_set(X_train, y_train)
            p_single = cp_single.compute_p_value(x_test, label)

            assert np.isclose(p_single, p_values[label], atol=1e-12)
            assert 0 <= p_single <= 1

    def test_predict_return_update_produces_consistent_state(self, classification_dataset):
        """Using predict(..., return_update=True) should preserve learn_one state equivalence."""
        X, y = classification_dataset
        label_space = np.unique(y)
        X_train, y_train = X[:30], y[:30]
        x_new, y_new = X[30], y[30]

        cp_with_update = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=7)
        cp_with_update.learn_initial_training_set(X_train, y_train)

        cp_reference = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=7)
        cp_reference.learn_initial_training_set(X_train, y_train)

        _, D = cp_with_update.predict(x_new, return_update=True)
        assert D.shape == (X_train.shape[0] + 1, X_train.shape[0] + 1)

        cp_with_update.learn_one(x_new, y_new, precomputed=D)
        cp_reference.learn_one(x_new, y_new)

        assert np.allclose(cp_with_update.X, cp_reference.X)
        assert np.allclose(cp_with_update.y, cp_reference.y)
        assert np.allclose(cp_with_update.D, cp_reference.D)

    def test_compute_p_value_return_update_produces_consistent_state(self, classification_dataset):
        """Using compute_p_value(..., return_update=True) should preserve learn_one state equivalence."""
        X, y = classification_dataset
        label_space = np.unique(y)
        X_train, y_train = X[:30], y[:30]
        x_new, y_new = X[31], y[31]

        cp_with_update = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=9)
        cp_with_update.learn_initial_training_set(X_train, y_train)

        cp_reference = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=9)
        cp_reference.learn_initial_training_set(X_train, y_train)

        p_value, D = cp_with_update.compute_p_value(x_new, y_new, return_update=True)
        assert 0 <= p_value <= 1
        assert D.shape == (X_train.shape[0] + 1, X_train.shape[0] + 1)

        cp_with_update.learn_one(x_new, y_new, precomputed=D)
        cp_reference.learn_one(x_new, y_new)

        assert np.allclose(cp_with_update.X, cp_reference.X)
        assert np.allclose(cp_with_update.y, cp_reference.y)
        assert np.allclose(cp_with_update.D, cp_reference.D)

    def test_multiclass_arbitrary_labels_with_k_edge_cases(self):
        """Nearest-neighbour p-values should work for multiclass arbitrary labels and large k."""
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [10.0, 10.0],
                [10.0, 11.0],
                [20.0, 20.0],
            ]
        )
        y = np.array([10, 10, 20, 30, 30])
        label_space = np.array([10, 20, 30])

        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=0)
        cp.learn_initial_training_set(X[:4], y[:4])

        Gamma, p_values = cp.predict(X[4], return_p_values=True)

        assert set(p_values.keys()) == {10, 20, 30}
        for p in p_values.values():
            assert np.isfinite(p)
            assert 0 <= p <= 1
        assert set(Gamma.elements).issubset({10, 20, 30})

        multi = cp.predict(X[4], epsilon=[0.05, 0.2, 0.5])
        assert isinstance(multi, MultiLevelPredictionSet)
        assert len(multi[0.05]) >= len(multi[0.2]) >= len(multi[0.5])

    def test_non_euclidean_distance_metric(self, classification_dataset):
        """Classifier should work with non-Euclidean scipy distance metrics."""
        X, y = classification_dataset
        label_space = np.unique(y)

        cp = ConformalNearestNeighboursClassifier(
            k=3,
            label_space=label_space,
            distance="cityblock",
            rnd_state=0,
        )
        cp.learn_initial_training_set(X[:25], y[:25])

        Gamma, p_values = cp.predict(X[25], return_p_values=True)

        assert set(p_values.keys()) == set(label_space)
        assert cp.D.shape == (25, 25)
        for p in p_values.values():
            assert 0 <= p <= 1
        assert set(Gamma.elements).issubset(set(label_space))

        p_true = cp.compute_p_value(X[25], y[25])
        assert 0 <= p_true <= 1

    def test_custom_distance_function_is_used(self):
        """Custom distance functions should determine the stored distance matrix and predictions."""
        calls = []

        def chebyshev_distance(X, y=None):
            X = np.atleast_2d(X)
            calls.append((X.shape, None if y is None else np.atleast_2d(y).shape))
            if y is None:
                diff = np.abs(X[:, None, :] - X[None, :, :])
            else:
                Y = np.atleast_2d(y)
                diff = np.abs(X[:, None, :] - Y[None, :, :])
            return diff.max(axis=2)

        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [4.0, 1.0],
                [5.0, 5.0],
            ]
        )
        y = np.array([0, 0, 1, 1])
        expected_D = np.array(
            [
                [0.0, 2.0, 4.0, 5.0],
                [2.0, 0.0, 3.0, 4.0],
                [4.0, 3.0, 0.0, 4.0],
                [5.0, 4.0, 4.0, 0.0],
            ]
        )

        cp = ConformalNearestNeighboursClassifier(
            k=1,
            label_space=np.array([0, 1]),
            distance_func=chebyshev_distance,
            rnd_state=0,
        )
        cp.learn_initial_training_set(X, y)

        assert cp.distance == "custom"
        assert np.allclose(cp.D, expected_D)

        Gamma, p_values = cp.predict(np.array([3.0, 3.0]), return_p_values=True)

        assert calls
        assert set(p_values.keys()) == {0, 1}
        for p in p_values.values():
            assert 0 <= p <= 1
        assert set(Gamma.elements).issubset({0, 1})

        p_label = cp.compute_p_value(np.array([3.0, 3.0]), 1)
        assert 0 <= p_label <= 1


class _ToyProbLearner:
    """Minimal sklearn-like learner with classes_ and predict_proba."""

    def __init__(self):
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        m = len(self.classes_)
        probs = np.tile(np.linspace(1.0, float(m), m), (n, 1))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


class LogisticRegression(_ToyProbLearner):
    """Name-based stand-in to exercise recommended-estimator warning logic."""


class _BadNormalizedProbLearner(_ToyProbLearner):
    def predict_proba(self, X):
        probs = super().predict_proba(X)
        return probs * 2.0


class _BadShapeProbLearner(_ToyProbLearner):
    def predict_proba(self, X):
        probs = super().predict_proba(X)
        return probs[:, 0]


class TestConformalClassifierWrapper:
    def test_soft_whitelist_warns_for_non_recommended_estimator(self):
        with pytest.warns(UserWarning) as record:
            ConformalClassifierWrapper(_ToyProbLearner(), label_space=np.array([10, 20]), rnd_state=0)
        assert any("not in the recommended set" in str(w.message) for w in record)

    def test_recommended_name_does_not_emit_support_tier_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConformalClassifierWrapper(LogisticRegression(), label_space=np.array([10, 20]), rnd_state=0)

        messages = [str(w.message) for w in caught]
        assert any("experimental, slow" in msg for msg in messages)
        assert not any("not in the recommended set" in msg for msg in messages)
        assert not any("supported with caution" in msg for msg in messages)

    def test_empty_training_predicts_all_labels(self):
        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(_ToyProbLearner(), label_space=np.array([10, 20]), rnd_state=0)

        Gamma, p_values = cp.predict(np.array([0.0, 1.0]), return_p_values=True)
        # With no training data, p-value for each label equals tau (uniform draw)
        assert set(p_values.keys()) == {10, 20}
        tau = list(p_values.values())[0]
        assert 0 < tau < 1
        assert all(v == tau for v in p_values.values())

    def test_learn_initial_training_set_and_arbitrary_labels(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([10, 30, 10])

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(_ToyProbLearner(), label_space=np.array([10, 20, 30]), rnd_state=0)

        cp.learn_initial_training_set(X, y)
        assert cp.X.shape == X.shape
        assert cp.y.shape == y.shape

        Gamma, p_values = cp.predict(np.array([3.0, 3.0]), return_p_values=True)
        assert set(p_values.keys()) == {10, 20, 30}
        assert set(Gamma.elements).issubset({10, 20, 30})
        for p in p_values.values():
            assert 0 <= p <= 1

    def test_score_alignment_with_missing_class(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([10, 10, 30, 30])

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(_ToyProbLearner(), label_space=np.array([10, 20, 30]), rnd_state=0)

        cp.learn_initial_training_set(X, y)
        _, p_values = cp.predict(np.array([4.0]), return_p_values=True)

        assert set(p_values.keys()) == {10, 20, 30}
        for p in p_values.values():
            assert 0 <= p <= 1

    def test_invalid_score_rows_warn_but_continue(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 0])

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(_BadNormalizedProbLearner(), label_space=np.array([0, 1]), rnd_state=0)

        cp.learn_initial_training_set(X, y)
        with pytest.warns(UserWarning) as record:
            Gamma, p_values = cp.predict(np.array([3.0]), return_p_values=True)
        assert any("not normalized" in str(w.message) for w in record)

        assert set(p_values.keys()) == {0, 1}
        assert set(Gamma.elements).issubset({0, 1})
        for p in p_values.values():
            assert 0 <= p <= 1

    def test_invalid_score_shape_falls_back_to_full_set(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 0])

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(_BadShapeProbLearner(), label_space=np.array([0, 1]), rnd_state=0)

        cp.learn_initial_training_set(X, y)
        with pytest.warns(UserWarning, match="must return a 2D array"):
            Gamma, p_values = cp.predict(np.array([3.0]), return_p_values=True)

        assert set(Gamma.elements) == {0, 1}
        # Fallback p-values are tau (uniform draw), same for all labels
        tau = p_values[0]
        assert 0 < tau < 1
        assert p_values == {0: tau, 1: tau}

    def test_wrapper_infers_label_space_from_data(self):
        """With label_space=None, wrapper infers from learn_initial_training_set."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([10, 20, 10])

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(_ToyProbLearner(), rnd_state=0)

        assert cp.label_space is None
        cp.learn_initial_training_set(X, y)
        assert set(cp.label_space) == {10, 20}

        Gamma, p_values = cp.predict(np.array([3.0, 3.0]), return_p_values=True)
        assert set(p_values.keys()) == {10, 20}


class TestKNNAggregation:
    """Tests for the configurable aggregation parameter (mean/median)."""

    def test_default_is_mean(self):
        cp = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
        assert cp.aggregation == "mean"

    def test_median_aggregation(self):
        cp = ConformalNearestNeighboursClassifier(k=3, aggregation="median", rnd_state=0)
        assert cp.aggregation == "median"
        # Should still produce valid predictions
        label_space = np.array([0, 1])
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, aggregation="median", rnd_state=0)
        X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        cp.learn_initial_training_set(X, y)
        Gamma, p_values = cp.predict(np.array([1.5]), return_p_values=True)
        assert p_values[0] > p_values[1]

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation must be"):
            ConformalNearestNeighboursClassifier(k=3, aggregation="sum")


class TestWrapperWarmStart:
    """Tests for the warm_start and n_jobs features of ConformalClassifierWrapper."""

    def _make_dataset(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_warm_start_predictions_match_cold(self):
        """Warm-start and cold-start must produce identical p-values."""
        from sklearn.linear_model import LogisticRegression

        X, y = self._make_dataset()

        with pytest.warns(UserWarning):
            cp_warm = ConformalClassifierWrapper(
                LogisticRegression(max_iter=200),
                label_space=np.array([0, 1]),
                rnd_state=123,
                warm_start=True,
            )
        with pytest.warns(UserWarning):
            cp_cold = ConformalClassifierWrapper(
                LogisticRegression(max_iter=200),
                label_space=np.array([0, 1]),
                rnd_state=123,
                warm_start=False,
            )

        cp_warm.learn_initial_training_set(X[:20], y[:20])
        cp_cold.learn_initial_training_set(X[:20], y[:20])

        _, pv_warm = cp_warm.predict(X[20], return_p_values=True)
        _, pv_cold = cp_cold.predict(X[20], return_p_values=True)

        for label in [0, 1]:
            assert abs(pv_warm[label] - pv_cold[label]) < 1e-10

    def test_warm_start_auto_detection(self):
        """LogisticRegression gets warm_start=True, RandomForest does not."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        with pytest.warns(UserWarning):
            cp_lr = ConformalClassifierWrapper(
                LogisticRegression(),
                label_space=np.array([0, 1]),
                rnd_state=0,
            )
        assert cp_lr._warm_start is True

        with pytest.warns(UserWarning):
            cp_rf = ConformalClassifierWrapper(
                RandomForestClassifier(n_estimators=5),
                label_space=np.array([0, 1]),
                rnd_state=0,
            )
        assert cp_rf._warm_start is False

    def test_n_jobs_parallel_predictions_match_sequential(self):
        """Parallel execution must match sequential results."""
        from sklearn.linear_model import LogisticRegression

        X, y = self._make_dataset()

        with pytest.warns(UserWarning):
            cp_seq = ConformalClassifierWrapper(
                LogisticRegression(max_iter=200),
                label_space=np.array([0, 1]),
                rnd_state=99,
                n_jobs=None,
            )
        with pytest.warns(UserWarning):
            cp_par = ConformalClassifierWrapper(
                LogisticRegression(max_iter=200),
                label_space=np.array([0, 1]),
                rnd_state=99,
                n_jobs=2,
            )

        cp_seq.learn_initial_training_set(X[:20], y[:20])
        cp_par.learn_initial_training_set(X[:20], y[:20])

        _, pv_seq = cp_seq.predict(X[20], return_p_values=True)
        _, pv_par = cp_par.predict(X[20], return_p_values=True)

        for label in [0, 1]:
            assert abs(pv_seq[label] - pv_par[label]) < 1e-10

    def test_base_fit_invalidated_on_learn_one(self):
        """After learn_one, the base fit cache must be invalidated."""
        from sklearn.linear_model import LogisticRegression

        X, y = self._make_dataset()

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(
                LogisticRegression(max_iter=200),
                label_space=np.array([0, 1]),
                rnd_state=0,
            )

        cp.learn_initial_training_set(X[:20], y[:20])
        assert cp._base_fitted is False

        # First predict builds the cache
        cp.predict(X[20])
        assert cp._base_fitted is True

        # learn_one invalidates it
        cp.learn_one(X[20], y[20])
        assert cp._base_fitted is False
        assert cp._base_learner is None

    def test_warm_start_force_on_random_forest(self):
        """warm_start=True forced on tree-based should still produce valid predictions."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = self._make_dataset()

        with pytest.warns(UserWarning):
            cp = ConformalClassifierWrapper(
                RandomForestClassifier(n_estimators=5, random_state=0),
                label_space=np.array([0, 1]),
                rnd_state=0,
                warm_start=True,
            )

        cp.learn_initial_training_set(X[:20], y[:20])
        # sklearn's warm-start fit (deferred until predict) emits its own
        # UserWarning; this test is about prediction validity, so absorb it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            Gamma, pv = cp.predict(X[20], return_p_values=True)

        assert set(pv.keys()) == {0, 1}
        for p in pv.values():
            assert 0 <= p <= 1


class TestSection75ClassifierSpeedup:
    """The §7.5 optimization for the k-NN *classifier*.

    The classifier's per-point score ``alpha = same_d / diff_d`` is
    *label-hypothesis-relative* (the test point joins a training point's
    same-label or different-label k-neighbour set depending on the hypothesised
    label), so the semi-neighbour set is per-hypothesis.  The cached predict
    recomputes ``alpha`` only for the semi-neighbours plus the test point,
    instead of recomputing the full ``n x n`` nonconformity matrix from scratch
    per hypothesised label.

    The core invariant is **bit-identity**: the cached path must produce
    exactly the same p-values as the dense-matrix reference path, including in
    the warm-up regime (``n < k + 1``), for tie-rich data (duplicates, rounded
    grids), and for every hypothesised label in a multiclass bag.
    """

    @staticmethod
    def _dense_predict_pvalues(cp, x):
        """Reference: recompute p-values via the full augmented distance matrix
        (the pre-§7.5 code path), bypassing the per-point score cache."""
        tau = cp.rnd_gen.uniform(0, 1)
        d = cp.distance_func(cp.X, x).ravel()
        D = cp.update_distance_matrix(cp.D, d)
        test_index = D.shape[0] - 1
        p_values = {}
        for label in cp.label_space:
            li = cp._extend_label_indices(cp._label_indices, label, test_index)
            same, diff = cp._find_nearest_distances(D, label_indices=li, update_cache=False)
            # A zero different-label distance (duplicate example) legitimately
            # yields +inf; suppress the expected divide warning.
            with np.errstate(divide="ignore", invalid="ignore"):
                Alpha = np.nan_to_num(same / diff, nan=np.inf)
            p_values[label] = cp._compute_p_value(Alpha, tau, "nonconformity")
        return p_values

    def _make_binary(self, n, seed=0, k=3, agg="mean"):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def _assert_bit_identical(self, cp, x, tag):
        """Compare the cached predict against the dense reference for one point."""
        cp.rnd_gen.bit_generator.state = state = cp.rnd_gen.bit_generator.state
        got = cp.predict(x, return_p_values=True)[1]
        cp.rnd_gen.bit_generator.state = state
        exp = self._dense_predict_pvalues(cp, x)
        for lb in got:
            assert np.isclose(got[lb], exp[lb], rtol=0, atol=0, equal_nan=True), (
                f"{tag} lb={lb}: {got[lb]} != {exp[lb]}"
            )

    def test_bit_identical_random_data(self):
        """Cached predict must match the dense reference on random data, for
        both aggregations and several k values. Test points are held out (not
        part of the training bag) so the augmented matrix has a fresh row."""
        for k in (1, 3):
            for agg in ("mean", "median"):
                X, y = self._make_binary(210, seed=1, k=k, agg=agg)
                cp = ConformalNearestNeighboursClassifier(k=k, rnd_state=0, aggregation=agg)
                cp.learn_initial_training_set(X[:50], y[:50])
                for i in range(50, 200):
                    cp.learn_one(X[i], y[i])
                for i in range(200, 210):
                    self._assert_bit_identical(cp, X[i], f"k={k} agg={agg} i={i}")

    def test_bit_identical_warmup_regime(self):
        """When ``n < k + 1`` every training point is a semi-neighbour (its k-set
        has room for the test point); the cached path must still match."""
        for n_init in (1, 2, 3):
            X, y = self._make_binary(n_init + 16, seed=2)
            cp = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
            cp.learn_initial_training_set(X[:n_init], y[:n_init])
            for i in range(n_init, n_init + 10):
                cp.learn_one(X[i], y[i])
            for i in range(n_init + 10, n_init + 16):
                self._assert_bit_identical(cp, X[i], f"n_init={n_init} i={i}")

    def test_bit_identical_single_label_point(self):
        """A label with a single training point is the degenerate case: when the
        hypothesised label equals that point's label, the test point is the
        point's *only* same-neighbour and its score must be recomputed."""
        cp = ConformalNearestNeighboursClassifier(
            k=1, label_space=[-1, 1], rnd_state=1337, epsilon=0.1
        )
        cp.learn_one(np.int64(3), 1)
        self._assert_bit_identical(cp, -2, "single-point")

    def test_bit_identical_tie_rich(self):
        """Ties in the k-th distance must resolve identically to the dense path
        (the reference selects by value, and ties at the k-th slot must not
        change the aggregated score)."""
        rng = np.random.default_rng(3)
        grid = np.round(rng.normal(size=(68, 2)), 1)  # 1-decimal rounding -> ties
        y = (grid[:, 0] > 0).astype(int)
        for k, agg in ((1, "mean"), (3, "median")):
            cp = ConformalNearestNeighboursClassifier(k=k, rnd_state=0, aggregation=agg)
            cp.learn_initial_training_set(grid[:30], y[:30])
            for i in range(30, 60):
                cp.learn_one(grid[i], y[i])
            for i in range(60, 68):
                self._assert_bit_identical(cp, grid[i], f"k={k} agg={agg} i={i}")

    def test_bit_identical_duplicates(self):
        """Duplicate rows give zero same-label distances; the score is a
        legitimate +inf which must be reproduced exactly."""
        rng = np.random.default_rng(4)
        base = rng.normal(size=(12, 2))
        X = np.repeat(base, 3, axis=0)
        y = (X[:, 0] > 0).astype(int)
        cp = ConformalNearestNeighboursClassifier(k=2, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])
        for i in range(20, len(X)):
            cp.learn_one(X[i], y[i])
        for i in range(3, 11):
            self._assert_bit_identical(cp, X[i], f"dup i={i}")

    def test_bit_identical_multiclass(self):
        """With three labels, each hypothesised label puts the test point into a
        different subset of training points' same/diff sets; all labels must
        match."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(86, 2))
        y = (X[:, 0] > 0).astype(int) * 2 + (X[:, 1] > 0).astype(int)  # labels 0, 1, 2
        cp = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:40], y[:40])
        for i in range(40, 80):
            cp.learn_one(X[i], y[i])
        for i in range(80, 86):
            self._assert_bit_identical(cp, X[i], f"multiclass i={i}")

    def test_cache_alpha_matches_dense(self):
        """The stored per-point ``_alpha`` must equal a from-scratch
        ``_find_nearest_distances`` on the test-free matrix."""
        X, y = self._make_binary(120, seed=6, k=3)
        cp = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:40], y[:40])
        for i in range(40, 120):
            cp.learn_one(X[i], y[i])
        same, diff = cp._find_nearest_distances(
            D=cp.D, label_indices=cp._label_indices, update_cache=False
        )
        expected = np.nan_to_num(same / diff, nan=np.inf)
        assert np.allclose(cp._alpha, expected, rtol=0, atol=0, equal_nan=True)

    def test_pruning_invalidates_cache(self):
        """After a pruning eviction the cache must be rebuilt from scratch
        (the bag's *contents* change, not just its size)."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 2))
        y = (X[:, 0] > 0).astype(int)
        cp = ConformalNearestNeighboursClassifier(k=3, rnd_state=0, pruning="reservoir", budget=15)
        cp.learn_initial_training_set(X[:20], y[:20])
        for i in range(20, 40):
            cp.predict(X[i], epsilon=0.1)
            cp.learn_one(X[i], y[i])
        # Cache is consistent with the current bag.
        assert cp._alpha is not None
        assert cp._alpha.shape[0] == cp.X.shape[0] == 15
        same, diff = cp._find_nearest_distances(
            D=cp.D, label_indices=cp._label_indices, update_cache=False
        )
        expected = np.nan_to_num(same / diff, nan=np.inf)
        assert np.allclose(cp._alpha, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_predict_batch_matches_sequential(self):
        """``predict_batch`` (which shares the cached path) must be
        bit-identical to the sequential ``predict`` loop."""
        X, y = self._make_binary(40, seed=8, k=3)
        cp = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])
        for i in range(20, 40):
            cp.learn_one(X[i], y[i])
        Xb = X[:8]
        state = cp.rnd_gen.bit_generator.state
        batch = cp.predict_batch(Xb, epsilon=0.1, return_p_values=True)
        cp.rnd_gen.bit_generator.state = state
        for i, x in enumerate(Xb):
            ref = cp.predict(x, epsilon=0.1, return_p_values=True)[1]
            for lb in ref:
                assert np.isclose(batch[i][1][lb], ref[lb], rtol=0, atol=0, equal_nan=True), (
                    f"batch i={i} lb={lb}: {batch[i][1][lb]} != {ref[lb]}"
                )

    def test_order_invariance(self):
        """The predictor must be permutation-invariant: building the same bag in
        two different orders must yield bit-identical p-values."""
        X, y = self._make_binary(60, seed=9, k=3)
        cp1 = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
        cp1.learn_initial_training_set(X[:30], y[:30])
        for i in range(30, 60):
            cp1.learn_one(X[i], y[i])
        idx = np.arange(59, -1, -1)  # reverse order
        cp2 = ConformalNearestNeighboursClassifier(k=3, rnd_state=0)
        cp2.learn_initial_training_set(X[idx[:30]], y[idx[:30]])
        for i in range(30, 60):
            cp2.learn_one(X[idx[i]], y[idx[i]])
        state1 = cp1.rnd_gen.bit_generator.state
        pv1 = cp1.predict(X[0], return_p_values=True)[1]
        cp1.rnd_gen.bit_generator.state = state1
        state2 = cp2.rnd_gen.bit_generator.state
        pv2 = cp2.predict(X[0], return_p_values=True)[1]
        cp2.rnd_gen.bit_generator.state = state2
        for lb in pv1:
            assert np.isclose(pv1[lb], pv2[lb], rtol=0, atol=0, equal_nan=True), (
                f"order invariance lb={lb}: {pv1[lb]} != {pv2[lb]}"
            )
