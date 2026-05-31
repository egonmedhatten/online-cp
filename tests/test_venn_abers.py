"""Tests for the Venn-Abers predictor (src/online_cp/venn.py)."""

import numpy as np
import pytest

from online_cp.venn import (
    VennAbersPrediction,
    VennAbersPredictor,
    _isotonic_calibrate,
    _pava_inplace,
    brier_point,
    log_loss_point,
)


# ---------------------------------------------------------------------------
# PAVA tests
# ---------------------------------------------------------------------------


class TestPAVA:
    def test_already_isotonic(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        result = _pava_inplace(y, w)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_reverse_order(self):
        y = np.array([4.0, 3.0, 2.0, 1.0])
        w = np.ones(4)
        result = _pava_inplace(y, w)
        # All should be merged to the mean
        np.testing.assert_array_almost_equal(result, [2.5, 2.5, 2.5, 2.5])

    def test_single_violation(self):
        y = np.array([1.0, 3.0, 2.0, 4.0])
        w = np.ones(4)
        result = _pava_inplace(y, w)
        np.testing.assert_array_almost_equal(result, [1.0, 2.5, 2.5, 4.0])

    def test_weighted(self):
        y = np.array([3.0, 1.0])
        w = np.array([1.0, 2.0])
        result = _pava_inplace(y, w)
        expected = (3.0 * 1 + 1.0 * 2) / 3.0
        np.testing.assert_array_almost_equal(result, [expected, expected])

    def test_empty(self):
        y = np.array([], dtype=np.float64)
        w = np.array([], dtype=np.float64)
        result = _pava_inplace(y, w)
        assert len(result) == 0

    def test_single_element(self):
        y = np.array([5.0])
        w = np.array([1.0])
        result = _pava_inplace(y, w)
        np.testing.assert_array_almost_equal(result, [5.0])


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------


class TestIsotonicCalibrate:
    def test_perfect_separation(self):
        # Scores perfectly separate labels
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        # Query is at index 3 (score=0.9, label=1)
        val = _isotonic_calibrate(scores, labels, 3)
        assert val == 1.0

    def test_query_in_low_region(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        val = _isotonic_calibrate(scores, labels, 0)
        assert val == 0.0


# ---------------------------------------------------------------------------
# VennAbersPrediction
# ---------------------------------------------------------------------------


class TestVennAbersPrediction:
    def test_pair_attributes(self):
        pred = VennAbersPrediction(0.2, 0.8)
        assert pred.p0 == 0.2
        assert pred.p1 == 0.8

    def test_no_point_attribute(self):
        pred = VennAbersPrediction(0.2, 0.8)
        assert not hasattr(pred, "point")

    def test_repr(self):
        pred = VennAbersPrediction(0.1, 0.9)
        assert "VennAbersPrediction" in repr(pred)
        assert "point" not in repr(pred)

    def test_str(self):
        pred = VennAbersPrediction(0.1, 0.9)
        assert "p0=" in str(pred) and "p1=" in str(pred)
        assert "point" not in str(pred)


# ---------------------------------------------------------------------------
# VennAbersPredictor — Ridge
# ---------------------------------------------------------------------------


class TestVennAbersRidge:
    @pytest.fixture
    def trained_vap(self):
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        vap.learn_initial_training_set(X[:30], y[:30])
        return vap, X, y

    def test_predictions_in_unit_interval(self, trained_vap):
        vap, X, y = trained_vap
        for i in range(30, 40):
            pred = vap.predict(X[i])
            assert 0 <= pred.p0 <= 1, f"p0={pred.p0} out of [0,1]"
            assert 0 <= pred.p1 <= 1, f"p1={pred.p1} out of [0,1]"

    def test_p1_geq_p0(self, trained_vap):
        """p1 should generally be >= p0 (more confident about y=1 when label is 1)."""
        vap, X, y = trained_vap
        for i in range(30, 40):
            pred = vap.predict(X[i])
            assert pred.p1 >= pred.p0, f"p1={pred.p1} < p0={pred.p0}"

    def test_streaming_with_learn_one(self, trained_vap):
        vap, X, y = trained_vap
        predictions = []
        for i in range(30, 40):
            pred = vap.predict(X[i])
            predictions.append(pred)
            vap.learn_one(X[i], y[i])
        # All predictions should be valid pairs
        for pred in predictions:
            assert 0 <= pred.p0 <= 1
            assert 0 <= pred.p1 <= 1

    def test_return_update_speeds_learn(self, trained_vap):
        """Using return_update should give same result and precomputed state."""
        vap, X, y = trained_vap
        pred, precomputed = vap.predict(X[30], return_update=True)
        assert "XTXinv" in precomputed
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1
        # learn_one with precomputed
        vap.learn_one(X[30], y[30], precomputed=precomputed)
        # Still works after learning
        pred2 = vap.predict(X[31])
        assert 0 <= pred2.p0 <= 1 and 0 <= pred2.p1 <= 1

    def test_cold_start(self):
        """Predictor with no data should return (0.5, 0.5)."""
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        pred = vap.predict(np.array([1.0, 2.0, 3.0]))
        assert pred.p0 == 0.5 and pred.p1 == 0.5

    def test_learn_from_scratch(self):
        """Learn one at a time from empty."""
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        np.random.seed(0)
        X = np.random.randn(10, 2)
        y = (X[:, 0] > 0).astype(int)
        for i in range(10):
            vap.learn_one(X[i], y[i])
        pred = vap.predict(np.array([1.0, 0.5]))
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1


# ---------------------------------------------------------------------------
# VennAbersPredictor — k-NN
# ---------------------------------------------------------------------------


class TestVennAbersKNN:
    @pytest.fixture
    def trained_vap_knn(self):
        np.random.seed(42)
        n = 60
        X = np.random.randn(n, 2)
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int)
        vap = VennAbersPredictor(scorer="knn", k=3)
        vap.learn_initial_training_set(X[:40], y[:40])
        return vap, X, y

    def test_predictions_in_unit_interval(self, trained_vap_knn):
        vap, X, y = trained_vap_knn
        for i in range(40, 50):
            pred = vap.predict(X[i])
            assert 0 <= pred.p0 <= 1, f"p0={pred.p0}"
            assert 0 <= pred.p1 <= 1, f"p1={pred.p1}"

    def test_p1_geq_p0(self, trained_vap_knn):
        vap, X, y = trained_vap_knn
        for i in range(40, 50):
            pred = vap.predict(X[i])
            assert pred.p1 >= pred.p0, f"p1={pred.p1} < p0={pred.p0}"

    def test_streaming(self, trained_vap_knn):
        vap, X, y = trained_vap_knn
        for i in range(40, 50):
            pred, precomputed = vap.predict(X[i], return_update=True)
            assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1
            vap.learn_one(X[i], y[i], precomputed=precomputed)

    def test_custom_distance(self):
        """Test with a custom distance function."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0] > 0).astype(int)

        def my_dist(X, Y=None):
            X = np.atleast_2d(X)
            if Y is None:
                from scipy.spatial.distance import pdist, squareform
                return squareform(pdist(X, metric="cityblock"))
            else:
                from scipy.spatial.distance import cdist
                Y = np.atleast_2d(Y)
                return cdist(X, Y, metric="cityblock")

        vap = VennAbersPredictor(scorer="knn", k=3, distance_func=my_dist)
        vap.learn_initial_training_set(X[:15], y[:15])
        pred = vap.predict(X[15])
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1


# ---------------------------------------------------------------------------
# Validity / calibration test (statistical)
# ---------------------------------------------------------------------------


class TestVennAbersValidity:
    """Statistical test: aggregated predictions for true class-1 should
    be higher than for true class-0."""

    def test_calibration_direction(self):
        np.random.seed(123)
        n_train, n_test = 100, 50
        X = np.random.randn(n_train + n_test, 4)
        true_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))
        y = (np.random.rand(n_train + n_test) < true_prob).astype(int)

        vap = VennAbersPredictor(scorer="ridge", a=0.1)
        vap.learn_initial_training_set(X[:n_train], y[:n_train])

        preds = []
        for i in range(n_train, n_train + n_test):
            pred = vap.predict(X[i])
            # Use log_loss_point to aggregate for evaluation
            preds.append(log_loss_point(pred.p0, pred.p1))
            vap.learn_one(X[i], y[i])

        preds = np.array(preds)
        y_test = y[n_train:]

        mean_pred_1 = preds[y_test == 1].mean()
        mean_pred_0 = preds[y_test == 0].mean()
        assert mean_pred_1 > mean_pred_0, (
            f"Mean pred for class 1 ({mean_pred_1:.3f}) should be > "
            f"class 0 ({mean_pred_0:.3f})"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_scorer(self):
        with pytest.raises(ValueError, match="scorer must be"):
            VennAbersPredictor(scorer="svm")

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError, match="aggregation must be"):
            VennAbersPredictor(scorer="knn", aggregation="sum")

    def test_non_binary_labels(self):
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        with pytest.raises(ValueError, match="binary"):
            vap.learn_initial_training_set(
                np.random.randn(10, 2), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
            )


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


class TestAggregationFunctions:
    def test_log_loss_point_symmetric(self):
        # p0=0.2, p1=0.8 → 0.8 / (1 - 0.2 + 0.8) = 0.8/1.6 = 0.5
        assert abs(log_loss_point(0.2, 0.8) - 0.5) < 1e-10

    def test_log_loss_point_extreme(self):
        assert abs(log_loss_point(0.0, 1.0) - 0.5) < 1e-10

    def test_log_loss_point_certain(self):
        # p0=0.9, p1=1.0 → 1.0 / (1 - 0.9 + 1.0) = 1.0/1.1 ≈ 0.909
        assert abs(log_loss_point(0.9, 1.0) - 1.0 / 1.1) < 1e-10

    def test_log_loss_point_degenerate(self):
        # denom = 1 - 1 + 0 = 0 → returns 0.5
        assert log_loss_point(1.0, 0.0) == 0.5

    def test_brier_point_symmetric(self):
        assert abs(brier_point(0.2, 0.8) - 0.5) < 1e-10

    def test_brier_point_is_mean(self):
        assert abs(brier_point(0.3, 0.7) - 0.5) < 1e-10
        assert abs(brier_point(0.1, 0.9) - 0.5) < 1e-10
        assert abs(brier_point(0.0, 0.6) - 0.3) < 1e-10

    def test_brier_point_extreme(self):
        assert abs(brier_point(0.0, 1.0) - 0.5) < 1e-10

    def test_log_loss_and_brier_differ(self):
        # They should give different results for asymmetric pairs
        p0, p1 = 0.1, 0.9
        ll = log_loss_point(p0, p1)
        br = brier_point(p0, p1)
        # In this case both are 0.5, but for asymmetric:
        p0, p1 = 0.3, 0.9
        ll = log_loss_point(p0, p1)  # 0.9 / (0.7 + 0.9) = 0.9/1.6 = 0.5625
        br = brier_point(p0, p1)  # (0.3 + 0.9) / 2 = 0.6
        assert abs(ll - 0.5625) < 1e-10
        assert abs(br - 0.6) < 1e-10
        assert ll != br
