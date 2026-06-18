"""Tests for the Venn-Abers predictor (src/online_cp/venn.py)."""

import numpy as np
import pytest

from online_cp.venn import (
    MulticlassVennPrediction,
    VennAbersPredictor,
    VennPrediction,
    VennPredictor,
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
# VennPrediction
# ---------------------------------------------------------------------------


class TestVennPrediction:
    def test_pair_attributes(self):
        pred = VennPrediction.binary(0.2, 0.8)
        assert pred.p0 == 0.2
        assert pred.p1 == 0.8

    def test_point_prediction(self):
        pred = VennPrediction.binary(0.2, 0.8)
        point = pred.point
        assert point.shape == (2,)
        assert np.isclose(point.sum(), 1.0)

    def test_repr(self):
        pred = VennPrediction.binary(0.1, 0.9)
        assert "VennPrediction" in repr(pred)

    def test_str(self):
        pred = VennPrediction.binary(0.1, 0.9)
        assert "p0=" in str(pred) and "p1=" in str(pred)


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
# VennAbersPredictor - Kernel Ridge
# ---------------------------------------------------------------------------


class TestVennAbersKernelRidge:
    @pytest.fixture
    def trained_vap_kernel_ridge(self):
        np.random.seed(43)
        n = 60
        X = np.random.randn(n, 3)
        y = (X[:, 0] + 0.7 * X[:, 1] - 0.4 * X[:, 2] > 0).astype(int)
        vap = VennAbersPredictor(scorer="kernel_ridge", a=1.0, kernel="rbf", sigma=1.0)
        vap.learn_initial_training_set(X[:40], y[:40])
        return vap, X, y

    def test_predictions_in_unit_interval(self, trained_vap_kernel_ridge):
        vap, X, y = trained_vap_kernel_ridge
        for i in range(40, 50):
            pred = vap.predict(X[i])
            assert 0 <= pred.p0 <= 1, f"p0={pred.p0} out of [0,1]"
            assert 0 <= pred.p1 <= 1, f"p1={pred.p1} out of [0,1]"

    def test_p1_geq_p0(self, trained_vap_kernel_ridge):
        vap, X, y = trained_vap_kernel_ridge
        for i in range(40, 50):
            pred = vap.predict(X[i])
            assert pred.p1 >= pred.p0, f"p1={pred.p1} < p0={pred.p0}"

    def test_streaming_with_learn_one(self, trained_vap_kernel_ridge):
        vap, X, y = trained_vap_kernel_ridge
        for i in range(40, 45):
            pred = vap.predict(X[i])
            assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1
            vap.learn_one(X[i], y[i])
        assert vap.K.shape == (45, 45)
        assert vap.Ka_inv.shape == (45, 45)

    def test_return_update_speeds_learn(self, trained_vap_kernel_ridge):
        vap, X, y = trained_vap_kernel_ridge
        pred, precomputed = vap.predict(X[40], return_update=True)
        assert "K" in precomputed
        assert "Ka_inv" in precomputed
        assert precomputed["K"].shape == (41, 41)
        assert precomputed["Ka_inv"].shape == (41, 41)
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1

        vap.learn_one(X[40], y[40], precomputed=precomputed)
        pred2 = vap.predict(X[41])
        assert 0 <= pred2.p0 <= 1 and 0 <= pred2.p1 <= 1

    def test_kernel_linear(self):
        np.random.seed(0)
        X = np.random.randn(30, 2)
        y = (X[:, 0] > 0).astype(int)
        vap = VennAbersPredictor(scorer="kernel_ridge", a=1.0, kernel="linear")
        vap.learn_initial_training_set(X[:20], y[:20])
        pred = vap.predict(X[20])
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1

    def test_cold_start(self):
        vap = VennAbersPredictor(scorer="kernel_ridge", a=1.0)
        pred = vap.predict(np.array([1.0, 2.0, 3.0]))
        assert pred.p0 == 0.5 and pred.p1 == 0.5

    def test_near_singular_kernel_state_stays_finite(self):
        """Near-duplicate points with tiny regularization should remain numerically stable."""
        rng = np.random.default_rng(7)
        base = np.array([0.2, -0.1])
        X_dup = np.tile(base, (12, 1)) + 1e-12 * rng.normal(size=(12, 2))
        y_dup = np.array([0, 1] * 6)

        vap = VennAbersPredictor(scorer="kernel_ridge", a=1e-14, kernel="linear")
        vap.learn_initial_training_set(X_dup[:8], y_dup[:8])

        pred, precomputed = vap.predict(X_dup[8], return_update=True)
        assert np.isfinite(pred.p0)
        assert np.isfinite(pred.p1)
        assert np.isfinite(precomputed["Ka_inv"]).all()

        vap.learn_one(X_dup[8], y_dup[8], precomputed=precomputed)
        pred2 = vap.predict(X_dup[9])
        assert np.isfinite(pred2.p0)
        assert np.isfinite(pred2.p1)


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
# SVM scorer tests
# ---------------------------------------------------------------------------


class TestVennAbersSVM:
    """Test SVM scoring function for Venn-Abers predictor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(99)
        N = 60
        X = np.random.randn(N, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        self.X, self.y = X, y
        self.n_train = 30

    def test_predictions_in_unit_interval(self):
        vap = VennAbersPredictor(scorer="svm", kernel="rbf", sigma=1.0, C=10.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        for i in range(self.n_train, self.n_train + 10):
            pred = vap.predict(self.X[i])
            assert 0 <= pred.p0 <= 1, f"p0={pred.p0} out of [0,1]"
            assert 0 <= pred.p1 <= 1, f"p1={pred.p1} out of [0,1]"
            vap.learn_one(self.X[i], self.y[i])

    def test_p1_geq_p0(self):
        vap = VennAbersPredictor(scorer="svm", kernel="rbf", sigma=1.0, C=10.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        for i in range(self.n_train, self.n_train + 10):
            pred = vap.predict(self.X[i])
            assert pred.p1 >= pred.p0, f"p1={pred.p1} < p0={pred.p0}"
            vap.learn_one(self.X[i], self.y[i])

    def test_streaming_with_learn_one(self):
        vap = VennAbersPredictor(scorer="svm", kernel="rbf", sigma=1.0, C=10.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        for i in range(self.n_train, self.n_train + 5):
            vap.predict(self.X[i])
            vap.learn_one(self.X[i], self.y[i])
        assert vap.K.shape == (self.n_train + 5, self.n_train + 5)

    def test_return_update_speeds_learn(self):
        vap = VennAbersPredictor(scorer="svm", kernel="rbf", sigma=1.0, C=10.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        pred, precomputed = vap.predict(self.X[self.n_train], return_update=True)
        assert "K" in precomputed
        assert precomputed["K"].shape == (self.n_train + 1, self.n_train + 1)
        vap.learn_one(self.X[self.n_train], self.y[self.n_train], precomputed=precomputed)
        assert vap.K.shape == (self.n_train + 1, self.n_train + 1)

    def test_kernel_linear(self):
        vap = VennAbersPredictor(scorer="svm", kernel="linear", C=1.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        pred = vap.predict(self.X[self.n_train])
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1

    def test_kernel_poly(self):
        vap = VennAbersPredictor(scorer="svm", kernel="poly", degree=2, C=1.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        pred = vap.predict(self.X[self.n_train])
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1

    def test_kernel_instance(self):
        from online_cp.kernels import GaussianKernel
        vap = VennAbersPredictor(scorer="svm", kernel=GaussianKernel(sigma=2.0), C=5.0)
        vap.learn_initial_training_set(self.X[:self.n_train], self.y[:self.n_train])
        pred = vap.predict(self.X[self.n_train])
        assert 0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1

    def test_cold_start(self):
        vap = VennAbersPredictor(scorer="svm", kernel="rbf", sigma=1.0, C=10.0)
        pred = vap.predict(self.X[0])
        assert pred.p0 == 0.5 and pred.p1 == 0.5


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
            VennAbersPredictor(scorer="invalid")

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError, match="aggregation must be"):
            VennAbersPredictor(scorer="knn", aggregation="sum")

    def test_non_binary_labels(self):
        """Multiclass labels are now accepted (adaptive label_space)."""
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        X = np.random.randn(10, 2)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        vap.learn_initial_training_set(X, y)
        assert len(vap.label_space) == 3
        np.testing.assert_array_equal(vap.label_space, [0, 1, 2])


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


# ---------------------------------------------------------------------------
# NearestNeighboursVennPredictor
# ---------------------------------------------------------------------------

from online_cp.venn import NearestNeighboursVennPredictor  # noqa: E402


class TestVennPredictorInheritance:
    """Both concrete predictors must be instances of the VennPredictor base."""

    def test_venn_abers_is_venn_predictor(self):
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        assert isinstance(vap, VennPredictor)

    def test_nearest_neighbours_is_venn_predictor(self):
        vp = NearestNeighboursVennPredictor(k=1)
        assert isinstance(vp, VennPredictor)


class TestNearestNeighboursVennPredictor:
    """Tests for the taxonomy-based Venn predictor."""

    def test_slide_example_k1(self):
        """Reproduce the exact k=1 example from Ilia's ALRW lecture slides.

        Training: (+) (0,3), (2,2), (3,3); (-) (-1,1), (-1,-1), (0,1)
        New: (1,2)
        Expected: s(+1,+1)=1, s(+1,-1)=0, s(-1,+1)=0.5, s(-1,-1)=0.5
        → p0=s(0,1)=0.5, p1=s(1,1)=1.0
        """
        # Labels: positive=1, negative=0
        X_train = np.array([
            [0, 3], [2, 2], [3, 3],       # positive
            [-1, 1], [-1, -1], [0, 1],     # negative
        ], dtype=float)
        y_train = np.array([1, 1, 1, 0, 0, 0])

        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_initial_training_set(X_train, y_train)

        x_new = np.array([1.0, 2.0])
        pred = vp.predict(x_new)

        # Under hypothesis v=1 (y_new=1):
        # The new point's taxonomy should place it with all positives
        # s(1, 1) = 4/4 = 1.0, s(1, 0) = 0/4 = 0
        assert abs(pred.p1 - 1.0) < 1e-10

        # Under hypothesis v=0 (y_new=0):
        # s(0, 1) = 0.5, s(0, 0) = 0.5
        assert abs(pred.p0 - 0.5) < 1e-10

    def test_prediction_bounds(self):
        """Check that p0 <= p1 for well-separated data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [2, 0],
            np.random.randn(20, 2) + [-2, 0],
        ])
        y = np.array([1] * 20 + [0] * 20)

        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_initial_training_set(X[:35], y[:35])

        # Test a clearly positive point
        pred = vp.predict(np.array([3.0, 0.0]))
        assert pred.p0 <= pred.p1

    def test_k_greater_than_1(self):
        """Test k=3 produces valid output with taxonomy in {0,1,2,3}."""
        np.random.seed(7)
        X = np.random.randn(30, 2)
        y = (X[:, 0] > 0).astype(int)

        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:25], y[:25])

        pred = vp.predict(X[25])
        assert 0 <= pred.p0 <= 1
        assert 0 <= pred.p1 <= 1

    def test_learn_one_incremental(self):
        """learn_one produces same result as batch initialisation."""
        np.random.seed(99)
        X = np.random.randn(10, 2)
        y = (X[:, 0] > 0).astype(int)

        # Batch
        vp_batch = NearestNeighboursVennPredictor(k=1)
        vp_batch.learn_initial_training_set(X, y)

        # Incremental
        vp_inc = NearestNeighboursVennPredictor(k=1)
        vp_inc.learn_initial_training_set(X[:5], y[:5])
        for i in range(5, 10):
            vp_inc.learn_one(X[i], y[i])

        x_test = np.array([0.5, 0.5])
        pred_batch = vp_batch.predict(x_test)
        pred_inc = vp_inc.predict(x_test)
        assert abs(pred_batch.p0 - pred_inc.p0) < 1e-10
        assert abs(pred_batch.p1 - pred_inc.p1) < 1e-10

    def test_k_capped_at_available(self):
        """k larger than n-1 should not error — use all available neighbours."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        y = np.array([0, 1, 1])

        vp = NearestNeighboursVennPredictor(k=10)  # k >> n
        vp.learn_initial_training_set(X, y)

        pred = vp.predict(np.array([1.5, 1.5]))
        assert 0 <= pred.p0 <= 1
        assert 0 <= pred.p1 <= 1

    def test_empty_predictor_returns_uniform(self):
        """Prediction with no training data returns (0.5, 0.5)."""
        vp = NearestNeighboursVennPredictor(k=1)
        pred = vp.predict(np.array([1.0, 2.0]))
        assert pred.p0 == 0.5
        assert pred.p1 == 0.5

    def test_single_training_point(self):
        """With one training example, prediction should not crash."""
        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_one(np.array([1.0, 0.0]), 1)
        pred = vp.predict(np.array([0.0, 0.0]))
        assert 0 <= pred.p0 <= 1
        assert 0 <= pred.p1 <= 1

    def test_invalid_k(self):
        """k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            NearestNeighboursVennPredictor(k=0)

    def test_non_binary_labels(self):
        """Non-binary labels are accepted (multiclass support)."""
        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_initial_training_set(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([0, 1, 2]),
        )
        pred = vp.predict(np.array([0.5, 0.5]))
        assert isinstance(pred, MulticlassVennPrediction)

    def test_output_type(self):
        """predict returns VennPrediction."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_initial_training_set(X, y)
        pred = vp.predict(np.array([2.5, 2.5]))
        assert isinstance(pred, VennPrediction)

    def test_compatible_with_log_loss_point(self):
        """Output works with log_loss_point and brier_point utilities."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_initial_training_set(X, y)
        pred = vp.predict(np.array([2.5, 2.5]))
        ll = log_loss_point(pred.p0, pred.p1)
        br = brier_point(pred.p0, pred.p1)
        assert 0 <= ll <= 1
        assert 0 <= br <= 1


class TestNearestNeighboursVennValidity:
    """Statistical validity: error bounds should be calibrated."""

    def test_validity_on_synthetic_data(self):
        """Check that empirical error rate falls within [L, U] bounds on average."""
        np.random.seed(123)
        n_train = 50
        n_test = 150
        X = np.random.randn(n_train + n_test, 2)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:n_train], y[:n_train])

        errors = 0
        sum_lower = 0.0
        sum_upper = 0.0

        for i in range(n_train, n_train + n_test):
            pred = vp.predict(X[i])

            # Predicted label: same rule as VAP (p0+p1 > 1 → predict 1)
            y_hat = 1 if pred.p0 + pred.p1 > 1 else 0

            # Error bounds for predicted label
            # L = 1 - max(s(v, y_hat)), U = 1 - min(s(v, y_hat))
            lower = 1 - max(pred.p0 if y_hat == 1 else 1 - pred.p0,
                            pred.p1 if y_hat == 1 else 1 - pred.p1)
            upper = 1 - min(pred.p0 if y_hat == 1 else 1 - pred.p0,
                            pred.p1 if y_hat == 1 else 1 - pred.p1)

            if y_hat != y[i]:
                errors += 1
            sum_lower += lower
            sum_upper += upper

            vp.learn_one(X[i], y[i])

        error_rate = errors / n_test
        avg_lower = sum_lower / n_test
        avg_upper = sum_upper / n_test

        # The empirical error rate should be roughly between
        # the average L and U (validity property).
        # Use generous tolerance since sample is finite.
        assert avg_lower - 0.05 <= error_rate <= avg_upper + 0.05


# ===========================================================================
# Multiclass NearestNeighboursVennPredictor tests
# ===========================================================================



class TestMulticlassNearestNeighboursVenn:
    """Tests for multiclass extension of NearestNeighboursVennPredictor."""

    def _make_3class_data(self, rng, n=90):
        """Well-separated 3-class Gaussian data, shuffled."""
        n_per = n // 3
        centers = np.array([[0, 0], [4, 0], [2, 4]])
        X = np.vstack([rng.normal(c, 0.5, size=(n_per, 2)) for c in centers])
        y = np.array([0] * n_per + [1] * n_per + [2] * n_per)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def test_output_type_multiclass(self):
        """Multiclass returns MulticlassVennPrediction."""
        rng = np.random.default_rng(42)
        X, y = self._make_3class_data(rng)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:40], y[:40])
        pred = vp.predict(X[40])
        assert isinstance(pred, MulticlassVennPrediction)

    def test_output_type_binary(self):
        """Binary data still returns VennPrediction (backward compat)."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 2))
        y = (X[:, 0] > 0).astype(int)
        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_initial_training_set(X[:20], y[:20])
        pred = vp.predict(X[20])
        assert isinstance(pred, VennPrediction)

    def test_rows_sum_to_one(self):
        """Each row of the multiprobability matrix sums to 1."""
        rng = np.random.default_rng(7)
        X, y = self._make_3class_data(rng)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:40], y[:40])

        for i in range(40, 50):
            pred = vp.predict(X[i])
            row_sums = pred.probs.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_probabilities_non_negative(self):
        """All entries in the matrix are >= 0."""
        rng = np.random.default_rng(8)
        X, y = self._make_3class_data(rng)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:40], y[:40])

        for i in range(40, 50):
            pred = vp.predict(X[i])
            assert np.all(pred.probs >= 0)

    def test_diagonal_dominance_well_separated(self):
        """On well-separated data, diagonal entries should be largest in row."""
        rng = np.random.default_rng(9)
        X, y = self._make_3class_data(rng, n=90)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:60], y[:60])

        # Most predictions should have diagonal dominance
        diag_dominant_count = 0
        for i in range(60, 90):
            pred = vp.predict(X[i])
            for row_idx in range(3):
                if pred.probs[row_idx, row_idx] == pred.probs[row_idx].max():
                    diag_dominant_count += 1

        # At least 80% of (hypothesis, prediction) pairs should be diag-dominant
        assert diag_dominant_count >= 0.8 * (30 * 3)

    def test_streaming(self):
        """predict → learn_one loop works correctly."""
        rng = np.random.default_rng(10)
        X, y = self._make_3class_data(rng, n=60)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:30], y[:30])

        for i in range(30, 50):
            pred = vp.predict(X[i])
            assert isinstance(pred, MulticlassVennPrediction)
            assert pred.probs.shape == (3, 3)
            vp.learn_one(X[i], y[i])

        # After streaming, internal state should have grown
        assert vp.X.shape[0] == 50

    def test_point_prediction(self):
        """Point prediction is a valid probability vector."""
        rng = np.random.default_rng(11)
        X, y = self._make_3class_data(rng)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:40], y[:40])

        pred = vp.predict(X[40])
        point = pred.point
        assert point.shape == (3,)
        assert np.all(point >= 0)
        np.testing.assert_allclose(point.sum(), 1.0, atol=1e-12)

    def test_5class(self):
        """Works with 5 classes."""
        rng = np.random.default_rng(12)
        n_per_class = 20
        centers = rng.normal(size=(5, 3)) * 5
        X = np.vstack(
            [rng.normal(c, 0.3, size=(n_per_class, 3)) for c in centers]
        )
        y = np.repeat(np.arange(5), n_per_class)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:80], y[:80])

        pred = vp.predict(X[80])
        assert isinstance(pred, MulticlassVennPrediction)
        assert pred.probs.shape == (5, 5)
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)

    def test_explicit_label_space(self):
        """Explicit label_space handles missing labels in training data."""
        rng = np.random.default_rng(13)
        X = rng.normal(size=(30, 2))
        # Training data only has labels 0, 1 but label_space declares 0,1,2
        y_train = (X[:20, 0] > 0).astype(int)
        vp = NearestNeighboursVennPredictor(k=3, label_space=[0, 1, 2])
        vp.learn_initial_training_set(X[:20], y_train)

        pred = vp.predict(X[20])
        assert isinstance(pred, MulticlassVennPrediction)
        assert pred.probs.shape == (3, 3)
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)

    def test_label_space_expanded_by_learn_one(self):
        """learn_one with unseen label expands label_space."""
        rng = np.random.default_rng(14)
        X = rng.normal(size=(25, 2))
        y = np.zeros(20, dtype=int)
        y[10:] = 1

        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X[:20], y)
        assert len(vp.label_space) == 2

        # Add a new class via learn_one
        vp.learn_one(X[20], 2)
        assert 2 in vp.label_space
        assert len(vp.label_space) == 3

        pred = vp.predict(X[21])
        assert isinstance(pred, MulticlassVennPrediction)
        assert pred.probs.shape == (3, 3)

    def test_cold_start(self):
        """Works from empty state with learn_one only."""
        rng = np.random.default_rng(15)
        X = rng.normal(size=(15, 2))
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

        vp = NearestNeighboursVennPredictor(k=2, label_space=[0, 1, 2])
        # Cold start prediction
        pred = vp.predict(X[0])
        assert isinstance(pred, MulticlassVennPrediction)

        # Build up from scratch
        for i in range(10):
            vp.learn_one(X[i], y[i])

        pred = vp.predict(X[10])
        assert isinstance(pred, MulticlassVennPrediction)
        assert pred.probs.shape == (3, 3)
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)

    def test_k_greater_than_n(self):
        """k > n doesn't crash; uses all available neighbours."""
        rng = np.random.default_rng(16)
        X = rng.normal(size=(10, 2))
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        vp = NearestNeighboursVennPredictor(k=20)  # k > n
        vp.learn_initial_training_set(X[:5], y[:5])

        pred = vp.predict(X[5])
        assert isinstance(pred, MulticlassVennPrediction)
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)


class TestLabelSpacePolicy:
    """Label-space policy: explicit = fixed (rejects unknowns), implicit = adaptive."""

    def test_fixed_rejects_unknown_learn_one(self):
        """Fixed label_space raises ValueError on unknown label in learn_one."""
        vp = NearestNeighboursVennPredictor(k=1, label_space=[0, 1])
        vp.learn_one(np.array([1.0, 0.0]), 0)
        with pytest.raises(ValueError, match="not in declared label_space"):
            vp.learn_one(np.array([2.0, 0.0]), 2)

    def test_fixed_rejects_unknown_learn_initial(self):
        """Fixed label_space raises ValueError on unknown label in learn_initial_training_set."""
        vp = NearestNeighboursVennPredictor(k=1, label_space=[0, 1])
        with pytest.raises(ValueError, match="not in declared label_space"):
            vp.learn_initial_training_set(
                np.array([[0, 0], [1, 1], [2, 2]], dtype=float),
                np.array([0, 1, 2]),
            )

    def test_adaptive_expands(self):
        """Adaptive label_space grows on new labels."""
        vp = NearestNeighboursVennPredictor(k=1)
        vp.learn_one(np.array([0.0, 0.0]), 0)
        assert list(vp.label_space) == [0]
        vp.learn_one(np.array([1.0, 1.0]), 1)
        assert list(vp.label_space) == [0, 1]
        vp.learn_one(np.array([2.0, 2.0]), 5)
        assert list(vp.label_space) == [0, 1, 5]

    def test_fixed_flag_set(self):
        """_label_space_fixed reflects construction argument."""
        vp_fixed = NearestNeighboursVennPredictor(k=1, label_space=[0, 1, 2])
        assert vp_fixed._label_space_fixed is True
        vp_adapt = NearestNeighboursVennPredictor(k=1)
        assert vp_adapt._label_space_fixed is False


# ===========================================================================
# Multiclass VennAbersPredictor tests
# ===========================================================================


class TestMulticlassVennAbersPredictor:
    """Tests for multiclass extension of VennAbersPredictor."""

    def _make_3class_data(self, rng, n=90):
        """Well-separated 3-class Gaussian data, shuffled."""
        n_per = n // 3
        centers = np.array([[0, 0], [4, 0], [2, 4]])
        X = np.vstack([rng.normal(c, 0.5, size=(n_per, 2)) for c in centers])
        y = np.array([0] * n_per + [1] * n_per + [2] * n_per)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    @pytest.mark.parametrize("scorer", ["ridge", "kernel_ridge", "knn", "svm"])
    def test_output_type_multiclass(self, scorer):
        """Multiclass returns VennPrediction with |Y|×|Y| matrix."""
        rng = np.random.default_rng(42)
        X, y = self._make_3class_data(rng)
        kwargs = {"scorer": scorer, "label_space": [0, 1, 2]}
        if scorer == "ridge":
            kwargs["a"] = 1.0
        elif scorer == "kernel_ridge":
            kwargs["a"] = 1.0
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
        elif scorer == "knn":
            kwargs["k"] = 3
        elif scorer == "svm":
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
            kwargs["C"] = 1.0
        vap = VennAbersPredictor(**kwargs)
        vap.learn_initial_training_set(X[:40], y[:40])
        pred = vap.predict(X[40])
        assert isinstance(pred, VennPrediction)
        assert pred.probs.shape == (3, 3)

    @pytest.mark.parametrize("scorer", ["ridge", "kernel_ridge", "knn", "svm"])
    def test_rows_sum_to_one(self, scorer):
        """Each row of the probs matrix sums to 1."""
        rng = np.random.default_rng(7)
        X, y = self._make_3class_data(rng)
        kwargs = {"scorer": scorer, "label_space": [0, 1, 2]}
        if scorer == "ridge":
            kwargs["a"] = 1.0
        elif scorer == "kernel_ridge":
            kwargs["a"] = 1.0
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
        elif scorer == "knn":
            kwargs["k"] = 3
        elif scorer == "svm":
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
            kwargs["C"] = 1.0
        vap = VennAbersPredictor(**kwargs)
        vap.learn_initial_training_set(X[:40], y[:40])
        for i in range(40, 45):
            pred = vap.predict(X[i])
            np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("scorer", ["ridge", "kernel_ridge", "knn", "svm"])
    def test_probabilities_non_negative(self, scorer):
        """All entries in the probs matrix are >= 0."""
        rng = np.random.default_rng(8)
        X, y = self._make_3class_data(rng)
        kwargs = {"scorer": scorer, "label_space": [0, 1, 2]}
        if scorer == "ridge":
            kwargs["a"] = 1.0
        elif scorer == "kernel_ridge":
            kwargs["a"] = 1.0
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
        elif scorer == "knn":
            kwargs["k"] = 3
        elif scorer == "svm":
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
            kwargs["C"] = 1.0
        vap = VennAbersPredictor(**kwargs)
        vap.learn_initial_training_set(X[:40], y[:40])
        for i in range(40, 45):
            pred = vap.predict(X[i])
            assert np.all(pred.probs >= -1e-12)

    @pytest.mark.parametrize("scorer", ["ridge", "kernel_ridge", "knn", "svm"])
    def test_streaming(self, scorer):
        """predict → learn_one loop works correctly."""
        rng = np.random.default_rng(10)
        X, y = self._make_3class_data(rng, n=60)
        kwargs = {"scorer": scorer, "label_space": [0, 1, 2]}
        if scorer == "ridge":
            kwargs["a"] = 1.0
        elif scorer == "kernel_ridge":
            kwargs["a"] = 1.0
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
        elif scorer == "knn":
            kwargs["k"] = 3
        elif scorer == "svm":
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
            kwargs["C"] = 1.0
        vap = VennAbersPredictor(**kwargs)
        vap.learn_initial_training_set(X[:30], y[:30])
        for i in range(30, 45):
            pred = vap.predict(X[i])
            assert pred.probs.shape == (3, 3)
            np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-10)
            vap.learn_one(X[i], y[i])
        assert vap.X.shape[0] == 45

    @pytest.mark.parametrize("scorer", ["ridge", "kernel_ridge", "knn", "svm"])
    def test_return_update(self, scorer):
        """return_update=True returns precomputed dict."""
        rng = np.random.default_rng(11)
        X, y = self._make_3class_data(rng)
        kwargs = {"scorer": scorer, "label_space": [0, 1, 2]}
        if scorer == "ridge":
            kwargs["a"] = 1.0
        elif scorer == "kernel_ridge":
            kwargs["a"] = 1.0
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
        elif scorer == "knn":
            kwargs["k"] = 3
        elif scorer == "svm":
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
            kwargs["C"] = 1.0
        vap = VennAbersPredictor(**kwargs)
        vap.learn_initial_training_set(X[:40], y[:40])
        result = vap.predict(X[40], return_update=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pred, precomputed = result
        assert isinstance(pred, VennPrediction)
        assert isinstance(precomputed, dict)

    def test_cold_start_multiclass(self):
        """Cold-start returns uniform probs matrix."""
        vap = VennAbersPredictor(scorer="ridge", a=1.0, label_space=[0, 1, 2])
        pred = vap.predict(np.array([1.0, 2.0]))
        assert pred.probs.shape == (3, 3)
        np.testing.assert_allclose(pred.probs, 1.0 / 3)

    def test_binary_backward_compat(self):
        """Binary data without label_space still gives binary VennPrediction."""
        rng = np.random.default_rng(99)
        X = rng.normal(size=(30, 2))
        y = (X[:, 0] > 0).astype(int)
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        vap.learn_initial_training_set(X[:20], y[:20])
        pred = vap.predict(X[20])
        assert isinstance(pred, VennPrediction)
        # Should still have p0/p1 since label_space=[0,1]
        assert 0 <= pred.p0 <= 1
        assert 0 <= pred.p1 <= 1

    def test_label_space_policy_fixed_rejects(self):
        """Fixed label_space rejects unknown labels."""
        vap = VennAbersPredictor(scorer="ridge", a=1.0, label_space=[0, 1, 2])
        vap.learn_initial_training_set(
            np.random.randn(9, 2), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        )
        with pytest.raises(ValueError, match="not in declared label_space"):
            vap.learn_one(np.array([0.0, 0.0]), 5)

    def test_label_space_policy_adaptive_expands(self):
        """Adaptive label_space grows when new labels arrive."""
        vap = VennAbersPredictor(scorer="ridge", a=1.0)
        vap.learn_initial_training_set(
            np.random.randn(6, 2), np.array([0, 1, 0, 1, 0, 1])
        )
        assert list(vap.label_space) == [0, 1]
        vap.learn_one(np.array([0.0, 0.0]), 2)
        assert 2 in vap.label_space

    @pytest.mark.parametrize("scorer", ["ridge", "kernel_ridge"])
    def test_binary_uses_binary_path(self, scorer):
        """With binary data and no explicit label_space, binary path is used."""
        rng = np.random.default_rng(77)
        X = rng.normal(size=(30, 2))
        y = (X[:, 0] > 0).astype(int)

        kwargs = {"scorer": scorer, "a": 1.0}
        if scorer == "kernel_ridge":
            kwargs["kernel"] = "rbf"
            kwargs["sigma"] = 1.0
        vap = VennAbersPredictor(**kwargs)
        vap.learn_initial_training_set(X[:20], y[:20])
        pred = vap.predict(X[20])
        # label_space should be [0,1] (inferred) → binary path
        assert len(vap.label_space) == 2
        assert 0 <= pred.p0 <= 1
        assert 0 <= pred.p1 <= 1
        # Rows must sum to 1
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)

    def test_point_prediction(self):
        """Point prediction is a valid probability vector."""
        rng = np.random.default_rng(12)
        X, y = self._make_3class_data(rng)
        vap = VennAbersPredictor(scorer="ridge", a=1.0, label_space=[0, 1, 2])
        vap.learn_initial_training_set(X[:40], y[:40])
        pred = vap.predict(X[40])
        point = pred.point
        assert point.shape == (3,)
        assert np.all(point >= 0)
        np.testing.assert_allclose(point.sum(), 1.0, atol=1e-12)
