"""Tests for API consistency fixes.

Covers: PeriodicKernel name, LinearCombinationKernel name, CPS predict() alias,
NearestNeighboursVennPredictor.predict(), k-NN regressor compute_p_value(),
Wrapper return_update, evaluate.py CPS tau fix.
"""

import numpy as np
import pytest

from online_cp import (
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    GaussianKernel,
    LinearKernel,
    PeriodicKernel,
    RidgePredictionMachine,
    NearestNeighboursVennPredictor,
)
from online_cp.classifiers import ConformalClassifierWrapper
from online_cp.kernels import LinearCombinationKernel
from online_cp.CPS import ConformalPredictiveDistributionFunction


# ---------------------------------------------------------------------------
# Phase 1: Bug fixes
# ---------------------------------------------------------------------------


class TestKernelNames:
    def test_periodic_kernel_name(self):
        k = PeriodicKernel(p=1.0, s=1.0)
        assert k.name == "Periodic"

    def test_gaussian_kernel_name(self):
        k = GaussianKernel(sigma=1.0)
        assert k.name == "Gaussian"

    def test_linear_kernel_name(self):
        k = LinearKernel()
        assert k.name == "Linear"

    def test_linear_combination_kernel_name(self):
        k1 = GaussianKernel(sigma=1.0)
        k2 = LinearKernel()
        lc = LinearCombinationKernel([k1, k2], [0.3, 0.7])
        assert lc.name == "0.3*Gaussian + 0.7*Linear"

    def test_linear_combination_kernel_name_default_weights(self):
        k1 = GaussianKernel(sigma=1.0)
        k2 = PeriodicKernel(p=1.0, s=1.0)
        lc = LinearCombinationKernel([k1, k2])
        assert lc.name == "1*Gaussian + 1*Periodic"


# ---------------------------------------------------------------------------
# Phase 2: Naming consistency
# ---------------------------------------------------------------------------


class TestCPSPredictAlias:
    @pytest.fixture
    def trained_cps(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.1, 30)
        cps = RidgePredictionMachine(a=1.0)
        cps.learn_initial_training_set(X, y)
        return cps, X, y

    def test_predict_returns_cpd(self, trained_cps):
        cps, X, y = trained_cps
        x_test = X[0]
        cpd = cps.predict(x_test)
        assert isinstance(cpd, ConformalPredictiveDistributionFunction)

    def test_predict_equals_predict_cpd(self, trained_cps):
        cps, X, y = trained_cps
        x_test = X[0]
        cpd1 = cps.predict(x_test)
        cpd2 = cps.predict_cpd(x_test)
        # Both should produce the same type
        assert type(cpd1) is type(cpd2)

    def test_predict_passes_kwargs(self, trained_cps):
        cps, X, y = trained_cps
        x_test = X[0]
        cpd, precomputed = cps.predict(x_test, return_update=True)
        assert isinstance(cpd, ConformalPredictiveDistributionFunction)
        assert isinstance(precomputed, dict)


class TestNearestNeighboursVennPredictorPredict:
    @pytest.fixture
    def trained_venn(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] > 0).astype(int)
        vp = NearestNeighboursVennPredictor(k=3)
        vp.learn_initial_training_set(X, y)
        return vp, X

    def test_predict_works(self, trained_venn):
        vp, X = trained_venn
        pred = vp.predict(X[0])
        assert hasattr(pred, "p0")
        assert hasattr(pred, "p1")
        assert 0 <= pred.p0 <= 1
        assert 0 <= pred.p1 <= 1

    def test_predict_one_deprecated(self, trained_venn):
        vp, X = trained_venn
        with pytest.warns(DeprecationWarning, match="predict_one.*deprecated"):
            vp.predict_one(X[0])

    def test_predict_and_predict_one_same_result(self, trained_venn):
        vp, X = trained_venn
        pred_new = vp.predict(X[0])
        with pytest.warns(DeprecationWarning):
            pred_old = vp.predict_one(X[0])
        assert pred_new.p0 == pred_old.p0
        assert pred_new.p1 == pred_old.p1


# ---------------------------------------------------------------------------
# Phase 3: Feature parity
# ---------------------------------------------------------------------------


class TestKNNRegressorComputePValue:
    @pytest.fixture
    def trained_knn(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(30)
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=7)
        cp.learn_initial_training_set(X, y)
        return cp, X, y

    def test_p_value_in_range(self, trained_knn):
        cp, X, y = trained_knn
        p = cp.compute_p_value(X[0], y[0])
        assert 0 <= p <= 1

    def test_p_value_smoothed_vs_unsmoothed(self, trained_knn):
        cp, X, y = trained_knn
        p_smooth = cp.compute_p_value(X[0], y[0], tau=0.5, smoothed=True)
        p_raw = cp.compute_p_value(X[0], y[0], smoothed=False)
        assert 0 <= p_smooth <= 1
        assert 0 <= p_raw <= 1

    def test_p_value_with_explicit_tau(self, trained_knn):
        cp, X, y = trained_knn
        p1 = cp.compute_p_value(X[0], y[0], tau=0.0)
        p2 = cp.compute_p_value(X[0], y[0], tau=1.0)
        # tau=0 gives smallest possible smoothed p, tau=1 gives largest
        assert p1 <= p2

    def test_extreme_y_has_small_p(self, trained_knn):
        cp, X, y = trained_knn
        # An extreme prediction should have a low p-value
        p = cp.compute_p_value(X[0], 1000.0, tau=0.5)
        assert p < 0.2  # should be very small

    def test_empty_training_returns_one(self):
        cp = ConformalNearestNeighboursRegressor(k=3)
        p = cp.compute_p_value(np.array([1.0, 2.0]), 3.0)
        assert p == 1.0


class TestWrapperReturnUpdate:
    @pytest.fixture
    def trained_wrapper(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] > 0).astype(int)
        wrapper = ConformalClassifierWrapper(
            LogisticRegression(), label_space=np.array([0, 1]), rnd_state=7
        )
        wrapper.learn_initial_training_set(X, y)
        return wrapper, X

    def test_return_update_returns_tuple(self, trained_wrapper):
        wrapper, X = trained_wrapper
        result = wrapper.predict(X[0], return_update=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        prediction, precomputed = result
        assert isinstance(precomputed, dict)

    def test_return_update_false_returns_prediction(self, trained_wrapper):
        wrapper, X = trained_wrapper
        result = wrapper.predict(X[0], return_update=False)
        # Should not be a tuple
        assert not isinstance(result, tuple)


class TestDeprecatedDParameter:
    def test_d_param_emits_deprecation(self):
        from online_cp import ConformalNearestNeighboursClassifier
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 2))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=np.array([0, 1]))
        cp.learn_initial_training_set(X, y)
        # Get a D matrix
        _, D = cp.predict(X[0], return_update=True)
        with pytest.warns(DeprecationWarning, match="'D' parameter is deprecated"):
            cp.learn_one(X[0], 0, D=D)
