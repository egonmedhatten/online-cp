"""Round-trip save/load tests for online-cp serialization.

Tests cover:
- Functional round-trip: save → load → predict equality for all Phase-A classes.
- RNG reproducibility: advance rnd_gen BEFORE save; post-load predictions match.
- Derived-matrix integrity: XTXinv, D, K arrays equal after load.
- Version / class mismatch: correct errors/warnings raised.
- Callable registry: registered function round-trips; lambda raises SerializationError.
- Symmetry: loaded model still order-invariant under training-data permutation.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from online_cp import (
    ConformalLassoRegressor,
    ConformalNearestNeighboursClassifier,
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    ConformalSupportVectorMachine,
    GaussianKernel,
    KernelConformalRidgeRegressor,
    KernelRidgePredictionMachine,
    RidgePredictionMachine,
)
from online_cp._serialization import (
    SerializationError,
    SerializableMixin,
    register_callable,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)
N_TRAIN = 40
D_FEATURES = 2

X_REG = RNG.standard_normal((N_TRAIN, D_FEATURES))
Y_REG = X_REG[:, 0] + RNG.standard_normal(N_TRAIN) * 0.15

X_CLF = np.vstack([
    RNG.standard_normal((20, 2)) + np.array([-2, 0]),
    RNG.standard_normal((20, 2)) + np.array([2, 0]),
])
Y_CLF = np.array([-1] * 20 + [1] * 20)

X_TEST_REG = np.array([0.5, -0.3])
X_TEST_CLF = np.array([1.5, 0.2])


def _train_regressor(cls, **kwargs):
    m = cls(**kwargs)
    m.learn_initial_training_set(X_REG, Y_REG)
    return m


def _train_classifier(cls, **kwargs):
    m = cls(**kwargs)
    m.learn_initial_training_set(X_CLF, Y_CLF)
    return m


# ---------------------------------------------------------------------------
# Functional round-trip — regressors
# ---------------------------------------------------------------------------

class TestRidgeRegressorRoundTrip:
    def test_predict_interval_equality(self, tmp_path):
        cp = _train_regressor(ConformalRidgeRegressor, a=1e-3, rnd_state=42)
        path = tmp_path / "ridge.joblib"
        cp.save(str(path))
        loaded = ConformalRidgeRegressor.load(str(path))
        orig = cp.predict(X_TEST_REG)
        back = loaded.predict(X_TEST_REG)
        assert abs(orig.lower - back.lower) < 1e-10
        assert abs(orig.upper - back.upper) < 1e-10

    def test_rng_reproducibility(self, tmp_path):
        """After advancing rnd_gen, save+load must produce identical p-values."""
        cp = _train_regressor(ConformalRidgeRegressor, a=1e-3, rnd_state=99)
        # Advance the RNG several steps
        for xi in X_REG[:8]:
            cp.compute_p_value(xi, 1.0)
        path = tmp_path / "ridge_rng.joblib"
        cp.save(str(path))
        loaded = ConformalRidgeRegressor.load(str(path))
        # Identical post-load p-values (same RNG stream position)
        p_orig = cp.compute_p_value(X_TEST_REG, 0.9)
        p_back = loaded.compute_p_value(X_TEST_REG, 0.9)
        assert abs(p_orig - p_back) < 1e-12

    def test_derived_matrix_integrity(self, tmp_path):
        cp = _train_regressor(ConformalRidgeRegressor, a=0.1, rnd_state=0)
        path = tmp_path / "ridge_mat.joblib"
        cp.save(str(path))
        loaded = ConformalRidgeRegressor.load(str(path))
        assert np.allclose(cp.XTXinv, loaded.XTXinv)
        assert np.array_equal(cp.X, loaded.X)
        assert np.array_equal(cp.y, loaded.y)

    def test_params_preserved(self, tmp_path):
        cp = _train_regressor(ConformalRidgeRegressor, a=0.5, studentised=True,
                              recompute_every=10, rnd_state=7)
        path = tmp_path / "ridge_params.joblib"
        cp.save(str(path))
        loaded = ConformalRidgeRegressor.load(str(path))
        assert loaded.a == 0.5
        assert loaded.studentised is True
        assert loaded.recompute_every == 10
        assert loaded.rnd_state == 7

    def test_online_after_load(self, tmp_path):
        """learn_one + predict still works after load."""
        cp = _train_regressor(ConformalRidgeRegressor, a=1e-3, rnd_state=0)
        path = tmp_path / "ridge_online.joblib"
        cp.save(str(path))
        loaded = ConformalRidgeRegressor.load(str(path))
        loaded.learn_one(np.array([0.1, 0.2]), 0.5)
        result = loaded.predict(X_TEST_REG)
        assert np.isfinite(result.lower)
        assert np.isfinite(result.upper)


class TestKNNRegressorRoundTrip:
    def test_predict_equality(self, tmp_path):
        cp = _train_regressor(ConformalNearestNeighboursRegressor, k=3, rnd_state=0)
        path = tmp_path / "knn_reg.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursRegressor.load(str(path))
        orig = cp.predict(X_TEST_REG)
        back = loaded.predict(X_TEST_REG)
        assert abs(orig.lower - back.lower) < 1e-10
        assert abs(orig.upper - back.upper) < 1e-10

    def test_distance_matrix_integrity(self, tmp_path):
        cp = _train_regressor(ConformalNearestNeighboursRegressor, k=2, rnd_state=1)
        path = tmp_path / "knn_reg_D.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursRegressor.load(str(path))
        assert np.allclose(cp.D, loaded.D)

    def test_default_distance_round_trips(self, tmp_path):
        """distance_func=None should survive round-trip as None (uses default)."""
        cp = _train_regressor(ConformalNearestNeighboursRegressor)
        path = tmp_path / "knn_default_dist.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursRegressor.load(str(path))
        # Smoke: prediction works
        loaded.predict(X_TEST_REG)

    def test_custom_registered_distance_func(self, tmp_path):
        @register_callable("test_custom_dist_reg")
        def my_dist(X, y=None):
            from scipy.spatial.distance import cdist, pdist, squareform
            X = np.atleast_2d(X)
            if y is None:
                return squareform(pdist(X, "euclidean"))
            return cdist(X, np.atleast_2d(y), "euclidean")

        cp = _train_regressor(ConformalNearestNeighboursRegressor,
                              distance_func=my_dist, rnd_state=5)
        path = tmp_path / "knn_custom_dist.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursRegressor.load(str(path))
        orig = cp.predict(X_TEST_REG)
        back = loaded.predict(X_TEST_REG)
        assert abs(orig.lower - back.lower) < 1e-10

    def test_lambda_raises_serialization_error(self, tmp_path):
        cp = _train_regressor(ConformalNearestNeighboursRegressor,
                              distance_func=lambda X, y=None: np.zeros((X.shape[0], X.shape[0]) if y is None else (X.shape[0], np.atleast_2d(y).shape[0])))
        path = tmp_path / "knn_lambda.joblib"
        with pytest.raises(SerializationError, match="lambda|closure|Cannot serialize"):
            cp.save(str(path))

    def test_rng_reproducibility(self, tmp_path):
        cp = _train_regressor(ConformalNearestNeighboursRegressor, k=2, rnd_state=123)
        for xi in X_REG[:5]:
            cp.compute_p_value(xi, 1.0)
        path = tmp_path / "knn_rng.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursRegressor.load(str(path))
        p1 = cp.compute_p_value(X_TEST_REG, 0.5)
        p2 = loaded.compute_p_value(X_TEST_REG, 0.5)
        assert abs(p1 - p2) < 1e-12


class TestLassoRegressorRoundTrip:
    def test_predict_equality(self, tmp_path):
        cp = _train_regressor(ConformalLassoRegressor, lam=0.05, rnd_state=3)
        path = tmp_path / "lasso.joblib"
        cp.save(str(path))
        loaded = ConformalLassoRegressor.load(str(path))
        orig = cp.predict(X_TEST_REG)
        back = loaded.predict(X_TEST_REG)
        assert abs(orig.lower - back.lower) < 1e-8
        assert abs(orig.upper - back.upper) < 1e-8

    def test_beta_preserved(self, tmp_path):
        cp = _train_regressor(ConformalLassoRegressor, lam=0.1)
        path = tmp_path / "lasso_beta.joblib"
        cp.save(str(path))
        loaded = ConformalLassoRegressor.load(str(path))
        assert np.allclose(cp.beta, loaded.beta)


class TestKernelRidgeRegressorRoundTrip:
    def test_predict_equality(self, tmp_path):
        kernel = GaussianKernel(sigma=1.0)
        cp = _train_regressor(KernelConformalRidgeRegressor, kernel=kernel,
                              a=1e-3, rnd_state=77)
        path = tmp_path / "kridge.joblib"
        cp.save(str(path))
        loaded = KernelConformalRidgeRegressor.load(str(path))
        orig = cp.predict(X_TEST_REG)
        back = loaded.predict(X_TEST_REG)
        assert abs(orig.lower - back.lower) < 1e-8
        assert abs(orig.upper - back.upper) < 1e-8

    def test_gram_matrix_preserved(self, tmp_path):
        kernel = GaussianKernel(sigma=0.5)
        cp = _train_regressor(KernelConformalRidgeRegressor, kernel=kernel, a=1e-2)
        path = tmp_path / "kridge_K.joblib"
        cp.save(str(path))
        loaded = KernelConformalRidgeRegressor.load(str(path))
        assert np.allclose(cp.K, loaded.K)
        assert np.allclose(cp.Kinv, loaded.Kinv)

    def test_rng_reproducibility(self, tmp_path):
        kernel = GaussianKernel(sigma=1.0)
        cp = _train_regressor(KernelConformalRidgeRegressor, kernel=kernel,
                              a=1e-3, rnd_state=55)
        for xi in X_REG[:5]:
            cp.compute_p_value(xi, 1.0)
        path = tmp_path / "kridge_rng.joblib"
        cp.save(str(path))
        loaded = KernelConformalRidgeRegressor.load(str(path))
        p1 = cp.compute_p_value(X_TEST_REG, 0.7)
        p2 = loaded.compute_p_value(X_TEST_REG, 0.7)
        assert abs(p1 - p2) < 1e-12


# ---------------------------------------------------------------------------
# Functional round-trip — CPS
# ---------------------------------------------------------------------------

class TestRidgePredictionMachineRoundTrip:
    def test_cpd_equality(self, tmp_path):
        cps = _train_regressor(RidgePredictionMachine, a=1e-3)
        path = tmp_path / "rpm.joblib"
        cps.save(str(path))
        loaded = RidgePredictionMachine.load(str(path))
        cpd_orig = cps.predict_cpd(X_TEST_REG)
        cpd_back = loaded.predict_cpd(X_TEST_REG)
        assert np.allclose(cpd_orig.C, cpd_back.C)

    def test_matrix_integrity(self, tmp_path):
        cps = _train_regressor(RidgePredictionMachine, a=0.5)
        path = tmp_path / "rpm_mat.joblib"
        cps.save(str(path))
        loaded = RidgePredictionMachine.load(str(path))
        assert np.allclose(cps.XTXinv, loaded.XTXinv)
        assert np.array_equal(cps.X, loaded.X)


class TestKernelRidgePredictionMachineRoundTrip:
    def test_cpd_equality(self, tmp_path):
        kernel = GaussianKernel(sigma=1.0)
        cps = _train_regressor(KernelRidgePredictionMachine, kernel=kernel, a=1e-3)
        path = tmp_path / "krpm.joblib"
        cps.save(str(path))
        loaded = KernelRidgePredictionMachine.load(str(path))
        cpd_orig = cps.predict_cpd(X_TEST_REG)
        cpd_back = loaded.predict_cpd(X_TEST_REG)
        assert np.allclose(cpd_orig.C, cpd_back.C)

    def test_state_integrity(self, tmp_path):
        kernel = GaussianKernel(sigma=0.5)
        cps = _train_regressor(KernelRidgePredictionMachine, kernel=kernel, a=1e-2)
        path = tmp_path / "krpm_state.joblib"
        cps.save(str(path))
        loaded = KernelRidgePredictionMachine.load(str(path))
        assert np.allclose(cps.K, loaded.K)
        assert np.allclose(cps.Kinv, loaded.Kinv)
        assert np.allclose(cps.h_diag, loaded.h_diag)
        assert np.allclose(cps.Hy, loaded.Hy)


# ---------------------------------------------------------------------------
# Functional round-trip — classifiers
# ---------------------------------------------------------------------------

class TestKNNClassifierRoundTrip:
    def test_predict_set_equality(self, tmp_path):
        cp = _train_classifier(ConformalNearestNeighboursClassifier,
                               k=3, label_space=[-1, 1], rnd_state=0)
        path = tmp_path / "knn_clf.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursClassifier.load(str(path))
        orig = cp.predict(X_TEST_CLF)
        back = loaded.predict(X_TEST_CLF)
        assert set(orig.elements.tolist()) == set(back.elements.tolist())

    def test_distance_matrix_preserved(self, tmp_path):
        cp = _train_classifier(ConformalNearestNeighboursClassifier, k=2, rnd_state=1)
        path = tmp_path / "knn_clf_D.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursClassifier.load(str(path))
        assert np.allclose(cp.D, loaded.D)

    def test_label_indices_preserved(self, tmp_path):
        cp = _train_classifier(ConformalNearestNeighboursClassifier, k=1, rnd_state=2)
        path = tmp_path / "knn_clf_labels.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursClassifier.load(str(path))
        for lbl in cp._label_indices:
            assert np.array_equal(cp._label_indices[lbl], loaded._label_indices[lbl])

    def test_rng_reproducibility(self, tmp_path):
        cp = _train_classifier(ConformalNearestNeighboursClassifier,
                               k=1, label_space=[-1, 1], rnd_state=77)
        # Advance RNG
        for xi in X_CLF[:5]:
            cp.predict(xi)
        path = tmp_path / "knn_clf_rng.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursClassifier.load(str(path))
        _, pv1 = cp.predict(X_TEST_CLF, return_p_values=True)
        _, pv2 = loaded.predict(X_TEST_CLF, return_p_values=True)
        for lbl in pv1:
            assert abs(pv1[lbl] - pv2[lbl]) < 1e-12

    def test_custom_registered_distance_func(self, tmp_path):
        @register_callable("test_clf_custom_dist")
        def clf_dist(X, y=None):
            from scipy.spatial.distance import cdist, pdist, squareform
            X = np.atleast_2d(X)
            if y is None:
                return squareform(pdist(X))
            return cdist(X, np.atleast_2d(y))

        cp = _train_classifier(ConformalNearestNeighboursClassifier,
                               k=2, distance_func=clf_dist, rnd_state=9)
        path = tmp_path / "knn_clf_custom.joblib"
        cp.save(str(path))
        loaded = ConformalNearestNeighboursClassifier.load(str(path))
        orig = cp.predict(X_TEST_CLF)
        back = loaded.predict(X_TEST_CLF)
        assert set(orig.elements.tolist()) == set(back.elements.tolist())


class TestSVMClassifierRoundTrip:
    def test_predict_set_equality(self, tmp_path):
        cp = _train_classifier(ConformalSupportVectorMachine,
                               kernel="rbf", sigma=1.0, C=10.0, rnd_state=0)
        path = tmp_path / "svm.joblib"
        cp.save(str(path))
        loaded = ConformalSupportVectorMachine.load(str(path))
        orig = cp.predict(X_TEST_CLF)
        back = loaded.predict(X_TEST_CLF)
        assert set(orig.elements.tolist()) == set(back.elements.tolist())

    def test_gram_matrix_preserved(self, tmp_path):
        cp = _train_classifier(ConformalSupportVectorMachine, kernel="rbf", sigma=1.0)
        path = tmp_path / "svm_K.joblib"
        cp.save(str(path))
        loaded = ConformalSupportVectorMachine.load(str(path))
        assert np.allclose(cp.K, loaded.K)

    def test_kernel_object_round_trip(self, tmp_path):
        kernel = GaussianKernel(sigma=1.5)
        cp = _train_classifier(ConformalSupportVectorMachine,
                               kernel=kernel, C=5.0, rnd_state=3)
        path = tmp_path / "svm_kernel_obj.joblib"
        cp.save(str(path))
        loaded = ConformalSupportVectorMachine.load(str(path))
        orig = cp.predict(X_TEST_CLF)
        back = loaded.predict(X_TEST_CLF)
        assert set(orig.elements.tolist()) == set(back.elements.tolist())

    def test_rng_reproducibility(self, tmp_path):
        cp = _train_classifier(ConformalSupportVectorMachine,
                               kernel="rbf", sigma=1.0, C=10.0, rnd_state=42)
        for xi in X_CLF[:5]:
            cp.predict(xi)
        path = tmp_path / "svm_rng.joblib"
        cp.save(str(path))
        loaded = ConformalSupportVectorMachine.load(str(path))
        _, pv1 = cp.predict(X_TEST_CLF, return_p_values=True)
        _, pv2 = loaded.predict(X_TEST_CLF, return_p_values=True)
        for lbl in pv1:
            assert abs(pv1[lbl] - pv2[lbl]) < 1e-12


# ---------------------------------------------------------------------------
# Error / safety cases
# ---------------------------------------------------------------------------

class TestVersionAndClassChecks:
    def test_class_mismatch_raises(self, tmp_path):
        cp = _train_regressor(ConformalRidgeRegressor, a=0.1)
        path = tmp_path / "wrong_class.joblib"
        cp.save(str(path))
        with pytest.raises(SerializationError, match="Class mismatch"):
            ConformalNearestNeighboursRegressor.load(str(path))

    def test_format_version_too_new_raises(self, tmp_path):
        import joblib
        cp = _train_regressor(ConformalRidgeRegressor)
        path = tmp_path / "future.joblib"
        cp.save(str(path))
        env = joblib.load(str(path))
        env["format_version"] = 999
        joblib.dump(env, str(path))
        with pytest.raises(SerializationError, match="format_version"):
            ConformalRidgeRegressor.load(str(path))

    def test_library_version_mismatch_warns(self, tmp_path):
        import joblib
        cp = _train_regressor(ConformalRidgeRegressor)
        path = tmp_path / "old_lib.joblib"
        cp.save(str(path))
        env = joblib.load(str(path))
        env["library_version"] = "0.0.0"
        joblib.dump(env, str(path))
        with pytest.warns(UserWarning, match="0.0.0"):
            ConformalRidgeRegressor.load(str(path))

    def test_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "corrupt.joblib"
        path.write_bytes(b"not a joblib file")
        with pytest.raises(SerializationError):
            ConformalRidgeRegressor.load(str(path))

    def test_lambda_kernel_raises(self, tmp_path):
        cp = _train_regressor(KernelConformalRidgeRegressor,
                              kernel=lambda X, Y=None: X @ X.T if Y is None else X @ Y.T)
        path = tmp_path / "lambda_kernel.joblib"
        with pytest.raises(SerializationError):
            cp.save(str(path))

    def test_unregistered_callable_raises_helpful_message(self, tmp_path):
        def _unregistered(X, y=None):
            from scipy.spatial.distance import pdist, squareform
            return squareform(pdist(X)) if y is None else np.zeros((X.shape[0], 1))
        # Make it non-picklable by wrapping in a closure
        outer = _unregistered

        def inner(X, y=None):
            return outer(X, y)  # closure over outer

        cp = _train_regressor(ConformalNearestNeighboursRegressor,
                              distance_func=inner)
        path = tmp_path / "closure.joblib"
        with pytest.raises(SerializationError) as exc_info:
            cp.save(str(path))
        assert "register_callable" in str(exc_info.value) or "lambda" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------

class TestCallableRegistry:
    def test_registered_callable_round_trips(self, tmp_path):
        @register_callable("test_registry_func_v1")
        def my_kernel(X, Y=None):
            X = np.atleast_2d(X)
            if Y is None:
                return X @ X.T
            return X @ np.atleast_2d(Y).T

        cp = _train_regressor(KernelConformalRidgeRegressor, kernel=my_kernel, a=1e-3)
        path = tmp_path / "registry_kernel.joblib"
        cp.save(str(path))
        loaded = KernelConformalRidgeRegressor.load(str(path))
        orig = cp.predict(X_TEST_REG)
        back = loaded.predict(X_TEST_REG)
        assert abs(orig.lower - back.lower) < 1e-8

    def test_unregistered_lookup_raises(self, tmp_path):
        import joblib
        cp = _train_regressor(ConformalRidgeRegressor)
        path = tmp_path / "missing_reg.joblib"
        cp.save(str(path))
        env = joblib.load(str(path))
        # Inject a fake registry token
        env["params"]["fake_callable"] = {"__type__": "registry", "name": "__nonexistent__"}
        # We can't test via load() directly without modifying the class, but we can
        # test from_token directly
        from online_cp._serialization import from_token
        with pytest.raises(SerializationError, match="not found in the registry"):
            from_token({"__type__": "registry", "name": "__nonexistent__"})


# ---------------------------------------------------------------------------
# Symmetry: loaded model is still order-invariant
# ---------------------------------------------------------------------------

class TestSymmetryAfterLoad:
    def test_ridge_permutation_invariant(self, tmp_path):
        """Permuting training data before save → same predictions after load."""
        rng = np.random.default_rng(0)
        perm = rng.permutation(N_TRAIN)
        X_perm, y_perm = X_REG[perm], Y_REG[perm]

        cp1 = ConformalRidgeRegressor(a=1e-3)
        cp1.learn_initial_training_set(X_REG, Y_REG)
        path1 = tmp_path / "sym1.joblib"
        cp1.save(str(path1))
        loaded1 = ConformalRidgeRegressor.load(str(path1))

        cp2 = ConformalRidgeRegressor(a=1e-3)
        cp2.learn_initial_training_set(X_perm, y_perm)
        path2 = tmp_path / "sym2.joblib"
        cp2.save(str(path2))
        loaded2 = ConformalRidgeRegressor.load(str(path2))

        # p-values (smoothed=False to avoid RNG dependence)
        p1 = loaded1.compute_p_value(X_TEST_REG, 0.7, smoothed=False)
        p2 = loaded2.compute_p_value(X_TEST_REG, 0.7, smoothed=False)
        assert abs(p1 - p2) < 1e-10


# ---------------------------------------------------------------------------
# Untrained model save/load
# ---------------------------------------------------------------------------

class TestUntrainedRoundTrip:
    def test_untrained_ridge_saves_and_loads(self, tmp_path):
        cp = ConformalRidgeRegressor(a=0.5, rnd_state=1)
        path = tmp_path / "untrained.joblib"
        cp.save(str(path))
        loaded = ConformalRidgeRegressor.load(str(path))
        # Should return (-inf, inf) on predict without training
        result = loaded.predict(X_TEST_REG)
        assert result.lower == -np.inf
        assert result.upper == np.inf


# ---------------------------------------------------------------------------
# Phase B — Venn predictors
# ---------------------------------------------------------------------------

class TestVennAbersRoundTrip:
    def test_ridge_scorer_round_trip(self, tmp_path):
        from online_cp.venn import VennAbersPredictor
        va = VennAbersPredictor(scorer="ridge", a=1e-3, label_space=[-1, 1])
        va.learn_initial_training_set(X_CLF, Y_CLF)
        path = tmp_path / "va_ridge.joblib"
        va.save(str(path))
        loaded = VennAbersPredictor.load(str(path))
        orig = va.predict(X_TEST_CLF)
        new = loaded.predict(X_TEST_CLF)
        np.testing.assert_allclose(orig.p0, new.p0)
        np.testing.assert_allclose(orig.p1, new.p1)

    def test_knn_scorer_round_trip(self, tmp_path):
        from online_cp.venn import VennAbersPredictor
        va = VennAbersPredictor(scorer="knn", k=5, label_space=[-1, 1])
        va.learn_initial_training_set(X_CLF, Y_CLF)
        path = tmp_path / "va_knn.joblib"
        va.save(str(path))
        loaded = VennAbersPredictor.load(str(path))
        orig = va.predict(X_TEST_CLF)
        new = loaded.predict(X_TEST_CLF)
        np.testing.assert_allclose(orig.p0, new.p0)
        np.testing.assert_allclose(orig.p1, new.p1)

    def test_svm_scorer_round_trip(self, tmp_path):
        from online_cp.venn import VennAbersPredictor
        va = VennAbersPredictor(scorer="svm", C=1.0, label_space=[-1, 1])
        va.learn_initial_training_set(X_CLF, Y_CLF)
        path = tmp_path / "va_svm.joblib"
        va.save(str(path))
        loaded = VennAbersPredictor.load(str(path))
        orig = va.predict(X_TEST_CLF)
        new = loaded.predict(X_TEST_CLF)
        np.testing.assert_allclose(orig.p0, new.p0)
        np.testing.assert_allclose(orig.p1, new.p1)


class TestNearestNeighboursVennRoundTrip:
    def test_round_trip(self, tmp_path):
        from online_cp.venn import NearestNeighboursVennPredictor
        va = NearestNeighboursVennPredictor(k=5, label_space=[-1, 1])
        va.learn_initial_training_set(X_CLF, Y_CLF)
        path = tmp_path / "nn_venn.joblib"
        va.save(str(path))
        loaded = NearestNeighboursVennPredictor.load(str(path))
        orig = va.predict(X_TEST_CLF)
        new = loaded.predict(X_TEST_CLF)
        np.testing.assert_allclose(orig.p0, new.p0)
        np.testing.assert_allclose(orig.p1, new.p1)


# ---------------------------------------------------------------------------
# Phase B — remaining CPS classes
# ---------------------------------------------------------------------------

class TestNearestNeighboursPredictionMachineRoundTrip:
    def test_round_trip(self, tmp_path):
        from online_cp.CPS import NearestNeighboursPredictionMachine
        m = NearestNeighboursPredictionMachine(k=5)
        m.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "nn_cps.joblib"
        m.save(str(path))
        loaded = NearestNeighboursPredictionMachine.load(str(path))
        orig = m.predict_cpd(X_TEST_REG)
        new = loaded.predict_cpd(X_TEST_REG)
        np.testing.assert_allclose(orig.Y, new.Y)
        np.testing.assert_allclose(orig.L, new.L)
        np.testing.assert_allclose(orig.U, new.U)


class TestDempsterHillRoundTrip:
    def test_round_trip(self, tmp_path):
        from online_cp.CPS import DempsterHillConformalPredictiveSystem
        m = DempsterHillConformalPredictiveSystem()
        m.learn_initial_training_set(Y_REG)
        path = tmp_path / "dempster.joblib"
        m.save(str(path))
        loaded = DempsterHillConformalPredictiveSystem.load(str(path))
        orig = m.predict_cpd()
        new = loaded.predict_cpd()
        np.testing.assert_allclose(orig.Y, new.Y)
        np.testing.assert_allclose(orig.L, new.L)
        np.testing.assert_allclose(orig.U, new.U)


# ---------------------------------------------------------------------------
# Phase B — Mondrian conformal wrappers
# ---------------------------------------------------------------------------

def _make_category_fn(n_bins: int = 2):
    """Module-level function (not lambda) needed for pickling."""
    # Return a module-level function-like object via a registered callable
    return lambda x: "low" if x[0] < 0 else "high"


@register_callable("_test_mondrian_category_reg")
def _mondrian_cat_reg(x):
    return "low" if x[0] < 0 else "high"


@register_callable("_test_mondrian_category_clf")
def _mondrian_cat_clf(x):
    return "low" if x[0] < 0 else "high"


class TestMondrianConformalRegressorRoundTrip:
    def test_round_trip(self, tmp_path):
        from online_cp.mondrian import MondrianConformalRegressor
        base = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        m = MondrianConformalRegressor(base, _mondrian_cat_reg)
        m.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "mondrian_reg.joblib"
        m.save(str(path))
        loaded = MondrianConformalRegressor.load(str(path))
        orig = m.predict(X_TEST_REG, epsilon=0.1)
        new = loaded.predict(X_TEST_REG, epsilon=0.1)
        assert abs(orig.lower - new.lower) < 1e-10
        assert abs(orig.upper - new.upper) < 1e-10


class TestMondrianConformalClassifierRoundTrip:
    def test_label_category_fn(self, tmp_path):
        from online_cp.mondrian import MondrianConformalClassifier
        from online_cp.classifiers import ConformalNearestNeighboursClassifier
        base = ConformalNearestNeighboursClassifier(k=5, label_space=[-1, 1])
        m = MondrianConformalClassifier(base, category_fn="label")
        m.learn_initial_training_set(X_CLF, Y_CLF)
        path = tmp_path / "mondrian_clf_label.joblib"
        m.save(str(path))
        loaded = MondrianConformalClassifier.load(str(path))
        orig = m.predict(X_TEST_CLF, epsilon=0.1)
        new = loaded.predict(X_TEST_CLF, epsilon=0.1)
        np.testing.assert_array_equal(orig.elements, new.elements)

    def test_callable_category_fn(self, tmp_path):
        from online_cp.mondrian import MondrianConformalClassifier
        from online_cp.classifiers import ConformalNearestNeighboursClassifier
        base = ConformalNearestNeighboursClassifier(k=5, label_space=[-1, 1])
        m = MondrianConformalClassifier(base, category_fn=_mondrian_cat_clf)
        m.learn_initial_training_set(X_CLF, Y_CLF)
        path = tmp_path / "mondrian_clf_fn.joblib"
        m.save(str(path))
        loaded = MondrianConformalClassifier.load(str(path))
        orig = m.predict(X_TEST_CLF, epsilon=0.1)
        new = loaded.predict(X_TEST_CLF, epsilon=0.1)
        np.testing.assert_array_equal(orig.elements, new.elements)


# ---------------------------------------------------------------------------
# Phase B — ConformalPredictiveDecisionMaker
# ---------------------------------------------------------------------------

@register_callable("_test_decision_utility")
def _test_utility_fn(x, y, d):
    """Simple utility: reward correct side."""
    return float(y * d)


class TestDecisionMakerRoundTrip:
    def test_round_trip(self, tmp_path):
        from online_cp.decision import ConformalPredictiveDecisionMaker, UtilityFunction
        from online_cp.CPS import RidgePredictionMachine
        utility = UtilityFunction(fn=_test_utility_fn, decisions=[-1.0, 1.0])
        dm = ConformalPredictiveDecisionMaker(utility, RidgePredictionMachine)
        dm.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "decision.joblib"
        dm.save(str(path))
        loaded = ConformalPredictiveDecisionMaker.load(str(path))
        orig = dm.predict(X_TEST_REG)
        new = loaded.predict(X_TEST_REG)
        assert orig == new

    def test_utility_fn_preserved(self, tmp_path):
        from online_cp.decision import ConformalPredictiveDecisionMaker, UtilityFunction
        from online_cp.CPS import RidgePredictionMachine
        utility = UtilityFunction(fn=_test_utility_fn, decisions=[-1.0, 1.0])
        dm = ConformalPredictiveDecisionMaker(utility, RidgePredictionMachine)
        path = tmp_path / "decision_fn.joblib"
        dm.save(str(path))
        loaded = ConformalPredictiveDecisionMaker.load(str(path))
        assert loaded.utility.fn is _test_utility_fn


# ---------------------------------------------------------------------------
# Phase B — Pipeline
# ---------------------------------------------------------------------------

class TestPipelineRoundTrip:
    def test_standard_scaler_round_trip(self, tmp_path):
        from online_cp import Pipeline
        from online_cp.preprocessing import StandardScaler
        scaler = StandardScaler(mode="frozen")
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(scaler, regressor)
        pipe.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "pipeline_scaler.joblib"
        pipe.save(str(path))
        loaded = Pipeline.load(str(path))
        orig = pipe.predict(X_TEST_REG, epsilon=0.1)
        new = loaded.predict(X_TEST_REG, epsilon=0.1)
        assert abs(orig.lower - new.lower) < 1e-10
        assert abs(orig.upper - new.upper) < 1e-10

    def test_estimator_state_preserved(self, tmp_path):
        from online_cp import Pipeline
        from online_cp.preprocessing import StandardScaler
        scaler = StandardScaler(mode="frozen")
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(scaler, regressor)
        pipe.learn_initial_training_set(X_REG, Y_REG)
        # Advance with learn_one
        pipe.learn_one(X_TEST_REG, 0.5)
        path = tmp_path / "pipeline_adv.joblib"
        pipe.save(str(path))
        loaded = Pipeline.load(str(path))
        orig = pipe.predict(X_TEST_REG, epsilon=0.1)
        new = loaded.predict(X_TEST_REG, epsilon=0.1)
        assert abs(orig.lower - new.lower) < 1e-10
        assert abs(orig.upper - new.upper) < 1e-10

    def test_bag_mode_raw_data_preserved(self, tmp_path):
        from online_cp import Pipeline
        from online_cp.preprocessing import StandardScaler
        scaler = StandardScaler(mode="bag")
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(scaler, regressor)
        pipe.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "pipeline_bag.joblib"
        pipe.save(str(path))
        loaded = Pipeline.load(str(path))
        assert loaded._X_raw is not None
        np.testing.assert_array_equal(loaded._X_raw, pipe._X_raw)
        np.testing.assert_array_equal(loaded._y_raw, pipe._y_raw)
        orig = pipe.predict(X_TEST_REG, epsilon=0.1)
        new = loaded.predict(X_TEST_REG, epsilon=0.1)
        assert abs(orig.lower - new.lower) < 1e-10
        assert abs(orig.upper - new.upper) < 1e-10

    def test_class_mismatch_raises(self, tmp_path):
        from online_cp import Pipeline
        from online_cp.preprocessing import StandardScaler
        scaler = StandardScaler(mode="frozen")
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(scaler, regressor)
        pipe.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "pipeline_mismatch.joblib"
        pipe.save(str(path))
        with pytest.raises(SerializationError, match="Class mismatch"):
            ConformalRidgeRegressor.load(str(path))


# ---------------------------------------------------------------------------
# Phase B — FuncTransformer inside a Pipeline
# ---------------------------------------------------------------------------

def _shift_features(x):
    """Module-level (picklable) transform: x -> x + 1.0."""
    return x + 1.0


class TestFuncTransformerRoundTrip:
    def test_numpy_ufunc_round_trip(self, tmp_path):
        from online_cp import Pipeline, FuncTransformer
        # log1p requires non-negative inputs.
        X = np.abs(X_REG) + 0.1
        ft = FuncTransformer(np.log1p)
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(ft, regressor)
        pipe.learn_initial_training_set(X, Y_REG)
        path = tmp_path / "pipeline_func.joblib"
        pipe.save(str(path))
        loaded = Pipeline.load(str(path))
        x_test = np.abs(X_TEST_REG) + 0.1
        orig = pipe.predict(x_test, epsilon=0.1)
        new = loaded.predict(x_test, epsilon=0.1)
        assert abs(orig.lower - new.lower) < 1e-10
        assert abs(orig.upper - new.upper) < 1e-10

    def test_named_function_round_trip(self, tmp_path):
        from online_cp import Pipeline, FuncTransformer
        ft = FuncTransformer(_shift_features)
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(ft, regressor)
        pipe.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "pipeline_named_func.joblib"
        pipe.save(str(path))
        loaded = Pipeline.load(str(path))
        assert isinstance(loaded.transformers[0], FuncTransformer)
        assert loaded.transformers[0].fn is _shift_features
        orig = pipe.predict(X_TEST_REG, epsilon=0.1)
        new = loaded.predict(X_TEST_REG, epsilon=0.1)
        assert abs(orig.lower - new.lower) < 1e-10
        assert abs(orig.upper - new.upper) < 1e-10

    def test_lambda_raises_serialization_error(self, tmp_path):
        from online_cp import Pipeline, FuncTransformer
        ft = FuncTransformer(lambda x: x + 1.0)
        regressor = ConformalRidgeRegressor(a=1e-3, rnd_state=42)
        pipe = Pipeline(ft, regressor)
        pipe.learn_initial_training_set(X_REG, Y_REG)
        path = tmp_path / "pipeline_lambda.joblib"
        with pytest.raises(SerializationError):
            pipe.save(str(path))

