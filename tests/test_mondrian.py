"""Tests for Mondrian conformal prediction (correct: single pooled model, filtered calibration)."""

import numpy as np
import pytest

from online_cp import ConformalRidgeRegressor, KernelConformalRidgeRegressor
from online_cp.kernels import GaussianKernel
from online_cp.mondrian import MondrianConformalClassifier, MondrianConformalRegressor


def _category_fn(x):
    """Assign category based on sign of first feature."""
    return "pos" if x[0] > 0 else "neg"


@pytest.fixture
def ridge_data():
    rng = np.random.default_rng(42)
    n_train, n_test, d = 60, 200, 3
    X = rng.standard_normal((n_train + n_test, d))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.5, n_train + n_test)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class TestMondrianConformalRegressorRidge:
    def test_pooled_model_contains_all_data(self, ridge_data):
        X_train, y_train, _, _ = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        # The pooled model must contain ALL training data
        assert wrapper.base_model.X.shape[0] == X_train.shape[0]

    def test_categories_recorded(self, ridge_data):
        X_train, y_train, _, _ = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        assert len(wrapper.categories_) == X_train.shape[0]
        assert wrapper.categories == {"pos", "neg"}

    def test_predict_returns_interval(self, ridge_data):
        X_train, y_train, X_test, _ = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        interval = wrapper.predict(X_test[0], epsilon=0.1)
        assert interval.lower < interval.upper
        assert np.isfinite(interval.lower)
        assert np.isfinite(interval.upper)

    def test_group_conditional_coverage(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        epsilon = 0.1
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)

        coverage = {"pos": [], "neg": []}
        for x_i, y_i in zip(X_test, y_test):
            interval = wrapper.predict(x_i, epsilon=epsilon)
            cat = _category_fn(x_i)
            coverage[cat].append(y_i in interval)
            wrapper.learn_one(x_i, y_i)

        # Group-conditional coverage should be approximately >= 1 - epsilon
        for cat in ["pos", "neg"]:
            cov = np.mean(coverage[cat])
            # Allow some slack for finite-sample effects
            assert cov >= 1 - epsilon - 0.1, (
                f"Category '{cat}' coverage {cov:.3f} too low (expected >= {1-epsilon-0.1:.3f})"
            )

    def test_learn_one_updates_pooled(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        n_before = wrapper.base_model.X.shape[0]
        wrapper.learn_one(X_test[0], y_test[0])
        assert wrapper.base_model.X.shape[0] == n_before + 1
        assert len(wrapper.categories_) == n_before + 1

    def test_compute_p_value(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        p = wrapper.compute_p_value(X_test[0], y_test[0])
        assert 0 <= p <= 1

    def test_multi_epsilon(self, ridge_data):
        X_train, y_train, X_test, _ = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        result = wrapper.predict(X_test[0], epsilon=[0.05, 0.1, 0.2])
        # Wider epsilon -> narrower interval
        assert result[0.05].width() >= result[0.1].width()
        assert result[0.1].width() >= result[0.2].width()


class TestMondrianConformalRegressorKernelRidge:
    def test_predict_and_coverage(self):
        rng = np.random.default_rng(123)
        n_train, n_test, d = 40, 100, 2
        X = rng.standard_normal((n_train + n_test, d))
        y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(n_train + n_test)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        wrapper = MondrianConformalRegressor(
            base_model=KernelConformalRidgeRegressor(a=0.5, kernel=GaussianKernel(sigma=1.0)),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)

        epsilon = 0.1
        covered = 0
        for x_i, y_i in zip(X_test, y_test):
            interval = wrapper.predict(x_i, epsilon=epsilon)
            if y_i in interval:
                covered += 1
            wrapper.learn_one(x_i, y_i)

        cov = covered / n_test
        assert cov >= 1 - epsilon - 0.1


class TestMondrianConformalClassifier:
    def test_knn_coverage(self):
        from online_cp import ConformalNearestNeighboursClassifier

        rng = np.random.default_rng(99)
        n_train, n_test, d = 60, 100, 4
        X = rng.standard_normal((n_train + n_test, d))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=5, label_space=np.array([0, 1])),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)

        epsilon = 0.1
        coverage = {"pos": [], "neg": []}
        for x_i, y_i in zip(X_test, y_test):
            pred_set = wrapper.predict(x_i, epsilon=epsilon)
            cat = _category_fn(x_i)
            coverage[cat].append(y_i in pred_set)
            wrapper.learn_one(x_i, y_i)

        for cat in ["pos", "neg"]:
            if len(coverage[cat]) > 10:
                cov = np.mean(coverage[cat])
                assert cov >= 1 - epsilon - 0.2, (
                    f"KNN category '{cat}' coverage {cov:.3f} too low"
                )

    def test_pooled_model_all_data(self):
        from online_cp import ConformalNearestNeighboursClassifier

        rng = np.random.default_rng(77)
        X = rng.standard_normal((30, 3))
        y = (X[:, 0] > 0).astype(int)

        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=3),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X, y)
        assert wrapper.base_model.X.shape[0] == 30
        assert len(wrapper.categories_) == 30


class TestMondrianTypeChecks:
    def test_regressor_rejects_invalid_model(self):
        with pytest.raises(TypeError, match="base_model must be"):
            MondrianConformalRegressor(
                base_model="not a model",
                category_fn=_category_fn,
            )

    def test_repr(self, ridge_data):
        X_train, y_train, _, _ = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        r = repr(wrapper)
        assert "MondrianConformalRegressor" in r
        assert "ConformalRidgeRegressor" in r
