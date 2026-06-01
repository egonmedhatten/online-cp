"""Tests for Mondrian conformal prediction (correct: single pooled model, filtered calibration)."""

import numpy as np
import pytest

from online_cp import (
    ConformalNearestNeighboursClassifier,
    ConformalRidgeRegressor,
    ConformalSupportVectorMachine,
    KernelConformalRidgeRegressor,
)
from online_cp.kernels import GaussianKernel, LinearKernel
from online_cp.mondrian import MondrianConformalClassifier, MondrianConformalRegressor
from online_cp.regressors import ConformalLassoRegressor


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


# ---------------------------------------------------------------------------
# Lasso regressor tests
# ---------------------------------------------------------------------------


@pytest.fixture
def lasso_data():
    """Sparse linear data suitable for Lasso."""
    rng = np.random.default_rng(55)
    n_train, n_test, d = 50, 80, 10
    X = rng.standard_normal((n_train + n_test, d))
    beta = np.array([3.0, 1.5, 0, 0, 2.0, 0, 0, 0, 0, 0])
    y = X @ beta + rng.normal(0, 0.5, n_train + n_test)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class TestMondrianConformalRegressorLasso:
    def test_predict_returns_interval(self, lasso_data):
        X_train, y_train, X_test, _ = lasso_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalLassoRegressor(lam=0.5),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        interval = wrapper.predict(X_test[0], epsilon=0.2)
        assert interval.lower < interval.upper
        assert np.isfinite(interval.lower)
        assert np.isfinite(interval.upper)

    def test_coverage(self, lasso_data):
        X_train, y_train, X_test, y_test = lasso_data
        epsilon = 0.2
        wrapper = MondrianConformalRegressor(
            base_model=ConformalLassoRegressor(lam=0.5),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)

        coverage = {"pos": [], "neg": []}
        for x_i, y_i in zip(X_test[:40], y_test[:40]):
            interval = wrapper.predict(x_i, epsilon=epsilon)
            cat = _category_fn(x_i)
            coverage[cat].append(y_i in interval)
            wrapper.learn_one(x_i, y_i)

        for cat in ["pos", "neg"]:
            if len(coverage[cat]) > 5:
                cov = np.mean(coverage[cat])
                assert cov >= 1 - epsilon - 0.15, (
                    f"Lasso category '{cat}' coverage {cov:.3f} too low"
                )

    def test_compute_p_value(self, lasso_data):
        X_train, y_train, X_test, y_test = lasso_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalLassoRegressor(lam=0.5),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        p = wrapper.compute_p_value(X_test[0], y_test[0])
        assert 0 <= p <= 1

    def test_learn_one_incremental(self, lasso_data):
        X_train, y_train, X_test, y_test = lasso_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalLassoRegressor(lam=0.5),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        n_before = len(wrapper.categories_)
        wrapper.learn_one(X_test[0], y_test[0])
        assert len(wrapper.categories_) == n_before + 1


# ---------------------------------------------------------------------------
# SVM classifier tests
# ---------------------------------------------------------------------------


@pytest.fixture
def svm_data():
    """Binary classification data for SVM."""
    rng = np.random.default_rng(88)
    n_train, n_test = 50, 60
    X_pos = rng.standard_normal((n_train + n_test, 2)) + [2, 0]
    X_neg = rng.standard_normal((n_train + n_test, 2)) + [-2, 0]
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_train + n_test) + [-1] * (n_train + n_test))
    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    return X[:n_train], y[:n_train], X[n_train : n_train + n_test], y[n_train : n_train + n_test]


class TestMondrianConformalClassifierSVM:
    def test_predict_returns_set(self, svm_data):
        X_train, y_train, X_test, _ = svm_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalSupportVectorMachine(
                kernel=LinearKernel(), C=10.0, label_space=np.array([-1, 1])
            ),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        pred = wrapper.predict(X_test[0], epsilon=0.1)
        # Should return a prediction set that supports `in`
        assert 1 in pred or -1 in pred

    def test_coverage(self, svm_data):
        X_train, y_train, X_test, y_test = svm_data
        epsilon = 0.2
        wrapper = MondrianConformalClassifier(
            base_model=ConformalSupportVectorMachine(
                kernel=LinearKernel(), C=10.0, label_space=np.array([-1, 1])
            ),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)

        covered = 0
        for x_i, y_i in zip(X_test, y_test):
            pred = wrapper.predict(x_i, epsilon=epsilon)
            if y_i in pred:
                covered += 1
            wrapper.learn_one(x_i, y_i)

        cov = covered / len(X_test)
        assert cov >= 1 - epsilon - 0.15, f"SVM coverage {cov:.3f} too low"

    def test_learn_one_incremental(self, svm_data):
        X_train, y_train, X_test, y_test = svm_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalSupportVectorMachine(
                kernel=LinearKernel(), C=10.0, label_space=np.array([-1, 1])
            ),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        n_before = len(wrapper.categories_)
        wrapper.learn_one(X_test[0], y_test[0])
        assert len(wrapper.categories_) == n_before + 1


# ---------------------------------------------------------------------------
# Bounds parameter tests (ridge regressor)
# ---------------------------------------------------------------------------


class TestMondrianBoundsParameter:
    @pytest.fixture
    def trained_ridge(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        return wrapper, X_test, y_test

    def test_bounds_lower(self, trained_ridge):
        wrapper, X_test, _ = trained_ridge
        interval = wrapper.predict(X_test[0], epsilon=0.1, bounds="lower")
        assert np.isfinite(interval.lower)
        assert interval.upper == np.inf

    def test_bounds_upper(self, trained_ridge):
        wrapper, X_test, _ = trained_ridge
        interval = wrapper.predict(X_test[0], epsilon=0.1, bounds="upper")
        assert interval.lower == -np.inf
        assert np.isfinite(interval.upper)

    def test_bounds_both_wider(self, trained_ridge):
        """bounds='both' should produce a finite interval containing the one-sided bounds."""
        wrapper, X_test, _ = trained_ridge
        both = wrapper.predict(X_test[0], epsilon=0.1, bounds="both")
        lower = wrapper.predict(X_test[0], epsilon=0.1, bounds="lower")
        upper = wrapper.predict(X_test[0], epsilon=0.1, bounds="upper")
        # The lower bound from "both" should be <= upper bound from "upper"
        assert both.lower <= upper.upper
        # The upper bound from "both" should be >= lower bound from "lower"
        assert both.upper >= lower.lower


class TestMondrianBoundsKernelRidge:
    def test_bounds_lower(self):
        rng = np.random.default_rng(321)
        X = rng.standard_normal((50, 2))
        y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(50)

        wrapper = MondrianConformalRegressor(
            base_model=KernelConformalRidgeRegressor(a=0.5, kernel=GaussianKernel(sigma=1.0)),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X[:40], y[:40])
        interval = wrapper.predict(X[40], epsilon=0.1, bounds="lower")
        assert np.isfinite(interval.lower)
        assert interval.upper == np.inf

    def test_bounds_upper(self):
        rng = np.random.default_rng(321)
        X = rng.standard_normal((50, 2))
        y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(50)

        wrapper = MondrianConformalRegressor(
            base_model=KernelConformalRidgeRegressor(a=0.5, kernel=GaussianKernel(sigma=1.0)),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X[:40], y[:40])
        interval = wrapper.predict(X[40], epsilon=0.1, bounds="upper")
        assert interval.lower == -np.inf
        assert np.isfinite(interval.upper)


# ---------------------------------------------------------------------------
# P-value tests (smoothed vs unsmoothed)
# ---------------------------------------------------------------------------


class TestMondrianPValues:
    def test_p_value_smoothed(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        p = wrapper.compute_p_value(X_test[0], y_test[0], smoothed=True)
        assert 0 <= p <= 1

    def test_p_value_unsmoothed(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        p = wrapper.compute_p_value(X_test[0], y_test[0], smoothed=False)
        assert 0 <= p <= 1

    def test_p_value_bounds_lower(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        p = wrapper.compute_p_value(X_test[0], y_test[0], bounds="lower")
        assert 0 <= p <= 1

    def test_p_value_bounds_upper(self, ridge_data):
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        p = wrapper.compute_p_value(X_test[0], y_test[0], bounds="upper")
        assert 0 <= p <= 1

    def test_p_values_uniform_distribution(self, ridge_data):
        """P-values should be approximately uniform under null."""
        X_train, y_train, X_test, y_test = ridge_data
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)

        p_vals = []
        for x_i, y_i in zip(X_test[:80], y_test[:80]):
            p = wrapper.compute_p_value(x_i, y_i)
            p_vals.append(p)
            wrapper.learn_one(x_i, y_i)

        p_vals = np.array(p_vals)
        # Under exchangeability, p-values are super-uniform
        # Check that at least some are spread across [0, 1]
        assert np.min(p_vals) < 0.3
        assert np.max(p_vals) > 0.7


# ---------------------------------------------------------------------------
# Classifier return_p_values tests
# ---------------------------------------------------------------------------


class TestMondrianClassifierReturnPValues:
    def test_return_p_values_knn(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(int)

        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=5, label_space=np.array([0, 1])),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X[:50], y[:50])
        result = wrapper.predict(X[50], epsilon=0.1, return_p_values=True)
        # Should be a tuple (pred_set, p_values)
        pred_set, p_values = result
        assert 0 in pred_set or 1 in pred_set
        assert isinstance(p_values, dict)
        for label, pval in p_values.items():
            assert 0 <= pval <= 1

    def test_return_p_values_svm(self, svm_data):
        X_train, y_train, X_test, _ = svm_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalSupportVectorMachine(
                kernel=LinearKernel(), C=10.0, label_space=np.array([-1, 1])
            ),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        result = wrapper.predict(X_test[0], epsilon=0.1, return_p_values=True)
        pred_set, p_values = result
        assert isinstance(p_values, dict)
        for label, pval in p_values.items():
            assert 0 <= pval <= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMondrianEdgeCases:
    def test_single_category(self, ridge_data):
        """All training data in one category should still work."""
        X_train, y_train, X_test, y_test = ridge_data
        # Category function that always returns same category
        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=lambda x: "all",
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        interval = wrapper.predict(X_test[0], epsilon=0.1)
        assert interval.lower < interval.upper

    def test_many_categories(self, ridge_data):
        """Many small categories should produce valid (possibly wide) intervals."""
        X_train, y_train, X_test, _ = ridge_data
        # Category per quartile of first two features
        def fine_cat(x):
            return f"{'p' if x[0] > 0 else 'n'}{'p' if x[1] > 0 else 'n'}"

        wrapper = MondrianConformalRegressor(
            base_model=ConformalRidgeRegressor(a=1.0),
            category_fn=fine_cat,
        )
        wrapper.learn_initial_training_set(X_train, y_train)
        # With 4 categories and 60 training points, ~15 per category
        interval = wrapper.predict(X_test[0], epsilon=0.2)
        assert np.isfinite(interval.lower) or np.isfinite(interval.upper)

    def test_classifier_rejects_invalid_model(self):
        """Classifier with unsupported base_model raises TypeError on predict."""
        # MondrianConformalClassifier doesn't type-check in __init__,
        # but raises TypeError in predict() for unsupported models
        wrapper = MondrianConformalClassifier(
            base_model="not a model",
            category_fn=_category_fn,
        )
        # It will fail when trying to call methods on the invalid model
        with pytest.raises((TypeError, AttributeError)):
            wrapper.learn_initial_training_set(np.zeros((10, 2)), np.zeros(10, dtype=int))

    def test_kernel_ridge_multi_epsilon(self):
        rng = np.random.default_rng(444)
        X = rng.standard_normal((50, 2))
        y = X[:, 0] ** 2 + 0.3 * rng.standard_normal(50)

        wrapper = MondrianConformalRegressor(
            base_model=KernelConformalRidgeRegressor(a=0.5, kernel=GaussianKernel(sigma=1.0)),
            category_fn=_category_fn,
        )
        wrapper.learn_initial_training_set(X[:40], y[:40])
        result = wrapper.predict(X[40], epsilon=[0.05, 0.1, 0.2])
        # Wider epsilon -> narrower interval
        assert result[0.05].width() >= result[0.1].width()
        assert result[0.1].width() >= result[0.2].width()


# ---------------------------------------------------------------------------
# Label-conditional Mondrian classifier tests
# ---------------------------------------------------------------------------


class TestLabelConditionalMondrianClassifier:
    """Tests for label-conditional Mondrian CP (category_fn='label')."""

    @pytest.fixture
    def multiclass_data(self):
        rng = np.random.default_rng(123)
        N = 400
        X = rng.standard_normal((N, 3))
        y = (X[:, 0] + X[:, 1] + 0.5 * rng.standard_normal(N))
        y = np.digitize(y, bins=[-1, 1])  # 3 classes: 0, 1, 2
        return X, y

    def test_label_string_shortcut(self, multiclass_data):
        """category_fn='label' creates a label-conditional predictor."""
        X, y = multiclass_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=3),
            category_fn="label",
        )
        wrapper.learn_initial_training_set(X[:30], y[:30])
        # Categories should be the labels themselves
        assert set(wrapper.categories_) == set(y[:30])

    def test_two_arg_callable(self, multiclass_data):
        """A 2-arg callable (x, y) -> category works as label-aware."""
        X, y = multiclass_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=3),
            category_fn=lambda x, y: y,
        )
        wrapper.learn_initial_training_set(X[:30], y[:30])
        assert set(wrapper.categories_) == set(y[:30])

    def test_one_arg_backward_compat(self, multiclass_data):
        """A 1-arg callable still works (object-conditional)."""
        X, y = multiclass_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=3),
            category_fn=lambda x: "pos" if x[0] > 0 else "neg",
        )
        wrapper.learn_initial_training_set(X[:30], y[:30])
        assert set(wrapper.categories_) == {"pos", "neg"}

    def test_label_conditional_per_class_validity(self, multiclass_data):
        """Label-conditional Mondrian should give valid coverage per class."""
        X, y = multiclass_data
        n_init = 30
        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=5),
            category_fn="label",
        )
        wrapper.learn_initial_training_set(X[:n_init], y[:n_init])

        epsilon = 0.1
        errors = {c: [] for c in np.unique(y)}
        for i in range(n_init, len(y)):
            Gamma = wrapper.predict(X[i], epsilon=epsilon)
            errors[y[i]].append(int(y[i] not in Gamma))
            wrapper.learn_one(X[i], y[i])

        # Each class should have error rate ≤ epsilon + tolerance
        # (statistical tolerance for finite sample)
        for c, errs in errors.items():
            if len(errs) >= 30:  # only check classes with enough samples
                rate = np.mean(errs)
                assert rate <= epsilon + 0.07, (
                    f"Class {c} error rate {rate:.3f} exceeds {epsilon} + tolerance"
                )

    def test_label_conditional_returns_p_values(self, multiclass_data):
        """return_p_values=True works with label-conditional."""
        X, y = multiclass_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=3),
            category_fn="label",
        )
        wrapper.learn_initial_training_set(X[:30], y[:30])
        Gamma, p_values = wrapper.predict(X[30], epsilon=0.1, return_p_values=True)
        assert isinstance(p_values, dict)
        for label in np.unique(y[:30]):
            assert label in p_values
            assert 0 <= p_values[label] <= 1

    def test_label_conditional_learn_one(self, multiclass_data):
        """learn_one correctly records label as category."""
        X, y = multiclass_data
        wrapper = MondrianConformalClassifier(
            base_model=ConformalNearestNeighboursClassifier(k=3),
            category_fn="label",
        )
        wrapper.learn_initial_training_set(X[:10], y[:10])
        n_before = len(wrapper.categories_)
        wrapper.learn_one(X[10], y[10])
        assert len(wrapper.categories_) == n_before + 1
        assert wrapper.categories_[-1] == y[10]

    def test_invalid_category_fn_raises(self):
        """Non-string, non-callable raises TypeError."""
        with pytest.raises(TypeError):
            MondrianConformalClassifier(
                base_model=ConformalNearestNeighboursClassifier(k=3),
                category_fn=42,
            )
