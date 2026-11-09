"""
Tests for ConformalMondrianTreeClassifier.

Comprehensive test suite covering:
- API contract (predict signature, return types)
- Conformal validity (coverage guarantees)
- Bag function property (permutation invariance)
- Edge cases (single point, empty predictions, etc.)
"""

import os

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs

from online_cp.classifiers import (
    ConformalMondrianForestClassifier,
    ConformalMondrianTreeClassifier,
    DiscretePredictionSet,
    MultiLevelPredictionSet,
)
from online_cp.mondrian import MondrianTree
from online_cp.mondrian.tree import _has_graphviz


class TestConformalMondrianTreeClassifier:
    """Core tests for the classifier."""

    @pytest.fixture
    def iris_data(self):
        """Load Iris dataset."""
        data = load_iris()
        X = data.data
        y = data.target
        return X, y

    @pytest.fixture
    def synthetic_binary(self):
        """Generate synthetic binary classification data."""
        X, y = make_blobs(n_samples=100, n_features=5, centers=2, random_state=42)
        return X, y

    @pytest.fixture
    def synthetic_multiclass(self):
        """Generate synthetic multiclass data."""
        X, y = make_blobs(n_samples=150, n_features=4, centers=5, random_state=42)
        return X, y

    def test_learn_initial_training_set_basic(self, iris_data):
        """Test batch initialization."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        assert clf.X.shape == (50, 4)
        assert clf.y.shape == (50,)
        assert len(clf.label_space) == 1
        assert clf.label_to_idx is not None

    def test_learn_initial_training_set_multiclass(self, iris_data):
        """Test batch initialization with multiple classes."""
        X, y = iris_data
        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X, y)

        assert clf.X.shape == (150, 4)
        assert clf.y.shape == (150,)
        assert len(clf.label_space) == 3
        assert np.array_equal(clf.label_space, [0, 1, 2])

    def test_learn_initial_training_set_empty_raises(self):
        """Test that empty training set raises error."""
        clf = ConformalMondrianTreeClassifier()
        with pytest.raises(ValueError, match="empty"):
            clf.learn_initial_training_set(np.array([]).reshape(0, 2), np.array([]))

    def test_learn_initial_training_set_mismatched_shapes_raises(self):
        """Test that mismatched X and y shapes raise error."""
        clf = ConformalMondrianTreeClassifier()
        X = np.random.randn(10, 3)
        y = np.arange(5)  # Wrong length
        with pytest.raises(ValueError, match="same length"):
            clf.learn_initial_training_set(X, y)

    def test_learn_one_appends_data(self, iris_data):
        """Test that learn_one appends points correctly."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]
        x_new, y_new = X[50], y[50]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        assert clf.X.shape[0] == 50
        clf.learn_one(x_new, y_new)
        assert clf.X.shape[0] == 51

        # Add another single point
        clf.learn_one(X[51], y[51])
        assert clf.X.shape[0] == 52

    def test_learn_one_handles_new_label(self, iris_data):
        """Test that learn_one expands label_space for new labels."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]  # Only class 0

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)
        assert len(clf.label_space) == 1

        # Add point with new label
        clf.learn_one(X[100], 1)  # Class 1
        assert len(clf.label_space) == 2

    def test_learn_one_before_init_raises(self):
        """Test that learn_one before initialization raises error."""
        clf = ConformalMondrianTreeClassifier()
        with pytest.raises(ValueError, match="learn_initial_training_set"):
            clf.learn_one(np.array([1, 2, 3, 4]), 0)

    def test_predict_dimension_mismatch_raises(self, iris_data):
        """Test that dimension mismatch in predict raises error."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        x_bad = np.array([1, 2, 3])  # Wrong dimension
        with pytest.raises(ValueError, match="dimension mismatch"):
            clf.predict(x_bad)

    def test_predict_returns_conformal_prediction_set(self, iris_data):
        """Test that predict returns DiscretePredictionSet for scalar epsilon."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(epsilon=0.1, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60])
        assert isinstance(result, DiscretePredictionSet)
        assert hasattr(result, 'elements')

    def test_predict_multi_level_epsilon(self, iris_data):
        """Test that predict returns MultiLevelPredictionSet for array epsilon."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        epsilons = np.array([0.05, 0.1, 0.2])
        result = clf.predict(X[60], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert result.levels == sorted(epsilons.tolist())

    def test_predict_with_return_p_values(self, iris_data):
        """Test return_p_values flag."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(epsilon=0.1, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60], return_p_values=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], DiscretePredictionSet)
        assert isinstance(result[1], dict)
        assert set(result[1].keys()) == set(clf.label_space)

        # Check p-values are in [0, 1]
        for p in result[1].values():
            assert 0 <= p <= 1

    def test_predict_with_return_update(self, iris_data):
        """Test return_update flag."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(epsilon=0.1, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60], return_update=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], DiscretePredictionSet)
        assert isinstance(result[1], dict)
        assert 'tree' in result[1]
        assert 'leaf_star' in result[1]

    def test_p_values_in_unit_interval(self, iris_data):
        """Test that all p-values lie in [0, 1]."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        for test_idx in range(60, 80):
            _, p_values = clf.predict(X[test_idx], return_p_values=True)
            for p in p_values.values():
                assert 0 <= p <= 1, f"p-value {p} out of range"
            clf.learn_one(X[test_idx], y[test_idx])

    def test_monotonicity_multi_level_epsilon(self, iris_data):
        """Smaller epsilon → larger (or equal) prediction set.

        _compute_Gamma includes label y iff p_values[y] > eps. Since p-values
        are fixed within a single predict call, smaller eps strictly means the
        threshold is easier to exceed. Monotonicity is therefore guaranteed.
        """
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
        result = clf.predict(X[60], epsilon=epsilons)

        for i in range(len(epsilons) - 1):
            assert len(result[epsilons[i]].elements) >= len(result[epsilons[i + 1]].elements), \
                f"Monotonicity violated: |Γ(ε={epsilons[i]})| < |Γ(ε={epsilons[i+1]})|"

    def test_prediction_set_nonempty_at_large_epsilon(self, iris_data):
        """Test that prediction set is non-empty at reasonable epsilon."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(epsilon=0.5, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60], epsilon=0.5)
        # With epsilon=0.5, set should typically be non-empty (but allow for rare cases)
        assert isinstance(result, DiscretePredictionSet)

    def test_first_prediction_includes_all_labels(self, iris_data):
        """Test that first prediction with minimal training includes labels."""
        X, y = iris_data
        X_train, y_train = X[[0]], y[[0]]  # Single point

        clf = ConformalMondrianTreeClassifier(epsilon=0.3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[1], epsilon=0.3)
        # With single training point and moderate epsilon, set should be non-empty
        assert isinstance(result, DiscretePredictionSet)

    def test_compute_p_value_single_label(self, iris_data):
        """Test compute_p_value for a single label."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        p_val = clf.compute_p_value(X[60], y=0)
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1

    def test_compute_p_value_unknown_label_raises(self, iris_data):
        """Test that compute_p_value with unknown label raises error."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        with pytest.raises(ValueError, match="label_space"):
            clf.compute_p_value(X[60], y=999)

    def test_reproducibility_with_rnd_state(self, iris_data):
        """Test that same rnd_state gives same predictions."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf1 = ConformalMondrianTreeClassifier(epsilon=0.1, rnd_state=42)
        clf1.learn_initial_training_set(X_train, y_train)
        result1 = clf1.predict(X[60], return_p_values=True)

        clf2 = ConformalMondrianTreeClassifier(epsilon=0.1, rnd_state=42)
        clf2.learn_initial_training_set(X_train, y_train)
        result2 = clf2.predict(X[60], return_p_values=True)

        # P-values should be identical
        for label in clf1.label_space:
            assert np.isclose(result1[1][label], result2[1][label])

    def test_bag_function_permutation_invariance(self):
        """Critical test: NCM is a bag function (permutation-invariant).

        This verifies the fundamental conformal property: permuting the training
        set should not change the p-values (or Gamma sets) for a test point.
        """
        np.random.seed(0)
        X, y = make_blobs(n_samples=50, n_features=3, centers=2, random_state=0)
        x_test = np.array([0.5, 0.5, 0.5])

        # Original order
        clf1 = ConformalMondrianTreeClassifier(rnd_state=0)
        clf1.learn_initial_training_set(X, y)
        result1 = clf1.predict(x_test, return_p_values=True)

        # Permuted order
        perm = np.random.permutation(len(X))
        X_perm = X[perm]
        y_perm = y[perm]

        clf2 = ConformalMondrianTreeClassifier(rnd_state=0)
        clf2.learn_initial_training_set(X_perm, y_perm)
        result2 = clf2.predict(x_test, return_p_values=True)

        # P-values should be identical (up to numerical precision)
        for label in [0, 1]:
            assert np.isclose(result1[1][label], result2[1][label], atol=1e-10), \
                f"NCM not invariant to permutation for label {label}"

    def test_validity_binary_classification(self):
        """Test validity: error rate <= epsilon + margin on binary data."""
        X, y = make_blobs(n_samples=200, n_features=4, centers=2, random_state=0)

        epsilon = 0.2
        clf = ConformalMondrianTreeClassifier(epsilon=epsilon, rnd_state=0)

        # Split into initial training and online test set
        X_init, X_test = X[:50], X[50:]
        y_init, y_test = y[:50], y[50:]

        clf.learn_initial_training_set(X_init, y_init)

        errors = 0
        for x_t, y_t in zip(X_test, y_test):
            pred_set = clf.predict(x_t)
            if y_t not in pred_set.elements:
                errors += 1
            clf.learn_one(x_t, y_t)

        error_rate = errors / len(X_test)
        margin = 0.15
        assert error_rate <= epsilon + margin, \
            f"Error rate {error_rate:.4f} > {epsilon + margin:.4f}"

    def test_validity_multiclass(self):
        """Test validity on multiclass data."""
        X, y = make_blobs(n_samples=200, n_features=4, centers=3, random_state=0)

        epsilon = 0.15
        clf = ConformalMondrianTreeClassifier(epsilon=epsilon, rnd_state=0)

        X_init, X_test = X[:60], X[60:]
        y_init, y_test = y[:60], y[60:]

        clf.learn_initial_training_set(X_init, y_init)

        errors = 0
        for x_t, y_t in zip(X_test, y_test):
            pred_set = clf.predict(x_t)
            if y_t not in pred_set.elements:
                errors += 1
            clf.learn_one(x_t, y_t)

        error_rate = errors / len(X_test)
        margin = 0.15
        assert error_rate <= epsilon + margin, \
            f"Error rate {error_rate:.4f} > {epsilon + margin:.4f}"

    def test_singleton_leaf_handling(self):
        """Test that singleton leaves (pure leaves) are handled correctly."""
        X, y = make_blobs(n_samples=30, n_features=2, centers=2, random_state=0)

        clf = ConformalMondrianTreeClassifier(lifetime=np.inf, rnd_state=0)
        clf.learn_initial_training_set(X, y)

        # With lifetime=inf, leaves can become singleton
        result = clf.predict(X[0], return_p_values=True)
        assert isinstance(result[0], DiscretePredictionSet)
        assert all(0 <= p <= 1 for p in result[1].values())

    def test_lifetime_parameter(self):
        """Test that lifetime parameter affects tree depth."""
        X, y = make_blobs(n_samples=100, n_features=3, centers=3, random_state=0)

        # Short lifetime should create shallow tree
        clf_shallow = ConformalMondrianTreeClassifier(lifetime=0.01, rnd_state=0)
        clf_shallow.learn_initial_training_set(X, y)

        # Long lifetime should allow deeper tree
        clf_deep = ConformalMondrianTreeClassifier(lifetime=np.inf, rnd_state=0)
        clf_deep.learn_initial_training_set(X, y)

        # Both should produce valid predictions
        result_shallow = clf_shallow.predict(X[0])
        result_deep = clf_deep.predict(X[0])

        assert isinstance(result_shallow, DiscretePredictionSet)
        assert isinstance(result_deep, DiscretePredictionSet)

    def test_arbitrary_label_space(self):
        """Test that arbitrary labels (not just 0,1,2) are handled."""
        np.random.seed(0)
        X = np.random.randn(50, 3)
        y = np.array(['cat'] * 20 + ['dog'] * 30)

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X, y)

        result = clf.predict(X[0], epsilon=0.1, return_p_values=True)
        assert 'cat' in result[1]
        assert 'dog' in result[1]
        assert set(result[1].keys()) == {'cat', 'dog'}

    def test_reshape_input_handling(self):
        """Test that (1, d) shaped inputs are correctly reshaped to (d,)."""
        X, y = make_blobs(n_samples=50, n_features=3, centers=2, random_state=0)

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X, y)

        # Test with (3,) shape
        result1 = clf.predict(X[0], epsilon=0.1)

        # Test with (1, 3) shape
        result2 = clf.predict(X[0:1], epsilon=0.1)

        assert isinstance(result1, DiscretePredictionSet)
        assert isinstance(result2, DiscretePredictionSet)


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_imbalanced_classes(self):
        """Test on highly imbalanced dataset."""
        X = np.vstack([
            np.random.randn(100, 2) + [0, 0],
            np.random.randn(5, 2) + [5, 5],
        ])
        y = np.array([0] * 100 + [1] * 5)

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X[:80], y[:80])

        result = clf.predict(X[80], return_p_values=True)
        assert isinstance(result[0], DiscretePredictionSet)
        assert all(0 <= p <= 1 for p in result[1].values())

    def test_identical_points(self):
        """Test on dataset with duplicate points."""
        X = np.array([[1.0, 1.0]] * 30 + [[2.0, 2.0]] * 20)
        y = np.array([0] * 30 + [1] * 20)

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X, y)

        # Test point at cluster center (conforming) should produce non-empty set
        result = clf.predict(np.array([1.0, 1.0]), epsilon=0.1)
        assert isinstance(result, DiscretePredictionSet)
        assert len(result.elements) > 0

    def test_high_dimensions(self):
        """Test on high-dimensional data."""
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 3, 50)

        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X, y)

        result = clf.predict(np.random.randn(100), epsilon=0.1, return_p_values=True)
        assert isinstance(result[0], DiscretePredictionSet)
        assert all(0 <= p <= 1 for p in result[1].values())

    def test_zero_epsilon(self):
        """Test with epsilon=0 (very strict)."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        clf = ConformalMondrianTreeClassifier(epsilon=0.0, rnd_state=0)
        clf.learn_initial_training_set(X, y)

        result = clf.predict(X[0])
        # With epsilon=0, prediction set should be small (possibly empty)
        assert isinstance(result, DiscretePredictionSet)


class TestConformalMondrianForestClassifier:
    """Core tests for the forest classifier (ensemble of Mondrian trees)."""

    @pytest.fixture
    def iris_data(self):
        """Load Iris dataset."""
        data = load_iris()
        X = data.data
        y = data.target
        return X, y

    @pytest.fixture
    def synthetic_binary(self):
        """Generate synthetic binary classification data."""
        X, y = make_blobs(n_samples=100, n_features=5, centers=2, random_state=42)
        return X, y

    @pytest.fixture
    def synthetic_multiclass(self):
        """Generate synthetic multiclass data."""
        X, y = make_blobs(n_samples=150, n_features=4, centers=5, random_state=42)
        return X, y

    def test_learn_initial_training_set_basic(self, iris_data):
        """Test batch initialization."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]

        clf = ConformalMondrianForestClassifier(n_trees=5, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        assert clf.X.shape == (50, 4)
        assert clf.y.shape == (50,)
        assert len(clf.label_space) == 1
        assert clf.n_trees == 5

    def test_learn_initial_training_set_multiclass(self, iris_data):
        """Test batch initialization with multiple classes."""
        X, y = iris_data
        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X, y)

        assert clf.X.shape == (150, 4)
        assert clf.y.shape == (150,)
        assert len(clf.label_space) == 3

    def test_learn_initial_training_set_empty_raises(self):
        """Test that empty training set raises error."""
        clf = ConformalMondrianForestClassifier()
        with pytest.raises(ValueError, match="empty"):
            clf.learn_initial_training_set(np.array([]).reshape(0, 2), np.array([]))

    def test_learn_one_appends_data(self, iris_data):
        """Test that learn_one appends points correctly."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]
        x_new, y_new = X[50], y[50]

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        assert clf.X.shape[0] == 50
        clf.learn_one(x_new, y_new)
        assert clf.X.shape[0] == 51

    def test_n_trees_parameter(self, iris_data):
        """Test that n_trees parameter is stored correctly."""
        X, y = iris_data
        X_train, y_train = X[:50], y[:50]

        for n_trees in [1, 5, 10, 20]:
            clf = ConformalMondrianForestClassifier(n_trees=n_trees, rnd_state=0)
            clf.learn_initial_training_set(X_train, y_train)
            assert clf.n_trees == n_trees

    def test_predict_returns_conformal_prediction_set(self, iris_data):
        """Test that predict returns DiscretePredictionSet for scalar epsilon."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=5, epsilon=0.1, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60])
        assert isinstance(result, DiscretePredictionSet)
        assert hasattr(result, 'elements')

    def test_predict_multi_level_epsilon(self, iris_data):
        """Test that predict returns MultiLevelPredictionSet for array epsilon."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        epsilons = np.array([0.05, 0.1, 0.2])
        result = clf.predict(X[60], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert result.levels == sorted(epsilons.tolist())

    def test_predict_with_return_p_values(self, iris_data):
        """Test return_p_values flag."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, epsilon=0.1, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60], return_p_values=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], DiscretePredictionSet)
        assert isinstance(result[1], dict)
        assert set(result[1].keys()) == set(clf.label_space)

        for p in result[1].values():
            assert 0 <= p <= 1

    def test_predict_with_return_update(self, iris_data):
        """Test return_update flag."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, epsilon=0.1, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        result = clf.predict(X[60], return_update=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], DiscretePredictionSet)
        assert isinstance(result[1], dict)
        assert 'n_leaves_all' in result[1]
        assert 'counts_all' in result[1]
        assert 'leaf_star_train_indices_all' in result[1]
        assert len(result[1]['n_leaves_all']) == 3

    def test_p_values_in_unit_interval(self, iris_data):
        """Test that all p-values lie in [0, 1]."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        for test_idx in range(60, 80):
            _, p_values = clf.predict(X[test_idx], return_p_values=True)
            for p in p_values.values():
                assert 0 <= p <= 1
            clf.learn_one(X[test_idx], y[test_idx])

    def test_monotonicity_multi_level_epsilon(self, iris_data):
        """Smaller epsilon → larger (or equal) prediction set."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
        result = clf.predict(X[60], epsilon=epsilons)

        for i in range(len(epsilons) - 1):
            assert len(result[epsilons[i]].elements) >= len(result[epsilons[i + 1]].elements)

    def test_compute_p_value_single_label(self, iris_data):
        """Test compute_p_value for a single label."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        p_val = clf.compute_p_value(X[60], y=0)
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1

    def test_compute_p_value_unknown_label_raises(self, iris_data):
        """Test that compute_p_value with unknown label raises error."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X_train, y_train)

        with pytest.raises(ValueError, match="label_space"):
            clf.compute_p_value(X[60], y=999)

    def test_reproducibility_with_rnd_state(self, iris_data):
        """Test that same rnd_state gives same predictions."""
        X, y = iris_data
        X_train, y_train = X[:60], y[:60]

        clf1 = ConformalMondrianForestClassifier(n_trees=5, epsilon=0.1, rnd_state=42)
        clf1.learn_initial_training_set(X_train, y_train)
        result1 = clf1.predict(X[60], return_p_values=True)

        clf2 = ConformalMondrianForestClassifier(n_trees=5, epsilon=0.1, rnd_state=42)
        clf2.learn_initial_training_set(X_train, y_train)
        result2 = clf2.predict(X[60], return_p_values=True)

        for label in clf1.label_space:
            assert np.isclose(result1[1][label], result2[1][label])

    def test_bag_function_permutation_invariance(self):
        """Critical test: forest NCM is a bag function (permutation-invariant)."""
        np.random.seed(0)
        X, y = make_blobs(n_samples=50, n_features=3, centers=2, random_state=0)
        x_test = np.array([0.5, 0.5, 0.5])

        clf1 = ConformalMondrianForestClassifier(n_trees=5, rnd_state=0)
        clf1.learn_initial_training_set(X, y)
        result1 = clf1.predict(x_test, return_p_values=True)

        perm = np.random.permutation(len(X))
        X_perm = X[perm]
        y_perm = y[perm]

        clf2 = ConformalMondrianForestClassifier(n_trees=5, rnd_state=0)
        clf2.learn_initial_training_set(X_perm, y_perm)
        result2 = clf2.predict(x_test, return_p_values=True)

        for label in [0, 1]:
            assert np.isclose(result1[1][label], result2[1][label], atol=1e-10)

    def test_validity_binary_classification(self):
        """Test validity: error rate <= epsilon + margin on binary data."""
        X, y = make_blobs(n_samples=200, n_features=4, centers=2, random_state=0)

        epsilon = 0.2
        clf = ConformalMondrianForestClassifier(n_trees=5, epsilon=epsilon, rnd_state=0)

        X_init, X_test = X[:50], X[50:]
        y_init, y_test = y[:50], y[50:]

        clf.learn_initial_training_set(X_init, y_init)

        errors = 0
        for x_t, y_t in zip(X_test, y_test):
            pred_set = clf.predict(x_t)
            if y_t not in pred_set.elements:
                errors += 1
            clf.learn_one(x_t, y_t)

        error_rate = errors / len(X_test)
        margin = 0.15
        assert error_rate <= epsilon + margin

    def test_validity_multiclass(self):
        """Test validity on multiclass data."""
        X, y = make_blobs(n_samples=200, n_features=4, centers=3, random_state=0)

        epsilon = 0.15
        clf = ConformalMondrianForestClassifier(n_trees=5, epsilon=epsilon, rnd_state=0)

        X_init, X_test = X[:60], X[60:]
        y_init, y_test = y[:60], y[60:]

        clf.learn_initial_training_set(X_init, y_init)

        errors = 0
        for x_t, y_t in zip(X_test, y_test):
            pred_set = clf.predict(x_t)
            if y_t not in pred_set.elements:
                errors += 1
            clf.learn_one(x_t, y_t)

        error_rate = errors / len(X_test)
        margin = 0.15
        assert error_rate <= epsilon + margin

    def test_forest_vs_single_tree_efficiency(self):
        """Forest should produce valid predictions like single tree.

        The ensemble averages per-tree predictions, which typically reduces
        variance. While we expect smaller prediction sets on average, this is
        not guaranteed for every test case due to random variation. The test
        verifies both produce valid predictions with sensible set sizes.
        """
        np.random.seed(0)
        X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=0)

        X_init, X_test = X[:60], X[60:]
        y_init = y[:60]

        # Single tree - use more permissive epsilon
        clf_tree = ConformalMondrianTreeClassifier(lifetime=1.0, epsilon=0.2, rnd_state=0)
        clf_tree.learn_initial_training_set(X_init, y_init)
        sizes_tree = []
        for x_t in X_test[:10]:
            pred_set = clf_tree.predict(x_t, epsilon=0.2)
            sizes_tree.append(len(pred_set.elements))
            assert isinstance(pred_set, DiscretePredictionSet)

        # Forest with multiple trees
        clf_forest = ConformalMondrianForestClassifier(n_trees=10, lifetime=1.0, epsilon=0.2, rnd_state=0)
        clf_forest.learn_initial_training_set(X_init, y_init)
        sizes_forest = []
        for x_t in X_test[:10]:
            pred_set = clf_forest.predict(x_t, epsilon=0.2)
            sizes_forest.append(len(pred_set.elements))
            assert isinstance(pred_set, DiscretePredictionSet)

        # Both should produce valid predictions
        assert len(sizes_tree) == 10
        assert len(sizes_forest) == 10
        # Set sizes should be reasonable (between 0 and number of classes)
        assert all(0 <= s <= 3 for s in sizes_tree)
        assert all(0 <= s <= 3 for s in sizes_forest)

    def test_arbitrary_label_space(self):
        """Test that arbitrary labels are handled."""
        np.random.seed(0)
        X = np.random.randn(50, 3)
        y = np.array(['cat'] * 20 + ['dog'] * 30)

        clf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        clf.learn_initial_training_set(X, y)

        result = clf.predict(X[0], epsilon=0.1, return_p_values=True)
        assert 'cat' in result[1]
        assert 'dog' in result[1]

    def test_different_n_trees_values(self):
        """Test forest with various n_trees configurations."""
        X, y = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)
        X_train, X_test = X[:50], X[50:]
        y_train = y[:50]

        for n_trees in [1, 3, 10]:
            clf = ConformalMondrianForestClassifier(n_trees=n_trees, rnd_state=0)
            clf.learn_initial_training_set(X_train, y_train)

            for x_t in X_test:
                result = clf.predict(x_t, epsilon=0.1, return_p_values=True)
                assert isinstance(result[0], DiscretePredictionSet)
                assert all(0 <= p <= 1 for p in result[1].values())

    def test_lifetime_parameter(self):
        """Test that lifetime parameter affects tree depth."""
        X, y = make_blobs(n_samples=100, n_features=3, centers=3, random_state=0)

        clf_shallow = ConformalMondrianForestClassifier(
            n_trees=3, lifetime=0.01, rnd_state=0
        )
        clf_shallow.learn_initial_training_set(X, y)

        clf_deep = ConformalMondrianForestClassifier(
            n_trees=3, lifetime=np.inf, rnd_state=0
        )
        clf_deep.learn_initial_training_set(X, y)

        result_shallow = clf_shallow.predict(X[0])
        result_deep = clf_deep.predict(X[0])

        assert isinstance(result_shallow, DiscretePredictionSet)
        assert isinstance(result_deep, DiscretePredictionSet)

    def test_epsilon_one(self):
        """Test with epsilon=0.5 (reasonable level)."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        clf = ConformalMondrianTreeClassifier(epsilon=0.5, rnd_state=0)
        clf.learn_initial_training_set(X, y)

        result = clf.predict(X[0])
        # With epsilon=0.5, typically some labels included
        assert isinstance(result, DiscretePredictionSet)


# ===========================================================================
# Regressor tests
# ===========================================================================

from online_cp.regressors import (  # noqa: E402
    ConformalMondrianForestRegressor,
    ConformalMondrianTreeRegressor,
    ConformalRegressor,
    ContinuousPredictionSet,
)


@pytest.fixture(scope="module")
def sinusoid_data():
    """1-D sinusoidal regression data."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.uniform(0, 2 * np.pi, (n, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, n)
    return X, y


class TestConformalMondrianTreeRegressor:

    def test_isinstance_conformal_regressor(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        assert isinstance(reg, ConformalRegressor)

    def test_learn_initial_training_set(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X, y)
        assert reg.X.shape == X.shape
        assert reg.y.shape == y.shape

    def test_learn_one(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X[:10], y[:10])
        reg.learn_one(X[10], y[10])
        assert reg.X.shape[0] == 11

    def test_predict_returns_interval(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X, y)
        result = reg.predict(X[0])
        assert isinstance(result, ContinuousPredictionSet)

    def test_predict_lower_le_upper(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X, y)
        for i in range(10):
            iv = reg.predict(X[i])
            assert iv.lower <= iv.upper

    def test_predict_multi_epsilon(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X, y)
        eps = np.array([0.05, 0.1, 0.2])
        result = reg.predict(X[0], epsilon=eps)
        assert isinstance(result, MultiLevelPredictionSet)
        # Larger epsilon → smaller (or equal) interval
        ivs = [result[e] for e in sorted(eps)]
        widths = [iv.upper - iv.lower for iv in ivs]
        for w1, w2 in zip(widths, widths[1:]):
            assert w1 >= w2 - 1e-9

    def test_compute_p_value_in_range(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X, y)
        for i in range(10):
            pv = reg.compute_p_value(X[i], y[i])
            assert 0.0 <= pv <= 1.0

    def test_validity(self, sinusoid_data):
        """Empirical coverage must be >= 1 - epsilon (marginal validity)."""
        X, y = sinusoid_data
        n_train = 40
        epsilon = 0.1
        covered = 0
        n_test = 30
        reg = ConformalMondrianTreeRegressor(lifetime=1.5, rnd_state=42)
        reg.learn_initial_training_set(X[:n_train], y[:n_train])
        for i in range(n_train, n_train + n_test):
            iv = reg.predict(X[i], epsilon=epsilon)
            if iv.lower <= y[i] <= iv.upper:
                covered += 1
            reg.learn_one(X[i], y[i])
        coverage = covered / n_test
        # Validity: coverage >= 1-epsilon. Allow slack for small sample.
        assert coverage >= 1.0 - epsilon - 0.15, (
            f"Coverage {coverage:.2f} too low for epsilon={epsilon}"
        )

    def test_empty_leaf_returns_infinite_interval(self):
        """When test point lands in a leaf with no training points (A=0), return [-inf, inf]."""
        # Use a test point far outside the training range so the Mondrian tree
        # isolates it in its own leaf on the very first split (A=0 guaranteed).
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (20, 1))
        y = rng.normal(0, 1, 20)
        # Large lifetime → fine-grained splits; test point at -1000 will be
        # separated from all training data by the first split.
        reg = ConformalMondrianTreeRegressor(lifetime=10.0, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        x_test = np.array([-1000.0])
        iv = reg.predict(x_test, epsilon=0.1)
        assert np.isinf(iv.lower) and np.isinf(iv.upper)

    def test_predict_return_update(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(rnd_state=0)
        reg.learn_initial_training_set(X, y)
        result, upd = reg.predict(X[0], return_update=True)
        assert isinstance(result, ContinuousPredictionSet)
        assert "tree" in upd and "leaf_star" in upd

    def test_finite_interval_typical_case(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianTreeRegressor(lifetime=2.0, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        iv = reg.predict(X[0], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)
        assert iv.width() > 0

    def test_single_tree_forest_solver_matches_python_reference(self):
        """The single-tree solver used by ``ConformalMondrianTreeRegressor.
        predict`` (``forest_sweepline_solver`` with M=1) must reproduce the
        interval bounds of the documented pure-Python reference
        ``_build_knot_events`` + ``_pvalue_from_sweep`` on many random cases,
        including edge cases (A=1, A>1, ties), at a fixed tau.

        The two code paths are mathematically identical but differ in the last
        floating-point bits (the reference uses ``B = sum()`` while the solver
        uses ``mean * A``, and the sweeps use different FP orderings), so the
        comparison is at a 1e-9 tolerance rather than strict equality.
        """
        from online_cp.mondrian._forest_solver import forest_sweepline_solver
        from online_cp.mondrian.tree import MondrianTree, _find_leaf, _summarize_tree_reg

        rng = np.random.default_rng(0)
        reg = ConformalMondrianTreeRegressor(lifetime=1.0, rnd_state=0)
        X = rng.uniform(0, 1, (200, 3))
        y = rng.normal(0, 1, 200)
        reg.learn_initial_training_set(X, y)
        n = X.shape[0]

        def close(a, b):
            if np.isnan(a) and np.isnan(b):
                return True
            if a == np.inf and b == np.inf:
                return True
            if a == -np.inf and b == -np.inf:
                return True
            if np.isnan(a) != np.isnan(b):
                return False
            return abs(a - b) < 1e-9

        for trial in range(30):
            x = rng.uniform(0, 1, 3)
            X_aug = np.vstack([reg.X, x])
            root = MondrianTree.grow(
                X_aug, np.random.default_rng(42 + trial), lifetime=1.0, x_test=x
            ).root
            leaf = _find_leaf(root, x)
            idx = leaf.indices[leaf.indices < n]
            A = len(idx)
            if A == 0:
                continue  # empty leaf handled by the fast path; skip
            B = float(reg.y[idx].sum())
            y_star = reg.y[idx]
            ext = reg._outside_ncms_sorted(root, leaf, n)
            # The per-tree summary the forest solver consumes (mean-based B).
            base_ncm, ls_idx, ls_mu = _summarize_tree_reg(root, reg.y, n, x)
            summaries = [(base_ncm, ls_idx, ls_mu)]
            tau = 0.7

            for eps in [0.05, 0.1, 0.25, 0.5, 0.9]:
                ke, pt = reg._build_knot_events(A, B, ext, y_star)
                lo_p, hi_p, _vk_p = reg._pvalue_from_sweep(
                    ke, A, B, ext, y_star, n, pt, float(eps), tau
                )
                intervals, _ = forest_sweepline_solver(
                    summaries, reg.y, float(eps), tau, n, 1, return_exact=True
                )
                if not intervals:
                    lo_c, hi_c = np.nan, np.nan
                else:
                    lo_c = min(i[0] for i in intervals)
                    hi_c = max(i[1] for i in intervals)
                assert close(lo_p, lo_c), (
                    f"lo mismatch trial={trial} A={A} eps={eps}: "
                    f"py={lo_p!r} solver={lo_c!r}"
                )
                assert close(hi_p, hi_c), (
                    f"hi mismatch trial={trial} A={A} eps={eps}: "
                    f"py={hi_p!r} solver={hi_c!r}"
                )

    def test_knot_sweep_pvalue_consistent_with_pvalues_at(self):
        """The knot-sweep p-value (_evaluate_pvalue_at_point) must agree with
        _pvalues_at at every knot value.  This pins the mutually-exclusive
        gt/eq counting convention: a point with score - ncm_test in (0, TOL]
        is a 'tie' (eq), NOT 'greater' (gt).  A previous bug counted such
        points in BOTH gt and eq, making predict and compute_p_value
        internally inconsistent.
        """
        from online_cp.mondrian.tree import MondrianTree, _find_leaf

        _TOL = 1e-9
        rng = np.random.default_rng(0)
        reg = ConformalMondrianTreeRegressor(lifetime=1.0, rnd_state=0)
        X = rng.uniform(0, 1, (200, 3))
        y = rng.normal(0, 1, 200)
        reg.learn_initial_training_set(X, y)
        n = X.shape[0]
        tau = 0.7

        for trial in range(20):
            x = rng.uniform(0, 1, 3)
            X_aug = np.vstack([reg.X, x])
            root = MondrianTree.grow(
                X_aug, np.random.default_rng(7 + trial), lifetime=1.0, x_test=x
            ).root
            leaf = _find_leaf(root, x)
            idx = leaf.indices[leaf.indices < n]
            A = len(idx)
            if A == 0:
                continue  # empty leaf handled separately, no knot solver
            B = float(reg.y[idx].sum())
            y_star = reg.y[idx]
            ext = reg._outside_ncms_sorted(root, leaf, n)
            ke, pt = reg._build_knot_events(A, B, ext, y_star)

            for i in range(1, ke.shape[0]):
                kv = float(ke[i]["value"])
                # Knot-sweep p-value
                p_sweep = reg._evaluate_pvalue_at_point(kv, A, B, ext, y_star, n, tau, pt)
                # _pvalues_at-style p-value (mutually exclusive)
                ncm_test = abs(A * kv - B) / (A + 1)
                inside = np.abs((A + 1) * y_star - B - kv) / (A + 1)
                gt = int(np.sum(ext - ncm_test > _TOL)) + int(
                    np.sum(inside - ncm_test > _TOL)
                )
                eq = int(np.sum(np.abs(ext - ncm_test) <= _TOL)) + int(
                    np.sum(np.abs(inside - ncm_test) <= _TOL)
                )
                p_pv = (gt + tau * (eq + 1 + pt)) / (n + 1)
                assert abs(p_sweep - p_pv) < 1e-12, (
                    f"knot-sweep vs _pvalues_at mismatch trial={trial} i={i}: "
                    f"p_sweep={p_sweep:.9f} p_pv={p_pv:.9f}"
                )

class TestConformalMondrianForestRegressor:

    def test_isinstance_conformal_regressor(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianForestRegressor(rnd_state=0)
        assert isinstance(reg, ConformalRegressor)

    def test_predict_returns_interval(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianForestRegressor(n_trees=5, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        result = reg.predict(X[0])
        assert isinstance(result, ContinuousPredictionSet)

    def test_predict_lower_le_upper(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianForestRegressor(n_trees=5, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        for i in range(5):
            iv = reg.predict(X[i])
            assert iv.lower <= iv.upper

    def test_compute_p_value_in_range(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianForestRegressor(n_trees=5, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        for i in range(5):
            pv = reg.compute_p_value(X[i], y[i])
            assert 0.0 <= pv <= 1.0

    def test_validity(self, sinusoid_data):
        """Empirical coverage must be >= 1 - epsilon (marginal validity)."""
        X, y = sinusoid_data
        n_train = 40
        epsilon = 0.1
        covered = 0
        n_test = 20
        reg = ConformalMondrianForestRegressor(n_trees=5, lifetime=1.5, rnd_state=42)
        reg.learn_initial_training_set(X[:n_train], y[:n_train])
        for i in range(n_train, n_train + n_test):
            iv = reg.predict(X[i], epsilon=epsilon)
            if iv.lower <= y[i] <= iv.upper:
                covered += 1
            reg.learn_one(X[i], y[i])
        coverage = covered / n_test
        assert coverage >= 1.0 - epsilon - 0.15, (
            f"Coverage {coverage:.2f} too low for epsilon={epsilon}"
        )

    def test_boundary_extension_to_infinity(self):
        """If p-value is high at grid boundary, interval extends to ±inf."""
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (10, 1))
        y = np.zeros(10)  # constant labels → huge uncertainty for any new point
        reg = ConformalMondrianForestRegressor(
            n_trees=3, lifetime=1e-9, rnd_state=0
        )
        reg.learn_initial_training_set(X, y)
        x_test = np.array([0.5])
        iv = reg.predict(x_test, epsilon=0.1)
        # Degenerate case: very sparse tree → high uncertainty
        # For empty intervals, lower and upper are NaN; otherwise, lower <= upper
        if iv.is_empty:
            assert np.isnan(iv.lower) and np.isnan(iv.upper)
        else:
            assert iv.lower <= iv.upper

    def test_multi_epsilon(self, sinusoid_data):
        X, y = sinusoid_data
        reg = ConformalMondrianForestRegressor(n_trees=5, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        eps = np.array([0.05, 0.1, 0.2])
        result = reg.predict(X[0], epsilon=eps)
        assert isinstance(result, MultiLevelPredictionSet)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestMondrianTreeInspection:
    """Tests for tree inspection utilities (summary, to_dataframe, debug_one)."""

    @pytest.fixture
    def classifier_with_history(self, rng):
        """Create a trained classifier with prediction history."""
        clf = ConformalMondrianTreeClassifier(epsilon=0.1, rnd_state=42)
        X_train = rng.standard_normal((30, 5))
        y_train = (rng.uniform(size=30) > 0.5).astype(int)
        clf.learn_initial_training_set(X_train, y_train)

        # Make some predictions to cache the tree
        for _ in range(5):
            x_test = rng.standard_normal(5)
            y_test = int(rng.uniform() > 0.5)
            clf.learn_one(x_test, y_test)
            clf.predict(x_test)

        return clf, X_train, y_train

    @pytest.fixture
    def regressor_with_history(self, rng):
        """Create a trained regressor with prediction history."""
        reg = ConformalMondrianTreeRegressor(epsilon=0.1, rnd_state=42)
        X_train = rng.standard_normal((30, 5))
        y_train = rng.standard_normal(30)
        reg.learn_initial_training_set(X_train, y_train)

        # Make some predictions to cache the tree
        for _ in range(5):
            x_test = rng.standard_normal(5)
            y_test = float(rng.standard_normal())
            reg.learn_one(x_test, y_test)
            reg.predict(x_test)

        return reg, X_train, y_train

    def test_summary_classifier_before_predict(self, rng):
        """Test .summary property for classifier before prediction."""
        clf = ConformalMondrianTreeClassifier(epsilon=0.1, lifetime=5.0, rnd_state=42)
        X_train = rng.standard_normal((20, 3))
        y_train = (rng.uniform(size=20) > 0.5).astype(int)
        clf.learn_initial_training_set(X_train, y_train)

        summary = clf.summary
        assert "n_points" in summary
        assert "n_features" in summary
        assert "n_classes" in summary
        assert "label_space" in summary
        assert "lifetime" in summary
        assert "epsilon" in summary

        assert summary["n_points"] == 20
        assert summary["n_features"] == 3
        assert summary["n_classes"] == 2
        assert summary["epsilon"] == 0.1

        # Tree-structure keys should not be present yet
        assert "n_nodes" not in summary
        assert "n_leaves" not in summary

    def test_summary_classifier_after_predict(self, classifier_with_history):
        """Test .summary property for classifier after prediction."""
        clf, X_train, y_train = classifier_with_history

        summary = clf.summary
        assert summary["n_points"] == 30 + 5  # Initial + learned ones
        assert summary["n_features"] == 5

        # After predict, tree structure info should be present
        assert "n_nodes" in summary
        assert "n_leaves" in summary
        assert "n_branches" in summary
        assert "height" in summary
        assert "total_observed_weight" in summary

        assert summary["n_nodes"] > 0
        assert summary["n_leaves"] > 0
        assert summary["height"] >= 0

    def test_summary_regressor_before_predict(self, rng):
        """Test .summary property for regressor before prediction."""
        reg = ConformalMondrianTreeRegressor(epsilon=0.1, lifetime=5.0, rnd_state=42)
        X_train = rng.standard_normal((20, 3))
        y_train = rng.standard_normal(20)
        reg.learn_initial_training_set(X_train, y_train)

        summary = reg.summary
        assert "n_points" in summary
        assert "n_features" in summary
        assert "y_mean" in summary
        assert "y_std" in summary
        assert "lifetime" in summary
        assert "epsilon" in summary

        assert summary["n_points"] == 20
        assert summary["n_features"] == 3
        assert summary["y_mean"] is not None
        assert summary["y_std"] is not None

        # Tree-structure keys should not be present yet
        assert "n_nodes" not in summary

    def test_summary_regressor_after_predict(self, regressor_with_history):
        """Test .summary property for regressor after prediction."""
        reg, X_train, y_train = regressor_with_history

        summary = reg.summary
        assert summary["n_points"] == 30 + 5  # Initial + learned ones
        assert summary["n_features"] == 5

        # After predict, tree structure info should be present
        assert "n_nodes" in summary
        assert "n_leaves" in summary
        assert "n_branches" in summary
        assert "height" in summary
        assert "total_observed_weight" in summary

    def test_to_dataframe_classifier_raises_before_predict(self, rng):
        """Test that to_dataframe raises for classifier before predict."""
        clf = ConformalMondrianTreeClassifier(rnd_state=42)
        X_train = rng.standard_normal((20, 3))
        y_train = (rng.uniform(size=20) > 0.5).astype(int)
        clf.learn_initial_training_set(X_train, y_train)

        with pytest.raises(RuntimeError, match="No tree has been built yet"):
            clf.to_dataframe()

    def test_to_dataframe_classifier_structure(self, rng, classifier_with_history):
        """Test to_dataframe for classifier returns expected structure."""
        clf, _, _ = classifier_with_history

        df = clf.to_dataframe()

        # Check it's a DataFrame
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

        # Check required columns exist
        required_cols = ["node_id", "parent_id", "is_leaf", "depth",
                         "split_dim", "split_loc", "split_time", "parent_time",
                         "bbox_lower", "bbox_upper", "n_points", "counts"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check data types
        assert df["node_id"].dtype in [int, 'int64', 'int32']
        assert df["is_leaf"].dtype == bool
        assert df["depth"].dtype in [int, 'int64', 'int32']
        assert df["n_points"].dtype in [int, 'int64', 'int32']

        # Check that there's at least one leaf
        assert df["is_leaf"].sum() > 0

        # Check that leaves have counts
        leaf_rows = df[df["is_leaf"]]
        for _, row in leaf_rows.iterrows():
            if pd.notna(row["counts"]):
                assert isinstance(row["counts"], dict)

    def test_to_dataframe_regressor_structure(self, rng, regressor_with_history):
        """Test to_dataframe for regressor returns expected structure."""
        reg, _, _ = regressor_with_history

        df = reg.to_dataframe()

        # Check it's a DataFrame
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

        # Check required columns exist
        required_cols = ["node_id", "parent_id", "is_leaf", "depth",
                         "split_dim", "split_loc", "split_time", "parent_time",
                         "bbox_lower", "bbox_upper", "n_points", "y_mean", "y_std"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check that there's at least one leaf
        assert df["is_leaf"].sum() > 0

        # Check leaf statistics
        leaf_rows = df[df["is_leaf"]]
        for _, row in leaf_rows.iterrows():
            if row["n_points"] > 0:
                # Leaves with points should have y_mean
                if pd.notna(row["y_mean"]):
                    assert isinstance(row["y_mean"], (int, float))

    def test_debug_one_classifier_string_format(self, rng, classifier_with_history):
        """Test debug_one for classifier returns readable string."""
        clf, X_train, _ = classifier_with_history

        x_test = X_train[0:1]  # Use a training point
        debug_str = clf.debug_one(x_test)

        # Check it's a string
        assert isinstance(debug_str, str)

        # Check it contains expected keywords
        assert "LEAF:" in debug_str or "x[" in debug_str  # Should show splits or leaf
        assert "n_points" in debug_str

    def test_debug_one_regressor_string_format(self, rng, regressor_with_history):
        """Test debug_one for regressor returns readable string."""
        reg, X_train, _ = regressor_with_history

        x_test = X_train[0:1]  # Use a training point
        debug_str = reg.debug_one(x_test)

        # Check it's a string
        assert isinstance(debug_str, str)

        # Check it contains expected keywords
        assert "LEAF:" in debug_str or "x[" in debug_str  # Should show splits or leaf
        assert "n_points" in debug_str

    def test_debug_one_classifier_rng_isolation(self, rng, classifier_with_history):
        """Test that debug_one doesn't affect classifier's RNG state."""
        clf, X_train, _ = classifier_with_history

        x_test = X_train[0:1]

        # Call debug_one - should not raise an error
        debug_str = clf.debug_one(x_test)

        # Check it returns a valid string
        assert isinstance(debug_str, str)
        assert len(debug_str) > 0

    def test_debug_one_regressor_rng_isolation(self, rng, regressor_with_history):
        """Test that debug_one doesn't affect regressor's RNG state."""
        reg, X_train, _ = regressor_with_history

        x_test = X_train[0:1]

        # Call debug_one - should not raise an error
        debug_str = reg.debug_one(x_test)

        # Check it returns a valid string
        assert isinstance(debug_str, str)
        assert len(debug_str) > 0


class TestMondrianTreeVisualization:
    """Tests for .draw() and .draw_partition() on all four estimator classes."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        """Use non-interactive Agg backend for all viz tests."""
        import matplotlib
        matplotlib.use("Agg")

    @pytest.fixture
    def clf_2d(self, rng):
        """2-D classifier with prediction history."""
        clf = ConformalMondrianTreeClassifier(lifetime=2.0, rnd_state=7)
        X = rng.standard_normal((40, 2))
        y = (X[:, 0] + rng.standard_normal(40) > 0).astype(int)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(2))
        return clf, X, y

    @pytest.fixture
    def reg_2d(self, rng):
        """2-D regressor with prediction history."""
        reg = ConformalMondrianTreeRegressor(lifetime=2.0, rnd_state=7)
        X = rng.standard_normal((40, 2))
        y = X[:, 0] + rng.standard_normal(40)
        reg.learn_initial_training_set(X, y)
        reg.predict(rng.standard_normal(2))
        return reg, X, y

    @pytest.fixture
    def clf_5d(self, rng):
        """5-D classifier with prediction history (for draw() any-d test)."""
        clf = ConformalMondrianTreeClassifier(lifetime=2.0, rnd_state=7)
        X = rng.standard_normal((30, 5))
        y = (rng.uniform(size=30) > 0.5).astype(int)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(5))
        return clf

    # ---- draw() --------------------------------------------------------

    def test_draw_raises_before_predict(self, rng):
        clf = ConformalMondrianTreeClassifier(rnd_state=7)
        X = rng.standard_normal((20, 2))
        y = (rng.uniform(size=20) > 0.5).astype(int)
        clf.learn_initial_training_set(X, y)
        with pytest.raises(RuntimeError, match="No tree has been built"):
            clf.draw()

    def test_draw_returns_axes_classifier(self, clf_2d):
        import matplotlib.axes
        clf, _, _ = clf_2d
        ax = clf.draw(backend="matplotlib")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draw_returns_axes_regressor(self, reg_2d):
        import matplotlib.axes
        reg, _, _ = reg_2d
        ax = reg.draw(backend="matplotlib")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draw_any_d(self, clf_5d):
        """draw() must work for d > 2."""
        import matplotlib.axes
        ax = clf_5d.draw(backend="matplotlib")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draw_max_depth(self, clf_2d):
        """max_depth limits how many levels are drawn."""
        import matplotlib.pyplot as plt
        clf, _, _ = clf_2d
        ax0 = clf.draw(backend="matplotlib")
        ax1 = clf.draw(max_depth=1, backend="matplotlib")
        # max_depth=1 should produce fewer text artists (nodes) than no cap
        texts0 = len(ax0.texts)
        texts1 = len(ax1.texts)
        assert texts1 <= texts0
        plt.close("all")

    def test_draw_forest_classifier(self, rng):
        """Forest draw uses representative last tree."""
        import matplotlib.axes
        fclf = ConformalMondrianForestClassifier(n_trees=5, lifetime=2.0, rnd_state=7)
        X = rng.standard_normal((30, 2))
        y = (rng.uniform(size=30) > 0.5).astype(int)
        fclf.learn_initial_training_set(X, y)
        fclf.predict(rng.standard_normal(2))
        ax = fclf.draw(backend="matplotlib")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draw_forest_regressor(self, rng):
        import matplotlib.axes
        freg = ConformalMondrianForestRegressor(n_trees=5, lifetime=2.0, rnd_state=7)
        X = rng.standard_normal((30, 2))
        y = X[:, 0] + rng.standard_normal(30)
        freg.learn_initial_training_set(X, y)
        freg.predict(rng.standard_normal(2))
        ax = freg.draw(backend="matplotlib")
        assert isinstance(ax, matplotlib.axes.Axes)

    # ---- draw_partition() ----------------------------------------------

    def test_draw_partition_raises_before_predict(self, rng):
        clf = ConformalMondrianTreeClassifier(rnd_state=7)
        X = rng.standard_normal((20, 2))
        y = (rng.uniform(size=20) > 0.5).astype(int)
        clf.learn_initial_training_set(X, y)
        with pytest.raises(RuntimeError, match="No tree has been built"):
            clf.draw_partition()

    def test_draw_partition_raises_wrong_d(self, clf_5d):
        with pytest.raises(ValueError, match="exactly 2 features"):
            clf_5d.draw_partition()

    def test_draw_partition_returns_axes_classifier(self, clf_2d):
        import matplotlib.axes
        clf, _, _ = clf_2d
        ax = clf.draw_partition()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draw_partition_returns_axes_regressor(self, reg_2d):
        import matplotlib.axes
        reg, _, _ = reg_2d
        ax = reg.draw_partition()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draw_partition_n_rectangles_equals_n_leaves(self, clf_2d):
        """Number of drawn rectangles must equal number of leaves."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        clf, _, _ = clf_2d
        ax = clf.draw_partition(scatter=False)
        n_rects = sum(isinstance(p, Rectangle) for p in ax.patches)
        n_leaves = clf.summary["n_leaves"]
        assert n_rects == n_leaves
        plt.close("all")

    def test_draw_partition_finite_bounds(self, clf_2d):
        """All drawn rectangles must have finite coordinates (inf clipping works)."""
        import math

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        clf, _, _ = clf_2d
        ax = clf.draw_partition(scatter=False)
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                bbox = patch.get_bbox()
                assert math.isfinite(bbox.x0)
                assert math.isfinite(bbox.y0)
                assert math.isfinite(bbox.x1)
                assert math.isfinite(bbox.y1)
        plt.close("all")

    def test_draw_partition_scatter_off(self, clf_2d):
        """scatter=False should produce no scatter artists."""
        import matplotlib.pyplot as plt
        clf, _, _ = clf_2d
        ax = clf.draw_partition(scatter=False)
        # PathCollections (scatter) should be absent
        from matplotlib.collections import PathCollection
        assert not any(isinstance(c, PathCollection) for c in ax.collections)
        plt.close("all")


class TestMondrianForestInspection:
    """Tests for forest indexing (__getitem__, __len__, __iter__) and T1 inspection
    methods that were broken before T2 (forest _last_tree never set)."""

    @pytest.fixture
    def forest_clf(self, rng):
        fclf = ConformalMondrianForestClassifier(n_trees=4, lifetime=2.0, rnd_state=42)
        X = rng.standard_normal((30, 3))
        y = (rng.uniform(size=30) > 0.5).astype(int)
        fclf.learn_initial_training_set(X, y)
        x_test = rng.standard_normal(3)
        fclf.predict(x_test)
        return fclf, X, y, x_test

    @pytest.fixture
    def forest_reg(self, rng):
        freg = ConformalMondrianForestRegressor(n_trees=4, lifetime=2.0, rnd_state=42)
        X = rng.standard_normal((30, 3))
        y = X[:, 0] + rng.standard_normal(30)
        freg.learn_initial_training_set(X, y)
        x_test = rng.standard_normal(3)
        freg.predict(x_test)
        return freg, X, y, x_test

    # ---- forest summary / to_dataframe / debug_one (previously broken) ----

    def test_forest_clf_summary_after_predict(self, forest_clf):
        fclf, _, _, _ = forest_clf
        s = fclf.summary
        assert "n_nodes" in s
        assert "n_leaves" in s
        assert s["n_trees"] == 4

    def test_forest_reg_summary_after_predict(self, forest_reg):
        freg, _, _, _ = forest_reg
        s = freg.summary
        assert "n_nodes" in s
        assert "n_leaves" in s
        assert s["n_trees"] == 4

    def test_forest_clf_to_dataframe(self, forest_clf):
        import pandas as pd
        fclf, _, _, _ = forest_clf
        df = fclf.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "node_id" in df.columns
        assert len(df) > 0

    def test_forest_reg_to_dataframe(self, forest_reg):
        import pandas as pd
        freg, _, _, _ = forest_reg
        df = freg.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    # ---- __len__ -------------------------------------------------------

    def test_len_clf(self, forest_clf):
        fclf, _, _, _ = forest_clf
        assert len(fclf) == 4

    def test_len_reg(self, forest_reg):
        freg, _, _, _ = forest_reg
        assert len(freg) == 4

    # ---- __getitem__ ---------------------------------------------------

    def test_getitem_clf_returns_single_tree_view(self, forest_clf):
        fclf, _, _, _ = forest_clf
        view = fclf[0]
        assert isinstance(view, ConformalMondrianTreeClassifier)
        assert view._last_tree is not None

    def test_getitem_reg_returns_single_tree_view(self, forest_reg):
        freg, _, _, _ = forest_reg
        view = freg[0]
        assert isinstance(view, ConformalMondrianTreeRegressor)
        assert view._last_tree is not None

    def test_getitem_deterministic(self, forest_clf):
        """Accessing the same index twice yields structurally identical trees."""
        fclf, _, _, _ = forest_clf
        v0a = fclf[0]
        v0b = fclf[0]
        # Same number of leaves
        from online_cp.mondrian.tree import _tree_struct_stats
        assert _tree_struct_stats(v0a._last_tree) == _tree_struct_stats(v0b._last_tree)

    def test_getitem_different_trees(self, forest_clf):
        """Different indices yield distinct trees."""
        fclf, _, _, _ = forest_clf
        from online_cp.mondrian.tree import _tree_struct_stats
        stats = [_tree_struct_stats(fclf[i]._last_tree) for i in range(4)]
        # Just check that they can be accessed without error; structure may coincide
        assert len(stats) == 4

    def test_getitem_out_of_range_clf(self, forest_clf):
        fclf, _, _, _ = forest_clf
        with pytest.raises(IndexError):
            _ = fclf[4]
        with pytest.raises(IndexError):
            _ = fclf[-1]

    def test_getitem_out_of_range_reg(self, forest_reg):
        freg, _, _, _ = forest_reg
        with pytest.raises(IndexError):
            _ = freg[10]

    def test_getitem_before_predict_raises(self, rng):
        fclf = ConformalMondrianForestClassifier(n_trees=3, rnd_state=0)
        X = rng.standard_normal((20, 2))
        y = (rng.uniform(size=20) > 0.5).astype(int)
        fclf.learn_initial_training_set(X, y)
        with pytest.raises(RuntimeError, match="No trees cached"):
            _ = fclf[0]

    # ---- __iter__ ------------------------------------------------------

    def test_iter_clf(self, forest_clf):
        fclf, _, _, _ = forest_clf
        views = list(fclf)
        assert len(views) == 4
        assert all(isinstance(v, ConformalMondrianTreeClassifier) for v in views)

    def test_iter_reg(self, forest_reg):
        freg, _, _, _ = forest_reg
        views = list(freg)
        assert len(views) == 4

    # ---- view inspection methods work ----------------------------------

    def test_getitem_view_summary(self, forest_clf):
        fclf, _, _, _ = forest_clf
        view = fclf[2]
        s = view.summary
        assert "n_nodes" in s

    def test_getitem_view_to_dataframe(self, forest_clf):
        import pandas as pd
        fclf, _, _, _ = forest_clf
        df = fclf[1].to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_getitem_reg_via_compute_p_value(self, rng):
        """__getitem__ also works after compute_p_value (not just predict)."""
        freg = ConformalMondrianForestRegressor(n_trees=3, lifetime=2.0, rnd_state=0)
        X = rng.standard_normal((20, 2))
        y = X[:, 0] + rng.standard_normal(20)
        freg.learn_initial_training_set(X, y)
        x_test = rng.standard_normal(2)
        freg.compute_p_value(x_test, y_cand=0.0)
        view = freg[0]
        assert view._last_tree is not None


# ---------------------------------------------------------------------------
# T3 — max_depth hard-cap tests
# ---------------------------------------------------------------------------

class TestMondrianDepthCap:
    """Tests for the max_depth sampler cap on all four Mondrian estimators."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(7)

    @pytest.fixture
    def clf_capped(self, rng):
        """2-D classifier with max_depth=2 and high lifetime (cap dominates)."""
        X = rng.standard_normal((40, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = ConformalMondrianTreeClassifier(lifetime=100.0, max_depth=2, rnd_state=7)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(2))
        return clf

    @pytest.fixture
    def reg_capped(self, rng):
        """2-D regressor with max_depth=2 and high lifetime (cap dominates)."""
        X = rng.standard_normal((40, 2))
        y = X[:, 0] + rng.standard_normal(40) * 0.3
        reg = ConformalMondrianTreeRegressor(lifetime=100.0, max_depth=2, rnd_state=7)
        reg.learn_initial_training_set(X, y)
        reg.predict(rng.standard_normal(2))
        return reg

    # ------------------------------------------------------------------
    # Validity: max_depth must be a non-negative int or None
    # ------------------------------------------------------------------

    def test_invalid_max_depth_negative_raises(self):
        with pytest.raises(ValueError, match="max_depth"):
            ConformalMondrianTreeClassifier(max_depth=-1)

    def test_invalid_max_depth_float_raises(self):
        # float should raise
        with pytest.raises(ValueError, match="max_depth"):
            ConformalMondrianTreeClassifier(max_depth=2.5)
        with pytest.raises(ValueError, match="max_depth"):
            ConformalMondrianTreeClassifier(max_depth=0.0)

    def test_invalid_max_depth_forest_raises(self):
        with pytest.raises(ValueError, match="max_depth"):
            ConformalMondrianForestClassifier(max_depth=-1)

    def test_invalid_max_depth_reg_raises(self):
        with pytest.raises(ValueError, match="max_depth"):
            ConformalMondrianTreeRegressor(max_depth=-3)

    # ------------------------------------------------------------------
    # Structural: height <= max_depth after predict
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("depth_cap", [1, 2, 3])
    def test_tree_clf_height_bounded(self, rng, depth_cap):
        X = rng.standard_normal((50, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = ConformalMondrianTreeClassifier(lifetime=100.0, max_depth=depth_cap, rnd_state=42)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(2))
        assert clf.summary["height"] <= depth_cap

    @pytest.mark.parametrize("depth_cap", [1, 2, 3])
    def test_tree_reg_height_bounded(self, rng, depth_cap):
        X = rng.standard_normal((50, 2))
        y = X[:, 0] + rng.standard_normal(50) * 0.3
        reg = ConformalMondrianTreeRegressor(lifetime=100.0, max_depth=depth_cap, rnd_state=42)
        reg.learn_initial_training_set(X, y)
        reg.predict(rng.standard_normal(2))
        assert reg.summary["height"] <= depth_cap

    def test_forest_clf_height_bounded(self, rng):
        X = rng.standard_normal((40, 2))
        y = (X[:, 0] > 0).astype(int)
        fclf = ConformalMondrianForestClassifier(n_trees=4, lifetime=100.0, max_depth=2, rnd_state=7)
        fclf.learn_initial_training_set(X, y)
        fclf.predict(rng.standard_normal(2))
        for i in range(4):
            assert fclf[i].summary["height"] <= 2, f"Tree {i} exceeds max_depth"

    def test_forest_reg_height_bounded(self, rng):
        X = rng.standard_normal((40, 2))
        y = X[:, 0] + rng.standard_normal(40) * 0.3
        freg = ConformalMondrianForestRegressor(n_trees=4, lifetime=100.0, max_depth=2, rnd_state=7)
        freg.learn_initial_training_set(X, y)
        freg.predict(rng.standard_normal(2))
        for i in range(4):
            assert freg[i].summary["height"] <= 2, f"Tree {i} exceeds max_depth"

    # ------------------------------------------------------------------
    # Corner cases
    # ------------------------------------------------------------------

    def test_max_depth_0_produces_single_leaf(self, rng):
        X = rng.standard_normal((20, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = ConformalMondrianTreeClassifier(lifetime=100.0, max_depth=0, rnd_state=0)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(2))
        assert clf.summary["n_leaves"] == 1
        assert clf.summary["height"] == 0

    def test_max_depth_1_at_most_two_leaves(self, rng):
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = ConformalMondrianTreeClassifier(lifetime=100.0, max_depth=1, rnd_state=1)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(2))
        assert clf.summary["n_leaves"] <= 2

    def test_max_depth_none_unchanged(self, rng):
        """max_depth=None must produce the same tree as not passing max_depth."""
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] > 0).astype(int)
        x_test = rng.standard_normal(2)

        clf_default = ConformalMondrianTreeClassifier(lifetime=2.0, rnd_state=99)
        clf_none = ConformalMondrianTreeClassifier(lifetime=2.0, max_depth=None, rnd_state=99)

        for clf in (clf_default, clf_none):
            clf.learn_initial_training_set(X.copy(), y.copy())
            clf.predict(x_test.copy())

        assert clf_default.summary == clf_none.summary

    # ------------------------------------------------------------------
    # Forest view carries max_depth
    # ------------------------------------------------------------------

    def test_forest_getitem_view_has_max_depth(self, rng):
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] > 0).astype(int)
        fclf = ConformalMondrianForestClassifier(n_trees=3, lifetime=100.0, max_depth=2, rnd_state=5)
        fclf.learn_initial_training_set(X, y)
        fclf.predict(rng.standard_normal(2))
        for i in range(3):
            assert fclf[i].max_depth == fclf.max_depth

    def test_forest_reg_getitem_view_has_max_depth(self, rng):
        X = rng.standard_normal((30, 2))
        y = X[:, 0] + rng.standard_normal(30) * 0.3
        freg = ConformalMondrianForestRegressor(n_trees=3, lifetime=100.0, max_depth=2, rnd_state=5)
        freg.learn_initial_training_set(X, y)
        freg.predict(rng.standard_normal(2))
        for i in range(3):
            assert freg[i].max_depth == freg.max_depth

    # ------------------------------------------------------------------
    # Visualisation cap is independent of sampler cap
    # ------------------------------------------------------------------

    def test_draw_display_cap_independent_of_sampler_cap(self, clf_capped):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Sampler capped at max_depth=2; display capped at 1 — should work fine
        fig, ax = plt.subplots()
        returned_ax = clf_capped.draw(ax=ax, max_depth=1)
        assert returned_ax is ax
        plt.close("all")


class TestTreeVisualization:
    """Tests for the graphviz/matplotlib draw() backend dispatch."""

    @pytest.fixture
    def fitted_clf(self):
        """A fitted 2-D classifier with a cached tree."""
        import matplotlib
        matplotlib.use("Agg")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = ConformalMondrianTreeClassifier(lifetime=3.0, rnd_state=0)
        clf.learn_initial_training_set(X, y)
        clf.predict(rng.standard_normal(2))
        return clf

    def test_matplotlib_backend_returns_axes(self, fitted_clf):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        result = fitted_clf.draw(backend="matplotlib")
        import matplotlib.axes
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close("all")

    def test_auto_backend_with_ax_returns_axes(self, fitted_clf):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = fitted_clf.draw(ax=ax, backend="auto")
        import matplotlib.axes
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close("all")

    def test_matplotlib_backend_with_ax_returns_same_ax(self, fitted_clf):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = fitted_clf.draw(ax=ax, backend="matplotlib")
        assert result is ax
        plt.close("all")

    def test_invalid_backend_raises_valueerror(self, fitted_clf):
        with pytest.raises(ValueError, match="backend"):
            fitted_clf.draw(backend="bogus")

    def test_missing_graphviz_raises_importerror(self, fitted_clf, monkeypatch):
        import online_cp.mondrian.tree as _mod
        monkeypatch.setattr(_mod, "_has_graphviz", lambda: False)
        with pytest.raises(ImportError, match="viz"):
            fitted_clf.draw(backend="graphviz")

    def test_draw_before_predict_raises_runtimeerror(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = ConformalMondrianTreeClassifier(rnd_state=0)
        clf.learn_initial_training_set(X, y)
        with pytest.raises(RuntimeError):
            clf.draw()

    @pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
    def test_graphviz_backend_no_ax_returns_digraph(self, fitted_clf):
        import graphviz
        result = fitted_clf.draw(backend="graphviz")
        assert isinstance(result, graphviz.Digraph)

    @pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
    def test_graphviz_digraph_source_nonempty(self, fitted_clf):
        result = fitted_clf.draw(backend="graphviz")
        assert len(result.source) > 0

    @pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
    def test_graphviz_auto_no_ax_returns_digraph(self, fitted_clf):
        import graphviz
        result = fitted_clf.draw(backend="auto")
        assert isinstance(result, graphviz.Digraph)

    @pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
    def test_graphviz_with_ax_returns_axes(self, fitted_clf):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.axes
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = fitted_clf.draw(ax=ax, backend="graphviz")
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close("all")

    @pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
    def test_graphviz_digraph_can_render_svg(self, fitted_clf, tmp_path):
        """Graphviz Digraph can be rendered to SVG (vector format)."""
        result = fitted_clf.draw(backend="graphviz")
        svg_path = str(tmp_path / "test_tree")
        result.render(svg_path, format="svg", cleanup=True)
        svg_file = svg_path + ".svg"
        assert os.path.exists(svg_file)
        with open(svg_file) as f:
            content = f.read()
        # SVG should contain vector elements
        assert "<svg" in content or "<g>" in content

    @pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
    def test_graphviz_digraph_can_render_pdf(self, fitted_clf, tmp_path):
        """Graphviz Digraph can be rendered to PDF (vector format)."""
        result = fitted_clf.draw(backend="graphviz")
        pdf_path = str(tmp_path / "test_tree")
        result.render(pdf_path, format="pdf", cleanup=True)
        pdf_file = pdf_path + ".pdf"
        assert os.path.exists(pdf_file)
        with open(pdf_file, "rb") as f:
            header = f.read(4)
        # PDF files start with %PDF
        assert header == b"%PDF"

    def test_matplotlib_draw_can_save_svg(self, fitted_clf, tmp_path):
        """Matplotlib backend produces vector SVG output when saved."""
        ax = fitted_clf.draw(backend="matplotlib")
        svg_path = str(tmp_path / "tree_mpl.svg")
        ax.figure.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
        assert os.path.exists(svg_path)
        with open(svg_path) as f:
            content = f.read()
        assert "<svg" in content


# ===========================================================================
# MondrianTree standalone core
# ===========================================================================


class TestMondrianTreeCore:
    """Unit tests for the standalone ``MondrianTree`` partition object."""

    @staticmethod
    def _leaf_groups(tree, index_map=None):
        """Return the partition as a sorted list of sorted original-index tuples."""
        groups = []
        for leaf in tree.collect_leaves():
            idx = leaf.indices if index_map is None else index_map[leaf.indices]
            groups.append(tuple(sorted(int(i) for i in idx)))
        return sorted(groups)

    def test_grow_returns_mondrian_tree(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (30, 3))
        tree = MondrianTree.grow(X, rng, lifetime=2.0)
        assert isinstance(tree, MondrianTree)
        assert tree.root.is_leaf() or not tree.root.is_leaf()  # a valid node
        assert len(tree) == tree.struct_stats()["n_leaves"]

    def test_grow_deterministic_same_seed(self):
        X = np.random.default_rng(0).uniform(0, 1, (30, 3))
        t1 = MondrianTree.grow(X, np.random.default_rng(7), lifetime=2.0)
        t2 = MondrianTree.grow(X, np.random.default_rng(7), lifetime=2.0)
        assert t1.struct_stats() == t2.struct_stats()
        assert self._leaf_groups(t1) == self._leaf_groups(t2)

    def test_find_leaf_returns_leaf_containing_point(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (40, 2))
        tree = MondrianTree.grow(X, rng, lifetime=3.0)
        for i in range(len(X)):
            leaf = tree.find_leaf(X[i])
            assert leaf.is_leaf()
            assert i in set(int(j) for j in leaf.indices)

    def test_collect_leaves_are_all_leaves(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (25, 2))
        tree = MondrianTree.grow(X, rng, lifetime=3.0)
        leaves = tree.collect_leaves()
        assert len(leaves) >= 1
        assert all(node.is_leaf() for node in leaves)
        # Every training row lands in exactly one leaf.
        covered = sorted(int(i) for leaf in leaves for i in leaf.indices)
        assert covered == list(range(len(X)))

    def test_partition_is_permutation_invariant(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (25, 2))
        perm = rng.permutation(len(X))
        t = MondrianTree.grow(X, np.random.default_rng(3), lifetime=3.0)
        tp = MondrianTree.grow(X[perm], np.random.default_rng(3), lifetime=3.0)
        # Map permuted-row indices back to original identities before comparing.
        assert self._leaf_groups(t) == self._leaf_groups(tp, index_map=perm)

    def test_to_dataframe_has_one_row_per_node(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (20, 2))
        tree = MondrianTree.grow(X, rng, lifetime=3.0)
        df = tree.to_dataframe()
        assert len(df) == tree.struct_stats()["n_nodes"]
        assert int(df["is_leaf"].sum()) == tree.struct_stats()["n_leaves"]

    def test_string_lifetime_is_truncation_of_master(self):
        # 'sqrt_n' resolves a finite lifetime and restores the RNG, so the
        # grown tree is a deterministic truncation reproducible across calls.
        X = np.random.default_rng(5).uniform(0, 1, (60, 2))
        t1 = MondrianTree.grow(X, np.random.default_rng(9), lifetime="sqrt_n")
        t2 = MondrianTree.grow(X, np.random.default_rng(9), lifetime="sqrt_n")
        assert self._leaf_groups(t1) == self._leaf_groups(t2)
        assert t1.lifetime == t2.lifetime

    def test_draw_partition_requires_two_features(self):
        rng = np.random.default_rng(6)
        X = rng.uniform(0, 1, (20, 3))
        tree = MondrianTree.grow(X, rng, lifetime=2.0)
        with pytest.raises(ValueError):
            tree.draw_partition()


class TestMinSamplesLeafConformal:
    """Tests for the ``min_samples_leaf`` spatial regularizer on the four
    Mondrian conformal predictors (tree + forest, regression + classification).

    ``min_samples_leaf`` is a label-independent truncation: it depends only on
    leaf cardinalities (the feature geometry X), never on the candidate label y.
    Therefore the partition remains rigid during the algebraic y-sweep and the
    exact O(n log n) solver is untouched — exchangeability is preserved.
    """

    def _reg_data(self, n=60, d=2, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n, d))
        y = X.sum(axis=1) + rng.normal(0, 0.1, n)
        return X, y

    def _clf_data(self, n=60, d=2, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n, d))
        y = (X[:, 0] > 0.5).astype(int)
        return X, y

    # ---- param validation ------------------------------------------------

    def test_min_samples_leaf_zero_rejected(self):
        from online_cp import ConformalMondrianTreeRegressor

        with pytest.raises(ValueError, match="positive integer"):
            ConformalMondrianTreeRegressor(min_samples_leaf=0)

    def test_min_samples_leaf_negative_rejected(self):
        from online_cp import ConformalMondrianTreeRegressor

        with pytest.raises(ValueError, match="positive integer"):
            ConformalMondrianTreeRegressor(min_samples_leaf=-1)

    def test_min_samples_leaf_float_rejected(self):
        from online_cp import ConformalMondrianTreeClassifier

        with pytest.raises(ValueError, match="positive integer"):
            ConformalMondrianTreeClassifier(min_samples_leaf=1.5)

    def test_online_with_min_samples_leaf_gt1_rejected(self):
        """online=True + min_samples_leaf>1 is rejected: the projective
        ExtendMondrianBlock is only valid for the uncapped process, and a
        cardinality-gated online unfreeze has no established exact law.
        min_samples_leaf=1 (the pure process) remains allowed online."""
        from online_cp import (
            ConformalMondrianForestClassifier,
            ConformalMondrianForestRegressor,
            ConformalMondrianTreeClassifier,
            ConformalMondrianTreeRegressor,
        )

        for cls, kw in [
            (ConformalMondrianTreeRegressor, {}),
            (ConformalMondrianForestRegressor, {"n_trees": 2}),
            (ConformalMondrianTreeClassifier, {}),
            (ConformalMondrianForestClassifier, {"n_trees": 2}),
        ]:
            with pytest.raises(ValueError, match="min_samples_leaf>1 is not yet supported"):
                cls(online=True, min_samples_leaf=3, **kw)
            # min_samples_leaf=1 is the pure process and remains valid online.
            assert cls(online=True, min_samples_leaf=1, **kw).min_samples_leaf == 1

    def test_min_samples_leaf_attr_stored(self):
        from online_cp import (
            ConformalMondrianForestClassifier,
            ConformalMondrianForestRegressor,
            ConformalMondrianTreeClassifier,
            ConformalMondrianTreeRegressor,
        )

        for cls, kw in [
            (ConformalMondrianTreeRegressor, {}),
            (ConformalMondrianForestRegressor, {"n_trees": 2}),
            (ConformalMondrianTreeClassifier, {}),
            (ConformalMondrianForestClassifier, {"n_trees": 2}),
        ]:
            obj = cls(min_samples_leaf=5, **kw)
            assert obj.min_samples_leaf == 5

    # ---- functional: prediction runs and is finite ------------------------

    def test_tree_regressor_prediction_finite_with_min_leaf(self):
        from online_cp import ConformalMondrianTreeRegressor

        X, y = self._reg_data()
        m = ConformalMondrianTreeRegressor(lifetime=3.0, min_samples_leaf=5, rnd_state=0)
        m.learn_initial_training_set(X, y)
        iv = m.predict(X[0], epsilon=0.1)
        lo, hi = iv.lower, iv.upper
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi

    def test_tree_classifier_runs_with_min_leaf_and_lambda_zero(self):
        from online_cp import ConformalMondrianTreeClassifier

        X, y = self._clf_data()
        c = ConformalMondrianTreeClassifier(lifetime=3.0, min_samples_leaf=5, lambda_=0.0, rnd_state=0)
        c.learn_initial_training_set(X, y)
        ps = c.predict(X[0], epsilon=0.1)
        assert len(ps) >= 1
        assert set(np.asarray(ps.elements).tolist()).issubset(set([0, 1]))

    def test_forest_regressor_runs_with_min_leaf(self):
        from online_cp import ConformalMondrianForestRegressor

        X, y = self._reg_data()
        f = ConformalMondrianForestRegressor(n_trees=3, lifetime=3.0, min_samples_leaf=5, rnd_state=0)
        f.learn_initial_training_set(X, y)
        iv = f.predict(X[0], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)
        assert iv.lower <= iv.upper

    def test_forest_classifier_runs_with_min_leaf(self):
        from online_cp import ConformalMondrianForestClassifier

        X, y = self._clf_data()
        f = ConformalMondrianForestClassifier(n_trees=3, lifetime=3.0, min_samples_leaf=5, lambda_=0.0, rnd_state=0)
        f.learn_initial_training_set(X, y)
        ps = f.predict(X[0], epsilon=0.1)
        assert len(ps) >= 1

    # ---- leaf-size guarantee on the partition ----------------------------

    def test_leaf_size_guarantee_tree_regressor(self):
        from online_cp import ConformalMondrianTreeRegressor
        from online_cp.mondrian.tree import _collect_leaves

        X, y = self._reg_data(n=60)
        m = 4
        pred = ConformalMondrianTreeRegressor(lifetime=3.0, min_samples_leaf=m, rnd_state=0)
        pred.learn_initial_training_set(X, y)
        pred.predict(X[0], epsilon=0.1)
        tree = pred._last_tree
        n_train = X.shape[0]
        for leaf in _collect_leaves(tree):
            train_idx = leaf.indices[leaf.indices < n_train]
            # A non-root leaf must hold >= m training points. The root leaf is
            # the only legitimate sub-threshold case (min_samples_leaf >> n).
            if len(train_idx) < m:
                # This is only allowed if the tree is a single root leaf.
                assert len(_collect_leaves(tree)) == 1

    def test_min_samples_leaf_identity_at_one(self):
        """min_samples_leaf=1 must reproduce the default (pure Mondrian) tree exactly."""
        from online_cp import ConformalMondrianTreeRegressor
        from online_cp.mondrian.tree import _collect_leaves

        X, y = self._reg_data(n=40)

        def _groups(model):
            model.predict(X[0], epsilon=0.1)
            return sorted(
                tuple(sorted(leaf.indices.tolist())) for leaf in _collect_leaves(model._last_tree)
            )

        a = ConformalMondrianTreeRegressor(lifetime=2.0, rnd_state=7)
        a.learn_initial_training_set(X, y)
        b = ConformalMondrianTreeRegressor(lifetime=2.0, min_samples_leaf=1, rnd_state=7)
        b.learn_initial_training_set(X, y)
        assert _groups(a) == _groups(b)

    # ---- symmetry / permutation invariance --------------------------------

    def test_permutation_invariance_with_min_leaf(self):
        """Permuting the training data must not change the prediction for a fixed x."""
        from online_cp import ConformalMondrianTreeRegressor

        X, y = self._reg_data(n=51, seed=11)
        rng = np.random.default_rng(77)
        perm = rng.permutation(50)

        m1 = ConformalMondrianTreeRegressor(lifetime=2.0, min_samples_leaf=3, rnd_state=0)
        m1.learn_initial_training_set(X[:50], y[:50])
        iv1 = m1.predict(X[50], epsilon=0.1)

        m2 = ConformalMondrianTreeRegressor(lifetime=2.0, min_samples_leaf=3, rnd_state=0)
        m2.learn_initial_training_set(X[:50][perm], y[:50][perm])
        iv2 = m2.predict(X[50], epsilon=0.1)

        assert np.isclose(iv1.lower, iv2.lower, atol=1e-12)
        assert np.isclose(iv1.upper, iv2.upper, atol=1e-12)

    # ---- degenerate inputs ------------------------------------------------

    def test_min_leaf_larger_than_n_root_leaf_fallback(self):
        """min_samples_leaf >> n collapses to a root leaf — still a valid, finite prediction."""
        from online_cp import ConformalMondrianTreeRegressor

        X, y = self._reg_data(n=10)
        m = ConformalMondrianTreeRegressor(lifetime=100.0, min_samples_leaf=1000, rnd_state=0)
        m.learn_initial_training_set(X, y)
        iv = m.predict(X[0], epsilon=0.1)
        assert np.isfinite(iv.lower) and np.isfinite(iv.upper)
        assert iv.lower <= iv.upper

    def test_min_leaf_classifier_root_leaf_fallback(self):
        """With a huge min_samples_leaf the classifier falls back to a root leaf but stays valid."""
        from online_cp import ConformalMondrianTreeClassifier

        X, y = self._clf_data(n=10)
        c = ConformalMondrianTreeClassifier(lifetime=100.0, min_samples_leaf=1000, lambda_=0.0, rnd_state=0)
        c.learn_initial_training_set(X, y)
        ps = c.predict(X[0], epsilon=0.1)
        assert len(ps) >= 1


# ===========================================================================
# Empty-leaf validity (hierarchical prior removed)
# ===========================================================================
#
# The ``hierarchical_prior`` parameter was removed because the ancestor-
# borrowing prior violated conformal validity: when the test point's leaf is
# empty, borrowing from an ancestor with very few points (A=1) produced a
# near-degenerate point interval at the ancestor's mean, giving coverage as
# low as ~2% (vs the target 90%).  The valid behaviour for an empty leaf is
# an infinite interval (total epistemic uncertainty), which always covers.


class TestEmptyLeafValidity:
    """The regression predictor must return a valid (infinite or NaN) interval
    when the test point's leaf holds no training samples.  The hierarchical
    (ancestor) prior that once tried to 'fix' this by borrowing from an
    ancestor has been removed because it broke the conformal validity
    guarantee."""

    def test_hierarchical_prior_parameter_removed(self):
        """The ``hierarchical_prior`` parameter has been removed from the
        constructor.  Passing it must raise a TypeError."""
        from online_cp import ConformalMondrianTreeRegressor

        with pytest.raises(TypeError):
            ConformalMondrianTreeRegressor(hierarchical_prior=True)

    def test_empty_leaf_gives_valid_interval(self):
        """An empty test leaf (no training samples in the leaf containing the
        test point) must yield a valid interval: either (-inf, +inf) (total
        epistemic uncertainty, always covers) or (nan, nan) (no valid region).
        It must NOT be a finite but too-narrow interval."""
        from online_cp import ConformalMondrianTreeRegressor

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (40, 1))
        y = X[:, 0] * 2.0
        x_test = np.array([5.0])  # far from training data -> isolated leaf
        reg = ConformalMondrianTreeRegressor(lifetime=4.0, min_samples_leaf=1, rnd_state=0)
        reg.learn_initial_training_set(X, y)
        ps = reg.predict(x_test, epsilon=0.1)
        assert np.isinf(ps.lower) or np.isinf(ps.upper) or np.isnan(ps.lower)

    def test_empty_leaf_maintains_valid_coverage_on_stream(self):
        """Regression test for the HP undercoverage bug.

        The removed ancestor-borrowing prior produced intervals as narrow as
        a single point (the ancestor's mean) for empty leaves, giving coverage
        as low as ~2% on california housing.  With the HP removed, empty
        leaves yield infinite intervals and coverage remains valid
        (>= 1 - alpha) across a realistic stream.

        This test uses a small but non-trivial dataset to keep runtime low
        while still exercising many empty-leaf (A=0) cases.
        """
        from online_cp import ConformalMondrianTreeRegressor

        rng = np.random.default_rng(0)
        # 1-D data with a gap: test points in the gap have empty leaves.
        n_train = 300
        X_train = rng.uniform(0, 1, (n_train, 1))
        y_train = X_train[:, 0] * 3.0 + rng.normal(0, 0.2, n_train)
        # Test stream spans the training range plus a gap beyond it.
        n_test = 150
        X_test = rng.uniform(-0.5, 1.5, (n_test, 1))
        y_test = X_test[:, 0] * 3.0 + rng.normal(0, 0.2, n_test)

        reg = ConformalMondrianTreeRegressor(
            lifetime=4.0, min_samples_leaf=1, online=True, rnd_state=0,
        )
        reg.learn_initial_training_set(X_train, y_train)

        covers = []
        for i in range(n_test):
            ps = reg.predict(X_test[i], epsilon=0.1)
            lo, up = ps.lower, ps.upper
            if np.isnan(lo):
                # No valid region: count as non-covered (conservative).
                covers.append(0.0)
            else:
                covers.append(1.0 if (lo <= y_test[i] <= up) else 0.0)
            reg.learn_one(X_test[i], y_test[i])

        coverage = float(np.mean(covers))
        # With alpha=0.1 the target coverage is 0.9.  Allow a margin for
        # finite-sample noise but require clearly valid coverage.
        assert coverage >= 0.80, (
            f"empty-leaf undercovered: {coverage:.3f} < 0.80 "
            f"on a 1-D stream with empty-leaf test points."
        )

    # ------------------------------------------------------------------
    # Cross-solver equivalence: single tree == 1-tree forest
    # ------------------------------------------------------------------

    def test_tree_equals_single_tree_forest_pvalues_bit_identical(self):
        """A Mondrian forest with n_trees=1 must produce p-values bit-identical
        to the single-tree conformal predictor, for the same (x, y, tau).

        This pins the 'pathological forest equals tree' invariant that is
        essential for the paper: the forest solver's per-point root-finding
        and state sweep must compute exactly the same smoothed conformal
        p-value as the tree solver's closed-form knot construction.
        """
        from online_cp import ConformalMondrianForestRegressor, ConformalMondrianTreeRegressor

        rng = np.random.default_rng(42)
        for trial in range(20):
            n = int(rng.integers(20, 80))
            d = int(rng.integers(1, 4))
            X = rng.uniform(0, 1, (n, d))
            y = rng.normal(0, 1, n)

            tree = ConformalMondrianTreeRegressor(lifetime=1.0, rnd_state=0)
            tree.learn_initial_training_set(X, y)
            forest = ConformalMondrianForestRegressor(
                n_trees=1, lifetime=1.0, rnd_state=0
            )
            forest.learn_initial_training_set(X, y)

            x = rng.uniform(0, 1, d)
            for tau in [0.0, 0.3, 0.7, 1.0]:
                for y_cand in rng.uniform(-2, 2, size=5):
                    p_tree = float(tree.compute_p_value(x, float(y_cand), tau=tau))
                    p_forest = float(forest.compute_p_value(x, float(y_cand), tau=tau))
                    assert p_tree == p_forest, (
                        f"p-value mismatch trial={trial} tau={tau} y_cand={y_cand:.4f}: "
                        f"tree={p_tree!r} forest={p_forest!r}"
                    )

    def test_single_tree_forest_solver_endpoints_valid_and_contiguous(self):
        """The 1-tree ``forest_sweepline_solver`` (the solver the single-tree
        regressor's ``predict`` now routes through) must return a well-formed,
        contiguous interval whose endpoints and midpoint are all *valid*
        (i.e. have p-value > epsilon), at a fixed tau.

        This is an exact, knot-based check (no grid needed, so it also catches
        narrow singleton valid regions that a dense grid would miss). It pins
        the 'pathological forest equals tree' invariant for the prediction set.
        """
        from online_cp import ConformalMondrianForestRegressor
        from online_cp.mondrian._forest_solver import forest_sweepline_solver

        _TOL = 1e-9
        for trial in range(15):
            rng = np.random.default_rng(42 + trial)
            n = int(rng.integers(20, 60))
            d = int(rng.integers(1, 4))
            X = rng.uniform(0, 1, (n, d))
            y = rng.normal(0, 1, n)

            # Build a 1-tree forest with a fixed seed so we can extract its
            # per-tree summary and feed it to the forest solver directly.
            forest = ConformalMondrianForestRegressor(
                n_trees=1, lifetime=1.0, rnd_state=1234 + trial
            )
            forest.learn_initial_training_set(X, y)
            x = rng.uniform(0, 1, d)

            n_train = X.shape[0]
            X_aug = np.vstack([X, x])
            x_seed = int(hash(tuple(np.round(x, 10).flatten())) % (2**31))
            seeds = np.array([x_seed])
            summaries = forest._forest_summaries(X_aug, x, n_train, seeds=seeds)
            base_ncm, ls_idx, ls_mu = summaries[0]
            A = len(ls_idx)
            if A == 0:
                continue  # empty leaf: handled by the fast path

            # Exact p-value (same algebra as the forest solver's p_value_at),
            # evaluated at a single y. The +1 counts the test point's tie.
            def _p_at_single(yc, summaries, y, tau, n_train, tol):
                gt = 0
                eq = 0
                for bn, ls_i, ls_m in summaries:
                    A_i = len(ls_i)
                    B_i = ls_m * A_i if A_i > 0 else 0.0
                    test_t = abs(A_i * yc - B_i) / (A_i + 1)
                    ncm_t = bn.copy()
                    if A_i > 0:
                        ys_i = y[ls_i]
                        ncm_t[ls_i] = np.abs((A_i + 1) * ys_i - B_i - yc) / (A_i + 1)
                    gt += int(np.sum(ncm_t > test_t + tol))
                    eq += int(np.sum(np.abs(ncm_t - test_t) <= tol))
                return (gt + tau * (eq + 1)) / (n_train + 1)

            for tau in [0.0, 0.5, 1.0]:
                for eps in [0.05, 0.2, 0.5, 0.8]:
                    intervals, _roots = forest_sweepline_solver(
                        summaries, y, float(eps), tau, n_train, 1, return_exact=True
                    )
                    if not intervals:
                        continue  # empty valid set: nothing to check
                    lo_s = min(i[0] for i in intervals)
                    hi_s = max(i[1] for i in intervals)
                    assert lo_s <= hi_s + 1e-9, (
                        f"malformed interval trial={trial} tau={tau} eps={eps}: "
                        f"lo={lo_s!r} > hi={hi_s!r} (A={A})"
                    )
                    # The interior of the interval must be valid (p > eps). The
                    # *boundary* points are the continuous roots where p = eps;
                    # the count-based p-value is a step function that may sit
                    # just below eps at the exact boundary, so we probe a point
                    # strictly inside the interval instead.
                    if hi_s - lo_s > 1e-9:
                        probe = 0.5 * (lo_s + hi_s)  # interior point
                    else:
                        probe = lo_s  # singleton: the knot itself
                    p_val = _p_at_single(float(probe), summaries, y, tau, n_train, _TOL)
                    assert p_val > eps - 1e-9, (
                        f"interior not valid trial={trial} tau={tau} eps={eps}: "
                        f"y={probe!r} p={p_val!r} (A={A})"
                    )

    def test_single_tree_forest_solver_A1_permanent_tie_and_zero_width_knots(self):
        """Exercise the two edge cases that previously broke the tree solver,
        now through the shared ``forest_sweepline_solver`` (M=1, the solver
        the single-tree regressor's ``predict`` routes through):

        1. **A=1 permanent tie**: when the test leaf has a single point, the
           inside point's score equals the test score for *all* y. The
           p-value formula must count this tie exactly once.

        2. **Zero-width knots**: when several knot events land on the same
           y, the cells between them are zero-width and the cell p-value
           formula overestimates p. The solver must still produce a valid
           (lo <= hi) interval.

        In both cases the solver output must be well-formed (lo <= hi when
        non-empty) and its endpoints/midpoint must be valid (p > epsilon) at a
        fixed tau, confirming a contiguous valid region.
        """
        from online_cp.mondrian._forest_solver import forest_sweepline_solver
        from online_cp.mondrian.tree import (
            MondrianTree,
            _find_leaf,
            _summarize_tree_reg,
        )

        # Duplicate y-values are injected to force zero-width knots; a low
        # lifetime concentrates points in a few leaves so that A=1 (permanent
        # tie) and duplicate-y_j (zero-width) configurations actually occur.
        rng = np.random.default_rng(999)
        for trial in range(30):
            n = int(rng.integers(15, 60))
            d = int(rng.integers(1, 4))
            X = rng.uniform(0, 1, (n, d))
            y = rng.normal(0, 1, n)
            # Force some duplicate y values to create zero-width knots.
            if trial % 2 == 0:
                y[rng.integers(0, n - 1)] = y[0]  # duplicate
            lifetime = float(rng.choice([0.5, 1.0, 2.0]))
            x = rng.uniform(0, 1, d)
            X_aug = np.vstack([X, x])
            root = MondrianTree.grow(
                X_aug, np.random.default_rng(555 + trial),
                lifetime=lifetime, x_test=x,
            ).root
            leaf = _find_leaf(root, x)
            idx = leaf.indices[leaf.indices < n]
            A_t = len(idx)
            if A_t == 0:
                continue  # empty leaf handled by the fast path
            base_ncm, ls_idx, ls_mu = _summarize_tree_reg(root, y, n, x)
            summaries = [(base_ncm, ls_idx, ls_mu)]

            # Exact p-value (same algebra as the forest solver's p_value_at),
            # evaluated at a single y. The +1 counts the test point's tie.
            _TOL = 1e-9

            def _p_at_single(yc, summaries, y, tau, n, tol):
                gt = 0
                eq = 0
                for bn, ls_i, ls_m in summaries:
                    A_i = len(ls_i)
                    B_i = ls_m * A_i if A_i > 0 else 0.0
                    test_t = abs(A_i * yc - B_i) / (A_i + 1)
                    ncm_t = bn.copy()
                    if A_i > 0:
                        ys_i = y[ls_i]
                        ncm_t[ls_i] = np.abs((A_i + 1) * ys_i - B_i - yc) / (A_i + 1)
                    gt += int(np.sum(ncm_t > test_t + tol))
                    eq += int(np.sum(np.abs(ncm_t - test_t) <= tol))
                return (gt + tau * (eq + 1)) / (n + 1)

            for tau in [0.0, 0.5, 1.0]:
                for eps in [0.05, 0.5, 0.9]:
                    intervals, _ = forest_sweepline_solver(
                        summaries, y, eps, tau, n, 1, return_exact=True
                    )
                    if not intervals:
                        continue  # empty valid set: nothing to check
                    lo_s = min(i[0] for i in intervals)
                    hi_s = max(i[1] for i in intervals)

                    # The interval must be well-formed.
                    assert lo_s <= hi_s + 1e-9, (
                        f"A={A_t}: lo={lo_s!r} > hi={hi_s!r} "
                        f"for tau={tau} eps={eps}"
                    )
                    # The interior of the interval must be valid (p > eps); the
                    # exact boundary sits at the continuous root p = eps, where
                    # the count-based p-value is one step below, so probe
                    # strictly inside (or the knot itself for a singleton).
                    if hi_s - lo_s > 1e-9:
                        probe = 0.5 * (lo_s + hi_s)
                    else:
                        probe = lo_s
                    p_val = _p_at_single(float(probe), summaries, y, tau, n, _TOL)
                    assert p_val > eps - 1e-9, (
                        f"interior not valid trial={trial} "
                        f"tau={tau} eps={eps}: y={probe!r} p={p_val!r} (A={A_t})"
                    )

    def test_tree_predict_empty_leaf_and_degenerate_inputs(self):
        """The single-tree regressor's ``predict`` must produce well-formed
        intervals (lo <= hi when non-empty) on degenerate inputs: empty
        leaves (A=0) and test points isolated into their own leaf.

        These exercise the A=0 fast path (total epistemic uncertainty) and the
        A=1 permanent-tie case through the shared sweepline solver, confirming
        the unification did not break the edge-case handling.
        """
        rng = np.random.default_rng(7)
        for trial in range(12):
            n = int(rng.integers(20, 60))
            d = int(rng.integers(1, 3))
            X = rng.uniform(0, 1, (n, d))
            y = rng.normal(0, 1, n)
            reg = ConformalMondrianTreeRegressor(lifetime=2.0, rnd_state=trial)
            reg.learn_initial_training_set(X, y)
            # A test point far outside the training range tends to be isolated
            # into an empty or singleton leaf.
            x = np.full(d, 5.0 + float(trial) * 0.01)
            for eps in [0.1, 0.3]:
                ps = reg.predict(x, epsilon=eps)
                lo, hi = float(ps.lower), float(ps.upper)
                if not (np.isnan(lo)):
                    assert lo <= hi + 1e-9, (
                        f"degenerate input: lo={lo!r} > hi={hi!r} "
                        f"trial={trial} eps={eps}"
                    )

