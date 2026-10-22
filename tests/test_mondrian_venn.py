"""Tests for MondrianVennPredictor (src/online_cp/venn.py).

Covers:
- Functional correctness (binary + multiclass, valid VennPrediction)
- Order-invariance / permutation symmetry — the key Venn bag-function property
- Batch vs incremental state equivalence
- Leaf-size control: min_samples_leaf guarantee, identity at default 1,
  root-leaf fallback, param validation
- Label-space management (fixed space, inference, rejection of unknowns)
- Integration with progressive_val_venn
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from online_cp.venn import MondrianVennPredictor, VennPrediction, VennPredictor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_data():
    X, y = make_blobs(n_samples=60, n_features=2, centers=2, random_state=0)
    y = (y > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    X, y = make_blobs(n_samples=120, n_features=3, centers=3, random_state=0)
    return X, y


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestMondrianVennPredictorFunctional:

    def test_isinstance_venn_predictor(self):
        assert isinstance(MondrianVennPredictor(rnd_state=0), VennPredictor)

    def test_binary_predict_type(self, binary_data):
        X, y = binary_data
        mvp = MondrianVennPredictor(rnd_state=42)
        mvp.learn_initial_training_set(X[:40], y[:40])
        pred = mvp.predict(X[40])
        assert isinstance(pred, VennPrediction)

    def test_binary_predict_p_in_unit_interval(self, binary_data):
        X, y = binary_data
        mvp = MondrianVennPredictor(rnd_state=42)
        mvp.learn_initial_training_set(X[:40], y[:40])
        pred = mvp.predict(X[40])
        assert 0.0 <= pred.p0 <= 1.0
        assert 0.0 <= pred.p1 <= 1.0

    def test_multiclass_predict_rows_sum_to_one(self, multiclass_data):
        X, y = multiclass_data
        mvp = MondrianVennPredictor(rnd_state=42)
        mvp.learn_initial_training_set(X[:80], y[:80])
        pred = mvp.predict(X[80])
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)

    def test_multiclass_predict_probs_in_unit_interval(self, multiclass_data):
        X, y = multiclass_data
        mvp = MondrianVennPredictor(rnd_state=42)
        mvp.learn_initial_training_set(X[:80], y[:80])
        pred = mvp.predict(X[80])
        assert np.all(pred.probs >= 0.0) and np.all(pred.probs <= 1.0)

    def test_multiclass_probs_shape(self, multiclass_data):
        X, y = multiclass_data
        mvp = MondrianVennPredictor(rnd_state=0)
        mvp.learn_initial_training_set(X[:80], y[:80])
        pred = mvp.predict(X[80])
        n_labels = len(np.unique(y[:80]))
        assert pred.probs.shape == (n_labels, n_labels)

    def test_empty_training_set_returns_uniform(self):
        mvp = MondrianVennPredictor(rnd_state=0)
        pred = mvp.predict(np.array([1.0, 2.0]))
        assert isinstance(pred, VennPrediction)
        assert pred.p0 == 0.5 and pred.p1 == 0.5

    def test_single_training_point(self):
        mvp = MondrianVennPredictor(rnd_state=0)
        mvp.learn_one(np.array([1.0, 2.0]), 1)
        pred = mvp.predict(np.array([1.5, 2.5]))
        assert isinstance(pred, VennPrediction)
        assert 0.0 <= pred.p0 <= 1.0
        assert 0.0 <= pred.p1 <= 1.0

    def test_multiple_predictions_consistent(self, binary_data):
        X, y = binary_data
        mvp = MondrianVennPredictor(rnd_state=7)
        mvp.learn_initial_training_set(X[:30], y[:30])
        for i in range(30, 40):
            pred = mvp.predict(X[i])
            assert 0.0 <= pred.p0 <= 1.0
            assert 0.0 <= pred.p1 <= 1.0
            mvp.learn_one(X[i], y[i])


# ---------------------------------------------------------------------------
# Permutation symmetry (the key Venn bag-function property)
# ---------------------------------------------------------------------------


class TestMondrianVennPredictorSymmetry:

    def test_permutation_invariance_binary(self, binary_data):
        X, y = binary_data
        perm = np.random.default_rng(999).permutation(40)

        mvp1 = MondrianVennPredictor(lifetime=1.0, rnd_state=42)
        mvp1.learn_initial_training_set(X[:40], y[:40])
        pred1 = mvp1.predict(X[40])

        mvp2 = MondrianVennPredictor(lifetime=1.0, rnd_state=42)
        mvp2.learn_initial_training_set(X[perm], y[perm])
        pred2 = mvp2.predict(X[40])

        assert abs(pred1.p0 - pred2.p0) < 1e-12
        assert abs(pred1.p1 - pred2.p1) < 1e-12

    def test_permutation_invariance_multiclass(self, multiclass_data):
        X, y = multiclass_data
        perm = np.random.default_rng(77).permutation(80)

        mvp1 = MondrianVennPredictor(lifetime=1.5, rnd_state=0)
        mvp1.learn_initial_training_set(X[:80], y[:80])
        pred1 = mvp1.predict(X[80])

        mvp2 = MondrianVennPredictor(lifetime=1.5, rnd_state=0)
        mvp2.learn_initial_training_set(X[perm], y[perm])
        pred2 = mvp2.predict(X[80])

        np.testing.assert_array_equal(pred1.label_space, pred2.label_space)
        np.testing.assert_allclose(pred1.probs, pred2.probs, atol=1e-12)

    def test_permutation_invariance_with_min_samples_leaf(self, binary_data):
        X, y = binary_data
        perm = np.random.default_rng(77).permutation(50)

        mvp1 = MondrianVennPredictor(lifetime=2.0, min_samples_leaf=3, rnd_state=0)
        mvp1.learn_initial_training_set(X[:50], y[:50])
        pred1 = mvp1.predict(X[50])

        mvp2 = MondrianVennPredictor(lifetime=2.0, min_samples_leaf=3, rnd_state=0)
        mvp2.learn_initial_training_set(X[perm], y[perm])
        pred2 = mvp2.predict(X[50])

        assert abs(pred1.p0 - pred2.p0) < 1e-12
        assert abs(pred1.p1 - pred2.p1) < 1e-12

    def test_permutation_invariance_with_max_depth(self, binary_data):
        X, y = binary_data
        perm = np.random.default_rng(55).permutation(40)

        mvp1 = MondrianVennPredictor(lifetime=5.0, max_depth=3, rnd_state=1)
        mvp1.learn_initial_training_set(X[:40], y[:40])
        pred1 = mvp1.predict(X[40])

        mvp2 = MondrianVennPredictor(lifetime=5.0, max_depth=3, rnd_state=1)
        mvp2.learn_initial_training_set(X[perm], y[perm])
        pred2 = mvp2.predict(X[40])

        assert abs(pred1.p0 - pred2.p0) < 1e-12
        assert abs(pred1.p1 - pred2.p1) < 1e-12


# ---------------------------------------------------------------------------
# Leaf-size control knobs
# ---------------------------------------------------------------------------


class TestMondrianVennPredictorLeafSize:

    def test_min_samples_leaf_guarantee(self, binary_data):
        """Every leaf in the augmented tree has >= min_samples_leaf points."""
        from online_cp.mondrian.tree import _collect_leaves, _sample_mondrian_tree

        X, y = binary_data
        n = 40
        x_test = X[40]
        X_aug = np.vstack([X[:n], x_test.reshape(1, -1)])
        indices = np.arange(n + 1)
        m = 3

        rng = np.random.default_rng(42)
        root = _sample_mondrian_tree(
            rng, X_aug, indices,
            parent_time=0.0, lifetime=2.0,
            min_samples_leaf=m,
        )
        leaves = _collect_leaves(root)
        for leaf in leaves:
            assert len(leaf.indices) >= m, (
                f"Leaf has {len(leaf.indices)} points, expected >= {m}"
            )

    def test_min_samples_leaf_default_is_identity(self, binary_data):
        """min_samples_leaf=1 produces the exact same tree as the default."""
        from online_cp.mondrian.tree import _collect_leaves, _sample_mondrian_tree

        X, _ = binary_data
        indices = np.arange(len(X))

        rng1 = np.random.default_rng(123)
        root1 = _sample_mondrian_tree(rng1, X, indices, 0.0, 1.0)
        leaves1 = sorted([tuple(sorted(leaf.indices.tolist())) for leaf in _collect_leaves(root1)])

        rng2 = np.random.default_rng(123)
        root2 = _sample_mondrian_tree(rng2, X, indices, 0.0, 1.0, min_samples_leaf=1)
        leaves2 = sorted([tuple(sorted(leaf.indices.tolist())) for leaf in _collect_leaves(root2)])

        assert leaves1 == leaves2

    def test_large_min_samples_leaf_root_leaf_fallback(self):
        """min_samples_leaf >> n collapses to a root leaf — valid prediction, no crash."""
        X, y = make_blobs(n_samples=10, n_features=2, centers=2, random_state=0)
        y = (y > 0).astype(int)
        mvp = MondrianVennPredictor(lifetime=100.0, min_samples_leaf=1000, rnd_state=0)
        mvp.learn_initial_training_set(X[:8], y[:8])
        pred = mvp.predict(X[8])
        assert isinstance(pred, VennPrediction)
        assert 0.0 <= pred.p0 <= 1.0
        assert 0.0 <= pred.p1 <= 1.0

    def test_min_samples_leaf_prediction_not_singleton(self, binary_data):
        """With min_samples_leaf=2, p0/p1 should not be forced 0 or 1 when
        the leaf containing the test point has mixed labels."""
        X, y = binary_data
        found_non_extreme = False
        for seed in range(20):
            mvp = MondrianVennPredictor(
                lifetime=0.5, min_samples_leaf=2, rnd_state=seed
            )
            mvp.learn_initial_training_set(X[:50], y[:50])
            pred = mvp.predict(X[50])
            if 0.0 < pred.p0 < 1.0 or 0.0 < pred.p1 < 1.0:
                found_non_extreme = True
                break
        assert found_non_extreme, "Expected at least one non-extreme prediction with min_samples_leaf=2"

    def test_param_validation_min_samples_leaf_zero(self):
        with pytest.raises(ValueError, match="min_samples_leaf"):
            MondrianVennPredictor(min_samples_leaf=0)

    def test_param_validation_min_samples_leaf_negative(self):
        with pytest.raises(ValueError, match="min_samples_leaf"):
            MondrianVennPredictor(min_samples_leaf=-1)

    def test_param_validation_max_depth_negative(self):
        with pytest.raises(ValueError, match="max_depth"):
            MondrianVennPredictor(max_depth=-1)

    def test_param_validation_max_depth_float(self):
        with pytest.raises(ValueError, match="max_depth"):
            MondrianVennPredictor(max_depth=1.5)

    def test_max_depth_zero_root_leaf(self, binary_data):
        """max_depth=0 → root is immediately a leaf → valid single-category prediction."""
        X, y = binary_data
        mvp = MondrianVennPredictor(lifetime=10.0, max_depth=0, rnd_state=0)
        mvp.learn_initial_training_set(X[:30], y[:30])
        pred = mvp.predict(X[30])
        assert isinstance(pred, VennPrediction)
        assert 0.0 <= pred.p0 <= 1.0
        assert 0.0 <= pred.p1 <= 1.0


# ---------------------------------------------------------------------------
# Batch vs incremental equivalence
# ---------------------------------------------------------------------------


class TestMondrianVennPredictorBatchIncremental:

    def test_batch_vs_incremental_state(self, binary_data):
        X, y = binary_data

        mvp_batch = MondrianVennPredictor(rnd_state=0)
        mvp_batch.learn_initial_training_set(X[:10], y[:10])

        mvp_inc = MondrianVennPredictor(rnd_state=0)
        for i in range(10):
            mvp_inc.learn_one(X[i], y[i])

        np.testing.assert_array_equal(mvp_batch.X, mvp_inc.X)
        np.testing.assert_array_equal(mvp_batch.y, mvp_inc.y)

    def test_batch_vs_incremental_prediction(self, binary_data):
        X, y = binary_data

        mvp_batch = MondrianVennPredictor(rnd_state=0)
        mvp_batch.learn_initial_training_set(X[:10], y[:10])

        mvp_inc = MondrianVennPredictor(rnd_state=0)
        for i in range(10):
            mvp_inc.learn_one(X[i], y[i])

        pred_batch = mvp_batch.predict(X[10])
        pred_inc = mvp_inc.predict(X[10])
        assert abs(pred_batch.p0 - pred_inc.p0) < 1e-12
        assert abs(pred_batch.p1 - pred_inc.p1) < 1e-12


# ---------------------------------------------------------------------------
# Label-space management
# ---------------------------------------------------------------------------


class TestMondrianVennPredictorLabelSpace:

    def test_fixed_label_space_learn_one_rejects_unknown(self):
        mvp = MondrianVennPredictor(label_space=[0, 1], rnd_state=0)
        with pytest.raises(ValueError):
            mvp.learn_one(np.array([1.0, 2.0]), 2)

    def test_fixed_label_space_batch_rejects_unknown(self):
        X = np.random.randn(10, 2)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 2])
        mvp = MondrianVennPredictor(label_space=[0, 1], rnd_state=0)
        with pytest.raises(ValueError):
            mvp.learn_initial_training_set(X, y)

    def test_inferred_label_space_binary(self, binary_data):
        X, y = binary_data
        mvp = MondrianVennPredictor(rnd_state=0)
        mvp.learn_initial_training_set(X[:20], y[:20])
        assert set(mvp.label_space.tolist()) == set(np.unique(y[:20]).tolist())

    def test_inferred_label_space_grows_with_learn_one(self):
        mvp = MondrianVennPredictor(rnd_state=0)
        mvp.learn_one(np.array([0.0, 0.0]), 0)
        assert 0 in mvp.label_space
        mvp.learn_one(np.array([1.0, 1.0]), 1)
        assert 1 in mvp.label_space

    def test_fixed_label_space_empty_prediction_uses_it(self):
        mvp = MondrianVennPredictor(label_space=[0, 1, 2], rnd_state=0)
        pred = mvp.predict(np.array([0.0, 0.0]))
        assert pred.probs.shape == (3, 3)
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Integration with progressive_val_venn
# ---------------------------------------------------------------------------


class TestMondrianVennPredictorProgressive:

    def test_progressive_val_venn_runs(self, binary_data):
        from online_cp.evaluate import progressive_val_venn

        X, y = binary_data
        mvp = MondrianVennPredictor(rnd_state=42)
        mvp.learn_initial_training_set(X[:10], y[:10])
        # Should not raise; returns a metric object
        result = progressive_val_venn(mvp, X[10:30], y[10:30])
        assert result is not None

    def test_progressive_val_venn_multiclass(self, multiclass_data):
        from online_cp.evaluate import progressive_val_venn

        X, y = multiclass_data
        mvp = MondrianVennPredictor(rnd_state=0)
        mvp.learn_initial_training_set(X[:20], y[:20])
        result = progressive_val_venn(mvp, X[20:40], y[20:40])
        assert result is not None
