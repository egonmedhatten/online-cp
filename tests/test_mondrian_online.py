"""Tests for the online ``ExtendMondrianBlock`` update (``MondrianTree.extend``).

Phase 1 of the online Mondrian work: verify that projecting points in one at a
time (a) produces a structurally valid tree, (b) never mutates the source tree
(so the transductive ``predict`` can score a temporary extension and discard it),
and (c) yields a tree with the **same law** as a batch ``grow`` — the projectivity
property (Roy & Teh 2009) that underpins online validity.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.datasets import make_moons

from online_cp import (
    ConformalMondrianForestClassifier,
    ConformalMondrianForestRegressor,
    ConformalMondrianTreeClassifier,
    ConformalMondrianTreeRegressor,
    ErrorRate,
    progressive_val,
)
from online_cp.mondrian import MondrianTree
from online_cp.mondrian.tree import _MondrianNode


def _build_by_extend(X, rng, lifetime):
    """Grow a single-leaf seed then project the remaining points in one by one."""
    tree = MondrianTree.grow(X[:1], rng, lifetime=lifetime)
    for i in range(1, len(X)):
        tree = tree.extend(X[i], rng)
    return tree


def _signature(node: _MondrianNode):
    """A hashable structural snapshot of a subtree (for mutation checks)."""
    if node.is_leaf():
        return ("leaf", tuple(sorted(int(i) for i in node.indices)),
                tuple(np.round(node.lower_bounds, 9)),
                tuple(np.round(node.upper_bounds, 9)))
    return ("split", node.split_dim, round(float(node.split_loc), 9),
            _signature(node.left), _signature(node.right))


class TestExtendStructure:
    def test_all_points_covered_exactly_once(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50, 3))
        tree = _build_by_extend(X, np.random.default_rng(1), lifetime=3.0)
        covered = sorted(int(j) for leaf in tree.collect_leaves() for j in leaf.indices)
        assert covered == list(range(len(X)))

    def test_find_leaf_membership(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (40, 2))
        tree = _build_by_extend(X, np.random.default_rng(3), lifetime=3.0)
        for i in range(len(X)):
            leaf = tree.find_leaf(X[i])
            assert i in {int(j) for j in leaf.indices}

    def test_extend_does_not_mutate_source(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (30, 2))
        tree = _build_by_extend(X, np.random.default_rng(5), lifetime=3.0)
        before = _signature(tree.root)
        x_new = rng.uniform(0, 1, 2)
        _ = tree.extend(x_new, np.random.default_rng(6))
        after = _signature(tree.root)
        assert before == after
        assert tree.X.shape[0] == len(X)  # source X unchanged too

    def test_reproducible_for_fixed_seed(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (35, 2))
        t1 = _build_by_extend(X, np.random.default_rng(11), lifetime=3.0)
        t2 = _build_by_extend(X, np.random.default_rng(11), lifetime=3.0)
        assert _signature(t1.root) == _signature(t2.root)

    def test_string_lifetime_rejected(self):
        rng = np.random.default_rng(8)
        X = rng.uniform(0, 1, (20, 2))
        tree = MondrianTree.grow(X, rng, lifetime=2.0)
        tree.lifetime = "sqrt_n"  # simulate an unresolved/auto-tuned lifetime
        with pytest.raises(ValueError, match="fixed float lifetime"):
            tree.extend(X[0], rng)


class TestExtendDistribution:
    """Projectivity: extend-built trees share the batch law."""

    def test_leaf_count_matches_batch(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (60, 2))
        lifetime = 4.0
        n = 300

        batch = np.array([
            len(MondrianTree.grow(X, np.random.default_rng(s), lifetime=lifetime))
            for s in range(n)
        ])
        extend = np.array([
            len(_build_by_extend(X, np.random.default_rng(s), lifetime=lifetime))
            for s in range(n, 2 * n)
        ])

        # Means agree closely and a two-sample KS test does not reject equality.
        assert abs(batch.mean() - extend.mean()) < 0.15 * batch.mean()
        assert stats.ks_2samp(batch, extend).pvalue > 0.02


def _clf_stream_error(seed, online):
    """Progressive-validation error rate of the (online or batch) tree classifier."""
    X, y = make_moons(n_samples=200, noise=0.3, random_state=seed)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    clf = ConformalMondrianTreeClassifier(
        lifetime=3.0, label_space=np.array([0, 1]), rnd_state=seed, online=online
    )
    clf.learn_initial_training_set(X[:70], y[:70])
    metric = progressive_val(clf, X[70:], y[70:], epsilon=0.1, metric=ErrorRate())
    return metric.get()


class TestOnlineClassifier:
    def test_predict_and_learn_grow_the_tree(self):
        X, y = make_moons(n_samples=120, noise=0.3, random_state=0)
        clf = ConformalMondrianTreeClassifier(
            lifetime=3.0, label_space=np.array([0, 1]), rnd_state=0, online=True
        )
        clf.learn_initial_training_set(X[:80], y[:80])
        assert clf._online_tree.X.shape[0] == 80
        Gamma = clf.predict(X[80], epsilon=0.1)
        assert set(Gamma.elements).issubset({0, 1})
        clf.learn_one(X[80], y[80])
        assert clf._online_tree.X.shape[0] == 81  # persistent tree grew by one

    def test_coverage_tracks_epsilon(self):
        errs = [_clf_stream_error(s, online=True) for s in range(15)]
        assert 0.06 < float(np.mean(errs)) < 0.15  # ε = 0.1

    def test_coverage_matches_batch(self):
        online = float(np.mean([_clf_stream_error(s, True) for s in range(10)]))
        batch = float(np.mean([_clf_stream_error(s, False) for s in range(10)]))
        assert abs(online - batch) < 0.04

    def test_save_load_round_trip(self, tmp_path):
        X, y = make_moons(n_samples=120, noise=0.3, random_state=1)
        clf = ConformalMondrianTreeClassifier(
            lifetime=3.0, label_space=np.array([0, 1]), rnd_state=1, online=True
        )
        clf.learn_initial_training_set(X[:80], y[:80])
        before = clf.predict(X[80], epsilon=0.1)
        path = tmp_path / "online_clf.joblib"
        clf.save(str(path))
        loaded = ConformalMondrianTreeClassifier.load(str(path))
        assert loaded.online is True
        after = loaded.predict(X[80], epsilon=0.1)
        np.testing.assert_array_equal(before.elements, after.elements)

    def test_rejects_string_lifetime(self):
        with pytest.raises(ValueError, match="fixed float lifetime"):
            ConformalMondrianTreeClassifier(lifetime="sqrt_n", online=True)

    def test_rejects_max_depth(self):
        with pytest.raises(ValueError, match="max_depth=None"):
            ConformalMondrianTreeClassifier(max_depth=5, online=True)


def _reg_stream_error(seed, online, n_total=160, n_train=60):
    r = np.random.default_rng(seed)
    X = r.uniform(-3, 3, (n_total, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * r.normal(size=n_total)
    reg = ConformalMondrianTreeRegressor(lifetime=3.0, rnd_state=seed, online=online)
    reg.learn_initial_training_set(X[:n_train], y[:n_train])
    metric = progressive_val(reg, X[n_train:], y[n_train:], epsilon=0.1, metric=ErrorRate())
    return metric.get()


class TestOnlineRegressor:
    def test_predict_and_learn_grow_the_tree(self):
        r = np.random.default_rng(0)
        X = r.uniform(-3, 3, (60, 2))
        y = np.sin(X[:, 0]) + X[:, 1]
        reg = ConformalMondrianTreeRegressor(lifetime=3.0, rnd_state=0, online=True)
        reg.learn_initial_training_set(X[:40], y[:40])
        assert reg._online_tree.X.shape[0] == 40
        interval = reg.predict(X[40], epsilon=0.2)
        assert interval.lower <= interval.upper
        reg.learn_one(X[40], y[40])
        assert reg._online_tree.X.shape[0] == 41

    def test_save_load_round_trip(self, tmp_path):
        r = np.random.default_rng(1)
        X = r.uniform(-3, 3, (100, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        reg = ConformalMondrianTreeRegressor(lifetime=3.0, rnd_state=1, online=True)
        reg.learn_initial_training_set(X[:70], y[:70])
        before = reg.predict(X[70], epsilon=0.2)
        path = tmp_path / "online_reg.joblib"
        reg.save(str(path))
        loaded = ConformalMondrianTreeRegressor.load(str(path))
        assert loaded.online is True
        after = loaded.predict(X[70], epsilon=0.2)
        np.testing.assert_allclose([before.lower, before.upper], [after.lower, after.upper])

    def test_rejects_string_lifetime(self):
        with pytest.raises(ValueError, match="fixed float lifetime"):
            ConformalMondrianTreeRegressor(lifetime="density", online=True)

    def test_rejects_max_depth(self):
        with pytest.raises(ValueError, match="max_depth=None"):
            ConformalMondrianTreeRegressor(max_depth=4, online=True)

    @pytest.mark.slow
    def test_coverage_tracks_epsilon(self):
        errs = [_reg_stream_error(s, online=True) for s in range(12)]
        assert 0.06 < float(np.mean(errs)) < 0.15  # ε = 0.1

    @pytest.mark.slow
    def test_coverage_matches_batch(self):
        online = float(np.mean([_reg_stream_error(s, True) for s in range(8)]))
        batch = float(np.mean([_reg_stream_error(s, False) for s in range(8)]))
        assert abs(online - batch) < 0.04


class TestSingleOmegaReuse:
    """The tree that ``predict`` scores must be the one ``learn_one`` persists.

    Only then do the per-step partitions form a single nested Mondrian process
    (one ``Ω``), which is what the online-validity proof relies on. Re-drawing a
    fresh extension in ``learn_one`` would still be marginally valid but would
    break that single-``Ω`` coupling.
    """

    def _reg(self):
        r = np.random.default_rng(0)
        X = r.uniform(-3, 3, (30, 2))
        y = np.sin(X[:, 0]) + X[:, 1]
        reg = ConformalMondrianTreeRegressor(lifetime=3.0, rnd_state=0, online=True)
        reg.learn_initial_training_set(X[:20], y[:20])
        return reg, X, y

    def test_regressor_reuses_scored_tree_via_precomputed(self):
        reg, X, y = self._reg()
        _, pc = reg.predict(X[20], epsilon=0.2, return_update=True)
        reg.learn_one(X[20], y[20], precomputed=pc)
        assert reg._online_tree is pc["online_tree"]

    def test_regressor_reuses_scored_tree_via_cache(self):
        reg, X, y = self._reg()
        _, pc = reg.predict(X[20], epsilon=0.2, return_update=True)
        reg.learn_one(X[20], y[20])  # no precomputed → internal cache
        assert reg._online_tree is pc["online_tree"]

    def test_regressor_stale_extension_falls_back(self):
        reg, X, y = self._reg()
        _, pc = reg.predict(X[20], epsilon=0.2, return_update=True)
        reg.learn_one(X[21], y[21])  # different point → must not reuse the stale tree
        assert reg._online_tree is not pc["online_tree"]
        assert np.array_equal(reg._online_tree.X[-1], X[21])

    def _clf(self):
        X, y = make_moons(n_samples=60, noise=0.3, random_state=0)
        clf = ConformalMondrianTreeClassifier(
            lifetime=3.0, label_space=np.array([0, 1]), rnd_state=0, online=True
        )
        clf.learn_initial_training_set(X[:40], y[:40])
        return clf, X, y

    def test_classifier_reuses_scored_tree_via_precomputed(self):
        clf, X, y = self._clf()
        _, pc = clf.predict(X[40], epsilon=0.1, return_update=True)
        clf.learn_one(X[40], y[40], precomputed=pc)
        assert clf._online_tree is pc["online_tree"]

    def test_classifier_reuses_scored_tree_via_cache(self):
        clf, X, y = self._clf()
        _, pc = clf.predict(X[40], epsilon=0.1, return_update=True)
        clf.learn_one(X[40], y[40])  # no precomputed → internal cache
        assert clf._online_tree is pc["online_tree"]

    def test_classifier_stale_extension_falls_back(self):
        clf, X, y = self._clf()
        _, pc = clf.predict(X[40], epsilon=0.1, return_update=True)
        clf.learn_one(X[41], y[41])
        assert clf._online_tree is not pc["online_tree"]
        assert np.array_equal(clf._online_tree.X[-1], X[41])


# ---------------------------------------------------------------------------
# Online forests (Phase 3): T persistent trees, extended one point at a time.
# ---------------------------------------------------------------------------

def _clf_forest_stream_error(seed, online, n_total=110, n_train=70):
    X, y = make_moons(n_samples=n_total, noise=0.3, random_state=seed)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    clf = ConformalMondrianForestClassifier(
        n_trees=5, lifetime=3.0, label_space=np.array([0, 1]), rnd_state=seed, online=online
    )
    clf.learn_initial_training_set(X[:n_train], y[:n_train])
    metric = progressive_val(clf, X[n_train:], y[n_train:], epsilon=0.1, metric=ErrorRate())
    return metric.get()


class TestOnlineForestClassifier:
    def _clf(self):
        X, y = make_moons(n_samples=60, noise=0.3, random_state=0)
        clf = ConformalMondrianForestClassifier(
            n_trees=5, lifetime=3.0, label_space=np.array([0, 1]), rnd_state=0, online=True
        )
        clf.learn_initial_training_set(X[:40], y[:40])
        return clf, X, y

    def test_predict_and_learn_grow_the_forest(self):
        clf, X, y = self._clf()
        assert all(t.X.shape[0] == 40 for t in clf._forest)
        Gamma = clf.predict(X[40], epsilon=0.1)
        assert set(Gamma.elements).issubset({0, 1})
        clf.learn_one(X[40], y[40])
        assert all(t.X.shape[0] == 41 for t in clf._forest)

    def test_getitem_returns_tree_views(self):
        clf, X, y = self._clf()
        clf.predict(X[40], epsilon=0.1)
        assert clf[0]._last_tree is not None
        assert len(list(clf)) == clf.n_trees

    def test_reuses_scored_forest_via_precomputed(self):
        clf, X, y = self._clf()
        _, pc = clf.predict(X[40], epsilon=0.1, return_update=True)
        clf.learn_one(X[40], y[40], precomputed=pc)
        assert clf._forest is pc["online_forest"]

    def test_reuses_scored_forest_via_cache(self):
        clf, X, y = self._clf()
        _, pc = clf.predict(X[40], epsilon=0.1, return_update=True)
        clf.learn_one(X[40], y[40])
        assert clf._forest is pc["online_forest"]

    def test_stale_forest_falls_back(self):
        clf, X, y = self._clf()
        _, pc = clf.predict(X[40], epsilon=0.1, return_update=True)
        clf.learn_one(X[41], y[41])
        assert clf._forest is not pc["online_forest"]
        assert all(np.array_equal(t.X[-1], X[41]) for t in clf._forest)

    def test_save_load_round_trip(self, tmp_path):
        X, y = make_moons(n_samples=120, noise=0.3, random_state=1)
        clf = ConformalMondrianForestClassifier(
            n_trees=5, lifetime=3.0, label_space=np.array([0, 1]), rnd_state=1, online=True
        )
        clf.learn_initial_training_set(X[:80], y[:80])
        before = clf.predict(X[80], epsilon=0.1)
        path = tmp_path / "online_forest_clf.joblib"
        clf.save(str(path))
        loaded = ConformalMondrianForestClassifier.load(str(path))
        assert loaded.online is True
        assert len(loaded._forest) == 5
        after = loaded.predict(X[80], epsilon=0.1)
        np.testing.assert_array_equal(before.elements, after.elements)

    def test_rejects_string_lifetime(self):
        with pytest.raises(ValueError, match="fixed float lifetime"):
            ConformalMondrianForestClassifier(lifetime="sqrt_n", online=True)

    def test_rejects_max_depth(self):
        with pytest.raises(ValueError, match="max_depth=None"):
            ConformalMondrianForestClassifier(max_depth=5, online=True)

    @pytest.mark.slow
    def test_coverage_tracks_epsilon(self):
        errs = [_clf_forest_stream_error(s, online=True) for s in range(8)]
        assert 0.05 < float(np.mean(errs)) < 0.16  # ε = 0.1

    @pytest.mark.slow
    def test_coverage_matches_batch(self):
        online = float(np.mean([_clf_forest_stream_error(s, True) for s in range(6)]))
        batch = float(np.mean([_clf_forest_stream_error(s, False) for s in range(6)]))
        assert abs(online - batch) < 0.05


def _reg_forest_stream_error(seed, online, n_total=100, n_train=70):
    r = np.random.default_rng(seed)
    X = r.uniform(-3, 3, (n_total, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * r.normal(size=n_total)
    reg = ConformalMondrianForestRegressor(n_trees=5, lifetime=3.0, rnd_state=seed, online=online)
    reg.learn_initial_training_set(X[:n_train], y[:n_train])
    metric = progressive_val(reg, X[n_train:], y[n_train:], epsilon=0.1, metric=ErrorRate())
    return metric.get()


class TestOnlineForestRegressor:
    def _reg(self):
        r = np.random.default_rng(0)
        X = r.uniform(-3, 3, (60, 2))
        y = np.sin(X[:, 0]) + X[:, 1]
        reg = ConformalMondrianForestRegressor(n_trees=5, lifetime=3.0, rnd_state=0, online=True)
        reg.learn_initial_training_set(X[:40], y[:40])
        return reg, X, y

    def test_predict_and_learn_grow_the_forest(self):
        reg, X, y = self._reg()
        assert all(t.X.shape[0] == 40 for t in reg._forest)
        interval = reg.predict(X[40], epsilon=0.2)
        assert interval.lower <= interval.upper
        reg.learn_one(X[40], y[40])
        assert all(t.X.shape[0] == 41 for t in reg._forest)

    def test_getitem_returns_tree_views(self):
        reg, X, y = self._reg()
        reg.predict(X[40], epsilon=0.2)
        assert reg[0]._last_tree is not None
        assert len(list(reg)) == reg.n_trees

    def test_reuses_scored_forest_via_precomputed(self):
        reg, X, y = self._reg()
        _, pc = reg.predict(X[40], epsilon=0.2, return_update=True)
        reg.learn_one(X[40], y[40], precomputed=pc)
        assert reg._forest is pc["online_forest"]

    def test_reuses_scored_forest_via_cache(self):
        reg, X, y = self._reg()
        _, pc = reg.predict(X[40], epsilon=0.2, return_update=True)
        reg.learn_one(X[40], y[40])
        assert reg._forest is pc["online_forest"]

    def test_stale_forest_falls_back(self):
        reg, X, y = self._reg()
        _, pc = reg.predict(X[40], epsilon=0.2, return_update=True)
        reg.learn_one(X[41], y[41])
        assert reg._forest is not pc["online_forest"]
        assert all(np.array_equal(t.X[-1], X[41]) for t in reg._forest)

    def test_save_load_round_trip(self, tmp_path):
        r = np.random.default_rng(1)
        X = r.uniform(-3, 3, (100, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        reg = ConformalMondrianForestRegressor(n_trees=5, lifetime=3.0, rnd_state=1, online=True)
        reg.learn_initial_training_set(X[:70], y[:70])
        before = reg.predict(X[70], epsilon=0.2)
        path = tmp_path / "online_forest_reg.joblib"
        reg.save(str(path))
        loaded = ConformalMondrianForestRegressor.load(str(path))
        assert loaded.online is True
        assert len(loaded._forest) == 5
        after = loaded.predict(X[70], epsilon=0.2)
        np.testing.assert_allclose([before.lower, before.upper], [after.lower, after.upper])

    def test_rejects_string_lifetime(self):
        with pytest.raises(ValueError, match="fixed float lifetime"):
            ConformalMondrianForestRegressor(lifetime="density", online=True)

    def test_rejects_max_depth(self):
        with pytest.raises(ValueError, match="max_depth=None"):
            ConformalMondrianForestRegressor(max_depth=4, online=True)

    @pytest.mark.slow
    def test_coverage_tracks_epsilon(self):
        errs = [_reg_forest_stream_error(s, online=True) for s in range(6)]
        assert 0.04 < float(np.mean(errs)) < 0.17  # ε = 0.1

    @pytest.mark.slow
    def test_coverage_matches_batch(self):
        online = float(np.mean([_reg_forest_stream_error(s, True) for s in range(5)]))
        batch = float(np.mean([_reg_forest_stream_error(s, False) for s in range(5)]))
        assert abs(online - batch) < 0.06
