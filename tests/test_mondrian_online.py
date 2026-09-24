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
