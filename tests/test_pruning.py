"""Tests for stream pruning (``online_cp._pruning``): reservoir sampling and sliding window.

Both reservoir sampling and the sliding window are value-based, order-invariant
sub-selections of the exchangeable stream, so both preserve exchangeability and
the exact finite-sample coverage guarantee. These tests verify:

* **Bit-identity**: with ``budget >= n`` seen, pruning is a no-op and predictions
  match the unpruned model exactly.
* **Bag cap**: the retained bag never exceeds ``budget``.
* **Index consistency** (classifier): ``_label_indices`` stay in sync with the
  pruned ``y``.
* **Coverage preservation**: pruned models still cover at the target rate.
* **Save/load**: the pruned state (bag, ``n_seen``, ``budget``, RNG position)
  round-trips and reproduces identical predictions.
"""

import numpy as np
import pytest

from online_cp import (
    ConformalNearestNeighboursClassifier,
    ConformalNearestNeighboursRegressor,
    ConformalSupportVectorMachine,
    KernelConformalRidgeRegressor,
)
from online_cp._pruning import (
    PruningStrategy,
    _evict_distance_matrix,
    _evict_masked_distance,
    _ReservoirBuffer,
    _WindowBuffer,
    validate_strategy,
)

# ---------------------------------------------------------------------------
# _ReservoirBuffer unit behaviour
# ---------------------------------------------------------------------------


def test_buffer_decide_always_keeps_when_under_budget():
    buf = _ReservoirBuffer(budget=10, rnd_gen=np.random.default_rng(0))
    for n in range(1, 11):
        assert buf.decide(n) == -1  # keep, no eviction


def test_buffer_decide_keeps_with_prob_budget_over_n():
    """When n > budget, the kept fraction over many draws should be ~budget/n."""
    rng = np.random.default_rng(0)
    budget, n = 10, 100
    buf = _ReservoirBuffer(budget=budget, rnd_gen=rng)
    kept = 0
    for _ in range(20000):
        decision = buf.decide(n)  # n fixed -> stationary decision
        kept += decision is not None
    # expected kept fraction = budget / n = 0.1
    assert abs(kept / 20000 - budget / n) < 0.01


def test_buffer_decide_evicted_position_in_range():
    rng = np.random.default_rng(1)
    budget, n = 5, 50
    buf = _ReservoirBuffer(budget=budget, rnd_gen=rng)
    for _ in range(5000):
        d = buf.decide(n)
        if d is not None:
            assert 0 <= d < budget


def test_buffer_requires_positive_budget():
    with pytest.raises(ValueError):
        _ReservoirBuffer(budget=0, rnd_gen=np.random.default_rng(0))
    with pytest.raises(ValueError):
        _ReservoirBuffer(budget=None, rnd_gen=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# eviction helpers
# ---------------------------------------------------------------------------


def test_evict_distance_matrix():
    D = np.arange(16.0).reshape(4, 4)
    out = _evict_distance_matrix(D, 1)
    assert out.shape == (3, 3)
    # row 1 and col 1 removed; remaining in order
    assert np.allclose(out, D[np.ix_([0, 2, 3], [0, 2, 3])])


def test_evict_masked_distance():
    D = np.arange(16.0).reshape(4, 4)
    keep = np.array([2, 0, 3])  # unsorted on purpose
    out = _evict_masked_distance(D, keep)
    # positions kept in ascending order
    assert np.allclose(out, D[np.ix_([0, 2, 3], [0, 2, 3])])
    mask = np.ones(4, dtype=bool)
    mask[1] = False
    assert np.allclose(_evict_masked_distance(D, mask), D[np.ix_([0, 2, 3], [0, 2, 3])])


# ---------------------------------------------------------------------------
# k-NN regressor: bit-identity, bag cap, coverage
# ---------------------------------------------------------------------------


def test_knn_reg_bit_identical_when_budget_ge_n():
    """budget >= n -> no eviction ever -> exact match with the unpruned model."""
    rng = np.random.default_rng(0)
    n, d = 200, 3
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)

    full = ConformalNearestNeighboursRegressor(k=3, rnd_state=1)
    pruned = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning="reservoir", budget=n)
    for i in range(n):
        gf = full.predict(X[i], epsilon=0.1, return_update=True)
        gp = pruned.predict(X[i], epsilon=0.1, return_update=True)
        assert np.isclose(gf[0].lower, gp[0].lower) and np.isclose(gf[0].upper, gp[0].upper), i
        full.learn_one(X[i], y[i], precomputed=gf[1])
        pruned.learn_one(X[i], y[i], precomputed=gp[1])
    assert full.X.shape[0] == pruned.X.shape[0] == n


def test_knn_reg_bag_stays_at_budget():
    rng = np.random.default_rng(1)
    n, d, budget = 500, 3, 50
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    model = ConformalNearestNeighboursRegressor(k=3, rnd_state=2, pruning="reservoir", budget=budget)
    for i in range(n):
        model.learn_one(X[i], y[i])
        assert model.X.shape[0] <= budget, i
        assert model._reservoir_n_seen == i + 1
    assert model.X.shape[0] == budget


def test_knn_reg_pruned_deterministic():
    """Two identically-seeded pruned models must produce identical predictions."""
    rng = np.random.default_rng(2)
    n, d, budget = 300, 3, 40
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    m1 = ConformalNearestNeighboursRegressor(k=3, rnd_state=3, pruning="reservoir", budget=budget)
    m2 = ConformalNearestNeighboursRegressor(k=3, rnd_state=3, pruning="reservoir", budget=budget)
    for i in range(n):
        g1 = m1.predict(X[i], epsilon=0.1)
        g2 = m2.predict(X[i], epsilon=0.1)
        assert np.isclose(g1.lower, g2.lower) and np.isclose(g1.upper, g2.upper), i
        m1.learn_one(X[i], y[i])
        m2.learn_one(X[i], y[i])


def test_knn_reg_coverage_preserved():
    rng = np.random.default_rng(3)
    n, d = 2000, 3
    X = rng.uniform(0, 1, (n, d))
    y = X[:, 0] * 2 + 0.2 * rng.standard_normal(n)
    eps = 0.1
    model = ConformalNearestNeighboursRegressor(k=5, rnd_state=4, pruning="reservoir", budget=200)
    model.learn_initial_training_set(X[:100], y[:100])
    covered = total = 0
    for i in range(100, n):
        interval = model.predict(X[i], epsilon=eps)
        covered += interval.lower <= y[i] <= interval.upper
        total += 1
        model.learn_one(X[i], y[i])
    rate = covered / total
    assert rate >= 1 - eps - 0.02, f"coverage {rate:.3f} < {1 - eps}"


# ---------------------------------------------------------------------------
# k-NN classifier: bit-identity, bag cap, index consistency, coverage
# ---------------------------------------------------------------------------


def test_clf_bit_identical_when_budget_ge_n():
    rng = np.random.default_rng(1)
    n, d = 200, 3
    X = rng.uniform(0, 1, (n, d))
    y = (X[:, 0] > 0.5).astype(int)
    full = ConformalNearestNeighboursClassifier(k=1, label_space=np.array([0, 1]), rnd_state=1)
    pruned = ConformalNearestNeighboursClassifier(
        k=1, label_space=np.array([0, 1]), rnd_state=1, pruning="reservoir", budget=n
    )
    for i in range(n):
        gf, _ = full.predict(X[i], return_update=True)
        gp, _ = pruned.predict(X[i], return_update=True)
        assert np.array_equal(np.asarray(gf.elements), np.asarray(gp.elements)), i
        full.learn_one(X[i], y[i])
        pruned.learn_one(X[i], y[i])
    assert full.X.shape[0] == pruned.X.shape[0] == n


def test_clf_bag_at_budget_and_consistent_indices():
    rng = np.random.default_rng(2)
    n, d, budget = 400, 3, 40
    X = rng.uniform(0, 1, (n, d))
    y = (X[:, 0] > 0.5).astype(int)
    model = ConformalNearestNeighboursClassifier(
        k=1, label_space=np.array([0, 1]), rnd_state=3, pruning="reservoir", budget=budget
    )
    for i in range(n):
        model.learn_one(X[i], y[i])
        assert model.X.shape[0] <= budget, i
        assert model.D.shape[0] == model.X.shape[0] == model.y.shape[0]
        # label indices must partition [0, n_bags) exactly once
        all_pos = np.concatenate([model._label_indices[l] for l in model._label_indices])
        assert set(np.sort(all_pos.tolist())) == set(range(model.y.shape[0])), i
        for lab, idx in model._label_indices.items():
            for pos in idx:
                assert model.y[pos] == lab, (i, pos, model.y[pos], lab)
    assert model.X.shape[0] == budget


def test_clf_coverage_preserved():
    """On well-separated data, pruning must keep coverage near the target."""
    rng = np.random.default_rng(4)
    n, d = 400, 2
    A = rng.normal(0, 1, (n, d))
    B = rng.normal(3, 1, (n, d))
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    eps = 0.1
    model = ConformalNearestNeighboursClassifier(
        k=1, label_space=np.array([0, 1]), rnd_state=5, pruning="reservoir", budget=150
    )
    model.learn_initial_training_set(X[:100], y[:100])
    covered = total = 0
    for i in range(100, 2 * n):
        Gamma = model.predict(X[i], epsilon=eps)
        covered += y[i] in Gamma  # built-in __contains__
        total += 1
        model.learn_one(X[i], y[i])
    rate = covered / total
    assert rate >= 1 - eps - 0.03, f"coverage {rate:.3f} < {1 - eps}"


# ---------------------------------------------------------------------------
# save / load round-trip with pruning
# ---------------------------------------------------------------------------


def test_knn_reg_saveload_preserves_pruned_state(tmp_path):
    rng = np.random.default_rng(0)
    n, d, budget = 120, 3, 30
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)
    m = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning="reservoir", budget=budget)
    for i in range(n):
        m.learn_one(X[i], y[i])
    bag, nseen = m.X.shape[0], m._reservoir_n_seen
    pred_before = m.predict(X[0], epsilon=0.1)
    fp = tmp_path / "reg.joblib"
    m.save(fp)
    m2 = ConformalNearestNeighboursRegressor.load(fp)
    assert m2.X.shape[0] == bag
    assert m2._reservoir_n_seen == nseen
    assert m2.budget == budget and m2.pruning == "reservoir"
    pred_after = m2.predict(X[0], epsilon=0.1)
    assert np.isclose(pred_before.lower, pred_after.lower) and np.isclose(pred_before.upper, pred_after.upper)


def test_clf_saveload_preserves_pruned_state(tmp_path):
    rng = np.random.default_rng(1)
    n, d, budget = 120, 2, 30
    X = rng.uniform(0, 1, (n, d))
    y = (X[:, 0] > 0.5).astype(int)
    m = ConformalNearestNeighboursClassifier(
        k=1, label_space=np.array([0, 1]), rnd_state=2, pruning="reservoir", budget=budget
    )
    for i in range(n):
        m.learn_one(X[i], y[i])
    bag, nseen = m.X.shape[0], m._reservoir_n_seen
    pred_before = m.predict(X[0], epsilon=0.1)
    fp = tmp_path / "clf.joblib"
    m.save(fp)
    m2 = ConformalNearestNeighboursClassifier.load(fp)
    assert m2.X.shape[0] == bag
    assert m2._reservoir_n_seen == nseen
    pred_after = m2.predict(X[0], epsilon=0.1)
    assert np.array_equal(np.asarray(pred_before.elements), np.asarray(pred_after.elements))
    all_pos = np.concatenate([m2._label_indices[l] for l in m2._label_indices])
    assert set(np.sort(all_pos.tolist())) == set(range(m2.y.shape[0]))


def test_pruning_requires_budget(tmp_path):
    with pytest.raises(ValueError):
        ConformalNearestNeighboursRegressor(k=1, pruning="reservoir")
    with pytest.raises(ValueError):
        ConformalNearestNeighboursClassifier(k=1, pruning="reservoir")
    with pytest.raises(ValueError):
        ConformalNearestNeighboursRegressor(k=1, pruning="window")


# ---------------------------------------------------------------------------
# Kernel-Ridge: bit-identity, bag cap, Kinv consistency
# ---------------------------------------------------------------------------


def test_krr_bit_identical_when_budget_ge_n():
    from online_cp.kernels import GaussianKernel

    rng = np.random.default_rng(0)
    n, d, budget = 150, 2, 150
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rng.standard_normal(n)
    full = KernelConformalRidgeRegressor(kernel=GaussianKernel(sigma=0.5), a=0.1, rnd_state=1)
    pruned = KernelConformalRidgeRegressor(
        kernel=GaussianKernel(sigma=0.5), a=0.1, rnd_state=1, pruning="reservoir", budget=budget
    )
    full.learn_initial_training_set(X[:10], y[:10])
    pruned.learn_initial_training_set(X[:10], y[:10])
    for i in range(10, n):
        gf = full.predict(X[i], epsilon=0.1)
        gp = pruned.predict(X[i], epsilon=0.1)
        assert np.isclose(gf.lower, gp.lower) and np.isclose(gf.upper, gp.upper), i
        full.learn_one(X[i], y[i])
        pruned.learn_one(X[i], y[i])
    assert full.X.shape[0] == pruned.X.shape[0] == n


def test_krr_bag_cap_and_kinv_consistency():
    from online_cp.kernels import GaussianKernel

    rng = np.random.default_rng(1)
    n, d, budget = 400, 2, 40
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rng.standard_normal(n)
    m = KernelConformalRidgeRegressor(
        kernel=GaussianKernel(sigma=0.5), a=0.1, rnd_state=2, pruning="reservoir", budget=budget
    )
    m.learn_initial_training_set(X[:5], y[:5])
    for i in range(5, n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
        assert m.K.shape == (m.X.shape[0], m.X.shape[0]), i
        # Kinv must be the true inverse of K + aI (the eviction recomputes it).
        n_k = m.K.shape[0]
        assert np.allclose(m.Kinv @ (m.K + m.a * np.identity(n_k)), np.identity(n_k), atol=1e-3), i
    assert m.X.shape[0] == budget


def test_krr_coverage_preserved():
    from online_cp.kernels import GaussianKernel

    rng = np.random.default_rng(3)
    n, d, budget = 300, 2, 60
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rng.standard_normal(n)
    m = KernelConformalRidgeRegressor(
        kernel=GaussianKernel(sigma=0.5), a=0.1, rnd_state=4, pruning="reservoir", budget=budget
    )
    m.learn_initial_training_set(X[:20], y[:20])
    eps = 0.1
    covered = total = 0
    for i in range(20, n):
        iv = m.predict(X[i], epsilon=eps)
        covered += iv.lower <= y[i] <= iv.upper
        total += 1
        m.learn_one(X[i], y[i])
    rate = covered / total
    assert rate >= 1 - eps - 0.03, f"coverage {rate:.3f} < {1 - eps}"


# ---------------------------------------------------------------------------
# SVM: bit-identity, bag cap, coverage
# ---------------------------------------------------------------------------


def test_svm_bit_identical_when_budget_ge_n():
    rng = np.random.default_rng(0)
    n, d, budget = 120, 2, 120
    A = rng.normal(-1, 1, (n, d))
    B = rng.normal(1, 1, (n, d))
    X = np.vstack([A, B])
    y = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    full = ConformalSupportVectorMachine(
        kernel="rbf", sigma=1.0, C=10.0, label_space=np.array([-1.0, 1.0]), rnd_state=1
    )
    pruned = ConformalSupportVectorMachine(
        kernel="rbf",
        sigma=1.0,
        C=10.0,
        label_space=np.array([-1.0, 1.0]),
        rnd_state=1,
        pruning="reservoir",
        budget=budget,
    )
    full.learn_initial_training_set(X[:10], y[:10])
    pruned.learn_initial_training_set(X[:10], y[:10])
    for i in range(10, n):
        _, pf = full.predict(X[i], epsilon=0.1, return_p_values=True)
        _, pp = pruned.predict(X[i], epsilon=0.1, return_p_values=True)
        for lab in (-1.0, 1.0):
            assert np.isclose(pf[lab], pp[lab]), (i, lab)
        full.learn_one(X[i], y[i])
        pruned.learn_one(X[i], y[i])
    assert full.X.shape[0] == pruned.X.shape[0] == n


def test_svm_bag_cap_and_gram_consistency():
    rng = np.random.default_rng(1)
    n, d, budget = 400, 2, 50
    A = rng.normal(-1, 1, (n, d))
    B = rng.normal(1, 1, (n, d))
    X = np.vstack([A, B])
    y = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    m = ConformalSupportVectorMachine(
        kernel="rbf",
        sigma=1.0,
        C=10.0,
        label_space=np.array([-1.0, 1.0]),
        rnd_state=2,
        pruning="reservoir",
        budget=budget,
    )
    m.learn_initial_training_set(X[:10], y[:10])
    for i in range(10, n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
        assert m.K.shape == (m.X.shape[0], m.X.shape[0]), i
        assert m.y.shape[0] == m.X.shape[0], i
    assert m.X.shape[0] == budget


def test_svm_coverage_preserved():
    rng = np.random.default_rng(3)
    n, d, budget = 200, 2, 80
    A = rng.normal(-1, 1, (n, d))
    B = rng.normal(1, 1, (n, d))
    X = np.vstack([A, B])
    y = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    m = ConformalSupportVectorMachine(
        kernel="rbf",
        sigma=1.0,
        C=10.0,
        label_space=np.array([-1.0, 1.0]),
        rnd_state=4,
        pruning="reservoir",
        budget=budget,
    )
    m.learn_initial_training_set(X[:20], y[:20])
    eps = 0.1
    covered = total = 0
    for i in range(20, n):
        Gamma = m.predict(X[i], epsilon=eps)
        covered += y[i] in Gamma
        total += 1
        m.learn_one(X[i], y[i])
    rate = covered / total
    assert rate >= 1 - eps - 0.03, f"coverage {rate:.3f} < {1 - eps}"


# ---------------------------------------------------------------------------
# Sliding-window pruning: bit-identity, bag cap, determinism, coverage,
# save/load
# ---------------------------------------------------------------------------


def test_window_requires_budget():
    with pytest.raises(ValueError):
        ConformalNearestNeighboursRegressor(k=1, pruning="window")
    with pytest.raises(ValueError):
        ConformalNearestNeighboursClassifier(k=1, pruning="window")
    with pytest.raises(ValueError):
        ConformalSupportVectorMachine(kernel="rbf", label_space=[-1, 1], pruning="window")
    with pytest.raises(ValueError):
        KernelConformalRidgeRegressor(kernel="rbf", pruning="window")


def test_window_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        ConformalNearestNeighboursRegressor(k=1, pruning="sv", budget=10)
    with pytest.raises(ValueError):
        ConformalNearestNeighboursClassifier(k=1, pruning="sv", budget=10)


def test_window_keeps_most_recent_and_caps_bag():
    """Window keeps the most recent `budget` examples; bag never exceeds budget."""
    rng = np.random.default_rng(0)
    n, d, budget = 80, 3, 20
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    m = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning="window", budget=budget)
    for i in range(n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
        assert m._reservoir_n_seen == i + 1
    assert m.X.shape[0] == budget
    # Window keeps the most recent: last row == X[-1], first row == X[n-budget]
    assert np.allclose(m.X[-1], X[-1])
    assert np.allclose(m.X[0], X[n - budget])
    assert np.allclose(m.y[-1], y[-1])
    assert np.allclose(m.y[0], y[n - budget])


def test_window_bit_identical_when_budget_ge_n():
    """budget >= n -> no eviction ever -> exact match with the unpruned model."""
    rng = np.random.default_rng(1)
    n, d = 120, 3
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    full = ConformalNearestNeighboursRegressor(k=3, rnd_state=1)
    win = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning="window", budget=n)
    for i in range(n):
        gf = full.predict(X[i], epsilon=0.1, return_update=True)
        gw = win.predict(X[i], epsilon=0.1, return_update=True)
        assert np.isclose(gf[0].lower, gw[0].lower) and np.isclose(gf[0].upper, gw[0].upper), i
        full.learn_one(X[i], y[i], precomputed=gf[1])
        win.learn_one(X[i], y[i], precomputed=gw[1])
    assert full.X.shape[0] == win.X.shape[0] == n


def test_window_deterministic_across_seeds():
    """Window eviction is deterministic (FIFO), so two models with different
    seeds but the same data/budget produce identical predictions."""
    rng = np.random.default_rng(2)
    n, d, budget = 200, 3, 30
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    m1 = ConformalNearestNeighboursRegressor(k=3, rnd_state=7, pruning="window", budget=budget)
    m2 = ConformalNearestNeighboursRegressor(k=3, rnd_state=999, pruning="window", budget=budget)
    for i in range(n):
        g1 = m1.predict(X[i], epsilon=0.1)
        g2 = m2.predict(X[i], epsilon=0.1)
        assert np.isclose(g1.lower, g2.lower) and np.isclose(g1.upper, g2.upper), i
        m1.learn_one(X[i], y[i])
        m2.learn_one(X[i], y[i])


def test_window_classifier_keeps_most_recent_and_indices_consistent():
    rng = np.random.default_rng(3)
    n, budget = 40, 15
    A = rng.normal(0, 1, (n, 2))
    B = rng.normal(3, 1, (n, 2))
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    m = ConformalNearestNeighboursClassifier(
        k=1, label_space=np.array([0, 1]), rnd_state=4, pruning="window", budget=budget
    )
    for i in range(2 * n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
        for lab, idx in m._label_indices.items():
            for pos in idx:
                assert m.y[pos] == lab, (i, pos, m.y[pos], lab)
    assert m.X.shape[0] == budget
    assert np.allclose(m.X[-1], X[-1])
    assert np.allclose(m.X[0], X[2 * n - budget])


def test_window_kernel_ridge_keeps_most_recent_and_kinv_consistent():
    from online_cp.kernels import GaussianKernel

    rng = np.random.default_rng(4)
    n, budget = 60, 15
    X = rng.uniform(0, 1, (n, 2))
    y = np.sin(4 * X[:, 0]) + 0.1 * rng.standard_normal(n)
    m = KernelConformalRidgeRegressor(
        kernel=GaussianKernel(sigma=0.5), a=0.1, rnd_state=1, pruning="window", budget=budget
    )
    for i in range(n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
        assert m.K.shape == (m.X.shape[0], m.X.shape[0]), i
        assert m.Kinv.shape == (m.X.shape[0], m.X.shape[0]), i
    assert m.X.shape[0] == budget
    assert np.allclose(m.X[-1], X[-1])
    assert np.allclose(m.X[0], X[n - budget])
    # K must equal the kernel of the retained X (consistency)
    assert np.allclose(m.K, m.kernel(m.X))


def test_window_svm_keeps_most_recent():
    rng = np.random.default_rng(5)
    n, budget = 40, 15
    A = rng.normal(-1, 1, (n, 2))
    B = rng.normal(1, 1, (n, 2))
    X = np.vstack([A, B])
    y = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    m = ConformalSupportVectorMachine(
        kernel="rbf",
        sigma=1.0,
        C=10.0,
        label_space=np.array([-1.0, 1.0]),
        rnd_state=6,
        pruning="window",
        budget=budget,
    )
    for i in range(2 * n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
    assert m.X.shape[0] == budget
    assert np.allclose(m.X[-1], X[-1])
    assert np.allclose(m.X[0], X[2 * n - budget])


def test_window_coverage_preserved():
    """Window pruning keeps coverage near the target (within a looser tolerance
    than reservoir, since the window is less robust on drifting streams)."""
    rng = np.random.default_rng(6)
    n, d = 2000, 3
    X = rng.uniform(0, 1, (n, d))
    y = X[:, 0] * 2 + 0.2 * rng.standard_normal(n)
    eps = 0.1
    model = ConformalNearestNeighboursRegressor(k=5, rnd_state=7, pruning="window", budget=200)
    model.learn_initial_training_set(X[:100], y[:100])
    covered = total = 0
    for i in range(100, n):
        interval = model.predict(X[i], epsilon=eps)
        covered += interval.lower <= y[i] <= interval.upper
        total += 1
        model.learn_one(X[i], y[i])
    rate = covered / total
    assert rate >= 1 - eps - 0.03, f"window coverage {rate:.3f} < {1 - eps - 0.03}"


def test_window_saveload_preserves_state(tmp_path):
    rng = np.random.default_rng(7)
    n, d, budget = 120, 3, 30
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    m = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning="window", budget=budget)
    for i in range(n):
        m.learn_one(X[i], y[i])
    bag, nseen = m.X.shape[0], m._reservoir_n_seen
    pred_before = m.predict(X[0], epsilon=0.1)
    fp = tmp_path / "win.joblib"
    m.save(fp)
    m2 = ConformalNearestNeighboursRegressor.load(fp)
    assert m2.X.shape[0] == bag
    assert m2._reservoir_n_seen == nseen
    assert m2.budget == budget and m2.pruning == "window"
    pred_after = m2.predict(X[0], epsilon=0.1)
    assert np.isclose(pred_before.lower, pred_after.lower) and np.isclose(pred_before.upper, pred_after.upper)


def test_window_seed_in_initial_training_set():
    """When learn_initial_training_set is called with n > budget, the window keeps
    the last `budget` examples (not a random sample)."""
    rng = np.random.default_rng(8)
    n, d, budget = 50, 3, 20
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)
    m = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning="window", budget=budget)
    m.learn_initial_training_set(X, y)
    assert m.X.shape[0] == budget
    # Window keeps the last budget examples
    assert np.allclose(m.X, X[n - budget:])
    assert np.allclose(m.y, y[n - budget:])


# ---------------------------------------------------------------------------
# Validator: data-blind (strictly-valid) contract
# ---------------------------------------------------------------------------


# A module-level data-blind strategy (picklable by reference, unlike the
# locally-defined helpers), used to exercise save/load reproducibility.
class _EveryKthTestStrategy(PruningStrategy):
    def __init__(self, budget, rnd_gen, period=3):
        super().__init__(budget, rnd_gen)
        self.period = period

    def decide(self, n_seen):
        if n_seen <= self.budget:
            return -1
        return (n_seen // self.period) % self.budget


def _valid_strategy(period=3):
    """A minimal data-blind strategy: evict position (n // period) % budget."""

    class _EveryPeriod(PruningStrategy):
        def __init__(self, budget, rnd_gen, period):
            super().__init__(budget, rnd_gen)
            self.period = period

        def decide(self, n_seen):
            if n_seen <= self.budget:
                return -1
            return (n_seen // self.period) % self.budget

    return _EveryPeriod(10, np.random.default_rng(0), period)


def _invalid_strategy_with_data_param():
    """A strategy that inspects the data via a 'y' parameter -> invalid."""

    class _LabelHugger:
        def __init__(self, budget, rnd_gen):
            self.budget = budget

        def decide(self, n_seen, y):  # <-- value-dependent parameter
            return 0

    return _LabelHugger(10, np.random.default_rng(0))


def test_validate_builtin_names_pass():
    validate_strategy("reservoir")
    validate_strategy("window")


def test_validate_rejects_unknown_name():
    for bad in ("sv", "leverage", "window2", ""):
        with pytest.raises(ValueError):
            validate_strategy(bad)


def test_validate_accepts_pruning_strategy_instance():
    # A PruningStrategy instance is trusted per its documented contract.
    validate_strategy(_valid_strategy())
    # Built-in strategies are also PruningStrategy instances.
    validate_strategy(_ReservoirBuffer(10, np.random.default_rng(0)))
    validate_strategy(_WindowBuffer(10, np.random.default_rng(0)))


def test_validate_accepts_data_blind_callable():
    class _EveryKth:
        def __init__(self, budget, rnd_gen):
            self.budget = budget

        def decide(self, n_seen, budget, period=3):
            if n_seen <= budget:
                return -1
            return (n_seen // period) % budget

    validate_strategy(_EveryKth(10, np.random.default_rng(0)))


def test_validate_rejects_callable_inspecting_data():
    """Each value-dependent parameter name must be rejected by the validator."""
    for param in ("y", "x", "model", "d", "leverage", "alpha", "point", "example", "bag", "labels", "features", "data", "train"):
        # Build a strategy whose `decide` signature names the offending parameter.
        src = (
            "class _S:\n"
            f"    def decide(self, n_seen, {param}):\n"
            "        return 0\n"
        )
        ns = {}
        exec(src, ns)
        strat = ns["_S"]()
        with pytest.raises(ValueError):
            validate_strategy(strat)


def test_validate_rejects_callable_without_decide():
    with pytest.raises(ValueError):
        validate_strategy(object())


def test_validate_rejects_ambiguous_k_param():
    class _AmbiguousK:
        def __init__(self, budget, rnd_gen):
            self.budget = budget

        def decide(self, n_seen, k):
            return 0

    with pytest.raises(ValueError):
        validate_strategy(_AmbiguousK(10, np.random.default_rng(0)))


def test_validate_accepts_keyword_only_k():
    class _KWOnlyK:
        def __init__(self, budget, rnd_gen):
            self.budget = budget

        def decide(self, n_seen, *, k=3):
            if n_seen <= self.budget:
                return -1
            return (n_seen // k) % self.budget

    validate_strategy(_KWOnlyK(10, np.random.default_rng(0)))


def test_model_rejects_invalid_string_pruning():
    from online_cp import ConformalNearestNeighboursRegressor

    with pytest.raises(ValueError):
        ConformalNearestNeighboursRegressor(k=3, pruning="leverage", budget=10)


def test_model_accepts_custom_data_blind_strategy():
    """A user-defined data-blind strategy is accepted and used by the model."""
    from online_cp import ConformalNearestNeighboursRegressor

    rng = np.random.default_rng(0)
    n, d, budget = 60, 2, 20
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)

    strat = _valid_strategy(period=5)
    m = ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning=strat, budget=budget)
    for i in range(n):
        m.learn_one(X[i], y[i])
        assert m.X.shape[0] <= budget, i
    assert m.X.shape[0] == budget


def test_model_rejects_data_dependent_callable_strategy():
    """A callable strategy that inspects the data is rejected at construction."""
    from online_cp import ConformalNearestNeighboursRegressor

    bad = _invalid_strategy_with_data_param()
    with pytest.raises(ValueError):
        ConformalNearestNeighboursRegressor(k=3, rnd_state=1, pruning=bad, budget=10)


def test_custom_strategy_saveload_reproducible(tmp_path):
    """A custom data-blind strategy round-trips via save/load and the model's
    injected RNG keeps the pruned stream reproducible."""
    from online_cp import ConformalNearestNeighboursRegressor

    rng = np.random.default_rng(0)
    n, d, budget = 120, 3, 30
    X = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal(n)

    m = ConformalNearestNeighboursRegressor(
        k=3,
        rnd_state=1,
        pruning=_EveryKthTestStrategy(budget, np.random.default_rng(0), period=7),
        budget=budget,
    )
    for i in range(n):
        m.learn_one(X[i], y[i])
    pred_before = m.predict(X[0], epsilon=0.1)
    fp = tmp_path / "custom.joblib"
    m.save(fp)
    m2 = ConformalNearestNeighboursRegressor.load(fp)
    pred_after = m2.predict(X[0], epsilon=0.1)
    assert np.isclose(pred_before.lower, pred_after.lower) and np.isclose(
        pred_before.upper, pred_after.upper
    )
    assert m2.X.shape[0] == m.X.shape[0] <= budget


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
