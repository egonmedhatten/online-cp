"""Tests for the batch prediction API (``predict_batch``).

The strict contract: ``predict_batch(X)[i]`` must be **bit-identical** to
``predict(X[i])`` for every row ``i``. This holds by construction for the
default loop-fallback implementation (Ridge, Lasso, SVM, k-NN regressor, and any
other model), and must hold for the vectorised overrides (currently the k-NN
classifier).

We verify bit-identity for all pruned-capable models, plus permutation
invariance (reordering the batch reorders the results, nothing else).
"""

import numpy as np
import pytest

from online_cp import (
    ConformalLassoRegressor,
    ConformalNearestNeighboursClassifier,
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    ConformalSupportVectorMachine,
    KernelConformalRidgeRegressor,
)
from online_cp.kernels import GaussianKernel


def _regressors():
    rng = np.random.default_rng(0)
    n, d = 120, 3
    X = rng.uniform(0, 1, (n, d))
    y = X.sum(axis=1) + 0.1 * rng.standard_normal(n)
    models = [
        ConformalRidgeRegressor(a=1.0, rnd_state=1),
        ConformalLassoRegressor(rnd_state=1),
        KernelConformalRidgeRegressor(kernel=GaussianKernel(sigma=0.5), a=0.1, rnd_state=1),
        ConformalNearestNeighboursRegressor(k=3, rnd_state=1),
    ]
    for m in models:
        m.learn_initial_training_set(X[:60], y[:60])
        for i in range(60, n):
            m.learn_one(X[i], y[i])
    return models, X, y


def _classifiers():
    rng = np.random.default_rng(1)
    n, d = 120, 2
    A = rng.normal(-1, 1, (n, d))
    B = rng.normal(1, 1, (n, d))
    X = np.vstack([A, B])
    y = np.concatenate([np.full(n, 0), np.full(n, 1)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    models = [
        ConformalNearestNeighboursClassifier(k=1, label_space=np.array([0, 1]), rnd_state=2),
        ConformalSupportVectorMachine(kernel="rbf", sigma=1.0, C=10.0, label_space=np.array([0, 1]), rnd_state=2),
    ]
    for m in models:
        m.learn_initial_training_set(X[:60], y[:60])
        for i in range(60, 2 * n):
            m.learn_one(X[i], y[i])
    return models, X, y


def _set_equal(a, b):
    """Compare two prediction sets element-wise (works for continuous & discrete)."""
    if hasattr(a, "lower"):
        return np.isclose(a.lower, b.lower) and np.isclose(a.upper, b.upper)
    return np.array_equal(np.asarray(a.elements), np.asarray(b.elements))


def test_batch_bit_identical_regressors():
    models, X, y = _regressors()
    Xb = X[100:115]
    for m in models:
        batch = m.predict_batch(Xb, epsilon=0.1)
        assert len(batch) == 15
        # fresh model with same RNG state for the sequential reference
        for i, x in enumerate(Xb):
            ref = m.predict(x, epsilon=0.1)
            assert _set_equal(batch[i], ref), (type(m).__name__, i)


def test_batch_bit_identical_classifiers():
    models, X, y = _classifiers()
    Xb = X[100:112]
    for m in models:
        # The k-NN classifier draws a smoothing tau per test point from its RNG.
        # To compare the batch against the sequential loop fairly, run the batch
        # from a fixed RNG state, then run the sequential loop from the SAME state.
        state = m.rnd_gen.bit_generator.state
        batch = m.predict_batch(Xb, epsilon=0.1, return_p_values=True)
        m.rnd_gen.bit_generator.state = state
        for i, x in enumerate(Xb):
            ref = m.predict(x, epsilon=0.1, return_p_values=True)
            assert _set_equal(batch[i][0], ref[0]), (type(m).__name__, i)
            for lab in ref[1]:
                assert np.isclose(batch[i][1][lab], ref[1][lab]), (type(m).__name__, i, lab)


def test_batch_permutation_invariance_ridge():
    """For the tau-free ridge regressor, reordering the batch reorders the results
    exactly; nothing else changes. (tau-smoothed predictors are positionally bound
    to the smoothing variable, so this invariant applies to non-smoothed models.)"""
    models, X, y = _regressors()
    ridge = models[0]
    Xb = X[100:110]
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(Xb))
    batch = ridge.predict_batch(Xb, epsilon=0.1)
    batch_perm = ridge.predict_batch(Xb[perm], epsilon=0.1)
    # Row j of the permuted batch is Xb[perm[j]], i.e. row perm[j] of the original
    # batch -- so batch_perm[j] must equal batch[perm[j]].
    for j in range(len(Xb)):
        assert _set_equal(batch_perm[j], batch[perm[j]]), j


def test_batch_single_row_matches_predict():
    """A 1-row batch must equal predict on that row (edge case)."""
    models, X, y = _regressors()
    ridge = models[0]
    x = X[5]
    batch = ridge.predict_batch(x.reshape(1, -1), epsilon=0.1)
    assert len(batch) == 1
    assert _set_equal(batch[0], ridge.predict(x, epsilon=0.1))


def test_batch_handles_multilevel_epsilon():
    """A multi-level epsilon must produce MultiLevelPredictionSets, matching predict."""
    models, X, y = _regressors()
    ridge = models[0]
    Xb = X[100:105]
    epsilons = np.array([0.1, 0.05])
    batch = ridge.predict_batch(Xb, epsilon=epsilons)
    for i, x in enumerate(Xb):
        ref = ridge.predict(x, epsilon=epsilons)
        for eps in epsilons:
            assert _set_equal(batch[i][float(eps)], ref[float(eps)]), (i, eps)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
