"""Layer 1: enumerative property-based tests (``leancheck``).

These tests assert *structural / mathematical invariants* that must hold across
a wide enumeration of discrete edge cases (insertion orders, delay sequences,
quantized p-value streams, significance grids). They are fast and run in the
default lane.

``leancheck`` selects test generators from type annotations, so each ``prop_*``
function takes typed arguments (``int``, ``list[int]``) that drive a
deterministic reconstruction of the numeric inputs inside the property.

Run with::

    pytest tests/test_properties.py
"""

# NOTE: do *not* add ``from __future__ import annotations`` here. leancheck
# reads ``__annotations__`` to select generators and needs real type objects,
# not stringized annotations (PEP 563), to enumerate ``int`` / ``list[int]``.
#
# leancheck is a declared test dependency (``dev`` and ``ci`` extras in
# pyproject.toml).  We use a hard import rather than ``pytest.importorskip``
# so that a missing package causes a loud collection error instead of silently
# skipping the entire module and letting CI stay green vacuously.

import leancheck
import numpy as np

from online_cp import (
    ConformalNearestNeighboursClassifier,
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    CUSUMWrapper,
    ErrorRate,
    ShiryaevRobertsWrapper,
    SimpleJumper,
    VennPrediction,
    VilleWrapper,
    progressive_val,
)
from online_cp.venn import _pava_inplace

# Number of enumerated cases per property. Kept modest so the structural lane
# stays fast (each case may fit a model or run a streaming loop).
MAX_TESTS = 200


def _check(prop) -> bool:
    """Run a leancheck property silently and return pass/fail as a bool."""
    return bool(leancheck.check(prop, max_tests=MAX_TESTS, silent=True))


# --------------------------------------------------------------------------- #
# Property 1: order-invariance of transductive conformal prediction.
#
# Conformal p-values are rank-based, so the prediction for a fixed test object
# must not depend on the order in which the calibration points were learned.
# --------------------------------------------------------------------------- #

_OI_RNG = np.random.default_rng(0)
_OI_N, _OI_D = 12, 2
_OI_X = _OI_RNG.normal(size=(_OI_N, _OI_D))
_OI_Y = _OI_X @ np.array([1.5, -0.7]) + 0.05 * _OI_RNG.normal(size=_OI_N)
_OI_XQ = np.array([0.3, -0.2])
_OI_EPS = 0.5  # needs ceil(2/eps)=4 points for a finite interval; we have 12


def _perm_from_keys(keys: list[int]) -> list[int]:
    """Map any ``list[int]`` to a valid permutation of ``range(_OI_N)``.

    Pads/truncates the enumerated keys to length ``_OI_N`` and returns a stable
    argsort, so every enumerated list yields a genuine reordering.
    """
    pad = [keys[i] if i < len(keys) else 0 for i in range(_OI_N)]
    return [int(j) for j in np.argsort(np.array(pad), kind="stable")]


def _fit_ridge_in_order(order: list[int]) -> ConformalRidgeRegressor:
    cp = ConformalRidgeRegressor(a=1.0, warnings=False)
    for i in order:
        cp.learn_one(_OI_X[i], float(_OI_Y[i]))
    return cp


def prop_ridge_prediction_order_invariant(keys: list[int]) -> bool:
    ref = _fit_ridge_in_order(list(range(_OI_N)))
    out = _fit_ridge_in_order(_perm_from_keys(keys))
    iv_ref = ref.predict(_OI_XQ, epsilon=_OI_EPS, bounds="both")
    iv_out = out.predict(_OI_XQ, epsilon=_OI_EPS, bounds="both")
    return bool(np.isclose(iv_ref.lower, iv_out.lower) and np.isclose(iv_ref.upper, iv_out.upper))


def test_ridge_prediction_order_invariant():
    assert _check(prop_ridge_prediction_order_invariant)


# --------------------------------------------------------------------------- #
# Property 2: streaming heap / delayed-label accounting integrity.
#
# Regardless of how labels are delayed (any permutation of arrival times),
# every labelled point must be drained exactly once and the metric's internal
# accounting must stay consistent. NOTE: the metric *value* is deliberately not
# asserted invariant - under delay the model learns later, so predictions
# legitimately change; only the bookkeeping is invariant.
# --------------------------------------------------------------------------- #

_ST_RNG = np.random.default_rng(1)
_ST_N, _ST_D = 10, 2
_ST_X = _ST_RNG.normal(size=(_ST_N, _ST_D))
_ST_Y = _ST_X @ np.array([1.0, 0.5]) + 0.1 * _ST_RNG.normal(size=_ST_N)


def _delay_array(delays: list[int]) -> np.ndarray:
    """Build a non-negative integer delay of length ``_ST_N`` from enumerated ints."""
    if len(delays) == 0:
        return np.zeros(_ST_N, dtype=int)
    base = np.array([abs(int(delays[i % len(delays)])) % 6 for i in range(_ST_N)], dtype=int)
    return base


def prop_streaming_accounting_integrity(delays: list[int]) -> bool:
    d = _delay_array(delays)
    model = ConformalRidgeRegressor(a=1.0, warnings=False)
    metric = ErrorRate()
    progressive_val(
        model,
        _ST_X,
        _ST_Y,
        epsilon=0.2,
        metric=metric,
        delay=lambda i, x, y: int(d[i]),
    )
    counted_all = metric._n == _ST_N
    list_consistent = len(metric._values) == metric._n
    sum_consistent = abs(metric._sum - float(np.sum(metric._values))) < 1e-9
    return bool(counted_all and list_consistent and sum_consistent)


def test_streaming_accounting_integrity():
    assert _check(prop_streaming_accounting_integrity)


# --------------------------------------------------------------------------- #
# Property 3: change-point detector running-statistic invariants.
#
# For ANY sequence of valid p-values:
#   - CUSUM gamma_n >= 1 always (log_gamma = logM - running_min >= 0).
#   - Ville running maximum is non-decreasing and never below the current logM.
#   - Shiryaev-Roberts statistic R_n >= 0 always.
# --------------------------------------------------------------------------- #


def _pvals(ks: list[int]) -> list[float]:
    """Map enumerated ints to quantized p-values in the open interval (0, 1)."""
    if len(ks) == 0:
        return [0.5]
    return [((abs(int(k)) % 100) + 0.5) / 100.0 for k in ks]


def prop_cusum_gamma_at_least_one(ks: list[int]) -> bool:
    cusum = CUSUMWrapper(SimpleJumper(J=0.01))
    for p in _pvals(ks):
        cusum.update(p)
        if cusum.gamma < 1.0 - 1e-9:
            return False
    return True


def prop_ville_max_monotone_and_dominates(ks: list[int]) -> bool:
    ville = VilleWrapper(SimpleJumper(J=0.01), threshold=20)
    prev_max = ville.log_max
    for p in _pvals(ks):
        ville.update(p)
        if ville.log_max < prev_max - 1e-12:
            return False
        if ville.log_max < ville.martingale.logM - 1e-9:
            return False
        prev_max = ville.log_max
    return True


def prop_shiryaev_roberts_nonnegative(ks: list[int]) -> bool:
    sr = ShiryaevRobertsWrapper(SimpleJumper(J=0.01))
    for p in _pvals(ks):
        sr.update(p)
        if sr.R < -1e-9:
            return False
    return True


def test_cusum_gamma_at_least_one():
    assert _check(prop_cusum_gamma_at_least_one)


def test_ville_max_monotone_and_dominates():
    assert _check(prop_ville_max_monotone_and_dominates)


def test_shiryaev_roberts_nonnegative():
    assert _check(prop_shiryaev_roberts_nonnegative)


# --------------------------------------------------------------------------- #
# Property 4: prediction intervals are nested in the significance level.
#
# Smaller epsilon => wider (or equal) interval: eps1 < eps2 => width1 >= width2.
# --------------------------------------------------------------------------- #

_NS_RNG = np.random.default_rng(2)
_NS_N, _NS_D = 60, 3
_NS_X = _NS_RNG.normal(size=(_NS_N, _NS_D))
_NS_Y = _NS_X @ np.array([1.0, -0.5, 0.25]) + 0.2 * _NS_RNG.normal(size=_NS_N)
_NS_XQ = np.array([0.1, 0.2, -0.1])
_NS_MODEL = ConformalRidgeRegressor(a=1.0, warnings=False)
_NS_MODEL.learn_initial_training_set(_NS_X, _NS_Y)


def _eps(k: int) -> float:
    """Map an int to a significance level in [0.05, 0.5]."""
    return 0.05 + (abs(int(k)) % 10) * 0.05


def prop_intervals_nested_in_epsilon(a: int, b: int) -> bool:
    e1, e2 = _eps(a), _eps(b)
    lo, hi = min(e1, e2), max(e1, e2)
    if hi - lo < 1e-9:
        return True  # identical levels: nothing to compare
    iv_small_eps = _NS_MODEL.predict(_NS_XQ, epsilon=lo, bounds="both")
    iv_large_eps = _NS_MODEL.predict(_NS_XQ, epsilon=hi, bounds="both")
    return bool(iv_small_eps.width() >= iv_large_eps.width() - 1e-9)


def test_intervals_nested_in_epsilon():
    assert _check(prop_intervals_nested_in_epsilon)


# --------------------------------------------------------------------------- #
# Property 5: PAVA isotonic regression invariants.
#
# _pava_inplace is the core of Venn-Abers calibration. As a pure function it
# is an ideal leancheck target. Four invariants must hold for any input:
#   (a) output is non-decreasing;
#   (b) output is bounded by [min(input), max(input)];
#   (c) the (unweighted) sum is preserved (unit weights => weighted-sum = sum);
#   (d) applying PAVA twice is idempotent.
# venn.py has a no-op njit fallback, so this works with or without numba.
# --------------------------------------------------------------------------- #


def _pava_array(ks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Map enumerated ints to a float64 value array and unit weight array."""
    vals = np.array([float(k % 20) for k in ks] if ks else [0.0], dtype=np.float64)
    return vals.copy(), np.ones(len(vals), dtype=np.float64)


def prop_pava_non_decreasing(ks: list[int]) -> bool:
    y, w = _pava_array(ks)
    _pava_inplace(y, w)
    return bool(np.all(np.diff(y) >= -1e-9))


def prop_pava_bounded_by_input(ks: list[int]) -> bool:
    if not ks:
        return True
    y, w = _pava_array(ks)
    lo, hi = y.min(), y.max()
    _pava_inplace(y, w)
    return bool(y.min() >= lo - 1e-9 and y.max() <= hi + 1e-9)


def prop_pava_preserves_sum(ks: list[int]) -> bool:
    y, w = _pava_array(ks)
    total_before = float(y.sum())
    _pava_inplace(y, w)
    return bool(abs(float(y.sum()) - total_before) < 1e-6)


def prop_pava_idempotent(ks: list[int]) -> bool:
    y, w = _pava_array(ks)
    _pava_inplace(y, w)
    y_once = y.copy()
    _pava_inplace(y, w)
    return bool(np.allclose(y, y_once))


def test_pava_non_decreasing():
    assert leancheck.check(prop_pava_non_decreasing, max_tests=500, silent=True)


def test_pava_bounded_by_input():
    assert leancheck.check(prop_pava_bounded_by_input, max_tests=500, silent=True)


def test_pava_preserves_sum():
    assert leancheck.check(prop_pava_preserves_sum, max_tests=500, silent=True)


def test_pava_idempotent():
    assert leancheck.check(prop_pava_idempotent, max_tests=500, silent=True)


# --------------------------------------------------------------------------- #
# Property 6: streaming metric value is delay-invariant when learn=False.
#
# When the model never updates (learn=False), every prediction is made from
# the same frozen model state, so the final mean metric must be identical
# regardless of how the labels are delayed. The ordering of per-step scores
# changes under delay, but the set of scores — and therefore the mean — is
# invariant. This catches scheduler bugs invisible to the accounting-only test.
# --------------------------------------------------------------------------- #

_DI_RNG = np.random.default_rng(3)
_DI_N, _DI_D = 10, 2
_DI_X = _DI_RNG.normal(size=(_DI_N, _DI_D))
_DI_Y = _DI_X @ np.array([1.0, 0.5]) + 0.1 * _DI_RNG.normal(size=_DI_N)
_DI_MODEL = ConformalRidgeRegressor(a=1.0, warnings=False)
_DI_MODEL.learn_initial_training_set(_DI_X, _DI_Y)
# Baseline: no delay, frozen model.
_DI_BASELINE = progressive_val(
    _DI_MODEL, _DI_X, _DI_Y, epsilon=0.2, metric=ErrorRate(), learn=False
).get()


def prop_metric_value_delay_invariant(delays: list[int]) -> bool:
    d = np.array(
        [abs(int(delays[i % len(delays)])) % 6 for i in range(_DI_N)] if delays else [0] * _DI_N,
        dtype=int,
    )
    metric = progressive_val(
        _DI_MODEL,
        _DI_X,
        _DI_Y,
        epsilon=0.2,
        metric=ErrorRate(),
        learn=False,
        delay=lambda i, x, y: int(d[i]),
    )
    return bool(abs(metric.get() - _DI_BASELINE) < 1e-9)


def test_metric_value_delay_invariant():
    assert _check(prop_metric_value_delay_invariant)


# --------------------------------------------------------------------------- #
# Property 7: classifier prediction sets are nested in the significance level.
#
# For ε₁ < ε₂ the prediction set at ε₁ must be a superset of the set at ε₂
# (smaller ε → larger set). We use a SINGLE predict() call with a sorted
# epsilon array so both levels share the same p-values / smoothing draw,
# making the nesting guarantee exact (not just probabilistic).
# --------------------------------------------------------------------------- #

_CS_RNG = np.random.default_rng(4)
_CS_N, _CS_D = 60, 3
_CS_LABELS = np.array([0, 1] * (_CS_N // 2))
_CS_X = _CS_RNG.normal(size=(_CS_N, _CS_D))
_CS_X[_CS_LABELS == 0] -= 1.5
_CS_X[_CS_LABELS == 1] += 1.5
_CS_XQ = np.array([0.1, 0.2, -0.1])
_CS_CLF = ConformalNearestNeighboursClassifier(k=1, label_space=np.array([0, 1]), rnd_state=0)
_CS_CLF.learn_initial_training_set(_CS_X, _CS_LABELS)


def prop_classifier_sets_nested_in_epsilon(a: int, b: int) -> bool:
    e1, e2 = _eps(a), _eps(b)  # reuse _eps helper from Property 4
    if abs(e1 - e2) < 1e-9:
        return True
    lo, hi = min(e1, e2), max(e1, e2)
    # Single predict call: both levels share the same smoothing draw.
    ml = _CS_CLF.predict(_CS_XQ, epsilon=np.array([lo, hi]))
    set_lo = set(ml[lo].elements.tolist())
    set_hi = set(ml[hi].elements.tolist())
    # Smaller ε (lo) → larger or equal set.
    superset_ok = set_hi.issubset(set_lo)
    size_ok = len(ml[lo]) >= len(ml[hi])
    return bool(superset_ok and size_ok)


def test_classifier_sets_nested_in_epsilon():
    assert _check(prop_classifier_sets_nested_in_epsilon)


# --------------------------------------------------------------------------- #
# Property 8: VennPrediction.binary structural bounds.
#
# For any (p0, p1) in [0, 1]:                                               
#   - All four matrix entries are in [0, 1].                                  
#   - Each row of the probability matrix sums to 1.                           
#   - .p0 and .p1 round-trip through the matrix.                              
#   - .point sums to 1 and point[1] == (p0 + p1) / 2.                        
# --------------------------------------------------------------------------- #


def _unit_prob(k: int) -> float:
    """Map an enumerated int to a probability in [0, 1]."""
    return (abs(int(k)) % 101) / 100.0


def prop_venn_binary_rows_sum_to_one(a: int, b: int) -> bool:
    p0, p1 = _unit_prob(a), _unit_prob(b)
    vp = VennPrediction.binary(p0, p1)
    row0_sum = float(vp.probs[0].sum())
    row1_sum = float(vp.probs[1].sum())
    return bool(abs(row0_sum - 1.0) < 1e-9 and abs(row1_sum - 1.0) < 1e-9)


def prop_venn_binary_entries_in_unit_interval(a: int, b: int) -> bool:
    p0, p1 = _unit_prob(a), _unit_prob(b)
    vp = VennPrediction.binary(p0, p1)
    return bool(np.all((vp.probs >= -1e-9) & (vp.probs <= 1.0 + 1e-9)))


def prop_venn_binary_point_sums_to_one(a: int, b: int) -> bool:
    p0, p1 = _unit_prob(a), _unit_prob(b)
    vp = VennPrediction.binary(p0, p1)
    point_sum = float(vp.point.sum())
    point1_expected = (p0 + p1) / 2.0
    return bool(abs(point_sum - 1.0) < 1e-9 and abs(float(vp.point[1]) - point1_expected) < 1e-9)


def test_venn_binary_rows_sum_to_one():
    assert _check(prop_venn_binary_rows_sum_to_one)


def test_venn_binary_entries_in_unit_interval():
    assert _check(prop_venn_binary_entries_in_unit_interval)


def test_venn_binary_point_sums_to_one():
    assert _check(prop_venn_binary_point_sums_to_one)


# --------------------------------------------------------------------------- #
# Property 9: order-invariance of KNN regressor prediction.
#
# Mirrors Property 1 for ConformalNearestNeighboursRegressor, extending the
# exchangeability guarantee to a second estimator family.
# --------------------------------------------------------------------------- #

_KNN_RNG = np.random.default_rng(5)
_KNN_N, _KNN_D = 12, 2
_KNN_X = _KNN_RNG.normal(size=(_KNN_N, _KNN_D))
_KNN_Y = _KNN_X @ np.array([1.5, -0.7]) + 0.05 * _KNN_RNG.normal(size=_KNN_N)
_KNN_XQ = np.array([0.3, -0.2])
_KNN_EPS = 0.5  # needs ceil(2/eps)=4 points; we have 12


def _perm_knn(keys: list[int]) -> list[int]:
    pad = [keys[i] if i < len(keys) else 0 for i in range(_KNN_N)]
    return [int(j) for j in np.argsort(np.array(pad), kind="stable")]


def _fit_knn_in_order(order: list[int]) -> ConformalNearestNeighboursRegressor:
    knn = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
    for i in order:
        knn.learn_one(_KNN_X[i], float(_KNN_Y[i]))
    return knn


def prop_knn_prediction_order_invariant(keys: list[int]) -> bool:
    ref = _fit_knn_in_order(list(range(_KNN_N)))
    out = _fit_knn_in_order(_perm_knn(keys))
    iv_ref = ref.predict(_KNN_XQ, epsilon=_KNN_EPS, bounds="both")
    iv_out = out.predict(_KNN_XQ, epsilon=_KNN_EPS, bounds="both")
    return bool(np.isclose(iv_ref.lower, iv_out.lower) and np.isclose(iv_ref.upper, iv_out.upper))


def test_knn_prediction_order_invariant():
    assert _check(prop_knn_prediction_order_invariant)
