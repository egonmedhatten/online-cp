"""Layer 1: enumerative property-based tests (``leancheck``).

These tests assert *structural / mathematical invariants* that must hold across
a wide enumeration of discrete edge cases (insertion orders, delay sequences,
quantized p-value streams, significance grids). They are fast and run in the
default lane.

``leancheck`` selects test generators from type annotations.  Domain-specific
types (``Epsilon``, ``UnitProb``, ``PValue``) enumerate values directly in the
correct range.  ``leancheck.precondition()`` skips vacuous inputs rather than
returning ``True`` early (which would count as a pass).

Run with pytest::

    pytest tests/test_properties.py

Or run all properties standalone with LeanCheck reporting::

    python tests/test_properties.py
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
    PCA,
    SVD,
    ConformalNearestNeighboursClassifier,
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    CUSUMWrapper,
    ErrorRate,
    IntervalWidth,
    ObservedExcess,
    PluginMartingale,
    RidgePredictionMachine,
    SetSize,  # noqa: F401 — imported for completeness; used in P16 comment
    ShiryaevRobertsWrapper,
    SimpleJumper,
    StandardScaler,
    VennPrediction,
    VilleWrapper,
    progressive_val,
)
from online_cp.classifiers import ConformalClassifier, ConformalPredictionSet
from online_cp.regressors import ConformalPredictionInterval, ConformalRegressor
from online_cp.venn import _pava_inplace

# Enumerated-case budgets.
#
# _TESTS_SLOW: one full model fit (or streaming loop) per test case.
#              ~0.5–1.2 ms/test → 360 tests ≈ 0.2–0.4 s per property.
# _TESTS_MED:  one O(n) loop or matrix decomposition per test case
#              (martingales, change detectors, PAVA, SVD/PCA).
#              ~0.1–0.6 ms/test → 720 tests ≈ 0.1–0.4 s per property.
# _TESTS_FAST: pure arithmetic / data-structure checks.
#              < 0.01 ms/test → 3600 tests ≈ negligible.
#
# Total target: ~8 s for the full property suite in CI.
_TESTS_SLOW = 360  # model-fitting or full streaming-pipeline properties
_TESTS_MED  = 720  # sequence-loop / matrix-decomposition properties
_TESTS_FAST = 3600 # pure-arithmetic / data-structure properties

# =========================================================================== #
# Domain types for LeanCheck enumeration.                                     #
#                                                                             #
# Registering float subclasses as Enumerator types lets property signatures  #
# express domain constraints directly rather than through int→domain helpers. #
# =========================================================================== #


class Epsilon(float):
    """Significance level in [0.05, 0.50]."""


class UnitProb(float):
    """Probability or tau value in [0.0, 1.0]."""


class PValue(float):
    """P-value in the open interval (0, 1)."""


# Epsilon: include practically important small values (0.01, 0.02) plus the
# standard 0.05–0.50 grid.  These properties exhaust at 12×12−12=132 pairs
# regardless of max_tests, giving complete pairwise coverage.
leancheck.Enumerator.register_choices(
    Epsilon,
    [0.01, 0.02] + [0.05 + i * 0.05 for i in range(10)],
)
leancheck.Enumerator.register_choices(UnitProb, [i / 100.0 for i in range(101)])
leancheck.Enumerator.register_choices(PValue, [(i + 1) / 100.0 for i in range(99)])


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
    assert leancheck.check(prop_ridge_prediction_order_invariant, max_tests=_TESTS_SLOW, silent=True)


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
    assert leancheck.check(prop_streaming_accounting_integrity, max_tests=_TESTS_SLOW, silent=True)


# --------------------------------------------------------------------------- #
# Property 3: change-point detector running-statistic invariants.
#
# For ANY sequence of valid p-values:
#   - CUSUM gamma_n >= 1 always (log_gamma = logM - running_min >= 0).
#   - Ville running maximum is non-decreasing and never below the current logM.
#   - Shiryaev-Roberts statistic R_n >= 0 always.
# --------------------------------------------------------------------------- #


def prop_cusum_gamma_at_least_one(pvs: list[PValue]) -> bool:
    leancheck.precondition(bool(pvs))
    cusum = CUSUMWrapper(SimpleJumper(J=0.01))
    for p in pvs:
        cusum.update(float(p))
        if cusum.gamma < 1.0 - 1e-9:
            return False
    return True


def prop_ville_max_monotone_and_dominates(pvs: list[PValue]) -> bool:
    leancheck.precondition(bool(pvs))
    ville = VilleWrapper(SimpleJumper(J=0.01), threshold=20)
    prev_max = ville.log_max
    for p in pvs:
        ville.update(float(p))
        if ville.log_max < prev_max - 1e-12:
            return False
        if ville.log_max < ville.martingale.logM - 1e-9:
            return False
        prev_max = ville.log_max
    return True


def prop_shiryaev_roberts_nonnegative(pvs: list[PValue]) -> bool:
    leancheck.precondition(bool(pvs))
    sr = ShiryaevRobertsWrapper(SimpleJumper(J=0.01))
    for p in pvs:
        sr.update(float(p))
        if sr.R < -1e-9:
            return False
    return True


def test_cusum_gamma_at_least_one():
    assert leancheck.check(prop_cusum_gamma_at_least_one, max_tests=_TESTS_MED, silent=True)


def test_ville_max_monotone_and_dominates():
    assert leancheck.check(prop_ville_max_monotone_and_dominates, max_tests=_TESTS_MED, silent=True)


def test_shiryaev_roberts_nonnegative():
    assert leancheck.check(prop_shiryaev_roberts_nonnegative, max_tests=_TESTS_MED, silent=True)


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


def prop_intervals_nested_in_epsilon(e1: Epsilon, e2: Epsilon) -> bool:
    leancheck.precondition(abs(float(e1) - float(e2)) >= 1e-9)
    lo, hi = min(float(e1), float(e2)), max(float(e1), float(e2))
    iv_small_eps = _NS_MODEL.predict(_NS_XQ, epsilon=lo, bounds="both")
    iv_large_eps = _NS_MODEL.predict(_NS_XQ, epsilon=hi, bounds="both")
    return bool(iv_small_eps.width() >= iv_large_eps.width() - 1e-9)


def test_intervals_nested_in_epsilon():
    assert leancheck.check(prop_intervals_nested_in_epsilon, max_tests=_TESTS_FAST, silent=True)


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
    leancheck.precondition(bool(ks))
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
    assert leancheck.check(prop_pava_non_decreasing, max_tests=_TESTS_MED, silent=True)


def test_pava_bounded_by_input():
    assert leancheck.check(prop_pava_bounded_by_input, max_tests=_TESTS_MED, silent=True)


def test_pava_preserves_sum():
    assert leancheck.check(prop_pava_preserves_sum, max_tests=_TESTS_MED, silent=True)


def test_pava_idempotent():
    assert leancheck.check(prop_pava_idempotent, max_tests=_TESTS_MED, silent=True)


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
    assert leancheck.check(prop_metric_value_delay_invariant, max_tests=_TESTS_SLOW, silent=True)


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


def prop_classifier_sets_nested_in_epsilon(e1: Epsilon, e2: Epsilon) -> bool:
    leancheck.precondition(abs(float(e1) - float(e2)) >= 1e-9)
    lo, hi = min(float(e1), float(e2)), max(float(e1), float(e2))
    # Single predict call: both levels share the same smoothing draw.
    ml = _CS_CLF.predict(_CS_XQ, epsilon=np.array([lo, hi]))
    set_lo = set(ml[lo].elements.tolist())
    set_hi = set(ml[hi].elements.tolist())
    # Smaller ε (lo) → larger or equal set.
    superset_ok = set_hi.issubset(set_lo)
    size_ok = len(ml[lo]) >= len(ml[hi])
    return bool(superset_ok and size_ok)


def test_classifier_sets_nested_in_epsilon():
    assert leancheck.check(prop_classifier_sets_nested_in_epsilon, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 8: VennPrediction.binary structural bounds.
#
# For any (p0, p1) in [0, 1]:
#   - All four matrix entries are in [0, 1].
#   - Each row of the probability matrix sums to 1.
#   - .p0 and .p1 round-trip through the matrix.
#   - .point sums to 1 and point[1] == (p0 + p1) / 2.
# --------------------------------------------------------------------------- #


def prop_venn_binary_rows_sum_to_one(p0: UnitProb, p1: UnitProb) -> bool:
    vp = VennPrediction.binary(float(p0), float(p1))
    row0_sum = float(vp.probs[0].sum())
    row1_sum = float(vp.probs[1].sum())
    return bool(abs(row0_sum - 1.0) < 1e-9 and abs(row1_sum - 1.0) < 1e-9)


def prop_venn_binary_entries_in_unit_interval(p0: UnitProb, p1: UnitProb) -> bool:
    vp = VennPrediction.binary(float(p0), float(p1))
    return bool(np.all((vp.probs >= -1e-9) & (vp.probs <= 1.0 + 1e-9)))


def prop_venn_binary_point_sums_to_one(p0: UnitProb, p1: UnitProb) -> bool:
    vp = VennPrediction.binary(float(p0), float(p1))
    point_sum = float(vp.point.sum())
    point1_expected = (float(p0) + float(p1)) / 2.0
    return bool(abs(point_sum - 1.0) < 1e-9 and abs(float(vp.point[1]) - point1_expected) < 1e-9)


def test_venn_binary_rows_sum_to_one():
    assert leancheck.check(prop_venn_binary_rows_sum_to_one, max_tests=_TESTS_FAST, silent=True)


def test_venn_binary_entries_in_unit_interval():
    assert leancheck.check(prop_venn_binary_entries_in_unit_interval, max_tests=_TESTS_FAST, silent=True)


def test_venn_binary_point_sums_to_one():
    assert leancheck.check(prop_venn_binary_point_sums_to_one, max_tests=_TESTS_FAST, silent=True)


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
    assert leancheck.check(prop_knn_prediction_order_invariant, max_tests=_TESTS_SLOW, silent=True)


# =========================================================================== #
# Shared helpers for P10-P18
# =========================================================================== #


def _alpha_array(ks: list[int]) -> np.ndarray:
    """Map enumerated ints to a nonconformity score array.

    The last element is the test-object score; at least one training-object
    score is prepended so the p-value formula is always well-defined.
    """
    vals = [float(abs(int(k)) % 20) for k in ks] if ks else [5.0]
    if len(vals) < 2:
        vals = [0.0] + vals
    return np.array(vals, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Property 10: _compute_p_value output is always in [0, 1].
#
# The static p-value method is the core of every conformal guarantee. It is a
# pure function: leancheck enumerates the nonconformity score array (via
# list[int]) and tau. Both the regressor and classifier variants are tested
# (they share the same formula but differ in default signature).
# --------------------------------------------------------------------------- #


def prop_regressor_p_value_in_unit_interval(ks: list[int], tau: UnitProb) -> bool:
    Alpha = _alpha_array(ks)
    p = ConformalRegressor._compute_p_value(Alpha, tau=float(tau))
    return bool(-1e-9 <= p <= 1.0 + 1e-9)


def prop_classifier_p_value_in_unit_interval(ks: list[int], tau: UnitProb) -> bool:
    Alpha = _alpha_array(ks)
    p = ConformalClassifier._compute_p_value(Alpha, tau=float(tau))
    return bool(-1e-9 <= p <= 1.0 + 1e-9)


def test_regressor_p_value_in_unit_interval():
    assert leancheck.check(prop_regressor_p_value_in_unit_interval, max_tests=_TESTS_FAST, silent=True)


def test_classifier_p_value_in_unit_interval():
    assert leancheck.check(prop_classifier_p_value_in_unit_interval, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 11: minimum_training_set is exactly ceil(2/eps) or ceil(1/eps).
#
# Pure closed-form formula; no data needed. Enumerated over a fine eps grid.
# --------------------------------------------------------------------------- #


def prop_minimum_training_set_both_bounds(eps: PValue) -> bool:
    # PValue enumerates (0.01, ..., 0.99) — exactly the eps grid needed.
    # LeanCheck exhausts all 99 values, so no modular arithmetic required.
    result = ConformalRegressor.minimum_training_set(float(eps), bounds="both")
    return result == int(np.ceil(2.0 / float(eps)))


def prop_minimum_training_set_one_sided(eps: PValue) -> bool:
    result = ConformalRegressor.minimum_training_set(float(eps), bounds="lower")
    return result == int(np.ceil(1.0 / float(eps)))


def test_minimum_training_set_both_bounds():
    assert leancheck.check(prop_minimum_training_set_both_bounds, max_tests=_TESTS_FAST, silent=True)


def test_minimum_training_set_one_sided():
    assert leancheck.check(prop_minimum_training_set_one_sided, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 12: ConformalPredictionInterval semantics.
#
# width() == upper - lower (exact arithmetic); __contains__ iff lower<=y<=upper
# (including boundary). Both are pure data-class operations.
# --------------------------------------------------------------------------- #


def prop_interval_width_equals_upper_minus_lower(a: int, b: int) -> bool:
    lo = float(a % 50)
    hi = lo + float(abs(b) % 30)
    iv = ConformalPredictionInterval(lo, hi, 0.1)
    return bool(abs(iv.width() - (hi - lo)) < 1e-12)


def prop_interval_containment_iff_in_bounds(a: int, b: int, c: int) -> bool:
    lo = float(a % 50)
    hi = lo + float(abs(b) % 30)
    y = float(c % 100 - 30)
    iv = ConformalPredictionInterval(lo, hi, 0.1)
    return bool((y in iv) == (lo <= y <= hi))


def test_interval_width_equals_upper_minus_lower():
    assert leancheck.check(prop_interval_width_equals_upper_minus_lower, max_tests=_TESTS_FAST, silent=True)


def test_interval_containment_iff_in_bounds():
    assert leancheck.check(prop_interval_containment_iff_in_bounds, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 13: Metric.cumulative_mean() matches the manual cumsum formula.
#
# Pure arithmetic over the accumulated _values list. ErrorRate is used as the
# concrete Metric; the 0/1 scores are driven by whether the interval covers
# y=0.0 (always-cover vs. never-cover sentinel intervals).
# --------------------------------------------------------------------------- #

_CM_IV_IN = ConformalPredictionInterval(-1e9, 1e9, 0.1)    # always covers y=0
_CM_IV_OUT = ConformalPredictionInterval(100.0, 200.0, 0.1)  # never covers y=0


def prop_cumulative_mean_matches_formula(ks: list[int]) -> bool:
    leancheck.precondition(bool(ks))
    metric = ErrorRate()
    for k in ks:
        iv = _CM_IV_IN if abs(k) % 2 == 0 else _CM_IV_OUT
        metric.update(y=0.0, Gamma=iv)
    cm = metric.cumulative_mean()
    expected = np.cumsum(metric._values) / np.arange(1, metric._n + 1)
    return bool(np.allclose(cm, expected))


def test_cumulative_mean_matches_formula():
    assert leancheck.check(prop_cumulative_mean_matches_formula, max_tests=_TESTS_MED, silent=True)


# --------------------------------------------------------------------------- #
# Property 14: PluginMartingale wealth-accounting invariants.
#
# After each update(p): logM == log_martingale_values[-1] (sync), the history
# list has grown by exactly one entry (started at [0.0]), and M == exp(logM).
# --------------------------------------------------------------------------- #


def prop_plugin_martingale_logM_synced(pvs: list[PValue]) -> bool:
    leancheck.precondition(bool(pvs))
    m = PluginMartingale()
    for i, p in enumerate(pvs):
        m.update(float(p))
        if not np.isclose(m.logM, m.log_martingale_values[-1]):
            return False
        if len(m.log_martingale_values) != i + 2:  # [0.0] + one per update
            return False
        if not np.isclose(m.M, np.exp(m.logM)):
            return False
    return True


def test_plugin_martingale_logM_synced():
    assert leancheck.check(prop_plugin_martingale_logM_synced, max_tests=_TESTS_MED, silent=True)


# --------------------------------------------------------------------------- #
# Property 15: Metrics composition broadcasts updates synchronously.
#
# After every update to a composite Metrics object, all sub-metrics must have
# identical _n (they are updated in lockstep inside Metrics.update).
# --------------------------------------------------------------------------- #

_COMP_IV = ConformalPredictionInterval(0.0, 10.0, 0.1)


def prop_metrics_composition_n_sync(ks: list[int]) -> bool:
    m = ErrorRate() + IntervalWidth()
    for _k in ks:
        m.update(y=0.0, Gamma=_COMP_IV)
        if m["ErrorRate"]._n != m["IntervalWidth"]._n:
            return False
    return True


def test_metrics_composition_n_sync():
    assert leancheck.check(prop_metrics_composition_n_sync, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 16: algebraic identities between metric scores.
#
# (a) ErrorRate(y, Gamma) + (y in Gamma) == 1 always: the error indicator and
#     the coverage indicator are complementary by construction.
# (b) ObservedExcess(y, Gamma) + (y in Gamma) == len(Gamma) always: OE counts
#     incorrect labels; adding back the one correct label (when covered)
#     recovers the full set size.
# --------------------------------------------------------------------------- #


def prop_error_rate_plus_coverage_equals_one(a: int, b: int, c: int) -> bool:
    lo = float(a % 50)
    hi = lo + float(abs(b) % 30)
    y = float(c % 100 - 30)
    iv = ConformalPredictionInterval(lo, hi, 0.1)
    error = ErrorRate()._score(y=y, Gamma=iv)
    coverage = float(y in iv)
    return bool(abs(error + coverage - 1.0) < 1e-12)


def prop_observed_excess_plus_coverage_equals_set_size(a: int, b: int) -> bool:
    n = 1 + abs(a) % 5  # set size: 1 to 5 labels
    include_y = b % 2 == 0
    y = 0
    elems = np.arange(n) if include_y else np.arange(1, n + 1)
    gs = ConformalPredictionSet(elems, epsilon=0.1)
    oe = ObservedExcess()._score(y=y, Gamma=gs)
    coverage = float(y in gs)
    return bool(abs(oe + coverage - float(len(gs))) < 1e-12)


def test_error_rate_plus_coverage_equals_one():
    assert leancheck.check(prop_error_rate_plus_coverage_equals_one, max_tests=_TESTS_FAST, silent=True)


def test_observed_excess_plus_coverage_equals_set_size():
    assert leancheck.check(prop_observed_excess_plus_coverage_equals_set_size, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 17: predict(return_update=True) + learn_one(precomputed) state
# equivalence.
#
# Two execution paths from the same initial model state must produce identical
# X, y, XTXinv after the update:
#   Path A — predict(x) [no cache], then learn_one(x, y).
#   Path B — predict(x, return_update=True) -> (interval, cache),
#             then learn_one(x, y, precomputed=cache).
# This tests the streaming-efficiency caching protocol that all online
# pipelines rely on.
# --------------------------------------------------------------------------- #

_PC_RNG = np.random.default_rng(10)
_PC_N, _PC_D = 15, 2
_PC_X = _PC_RNG.normal(size=(_PC_N, _PC_D))
_PC_Y = _PC_X @ np.array([1.0, -0.5]) + 0.1 * _PC_RNG.normal(size=_PC_N)


def _new_point(ks: list[int]) -> tuple:
    """Derive a new (x, y) point from an enumerated key list."""
    seed = [ks[i] if i < len(ks) else 0 for i in range(_PC_D + 1)]
    x = np.array([float(seed[i] % 20 - 10) / 5.0 for i in range(_PC_D)])
    y = float(seed[_PC_D] % 20 - 10) / 5.0
    return x, y


def prop_predict_return_update_state_equivalence(ks: list[int]) -> bool:
    x_new, y_new = _new_point(ks)

    # Path A: uncached predict + normal learn_one.
    cp_a = ConformalRidgeRegressor(a=1.0, warnings=False)
    cp_a.learn_initial_training_set(_PC_X.copy(), _PC_Y.copy())
    cp_a.predict(x_new, epsilon=0.3)
    cp_a.learn_one(x_new, y_new)

    # Path B: cached predict + learn_one with precomputed state.
    cp_b = ConformalRidgeRegressor(a=1.0, warnings=False)
    cp_b.learn_initial_training_set(_PC_X.copy(), _PC_Y.copy())
    _, cache = cp_b.predict(x_new, epsilon=0.3, return_update=True)
    cp_b.learn_one(x_new, y_new, precomputed=cache)

    return bool(
        np.allclose(cp_a.XTXinv, cp_b.XTXinv, atol=1e-8)
        and np.allclose(cp_a.y, cp_b.y)
        and np.allclose(cp_a.X, cp_b.X)
    )


def test_predict_return_update_state_equivalence():
    assert leancheck.check(prop_predict_return_update_state_equivalence, max_tests=_TESTS_SLOW, silent=True)


# --------------------------------------------------------------------------- #
# Property 18: Conformal Predictive Distribution (CPD) satisfies CDF axioms.
#
# RidgePredictiveDistributionFunction (returned by RidgePredictionMachine.
# predict_cpd) must satisfy the four standard CDF axioms for any tau in [0,1]:
#   (a) Values in [0, 1].
#   (b) Non-decreasing in y.
#   (c) Boundary: F(-inf, tau) = 0 and F(+inf, tau) = 1.
#   (d) Quantile lower-bound: F(quantile(p, tau), tau) >= p  (smallest y
#       such that F(y) >= p, so evaluating F at that point must recover p).
# --------------------------------------------------------------------------- #

_CPD_RNG = np.random.default_rng(12)
_CPD_N, _CPD_D = 20, 2
_CPD_X = _CPD_RNG.normal(size=(_CPD_N, _CPD_D))
_CPD_Y = _CPD_X @ np.array([1.0, -0.5]) + 0.1 * _CPD_RNG.normal(size=_CPD_N)
_CPD_XQ = np.array([0.1, 0.2])
_CPD_MODEL = RidgePredictionMachine(a=1.0)
_CPD_MODEL.learn_initial_training_set(_CPD_X, _CPD_Y)
_CPD = _CPD_MODEL.predict_cpd(_CPD_XQ)


def prop_cpd_values_in_unit_interval(a: int, tau: UnitProb) -> bool:
    y = float(a % 40 - 10)
    p = _CPD(y, float(tau))
    return bool(-1e-9 <= p <= 1.0 + 1e-9)


def prop_cpd_monotone_in_y(a: int, b: int, tau: UnitProb) -> bool:
    y1 = float(a % 40 - 10)
    gap = float(abs(b) % 20)
    leancheck.precondition(gap >= 1e-9)
    y2 = y1 + gap
    return bool(_CPD(y1, float(tau)) <= _CPD(y2, float(tau)) + 1e-9)


def prop_cpd_boundary_conditions(tau: UnitProb) -> bool:
    return bool(
        abs(_CPD(-np.inf, float(tau))) < 1e-9 and abs(_CPD(np.inf, float(tau)) - 1.0) < 1e-9
    )


def prop_cpd_quantile_lower_bound(p: PValue, tau: UnitProb) -> bool:
    q = _CPD.quantile(float(p), float(tau))
    leancheck.precondition(np.isfinite(q))
    return bool(_CPD(q, float(tau)) >= float(p) - 1e-9)


def test_cpd_values_in_unit_interval():
    assert leancheck.check(prop_cpd_values_in_unit_interval, max_tests=_TESTS_FAST, silent=True)


def test_cpd_monotone_in_y():
    assert leancheck.check(prop_cpd_monotone_in_y, max_tests=_TESTS_FAST, silent=True)


def test_cpd_boundary_conditions():
    assert leancheck.check(prop_cpd_boundary_conditions, max_tests=_TESTS_FAST, silent=True)


def test_cpd_quantile_lower_bound():
    assert leancheck.check(prop_cpd_quantile_lower_bound, max_tests=_TESTS_FAST, silent=True)


# --------------------------------------------------------------------------- #
# Property 21: bag-pipeline p-value is order-invariant over the training bag. #
#                                                                              #
# The bag-mode StandardScaler fits on [X_train, x_test] symmetrically        #
# (objects-only, label-free) → permutation-equivariant feature map.          #
# Combined with ConformalRidgeRegressor (already permutation-invariant),      #
# the pipeline p-value must be identical regardless of training-bag order.    #
# This is the empirical exchangeability proof for the bag-fit regime.         #
# --------------------------------------------------------------------------- #

_BAG_RNG = np.random.default_rng(7)
_BAG_N, _BAG_D = 14, 2
# Poorly-scaled features so the bag scaler actually changes the geometry.
_BAG_X = _BAG_RNG.normal(size=(_BAG_N, _BAG_D)) * np.array([100.0, 0.001])
_BAG_Y = _BAG_X @ np.array([0.01, 1000.0]) + 0.1 * _BAG_RNG.normal(size=_BAG_N)
_BAG_XQ = _BAG_RNG.normal(size=_BAG_D) * np.array([100.0, 0.001])
_BAG_YQ = float(_BAG_Y[0])
_BAG_TAU = 0.5


def _perm_from_keys_bag(keys: list[int]) -> list[int]:
    pad = [keys[i] if i < len(keys) else 0 for i in range(_BAG_N)]
    return [int(j) for j in np.argsort(np.array(pad), kind="stable")]


def _fit_bag_pipeline_in_order(order: list[int]):
    pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0, warnings=False)
    for i in order:
        pipe.learn_one(_BAG_X[i], float(_BAG_Y[i]))
    return pipe


def prop_bag_pipeline_p_value_order_invariant(keys: list[int]) -> bool:
    """P-value must not change under permutation of the training-bag insertion order."""
    ref = _fit_bag_pipeline_in_order(list(range(_BAG_N)))
    out = _fit_bag_pipeline_in_order(_perm_from_keys_bag(keys))
    p_ref = ref.compute_p_value(_BAG_XQ, _BAG_YQ, tau=_BAG_TAU)
    p_out = out.compute_p_value(_BAG_XQ, _BAG_YQ, tau=_BAG_TAU)
    return bool(np.isclose(p_ref, p_out, atol=1e-10))


def test_bag_pipeline_p_value_order_invariant():
    assert leancheck.check(prop_bag_pipeline_p_value_order_invariant, max_tests=_TESTS_SLOW, silent=True)


# --------------------------------------------------------------------------- #
# Property 22: PCA components_ are orthonormal across random data shapes.    #
#                                                                              #
# V @ V^T must equal I_k for any valid (n ≥ 2, d ≥ 1) batch; this also      #
# validates the sign-flip convention leaves the vectors normalised.           #
# --------------------------------------------------------------------------- #

_PROP22_RNG = np.random.default_rng(22)
_PROP22_SIZES = [(n, d) for n in [4, 8, 20] for d in [1, 2, 5]]


def prop_pca_components_orthonormal(idx: int) -> bool:
    """PCA components_ @ components_.T == I_k for a fixed set of data shapes."""
    n, d = _PROP22_SIZES[idx % len(_PROP22_SIZES)]
    X = _PROP22_RNG.normal(size=(n, d))
    pca = PCA()
    pca.fit(X)
    gram = pca.components_ @ pca.components_.T
    return bool(np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10))


def test_pca_components_orthonormal():
    assert leancheck.check(prop_pca_components_orthonormal, max_tests=_TESTS_MED, silent=True)


# --------------------------------------------------------------------------- #
# Property 23: SVD components_ are orthonormal (center=True and False).      #
# --------------------------------------------------------------------------- #

def prop_svd_components_orthonormal(idx: int) -> bool:
    """SVD components_ @ components_.T == I_k regardless of center flag."""
    n, d = _PROP22_SIZES[idx % len(_PROP22_SIZES)]
    center = bool(idx % 2)
    X = _PROP22_RNG.normal(size=(n, d))
    svd = SVD(center=center)
    svd.fit(X)
    gram = svd.components_ @ svd.components_.T
    return bool(np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10))


def test_svd_components_orthonormal():
    assert leancheck.check(prop_svd_components_orthonormal, max_tests=_TESTS_MED, silent=True)


if __name__ == "__main__":
    leancheck.main(verbose=True, exit_on_failure=False)
