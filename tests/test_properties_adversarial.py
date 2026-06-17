"""Layer 1 adversarial: LeanCheck falsification on degenerate inputs.

Each property *should* hold as a mathematical invariant.  The test is written
BEFORE the corresponding library fix so that LeanCheck can produce a minimal
counterexample.  After each fix the property turns green and becomes a
permanent regression guard.

Attacks implemented here
------------------------
A1  KNN regressor order-invariance on tie-rich data
    ``np.argpartition`` is documented as unstable on ties; with equidistant
    training points at the k-boundary, different insertion orders yield
    different neighbour selections and thus different prediction intervals.
    Fix: use ``np.lexsort((y, distances))`` as a canonical tie-break.

A4  CPD quantile lower-bound with duplicate training rows
    ``RidgePredictiveDistributionFunction._compute_quantile`` assumes that
    all interior C values are distinct ("from A*y+B formula").  Duplicate
    training rows produce tied C values; the closed-form formula then returns
    a quantile Q where Pi(Q, tau) < p, violating the quantile lower-bound.
    Fix: check the actual ``_cdf_bounds(C[j_at])`` before accepting j_at.

A5  ``VennPrediction.binary`` entries in [0, 1]
    No bounds check on p0, p1; ``binary(-0.5, 0.5)`` silently produces
    probability entries outside [0, 1].
    Fix: raise ``ValueError`` when p0 or p1 is outside [0, 1].

A6  ``log_loss_point`` output in [0, 1]
    No bounds check on p0, p1; ``log_loss_point(-0.5, -0.5) == -0.5``.
    Fix: raise ``ValueError`` when p0 or p1 is outside [0, 1].

NOTE: do *not* add ``from __future__ import annotations`` — leancheck reads
``__annotations__`` and needs real type objects, not PEP-563 strings.
"""

import leancheck
import numpy as np

from online_cp import (
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    RidgePredictionMachine,
    SimpleLegendreJumper,
    StandardScaler,
    VariationalLegendreJumper,
    VennPrediction,
)
from online_cp.venn import log_loss_point

_TESTS_SLOW = 360  # model-fitting or full streaming-pipeline properties
_TESTS_MED  = 720  # sequence-loop / matrix-decomposition properties
_TESTS_FAST = 3600 # pure-arithmetic / data-structure properties


class UnitProb(float):
    """Probability or tau value in [0.0, 1.0]."""


class PValue(float):
    """P-value in the open interval (0, 1)."""


leancheck.Enumerator.register_choices(UnitProb, [i / 100.0 for i in range(101)])
leancheck.Enumerator.register_choices(PValue, [(i + 1) / 100.0 for i in range(99)])


def _perm_from_keys(keys: list[int], n: int) -> list[int]:
    """Map any ``list[int]`` to a permutation of ``range(n)`` via stable argsort."""
    pad = [keys[i] if i < len(keys) else 0 for i in range(n)]
    return [int(j) for j in np.argsort(np.array(pad), kind="stable")]


# ======================================================================= #
# A1 – KNN regressor order-invariance on tie-rich data                    #
#                                                                          #
# Dataset: 8 training points.  The first 4 are equidistant from the query #
# at d=1.0 and carry *distinct* y-values (0, 25, 50, 75).  The last 4 are #
# far away (d=10) and carry the same y-value so they cannot contribute to  #
# tie ambiguity.  With k=2 and epsilon=0.25 (needs ≥ 8 points), the two   #
# nearest neighbours for the test query are always drawn from the 4 tied   #
# equidistant points – but which two depends on argpartition tie-breaking. #
# ======================================================================= #

_TIE_N = 8
_TIE_X = np.array(
    [
        [ 1.0,  0.0],  # 0 – d=1.0 from origin, y=0
        [-1.0,  0.0],  # 1 – d=1.0 from origin, y=25
        [ 0.0,  1.0],  # 2 – d=1.0 from origin, y=50
        [ 0.0, -1.0],  # 3 – d=1.0 from origin, y=75
        [10.0,  0.0],  # 4 – far, y=37.5
        [-10.0, 0.0],  # 5 – far, y=37.5
        [ 0.0, 10.0],  # 6 – far, y=37.5
        [ 0.0,-10.0],  # 7 – far, y=37.5
    ],
    dtype=float,
)
_TIE_Y = np.array([0.0, 25.0, 50.0, 75.0, 37.5, 37.5, 37.5, 37.5])
_TIE_XQ = np.array([0.0, 0.0])
_TIE_K = 2
_TIE_EPS = 0.25  # needs ceil(2/0.25)=8 ≤ n_aug=9; n_aug=9 ≥ 2/0.25=8 ✓


def _fit_knn_reg_in_order(order: list[int]) -> ConformalNearestNeighboursRegressor:
    m = ConformalNearestNeighboursRegressor(k=_TIE_K, rnd_state=0)
    for i in order:
        m.learn_one(_TIE_X[i], float(_TIE_Y[i]))
    return m


def prop_knn_regressor_tie_order_invariant(keys: list[int]) -> bool:
    """Prediction interval must be identical regardless of insertion order.

    Different insertion orders produce permuted distance arrays; on tie-rich
    data ``np.argpartition`` selects different tied neighbours → different
    predictions → property falsifies BEFORE the lexsort fix.
    """
    ref = _fit_knn_reg_in_order(list(range(_TIE_N)))
    out = _fit_knn_reg_in_order(_perm_from_keys(keys, _TIE_N))
    iv_ref = ref.predict(_TIE_XQ, epsilon=_TIE_EPS, bounds="both")
    iv_out = out.predict(_TIE_XQ, epsilon=_TIE_EPS, bounds="both")
    return bool(
        np.isclose(iv_ref.lower, iv_out.lower) and np.isclose(iv_ref.upper, iv_out.upper)
    )


def test_knn_regressor_tie_order_invariant():
    assert leancheck.check(prop_knn_regressor_tie_order_invariant, max_tests=_TESTS_SLOW, silent=True)


# ======================================================================= #
# A4 – CPD quantile lower-bound with duplicate training rows               #
#                                                                          #
# 4 identical training rows cause ``predict_cpd`` to produce              #
# C = [-inf, v, v, v, v, +inf] (all interior C values equal).             #
# The Ridge closed-form ``_compute_quantile`` assumes distinct C; it then  #
# may return Q where Pi(Q, tau) < p.                                       #
# ======================================================================= #

_CPD_N_DUPS = 4
_CPD_X_DUP = np.tile(np.array([[1.0, 0.0]]), (_CPD_N_DUPS, 1))
_CPD_Y_DUP = np.ones(_CPD_N_DUPS)
_CPD_XQ = np.array([0.5, 0.0])


def prop_cpd_quantile_lower_bound_with_dups(p: PValue, tau: UnitProb) -> bool:
    """F(Q(p, tau), tau) ≥ p must hold even when C has tied values.

    Before the _compute_quantile fix: with C=[-inf,v,v,v,v,+inf] and
    p=0.6, tau=0.5 the formula returns Q=v but Pi(v,0.5)=0.5 < 0.6.

    Degenerate tails (Q = ±inf) are skipped via precondition, matching P18.
    """
    m = RidgePredictionMachine(a=1.0, warnings=False)
    m.learn_initial_training_set(_CPD_X_DUP, _CPD_Y_DUP)
    cpd = m.predict_cpd(_CPD_XQ)
    Q = cpd.quantile(float(p), float(tau))
    leancheck.precondition(np.isfinite(Q))
    pi = cpd(Q, float(tau))
    return bool(pi >= float(p) - 1e-9)


def test_cpd_quantile_lower_bound_with_dups():
    assert leancheck.check(prop_cpd_quantile_lower_bound_with_dups, max_tests=_TESTS_MED, silent=True)


# ======================================================================= #
# A5 – VennPrediction.binary: probability matrix entries in [0, 1]         #
#                                                                          #
# ``binary(p0, p1)`` builds [[1-p0, p0], [1-p1, p1]].  Without a bounds   #
# check, out-of-range p0 / p1 produce entries < 0 or > 1.                 #
# After fix: ValueError is raised for out-of-range inputs, which the       #
# property treats as "correctly rejected" (returns True).                  #
# ======================================================================= #

_P_VALS = [-0.5, 0.0, 0.1, 0.5, 0.9, 1.0, 1.5]  # includes out-of-range


def _pv(k: int) -> float:
    return _P_VALS[abs(k) % len(_P_VALS)]


def prop_venn_binary_entries_in_unit_interval(k0: int, k1: int) -> bool:
    """Probability matrix entries must lie in [0, 1] or a ValueError must be raised.

    Before fix: ``VennPrediction.binary(-0.5, 0.5)`` silently returns an
    entry of -0.5 → property falsifies.
    After fix: ValueError raised for out-of-range inputs → True.
    """
    p0, p1 = _pv(k0), _pv(k1)
    try:
        vp = VennPrediction.binary(p0, p1)
    except (ValueError, TypeError):
        return True  # out-of-range input correctly rejected
    return bool(np.all(vp.probs >= 0.0) and np.all(vp.probs <= 1.0))


def test_venn_binary_entries_in_unit_interval():
    assert leancheck.check(prop_venn_binary_entries_in_unit_interval, max_tests=_TESTS_FAST, silent=True)


# ======================================================================= #
# A6 – log_loss_point: output in [0, 1]                                   #
#                                                                          #
# ``log_loss_point(p0, p1) = p1 / (1 - p0 + p1)`` guards only denom==0.  #
# Out-of-range inputs (e.g. p0=p1=-0.5) produce negative outputs.         #
# After fix: ValueError raised for out-of-range inputs → True.            #
# ======================================================================= #


def prop_log_loss_point_in_unit_interval(k0: int, k1: int) -> bool:
    """Output must lie in [0, 1] or a ValueError must be raised.

    Before fix: ``log_loss_point(-0.5, -0.5) == -0.5`` → property
    falsifies immediately (k0=k1=0 hits p0=p1=-0.5).
    After fix: ValueError raised → True.
    """
    p0, p1 = _pv(k0), _pv(k1)
    try:
        result = log_loss_point(p0, p1)
    except (ValueError, TypeError):
        return True  # out-of-range input correctly rejected
    return bool(0.0 <= result <= 1.0)


def test_log_loss_point_in_unit_interval():
    assert leancheck.check(prop_log_loss_point_in_unit_interval, max_tests=_TESTS_FAST, silent=True)


# ======================================================================= #
# A8 – Bag pipeline order-invariance on degenerate / zero-variance data    #
#                                                                          #
# Scenario: one constant column (std=0 → clamped to 1 by scaler guard)    #
# and one tie-rich column (very small spread → nearly degenerate).         #
# The pipeline's p-value must be order-invariant even in this pathological  #
# case, confirming the exchangeability guarantee holds under the zero-var   #
# guard and tie-breaking paths.                                             #
# ======================================================================= #

_BAG_ADV_N = 10
_BAG_ADV_RNG = np.random.default_rng(99)
# Column 0: constant (zero variance) → bag scaler std=0 guard fires
# Column 1: tie-rich (very small spread, rounded to 2 dp)
_BAG_ADV_X = np.column_stack([
    np.ones(_BAG_ADV_N),
    np.round(_BAG_ADV_RNG.normal(size=_BAG_ADV_N) * 0.01, 2),
])
_BAG_ADV_Y = _BAG_ADV_RNG.normal(size=_BAG_ADV_N)
_BAG_ADV_XQ = np.array([1.0, 0.0])
_BAG_ADV_YQ = float(_BAG_ADV_Y[0])


def _fit_bag_adv_in_order(order: list[int]):
    pipe = StandardScaler(mode="bag") | ConformalRidgeRegressor(a=1.0, warnings=False)
    for i in order:
        pipe.learn_one(_BAG_ADV_X[i], float(_BAG_ADV_Y[i]))
    return pipe


def prop_bag_pipeline_degenerate_order_invariant(keys: list[int]) -> bool:
    """P-value must be order-invariant even on degenerate (zero-variance) data.

    Column 0 is constant → bag scaler's zero-variance guard (std→1) fires.
    Column 1 is tie-rich → many identical scaled values.  In both cases the
    p-value must not depend on the training insertion order.
    """
    ref = _fit_bag_adv_in_order(list(range(_BAG_ADV_N)))
    out = _fit_bag_adv_in_order(_perm_from_keys(keys, _BAG_ADV_N))
    p_ref = ref.compute_p_value(_BAG_ADV_XQ, _BAG_ADV_YQ, tau=0.5)
    p_out = out.compute_p_value(_BAG_ADV_XQ, _BAG_ADV_YQ, tau=0.5)
    return bool(np.isclose(p_ref, p_out, atol=1e-10))


def test_bag_pipeline_degenerate_order_invariant():
    assert leancheck.check(prop_bag_pipeline_degenerate_order_invariant, max_tests=_TESTS_SLOW, silent=True)


# ======================================================================= #
# A7 – SimpleLegendreJumper betting density validity after any sequence   #
#                                                                          #
# Attack: feed any sequence of p-values (including near-boundary values)  #
# and assert that b_n integrates to 1 and M stays positive and finite.    #
# ======================================================================= #

def prop_slj_betting_density_valid(pvs: list[PValue]) -> bool:
    """b_n must integrate to 1 and M must be positive and finite after any sequence."""
    from scipy.integrate import quad

    leancheck.precondition(bool(pvs))
    slj = SimpleLegendreJumper(order=1, J=0.01)
    for p in pvs[:20]:
        slj.update(float(p))
    if not (np.isfinite(slj.logM) and slj.M > 0):
        return False
    integral, _ = quad(slj.b_n, 0.0, 1.0, limit=50)
    return bool(abs(integral - 1.0) < 0.02)


def test_slj_betting_density_valid():
    assert leancheck.check(prop_slj_betting_density_valid, max_tests=_TESTS_MED, silent=True)


# ======================================================================= #
# A8 – VariationalLegendreJumper logM sync and positivity                 #
#                                                                          #
# Attack: feed any mixed sequence and confirm logM == log_martingale_     #
# values[-1] (internal sync) and M remains positive.                      #
# ======================================================================= #


def prop_vlj_logM_synced_and_positive(pvs: list[PValue]) -> bool:
    """logM must equal log_martingale_values[-1] and M > 0 after any sequence."""
    leancheck.precondition(bool(pvs))
    vlj = VariationalLegendreJumper(orders=[1, 2], J=0.01)
    for p in pvs[:20]:
        vlj.update(float(p))
    if not (np.isfinite(vlj.logM) and vlj.M > 0):
        return False
    return bool(np.isclose(vlj.logM, vlj.log_martingale_values[-1], atol=1e-12))


def test_vlj_logM_synced_and_positive():
    assert leancheck.check(prop_vlj_logM_synced_and_positive, max_tests=_TESTS_MED, silent=True)


if __name__ == "__main__":
    leancheck.main(verbose=True, exit_on_failure=False)
