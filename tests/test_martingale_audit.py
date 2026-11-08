"""Audit regression tests: trajectory faithfulness vs. reference algorithms,
the false '0.2 floor' claim, and J=1 analytic constants.

These tests pin the *exact* trajectory a deterministic sequence should
produce, using independent reference implementations of the published
algorithms (ALRW2 Algorithms 8.1, 9.4, 9.5; Legendre Jumper preprint
Algorithms 2, 3), plus analytic checks (J=1 constants, the ``Z`` normalizer,
and the epsilon-grid admissibility bounds).
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy.special import eval_legendre

from online_cp.martingale import (
    IndependentTransitionProductLegendreJumper,
    ProductLegendreJumper,
    SimpleJumper,
    SimpleLegendreJumper,
    SleeperStayer,
)
from online_cp.martingale.legendre import compute_normalization_Z

STANDARD_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)


def deterministic_seq_7bit(n_blocks=1):
    block = [int(format(j, "07b")[::-1], 2) / 127 for j in range(128)]
    return block * n_blocks


def _run(model, seq):
    vals = [1.0]
    for p in seq:
        model.update(float(p))
        vals.append(model.M)
    return vals[1:]


def ref_simple_jumper(p_seq, J=0.01, E=None):
    if E is None:
        E = [-1, -0.5, 0, 0.5, 1]
    C = {e: 1.0 / len(E) for e in E}
    C_total = 1.0
    out = []
    for p in p_seq:
        for e in E:
            C[e] = (1 - J) * C[e] + (J / len(E)) * C_total
        for e in E:
            C[e] = C[e] * (1 + e * (p - 0.5))
        out.append(C_total := sum(C.values()))
    return out


def ref_sleeper_stayer(p_seq, R=0.001, G=10):
    grid_vals = [i / G for i in range(1, G)]
    G_grid = [(a, b) for a in grid_vals for b in grid_vals]
    n = len(G_grid)
    S_active = {ab: 0.0 for ab in G_grid}
    S_sleep = 1.0
    out = []
    for p in p_seq:
        for (a, b) in G_grid:
            S_active[(a, b)] = S_active[(a, b)] * (b / a if p <= a else (1 - b) / (1 - a))
        out.append(S_sleep + sum(S_active.values()))
        for (a, b) in G_grid:
            S_active[(a, b)] = S_active[(a, b)] + R * S_sleep / n
        S_sleep = (1 - R) * S_sleep
    return out


def ref_SLJ(p_seq, order, J=0.01, E=None):
    if E is None:
        E = STANDARD_GRID
    C = {e: 1.0 / len(E) for e in E}
    C_total = 1.0
    out = []
    for p in p_seq:
        for e in E:
            C[e] = (1 - J) * C[e] + (J / len(E)) * C_total
        for e in E:
            C[e] = C[e] * (1 + e * eval_legendre(order, 2.0 * p - 1.0))
        out.append(C_total := sum(C.values()))
    return out


def ref_PLJ(p_seq, orders, J=0.01, E=None):
    if E is None:
        E = STANDARD_GRID
    states = np.array(list(itertools.product(E, repeat=len(orders))))
    Z = np.array([compute_normalization_Z(orders, s) for s in states])
    capital = np.full(len(states), 1.0 / len(states))
    out = []
    for p in p_seq:
        x = 2 * p - 1
        L = np.array([eval_legendre(k, x) for k in orders])
        bets = (1 + states * L).prod(axis=1) / Z
        total = capital.sum()
        capital = ((1 - J) * capital + (J / len(states)) * total) * bets
        out.append(capital.sum())
    return out


class TestTrajectoryFaithfulness:
    def test_simple_jumper_matches_algorithm_8_1(self):
        seq = deterministic_seq_7bit(1)
        vals = _run(SimpleJumper(J=0.1, store_p_values=False), seq)
        ref = ref_simple_jumper(seq, J=0.1)
        assert np.allclose(vals, ref, rtol=1e-12, atol=1e-12)

    def test_sleeper_stayer_matches_algorithm_9_4(self):
        seq = deterministic_seq_7bit(1)
        vals = _run(SleeperStayer(R=0.01, G=5, store_p_values=False), seq)
        ref = ref_sleeper_stayer(seq, R=0.01, G=5)
        assert np.allclose(vals, ref, rtol=1e-10, atol=1e-12)

    def test_simple_legendre_jumper_matches_algorithm_2(self):
        seq = deterministic_seq_7bit(1)
        vals = _run(SimpleLegendreJumper(order=2, J=0.01, store_p_values=False), seq)
        ref = ref_SLJ(seq, order=2, J=0.01)
        assert np.allclose(vals, ref, rtol=1e-12, atol=1e-12)

    def test_product_legendre_jumper_matches_algorithm_3(self):
        seq = deterministic_seq_7bit(1)
        vals = _run(ProductLegendreJumper(orders=[1, 2, 3], J=0.01, store_p_values=False), seq)
        ref = ref_PLJ(seq, orders=[1, 2, 3], J=0.01)
        assert np.allclose(vals, ref, rtol=1e-12, atol=1e-12)

    def test_itplj_j1_equals_plj_j1(self):
        seq = deterministic_seq_7bit(1)
        a = _run(IndependentTransitionProductLegendreJumper(orders=[1, 2], J=1.0, store_p_values=False), seq)
        b = _run(ProductLegendreJumper(orders=[1, 2], J=1.0, store_p_values=False), seq)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


class TestZeroPointTwoFloorClaim:
    """The Legendre Jumper preprint's '0.2 floor' claim is false for the
    normalized PLJ composite: running the per-rate PLJ components over the
    100-block (12,800-step) 7-bit-reversal sequence yields a composite mean
    capital of ~0.1973, not 0.2 (the referee's Appendix counterexample)."""

    def test_composite_mean_is_below_zero_two(self):
        rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        seq = np.array(deterministic_seq_7bit(n_blocks=100))
        comp = []
        for J in rates:
            m = ProductLegendreJumper(orders=[1, 2, 3], J=J, store_p_values=False)
            for p in seq:
                m.update(float(p))
            comp.append(m.M)
        assert np.isclose(np.mean(comp), 0.197326997687, rtol=1e-9)


class TestJ1AnalyticConstants:
    def test_J1_PLJ_a_constant(self):
        grid_fr = [Fraction(-1, 2), Fraction(-1, 4), Fraction(0, 1), Fraction(1, 4), Fraction(1, 2)]
        a_exact = sum(
            Fraction(1, 1) / (1 + Fraction(3, 35) * (e1 * e2 * e3))
            for e1 in grid_fr for e2 in grid_fr for e3 in grid_fr
        ) / 125
        assert abs(float(a_exact) - 1.000014350501523) < 1e-12

    def test_J1_PLJ_h_at_1(self):
        grid_fr = [Fraction(-1, 2), Fraction(-1, 4), Fraction(0, 1), Fraction(1, 4), Fraction(1, 2)]
        a_exact = sum(
            Fraction(1, 1) / (1 + Fraction(3, 35) * (e1 * e2 * e3))
            for e1 in grid_fr for e2 in grid_fr for e3 in grid_fr
        ) / 125
        h1 = float(a_exact) + (1 - float(a_exact)) / Fraction(3, 35)
        assert abs(h1 - 0.999846927983758) < 1e-9


class TestZCorrectness:
    def test_all_grid_states_K1234(self):
        orders = [1, 2, 3, 4]
        states = list(itertools.product(STANDARD_GRID, repeat=4))
        from scipy.integrate import quad

        max_err = 0.0
        for eps in states:
            Z = compute_normalization_Z(orders, eps)
            integral, _ = quad(
                lambda u, _e=eps, _o=orders, _Z=Z:
                float(np.prod([1.0 + e * eval_legendre(k, 2 * u - 1) for k, e in zip(_o, _e)])) / _Z,
                0, 1, limit=200,
            )
            max_err = max(max_err, abs(integral - 1.0))
        assert max_err < 1e-13


class TestEpsilonGridValidation:
    def test_default_grid_accepted(self):
        SimpleLegendreJumper(order=2)  # should not raise
        ProductLegendreJumper(orders=[1, 2, 3])
        IndependentTransitionProductLegendreJumper(orders=[1, 2, 3])

    def test_out_of_range_grid_rejected(self):
        bad = (-1.0, -0.5, 0.0, 0.5, 1.0)  # |eps| == 1 at the endpoints
        with pytest.raises(ValueError, match="admissible range"):
            SimpleLegendreJumper(order=2, epsilon_grid=bad)
        with pytest.raises(ValueError, match="admissible range"):
            ProductLegendreJumper(orders=[1, 2, 3], epsilon_grid=bad)
        with pytest.raises(ValueError, match="admissible range"):
            IndependentTransitionProductLegendreJumper(orders=[1, 2, 3], epsilon_grid=bad)


class TestEpsilonGridValidationOrderDependent:
    """The admissible range of ``epsilon_grid`` depends on the polynomial
    orders: it is the intersection over ``k in orders`` of
    ``(-1 / max_p P_k, -1 / min_p P_k)``.

    Shifted Legendre polynomials satisfy ``max_p P_k = +1`` always, so the
    lower bound is always ``-1``.  The upper bound is ``-1 / (min_p P_k)``
    for the order with the most-negative minimum: odd orders reach ``-1``
    (upper bound ``+1``), while even orders only reach an interior minimum of
    magnitude ``< 1`` (upper bound ``> 1``, e.g. ``+2.0`` for order 2).

    This pins down that the bound is *sharp and order-dependent*, not the
    coarser ``|eps| < 1`` (which would wrongly reject valid even-only grids
    with ``eps > 1``).
    """

    def test_default_grid_always_accepted(self):
        for orders in ([2], [1, 2], [2, 4], [1, 2, 3], [2, 4, 6]):
            ProductLegendreJumper(orders=orders)  # default grid, no raise

    def test_lower_bound_is_minus_one_independent_of_orders(self):
        # -1 is always rejected (max P_k = 1 everywhere -> 1 - 1*1 = 0 at p).
        bad = (-1.0, 0.0, 0.5)
        for orders in ([2], [4], [2, 4, 6], [1]):
            with pytest.raises(ValueError, match="admissible range"):
                ProductLegendreJumper(orders=orders, epsilon_grid=bad)

    def test_even_only_grid_allows_eps_above_one(self):
        # Order 2 has min_p P_2 = -0.5 -> upper bound +2.0.  So eps up to (but
        # not including) 2.0 is admissible; the old |eps| < 1 check would have
        # wrongly rejected these.
        ProductLegendreJumper(orders=[2], epsilon_grid=(0.0, 1.5, 1.9))
        with pytest.raises(ValueError, match="admissible range"):
            ProductLegendreJumper(orders=[2], epsilon_grid=(0.0, 2.0))  # boundary
        with pytest.raises(ValueError, match="admissible range"):
            ProductLegendreJumper(orders=[2], epsilon_grid=(0.0, 2.5))

    def test_order_two_upper_bound_is_exactly_two(self):
        # Boundary at eps = 2.0 (min P_2 = -1/2) is strictly excluded.
        with pytest.raises(ValueError, match="admissible range"):
            SimpleLegendreJumper(order=2, epsilon_grid=(1.999, 2.0))

    def test_odd_order_restores_unit_upper_bound(self):
        # Any odd order reaches -1, so the upper bound collapses to +1.  Even if
        # the set also contains even orders, the odd order is binding.
        with pytest.raises(ValueError, match="admissible range"):
            ProductLegendreJumper(orders=[1, 2], epsilon_grid=(0.0, 1.5))
        with pytest.raises(ValueError, match="admissible range"):
            ProductLegendreJumper(orders=[2, 3], epsilon_grid=(0.0, 1.01))
        ProductLegendreJumper(orders=[1, 2], epsilon_grid=(0.0, 0.99))

    def test_mixed_even_orders_bound_is_most_negative_minimum(self):
        # For [4, 6, 8], the binding minimum is order 4 (min P_4 = -3/7), giving
        # upper bound 7/3 ~ 2.333.  eps below that is admissible, above is not.
        ProductLegendreJumper(orders=[4, 6, 8], epsilon_grid=(0.0, 2.3))
        with pytest.raises(ValueError, match="admissible range"):
            ProductLegendreJumper(orders=[4, 6, 8], epsilon_grid=(0.0, 2.34))

    def test_empty_orders_is_trivially_valid(self):
        # An empty orders list imposes no constraint (no polynomial, no bound).
        from online_cp.martingale.legendre import _validate_epsilon_grid

        _validate_epsilon_grid([], [10.0])  # should not raise


class TestSimpleMixtureMartingale:
    """``SimpleMixtureMartingale`` is the canonical parameter-free exchangeability
    test martingale (Vovk 2013, eq. 7.5; Fedorova et al. 2012).

    It uniformly mixes the *power* betting functions
    ``f_eps(p) = eps * p**(eps - 1)`` over ``eps in [0, 1]``.  Each component
    is a valid martingale (its betting function integrates to 1), and a
    fixed-weight mixture of martingales is a martingale, so the simple mixture
    is a **true martingale**: ``E[M_n | F_{n-1}] == M_{n-1}``, ``M_0 = 1``, and
    the betting function satisfies ``int_0^1 b_n(p) dp = 1`` exactly (Fubini).

    The 2026-11-07 audit originally flagged this class as "not a valid
    martingale" because it tested the criterion naively with numerical
    quadrature that mishandled the ``p**(eps - 1)`` singularity at ``p = 0``;
    both the capital formula and the martingale property are in fact correct.
    The only real defect was the exposed CDF ``B_n``, which is now given in
    closed form (``B_n(p) = p * M_n(slp + log p) / M_n(slp)``) to avoid that
    singularity.  This test pins down the correct invariants.
    """

    def test_initial_capital_one(self):
        from online_cp.martingale import SimpleMixtureMartingale

        assert SimpleMixtureMartingale().M == 1.0

    def test_betting_function_integrates_to_one(self):
        """``B_n(1.0)`` (the integral of the betting function) must be 1.0
        exactly, which is the defining true-martingale property."""
        import numpy as np

        from online_cp.martingale import SimpleMixtureMartingale

        rng = np.random.default_rng(123)
        for start in [1, 2, 3, 5, 8]:
            m = SimpleMixtureMartingale()
            for _ in range(start):
                m.update(float(rng.uniform(0.05, 0.95)))
            assert abs(m.B_n(1.0) - 1.0) < 1e-12, (
                f"B_n(1) = {m.B_n(1.0)} != 1 after {start} steps "
                f"(betting function does not integrate to 1)"
            )

    def test_B_n_closed_form_matches_reference(self):
        """``B_n(p)`` must agree with the singularity-free wealth-weighted
        reference ``int_0^1 eps * p**eps * w(eps) d eps`` to high precision."""
        import math

        import numpy as np
        from scipy.integrate import quad

        from online_cp.martingale import SimpleMixtureMartingale

        def B_ref(p, b, n):
            den = quad(lambda t: t**n * math.exp(-b * t), 0, 1, limit=300)[0]
            num = quad(
                lambda eps: (eps**n * math.exp(-b * eps) / den) * p**eps,
                0, 1, limit=300,
            )[0]
            return num

        rng = np.random.default_rng(7)
        worst = 0.0
        for start in [1, 2, 4, 7]:
            m = SimpleMixtureMartingale()
            for _ in range(start):
                m.update(float(rng.uniform(0.05, 0.95)))
            b = -m.sum_log_p
            n = m.n
            for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
                worst = max(worst, abs(m.B_n(p) - B_ref(p, b, n)))
        assert worst < 1e-9, f"B_n closed form deviates from reference by {worst}"

    def test_B_n_is_valid_cdf(self):
        """``B_n`` must be a proper CDF: nonnegative, monotone, with
        ``B_n(0) = 0`` and ``B_n(1) = 1``."""
        import numpy as np

        from online_cp.martingale import SimpleMixtureMartingale

        rng = np.random.default_rng(2024)
        m = SimpleMixtureMartingale()
        for _ in range(6):
            m.update(float(rng.uniform(0.02, 0.98)))
        assert m.B_n(0.0) == 0.0
        assert abs(m.B_n(1.0) - 1.0) < 1e-12
        prev = 0.0
        for p in np.linspace(0.05, 0.95, 19):
            val = m.B_n(float(p))
            assert val >= prev - 1e-12, f"B_n not monotone at p={p}"
            prev = val

    def test_grows_on_small_p(self):
        from online_cp.martingale import SimpleMixtureMartingale

        m = SimpleMixtureMartingale()
        m.update(0.001)
        assert m.M > 1.0, (
            f"Simple mixture should grow on a small p-value, got M_1 = {m.M}"
        )
