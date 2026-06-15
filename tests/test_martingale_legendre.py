"""Tests for martingale_legendre_dev.py (Algorithms 2 & 3 from the paper)."""

import numpy as np
import pytest
from scipy.integrate import quad

from online_cp.martingale import (
    CompositeLegendreJumper,
    ProductLegendreJumper,
    SimpleLegendreJumper,
    VariationalLegendreJumper,
    compute_normalization_Z,
    shifted_legendre_poly,
)

# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestNormalizationZ:
    """Verify Z(eps) matches known closed-form results."""

    def test_Z_equals_1_for_single_order(self):
        """For |K|=1, Z=1 always (orthogonality)."""
        for k in [1, 2, 3, 4]:
            for eps in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                Z = compute_normalization_Z([k], [eps])
                assert np.isclose(Z, 1.0, atol=1e-12), f"k={k}, eps={eps}: Z={Z}"

    def test_Z_equals_1_for_two_orders(self):
        """For |K|=2, Z=1 always (pairwise orthogonality kills cross-term)."""
        for eps1 in [-0.5, 0.25, 0.5]:
            for eps2 in [-0.5, 0.25, 0.5]:
                Z = compute_normalization_Z([1, 2], [eps1, eps2])
                assert np.isclose(Z, 1.0, atol=1e-12), f"eps=({eps1},{eps2}): Z={Z}"

    def test_Z_for_K123_matches_paper(self):
        """For K={1,2,3}, Z = 1 + (3/35)*eps1*eps2*eps3 (from the paper)."""
        test_cases = [
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.25),
            (0.25, -0.25, 0.5),
            (0.0, 0.5, 0.5),  # Should give Z=1 (one eps is 0)
        ]
        for eps1, eps2, eps3 in test_cases:
            Z_computed = compute_normalization_Z([1, 2, 3], [eps1, eps2, eps3])
            Z_expected = 1.0 + (3.0 / 35.0) * eps1 * eps2 * eps3
            assert np.isclose(Z_computed, Z_expected, atol=1e-12), (
                f"eps=({eps1},{eps2},{eps3}): got {Z_computed}, expected {Z_expected}"
            )


class TestShiftedLegendrePoly:
    """Basic properties of shifted Legendre polynomials."""

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_integrates_to_zero(self, k):
        """integral_0^1 P_k(2u-1) du = 0 for k >= 1."""
        P = shifted_legendre_poly(k)
        antideriv = P.integ()
        integral = float(antideriv(1.0) - antideriv(0.0))
        assert np.isclose(integral, 0.0, atol=1e-12)

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_value_at_one(self, k):
        """P_k(2*1-1) = P_k(1) = 1."""
        P = shifted_legendre_poly(k)
        assert np.isclose(float(P(1.0)), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# SimpleLegendreJumper tests
# ---------------------------------------------------------------------------


class TestSimpleLegendreJumper:
    def test_starts_at_one(self):
        slj = SimpleLegendreJumper(order=1)
        assert slj.logM == 0.0
        assert np.isclose(slj.M, 1.0)

    def test_grows_under_alternative(self):
        """Under H1 (p-values skewed towards 0), martingale should grow."""
        slj = SimpleLegendreJumper(order=1, J=0.1)
        rng = np.random.default_rng(42)
        p_values = rng.beta(0.5, 2, size=200)
        for p in p_values:
            slj.update(p)
        assert slj.logM > 0, f"Expected growth, got logM={slj.logM}"

    def test_order2_detects_variance_shift(self):
        """Order 2 should detect p-values concentrated at boundaries."""
        slj = SimpleLegendreJumper(order=2, J=0.1)
        rng = np.random.default_rng(42)
        # Beta(0.3, 0.3) concentrates at 0 and 1 (high variance)
        p_values = rng.beta(0.3, 0.3, size=200)
        for p in p_values:
            slj.update(p)
        assert slj.logM > 0, f"Expected growth for variance shift, got logM={slj.logM}"

    def test_bounded_under_null(self):
        """Under H0 (uniform), average logM should stay near 0."""
        rng = np.random.default_rng(123)
        final_logMs = []
        for _ in range(20):
            slj = SimpleLegendreJumper(order=1, J=0.1)
            for p in rng.uniform(size=200):
                slj.update(p)
            final_logMs.append(slj.logM)
        assert np.mean(final_logMs) < 5.0

    def test_b_n_nonnegative(self):
        """b_n should be non-negative on [0, 1]."""
        slj = SimpleLegendreJumper(order=2, J=0.1)
        rng = np.random.default_rng(42)
        for p in rng.uniform(size=30):
            slj.update(p)
        xs = np.linspace(0, 1, 50)
        for x in xs:
            assert slj.b_n(x) >= -1e-10, f"b_n({x}) = {slj.b_n(x)}"

    def test_b_n_integrates_to_one(self):
        """integral_0^1 b_n(u) du should equal 1."""
        slj = SimpleLegendreJumper(order=1, J=0.1)
        rng = np.random.default_rng(42)
        for p in rng.uniform(size=30):
            slj.update(p)
        integral, _ = quad(slj.b_n, 0, 1)
        assert np.isclose(integral, 1.0, atol=1e-8)

    def test_B_n_boundary_values(self):
        """B_n(0)=0, B_n(1)=1."""
        slj = SimpleLegendreJumper(order=1, J=0.1)
        rng = np.random.default_rng(42)
        for p in rng.uniform(size=20):
            slj.update(p)
        assert np.isclose(slj.B_n(0), 0.0, atol=1e-10)
        assert np.isclose(slj.B_n(1), 1.0, atol=1e-10)

    def test_single_order_PLJ_matches_SLJ(self):
        """ProductLegendreJumper with K={k} should match SimpleLegendreJumper(k)."""
        rng = np.random.default_rng(99)
        p_values = rng.uniform(size=50)

        slj = SimpleLegendreJumper(order=2, J=0.05)
        plj = ProductLegendreJumper(orders=[2], J=0.05)

        for p in p_values:
            slj.update(p)
            plj.update(p)

        assert np.isclose(slj.logM, plj.logM, atol=1e-10), (
            f"SLJ logM={slj.logM}, PLJ logM={plj.logM}"
        )


# ---------------------------------------------------------------------------
# ProductLegendreJumper tests
# ---------------------------------------------------------------------------


class TestProductLegendreJumper:
    def test_starts_at_one(self):
        plj = ProductLegendreJumper(orders=[1, 2])
        assert plj.logM == 0.0
        assert np.isclose(plj.M, 1.0)

    def test_grows_under_alternative(self):
        """Under H1, martingale should grow."""
        plj = ProductLegendreJumper(orders=[1, 2], J=0.1)
        rng = np.random.default_rng(42)
        p_values = rng.beta(0.5, 2, size=200)
        for p in p_values:
            plj.update(p)
        assert plj.logM > 0, f"Expected growth, got logM={plj.logM}"

    def test_bounded_under_null(self):
        """Under H0, average logM should stay near 0."""
        rng = np.random.default_rng(456)
        final_logMs = []
        for _ in range(10):
            plj = ProductLegendreJumper(orders=[1, 2], J=0.1)
            for p in rng.uniform(size=100):
                plj.update(p)
            final_logMs.append(plj.logM)
        assert np.mean(final_logMs) < 5.0

    def test_b_n_nonnegative(self):
        """b_n should be non-negative on [0, 1]."""
        plj = ProductLegendreJumper(orders=[1, 2], J=0.1)
        rng = np.random.default_rng(42)
        for p in rng.uniform(size=20):
            plj.update(p)
        xs = np.linspace(0, 1, 30)
        for x in xs:
            assert plj.b_n(x) >= -1e-10, f"b_n({x}) = {plj.b_n(x)}"

    def test_b_n_integrates_to_one(self):
        """integral_0^1 b_n(u) du should equal 1."""
        plj = ProductLegendreJumper(orders=[1, 2], J=0.1)
        rng = np.random.default_rng(42)
        for p in rng.uniform(size=20):
            plj.update(p)
        integral, _ = quad(plj.b_n, 0, 1)
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_B_n_boundary_values(self):
        """B_n(0)=0, B_n(1)=1."""
        plj = ProductLegendreJumper(orders=[1, 2], J=0.1)
        rng = np.random.default_rng(42)
        for p in rng.uniform(size=20):
            plj.update(p)
        assert np.isclose(plj.B_n(0), 0.0, atol=1e-10)
        assert np.isclose(plj.B_n(1), 1.0, atol=1e-4)

    def test_state_space_size(self):
        """Verify Cartesian product size is g^K."""
        plj = ProductLegendreJumper(orders=[1, 2, 3])
        assert plj.N == 5**3  # 125

    def test_three_orders_uses_normalization(self):
        """For K={1,2,3}, Z deviates from 1 at non-zero corners."""
        plj = ProductLegendreJumper(orders=[1, 2, 3])
        # Find a corner state like (0.5, 0.5, 0.5)
        idx = plj.states.index((0.5, 0.5, 0.5))
        Z = np.exp(plj._log_Z[idx])
        expected = 1.0 + (3.0 / 35.0) * 0.5 * 0.5 * 0.5
        assert np.isclose(Z, expected, atol=1e-12)

    def test_warns_on_large_state_space(self):
        """Should warn when state space exceeds 500."""
        with pytest.warns(UserWarning, match="Product state space has"):
            ProductLegendreJumper(orders=[1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Algorithm 4: Variational Legendre Jumper tests
# ---------------------------------------------------------------------------


class TestVariationalLegendreJumperInit:
    """Basic initialisation tests for VLJ."""

    def test_default_init(self):
        vlj = VariationalLegendreJumper()
        assert vlj.orders == [1, 2, 3]
        assert vlj.g == 5
        assert vlj.K == 3

    def test_custom_orders(self):
        vlj = VariationalLegendreJumper(orders=[1, 2])
        assert vlj.K == 2
        assert vlj.g == 5


class TestVLJEquivalences:
    """VLJ should reduce to known simpler algorithms in special cases."""

    def test_single_order_equals_slj(self):
        """For |K|=1, VLJ must produce identical output to SLJ."""
        rng = np.random.default_rng(42)
        p_values = rng.uniform(0, 1, size=50)

        slj = SimpleLegendreJumper(order=2, J=0.05)
        vlj = VariationalLegendreJumper(orders=[2], J=0.05)

        for p in p_values:
            slj.update(p)
            vlj.update(p)

        assert np.isclose(slj.logM, vlj.logM, atol=1e-10), (
            f"SLJ logM={slj.logM}, VLJ logM={vlj.logM}"
        )

    def test_two_orders_equals_product_of_sljs(self):
        """For |K|=2, VLJ factors into independent SLJs: S_n = prod_k C_k."""
        rng = np.random.default_rng(123)
        p_values = rng.uniform(0, 1, size=100)

        slj1 = SimpleLegendreJumper(order=1, J=0.05)
        slj2 = SimpleLegendreJumper(order=2, J=0.05)
        vlj = VariationalLegendreJumper(orders=[1, 2], J=0.05)

        for p in p_values:
            slj1.update(p)
            slj2.update(p)
            vlj.update(p)

        expected_logM = slj1.logM + slj2.logM
        assert np.isclose(vlj.logM, expected_logM, atol=1e-10), (
            f"VLJ logM={vlj.logM}, prod SLJ logM={expected_logM}"
        )

    def test_two_orders_trajectory_matches_product(self):
        """Full trajectory for |K|=2 should match product of SLJ trajectories."""
        rng = np.random.default_rng(77)
        p_values = rng.uniform(0, 1, size=30)

        slj1 = SimpleLegendreJumper(order=1, J=0.1)
        slj2 = SimpleLegendreJumper(order=3, J=0.1)
        vlj = VariationalLegendreJumper(orders=[1, 3], J=0.1)

        for p in p_values:
            slj1.update(p)
            slj2.update(p)
            vlj.update(p)

        expected = np.array(slj1.log_martingale_values) + np.array(slj2.log_martingale_values)
        actual = np.array(vlj.log_martingale_values)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestVLJBehaviour:
    """Statistical behaviour of VLJ."""

    def test_grows_under_alternative(self):
        """VLJ should detect deviation from uniformity."""
        rng = np.random.default_rng(0)
        p_values = rng.beta(0.3, 1.5, size=200)

        vlj = VariationalLegendreJumper(orders=[1, 2], J=0.05)
        for p in p_values:
            vlj.update(p)

        assert vlj.logM > 10.0, f"Expected growth, got logM={vlj.logM}"

    def test_bounded_under_null(self):
        """Under H0 (uniform p-values), VLJ should not grow systematically."""
        rng = np.random.default_rng(999)
        p_values = rng.uniform(0, 1, size=500)

        vlj = VariationalLegendreJumper(orders=[1, 2, 3], J=0.05)
        for p in p_values:
            vlj.update(p)

        assert vlj.logM < 10.0, f"Unexpectedly large under null: logM={vlj.logM}"

    def test_differs_from_plj(self):
        """VLJ and PLJ should differ (different transition dynamics)."""
        rng = np.random.default_rng(55)
        p_values = rng.beta(0.5, 0.5, size=100)

        vlj = VariationalLegendreJumper(orders=[1, 2, 3], J=0.1)
        plj = ProductLegendreJumper(orders=[1, 2, 3], J=0.1)

        for p in p_values:
            vlj.update(p)
            plj.update(p)

        # They should produce different values (same bet, different transition)
        assert not np.isclose(vlj.logM, plj.logM, atol=1e-6), (
            f"VLJ and PLJ should differ but got logM={vlj.logM} vs {plj.logM}"
        )

    def test_b_n_integrates_to_one(self):
        """The betting density b_n should integrate to 1 over [0,1]."""
        rng = np.random.default_rng(11)
        p_values = rng.uniform(0, 1, size=10)

        vlj = VariationalLegendreJumper(orders=[1, 2], J=0.1)
        for p in p_values:
            vlj.update(p)

        integral, _ = quad(vlj.b_n, 0, 1)
        assert np.isclose(integral, 1.0, atol=1e-8), f"b_n integrates to {integral}"

    def test_three_orders_valid_martingale(self):
        """For |K|=3, VLJ should still be a valid test martingale (bounded under H0)."""
        # Run multiple short sequences, check none blow up
        for seed in range(10):
            rng2 = np.random.default_rng(seed + 1000)
            p_values = rng2.uniform(0, 1, size=200)
            vlj = VariationalLegendreJumper(orders=[1, 2, 3], J=0.05)
            for p in p_values:
                vlj.update(p)
            # Under null, logM should stay moderate (no systematic growth)
            assert vlj.logM < 20.0, f"seed={seed}: logM={vlj.logM} too large under null"


# ---------------------------------------------------------------------------
# CompositeLegendreJumper tests
# ---------------------------------------------------------------------------


class TestCompositeLegendreJumper:
    def test_default_creates_five_jumpers(self):
        clj = CompositeLegendreJumper()
        assert len(clj._jumpers) == 5
        assert clj.J == [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    def test_single_rate_matches_base(self):
        """With a single jump rate, composite should equal base class."""
        rng = np.random.default_rng(42)
        p_values = rng.beta(0.3, 1.5, size=100)

        slj = SimpleLegendreJumper(order=1, J=0.01)
        clj = CompositeLegendreJumper(J=[0.01], order=1)

        for p in p_values:
            slj.update(p)
            clj.update(p)

        assert np.isclose(slj.logM, clj.logM, atol=1e-12)

    def test_with_product_legendre_jumper(self):
        """Composite wrapping PLJ should work and detect shifts."""
        rng = np.random.default_rng(42)
        p_values = rng.beta(0.3, 1.5, size=100)

        clj = CompositeLegendreJumper(
            base_class=ProductLegendreJumper, orders=[1, 2]
        )
        for p in p_values:
            clj.update(p)

        assert clj.M > 1.0

    def test_with_variational_legendre_jumper(self):
        """Composite wrapping VLJ should work and detect shifts."""
        rng = np.random.default_rng(42)
        p_values = rng.beta(0.3, 1.5, size=100)

        clj = CompositeLegendreJumper(
            base_class=VariationalLegendreJumper, orders=[1, 2]
        )
        for p in p_values:
            clj.update(p)

        assert clj.M > 1.0

    def test_stable_under_uniform(self):
        """Under H0, composite should not grow systematically."""
        rng = np.random.default_rng(123)
        p_values = rng.uniform(size=500)

        clj = CompositeLegendreJumper(order=2)
        for p in p_values:
            clj.update(p)

        assert clj.logM < 10.0

    def test_detects_shift(self):
        """Under alternative, composite should accumulate evidence."""
        rng = np.random.default_rng(42)
        p_values = rng.beta(0.3, 1.5, size=200)

        clj = CompositeLegendreJumper(order=1)
        for p in p_values:
            clj.update(p)

        assert clj.logM > 5.0
