"""Unit tests for the betting strategies in ``online_cp.betting``.

These tests target the *pure* betting strategies directly (rather than only via
the martingale wrappers in :mod:`online_cp.martingale`). Each strategy estimates
a density ``g`` on ``[0, 1]`` and exposes three methods::

    bet(p)        -> g(p)           (density, must be >= 0)
    integrate(p)  -> G(p)           (CDF, monotone, G(0)=0, G(1)=1)
    update(p)     -> incorporate p  (predict-then-learn order)

The tests verify the defining mathematical properties of each strategy:

* positivity and non-negativity of the density,
* CDF boundary conditions (``integrate(0) ~ 0``, ``integrate(1) ~ 1``),
* monotonicity of the CDF,
* the *predict-then-learn* order (a ``bet`` before any data must be the
  uniform/no-betting default, i.e. density 1 and CDF ``p``),
* and, where the strategy is *adaptive*, that it concentrates mass on the
  side of ``[0, 1]`` where the observed p-values cluster.

The ``GaussianKDE`` tests additionally cross-check the reflected-kernel
numerics against a slow pure-Python reference, which is the guard that matters
for the numba-accelerated path (it must produce identical results to Python).
"""

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, uniform

from online_cp.betting import (
    BetaKernel,
    BetaMLE,
    BetaMoments,
    BettingStrategy,
    ExpertAggregationStrategy,
    FixedStrategy,
    GaussianKDE,
    ParticleFilterStrategy,
    PiecewiseConstantBetting,
    _reflected_kde_cdf_numba,
    _reflected_kde_pdf_numba,
)


def small_p_values():
    """50 p-values clustered near 0 (a 'skewed' stream)."""
    rng = np.random.default_rng(0)
    return rng.beta(0.5, 2.0, size=50)


def large_p_values():
    """50 p-values clustered near 1."""
    rng = np.random.default_rng(1)
    return 1.0 - rng.beta(0.5, 2.0, size=50)


def uniform_p_values():
    rng = np.random.default_rng(2)
    return rng.uniform(size=50)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class TestBettingStrategyBase:
    def test_uniform_density(self):
        s = BettingStrategy()
        assert s.bet(0.25) == 1.0
        assert s.bet(0.9) == 1.0

    def test_uniform_cdf_is_identity(self):
        s = BettingStrategy()
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert s.integrate(p) == pytest.approx(p)

    def test_update_is_noop(self):
        s = BettingStrategy()
        s.update(0.3)
        # Still uniform after an update: the base class does not learn.
        assert s.bet(0.3) == 1.0
        assert s.integrate(0.3) == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# GaussianKDE
# --------------------------------------------------------------------------- #


class TestGaussianKDE:
    def test_empty_is_uniform(self):
        g = GaussianKDE(bandwidth=0.1)
        assert g.bet(0.5) == 1.0
        assert g.integrate(0.5) == pytest.approx(0.5)

    def test_single_point_is_uniform(self):
        g = GaussianKDE(bandwidth=0.1)
        g.update(0.5)  # <2 observations -> still uniform
        assert g.bet(0.5) == 1.0
        assert g.integrate(0.5) == pytest.approx(0.5)

    def test_positive_density(self):
        g = GaussianKDE(bandwidth="silverman")
        for p in small_p_values():
            g.update(p)
        for probe in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert g.bet(probe) >= 0.0

    def test_concentrates_on_small_p(self):
        g = GaussianKDE(bandwidth="silverman")
        for p in small_p_values():
            g.update(p)
        # After seeing many small p-values the density should peak near 0.
        assert g.bet(0.05) > g.bet(0.95)

    def test_density_integrates_to_one(self):
        g = GaussianKDE(bandwidth="silverman")
        for p in small_p_values():
            g.update(p)
        total, _ = quad(lambda x: g.bet(x), 0, 1, limit=200)
        assert total == pytest.approx(1.0, abs=1e-2)

    def test_cdf_boundaries(self):
        g = GaussianKDE(bandwidth="silverman")
        for p in small_p_values():
            g.update(p)
        assert g.integrate(0.0) == pytest.approx(0.0, abs=1e-6)
        assert g.integrate(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_cdf_monotone(self):
        g = GaussianKDE(bandwidth="silverman")
        for p in small_p_values():
            g.update(p)
        grid = np.linspace(0.0, 1.0, 40)
        cdf = [g.integrate(x) for x in grid]
        diffs = np.diff(cdf)
        assert np.all(diffs >= -1e-9)

    def test_cdf_matches_density_integral(self):
        g = GaussianKDE(bandwidth="silverman")
        for p in small_p_values():
            g.update(p)
        for probe in (0.2, 0.5, 0.8):
            num, _ = quad(lambda x: g.bet(x), 0, probe, limit=200)
            assert num == pytest.approx(g.integrate(probe), abs=1e-2)

    def test_lcv_bandwidth_selects_positive(self):
        g = GaussianKDE(bandwidth="lcv")
        for p in small_p_values():
            g.update(p)
        assert g._current_bw is not None
        assert g._current_bw > 0

    def test_window_size_limits_memory(self):
        g = GaussianKDE(bandwidth=0.1, window_size=5)
        for p in uniform_p_values():
            g.update(p)
        assert len(g._data) == 50  # _data keeps the full history
        assert len(g._get_data()) == 5  # but the windowed view is capped


class TestReflectedKDENumerics:
    """Guard the numba-accelerated kernels against the pure-Python reference."""

    def test_pdf_matches_python_reference(self):
        data = np.array([0.1, 0.15, 0.2, 0.05, 0.12])
        h = 0.1
        for x in (0.0, 0.1, 0.5, 0.9, 1.0):
            expected = (
                norm.pdf(x, loc=data, scale=h).mean()
                + norm.pdf(x, loc=-data, scale=h).mean()
                + norm.pdf(x, loc=2 - data, scale=h).mean()
            )
            assert _reflected_kde_pdf_numba(x, data, h) == pytest.approx(expected, abs=1e-9)

    def test_cdf_matches_python_reference(self):
        data = np.array([0.1, 0.15, 0.2, 0.05, 0.12])
        h = 0.1
        for x in (0.0, 0.1, 0.5, 0.9, 1.0):
            expected = (
                norm.cdf(x, loc=data, scale=h).mean()
                + norm.cdf(x, loc=-data, scale=h).mean()
                + norm.cdf(x, loc=2 - data, scale=h).mean()
                - 1.0
            )
            assert _reflected_kde_cdf_numba(x, data, h) == pytest.approx(expected, abs=1e-9)

    def test_public_pdf_matches_numpy_path(self):
        """Force the pure-Python branch and compare to the numba-accelerated one."""
        data = np.array([0.1, 0.15, 0.2, 0.05, 0.12])
        h = 0.1
        py_pdf = (
            norm.pdf(0.5, loc=data, scale=h).mean()
            + norm.pdf(0.5, loc=-data, scale=h).mean()
            + norm.pdf(0.5, loc=2 - data, scale=h).mean()
        )
        numba_pdf = _reflected_kde_pdf_numba(0.5, data, h)
        assert numba_pdf == pytest.approx(py_pdf, abs=1e-9)


# --------------------------------------------------------------------------- #
# BetaMoments
# --------------------------------------------------------------------------- #


class TestBetaMoments:
    def test_starts_as_uniform_beta(self):
        bm = BetaMoments()
        # Beta(1,1) == uniform.
        assert bm.bet(0.5) == pytest.approx(1.0)
        assert bm.integrate(0.5) == pytest.approx(0.5)

    def test_positive_density(self):
        bm = BetaMoments()
        for p in small_p_values():
            bm.update(p)
        for probe in (0.01, 0.5, 0.99):
            assert bm.bet(probe) >= 0.0

    def test_concentrates_on_small_p(self):
        bm = BetaMoments()
        for p in small_p_values():
            bm.update(p)
        assert bm.bet(0.05) > bm.bet(0.95)

    def test_moments_track_mean(self):
        bm = BetaMoments()
        for p in small_p_values():
            bm.update(p)
        # After fitting, the Beta mean a/(a+b) should be below 0.5.
        assert bm._ahat / (bm._ahat + bm._bhat) < 0.5

    def test_single_observation_no_fit(self):
        bm = BetaMoments()
        bm.update(0.01)  # n=1 -> parameters unchanged (still uniform)
        assert (bm._ahat, bm._bhat) == (1.0, 1.0)

    def test_degenerate_variance_keeps_uniform(self):
        bm = BetaMoments()
        for _ in range(5):
            bm.update(0.5)  # zero sample variance
        # No fitting occurs with zero variance: stays at initial params.
        assert (bm._ahat, bm._bhat) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# BetaMLE
# --------------------------------------------------------------------------- #


class TestBetaMLE:
    def test_starts_as_uniform_beta(self):
        bm = BetaMLE()
        assert bm.bet(0.5) == pytest.approx(1.0)
        assert bm.integrate(0.5) == pytest.approx(0.5)

    def test_positive_density(self):
        bm = BetaMLE()
        for p in small_p_values():
            bm.update(p)
        for probe in (0.01, 0.5, 0.99):
            assert bm.bet(probe) >= 0.0

    def test_concentrates_on_small_p(self):
        bm = BetaMLE()
        for p in small_p_values():
            bm.update(p)
        assert bm.bet(0.05) > bm.bet(0.95)

    def test_mle_recovers_skewed_shape(self):
        bm = BetaMLE()
        for p in small_p_values():
            bm.update(p)
        # MLE should produce a Beta with mean below 0.5.
        assert bm._ahat / (bm._ahat + bm._bhat) < 0.5

    def test_parameters_positive(self):
        bm = BetaMLE()
        for p in small_p_values():
            bm.update(p)
        assert bm._ahat > 0
        assert bm._bhat > 0

    def test_clips_extreme_inputs(self):
        # p must stay in (0,1); feeding 0 and 1 must not produce inf/nan.
        bm = BetaMLE()
        for p in (0.0, 1.0, 0.0, 1.0):
            bm.update(p)
        assert math.isfinite(bm._ahat)
        assert math.isfinite(bm._bhat)



# --------------------------------------------------------------------------- #
# ParticleFilterStrategy
# --------------------------------------------------------------------------- #


class TestParticleFilterStrategy:
    def test_positive_density(self):
        pf = ParticleFilterStrategy(num_particles=200, seed=42)
        for p in small_p_values():
            pf.update(p)
        for probe in (0.01, 0.5, 0.99):
            assert pf.bet(probe) >= 0.0

    def test_concentrates_on_small_p(self):
        pf = ParticleFilterStrategy(num_particles=200, seed=42)
        for p in small_p_values():
            pf.update(p)
        assert pf.bet(0.05) > pf.bet(0.95)

    def test_deterministic_with_int_seed(self):
        data = small_p_values()
        pf1 = ParticleFilterStrategy(num_particles=100, seed=7)
        pf2 = ParticleFilterStrategy(num_particles=100, seed=7)
        for p in data:
            pf1.update(p)
            pf2.update(p)
        assert pf1.bet(0.1) == pytest.approx(pf2.bet(0.1))

    def test_generator_seed(self):
        rng = np.random.default_rng(123)
        pf = ParticleFilterStrategy(num_particles=50, seed=rng)
        assert pf.rng is rng

    def test_adaptive_noise_mode(self):
        pf = ParticleFilterStrategy(num_particles=100, process_noise_std="auto", seed=1)
        for p in small_p_values():
            pf.update(p)
        assert all(math.isfinite(pf.bet(x)) for x in (0.05, 0.5, 0.95))
        # sigma history should be recorded in adaptive mode.
        assert len(pf.sigma_history) > 0

    def test_history_recorded(self):
        pf = ParticleFilterStrategy(num_particles=100, seed=3)
        data = small_p_values()
        for p in data:
            pf.update(p)
        assert len(pf.alpha_history) == len(data)
        assert len(pf.beta_history) == len(data)

    def test_particles_never_overflow(self):
        pf = ParticleFilterStrategy(num_particles=100, seed=5)
        # Feed a long stream; _get_alphas_betas must clip so exp() stays finite.
        for p in small_p_values():
            pf.update(p)
        alphas, betas = pf._get_alphas_betas()
        assert np.all(np.isfinite(alphas))
        assert np.all(np.isfinite(betas))


# --------------------------------------------------------------------------- #
# FixedStrategy
# --------------------------------------------------------------------------- #


class TestFixedStrategy:
    def test_from_distribution(self):
        fs = FixedStrategy(distribution=uniform())
        assert fs.bet(0.5) == pytest.approx(1.0)
        assert fs.integrate(0.5) == pytest.approx(0.5)

    def test_explicit_pdf_and_cdf(self):
        # pdf = 2(1-x) integrates to 1 on [0,1]; cdf = 1-(1-x)^2.
        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), cdf=lambda x: 1 - (1 - x) ** 2)
        assert fs.bet(0.1) == pytest.approx(1.8)
        assert fs.integrate(0.5) == pytest.approx(0.75)

    def test_explicit_pdf_derives_cdf_by_integration(self):
        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x))
        # cdf(0.5) = integral_0^0.5 2(1-x) dx = [2x - x^2]_0^0.5 = 0.75
        assert fs.integrate(0.5) == pytest.approx(0.75, abs=1e-6)

    def test_pdf_integrating_to_one_passes_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            fs = FixedStrategy(pdf=lambda x: 2 * (1 - x))  # integrates to 1
        assert fs.bet(0.5) == pytest.approx(1.0)

    def test_misnormalised_pdf_warns(self):
        with pytest.warns(UserWarning, match="does not integrate to 1"):
            # integrates to 2 on [0,1], not 1
            FixedStrategy(pdf=lambda x: 2.0)

    def test_check_integration_can_be_disabled(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Would warn if checked, but check_integration=False silences it.
            fs = FixedStrategy(pdf=lambda x: 1.0, check_integration=False)
        assert fs.bet(0.5) == pytest.approx(1.0)



# --------------------------------------------------------------------------- #
# PiecewiseConstantBetting
# --------------------------------------------------------------------------- #


class TestPiecewiseConstantBetting:
    def test_values_match_definition(self):
        pcb = PiecewiseConstantBetting(a=0.3, b=0.5)
        assert pcb.bet(0.1) == pytest.approx(0.5 / 0.3)
        assert pcb.bet(0.8) == pytest.approx(0.5 / 0.7)

    def test_threshold_is_left_closed(self):
        pcb = PiecewiseConstantBetting(a=0.3, b=0.5)
        assert pcb.bet(0.3) == pytest.approx(0.5 / 0.3)  # p == a -> left
        assert pcb.bet(0.3000001) == pytest.approx(0.5 / 0.7)

    def test_integrates_to_one(self):
        pcb = PiecewiseConstantBetting(a=0.3, b=0.5)
        total, _ = quad(lambda x: pcb.bet(x), 0, 1, limit=50)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_cdf_boundaries_and_continuity(self):
        pcb = PiecewiseConstantBetting(a=0.3, b=0.5)
        assert pcb.integrate(0.0) == pytest.approx(0.0)
        assert pcb.integrate(1.0) == pytest.approx(1.0)
        # continuous at the threshold: left limit == right limit == b
        assert pcb.integrate(pcb.a) == pytest.approx(pcb.b)
        assert pcb.integrate(pcb.a + 1e-12) == pytest.approx(pcb.b, abs=1e-9)

    def test_invalid_a_raises(self):
        with pytest.raises(ValueError):
            PiecewiseConstantBetting(a=0.0, b=0.5)
        with pytest.raises(ValueError):
            PiecewiseConstantBetting(a=1.0, b=0.5)

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError):
            PiecewiseConstantBetting(a=0.5, b=0.0)
        with pytest.raises(ValueError):
            PiecewiseConstantBetting(a=0.5, b=1.0)


# --------------------------------------------------------------------------- #
# ExpertAggregationStrategy
# --------------------------------------------------------------------------- #


class TestExpertAggregationStrategy:
    def test_initial_weights_uniform(self):
        experts = [
            FixedStrategy(distribution=uniform()),
            FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False),
        ]
        agg = ExpertAggregationStrategy(experts=experts)
        w = agg.get_current_weights()
        assert np.allclose(w, [0.5, 0.5])

    def test_weights_are_probabilities(self):
        experts = [
            FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False),
            FixedStrategy(distribution=uniform()),
        ]
        agg = ExpertAggregationStrategy(experts=experts, learning_rate=1.0)
        for p in small_p_values():
            agg.update(p)
        w = agg.get_current_weights()
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= 0.0)

    def test_rewards_better_expert(self):
        # A density peaking near 0 is a "good" expert for a stream of small p.
        good = FixedStrategy(
            pdf=lambda x: norm.pdf(x, 0.1, 0.1) / (norm.cdf(1, 0.1, 0.1) - norm.cdf(0, 0.1, 0.1)),
            check_integration=False,
        )
        bad = FixedStrategy(distribution=uniform())
        agg = ExpertAggregationStrategy(experts=[good, bad], learning_rate=0.5)
        # Use a very concentrated stream (uniform(0, 0.2)) so the good expert
        # consistently outperforms the uniform one.
        rng = np.random.default_rng(99)
        concentrated = rng.uniform(0.0, 0.2, size=50)
        for p in concentrated:
            agg.update(p)
        w = agg.get_current_weights()
        assert w[0] > w[1]

    def test_bet_is_weighted_average(self):
        experts = [
            FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False),
            FixedStrategy(distribution=uniform()),
        ]
        agg = ExpertAggregationStrategy(experts=experts, learning_rate=0.0)
        # learning_rate=0 -> weights stay uniform -> bet == mean of expert bets.
        p = 0.5
        expected = (experts[0].bet(p) + experts[1].bet(p)) / 2
        assert agg.bet(p) == pytest.approx(expected)

    def test_integration_is_weighted_average(self):
        experts = [
            FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False),
            FixedStrategy(distribution=uniform()),
        ]
        agg = ExpertAggregationStrategy(experts=experts, learning_rate=0.0)
        p = 0.5
        expected = (experts[0].integrate(p) + experts[1].integrate(p)) / 2
        assert agg.integrate(p) == pytest.approx(expected)

    def test_weight_history_recorded(self):
        experts = [FixedStrategy(distribution=uniform()) for _ in range(3)]
        agg = ExpertAggregationStrategy(experts=experts)
        for p in small_p_values()[:10]:
            agg.update(p)
        assert len(agg.expert_weights_history) == 10
        assert np.allclose(agg.expert_weights_history[-1].sum(), 1.0)



# --------------------------------------------------------------------------- #
# Cross-cutting property sweeps
# --------------------------------------------------------------------------- #


class TestStrategyInvariants:
    """Properties that should hold for *every* adaptive strategy."""

    @pytest.mark.parametrize(
        "strategy",
        [
            GaussianKDE(bandwidth="silverman"),
            BetaMoments(),
            BetaMLE(),
            ParticleFilterStrategy(num_particles=200, seed=11),
            PiecewiseConstantBetting(a=0.5, b=0.5),
        ],
        ids=["GaussianKDE", "BetaMoments", "BetaMLE", "ParticleFilter", "PiecewiseConstant"],
    )
    def test_density_nonnegative(self, strategy):
        for p in small_p_values():
            strategy.update(p)
        for probe in np.linspace(0.0, 1.0, 11):
            assert strategy.bet(probe) >= 0.0

    @pytest.mark.parametrize(
        "strategy",
        [
            GaussianKDE(bandwidth="silverman"),
            BetaMoments(),
            BetaMLE(),
            ParticleFilterStrategy(num_particles=200, seed=11),
        ],
        ids=["GaussianKDE", "BetaMoments", "BetaMLE", "ParticleFilter"],
    )
    def test_cdf_anchored_at_boundaries(self, strategy):
        for p in small_p_values():
            strategy.update(p)
        assert strategy.integrate(0.0) == pytest.approx(0.0, abs=1e-3)
        assert strategy.integrate(1.0) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# BetaKernel
# --------------------------------------------------------------------------- #


def _beta_kde_available():
    try:
        import beta_kde  # noqa: F401

        return True
    except ImportError:
        return False


class TestBetaKernel:
    """Tests that hold whether or not the optional ``beta_kde`` package is present."""

    def test_empty_is_uniform(self):
        bk = BetaKernel()
        assert bk.bet(0.5) == 1.0
        assert bk.integrate(0.5) == pytest.approx(0.5)

    def test_single_point_is_uniform(self):
        bk = BetaKernel()
        bk.update(0.5)
        assert bk.bet(0.5) == 1.0
        assert bk.integrate(0.5) == pytest.approx(0.5)

    def test_positive_density(self):
        bk = BetaKernel()
        for p in small_p_values():
            bk.update(p)
        for probe in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert bk.bet(probe) >= 0.0

    @pytest.mark.skipif(not _beta_kde_available(), reason="beta_kde not installed")
    def test_concentrates_on_small_p_with_backend(self):
        bk = BetaKernel()
        for p in small_p_values():
            bk.update(p)
        assert bk.bet(0.05) > bk.bet(0.95)

    def test_window_size_capped(self):
        bk = BetaKernel(window_size=5)
        for p in uniform_p_values():
            bk.update(p)
        assert len(bk._data) == 50
        # The last window is what is fitted.
        assert len(bk._data[-5:]) == 5

    def test_warns_without_beta_kde(self):
        if _beta_kde_available():
            pytest.skip("beta_kde installed; fallback path not exercised")
        # Re-instantiate to force the fallback warning path. The module-level
        # stub sets _warned=True after the first instantiation, so reset it
        # to observe the warning on a fresh object.
        from online_cp import betting as betting_mod

        if hasattr(betting_mod.BetaKDE, "_warned"):
            betting_mod.BetaKDE._warned = False
        with pytest.warns(UserWarning, match="beta_kde package not installed"):
            BetaKernel()

    def test_fallback_density_is_uniform_without_beta_kde(self):
        if _beta_kde_available():
            pytest.skip("beta_kde installed; fallback path not exercised")
        bk = BetaKernel()
        for p in small_p_values():
            bk.update(p)
        # Without the backend the stub returns a flat density of 1.
        assert bk.bet(0.5) == pytest.approx(1.0)

