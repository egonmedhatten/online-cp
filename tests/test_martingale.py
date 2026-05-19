import numpy as np
import pytest
from scipy.stats import uniform

from online_cp.martingale import (
    BetaMLE,
    BetaMoments,
    CompositeJumper,
    ExpertAggregationStrategy,
    FixedStrategy,
    GaussianKDE,
    ParticleFilterStrategy,
    PluginMartingale,
    SimpleJumper,
    SimpleMixtureMartingale,
)

# ─── Base martingale properties ───────────────────────────────────────────────


class TestMartingaleBaseProperties:
    """Properties that hold for any ConformalTestMartingale subclass."""

    @pytest.mark.parametrize(
        "MartingaleClass,kwargs",
        [
            (SimpleJumper, {"warnings": False}),
            (CompositeJumper, {"warnings": False}),
            (SimpleMixtureMartingale, {"warnings": False}),
        ],
    )
    def test_starts_at_one(self, MartingaleClass, kwargs):
        m = MartingaleClass(**kwargs)
        assert m.logM == 0.0
        assert np.isclose(m.M, 1.0)
        assert m.log_martingale_values == [0.0]

    @pytest.mark.parametrize(
        "MartingaleClass,kwargs",
        [
            (SimpleJumper, {"warnings": False}),
            (CompositeJumper, {"warnings": False}),
            (SimpleMixtureMartingale, {"warnings": False}),
        ],
    )
    def test_M_equals_exp_logM(self, MartingaleClass, kwargs, uniform_p_values):
        m = MartingaleClass(**kwargs)
        for p in uniform_p_values[:50]:
            m.update(p)
        assert np.isclose(m.M, np.exp(m.logM))

    @pytest.mark.parametrize(
        "MartingaleClass,kwargs",
        [
            (SimpleJumper, {"warnings": False}),
            (CompositeJumper, {"warnings": False}),
            (SimpleMixtureMartingale, {"warnings": False}),
        ],
    )
    def test_max_geq_current(self, MartingaleClass, kwargs, uniform_p_values):
        m = MartingaleClass(**kwargs)
        for p in uniform_p_values[:50]:
            m.update(p)
            assert m.log_max >= m.logM


# ─── SimpleJumper ─────────────────────────────────────────────────────────────


class TestSimpleJumper:
    def test_grows_under_alternative(self, skewed_p_values):
        """Under H1 (skewed towards 0), martingale should grow."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in skewed_p_values:
            m.update(p)
        assert m.logM > 0, f"Expected growth under H1, got logM={m.logM}"

    def test_bounded_under_null(self):
        """Under H0 (uniform), average logM should stay near 0."""
        rng = np.random.default_rng(123)
        final_logMs = []
        for _ in range(20):
            m = SimpleJumper(J=0.1, warnings=False)
            pvals = rng.uniform(size=200)
            for p in pvals:
                m.update(p)
            final_logMs.append(m.logM)
        # Mean of logM should not be significantly positive
        assert np.mean(final_logMs) < 5.0

    def test_B_n_boundary_values(self, uniform_p_values):
        """B_n(0) should be 0, B_n(1) should be 1."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in uniform_p_values[:20]:
            m.update(p)
        assert np.isclose(m.B_n(0), 0, atol=1e-10)
        assert np.isclose(m.B_n(1), 1, atol=1e-10)

    def test_B_n_monotone(self, uniform_p_values):
        """B_n should be monotone increasing on [0, 1]."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in uniform_p_values[:30]:
            m.update(p)
        xs = np.linspace(0, 1, 50)
        Bs = [m.B_n(x) for x in xs]
        for i in range(len(Bs) - 1):
            assert Bs[i] <= Bs[i + 1] + 1e-10

    def test_b_n_nonnegative(self, uniform_p_values):
        """b_n should be non-negative on [0,1]."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in uniform_p_values[:30]:
            m.update(p)
        xs = np.linspace(0, 1, 50)
        for x in xs:
            assert m.b_n(x) >= -1e-10

    def test_p_values_stored(self, uniform_p_values):
        m = SimpleJumper(warnings=False)
        for p in uniform_p_values[:10]:
            m.update(p)
        assert len(m.p_values) == 10
        assert np.allclose(m.p_values, uniform_p_values[:10])


# ─── CompositeJumper ──────────────────────────────────────────────────────────


class TestCompositeJumper:
    def test_grows_under_alternative(self, skewed_p_values):
        m = CompositeJumper(warnings=False)
        for p in skewed_p_values:
            m.update(p)
        assert m.logM > 0

    def test_M_is_mean_of_sub_jumpers(self, uniform_p_values):
        """M should equal the arithmetic mean of sub-jumper Ms."""
        m = CompositeJumper(warnings=False)
        for p in uniform_p_values[:30]:
            m.update(p)
        sub_Ms = [j.M for j in m.Jumpers.values()]
        expected_M = np.mean(sub_Ms)
        assert np.isclose(m.M, expected_M, rtol=1e-8)

    def test_default_jump_rates(self):
        m = CompositeJumper(warnings=False)
        assert m.J == [1e-4, 1e-3, 1e-2, 1e-1, 1]


# ─── PluginMartingale ─────────────────────────────────────────────────────────


class TestPluginMartingale:
    @pytest.mark.parametrize("strategy_cls", [GaussianKDE, BetaMoments, BetaMLE])
    def test_grows_under_alternative(self, strategy_cls, skewed_p_values):
        strategy = strategy_cls()
        m = PluginMartingale(betting_strategy=strategy, min_sample_size=10, warnings=False)
        for p in skewed_p_values[:200]:
            m.update(p)
        assert m.logM > 0, f"{strategy_cls.__name__}: expected growth, got logM={m.logM}"

    def test_fixed_strategy(self):
        """FixedStrategy with pdf=2*(1-x) should grow on small p-values."""
        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        m = PluginMartingale(betting_strategy=fs, min_sample_size=0, warnings=False)
        for _ in range(20):
            m.update(0.1)
        assert m.logM > 0

    def test_default_strategy_is_gaussian_kde(self):
        m = PluginMartingale(warnings=False)
        assert isinstance(m.strategy, GaussianKDE)

    def test_p_values_stored(self, uniform_p_values):
        m = PluginMartingale(warnings=False)
        for p in uniform_p_values[:10]:
            m.update(p)
        assert len(m.p_values) == 10


# ─── SimpleMixtureMartingale ─────────────────────────────────────────────────


class TestSimpleMixtureMartingale:
    def test_grows_under_alternative(self, skewed_p_values):
        m = SimpleMixtureMartingale(warnings=False)
        for p in skewed_p_values:
            m.update(p)
        assert m.logM > 0

    def test_bounded_under_null(self):
        rng = np.random.default_rng(77)
        final_logMs = []
        for _ in range(20):
            m = SimpleMixtureMartingale(warnings=False)
            for p in rng.uniform(size=100):
                m.update(p)
            final_logMs.append(m.logM)
        # Should not explode under null
        assert np.mean(final_logMs) < 5.0


# ─── Betting Strategies ───────────────────────────────────────────────────────


class TestBettingStrategies:
    @pytest.mark.parametrize(
        "StrategyClass,kwargs",
        [
            (BetaMoments, {}),
            (BetaMLE, {}),
            (GaussianKDE, {"bandwidth": "silverman"}),
        ],
    )
    def test_bet_and_integrate_callable(self, StrategyClass, kwargs):
        strategy = StrategyClass(**kwargs)
        pvals = [0.1, 0.2, 0.3, 0.4, 0.5]
        for p in pvals:
            strategy.update(p)
        assert callable(strategy.bet)
        assert callable(strategy.integrate)

    @pytest.mark.parametrize(
        "StrategyClass,kwargs",
        [
            (BetaMoments, {}),
            (BetaMLE, {}),
            (GaussianKDE, {"bandwidth": "silverman"}),
        ],
    )
    def test_b_n_nonnegative(self, StrategyClass, kwargs):
        strategy = StrategyClass(**kwargs)
        pvals = [0.1, 0.2, 0.3, 0.15, 0.25, 0.05]
        for p in pvals:
            strategy.update(p)
        xs = np.linspace(0.01, 0.99, 20)
        for x in xs:
            assert strategy.bet(x) >= -1e-10, f"bet({x}) = {strategy.bet(x)} is negative"

    @pytest.mark.parametrize(
        "StrategyClass,kwargs",
        [
            (BetaMoments, {}),
            (BetaMLE, {}),
            (GaussianKDE, {"bandwidth": "silverman"}),
        ],
    )
    def test_B_n_boundaries(self, StrategyClass, kwargs):
        strategy = StrategyClass(**kwargs)
        pvals = [0.1, 0.2, 0.3, 0.15, 0.25, 0.05]
        for p in pvals:
            strategy.update(p)
        assert np.isclose(strategy.integrate(0), 0, atol=0.05)
        assert np.isclose(strategy.integrate(1), 1, atol=0.05)


class TestFixedStrategy:
    def test_returns_same_function_regardless_of_input(self):
        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        xs = np.linspace(0, 1, 10)
        vals_before = [fs.bet(x) for x in xs]
        fs.update(0.1)
        fs.update(0.2)
        vals_after = [fs.bet(x) for x in xs]
        for v1, v2 in zip(vals_before, vals_after):
            assert np.isclose(v1, v2)

    def test_with_scipy_distribution(self):
        fs = FixedStrategy(distribution=uniform())
        assert np.isclose(fs.bet(0.5), 1.0)
        assert np.isclose(fs.integrate(0.5), 0.5)


class TestExpertAggregationStrategy:
    def test_weights_sum_to_one(self):
        experts = [
            FixedStrategy(distribution=uniform()),
            FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False),
        ]
        agg = ExpertAggregationStrategy(experts=experts)
        w = agg.get_current_weights()
        assert np.isclose(np.sum(w), 1.0)

    def test_better_expert_gains_weight(self):
        """An expert that matches the data should gain weight."""
        expert_good = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        expert_bad = FixedStrategy(distribution=uniform())
        agg = ExpertAggregationStrategy(experts=[expert_good, expert_bad])

        rng = np.random.default_rng(0)
        pvals = rng.beta(0.5, 2, size=50).tolist()
        for p in pvals:
            agg.update(p)

        w = agg.get_current_weights()
        assert w[0] > w[1], f"Expected expert_good to have higher weight: {w}"


class TestParticleFilterStrategy:
    def test_bet_and_integrate_callable(self):
        pf = ParticleFilterStrategy(num_particles=50, seed=42)
        pvals = [0.1, 0.2, 0.1, 0.05, 0.1]
        for p in pvals:
            pf.update(p)
        assert callable(pf.bet)
        assert callable(pf.integrate)

    def test_b_n_nonnegative(self):
        pf = ParticleFilterStrategy(num_particles=50, seed=42)
        pvals = [0.1, 0.2, 0.1, 0.05, 0.1]
        for p in pvals:
            pf.update(p)
        xs = np.linspace(0.01, 0.99, 20)
        for x in xs:
            assert pf.bet(x) >= -1e-10
