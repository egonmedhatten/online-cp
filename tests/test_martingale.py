import numpy as np
import pytest
from copy import deepcopy
from scipy.integrate import quad

from online_cp.martingale import (
    SimpleJumper,
    CompositeJumper,
    PluginMartingale,
    OnionMartingale,
    SimpleMixtureMartingale,
    BetaMoments,
    BetaMLE,
    GaussianKDE,
    BetaKernel,
    FixedStrategy,
    ExpertAggregationStrategy,
    ParticleFilterStrategy,
)
from scipy.stats import uniform, norm


# ─── Base martingale properties ───────────────────────────────────────────────


class TestMartingaleBaseProperties:
    """Properties that hold for any ConformalTestMartingale subclass."""

    @pytest.mark.parametrize("MartingaleClass,kwargs", [
        (SimpleJumper, {"warnings": False}),
        (CompositeJumper, {"warnings": False}),
        (SimpleMixtureMartingale, {"warnings": False}),
    ])
    def test_starts_at_one(self, MartingaleClass, kwargs):
        m = MartingaleClass(**kwargs)
        assert m.logM == 0.0
        assert np.isclose(m.M, 1.0)
        assert m.log_martingale_values == [0.0]

    @pytest.mark.parametrize("MartingaleClass,kwargs", [
        (SimpleJumper, {"warnings": False}),
        (CompositeJumper, {"warnings": False}),
        (SimpleMixtureMartingale, {"warnings": False}),
    ])
    def test_M_equals_exp_logM(self, MartingaleClass, kwargs, uniform_p_values):
        m = MartingaleClass(**kwargs)
        for p in uniform_p_values[:50]:
            m.update_martingale_value(p)
        assert np.isclose(m.M, np.exp(m.logM))

    @pytest.mark.parametrize("MartingaleClass,kwargs", [
        (SimpleJumper, {"warnings": False}),
        (CompositeJumper, {"warnings": False}),
        (SimpleMixtureMartingale, {"warnings": False}),
    ])
    def test_max_geq_current(self, MartingaleClass, kwargs, uniform_p_values):
        m = MartingaleClass(**kwargs)
        for p in uniform_p_values[:50]:
            m.update_martingale_value(p)
            assert m.log_max >= m.logM


# ─── SimpleJumper ─────────────────────────────────────────────────────────────


class TestSimpleJumper:
    def test_grows_under_alternative(self, skewed_p_values):
        """Under H1 (skewed towards 0), martingale should grow."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in skewed_p_values:
            m.update_martingale_value(p)
        assert m.logM > 0, f"Expected growth under H1, got logM={m.logM}"

    def test_bounded_under_null(self):
        """Under H0 (uniform), average logM should stay near 0."""
        rng = np.random.default_rng(123)
        final_logMs = []
        for _ in range(20):
            m = SimpleJumper(J=0.1, warnings=False)
            pvals = rng.uniform(size=200)
            for p in pvals:
                m.update_martingale_value(p)
            final_logMs.append(m.logM)
        # Mean of logM should not be significantly positive
        assert np.mean(final_logMs) < 5.0

    def test_B_n_boundary_values(self, uniform_p_values):
        """B_n(0) should be 0, B_n(1) should be 1."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in uniform_p_values[:20]:
            m.update_martingale_value(p)
        assert np.isclose(m.B_n(0), 0, atol=1e-10)
        assert np.isclose(m.B_n(1), 1, atol=1e-10)

    def test_B_n_monotone(self, uniform_p_values):
        """B_n should be monotone increasing on [0, 1]."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in uniform_p_values[:30]:
            m.update_martingale_value(p)
        xs = np.linspace(0, 1, 50)
        Bs = [m.B_n(x) for x in xs]
        for i in range(len(Bs) - 1):
            assert Bs[i] <= Bs[i + 1] + 1e-10

    def test_b_n_nonnegative(self, uniform_p_values):
        """b_n should be non-negative on [0,1]."""
        m = SimpleJumper(J=0.1, warnings=False)
        for p in uniform_p_values[:30]:
            m.update_martingale_value(p)
        xs = np.linspace(0, 1, 50)
        for x in xs:
            assert m.b_n(x) >= -1e-10

    def test_p_values_stored(self, uniform_p_values):
        m = SimpleJumper(warnings=False)
        for p in uniform_p_values[:10]:
            m.update_martingale_value(p)
        assert len(m.p_values) == 10
        assert np.allclose(m.p_values, uniform_p_values[:10])


# ─── CompositeJumper ──────────────────────────────────────────────────────────


class TestCompositeJumper:
    def test_grows_under_alternative(self, skewed_p_values):
        m = CompositeJumper(warnings=False)
        for p in skewed_p_values:
            m.update_martingale_value(p)
        assert m.logM > 0

    def test_M_is_mean_of_sub_jumpers(self, uniform_p_values):
        """M should equal the arithmetic mean of sub-jumper Ms."""
        m = CompositeJumper(warnings=False)
        for p in uniform_p_values[:30]:
            m.update_martingale_value(p)
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
        strategy = strategy_cls(min_sample_size=10)
        m = PluginMartingale(betting_strategy=strategy, warnings=False)
        for p in skewed_p_values[:200]:
            m.update_martingale_value(p)
        assert m.logM > 0, f"{strategy_cls.__name__}: expected growth, got logM={m.logM}"

    def test_fixed_strategy(self):
        """FixedStrategy with pdf=2*(1-x) should grow on small p-values."""
        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        m = PluginMartingale(betting_strategy=fs, warnings=False)
        for _ in range(20):
            m.update_martingale_value(0.1)
        assert m.logM > 0

    def test_default_strategy_is_gaussian_kde(self):
        m = PluginMartingale(warnings=False)
        assert isinstance(m.betting_strategy, GaussianKDE)

    def test_p_values_stored(self, uniform_p_values):
        m = PluginMartingale(warnings=False)
        for p in uniform_p_values[:10]:
            m.update_martingale_value(p)
        assert len(m.p_values) == 10


# ─── OnionMartingale ──────────────────────────────────────────────────────────


class TestOnionMartingale:
    def test_logM_is_sum_of_layers(self, uniform_p_values):
        layers = [SimpleJumper(J=j, warnings=False) for j in [0.01, 0.1, 1.0]]
        om = OnionMartingale(layers, warnings=False)
        for p in uniform_p_values[:50]:
            om.update_martingale_value(p)
        expected_logM = sum(layer.logM for layer in om.layers)
        assert np.isclose(om.logM, expected_logM, atol=1e-12)

    def test_protected_p_values_in_unit_interval(self, uniform_p_values):
        layers = [SimpleJumper(J=j, warnings=False) for j in [0.01, 0.1]]
        om = OnionMartingale(layers, warnings=False)
        for p in uniform_p_values[:100]:
            om.update_martingale_value(p)
        for pv in om.p_values:
            assert 0 <= pv <= 1 + 1e-10, f"Protected p-value {pv} outside [0,1]"

    def test_B_n_boundary_values(self, uniform_p_values):
        layers = [SimpleJumper(J=0.1, warnings=False) for _ in range(3)]
        om = OnionMartingale(layers, warnings=False)
        for p in uniform_p_values[:30]:
            om.update_martingale_value(p)
        assert np.isclose(om.B_n(0), 0, atol=1e-10)
        assert np.isclose(om.B_n(1), 1, atol=1e-10)

    def test_grows_under_alternative(self, skewed_p_values):
        layers = [SimpleJumper(J=j, warnings=False) for j in [0.01, 0.1, 1.0]]
        om = OnionMartingale(layers, warnings=False)
        for p in skewed_p_values:
            om.update_martingale_value(p)
        assert om.logM > 0

    def test_protected_p_values_approx_uniform_under_H0(self):
        """Under H0, protected p-values should be approximately uniform."""
        rng = np.random.default_rng(99)
        pvals = rng.uniform(size=500)
        layers = [SimpleJumper(J=0.1, warnings=False) for _ in range(3)]
        om = OnionMartingale(layers, warnings=False)
        for p in pvals:
            om.update_martingale_value(p)
        # KS-like check: mean should be ~0.5
        assert 0.35 < np.mean(om.p_values) < 0.65

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError):
            OnionMartingale([], warnings=False)

    def test_each_layer_stores_p_values(self, uniform_p_values):
        layers = [SimpleJumper(J=0.01, warnings=False) for _ in range(3)]
        om = OnionMartingale(layers, warnings=False)
        for p in uniform_p_values[:20]:
            om.update_martingale_value(p)
        for layer in om.layers:
            assert len(layer.p_values) == 20


# ─── SimpleMixtureMartingale ─────────────────────────────────────────────────


class TestSimpleMixtureMartingale:
    def test_grows_under_alternative(self, skewed_p_values):
        m = SimpleMixtureMartingale(warnings=False)
        for p in skewed_p_values:
            m.update_martingale_value(p)
        assert m.logM > 0

    def test_bounded_under_null(self):
        rng = np.random.default_rng(77)
        final_logMs = []
        for _ in range(20):
            m = SimpleMixtureMartingale(warnings=False)
            for p in rng.uniform(size=100):
                m.update_martingale_value(p)
            final_logMs.append(m.logM)
        # Should not explode under null
        assert np.mean(final_logMs) < 5.0


# ─── Betting Strategies ───────────────────────────────────────────────────────


class TestBettingStrategies:
    @pytest.mark.parametrize("StrategyClass,kwargs", [
        (BetaMoments, {"min_sample_size": 10}),
        (BetaMLE, {"min_sample_size": 10}),
        (GaussianKDE, {"min_sample_size": 10, "bandwidth": "silverman"}),
    ])
    def test_returns_callables(self, StrategyClass, kwargs):
        strategy = StrategyClass(**kwargs)
        pvals = [0.1, 0.2, 0.3, 0.4, 0.5]
        for p in pvals:
            b_n, B_n = strategy.update_betting_function([p])
        assert callable(b_n)
        assert callable(B_n)

    @pytest.mark.parametrize("StrategyClass,kwargs", [
        (BetaMoments, {"min_sample_size": 5}),
        (BetaMLE, {"min_sample_size": 5}),
        (GaussianKDE, {"min_sample_size": 5, "bandwidth": "silverman"}),
    ])
    def test_b_n_nonnegative(self, StrategyClass, kwargs):
        strategy = StrategyClass(**kwargs)
        pvals = [0.1, 0.2, 0.3, 0.15, 0.25, 0.05]
        b_n, B_n = strategy.update_betting_function(pvals)
        xs = np.linspace(0.01, 0.99, 20)
        for x in xs:
            assert b_n(x) >= -1e-10, f"b_n({x}) = {b_n(x)} is negative"

    @pytest.mark.parametrize("StrategyClass,kwargs", [
        (BetaMoments, {"min_sample_size": 5}),
        (BetaMLE, {"min_sample_size": 5}),
        (GaussianKDE, {"min_sample_size": 5, "bandwidth": "silverman"}),
    ])
    def test_B_n_boundaries(self, StrategyClass, kwargs):
        strategy = StrategyClass(**kwargs)
        pvals = [0.1, 0.2, 0.3, 0.15, 0.25, 0.05]
        b_n, B_n = strategy.update_betting_function(pvals)
        assert np.isclose(B_n(0), 0, atol=0.05)
        assert np.isclose(B_n(1), 1, atol=0.05)


class TestFixedStrategy:
    def test_returns_same_function_regardless_of_input(self):
        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        b1, _ = fs.update_betting_function([])
        b2, _ = fs.update_betting_function([0.1, 0.2, 0.3])
        xs = np.linspace(0, 1, 10)
        for x in xs:
            assert np.isclose(b1(x), b2(x))

    def test_with_scipy_distribution(self):
        fs = FixedStrategy(distribution=uniform())
        b, B = fs.update_betting_function([])
        assert np.isclose(b(0.5), 1.0)
        assert np.isclose(B(0.5), 0.5)


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
        # Expert 1 bets on small p-values (good when p ~ Beta(0.5, 2))
        expert_good = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        # Expert 2 is uniform (neutral)
        expert_bad = FixedStrategy(distribution=uniform())
        agg = ExpertAggregationStrategy(experts=[expert_good, expert_bad])

        rng = np.random.default_rng(0)
        pvals = rng.beta(0.5, 2, size=50).tolist()
        for i in range(len(pvals)):
            agg.update_betting_function(pvals[:i + 1])

        w = agg.get_current_weights()
        assert w[0] > w[1], f"Expected expert_good to have higher weight: {w}"


class TestParticleFilterStrategy:
    def test_returns_callables(self):
        pf = ParticleFilterStrategy(num_particles=50, seed=42, min_sample_size=5)
        pvals = [0.1, 0.2, 0.1, 0.05, 0.1]
        b_n, B_n = pf.update_betting_function(pvals)
        assert callable(b_n)
        assert callable(B_n)

    def test_b_n_nonnegative(self):
        pf = ParticleFilterStrategy(num_particles=50, seed=42, min_sample_size=5)
        pvals = [0.1, 0.2, 0.1, 0.05, 0.1]
        b_n, B_n = pf.update_betting_function(pvals)
        xs = np.linspace(0.01, 0.99, 20)
        for x in xs:
            assert b_n(x) >= -1e-10
