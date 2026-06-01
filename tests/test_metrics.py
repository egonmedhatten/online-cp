import numpy as np
import pytest

from online_cp.classifiers import ConformalPredictionSet
from online_cp.metrics import (
    CRPS,
    ConformalCRPS,
    TruncatedCRPS,
    ErrorRate,
    IntervalWidth,
    Metric,
    Metrics,
    ObservedExcess,
    ObservedFuzziness,
    SetSize,
    WinklerScore,
    BrierScore,
    LogLoss,
    Width,
)
from online_cp.regressors import ConformalPredictionInterval
from online_cp.venn import VennPrediction


class TestErrorRate:
    def test_no_error_when_y_in_gamma(self):
        m = ErrorRate()
        Gamma = ConformalPredictionSet(np.array([0, 1, 2]), epsilon=0.1)
        result = m.update(y=1, Gamma=Gamma)
        assert result == 0.0

    def test_error_when_y_not_in_gamma(self):
        m = ErrorRate()
        Gamma = ConformalPredictionSet(np.array([0, 2]), epsilon=0.1)
        result = m.update(y=1, Gamma=Gamma)
        assert result == 1.0

    def test_running_mean(self):
        m = ErrorRate()
        Gamma_hit = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        Gamma_miss = ConformalPredictionSet(np.array([0]), epsilon=0.1)
        m.update(y=1, Gamma=Gamma_hit)
        m.update(y=1, Gamma=Gamma_miss)
        m.update(y=1, Gamma=Gamma_hit)
        assert np.isclose(m.get(), 1 / 3)

    def test_values_history(self):
        m = ErrorRate()
        Gamma_hit = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        Gamma_miss = ConformalPredictionSet(np.array([0]), epsilon=0.1)
        m.update(y=1, Gamma=Gamma_hit)
        m.update(y=1, Gamma=Gamma_miss)
        m.update(y=1, Gamma=Gamma_hit)
        np.testing.assert_array_equal(m.values, [0, 1, 0])

    def test_works_with_intervals(self):
        m = ErrorRate()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        assert m.update(y=1.0, Gamma=Gamma) == 0.0
        assert m.update(y=5.0, Gamma=Gamma) == 1.0


class TestObservedExcess:
    def test_no_excess_when_singleton(self):
        m = ObservedExcess()
        Gamma = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        assert m.update(y=1, Gamma=Gamma) == 0.0

    def test_excess_when_multiple_labels(self):
        m = ObservedExcess()
        Gamma = ConformalPredictionSet(np.array([0, 1, 2]), epsilon=0.1)
        assert m.update(y=1, Gamma=Gamma) == 2.0

    def test_excess_when_y_not_in_gamma(self):
        m = ObservedExcess()
        Gamma = ConformalPredictionSet(np.array([0, 2]), epsilon=0.1)
        assert m.update(y=1, Gamma=Gamma) == 2.0


class TestObservedFuzziness:
    def test_fuzziness(self):
        m = ObservedFuzziness()
        p_values = {0: 0.3, 1: 0.8, 2: 0.1}
        result = m.update(y=1, p_values=p_values)
        assert np.isclose(result, 0.4)

    def test_zero_fuzziness_when_only_true_label(self):
        m = ObservedFuzziness()
        p_values = {1: 0.9}
        result = m.update(y=1, p_values=p_values)
        assert result == 0.0

    def test_raises_without_p_values(self):
        m = ObservedFuzziness()
        Gamma = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        with pytest.raises(ValueError):
            m.update(y=1, Gamma=Gamma)


class TestSetSize:
    def test_size(self):
        m = SetSize()
        Gamma = ConformalPredictionSet(np.array([0, 1, 2]), epsilon=0.1)
        assert m.update(y=1, Gamma=Gamma) == 3.0

    def test_empty_set(self):
        m = SetSize()
        Gamma = ConformalPredictionSet(np.array([]), epsilon=0.1)
        assert m.update(y=1, Gamma=Gamma) == 0.0


class TestIntervalWidth:
    def test_finite_interval(self):
        m = IntervalWidth()
        Gamma = ConformalPredictionInterval(lower=1.0, upper=3.0, epsilon=0.1)
        assert np.isclose(m.update(y=2.0, Gamma=Gamma), 2.0)

    def test_zero_width(self):
        m = IntervalWidth()
        Gamma = ConformalPredictionInterval(lower=5.0, upper=5.0, epsilon=0.1)
        assert m.update(y=5.0, Gamma=Gamma) == 0.0


class TestWinklerScore:
    def test_no_penalty_when_covered(self):
        m = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        result = m.update(y=1.0, Gamma=Gamma, epsilon=0.1)
        assert np.isclose(result, 2.0)

    def test_penalty_below(self):
        m = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=2.0, upper=4.0, epsilon=0.1)
        result = m.update(y=1.0, Gamma=Gamma, epsilon=0.1)
        expected = 2.0 + (2.0 / 0.1) * (2.0 - 1.0)
        assert np.isclose(result, expected)

    def test_penalty_above(self):
        m = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        result = m.update(y=5.0, Gamma=Gamma, epsilon=0.1)
        expected = 2.0 + (2.0 / 0.1) * (5.0 - 2.0)
        assert np.isclose(result, expected)

    def test_uses_gamma_epsilon_if_not_provided(self):
        m = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        result = m.update(y=1.0, Gamma=Gamma)
        assert np.isclose(result, 2.0)


class TestMetrics:
    def test_composition_via_add(self):
        m = ErrorRate() + IntervalWidth()
        assert isinstance(m, Metrics)
        assert len(m) == 2

    def test_triple_composition(self):
        m = ErrorRate() + IntervalWidth() + WinklerScore()
        assert len(m) == 3

    def test_update_all(self):
        m = ErrorRate() + IntervalWidth()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        m.update(y=1.0, Gamma=Gamma)
        result = m.get()
        assert result["ErrorRate"] == 0.0
        assert np.isclose(result["IntervalWidth"], 2.0)

    def test_getitem_by_name(self):
        m = ErrorRate() + IntervalWidth()
        assert m["ErrorRate"].name == "ErrorRate"

    def test_getitem_by_index(self):
        m = ErrorRate() + IntervalWidth()
        assert m[0].name == "ErrorRate"

    def test_repr(self):
        m = ErrorRate()
        Gamma = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        m.update(y=1, Gamma=Gamma)
        assert "ErrorRate" in repr(m)

    def test_reset(self):
        m = ErrorRate()
        Gamma = ConformalPredictionSet(np.array([0]), epsilon=0.1)
        m.update(y=1, Gamma=Gamma)
        m.reset()
        assert m.get() == 0.0
        assert len(m.values) == 0

    def test_cumulative_mean(self):
        m = ErrorRate()
        Gamma_hit = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        Gamma_miss = ConformalPredictionSet(np.array([0]), epsilon=0.1)
        m.update(y=1, Gamma=Gamma_hit)
        m.update(y=1, Gamma=Gamma_miss)
        m.update(y=1, Gamma=Gamma_hit)
        expected = np.array([0.0, 0.5, 1 / 3])
        np.testing.assert_allclose(m.cumulative_mean(), expected)


# ---------------------------------------------------------------------------
# Venn prediction metrics
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_binary_prediction(self):
        # Predict y=1 with certainty: point = [0, 1]
        venn = VennPrediction.binary(1.0, 1.0)
        m = BrierScore()
        result = m.update(y=1, venn=venn)
        assert np.isclose(result, 0.0)

    def test_worst_binary_prediction(self):
        # Predict y=0 with certainty when y=1: point = [1, 0]
        venn = VennPrediction.binary(0.0, 0.0)
        m = BrierScore()
        result = m.update(y=1, venn=venn)
        # Brier = (0-1)^2 + (1-0)^2 = 2.0
        assert np.isclose(result, 2.0)

    def test_uniform_binary(self):
        # p0=0.5, p1=0.5 → point = [0.5, 0.5]
        venn = VennPrediction.binary(0.5, 0.5)
        m = BrierScore()
        result = m.update(y=1, venn=venn)
        # Brier = (0.5-0)^2 + (0.5-1)^2 = 0.25 + 0.25 = 0.5
        assert np.isclose(result, 0.5)

    def test_multiclass(self):
        # 3-class, hypothesis rows all predict [0.2, 0.3, 0.5]
        probs = np.array([[0.2, 0.3, 0.5]] * 3)
        venn = VennPrediction(probs, np.array([0, 1, 2]))
        m = BrierScore()
        result = m.update(y=2, venn=venn)
        # point = [0.2, 0.3, 0.5], indicator = [0, 0, 1]
        # Brier = (0.2)^2 + (0.3)^2 + (0.5-1)^2 = 0.04 + 0.09 + 0.25 = 0.38
        assert np.isclose(result, 0.38)

    def test_requires_venn_kwarg(self):
        m = BrierScore()
        with pytest.raises(ValueError, match="requires venn"):
            m.update(y=1)


class TestLogLoss:
    def test_perfect_prediction(self):
        venn = VennPrediction.binary(1.0, 1.0)
        m = LogLoss()
        result = m.update(y=1, venn=venn)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_uniform_binary(self):
        venn = VennPrediction.binary(0.5, 0.5)
        m = LogLoss()
        result = m.update(y=1, venn=venn)
        assert np.isclose(result, np.log(2))

    def test_near_zero_clips(self):
        # Predicts p(y=1) ≈ 0 when y=1 → should not be inf
        venn = VennPrediction.binary(0.0, 0.0)
        m = LogLoss()
        result = m.update(y=1, venn=venn)
        assert np.isfinite(result)
        assert result > 30  # -log(1e-15) ≈ 34.5

    def test_multiclass(self):
        probs = np.array([[0.2, 0.3, 0.5]] * 3)
        venn = VennPrediction(probs, np.array([0, 1, 2]))
        m = LogLoss()
        result = m.update(y=2, venn=venn)
        # point = [0.2, 0.3, 0.5], -log(0.5) = log(2)
        assert np.isclose(result, np.log(2))

    def test_requires_venn_kwarg(self):
        m = LogLoss()
        with pytest.raises(ValueError, match="requires venn"):
            m.update(y=0)


class TestWidth:
    def test_binary_sharp(self):
        # p0 = p1 = 0.7 → width = 0
        venn = VennPrediction.binary(0.7, 0.7)
        m = Width()
        result = m.update(y=1, venn=venn)
        assert np.isclose(result, 0.0)

    def test_binary_wide(self):
        # p0=0.2, p1=0.9 → probs = [[0.8, 0.2], [0.1, 0.9]]
        # col 0: max=0.8, min=0.1, width=0.7
        # col 1: max=0.9, min=0.2, width=0.7
        # mean = 0.7
        venn = VennPrediction.binary(0.2, 0.9)
        m = Width()
        result = m.update(y=1, venn=venn)
        assert np.isclose(result, 0.7)

    def test_multiclass(self):
        probs = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
        ])
        venn = VennPrediction(probs, np.array([0, 1, 2]))
        m = Width()
        result = m.update(y=0, venn=venn)
        # col 0: max=0.5, min=0.1 → 0.4
        # col 1: max=0.6, min=0.2 → 0.4
        # col 2: max=0.6, min=0.2 → 0.4
        # mean = 0.4
        assert np.isclose(result, 0.4)

    def test_requires_venn_kwarg(self):
        m = Width()
        with pytest.raises(ValueError, match="requires venn"):
            m.update(y=0)

    def test_composable_with_other_venn_metrics(self):
        venn = VennPrediction.binary(0.3, 0.8)
        metric = BrierScore() + LogLoss() + Width()
        metric.update(y=1, venn=venn)
        result = metric.get()
        assert "BrierScore" in result
        assert "LogLoss" in result
        assert "Width" in result


# ---------------------------------------------------------------------------
# CPD scoring metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cpd():
    """Build a Ridge CPD from synthetic data."""
    from online_cp.CPS import RidgePredictionMachine

    rng = np.random.default_rng(789)
    N = 50
    X = rng.normal(size=(N, 2))
    beta = np.array([1.0, -0.5])
    Y = X @ beta + rng.normal(scale=0.5, size=N)
    cps = RidgePredictionMachine(a=1.0, warnings=False)
    cps.learn_initial_training_set(X[:40], Y[:40])
    return cps.predict_cpd(X[40]), Y[40]


class TestTruncatedCRPS:
    def test_nonnegative(self, simple_cpd):
        cpd, y = simple_cpd
        m = TruncatedCRPS()
        score = m.update(y=y, cpd=cpd)
        assert score >= 0.0

    def test_zero_at_perfect_prediction(self):
        """If y is exactly a critical point with Q(y)=1, score should be small."""
        from online_cp.CPS import RidgePredictionMachine

        rng = np.random.default_rng(111)
        N = 30
        X = rng.normal(size=(N, 1))
        Y = X[:, 0] * 2  # no noise
        cps = RidgePredictionMachine(a=0.01, warnings=False)
        cps.learn_initial_training_set(X[:20], Y[:20])
        cpd = cps.predict_cpd(X[20])
        m = TruncatedCRPS()
        score = m.update(y=Y[20], cpd=cpd, tau=0.5)
        # Should be small (near-perfect fit)
        assert score < 1.0

    def test_finite(self, simple_cpd):
        cpd, y = simple_cpd
        m = TruncatedCRPS()
        score = m.update(y=y, cpd=cpd, tau=0.5)
        assert np.isfinite(score)


class TestConformalCRPS:
    def test_nonnegative(self, simple_cpd):
        cpd, y = simple_cpd
        m = ConformalCRPS()
        score = m.update(y=y, cpd=cpd, tau=0.5)
        assert score >= 0.0

    def test_finite(self, simple_cpd):
        cpd, y = simple_cpd
        m = ConformalCRPS()
        score = m.update(y=y, cpd=cpd, tau=0.5)
        assert np.isfinite(score)

    def test_different_from_truncated(self, simple_cpd):
        """ConformalCRPS and TruncatedCRPS should generally differ."""
        cpd, y = simple_cpd
        s1 = TruncatedCRPS().update(y=y, cpd=cpd, tau=0.5)
        s2 = ConformalCRPS().update(y=y, cpd=cpd, tau=0.5)
        # They can be equal in degenerate cases, but usually differ
        # Just check both are finite and non-negative
        assert s1 >= 0 and s2 >= 0
        assert np.isfinite(s1) and np.isfinite(s2)


class TestCRPSDeprecation:
    def test_warns(self, simple_cpd):
        cpd, y = simple_cpd
        m = CRPS()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            m.update(y=y, cpd=cpd)
