import numpy as np
import pytest

from online_cp.classifiers import ConformalPredictionSet
from online_cp.metrics import (
    CRPS,
    ErrorRate,
    IntervalWidth,
    Metric,
    Metrics,
    ObservedExcess,
    ObservedFuzziness,
    SetSize,
    WinklerScore,
)
from online_cp.regressors import ConformalPredictionInterval


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
