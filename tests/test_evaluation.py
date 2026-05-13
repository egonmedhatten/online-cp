import numpy as np
import pytest

from online_cp.classifiers import ConformalPredictionSet
from online_cp.evaluation import OE, OF, Err, Evaluation, Width, WinklerScore
from online_cp.regressors import ConformalPredictionInterval


class TestErr:
    def test_no_error_when_y_in_gamma(self):
        err = Err()
        Gamma = ConformalPredictionSet(np.array([0, 1, 2]), epsilon=0.1)
        result = err._update(1, Gamma)
        assert result == 0

    def test_error_when_y_not_in_gamma(self):
        err = Err()
        Gamma = ConformalPredictionSet(np.array([0, 2]), epsilon=0.1)
        result = err._update(1, Gamma)
        assert result == 1

    def test_cumulative(self):
        err = Err()
        Gamma_hit = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        Gamma_miss = ConformalPredictionSet(np.array([0]), epsilon=0.1)
        err._update(1, Gamma_hit)
        err._update(1, Gamma_miss)
        err._update(1, Gamma_hit)
        assert err.value == 1
        assert err.n == 3


class TestOE:
    def test_no_excess_when_singleton(self):
        oe = OE()
        Gamma = ConformalPredictionSet(np.array([1]), epsilon=0.1)
        result = oe._update(1, Gamma)
        assert result == 0

    def test_excess_when_multiple_labels(self):
        oe = OE()
        Gamma = ConformalPredictionSet(np.array([0, 1, 2]), epsilon=0.1)
        result = oe._update(1, Gamma)
        assert result == 2  # 3 labels - 1 correct

    def test_excess_when_y_not_in_gamma(self):
        oe = OE()
        Gamma = ConformalPredictionSet(np.array([0, 2]), epsilon=0.1)
        result = oe._update(1, Gamma)
        assert result == 2  # all labels are "excess"


class TestOF:
    def test_fuzziness(self):
        of = OF()
        p_values = {0: 0.3, 1: 0.8, 2: 0.1}
        result = of._update(p_values, 1)
        # OF = sum of p-values for labels != y = 0.3 + 0.1 = 0.4
        assert np.isclose(result, 0.4)

    def test_zero_fuzziness_when_only_true_label(self):
        of = OF()
        p_values = {1: 0.9}
        result = of._update(p_values, 1)
        assert result == 0


class TestWidth:
    def test_finite_interval(self):
        w = Width()
        Gamma = ConformalPredictionInterval(lower=1.0, upper=3.0, epsilon=0.1)
        result = w._update(Gamma)
        assert np.isclose(result, 2.0)

    def test_nonnegative(self):
        w = Width()
        Gamma = ConformalPredictionInterval(lower=5.0, upper=5.0, epsilon=0.1)
        result = w._update(Gamma)
        assert result >= 0


class TestWinklerScore:
    def test_no_penalty_when_covered(self):
        ws = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        result = ws._update(Gamma, 1.0, 0.1)
        # Score = width = 2.0, no penalty since y in [0, 2]
        assert np.isclose(result, 2.0)

    def test_penalty_below(self):
        ws = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=2.0, upper=4.0, epsilon=0.1)
        result = ws._update(Gamma, 1.0, 0.1)
        # width = 2, penalty = (2/0.1)*(2-1) = 20
        expected = 2.0 + 20.0
        assert np.isclose(result, expected)

    def test_penalty_above(self):
        ws = WinklerScore()
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        result = ws._update(Gamma, 5.0, 0.1)
        # width = 2, penalty = (2/0.1)*(5-2) = 60
        expected = 2.0 + 60.0
        assert np.isclose(result, expected)


class TestEvaluation:
    def test_initialization_with_metrics(self):
        ev = Evaluation(metrics=["err", "oe", "of", "width", "winkler"])
        assert "err" in ev.metrics
        assert "oe" in ev.metrics
        assert "of" in ev.metrics
        assert "width" in ev.metrics
        assert "winkler" in ev.metrics

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            Evaluation(metrics=["nonexistent"])

    def test_update_and_summarize(self):
        ev = Evaluation(metrics=["err", "width"])
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        ev.update(y=1.0, Gamma=Gamma)
        ev.update(y=5.0, Gamma=Gamma)

        summary = ev.summarize()
        assert "err" in summary
        assert "width" in summary
        assert summary["err"]["mean"] == 0.5  # 1 hit, 1 miss
        assert summary["width"]["mean"] == 2.0

    def test_cumulative(self):
        ev = Evaluation(metrics=["err"])
        Gamma_hit = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        Gamma_miss = ConformalPredictionInterval(lower=3.0, upper=4.0, epsilon=0.1)
        ev.update(y=1.0, Gamma=Gamma_hit)
        ev.update(y=1.0, Gamma=Gamma_miss)
        ev.update(y=1.0, Gamma=Gamma_hit)

        cum = ev.cumulative("err")
        assert list(cum) == [0, 1, 1]

    def test_n_increments(self):
        ev = Evaluation(metrics=["err"])
        Gamma = ConformalPredictionInterval(lower=0.0, upper=2.0, epsilon=0.1)
        ev.update(y=1.0, Gamma=Gamma)
        ev.update(y=1.0, Gamma=Gamma)
        assert ev.n == 2
