"""Focused unit tests for the private helpers in ``online_cp.evaluate``.

These helpers (``_needs_p_values``, ``_needs_cpd``, ``_iter_data``,
``_should_learn``, ``_wrap_progress``, ``_predict``, ``_predict_and_update``,
``_resolve_delay``, ``_metric_postfix``, ``_drain``) are the glue that the four
``progressive_val`` variants share. They are exercised incidentally through the
public functions elsewhere, but the *edge cases* (streaming input shapes,
metric-kind dispatch, delayed-feedback draining, learn gating) are not
isolated anywhere. This file pins them down directly.
"""

from __future__ import annotations

import heapq

import numpy as np

from online_cp.evaluate import (
    _drain,
    _iter_data,
    _metric_postfix,
    _needs_cpd,
    _needs_p_values,
    _predict,
    _predict_and_update,
    _resolve_delay,
    _should_learn,
    _wrap_progress,
)
from online_cp.metrics import (
    CRPS,
    BrierScore,
    ConformalCRPS,
    ErrorRate,
    IntervalWidth,
    Metrics,
    ObservedFuzziness,
    TruncatedCRPS,
)
from online_cp.prediction_set import DiscretePredictionSet


# --------------------------------------------------------------------------- #
# _needs_p_values / _needs_cpd: metric-kind dispatch
# --------------------------------------------------------------------------- #
class TestNeedsHelpers:
    def test_observed_fuzziness_needs_p_values(self):
        assert _needs_p_values(ObservedFuzziness()) is True

    def test_error_rate_does_not_need_p_values(self):
        assert _needs_p_values(ErrorRate()) is False

    def test_cpd_needs_composite(self):
        # ErrorRate + IntervalWidth + CRPS: only CRPS needs a CPD.
        m = ErrorRate() + IntervalWidth() + CRPS()
        assert _needs_cpd(m) is True
        assert _needs_p_values(m) is False

    def test_no_cpd_when_only_set_metrics(self):
        m = ErrorRate() + IntervalWidth()
        assert _needs_cpd(m) is False

    def test_cpd_helpers_recognise_all_cpd_metric_kinds(self):
        for cls in (CRPS, TruncatedCRPS, ConformalCRPS):
            assert _needs_cpd(cls()) is True

    def test_p_values_helper_recognises_only_observed_fuzziness(self):
        assert _needs_p_values(ErrorRate() + IntervalWidth()) is False
        assert _needs_p_values(ObservedFuzziness()) is True


# --------------------------------------------------------------------------- #
# _iter_data: input normalisation
# --------------------------------------------------------------------------- #
class TestIterData:
    def test_array_input_yields_triples(self):
        X = np.arange(6, dtype=float).reshape(3, 2)
        y = np.array([1.0, 2.0, 3.0])
        triples = list(_iter_data(X, y))
        assert len(triples) == 3
        x0, y0, t0 = triples[0]
        x2, y2, t2 = triples[2]
        assert np.array_equal(x0, X[0]) and y0 == 1.0 and t0 == 0
        assert np.array_equal(x2, X[2]) and y2 == 3.0 and t2 == 2

    def test_streaming_xy_tuples(self):
        stream = [(np.array([0.0]), 1.0), (np.array([1.0]), 2.0)]
        triples = list(_iter_data(stream))  # y is None -> streaming mode
        assert len(triples) == 2
        x0, y0, t0 = triples[0]
        x1, y1, t1 = triples[1]
        assert np.array_equal(x0, np.array([0.0])) and y0 == 1.0 and t0 == 0
        assert np.array_equal(x1, np.array([1.0])) and y1 == 2.0 and t1 == 1

    def test_streaming_xyz_triples_preserves_t(self):
        stream = [(np.array([0.0]), 1.0, 5.0), (np.array([1.0]), 2.0, 7.0)]
        triples = list(_iter_data(stream))
        # The user-supplied t values (5, 7) are preserved, not replaced by i.
        assert triples[0][2] == 5.0
        assert triples[1][2] == 7.0

    def test_empty_array(self):
        assert list(_iter_data(np.empty((0, 2)), np.empty(0))) == []


# --------------------------------------------------------------------------- #
# _should_learn: learn-mode dispatch
# --------------------------------------------------------------------------- #
class TestShouldLearn:
    def test_true_and_false(self):
        assert _should_learn(True, 0, 0.0, 1.0) is True
        assert _should_learn(False, 0, 0.0, 1.0) is False

    def test_callable_is_invoked_with_step_and_obs(self):
        calls = []

        def learn(i, x, y):
            calls.append((i, x, y))
            return i % 2 == 0

        assert _should_learn(learn, 0, 10.0, 20.0) is True
        assert _should_learn(learn, 1, 10.0, 20.0) is False
        assert calls == [(0, 10.0, 20.0), (1, 10.0, 20.0)]


# --------------------------------------------------------------------------- #
# _wrap_progress: tqdm wrapper (disabled path is deterministic)
# --------------------------------------------------------------------------- #
class TestWrapProgress:
    def test_disabled_returns_none_pbar(self):
        it = iter([1, 2, 3])
        wrapped, pbar = _wrap_progress(it, total=3, enabled=False)
        assert pbar is None
        # The iterable is returned untouched and fully consumable.
        assert list(wrapped) == [1, 2, 3]

    def test_enabled_without_total(self):
        # tqdm is installed in the CI env; with no total it should still work.
        wrapped, pbar = _wrap_progress(iter([1, 2, 3]), enabled=True)
        assert pbar is not None
        assert list(wrapped) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# _resolve_delay
# --------------------------------------------------------------------------- #
class TestResolveDelay:
    def test_constant_delay(self):
        assert _resolve_delay(2, 5, 0.0, 1.0) == 2
        assert _resolve_delay(0, 5, 0.0, 1.0) == 0

    def test_callable_delay(self):
        assert _resolve_delay(lambda i, x, y: i, 3, 0.0, 1.0) == 3
        assert _resolve_delay(lambda i, x, y: i + 1, 4, 0.0, 1.0) == 5


# --------------------------------------------------------------------------- #
# _predict: prediction-artifact dispatch (needs_p / needs_cpd / plain)
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Minimal stand-in exposing the predict / predict_cpd surface."""

    def __init__(self, Gamma):
        self._Gamma = Gamma

    def predict(self, x, return_p_values=False, **kw):
        if return_p_values:
            return self._Gamma, np.array([0.5])
        return self._Gamma

    def predict_cpd(self, x):
        class _Cpd:
            def predict_set(self, tau, epsilon=None):
                return f"set@{epsilon}"

        return _Cpd()


class TestPredict:
    def test_plain_predict_no_epsilon(self):
        Gamma, kw = _predict(_FakeModel("G"), 0.0, False, False, {}, None, None)
        assert Gamma == "G"
        assert kw == {}

    def test_plain_predict_with_epsilon_stored(self):
        Gamma, kw = _predict(_FakeModel("G"), 0.0, False, False, {}, 0.1, None)
        assert Gamma == "G"
        assert kw == {"epsilon": 0.1}

    def test_needs_p_values_requests_p_values(self):
        Gamma, kw = _predict(_FakeModel("G"), 0.0, True, False, {}, None, None)
        assert Gamma == "G"
        assert "p_values" in kw
        assert np.array_equal(kw["p_values"], np.array([0.5]))

    def test_needs_cpd_requests_cpd_and_set(self):
        Gamma, kw = _predict(_FakeModel("G"), 0.0, False, True, {}, 0.2, np.random.default_rng(0))
        assert "cpd" in kw
        assert Gamma == "set@0.2"  # predict_set was called with epsilon=0.2


# --------------------------------------------------------------------------- #
# _predict_and_update
# --------------------------------------------------------------------------- #
class TestPredictAndUpdate:
    def test_update_receives_gamma_and_kw(self):
        captured = {}

        class _Metric:
            def update(self, y, Gamma, **kw):
                captured["y"] = y
                captured["Gamma"] = Gamma
                captured["kw"] = kw

        _predict_and_update(
            _FakeModel("G"), 0.0, 7.0, _Metric(), False, False, {}, 0.1, None
        )
        assert captured["y"] == 7.0
        assert captured["Gamma"] == "G"
        assert captured["kw"].get("epsilon") == 0.1


# --------------------------------------------------------------------------- #
# _metric_postfix
# --------------------------------------------------------------------------- #
class TestMetricPostfix:
    def test_single_metric(self):
        m = ErrorRate()
        m.update(y=1, Gamma={1})
        m.update(y=0, Gamma={0, 1})
        out = _metric_postfix(m)
        assert out == {"ErrorRate": f"{m.get():.4f}"}

    def test_composite(self):
        m = ErrorRate() + IntervalWidth()
        gamma = DiscretePredictionSet(np.array([0, 1]), epsilon=0.1)
        m.update(y=1, Gamma=gamma)
        out = _metric_postfix(m)
        assert set(out) == {"ErrorRate", "IntervalWidth"}
        for v in out.values():
            assert isinstance(v, str)


# --------------------------------------------------------------------------- #
# _drain: delayed-feedback heap draining
# --------------------------------------------------------------------------- #
class _RecordingMetric:
    def __init__(self, sink):
        self._sink = sink

    def record(self, y, payload):
        self._sink.append((y, payload))


class TestDrain:
    def test_drain_only_updates_arrived_entries(self):
        # (arrival, orig_step, x, y, payload)
        heap = [(2, 0, 0.0, 1.0, "A"), (5, 1, 0.0, 2.0, "B"), (1, 2, 0.0, 3.0, "C")]
        heapq.heapify(heap)
        applied = []
        metric = _RecordingMetric(applied)
        # Drain up to step 2 -> only arrival 1 and 2 should pop.
        _drain(
            heap, 2, metric, model=None, learn=False,
            apply_update=lambda m, y, p: m.record(y, p),
        )
        assert applied == [(3.0, "C"), (1.0, "A")]
        # One entry (arrival 5) remains.
        assert len(heap) == 1

    def test_drain_all_with_inf_and_learn(self):
        heap = [(1, 0, 0.0, 1.0, "A"), (1, 1, 0.0, 2.0, "B")]
        heapq.heapify(heap)
        learned = []

        class _Model:
            def learn_one(self, x, y):
                learned.append((x, y))

        applied = []
        metric = _RecordingMetric(applied)
        _drain(
            heap, float("inf"), metric, model=_Model(), learn=True,
            apply_update=lambda m, y, p: m.record(y, p),
        )
        assert applied == [(1.0, "A"), (2.0, "B")]
        assert learned == [(0.0, 1.0), (0.0, 2.0)]
        assert heap == []


def test_metrics_composite_is_iterable_for_helpers():
    # The helper dispatch relies on iterating a Metrics composite; pin that.
    m = ErrorRate() + BrierScore()
    assert isinstance(m, Metrics)
    assert [x.name for x in m] == ["ErrorRate", "BrierScore"]
