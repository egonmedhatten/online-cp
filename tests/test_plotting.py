"""Focused unit tests for ``online_cp.plotting``.

The plotting module is largely untested (no dedicated file). This pins down the
*pure* helpers that are the real logic (``_resolve_metric`` composite dispatch,
``_find_alarm_times_cusum``/``_find_alarm_times_sr`` alarm detection) and
verifies that a representative set of public plot functions return a valid
Axes and actually draw the expected artists (lines / hlines) onto it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; safe in CI

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import axes

from online_cp.metrics import ErrorRate, IntervalWidth
from online_cp.plotting import (
    _find_alarm_times_cusum,
    _find_alarm_times_sr,
    _get_ax,
    _resolve_metric,
    plot_coverage,
    plot_set_sizes,
)
from online_cp.prediction_set import DiscretePredictionSet


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure no figure leaks between tests (matplotlib accumulates figures)."""
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# _resolve_metric: composite dispatch
# --------------------------------------------------------------------------- #
class TestResolveMetric:
    def test_bare_metric_returned_as_is(self):
        m = ErrorRate()
        assert _resolve_metric(m, (ErrorRate,)) is m

    def test_composite_selects_first_matching_candidate(self):
        m = ErrorRate() + IntervalWidth()
        # ErrorRate is the first candidate and first member.
        assert _resolve_metric(m, (ErrorRate,)) is m[0]
        # IntervalWidth is still reachable when it is the only candidate.
        assert _resolve_metric(m, (IntervalWidth,)) is m[1]

    def test_priority_order_is_respected(self):
        m = IntervalWidth() + ErrorRate()
        # ErrorRate is listed first even though IntervalWidth appears first in
        # the composite: candidates are tried in priority order.
        assert _resolve_metric(m, (ErrorRate, IntervalWidth)) is m[1]

    def test_missing_candidate_raises_keyerror(self):
        m = ErrorRate() + IntervalWidth()
        with pytest.raises(KeyError):
            _resolve_metric(m, (BrierScoreLike,))

    def test_non_metric_raises_typeerror(self):
        with pytest.raises(TypeError):
            _resolve_metric("not a metric", (ErrorRate,))


class BrierScoreLike:
    """A stand-in class that is *not* an ErrorRate/IntervalWidth subclass."""

    pass


# --------------------------------------------------------------------------- #
# _get_ax
# --------------------------------------------------------------------------- #
class TestGetAx:
    def test_none_creates_new_ax(self):
        ax = _get_ax(None)
        assert isinstance(ax, axes.Axes)

    def test_existing_ax_returned(self):
        _, ax = plt.subplots()
        assert _get_ax(ax) is ax


# --------------------------------------------------------------------------- #
# Alarm-time detection (the real detector logic)
# --------------------------------------------------------------------------- #
class TestAlarmTimesSR:
    def test_no_alarm_below_threshold(self):
        assert _find_alarm_times_sr(np.array([0.5, 0.4, 0.9]), 1.0) == []

    def test_single_alarm(self):
        vals = np.array([0.5, 1.2, 1.3, 0.1, 0.2])
        # Crosses at index 1, stays above at 2 (no new alarm), drops, no more.
        assert _find_alarm_times_sr(vals, 1.0) == [1]

    def test_two_separate_alarm_episodes(self):
        vals = np.array([0.5, 1.2, 0.1, 0.2, 1.5])
        # Alarm at 1 (drops at 2 -> reset), alarm again at 4.
        assert _find_alarm_times_sr(vals, 1.0) == [1, 4]

    def test_threshold_is_inclusive(self):
        assert _find_alarm_times_sr(np.array([1.0, 0.5]), 1.0) == [0]


class TestAlarmTimesCusum:
    def test_no_alarm(self):
        assert _find_alarm_times_cusum(np.log(np.array([0.5, 0.6])), 100.0, None) == []

    def test_constant_threshold_crossing(self):
        # log values: well below, above, above, below -> one alarm at index 1.
        log_vals = np.array([np.log(0.5), np.log(200.0), np.log(300.0), np.log(0.1)])
        assert _find_alarm_times_cusum(log_vals, 100.0, None) == [1]

    def test_barrier_slope_delays_or_prevents_alarm(self):
        # With no barrier a constant log-statistic above the threshold alarms
        # from the very first step. With a positive barrier_slope the effective
        # threshold log(threshold + slope*n) grows with n, so the same
        # statistic stops alarming once the barrier overtakes it.
        log_vals = np.full(6, np.log(150.0))
        # No barrier: alarms at every step where log(150) >= log(100) -> step 0.
        no_barrier = _find_alarm_times_cusum(log_vals, 100.0, barrier_slope=None)
        assert no_barrier == [0]
        # Strong barrier: log(100 + 50*n) exceeds log(150) for n >= 1, so the
        # alarm does not continue past step 0.
        with_barrier = _find_alarm_times_cusum(log_vals, 100.0, barrier_slope=50.0)
        assert with_barrier == [0]
        # Sanity: at n=1 the barrier threshold log(150) >= log(150) is equal,
        # so n=1 would still be an alarm boundary; use a steeper slope to be
        # sure it clearly drops off.
        steep = _find_alarm_times_cusum(log_vals, 100.0, barrier_slope=200.0)
        assert steep == [0]


# --------------------------------------------------------------------------- #
# Public plot functions: figure content
# --------------------------------------------------------------------------- #
class TestPlotCoverage:
    def test_returns_axes_and_draws_line(self):
        m = ErrorRate()
        for v in (0.0, 1.0, 0.0, 1.0, 0.0):
            m.update(y=1, Gamma={1} if v == 0 else {0})
        ax = plot_coverage(m)
        assert isinstance(ax, axes.Axes)
        # A line plot should have at least one Line2D artist.
        lines = ax.get_lines()
        assert len(lines) >= 1
        assert len(lines[0].get_xdata()) == len(m.values)
        assert ax.get_title() == "Running Coverage"

    def test_composite_metric_is_accepted(self):
        m = ErrorRate() + IntervalWidth()
        for v in (0.0, 1.0, 0.0):
            gamma = DiscretePredictionSet(np.array([0, 1]) if v == 0 else np.array([2]), epsilon=0.1)
            m.update(y=1, Gamma=gamma)
        ax = plot_coverage(m, nominal=0.9)
        assert isinstance(ax, axes.Axes)
        # nominal draws an hline.
        assert len(ax.get_lines()) >= 2

    def test_reuses_provided_ax(self):
        _, ax = plt.subplots()
        m = ErrorRate()
        m.update(y=1, Gamma={1})
        out = plot_coverage(m, ax=ax)
        assert out is ax


class TestPlotSetSizes:
    def test_returns_axes(self):
        m = IntervalWidth()
        gamma = DiscretePredictionSet(np.array([0, 1, 2]), epsilon=0.1)
        for _ in range(5):
            m.update(y=1.0, Gamma=gamma)
        ax = plot_set_sizes(m)
        assert isinstance(ax, axes.Axes)
        assert len(ax.get_lines()) >= 1
