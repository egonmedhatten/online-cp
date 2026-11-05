"""Tests for streaming evaluation and plotting utilities."""

import numpy as np
import pytest

from online_cp import ConformalRidgeRegressor
from online_cp.evaluate import iter_progressive_val, iter_progressive_val_venn, progressive_val, progressive_val_venn
from online_cp.metrics import BrierScore, ErrorRate, IntervalWidth


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test.

    Plotting tests create figures via ``plt.subplots()`` (through the
    ``plot_*`` helpers) and do not need to keep them open. Leaving >20
    figures alive triggers matplotlib's ``figure.max_open_warning``
    RuntimeWarning. Closing them after each test keeps the run quiet.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def ridge_model(linear_dataset):
    """A ridge regressor with initial training set."""
    X, y = linear_dataset
    model = ConformalRidgeRegressor(a=1.0)
    model.learn_initial_training_set(X[:50], y[:50])
    return model, X[50:], y[50:]


class TestStreamingEval:
    """Test streaming iterable input to progressive_val."""

    def test_array_backward_compat(self, ridge_model):
        """Classic (X, y) array interface still works."""
        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        result = progressive_val(model, X_test, y_test, epsilon=0.1, metric=metric)
        assert result is metric
        assert len(metric.values) == len(y_test)

    def test_streaming_tuples(self, linear_dataset):
        """Accept iterable of (x, y) tuples."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])

        stream = list(zip(X[50:100], y[50:100]))
        metric = ErrorRate()
        progressive_val(model, stream, epsilon=0.1, metric=metric)
        assert len(metric.values) == 50

    def test_streaming_with_timestamps(self, linear_dataset):
        """Accept iterable of (x, y, t) triples."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])

        stream = [(X[i], y[i], f"t{i}") for i in range(50, 80)]
        metric = ErrorRate()
        progressive_val(model, stream, epsilon=0.1, metric=metric)
        assert len(metric.values) == 30  # 30 tuples in stream

    def test_learn_false(self, linear_dataset):
        """learn=False means model doesn't update."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])

        n_before = model.X.shape[0]
        progressive_val(model, X[50:60], y[50:60], epsilon=0.1, learn=False)
        assert model.X.shape[0] == n_before

    def test_learn_callable(self, linear_dataset):
        """learn=callable controls per-step learning."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])

        n_before = model.X.shape[0]
        # Learn only on even steps
        progressive_val(model, X[50:60], y[50:60], epsilon=0.1, learn=lambda i, x, y: i % 2 == 0)
        # Should have learned 5 times (i=0,2,4,6,8)
        assert model.X.shape[0] == n_before + 5

    def test_iter_yields_timestamps(self, linear_dataset):
        """iter_progressive_val yields 't' field."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])

        stream = [(X[i], y[i], 1000 + i) for i in range(50, 60)]
        snapshots = list(iter_progressive_val(model, stream, epsilon=0.1, step=5))
        assert len(snapshots) == 2
        assert "t" in snapshots[0]
        assert snapshots[0]["t"] == 1000 + 54  # 5th element (i=4) has t=1054
        assert snapshots[0]["step"] == 5

    def test_iter_step(self, ridge_model):
        """iter_progressive_val yields every `step` samples."""
        model, X_test, y_test = ridge_model
        snapshots = list(iter_progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, step=10))
        assert len(snapshots) == 2
        assert snapshots[0]["step"] == 10
        assert snapshots[1]["step"] == 20

    def test_delay_int_resolves_all_labels(self, linear_dataset):
        """Fixed delay: all labels are resolved after progressive_val returns."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])
        metric = ErrorRate()
        progressive_val(model, X[50:100], y[50:100], epsilon=0.1, metric=metric, delay=3)
        assert len(metric.values) == 50

    def test_delay_callable(self, linear_dataset):
        """Callable delay returning 0 behaves identically to delay=0."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])
        metric = ErrorRate()
        progressive_val(
            model,
            X[50:80],
            y[50:80],
            epsilon=0.1,
            metric=metric,
            delay=lambda i, x, y: 0,
        )
        assert len(metric.values) == 30

    def test_lazy_teacher_none_labels(self, linear_dataset):
        """y=None in stream skips metric update and learning."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])
        n_before = model.X.shape[0]
        # Every other label is missing.
        stream = [(X[i], y[i] if i % 2 == 0 else None) for i in range(50, 70)]
        metric = ErrorRate()
        progressive_val(model, stream, epsilon=0.1, metric=metric)
        assert len(metric.values) == 10  # only even-index labels
        assert model.X.shape[0] == n_before + 10

    def test_iter_delay_teardown(self, linear_dataset):
        """iter_progressive_val emits a teardown snapshot when delay > 0."""
        X, y = linear_dataset

        # delay=0: 20 samples, step=10 -> exactly 2 snapshots, no teardown.
        model1 = ConformalRidgeRegressor(a=1.0)
        model1.learn_initial_training_set(X[:50], y[:50])
        snaps_no_delay = list(iter_progressive_val(model1, X[50:70], y[50:70], epsilon=0.1, step=10, delay=0))
        assert len(snaps_no_delay) == 2

        # delay=5: items 15-19 have arrival steps 20-24, resolved in teardown.
        model2 = ConformalRidgeRegressor(a=1.0)
        model2.learn_initial_training_set(X[:50], y[:50])
        snaps_delayed = list(iter_progressive_val(model2, X[50:70], y[50:70], epsilon=0.1, step=10, delay=5))
        # step=10 snapshot, step=20 snapshot, teardown snapshot.
        assert len(snaps_delayed) == 3
        assert snaps_delayed[-1]["step"] == 20

    def test_progressive_val_with_progress_bar(self, ridge_model):
        """progress=True displays running metrics via tqdm postfix (smoke test)."""
        try:
            import tqdm  # noqa: F401
        except ImportError:
            pytest.skip("tqdm not installed; skipping progress bar test")

        model, X_test, y_test = ridge_model
        metric = ErrorRate() + IntervalWidth()
        result = progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric, progress=True)
        assert result is metric
        # Verify metrics were updated (check via get() dict).
        metric_dict = metric.get()
        assert "ErrorRate" in metric_dict
        assert "IntervalWidth" in metric_dict


class TestTrace:
    """Tests for the ``trace=True`` option on the progressive_val family.

    ``trace=True`` records the raw per-step (y, Gamma) data on the returned
    metric, in label-arrival order, so it can be inspected or plotted
    directly (e.g. ``plot_intervals(metric)``) without re-running the stream.
    """

    def test_trace_records_y_and_gamma(self, ridge_model):
        """trace=True sets trace_y / trace_gamma aligned with metric.values."""
        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        result = progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric, trace=True)
        assert result is metric
        assert len(metric.trace_y) == len(metric.values) == 30
        assert len(metric.trace_gamma) == 30
        # trace_y matches the true labels (arrival order == stream order here).
        assert np.allclose(np.array(metric.trace_y), y_test[:30])

    def test_trace_gamma_is_the_prediction_set(self, ridge_model):
        """Each traced Gamma is the conformal prediction set for that step."""
        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, metric=metric, trace=True)
        for i, Gamma in enumerate(metric.trace_gamma):
            # A fresh prediction from the *same* (already-trained) model state
            # is order-dependent, so instead verify the invariants that must
            # hold: continuous, single interval, and y either in or out.
            assert Gamma.is_continuous
            lo, hi = Gamma.intervals[0]
            assert lo <= hi
            covered = lo <= y_test[i] <= hi
            assert covered == (y_test[i] in Gamma)

    def test_trace_default_off(self, ridge_model):
        """trace=False (default) creates no trace_* attributes (no bloat)."""
        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, metric=metric)
        assert not hasattr(metric, "trace_y")
        assert not hasattr(metric, "trace_gamma")
        # And the metric still works as before.
        assert len(metric.values) == 20

    def test_trace_composite_metric(self, ridge_model):
        """trace=True works with a Metrics composite and both members update."""
        model, X_test, y_test = ridge_model
        metric = ErrorRate() + IntervalWidth()
        progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric, trace=True)
        # Both scalar members recorded 30 steps.
        assert len(metric["ErrorRate"].values) == 30
        assert len(metric["IntervalWidth"].values) == 30
        # The shared trace is also 30 long.
        assert len(metric.trace_y) == 30
        assert len(metric.trace_gamma) == 30

    def test_trace_lazy_teacher_skips(self, linear_dataset):
        """Lazy-teacher steps (y=None) are excluded from the trace."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])
        stream = [(X[i], y[i] if i % 2 == 0 else None) for i in range(50, 70)]
        metric = ErrorRate()
        progressive_val(model, stream, epsilon=0.1, metric=metric, trace=True)
        # Only the 10 even-index labels are evaluated.
        assert len(metric.trace_y) == len(metric.values) == 10
        assert len(metric.trace_gamma) == 10

    def test_trace_with_delay(self, linear_dataset):
        """Under delay>0 the trace is still fully resolved and length-correct."""
        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])
        metric = ErrorRate()
        progressive_val(model, X[50:80], y[50:80], epsilon=0.1, metric=metric, trace=True, delay=3)
        assert len(metric.trace_y) == len(metric.values) == 30

    def test_iter_progressive_val_trace(self, ridge_model):
        """iter_progressive_val(trace=True) carries trace data in each snapshot."""
        model, X_test, y_test = ridge_model
        snapshots = list(iter_progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, step=5, trace=True))
        assert len(snapshots) == 4
        for snap in snapshots:
            assert "trace_y" in snap
            assert "trace_gamma" in snap
        # Each snapshot covers a distinct chunk; total = 20.
        total = sum(len(s["trace_y"]) for s in snapshots)
        assert total == 20

    def test_iter_trace_default_off(self, ridge_model):
        """iter_progressive_val default (trace=False) snapshots have no trace keys."""
        model, X_test, y_test = ridge_model
        snapshots = list(iter_progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, step=5))
        for snap in snapshots:
            assert "trace_y" not in snap
            assert "trace_gamma" not in snap


class TestVennEval:
    """Tests for progressive_val_venn and iter_progressive_val_venn."""

    @pytest.fixture
    def venn_model(self):
        from online_cp import VennAbersPredictor

        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(loc=[2, 0], size=(30, 2)), rng.normal(loc=[-2, 0], size=(30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        idx = rng.permutation(60)
        X, y = X[idx], y[idx]
        model = VennAbersPredictor(scorer="knn", k=3)
        model.learn_initial_training_set(X[:40], y[:40])
        return model, X[40:], y[40:]

    def test_progressive_val_venn_basic(self, venn_model):
        model, X_test, y_test = venn_model
        metric = BrierScore()
        result = progressive_val_venn(model, X_test, y_test, metric=metric)
        assert result is metric
        assert metric._n == len(X_test)
        assert 0 <= metric.get() <= 1

    def test_progressive_val_venn_streaming(self, venn_model):
        model, X_test, y_test = venn_model
        stream = [(X_test[i], y_test[i]) for i in range(len(X_test))]
        metric = BrierScore()
        progressive_val_venn(model, stream, metric=metric)
        assert metric._n == len(X_test)

    def test_progressive_val_venn_learn_false(self, venn_model):
        model, X_test, y_test = venn_model
        metric = BrierScore()
        progressive_val_venn(model, X_test, y_test, metric=metric, learn=False)
        assert metric._n == len(X_test)

    def test_iter_progressive_val_venn(self, venn_model):
        model, X_test, y_test = venn_model
        snapshots = list(iter_progressive_val_venn(model, X_test, y_test, step=5))
        assert len(snapshots) == len(X_test) // 5
        assert snapshots[0]["step"] == 5
        assert "brier_score" in snapshots[0] or "BrierScore" in str(snapshots[0])

    def test_iter_progressive_val_venn_timestamps(self, venn_model):
        model, X_test, y_test = venn_model
        stream = [(X_test[i], y_test[i], f"t{i}") for i in range(len(X_test))]
        snapshots = list(iter_progressive_val_venn(model, stream, step=10))
        assert snapshots[0]["t"] == "t9"

    def test_progressive_val_venn_delay(self, venn_model):
        """Fixed delay: all labels resolved after progressive_val_venn returns."""
        model, X_test, y_test = venn_model
        metric = BrierScore()
        progressive_val_venn(model, X_test, y_test, metric=metric, delay=3)
        assert metric._n == len(X_test)

    def test_lazy_teacher_venn(self, venn_model):
        """y=None in stream skips metric update and learning for Venn predictor."""
        model, X_test, y_test = venn_model
        # Every other label is missing.
        stream = [(X_test[i], y_test[i] if i % 2 == 0 else None) for i in range(len(X_test))]
        metric = BrierScore()
        progressive_val_venn(model, stream, metric=metric)
        expected = sum(1 for i in range(len(X_test)) if i % 2 == 0)
        assert metric._n == expected

    def test_iter_progressive_val_venn_delay_teardown(self, venn_model):
        """iter_progressive_val_venn emits a teardown snapshot when delay > 0."""
        model, X_test, y_test = venn_model
        n = len(X_test)  # 20
        snaps = list(iter_progressive_val_venn(model, X_test, y_test, step=5, delay=3))
        # n//step regular snapshots + 1 teardown (delay=3 leaves last 3 unresolved).
        assert len(snaps) == n // 5 + 1
        assert snaps[-1]["step"] == n

    def test_trace_venn(self, venn_model):
        """progressive_val_venn(trace=True) records (y, VennPrediction) per step."""
        from online_cp.venn import VennPrediction

        model, X_test, y_test = venn_model
        metric = BrierScore()
        progressive_val_venn(model, X_test, y_test, metric=metric, trace=True)
        assert len(metric.trace_y) == metric._n == len(X_test)
        assert len(metric.trace_gamma) == len(X_test)
        assert all(isinstance(g, VennPrediction) for g in metric.trace_gamma)

    def test_iter_trace_venn(self, venn_model):
        """iter_progressive_val_venn(trace=True) carries trace data in snapshots."""
        model, X_test, y_test = venn_model
        snapshots = list(iter_progressive_val_venn(model, X_test, y_test, step=5, trace=True))
        total = sum(len(s["trace_gamma"]) for s in snapshots)
        assert total == len(X_test)


class TestPlotting:
    """Smoke tests for plotting functions."""

    def test_plot_coverage(self, ridge_model):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_coverage

        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric)

        ax = plot_coverage(metric, nominal=0.9)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_intervals(self, linear_dataset):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        X, y = linear_dataset
        model = ConformalRidgeRegressor(a=1.0)
        model.learn_initial_training_set(X[:50], y[:50])

        intervals = []
        for i in range(50, 70):
            iv = model.predict(X[i], epsilon=0.1)
            intervals.append(iv)
            model.learn_one(X[i], y[i])

        ax = plot_intervals(y[50:70], intervals)
        assert ax is not None

    def test_plot_set_sizes(self, ridge_model):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_set_sizes

        model, X_test, y_test = ridge_model
        metric = IntervalWidth()
        progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric)

        ax = plot_set_sizes(metric)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_coverage_accepts_composite(self, ridge_model):
        """plot_coverage picks the ErrorRate member out of a Metrics composite."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_coverage

        model, X_test, y_test = ridge_model
        metric = ErrorRate() + IntervalWidth()
        progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric)
        ax = plot_coverage(metric, nominal=0.9)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_set_sizes_accepts_composite(self, ridge_model):
        """plot_set_sizes picks IntervalWidth out of a Metrics composite."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_set_sizes

        model, X_test, y_test = ridge_model
        metric = ErrorRate() + IntervalWidth()
        progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric)
        ax = plot_set_sizes(metric)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_intervals_from_traced_metric(self, ridge_model):
        """plot_intervals(metric) works when metric was run with trace=True."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        model, X_test, y_test = ridge_model
        metric = ErrorRate() + IntervalWidth()
        progressive_val(model, X_test[:30], y_test[:30], epsilon=0.1, metric=metric, trace=True)
        ax = plot_intervals(metric)
        assert ax is not None

    def test_plot_intervals_metric_and_explicit_conflict(self, ridge_model):
        """Passing both a traced metric and intervals raises ValueError."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:10], y_test[:10], epsilon=0.1, metric=metric, trace=True)
        with pytest.raises(ValueError, match="do not also pass"):
            plot_intervals(metric, y_test[:10])

    def _legend_labels(self, ax):
        """Return the list of legend text entries on *ax*."""
        return [t.get_text() for t in ax.get_legend().get_texts()]

    def test_plot_intervals_default_label(self, ridge_model):
        """label=None keeps the 'Prediction interval' default in the legend."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, metric=metric, trace=True)
        ax = plot_intervals(metric)
        assert "Prediction interval" in self._legend_labels(ax)

    def test_plot_intervals_custom_label(self, ridge_model):
        """label= overrides the default legend label for the interval band."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, metric=metric, trace=True)
        ax = plot_intervals(metric, label="My Model")
        labels = self._legend_labels(ax)
        assert "My Model" in labels
        assert "Prediction interval" not in labels

    def test_plot_intervals_overlay_two_models(self, ridge_model):
        """Overlaying two models on one ax draws y_true only once, with distinct colours."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        model, X_test, y_test = ridge_model
        m1 = ErrorRate()
        progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, metric=m1, trace=True)
        # A second (independent) run for a second model.
        model2, X_test2, y_test2 = ridge_model
        m2 = ErrorRate()
        progressive_val(model2, X_test2[:20], y_test2[:20], epsilon=0.1, metric=m2, trace=True)

        ax = plot_intervals(m1, label="Model A", interval_kwargs={"color": "steelblue"})
        plot_intervals(m2, label="Model B", ax=ax, show_points=False, interval_kwargs={"color": "orange"})

        labels = self._legend_labels(ax)
        assert "Model A" in labels
        assert "Model B" in labels
        # y_true appears in the legend exactly once (its label is r"$y_{true}$").
        n_true = sum(1 for lab in labels if "y_{true}" in lab)
        assert n_true == 1
        # Two interval bands + one true-value scatter = 3 collections.
        assert len(ax.collections) == 3
        # The two bands have the requested colours.
        colors = {tuple(np.round(c.get_facecolor()[0], 3)) for c in ax.collections[:2]}
        assert len(colors) == 2  # two distinct colours

    def test_plot_intervals_show_points_false_no_scatter(self, ridge_model):
        """show_points=False omits the true-value scatter entirely."""
        import matplotlib

        matplotlib.use("Agg")
        from online_cp.plotting import plot_intervals

        model, X_test, y_test = ridge_model
        metric = ErrorRate()
        progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, metric=metric, trace=True)
        ax = plot_intervals(metric, show_points=False)
        # Only the interval band, no scatter.
        assert len(ax.collections) == 1

    def test_plot_martingale(self, uniform_p_values):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import FixedStrategy, PluginMartingale
        from online_cp.plotting import plot_martingale

        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        mart = PluginMartingale(betting_strategy=fs)
        for p in uniform_p_values[:50]:
            mart.update(p)

        ax = plot_martingale(mart)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_detector_ville(self, uniform_p_values):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import SimpleJumper, VilleWrapper
        from online_cp.plotting import plot_detector

        sj = SimpleJumper(J=0.1)
        ville = VilleWrapper(sj, threshold=20)
        for p in uniform_p_values[:50]:
            ville.update(p)

        ax = plot_detector(ville)
        assert ax is not None
        assert len(ax.lines) >= 2  # trajectory + running max + threshold

    def test_plot_detector_ville_with_alarm(self):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import SimpleJumper, VilleWrapper
        from online_cp.plotting import plot_detector

        sj = SimpleJumper(J=0.1)
        ville = VilleWrapper(sj, threshold=5)
        # Feed very small p-values to trigger alarm
        for _ in range(50):
            ville.update(0.01)

        assert ville.rejected
        assert ville.rejection_time is not None
        ax = plot_detector(ville, change_point=10)
        assert ax is not None

    def test_plot_detector_cusum(self, uniform_p_values):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import CUSUMWrapper, SimpleJumper
        from online_cp.plotting import plot_detector

        sj = SimpleJumper(J=0.1)
        cusum = CUSUMWrapper(sj)
        for p in uniform_p_values[:50]:
            cusum.update(p)

        ax = plot_detector(cusum, threshold=100)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_detector_cusum_with_barrier(self):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import CUSUMWrapper, SimpleJumper
        from online_cp.plotting import plot_detector

        sj = SimpleJumper(J=0.1)
        cusum = CUSUMWrapper(sj, barrier_slope=0.01)
        for _ in range(50):
            cusum.update(0.5)

        ax = plot_detector(cusum, threshold=100)
        assert ax is not None

    def test_plot_detector_sr(self, uniform_p_values):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import ShiryaevRobertsWrapper, SimpleJumper
        from online_cp.plotting import plot_detector

        sj = SimpleJumper(J=0.1)
        sr = ShiryaevRobertsWrapper(sj)
        for p in uniform_p_values[:50]:
            sr.update(p)

        ax = plot_detector(sr, threshold=100)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_detector_sr_with_alarm(self):
        import matplotlib

        matplotlib.use("Agg")
        from online_cp import ShiryaevRobertsWrapper, SimpleJumper
        from online_cp.plotting import plot_detector

        sj = SimpleJumper(J=0.1)
        sr = ShiryaevRobertsWrapper(sj)
        # Small p-values to trigger alarm
        for _ in range(50):
            sr.update(0.01)

        ax = plot_detector(sr, threshold=20, change_point=5)
        assert ax is not None

    def test_plot_detector_invalid_type(self):
        from online_cp.plotting import plot_detector

        with pytest.raises(TypeError, match="Expected VilleWrapper"):
            plot_detector("not a wrapper")

    def test_ville_rejection_time(self):
        from online_cp import SimpleJumper, VilleWrapper

        sj = SimpleJumper(J=0.1)
        ville = VilleWrapper(sj, threshold=5)
        # Uniform p-values → no rejection
        rng = np.random.default_rng(123)
        for p in rng.uniform(size=20):
            ville.update(p)
        assert ville.rejection_time is None

        # Now push small values
        for _ in range(100):
            ville.update(0.01)
        assert ville.rejected
        assert 21 <= ville.rejection_time <= 120  # fires within small-p region


# ---------------------------------------------------------------------------
# Calibration & Venn plotting tests
# ---------------------------------------------------------------------------


class TestPlotReliabilityDiagram:
    def test_from_arrays(self):
        from online_cp.plotting import plot_reliability_diagram

        predicted = np.array([0.1, 0.4, 0.6, 0.9, 0.8, 0.3])
        observed = np.array([0, 0, 1, 1, 1, 0])
        ax = plot_reliability_diagram(predicted, observed, n_bins=3)
        assert ax is not None
        assert len(ax.lines) >= 1  # at least diagonal + curve

    def test_from_calibration_error(self):
        from online_cp.metrics import CalibrationError
        from online_cp.plotting import plot_reliability_diagram
        from online_cp.venn import VennPrediction

        m = CalibrationError()
        for _ in range(20):
            venn = VennPrediction.binary(0.7, 0.7)
            m.update(y=1, venn=venn)
        ax = plot_reliability_diagram(m, n_bins=5)
        assert ax is not None

    def test_requires_observed_for_arrays(self):
        from online_cp.plotting import plot_reliability_diagram

        with pytest.raises(ValueError, match="observed is required"):
            plot_reliability_diagram(np.array([0.5, 0.5]))


class TestPlotReliabilityDiagramVenn:
    def test_smoke_binary(self):
        from online_cp.plotting import plot_reliability_diagram_venn
        from online_cp.venn import VennPrediction

        preds = [VennPrediction.binary(0.3, 0.8) for _ in range(20)]
        labels = np.array([0, 1] * 10)
        ax = plot_reliability_diagram_venn(preds, labels, n_bins=5)
        assert ax is not None
        assert len(ax.lines) >= 2  # diagonal + at least one curve

    def test_which_point_only(self):
        from online_cp.plotting import plot_reliability_diagram_venn
        from online_cp.venn import VennPrediction

        preds = [VennPrediction.binary(0.4, 0.9) for _ in range(20)]
        labels = np.array([0, 1] * 10)
        ax = plot_reliability_diagram_venn(preds, labels, which="point", n_bins=5)
        assert ax is not None

    def test_unknown_label_raises(self):
        from online_cp.plotting import plot_reliability_diagram_venn
        from online_cp.venn import VennPrediction

        preds = [VennPrediction.binary(0.5, 0.5)]
        labels = np.array([99])  # not in label_space [0, 1]
        with pytest.raises(ValueError, match="not found in label_space"):
            plot_reliability_diagram_venn(preds, labels)


class TestPlotSharpness:
    def test_smoke(self):
        from online_cp.plotting import plot_sharpness
        from online_cp.venn import VennPrediction

        preds = [VennPrediction.binary(0.2, 0.9) for _ in range(30)]
        ax = plot_sharpness(preds, n_bins=10)
        assert ax is not None

    def test_empty_predictions(self):
        from online_cp.plotting import plot_sharpness

        ax = plot_sharpness([], n_bins=10)
        assert ax is not None  # returns empty axes without crashing


class TestPlotPitHistogram:
    def test_smoke(self):
        from online_cp.plotting import plot_pit_histogram

        rng = np.random.default_rng(42)
        pit_values = rng.uniform(size=100)
        ax = plot_pit_histogram(pit_values, n_bins=10)
        assert ax is not None
        assert len(ax.patches) >= 10  # histogram bars


class TestPlotCalibrationConditional:
    def test_smoke(self):
        from online_cp.metrics import CalibrationError
        from online_cp.plotting import plot_calibration_conditional
        from online_cp.venn import VennPrediction

        m1 = CalibrationError()
        m2 = CalibrationError()
        venn = VennPrediction.binary(0.7, 0.7)
        for _ in range(20):
            m1.update(y=1, venn=venn)
            m2.update(y=0, venn=venn)
        ax = plot_calibration_conditional({"Group A": m1, "Group B": m2}, n_bins=5)
        assert ax is not None
        assert len(ax.lines) >= 1  # at least diagonal


class TestPlotValidation:
    def test_venn_reliability_invalid_which_raises(self):
        from online_cp.plotting import plot_reliability_diagram_venn
        from online_cp.venn import VennPrediction

        preds = [VennPrediction.binary(0.5, 0.5)]
        labels = np.array([1])
        with pytest.raises(ValueError, match="which must be"):
            plot_reliability_diagram_venn(preds, labels, which="typo")

    def test_intervals_length_mismatch_raises(self):
        from online_cp.plotting import plot_intervals

        y_true = np.array([1.0, 2.0, 3.0])
        intervals = [(0.5, 1.5), (1.5, 2.5)]  # only 2, need 3
        with pytest.raises(ValueError, match="Length mismatch"):
            plot_intervals(y_true, intervals)

    def test_pit_histogram_empty(self):
        from online_cp.plotting import plot_pit_histogram

        ax = plot_pit_histogram(np.array([]))
        assert ax is not None
        assert "no data" in ax.get_title().lower()
