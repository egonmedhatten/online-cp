"""Tests for streaming evaluation and plotting utilities."""

import numpy as np
import pytest

from online_cp import ConformalRidgeRegressor
from online_cp.evaluate import progressive_val, iter_progressive_val
from online_cp.metrics import ErrorRate, IntervalWidth


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
        progressive_val(
            model, X[50:60], y[50:60], epsilon=0.1,
            learn=lambda i, x, y: i % 2 == 0
        )
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
        snapshots = list(
            iter_progressive_val(model, X_test[:20], y_test[:20], epsilon=0.1, step=10)
        )
        assert len(snapshots) == 2
        assert snapshots[0]["step"] == 10
        assert snapshots[1]["step"] == 20


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

    def test_plot_martingale(self, uniform_p_values):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_martingale
        from online_cp import PluginMartingale, FixedStrategy

        fs = FixedStrategy(pdf=lambda x: 2 * (1 - x), check_integration=False)
        mart = PluginMartingale(betting_strategy=fs, min_sample_size=0)
        for p in uniform_p_values[:50]:
            mart.update(p)

        ax = plot_martingale(mart)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_plot_detector_ville(self, uniform_p_values):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_detector
        from online_cp import SimpleJumper, VilleWrapper

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
        from online_cp.plotting import plot_detector
        from online_cp import SimpleJumper, VilleWrapper

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
        from online_cp.plotting import plot_detector
        from online_cp import SimpleJumper, CUSUMWrapper

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
        from online_cp.plotting import plot_detector
        from online_cp import SimpleJumper, CUSUMWrapper

        sj = SimpleJumper(J=0.1)
        cusum = CUSUMWrapper(sj, barrier_slope=0.01)
        for _ in range(50):
            cusum.update(0.5)

        ax = plot_detector(cusum, threshold=100)
        assert ax is not None

    def test_plot_detector_sr(self, uniform_p_values):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_detector
        from online_cp import SimpleJumper, ShiryaevRobertsWrapper

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
        from online_cp.plotting import plot_detector
        from online_cp import SimpleJumper, ShiryaevRobertsWrapper

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
