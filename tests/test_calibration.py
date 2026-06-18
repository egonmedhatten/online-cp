"""Tests for calibration diagnostics (CalibrationError metric + plotting)."""

import numpy as np
import pytest

from online_cp.metrics import CalibrationError
from online_cp.venn import VennPrediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_predictions_calibrated():
    """Perfectly calibrated binary predictions (p_y = y for each hypothesis)."""
    rng = np.random.default_rng(42)
    n = 200
    probs = rng.uniform(0.1, 0.9, size=n)
    labels = (rng.random(n) < probs).astype(int)
    predictions = [VennPrediction.binary(p, p) for p in probs]  # p0 = p1 = p (degenerate)
    return predictions, labels


@pytest.fixture
def binary_predictions_spread():
    """Binary predictions with spread (p0 != p1)."""
    rng = np.random.default_rng(123)
    n = 200
    predictions = []
    labels = []
    for _ in range(n):
        p0 = rng.uniform(0.1, 0.5)
        p1 = rng.uniform(0.5, 0.9)
        pred = VennPrediction.binary(p0, p1)
        y = rng.choice([0, 1])
        predictions.append(pred)
        labels.append(y)
    return predictions, np.array(labels)


# ---------------------------------------------------------------------------
# CalibrationError metric
# ---------------------------------------------------------------------------


class TestCalibrationError:
    def test_requires_venn_kwarg(self):
        metric = CalibrationError()
        with pytest.raises(ValueError, match="requires venn"):
            metric.update(y=1)

    def test_accumulates_values(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions, labels):
            metric.update(y=y, venn=pred)
        assert len(metric.values) == len(labels)
        assert len(metric._pairs) == len(labels)

    def test_ece_returns_float(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions, labels):
            metric.update(y=y, venn=pred)
        ece = metric.ece(n_bins=10)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_ece_empty_returns_zero(self):
        metric = CalibrationError()
        assert metric.ece() == 0.0

    def test_ece_perfect_calibration_near_zero(self):
        """Perfectly calibrated point predictions should give ECE ≈ 0."""
        rng = np.random.default_rng(99)
        n = 5000
        metric = CalibrationError()
        for _ in range(n):
            p = rng.uniform(0.1, 0.9)
            y = int(rng.random() < p)
            pred = VennPrediction.binary(p, p)
            metric.update(y=y, venn=pred)
        ece = metric.ece(n_bins=10)
        # With 5000 samples, ECE should be small (< 0.05)
        assert ece < 0.05

    def test_use_hypothesis_mode(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric_point = CalibrationError(use_hypothesis=False)
        metric_hyp = CalibrationError(use_hypothesis=True)
        for pred, y in zip(predictions, labels):
            metric_point.update(y=y, venn=pred)
            metric_hyp.update(y=y, venn=pred)
        # Both should produce valid ECE values
        assert 0.0 <= metric_point.ece() <= 1.0
        assert 0.0 <= metric_hyp.ece() <= 1.0

    def test_max_history(self):
        metric = CalibrationError(max_history=50)
        rng = np.random.default_rng(7)
        for _ in range(100):
            p = rng.uniform(0.2, 0.8)
            pred = VennPrediction.binary(p, p)
            metric.update(y=rng.choice([0, 1]), venn=pred)
        assert len(metric._pairs) == 50

    def test_predicted_observed_arrays(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions, labels):
            metric.update(y=y, venn=pred)
        assert metric.predicted.shape == (len(labels),)
        assert metric.observed.shape == (len(labels),)
        assert np.all((metric.predicted >= 0) & (metric.predicted <= 1))

    def test_bin_data(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions, labels):
            metric.update(y=y, venn=pred)
        mean_pred, frac_pos, counts = metric.bin_data(n_bins=5)
        assert len(mean_pred) <= 5
        assert len(frac_pos) <= 5
        assert counts.sum() == len(labels)

    def test_quantile_strategy(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions, labels):
            metric.update(y=y, venn=pred)
        ece_uniform = metric.ece(n_bins=10, strategy="uniform")
        ece_quantile = metric.ece(n_bins=10, strategy="quantile")
        assert 0.0 <= ece_uniform <= 1.0
        assert 0.0 <= ece_quantile <= 1.0

    def test_invalid_strategy_raises(self, binary_predictions_spread):
        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions[:10], labels[:10]):
            metric.update(y=y, venn=pred)
        with pytest.raises(ValueError, match="Unknown strategy"):
            metric.ece(strategy="bogus")


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------


class TestPlotReliabilityDiagram:
    def test_from_arrays(self):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_reliability_diagram

        rng = np.random.default_rng(42)
        predicted = rng.uniform(0, 1, 100)
        observed = (rng.random(100) < predicted).astype(int)
        ax = plot_reliability_diagram(predicted, observed, n_bins=5)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_from_metric(self, binary_predictions_spread):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_reliability_diagram

        predictions, labels = binary_predictions_spread
        metric = CalibrationError()
        for pred, y in zip(predictions, labels):
            metric.update(y=y, venn=pred)
        ax = plot_reliability_diagram(metric, n_bins=5)
        assert ax is not None

    def test_requires_observed_for_arrays(self):
        from online_cp.plotting import plot_reliability_diagram

        with pytest.raises(ValueError, match="observed is required"):
            plot_reliability_diagram(np.array([0.5, 0.6]))


class TestPlotReliabilityDiagramVenn:
    def test_which_point(self, binary_predictions_spread):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_reliability_diagram_venn

        predictions, labels = binary_predictions_spread
        ax = plot_reliability_diagram_venn(predictions, labels, which="point")
        assert ax is not None

    def test_which_hypothesis(self, binary_predictions_spread):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_reliability_diagram_venn

        predictions, labels = binary_predictions_spread
        ax = plot_reliability_diagram_venn(predictions, labels, which="hypothesis")
        assert ax is not None

    def test_which_both(self, binary_predictions_spread):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_reliability_diagram_venn

        predictions, labels = binary_predictions_spread
        ax = plot_reliability_diagram_venn(predictions, labels, which="both")
        assert ax is not None
        # Should have diagonal + 2 calibration curves
        assert len(ax.lines) >= 3


class TestPlotSharpness:
    def test_smoke(self, binary_predictions_spread):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_sharpness

        predictions, _ = binary_predictions_spread
        ax = plot_sharpness(predictions, n_bins=10)
        assert ax is not None


class TestPlotPitHistogram:
    def test_uniform_pit(self):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_pit_histogram

        rng = np.random.default_rng(42)
        pit_values = rng.uniform(0, 1, 500)
        ax = plot_pit_histogram(pit_values, n_bins=10)
        assert ax is not None

    def test_nonuniform_pit(self):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_pit_histogram

        # All PITs near 0 = mis-calibrated
        pit_values = np.random.default_rng(1).beta(0.5, 5, 200)
        ax = plot_pit_histogram(pit_values, n_bins=10)
        assert ax is not None


class TestPlotCalibrationConditional:
    def test_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        from online_cp.plotting import plot_calibration_conditional

        rng = np.random.default_rng(42)
        groups = {}
        for name in ("Group A", "Group B"):
            metric = CalibrationError()
            for _ in range(100):
                p = rng.uniform(0.2, 0.8)
                pred = VennPrediction.binary(p, p + rng.uniform(0, 0.1))
                metric.update(y=rng.choice([0, 1]), venn=pred)
            groups[name] = metric

        ax = plot_calibration_conditional(groups, n_bins=5)
        assert ax is not None
        assert len(ax.lines) >= 3  # diagonal + 2 group curves
