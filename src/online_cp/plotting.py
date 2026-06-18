"""Plotting utilities for online conformal prediction.

Provides convenience functions for visualising martingale trajectories,
coverage curves, prediction intervals, and set sizes.

All functions return a matplotlib ``Axes`` object for composability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from online_cp.martingale import ConformalTestMartingale, CUSUMWrapper, ShiryaevRobertsWrapper, VilleWrapper
    from online_cp.metrics import Metric

__all__ = [
    "plot_coverage",
    "plot_detector",
    "plot_martingale",
    "plot_intervals",
    "plot_set_sizes",
    "plot_reliability_diagram",
    "plot_reliability_diagram_venn",
    "plot_sharpness",
    "plot_pit_histogram",
    "plot_calibration_conditional",
]


def _get_ax(ax):
    """Get or create a matplotlib Axes."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def plot_coverage(metric: Metric, *, nominal: float | None = None, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Plot the running coverage (1 - error rate) over time.

    Parameters
    ----------
    metric : Metric
        A metric object with a ``.values`` attribute (list of per-step scores).
        Typically an ``ErrorRate`` metric.
    nominal : float, optional
        Nominal coverage level (e.g. 0.9). Draws a horizontal reference line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    values = np.array(metric.values)
    # Coverage = 1 - cumulative error rate
    cumulative_coverage = 1.0 - np.cumsum(values) / np.arange(1, len(values) + 1)

    kwargs.setdefault("label", "Coverage")
    ax.plot(np.arange(1, len(values) + 1), cumulative_coverage, **kwargs)

    if nominal is not None:
        ax.axhline(nominal, color="red", linestyle="--", alpha=0.7, label=f"Nominal ({nominal})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage")
    ax.set_title("Running Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_martingale(martingale: ConformalTestMartingale, *, log_scale: bool = True, threshold: float | None = 100, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Plot a martingale trajectory.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        A martingale object with ``.log_martingale_values`` or ``.martingale_values``.
    log_scale : bool
        If True (default), plot log10(M_n). If False, plot M_n.
    threshold : float or None
        Draw a horizontal line at this rejection threshold. None = no line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    if log_scale:
        values = np.array(martingale.log_martingale_values) / np.log(10)
        ylabel = r"$\log_{10} M_n$"
        if threshold is not None:
            ax.axhline(
                np.log10(threshold), color="red", linestyle="--", alpha=0.7,
                label=f"Threshold ({threshold})"
            )
    else:
        values = np.array(martingale.martingale_values)
        ylabel = r"$M_n$"
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({threshold})")

    kwargs.setdefault("label", "Martingale")
    ax.plot(np.arange(len(values)), values, **kwargs)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title("Martingale Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_detector(wrapper: VilleWrapper | CUSUMWrapper | ShiryaevRobertsWrapper, *, threshold: float | None = None, log_scale: bool = True, change_point: int | None = None, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Plot a detection wrapper's statistic trajectory with alarm markers.

    Accepts a VilleWrapper, CUSUMWrapper, or ShiryaevRobertsWrapper and
    renders the appropriate diagnostic plot including threshold lines and
    alarm time markers.

    Parameters
    ----------
    wrapper : VilleWrapper, CUSUMWrapper, or ShiryaevRobertsWrapper
        A detection wrapper that has been updated with p-values.
    threshold : float or None
        Alarm threshold to draw. For VilleWrapper, defaults to ``wrapper.threshold``.
        For CUSUM/SR wrappers, must be provided to mark alarm times.
    log_scale : bool
        If True (default), plot in log₁₀ scale. If False, plot on natural scale.
    change_point : int or None
        If provided, draw a vertical dashed line at this step (true change-point).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    **kwargs
        Passed to the main trajectory ``ax.plot()`` call.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from online_cp.martingale import CUSUMWrapper, ShiryaevRobertsWrapper, VilleWrapper

    ax = _get_ax(ax)

    if isinstance(wrapper, VilleWrapper):
        _plot_ville(wrapper, threshold=threshold, log_scale=log_scale, ax=ax, **kwargs)
    elif isinstance(wrapper, CUSUMWrapper):
        _plot_cusum(wrapper, threshold=threshold, log_scale=log_scale, ax=ax, **kwargs)
    elif isinstance(wrapper, ShiryaevRobertsWrapper):
        _plot_sr(wrapper, threshold=threshold, log_scale=log_scale, ax=ax, **kwargs)
    else:
        raise TypeError(
            f"Expected VilleWrapper, CUSUMWrapper, or ShiryaevRobertsWrapper, "
            f"got {type(wrapper).__name__}"
        )

    if change_point is not None:
        ax.axvline(change_point, color="black", linestyle=":", linewidth=1.5,
                   alpha=0.7, label=f"Change-point (t={change_point})")

    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def _plot_ville(wrapper, *, threshold, log_scale, ax, **kwargs):
    """Plot VilleWrapper: martingale trajectory + running-max envelope + threshold."""
    thresh = threshold if threshold is not None else wrapper.threshold
    log_values = np.array(wrapper.martingale.log_martingale_values)
    steps = np.arange(len(log_values))

    # Compute running maximum
    log_running_max = np.maximum.accumulate(log_values)

    if log_scale:
        scale = 1.0 / np.log(10)
        trajectory = log_values * scale
        envelope = log_running_max * scale
        thresh_val = np.log10(thresh)
        ylabel = r"$\log_{10} M_n$"
    else:
        trajectory = np.exp(log_values)
        envelope = np.exp(log_running_max)
        thresh_val = thresh
        ylabel = r"$M_n$"

    kwargs.setdefault("label", "Martingale")
    ax.plot(steps, trajectory, **kwargs)
    ax.plot(steps, envelope, color="tab:purple", alpha=0.6, linestyle="--",
            linewidth=1, label="Running max")
    ax.axhline(thresh_val, color="red", linestyle="--", alpha=0.7,
               label=f"Threshold ({thresh})")

    # Mark rejection time
    if wrapper.rejection_time is not None:
        ax.axvline(wrapper.rejection_time, color="red", linestyle="-", alpha=0.5,
                   linewidth=1.5, label=f"Alarm (t={wrapper.rejection_time})")

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title("Ville Procedure")


def _plot_cusum(wrapper, *, threshold, log_scale, ax, **kwargs):
    """Plot CUSUMWrapper: CUSUM statistic + threshold (with optional barrier) + alarms."""
    log_values = np.array(wrapper.log_cusum_values, dtype=float)
    steps = np.arange(len(log_values))

    if log_scale:
        scale = 1.0 / np.log(10)
        trajectory = log_values * scale
        ylabel = r"$\log_{10} \gamma_n$"
    else:
        trajectory = np.exp(log_values)
        ylabel = r"$\gamma_n$"

    kwargs.setdefault("label", "CUSUM")
    kwargs.setdefault("color", "tab:orange")
    ax.plot(steps, trajectory, **kwargs)

    # Threshold line (flat or with barrier slope)
    if threshold is not None:
        if wrapper.barrier_slope is not None:
            # Time-varying threshold: threshold + slope * n
            effective = np.array([threshold + wrapper.barrier_slope * n for n in steps])
            if log_scale:
                thresh_line = np.log10(np.maximum(effective, 1e-300))
            else:
                thresh_line = effective
            ax.plot(steps, thresh_line, color="red", linestyle="--", alpha=0.7,
                    label=f"Barrier (slope={wrapper.barrier_slope})")
        else:
            if log_scale:
                thresh_val = np.log10(threshold)
            else:
                thresh_val = threshold
            ax.axhline(thresh_val, color="red", linestyle="--", alpha=0.7,
                       label=f"Threshold ({threshold})")

        # Mark alarm times (threshold crossings)
        alarm_times = _find_alarm_times_cusum(log_values, threshold, wrapper.barrier_slope)
        for i, t in enumerate(alarm_times):
            label = "Alarm" if i == 0 else None
            ax.axvline(t, color="red", linestyle="-", alpha=0.5, linewidth=1.5, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title("CUSUM Procedure")


def _plot_sr(wrapper, *, threshold, log_scale, ax, **kwargs):
    """Plot ShiryaevRobertsWrapper: SR statistic + threshold + alarms."""
    sr_values = np.array(wrapper.sr_values, dtype=float)
    steps = np.arange(len(sr_values))

    if log_scale:
        # Clamp at 1.0: R_n < 1 means no evidence of change, shown as 0 in log-space
        trajectory = np.log10(np.maximum(sr_values, 1.0))
        ylabel = r"$\log_{10} R_n$"
    else:
        trajectory = sr_values
        ylabel = r"$R_n$"

    kwargs.setdefault("label", "Shiryaev-Roberts")
    kwargs.setdefault("color", "tab:green")
    ax.plot(steps, trajectory, **kwargs)

    if threshold is not None:
        if log_scale:
            thresh_val = np.log10(threshold)
        else:
            thresh_val = threshold
        ax.axhline(thresh_val, color="red", linestyle="--", alpha=0.7,
                   label=f"Threshold ({threshold})")

        # Mark alarm times
        alarm_times = _find_alarm_times_sr(sr_values, threshold)
        for i, t in enumerate(alarm_times):
            label = "Alarm" if i == 0 else None
            ax.axvline(t, color="red", linestyle="-", alpha=0.5, linewidth=1.5, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title("Shiryaev-Roberts Procedure")


def _find_alarm_times_cusum(log_values, threshold, barrier_slope):
    """Find steps where CUSUM statistic crosses the threshold."""
    alarm_times = []
    log_thresh = np.log(threshold)
    alarming = False
    for n, lv in enumerate(log_values):
        if barrier_slope is None:
            effective = log_thresh
        else:
            effective = np.log(max(threshold + barrier_slope * n, 1e-300))
        if lv >= effective:
            if not alarming:
                alarm_times.append(n)
                alarming = True
        else:
            alarming = False
    return alarm_times


def _find_alarm_times_sr(sr_values, threshold):
    """Find steps where SR statistic crosses the threshold."""
    alarm_times = []
    alarming = False
    for n, v in enumerate(sr_values):
        if v >= threshold:
            if not alarming:
                alarm_times.append(n)
                alarming = True
        else:
            alarming = False
    return alarm_times


def plot_intervals(y_true: NDArray[np.floating[Any]] | Sequence[float], intervals: Sequence[Any], *, ax: Axes | None = None, point_kwargs: dict[str, Any] | None = None, interval_kwargs: dict[str, Any] | None = None) -> Axes:
    """Plot prediction intervals with true values overlaid.

    Parameters
    ----------
    y_true : array-like
        True response values.
    intervals : list of tuples or objects with .lower/.upper attributes
        Prediction intervals. Each element is either a (lower, upper) tuple
        or an object with ``.lower`` and ``.upper`` attributes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    point_kwargs : dict, optional
        Keyword arguments for the true-value scatter plot.
    interval_kwargs : dict, optional
        Keyword arguments for the interval vertical lines.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    y_true = np.asarray(y_true)
    n = len(y_true)
    if len(intervals) != n:
        raise ValueError(
            f"Length mismatch: y_true has {n} elements but intervals has {len(intervals)}"
        )
    steps = np.arange(n)

    # Extract lower/upper bounds
    lowers = np.empty(n)
    uppers = np.empty(n)
    for i, iv in enumerate(intervals):
        if hasattr(iv, "lower") and hasattr(iv, "upper"):
            lowers[i] = iv.lower
            uppers[i] = iv.upper
        else:
            lowers[i] = iv[0]
            uppers[i] = iv[1]

    # Plot intervals
    ikw = {"color": "steelblue", "alpha": 0.3, "label": "Prediction interval"}
    if interval_kwargs:
        ikw.update(interval_kwargs)
    ax.fill_between(steps, lowers, uppers, **ikw)

    # Plot true values
    pkw = {"color": "black", "s": 8, "zorder": 5, "label": r"$y_{true}$"}
    if point_kwargs:
        pkw.update(point_kwargs)
    ax.scatter(steps, y_true, **pkw)

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title("Prediction Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_set_sizes(metric: Metric, *, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Plot running average set size (or interval width) over time.

    Parameters
    ----------
    metric : Metric
        A metric with a ``.values`` attribute (e.g. SetSize or IntervalWidth).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    values = np.array(metric.values)
    running_mean = np.cumsum(values) / np.arange(1, len(values) + 1)

    kwargs.setdefault("label", f"Running mean {metric.name}")
    ax.plot(np.arange(1, len(values) + 1), running_mean, **kwargs)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.name)
    ax.set_title(f"Running {metric.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    predicted: NDArray | Any,
    observed: NDArray | None = None,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a reliability diagram (calibration curve).

    Bins predicted probabilities and plots mean predicted vs observed
    frequency, with a diagonal reference line for perfect calibration.

    Parameters
    ----------
    predicted : ndarray or CalibrationError
        Array of predicted probabilities in [0, 1], or a
        ``CalibrationError`` metric object (from which stored pairs are
        extracted).
    observed : ndarray or None
        Binary array of outcomes (1 = positive, 0 = negative).
        Required if ``predicted`` is an ndarray, ignored if a metric.
    n_bins : int, default 10
        Number of bins.
    strategy : str, default "uniform"
        ``"uniform"`` (equal-width) or ``"quantile"`` (equal-mass).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to the main ``ax.plot()`` call.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from online_cp.metrics import CalibrationError

    ax = _get_ax(ax)

    if isinstance(predicted, CalibrationError):
        mean_pred, frac_pos, counts = predicted.bin_data(n_bins, strategy)
    else:
        predicted = np.asarray(predicted)
        if observed is None:
            raise ValueError("observed is required when predicted is an array")
        observed = np.asarray(observed)
        # Bin manually
        if strategy == "uniform":
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            bin_edges = np.quantile(predicted, quantiles)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy {strategy!r}.")

        mean_pred_list, frac_pos_list, counts_list = [], [], []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            else:
                mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            mean_pred_list.append(predicted[mask].mean())
            frac_pos_list.append(observed[mask].mean())
            counts_list.append(n_bin)
        mean_pred = np.array(mean_pred_list)
        frac_pos = np.array(frac_pos_list)
        _counts = np.array(counts_list)  # noqa: F841 (retained for potential future use)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Calibration curve
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("label", "Model")
    ax.plot(mean_pred, frac_pos, **kwargs)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_reliability_diagram_venn(
    predictions: Sequence[Any],
    labels: NDArray | Sequence,
    *,
    n_bins: int = 10,
    which: str = "both",
    target_label: int | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Reliability diagram for Venn multiprobability predictions.

    Demonstrates the Venn calibration guarantee by plotting:

    - **point**: reliability of the aggregated point estimate
    - **hypothesis**: reliability of the correct-hypothesis probability
      :math:`P^y(y)` (theoretically calibrated)
    - **both**: overlays both curves

    Parameters
    ----------
    predictions : sequence of VennPrediction
        Multiprobability predictions.
    labels : array-like
        True labels corresponding to each prediction.
    n_bins : int, default 10
        Number of bins.
    which : str, default "both"
        ``"point"``, ``"hypothesis"``, or ``"both"``.
    target_label : int or None
        For binary predictions, which label to plot P(y=target_label).
        Defaults to label_space[1] (the positive class).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to ``ax.plot()`` calls.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if which not in ("point", "hypothesis", "both"):
        raise ValueError(
            f"which must be 'point', 'hypothesis', or 'both', got {which!r}"
        )

    ax = _get_ax(ax)
    labels = np.asarray(labels)

    point_probs = []
    hyp_probs = []
    indicators = []

    for pred, y in zip(predictions, labels):
        label_idx = int(np.searchsorted(pred.label_space, y))
        if label_idx >= len(pred.label_space) or pred.label_space[label_idx] != y:
            raise ValueError(
                f"y={y!r} not found in label_space={pred.label_space.tolist()}"
            )
        if target_label is not None:
            target_idx = int(np.searchsorted(pred.label_space, target_label))
            if target_idx >= len(pred.label_space) or pred.label_space[target_idx] != target_label:
                raise ValueError(
                    f"target_label={target_label!r} not in "
                    f"label_space={pred.label_space.tolist()}"
                )
        else:
            target_idx = min(1, len(pred.label_space) - 1)

        point_probs.append(pred.point[target_idx])
        hyp_probs.append(float(pred.probs[label_idx, target_idx]))
        indicators.append(1 if label_idx == target_idx else 0)

    point_probs = np.array(point_probs)
    hyp_probs = np.array(hyp_probs)
    indicators = np.array(indicators)

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    def _bin_and_plot(predicted, label, **pkw):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        mp, fp = [], []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            else:
                mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            mp.append(predicted[mask].mean())
            fp.append(indicators[mask].mean())
        if mp:
            ax.plot(mp, fp, marker="o", label=label, **pkw)

    if which in ("point", "both"):
        _bin_and_plot(point_probs, "Point estimate", **kwargs)
    if which in ("hypothesis", "both"):
        kw2 = dict(kwargs)
        kw2.setdefault("linestyle", "--")
        _bin_and_plot(hyp_probs, r"Correct-hypothesis $P^y(y)$", **kw2)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Venn Reliability Diagram")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_sharpness(
    predictions: Sequence[Any],
    *,
    n_bins: int = 20,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Histogram of Venn multiprobability widths (sharpness).

    For binary predictions, plots the distribution of :math:`|p_1 - p_0|`.
    For multiclass, plots the mean per-label width (max − min across
    hypotheses).

    Parameters
    ----------
    predictions : sequence of VennPrediction
        Multiprobability predictions.
    n_bins : int, default 20
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to ``ax.hist()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    widths = []
    for pred in predictions:
        probs = pred.probs
        per_label_widths = probs.max(axis=0) - probs.min(axis=0)
        widths.append(per_label_widths.mean())

    widths = np.array(widths)

    if len(widths) == 0:
        ax.set_title("Sharpness (no data)")
        return ax

    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("alpha", 0.7)
    ax.hist(widths, bins=n_bins, **kwargs)

    ax.axvline(widths.mean(), color="red", linestyle="--", alpha=0.7,
               label=f"Mean = {widths.mean():.3f}")

    ax.set_xlabel("Multiprobability width")
    ax.set_ylabel("Count")
    ax.set_title("Sharpness (Multiprobability Width Distribution)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_pit_histogram(
    pit_values: NDArray | Sequence[float],
    *,
    n_bins: int = 10,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """PIT histogram for conformal predictive distributions.

    Plots a histogram of Probability Integral Transform values
    :math:`F(y_{\\text{true}})` and compares to the Uniform[0,1]
    reference. Under exchangeability, PIT values should be uniform.

    Parameters
    ----------
    pit_values : array-like
        PIT values, typically computed as ``cpd(y_true, tau)`` for each
        test point.
    n_bins : int, default 10
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to ``ax.bar()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    pit_values = np.asarray(pit_values)
    n_total = len(pit_values)

    if n_total == 0:
        ax.set_title("PIT Histogram (no data)")
        return ax

    counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    freq = counts / n_total

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = 1.0 / n_bins

    kwargs.setdefault("alpha", 0.7)
    kwargs.setdefault("edgecolor", "black")
    ax.bar(bin_centers, freq, width=bin_width * 0.9, **kwargs)

    # Uniform reference
    uniform_level = 1.0 / n_bins
    ax.axhline(uniform_level, color="red", linestyle="--", alpha=0.7,
               label=f"Uniform (1/{n_bins})")

    # 95% confidence band (Binomial)
    se = np.sqrt(uniform_level * (1 - uniform_level) / n_total)
    ax.axhspan(uniform_level - 1.96 * se, uniform_level + 1.96 * se,
               alpha=0.1, color="red", label="95% band")

    ax.set_xlabel("PIT value")
    ax.set_ylabel("Relative frequency")
    ax.set_title("PIT Histogram (CPS Calibration)")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_calibration_conditional(
    metric_dict: dict[str, Any],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Overlay reliability diagrams for multiple groups (Mondrian lens).

    Parameters
    ----------
    metric_dict : dict[str, CalibrationError]
        Mapping from group name to ``CalibrationError`` metric.
    n_bins : int, default 10
        Number of bins per group.
    strategy : str, default "uniform"
        Binning strategy: ``"uniform"`` or ``"quantile"``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to each ``ax.plot()`` call.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for i, (name, metric) in enumerate(metric_dict.items()):
        mean_pred, frac_pos, counts = metric.bin_data(n_bins, strategy)
        if len(mean_pred) == 0:
            continue
        kw = dict(kwargs)
        kw.setdefault("marker", "o")
        kw.setdefault("label", name)
        kw.setdefault("color", f"C{i % 10}")
        ax.plot(mean_pred, frac_pos, **kw)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Conditional Calibration (Mondrian)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax
