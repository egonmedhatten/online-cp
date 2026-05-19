"""Plotting utilities for online conformal prediction.

Provides convenience functions for visualising martingale trajectories,
coverage curves, prediction intervals, and set sizes.

All functions return a matplotlib ``Axes`` object for composability.
"""

import numpy as np

__all__ = [
    "plot_coverage",
    "plot_martingale",
    "plot_intervals",
    "plot_set_sizes",
]


def _get_ax(ax):
    """Get or create a matplotlib Axes."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def plot_coverage(metric, *, nominal=None, ax=None, **kwargs):
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


def plot_martingale(martingale, *, log_scale=True, threshold=100, ax=None, **kwargs):
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


def plot_intervals(y_true, intervals, *, ax=None, point_kwargs=None, interval_kwargs=None):
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


def plot_set_sizes(metric, *, ax=None, **kwargs):
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
