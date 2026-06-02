"""Evaluation metrics for online conformal prediction.

Provides stateful metric objects that accumulate results step-by-step
in the online learning protocol. Metrics are composable via the ``+``
operator and share a uniform ``.update()`` interface.

Example
-------
>>> from online_cp.metrics import ErrorRate, IntervalWidth
>>> metric = ErrorRate() + IntervalWidth()
>>> # In your online loop:
>>> # metric.update(y=y_true, Gamma=prediction_set)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Metric",
    "Metrics",
    "ErrorRate",
    "ObservedExcess",
    "ObservedFuzziness",
    "SetSize",
    "IntervalWidth",
    "WinklerScore",
    "CRPS",
    "BrierScore",
    "LogLoss",
    "Width",
    "CalibrationError",
]


class Metric:
    """Base class for online evaluation metrics.

    Subclasses must implement ``_score(self, y, Gamma, **kw)`` which
    returns a single scalar for one observation.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self) -> None:
        self._values: list[float] = []
        self._sum = 0.0
        self._n = 0

    def update(self, y: Any = None, Gamma: Any = None, **kw: Any) -> float:
        """Record one observation.

        Parameters
        ----------
        y : scalar
            True label / response.
        Gamma : ConformalPredictionSet or ConformalPredictionInterval
            Prediction output from a conformal predictor.
        **kw : dict
            Additional keyword arguments (p_values, cpd, epsilon, etc.).
            Each metric picks what it needs.

        Returns
        -------
        float
            The metric value for this observation.
        """
        val = self._score(y=y, Gamma=Gamma, **kw)
        self._values.append(val)
        self._sum += val
        self._n += 1
        return val

    def _score(self, y, Gamma, **kw):
        raise NotImplementedError

    def get(self) -> float:
        """Return the running mean of the metric."""
        if self._n == 0:
            return 0.0
        return self._sum / self._n

    @property
    def values(self) -> NDArray[np.floating[Any]]:
        """Per-step history as a numpy array."""
        return np.asarray(self._values)

    def cumulative_mean(self) -> NDArray[np.floating[Any]]:
        """Cumulative running mean at each step."""
        return np.cumsum(self._values) / np.arange(1, self._n + 1)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        self._values = []
        self._sum = 0.0
        self._n = 0

    def __repr__(self) -> str:
        return f"{self.name}: {self.get():.4f}"

    def __add__(self, other: Metric | Metrics) -> Metrics:
        if isinstance(other, Metrics):
            return Metrics([self] + other._metrics)
        if isinstance(other, Metric):
            return Metrics([self, other])
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented


class Metrics:
    """Composite of multiple metrics, created via the ``+`` operator.

    Example
    -------
    >>> metric = ErrorRate() + IntervalWidth()
    >>> metric.update(y=1.0, Gamma=interval)
    """

    def __init__(self, metrics: list[Metric]) -> None:
        self._metrics = list(metrics)

    def update(self, y: Any = None, Gamma: Any = None, **kw: Any) -> None:
        """Update all contained metrics."""
        for m in self._metrics:
            m.update(y=y, Gamma=Gamma, **kw)

    def get(self) -> dict[str, float]:
        """Return a dict of {name: running_mean} for all metrics."""
        return {m.name: m.get() for m in self._metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        for m in self._metrics:
            m.reset()

    def __repr__(self) -> str:
        return "\n".join(repr(m) for m in self._metrics)

    def __add__(self, other: Metrics | Metric) -> Metrics:
        if isinstance(other, Metrics):
            return Metrics(self._metrics + other._metrics)
        if isinstance(other, Metric):
            return Metrics(self._metrics + [other])
        return NotImplemented

    def __getitem__(self, key):
        """Access a metric by name or index."""
        if isinstance(key, int):
            return self._metrics[key]
        for m in self._metrics:
            if m.name == key:
                return m
        raise KeyError(f"Metric '{key}' not found")

    def __iter__(self):
        return iter(self._metrics)

    def __len__(self):
        return len(self._metrics)


# ---------------------------------------------------------------------------
# Concrete metrics
# ---------------------------------------------------------------------------


class ErrorRate(Metric):
    """Fraction of times the true label falls outside the prediction set.

    Works for both classifiers (prediction sets) and regressors (intervals).
    """

    def _score(self, y, Gamma, **kw):
        return float(y not in Gamma)


class ObservedExcess(Metric):
    """Number of incorrect labels in the prediction set (OE).

    For classifiers: |Gamma| - 1 if y in Gamma, else |Gamma|.
    A conditionally proper efficiency criterion.
    """

    def _score(self, y, Gamma, **kw):
        if y in Gamma:
            return float(len(Gamma) - 1)
        return float(len(Gamma))


class ObservedFuzziness(Metric):
    """Sum of p-values for incorrect labels (OF).

    Requires ``p_values`` keyword argument (dict: label -> p-value).
    A conditionally proper efficiency criterion independent of epsilon.
    """

    def _score(self, y, Gamma=None, *, p_values=None, **kw):
        if p_values is None:
            raise ValueError("ObservedFuzziness requires p_values keyword argument")
        return float(sum(p for label, p in p_values.items() if label != y))


class SetSize(Metric):
    """Size of the prediction set (for classifiers)."""

    def _score(self, y, Gamma, **kw):
        return float(len(Gamma))


class IntervalWidth(Metric):
    """Width of the prediction interval (for regressors)."""

    def _score(self, y, Gamma, **kw):
        return float(Gamma.width())


class WinklerScore(Metric):
    """Winkler interval score — a proper scoring rule for interval forecasts.

    Requires the prediction interval to have ``.lower`` and ``.upper``
    attributes, and ``epsilon`` to be provided.
    """

    def _score(self, y, Gamma, *, epsilon=None, **kw):
        if epsilon is None:
            epsilon = getattr(Gamma, "epsilon", 0.1)
        lower = Gamma.lower
        upper = Gamma.upper
        if not np.isfinite(lower) or not np.isfinite(upper):
            return np.inf
        width = upper - lower
        if y < lower:
            return width + (2.0 / epsilon) * (lower - y)
        elif y > upper:
            return width + (2.0 / epsilon) * (y - upper)
        return width


class CRPS(Metric):
    """Continuous Ranked Probability Score for conformal predictive distributions.

    .. deprecated::
        This class delegates to :class:`TruncatedCRPS`. Prefer using
        ``TruncatedCRPS`` or ``ConformalCRPS`` explicitly.

    Requires ``cpd`` keyword argument (a conformal predictive distribution object).
    """

    def _score(self, y, Gamma=None, *, cpd=None, **kw):
        import warnings
        warnings.warn(
            "CRPS is deprecated. Use TruncatedCRPS or ConformalCRPS instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        return TruncatedCRPS()._score(y, Gamma, cpd=cpd, **kw)


class TruncatedCRPS(Metric):
    """Truncated CRPS: integrate over data support only.

    Computes the standard CRPS formula but restricted to the interval
    [C_1, C_{n-1}] (the finite critical points), avoiding divergence
    from the mass-deficient tails. Uses exact piecewise-constant summation.

    .. math::

        \\text{TruncatedCRPS} = \\sum_{j=1}^{m-1}
            [Q_\\tau(C_j) - \\mathbf{1}(y \\leq C_j)]^2 \\, (C_{j+1} - C_j)

    where :math:`Q_\\tau` is the CPD evaluated at randomisation :math:`\\tau`
    and the sum runs over finite critical points.

    NOT strictly proper (due to truncation), but pragmatic and common.

    Requires ``cpd`` keyword argument and optionally ``tau`` (default 0.5).
    """

    def _score(self, y, Gamma=None, *, cpd=None, tau=0.5, **kw):
        if cpd is None:
            raise ValueError("TruncatedCRPS requires cpd keyword argument")

        # Get finite critical points (exclude -inf and +inf sentinels)
        Y = cpd.Y
        finite_mask = np.isfinite(Y)
        C = Y[finite_mask]

        if len(C) < 2:
            return 0.0

        # CDF values at finite critical points
        Q = np.array([cpd(c, tau) for c in C])
        # Indicator: 1{y <= C_j}
        indicator = (y <= C).astype(float)

        # Piecewise-constant integrand: [Q(C_j) - I(y <= C_j)]^2 * (C_{j+1} - C_j)
        widths = np.diff(C)
        integrand = (Q[:-1] - indicator[:-1]) ** 2

        return float(np.sum(integrand * widths))


class ConformalCRPS(Metric):
    """Conformal CRPS: replace the indicator with the CPD's conformal indicator.

    Instead of the standard Heaviside indicator :math:`\\mathbf{1}(y \\leq x)`,
    uses the CPD's own CDF evaluated at the true outcome :math:`Q_\\tau(y)` as
    the "conformal indicator", ensuring the integrand shares the CPD's
    mass-deficiency bounds.

    .. math::

        \\text{ConformalCRPS} = \\sum_{j=1}^{m-1}
            [Q_\\tau(C_j) - Q_\\tau(y)]^2 \\, (C_{j+1} - C_j)

    Finite by construction (both terms bounded by CPD mass). NOT strictly
    proper, but theoretically motivated.

    Requires ``cpd`` keyword argument and optionally ``tau`` (default 0.5).
    """

    def _score(self, y, Gamma=None, *, cpd=None, tau=0.5, **kw):
        if cpd is None:
            raise ValueError("ConformalCRPS requires cpd keyword argument")

        # Get finite critical points
        Y = cpd.Y
        finite_mask = np.isfinite(Y)
        C = Y[finite_mask]

        if len(C) < 2:
            return 0.0

        # CDF at each critical point and at y
        Q = np.array([cpd(c, tau) for c in C])
        Q_y = float(cpd(y, tau))

        # Piecewise-constant: [Q(C_j) - Q(y)]^2 * (C_{j+1} - C_j)
        widths = np.diff(C)
        integrand = (Q[:-1] - Q_y) ** 2

        return float(np.sum(integrand * widths))


# ---------------------------------------------------------------------------
# Venn prediction metrics
# ---------------------------------------------------------------------------


class BrierScore(Metric):
    """Brier score for Venn predictor outputs.

    Evaluates the aggregated point probability from a ``VennPrediction``
    using the standard Brier score: :math:`(p_{\\text{point}} - \\mathbf{1}\\{y = k\\})^2`
    summed over all labels.

    Requires ``venn`` keyword argument (a ``VennPrediction`` object).
    """

    def _score(self, y, Gamma=None, *, venn=None, **kw):
        if venn is None:
            raise ValueError("BrierScore requires venn keyword argument")
        point = venn.point  # shape (|Y|,), sums to 1
        label_idx = np.searchsorted(venn.label_space, y)
        indicator = np.zeros(len(venn.label_space))
        indicator[label_idx] = 1.0
        return float(np.sum((point - indicator) ** 2))


class LogLoss(Metric):
    """Log loss for Venn predictor outputs.

    Evaluates the aggregated point probability from a ``VennPrediction``
    using negative log-likelihood: :math:`-\\log(p_{\\text{point}}[y])`.

    Requires ``venn`` keyword argument (a ``VennPrediction`` object).
    """

    _EPS = 1e-15  # clip to avoid log(0)

    def _score(self, y, Gamma=None, *, venn=None, **kw):
        if venn is None:
            raise ValueError("LogLoss requires venn keyword argument")
        point = venn.point
        label_idx = np.searchsorted(venn.label_space, y)
        prob_y = np.clip(point[label_idx], self._EPS, 1.0 - self._EPS)
        return float(-np.log(prob_y))


class Width(Metric):
    """Width (sharpness) of a Venn multiprobability prediction.

    For binary predictions: :math:`p_1 - p_0`.
    For multiclass: mean over labels of (max − min) probability across
    hypotheses.

    Requires ``venn`` keyword argument (a ``VennPrediction`` object).
    """

    def _score(self, y, Gamma=None, *, venn=None, **kw):
        if venn is None:
            raise ValueError("Width requires venn keyword argument")
        probs = venn.probs  # shape (|Y|, |Y|)
        # For each label (column), compute max - min across hypotheses (rows)
        widths = probs.max(axis=0) - probs.min(axis=0)
        return float(widths.mean())


class CalibrationError(Metric):
    """Expected Calibration Error (ECE) for Venn predictor outputs.

    Accumulates (predicted probability, true indicator) pairs from a
    stream of ``VennPrediction`` objects, enabling post-hoc ECE
    computation via binning.

    Two modes:

    - ``use_hypothesis=False`` (default): evaluates the *point estimate*
      from ``venn.point``. This is the aggregated probability and is
      typically well-calibrated empirically.
    - ``use_hypothesis=True``: evaluates the correct-hypothesis probability
      :math:`P^y(y)`, which is *theoretically calibrated* by the Venn
      validity guarantee (ALRW2 Theorem 6.4).

    The per-step ``_score()`` returns :math:`|p - \\mathbf{1}\\{y = k\\}|`
    (absolute calibration gap), so ``metric.value`` gives the running mean
    absolute error. Use :meth:`ece` for the standard binned ECE.

    For binary classification, the predicted probability is :math:`P(y=1)`.
    For multiclass, probabilities are stored per-class (one-vs-rest) and
    ECE is computed as a weighted average across classes.

    Requires ``venn`` keyword argument (a ``VennPrediction`` object).

    Parameters
    ----------
    use_hypothesis : bool, default False
        If True, use the correct-hypothesis probability :math:`P^y(y)`
        instead of the point estimate.
    max_history : int or None, default None
        Maximum number of (predicted, observed) pairs to store.
        If None, stores all. When exceeded, oldest pairs are discarded.
    """

    def __init__(self, use_hypothesis: bool = False, max_history: int | None = None) -> None:
        super().__init__()
        self.use_hypothesis = use_hypothesis
        self.max_history = max_history
        self._pairs: list[tuple[float, int]] = []  # (predicted_prob, true_indicator)

    def _score(self, y, Gamma=None, *, venn=None, **kw):
        if venn is None:
            raise ValueError("CalibrationError requires venn keyword argument")

        label_idx = int(np.searchsorted(venn.label_space, y))
        # For calibration we track P(y = positive_class) vs 1{y = positive_class}
        # For binary: positive_class = label_space[1]
        # For multiclass: use label_space[-1] (or user can filter externally)
        pos_idx = min(1, len(venn.label_space) - 1)

        if self.use_hypothesis:
            # P^y(positive_class): probability of positive class under correct hypothesis
            pred_prob = float(venn.probs[label_idx, pos_idx])
        else:
            # Point estimate probability for positive class
            pred_prob = float(venn.point[pos_idx])

        indicator = int(label_idx == pos_idx)
        self._pairs.append((pred_prob, indicator))

        if self.max_history is not None and len(self._pairs) > self.max_history:
            self._pairs.pop(0)

        return abs(pred_prob - indicator)

    @property
    def predicted(self) -> NDArray:
        """Array of stored predicted probabilities."""
        if not self._pairs:
            return np.array([])
        return np.array([p for p, _ in self._pairs])

    @property
    def observed(self) -> NDArray:
        """Array of stored true indicators (always 1 for correct-class prob)."""
        if not self._pairs:
            return np.array([])
        return np.array([o for _, o in self._pairs])

    def ece(self, n_bins: int = 10, strategy: str = "uniform") -> float:
        """Compute binned Expected Calibration Error.

        Parameters
        ----------
        n_bins : int, default 10
            Number of bins.
        strategy : str, default "uniform"
            Binning strategy: ``"uniform"`` (equal-width) or
            ``"quantile"`` (equal-mass).

        Returns
        -------
        float
            Weighted average of |mean_predicted - fraction_positive| across
            bins, weighted by bin count.
        """
        if not self._pairs:
            return 0.0

        predicted = self.predicted
        observed = self.observed

        if strategy == "uniform":
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            bin_edges = np.quantile(predicted, quantiles)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose 'uniform' or 'quantile'.")

        ece_val = 0.0
        n_total = len(predicted)

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            else:
                mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])

            n_bin = mask.sum()
            if n_bin == 0:
                continue

            mean_pred = predicted[mask].mean()
            frac_pos = observed[mask].mean()
            ece_val += (n_bin / n_total) * abs(mean_pred - frac_pos)

        return float(ece_val)

    def bin_data(self, n_bins: int = 10, strategy: str = "uniform") -> tuple[NDArray, NDArray, NDArray]:
        """Return binned calibration data for plotting.

        Parameters
        ----------
        n_bins : int, default 10
            Number of bins.
        strategy : str, default "uniform"
            Binning strategy: ``"uniform"`` or ``"quantile"``.

        Returns
        -------
        mean_predicted : ndarray
            Mean predicted probability per bin.
        fraction_positive : ndarray
            Fraction of positive outcomes per bin.
        bin_counts : ndarray
            Number of samples per bin.
        """
        if not self._pairs:
            return np.array([]), np.array([]), np.array([])

        predicted = self.predicted
        observed = self.observed

        if strategy == "uniform":
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            bin_edges = np.quantile(predicted, quantiles)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy {strategy!r}.")

        mean_preds = []
        frac_pos = []
        counts = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            else:
                mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])

            n_bin = mask.sum()
            if n_bin == 0:
                continue

            mean_preds.append(predicted[mask].mean())
            frac_pos.append(observed[mask].mean())
            counts.append(n_bin)

        return np.array(mean_preds), np.array(frac_pos), np.array(counts)
