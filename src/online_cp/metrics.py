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
    "TruncatedCRPS",
    "ConformalCRPS",
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
        names = [m.name for m in self._metrics]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            raise ValueError(
                f"Duplicate metric names: {sorted(set(dupes))}. "
                "Subclass and override .name to disambiguate."
            )

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
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
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
            stacklevel=3,
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
        label_idx = int(np.searchsorted(venn.label_space, y))
        if label_idx >= len(venn.label_space) or venn.label_space[label_idx] != y:
            raise ValueError(
                f"y={y!r} not found in label_space={venn.label_space.tolist()}"
            )
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
        label_idx = int(np.searchsorted(venn.label_space, y))
        if label_idx >= len(venn.label_space) or venn.label_space[label_idx] != y:
            raise ValueError(
                f"y={y!r} not found in label_space={venn.label_space.tolist()}"
            )
        prob_y = np.clip(point[label_idx], self._EPS, None)
        return float(-np.log(prob_y))


class Width(Metric):
    """Width (sharpness) of a Venn multiprobability prediction.

    Mean over labels of (max − min) probability across hypotheses.
    For binary predictions, this equals :math:`|p_1 - p_0|`.

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
    (absolute calibration gap), so ``metric.get()`` gives the running mean
    absolute error. Use :meth:`ece` for the standard binned ECE.

    For binary classification, the tracked class defaults to ``label_space[1]``
    (the "positive" class). For multiclass, specify ``target_class`` explicitly.

    Requires ``venn`` keyword argument (a ``VennPrediction`` object).

    Parameters
    ----------
    use_hypothesis : bool, default False
        If True, use the correct-hypothesis probability :math:`P^y(y)`
        instead of the point estimate.
    target_class : int or None, default None
        Which class to track calibration for. If None, defaults to
        ``label_space[1]`` (binary positive class). For multiclass problems,
        this must be specified explicitly.
    max_history : int or None, default None
        Maximum number of (predicted, observed) pairs to store.
        If None, stores all. When exceeded, oldest pairs are discarded.
    """

    def __init__(
        self,
        use_hypothesis: bool = False,
        target_class: Any = None,
        max_history: int | None = None,
    ) -> None:
        super().__init__()
        self.use_hypothesis = use_hypothesis
        self.target_class = target_class
        self.max_history = max_history
        self._pairs: list[tuple[float, int]] = []  # (predicted_prob, true_indicator)

    def reset(self) -> None:
        """Reset the metric and clear stored calibration pairs."""
        super().reset()
        self._pairs = []

    def update(self, y: Any = None, Gamma: Any = None, **kw: Any) -> float:
        """Record one observation, respecting max_history for all state."""
        val = super().update(y=y, Gamma=Gamma, **kw)
        if self.max_history is not None and len(self._values) > self.max_history:
            removed = self._values.pop(0)
            self._sum -= removed
            self._n -= 1
        return val

    def _score(self, y, Gamma=None, *, venn=None, **kw):
        if venn is None:
            raise ValueError("CalibrationError requires venn keyword argument")

        label_idx = int(np.searchsorted(venn.label_space, y))
        if label_idx >= len(venn.label_space) or venn.label_space[label_idx] != y:
            raise ValueError(
                f"y={y!r} not found in label_space={venn.label_space.tolist()}"
            )

        # Determine target class index
        if self.target_class is not None:
            pos_idx = int(np.searchsorted(venn.label_space, self.target_class))
            if pos_idx >= len(venn.label_space) or venn.label_space[pos_idx] != self.target_class:
                raise ValueError(
                    f"target_class={self.target_class!r} not in "
                    f"label_space={venn.label_space.tolist()}"
                )
        elif len(venn.label_space) <= 2:
            # Binary: default to label_space[1] (positive class)
            pos_idx = min(1, len(venn.label_space) - 1)
        else:
            raise ValueError(
                "For multiclass (|Y| > 2), target_class must be specified. "
                f"label_space={venn.label_space.tolist()}"
            )

        if self.use_hypothesis:
            # P^y(positive_class): probability of target class under correct hypothesis
            pred_prob = float(venn.probs[label_idx, pos_idx])
        else:
            # Point estimate probability for target class
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

    def _bin_edges(self, predicted: NDArray, n_bins: int, strategy: str) -> NDArray:
        """Compute bin edges for calibration binning."""
        if strategy == "uniform":
            return np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(predicted, quantiles)
            edges[0] = 0.0
            edges[-1] = 1.0
            return edges
        else:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose 'uniform' or 'quantile'.")

    def _bin_masks(self, predicted: NDArray, bin_edges: NDArray, n_bins: int) -> list[NDArray]:
        """Compute boolean masks for each bin."""
        masks = []
        for i in range(n_bins):
            if i < n_bins - 1:
                masks.append((predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1]))
            else:
                masks.append((predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1]))
        return masks

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
        bin_edges = self._bin_edges(predicted, n_bins, strategy)
        masks = self._bin_masks(predicted, bin_edges, n_bins)

        ece_val = 0.0
        n_total = len(predicted)

        for mask in masks:
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
        bin_edges = self._bin_edges(predicted, n_bins, strategy)
        masks = self._bin_masks(predicted, bin_edges, n_bins)

        mean_preds = []
        frac_pos = []
        counts = []

        for mask in masks:
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            mean_preds.append(predicted[mask].mean())
            frac_pos.append(observed[mask].mean())
            counts.append(n_bin)

        return np.array(mean_preds), np.array(frac_pos), np.array(counts)
