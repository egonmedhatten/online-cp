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

import numpy as np

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
]


class Metric:
    """Base class for online evaluation metrics.

    Subclasses must implement ``_score(self, y, Gamma, **kw)`` which
    returns a single scalar for one observation.
    """

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self):
        self._values = []
        self._sum = 0.0
        self._n = 0

    def update(self, y=None, Gamma=None, **kw):
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

    def get(self):
        """Return the running mean of the metric."""
        if self._n == 0:
            return 0.0
        return self._sum / self._n

    @property
    def values(self):
        """Per-step history as a numpy array."""
        return np.asarray(self._values)

    def cumulative_mean(self):
        """Cumulative running mean at each step."""
        return np.cumsum(self._values) / np.arange(1, self._n + 1)

    def reset(self):
        """Reset the metric to its initial state."""
        self._values = []
        self._sum = 0.0
        self._n = 0

    def __repr__(self):
        return f"{self.name}: {self.get():.4f}"

    def __add__(self, other):
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

    def __init__(self, metrics):
        self._metrics = list(metrics)

    def update(self, y=None, Gamma=None, **kw):
        """Update all contained metrics."""
        for m in self._metrics:
            m.update(y=y, Gamma=Gamma, **kw)

    def get(self):
        """Return a dict of {name: running_mean} for all metrics."""
        return {m.name: m.get() for m in self._metrics}

    def reset(self):
        """Reset all metrics."""
        for m in self._metrics:
            m.reset()

    def __repr__(self):
        return "\n".join(repr(m) for m in self._metrics)

    def __add__(self, other):
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

    Requires ``cpd`` keyword argument (a conformal predictive distribution object).
    """

    def _score(self, y, Gamma=None, *, cpd=None, **kw):
        if cpd is None:
            raise ValueError("CRPS requires cpd keyword argument")

        def integrand(x):
            if x <= y:
                return (cpd(x, 0) - int(y <= x)) ** 2
            else:
                return (cpd(x, 1) - int(y <= x)) ** 2

        vals = cpd.y_vals[1:-1]
        return float(np.trapz([integrand(x) for x in vals], vals))


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
