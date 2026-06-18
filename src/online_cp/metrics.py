r"""Evaluation metrics for online conformal prediction.

Provides stateful metric objects that accumulate results step-by-step
in the online learning protocol. Metrics are composable via the ``+``
operator and share a uniform ``.update()`` interface.

The metrics fall into three theoretical families:

- **Criteria of efficiency** for conformal predictors and transducers
  [ALRW2 §3.1]. These come in *prior* (label-independent) and *observed*
  (label-dependent) variants, and in *$\epsilon$-dependent* and *$\epsilon$-free*
  forms. :class:`SetSize` (N), :class:`ObservedExcess` (OE) and
  :class:`ObservedFuzziness` (OF) are **conditionally strongly proper**
  [ALRW2 §3.1.5, Thm 3.1]; :class:`ErrorRate` measures validity rather than
  efficiency.
- **Proper scoring rules** for distributional / probabilistic forecasts:
  :class:`WinklerScore`, :class:`BrierScore`, :class:`LogLoss`, and the CRPS
  family (:class:`TruncatedCRPS`, :class:`ConformalCRPS`).
- **Calibration diagnostics**: :class:`CalibrationError`, :class:`Width`.

References to ALRW2 are to V. Vovk, A. Gammerman and G. Shafer,
*Algorithmic Learning in a Random World*, 2nd ed., Springer, 2022.

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

    Calling :meth:`update` forwards the observation to every contained metric,
    and :meth:`get` returns a ``{name: running_mean}`` dictionary.

    Example
    -------
    >>> metric = ErrorRate() + SetSize()
    >>> metric.update(y=1, Gamma={1, 2})
    >>> metric.get()
    {'ErrorRate': 0.0, 'SetSize': 2.0}
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
    r"""Error rate: fraction of times the true label is excluded.

    For a prediction set / interval $\Gamma^\epsilon$ at significance level
    $\epsilon$, the per-step score is the miss indicator

    $$
    \mathrm{err}^\epsilon = \mathbf{1}\{y \notin \Gamma^\epsilon\},
    $$

    and ``get()`` returns the running error rate $\mathrm{Err}_n^\epsilon / n$.
    This is a measure of *validity* rather than efficiency: for an (exactly)
    valid conformal predictor the errors at level $\epsilon$ behave like
    independent Bernoulli($\epsilon$) trials, so
    $\mathrm{Err}_n^\epsilon / n \to \epsilon$ almost surely [ALRW2 §2.1, §3.1].
    Works for both classifiers (prediction sets) and regressors (intervals).
    """

    def _score(self, y, Gamma, **kw):
        return float(y not in Gamma)


class ObservedExcess(Metric):
    r"""Observed excess (OE): number of *false* labels in the prediction set.

    Per-step score at significance level $\epsilon$,

    $$
    \mathrm{OE}^\epsilon = \bigl|\Gamma^\epsilon \setminus \{y\}\bigr|
        = |\Gamma^\epsilon| - \mathbf{1}\{y \in \Gamma^\epsilon\},
    $$

    i.e. $|\Gamma| - 1$ when the true label is covered and $|\Gamma|$ when it is
    not. ``get()`` returns the average over the stream.

    **Criterion type.** OE is an *observed* (it depends on the realised label
    $y$) and *$\epsilon$-dependent* criterion of efficiency. By
    [ALRW2 §3.1.5, Thm 3.1] it is one of the four criteria (S, N, OF, OE) that
    are **conditionally strongly proper**: the conditional-probability idealised
    conformity measure $A(x, y) = Q(y \mid x)$, and every refinement of it, is
    optimal for OE. Intuitively, a conditionally proper criterion rewards
    conformity scores that rank labels by their true conditional probability, so
    optimising OE does not pull the underlying scorer away from the
    Bayes-optimal ordering. Contrast with the prior $\epsilon$-dependent
    counterpart :class:`SetSize` (N) and the $\epsilon$-free observed
    counterpart :class:`ObservedFuzziness` (OF).

    Smaller is better.
    """

    def _score(self, y, Gamma, **kw):
        if y in Gamma:
            return float(len(Gamma) - 1)
        return float(len(Gamma))


class ObservedFuzziness(Metric):
    r"""Observed fuzziness (OF): sum of p-values for the *false* labels.

    Given the system of conformal p-values $(p^{y'} : y' \in \mathbf{Y})$ for the
    test object, the per-step score is

    $$
    \mathrm{OF} = \sum_{y' \neq y} p^{y'},
    $$

    the sum of p-values over all labels other than the realised one. ``get()``
    returns the average over the stream.

    **Criterion type.** OF is an *observed* and *$\epsilon$-free* criterion of
    efficiency (it needs no significance level, only the p-values). By
    [ALRW2 §3.1.5, Thm 3.1] it is **conditionally strongly proper** — it shares
    the optimal idealised conformity measures of S, N and OE, namely all
    refinements of the conditional-probability measure $A(x, y) = Q(y \mid x)$.
    Being $\epsilon$-free, it summarises efficiency across all significance
    levels at once, and is the natural $\epsilon$-free analogue of
    :class:`ObservedExcess` (OE).

    Requires the ``p_values`` keyword argument (a ``dict`` mapping each label to
    its conformal p-value). Smaller is better.
    """

    def _score(self, y, Gamma=None, *, p_values=None, **kw):
        if p_values is None:
            raise ValueError("ObservedFuzziness requires p_values keyword argument")
        return float(sum(p for label, p in p_values.items() if label != y))


class SetSize(Metric):
    r"""Set size (N): number of labels in the prediction set.

    Per-step score $|\Gamma^\epsilon|$ at significance level $\epsilon$;
    ``get()`` returns the average. Intended for classification; for regression
    use :class:`IntervalWidth` as the analogous sharpness measure.

    **Criterion type.** N is a *prior* criterion (it ignores the realised label
    $y$) and is *$\epsilon$-dependent*. By [ALRW2 §3.1.5, Thm 3.1] it is
    **conditionally strongly proper**, sharing the optimal idealised conformity
    measures of S, OF and OE. It is the prior counterpart of the observed
    criterion :class:`ObservedExcess` (OE): where OE counts only false labels, N
    counts all labels — including the true one, which a valid predictor covers
    with probability $1 - \epsilon$.

    Smaller is better.
    """

    def _score(self, y, Gamma, **kw):
        return float(len(Gamma))


class IntervalWidth(Metric):
    r"""Interval width: sharpness measure for regression.

    Per-step score is the length $u - \ell$ of the prediction interval
    $\Gamma^\epsilon = [\ell, u]$ produced at significance level $\epsilon$;
    ``get()`` returns the average. This is the regression analogue of
    :class:`SetSize` (N): given (approximately) valid coverage, narrower
    intervals are more informative.

    Width alone does not penalise miscoverage, so it should be read together with
    :class:`ErrorRate`; see :class:`WinklerScore` for a single proper score that
    combines width and coverage.
    """

    def _score(self, y, Gamma, **kw):
        return float(Gamma.width())


class WinklerScore(Metric):
    r"""Winkler interval score — a proper scoring rule for interval forecasts.

    For a central $(1 - \epsilon)$ prediction interval $[\ell, u]$ the per-step
    score is

    $$
    W^\epsilon = (u - \ell)
    + \frac{2}{\epsilon}(\ell - y)\,\mathbf{1}\{y < \ell\}
    + \frac{2}{\epsilon}(y - u)\,\mathbf{1}\{y > u\},
    $$

    i.e. the interval width plus a coverage penalty, scaled by $2/\epsilon$, for
    realisations falling outside the interval. Unlike the pure efficiency
    criteria (:class:`SetSize`, :class:`IntervalWidth`) this is a **proper
    scoring rule**: it is minimised in expectation by the true central
    $(1-\epsilon)$ interval, so it rewards calibration and sharpness jointly
    rather than width alone. Smaller is better; an infinite interval yields an
    infinite score.

    Notes
    -----
    Requires the prediction interval to expose ``.lower`` and ``.upper``.

    The significance level used for the $2/\epsilon$ tail penalty is resolved
    per score: an explicit ``epsilon`` keyword takes precedence; otherwise the
    interval's own ``Gamma.epsilon`` attribute is used; if neither is available
    it falls back to ``0.1``. It must be positive.
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
    r"""Continuous Ranked Probability Score (CRPS) for conformal predictive systems.

    The CRPS of a predictive CDF $F$ against an outcome $y$ is
    $\int_{-\infty}^{\infty} (F(t) - \mathbf{1}\{y \leq t\})^2 \, dt$, a
    **proper scoring rule** for distributional forecasts. For a conformal
    predictive distribution the integral is taken against the (randomised)
    predictive CDF $Q_\tau$.

    !!! warning "Deprecated"
        This class is retained for backward compatibility and simply delegates
        to :class:`TruncatedCRPS`. The raw CRPS integral diverges for a conformal
        predictive distribution because its tails are mass-deficient (they do not
        reach 0 and 1); prefer :class:`TruncatedCRPS` or :class:`ConformalCRPS`,
        which keep the score finite by truncating the integral or replacing the
        Heaviside indicator, respectively.

    Requires the ``cpd`` keyword argument (a conformal predictive distribution).
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
    r"""Truncated CRPS: integrate the CRPS over the data support only.

    Computes the CRPS integrand but restricts it to the interval
    $[C_1, C_{m-1}]$ spanned by the *finite* critical points of the conformal
    predictive distribution, avoiding the divergence caused by its
    mass-deficient tails. Using the exact piecewise-constant form,

    $$
    \mathrm{TruncatedCRPS} = \sum_{j=1}^{m-1}
        \bigl[Q_\tau(C_j) - \mathbf{1}\{y \leq C_j\}\bigr]^2 \,
        (C_{j+1} - C_j),
    $$

    where $C_1 < \dots < C_m$ are the finite critical points (the sorted
    conformal scores), $Q_\tau$ is the conformal predictive CDF evaluated at
    randomisation $\tau$, and the indicator is the usual Heaviside step.

    Because of the truncation it is **not strictly proper**, but it is finite,
    cheap, and a common pragmatic choice. See :class:`ConformalCRPS` for a
    variant that replaces the Heaviside indicator instead of truncating.

    Requires the ``cpd`` keyword argument and optionally ``tau`` (default 0.5).
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
    r"""Conformal CRPS: replace the Heaviside indicator with the CPD's own CDF.

    Instead of truncating the integral, this variant replaces the Heaviside
    indicator $\mathbf{1}\{y \leq C_j\}$ with the conformal predictive CDF
    evaluated at the true outcome, $Q_\tau(y)$, used as a "conformal indicator".
    This keeps both terms bounded by the CPD's mass, so the integrand inherits
    the same mass-deficiency bounds:

    $$
    \mathrm{ConformalCRPS} = \sum_{j=1}^{m-1}
        \bigl[Q_\tau(C_j) - Q_\tau(y)\bigr]^2 \, (C_{j+1} - C_j),
    $$

    with $C_1 < \dots < C_m$ the finite critical points and $Q_\tau$ the
    conformal predictive CDF at randomisation $\tau$. Finite by construction;
    **not strictly proper**, but theoretically motivated and a natural companion
    to :class:`TruncatedCRPS`.

    Requires the ``cpd`` keyword argument and optionally ``tau`` (default 0.5).
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
    r"""Brier score for Venn predictor outputs — a proper scoring rule.

    Scores the aggregated point probability $p_{\text{point}}$ of a
    :class:`~online_cp.venn.VennPrediction` with the multiclass Brier score

    $$
    \mathrm{BS} = \sum_{k \in \mathbf{Y}}
        \bigl(p_{\text{point}}[k] - \mathbf{1}\{y = k\}\bigr)^2 ,
    $$

    the squared $\ell_2$ distance between the probability vector and the one-hot
    outcome. The Brier score is a **proper scoring rule**, so it is minimised in
    expectation by the true conditional class probabilities and rewards both
    calibration and sharpness. The Venn point probability is the multiprobability
    average and requires no renormalisation. Smaller is better; compare with
    :class:`LogLoss`.

    Requires the ``venn`` keyword argument (a ``VennPrediction`` object).
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
    r"""Log loss (negative log-likelihood) for Venn outputs — a proper scoring rule.

    Scores the aggregated point probability $p_{\text{point}}$ of a
    :class:`~online_cp.venn.VennPrediction` by

    $$
    \mathrm{LogLoss} = -\log p_{\text{point}}[y] ,
    $$

    the negative log-probability assigned to the realised label. Like the Brier
    score it is a **proper scoring rule**, but it penalises confident mistakes
    far more severely (unboundedly as $p_{\text{point}}[y] \to 0$). The
    probability is clipped to $[10^{-15}, 1]$ before taking the logarithm to
    avoid $-\infty$. Smaller is better; compare with :class:`BrierScore`.

    Requires the ``venn`` keyword argument (a ``VennPrediction`` object).
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
    r"""Width (sharpness) of a Venn multiprobability prediction.

    A Venn predictor outputs, for each label, an *interval* of probabilities
    (one entry per hypothesised label). This metric reports the mean width of
    those intervals,

    $$
    \mathrm{Width} = \frac{1}{|\mathbf{Y}|} \sum_{k \in \mathbf{Y}}
        \Bigl(\max_{h} P^{h}(k) - \min_{h} P^{h}(k)\Bigr) ,
    $$

    averaged over labels $k$ and taken across hypotheses $h$. For binary
    predictions this reduces to $|p_1 - p_0|$. Smaller width means a tighter
    (more decisive) multiprobability prediction; it measures sharpness, not
    validity, and is the Venn analogue of interval width.

    Requires the ``venn`` keyword argument (a ``VennPrediction`` object).
    """

    def _score(self, y, Gamma=None, *, venn=None, **kw):
        if venn is None:
            raise ValueError("Width requires venn keyword argument")
        probs = venn.probs  # shape (|Y|, |Y|)
        # For each label (column), compute max - min across hypotheses (rows)
        widths = probs.max(axis=0) - probs.min(axis=0)
        return float(widths.mean())


class CalibrationError(Metric):
    r"""Expected Calibration Error (ECE) for Venn predictor outputs.

    Accumulates (predicted probability, true indicator) pairs from a stream of
    :class:`~online_cp.venn.VennPrediction` objects, enabling post-hoc ECE
    computation by binning. With $B$ bins, the binned ECE is

    $$
    \mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}
        \bigl| \bar{p}_b - \bar{y}_b \bigr| ,
    $$

    where $n_b$ is the number of predictions in bin $b$, $\bar{p}_b$ their mean
    predicted probability and $\bar{y}_b$ the observed frequency of the target
    class; see :meth:`ece`.

    Two modes:

    - ``use_hypothesis=False`` (default): evaluates the *point estimate*
      from ``venn.point``. This is the aggregated probability and is
      typically well-calibrated empirically.
    - ``use_hypothesis=True``: evaluates the correct-hypothesis probability
      $P^y(y)$, which is *theoretically calibrated* by the Venn validity
      guarantee [ALRW2, Thm 6.4].

    The per-step ``_score()`` returns $|p - \mathbf{1}\{y = k\}|$ (the absolute
    calibration gap), so ``metric.get()`` gives the running mean absolute error.
    Use :meth:`ece` for the standard binned ECE.

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
