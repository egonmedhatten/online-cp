"""Evaluation protocol for online conformal predictors.

Provides ``progressive_val`` — the standard test-then-train loop — as a
standalone function that decouples evaluation logic from model classes.

Supports both array inputs (classic batch) and streaming iterables for
truly online workflows. The ``learn`` parameter controls whether and when
the model learns from each observation.

Example
-------
>>> from online_cp import ConformalRidgeRegressor
>>> from online_cp.metrics import ErrorRate, IntervalWidth
>>> from online_cp.evaluate import progressive_val
>>>
>>> model = ConformalRidgeRegressor(a=1.0)
>>> model.learn_initial_training_set(X_train, y_train)
>>> metric = ErrorRate() + IntervalWidth()
>>> progressive_val(model, X_test, y_test, epsilon=0.1, metric=metric)
>>> print(metric)
"""

from __future__ import annotations

from typing import Any, Callable, Generator, Iterable

import numpy as np
from numpy.typing import NDArray

if __name__ != "__main__":
    from online_cp.metrics import Metric, Metrics, ObservedFuzziness, CRPS, BrierScore, LogLoss, Width

__all__ = [
    "progressive_val",
    "iter_progressive_val",
    "progressive_val_venn",
    "iter_progressive_val_venn",
]


def _needs_p_values(metric):
    """Check if any metric in the composite requires p_values."""
    if isinstance(metric, Metrics):
        return any(_needs_p_values(m) for m in metric)
    return isinstance(metric, ObservedFuzziness)


def _needs_cpd(metric):
    """Check if any metric in the composite requires a CPD."""
    if isinstance(metric, Metrics):
        return any(_needs_cpd(m) for m in metric)
    return isinstance(metric, CRPS)


def _needs_venn(metric):
    """Check if any metric in the composite requires a VennPrediction."""
    if isinstance(metric, Metrics):
        return any(_needs_venn(m) for m in metric)
    return isinstance(metric, (BrierScore, LogLoss, Width))


def _iter_data(X_or_stream, y=None):
    """Normalize input to an iterable of (x, y, t) triples.

    Accepts:
    - X array + y array (classic): yields (x_i, y_i, i)
    - Iterable of (x, y) tuples: yields (x, y, i)
    - Iterable of (x, y, t) triples: yields (x, y, t)
    """
    if y is not None:
        # Classic array input
        for i, (x_i, y_i) in enumerate(zip(X_or_stream, y)):
            yield x_i, y_i, i
    else:
        # Streaming iterable
        for i, item in enumerate(X_or_stream):
            if len(item) == 3:
                yield item[0], item[1], item[2]
            else:
                yield item[0], item[1], i


def _should_learn(learn, i, x_i, y_i):
    """Determine whether to learn from this example."""
    if learn is True:
        return True
    if learn is False:
        return False
    # Callable: learn(i, x, y) -> bool
    return learn(i, x_i, y_i)


def progressive_val(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    epsilon: float | NDArray[np.floating[Any]] | None = None,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    print_every: int = 0,
) -> Metric | Metrics:
    """Run the progressive validation (test-then-train) protocol.

    For each sample: predict, update metric, then optionally learn.

    Parameters
    ----------
    model : conformal predictor
        Must implement ``predict(x, epsilon=...)`` and ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array of shape (n_samples, n_features), OR an
        iterable of ``(x, y)`` tuples or ``(x, y, t)`` triples for streaming.
    y : array-like, optional
        True labels / responses. Required when X is an array; omitted when
        X is a streaming iterable.
    epsilon : float, optional
        Significance level. If None, uses model default.
    metric : Metric or Metrics, optional
        Metric(s) to update at each step. Defaults to ErrorRate.
    learn : bool or callable, optional
        Whether the model learns from each example (default: True).
        If callable, called as ``learn(i, x, y) -> bool`` to decide per step.
    print_every : int, optional
        Print metric summary every N steps. 0 = no printing.

    Returns
    -------
    metric : Metric or Metrics
        The same metric object, updated in place.
    """
    if metric is None:
        from online_cp.metrics import ErrorRate
        metric = ErrorRate()

    needs_p = _needs_p_values(metric)
    needs_cpd = _needs_cpd(metric)

    predict_kw = {}
    if epsilon is not None:
        predict_kw["epsilon"] = epsilon

    for i, (x_i, y_i, t_i) in enumerate(_iter_data(X, y)):
        # Predict
        kw = {}
        if needs_p:
            result = model.predict(x_i, return_p_values=True, **predict_kw)
            Gamma, p_values = result
            kw["p_values"] = p_values
        elif needs_cpd:
            cpd = model.predict_cpd(x_i)
            Gamma = cpd.predict_set(epsilon=epsilon) if epsilon else None
            kw["cpd"] = cpd
        else:
            Gamma = model.predict(x_i, **predict_kw)

        if epsilon is not None:
            kw["epsilon"] = epsilon

        # Update metric
        metric.update(y=y_i, Gamma=Gamma, **kw)

        # Learn (conditional)
        if _should_learn(learn, i, x_i, y_i):
            model.learn_one(x_i, y_i)

        # Progress printing
        if print_every > 0 and (i + 1) % print_every == 0:
            print(f"[{i + 1}] {metric}")

    return metric


def iter_progressive_val(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    epsilon: float | NDArray[np.floating[Any]] | None = None,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    step: int = 1,
) -> Generator[dict[str, Any], None, None]:
    """Iterate the progressive validation protocol, yielding checkpoints.

    Yields a snapshot of the metric state every ``step`` samples.
    Useful for plotting learning curves.

    Parameters
    ----------
    model : conformal predictor
        Must implement ``predict(x, epsilon=...)`` and ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array, or iterable of ``(x, y)`` / ``(x, y, t)`` tuples.
    y : array-like, optional
        True labels. Required when X is an array; omitted for streaming.
    epsilon : float, optional
        Significance level.
    metric : Metric or Metrics, optional
        Metric(s) to update. Defaults to ErrorRate.
    learn : bool or callable, optional
        Whether the model learns from each example. If callable, called as
        ``learn(i, x, y) -> bool``.
    step : int
        Yield every ``step`` samples.

    Yields
    ------
    dict
        Contains ``"step"`` (int, 1-indexed count), ``"t"`` (timestamp or index),
        and metric values.
    """
    if metric is None:
        from online_cp.metrics import ErrorRate
        metric = ErrorRate()

    needs_p = _needs_p_values(metric)
    needs_cpd = _needs_cpd(metric)

    predict_kw = {}
    if epsilon is not None:
        predict_kw["epsilon"] = epsilon

    for i, (x_i, y_i, t_i) in enumerate(_iter_data(X, y)):
        kw = {}
        if needs_p:
            result = model.predict(x_i, return_p_values=True, **predict_kw)
            Gamma, p_values = result
            kw["p_values"] = p_values
        elif needs_cpd:
            cpd = model.predict_cpd(x_i)
            Gamma = cpd.predict_set(epsilon=epsilon) if epsilon else None
            kw["cpd"] = cpd
        else:
            Gamma = model.predict(x_i, **predict_kw)

        if epsilon is not None:
            kw["epsilon"] = epsilon

        metric.update(y=y_i, Gamma=Gamma, **kw)

        if _should_learn(learn, i, x_i, y_i):
            model.learn_one(x_i, y_i)

        if (i + 1) % step == 0:
            if isinstance(metric, Metrics):
                snapshot = {"step": i + 1, "t": t_i}
                snapshot.update(metric.get())
            else:
                snapshot = {"step": i + 1, "t": t_i, metric.name: metric.get()}
            yield snapshot


# ---------------------------------------------------------------------------
# Venn predictor evaluation
# ---------------------------------------------------------------------------


def progressive_val_venn(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    print_every: int = 0,
) -> Metric | Metrics:
    """Run progressive validation for Venn predictors.

    Same test-then-train protocol as ``progressive_val``, but adapted for
    Venn predictors that return ``VennPrediction`` objects (no epsilon).

    Parameters
    ----------
    model : Venn predictor
        Must implement ``predict(x)`` returning a ``VennPrediction`` and
        ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array of shape (n_samples, n_features), OR an
        iterable of ``(x, y)`` tuples or ``(x, y, t)`` triples for streaming.
    y : array-like, optional
        True labels. Required when X is an array; omitted for streaming.
    metric : Metric or Metrics, optional
        Metric(s) to update at each step. Defaults to ``BrierScore()``.
    learn : bool or callable, optional
        Whether the model learns from each example (default: True).
        If callable, called as ``learn(i, x, y) -> bool``.
    print_every : int, optional
        Print metric summary every N steps. 0 = no printing.

    Returns
    -------
    metric : Metric or Metrics
        The same metric object, updated in place.
    """
    if metric is None:
        metric = BrierScore()

    for i, (x_i, y_i, t_i) in enumerate(_iter_data(X, y)):
        venn_pred = model.predict(x_i)
        metric.update(y=y_i, Gamma=None, venn=venn_pred)

        if _should_learn(learn, i, x_i, y_i):
            model.learn_one(x_i, y_i)

        if print_every > 0 and (i + 1) % print_every == 0:
            print(f"[{i + 1}] {metric}")

    return metric


def iter_progressive_val_venn(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    step: int = 1,
) -> Generator[dict[str, Any], None, None]:
    """Iterate progressive validation for Venn predictors, yielding checkpoints.

    Parameters
    ----------
    model : Venn predictor
        Must implement ``predict(x)`` returning a ``VennPrediction`` and
        ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array, or iterable of ``(x, y)`` / ``(x, y, t)`` tuples.
    y : array-like, optional
        True labels. Required when X is an array; omitted for streaming.
    metric : Metric or Metrics, optional
        Metric(s) to update. Defaults to ``BrierScore()``.
    learn : bool or callable, optional
        Whether the model learns from each example. If callable, called as
        ``learn(i, x, y) -> bool``.
    step : int
        Yield every ``step`` samples.

    Yields
    ------
    dict
        Contains ``"step"``, ``"t"``, and metric values.
    """
    if metric is None:
        metric = BrierScore()

    for i, (x_i, y_i, t_i) in enumerate(_iter_data(X, y)):
        venn_pred = model.predict(x_i)
        metric.update(y=y_i, Gamma=None, venn=venn_pred)

        if _should_learn(learn, i, x_i, y_i):
            model.learn_one(x_i, y_i)

        if (i + 1) % step == 0:
            if isinstance(metric, Metrics):
                snapshot = {"step": i + 1, "t": t_i}
                snapshot.update(metric.get())
            else:
                snapshot = {"step": i + 1, "t": t_i, metric.name: metric.get()}
            yield snapshot
