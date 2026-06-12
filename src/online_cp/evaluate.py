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

import heapq
from typing import Any, Callable, Generator, Iterable

import numpy as np
from numpy.typing import NDArray

if __name__ != "__main__":
    from online_cp.metrics import Metric, Metrics, ObservedFuzziness, CRPS, TruncatedCRPS, ConformalCRPS, BrierScore

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
    return isinstance(metric, (CRPS, TruncatedCRPS, ConformalCRPS))


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


def _wrap_progress(iterable, total=None, enabled=False, desc=None):
    """Optionally wrap an iterable with a tqdm progress bar.
    
    Returns a tuple (wrapped_iterable, pbar_instance) where pbar_instance
    is the tqdm instance (useful for set_postfix) or None if disabled.
    """
    if not enabled:
        return iterable, None
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required for progress bars. "
            "Install it with: pip install online-cp[progress]"
        ) from exc
    pbar = tqdm(iterable, total=total, desc=desc)
    return pbar, pbar


def _predict(model, x_i, needs_p, needs_cpd, predict_kw, epsilon, rng):
    """Return prediction artifacts ``(Gamma, kw)`` without updating any metric."""
    kw = {}
    if needs_p:
        result = model.predict(x_i, return_p_values=True, **predict_kw)
        Gamma, p_values = result
        kw["p_values"] = p_values
    elif needs_cpd:
        cpd = model.predict_cpd(x_i)
        tau = rng.uniform(0, 1)
        Gamma = cpd.predict_set(tau, epsilon=epsilon) if epsilon is not None else None
        kw["cpd"] = cpd
    else:
        Gamma = model.predict(x_i, **predict_kw)

    if epsilon is not None:
        kw["epsilon"] = epsilon

    return Gamma, kw


def _predict_and_update(model, x_i, y_i, metric, needs_p, needs_cpd, predict_kw, epsilon, rng):
    """Core predict-then-update step shared by progressive_val variants."""
    Gamma, kw = _predict(model, x_i, needs_p, needs_cpd, predict_kw, epsilon, rng)
    metric.update(y=y_i, Gamma=Gamma, **kw)


def _resolve_delay(delay, i, x, y):
    """Return the integer delay for step *i*."""
    if callable(delay):
        return int(delay(i, x, y))
    return int(delay)


def _metric_postfix(metric):
    """Format metric values as a dict for tqdm.set_postfix()."""
    if isinstance(metric, Metrics):
        return {name: f"{value:.4f}" for name, value in metric.get().items()}
    else:
        return {metric.name: f"{metric.get():.4f}"}


def _drain(heap, upto, metric, model, learn, apply_update):
    """Pop heap entries with ``arrival_step <= upto``, updating metric and model.

    Parameters
    ----------
    heap : list
        Min-heap of ``(arrival_step, orig_step, x, y, payload)`` tuples managed
        by :mod:`heapq`.  ``orig_step`` is a unique int so heap ordering never
        falls through to numpy-array comparison.
    upto : int or float
        Drain all entries whose arrival step is ``<= upto``.
    metric : Metric or Metrics
        Updated in place via ``apply_update``.
    model : conformal predictor
        ``learn_one`` is called for each resolved entry (gated by ``learn``).
    learn : bool or callable
        Forwarded to ``_should_learn``.
    apply_update : callable
        ``apply_update(metric, y, payload)`` — applies the stored prediction
        artifacts to the metric.
    """
    while heap and heap[0][0] <= upto:
        _arrival, orig_step, x, y, payload = heapq.heappop(heap)
        apply_update(metric, y, payload)
        if _should_learn(learn, orig_step, x, y):
            model.learn_one(x, y)


def progressive_val(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    epsilon: float | NDArray[np.floating[Any]] | None = None,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    delay: int | Callable[[int, Any, Any], int] = 0,
    print_every: int = 0,
    progress: bool = False,
) -> Metric | Metrics:
    """Run the progressive validation (test-then-train) protocol.

    For each sample: predict, enqueue the label at its arrival step, resolve
    all labels that have arrived, then optionally learn from them.

    This function implements Vovk's **Weak Teacher** and **Lazy Teacher**
    paradigms (*Algorithmic Learning in a Random World*, 2nd ed., §3.3) via a
    priority-queue scheduling architecture:

    - **Weak / slow teacher** — set ``delay > 0`` to model a fixed lag
      :math:`l` (or pass a callable ``(step, x, y) → int`` for dynamic
      latency). Labels are placed in a min-heap keyed on their arrival step
      :math:`\\mathcal{L}(n) = n - l` and resolved when they become
      available, mirroring the ALRW2 teaching-schedule formalism.
    - **Lazy teacher** — yield ``y = None`` in the stream for steps where no
      feedback is available. The predictor still produces a prediction set,
      but that step contributes nothing to the metric or to learning.

    **Validity guarantees** (ALRW2, §3.3). Asymptotic validity is preserved
    for invariant conformal predictors (i.e. predictions are order-independent
    — satisfied by every predictor in this package):

    - *Weak validity* (Thm 3.7 / Cor 3.8): :math:`\\mathrm{Err}_n^\\epsilon/n
      \\to \\epsilon` in probability, provided feedback gaps grow
      sub-exponentially (:math:`\\lim_k n_k / n_{k-1} = 1`).
    - *Strong validity* (Thm 3.9 / Cor 3.10): a.s. convergence under
      :math:`\\sum_k (n_k / n_{k-1} - 1)^2 < \\infty`.
    - *LIL validity* (Thm 3.11): with equally-spaced feedback
      (:math:`n_k = O(k)`), the strongest guarantee holds:
      :math:`|\\mathrm{Err}_n^\\epsilon / n - \\epsilon| = O(\\sqrt{\\ln\\ln n / n})`
      a.s. A fixed lag ``delay=l`` satisfies :math:`n_k = O(k)` and therefore
      enjoys this full LIL guarantee.

    The default ``delay=0`` is the ideal teacher and reproduces the standard
    synchronous test-then-train loop exactly.

    Parameters
    ----------
    model : conformal predictor
        Must implement ``predict(x, epsilon=...)`` and ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array of shape (n_samples, n_features), OR an
        iterable of ``(x, y)`` tuples or ``(x, y, t)`` triples for streaming.
        Pass ``y = None`` in stream items to invoke the Lazy Teacher protocol.
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
        Under delayed feedback the decision uses the original prediction step.
    delay : int or callable, optional
        Label arrival delay in steps (default: ``0`` — ideal teacher).
        An integer sets a fixed lag; a callable ``(step_index, x, y) -> int``
        provides a dynamic delay.  All in-flight labels are flushed in a
        teardown pass after the stream is exhausted.
    print_every : int, optional
        Print metric summary every N steps (keyed on prediction step, not
        arrival step).  Under ``delay > 0`` the printed metric reflects only
        labels that have arrived so far.  0 = no printing.
    progress : bool, optional
        Show a tqdm progress bar (requires ``pip install online-cp[progress]``).

    Returns
    -------
    metric : Metric or Metrics
        The same metric object, updated in place and fully resolved.
    """
    if metric is None:
        from online_cp.metrics import ErrorRate
        metric = ErrorRate()

    needs_p = _needs_p_values(metric)
    needs_cpd = _needs_cpd(metric)
    rng = np.random.default_rng()

    predict_kw = {}
    if epsilon is not None:
        predict_kw["epsilon"] = epsilon

    def _apply_update_cp(metric, y, payload):
        Gamma, kw = payload
        metric.update(y=y, Gamma=Gamma, **kw)

    heap: list = []
    total = len(y) if y is not None and hasattr(y, "__len__") else (len(X) if hasattr(X, "__len__") else None)
    data_iter, pbar = _wrap_progress(_iter_data(X, y), total=total, enabled=progress)
    for i, (x_i, y_i, t_i) in enumerate(data_iter):
        if y_i is None:
            # Lazy Teacher: emit prediction but exclude from metric and learning.
            _predict(model, x_i, needs_p, needs_cpd, predict_kw, epsilon, rng)
        else:
            Gamma, kw = _predict(model, x_i, needs_p, needs_cpd, predict_kw, epsilon, rng)
            arrival = i + _resolve_delay(delay, i, x_i, y_i)
            heapq.heappush(heap, (arrival, i, x_i, y_i, (Gamma, kw)))

        _drain(heap, i, metric, model, learn, _apply_update_cp)
        if pbar is not None:
            pbar.set_postfix(_metric_postfix(metric), refresh=False)

        if print_every > 0 and (i + 1) % print_every == 0:
            print(f"[{i + 1}] {metric}")

    # Teardown: flush all labels still in-flight after the stream is exhausted.
    _drain(heap, float("inf"), metric, model, learn, _apply_update_cp)

    return metric


def iter_progressive_val(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    epsilon: float | NDArray[np.floating[Any]] | None = None,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    delay: int | Callable[[int, Any, Any], int] = 0,
    step: int = 1,
    progress: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Iterate the progressive validation protocol, yielding checkpoints.

    Yields a snapshot of the metric state every ``step`` samples.
    Useful for plotting learning curves.

    Implements the same **Weak Teacher** and **Lazy Teacher** paradigms as
    :func:`progressive_val` (see its docstring for the full validity-guarantee
    discussion, ALRW2 §3.3). Snapshots reflect the metric state at the moment
    they are yielded, i.e. only labels that have already arrived under the
    teaching schedule contribute. A final teardown snapshot is yielded after
    the stream is exhausted if any delayed labels remain, so the last snapshot
    always reflects the fully-resolved metric.

    Parameters
    ----------
    model : conformal predictor
        Must implement ``predict(x, epsilon=...)`` and ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array, or iterable of ``(x, y)`` / ``(x, y, t)``
        tuples. Pass ``y = None`` in stream items for the Lazy Teacher.
    y : array-like, optional
        True labels. Required when X is an array; omitted for streaming.
    epsilon : float, optional
        Significance level.
    metric : Metric or Metrics, optional
        Metric(s) to update. Defaults to ErrorRate.
    learn : bool or callable, optional
        Whether the model learns from each example. If callable, called as
        ``learn(i, x, y) -> bool``.  Under delayed feedback the decision uses
        the original prediction step.
    delay : int or callable, optional
        Label arrival delay in steps (default: ``0`` — ideal teacher).
        See :func:`progressive_val` for full semantics.
    step : int
        Yield every ``step`` samples.
    progress : bool, optional
        Show a tqdm progress bar.

    Yields
    ------
    dict
        Contains ``"step"`` (int, 1-indexed prediction count), ``"t"``
        (timestamp or index), and metric values.
    """
    if metric is None:
        from online_cp.metrics import ErrorRate
        metric = ErrorRate()

    needs_p = _needs_p_values(metric)
    needs_cpd = _needs_cpd(metric)
    rng = np.random.default_rng()

    predict_kw = {}
    if epsilon is not None:
        predict_kw["epsilon"] = epsilon

    def _apply_update_cp(metric, y, payload):
        Gamma, kw = payload
        metric.update(y=y, Gamma=Gamma, **kw)

    heap: list = []
    last_i = -1
    last_t = None
    total = len(y) if y is not None and hasattr(y, "__len__") else (len(X) if hasattr(X, "__len__") else None)
    data_iter, pbar = _wrap_progress(_iter_data(X, y), total=total, enabled=progress)
    for i, (x_i, y_i, t_i) in enumerate(data_iter):
        last_i = i
        last_t = t_i
        if y_i is None:
            # Lazy Teacher: emit prediction but exclude from metric and learning.
            _predict(model, x_i, needs_p, needs_cpd, predict_kw, epsilon, rng)
        else:
            Gamma, kw = _predict(model, x_i, needs_p, needs_cpd, predict_kw, epsilon, rng)
            arrival = i + _resolve_delay(delay, i, x_i, y_i)
            heapq.heappush(heap, (arrival, i, x_i, y_i, (Gamma, kw)))

        _drain(heap, i, metric, model, learn, _apply_update_cp)
        if pbar is not None:
            pbar.set_postfix(_metric_postfix(metric), refresh=False)

        if (i + 1) % step == 0:
            if isinstance(metric, Metrics):
                snapshot = {"step": i + 1, "t": t_i}
                snapshot.update(metric.get())
            else:
                snapshot = {"step": i + 1, "t": t_i, metric.name: metric.get()}
            yield snapshot

    # Teardown: flush in-flight labels and emit a final snapshot if needed.
    if heap:
        _drain(heap, float("inf"), metric, model, learn, _apply_update_cp)
        if isinstance(metric, Metrics):
            snapshot = {"step": last_i + 1, "t": last_t}
            snapshot.update(metric.get())
        else:
            snapshot = {"step": last_i + 1, "t": last_t, metric.name: metric.get()}
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
    delay: int | Callable[[int, Any, Any], int] = 0,
    print_every: int = 0,
    progress: bool = False,
) -> Metric | Metrics:
    """Run progressive validation for Venn predictors.

    Same test-then-train protocol as :func:`progressive_val`, but adapted for
    Venn predictors that return ``VennPrediction`` objects (no epsilon).

    Supports the same **Weak Teacher** and **Lazy Teacher** delay semantics;
    see :func:`progressive_val` for the full validity-guarantee discussion
    (ALRW2 §3.3).

    Parameters
    ----------
    model : Venn predictor
        Must implement ``predict(x)`` returning a ``VennPrediction`` and
        ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array of shape (n_samples, n_features), OR an
        iterable of ``(x, y)`` tuples or ``(x, y, t)`` triples for streaming.
        Pass ``y = None`` in stream items to invoke the Lazy Teacher protocol.
    y : array-like, optional
        True labels. Required when X is an array; omitted for streaming.
    metric : Metric or Metrics, optional
        Metric(s) to update at each step. Defaults to ``BrierScore()``.
    learn : bool or callable, optional
        Whether the model learns from each example (default: True).
        If callable, called as ``learn(i, x, y) -> bool``.
        Under delayed feedback the decision uses the original prediction step.
    delay : int or callable, optional
        Label arrival delay in steps (default: ``0`` — ideal teacher).
        See :func:`progressive_val` for full semantics.
    print_every : int, optional
        Print metric summary every N steps. 0 = no printing.
    progress : bool, optional
        Show a tqdm progress bar.

    Returns
    -------
    metric : Metric or Metrics
        The same metric object, updated in place and fully resolved.
    """
    if metric is None:
        metric = BrierScore()

    def _apply_update_venn(metric, y, payload):
        metric.update(y=y, Gamma=None, venn=payload)

    heap: list = []
    total = len(y) if y is not None and hasattr(y, "__len__") else (len(X) if hasattr(X, "__len__") else None)
    data_iter, pbar = _wrap_progress(_iter_data(X, y), total=total, enabled=progress)
    for i, (x_i, y_i, t_i) in enumerate(data_iter):
        if y_i is None:
            # Lazy Teacher: emit prediction but exclude from metric and learning.
            model.predict(x_i)
        else:
            venn_pred = model.predict(x_i)
            arrival = i + _resolve_delay(delay, i, x_i, y_i)
            heapq.heappush(heap, (arrival, i, x_i, y_i, venn_pred))

        _drain(heap, i, metric, model, learn, _apply_update_venn)
        if pbar is not None:
            pbar.set_postfix(_metric_postfix(metric), refresh=False)

        if print_every > 0 and (i + 1) % print_every == 0:
            print(f"[{i + 1}] {metric}")

    # Teardown: flush all labels still in-flight after the stream is exhausted.
    _drain(heap, float("inf"), metric, model, learn, _apply_update_venn)

    return metric


def iter_progressive_val_venn(
    model: Any,
    X: NDArray[np.floating[Any]] | Iterable,
    y: NDArray[Any] | None = None,
    *,
    metric: Metric | Metrics | None = None,
    learn: bool | Callable[[int, NDArray[np.floating[Any]], Any], bool] = True,
    delay: int | Callable[[int, Any, Any], int] = 0,
    step: int = 1,
    progress: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Iterate progressive validation for Venn predictors, yielding checkpoints.

    Supports the same **Weak Teacher** and **Lazy Teacher** delay semantics as
    :func:`progressive_val`; see its docstring for the full validity-guarantee
    discussion (ALRW2 §3.3). A final teardown snapshot is yielded after the
    stream is exhausted if any delayed labels remain.

    Parameters
    ----------
    model : Venn predictor
        Must implement ``predict(x)`` returning a ``VennPrediction`` and
        ``learn_one(x, y)``.
    X : array-like or iterable
        Feature vectors as array, or iterable of ``(x, y)`` / ``(x, y, t)``
        tuples. Pass ``y = None`` in stream items for the Lazy Teacher.
    y : array-like, optional
        True labels. Required when X is an array; omitted for streaming.
    metric : Metric or Metrics, optional
        Metric(s) to update. Defaults to ``BrierScore()``.
    learn : bool or callable, optional
        Whether the model learns from each example. If callable, called as
        ``learn(i, x, y) -> bool``.  Under delayed feedback the decision uses
        the original prediction step.
    delay : int or callable, optional
        Label arrival delay in steps (default: ``0`` — ideal teacher).
        See :func:`progressive_val` for full semantics.
    step : int
        Yield every ``step`` samples.
    progress : bool, optional
        Show a tqdm progress bar.

    Yields
    ------
    dict
        Contains ``"step"``, ``"t"``, and metric values.
    """
    if metric is None:
        metric = BrierScore()

    def _apply_update_venn(metric, y, payload):
        metric.update(y=y, Gamma=None, venn=payload)

    heap: list = []
    last_i = -1
    last_t = None
    total = len(y) if y is not None and hasattr(y, "__len__") else (len(X) if hasattr(X, "__len__") else None)
    data_iter, pbar = _wrap_progress(_iter_data(X, y), total=total, enabled=progress)
    for i, (x_i, y_i, t_i) in enumerate(data_iter):
        last_i = i
        last_t = t_i
        if y_i is None:
            # Lazy Teacher: emit prediction but exclude from metric and learning.
            model.predict(x_i)
        else:
            venn_pred = model.predict(x_i)
            arrival = i + _resolve_delay(delay, i, x_i, y_i)
            heapq.heappush(heap, (arrival, i, x_i, y_i, venn_pred))

        _drain(heap, i, metric, model, learn, _apply_update_venn)
        if pbar is not None:
            pbar.set_postfix(_metric_postfix(metric), refresh=False)

        if (i + 1) % step == 0:
            if isinstance(metric, Metrics):
                snapshot = {"step": i + 1, "t": t_i}
                snapshot.update(metric.get())
            else:
                snapshot = {"step": i + 1, "t": t_i, metric.name: metric.get()}
            yield snapshot

    # Teardown: flush in-flight labels and emit a final snapshot if needed.
    if heap:
        _drain(heap, float("inf"), metric, model, learn, _apply_update_venn)
        if isinstance(metric, Metrics):
            snapshot = {"step": last_i + 1, "t": last_t}
            snapshot.update(metric.get())
        else:
            snapshot = {"step": last_i + 1, "t": last_t, metric.name: metric.get()}
        yield snapshot
