"""Evaluation protocol for online conformal predictors.

Provides ``progressive_val`` — the standard test-then-train loop — as a
standalone function that decouples evaluation logic from model classes.

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

from online_cp.metrics import Metric, Metrics, ObservedFuzziness, CRPS

__all__ = [
    "progressive_val",
    "iter_progressive_val",
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


def progressive_val(model, X, y, *, epsilon=None, metric=None, print_every=0):
    """Run the progressive validation (test-then-train) protocol.

    For each sample: predict, update metric, then learn.

    Parameters
    ----------
    model : conformal predictor
        Must implement ``predict(x, epsilon=...)`` and ``learn_one(x, y)``.
        If metrics require p-values, must support
        ``predict(x, epsilon=..., return_p_values=True)``.
    X : array-like of shape (n_samples, n_features)
        Feature vectors.
    y : array-like of shape (n_samples,)
        True labels / responses.
    epsilon : float, optional
        Significance level. If None, uses model default.
    metric : Metric or Metrics
        Metric(s) to update at each step.
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

    for i, (x_i, y_i) in enumerate(zip(X, y)):
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

        # Learn
        model.learn_one(x_i, y_i)

        # Progress printing
        if print_every > 0 and (i + 1) % print_every == 0:
            print(f"[{i + 1}] {metric}")

    return metric


def iter_progressive_val(model, X, y, *, epsilon=None, metric=None, step=1):
    """Iterate the progressive validation protocol, yielding checkpoints.

    Yields a snapshot of the metric state every ``step`` samples.
    Useful for plotting learning curves.

    Parameters
    ----------
    model : conformal predictor
        Must implement ``predict(x, epsilon=...)`` and ``learn_one(x, y)``.
    X : array-like of shape (n_samples, n_features)
        Feature vectors.
    y : array-like of shape (n_samples,)
        True labels / responses.
    epsilon : float, optional
        Significance level.
    metric : Metric or Metrics
        Metric(s) to update.
    step : int
        Yield every ``step`` samples.

    Yields
    ------
    dict
        ``{"step": i, **metric.get()}`` if Metrics, or
        ``{"step": i, metric.name: metric.get()}`` if single Metric.
    """
    if metric is None:
        from online_cp.metrics import ErrorRate
        metric = ErrorRate()

    needs_p = _needs_p_values(metric)
    needs_cpd = _needs_cpd(metric)

    predict_kw = {}
    if epsilon is not None:
        predict_kw["epsilon"] = epsilon

    for i, (x_i, y_i) in enumerate(zip(X, y)):
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
        model.learn_one(x_i, y_i)

        if (i + 1) % step == 0:
            if isinstance(metric, Metrics):
                snapshot = {"step": i + 1}
                snapshot.update(metric.get())
            else:
                snapshot = {"step": i + 1, metric.name: metric.get()}
            yield snapshot
