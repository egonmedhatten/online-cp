"""Core benchmark runner.

Runs progressive validation for a single (model, dataset) pair and
returns a results dict with metric values and timing.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

from online_cp.evaluate import progressive_val, progressive_val_venn
from online_cp.metrics import (
    BrierScore,
    CRPS,
    ErrorRate,
    IntervalWidth,
    LogLoss,
    Metrics,
    ObservedExcess,
    ObservedFuzziness,
    SetSize,
    Width,
    WinklerScore,
)


def _metrics_for_task(task: str) -> Metrics:
    """Build the appropriate metric composition for a model task type."""
    if task == "regression":
        return ErrorRate() + IntervalWidth() + WinklerScore()
    elif task == "classification":
        return ErrorRate() + SetSize() + ObservedExcess() + ObservedFuzziness()
    elif task == "cps":
        return ErrorRate() + IntervalWidth() + CRPS()
    elif task == "venn":
        return BrierScore() + LogLoss() + Width()
    else:
        raise ValueError(f"Unknown task: {task}")


def run_benchmark(
    config: dict[str, Any],
    dataset: dict[str, Any],
    *,
    epsilon: float = 0.1,
    train_fraction: float = 0.3,
    cap: int | None = None,
) -> dict[str, Any] | None:
    """Run a single benchmark: progressive_val on (model, dataset).

    Parameters
    ----------
    config : dict
        Model configuration from configs.get_configs().
    dataset : dict
        Dataset from datasets.load_datasets().
    epsilon : float
        Significance level (for regressors/classifiers/CPS).
    train_fraction : float
        Fraction of data used for learn_initial_training_set.
    cap : int or None
        If set, limit the test set to this many examples.

    Returns
    -------
    dict or None
        Results dict with metric values and timing, or None if skipped.
    """
    task = config["task"]
    ds_task = dataset["metadata"]["task"]

    # Task compatibility check
    if task in ("regression", "cps") and ds_task != "regression":
        return None
    if task in ("classification", "venn") and ds_task != "classification":
        return None

    # Binary-only check
    if config.get("binary_only") and dataset["metadata"].get("n_classes", 2) > 2:
        return None

    X = dataset["X"]
    y = dataset["y"]
    n = len(y)
    n_train = max(10, int(n * train_fraction))

    if cap is not None:
        n_test = min(cap, n - n_train)
    else:
        n_test = n - n_train

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train : n_train + n_test], y[n_train : n_train + n_test]

    # Instantiate model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = config["factory"]()

    # Set label_space for classifiers if needed
    if task == "classification" and hasattr(model, "label_space"):
        if model.label_space is None:
            model.label_space = np.unique(y)

    # Learn initial training set
    model.learn_initial_training_set(X_train, y_train)

    # Build metrics
    metric = _metrics_for_task(task)

    # Run benchmark with timing
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "venn":
            # Handle both predict() and predict_one() interfaces
            predict_fn = getattr(model, "predict", None) or getattr(model, "predict_one")
            for x_i, y_i in zip(X_test, y_test):
                venn_pred = predict_fn(x_i)
                metric.update(y=y_i, Gamma=None, venn=venn_pred)
                model.learn_one(x_i, y_i)
        elif task == "cps":
            # CPS needs tau for predict_set; use custom loop
            rng = np.random.default_rng(42)
            for x_i, y_i in zip(X_test, y_test):
                cpd = model.predict_cpd(x_i)
                tau = rng.uniform(0, 1)
                Gamma = cpd.predict_set(tau, epsilon=epsilon)
                metric.update(y=y_i, Gamma=Gamma, cpd=cpd, epsilon=epsilon)
                model.learn_one(x_i, y_i)
        else:
            progressive_val(model, X_test, y_test, epsilon=epsilon, metric=metric)
    elapsed = time.perf_counter() - t0

    # Collect results
    result = {
        "model": config["name"],
        "dataset": dataset["metadata"]["name"],
        "task": task,
        "n_train": n_train,
        "n_test": n_test,
        "time_s": elapsed,
    }

    # Extract metric values
    if isinstance(metric, Metrics):
        for m in metric._metrics:
            result[type(m).__name__] = m.get()
    else:
        result[type(metric).__name__] = metric.get()

    return result
