"""Dataset registry for model benchmarking.

Provides standardized access to sklearn built-in datasets with fixed-seed
shuffling for reproducible benchmarks.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_digits,
)


def _load_and_shuffle(loader, name, task, seed=42, **kwargs):
    data = loader()
    X, y = data.data, data.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    meta = {
        "name": name,
        "task": task,
        "n": X.shape[0],
        "d": X.shape[1],
    }
    if task == "classification":
        meta["n_classes"] = len(np.unique(y))
    return {"X": X, "y": y, "metadata": meta}


def load_datasets(task="all"):
    """Load benchmark datasets.

    Parameters
    ----------
    task : str
        One of "classification", "regression", or "all".

    Returns
    -------
    list[dict]
        Each dict has keys: "X", "y", "metadata".
    """
    datasets = []

    if task in ("classification", "all"):
        datasets.append(_load_and_shuffle(load_iris, "iris", "classification"))
        datasets.append(_load_and_shuffle(load_breast_cancer, "breast_cancer", "classification"))
        datasets.append(_load_and_shuffle(load_digits, "digits", "classification"))

    if task in ("regression", "all"):
        datasets.append(_load_and_shuffle(load_diabetes, "diabetes", "regression"))
        # california_housing requires network fetch on first call; use a
        # synthetic alternative that's always available.
        datasets.append(_make_friedman1())

    return datasets


def _make_friedman1(n=500, seed=42):
    """Friedman #1 synthetic regression (sklearn.datasets.make_friedman1)."""
    from sklearn.datasets import make_friedman1

    X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    return {
        "X": X,
        "y": y,
        "metadata": {
            "name": "friedman1",
            "task": "regression",
            "n": X.shape[0],
            "d": X.shape[1],
        },
    }
