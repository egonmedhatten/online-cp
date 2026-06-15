"""Conformal-safe preprocessing transformers.

Two operation modes are supported:

- **frozen** (``mode="frozen"``, default) — parameters are computed once from
  the initial training set (via :meth:`~Transformer.fit`) and held constant
  thereafter.  Preserves *training-conditional conformal validity* (ALRW2 §4.7):
  the preprocessing is a fixed function of the training data.
- **bag** (``mode="bag"``) — parameters are recomputed at each prediction from
  the *augmented object bag* ``[X_train, x_test]`` (label-free).  Because the
  fit is symmetric over the full augmented bag, the resulting feature map is
  permutation-equivariant and preserves *exact finite-sample conformal validity*.
  No initial training set is required; the bag grows via :meth:`learn_one`.
  Cost: loses the Sherman-Morrison incremental speedup (O(n·d²+d³) per predict).

The transformers integrate with :class:`~online_cp.pipeline.Pipeline` through
the :class:`~online_cp.pipeline.Transformer` protocol; they should not be used
standalone (call :meth:`fit` before :meth:`transform` / :meth:`transform_one`).

Example
-------
>>> import numpy as np
>>> from online_cp import Pipeline, StandardScaler, ConformalRidgeRegressor
>>> rng = np.random.default_rng(0)
>>> X_tr = rng.normal(loc=[10.0, 0.001], scale=[100.0, 0.001], size=(100, 2))
>>> y_tr = X_tr[:, 0] * 0.01 + X_tr[:, 1] * 1.5 + rng.normal(scale=0.1, size=100)
>>> pipe = StandardScaler() | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
>>> pipe.learn_initial_training_set(X_tr, y_tr)
>>> x_new = rng.normal(loc=[10.0, 0.001], scale=[100.0, 0.001])
>>> interval = pipe.predict(x_new, epsilon=0.1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from online_cp.pipeline import Transformer


class StandardScaler(Transformer):
    """Standardise features to zero mean and unit variance.

    Parameters are computed once from the initial training batch (via
    :meth:`fit`) and held constant thereafter (``mode="frozen"``).  This
    preserves training-conditional conformal validity.

    Zero-variance features are left unchanged (their scale is set to 1 to
    avoid division by zero).

    Parameters
    ----------
    with_mean : bool, default True
        Subtract the training mean from each feature.
    with_std : bool, default True
        Divide each feature by its training standard deviation.

    Attributes
    ----------
    mean_ : ndarray of shape (d,) or None
        Per-feature training mean.  ``None`` before :meth:`fit` is called.
    scale_ : ndarray of shape (d,) or None
        Per-feature training standard deviation (with zero-variance guard).
        ``None`` before :meth:`fit` is called.

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import StandardScaler, Pipeline, ConformalRidgeRegressor
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(loc=5.0, scale=10.0, size=(50, 3))
    >>> y = X[:, 0] + rng.normal(scale=0.5, size=50)
    >>> pipe = StandardScaler() | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    >>> pipe.learn_initial_training_set(X, y)
    >>> interval = pipe.predict(X[0], epsilon=0.1)
    """

    mode: str = "frozen"  # class-level default; overridden by instance in __init__

    def __init__(self, with_mean: bool = True, with_std: bool = True, mode: str = "frozen") -> None:
        if mode not in ("frozen", "bag"):
            raise ValueError(
                f"StandardScaler mode must be 'frozen' or 'bag', got {mode!r}."
            )
        self.mode = mode  # instance attribute
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: NDArray | None = None
        self.scale_: NDArray | None = None

    def fit(self, X: NDArray) -> None:
        """Compute mean and standard deviation from *X*.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training batch.
        """
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            std = X.std(axis=0)
            std[std == 0.0] = 1.0  # guard: zero-variance feature → no scaling
            self.scale_ = std
        else:
            self.scale_ = np.ones(X.shape[1])

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "StandardScaler has not been fitted yet. "
                "Call Pipeline.learn_initial_training_set() before predict()."
            )

    def transform(self, X: NDArray) -> NDArray:
        self._check_fitted()
        return (X - self.mean_) / self.scale_

    def transform_one(self, x: NDArray) -> NDArray:
        self._check_fitted()
        return (x - self.mean_) / self.scale_

    def __repr__(self) -> str:
        return f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std}, mode={self.mode!r})"


class MinMaxScaler(Transformer):
    """Scale features to a fixed range ``[feature_range[0], feature_range[1]]``.

    Parameters are computed once from the initial training batch (via
    :meth:`fit`) and held constant thereafter (``mode="frozen"``).

    Features whose training range is zero (constant column) are mapped to the
    lower bound of ``feature_range`` without raising an error.

    Parameters
    ----------
    feature_range : tuple of (float, float), default (0, 1)
        Target range for the scaled features.

    Attributes
    ----------
    data_min_ : ndarray of shape (d,) or None
        Per-feature minimum over the training set.
    data_range_ : ndarray of shape (d,) or None
        Per-feature ``max - min`` over the training set (zero-range guarded).

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import MinMaxScaler, Pipeline, ConformalRidgeRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(low=-100.0, high=100.0, size=(50, 2))
    >>> y = X[:, 0] * 0.5 + rng.normal(scale=1.0, size=50)
    >>> pipe = MinMaxScaler() | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    >>> pipe.learn_initial_training_set(X, y)
    >>> interval = pipe.predict(X[0], epsilon=0.1)
    """

    mode: str = "frozen"  # class-level default; overridden by instance in __init__

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), mode: str = "frozen") -> None:
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(
                f"feature_range must satisfy lo < hi, got ({lo}, {hi})."
            )
        if mode not in ("frozen", "bag"):
            raise ValueError(
                f"MinMaxScaler mode must be 'frozen' or 'bag', got {mode!r}."
            )
        self.mode = mode  # instance attribute
        self.feature_range = feature_range
        self.data_min_: NDArray | None = None
        self.data_range_: NDArray | None = None

    def fit(self, X: NDArray) -> None:
        """Compute min and range from *X*.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training batch.
        """
        self.data_min_ = X.min(axis=0)
        data_range = X.max(axis=0) - self.data_min_
        data_range[data_range == 0.0] = 1.0  # guard: constant feature
        self.data_range_ = data_range

    def _check_fitted(self) -> None:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError(
                "MinMaxScaler has not been fitted yet. "
                "Call Pipeline.learn_initial_training_set() before predict()."
            )

    def _scale(self, v: NDArray) -> NDArray:
        lo, hi = self.feature_range
        return lo + (v - self.data_min_) / self.data_range_ * (hi - lo)

    def transform(self, X: NDArray) -> NDArray:
        self._check_fitted()
        return self._scale(X)

    def transform_one(self, x: NDArray) -> NDArray:
        self._check_fitted()
        return self._scale(x)

    def __repr__(self) -> str:
        return f"MinMaxScaler(feature_range={self.feature_range!r}, mode={self.mode!r})"
