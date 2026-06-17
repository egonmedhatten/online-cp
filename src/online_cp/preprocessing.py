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

import numpy as np
from numpy.typing import NDArray

from online_cp._serialization import SerializableMixin
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


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _sign_flip_components(components: NDArray) -> NDArray:
    """Enforce a deterministic sign convention for eigenvectors.

    For each row (component), flip the sign so that the element with the
    largest absolute value is positive.  This makes eigenvectors unique
    (up to zero-measure ties) and permutation-invariant.

    Parameters
    ----------
    components : ndarray of shape (k, d)

    Returns
    -------
    ndarray of shape (k, d)
    """
    max_abs_idx = np.argmax(np.abs(components), axis=1)  # (k,)
    signs = np.sign(components[np.arange(components.shape[0]), max_abs_idx])  # (k,)
    signs[signs == 0] = 1.0  # guard: all-zero row (degenerate)
    return components * signs[:, np.newaxis]


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class PCA(Transformer, SerializableMixin):
    """Rotate features into their principal-component basis.

    Parameters are computed once from the initial training batch (via
    :meth:`fit`) and held constant thereafter (``mode="frozen"``), or
    recomputed at each prediction from the augmented bag (``mode="bag"``).

    ``mode="frozen"`` preserves *training-conditional conformal validity*
    (parameters are a fixed function of the training data).  ``mode="bag"``
    recomputes the rotation from the label-free augmented bag ``[X_train,
    x_test]`` at each prediction, preserving *exact finite-sample conformal
    validity* at the cost of O(n·d²+d³) per predict.

    The transformation is the standard PCA projection: subtract the training
    mean, then project onto the top-*k* eigenvectors of the (unbiased) sample
    covariance matrix.  Eigenvectors are ordered by descending explained
    variance and given a deterministic sign (largest-magnitude element
    positive).

    **Intended use-case**: axis-aligning features before
    :class:`~online_cp.mondrian.MondrianConformalRegressor` /
    :class:`~online_cp.mondrian.MondrianConformalClassifier`.  Mondrian
    methods partition the feature space by hyperplanes; after PCA rotation the
    axes align with the directions of greatest variance, yielding tighter and
    more balanced partitions.

    Parameters
    ----------
    n_components : int or None, default None
        Number of principal components to retain.  ``None`` keeps all
        components (``min(n-1, d)`` after fitting *n* samples with *d*
        features).
    mode : {"frozen", "bag"}, default "frozen"
        See module docstring for the two operation modes.

    Attributes
    ----------
    n_ : int or None
        Number of samples seen during :meth:`fit`.
    mean_ : ndarray of shape (d,) or None
        Per-feature training mean.
    components_ : ndarray of shape (k, d) or None
        Principal components as row vectors (sorted by descending variance).
    singular_values_ : ndarray of shape (k,) or None
        Square roots of the retained eigenvalues (≈ singular values of the
        centred data matrix, up to the ``n-1`` denominator).

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import PCA, Pipeline, ConformalRidgeRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(loc=[0.0, 0.0], scale=[10.0, 0.1], size=(60, 2))
    >>> y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=60)
    >>> pipe = PCA() | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
    >>> pipe.learn_initial_training_set(X, y)
    >>> interval = pipe.predict(X[0], epsilon=0.1)
    """

    mode: str = "frozen"

    _SAVE_PARAMS: tuple[str, ...] = ("n_components", "mode")
    _SAVE_STATE: tuple[str, ...] = ("n_", "mean_", "components_", "singular_values_")

    def __init__(
        self,
        n_components: int | None = None,
        mode: str = "frozen",
    ) -> None:
        if mode not in ("frozen", "bag"):
            raise ValueError(
                f"PCA mode must be 'frozen' or 'bag', got {mode!r}."
            )
        if n_components is not None and n_components < 1:
            raise ValueError(
                f"n_components must be a positive integer or None, got {n_components!r}."
            )
        self.mode = mode
        self.n_components = n_components
        self.n_: int | None = None
        self.mean_: NDArray | None = None
        self.components_: NDArray | None = None
        self.singular_values_: NDArray | None = None

    def fit(self, X: NDArray) -> None:
        """Compute principal components from *X*.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training batch.  Requires ``n ≥ 2``.
        """
        n, d = X.shape
        if n < 2:
            raise ValueError(
                f"PCA requires at least 2 samples to fit, got {n}."
            )
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = X_c.T @ X_c / (n - 1)  # (d, d) unbiased sample covariance

        vals, vecs = np.linalg.eigh(cov)  # ascending order, real symmetric
        idx = np.argsort(vals)[::-1]      # descending variance order

        k_max = min(n - 1, d)
        k = min(self.n_components, k_max) if self.n_components is not None else k_max

        components = vecs[:, idx[:k]].T   # (k, d)
        self.components_ = _sign_flip_components(components)
        self.singular_values_ = np.sqrt(np.maximum(vals[idx[:k]], 0.0))
        self.n_ = n

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError(
                "PCA has not been fitted yet. "
                "Call Pipeline.learn_initial_training_set() before predict()."
            )

    def transform(self, X: NDArray) -> NDArray:
        """Project *X* onto the principal-component basis.

        Parameters
        ----------
        X : ndarray of shape (n, d)

        Returns
        -------
        ndarray of shape (n, k)
        """
        self._check_fitted()
        return (X - self.mean_) @ self.components_.T

    def transform_one(self, x: NDArray) -> NDArray:
        """Project a single feature vector *x*.

        Parameters
        ----------
        x : ndarray of shape (d,)

        Returns
        -------
        ndarray of shape (k,)
        """
        self._check_fitted()
        return (x - self.mean_) @ self.components_.T

    def __repr__(self) -> str:
        return f"PCA(n_components={self.n_components!r}, mode={self.mode!r})"


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

class SVD(Transformer, SerializableMixin):
    """Project features onto the top-*k* right singular vectors of *X*.

    Computes the eigen-decomposition of the (optionally centred) Gram matrix
    ``X^T X`` and projects data onto the leading eigenvectors.  This is
    equivalent to truncated SVD on the data matrix and is the standard
    approach for dimensionality reduction and multicollinearity removal before
    ridge-based methods.

    When ``center=True`` the Gram matrix is built from the mean-centred data
    ``X - mean(X)``, making this *identical* to :class:`PCA` in the projected
    subspace.  Use ``center=False`` when the data is already mean-zero or when
    you want the uncentred factorisation (e.g. for non-negative data).

    **Intended use-case**: dimensionality reduction and multicollinearity
    removal before :class:`~online_cp.cps.RidgePredictionMachine` and
    :class:`~online_cp.regressors.ConformalRidgeRegressor`.  Reducing *d* to
    *k < d* speeds up the O(d³) matrix inversion and stabilises it when
    features are near-collinear.

    Parameters
    ----------
    n_components : int or None, default None
        Number of right singular vectors to retain.  ``None`` keeps all
        (``min(n-1, d)`` when ``center=True``, or ``min(n, d)`` when
        ``center=False``).
    mode : {"frozen", "bag"}, default "frozen"
        See module docstring for the two operation modes.
    center : bool, default True
        If ``True``, subtract the training mean before computing the Gram
        matrix (uses unbiased ``n-1`` denominator).  If ``False``, use the raw
        data (uses ``n`` denominator).

    Attributes
    ----------
    n_ : int or None
        Number of samples seen during :meth:`fit`.
    mean_ : ndarray of shape (d,) or None
        Per-feature training mean.  ``None`` when ``center=False``.
    components_ : ndarray of shape (k, d) or None
        Right singular vectors as row vectors (sorted by descending singular
        value).
    singular_values_ : ndarray of shape (k,) or None
        Retained singular values (square roots of the retained eigenvalues of
        the Gram matrix).

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import SVD, Pipeline, ConformalRidgeRegressor
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(50, 5))
    >>> X[:, 2] = X[:, 0] + 0.01 * rng.normal(size=50)  # near-collinear
    >>> y = X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=50)
    >>> pipe = SVD(n_components=3) | ConformalRidgeRegressor(a=1e-3, epsilon=0.1)
    >>> pipe.learn_initial_training_set(X, y)
    >>> interval = pipe.predict(X[0], epsilon=0.1)
    """

    mode: str = "frozen"

    _SAVE_PARAMS: tuple[str, ...] = ("n_components", "mode", "center")
    _SAVE_STATE: tuple[str, ...] = ("n_", "mean_", "components_", "singular_values_")

    def __init__(
        self,
        n_components: int | None = None,
        mode: str = "frozen",
        center: bool = True,
    ) -> None:
        if mode not in ("frozen", "bag"):
            raise ValueError(
                f"SVD mode must be 'frozen' or 'bag', got {mode!r}."
            )
        if n_components is not None and n_components < 1:
            raise ValueError(
                f"n_components must be a positive integer or None, got {n_components!r}."
            )
        self.mode = mode
        self.n_components = n_components
        self.center = center
        self.n_: int | None = None
        self.mean_: NDArray | None = None
        self.components_: NDArray | None = None
        self.singular_values_: NDArray | None = None

    def fit(self, X: NDArray) -> None:
        """Compute right singular vectors from *X*.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training batch.  Requires ``n ≥ 2``.
        """
        n, d = X.shape
        if n < 2:
            raise ValueError(
                f"SVD requires at least 2 samples to fit, got {n}."
            )
        if self.center:
            self.mean_ = X.mean(axis=0)
            X_c = X - self.mean_
            gram = X_c.T @ X_c / (n - 1)  # unbiased
        else:
            self.mean_ = None
            X_c = X
            gram = X_c.T @ X_c / n        # biased (n denominator)

        vals, vecs = np.linalg.eigh(gram)  # ascending order
        idx = np.argsort(vals)[::-1]       # descending singular-value order

        k_max = min(n - 1, d) if self.center else min(n, d)
        k = min(self.n_components, k_max) if self.n_components is not None else k_max

        components = vecs[:, idx[:k]].T    # (k, d)
        self.components_ = _sign_flip_components(components)
        self.singular_values_ = np.sqrt(np.maximum(vals[idx[:k]], 0.0))
        self.n_ = n

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError(
                "SVD has not been fitted yet. "
                "Call Pipeline.learn_initial_training_set() before predict()."
            )

    def transform(self, X: NDArray) -> NDArray:
        """Project *X* onto the retained right singular vectors.

        Parameters
        ----------
        X : ndarray of shape (n, d)

        Returns
        -------
        ndarray of shape (n, k)
        """
        self._check_fitted()
        if self.center:
            return (X - self.mean_) @ self.components_.T
        return X @ self.components_.T

    def transform_one(self, x: NDArray) -> NDArray:
        """Project a single feature vector *x*.

        Parameters
        ----------
        x : ndarray of shape (d,)

        Returns
        -------
        ndarray of shape (k,)
        """
        self._check_fitted()
        if self.center:
            return (x - self.mean_) @ self.components_.T
        return x @ self.components_.T

    def __repr__(self) -> str:
        return (
            f"SVD(n_components={self.n_components!r}, "
            f"mode={self.mode!r}, center={self.center!r})"
        )
