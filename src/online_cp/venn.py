"""Online Venn predictors.

This module implements Venn predictors that produce calibrated probability
predictions. Currently includes the Venn-Abers predictor (Algorithm 6.1,
ALRW2 §6.4) — the first known Python implementation of the full/transductive
variant.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

__all__ = [
    "VennAbersPredictor",
    "NearestNeighboursVennPredictor",
    "VennPrediction",
    "MulticlassVennPrediction",
    "log_loss_point",
    "brier_point",
]


# ---------------------------------------------------------------------------
# PAVA (Pool Adjacent Violators Algorithm) — O(n) isotonic regression
# ---------------------------------------------------------------------------


@njit(cache=True)
def _pava_inplace(y, w):
    """Weighted pool-adjacent-violators algorithm (in-place).

    Given values y[0..n-1] and weights w[0..n-1] (assumed sorted by some
    external key), computes the isotonic (non-decreasing) regression of y
    with respect to weights w using the PAVA. Modifies y in-place.

    Parameters
    ----------
    y : ndarray, shape (n,), float64
        Values to make isotonic. Modified in-place.
    w : ndarray, shape (n,), float64
        Positive weights for each observation.

    Returns
    -------
    y : ndarray
        The same array, now containing isotonic values.
    """
    n = len(y)
    if n <= 1:
        return y

    # block[i] = (start_index, weighted_sum, weight_sum)
    # We use parallel arrays for numba compatibility
    block_start = np.empty(n, dtype=np.int64)
    block_wsum = np.empty(n, dtype=np.float64)
    block_wweight = np.empty(n, dtype=np.float64)

    # Initialize: each point is its own block
    num_blocks = 0
    for i in range(n):
        block_start[num_blocks] = i
        block_wsum[num_blocks] = w[i] * y[i]
        block_wweight[num_blocks] = w[i]

        # Merge with previous blocks while violating monotonicity
        while num_blocks > 0:
            prev_val = block_wsum[num_blocks - 1] / block_wweight[num_blocks - 1]
            curr_val = block_wsum[num_blocks] / block_wweight[num_blocks]
            if prev_val > curr_val:
                # Merge current into previous
                block_wsum[num_blocks - 1] += block_wsum[num_blocks]
                block_wweight[num_blocks - 1] += block_wweight[num_blocks]
                num_blocks -= 1
            else:
                break
        num_blocks += 1

    # Write back isotonic values
    for b in range(num_blocks):
        val = block_wsum[b] / block_wweight[b]
        start = block_start[b]
        end = block_start[b + 1] if b + 1 < num_blocks else n
        for i in range(start, end):
            y[i] = val

    return y


def _isotonic_calibrate(scores, labels, query_idx):
    """Apply isotonic calibration and return the calibrated value at query_idx.

    Sorts examples by score, applies PAVA to get a non-decreasing mapping
    from scores to calibrated probabilities, and returns the value at the
    position corresponding to query_idx (index into the original arrays).

    Parameters
    ----------
    scores : ndarray, shape (n,)
        Raw scores for all examples (including the query point).
    labels : ndarray, shape (n,)
        Binary labels (0 or 1) for all examples.
    query_idx : int
        Index of the query point in the original (unsorted) arrays.

    Returns
    -------
    calibrated_value : float
        The isotonic-calibrated probability at the query point.
    """
    order = np.argsort(scores, kind="stable")
    y_sorted = labels[order].astype(np.float64).copy()
    w_sorted = np.ones(len(y_sorted), dtype=np.float64)

    _pava_inplace(y_sorted, w_sorted)

    # Find which position in the sorted array corresponds to query_idx
    pos = np.flatnonzero(order == query_idx)[0]
    return y_sorted[pos]


# ---------------------------------------------------------------------------
# VennPrediction output type
# ---------------------------------------------------------------------------


class VennPrediction:
    """Multiprobability prediction from a Venn predictor (ALRW2 §6.2).

    Represents the family P = {P^v : v ∈ Y} of probability distributions
    over the label space Y, one for each hypothesis about the test label.
    Internally stored as a |Y| × |Y| matrix where row v gives P^v.

    For binary classification (|Y| = 2 with labels {0, 1}), the convenient
    ``.p0`` and ``.p1`` properties give the calibrated P(y=1) under each
    hypothesis, compatible with :func:`log_loss_point` and :func:`brier_point`.

    Attributes
    ----------
    probs : ndarray, shape (|Y|, |Y|)
        ``probs[i, j]`` = P^{label_space[i]}(label_space[j]).
    label_space : ndarray
        Sorted array of distinct labels.

    Examples
    --------
    >>> import numpy as np
    >>> pred = VennPrediction.binary(0.2, 0.8)
    >>> pred.p0, pred.p1
    (0.2, 0.8)
    >>> pred.probs.shape
    (2, 2)
    >>> pred.point
    array([0.5, 0.5])
    """

    def __init__(self, probs: NDArray[np.floating[Any]], label_space: NDArray[Any]) -> None:
        self.probs = np.asarray(probs, dtype=np.float64)
        self.label_space = np.asarray(label_space)

    @classmethod
    def binary(cls, p0, p1):
        """Create a binary VennPrediction from p0 and p1.

        Parameters
        ----------
        p0 : float
            P(y=1) under hypothesis y=0.  Must be in [0, 1].
        p1 : float
            P(y=1) under hypothesis y=1.  Must be in [0, 1].

        Raises
        ------
        ValueError
            If p0 or p1 is outside [0, 1].
        """
        if not (0.0 <= p0 <= 1.0):
            raise ValueError(f"p0 must be in [0, 1], got {p0}")
        if not (0.0 <= p1 <= 1.0):
            raise ValueError(f"p1 must be in [0, 1], got {p1}")
        probs = np.array([[1.0 - p0, p0], [1.0 - p1, p1]])
        return cls(probs, np.array([0, 1]))

    @property
    def p0(self):
        """P(y=1) under hypothesis y=0. Only valid for binary (|Y|=2)."""
        if len(self.label_space) != 2:
            raise AttributeError(
                "p0 is only defined for binary predictions (|Y|=2)"
            )
        return float(self.probs[0, 1])

    @property
    def p1(self):
        """P(y=1) under hypothesis y=1. Only valid for binary (|Y|=2)."""
        if len(self.label_space) != 2:
            raise AttributeError(
                "p1 is only defined for binary predictions (|Y|=2)"
            )
        return float(self.probs[1, 1])

    @property
    def point(self):
        """Aggregate point prediction: mean of all hypothesis rows.

        Each row is already a valid probability distribution (sums to 1),
        so their mean also sums to 1 — no renormalization needed.
        """
        return self.probs.mean(axis=0)

    def __repr__(self):
        if len(self.label_space) == 2:
            return f"VennPrediction(p0={self.p0:.4f}, p1={self.p1:.4f})"
        n = len(self.label_space)
        return f"VennPrediction(|Y|={n}, labels={self.label_space.tolist()})"

    def __str__(self):
        if len(self.label_space) == 2:
            return f"(p0={self.p0:.4f}, p1={self.p1:.4f})"
        return f"VennPrediction\n{self.probs}"


# Backward-compatible alias
MulticlassVennPrediction = VennPrediction


# ---------------------------------------------------------------------------
# Aggregation utilities (ALRW2 §6.4)
# ---------------------------------------------------------------------------


def log_loss_point(p0, p1):
    """Merge a multiprobability pair into a single probability minimising log loss.

    Given the Venn-Abers output (p0, p1), returns the probability p that
    minimises the expected logarithmic loss. This is the formula from
    ALRW2 §6.4: p = p1 / (1 - p0 + p1).

    Parameters
    ----------
    p0 : float
        Calibrated P(y=1) under hypothesis y=0.
    p1 : float
        Calibrated P(y=1) under hypothesis y=1.

    Returns
    -------
    float
        Single probability minimising expected log loss.

    Raises
    ------
    ValueError
        If p0 or p1 is outside [0, 1].

    Examples
    --------
    >>> log_loss_point(0.2, 0.8)
    0.5
    >>> log_loss_point(0.0, 1.0)
    0.5
    """
    if not (0.0 <= p0 <= 1.0):
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if not (0.0 <= p1 <= 1.0):
        raise ValueError(f"p1 must be in [0, 1], got {p1}")
    denom = 1 - p0 + p1
    if denom == 0:
        return 0.5
    return p1 / denom


def brier_point(p0, p1):
    """Merge a multiprobability pair into a single probability minimising Brier loss.

    Given the Venn-Abers output (p0, p1), returns the probability p that
    minimises the expected Brier (squared) loss: p = (p0 + p1) / 2.

    Parameters
    ----------
    p0 : float
        Calibrated P(y=1) under hypothesis y=0.
    p1 : float
        Calibrated P(y=1) under hypothesis y=1.

    Returns
    -------
    float
        Single probability minimising expected Brier loss.

    Examples
    --------
    >>> brier_point(0.2, 0.8)
    0.5
    >>> brier_point(0.0, 1.0)
    0.5
    """
    return (p0 + p1) / 2


# ---------------------------------------------------------------------------
# VennAbersPredictor
# ---------------------------------------------------------------------------


class VennAbersPredictor:
    """
    Full online Venn-Abers predictor (Algorithm 6.1, ALRW2 §6.4).

    Produces calibrated probability predictions for binary classification.
    This is the full/transductive variant — no data splitting. The scorer
    is retrained on the augmented dataset (training + hypothesized test label)
    for each prediction.

    Supports ridge regression, kernel ridge regression, k-NN, and SVM
    scoring functions.

    >>> import numpy as np
    >>> np.random.seed(42)
    >>> N = 50
    >>> X = np.random.randn(N, 2)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> vap = VennAbersPredictor(scorer="ridge", a=1.0)
    >>> vap.learn_initial_training_set(X[:30], y[:30])
    >>> pred = vap.predict(X[30])
    >>> bool(0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1)
    True
    """

    def __init__(
        self,
        scorer="ridge",
        a=0.0,
        k=1,
        distance="euclidean",
        distance_func=None,
        aggregation="mean",
        kernel="rbf",
        C=1.0,
        sigma=1.0,
        degree=3,
        coef0=0.0,
        smo_tol=1e-3,
        smo_max_iter=5000,
        label_space=None,
    ):
        """
        Parameters
        ----------
        scorer : {'ridge', 'kernel_ridge', 'knn', 'svm'}
            Scoring function to use.
        a : float
            Ridge regularisation parameter (for scorer='ridge').
        k : int
            Number of nearest neighbours (for scorer='knn').
        distance : str
            Distance metric for k-NN (passed to scipy).
        distance_func : callable, optional
            Custom distance function for k-NN.
        aggregation : {'mean', 'median'}
            Aggregation for k-NN distances.
        kernel : str or Kernel instance or callable
            Kernel for kernel-ridge and SVM scorers. Strings: 'rbf',
            'linear', 'poly'.
        C : float
            SVM regularisation parameter (upper bound on alpha).
        sigma : float
            Bandwidth for RBF kernel.
        degree : int
            Degree for polynomial kernel.
        coef0 : float
            Constant term for polynomial kernel.
        smo_tol : float
            KKT violation tolerance for SMO solver.
        smo_max_iter : int
            Maximum iterations for SMO solver.
        label_space : array-like or None
            Explicit set of possible labels. If None, inferred from data.
            When provided, the label space is fixed and labels outside it
            are rejected. When None (default), binary {0,1} is inferred
            for backward compatibility unless multiclass labels appear.
        """
        if scorer not in ("ridge", "kernel_ridge", "knn", "svm"):
            raise ValueError(
                f"scorer must be 'ridge', 'kernel_ridge', 'knn', or 'svm', got '{scorer}'"
            )
        if aggregation not in ("mean", "median"):
            raise ValueError(f"aggregation must be 'mean' or 'median', got '{aggregation}'")

        self.scorer = scorer
        self.a = a
        self.k = k
        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = "custom"
        self.aggregation = aggregation

        # SVM parameters
        self.C = C
        self.sigma = sigma
        self.degree = degree
        self.coef0 = coef0
        self.smo_tol = smo_tol
        self.smo_max_iter = smo_max_iter

        # Label-space policy
        self._label_space_fixed = label_space is not None
        self.label_space = (
            np.asarray(sorted(label_space), dtype=int)
            if label_space is not None
            else None
        )

        self.X = None
        self.y = None

        # Ridge state
        self.XTXinv = None

        # k-NN state
        self.D = None

        # Kernel/SVM state
        self.K = None  # Gram matrix
        self.Ka_inv = None  # (K + aI)^-1 for kernel-ridge
        if scorer in ("svm", "kernel_ridge"):
            self._kernel = self._resolve_kernel(kernel)

    def _standard_distance_func(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            return squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            return cdist(X, y, metric=self.distance)

    def _resolve_kernel(self, kernel):
        """Resolve kernel specification to a callable."""
        from online_cp.kernels import GaussianKernel, Kernel, LinearKernel, PolynomialKernel

        if isinstance(kernel, Kernel):
            return kernel
        elif isinstance(kernel, str):
            if kernel == "rbf":
                return GaussianKernel(sigma=self.sigma)
            elif kernel == "linear":
                return LinearKernel()
            elif kernel == "poly":
                return PolynomialKernel(d=self.degree, c=self.coef0)
            else:
                raise ValueError(f"Unknown kernel string: '{kernel}'. Use 'rbf', 'linear', or 'poly'.")
        elif callable(kernel):
            return kernel
        else:
            raise TypeError(f"kernel must be a string, Kernel instance, or callable, got {type(kernel)}")

    def _augment_kernel_state(self, x):
        """Build augmented K and (K+aI)^-1 for one new point with robust fallback."""
        x = np.asarray(x).ravel()
        n = self.K.shape[0]

        k_row = np.atleast_1d(self._kernel(self.X, x))
        kappa = float(self._kernel(x.reshape(1, -1))[0, 0])
        K_aug = np.empty((n + 1, n + 1), dtype=np.float64)
        K_aug[:n, :n] = self.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        # Block inverse update for (K_aug + aI)^-1
        b = k_row
        d = kappa + self.a
        A_inv_b = self.Ka_inv @ b
        s = float(d - b.T @ A_inv_b)

        # Use a relative tolerance and finite checks; fallback to recomputation when unstable
        scale = 1.0 + abs(d) + np.linalg.norm(b) * max(np.linalg.norm(A_inv_b), 1.0)
        tol = 1e-12 * scale
        use_fallback = (not np.isfinite(s)) or (s <= tol)

        if use_fallback:
            Ka_aug = K_aug + self.a * np.identity(n + 1)
            try:
                Ka_inv_aug = np.linalg.inv(Ka_aug)
            except np.linalg.LinAlgError:
                Ka_inv_aug = np.linalg.pinv(Ka_aug)
        else:
            Ka_inv_aug = np.empty((n + 1, n + 1), dtype=np.float64)
            Ka_inv_aug[:n, :n] = self.Ka_inv + np.outer(A_inv_b, A_inv_b) / s
            Ka_inv_aug[:n, n] = -A_inv_b / s
            Ka_inv_aug[n, :n] = -A_inv_b / s
            Ka_inv_aug[n, n] = 1.0 / s

        # Keep numerical symmetry of the inverse matrix
        Ka_inv_aug = 0.5 * (Ka_inv_aug + Ka_inv_aug.T)
        return K_aug, Ka_inv_aug

    def learn_initial_training_set(self, X, y):
        """Batch-initialize with training data.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Training feature vectors.
        y : ndarray, shape (n,)
            Integer labels. Binary {0, 1} or multiclass.
        """
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=int)

        # Label-space policy
        if self._label_space_fixed:
            unknown = set(np.unique(y)) - set(self.label_space)
            if unknown:
                raise ValueError(
                    f"Labels {sorted(unknown)} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.unique(y)
        else:
            self.label_space = np.sort(
                np.unique(np.concatenate([self.label_space, np.unique(y)]))
            )

        self.X = X
        self.y = y

        if self.scorer == "ridge":
            Id = np.identity(X.shape[1])
            try:
                self.XTXinv = np.linalg.inv(X.T @ X + self.a * Id)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise."
                ) from None
        elif self.scorer == "kernel_ridge":
            self.K = self._kernel(X)
            Ka = self.K + self.a * np.identity(self.K.shape[0])
            try:
                self.Ka_inv = np.linalg.inv(Ka)
            except np.linalg.LinAlgError:
                self.Ka_inv = np.linalg.pinv(Ka)
        elif self.scorer == "knn":
            self.D = self.distance_func(X)
        elif self.scorer == "svm":
            self.K = self._kernel(X)

    def learn_one(self, x: NDArray[np.floating[Any]], y: int, precomputed: dict[str, Any] | None = None) -> None:
        """Incrementally add one observation after the true label is revealed.

        Parameters
        ----------
        x : array-like, shape (d,)
            Feature vector.
        y : int
            True label.
        precomputed : dict, optional
            Cached state from predict(return_update=True).
            For ridge: {'XTXinv': ...}
            For kernel_ridge: {'K': ..., 'Ka_inv': ...}
            For knn: {'D': ...}
        """
        x = np.asarray(x).ravel()
        y = int(y)

        # Label-space policy
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(
                    f"Label {y} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.array([y], dtype=int)
        elif y not in self.label_space:
            self.label_space = np.sort(np.append(self.label_space, y))

        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=int)
            if self.scorer == "ridge":
                Id = np.identity(self.X.shape[1])
                try:
                    self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * Id)
                except np.linalg.LinAlgError:
                    raise ValueError(
                        "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise."
                    ) from None
            elif self.scorer == "kernel_ridge":
                self.K = self._kernel(self.X)
                Ka = self.K + self.a * np.identity(1)
                try:
                    self.Ka_inv = np.linalg.inv(Ka)
                except np.linalg.LinAlgError:
                    self.Ka_inv = np.linalg.pinv(Ka)
            elif self.scorer == "knn":
                self.D = self.distance_func(self.X)
            elif self.scorer == "svm":
                self.K = self._kernel(self.X)
        else:
            if self.scorer == "ridge":
                if precomputed is not None and "XTXinv" in precomputed:
                    self.XTXinv = precomputed["XTXinv"]
                else:
                    self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (
                        1 + x.T @ self.XTXinv @ x
                    )
                self.X = np.vstack([self.X, x.reshape(1, -1)])
                self.y = np.append(self.y, y)

            elif self.scorer == "knn":
                if precomputed is not None and "D" in precomputed:
                    self.D = precomputed["D"]
                else:
                    d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
                    n = self.D.shape[0]
                    D_new = np.empty((n + 1, n + 1), dtype=np.float64)
                    D_new[:n, :n] = self.D
                    D_new[:n, n] = d
                    D_new[n, :n] = d
                    D_new[n, n] = 0.0
                    self.D = D_new
                self.X = np.vstack([self.X, x.reshape(1, -1)])
                self.y = np.append(self.y, y)

            elif self.scorer == "kernel_ridge":
                if precomputed is not None and "K" in precomputed and "Ka_inv" in precomputed:
                    self.K = precomputed["K"]
                    self.Ka_inv = precomputed["Ka_inv"]
                else:
                    self.K, self.Ka_inv = self._augment_kernel_state(x)

                self.X = np.vstack([self.X, x.reshape(1, -1)])
                self.y = np.append(self.y, y)

            elif self.scorer == "svm":
                if precomputed is not None and "K" in precomputed:
                    self.K = precomputed["K"]
                else:
                    k_row = np.atleast_1d(self._kernel(self.X, x))
                    kappa = float(self._kernel(x.reshape(1, -1))[0, 0])
                    n = self.K.shape[0]
                    K_new = np.empty((n + 1, n + 1), dtype=np.float64)
                    K_new[:n, :n] = self.K
                    K_new[:n, n] = k_row
                    K_new[n, :n] = k_row
                    K_new[n, n] = kappa
                    self.K = K_new
                self.X = np.vstack([self.X, x.reshape(1, -1)])
                self.y = np.append(self.y, y)

    def predict(self, x: NDArray[np.floating[Any]], return_update: bool = False) -> VennPrediction | tuple[VennPrediction, dict[str, Any]]:
        """Produce a Venn-Abers multi-probability prediction.

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.
        return_update : bool
            If True, return precomputed state for efficient learn_one.

        Returns
        -------
        prediction : VennPrediction
            Binary: contains p0, p1. Multiclass: |Y|×|Y| probs matrix.
        precomputed : dict, optional
            Returned if return_update=True.
        """
        x = np.asarray(x).ravel()

        if self.X is None or len(self.y) == 0:
            if self.label_space is not None and len(self.label_space) > 2:
                n_labels = len(self.label_space)
                uniform = np.full((n_labels, n_labels), 1.0 / n_labels)
                pred = VennPrediction(uniform, self.label_space)
            else:
                pred = VennPrediction.binary(0.5, 0.5)
            if return_update:
                return pred, {}
            return pred

        # Dispatch: binary vs multiclass
        # Binary path requires labels {0, 1}; other 2-class problems use multiclass OVR.
        # Single-class label_space also uses binary path (hypothesizes both 0 and 1).
        _is_binary = (
            self.label_space is not None
            and len(self.label_space) <= 2
            and (len(self.label_space) <= 1 or (self.label_space[0] == 0 and self.label_space[1] == 1))
        )
        if not _is_binary:
            if self.scorer == "ridge":
                return self._predict_multiclass_ridge(x, return_update)
            elif self.scorer == "kernel_ridge":
                return self._predict_multiclass_kernel_ridge(x, return_update)
            elif self.scorer == "knn":
                return self._predict_multiclass_knn(x, return_update)
            elif self.scorer == "svm":
                return self._predict_multiclass_svm(x, return_update)

        if self.scorer == "ridge":
            return self._predict_ridge(x, return_update)
        elif self.scorer == "kernel_ridge":
            return self._predict_kernel_ridge(x, return_update)
        elif self.scorer == "knn":
            return self._predict_knn(x, return_update)
        elif self.scorer == "svm":
            return self._predict_svm(x, return_update)

    def _predict_ridge(self, x, return_update):
        """Ridge scoring: S(x_i) = fitted value from ridge on augmented set."""
        n = self.X.shape[0]

        # Augment X with test point
        X_aug = np.vstack([self.X, x.reshape(1, -1)])

        # Sherman-Morrison update for augmented XTXinv
        XTXinv_aug = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (
            1 + x.T @ self.XTXinv @ x
        )

        # Scores for hypothesis y=0:
        # beta_0 = XTXinv_aug @ X_aug^T @ [y_train; 0]
        y_ext_0 = np.append(self.y.astype(np.float64), 0.0)
        beta_0 = XTXinv_aug @ X_aug.T @ y_ext_0
        scores_0 = X_aug @ beta_0  # fitted values = scores

        # Scores for hypothesis y=1:
        # scores_1 = scores_0 + h_col (last column of hat matrix)
        h_col = X_aug @ XTXinv_aug @ X_aug[-1]
        scores_1 = scores_0 + h_col

        # Labels for each hypothesis
        labels_0 = np.append(self.y, 0).astype(np.float64)
        labels_1 = np.append(self.y, 1).astype(np.float64)

        # Isotonic calibration
        test_idx = n  # last position in augmented arrays
        p0 = _isotonic_calibrate(scores_0, labels_0, test_idx)
        p1 = _isotonic_calibrate(scores_1, labels_1, test_idx)

        pred = VennPrediction.binary(p0, p1)

        if return_update:
            return pred, {"XTXinv": XTXinv_aug}
        return pred

    def _predict_kernel_ridge(self, x, return_update):
        """Kernel-ridge scoring: S(x_i) = fitted values on augmented set."""
        n = self.X.shape[0]

        # Augment Gram and inverse state with robust fallback when needed
        K_aug, Ka_inv_aug = self._augment_kernel_state(x)

        # Scores for hypothesis y=0
        y_ext_0 = np.append(self.y.astype(np.float64), 0.0)
        beta_0 = Ka_inv_aug @ y_ext_0
        scores_0 = K_aug @ beta_0

        # Scores for hypothesis y=1
        # scores_1 = scores_0 + last column of kernel-ridge hat matrix
        h_col = K_aug @ Ka_inv_aug[:, -1]
        scores_1 = scores_0 + h_col

        labels_0 = np.append(self.y, 0).astype(np.float64)
        labels_1 = np.append(self.y, 1).astype(np.float64)

        test_idx = n
        p0 = _isotonic_calibrate(scores_0, labels_0, test_idx)
        p1 = _isotonic_calibrate(scores_1, labels_1, test_idx)

        pred = VennPrediction.binary(p0, p1)

        if return_update:
            return pred, {"K": K_aug, "Ka_inv": Ka_inv_aug}
        return pred

    def _predict_knn(self, x, return_update):
        """k-NN scoring: S(x_i) = agg(d_same) - agg(d_diff)."""
        n = self.X.shape[0]
        k = self.k
        agg_func = np.mean if self.aggregation == "mean" else np.median

        # Augment distance matrix with test point
        d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
        D_aug = np.empty((n + 1, n + 1), dtype=np.float64)
        D_aug[:n, :n] = self.D
        D_aug[:n, n] = d
        D_aug[n, :n] = d
        D_aug[n, n] = 0.0

        test_idx = n

        # For each hypothesis y ∈ {0, 1}, compute scores for all n+1 points
        results = []
        for y_hyp in (0, 1):
            labels_aug = np.append(self.y, y_hyp)
            scores = self._compute_knn_scores(D_aug, labels_aug, k, agg_func)
            p = _isotonic_calibrate(scores, labels_aug.astype(np.float64), test_idx)
            results.append(p)

        pred = VennPrediction.binary(results[0], results[1])

        if return_update:
            return pred, {"D": D_aug}
        return pred

    def _predict_svm(self, x, return_update):
        """SVM scoring: S(x_i) = decision function value from SVM on augmented set."""
        from online_cp.classifiers import _smo_solve

        n = self.X.shape[0]

        # Augment Gram matrix with test point
        k_row = np.atleast_1d(self._kernel(self.X, x))
        kappa = float(self._kernel(x.reshape(1, -1))[0, 0])
        K_aug = np.empty((n + 1, n + 1), dtype=np.float64)
        K_aug[:n, :n] = self.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        test_idx = n

        # For each hypothesis y ∈ {0, 1}, solve SVM and compute decision function
        results = []
        for y_hyp in (0, 1):
            labels_aug = np.append(self.y, y_hyp)
            y_binary = (2 * labels_aug - 1).astype(np.float64)  # {0,1} → {-1,+1}

            alpha, b = _smo_solve(K_aug, y_binary, self.C, self.smo_tol, self.smo_max_iter)

            # Decision function: f(x_i) = K_aug[i] @ (alpha * y_binary) + b
            scores = K_aug @ (alpha * y_binary) + b

            p = _isotonic_calibrate(scores, labels_aug.astype(np.float64), test_idx)
            results.append(p)

        pred = VennPrediction.binary(results[0], results[1])

        if return_update:
            return pred, {"K": K_aug}
        return pred

    # ------------------------------------------------------------------
    # Multiclass prediction methods (OVR isotonic calibration)
    # ------------------------------------------------------------------

    def _predict_multiclass_ridge(self, x, return_update):
        """Multiclass ridge: 2|Y| PAVA calls via hat-matrix decomposition."""
        n = self.X.shape[0]
        n_labels = len(self.label_space)

        # Augment X with test point
        X_aug = np.vstack([self.X, x.reshape(1, -1)])

        # Sherman-Morrison update for augmented XTXinv
        XTXinv_aug = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (
            1 + x.T @ self.XTXinv @ x
        )

        # Hat matrix last column (shared across all target classes)
        h_col = X_aug @ XTXinv_aug @ X_aug[-1]

        test_idx = n
        probs = np.empty((n_labels, n_labels), dtype=np.float64)

        for j, y_prime in enumerate(self.label_space):
            # Indicator for target class y' (test point = 0 for off-diagonal)
            ind_train = (self.y == y_prime).astype(np.float64)
            ind_off = np.append(ind_train, 0.0)
            ind_on = np.append(ind_train, 1.0)

            # Base scores (hypothesis v ≠ y': test entry contributes 0)
            base_scores = X_aug @ XTXinv_aug @ X_aug.T @ ind_off

            # Diagonal scores (hypothesis v = y': test entry contributes 1)
            scores_on = base_scores + h_col

            # Off-diagonal: all v ≠ y' share the same calibrated value
            p_off = _isotonic_calibrate(base_scores, ind_off, test_idx)
            # Diagonal: v = y'
            p_on = _isotonic_calibrate(scores_on, ind_on, test_idx)

            for i, v in enumerate(self.label_space):
                if v == y_prime:
                    probs[i, j] = p_on
                else:
                    probs[i, j] = p_off

        # Normalize rows
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs /= row_sums

        pred = VennPrediction(probs, self.label_space)

        if return_update:
            return pred, {"XTXinv": XTXinv_aug}
        return pred

    def _predict_multiclass_kernel_ridge(self, x, return_update):
        """Multiclass kernel ridge: 2|Y| PAVA calls via kernel hat-matrix."""
        n = self.X.shape[0]
        n_labels = len(self.label_space)

        # Augment Gram and inverse
        K_aug, Ka_inv_aug = self._augment_kernel_state(x)

        # Hat matrix last column (shared)
        h_col = K_aug @ Ka_inv_aug[:, -1]

        test_idx = n
        probs = np.empty((n_labels, n_labels), dtype=np.float64)

        for j, y_prime in enumerate(self.label_space):
            ind_train = (self.y == y_prime).astype(np.float64)
            ind_off = np.append(ind_train, 0.0)
            ind_on = np.append(ind_train, 1.0)

            base_scores = K_aug @ Ka_inv_aug @ ind_off
            scores_on = base_scores + h_col

            p_off = _isotonic_calibrate(base_scores, ind_off, test_idx)
            p_on = _isotonic_calibrate(scores_on, ind_on, test_idx)

            for i, v in enumerate(self.label_space):
                if v == y_prime:
                    probs[i, j] = p_on
                else:
                    probs[i, j] = p_off

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs /= row_sums

        pred = VennPrediction(probs, self.label_space)

        if return_update:
            return pred, {"K": K_aug, "Ka_inv": Ka_inv_aug}
        return pred

    def _predict_multiclass_knn(self, x, return_update):
        """Multiclass kNN: 2|Y| score computations via OVR binarization."""
        n = self.X.shape[0]
        n_labels = len(self.label_space)
        k = self.k
        agg_func = np.mean if self.aggregation == "mean" else np.median

        # Augment distance matrix
        d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
        D_aug = np.empty((n + 1, n + 1), dtype=np.float64)
        D_aug[:n, :n] = self.D
        D_aug[:n, n] = d
        D_aug[n, :n] = d
        D_aug[n, n] = 0.0

        test_idx = n
        probs = np.empty((n_labels, n_labels), dtype=np.float64)

        for j, y_prime in enumerate(self.label_space):
            # Binarize: 1 if label == y', 0 otherwise
            ind_train = (self.y == y_prime).astype(np.float64)
            ind_off = np.append(ind_train, 0.0)  # test point NOT in class y'
            ind_on = np.append(ind_train, 1.0)  # test point IN class y'

            # Compute OVR kNN scores for both variants
            scores_off = self._compute_knn_scores_binary(D_aug, ind_off, k, agg_func)
            scores_on = self._compute_knn_scores_binary(D_aug, ind_on, k, agg_func)

            p_off = _isotonic_calibrate(scores_off, ind_off, test_idx)
            p_on = _isotonic_calibrate(scores_on, ind_on, test_idx)

            for i, v in enumerate(self.label_space):
                if v == y_prime:
                    probs[i, j] = p_on
                else:
                    probs[i, j] = p_off

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs /= row_sums

        pred = VennPrediction(probs, self.label_space)

        if return_update:
            return pred, {"D": D_aug}
        return pred

    def _predict_multiclass_svm(self, x, return_update):
        """Multiclass SVM: 2|Y| SVM solves via OVR binarization."""
        from online_cp.classifiers import _smo_solve

        n = self.X.shape[0]
        n_labels = len(self.label_space)

        # Augment Gram matrix
        k_row = np.atleast_1d(self._kernel(self.X, x))
        kappa = float(self._kernel(x.reshape(1, -1))[0, 0])
        K_aug = np.empty((n + 1, n + 1), dtype=np.float64)
        K_aug[:n, :n] = self.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        test_idx = n
        probs = np.empty((n_labels, n_labels), dtype=np.float64)

        for j, y_prime in enumerate(self.label_space):
            ind_train = (self.y == y_prime).astype(np.float64)
            ind_off = np.append(ind_train, 0.0)
            ind_on = np.append(ind_train, 1.0)

            # OVR SVM: labels {-1, +1} from indicator
            y_bin_off = (2 * ind_off - 1).astype(np.float64)
            y_bin_on = (2 * ind_on - 1).astype(np.float64)

            alpha_off, b_off = _smo_solve(
                K_aug, y_bin_off, self.C, self.smo_tol, self.smo_max_iter
            )
            scores_off = K_aug @ (alpha_off * y_bin_off) + b_off

            alpha_on, b_on = _smo_solve(
                K_aug, y_bin_on, self.C, self.smo_tol, self.smo_max_iter
            )
            scores_on = K_aug @ (alpha_on * y_bin_on) + b_on

            p_off = _isotonic_calibrate(scores_off, ind_off, test_idx)
            p_on = _isotonic_calibrate(scores_on, ind_on, test_idx)

            for i, v in enumerate(self.label_space):
                if v == y_prime:
                    probs[i, j] = p_on
                else:
                    probs[i, j] = p_off

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs /= row_sums

        pred = VennPrediction(probs, self.label_space)

        if return_update:
            return pred, {"K": K_aug}
        return pred

    @staticmethod
    def _compute_knn_scores_binary(D, labels, k, agg_func):
        """Compute binary kNN scores for OVR isotonic calibration.

        For each point i with binary labels (0/1), compute:
        score = agg(d_to_class_0) - agg(d_to_class_1)
        Higher score → more likely to be class 1 (monotone for PAVA).
        """
        n = len(labels)
        k_use = min(k, n - 1)
        if k_use == 0:
            return np.zeros(n)

        scores = np.empty(n, dtype=np.float64)
        D_work = D.copy()
        np.fill_diagonal(D_work, np.inf)

        idx_0 = np.flatnonzero(labels == 0)
        idx_1 = np.flatnonzero(labels == 1)

        for i in range(n):
            # Distance to class 0
            others_0 = idx_0[idx_0 != i]
            if len(others_0) == 0:
                d_to_0 = np.inf
            else:
                d_0_all = D_work[i, others_0]
                k_0 = min(k_use, len(others_0))
                d_to_0 = agg_func(np.partition(d_0_all, k_0 - 1)[:k_0])

            # Distance to class 1
            others_1 = idx_1[idx_1 != i]
            if len(others_1) == 0:
                d_to_1 = np.inf
            else:
                d_1_all = D_work[i, others_1]
                k_1 = min(k_use, len(others_1))
                d_to_1 = agg_func(np.partition(d_1_all, k_1 - 1)[:k_1])

            # Higher score → more likely class 1 (monotone)
            scores[i] = d_to_0 - d_to_1

        return scores

    @staticmethod
    def _compute_knn_scores(D, labels, k, agg_func):
        """Compute monotone k-NN scores for binary isotonic calibration.

        For each point i, computes agg(d_to_class_0) - agg(d_to_class_1)
        using the k nearest neighbours of the appropriate class (LOO).
        Higher score → more likely class 1 (monotone for PAVA).
        """
        n = len(labels)
        k_use = min(k, n - 1)
        if k_use == 0:
            return np.zeros(n)

        scores = np.empty(n, dtype=np.float64)

        D_work = D.copy()
        np.fill_diagonal(D_work, np.inf)

        idx_0 = np.flatnonzero(labels == 0)
        idx_1 = np.flatnonzero(labels == 1)

        for i in range(n):
            if labels[i] == 0:
                same_idx = idx_0
                diff_idx = idx_1
            else:
                same_idx = idx_1
                diff_idx = idx_0

            # Distances to same-class (excluding self)
            same_mask = same_idx[same_idx != i]
            if len(same_mask) == 0:
                d_same = 0.0
            else:
                d_same_all = D_work[i, same_mask]
                k_s = min(k_use, len(same_mask))
                d_same = agg_func(np.partition(d_same_all, k_s - 1)[:k_s])

            # Distances to different-class
            if len(diff_idx) == 0:
                d_diff = np.inf
            else:
                d_diff_all = D_work[i, diff_idx]
                k_d = min(k_use, len(diff_idx))
                d_diff = agg_func(np.partition(d_diff_all, k_d - 1)[:k_d])

            # Score: higher means more like class 1
            # d_diff - d_same: if same-class is close and diff is far, score is high
            # For class-1 points: d_same=dist to other 1s, d_diff=dist to 0s
            # For class-0 points: d_same=dist to other 0s, d_diff=dist to 1s
            # We want monotone: higher score → higher P(y=1)
            # Use: d_same_to_0 - d_same_to_1 (distance to class 0 minus distance to class 1)
            if labels[i] == 1:
                scores[i] = d_diff - d_same  # far from 0, close to 1 → high
            else:
                scores[i] = d_same - d_diff  # close to 0, far from 1 → low

        return scores


# ---------------------------------------------------------------------------
# NearestNeighboursVennPredictor
# ---------------------------------------------------------------------------


class NearestNeighboursVennPredictor:
    """Online Venn predictor with k-NN voting taxonomy.

    Uses the k-nearest-neighbour voting taxonomy from ALRW (§6.2): for each
    example *i* the taxonomy value is the number of positive labels among
    the *k* nearest neighbours of *x_i* (leave-one-out). This gives *k* + 1
    categories {0, 1, …, k}. Under each hypothesis *v* ∈ {0, 1} for the new
    example, empirical frequencies among examples sharing the new example's
    taxonomy category give the multiprobability output.

    For binary labels (|Y| = 2), the output is ``VennPrediction(p0, p1)``
    compatible with :func:`log_loss_point` and :func:`brier_point`.
    For multiclass labels (|Y| > 2), the output is a
    :class:`MulticlassVennPrediction` containing the full |Y| × |Y|
    multiprobability matrix.

    Parameters
    ----------
    k : int
        Number of nearest neighbours for the voting taxonomy (default 1).
    metric : str
        Distance metric passed to ``scipy.spatial.distance`` (default
        ``'euclidean'``).
    label_space : array-like or None
        Explicit set of possible labels. If None, inferred from training
        data. Useful when the initial training set may not contain all labels.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(30, 2)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> vp = NearestNeighboursVennPredictor(k=1)
    >>> vp.learn_initial_training_set(X[:20], y[:20])
    >>> pred = vp.predict(X[20])
    >>> bool(0 <= pred.p0 <= 1 and 0 <= pred.p1 <= 1)
    True
    """

    def __init__(self, k=1, metric="euclidean", label_space=None):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.metric = metric
        self._label_space_fixed = label_space is not None
        self.label_space = (
            np.asarray(sorted(label_space), dtype=int)
            if label_space is not None
            else None
        )
        self.X = None
        self.y = None
        self.D = None  # distance matrix (n×n)

    def _distance(self, X, Y=None):
        """Compute pairwise distances."""
        X = np.atleast_2d(X)
        if Y is None:
            return squareform(pdist(X, metric=self.metric))
        else:
            Y = np.atleast_2d(Y)
            return cdist(X, Y, metric=self.metric)

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]) -> None:
        """Batch-initialise with training data.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Training feature vectors.
        y : array-like, shape (n,)
            Integer labels (binary {0, 1} or multiclass).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=int)
        if self._label_space_fixed:
            unknown = set(np.unique(y)) - set(self.label_space)
            if unknown:
                raise ValueError(
                    f"Labels {sorted(unknown)} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.unique(y)
        else:
            self.label_space = np.sort(
                np.unique(np.concatenate([self.label_space, np.unique(y)]))
            )
        self.X = X
        self.y = y
        self.D = self._distance(X)

    def learn_one(self, x, y):
        """Incrementally add one observation.

        Parameters
        ----------
        x : array-like, shape (d,)
            Feature vector.
        y : int
            True label.

        Raises
        ------
        ValueError
            If ``label_space`` was declared at construction and ``y`` is not
            in it.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = int(y)
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(
                    f"Label {y} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.array([y], dtype=int)
        elif y not in self.label_space:
            self.label_space = np.sort(np.append(self.label_space, y))
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=int)
            self.D = np.zeros((1, 1))
        else:
            d = self._distance(self.X, x.reshape(1, -1)).ravel()
            n = self.D.shape[0]
            D_new = np.empty((n + 1, n + 1), dtype=np.float64)
            D_new[:n, :n] = self.D
            D_new[:n, n] = d
            D_new[n, :n] = d
            D_new[n, n] = 0.0
            self.D = D_new
            self.X = np.vstack([self.X, x.reshape(1, -1)])
            self.y = np.append(self.y, y)

    def predict_one(self, x):
        """Produce a Venn multiprobability prediction.

        .. deprecated:: 0.3.0
            Use :meth:`predict` instead.

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.

        Returns
        -------
        VennPrediction or MulticlassVennPrediction
            Binary: VennPrediction(p0, p1).
            Multiclass: MulticlassVennPrediction with |Y| × |Y| matrix.
        """
        warnings.warn(
            "predict_one() is deprecated, use predict() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.predict(x)

    def predict(self, x):
        """Produce a Venn multiprobability prediction.

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.

        Returns
        -------
        VennPrediction or MulticlassVennPrediction
            Binary: VennPrediction(p0, p1).
            Multiclass: MulticlassVennPrediction with |Y| × |Y| matrix.
        """
        x = np.asarray(x, dtype=np.float64).ravel()

        if self.X is None or len(self.y) == 0:
            if self.label_space is not None and len(self.label_space) > 2:
                n_labels = len(self.label_space)
                uniform = np.full(
                    (n_labels, n_labels), 1.0 / n_labels
                )
                return VennPrediction(uniform, self.label_space)
            return VennPrediction.binary(0.5, 0.5)

        n = len(self.y)

        # Augment distance matrix with the test point
        d = self._distance(self.X, x.reshape(1, -1)).ravel()
        D_aug = np.empty((n + 1, n + 1), dtype=np.float64)
        D_aug[:n, :n] = self.D
        D_aug[:n, n] = d
        D_aug[n, :n] = d
        D_aug[n, n] = 0.0

        # Effective k (cap at n-1 since leave-one-out among n+1 points
        # means each point has n neighbours available)
        k_eff = min(self.k, n)

        test_idx = n  # index of new example in augmented arrays

        # Binary path requires labels {0, 1}; other 2-class problems use multiclass
        _is_binary = (
            len(self.label_space) == 2
            and self.label_space[0] == 0
            and self.label_space[1] == 1
        )
        if _is_binary or len(self.label_space) <= 1:
            return self._predict_binary(D_aug, k_eff, test_idx)
        else:
            return self._predict_multiclass(D_aug, k_eff, test_idx)

    def _predict_binary(self, D_aug, k_eff, test_idx):
        """Binary prediction path (backward-compatible)."""
        results = []
        for v in (0, 1):
            labels_aug = np.append(self.y, v)
            taxonomies = self._compute_taxonomies(D_aug, labels_aug, k_eff)
            tau_new = taxonomies[test_idx]
            mask = taxonomies == tau_new
            matching_labels = labels_aug[mask]
            s_v_1 = np.sum(matching_labels) / len(matching_labels)
            results.append(s_v_1)
        return VennPrediction.binary(results[0], results[1])

    def _predict_multiclass(self, D_aug, k_eff, test_idx):
        """Multiclass prediction path (|Y| > 2)."""
        n_labels = len(self.label_space)
        probs = np.empty((n_labels, n_labels), dtype=np.float64)

        for i, v in enumerate(self.label_space):
            labels_aug = np.append(self.y, v)
            taxonomies = self._compute_taxonomies_multiclass(
                D_aug, labels_aug, k_eff
            )
            tau_new = taxonomies[test_idx]
            mask = taxonomies == tau_new
            matching_labels = labels_aug[mask]

            # Frequency of each label among matching examples
            for j, y_prime in enumerate(self.label_space):
                probs[i, j] = np.sum(matching_labels == y_prime) / len(
                    matching_labels
                )

        return VennPrediction(probs, self.label_space)

    @staticmethod
    def _compute_taxonomies(D, labels, k):
        """Compute kNN voting taxonomy for all examples.

        For each example i, taxonomy τᵢ = sum of labels of k nearest
        neighbours (leave-one-out).

        Parameters
        ----------
        D : ndarray, shape (n, n)
            Pairwise distance matrix.
        labels : ndarray, shape (n,)
            Binary labels.
        k : int
            Number of neighbours.

        Returns
        -------
        taxonomies : ndarray, shape (n,), dtype int
            Taxonomy value for each example (in {0, 1, ..., k}).
        """
        n = len(labels)
        taxonomies = np.empty(n, dtype=int)

        # Set diagonal to inf for leave-one-out
        D_work = D.copy()
        np.fill_diagonal(D_work, np.inf)

        for i in range(n):
            # Find k nearest neighbours
            if k >= n - 1:
                # Use all other points
                nn_idx = np.arange(n)
                nn_idx = nn_idx[nn_idx != i]
            else:
                # Partial sort to find k nearest
                nn_idx = np.argpartition(D_work[i], k)[:k]

            taxonomies[i] = np.sum(labels[nn_idx])

        return taxonomies

    @staticmethod
    def _compute_taxonomies_multiclass(D, labels, k):
        """Compute same-class-count taxonomy for multiclass labels.

        For each example i, taxonomy τᵢ = number of k nearest neighbours
        that share the same label as example i (leave-one-out).

        Parameters
        ----------
        D : ndarray, shape (n, n)
            Pairwise distance matrix.
        labels : ndarray, shape (n,)
            Integer labels.
        k : int
            Number of neighbours.

        Returns
        -------
        taxonomies : ndarray, shape (n,), dtype int
            Taxonomy value for each example (in {0, 1, ..., k}).
        """
        n = len(labels)
        taxonomies = np.empty(n, dtype=int)

        D_work = D.copy()
        np.fill_diagonal(D_work, np.inf)

        for i in range(n):
            if k >= n - 1:
                nn_idx = np.arange(n)
                nn_idx = nn_idx[nn_idx != i]
            else:
                nn_idx = np.argpartition(D_work[i], k)[:k]

            taxonomies[i] = np.sum(labels[nn_idx] == labels[i])

        return taxonomies

