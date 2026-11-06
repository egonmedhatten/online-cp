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
    from ._serialization import SerializableMixin
except ImportError:
    from _serialization import SerializableMixin

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


from online_cp.mondrian.tree import MondrianTree

__all__ = [
    "VennPredictor",
    "VennAbersPredictor",
    "NearestNeighboursVennPredictor",
    "MondrianVennPredictor",
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
            raise AttributeError("p0 is only defined for binary predictions (|Y|=2)")
        return float(self.probs[0, 1])

    @property
    def p1(self):
        """P(y=1) under hypothesis y=1. Only valid for binary (|Y|=2)."""
        if len(self.label_space) != 2:
            raise AttributeError("p1 is only defined for binary predictions (|Y|=2)")
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
# VennPredictor — abstract base class
# ---------------------------------------------------------------------------


class VennPredictor(SerializableMixin):
    """Abstract base class for online Venn predictors (ALRW2 §6.2).

    Provides shared label-space management, the empty-training-set fallback,
    and the generic taxonomy → multiprobability prediction loop used by
    taxonomy-based Venn predictors (:class:`NearestNeighboursVennPredictor`,
    :class:`MondrianVennPredictor`).

    Subclasses must implement :meth:`learn_initial_training_set`,
    :meth:`learn_one`, and :meth:`predict`. Taxonomy-based subclasses should
    also implement :meth:`_categories_for_hypothesis` and call
    :meth:`_venn_predict_from_taxonomy` from their ``predict`` method.

    This mirrors the role of
    :class:`~online_cp.classifiers.ConformalClassifier` and
    :class:`~online_cp.regressors.ConformalRegressor` in the conformal
    prediction hierarchy.
    """

    _SAVE_PARAMS: tuple = ("label_space",)
    _SAVE_STATE: tuple = ("label_space", "_label_space_fixed")

    def __init__(self, label_space=None) -> None:
        self._label_space_fixed: bool = label_space is not None
        self.label_space = np.asarray(sorted(label_space), dtype=int) if label_space is not None else None

    # ------------------------------------------------------------------
    # Label-space policy helpers
    # ------------------------------------------------------------------

    def _update_label_space_batch(self, y: NDArray) -> None:
        """Update label space from a batch of labels."""
        if self._label_space_fixed:
            unknown = set(np.unique(y)) - set(self.label_space)
            if unknown:
                raise ValueError(f"Labels {sorted(unknown)} not in declared label_space {self.label_space.tolist()}")
        elif self.label_space is None:
            self.label_space = np.unique(y)
        else:
            self.label_space = np.sort(np.unique(np.concatenate([self.label_space, np.unique(y)])))

    def _update_label_space_one(self, y: int) -> None:
        """Update label space from a single label."""
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(f"Label {y} not in declared label_space {self.label_space.tolist()}")
        elif self.label_space is None:
            self.label_space = np.array([y], dtype=int)
        elif y not in self.label_space:
            self.label_space = np.sort(np.append(self.label_space, y))

    # ------------------------------------------------------------------
    # Shared prediction helpers
    # ------------------------------------------------------------------

    def _empty_prediction(self) -> VennPrediction:
        """Uniform multiprobability prediction when no training data is available."""
        if self.label_space is not None and len(self.label_space) > 2:
            n_labels = len(self.label_space)
            uniform = np.full((n_labels, n_labels), 1.0 / n_labels)
            return VennPrediction(uniform, self.label_space)
        return VennPrediction.binary(0.5, 0.5)

    @property
    def _is_binary_label_space(self) -> bool:
        """Whether the (declared) label space is the canonical binary ``{0, 1}``.

        True only when ``label_space`` is explicitly declared with at most two
        entries that are exactly ``{0, 1}`` (or a single label). A label space of
        two *other* values (e.g. ``{3, 7}``) is treated as multiclass so that the
        per-hypothesis loop covers every label. Centralised here so the k-NN
        taxonomy / OVR scoring paths and the generic prediction loop all agree
        on which branch they take.
        """
        ls = self.label_space
        return (
            ls is not None
            and len(ls) <= 2
            and (len(ls) <= 1 or (int(ls[0]) == 0 and int(ls[1]) == 1))
        )

    def _categories_for_hypothesis(self, taxonomy_data: Any, v: int) -> tuple:
        """Return (categories, labels_aug, test_idx) for hypothesis y_test = v.

        Parameters
        ----------
        taxonomy_data : Any
            Precomputed data from the calling ``predict`` method (e.g. an
            augmented distance matrix for k-NN, or Mondrian leaf IDs).
        v : int
            Hypothesised test label.

        Returns
        -------
        categories : ndarray, shape (n+1,)
            Taxonomy category for each point in the augmented bag.
        labels_aug : ndarray, shape (n+1,)
            Labels for the augmented bag (last entry = v).
        test_idx : int
            Index of the test point in the augmented arrays (always n).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _categories_for_hypothesis")

    def _venn_predict_from_taxonomy(self, taxonomy_data: Any) -> VennPrediction:
        """Generic taxonomy → multiprobability prediction loop.

        Calls :meth:`_categories_for_hypothesis` once per hypothesis label,
        groups examples by taxonomy category, and returns empirical label
        frequencies within the test point's category.  Handles both binary
        (|Y| ≤ 2 with labels {0, 1}) and multiclass (|Y| > 2) settings.

        Parameters
        ----------
        taxonomy_data : Any
            Passed through to :meth:`_categories_for_hypothesis`.

        Returns
        -------
        VennPrediction
        """
        _is_binary = self._is_binary_label_space

        if _is_binary:
            results = []
            for v in (0, 1):
                categories, labels_aug, test_idx = self._categories_for_hypothesis(taxonomy_data, v)
                tau_new = categories[test_idx]
                mask = categories == tau_new
                matching = labels_aug[mask]
                results.append(float(np.sum(matching) / len(matching)))
            return VennPrediction.binary(results[0], results[1])

        n_labels = len(self.label_space)
        probs = np.empty((n_labels, n_labels), dtype=np.float64)
        for i, v in enumerate(self.label_space):
            categories, labels_aug, test_idx = self._categories_for_hypothesis(taxonomy_data, v)
            tau_new = categories[test_idx]
            mask = categories == tau_new
            matching = labels_aug[mask]
            for j, y_prime in enumerate(self.label_space):
                probs[i, j] = float(np.sum(matching == y_prime) / len(matching))
        return VennPrediction(probs, self.label_space)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Batch-initialise with a training set. Must be overridden by subclasses."""
        raise NotImplementedError

    def learn_one(self, x: NDArray, y: int) -> None:
        """Incrementally add one labelled observation. Must be overridden by subclasses."""
        raise NotImplementedError

    def predict(self, x: NDArray) -> VennPrediction:
        """Produce a Venn multiprobability prediction. Must be overridden by subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VennAbersPredictor
# ---------------------------------------------------------------------------


class VennAbersPredictor(VennPredictor):
    r"""Full online Venn-Abers predictor (Algorithm 6.1, [ALRW2 §6.4]).

    Produces *calibrated* probability predictions for binary classification.
    Rather than a single probability, a Venn predictor outputs a small set of
    probabilities — here the pair ``(p0, p1)`` obtained by isotonic-calibrating
    the score under each hypothesised test label — one of which is guaranteed to
    be (multiprobability-)calibrated. This *automatic calibration under
    exchangeability* is the defining validity property of Venn predictors
    ([ALRW2 Thm 6.4]); the resulting interval ``[p0, p1]`` quantifies the
    residual calibration uncertainty.

    This is the full/transductive variant — no data splitting. The scorer is
    retrained on the augmented dataset (training + hypothesised test label) for
    each prediction. Supports ridge regression, kernel ridge regression, k-NN,
    and SVM scoring functions.

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

    _SAVE_PARAMS: tuple = (
        "scorer",
        "a",
        "k",
        "distance",
        "distance_func",
        "aggregation",
        "kernel",
        "C",
        "sigma",
        "degree",
        "coef0",
        "smo_tol",
        "smo_max_iter",
        "label_space",
    )
    _SAVE_STATE: tuple = (
        "X",
        "y",
        "label_space",
        "_label_space_fixed",
        "XTXinv",
        "D",
        "K",
        "Ka_inv",
        "_kernel",
    )
    _SAVE_CALLABLES: tuple = ("distance_func", "kernel")
    _PARAM_MAP: dict = {
        "distance_func": "_distance_func_arg",
        "kernel": "kernel",
    }

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
            raise ValueError(f"scorer must be 'ridge', 'kernel_ridge', 'knn', or 'svm', got '{scorer}'")
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

        # Store raw constructor args for serialization
        self.kernel = kernel
        self._distance_func_arg = distance_func

        super().__init__(label_space=label_space)

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

        self._update_label_space_batch(y)

        self.X = X
        self.y = y

        if self.scorer == "ridge":
            Id = np.identity(X.shape[1])
            try:
                self.XTXinv = np.linalg.inv(X.T @ X + self.a * Id)
            except np.linalg.LinAlgError:
                raise ValueError("X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None
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

        self._update_label_space_one(y)

        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=int)
            if self.scorer == "ridge":
                Id = np.identity(self.X.shape[1])
                try:
                    self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * Id)
                except np.linalg.LinAlgError:
                    raise ValueError("X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None
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
                    # Update (X^T X + aI)^{-1} via the matrix-inversion lemma,
                    # symmetrized to counteract round-off drift over long streams.
                    v = self.XTXinv @ x
                    M = self.XTXinv - np.outer(v, v) / (1.0 + x @ v)
                    self.XTXinv = 0.5 * (M + M.T)
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

    def predict(
        self, x: NDArray[np.floating[Any]], return_update: bool = False
    ) -> VennPrediction | tuple[VennPrediction, dict[str, Any]]:
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
            pred = self._empty_prediction()
            if return_update:
                return pred, {}
            return pred

        # Dispatch: binary vs multiclass
        # Binary path requires labels {0, 1}; other 2-class problems use multiclass OVR.
        # Single-class label_space also uses binary path (hypothesizes both 0 and 1).
        _is_binary = self._is_binary_label_space
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
        XTXinv_aug = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

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
        XTXinv_aug = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

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

            alpha_off, b_off = _smo_solve(K_aug, y_bin_off, self.C, self.smo_tol, self.smo_max_iter)
            scores_off = K_aug @ (alpha_off * y_bin_off) + b_off

            alpha_on, b_on = _smo_solve(K_aug, y_bin_on, self.C, self.smo_tol, self.smo_max_iter)
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


class NearestNeighboursVennPredictor(VennPredictor):
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

    _SAVE_PARAMS: tuple = ("k", "metric", "label_space")
    _SAVE_STATE: tuple = ("X", "y", "D", "label_space", "_label_space_fixed")

    def __init__(self, k=1, metric="euclidean", label_space=None):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.metric = metric
        super().__init__(label_space=label_space)
        self.X = None
        self.y = None
        self.D = None  # distance matrix (n×n)

        # --- Test-free per-point k-NN neighbour cache (Section 7.5 speedup) ---
        # The taxonomy τᵢ of a training point depends on its k-nearest-neighbour
        # *set*, which is test-free (it depends only on the training bag). The test
        # point only changes a training point's k-NN set when it enters that
        # point's (k+1)-nearest set (a *semi-neighbour*). Caching each point's
        # test-free k-NN set + the (k+1)-th neighbour distance lets ``predict``
        # refresh only the (few) semi-neighbours per hypothesised label instead of
        # recomputing a lexsort over every point (O(n log n) each) — the same §7.5
        # recipe as the k-NN CPS / classifier.
        # _knn_idx[i]   : indices of point i's k nearest *training* neighbours
        #                 (test-free; canonical (distance, label) tie-break).
        # _knn_labels[i]: the labels of those k neighbours (so the binary
        #                 taxonomy is sum(_knn_labels[i]) and the multiclass
        #                 taxonomy is the count of _knn_labels[i] == y[i]).
        # _kth_dist[i]  : the (k+1)-th smallest distance from point i to the rest
        #                 of the training bag — the semi-neighbour threshold.
        # _count[i]     : number of available training neighbours of point i
        #                 (n-1 when unpruned; warm-up regime n < k+1).
        self._knn_idx = None
        self._knn_labels = None
        self._kth_dist = None
        self._count = None

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
        self._update_label_space_batch(y)
        self.X = X
        self.y = y
        self.D = self._distance(X)
        # Section 7.5: the initial bag is set from scratch, so build the per-point
        # k-NN neighbour cache.
        self._build_per_point_cache()

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
        self._update_label_space_one(y)
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
        # Section 7.5: refresh the per-point k-NN neighbour cache for the new
        # example (semi-neighbour refresh; rebuilds from scratch if the cache is
        # empty or its size no longer matches, e.g. after a load).
        self._refresh_cache()

    def _ensure_cache(self) -> None:
        """Build the per-point k-NN neighbour cache if it is not populated.

        The cache is derived state (not persisted) and is (re)built lazily after a
        :meth:`load` (where only the distance matrix is restored) or before the
        first prediction on a fresh model.
        """
        if self.X is not None and self.X.shape[0] > 0:
            if self._knn_idx is None or self._knn_idx.shape[0] != self.X.shape[0]:
                self._build_per_point_cache()

    def _build_per_point_cache(self) -> None:
        """Populate the per-point k-NN neighbour cache from the current bag.

        For every training point *i* store its *k* nearest training neighbours
        (test-free; the canonical (distance, label) tie-break that keeps the
        taxonomy a bag-function of ``(X, y)``) and the (k+1)-th neighbour
        distance (the semi-neighbour threshold). Costs ``O(n^2 log n)`` to build
        from scratch but is amortised across streaming steps.
        """
        if self.X is None or self.X.shape[0] == 0:
            self._knn_idx = None
            return
        D = self.D
        n = D.shape[0]
        k = self.k
        y = self.y

        # Mask the diagonal (self-distance) so it never enters the k-NN set.
        D_work = D.copy()
        np.fill_diagonal(D_work, np.inf)

        # k-th smallest distance = the semi-neighbour threshold: when there are
        # at least k training neighbours (n-1 >= k), the test point enters point
        # i's k-NN set (displacing the farthest) iff its distance to i is at most
        # this. Fewer than k available training neighbours -> warm-up, the test
        # point is always in the set (handled via count, not via this threshold).
        if n - 1 >= k:
            # np.partition(D_work, k-1, axis=1) puts the k-th smallest
            # (0-indexed position k-1) at column k-1 of each row.
            kth_dist = np.partition(D_work, k - 1, axis=1)[:, k - 1]
        else:
            kth_dist = np.full(n, np.inf)
        count = np.full(n, n - 1, dtype=int)

        # k-NN width for the current bag (n-1 other points).
        nw = min(k, n - 1)
        knn_idx = np.empty((n, nw), dtype=int)
        for i in range(n):
            # (distance, label) tie-break: matches the original
            # np.lexsort((labels, D_work[i]))[:k] selection exactly, so the
            # taxonomy is bit-identical to the pre-§7.5 implementation.
            knn_idx[i] = np.lexsort((y, D_work[i]))[:nw]
        knn_labels = y[knn_idx]

        self._knn_idx = knn_idx
        self._knn_labels = knn_labels
        self._kth_dist = kth_dist
        self._count = count

    def _refresh_cache(self) -> None:
        """Populate/refresh the per-point k-NN neighbour cache after a state change.

        Builds from scratch when the cache is empty or its size no longer matches
        ``self.X`` (e.g. after a :meth:`load`); otherwise, when the bag grew by
        exactly one (a normal streaming ``learn_one``), does an *incremental
        append* — grow the cache by one row, compute the new example's k-NN set
        from its distance row, and refresh only the *semi-neighbours* of the new
        example (``O(S n log n)``, ``O(S k)`` when warm) instead of ``O(n^2 log
        n)``.

        A point *i* is a *semi-neighbour* of the new example iff the new example's
        distance to *i* is at most *i*'s cached k-th neighbour distance (when
        there are at least k training neighbours) — i.e. the new example enters
        (or ties at) *i*'s k-NN set. In warm-up (fewer than k training neighbours)
        the new example is always in *i*'s k-NN set, so *i* is a semi-neighbour.
        Only those points' k-NN sets / k-th distances are recomputed; for all
        others the cached values are unchanged.

        .. note::
            This is the §7.5 neighbour-list cache (ALRW2 Sec. 7.5); it is the
            Venn analogue of the near-identical ``_build_per_point_cache`` /
            ``_refresh_cache`` pair on
            :class:`~online_cp.CPS.NearestNeighboursPredictionMachine` (k-NN CPS).
            The two differ in two *deliberate* ways that encode each model's
            distinct invariant — do not "synchronise" them away: (a) the Venn
            tie-breaks on ``(label, distance)`` (a bag-function of ``(X, y)``)
            while the CPS tie-breaks on ``(global-index, distance)`` (a
            bag-function of ``X`` only, matching its ``argpartition`` + ``argsort``
            reference); and (b) the Venn stores the **k-th** neighbour distance
            (the test point enters a training point's *k*-NN set) while the CPS
            stores the **(k+1)-th** (its ``predict_cpd`` works with (k+1)-NN
            sets). Keep the two implementations in lock-step on the shared
            incremental-append skeleton (grow-by-one-row, recompute the new row,
            refresh only the semi-neighbours) but leave the tie-break key and the
            partition rank model-specific.
        """
        if self.X is None or self.X.shape[0] == 0:
            self._knn_idx = None
            return
        n = self.X.shape[0]
        if self._knn_idx is None:
            self._build_per_point_cache()
            return
        old_n = self._knn_idx.shape[0]
        if old_n != n - 1:
            # Size did not grow by exactly one (load / init): rebuild from scratch.
            self._build_per_point_cache()
            return

        D = self.D
        k = self.k
        y = self.y
        # The new example is at position n-1 (0-indexed) in the grown bag. Its
        # distances to the existing n-1 points (positions 0..n-2) are D[n-1, :n-1].
        d_new = D[-1, :-1]

        nw = min(k, n - 1)  # k-NN width for the current bag (n-1 other points)

        # Grow the caches by one row (the new example is at position n-1). The
        # old cache has `old_w` columns (possibly < nw in warm-up); pad the right
        # with placeholder entries that are recomputed for the semi-neighbours
        # (and the new row) below.
        old_w = self._knn_idx.shape[1]
        knn_idx = np.zeros((n, nw), dtype=int)
        knn_labels = np.zeros((n, nw), dtype=int)
        knn_idx[:n - 1, :old_w] = self._knn_idx
        knn_labels[:n - 1, :old_w] = self._knn_labels
        kth_dist = np.append(
            self._kth_dist.copy(),
            np.partition(d_new, k - 1)[k - 1] if d_new.size >= k else np.inf,
        )
        count = np.append(self._count.copy(), d_new.size)

        # Semi-neighbour mask over the existing n-1 points (positions 0..n-2).
        # count < k means fewer than k training neighbours are available (warm-up)
        # — those points' sets extend as the bag grows, so they must be refreshed.
        # Otherwise the new example enters point i's k-NN set iff d_new[i] <= the
        # cached k-th neighbour distance.
        semi = (count[:-1] < k) | (d_new <= kth_dist[:-1])

        for j in np.flatnonzero(semi):
            ii = int(j)
            dists = D[ii, :].copy()
            dists[ii] = np.inf  # exclude self
            # (distance, label) tie-break, matching _build_per_point_cache.
            order = np.lexsort((y, dists))
            knn_idx[ii] = order[:nw]
            knn_labels[ii] = y[order[:nw]]
            kth_dist[ii] = np.partition(dists, k - 1)[k - 1] if dists.size >= k else np.inf

        # The new example's own k-NN set, from its distance row (n-1 candidates:
        # the existing points at positions 0..n-2).
        order = np.lexsort((y[:-1], d_new))
        knn_idx[-1] = order[:nw]
        knn_labels[-1] = y[order[:nw]]

        self._knn_idx = knn_idx
        self._knn_labels = knn_labels
        self._kth_dist = kth_dist
        self._count = count

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
            return self._empty_prediction()

        n = len(self.y)

        # Section 7.5 speedup: instead of building the full (n+1) x (n+1)
        # distance matrix and re-running a lexsort over every point per hypothesis,
        # (a) compute the test point's distances to the training bag, (b) identify
        # its *semi-neighbours* (training points whose k-NN set the test point
        # enters), and (c) let _categories_for_hypothesis refresh only those
        # points' cached taxonomies per hypothesised label.
        self._ensure_cache()
        d = self._distance(self.X, x.reshape(1, -1)).ravel()

        # Effective k (cap at n-1 since leave-one-out among n+1 points
        # means each point has n neighbours available).
        k_eff = min(self.k, n)

        # Semi-neighbour threshold: when a training point i has at least k
        # training neighbours (count[i] >= k), the test point enters i's k-NN set
        # (displacing the farthest) iff d[i] <= i's k-th neighbour distance. In
        # warm-up (count[i] < k) the test point is always in i's k-NN set, so all
        # such points are semi-neighbours.
        count = self._count
        kth = self._kth_dist
        if count is not None and kth is not None:
            semi_mask = (count < self.k) | (d <= kth)
        else:
            # Defensive: cache should always be populated here; fall back to all.
            semi_mask = np.ones(n, dtype=bool)

        labels_aug = np.append(self.y, 0)  # placeholder; the test label is set per hypothesis
        taxonomy_data = (d, k_eff, n, labels_aug, semi_mask)
        return self._venn_predict_from_taxonomy(taxonomy_data)

    def _categories_for_hypothesis(self, taxonomy_data: Any, v: int) -> tuple:
        """Return (taxonomies, labels_aug, test_idx) for hypothesis y_test = v.

        Uses the per-point k-NN neighbour cache (Section 7.5) to compute the
        taxonomies: the cached (test-free) taxonomies are reused for every
        training point that is *not* a semi-neighbour of the test point, and the
        (few) semi-neighbours are re-derived with the test point (label ``v``)
        inserted into their k-NN set.

        Binary labels {0, 1} use the voting taxonomy (count of 1s among k NN);
        multiclass labels use the same-class-count taxonomy. This preserves the
        original per-class taxonomy semantics (ALRW2 §6.2) and is bit-identical
        to the pre-§7.5 ``_compute_taxonomies`` on the same bag.

        Parameters
        ----------
        taxonomy_data : tuple
            ``(d, k_eff, test_idx, labels_aug, semi_mask)`` precomputed by
            :meth:`predict`, where ``d`` is the test point's distance vector to
            the training bag and ``semi_mask`` marks the training points whose
            k-NN set the test point enters.
        v : int
            Hypothesised test label.

        Returns
        -------
        taxonomies : ndarray, shape (n+1,)
        labels_aug : ndarray, shape (n+1,)
        test_idx : int
        """
        d, k_eff, test_idx, labels_aug, semi_mask = taxonomy_data
        n = len(self.y)
        labels_aug = np.append(self.y, v)

        # Binary {0,1}: voting taxonomy (count of 1s in k NN).
        # Multiclass / non-{0,1} 2-class: same-class-count taxonomy.
        _is_binary = self._is_binary_label_space

        # Cached test-free taxonomies for every training point (the binary
        # voting sum, or the same-class count, over the point's cached k-NN set).
        y = self.y
        knn_labels = self._knn_labels
        if _is_binary:
            tax_train = np.sum(knn_labels, axis=1)
        else:
            tax_train = np.sum(knn_labels == y[:, None], axis=1)

        # Start from the cached taxonomies; refresh the (few) semi-neighbours
        # whose k-NN set the test point (label v) enters.
        taxonomies = np.empty(n + 1, dtype=int)
        taxonomies[:n] = tax_train

        if np.any(semi_mask):
            knn_idx = self._knn_idx
            D = self.D
            for i in np.flatnonzero(semi_mask):
                ii = int(i)
                # Point ii's augmented k-NN set is the k_eff nearest among its
                # cached training neighbours plus the test point (label v).
                # Fetch only the cached neighbours' distances (O(k)) and the test
                # distance, then select the k_eff nearest by (distance, label).
                nbr = knn_idx[ii]
                cand_dist = np.append(D[ii, nbr], d[ii])
                cand_lab = np.append(knn_labels[ii], v)
                order = np.lexsort((cand_lab, cand_dist))[:k_eff]
                if _is_binary:
                    taxonomies[ii] = int(np.sum(cand_lab[order]))
                else:
                    taxonomies[ii] = int(np.sum(cand_lab[order] == y[ii]))

        # Test point's own taxonomy: its k-NN set is the k_eff nearest of the n
        # training points, selected by (distance, label); its label is v.
        order_test = np.lexsort((y, d))[:k_eff]
        if _is_binary:
            taxonomies[n] = int(np.sum(y[order_test]))
        else:
            taxonomies[n] = int(np.sum(y[order_test] == v))

        return taxonomies, labels_aug, test_idx


# ---------------------------------------------------------------------------
# Mondrian Venn predictor
# ---------------------------------------------------------------------------


class MondrianVennPredictor(VennPredictor):
    """Online Venn predictor using the Mondrian tree partition as taxonomy.

    Uses a Mondrian tree leaf as the Venn taxonomy category (ALRW2 §6.2).
    At each call to ``predict(x)`` a single Mondrian tree is built from the
    augmented dataset {X_train ∪ {x_test}}.  Each example is assigned to a
    leaf; the multiprobability prediction is the empirical label distribution
    within the test point's leaf.

    The Mondrian partition depends only on X-values (bounding boxes), so the
    same tree is reused for all hypothesised test labels v ∈ Y, making each
    ``predict`` call build only one tree regardless of label-space size.
    The partition is a bag function (permutation-invariant in the augmented
    training set), guaranteeing exact Venn validity under exchangeability
    (ALRW2 Thm 6.4).

    Parameters
    ----------
    lifetime : float or {'sqrt_n', 'density'}
        Mondrian time budget. Larger → more splits → smaller leaves.
        Leaves are created when the cumulative split time exceeds ``lifetime``.
        Default 1.0.

        Can also be a string for unsupervised automatic tuning (only X is
        used; labels are never accessed):

        - ``'sqrt_n'``: halt when the leaf count drops to ≤ √n. Coarsens
          the partition in sparse regions.
        - ``'density'``: choose τ maximising the histogram log-likelihood
          Σ_leaf n_leaf · log(n_leaf / V_leaf). Adapts to feature density.
    max_depth : int or None
        Hard depth cap on the tree. Nodes at depth >= max_depth become leaves
        regardless of remaining ``lifetime``. None = no cap.
    min_samples_leaf : int
        Minimum number of points in each leaf. A split that would produce a
        child with fewer than ``min_samples_leaf`` points is refused. Default 1
        (no restriction; equivalent to the canonical Mondrian algorithm). Set
        to 2 to eliminate singleton leaves and avoid extreme p0/p1 outputs.
    label_space : array-like or None
        Explicit set of possible labels. If None, inferred from training data.
        When provided, the label space is fixed and labels outside it are
        rejected.
    rnd_state : int or None
        Seed for the random generator used to build Mondrian trees. Predictions
        are reproducible for a fixed seed and fixed call order.
    feature_weights : {'variance'} or array-like of shape (d,) or None
        Unsupervised feature importance for split dimension sampling. Labels
        are never accessed.

        - ``None`` (default): uniform — standard isotropic Mondrian.
        - ``'variance'``: split probabilities ∝ population variance per
          feature. High-variance dimensions are sampled more often.
        - array of shape (d,): explicit non-negative weights (normalised
          internally; need not sum to 1).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(40, 2)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> mvp = MondrianVennPredictor(lifetime=1.0, rnd_state=42)
    >>> mvp.learn_initial_training_set(X[:30], y[:30])
    >>> pred = mvp.predict(X[30])
    >>> bool(0.0 <= pred.p0 <= 1.0 and 0.0 <= pred.p1 <= 1.0)
    True
    """

    _SAVE_PARAMS: tuple = (
        "lifetime", "max_depth", "min_samples_leaf", "label_space", "rnd_state",
        "feature_weights",
    )
    _SAVE_STATE: tuple = ("X", "y", "label_space", "_label_space_fixed")

    def __init__(
        self,
        lifetime: float | str = 1.0,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        label_space=None,
        rnd_state: int | None = None,
        feature_weights: str | NDArray | None = None,
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if isinstance(lifetime, str) and lifetime not in ("sqrt_n", "density"):
            raise ValueError(
                f"Unknown lifetime string {lifetime!r}. "
                "Recognised values: 'sqrt_n', 'density'."
            )
        if isinstance(feature_weights, str) and feature_weights not in ("variance",):
            raise ValueError(
                f"Unknown feature_weights string {feature_weights!r}. "
                "Recognised value: 'variance'."
            )

        super().__init__(label_space=label_space)

        self.lifetime = lifetime
        self.feature_weights = feature_weights
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rnd_state = rnd_state
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.X: NDArray | None = None
        self.y: NDArray | None = None

    # ------------------------------------------------------------------
    # Online learning interface
    # ------------------------------------------------------------------

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Batch-initialise with a training set.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Training feature vectors.
        y : array-like, shape (n,)
            Integer class labels.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=int)
        self._update_label_space_batch(y)
        self.X = X
        self.y = y

    def learn_one(self, x: NDArray, y: int) -> None:
        """Incrementally add one labelled observation.

        Parameters
        ----------
        x : array-like, shape (d,)
            Feature vector.
        y : int
            True label.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = int(y)
        self._update_label_space_one(y)
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=int)
        else:
            self.X = np.vstack([self.X, x.reshape(1, -1)])
            self.y = np.append(self.y, y)

    # ------------------------------------------------------------------
    # Mondrian taxonomy helpers
    # ------------------------------------------------------------------

    def _build_augmented_leaf_ids(self, x: NDArray) -> NDArray:
        """Build one Mondrian tree from X_aug = [X_train; x] and return leaf IDs.

        The leaf assignment is label-independent — the same tree is used for
        all hypothesised test labels v, so we build it only once per
        ``predict()`` call.

        Uses ``leaf.indices`` stored on each leaf node (O(n) total, avoids
        per-point tree traversals).

        Parameters
        ----------
        x : ndarray, shape (d,)
            Test feature vector (already flattened).

        Returns
        -------
        leaf_ids : ndarray of int, shape (n+1,)
            Integer leaf identifier for each point in the augmented bag.
            Points sharing the same id belong to the same Mondrian leaf.
        """
        n = len(self.y)
        X_aug = np.vstack([self.X, x.reshape(1, -1)])

        # Delegate partition construction to the standalone MondrianTree core.
        # grow() resolves feature weights and (string) lifetimes and advances
        # self.rnd_gen in the exact same order as the raw engine calls, so
        # seeded predictions are unchanged.
        tree = MondrianTree.grow(
            X_aug,
            self.rnd_gen,
            lifetime=self.lifetime,
            x_test=x,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            feature_weights=self.feature_weights,
        )

        # Assign leaf IDs from leaf.indices — O(n), no redundant tree walks.
        leaves = tree.collect_leaves()
        leaf_ids = np.empty(n + 1, dtype=np.intp)
        for leaf_id, leaf in enumerate(leaves):
            leaf_ids[leaf.indices] = leaf_id

        return leaf_ids

    def _categories_for_hypothesis(
        self, taxonomy_data: Any, v: int
    ) -> tuple:
        """Return (leaf_ids, labels_aug, test_idx) for hypothesis y_test = v.

        Parameters
        ----------
        taxonomy_data : ndarray, shape (n+1,)
            Leaf IDs precomputed by :meth:`_build_augmented_leaf_ids`.
        v : int
            Hypothesised test label.

        Returns
        -------
        leaf_ids : ndarray, shape (n+1,)
        labels_aug : ndarray, shape (n+1,)
        test_idx : int
        """
        leaf_ids = taxonomy_data
        test_idx = len(self.y)
        labels_aug = np.append(self.y, v)
        return leaf_ids, labels_aug, test_idx

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x: NDArray) -> VennPrediction:
        """Produce a Venn multiprobability prediction.

        Builds one Mondrian tree from the augmented dataset {X_train ∪ {x}},
        assigns each point to a leaf, then computes empirical label frequencies
        within the test point's leaf under each hypothesis v ∈ Y.

        Parameters
        ----------
        x : array-like, shape (d,)
            Test feature vector.

        Returns
        -------
        VennPrediction
            Binary: contains p0, p1. Multiclass: |Y|×|Y| probs matrix.
        """
        x = np.asarray(x, dtype=np.float64).ravel()

        if self.X is None or len(self.y) == 0:
            return self._empty_prediction()

        leaf_ids = self._build_augmented_leaf_ids(x)
        return self._venn_predict_from_taxonomy(leaf_ids)
