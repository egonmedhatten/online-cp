"""Online Venn predictors.

This module implements Venn predictors that produce calibrated probability
predictions. Currently includes the Venn-Abers predictor (Algorithm 6.1,
ALRW2 §6.4) — the first known Python implementation of the full/transductive
variant.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    HAS_NUMBA = False

__all__ = [
    "VennAbersPredictor",
    "VennAbersPrediction",
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

    # Find where query_idx landed in the sorted order
    query_pos = np.searchsorted(order, query_idx)
    # searchsorted won't work directly — we need the position of query_idx in `order`
    # Actually we need: which position in the sorted array corresponds to query_idx
    pos = np.flatnonzero(order == query_idx)[0]
    return y_sorted[pos]


# ---------------------------------------------------------------------------
# VennAbersPrediction output type
# ---------------------------------------------------------------------------


class VennAbersPrediction:
    """Multiprobability prediction from a Venn-Abers predictor.

    The prediction is the pair (p0, p1) — two calibrated probabilities
    of class 1 under the two possible label hypotheses. This pair IS the
    prediction; it is not an interval or a point estimate.

    To merge into a single probability for decision-making, use the
    standalone functions :func:`log_loss_point` or :func:`brier_point`.

    Attributes
    ----------
    p0 : float
        Calibrated P(y=1) under the hypothesis that the true label is 0.
    p1 : float
        Calibrated P(y=1) under the hypothesis that the true label is 1.
    """

    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def __repr__(self):
        return f"VennAbersPrediction(p0={self.p0:.4f}, p1={self.p1:.4f})"

    def __str__(self):
        return f"(p0={self.p0:.4f}, p1={self.p1:.4f})"


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

    Examples
    --------
    >>> log_loss_point(0.2, 0.8)
    0.5
    >>> log_loss_point(0.0, 1.0)
    0.5
    """
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

    Supports ridge regression and k-NN scoring functions.

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
    ):
        """
        Parameters
        ----------
        scorer : {'ridge', 'knn'}
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
        """
        if scorer not in ("ridge", "knn"):
            raise ValueError(f"scorer must be 'ridge' or 'knn', got '{scorer}'")
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

        self.X = None
        self.y = None

        # Ridge state
        self.XTXinv = None
        self.p = None
        self.Id = None

        # k-NN state
        self.D = None
        self._label_indices = None

    def _standard_distance_func(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            return squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            return cdist(X, y, metric=self.distance)

    def learn_initial_training_set(self, X, y):
        """Batch-initialize with training data.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Training feature vectors.
        y : ndarray, shape (n,)
            Binary labels, must be in {0, 1}.
        """
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=int)
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Labels must be binary {0, 1}")

        self.X = X
        self.y = y

        if self.scorer == "ridge":
            self.p = X.shape[1]
            self.Id = np.identity(self.p)
            self.XTXinv = np.linalg.inv(X.T @ X + self.a * self.Id)
        elif self.scorer == "knn":
            self.D = self.distance_func(X)
            self._label_indices = {0: np.flatnonzero(y == 0), 1: np.flatnonzero(y == 1)}

    def learn_one(self, x, y, precomputed=None):
        """Incrementally add one observation after the true label is revealed.

        Parameters
        ----------
        x : array-like, shape (d,)
            Feature vector.
        y : int
            True binary label (0 or 1).
        precomputed : dict, optional
            Cached state from predict(return_update=True).
            For ridge: {'XTXinv': ...}
            For knn: {'D': ...}
        """
        x = np.asarray(x).ravel()

        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=int)
            if self.scorer == "ridge":
                self.p = self.X.shape[1]
                self.Id = np.identity(self.p)
                self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
            elif self.scorer == "knn":
                self.D = self.distance_func(self.X)
                self._label_indices = {0: np.flatnonzero(self.y == 0), 1: np.flatnonzero(self.y == 1)}
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
                self._label_indices = {0: np.flatnonzero(self.y == 0), 1: np.flatnonzero(self.y == 1)}

    def predict(self, x, return_update=False):
        """Produce a Venn-Abers multi-probability prediction.

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.
        return_update : bool
            If True, return precomputed state for efficient learn_one.

        Returns
        -------
        prediction : VennAbersPrediction
            Contains p0, p1, and point prediction.
        precomputed : dict, optional
            Returned if return_update=True.
        """
        x = np.asarray(x).ravel()

        if self.X is None or len(self.y) == 0:
            pred = VennAbersPrediction(0.5, 0.5)
            if return_update:
                return pred, {}
            return pred

        if self.scorer == "ridge":
            return self._predict_ridge(x, return_update)
        elif self.scorer == "knn":
            return self._predict_knn(x, return_update)

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

        pred = VennAbersPrediction(p0, p1)

        if return_update:
            return pred, {"XTXinv": XTXinv_aug}
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

        pred = VennAbersPrediction(results[0], results[1])

        if return_update:
            return pred, {"D": D_aug}
        return pred

    @staticmethod
    def _compute_knn_scores(D, labels, k, agg_func):
        """Compute k-NN nonconformity scores: agg(d_same) - agg(d_diff).

        For each point i, compute the aggregated distance to k nearest
        same-class neighbours minus aggregated distance to k nearest
        different-class neighbours. Higher score = more likely class 1.

        We use d_diff - d_same so that higher scores correspond to higher
        P(y=1) when y=1 points cluster together.
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


if __name__ == "__main__":
    import doctest
    import sys

    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
