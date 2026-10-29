r"""Online conformal regressors.

Conformal regressors turn any point predictor into an interval predictor with a
finite-sample, distribution-free coverage guarantee. Given a *nonconformity
measure* $A$ that scores how unusual a labelled example $(x, y)$ looks relative
to the others, the conformal predictor outputs the set of candidate labels whose
conformal p-value exceeds the significance level $\epsilon$. Under
exchangeability of the data stream this set is *valid*:

$$
\mathbb{P}\bigl(y_n \in \Gamma^\epsilon(x_n)\bigr) \geq 1 - \epsilon ,
$$

and in the online (sequential) setting the long-run error rate converges to
$\epsilon$ (the predictor is *well calibrated*; see [ALRW2 Ch.2]).

The regressors differ only in their underlying point predictor and hence in the
nonconformity measure they induce:

- :class:`ConformalRidgeRegressor` — ridge regression; NCM is the (optionally
  studentised) residual $|y - \hat y|$.
- :class:`ConformalNearestNeighboursRegressor` — $k$-NN; NCM is the
  leave-one-out $k$-NN residual.
- :class:`KernelConformalRidgeRegressor` — kernel ridge regression in an RKHS.
- :class:`ConformalLassoRegressor` — Lasso / elastic net via exact homotopy.

References
----------
[ALRW2] Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*,
2nd ed., Springer, 2022.
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    from ._serialization import SerializableMixin
except ImportError:
    from _serialization import SerializableMixin

try:
    from ._pruning import _ReservoirBuffer, _evict_distance_matrix, _evict_masked_distance
except ImportError:
    from _pruning import _ReservoirBuffer, _evict_distance_matrix, _evict_masked_distance

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform

# Optional numba for Lasso homotopy speedup
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


from online_cp.mondrian._forest_solver import forest_sweepline_solver
from online_cp.mondrian._inspection import _MondrianRegressorInspection
from online_cp.mondrian.tree import (
    MondrianTree,
    _build_tree_summary_reg,
    _collect_leaves,
    _find_leaf,
    _MondrianNode,
    _summarize_tree_reg,
)
from online_cp.prediction_set import (
    ContinuousPredictionSet,
    DiscretePredictionSet,
    EmptyPredictionSet,
    MultiLevelPredictionSet,
)

__all__ = [
    "ConformalRidgeRegressor",
    "ConformalNearestNeighboursRegressor",
    "KernelConformalRidgeRegressor",
    "ConformalLassoRegressor",
    "ContinuousPredictionSet",
    "DiscretePredictionSet",
    "EmptyPredictionSet",
    "MultiLevelPredictionSet",
    "ConformalPredictionInterval",  # deprecated
    "MultiLevelPredictionInterval",  # deprecated
    "ConformalMondrianTreeRegressor",
    "ConformalMondrianForestRegressor",
]


default_epsilon = 0.1


def _rank1_inverse_update(inv, x):
    r"""Return ``(inv - inv xx' inv) / (1 + x' inv x)``, symmetrized.

    This is the inverse of ``A + x x'`` for the positive-definite matrix ``A``
    whose inverse is ``inv`` (the matrix-inversion lemma). The update is
    symmetrized as ``(M + M') / 2`` after each step to counteract the asymmetric
    round-off that accumulates when this rank-1 update is applied repeatedly
    over a long online stream; symmetry is an invariant of every intermediate
    ``A^{-1}`` since ``A`` is symmetric.
    """
    v = inv @ x
    M = inv - np.outer(v, v) / (1.0 + x @ v)
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# Deprecated wrappers
#
# The old ``ConformalPredictionInterval`` (regressor) and
# ``MultiLevelPredictionInterval`` (regressor) classes have been unified into
# the ``ContinuousPredictionSet`` / ``MultiLevelPredictionSet`` hierarchy in
# :mod:`online_cp.prediction_set`. The names below are kept for backward
# compatibility and emit a :class:`DeprecationWarning` on construction.
# ---------------------------------------------------------------------------
def _deprecation_msg(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated and will be removed in a future release. "
        f"Use `{new}` from `online_cp.prediction_set` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


class ConformalPredictionInterval(ContinuousPredictionSet):
    """Deprecated. Use :class:`ContinuousPredictionSet` instead.

    Parameters are ``(lower, upper, epsilon)`` for backward compatibility.
    ``.lower`` / ``.upper`` are preserved as read-only properties.
    """

    def __init__(self, lower: float, upper: float, epsilon: float) -> None:
        _deprecation_msg("ConformalPredictionInterval", "ContinuousPredictionSet")
        super().__init__([(lower, upper)], epsilon)

    @property
    def lower(self) -> float:
        """Lower bound of the (single) interval. Preserved for backward compat."""
        return self.intervals[0][0] if self.intervals else float("nan")

    @property
    def upper(self) -> float:
        """Upper bound of the (single) interval. Preserved for backward compat."""
        return self.intervals[-1][1] if self.intervals else float("nan")

    def __repr__(self):
        if self.is_empty:
            return "()"
        return f"({self.lower}, {self.upper})"

    def __str__(self):
        if self.is_empty:
            return "()"
        return f"({self.lower}, {self.upper})"


class MultiLevelPredictionInterval(MultiLevelPredictionSet):
    """Deprecated. Use :class:`MultiLevelPredictionSet` instead."""

    def __init__(self, predictions: dict[float, ContinuousPredictionSet]) -> None:
        _deprecation_msg("MultiLevelPredictionInterval", "MultiLevelPredictionSet")
        super().__init__(predictions)

    def __repr__(self):
        parts = [f"  ε={eps}: {pset!r}" for eps, pset in self._predictions.items()]
        return "MultiLevelPredictionInterval(\n" + "\n".join(parts) + "\n)"


class ConformalRegressor(SerializableMixin):
    """Base class for online conformal regressors.

    Provides shared methods for computing p-values, constructing prediction
    intervals, and processing datasets in the online setting.
    """

    _SAVE_PARAMS: tuple = ("epsilon",)
    _SAVE_STATE: tuple = ()

    def __init__(self, epsilon: float | NDArray[np.floating[Any]] = default_epsilon) -> None:
        self.epsilon = epsilon

    def _construct_Gamma(self, Gamma_val, epsilon):
        """Construct a prediction set from the solver output.

        For single-interval mode (default): Gamma_val is (lo, hi) tuple
        For multi-interval mode: Gamma_val is list of (lo, hi) tuples
        For empty set: Gamma_val is (nan, nan) tuple or empty list []
        """
        if isinstance(Gamma_val, tuple):
            # Single interval case: (lo, hi) or (nan, nan) for empty
            lo, hi = Gamma_val
            if np.isnan(lo) and np.isnan(hi):
                # Return empty ContinuousPredictionSet for backward compatibility
                # with tests that expect .lower and .upper properties
                return ContinuousPredictionSet([], epsilon)
            else:
                return ContinuousPredictionSet([(lo, hi)], epsilon)
        else:
            # Multiple intervals case: list of (lo, hi) tuples
            intervals = Gamma_val
            if not intervals:
                return ContinuousPredictionSet([], epsilon)
            return ContinuousPredictionSet(intervals, epsilon)

    @staticmethod
    def _safe_size_check(X):
        if X is None:
            size = 0
        else:
            size = X.shape[0]
        return size

    @staticmethod
    def _compute_p_value(Alpha, tau=None, c_type="nonconformity"):
        """
        Method to compute the smoothed p-value, given an array of nonconformity scores where the last element corresponds
        to the test object, and a random number tau. If tau is None, the non-smoothed p-value is returned.
        """
        if c_type == "nonconformity":
            alpha_y = Alpha[-1]
            if tau is not None:
                gt = np.where(Alpha > alpha_y)[0].size
                eq = np.where(Alpha == alpha_y)[0].size
                p_y = (gt + tau * eq) / Alpha.size
            else:
                geq = np.where(Alpha >= alpha_y)[0].size
                p_y = geq / Alpha.size
        elif c_type == "conformity":
            alpha_y = Alpha[-1]
            if tau is not None:
                lt = np.where(Alpha < alpha_y)[0].size
                eq = np.where(Alpha == alpha_y)[0].size
                p_y = (lt + tau * eq) / Alpha.size
            else:
                leq = np.where(Alpha <= alpha_y)[0].size
                p_y = leq / Alpha.size
        else:
            raise ValueError(f"c_type must be 'nonconformity' or 'conformity', got '{c_type}'")
        return p_y

    @staticmethod
    def _get_upper(u_dic, epsilon, n):
        r"""Upper bound from Algorithm 2.2: the order statistic $t_{(\lceil(1-\epsilon)n\rceil)}$.

        ``u_dic`` maps 1-based rank ``k`` (``1 .. n - 1``) to the ``k``-th
        smallest training cut-point. The upper bound is the order statistic at
        rank ``k = ceil((1 - epsilon) * n)``.

        Parameters
        ----------
        u_dic : dict
            Mapping ``rank -> value`` for the upper order statistic, ranks
            ``1 .. n - 1`` (see :meth:`_vectorised_l_and_u`).
        epsilon : float
            Significance level for this tail (callers pass ``epsilon / 2`` for a
            two-sided set).
        n : int
            Total number of points (training + test) = number of cut-points + 1.
        """
        k = int(np.ceil((1 - epsilon) * n))
        # A required rank outside the available range (k < 1 or k > n - 1) means
        # the order statistic does not exist -> unbounded ray (+inf). This is
        # the explicit form of the previous KeyError fallback.
        return u_dic.get(k, np.inf)

    @staticmethod
    def _get_lower(l_dic, epsilon, n):
        r"""Lower bound from Algorithm 2.3: the order statistic $t_{(\lfloor\epsilon n\rfloor)}$.

        ``l_dic`` maps 1-based rank ``k`` (``1 .. n - 1``) to the ``k``-th
        smallest training cut-point. The lower bound is the order statistic at
        rank ``k = floor(epsilon * n)``.

        Parameters
        ----------
        l_dic : dict
            Mapping ``rank -> value`` for the lower order statistic, ranks
            ``1 .. n - 1`` (see :meth:`_vectorised_l_and_u`).
        epsilon : float
            Significance level for this tail (callers pass ``epsilon / 2`` for a
            two-sided set).
        n : int
            Total number of points (training + test) = number of cut-points + 1.
        """
        k = int(np.floor(epsilon * n))
        # A required rank outside the available range (k < 1 or k > n - 1) means
        # the order statistic does not exist -> unbounded ray (-inf). This is
        # the explicit form of the previous KeyError fallback.
        return l_dic.get(k, -np.inf)

    @staticmethod
    def _vectorised_l_and_u(A, B):
        """Compute the lower/upper order-statistic dictionaries.

        ``A`` and ``B`` are length-``n`` columns (last element = test point).
        For each training point ``i`` the cut-point is
        ``t_i = (a_i - a_n) / (b_n - b_i)``. Two dictionaries are returned:

        - ``l_dic``: ascending-sorted cut-points with anomalous points
          (``b_n <= b_i``) sent to ``-inf`` (they can never be the lower bound),
          the overall minimum dropped. Ranks ``1 .. n - 1``.
        - ``u_dic``: ascending-sorted cut-points with anomalous points sent to
          ``+inf`` (they can never be the upper bound), the overall maximum
          dropped. Ranks ``1 .. n - 1``.

        This is the input to :meth:`_get_lower` / :meth:`_get_upper`.
        """
        differences = B[-1] - B

        # Create an array to store results
        l = np.empty_like(B, dtype=float)
        u = np.empty_like(B, dtype=float)

        # Calculate values where differences are positive
        mask = differences > 0
        l[mask] = (A[mask] - A[-1]) / differences[mask]
        u[mask] = (A[mask] - A[-1]) / differences[mask]

        # Assign infinity where differences are non-positive (anomalous case,
        # ALRW2 §2.3.2): the point's cut-point cannot be a lower bound (-inf)
        # nor an upper bound (+inf).
        l[~mask] = -np.inf
        u[~mask] = np.inf

        # Drop the test point's own entry and one extreme sentinel each, so each
        # dictionary holds exactly n - 1 values, 1-based ranked.
        l = np.sort(l, axis=0)[1:]
        u = np.sort(u, axis=0)[:-1]

        l_dic = {i + 1: val for i, val in enumerate(l)}
        u_dic = {i + 1: val for i, val in enumerate(u)}

        return l_dic, u_dic

    @staticmethod
    def minimum_training_set(epsilon, bounds="both"):
        """
        Returns the minimum initial training set size needed to output informative (finite) prediciton sets

        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(0.1)
        20

        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(0.1, "upper")
        10

        >>> import numpy as np
        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(np.array([0.1, 0.05]))
        40

        """
        if not hasattr(epsilon, "shape"):
            # Then it is a scalar
            if bounds == "both":
                return int(np.ceil(2 / epsilon))
            else:
                return int(np.ceil(1 / epsilon))
        else:
            # Then it is a vector
            if bounds == "both":
                return int(np.ceil(2 / epsilon.min()))
            else:
                return int(np.ceil(1 / epsilon.min()))

    def predict_batch(
        self,
        X: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] = default_epsilon,
        **kwargs: Any,
    ) -> list:
        r"""Predict a batch of test objects.

        Default implementation: call :meth:`predict` once per row of ``X``. The
        result ``i`` is **bit-identical** to ``predict(X[i], epsilon=epsilon,
        **kwargs)`` by construction. Subclasses that can vectorise the
        nonconformity computation override this to be faster while still
        returning the same values.

        Parameters
        ----------
        X : ndarray, shape (m, d)
            ``m`` test objects, one per row.
        epsilon : float or array-like
            Significance level (or levels). Passed through to :meth:`predict`.
        **kwargs
            Forwarded to :meth:`predict` (e.g. ``bounds``).

        Returns
        -------
        list
            A list of ``m`` prediction sets, ``results[i] == predict(X[i], ...)``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return [self.predict(x, epsilon=epsilon, **kwargs) for x in X]


class ConformalRidgeRegressor(ConformalRegressor):
    r"""Conformal ridge regression ([ALRW2 §2.3], Algorithm 2.4).

    Online conformal predictor built on ridge regression. The nonconformity
    measure is the residual $\alpha_i = |y_i - \hat y_i|$ of the ridge fit; with
    ``studentised=True`` it is divided by $\sqrt{1 - h_{ii}}$ (the leverage
    correction), which makes the scores more exchangeable when leverages differ.
    Because ridge residuals are an affine function of the hypothesised test
    label, the whole prediction interval is computed in closed form (no grid
    search) and the design-matrix inverse is maintained online via the
    Sherman–Morrison update.

    Under exchangeability the resulting intervals satisfy
    $\mathbb{P}(y \in \Gamma^\epsilon(x)) \geq 1 - \epsilon$.

    Let's create a dataset with noisy evaluations of the function f(x1,x2) = x1+x2:

    >>> import numpy as np
    >>> np.random.seed(31337)  # only needed for doctests
    >>> N = 30
    >>> X = np.random.uniform(0, 1, (N, 2))
    >>> y = X.sum(axis=1) + np.random.normal(0, 0.1, N)

    Import the library and create a regressor:

    >>> cp = ConformalRidgeRegressor()

    Learn the whole dataset:

    >>> cp.learn_initial_training_set(X, y)

    Predict an object (output may not be exactly the same, as the dataset
    depends on the random seed):
    >>> interval = cp.predict(np.array([0.5, 0.5]), bounds="both")
    >>> print("(%.2f, %.2f)" % (interval.lower, interval.upper))
    (0.73, 1.23)

    You can of course learn a new data point online:

    >>> cp.learn_one(np.array([0.5, 0.5]), 1.0)

    The prediction set is the closed interval whose boundaries are indicated by the output.

    We can then predict again:

    >>> interval = cp.predict(np.array([2, 4]), bounds="both")
    >>> print("(%.2f, %.2f)" % (interval.lower, interval.upper))
    (5.39, 6.33)
    """

    _SAVE_PARAMS: tuple = (
        "a",
        "warnings",
        "autotune",
        "verbose",
        "studentised",
        "epsilon",
        "recompute_every",
        "rnd_state",
    )
    _SAVE_STATE: tuple = ("X", "y", "p", "Id", "XTXinv", "_n_sm_updates")

    def __init__(
        self,
        a=0,
        warnings=True,
        autotune=False,
        verbose=0,
        rnd_state=None,
        studentised=False,
        epsilon=default_epsilon,
        recompute_every: int | None = None,
    ) -> None:
        r"""Create a conformal ridge regressor.

        Parameters
        ----------
        a : float, default 0
            Ridge (L2) regularisation parameter. ``a > 0`` is required if
            $X^\top X$ is singular (e.g. fewer examples than features).
        warnings : bool, default True
            Whether to emit numerical-stability warnings (e.g. ill-conditioned
            inverse, significance level too small for the training set).
        autotune : bool, default False
            If True, tune ``a`` by generalized cross-validation when
            :meth:`learn_initial_training_set` is called.
        verbose : int, default 0
            Verbosity level.
        rnd_state : int, np.random.Generator, or None, default None
            Seed or Generator for the random number generator used to draw the smoothing
            variable $\tau$ for smoothed p-values.
        studentised : bool, default False
            If True, use studentised residuals $|y - \hat y| / \sqrt{1 - h_{ii}}$
            as the nonconformity measure instead of raw residuals.
        epsilon : float, default 0.1
            Default significance level.
        recompute_every : int or None, default None
            If set, recompute the full matrix inverse from scratch every N
            Sherman-Morrison updates to correct accumulated floating-point drift.
        """
        super().__init__(epsilon=epsilon)

        self.a = a
        self.X = None
        self.y = None
        self.p = None
        self.Id = None
        self.XTXinv = None

        # Should we raise warnings
        self.warnings = warnings
        # Do we autotune ridge prarmeter on warning
        self.autotune = autotune

        self.verbose = verbose
        self.rnd_state = rnd_state
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
            self.rnd_gen = np.random.default_rng(rnd_state)

        # Do we use the studentised residuals
        self.studentised = studentised

        # Sherman-Morrison stability
        self.recompute_every = recompute_every
        self._n_sm_updates = 0

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        """Batch-fit the ridge model on an initial training set.

        Stores ``X``/``y`` and computes the design-matrix inverse
        $(X^\\top X + aI)^{-1}$ that is updated online thereafter. If
        ``autotune`` is set, the ridge parameter is tuned first.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training objects.
        y : ndarray of shape (n,)
            Training responses.
        """
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.Id = np.identity(self.p)
        self._n_sm_updates = 0
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            try:
                self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise, "
                    "or provide more linearly independent training examples."
                ) from None

    def recompute_inverse(self) -> None:
        """Recompute (X^T X + aI)^{-1} from scratch to correct numerical drift.

        Call this periodically during long online streams, or set
        ``recompute_every`` in the constructor for automatic periodic
        recomputation.

        >>> import numpy as np
        >>> cp = ConformalRidgeRegressor(a=1)
        >>> cp.learn_initial_training_set(np.eye(3), np.array([1.0, 2.0, 3.0]))
        >>> cp.recompute_inverse()
        >>> np.allclose(cp.XTXinv, np.linalg.inv(cp.X.T @ cp.X + cp.a * cp.Id))
        True
        """
        if self.X is None or self.Id is None:
            return
        try:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
        except np.linalg.LinAlgError:
            raise ValueError(
                "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise, "
                "or provide more linearly independent training examples."
            ) from None
        self._n_sm_updates = 0

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a single example. If we have already computed X and XTXinv, use them for update. Then the last row of X is the object with label y.
        >>> cp = ConformalRidgeRegressor()
        >>> cp.learn_one(np.array([1, 0]), 1)
        >>> cp.X
        array([[1, 0]])
        >>> cp.y
        array([1])
        """
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)

        if precomputed is not None:
            X = precomputed["X"]
            XTXinv = precomputed["XTXinv"]

            if X is not None:
                self.X = X
                self.p = self.X.shape[1]
                self.Id = np.identity(self.p)

            if XTXinv is not None:
                self.XTXinv = XTXinv

            else:
                if self.X.shape[0] == 1:
                    try:
                        self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
                    except np.linalg.LinAlgError:
                        raise ValueError(
                            "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise, "
                            "or provide more linearly independent training examples."
                        ) from None
                else:
                    # Update XTX_inv using the matrix-inversion lemma
                    # (inverse of (X'X + aI) -> inverse of ((X+x)(X+x)' + aI)).
                    self.XTXinv = _rank1_inverse_update(self.XTXinv, x)
                    self._n_sm_updates += 1

                    if self.recompute_every and self._n_sm_updates % self.recompute_every == 0:
                        self.recompute_inverse()
                    elif self.warnings:
                        cond = np.linalg.cond(self.XTXinv)
                        if cond > 1e12:
                            warnings.warn(
                                f"(X^T X + aI)^{{-1}} has condition number {cond:.2e}. "
                                f"Consider calling recompute_inverse() or setting recompute_every.",
                                stacklevel=2,
                            )

        else:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1, -1)
                self.p = self.X.shape[1]
                self.Id = np.identity(self.p)
            elif self.X.shape[0] == 1:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                try:
                    self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
                except np.linalg.LinAlgError:
                    raise ValueError(
                        "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise, "
                        "or provide more linearly independent training examples."
                    ) from None
            else:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                # Update XTX_inv using the matrix-inversion lemma
                # (inverse of (X'X + aI) -> inverse of ((X+x)(X+x)' + aI)).
                self.XTXinv = _rank1_inverse_update(self.XTXinv, x)
                self._n_sm_updates += 1

                if self.recompute_every and self._n_sm_updates % self.recompute_every == 0:
                    self.recompute_inverse()
                elif self.warnings:
                    cond = np.linalg.cond(self.XTXinv)
                    if cond > 1e12:
                        warnings.warn(
                            f"(X^T X + aI)^{{-1}} has condition number {cond:.2e}. "
                            f"Consider calling recompute_inverse() or setting recompute_every.",
                            stacklevel=2,
                        )

    def compute_A_and_B(self, X, XTXinv, y):
        """Compute A and B vectors for conformal ridge regression.

        Given the augmented design matrix X (last row = test object) and its
        corresponding (X^T X + aI)^{-1}, compute the vectors A and B such that
        the nonconformity score of point i at hypothesised test label y is
        ``A[i] + y * B[i]``.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Augmented design matrix. Last row is the test object.
        XTXinv : ndarray, shape (d, d)
            Inverse of X.T @ X + a*I for the augmented X.
        y : ndarray, shape (n-1,)
            Training labels (test label is not included; appended as 0
            internally).
        """
        y_ext = np.append(y, 0)  # y with test point (last row) as 0

        # Compute beta using the augmented X and y_ext (just like the old code)
        beta = XTXinv @ X.T @ y_ext  # (d, d) @ (d, n) @ (n,) -> (d,)

        # Fitted values for all points (including test)
        y_hat = X @ beta  # (n, d) @ (d,) -> (n,)

        # Compute hat matrix diagonal for all points using XTXinv (augmented)
        H_diag = np.sum(X @ XTXinv * X, axis=1)  # (n,)

        # Compute last column of H efficiently
        h_col = X @ XTXinv @ X[-1]  # (n, d) @ (d,) -> (n,)

        # A and B for each point
        A = y_ext - y_hat
        B = -h_col
        B[-1] += 1  # e_{-1}[-1] = 1

        if self.studentised:
            leverage = np.sqrt(np.clip(1 - H_diag, 1e-12, None))
            A = A / leverage
            B = B / leverage

        return A, B

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        bounds: str = "both",
        return_update: bool = False,
    ) -> (
        ContinuousPredictionSet
        | MultiLevelPredictionSet
        | tuple[ContinuousPredictionSet | MultiLevelPredictionSet, dict[str, Any]]
    ):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), bounds="both")
        ContinuousPredictionSet([(-inf, inf)])
        """

        def build_precomputed(X, XTXinv, A, B):
            computed = {
                "X": X,  # The updated matrix of objects
                "XTXinv": XTXinv,  # The updated kernel matrix
                "A": A,
                "B": B,
            }
            return computed

        if epsilon is None:
            epsilon = self.epsilon

        if self._safe_size_check(self.X) > 0:
            # Add row to X matrix
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            n = X.shape[0]
            XTXinv = None

            # Check that the significance level is not too small. If it is, return infinite prediction interval
            # For multi-level: only bail out if even the largest epsilon is too small
            eps_check = max(epsilon) if hasattr(epsilon, "__iter__") else epsilon
            if bounds == "both":
                if not (eps_check >= 2 / n):
                    if self.warnings:
                        eps_warn = min(epsilon) if hasattr(epsilon, "__iter__") else epsilon
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(2 / eps_warn))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if hasattr(epsilon, "__iter__"):
                        predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                        result = MultiLevelPredictionSet(predictions)
                    else:
                        result = self._construct_Gamma((-np.inf, np.inf), epsilon)
                    if return_update:
                        return result, build_precomputed(X, XTXinv, None, None)
                    else:
                        return result
            else:
                if not (eps_check >= 1 / n):
                    if self.warnings:
                        eps_warn = min(epsilon) if hasattr(epsilon, "__iter__") else epsilon
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(1 / eps_warn))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if hasattr(epsilon, "__iter__"):
                        predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                        result = MultiLevelPredictionSet(predictions)
                    else:
                        result = self._construct_Gamma((-np.inf, np.inf), epsilon)
                    if return_update:
                        return result, build_precomputed(X, XTXinv, None, None)
                    else:
                        return result

            # Update XTX_inv using Sherman-Morrison formula
            XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

            A, B = self.compute_A_and_B(X, XTXinv, self.y)

            if self.studentised:
                diffs = B[-1] - B[:-1]
                safe = np.abs(diffs) > 1e-12
                t = np.where(safe, (A[:-1] - A[-1]) / diffs, np.inf)
                t.sort()
                l_dic = {i + 1: val for i, val in enumerate(t)}
                u_dic = {i + 1: val for i, val in enumerate(t)}
            else:
                l_dic, u_dic = self._vectorised_l_and_u(A, B)

            if bounds == "both":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        lo = self._get_lower(l_dic=l_dic, epsilon=eps / 2, n=n)
                        up = self._get_upper(u_dic=u_dic, epsilon=eps / 2, n=n)
                        predictions[eps] = self._construct_Gamma((lo, up), eps)
                    result = MultiLevelPredictionSet(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
                    result = self._construct_Gamma((lower, upper), epsilon)
            elif bounds == "lower":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        lo = self._get_lower(l_dic=l_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma((lo, np.inf), eps)
                    result = MultiLevelPredictionSet(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma((lower, np.inf), epsilon)
            elif bounds == "upper":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        up = self._get_upper(u_dic=u_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma((-np.inf, up), eps)
                    result = MultiLevelPredictionSet(predictions)
                else:
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma((-np.inf, upper), epsilon)
            else:
                raise ValueError('bounds must be "both", "lower", or "upper"')
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            X = x.reshape(1, -1)
            XTXinv = None
            A = None
            B = None

            if hasattr(epsilon, "__iter__"):
                predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                result = MultiLevelPredictionSet(predictions)
            else:
                result = self._construct_Gamma((-np.inf, np.inf), epsilon)

        if return_update:
            return result, build_precomputed(X, XTXinv, A, B)
        else:
            return result

    def compute_p_value(self, x, y, bounds="both", precomputed=None, tau=None, smoothed=True):
        r"""Conformal p-value of the candidate example $(x, y)$.

        The p-value is the (smoothed) rank of the test example's nonconformity
        score among all scores; a candidate label $y$ is included in
        $\Gamma^\epsilon(x)$ iff its p-value exceeds $\epsilon$.

        Parameters
        ----------
        x : ndarray of shape (d,)
            Test object.
        y : float
            Candidate response.
        bounds : {"both", "lower", "upper"}, default "both"
            Which tail(s) of the conformity region to score.
        precomputed : dict or None
            Cached ``{X, XTXinv, A, B}`` from :meth:`predict` (with
            ``return_update=True``) to avoid recomputation.
        tau : float or None
            Smoothing variable in $[0, 1]$. If None and ``smoothed`` is True it
            is drawn uniformly at random.
        smoothed : bool, default True
            Whether to return the smoothed (exactly valid) p-value.

        Returns
        -------
        float
            The conformal p-value.
        """
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            if not np.allclose(x, precomputed["X"][-1]):
                raise ValueError("x does not match the last row of precomputed['X']")
            A = precomputed["A"]
            B = precomputed["B"]
        else:
            if self.XTXinv is not None:
                X = np.append(self.X, x.reshape(1, -1), axis=0)
                XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
                A, B = self.compute_A_and_B(X, XTXinv, self.y)
            else:
                A, B = None, None

        if A is not None and B is not None:
            if bounds == "both":
                E = A + y * B
                Alpha = np.zeros_like(A)
                for i, e in enumerate(E):
                    alpha = min((E >= e).sum(), (E <= e).sum())
                    Alpha[i] = alpha
                c_type = "conformity"
            elif bounds == "lower":
                Alpha = -(A + y * B)
                c_type = "nonconformity"
            elif bounds == "upper":
                Alpha = A + y * B
                c_type = "nonconformity"
            else:
                raise ValueError('bounds must be "both", "lower", or "upper"')

            if smoothed:
                p = self._compute_p_value(Alpha, tau, c_type=c_type)
            else:
                p = self._compute_p_value(Alpha, c_type=c_type)
        else:
            if smoothed:
                p = tau
            else:
                p = 1

        return p

    def change_ridge_parameter(self, a):
        """
        Change the ridge parameter
        >>> cp = ConformalRidgeRegressor()
        >>> cp.learn_one(np.array([1, 0]), 1)
        >>> cp.change_ridge_parameter(1)
        >>> cp.a
        1
        """
        self.a = a
        if self.X is not None:
            try:
                self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "X^T X + aI is singular. Set a > 0 (ridge parameter) to regularise, "
                    "or provide more linearly independent training examples."
                ) from None
            self._n_sm_updates = 0

    def _tune_ridge_parameter(self, a0=None):
        """
        Tune ridge parameter with Generalized cross validation
        (https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/golub.heath.wahba.pdf).

        The GCV score only depends on the ridge hat matrix
        ``A = X (X^T X + a I)^{-1} X^T`` through ``||y - A y||`` and
        ``trace(A)``.  We factor ``X^T X`` **once** into its eigendecomposition
        (``X^T X = V diag(lam) V^T``); for a ridge parameter ``a`` the hat
        matrix is ``A = (X V) diag(lam / (lam + a)) (X V)^T``, so evaluating the
        GCV at any ``a`` reduces to a handful of vector operations (O(n p))
        instead of a fresh p x p matrix inverse per evaluation (O(p^3)).  The
        one-dimensional GCV is then minimised with :func:`minimize_scalar`
        (bound method) rather than a general-purpose minimiser.

        The result is identical to the original formulation (same GCV, same
        ``a >= 1e-6`` bound, same ``max_diag_H > 1`` guard); only the
        evaluation order changes, so the minimised ``a`` is (up to optimiser
        tolerance) the same.
        """
        XTX = self.X.T @ self.X
        n = self.X.shape[0]
        y = self.y

        # One-time eigendecomposition of X^T X (symmetric, p x p).
        lam_eig, V = np.linalg.eigh(XTX)
        # Columns of W = X V;  W^T W = diag(lam_eig) (orthonormal up to lam).
        W = self.X @ V
        # Precompute V^T (X^T y) and W^T y so the per-a work is just vector ops.
        Xt_y = self.X.T @ y  # (p,)
        Vt_Xt_y = V.T @ Xt_y  # (p,)

        def GCV(a):
            # Hat-matrix eigenvalues are lam / (lam + a) in (0, 1) for a > 0.
            denom = lam_eig + a
            # trace(A) = sum_j lam_j / (lam_j + a)
            trA = np.sum(lam_eig / denom)
            # A y = W ( diag(lam/denom) @ (V^T X^T y) ) ... see docstring.
            #   A y = (X V) diag(1/(lam+a)) (V^T) (X^T y)
            #        = W @ ( (Vt_Xt_y) / denom )
            A_y = W @ (Vt_Xt_y / denom)
            # ||(I - A) y||^2 = ||y - A y||^2
            r = y - A_y
            num = (1.0 / n) * float(r @ r)
            den = (1.0 / n) * (n - trA)
            if den <= 0.0:
                return np.inf
            # Equivalent to the original max_diag_H > 1 guard (inert for ridge,
            # kept for safety): A = W diag(1/(lam+a)) W^T, so the i-th hat
            # diagonal is sum_j W_ij^2 / (lam_j + a).
            max_diag_H = np.max(np.einsum("ij,j,ij->i", W, 1.0 / denom, W))
            if max_diag_H > 1.0:
                return np.inf
            return num / (den * den)

        # Upper bound for the bounded 1-D minimiser.  The original used an
        # unbounded Nelder-Mead with a lower bound only; here we need a finite
        # upper bound.  For ridge the GCV plateaus once a is large enough that
        # the hat matrix is negligible, and the optimum always lies below
        # O(n * var(y)) (the scale at which the fit is essentially the mean).
        upper = max(1.0, n * max(float(np.var(y)), 1e-12))

        # Bounds to ensure a >= 0
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(GCV, bounds=(1e-6, upper), method="bounded", options={"xatol": 1e-8})
        a = res.x

        if self.verbose > 0:
            print(f"New ridge parameter: {a}")
        self.change_ridge_parameter(a)


class ConformalNearestNeighboursRegressor(ConformalRegressor):
    """
    Conformal k-nearest neighbours regressor (ALRW2 §2.4).

    Produces prediction intervals with guaranteed coverage using the
    nonconformity measure |y - ŷ_kNN| where ŷ_kNN is the leave-one-out
    k-nearest neighbours prediction.

    This implementation uses an optimized exact algorithm with:
    - Cached k-th nearest neighbor distances for efficient reverse-neighbour detection
    - O(n) prediction time (instead of O(n²) in the naive implementation)

    For very large datasets, consider using a k-d tree for O(n log n) complexity
    (not yet implemented, but the exact path is highly optimized).

    >>> import numpy as np
    >>> np.random.seed(42)
    >>> N = 30
    >>> X = np.random.uniform(0, 1, (N, 1))
    >>> y = np.sin(2 * np.pi * X[:, 0]) + np.random.normal(0, 0.1, N)
    >>> cp = ConformalNearestNeighboursRegressor(k=3)
    >>> cp.learn_initial_training_set(X, y)
    >>> interval = cp.predict(np.array([0.25]))
    >>> bool(interval.lower < np.sin(2 * np.pi * 0.25) < interval.upper)
    True
    """

    _SAVE_PARAMS: tuple = (
        "k",
        "distance",
        "distance_func",
        "aggregation",
        "verbose",
        "rnd_state",
        "epsilon",
        "pruning",
        "budget",
    )
    _SAVE_STATE: tuple = ("X", "y", "D", "_a_const", "_kth_distances", "_reservoir_n_seen")
    _SAVE_CALLABLES: tuple = ("distance_func",)
    _PARAM_MAP: dict = {"distance_func": "_distance_func_arg"}

    def __init__(
        self,
        k=1,
        distance="euclidean",
        distance_func=None,
        aggregation="mean",
        verbose=0,
        rnd_state=None,
        epsilon=default_epsilon,
        pruning: str | None = None,
        budget: int | None = None,
    ):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours.
        distance : str
            Distance metric (passed to scipy.spatial.distance).
        distance_func : callable, optional
            Custom distance function. If provided, `distance` is ignored.
            Signature: distance_func(X, y=None) where X is (n, d) and y is
            (m, d) or None. Returns (n, n) if y is None, else (n, m).
        aggregation : {'mean', 'median'}
            How to aggregate k nearest neighbour labels.
        verbose : int, default 0
            Verbosity level.
        rnd_state : int, np.random.Generator, or None, default None
            Seed or Generator for the random number generator.
        epsilon : float
            Default significance level.
        pruning : {'reservoir', 'window'} or None, default None
            Stream-pruning strategy that bounds the retained training bag.
            Both strategies preserve exchangeability (and therefore the exact
            finite-sample coverage guarantee):
            ``'reservoir'`` keeps a uniform random sample of ``budget`` examples;
            ``'window'`` keeps the most recent ``budget`` examples (FIFO). The
            window is less robust than the reservoir on drifting / non-i.i.d.
            streams but is trivial to maintain. ``None`` (default) keeps every
            example. See :mod:`online_cp._pruning`.
        budget : int or None, default None
            Retained-bag size for reservoir pruning. Must be set together with
            ``pruning='reservoir'``. With ``budget >= n`` seen, behaviour is
            identical to no pruning (exact, not just close).
        """
        super().__init__(epsilon=epsilon)
        self.k = k
        if aggregation not in ("mean", "median"):
            raise ValueError(f"aggregation must be 'mean' or 'median', got '{aggregation}'")
        self.aggregation = aggregation
        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = "custom"
        self.X = None
        self.y = None
        self.D = None
        self.verbose = verbose
        self.rnd_state = rnd_state
        self._distance_func_arg = distance_func
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
            self.rnd_gen = np.random.default_rng(rnd_state)

        # --- Reservoir pruning (budget-bounded stream) ---
        self.pruning = pruning
        self.budget = budget
        if pruning is not None and pruning not in ("reservoir", "window"):
            raise ValueError(
                f"pruning must be 'reservoir', 'window', or None, got {pruning!r}. "
                "Other strategies (sv, leverage, ...) select on model state and "
                "break exchangeability; they are not yet implemented."
            )
        if pruning in ("reservoir", "window"):
            if budget is None:
                raise ValueError(f"pruning={pruning!r} requires a positive integer budget")
        self._reservoir = (
            _ReservoirBuffer(int(budget), self.rnd_gen) if pruning is not None else None
        )
        self._reservoir_n_seen = 0

    def _standard_distance_func(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            return squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            return cdist(X, y, metric=self.distance)

    @staticmethod
    def _update_distance_matrix(D, d):
        """Extend (n, n) distance matrix to (n+1, n+1) given new distances d."""
        d = np.asarray(d).ravel()
        n = D.shape[0]
        D_new = np.empty((n + 1, n + 1), dtype=np.result_type(D.dtype, d.dtype))
        D_new[:n, :n] = D
        D_new[:n, n] = d
        D_new[n, :n] = d
        D_new[n, n] = 0.0
        return D_new

    def _knn_predictions(self, D, y):
        """
        Compute leave-one-out k-NN predictions for all points.

        For point i, the prediction is the aggregation of the labels of the
        k nearest neighbours of i (excluding i itself).

        Parameters
        ----------
        D : ndarray, shape (n, n)
            Distance matrix.
        y : ndarray, shape (n,)
            Labels.

        Returns
        -------
        y_hat : ndarray, shape (n,)
            Leave-one-out k-NN predictions.
        """
        n = D.shape[0]
        k = min(self.k, n - 1)  # can't have more neighbours than n-1
        if k == 0:
            return np.zeros(n)

        agg_func = np.mean if self.aggregation == "mean" else np.median

        # Set diagonal to inf so a point is never its own neighbour
        D_work = D.copy()
        np.fill_diagonal(D_work, np.inf)

        # Find k nearest neighbour indices for all points at once.
        # Use lexsort (distance primary, y-value secondary) so that ties in
        # distance are broken canonically by y rather than by insertion order.
        # This makes the result invariant to the order in which training points
        # were added (np.argpartition is unstable on ties).
        knn_idx = np.array([np.lexsort((y, D_work[i]))[:k] for i in range(n)])

        # Gather neighbour labels and aggregate
        y_neighbours = y[knn_idx]  # shape (n, k)
        y_hat = agg_func(y_neighbours, axis=1)
        return y_hat

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        """Batch-initialize with training data.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Training objects.
        y : ndarray, shape (n,)
            Training labels.
        """
        self.X = np.atleast_2d(X)
        self.y = np.asarray(y, dtype=float)
        self.D = self.distance_func(self.X)
        # Reservoir pruning: seed the reservoir from a uniform random subset of
        # the initial bag (valid when the bag is a sample of the stream). When
        # n <= budget, keep everything (exact, no pruning occurs yet).
        if self._reservoir is not None:
            n = self.X.shape[0]
            self._reservoir_n_seen = n
            if n > self.budget:
                if self.pruning == "reservoir":
                    keep = np.sort(self.rnd_gen.choice(n, size=self.budget, replace=False))
                else:  # window: keep the most recent `budget` examples
                    keep = np.arange(n - self.budget, n)
                self.X = self.X[keep]
                self.y = self.y[keep]
                self.D = _evict_masked_distance(self.D, keep)
        # Cache constant parts of affine scores: a_i = y_i - ŷ_i (with y_n=0)
        self._update_residuals()
        # Cache k-th nearest distances for efficient reverse-neighbour detection
        self._update_kth_distances()

    def _update_residuals(self) -> None:
        """Compute and cache leave-one-out k-NN residuals for conformal prediction.

        For each training point i, compute a_i = y_i - ŷ_i where ŷ_i is the
        leave-one-out k-NN prediction. This is the constant part of the affine
        score α_i(y) = a_i + b_i*y.
        """
        if self.X is None or self.y is None or self.D is None:
            return

        n = self.X.shape[0]
        if n == 0:
            return

        # Compute leave-one-out k-NN predictions on the original training set
        y_hat_train = self._knn_predictions(self.D, self.y)

        # Cache constant parts of affine scores (signed residuals, not absolute)
        self._a_const = self.y - y_hat_train

    def _update_kth_distances(self) -> None:
        """Compute and cache k-th nearest neighbor distances for each training point.

        For each training point i, compute the k-th smallest distance to other
        training points (excluding self). This is used for efficient reverse-neighbour
        detection during prediction.
        """
        if self.X is None or self.y is None or self.D is None:
            return

        n = self.X.shape[0]
        if n == 0:
            return

        k_actual = min(self.k, n - 1)
        if k_actual == 0:
            self._kth_distances = np.zeros(n)
            return

        # For each training point, get k-th nearest distance (excluding self)
        self._kth_distances = np.empty(n)
        for i in range(n):
            d_i = self.D[i].copy()
            d_i[i] = np.inf
            self._kth_distances[i] = np.partition(d_i, k_actual - 1)[k_actual - 1]

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """Incrementally add one observation.

        Parameters
        ----------
        x : array-like, shape (d,)
            New object.
        y : float
            New label.
        precomputed : dict, optional
            If provided, should contain 'D' (the already-updated distance matrix).
        """
        x = np.asarray(x).ravel()

        # Reservoir pruning: decide accept/evict *before* touching state.
        # Returns -1 (keep, no eviction), i >= 0 (keep & evict i), or None (discard).
        if self._reservoir is not None:
            self._reservoir_n_seen += 1
            eviction = self._reservoir.decide(self._reservoir_n_seen, strategy=self.pruning)
        else:
            eviction = -1

        if self.X is None:
            # First example ever: always keep (reservoir trivially holds it).
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=float)
            self.D = self.distance_func(self.X)
            # Cache residuals for efficient conformal prediction
            self._update_residuals()
            self._update_kth_distances()
            return

        # Discard: the reservoir is at capacity and this example lost the
        # accept/evict draw -- state is unchanged.
        if eviction is None:
            return

        # Append the new example (it occupies the last position).
        if precomputed is not None and "D" in precomputed:
            self.D = precomputed["D"]
        else:
            d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
            self.D = self._update_distance_matrix(self.D, d)
        self.X = np.vstack([self.X, x.reshape(1, -1)])
        self.y = np.append(self.y, y)

        # If an existing position was evicted, drop it (row/col i of D, row i of
        # X, element i of y). The appended example is at the last position and is
        # kept. After the eviction, the bag is back at size `budget`.
        if eviction >= 0:
            i = eviction
            self.D = _evict_distance_matrix(self.D, i)
            self.X = np.delete(self.X, i, axis=0)
            self.y = np.delete(self.y, i, axis=0)

        # Refresh cached residuals and k-th distances from the (possibly pruned)
        # matrices. These are rebuilt from scratch each step, so they automatically
        # track the pruned state.
        self._update_residuals()
        self._update_kth_distances()

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        bounds: str = "both",
        return_update: bool = False,
    ) -> (
        ContinuousPredictionSet
        | MultiLevelPredictionSet
        | tuple[ContinuousPredictionSet | MultiLevelPredictionSet, dict[str, Any]]
    ):
        """Predict a conformal prediction interval for test object x.

        This implements the conformal predictor from ALRW2 §2.4 using the
        deleted nonconformity scores with affine representation α_i(y) = a_i + b_i * y.

        This implementation uses an optimized exact algorithm with:
        - Cached k-th nearest neighbor distances for efficient reverse-neighbour detection
        - O(n) prediction time per call (after O(n²) preprocessing in training)

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.
        epsilon : float or array-like, optional
            Significance level(s). If None, uses self.epsilon.
        bounds : {'both', 'lower', 'upper'}
            Which bounds to compute.
        return_update : bool
            If True, also return precomputed dict for subsequent learn_one.

        Returns
        -------
        result : ContinuousPredictionSet or MultiLevelPredictionSet
        precomputed : dict, optional
            Returned if return_update=True. Contains 'D'.

        Examples
        --------
        >>> import numpy as np
        >>> from online_cp.regressors import ConformalNearestNeighboursRegressor
        >>> X = np.array([[0], [1], [2], [3], [4]])
        >>> y = np.array([0, 1, 2, 3, 4])
        >>> cp = ConformalNearestNeighboursRegressor(k=2)
        >>> cp.learn_initial_training_set(X, y)
        >>> result = cp.predict(np.array([1.5]))
        >>> isinstance(result, ContinuousPredictionSet)
        True
        """
        x = np.asarray(x).ravel()

        if epsilon is None:
            epsilon = self.epsilon

        n = self._safe_size_check(self.X)

        if n == 0:
            # No training data
            if hasattr(epsilon, "__iter__"):
                predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                result = MultiLevelPredictionSet(predictions)
            else:
                result = self._construct_Gamma((-np.inf, np.inf), epsilon)
            if return_update:
                return result, {"D": None}
            return result

        # Augment distance matrix with test point
        d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
        D_aug = self._update_distance_matrix(self.D, d)

        # Compute affine scores a_i, b_i for all training points (ALRW2 §2.4).
        #
        # x_i is a "reverse neighbour" iff x_n is among the k nearest neighbours
        # of x_i, i.e. d(x_i, x_n) is within the k smallest of {d(x_i, x_j) : j != i}.
        # For such points the k-NN prediction of x_i includes the (unknown) test
        # label, giving  alpha_i(y) = a_i + b_i*y  with b_i = -1/k.
        # Otherwise the prediction is independent of y, giving b_i = 0.
        k_actual = min(self.k, n)
        agg_func = np.mean if self.aggregation == "mean" else np.median

        # b_i: -1/k if x_n in NN_k(x_i) else 0
        # Use cached k-th nearest distances for efficient reverse-neighbour detection.
        # x_n is a reverse neighbour iff d(x_i, x_n) <= k-th nearest distance of x_i.
        b = np.zeros(n, dtype=float)
        for i in range(n):
            # d(x_i, x_n)
            d_test_i = d[i]
            # Use cached k-th nearest distance (already excludes self)
            if d_test_i <= self._kth_distances[i]:
                b[i] = -1.0 / self.k

        # Test point: a_n = -mean(k nearest training labels), b_n = 1
        # Use lexsort for canonical tie-breaking.
        knn_idx = np.lexsort((self.y, d))[:k_actual]
        y_hat_test = agg_func(self.y[knn_idx])
        a_n = -y_hat_test
        b_n = 1.0

        # For each training point, compute a_i with y_n = 0.
        # alpha_i(y) = a_i + b_i*y where y is the hypothesized test label.
        a = np.empty(n, dtype=float)
        for i in range(n):
            if b[i] == 0.0:
                # x_n not a reverse neighbour: prediction independent of y.
                # a_i = y_i - (mean of k nearest training labels of x_i)
                # This is exactly the cached leave-one-out residual.
                a[i] = self._a_const[i]
            else:
                # x_n is a reverse neighbour: the k-NN prediction of x_i in the
                # augmented set includes the test point, whose label is set to 0
                # for the constant term a_i (the y-dependence is captured by b_i).
                d_i = self.D[i].copy()
                d_i[i] = np.inf
                # Use partition to find k nearest (excluding self) among training + test
                d_full = np.append(d_i, d[i])
                y_full = np.append(self.y, 0.0)
                # Get k+1 nearest (test might be among them), then filter
                nn_idx = np.lexsort((y_full, d_full))[:k_actual]
                y_hat_i = agg_func(y_full[nn_idx])
                a[i] = self.y[i] - y_hat_i  # constant term with y_n = 0

        # Compute t_i = (a_i - a_n) / (b_n - b_i) for i = 1..n
        # For k-NN: b_n - b_i = 1 - (-1/k) = 1 + 1/k for reverse neighbours, 1 otherwise
        denom = b_n - b
        t = (a - a_n) / denom

        # Sort t values and extract bounds
        t_sorted = np.sort(t)

        # ALRW2 §2.4: The p-value uses total points N = n+1 (training + test).
        # Upper bound: t_(ceil((1-ε)*N))  [Algorithm 2.2]
        # Lower bound: t_(floor(ε*N))    [Algorithm 2.3]
        # Two-sided: split ε between tails [Algorithm 2.4]
        # t_sorted has n values (one per training point).
        # Indices are 1-based in the book, convert to 0-based here.

        N = n + 1  # total number of points

        def get_upper(eps):
            """Upper bound from Algorithm 2.2: t_(ceil((1-eps)*N))."""
            idx = int(np.ceil((1 - eps) * N))  # 1-based index
            if idx < 1:
                return -np.inf
            if idx > n:
                return np.inf
            return float(t_sorted[idx - 1])

        def get_lower(eps):
            """Lower bound from Algorithm 2.3: t_(floor(eps*N))."""
            idx = int(np.floor(eps * N))  # 1-based index
            if idx < 1:
                return -np.inf
            if idx > n:
                return np.inf
            return float(t_sorted[idx - 1])

        def get_interval(eps):
            if bounds == "both":
                # Algorithm 2.4: split epsilon between tails
                lo = get_lower(eps / 2)
                hi = get_upper(eps / 2)
                return lo, hi
            elif bounds == "lower":
                lo = get_lower(eps)
                return lo, np.inf
            elif bounds == "upper":
                hi = get_upper(eps)
                return -np.inf, hi
            else:
                raise ValueError(f"bounds must be 'both', 'lower', or 'upper', got '{bounds}'")

        if hasattr(epsilon, "__iter__"):
            predictions = {}
            for eps in epsilon:
                lo, hi = get_interval(eps)
                predictions[eps] = self._construct_Gamma((lo, hi), eps)
            result = MultiLevelPredictionSet(predictions)
        else:
            lo, hi = get_interval(epsilon)
            result = self._construct_Gamma((lo, hi), epsilon)

        if return_update:
            return result, {"D": D_aug}
        return result

    @staticmethod
    def _compute_interval(alpha_sorted, y_hat, epsilon, n, bounds):
        """
        Compute prediction interval from sorted training residuals.

        The prediction set is {y : |y - y_hat| <= alpha_(j)} where
        j = ceil((1 - epsilon) * n) - 1 (0-indexed into sorted alphas).
        For two-sided: split epsilon equally.
        """
        if bounds == "both":
            # Two-sided: the interval is symmetric around y_hat
            # We need rank such that p-value > epsilon
            # threshold index: we want the (ceil((1-eps)*n) - 1)-th sorted alpha
            # (0-indexed), but there are only n-1 training alphas.
            idx = int(np.ceil((1 - epsilon) * n)) - 1
            if idx >= len(alpha_sorted):
                # All training residuals are included
                threshold = alpha_sorted[-1] if len(alpha_sorted) > 0 else np.inf
            elif idx < 0:
                return -np.inf, np.inf
            else:
                threshold = alpha_sorted[idx]
            return y_hat - threshold, y_hat + threshold
        elif bounds == "lower":
            idx = int(np.ceil((1 - epsilon) * n)) - 1
            if idx >= len(alpha_sorted):
                threshold = alpha_sorted[-1] if len(alpha_sorted) > 0 else np.inf
            elif idx < 0:
                return -np.inf, np.inf
            else:
                threshold = alpha_sorted[idx]
            return y_hat - threshold, np.inf
        elif bounds == "upper":
            idx = int(np.ceil((1 - epsilon) * n)) - 1
            if idx >= len(alpha_sorted):
                threshold = alpha_sorted[-1] if len(alpha_sorted) > 0 else np.inf
            elif idx < 0:
                return -np.inf, np.inf
            else:
                threshold = alpha_sorted[idx]
            return -np.inf, y_hat + threshold
        else:
            raise ValueError(f"bounds must be 'both', 'lower', or 'upper', got '{bounds}'")

    def compute_p_value(
        self, x: NDArray[np.floating[Any]], y: float, tau: float | None = None, smoothed: bool = True
    ) -> float:
        """Compute conformal p-value for a test pair (x, y).

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.
        y : float
            Hypothesized label.
        tau : float, optional
            Randomization value in [0, 1] for smoothed p-value.
            If None and smoothed=True, drawn uniformly at random.
        smoothed : bool
            Whether to compute the smoothed p-value (default True).

        Returns
        -------
        float
            The conformal p-value.
        """
        x = np.asarray(x).ravel()
        n = self._safe_size_check(self.X)
        if n == 0:
            return 1.0

        if smoothed and tau is None:
            tau = self.rnd_gen.uniform(0, 1)

        # Training residuals (leave-one-out k-NN)
        y_hat_train = self._knn_predictions(self.D, self.y)
        alpha_train = np.abs(self.y - y_hat_train)

        # Test point k-NN prediction.
        # Use lexsort (distance primary, y-value secondary) so that ties in
        # distance are broken canonically by y rather than by insertion order,
        # matching the training path in _knn_predictions (np.argpartition is
        # unstable on ties and would break permutation invariance).
        d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
        k_test = min(self.k, n)
        agg_func = np.mean if self.aggregation == "mean" else np.median
        test_knn_idx = np.lexsort((self.y, d))[:k_test]
        y_hat_test = agg_func(self.y[test_knn_idx])

        # Test nonconformity score
        alpha_test = np.abs(y - y_hat_test)

        # Compute p-value: fraction of training alphas >= test alpha
        Alpha = np.append(alpha_train, alpha_test)
        if smoothed:
            return self._compute_p_value(Alpha, tau, c_type="nonconformity")
        else:
            return self._compute_p_value(Alpha, c_type="nonconformity")


class KernelConformalRidgeRegressor(ConformalRegressor):
    r"""Kernel conformal ridge regression ([ALRW2 §2.3, §7.4]).

    The "kernel trick" applied to :class:`ConformalRidgeRegressor`: ridge
    regression is performed implicitly in the reproducing-kernel Hilbert space
    induced by ``kernel``, so the predictor can fit non-linear targets while
    keeping the same closed-form, exchangeability-based validity. The
    nonconformity measure is the (optionally studentised) residual in feature
    space; the kernel matrix inverse is maintained online by a rank-1 update.

    Some ready-made kernels live in :mod:`online_cp.kernels`, but most
    scikit-learn kernels and any callable ``kernel(X, X') -> ndarray`` also work.
    """

    _SAVE_PARAMS: tuple = (
        "kernel",
        "a",
        "warnings",
        "verbose",
        "rnd_state",
        "epsilon",
        "recompute_every",
        "studentised",
        "pruning",
        "budget",
    )
    _SAVE_STATE: tuple = ("X", "y", "K", "Kinv", "_n_sm_updates", "_reservoir_n_seen")
    _SAVE_CALLABLES: tuple = ("kernel",)

    def __init__(
        self,
        kernel: Any,
        a: float = 0,
        warnings: bool = True,
        verbose: int = 0,
        rnd_state: int | None = None,
        epsilon: float = default_epsilon,
        recompute_every: int | None = None,
        studentised: bool = False,
        pruning: str | None = None,
        budget: int | None = None,
    ) -> None:
        """
        KernelConformalRidgeRegressor requires a kernel. Some common kernels are found in kernels.py, but it is
        also compatible with (most) kernels from e.g. scikit-learn.
        Custom kernels can also be passed as callable functions.

        Parameters
        ----------
        recompute_every : int or None
            If set, recompute the full matrix inverse from scratch every N
            rank-1 updates to correct accumulated floating-point drift.
            Note: recomputation is O(n³) where n is the current training size.
            Default None (no periodic recomputation).
        studentised : bool
            If True, use the studentised conformity measure (ALRW2 §7.4).
            Default False.
        rnd_state : int, np.random.Generator, or None, default None
            Seed or Generator for the random number generator.
        pruning : {'reservoir', 'window'} or None, default None
            Stream-pruning strategy that bounds the retained training bag.
            Both strategies preserve exchangeability (and therefore the exact
            finite-sample coverage guarantee):
            ``'reservoir'`` keeps a uniform random sample of ``budget`` examples;
            ``'window'`` keeps the most recent ``budget`` examples (FIFO). The
            window is less robust than the reservoir on drifting / non-i.i.d.
            streams but is trivial to maintain. On each eviction the kernel
            inverse is recomputed from scratch at size ``budget``
            (O(budget³), bounded by the user's budget). ``None`` (default)
            keeps every example.
        budget : int or None, default None
            Retained-bag size for reservoir pruning. Must be set together with
            ``pruning='reservoir'``. With ``budget >= n`` seen, behaviour is
            identical to no pruning (exact, not just close).
        """
        super().__init__(epsilon=epsilon)

        self.a = a
        self.X = None
        self.y = None
        self.p = None
        self.Id = None
        self.K = None
        self.Kinv = None

        self.kernel = kernel

        # Should we raise warnings
        self.warnings = warnings

        self.verbose = verbose
        self.rnd_state = rnd_state
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
            self.rnd_gen = np.random.default_rng(rnd_state)

        # Do we use the studentised residuals
        self.studentised = studentised

        # Sherman-Morrison stability
        self.recompute_every = recompute_every
        self._n_sm_updates = 0

        # --- Reservoir pruning (budget-bounded stream) ---
        self.pruning = pruning
        self.budget = budget
        if pruning is not None and pruning not in ("reservoir", "window"):
            raise ValueError(
                f"pruning must be 'reservoir', 'window', or None, got {pruning!r}. "
                "Other strategies (sv, leverage, ...) select on model state and "
                "break exchangeability; they are not yet implemented."
            )
        if pruning in ("reservoir", "window"):
            if budget is None:
                raise ValueError(f"pruning={pruning!r} requires a positive integer budget")
        self._reservoir = (
            _ReservoirBuffer(int(budget), self.rnd_gen) if pruning is not None else None
        )
        self._reservoir_n_seen = 0

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        self.X = X
        self.y = y
        Id = np.identity(self.X.shape[0])

        self.K = self.kernel(self.X)
        try:
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
        except np.linalg.LinAlgError:
            raise ValueError("K + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None
        self._n_sm_updates = 0
        # Reservoir pruning: seed the reservoir from a uniform random subset of
        # the initial bag. With n > budget, keep a size-`budget` sample and rebuild
        # K / Kinv on the retained positions.
        if self._reservoir is not None:
            n = self.X.shape[0]
            self._reservoir_n_seen = n
            if n > self.budget:
                if self.pruning == "reservoir":
                    keep = np.sort(self.rnd_gen.choice(n, size=self.budget, replace=False))
                else:  # window: keep the most recent `budget` examples
                    keep = np.arange(n - self.budget, n)
                self.X = self.X[keep]
                self.y = self.y[keep]
                self.K = _evict_masked_distance(self.K, keep)
                Id = np.identity(self.X.shape[0])
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
                self._n_sm_updates = 0

    def recompute_inverse(self) -> None:
        """Recompute (K + aI)^{-1} from scratch to correct numerical drift.

        Call this periodically during long online streams, or set
        ``recompute_every`` in the constructor for automatic periodic
        recomputation.

        Note: This is O(n³) where n is the current training set size.
        """
        if self.K is None:
            return
        n = self.K.shape[0]
        if self.verbose > 0 and n > 1000:
            print(f"KernelConformalRidgeRegressor.recompute_inverse(): n={n}, O(n³) recomputation.")
        Id = np.identity(n)
        try:
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
        except np.linalg.LinAlgError:
            raise ValueError("K + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None
        self._n_sm_updates = 0

    @staticmethod
    def _update_Kinv(Kinv, k, kappa):
        """Block matrix-inversion lemma (ALRW2 eq. 2.56) to extend ``Kinv``.

        Symmetrizes the result to counteract round-off drift from repeated
        rank-1 extensions over a long online stream.
        """
        d = 1 / (kappa - k.T @ Kinv @ k)
        M = np.block([[Kinv + d * Kinv @ k @ k.T @ Kinv, -d * Kinv @ k], [-d * k.T @ Kinv, d]])
        return 0.5 * (M + M.T)

    @staticmethod
    def _update_K(K, k, kappa):
        return np.block([[K, k], [k.T, kappa]])

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a single example
        """
        # Reservoir pruning: decide accept/evict *before* touching state.
        # Returns -1 (keep, no eviction), i >= 0 (keep & evict i), or None (discard).
        if self._reservoir is not None:
            self._reservoir_n_seen += 1
            eviction = self._reservoir.decide(self._reservoir_n_seen, strategy=self.pruning)
        else:
            eviction = -1

        # Discard: reservoir at capacity and this example lost the draw -- state
        # unchanged (no append, no kernel/inverse update).
        if eviction is None:
            return

        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)

        if precomputed is not None:
            X = precomputed["X"]
            K = precomputed["K"]
            Kinv = precomputed["Kinv"]

            if X is not None:
                self.X = X

            if K is not None and Kinv is not None:
                self.K = K
                self.Kinv = Kinv

            else:
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                try:
                    self.Kinv = np.linalg.inv(self.K + self.a * Id)
                except np.linalg.LinAlgError:
                    raise ValueError("K + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None

        else:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1, -1)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                try:
                    self.Kinv = np.linalg.inv(self.K + self.a * Id)
                except np.linalg.LinAlgError:
                    raise ValueError("K + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None
            elif self.X.shape[0] == 1:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                try:
                    self.Kinv = np.linalg.inv(self.K + self.a * Id)
                except np.linalg.LinAlgError:
                    raise ValueError("K + aI is singular. Set a > 0 (ridge parameter) to regularise.") from None
            else:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                self.K = self._update_K(self.K, k, kappa)
                self.Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                self._n_sm_updates += 1

                if self.recompute_every and self._n_sm_updates % self.recompute_every == 0:
                    self.recompute_inverse()

        # If an existing position was evicted, drop row/col i of K, row i of X and
        # element i of y, then rebuild Kinv from scratch on the pruned kernel. The
        # recomputation is O(n³) with n == budget (the bag is capped), so it is
        # bounded by a constant set by the user's budget -- this is the intended
        # "flatten at budget" behaviour of reservoir pruning.
        if eviction >= 0:
            i = eviction
            self.K = _evict_distance_matrix(self.K, i)
            self.X = np.delete(self.X, i, axis=0)
            self.y = np.delete(self.y, i, axis=0)
            Id = np.identity(self.X.shape[0])
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
            self._n_sm_updates = 0

    def compute_A_and_B(self, X, K, Kinv, y):
        """Compute A and B vectors for kernel conformal ridge regression.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Augmented design matrix. Last row is the test object.
        K : ndarray, shape (n, n)
            Augmented kernel matrix.
        Kinv : ndarray, shape (n, n)
            Inverse of K + a*I for the augmented kernel matrix.
        y : ndarray, shape (n-1,)
            Training labels (test label appended as 0 internally).
        """
        n = X.shape[0]
        H = Kinv @ K
        C = np.identity(n) - H
        A = C @ np.append(y, 0)
        B = C @ np.append(np.zeros((n - 1,)), 1)

        if self.studentised:
            H_diag = np.diag(H)
            leverage = np.sqrt(np.clip(1 - H_diag, 1e-12, None))
            A = A / leverage
            B = B / leverage

        return A, B

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        bounds: str = "both",
        return_update: bool = False,
    ) -> (
        ContinuousPredictionSet
        | MultiLevelPredictionSet
        | tuple[ContinuousPredictionSet | MultiLevelPredictionSet, dict[str, Any]]
    ):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), bounds="both")
        ContinuousPredictionSet([(-inf, inf)])
        """

        def build_precomputed(X, K, Kinv, A, B):
            computed = {
                "X": X,  # The updated matrix of objects
                "K": K,  # The updated kernel matrix
                "Kinv": Kinv,
                "A": A,
                "B": B,
            }
            return computed

        if epsilon is None:
            epsilon = self.epsilon

        if self.X is not None:
            # Add row to X matrix
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            n = X.shape[0]

            # Check that the significance level is not too small. If it is, return infinite prediction interval
            # For multi-level: only bail out if even the largest epsilon is too small
            eps_check = max(epsilon) if hasattr(epsilon, "__iter__") else epsilon
            if bounds == "both":
                if not (eps_check >= 2 / n):
                    if self.warnings:
                        eps_warn = min(epsilon) if hasattr(epsilon, "__iter__") else epsilon
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(2 / eps_warn))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if hasattr(epsilon, "__iter__"):
                        predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                        result = MultiLevelPredictionSet(predictions)
                    else:
                        result = self._construct_Gamma((-np.inf, np.inf), epsilon)
                    if return_update:
                        k = self.kernel(self.X, x).reshape(-1, 1)
                        kappa = self.kernel(x, x)
                        K = self._update_K(self.K, k, kappa)
                        Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                        return result, build_precomputed(X, K, Kinv, None, None)
                    else:
                        return result
            else:
                if not (eps_check >= 1 / n):
                    if self.warnings:
                        eps_warn = min(epsilon) if hasattr(epsilon, "__iter__") else epsilon
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(1 / eps_warn))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if hasattr(epsilon, "__iter__"):
                        predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                        result = MultiLevelPredictionSet(predictions)
                    else:
                        result = self._construct_Gamma((-np.inf, np.inf), epsilon)
                    if return_update:
                        k = self.kernel(self.X, x).reshape(-1, 1)
                        kappa = self.kernel(x, x)
                        K = self._update_K(self.K, k, kappa)
                        Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                        return result, build_precomputed(X, K, Kinv, None, None)
                    else:
                        return result

            # Update kernel matrix (deferred until after feasibility check)
            k = self.kernel(self.X, x).reshape(-1, 1)
            kappa = self.kernel(x, x)
            K = self._update_K(self.K, k, kappa)
            Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)

            A, B = self.compute_A_and_B(X, K, Kinv, self.y)

            if self.studentised:
                diffs = B[-1] - B[:-1]
                safe = np.abs(diffs) > 1e-12
                t = np.where(safe, (A[:-1] - A[-1]) / diffs, np.inf)
                t.sort()
                l_dic = {i + 1: val for i, val in enumerate(t)}
                u_dic = {i + 1: val for i, val in enumerate(t)}
            else:
                l_dic, u_dic = self._vectorised_l_and_u(A, B)

            if bounds == "both":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        lo = self._get_lower(l_dic=l_dic, epsilon=eps / 2, n=n)
                        up = self._get_upper(u_dic=u_dic, epsilon=eps / 2, n=n)
                        predictions[eps] = self._construct_Gamma((lo, up), eps)
                    result = MultiLevelPredictionSet(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
                    result = self._construct_Gamma((lower, upper), epsilon)
            elif bounds == "lower":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        lo = self._get_lower(l_dic=l_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma((lo, np.inf), eps)
                    result = MultiLevelPredictionSet(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma((lower, np.inf), epsilon)
            elif bounds == "upper":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        up = self._get_upper(u_dic=u_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma((-np.inf, up), eps)
                    result = MultiLevelPredictionSet(predictions)
                else:
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma((-np.inf, upper), epsilon)
            else:
                raise ValueError('bounds must be "both", "lower", or "upper"')
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            X = x.reshape(1, -1)
            K = None
            Kinv = None
            A = None
            B = None

            if hasattr(epsilon, "__iter__"):
                predictions = {eps: self._construct_Gamma((-np.inf, np.inf), eps) for eps in epsilon}
                result = MultiLevelPredictionSet(predictions)
            else:
                result = self._construct_Gamma((-np.inf, np.inf), epsilon)

        if return_update:
            return result, build_precomputed(X, K, Kinv, A, B)
        else:
            return result

    def compute_p_value(self, x, y, bounds="both", precomputed=None, tau=None, smoothed=True):
        """
        Computes the smoothed p-value of the example (x, y).
        """
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            if not np.allclose(x, precomputed["X"][-1]):
                raise ValueError("x does not match the last row of precomputed['X']")
            A = precomputed["A"]
            B = precomputed["B"]

        else:
            if self.Kinv is not None:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                K = self._update_K(self.K, k, kappa)
                Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                X = np.append(self.X, x.reshape(1, -1), axis=0)
                A, B = self.compute_A_and_B(X, K, Kinv, self.y)
            else:
                A, B = None, None

        if A is not None and B is not None:
            if bounds == "both":
                E = A + y * B
                Alpha = np.zeros_like(A)
                for i, e in enumerate(E):
                    alpha = min((E >= e).sum(), (E <= e).sum())
                    Alpha[i] = alpha
                c_type = "conformity"
            elif bounds == "lower":
                Alpha = -(A + y * B)
                c_type = "nonconformity"
            elif bounds == "upper":
                Alpha = A + y * B
                c_type = "nonconformity"
            else:
                raise ValueError('bounds must be "both", "lower", or "upper"')

            if smoothed:
                p = self._compute_p_value(Alpha, tau, c_type=c_type)
            else:
                p = self._compute_p_value(Alpha, c_type=c_type)
        else:
            if smoothed:
                p = tau
            else:
                p = 1

        return p


# ===========================================================================
# Conformalised Lasso Regressor
# ===========================================================================


def _soft_threshold(z, lam):
    """Soft-thresholding operator for Lasso coordinate descent."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def _solve_lasso(X, y, lam, rho=0.0, max_iter=1000, tol=1e-6, warm_start=None):
    """
    Solve the elastic net problem via coordinate descent:
        min_{beta} (1/2) ||y - X beta||^2 + lam * ||beta||_1 + (rho/2) * ||beta||_2^2

    When rho=0 this is pure Lasso.

    Parameters
    ----------
    X : ndarray of shape (n, p)
    y : ndarray of shape (n,)
    lam : float, L1 regularization parameter
    rho : float, L2 regularization parameter (default 0.0)
    max_iter : int
    tol : float, convergence tolerance
    warm_start : ndarray of shape (p,) or None

    Returns
    -------
    beta : ndarray of shape (p,)
    """
    n, p = X.shape
    if warm_start is not None:
        beta = warm_start.copy()
    else:
        beta = np.zeros(p)

    # Precompute the diagonal of X^T X (column norm squared) used in the
    # soft-thresholding update.  The residual is maintained incrementally so a
    # sweep costs only the O(p) coordinate updates rather than a fresh O(n p)
    # matrix-vector product.
    col_norms_sq = np.sum(X**2, axis=0)  # (p,)
    residual = y - X @ beta

    def _kkt_residual(b):
        """Max |KKT violation| of b under the elastic-net objective.

        Active j : subgrad_j == sign(b_j) * lam   (subgrad = X^T (y - X b) + rho b)
        Inactive j : |subgrad_j| <= lam.
        """
        s = X.T @ (y - X @ b) + rho * b
        act = np.abs(b) > 1e-12
        l1 = np.max(np.abs(s[act] - np.sign(b[act]) * lam)) if act.any() else 0.0
        l2 = np.max(np.abs(s[~act])) if (~act).any() else 0.0
        return max(l1, l2)

    # Initial optimality check: a warm start may already be optimal.  This is
    # the key speed-up in the online loop, where every augmented-data solve
    # warm-starts from the previous solution (the Lasso path is continuous, so
    # a single augmented point is usually a no-op for the solution).
    if _kkt_residual(beta) < tol:
        return beta

    # Cyclic coordinate descent with the *true* KKT stopping rule.
    #
    # The original implementation stopped on ``max|delta beta| < tol`` with
    # ``tol = 1e-6``.  That test is *insufficient*: a single coefficient can
    # oscillate below the threshold each sweep while the subgradient is still
    # far from +/-lam, so the loop halts early and returns a solution that is
    # not the Lasso/elastic-net optimum.  A non-optimum starting solution is
    # exactly what made the conformal-Lasso homotopy diverge from the true
    # solution path and under-cover.  Here we instead check the optimality
    # condition itself: for active j the subgradient must equal sign(beta_j)
    # * lam, and for inactive j it must lie in [-lam, lam].  This converges to
    # the (unique) optimum and, because it is an actual stopping test, also
    # stops as soon as the warm start is already optimal.
    for _iteration in range(max_iter):
        for j in range(p):
            if col_norms_sq[j] == 0:
                continue
            # Unconstrained coordinate update (partial residual re-adding the
            # j-th contribution): beta_j^* = S_lam( x_j^T (r + x_j beta_j) ) /
            # (||x_j||^2 + rho), with r = y - X beta.
            residual += X[:, j] * beta[j]
            rho_j = X[:, j] @ residual
            beta[j] = _soft_threshold(rho_j, lam) / (col_norms_sq[j] + rho)
            residual -= X[:, j] * beta[j]
        if _kkt_residual(beta) < tol:
            break

    return beta


@njit(cache=True)
def _compute_crossings(r_train, slopes_train, r_test, slope_test, n, dt_k):
    """Find all delta values where |r_i(delta)| = |r_{n+1}(delta)| for some i.

    Returns an array of valid crossing times within [0, dt_k].
    """
    # Maximum possible crossings: 2 per training point (two sign cases)
    crossings = np.empty(2 * n)
    count = 0
    for i in range(n):
        for s in (1.0, -1.0):
            a = r_train[i] - s * r_test
            b = slopes_train[i] - s * slope_test
            if abs(b) > 1e-15:
                d = -a / b
                if -1e-12 < d < dt_k + 1e-12:
                    if d < 0.0:
                        d = 0.0
                    elif d > dt_k:
                        d = dt_k
                    crossings[count] = d
                    count += 1
    return crossings[:count]


@njit(cache=True)
def _compute_crossings_with_index(r_train, slopes_train, r_test, slope_test, n, dt_k):
    """Find all delta values where |r_i(delta)| = |r_{n+1}(delta)|, with identity.

    Same crossing logic as :func:`_compute_crossings`, but additionally returns
    the index ``i`` of the training point that crossed and the sign-case ``s``
    (``+1`` for ``r_i = r_{n+1}``, ``-1`` for ``r_i = -r_{n+1}``). This lets the
    caller sweep the conformal rank incrementally in ``O(n + m)`` instead of
    building an ``O(n * m)`` residual matrix (see T2 speed target).

    Returns
    -------
    d : (m,) float array of valid crossing times within [0, dt_k]
    i : (m,) int array of the training-point index for each crossing
    s : (m,) float array of the sign-case (+1 / -1) for each crossing
    """
    nmax = 2 * n
    d = np.empty(nmax)
    i = np.empty(nmax, dtype=np.int64)
    s = np.empty(nmax)
    count = 0
    for idx in range(n):
        for sg in (1.0, -1.0):
            a = r_train[idx] - sg * r_test
            b = slopes_train[idx] - sg * slope_test
            if abs(b) > 1e-15:
                dd = -a / b
                if -1e-12 < dd < dt_k + 1e-12:
                    if dd < 0.0:
                        dd = 0.0
                    elif dd > dt_k:
                        dd = dt_k
                    d[count] = dd
                    i[count] = idx
                    s[count] = sg
                    count += 1
    return d[:count], i[:count], s[:count]


class ConformalLassoRegressor(ConformalRegressor):
    """
    Conformal prediction with Lasso/elastic net using the piecewise linear
    homotopy from Lei (2019). Computes exact conformal prediction sets
    without grid search by tracing how residuals evolve as the test label varies.

    References
    ----------
    Lei, J. (2019). Fast exact conformalization of the Lasso using piecewise
    linear homotopy. *Biometrika*, 106(4), 751–767.

    When rho=0 (default), this is pure Lasso. When rho>0, this solves the
    elastic net: min (1/2)||y - Xβ||² + lam||β||₁ + (rho/2)||β||₂²

    Parameters
    ----------
    lam : float
        L1 regularization parameter (lambda). Must be non-negative.
    rho : float
        L2 regularization parameter (default 0.0). When rho=0, pure Lasso.
    epsilon : float
        Significance level for prediction sets (default 0.1).
    autotune : bool
        If True, tune lambda via K-fold cross-validation in learn_initial_training_set.
    n_folds : int
        Number of CV folds for autotuning (default 5).
    search_range_factor : float
        Factor to extend the y search range beyond [y_min, y_max] (default 0.25).
    max_homotopy_steps : int
        Maximum number of homotopy breakpoints to trace per direction (default 1000).
    verbose : int
        Verbosity level.
    warnings : bool
        Whether to emit warnings.
    rnd_state : int, np.random.Generator, or None
        Random seed or Generator.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> N = 50
    >>> X = np.random.normal(size=(N, 10))
    >>> beta_true = np.array([3, 1.5, 0, 0, 2, 0, 0, 0, 0, 0])
    >>> y = X @ beta_true + np.random.normal(scale=0.5, size=N)
    >>> cp = ConformalLassoRegressor(lam=0.5)
    >>> cp.learn_initial_training_set(X[:30], y[:30])
    >>> interval = cp.predict(X[30], epsilon=0.1)
    >>> y[30] in interval
    True
    """

    _SAVE_PARAMS: tuple = (
        "lam",
        "rho",
        "epsilon",
        "autotune",
        "n_folds",
        "search_range_factor",
        "max_homotopy_steps",
        "verbose",
        "warnings",
        "rnd_state",
    )
    _SAVE_STATE: tuple = ("X", "y", "beta")

    def __init__(
        self,
        lam=1.0,
        rho=0.0,
        epsilon=default_epsilon,
        autotune=False,
        n_folds=5,
        search_range_factor=0.25,
        max_homotopy_steps=1000,
        verbose=0,
        warnings=True,
        rnd_state=None,
    ):
        super().__init__(epsilon=epsilon)
        self.lam = lam
        self.rho = rho
        self.autotune = autotune
        self.n_folds = n_folds
        self.search_range_factor = search_range_factor
        self.max_homotopy_steps = max_homotopy_steps
        self.verbose = verbose
        self.warnings = warnings
        self.rnd_state = rnd_state
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
            self.rnd_gen = np.random.default_rng(rnd_state)

        self.X = None
        self.y = None
        self.beta = None  # Current Lasso solution

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        """Fit initial Lasso on the training data."""
        self.X = X.copy()
        self.y = y.copy()

        if self.autotune:
            self._tune_lambda()

        self.beta = _solve_lasso(self.X, self.y, self.lam, rho=self.rho)

    def _tune_lambda(self):
        """Tune lambda via K-fold cross-validation.

        The lambda grid is walked in *descending* order and each fold's
        coordinate-descent fit is warm-started from the previous lambda's
        solution for that fold (the Lasso solution is monotone in lambda, so a
        nearby starting point converges in far fewer sweeps). This preserves
        the exact same CV procedure -- same folds, same grid, same MSE -- while
        turning ~n_folds * grid_size cold fits into a near-warm homotopy path.
        """
        n = self.X.shape[0]
        indices = np.arange(n)
        self.rnd_gen.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        # Precompute per-fold train/validation arrays once (avoid re-indexing
        # and concatenating on every (lambda, fold) iteration).
        train_arrays = []
        val_arrays = []
        for k in range(self.n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != k])
            train_arrays.append((self.X[train_idx], self.y[train_idx]))
            val_arrays.append((self.X[val_idx], self.y[val_idx]))

        # Lambda grid: geometric sequence, walked descending (large -> small).
        lam_max = np.max(np.abs(self.X.T @ self.y)) / n
        lam_grid = np.geomspace(lam_max * 1e-3, lam_max, num=50)

        best_lam = self.lam
        best_mse = np.inf

        # Per-fold warm-started solutions carried across lambdas (descending).
        fold_betas = [None] * self.n_folds
        for lam in lam_grid:
            mse = 0.0
            for k in range(self.n_folds):
                X_tr, y_tr = train_arrays[k]
                X_val, y_val = val_arrays[k]
                beta_k = _solve_lasso(X_tr, y_tr, lam, rho=self.rho, warm_start=fold_betas[k])
                fold_betas[k] = beta_k
                mse += np.mean((y_val - X_val @ beta_k) ** 2)
            mse /= self.n_folds
            if mse < best_mse:
                best_mse = mse
                best_lam = lam

        self.lam = best_lam
        if self.verbose > 0:
            print(f"Tuned lambda: {self.lam:.6f} (CV MSE: {best_mse:.4f})")

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a new data point. Updates the training set and refits Lasso.

        If precomputed is provided (from predict with return_update=True),
        uses the cached Lasso solution to avoid refitting.
        """
        x = np.atleast_1d(x).ravel()

        # Update training set
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y])
        else:
            self.X = np.vstack([self.X, x.reshape(1, -1)])
            self.y = np.append(self.y, y)

        if precomputed is not None and precomputed.get("beta") is not None:
            self.beta = precomputed["beta"]
        else:
            # Refit from scratch (warm-started from current beta)
            self.beta = _solve_lasso(self.X, self.y, self.lam, rho=self.rho, warm_start=self.beta)

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        return_update: bool = False,
    ) -> (
        ContinuousPredictionSet
        | MultiLevelPredictionSet
        | tuple[ContinuousPredictionSet | MultiLevelPredictionSet, dict[str, Any]]
    ):
        """
        Compute the conformal prediction set at x using the homotopy algorithm.

        If epsilon is a list/array, returns a MultiLevelPredictionSet.
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Handle multi-level epsilon
        if hasattr(epsilon, "__iter__"):
            predictions = {}
            for eps in epsilon:
                result = self.predict(x, epsilon=eps, return_update=False)
                predictions[eps] = result
            result = MultiLevelPredictionSet(predictions)
            if return_update:
                return result, {"beta": None}
            return result

        x = np.atleast_1d(x).ravel()

        if self.X is None or self.X.shape[0] < 2:
            result = self._construct_Gamma((-np.inf, np.inf), epsilon)
            if return_update:
                return result, {"beta": None}
            return result

        # y_{n+1}(0) = x_{n+1}^T beta_hat — the "neutral" label
        y0 = x @ self.beta

        # Compute search range
        y_range = self.y.max() - self.y.min()
        if y_range == 0:
            y_range = 1.0
        t_max = (self.y.max() - y0) + self.search_range_factor * y_range
        t_min = (self.y.min() - y0) - self.search_range_factor * y_range

        # Run homotopy in both directions (only if search range is non-trivial)
        intervals_pos = self._run_homotopy(x, direction=+1, t_bound=t_max, epsilon=epsilon) if t_max > 0 else []
        intervals_neg = self._run_homotopy(x, direction=-1, t_bound=t_min, epsilon=epsilon) if t_min < 0 else []

        # Merge intervals (shift from t-space to y-space)
        all_intervals = []
        for a, b in intervals_neg:
            all_intervals.append((y0 + a, y0 + b))
        # Add t=0 point
        # Check if t=0 is in the prediction set
        if self._t_in_prediction_set(x, 0.0, epsilon):
            all_intervals.append((y0, y0))
        for a, b in intervals_pos:
            all_intervals.append((y0 + a, y0 + b))

        # Merge overlapping/adjacent intervals
        merged = self._merge_intervals(all_intervals)

        if not merged:
            result = self._construct_Gamma((np.nan, np.nan), epsilon)
        elif len(merged) == 1:
            result = self._construct_Gamma((merged[0][0], merged[0][1]), epsilon)
        else:
            # Return the smallest enclosing interval (conservative)
            result = self._construct_Gamma((merged[0][0], merged[-1][1]), epsilon)

        # Build precomputed for learn_one
        precomputed_dict = None
        if return_update:
            precomputed_dict = {"beta": None}  # Will be set if we can extract from homotopy

        if return_update:
            return result, precomputed_dict
        return result

    def compute_p_value(
        self, x: NDArray[np.floating[Any]], y: float, tau: float | None = None, smoothed: bool = True
    ) -> float:
        """
        Compute the conformal p-value for (x, y) given current training set.
        """
        x = np.atleast_1d(x).ravel()

        if self.X is None:
            if smoothed:
                return self.rnd_gen.uniform() if tau is None else tau
            return 1.0

        if tau is None and smoothed:
            tau = self.rnd_gen.uniform()

        # Augment training set with (x, y)
        X_aug = np.vstack([self.X, x.reshape(1, -1)])
        y_aug = np.append(self.y, y)

        # Fit Lasso on augmented data
        beta_aug = _solve_lasso(X_aug, y_aug, self.lam, rho=self.rho, warm_start=self.beta)

        # Compute residuals
        residuals = np.abs(y_aug - X_aug @ beta_aug)

        # p-value: fraction of residuals >= residual of test point
        r_test = residuals[-1]
        if smoothed and tau is not None:
            gt = np.sum(residuals > r_test)
            eq = np.sum(residuals == r_test)
            p = (gt + tau * eq) / len(residuals)
        else:
            geq = np.sum(residuals >= r_test)
            p = geq / len(residuals)

        return p

    def _t_in_prediction_set(self, x_new, t, epsilon):
        """Check if a specific t value puts y_{n+1}(t) in the prediction set."""
        n = self.X.shape[0]
        threshold = int(np.ceil((n + 1) * (1 - epsilon)))

        # Augment
        X_aug = np.vstack([self.X, x_new.reshape(1, -1)])
        y_aug = np.append(self.y, x_new @ self.beta + t)

        # Fit Lasso
        beta_t = _solve_lasso(X_aug, y_aug, self.lam, rho=self.rho, warm_start=self.beta)

        # Residuals
        abs_res = np.abs(y_aug - X_aug @ beta_t)

        # Rank of |r_{n+1}| among all (n+1) residuals, in ascending order
        # (1 = smallest): count of residuals with |r_i| <= |r_{n+1}|, which
        # always includes the test point itself.  Lei (2019), prelim.tex line
        # 141: y_{n+1}(t) is in the set iff this ascending rank is NO HIGHER
        # than ceil((n+1)(1-alpha)) -- i.e. the test residual must be among
        # the SMALLEST in magnitude (a candidate the Lasso fits well).
        rank = np.sum(abs_res <= abs_res[-1])
        return rank <= threshold

    def _run_homotopy(self, x_new, direction, t_bound, epsilon):
        """
        Run the piecewise linear homotopy in one direction.

        Returns a list of (t_start, t_end) intervals that are IN the prediction set.
        """
        n = self.X.shape[0]
        p = self.X.shape[1]
        lam = self.lam
        threshold = int(np.ceil((n + 1) * (1 - epsilon)))

        # sign of direction
        sign = 1 if direction > 0 else -1
        t_bound_abs = abs(t_bound)

        # Initial state
        beta_k = self.beta.copy()
        J_k = np.where(np.abs(beta_k) > 1e-12)[0]  # active set
        signs_k = np.sign(beta_k[J_k])  # signs on active set

        # Dual variable on inactive set: v = X^T(y - X*beta) evaluated at t=0
        # At t=0 the augmented data is (X; x_new) with label y_{n+1}=x_new^T beta
        # The augmented problem's subgradient:
        # v_j = sum_i (y_i - x_i^T beta) x_{i,j} + (y_{n+1}(0) - x_new^T beta) * x_new_j
        # But y_{n+1}(0) = x_new^T beta, so the second term is 0
        # v = X^T (y - X beta) = X^T r (initial residuals)
        residuals = self.y - self.X @ beta_k
        v_full = self.X.T @ residuals  # subgradient (not including test point contribution at t=0)
        # Note: for the augmented problem at t=0, the n+1-th contribution is 0

        J_c_k = np.setdiff1d(np.arange(p), J_k)  # inactive set
        v_inactive = v_full[J_c_k]

        # Residuals at t=0 for training points
        r_train = residuals.copy()  # r_i(0) = y_i - x_i^T beta for i=1..n
        r_test = 0.0  # r_{n+1}(0) = y_{n+1}(0) - x_new^T beta(0) = 0

        # The Sigma hat used in the paper: (1/n) X^T X
        # But actually the formula uses sum_{i=1}^{n+1} x_i x_i^T = n*Sigma + x_new*x_new^T
        # Let's define Sigma_aug = X^T X + x_new x_new^T (unnormalized)
        XtX = self.X.T @ self.X  # n * Sigma
        XtX_aug = XtX + np.outer(x_new, x_new)  # sum_{i=1}^{n+1} x_i x_i^T

        t_accumulated = 0.0
        intervals_in_set = []

        for _step in range(self.max_homotopy_steps):
            if t_accumulated >= t_bound_abs:
                break

            # Compute eta(k) and gamma(k)
            # eta(k) = (Sigma_aug_{J_k})^{-1} x_{n+1,J_k} / (1 + x_{n+1,J_k}^T (Sigma_aug_{J_k})^{-1} x_{n+1,J_k})
            # But the paper uses n^{-1} * Sigma_hat = (1/n) sum x_i x_i^T, so Sigma_hat_{J_k} = XtX_aug[J_k][:,J_k]/n?
            # Actually re-reading: the paper defines Sigma_hat = (1/n) sum_{i=1}^n x_i x_i^T
            # And the formula has (sum_{i=1}^{n+1} x_{i,J} x_{i,J}^T)^{-1} = (n Sigma_hat_J + x_{n+1,J} x_{n+1,J}^T)^{-1}
            # which equals XtX_aug[J,J]^{-1}

            if len(J_k) == 0:
                # All variables inactive — beta(t) = 0 for all t in this piece
                # r_{n+1}(t) grows linearly with slope 1 (since eta=0)
                # This piece extends until some dual variable hits +/-lambda
                if len(J_c_k) == 0:
                    break
                # gamma(k) = x_{n+1,J_c} (since J is empty, no correction term)
                gamma_k = sign * x_new[J_c_k]

                # Breakpoint: dual variable hits boundary
                dt_dual = np.full(len(J_c_k), np.inf)
                for idx, _j in enumerate(J_c_k):
                    g = gamma_k[idx]
                    if g > 1e-15:
                        dt_dual[idx] = (lam - sign * v_inactive[idx]) / g
                    elif g < -1e-15:
                        dt_dual[idx] = (-lam - sign * v_inactive[idx]) / g
                dt_dual = np.where(dt_dual > 1e-12, dt_dual, np.inf)

                dt_k = np.min(dt_dual)
                dt_k = min(dt_k, t_bound_abs - t_accumulated)

                # In this piece: r_i(t) = r_train_i (constant), r_{n+1}(t) = sign*t (grows)
                # Find sub-intervals in prediction set
                sub_intervals = self._find_intervals_in_piece(
                    r_train,
                    r_test,
                    slopes_train=np.zeros(n),
                    slope_test=sign * 1.0,
                    dt_k=dt_k,
                    t_accumulated=t_accumulated,
                    sign=sign,
                    threshold=threshold,
                    n=n,
                )
                intervals_in_set.extend(sub_intervals)

                # Advance
                t_accumulated += dt_k
                r_test += sign * 1.0 * dt_k
                v_inactive += gamma_k * dt_k

                # Update active set
                if dt_k < t_bound_abs - t_accumulated + 1e-12:
                    entering = J_c_k[np.argmin(dt_dual)]
                    J_k = np.append(J_k, entering)
                    signs_k = np.append(signs_k, np.sign(v_inactive[np.argmin(dt_dual)]))
                    J_c_k = np.setdiff1d(np.arange(p), J_k)
                    # Recompute v_full with updated r_test
                    v_full = self.X.T @ r_train + x_new * r_test
                    v_inactive = v_full[J_c_k]
                continue

            # Normal case: |J_k| > 0
            Sigma_J = XtX_aug[np.ix_(J_k, J_k)] + self.rho * np.eye(len(J_k))
            x_J = x_new[J_k]

            try:
                # Solve Sigma_J * z = x_J directly (avoids forming the full
                # inverse; numerically preferred and identical up to rounding).
                Sigma_J_inv_x = np.linalg.solve(Sigma_J, x_J)
            except np.linalg.LinAlgError:
                # Singular — cannot continue homotopy
                break

            # eta(k) = (XtX_aug_J)^{-1} x_{n+1,J}  [paper eq. (5), first form]
            # Note: the Sherman-Morrison form with training-only covariance has a
            # 1/(1 + ...) denominator, but that is already baked into the
            # inverse of XtX_aug.  Do NOT divide by denom again.
            eta_k = sign * Sigma_J_inv_x  # slope of beta_{J_k}(t) w.r.t. |t|

            # gamma(k) for inactive variables  [paper eq. (10), first form]
            if len(J_c_k) > 0:
                Sigma_JcJ = XtX_aug[np.ix_(J_c_k, J_k)]
                gamma_k = sign * (x_new[J_c_k] - Sigma_JcJ @ Sigma_J_inv_x)
            else:
                gamma_k = np.array([])

            # Slopes of residuals:
            # dr_i/d(delta) = -x_{i,J}^T eta_k  for training points
            # dr_{n+1}/d(delta) = sign * (1 - x_J^T eta_unsigned)
            slopes_train = -(self.X[:, J_k] @ eta_k)  # (n,) — dr_i/d(delta_t)
            slope_test = sign * (1.0 - x_J @ Sigma_J_inv_x)  # dr_{n+1}/d(delta_t)

            # Find breakpoint t_{k+1}
            # Primal: beta_j(t_k) + eta_j(k) * dt = 0 for j in J_k.
            # dt_j = -beta_J_j / eta_j (vectorised, masked by |eta| > 1e-15).
            beta_J = beta_k[J_k]
            dt_primal = np.full(len(J_k), np.inf)
            active_eta = np.abs(eta_k) > 1e-15
            if np.any(active_eta):
                cand = np.full(len(J_k), np.inf)
                cand[active_eta] = -beta_J[active_eta] / eta_k[active_eta]
                cand = np.where(cand > 1e-12, cand, np.inf)
                dt_primal = cand

            # Dual: |v_j(t_k) + gamma_j * dt| = lambda for j in J_c.
            # For each j the two candidate breakpoints are (lam - v_j)/g and
            # (-lam - v_j)/g; take the smallest one that is positive.
            dt_dual = np.full(len(J_c_k), np.inf)
            if len(J_c_k) > 0:
                g = gamma_k
                v_j = v_inactive
                valid_g = np.abs(g) > 1e-15
                if np.any(valid_g):
                    dt1 = np.full(len(J_c_k), np.inf)
                    dt2 = np.full(len(J_c_k), np.inf)
                    dt1[valid_g] = (lam - v_j[valid_g]) / g[valid_g]
                    dt2[valid_g] = (-lam - v_j[valid_g]) / g[valid_g]
                    # Per-j: min of the candidates that are > 1e-12 (else inf).
                    dt1 = np.where(dt1 > 1e-12, dt1, np.inf)
                    dt2 = np.where(dt2 > 1e-12, dt2, np.inf)
                    dt_dual = np.minimum(dt1, dt2)

            dt_k = min(
                np.min(dt_primal) if len(dt_primal) > 0 else np.inf,
                np.min(dt_dual) if len(dt_dual) > 0 else np.inf,
            )
            dt_k = min(dt_k, t_bound_abs - t_accumulated)

            if dt_k <= 0 or not np.isfinite(dt_k):
                break

            # Find sub-intervals in this piece that are in the prediction set
            sub_intervals = self._find_intervals_in_piece(
                r_train,
                r_test,
                slopes_train=slopes_train,
                slope_test=slope_test,
                dt_k=dt_k,
                t_accumulated=t_accumulated,
                sign=sign,
                threshold=threshold,
                n=n,
            )
            intervals_in_set.extend(sub_intervals)

            # Advance state
            beta_k[J_k] += eta_k * dt_k
            r_train += slopes_train * dt_k
            r_test += slope_test * dt_k
            if len(J_c_k) > 0:
                v_inactive += gamma_k * dt_k
            t_accumulated += dt_k

            # Update active set based on what hit the boundary
            min_primal = np.min(dt_primal) if len(dt_primal) > 0 else np.inf
            min_dual = np.min(dt_dual) if len(dt_dual) > 0 else np.inf

            if dt_k >= t_bound_abs - (t_accumulated - dt_k):
                break  # Reached search boundary

            if min_primal <= min_dual:
                # A variable leaves the active set
                leaving_idx = np.argmin(dt_primal)
                leaving_var = J_k[leaving_idx]
                beta_k[leaving_var] = 0.0
                J_k = np.delete(J_k, leaving_idx)
                signs_k = np.delete(signs_k, leaving_idx)
            else:
                # A variable enters the active set
                entering_idx = np.argmin(dt_dual)
                entering_var = J_c_k[entering_idx]
                entering_sign = np.sign(v_inactive[entering_idx])
                if entering_sign == 0.0:
                    entering_sign = 1.0
                J_k = np.append(J_k, entering_var)
                signs_k = np.append(signs_k, entering_sign)

            J_c_k = np.setdiff1d(np.arange(p), J_k)
            # Recompute v_inactive for new inactive set
            # v = X_aug^T * r_aug where r_aug includes test point
            # Since we track r_train and r_test:
            v_full = self.X.T @ r_train + x_new * r_test
            v_inactive = v_full[J_c_k]

        return intervals_in_set

    def _find_intervals_in_piece(
        self, r_train, r_test, slopes_train, slope_test, dt_k, t_accumulated, sign, threshold, n
    ):
        """
        Within one homotopy piece of length dt_k, find sub-intervals where
        the ascending rank of |r_{n+1}(t)| among all |r_i(t)| is NO HIGHER than
        threshold (Lei 2019, prelim.tex line 141: the test residual must be
        among the SMALLEST in magnitude for the candidate to be in the set).

        Residuals are linear in delta_t (local parameter within the piece):
            r_i(delta) = r_train[i] + slopes_train[i] * delta   for i=0..n-1
            r_{n+1}(delta) = r_test + slope_test * delta

        Returns intervals in GLOBAL t-space (t_accumulated + sign*delta mapped to t).
        """
        # Find all crossing points WITH identity (index + sign-case) so the
        # conformal rank can be swept incrementally in O(n + m) instead of the
        # O(n * m) residual-matrix materialization used previously (T2 fix).
        raw_d, raw_i, raw_s = _compute_crossings_with_index(r_train, slopes_train, r_test, slope_test, n, dt_k)

        # Build sorted unique crossings with endpoints (C-level, no Python set/sort)
        crossings = np.unique(np.concatenate(([0.0, dt_k], raw_d)))

        if len(crossings) < 2:
            return []

        # Sub-intervals are the gaps between consecutive crossings.
        d_starts = crossings[:-1]
        d_ends = crossings[1:]
        widths = d_ends - d_starts

        # Ascending rank per sub-interval via an O(n + m) incremental sweep.
        #
        # Within a sub-interval (d_starts[j], d_ends[j]) the *relative order* of
        # |r_i| vs |r_{n+1}| is constant, so the rank is constant there. The rank
        # only changes at the crossings. Start from the rank at d = 0 (O(n)) and
        # update by +/-1 at each crossing: a crossing flips exactly one training
        # point's contribution, and the flip direction is set by the derivative of
        # (|r_i| - |r_{n+1}|) at the crossing.
        #
        # Lei (2019) conformal criterion (prelim.tex line 141 / method.tex line
        # 459): a candidate is kept iff its ascending rank is NO HIGHER than
        # ceil((n+1)(1-alpha)), i.e. the test residual must be among the SMALLEST
        # in magnitude. The +1 below counts the test point itself.
        raw_d_sorted = raw_d
        if raw_d_sorted.size:
            order = np.argsort(raw_d_sorted)
            cs_d = raw_d_sorted[order]
            cs_i = raw_i[order]
        else:
            cs_d = np.array([])
            cs_i = np.array([], dtype=np.int64)

        # Rank at d = 0: #training |r_i(0)| <= |r_{n+1}(0)|  (+1 for the test point).
        mag_train0 = np.abs(r_train)
        mag_test0 = np.abs(r_test)
        rank0 = int((mag_train0 <= mag_test0).sum()) + 1

        # Per-crossing flip: -1 if (|r_i| - |r_{n+1}|) is increasing at the
        # crossing (point moves from below to above -> drops out of the rank),
        # +1 if decreasing (point moves from above to below -> enters the rank).
        flips = np.zeros(cs_d.size, dtype=np.int64)
        for k in range(cs_d.size):
            idx = int(cs_i[k])
            dd = cs_d[k]
            ri = r_train[idx] + slopes_train[idx] * dd
            rn = r_test + slope_test * dd
            sri = 1.0 if ri > 0 else (-1.0 if ri < 0 else 0.0)
            srn = 1.0 if rn > 0 else (-1.0 if rn < 0 else 0.0)
            deriv = sri * slopes_train[idx] - srn * slope_test
            if deriv > 0:
                flips[k] = -1
            elif deriv < 0:
                flips[k] = 1

        # Accumulate flips onto the sorted-unique crossing grid, then take the
        # cumulative sum so sub-interval j (between crossings[j] and crossings[j+1])
        # sees every crossing at crossings[0..j] as already passed.
        flip_at = np.zeros(crossings.size, dtype=np.int64)
        if cs_d.size:
            idx_at = np.searchsorted(crossings, cs_d, side="left")
            np.add.at(flip_at, idx_at, flips)
        cum = np.cumsum(flip_at)
        # Rank in sub-interval j (between crossings[j] and crossings[j+1]) is
        # rank0 + cum[j]: a crossing exactly at crossings[j] sits on the sub-interval's
        # left boundary, so it has already been passed by the sub-interval interior.
        ranks = rank0 + cum[:-1]  # length == len(crossings) - 1

        # Drop degenerate sub-intervals (zero width) from all three arrays at once.
        valid = widths >= 1e-14
        if not np.any(valid):
            return []
        d_starts = d_starts[valid]
        d_ends = d_ends[valid]
        ranks = ranks[valid]

        in_set = ranks <= threshold

        if not np.any(in_set):
            return []

        d_starts = d_starts[in_set]
        d_ends = d_ends[in_set]

        # Map to global t-space. With sign > 0, d_starts < d_ends gives
        # t_starts < t_ends; with sign < 0 the order reverses, so swap to keep
        # the convention t_start <= t_end.
        t_left = sign * (t_accumulated + np.minimum(d_starts, d_ends))
        t_right = sign * (t_accumulated + np.maximum(d_starts, d_ends))
        if sign < 0:
            t_left, t_right = t_right, t_left

        return list(zip(t_left, t_right))

    @staticmethod
    def _merge_intervals(intervals):
        """Merge overlapping or adjacent intervals."""
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for a, b in sorted_intervals[1:]:
            if a <= merged[-1][1] + 1e-12:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged


# ---------------------------------------------------------------------------
# Mondrian tree / forest conformal regressors
# ---------------------------------------------------------------------------


class ConformalMondrianTreeRegressor(_MondrianRegressorInspection, ConformalRegressor):
    """Conformal predictor using a single Mondrian tree for regression.

    A full (transductive) conformal predictor that rebuilds one Mondrian tree
    at each prediction step from the augmented bag ``[X_train; x_test]``.
    The nonconformity measure (NCM) is the absolute residual from the leaf mean:

        α_i = |y_i − μ_leaf(x_i)|

    where μ_leaf is the mean of **all** y-values (including the candidate) in
    the leaf.  Because the mean is computed over the full augmented multiset
    this is a valid bag function and the resulting conformal predictor has
    exact, finite-sample validity (ALRW2 §2.2.9).

    The prediction interval is computed **algebraically** via an O(n) knot-point
    solver: the p-value p(y) is piecewise-constant between O(n) critical y-values
    where NCM ranks change.  No grid search is needed.

    Mathematical note
    -----------------
    Let A = number of training points in ``leaf_star`` and B = their y-sum.
    Under candidate y:

    * Test NCM:      ``|A·y − B| / (A+1)``        slope ±A/(A+1)
    * Inside NCM_j:  ``|(A+1)·y_j − B − y| / (A+1)``  slope ∓1/(A+1)

    Since A ≥ 1, the test NCM grows strictly faster than any inside NCM as
    y → ±∞, so p(y) is unimodal (guaranteed contiguous interval).

    Parameters
    ----------
    lifetime : float or {'sqrt_n', 'density'}
        Mondrian tree depth budget L. Larger → more splits → smaller leaves.
        Default 1.0.

        Can also be a string for unsupervised automatic tuning (only X is
        used; labels are never accessed):

        - ``'sqrt_n'``: halt when the leaf count drops to ≤ √n.
        - ``'density'``: choose τ maximising Σ_leaf n_leaf · log(n_leaf / V_leaf).
    epsilon : float
        Default significance level. Default 0.1.
    rnd_state : int, optional
        Seed for reproducibility.
    verbose : int
        Verbosity level (0 = silent).
    max_depth : int or None
        Hard depth cap. Nodes at depth >= max_depth become leaves regardless
        of remaining lifetime. None = no cap.
    feature_weights : {'variance'} or array-like of shape (d,) or None
        Unsupervised feature importance for split dimension sampling. Labels
        are never accessed.

        - ``None`` (default): uniform — standard isotropic Mondrian.
        - ``'variance'``: split probabilities ∝ population variance per
          feature. High-variance dimensions are sampled more often.
        - array of shape (d,): explicit non-negative weights (normalised
          internally; need not sum to 1).
    """

    _SAVE_PARAMS: tuple = (
        "lifetime",
        "epsilon",
        "rnd_state",
        "verbose",
        "max_depth",
        "feature_weights",
        "online",
    )
    _SAVE_STATE: tuple = ("X", "y", "_online_tree")

    def __init__(
        self,
        lifetime: float | str = 1.0,
        epsilon: float = 0.1,
        rnd_state: int | None = None,
        verbose: int = 0,
        max_depth: int | None = None,
        feature_weights: str | NDArray | None = None,
        online: bool = False,
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        if isinstance(lifetime, str) and lifetime not in ("sqrt_n", "density"):
            raise ValueError(f"Unknown lifetime string {lifetime!r}. Recognised values: 'sqrt_n', 'density'.")
        if isinstance(feature_weights, str) and feature_weights not in ("variance",):
            raise ValueError(f"Unknown feature_weights string {feature_weights!r}. Recognised value: 'variance'.")
        if online:
            if not isinstance(lifetime, (int, float)):
                raise ValueError(
                    "online=True requires a fixed float lifetime; string/auto-tuned "
                    "lifetimes recompute a master tree per step and are batch-only."
                )
            if max_depth is not None:
                raise ValueError("online=True requires max_depth=None (the projective Mondrian regime).")
        super().__init__(epsilon=epsilon)
        self.lifetime = lifetime
        self.feature_weights = feature_weights
        self.rnd_state = rnd_state
        self.verbose = verbose
        self.max_depth = max_depth
        self.online = online
        self.X: NDArray | None = None
        self.y: NDArray | None = None
        self.rnd_gen = np.random.default_rng(rnd_state)
        self._last_tree = None
        self._online_tree = None
        self._pending_tree = None  # extension scored by the last predict (online)

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Batch-initialize with training data.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        y : ndarray, shape (n,)  — real-valued labels
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] == 0:
            raise ValueError("Training set cannot be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")
        self.X = X.copy()
        self.y = y.copy()
        if self.online:
            self._online_tree = MondrianTree.grow(
                self.X,
                self.rnd_gen,
                lifetime=self.lifetime,
                feature_weights=self.feature_weights,
            )

    def learn_one(
        self,
        x: NDArray,
        y: float,
        precomputed: dict | None = None,
    ) -> None:
        """Incrementally add one observation.

        Parameters
        ----------
        x : array-like, shape (d,)
        y : float
        precomputed : dict, optional
            The update dict from ``predict(..., return_update=True)`` (or
            ``compute_p_value(..., return_update=True)``) for this same ``x``.
            In online mode, if it carries the extension that scored ``x`` the
            persisted tree becomes that exact extension, so the realised stream
            is a single consistent Mondrian process (exact online validity).
            Without it — or if it is stale — a fresh ``extend`` is drawn (still
            valid). The last ``predict`` is also cached, so the plain
            ``predict(x)`` → ``learn_one(x, y)`` loop reuses it automatically.
        """
        x = np.asarray(x, dtype=float).ravel()
        if self.X is None:
            # First training example: initialize with a single-point tree
            self.X = x.reshape(1, -1)
            self.y = np.array([float(y)])
            if self.online:
                self._online_tree = MondrianTree.grow(
                    self.X,
                    self.rnd_gen,
                    lifetime=self.lifetime,
                    feature_weights=self.feature_weights,
                )
            return
        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, float(y))
        if self.online:
            cand = precomputed.get("online_tree") if precomputed else None
            if not self._is_consistent_extension(cand, x):
                cand = getattr(self, "_pending_tree", None)
            self._pending_tree = None
            self._online_tree = (
                cand if self._is_consistent_extension(cand, x) else self._online_tree.extend(x, self.rnd_gen)
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_augmented_tree(self, x: NDArray):
        """Build (batch) or extend (online) the partition on [X_train; x].

        In online mode the extension is cached (``_pending_tree``) so a
        subsequent ``learn_one`` on the same ``x`` can persist the very tree
        that scored it — making the realised stream a single consistent
        Mondrian process (see :meth:`learn_one`).

        Returns ``(root, leaf_star, n)``.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self.X.shape[0]
        if self.online:
            ext_tree = self._online_tree.extend(x, self.rnd_gen)
            self._pending_tree = ext_tree
            tree = ext_tree.root
        else:
            self._pending_tree = None
            X_aug = np.vstack([self.X, x])
            tree = MondrianTree.grow(
                X_aug,
                self.rnd_gen,
                lifetime=self.lifetime,
                x_test=x,
                max_depth=self.max_depth,
                feature_weights=self.feature_weights,
                verbose=self.verbose,
            ).root
        leaf_star = _find_leaf(tree, x)
        return tree, leaf_star, n

    def _build_augmented_tree_with_rng(self, x: NDArray, rnd_gen: np.random.RandomState):
        """Build (batch) or extend (online) the partition on [X_train; x] with a given random generator.

        This is a variant of _build_augmented_tree that takes a random generator as input,
        allowing the tree to be built deterministically without advancing self.rnd_gen.
        This is essential for compute_p_value, where the tree must be the same for all
        candidate y values.

        Returns ``(root, leaf_star, n)``.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self.X.shape[0]
        if self.online:
            # For online mode, we can't easily support a custom rng without
            # modifying the online tree structure. Fall back to using self.rnd_gen.
            ext_tree = self._online_tree.extend(x, rnd_gen)
            self._pending_tree = ext_tree
            tree = ext_tree.root
        else:
            self._pending_tree = None
            X_aug = np.vstack([self.X, x])
            tree = MondrianTree.grow(
                X_aug,
                rnd_gen,
                lifetime=self.lifetime,
                x_test=x,
                max_depth=self.max_depth,
                feature_weights=self.feature_weights,
                verbose=self.verbose,
            ).root
        leaf_star = _find_leaf(tree, x)
        return tree, leaf_star, n

    def _is_consistent_extension(self, cand, x) -> bool:
        """True iff ``cand`` is the persisted online tree extended by exactly ``x``.

        Guards reuse of a scored extension: it must have one more row than the
        persisted tree and that row must be ``x`` — so a stale extension (the
        base has advanced since it was scored) falls back to a fresh ``extend``.
        """
        return (
            cand is not None
            and self._online_tree is not None
            and cand.X.shape[0] == self._online_tree.X.shape[0] + 1
            and np.array_equal(cand.X[-1], np.asarray(x, dtype=float).ravel())
        )

    def _outside_ncms_sorted(self, tree: _MondrianNode, leaf_star: _MondrianNode, n_train: int) -> NDArray:
        """Sorted nonconformity scores of all training points *outside* leaf_star.

        These scores are independent of the candidate ``y`` (the test point only
        affects its own leaf), so they are computed once per prediction — a
        single O(n) tree traversal + O(n log n) sort — and then reused for every
        candidate via binary search (see :meth:`_pvalues_at`).
        """
        outside: list[NDArray] = []
        for leaf in _collect_leaves(tree):
            if leaf is leaf_star:
                continue
            train_idx = leaf.indices[leaf.indices < n_train]
            if len(train_idx) == 0:
                continue
            mu = self.y[train_idx].mean()
            outside.append(np.abs(self.y[train_idx] - mu))
        ext = np.concatenate(outside) if outside else np.empty(0)
        ext.sort()
        return ext

    def _pvalues_at(
        self,
        y_grid: NDArray,
        A: int,
        B: float,
        ext_sorted: NDArray,
        y_star: NDArray,
        n_train: int,
        tau: float,
    ) -> NDArray:
        """Smoothed conformal p-values at candidate values ``y_grid`` (vectorised).

        The candidate-independent ``ext_sorted`` scores are counted by binary
        search (O(log n) per candidate); only the O(A) leaf-local scores are
        recomputed per candidate. This makes the exact solver O(n log n) instead
        of O(n^2). Numerically identical to evaluating the per-candidate p-value
        directly.
        """
        y_grid = np.atleast_1d(np.asarray(y_grid, dtype=float))
        ncm_test = np.abs(A * y_grid - B) / (A + 1)
        _TOL = 1e-9
        n_out = ext_sorted.size

        right = np.searchsorted(ext_sorted, ncm_test + _TOL, side="right")
        left = np.searchsorted(ext_sorted, ncm_test - _TOL, side="left")
        gt = (n_out - right).astype(np.int64)
        eq = (right - left).astype(np.int64)

        if A > 0 and y_star.size > 0:
            inside = np.abs((A + 1) * y_star[None, :] - B - y_grid[:, None]) / (A + 1)
            d = inside - ncm_test[:, None]
            gt = gt + np.sum(d > _TOL, axis=1)
            eq = eq + np.sum(np.abs(d) <= _TOL, axis=1)

        # eq now counts all points with alpha_i == alpha_n (excluding test point itself)
        # Apply tau to the ties, and add 1 for the test point comparing to itself
        p_values = (gt + tau * eq + tau) / (n_train + 1)
        return p_values

    # ------------------------------------------------------------------
    # Event-driven O(n log n) knot sweep
    # ------------------------------------------------------------------
    def _build_knot_events(
        self,
        A: int,
        B: float,
        ext_sorted: NDArray,
        y_star: NDArray,
    ) -> tuple[NDArray, int]:
        """Build knot events with crossing-direction flags for the O(n log n) sweep.

        Returns:
            knots_events: sorted array of (knot_value, flag) tuples, where flag is +1
                         when a point *enters* the {α_i > α_n} set as y increases through
                         the knot, and -1 when it *exits*.
            permanent_ties: count of training points j with α_j(y) ≡ α_n(y) for all y
                           (only happens when A = 1, in which case the single inside
                           point is permanently tied with the test point).

        The algorithm matches mondrian_trees.tex Appendix reg-knots and Prop. reg-exact.
        """
        knots: list[tuple[float, int]] = []
        permanent_ties = 0

        # ------------------------------------------------------------------
        # Outside-leaf points: α_i is constant (independent of y)
        # Each outside point i generates two knots where α_i crosses α_n:
        #   α_n(y) = |A·y - B| / (A+1)      (slope ±A/(A+1), vertex at B/A)
        #   α_i is fixed
        # Crossings at y = B/A ± ((A+1)/A)·α_i
        # Lower root (y_lower = B/A - ((A+1)/A)·α_i): as y increases, α_n overtakes α_i
        #   → for y < y_lower: α_n > α_i (test score higher)
        #   → for y > y_lower: α_i > α_n (training score higher) → *enter* G_i
        #   → tag y_lower with +1
        # Upper root (y_upper = B/A + ((A+1)/A)·α_i): as y increases, α_n moves away
        #   → for y < y_upper: α_i > α_n
        #   → for y > y_upper: α_n > α_i → *exit* G_i
        #   → tag y_upper with -1
        # ------------------------------------------------------------------
        n_out = ext_sorted.size
        if n_out > 0:
            fac = (A + 1) / A
            lower_knots = (B / A) - fac * ext_sorted  # +1 flag
            upper_knots = (B / A) + fac * ext_sorted  # -1 flag
            for y_knot in lower_knots:
                knots.append((float(y_knot), 1))  # entering
            for y_knot in upper_knots:
                knots.append((float(y_knot), -1))  # exiting

        # ------------------------------------------------------------------
        # Inside-leaf points: α_j(y) = |(A+1)·y_j - B - y| / (A+1)
        # This is a V-shape with vertex at y = y_j (slope ∓1/(A+1))
        # The test score α_n has vertex at y = B/A.
        # ------------------------------------------------------------------
        if A == 0:
            # Empty leaf: no inside points. Should not happen here (handled earlier).
            pass
        elif A == 1:
            # Singleton test leaf: the single inside point j has y_j = B/A (since A=1).
            # Corollary cor:slope: α_j(y) ≡ α_n(y) for all y (permanent tie).
            # No crossing knots — the inside point is always tied with the test point.
            permanent_ties = 1
        else:
            # A > 1: each inside point generates two knots:
            #   (i) y = y_j: test score vertex crosses inside score vertex
            #       For y < y_j: α_n > α_j (if y_j > B/A) or α_j > α_n (if y_j < B/A)
            #       At y = y_j: both scores have a kink
            #       As y increases through y_j, α_n overtakes α_j → *enter* G_j → +1
            #   (ii) y = (2B - (A+1)·y_j)/(A-1): reflection root where α_j = α_n again
            #       As y increases through this, α_n moves away → *exit* G_j → -1
            for y_j in y_star:
                # The valid crossing region G_j is bounded by the min and max roots
                refl = (2.0 * B - (A + 1) * y_j) / (A - 1)
                left_root = min(y_j, refl)
                right_root = max(y_j, refl)
                # Enter G_j at left root, exit at right root
                knots.append((float(left_root), 1))
                knots.append((float(right_root), -1))

        if not knots:
            # Degenerate: no outside points and A <= 1 (empty leaf or singleton)
            # Singleton handled via permanent_ties, so this must be empty leaf.
            # But empty leaf should be caught earlier. Return empty events.
            return np.array([], dtype=[("value", float), ("flag", int)]), permanent_ties

        # Sort by knot value, then by flag (process exits before entries to be safe)
        # Actually, we want to aggregate flags at coincident knots, so sort only by value
        knots.sort(key=lambda x: x[0])
        knots_arr = np.array(knots, dtype=[("value", float), ("flag", int)])
        return knots_arr, permanent_ties

    def _evaluate_pvalue_at_point(self, y, A, B, ext_sorted, y_star, n_train, tau, permanent_ties):
        """Evaluate smoothed conformal p-value at an arbitrary y (not just cell midpoints).

        This is used to check knot locations exactly, where p(y) may spike due to
        tie-breaking with tau, potentially making a singleton valid even if
        neighboring cells are invalid.

        Parameters
        ----------
        y : float
            Candidate label at which to evaluate p-value.
        A : int
            Number of training points in test leaf.
        B : float
            Sum of y-values in test leaf.
        ext_sorted : ndarray
            Sorted nonconformity scores of training points outside test leaf.
        y_star : ndarray
            y-values of training points inside test leaf.
        n_train : int
            Number of training points.
        tau : float
            Randomization variable in [0, 1] for smoothed p-value.
        permanent_ties : int
            Count of points permanently tied with test point (A == 1 case).

        Returns
        -------
        float
            Smoothed conformal p-value at y.
        """
        # Test nonconformity score: alpha_n(y) = |A*y - B| / (A+1)
        ncm_test = abs(A * y - B) / (A + 1)

        # Count outside points with alpha_i > ncm_test
        gt = int(np.sum(ext_sorted > ncm_test))

        # Count outside points with alpha_i == ncm_test (for tie-breaking)
        _TOL = 1e-9
        eq = int(np.sum(np.abs(ext_sorted - ncm_test) <= _TOL))

        # Add contributions from inside points (in leaf_star)
        if A > 0 and y_star.size > 0:
            # Inside scores: alpha_j(y) = |(A+1)*y_j - B - y| / (A+1)
            inside_ncm = np.abs((A + 1) * y_star - B - y) / (A + 1)
            gt += int(np.sum(inside_ncm > ncm_test))
            eq += int(np.sum(np.abs(inside_ncm - ncm_test) <= _TOL))

        # p-value = (gt + tau * (eq + 1 + permanent_ties)) / (n_train + 1)
        # The +1 is for the test point itself comparing to itself
        p_val = (gt + tau * (eq + 1 + permanent_ties)) / (n_train + 1)
        return p_val

    def _pvalue_from_sweep(
        self,
        knots_events: NDArray,
        A: int,
        B: float,
        ext_sorted: NDArray,
        y_star: NDArray,
        n_train: int,
        permanent_ties: int,
        epsilon: float,
        tau: float,
    ) -> tuple[float, float, list]:
        """Left-to-right sweep to find p(y) > epsilon interval.

        Returns (lower_bound, upper_bound, valid_knots) where valid_knots is a
        list of knot values where p(y) > epsilon (singleton valid regions).
        Uses O(1) amortised update per knot → O(n) total for all p-value evals.
        """
        # Edge case: no knots (A == 0 with no outside points, or A == 1 with no outside points)
        if knots_events.size == 0:
            # No knots → p(y) is constant everywhere
            # For A == 0: p = tau / (n+1) (no points to compare)
            # For A == 1 with permanent_ties: p = (0 + tau * (1 + 1)) / (n+1) = 2*tau/(n+1)
            p_const = tau * (1 + permanent_ties) / (n_train + 1)
            if p_const > epsilon:
                return -np.inf, np.inf, []
            else:
                # No valid region — return degenerate interval at peak
                peak = B / A if A > 0 else 0.0
                return float(peak), float(peak), [float(peak)]

        # ------------------------------------------------------------------
        # Baseline tally at y_start (far left of all knots)
        # Pick y_start strictly less than first knot
        y_start = knots_events[0]["value"] - 1.0
        # Evaluate all scores at y_start
        ncm_test_start = abs(A * y_start - B) / (A + 1)
        # Outside scores (fixed)
        greater_count = int(np.sum(ext_sorted > ncm_test_start))
        # Inside scores
        if A > 0 and y_star.size > 0:
            for y_j in y_star:
                ncm_j = abs((A + 1) * y_j - B - y_start) / (A + 1)
                if ncm_j > ncm_test_start:
                    greater_count += 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Sweep left-to-right, updating greater_count at each knot
        # ------------------------------------------------------------------
        # We'll evaluate p(y) at the midpoint of each open cell
        # and track the first valid cell (for lower bound) and the last valid cell (for upper)
        # Also track valid knots (singletons at boundary points)
        first_valid_idx = -1
        last_valid_idx = -1
        valid_knots: list[float] = []
        prev_knot = None

        for i in range(len(knots_events) + 1):
            # Determine the cell's right boundary (current knot) and left boundary (prev knot)
            if i < len(knots_events):
                knot_val = knots_events[i]["value"]
                flag = knots_events[i]["flag"]
            else:
                knot_val = None  # rightmost end-cell

            # Compute midpoint of current cell (for potential debugging/logging)
            if prev_knot is None:
                # Leftmost end-cell: extend marginally beyond first knot
                cell_right = knots_events[0]["value"]
                _ = cell_right - max(abs(cell_right), 1.0) * 10.0
            elif knot_val is None:
                # Rightmost end-cell
                cell_left = prev_knot
                _ = cell_left + max(abs(cell_left), 1.0) * 10.0
            else:
                cell_left = prev_knot
                cell_right = knot_val
                _ = (cell_left + cell_right) / 2.0

            # Compute p-value for this cell
            # Apply tau to the ties (test point + any permanent ties)
            p_val = (greater_count + tau * (1 + permanent_ties)) / (n_train + 1)

            # Check validity of cell interior
            if p_val > epsilon:
                if first_valid_idx == -1:
                    first_valid_idx = i  # first valid cell index
                last_valid_idx = i  # update last valid cell index

            # Check validity of the knot itself (if not the first knot)
            # Evaluate p(y) exactly at the knot location
            if knot_val is not None and prev_knot is not None:
                # We're at knot i, which is the boundary between cell i-1 and cell i
                # Check if this knot is valid
                knot_p_val = self._evaluate_pvalue_at_point(
                    knot_val, A, B, ext_sorted, y_star, n_train, tau, permanent_ties
                )
                if knot_p_val > epsilon:
                    valid_knots.append(float(knot_val))

            # Update for next iteration: apply flags at this knot (if any)
            if knot_val is not None:
                greater_count += flag
                prev_knot = knot_val
            elif prev_knot is not None:
                # Already processed all knots, we're done
                break
            else:
                prev_knot = knots_events[0]["value"]

        # ------------------------------------------------------------------
        # Extract interval from valid cells, incorporating valid knots
        # ------------------------------------------------------------------
        if first_valid_idx == -1 and not valid_knots:
            # No valid region at all — return empty interval
            return np.nan, np.nan, []

        # Build the exact prediction set as union of valid cells and valid knots
        # Valid cells: cell i corresponds to open interval (knots[i-1], knots[i]) for i=1..len
        # Cell 0 is (-∞, knots[0]), cell len(knots) is (knots[-1], +∞)

        # If we have valid knots but no valid cells, return singleton(s)
        if first_valid_idx == -1:
            # All cells invalid, only knots valid
            if len(valid_knots) == 1:
                return valid_knots[0], valid_knots[0], valid_knots
            else:
                # Multiple disjoint singletons — return the smallest enclosing interval
                # (conservative but valid)
                return min(valid_knots), max(valid_knots), valid_knots

        # Determine bounds from first and last valid cells, incorporating valid knots
        lo = None
        hi = None

        if first_valid_idx == 0:
            # Cell 0 is (-∞, knots[0])
            lo = -np.inf
            # Check if knot[0] is valid (was already added to valid_knots above)
        else:
            lo = float(knots_events[first_valid_idx - 1]["value"])

        if last_valid_idx == len(knots_events):
            # Rightmost end-cell is valid
            hi = np.inf
        else:
            hi = float(knots_events[last_valid_idx]["value"])

        # If lo is a valid knot but not part of a valid cell, adjust to singleton
        if valid_knots and lo is not None and hi is not None:
            lo_knot = float(knots_events[first_valid_idx - 1]["value"]) if first_valid_idx > 0 else None

            # If lo_knot is valid and the cell to its left is invalid, lo should be singleton
            if lo_knot in valid_knots and first_valid_idx > 0:
                # The knot is valid but the cell to its left is not
                # This is a singleton at the left boundary
                pass  # Keep lo as is, valid_knots will record it

        return (
            float(lo) if lo is not None else float(knots_events[0]["value"]),
            float(hi) if hi is not None else float(knots_events[-1]["value"]),
            valid_knots,
        )

    # ------------------------------------------------------------------
    # compute_p_value
    # ------------------------------------------------------------------

    def compute_p_value(
        self,
        x: NDArray,
        y_cand: float,
        return_update: bool = False,
        tau: float | None = None,
    ) -> float | tuple[float, dict]:
        """Compute conformal p-value for (x, y_cand).

        Parameters
        ----------
        x : array-like, shape (d,)
        y_cand : float
        return_update : bool
            If True, also return a dict with tree info.
        tau : float or None
            Randomization variable for smoothed p-value. If None, draws from U(0,1).
            For property testing and deterministic behavior, pass a fixed tau.

        Returns
        -------
        float or (float, dict)
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self._safe_size_check(self.X)
        if n == 0:
            return (1.0, {}) if return_update else 1.0

        y_cand = float(y_cand)

        # FIX: Use a deterministic seed based on the test point x for tree building
        # This ensures the same tree is built for all candidate y values
        x_seed = int(hash(tuple(np.round(x, 10).flatten())) % (2**31))
        tree_rnd_gen = np.random.default_rng(x_seed)

        # tau must be INDEPENDENT of the tree, drawn from a separate source
        # If tau is not provided, draw it from a deterministic but SEPARATE source
        if tau is None:
            # Use a different deterministic seed for tau (independent of tree)
            tau_seed = (x_seed * 2654435761) % (2**32)  # Prime multiplier for decorrelation
            tau_rnd_gen = np.random.default_rng(tau_seed)
            tau = tau_rnd_gen.uniform(0.0, 1.0)

        tree, leaf_star, _ = self._build_augmented_tree_with_rng(x, tree_rnd_gen)
        leaf_star_train_idx = leaf_star.indices[leaf_star.indices < n]
        A = len(leaf_star_train_idx)
        B = float(self.y[leaf_star_train_idx].sum()) if A > 0 else 0.0
        ext_sorted = self._outside_ncms_sorted(tree, leaf_star, n)
        # tau was already set above if None
        p_val = float(self._pvalues_at(y_cand, A, B, ext_sorted, self.y[leaf_star_train_idx], n, tau)[0])

        # Cache the tree for inspection utilities
        self._last_tree = tree

        if return_update:
            return p_val, {"tree": tree, "leaf_star": leaf_star, "online_tree": self._pending_tree}
        return p_val

    # ------------------------------------------------------------------
    # predict  (exact knot-point solver)
    # ------------------------------------------------------------------

    def predict(
        self,
        x: NDArray,
        epsilon: float | NDArray | None = None,
        return_update: bool = False,
    ) -> ContinuousPredictionSet | MultiLevelPredictionSet | tuple:
        """Predict a conformal interval for test object x.

        Uses an exact O(n) algebraic knot-point solver.  No grid search.

        Parameters
        ----------
        x : array-like, shape (d,)
        epsilon : float or array-like, optional
            Significance level(s).  Defaults to ``self.epsilon``.
        return_update : bool
            If True, return ``(interval, update_dict)`` for ``learn_one``.

        Returns
        -------
        ContinuousPredictionSet or MultiLevelPredictionSet
        """
        x = np.asarray(x, dtype=float).ravel()
        if epsilon is None:
            epsilon = self.epsilon

        n = self._safe_size_check(self.X)
        _inf_interval = self._construct_Gamma((-np.inf, np.inf), epsilon)

        if n == 0:
            if return_update:
                return _inf_interval, {}
            return _inf_interval

        tree, leaf_star, _ = self._build_augmented_tree(x)
        # Cache the tree for inspection utilities (draw_partition etc.),
        # regardless of which prediction branch is taken below.
        self._last_tree = tree

        leaf_star_train_idx = leaf_star.indices[leaf_star.indices < n]
        A = len(leaf_star_train_idx)
        B = self.y[leaf_star_train_idx].sum() if A > 0 else 0.0

        # ---- Draw ONE tau for the entire predict call to ensure deterministic behavior.
        # Use it for smoothed p-value calculation. This must be done BEFORE any A==0 checks.
        tau = self.rnd_gen.uniform(0.0, 1.0)

        # ---- A = 0: empty leaf → total epistemic uncertainty -----------
        if A == 0:
            ext_sorted = self._outside_ncms_sorted(tree, leaf_star, n)

            # Since test score is 0, rank depends entirely on outside scores >= 0
            gt = int(np.sum(ext_sorted > 0))
            eq = int(np.sum(ext_sorted == 0))
            p_const = (gt + tau * (eq + 1)) / (n + 1)

            eps_arr = np.atleast_1d(np.asarray(epsilon, dtype=float))
            predictions = {}
            for eps_val in eps_arr:
                if p_const > eps_val:
                    predictions[float(eps_val)] = self._construct_Gamma((-np.inf, np.inf), float(eps_val))
                else:
                    predictions[float(eps_val)] = self._construct_Gamma((np.nan, np.nan), float(eps_val))

            if hasattr(epsilon, "__iter__"):
                result = MultiLevelPredictionSet(predictions)
            else:
                result = predictions[float(eps_arr[0])]

            if return_update:
                return result, {"tree": tree, "leaf_star": leaf_star, "online_tree": self._pending_tree}
            return result

        # ---- Collect knot points ----------------------------------------
        # The candidate-independent outside-leaf scores are computed once here
        # and reused for the knots and for every p-value evaluation below.
        ext_sorted = self._outside_ncms_sorted(tree, leaf_star, n)

        # Outside-leaf_star points: ncm_i is fixed → rank changes when
        #   ncm_test(y) = ncm_i  ↔  y = (B ± ncm_i*(A+1)) / A
        knots: list[float] = []
        if ext_sorted.size:
            fac = (A + 1) / A
            knots.extend(((B / A) + fac * ext_sorted).tolist())
            knots.extend(((B / A) - fac * ext_sorted).tolist())

        # Inside-leaf_star points:
        #   always: y_j
        #   if A > 1: (2B − (A+1)*y_j) / (A−1)
        y_star = self.y[leaf_star_train_idx]
        knots.extend(y_star.tolist())
        if A > 1:
            knots.extend(((2.0 * B - (A + 1) * y_star) / (A - 1)).tolist())

        # Sort and de-duplicate (for potential future use)

        # Build knot events with +1/-1 flags
        knots_events, permanent_ties = self._build_knot_events(A, B, ext_sorted, y_star)

        # Perform left-to-right sweep to find valid region
        lo, hi, valid_knots = self._pvalue_from_sweep(
            knots_events,
            A,
            B,
            ext_sorted,
            y_star,
            n,
            permanent_ties,
            epsilon if not hasattr(epsilon, "__iter__") else epsilon[0],
            tau,
        )

        # Build result with correct topology
        eps_arr = np.atleast_1d(np.asarray(epsilon, dtype=float))
        predictions = {}

        for eps_val in eps_arr:
            lo, hi, valid_knots = self._pvalue_from_sweep(
                knots_events, A, B, ext_sorted, y_star, n, permanent_ties, float(eps_val), tau
            )

            if len(valid_knots) == 1 and lo == hi == valid_knots[0]:
                predictions[float(eps_val)] = self._construct_Gamma((lo, hi), float(eps_val))
            else:
                predictions[float(eps_val)] = self._construct_Gamma((lo, hi), float(eps_val))

        if hasattr(epsilon, "__iter__"):
            result = MultiLevelPredictionSet(predictions)
        else:
            result = predictions[float(eps_arr[0])]

        if return_update:
            return result, {"tree": tree, "leaf_star": leaf_star, "online_tree": self._pending_tree}

        return result

    @staticmethod
    def _extract_interval(
        knots: NDArray,
        midpoints: NDArray,
        p_vals: NDArray,
        epsilon: float,
    ) -> tuple[float, float]:
        """Return (lower, upper) convex hull of valid region.

        The prediction set is {y : p(y) > epsilon} (ALRW2 convention).
        """
        threshold = epsilon
        valid = p_vals > threshold
        if not np.any(valid):
            # No candidate passes — return empty interval
            return np.nan, np.nan
        # Knot boundaries: the valid midpoints index into intervals between knots.
        # Interval i spans [knots[i-1], knots[i]] with midpoint midpoints[i].
        # midpoints[0] < knots[0] → left-open interval, midpoints[-1] > knots[-1] → right-open.
        valid_idx = np.where(valid)[0]
        lo_idx = valid_idx[0]
        hi_idx = valid_idx[-1]
        # Lower bound: left edge of first valid segment
        lo = -np.inf if lo_idx == 0 else float(knots[lo_idx - 1])
        # Upper bound: right edge of last valid segment
        hi = np.inf if hi_idx == len(knots) else float(knots[hi_idx])
        return lo, hi

    def _make_result(self, epsilon, lo: float, hi: float):
        """Construct a ContinuousPredictionSet or MultiLevelPredictionSet."""
        if hasattr(epsilon, "__iter__"):
            return MultiLevelPredictionSet({float(eps): self._construct_Gamma((lo, hi), float(eps)) for eps in epsilon})
        return self._construct_Gamma((lo, hi), float(epsilon))

    def _make_multi_result(
        self,
        epsilon,
        knots: NDArray,
        midpoints: NDArray,
        p_vals: NDArray,
    ):
        predictions = {}
        for eps in epsilon:
            lo, hi = self._extract_interval(knots, midpoints, p_vals, float(eps))
            predictions[float(eps)] = self._construct_Gamma((lo, hi), float(eps))
        return MultiLevelPredictionSet(predictions)


class ConformalMondrianForestRegressor(_MondrianRegressorInspection, ConformalRegressor):
    """Conformal predictor using an ensemble of Mondrian trees for regression.

    Builds T Mondrian trees per prediction step from the same augmented bag
    with different seeds. The per-tree NCMs are averaged to give a single
    smoothed NCM per point, reducing variance while maintaining exact validity.

        α_i = (1/T) Σ_t |y_i − μ_{leaf_t(x_i)}|

    The prediction set is found via an exact O(Mn log(Mn)) sweepline solver.
    For forests, the prediction set may be a union of disjoint intervals (the
    p-value is not guaranteed to be quasi-concave). By default, the convex hull
    is returned for compatibility with downstream tasks; set `return_interval`
    to `False` to get the exact union of intervals.

    Parameters
    ----------
    n_trees : int
        Number of Mondrian trees. Default 10.
    lifetime : float
        Mondrian depth budget L. Default 1.0.
    epsilon : float
        Default significance level. Default 0.1.
    rnd_state : int, optional
        Seed for the root RNG (each tree gets a derived seed).
    verbose : int
        Verbosity level (0 = silent).
    n_jobs : int
        Number of parallel workers for tree construction. Default 1.
    grid_resolution : int
        Number of grid points for the initial interval search. Default 200.
    bisection_tol : float
        Convergence tolerance for bisection refinement. Default 1e-6.
    return_interval : bool, optional
        If True, return the convex hull (single interval) of the exact prediction
        set. If False, return the exact prediction set as a union of intervals.
        Default True for backward compatibility.
    max_depth : int or None, optional
        Hard depth cap for Mondrian trees. Default None.
    online : bool, optional
        If True, use the online Mondrian forest mode. Default False.
    """

    _SAVE_PARAMS: tuple = (
        "n_trees",
        "lifetime",
        "epsilon",
        "rnd_state",
        "verbose",
        "n_jobs",
        "grid_resolution",
        "bisection_tol",
        "max_depth",
        "online",
        "return_interval",
    )
    _SAVE_STATE: tuple = ("X", "y", "_forest")

    def __init__(
        self,
        n_trees: int = 10,
        lifetime: float = 1.0,
        epsilon: float = 0.1,
        rnd_state: int | None = None,
        verbose: int = 0,
        n_jobs: int = 1,
        grid_resolution: int = 200,
        bisection_tol: float = 1e-6,
        max_depth: int | None = None,
        online: bool = False,
        return_interval: bool = True,
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        if online:
            if not isinstance(lifetime, (int, float)):
                raise ValueError(
                    "online=True requires a fixed float lifetime; string/auto-tuned "
                    "lifetimes recompute a master tree per step and are batch-only."
                )
            if max_depth is not None:
                raise ValueError("online=True requires max_depth=None (the projective Mondrian regime).")
        super().__init__(epsilon=epsilon)
        self.n_trees = n_trees
        self.lifetime = lifetime
        self.rnd_state = rnd_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.grid_resolution = grid_resolution
        self.bisection_tol = bisection_tol
        self.max_depth = max_depth
        self.online = online
        self.return_interval = return_interval
        self.X: NDArray | None = None
        self.y: NDArray | None = None
        self.rnd_gen = np.random.default_rng(rnd_state)
        self._last_tree = None
        self._last_seeds = None
        self._last_x = None
        self._forest = None
        self._pending_forest = None

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Batch-initialize with training data."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] == 0:
            raise ValueError("Training set cannot be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")
        self.X = X.copy()
        self.y = y.copy()
        if self.online:
            seeds = self.rnd_gen.integers(0, 2**31, self.n_trees)
            self._forest = [
                MondrianTree.grow(self.X, np.random.default_rng(int(s)), lifetime=self.lifetime) for s in seeds
            ]

    def learn_one(
        self,
        x: NDArray,
        y: float,
        precomputed: dict | None = None,
    ) -> None:
        """Incrementally add one observation.

        In online mode the forest extensions that scored ``x`` (from
        ``precomputed`` or the last ``predict``) are persisted, so each tree's
        stream is a single consistent Mondrian process; otherwise fresh
        extensions are drawn.
        """
        x = np.asarray(x, dtype=float).ravel()
        if self.X is None:
            raise ValueError("Must call learn_initial_training_set first")
        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, float(y))
        if self.online:
            cand = precomputed.get("online_forest") if precomputed else None
            if not self._is_consistent_forest(cand, x):
                cand = getattr(self, "_pending_forest", None)
            self._pending_forest = None
            if self._is_consistent_forest(cand, x):
                self._forest = cand
            else:
                self._forest = [t.extend(x, self.rnd_gen) for t in self._forest]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_consistent_forest(self, cand, x) -> bool:
        """True iff ``cand`` is the persisted forest with every tree extended by exactly ``x``."""
        if cand is None or self._forest is None or len(cand) != len(self._forest):
            return False
        xr = np.asarray(x, dtype=float).ravel()
        base_n = self._forest[0].X.shape[0]
        return all(t.X.shape[0] == base_n + 1 and np.array_equal(t.X[-1], xr) for t in cand)

    def _forest_summaries(self, X_aug, x, n, seeds: NDArray | None = None):
        """Per-tree ``(base_ncm, ls_idx, ls_mu)`` summaries + viz/reuse caches.

        Batch: T fresh trees from seeds (joblib). Online: extend each persistent
        tree and summarise, caching the extensions for ``learn_one`` reuse.

        Parameters
        ----------
        X_aug : array-like, shape (n+1, d)
            Augmented feature matrix with test point appended.
        x : array-like, shape (d,)
            Test point features.
        n : int
            Number of training points.
        seeds : array-like or None
            Seeds for each tree. If None, uses _tree_seeds().
        """
        if self.online:
            # In online mode, use deterministic seeds if provided
            if seeds is None:
                pending = [t.extend(x, self.rnd_gen) for t in self._forest]
            else:
                pending = [t.extend(x, np.random.default_rng(int(s))) for t, s in zip(self._forest, seeds)]
            self._pending_forest = pending
            self._last_seeds = None
            self._last_x = x.copy()
            self._last_tree = pending[0].root
            return [_summarize_tree_reg(t.root, self.y, n, x) for t in pending]

        # Batch mode
        if seeds is None:
            seeds = self._tree_seeds()
        self._last_seeds = seeds.copy()
        self._last_x = x.copy()
        self._pending_forest = None
        _rng0 = np.random.default_rng(int(seeds[0]))
        self._last_tree = MondrianTree.grow(X_aug, _rng0, lifetime=self.lifetime, max_depth=self.max_depth).root
        return Parallel(n_jobs=self.n_jobs)(
            delayed(_build_tree_summary_reg)(s, X_aug, self.y, n, self.lifetime, self.max_depth) for s in seeds
        )

    def _tree_seeds(self) -> NDArray:
        """Generate n_trees integer seeds from the root RNG."""
        return self.rnd_gen.integers(0, 2**31 - 1, size=self.n_trees)

    def _compute_avg_ncms(self, x: NDArray, y_cand: float) -> tuple[NDArray, float]:
        """Compute averaged NCMs for all training points + test point.

        Returns
        -------
        avg_train_ncms : ndarray, shape (n,)
            Averaged NCM for each training point.
        avg_test_ncm : float
            Averaged NCM for the test point.
        """
        n = self.X.shape[0]
        X_aug = np.vstack([self.X, x.reshape(1, -1)])
        results = self._forest_summaries(X_aug, x, n)

        # Accumulate per-tree NCMs
        # base_ncm[i] = |y_i - mu_leaf_i| (training-only mean)
        # For test point: ncm_test = |A*y_cand - B| / (A+1)
        # For inside points j: ncm_j = |(A+1)*y_j - B - y_cand| / (A+1)
        sum_train_ncms = np.zeros(n, dtype=float)
        sum_test_ncm = 0.0

        for base_ncm, ls_train_idx, ls_mu_train in results:
            A = len(ls_train_idx)
            B = ls_mu_train * A if A > 0 else 0.0

            # Test NCM for this tree
            test_ncm_t = abs(A * y_cand - B) / (A + 1)
            sum_test_ncm += test_ncm_t

            # Training NCMs: start from base, override leaf_star
            ncm_t = base_ncm.copy()
            if A > 0:
                y_star = self.y[ls_train_idx]
                ncm_t[ls_train_idx] = np.abs((A + 1) * y_star - B - y_cand) / (A + 1)

            sum_train_ncms += ncm_t

        T = float(self.n_trees)
        return sum_train_ncms / T, sum_test_ncm / T

    def _compute_avg_ncms_with_seeds(self, x: NDArray, y_cand: float, seeds: NDArray) -> tuple[NDArray, float]:
        """Compute averaged NCMs with deterministic seeds.

        Parameters
        ----------
        x : array-like, shape (d,)
        y_cand : float
        seeds : array-like, shape (n_trees,)
            Deterministic seeds for each tree.

        Returns
        -------
        avg_train_ncms : ndarray, shape (n,)
        avg_test_ncm : float
        """
        n = self.X.shape[0]
        X_aug = np.vstack([self.X, x.reshape(1, -1)])
        results = self._forest_summaries(X_aug, x, n, seeds=seeds)

        sum_train_ncms = np.zeros(n, dtype=float)
        sum_test_ncm = 0.0

        for base_ncm, ls_train_idx, ls_mu_train in results:
            A = len(ls_train_idx)
            B = ls_mu_train * A if A > 0 else 0.0

            test_ncm_t = abs(A * y_cand - B) / (A + 1)
            sum_test_ncm += test_ncm_t

            ncm_t = base_ncm.copy()
            if A > 0:
                y_star = self.y[ls_train_idx]
                ncm_t[ls_train_idx] = np.abs((A + 1) * y_star - B - y_cand) / (A + 1)

            sum_train_ncms += ncm_t

        T = float(self.n_trees)
        return sum_train_ncms / T, sum_test_ncm / T

    # ------------------------------------------------------------------
    # compute_p_value
    # ------------------------------------------------------------------

    def compute_p_value(
        self,
        x: NDArray,
        y_cand: float,
        return_update: bool = False,
        tau: float | None = None,
    ) -> float | tuple[float, dict]:
        """Compute conformal p-value for (x, y_cand).

        Parameters
        ----------
        x : array-like, shape (d,)
        y_cand : float
        return_update : bool
        tau : float or None
            Randomization variable for smoothed p-value. If None, draws from U(0,1).
            For property testing and deterministic behavior, pass a fixed tau.

        Returns
        -------
        float or (float, dict)
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self._safe_size_check(self.X)
        if n == 0:
            return (1.0, {}) if return_update else 1.0

        # FIX: Use deterministic seeds based on x to ensure the same forest for all y
        # The seed for each tree must be deterministic based on x, not on self.rnd_gen
        # Use a hash of x to create reproducible seeds for each tree
        x_seed = int(hash(tuple(np.round(x, 10).flatten())) % (2**31))
        seeds = np.array([x_seed + i for i in range(self.n_trees)])

        # tau is drawn ONCE in predict() and passed to compute_p_value
        # If tau is not provided, draw it from a SEPARATE deterministic source
        # Tau must be independent of the tree-building process
        if tau is None:
            # Use a different deterministic seed for tau (independent of tree seeds)
            tau_seed = (x_seed * 2654435761) % (2**32)  # Prime multiplier for decorrelation
            tau = np.random.default_rng(tau_seed).uniform(0.0, 1.0)

        # Compute NCMs with deterministic seeds
        avg_train_ncms, avg_test_ncm = self._compute_avg_ncms_with_seeds(x, float(y_cand), seeds)

        all_ncms = np.append(avg_train_ncms, avg_test_ncm)
        p_val = float(self._compute_p_value(all_ncms, tau))

        if return_update:
            return p_val, {"online_forest": getattr(self, "_pending_forest", None)}
        return p_val

    # ------------------------------------------------------------------
    # predict  (grid + bisection)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Forest indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of trees in the forest."""
        return self.n_trees

    def __iter__(self):
        """Iterate over individual tree views."""
        for i in range(self.n_trees):
            yield self[i]

    def __getitem__(self, i: int) -> ConformalMondrianTreeRegressor:
        """Return a single-tree view for tree ``i`` from the last predict/compute_p_value call.

        The returned object is a :class:`ConformalMondrianTreeRegressor` whose
        ``_last_tree`` is the exact i-th tree from the last call, enabling all
        T1/T2 inspection and visualisation methods. Its own ``predict()`` would
        rebuild a fresh tree — this object is intended for *inspection only*.

        Parameters
        ----------
        i : int
            Tree index, ``0 <= i < n_trees``.

        Raises
        ------
        RuntimeError
            If ``predict()`` or ``compute_p_value()`` has not been called yet.
        IndexError
            If ``i`` is out of range.
        """
        if self.online:
            pending = getattr(self, "_pending_forest", None)
            if pending is None:
                raise RuntimeError("No trees cached. Call predict() or compute_p_value() first.")
            if not (0 <= i < self.n_trees):
                raise IndexError(f"Tree index {i} out of range [0, {self.n_trees}).")
            view = ConformalMondrianTreeRegressor(
                lifetime=self.lifetime,
                epsilon=self.epsilon,
                max_depth=self.max_depth,
            )
            view.X = self.X
            view.y = self.y
            view._last_tree = pending[i].root
            return view

        if self._last_seeds is None:
            raise RuntimeError("No trees cached. Call predict() or compute_p_value() first.")
        if not (0 <= i < self.n_trees):
            raise IndexError(f"Tree index {i} out of range [0, {self.n_trees}).")

        X_aug = np.vstack([self.X, self._last_x])
        rng_i = np.random.default_rng(int(self._last_seeds[i]))
        tree_i = MondrianTree.grow(X_aug, rng_i, lifetime=self.lifetime, max_depth=self.max_depth).root

        view = ConformalMondrianTreeRegressor(
            lifetime=self.lifetime,
            epsilon=self.epsilon,
            max_depth=self.max_depth,
        )
        view.X = self.X
        view.y = self.y
        view._last_tree = tree_i
        return view

    def predict(
        self,
        x: NDArray,
        epsilon: float | NDArray | None = None,
        return_update: bool = False,
        return_interval: bool | None = None,
    ) -> ContinuousPredictionSet | MultiLevelPredictionSet | tuple:
        """Predict a conformal interval for test object x.

        Uses an exact O(Mn log(Mn)) sweepline solver to locate p-value crossings.
        For forests, the prediction set may be a union of disjoint intervals.

        Parameters
        ----------
        x : array-like, shape (d,)
        epsilon : float or array-like, optional
        return_update : bool
        return_interval : bool or None, optional
            If True, return the convex hull (single interval) of the exact prediction
            set. If False, return the exact union of intervals as a DiscretePredictionSet.
            If None (default), use ``self.return_interval``.

        Returns
        -------
        ContinuousPredictionSet or MultiLevelPredictionSet
            Returns a DiscretePredictionSet with multiple intervals if return_interval=False
            and the exact prediction set is disconnected.
        """
        x = np.asarray(x, dtype=float).ravel()
        if epsilon is None:
            epsilon = self.epsilon

        n = self._safe_size_check(self.X)
        if n == 0:
            result = self._make_result(epsilon, -np.inf, np.inf)
            if return_update:
                return result, {}
            return result

        # Use return_interval parameter if provided, otherwise fall back to self.return_interval
        if return_interval is None:
            return_interval = self.return_interval

        # ---- Build all T trees ONCE (batch: seeds; online: extend forest) --
        X_aug = np.vstack([self.X, x.reshape(1, -1)])
        summaries = self._forest_summaries(X_aug, x, n)

        T = float(self.n_trees)
        _TOL = 1e-9
        # Draw ONE tau for the entire predict call to ensure deterministic save/load.
        # Use it for smoothed p-value calculation.
        tau = self.rnd_gen.uniform(0.0, 1.0)

        def p_value_at(y_cand: float) -> float:
            """Evaluate p(y_cand) using pre-built tree summaries and smoothed tau."""
            sum_train = np.zeros(n, dtype=float)
            sum_test = 0.0
            for base_ncm, ls_idx, ls_mu in summaries:
                A = len(ls_idx)
                B = ls_mu * A if A > 0 else 0.0
                test_t = abs(A * y_cand - B) / (A + 1)
                sum_test += test_t
                ncm_t = base_ncm.copy()
                if A > 0:
                    y_star = self.y[ls_idx]
                    ncm_t[ls_idx] = np.abs((A + 1) * y_star - B - y_cand) / (A + 1)
                sum_train += ncm_t
            avg_train = sum_train / T
            avg_test = sum_test / T
            alpha_y = avg_test
            gt = int(np.sum(avg_train > alpha_y + _TOL))
            eq = int(np.sum(np.abs(avg_train - alpha_y) <= _TOL))  # exclude test point, apply tau below
            return float((gt + tau * eq) / (n + 1))

        # ---- Exact O(Mn log(Mn)) Sweepline Solver ----------------------
        # Use the Numba-accelerated solver from _forest_solver module
        M = len(summaries)
        eps_arr = np.atleast_1d(np.asarray(epsilon, dtype=float))
        predictions = {}

        for eps_val in eps_arr:
            if return_interval:
                lo, hi = forest_sweepline_solver(summaries, self.y, float(eps_val), tau, n, M, return_exact=False)
                # Single interval mode: pass as (lo, hi) tuple
                if np.isnan(lo) and np.isnan(hi):
                    predictions[float(eps_val)] = self._construct_Gamma((np.nan, np.nan), float(eps_val))
                else:
                    predictions[float(eps_val)] = self._construct_Gamma((lo, hi), float(eps_val))
            else:
                intervals, Y_eval = forest_sweepline_solver(
                    summaries, self.y, float(eps_val), tau, n, M, return_exact=True
                )
                # intervals is a list of (lo, hi) tuples for the exact prediction set
                if not intervals:
                    predictions[float(eps_val)] = self._construct_Gamma([], float(eps_val))
                else:
                    predictions[float(eps_val)] = self._construct_Gamma(intervals, float(eps_val))

        if hasattr(epsilon, "__iter__"):
            result = MultiLevelPredictionSet(predictions)
        else:
            result = predictions[float(eps_arr[0])]

        if return_update:
            return result, {"online_forest": getattr(self, "_pending_forest", None)}
        return result

    @staticmethod
    def _make_result(epsilon, lo: float, hi: float):
        if hasattr(epsilon, "__iter__"):
            return MultiLevelPredictionSet(
                {float(eps): ContinuousPredictionSet([(lo, hi)], float(eps)) for eps in epsilon}
            )
        return ContinuousPredictionSet([(lo, hi)], float(epsilon))
