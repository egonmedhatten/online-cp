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

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize
from scipy.spatial.distance import cdist, pdist, squareform

# Optional numba for Lasso homotopy speedup
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


__all__ = [
    "ConformalRidgeRegressor",
    "ConformalNearestNeighboursRegressor",
    "KernelConformalRidgeRegressor",
    "ConformalLassoRegressor",
    "ConformalPredictionInterval",
    "MultiLevelPredictionInterval",
]


default_epsilon = 0.1


class ConformalPredictionInterval:
    """A prediction interval produced by a conformal regressor.

    Parameters
    ----------
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.
    epsilon : float
        Significance level at which the interval was constructed.
    """

    def __init__(self, lower: float, upper: float, epsilon: float) -> None:
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon

    def __contains__(self, y: float) -> bool:
        return self.lower <= y <= self.upper

    def width(self) -> float:
        return self.upper - self.lower

    def __repr__(self):
        return repr((self.lower, self.upper))

    def __str__(self):
        return f"({self.lower}, {self.upper})"


class MultiLevelPredictionInterval:
    """Prediction intervals at multiple significance levels.

    Returned when ``predict`` is called with an array-like ``epsilon``.

    Parameters
    ----------
    predictions : dict
        Mapping ``{epsilon: ConformalPredictionInterval}``.
    """

    def __init__(self, predictions: dict[float, ConformalPredictionInterval]) -> None:
        self._predictions = dict(sorted(predictions.items()))

    @property
    def levels(self) -> list[float]:
        """Sorted list of significance levels."""
        return list(self._predictions.keys())

    def __getitem__(self, eps: float) -> ConformalPredictionInterval:
        return self._predictions[eps]

    def __iter__(self):
        return iter(self._predictions.items())

    def __len__(self) -> int:
        return len(self._predictions)

    def __contains__(self, y: float) -> bool:
        """True if y is covered at all levels."""
        return all(y in interval for interval in self._predictions.values())

    def coverage(self, y: float) -> dict[float, bool]:
        """Return dict of {epsilon: bool} indicating coverage at each level."""
        return {eps: (y in interval) for eps, interval in self._predictions.items()}

    def __repr__(self):
        parts = [f"  ε={eps}: {interval}" for eps, interval in self._predictions.items()]
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

    def _construct_Gamma(self, lower, upper, epsilon):
        return ConformalPredictionInterval(lower, upper, epsilon)

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
        try:
            upper = u_dic[int(np.ceil((1 - epsilon) * n))]
        except KeyError:
            upper = np.inf
        return upper

    @staticmethod
    def _get_lower(l_dic, epsilon, n):
        try:
            lower = l_dic[int(np.floor(epsilon * n))]
        except KeyError:
            lower = -np.inf
        return lower

    @staticmethod
    def _vectorised_l_and_u(A, B):
        """A and B are columns"""
        # Calculate differences
        differences = B[-1] - B

        # Create an array to store results
        l = np.empty_like(B, dtype=float)
        u = np.empty_like(B, dtype=float)

        # Calculate values where differences are positive
        mask = differences > 0
        l[mask] = (A[mask] - A[-1]) / differences[mask]
        u[mask] = (A[mask] - A[-1]) / differences[mask]

        # Assign positive infinity where differences are non-positive
        l[~mask] = -np.inf
        u[~mask] = np.inf

        l = np.sort(l, axis=0)[1:]
        u = np.sort(u, axis=0)[:-1]

        # These are just to avoid messing with the python indexing. Could probably be removed for efficiency
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
                    # Update XTX_inv using the Sherman-Morrison formula
                    # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
                    self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
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
                # Update XTX_inv using the Sherman-Morrison formula
                # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
                self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
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
        ConformalPredictionInterval
        | MultiLevelPredictionInterval
        | tuple[ConformalPredictionInterval | MultiLevelPredictionInterval, dict[str, Any]]
    ):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), bounds="both")
        (-inf, inf)
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
                        predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                        result = MultiLevelPredictionInterval(predictions)
                    else:
                        result = self._construct_Gamma(-np.inf, np.inf, epsilon)
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
                        predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                        result = MultiLevelPredictionInterval(predictions)
                    else:
                        result = self._construct_Gamma(-np.inf, np.inf, epsilon)
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
                        predictions[eps] = self._construct_Gamma(lo, up, eps)
                    result = MultiLevelPredictionInterval(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
                    result = self._construct_Gamma(lower, upper, epsilon)
            elif bounds == "lower":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        lo = self._get_lower(l_dic=l_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma(lo, np.inf, eps)
                    result = MultiLevelPredictionInterval(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma(lower, np.inf, epsilon)
            elif bounds == "upper":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        up = self._get_upper(u_dic=u_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma(-np.inf, up, eps)
                    result = MultiLevelPredictionInterval(predictions)
                else:
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma(-np.inf, upper, epsilon)
            else:
                raise ValueError('bounds must be "both", "lower", or "upper"')
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            X = x.reshape(1, -1)
            XTXinv = None
            A = None
            B = None

            if hasattr(epsilon, "__iter__"):
                predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                result = MultiLevelPredictionInterval(predictions)
            else:
                result = self._construct_Gamma(-np.inf, np.inf, epsilon)

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
        Tune ridge parameter with Generalized cross validation https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/golub.heath.wahba.pdf
        """
        XTX = self.X.T @ self.X
        n = self.X.shape[0]
        In = np.identity(n)

        def GCV(a):
            try:
                A = self.X @ np.linalg.inv(XTX + a * self.Id) @ self.X.T
                max_diag_H = np.max(np.diag(A))
                if max_diag_H > 1:
                    return np.inf
                return (1 / n) * np.linalg.norm((In - A) @ self.y) ** 2 / ((1 / n) * np.trace(In - A)) ** 2
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return np.inf

        # Initial guess
        if a0 is None:
            a0 = 1e-6  # Just a small pertubation to avoid numerical issues

        # Bounds to ensure a >= 0
        res = minimize(
            GCV, x0=a0, bounds=Bounds(lb=1e-6, keep_feasible=True)
        )  # May be relevant to pass some arguments here, or even use another minimizer.
        a = res.x[0]

        if self.verbose > 0:
            print(f"New ridge parameter: {a}")
        self.change_ridge_parameter(a)


class ConformalNearestNeighboursRegressor(ConformalRegressor):
    """
    Conformal k-nearest neighbours regressor (ALRW2 §2.4).

    Produces prediction intervals with guaranteed coverage using the
    nonconformity measure |y - ŷ_kNN| where ŷ_kNN is the leave-one-out
    k-nearest neighbours prediction.

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
    )
    _SAVE_STATE: tuple = ("X", "y", "D")
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

        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=float)
            self.D = self.distance_func(self.X)
        else:
            if precomputed is not None and "D" in precomputed:
                self.D = precomputed["D"]
            else:
                d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
                self.D = self._update_distance_matrix(self.D, d)
            self.X = np.vstack([self.X, x.reshape(1, -1)])
            self.y = np.append(self.y, y)

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        bounds: str = "both",
        return_update: bool = False,
    ) -> (
        ConformalPredictionInterval
        | MultiLevelPredictionInterval
        | tuple[ConformalPredictionInterval | MultiLevelPredictionInterval, dict[str, Any]]
    ):
        """Predict a conformal prediction interval for test object x.

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
        result : ConformalPredictionInterval or MultiLevelPredictionInterval
        precomputed : dict, optional
            Returned if return_update=True. Contains 'D'.
        """
        x = np.asarray(x).ravel()

        if epsilon is None:
            epsilon = self.epsilon

        n = self._safe_size_check(self.X)

        if n == 0:
            # No training data
            if hasattr(epsilon, "__iter__"):
                predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                result = MultiLevelPredictionInterval(predictions)
            else:
                result = self._construct_Gamma(-np.inf, np.inf, epsilon)
            if return_update:
                return result, {"D": None}
            return result

        # Augment distance matrix with test point
        d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
        D_aug = self._update_distance_matrix(self.D, d)
        n_aug = n + 1  # augmented size

        # Check significance level is feasible
        eps_check = max(epsilon) if hasattr(epsilon, "__iter__") else epsilon
        min_needed = 2 if bounds == "both" else 1
        if not (eps_check >= min_needed / n_aug):
            if hasattr(epsilon, "__iter__"):
                predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                result = MultiLevelPredictionInterval(predictions)
            else:
                result = self._construct_Gamma(-np.inf, np.inf, epsilon)
            if return_update:
                return result, {"D": D_aug}
            return result

        # Compute leave-one-out k-NN predictions for all training points
        # in the augmented set. The test point's label is unknown, so we
        # need to determine the interval of y values that would be included.
        #
        # Key insight: For the n training points, their k-NN predictions
        # may or may not change when the test point is added (only if the
        # test point enters their k-neighbourhood). For the test point,
        # its k-NN prediction is always computed from training labels only
        # (since we exclude self).
        #
        # Strategy: compute all training nonconformity scores under the
        # augmented distance matrix, then find the threshold.

        agg_func = np.mean if self.aggregation == "mean" else np.median

        # For each training point i (0..n-1), compute its leave-one-out
        # k-NN prediction in the augmented set (which includes the test point)
        D_work = D_aug.copy()
        np.fill_diagonal(D_work, np.inf)

        # For training points (0..n-1): their predictions don't depend on
        # the test label since we only use labels of their k neighbours.
        # BUT if the test point (index n) is among a training point's k-NN,
        # its label y_test would appear in the average. We handle this by
        # noting that y_test is what we're searching over.
        #
        # However, for the standard conformal regressor approach, we compute
        # the nonconformity scores as |y_i - hat_y_i| where hat_y_i is
        # computed with ALL labels known (including the hypothesized test label).
        # This makes the prediction interval depend on the test label y in a
        # complex way when the test point enters a training point's k-neighbourhood.
        #
        # Simplification (valid and common): Use the "deleted" approach where
        # for each training point i, the k-NN prediction uses the bag without i.
        # The test point's label only affects training points that have it as
        # a k-NN. For the test point itself, its prediction only uses training
        # labels.
        #
        # Even simpler (and what most practical k-NN conformal regressors do):
        # Compute training residuals on the ORIGINAL training set (without the
        # test point), then compare the test point's residual against them.
        # This is valid because the nonconformity scores are exchangeable.

        # Training residuals (on original training set, leave-one-out)
        y_hat_train = self._knn_predictions(self.D, self.y)
        alpha_train = np.abs(self.y - y_hat_train)

        # Test point k-NN prediction (using training data only)
        # Its k nearest neighbours among training points.
        # Use lexsort (distance primary, y-value secondary) for canonical
        # tie-breaking that is independent of insertion order.
        k_test = min(self.k, n)
        test_knn_idx = np.lexsort((self.y, d))[:k_test]
        y_hat_test = agg_func(self.y[test_knn_idx])

        # The test nonconformity score is |y - y_hat_test| and we need
        # p(y) = |{i: alpha_i >= |y - y_hat_test|}| / (n+1) > epsilon
        # The prediction interval is: y_hat_test +/- alpha_(ceil((1-eps)(n+1)))

        # Sort training residuals
        alpha_sorted = np.sort(alpha_train)

        # Build the interval
        if hasattr(epsilon, "__iter__"):
            predictions = {}
            for eps in epsilon:
                lo, up = self._compute_interval(alpha_sorted, y_hat_test, eps, n_aug, bounds)
                predictions[eps] = self._construct_Gamma(lo, up, eps)
            result = MultiLevelPredictionInterval(predictions)
        else:
            lo, up = self._compute_interval(alpha_sorted, y_hat_test, epsilon, n_aug, bounds)
            result = self._construct_Gamma(lo, up, epsilon)

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

        # Test point k-NN prediction
        d = self.distance_func(self.X, x.reshape(1, -1)).ravel()
        k_test = min(self.k, n)
        agg_func = np.mean if self.aggregation == "mean" else np.median
        test_knn_idx = np.argpartition(d, k_test - 1)[:k_test]
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
    )
    _SAVE_STATE: tuple = ("X", "y", "K", "Kinv", "_n_sm_updates")
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
        d = 1 / (kappa - k.T @ Kinv @ k)
        return np.block([[Kinv + d * Kinv @ k @ k.T @ Kinv, -d * Kinv @ k], [-d * k.T @ Kinv, d]])

    @staticmethod
    def _update_K(K, k, kappa):
        return np.block([[K, k], [k.T, kappa]])

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a single example
        """
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
        ConformalPredictionInterval
        | MultiLevelPredictionInterval
        | tuple[ConformalPredictionInterval | MultiLevelPredictionInterval, dict[str, Any]]
    ):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), bounds="both")
        (-inf, inf)
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
                        predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                        result = MultiLevelPredictionInterval(predictions)
                    else:
                        result = self._construct_Gamma(-np.inf, np.inf, epsilon)
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
                        predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                        result = MultiLevelPredictionInterval(predictions)
                    else:
                        result = self._construct_Gamma(-np.inf, np.inf, epsilon)
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
                        predictions[eps] = self._construct_Gamma(lo, up, eps)
                    result = MultiLevelPredictionInterval(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
                    result = self._construct_Gamma(lower, upper, epsilon)
            elif bounds == "lower":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        lo = self._get_lower(l_dic=l_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma(lo, np.inf, eps)
                    result = MultiLevelPredictionInterval(predictions)
                else:
                    lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma(lower, np.inf, epsilon)
            elif bounds == "upper":
                if hasattr(epsilon, "__iter__"):
                    predictions = {}
                    for eps in epsilon:
                        up = self._get_upper(u_dic=u_dic, epsilon=eps, n=n)
                        predictions[eps] = self._construct_Gamma(-np.inf, up, eps)
                    result = MultiLevelPredictionInterval(predictions)
                else:
                    upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
                    result = self._construct_Gamma(-np.inf, upper, epsilon)
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
                predictions = {eps: self._construct_Gamma(-np.inf, np.inf, eps) for eps in epsilon}
                result = MultiLevelPredictionInterval(predictions)
            else:
                result = self._construct_Gamma(-np.inf, np.inf, epsilon)

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

    # Precompute X^T X diagonal and X^T y
    # For coordinate descent: update_j = X_j^T (y - X beta + X_j beta_j)
    # We use the "naive" update which is simple and correct.
    # Column norms squared (for normalization)
    col_norms_sq = np.sum(X**2, axis=0)  # (p,)

    for _iteration in range(max_iter):
        beta_old = beta.copy()
        residual = y - X @ beta

        for j in range(p):
            if col_norms_sq[j] == 0:
                continue
            # Partial residual including j-th component
            residual += X[:, j] * beta[j]
            # Unconstrained update
            rho_j = X[:, j] @ residual
            # Soft threshold
            beta[j] = _soft_threshold(rho_j, lam) / (col_norms_sq[j] + rho)
            # Update residual
            residual -= X[:, j] * beta[j]

        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
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
        """Tune lambda via K-fold cross-validation."""
        n = self.X.shape[0]
        indices = np.arange(n)
        self.rnd_gen.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        # Lambda grid: geometric sequence
        lam_max = np.max(np.abs(self.X.T @ self.y)) / n
        lam_grid = np.geomspace(lam_max, lam_max * 1e-3, num=50)

        best_lam = self.lam
        best_mse = np.inf

        for lam in lam_grid:
            mse = 0.0
            for k in range(self.n_folds):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != k])
                X_tr, y_tr = self.X[train_idx], self.y[train_idx]
                X_val, y_val = self.X[val_idx], self.y[val_idx]
                beta_k = _solve_lasso(X_tr, y_tr, lam, rho=self.rho)
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
        ConformalPredictionInterval
        | MultiLevelPredictionInterval
        | tuple[ConformalPredictionInterval | MultiLevelPredictionInterval, dict[str, Any]]
    ):
        """
        Compute the conformal prediction set at x using the homotopy algorithm.

        If epsilon is a list/array, returns a MultiLevelPredictionInterval.
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Handle multi-level epsilon
        if hasattr(epsilon, "__iter__"):
            predictions = {}
            for eps in epsilon:
                result = self.predict(x, epsilon=eps, return_update=False)
                predictions[eps] = result
            result = MultiLevelPredictionInterval(predictions)
            if return_update:
                return result, {"beta": None}
            return result

        x = np.atleast_1d(x).ravel()

        if self.X is None or self.X.shape[0] < 2:
            result = self._construct_Gamma(-np.inf, np.inf, epsilon)
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
            result = self._construct_Gamma(np.nan, np.nan, epsilon)
        elif len(merged) == 1:
            result = self._construct_Gamma(merged[0][0], merged[0][1], epsilon)
        else:
            # Return the smallest enclosing interval (conservative)
            result = self._construct_Gamma(merged[0][0], merged[-1][1], epsilon)

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

        # Rank of |r_{n+1}| (how many are <= it, in increasing order)
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
                Sigma_J_inv = np.linalg.inv(Sigma_J)
            except np.linalg.LinAlgError:
                # Singular — cannot continue homotopy
                break

            Sigma_J_inv_x = Sigma_J_inv @ x_J

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
            # Primal: beta_j(t_k) + eta_j(k) * dt = 0 for j in J_k
            beta_J = beta_k[J_k]
            dt_primal = np.full(len(J_k), np.inf)
            for idx in range(len(J_k)):
                if abs(eta_k[idx]) > 1e-15:
                    dt = -beta_J[idx] / eta_k[idx]
                    if dt > 1e-12:
                        dt_primal[idx] = dt

            # Dual: |v_j(t_k) + gamma_j * dt| = lambda for j in J_c
            dt_dual = np.full(len(J_c_k), np.inf)
            for idx in range(len(J_c_k)):
                g = gamma_k[idx]
                v_j = v_inactive[idx]
                if abs(g) > 1e-15:
                    # v_j + g*dt = +lambda or -lambda
                    dt1 = (lam - v_j) / g
                    dt2 = (-lam - v_j) / g
                    candidates = []
                    if dt1 > 1e-12:
                        candidates.append(dt1)
                    if dt2 > 1e-12:
                        candidates.append(dt2)
                    if candidates:
                        dt_dual[idx] = min(candidates)

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
        |r_{n+1}(t)| has rank <= threshold among all |r_i(t)|.

        Residuals are linear in delta_t (local parameter within the piece):
            r_i(delta) = r_train[i] + slopes_train[i] * delta   for i=0..n-1
            r_{n+1}(delta) = r_test + slope_test * delta

        Returns intervals in GLOBAL t-space (t_accumulated + sign*delta mapped to t).
        """
        # Find all crossing points using the JIT-compiled function
        raw_crossings = _compute_crossings(r_train, slopes_train, r_test, slope_test, n, dt_k)

        # Build sorted unique crossings with endpoints
        crossings = [0.0]
        for c in raw_crossings:
            crossings.append(c)
        crossings.append(dt_k)
        crossings = sorted(set(crossings))

        # For each sub-interval, check rank at midpoint
        result_intervals = []
        for idx in range(len(crossings) - 1):
            d_start = crossings[idx]
            d_end = crossings[idx + 1]
            if d_end - d_start < 1e-14:
                continue
            d_mid = (d_start + d_end) / 2

            # Compute residuals at midpoint
            r_i_mid = r_train + slopes_train * d_mid
            r_test_mid = r_test + slope_test * d_mid

            abs_r_i = np.abs(r_i_mid)
            abs_r_test = np.abs(r_test_mid)

            # Rank: number of |r_j| <= |r_{n+1}| (including n+1 itself)
            rank = np.sum(abs_r_i <= abs_r_test) + 1  # +1 for itself

            if rank <= threshold:
                # This sub-interval is in the prediction set
                # Map to global t-space
                t_start = sign * (t_accumulated + d_start)
                t_end = sign * (t_accumulated + d_end)
                if t_start > t_end:
                    t_start, t_end = t_end, t_start
                result_intervals.append((t_start, t_end))

        return result_intervals

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
