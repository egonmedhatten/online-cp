r"""Conformal predictive systems (CPS) and conformal predictive distributions.

A *conformal predictive system* extends conformal regression from prediction
*intervals* to a full predictive *distribution*. Instead of a single coverage
level, a CPS outputs, for each test object $x$, a *conformal predictive
distribution* (CPD): a function $\Pi(y)$ that behaves like a cumulative
distribution function over the unknown label and from which prediction intervals
at **any** significance level can be read off as quantiles.

Because the rank of the test label is uniform under exchangeability, the CPD is
*calibrated* in the strong sense that $\Pi(y_n)$ is (asymptotically) uniform on
$[0, 1]$ ([ALRW2 Ch.7], Vovk et al. 2019). The CPD is reported as a pair of
lower/upper step functions $(\Pi_0, \Pi_1)$ bracketing the true value at the
jumps; a smoothing variable $\tau \in [0, 1]$ interpolates between them to give
an exactly calibrated randomised value $(1 - \tau)\Pi_0 + \tau\Pi_1$.

This module provides CPS based on ridge regression, kernel ridge regression,
nearest neighbours, and the label-only Dempster–Hill construction.

References
----------
[ALRW2] Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*,
2nd ed., Springer, 2022.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize
from scipy.spatial.distance import cdist, pdist, squareform


# Lazy imports to avoid circular dependencies
def _get_mondrian_tree():
    """Lazy import of MondrianTree."""
    from online_cp.mondrian.tree import MondrianTree
    return MondrianTree

def _get_find_leaf():
    """Lazy import of _find_leaf."""
    from online_cp.mondrian.tree import _find_leaf
    return _find_leaf

try:
    from ._serialization import SerializableMixin
except ImportError:
    from _serialization import SerializableMixin

__all__ = [
    "RidgePredictionMachine",
    "KernelRidgePredictionMachine",
    "NearestNeighboursPredictionMachine",
    "DempsterHillConformalPredictiveSystem",
    "MondrianTreePredictionMachine",
]


default_epsilon = 0.1


def get_ContinuousPredictionSet():
    """Return the :class:`ContinuousPredictionSet` class (lazy import).

    .. deprecated::
        Use ``online_cp.prediction_set.ContinuousPredictionSet`` directly.
    """
    from .prediction_set import ContinuousPredictionSet

    return ContinuousPredictionSet


class ConformalPredictiveSystem(SerializableMixin):
    """Base class for conformal predictive systems."""

    _SAVE_PARAMS: tuple = ("epsilon",)
    _SAVE_STATE: tuple = ()

    def __init__(self, epsilon: float = default_epsilon) -> None:
        self.epsilon = epsilon

    def learn_many(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        for x1, y1 in zip(X, y):
            self.learn_one(x1, y1)

    def predict(self, x, **kwargs):
        """Produce a conformal predictive distribution for test object x.

        Alias for :meth:`predict_cpd`. Provides a uniform interface
        consistent with classifiers and regressors.

        Parameters
        ----------
        x : array-like, shape (d,)
            Test object.
        **kwargs
            Passed through to ``predict_cpd``.

        Returns
        -------
        ConformalPredictiveDistributionFunction
            The conformal predictive distribution for x.
        """
        return self.predict_cpd(x, **kwargs)


class RidgePredictionMachine(ConformalPredictiveSystem):
    """Conformal predictive system based on ridge regression ([ALRW2 §7.1]).

    The Ridge Prediction Machine (RPM) is the predictive-distribution analogue
    of :class:`~online_cp.regressors.ConformalRidgeRegressor`: it uses the
    studentised ridge residual as its conformity measure and, because that score
    is monotone in the hypothesised label, inverts it in closed form to produce
    a full conformal predictive distribution rather than a single interval.

    Parameters
    ----------
    a : float, optional
        Ridge regularisation parameter (default 0).
    warnings : bool, optional
        Whether to raise warnings on rank-deficient matrices (default True).
    autotune : bool, optional
        Whether to automatically tune the ridge parameter via GCV (default False).
    verbose : int, optional
        Verbosity level (default 0).
    epsilon : float, optional
        Default significance level (default 0.1).
    """

    _SAVE_PARAMS: tuple = ("a", "warnings", "autotune", "verbose", "epsilon")
    _SAVE_STATE: tuple = ("X", "y", "p", "Id", "XTXinv")

    def __init__(self, a=0, warnings=True, autotune=False, verbose=0, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.a = a
        self.X = None
        self.y = None
        self.p = None
        self.Id = None
        self.XTXinv = None

        # Should we raise warnings
        self.warnings = warnings
        # Do we autotune ridge parameter on warning
        self.autotune = autotune

        self.verbose = verbose

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.Id = np.identity(self.p)
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a single example. If we have already computed X and XTXinv, use them for update. Then the last row of X is the object with label y.
        >>> cps = RidgePredictionMachine()
        >>> cps.learn_one(np.array([1, 0]), 1)
        >>> cps.X
        array([[1, 0]])
        >>> cps.y
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
                    self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
                else:
                    # Update (X^T X + aI)^{-1} via Sherman-Morrison
                    self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

                    # Check the rank
                    if self.warnings:
                        self.check_matrix_rank(self.XTXinv)

        else:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1, -1)
                self.p = self.X.shape[1]
                self.Id = np.identity(self.p)
            elif self.X.shape[0] == 1:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
            else:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                # Update (X^T X + aI)^{-1} via Sherman-Morrison
                self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

                # Check the rank
                if self.warnings:
                    self.check_matrix_rank(self.XTXinv)

    def change_ridge_parameter(self, a):
        """
        Change the ridge parameter
        >>> cps = RidgePredictionMachine()
        >>> cps.learn_one(np.array([1, 0]), 1)
        >>> cps.change_ridge_parameter(1)
        >>> cps.a
        1
        """
        self.a = a
        if self.X is not None:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)

    def check_matrix_rank(self, M):
        """
        Check if a matrix has full rank <==> is invertible
        Returns False if matrix is rank deficient
        NOTE In numerical linear algebra it is a bit more subtle. The condition number can tell us more.

        >>> cps = RidgePredictionMachine(warnings=False)
        >>> cps.check_matrix_rank(np.array([[1, 0], [1, 0]]))
        False
        >>> cps.check_matrix_rank(np.array([[1, 0], [0, 1]]))
        True
        """
        if np.linalg.matrix_rank(M) < M.shape[0]:
            if self.warnings:
                warnings.warn(
                    f"The matrix X is rank deficient. Condition number: {np.linalg.cond(M)}. Consider changing the ridge parameter",
                    stacklevel=2,
                )
            return False
        else:
            return True

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
                max_diag_H = np.max(np.diag(A))  # Maximum diagonal element of the hat matrix
                if max_diag_H > 1:
                    return np.inf
                return (1 / n) * np.linalg.norm((In - A) @ self.y) ** 2 / ((1 / n) * np.trace(In - A)) ** 2
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return np.inf

        # Initial guess
        if a0 is None:
            a0 = 1e-6  # Small perturbation to avoid numerical issues

        # Bounds to ensure a >= 0
        res = minimize(
            GCV, x0=a0, bounds=Bounds(lb=1e-6, keep_feasible=True)
        )  # May be relevant to pass some arguments here, or even use another minimizer.
        a = res.x[0]

        if self.verbose > 0:
            print(f"New ridge parameter: {a}")
        self.change_ridge_parameter(a)

    def predict_cpd(self, x, return_update=False):
        def build_precomputed(X, XTXinv):
            computed = {
                "X": X,  # The updated matrix of objects
                "XTXinv": XTXinv,  # The updated inverse
            }
            return computed

        # Add row to X matrix
        X = np.append(self.X, x.reshape(1, -1), axis=0)
        n = X.shape[0]
        y = self.y

        # Update XTX_inv via Sherman-Morrison formula
        XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

        # Efficient computation avoiding full O(n²d) hat matrix.
        # Only compute the diagonal, last row, and H[:-1,:-1]@y in O(nd²).
        XTXinv_x = XTXinv @ x  # (d,)   — O(d²)
        h = np.sum((X @ XTXinv) * X, axis=1)  # diag(H) — O(nd²)
        h_last_row = X[:-1] @ XTXinv_x  # H[-1, :-1] — O(nd)
        Hy = X[:-1] @ (XTXinv @ (X[:-1].T @ y))  # H[:-1,:-1] @ y — O(nd)

        sqrt_one_minus_h = np.sqrt(1 - h[:-1])
        A = np.dot(h_last_row, y) / np.sqrt(1 - h[-1]) + (y - Hy) / sqrt_one_minus_h
        B = np.sqrt(1 - h[-1]) * np.ones(n - 1) + h_last_row / sqrt_one_minus_h
        C = np.zeros(n + 1)
        C[1:-1] = A / B
        C[0] = -np.inf
        C[-1] = np.inf
        C.sort()

        cpd = RidgePredictiveDistributionFunction(C=C, epsilon=self.epsilon)

        if return_update:
            return cpd, build_precomputed(X, XTXinv)
        else:
            return cpd


class KernelRidgePredictionMachine(ConformalPredictiveSystem):
    """
    This conformal predictive system uses the "studentised residuals" as conformity measure.
    Algorithm 7.3 in Algorithmic Learning in a Random World (2nd edition).
    """

    _SAVE_PARAMS: tuple = ("kernel", "a", "autotune", "verbose", "epsilon")
    _SAVE_STATE: tuple = ("X", "y", "K", "Kinv", "h_diag", "Hy")
    _SAVE_CALLABLES: tuple = ("kernel",)

    def __init__(self, kernel, a=0, autotune=False, verbose=0, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)

        self.kernel = kernel

        self.a = a
        self.X = None
        self.y = None

        # Do we autotune ridge parameter on warning
        self.autotune = autotune

        self.verbose = verbose

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        self.X = X
        self.y = y
        self.K = self.kernel(self.X)
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            Id = np.identity(self.X.shape[0])
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
            H = self.K @ self.Kinv
            self.h_diag = H.diagonal().copy()
            self.Hy = H @ y

    def _tune_ridge_parameter(self, a0=None):
        """
        Tune ridge parameter with Generalized Cross Validation (GCV) in the kernel space.
        """
        n = self.K.shape[0]
        In = np.identity(n)

        def GCV(a):
            try:
                A = self.K @ np.linalg.inv(self.K + a * In)
                max_diag_H = np.max(np.diag(A))  # Maximum diagonal element of the hat matrix
                if max_diag_H > 1:
                    return np.inf
                return (1 / n) * np.linalg.norm((In - A) @ self.y) ** 2 / ((1 / n) * np.trace(In - A)) ** 2
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return np.inf

        # Initial guess
        if a0 is None:
            a0 = 1e-6  # Small perturbation to avoid numerical issues

        # Bounds to ensure a >= 0
        res = minimize(GCV, x0=a0, bounds=Bounds(lb=1e-6, keep_feasible=True))
        a = res.x[0]

        if self.verbose > 0:
            print(f"New ridge parameter: {a}")
        self.change_ridge_parameter(a)

    def change_ridge_parameter(self, a):
        """
        Change the ridge parameter and recompute cached intermediates.
        """
        self.a = a
        if self.X is not None:
            Id = np.identity(self.X.shape[0])

            self.K = self.kernel(self.X)
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
            H = self.K @ self.Kinv
            self.h_diag = H.diagonal().copy()
            self.Hy = H @ self.y

    def _update_Kinv(self, Kinv, k, d):
        return np.block([[Kinv + d * Kinv @ k @ k.T @ Kinv, -d * Kinv @ k], [-d * k.T @ Kinv, d]])

    @staticmethod
    def _update_K(K, k, kappa):
        return np.block([[K, k], [k.T, kappa]])

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a single example
        """
        x = np.atleast_2d(x)
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)

        if precomputed is not None:
            # Incremental update of h_diag and Hy from precomputed intermediates
            v = precomputed["v"]
            d_val = precomputed["d"]
            k = precomputed["k"]
            kappa = precomputed["kappa"]
            a_d = self.a * d_val

            # h_diag_new[:-1] = h_diag_old - a_d * v^2
            # h_diag_new[-1] = 1 - a_d  (= d*kappa - d*k^T v = d*(kappa - k^T Kinv k))
            v_flat = v.ravel()
            new_h_last = (d_val * kappa - d_val * (k.T @ v)).item()
            self.h_diag = np.append(self.h_diag - a_d * v_flat**2, new_h_last)

            # Hy_new[:-1] = Hy_old - a_d * v * (v^T @ y_old) + a_d * v * y_new
            #             = Hy_old + a_d * v * (y_new - v^T @ y_old)
            # Hy_new[-1]  = a_d * (v^T @ y_old) + new_h_last * y_new
            # But y_new is the label we just appended: self.y[-1] = y
            y_old = self.y[:-1]
            vTy_old = float(v_flat @ y_old)
            self.Hy = np.append(self.Hy + a_d * v_flat * (y - vTy_old), a_d * vTy_old + new_h_last * y)

            self.K = self._update_K(self.K, k, kappa)
            self.Kinv = self._update_Kinv(self.Kinv, k, d_val)
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)
        else:
            if self.X is None:
                self.X = x.reshape(1, -1)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
                H = self.K @ self.Kinv
                self.h_diag = H.diagonal().copy()
                self.Hy = H @ self.y
            else:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                d_val = (1 / (kappa + self.a - k.T @ self.Kinv @ k)).item()
                a_d = self.a * d_val

                # Compute v = Kinv @ k
                v = self.Kinv @ k  # (n, 1)
                v_flat = v.ravel()

                # Incremental update of h_diag and Hy
                new_h_last = (d_val * kappa - d_val * (k.T @ v)).item()
                self.h_diag = np.append(self.h_diag - a_d * v_flat**2, new_h_last)

                y_old = self.y[:-1]
                vTy_old = float(v_flat @ y_old)
                self.Hy = np.append(self.Hy + a_d * v_flat * (y - vTy_old), a_d * vTy_old + new_h_last * y)

                self.K = self._update_K(self.K, k, kappa)
                self.Kinv = self._update_Kinv(self.Kinv, k, d_val)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    def predict_cpd(self, x, return_update=False):

        def build_precomputed(v, d_val, k, kappa):
            computed = {
                "v": v,
                "d": d_val,
                "k": k,
                "kappa": kappa,
            }
            return computed

        x = np.atleast_2d(x)
        # Temporarily update kernel matrix
        k = self.kernel(self.X, x).reshape(-1, 1)
        kappa = self.kernel(x, x)
        d_val = (1 / (kappa + self.a - k.T @ self.Kinv @ k)).item()
        a_d = self.a * d_val
        y = self.y

        # Compute v = Kinv @ k (the key intermediate)
        v = self.Kinv @ k  # (n, 1)
        v_flat = v.ravel()

        # Efficient O(n) computation — avoid forming full (n+1)×(n+1) hat matrix.
        # H_new diagonal: h_diag_new[:-1] = h_diag_old - a_d * v^2, h_new[-1] = d*kappa - d*k^T*v
        h_train = self.h_diag - a_d * v_flat**2
        h_last = (d_val * kappa - d_val * (k.T @ v)).item()

        # H_new last row (= last col by symmetry): H[-1, :-1] = a_d * v^T
        h_last_row = a_d * v_flat

        # H_new[:-1,:-1] @ y = Hy_old - a_d * v * (v^T @ y)
        Hy_train = self.Hy - a_d * v_flat * float(v_flat @ y)

        n = len(y) + 1  # augmented size

        sqrt_one_minus_h = np.sqrt(1 - h_train)
        A = np.dot(h_last_row, y) / np.sqrt(1 - h_last) + (y - Hy_train) / sqrt_one_minus_h
        B = np.sqrt(1 - h_last) * np.ones(n - 1) + h_last_row / sqrt_one_minus_h

        C = np.zeros(n + 1)
        C[1:-1] = A / B
        C[0] = -np.inf
        C[-1] = np.inf
        assert not np.isnan(C).any(), "C contains NaN values"
        C.sort()

        cpd = RidgePredictiveDistributionFunction(C=C, epsilon=self.epsilon)

        if return_update:
            return cpd, build_precomputed(v, d_val, k, kappa)
        else:
            return cpd


class NearestNeighboursPredictionMachine(ConformalPredictiveSystem):
    _SAVE_PARAMS: tuple = ("k", "distance", "distance_func", "epsilon")
    _SAVE_STATE: tuple = ("X", "y", "D")
    _SAVE_CALLABLES: tuple = ("distance_func",)
    _PARAM_MAP: dict = {"distance_func": "_distance_func_arg"}

    def __init__(
        self,
        k,
        distance="euclidean",
        distance_func=None,
        epsilon=default_epsilon,
    ):
        super().__init__(epsilon=epsilon)

        self.k = k

        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = "custom"
        self._distance_func_arg = distance_func

        self.X = None
        self.y = None
        self.D = None

    def _standard_distance_func(self, X, y=None):
        """
        By default we use scipy to compute distances
        """
        X = np.atleast_2d(X)
        if y is None:
            dists = squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            dists = cdist(X, y, metric=self.distance)
        return dists

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        """
        >>> cps = NearestNeighboursPredictionMachine(k=3)
        >>> X = np.array([[1], [2]])
        >>> y = np.array([1, 2])
        >>> cps.learn_initial_training_set(X, y)
        >>> cps.X
        array([[1],
               [2]])
        >>> cps.y
        array([1, 2])
        >>> cps.D
        array([[0., 1.],
               [1., 0.]])
        """
        self.X = X
        self.D = self.distance_func(X)
        self.y = y

    @staticmethod
    def update_distance_matrix(D, d):
        return np.block([[D, d], [d.T, np.array([0])]])

    def learn_one(self, x: NDArray[np.floating[Any]], y: float, precomputed: dict[str, Any] | None = None) -> None:
        """
        Learn a single example.

        precomputed is a dictionary
        {
            'X': X,
            'D': D,
        }
        >>> cps = NearestNeighboursPredictionMachine(k=3)
        >>> X = np.array([[1], [2]])
        >>> y = np.array([1, 2])
        >>> cps.learn_initial_training_set(X, y)
        >>> cps.learn_one(np.array([3]), 3)
        >>> cps.y
        array([1, 2, 3])
        >>> cps.X
        array([[1],
               [2],
               [3]])
        >>> cps.D
        array([[0., 1., 2.],
               [1., 0., 1.],
               [2., 1., 0.]])
        """
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)

        if precomputed is None:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1, -1)
                self.D = self.distance_func(self.X)
            else:
                d = self.distance_func(self.X, x)
                self.D = self.update_distance_matrix(self.D, d)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
        else:
            self.X = precomputed["X"]
            self.D = precomputed["D"]

    def predict_cpd(self, x, return_update=False):
        """
        >>> import numpy as np
        >>> rnd_gen = np.random.default_rng(2024)
        >>> X = rnd_gen.normal(loc=0, scale=1, size=(100, 4))
        >>> beta = np.array([2, 1, 0, 0])
        >>> Y = X @ beta + rnd_gen.normal(loc=0, scale=1, size=100)
        >>> cps = NearestNeighboursPredictionMachine(k=3)
        >>> cps.learn_initial_training_set(X, Y)
        >>> x = rnd_gen.normal(loc=0, scale=1, size=(1, 4))
        >>> cpd = cps.predict_cpd(x)
        >>> cpd.L
        array([0.        , 0.        , 0.18811881, 0.53465347, 0.76237624])
        >>> cpd.U
        array([0.17821782, 0.18811881, 0.54455446, 0.76237624, 1.        ])
        """
        # Temporarily update the distance matrix
        if self.X.shape[0] <= self.k:
            raise ValueError("Training set is too small for k-NN prediction")
        d = self.distance_func(self.X, x)
        D = self.update_distance_matrix(self.D, d)
        y = np.append(self.y, -np.inf)  # Initialise label as -inf

        # Find all neighbours and semi-neighbours
        # Use argpartition for O(n) selection of k+1 smallest, then sort them
        # for deterministic tie-breaking consistent with full argsort.
        top_k1 = np.argpartition(D, self.k + 1, axis=0)[: self.k + 1]
        for col in range(D.shape[1]):
            idx = top_k1[:, col]
            order = np.argsort(D[idx, col])
            top_k1[:, col] = idx[order]
        k_nearest = top_k1[1:]  # skip self (distance=0, always first after sort)

        n = self.X.shape[0]

        full_neighbours = set()
        single_neighbours = set()
        semi_neighbours = set()

        k_nearest_of_n = set(k_nearest.T[-1])

        for i, col in enumerate(k_nearest.T):
            i_is_neighbour_of_n = i in k_nearest_of_n
            n_is_neighbour_of_i = n in col
            if i_is_neighbour_of_n and n_is_neighbour_of_i:
                full_neighbours.add(i)
            elif i_is_neighbour_of_n:
                single_neighbours.add(i)
            elif n_is_neighbour_of_i:
                semi_neighbours.add(i)

        neighbours = full_neighbours | single_neighbours
        full_or_semi = full_neighbours | semi_neighbours
        idx_all_neighbours_and_semi_neighbours = np.array(sorted(full_neighbours | single_neighbours | semi_neighbours))

        # Line 1
        Kprime = len(idx_all_neighbours_and_semi_neighbours)
        # Line 2 and 3
        Y = np.zeros(shape=Kprime + 2)
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = y[idx_all_neighbours_and_semi_neighbours]
        idx_mem = {i: idx_all_neighbours_and_semi_neighbours[i - 1] for i in range(1, Kprime + 1)}
        sorted_indices = np.argsort(Y)[1:-1]
        Y.sort()

        # Line 4: conformity scores and histogram
        Alpha = np.array([(y[k_nearest.T[i]] <= y_i).sum() for i, y_i in enumerate(y)])
        N = np.array([(Alpha == k).sum() for k in range(self.k + 1)])

        # Line 5
        L = -np.inf * np.ones(Kprime + 1)
        U = -np.inf * np.ones(Kprime + 1)
        L[0] = 0
        U[0] = N[0] / (n + 1)

        # Line 6
        for k in range(1, Kprime + 1):
            idx = idx_mem[sorted_indices[k - 1]]
            if idx in neighbours:
                N[Alpha[-1]] -= 1
                Alpha[-1] += 1
                N[Alpha[-1]] += 1
            if idx in full_or_semi:
                N[Alpha[idx]] -= 1
                Alpha[idx] -= 1
                N[Alpha[idx]] += 1
            L[k] = N[: Alpha[-1]].sum() / (n + 1) if Alpha[-1] != 0 else 0
            U[k] = N[: Alpha[-1] + 1].sum() / (n + 1) if Alpha[-1] != 0 else N[0] / (n + 1)

        # Line 12
        cpd = NearestNeighboursPredictiveDistributionFunction(L, U, Y, epsilon=self.epsilon)

        if return_update:
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            return cpd, {"X": X, "D": D}
        else:
            return cpd


class DempsterHillConformalPredictiveSystem(ConformalPredictiveSystem):
    r"""Label-only conformal predictive system (Dempster–Hill).

    The simplest CPS: it ignores the objects $x$ entirely and builds the
    predictive distribution from the labels alone, placing the test label among
    the sorted training labels. This is the conformalised version of Hill's
    $A_{(n)}$ assumption / Dempster's direct probabilities, and serves as a
    distribution-free baseline (the predictive distribution of an exchangeable
    sequence with no covariate information). See [ALRW2 §7.5].
    """

    _SAVE_PARAMS: tuple = ("epsilon",)
    _SAVE_STATE: tuple = ("y",)

    def __init__(self, epsilon=default_epsilon):
        """Create a Dempster–Hill CPS.

        Only the labels are used, so objects passed to the ``learn_*`` methods
        are ignored.

        Parameters
        ----------
        epsilon : float, default 0.1
            Default significance level.
        """
        super().__init__(epsilon=epsilon)
        self.y = None

    def learn_initial_training_set(self, y):
        self.y = y

    def learn_one(self, y):
        self.y = np.append(self.y, y)

    def learn_many(self, y):
        self.y = np.append(self.y, y)

    def predict_cpd(self, x=None, return_update=False):
        """Compute the conformal predictive distribution (labels only).

        Parameters
        ----------
        x : array-like, optional
            Accepted for interface uniformity with other CPS classes; ignored
            because the Dempster–Hill model uses only the labels.
        return_update : bool, default False
            If True, return ``(cpd, {"labels_only": y})``.

        Returns
        -------
        DempsterHillConformalPredictiveDistribution
        """
        Y = np.zeros(shape=self.y.shape[0] + 2)
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = self.y
        Y.sort()
        cpd = DempsterHillConformalPredictiveDistribution(Y, epsilon=self.epsilon)
        if return_update:
            return cpd, {"labels_only": self.y.copy()}
        return cpd


class ConformalPredictiveDistributionFunction:
    """Base class for conformal predictive distributions.

    A conformal predictive distribution (CPD) contains all the information
    needed to form prediction sets at any significance level. Supports
    quantile computation, prediction set construction, and conformal
    expectation.
    """

    def __init__(self, epsilon=default_epsilon):
        self.epsilon = epsilon

    def __call__(self, y, tau=None):
        r"""Evaluate the conformal predictive distribution at ``y``.

        The CPD is a step function with jumps at the observed labels, so its
        value at ``y`` is reported as a lower/upper pair $(\Pi_0, \Pi_1)$ that
        brackets the jump. Supplying a smoothing variable $\tau \in [0, 1]$
        returns the interpolated, exactly calibrated value
        $(1 - \tau)\Pi_0 + \tau\Pi_1$.

        Parameters
        ----------
        y : float
            Point at which to evaluate the predictive distribution.
        tau : float or None
            Smoothing variable in $[0, 1]$. If None, the $(\Pi_0, \Pi_1)$ pair
            is returned.

        Returns
        -------
        tuple of float, or float
            $(\Pi_0, \Pi_1)$ if ``tau`` is None, else the interpolated value.
        """
        Pi0, Pi1 = self._cdf_bounds(y)
        if tau is None:
            return Pi0, Pi1
        return (1 - tau) * Pi0 + tau * Pi1

    def _cdf_bounds(self, y):
        """Compute lower and upper CDF bounds at y.

        Default implementation for CPDs with sorted Y array where
        len(Y) == len(L) == len(U), Y[0]=-inf, Y[-1]=+inf.
        Override for different array conventions.
        """
        if y == self.Y[0]:
            return 0.0, 0.0
        if y == self.Y[-1]:
            return 1.0, 1.0
        Y_trimmed = self.Y[:-1]
        n = len(Y_trimmed)
        left = np.searchsorted(Y_trimmed, y, side="left")
        right = np.searchsorted(Y_trimmed, y, side="right")
        if left < right:
            # y matches one or more breakpoints
            return (left - 1) / n, right / n
        else:
            # y is between breakpoints
            i = left - 1
            return i / n, (i + 1) / n

    def quantile(self, p, tau=None):
        """Compute the p-quantile of the CPD.

        If tau is None, returns (quantile at tau=0, quantile at tau=1).
        """
        if tau is not None:
            return self._compute_quantile(p, tau)
        return self._compute_quantile(p, 0), self._compute_quantile(p, 1)

    def _compute_quantile(self, p, tau):
        raise NotImplementedError("Subclass must implement _compute_quantile")

    def predict_set(self, tau, epsilon=None, bounds="both", minimise_width=False):
        """
        The convex hull of the epsilon/2 and 1-epsilon/2 quantiles make up
        the prediction set Gamma(epsilon)

        If epsilon is a list/array, returns a MultiLevelPredictionSet.
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Handle multi-level epsilon
        if hasattr(epsilon, "__iter__"):
            from .prediction_set import MultiLevelPredictionSet

            predictions = {}
            for eps in epsilon:
                predictions[eps] = self.predict_set(tau, epsilon=eps, bounds=bounds, minimise_width=minimise_width)
            return MultiLevelPredictionSet(predictions)

        if minimise_width:
            if bounds != "both":
                raise ValueError('bounds must be "both" when minimise_width=True')

            # Find narrowest [a, b] = [quantile(l), quantile(l+1-eps)]
            # over l. The width W(l) is piecewise-constant, changing only
            # when l or l+target crosses a CDF level. So candidate l's are
            # {all CDF levels} ∪ {all CDF levels - target}.
            target_coverage = 1 - epsilon

            # Collect all distinct CDF levels the step function can take
            if len(self.L) == len(self.Y):
                # Ridge/DH: at-breakpoint levels from stored L/U;
                # between-breakpoint levels = (j+tau)/n from _cdf_bounds
                n_trim = len(self.Y) - 1
                at_levels = (1 - tau) * self.L + tau * self.U
                btwn_levels = (np.arange(n_trim) + tau) / n_trim
                all_cdf_levels = np.unique(np.concatenate([at_levels, btwn_levels]))
            else:
                # NN: between-levels from L/U, at-levels from L[:-1]/U[1:]
                btwn_levels = (1 - tau) * self.L + tau * self.U
                at_levels = (1 - tau) * self.L[:-1] + tau * self.U[1:]
                all_cdf_levels = np.unique(np.concatenate([btwn_levels, at_levels]))

            # Candidates: CDF levels AND (CDF levels - target)
            shifted = all_cdf_levels - target_coverage
            candidate_levels = np.unique(np.concatenate([all_cdf_levels, shifted]))
            candidate_levels = candidate_levels[(candidate_levels >= 0) & (candidate_levels + target_coverage <= 1.0)]

            best_width = np.inf
            best_lower = -np.inf
            best_upper = np.inf

            for l in candidate_levels:
                lower_q = self.quantile(l, tau)
                upper_q = self.quantile(l + target_coverage, tau)
                w = upper_q - lower_q
                if w < best_width:
                    best_width = w
                    best_lower = lower_q
                    best_upper = upper_q

            lower = best_lower
            upper = best_upper

        else:
            q1 = epsilon / 2
            q2 = 1 - epsilon / 2
            if bounds == "both":
                lower = self.quantile(q1, tau)
                upper = self.quantile(q2, tau)
            elif bounds == "lower":
                lower = self.quantile(q1, tau)
                upper = np.inf
            elif bounds == "upper":
                lower = -np.inf
                upper = self.quantile(q2, tau)
            else:
                raise ValueError('bounds must be "both", "lower", or "upper"')

        CPS = get_ContinuousPredictionSet()
        return CPS([(lower, upper)], epsilon)

    def find_smallest_epsilon(self, tau, increment=0.001):
        """
        Find the smallest epsilon such that the prediction set is finite.
        Returns None if no such epsilon <= 1 exists.
        """
        epsilon = 0.0
        while epsilon <= 1.0:
            prediction_set = self.predict_set(tau=tau, epsilon=epsilon)
            if np.isfinite(prediction_set.width()):
                return epsilon
            epsilon += increment
        return None

    def median(self, tau: float = 0.5):
        """Median of the CPD: shortcut for ``quantile(0.5, tau)``.

        Parameters
        ----------
        tau : float, default 0.5
            Randomisation parameter.
        """
        return self.quantile(0.5, tau)

    # These methods relate to when the cpd is used to predict sets
    @staticmethod
    def err(Gamma, y):
        return int(y not in Gamma)

    @staticmethod
    def width(Gamma):
        return Gamma.width()

    def plot(self, tau=None):
        """Plot the CPD. Assumes self.Y and self.L/self.U have the same length."""
        if tau is None:
            fig, ax = plt.subplots()
            ax.step(self.Y, self.L, label=r"$\Pi(y, 0)$", where="pre")
            ax.step(self.Y, self.U, label=r"$\Pi(y, 1)$", where="pre")
            ax.fill_between(self.Y, self.L, self.U, step="pre", alpha=0.5, color="green")
            ax.legend()
        else:
            fig, ax = plt.subplots()
            ax.step(self.Y, (1 - tau) * self.L + tau * self.U, label=r"$\Pi(y, \tau)$", where="pre")
            ax.legend()
        ax.set_ylabel("cumulative probability")
        ax.set_xlabel(r"$y$")
        fig.tight_layout()
        plt.close(fig)  # Prevent implicit display
        return fig


class RidgePredictiveDistributionFunction(ConformalPredictiveDistributionFunction):
    def __init__(self, C, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.C = C
        self.Y = C

        # Analytical computation of L, U (O(n) instead of O(n²))
        # C is sorted with C[0]=-inf, C[-1]=+inf. Trimmed array has n = len(C)-1 elements.
        n = len(C) - 1
        # For Ridge, C values are all distinct (from A*y+B formula)
        j = np.arange(len(C))
        self.L = np.where(j == 0, 0.0, np.where(j == len(C) - 1, 1.0, (j - 1) / n))
        self.U = np.where(j == 0, 0.0, np.where(j == len(C) - 1, 1.0, (j + 1) / n))

    def _compute_quantile(self, p, tau):
        # Analytical inversion: find smallest y where Pi(y, tau) >= p
        if p <= 0:
            return -np.inf
        if p > 1:
            return np.inf
        # Pi values at C[j] are (1-tau)*L[j] + tau*U[j]
        # Pi values just above C[j] (between C[j] and C[j+1]) are (j+tau)/n
        n = len(self.C) - 1
        # Check "between" levels: (j+tau)/n for j=0..n-1
        # The smallest j where (j+tau)/n >= p is j = ceil(p*n - tau)
        j_between = max(0, int(np.ceil(p * n - tau)))
        # Check "at breakpoint" levels for j = 1..n-1.
        # For distinct C: Pi at C[j] = (j-1+2*tau)/n.
        # For tied C (duplicate training rows), the true Pi at C[j] is
        # determined by _cdf_bounds (searchsorted-based) and may differ from
        # the index-based formula.  We therefore verify the candidate with
        # _cdf_bounds before accepting it.
        # Smallest j where (j-1+2*tau)/n >= p: j = ceil(p*n - 2*tau + 1)
        j_at = max(0, int(np.ceil(p * n - 2 * tau + 1)))

        # The quantile is the smallest y: either C[j_at] or C[j_between]+eps
        # C[j_between]+eps corresponds to y just above C[j_between]
        if 1 <= j_at < len(self.C) - 1 and j_at <= j_between:
            # Validate using the tie-aware _cdf_bounds (handles duplicate C values)
            Pi_lo, Pi_hi = self._cdf_bounds(self.C[j_at])
            Pi_at = (1 - tau) * Pi_lo + tau * Pi_hi
            if Pi_at >= p - 1e-12:
                return self.C[j_at]
            # Pi at C[j_at] is insufficient due to ties; fall through
        if 0 <= j_between < len(self.C) - 1:
            # Return a value just above C[j_between]
            if j_between == 0:
                return -np.inf
            return np.nextafter(self.C[j_between], np.inf)
        else:
            return np.inf


class NearestNeighboursPredictiveDistributionFunction(ConformalPredictiveDistributionFunction):
    def __init__(self, L, U, Y, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.L = L
        self.U = U
        self.Y = Y

    def _cdf_bounds(self, y):
        """NN override: len(L) == len(Y) - 1, different indexing convention."""
        if y == self.Y[0]:
            return 0.0, 0.0
        if y == self.Y[-1]:
            return 1.0, 1.0
        Y_trimmed = self.Y[:-1]
        left = np.searchsorted(Y_trimmed, y, side="left")
        right = np.searchsorted(Y_trimmed, y, side="right")
        if left < right:
            # y matches breakpoint(s): Pi = [L[left-1], U[right-1]]
            return self.L[left - 1], self.U[right - 1]
        else:
            # y is between breakpoints: Pi = [L[k], U[k]]
            k = left - 1
            return self.L[k], self.U[k]

    def _compute_quantile(self, p, tau):
        # NN CPD: Pi(Y[k], tau) = (1-tau)*L[k-1] + tau*U[k] at breakpoint Y[k] (k=1..K')
        # Between Y[k] and Y[k+1]: Pi = (1-tau)*L[k] + tau*U[k]
        # L and U have K'+1 entries (indices 0..K')
        # Find smallest y where Pi(y, tau) >= p
        # (K' = len(self.L) - 1 entries)

        # Check "between" levels: (1-tau)*L[k] + tau*U[k] for k=0..K'
        between_levels = (1 - tau) * self.L + tau * self.U

        # Check "at breakpoint" levels: (1-tau)*L[k-1] + tau*U[k] for k=1..K'
        at_levels = (1 - tau) * self.L[:-1] + tau * self.U[1:]

        # Find first index where level >= p
        between_idx = np.where(between_levels >= p)[0]
        at_idx = np.where(at_levels >= p)[0]

        best_y = np.inf
        if len(at_idx) > 0:
            k = at_idx[0] + 1  # +1 because at_levels[i] corresponds to Y[i+1]
            best_y = self.Y[k]
        if len(between_idx) > 0:
            k = between_idx[0]
            # "between Y[k] and Y[k+1]" — infimum is Y[k] (or -inf if k=0)
            if k == 0:
                candidate = -np.inf
            else:
                candidate = np.nextafter(self.Y[k], np.inf)
            if candidate < best_y:
                best_y = candidate
        return best_y

    def plot(self, tau=None):
        # Override: NN has len(L) = len(Y) - 1, so x-axis uses Y[1:]
        if tau is None:
            fig, ax = plt.subplots()
            ax.step(self.Y[1:], self.L, label=r"$\Pi(y, 0)$", where="pre")
            ax.step(self.Y[1:], self.U, label=r"$\Pi(y, 1)$", where="pre")
            ax.fill_between(self.Y[1:], self.L, self.U, step="pre", alpha=0.5, color="green")
            ax.legend()
        else:
            fig, ax = plt.subplots()
            ax.step(self.Y[1:], (1 - tau) * self.L + tau * self.U, label=r"$\Pi(y, \tau)$", where="pre")
            ax.legend()
        ax.set_ylabel("cumulative probability")
        ax.set_xlabel(r"$y$")
        fig.tight_layout()
        plt.close(fig)  # Prevent implicit display
        return fig


class DempsterHillConformalPredictiveDistribution(ConformalPredictiveDistributionFunction):
    def __init__(self, Y, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.Y = Y

        # Vectorized computation of L, U
        # Y is sorted with Y[0]=-inf, Y[-1]=+inf.
        n = len(Y) - 1  # size of trimmed array Y[:-1]
        self.L = np.zeros(len(Y))
        self.U = np.zeros(len(Y))
        self.L[-1] = 1.0
        self.U[-1] = 1.0
        # For interior values, handle ties using searchsorted (vectorized)
        trimmed = Y[:-1]
        interior = Y[1:-1]
        left = np.searchsorted(trimmed, interior, side="left")
        right = np.searchsorted(trimmed, interior, side="right") - 1
        self.L[1:-1] = (left - 1) / n
        self.U[1:-1] = (right + 1) / n

    def _compute_quantile(self, p, tau):
        # CDF at breakpoint Y[j]: (1-tau)*L[j] + tau*U[j]
        # CDF between Y[j] and Y[j+1]: (j + tau) / n
        n = len(self.Y) - 1  # size of trimmed Y[:-1]

        # First j where at-knot level >= p
        pi_at = (1 - tau) * self.L + tau * self.U
        at_candidates = np.where(pi_at >= p)[0]
        j_at = at_candidates[0] if len(at_candidates) > 0 else len(self.Y)

        # First j where between-level (j+tau)/n >= p
        j_between = max(0, int(np.ceil(p * n - tau)))

        if j_at <= j_between and j_at < len(self.Y):
            # At-breakpoint reached first
            if j_at == 0:
                return -np.inf
            return self.Y[j_at]
        elif j_between < len(self.Y) - 1:
            # Between-level reached first; quantile is just above Y[j_between]
            if j_between == 0:
                return -np.inf
            return np.nextafter(self.Y[j_between], np.inf)
        else:
            return np.inf


class MondrianTreePredictionMachine(ConformalPredictiveSystem):
    r"""Conformal predictive system based on Mondrian trees with signed residuals.

    This CPS uses the Mondrian tree partition and the signed residual as the
    conformity measure:

        α_i = y_i − μ_leaf(x_i)

    where μ_leaf is the mean of **all** y-values (including the candidate) in
    the leaf.  For Mondrian trees, the leaf structure depends only on the
    features X (not on y), so the test point (x, y) always falls into the same
    leaf for any candidate y.

    Mathematical insight
    --------------------
    Let A = number of training points in ``leaf_star`` and B = their y-sum.
    Under candidate y:

    * Test signed residual:  ``(A·y − B) / (A+1)``    slope A/(A+1)
    * Inside signed residual: ``((A+1)·y_j − B − y) / (A+1)``    slope ∓1/(A+1)

    The difference α_n^y − α_i^y = y − y_i is strictly increasing in y (slope 1),
    so the critical values are simply C_i = y_i (the training labels).  No
    studentisation is needed.

    This yields an O(n) conformal predictive distribution with knots at the
    sorted training labels, making it extremely efficient.

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
        "lifetime", "epsilon", "rnd_state", "verbose", "max_depth",
        "feature_weights",
    )
    _SAVE_STATE: tuple = ("X", "y")

    def __init__(
        self,
        lifetime: float | str = 1.0,
        epsilon: float = 0.1,
        rnd_state: int | None = None,
        verbose: int = 0,
        max_depth: int | None = None,
        feature_weights: str | NDArray | None = None,
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
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
        super().__init__(epsilon=epsilon)
        self.lifetime = lifetime
        self.feature_weights = feature_weights
        self.rnd_state = rnd_state
        self.verbose = verbose
        self.max_depth = max_depth
        self.X: NDArray | None = None
        self.y: NDArray | None = None
        self.rnd_gen = np.random.default_rng(rnd_state)

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

    def learn_one(self, x: NDArray, y: float, precomputed: dict | None = None) -> None:
        """Incrementally add one observation.

        Parameters
        ----------
        x : array-like, shape (d,)
        y : float
        precomputed : dict, optional
            Not used. Present for API consistency.
        """
        x = np.asarray(x, dtype=float).ravel()
        if self.X is None:
            raise ValueError("Must call learn_initial_training_set first")
        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, float(y))

    def predict_cpd(self, x, return_update=False):
        r"""Compute the conformal predictive distribution for test object x.

        Parameters
        ----------
        x : array-like, shape (d,)
        return_update : bool
            If True, return ``(cpd, precomputed)`` where ``precomputed`` is a
            dict that can be passed to ``learn_one``. For Mondrian trees this
            is currently empty but reserved for future incremental updates.

        Returns
        -------
        MondrianPredictiveDistributionFunction
            The conformal predictive distribution for x.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = self._safe_size_check(self.X)

        if n == 0:
            # Empty training set → completely non-informative CPS
            Y = np.array([-np.inf, np.inf])
            cpd = MondrianPredictiveDistributionFunction(Y, np.array([0.0, 1.0]), np.array([0.0, 1.0]), epsilon=self.epsilon)
            if return_update:
                return cpd, {}
            return cpd

        # Lazy imports
        MondrianTree = _get_mondrian_tree()
        _find_leaf = _get_find_leaf()

        # Build Mondrian tree on augmented data [X; x]
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

        # Find leaf containing x (leaf_star).  For Mondrian trees the leaf
        # depends only on X, so the same leaf is visited for every candidate y.
        # The partition is required by the theory even though the signed-
        # residual monotonicity makes the CPD depend only on the labels.
        leaf_star = _find_leaf(tree, x)
        leaf_star_train_idx = leaf_star.indices[leaf_star.indices < n]  # noqa: F841

        # Critical values C_i = y_i (the training labels).
        # The signed-residual difference α_n^y − α_i^y = y − y_i is strictly
        # increasing in y (slope 1), so the p-value jumps exactly at each y_i.
        # Consequently the leaf mean B does not appear in the final CPD.
        Y = np.zeros(n + 2)
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = self.y.copy()
        Y.sort()

        # For Mondrian signed residual CPS:
        # L[j] = (number of labels < Y[j]) / n
        # U[j] = (number of labels <= Y[j]) / n
        # For Y[0] = -inf: L[0] = 0, U[0] = 0
        # For Y[-1] = +inf: L[-1] = 1, U[-1] = 1
        L = np.zeros(len(Y))
        U = np.zeros(len(Y))
        L[-1] = 1.0
        U[-1] = 1.0
        # For each breakpoint Y[j] (j=1..n), count how many training labels
        # are strictly less than Y[j] (for L) and how many are <= Y[j] (for U).
        for j in range(1, n + 1):
            y_j = Y[j]
            L[j] = np.sum(self.y < y_j) / n
            U[j] = np.sum(self.y <= y_j) / n

        cpd = MondrianPredictiveDistributionFunction(Y, L, U, epsilon=self.epsilon)

        if return_update:
            return cpd, {}
        else:
            return cpd

    def _safe_size_check(self, X):
        """Helper to check if X is None or empty."""
        if X is None:
            return 0
        return X.shape[0]


class MondrianPredictiveDistributionFunction(ConformalPredictiveDistributionFunction):
    r"""Conformal predictive distribution for the Mondrian signed-residual CPS.

    The CPD is a step function whose knots are at the sorted training labels
    ``Y[1:-1]``.  At each knot ``Y[j]`` the lower and upper CDF bounds are

    .. math::

        L[j] = \frac{|\{i : y_i < Y[j]\}|}{n}, \qquad
        U[j] = \frac{|\{i : y_i \le Y[j]\}|}{n},

    where :math:`n` is the training-set size.  The use of strict ``<`` for
    ``L`` and ``<=`` for ``U`` follows the interval-valued CPD definition in
    [ALRW2 §7.2]: the lower bound excludes ties whereas the upper bound
    includes them, so that :math:`L(y) \le U(y)` for every :math:`y`.

    Parameters
    ----------
    Y : ndarray
        Sorted array of length ``n+2`` with ``Y[0] = -inf``, ``Y[-1] = +inf``,
        and ``Y[1:-1]`` = the sorted training labels.
    L : ndarray
        Lower CDF bounds, same length as ``Y``.
    U : ndarray
        Upper CDF bounds, same length as ``Y``.
    epsilon : float, default 0.1
        Default significance level.
    """

    def __init__(self, Y, L, U, epsilon=default_epsilon):
        """Initialise the CPD.

        Parameters
        ----------
        Y : ndarray
            Sorted knot locations with ``Y[0] = -inf``, ``Y[-1] = +inf``.
        L : ndarray
            Lower CDF bounds at each knot.
        U : ndarray
            Upper CDF bounds at each knot.
        epsilon : float, optional
            Default significance level.
        """
        super().__init__(epsilon=epsilon)
        self.Y = Y
        self.L = L
        self.U = U

    def _cdf_bounds(self, y):
        r"""Compute the lower and upper CDF bounds at ``y``.

        Uses binary search (``np.searchsorted``) on ``Y[1:-1]`` to locate
        ``y``.  When ``y`` coincides with one or more knots, the lower bound
        is ``L[left]`` (first matching knot's ``L``) and the upper bound is
        ``U[right-1]`` (last matching knot's ``U``).  When ``y`` lies strictly
        between two knots, both bounds equal ``L[i] = U[i]`` for the bracket
        index ``i``.
        """
        if y == self.Y[0]:
            return 0.0, 0.0
        if y == self.Y[-1]:
            return 1.0, 1.0
        # Search in Y[1:-1] which contains the actual data points
        left = np.searchsorted(self.Y[1:-1], y, side="left") + 1
        right = np.searchsorted(self.Y[1:-1], y, side="right") + 1
        if left < right:
            # y matches one or more breakpoints (multiple occurrences in Y[1:-1])
            # left and right are indices in Y (after +1 offset)
            # Matches span from Y[left] to Y[right-1] (in Y indexing)
            return self.L[left], self.U[right - 1]
        else:
            # y is between breakpoints
            i = left
            return self.L[i], self.U[i]

    def _compute_quantile(self, p, tau):
        """Compute the p-quantile of the CPD.

        Returns the smallest ``y`` such that :math:`\\Pi(y, \\tau) \\ge p`.

        Two types of CDF level are checked:

        * **At breakpoint** ``Y[j]`` (``j = 1,\\dots,n``): the smoothed CDF
          is :math:`(1-\\tau) L[j] + \\tau U[j]`.
        * **Between** ``Y[j]`` and ``Y[j+1]`` (``j = 0,\\dots,n``): the CDF
          is constant at :math:`(1-\\tau) L[j] + \\tau U[j]`.

        The quantile is whichever yields the smaller ``y``: the knot
        ``Y[j_{\\text{at}}]`` or the value just above ``Y[j_{\\text{between}}]``.
        Tie-breaking is consistent with :meth:`_cdf_bounds` — ties cause
        ``L`` and ``U`` to differ but the smallest feasible ``y`` is always
        returned.
        """
        n = len(self.Y) - 2  # number of training labels

        if p <= 0:
            return -np.inf
        if p > 1:
            return np.inf

        # At-breakpoint CDF values for j = 1..n.
        at_levels = (1 - tau) * self.L[1:-1] + tau * self.U[1:-1]
        at_idx = np.where(at_levels >= p)[0]
        j_at = at_idx[0] + 1 if len(at_idx) > 0 else len(self.Y)

        # Between-interval CDF values for j = 0..n.
        between_levels = (1 - tau) * self.L[: n + 1] + tau * self.U[: n + 1]
        between_idx = np.where(between_levels >= p)[0]
        j_between = between_idx[0] if len(between_idx) > 0 else n + 1

        if j_at <= j_between and j_at < len(self.Y):
            # At-breakpoint reached first (or simultaneously)
            return self.Y[j_at]
        elif j_between < len(self.Y):
            # Between-level reached first; quantile is just above Y[j_between]
            if j_between == 0:
                return -np.inf
            return np.nextafter(self.Y[j_between], np.inf)
        else:
            return np.inf

    def plot(self, tau=None, ax=None):
        """Plot the conformal predictive distribution.

        Parameters
        ----------
        tau : float, optional
            Smoothing variable in ``[0, 1]``.  If ``None``, plots both
            bounds ``L(y)`` and ``U(y)`` with a shaded region.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  If ``None``, a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if tau is None:
            ax.step(self.Y[1:-1], self.L[1:-1], label=r"$\Pi(y, 0)$", where="post")
            ax.step(self.Y[1:-1], self.U[1:-1], label=r"$\Pi(y, 1)$", where="post")
            ax.fill_between(self.Y[1:-1], self.L[1:-1], self.U[1:-1], step="post", alpha=0.5, color="green")
            ax.legend()
        else:
            ax.step(self.Y[1:-1], (1 - tau) * self.L[1:-1] + tau * self.U[1:-1], label=r"$\Pi(y, \tau)$", where="post")
            ax.legend()
        ax.set_ylabel("cumulative probability")
        ax.set_xlabel(r"$y$")
        fig.tight_layout()
        plt.close(fig)  # Prevent implicit display
        return fig
