"""Online conformal regressors.

This module implements conformal regressors that produce prediction intervals
with guaranteed coverage. Includes ridge regression, kernel ridge regression,
and Lasso-based conformal regressors.
"""

import time
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

__all__ = [
    "ConformalRidgeRegressor",
    "KernelConformalRidgeRegressor",
    "ConformalLassoRegressor",
    "ConformalPredictionInterval",
]


def MACHINE_EPSILON(x):
    return np.abs(x) * np.finfo(np.float64).eps


default_epsilon = 0.1

# FIXME: The p-values for all regressors should be modified. We should be able to compute the upper, lower, and combined p-value


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

    def __init__(self, lower, upper, epsilon):
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon

    def __contains__(self, y):
        return self.lower <= y <= self.upper

    def width(self):
        return self.upper - self.lower

    def __repr__(self):
        return repr((self.lower, self.upper))

    def __str__(self):
        return f"({self.lower}, {self.upper})"


class ConformalRegressor:
    """Base class for online conformal regressors.

    Provides shared methods for computing p-values, constructing prediction
    intervals, and processing datasets in the online setting.
    """

    def __init__(self, epsilon=default_epsilon):
        self.epsilon = epsilon

    # TODO The methods _get_upper and _get_lower could be called with a vector of significance levels,
    #      say one casual and one highly confident. In pracice, this could be useful when the aim is
    #      descision support. This would then have to play nice wiht everything else...

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
    def _calculate_p(Alpha, tau=None, c_type="nonconformity"):
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
            raise Exception()
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

    @staticmethod
    def err(Gamma, y):
        return int(y not in Gamma)

    @staticmethod
    def width(Gamma):
        return Gamma.width()

    def learn_many(self, X, y):
        for x1, y1 in zip(X, y):
            self.learn_one(x1, y1)

    def process_dataset(self, X, y, epsilon=None, init_train=0, return_results=False):
        if epsilon is None:
            epsilon = self.epsilon

        Err = 0
        Width = 0

        X_train = X[:init_train]
        y_train = y[:init_train]
        X_run = X[init_train:]
        y_run = y[init_train:]

        if return_results:
            res = np.zeros(shape=(y_run.shape[0], 2))
            prediction_sets = {}

        self.learn_initial_training_set(X=X_train, y=y_train)

        time_init = time.time()
        for i, (obj, lab) in enumerate(zip(X_run, y_run)):
            # Make prediction
            Gamma = self.predict(obj, epsilon=epsilon)

            # Check error
            Err += self.err(Gamma, lab)

            # Learn the label
            self.learn_one(obj, lab)

            # Width of interval
            width = self.width(Gamma)
            Width += width

            if return_results:
                res[i, 0] = Err
                res[i, 1] = width
                prediction_sets[i] = Gamma

        time_process = time.time() - time_init

        result = {
            "Efficiency": {
                "Average error": Err / self.y.shape[0],
                "Average width": Width / self.y.shape[0],
                "Time": time_process,
            }
        }
        if return_results:
            result["Prediction sets"] = prediction_sets
            result["Cummulative Err"] = res[:, 0]
            result["Width"] = res[:, 1]

        return result


class ConformalRidgeRegressor(ConformalRegressor):
    """
    Conformal ridge regression (Algorithm 2.4 in Algorithmic Learning in a Random World)

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

    # TODO: Fix gracefull error handling when the matrix is singular. It should raise an exception, but we could
    #       specify that it can be handled by changing the ridge parameter.

    def __init__(
        self, a=0, warnings=True, autotune=False, verbose=0, rnd_state=None, studentised=False, epsilon=default_epsilon
    ):
        """
        The ridge parameter (L2 regularisation) is a.
        Setting autotune=True automatically tunes the ridge parameter using generalized cross validation when learning initial training set.
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
        self.rnd_gen = np.random.default_rng(rnd_state)

        # Do we use the studentised residuals
        self.studentised = studentised

    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.Id = np.identity(self.p)
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)

    def learn_one(self, x, y, precomputed=None):
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
            if hasattr(self, "h"):
                self.y = np.append(self.y, y.reshape(1, self.h), axis=0)
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
                    # print(self.X)
                    # print(self.Id)
                    # print(self.a)
                    self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
                else:
                    # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                    # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
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
                # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
                self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)

                # Check the rank
                if self.warnings:
                    self.check_matrix_rank(self.XTXinv)

    def compute_A_and_B_OLD(self, X, XTXinv, y):
        n = X.shape[0]
        # Hat matrix (This block is the time consuming one...)
        H = X @ XTXinv @ X.T
        C = np.identity(n) - H
        A = C @ np.append(y, 0)  # Elements of this vector are denoted ai
        B = C @ np.append(np.zeros((n - 1,)), 1)  # Elements of this vector are denoted bi
        if self.studentised:
            h = H.diagonal()
            A = A / np.sqrt(1 - h)
            B = B / np.sqrt(1 - h)
        # Nonconformity scores are A + yB = y - yhat
        return A, B

    def compute_A_and_B(self, X, XTXinv, y):
        """
        Efficient and correct computation of A and B for conformal ridge regression.
        X: (n, d) augmented matrix (last row is test object)
        XTXinv: (d, d) inverse of X.T @ X + a*I for augmented X
        y: (n-1,) training labels (no test label)
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
            A = A / np.sqrt(1 - H_diag + 1e-12)
            B = B / np.sqrt(1 - H_diag + 1e-12)

        return A, B

    def predict(self, x, epsilon=None, bounds="both", return_update=False, debug_time=False):
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
            tic = time.time()
            # Add row to X matrix
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            toc_add_row = time.time() - tic
            n = X.shape[0]
            XTXinv = None

            # Check that the significance level is not too small. If it is, return infinite prediction interval
            if bounds == "both":
                if not (epsilon >= 2 / n):
                    if self.warnings:
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(2 / epsilon))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, XTXinv, None, None)
                    else:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)
            else:
                if not (epsilon >= 1 / n):
                    if self.warnings:
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(1 / epsilon))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, XTXinv, None, None)
                    else:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)

            tic = time.time()
            # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
            # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

            XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
            toc_update_XTXinv = time.time() - tic

            tic = time.time()
            A, B = self.compute_A_and_B(X, XTXinv, self.y)
            toc_nc = time.time() - tic

            if self.studentised:
                tic = time.time()
                t = (A[:-1] - A[-1]) / (B[-1] - B[:-1])
                t.sort()
                l_dic = {i + 1: val for i, val in enumerate(t)}
                u_dic = {i + 1: val for i, val in enumerate(t)}
                toc_dics = time.time() - tic
            else:
                tic = time.time()
                l_dic, u_dic = self._vectorised_l_and_u(A, B)
                toc_dics = time.time() - tic

            if bounds == "both":
                lower = self._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
                upper = self._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
            elif bounds == "lower":
                lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                upper = np.inf
            elif bounds == "upper":
                lower = -np.inf
                upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
            else:
                raise Exception

            if debug_time:
                print(f"Add row: {toc_add_row}")
                print(f"Update kernel: {toc_update_XTXinv}")
                print(f"NC scores: {toc_nc}")
                print(f"l and u: {toc_dics}")
                print()
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            X = x.reshape(1, -1)
            XTXinv = None
            A = None
            B = None

            lower = -np.inf
            upper = np.inf

        if return_update:
            return self._construct_Gamma(lower, upper, epsilon), build_precomputed(X, XTXinv, A, B)
        else:
            return self._construct_Gamma(lower, upper, epsilon)

    def compute_p_value(self, x, y, bounds="both", precomputed=None, tau=None, smoothed=True):
        """
        Computes the smoothed p-value of the example (x, y).
        """
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            assert np.allclose(x, precomputed["X"][-1])
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
                raise Exception('bounds must be one of "both", "lower", "upper"')

            if smoothed:
                p = self._calculate_p(Alpha, tau, c_type=c_type)
            else:
                p = self._calculate_p(Alpha, c_type=c_type)
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
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)

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

    # TODO
    def prune_training_set(self):
        """
        Just an idea at the moment, but perhaps we should have some inclusion criteria for examples to only include the informative ones. Could improve accuracy, but also significantly decrease computation time if we have a large dataset.
        """
        raise NotImplementedError

    def check_matrix_rank(self, M):
        """
        Check if a matrix has full rank <==> is invertible
        Returns False if matrix is rank deficient
        NOTE In numerical linear algebra it is a bit more subtle. The condition number can tell us more.

        >>> cp = ConformalRidgeRegressor(warnings=False)
        >>> cp.check_matrix_rank(np.array([[1, 0], [1, 0]]))
        False
        >>> cp.check_matrix_rank(np.array([[1, 0], [0, 1]]))
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


class KernelConformalRidgeRegressor(ConformalRegressor):
    # TODO Add doctests to methods where applicable

    def __init__(self, kernel, a=0, warnings=True, verbose=0, rnd_state=None, epsilon=default_epsilon):
        """
        KernelConformalRidgeRegressor requires a kernel. Some common kernels are found in kernels.py, but it is
        also compatible with (most) kernels from e.g. scikit-learn.
        Custom kernels can also be passed as callable functions.
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

        self.rnd_gen = np.random.default_rng(rnd_state)

    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        Id = np.identity(self.X.shape[0])

        self.K = self.kernel(self.X)
        self.Kinv = np.linalg.inv(self.K + self.a * Id)

    @staticmethod
    def _update_Kinv(Kinv, k, kappa):
        # print(f'K: {K}')
        # print(f'k: {k}')
        # print(f'kappa: {kappa}')
        d = 1 / (kappa - k.T @ Kinv @ k)
        return np.block([[Kinv + d * Kinv @ k @ k.T @ Kinv, -d * Kinv @ k], [-d * k.T @ Kinv, d]])

    @staticmethod
    def _update_K(K, k, kappa):
        # print(f'K: {K}')
        # print(f'k: {k}')
        # print(f'kappa: {kappa}')
        return np.block([[K, k], [k.T, kappa]])

    def learn_one(self, x, y, precomputed=None):
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
                self.Kinv = np.linalg.inv(self.K + self.a * Id)

        else:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1, -1)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
            elif self.X.shape[0] == 1:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
            else:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                self.K = self._update_K(self.K, k, kappa)
                self.Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    @staticmethod
    def compute_A_and_B(X, K, Kinv, y):
        # print(f'X: {X}')
        # print(f'K: {K}')
        # print(f'Kinv: {Kinv}')
        # print(f'y: {y}')
        n = X.shape[0]
        H = Kinv @ K
        C = np.identity(n) - H
        A = C @ np.append(y, 0)  # Elements of this vector are denoted ai
        B = C @ np.append(np.zeros((n - 1,)), 1)  # Elements of this vector are denoted bi
        # Nonconformity scores are A + yB = y - yhat
        return A, B

    def predict(self, x, epsilon=None, bounds="both", return_update=False, debug_time=False):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        TODO Add possibility to learn object to save time

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
            tic = time.time()

            # Temporarily update kernel matrix
            k = self.kernel(self.X, x).reshape(-1, 1)
            kappa = self.kernel(x, x)
            K = self._update_K(self.K, k, kappa)
            Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)

            toc_update_kernel = time.time() - tic

            tic = time.time()
            # Add row to X matrix
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            toc_add_row = time.time() - tic
            n = X.shape[0]

            # Check that the significance level is not too small. If it is, return infinite prediction interval
            if bounds == "both":
                if not (epsilon >= 2 / n):
                    if self.warnings:
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(2 / epsilon))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(
                            X, K, Kinv, None, None
                        )
                    else:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)
            else:
                if not (epsilon >= 1 / n):
                    if self.warnings:
                        warnings.warn(
                            f"Significance level epsilon is too small for training set. Need at least {int(np.ceil(1 / epsilon))} examples. Increase or add more examples",
                            stacklevel=2,
                        )
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(
                            X, K, Kinv, None, None
                        )
                    else:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)

            tic = time.time()
            A, B = self.compute_A_and_B(X, K, Kinv, self.y)
            toc_nc = time.time() - tic

            tic = time.time()
            l_dic, u_dic = self._vectorised_l_and_u(A, B)
            toc_dics = time.time() - tic

            if bounds == "both":
                lower = self._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
                upper = self._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
            elif bounds == "lower":
                lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                upper = np.inf
            elif bounds == "upper":
                lower = -np.inf
                upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
            else:
                raise Exception

            if debug_time:
                print(f"Add row: {toc_add_row}")
                print(f"Update kernel: {toc_update_kernel}")
                print(f"NC scores: {toc_nc}")
                print(f"l and u: {toc_dics}")
                print()
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            X = x.reshape(1, -1)
            K = None
            Kinv = None
            A = None
            B = None

            lower = -np.inf
            upper = np.inf

        if return_update:
            return self._construct_Gamma(lower, upper, epsilon), build_precomputed(X, K, Kinv, A, B)
        else:
            return self._construct_Gamma(lower, upper, epsilon)

    def compute_p_value(self, x, y, bounds="both", precomputed=None, tau=None, smoothed=True):
        """
        Computes the smoothed p-value of the example (x, y).
        """
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            assert np.allclose(x, precomputed["X"][-1])
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
                raise Exception('bounds must be one of "both", "lower", "upper"')

            if smoothed:
                p = self._calculate_p(Alpha, tau, c_type=c_type)
            else:
                p = self._calculate_p(Alpha, c_type=c_type)
        else:
            if smoothed:
                p = tau
            else:
                p = 1

        return p

    def compute_smoothed_p_value(self, x, y, precomputed=None):
        """
        Computes the smoothed p-value of the example (x, y).
        Smoothed p-values can be used to test the exchangeability assumption.
        """

        # Inner method to compute the p-value from NC scores
        def calc_p(A, B, y):
            # Nonconformity scores are A + yB = y - yhat
            Alpha = A + y * B
            alpha_y = Alpha[-1]
            gt = np.where(Alpha > alpha_y)[0].shape[0]
            eq = np.where(Alpha == alpha_y)[0].shape[0]
            tau = self.rnd_gen.uniform(0, 1)
            p_y = (gt + tau * eq) / Alpha.shape[0]
            return p_y

        if precomputed is not None:
            A = precomputed["A"]
            B = precomputed["B"]
            X = precomputed["X"]
            K = precomputed["K"]
            Kinv = precomputed["Kinv"]

            if A is not None and B is not None:
                p_y = calc_p(A, B, y)
            else:
                if Kinv is not None and X is not None and K is not None:
                    A, B = self.compute_A_and_B(X, K, Kinv, self.y)
                    p_y = calc_p(A, B, y)

                else:
                    p_y = self.rnd_gen.uniform(0, 1)

        else:
            if self.Kinv is not None:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                K = self._update_K(self.K, k, kappa)
                Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                X = np.append(self.X, x.reshape(1, -1), axis=0)

                A, B = self.compute_A_and_B(X, K, Kinv, self.y)
                p_y = calc_p(A, B, y)

            else:
                p_y = self.rnd_gen.uniform(0, 1)
        return p_y


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


class ConformalLassoRegressor(ConformalRegressor):
    """
    Conformal prediction with Lasso/elastic net using the piecewise linear
    homotopy from Lei & Fithian. Computes exact conformal prediction sets
    without grid search by tracing how residuals evolve as the test label varies.

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
    rnd_state : int or None
        Random seed.

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
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.X = None
        self.y = None
        self.beta = None  # Current Lasso solution
        self.Sigma = None  # X^T X / n (sample covariance)

    def learn_initial_training_set(self, X, y):
        """Fit initial Lasso on the training data."""
        self.X = X.copy()
        self.y = y.copy()
        n, p = X.shape
        self.Sigma = X.T @ X / n

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

    def learn_one(self, x, y, precomputed=None):
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

        # Update sample covariance incrementally: Sigma_new = (n*Sigma_old + x x^T) / (n+1)
        n = self.X.shape[0]
        self.Sigma = ((n - 1) * self.Sigma + np.outer(x, x)) / n

        if precomputed is not None and precomputed.get("beta") is not None:
            self.beta = precomputed["beta"]
        else:
            # Refit from scratch (warm-started from current beta)
            self.beta = _solve_lasso(self.X, self.y, self.lam, rho=self.rho, warm_start=self.beta)

    def predict(self, x, epsilon=None, return_update=False):
        """
        Compute the conformal prediction set at x using the homotopy algorithm.

        Returns a ConformalPredictionInterval (if the set is a single interval)
        or a list of (lower, upper) tuples otherwise.
        """
        if epsilon is None:
            epsilon = self.epsilon

        x = np.atleast_1d(x).ravel()
        n = self.X.shape[0]

        if n < 2:
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

        # Run homotopy in both directions
        intervals_pos = self._run_homotopy(x, direction=+1, t_bound=t_max)
        intervals_neg = self._run_homotopy(x, direction=-1, t_bound=t_min)

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

    def compute_p_value(self, x, y, smoothed=True, tau=None):
        """
        Compute the conformal p-value for (x, y) given current training set.
        """
        x = np.atleast_1d(x).ravel()

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

    def _run_homotopy(self, x_new, direction, t_bound):
        """
        Run the piecewise linear homotopy in one direction.

        Returns a list of (t_start, t_end) intervals that are IN the prediction set.
        """
        n = self.X.shape[0]
        p = self.X.shape[1]
        lam = self.lam
        epsilon = self.epsilon
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
                    v_inactive = v_full[J_c_k]  # will be recomputed below
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
        # Find all crossing points where |r_i(delta)| = |r_{n+1}(delta)| for some i
        crossings = []

        for i in range(n):
            # |r_i(d)| = |r_{n+1}(d)|
            # Case 1: r_i(d) = r_{n+1}(d)  => (r_train[i] - r_test) + (slopes_train[i] - slope_test)*d = 0
            # Case 2: r_i(d) = -r_{n+1}(d) => (r_train[i] + r_test) + (slopes_train[i] + slope_test)*d = 0
            # Case 3: -r_i(d) = r_{n+1}(d) => -(r_train[i] + r_test) - (slopes_train[i] + slope_test)*d = 0
            #        same as case 2
            # Case 4: -r_i(d) = -r_{n+1}(d) => same as case 1

            # So we need: r_i(d) = +/- r_{n+1}(d)
            for s in [1, -1]:
                a = r_train[i] - s * r_test
                b = slopes_train[i] - s * slope_test
                if abs(b) > 1e-15:
                    d = -a / b
                    if -1e-12 < d < dt_k + 1e-12:
                        d = np.clip(d, 0, dt_k)
                        crossings.append(d)

        # Add endpoints
        crossings = [0.0] + sorted(set(crossings)) + [dt_k]

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


if __name__ == "__main__":
    import doctest
    import sys

    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
