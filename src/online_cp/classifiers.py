"""Online conformal classifiers.

This module implements conformal classifiers that produce prediction sets
with guaranteed coverage. Includes nearest neighbours and support vector
machine-based conformal classifiers.
"""

import time

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "ConformalNearestNeighboursClassifier",
    "ConformalSupportVectorMachine",
    "ConformalPredictionSet",
]

default_epsilon = 0.1


class ConformalPredictionSet:
    """A prediction set produced by a conformal classifier.

    Parameters
    ----------
    Gamma : np.ndarray
        Array of predicted labels in the set.
    epsilon : float
        Significance level at which the set was constructed.
    """

    def __init__(self, Gamma: np.array, epsilon):
        self.elements = Gamma
        self.epsilon = epsilon

    def __contains__(self, y):
        return y in self.elements

    def __len__(self):
        return self.elements.shape[0]

    def __repr__(self):
        return repr(self.elements)

    def __str__(self):
        return str(self.elements)

    def size(self):
        return self.__len__()


class ConformalClassifier:
    """Base class for online conformal classifiers.

    Provides shared methods for computing p-values, constructing prediction
    sets, and tracking efficiency criteria (Err, OE, OF).
    """

    def __init__(self, epsilon=default_epsilon):
        self.Err = 0
        # Preferred efficiency criteria (See Protocol 3.1 ALRW)
        self.OE = 0
        self.OF = 0
        self.epsilon = epsilon

    @staticmethod
    def _compute_p_value(Alpha, tau=1, score_type="nonconformity", return_string=False):
        """
        Assumes that the (non) conformity scores are organised so that the
        test example is the last element.
        If tau is not provided, the non-smoothed p-value is computed.
        """
        alpha_n = Alpha[-1]
        if score_type == "nonconformity":
            gt = np.sum(Alpha > alpha_n)
            eq = np.sum(Alpha == alpha_n)
            p = (gt + tau * eq) / Alpha.shape[0]
            string = f"({gt} + {eq}*tau)/{Alpha.shape[0]}"

        elif score_type == "conformity":
            lt = np.sum(Alpha < alpha_n)
            eq = np.sum(Alpha == alpha_n)
            p = (lt + tau * eq) / Alpha.shape[0]
            string = f"({lt} + {eq}*tau)/{Alpha.shape[0]}"

        if return_string:
            return float(p), string
        else:
            return float(p)

    def _compute_Gamma(self, p_values, epsilon):
        Gamma = []
        for y in self.label_space:
            if p_values[y] > epsilon:
                Gamma.append(y)
        return ConformalPredictionSet(np.array(Gamma), epsilon)

    def err(self, Gamma, y):
        err = int(y not in Gamma)
        self.Err += err
        return err

    def oe(self, Gamma, y):
        if y in Gamma:
            oe = len(Gamma) - 1
        else:
            oe = len(Gamma)
        self.OE += oe
        return oe

    def of(self, p_values, y):
        of = 0
        for label, p in p_values.items():
            if not label == y:
                of += p
        self.OF += of
        return of

    def learn_many(self, X, y):
        for x1, y1 in zip(X, y):
            self.learn_one(x1, y1)

    # TEST
    def process_dataset(self, X, y, epsilon=0.1, init_train=0, return_results=False):

        self.label_space = np.unique(y)

        X_train = X[:init_train]
        y_train = y[:init_train]
        X_run = X[init_train:]
        y_run = y[init_train:]

        if return_results:
            res = np.zeros(shape=(y_run.shape[0], 3))
            prediction_sets = {}

        self.learn_initial_training_set(X=X_train, y=y_train)

        time_init = time.time()
        for i, (obj, lab) in enumerate(zip(X_run, y_run)):
            # Make prediction
            Gamma, p_values = self.predict(obj, epsilon=epsilon, return_p_values=True)

            # Check error
            self.err(Gamma, lab)

            # Learn the label
            self.learn_one(obj, lab)

            # Prefferred efficiency criteria

            # Observed excess
            self.oe(Gamma, lab)

            # Observed fuzziness
            self.of(p_values, lab)

            if return_results:
                res[i, 0] = self.OE
                res[i, 1] = self.OF
                res[i, 2] = self.Err
                prediction_sets[i] = Gamma

        time_process = time.time() - time_init

        result = {
            "Efficiency": {
                "Average error": self.Err / self.y.shape[0],
                "Average OE": self.OE / self.y.shape[0],
                "Average OF": self.OF / self.y.shape[0],
                "Time": time_process,
            }
        }
        if return_results:
            result["Prediction sets"] = (prediction_sets,)
            result["Cummulative Err"] = res[:, 2]
            result["Cummulative OE"] = res[:, 0]
            result["Cummulative OF"] = res[:, 1]

        return result


class ConformalNearestNeighboursClassifier(ConformalClassifier):
    """
    Classifier using nearest neighbours as the nonconformity measure.

    >>> cp = ConformalNearestNeighboursClassifier(k=1, rnd_state=1337, epsilon=0.1)
    >>> Gamma, p_values = cp.predict(3, return_p_values=True)
    >>> Gamma  # predict both labels, as this is the first
    array([-1,  1])
    >>> [p_values[i] for i in [-1, 1]]
    [0.8781019003471183, 0.8781019003471183]

    >>> cp.learn_one(np.int64(3), 1)

    >>> Gamma, p_values = cp.predict(-2, return_p_values=True)
    >>> Gamma  # predict both labels, as this is the first
    array([-1,  1])
    >>> [p_values[i] for i in [-1, 1]]
    [0.18552796163759344, 0.18552796163759344]
    """

    # TODO: implement: cp.learn_several([[3,1],[4,7],[5,2]], [1, -1, 1])

    # TODO Write tests

    def __init__(
        self,
        k=1,
        label_space=None,
        distance="euclidean",
        distance_func=None,
        verbose=0,
        rnd_state=None,
        n_jobs=None,
        epsilon=default_epsilon,
    ):
        super().__init__(epsilon=epsilon)
        self.label_space = label_space if label_space is not None else np.array([-1, 1])

        self.k = k

        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = "custom"

        self.y = np.empty(0)
        self.X = None
        self.D = None

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.n_jobs = n_jobs

    def reset(self):

        self.__init__(self.k, self.label_space)

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

    def learn_initial_training_set(self, X, y):
        if X.shape[0] > 0:
            self.X = X
            self.y = y
            self.D = self.distance_func(X)

    @staticmethod
    def update_distance_matrix(D, d):
        return np.block([[D, d], [d.T, np.array([0])]])

    def _find_nearest_distances(self, D, y):
        n = D.shape[0]

        # Initialize arrays to store the results
        same_label_distances = np.full(n, np.inf)
        different_label_distances = np.full(n, np.inf)

        for i in range(n):
            # Create a mask for the same and different labels
            same_label_mask = y == y[i]
            different_label_mask = y != y[i]

            # Ignore the distance to itself by setting it to np.inf
            same_label_mask[i] = False

            # Extract distances for the same label
            if np.any(same_label_mask):
                same_label_distances[i] = np.sort(D[i, same_label_mask])[: self.k].mean()

            # Extract distances for the different label
            if np.any(different_label_mask):
                different_label_distances[i] = np.sort(D[i, different_label_mask])[: self.k].mean()

        return same_label_distances, different_label_distances

    def learn_one(self, x, y, D=None):
        # Learn label y
        self.y = np.append(self.y, y)

        # Learn object
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.D = self.distance_func(self.X)
        else:
            if D is None:
                d = self.distance_func(self.X, x)
                D = self.update_distance_matrix(self.D, d)
            self.D = D
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    def predict(self, x, epsilon=None, return_p_values=False, return_update=False, verbose=0):
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.y.shape[0] >= 1:
            tic = time.time()
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            time_update_D = time.time() - tic

            tic = time.time()
            if self.n_jobs is not None:

                def process_label(label):
                    y = np.append(self.y, label)
                    same_label_distances, different_label_distances = self._find_nearest_distances(D, y)

                    Alpha = same_label_distances / different_label_distances
                    if verbose > 10:
                        print(f"Nonconformity scores for hypothesis y={label}: {Alpha}")
                        _, string = self._compute_p_value(Alpha, tau, "nonconformity", return_string=True)
                        print(f"p-value for hypothesis y={label}: {string}")

                    return label, self._compute_p_value(Alpha, tau, "nonconformity")

                results = Parallel(n_jobs=self.n_jobs)(delayed(process_label)(label) for label in self.label_space)
                p_values = dict(results)
            else:
                for label in self.label_space:
                    y = np.append(self.y, label)

                    same_label_distances, different_label_distances = self._find_nearest_distances(D, y)

                    Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)

                    if verbose > 10:
                        print(f"Nonconformity scores for hypothesis y={label}: {Alpha}")
                        p_values[label], string = self._compute_p_value(Alpha, tau, "nonconformity", return_string=True)
                        print(f"p-value for hypothesis y={label}: {string}")

                    p_values[label] = self._compute_p_value(Alpha, tau, "nonconformity")
            time_compute_p_values = time.time() - tic

            tic = time.time()
            Gamma = self._compute_Gamma(p_values, epsilon)
            time_Gamma = time.time() - tic

            self.time_dict = {
                "Update distance matrix": time_update_D,
                "Compute p-values": time_compute_p_values,
                "Compute Gamma": time_Gamma,
            }

        else:
            for label in self.label_space:
                Alpha = np.array([np.inf])
                if verbose > 10:
                    print(f"Nonconformity scores for hypothesis y={label}: {Alpha}")
                    p_values[label], string = self._compute_p_value(Alpha, tau, "nonconformity", return_string=True)
                    print(f"p-value for hypothesis y={label}: {string}")
                p_values[label] = self._compute_p_value(Alpha, tau, "nonconformity")
            Gamma = self._compute_Gamma(p_values, epsilon)
            D = None
            self.time_dict = {}

        if return_update:
            if return_p_values:
                return Gamma, p_values, D
            else:
                return Gamma, D
        else:
            if return_p_values:
                return Gamma, p_values
            else:
                return Gamma


class ConformalClassifierWrapper(ConformalClassifier):
    """
    The following scikit-learn classifiers should in priciple be compatible (they have a predict_proba method):
    AdaBoostClassifier
    BaggingClassifier
    BernoulliNB
    CalibratedClassifierCV
    CategoricalNB
    ComplementNB
    DecisionTreeClassifier
    DummyClassifier
    ExtraTreeClassifier
    ExtraTreesClassifier
    GaussianNB
    GaussianProcessClassifier
    GradientBoostingClassifier
    HistGradientBoostingClassifier
    KNeighborsClassifier
    LabelPropagation
    LabelSpreading
    LinearDiscriminantAnalysis
    LogisticRegression
    LogisticRegressionCV
    MLPClassifier
    MultinomialNB
    NearestCentroid
    QuadraticDiscriminantAnalysis
    RadiusNeighborsClassifier
    RandomForestClassifier
    """

    def __init__(self, learner, label_space=None, epsilon=default_epsilon, verbose=0, rnd_state=None, n_jobs=None):
        super().__init__(epsilon)

        assert hasattr(learner, "predict_proba")

        self.learner = learner

        self.label_space = label_space if label_space is not None else np.array([-1, 1])

        self.y = np.empty(0)
        self.X = None

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.n_jobs = n_jobs

    def learn_one(self, x, y, D=None):
        # Learn label y
        self.y = np.append(self.y, y)
        # Learn object
        if self.X is None:
            self.X = x.reshape(1, -1)
        else:
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    def predict(self, x, epsilon=None, return_p_values=False, verbose=0):
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        # TODO: Fix parallellisation

        # TODO: Some models have minimum requirements for the training set.
        # if self.y.shape[0] >= 1:
        try:
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            # Label loop
            for y in self.label_space:
                Y = np.append(self.y, y)
                self.learner.fit(X, Y)

                Prob = self.learner.predict_proba(X)
                if Prob.shape[1] < self.label_space.size:
                    zeros_to_add = self.label_space.size - Prob.shape[1]
                    Prob = np.hstack([Prob, np.zeros((Prob.shape[0], zeros_to_add))])

                Alpha = Prob[np.arange(len(Y)), Y.astype("int")]
                p_values[y] = self._compute_p_value(Alpha, tau, "conformity")
            Gamma = self._compute_Gamma(p_values, epsilon)
        # else:
        except ValueError:
            for label in self.label_space:
                Alpha = np.array([np.inf])
                p_values[label] = self._compute_p_value(Alpha, tau, "conformity")
            Gamma = self._compute_Gamma(p_values, epsilon)

        if return_p_values:
            return Gamma, p_values
        else:
            return Gamma


class ConformalSupportVectorMachine(ConformalClassifier):
    """
    Conformal classifier using the Support Vector Machine with Lagrange
    multiplier nonconformity measure (ALRW Ch. 3).

    For each candidate label, one-vs-rest binarization is applied and the
    SVM dual is solved on the augmented training set. The Lagrange multiplier
    alpha_i is the nonconformity score for example i: alpha_i = 0 means well
    inside the margin (conforming), alpha_i = C means on the margin boundary
    or misclassified (maximally nonconforming).

    Supports multi-class classification via one-vs-rest decomposition.
    The Gram matrix is label-independent and reused across all candidate labels.

    Parameters
    ----------
    kernel : Kernel, callable, or str
        - An online_cp.kernels.Kernel instance (native).
        - A callable f(X, Y) -> (n, m) Gram matrix (sklearn-style).
        - A string: 'linear', 'rbf', 'poly'.
    C : float
        Regularization parameter (upper bound on alpha_i). Default 1.0.
    label_space : array-like
        The set of possible labels. Supports any number of classes.
        Default [-1, 1].
    sigma : float
        Bandwidth for RBF kernel when kernel='rbf'. Default 1.0.
    degree : int
        Degree for polynomial kernel when kernel='poly'. Default 3.
    coef0 : float
        Constant for polynomial kernel. Default 1.0.
    smo_tol : float
        Tolerance for SMO convergence. Default 1e-3.
    smo_max_iter : int
        Maximum SMO iterations. Default 1000.
    epsilon : float
        Significance level. Default 0.1.
    rnd_state : int or None
        Random seed.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.vstack([np.random.normal(loc=-1, size=(20, 2)), np.random.normal(loc=1, size=(20, 2))])
    >>> y = np.array([-1] * 20 + [1] * 20)
    >>> svm = ConformalSupportVectorMachine(kernel="rbf", sigma=1.0, C=10.0)
    >>> svm.learn_initial_training_set(X[:30], y[:30])
    >>> Gamma = svm.predict(X[30])
    >>> y[30] in Gamma
    True
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        label_space=None,
        sigma=1.0,
        degree=3,
        coef0=1.0,
        smo_tol=1e-3,
        smo_max_iter=1000,
        epsilon=default_epsilon,
        rnd_state=None,
    ):
        super().__init__(epsilon=epsilon)
        self.C = C
        self.label_space = label_space if label_space is not None else np.array([-1, 1])
        self.sigma = sigma
        self.degree = degree
        self.coef0 = coef0
        self.smo_tol = smo_tol
        self.smo_max_iter = smo_max_iter
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.X = None
        self.y = np.empty(0)
        self.K = None  # Cached Gram matrix

        # Resolve kernel
        self._kernel = self._resolve_kernel(kernel)

    def _resolve_kernel(self, kernel):
        """Resolve kernel specification into a callable with our interface."""
        try:
            from online_cp.kernels import GaussianKernel, Kernel, LinearKernel, PolynomialKernel
        except ModuleNotFoundError:
            from kernels import GaussianKernel, Kernel, LinearKernel, PolynomialKernel

        if isinstance(kernel, Kernel):
            return kernel
        elif isinstance(kernel, str):
            if kernel == "linear":
                return LinearKernel()
            elif kernel == "rbf":
                return GaussianKernel(sigma=self.sigma)
            elif kernel == "poly":
                return PolynomialKernel(d=self.degree, c=self.coef0)
            else:
                raise ValueError(f"Unknown kernel string: '{kernel}'. Use 'linear', 'rbf', or 'poly'.")
        elif callable(kernel):
            # Wrap sklearn-style callable: f(X, Y) -> matrix
            return _SklearnKernelAdapter(kernel)
        else:
            raise TypeError(f"kernel must be a Kernel instance, callable, or string, got {type(kernel)}")

    def _compute_gram(self, X):
        """Compute full Gram matrix."""
        return self._kernel(X)

    def _compute_kernel_row(self, X, x):
        """Compute kernel between all rows of X and a single point x."""
        return self._kernel(X, x).ravel()

    def learn_initial_training_set(self, X, y):
        """Store training data and precompute Gram matrix."""
        self.X = X.copy()
        self.y = y.copy().astype(float)
        self.K = self._compute_gram(X)

    def learn_one(self, x, y):
        """Learn a new example, updating stored data and Gram matrix."""
        x = np.atleast_1d(x).ravel()
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=float)
            self.K = self._compute_gram(self.X)
        else:
            # Compute new kernel row
            k_row = self._compute_kernel_row(self.X, x)
            kappa = (
                self._kernel(x.reshape(1, -1)).item()
                if self._kernel(x.reshape(1, -1)).ndim > 0
                else self._kernel(x.reshape(1, -1))
            )
            # Extend Gram matrix
            n = self.K.shape[0]
            K_new = np.empty((n + 1, n + 1))
            K_new[:n, :n] = self.K
            K_new[:n, n] = k_row
            K_new[n, :n] = k_row
            K_new[n, n] = kappa
            self.K = K_new
            self.X = np.vstack([self.X, x.reshape(1, -1)])
            self.y = np.append(self.y, float(y))

    def predict(self, x, epsilon=None, return_p_values=False):
        """
        Predict the conformal prediction set for object x.

        For each candidate label, augment the training set with (x, label),
        solve the SVM dual, and use alpha_i as nonconformity scores.
        """
        if epsilon is None:
            epsilon = self.epsilon

        x = np.atleast_1d(x).ravel()
        tau = self.rnd_gen.uniform()
        p_values = {}

        if self.X is None or self.y.shape[0] == 0:
            # No training data — predict all labels
            for label in self.label_space:
                p_values[label] = 1.0
            Gamma = self._compute_Gamma(p_values, epsilon)
            if return_p_values:
                return Gamma, p_values
            return Gamma

        # Compute kernel row between training set and test point
        k_row = self._compute_kernel_row(self.X, x)
        kappa = self._kernel(x.reshape(1, -1))
        if np.ndim(kappa) > 0:
            kappa = kappa.item()

        # Build augmented Gram matrix (n+1 x n+1)
        n = self.K.shape[0]
        K_aug = np.empty((n + 1, n + 1))
        K_aug[:n, :n] = self.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        # For each candidate label, solve SVM and compute p-value
        for label in self.label_space:
            y_aug = np.append(self.y, float(label))

            # Binarize: one-vs-rest (label -> +1, everything else -> -1)
            y_binary = np.where(y_aug == label, 1.0, -1.0)

            # Solve SVM dual (no warm start: solutions differ greatly across labels)
            alpha, _ = _smo_solve(K_aug, y_binary, self.C, tol=self.smo_tol, max_iter=self.smo_max_iter)

            # NCM = alpha_i (nonconformity: larger alpha = more nonconforming)
            p_values[label] = self._compute_p_value(alpha, tau, "nonconformity")

        Gamma = self._compute_Gamma(p_values, epsilon)

        if return_p_values:
            return Gamma, p_values
        return Gamma

    def compute_p_value(self, x, y):
        """Compute the conformal p-value for (x, y) given current training set."""
        x = np.atleast_1d(x).ravel()
        tau = self.rnd_gen.uniform()

        if self.X is None or self.y.shape[0] == 0:
            return 1.0

        # Build augmented Gram matrix
        k_row = self._compute_kernel_row(self.X, x)
        kappa = self._kernel(x.reshape(1, -1))
        if np.ndim(kappa) > 0:
            kappa = kappa.item()

        n = self.K.shape[0]
        K_aug = np.empty((n + 1, n + 1))
        K_aug[:n, :n] = self.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        y_aug = np.append(self.y, float(y))

        # Binarize: one-vs-rest (label -> +1, everything else -> -1)
        y_binary = np.where(y_aug == y, 1.0, -1.0)

        alpha, _ = _smo_solve(K_aug, y_binary, self.C, tol=self.smo_tol, max_iter=self.smo_max_iter)

        return self._compute_p_value(alpha, tau, "nonconformity")


class _SklearnKernelAdapter:
    """Adapter to make sklearn-style kernel callables work with our interface."""

    def __init__(self, kernel_func):
        self.kernel_func = kernel_func

    def __call__(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            return self.kernel_func(X, X)
        else:
            Y = np.atleast_2d(y)
            K = self.kernel_func(X, Y)
            return K.ravel()


def _smo_solve(K, y, C, tol=1e-3, max_iter=1000, warm_start=None):
    """
    Solve the SVM dual QP via Sequential Minimal Optimization (SMO).

    max_alpha  sum(alpha) - 0.5 * alpha^T (y y^T * K) alpha
    s.t.       0 <= alpha_i <= C,  sum(alpha_i * y_i) = 0

    Parameters
    ----------
    K : ndarray (n, n), precomputed Gram matrix
    y : ndarray (n,), labels in {-1, +1}
    C : float, upper bound on alpha
    tol : float, KKT violation tolerance
    max_iter : int
    warm_start : ndarray (n,) or None, initial alpha values

    Returns
    -------
    alpha : ndarray (n,)
    b : float, bias term
    """
    n = len(y)
    if warm_start is not None and len(warm_start) == n:
        alpha = warm_start.copy()
        # Ensure feasibility
        alpha = np.clip(alpha, 0, C)
    else:
        alpha = np.zeros(n)

    # f_cache: f_i = sum_j alpha_j y_j K_{ij} - y_i  (negated gradient component)
    # Actually the decision function value: f(x_i) = sum_j alpha_j y_j K_{ij} + b
    # For KKT checking we use E_i = f(x_i) - y_i
    # f(x_i) = (alpha * y) @ K[:, i] + b, but we track without b for simplicity
    # and compute b at the end.

    # Precompute Q = y_i * y_j * K_{ij}
    # E_i = sum_j alpha_j y_j K_{ij} - y_i (without bias, we'll account for it)

    # Use the simplified approach: track E_i = f(x_i) - y_i
    # where f(x_i) = sum_j (alpha_j * y_j * K[j,i]) + b
    # Start with b=0
    b = 0.0
    E = np.zeros(n)
    for i in range(n):
        E[i] = (alpha * y) @ K[:, i] + b - y[i]

    for _iteration in range(max_iter):
        num_changed = 0

        for i in range(n):
            # Check KKT conditions for alpha[i]
            r_i = E[i] * y[i]  # = y_i * (f(x_i) - y_i) = y_i*f(x_i) - 1

            if (r_i < -tol and alpha[i] < C) or (r_i > tol and alpha[i] > 0):
                # Select j: maximum |E_i - E_j|
                j = _select_j(i, E, n)

                # Save old alphas
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]

                # Compute bounds
                if y[i] != y[j]:
                    L = max(0.0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0.0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L >= H:
                    continue

                # Compute eta
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Update alpha[j]
                alpha[j] -= y[j] * (E[i] - E[j]) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < 1e-8:
                    continue

                # Update alpha[i]
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                # Update bias
                b1 = b - E[i] - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - E[j] - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # Update E cache
                for k in range(n):
                    E[k] = (alpha * y) @ K[:, k] + b - y[k]

                num_changed += 1

        if num_changed == 0:
            break

    return alpha, b


def _select_j(i, E, n):
    """Select the second index j that maximizes |E_i - E_j|."""
    E_i = E[i]
    max_delta = -1.0
    j = 0
    for k in range(n):
        if k == i:
            continue
        delta = abs(E_i - E[k])
        if delta > max_delta:
            max_delta = delta
            j = k
    return j


if __name__ == "__main__":
    import doctest
    import sys

    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
