"""Online conformal classifiers.

This module implements conformal classifiers that produce prediction sets
with guaranteed coverage. Includes nearest neighbours and support vector
machine-based conformal classifiers.
"""

from __future__ import annotations

import copy
import time
import warnings
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform

try:
    from ._serialization import SerializableMixin
except ImportError:
    from _serialization import SerializableMixin

# Numba is optional — provides ~10x speedup for the SMO solver
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:

    def njit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    HAS_NUMBA = False

__all__ = [
    "ConformalNearestNeighboursClassifier",
    "ConformalSupportVectorMachine",
    "ConformalPredictionSet",
    "MultiLevelPredictionSet",
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

    def __init__(self, Gamma: NDArray[Any], epsilon: float) -> None:
        self.elements = Gamma
        self.epsilon = epsilon

    def __contains__(self, y: Any) -> bool:
        return y in self.elements

    def __len__(self) -> int:
        return self.elements.shape[0]

    def __repr__(self):
        return repr(self.elements)

    def __str__(self):
        return str(self.elements)

    def size(self):
        return self.__len__()


class MultiLevelPredictionSet:
    """Prediction sets at multiple significance levels.

    Returned when ``predict`` is called with an array-like ``epsilon``.

    Parameters
    ----------
    predictions : dict
        Mapping ``{epsilon: ConformalPredictionSet}``.
    """

    def __init__(self, predictions: dict[float, ConformalPredictionSet]) -> None:
        self._predictions = dict(sorted(predictions.items()))

    @property
    def levels(self) -> list[float]:
        """Sorted list of significance levels."""
        return list(self._predictions.keys())

    def __getitem__(self, eps: float) -> ConformalPredictionSet:
        return self._predictions[eps]

    def __iter__(self):
        return iter(self._predictions.items())

    def __len__(self) -> int:
        return len(self._predictions)

    def __contains__(self, y: Any) -> bool:
        """True if y is covered at all levels."""
        return all(y in gamma for gamma in self._predictions.values())

    def coverage(self, y: Any) -> dict[float, bool]:
        """Return dict of {epsilon: bool} indicating coverage at each level."""
        return {eps: (y in gamma) for eps, gamma in self._predictions.items()}

    def __repr__(self):
        parts = [f"  ε={eps}: {gamma}" for eps, gamma in self._predictions.items()]
        return "MultiLevelPredictionSet(\n" + "\n".join(parts) + "\n)"


class ConformalClassifier(SerializableMixin):
    """Base class for online conformal classifiers.

    Provides shared methods for computing p-values and constructing prediction sets.
    """

    _SAVE_PARAMS: tuple = ("epsilon",)
    _SAVE_STATE: tuple = ()

    def __init__(self, epsilon: float | NDArray[np.floating[Any]] = default_epsilon) -> None:
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

        else:
            raise ValueError(f"score_type must be 'nonconformity' or 'conformity', got '{score_type}'")

        if return_string:
            return float(p), string
        else:
            return float(p)

    def _compute_Gamma(self, p_values, epsilon):
        if hasattr(epsilon, '__iter__'):
            predictions = {}
            for eps in epsilon:
                Gamma = []
                for y in self.label_space:
                    if p_values[y] > eps:
                        Gamma.append(y)
                predictions[eps] = ConformalPredictionSet(np.array(Gamma), eps)
            return MultiLevelPredictionSet(predictions)
        Gamma = []
        for y in self.label_space:
            if p_values[y] > epsilon:
                Gamma.append(y)
        return ConformalPredictionSet(np.array(Gamma), epsilon)


class ConformalNearestNeighboursClassifier(ConformalClassifier):
    """
    Classifier using nearest neighbours as the nonconformity measure.

    >>> cp = ConformalNearestNeighboursClassifier(k=1, label_space=[-1, 1], rnd_state=1337, epsilon=0.1)
    >>> Gamma, p_values = cp.predict(3, return_p_values=True)
    >>> Gamma  # predict both labels, as this is the first
    array([-1,  1])
    >>> [round(p_values[i], 4) for i in [-1, 1]]
    [0.8781, 0.8781]

    >>> cp.learn_one(np.int64(3), 1)

    >>> Gamma, p_values = cp.predict(-2, return_p_values=True)
    >>> Gamma  # predict both labels, as this is the first
    array([-1,  1])
    >>> [round(p_values[i], 4) for i in [-1, 1]]
    [0.1855, 0.1855]
    """

    _SAVE_PARAMS: tuple = (
        "k", "label_space", "distance", "distance_func", "aggregation",
        "verbose", "rnd_state", "n_jobs", "epsilon",
    )
    _SAVE_STATE: tuple = (
        "X", "y", "D", "_label_indices", "_label_space_fixed", "label_space",
    )
    _SAVE_CALLABLES: tuple = ("distance_func",)
    _PARAM_MAP: dict = {"distance_func": "_distance_func_arg"}

    def __init__(
        self,
        k=1,
        label_space=None,
        distance="euclidean",
        distance_func=None,
        aggregation="mean",
        verbose=0,
        rnd_state=None,
        n_jobs=None,
        epsilon=default_epsilon,
    ):
        super().__init__(epsilon=epsilon)
        self._label_space_fixed = label_space is not None
        self.label_space = np.asarray(label_space) if label_space is not None else None

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
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

        self.y = np.empty(0)
        self.X = None
        self.D = None
        self._label_indices = {}

        self.verbose = verbose
        self.rnd_state = rnd_state
        self._distance_func_arg = distance_func
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.n_jobs = n_jobs

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
            self._label_indices = self._build_label_indices(y)
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

    @staticmethod
    def update_distance_matrix(D, d):
        d = np.asarray(d).reshape(-1)
        n = D.shape[0]
        D_new = np.empty((n + 1, n + 1), dtype=np.result_type(D.dtype, d.dtype))
        D_new[:n, :n] = D
        D_new[:n, n] = d
        D_new[n, :n] = d
        D_new[n, n] = 0
        return D_new

    @staticmethod
    def _build_label_indices(y):
        return {label: np.flatnonzero(y == label) for label in np.unique(y)}

    @staticmethod
    def _extend_label_indices(label_indices, label, new_index):
        extended = label_indices.copy()
        if label in extended:
            extended[label] = np.concatenate((extended[label], np.array([new_index], dtype=int)))
        else:
            extended[label] = np.array([new_index], dtype=int)
        return extended

    def _find_nearest_distances(self, D, y=None, label_indices=None):
        """Vectorized nearest same/different class distances for any k.

        Aggregates the k nearest distances using self.aggregation ('mean' or 'median').
        This extends the 1-NN nonconformity measure of ALRW2 §2.3 to k-NN.
        """
        n = D.shape[0]
        k = self.k
        agg_func = np.mean if self.aggregation == "mean" else np.median
        same_label_distances = np.full(n, np.inf)
        different_label_distances = np.full(n, np.inf)

        if label_indices is None:
            if y is None:
                raise ValueError("Either y or label_indices must be provided")
            label_indices = self._build_label_indices(y)

        all_idx = np.arange(n)
        for idx in label_indices.values():
            not_mask = np.ones(n, dtype=bool)
            not_mask[idx] = False
            not_idx = all_idx[not_mask]

            # Same-class: for points of this label, k nearest same-label neighbors
            if len(idx) > 1:
                D_sub = D[np.ix_(idx, idx)].copy()
                np.fill_diagonal(D_sub, np.inf)
                m = len(idx) - 1  # available neighbors (excluding self)
                if m >= k:
                    same_label_distances[idx] = agg_func(np.partition(D_sub, k - 1, axis=1)[:, :k], axis=1)
                else:
                    # Fewer than k same-class neighbors: use all available
                    same_label_distances[idx] = agg_func(np.sort(D_sub, axis=1)[:, :m], axis=1)

            # Different-class: for points OF this label, k nearest among all other labels
            if len(idx) > 0 and len(not_idx) > 0:
                D_sub = D[np.ix_(idx, not_idx)]
                if len(not_idx) >= k:
                    different_label_distances[idx] = agg_func(np.partition(D_sub, k - 1, axis=1)[:, :k], axis=1)
                else:
                    different_label_distances[idx] = agg_func(D_sub, axis=1)

        return same_label_distances, different_label_distances

    def learn_one(self, x: NDArray[np.floating[Any]], y: Any, precomputed: NDArray[np.floating[Any]] | None = None) -> None:
        new_index = 0 if self.X is None else self.X.shape[0]

        # Enforce label-space policy
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(
                    f"Label {y} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.array([y])
        elif y not in self.label_space:
            self.label_space = np.sort(np.append(self.label_space, y))

        # Learn label y
        self.y = np.append(self.y, y)
        if y in self._label_indices:
            self._label_indices[y] = np.concatenate((self._label_indices[y], np.array([new_index], dtype=int)))
        else:
            self._label_indices[y] = np.array([new_index], dtype=int)

        # Learn object
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.D = self.distance_func(self.X)
        else:
            if precomputed is None:
                d = self.distance_func(self.X, x)
                precomputed = self.update_distance_matrix(self.D, d)
            self.D = precomputed
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    def compute_p_value(self, x: NDArray[np.floating[Any]], y: Any, return_update: bool = False) -> float | tuple[float, NDArray[np.floating[Any]] | None]:
        """Compute conformal p-value for a single (x, y) pair.

        Only tests the given label y (not the full label space),
        making this faster than predict() when only one p-value is needed.

        Parameters
        ----------
        x : array-like
            Test object.
        y : scalar
            Hypothesized label.
        return_update : bool
            If True, also return the updated distance matrix D.

        Returns
        -------
        p_value : float
            Smoothed conformal p-value for the hypothesis that x has label y.
        D : ndarray, optional
            Updated distance matrix (only if return_update=True).
        """
        tau = self.rnd_gen.uniform(0, 1)

        if self.y.shape[0] >= 1:
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            label_indices = self._extend_label_indices(self._label_indices, y, D.shape[0] - 1)
            same_label_distances, different_label_distances = self._find_nearest_distances(D, label_indices=label_indices)
            Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)
            p_value = self._compute_p_value(Alpha, tau, "nonconformity")
        else:
            D = None
            p_value = self._compute_p_value(np.array([np.inf]), tau, "nonconformity")

        if return_update:
            return p_value, D
        return p_value

    def predict(self, x: NDArray[np.floating[Any]], epsilon: float | NDArray[np.floating[Any]] | None = None, return_p_values: bool = False, return_update: bool = False, verbose: int = 0) -> ConformalPredictionSet | MultiLevelPredictionSet:
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.label_space is None:
            Gamma = ConformalPredictionSet(np.array([]), epsilon if not hasattr(epsilon, '__iter__') else epsilon[0])
            if hasattr(epsilon, '__iter__'):
                Gamma = MultiLevelPredictionSet({eps: ConformalPredictionSet(np.array([]), eps) for eps in epsilon})
            if return_update:
                return (Gamma, {}, None) if return_p_values else (Gamma, None)
            return (Gamma, {}) if return_p_values else Gamma

        if self.y.shape[0] >= 1:
            tic = time.time()
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            time_update_D = time.time() - tic
            base_label_indices = self._label_indices
            test_index = D.shape[0] - 1

            tic = time.time()
            if self.n_jobs is not None:

                def process_label(label):
                    label_indices = self._extend_label_indices(base_label_indices, label, test_index)
                    same_label_distances, different_label_distances = self._find_nearest_distances(
                        D, label_indices=label_indices
                    )

                    Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)
                    return label, self._compute_p_value(Alpha, tau, "nonconformity")

                results = Parallel(n_jobs=self.n_jobs)(delayed(process_label)(label) for label in self.label_space)
                p_values = dict(results)
            else:
                for label in self.label_space:
                    label_indices = self._extend_label_indices(base_label_indices, label, test_index)

                    same_label_distances, different_label_distances = self._find_nearest_distances(
                        D, label_indices=label_indices
                    )

                    Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)
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
    Experimental convenience adapter for wrapping a sklearn-style classifier
    with ``predict_proba`` in a transductive conformal loop.

    Caveats
    -------
    - Slow by design: the wrapped learner is refit once per candidate label.
    - Not a first-class supported classifier in this package.
    - Semantics are narrow: scores are aligned via ``learner.classes_``.
    - Labels absent from the current fit are assigned zero score.

    Validity assumptions
    --------------------
    - Data are exchangeable.
    - The wrapped learner has stable ``predict_proba`` semantics.
    - Reproducibility may require controlling learner randomness externally.
    """

    _RECOMMENDED_ESTIMATORS = frozenset(
        {
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "HistGradientBoostingClassifier",
            "GaussianNB",
        }
    )

    _CAUTION_ESTIMATORS = frozenset(
        {
            "MLPClassifier",
            "GaussianProcessClassifier",
            "KNeighborsClassifier",
        }
    )

    _WARM_START_BENEFICIAL = frozenset(
        {
            "LogisticRegression",
            "MLPClassifier",
            "SGDClassifier",
            "Perceptron",
            "PassiveAggressiveClassifier",
        }
    )

    def __init__(self, learner, label_space=None, epsilon=default_epsilon, verbose=0, rnd_state=None, n_jobs=None, warm_start="auto"):
        super().__init__(epsilon)

        warnings.warn(
            "ConformalClassifierWrapper is experimental, slow, and only reliable "
            "for narrowly aligned label/score conventions.",
            UserWarning,
            stacklevel=2,
        )

        if not hasattr(learner, "predict_proba"):
            raise TypeError("Wrapped learner must implement predict_proba")

        self.learner = learner

        self._label_space_fixed = label_space is not None
        self.label_space = np.asarray(label_space) if label_space is not None else None

        self.y = np.empty(0)
        self.X = None

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.n_jobs = n_jobs

        # Warm-start configuration
        if warm_start == "auto":
            self._warm_start = type(learner).__name__ in self._WARM_START_BENEFICIAL
        else:
            self._warm_start = bool(warm_start)

        # Base fit cache (invalidated on learn_one)
        self._base_learner = None
        self._base_fitted = False

        self._warn_estimator_support_tier()

    def _warn_estimator_support_tier(self):
        estimator_name = type(self.learner).__name__
        if estimator_name in self._RECOMMENDED_ESTIMATORS:
            return
        if estimator_name in self._CAUTION_ESTIMATORS:
            warnings.warn(
                f"Wrapped estimator '{estimator_name}' is supported with caution; "
                "results can be unstable or slow.",
                UserWarning,
                stacklevel=2,
            )
            return
        warnings.warn(
            f"Wrapped estimator '{estimator_name}' is not in the recommended set "
            "for ConformalClassifierWrapper and may behave unexpectedly.",
            UserWarning,
            stacklevel=2,
        )

    def learn_one(self, x: NDArray[np.floating[Any]], y: Any) -> None:
        # Invalidate base fit cache
        self._base_fitted = False
        self._base_learner = None

        # Enforce label-space policy
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(
                    f"Label {y} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.array([y])
        elif y not in self.label_space:
            self.label_space = np.sort(np.append(self.label_space, y))

        # Learn label y
        self.y = np.append(self.y, y)
        # Learn object
        if self.X is None:
            self.X = x.reshape(1, -1)
        else:
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[Any]) -> None:
        # Invalidate base fit cache
        self._base_fitted = False
        self._base_learner = None

        if X.shape[0] > 0:
            self.X = X
            self.y = y
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

    def _align_scores(self, S, classes):
        """Align predict_proba score columns to self.label_space order."""
        aligned = np.zeros((S.shape[0], self.label_space.size), dtype=S.dtype)
        class_to_col = {cls: i for i, cls in enumerate(classes)}
        for j, label in enumerate(self.label_space):
            col = class_to_col.get(label)
            if col is not None:
                aligned[:, j] = S[:, col]
        return aligned

    def _fallback_prediction(self, epsilon, tau):
        p_values = {label: tau for label in self.label_space}
        return self._compute_Gamma(p_values, epsilon), p_values

    def _validate_scores(self, S, classes, expected_rows):
        if S.ndim != 2:
            warnings.warn("predict_proba must return a 2D array", UserWarning, stacklevel=3)
            return False
        if S.shape[0] != expected_rows:
            warnings.warn("predict_proba row count does not match fitted data", UserWarning, stacklevel=3)
            return False
        if S.shape[1] != len(classes):
            warnings.warn("predict_proba columns do not match learner.classes_", UserWarning, stacklevel=3)
            return False
        if not np.all(np.isfinite(S)):
            warnings.warn("predict_proba contains non-finite values", UserWarning, stacklevel=3)
            return False

        row_sums = S.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            warnings.warn(
                "predict_proba rows are not normalized to 1.0; continuing with provided scores",
                UserWarning,
                stacklevel=3,
            )
        if np.any(S < 0) or np.any(S > 1):
            warnings.warn(
                "predict_proba contains values outside [0, 1]; continuing with provided scores",
                UserWarning,
                stacklevel=3,
            )
        return True

    def _ensure_base_fit(self):
        """Lazily fit the base learner on (X_train, y_train) and cache a copy."""
        if not self._base_fitted:
            try:
                self.learner.fit(self.X, self.y)
                self._base_learner = copy.deepcopy(self.learner)
                self._base_fitted = True
            except Exception:
                # If base fit fails, proceed without caching
                self._base_learner = None
                self._base_fitted = False

    def _fit_label(self, learner, X, Y, label_to_idx, tau, y_candidate):
        """Fit learner for a single candidate label and return (label, p_value) or None on failure."""
        Y_aug = np.append(self.y, y_candidate)
        try:
            learner.fit(X, Y_aug)
            S = learner.predict_proba(X)
        except Exception:
            return None  # Signal failure

        classes = getattr(learner, "classes_", None)
        if classes is None:
            return None

        if not self._validate_scores(S, classes, expected_rows=len(Y_aug)):
            return None

        S = self._align_scores(S, classes)
        label_idx = np.array([label_to_idx.get(label, -1) for label in Y_aug], dtype=int)
        if np.any(label_idx < 0):
            return None

        Alpha = S[np.arange(len(Y_aug)), label_idx]
        p_value = self._compute_p_value(Alpha, tau, "conformity")
        return (y_candidate, p_value)

    def predict(self, x: NDArray[np.floating[Any]], epsilon: float | NDArray[np.floating[Any]] | None = None, return_p_values: bool = False, return_update: bool = False, verbose: int = 0) -> ConformalPredictionSet | MultiLevelPredictionSet:
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.label_space is None or self.X is None or self.y.shape[0] == 0:
            if self.label_space is None:
                Gamma = ConformalPredictionSet(np.array([]), epsilon if not hasattr(epsilon, '__iter__') else epsilon[0])
                if hasattr(epsilon, '__iter__'):
                    Gamma = MultiLevelPredictionSet({eps: ConformalPredictionSet(np.array([]), eps) for eps in epsilon})
                p_values = {}
            else:
                Gamma, p_values = self._fallback_prediction(epsilon, tau)
            if return_p_values:
                return Gamma, p_values
            if return_update:
                return Gamma, {}
            return Gamma

        label_to_idx = {label: i for i, label in enumerate(self.label_space)}

        if np.any(np.array([label_to_idx.get(label, -1) for label in self.y], dtype=int) < 0):
            warnings.warn("Observed training labels are not present in label_space", UserWarning, stacklevel=2)
            Gamma, p_values = self._fallback_prediction(epsilon, tau)
            if return_p_values:
                return Gamma, p_values
            if return_update:
                return Gamma, {}
            return Gamma

        X = np.append(self.X, x.reshape(1, -1), axis=0)

        # Build base fit cache (lazy)
        self._ensure_base_fit()

        # Parallel execution
        if self.n_jobs is not None and self.n_jobs != 1:

            def process_label(y_candidate):
                learner_copy = copy.deepcopy(self._base_learner) if self._base_learner is not None else copy.deepcopy(self.learner)
                if self._warm_start and hasattr(learner_copy, "warm_start"):
                    learner_copy.warm_start = True
                return self._fit_label(learner_copy, X, self.y, label_to_idx, tau, y_candidate)

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_label)(y_candidate) for y_candidate in self.label_space
            )
            for result in results:
                if result is None:
                    Gamma, p_values = self._fallback_prediction(epsilon, tau)
                    if return_p_values:
                        return Gamma, p_values
                    return Gamma
                p_values[result[0]] = result[1]

        # Sequential execution (with warm-start chaining)
        else:
            use_warm = self._warm_start and hasattr(self.learner, "warm_start")
            orig_warm_start = getattr(self.learner, "warm_start", None)

            if use_warm and self._base_learner is not None:
                # Restore from base fit and enable warm_start for chaining
                self.learner = copy.deepcopy(self._base_learner)
                self.learner.warm_start = True

            for y_candidate in self.label_space:
                result = self._fit_label(self.learner, X, self.y, label_to_idx, tau, y_candidate)
                if result is None:
                    # Restore learner state on failure
                    if use_warm and orig_warm_start is not None:
                        self.learner.warm_start = orig_warm_start
                    Gamma, p_values = self._fallback_prediction(epsilon, tau)
                    if return_p_values:
                        return Gamma, p_values
                    return Gamma
                p_values[result[0]] = result[1]

            # Restore original warm_start setting
            if use_warm and orig_warm_start is not None:
                self.learner.warm_start = orig_warm_start

        Gamma = self._compute_Gamma(p_values, epsilon)

        if return_p_values:
            return Gamma, p_values
        if return_update:
            return Gamma, {}
        return Gamma


class ConformalSupportVectorMachine(ConformalClassifier):
    """
    Conformal classifier using the Support Vector Machine.

    For each candidate label, one-vs-rest binarization is applied and the
    SVM dual is solved on the augmented training set.  Two nonconformity
    measures (NCMs) are available via the ``nonconformity`` parameter:

    ``'margin'`` *(default)* — signed-margin NCM:
        ``ncm_i = -(y_i · f(x_i))``  where  ``f(x) = K·(α·y) + b``.
        Negative for well-classified examples (conforming), positive for
        misclassified ones (nonconforming).  Produces a continuous score
        with no ties, giving tighter prediction sets on noisy data.

    ``'alpha'`` — Lagrange-multiplier NCM (ALRW Ch. 3):
        ``ncm_i = α_i``.  ``α_i = 0`` means well inside the margin
        (conforming); ``α_i = C`` means misclassified (maximally
        nonconforming).  Discrete score with many ties at 0 on
        well-separated data.

    Both measures are valid (coverage-guaranteed).  ``'margin'`` is
    generally more efficient (smaller prediction sets) when classes
    overlap; ``'alpha'`` can be preferable on small, cleanly separable
    problems.

    Supports multi-class classification via one-vs-rest decomposition.
    The Gram matrix is label-independent and reused across all candidate
    labels.

    Parameters
    ----------
    kernel : Kernel, callable, or str
        - An online_cp.kernels.Kernel instance (native).
        - A callable f(X, Y) -> (n, m) Gram matrix (sklearn-style).
        - A string: 'linear', 'rbf', 'poly'.
    C : float
        Regularization parameter (upper bound on alpha_i). Default 1.0.
    nonconformity : str
        Nonconformity measure: ``'margin'`` (default) or ``'alpha'``.
    label_space : array-like or None
        The set of possible labels. Supports any number of classes.
        If None, inferred from the first training data.
    sigma : float
        Bandwidth for RBF kernel when kernel='rbf'. Default 1.0.
    degree : int
        Degree for polynomial kernel when kernel='poly'. Default 3.
    coef0 : float
        Constant for polynomial kernel. Default 1.0.
    smo_tol : float
        Tolerance for SMO convergence. Default 1e-3.
    smo_max_iter : int
        Maximum SMO iterations. Default 5000.
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

    _SAVE_PARAMS: tuple = (
        "kernel", "C", "nonconformity", "label_space", "sigma", "degree", "coef0",
        "smo_tol", "smo_max_iter", "epsilon", "rnd_state",
    )
    _SAVE_STATE: tuple = ("X", "y", "K", "label_space", "_label_space_fixed")
    _SAVE_CALLABLES: tuple = ("kernel",)

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        nonconformity="margin",
        label_space=None,
        sigma=1.0,
        degree=3,
        coef0=1.0,
        smo_tol=1e-3,
        smo_max_iter=5000,
        epsilon=default_epsilon,
        rnd_state=None,
    ):
        if nonconformity not in ("margin", "alpha"):
            raise ValueError(f"nonconformity must be 'margin' or 'alpha', got '{nonconformity}'")
        super().__init__(epsilon=epsilon)
        self.kernel = kernel
        self.C = C
        self.nonconformity = nonconformity
        self._label_space_fixed = label_space is not None
        self.label_space = np.asarray(label_space) if label_space is not None else None
        self.sigma = sigma
        self.degree = degree
        self.coef0 = coef0
        self.smo_tol = smo_tol
        self.smo_max_iter = smo_max_iter
        self.rnd_state = rnd_state
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

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[Any]) -> None:
        """Store training data and precompute Gram matrix."""
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
        self.X = X.copy()
        self.y = y.copy().astype(float)
        self.K = self._compute_gram(X)

    def learn_one(self, x: NDArray[np.floating[Any]], y: Any) -> None:
        """Learn a new example, updating stored data and Gram matrix."""
        x = np.atleast_1d(x).ravel()

        # Enforce label-space policy
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(
                    f"Label {y} not in declared label_space "
                    f"{self.label_space.tolist()}"
                )
        elif self.label_space is None:
            self.label_space = np.array([y])
        elif y not in self.label_space:
            self.label_space = np.sort(np.append(self.label_space, y))

        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y], dtype=float)
            self.K = self._compute_gram(self.X)
        else:
            # Compute new kernel row
            k_row = self._compute_kernel_row(self.X, x)
            kappa = self._kernel(x.reshape(1, -1))
            if np.ndim(kappa) > 0:
                kappa = kappa.item()
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

    def predict(self, x: NDArray[np.floating[Any]], epsilon: float | NDArray[np.floating[Any]] | None = None, return_p_values: bool = False) -> ConformalPredictionSet | MultiLevelPredictionSet:
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

        if self.label_space is None or self.X is None or self.y.shape[0] == 0:
            # No training data — predict all labels (or empty if no label_space)
            if self.label_space is None:
                Gamma = ConformalPredictionSet(np.array([]), epsilon if not hasattr(epsilon, '__iter__') else epsilon[0])
                if hasattr(epsilon, '__iter__'):
                    Gamma = MultiLevelPredictionSet({eps: ConformalPredictionSet(np.array([]), eps) for eps in epsilon})
                if return_p_values:
                    return Gamma, {}
                return Gamma
            for label in self.label_space:
                p_values[label] = tau
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

            alpha, b = _smo_solve(K_aug, y_binary, self.C, tol=self.smo_tol, max_iter=self.smo_max_iter)

            # For multiclass (>2 labels) the one-vs-rest binarization makes the
            # Gram matrix Q depend on the hypothesised label, so the NCM is
            # equivariant only to within-class permutations.  Both NCMs restrict
            # to same-class (positive) entries in the multiclass case; for binary
            # problems all entries are exchangeable and the full vector is used.
            multiclass = len(self.label_space) > 2
            if self.nonconformity == "margin":
                f = K_aug @ (alpha * y_binary) + b   # decision function
                ncm = -(y_binary * f)                # large => nonconforming
                scores = ncm[y_binary == 1.0] if multiclass else ncm
            else:  # 'alpha'
                scores = alpha[y_binary == 1.0] if multiclass else alpha
            p_values[label] = self._compute_p_value(scores, tau, "nonconformity")

        Gamma = self._compute_Gamma(p_values, epsilon)

        if return_p_values:
            return Gamma, p_values
        return Gamma

    def compute_p_value(self, x, y):
        """Compute the conformal p-value for (x, y) given current training set."""
        x = np.atleast_1d(x).ravel()
        tau = self.rnd_gen.uniform()

        if self.X is None or self.y.shape[0] == 0:
            return tau

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

        alpha, b = _smo_solve(K_aug, y_binary, self.C, tol=self.smo_tol, max_iter=self.smo_max_iter)

        multiclass = len(self.label_space) > 2
        if self.nonconformity == "margin":
            f = K_aug @ (alpha * y_binary) + b
            ncm = -(y_binary * f)
            scores = ncm[y_binary == 1.0] if multiclass else ncm
        else:  # 'alpha'
            scores = alpha[y_binary == 1.0] if multiclass else alpha
        return self._compute_p_value(scores, tau, "nonconformity")


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


@njit(cache=True)
def _smo_loop(K, y, C, tol, max_iter, alpha, G, Q_diag):
    """Numba-jitted SMO inner loop with WSS3 working set selection."""
    n = len(y)
    for _iteration in range(max_iter):
        # Working set selection (WSS3: second-order)
        # Find i from I_up with max -y_i*G_i
        m_val = -np.inf
        i = -1
        for k in range(n):
            if ((alpha[k] < C and y[k] > 0) or (alpha[k] > 0 and y[k] < 0)):
                val = -y[k] * G[k]
                if val > m_val:
                    m_val = val
                    i = k
        if i == -1:
            break

        # WSS3: select j from I_low to maximize gain
        best_gain = -np.inf
        j = -1
        K_ii = Q_diag[i]
        for k in range(n):
            if ((alpha[k] < C and y[k] < 0) or (alpha[k] > 0 and y[k] > 0)):
                yG_k = -y[k] * G[k]
                if yG_k < m_val:
                    a_ij = K_ii + Q_diag[k] - 2.0 * K[i, k]
                    if a_ij <= 0:
                        a_ij = 1e-12
                    gain = (m_val - yG_k) ** 2 / a_ij
                    if gain > best_gain:
                        best_gain = gain
                        j = k
        if j == -1:
            break

        M_val = -y[j] * G[j]
        # Check convergence
        if m_val - M_val <= tol:
            break

        # Quadratic coefficient
        a = Q_diag[i] + Q_diag[j] - 2.0 * K[i, j]
        if a <= 0:
            a = 1e-12

        # Compute bounds
        old_ai = alpha[i]
        old_aj = alpha[j]
        s = y[i] * y[j]

        if s > 0:
            L = max(0.0, old_ai + old_aj - C)
            H = min(C, old_ai + old_aj)
        else:
            L = max(0.0, old_aj - old_ai)
            H = min(C, C + old_aj - old_ai)

        if L >= H:
            continue

        # Update alpha_j
        new_aj = old_aj + (s * G[i] - G[j]) / a
        if new_aj < L:
            new_aj = L
        elif new_aj > H:
            new_aj = H
        new_ai = old_ai + s * (old_aj - new_aj)

        d_i = new_ai - old_ai
        d_j = new_aj - old_aj
        if abs(d_i) < 1e-15 and abs(d_j) < 1e-15:
            continue

        alpha[i] = new_ai
        alpha[j] = new_aj

        # Update gradient
        ci = d_i * y[i]
        cj = d_j * y[j]
        for k in range(n):
            G[k] += y[k] * (ci * K[i, k] + cj * K[j, k])

    return alpha, G


def _smo_solve(K, y, C, tol=1e-3, max_iter=5000, warm_start=None):
    """
    Solve the SVM dual QP using Sequential Minimal Optimization (SMO).

    max_alpha  sum(alpha) - 0.5 * alpha^T (y y^T * K) alpha
    s.t.       0 <= alpha_i <= C,  sum(alpha_i * y_i) = 0

    Uses WSS3 (second-order) working set selection with a numba-jitted
    inner loop for performance (Fan, Chen & Lin 2005 / libsvm).

    Parameters
    ----------
    K : ndarray (n, n), precomputed Gram matrix
    y : ndarray (n,), labels in {-1, +1}
    C : float, upper bound on alpha
    tol : float, KKT violation tolerance for convergence
    max_iter : int, maximum number of pair updates
    warm_start : ndarray (n,) or None, initial alpha values

    Returns
    -------
    alpha : ndarray (n,)
    b : float, bias term
    """
    n = len(y)

    # Initialize alpha and gradient
    if warm_start is not None and len(warm_start) == n:
        alpha = np.clip(warm_start.copy(), 0.0, C)
        if abs(y @ alpha) > tol:
            alpha = np.zeros(n)
            G = -np.ones(n)
        else:
            G = (y * (K @ (y * alpha))) - 1.0
    else:
        alpha = np.zeros(n)
        G = -np.ones(n)

    Q_diag = np.diag(K).copy()

    # Ensure contiguous arrays for numba
    K = np.ascontiguousarray(K)
    y = np.ascontiguousarray(y)
    alpha = np.ascontiguousarray(alpha)
    G = np.ascontiguousarray(G)
    Q_diag = np.ascontiguousarray(Q_diag)

    alpha, G = _smo_loop(K, y, C, tol, max_iter, alpha, G, Q_diag)

    # Snap alpha values near boundaries
    snap_tol = max(tol * 1e-2, 1e-10)
    alpha[alpha < snap_tol] = 0.0
    alpha[alpha > C - snap_tol] = C

    # Compute bias from support vectors (0 < alpha < C)
    sv_mask = (alpha > 0) & (alpha < C)
    if np.any(sv_mask):
        decision = (alpha * y) @ K
        b = np.mean(y[sv_mask] - decision[sv_mask])
    else:
        b = 0.0

    return alpha, b



