r"""Online conformal classifiers.

Conformal classifiers output, for each test object, a *prediction set* of labels
rather than a single point prediction. Given a *nonconformity measure* $A$ that
scores how unusual a candidate labelled example looks, the predictor includes
every label whose conformal p-value exceeds the significance level $\epsilon$.
Under exchangeability the set is *valid*,

$$
\mathbb{P}\bigl(y_n \in \Gamma^\epsilon(x_n)\bigr) \geq 1 - \epsilon ,
$$

and in the online setting the long-run error rate converges to $\epsilon$
([ALRW2 Ch.2–3]).

The classifiers differ in the underlying model and nonconformity measure:

- :class:`ConformalNearestNeighboursClassifier` — NCM is the ratio of the
  ($k$ nearest) same-class distance to the nearest different-class distance.
- :class:`ConformalSupportVectorMachine` — NCM is the SVM signed margin
  (default) or the Lagrange multiplier $\alpha_i$ ([ALRW2 Ch.3]); multiclass
  via one-vs-rest.

References
----------
[ALRW2] Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*,
2nd ed., Springer, 2022.
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

from online_cp.mondrian.tree import (
    _assign_counts,
    _build_tree_summary,
    _collect_leaves,
    _draw_partition,
    _find_leaf,
    _get_ax,
    _iter_nodes,
    _render_tree,
    _resolve_feature_weights,
    _resolve_lifetime,
    _sample_mondrian_tree,
    _tree_struct_stats,
)

__all__ = [
    "ConformalNearestNeighboursClassifier",
    "ConformalSupportVectorMachine",
    "ConformalPredictionSet",
    "MultiLevelPredictionSet",
    "ConformalMondrianTreeClassifier",
    "ConformalMondrianForestClassifier",
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
        if hasattr(epsilon, "__iter__"):
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
    r"""Conformal $k$-nearest-neighbours classifier ([ALRW2 §2.3]).

    The nonconformity measure of a labelled example is the ratio

    $$
    A\bigl(x, y\bigr) =
        \frac{\text{distance to the } k \text{ nearest same-class objects}}
             {\text{distance to the } k \text{ nearest different-class objects}},
    $$

    aggregated by mean or median. An example is *nonconforming* (large score)
    when it sits far from its own class but close to another — exactly the
    1-NN measure of [ALRW2 §2.3] generalised to $k$ neighbours. Under
    exchangeability the prediction sets are valid at every $\epsilon$.

    >>> cp = ConformalNearestNeighboursClassifier(k=1, label_space=[-1, 1], rnd_state=1337, epsilon=0.1)
    >>> Gamma, p_values = cp.predict(3, return_p_values=True)
    >>> Gamma  # predict both labels, as this is the first
    array([-1,  1])
    >>> tuple(round(p_values[i], 4) for i in (-1, 1))
    (0.8781, 0.8781)

    >>> cp.learn_one(np.int64(3), 1)

    >>> Gamma, p_values = cp.predict(-2, return_p_values=True)
    >>> Gamma  # predict both labels, as this is the first
    array([-1,  1])
    >>> tuple(round(p_values[i], 4) for i in (-1, 1))
    (0.1855, 0.1855)
    """

    _SAVE_PARAMS: tuple = (
        "k",
        "label_space",
        "distance",
        "distance_func",
        "aggregation",
        "verbose",
        "rnd_state",
        "n_jobs",
        "epsilon",
    )
    _SAVE_STATE: tuple = (
        "X",
        "y",
        "D",
        "_label_indices",
        "_label_space_fixed",
        "label_space",
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
        """Create a conformal nearest-neighbours classifier.

        Parameters
        ----------
        k : int, default 1
            Number of nearest neighbours used in the nonconformity ratio.
        label_space : array-like or None, default None
            The set of possible labels. If None, it is inferred (and grows)
            from the data seen so far.
        distance : str, default "euclidean"
            Distance metric passed to ``scipy.spatial.distance``.
        distance_func : callable, optional
            Custom distance ``(X, y=None) -> ndarray``. If given, ``distance``
            is ignored.
        aggregation : {"mean", "median"}, default "mean"
            How to aggregate the k nearest same/different-class distances.
        verbose : int, default 0
            Verbosity level.
        rnd_state : int, np.random.Generator, or None, default None
            Seed or Generator for the smoothing-variable generator.
        n_jobs : int or None, default None
            Number of parallel jobs for per-label p-value computation in
            :meth:`predict`.
        epsilon : float, default 0.1
            Default significance level.
        """
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
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
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
        """Batch-learn an initial training set.

        Stores the objects/labels, precomputes the pairwise distance matrix,
        and indexes examples by label. Updates the inferred label space unless
        a fixed ``label_space`` was supplied.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training objects.
        y : ndarray of shape (n,)
            Training labels.
        """
        if X.shape[0] > 0:
            self.X = X
            self.y = y
            self.D = self.distance_func(X)
            self._label_indices = self._build_label_indices(y)
            if self._label_space_fixed:
                unknown = set(np.unique(y)) - set(self.label_space)
                if unknown:
                    raise ValueError(
                        f"Labels {sorted(unknown)} not in declared label_space {self.label_space.tolist()}"
                    )
            elif self.label_space is None:
                self.label_space = np.unique(y)
            else:
                self.label_space = np.sort(np.unique(np.concatenate([self.label_space, np.unique(y)])))

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

    def learn_one(
        self, x: NDArray[np.floating[Any]], y: Any, precomputed: NDArray[np.floating[Any]] | None = None
    ) -> None:
        """Update the classifier with a single new example.

        Appends ``(x, y)`` to the stored data, extends the distance matrix and
        label index, and grows the label space if needed.

        Parameters
        ----------
        x : ndarray of shape (d,)
            New object.
        y : hashable
            Observed label.
        precomputed : ndarray or None
            A pre-extended distance matrix (e.g. from a previous
            :meth:`predict` call) to avoid recomputing distances.
        """
        new_index = 0 if self.X is None else self.X.shape[0]

        # Enforce label-space policy
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(f"Label {y} not in declared label_space {self.label_space.tolist()}")
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

    def compute_p_value(
        self, x: NDArray[np.floating[Any]], y: Any, return_update: bool = False
    ) -> float | tuple[float, NDArray[np.floating[Any]] | None]:
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
            same_label_distances, different_label_distances = self._find_nearest_distances(
                D, label_indices=label_indices
            )
            Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)
            p_value = self._compute_p_value(Alpha, tau, "nonconformity")
        else:
            D = None
            p_value = self._compute_p_value(np.array([np.inf]), tau, "nonconformity")

        if return_update:
            return p_value, D
        return p_value

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        return_p_values: bool = False,
        return_update: bool = False,
        verbose: int = 0,
    ) -> ConformalPredictionSet | MultiLevelPredictionSet:
        """Compute the conformal prediction set for object ``x``.

        For every candidate label the nonconformity ratio is evaluated as if
        ``x`` carried that label, and the label is kept when its conformal
        p-value exceeds ``epsilon``.

        Parameters
        ----------
        x : ndarray of shape (d,)
            Test object.
        epsilon : float, array-like, or None
            Significance level(s). If None, uses ``self.epsilon``. An iterable
            yields a :class:`MultiLevelPredictionSet`.
        return_p_values : bool, default False
            If True, also return the ``{label: p_value}`` dict.
        return_update : bool, default False
            If True, also return the extended distance matrix to reuse in a
            subsequent :meth:`learn_one`.
        verbose : int, default 0
            Verbosity level.

        Returns
        -------
        ConformalPredictionSet or MultiLevelPredictionSet, optionally followed
        by the p-value dict and/or the updated distance matrix.
        """
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.label_space is None:
            Gamma = ConformalPredictionSet(np.array([]), epsilon if not hasattr(epsilon, "__iter__") else epsilon[0])
            if hasattr(epsilon, "__iter__"):
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

    def __init__(
        self,
        learner,
        label_space=None,
        epsilon=default_epsilon,
        verbose=0,
        rnd_state=None,
        n_jobs=None,
        warm_start="auto",
    ):
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
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
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
                f"Wrapped estimator '{estimator_name}' is supported with caution; results can be unstable or slow.",
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
                raise ValueError(f"Label {y} not in declared label_space {self.label_space.tolist()}")
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
                        f"Labels {sorted(unknown)} not in declared label_space {self.label_space.tolist()}"
                    )
            elif self.label_space is None:
                self.label_space = np.unique(y)
            else:
                self.label_space = np.sort(np.unique(np.concatenate([self.label_space, np.unique(y)])))

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

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        return_p_values: bool = False,
        return_update: bool = False,
        verbose: int = 0,
    ) -> ConformalPredictionSet | MultiLevelPredictionSet:
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if self.label_space is None or self.X is None or self.y.shape[0] == 0:
            if self.label_space is None:
                Gamma = ConformalPredictionSet(
                    np.array([]), epsilon if not hasattr(epsilon, "__iter__") else epsilon[0]
                )
                if hasattr(epsilon, "__iter__"):
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
                learner_copy = (
                    copy.deepcopy(self._base_learner) if self._base_learner is not None else copy.deepcopy(self.learner)
                )
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
    rnd_state : int, np.random.Generator, or None
        Random seed or Generator.

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
        "kernel",
        "C",
        "nonconformity",
        "label_space",
        "sigma",
        "degree",
        "coef0",
        "smo_tol",
        "smo_max_iter",
        "epsilon",
        "rnd_state",
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
        if isinstance(rnd_state, np.random.Generator):
            self.rnd_gen = rnd_state
        else:
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
                raise ValueError(f"Labels {sorted(unknown)} not in declared label_space {self.label_space.tolist()}")
        elif self.label_space is None:
            self.label_space = np.unique(y)
        else:
            self.label_space = np.sort(np.unique(np.concatenate([self.label_space, np.unique(y)])))
        self.X = X.copy()
        self.y = y.copy().astype(float)
        self.K = self._compute_gram(X)

    def learn_one(self, x: NDArray[np.floating[Any]], y: Any) -> None:
        """Learn a new example, updating stored data and Gram matrix."""
        x = np.atleast_1d(x).ravel()

        # Enforce label-space policy
        if self._label_space_fixed:
            if y not in self.label_space:
                raise ValueError(f"Label {y} not in declared label_space {self.label_space.tolist()}")
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

    def predict(
        self,
        x: NDArray[np.floating[Any]],
        epsilon: float | NDArray[np.floating[Any]] | None = None,
        return_p_values: bool = False,
    ) -> ConformalPredictionSet | MultiLevelPredictionSet:
        r"""Compute the conformal prediction set for object ``x``.

        For each candidate label the training set is augmented with
        ``(x, label)``, one-vs-rest binarised, and the SVM dual is solved on the
        shared (label-independent) Gram matrix. The configured nonconformity
        measure (signed margin or $\alpha_i$) then yields a conformal p-value
        per label; labels with p-value above ``epsilon`` form the set.

        Parameters
        ----------
        x : ndarray of shape (d,)
            Test object.
        epsilon : float, array-like, or None
            Significance level(s). If None, uses ``self.epsilon``.
        return_p_values : bool, default False
            If True, also return the ``{label: p_value}`` dict.

        Returns
        -------
        ConformalPredictionSet or MultiLevelPredictionSet, optionally with the
        p-value dict.
        """
        if epsilon is None:
            epsilon = self.epsilon

        x = np.atleast_1d(x).ravel()
        tau = self.rnd_gen.uniform()
        p_values = {}

        if self.label_space is None or self.X is None or self.y.shape[0] == 0:
            # No training data — predict all labels (or empty if no label_space)
            if self.label_space is None:
                Gamma = ConformalPredictionSet(
                    np.array([]), epsilon if not hasattr(epsilon, "__iter__") else epsilon[0]
                )
                if hasattr(epsilon, "__iter__"):
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
                f = K_aug @ (alpha * y_binary) + b  # decision function
                ncm = -(y_binary * f)  # large => nonconforming
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
            if (alpha[k] < C and y[k] > 0) or (alpha[k] > 0 and y[k] < 0):
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
            if (alpha[k] < C and y[k] < 0) or (alpha[k] > 0 and y[k] > 0):
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


# ---------------------------------------------------------------------------
# Mondrian tree / forest conformal classifiers
# ---------------------------------------------------------------------------


class ConformalMondrianTreeClassifier(ConformalClassifier):
    """Conformal predictor using Mondrian trees.

    A full (transductive) conformal predictor. At each prediction step the tree
    is rebuilt from the augmented bag {train ∪ (x_test, y_cand)}, guaranteeing
    exact validity under exchangeability (ALRW2 §2.2.9).

    The nonconformity measure is (Leaf-Local Laplace Smoothing):

        α_i = 1 - (C_{leaf(x_i), y_i} + λ) / (N_leaf(x_i) + K * λ)

    where C_{leaf, k} is the count of class k in the leaf, N_leaf is the
    number of training points in the leaf, K is the number of classes, and
    λ (`lambda_`) is the Laplace smoothing parameter (default 1.0). This
    eliminates the discrete ties of raw fractions while confining the
    transductive delta strictly to leaf_star (O(1) blast radius).

    Parameters
    ----------
    lifetime : float or {'sqrt_n', 'density'}
        Depth budget L for the Mondrian tree. Splits occur at times sampled
        from Exp(sum of feature ranges). A split is created only if its
        cumulative time is below L; otherwise the node becomes a leaf.

        Can also be a string for unsupervised automatic tuning (only X is
        used; labels are never accessed):

        - ``'sqrt_n'``: descend the master tree toward the test point and
          halt when the leaf count drops to ≤ √n. Coarsens the partition in
          sparse regions.
        - ``'density'``: sweep all split events of a master tree and choose
          the τ that maximises the histogram log-likelihood
          Σ_leaf n_leaf · log(n_leaf / V_leaf). Adapts to feature density.

        **Effect on prediction quality:**
        - Too small (e.g. 0.01): almost no splits → one big leaf → NCM is
          just the global class frequency → very weak discrimination.
        - Too large (e.g. np.inf): splits until every leaf is a singleton →
          NCM is 0 for every point → p-values are all equal (τ) → prediction
          sets are either full or empty, with no discrimination.
        - Well-chosen (default 1.0): leaves contain a handful of points,
          NCMs vary meaningfully, prediction sets are informative.

        **Scaling guidance:** `lifetime` is in the same units as the sum of
        feature ranges. For standardised data (zero mean, unit variance)
        a value of 1–5 is a reasonable starting range. For raw (unstandardised)
        features with large ranges (e.g. pixel values 0–255) you may need
        lifetime in the hundreds. When in doubt, standardise your features
        first and use the default.
    lambda_ : float
        Laplace smoothing parameter λ. Controls the strength of the uniform
        prior added to each leaf's class counts. Default 1.0 (standard Laplace).
        Larger values push NCMs towards 1 - 1/K (uniform), reducing the
        influence of small leaves. Set to 0.0 to recover raw fraction NCMs.
    label_space : array-like, optional
        Set of possible labels. If None, inferred from training data.
    epsilon : float
        Default significance level for predictions.
    rnd_state : int, optional
        Seed for reproducible tree construction.
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

    Attributes
    ----------
    X : ndarray of shape (n, d)
        Training feature array.
    y : ndarray of shape (n,)
        Training labels.
    label_space : ndarray
        Sorted unique labels.
    label_to_idx : dict
        Mapping from label to integer index 0..K-1.
    rnd_gen : numpy.random.Generator
        Random generator instance.
    """

    _SAVE_PARAMS: tuple = (
        "lifetime", "lambda_", "label_space", "epsilon", "rnd_state",
        "verbose", "max_depth", "feature_weights",
    )
    _SAVE_STATE: tuple = ("X", "y", "label_to_idx", "label_space")

    def __init__(
        self,
        lifetime: float | str = 1.0,
        lambda_: float = 1.0,
        label_space: NDArray | None = None,
        epsilon: float = 0.1,
        rnd_state: int | None = None,
        verbose: int = 0,
        max_depth: int | None = None,
        feature_weights: str | NDArray | None = None,
    ):
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
        self.lambda_ = lambda_
        self.label_space = label_space
        self.rnd_state = rnd_state
        self.verbose = verbose
        self.max_depth = max_depth

        self.X = None
        self.y = None
        self.label_to_idx = None
        self.rnd_gen = np.random.default_rng(rnd_state)
        self._last_tree = None

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Batch training phase.

        Args:
            X: Training features (n, d)
            y: Training labels (n,) — any hashable values
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] == 0:
            raise ValueError("Training set cannot be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")

        self.X = X.copy()
        self.y = y.copy()

        # Infer or validate label space
        if self.label_space is None:
            self.label_space = np.unique(y)
        else:
            self.label_space = np.asarray(self.label_space)

        # Build label-to-index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_space)}

        if self.verbose >= 1:
            print(
                f"[MondrianTree] Initialized with {X.shape[0]} training points, "
                f"{X.shape[1]} dimensions, {len(self.label_space)} classes"
            )

    def learn_one(self, x: NDArray, y: Any, precomputed: dict | None = None) -> None:
        """Online learning step.

        Args:
            x: New feature vector (d,) or (1, d)
            y: New label (any hashable value in label_space)
            precomputed: Optional dict from previous predict (not used here)
        """
        x = np.asarray(x).ravel()

        if self.X is None:
            raise ValueError("Must call learn_initial_training_set first")

        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")

        if y not in self.label_to_idx:
            # New label encountered
            new_idx = len(self.label_space)
            self.label_space = np.append(self.label_space, y)
            self.label_to_idx[y] = new_idx

        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, y)


    @property
    def summary(self) -> dict:
        """Summary statistics of the classifier.

        Returns
        -------
        dict
            Contains: n_points, n_features, n_classes, label_space, lifetime, epsilon.
            If a tree has been built (after predict() or compute_p_value()), also includes:
            n_nodes, n_leaves, n_branches, height, total_observed_weight.
        """
        stats = {
            "n_points": len(self.y) if self.y is not None else 0,
            "n_features": self.X.shape[1] if self.X is not None else 0,
            "n_classes": len(self.label_space) if self.label_space is not None else 0,
            "label_space": list(self.label_space) if self.label_space is not None else [],
            "lifetime": self.lifetime,
            "epsilon": self.epsilon,
        }
        if self._last_tree is not None:
            tree_stats = _tree_struct_stats(self._last_tree)
            stats.update(tree_stats)
            stats["total_observed_weight"] = len(self.y)
        return stats

    def to_dataframe(self):
        """Export tree structure as a pandas DataFrame.

        Returns a DataFrame with one row per node, including node ID, parent ID,
        depth, split information, and leaf-specific data (class counts for classifier).

        Raises
        ------
        RuntimeError
            If no tree has been built yet (call predict() first).

        Returns
        -------
        pd.DataFrame
            Tree structure with columns: node_id, parent_id, is_leaf, depth,
            split_dim, split_loc, split_time, parent_time, bbox_lower, bbox_upper,
            n_points, and counts (dict, classifier only).
        """
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )
        import pandas as pd

        rows = []
        node_id_map = {}  # Map string node_id to integer
        next_id = 0

        for node, depth, parent_id_str, node_id_str in _iter_nodes(self._last_tree):
            # Convert string node_id to integer
            if node_id_str not in node_id_map:
                node_id_map[node_id_str] = next_id
                next_id += 1
            node_id_int = node_id_map[node_id_str]

            # Convert parent_id_str to integer
            if parent_id_str is None:
                parent_id_int = None
            else:
                if parent_id_str not in node_id_map:
                    node_id_map[parent_id_str] = next_id
                    next_id += 1
                parent_id_int = node_id_map[parent_id_str]

            bbox_lower = node.lower_bounds.tolist() if len(node.lower_bounds) > 0 else []
            bbox_upper = node.upper_bounds.tolist() if len(node.upper_bounds) > 0 else []

            row = {
                "node_id": node_id_int,
                "parent_id": parent_id_int,
                "is_leaf": node.is_leaf(),
                "depth": depth,
                "split_dim": node.split_dim,
                "split_loc": float(node.split_loc) if not node.is_leaf() else None,
                "split_time": float(node.split_time) if not node.is_leaf() else None,
                "parent_time": float(node.parent_time),
                "bbox_lower": bbox_lower,
                "bbox_upper": bbox_upper,
                "n_points": node.n_points(),
            }
            if node.is_leaf() and node.counts is not None:
                counts_dict = {self.label_space[i]: int(node.counts[i]) for i in range(len(self.label_space))}
                row["counts"] = counts_dict
            rows.append(row)

        return pd.DataFrame(rows)

    def debug_one(self, x):
        """Trace the path of a single example through a representative tree.

        Builds a fresh representative tree (using a deterministic RNG seed)
        and traces the feature vector through it, showing split decisions
        and leaf information.

        Parameters
        ----------
        x : array-like, shape (d,)
            Feature vector to trace.

        Returns
        -------
        str
            Human-readable string showing the decision path and final leaf stats.

        Notes
        -----
        This uses a representative tree sampled from the same distribution as
        the tree in the last predict() call, NOT the exact tree itself.
        This method does not affect the random state of the main RNG.
        """
        x = np.asarray(x, dtype=float).ravel()

        if self.X is None:
            raise RuntimeError("Must call learn_initial_training_set first")

        # Build a representative tree using a fixed seed
        rng_debug = np.random.default_rng(self.rnd_state if self.rnd_state is not None else 0)
        X_aug = np.vstack([self.X, x])
        indices_all = np.arange(len(X_aug))
        fw_debug = _resolve_feature_weights(X_aug, self.feature_weights)
        lt_debug = _resolve_lifetime(X_aug, X_aug[-1], self.lifetime, rng_debug, fw_debug)
        tree_repr = _sample_mondrian_tree(
            rng_debug,
            X_aug,
            indices_all,
            parent_time=0.0,
            lifetime=lt_debug,
            verbose=0,
            max_depth=self.max_depth,
            feature_weights=fw_debug,
        )

        # Trace the path
        path_lines = []
        current = tree_repr
        depth = 0

        while not current.is_leaf():
            split_str = f"  {'  ' * depth}x[{current.split_dim}] <= {current.split_loc:.6f}"
            if x[current.split_dim] <= current.split_loc:
                path_lines.append(split_str + "  [TRUE → left]")
                current = current.left
            else:
                path_lines.append(split_str + "  [FALSE → right]")
                current = current.right
            depth += 1

        # Leaf info
        leaf_str = f"  {'  ' * depth}LEAF: n_points={current.n_points()}"
        if current.counts is not None:
            counts_str = {self.label_space[i]: int(current.counts[i]) for i in range(len(self.label_space))}
            leaf_str += f", counts={counts_str}"
        path_lines.append(leaf_str)

        return "\n".join(path_lines)

    def draw(self, ax=None, max_depth=None, backend="auto", **kwargs):
        """Draw a node-link tree diagram of the last cached tree.

        **Node types in the diagram**

        - *Internal nodes* (orange): split decisions — ``x[i] ≤ threshold``.
          Left branch follows ``≤``, right branch follows ``>``.
        - *Leaf nodes* (blue): ``→ class C  (P%)`` — the majority class C among
          training points in this cell, with purity P (fraction belonging to C).
          The second line ``n=k  [c0:n0  c1:n1 ...]`` shows the total count and
          per-class breakdown (zero-count classes omitted).
        - *"empty"* leaf: the Mondrian process created this rectangular partition
          cell, but no training points landed in it.  This is normal for small
          datasets or high lifetime values; such leaves contribute a uniform
          conformal score that weakly widens prediction sets.

        .. note::
           The tree shown is the *augmented* tree built on ``{train ∪ x_test}``
           during the most recent ``predict()`` or ``compute_p_value()`` call.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes.  If ``None`` and the graphviz backend is active, a
            :class:`graphviz.Digraph` is returned instead (renders as inline SVG
            in Jupyter notebooks).
        max_depth : int or None
            Maximum depth to display (display-only; does not affect the model).
            Nodes below this depth are shown as collapsed leaf boxes.
        backend : {'auto', 'graphviz', 'matplotlib'}
            Rendering backend.  ``'auto'`` (default) uses graphviz when available
            and falls back to matplotlib.  ``'graphviz'`` raises
            :class:`ImportError` with an install hint if the package is missing.
            ``'matplotlib'`` forces the built-in renderer.
            Install graphviz with ``pip install online-cp[viz]``.
        **kwargs
            Reserved for future style options.

        Returns
        -------
        graphviz.Digraph or matplotlib.axes.Axes
            A :class:`graphviz.Digraph` when graphviz is used and *ax* is
            ``None``; otherwise a :class:`~matplotlib.axes.Axes`.

        Raises
        ------
        RuntimeError
            If no tree has been built yet (call ``predict()`` first).
        """
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )

        def label_fn(node, depth, *, collapsed=False):
            n = node.n_points()
            if node.is_leaf() or collapsed:
                if n == 0:
                    return "empty"
                if node.counts is not None:
                    maj_idx = int(np.argmax(node.counts))
                    maj_label = self.label_space[maj_idx]
                    counts_str = "  ".join(
                        f"{self.label_space[i]}:{int(node.counts[i])}"
                        for i in range(len(self.label_space))
                        if node.counts[i] > 0
                    )
                    purity = int(node.counts[maj_idx]) / n
                    return f"\u2192 class {maj_label}  ({purity:.0%})\nn={n}  [{counts_str}]"
                return f"n={n}"
            return f"x[{node.split_dim}] ≤\n{node.split_loc:.3g}"

        return _render_tree(
            self._last_tree, ax, label_fn, max_depth, backend,
            title="Mondrian Tree (classifier)",
        )

    def draw_partition(self, ax=None, scatter=True, **kwargs):
        """Draw the 2-D Mondrian box-partition of the last cached tree.

        Each leaf is drawn as a rectangle coloured by its majority class.
        Training points can be overlaid as a scatter plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        scatter : bool
            If True (default), overlay training points coloured by label.
        **kwargs
            Currently unused; reserved for future style options.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If no tree has been built yet.
        ValueError
            If the number of features is not exactly 2.
        """
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )
        if self.X.shape[1] != 2:
            raise ValueError(
                f"draw_partition() requires exactly 2 features, got {self.X.shape[1]}. "
                "The leaf bounding boxes only tile feature space exactly at d=2."
            )
        from matplotlib import colormaps
        from matplotlib.lines import Line2D

        ax = _get_ax(ax)

        K = len(self.label_space)
        cmap = colormaps["tab10"].resampled(K)
        label_to_color = {lbl: cmap(i) for i, lbl in enumerate(self.label_space)}

        def leaf_color_fn(leaf):
            if leaf.counts is None or leaf.counts.sum() == 0:
                return "#cccccc"
            majority_idx = int(np.argmax(leaf.counts))
            return label_to_color[self.label_space[majority_idx]]

        scatter_X = self.X if scatter else None
        scatter_y = [list(self.label_space).index(yi) for yi in self.y] if scatter else None
        _draw_partition(
            self._last_tree, self.X, ax, leaf_color_fn,
            scatter_X=scatter_X, scatter_y=scatter_y, scatter_cmap="tab10",
        )

        # Legend
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lbl],
                   markersize=8, label=str(lbl))
            for lbl in self.label_space
        ]
        ax.legend(handles=handles, title="Class", fontsize=7, title_fontsize=7)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title("Mondrian Partition (classifier)", fontsize=9)
        return ax


    def predict(
        self,
        x: NDArray,
        epsilon: float | NDArray | None = None,
        return_p_values: bool = False,
        return_update: bool = False,
    ) -> ConformalPredictionSet | MultiLevelPredictionSet | tuple:
        """Make conformal prediction.

        Rebuilds the tree from the augmented bag [X, y, x, ?] where ? ranges
        over label_space. For each candidate label, computes nonconformity
        scores and p-values.

        Args:
            x: Test feature vector (d,) or (1, d)
            epsilon: Significance level(s). If None, uses self.epsilon.
                     Can be scalar or array-like.
            return_p_values: If True, also return dict of p-values keyed by label.
            return_update: If True, also return dict with tree info for efficiency.

        Returns:
            ConformalPredictionSet: If epsilon is scalar and neither flag is True
            MultiLevelPredictionSet: If epsilon is array-like
            Tuple: If return_p_values or return_update flag is set
        """
        x = np.asarray(x).ravel()

        if self.X is None:
            raise ValueError("Must call learn_initial_training_set first")

        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")

        if epsilon is None:
            epsilon = self.epsilon

        # Form augmented feature matrix (same for all candidates)
        X_aug = np.vstack([self.X, x])
        n = self.X.shape[0]
        n_total = n + 1
        K = len(self.label_space)

        # Resolve adaptive lifetime and feature weights (unsupervised; no labels used)
        fw_arr  = _resolve_feature_weights(X_aug, self.feature_weights)
        lt_val  = _resolve_lifetime(X_aug, x, self.lifetime, self.rnd_gen, fw_arr)

        # Build tree from augmented features
        indices_all = np.arange(n_total)
        tree = _sample_mondrian_tree(
            self.rnd_gen,
            X_aug,
            indices_all,
            parent_time=0.0,
            lifetime=lt_val,
            verbose=self.verbose,
            max_depth=self.max_depth,
            feature_weights=fw_arr,
        )

        # Find leaf containing test point (index n in X_aug)
        leaf_star = _find_leaf(tree, X_aug[n])

        # Base state: assign training-only counts (test point excluded from counts)
        _assign_counts(tree, self.y, self.label_to_idx, K, n_train=n)

        n_star = leaf_star.n_points()  # leaf size including test point's structural slot
        n_star_train = n_star - 1  # training-only count in leaf_star
        c = leaf_star.counts.copy()  # training-only class counts per class, shape (K,)

        # Leaf-Local Laplace Smoothing
        LAMBDA = self.lambda_
        base_ncm = np.empty(n)
        for leaf in _collect_leaves(tree):
            n_leaf = leaf.n_points()
            has_test = int(n) in leaf.indices
            n_train_leaf = n_leaf - 1 if has_test else n_leaf
            denom = n_train_leaf + K * LAMBDA
            for idx in leaf.indices:
                if idx >= n:
                    continue  # skip the test point's structural slot
                yi = self.label_to_idx[self.y[idx]]
                base_ncm[idx] = 1.0 - (leaf.counts[yi] + LAMBDA) / denom

        # ncm_test[k]: test point's NCM under candidate label k (Laplace)
        denom_test = n_star_train + 1.0 + K * LAMBDA
        ncm_test = 1.0 - (c + 1.0 + LAMBDA) / denom_test  # shape (K,)

        # Vectorised (K, n) broadcast: compare all training NCMs against all ncm_test[k]
        _TOL = 1e-9
        diff = base_ncm[np.newaxis, :] - ncm_test[:, np.newaxis]  # (K, n)
        gt = np.sum(diff > _TOL, axis=1).astype(np.int64)  # (K,)
        eq = np.sum(np.abs(diff) <= _TOL, axis=1).astype(np.int64)  # (K,)

        # Delta correction: O(K²) — only leaf_star training points are affected.
        # Under candidate k, both numerator (if j==k) and denominator change.
        # All c[j] points of class j have identical base and actual NCMs.
        denom_base_star = n_star_train + K * LAMBDA
        for j in range(K):
            if c[j] == 0:
                continue
            base_j = 1.0 - (c[j] + LAMBDA) / denom_base_star
            count_j = int(c[j])
            for k in range(K):
                actual_j = 1.0 - (c[j] + LAMBDA + (1.0 if j == k else 0.0)) / denom_test
                ncm_test_k = ncm_test[k]
                # Classify base_j vs ncm_test_k
                if base_j > ncm_test_k + _TOL:
                    before = 0  # GT
                elif abs(base_j - ncm_test_k) <= _TOL:
                    before = 1  # EQ
                else:
                    before = 2  # LT
                # Classify actual_j vs ncm_test_k
                if actual_j > ncm_test_k + _TOL:
                    after = 0
                elif abs(actual_j - ncm_test_k) <= _TOL:
                    after = 1
                else:
                    after = 2
                if before == after:
                    continue
                if before == 0 and after == 1:
                    gt[k] -= count_j
                    eq[k] += count_j
                elif before == 0 and after == 2:
                    gt[k] -= count_j
                elif before == 1 and after == 2:
                    eq[k] -= count_j
                elif before == 1 and after == 0:
                    eq[k] -= count_j
                    gt[k] += count_j
                elif before == 2 and after == 1:
                    eq[k] += count_j
                elif before == 2 and after == 0:
                    gt[k] += count_j

        eq += 1  # test point always EQ to itself
        tau = self.rnd_gen.uniform(0, 1)
        p_values_arr = (gt + tau * eq) / n_total
        p_values = {label: float(p_values_arr[ki]) for ki, label in enumerate(self.label_space)}

        # Construct prediction set(s)
        result = self._compute_Gamma(p_values, epsilon)

        # Cache the tree for inspection utilities
        self._last_tree = tree

        # Handle return flags
        if return_update and return_p_values:
            return result, p_values, {"tree": tree, "leaf_star": leaf_star}
        elif return_p_values:
            return result, p_values
        elif return_update:
            return result, {"tree": tree, "leaf_star": leaf_star}
        else:
            return result

    def compute_p_value(self, x: NDArray, y: Any, return_update: bool = False) -> float | tuple[float, dict]:
        """Compute p-value for a single candidate label.

        Args:
            x: Test feature vector (d,)
            y: Candidate label
            return_update: If True, also return dict with tree info

        Returns:
            P-value (float) or (p-value, dict) if return_update=True
        """
        x = np.asarray(x).ravel()

        if y not in self.label_to_idx:
            raise ValueError(f"Label {y} not in label_space")

        # Build augmented dataset
        X_aug = np.vstack([self.X, x])
        n = self.X.shape[0]
        n_total = n + 1
        K = len(self.label_space)
        label_idx = self.label_to_idx[y]

        # Resolve adaptive lifetime and feature weights (unsupervised; no labels used)
        fw_arr  = _resolve_feature_weights(X_aug, self.feature_weights)
        lt_val  = _resolve_lifetime(X_aug, X_aug[n], self.lifetime, self.rnd_gen, fw_arr)

        # Build tree
        indices_all = np.arange(n_total)
        tree = _sample_mondrian_tree(
            self.rnd_gen,
            X_aug,
            indices_all,
            parent_time=0.0,
            lifetime=lt_val,
            verbose=self.verbose,
            max_depth=self.max_depth,
            feature_weights=fw_arr,
        )

        # Find leaf and assign training-only counts
        leaf_star = _find_leaf(tree, X_aug[n])
        _assign_counts(tree, self.y, self.label_to_idx, K, n_train=n)

        n_star = leaf_star.n_points()
        n_star_train = n_star - 1
        c = leaf_star.counts.copy()

        # Leaf-Local Laplace Smoothing
        LAMBDA = self.lambda_
        denom_test = n_star_train + 1.0 + K * LAMBDA
        ncm_test_k = 1.0 - (float(c[label_idx]) + 1.0 + LAMBDA) / denom_test

        # Build base_ncm for all training points (Laplace)
        base_ncm = np.empty(n)
        for leaf in _collect_leaves(tree):
            n_leaf = leaf.n_points()
            has_test = int(n) in leaf.indices
            n_train_leaf = n_leaf - 1 if has_test else n_leaf
            denom = n_train_leaf + K * LAMBDA
            for idx in leaf.indices:
                if idx >= n:
                    continue
                yi = self.label_to_idx[self.y[idx]]
                base_ncm[idx] = 1.0 - (leaf.counts[yi] + LAMBDA) / denom

        _TOL = 1e-9
        diff = base_ncm - ncm_test_k
        gt = int(np.sum(diff > _TOL))
        eq = int(np.sum(np.abs(diff) <= _TOL))

        # Delta correction: O(K) — for each class j in leaf_star
        denom_base_star = n_star_train + K * LAMBDA
        for j in range(K):
            if c[j] == 0:
                continue
            base_j = 1.0 - (c[j] + LAMBDA) / denom_base_star
            actual_j = 1.0 - (c[j] + LAMBDA + (1.0 if j == label_idx else 0.0)) / denom_test
            if base_j > ncm_test_k + _TOL:
                before = 0
            elif abs(base_j - ncm_test_k) <= _TOL:
                before = 1
            else:
                before = 2
            if actual_j > ncm_test_k + _TOL:
                after = 0
            elif abs(actual_j - ncm_test_k) <= _TOL:
                after = 1
            else:
                after = 2
            if before == after:
                continue
            count_j = int(c[j])
            if before == 0 and after == 1:
                gt -= count_j
                eq += count_j
            elif before == 0 and after == 2:
                gt -= count_j
            elif before == 1 and after == 2:
                eq -= count_j
            elif before == 1 and after == 0:
                eq -= count_j
                gt += count_j
            elif before == 2 and after == 1:
                eq += count_j
            elif before == 2 and after == 0:
                gt += count_j

        eq += 1  # test point always EQ to itself
        tau = self.rnd_gen.uniform(0, 1)
        p_val = float((gt + tau * eq) / n_total)

        # Cache the tree for inspection utilities
        self._last_tree = tree

        if return_update:
            return p_val, {"tree": tree, "leaf_star": leaf_star}
        else:
            return p_val


class ConformalMondrianForestClassifier(ConformalClassifier):
    """Conformal predictor using Mondrian forests (ensembles of Mondrian trees).

    A full (transductive) conformal predictor that builds T Mondrian trees per
    prediction step from the same augmented bag but with different random seeds.
    The ensemble averages per-tree leaf posteriors, reducing variance while
    maintaining exact validity (average of invariant predictors is invariant).

    The nonconformity measure is (Leaf-Local Laplace, averaged):

        α_i = 1 - (1/T) Σ_t (C_{leaf_t(x_i), y_i} + λ) / (N_leaf_t(x_i) + K * λ)

    where the sum averages the Laplace-smoothed class posterior for label y_i
    across all T trees. This smooths predictions, reduces discrete ties, and
    produces smaller (more efficient) prediction sets than a single tree.

    Parameters
    ----------
    n_trees : int
        Number of Mondrian trees in the forest. Default: 10. Larger values
        reduce variance but increase computation per predict.
    lifetime : float
        Depth budget L for each tree (see `ConformalMondrianTreeClassifier`).
        Default: 1.0.
    lambda_ : float
        Laplace smoothing parameter λ (see `ConformalMondrianTreeClassifier`).
        Default: 1.0.
    label_space : array-like, optional
        Set of possible labels. If None, inferred from training data.
    epsilon : float
        Default significance level for predictions.
    rnd_state : int, optional
        Seed for reproducible forest construction.
    verbose : int
        Verbosity level (0 = silent).

    Attributes
    ----------
    X : ndarray of shape (n, d)
        Training feature array.
    y : ndarray of shape (n,)
        Training labels.
    label_space : ndarray
        Sorted unique labels.
    label_to_idx : dict
        Mapping from label to integer index 0..K-1.
    rnd_gen : numpy.random.Generator
        Root random generator instance.
    """

    _SAVE_PARAMS: tuple = (
        "n_trees", "lifetime", "lambda_", "label_space", "epsilon",
        "rnd_state", "verbose", "n_jobs", "max_depth",
    )
    _SAVE_STATE: tuple = ("X", "y", "label_to_idx", "label_space")

    def __init__(
        self,
        n_trees: int = 10,
        lifetime: float = 1.0,
        lambda_: float = 1.0,
        label_space: NDArray | None = None,
        epsilon: float = 0.1,
        rnd_state: int | None = None,
        verbose: int = 0,
        n_jobs: int = 1,
        max_depth: int | None = None,
    ):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        super().__init__(epsilon=epsilon)
        self.n_trees = n_trees
        self.lifetime = lifetime
        self.lambda_ = lambda_
        self.label_space = label_space
        self.rnd_state = rnd_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.max_depth = max_depth

        self.X = None
        self.y = None
        self.label_to_idx = None
        self.rnd_gen = np.random.default_rng(rnd_state)
        self._last_tree = None
        self._last_seeds = None
        self._last_x = None

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Batch training phase.

        Args:
            X: Training features (n, d)
            y: Training labels (n,) — any hashable values
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] == 0:
            raise ValueError("Training set cannot be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")

        self.X = X.copy()
        self.y = y.copy()

        # Infer or validate label space
        if self.label_space is None:
            self.label_space = np.unique(y)
        else:
            self.label_space = np.asarray(self.label_space)

        # Build label-to-index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_space)}

        if self.verbose >= 1:
            print(
                f"[MondrianForest] Initialized with {X.shape[0]} training points, "
                f"{X.shape[1]} dimensions, {len(self.label_space)} classes, {self.n_trees} trees"
            )

    def learn_one(self, x: NDArray, y: Any, precomputed: dict | None = None) -> None:
        """Online learning step.

        Args:
            x: New feature vector (d,) or (1, d)
            y: New label (any hashable value in label_space)
            precomputed: Optional dict from previous predict (not used here)
        """
        x = np.asarray(x).ravel()

        if self.X is None:
            raise ValueError("Must call learn_initial_training_set first")

        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")

        if y not in self.label_to_idx:
            # New label encountered
            new_idx = len(self.label_space)
            self.label_space = np.append(self.label_space, y)
            self.label_to_idx[y] = new_idx

        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, y)

    # ------------------------------------------------------------------
    # Forest indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of trees in the forest."""
        return self.n_trees

    def __iter__(self):
        """Iterate over individual tree views (calls predict first if needed)."""
        for i in range(self.n_trees):
            yield self[i]

    def __getitem__(self, i: int) -> ConformalMondrianTreeClassifier:
        """Return a single-tree view for tree ``i`` from the last predict/compute_p_value call.

        The returned object is a :class:`ConformalMondrianTreeClassifier` whose
        ``_last_tree`` is the exact i-th tree from the last call, enabling all
        T1/T2 inspection and visualisation methods (".summary", ".to_dataframe()",
        ".draw()", etc.). Its own ``predict()`` would rebuild a fresh tree (not
        tree i) — this object is intended for *inspection only*.

        Parameters
        ----------
        i : int
            Tree index, ``0 <= i < n_trees``.

        Returns
        -------
        ConformalMondrianTreeClassifier
            Single-tree view with ``_last_tree`` set to tree i.

        Raises
        ------
        RuntimeError
            If ``predict()`` or ``compute_p_value()`` has not been called yet.
        IndexError
            If ``i`` is out of range.
        """
        if self._last_seeds is None:
            raise RuntimeError(
                "No trees cached. Call predict() or compute_p_value() first."
            )
        if not (0 <= i < self.n_trees):
            raise IndexError(f"Tree index {i} out of range [0, {self.n_trees}).")

        n = self.X.shape[0]
        X_aug = np.vstack([self.X, self._last_x])
        K = len(self.label_space)
        rng_i = np.random.default_rng(int(self._last_seeds[i]))
        tree_i = _sample_mondrian_tree(rng_i, X_aug, np.arange(n + 1), 0.0, self.lifetime, max_depth=self.max_depth)
        _assign_counts(tree_i, self.y, self.label_to_idx, K, n_train=n)

        view = ConformalMondrianTreeClassifier(
            lifetime=self.lifetime,
            lambda_=self.lambda_,
            label_space=self.label_space.copy(),
            epsilon=self.epsilon,
            max_depth=self.max_depth,
        )
        view.X = self.X
        view.y = self.y
        view.label_to_idx = self.label_to_idx
        view._last_tree = tree_i
        return view

    @property
    def summary(self) -> dict:
        """Summary statistics of the forest.

        Returns
        -------
        dict
            Contains: n_points, n_features, n_classes, label_space, lifetime, epsilon, n_trees.
            If a tree has been built (after predict() or compute_p_value()), also includes
            stats from the last cached tree: n_nodes, n_leaves, n_branches, height.
        """
        stats = {
            "n_points": len(self.y) if self.y is not None else 0,
            "n_features": self.X.shape[1] if self.X is not None else 0,
            "n_classes": len(self.label_space) if self.label_space is not None else 0,
            "label_space": list(self.label_space) if self.label_space is not None else [],
            "lifetime": self.lifetime,
            "epsilon": self.epsilon,
            "n_trees": self.n_trees,
        }
        if self._last_tree is not None:
            tree_stats = _tree_struct_stats(self._last_tree)
            stats.update(tree_stats)
            stats["total_observed_weight"] = len(self.y)
        return stats

    def to_dataframe(self):
        """Export the last tree structure as a pandas DataFrame.

        Returns a DataFrame with one row per node from the last cached tree,
        including node ID, parent ID, depth, split information, and leaf-specific
        data (class counts for classifier).

        Raises
        ------
        RuntimeError
            If no tree has been built yet (call predict() first).

        Returns
        -------
        pd.DataFrame
            Tree structure with columns: node_id, parent_id, is_leaf, depth,
            split_dim, split_loc, split_time, parent_time, bbox_lower, bbox_upper,
            n_points, and counts (dict, classifier only).
        """
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )
        import pandas as pd

        rows = []
        node_id_map = {}  # Map string node_id to integer
        next_id = 0

        for node, depth, parent_id_str, node_id_str in _iter_nodes(self._last_tree):
            # Convert string node_id to integer
            if node_id_str not in node_id_map:
                node_id_map[node_id_str] = next_id
                next_id += 1
            node_id_int = node_id_map[node_id_str]

            # Convert parent_id_str to integer
            if parent_id_str is None:
                parent_id_int = None
            else:
                if parent_id_str not in node_id_map:
                    node_id_map[parent_id_str] = next_id
                    next_id += 1
                parent_id_int = node_id_map[parent_id_str]

            bbox_lower = node.lower_bounds.tolist() if len(node.lower_bounds) > 0 else []
            bbox_upper = node.upper_bounds.tolist() if len(node.upper_bounds) > 0 else []

            row = {
                "node_id": node_id_int,
                "parent_id": parent_id_int,
                "is_leaf": node.is_leaf(),
                "depth": depth,
                "split_dim": node.split_dim,
                "split_loc": float(node.split_loc) if not node.is_leaf() else None,
                "split_time": float(node.split_time) if not node.is_leaf() else None,
                "parent_time": float(node.parent_time),
                "bbox_lower": bbox_lower,
                "bbox_upper": bbox_upper,
                "n_points": node.n_points(),
            }
            if node.is_leaf() and node.counts is not None:
                counts_dict = {self.label_space[i]: int(node.counts[i]) for i in range(len(self.label_space))}
                row["counts"] = counts_dict
            rows.append(row)

        return pd.DataFrame(rows)

    def debug_one(self, x):
        """Trace the path of a single example through a representative tree from the forest.

        Builds a fresh representative tree (using a deterministic RNG seed)
        and traces the feature vector through it, showing split decisions
        and leaf information.

        Parameters
        ----------
        x : array-like, shape (d,)
            Feature vector to trace.

        Returns
        -------
        str
            Human-readable string showing the decision path and final leaf stats.

        Notes
        -----
        This uses a representative tree sampled from the same distribution as
        the forest, NOT the exact tree from the ensemble.
        This method does not affect the random state of the main RNG.
        """
        x = np.asarray(x, dtype=float).ravel()

        if self.X is None:
            raise RuntimeError("Must call learn_initial_training_set first")

        # Build a representative tree using a fixed seed
        rng_debug = np.random.default_rng(self.rnd_state if self.rnd_state is not None else 0)
        X_aug = np.vstack([self.X, x])
        indices_all = np.arange(len(X_aug))
        tree_repr = _sample_mondrian_tree(
            rng_debug,
            X_aug,
            indices_all,
            parent_time=0.0,
            lifetime=self.lifetime,
            verbose=0,
            max_depth=self.max_depth,
        )

        # Trace the path
        path_lines = []
        current = tree_repr
        depth = 0

        while not current.is_leaf():
            split_str = f"  {'  ' * depth}x[{current.split_dim}] <= {current.split_loc:.6f}"
            if x[current.split_dim] <= current.split_loc:
                path_lines.append(split_str + "  [TRUE → left]")
                current = current.left
            else:
                path_lines.append(split_str + "  [FALSE → right]")
                current = current.right
            depth += 1

        # Leaf info
        leaf_str = f"  {'  ' * depth}LEAF: n_points={current.n_points()}"
        if current.counts is not None:
            counts_str = {self.label_space[i]: int(current.counts[i]) for i in range(len(self.label_space))}
            leaf_str += f", counts={counts_str}"
        path_lines.append(leaf_str)

        return "\n".join(path_lines)

    def draw(self, ax=None, max_depth=None, backend="auto", **kwargs):
        """Draw a node-link tree diagram of the last cached tree.

        **Node types in the diagram**

        - *Internal nodes* (orange): split decisions — ``x[i] ≤ threshold``.
          Left branch follows ``≤``, right branch follows ``>``.
        - *Leaf nodes* (blue): ``→ class C  (P%)`` — the majority class C among
          training points in this cell, with purity P (fraction belonging to C).
          The second line ``n=k  [c0:n0  c1:n1 ...]`` shows the total count and
          per-class breakdown (zero-count classes omitted).
        - *"empty"* leaf: the Mondrian process created this rectangular partition
          cell, but no training points landed in it.  This is normal for small
          datasets or high lifetime values; such leaves contribute a uniform
          conformal score that weakly widens prediction sets.

        .. note::
           The tree shown is the *augmented* tree built on ``{train ∪ x_test}``
           during the most recent ``predict()`` or ``compute_p_value()`` call.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes.  If ``None`` and the graphviz backend is active, a
            :class:`graphviz.Digraph` is returned instead (renders as inline SVG
            in Jupyter notebooks).
        max_depth : int or None
            Maximum depth to display (display-only; does not affect the model).
            Nodes below this depth are shown as collapsed leaf boxes.
        backend : {'auto', 'graphviz', 'matplotlib'}
            Rendering backend.  ``'auto'`` (default) uses graphviz when available
            and falls back to matplotlib.  ``'graphviz'`` raises
            :class:`ImportError` with an install hint if the package is missing.
            ``'matplotlib'`` forces the built-in renderer.
            Install graphviz with ``pip install online-cp[viz]``.
        **kwargs
            Reserved for future style options.

        Returns
        -------
        graphviz.Digraph or matplotlib.axes.Axes
            A :class:`graphviz.Digraph` when graphviz is used and *ax* is
            ``None``; otherwise a :class:`~matplotlib.axes.Axes`.

        Raises
        ------
        RuntimeError
            If no tree has been built yet (call ``predict()`` first).
        """
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )

        def label_fn(node, depth, *, collapsed=False):
            n = node.n_points()
            if node.is_leaf() or collapsed:
                if n == 0:
                    return "empty"
                if node.counts is not None:
                    maj_idx = int(np.argmax(node.counts))
                    maj_label = self.label_space[maj_idx]
                    counts_str = "  ".join(
                        f"{self.label_space[i]}:{int(node.counts[i])}"
                        for i in range(len(self.label_space))
                        if node.counts[i] > 0
                    )
                    purity = int(node.counts[maj_idx]) / n
                    return f"\u2192 class {maj_label}  ({purity:.0%})\nn={n}  [{counts_str}]"
                return f"n={n}"
            return f"x[{node.split_dim}] ≤\n{node.split_loc:.3g}"

        return _render_tree(
            self._last_tree, ax, label_fn, max_depth, backend,
            title="Mondrian Tree (classifier)",
        )

    def draw_partition(self, ax=None, scatter=True, **kwargs):
        """Draw the 2-D Mondrian box-partition of the last cached tree.

        Each leaf is drawn as a rectangle coloured by its majority class.
        Training points can be overlaid as a scatter plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        scatter : bool
            If True (default), overlay training points coloured by label.
        **kwargs
            Currently unused; reserved for future style options.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If no tree has been built yet.
        ValueError
            If the number of features is not exactly 2.
        """
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )
        if self.X.shape[1] != 2:
            raise ValueError(
                f"draw_partition() requires exactly 2 features, got {self.X.shape[1]}. "
                "The leaf bounding boxes only tile feature space exactly at d=2."
            )
        from matplotlib import colormaps
        from matplotlib.lines import Line2D

        ax = _get_ax(ax)

        K = len(self.label_space)
        cmap = colormaps["tab10"].resampled(K)
        label_to_color = {lbl: cmap(i) for i, lbl in enumerate(self.label_space)}

        def leaf_color_fn(leaf):
            if leaf.counts is None or leaf.counts.sum() == 0:
                return "#cccccc"
            majority_idx = int(np.argmax(leaf.counts))
            return label_to_color[self.label_space[majority_idx]]

        scatter_X = self.X if scatter else None
        scatter_y = [list(self.label_space).index(yi) for yi in self.y] if scatter else None
        _draw_partition(
            self._last_tree, self.X, ax, leaf_color_fn,
            scatter_X=scatter_X, scatter_y=scatter_y, scatter_cmap="tab10",
        )

        # Legend
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lbl],
                   markersize=8, label=str(lbl))
            for lbl in self.label_space
        ]
        ax.legend(handles=handles, title="Class", fontsize=7, title_fontsize=7)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title("Mondrian Partition (classifier)", fontsize=9)
        return ax


    def predict(
        self,
        x: NDArray,
        epsilon: float | NDArray | None = None,
        return_p_values: bool = False,
        return_update: bool = False,
    ) -> ConformalPredictionSet | MultiLevelPredictionSet | tuple:
        """Make conformal prediction using ensemble of Mondrian trees.

        For each candidate label, builds T trees from the augmented bag and
        computes the average leaf posterior across all trees. The NCM is then
        1 - avg_posterior, which is used to rank all n+1 points and compute
        the p-value.

        Args:
            x: Test feature vector (d,) or (1, d)
            epsilon: Significance level(s). If None, uses self.epsilon.
                     Can be scalar or array-like.
            return_p_values: If True, also return dict of p-values keyed by label.
            return_update: If True, also return dict with forest info.

        Returns:
            ConformalPredictionSet: If epsilon is scalar and neither flag is True
            MultiLevelPredictionSet: If epsilon is array-like
            Tuple: If return_p_values or return_update flag is set
        """
        x = np.asarray(x).ravel()

        if self.X is None:
            raise ValueError("Must call learn_initial_training_set first")

        if x.shape[0] != self.X.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[0]}, expected {self.X.shape[1]}")

        if epsilon is None:
            epsilon = self.epsilon

        # Form augmented feature matrix (same for all candidates and all trees)
        X_aug = np.vstack([self.X, x])
        n = self.X.shape[0]
        n_total = n + 1
        K = len(self.label_space)

        # Build T tree summaries — possibly in parallel (n_jobs follows sklearn semantics)
        seeds = self.rnd_gen.integers(0, 2**31, self.n_trees)
        # Cache seeds + test point for __getitem__ and representative-tree viz
        self._last_seeds = seeds.copy()
        self._last_x = x.copy()
        _rng0 = np.random.default_rng(int(seeds[0]))
        _tree0 = _sample_mondrian_tree(_rng0, X_aug, np.arange(n_total), 0.0, self.lifetime, max_depth=self.max_depth)
        _assign_counts(_tree0, self.y, self.label_to_idx, K, n_train=n)
        self._last_tree = _tree0
        tree_summaries = Parallel(n_jobs=self.n_jobs)(
            delayed(_build_tree_summary)(int(seed), X_aug, self.y, self.label_to_idx, K, n, self.lifetime, self.max_depth)
            for seed in seeds
        )
        n_leaves_all = [ts[0] for ts in tree_summaries]  # T x (n_total,)
        counts_all = [ts[1] for ts in tree_summaries]  # T x (n_total, K)
        ls_train_idx = [ts[2] for ts in tree_summaries]  # T x variable

        # Integer label indices for all training points
        LAMBDA = self.lambda_
        y_idx = np.array([self.label_to_idx[self.y[i]] for i in range(n)], dtype=np.int64)
        arange_n = np.arange(n)

        # base_ncm[i] = 1 - (1/T) * sum_t (counts_t[i, y_idx[i]] + λ) / (n_train_leaf_t[i] + K*λ)
        base_ncm_sum = np.zeros(n)
        for t in range(self.n_trees):
            nl = n_leaves_all[t][:n].astype(np.float64)  # (n,) structural sizes
            ct = counts_all[t][:n]  # (n, K)
            # leaf_star points have structural size including test slot
            n_train_leaf = nl.copy()
            for idx in ls_train_idx[t]:
                n_train_leaf[int(idx)] -= 1.0
            denom = n_train_leaf + K * LAMBDA
            base_ncm_sum += (ct[arange_n, y_idx].astype(np.float64) + LAMBDA) / denom
        base_ncm = 1.0 - base_ncm_sum / self.n_trees

        # ncm_test[k] = 1 - (1/T) * sum_t (counts_t[n, k] + 1 + λ) / (n_star_train_t + 1 + K*λ)
        ncm_test = np.zeros(K)
        for t in range(self.n_trees):
            n_star_train_t = float(n_leaves_all[t][n]) - 1.0
            denom_test_t = n_star_train_t + 1.0 + K * LAMBDA
            ncm_test += (counts_all[t][n].astype(np.float64) + 1.0 + LAMBDA) / denom_test_t
        ncm_test = 1.0 - ncm_test / self.n_trees  # shape (K,)

        # Vectorised (K, n) broadcast
        _TOL = 1e-9
        diff = base_ncm[np.newaxis, :] - ncm_test[:, np.newaxis]  # (K, n)
        gt = np.sum(diff > _TOL, axis=1).astype(np.int64)
        eq = np.sum(np.abs(diff) <= _TOL, axis=1).astype(np.int64)

        # Delta correction (Laplace): for each tree t, compute per-class delta
        # in leaf_star_t, then accumulate per-point adjustments.
        # Build per-tree delta matrices: delta_jk[j, k] = actual_prob - base_prob
        tree_deltas = []
        for t in range(self.n_trees):
            n_star_t = float(n_leaves_all[t][n])
            c_t = counts_all[t][n].astype(np.float64)
            denom_base_t = (n_star_t - 1.0) + K * LAMBDA
            denom_actual_t = n_star_t + K * LAMBDA
            delta_jk = np.empty((K, K))
            for j in range(K):
                base_p = (c_t[j] + LAMBDA) / denom_base_t
                for k in range(K):
                    actual_p = (c_t[j] + LAMBDA + (1.0 if j == k else 0.0)) / denom_actual_t
                    delta_jk[j, k] = actual_p - base_p
            tree_deltas.append(delta_jk)

        # Build point -> trees membership
        point_trees: dict[int, list] = {}
        for t in range(self.n_trees):
            for idx in ls_train_idx[t]:
                i_int = int(idx)
                if i_int not in point_trees:
                    point_trees[i_int] = []
                point_trees[i_int].append(t)

        # Reclassify affected points
        for i_int, trees_list in point_trees.items():
            j = int(y_idx[i_int])
            delta_sum_k = np.zeros(K)
            for t in trees_list:
                delta_sum_k += tree_deltas[t][j, :]
            delta_avg_k = delta_sum_k / self.n_trees

            base_val = base_ncm[i_int]
            for k in range(K):
                actual_val = base_val - delta_avg_k[k]
                ncm_test_k = ncm_test[k]
                # Before bucket
                if base_val > ncm_test_k + _TOL:
                    before = 0
                elif abs(base_val - ncm_test_k) <= _TOL:
                    before = 1
                else:
                    before = 2
                # After bucket
                if actual_val > ncm_test_k + _TOL:
                    after = 0
                elif abs(actual_val - ncm_test_k) <= _TOL:
                    after = 1
                else:
                    after = 2
                if before == after:
                    continue
                if before == 0 and after == 1:
                    gt[k] -= 1
                    eq[k] += 1
                elif before == 0 and after == 2:
                    gt[k] -= 1
                elif before == 1 and after == 2:
                    eq[k] -= 1
                elif before == 1 and after == 0:
                    eq[k] -= 1
                    gt[k] += 1
                elif before == 2 and after == 1:
                    eq[k] += 1
                elif before == 2 and after == 0:
                    gt[k] += 1

        eq += 1  # test point always EQ to itself
        tau = self.rnd_gen.uniform(0, 1)
        p_values_arr = (gt + tau * eq) / n_total
        p_values = {label: float(p_values_arr[ki]) for ki, label in enumerate(self.label_space)}

        result = self._compute_Gamma(p_values, epsilon)

        # Handle return flags
        _update = {"n_leaves_all": n_leaves_all, "counts_all": counts_all, "leaf_star_train_indices_all": ls_train_idx}
        if return_update and return_p_values:
            return result, p_values, _update
        elif return_p_values:
            return result, p_values
        elif return_update:
            return result, _update
        else:
            return result

    def compute_p_value(self, x: NDArray, y: Any, return_update: bool = False) -> float | tuple[float, dict]:
        """Compute p-value for a single candidate label.

        Args:
            x: Test feature vector (d,)
            y: Candidate label
            return_update: If True, also return dict with forest info

        Returns:
            P-value (float) or (p-value, dict) if return_update=True
        """
        x = np.asarray(x).ravel()

        if y not in self.label_to_idx:
            raise ValueError(f"Label {y} not in label_space")

        # Build augmented dataset
        X_aug = np.vstack([self.X, x])
        n = self.X.shape[0]
        n_total = n + 1
        K = len(self.label_space)
        label_idx = self.label_to_idx[y]

        # Build T tree summaries (possibly in parallel)
        seeds = self.rnd_gen.integers(0, 2**31, self.n_trees)
        # Cache seeds + test point for __getitem__ and representative-tree viz
        self._last_seeds = seeds.copy()
        self._last_x = x.copy()
        _rng0 = np.random.default_rng(int(seeds[0]))
        _tree0 = _sample_mondrian_tree(_rng0, X_aug, np.arange(n_total), 0.0, self.lifetime, max_depth=self.max_depth)
        _assign_counts(_tree0, self.y, self.label_to_idx, K, n_train=n)
        self._last_tree = _tree0
        tree_summaries = Parallel(n_jobs=self.n_jobs)(
            delayed(_build_tree_summary)(int(seed), X_aug, self.y, self.label_to_idx, K, n, self.lifetime, self.max_depth)
            for seed in seeds
        )
        n_leaves_all = [ts[0] for ts in tree_summaries]
        counts_all = [ts[1] for ts in tree_summaries]
        ls_train_idx = [ts[2] for ts in tree_summaries]

        LAMBDA = self.lambda_
        y_idx = np.array([self.label_to_idx[self.y[i]] for i in range(n)], dtype=np.int64)
        arange_n = np.arange(n)

        # base_ncm for training points (Laplace)
        base_ncm_sum = np.zeros(n)
        for t in range(self.n_trees):
            nl = n_leaves_all[t][:n].astype(np.float64)
            ct = counts_all[t][:n]
            n_train_leaf = nl.copy()
            for idx in ls_train_idx[t]:
                n_train_leaf[int(idx)] -= 1.0
            denom = n_train_leaf + K * LAMBDA
            base_ncm_sum += (ct[arange_n, y_idx].astype(np.float64) + LAMBDA) / denom
        base_ncm = 1.0 - base_ncm_sum / self.n_trees

        # ncm_test for this single candidate (Laplace)
        ncm_test_k = 0.0
        for t in range(self.n_trees):
            n_star_train_t = float(n_leaves_all[t][n]) - 1.0
            denom_test_t = n_star_train_t + 1.0 + K * LAMBDA
            ncm_test_k += (float(counts_all[t][n, label_idx]) + 1.0 + LAMBDA) / denom_test_t
        ncm_test_k = 1.0 - ncm_test_k / self.n_trees

        _TOL = 1e-9
        diff = base_ncm - ncm_test_k
        gt = int(np.sum(diff > _TOL))
        eq = int(np.sum(np.abs(diff) <= _TOL))

        # Delta correction (Laplace): per-tree delta for leaf_star points
        tree_deltas = []
        for t in range(self.n_trees):
            n_star_t = float(n_leaves_all[t][n])
            c_t = counts_all[t][n].astype(np.float64)
            denom_base_t = (n_star_t - 1.0) + K * LAMBDA
            denom_actual_t = n_star_t + K * LAMBDA
            # delta[j] = actual_prob_j - base_prob_j for candidate label_idx
            delta_j = np.empty(K)
            for j in range(K):
                base_p = (c_t[j] + LAMBDA) / denom_base_t
                actual_p = (c_t[j] + LAMBDA + (1.0 if j == label_idx else 0.0)) / denom_actual_t
                delta_j[j] = actual_p - base_p
            tree_deltas.append(delta_j)

        # Build point -> trees membership
        point_trees: dict[int, list] = {}
        for t in range(self.n_trees):
            for idx in ls_train_idx[t]:
                i_int = int(idx)
                if i_int not in point_trees:
                    point_trees[i_int] = []
                point_trees[i_int].append(t)

        # Reclassify affected points
        for i_int, trees_list in point_trees.items():
            j = int(y_idx[i_int])
            delta_sum = 0.0
            for t in trees_list:
                delta_sum += tree_deltas[t][j]
            delta_avg = delta_sum / self.n_trees

            base_val = base_ncm[i_int]
            actual_val = base_val - delta_avg
            # Before bucket
            if base_val > ncm_test_k + _TOL:
                before = 0
            elif abs(base_val - ncm_test_k) <= _TOL:
                before = 1
            else:
                before = 2
            # After bucket
            if actual_val > ncm_test_k + _TOL:
                after = 0
            elif abs(actual_val - ncm_test_k) <= _TOL:
                after = 1
            else:
                after = 2
            if before == after:
                continue
            if before == 0 and after == 1:
                gt -= 1
                eq += 1
            elif before == 0 and after == 2:
                gt -= 1
            elif before == 1 and after == 2:
                eq -= 1
            elif before == 1 and after == 0:
                eq -= 1
                gt += 1
            elif before == 2 and after == 1:
                eq += 1
            elif before == 2 and after == 0:
                gt += 1

        eq += 1  # test point always EQ to itself
        tau = self.rnd_gen.uniform(0, 1)
        p_val = float((gt + tau * eq) / n_total)

        if return_update:
            return p_val, {
                "n_leaves_all": n_leaves_all,
                "counts_all": counts_all,
                "leaf_star_train_indices_all": ls_train_idx,
            }
        else:
            return p_val


