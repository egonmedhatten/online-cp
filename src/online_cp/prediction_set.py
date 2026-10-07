"""Unified conformal prediction set hierarchy.

This module introduces the ``ConformalPredictionSet`` hierarchy — a single
abstract base with three concrete subclasses covering the three kinds of
prediction sets produced by the online conformal predictors in this library:

* ``EmptyPredictionSet`` — the empty set ``∅`` (no ``y`` satisfies
  ``p(y) > ε``);
* ``DiscretePredictionSet`` — a finite set of labels, produced by
  conformal classifiers;
* ``ContinuousPredictionSet`` — one or more **disjoint closed** intervals
  ``[a, b]`` (including rays and ``(-∞, ∞)``), produced by conformal
  regressors. The interval endpoints are numeric real bounds and are
  **closed** — the inclusion criterion is the super-level set
  ``{y : p(y) > ε}`` which, for a continuous nonconformity measure
  ``α(y)``, is algebraically ``{y : α(y) ≤ c}`` — a **closed** set.

The old class names (``ConformalPredictionInterval`` in
``online_cp.regressors`` and ``ConformalPredictionSet`` /
``MultiLevelPredictionSet`` in ``online_cp.classifiers``) are kept as
``DeprecationWarning``-raising wrappers in their home modules for
downstream users.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ConformalPredictionSet(ABC):
    """Abstract base for all conformal prediction sets.

    A concrete prediction set is one of ``EmptyPredictionSet``,
    ``DiscretePredictionSet`` or ``ContinuousPredictionSet``. Instances of
    the base class are not meant to be created directly — use a subclass.

    Shared interface
    ----------------
    * ``is_empty`` — ``True`` if the set contains no values.
    * ``is_discrete`` — ``True`` for discrete (classifier) sets.
    * ``is_continuous`` — ``True`` for continuous (interval) sets.
    * ``width()`` — total "width" (0 for empty, ∞ for unbounded, number of
      elements for discrete, sum of interval lengths for continuous).
    * ``__contains__(y)`` / ``coverage(y)`` — membership test.
    """

    @property
    def is_empty(self) -> bool:
        """``True`` if the prediction set contains no values.

        For the base class the default is ``True`` — a bare
        ``ConformalPredictionSet()`` represents the empty set. Subclasses
        override with the correct semantics.
        """
        return True

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """``True`` if the set contains discrete elements (classifiers)."""

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """``True`` if the set contains continuous intervals (regressors)."""

    @abstractmethod
    def __contains__(self, y: Any) -> bool:
        """``True`` if ``y`` is in the prediction set."""

    @abstractmethod
    def width(self) -> float:
        """Total width of the prediction set.

        Returns ``0.0`` for empty sets, ``np.inf`` for unbounded continuous
        sets, the number of elements for discrete sets, and the sum of
        interval lengths for bounded continuous sets.
        """

    def coverage(self, y: Any) -> bool:
        """Alias for ``__contains__`` for API symmetry."""
        return y in self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConformalPredictionSet):
            return NotImplemented
        if self.is_discrete != other.is_discrete:
            return False
        if self.is_continuous != other.is_continuous:
            return False
        if self.is_empty != other.is_empty:
            return False
        if self.is_discrete:
            return np.array_equal(
                np.asarray(self.elements, dtype=object),  # type: ignore[attr-defined]
                np.asarray(other.elements, dtype=object),  # type: ignore[attr-defined]
            )
        if self.is_continuous:
            return self.intervals == other.intervals  # type: ignore[attr-defined]
        return True

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self.is_empty:
            return f"{cls}(∅)"
        if self.is_discrete:
            return f"{cls}({np.asarray(self.elements)})"  # type: ignore[attr-defined]
        if self.is_continuous:
            return f"{cls}({self.intervals})"  # type: ignore[attr-defined]
        return f"{cls}()"


class EmptyPredictionSet(ConformalPredictionSet):
    """The empty prediction set ``∅``.

    Returned when no ``y`` satisfies the conformal condition ``p(y) > ε``
    (i.e. the super-level set is empty).
    """

    def __init__(self) -> None:
        pass

    @property
    def is_empty(self) -> bool:
        return True

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def is_continuous(self) -> bool:
        return False

    def __contains__(self, y: Any) -> bool:
        return False

    def width(self) -> float:
        return 0.0

    @property
    def elements(self) -> NDArray[Any]:
        """An empty numpy array (discrete-style accessor)."""
        return np.array([], dtype=object)

    @property
    def intervals(self) -> list:
        """An empty list of intervals (continuous-style accessor)."""
        return []

    def to_single_interval(self) -> tuple[float, float] | None:
        """Return ``None`` — the empty set has no single interval."""
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConformalPredictionSet) and other.is_empty and not other.is_discrete and not other.is_continuous

    def __repr__(self) -> str:
        return "EmptyPredictionSet(∅)"


class DiscretePredictionSet(ConformalPredictionSet):
    """A discrete prediction set (classifiers).

    Parameters
    ----------
    elements : array-like
        The set of labels contained in the prediction set.
    epsilon : float
        Significance level at which the set was constructed.

    Examples
    --------
    >>> s = DiscretePredictionSet(np.array([0, 1, 2]), epsilon=0.1)
    >>> s.is_discrete
    True
    >>> s.width()
    3.0
    >>> 1 in s
    True
    >>> 3 in s
    False
    """

    def __init__(self, elements: NDArray[Any] | Any, epsilon: float) -> None:
        self.elements = np.asarray(elements)
        self.epsilon = epsilon

    @property
    def is_empty(self) -> bool:
        return self.elements.size == 0

    @property
    def is_discrete(self) -> bool:
        return True

    @property
    def is_continuous(self) -> bool:
        return False

    def __contains__(self, y: Any) -> bool:
        if self.elements.size == 0:
            return False
        # Use numpy membership test so that array elements (e.g. ints) compare
        # element-wise without raising.
        try:
            return bool(np.any(self.elements == y))
        except (ValueError, TypeError):
            return y in self.elements

    def width(self) -> float:
        return float(self.elements.size)

    def size(self) -> int:
        """Number of elements in the prediction set (alias for ``width()``)."""
        return int(self.elements.size)

    def __len__(self) -> int:
        return int(self.elements.size)

    def __eq__(self, other: object) -> bool:
        """Order-insensitive equality (this is a *set* of labels)."""
        if not isinstance(other, DiscretePredictionSet):
            return NotImplemented
        return set(np.asarray(self.elements).tolist()) == set(np.asarray(other.elements).tolist())

    def __repr__(self) -> str:
        # Matches the legacy classifier ``ConformalPredictionSet.__repr__``
        # (which returned ``repr(self.elements)``) for backward compatibility.
        return repr(self.elements)

    def __str__(self) -> str:
        return str(self.elements)


class ContinuousPredictionSet(ConformalPredictionSet):
    """A continuous prediction set: one or more **disjoint closed** intervals.

    All intervals are **closed** ``[a, b]``. The endpoints are *numeric
    real* bounds; the special value ``np.inf`` is allowed for rays. A
    degenerate ``[a, a]`` interval represents the single point ``{a}``.

    Parameters
    ----------
    intervals : list[tuple[float, float]]
        One or more closed intervals ``[(a, b), ...]``. May be empty
        (equivalent to ``EmptyPredictionSet``) or contain an explicit
        ``(np.nan, np.nan)`` pair (also treated as empty for backward
        compatibility with the legacy ``ConformalPredictionInterval(nan, nan)``
        convention).
    epsilon : float
        Significance level at which the set was constructed.

    Examples
    --------
    >>> s = ContinuousPredictionSet([(1.2, 3.7)], epsilon=0.1)
    >>> s.is_continuous
    True
    >>> s.is_single_interval
    True
    >>> s.width()
    2.5
    >>> 1.2 in s  # boundary included (closed)
    True
    >>> 3.7 in s  # boundary included (closed)
    True
    >>> 2.0 in s
    True
    """

    def __init__(
        self,
        intervals: list[tuple[float, float]] | tuple[float, float] | Any,
        epsilon: float,
    ) -> None:
        # Accept a single (lower, upper) pair or a list of intervals.
        self._intervals = self._normalize(intervals)
        self.epsilon = epsilon

    @staticmethod
    def _normalize(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Sort and merge intervals into a canonical form.

        Rules:
        - Intervals whose endpoints are both NaN are dropped (they represent
          the empty set in the legacy API).
        - Intervals whose endpoints are not finite AND not ±inf are dropped
          (defensive; NaN already handled).
        - Intervals are sorted by lower bound, then merged when touching or
          overlapping (closed-interval semantics: ``b_i >= a_{i+1}``).
        """
        if intervals is None:
            return []
        if isinstance(intervals, (tuple, list)) and len(intervals) == 2 and not isinstance(intervals[0], (tuple, list)):
            # A single (lower, upper) pair.
            intervals = [tuple(intervals)]
        cleaned = []
        for lo, hi in intervals:
            lo_f = float(lo)
            hi_f = float(hi)
            # Drop the legacy empty sentinel (nan, nan) and any interval with a
            # NaN endpoint — both are nonsensical and treated as empty.
            if np.isnan(lo_f) or np.isnan(hi_f):
                continue
            if hi_f < lo_f:
                # Inverted interval is invalid — drop (keeps single point a==a).
                continue
            cleaned.append((lo_f, hi_f))
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: x[0])
        merged = [cleaned[0]]
        for lo, hi in cleaned[1:]:
            prev_lo, prev_hi = merged[-1]
            # Closed-interval semantics: merge when touching (lo == prev_hi) or
            # overlapping (lo < prev_hi). This makes [1, 2] and [2, 3] -> [1, 3].
            if lo <= prev_hi:
                merged[-1] = (prev_lo, max(prev_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return len(self._intervals) == 0

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def is_continuous(self) -> bool:
        return True

    @property
    def intervals(self) -> list[tuple[float, float]]:
        """Return the list of closed intervals in canonical form."""
        return list(self._intervals)

    @property
    def is_single_interval(self) -> bool:
        """``True`` if the set is a single connected component."""
        return len(self._intervals) == 1

    def width(self) -> float:
        total = 0.0
        for lo, hi in self._intervals:
            if np.isinf(lo) or np.isinf(hi):
                return np.inf
            total += hi - lo
        return total

    def __contains__(self, y: Any) -> bool:
        try:
            y_f = float(y)
        except (TypeError, ValueError):
            return False
        return any(lo <= y_f <= hi for lo, hi in self._intervals)

    def to_single_interval(self) -> tuple[float, float] | None:
        """Return ``(lower, upper)`` if single interval, else ``None``."""
        if len(self._intervals) != 1:
            return None
        return self._intervals[0]

    # ------------------------------------------------------------------
    # Backward-compatible accessors (legacy ``ConformalPredictionInterval``)
    # ------------------------------------------------------------------
    @property
    def lower(self) -> float:
        """Lower bound of the *first* interval.

        .. deprecated::
            Prefer ``.intervals[0][0]``. Retained for backward compatibility
            with the old ``ConformalPredictionInterval`` class.
        """
        if not self._intervals:
            return float("nan")
        return self._intervals[0][0]

    @property
    def upper(self) -> float:
        """Upper bound of the *last* interval.

        .. deprecated::
            Prefer ``.intervals[-1][1]``. Retained for backward compatibility
            with the old ``ConformalPredictionInterval`` class.
        """
        if not self._intervals:
            return float("nan")
        return self._intervals[-1][1]

    def __repr__(self) -> str:
        return f"ContinuousPredictionSet({self._intervals})"


class MultiLevelPredictionSet:
    """Prediction sets at multiple significance levels (unified).

    Returned by ``predict`` (and ``predict_set``) when ``epsilon`` is an
    array-like. Replaces the legacy ``MultiLevelPredictionInterval``
    (regressors) and ``MultiLevelPredictionSet`` (classifiers) with a single
    class that works for both discrete and continuous prediction sets.

    Parameters
    ----------
    predictions : dict[float, ConformalPredictionSet]
        Mapping ``{epsilon: prediction_set}``. Keys are sorted.

    Examples
    --------
    >>> from online_cp.prediction_set import DiscretePredictionSet
    >>> ml = MultiLevelPredictionSet({
    ...     0.1: DiscretePredictionSet(np.array([0, 1]), 0.1),
    ...     0.2: DiscretePredictionSet(np.array([0, 1, 2]), 0.2),
    ... })
    >>> ml.levels
    [0.1, 0.2]
    >>> 0 in ml
    True
    >>> 2 in ml[0.2]
    True
    """

    def __init__(self, predictions: dict[float, ConformalPredictionSet]) -> None:
        self._predictions = dict(sorted(predictions.items()))

    @property
    def levels(self) -> list[float]:
        """Sorted list of significance levels."""
        return list(self._predictions.keys())

    @property
    def is_empty(self) -> bool:
        """``True`` if the prediction set is empty at **any** level."""
        return any(pset.is_empty for pset in self._predictions.values())

    def __getitem__(self, eps: float) -> ConformalPredictionSet:
        return self._predictions[eps]

    def __iter__(self):
        return iter(self._predictions.items())

    def __len__(self) -> int:
        return len(self._predictions)

    def __contains__(self, y: Any) -> bool:
        """``True`` if ``y`` is covered at **all** levels."""
        if self.is_empty:
            return False
        return all(y in pset for pset in self._predictions.values())

    def coverage(self, y: Any) -> dict[float, bool]:
        """Return ``{epsilon: bool}`` indicating coverage at each level."""
        return {eps: (y in pset) for eps, pset in self._predictions.items()}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiLevelPredictionSet):
            return NotImplemented
        if self.levels != other.levels:
            return False
        return all(self[eps] == other[eps] for eps in self.levels)

    def __repr__(self) -> str:
        parts = [f"  ε={eps}: {pset!r}" for eps, pset in self._predictions.items()]
        return "MultiLevelPredictionSet(\n" + "\n".join(parts) + "\n)"


__all__ = [
    "ConformalPredictionSet",
    "EmptyPredictionSet",
    "DiscretePredictionSet",
    "ContinuousPredictionSet",
    "MultiLevelPredictionSet",
]
