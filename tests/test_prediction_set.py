"""Tests for the unified ``ConformalPredictionSet`` hierarchy.

Covers:
- the base class is abstract and shares a consistent interface;
- ``EmptyPredictionSet`` / ``DiscretePredictionSet`` / ``ContinuousPredictionSet``
  semantics (closed intervals, rays, degenerate points, disjoint unions,
  touching-merge, NaN-empty sentinel, ``lower``/``upper`` backward-compat);
- ``MultiLevelPredictionSet`` nesting / coverage;
- the deprecated wrappers (``ConformalPredictionInterval`` in regressors and
  ``classifiers.ConformalPredictionSet``) emit ``DeprecationWarning`` and keep
  the legacy ``(lower, upper, epsilon)`` signature and ``.lower``/``.upper``
  access;
- mathematical invariants (closed boundaries, width, disjointness, nesting).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from online_cp import (
    ConformalPredictionSet,
    ContinuousPredictionSet,
    DiscretePredictionSet,
    EmptyPredictionSet,
    MultiLevelPredictionSet,
)
from online_cp.classifiers import ConformalPredictionSet as _ClassifierLegacyCPSet
from online_cp.regressors import (
    ConformalPredictionInterval,
    MultiLevelPredictionInterval,
)


# ---------------------------------------------------------------------------
# Base class / hierarchy
# ---------------------------------------------------------------------------
def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        ConformalPredictionSet()  # type: ignore[abstract]


def test_subclasses_are_instances_of_base():
    assert issubclass(EmptyPredictionSet, ConformalPredictionSet)
    assert issubclass(DiscretePredictionSet, ConformalPredictionSet)
    assert issubclass(ContinuousPredictionSet, ConformalPredictionSet)


def test_base_interface_is_present_on_all():
    """Every concrete class must expose the shared interface."""
    for cls in (EmptyPredictionSet, DiscretePredictionSet, ContinuousPredictionSet):
        for attr in ("is_empty", "is_discrete", "is_continuous", "width", "coverage"):
            assert hasattr(cls, attr), f"{cls.__name__} missing {attr}"


# ---------------------------------------------------------------------------
# EmptyPredictionSet
# ---------------------------------------------------------------------------
def test_empty_set_semantics():
    e = EmptyPredictionSet()
    assert e.is_empty
    assert not e.is_discrete
    assert not e.is_continuous
    assert e.width() == 0.0
    assert 0 not in e
    assert 1.5 not in e
    assert e.intervals == []
    assert e.elements.size == 0
    assert e.to_single_interval() is None


def test_nan_interval_is_empty():
    """The legacy NaN interval is now just the empty continuous set."""
    c = ContinuousPredictionSet([], 0.1)
    assert c.is_empty
    assert c.width() == 0.0
    assert c.intervals == []
    assert c == ContinuousPredictionSet([], 0.1)


# ---------------------------------------------------------------------------
# DiscretePredictionSet
# ---------------------------------------------------------------------------
def test_discrete_semantics():
    d = DiscretePredictionSet(np.array([0, 1, 2]), epsilon=0.1)
    assert d.is_discrete
    assert not d.is_continuous
    assert not d.is_empty
    assert d.width() == 3
    assert d.size() == 3
    assert len(d) == 3
    assert 0 in d and 1 in d and 2 in d
    assert 3 not in d
    assert d.coverage(1) is True
    assert d.coverage(5) is False


def test_discrete_empty():
    d = DiscretePredictionSet(np.array([]), epsilon=0.1)
    assert d.is_empty
    assert d.width() == 0
    assert 0 not in d


def test_discrete_repr_matches_legacy():
    # Backward compatibility with the old classifier repr (plain array).
    d = DiscretePredictionSet(np.array([0, 1, 2]), epsilon=0.1)
    assert repr(d) == repr(np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# ContinuousPredictionSet
# ---------------------------------------------------------------------------
def test_single_closed_interval():
    c = ContinuousPredictionSet([(1.2, 3.7)], epsilon=0.1)
    assert c.is_continuous
    assert not c.is_discrete
    assert not c.is_empty
    assert c.is_single_interval
    assert c.intervals == [(1.2, 3.7)]
    assert c.width() == pytest.approx(2.5)
    # Closed interval: boundaries included.
    assert 1.2 in c
    assert 3.7 in c
    assert 2.0 in c
    assert 1.1 not in c
    assert 3.8 not in c
    assert c.to_single_interval() == (1.2, 3.7)


def test_degenerate_single_point():
    c = ContinuousPredictionSet([(5.0, 5.0)], epsilon=0.1)
    assert not c.is_empty
    assert c.width() == 0.0
    assert 5.0 in c
    assert 4.9 not in c
    assert 5.1 not in c
    assert c.to_single_interval() == (5.0, 5.0)


def test_rays_and_all_reals():
    lo = ContinuousPredictionSet([(-np.inf, 3.0)], epsilon=0.1)
    assert lo.width() == np.inf
    assert -100.0 in lo
    assert 3.0 in lo  # closed numeric bound
    assert 3.1 not in lo

    hi = ContinuousPredictionSet([(3.0, np.inf)], epsilon=0.1)
    assert hi.width() == np.inf
    assert 100.0 in hi
    assert 3.0 in hi
    assert 2.9 not in hi

    allr = ContinuousPredictionSet([(-np.inf, np.inf)], epsilon=0.1)
    assert allr.width() == np.inf
    assert 1e9 in allr and -1e9 in allr


def test_disjoint_intervals():
    c = ContinuousPredictionSet(
        [(-np.inf, -1.0), (0.0, 1.0), (2.0, np.inf)], epsilon=0.1
    )
    assert not c.is_single_interval
    assert c.intervals == [(-np.inf, -1.0), (0.0, 1.0), (2.0, np.inf)]
    assert c.width() == np.inf
    # Membership respects the gaps.
    assert -2.0 in c
    assert -1.0 in c  # closed boundary
    assert -0.5 not in c  # gap
    assert 0.0 in c and 1.0 in c
    assert 1.5 not in c  # gap
    assert 2.0 in c and 100.0 in c
    assert c.to_single_interval() is None


def test_touching_intervals_merge():
    """Closed-interval semantics: [1,2] and [2,3] touch -> [1,3]."""
    c = ContinuousPredictionSet([(1.0, 2.0), (2.0, 3.0)], epsilon=0.1)
    assert c.intervals == [(1.0, 3.0)]
    assert c.is_single_interval
    assert 2.0 in c


def test_overlapping_intervals_merge():
    c = ContinuousPredictionSet([(1.0, 3.0), (2.0, 5.0)], epsilon=0.1)
    assert c.intervals == [(1.0, 5.0)]


def test_intervals_are_sorted_and_normalized():
    c = ContinuousPredictionSet([(5.0, 6.0), (0.0, 1.0), (3.0, 4.0)], epsilon=0.1)
    assert c.intervals == [(0.0, 1.0), (3.0, 4.0), (5.0, 6.0)]


def test_single_pair_constructor_accepted():
    """A bare (lower, upper) pair is accepted, matching legacy ergonomics."""
    c = ContinuousPredictionSet((1.0, 5.0), epsilon=0.1)
    assert c.intervals == [(1.0, 5.0)]
    assert 1.0 in c and 5.0 in c


def test_nan_sentinel_is_empty():
    c = ContinuousPredictionSet([(np.nan, np.nan)], epsilon=0.1)
    assert c.is_empty
    assert c.width() == 0.0
    assert 0.0 not in c


def test_lower_upper_backward_compat():
    c = ContinuousPredictionSet([(1.2, 3.7)], epsilon=0.1)
    assert c.lower == pytest.approx(1.2)
    assert c.upper == pytest.approx(3.7)


def test_contains_non_numeric_is_false():
    c = ContinuousPredictionSet([(1.0, 3.0)], epsilon=0.1)
    assert "foo" not in c
    assert None not in c


# ---------------------------------------------------------------------------
# Invariants (property-style)
# ---------------------------------------------------------------------------
def test_interval_membership_invariant():
    """For a bounded closed interval, membership == (lo <= y <= hi)."""
    lo, hi = 0.3, 7.7
    c = ContinuousPredictionSet([(lo, hi)], epsilon=0.1)
    for y in np.linspace(lo - 1, hi + 1, 41):
        assert (y in c) == (lo <= y <= hi)


def test_width_is_sum_of_lengths():
    c = ContinuousPredictionSet([(0.0, 1.0), (3.0, 5.5), (9.0, 9.0)], epsilon=0.1)
    assert c.width() == pytest.approx(1.0 + 2.5 + 0.0)


def test_discrete_membership_invariant():
    labels = np.array([0, 2, 5, 7])
    d = DiscretePredictionSet(labels, epsilon=0.1)
    for y in range(10):
        assert (y in d) == (y in set(labels.tolist()))


# ---------------------------------------------------------------------------
# MultiLevelPredictionSet
# ---------------------------------------------------------------------------
def test_multi_level_basic():
    ml = MultiLevelPredictionSet(
        {
            0.1: DiscretePredictionSet(np.array([0, 1, 2]), 0.1),
            0.2: DiscretePredictionSet(np.array([0, 1]), 0.2),
        }
    )
    assert ml.levels == [0.1, 0.2]
    assert len(ml) == 2
    # Covered at ALL levels only.
    assert 0 in ml and 1 in ml
    assert 2 not in ml  # only in the 0.1 level
    assert ml.coverage(0) == {0.1: True, 0.2: True}
    assert ml.coverage(2) == {0.1: True, 0.2: False}
    assert not ml.is_empty


def test_multi_level_is_empty_if_any_level_empty():
    ml = MultiLevelPredictionSet(
        {
            0.1: EmptyPredictionSet(),
            0.2: DiscretePredictionSet(np.array([0]), 0.2),
        }
    )
    assert ml.is_empty
    assert 0 not in ml


def test_multi_level_nesting_in_continuous():
    """Larger epsilon (stricter) -> subset; sets nest by epsilon."""
    ml = MultiLevelPredictionSet(
        {
            0.1: ContinuousPredictionSet([(-np.inf, np.inf)], 0.1),
            0.5: ContinuousPredictionSet([(-5.0, 5.0)], 0.5),
            0.9: ContinuousPredictionSet([(-1.0, 1.0)], 0.9),
        }
    )
    # A point covered at the strictest level must be covered at looser ones.
    assert 0.5 in ml[0.9] and 0.5 in ml[0.5] and 0.5 in ml[0.1]
    # A point only in the loose level is not covered at all levels.
    assert 3.0 in ml[0.5]
    assert 3.0 not in ml  # not in the 0.9 level


# ---------------------------------------------------------------------------
# Equality / hashing
# ---------------------------------------------------------------------------
def test_equality_continuous():
    a = ContinuousPredictionSet([(1.0, 2.0), (3.0, 4.0)], 0.1)
    b = ContinuousPredictionSet([(3.0, 4.0), (1.0, 2.0)], 0.1)
    assert a == b  # order-insensitive after normalization
    assert a != ContinuousPredictionSet([(1.0, 2.0)], 0.1)


def test_equality_discrete():
    a = DiscretePredictionSet(np.array([0, 1, 2]), 0.1)
    b = DiscretePredictionSet(np.array([2, 1, 0]), 0.1)
    assert a == b
    assert a != DiscretePredictionSet(np.array([0, 1]), 0.1)


# ---------------------------------------------------------------------------
# Deprecation wrappers
# ---------------------------------------------------------------------------
def test_regressor_interval_wrapper_warns_and_works():
    with pytest.warns(DeprecationWarning):
        cpi = ConformalPredictionInterval(1.0, 5.0, 0.1)
    # Keeps legacy signature + attributes.
    assert cpi.lower == pytest.approx(1.0)
    assert cpi.upper == pytest.approx(5.0)
    assert cpi.width() == pytest.approx(4.0)
    assert 3.0 in cpi
    # And it is a real ContinuousPredictionSet.
    assert isinstance(cpi, ContinuousPredictionSet)


def test_regressor_multilevel_wrapper_warns_and_works():
    inner = ContinuousPredictionSet([(1.0, 2.0)], 0.1)
    with pytest.warns(DeprecationWarning):
        ml = MultiLevelPredictionInterval({0.1: inner})
    assert isinstance(ml, MultiLevelPredictionSet)
    assert ml.levels == [0.1]
    assert 1.5 in ml


def test_classifier_cpset_wrapper_warns_and_works():
    with pytest.warns(DeprecationWarning):
        d = _ClassifierLegacyCPSet(np.array([0, 1, 2]), 0.1)
    assert d.is_discrete
    assert d.width() == 3
    assert 1 in d
    assert isinstance(d, DiscretePredictionSet)
    # Legacy repr preserved.
    assert repr(d) == repr(np.array([0, 1, 2]))


def test_wrapper_nan_empty_preserved():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpi = ConformalPredictionInterval(np.nan, np.nan, 0.1)
    assert cpi.is_empty
    assert cpi.width() == 0.0
    assert 0.0 not in cpi
    assert repr(cpi) == "()"


# ---------------------------------------------------------------------------
# Integration: regressor / classifier / CPS return the new types
# ---------------------------------------------------------------------------
def test_regressor_returns_new_types():
    from online_cp import ConformalRidgeRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (40, 2))
    y = X.sum(axis=1) + rng.normal(0, 0.1, 40)
    m = ConformalRidgeRegressor()
    m.learn_initial_training_set(X, y)

    single = m.predict(X[0], epsilon=0.1)
    assert isinstance(single, ContinuousPredictionSet)

    multi = m.predict(X[0], epsilon=[0.1, 0.2])
    assert isinstance(multi, MultiLevelPredictionSet)
    assert multi.levels == [0.1, 0.2]


def test_classifier_returns_new_types():
    from online_cp import ConformalNearestNeighboursClassifier

    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (40, 2))
    y = (X.sum(axis=1) > 1).astype(int)
    c = ConformalNearestNeighboursClassifier(label_space=[0, 1], rnd_state=1)
    c.learn_initial_training_set(X, y)

    single = c.predict(X[0], epsilon=0.5)
    assert isinstance(single, DiscretePredictionSet)

    multi = c.predict(X[0], epsilon=[0.3, 0.6])
    assert isinstance(multi, MultiLevelPredictionSet)
