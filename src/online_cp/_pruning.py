"""Budget-bounded stream pruning for online conformal learning.

This module implements the pruning strategies that preserve **exchangeability**,
and therefore the exact finite-sample coverage guarantee.

The validity criterion
----------------------
A pruning strategy is *strictly valid* iff the retained training bag is a
**bag-function**: a symmetric (permutation-invariant) function of the stream's
*values*, possibly randomised with randomness **independent of the data values**.

Why: the exact coverage guarantee at step ``n`` holds provided the new example is
*exchangeable with the other points in the bag it is bagged against* -- i.e. its
nonconformity rank among the bag is uniform. Under the library's stream-
exchangeability assumption (the whole stream is exchangeable), this holds exactly
when the retained bag is a value-independent (data-blind) sub-selection of the
stream. Any selection that *inspects the data values* -- stratification by
label, importance by support-vector / leverage / LOO p-value, geometry by
distance, "keep rare classes" -- makes the new example's *membership in the bag*
correlated with a value-dependent variable, which breaks that conditional
exchangeability. Such strategies are **not** valid and are **not** implemented
(see ``design/pruning.md``, "Invalid strategies (reference only)").

The two built-in strictly-valid strategies are:

- **Reservoir** (``"reservoir"``) -- a uniform random sample of ``budget``
  examples from the stream (Vitter's Algorithm R). Data-blind.
- **Window** (``"window"``) -- the most recent ``budget`` examples (FIFO).
  Data-blind (order-based; valid under the stream-exchangeability assumption).

Both *expand* to ``budget`` examples as the stream arrives and then hold the bag
at size ``budget`` (the window slides; the reservoir keeps a random subset).
With ``budget >= n`` seen, behaviour is *bit-identical* to no pruning.

Custom strategies
-----------------
A user may supply their own strategy as a subclass of :class:`PruningStrategy`
(or a lightweight object exposing ``decide(n_seen)``). The library enforces the
data-blind contract at the API boundary via :func:`validate_strategy`: a
strategy is accepted only if it does **not** receive the data (``X``, ``y``,
``D``, ``K``, the model, or the new point) -- the only arguments it may use are
the stream count, the budget, and an RNG. See ``design/pruning.md`` for the full
contract.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.random as _npr


# ---------------------------------------------------------------------------
# Strategy base classes
# ---------------------------------------------------------------------------


class PruningStrategy:
    """User-facing base class for a budget-bounded stream-pruning strategy.

    A strategy is *strictly valid* iff it is **data-blind**: its accept/evict
    decision is a function only of (i) the number of examples seen so far,
    (ii) the budget, and (iii) an RNG independent of the data. It must *not*
    inspect the feature values ``X``, the labels ``y``, the distance/kernel
    matrices ``D``/``K``, the fitted model state, or the new point. This is
    exactly the condition for the retained bag to be a bag-function of the
    exchangeable stream, which is what preserves the conformal guarantee.

    Subclass this and implement :meth:`decide` to add a custom *valid* strategy.
    The model calls ``decide(n_seen)`` at each ``learn_one`` (where ``n_seen``
    includes the just-arrived example). The model owns ``n_seen`` (so it
    serialises cleanly). At construction the model injects its own ``budget`` and
    ``rnd_gen`` into the strategy, overriding whatever you passed, so the
    strategy's draws stay on the model's (serialized) RNG for reproducibility.
    Pass the budget you intend to use and any RNG (the model will reconcile
    both).

    Example
    -------
    >>> import numpy as np
    >>> from online_cp._pruning import PruningStrategy
    >>> class EveryKth(PruningStrategy):
    ...     def __init__(self, budget, rnd_gen, period=3):
    ...         super().__init__(budget, rnd_gen)
    ...         self.period = period
    ...     def decide(self, n_seen):
    ...         if n_seen <= self.budget:
    ...             return -1
    ...         return (n_seen // self.period) % self.budget
    # Pass the intended budget and any RNG; the model overrides both.
    >>> strat = EveryKth(1000, np.random.default_rng(0), period=3)
    """

    def __init__(self, budget: int, rnd_gen: _npr.Generator) -> None:
        if budget is None or int(budget) < 1:
            raise ValueError(f"budget must be a positive integer, got {budget!r}")
        self.budget = int(budget)
        self.rnd_gen = rnd_gen

    def decide(self, n_seen: int) -> int | None:
        """Accept/evict decision for the example that brings the stream to
        ``n_seen`` examples in total (``n_seen`` counts the new example).

        Returns
        -------
        - ``-1`` (int)  -- *keep*, evict nothing (bag not yet full).
        - ``i >= 0``    -- *keep* the new example; evict the existing element at
          position ``i`` (``0 <= i < budget``).
        - ``None``      -- *discard* the new example (state unchanged).

        The default implementation is reservoir sampling (Vitter's Algorithm R);
        override for a different (data-blind) rule.
        """
        return _reservoir_decide(n_seen, self.budget, self.rnd_gen)


class _ReservoirBuffer(PruningStrategy):
    """Built-in reservoir-sampling strategy (Vitter's Algorithm R).

    Data-blind: the accept/evict draws use only the stream count and the RNG.
    For ``n_seen > budget`` the new example is kept with probability
    ``budget / n_seen``; when kept, the evicted position is uniform over
    ``{0, ..., budget - 1}``. For ``n_seen <= budget`` every example is kept.
    """

    def decide(self, n_seen: int) -> int | None:
        return _reservoir_decide(n_seen, self.budget, self.rnd_gen)


class _WindowBuffer(PruningStrategy):
    """Built-in sliding-window (FIFO) strategy.

    Data-blind: the decision depends only on the stream count. The new example
    is kept unconditionally; when the bag is already full (``n_seen > budget``)
    the oldest element (position ``0``) is evicted. For ``n_seen <= budget``
    every example is kept (the window *expands* to ``budget`` then slides).
    """

    def decide(self, n_seen: int) -> int | None:
        if n_seen <= self.budget:
            return -1
        return 0


def _reservoir_decide(n_seen: int, budget: int, rnd_gen: _npr.Generator) -> int | None:
    """Vitter's Algorithm R decision (module-level helper, used by the default
    :class:`PruningStrategy` and :class:`_ReservoirBuffer`)."""
    if n_seen <= budget:
        return -1
    j = int(rnd_gen.integers(0, n_seen))  # uniform on {0, ..., n_seen-1}
    if j < budget:
        return int(j)
    return None


# ---------------------------------------------------------------------------
# Strategy registry & validation
# ---------------------------------------------------------------------------

#: The built-in, strictly-valid strategies and their constructors.
_BUILTIN_STRATEGIES: dict[str, type[PruningStrategy]] = {
    "reservoir": _ReservoirBuffer,
    "window": _WindowBuffer,
}


def validate_strategy(pruning) -> None:
    """Validate that ``pruning`` is a strictly-valid (data-blind) strategy.

    A strictly-valid strategy is a **bag-function**: a symmetric, value-blind
    function of the stream, randomised only by data-independent randomness. The
    library cannot *prove* non-value-dependence, so it enforces the practical
    necessary condition: the strategy must not receive the data.

    - A string name is valid iff it is a built-in (``"reservoir"``,
      ``"window"``); other names raise.
    - A :class:`PruningStrategy` instance is accepted (its contract is its
      docstring; users are responsible for data-blindness).
    - Any other object is accepted iff it exposes a ``decide`` method whose
      parameters are drawn *only* from the data-blind set ``{n_seen, budget,
      rng, rnd_gen, period, k (keyword-only)}``. Any other parameter name
      (e.g. ``x``, ``y``, ``model``, ``d``, ``alpha``, ``leverage``, ``point``,
      ``example``, ``bag``, ``labels``, ``features``, ``data``, ``z``,
      ``train``, or ``**kwargs``) is rejected with ``ValueError``.

    This is a *heuristic necessary-condition* guard, not a proof. The
    authoritative contract is the docstring of :class:`PruningStrategy`.

    Raises
    ------
    ValueError
        If ``pruning`` is not a recognised valid strategy.
    """
    if isinstance(pruning, str):
        if pruning not in _BUILTIN_STRATEGIES:
            raise ValueError(
                f"pruning must be 'reservoir' or 'window' (or a PruningStrategy "
                f"/ callable); the string {pruning!r} is not a built-in strategy. "
                f"Model-state / value-dependent selectors break exchangeability "
                f"and are not supported."
            )
        return
    if isinstance(pruning, PruningStrategy):
        # Trust the subclass contract (documented); the guard is a no-op here.
        return
    # Duck-typing: accept any object exposing a data-blind `decide` method
    # (whether or not the object itself is callable).
    target = getattr(pruning, "decide", None)
    if target is None:
        raise ValueError(
            f"pruning must be a strategy name, a PruningStrategy, or an object "
            f"with a decide(n_seen) method; {pruning!r} has no 'decide'."
        )
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        raise ValueError(
            f"Cannot introspect the signature of {pruning!r}.decide; "
            "provide a plain Python function with an inspectable signature."
        ) from None
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in ("n_seen", "budget", "rng", "rnd_gen", "period"):
            continue
        if name == "k":
            # 'k' is ambiguous (neighbour count vs. period); allow only if
            # keyword-only or defaulted.
            if param.default is inspect.Parameter.empty and param.kind != inspect.Parameter.KEYWORD_ONLY:
                raise ValueError(
                    "Parameter 'k' is ambiguous; rename it to 'period' or "
                    "make it keyword-only to be accepted as a data-blind "
                    "hyper-parameter."
                )
            continue
        raise ValueError(
            f"Pruning strategy {pruning!r} inspects the data via parameter "
            f"'{name}', which breaks exchangeability. A strictly-valid "
            "strategy must be data-blind: its decision may depend only on "
            "the stream count, the budget, and a data-independent RNG."
        )
    return


def _make_strategy_buffer(pruning, budget: int, rnd_gen: _npr.Generator) -> PruningStrategy:
    """Build the strategy buffer from a ``pruning`` argument.

    ``pruning`` is one of:
    - ``"reservoir"`` or ``"window"`` (built-in strategies);
    - a subclass/instance of :class:`PruningStrategy`;
    - a callable (object) exposing ``decide(n_seen)``.

    Returns a :class:`PruningStrategy` instance. Raises ``ValueError`` if the
    strategy is not recognised or fails the data-blind contract (see
    :func:`validate_strategy`).
    """
    if isinstance(pruning, str):
        try:
            cls = _BUILTIN_STRATEGIES[pruning]
        except KeyError:
            raise ValueError(
                f"pruning must be 'reservoir', 'window', a PruningStrategy "
                f"subclass/instance, or an object with decide(n_seen); "
                f"got {pruning!r}. (Model-state / value-dependent selectors "
                f"break exchangeability and are not supported.)"
            ) from None
        return cls(budget, rnd_gen)
    if isinstance(pruning, PruningStrategy):
        # The model's budget and RNG are authoritative; reconcile the strategy's
        # caches so decide()'s `self.budget` matches the bag the model actually
        # maintains, and so the strategy draws from the model's (serialized)
        # RNG for full reproducibility across save/load.
        pruning.budget = int(budget)
        pruning.rnd_gen = rnd_gen
        return pruning
    # Duck-typed strategy object (validates as data-blind).
    validate_strategy(pruning)

    class _Wrapped(PruningStrategy):
        def __init__(self, budget, rnd_gen, strategy):
            super().__init__(budget, rnd_gen)
            self._strategy = strategy

        def decide(self, n_seen):
            return self._strategy.decide(n_seen)

    return _Wrapped(budget, rnd_gen, pruning)


# ---------------------------------------------------------------------------
# Eviction helpers (model-specific state surgery)
# ---------------------------------------------------------------------------


def _evict_distance_matrix(D, i: int):
    """Remove row ``i`` *and* column ``i`` from a square distance matrix ``D``.

    Returns the (n-1, n-1) matrix with the ``i``-th row and column removed, which
    keeps the remaining rows/columns in their original relative order.
    """
    n = D.shape[0]
    keep = np.ones(n, dtype=bool)
    keep[i] = False
    return D[np.ix_(keep, keep)].copy()


def _evict_masked_distance(D, keep):
    """Return the sub-matrix of ``D`` restricted to the rows/cols in ``keep``.

    ``keep`` is a 1-D boolean array (or an integer position array) of length /
    elements matching ``D``'s first axis. The result is the square matrix on the
    retained positions, preserving their original relative order.
    """
    keep = np.asarray(keep)
    if keep.dtype == bool:
        idx = np.flatnonzero(keep)
    else:
        idx = np.sort(keep)
    return D[np.ix_(idx, idx)].copy()
