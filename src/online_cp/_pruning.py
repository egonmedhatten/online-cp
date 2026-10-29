"""Reservoir-sampling stream pruning for budget-bounded online conformal learning.

This module implements the two exchangeability-preserving stream-pruning
strategies: **reservoir sampling** (Vitter's Algorithm R) and a **sliding
window** (FIFO). Both keep a value-based, order-invariant sub-selection of the
stream, so the retained bag is exchangeable with the full stream and the
conformal guarantee is maintained unchanged.

- **Reservoir** keeps a uniform random sample of size ``budget``.
- **Window** keeps the most recent ``budget`` examples.

A pruning strategy preserves exchangeability -- and hence the exact
finite-sample coverage guarantee -- whenever its retention rule is a function
of the *values* of the stream that is *order-invariant*. Both reservoir
sampling and the sliding window satisfy this. What *does* break exchangeability
are model-state-based selectors (SV-informed, leverage, core-set, self-pruning),
whose selection correlates with the test point's own nonconformity.

Design
------
The :class:`_ReservoirBuffer` is a thin, model-agnostic controller. It is
stateless with respect to the stream count (the model owns ``n_seen`` so it
serialises cleanly). At each ``learn_one``, ``decide(n_seen, strategy)``
reports whether the just-arrived example is *kept* or *discarded*:

- reservoir: kept with probability ``budget / n_seen`` (Vitter's Algorithm R);
  when kept, it also reports which existing position (``0 .. budget - 1``) was
  evicted, so the model can drop that row/column from its own state;
- window: kept unconditionally; when the bag is full it evicts position ``0``
  (the oldest element);
- discarded otherwise (state unchanged).

The eviction mechanic is model-specific (k-NN drops row/col ``i`` of the
distance matrix and row ``i`` of ``X``/``y``; the classifier additionally
rebuilds ``_label_indices``; kernel-ridge recomputes ``Kinv`` from scratch at
size ``budget``; the SVM drops row/col ``i`` of the Gram matrix), so the model
applies the eviction itself. The buffer owns only the accept/evict decision.

Reproducibility: all randomness is drawn from the model's own ``rnd_gen``
(``np.random.Generator``), which is already serialised via ``bit_generator.state``
(see the save/load machinery), so the reservoir is fully reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.random as _npr


class _ReservoirBuffer:
    """Pruning budget controller (stateless w.r.t. the stream count).

    The model owns the running count of examples seen (``n_seen``) so it can be
    serialised/restored trivially; the buffer only needs the budget and the RNG.
    ``decide(n_seen, strategy)`` reports the accept/evict decision for the
    *just-arrived* example, given that ``n_seen`` (including it) is the total
    seen so far. ``strategy`` is ``"reservoir"`` (default, random eviction) or
    ``"window"`` (FIFO eviction of the oldest element).

    Parameters
    ----------
    budget : int
        Maximum number of examples retained (the reservoir / window size).
    rnd_gen : numpy.random.Generator
        The model's random generator (advanced in place; serialised with the
        model). Used for the accept/evict draws in reservoir mode (unused by
        the window, which is deterministic).
    """

    def __init__(self, budget: int, rnd_gen: "_npr.Generator") -> None:
        if budget is None or int(budget) < 1:
            raise ValueError(f"budget must be a positive integer, got {budget!r}")
        self.budget = int(budget)
        self.rnd_gen = rnd_gen

    def decide(self, n_seen: int, strategy: str = "reservoir") -> int | None:
        """Accept/evict decision for the example that brings the stream to
        ``n_seen`` examples in total (``n_seen`` counts the new example).

        Returns
        -------
        - ``-1`` (int)  -- *keep*, evict nothing (reservoir not yet full).
        - ``i >= 0``    -- *keep* the new example; evict the existing element at
          position ``i`` (``0 <= i < budget``).
        - ``None``      -- *discard* the new example (state unchanged).

        Two strategies:

        - ``strategy="reservoir"`` (default): Vitter's Algorithm R. For
          ``n_seen > budget`` the new example is kept with probability
          ``budget / n_seen``; when kept, the evicted position is uniform over
          ``{0, ..., budget - 1}``. For ``n_seen <= budget`` every example is
          kept.

        - ``strategy="window"``: keep the new example unconditionally; when the
          bag is already full (``n_seen > budget``) evict position ``0`` (the
          oldest element). For ``n_seen <= budget`` every example is kept. The
          window is a value-based, order-invariant sub-selection of the
          exchangeable stream, so it preserves exchangeability and the exact
          finite-sample coverage guarantee (less robust on drifting streams,
          see ``design/pruning.md``).
        """
        if n_seen <= self.budget:
            return -1  # reservoir not yet full: keep, no eviction
        if strategy == "window":
            return 0  # window: always keep the new example; evict the oldest
        # Reservoir at capacity. Keep with prob budget / n_seen.
        j = int(self.rnd_gen.integers(0, n_seen))  # uniform on {0, ..., n_seen-1}
        if j < self.budget:
            return int(j)  # keep the new example; evict position j
        return None


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
