"""ConformalTestMartingale base class."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class ConformalTestMartingale:
    r"""Base class for conformal test martingales.

    A conformal test martingale is a non-negative process $M_0 = 1, M_1, M_2,
    \dots$ that is a martingale under the hypothesis that the input p-values are
    i.i.d. uniform on $[0, 1]$ — equivalently, under exchangeability of the data
    stream. It *grows* when that hypothesis is violated, so a large value is
    evidence against exchangeability and can be used for online change-point
    detection.

    Validity follows from **Ville's inequality**: for a non-negative martingale
    started at $1$,

    $$
    \mathbb{P}\bigl(\sup_n M_n \geq c\bigr) \leq 1 / c ,
    $$

    so rejecting exchangeability when $M_n \geq 1/\alpha$ controls the false-alarm
    probability at level $\alpha$ uniformly over time (an *anytime-valid* test).
    Each step multiplies the martingale by a betting function $b_n(p)$ (a density
    on $[0, 1]$ with mean 1); see :mod:`online_cp.betting`.

    It exposes:

    - ``update(p)``: template method that incorporates a new p-value and
      advances the martingale (do not override; implement ``_step`` instead)
    - ``M``: the current martingale value (property, $\exp(\texttt{logM})$)
    - ``logM``: the log martingale value (attribute)
    - ``log_martingale_values``: list of $\log M_0, \log M_1, \ldots, \log M_n$
    - ``b_n(p)``: the current betting function (density) for the next step
    - ``B_n(p)``: the current protection function (CDF) for the next step
    - ``p_n``: the last p-value observed (``None`` before the first update)
    - ``n``: the number of p-values processed so far

    To define a custom martingale, subclass this class and implement
    :meth:`_step` (required) and :meth:`_update_exposed_functions` (optional).
    See the :meth:`update` docstring for a minimal working example.

    Attributes
    ----------
    logM : float
        Log martingale value.
    M : float
        Current martingale value (property).
    log_martingale_values : list of float
        All log martingale values $\log M_0, \log M_1, \ldots$ (always recorded).
    p_values : list of float
        All p-values observed (recorded only when ``store_p_values=True``).
    b_n : callable
        Betting function for the next step.
    B_n : callable
        Protection function (CDF) for the next step.
    p_n : float or None
        Last p-value observed (always available).
    n : int
        Number of p-values processed.
    """

    def __init__(self, store_p_values: bool = True) -> None:
        self.logM = 0.0
        self.p_values: list[float] = []
        self.store_p_values = store_p_values
        self.log_martingale_values: list[float] = [0.0]
        self._last_p: float | None = None
        self._b_n_cache = lambda x: 1.0
        self._B_n_cache = lambda x: x
        self._B_n_inv_cache = None
        self._b_n_stale = False

    def _mark_stale(self):
        """Invalidate all cached betting/protection functions."""
        self._b_n_stale = True
        self._B_n_inv_cache = None

    @property
    def b_n(self):
        """Betting function for the next step (lazily computed)."""
        if self._b_n_stale:
            self._B_n_inv_cache = None
            self._update_exposed_functions()
            self._b_n_stale = False
        return self._b_n_cache

    @b_n.setter
    def b_n(self, value):
        self._b_n_cache = value
        self._b_n_stale = False

    @property
    def B_n(self):
        """Protection function (CDF) for the next step (lazily computed)."""
        if self._b_n_stale:
            self._B_n_inv_cache = None
            self._update_exposed_functions()
            self._b_n_stale = False
        return self._B_n_cache

    @B_n.setter
    def B_n(self, value):
        self._B_n_cache = value
        self._b_n_stale = False

    @property
    def B_n_inv(self):
        """Inverse protection function (lazily computed with numerical fallback)."""
        if self._b_n_stale:
            self._B_n_inv_cache = None
            self._update_exposed_functions()
            self._b_n_stale = False
        if self._B_n_inv_cache is None:
            # Numerical fallback: invert the current B_n via brentq
            B_n_frozen = self._B_n_cache

            def _inv(x, _B=B_n_frozen):
                if x <= 0.0:
                    return 0.0
                if x >= 1.0:
                    return 1.0
                return brentq(lambda u: _B(u) - x, 0.0, 1.0)

            self._B_n_inv_cache = _inv
        return self._B_n_inv_cache

    @B_n_inv.setter
    def B_n_inv(self, value):
        self._B_n_inv_cache = value

    def _update_exposed_functions(self):
        """Compute and cache b_n/B_n. Subclasses override this."""
        pass

    @property
    def M(self):
        """Current martingale value $M_n = \\exp(\\log M_n)$."""
        return float(np.exp(self.logM))

    @property
    def martingale_values(self):
        return np.exp(self.log_martingale_values)

    @property
    def log10_martingale_values(self):
        return [logM / np.log(10) for logM in self.log_martingale_values]

    @property
    def p_n(self):
        """Last p-value observed, or ``None`` if no updates have occurred.

        This is available regardless of the ``store_p_values`` flag because
        the base template method always records the most recent p-value.
        When ``store_p_values=True``, this is equivalent to
        ``self.p_values[-1]``.
        """
        return self._last_p

    @property
    def n(self):
        """Number of p-values processed so far (i.e. the current step index).

        At construction time ``n == 0``. After the first ``update(p)`` call,
        ``n == 1``, and so on.
        """
        return len(self.log_martingale_values) - 1

    def update(self, p: float) -> None:
        r"""Incorporate a new p-value and advance the martingale by one step.

        This is the **template method** — do not override it in subclasses.
        It performs the shared bookkeeping (storing the p-value, recording the
        martingale value, invalidating cached functions) and then delegates to
        the subclass's :meth:`_step` for the core mathematical update.

        Protocol
        --------
        1. Record ``p`` as the most recent p-value (always; enables the
           :attr:`p_n` property).
        2. Append ``p`` to ``self.p_values`` (only if ``store_p_values``
           is True).
        3. Call ``self._step(p)`` — the subclass updates its internal state
           and sets ``self.logM``.
        4. Append ``self.logM`` to ``self.log_martingale_values``.
        5. Invalidate the cached betting/protection functions.

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.

        Examples
        --------
        Here is a minimal custom martingale using only a fixed betting
        function $b(p) = 1 + 2\,(p - 0.5) = 2p$ (a valid density on $[0,1]$):

        >>> import numpy as np
        >>> class MyMartingale(ConformalTestMartingale):
        ...     def _step(self, p: float) -> None:
        ...         # Multiply wealth by the betting function evaluated at p.
        ...         self.logM += np.log(2.0 * p)
        ...     def _update_exposed_functions(self):
        ...         self.b_n = lambda x: 2.0 * x
        ...         self.B_n = lambda x: x * x
        >>> m = MyMartingale()
        >>> m.M
        1.0
        >>> m.update(0.25)
        >>> bool(np.isclose(m.M, 0.5))
        True
        >>> m.p_n
        0.25
        >>> m.n
        1
        """
        self._last_p = p
        if self.store_p_values:
            self.p_values.append(p)
        self._step(p)
        self.log_martingale_values.append(self.logM)
        self._mark_stale()

    def _step(self, p: float) -> None:
        r"""Core mathematical update — **subclasses must implement this**.

        This is the only method a minimal custom martingale *must* define
        (along with :meth:`_update_exposed_functions` if it wants to expose
        betting/protection functions). The method receives the new p-value
        ``p`` and must update the subclass's internal state so that
        ``self.logM`` reflects the martingale value after incorporating
        ``p``.

        The subclass is responsible for:

        - Updating whatever internal state it maintains (expert weights,
          running sums, polynomial coefficients, etc.).
        - Setting ``self.logM`` to the new log-martingale value.

        The subclass should **not** touch ``self.p_values``,
        ``self.log_martingale_values``, or call ``self._mark_stale()`` —
        those are handled by the base template method :meth:`update`.

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.
        """
        raise NotImplementedError
