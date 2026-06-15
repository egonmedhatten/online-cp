"""ConformalTestMartingale base class."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class ConformalTestMartingale:
    """Base class for conformal test martingales.

    A conformal test martingale is a non-negative process starting at 1
    that grows when the exchangeability assumption is violated. It exposes:
    - ``b_n(p)``: the current betting function (density) for the next step
    - ``B_n(p)``: the current protection function (CDF) for the next step
    - ``update(p)``: incorporate a new p-value and advance the martingale

    Attributes
    ----------
    logM : float
        Log martingale value.
    M : float
        Current martingale value (property).
    b_n : callable
        Betting function for the next step.
    B_n : callable
        Protection function (CDF) for the next step.
    """

    def __init__(self, store_p_values: bool = True) -> None:
        self.logM = 0.0
        self.p_values: list[float] = []
        self.store_p_values = store_p_values
        self.log_martingale_values: list[float] = [0.0]
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
        return np.exp(self.logM)

    @property
    def martingale_values(self):
        return np.exp(self.log_martingale_values)

    @property
    def log10_martingale_values(self):
        return [logM / np.log(10) for logM in self.log_martingale_values]

    def update(self, p: float) -> None:
        """Incorporate a new p-value. Subclasses must implement this."""
        raise NotImplementedError


