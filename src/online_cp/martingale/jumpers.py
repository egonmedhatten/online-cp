"""Jumper and mixture martingale implementations."""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaln, logsumexp

from ..betting import (
    BettingStrategy,
    GaussianKDE,
)
from .base import ConformalTestMartingale


class PluginMartingale(ConformalTestMartingale):
    """Plugin martingale using a betting strategy for density estimation.

    The martingale wraps a ``BettingStrategy`` whose ``bet(p)`` method provides
    the predictive density (betting function) at each step. The strategy is
    updated *after* the bet — predict-then-learn order — preserving the
    martingale property.

    Protocol per step:
    1. **Predict**: evaluate ``strategy.bet(p)`` (uses past data only)
    2. **Accumulate**: ``logM += log(bet)``
    3. **Learn**: call ``strategy.update(p)``
    4. **Expose**: set ``b_n`` and ``B_n`` for the *next* step

    For cautious behaviour during early steps (small sample), use
    :class:`ExpertAggregationStrategy` to mix the primary strategy with a
    uniform baseline, or wrap in a :class:`SleeperStayer`.

    Parameters
    ----------
    betting_strategy : BettingStrategy or type
        An instantiated strategy, or a class to be instantiated with kwargs.
    **kwargs
        Passed to the strategy constructor if a class is given.

    Examples
    --------
    >>> strat = FixedStrategy(pdf=lambda x: 2 if x < 0.5 else 0, check_integration=False)
    >>> m = PluginMartingale(betting_strategy=strat)
    >>> m.update(0.1)
    >>> bool(np.isclose(m.M, 2.0))
    True
    >>> m.update(0.9)
    >>> bool(m.M == 0.0)
    True
    """

    def __init__(
        self,
        betting_strategy: type[BettingStrategy] | BettingStrategy = GaussianKDE,
        store_p_values: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(store_p_values)

        if isinstance(betting_strategy, BettingStrategy):
            self.strategy = betting_strategy
        else:
            betting_kwargs = kwargs if kwargs else {}
            self.strategy = betting_strategy(**betting_kwargs)

        # Expose initial b_n / B_n
        self._update_exposed_functions()

    def _update_exposed_functions(self):
        """Set b_n and B_n for the *next* step (using current strategy state)."""
        strategy_snapshot = deepcopy(self.strategy)

        def b_n(x, _s=strategy_snapshot):
            return _s.bet(x)

        def B_n(x, _s=strategy_snapshot):
            return _s.integrate(x)

        self.b_n = b_n
        self.B_n = B_n

    def update(self, p: float) -> None:
        # 1. Predict: evaluate current betting function
        b = self.strategy.bet(p)

        # 2. Safeguard: ensure b is a valid positive density
        # Betting functions should be > 0 (probability densities/likelihoods)
        if not np.isfinite(b) or b <= 0:
            warnings.warn(
                f"Betting function returned invalid value b={b} for p={p}. "
                f"Using fallback b=1.0 (uniform betting)",
                RuntimeWarning,
                stacklevel=2
            )
            b = 1.0

        # 3. Accumulate wealth
        self.logM += np.log(b)
        self.log_martingale_values.append(self.logM)

        if self.store_p_values:
            self.p_values.append(p)

        # 4. Learn
        self.strategy.update(p)

        # 5. Mark b_n/B_n stale (lazy recomputation on access)
        self._mark_stale()


class SimpleJumper(ConformalTestMartingale):
    """Simple Jumper betting martingale (Algorithm 8.1 of ALRW2).

    Uses a set of experts indexed by epsilon with betting functions
    f_epsilon(p) = 1 + epsilon*(p - 0.5). A Markov chain with jump rate J
    tracks the best expert, enabling adaptation to changing alternatives.

    Parameters
    ----------
    J : float
        Jump rate (probability of switching expert per step).
    E : list of float or None
        Expert grid. Default is [-1, -0.5, 0, 0.5, 1] (Algorithm 8.1).

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, Algorithm 8.1. Cambridge University Press.

    Examples
    --------
    >>> sj = SimpleJumper(J=0.1)
    >>> for _ in range(5):
    ...     sj.update(0.0001)
    >>> bool(sj.M > 1.0)
    True
    """

    def __init__(self, J=0.01, E=None, store_p_values=True, **kwargs):
        super().__init__(store_p_values)
        self.J = J
        if E is None:
            self.E = [-1, -0.5, 0, 0.5, 1]
        else:
            self.E = list(E)
        self._n_experts = len(self.E)
        self.log_C_epsilon = {eps: -np.log(self._n_experts) for eps in self.E}
        self.log_C = 0.0

        self.b_epsilon = lambda u, epsilon: 1 + epsilon * (u - 1 / 2)
        self.B_n_inv = lambda x: x

    def update(self, p: float) -> None:
        if self.store_p_values:
            self.p_values.append(p)

        if self.J == 1:
            log_1_minus_J = -np.inf
        else:
            log_1_minus_J = np.log(1 - self.J)

        log_J_div_E = np.log(self.J / self._n_experts)

        new_log_C_epsilon = {}

        for epsilon in self.E:
            term1 = log_1_minus_J + self.log_C_epsilon[epsilon]
            term2 = log_J_div_E + self.log_C
            log_C_mixed = np.logaddexp(term1, term2)
            bet_val = self.b_epsilon(p, epsilon)
            # Safeguard: betting function must be > 0
            if bet_val <= 0 or not np.isfinite(bet_val):
                warnings.warn(
                    f"Betting function returned invalid value for epsilon={epsilon}, "
                    f"p={p}: b={bet_val}. Using fallback b=1.0",
                    RuntimeWarning,
                    stacklevel=2
                )
                bet_val = 1.0
            new_log_C_epsilon[epsilon] = log_C_mixed + np.log(bet_val)

        self.log_C_epsilon = new_log_C_epsilon
        self.log_C = logsumexp(list(self.log_C_epsilon.values()))

        self.logM = self.log_C
        self.log_martingale_values.append(self.logM)

        # Mark b_n/B_n stale (lazy recomputation on access)
        self._mark_stale()

    def _update_exposed_functions(self):
        """Compute wealth-weighted effective betting function for next step."""
        # b_n(u) = sum_eps w_eps * (1 + eps*(u - 0.5))
        #        = 1 + eps_bar * (u - 0.5)
        # where eps_bar = (1-J) * sum_eps eps * exp(log_C_eps - log_C)
        weights = {eps: np.exp(self.log_C_epsilon[eps] - self.log_C) for eps in self.E}
        epsilon_bar = (1 - self.J) * sum(eps * weights[eps] for eps in self.E)

        self.b_n = lambda u: 1 + epsilon_bar * (u - 1 / 2)
        self.B_n = lambda u: (epsilon_bar / 2) * u**2 + (1 - epsilon_bar / 2) * u
        self.B_n_inv = lambda u: (
            (epsilon_bar - 2) / (2 * epsilon_bar)
            + np.sqrt(epsilon_bar * (8 * u + epsilon_bar - 4) + 4) / (2 * epsilon_bar)
            if abs(epsilon_bar) > 1e-9
            else u
        )


class CompositeJumper(ConformalTestMartingale):
    """Composite Jumper that averages over multiple jump rates.

    Examples
    --------
    >>> cj = CompositeJumper()
    >>> for _ in range(5):
    ...     cj.update(0.001)
    >>> bool(cj.M > 1)
    True
    """

    def __init__(self, J=None, store_p_values=True):
        super().__init__(store_p_values)
        if J is None:
            self.J = [10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1]
        else:
            self.J = J

        self.Jumpers = {j: SimpleJumper(J=j, store_p_values=False) for j in self.J}

    def update(self, p: float) -> None:
        if self.store_p_values:
            self.p_values.append(p)

        for m in self.Jumpers.values():
            m.update(p)

        log_M_values = [m.logM for m in self.Jumpers.values()]
        self.logM = logsumexp(log_M_values) - np.log(len(self.Jumpers))

        self.log_martingale_values.append(self.logM)

        # Mark b_n/B_n stale (lazy recomputation on access)
        self._mark_stale()

    def _update_exposed_functions(self):
        """Compute wealth-weighted betting function from sub-jumpers."""
        log_M_values = [m.logM for m in self.Jumpers.values()]
        log_sum_M = self.logM + np.log(len(self.Jumpers))
        weights = np.exp(np.array(log_M_values) - log_sum_M)

        current_b_ns = [m.b_n for m in self.Jumpers.values()]
        current_B_ns = [m.B_n for m in self.Jumpers.values()]

        self.b_n = lambda u: np.dot(weights, [f(u) for f in current_b_ns])
        self.B_n = lambda u: np.dot(weights, [F(u) for F in current_B_ns])


class SimpleMixtureMartingale(ConformalTestMartingale):
    """Simple Mixture Martingale using the incomplete gamma function.

    This is the canonical "parameter-free" test martingale that averages over
    all power alternatives t^epsilon with epsilon ~ Exp(1). It has a closed-form
    solution based on the regularized incomplete gamma function.

    After each step, ``b_n`` and ``B_n`` are set analytically:
    - b_n(p) = M_n(p) / M_{n-1} where M_n(p) is the martingale value if the
      next observation were p.
    - B_n(p) = integral of b_n from 0 to p.

    Examples
    --------
    >>> sm = SimpleMixtureMartingale()
    >>> sm.update(0.01)
    >>> sm.update(0.01)
    >>> bool(sm.M > 1)
    True
    >>> sm.update(1.0)
    >>> bool(sm.M < sm.martingale_values[-2])
    True
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 0
        self.sum_log_p = 0.0

    def _compute_logM(self, n, sum_log_p):
        """Compute log-martingale value from sufficient statistics."""
        if n == 0:
            return 0.0
        L = sum_log_p
        if np.isclose(L, 0):
            return -np.log(n + 1)
        arg = -L
        log_gamma_n_plus_1 = gammaln(n + 1)
        val_gammainc = gammainc(n + 1, arg)
        if val_gammainc <= 0:
            log_incomplete_gamma = -700  # underflow floor (~1e-304)
        else:
            log_incomplete_gamma = log_gamma_n_plus_1 + np.log(val_gammainc)
        return -L + log_incomplete_gamma - (n + 1) * np.log(arg)

    def _update_exposed_functions(self):
        """Set analytic b_n and B_n for the next step.

        b_n(p) = M(n+1, sum_log_p + log(p)) / M(n, sum_log_p)
        which simplifies to an analytic ratio of incomplete gamma terms.
        """
        n = self.n
        slp = self.sum_log_p
        current_logM = self.logM

        def b_n(p, _n=n, _slp=slp, _logM=current_logM):
            p_c = np.clip(p, 1e-12, 1.0)
            next_logM = self._compute_logM(_n + 1, _slp + np.log(p_c))
            return np.exp(next_logM - _logM)

        def B_n(p, _n=n, _slp=slp, _logM=current_logM):
            # B_n(p) = integral_0^p b_n(t) dt
            # We compute this via numerical integration (the ratio doesn't simplify to closed-form CDF)
            if p <= 1e-12:
                return 0.0
            if p >= 1.0 - 1e-12:
                return 1.0
            val, _ = quad(b_n, 1e-12, p, limit=50)
            return val

        self.b_n = b_n
        self.B_n = B_n

    def update(self, p: float) -> None:
        self.n += 1
        p_clipped = np.clip(p, 1e-12, 1.0)
        self.sum_log_p += np.log(p_clipped)

        self.logM = self._compute_logM(self.n, self.sum_log_p)

        if self.store_p_values:
            self.p_values.append(p)
        self.log_martingale_values.append(self.logM)

        self._mark_stale()


