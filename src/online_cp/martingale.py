"""Conformal test martingales.

This module implements conformal test martingales for testing the exchangeability
assumption online. Martingales combine betting strategies (imported from betting.py)
with various aggregation schemes (Simple Jumper, Sleeper variants, etc.).
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammainc, gammaln, logsumexp
from scipy.stats import norm

from .betting import (
    BettingStrategy,
    BetaKernel,
    BetaMLE,
    BetaMoments,
    ExpertAggregationStrategy,
    FixedStrategy,
    GaussianKDE,
    ParticleFilterStrategy,
    PiecewiseConstantBetting,
)

__all__ = [
    "PluginMartingale",
    "SimpleJumper",
    "CompositeJumper",
    "SimpleMixtureMartingale",
    "SleeperStayer",
    "SleeperDrifter",
    "VilleWrapper",
    "CUSUMWrapper",
    "ShiryaevRobertsWrapper",
    # Re-export betting strategies for backward compat
    "BettingStrategy",
    "BetaKernel",
    "GaussianKDE",
    "BetaMoments",
    "BetaMLE",
    "ParticleFilterStrategy",
    "FixedStrategy",
    "PiecewiseConstantBetting",
    "ExpertAggregationStrategy",
]


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


class SleeperStayer(ConformalTestMartingale):
    """Sleeper/Stayer conformal test martingale (Algorithm 9.4 of ALRW2).

    Maintains a grid of piecewise-constant betting experts indexed by (a, b)
    together with a sleeping capital account. At each step, a fraction R of the
    sleeping capital is redistributed equally to all active experts.

    Each expert uses the betting function f_{(a,b)}(p) = b/a if p <= a, else
    (1-b)/(1-a). This targets change-points where the conformal p-values shift
    from Uniform to having mass b below threshold a.

    Parameters
    ----------
    R : float
        Wake-up rate: fraction of sleeping capital redistributed per step.
    G : int
        Grid resolution. The grid is {1/G, 2/G, ..., (G-1)/G}^2.

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, Algorithm 9.4 (Sleeper). Cambridge University Press.

    Examples
    --------
    >>> ss = SleeperStayer(R=0.01, G=5)
    >>> for _ in range(50):
    ...     ss.update(0.05)
    >>> bool(ss.M > 1.0)
    True
    """

    def __init__(self, R=0.001, G=10, store_p_values=True):
        super().__init__(store_p_values)
        self.R = R
        self.G = G

        # Build the grid: (a, b) pairs with a, b in {1/G, ..., (G-1)/G}
        grid_vals = np.arange(1, G) / G
        self._grid = [(a, b) for a in grid_vals for b in grid_vals]
        self._n_experts = len(self._grid)

        # Precompute betting values for each expert in log-space
        self._left = np.array([b / a for a, b in self._grid])
        self._right = np.array([(1 - b) / (1 - a) for a, b in self._grid])
        self._log_left = np.log(self._left)
        self._log_right = np.log(self._right)
        self._thresholds = np.array([a for a, _ in self._grid])

        # Capital in log-space: all starts in sleeping account
        self._log_S_active = np.full(self._n_experts, -np.inf)  # no capital yet
        self._log_S_sleep = 0.0  # log(1) = 0
        self._log_1_minus_R = math.log(1.0 - R)
        self._log_R_div_n = math.log(R / self._n_experts)
        self._n = 0

    def update(self, p: float) -> None:
        if self.store_p_values:
            self.p_values.append(p)

        self._n += 1

        # Step 1: Bet — add log(bet) to each active expert's log-capital
        log_bets = np.where(p <= self._thresholds, self._log_left, self._log_right)
        self._log_S_active += log_bets

        # Step 2: Output — total capital = exp(log_S_sleep) + sum(exp(log_S_active))
        # Compute in log-space via logsumexp
        log_active_sum = logsumexp(self._log_S_active)
        self.logM = np.logaddexp(self._log_S_sleep, log_active_sum)
        self.log_martingale_values.append(self.logM)

        # Step 3: Redistribute — move fraction R of sleeping capital to active experts
        # log(transfer) = log(R / n_experts) + log_S_sleep
        log_transfer = self._log_R_div_n + self._log_S_sleep
        self._log_S_active = np.logaddexp(self._log_S_active, log_transfer)
        self._log_S_sleep += self._log_1_minus_R

        # Mark b_n/B_n stale (lazy recomputation on access)
        self._mark_stale()

    def _update_exposed_functions(self):
        """Set b_n/B_n as wealth-weighted combination of active experts."""
        log_active_sum = logsumexp(self._log_S_active)
        if np.isfinite(log_active_sum):
            log_weights = self._log_S_active - log_active_sum
            weights = np.exp(log_weights)
            left_vals = self._left
            right_vals = self._right
            thresholds = self._thresholds

            def _b_n(u, w=weights, l=left_vals, r=right_vals, t=thresholds):
                bets = np.where(u <= t, l, r)
                return float(np.dot(w, bets))

            def _B_n(u, w=weights, l=left_vals, r=right_vals, t=thresholds,
                     grid=self._grid):
                # CDF of weighted mixture
                cdfs = np.where(
                    u <= t,
                    l * u,
                    np.array([b + r_i * (u - a) for (a, b), r_i in zip(grid, r)])
                )
                return float(np.dot(w, cdfs))

            self.b_n = _b_n
            self.B_n = _B_n


class SleeperDrifter(ConformalTestMartingale):
    """Sleeper/Drifter conformal test martingale (Algorithm 9.5 of ALRW2).

    Extension of the Sleeper/Stayer that wakes experts in batches every M steps
    and uses a drifting threshold that interpolates between the initial guess a
    and the target b over time.

    The drifting threshold for expert (i, a, b) at step n is:
        a' = (i*M/n)*a + (1 - i*M/n)*b

    This makes the martingale more sensitive to gradual distribution shifts.

    Parameters
    ----------
    R : float
        Wake-up rate per batch: fraction of sleeping capital allocated when
        a new batch wakes up.
    G : int
        Grid resolution. The grid is {1/G, 2/G, ..., (G-1)/G}^2.
    M : int
        Batch interval: new experts wake up every M steps.

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, Algorithm 9.5 (Drifter). Cambridge University Press.

    Examples
    --------
    >>> sd = SleeperDrifter(R=0.01, G=5, M=10)
    >>> for _ in range(50):
    ...     sd.update(0.05)
    >>> bool(sd.M > 1.0)
    True
    """

    def __init__(self, R=0.001, G=10, M=100, store_p_values=True):
        super().__init__(store_p_values)
        self.R = R
        self.G = G
        self._batch_interval = M

        # Grid of (a, b) pairs
        grid_vals = np.arange(1, G) / G
        self._grid = [(a, b) for a in grid_vals for b in grid_vals]
        self._n_grid = len(self._grid)

        # Active experts: dict of (batch_index, grid_index) -> log-capital
        self._experts = {}  # (i, j) -> log_capital
        self._log_S_sleep = 0.0  # log(1) = 0
        self._n = 0
        self._log_prune_threshold = math.log(1e-15)

    def _get_drifted_threshold(self, batch_idx, grid_idx):
        """Compute a' = (i*M/n)*a + (1 - i*M/n)*b for expert (i, a, b)."""
        a, b = self._grid[grid_idx]
        ratio = (batch_idx * self._batch_interval) / self._n
        ratio = min(ratio, 1.0)  # Clamp
        return ratio * a + (1 - ratio) * b

    def update(self, p: float) -> None:
        if self.store_p_values:
            self.p_values.append(p)

        self._n += 1

        # Step 1: Bet — update all active experts in log-space
        keys_to_remove = []
        for key, log_capital in self._experts.items():
            batch_idx, grid_idx = key
            a_prime = self._get_drifted_threshold(batch_idx, grid_idx)
            _, b = self._grid[grid_idx]

            # Bet using f_{(a', b)}(p)
            if a_prime <= 0 or a_prime >= 1:
                log_bet_val = 0.0  # log(1) = no bet
            else:
                bet_val = b / a_prime if p <= a_prime else (1 - b) / (1 - a_prime)
                log_bet_val = math.log(bet_val)

            new_log_capital = log_capital + log_bet_val
            if new_log_capital < self._log_prune_threshold:
                keys_to_remove.append(key)
            else:
                self._experts[key] = new_log_capital

        for key in keys_to_remove:
            del self._experts[key]

        # Step 2: Output — total capital in log-space
        if self._experts:
            log_active_sum = logsumexp(list(self._experts.values()))
            self.logM = np.logaddexp(self._log_S_sleep, log_active_sum)
        else:
            self.logM = self._log_S_sleep
        self.log_martingale_values.append(self.logM)

        # Step 3: Wake new batch (if n is divisible by M) — prepares for next step
        if self._n % self._batch_interval == 0:
            batch_idx = self._n // self._batch_interval
            # log(transfer_per_expert) = log(R * M) + log_S_sleep - log(n_grid)
            log_transfer = (math.log(self.R * self._batch_interval)
                           + self._log_S_sleep
                           - math.log(self._n_grid))
            # S_sleep *= (1 - R*M), clamped to avoid negative
            rm = self.R * self._batch_interval
            if rm >= 1.0:
                self._log_S_sleep = -np.inf
            else:
                self._log_S_sleep += math.log(1.0 - rm)
            for j in range(self._n_grid):
                key = (batch_idx, j)
                if key in self._experts:
                    self._experts[key] = np.logaddexp(self._experts[key], log_transfer)
                else:
                    self._experts[key] = log_transfer

        # Mark b_n/B_n stale (lazy recomputation on access)
        self._mark_stale()

    def _update_exposed_functions(self):
        """Set b_n/B_n as wealth-weighted function for next step."""
        if self._experts:
            log_vals = list(self._experts.values())
            log_total_active = logsumexp(log_vals)
            expert_items = list(self._experts.items())
            log_total = log_total_active
            n_next = self._n + 1

            def _b_n(u, _items=expert_items, _log_total=log_total, _n=n_next,
                     _grid=self._grid, _M=self._batch_interval):
                val = 0.0
                for (bi, gi), log_cap in _items:
                    a, b = _grid[gi]
                    ratio = min((bi * _M) / _n, 1.0)
                    a_prime = ratio * a + (1 - ratio) * b
                    if 0 < a_prime < 1:
                        f = b / a_prime if u <= a_prime else (1 - b) / (1 - a_prime)
                    else:
                        f = 1.0
                    val += math.exp(log_cap - _log_total) * f
                return val

            def _B_n(u, _items=expert_items, _log_total=log_total, _n=n_next,
                     _grid=self._grid, _M=self._batch_interval):
                if u <= 0.0:
                    return 0.0
                if u >= 1.0:
                    return 1.0
                val = 0.0
                for (bi, gi), log_cap in _items:
                    a, b = _grid[gi]
                    ratio = min((bi * _M) / _n, 1.0)
                    a_prime = ratio * a + (1 - ratio) * b
                    w = math.exp(log_cap - _log_total)
                    if 0 < a_prime < 1:
                        if u <= a_prime:
                            val += w * (b / a_prime) * u
                        else:
                            val += w * (b + (1 - b) / (1 - a_prime) * (u - a_prime))
                    else:
                        val += w * u
                return val

            self.b_n = _b_n
            self.B_n = _B_n


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


class VilleWrapper:
    """Ville's inequality procedure for change-point detection.

    The simplest test based on a conformal test martingale: reject the
    exchangeability hypothesis when the running maximum of the martingale
    exceeds a threshold c. By Ville's inequality:

        P(∃n : S_n >= c) <= 1/c

    So threshold c = 20 gives a 5% significance level, c = 100 gives 1%, etc.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        The underlying martingale to wrap.
    threshold : float
        Default alarm threshold (default 20, i.e. 5% significance).

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §8.4.1 (The Ville Procedure). Cambridge University Press.

    Examples
    --------
    >>> from online_cp.martingale import SimpleJumper, VilleWrapper
    >>> sj = SimpleJumper(J=0.1)
    >>> ville = VilleWrapper(sj, threshold=20)
    >>> for _ in range(10):
    ...     ville.update(0.5)
    >>> bool(ville.rejected)
    False
    """

    def __init__(self, martingale, threshold=20):
        self.martingale = martingale
        self.threshold = threshold
        self._log_max = 0.0
        self._n = 0
        self._rejection_time = None

    @property
    def log_max(self):
        """Log of the running maximum of the martingale."""
        return self._log_max

    @property
    def max(self):
        """Running maximum of the martingale."""
        return np.exp(self._log_max)

    @property
    def rejected(self):
        """Whether the exchangeability hypothesis has been rejected."""
        return self._log_max >= np.log(self.threshold)

    @property
    def rejection_time(self):
        """Step at which the hypothesis was first rejected, or None."""
        return self._rejection_time

    def update(self, p: float) -> None:
        """Update the inner martingale and track the running maximum."""
        self.martingale.update(p)
        self._n += 1
        if self.martingale.logM > self._log_max:
            self._log_max = self.martingale.logM
        if self._rejection_time is None and self._log_max >= np.log(self.threshold):
            self._rejection_time = self._n

    def alarm(self, threshold=None):
        """Check whether max(S_n) exceeds the threshold.

        Parameters
        ----------
        threshold : float or None
            Override threshold. If None, uses the threshold set at construction.

        Returns
        -------
        bool
            True if the running maximum exceeds the threshold.
        """
        t = threshold if threshold is not None else self.threshold
        return self._log_max >= np.log(t)


class CUSUMWrapper:
    """CUSUM change-detection wrapper for any conformal test martingale.

    Computes the Page CUSUM statistic as the ratio of the current martingale
    value to its running minimum:

        gamma_n = S_n / min_{i <= n} S_i

    In log-space: log(gamma_n) = logM_n - min_{i <= n} logM_i

    This removes any accumulated "debt" from an initial in-control period,
    giving faster detection after the change-point. Optionally accepts a linear
    barrier for controlling the false alarm rate over long horizons.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        The underlying martingale to wrap.
    barrier_slope : float or None
        If not None, the alarm threshold grows linearly as barrier_slope * n.

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §8.3. Cambridge University Press.

    Examples
    --------
    >>> from online_cp.martingale import SimpleJumper, CUSUMWrapper
    >>> sj = SimpleJumper(J=0.01)
    >>> cusum = CUSUMWrapper(sj)
    >>> for _ in range(10):
    ...     cusum.update(0.5)
    >>> bool(cusum.gamma >= 1.0)  # gamma is always >= 1 (since S_n >= min S_i is not guaranteed)
    True
    """

    def __init__(self, martingale, barrier_slope=None):
        self.martingale = martingale
        self.barrier_slope = barrier_slope
        self._log_min = 0.0  # log(S_0) = 0
        self._n = 0
        self._log_gamma_values = [0.0]

    @property
    def gamma(self):
        """Current CUSUM statistic."""
        return np.exp(self._log_gamma_values[-1])

    @property
    def log_gamma(self):
        """Current log CUSUM statistic."""
        return self._log_gamma_values[-1]

    @property
    def cusum_values(self):
        """All CUSUM statistic values."""
        return np.exp(self._log_gamma_values)

    @property
    def log_cusum_values(self):
        """All log CUSUM statistic values."""
        return self._log_gamma_values

    def update(self, p: float) -> None:
        """Update the inner martingale and recompute the CUSUM statistic."""
        self.martingale.update(p)
        self._n += 1

        logM = self.martingale.logM
        if logM < self._log_min:
            self._log_min = logM

        log_gamma = logM - self._log_min
        self._log_gamma_values.append(log_gamma)

    def alarm(self, threshold):
        """Check whether gamma_n exceeds the threshold (optionally with barrier).

        Parameters
        ----------
        threshold : float
            The alarm threshold. If barrier_slope is set, the effective
            threshold at step n is threshold + barrier_slope * n.

        Returns
        -------
        bool
            True if gamma_n exceeds the (possibly time-varying) threshold.
        """
        effective = threshold
        if self.barrier_slope is not None:
            effective = threshold + self.barrier_slope * self._n
        return self.gamma > effective


class ShiryaevRobertsWrapper:
    """Shiryaev-Roberts change-detection wrapper for any conformal test martingale.

    Computes the Shiryaev-Roberts statistic as:

        R_n = sum_{i=1}^{n} S_n / S_i

    In log-space: R_n = sum_{i=1}^{n} exp(logM_n - logM_{i-1})

    This is always >= the CUSUM statistic (sum >= max), giving a slightly
    different power/false-alarm trade-off.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        The underlying martingale to wrap.

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §8.3. Cambridge University Press.

    Examples
    --------
    >>> from online_cp.martingale import SimpleJumper, ShiryaevRobertsWrapper
    >>> sj = SimpleJumper(J=0.01)
    >>> sr = ShiryaevRobertsWrapper(sj)
    >>> for _ in range(10):
    ...     sr.update(0.5)
    >>> sr.R >= 0
    True
    """

    def __init__(self, martingale):
        self.martingale = martingale
        self._n = 0
        self._sr_values = [0.0]  # R_0 = 0 (no terms in the sum)
        self._prev_logM = 0.0  # logM_{n-1}, starts at 0

    @property
    def R(self):
        """Current Shiryaev-Roberts statistic."""
        return self._sr_values[-1]

    @property
    def sr_values(self):
        """All Shiryaev-Roberts statistic values."""
        return self._sr_values

    def update(self, p: float) -> None:
        """Update the inner martingale and recompute the SR statistic.

        Uses the O(1) recursive formula (eq. 8.18 of ALRW2):
            R_n = (S_n / S_{n-1}) * (R_{n-1} + 1)
        """
        self.martingale.update(p)
        self._n += 1

        logM_n = self.martingale.logM
        # S_n / S_{n-1} = exp(logM_n - logM_{n-1})
        ratio = np.exp(logM_n - self._prev_logM)
        R_n = ratio * (self._sr_values[-1] + 1)
        self._sr_values.append(float(R_n))
        self._prev_logM = logM_n

    def alarm(self, threshold):
        """Check whether R_n exceeds the threshold.

        Parameters
        ----------
        threshold : float
            The alarm threshold.

        Returns
        -------
        bool
            True if R_n exceeds the threshold.
        """
        return self.R > threshold


