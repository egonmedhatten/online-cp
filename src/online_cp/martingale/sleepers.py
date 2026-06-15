"""Sleeper/Stayer and Sleeper/Drifter martingale implementations."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from .base import ConformalTestMartingale


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
