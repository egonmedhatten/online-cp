"""Legendre Jumper martingale implementations.

Promoted from development for release v0.3.0.

This module implements Algorithm 2 (Simple Legendre Jumper) and
Algorithm 3 (Product Legendre Jumper) as described in:

    "Legendre Jumper Martingale" by Johan Hallberg Szabadváry (2026).

Sign convention: f_eps^(k)(p) = 1 + eps * P_k(p)  (plus sign).
Default grid: E = {-1/2, -1/4, 0, 1/4, 1/2}.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre
from scipy.special import eval_legendre, logsumexp

from .base import ConformalTestMartingale

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

#: Standard 5-point epsilon grid from the paper.
STANDARD_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)

# Bounded cache for exact Gaunt coefficients keyed by order tuple.
GAUNT_CACHE_MAXSIZE = 32


def shifted_legendre_poly(k):
    """Return the k-th shifted Legendre polynomial P_k(2u-1) as a Polynomial in u.

    Parameters
    ----------
    k : int
        Order of the polynomial (k >= 0).

    Returns
    -------
    numpy.polynomial.Polynomial
        Polynomial in the monomial basis, orthogonal on [0, 1].
    """
    return Legendre([0] * k + [1], domain=[0, 1]).convert(kind=Polynomial)


def _poly_add_exact(a, b):
    n = max(len(a), len(b))
    out = [Fraction(0, 1)] * n
    for i in range(n):
        ai = a[i] if i < len(a) else Fraction(0, 1)
        bi = b[i] if i < len(b) else Fraction(0, 1)
        out[i] = ai + bi
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_scale_exact(a, c):
    cf = Fraction(c, 1)
    return [cf * x for x in a]


def _poly_mul_exact(a, b):
    out = [Fraction(0, 1)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_mul_linear_2u_minus_1_exact(a):
    out = [Fraction(0, 1)] * (len(a) + 1)
    for i, ai in enumerate(a):
        out[i] -= ai
        out[i + 1] += 2 * ai
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _integral_unit_interval_exact(poly):
    total = Fraction(0, 1)
    for i, ci in enumerate(poly):
        total += ci / Fraction(i + 1, 1)
    return total


def _build_shifted_legendre_exact(orders):
    if not orders:
        return {}
    max_order = max(orders)
    polys = {0: [Fraction(1, 1)]}
    if max_order >= 1:
        polys[1] = [Fraction(-1, 1), Fraction(2, 1)]

    for n in range(1, max_order):
        term1 = _poly_scale_exact(
            _poly_mul_linear_2u_minus_1_exact(polys[n]),
            2 * n + 1,
        )
        term2 = _poly_scale_exact(polys[n - 1], -n)
        numer = _poly_add_exact(term1, term2)
        denom = Fraction(n + 1, 1)
        polys[n + 1] = [c / denom for c in numer]

    return {k: polys[k] for k in orders}


@lru_cache(maxsize=GAUNT_CACHE_MAXSIZE)
def _gaunt_terms_exact_cached(orders_tuple):
    """Return exact Gaunt coefficients for a fixed order tuple.

    Output is a tuple of ``(indices, coefficient_float)`` where ``indices`` are
    positions in ``orders_tuple`` (not polynomial degrees themselves).
    """
    K = len(orders_tuple)
    if K < 3:
        return ()

    unique_orders = sorted(set(orders_tuple))
    exact_shifted = _build_shifted_legendre_exact(unique_orders)

    out = []
    for size in range(3, K + 1):
        for indices in itertools.combinations(range(K), size):
            poly = [Fraction(1, 1)]
            for idx in indices:
                poly = _poly_mul_exact(poly, exact_shifted[orders_tuple[idx]])
            coeff = float(_integral_unit_interval_exact(poly))
            if abs(coeff) > 1e-15:
                out.append((indices, coeff))
    return tuple(out)


def _compensated_sum(values):
    """Neumaier compensated summation for improved numerical stability."""
    total = np.longdouble(0.0)
    compensation = np.longdouble(0.0)
    for x in values:
        t = total + x
        if abs(total) >= abs(x):
            compensation += (total - t) + x
        else:
            compensation += (x - t) + total
        total = t
    return total + compensation


def _evaluate_Z_from_gaunt(eps, gaunt_terms):
    eps_ld = np.asarray(eps, dtype=np.longdouble)
    if not gaunt_terms:
        return 1.0
    z_terms = []
    for indices, G_S in gaunt_terms:
        prod_all = np.longdouble(1.0)
        for idx in indices:
            prod_all *= eps_ld[idx]
        z_terms.append(np.longdouble(G_S) * prod_all)
    z_terms.sort(key=lambda x: abs(x))
    Z = np.longdouble(1.0) + _compensated_sum(z_terms)
    return float(Z)


def _evaluate_Z_and_grad_from_gaunt(eps, gaunt_terms, K):
    eps_ld = np.asarray(eps, dtype=np.longdouble)
    if not gaunt_terms:
        return 1.0, np.zeros(K, dtype=float)

    z_terms = []
    grad_terms = [[] for _ in range(K)]

    for indices, G_S in gaunt_terms:
        G_ld = np.longdouble(G_S)
        prod_all = np.longdouble(1.0)
        for idx in indices:
            prod_all *= eps_ld[idx]
        z_terms.append(G_ld * prod_all)

        for idx in indices:
            prod_excl = np.longdouble(1.0)
            for j in indices:
                if j != idx:
                    prod_excl *= eps_ld[j]
            grad_terms[idx].append(G_ld * prod_excl)

    z_terms.sort(key=lambda x: abs(x))
    Z = np.longdouble(1.0) + _compensated_sum(z_terms)

    grad_Z = np.zeros(K, dtype=np.longdouble)
    for k in range(K):
        if grad_terms[k]:
            grad_terms[k].sort(key=lambda x: abs(x))
            grad_Z[k] = _compensated_sum(grad_terms[k])

    return float(Z), np.asarray(grad_Z, dtype=float)


def compute_normalization_Z(orders, epsilon_vec):
    """Compute Z(eps) = integral_0^1 prod_k (1 + eps_k * P_k(p)) dp.

    For |K| <= 2, Z = 1 exactly (by orthogonality).
    For |K| >= 3, Z deviates from 1 via Gaunt-type cross-terms.

    Parameters
    ----------
    orders : list of int
        Polynomial orders included in the product.
    epsilon_vec : array-like
        Epsilon values for each order.

    Returns
    -------
    float
        The normalization constant.
    """
    orders_tuple = tuple(int(k) for k in orders)
    gaunt_terms = _gaunt_terms_exact_cached(orders_tuple)
    return _evaluate_Z_from_gaunt(epsilon_vec, gaunt_terms)


def product_betting_value(orders, epsilon_vec, p, Z=None):
    """Evaluate f_eps^K(p) = prod_k(1 + eps_k * P_k(p)) / Z(eps).

    Parameters
    ----------
    orders : list of int
        Polynomial orders.
    epsilon_vec : array-like
        Epsilon values for each order.
    p : float
        The p-value at which to evaluate.
    Z : float or None
        Pre-computed normalization. If None, computed on the fly.

    Returns
    -------
    float
        Value of the normalized product betting function at p.
    """
    val = 1.0
    for k, eps in zip(orders, epsilon_vec):
        val *= (1.0 + eps * eval_legendre(k, 2.0 * p - 1.0))
    if Z is None:
        Z = compute_normalization_Z(orders, epsilon_vec)
    return val / Z


# ---------------------------------------------------------------------------
# Algorithm 2: Simple Legendre Jumper
# ---------------------------------------------------------------------------


class SimpleLegendreJumper(ConformalTestMartingale):
    """Simple Legendre Jumper betting martingale (Algorithm 2).

    Uses the betting function f_eps^(k)(p) = 1 + eps * P_k(2p-1)
    with a Markov chain over the state space E and jump rate J.

    Parameters
    ----------
    order : int
        Degree k >= 1 of the shifted Legendre polynomial.
    J : float
        Jump rate in (0, 1].
    epsilon_grid : tuple of float
        State space E. Default is the paper's 5-point grid.

    Examples
    --------
    >>> slj = SimpleLegendreJumper(order=1, J=0.01)
    >>> for _ in range(10):
    ...     slj.update(0.01)
    >>> slj.M > 1.0
    True
    """

    def __init__(self, order: int = 1, J: float = 0.01, epsilon_grid: tuple[float, ...] = STANDARD_GRID,
                 store_p_values: bool = True) -> None:
        super().__init__(store_p_values)
        if order < 1:
            raise ValueError("order must be >= 1")
        if not (0 < J <= 1):
            raise ValueError("J must be in (0, 1]")
        self.order = order
        self.J = J
        self.epsilon_grid = tuple(epsilon_grid)
        self.N = len(self.epsilon_grid)

        # Log-space state: log(C_eps) for each expert
        self._log_C = np.full(self.N, -np.log(self.N))  # uniform: 1/|E|
        self._log_total = 0.0  # log(C) = log(sum C_eps) = 0 initially

    def _P(self, u):
        """Evaluate shifted Legendre polynomial P_k(2u-1)."""
        return eval_legendre(self.order, 2.0 * u - 1.0)

    def update(self, p: float) -> None:
        r"""Advance the simple Legendre jumper by one p-value.

        Performs the jump (mixing over the $\epsilon$ grid) then bets with the
        shifted-Legendre density of the configured order, all in log-space
        ([LegendreJumper, preprint]).

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.
        """
        if self.store_p_values:
            self.p_values.append(p)

        # --- Step 1: Jump ---
        # C_eps <- (1-J) * C_eps + (J / |E|) * C
        # In log-space: log((1-J)*C_eps + (J/N)*C)
        #             = logaddexp(log(1-J) + log_C_eps, log(J/N) + log_total)
        if self.J == 1.0:
            log_1_minus_J = -np.inf
        else:
            log_1_minus_J = np.log(1.0 - self.J)
        log_J_div_N = np.log(self.J / self.N)

        term1 = log_1_minus_J + self._log_C
        term2 = log_J_div_N + self._log_total
        self._log_C = np.logaddexp(term1, term2)

        # --- Step 2: Bet ---
        # C_eps <- C_eps * f_eps^(k)(p)
        # f_eps(p) = 1 + eps * P_k(p)
        P_p = self._P(p)
        bet_values = np.array([1.0 + eps * P_p for eps in self.epsilon_grid])
        self._log_C += np.log(bet_values)

        # --- Step 3: Accumulate ---
        # C = sum(C_eps)
        self._log_total = logsumexp(self._log_C)
        self.logM = self._log_total
        self.log_martingale_values.append(self.logM)

        # --- Mark b_n/B_n stale (lazy recomputation on access) ---
        self._mark_stale()

    def _update_exposed_functions(self):
        """Set b_n and B_n as the wealth-weighted effective betting function."""
        # Post-jump weights for next step
        if self.J == 1.0:
            log_1_minus_J = -np.inf
        else:
            log_1_minus_J = np.log(1.0 - self.J)
        log_J_div_N = np.log(self.J / self.N)

        term1 = log_1_minus_J + self._log_C
        term2 = log_J_div_N + self._log_total
        log_weights_next = np.logaddexp(term1, term2)
        log_sum = logsumexp(log_weights_next)
        weights = np.exp(log_weights_next - log_sum)

        # Effective epsilon: eps_bar = sum_i w_i * eps_i
        eps_bar = np.dot(weights, self.epsilon_grid)
        order = self.order

        def _b_n(u, _eps=eps_bar, _k=order):
            return 1.0 + _eps * eval_legendre(_k, 2.0 * u - 1.0)

        def _B_n(u, _eps=eps_bar, _k=order):
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            # B_n(u) = u + eps * Q_k(u) where Q_k = integral_0^u P_k(t) dt
            P_kp1 = eval_legendre(_k + 1, 2.0 * u - 1.0)
            P_km1 = eval_legendre(_k - 1, 2.0 * u - 1.0)
            Q = (P_kp1 - P_km1) / (2.0 * (2 * _k + 1))
            return u + _eps * Q

        self.b_n = _b_n
        self.B_n = _B_n

        # Closed-form inverse for k=1 (quadratic); otherwise numerical fallback
        if order == 1:
            a = eps_bar

            def _B_n_inv(x, _a=a):
                if x <= 0.0:
                    return 0.0
                if x >= 1.0:
                    return 1.0
                if abs(_a) < 1e-12:
                    return x
                return (-(1 - _a) + np.sqrt((1 - _a) ** 2 + 4 * _a * x)) / (2 * _a)

            self.B_n_inv = _B_n_inv


class ProductLegendreJumper(ConformalTestMartingale):
    """Product Legendre Jumper betting martingale (Algorithm 3).

    Maintains a single Markov chain over the full Cartesian product state
    space E = E_1 x E_2 x ... x E_K, with betting function:

        f_eps^K(p) = prod_k (1 + eps_k * P_k(p)) / Z(eps)

    Parameters
    ----------
    orders : list of int
        Set K of Legendre polynomial degrees (each >= 1).
    J : float
        Jump rate in (0, 1].
    epsilon_grid : tuple of float
        Per-order state space E_k. Default is the paper's 5-point grid.

    Examples
    --------
    >>> plj = ProductLegendreJumper(orders=[1, 2], J=0.01)
    >>> for _ in range(10):
    ...     plj.update(0.01)
    >>> plj.M > 1.0
    True
    """

    def __init__(self, orders: list[int] | None = None, J: float = 0.01, epsilon_grid: tuple[float, ...] = STANDARD_GRID,
                 store_p_values: bool = True) -> None:
        super().__init__(store_p_values)
        if orders is None:
            orders = [1, 2, 3]
        if not orders:
            raise ValueError("At least one order is required.")
        if any(k < 1 for k in orders):
            raise ValueError("All orders must be >= 1.")
        if not (0 < J <= 1):
            raise ValueError("J must be in (0, 1]")
        self.orders = list(orders)
        self.K = len(self.orders)
        self.J = J
        self.epsilon_grid = tuple(epsilon_grid)

        # Build full Cartesian product state space
        self.states = list(itertools.product(self.epsilon_grid, repeat=self.K))
        self.N = len(self.states)
        self._states_array = np.array(self.states)  # shape (N, K) for vectorized _eval_bets

        if self.N > 500:
            import warnings as _w
            _w.warn(
                f"Product state space has {self.N} experts "
                f"(orders={self.orders}, grid size={len(self.epsilon_grid)}). "
                f"Consider using fewer orders or a coarser grid.",
                stacklevel=2,
            )

        # Pre-compute Z(eps) for every state (done once at init)
        self._log_Z = np.zeros(self.N)
        for i, eps_vec in enumerate(self.states):
            Z = compute_normalization_Z(self.orders, eps_vec)
            self._log_Z[i] = np.log(Z)

        # Log-space state: uniform prior over all experts
        self._log_C = np.full(self.N, -np.log(self.N))
        self._log_total = 0.0

    def _eval_bets(self, p):
        """Evaluate f_eps^K(p) for all states. Returns array of shape (N,)."""
        # Compute P_k(p) for each order once
        Pk_vals = np.array([eval_legendre(k, 2.0 * p - 1.0) for k in self.orders])

        # Vectorized: (N, K) * (K,) -> (N, K), then sum over K axis
        return np.log1p(self._states_array * Pk_vals).sum(axis=1) - self._log_Z

    def update(self, p: float) -> None:
        r"""Advance the product Legendre jumper by one p-value.

        Bets with a *product* of shifted-Legendre densities over several orders,
        after the jump-mixing step over the $\epsilon$ grid
        ([LegendreJumper, preprint]).

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.
        """
        if self.store_p_values:
            self.p_values.append(p)

        # --- Step 1: Jump ---
        # C_eps <- (1-J) * C_eps + (J / |E|) * C
        if self.J == 1.0:
            log_1_minus_J = -np.inf
        else:
            log_1_minus_J = np.log(1.0 - self.J)
        log_J_div_N = np.log(self.J / self.N)

        term1 = log_1_minus_J + self._log_C
        term2 = log_J_div_N + self._log_total
        self._log_C = np.logaddexp(term1, term2)

        # --- Step 2: Bet ---
        # C_eps <- C_eps * f_eps^K(p)
        log_bets = self._eval_bets(p)
        self._log_C += log_bets

        # --- Step 3: Accumulate ---
        self._log_total = logsumexp(self._log_C)
        self.logM = self._log_total
        self.log_martingale_values.append(self.logM)

        # --- Mark b_n/B_n stale (lazy recomputation on access) ---
        self._mark_stale()

    def _update_exposed_functions(self):
        """Set b_n and B_n as the wealth-weighted mixture of expert betting functions."""
        # Post-jump weights for next step
        if self.J == 1.0:
            log_1_minus_J = -np.inf
        else:
            log_1_minus_J = np.log(1.0 - self.J)
        log_J_div_N = np.log(self.J / self.N)

        term1 = log_1_minus_J + self._log_C
        term2 = log_J_div_N + self._log_total
        log_weights_next = np.logaddexp(term1, term2)
        log_sum = logsumexp(log_weights_next)
        weights = np.exp(log_weights_next - log_sum)

        # Capture state for closures
        orders = self.orders
        states = self.states
        log_Z = self._log_Z

        def _b_n(u, _w=weights, _orders=orders, _states=states, _log_Z=log_Z):
            Pk_vals = [eval_legendre(k, 2.0 * u - 1.0) for k in _orders]
            total = 0.0
            for i, eps_vec in enumerate(_states):
                prod = 1.0
                for j, eps in enumerate(eps_vec):
                    prod *= (1.0 + eps * Pk_vals[j])
                total += _w[i] * prod / np.exp(_log_Z[i])
            return total

        def _B_n(u, _b_n_func=None):
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            # Numerical integration (the mixture doesn't simplify to a closed-form CDF)
            from scipy.integrate import quad
            val, _ = quad(_b_n, 1e-12, u, limit=50)
            return val

        self.b_n = _b_n
        self.B_n = _B_n


# ---------------------------------------------------------------------------
# Algorithm 4: Variational Legendre Jumper
# ---------------------------------------------------------------------------


class VariationalLegendreJumper(ConformalTestMartingale):
    """Variational Legendre Jumper betting martingale (Algorithm 4).

    Runs |K| independent sub-jumpers (each an instance of the SLJ logic),
    one per polynomial degree. At each step, consensus parameters are
    computed as the wealth-weighted mean epsilon from each sub-jumper,
    and the global martingale bets with the Z-normalised product betting
    function evaluated at these consensus parameters.

    Computational cost: O(|K| * g) per step (linear, not exponential).

    Parameters
    ----------
    orders : list of int
        Set K of Legendre polynomial degrees (each >= 1).
    J : float
        Jump rate in (0, 1].
    epsilon_grid : tuple of float
        Per-order state space E_k. Default is the paper's 5-point grid.

    Examples
    --------
    >>> vlj = VariationalLegendreJumper(orders=[1, 2], J=0.01)
    >>> for _ in range(10):
    ...     vlj.update(0.01)
    >>> vlj.M > 1.0
    True
    """

    def __init__(self, orders: list[int] | None = None, J: float = 0.01, epsilon_grid: tuple[float, ...] = STANDARD_GRID,
                 store_p_values: bool = True) -> None:
        super().__init__(store_p_values)
        if orders is None:
            orders = [1, 2, 3]
        if not orders:
            raise ValueError("At least one order is required.")
        if any(k < 1 for k in orders):
            raise ValueError("All orders must be >= 1.")
        if not (0 < J <= 1):
            raise ValueError("J must be in (0, 1]")
        self.orders = list(orders)
        self.K = len(self.orders)
        self.J = J
        self.epsilon_grid = tuple(epsilon_grid)
        self.g = len(self.epsilon_grid)
        self._eps_array = np.array(self.epsilon_grid)

        # Log-space sub-jumper state: log(C_{k, eps})
        # Shape: (K, g). Initialised to log(1/g) = -log(g).
        self._log_C = np.full((self.K, self.g), -np.log(self.g))
        # log(C_k) = log(sum_eps C_{k, eps}) = 0 initially
        self._log_C_k = np.zeros(self.K)

        # Global martingale value in log-space
        self._log_S = 0.0

        # Pre-compute log constants for the jump step
        if self.J == 1.0:
            self._log_1_minus_J = -np.inf
        else:
            self._log_1_minus_J = np.log(1.0 - self.J)
        self._log_J_div_g = np.log(self.J / self.g)

        # Pre-compute Gaunt coefficients for fast Z evaluation.
        # Z(eps) = 1 + sum_{|S|>=3} (prod_{k in S} eps_k) * G_S
        # where G_S = integral_0^1 prod_{k in S} P_k(p) dp.
        # For |K| <= 2, Z = 1 always (by orthogonality).
        self._gaunt_terms = list(_gaunt_terms_exact_cached(tuple(self.orders)))

    def _fast_Z(self, eps_bar):
        """Compute Z(eps_bar) using precomputed Gaunt coefficients. O(1) for typical |K|."""
        return _evaluate_Z_from_gaunt(eps_bar, self._gaunt_terms)

    def update(self, p: float) -> None:
        r"""Advance the variational Legendre jumper by one p-value.

        Maintains per-sub-jumper consensus parameters $\bar\epsilon_k$ via a
        variational update before betting ([LegendreJumper, preprint]).

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.
        """
        if self.store_p_values:
            self.p_values.append(p)

        # --- Step 1: Jump (per sub-jumper) ---
        # C_{k,eps} <- (1-J) * C_{k,eps} + (J/g) * C_k
        # In log-space: logaddexp(log(1-J) + log_C, log(J/g) + log_C_k)
        term1 = self._log_1_minus_J + self._log_C
        term2 = self._log_J_div_g + self._log_C_k[:, np.newaxis]
        self._log_C = np.logaddexp(term1, term2)

        # --- Step 2: Consensus parameters ---
        # eps_bar_k = sum_eps eps * (C_{k,eps} / C_k)
        # weights = exp(log_C - log_C_k) are normalized probabilities
        weights = np.exp(self._log_C - self._log_C_k[:, np.newaxis])
        eps_bar = (weights * self._eps_array[np.newaxis, :]).sum(axis=1)

        # --- Step 3: Global bet ---
        # S_n = S_{n-1} * prod_k(1 + eps_bar_k * P_k(p)) / Z(eps_bar)
        Pk_vals = np.array([eval_legendre(k, 2.0 * p - 1.0) for k in self.orders])
        Z = self._fast_Z(eps_bar)
        log_global_bet = np.sum(np.log(1.0 + eps_bar * Pk_vals)) - np.log(Z)
        self._log_S += log_global_bet

        # --- Step 4: Update sub-jumpers ---
        # C_{k,eps} <- C_{k,eps} * (1 + eps * P_k(p))
        marginal_bets = 1.0 + self._eps_array[np.newaxis, :] * Pk_vals[:, np.newaxis]
        self._log_C += np.log(marginal_bets)

        # --- Step 5: Recompute log_C_k ---
        self._log_C_k = logsumexp(self._log_C, axis=1)

        # --- Update martingale ---
        self.logM = self._log_S
        self.log_martingale_values.append(self.logM)

        # --- Mark b_n/B_n stale (lazy recomputation on access) ---
        self._mark_stale()

    def _update_exposed_functions(self):
        """Set b_n and B_n using consensus parameters for the next step."""
        # Simulate post-jump state (what would happen next step)
        term1 = self._log_1_minus_J + self._log_C
        term2 = self._log_J_div_g + self._log_C_k[:, np.newaxis]
        log_C_next = np.logaddexp(term1, term2)
        log_C_k_next = logsumexp(log_C_next, axis=1)

        weights = np.exp(log_C_next - log_C_k_next[:, np.newaxis])
        eps_bar = (weights * self._eps_array[np.newaxis, :]).sum(axis=1)

        orders = self.orders
        gaunt_terms = self._gaunt_terms

        def _b_n(u, _eps_bar=eps_bar.copy(), _orders=orders, _gaunt=gaunt_terms):  # noqa: B008
            Pk_vals = [eval_legendre(k, 2.0 * u - 1.0) for k in _orders]
            # Fast Z from precomputed Gaunt coefficients
            Z = _evaluate_Z_from_gaunt(_eps_bar, _gaunt)
            val = 1.0
            for j, eps in enumerate(_eps_bar):
                val *= (1.0 + eps * Pk_vals[j])
            return val / Z

        def _B_n(u):
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            from scipy.integrate import quad
            val, _ = quad(_b_n, 1e-12, u, limit=50)
            return val

        self.b_n = _b_n
        self.B_n = _B_n


# ---------------------------------------------------------------------------
# Optimistic FTRL + Barrier Legendre martingale
# ---------------------------------------------------------------------------


class OptimisticFTRLBarrierLegendreMartingale(ConformalTestMartingale):
    """Jumpless multivariate Legendre martingale via Optimistic-FTRL + Barrier.

    Uses the same product betting function as ``ProductLegendreJumper``:

        f_eps^K(p) = prod_k (1 + eps_k * P_k(p)) / Z(eps),

    but updates the parameter vector ``eps`` continuously with the
    Optimistic-FTRL + Barrier closed-form update from Wang et al. (Algorithm 3).

    Parameters
    ----------
    orders : list of int or None
        Set K of Legendre polynomial degrees (each >= 1).
    eta : float
        Learning-rate parameter for Optimistic-FTRL. Must satisfy
        ``0 < eta <= 0.25``.
    store_p_values : bool
        Whether to store incoming p-values.

    Examples
    --------
    >>> oftrl = OptimisticFTRLBarrierLegendreMartingale(orders=[1, 2], eta=0.25)
    >>> for _ in range(10):
    ...     oftrl.update(0.01)
    >>> oftrl.M > 1.0
    True
    """

    def __init__(self, orders: list[int] | None = None, eta: float = 0.25,
                 store_p_values: bool = True) -> None:
        super().__init__(store_p_values)
        if orders is None:
            orders = [1, 2, 3]
        if not orders:
            raise ValueError("At least one order is required.")
        if any(k < 1 for k in orders):
            raise ValueError("All orders must be >= 1.")
        if not (0.0 < eta <= 0.25):
            raise ValueError("eta must be in (0, 0.25].")

        self.orders = list(orders)
        self.K = len(self.orders)
        self.eta = float(eta)
        self._z_floor = 1e-14

        # Current parameter vector eps_t and cumulative gradient G_t.
        self._eps = np.zeros(self.K, dtype=float)
        self._G = np.zeros(self.K, dtype=float)

        # Pre-compute Gaunt coefficients for exact Z and exact grad(Z).
        # Z(eps) = 1 + sum_{|S|>=3} (prod_{k in S} eps_k) * G_S.
        self._gaunt_terms = list(_gaunt_terms_exact_cached(tuple(self.orders)))

    def _compute_Z_only_strict(self, eps):
        """Strict normalization recomputation in longdouble from Gaunt terms."""
        return np.longdouble(_evaluate_Z_from_gaunt(eps, self._gaunt_terms))

    def _compute_Z_and_gradient(self, eps):
        """Compute exact Z(eps) and exact grad Z(eps) from Gaunt coefficients."""
        return _evaluate_Z_and_grad_from_gaunt(eps, self._gaunt_terms, self.K)

    def update(self, p: float) -> None:
        r"""Advance martingale by one p-value using exact Optimistic-FTRL updates.

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.
        """
        if self.store_p_values:
            self.p_values.append(p)

        Pk_vals = np.array([eval_legendre(k, 2.0 * p - 1.0) for k in self.orders], dtype=float)
        Z, grad_Z = self._compute_Z_and_gradient(self._eps)
        if (not np.isfinite(Z)) or (Z <= self._z_floor):
            Z_strict = self._compute_Z_only_strict(self._eps)
            if np.isfinite(Z_strict) and Z_strict > self._z_floor:
                Z = float(Z_strict)
            else:
                raise FloatingPointError(f"Invalid normalization constant Z={Z} (strict={Z_strict}).")
        if Z <= 0.0:
            raise FloatingPointError(f"Invalid normalization constant Z={Z}.")

        denom = 1.0 + self._eps * Pk_vals
        if np.any(denom <= 0.0):
            raise FloatingPointError("Encountered non-positive term in product betting function.")

        # Bet with current eps_t.
        self.logM += float(np.sum(np.log(denom)) - np.log(Z))
        self.log_martingale_values.append(self.logM)

        # Exact gradient of loss l_t(eps).
        grad = -(Pk_vals / denom) + (grad_Z / Z)

        # Optimistic-FTRL state update: G_t += grad_t, then eps_{t+1} from G_t + m_{t+1}
        # with m_{t+1} = grad_t.
        self._G += grad
        a = self.eta * (self._G + grad)

        eps_next = np.zeros_like(self._eps)
        mask = np.abs(a) > 1e-15
        eps_next[mask] = (1.0 - np.sqrt(1.0 + a[mask] ** 2)) / a[mask]
        self._eps = eps_next

        self._mark_stale()

    def _update_exposed_functions(self):
        """Set b_n and B_n at the current Optimistic-FTRL parameter eps."""
        eps = self._eps.copy()
        orders = self.orders
        Z, _ = self._compute_Z_and_gradient(eps)

        def _b_n(u, _orders=orders, _eps=eps, _Z=Z):
            return product_betting_value(_orders, _eps, u, Z=_Z)

        def _B_n(u):
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            from scipy.integrate import quad
            val, _ = quad(_b_n, 1e-12, u, limit=50)
            return val

        self.b_n = _b_n
        self.B_n = _B_n


# ---------------------------------------------------------------------------
# Composite Legendre Jumper (averages over multiple jump rates)
# ---------------------------------------------------------------------------


class CompositeLegendreJumper(ConformalTestMartingale):
    """Composite Legendre Jumper that averages over multiple jump rates.

    Creates multiple instances of a base Legendre Jumper class (one per
    jumping rate) and computes the martingale as their arithmetic mean.
    This is the direct analogue of the Composite Jumper for Legendre
    martingales.

    Parameters
    ----------
    base_class : class
        The base martingale class to instantiate. Must accept a ``J``
        parameter. Default is ``SimpleLegendreJumper``.
    J : list of float or None
        List of jumping rates. Default is [1e-4, 1e-3, 1e-2, 1e-1, 1.0].
    **kwargs
        Additional keyword arguments forwarded to ``base_class`` (e.g.
        ``order``, ``orders``, ``epsilon_grid``).

    Examples
    --------
    >>> clj = CompositeLegendreJumper()
    >>> for _ in range(5):
    ...     clj.update(0.01)
    >>> clj.M > 1.0
    True

    >>> from online_cp.martingale import VariationalLegendreJumper
    >>> clj = CompositeLegendreJumper(
    ...     base_class=VariationalLegendreJumper, orders=[1, 2]
    ... )
    >>> for _ in range(5):
    ...     clj.update(0.01)
    >>> clj.M > 1.0
    True
    """

    def __init__(self, base_class: type[ConformalTestMartingale] | None = None, J: list[float] | None = None,
                 store_p_values: bool = True, **kwargs: Any) -> None:
        super().__init__(store_p_values)
        if base_class is None:
            base_class = SimpleLegendreJumper
        if J is None:
            J = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

        self.J = list(J)
        self.base_class = base_class
        self._jumpers = [
            base_class(J=j, store_p_values=False, **kwargs)
            for j in self.J
        ]
        self._n_jumpers = len(self._jumpers)
        self._log_n = np.log(self._n_jumpers)

    def update(self, p: float) -> None:
        """Advance every component Legendre jumper and pool them.

        Updates each sub-jumper on ``p`` and sets the composite log-martingale to
        the equal-weight log-mean of the components ([LegendreJumper, preprint]).

        Parameters
        ----------
        p : float
            New p-value in $[0, 1]$.
        """
        if self.store_p_values:
            self.p_values.append(p)

        for m in self._jumpers:
            m.update(p)

        log_Ms = np.array([m.logM for m in self._jumpers])
        self.logM = logsumexp(log_Ms) - self._log_n

        self.log_martingale_values.append(self.logM)

        self._mark_stale()
