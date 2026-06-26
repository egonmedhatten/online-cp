r"""Betting strategies for conformal test martingales.

A *betting strategy* is a probability density $g$ on $[0, 1]$ used to gamble
against the null hypothesis that conformal p-values are i.i.d. uniform. At each
step the martingale's wealth is multiplied by $g(p_n)$. Because any density
integrates to one,

$$
\mathbb{E}_{p \sim \mathrm{U}[0,1]}\bigl[g(p)\bigr] = \int_0^1 g(u)\,du = 1 ,
$$

the wealth process is a non-negative martingale under the null \u2014 fair betting
that can only grow in expectation when the p-values are *not* uniform. The art
is to choose $g$ to concentrate mass where the empirical p-value density departs
from uniform (a Kelly-style bet); the strategies here are online density
estimators on $[0, 1]$ that learn $g$ from the past p-values.

Each strategy exposes ``bet(p)`` (the density $g(p)$), ``integrate(p)`` (its CDF,
used as a protection function) and ``update(p)`` (incorporate a new p-value),
following the *predict-then-learn* order that preserves the martingale property.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, logsumexp
from scipy.stats import beta, norm, uniform

# Optional numba for KDE speedup
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    HAS_NUMBA = False

__all__ = [
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

# Handle optional dependency for BetaKDE
try:
    from beta_kde import BetaKDE
except ImportError:

    class BetaKDE:
        _warned = False

        def __init__(self, bandwidth="beta-reference"):
            if not BetaKDE._warned:
                warnings.warn(
                    "beta_kde package not installed. BetaKernel strategy will "
                    "use uniform density (no betting power). Install with: "
                    "pip install beta_kde",
                    stacklevel=2,
                )
                BetaKDE._warned = True

        def fit(self, data, compute_normalization=True):
            pass

        def pdf(self, x, normalized=True):
            return np.ones_like(x) if np.ndim(x) > 0 else 1.0


class BettingStrategy:
    """Base class for betting strategies (pure density estimators on [0,1]).

    A betting strategy estimates the density of conformal p-values. It exposes:
    - ``bet(p)``: evaluate the current density at p (using past data only)
    - ``integrate(p)``: evaluate the current CDF at p (protection function)
    - ``update(p)``: incorporate a new p-value into the estimate

    The critical invariant is **predict then learn**: ``bet(p)`` uses only data
    seen before p, ensuring the martingale property.

    Strategies are *pure* — they do not apply cautious mixing. That is the
    responsibility of the martingale that wraps them.
    """

    def bet(self, p: float) -> float:
        """Return the density f(p) using current state (past data only).

        Must satisfy f(p) >= 0 and integrate to ~1 over [0,1].
        """
        return 1.0  # uniform = no betting

    def integrate(self, p: float) -> float:
        """Return the CDF F(p) using current state (protection function).

        Must satisfy F(0) = 0, F(1) = 1, monotone increasing.
        """
        return p  # uniform CDF

    def update(self, p: float) -> None:
        """Incorporate a new p-value observation into the density estimate."""
        pass


class BetaKernel(BettingStrategy):
    """Beta Kernel Density Estimation betting strategy.

    Uses the ``beta_kde`` package (if installed) to estimate the density
    of p-values with a Beta kernel, which handles [0,1] boundaries well.

    Parameters
    ----------
    bandwidth : str or float
        Bandwidth selection method (default: "beta-reference").
    window_size : int or None
        If set, only use the last ``window_size`` observations.
    normalize : bool
        Whether to normalize the KDE.

    Examples
    --------
    >>> bk = BetaKernel()
    >>> for p in [0.1, 0.2, 0.15, 0.05, 0.1]:
    ...     bk.update(p)
    >>> bool(bk.bet(0.1) >= 1.0)  # density peaks near the data (== 1.0 without beta_kde)
    True
    """

    def __init__(self, bandwidth="beta-reference", window_size=None, normalize=True):
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.normalize = normalize
        self._data = []
        self._kde = BetaKDE(bandwidth=self.bandwidth)

    def bet(self, p: float) -> float:
        if len(self._data) < 2:
            return 1.0
        return float(self._kde.pdf(p, normalized=self.normalize))

    def integrate(self, p: float) -> float:
        if len(self._data) < 2:
            return p
        val, _ = quad(lambda x: self._kde.pdf(x, normalized=self.normalize), 0, p, limit=50)
        return float(val)

    def update(self, p: float) -> None:
        self._data.append(p)
        if len(self._data) >= 2:
            window = self.window_size or len(self._data)
            data = np.array(self._data[-window:])
            self._kde.fit(data.reshape(-1, 1), compute_normalization=self.normalize)


# ─── Numba-accelerated reflected Gaussian KDE helpers ─────────────────────────

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@njit(cache=False)
def _reflected_kde_pdf_numba(x, data, h):
    """Reflected Gaussian KDE PDF evaluated at a single point x.

    Computes sum over data of [phi(x-d) + phi(x+d) + phi(x-(2-d))] / (n*h)
    without allocating intermediate arrays.
    """
    n = data.shape[0]
    total = 0.0
    inv_h = 1.0 / h
    for i in range(n):
        d = data[i]
        z1 = (x - d) * inv_h
        z2 = (x + d) * inv_h
        z3 = (x - (2.0 - d)) * inv_h
        total += np.exp(-0.5 * z1 * z1) + np.exp(-0.5 * z2 * z2) + np.exp(-0.5 * z3 * z3)
    return total * _INV_SQRT_2PI * inv_h / n


@njit(cache=False)
def _reflected_kde_cdf_numba(x, data, h):
    """Reflected Gaussian KDE CDF evaluated at a single point x.

    Uses the identity: Phi(z) = 0.5 * erfc(-z / sqrt(2)).
    """
    n = data.shape[0]
    total = 0.0
    inv_h = 1.0 / h
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for i in range(n):
        d = data[i]
        z1 = (x - d) * inv_h
        z2 = (x + d) * inv_h
        z3 = (x - (2.0 - d)) * inv_h
        total += 0.5 * math.erfc(-z1 * inv_sqrt2) + 0.5 * math.erfc(-z2 * inv_sqrt2) + 0.5 * math.erfc(-z3 * inv_sqrt2)
    return total / n - 1.0  # subtract 1 for the reflection normalization


class GaussianKDE(BettingStrategy):
    """Gaussian Kernel Density Estimation betting strategy with boundary reflection.

    Uses a reflected Gaussian kernel to properly handle the [0,1] boundary.

    Parameters
    ----------
    bandwidth : str or float
        Bandwidth selection: "silverman" (rule of thumb), "lcv" (likelihood
        cross-validation), or a fixed float value.
    window_size : int or None
        If set, only use the last ``window_size`` observations.
    max_iter : int
        Maximum iterations for LCV bandwidth optimization.
    bw_min, bw_max : float
        Bandwidth search bounds for LCV.
    growth_factor : float
        Re-optimize bandwidth when sample size grows by this factor.

    Examples
    --------
    >>> gkde = GaussianKDE(bandwidth=0.1)
    >>> for p in [0.1, 0.11, 0.09, 0.12, 0.08]:
    ...     gkde.update(p)
    >>> bool(gkde.bet(0.1) > 1.0)  # should peak near the data
    True
    >>> bool(gkde.bet(0.9) < 1.0)
    True
    """

    def __init__(
        self,
        bandwidth="silverman",
        window_size=None,
        max_iter=20,
        bw_min=0.001,
        bw_max=0.5,
        growth_factor=1.1,
    ):
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.max_iter = max_iter
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.growth_factor = growth_factor

        self._data = []
        self._current_bw = None
        self._n_last_update = 0

    def _kernel_pdf_reflect(self, x, data, h):
        """Reflected Gaussian kernel PDF."""
        if HAS_NUMBA:
            return _reflected_kde_pdf_numba(float(x), data, h)
        x = np.atleast_1d(x)
        pdf_orig = np.mean(norm.pdf(x[:, None], loc=data, scale=h), axis=1)
        pdf_neg = np.mean(norm.pdf(x[:, None], loc=-data, scale=h), axis=1)
        pdf_two = np.mean(norm.pdf(x[:, None], loc=2 - data, scale=h), axis=1)
        pdf_values = pdf_orig + pdf_neg + pdf_two
        return pdf_values.item() if pdf_values.size == 1 else pdf_values

    def _kernel_cdf_reflect(self, x, data, h):
        """Reflected Gaussian kernel CDF."""
        if HAS_NUMBA:
            return _reflected_kde_cdf_numba(float(x), data, h)
        x = np.atleast_1d(x)
        cdf_orig = np.mean(norm.cdf(x[:, None], loc=data, scale=h), axis=1)
        cdf_neg = np.mean(norm.cdf(x[:, None], loc=-data, scale=h), axis=1)
        cdf_two = np.mean(norm.cdf(x[:, None], loc=2 - data, scale=h), axis=1)
        cdf_values = cdf_orig + cdf_neg + cdf_two - 1
        return cdf_values.item() if cdf_values.size == 1 else cdf_values

    def _likelihood_lcv_bw(self, data):
        """Optimal bandwidth via Likelihood Cross-Validation."""
        n = len(data)
        data_col = data[:, np.newaxis]
        diff_orig = data_col - data
        diff_neg = data_col + data
        diff_two = data_col - (2 - data)

        def objective_func(h):
            epsilon = 1e-10
            kernel_matrix_orig = norm.pdf(diff_orig / h)
            kernel_matrix_neg = norm.pdf(diff_neg / h)
            kernel_matrix_two = norm.pdf(diff_two / h)
            total_kernel_matrix = kernel_matrix_orig + kernel_matrix_neg + kernel_matrix_two
            np.fill_diagonal(total_kernel_matrix, 0)
            loo_pdfs = np.sum(total_kernel_matrix, axis=1) / ((n - 1) * h)
            loo_log_likelihood = np.sum(np.log(loo_pdfs + epsilon))
            return -loo_log_likelihood

        if n < 50 or self._current_bw is None:
            result = minimize_scalar(
                objective_func, bounds=(self.bw_min, self.bw_max), method="bounded", options={"maxiter": self.max_iter}
            )
        else:
            search_bounds = (
                max(self._current_bw * 0.8, self.bw_min),
                min(self._current_bw * 1.2, self.bw_max),
            )
            result = minimize_scalar(
                objective_func, bounds=search_bounds, method="bounded", options={"maxiter": self.max_iter}
            )
        return result

    def _calculate_bandwidth(self, data):
        """Calculate bandwidth for current data."""
        n = data.size
        if self.bandwidth == "silverman":
            sigma = np.std(data)
            if sigma > 1e-14:  # avoid zero or near-zero bandwidth
                h = ((4 * sigma**5) / (3 * n)) ** (1 / 5)
            else:
                h = self.bw_min  # fallback for zero variance
        elif self.bandwidth == "lcv":
            should_recalculate = (self._current_bw is None) or (n >= self._n_last_update * self.growth_factor)
            if should_recalculate and n > 1:
                result = self._likelihood_lcv_bw(data)
                h = result.x
                self._n_last_update = n
            else:
                h = self._current_bw if self._current_bw is not None else self.bw_min
        else:
            h = float(self.bandwidth)
        # Clamp to valid bounds
        h = np.clip(h, self.bw_min, self.bw_max)
        return h

    def _get_data(self):
        """Get the current data array (windowed if applicable)."""
        window = self.window_size or len(self._data)
        return np.array(self._data[-window:])

    def bet(self, p: float) -> float:
        if len(self._data) < 2:
            return 1.0
        data = self._get_data()
        if data.std() < 1e-6:
            return 1.0
        # _current_bw should always be set by update(), but guard against None
        if self._current_bw is None:
            return 1.0
        return self._kernel_pdf_reflect(p, data, self._current_bw)

    def integrate(self, p: float) -> float:
        if len(self._data) < 2:
            return p
        data = self._get_data()
        if data.std() < 1e-6:
            return p
        # _current_bw should always be set by update(), but guard against None
        if self._current_bw is None:
            return p
        return self._kernel_cdf_reflect(p, data, self._current_bw)

    def update(self, p: float) -> None:
        self._data.append(p)
        if len(self._data) >= 2:
            data = self._get_data()
            if data.std() >= 1e-6:
                self._current_bw = self._calculate_bandwidth(data)


class BetaMoments(BettingStrategy):
    """Betting strategy based on Beta distribution with method of moments.

    Maintains online running mean and variance (Welford's algorithm)
    and uses method-of-moments to fit Beta(a, b) parameters.

    Examples
    --------
    >>> bm = BetaMoments()
    >>> for p in [0.01, 0.02, 0.05, 0.01, 0.03]:
    ...     bm.update(p)
    >>> bool(bm.bet(0.01) > 1.0)  # should favor small p-values
    True
    >>> bool(bm.bet(0.99) < 1.0)
    True
    """

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._ahat = 1.0
        self._bhat = 1.0

    def bet(self, p: float) -> float:
        return beta.pdf(p, self._ahat, self._bhat)

    def integrate(self, p: float) -> float:
        return beta.cdf(p, self._ahat, self._bhat)

    def update(self, p: float) -> None:
        # Welford online update
        self._n += 1
        delta = p - self._mean
        self._mean += delta / self._n
        delta2 = p - self._mean
        self._M2 += delta * delta2

        # Re-estimate parameters
        if self._n >= 2:
            sample_variance = self._M2 / (self._n - 1)
            if sample_variance > 0 and 0 < self._mean < 1:
                common_factor = (self._mean * (1 - self._mean) / sample_variance) - 1
                if common_factor > 0:
                    self._ahat = max(self._mean * common_factor, 1e-4)
                    self._bhat = max((1 - self._mean) * common_factor, 1e-4)


class BetaMLE(BettingStrategy):
    """Betting strategy based on MLE for Beta distribution parameters.

    Maintains sufficient statistics (sum of log) and re-optimizes
    Beta(a, b) via maximum likelihood at each step.

    Examples
    --------
    >>> bmle = BetaMLE()
    >>> for p in [0.01, 0.02, 0.01, 0.02, 0.01]:
    ...     bmle.update(p)
    >>> bool(bmle.bet(0.01) > 1.0)  # should favor small p-values
    True
    """

    def __init__(self):
        self._n = 0
        self._log_sum_x = 0.0
        self._log_sum_1_minus_x = 0.0
        self._ahat = 1.0
        self._bhat = 1.0

    def bet(self, p: float) -> float:
        return beta.pdf(p, self._ahat, self._bhat)

    def integrate(self, p: float) -> float:
        return beta.cdf(p, self._ahat, self._bhat)

    def update(self, p: float) -> None:
        p_safe = np.clip(p, 1e-12, 1 - 1e-12)
        self._n += 1
        self._log_sum_x += np.log(p_safe)
        self._log_sum_1_minus_x += np.log(1 - p_safe)

        if self._n >= 2:
            n = self._n
            ls_x = self._log_sum_x
            ls_1mx = self._log_sum_1_minus_x

            def neg_ll(params):
                a, b = params
                if a <= 0 or b <= 0:
                    return np.inf
                return -((a - 1) * ls_x + (b - 1) * ls_1mx - n * betaln(a, b))

            result = minimize(neg_ll, [self._ahat, self._bhat], bounds=[(1e-5, 1e5), (1e-5, 1e5)], method="L-BFGS-B")
            if result.success and np.all(np.isfinite(result.x)):
                self._ahat, self._bhat = result.x


class ParticleFilterStrategy(BettingStrategy):
    """Particle filter betting strategy for adaptive Beta distribution estimation.

    Maintains a particle cloud in (log-alpha, log-beta) space and uses
    sequential Monte Carlo to track the evolving distribution of p-values.

    Parameters
    ----------
    num_particles : int
        Number of particles.
    process_noise_std : float or "auto"
        Standard deviation of the random walk noise. "auto" learns volatility.
    vol_noise_std : float
        Noise on the log-volatility process (when process_noise_std="auto").
    seed : int, np.random.Generator, or None
        Random seed or Generator for reproducibility.

    Examples
    --------
    >>> pf = ParticleFilterStrategy(num_particles=100, seed=42)
    >>> for p in [0.1, 0.1, 0.1]:
    ...     pf.update(p)
    >>> bool(pf.bet(0.1) > 1.0)  # should favor 0.1
    True
    """

    def __init__(self, num_particles=1000, process_noise_std=0.05, vol_noise_std=0.01, seed=None):
        self.N = num_particles
        self.adaptive_noise = process_noise_std == "auto"
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

        if self.adaptive_noise:
            self.dim_x = 3
            self.vol_noise_std = vol_noise_std
            self.particles = self.rng.standard_normal((self.N, 3)) * 0.1
            self.particles[:, 2] = np.log(0.05) + self.rng.standard_normal(self.N) * 0.1
        else:
            self.dim_x = 2
            self.Q = np.eye(2) * (process_noise_std**2)
            self.particles = self.rng.standard_normal((self.N, 2)) * 0.1

        self.weights = np.ones(self.N) / self.N
        self.sigma_history = []
        self.alpha_history = []
        self.beta_history = []

    def _predict(self):
        """Propagate particles according to the motion model."""
        if self.adaptive_noise:
            vol_noise = self.rng.standard_normal(self.N) * self.vol_noise_std
            self.particles[:, 2] += vol_noise
            current_sigmas = np.exp(self.particles[:, 2])
            main_noise = self.rng.standard_normal((self.N, 2)) * current_sigmas[:, np.newaxis]
            self.particles[:, :2] += main_noise
            self.sigma_history.append(np.median(current_sigmas))
        else:
            noise = self.rng.multivariate_normal(mean=np.zeros(2), cov=self.Q, size=self.N)
            self.particles += noise

    def _update_weights(self, p_obs):
        """Calculate particle weights based on the likelihood of p_obs."""
        p_obs_clipped = np.clip(p_obs, 1e-9, 1 - 1e-9)
        alphas = np.exp(self.particles[:, 0])
        betas_param = np.exp(self.particles[:, 1])
        likelihood = beta.pdf(p_obs_clipped, a=alphas, b=betas_param) + 1e-9

        if np.sum(likelihood) > 0:
            self.weights = likelihood / np.sum(likelihood)
        else:
            self.weights.fill(1.0 / self.N)

    def _resample(self):
        """Resample particles using systematic resampling."""
        indices = self._systematic_resample(self.weights, self.rng)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.N)

    @staticmethod
    def _systematic_resample(weights, rng):
        N = len(weights)
        positions = (rng.random() + np.arange(N)) / N
        indexes = np.zeros(N, "i")
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def _get_alphas_betas(self):
        """Safe extraction of alpha and beta values with overflow protection."""
        # Clip particles to prevent overflow (max exponent for float64 is ~709)
        log_alphas = np.clip(self.particles[:, 0], -700, 700)
        log_betas = np.clip(self.particles[:, 1], -700, 700)
        alphas = np.exp(log_alphas)
        betas = np.exp(log_betas)
        # Replace any inf or nan with sensible default
        alphas = np.where(np.isfinite(alphas), alphas, 1.0)
        betas = np.where(np.isfinite(betas), betas, 1.0)
        return alphas, betas

    def bet(self, p: float) -> float:
        alphas, betas_param = self._get_alphas_betas()
        return float(np.mean(beta.pdf(p, a=alphas, b=betas_param)))

    def integrate(self, p: float) -> float:
        alphas, betas_param = self._get_alphas_betas()
        return float(np.mean(beta.cdf(p, a=alphas, b=betas_param)))

    def update(self, p: float) -> None:
        self._predict()
        self._update_weights(p)
        self._resample()

        alphas, betas_param = self._get_alphas_betas()
        # Only record finite values
        alpha_vals = alphas[np.isfinite(alphas)]
        beta_vals = betas_param[np.isfinite(betas_param)]
        if len(alpha_vals) > 0 and len(beta_vals) > 0:
            self.alpha_history.append(np.percentile(alpha_vals, [50, 2.5, 97.5]))
            self.beta_history.append(np.percentile(beta_vals, [50, 2.5, 97.5]))

    def plot_parameters(self, title="Particle Filter Parameter Evolution"):
        """Plot the evolution of the learned Beta parameters."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib is required for plotting.", stacklevel=2)
            return

        if not self.alpha_history:
            warnings.warn("No history to plot.", stacklevel=2)
            return

        alphas = np.array(self.alpha_history)
        betas = np.array(self.beta_history)
        steps = np.arange(len(alphas))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(steps, alphas[:, 0], label="Alpha (Median)", color="blue")
        ax1.fill_between(steps, alphas[:, 1], alphas[:, 2], color="blue", alpha=0.2, label="95% CI")
        ax1.set_ylabel("Alpha")
        ax1.set_title("Evolution of Alpha")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, betas[:, 0], label="Beta (Median)", color="green")
        ax2.fill_between(steps, betas[:, 1], betas[:, 2], color="green", alpha=0.2, label="95% CI")
        ax2.set_ylabel("Beta")
        ax2.set_xlabel("Time Step")
        ax2.set_title("Evolution of Beta")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class FixedStrategy(BettingStrategy):
    """A strategy that uses a fixed, unchanging density function.

    Accepts either a scipy-like distribution object OR explicit pdf/cdf callables.

    Parameters
    ----------
    distribution : object with .pdf() and .cdf() methods
        E.g., a scipy.stats distribution.
    pdf : callable
        Explicit PDF (overrides distribution).
    cdf : callable
        Explicit CDF. If pdf given without cdf, numerical integration is used.
    check_integration : bool
        Verify that the PDF integrates to 1.

    Examples
    --------
    >>> fs = FixedStrategy(distribution=uniform())
    >>> fs.bet(0.5)
    1.0
    >>> fs2 = FixedStrategy(pdf=lambda x: 2 * (1 - x))
    >>> fs2.bet(0.1)
    1.8
    """

    def __init__(self, distribution=None, pdf=None, cdf=None, check_integration=True):
        if pdf is not None:
            self._pdf = pdf
            if cdf is not None:
                self._cdf = cdf
            else:

                def numerical_cdf(x):
                    if np.ndim(x) == 0:
                        return quad(self._pdf, 0, x, limit=50)[0]
                    return np.array([quad(self._pdf, 0, val, limit=50)[0] for val in x])

                self._cdf = numerical_cdf
        elif distribution is not None:
            self._pdf = distribution.pdf
            self._cdf = distribution.cdf
        else:
            u = uniform()
            self._pdf = u.pdf
            self._cdf = u.cdf

        if check_integration:
            try:
                total_prob, _ = quad(self._pdf, 0, 1, limit=100)
                if not np.isclose(total_prob, 1.0, atol=1e-3):
                    warnings.warn(
                        f"FixedStrategy PDF does not integrate to 1 "
                        f"(integral={total_prob:.4f}). This may invalidate the martingale.",
                        stacklevel=2,
                    )
            except Exception as e:
                warnings.warn(f"Could not verify FixedStrategy PDF integration: {e}", stacklevel=2)

    def bet(self, p: float) -> float:
        return float(self._pdf(p))

    def integrate(self, p: float) -> float:
        return float(self._cdf(p))


class PiecewiseConstantBetting(BettingStrategy):
    """Piecewise-constant betting function f_{(a,b)} from ALRW2 §9.2.

    The betting function is defined as:

        f_{(a,b)}(p) = b/a          if p <= a
        f_{(a,b)}(p) = (1-b)/(1-a)  if p > a

    This integrates to 1 over [0,1] for any a, b in (0,1), making it a valid
    betting density. It bets that fraction b of the probability mass falls
    below threshold a.

    Parameters
    ----------
    a : float
        Threshold in (0, 1). Splits the domain into [0, a] and (a, 1].
    b : float
        Probability mass allocated to [0, a]. Must be in (0, 1).

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §9.2. Cambridge University Press.

    Examples
    --------
    >>> pcb = PiecewiseConstantBetting(a=0.3, b=0.5)
    >>> pcb.bet(0.1)  # b/a = 0.5/0.3
    1.6666666666666667
    >>> pcb.bet(0.8)  # (1-b)/(1-a) = 0.5/0.7
    0.7142857142857143
    >>> abs(pcb.integrate(1.0) - 1.0) < 1e-10
    True
    """

    def __init__(self, a, b):
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}")
        if not (0 < b < 1):
            raise ValueError(f"b must be in (0, 1), got {b}")
        self.a = a
        self.b = b
        self._left = b / a
        self._right = (1 - b) / (1 - a)

    def bet(self, p: float) -> float:
        """Evaluate f_{(a,b)}(p)."""
        return self._left if p <= self.a else self._right

    def integrate(self, p: float) -> float:
        """CDF: (b/a)*p if p <= a, else b + ((1-b)/(1-a))*(p - a)."""
        if p <= self.a:
            return self._left * p
        return self.b + self._right * (p - self.a)

    def update(self, p: float) -> None:
        """No-op: the piecewise-constant function is static."""
        pass


class ExpertAggregationStrategy(BettingStrategy):
    """Exponentially Weighted Average aggregation of expert betting strategies.

    Maintains a portfolio over multiple expert strategies and reweights
    based on their performance (log-gains).

    Parameters
    ----------
    experts : list of BettingStrategy
        The expert strategies to aggregate.
    learning_rate : float
        Step size for the exponential weights update.
    base_alpha : float
        Base mixing rate towards uniform for regularization.

    Examples
    --------
    >>> expert_good = FixedStrategy(
    ...     pdf=lambda x: norm.pdf(x, 0.1, 0.1) / (norm.cdf(1, 0.1, 0.1) - norm.cdf(0, 0.1, 0.1)),
    ...     check_integration=False,
    ... )
    >>> expert_bad = FixedStrategy(distribution=uniform())
    >>> agg = ExpertAggregationStrategy(experts=[expert_good, expert_bad])
    >>> agg.update(0.1)
    >>> w = agg.get_current_weights()
    >>> bool(w[0] > w[1])
    True
    """

    def __init__(self, experts, learning_rate=0.1, base_alpha=0.01):
        self.experts = experts
        self.num_experts = len(experts)
        self.learning_rate = learning_rate
        self.base_alpha = base_alpha
        self.log_weights = np.zeros(self.num_experts)
        self.expert_weights_history = []

    def get_current_weights(self):
        """Calculate current linear weights from log-weights."""
        lse = logsumexp(self.log_weights)
        return np.exp(self.log_weights - lse)

    def bet(self, p: float) -> float:
        weights = self.get_current_weights()
        return float(np.dot(weights, [expert.bet(p) for expert in self.experts]))

    def integrate(self, p: float) -> float:
        weights = self.get_current_weights()
        return float(np.dot(weights, [expert.integrate(p) for expert in self.experts]))

    def update(self, p: float) -> None:
        # 1. Update log-weights based on expert performance
        gains = np.array([expert.bet(p) for expert in self.experts])
        log_gains = np.log(np.maximum(gains, 1e-12))
        self.log_weights += self.learning_rate * log_gains

        # 2. Regularize: mix towards uniform (prevents weight collapse)
        weights = self.get_current_weights()
        final_weights = (1 - self.base_alpha) * weights + self.base_alpha / self.num_experts
        self.log_weights = np.log(np.maximum(final_weights, 1e-12))

        self.expert_weights_history.append(final_weights.copy())

        # 3. Update each expert (predict-then-learn order)
        for expert in self.experts:
            expert.update(p)

    def plot_weights(self, expert_names=None, title="Evolution of Expert Weights", ax=None):
        """Plot the evolution of expert weights over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib is required for plotting.", stacklevel=2)
            return

        if not self.expert_weights_history:
            warnings.warn("No weight history to plot.", stacklevel=2)
            return

        history = np.array(self.expert_weights_history)
        steps = np.arange(len(history))

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(self.num_experts):
            label = expert_names[i] if expert_names and i < len(expert_names) else f"Expert {i + 1}"
            ax.plot(steps, history[:, i], label=label)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Weight")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()
            plt.show()
