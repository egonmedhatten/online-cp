"""Conformal test martingales and betting strategies.

This module implements conformal test martingales for testing the exchangeability
assumption online, along with various betting strategies (parametric, non-parametric,
particle filter, expert aggregation) for constructing the martingale.
"""

import math
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammainc, gammaln, logsumexp
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

_NUMBA_BROKEN = False  # Set True on first numba runtime failure

__all__ = [
    "PluginMartingale",
    "SimpleJumper",
    "CompositeJumper",
    "SimpleMixtureMartingale",
    "SleeperStayer",
    "SleeperDrifter",
    "CUSUMWrapper",
    "ShiryaevRobertsWrapper",
    "PiecewiseConstantBetting",
    "BetaKernel",
    "GaussianKDE",
    "BetaMoments",
    "BetaMLE",
    "ParticleFilterStrategy",
    "FixedStrategy",
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

# ═══════════════════════════════════════════════════════════════════════════════
# BETTING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════


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

    def bet(self, p):
        """Return the density f(p) using current state (past data only).

        Must satisfy f(p) >= 0 and integrate to ~1 over [0,1].
        """
        return 1.0  # uniform = no betting

    def integrate(self, p):
        """Return the CDF F(p) using current state (protection function).

        Must satisfy F(0) = 0, F(1) = 1, monotone increasing.
        """
        return p  # uniform CDF

    def update(self, p):
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

    def bet(self, p):
        if len(self._data) < 2:
            return 1.0
        return float(self._kde.pdf(p, normalized=self.normalize))

    def integrate(self, p):
        if len(self._data) < 2:
            return p
        val, _ = quad(lambda x: self._kde.pdf(x, normalized=self.normalize), 0, p, limit=50)
        return float(val)

    def update(self, p):
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
        total += (
            0.5 * math.erfc(-z1 * inv_sqrt2)
            + 0.5 * math.erfc(-z2 * inv_sqrt2)
            + 0.5 * math.erfc(-z3 * inv_sqrt2)
        )
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
        global _NUMBA_BROKEN
        if HAS_NUMBA and not _NUMBA_BROKEN:
            try:
                return _reflected_kde_pdf_numba(float(x), data, h)
            except (ModuleNotFoundError, RuntimeError, TypeError):
                _NUMBA_BROKEN = True
        x = np.atleast_1d(x)
        pdf_orig = np.mean(norm.pdf(x[:, None], loc=data, scale=h), axis=1)
        pdf_neg = np.mean(norm.pdf(x[:, None], loc=-data, scale=h), axis=1)
        pdf_two = np.mean(norm.pdf(x[:, None], loc=2 - data, scale=h), axis=1)
        pdf_values = pdf_orig + pdf_neg + pdf_two
        return pdf_values.item() if pdf_values.size == 1 else pdf_values

    def _kernel_cdf_reflect(self, x, data, h):
        """Reflected Gaussian kernel CDF."""
        global _NUMBA_BROKEN
        if HAS_NUMBA and not _NUMBA_BROKEN:
            try:
                return _reflected_kde_cdf_numba(float(x), data, h)
            except (ModuleNotFoundError, RuntimeError, TypeError):
                _NUMBA_BROKEN = True
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
                objective_func, bounds=(self.bw_min, self.bw_max),
                method="bounded", options={"maxiter": self.max_iter}
            )
        else:
            search_bounds = (
                max(self._current_bw * 0.8, self.bw_min),
                min(self._current_bw * 1.2, self.bw_max),
            )
            result = minimize_scalar(
                objective_func, bounds=search_bounds,
                method="bounded", options={"maxiter": self.max_iter}
            )
        return result

    def _calculate_bandwidth(self, data):
        """Calculate bandwidth for current data."""
        n = data.size
        if self.bandwidth == "silverman":
            sigma = np.std(data)
            h = ((4 * sigma**5) / (3 * n)) ** (1 / 5)
        elif self.bandwidth == "lcv":
            should_recalculate = (
                (self._current_bw is None)
                or (n >= self._n_last_update * self.growth_factor)
            )
            if should_recalculate and n > 1:
                result = self._likelihood_lcv_bw(data)
                h = result.x
                self._n_last_update = n
            else:
                h = self._current_bw
        else:
            h = float(self.bandwidth)
        return h

    def _get_data(self):
        """Get the current data array (windowed if applicable)."""
        window = self.window_size or len(self._data)
        return np.array(self._data[-window:])

    def bet(self, p):
        if len(self._data) < 2:
            return 1.0
        data = self._get_data()
        if data.std() < 1e-6:
            return 1.0
        return self._kernel_pdf_reflect(p, data, self._current_bw)

    def integrate(self, p):
        if len(self._data) < 2:
            return p
        data = self._get_data()
        if data.std() < 1e-6:
            return p
        return self._kernel_cdf_reflect(p, data, self._current_bw)

    def update(self, p):
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

    def bet(self, p):
        return beta.pdf(p, self._ahat, self._bhat)

    def integrate(self, p):
        return beta.cdf(p, self._ahat, self._bhat)

    def update(self, p):
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

    def bet(self, p):
        return beta.pdf(p, self._ahat, self._bhat)

    def integrate(self, p):
        return beta.cdf(p, self._ahat, self._bhat)

    def update(self, p):
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

            result = minimize(
                neg_ll, [self._ahat, self._bhat],
                bounds=[(1e-5, None), (1e-5, None)]
            )
            if result.success:
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
    seed : int or None
        Random seed for reproducibility.

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
        return np.exp(self.particles[:, 0]), np.exp(self.particles[:, 1])

    def bet(self, p):
        alphas, betas_param = self._get_alphas_betas()
        return float(np.mean(beta.pdf(p, a=alphas, b=betas_param)))

    def integrate(self, p):
        alphas, betas_param = self._get_alphas_betas()
        return float(np.mean(beta.cdf(p, a=alphas, b=betas_param)))

    def update(self, p):
        self._predict()
        self._update_weights(p)
        self._resample()

        alphas, betas_param = self._get_alphas_betas()
        self.alpha_history.append(np.percentile(alphas, [50, 2.5, 97.5]))
        self.beta_history.append(np.percentile(betas_param, [50, 2.5, 97.5]))

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

    def bet(self, p):
        return float(self._pdf(p))

    def integrate(self, p):
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

    def bet(self, p):
        """Evaluate f_{(a,b)}(p)."""
        return self._left if p <= self.a else self._right

    def integrate(self, p):
        """CDF: (b/a)*p if p <= a, else b + ((1-b)/(1-a))*(p - a)."""
        if p <= self.a:
            return self._left * p
        return self.b + self._right * (p - self.a)

    def update(self, p):
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

    def bet(self, p):
        weights = self.get_current_weights()
        return float(np.dot(weights, [expert.bet(p) for expert in self.experts]))

    def integrate(self, p):
        weights = self.get_current_weights()
        return float(np.dot(weights, [expert.integrate(p) for expert in self.experts]))

    def update(self, p):
        # Update weights based on expert performance
        gains = np.array([expert.bet(p) for expert in self.experts])
        master_prediction = np.dot(self.get_current_weights(), gains)
        loss = 1.0 - np.clip(master_prediction / self.num_experts, 0, 1)
        alpha_t = self.base_alpha * loss

        log_gains = np.log(np.maximum(gains, 1e-12))
        self.log_weights += self.learning_rate * log_gains

        intermediate_weights = self.get_current_weights()
        final_weights = (1 - alpha_t) * intermediate_weights + alpha_t / self.num_experts
        final_weights /= np.sum(final_weights)
        self.log_weights = np.log(np.maximum(final_weights, 1e-12))

        self.expert_weights_history.append(final_weights.copy())

        # Then update each expert
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


# ═══════════════════════════════════════════════════════════════════════════════
# MARTINGALES
# ═══════════════════════════════════════════════════════════════════════════════


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

    def __init__(self, warnings=True, warning_level=100, store_p_values=True):
        self.logM = 0.0
        self.log_max = 0.0
        self.p_values = []
        self.store_p_values = store_p_values
        self.log_martingale_values = [0.0]
        self.warning_level = warning_level
        self.warnings = warnings
        self._warned = False
        self.b_n = lambda x: 1.0
        self.B_n = lambda x: x

    @property
    def M(self):
        return np.exp(self.logM)

    @property
    def max(self):
        return np.exp(self.log_max)

    @property
    def martingale_values(self):
        return np.exp(self.log_martingale_values)

    @property
    def log10_martingale_values(self):
        return [logM / np.log(10) for logM in self.log_martingale_values]

    def check_warning(self):
        if self.log_max >= np.log(self.warning_level) and self.warnings and not self._warned:
            warnings.warn(
                f"Exchangeability assumption likely violated: Max martingale value is {self.max}", stacklevel=2
            )
            self._warned = True

    def update(self, p):
        """Incorporate a new p-value. Subclasses must implement this."""
        raise NotImplementedError

    # Backward compatibility alias
    def update_martingale_value(self, p):
        """Deprecated: use update(p) instead."""
        return self.update(p)


class PluginMartingale(ConformalTestMartingale):
    """Plugin martingale using a betting strategy with cautious-start mixing.

    The martingale wraps a ``BettingStrategy`` and applies cautious mixing:
    during the first ``min_sample_size`` observations, the strategy's density
    is linearly mixed towards uniform to avoid catastrophic loss before the
    density estimate is reliable.

    Protocol per step:
    1. **Predict**: evaluate ``strategy.bet(p)`` (uses past data only)
    2. **Mix**: apply cautious start ``b = λ * f + (1 - λ)``
    3. **Accumulate**: ``logM += log(b)``
    4. **Learn**: call ``strategy.update(p)``
    5. **Expose**: set ``b_n`` and ``B_n`` for the *next* step

    Parameters
    ----------
    betting_strategy : BettingStrategy or type
        An instantiated strategy, or a class to be instantiated with kwargs.
    min_sample_size : int
        Number of steps over which to linearly ramp up from uniform to full betting.
    **kwargs
        Passed to the strategy constructor if a class is given.

    Examples
    --------
    >>> strat = FixedStrategy(pdf=lambda x: 2 if x < 0.5 else 0, check_integration=False)
    >>> m = PluginMartingale(betting_strategy=strat, min_sample_size=0)
    >>> m.update(0.1)
    >>> bool(np.isclose(m.M, 2.0))
    True
    >>> m.update(0.9)
    >>> bool(m.M == 0.0)
    True
    """

    def __init__(
        self,
        betting_strategy=GaussianKDE,
        min_sample_size=100,
        warnings=True,
        warning_level=100,
        store_p_values=True,
        **kwargs,
    ):
        super().__init__(warnings, warning_level, store_p_values)
        self.min_sample_size = min_sample_size
        self._n = 0

        if isinstance(betting_strategy, BettingStrategy):
            self.strategy = betting_strategy
        else:
            betting_kwargs = kwargs if kwargs else {}
            self.strategy = betting_strategy(**betting_kwargs)

        # Expose initial b_n / B_n (with mixing at step 0)
        self._update_exposed_functions()

    def _mixing_parameter(self, n):
        """Linear ramp from 0 to 1 over min_sample_size steps."""
        if self.min_sample_size == 0:
            return 1.0
        return min(n / self.min_sample_size, 1.0)

    def _update_exposed_functions(self):
        """Set b_n and B_n for the *next* step (using current strategy state)."""
        lam = self._mixing_parameter(self._n)
        strategy = self.strategy

        def b_n(x, _lam=lam, _s=strategy):
            return _lam * _s.bet(x) + (1 - _lam)

        def B_n(x, _lam=lam, _s=strategy):
            return _lam * _s.integrate(x) + (1 - _lam) * x

        self.b_n = b_n
        self.B_n = B_n

    def update(self, p):
        # 1. Predict: evaluate current betting function
        f = self.strategy.bet(p)
        F = self.strategy.integrate(p)

        # 2. Mix (cautious start)
        lam = self._mixing_parameter(self._n)
        b = lam * f + (1 - lam)

        # 3. Accumulate wealth
        self.logM += np.log(b)
        self.log_martingale_values.append(self.logM)

        if self.store_p_values:
            self.p_values.append(p)

        # 4. Learn
        self._n += 1
        self.strategy.update(p)

        # 5. Expose next step's functions
        self._update_exposed_functions()

        if self.logM > self.log_max:
            self.log_max = self.logM

        self.check_warning()


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

    def __init__(self, J=0.01, E=None, warning_level=100, warnings=True, store_p_values=True, **kwargs):
        super().__init__(warnings, warning_level, store_p_values)
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

    def update(self, p):
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
            new_log_C_epsilon[epsilon] = log_C_mixed + np.log(bet_val)

        self.log_C_epsilon = new_log_C_epsilon
        self.log_C = logsumexp(list(self.log_C_epsilon.values()))

        self.logM = self.log_C
        self.log_martingale_values.append(self.logM)

        # Effective betting function for next step: wealth-weighted mixture
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

        if self.logM > self.log_max:
            self.log_max = self.logM

        self.check_warning()


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

    def __init__(self, J=None, warnings=True, warning_level=100, store_p_values=True):
        super().__init__(warnings, warning_level, store_p_values)
        if J is None:
            self.J = [10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1]
        else:
            self.J = J

        self.Jumpers = {j: SimpleJumper(J=j, warnings=False, store_p_values=False) for j in self.J}

    def update(self, p):
        if self.store_p_values:
            self.p_values.append(p)

        for m in self.Jumpers.values():
            m.update(p)

        log_M_values = [m.logM for m in self.Jumpers.values()]
        self.logM = logsumexp(log_M_values) - np.log(len(self.Jumpers))

        self.log_martingale_values.append(self.logM)

        # Wealth-weighted betting function
        log_sum_M = self.logM + np.log(len(self.Jumpers))
        weights = np.exp(np.array(log_M_values) - log_sum_M)

        current_b_ns = [m.b_n for m in self.Jumpers.values()]
        current_B_ns = [m.B_n for m in self.Jumpers.values()]

        self.b_n = lambda u: np.dot(weights, [f(u) for f in current_b_ns])
        self.B_n = lambda u: np.dot(weights, [F(u) for F in current_B_ns])

        if self.logM > self.log_max:
            self.log_max = self.logM

        self.check_warning()


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

    def __init__(self, R=0.001, G=10, warnings=True, warning_level=100, store_p_values=True):
        super().__init__(warnings, warning_level, store_p_values)
        self.R = R
        self.G = G

        # Build the grid: (a, b) pairs with a, b in {1/G, ..., (G-1)/G}
        grid_vals = np.arange(1, G) / G
        self._grid = [(a, b) for a in grid_vals for b in grid_vals]
        self._n_experts = len(self._grid)

        # Precompute betting values for each expert: left = b/a, right = (1-b)/(1-a)
        self._left = np.array([b / a for a, b in self._grid])
        self._right = np.array([(1 - b) / (1 - a) for a, b in self._grid])
        self._thresholds = np.array([a for a, _ in self._grid])

        # Capital: all starts in sleeping account
        self._S_active = np.zeros(self._n_experts)
        self._S_sleep = 1.0
        self._n = 0

    def update(self, p):
        if self.store_p_values:
            self.p_values.append(p)

        self._n += 1

        # Step 1: Bet — multiply each active expert by f_{(a,b)}(p)
        # (Only experts that received capital in previous steps bet)
        bets = np.where(p <= self._thresholds, self._left, self._right)
        self._S_active *= bets

        # Step 2: Output — total capital is the martingale value
        S_n = self._S_sleep + self._S_active.sum()
        self.logM = np.log(max(S_n, 1e-300))
        self.log_martingale_values.append(self.logM)

        if self.logM > self.log_max:
            self.log_max = self.logM

        # Step 3: Redistribute — move fraction R of sleeping capital to active experts
        # (prepares experts for next step's bet)
        transfer = self.R * self._S_sleep / self._n_experts
        self._S_active += transfer
        self._S_sleep *= (1 - self.R)

        # Expose b_n / B_n as wealth-weighted combination of active experts
        total_active = self._S_active.sum()
        if total_active > 0:
            weights = self._S_active / total_active
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

        self.check_warning()


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

    def __init__(self, R=0.001, G=10, M=100, warnings=True, warning_level=100,
                 store_p_values=True):
        super().__init__(warnings, warning_level, store_p_values)
        self.R = R
        self.G = G
        self._batch_interval = M

        # Grid of (a, b) pairs
        grid_vals = np.arange(1, G) / G
        self._grid = [(a, b) for a in grid_vals for b in grid_vals]
        self._n_grid = len(self._grid)

        # Active experts: dict of (batch_index, grid_index) -> capital
        self._experts = {}  # (i, j) -> capital
        self._S_sleep = 1.0
        self._n = 0
        self._prune_threshold = 1e-15

    def _get_drifted_threshold(self, batch_idx, grid_idx):
        """Compute a' = (i*M/n)*a + (1 - i*M/n)*b for expert (i, a, b)."""
        a, b = self._grid[grid_idx]
        ratio = (batch_idx * self._batch_interval) / self._n
        ratio = min(ratio, 1.0)  # Clamp
        return ratio * a + (1 - ratio) * b

    def update(self, p):
        if self.store_p_values:
            self.p_values.append(p)

        self._n += 1

        # Step 1: Bet — update all active experts
        keys_to_remove = []
        for key, capital in self._experts.items():
            batch_idx, grid_idx = key
            a_prime = self._get_drifted_threshold(batch_idx, grid_idx)
            _, b = self._grid[grid_idx]

            # Bet using f_{(a', b)}(p)
            if a_prime <= 0 or a_prime >= 1:
                bet_val = 1.0  # Degenerate case, no bet
            else:
                bet_val = b / a_prime if p <= a_prime else (1 - b) / (1 - a_prime)

            new_capital = capital * bet_val
            if new_capital < self._prune_threshold:
                keys_to_remove.append(key)
            else:
                self._experts[key] = new_capital

        for key in keys_to_remove:
            del self._experts[key]

        # Step 2: Output — total capital
        total_active = sum(self._experts.values())
        S_n = self._S_sleep + total_active
        self.logM = np.log(max(S_n, 1e-300))
        self.log_martingale_values.append(self.logM)

        if self.logM > self.log_max:
            self.log_max = self.logM

        # Step 3: Wake new batch (if n is divisible by M) — prepares for next step
        if self._n % self._batch_interval == 0:
            batch_idx = self._n // self._batch_interval
            transfer_per_expert = self.R * self._batch_interval * self._S_sleep / self._n_grid
            self._S_sleep *= (1 - self.R * self._batch_interval)
            # Clamp sleeping capital
            self._S_sleep = max(self._S_sleep, 0.0)
            for j in range(self._n_grid):
                key = (batch_idx, j)
                self._experts[key] = self._experts.get(key, 0.0) + transfer_per_expert

        # Expose b_n / B_n as wealth-weighted function for next step
        total_active = sum(self._experts.values())
        if total_active > 0 and self._experts:
            expert_items = list(self._experts.items())
            total = total_active
            n_next = self._n + 1

            def _b_n(u, _items=expert_items, _total=total, _n=n_next,
                     _grid=self._grid, _M=self._batch_interval):
                val = 0.0
                for (bi, gi), cap in _items:
                    a, b = _grid[gi]
                    ratio = min((bi * _M) / _n, 1.0)
                    a_prime = ratio * a + (1 - ratio) * b
                    if 0 < a_prime < 1:
                        f = b / a_prime if u <= a_prime else (1 - b) / (1 - a_prime)
                    else:
                        f = 1.0
                    val += (cap / _total) * f
                return val

            self.b_n = _b_n
            self.B_n = lambda u: u  # No analytic CDF for drifting mixture

        self.check_warning()


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
            log_incomplete_gamma = -700
        else:
            log_incomplete_gamma = log_gamma_n_plus_1 + np.log(val_gammainc)
        return -L + log_incomplete_gamma - (n + 1) * np.log(arg)

    def _update_b_n(self):
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

    def update(self, p):
        self.n += 1
        p_clipped = np.clip(p, 1e-12, 1.0)
        self.sum_log_p += np.log(p_clipped)

        self.logM = self._compute_logM(self.n, self.sum_log_p)

        if self.store_p_values:
            self.p_values.append(p)
        self.log_martingale_values.append(self.logM)

        if self.logM > self.log_max:
            self.log_max = self.logM

        self._update_b_n()
        self.check_warning()


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

    def update(self, p):
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

    def update(self, p):
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


if __name__ == "__main__":
    import doctest
    import sys

    # Run tests and provide feedback
    print("Running doctests...")
    # Suppress warnings during testing to keep output clean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (failures, tests) = doctest.testmod()

    if failures:
        print(f"FAILED: {failures} out of {tests} tests failed.")
        sys.exit(1)
    else:
        print(f"Success: {tests} tests passed.")
