"""Conformal test martingales and betting strategies.

This module implements conformal test martingales for testing the exchangeability
assumption online, along with various betting strategies (parametric, non-parametric,
particle filter, expert aggregation) for constructing the martingale.
"""

import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammainc, gammaln, logsumexp
from scipy.stats import beta, norm, uniform

__all__ = [
    "PluginMartingale",
    "SimpleJumper",
    "CompositeJumper",
    "OnionMartingale",
    "SimpleMixtureMartingale",
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

# TODO
# * Remove online update of statistics in the parametric models. It is better to have the possibility to use a window
# * Tune the COEFFICIENTS in the adaptive rule
# * Consider adding plotting functions for the ParticleFilter and ExpertAggregation (as in the notebook)


class BettingStrategy:
    """
    Base class for betting strategies.

    Attributes:
        min_sample_size (int): Minimum number of samples required before full betting.
        mixing_parameter (float): Mixing parameter for cautious betting.
    """

    def __init__(self, min_sample_size=100):
        """
        Initializes the BettingStrategy class.

        Args:
            min_sample_size (int): Minimum number of samples required before full betting.
        """
        self.min_sample_size = min_sample_size
        self.mixing_parameter = 0

    def update_mixing_parameter(self, n):
        """
        Updates the mixing parameter based on the number of observations.

        Args:
            n (int): Number of observations.
        """
        # Avoid division by zero if min_sample_size is 0
        if self.min_sample_size == 0:
            self.mixing_parameter = 1.0
        else:
            self.mixing_parameter = min(n / self.min_sample_size, 1)


class BetaKernel(BettingStrategy):
    """
    Implements a Beta Kernel Density Estimation (KDE) betting strategy.

    Attributes:
        bandwidth (str or float): Bandwidth selection method.
        window_size (str or int): Window size for adaptive KDE.
        growth_factor (float): The sample size must increase by this factor to trigger a
                               new bandwidth calculation.

    Examples:
        >>> bk = BetaKernel(min_sample_size=1000)
        >>> p_vals = [0.1, 0.2, 0.15, 0.05, 0.1]
        >>> # With large min_sample_size, mixing parameter is small, result is close to uniform
        >>> b_func, B_func = bk.update_betting_function(p_vals)
        >>> abs(b_func(0.5) - 1.0) < 0.1
        True
    """

    def __init__(
        self,
        bandwidth="beta-reference",
        window_size=None,
        normalize=True,
        min_sample_size=100,
        max_iter=20,
        bw_min=0.001,
        bw_max=0.5,
        growth_factor=1.1,
    ):
        super().__init__(min_sample_size)
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.max_iter = max_iter
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.growth_factor = growth_factor

        self.current_bw = None
        self.n_last_update = 0  # To track sample size at last bandwidth update

        self.kde = BetaKDE(bandwidth=self.bandwidth)
        self.normalize = normalize

    def update_betting_function(self, p_values):
        """
        Updates the betting function based on the provided p-values.
        """
        if len(p_values) < 2:
            pdf = lambda x: beta.pdf(x, 1, 1)
            # cdf for uniform(0,1) is x
            cdf = lambda x: x
        else:
            # OPTIMIZATION: Slice the list BEFORE converting to numpy array
            window = self.calculate_window_size(p_values)
            data = np.array(p_values[-window:])

            self.kde.fit(data.reshape(-1, 1), compute_normalization=self.normalize)

            # Define the learned pdf
            # We capture self.kde.pdf carefully.
            # Note: self.kde changes state, so the lambda binds to the object, which is correct for "current" state.
            def learned_pdf(x):
                return self.kde.pdf(x, normalized=self.normalize)

            pdf = learned_pdf

            # Define CDF via numerical integration if analytical not available
            def learned_cdf(x):
                # Safely handle array inputs for integration
                if np.ndim(x) == 0:
                    val, _ = quad(pdf, 0, x, limit=50)
                    return val
                else:
                    return np.array([quad(pdf, 0, val, limit=50)[0] for val in x])

            cdf = learned_cdf

        self.update_mixing_parameter(len(p_values))

        b_n = lambda x: self.mixing_parameter * pdf(x) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * cdf(x) + (1 - self.mixing_parameter) * x

        return b_n, B_n

    def calculate_window_size(self, p_values):
        if self.window_size is None:
            return len(p_values)
        else:
            return self.window_size


class GaussianKDE(BettingStrategy):
    """
    Implements a Gaussian Kernel Density Estimation (KDE) betting strategy.

    Examples:
        >>> # Initialize strategy
        >>> gkde = GaussianKDE(min_sample_size=10, bandwidth=0.1)
        >>> # Feed it some data clustered around 0.1
        >>> p_vals = [0.1, 0.11, 0.09, 0.12, 0.08]
        >>> b_func, B_func = gkde.update_betting_function(p_vals)
        >>> # After 5 samples with min_sample_size=10, mixing param is 0.5.
        >>> # The kernel should peak around 0.1.
        >>> b_func(0.1) > 1.0
        True
        >>> b_func(0.9) < 1.0
        True
    """

    def __init__(
        self,
        bandwidth="silverman",
        window_size=None,
        min_sample_size=100,
        max_iter=20,
        bw_min=0.001,
        bw_max=0.5,
        growth_factor=1.1,
    ):
        super().__init__(min_sample_size)
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.max_iter = max_iter
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.growth_factor = growth_factor

        self.current_bw = None
        self.n_last_update = 0

    def _kernel_pdf_reflect(self, x, data, h):
        """Helper to compute the reflected kernel PDF for a given dataset and bandwidth."""
        x = np.atleast_1d(x)
        # Using broadcasting to compute differences
        # data shape: (N,), x shape: (M,) -> (M, N)
        # However, to be memory efficient with large N, we stick to the original structure or loop if needed.
        # The original code used x[:, None] (M,1) against data (N,) which creates (M,N) matrix.
        pdf_orig = np.mean(norm.pdf(x[:, None], loc=data, scale=h), axis=1)
        pdf_neg = np.mean(norm.pdf(x[:, None], loc=-data, scale=h), axis=1)
        pdf_two = np.mean(norm.pdf(x[:, None], loc=2 - data, scale=h), axis=1)
        pdf_values = pdf_orig + pdf_neg + pdf_two
        return pdf_values.item() if pdf_values.size == 1 else pdf_values

    def _kernel_cdf_reflect(self, x, data, h):
        """Helper to compute the reflected kernel CDF for a given dataset and bandwidth."""
        x = np.atleast_1d(x)
        cdf_orig = np.mean(norm.cdf(x[:, None], loc=data, scale=h), axis=1)
        cdf_neg = np.mean(norm.cdf(x[:, None], loc=-data, scale=h), axis=1)
        cdf_two = np.mean(norm.cdf(x[:, None], loc=2 - data, scale=h), axis=1)
        cdf_values = cdf_orig + cdf_neg + cdf_two - 1
        return cdf_values.item() if cdf_values.size == 1 else cdf_values

    def _likelihood_lcv_bw(self, data):
        """
        Finds the optimal bandwidth by optimizing the Likelihood Cross-Validation (LCV) score.
        """
        n = len(data)
        data_col = data[:, np.newaxis]
        diff_orig = data_col - data
        diff_neg = data_col + data
        diff_two = data_col - (2 - data)

        def objective_func(h):
            """The negative LCV log-likelihood score function, to be minimized."""
            epsilon = 1e-10
            # Calculation of the leave-one-out density
            # Note: We subtract the diagonal term (kernel at 0 distance) later or fill diagonal
            kernel_matrix_orig = norm.pdf(diff_orig / h)
            kernel_matrix_neg = norm.pdf(diff_neg / h)
            kernel_matrix_two = norm.pdf(diff_two / h)
            total_kernel_matrix = kernel_matrix_orig + kernel_matrix_neg + kernel_matrix_two

            # Remove the contribution of the point itself (diagonal of the first matrix)
            # The reflection points (neg and two) are far enough that they are not 'the point itself' in the LOO sense usually,
            # but technically LOO means we remove x_i from the dataset.
            # If we remove x_i, we remove it from orig, neg, and two reflections.
            np.fill_diagonal(total_kernel_matrix, 0)

            loo_pdfs = np.sum(total_kernel_matrix, axis=1) / ((n - 1) * h)
            loo_log_likelihood = np.sum(np.log(loo_pdfs + epsilon))
            return -loo_log_likelihood

        if n < self.min_sample_size or self.current_bw is None:
            result = minimize_scalar(
                objective_func, bounds=(self.bw_min, self.bw_max), method="bounded", options={"maxiter": self.max_iter}
            )
        else:
            search_bounds = (max(self.current_bw * 0.8, self.bw_min), min(self.current_bw * 1.2, self.bw_max))
            result = minimize_scalar(
                objective_func, bounds=search_bounds, method="bounded", options={"maxiter": self.max_iter}
            )

        return result

    def update_betting_function(self, p_values):
        """
        Updates the betting function based on the provided p-values.
        """
        if len(p_values) < 2:
            pdf = lambda x: beta.pdf(x, 1, 1)
            cdf = lambda x: beta.cdf(x, 1, 1)
        else:
            # OPTIMIZATION: Slice list before array conversion
            window = self.calculate_window_size(p_values)
            data = np.array(p_values[-window:])

            if data.std() < 1e-6:
                pdf = lambda x: beta.pdf(x, 1, 1)
                cdf = lambda x: beta.cdf(x, 1, 1)
            else:
                h = self.calculate_bandwidth(data)
                self.current_bw = h

                # Bind data and h to the lambda
                pdf = lambda x: self._kernel_pdf_reflect(x, data, h)
                cdf = lambda x: self._kernel_cdf_reflect(x, data, h)

        self.update_mixing_parameter(len(p_values))

        b_n = lambda x: self.mixing_parameter * pdf(x) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * cdf(x) + (1 - self.mixing_parameter) * x

        return b_n, B_n

    def calculate_bandwidth(self, data):
        """
        Calculates the bandwidth, using the growth_factor to decide when to run the expensive optimization.
        """
        n = data.size

        if self.bandwidth == "silverman":
            sigma = np.std(data)
            h = ((4 * sigma**5) / (3 * n)) ** (1 / 5)  # Silverman's rule of thumb

        elif self.bandwidth == "lcv":
            should_recalculate = (
                (self.current_bw is None)
                or (n < self.min_sample_size)
                or (n >= self.n_last_update * self.growth_factor)
            )

            if should_recalculate and n > 1:
                self.opt_result = self._likelihood_lcv_bw(data)
                h = self.opt_result.x
                self.n_last_update = n
            else:
                h = self.current_bw

        else:  # Fixed bandwidth
            h = float(self.bandwidth)
        return h

    def calculate_window_size(self, p_values):
        if self.window_size is None:
            return len(p_values)
        else:
            return self.window_size


class BetaMoments(BettingStrategy):
    """
    Implements a betting strategy based on Beta distribution moments.

    Examples:
        >>> bm = BetaMoments(min_sample_size=10)
        >>> # Feed highly skewed data (lots of small p-values)
        >>> p_vals = [0.01, 0.02, 0.05, 0.01, 0.03]
        >>> for p in p_vals:
        ...     _ = bm.update_betting_function([p])
        >>> b_func, _ = bm.update_betting_function([0.01])
        >>> # Strategy should now bet heavily on small p-values
        >>> b_func(0.01) > 1.0
        True
        >>> b_func(0.99) < 1.0
        True
    """

    def __init__(self, min_sample_size=100):
        super().__init__(min_sample_size)
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.ahat = 1.0
        self.bhat = 1.0

    def update_betting_function(self, p_values):
        if not p_values:
            b_n = lambda x: beta.pdf(x, 1, 1)
            B_n = lambda x: beta.cdf(x, 1, 1)
            return b_n, B_n

        p = p_values[-1]
        if self.n < 2:
            self.ahat = 1.0
            self.bhat = 1.0
        else:
            sample_variance = self.M2 / (self.n - 1) if self.n > 1 else 0
            if sample_variance <= 0:
                self.ahat = 1
                self.bhat = 1
            else:
                # Method of moments estimation for Beta
                common_factor = (self.mean * (1 - self.mean) / sample_variance) - 1
                self.ahat = max(self.mean * common_factor, 1e-4)  # Avoid <= 0
                self.bhat = max((1 - self.mean) * common_factor, 1e-4)

        self.n += 1
        delta = p - self.mean
        self.mean += delta / self.n
        delta2 = p - self.mean
        self.M2 += delta * delta2

        self.update_mixing_parameter(self.n)

        # Capture current ahat/bhat
        current_a, current_b = self.ahat, self.bhat

        b_n = lambda x: self.mixing_parameter * beta.pdf(x, current_a, current_b) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * beta.cdf(x, current_a, current_b) + (1 - self.mixing_parameter) * x

        return b_n, B_n


class BetaMLE(BettingStrategy):
    """
    Implements a betting strategy based on Maximum Likelihood Estimation (MLE) for Beta distribution parameters.

    Examples:
        >>> bmle = BetaMLE(min_sample_size=10)
        >>> # Feed skewed data
        >>> p_vals = [0.01, 0.02, 0.01]
        >>> for p in p_vals:
        ...     _ = bmle.update_betting_function([p])
        >>> b_func, _ = bmle.update_betting_function([0.01])
        >>> # Should bet on small values
        >>> b_func(0.01) > 1.0
        True
    """

    def __init__(self, min_sample_size=100):
        super().__init__(min_sample_size)
        self.n = 0
        self.log_sum_x = 0.0
        self.log_sum_1_minus_x = 0.0
        self.ahat = 1.0
        self.bhat = 1.0

    def update_betting_function(self, p_values):
        if not p_values:
            b_n = lambda x: beta.pdf(x, 1, 1)
            B_n = lambda x: beta.cdf(x, 1, 1)
            return b_n, B_n

        p = p_values[-1]

        def negative_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            log_likelihood = (
                (alpha - 1) * self.log_sum_x + (beta - 1) * self.log_sum_1_minus_x - self.n * betaln(alpha, beta)
            )
            return -log_likelihood

        # Update sufficient statistics
        # Clip p to avoid log(0)
        p_safe = np.clip(p, 1e-12, 1 - 1e-12)
        self.n += 1
        self.log_sum_x += np.log(p_safe)
        self.log_sum_1_minus_x += np.log(1 - p_safe)

        if self.n < 2:
            self.ahat = 1.0
            self.bhat = 1.0
        else:
            initial_guess = [self.ahat, self.bhat]  # Use previous estimate as warm start
            result = minimize(negative_log_likelihood, initial_guess, bounds=[(1e-5, None), (1e-5, None)])
            if result.success:
                self.ahat, self.bhat = result.x
            else:
                # If optimization fails, stick to previous or default
                pass

        self.update_mixing_parameter(self.n)

        current_a, current_b = self.ahat, self.bhat

        b_n = lambda x: self.mixing_parameter * beta.pdf(x, current_a, current_b) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * beta.cdf(x, current_a, current_b) + (1 - self.mixing_parameter) * x

        return b_n, B_n


class ParticleFilterStrategy(BettingStrategy):
    """
    A betting strategy based on a particle filter.

    Examples:
        >>> # Test with seed for reproducibility
        >>> pf = ParticleFilterStrategy(num_particles=100, seed=42, min_sample_size=10)
        >>> p_vals = [0.1, 0.1, 0.1]
        >>> # Update logic
        >>> b_func, _ = pf.update_betting_function(p_vals)
        >>> # Given low p-values, the density at 0.1 should be high
        >>> b_func(0.1) > 1.0
        True
    """

    def __init__(self, num_particles=1000, process_noise_std=0.05, vol_noise_std=0.01, min_sample_size=100, seed=None):

        super().__init__(min_sample_size)
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
        """Propagates particles according to the motion model."""
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

    def _update(self, p_obs):
        """Calculates particle weights based on the likelihood of p_obs."""
        p_obs_clipped = np.clip(p_obs, 1e-9, 1 - 1e-9)
        alpha, beta_param = np.exp(self.particles[:, 0]), np.exp(self.particles[:, 1])
        likelihood = beta.pdf(p_obs_clipped, a=alpha, b=beta_param) + 1e-9

        if np.sum(likelihood) > 0:
            self.weights = likelihood / np.sum(likelihood)
        else:
            self.weights.fill(1.0 / self.N)

    def _resample(self):
        """Resamples particles using a reproducible systematic resampling."""
        indices = self.systematic_resample(self.weights, self.rng)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.N)

    @staticmethod
    def systematic_resample(weights, rng):
        """Performs the systemic resampling algorithm used by particle filters."""
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

    def update_betting_function(self, p_values):
        """Updates the filter and returns the predictive functions for the next step."""
        if not p_values:
            return (lambda x: 1.0), (lambda x: x)
        p = p_values[-1]
        self._predict()
        self._update(p)
        self._resample()

        alphas = np.exp(self.particles[:, 0])
        betas = np.exp(self.particles[:, 1])

        self.alpha_history.append(np.percentile(alphas, [50, 2.5, 97.5]))
        self.beta_history.append(np.percentile(betas, [50, 2.5, 97.5]))

        # Use simple mean of the particle distributions for the predictive density
        def learned_pdf(x):
            return np.mean(beta.pdf(x, a=alphas, b=betas))

        def learned_cdf(x):
            return np.mean(beta.cdf(x, a=alphas, b=betas))

        self.update_mixing_parameter(len(p_values))

        def b_n(x):
            return self.mixing_parameter * learned_pdf(x) + (1 - self.mixing_parameter) * 1.0

        def B_n(x):
            return self.mixing_parameter * learned_cdf(x) + (1 - self.mixing_parameter) * x

        return b_n, B_n

    def plot_parameters(self, title="Particle Filter Parameter Evolution"):
        """
        Plots the evolution of the learned Beta parameters (alpha and beta).
        Shows median and 95% confidence intervals.
        """
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

        # Alpha plot
        ax1.plot(steps, alphas[:, 0], label="Alpha (Median)", color="blue")
        ax1.fill_between(steps, alphas[:, 1], alphas[:, 2], color="blue", alpha=0.2, label="95% CI")
        ax1.set_ylabel("Alpha")
        ax1.set_title("Evolution of Alpha")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Beta plot
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
    """
    A strategy that uses a fixed, unchanging betting function.
    Accepts either a scipy-like distribution object OR explicit pdf/cdf callables.

    This strategy does NOT use the mixing parameter logic.

    Examples:
        >>> # Example 1: Using scipy distribution
        >>> fs = FixedStrategy(distribution=uniform())
        >>> b, _ = fs.update_betting_function([])
        >>> b(0.5)
        1.0

        >>> # Example 2: Explicit PDF
        >>> # A density that bets on small p-values: f(x) = 2(1-x)
        >>> fs2 = FixedStrategy(pdf=lambda x: 2 * (1 - x))
        >>> b, _ = fs2.update_betting_function([])
        >>> b(0.1)
        1.8
    """

    def __init__(self, distribution=None, pdf=None, cdf=None, check_integration=True):
        """
        Initializes the FixedStrategy.

        Args:
            distribution: An object with .pdf() and .cdf() methods (e.g., from scipy.stats).
            pdf (callable): A probability density function (optional, overrides distribution if both provided).
            cdf (callable): A cumulative distribution function (optional). If pdf is provided but cdf is not,
                            numerical integration will be used.
            check_integration (bool): Whether to verify that the PDF integrates to 1.
        """
        # Pass 0 to super to imply full mixing, though we override the logic below anyway.
        super().__init__(min_sample_size=0)

        self.base_pdf = None
        self.base_cdf = None

        # Logic to determine pdf/cdf
        if pdf is not None:
            self.base_pdf = pdf
            if cdf is not None:
                self.base_cdf = cdf
            else:
                # Fallback to numerical integration for CDF
                def numerical_cdf(x):
                    if np.ndim(x) == 0:
                        return quad(self.base_pdf, 0, x, limit=50)[0]
                    return np.array([quad(self.base_pdf, 0, val, limit=50)[0] for val in x])

                self.base_cdf = numerical_cdf
        elif distribution is not None:
            self.base_pdf = distribution.pdf
            self.base_cdf = distribution.cdf
        else:
            # Default to uniform
            u = uniform()
            self.base_pdf = u.pdf
            self.base_cdf = u.cdf

        if check_integration:
            try:
                total_prob, _ = quad(self.base_pdf, 0, 1, limit=100)
                if not np.isclose(total_prob, 1.0, atol=1e-3):
                    warnings.warn(
                        f"FixedStrategy PDF does not integrate to 1 (integral={total_prob:.4f}). This may invalidate the martingale.",
                        stacklevel=2,
                    )
            except Exception as e:
                warnings.warn(f"Could not verify FixedStrategy PDF integration: {e}", stacklevel=2)

    def update_betting_function(self, p_values):
        """
        Returns the fixed betting function regardless of p_values.
        """
        return self.base_pdf, self.base_cdf


class ExpertAggregationStrategy(BettingStrategy):
    """
    An aggregation strategy using an adaptive Exponentially Weighted Average forecaster.

    Examples:
        >>> # Two experts: one good (FixedStrategy betting on 0.1), one bad (Uniform)
        >>> expert_good = FixedStrategy(
        ...     pdf=lambda x: norm.pdf(x, 0.1, 0.1) / (norm.cdf(1, 0.1, 0.1) - norm.cdf(0, 0.1, 0.1)),
        ...     check_integration=False,
        ... )
        >>> expert_bad = FixedStrategy(distribution=uniform())
        >>> agg = ExpertAggregationStrategy(experts=[expert_good, expert_bad])
        >>> # After seeing p=0.1, expert_good should have higher weight
        >>> _ = agg.update_betting_function([0.1])
        >>> w = agg.get_current_weights()
        >>> w[0] > w[1]
        True
    """

    def __init__(self, experts, learning_rate=0.1, base_alpha=0.01):
        super().__init__(min_sample_size=0)
        self.experts = experts
        self.num_experts = len(experts)
        self.learning_rate = learning_rate
        self.base_alpha = base_alpha

        self.log_weights = np.zeros(self.num_experts)
        self.expert_weights_history = []

        self.expert_pdfs = [lambda x: 1.0 for _ in self.experts]
        self.expert_cdfs = [lambda x: x for _ in self.experts]

        self.update_betting_function([])

    def get_current_weights(self):
        """Calculates current linear weights from log-weights."""
        lse = logsumexp(self.log_weights)
        return np.exp(self.log_weights - lse)

    def update_betting_function(self, p_values):
        if not p_values:
            for i, expert in enumerate(self.experts):
                self.expert_pdfs[i], self.expert_cdfs[i] = expert.update_betting_function(p_values)
        else:
            previous_weights = self.get_current_weights()
            p = p_values[-1]

            gains = np.array([pdf(p) for pdf in self.expert_pdfs])

            master_prediction = np.dot(previous_weights, gains)
            loss = 1.0 - np.clip(master_prediction / self.num_experts, 0, 1)  # Optional scaling
            alpha_t = self.base_alpha * loss

            # FIX: Added epsilon or clip to avoid log(0) which causes NaN propagation
            log_gains = np.log(np.maximum(gains, 1e-12))

            self.log_weights += self.learning_rate * log_gains

            intermediate_weights = self.get_current_weights()
            final_weights = (1 - alpha_t) * intermediate_weights + alpha_t / self.num_experts
            final_weights /= np.sum(final_weights)

            # Update log_weights safely
            self.log_weights = np.log(np.maximum(final_weights, 1e-12))

            for i, expert in enumerate(self.experts):
                self.expert_pdfs[i], self.expert_cdfs[i] = expert.update_betting_function(p_values)

        final_weights_for_next_step = self.get_current_weights()
        self.expert_weights_history.append(final_weights_for_next_step.copy())

        # We must capture the CURRENT state of pdfs/cdfs.
        # Since self.expert_pdfs is a list that changes elements, we need to be careful.
        # However, we only replace the elements, so 'current_pdfs' capturing the list object is okay
        # as long as we don't modify the list *during* the execution of next_agg_pdf.
        current_pdfs = self.expert_pdfs[:]
        current_cdfs = self.expert_cdfs[:]

        def next_agg_pdf(x):
            return np.dot(final_weights_for_next_step, [pdf(x) for pdf in current_pdfs])

        def next_agg_cdf(x):
            return np.dot(final_weights_for_next_step, [cdf(x) for cdf in current_cdfs])

        return next_agg_pdf, next_agg_cdf

    def plot_weights(self, expert_names=None, title="Evolution of Expert Weights", ax=None):
        """
        Plots the evolution of expert weights over time.

        Args:
            expert_names (list of str, optional): Names for the experts for the legend.
            title (str, optional): Title of the plot.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure is created.
        """
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

        if ax is None:  # Only show if we created the figure
            plt.tight_layout()
            plt.show()


# MARTINGALES
class ConformalTestMartingale:
    """
    Parent class for conformal test martingales.

    Attributes:
        logM (float): Logarithm of the martingale value.
        max (float): Maximum martingale value observed so far.
        p_values (list): List of observed p-values.
        store_p_values (bool): Whether to store the full history of p-values.
    """

    def __init__(self, warnings=True, warning_level=100, store_p_values=True):
        self.logM = 0.0
        self.log_max = 0.0
        self.p_values = []
        self.store_p_values = store_p_values
        self.log_martingale_values = [0.0]
        self.warning_level = warning_level
        self.warnings = warnings
        self._warned = False  # Track if we have already warned
        self.b_n = lambda x: beta.pdf(x, 1, 1)
        self.B_n = lambda x: beta.cdf(x, 1, 1)

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
        # Perform check in log-space to avoid overflow
        if self.log_max >= np.log(self.warning_level) and self.warnings and not self._warned:
            warnings.warn(
                f"Exchangeability assumption likely violated: Max martingale value is {self.max}", stacklevel=2
            )
            self._warned = True


class PluginMartingale(ConformalTestMartingale):
    """
    Implements a plugin martingale using a specified betting strategy.

    Examples:
        >>> # Use a fixed strategy betting on p < 0.5
        >>> strat = FixedStrategy(pdf=lambda x: 2 if x < 0.5 else 0, check_integration=False)
        >>> martingale = PluginMartingale(betting_strategy=strat)
        >>> # A low p-value should double the wealth
        >>> martingale.update_martingale_value(0.1)
        >>> np.isclose(martingale.M, 2.0)
        True
        >>> # A high p-value should loose everything (logM becomes -inf)
        >>> martingale.update_martingale_value(0.9)
        >>> martingale.M == 0.0
        True
    """

    def __init__(self, betting_strategy=GaussianKDE, warnings=True, warning_level=100, store_p_values=True, **kwargs):
        super().__init__(warnings, warning_level, store_p_values)

        if isinstance(betting_strategy, BettingStrategy):
            self.betting_strategy = betting_strategy
        else:
            betting_kwargs = (
                kwargs
                if kwargs
                else {
                    "bandwidth": "silverman",
                    "window_size": None,
                    "min_sample_size": 100,
                }
            )
            self.betting_strategy = betting_strategy(**betting_kwargs)

        # Initialize betting functions from strategy (for the first step)
        self.b_n, self.B_n = self.betting_strategy.update_betting_function([])

    def update_martingale_value(self, p):
        self.logM += np.log(self.b_n(p))
        self.log_martingale_values.append(self.logM)

        if self.store_p_values:
            self.p_values.append(p)

        # NOTE: Betting strategies usually require history.
        # If store_p_values is False, this might break strategies that rely on it.
        # However, for PluginMartingale, it is generally expected that p_values are stored.
        # If one disables storage, they must ensure the strategy handles it (e.g. requires no history like FixedStrategy).
        self.b_n, self.B_n = self.betting_strategy.update_betting_function(self.p_values)

        if self.logM > self.log_max:
            self.log_max = self.logM

        self.check_warning()


class SimpleJumper(ConformalTestMartingale):
    """
    Implements a simple jumper martingale in log-space to avoid overflow.

    Examples:
        >>> # Initialize a jumper
        >>> sj = SimpleJumper(J=0.1)
        >>> # A sequence of small p-values should increase the wealth
        >>> for _ in range(5):
        ...     sj.update_martingale_value(0.0001)
        >>> sj.M > 1.0
        True
        >>> # A p-value of 0.5 should keep wealth roughly stable or decaying slightly if prior was wrong
        >>> sj.update_martingale_value(0.5)
    """

    def __init__(self, J=0.01, warning_level=100, warnings=True, store_p_values=True, **kwargs):
        super().__init__(warnings, warning_level, store_p_values)
        self.J = J
        # Initialize log wealths. log(1/3) approx -1.0986
        self.log_C_epsilon = {-1: -np.log(3), 0: -np.log(3), 1: -np.log(3)}
        self.log_C = 0.0  # log(1)

        self.b_epsilon = lambda u, epsilon: 1 + epsilon * (u - 1 / 2)
        self.B_n_inv = lambda x: x

    def update_martingale_value(self, p):
        if self.store_p_values:
            self.p_values.append(p)

        # Precompute mixing logs
        if self.J == 1:
            log_1_minus_J = -np.inf  # effectively 0 in linear space
        else:
            log_1_minus_J = np.log(1 - self.J)

        log_J_div_3 = np.log(self.J / 3)

        new_log_C_epsilon = {}

        for epsilon in [-1, 0, 1]:
            # Mixing step in log space: log((1-J)*C_eps + (J/3)*C)
            # = logaddexp(log(1-J) + log_C_eps, log(J/3) + log_C)
            term1 = log_1_minus_J + self.log_C_epsilon[epsilon]
            term2 = log_J_div_3 + self.log_C
            log_C_mixed = np.logaddexp(term1, term2)

            # Betting step
            bet_val = self.b_epsilon(p, epsilon)
            # bet_val is always in [0.5, 1.5] for eps in [-1, 1], so log is safe
            new_log_C_epsilon[epsilon] = log_C_mixed + np.log(bet_val)

        self.log_C_epsilon = new_log_C_epsilon

        # Calculate total wealth via logsumexp
        self.log_C = logsumexp(list(self.log_C_epsilon.values()))

        self.logM = self.log_C
        self.log_martingale_values.append(self.logM)

        # Calculate epsilon_bar for the effective betting function at the NEXT step.
        # After mixing, the pre-bet weight for expert ε is:
        #   w_ε = (1-J)*(C_ε,n / C_n) + J/3
        # The effective epsilon_bar is w_1 - w_{-1} = (1-J)*(C_1,n/C_n - C_{-1,n}/C_n).
        w_1 = np.exp(self.log_C_epsilon[1] - self.log_C)
        w_minus_1 = np.exp(self.log_C_epsilon[-1] - self.log_C)
        epsilon_bar = (1 - self.J) * (w_1 - w_minus_1)

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
    """
    Composite Jumper that averages over multiple jump rates.

    Examples:
        >>> # Standard initialization
        >>> cj = CompositeJumper()
        >>> # Update with small p sequence
        >>> for _ in range(5):
        ...     cj.update_martingale_value(0.001)
        >>> cj.M > 1
        True
    """

    # FIX: Changed mutable default argument J=[] to J=None
    def __init__(self, J=None, warnings=True, warning_level=100, store_p_values=True):
        super().__init__(warnings, warning_level, store_p_values)
        if J is None:
            self.J = [10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1]
        else:
            self.J = J

        # FIX: Initialize sub-jumpers with store_p_values=False to save memory.
        # The CompositeJumper itself will store the history if requested.
        self.Jumpers = {j: SimpleJumper(J=j, warnings=False, store_p_values=False) for j in self.J}

    def update_martingale_value(self, p):
        if self.store_p_values:
            self.p_values.append(p)

        # Update individual jumpers
        for m in self.Jumpers.values():
            m.update_martingale_value(p)

        # Log-space aggregation to prevent overflow
        # M_composite = mean(M_i)
        # logM = log(sum(exp(logM_i))) - log(N)
        log_M_values = [m.logM for m in self.Jumpers.values()]
        self.logM = logsumexp(log_M_values) - np.log(len(self.Jumpers))

        self.log_martingale_values.append(self.logM)

        # Calculate wealth-weighted betting function for consistency
        # weights w_i = M_i / \sum M_k = exp(logM_i - log(\sum M_k))
        # log(\sum M_k) = self.logM + log(N)
        log_sum_M = self.logM + np.log(len(self.Jumpers))
        weights = np.exp(np.array(log_M_values) - log_sum_M)

        # We need to capture the current b_n functions of the jumpers
        current_b_ns = [m.b_n for m in self.Jumpers.values()]
        current_B_ns = [m.B_n for m in self.Jumpers.values()]

        # The composite b_n is the weighted average of individual b_ns
        self.b_n = lambda u: np.dot(weights, [f(u) for f in current_b_ns])
        self.B_n = lambda u: np.dot(weights, [F(u) for F in current_B_ns])

        if self.logM > self.log_max:
            self.log_max = self.logM

        self.check_warning()


class OnionMartingale(ConformalTestMartingale):
    """
    Multi-layer protection martingale (the "onion").

    Composes a list of ConformalTestMartingale instances into a layered
    protection scheme. Each layer bets against the uniformity of the
    previous layer's protected p-values. The combined martingale value
    is the product of all layer martingales (Proposition 5.1).

    Each layer is a full ConformalTestMartingale and maintains its own
    state, so individual trajectories and p-values can be inspected via
    ``self.layers[i]``.

    Examples:
        >>> from copy import deepcopy
        >>> base = SimpleJumper(warnings=False)
        >>> onion = OnionMartingale([deepcopy(base) for _ in range(3)])
        >>> for _ in range(10):
        ...     onion.update_martingale_value(0.01)
        >>> onion.M > 1.0
        True
        >>> # Each layer's p-values are accessible
        >>> len(onion.layers[0].p_values) == 10
        True
        >>> # Composite B_n maps [0,1] to [0,1]
        >>> abs(onion.B_n(0.0)) < 1e-12
        True
        >>> abs(onion.B_n(1.0) - 1.0) < 1e-12
        True
    """

    def __init__(self, layers, warnings=True, warning_level=100, store_p_values=True):
        super().__init__(warnings, warning_level, store_p_values)
        if not layers:
            raise ValueError("At least one layer is required.")
        self.layers = layers

    def update_martingale_value(self, p):
        p_i = p
        for layer in self.layers:
            p_i_next = layer.B_n(p_i)  # protect BEFORE update (uses B_j)
            layer.update_martingale_value(p_i)  # updates martingale, then sets B_n = B_{j+1}
            p_i = p_i_next

        if self.store_p_values:
            self.p_values.append(p_i)  # final protected p-value

        self.logM = sum(layer.logM for layer in self.layers)
        self.log_martingale_values.append(self.logM)

        # Composite betting function: b_total(p) = prod_i b^(i)(B^(i-1)(...(p)...))
        current_b_ns = [layer.b_n for layer in self.layers]
        current_B_ns = [layer.B_n for layer in self.layers]

        def _composite_b(u):
            val = 1.0
            x = u
            for _i, (b, B) in enumerate(zip(current_b_ns, current_B_ns)):
                val *= b(x)
                x = B(x)
            return val

        def _composite_B(u):
            x = u
            for B in current_B_ns:
                x = B(x)
            return x

        self.b_n = _composite_b
        self.B_n = _composite_B

        if self.logM > self.log_max:
            self.log_max = self.logM

        self.check_warning()


class SimpleMixtureMartingale(ConformalTestMartingale):
    """
    Implements the Simple Mixture Martingale using the efficient closed-form
    solution based on the incomplete gamma function.

    Examples:
        >>> sm = SimpleMixtureMartingale()
        >>> # A series of small p-values should grow the martingale
        >>> sm.update_martingale_value(0.01)
        >>> sm.update_martingale_value(0.01)
        >>> sm.M > 1
        True
        >>> # A p-value of 1.0 is bad for the alternative, should decrease wealth
        >>> sm.update_martingale_value(1.0)
        >>> sm.M < sm.martingale_values[-2]
        True
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 0
        self.sum_log_p = 0.0

    def update_martingale_value(self, p):
        # 1. Update the sufficient statistics
        self.n += 1
        p_clipped = np.clip(p, 1e-12, 1.0)  # Clip away from zero for log
        self.sum_log_p += np.log(p_clipped)

        # 2. Calculate the new martingale value using the formula
        n = self.n
        L = self.sum_log_p

        # Handle the edge case where all p-values were 1.0, so L=0
        if np.isclose(L, 0):
            # new_M = 1 / (n + 1) -> logM = -log(n+1)
            self.logM = -np.log(n + 1)
        else:
            # The argument for the gamma functions is -L
            arg = -L

            # Use log-gamma for numerical stability
            log_gamma_n_plus_1 = gammaln(n + 1)

            # γ(n+1, -L) = Γ(n+1) * P(n+1, -L)
            # where P is the regularized incomplete gamma function (gammainc)
            # FIX: Check for underflow in gammainc
            val_gammainc = gammainc(n + 1, arg)
            if val_gammainc <= 0:
                # If we underflow, we are deep in the tail.
                # This happens if arg (sum log p) is massive, meaning p-values are tiny.
                # In this region, martingale value is huge.
                # Fallback or strict lower bound might be needed, but usually 1e-300 is fine.
                log_incomplete_gamma = -700  # Approx log(tiny)
            else:
                log_incomplete_gamma = log_gamma_n_plus_1 + np.log(val_gammainc)

            # The full formula in log-space to avoid large numbers
            # log_M = -L + log_incomplete_gamma - (n + 1) * np.log(arg)
            self.logM = -L + log_incomplete_gamma - (n + 1) * np.log(arg)

        if self.store_p_values:
            self.p_values.append(p)
        self.log_martingale_values.append(self.logM)

        if self.logM > self.log_max:
            self.log_max = self.logM
        self.check_warning()


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
