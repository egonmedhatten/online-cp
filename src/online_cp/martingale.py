# TODO
# * Remove online update of statistics in the parametric models. It is better to have the possibility to use a window
# * Remove B_n^{-1} whenever it appears. If a user wants that, they will have to invert numerically. It is really only relevant for protection.
# * Tune the COEFFICIENTS in the adaptive rule
# * COnsider adding plotting functions for the ParticleFilter and ExpertAggregation (as in the notebook)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gamma, gammaln, logsumexp
from scipy.stats import beta, norm, uniform
import warnings

# TODO:
# Add a betting strategy based on the beta kernel density estimation package beta-kde


class BettingStrategy:
    """
    Base class for betting strategies.

    Attributes:
        min_sample_size (int): Minimum number of samples required before full betting.
        mixing_exponent (float): Exponent controlling the mixing parameter's growth.
        mixing_parameter (float): Mixing parameter for cautious betting.
    """

    def __init__(self, min_sample_size=100, mixing_exponent=1):
        """
        Initializes the BettingStrategy class.

        Args:
            min_sample_size (int): Minimum number of samples required before full betting.
            mixing_exponent (float): Exponent controlling the mixing parameter's growth.
        """
        self.min_sample_size = min_sample_size
        self.mixing_exponent = mixing_exponent
        self.mixing_parameter = 0
    
    def update_mixing_parameter(self, n):
        """
        Updates the mixing parameter based on the number of observations.

        Args:
            n (int): Number of observations.
        """
        self.mixing_parameter = min((n / self.min_sample_size)**self.mixing_exponent, 1)


class GaussianKDE(BettingStrategy):
    """
    Implements a Gaussian Kernel Density Estimation (KDE) betting strategy.
    This version includes a 'growth_factor' to control the frequency of
    computationally expensive bandwidth re-tuning.

    Attributes:
        bandwidth (str or float): Bandwidth selection method ('silverman', 'lcv', or a float).
        window_size (str or int): Window size for adaptive KDE.
        growth_factor (float): The sample size must increase by this factor to trigger a
                               new bandwidth calculation. A factor of 1.0 means tuning every time.
    """

    def __init__(self, bandwidth='silverman', window_size=None, min_sample_size=100, mixing_exponent=1, max_iter=20, bw_min=0.001, bw_max=0.5, growth_factor=1.1):
        super().__init__(min_sample_size, mixing_exponent)
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.max_iter = max_iter
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.growth_factor = growth_factor

        self.current_bw = None
        self.n_last_update = 0  # To track sample size at last bandwidth update

    def _kernel_pdf_reflect(self, x, data, h):
        """Helper to compute the reflected kernel PDF for a given dataset and bandwidth."""
        x = np.atleast_1d(x)
        pdf_orig = np.mean(norm.pdf(x[:, None], loc=data, scale=h), axis=1)
        pdf_neg = np.mean(norm.pdf(x[:, None], loc=-data, scale=h), axis=1)
        pdf_two = np.mean(norm.pdf(x[:, None], loc=2-data, scale=h), axis=1)
        pdf_values = pdf_orig + pdf_neg + pdf_two
        return pdf_values.item() if pdf_values.size == 1 else pdf_values

    def _kernel_cdf_reflect(self, x, data, h):
        """Helper to compute the reflected kernel CDF for a given dataset and bandwidth."""
        x = np.atleast_1d(x)
        cdf_orig = np.mean(norm.cdf(x[:, None], loc=data, scale=h), axis=1)
        cdf_neg = np.mean(norm.cdf(x[:, None], loc=-data, scale=h), axis=1)
        cdf_two = np.mean(norm.cdf(x[:, None], loc=2-data, scale=h), axis=1)
        cdf_values = cdf_orig + cdf_neg + cdf_two - 1
        return cdf_values.item() if cdf_values.size == 1 else cdf_values
    
    def _likelihood_lcv_bw(self, data):
        """
        Finds the optimal bandwidth by optimizing the Likelihood Cross-Validation (LCV) score.
        This version pre-calculates distance matrices for efficiency.
        """
        n = len(data)
        data_col = data[:, np.newaxis]
        diff_orig = data_col - data
        diff_neg = data_col + data
        diff_two = data_col - (2 - data)

        def objective_func(h):
            """The negative LCV log-likelihood score function, to be minimized."""
            epsilon = 1e-10
            kernel_matrix_orig = norm.pdf(diff_orig / h)
            kernel_matrix_neg = norm.pdf(diff_neg / h)
            kernel_matrix_two = norm.pdf(diff_two / h)
            total_kernel_matrix = kernel_matrix_orig + kernel_matrix_neg + kernel_matrix_two
            np.fill_diagonal(total_kernel_matrix, 0)
            loo_pdfs = np.sum(total_kernel_matrix, axis=1) / ((n - 1) * h)
            loo_log_likelihood = np.sum(np.log(loo_pdfs + epsilon))
            return -loo_log_likelihood

        if n < self.min_sample_size or self.current_bw is None:
            result = minimize_scalar(objective_func, bounds=(self.bw_min, self.bw_max), method='bounded', options={'maxiter': self.max_iter})
        else:
            search_bounds = (max(self.current_bw * 0.8, self.bw_min), min(self.current_bw * 1.2, self.bw_max))
            result = minimize_scalar(objective_func, bounds=search_bounds, method='bounded', options={'maxiter': self.max_iter})
        
        return result

    def update_betting_function(self, p_values):
        """
        Updates the betting function based on the provided p-values.
        """
        if len(p_values) < 2:
            pdf = lambda x: beta.pdf(x, 1, 1)
            cdf = lambda x: beta.cdf(x, 1, 1)
        else:
            data = np.array(p_values)[-self.calculate_window_size(p_values):]
            if data.std() < 1e-6:
                pdf = lambda x: beta.pdf(x, 1, 1)
                cdf = lambda x: beta.cdf(x, 1, 1)
            else:
                h = self.calculate_bandwidth(data)
                self.current_bw = h
                pdf = lambda x: self._kernel_pdf_reflect(x, data, h)
                cdf = lambda x: self._kernel_cdf_reflect(x, data, h)
        
        b_n = lambda x: self.mixing_parameter * pdf(x) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * cdf(x) + (1 - self.mixing_parameter) * x
        self.update_mixing_parameter(len(p_values))
        
        return b_n, B_n
    
    def calculate_bandwidth(self, data):
        """
        Calculates the bandwidth, using the growth_factor to decide when to run the expensive optimization.
        """
        n = data.size
        
        if self.bandwidth == 'silverman':
            sigma = np.std(data)
            h = 0.9 * sigma * (n**(-0.2))
        
        elif self.bandwidth == 'lcv':
            # Determine if a recalculation is needed based on growth factor
            should_recalculate = (self.current_bw is None) or \
                                 (n < self.min_sample_size) or \
                                 (n >= self.n_last_update * self.growth_factor)
            
            if should_recalculate and n > 1:
                self.opt_result = self._likelihood_lcv_bw(data)
                h = self.opt_result.x
                self.n_last_update = n
            else:
                h = self.current_bw
        
        else: # Fixed bandwidth
            h = float(self.bandwidth)
        return h

    def calculate_window_size(self, p_values):
        if self.window_size is None:
            return len(p_values)
        else:
            return self.window_size

    
class BetaMoments(BettingStrategy):
    # TODO: Remove online update of statistics, and add possibility of sliding window
    """
    Implements a betting strategy based on Beta distribution moments.

    Attributes:
        n (int): Number of observations.
        mean (float): Running mean of observations.
        M2 (float): Sum of squared differences from the mean (for variance).
        ahat (float): Alpha parameter of the Beta distribution.
        bhat (float): Beta parameter of the Beta distribution.
    """

    def __init__(self, min_sample_size=100, mixing_exponent=1):
        """
        Initializes the BetaMoments class.

        Args:
            min_sample_size (int): Minimum number of samples required before full betting.
            mixing_exponent (float): Exponent controlling the mixing parameter's growth.
        """
        super().__init__(min_sample_size, mixing_exponent)
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.ahat = 1.0
        self.bhat = 1.0

    def update_betting_function(self, p_values):
        """
        Updates the betting function based on the provided p-value.

        Args:
            p (float): A single p-value.

        Returns:
            tuple: Updated betting functions (b_n, B_n).
        """
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
                common_factor = (self.mean * (1 - self.mean) / sample_variance) - 1
                self.ahat = self.mean * common_factor
                self.bhat = (1 - self.mean) * common_factor

        self.n += 1
        delta = p - self.mean
        self.mean += delta / self.n
        delta2 = p - self.mean
        self.M2 += delta * delta2

        b_n = lambda x: self.mixing_parameter * beta.pdf(x, self.ahat, self.bhat) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * beta.cdf(x, self.ahat, self.bhat) + (1 - self.mixing_parameter) * x
        
        self.update_mixing_parameter(self.n)

        return b_n, B_n

class BetaMLE(BettingStrategy):
    # TODO: Remove online update of statistics, and add possibility of sliding window
    """
    Implements a betting strategy based on Maximum Likelihood Estimation (MLE) for Beta distribution parameters.

    Attributes:
        n (int): Number of observations.
        log_sum_x (float): Sum of logarithms of observations.
        log_sum_1_minus_x (float): Sum of logarithms of (1 - observations).
        ahat (float): Alpha parameter of the Beta distribution.
        bhat (float): Beta parameter of the Beta distribution.
    """

    def __init__(self, min_sample_size=100, mixing_exponent=1):
        """
        Initializes the BetaMLE class.

        Args:
            min_sample_size (int): Minimum number of samples required before full betting.
            mixing_exponent (float): Exponent controlling the mixing parameter's growth.
        """
        super().__init__(min_sample_size, mixing_exponent)
        self.n = 0
        self.log_sum_x = 0.0
        self.log_sum_1_minus_x = 0.0
        self.ahat = 1.0
        self.bhat = 1.0

    def update_betting_function(self, p_values):
        """
        Updates the betting function based on the provided p-value.

        Args:
            p (float): A single p-value.

        Returns:
            tuple: Updated betting functions (b_n, B_n).
        """
        p = p_values[-1]
        def negative_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            log_likelihood = (
                (alpha - 1) * self.log_sum_x +
                (beta - 1) * self.log_sum_1_minus_x -
                self.n * betaln(alpha, beta)
            )
            return -log_likelihood

        if self.n < 2:
            self.ahat = 1.0
            self.bhat = 1.0
        else:
            initial_guess = [1.0, 1.0]
            result = minimize(negative_log_likelihood, initial_guess, bounds=[(1e-5, None), (1e-5, None)])
            if result.success:
                self.ahat, self.bhat = result.x
            else:
                self.ahat, self.bhat = 1.0, 1.0

        self.n += 1
        self.log_sum_x += np.log(p)
        self.log_sum_1_minus_x += np.log(1 - p)

        b_n = lambda x: self.mixing_parameter * beta.pdf(x, self.ahat, self.bhat) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * beta.cdf(x, self.ahat, self.bhat) + (1 - self.mixing_parameter) * x
        
        self.update_mixing_parameter(self.n)

        return b_n, B_n
    
class ParticleFilterStrategy(BettingStrategy):
    """
    A betting strategy based on a particle filter, modified to be fully
    reproducible by accepting a seed for its random number generator.
    """
    # MODIFIED: Added a 'seed' argument
    def __init__(self, num_particles=1000, process_noise_std=0.05, vol_noise_std=0.01, 
                 min_sample_size=100, mixing_exponent=1.0, seed=None):
        
        super().__init__(min_sample_size, mixing_exponent)
        self.N = num_particles
        self.adaptive_noise = (process_noise_std == 'auto')
        
        # MODIFIED: Create a private random number generator from the seed
        self.rng = np.random.default_rng(seed)

        if self.adaptive_noise:
            self.dim_x = 3
            self.vol_noise_std = vol_noise_std
            # MODIFIED: Use the private rng object
            self.particles = self.rng.standard_normal((self.N, 3)) * 0.1
            self.particles[:, 2] = np.log(0.05) + self.rng.standard_normal(self.N) * 0.1
        else:
            self.dim_x = 2
            self.Q = np.eye(2) * (process_noise_std ** 2)
            # MODIFIED: Use the private rng object
            self.particles = self.rng.standard_normal((self.N, 2)) * 0.1

        self.weights = np.ones(self.N) / self.N
        self.sigma_history = []
        self.alpha_history = []
        self.beta_history = []


    def _predict(self):
        """Propagates particles according to the motion model."""
        if self.adaptive_noise:
            # MODIFIED: Use the private rng object
            vol_noise = self.rng.standard_normal(self.N) * self.vol_noise_std
            self.particles[:, 2] += vol_noise
            
            current_sigmas = np.exp(self.particles[:, 2])
            # MODIFIED: Use the private rng object
            main_noise = self.rng.standard_normal((self.N, 2)) * current_sigmas[:, np.newaxis]
            self.particles[:, :2] += main_noise
            self.sigma_history.append(np.median(current_sigmas))
        else:
            # MODIFIED: Use the private rng object
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
        # MODIFIED: Pass the private rng object to the resampling function
        indices = self.systematic_resample(self.weights, self.rng)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.N)

    # NOTE: This was lifted from filterpy, with the addition of passing a range to the function
    @staticmethod
    def systematic_resample(weights, rng):
        """ Performs the systemic resampling algorithm used by particle filters.

        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.

        Parameters
        ----------
        weights : list-like of float
            list of weights as floats

        Returns
        -------

        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (rng.random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
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
        
        def learned_pdf(x): return np.mean(beta.pdf(x, a=alphas, b=betas))
        def learned_cdf(x): return np.mean(beta.cdf(x, a=alphas, b=betas))
        
        self.update_mixing_parameter(len(p_values))

        def b_n(x): return self.mixing_parameter * learned_pdf(x) + (1 - self.mixing_parameter) * 1.0
        def B_n(x): return self.mixing_parameter * learned_cdf(x) + (1 - self.mixing_parameter) * x

        return b_n, B_n

class FixedStrategy(BettingStrategy):
    """
    A strategy that uses a fixed, unchanging betting function from a
    provided distribution object.
    """
    def __init__(self, distribution=None, min_sample_size=20, max_mixing_parameter=1.0):
        """
        Initializes the FixedStrategy.
        
        Args:
            distribution: An object with .pdf() and .cdf() methods (e.g., from scipy.stats).
                          Defaults to a uniform distribution if None.
            max_mixing_parameter (float): The maximum confidence in the base_pdf.
                                          Set < 1.0 for permanent cautiousness.
        """
        super().__init__(min_sample_size=min_sample_size)
        if distribution is None:
            self.distribution = uniform()
        else:
            self.distribution = distribution
            
        self.max_mixing_parameter = max_mixing_parameter
        self.base_pdf = self.distribution.pdf
        self.base_cdf = self.distribution.cdf

    def update_betting_function(self, p_values):
        # This strategy's base distribution is fixed, but we apply cautious mixing.
        self.update_mixing_parameter(len(p_values))
        
        # Cap the mixing parameter to enforce permanent caution.
        current_mix = min(self.mixing_parameter, self.max_mixing_parameter)
        
        def b_n(x):
            return current_mix * self.base_pdf(x) + (1 - current_mix) * 1.0
        def B_n(x):
            return current_mix * self.base_cdf(x) + (1 - current_mix) * x
            
        return b_n, B_n
    
class ExpertAggregationStrategy(BettingStrategy):
    """
    An aggregation strategy using an adaptive Exponentially Weighted Average forecaster.
    
    This version implements the Variable Share algorithm, uses log-domain updates
    for stability, and logs the weight history for analysis.
    """
    def __init__(self, experts, learning_rate=0.1, base_alpha=0.01):
        super().__init__(min_sample_size=0)
        self.experts = experts
        self.num_experts = len(experts)
        self.learning_rate = learning_rate
        self.base_alpha = base_alpha
        
        self.log_weights = np.zeros(self.num_experts)
        
        # RESTORED: Initialize the list to store weight history
        self.expert_weights_history = []
        
        self.expert_pdfs = [lambda x: 1.0 for _ in self.experts]
        self.expert_cdfs = [lambda x: x for _ in self.experts]
        
        # Prime the experts and log the initial uniform weights
        self.update_betting_function([])

    def get_current_weights(self):
        """Calculates current linear weights from log-weights."""
        lse = logsumexp(self.log_weights)
        return np.exp(self.log_weights - lse)

    def update_betting_function(self, p_values):
        if not p_values:
            # Initialization case: just prime the experts
            for i, expert in enumerate(self.experts):
                self.expert_pdfs[i], self.expert_cdfs[i] = expert.update_betting_function(p_values)
        else:
            # Main update logic for t > 0
            previous_weights = self.get_current_weights()
            p = p_values[-1]
            
            gains = np.array([pdf(p) for pdf in self.expert_pdfs])
            
            master_prediction = np.dot(previous_weights, gains)
            loss = 1.0 - np.clip(master_prediction / self.num_experts, 0, 1)
            alpha_t = self.base_alpha * loss

            log_gains = np.log(gains)# + 1e-12)
            self.log_weights += self.learning_rate * log_gains
            
            intermediate_weights = self.get_current_weights()
            final_weights = (1 - alpha_t) * intermediate_weights + alpha_t / self.num_experts
            final_weights /= np.sum(final_weights)
            
            self.log_weights = np.log(final_weights)# + 1e-12)
            
            for i, expert in enumerate(self.experts):
                self.expert_pdfs[i], self.expert_cdfs[i] = expert.update_betting_function(p_values)

        # Get the final weights for this step and log them
        final_weights_for_next_step = self.get_current_weights()
        
        # RESTORED: Log the final weights for this step
        self.expert_weights_history.append(final_weights_for_next_step.copy())
        
        # Create aggregate functions for the NEXT prediction
        def next_agg_pdf(x): return np.dot(final_weights_for_next_step, [pdf(x) for pdf in self.expert_pdfs])
        def next_agg_cdf(x): return np.dot(final_weights_for_next_step, [cdf(x) for cdf in self.expert_cdfs])

        return next_agg_pdf, next_agg_cdf
    
# MARTINGALES
class ConformalTestMartingale:
    """
    Parent class for conformal test martingales.

    Attributes:
        logM (float): Logarithm of the martingale value.
        max (float): Maximum martingale value observed so far.
        p_values (list): List of observed p-values.
        log_martingale_values (list): Logarithm of martingale values over time.
        warning_level (float): Threshold for raising warnings about exchangeability violations.
        warnings (bool): Whether to raise warnings when the threshold is exceeded.
        b_n (function): Current betting function for density.
        B_n (function): Current betting function for cumulative density.
    """

    def __init__(self, warnings=True, warning_level=100):
        """
        Initializes the ConfromalTestMartingale class.

        Args:
            warnings (bool): Whether to raise warnings when the threshold is exceeded.
            warning_level (float): Threshold for raising warnings about exchangeability violations.
        """
        self.logM = 0.0
        self.max = 1.0
        self.p_values = []
        self.log_martingale_values = [0.0]
        self.warning_level = warning_level
        self.warnings = warnings
        self.b_n = lambda x: beta.pdf(x, 1, 1)
        self.B_n = lambda x: beta.cdf(x, 1, 1)

    @property
    def M(self):
        """
        Returns the current martingale value.

        Returns:
            float: Martingale value.
        """
        return np.exp(self.logM)
    
    @property
    def martingale_values(self):
        """
        Returns the martingale values over time.

        Returns:
            list: Martingale values.
        """
        return np.exp(self.log_martingale_values)
    
    @property
    def log10_martingale_values(self):
        """
        Returns the base-10 logarithm of martingale values over time.

        Returns:
            list: Log10 martingale values.
        """
        return [logM / np.log(10) for logM in self.log_martingale_values] # np.log10(self.martingale_values)
    
    def check_warning(self):
        """
        Checks if the martingale value exceeds the warning threshold and raises a warning if necessary.
        """
        if self.max >= self.warning_level and self.warnings:
            warnings.warn(f'Exchangeability assumption likely violated: Max martingale value is {self.max}')


class PluginMartingale(ConformalTestMartingale):
    """
    Implements a plugin martingale using a specified betting strategy.

    Attributes:
        betting_strategy (BettingStrategy): The betting strategy to use.
    """

    def __init__(self, betting_strategy=GaussianKDE, warnings=True, warning_level=100, **kwargs):
        """
        Initializes the PluginMartingale class.

        Args:
            betting_strategy (BettingStrategy or type): The betting strategy to use.
            warnings (bool): Whether to raise warnings when the threshold is exceeded.
            warning_level (float): Threshold for raising warnings about exchangeability violations.
            **kwargs: Additional arguments for the betting strategy.
        """
        super().__init__(warnings, warning_level)

        if isinstance(betting_strategy, BettingStrategy):
            self.betting_strategy = betting_strategy
        else:
            betting_kwargs = kwargs if kwargs else {
                'bandwidth': 'silverman', 
                'window_size': None, 
                'min_sample_size': 100, 
                'mixing_exponent': 1.
            }
            self.betting_strategy = betting_strategy(**betting_kwargs)

    def update_martingale_value(self, p):
        """
        Updates the martingale value based on the provided p-value.

        Args:
            p (float): A single p-value.
        """
        self.logM += np.log(self.b_n(p))
        self.log_martingale_values.append(self.logM)
        self.p_values.append(p)
        
        self.b_n, self.B_n = self.betting_strategy.update_betting_function(self.p_values)
        
        if self.M > self.max:
            self.max = self.M

        self.check_warning()


class SimpleJumper(ConformalTestMartingale):
    """
    Implements a simple jumper martingale.

    Attributes:
        J (float): Jump size parameter.
        C_epsilon (dict): Dictionary of martingale values for different epsilon values.
        C (float): Combined martingale value.
        b_epsilon (function): Betting function for density.
        B_n_inv (function): Inverse of the cumulative betting function.
    """

    def __init__(self, J=0.01, warning_level=100, warnings=True, **kwargs):
        """
        Initializes the SimpleJumper class.

        Args:
            J (float): Jump size parameter.
            warning_level (float): Threshold for raising warnings about exchangeability violations.
            warnings (bool): Whether to raise warnings when the threshold is exceeded.
            **kwargs: Additional arguments.
        """
        super().__init__(warnings, warning_level)
        self.J = J
        self.C_epsilon = {-1: 1/3, 0: 1/3, 1: 1/3}
        self.C = 1
        self.b_epsilon = lambda u, epsilon: 1 + epsilon * (u - 1/2)
        self.B_n_inv = lambda x: x
    
    def update_martingale_value(self, p):
        """
        Updates the martingale value based on the provided p-value.

        Args:
            p (float): A single p-value.
        """
        self.p_values.append(p)
        for epsilon in [-1, 0, 1]:
            self.C_epsilon[epsilon] = (1 - self.J) * self.C_epsilon[epsilon] + (self.J / 3) * self.C
            self.C_epsilon[epsilon] = self.C_epsilon[epsilon] * self.b_epsilon(p, epsilon)
        self.C = self.C_epsilon[-1] + self.C_epsilon[0] + self.C_epsilon[1]
        self.logM = np.log(self.C)
        self.log_martingale_values.append(self.logM)

        epsilon_bar = (self.C_epsilon[1] - self.C_epsilon[-1]) / self.C
        self.b_n = lambda u: 1 + epsilon_bar * (u - 1/2)
        self.B_n = lambda u: (epsilon_bar / 2) * u**2 + (1 - epsilon_bar / 2) * u
        self.B_n_inv = lambda u: (epsilon_bar - 2) / (2 * epsilon_bar) + np.sqrt(epsilon_bar * (8 * u + epsilon_bar - 4) + 4) / (2 * epsilon_bar)

        if self.M > self.max:
            self.max = self.M

        self.check_warning()

class CompositeJumper(ConformalTestMartingale):

    def __init__(self, J=[10**(-4), 10**(-3), 10**(-2), 10**(-1), 1], warnings=True, warning_level=100):
        super().__init__(warnings, warning_level)
        self.J = J
        self.Jumpers = {j: SimpleJumper(J=j, warnings=False) for j in self.J}

    def update_martingale_value(self, p):
        self.p_values.append(p)
        # Update individual jumpers
        for m in self.Jumpers.values():
            m.update_martingale_value(p) # NOTE: There will be a memory bloat here, as each Jumper keeps its own list of p-values.
        
        self.logM = np.log(np.mean([m.M for m in self.Jumpers.values()]))
        self.log_martingale_values.append(self.logM)
        
        self.b_n = lambda u: np.mean([m.b_n(u) for m in self.Jumpers.values()])
        self.B_n = lambda u: np.mean([m.b_n(u) for m in self.Jumpers.values()])

        if self.M > self.max:
            self.max = self.M

        self.check_warning()
        
if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
