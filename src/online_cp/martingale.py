import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta, norm
import warnings


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

    Attributes:
        bandwidth (str or float): Bandwidth selection method or fixed value.
        window_size (str or int): Window size for adaptive KDE.
    """

    def __init__(self, bandwidth='silverman', window_size=None, min_sample_size=100, mixing_exponent=1):
        """
        Initializes the GaussianKDE class.

        Args:
            bandwidth (str or float): Bandwidth selection method or fixed value.
            window_size (str or int): Window size for adaptive KDE.
            min_sample_size (int): Minimum number of samples required before full betting.
            mixing_exponent (float): Exponent controlling the mixing parameter's growth.
        """
        super().__init__(min_sample_size, mixing_exponent)
        self.bandwidth = bandwidth
        self.window_size = window_size

    def calculate_bandwidth(self, data, sigma=None):
        """
        Calculates the bandwidth for KDE.

        Args:
            data (np.ndarray): Data points.
            sigma (float): Standard deviation of the data.

        Returns:
            float: Bandwidth value.
        """
        if self.bandwidth == 'silverman':
            assert sigma is not None
            h = ((4 * sigma**5) / (3 * data.size))**(1/5)
        else:
            h = self.bandwidth
        return h
    
    def calculate_window_size(self, p_values):
        """
        Calculates the window size for adaptive KDE.

        Args:
            p_values (list): List of p-values.

        Returns:
            int: Window size.
        """
        if self.window_size == 'adaptive':
            window_param = np.log(1.001)
            min_size = self.min_sample_size
            max_size = len(p_values)
            return max(min_size, int(max_size * np.exp(-window_param * self.M)))
        elif self.window_size is None:
            return 0
        else:
            return self.window_size

    def update_betting_function(self, p_values):
        """
        Updates the betting function based on the provided p-values.

        Args:
            p_values (list): List of p-values.

        Returns:
            tuple: Updated betting functions (b_n, B_n).
        """
        if len(p_values) < 2:  # Too little information to bet
            pdf = lambda x: beta.pdf(x, 1, 1)
            cdf = lambda x: beta.cdf(x, 1, 1)
        else:
            data = np.array(p_values)[-self.calculate_window_size(p_values):]
            # sigma = np.array([data, 2 - data, -data]).flatten().std()
            sigma = np.array(data).std()
            if sigma == 0:
                # If there is no variability: do not bet at all.
                pdf = lambda x: beta.pdf(x, 1, 1)
                cdf = lambda x: beta.cdf(x, 1, 1)
            else:
                h = self.calculate_bandwidth(data=data, sigma=sigma)

                def kernel_pdf_raw(x):
                    """
                    Computes the raw kernel PDF.

                    Args:
                        x (np.ndarray): Input values.

                    Returns:
                        np.ndarray or float: PDF values.
                    """
                    x = np.atleast_1d(x)
                    pdf_values = np.mean(norm.pdf(x[:, None], loc=data, scale=h * sigma), axis=1)
                    if pdf_values.size == 1:
                        return pdf_values.item()
                    return pdf_values

                def kernel_cdf_raw(x):
                    """
                    Computes the raw kernel CDF.

                    Args:
                        x (np.ndarray): Input values.

                    Returns:
                        np.ndarray or float: CDF values.
                    """
                    x = np.atleast_1d(x)
                    cdf_values = np.mean(norm.cdf(x[:, None], loc=data, scale=h * sigma), axis=1)
                    if cdf_values.size == 1:
                        return cdf_values.item()
                    return cdf_values

                def kernel_pdf_reflect(x):
                    """
                    Computes the reflected kernel PDF.

                    Args:
                        x (np.ndarray): Input values.

                    Returns:
                        np.ndarray or float: Reflected PDF values.
                    """
                    x = np.atleast_1d(x)
                    pdf_reflect_values = kernel_pdf_raw(x) + kernel_pdf_raw(-x) + kernel_pdf_raw(2 - x)
                    if np.isscalar(pdf_reflect_values):
                        return pdf_reflect_values
                    return pdf_reflect_values

                def kernel_cdf_reflect(x):
                    """
                    Computes the reflected kernel CDF.

                    Args:
                        x (np.ndarray): Input values.

                    Returns:
                        np.ndarray or float: Reflected CDF values.
                    """
                    x = np.atleast_1d(x)
                    cdf_reflect_values = kernel_cdf_raw(x) - kernel_cdf_raw(-x) + 1 - kernel_cdf_raw(2 - x)
                    if np.isscalar(cdf_reflect_values):
                        return cdf_reflect_values
                    return cdf_reflect_values
                
                pdf = kernel_pdf_reflect
                cdf = kernel_cdf_reflect
        
        # Cautious betting until min_sample_size is reached
        b_n = lambda x: self.mixing_parameter * pdf(x) + (1 - self.mixing_parameter)
        B_n = lambda x: self.mixing_parameter * cdf(x) + (1 - self.mixing_parameter) * x
        
        self.update_mixing_parameter(len(p_values))

        return b_n, B_n


class BetaMoments(BettingStrategy):
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

    def update_betting_function(self, p):
        """
        Updates the betting function based on the provided p-value.

        Args:
            p (float): A single p-value.

        Returns:
            tuple: Updated betting functions (b_n, B_n).
        """
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

    def update_betting_function(self, p):
        """
        Updates the betting function based on the provided p-value.

        Args:
            p (float): A single p-value.

        Returns:
            tuple: Updated betting functions (b_n, B_n).
        """
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
        return np.log10(self.martingale_values)
    
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
        
        if isinstance(self.betting_strategy, GaussianKDE):
            self.b_n, self.B_n = self.betting_strategy.update_betting_function(self.p_values)
        else:
            self.b_n, self.B_n = self.betting_strategy.update_betting_function(p)
        
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

        
if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
