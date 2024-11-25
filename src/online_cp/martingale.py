import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta
import warnings


class PluginMartingale:
    '''
    A conformal test martingale to test exchangeability online.
    We reject exchangeability with confidence 1-1/alpha if we ever observe M >= alpha for any alpha>0

    >>> martingale = PluginMartingale()
    >>> print(martingale.M)
    1.0
    '''

    def __init__(self, betting_function='kernel', warning_level=100, warnings=True, **kwargs):
        '''
        For numerical reasons, it is better to update the martingale in log scale
        Warning level set to 100 means that we warn if the exchangeability hypothesis can 
        be discarded with confidence at least 0.99.
        If you do not want to be warned, set to np.inf
        - betting_function (str): The method to use ("kernal", "beta_bayes", "beta_mle", "beta_moments").
        - kwargs: Additional parameters for the selected method.
            - For "beta_bayes": prior_alpha, prior_beta (default 5 each).
            - For "beta_mle" and "beta_moments": min_sample_size (default 30 for "beta_mle" and 20 for "beta_moments)
        '''
        # TODO: Make it possible to pass parameters to the kernel method as well.
        self.logM = 0.0
        self.max = 1 # The maximum value ever reachec by the maringale

        self.params = kwargs

        # At the moment, we just have one betting function, but more can be added.
        self.betting_function = betting_function
        if self.betting_function == 'beta_moments':
            min_sample_size = self.params.get("min_sample_size", 20)
            self.beta_stats = OnlineBetaMOM(min_sample_size)
        if self.betting_function == 'beta_mle':
            min_sample_size = self.params.get("min_sample_size", 30)
            self.beta_stats = OnlineBetaMLE(min_sample_size)
        if self.betting_function == 'beta_bayes':
            self.alpha = self.params.get("prior_alpha", 5)
            self.beta = self.params.get("prior_beta", 5)
            self.beta_stats = BayesianBetaEstimator(self.alpha, self.beta)

        self.warning_level = warning_level
        self.warnings = warnings

        self.p_values = []
        self.martingale_values = []

        # FIXME: This is a bit ad hoc... 
        self.current_betting_function = lambda p: 1

    def kernel_density_betting_function(self):
        '''
        Betting function from Vovk paper
        TODO: Make it possible to extract the current betting funciton. It can be used for protected probabilistic regression (https://www.alrw.net/articles/34.pdf)
        '''
        def get_density_estimate(p_values):
            P = np.array([[-p, p, 2-p] for p in p_values]).flatten()[:, np.newaxis]
            if len(P) == 0:
                return None, None
            kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(P)
            func = lambda p :np.exp(kde.score_samples([[p]])[0])
            norm_fac = quad(func, 0, 1)
            return kde, norm_fac
        
        def betting_function(p, d, norm_fac):
            if not d: 
                return 1
            if 0 <= p <= 1:
                return np.exp(d.score_samples([[p]])[0]) / norm_fac[0]
            else:
                return 0.0
            
        d, norm_fac = get_density_estimate(self.p_values[:-1])

        # FIXME: This is a bit ad hoc . Something nicer would be good.
        self.current_betting_function = lambda p: betting_function(p, d, norm_fac)

        return betting_function(self.p_values[-1], d, norm_fac)
    
    def beta_betting_function(self):
        '''
        Method of moments to fit a beta distribution to the p-values
        Seems to be very slow... Update sample mean and variance online instead.
        '''
        if len(self.p_values) > 1:
            self.beta_stats.update(self.p_values[-2])

        self.ahat, self.bhat = self.beta_stats.get_params()

        self.current_betting_function = lambda p: beta.pdf(p, self.ahat, self.bhat)
        return beta.pdf(self.p_values[-1], self.ahat, self.bhat)
    
    def update_log_martingale(self, p):
        self.p_values.append(p)
        if self.betting_function == 'kernel':
            self.logM += np.log(self.kernel_density_betting_function())
        elif 'beta' in self.betting_function:
            self.logM += np.log(self.beta_betting_function())
        else:
            raise NotImplementedError('Currently only kernel betting function is available. More to come...')
        # Update the running max
        if self.M > self.max:
            self.max = self.M
        
        self.martingale_values.append(self.M)

        if self.max >= self.warning_level and self.warnings:
            warnings.warn(f'Exchangeability assumption likely violated: Max martingale value is {self.max}')

    @property
    def M(self):
        return np.exp(self.logM)
    
    @property
    def log_martingale_values(self):
        return np.log(self.martingale_values)
    
# FIXME: The problem with both methods, is that they struggle with small sample sizes, leading possibly to very
#        poor bets early on. If we could make them tend towards uniformity rather that some other extreme, at least
#        we would not loos a lot of capital in the beginning. One simple idea, is to keep them fixed at uniform for
#        the first M iterations...
# TODO:  The solution to this seems to be a Bayesian approach to beta approximation. We bias towards uniformity
#        by setting the uniform distribution as our prior. We could consider keeping the others, but adjusting the
#        estimates for small sample sizes so that they are uniform (refrain from betting) in the beginning.
# TODO:  I don't like having the beta estimators in the same file as the martingales. They should be kept separately,
#        but we have to solve the imports.
class OnlineBetaMLE:
    def __init__(self, min_sample_size=30):
        self.n = 0
        self.log_sum_x = 0.0
        self.log_sum_1_minus_x = 0.0
        self.min_sample_size = min_sample_size
    
    def update(self, x):
        """
        Update sufficient statistics with a new data point x.
        """
        if x <= 0 or x >= 1:
            raise ValueError("Data points must be in the range (0, 1).")
        self.n += 1
        self.log_sum_x += np.log(x)
        self.log_sum_1_minus_x += np.log(1 - x)
    
    def mle(self):
        """
        Compute the MLE for alpha and beta based on the current sufficient statistics.
        """
        def negative_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf  # Invalid parameters
            log_likelihood = (
                (alpha - 1) * self.log_sum_x +
                (beta - 1) * self.log_sum_1_minus_x -
                self.n * betaln(alpha, beta)
            )
            return -log_likelihood  # Negate for minimization
        
        if self.n < self.min_sample_size: 
                return 1.0, 1.0 # If we can not say anything, choose uniform as default
        # Initial guesses for alpha and beta
        initial_guess = [1.0, 1.0]
        result = minimize(negative_log_likelihood, initial_guess,
                          bounds=[(1e-5, None), (1e-5, None)])
        if result.success:
            return tuple(result.x)  # Optimal alpha, beta
        else:
            return 1.0, 1.0 # If we can not say anything, choose uniform as default
        
    def get_params(self):
        return self.mle()
        
class OnlineBetaMOM:
    def __init__(self, min_sample_size=20):
        self.n = 0  # Number of observations
        self.mean = 0.0  # Running mean
        self.M2 = 0.0  # Sum of squared differences from the mean (for variance)
        self.min_sample_size = min_sample_size
    
    def update(self, x):
        """
        Update the running mean and variance with a new data point x.
        """
        if x <= 0 or x >= 1:
            raise ValueError("Data points must be in the range (0, 1).")
        
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2  # Incremental update for variance

    def get_mean_variance(self):
        """
        Compute the current mean and sample variance.
        """
        if self.n < 2:
            return self.mean, float('nan')  # Variance undefined for n < 2
        sample_variance = self.M2 / (self.n - 1)
        return self.mean, sample_variance

    def estimate_params(self):
        """
        Estimate alpha and beta using the method of moments.
        """
        mean, variance = self.get_mean_variance()
        if self.n < self.min_sample_size or variance <= 0:
            return 1.0, 1.0  # Parameters undefined, choose uniform as default
        
        common_factor = (mean * (1 - mean) / variance) - 1
        alpha = mean * common_factor
        beta = (1 - mean) * common_factor
        return alpha, beta

    def get_params(self):
        return self.estimate_params()
    

class BayesianBetaEstimator:
    def __init__(self, prior_alpha=10, prior_beta=10):
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.n = 0  # Keep track of sample size

    def update(self, x):
        if not (0 <= x <= 1):
            raise ValueError("Data points must be in the range [0, 1].")
        self.alpha += x
        self.beta += (1 - x)
        self.n += 1

    def point_estimate(self, scale=2):
        """Estimate the parameters of the data-generating distribution."""
        total = self.alpha + self.beta
        alpha_hat = (self.alpha / total) * scale
        beta_hat = (self.beta / total) * scale
        return alpha_hat, beta_hat

    def get_posterior_params(self):
        """Get raw posterior parameters."""
        return self.alpha, self.beta
    
    def get_params(self, scale=2):
        return self.point_estimate(scale=scale)

if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
