import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta, gaussian_kde, norm
import warnings

# TODO: Restructure this so that we have two cases regulated by a keyword (or possibly two classes)
#       1) Kernel estimate: **kwargs for kernel and bandwidth. Kernel should also be able to give the corresponding CDF, that can be used for protection.
#       2) Beta estimate: **kwargs for inference method. There are essentially three choices; moments, maximum likelihood, and Bayesian (the latter needs some research to implement properly)
#       The martingale should maintain the current PDF (betting function b_n), and the current CDF (protection B_n), and we must have B_n = db_n/dp.
#       A starting point is https://chatgpt.com/share/674574bf-d12c-8007-bc32-0fcf48e81c9f


class PluginMartingale:
    '''
    A conformal test martingale to test exchangeability online.
    We reject exchangeability with confidence 1-1/alpha if we ever observe M >= alpha for any alpha>0

    '''

    # NOTE: Initially, we could protect against poor density estimates by mixing the kernel density with a uniform distribution, 
    #       letting the mixing parameter decay with increased observations. This would be some kind of cautions betting function 
    #       that avoids betting with little information. This may be better than min_sample_size.
    #       Such a mixing parameter should go from 0 to 1 with increasing observations.
    #       I think it makes sense to have a concave function. Could be something like linear, quadratic, etc, and be
    #       regulated by passing an exponent.
    
    def __init__(self, method='kernel', warning_level=100, warnings=True, **kwargs):
        '''
        We can either use a kernel density estimate, or a parametric Beta model to estimate the density
        - method (str): The method for estimation ("kernel" of "beta")
        - kwargs: Additional parameters for the selected method.
            - For "kernel": "kernel_method" and "bandwidth"
            - For "beta": "mle" or "moments" or "bayes"

        TODO: Clear up the documentation. This could be messy.
        '''
        self.logM = 0.0
        self.max = 1.0 # Keeps track of the maximum value so far.

        self.p_values = []
        self.log_martingale_values = [0.0] # NOTE: It may be better not to store the initial value...

        self.warning_level = warning_level
        self.warnings = warnings # Do we raise a user warning or not, when the warning level is reached

        self.method = method

        self.params = kwargs

        # Initially, uniform betting function.
        self.b_n = lambda x: beta.pdf(x, 1, 1)
        self.B_n = lambda x: beta.cdf(x, 1, 1)

        if self.method == 'kernel':
            self.kernel = self.params.get("kernel", 'gaussian' )
            self.min_sample_size = self.params.get("min_sample_size", 5)
            if self.kernel == 'gaussian':
                self.kernel_method = self.params.get("kernel_method", 'reflect') # 'reflect' or 'logit'
                if self.kernel_method == 'logit':
                    self.edge_adjustment = self.params.get("edge_adjustment", 0.0)
            self.bandwidth = self.params.get("bandwidth", 'silverman') # 'silverman' or 'scott'

        elif self.method == 'beta':
            # Default is not to bet
            self.ahat = 1
            self.bhat = 1
            self.beta_method = self.params.get("beta_method", 'moment')

            if self.beta_method == 'moment':
                self.min_sample_size = self.params.get("min_sample_size", 20)
                # Initialise online statistics
                self.n = 0  # Number of observations
                self.mean = 0.0  # Running mean
                self.M2 = 0.0  # Sum of squared differences from the mean (for variance)

            elif self.beta_method == 'mle':
                self.min_sample_size = self.params.get("min_sample_size", 20)
                # Initialise online statistics
                self.n = 0  # Number of observations
                self.log_sum_x = 0.0
                self.log_sum_1_minus_x = 0.0

            elif self.beta_method == 'bayes':
                raise NotImplementedError('Working on this...')
                # FIXME: Figure out how to implement the Bayesian method. It is intuitively great to have the uniform distribution as the prior, 
                #        but I am confused as to how to implement it.
                self.prior_alpha = self.params.get("prior_alpha", 1)
                self.prior_beta = self.params.get("prior_beta", 1)
            else:
                raise Exception('beta_method must be one of "moment", "mle", bayes"')

    @property
    def M(self):
        return np.exp(self.logM)
    
    @property
    def martingale_values(self):
        return np.exp(self.log_martingale_values)
    
    @property
    def log10_martingale_values(self):
        return np.log10(self.martingale_values)

    def update_martingale_value(self, p):
        if self.method == 'beta':
            self.logM += np.log(self.beta_betting_function(p))
        elif self.method == 'kernel':
            self.logM += np.log(self.kernel_betting_function(p))

        self.log_martingale_values.append(self.logM)
        self.p_values.append(p)

        if self.M > self.max:
            self.max = self.M

        if self.max >= self.warning_level and self.warnings:
            # TODO: Figure out how to warn only once!
            warnings.warn(f'Exchangeability assumption likely violated: Max martingale value is {self.max}')


    def kernel_betting_function(self, p):
        if self.kernel == 'gaussian':
            gain, b_n, B_n = self.kernel_gaussian_betting_function(p)
            self.b_n = b_n
            self.B_n = B_n
            return gain
        else:
            raise NotImplementedError('There is currently only support for gaussian kernel')

    def kernel_gaussian_betting_function(self, p):

        if len(self.p_values) < self.min_sample_size:
            b_n = lambda x: beta.pdf(x, 1, 1)
            B_n = lambda x: beta.cdf(x, 1, 1)
            return 1, b_n, B_n
        else:
            if self.kernel_method == 'reflect':
                data = np.array(self.p_values)
                kde = gaussian_kde(data, bw_method=self.bandwidth)
                h = kde.factor # The bandwidth

                sigma = data.std()

                # Ensure `x` works for both scalars and arrays
                def kernel_pdf_raw(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    pdf_values = np.mean(norm.pdf(x[:, None], loc=data, scale=h * sigma), axis=1)
                    if pdf_values.size == 1:  # Check if the result is a single value
                        return pdf_values.item()  # Convert single-element array to scalar
                    return pdf_values

                def kernel_cdf_raw(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    cdf_values = np.mean(norm.cdf(x[:, None], loc=data, scale=h * sigma), axis=1)
                    if cdf_values.size == 1:  # Check if the result is a single value
                        return cdf_values.item()  # Convert single-element array to scalar
                    return cdf_values

                def kernel_pdf_reflect(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    pdf_reflect_values = kernel_pdf_raw(x) + kernel_pdf_raw(-x) + kernel_pdf_raw(2 - x)
                    if np.isscalar(pdf_reflect_values):  # Check if the result is scalar
                        return pdf_reflect_values
                    return pdf_reflect_values

                def kernel_cdf_reflect(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    cdf_reflect_values = kernel_cdf_raw(x) - kernel_cdf_raw(-x) + 1 - kernel_cdf_raw(2 - x)
                    if np.isscalar(cdf_reflect_values):  # Check if the result is scalar
                        return cdf_reflect_values
                    return cdf_reflect_values

                return kernel_pdf_reflect(p), kernel_pdf_reflect, kernel_cdf_reflect
            
            elif self.kernel_method == 'logit':
                def logit(x):
                    return np.log(x / (1-x))
                def edge_adjustment(x, delta=0.001):
                    return delta + (1-2*delta)*x
                def transform_with_edge_adjustment(x, delta=0.001):
                    return logit(edge_adjustment(x, delta))
                def derivative_transform_with_edge_adjustment(x, delta=0.001):
                    return (2*delta - 1) / ((2*delta*x - delta - x)*(2*delta*x - delta - x + 1))
                
                data = np.array(self.p_values)

                transformed_data = transform_with_edge_adjustment(data, self.edge_adjustment)

                kde = gaussian_kde(transformed_data, bw_method=self.bandwidth)
                h = kde.factor
                sigma = transformed_data.std()

                def kernel_pdf_transformed(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    pdf_values = np.mean(norm.pdf(x[:, None], loc=transformed_data, scale=h * sigma), axis=1)
                    if pdf_values.size == 1:  # Check if the result is a single value
                        return pdf_values.item()  # Convert single-element array to scalar
                    return pdf_values

                def kernel_cdf_transformed(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    cdf_values = np.mean(norm.cdf(x[:, None], loc=transformed_data, scale=h * sigma), axis=1)
                    if cdf_values.size == 1:  # Check if the result is a single value
                        return cdf_values.item()  # Convert single-element array to scalar
                    return cdf_values

                def kernel_pdf_logit(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    pdf_transform_values = kernel_pdf_transformed(transform_with_edge_adjustment(x, self.edge_adjustment)) * derivative_transform_with_edge_adjustment(x, self.edge_adjustment)
                    if pdf_transform_values.shape[0] == 1: 
                        return pdf_transform_values[0]
                    return pdf_transform_values

                def kernel_cdf_logit(x):
                    x = np.atleast_1d(x)  # Ensure x is an array
                    cdf_transform_values = kernel_cdf_transformed(transform_with_edge_adjustment(x, self.edge_adjustment))
                    return cdf_transform_values
                
                return kernel_pdf_logit(p), kernel_pdf_logit, kernel_cdf_logit
            
            else:
                raise NotImplementedError(f'Kernel method {self.kernel_method} does not exist.')


    def beta_betting_function(self, p):
        if self.beta_method == 'moment':
            gain = self.beta_moment_betting_function(p) # Is gain really a good name? I mean the outcome of the bet...
            self.b_n = lambda x: beta.pdf(x, self.ahat, self.bhat)
            self.B_n = lambda x: beta.cdf(x, self.ahat, self.bhat)
            return gain
            
        elif self.beta_method == 'mle':
            gain = self.beta_mle_betting_function(p) # Is gain really a good name? I mean the outcome of the bet...
            self.b_n = lambda x: beta.pdf(x, self.ahat, self.bhat)
            self.B_n = lambda x: beta.cdf(x, self.ahat, self.bhat)
            return gain
        
        elif self.beta_method == 'bayes':
            raise NotImplementedError()
        
    def beta_moment_betting_function(self, p):
        if self.n < self.min_sample_size:
            self.ahat = 1.0  # Parameters undefined, choose uniform as default
            self.bhat = 1.0
        
        else:
            sample_variance = self.M2 / (self.n - 1) if self.n > 1 else 0
            
            if sample_variance <= 0:
                self.ahat = 1  # Parameters undefined, choose uniform as default
                self.bhat = 1
            
            else:
                common_factor = (self.mean * (1 - self.mean) / sample_variance) - 1
                self.ahat = self.mean * common_factor
                self.bhat = (1 - self.mean) * common_factor

        # Update statistics
        self.n += 1
        delta = p - self.mean
        self.mean += delta / self.n
        delta2 = p - self.mean
        self.M2 += delta * delta2  # Incremental update for variance

        return beta.pdf(p, self.ahat, self.bhat)
    
    def beta_mle_betting_function(self, p):
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
            self.ahat = 1.0  # Parameters undefined, choose uniform as default
            self.bhat = 1.0
        else:
            initial_guess = [1.0, 1.0]
            result = minimize(negative_log_likelihood, initial_guess,
                            bounds=[(1e-5, None), (1e-5, None)])
            if result.success:
                self.ahat, self.bhat =  tuple(result.x)  # Optimal alpha, beta
            else:
                self.ahat, self.bhat = 1.0, 1.0 # If we can not say anything, choose uniform as default
            
        # Update statistics
        self.n += 1
        self.log_sum_x += np.log(p)
        self.log_sum_1_minus_x += np.log(1 - p)

        return beta.pdf(p, self.ahat, self.bhat)



class PluginMartingaleOld:
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
            self.ahat = 1
            self.bhat = 1

        if self.betting_function == 'beta_mle':
            min_sample_size = self.params.get("min_sample_size", 30)
            self.beta_stats = OnlineBetaMLE(min_sample_size)
            self.ahat = 1
            self.bhat = 1

        if self.betting_function == 'beta_bayes':
            raise NotImplementedError('This estimator is under construction')
            self.prior_alpha = self.params.get("prior_alpha", 5)
            self.prior_beta = self.params.get("prior_beta", 5)
            self.ahat = self.prior_alpha
            self.bhat = self.prior_beta
            self.beta_stats = BayesianBetaEstimator(self.prior_alpha, self.prior_beta)

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
# FIXME: The Bayesian estimator is unreliable. I think it can be fixed by storing all points and runing a batch estimate
#        at each step instead.

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
