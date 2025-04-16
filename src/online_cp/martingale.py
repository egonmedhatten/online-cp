import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.spatial import KDTree
from scipy.stats import beta, gaussian_kde, norm
from scipy.integrate import quad
import warnings


class ConfromalTestMartingale:
    '''
    Parent class for conformal test martingales
    '''
    def __init__(self):
        pass


class PluginMartingale(ConfromalTestMartingale):
    '''
    A conformal test martingale to test exchangeability online.
    We reject exchangeability with confidence 1-1/alpha if we ever observe M >= alpha for any alpha>0
    '''
    
    def __init__(self, method='kernel', window_size=None, warning_level=100, warnings=True, **kwargs):
        '''
        We can either use a kernel density estimate, or a parametric Beta model to estimate the density
        - method (str): The method for estimation ("kernel" of "beta")
        - kwargs: Additional parameters for the selected method.
            - For "kernel": "kernel_method" and "bandwidth"
            - For "beta": "mle" or "moments" or "bayes"

        TODO: Clear up the documentation. This could be messy.
        NOTE: It is possible to use a beta kernel for KDE. It is bounded by definition, and so, should have 
              no truoble with bounded data.
        TODO: Should we really have a parametric plugin? 
        '''
        self.logM = 0.0
        self.max = 1.0 # Keeps track of the maximum value so far.

        self.p_values = []
        self.log_martingale_values = [0.0] # NOTE: It may be better not to store the initial value...

        self.warning_level = warning_level
        self.warnings = warnings # Do we raise a user warning or not, when the warning level is reached

        self.method = method

        self.window_size = window_size

        self.params = kwargs

        # Initially, uniform betting function.
        self.b_n = lambda x: beta.pdf(x, 1, 1)
        self.B_n = lambda x: beta.cdf(x, 1, 1)

        self.min_sample_size = self.params.get("min_sample_size", 100) # After this many points, we use pure density estimate
        self.mixing_exponent = self.params.get("mixing_exponent", 1) # How fast does the mixing parameter grow to 1

        self.mixing_parameter = 0

        if self.method == 'kernel':
            self.kernel = self.params.get("kernel", 'gaussian' )
            if self.kernel == 'gaussian':
                self.kernel_method = self.params.get("kernel_method", 'reflect') # 'reflect' or 'logit'
                if self.kernel_method == 'logit':
                    self.edge_adjustment = self.params.get("edge_adjustment", 0.0)
                self.bandwidth = self.params.get("bandwidth", 'silverman') # 'silverman' or 'scott'
            elif self.kernel == 'beta':
                self.C = self.params.get("C", 1)
                self.bandwidth = self.params.get("bandwidth", 1.0)

        elif self.method == 'beta':
            # Default is not to bet
            self.ahat = 1
            self.bhat = 1
            self.beta_method = self.params.get("beta_method", 'moment')
            if self.window_size is not None:
                print('Parametric beta family can not use a roling window.') # FIXME: Perhaps a user warning instead

            if self.beta_method == 'moment':
                # Initialise online statistics
                self.n = 0  # Number of observations
                self.mean = 0.0  # Running mean
                self.M2 = 0.0  # Sum of squared differences from the mean (for variance)

            elif self.beta_method == 'mle':
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

    def update_mixing_parameter(self):
        self.mixing_parameter = min((len(self.p_values) / self.min_sample_size)**self.mixing_exponent, 1)

    @property
    def M(self):
        return np.exp(self.logM)
    
    @staticmethod
    def calculate_bandwidth_gaussian(data, bw_mehtod='silverman', sigma=None):
        if bw_mehtod == 'silverman':
            assert sigma is not None
            h = ((4 * sigma**5) / (3 * data.size))**(1/5)
        else:
            raise NotImplementedError
        return h

    # FIXME: This is not very good...
    @staticmethod
    def calculate_bandwidth_beta(data, C, sigma=None):
        assert sigma is not None
        h = C * (sigma * data.size)**(-2/5)
        return h
    
    @property
    def martingale_values(self):
        return np.exp(self.log_martingale_values)
    
    @property
    def log10_martingale_values(self):
        return np.log10(self.martingale_values)
    
    # FIXME Need some calibration...
    def calculate_window_size(self):
        if self.window_size == 'adaptive':
            window_param = np.log(1.001)
            min_size = self.min_sample_size
            max_size = len(self.p_values)
            return max(min_size, int(max_size*np.exp(-window_param*self.M)))
        elif self.window_size is None:
            return 0
        else:
            return self.window_size

    def update_martingale_value(self, p):
        if self.method == 'beta':
            self.logM += np.log(self.beta_betting_function(p))
        elif self.method == 'kernel':
            self.logM += np.log(self.kernel_betting_function(p))

        self.log_martingale_values.append(self.logM)
        self.p_values.append(p)

        self.update_mixing_parameter()

        if self.M > self.max:
            self.max = self.M

        if self.max >= self.warning_level and self.warnings:
            # TODO: Figure out how to warn only once!
            warnings.warn(f'Exchangeability assumption likely violated: Max martingale value is {self.max}')


    def kernel_betting_function(self, p):
        if self.kernel == 'gaussian':
            gain, b_n, B_n = self.kernel_gaussian_betting_function(p)
        elif self.kernel == 'beta':
            gain, b_n, B_n = self.kernel_beta_betting_function(p)
        else:
            raise NotImplementedError('There is currently only support for gaussian kernel')
        assert not np.isnan(gain)
        self.b_n = lambda x: self.mixing_parameter * b_n(x) + (1-self.mixing_parameter)
        self.B_n = lambda x: self.mixing_parameter * B_n(x) + (1-self.mixing_parameter)*x
        return self.mixing_parameter * gain + (1-self.mixing_parameter)


    def _K(self, x, b, t):
        if 2*b <= x <= 1-2*b:
            return beta.pdf(t, x/b, (1-x)/b)
        elif 0 <= x < 2*b:
            return beta.pdf(t, self._rho(x, b), (1-x)/b)
        elif 1 - 2*b < x <= 1:
            return beta.pdf(t, x/b, self._rho(1-x, b))

    @staticmethod
    def _rho(x, b):
        return 2*b**2 + 2.5 - np.sqrt(4*b**4 + 6*b**2 + 2.25 - x**2 - x/b)

    def kernel_beta_betting_function(self, p):
        if len(self.p_values) < 2:
            b_n = lambda x: beta.pdf(x, 1, 1)
            B_n = lambda x: beta.cdf(x, 1, 1)
            return 1, b_n, B_n
        else:
            data = np.array(self.p_values)[-self.calculate_window_size():]
            
            sigma = data.std()

            # FIXME: Is this reasonable? Maybe rather something very concentrated
            if sigma == 0:
                'If there is no variability: do not bet at all.'
                b_n = lambda x: beta.pdf(x, 1, 1)
                B_n = lambda x: beta.cdf(x, 1, 1)
                return 1, b_n, B_n
            
            # b = self.calculate_bandwidth_beta(data=data, C=self.C, sigma=sigma)
            # b = min(self.calculate_bandwidth_beta_dev(data=data, debug=True), 0.25)

            b = self.bandwidth

            def kernel_pdf(x):
                '''
                Vectorized implementation of the kernel PDF.
                '''
                x = np.atleast_1d(x)
                pdf_values = np.array([self._K(xi, b, data) for xi in x])
                mean_pdf_values = np.mean(pdf_values, axis=1) if pdf_values.ndim > 1 else np.mean(pdf_values)
                return mean_pdf_values.item() if mean_pdf_values.size == 1 else mean_pdf_values
            
            kernel_cdf = lambda x: np.clip(quad(kernel_pdf, 0, x)[0], a_min=0, a_max=1) # NOTE: This is slow, but an analytic solution may not even be possible

        return kernel_pdf(p), kernel_pdf, np.vectorize(kernel_cdf)

    def kernel_gaussian_betting_function(self, p):

        if len(self.p_values) < 2:
            b_n = lambda x: beta.pdf(x, 1, 1)
            B_n = lambda x: beta.cdf(x, 1, 1)
            return 1, b_n, B_n
        else:
            data = np.array(self.p_values)[-self.calculate_window_size():]
            
            if self.kernel_method == 'reflect':
                
                # NOTE: We could either use the standard deviation of the original data, or the augmented, reflected data.
                # sigma = data.std()
                sigma = np.array([data, 2-data, -data]).flatten().std()

                # FIXME: Is this reasonable? Maybe rather something very concentrated
                if sigma == 0:
                    'If there is no variability: do not bet at all.'
                    b_n = lambda x: beta.pdf(x, 1, 1)
                    B_n = lambda x: beta.cdf(x, 1, 1)
                    return 1, b_n, B_n

                # NOTE: This computes bandwidth for the original data, not including the reflected points.
                h = self.calculate_bandwidth_gaussian(data=data, bw_mehtod=self.bandwidth, sigma=sigma)

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
                
                transformed_data = transform_with_edge_adjustment(data, self.edge_adjustment)

                # kde = gaussian_kde(transformed_data, bw_method=self.bandwidth)
                # h = kde.factor
                sigma = transformed_data.std()

                if sigma == 0:
                    'If there is no variability: do not bet at all.'
                    b_n = lambda x: beta.pdf(x, 1, 1)
                    B_n = lambda x: beta.cdf(x, 1, 1)
                    return 1, b_n, B_n


                h = self.calculate_bandwidth_gaussian(data=data, bw_mehtod=self.bandwidth, sigma=sigma)

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
            self.b_n = lambda x: self.mixing_parameter * beta.pdf(x, self.ahat, self.bhat) + (1-self.mixing_parameter )
            self.B_n = lambda x: self.mixing_parameter * beta.cdf(x, self.ahat, self.bhat) + (1-self.mixing_parameter )*x
            return self.mixing_parameter * gain + (1-self.mixing_parameter)
            
        elif self.beta_method == 'mle':
            gain = self.beta_mle_betting_function(p) # Is gain really a good name? I mean the outcome of the bet...
            self.b_n = lambda x: self.mixing_parameter * beta.pdf(x, self.ahat, self.bhat) + (1-self.mixing_parameter )
            self.B_n = lambda x: self.mixing_parameter * beta.cdf(x, self.ahat, self.bhat) + (1-self.mixing_parameter )*x
            return self.mixing_parameter * gain + (1-self.mixing_parameter)
        
        elif self.beta_method == 'bayes':
            raise NotImplementedError()
        
    def beta_moment_betting_function(self, p):
        if self.n < 2:
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

        if self.n < 2:
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
    
class SimpleJumper(ConfromalTestMartingale):

    def __init__(self, J=0.01, warning_level=100, warnings=True, **kwargs):
        self.logM = 0.0
        self.max = 1.0 # Keeps track of the maximum value so far.

        self.p_values = []
        self.log_martingale_values = [0.0] # NOTE: It may be better not to store the initial value...

        self.warning_level = warning_level
        self.warnings = warnings # Do we raise a user warning or not, when the warning level is reached

        self.J = J

        self.C_epsilon = {-1: 1/3, 0: 1/3, 1: 1/3}
        self.C = 1

        self.b_epsilon = lambda u, epsilon: 1 + epsilon*(u - 1/2)

        self.b_n = lambda x: 1
        self.B_n = lambda x: x
        self.B_n_inv = lambda x: x
    
    def update_martingale_value(self, p):
        self.p_values.append(p)
        for epsilon in [-1, 0, 1]:
            self.C_epsilon[epsilon] = (1 - self.J)*self.C_epsilon[epsilon] + (self.J / 3)*self.C
            self.C_epsilon[epsilon] = self.C_epsilon[epsilon] * self.b_epsilon(p, epsilon)
        self.C = self.C_epsilon[-1] + self.C_epsilon[0] + self.C_epsilon[1]
        self.logM = np.log(self.C)
        self.log_martingale_values.append(self.logM)

        # Betting function
        epsilon_bar = (self.C_epsilon[1] - self.C_epsilon[-1])/self.C
        self.b_n = lambda u: 1 + epsilon_bar*(u - 1/2)
        self.B_n = lambda u: (epsilon_bar/2) * u**2 + (1 - epsilon_bar/2)*u
        self.B_n_inv = lambda u: (epsilon_bar - 2) / (2*epsilon_bar) + np.sqrt(epsilon_bar*(8*u + epsilon_bar - 4) + 4) / (2*epsilon_bar)

        if self.M > self.max:
            self.max = self.M

        if self.max >= self.warning_level and self.warnings:
            # TODO: Figure out how to warn only once!
            warnings.warn(f'Exchangeability assumption likely violated: Max martingale value is {self.max}')
    
    # TODO: These should live in a parent class
    @property
    def M(self):
        return np.exp(self.logM)
    
    @property
    def martingale_values(self):
        return np.exp(self.log_martingale_values)
    
    @property
    def log10_martingale_values(self):
        return np.log10(self.martingale_values)


if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
