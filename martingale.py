import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
import warnings


class PluginMartingale:
    '''
    A conformal test martingale to test exchangeability online.
    We reject exchangeability with confidence 1-1/alpha if we ever observe M >= alpha for any alpha>0

    >>> martingale = PluginMartingale()
    >>> martingale.M
    1.0
    '''

    def __init__(self, betting_function='kernel', warning_level=100, warnings=True):
        '''
        For numerical reasons, it is better to update the martingale in log scale
        Warning level set to 100 means that we warn if the exchangeability hypothesis can 
        be discarded with confidence at least 0.99.
        If you do not want to be warned, set to np.inf
        '''
        self.logM = 0.0
        self.max = 1 # The maximum value ever reachec by the maringale

        # At the moment, we just have one betting function, but more can be added.
        self.betting_function = betting_function

        self.warning_level = warning_level
        self.warnings = warnings

        self.p_values = []

    def kernel_density_betting_function(self, p_values):
        '''
        Betting function from Vovk paper
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
            
        d, norm_fac = get_density_estimate(p_values[:-1])
        return betting_function(p_values[-1], d, norm_fac)
    
    def update_log_martingale(self, p):
        self.p_values.append(p)
        if self.betting_function == 'kernel':
            self.logM += np.log(self.kernel_density_betting_function(self.p_values))
        else:
            raise NotImplementedError('Currently only kernel betting function is available. More to come...')
        # Update the running max
        if self.M > self.max:
            self.max = self.M

        if self.max >= self.warning_level and self.warnings:
            warnings.warn(f'Exchangeability assumption likely violated: Max martingale value is {self.max}')


    @property
    def M(self):
        return np.exp(self.logM)
    

    def offline_martingale(self, p_values):
        '''
        Compute martingale offline, given a sequence of p-values
        '''
        if self.betting_function == 'kernel':
            log_betting_function = lambda p_values: np.log(self.kernel_density_betting_function(p_values))
        else:
            raise NotImplementedError('Currently only kernel betting function is available. More to come...')
        p_values = np.array(p_values).flatten() # Could be a list or a vector, so make it into a vector either way
        logM = 0.0
        logM_arr = np.zeros((p_values.shape[0] + 1, ))
        logM_arr[0] = logM
        for i, _ in enumerate(p_values):
            logM += log_betting_function(p_values[:i+1])
            logM_arr[i+1] = logM
        
        M_arr = np.exp(logM_arr)
        max = M_arr.max()
        M = np.exp(logM)
        return max, M, M_arr

if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
