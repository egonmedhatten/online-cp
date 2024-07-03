import numpy as np
import time
import warnings
from scipy.optimize import minimize_scalar

'''
The most basic version of CRR predicts a single target. 
This conformal predictor is not smoothed, so p-values can not be used to test exchangeability.
If we want to ad an exchangeability test, we have to use some other conformal transducer, e.g. 1-NN as in https://www.alrw.net/articles/04.pdf
We could also add kernel ridge regression.
'''


class ConformalRidgeRegressor:
    '''
        Conformal ridge regression (Algorithm 2.4 in Algorithmic Learning in a Random World)
    '''

    def __init__(self, a=0, warnings=True):
        '''
            Initialise.
            Maybe input ridge parameter. Maybe input target miss-coverage level.
        '''
        self.a = a
        self.X = None
        self.y = None
        self.p = None
        self.Id = None
        self.XTXinv = None

        # Should we raise warnings
        self.warnings = warnings


    @staticmethod
    def _get_upper(u_dic, epsilon, n):
        try:
            upper = u_dic[int(np.ceil((1 - epsilon)*n))]
        except KeyError:
            upper = np.inf
        return upper


    @staticmethod
    def _get_lower(l_dic, epsilon, n):
        try:
            lower = l_dic[int(np.floor(epsilon*n))]
        except KeyError:
            lower = -np.inf
        return lower


    @staticmethod
    def _vectorised_l_and_u(A, B):
        '''A and B are columns'''
        # Calculate differences
        differences = B[-1] - B
        
        # Create an array to store results
        l = np.empty_like(B, dtype=float)
        u = np.empty_like(B, dtype=float)
        
        # Calculate values where differences are positive
        mask = differences > 0
        l[mask] = (A[mask] - A[-1]) / differences[mask]
        u[mask] = (A[mask] - A[-1]) / differences[mask]
        
        # Assign positive infinity where differences are non-positive
        l[~mask] = -np.inf
        u[~mask] = np.inf
        
        l = np.sort(u, axis=0)[:-1]
        u = np.sort(u, axis=0)[:-1]

        # These are just to avoid messing with the python indexing. Could probably be removed for efficiency
        l_dic = {i+1: val for i, val in enumerate(l)}
        u_dic = {i+1: val for i, val in enumerate(u)}

        return l_dic, u_dic


    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.Id = np.identity(self.p)
        self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a*self.Id)
    

    def learn_one(self, x, y):
        '''
        Learn a single example
        >>> cp = ConformalRidgeRegressor()
        >>> cp.learn_one(np.array([1,0]), 1)
        >>> cp.X
        array([[1, 0]])
        >>> cp.y
        array([1])
        '''
        # Learn object x
        if self.X is None:
            self.X = x.reshape(1,-1)
            self.p = self.X.shape[1]
            self.Id = np.identity(self.p)
        elif self.X.shape[0] == 1:
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
            self.check_matrix_rank(self.XTXinv)
        else:
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)
            # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                    #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
            self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
            self.check_matrix_rank(self.XTXinv)
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)


    def predict(self, x, epsilon=0.1, bounds='both', debug_time=False):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), epsilon=0.1, bounds='both')
        (-inf, inf)
        """

        if self.X is not None:

            tic = time.time()
            # Add row to X matrix
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            toc_add_row = time.time() - tic
            n = X.shape[0]

            # Check that the significance level is not too small. If it is, return infinite prediction interval
            if bounds=='both':
                if not (epsilon >= 2/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(2/epsilon))} examples. Increase or add more examples')
                    return (-np.inf, np.inf)
            else: 
                if not (epsilon >= 1/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(1/epsilon))} examples. Increase or add more examples')
                    return (-np.inf, np.inf)

            tic = time.time()
            # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                    #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

            XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
            toc_update_XTXinv = time.time() - tic

            tic = time.time()
            # Hat matrix (This block is the time consuming one...)
            H = X @ XTXinv @ X.T
            C = np.identity(n) - H
            A = C @ np.append(self.y, 0) # Elements of this vector are denoted ai
            B = C @ np.append(np.zeros((n-1,)), 1) # Elements of this vector are denoted bi
            # Nonconformity scores are A + yB = y - yhat
            toc_nc = time.time() - tic

            tic = time.time()
            l_dic, u_dic = self._vectorised_l_and_u(A, B)
            toc_dics = time.time() - tic

            if bounds=='both':
                lower = self._get_lower(l_dic=l_dic, epsilon=epsilon/2, n=n)
                upper = self._get_upper(u_dic=u_dic, epsilon=epsilon/2, n=n)
            elif bounds=='lower':
                lower = self._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
                upper = np.inf
            elif bounds=='upper':
                lower = -np.inf
                upper = self._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
            else: 
                raise Exception

            if debug_time:
                print(f'Add row: {toc_add_row}')
                print(f'Update kernel: {toc_update_XTXinv}')
                print(f'NC scores: {toc_nc}')
                print(f'l and u: {toc_dics}')
                print()
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            lower = -np.inf
            upper = np.inf

        return lower, upper
    

    @staticmethod
    def err(Gamma, y):
        return int(not(Gamma[0] <= y <= Gamma[1]))
    

    @staticmethod
    def width(Gamma):
        return Gamma[1] - Gamma[0]


    def change_ridge_parameter(self, a):
        '''
        Change the ridge parameter
        >>> cp = ConformalRidgeRegressor()
        >>> cp.learn_one(np.array([1,0]), 1)
        >>> cp.change_ridge_parameter(1)
        >>> cp.a
        1
        '''
        self.a = a
        if self.XTXinv is not None:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)


    def tune_ridge_parameter(self):
        '''
        Tune ridge parameter with Generalized cross validation https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/golub.heath.wahba.pdf
        '''
        XTX = self.X.T @ self.X
        n = self.X.shape[0]
        In = np.identity(n)
        def GCV(a):
            A = self.X @ np.linalg.inv(XTX + a*self.Id) @ self.X.T
            return (1/n)*np.linalg.norm((In - A) @ self.y)**2 / ((1/n)* np.trace(In- A))**2
        
        res = minimize_scalar(GCV) # May be relevant to pass some arguments here, or even use another minimizer.
        a = res.x
        print(f'New ridge parameter: {a}')
        self.change_ridge_parameter(a)


    def prune_training_set(self):
        '''
        Just an idea at the moment, but perhaps we should have some inclusion criteria for examples to only include the informative ones. Could improve accuracy, but also significantly decrease computation time if we have a large dataset.
        '''
        pass


    def check_matrix_rank(self, M):
        '''
        Check if a matrix has full rank <==> is invertible
        NOTE In numerical linear algebra it is a bit more subtle. The condition number can tell us more.

        >>> cp = ConformalRidgeRegressor(warnings=False)
        >>> cp.check_matrix_rank(np.array([[1, 0], [1, 0]]))
        False
        >>> cp.check_matrix_rank(np.array([[1, 0], [0, 1]]))
        True
        '''
        if np.linalg.matrix_rank(M) < M.shape[0]:
            if self.warnings:
                warnings.warn(f'The matrix X is rank deficient. Condition number: {np.linalg.cond(M)}. Consider changing the ridge prarmeter')
            return False
        else:
            return True


# TODO
'''
    Should we add some scaler? Don't know if it is neccesary for Ridge
    Possibly add a class MimoConformalRidgeRegressor
    Possibly add a class ExchangeabilityMartingale that takes a betting function as argument.
    Possibly add CPS version of ridge regressor?
    Possibly add a TeachingSchedule?
    Possibly add ACI, both for single, and MIMO CRR?
    (Possibly add C-MFAC as for ACI?)
    Add references to papers and books to README
'''


if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
