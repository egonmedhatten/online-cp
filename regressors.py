import numpy as np
import time
import warnings
from scipy.optimize import minimize_scalar, minimize, Bounds
from scipy.spatial.distance import pdist, cdist, squareform


MACHINE_EPSILON = np.finfo(np.float64).eps


class ConformalRegressor:
    '''
    Parent class for the different ridge regressors. Holds common methods
    '''

    def __init__(self):
        pass
    

    '''
    TODO The methods _get_upper and _get_lower could be called with a vector of significance levels,
         say one casual and one highly confident. In pracice, this could be useful when the aim is 
         descision support. This would then have to play nice wiht everything else...
    '''

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
    

    @staticmethod
    def minimum_training_set(epsilon, bounds='both'):
        '''
        Returns the minimum initial training set size needed to output informative (finite) prediciton sets

        >>> from CRR import ConformalRegressor
        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(0.1)
        20

        >>> from CRR import ConformalRegressor
        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(0.1, 'upper')
        10

        >>> from CRR import ConformalRegressor
        >>> import numpy as np
        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(np.array([0.1, 0.05]))
        40

        '''
        if not hasattr(epsilon, 'shape'):
            # Then it is a scalar
            if bounds == 'both':
                return int(np.ceil(2/epsilon)) 
            else: 
                return int(np.ceil(1/epsilon)) 
        else:
            # Then it is a vector
            if bounds == 'both':
                return int(np.ceil(2/epsilon.min())) 
            else: 
                return int(np.ceil(1/epsilon.min())) 
    

    @staticmethod
    def err(Gamma, y):
        '''
        Returns 0 if y is contained in the closed (possibly infinite) inteval indicated by the endpoints, and 1 otherwise.
        
        >>> from CRR import ConformalRegressor
        >>> cp = ConformalRegressor()
        >>> Gamma = (-1, 1)
        >>> cp.err(Gamma, 0)
        0

        >>> from CRR import ConformalRegressor
        >>> cp = ConformalRegressor()
        >>> Gamma = (-1, 1)
        >>> cp.err(Gamma, 2)
        1
        '''
        return int(not(Gamma[0] <= y <= Gamma[1]))
    

    @staticmethod
    def width(Gamma):
        '''
        Returns the width of the closed (possibly infinite) inteval indicated by the endpoints. 

        >>> from CRR import ConformalRegressor
        >>> cp = ConformalRegressor()
        >>> Gamma = (-1, 1)
        >>> cp.width(Gamma)
        2
        '''
        return Gamma[1] - Gamma[0]
    

class ConformalRidgeRegressor(ConformalRegressor):
    '''
    Conformal ridge regression (Algorithm 2.4 in Algorithmic Learning in a Random World)

    Let's create a dataset with noisy evaluations of the function f(x1,x2) = x1+x2:

    >>> import numpy as np
    >>> np.random.seed(31337) # only needed for doctests
    >>> N = 30
    >>> X = np.random.uniform(0, 1, (N, 2))
    >>> y = X.sum(axis=1) + np.random.normal(0, 0.1, N)

    Import the library and create a regressor:

    >>> from CRR import ConformalRidgeRegressor
    >>> cp = ConformalRidgeRegressor()

    Learn the whole dataset:

    >>> cp.learn_initial_training_set(X, y)

    Predict an object (output may not be exactly the same, as the dataset
    depends on the random seed):
    >>> print("(%.2f, %.2f)" % cp.predict(np.array([0.5, 0.5]), epsilon=0.1, bounds='both'))
    (0.73, 1.23)

    You can of course learn a new data point online:

    >>> cp.learn_one(np.array([0.5, 0.5]), 1.0)

    The prediction set is the closed interval whose boundaries are indicated by the output.

    We can then predict again:

    >>> print("(%.2f, %.2f)" % cp.predict(np.array([2,4]), epsilon=0.1, bounds='both'))
    (5.39, 6.33)
    '''

    def __init__(self, a=0, warnings=True, autotune=False, verbose=0, rnd_state=2024):
        '''
        Setting autotune=True automatically tunes the ridge parameter using generalized cross validation when learning initial training set.
        '''
        
        self.a = a
        self.X = None
        self.y = None
        self.p = None
        self.Id = None
        self.XTXinv = None

        # Should we raise warnings
        self.warnings = warnings
        # Do we autotune ridge prarmeter on warning
        self.autotune = autotune

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)


    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.Id = np.identity(self.p)
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a*self.Id)
    

    def learn_one(self, x, y, precomputed=None):
        '''
        Learn a single example. If we have already computed X and XTXinv, use them for update. Then the last row of X is the object with label y.
        >>> cp = ConformalRidgeRegressor()
        >>> cp.learn_one(np.array([1,0]), 1)
        >>> cp.X
        array([[1, 0]])
        >>> cp.y
        array([1])
        '''
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            if hasattr(self, 'h'):
                self.y = np.append(self.y, y.reshape(1, self.h), axis=0)
            else:
                self.y = np.append(self.y, y)
    
        if precomputed is not None:
            X = precomputed['X']
            XTXinv = precomputed['XTXinv']

            if X is not None:
                self.X = X
                self.p = self.X.shape[1]
                self.Id = np.identity(self.p)
            
            if XTXinv is not None:
                self.XTXinv = XTXinv
                
            else:
                if self.X.shape[0] == 1:
                    # print(self.X)
                    # print(self.Id)
                    # print(self.a)
                    self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
                else:
                    # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                            #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
                    self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
            
                    # Check the rank
                    rank_deficient = not(self.check_matrix_rank(self.XTXinv))
                    
        else:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1,-1)
                self.p = self.X.shape[1]
                self.Id = np.identity(self.p)
            elif self.X.shape[0] == 1:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)
            else:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                        #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
                self.XTXinv -= (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
        
                # Check the rank
                rank_deficient = not(self.check_matrix_rank(self.XTXinv))


    @staticmethod
    def compute_A_and_B(X, XTXinv, y):
        n = X.shape[0]
        # Hat matrix (This block is the time consuming one...)
        H = X @ XTXinv @ X.T
        C = np.identity(n) - H
        A = C @ np.append(y, 0) # Elements of this vector are denoted ai
        B = C @ np.append(np.zeros((n-1,)), 1) # Elements of this vector are denoted bi
        # Nonconformity scores are A + yB = y - yhat
        return A, B
    

    def predict(self, x, epsilon=0.1, bounds='both', return_update=False, debug_time=False):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([1, 1, 1]), epsilon=0.1, bounds='both')
        (-inf, inf)
        """
        def build_precomputed(X, XTXinv, A, B):
            computed = {
                'X': X, # The updated matrix of objects
                'XTXinv': XTXinv, # The updated kernel matrix
                'A': A,
                'B': B,
            } 
            return computed

        if self.X is not None:

            tic = time.time()
            # Add row to X matrix
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            toc_add_row = time.time() - tic
            n = X.shape[0]
            XTXinv = None

            # Check that the significance level is not too small. If it is, return infinite prediction interval
            if bounds=='both':
                if not (epsilon >= 2/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(2/epsilon))} examples. Increase or add more examples')
                    if return_update:
                        return (-np.inf, np.inf), build_precomputed(X, XTXinv, None, None)
                    else: 
                        return (-np.inf, np.inf)
            else: 
                if not (epsilon >= 1/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(1/epsilon))} examples. Increase or add more examples')
                    if return_update:
                        return (-np.inf, np.inf), build_precomputed(X, XTXinv, None, None)
                    else: 
                        return (-np.inf, np.inf)

            tic = time.time()
            # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                    #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

            XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
            toc_update_XTXinv = time.time() - tic

            tic = time.time()
            A, B = self.compute_A_and_B(X, XTXinv, self.y)
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
            X = x.reshape(1,-1)
            XTXinv = None
            A = None
            B = None

            lower = -np.inf
            upper = np.inf
    
        if return_update:
            return (lower, upper), build_precomputed(X, XTXinv, A, B)
        else:
            return (lower, upper)
            
    
    def compute_smoothed_p_value(self, x, y, precomputed=None):
        '''
        Computes the smoothed p-value of the example (x, y).
        Smoothed p-values can be used to test the exchangeability assumption.
        If X and XTXinv are passed, x must be the last row of X.
        '''
        # Inner method to compute the p-value from NC scores
        def calc_p(A, B, y):
            if hasattr(self, 'h'):
                raise NotImplementedError('MimoConformalRidgeRegressor can not compute p-values at the moment. Working on it...')
            # Nonconformity scores are A + yB = y - yhat
            Alpha = A + y*B
            alpha_y = Alpha[-1]
            lt = np.where(Alpha < alpha_y)[0].shape[0]
            eq = np.where(Alpha == alpha_y)[0].shape[0]
            tau = self.rnd_gen.uniform(0, 1)
            p_y = (lt + tau * eq)/Alpha.shape[0]
            return p_y

        if precomputed is not None:
            A = precomputed['A']
            B = precomputed['B']
            if A is not None and B is not None:
                p_y = calc_p(A, B, y)
            else:
                if self.XTXinv is not None:
                    X = np.append(self.X, x.reshape(1, -1), axis=0)
                    XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
                    A, B = self.compute_A_and_B(X, XTXinv, self.y)
                    p_y = calc_p(A, B, y)
                    
                else:
                    p_y = self.rnd_gen.uniform(0, 1)
        
        else:
            if self.XTXinv is not None:
                X = np.append(self.X, x.reshape(1, -1), axis=0)
                XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
                A, B = self.compute_A_and_B(X, XTXinv, self.y)
                p_y = calc_p(A, B, y)
                
            else:
                p_y = self.rnd_gen.uniform(0, 1)

        return p_y


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
        if self.X is not None:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)


    def _tune_ridge_parameter(self, a0=None):
        '''
        Tune ridge parameter with Generalized cross validation https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/golub.heath.wahba.pdf
        '''
        XTX = self.X.T @ self.X
        n = self.X.shape[0]
        In = np.identity(n)
        def GCV(a):
            try:
                A = self.X @ np.linalg.inv(XTX + a*self.Id) @ self.X.T
                return (1/n)*np.linalg.norm((In - A) @ self.y)**2 / ((1/n)* np.trace(In- A))**2
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return np.inf
        
        # Initial guess
        if a0 is None:
            a0 = MACHINE_EPSILON # Just a small pertubation to avoid numerical issues

        # Bounds to ensure a >= 0
        res = minimize(GCV, x0=a0, bounds=Bounds(lb=MACHINE_EPSILON, keep_feasible=True)) # May be relevant to pass some arguments here, or even use another minimizer.
        a = res.x[0]

        if self.verbose > 0:
            print(f'New ridge parameter: {a}')
        self.change_ridge_parameter(a)


    # TODO
    def prune_training_set(self):
        '''
        Just an idea at the moment, but perhaps we should have some inclusion criteria for examples to only include the informative ones. Could improve accuracy, but also significantly decrease computation time if we have a large dataset.
        '''
        raise NotImplementedError


    def check_matrix_rank(self, M):
        '''
        Check if a matrix has full rank <==> is invertible
        Returns False if matrix is rank deficient
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
        

if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
