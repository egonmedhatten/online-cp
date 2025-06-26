import numpy as np
import time
import warnings
from scipy.optimize import minimize_scalar, minimize, Bounds
from scipy.spatial.distance import pdist, cdist, squareform


MACHINE_EPSILON = lambda x: np.abs(x) * np.finfo(np.float64).eps

default_epsilon = 0.1

# FIXME: The p-values for all regressors should be modified. We should be able to compute the upper, lower, and combined p-value

class ConformalPredictionInterval:

    def __init__(self, lower, upper, epsilon):
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon

    def __contains__(self, y):
        return self.lower <= y <= self.upper
    
    def width(self):
        return self.upper - self.lower
    
    def __repr__(self):
        return repr((self.lower, self.upper))

    def __str__(self):
        return f'({self.lower}, {self.upper})'

class ConformalRegressor:
    '''
    Parent class for the different ridge regressors. Holds common methods
    '''

    def __init__(self, epsilon=default_epsilon):
        self.epsilon = epsilon
    

    '''
    TODO The methods _get_upper and _get_lower could be called with a vector of significance levels,
         say one casual and one highly confident. In pracice, this could be useful when the aim is 
         descision support. This would then have to play nice wiht everything else...
    '''
    def _construct_Gamma(self, lower, upper, epsilon):
        return ConformalPredictionInterval(lower, upper, epsilon)

    @staticmethod
    def _safe_size_check(X):
        if X is None:
            size = 0
        else:
            size = X.shape[0]
        return size
    
    @staticmethod
    def _calculate_p(Alpha, tau=None, c_type='nonconformity'):
        '''
        Method to compute the smoothed p-value, given an array of nonconformity scores where the last element corresponds 
        to the test object, and a random number tau. If tau is None, the non-smoothed p-value is returned.
        '''
        if c_type == 'nonconformity':
            alpha_y = Alpha[-1]
            if tau is not None:
                gt = np.where(Alpha > alpha_y)[0].size
                eq = np.where(Alpha == alpha_y)[0].size
                p_y = (gt + tau * eq) / Alpha.size
            else:
                geq = np.where(Alpha >= alpha_y)[0].size
                p_y = geq / Alpha.size
        elif c_type == 'conformity':
            alpha_y = Alpha[-1]
            if tau is not None:
                lt = np.where(Alpha < alpha_y)[0].size
                eq = np.where(Alpha == alpha_y)[0].size
                p_y = (lt + tau * eq) / Alpha.size
            else:
                leq = np.where(Alpha <= alpha_y)[0].size
                p_y = leq / Alpha.size
        else:
            raise Exception()
        return p_y

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

        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(0.1)
        20

        >>> cp = ConformalRegressor()
        >>> cp.minimum_training_set(0.1, 'upper')
        10

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
        return int(not(y in Gamma))
    

    @staticmethod
    def width(Gamma):
        return Gamma.width()
    

    def learn_many(self, X, y):
        for x1, y1 in zip(X, y):
            self.learn_one(x1,y1)

    
    def process_dataset(self, X, y, epsilon=None, init_train=0, return_results=False):
        if epsilon is None:
            epsilon = self.epsilon

        Err = 0
        Width = 0

        X_train = X[:init_train]
        y_train = y[:init_train]
        X_run = X[init_train:]
        y_run = y[init_train:]

        if return_results:
            res = np.zeros(shape=(y_run.shape[0], 2))
            prediction_sets = {}

        self.learn_initial_training_set(X=X_train, y=y_train)

        time_init = time.time()
        for i, (obj, lab) in enumerate(zip(X_run, y_run)):
            
            # Make prediction
            Gamma = self.predict(obj, epsilon=epsilon) 

            # Check error
            Err += self.err(Gamma, lab)

            # Learn the label
            self.learn_one(obj, lab)
            
            # Width of interval
            width = self.width(Gamma)
            Width += width

            if return_results:
                res[i, 0] = Err
                res[i, 1] = width
                prediction_sets[i] = Gamma

        time_process = time.time() - time_init

        result = {
            'Efficiency': {
                'Average error': Err/self.y.shape[0],
                'Average width': Width/self.y.shape[0],
                'Time': time_process
                }
            }
        if return_results:
            result['Prediction sets'] = prediction_sets
            result['Cummulative Err'] = res[:, 0]
            result['Width'] = res[:, 1]
        
        return result

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

    >>> cp = ConformalRidgeRegressor()

    Learn the whole dataset:

    >>> cp.learn_initial_training_set(X, y)

    Predict an object (output may not be exactly the same, as the dataset
    depends on the random seed):
    >>> interval = cp.predict(np.array([0.5, 0.5]), bounds='both')
    >>> print("(%.2f, %.2f)" % (interval.lower, interval.upper))
    (0.73, 1.23)

    You can of course learn a new data point online:

    >>> cp.learn_one(np.array([0.5, 0.5]), 1.0)

    The prediction set is the closed interval whose boundaries are indicated by the output.

    We can then predict again:

    >>> interval = cp.predict(np.array([2,4]), bounds='both')
    >>> print("(%.2f, %.2f)" % (interval.lower, interval.upper))
    (5.39, 6.33)
    '''
    # TODO: Fix gracefull error handling when the matrix is singular. It should raise an exception, but we could
    #       specify that it can be handled by changing the ridge parameter.

    def __init__(self, a=0, warnings=True, autotune=False, verbose=0, rnd_state=None, studentised=False, epsilon=default_epsilon):
        '''
        The ridge parameter (L2 regularisation) is a.
        Setting autotune=True automatically tunes the ridge parameter using generalized cross validation when learning initial training set.
        '''
        super().__init__(epsilon=epsilon)
        
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

        # Do we use the studentised residuals
        self.studentised = studentised


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
                    if self.warnings:
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
                if self.warnings:
                    rank_deficient = not(self.check_matrix_rank(self.XTXinv))


    
    def compute_A_and_B_OLD(self, X, XTXinv, y):
        n = X.shape[0]
        # Hat matrix (This block is the time consuming one...)
        H = X @ XTXinv @ X.T
        C = np.identity(n) - H
        A = C @ np.append(y, 0) # Elements of this vector are denoted ai
        B = C @ np.append(np.zeros((n-1,)), 1) # Elements of this vector are denoted bi
        if self.studentised:
            h = H.diagonal()
            A = A / np.sqrt(1 - h)
            B = B / np.sqrt(1 - h)
        # Nonconformity scores are A + yB = y - yhat
        return A, B
    
    def compute_A_and_B(self, X, XTXinv, y):
        """
        Efficient and correct computation of A and B for conformal ridge regression.
        X: (n, d) augmented matrix (last row is test object)
        XTXinv: (d, d) inverse of X.T @ X + a*I for augmented X
        y: (n-1,) training labels (no test label)
        """
        y_ext = np.append(y, 0)  # y with test point (last row) as 0

        # Compute beta using the augmented X and y_ext (just like the old code)
        beta = XTXinv @ X.T @ y_ext  # (d, d) @ (d, n) @ (n,) -> (d,)

        # Fitted values for all points (including test)
        y_hat = X @ beta  # (n, d) @ (d,) -> (n,)

        # Compute hat matrix diagonal for all points using XTXinv (augmented)
        H_diag = np.sum(X @ XTXinv * X, axis=1)  # (n,)

        # Compute last column of H efficiently
        h_col = X @ XTXinv @ X[-1]  # (n, d) @ (d,) -> (n,)

        # A and B for each point
        A = y_ext - y_hat
        B = -h_col
        B[-1] += 1  # e_{-1}[-1] = 1

        if self.studentised:
            A = A / np.sqrt(1 - H_diag + 1e-12)
            B = B / np.sqrt(1 - H_diag + 1e-12)

        return A, B
    

    def predict(self, x, epsilon=None, bounds='both', return_update=False, debug_time=False):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), bounds='both')
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

        if epsilon is None:
            epsilon = self.epsilon

        if self._safe_size_check(self.X) > 0:

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
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, XTXinv, None, None)
                    else: 
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)
            else: 
                if not (epsilon >= 1/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(1/epsilon))} examples. Increase or add more examples')
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, XTXinv, None, None)
                    else: 
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)

            tic = time.time()
            # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                    #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

            XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
            toc_update_XTXinv = time.time() - tic
            
            tic = time.time()
            A, B = self.compute_A_and_B(X, XTXinv, self.y)
            toc_nc = time.time() - tic

            if self.studentised:
                tic = time.time()
                t = (A[:-1] - A[-1]) / (B[-1] - B[:-1])
                t.sort()
                l_dic = {i+1: val for i, val in enumerate(t)}
                u_dic = {i+1: val for i, val in enumerate(t)}
                toc_dics = time.time() - tic
            else:
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
            return self._construct_Gamma(lower, upper, epsilon), build_precomputed(X, XTXinv, A, B)
        else:
            return self._construct_Gamma(lower, upper, epsilon)

    def compute_p_value(self, x, y, bounds='both', precomputed=None, tau=None, smoothed=True):
        '''
        Computes the smoothed p-value of the example (x, y).
        '''
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            
            assert np.allclose(x, precomputed['X'][-1])
            A = precomputed['A']
            B = precomputed['B']
        else:
            if self.XTXinv is not None:
                X = np.append(self.X, x.reshape(1, -1), axis=0)
                XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
                A, B = self.compute_A_and_B(X, XTXinv, self.y)
            else:
                A, B = None, None 
        
        if A is not None and B is not None:
            if bounds == 'both':
                E = A + y*B
                Alpha = np.zeros_like(A)
                for i, e in enumerate(E):
                    alpha = min((E >= e).sum(), (E<=e).sum())
                    Alpha[i] = alpha
                c_type = 'conformity'
            elif bounds == 'lower':
                Alpha = -(A + y*B)
                c_type = 'nonconformity'
            elif bounds=='upper':
                Alpha = A + y*B
                c_type = 'nonconformity'
            else:
                raise Exception('bounds must be one of "both", "lower", "upper"')

            if smoothed:
                p = self._calculate_p(Alpha, tau, c_type=c_type)
            else: 
                p = self._calculate_p(Alpha, c_type=c_type)
        else:
            if smoothed:
                p = tau
            else:
                p = 1

        return p
    

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
            a0 = 1e-6 # Just a small pertubation to avoid numerical issues

        # Bounds to ensure a >= 0
        res = minimize(GCV, x0=a0, bounds=Bounds(lb=1e-6, keep_feasible=True)) # May be relevant to pass some arguments here, or even use another minimizer.
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


class KernelConformalRidgeRegressor(ConformalRegressor):

    # TODO Add doctests to methods where applicable

    def __init__(self, kernel, a=0, warnings=True, verbose=0, rnd_state=None, epsilon=default_epsilon):
        '''
        KernelConformalRidgeRegressor requires a kernel. Some common kernels are found in kernels.py, but it is 
        also compatible with (most) kernels from e.g. scikit-learn.
        Custom kernels can also be passed as callable functions.
        '''
        super().__init__(epsilon=epsilon)

        self.a = a
        self.X = None
        self.y = None
        self.p = None
        self.Id = None
        self.K = None
        self.Kinv = None

        self.kernel = kernel

        # Should we raise warnings
        self.warnings = warnings
        
        self.verbose = verbose

        self.rnd_gen = np.random.default_rng(rnd_state)

    
    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        Id = np.identity(self.X.shape[0])
       
        self.K = self.kernel(self.X)
        self.Kinv = np.linalg.inv(self.K + self.a * Id)


    @staticmethod
    def _update_Kinv(Kinv, k, kappa):
        # print(f'K: {K}')
        # print(f'k: {k}')
        # print(f'kappa: {kappa}')
        d = 1 / (kappa - k.T @ Kinv @ k)
        return np.block([[Kinv + d * Kinv @ k @ k.T @ Kinv, -d * Kinv @ k], [ -d * k.T @ Kinv, d]])


    @staticmethod
    def _update_K(K, k, kappa):
        # print(f'K: {K}')
        # print(f'k: {k}')
        # print(f'kappa: {kappa}')
        return np.block([[K, k], [k.T, kappa]])


    def learn_one(self, x, y, precomputed=None):
        '''
        Learn a single example
        '''
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)
        
        if precomputed is not None:
            X = precomputed['X']
            K = precomputed['K']
            Kinv = precomputed['Kinv']

            if X is not None:
                self.X = X

            if K is not None and Kinv is not None:
                self.K = K
                self.Kinv = Kinv

            else:
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
        
        else:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1,-1)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
            elif self.X.shape[0] == 1:
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
            else:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                self.K = self._update_K(self.K, k, kappa)
                self.Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)


    @staticmethod
    def compute_A_and_B(X, K, Kinv, y):
        # print(f'X: {X}')
        # print(f'K: {K}')
        # print(f'Kinv: {Kinv}')
        # print(f'y: {y}')
        n = X.shape[0]
        H = Kinv @ K
        C = np.identity(n) - H
        A = C @ np.append(y, 0) # Elements of this vector are denoted ai
        B = C @ np.append(np.zeros((n-1,)), 1) # Elements of this vector are denoted bi
        # Nonconformity scores are A + yB = y - yhat
        return A, B
    
    
    def predict(self, x, epsilon=None, bounds='both', return_update=False, debug_time=False):
        """
        This function makes a prediction.

        If you start with no training,
        you get a null prediciton between
        -infinity and +infinity.

        TODO Add possibility to learn object to save time

        >>> cp = ConformalRidgeRegressor()
        >>> cp.predict(np.array([0.506, 0.22, -0.45]), bounds='both')
        (-inf, inf)
        """
        def build_precomputed(X, K, Kinv, A, B):
            computed = {
                'X': X, # The updated matrix of objects
                'K': K, # The updated kernel matrix
                'Kinv': Kinv,
                'A': A,
                'B': B,
            } 
            return computed

        if epsilon is None:
            epsilon = self.epsilon
        
        if self.X is not None:

            tic = time.time()
            
            # Temporarily update kernel matrix
            k = self.kernel(self.X, x).reshape(-1, 1)
            kappa = self.kernel(x, x)
            K = self._update_K(self.K, k, kappa)
            Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)

            toc_update_kernel = time.time() - tic

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
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, K, Kinv, None, None)
                    else: 
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)
            else: 
                if not (epsilon >= 1/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(1/epsilon))} examples. Increase or add more examples')
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, K, Kinv, None, None)
                    else: 
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)

            
            tic = time.time()
            A, B = self.compute_A_and_B(X, K, Kinv, self.y)
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
                print(f'Update kernel: {toc_update_kernel}')
                print(f'NC scores: {toc_nc}')
                print(f'l and u: {toc_dics}')
                print()
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            X = x.reshape(1,-1)
            K = None
            Kinv = None
            A = None
            B = None
            
            lower = -np.inf
            upper = np.inf

        if return_update:
            return self._construct_Gamma(lower, upper, epsilon), build_precomputed(X, K, Kinv, A, B)
        else:
            return self._construct_Gamma(lower, upper, epsilon)
        

    def compute_p_value(self, x, y, bounds='both', precomputed=None, tau=None, smoothed=True):
        '''
        Computes the smoothed p-value of the example (x, y).
        '''
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            
            assert np.allclose(x, precomputed['X'][-1])
            A = precomputed['A']
            B = precomputed['B']

        else:
            if self.Kinv is not None:

                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                K = self._update_K(self.K, k, kappa)
                Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                X = np.append(self.X, x.reshape(1, -1), axis=0)
                A, B = self.compute_A_and_B(X, K, Kinv, self.y)
            else:
                A, B = None, None 
        
        if A is not None and B is not None:
            if bounds == 'both':
                E = A + y*B
                Alpha = np.zeros_like(A)
                for i, e in enumerate(E):
                    alpha = min((E >= e).sum(), (E<=e).sum())
                    Alpha[i] = alpha
                c_type = 'conformity'
            elif bounds == 'lower':
                Alpha = -(A + y*B)
                c_type = 'nonconformity'
            elif bounds=='upper':
                Alpha = A + y*B
                c_type = 'nonconformity'
            else:
                raise Exception('bounds must be one of "both", "lower", "upper"')

            if smoothed:
                p = self._calculate_p(Alpha, tau, c_type=c_type)
            else: 
                p = self._calculate_p(Alpha, c_type=c_type)
        else:
            if smoothed:
                p = tau
            else:
                p = 1

        return p
    

    def compute_smoothed_p_value(self, x, y, precomputed=None):
        '''
        Computes the smoothed p-value of the example (x, y).
        Smoothed p-values can be used to test the exchangeability assumption.
        '''

        # Inner method to compute the p-value from NC scores
        def calc_p(A, B, y):
            # Nonconformity scores are A + yB = y - yhat
            Alpha = A + y*B
            alpha_y = Alpha[-1]
            gt = np.where(Alpha > alpha_y)[0].shape[0]
            eq = np.where(Alpha == alpha_y)[0].shape[0]
            tau = self.rnd_gen.uniform(0, 1)
            p_y = (gt + tau * eq)/Alpha.shape[0]
            return p_y
        if precomputed is not None:
            A = precomputed['A']
            B = precomputed['B']
            X = precomputed['X']
            K = precomputed['K']
            Kinv = precomputed['Kinv']

            if A is not None and B is not None:
                p_y = calc_p(A, B, y)
            else:
                if Kinv is not None and X is not None and K is not None:

                    A, B = self.compute_A_and_B(X, K, Kinv, self.y)
                    p_y = calc_p(A, B, y)

                else:
                    p_y = self.rnd_gen.uniform(0, 1)
                
        else:
            if self.Kinv is not None:
                
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                K = self._update_K(self.K, k, kappa)
                Kinv = self._update_Kinv(self.Kinv, k, kappa + self.a)
                X = np.append(self.X, x.reshape(1, -1), axis=0)

                A, B = self.compute_A_and_B(X, K, Kinv, self.y)
                p_y = calc_p(A, B, y)

            else:
                p_y = self.rnd_gen.uniform(0, 1)
        return p_y


class ConformalNearestNeighboursRegressor(ConformalRegressor):

    def __init__(self, k, distance='euclidean', distance_func=None, aggregation_method='mean', warnings=True, verbose=0, rnd_state=None, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        
        self.k = k
        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = 'custom'

        self.aggregation_method = aggregation_method
        if aggregation_method == 'mean':
            self.agg_func = np.mean
        elif aggregation_method == 'median':
            self.agg_func = np.median

        self.X = None
        self.y = None
        self.D = None

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

        self.warnings = warnings


    def _standard_distance_func(self, X, y=None):
        '''
        By default we use scipy to compute distances
        '''
        X = np.atleast_2d(X)
        if y is None:
            dists = squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            dists = cdist(X, y, metric=self.distance)
        return dists
    

    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        self.D = self.distance_func(X)

    @staticmethod
    def update_distance_matrix(D, d):
        return np.block([[D, d], [d.T, np.array([0])]])

    def learn_one(self, x, y, precomputed=None):
        '''
        precomputed is a dictionary
        {
            'X': X,
            'D': D,
            'A': A,
            'B': B
        }
        '''
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)

        if precomputed is None:
            # Learn object x
            if self.X is None:
                self.X = x.reshape(1,-1)
                self.D = self.distance_func(self.X)
            else:
                d = self.distance_func(self.X, x)
                self.D = self.update_distance_matrix(self.D, d)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
        else:
            self.X = precomputed['X']
            self.D = precomputed['D']
    

    def predict(self, x, epsilon=None, bounds='both', return_update=False, debug_time=False):

        def build_precomputed(X, D, A, B):
            computed = {
                'X': X,
                'D': D,
                'A': A,
                'B': B
            }
            return computed

        if epsilon is None:
            epsilon = self.epsilon
        
        if self._safe_size_check(self.X) > self.k:

            # Temporarily update distance matrix
            tic = time.time()
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            toc_update_D = time.time() - tic

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
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, D, None, None)
                    else: 
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)
            else: 
                if not (epsilon >= 1/n):
                    if self.warnings:
                        warnings.warn(f'Significance level epsilon is too small for training set. Need at least {int(np.ceil(1/epsilon))} examples. Increase or add more examples')
                    if return_update:
                        return self._construct_Gamma(-np.inf, np.inf, epsilon), build_precomputed(X, D, None, None)
                    else: 
                        return self._construct_Gamma(-np.inf, np.inf, epsilon)
            
            tic = time.time()
            A, B = self.compute_A_and_B(D, self.y, self.k)
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
                print(f'Update D: {toc_update_D}')
                print(f'NC scores: {toc_nc}')
                print(f'l and u: {toc_dics}')
                print()
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            lower = -np.inf
            upper = np.inf
            X = x.reshape(1,-1)
            D = self.distance_func(X)
            if self._safe_size_check(self.X) > 0:
                A, B = self.compute_A_and_B(D, self.y, self.k)
            else:
                A, B = None, None
        
        if return_update:
            return self._construct_Gamma(lower, upper, epsilon), build_precomputed(X, D, A, B)
        else:
            return self._construct_Gamma(lower, upper, epsilon)


    # FIXME This is very wrong!!
    def compute_A_and_B(self, D, y, k):
        y = np.append(y, 0)
        n = D.shape[0] - 1 
        A = np.zeros(n + 1)
        B = np.zeros(n + 1)

        k_nearest = D.argsort(axis=0)[1:k+1]

        for i, col in enumerate(k_nearest.T): # Transpose to iterate over columns
            if i < n:
                A[i] = y[i] - self.agg_func(y[col])
                if n in col:
                    B[i] = -1/k
                else:
                    B[i] = 0
            else:
                A[i] = - self.agg_func(y[col])
                B[i] = 1
        print(A)
        print(B)
        return A, B
    

    @staticmethod
    def _calculate_p(Alpha, tau=None):
        '''
        Method to compute the smoothed p-value, given an array of nonconformity scores where the last element corresponds 
        to the test object, and a random number tau. If tau is None, the non-smoothed p-value is returned.
        '''
        alpha_y = Alpha[-1]
        if tau is not None:
            gt = np.where(Alpha > alpha_y)[0].size
            eq = np.where(Alpha == alpha_y)[0].size
            p_y = (gt + tau * eq) / Alpha.size
        else:
            geq = np.where(Alpha >= alpha_y)[0].size
            p_y = geq / Alpha.size
        return p_y

    def compute_p_value(self, x, y, bounds='both', precomputed=None, tau=None, smoothed=True):
        '''
        Computes the smoothed p-value of the example (x, y).
        '''
        if tau is None and smoothed:
            tau = self.rnd_gen.uniform(0, 1)
        if precomputed is not None:
            
            assert np.allclose(x, precomputed['X'][-1])
            D = precomputed['D']
            A, B = self.compute_A_and_B(D, self.y, self.k)
        else:
            if self._safe_size_check(self.X) > self.k:
                d = self.distance_func(self.X, x)
                D = self.update_distance_matrix(self.D, d)
                A, B = self.compute_A_and_B(D, self.y, self.k)
            else:
                X = x.reshape(1,-1)
                D = self.distance_func(X)
                if self._safe_size_check(self.X) > 0:
                    self.compute_A_and_B(D, self.y, self.k)
                else:
                    A, B = None, None 
        
        if A is not None and B is not None:
            if bounds == 'both':
                E = A + y*B
                Alpha = np.zeros_like(A)
                for i, e in enumerate(E):
                    alpha = min((E >= e).sum(), (E<=e).sum())
                    Alpha[i] = alpha
            elif bounds == 'lower':
                Alpha = -(A + y*B)
            elif bounds=='upper':
                Alpha = A + y*B
            else:
                raise Exception('bounds must be one of "both", "lower", "upper"')

            if smoothed:
                p = self._calculate_p(Alpha, tau)
            else: 
                p = self._calculate_p(Alpha)
        else:
            if smoothed:
                p = tau
            else:
                p = 1

        return p
    

    # def compute_smoothed_p_value(self, x, y, precomputed=None):
    #     '''
    #     Computes the smoothed p-value of the example (x, y).
    #     Smoothed p-values can be used to test the exchangeability assumption.
    #     If X and XTXinv are passed, x must be the last row of X.
    #     '''

    #     # Inner method to compute the p-value from NC scores
    #     def calc_p(A, B, y):
    #         # Nonconformity scores are A + yB = y - yhat
    #         Alpha = A + y*B
    #         alpha_y = Alpha[-1]
    #         gt = np.where(Alpha < alpha_y)[0].shape[0]
    #         eq = np.where(Alpha == alpha_y)[0].shape[0]
    #         tau = self.rnd_gen.uniform(0, 1)
    #         p_y = (gt + tau * eq)/Alpha.shape[0]
    #         return p_y
        
    #     if precomputed is not None:
    #         A = precomputed['A']
    #         B = precomputed['B']
    #         if A is not None and B is not None:
    #             p_y = calc_p(A, B, y)
    #         else:
    #             X = precomputed['X']
    #             if self.X is not None:
    #                 assert np.allclose(x, X[-1])
    #                 D = precomputed['D']
    #                 A, B = self.compute_A_and_B(D, self.y, self.k)
    #                 p_y = calc_p(A, B, y)
    #             else:
    #                 p_y = self.rnd_gen.uniform(0, 1)
    #     else:
    #         if self.X is not None:
    #             d = self.distance_func(self.X, x)
    #             D = self.update_distance_matrix(self.D, d)
    #             A, B = self.compute_A_and_B(D, self.y, self.k)
    #             p_y = calc_p(A, B, y)
    #         else:
    #             p_y = self.rnd_gen.uniform(0, 1)
    #     return p_y
    

if __name__ == "__main__":
    import doctest
    import sys

    (failures, _) = doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE) # XXX: DO NOT COMMIT ME!!!!!!!!!!!!!!!!!!!!!!
    if failures:
        sys.exit(1)

        

    def test_equivalence_of_A_and_B():
        """
        Test that compute_A_and_B_OLD and compute_A_and_B return (almost) identical results.
        Test on the California Housing dataset, rather than synthetic data.
        """
        # np.random.seed(42)
        # n, d = 10, 3
        # X = np.random.randn(n, d)
        # y = np.random.randn(n-1)

        # Load California housing dataset
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = np.array(data.data)
        y = np.array(data.target)
        a=0

        # Create a dummy regressor instance
        cp = ConformalRidgeRegressor(a=a)
        cp.learn_initial_training_set(X=X[:-1], y=y[:-1])

        x = X[-1]
        X_internal = np.append(cp.X, x.reshape(1, -1), axis=0)
        XTXinv_internal = cp.XTXinv - (cp.XTXinv @ np.outer(x, x) @ cp.XTXinv) / (1 + x.T @ cp.XTXinv @ x)

        # OLD
        A_old, B_old = cp.compute_A_and_B_OLD(X_internal, XTXinv_internal, cp.y)
        # NEW
        A_new, B_new = cp.compute_A_and_B(X_internal, XTXinv_internal, cp.y)

        print("A_old:", A_old)
        print("A_new:", A_new)
        print("B_old:", B_old)
        print("B_new:", B_new)

        assert np.allclose(A_old, A_new, atol=1e-8), "A vectors differ!"
        assert np.allclose(B_old, B_new, atol=1e-8), "B vectors differ!"
        print("Test passed: compute_A_and_B_OLD and compute_A_and_B are equivalent.")

    # Uncomment to run the test
    test_equivalence_of_A_and_B()