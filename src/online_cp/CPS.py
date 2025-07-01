import numpy as np
import time
import warnings
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize, Bounds
MACHINE_EPSILON = lambda x: np.abs(x) * np.finfo(np.float64).eps


default_epsilon = 0.1


def get_ConformalPredictionInterval():
    from .regressors import ConformalPredictionInterval  # Lazy import
    return ConformalPredictionInterval

class ConformalPredictiveSystem:
    '''
    Parent class for conformal predictive systems. Unclear if some methods are common to all, so perhaps we don't need it.
    '''

    def __init__(self, epsilon=default_epsilon):
        self.epsilon = epsilon

    def learn_many(self, X, y):
        for x1, y1 in zip(X, y):
            self.learn_one(x1,y1)

class RidgePredictionMachine(ConformalPredictiveSystem):
    'This conformal predictive system uses the "studentised residuals as conformity measure'

    def __init__(self, a=0, warnings=True, autotune=False, verbose=0, rnd_state=None, epsilon=default_epsilon):
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
        >>> cps = RidgePredictionMachine()
        >>> cps.learn_one(np.array([1,0]), 1)
        >>> cps.X
        array([[1, 0]])
        >>> cps.y
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


    def change_ridge_parameter(self, a):
        '''
        Change the ridge parameter
        >>> cps = RidgePredictionMachine()
        >>> cps.learn_one(np.array([1,0]), 1)
        >>> cps.change_ridge_parameter(1)
        >>> cps.a
        1
        '''
        self.a = a
        if self.X is not None:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a * self.Id)


    def check_matrix_rank(self, M):
        '''
        Check if a matrix has full rank <==> is invertible
        Returns False if matrix is rank deficient
        NOTE In numerical linear algebra it is a bit more subtle. The condition number can tell us more.

        >>> cps = RidgePredictionMachine(warnings=False)
        >>> cps.check_matrix_rank(np.array([[1, 0], [1, 0]]))
        False
        >>> cps.check_matrix_rank(np.array([[1, 0], [0, 1]]))
        True
        '''
        if np.linalg.matrix_rank(M) < M.shape[0]:
            if self.warnings:
                warnings.warn(f'The matrix X is rank deficient. Condition number: {np.linalg.cond(M)}. Consider changing the ridge prarmeter')
            return False
        else:
            return True


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
                max_diag_H = np.max(np.diag(A))  # Maximum diagonal element of the hat matrix
                if max_diag_H > 1:
                    return np.inf 
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


    def predict_cpd(self, x, return_update=False, save_time=False):
        def build_precomputed(X, XTXinv, C):
            computed = {
                'X': X, # The updated matrix of objects
                'XTXinv': XTXinv, # The updated kernel matrix
                'C': C
            } 
            return computed

        tic = time.time()
        # Add row to X matrix
        X = np.append(self.X, x.reshape(1, -1), axis=0)
        toc_add_row = time.time() - tic
        n = X.shape[0]
        y = self.y

        tic = time.time()
        # Update XTX_inv (inverse of Kernel matrix plus regularisation) Use the Sherman-Morrison formula to update the hat matrix
                #https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

        XTXinv = self.XTXinv - (self.XTXinv @ np.outer(x, x) @ self.XTXinv) / (1 + x.T @ self.XTXinv @ x)
        toc_update_XTXinv = time.time() - tic

        tic = time.time()

        H = X @ XTXinv @ X.T
        h = H.diagonal()
        sqrt_one_minus_h = np.sqrt(1 - h[:-1])
        A = np.dot(H[-1, :-1], y) / np.sqrt(1 - h[-1])  + (y - H[:-1, :-1] @ y) / sqrt_one_minus_h
        B = np.sqrt(1 - h[-1]) * np.ones(n-1) + H[-1, :-1] / sqrt_one_minus_h
        C = np.zeros(n + 1)
        C[1:-1] = A / B
        C[0] = -np.inf
        C[-1] = np.inf
        C.sort()


        # NOTE: Keep the loop version for a bit, if we run into problems...
        # C = np.empty(n+1)
        # for i in range(n - 1):
        #     a_i = sum([H[j, -1] * y[j] for j in range(n-1)]) / np.sqrt(1 - h[-1]) + (y[i] - sum([H[i, j] * y[j] for j in range(n-1)])) / np.sqrt(1 - h[i])
        #     b_i = np.sqrt(1 - h[-1]) + H[i, -1] / np.sqrt(1 - h[i])
        #     C[i+1] = a_i / b_i
        # C[0] = -np.inf
        # C[-1] = np.inf
        # C.sort()

        toc_compute_C = time.time() - tic

        time_dict = {
            'Update hat matrix': toc_update_XTXinv,
            'Compute C': toc_compute_C
        }
        time_dict = time_dict if save_time else None
        cpd = RidgePredictiveDistributionFunction(C=C, time_dict=time_dict, epsilon=self.epsilon)

        if return_update:
            return cpd, build_precomputed(X, XTXinv, C)
        else:
            return cpd
    

class KernelRidgePredictionMachine(ConformalPredictiveSystem):
    '''
        This conformal predictive system uses the "studentised residuals as conformity measure.
        Algorithm 7,3 in Algorithmic Learning in a Random World 2nd edition.
    '''

    def __init__(self, kernel, a=0, warnings=True, autotune=False, verbose=0, rnd_state=None, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)

        self.kernel = kernel

        self.a = a
        self.X = None
        self.y = None
        

        # Should we raise warnings
        self.warnings = warnings
        # Do we autotune ridge prarmeter on warning
        self.autotune = autotune

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)

    def learn_initial_training_set(self, X, y):
        self.X = X
        self.y = y
        Id = np.identity(self.X.shape[0])
        self.K = self.kernel(self.X)
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
        self.H = self.K @ self.Kinv

    def _tune_ridge_parameter(self, a0=None):
        '''
        Tune ridge parameter with Generalized Cross Validation (GCV) in the kernel space.
        '''
        n = self.K.shape[0]
        In = np.identity(n)

        def GCV(a):
            try:
                A = self.K @ np.linalg.inv(self.K + a * In)
                max_diag_H = np.max(np.diag(A))  # Maximum diagonal element of the hat matrix
                if max_diag_H > 1:
                    return np.inf 
                return (1 / n) * np.linalg.norm((In - A) @ self.y) ** 2 / ((1 / n) * np.trace(In - A)) ** 2
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return np.inf

        # Initial guess
        if a0 is None:
            a0 = 1e-6  # Small perturbation to avoid numerical issues

        # Bounds to ensure a >= 0
        res = minimize(GCV, x0=a0, bounds=Bounds(lb=1e-6, keep_feasible=True))
        a = res.x[0]

        if self.verbose > 0:
            print(f'New ridge parameter: {a}')
        self.change_ridge_parameter(a)

    def change_ridge_parameter(self, a):
        '''
        Change the ridge parameter
        >>> cps = RidgePredictionMachine()
        >>> cps.learn_one(np.array([1,0]), 1)
        >>> cps.change_ridge_parameter(1)
        >>> cps.a
        1
        '''
        self.a = a
        if self.X is not None:
            Id = np.identity(self.X.shape[0])
       
            self.K = self.kernel(self.X)
            self.Kinv = np.linalg.inv(self.K + self.a * Id)
            self.H = self.K @ self.Kinv

    def _update_Kinv(self, Kinv, k, d):
        # print(f'K: {K}')
        # print(f'k: {k}')
        # print(f'kappa: {kappa}')
        return np.block([[Kinv + d * Kinv @ k @ k.T @ Kinv, -d * Kinv @ k], [ -d * k.T @ Kinv, d]])


    @staticmethod
    def _update_K(K, k, kappa):
        # print(f'K: {K}')
        # print(f'k: {d}')
        # print(f'kappa: {kappa}')
        return np.block([[K, k], [k.T, kappa]])


    def learn_one(self, x, y, precomputed=None):
        '''
        Learn a single example
        '''

        # # TEST DUMB UPDATE
        # self.y = np.append(self.y, y)
        # self.X = np.append(self.X, x.reshape(1, -1), axis=0)
        # Id = np.identity(self.X.shape[0])
       
        # self.K = self.kernel(self.X)
        # self.Kinv = np.linalg.inv(self.K + self.a * Id)
        # self.H = self.K @ self.Kinv

        x = np.atleast_2d(x)
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)
        
        if precomputed is not None:
            self.H = precomputed['H']
            self.K = self._update_K(self.K, precomputed['k'], precomputed['kappa'])
            self.Kinv = self._update_Kinv(self.Kinv, precomputed['k'], precomputed['d'])
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)
        else:
            if self.X is None:
                self.X = x.reshape(1,-1)
                Id = np.identity(self.X.shape[0])
                self.K = self.kernel(self.X)
                self.Kinv = np.linalg.inv(self.K + self.a * Id)
                self.H = self.K @ self.Kinv
            else:
                k = self.kernel(self.X, x).reshape(-1, 1)
                kappa = self.kernel(x, x)
                d = 1 / (kappa + self.a - k.T @ self.Kinv @ k)
                y = self.y
                self.H = np.block([
                    [
                        self.H - self.a * d * self.Kinv @ k @ k.T @ self.Kinv, 
                        self.a * d * self.Kinv @ k
                    ], 
                    [
                        self.a * d * k.T @ self.Kinv, 
                        d * kappa - d*k.T @ self.Kinv @ k
                        ]
                    ]
                    )
                self.K = self._update_K(self.K, k, kappa)
                self.Kinv = self._update_Kinv(self.Kinv, k, d)
                self.X = np.append(self.X, x.reshape(1, -1), axis=0)
    
    def predict_cpd(self, x, return_update=False, save_time=False):
        
        def build_precomputed(H, k, kappa, d):
            computed = {
                'H': H,
                'k': k,
                'kappa': kappa,
                'd': d,
            } 
            return computed
        x = np.atleast_2d(x)        
        # Temporarily update kernel matrix
        k = self.kernel(self.X, x).reshape(-1, 1)
        kappa = self.kernel(x, x)
        d = 1 / (kappa + self.a - k.T @ self.Kinv @ k)
        y = self.y
        H = np.block([
            [
                self.H - self.a * d * self.Kinv @ k @ k.T @ self.Kinv, 
                self.a * d * self.Kinv @ k
             ], 
             [
                self.a * d * k.T @ self.Kinv, 
                d * kappa - d*k.T @ self.Kinv @ k
                ]
            ]
            )
        
        n = H.shape[0]

        h = H.diagonal()
        sqrt_one_minus_h = np.sqrt(1 - h[:-1])
        A = np.dot(H[-1, :-1], y) / np.sqrt(1 - h[-1])  + (y - H[:-1, :-1] @ y) / sqrt_one_minus_h
        B = np.sqrt(1 - h[-1]) * np.ones(n-1) + H[-1, :-1] / sqrt_one_minus_h

        
        C = np.zeros(n + 1)
        C[1:-1] = A / B
        C[0] = -np.inf
        C[-1] = np.inf
        assert not np.isnan(C).any(), "C contains NaN values"
        C.sort()

        time_dict = None
        cpd = RidgePredictiveDistributionFunction(C=C, time_dict=time_dict, epsilon=self.epsilon)

        if return_update:
            return cpd, build_precomputed(H, k, kappa, d)
        else:
            return cpd

class NearestNeighboursPredictionMachine(ConformalPredictiveSystem):

    def __init__(self, k, distance='euclidean', distance_func=None, warnings=True, verbose=0, rnd_state=None, epsilon=default_epsilon):
        '''
        Consider adding possibility to update self.k as the training set grows, e.g. by some heuristic or something.
        Two rules of thumb are quite simple:
            1. Choose k close to sqrt(n) where n is the training set size
            2. If the data has large variance, choose k larger. If the variance is small, choose k smaller. This is less clear, however.
        '''
        # TODO: The sorting of conformity scores is the most time consuming step. Can it be done with parallel processing to speed things up?
        super().__init__(epsilon=epsilon)

        self.k = k

        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = 'custom'

        self.X = None
        self.y = None
        self.D = None

        # Should we raise warnings
        self.warnings = warnings

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)


    def _standard_distance_func(self, X, y=None):
        '''
        By default we use scipy to compute distances
        '''
        # TODO: Can this be done using parallel processing? 
        X = np.atleast_2d(X)
        if y is None:
            dists = squareform(pdist(X, metric=self.distance))
        else:
            y = np.atleast_2d(y)
            dists = cdist(X, y, metric=self.distance)
        return dists
    

    def learn_initial_training_set(self, X, y):
        '''
        The Nearest neighbours prediction machine assumes all labels are unique. If they are not, we add noise to break ties.

        >>> cps = NearestNeighboursPredictionMachine(k=3)
        >>> X = np.array([[1], [2]])
        >>> y = np.array([1, 2])
        >>> cps.learn_initial_training_set(X, y)
        >>> cps.X
        array([[1],
               [2]])
        >>> cps.y
        array([1, 2])
        >>> cps.D
        array([[0., 1.],
               [1., 0.]])
        '''
        # FIXME: It also assumes all distances are unique. Figure out how to handle this
        self.X = X
        self.D = self.distance_func(X)

        self.y = y
        
        if np.unique(self.y).size != self.y.size:
            raise Exception('All labels y must be distinct for the NearestNeighboursPredictionMachine to be valid')

    
    @staticmethod
    def update_distance_matrix(D, d):
        return np.block([[D, d], [d.T, np.array([0])]])
    

    def learn_one(self, x, y, precomputed=None):
        '''
        The Nearest neighbours prediction machine assumes all labels are unique. If they are not, we add noise to break ties.
        precomputed is a dictionary
        {
            'X': X,
            'D': D,
        }
        >>> cps = NearestNeighboursPredictionMachine(k=3, rnd_state=2024)
        >>> X = np.array([[1], [2]])
        >>> y = np.array([1, 2])
        >>> cps.learn_initial_training_set(X, y)
        >>> cps.learn_one(np.array([3]), 3)
        >>> cps.y
        array([1, 2, 3])
        >>> cps.X
        array([[1],
               [2],
               [3]])
        >>> cps.D
        array([[0., 1., 2.],
               [1., 0., 1.],
               [2., 1., 0.]])
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

        if np.unique(self.y).size != self.y.size:
            raise Exception('All labels y must be distinct for the NearestNeighboursPredictionMachine to be valid')

    
    def predict_cpd(self, x, return_update=False, save_time=False):
        '''
        >>> import numpy as np
        >>> rnd_gen = np.random.default_rng(2024)
        >>> X = rnd_gen.normal(loc=0, scale=1, size=(100, 4))
        >>> beta = np.array([2, 1, 0, 0])
        >>> Y = X @ beta + rnd_gen.normal(loc=0, scale=1, size=100)
        >>> cps = NearestNeighboursPredictionMachine(k=3)
        >>> cps.learn_initial_training_set(X, Y)
        >>> x = rnd_gen.normal(loc=0, scale=1, size=(1, 4))
        >>> cpd = cps.predict_cpd(x)
        >>> cpd.L
        array([0.        , 0.        , 0.18811881, 0.53465347, 0.76237624])
        >>> cpd.U
        array([0.17821782, 0.18811881, 0.54455446, 0.76237624, 1.        ])
        '''
        tic = time.time()
        # Temporarily update the distance matrix
        if self.X.shape[0] < self.k:
            # FIXME: Make some graceful error handling here
            raise Exception('Training set is too small...')
        else:
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            y = np.append(self.y, -np.inf) # Initialise label as -inf
        toc_dist = time.time()-tic

        tic = time.time()
        # Find all neighbours and semi-neighbours
        # NOTE: This is the time consuming step. The distance matrix has to be sorted. Is there any way to speed this up?
        k_nearest = D.argsort(axis=0)[1:self.k+1]
        toc_sort = time.time() - tic

        tic = time.time()
        n = self.X.shape[0]

        full_neighbours = []
        single_neighbours = []
        semi_neighbours = []
        idx_all_neighbours_and_semi_neighbours = []

        k_nearest_of_n = k_nearest.T[-1]

        # FIXME: How do we save the full, single and semi-neighbours so that we can acess them later in a nice way?
        for i, col in enumerate(k_nearest.T):
            if i in k_nearest_of_n and n in col:
                # print(f'z_{i} is a full neighbour')
                # idx_all_neighbours_and_semi_neighbours.append(i)
                full_neighbours.append(i)
            if i in k_nearest_of_n and not n in col:
                # print(f'z_{i} is a single neighbour')
                # idx_all_neighbours_and_semi_neighbours.append(i)
                single_neighbours.append(i)
            if not i in k_nearest_of_n and n in col:
                # print(f'z_{i} is a semi-neighbour')
                # idx_all_neighbours_and_semi_neighbours.append(i)
                semi_neighbours.append(i)
        idx_all_neighbours_and_semi_neighbours = np.array(full_neighbours + single_neighbours + semi_neighbours)
        toc_find_neighbours = time.time() - tic
        
        # Line 1
        Kprime = len(idx_all_neighbours_and_semi_neighbours)
        # Line 2 and 3
        Y = np.zeros(shape=Kprime + 2)
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = y[idx_all_neighbours_and_semi_neighbours]
        idx_mem = {i: idx_all_neighbours_and_semi_neighbours[i-1] for i in range(1, Kprime+1)}
        sorted_indices = np.argsort(Y)[1:-1]
        # print(f'idx_mem: {idx_mem}')
        # print(f'idx_all_neighbours_and_semi_neighbours: {idx_all_neighbours_and_semi_neighbours}')
        # print(f'sorted_indices: {sorted_indices}')
        Y.sort()
        # print(f'Y: {Y}')

        # Line 4
        Alpha = np.array([(y[k_nearest.T[i]] <= y_i).sum() for i, y_i in enumerate(y)])
        # FIXME: Based on the description in ALRW, alpha_n = 0 initially, which seems to imply that 
        #        they consider -inf <= -inf to be false. Or possibly it is a consequence of assuming
        #        all labels and distances are distinct...
        N = np.array([(Alpha == k).sum() for k in range(self.k+1)])

        # Line 5
        L = -np.inf * np.ones(Kprime+1) # Initialize at something unreasonable
        U = -np.inf * np.ones(Kprime+1) # Initialize at something unreasonable
        L[0] = 0
        U[0] = N[0]/(n+1)

        # if Alpha[-1] > 0:
        #     print(f'n: {D.shape[0]}')
        #     print(f'Alpha: {Alpha}')
        #     print(f'N: {N}')
        #     print(f'y: {y}')
        #     print(f'{[k_nearest.T[i] for i, y_i in enumerate(y)]}')
        #     print(f'k_nearest: {k_nearest}')
            # print(f'Kprime: {Kprime}')

        # print(f'Kprime: {Kprime}')

        tic = time.time()
        # Line 6
        for k in range(1, Kprime+1):
            idx = idx_mem[sorted_indices[k-1]]
            # print(f'idx: {idx}')
            if (idx in full_neighbours + single_neighbours):
                # print(f'{idx} is a full or a single neighbour')
                N[Alpha[-1]] -= 1
                Alpha[-1] += 1
                N[Alpha[-1]] += 1
            if (idx in full_neighbours + semi_neighbours):
                # print(f'{idx} is a full or a semi-neighbour')
                N[Alpha[idx]] -= 1
                Alpha[idx] -= 1
                N[Alpha[idx]] += 1
            L[k] = N[:Alpha[-1]].sum() / (n+1) if Alpha[-1] != 0  else 0
            U[k] = N[:Alpha[-1] + 1].sum() / (n+1) if Alpha[-1] != 0  else N[0] / (n+1)
            # print(f'Alpha: {Alpha}')
            # print(f'Alpha_n: {Alpha[-1]}')
            # print(f'L[k]: {L[k]}')
            # print(f'N: {N}')
        # print(f'full_neighbours: {full_neighbours}')
        # print(f'single neighbours: {single_neighbours}')
        # print(f'semi_neighbours: {semi_neighbours}')
        toc_loop = time.time() - tic

        time_dict = {
            'Compute distance matrix': toc_dist,
            'Sort distance matrix': toc_sort,
            'Find all neighbours and semi-neighbours': toc_find_neighbours,
            'Loop': toc_loop
        }
        time_dict = time_dict if save_time else None
        # Line 12
        cpd = NearestNeighboursPredictiveDistributionFunction(L, U, Y, time_dict, epsilon=self.epsilon)

        if return_update:
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            return cpd, {'X': X, 'D': D}
        else:
            return cpd
    

class DempsterHillConformalPredictiveSystem(ConformalPredictiveSystem):

    def __init__(self, verbose=0, rnd_state=None, epsilon=default_epsilon):
        '''
        The Dempster-Hill conformal predictive system uses only the labels of the examples, so the latter can be ignored.
        '''
        super().__init__(epsilon=epsilon)
        self.y = None
        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)
    
    def learn_initial_training_set(self, y):
        self.y = y

    def learn_one(self, y):
        self.y = np.append(self.y, y)

    def predict_cpd(self, save_time=False):
        tic = time.time()
        Y = np.zeros(shape=self.y.shape[0] + 2)
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = self.y
        Y.sort()
        time_sort = time.time() - tic

        time_dict = {'Sort labels': time_sort} if save_time else None

        return DempsterHillConformalPredictiveDistribution(Y, time_dict, epsilon=self.epsilon)


class ConformalPredictiveDistributionFunction:

    '''
    NOTE
    The CPD contains all the information needed to form a
    prediction set. We can take quantiles and so on.
    '''

    def __init__(self, epsilon=default_epsilon):
        self.epsilon=epsilon

    def quantile(self, p, tau=None):
        raise NotImplementedError('Parent class has no quantile function')


    def predict_set(self, tau, epsilon=None, bounds='both', minimise_width=False):
        '''
        The convex hull of the epsilon/2 and 1-epsilon/2 quantiles make up
        the prediction set Gamma(epsilon)
        '''
        if epsilon is None:
            epsilon = self.epsilon

        if minimise_width:
            if bounds != 'both':
                raise Exception('bounds must be "both" if we are to minimise the interval width')
            def compute_width(delta, epsilon):
                epsilon_minus_delta = epsilon - delta
                return self.quantile(1-epsilon_minus_delta, tau) - self.quantile(delta, tau)

            delta = minimize_scalar(lambda x: compute_width(x, epsilon), bounds=(0, epsilon)).x
            eps_minus_delta = epsilon - delta
            lower = self.quantile(delta, tau)
            upper = self.quantile(1-eps_minus_delta, tau)

        
        else:
            q1 = epsilon/2
            q2 = 1 - epsilon/2
            if bounds=='both':
                lower = self.quantile(q1, tau)
                upper = self.quantile(q2, tau)
            elif bounds=='lower':
                lower = self.quantile(q1, tau)
                upper = np.inf
            elif bounds=='upper':
                lower = -np.inf
                upper = self.quantile(q2, tau)
            else: 
                raise Exception

        # print(f'Lower: {lower}')
        # print(f'Upper: {upper}')
        CP_int = get_ConformalPredictionInterval()
        return CP_int(lower, upper, epsilon)
    
    def find_smallest_epsilon(self, tau, increment=0.001):
        '''
        Find the smallest epsilon such that the prediction set is finite
        '''
        epsilon = 0
        while True:
            prediction_set = self.predict_set(tau=tau, epsilon=epsilon)
            if np.isfinite(prediction_set.width()):
                return epsilon
            epsilon += increment  # Increment epsilon by a small amount

    # These methods relate to when the cpd is used to predict sets
    @staticmethod
    def err(Gamma, y):
        return int(not(y in Gamma))
    

    @staticmethod
    def width(Gamma):
        return Gamma.width()
    

class RidgePredictiveDistributionFunction(ConformalPredictiveDistributionFunction):

    def __init__(self, C, time_dict=None, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.C = C
        self.L = np.array([self.__call__(y, 0) for y in self.C])
        self.U = np.array([self.__call__(y, 1) for y in self.C])
        self.Y = C

        self.time_dict = time_dict

        # What about
        self.y_vals = np.array(sorted([-np.inf, np.inf] + self.C[1: -1].tolist() + (self.C[1: -1] + MACHINE_EPSILON(self.C[1: -1])).tolist() + (self.C[1: -1] - MACHINE_EPSILON(self.C[1: -1])).tolist()))
        self.lowers = np.array([self.__call__(y, 0) for y in self.y_vals])
        self.uppers = np.array([self.__call__(y, 1) for y in self.y_vals])
        # Then the quantile can be computed by
        # self.y_vals[np.where((1 - tau) * self.L + tau * self.U >= p)[0].min()]

    def __call__(self, y, tau=None):
        if y == -np.inf:
            Pi0, Pi1 = 0.0, 0.0
        elif y == np.inf:
            Pi0, Pi1 = 1.0, 1.0
        else:
            C = self.C[:-1]
            idx_eq = np.where(y == C)[0]
            if idx_eq.shape[0] > 0:
                i_prime = idx_eq.min()
                i_bis = idx_eq.max()
                interval = ((i_prime - 1) / C.shape[0], (i_bis + 1) / C.shape[0])
            else:
                i = np.where(C <= y)[0].max()
                interval = (i / C.shape[0], (i + 1) / C.shape[0])

            Pi0 = interval[0]
            Pi1 = interval[1]            

        if tau is None:
            return Pi0, Pi1
        else:
            return (1 - tau) * Pi0 + tau * Pi1
        
    
    # NOTE: This takes forever if we have a large training set. 
    # Why not just invert?
    def quantile(self, p, tau=None):
        def compute_quantile(p, tau):
            # q = np.inf
            # y_vals = np.array(sorted([-np.inf, np.inf] + self.C[1: -1].tolist() + (self.C[1: -1] + MACHINE_EPSILON(self.C[1: -1])).tolist() + (self.C[1: -1] - MACHINE_EPSILON(self.C[1: -1])).tolist()))
            # # This loop is not very nice. Can we get rid of it?
            # for y in y_vals[::-1]:
            #     if self.__call__(y, tau) >= p:
            #         q = y
            #     else:
            #         return q
            # return q
            q = self.y_vals[np.where((1 - tau) * self.lowers + tau * self.uppers >= p)[0].min()]
            return q
        
        if tau is not None:
            q = compute_quantile(p, tau)
            return q
        else:
            q0 = compute_quantile(p, 0)
            q1 = compute_quantile(p, 1)
            return q0, q1
    
        
    def plot(self, tau=None):
        if tau is None:
            fig, ax = plt.subplots()
            ax.step(self.C, self.L, label=r'$\Pi(y, 0)$')
            ax.step(self.C, self.U, label=r'$\Pi(y, 1)$')
            ax.fill_between(self.C, self.L, self.U, step='pre', alpha=0.5, color='green')
            ax.set_ylabel('cumulative probability')
            ax.set_xlabel(r'$y$')
            ax.legend()
        else: 
            fig, ax = plt.subplots()
            ax.step(self.C, (1 - tau) * self.L + tau * self.U, label=r'$\Pi(y, \tau)$')
            ax.set_ylabel('cumulative probability')
            ax.set_xlabel(r'$y$')
            ax.legend()

        plt.close(fig)  # Prevent implicit display
        return fig


class NearestNeighboursPredictiveDistributionFunction(ConformalPredictiveDistributionFunction):
    '''
    TODO: Write tests
    '''
    # NOTE: It would be possible to pass a protection function here...
    def __init__(self, L, U, Y, time_dict=None, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.L = L 
        self.U = U
        self.Y = Y

        self.time_dict = time_dict
        
        Y_temp = Y[np.isfinite(Y)]
        self.y_vals = np.array(
            sorted(
                [-np.inf, np.inf] + 
                Y_temp.tolist() + 
                (Y_temp + MACHINE_EPSILON(Y_temp)).tolist() + 
                (Y_temp - MACHINE_EPSILON(Y_temp)).tolist())
        )
        self.lowers = np.array([self.__call__(y, 0) for y in self.y_vals])
        self.uppers = np.array([self.__call__(y, 1) for y in self.y_vals])

    def __call__(self, y, tau=None):
        # TODO: Check carefully that this is correct
        if y == self.Y[0]:
            Pi0, Pi1 = 0.0, 0.0
        elif y == self.Y[-1]:
            Pi0, Pi1 = 1.0, 1.0
        else:
            Y = self.Y[:-1]
            idx_eq = np.where(y == Y)[0]
            if idx_eq.shape[0] > 0:
                k = idx_eq.min()
                interval = (self.L[k-1], self.U[k])
            else:
                k = np.where(Y <= y)[0].max()
                interval = (self.L[k], self.U[k])
            
            Pi0 = interval[0]
            Pi1 = interval[1]            

        if tau is None:
            return Pi0, Pi1
        else:
            return (1 - tau) * Pi0 + tau * Pi1
        
    
    def quantile(self, p, tau=None):
        def compute_quantile(p, tau):
            # q = np.inf
            # y_vals = np.array(sorted([-np.inf, np.inf] + self.Y[1: -1].tolist() + (self.Y[1: -1] + MACHINE_EPSILON(self.Y[1: -1])).tolist() + (self.Y[1: -1] - MACHINE_EPSILON(self.Y[1: -1])).tolist()))
            # for y in y_vals[::-1]:
            #     if self.__call__(y, tau) >= p:
            #         q = y
            #     else:
            #         return q
            q = self.y_vals[np.where((1 - tau) * self.lowers + tau * self.uppers >= p)[0].min()]
            return q
        if tau is not None:
            q = compute_quantile(p, tau)
            return q
        else:
            q0 = compute_quantile(p, 0)
            q1 = compute_quantile(p, 1)
            return q0, q1
    
    def plot(self, tau=None):
        if tau is None:
            fig, ax = plt.subplots(sharex=True)
            ax.step(self.Y[1:], self.L, label=r'$\Pi(y, 0)$')
            ax.step(self.Y[1:], self.U, label=r'$\Pi(y, 1)$')
            ax.fill_between(self.Y[1:], self.L, self.U, step='pre', alpha=0.5, color='green')
            ax.legend()
        else: 
            fig, ax = plt.subplots()
            ax.step(self.Y[1:], (1 - tau) * self.L + tau * self.U, label=r'$\Pi(y, \tau)$')
            ax.legend()
        fig.tight_layout()
        plt.close(fig)  # Prevent implicit display
        return fig


class DempsterHillConformalPredictiveDistribution(ConformalPredictiveDistributionFunction):

    def __init__(self, Y, time_dict=None, epsilon=default_epsilon):
        super().__init__(epsilon=epsilon)
        self.Y = Y
        self.time_dict = time_dict

        self.y_vals = np.array(sorted([-np.inf, np.inf] + self.Y[1: -1].tolist() + (self.Y[1: -1] + MACHINE_EPSILON(self.Y[1: -1])).tolist() + (self.Y[1: -1] - MACHINE_EPSILON(self.Y[1: -1])).tolist()))
        self.lowers = np.array([self.__call__(y, 0) for y in self.y_vals])
        self.uppers = np.array([self.__call__(y, 1) for y in self.y_vals])

    
    def __call__(self, y, tau=None):
        if y == self.Y[0]:
            Pi0, Pi1 = 0.0, 0.0
        elif y == self.Y[-1]:
            Pi0, Pi1 = 1.0, 1.0
        else:
            Y = self.Y[:-1]
            idx_eq = np.where(y == Y)[0]
            if idx_eq.shape[0] > 0:
                k = idx_eq.min()
                interval = ((k-1)/(Y.shape[0]), (k+1)/(Y.shape[0]))
            else:
                k = np.where(Y <= y)[0].max()
                interval = ((k)/(Y.shape[0]), (k+1)/(Y.shape[0]))
            
            Pi0 = interval[0]
            Pi1 = interval[1]

        if tau is None:
            return Pi0, Pi1
        else:
            return (1 - tau) * Pi0 + tau * Pi1
        
    def quantile(self, p, tau=None):
        def compute_quantile(p, tau):
            # q = np.inf
            # y_vals = np.array(sorted([-np.inf, np.inf] + self.Y[1: -1].tolist() + (self.Y[1: -1] + MACHINE_EPSILON(self.Y[1: -1])).tolist() + (self.Y[1: -1] - MACHINE_EPSILON(self.Y[1: -1])).tolist()))
            # for y in y_vals[::-1]:
            #     if self.__call__(y, tau) >= p:
            #         q = y
            #     else:
            #         return q
            q = self.y_vals[np.where((1 - tau) * self.lowers + tau * self.uppers >= p)[0].min()]
            return q
        if tau is not None:
            q = compute_quantile(p, tau)
            return q
        else:
            q0 = compute_quantile(p, 0)
            q1 = compute_quantile(p, 1)
            return q0, q1

if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
