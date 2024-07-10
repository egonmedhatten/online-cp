import numpy as np
import time
import warnings
from scipy.spatial.distance import pdist, cdist, squareform

MACHINE_EPSILON = np.finfo(np.float64).eps

class ConformalPredictiveSystem:
    '''
    Parent class for conformal predictive systems. Unclear if some methods are common to all, so perhaps we don't need it.
    '''


class NearestNeighboursPredictionMachine(ConformalPredictiveSystem):

    def __init__(self, k, distance='euclidean', distance_func=None, warnings=True, verbose=0, rnd_state=2024):
        
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

    
    def predict_cpd(self, x, epsilon=0.1):

        '''
        TODO Add possibility to return precomputed as we did with ConformalRegressor.
        '''
        
        # Temporarily update the distance matrix
        if self.X is None:
            X = x.reshape(1,-1)
            D = self.distance_func(self.X)
            y = np.array(-np.inf) # Initialise label as -inf
        else:
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            y = np.append(self.y, -np.inf) # Initialise label as -inf

        # Find all neighbours and semi-neighbours
        k_nearest = D.argsort(axis=0)[1:self.k+1]

        idx_all_neighbours_and_semi_neighbours = []

        full_neighbours = []
        single_neighbours = []
        semi_neighbours = []

        n = D.shape[0] - 1
        k_nearest_of_n = k_nearest.T[n]
        for i, col in enumerate(k_nearest.T):
            if i in k_nearest_of_n and n in col:
                # print(f'{i} is a full neighbour')
                idx_all_neighbours_and_semi_neighbours.append(i)
                full_neighbours.append(y[i])
            if i in k_nearest_of_n and n not in col:
                # print(f'{i} is a single neighbour')
                idx_all_neighbours_and_semi_neighbours.append(i)
                single_neighbours.append(y[i])
            if i not in k_nearest_of_n and n in col:
                # print(f'{i} is a semi-neighbour')
                idx_all_neighbours_and_semi_neighbours.append(i)
                semi_neighbours.append(y[i])
        
        # Line 1
        Kprime = len(idx_all_neighbours_and_semi_neighbours)

        # Line 2 and 3
        Y = np.zeros(shape=(Kprime+2,))
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = y[idx_all_neighbours_and_semi_neighbours]
        Y.sort()

        # Line 4
        Alpha = -np.inf * np.ones(n + 1) # Initialize at something unreasonable
        N = -np.inf * np.ones(self.k + 1) # Initialize at something unreasonable
        for i in range(n + 1):
            J = k_nearest.T[i]
            Alpha[i] = np.where(y[J] <= y[i])[0].shape[0]
        for k in range(self.k + 1):
            N[k] = np.where(Alpha == k)[0].shape[0]
        
        # Line 5
        L = -np.inf * np.ones(Kprime+1) # Initialize at something unreasonable
        U = -np.inf * np.ones(Kprime+1) # Initialize at something unreasonable
        L[0] = 0
        U[0] = N[0]/(n+1)

        # Line 6
        for k in range(1, Kprime + 1):
            # FIXME Something is wrong with this loop... Very difficult to tell what.
            # Line 7
            
            if Y[k] in full_neighbours or Y[k] in single_neighbours:
                # Line 8
                N[int(Alpha[-1])] -= 1
                Alpha[n] += 1
                N[int(Alpha[-1])] += 1

            # Line 9
            if Y[k] in full_neighbours or Y[k] in semi_neighbours:
                # Line 10
                N[int(Alpha[k])] -= 1
                Alpha[k] -= 1
                N[int(Alpha[k])] += 1
            
            # Line 11
            L[k] = N[:int(Alpha[-1])].sum() / (n + 1) if Alpha[-1] != 0  else 0
            U[k] = N[:int(Alpha[-1]) + 1].sum() / (n + 1)
        
        # Line 12
        cps = KnnConformalPredictiveDistributionFunction(L, U, Y)
        return cps
    

class ConformalPredictiveDistributionFunction:

    '''
    NOTE
    The CPD contains all the information needed to form a
    prediction set. We can take quantiles and so on.
    '''

    def quantile(self):
        pass

    def predict_set(self):
        pass


class KnnConformalPredictiveDistributionFunction(ConformalPredictiveDistributionFunction):

    def __init__(self, L, U, Y):
        self.L = L 
        self.U = U
        self.Y = Y

    def __call__(self, y, tau=None):
        Y = self.Y[:-1]
        idx_eq = np.where(y == Y)[0]
        if idx_eq.shape[0] > 0:
            print('Here')
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
            return tau * Pi0 + (1 - tau) * Pi1
        
    
