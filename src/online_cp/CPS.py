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

    def __init__(self, k, distance='euclidean', distance_func=None, warnings=True, verbose=0, rnd_state=None):
        '''
        Consider adding possibility to update self.k as the training set grows, e.g. by some heuristic or something.
        Two rules of thumb are quite simple:
            1. Choose k close to sqrt(n) where n is the training set size
            2. If the data has large variance, choose k larger. If the variance is small, choose k smaller. This is less clear, however.
        '''
        
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
    

    def learn_initial_training_set(self, X, y, noise_range=1e-6):
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

        # # Ensure all labels are unique
        # y = y
        # if np.unique(y).shape[0] < y.shape[0] and self.verbose > 1:
        #     print('Duplicate labels detected. Breaking ties with random noise.')
        # while np.unique(y).shape[0] < y.shape[0]:
        #     # Find duplicates
        #     unique, counts = np.unique(y, return_counts=True)
        #     duplicates = unique[counts > 1]

        #     # Add noise to duplicate entries
        #     for label in duplicates:
        #         indices = np.where(y == label)[0]
        #         noise = self.rnd_gen.uniform(low=-noise_range, high=noise_range, size=len(indices))
        #         y[indices] += noise
        self.y = y

    
    @staticmethod
    def update_distance_matrix(D, d):
        return np.block([[D, d], [d.T, np.array([0])]])
    

    def learn_one(self, x, y, precomputed=None, noise_range=1e-6):
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
        >>> cps.learn_one(np.array([3]), 1)
        >>> cps.y
        array([1, 2, 1])
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
            # if y in self.y and self.verbose > 1:
            #     print('Duplicate label. Breaking tie with random noise.')
            # while y in self.y:
            #     y = y + self.rnd_gen.uniform(low=-noise_range, high=noise_range)
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
        N = np.array([(Alpha == k).sum() for k in range(self.k+1)])

        # Line 5
        L = -np.inf * np.ones(Kprime+1) # Initialize at something unreasonable
        U = -np.inf * np.ones(Kprime+1) # Initialize at something unreasonable
        L[0] = 0
        U[0] = N[0]/(n+1)

        # print(f'Alpha: {Alpha}')
        # print(f'N: {N}')

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
        cps = NearestNeighboursPredictiveDistributionFunction(L, U, Y, time_dict)

        if return_update:
            X = np.append(self.X, x.reshape(1, -1), axis=0)
            return cps, {'X': X, 'D': D}
        else:
            return cps
    

class DempsterHillConformalPredictiveSystem(ConformalPredictiveSystem):

    def __init__(self, verbose=0, rnd_state=None):
        '''
        The Dempster-Hill conformal predictive system uses only the labels of the examples, so the latter can be ignored.
        '''
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

        return DempsterHillConformalPredictiveDistribution(Y, time_dict)


class ConformalPredictiveDistributionFunction:

    '''
    NOTE
    The CPD contains all the information needed to form a
    prediction set. We can take quantiles and so on.
    '''

    def quantile(self, quantile, tau):
        raise NotImplementedError('Parent class has not quantile function')


    def predict_set(self, tau, epsilon=0.1, bounds='both'):
        '''
        The convex hull of the epsilon/2 and 1-epsilon/2 quantiles make up
        the prediction set Gamma(epsilon)
        '''
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
        return lower, upper
    

    # These methods relate to when the cpd is used to predict sets
    @staticmethod
    def err(Gamma, y):
        return int(not(Gamma[0] <= y <= Gamma[1]))
    

    @staticmethod
    def width(Gamma):
        return Gamma[1] - Gamma[0]



class NearestNeighboursPredictiveDistributionFunction(ConformalPredictiveDistributionFunction):
    '''
    TODO: Write tests
    '''

    def __init__(self, L, U, Y, time_dict=None):
        self.L = L 
        self.U = U
        self.Y = Y

        self.time_dict = time_dict

    def __call__(self, y, tau=None):
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
        
    
    def quantile(self, quantile, tau):
        q = np.inf
        for y in self.Y[::-1]:
            if self.__call__(y, tau) >= quantile:
                q = y
            else:
                return q
        return q
    
class DempsterHillConformalPredictiveDistribution(ConformalPredictiveDistributionFunction):

    def __init__(self, Y, time_dict=None):
        self.Y = Y
        self.time_dict = time_dict

    
    def __call__(self, y, tau=None):
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
        
    def quantile(self, quantile, tau):
        q = np.inf
        for y in self.Y[::-1]:
            if self.__call__(y, tau) >= quantile:
                q = y
            else:
                return q
        return q

if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)