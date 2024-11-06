import numpy as np
import time
import warnings
from scipy.spatial.distance import pdist, cdist, squareform

class ConformalClssifier:
    '''
    Parent class for classifiers
    '''

    def __init__(self):
        self.Err = 0
        # Preferred efficiency criteria (See Protocol 3.1 ALRW)
        self.OE = 0
        self.OF = 0


    @staticmethod
    def _compute_p_value(Alpha, tau=1, score_type='nonconformity'):
        '''
        Assumes that the (non) conformity scores are organised so that the 
        test example is the last element.
        If tau is not provided, the non-smoothed p-value is computed.
        '''
        if score_type == 'nonconformity':
            gt = np.count_nonzero(Alpha > Alpha[-1])
            eq = np.count_nonzero(Alpha == Alpha[-1])
            p = (gt + tau * eq) / Alpha.shape[0]

        elif score_type == 'conformity':
            lt = np.count_nonzero(Alpha < Alpha[-1])
            eq = np.count_nonzero(Alpha == Alpha[-1])

            p = (lt + tau * eq) / Alpha.shape[0]
        
        return p


    def _compute_Gamma(self, p_values, epsilon):
        Gamma = []
        for y in self.label_space:
            if p_values[y] > epsilon:
                Gamma.append(y)
        return np.array(Gamma) #self.label_space[np.where(p_values > epsilon)[0]]
    

    def err(self, Gamma, y):
        err = int(not(y in Gamma))
        self.Err += err
        return err
    

    def oe(self, Gamma, y):
        if y in Gamma:
            oe = Gamma.shape[0] - 1
        else:
            oe = Gamma.shape[0]
        self.OE += oe
        return oe
    

    def of(self, p_values, y):
        of = 0
        for label, p in p_values.items():
            if not label == y:
                of += p
        self.OF += of
        return of


class ConformalPassiveAggressive(ConformalClssifier):
    '''
    One of three passive-aggressive algorithms described in
    https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
    
    This is a binary classifier with labels -1 and 1

    Conformity score is the label times the confdence in prediciton

    TODO add high_class name and low_class name to avoid reformulating

    NOTE We can easily add on a kernel by simply changing the loss function, as described 
    just before Section 4 in the paper
    '''

    def __init__(self, d, warnings=True, verbose=0, rnd_state=2024):
        super().__init__()
        
        self.label_space = np.array([-1, 1]) # Binary classifier with labels -1 and 1

        self.y = np.empty(0)
        self.X = np.empty((0, d))

        self.w = np.zeros(d)
        self.d = d

        # Should we raise warnings
        self.warnings = warnings

        self.verbose = verbose
        self.rnd_gen = np.random.default_rng(rnd_state)


    @staticmethod
    def _sign(x):
        '''
        The label space is {-1, 1}, so we have to adjust the sign function from numpy
        '''
        return np.sign(x) if x != 0 else 1


    def _loss(self, x, y):
        return max(0, 1 - y * np.dot(self.w, x))
    

    @staticmethod
    def _compute_theta(l, x):
        return l / np.linalg.norm(x)**2


    def learn_one(self, x, y):

        # Update vector of weights
        l = self._loss(x, y)
        theta = self._compute_theta(l, x)

        self.w += theta * y * x
        
        # Learn label
        self.y = np.append(self.y, y)

        # Learn object
        self.X = np.append(self.X, x.reshape(1, -1), axis=0)


    def predict_set(self, x, epsilon=0.1, return_p_values=False):
        p_values = {}

        tau = self.rnd_gen.uniform(0, 1)
        X = np.append(self.X, x.reshape(1, -1), axis=0)

        for label in self.label_space:
            
            y = np.append(self.y, label)
            
            Beta = y * (X @ self.w)

            p_values[label] = self._compute_p_value(Beta, tau, 'conformity')
        
        Gamma = self._compute_Gamma(p_values, epsilon)

        if return_p_values:
            return Gamma, p_values
        else:
            return Gamma
        
    def predict_p(self, x):
        '''
        Predict just the p-values
        '''

        p_values = {}

        tau = self.rnd_gen.uniform(0, 1)
        X = np.append(self.X, x.reshape(1, -1), axis=0)

        for label in self.label_space:
            
            y = np.append(self.y, label)
            
            Beta = y * (X @ self.w)

            p_values[label] = self._compute_p_value(Beta, tau, 'conformity')
        return p_values
            

class ConformalPassiveAggressive_I(ConformalPassiveAggressive):

    def __init__(self, d, C, warnings=True, verbose=0, rnd_state=2024):
        super().__init__(d, warnings, verbose, rnd_state)
        self.C = C


    def _compute_theta(self,l, x):
        return min(self.C, l / np.linalg.norm(x)**2)
    

class ConformalPassiveAggressive_II(ConformalPassiveAggressive):

    def __init__(self, d, C, warnings=True, verbose=0, rnd_state=2024):
        super().__init__(d, warnings, verbose, rnd_state)
        self.C = C

    
    def _compute_theta(self, l, x):
        return l / ( np.linalg.norm(x)**2 + 1 / (2*self.C))
    

class ConformalOneNearestNeighbours(ConformalClssifier):

    def __init__(self, distance='euclidean', distance_func=None, verbose=0, rnd_state=2024):
        super().__init__()
        self.label_space = np.array([-1, 1]) # Binary classifier with labels -1 and 1

        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = 'custom'

        self.y = np.empty(0)
        self.X = None
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
    

    @staticmethod
    def _find_nearest_distances(D, y):
        n = D.shape[0]
        
        # Initialize arrays to store the results
        same_label_distances = np.full(n, np.inf)
        different_label_distances = np.full(n, np.inf)

        for i in range(n):
            # Create a mask for the same and different labels
            same_label_mask = (y == y[i])
            different_label_mask = (y != y[i])

            # Ignore the distance to itself by setting it to np.inf
            same_label_mask[i] = False

            # Extract distances for the same label
            if np.any(same_label_mask):
                same_label_distances[i] = np.min(D[i, same_label_mask])
            
            # Extract distances for the different label
            if np.any(different_label_mask):
                different_label_distances[i] = np.min(D[i, different_label_mask])
        
        return same_label_distances, different_label_distances
    

    def learn_one(self, x, y):
        # Learn label y
        self.y = np.append(self.y, y)

        # Learn object
        if self.X is None:
            self.X = x.reshape(1,-1)
            self.D = self.distance_func(self.X)
        else:
            d = self.distance_func(self.X, x)
            self.D = self.update_distance_matrix(self.D, d)
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)


    def predict_set(self, x, epsilon=0.1, return_p_values=False):
        '''
        TODO Fix the precomputed option as in the regression setting.
        '''
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)

        if self.y.shape[0] >= 1: 
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            
            for label in self.label_space:
                y = np.append(self.y, label)
                
                same_label_distances, different_label_distances = self._find_nearest_distances(D, y)

                Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)

                p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
        
            Gamma = self._compute_Gamma(p_values, epsilon)
                        
            if return_p_values:
                return Gamma, p_values
            else:
                return Gamma
            
        else:
            for label in self.label_space:
                Alpha = np.array([0])
                p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
            Gamma = self._compute_Gamma(p_values, epsilon)

            if return_p_values:
                return self.label_space, {-1: self.rnd_gen.uniform(0, 1), 1: self.rnd_gen.uniform(0, 1)}
            else:
                return self.label_space
            

    def predict_p(self, x):
        p_values = {}
        tau = self.rnd_gen.uniform(0, 1)
        
        if self.y.shape[0] >= 1: 
            d = self.distance_func(self.X, x)
            D = self.update_distance_matrix(self.D, d)
            
            for label in self.label_space:
                y = np.append(self.y, label)
                
                same_label_distances, different_label_distances = self._find_nearest_distances(D, y)

                Alpha = np.nan_to_num(same_label_distances / different_label_distances, nan=np.inf)

                p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
            
        else:
            for label in self.label_space:
                Alpha = np.array([0])
                p_values[label] = self._compute_p_value(Alpha, tau, 'nonconformity')
        
        return p_values
            

class OnlineConfidencePrediction:

    def __init__(self, cp, Epsilon):
        '''
        Protocol 3.1 ALRW
        Epsilon is a grid of significance levels

        NOTE We can use the predict_p method to produce p-values and then prediction sets at each significance level
        Err, OE and OF will have to be computed at each significance level, and the internal counter of the conformal predictor 
        has to be ignored
        '''
        self.cp = cp
        self.Epsilon = Epsilon

        # Initialise the metrics
        self.Err = {epsilon: 0 for epsilon in self.Epsilon}
        self.OE = {epsilon: 0 for epsilon in self.Epsilon}
        self.OF = 0

    
    def learn_one(self, x, y):
        self.cp.learn_one(x, y)

    
    def predict_p(self, x):
        return self.cp.predict_p(x)

    
    def compute_Gamma(self, p_values):
        Gammas = {}
        for epsilon in self.Epsilon:
            Gammas[epsilon] = self.cp._compute_Gamma(p_values, epsilon)
        return Gammas
    

    def update_metrics(self, p_values, Gammas, y):
        for epsilon, Gamma in Gammas.items():
            self.Err[epsilon] += self.cp.err(Gamma, y)
            self.OE[epsilon] += self.cp.oe(Gamma, y)
        self.OF += self.cp.of(p_values, y)