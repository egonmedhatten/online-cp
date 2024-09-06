import numpy as np
import time
import warnings
from scipy.stats import t as t_dist
from scipy.spatial.distance import pdist, cdist, squareform


class OnlineRidgeRegressor:

    def __init__(self, a=0, warnings=True, autotune=False, verbose=0):
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

    def predict_point(self, x):
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

            yhat = x.T @ self.XTXinv @ self.X.T @ self.y
            
        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            yhat = None
        return yhat
    
    def predict_interval(self, x, epsilon=0.1):

        if self.X is not None:
            omega  = self.XTXinv @ self.X.T @ self.y
            degrees_of_freedom = self.X.shape[0] - self.X.shape[1]  

            yhat = x.T @ omega

            Yhats = self.X @ omega

            mse = np.sum((self.y - Yhats)**2)

            sigma_hat_sq = mse / degrees_of_freedom

            if epsilon <= 0:
                lower = -np.inf
                upper = np.inf
                
            elif epsilon >= 1:
                lower = yhat
                upper = yhat

            else:
                lower = yhat - t_dist.ppf(1 - epsilon/2, degrees_of_freedom) * np.sqrt(sigma_hat_sq * (1 + x.T @ self.XTXinv @ x))
                upper = yhat + t_dist.ppf(1 - epsilon/2, degrees_of_freedom) * np.sqrt(sigma_hat_sq * (1 + x.T @ self.XTXinv @ x))

        else:
            # With just one object, and no label, we cannot predict any meaningful interval
            lower = -np.inf
            upper = np.inf

        
        return lower, upper    

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
        
    @staticmethod
    def err(Gamma, y):
        return int(not(Gamma[0] <= y <= Gamma[1]))
    
    @staticmethod
    def width(Gamma):
        return Gamma[1] - Gamma[0]
    

class RidgeICP:
    '''
    This class should implement an ICP based on the nonconformity measure |y - yhat|. 
    It must have fit and calibrate methods. Then it can be run either in the batch or the online setting
    '''

    def __init__(self, a=0, warnings=True, autotune=False, verbose=0):
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
        self.is_calibrated = False
        self.is_fitted = False

    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.Id = np.identity(self.p)
        if self.autotune:
            self._tune_ridge_parameter()
        else:
            self.XTXinv = np.linalg.inv(self.X.T @ self.X + self.a*self.Id)
        self.is_fitted = True

    
    def calibrate(self, X_cal, y_cal):
        assert self.is_fitted
        self.h = y_cal.shape[0] # Calibration set size.
        yhat = X_cal @ self.XTXinv @ self.X.T @ self.y

        self.Alphas = np.abs(y_cal - yhat)
        self.Alphas.sort()
        self.is_calibrated = True

    
    def predict_interval(self, x, epsilon=0.1):
        assert self.is_calibrated

        idx = int(np.ceil((self.h+1)*(1-epsilon))-1)
        qhat = self.Alphas[idx]

        yhat = x.T @ self.XTXinv @ self.X.T @ self.y

        lower = yhat - qhat
        upper= yhat + qhat
        return lower, upper
    
    @staticmethod
    def err(Gamma, y):
        return int(not(Gamma[0] <= y <= Gamma[1]))
    
    @staticmethod
    def width(Gamma):
        return Gamma[1] - Gamma[0]
    

class ConfidencePredictorKNN:
    
    def __init__(self, k, distance='euclidean', distance_func=None, warnings=True, verbose=0):
        
        self.k = k
        self.distance = distance
        if distance_func is None:
            self.distance_func = self._standard_distance_func
        else:
            self.distance_func = distance_func
            self.distance = 'custom'

        self.X = None
        self.y = None

        self.verbose = verbose

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

    
    def learn_one(self, x, y):
 
        # Learn label y
        if self.y is None:
            self.y = np.array([y])
        else:
            self.y = np.append(self.y, y)

      
        # Learn object x
        if self.X is None:
            self.X = x.reshape(1,-1)
            # self.D = self.distance_func(self.X)
        else:
            # d = self.distance_func(self.X, x)
            # self.D = self.update_distance_matrix(self.D, d)
            self.X = np.append(self.X, x.reshape(1, -1), axis=0)


    @staticmethod
    def err(Gamma, y):
        return int(not(Gamma[0] <= y <= Gamma[1]))
    

    @staticmethod
    def width(Gamma):
        return Gamma[1] - Gamma[0]
    

    def predict(self, x, epsilon=0.1, return_update=False):
        assert self.X.shape[0] >= self.k

        # Calculate distances
        d = self.distance_func(self.X, x)

        k_nearest = d.argsort(axis=0)[1:self.k+1]

        k_nearest_lables = self.y[k_nearest]

        bounds = np.quantile(k_nearest_lables, [epsilon/2, 1-epsilon/2])

        lower = bounds[0]
        upper = bounds[1]

        return lower, upper


class ConfidencePredictor:

    def __init__(self, predictor):
        self.predictor = predictor

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.predictor.fit(X, y)

    
    def predict_interval(self, x, epsilon=0.1):
        
        Yhat = self.predictor.predict(self.X)
        self.Alphas = np.abs(self.y - Yhat)
        self.Alphas.sort()
        self.h = self.y.shape[0]

        idx = int(np.ceil((self.h+1)*(1-epsilon))-1)
        if idx >= self.h:
            qhat = np.inf
        elif idx <= 0:
            qhat = 0
        else:
            qhat = self.Alphas[idx]

        yhat = self.predictor.predict(x.reshape(1,-1))

        lower = yhat - qhat
        upper = yhat + qhat

        # if epsilon <= 0:
        #     lower = -np.inf
        #     upper = np.inf
                
        # elif epsilon >= 1:
        #     lower = yhat
        #     upper = yhat

        return lower, upper
    
    @staticmethod
    def err(Gamma, y):
        return int(not(Gamma[0] <= y <= Gamma[1]))
    

    @staticmethod
    def width(Gamma):
        return Gamma[1] - Gamma[0]