import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class Kernel:
    def __init__(self):
        pass

class RBF(Kernel):
    '''
    NOTE I think this is correct, but pdist and cdist are a bit confusing...
    X is a matrix and Y a column vector
    '''

    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'RBF'

    def __call__(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            sq_dists = pdist(X, metric="sqeuclidean")
            K = np.exp(-sq_dists / (2*self.sigma**2))
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            sq_dists = cdist(X, Y, metric="sqeuclidean")
            K = np.exp(-sq_dists / (2*self.sigma**2))
            K = K.reshape(-1, )
        return K
    
class LinearKernel(Kernel):

    def __init__(self):
        self.name = 'Linear'

    def __call__(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            K = X @ X.T
        else:
            Y = np.atleast_1d(y)
            K = X @ Y
            K = K.reshape(-1, )
        return K
    
# TODO Add doctest