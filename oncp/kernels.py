import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class Kernel:

    def __call__(self, x):
        raise NotImplementedError("Subclasses should implement this!")

class GaussianKernel(Kernel):
    '''
    NOTE I think this is correct, but pdist and cdist are a bit confusing...
    X is a matrix and Y a column vector
    '''

    def __init__(self, sigma, distance='sqeuclidean'):
        self.sigma = sigma
        self.distance = distance
        self.name = 'Gaussian'
    

    def __call__(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            dists = pdist(X, metric=self.distance)
            K = np.exp(-dists / (2*self.sigma**2))
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            dists = cdist(X, Y, metric=self.distance)
            K = np.exp(-dists / (2*self.sigma**2))
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
    
class PolynomialKernel(Kernel):

    def __init__(self, d, c):
        self.name = 'Polynomial'
        self.d = d
        self.c = c

    def __call__(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            K = (X @ X.T + self.c)**self.d
        else:
            Y = np.atleast_1d(y)
            K = (X @ Y + self.c)**self.d
            K = K.reshape(-1, )
        return K
    

class PeriodicKernel(Kernel):
    '''
    p is period
    s is smoothing
    '''

    def __init__(self, p, s, distance='euclidean'):
        self.p = p
        self.s = s
        self.distance = distance
        self.name = 'Gaussian'
    

    def __call__(self, X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            dists = pdist(X, metric=self.distance)
            K = np.exp(-2*np.sin(np.pi*dists)**2 / (self.s))
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            dists = cdist(X, Y, metric=self.distance)
            K = np.exp(-2*np.sin(np.pi*dists)**2 / (self.s))
            K = K.reshape(-1, )
        return K
    

class LinearCombinationKernel(Kernel):
    '''
    Positive  combinations of kernels is still a kernel
    >>> k1 = LinearKernel()
    >>> k2 = LinearKernel()
    >>> kernels = [k1, k2]
    >>> weights = [1/2, 1/2]
    >>> k = LinearCombinationKernel(kernels, weights)
    >>> X = np.array([[1,2], [3,4]]) 
    >>> np.allclose(k(X), (k1(X) + k2(X))/2)
    True
    '''
    def __init__(self, kernels, weights=None):
        
        if weights is None:
            weights = [1 for _ in kernels]
        if len(kernels) != len(weights):
            raise ValueError("The number of kernels and weights must be the same")
        self.kernels = kernels
        self.weights = weights

        self.name = f'LinearCombinationKernel' # TODO Write out the explicit combination as name, e.g. '0.3*Polynomial + 0.4*Gaussian

    def __call__(self, X, y=None):
        return sum(weight * kernel(X, y) for weight, kernel in zip(self.weights, self.kernels))
    

class ProductKernel(Kernel):
    '''
    TODO Implement. Need to understand what product is the relevant one. Inner? Matrix product?
    '''
    pass

if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
