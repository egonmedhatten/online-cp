"""Kernel functions for kernel-based conformal predictors.

Provides Gaussian (RBF), linear, polynomial, periodic, and composite kernels.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "GaussianKernel",
    "LinearKernel",
    "PolynomialKernel",
    "PeriodicKernel",
    "LinearCombinationKernel",
    "kernel_induced_distance",
    "kernel_matrix_to_distance_matrix",
]


class Kernel:
    """Base class for kernel functions."""

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        raise NotImplementedError("Subclasses should implement this!")


class GaussianKernel(Kernel):
    """Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||² / (2σ²)).

    Parameters
    ----------
    sigma : float
        Bandwidth parameter.
    distance : str, optional
        Distance metric passed to scipy (default ``'sqeuclidean'``).
    """

    def __init__(self, sigma: float, distance: str = "sqeuclidean") -> None:
        self.sigma = sigma
        self.distance = distance
        self.name = "Gaussian"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        X = np.atleast_2d(X)
        if y is None:
            dists = pdist(X, metric=self.distance)
            K = np.exp(-dists / (2 * self.sigma**2))
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            dists = cdist(X, Y, metric=self.distance)
            K = np.exp(-dists / (2 * self.sigma**2))
            K = K.reshape(
                -1,
            )
        return K


class LinearKernel(Kernel):
    """Linear kernel: k(x, y) = x · y."""

    def __init__(self) -> None:
        self.name = "Linear"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        X = np.atleast_2d(X)
        if y is None:
            K = X @ X.T
        else:
            Y = np.atleast_1d(y)
            K = X @ Y
            K = K.reshape(
                -1,
            )
        return K


class PolynomialKernel(Kernel):
    """Polynomial kernel: k(x, y) = (x · y + c)^d.

    Parameters
    ----------
    d : int
        Degree of the polynomial.
    c : float
        Offset constant.
    """

    def __init__(self, d: int, c: float) -> None:
        self.name = "Polynomial"
        self.d = d
        self.c = c

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        X = np.atleast_2d(X)
        if y is None:
            K = (X @ X.T + self.c) ** self.d
        else:
            Y = np.atleast_1d(y)
            K = (X @ Y + self.c) ** self.d
            K = K.reshape(
                -1,
            )
        return K


class PeriodicKernel(Kernel):
    """Periodic kernel: k(x, y) = exp(-2 sin²(π||x-y||) / s).

    Parameters
    ----------
    p : float
        Period.
    s : float
        Smoothing parameter.
    distance : str, optional
        Distance metric passed to scipy (default ``'euclidean'``).
    """

    def __init__(self, p: float, s: float, distance: str = "euclidean") -> None:
        self.p = p
        self.s = s
        self.distance = distance
        self.name = "Gaussian"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        X = np.atleast_2d(X)
        if y is None:
            dists = pdist(X, metric=self.distance)
            K = np.exp(-2 * np.sin(np.pi * dists) ** 2 / (self.s))
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            dists = cdist(X, Y, metric=self.distance)
            K = np.exp(-2 * np.sin(np.pi * dists) ** 2 / (self.s))
            K = K.reshape(
                -1,
            )
        return K


class LinearCombinationKernel(Kernel):
    """
    Positive  combinations of kernels is still a kernel
    >>> k1 = LinearKernel()
    >>> k2 = LinearKernel()
    >>> kernels = [k1, k2]
    >>> weights = [1 / 2, 1 / 2]
    >>> k = LinearCombinationKernel(kernels, weights)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> np.allclose(k(X), (k1(X) + k2(X)) / 2)
    True
    """

    def __init__(self, kernels: Sequence[Kernel], weights: Sequence[float] | None = None) -> None:

        if weights is None:
            weights = [1 for _ in kernels]
        if len(kernels) != len(weights):
            raise ValueError("The number of kernels and weights must be the same")
        self.kernels = kernels
        self.weights = weights

        self.name = "LinearCombinationKernel"  # TODO Write out the explicit combination as name, e.g. '0.3*Polynomial + 0.4*Gaussian

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        return sum(weight * kernel(X, y) for weight, kernel in zip(self.weights, self.kernels))


class ProductKernel(Kernel):
    """
    TODO Implement. Need to understand what product is the relevant one. Inner? Matrix product?
    """

    pass


def kernel_induced_distance(kernel: Kernel) -> Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]] | None], NDArray[np.floating[Any]]]:
    """Create a distance function from a kernel, compatible with ``distance_func``.

    The kernel-induced distance is:

    .. math::
        d_K(x, x') = \\sqrt{K(x,x) - 2\\,K(x,x') + K(x',x')}

    Parameters
    ----------
    kernel : Kernel
        A kernel object with signature ``kernel(X, y=None)``.

    Returns
    -------
    distance_func : callable
        A function ``distance_func(X, y=None)`` returning:

        - If ``y is None``: symmetric distance matrix of shape ``(n, n)``.
        - If ``y`` is given: column vector of distances of shape ``(n, 1)``.

        Compatible with the ``distance_func`` parameter of
        :class:`~online_cp.classifiers.ConformalNearestNeighboursClassifier` and
        :class:`~online_cp.CPS.NearestNeighboursPredictionMachine`.

    Examples
    --------
    >>> from online_cp.kernels import GaussianKernel, LinearKernel, kernel_induced_distance
    >>> import numpy as np
    >>> dist_fn = kernel_induced_distance(LinearKernel())
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> D = dist_fn(X)
    >>> np.allclose(D, D.T)
    True
    >>> np.allclose(np.diag(D), 0.0)
    True
    """

    def _distance(X, y=None):
        X = np.atleast_2d(X)
        if y is None:
            K = kernel(X)
            diag = np.diag(K)
            D_sq = diag[:, None] - 2 * K + diag[None, :]
            return np.sqrt(np.maximum(D_sq, 0.0))
        else:
            k_cross = kernel(X, y).ravel()
            Kyy = kernel(np.atleast_2d(y)).item()
            diag_X = np.array([kernel(X[i : i + 1]).item() for i in range(len(X))])
            D_sq = diag_X - 2 * k_cross + Kyy
            return np.sqrt(np.maximum(D_sq, 0.0)).reshape(-1, 1)

    return _distance


def kernel_matrix_to_distance_matrix(K: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Convert a kernel (Gram) matrix to a distance matrix.

    .. math::
        D_{ij} = \\sqrt{K_{ii} - 2\\,K_{ij} + K_{jj}}

    Parameters
    ----------
    K : numpy.ndarray of shape (n, n)
        A symmetric positive semi-definite kernel matrix.

    Returns
    -------
    D : numpy.ndarray of shape (n, n)
        The induced distance matrix.

    Examples
    --------
    >>> from online_cp.kernels import LinearKernel, kernel_matrix_to_distance_matrix
    >>> import numpy as np
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> K = LinearKernel()(X)
    >>> D = kernel_matrix_to_distance_matrix(K)
    >>> np.allclose(D[0, 1], np.sqrt(2))
    True
    """
    diag = np.diag(K)
    D_sq = diag[:, None] - 2 * K + diag[None, :]
    return np.sqrt(np.maximum(D_sq, 0.0))


if __name__ == "__main__":
    import doctest
    import sys

    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
