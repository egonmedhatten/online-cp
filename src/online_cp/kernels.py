"""Kernel functions for kernel-based conformal predictors.

Provides Gaussian (RBF), linear, polynomial, periodic, and composite kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "CustomKernel",
    "GaussianKernel",
    "Kernel",
    "LinearCombinationKernel",
    "LinearKernel",
    "PeriodicKernel",
    "PolynomialKernel",
    "ProductKernel",
    "kernel_induced_distance",
    "kernel_matrix_to_distance_matrix",
]


class Kernel:
    """Base class for kernel functions.

    Subclasses implement ``__call__(X, y=None)`` with the convention:

    - ``kernel(X)`` returns the ``(n, n)`` Gram matrix.
    - ``kernel(X, y)`` returns an ``(n,)`` vector of similarities between
      each row of X and the single point y.
    """

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        raise NotImplementedError("Subclasses should implement this!")


class GaussianKernel(Kernel):
    """Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||² / (2σ²)).

    Parameters
    ----------
    sigma : float
        Bandwidth parameter.
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma
        self.name = "Gaussian"

    def __repr__(self) -> str:
        return f"GaussianKernel(sigma={self.sigma!r})"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        X = np.atleast_2d(X)
        if y is None:
            dists = pdist(X, metric="sqeuclidean")
            K = np.exp(-dists / (2 * self.sigma**2))
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            dists = cdist(X, Y, metric="sqeuclidean")
            K = np.exp(-dists / (2 * self.sigma**2))
            K = K.reshape(
                -1,
            )
        return K


class LinearKernel(Kernel):
    """Linear kernel: k(x, y) = x · y."""

    def __init__(self) -> None:
        self.name = "Linear"

    def __repr__(self) -> str:
        return "LinearKernel()"

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

    def __repr__(self) -> str:
        return f"PolynomialKernel(d={self.d!r}, c={self.c!r})"

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
    """Periodic kernel: k(x, y) = exp(-2 sin²(π||x-y|| / p) / s).

    Guaranteed PSD for scalar inputs; for higher dimensions the Gram matrix
    may not be strictly PSD.

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
        self.name = "Periodic"

    def __repr__(self) -> str:
        return f"PeriodicKernel(p={self.p!r}, s={self.s!r})"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        X = np.atleast_2d(X)
        if y is None:
            dists = pdist(X, metric=self.distance)
            K = np.exp(-2 * np.sin(np.pi * dists / self.p) ** 2 / self.s)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            Y = np.atleast_2d(y)
            dists = cdist(X, Y, metric=self.distance)
            K = np.exp(-2 * np.sin(np.pi * dists / self.p) ** 2 / self.s)
            K = K.reshape(
                -1,
            )
        return K


class LinearCombinationKernel(Kernel):
    """Positive linear combination of kernels (which is still a valid kernel).

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
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        self.kernels = kernels
        self.weights = weights

        self.name = " + ".join(f"{w}*{k.name}" for w, k in zip(weights, kernels))

    def __repr__(self) -> str:
        return f"LinearCombinationKernel({self.name})"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        return sum(weight * kernel(X, y) for weight, kernel in zip(self.weights, self.kernels))


class ProductKernel(Kernel):
    """Element-wise (Schur/Hadamard) product of kernels (which is still a valid kernel).

    >>> k1 = LinearKernel()
    >>> k2 = LinearKernel()
    >>> k = ProductKernel([k1, k2])
    >>> X = np.array([[1, 2], [3, 4]])
    >>> np.allclose(k(X), k1(X) * k2(X))
    True
    """

    def __init__(self, kernels: Sequence[Kernel]) -> None:
        if len(kernels) < 2:
            raise ValueError("ProductKernel requires at least two kernels")
        self.kernels = kernels
        self.name = " * ".join(k.name for k in kernels)

    def __repr__(self) -> str:
        return f"ProductKernel({self.name})"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        result = self.kernels[0](X, y)
        for kernel in self.kernels[1:]:
            result = result * kernel(X, y)
        return result


class CustomKernel(Kernel):
    """Wrapper for a user-provided kernel function.

    Parameters
    ----------
    func : callable
        A function with signature ``func(X, y=None)`` following the kernel
        convention: returns ``(n, n)`` Gram matrix if ``y is None``, else
        ``(n,)`` similarities.
    name : str, optional
        Name for display purposes.

    Examples
    --------
    >>> k = CustomKernel(lambda X, y=None: LinearKernel()(X, y), name="MyLinear")
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> np.allclose(k(X), X @ X.T)
    True
    """

    def __init__(self, func: Callable, name: str = "Custom") -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        self._func = func
        self.name = name

    def __repr__(self) -> str:
        return f"CustomKernel(name={self.name!r})"

    def __call__(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]] | None = None) -> NDArray[np.floating[Any]]:
        return self._func(X, y)


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
