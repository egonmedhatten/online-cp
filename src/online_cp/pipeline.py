"""Conformal-safe pipeline composition.

This module provides a :class:`Pipeline` class, a :class:`Transformer` base,
and the composition utilities :class:`TransformerUnion`, :class:`Select`, and
:class:`Discard` that let feature maps be chained with any ``online-cp``
conformal predictor while preserving exchangeability.

Two sound transformer regimes are supported:

- **fixed** (``mode="fixed"``) — stateless maps applied identically to every
  example (e.g. :class:`FuncTransformer`, :class:`Select`, :class:`Discard`).
  Always safe; exchangeability is trivially preserved.
- **frozen** (``mode="frozen"``) — parameters computed once on the initial
  training set and held constant thereafter (e.g. :class:`StandardScaler`,
  :class:`MinMaxScaler` from :mod:`online_cp.preprocessing`).  Preserves
  training-conditional validity (ALRW2 §4.7).

Bag-fit (transductive) mode — in which parameters are recomputed from the
augmented bag on every prediction for exact finite-sample validity — is
planned for a future release.

Example
-------
>>> import numpy as np
>>> from online_cp import Pipeline, FuncTransformer, ConformalRidgeRegressor
>>> pipe = FuncTransformer(np.log1p) | ConformalRidgeRegressor(a=1.0, epsilon=0.1)
>>> X_tr = np.abs(np.random.default_rng(0).normal(size=(50, 4))) + 1
>>> y_tr = X_tr[:, 0] + 0.1 * np.random.default_rng(0).normal(size=50)
>>> pipe.learn_initial_training_set(X_tr, y_tr)
>>> x_new = np.abs(np.random.default_rng(1).normal(size=4)) + 1
>>> interval = pipe.predict(x_new, epsilon=0.1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Transformer(ABC):
    """Base class for conformal-safe feature transformers.

    Subclasses must implement :meth:`transform` (batch) and
    :meth:`transform_one` (single vector) and set the ``mode`` attribute.
    Stateful transformers should also override :meth:`fit` to compute
    parameters from a batch of training examples.

    The ``|`` operator composes a transformer with another transformer or a
    conformal predictor into a :class:`Pipeline`.  The ``+`` operator
    composes two transformers into a :class:`TransformerUnion`.
    """

    mode: str = "fixed"

    def fit(self, X: NDArray) -> None:
        """Fit the transformer on a batch of training examples.

        The default implementation marks the transformer as fitted and returns
        (suitable for stateless ``mode="fixed"`` transformers).  Stateful
        transformers such as :class:`~online_cp.preprocessing.StandardScaler`
        override this method to compute and store their parameters; they should
        call ``super().fit(X)`` to keep the ``_fitted`` flag up to date.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training batch *after* preceding transformers have been applied.
        """
        self._fitted: bool = True

    @abstractmethod
    def transform(self, X: NDArray) -> NDArray:
        """Apply the transformation to a batch of examples.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Input matrix.

        Returns
        -------
        ndarray
            Transformed matrix, same number of rows as *X*.
        """

    @abstractmethod
    def transform_one(self, x: NDArray) -> NDArray:
        """Apply the transformation to a single example.

        Parameters
        ----------
        x : ndarray of shape (d,)
            Input vector.

        Returns
        -------
        ndarray
            Transformed vector.
        """

    def __or__(self, other: Any) -> "Pipeline":
        """Build a :class:`Pipeline` via the ``|`` operator.

        Parameters
        ----------
        other : Transformer or conformal predictor
            The next step.  If *other* is already a :class:`Pipeline`, the
            current transformer is prepended to its steps; otherwise a new
            two-step pipeline is created.

        Returns
        -------
        Pipeline
        """
        if isinstance(other, Pipeline):
            return Pipeline(self, *other.steps)
        return Pipeline(self, other)

    def __add__(self, other: "Transformer") -> "TransformerUnion":
        """Build a :class:`TransformerUnion` via the ``+`` operator.

        Parameters
        ----------
        other : Transformer
            The transformer whose output will be concatenated with this one.

        Returns
        -------
        TransformerUnion
        """
        return TransformerUnion(self, other)


class FuncTransformer(Transformer):
    """Apply a fixed, stateless callable to every example.

    Because the function is data-independent, it is always conformally safe:
    exchangeability of the example sequence is preserved.

    The :meth:`fit` method is a no-op (inherited from :class:`Transformer`).

    Parameters
    ----------
    fn : callable
        A callable that accepts an ndarray and returns an ndarray of the same
        or different shape.  It must work on both a 2-D batch ``(n, d)`` and a
        1-D vector ``(d,)`` — ``numpy`` ufuncs (e.g. ``np.log``, ``np.sqrt``)
        and lambda functions satisfy this automatically.

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import FuncTransformer
    >>> ft = FuncTransformer(np.log1p)
    >>> ft.transform_one(np.array([0.0, 1.0, 2.0]))
    array([0.        , 0.69314718, 1.09861229])
    """

    mode: str = "fixed"

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def transform(self, X: NDArray) -> NDArray:
        return self.fn(X)

    def transform_one(self, x: NDArray) -> NDArray:
        return self.fn(x)


# Sound transformer modes — any other value triggers the validity guard.
_SOUND_MODES = frozenset({"fixed", "frozen"})


class Pipeline:
    """A conformal-safe pipeline of transformers ending in a conformal predictor.

    The pipeline re-exposes the full ``online-cp`` predictor API so that a
    composed object is a drop-in replacement for any bare predictor in
    :func:`~online_cp.evaluate.progressive_val` or
    :func:`~online_cp.evaluate.iter_progressive_val`.

    **Validity guarantee.** By default the pipeline only accepts transformers
    whose ``mode`` is ``"fixed"`` or ``"frozen"``; both regimes preserve
    conformal validity (ALRW2 §4.5 / §4.7).  Transformers with any other mode
    (e.g. ``"incremental"``) break the exchangeability of the example sequence
    and are **rejected at construction time** unless you explicitly pass
    ``unsafe_incremental=True`` — which opts out of the finite-sample validity
    guarantee.

    **Bare-callable auto-wrap.** If a transformer step is a plain callable
    (rather than a :class:`Transformer` instance), it is silently wrapped in a
    :class:`FuncTransformer`.  Non-callable, non-Transformer steps raise a
    :exc:`TypeError`.

    Parameters
    ----------
    *steps :
        One or more transformer steps followed by exactly one conformal
        predictor.  At least two steps are required.  Each transformer step
        may be a :class:`Transformer` instance or a bare callable.
    unsafe_incremental : bool, default False
        Set to ``True`` to allow transformers with modes outside
        ``{"fixed", "frozen"}``.  This voids the conformal validity guarantee.

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import Pipeline, FuncTransformer, ConformalRidgeRegressor
    >>> pipe = Pipeline(FuncTransformer(np.log1p), ConformalRidgeRegressor(a=1.0))
    >>> # Equivalent — bare callable is auto-wrapped:
    >>> pipe2 = Pipeline(np.log1p, ConformalRidgeRegressor(a=1.0))
    >>> # Operator sugar:
    >>> pipe3 = FuncTransformer(np.log1p) | ConformalRidgeRegressor(a=1.0)
    """

    def __init__(self, *steps: Any, unsafe_incremental: bool = False) -> None:
        if len(steps) < 2:
            raise ValueError(
                "Pipeline requires at least two steps: one or more transformers "
                "followed by a conformal predictor."
            )
        # Auto-wrap bare callables; reject non-callable non-Transformers.
        transformer_steps = []
        for i, step in enumerate(steps[:-1]):
            if isinstance(step, Transformer):
                transformer_steps.append(step)
            elif callable(step):
                transformer_steps.append(FuncTransformer(step))
            else:
                raise TypeError(
                    f"Pipeline step {i} is neither a Transformer nor a callable "
                    f"(got {type(step).__name__!r}).  Wrap it in a FuncTransformer "
                    "or provide a callable."
                )
        # Validity guard: reject non-sound modes unless opted in.
        if not unsafe_incremental:
            for t in transformer_steps:
                if t.mode not in _SOUND_MODES:
                    raise ValueError(
                        f"Transformer {t!r} has mode={t.mode!r}, which is not in "
                        f"the set of sound modes {set(_SOUND_MODES)}.  "
                        "Such transformers may break the exchangeability of the "
                        "example sequence and invalidate the conformal coverage "
                        "guarantee (ALRW2 §4.5).  "
                        "Pass unsafe_incremental=True to Pipeline to opt in, "
                        "accepting that finite-sample validity is no longer assured."
                    )
        self.transformers: list[Transformer] = transformer_steps
        self.estimator: Any = steps[-1]
        self.steps: tuple[Any, ...] = (tuple(transformer_steps) + (steps[-1],))
        self._unsafe_incremental = unsafe_incremental

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform_batch(self, X: NDArray) -> NDArray:
        """Apply all transformers to a batch matrix."""
        Xt = X
        for t in self.transformers:
            Xt = t.transform(Xt)
        return Xt

    def _transform_one(self, x: NDArray) -> NDArray:
        """Apply all transformers to a single vector."""
        xt = x
        for t in self.transformers:
            xt = t.transform_one(xt)
        return xt

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def learn_initial_training_set(self, X: NDArray, y: NDArray) -> None:
        """Fit all transformers then delegate to the estimator's initial fit.

        For each transformer in order: ``fit(Xt)`` is called to let stateful
        transformers (e.g. frozen scalers) compute their parameters, then
        ``transform(Xt)`` is applied to produce the input for the next step.
        The estimator receives the fully-transformed ``Xt``.

        Parameters
        ----------
        X : ndarray of shape (n, d)
        y : ndarray of shape (n,)
        """
        Xt = X
        for t in self.transformers:
            t.fit(Xt)
            Xt = t.transform(Xt)
        self.estimator.learn_initial_training_set(Xt, y)

    def learn_one(self, x: NDArray, y: Any, precomputed: Any = None) -> None:
        """Transform *x* and delegate to the estimator's online update.

        ``precomputed`` is always dropped at the pipeline boundary: the
        transform changes the geometry so any cached intermediate is invalid.

        Parameters
        ----------
        x : ndarray of shape (d,)
        y : label / target value
        precomputed : ignored
            Accepted for API compatibility; always discarded.
        """
        xt = self._transform_one(x)
        self.estimator.learn_one(xt, y)

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict(self, x: NDArray, **kwargs: Any) -> Any:
        """Transform *x* then call ``estimator.predict``.

        All keyword arguments (``epsilon``, ``return_p_values``, ``bounds``,
        ``return_update``, …) are forwarded unchanged.

        Parameters
        ----------
        x : ndarray of shape (d,)
        **kwargs
            Forwarded to ``estimator.predict``.
        """
        xt = self._transform_one(x)
        return self.estimator.predict(xt, **kwargs)

    def predict_cpd(self, x: NDArray, **kwargs: Any) -> Any:
        """Transform *x* then call ``estimator.predict_cpd``.

        Parameters
        ----------
        x : ndarray of shape (d,)
        **kwargs
            Forwarded to ``estimator.predict_cpd``.
        """
        xt = self._transform_one(x)
        return self.estimator.predict_cpd(xt, **kwargs)

    def compute_p_value(self, x: NDArray, y: Any, **kwargs: Any) -> Any:
        """Transform *x* then call ``estimator.compute_p_value``.

        Parameters
        ----------
        x : ndarray of shape (d,)
        y : label / target value
        **kwargs
            Forwarded to ``estimator.compute_p_value``.
        """
        xt = self._transform_one(x)
        return self.estimator.compute_p_value(xt, y, **kwargs)

    # ------------------------------------------------------------------
    # Attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, so self.transformers
        # and self.estimator are always found via the normal path.
        try:
            return getattr(self.__dict__["estimator"], name)
        except KeyError:
            raise AttributeError(name) from None

    # ------------------------------------------------------------------
    # Operator sugar
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a human-readable summary of the pipeline structure.

        Returns
        -------
        dict
            A dict with the following keys:

            ``n_steps`` : int
                Total number of steps (transformers + estimator).
            ``transformers`` : list of dict
                One entry per transformer with keys ``type`` (class name),
                ``mode`` (e.g. ``"fixed"`` or ``"frozen"``), and ``fitted``
                (``True`` after :meth:`learn_initial_training_set` has run).
            ``estimator`` : dict
                ``{"type": <class name>}``.
            ``unsafe_incremental`` : bool
                Whether the validity guard was disabled at construction.

        Examples
        --------
        >>> import numpy as np
        >>> from online_cp import Pipeline, FuncTransformer, ConformalRidgeRegressor
        >>> pipe = Pipeline(FuncTransformer(np.abs), ConformalRidgeRegressor(a=1.0))
        >>> pipe.summary()["n_steps"]
        2
        """
        def _is_fitted(t: Transformer) -> bool:
            # A transformer is fitted if the base fit() set _fitted,
            # OR if it has frozen-scaler attributes (mean_, data_min_).
            return bool(
                getattr(t, "_fitted", False)
                or getattr(t, "mean_", None) is not None
                or getattr(t, "data_min_", None) is not None
            )

        return {
            "n_steps": len(self.transformers) + 1,
            "transformers": [
                {
                    "type": type(t).__name__,
                    "mode": t.mode,
                    "fitted": _is_fitted(t),
                }
                for t in self.transformers
            ],
            "estimator": {"type": type(self.estimator).__name__},
            "unsafe_incremental": self._unsafe_incremental,
        }

    def __or__(self, other: Any) -> "Pipeline":
        """Append a step to this pipeline via the ``|`` operator."""
        return Pipeline(*self.steps, other,
                        unsafe_incremental=self._unsafe_incremental)

    def __repr__(self) -> str:
        step_reprs = " | ".join(repr(s) for s in self.steps)
        return f"Pipeline({step_reprs})"


# ---------------------------------------------------------------------------
# Composition utilities
# ---------------------------------------------------------------------------


class TransformerUnion(Transformer):
    """Concatenate the outputs of several transformers along the feature axis.

    Each constituent transformer is applied to the same input; their outputs
    are column-stacked to produce a wider feature matrix.  All constituent
    transformers are fitted independently during
    :meth:`~Pipeline.learn_initial_training_set`.

    The union's ``mode`` is ``"fixed"`` when all constituents are fixed, and
    ``"frozen"`` otherwise (most restrictive member wins).

    Parameters
    ----------
    *transformers : Transformer
        Two or more transformers whose outputs will be concatenated.

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import FuncTransformer, Select, TransformerUnion
    >>> # Build polynomial-like features: [x, x**2]
    >>> union = FuncTransformer(lambda x: x) + FuncTransformer(lambda x: x ** 2)
    >>> X = np.arange(6, dtype=float).reshape(3, 2)
    >>> union.fit(X)
    >>> union.transform(X).shape
    (3, 4)
    """

    def __init__(self, *transformers: Transformer) -> None:
        if len(transformers) < 2:
            raise ValueError("TransformerUnion requires at least two transformers.")
        self._transformers = list(transformers)
        modes = {t.mode for t in self._transformers}
        self.mode = "frozen" if "frozen" in modes else "fixed"

    def fit(self, X: NDArray) -> None:
        for t in self._transformers:
            t.fit(X)

    def transform(self, X: NDArray) -> NDArray:
        return np.hstack([t.transform(X) for t in self._transformers])

    def transform_one(self, x: NDArray) -> NDArray:
        return np.concatenate([t.transform_one(x) for t in self._transformers])

    def __repr__(self) -> str:
        parts = " + ".join(repr(t) for t in self._transformers)
        return f"TransformerUnion({parts})"


class Select(Transformer):
    """Keep a subset of input columns.

    This is a stateless (``mode="fixed"``), always-safe transform.

    Parameters
    ----------
    indices : array-like of int
        Column indices to retain, in the desired output order.

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import Select
    >>> sel = Select([0, 2])
    >>> X = np.arange(12, dtype=float).reshape(4, 3)
    >>> sel.fit(X)
    >>> sel.transform(X)
    array([[ 0.,  2.],
           [ 3.,  5.],
           [ 6.,  8.],
           [ 9., 11.]])
    """

    mode: str = "fixed"

    def __init__(self, indices: Any) -> None:
        self._indices = list(indices)

    def transform(self, X: NDArray) -> NDArray:
        return X[:, self._indices]

    def transform_one(self, x: NDArray) -> NDArray:
        return x[self._indices]

    def __repr__(self) -> str:
        return f"Select({self._indices!r})"


class Discard(Transformer):
    """Drop a subset of input columns, keeping all others.

    This is a stateless (``mode="fixed"``), always-safe transform.
    The remaining columns are returned in their original order.

    Parameters
    ----------
    indices : array-like of int
        Column indices to discard.

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import Discard
    >>> drop = Discard([1])
    >>> X = np.arange(12, dtype=float).reshape(4, 3)
    >>> drop.fit(X)
    >>> drop.transform(X)
    array([[ 0.,  2.],
           [ 3.,  5.],
           [ 6.,  8.],
           [ 9., 11.]])
    """

    mode: str = "fixed"

    def __init__(self, indices: Any) -> None:
        self._discard = set(indices)

    def _keep(self, d: int) -> list[int]:
        return [i for i in range(d) if i not in self._discard]

    def transform(self, X: NDArray) -> NDArray:
        return X[:, self._keep(X.shape[1])]

    def transform_one(self, x: NDArray) -> NDArray:
        keep = self._keep(x.shape[0])
        return x[keep]

    def __repr__(self) -> str:
        return f"Discard({sorted(self._discard)!r})"
