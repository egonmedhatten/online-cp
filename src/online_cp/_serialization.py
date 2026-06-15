"""Serialization utilities for online-cp models.

Provides the :class:`SerializableMixin` base class and a named-callable
registry (:func:`register_callable`) so that model classes can be saved to
disk and restored with exact numerical state — including the precise position
of their random-number generators.

.. warning::
    Model files use Python :mod:`pickle` internally via :mod:`joblib`.
    **Only load files from trusted sources** — loading a malicious file can
    execute arbitrary code.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Callable

__all__ = ["SerializableMixin", "SerializationError", "register_callable"]

# ---------------------------------------------------------------------------
# Named-callable registry
# ---------------------------------------------------------------------------

_CALLABLE_REGISTRY: dict[str, Callable] = {}


class SerializationError(Exception):
    """Raised when a model cannot be serialized or deserialized."""


def register_callable(name: str) -> Callable:
    """Register a callable under *name* for serialization.

    Use this decorator on module-level functions or classes that you want to
    pass as ``kernel``, ``distance_func``, or similar arguments to a model and
    then be able to save/load that model.

    Parameters
    ----------
    name : str
        Unique registry key.  Must be the same in the saving *and* loading
        process.

    Examples
    --------
    >>> @register_callable("my_distance")
    ... def my_distance(X, y=None):
    ...     import numpy as np
    ...     from scipy.spatial.distance import cdist, pdist, squareform
    ...     if y is None:
    ...         return squareform(pdist(X))
    ...     return cdist(X, y)
    """

    def decorator(fn: Callable) -> Callable:
        _CALLABLE_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def to_token(fn: Any) -> Any:
    """Convert a callable *fn* to a serialisable token.

    Non-callable values (strings, ints, numpy arrays, dataclass Kernel
    objects, …) are returned unchanged.  ``None`` is returned unchanged.

    Callable handling priority:
    1. Registered via :func:`register_callable` → registry token dict.
    2. Directly picklable (named functions, class instances, …) → returned
       as-is for joblib to handle.
    3. Anything else (lambdas, closures) → :class:`SerializationError`.
    """
    if fn is None:
        return None
    if not callable(fn):
        return fn  # string, int, ndarray, Kernel dataclass, etc.

    # Priority 1: explicit registry
    for reg_name, registered in _CALLABLE_REGISTRY.items():
        if fn is registered:
            return {"__type__": "registry", "name": reg_name}

    # Priority 2: picklable (covers named functions, Kernel instances, sklearn
    # estimators, …)
    try:
        pickle.dumps(fn)
        return fn  # joblib will handle it during dump
    except Exception:
        pass

    # Failed — give a helpful error message
    fn_repr = getattr(fn, "__qualname__", None) or repr(fn)
    raise SerializationError(
        f"Cannot serialize callable {fn_repr!r}.\n"
        "To fix this, choose one of:\n"
        "  1. Replace the lambda/closure with a named (module-level) function.\n"
        "  2. Register it before saving: @register_callable('my_name')."
    )


def from_token(token: Any) -> Any:
    """Restore a callable from a token produced by :func:`to_token`.

    Non-token values (anything that is not a ``{"__type__": ...}`` dict) are
    returned unchanged — this covers strings, raw picklable callables, and
    ``None``.
    """
    if token is None:
        return None
    if not isinstance(token, dict) or "__type__" not in token:
        return token  # raw picklable value passed through

    if token["__type__"] == "registry":
        name = token["name"]
        if name not in _CALLABLE_REGISTRY:
            raise SerializationError(
                f"Callable '{name}' not found in the registry.\n"
                f"Ensure the module that calls @register_callable('{name}') "
                "has been imported before loading."
            )
        return _CALLABLE_REGISTRY[name]

    raise SerializationError(f"Unknown token type: {token['__type__']!r}")


# ---------------------------------------------------------------------------
# Serializable mixin
# ---------------------------------------------------------------------------

class SerializableMixin:
    """Mixin that adds :meth:`save` / :meth:`load` to online-cp model classes.

    Subclasses declare class-level attribute tuples to drive the default
    implementation:

    ``_SAVE_PARAMS : tuple[str, ...]``
        Names of ``__init__`` keyword-argument parameters whose current values
        should be saved (hyperparameters, ``rnd_state``, etc.).  These are
        used to reconstruct the model via ``cls(**params)`` on load.

    ``_SAVE_STATE : tuple[str, ...]``
        Names of learned-state instance attributes (e.g. ``X``, ``y``,
        ``XTXinv``).  Saved and restored *after* construction so that the
        exact post-training state is reproduced.

    ``_SAVE_CALLABLES : tuple[str, ...]``
        Subset of ``_SAVE_PARAMS`` whose values are callable (``kernel``,
        ``distance_func``, …).  Values are routed through the callable
        registry / pickling logic on save and restored on load.

    ``_PARAM_MAP : dict[str, str]``
        Optional ``{kwarg_name: attr_name}`` mapping for cases where the
        constructor kwarg name differs from the attribute used to store it.
        Example: ``{"distance_func": "_distance_func_arg"}`` tells the mixin
        "read ``self._distance_func_arg`` when building the ``distance_func``
        constructor kwarg".

    In addition to the declared state, the mixin automatically handles the
    RNG: if the model has a ``rnd_gen`` attribute (a
    :class:`numpy.random.Generator`) its exact bit-generator state is saved
    and restored so that post-load predictions are numerically identical to
    what the original model would have produced.
    """

    _SAVE_PARAMS: tuple[str, ...] = ()
    _SAVE_STATE: tuple[str, ...] = ()
    _SAVE_CALLABLES: tuple[str, ...] = ()
    _PARAM_MAP: dict[str, str] = {}

    # ------------------------------------------------------------------ save

    def save(self, filepath: str | os.PathLike, *, compress: int = 3) -> None:
        """Save the model to *filepath* using :mod:`joblib`.

        .. warning::
            Model files use Python :mod:`pickle` internally.  Only load files
            from **trusted sources**.

        Parameters
        ----------
        filepath : str or path-like
            Destination path.  The ``.joblib`` extension is recommended.
        compress : int
            :mod:`joblib` compression level 0–9 (default 3).

        Examples
        --------
        >>> import numpy as np, tempfile, os
        >>> from online_cp import ConformalRidgeRegressor
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((30, 2))
        >>> y = X[:, 0] + rng.standard_normal(30) * 0.1
        >>> cp = ConformalRidgeRegressor(a=1e-3)
        >>> cp.learn_initial_training_set(X, y)
        >>> with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        ...     path = f.name
        >>> cp.save(path)
        >>> loaded = ConformalRidgeRegressor.load(path)
        >>> os.unlink(path)
        >>> orig = cp.predict(np.array([0.5, 0.5]))
        >>> back = loaded.predict(np.array([0.5, 0.5]))
        >>> (round(orig.lower, 8), round(orig.upper, 8)) == (round(back.lower, 8), round(back.upper, 8))
        True
        """
        import joblib

        try:
            from online_cp import __version__ as _lib_version
        except Exception:
            _lib_version = "unknown"

        try:
            envelope: dict[str, Any] = {
                "format_version": 1,
                "library_version": _lib_version,
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "params": self._get_params(),
                "state": self._get_state(),
            }
            joblib.dump(envelope, filepath, compress=compress)
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Failed to save model to {filepath!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, filepath: str | os.PathLike) -> SerializableMixin:
        """Load a model from *filepath*.

        .. warning::
            Only load files from **trusted sources**.

        Parameters
        ----------
        filepath : str or path-like
            Path to a file previously created by :meth:`save`.

        Returns
        -------
        instance of *cls*
            A fully restored model whose predictions will be identical to
            those of the original after the save point.
        """
        import joblib

        try:
            from online_cp import __version__ as _lib_version
        except Exception:
            _lib_version = "unknown"

        try:
            envelope = joblib.load(filepath)
        except Exception as exc:
            raise SerializationError(
                f"Failed to read model file {filepath!r}: {exc}"
            ) from exc

        fmt_ver = envelope.get("format_version", 0)
        if fmt_ver > 1:
            raise SerializationError(
                f"Unsupported format_version {fmt_ver}. "
                "Update online-cp to load this file."
            )

        lib_ver = envelope.get("library_version", "unknown")
        if lib_ver != _lib_version:
            warnings.warn(
                f"Model was saved with online-cp {lib_ver!r}, "
                f"but you are using {_lib_version!r}. "
                "Predictions may differ.",
                UserWarning,
                stacklevel=2,
            )

        expected = f"{cls.__module__}.{cls.__qualname__}"
        saved = envelope.get("class", "")
        if saved != expected:
            raise SerializationError(
                f"Class mismatch: file contains '{saved}', "
                f"expected '{expected}'. Use the correct class to load."
            )

        # Decode params (restore callables from tokens)
        raw_params: dict[str, Any] = envelope["params"]
        params: dict[str, Any] = {}
        for kwarg, val in raw_params.items():
            if kwarg in cls._SAVE_CALLABLES:
                val = from_token(val)
            params[kwarg] = val

        # Reconstruct via constructor (runs validation, sets defaults, inits rnd_gen)
        model = cls(**params)

        # Restore learned state (overwrites __init__ defaults)
        model._set_state(envelope["state"])

        return model

    # ------------------------------------------------------------------ helpers

    def _get_params(self) -> dict[str, Any]:
        """Build the params dict for the envelope."""
        params: dict[str, Any] = {}
        for kwarg in self._SAVE_PARAMS:
            attr = self._PARAM_MAP.get(kwarg, kwarg)
            val = getattr(self, attr, None)
            if kwarg in self._SAVE_CALLABLES:
                val = to_token(val)
            params[kwarg] = val
        return params

    def _get_state(self) -> dict[str, Any]:
        """Build the state dict for the envelope."""
        state: dict[str, Any] = {}
        for attr in self._SAVE_STATE:
            state[attr] = getattr(self, attr, None)
        # Capture exact RNG stream position so post-load predictions match
        rnd_gen = getattr(self, "rnd_gen", None)
        if rnd_gen is not None:
            state["__rng_state__"] = rnd_gen.bit_generator.state
        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        """Restore learned-state attributes from the loaded state dict."""
        for attr in self._SAVE_STATE:
            if attr in state:
                setattr(self, attr, state[attr])
        # Restore exact RNG stream position
        rnd_gen = getattr(self, "rnd_gen", None)
        if rnd_gen is not None and "__rng_state__" in state:
            rnd_gen.bit_generator.state = state["__rng_state__"]
