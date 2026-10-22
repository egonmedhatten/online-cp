"""Mondrian process components for ``online_cp``.

This subpackage centralises everything built on the Mondrian process:

- :class:`~online_cp.mondrian.tree.MondrianTree` — the standalone partition core
  (grow / traverse / visualise). Re-exported here.
- ``MondrianConformalClassifier`` / ``MondrianConformalRegressor`` — the
  group-conditional conformal *framework* wrappers (a taxonomy applied on top of
  any base predictor). These live in :mod:`online_cp.mondrian.framework` and are
  imported lazily to avoid an import cycle with :mod:`online_cp.regressors`.

The Mondrian *tree/forest* conformal predictors and the Mondrian Venn predictor
are thin adapters that live next to their peers in :mod:`online_cp.classifiers`,
:mod:`online_cp.regressors` and :mod:`online_cp.venn`.
"""

from online_cp.mondrian.tree import MondrianTree as MondrianTree

__all__ = [
    "MondrianTree",
    "MondrianConformalClassifier",
    "MondrianConformalRegressor",
]

_LAZY = {
    "MondrianConformalClassifier": "framework",
    "MondrianConformalRegressor": "framework",
}


def __getattr__(name: str):
    """Lazily import the framework wrappers (PEP 562).

    Deferring the ``framework`` import breaks the cycle
    ``online_cp.regressors`` → ``online_cp.mondrian.tree`` →
    ``online_cp.mondrian`` (package init) that would otherwise trigger
    ``framework`` → ``online_cp.regressors`` before the latter finishes loading.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__():
    return sorted(list(globals()) + __all__)
