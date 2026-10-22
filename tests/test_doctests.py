"""Gate the doctests of the predictor / Mondrian public API in the default lane.

Project-wide ``--doctest-modules`` is deliberately *not* enabled: a handful of
legacy ``martingale`` doctests predate NumPy's scalar ``repr`` change (they
print ``np.True_`` etc.) and would fail. This test keeps the doctests that back
the public predictor and Mondrian API honest without dragging those in — if an
example in one of these modules stops working, CI goes red.
"""

import doctest
import importlib

import pytest

DOCTEST_MODULES = [
    "online_cp.mondrian.tree",
    "online_cp.mondrian.framework",
    "online_cp.classifiers",
    "online_cp.regressors",
    "online_cp.venn",
]


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_module_doctests(module_name):
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False)
    assert result.failed == 0, (
        f"{result.failed} of {result.attempted} doctest(s) failed in {module_name}"
    )
