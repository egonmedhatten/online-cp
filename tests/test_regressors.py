#!/usr/bin/env python

from online_cp.regressors import *
import leancheck

def prop_predict_empty(x: list[float]) -> bool:
    cp = ConformalRidgeRegressor()
    i = cp.predict(np.array(x))
    # alt: return i.lower == -np.inf and i.upper == np.inf
    return cp.predict(np.array(x)) == \
           ConformalPredictionInterval(-np.inf, np.inf)

leancheck.main()
