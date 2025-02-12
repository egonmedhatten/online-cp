#!/usr/bin/env python

print("Initializing __init__.py")
from online_cp.regressors import ConformalRidgeRegressor
from online_cp.classifiers import ConformalNearestNeighboursClassifier
from online_cp.martingale import PluginMartingale
from online_cp.CPS import RidgePredictionMachine