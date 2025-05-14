#!/usr/bin/env python

from online_cp.regressors import ConformalRidgeRegressor
from online_cp.classifiers import ConformalNearestNeighboursClassifier
from online_cp.martingale import PluginMartingale
from online_cp.CPS import RidgePredictionMachine
from online_cp.evaluation import Evaluation, Err, OE, OF, WinklerScore, Width, CRPS