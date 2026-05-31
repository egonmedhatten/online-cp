#!/usr/bin/env python
"""online-cp: Online Conformal Prediction.

Provides conformal regressors, classifiers, conformal predictive systems,
conformal test martingales, and evaluation metrics.
"""

__version__ = "0.2.0"

from online_cp.classifiers import (
    ConformalNearestNeighboursClassifier as ConformalNearestNeighboursClassifier,
)
from online_cp.classifiers import (
    ConformalSupportVectorMachine as ConformalSupportVectorMachine,
)
from online_cp.CPS import (
    DempsterHillConformalPredictiveSystem as DempsterHillConformalPredictiveSystem,
)
from online_cp.CPS import (
    KernelRidgePredictionMachine as KernelRidgePredictionMachine,
)
from online_cp.CPS import (
    NearestNeighboursPredictionMachine as NearestNeighboursPredictionMachine,
)
from online_cp.CPS import (
    RidgePredictionMachine as RidgePredictionMachine,
)
from online_cp.evaluate import (
    progressive_val as progressive_val,
)
from online_cp.evaluate import (
    iter_progressive_val as iter_progressive_val,
)
from online_cp.metrics import (
    CRPS as CRPS,
)
from online_cp.metrics import (
    ErrorRate as ErrorRate,
)
from online_cp.metrics import (
    IntervalWidth as IntervalWidth,
)
from online_cp.metrics import (
    Metric as Metric,
)
from online_cp.metrics import (
    Metrics as Metrics,
)
from online_cp.metrics import (
    ObservedExcess as ObservedExcess,
)
from online_cp.metrics import (
    ObservedFuzziness as ObservedFuzziness,
)
from online_cp.metrics import (
    SetSize as SetSize,
)
from online_cp.metrics import (
    WinklerScore as WinklerScore,
)
from online_cp.kernels import (
    GaussianKernel as GaussianKernel,
)
from online_cp.kernels import (
    LinearCombinationKernel as LinearCombinationKernel,
)
from online_cp.kernels import (
    LinearKernel as LinearKernel,
)
from online_cp.kernels import (
    PeriodicKernel as PeriodicKernel,
)
from online_cp.kernels import (
    PolynomialKernel as PolynomialKernel,
)
from online_cp.martingale import (
    BetaKernel as BetaKernel,
)
from online_cp.martingale import (
    BetaMLE as BetaMLE,
)
from online_cp.martingale import (
    BetaMoments as BetaMoments,
)
from online_cp.martingale import (
    CompositeJumper as CompositeJumper,
)
from online_cp.martingale import (
    ExpertAggregationStrategy as ExpertAggregationStrategy,
)
from online_cp.martingale import (
    FixedStrategy as FixedStrategy,
)
from online_cp.martingale import (
    GaussianKDE as GaussianKDE,
)
from online_cp.martingale import (
    ParticleFilterStrategy as ParticleFilterStrategy,
)
from online_cp.martingale import (
    PluginMartingale as PluginMartingale,
)
from online_cp.martingale import (
    SimpleJumper as SimpleJumper,
)
from online_cp.martingale import (
    SimpleMixtureMartingale as SimpleMixtureMartingale,
)
from online_cp.martingale import (
    SleeperStayer as SleeperStayer,
)
from online_cp.martingale import (
    SleeperDrifter as SleeperDrifter,
)
from online_cp.martingale import (
    VilleWrapper as VilleWrapper,
)
from online_cp.martingale import (
    CUSUMWrapper as CUSUMWrapper,
)
from online_cp.martingale import (
    ShiryaevRobertsWrapper as ShiryaevRobertsWrapper,
)
from online_cp.martingale import (
    PiecewiseConstantBetting as PiecewiseConstantBetting,
)
from online_cp.regressors import (
    ConformalLassoRegressor as ConformalLassoRegressor,
)
from online_cp.regressors import (
    ConformalNearestNeighboursRegressor as ConformalNearestNeighboursRegressor,
)
from online_cp.regressors import (
    ConformalRidgeRegressor as ConformalRidgeRegressor,
)
from online_cp.regressors import (
    KernelConformalRidgeRegressor as KernelConformalRidgeRegressor,
)
from online_cp.mondrian import (
    MondrianConformalClassifier as MondrianConformalClassifier,
)
from online_cp.mondrian import (
    MondrianConformalRegressor as MondrianConformalRegressor,
)
from online_cp.plotting import (
    plot_coverage as plot_coverage,
)
from online_cp.plotting import (
    plot_detector as plot_detector,
)
from online_cp.plotting import (
    plot_intervals as plot_intervals,
)
from online_cp.plotting import (
    plot_martingale as plot_martingale,
)
from online_cp.plotting import (
    plot_set_sizes as plot_set_sizes,
)
from online_cp.venn import (
    VennPrediction as VennPrediction,
)
from online_cp.venn import (
    VennAbersPredictor as VennAbersPredictor,
)
from online_cp.venn import (
    NearestNeighboursVennPredictor as NearestNeighboursVennPredictor,
)
from online_cp.venn import (
    brier_point as brier_point,
)
from online_cp.venn import (
    log_loss_point as log_loss_point,
)
