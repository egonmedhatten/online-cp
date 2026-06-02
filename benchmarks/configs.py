"""Model configurations for benchmarking.

Each configuration is a dict with keys:
  - "name": display name for tables
  - "factory": callable() -> model instance
  - "task": "classification" | "regression" | "cps" | "venn"
  - "binary_only": bool (some models only work with 2 classes)
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def get_configs(task="all"):
    """Return model configurations for the given task.

    Parameters
    ----------
    task : str
        One of "classification", "regression", "cps", "venn", or "all".

    Returns
    -------
    list[dict]
    """
    configs = []

    if task in ("regression", "all"):
        configs.extend(_regression_configs())
    if task in ("classification", "all"):
        configs.extend(_classification_configs())
    if task in ("cps", "all"):
        configs.extend(_cps_configs())
    if task in ("venn", "all"):
        configs.extend(_venn_configs())

    return configs


def _regression_configs():
    from online_cp import ConformalRidgeRegressor, KernelConformalRidgeRegressor, GaussianKernel

    return [
        {
            "name": "Ridge(a=1)",
            "factory": lambda: ConformalRidgeRegressor(a=1.0),
            "task": "regression",
            "binary_only": False,
        },
        {
            "name": "KernelRidge(σ=1)",
            "factory": lambda: KernelConformalRidgeRegressor(kernel=GaussianKernel(sigma=1.0), a=1.0),
            "task": "regression",
            "binary_only": False,
        },
    ]


def _classification_configs():
    from online_cp import (
        ConformalNearestNeighboursClassifier,
        ConformalSupportVectorMachine,
    )
    from online_cp.classifiers import ConformalClassifierWrapper

    return [
        {
            "name": "KNN(k=3)",
            "factory": lambda: ConformalNearestNeighboursClassifier(k=3),
            "task": "classification",
            "binary_only": False,
        },
        {
            "name": "KNN(k=7)",
            "factory": lambda: ConformalNearestNeighboursClassifier(k=7),
            "task": "classification",
            "binary_only": False,
        },
        {
            "name": "SVM(rbf)",
            "factory": lambda: ConformalSupportVectorMachine(kernel="rbf", C=1.0, sigma=1.0),
            "task": "classification",
            "binary_only": False,
        },
        {
            "name": "Wrapper(LR)",
            "factory": lambda: ConformalClassifierWrapper(
                LogisticRegression(max_iter=200, solver="lbfgs"),
            ),
            "task": "classification",
            "binary_only": False,
        },
    ]


def _cps_configs():
    from online_cp import RidgePredictionMachine, KernelRidgePredictionMachine, NearestNeighboursPredictionMachine, GaussianKernel

    return [
        {
            "name": "RidgeCPS(a=1)",
            "factory": lambda: RidgePredictionMachine(a=1.0),
            "task": "cps",
            "binary_only": False,
        },
        {
            "name": "KernelCPS(σ=1)",
            "factory": lambda: KernelRidgePredictionMachine(kernel=GaussianKernel(sigma=1.0), a=1.0),
            "task": "cps",
            "binary_only": False,
        },
        {
            "name": "NN-CPS(k=5)",
            "factory": lambda: NearestNeighboursPredictionMachine(k=5),
            "task": "cps",
            "binary_only": False,
        },
    ]


def _venn_configs():
    from online_cp import VennAbersPredictor, NearestNeighboursVennPredictor

    return [
        {
            "name": "VA(ridge)",
            "factory": lambda: VennAbersPredictor(scorer="ridge", a=1.0),
            "task": "venn",
            "binary_only": True,
        },
        {
            "name": "VA(knn,k=5)",
            "factory": lambda: VennAbersPredictor(scorer="knn", k=5),
            "task": "venn",
            "binary_only": True,
        },
        {
            "name": "VennNN(k=5)",
            "factory": lambda: NearestNeighboursVennPredictor(k=5),
            "task": "venn",
            "binary_only": False,
        },
    ]
