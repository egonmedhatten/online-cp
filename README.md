# online-cp — Online Conformal Prediction

[![online-cp's Build Status][build-status]][build-log]
[![online-cp on PyPI][pypi-version]][online-cp-on-pypi]

A Python library for **online conformal prediction** — valid prediction sets and intervals with guaranteed coverage, updated one example at a time.

## Quick start

```bash
pip install online-cp
```

### Conformal regression

```python
import numpy as np
from online_cp import ConformalRidgeRegressor

# Synthetic data: f(x) = x₁ + x₂ + noise
N = 100
X = np.random.uniform(0, 1, (N, 2))
y = X.sum(axis=1) + np.random.normal(0, 0.1, N)

# Create regressor and learn an initial training set
cp = ConformalRidgeRegressor(a=1.0, epsilon=0.1)
cp.learn_initial_training_set(X[:50], y[:50])

# Online loop: predict then learn
for i in range(50, N):
    interval = cp.predict(X[i], epsilon=0.1)
    print(f"Prediction interval: {interval}")
    cp.learn_one(X[i], y[i])
```

### Conformal classification

```python
from online_cp import ConformalNearestNeighboursClassifier

cp = ConformalNearestNeighboursClassifier(k=3, label_space=np.array([0, 1, 2]))
cp.learn_initial_training_set(X_train, y_train)

Gamma = cp.predict(x_new, epsilon=0.1)
print(f"Prediction set: {Gamma}")  # e.g. array([1])
```

### Multi-level predictions

All predictors support multiple significance levels in a single call:

```python
result = cp.predict(x, epsilon=[0.01, 0.05, 0.1, 0.2])
result[0.1]          # prediction at ε=0.1
result.levels        # [0.01, 0.05, 0.1, 0.2]
result.coverage(y)   # {0.01: True, 0.05: True, 0.1: True, 0.2: False}
```

### Evaluation

Composable metrics and a standalone evaluation loop:

```python
from online_cp import ErrorRate, IntervalWidth, WinklerScore
from online_cp.evaluate import progressive_val

metric = ErrorRate() + IntervalWidth() + WinklerScore()
progressive_val(model, X_test, y_test, epsilon=0.1, metric=metric)
print(metric)
# ErrorRate: 0.0900
# IntervalWidth: 0.4123
# WinklerScore: 0.5012
```

### Conformal test martingales

Test the exchangeability assumption online:

```python
from online_cp import PluginMartingale, GaussianKDE

martingale = PluginMartingale(betting_strategy=GaussianKDE())
for i in range(n_train, N):
    p = cp.compute_p_value(X[i], y[i])
    martingale.update_martingale_value(p)
    cp.learn_one(X[i], y[i])

# If martingale grows large → evidence against exchangeability
print(f"Martingale: {martingale.M:.2f}")
```

## Features

| Module | Description |
|--------|-------------|
| **Regressors** | `ConformalRidgeRegressor`, `KernelConformalRidgeRegressor`, `ConformalLassoRegressor` |
| **Classifiers** | `ConformalNearestNeighboursClassifier`, `ConformalSupportVectorMachine` |
| **Predictive Systems** | `RidgePredictionMachine`, `KernelRidgePredictionMachine`, `NearestNeighboursPredictionMachine`, `DempsterHillConformalPredictiveSystem` |
| **Metrics** | `ErrorRate`, `ObservedExcess`, `ObservedFuzziness`, `SetSize`, `IntervalWidth`, `WinklerScore`, `CRPS` |
| **Evaluation** | `progressive_val()`, `iter_progressive_val()` |
| **Martingales** | `PluginMartingale`, `SimpleMixtureMartingale`, `SimpleJumper`, `CompositeJumper`, `OnionMartingale` |
| **Kernels** | `GaussianKernel`, `LinearKernel`, `PolynomialKernel`, `PeriodicKernel`, `LinearCombinationKernel` |

## API pattern

All models follow the same interface:

```python
model = ConformalRidgeRegressor(a=1.0, epsilon=0.1)

# Learn
model.learn_initial_training_set(X_train, y_train)  # batch
model.learn_one(x, y)                                # online

# Predict
Gamma = model.predict(x, epsilon=0.1)      # single level
result = model.predict(x, epsilon=[...])   # multi-level

# P-value
p = model.compute_p_value(x, y)
```

## Tutorial

See [`notebooks/tutorial.ipynb`][] for a comprehensive walkthrough covering regression, classification, conformal predictive systems, evaluation, and test martingales.

## Links

* [online-cp on GitHub][online-cp-on-github]
* [online-cp on PyPI][online-cp-on-pypi]


## References

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World* (2nd ed). Springer Nature, 2022.


[`notebooks/tutorial.ipynb`]: https://github.com/egonmedhatten/online-cp/blob/main/notebooks/tutorial.ipynb
[online-cp-on-pypi]: https://pypi.org/project/online-cp/
[online-cp-on-github]: https://github.com/egonmedhatten/online-cp
[pypi-version]: https://img.shields.io/pypi/v/online-cp
[build-log]:    https://github.com/egonmedhatten/online-cp/actions/workflows/test.yml
[build-status]: https://github.com/egonmedhatten/online-cp/actions/workflows/test.yml/badge.svg

## 📄 Citing `online-cp`

If you use `online-cp` in your work, please cite the following paper. It helps support the ongoing development of this package.

### BibTeX

For users of LaTeX and bibliography managers, please use this BibTeX entry:

```bibtex
@InProceedings{pmlr-v266-hallberg-szabadvary25a,
  title = 	 {online-cp: a Python Package for Online Conformal Prediction, Conformal Predictive Systems and Conformal Test Martingales},
  author =       {Hallberg Szabadv\'{a}ry, Johan and L\"{o}fstr\"{o}m, Tuwe and Matela, Rudy},
  booktitle = 	 {Proceedings of the Fourteenth Symposium on Conformal and Probabilistic Prediction with Applications},
  pages = 	 {595--614},
  year = 	 {2025},
  editor = 	 {Nguyen, Khuong An and Luo, Zhiyuan and Papadopoulos, Harris and L\"ofstr\"om, Tuwe and Carlsson, Lars and Bostr\"om, Henrik},
  volume = 	 {266},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--12 Sep},
  publisher =    {PMLR},
  pdf = 	 {[https://raw.githubusercontent.com/mlresearch/v266/main/assets/hallberg-szabadvary25a/hallberg-szabadvary25a.pdf](https://raw.githubusercontent.com/mlresearch/v266/main/assets/hallberg-szabadvary25a/hallberg-szabadvary25a.pdf)},
  url = 	 {[https://proceedings.mlr.press/v266/hallberg-szabadvary25a.html](https://proceedings.mlr.press/v266/hallberg-szabadvary25a.html)}
}
```

### Formatted Citation (APA Style)

Hallberg Szabadváry, J., Löfström, T., & Matela, R. (2025). online-cp: a Python Package for Online Conformal Prediction, Conformal Predictive Systems and Conformal Test Martingales. In K. A. Nguyen, Z. Luo, H. Papadopoulos, T. Löfström, L. Carlsson, & H. Boström (Eds.), *Proceedings of the Fourteenth Symposium on Conformal and Probabilistic Prediction with Applications* (Vol. 266, pp. 595–614). PMLR. https://proceedings.mlr.press/v266/hallberg-szabadvary25a.html