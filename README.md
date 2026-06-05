# online-cp — Online Conformal Prediction

[![PyPI version](https://badge.fury.io/py/online-cp.svg)](https://badge.fury.io/py/online-cp)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://egonmedhatten.github.io/online-cp/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/egonmedhatten/online-cp/actions/workflows/test.yml/badge.svg)](https://github.com/egonmedhatten/online-cp/actions/workflows/test.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egonmedhatten/online-cp/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fquickstart.ipynb)

A Python library for **online conformal prediction** — valid prediction sets and intervals with guaranteed coverage, updated one example at a time.

## Quick start

```bash
pip install online-cp

# Optional: install with numba for faster Lasso homotopy and KDE
pip install online-cp[fast]
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


Other estimators may still run, but should be treated as unverified and are
expected to be less stable or slower in this transductive setting.

### Multi-level predictions

All predictors support multiple significance levels in a single call:

```python
result = cp.predict(x, epsilon=[0.01, 0.05, 0.1, 0.2])
result[0.1]          # prediction at ε=0.1
result.levels        # [0.01, 0.05, 0.1, 0.2]
result.coverage(y)   # {0.01: True, 0.05: True, 0.1: True, 0.2: False}
```

### Venn-Abers predictor

Calibrated probability predictions for binary classification via the full/transductive Venn-Abers predictor (Algorithm 6.1, ALRW2 §6.4). The prediction is a **multiprobability pair** $(p^0, p^1)$ — not a point estimate:

```python
from online_cp import VennAbersPredictor, log_loss_point, brier_point

vap = VennAbersPredictor(scorer="ridge", a=1.0)
vap.learn_initial_training_set(X_train, y_train)

pred = vap.predict(x_new)
print(pred.p0, pred.p1)  # the multiprobability pair IS the prediction

# For decision-making, merge into a single probability:
log_loss_point(pred.p0, pred.p1)  # minimises log loss
brier_point(pred.p0, pred.p1)     # minimises Brier loss

vap.learn_one(x_new, y_new)
```

Also supports k-NN scoring (`VennAbersPredictor(scorer="knn", k=5)`) and SVM scoring (`VennAbersPredictor(scorer="svm", kernel="rbf", sigma=1.0, C=10.0)`).

### Nearest-neighbours Venn predictor

Taxonomy-based Venn predictor using the k-NN voting taxonomy (ALRW2 §6.2). Supports **binary and multiclass** labels. The taxonomy categorises each example by the number of same-class labels among its k nearest neighbours:

```python
from online_cp import NearestNeighboursVennPredictor, log_loss_point

# Binary
vp = NearestNeighboursVennPredictor(k=3)
vp.learn_initial_training_set(X_train, y_train)
pred = vp.predict(x_new)
print(pred.p0, pred.p1)            # multiprobability pair
log_loss_point(pred.p0, pred.p1)   # merge for decisions

# Multiclass (label_space inferred from data, or pass explicitly)
vp = NearestNeighboursVennPredictor(k=5, label_space=[0, 1, 2])
vp.learn_initial_training_set(X_train, y_train)
pred = vp.predict(x_new)
print(pred.point)                  # calibrated class probabilities

vp.learn_one(x_new, y_new)
```

### Mondrian conformal prediction

Group-conditional coverage via a single pooled model with category-filtered calibration:

```python
from online_cp import ConformalRidgeRegressor
from online_cp.mondrian import MondrianConformalRegressor

wrapper = MondrianConformalRegressor(
    base_model=ConformalRidgeRegressor(a=1.0),
    category_fn=lambda x: "high" if x[0] > 0 else "low",
)
wrapper.learn_initial_training_set(X_train, y_train)

# Guarantees: P(y ∈ Γ | category = k) ≥ 1 − ε  for each k
interval = wrapper.predict(x_new, epsilon=0.1)
```

### Streaming evaluation

River-style test-then-train loop with composable metrics:

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

Supports streaming iterables and conditional learning:

```python
from online_cp.evaluate import iter_progressive_val

# Stream from any iterable of (x, y) pairs
stream = ((x, y) for x, y in data_source)
for snapshot in iter_progressive_val(model, stream, epsilon=0.1, step=50):
    print(snapshot)  # periodic metric checkpoints

# Conditional learning: only learn from some examples
progressive_val(model, X, y, learn=lambda i, x, y: i % 2 == 0)
```

### Plotting utilities

```python
from online_cp.plotting import plot_coverage, plot_martingale, plot_intervals

plot_coverage(error_rate_metric, nominal=0.9)
plot_martingale(martingale)
plot_intervals(y_true, intervals)
```

### Conformal test martingales

Test the exchangeability assumption online:

```python
from online_cp import PluginMartingale, GaussianKDE

martingale = PluginMartingale(betting_strategy=GaussianKDE())
for i in range(n_train, N):
    p = cp.compute_p_value(X[i], y[i])
    martingale.update(p)
    cp.learn_one(X[i], y[i])

# If martingale grows large → evidence against exchangeability
print(f"Martingale: {martingale.M:.2f}")
```

Use `VilleWrapper` to turn any martingale into a statistical test with Ville's inequality ($P(\exists n: M_n \geq c) \leq 1/c$):

```python
from online_cp import SimpleJumper, VilleWrapper

ville = VilleWrapper(SimpleJumper(), threshold=20)  # 5% significance
for p in p_values:
    ville.update(p)
    if ville.rejected:
        print(f"Exchangeability rejected at observation {ville.martingale.n}")
        break
```

### Conformal predictive decision making

Make optimal decisions under uncertainty using conformal predictive distributions:

```python
from online_cp import UtilityFunction, ConformalPredictiveDecisionMaker

# Define a utility: U(x, y, decision) -> payoff
U = UtilityFunction(
    lambda x, y, d: -abs(y - d),  # penalise distance from decision to outcome
    decisions=[0.0, 0.5, 1.0],
)

cdm = ConformalPredictiveDecisionMaker(U, a=1.0)
cdm.learn_initial_training_set(X_train, y_train)

for i in range(len(X_test)):
    decision = cdm.predict(X_test[i])       # maximises expected utility
    cdm.learn_one(X_test[i], y_test[i])
```

## Features

| Module | Description |
|--------|-------------|
| **Regressors** | `ConformalRidgeRegressor`, `KernelConformalRidgeRegressor`, `ConformalNearestNeighboursRegressor`, `ConformalLassoRegressor` |
| **Classifiers** | `ConformalNearestNeighboursClassifier`, `ConformalSupportVectorMachine` |
| **Venn Predictors** | `VennAbersPredictor` (ridge, k-NN, SVM scoring), `NearestNeighboursVennPredictor` (binary & multiclass), `log_loss_point`, `brier_point` |
| **Mondrian CP** | `MondrianConformalRegressor`, `MondrianConformalClassifier` — group-conditional coverage |
| **Predictive Systems** | `RidgePredictionMachine`, `KernelRidgePredictionMachine`, `NearestNeighboursPredictionMachine`, `DempsterHillConformalPredictiveSystem` |
| **Decision Making** | `ConformalPredictiveDecisionMaker`, `UtilityFunction`, `cps_decision`, `venn_decision` |
| **Metrics** | `ErrorRate`, `ObservedExcess`, `ObservedFuzziness`, `SetSize`, `IntervalWidth`, `WinklerScore`, `CRPS` |
| **Evaluation** | `progressive_val()`, `iter_progressive_val()` — streaming test-then-train |
| **Plotting** | `plot_coverage`, `plot_martingale`, `plot_detector`, `plot_intervals`, `plot_set_sizes` |
| **Martingales** | `PluginMartingale`, `SimpleMixtureMartingale`, `SimpleJumper`, `CompositeJumper`, `SleeperStayer`, `SleeperDrifter` |  
| **Detection Wrappers** | `VilleWrapper`, `CUSUMWrapper`, `ShiryaevRobertsWrapper` |
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

Start with [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for a 5-minute introduction ([run on Binder](https://mybinder.org/v2/gh/egonmedhatten/online-cp/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fquickstart.ipynb)), then see [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) for a comprehensive walkthrough covering regression, classification, Mondrian CP, conformal predictive systems, martingales, and evaluation.

## Links

* [Documentation](https://egonmedhatten.github.io/online-cp/)
* [online-cp on GitHub][online-cp-on-github]
* [online-cp on PyPI][online-cp-on-pypi]
* [Changelog](CHANGELOG.md)

## Looking for Inductive (Split) Conformal Prediction?

This package focuses on **online** (transductive) conformal prediction, where models are updated one example at a time and predictions are valid without a separate calibration set.

For **inductive** (split) conformal prediction — where you have a fixed pre-trained model and a held-out calibration set — we recommend the excellent [`crepes`](https://github.com/henrikbostrom/crepes) package.


## References

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World* (2nd ed). Springer Nature, 2022.

Jing Lei. Fast exact conformalization of the Lasso using piecewise linear homotopy. *Biometrika*, 106(4):751–767, 2019.

Vladimir Vovk and Claus Bendtsen. Conformal predictive decision making. *Proceedings of Machine Learning Research*, 91:52–62, 2018.


[`notebooks/tutorial.ipynb`]: https://github.com/egonmedhatten/online-cp/blob/main/notebooks/tutorial.ipynb
[online-cp-on-pypi]: https://pypi.org/project/online-cp/
[online-cp-on-github]: https://github.com/egonmedhatten/online-cp

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