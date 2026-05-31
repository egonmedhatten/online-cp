# Quickstart

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egonmedhatten/online-cp/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fquickstart.ipynb)

For an interactive introduction, [open the quickstart notebook on Binder](https://mybinder.org/v2/gh/egonmedhatten/online-cp/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fquickstart.ipynb).

For the full walkthrough covering regression, classification, Venn-Abers calibrated probabilities, Mondrian CP, conformal predictive systems, martingales, and evaluation, see the [tutorial notebook](https://mybinder.org/v2/gh/egonmedhatten/online-cp/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Ftutorial.ipynb).

## Installation

```bash
pip install online-cp
```

## Minimal Example — Regression

```python
import numpy as np
from online_cp import ConformalRidgeRegressor, ErrorRate, IntervalWidth
from online_cp.evaluate import progressive_val

X = np.random.randn(500, 5)
y = X @ np.array([1, 2, 0, -1, 0.5]) + 0.3 * np.random.randn(500)

model = ConformalRidgeRegressor(a=1.0)
metric = ErrorRate() + IntervalWidth()
progressive_val(model, X, y, epsilon=0.1, metric=metric)
print(metric)
```

## Minimal Example — Classification

```python
from sklearn.datasets import load_iris
from online_cp import ConformalNearestNeighboursClassifier, ErrorRate, SetSize
from online_cp.evaluate import progressive_val

X, y = load_iris(return_X_y=True)
model = ConformalNearestNeighboursClassifier(k=5)
metric = ErrorRate() + SetSize()
progressive_val(model, X, y, epsilon=0.05, metric=metric)
print(metric)
```

## Minimal Example — Calibrated Probabilities (Venn-Abers)

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from online_cp import VennAbersPredictor, log_loss_point

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(42)
perm = rng.permutation(len(y))
X, y = X[perm], y[perm]

vap = VennAbersPredictor(scorer="ridge", a=1.0)
vap.learn_initial_training_set(X[:50], y[:50])

# Stream test points — the prediction is the pair (p0, p1)
for i in range(50, 60):
    pred = vap.predict(X[i])
    print(f"y={y[i]}  prediction=({pred.p0:.3f}, {pred.p1:.3f})")
    vap.learn_one(X[i], y[i])

# To merge into a single probability (e.g. for decisions), use:
# log_loss_point(pred.p0, pred.p1) or brier_point(pred.p0, pred.p1)
```
