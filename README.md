# OnlineConformalPrediction

This project is an implementation of Online Conformal Prediction.

For now, take a look at [`example.ipynb`](example.ipynb) to see how to use the library.


## Quick start
Let's create a dataset with noisy evaluations of the function $f(x_1, x_2) = x_1 + x_2$.

```py
import numpy as np
N = 30
X = np.random.uniform(0, 1, (N, 2))
y = X.sum(axis=1) + np.random.normal(0, 0.1, N)
cp.learn_initial_training_set(X, y)
```

Import the library and create a regressor:

```py
from CRR import ConformalRidgeRegressor
cp = ConformalRidgeRegressor()
```

Alternative 1: Learn the whole dataset online
```py
cp.learn_initial_training_set(X, y)
```

Predict an object (your output may not be exactly the same, as the dataset depends on the random seed).
```py
cp.predict(np.array([0.5, 0.5]), epsilon=0.1, bounds='both')
(0.8065748777057368, 1.2222461945130274)
```
The prediction set is the closed interval whose boundaries are indicated by the output.

Alternative 2: Learn the dataset sequentially online, and make predictions as we go. In order to output nontrivial prediction at significance level $\epsilon=0.1$, we need to have learned at least 20 examples.
```py
cp = ConformalRidgeRegressor()
for i, (obj, lab) in enumerate(zip(X, y)):
    print(cp.predict(obj, epsilon=0.1, bounds='both'))
    cp.learn_one(obj, lab)
```
The output will be ```(inf, inf)``` for the first 19 predictions, after which we will typically see meaningful prediction sets.
