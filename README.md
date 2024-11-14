# online-cp -- Online Conformal Prediction

This project is an implementation of Online Conformal Prediction.

For now, take a look at [`example.ipynb`](example.ipynb) to see how to use the library.


## Quick start

The `online-cp` package is available on PyPI, to install just:

```bash
pip install online-cp
```

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
from online_cp import ConformalRidgeRegressor
cp = ConformalRidgeRegressor()
```

To predict, simply do
```py
cp.predict(X[0], epsilon=0.1)
(-inf, inf)
```
The output is non-informative since we have not learned anything yet. The parameter `epsilon` is the significance level.

Alternative 1: Learn the dataset sequentially online, and make predictions as we go. In order to output nontrivial prediction at significance level `epsilon=0.1`, we need to have learned at least 20 examples.
```py
for x, y in zip(X[-1], Y[-1]):
    print(f'Prediction set: {cp.predict(x, epsilon=0.1)}')
    cp.learn_one(x, y)
```
In the online setting, we first observe the object $x$, which is used to make a prediction, only then to observe the label $y$. The output will be ```(inf, inf)``` for the first 19 predictions, after which we will typically see meaningful prediction sets. The snippet above learned all but the last example. To predict it, do (your output may not be exactly the same, as the dataset depends on the random seed).
```py
cp.predict(X[-1], epsilon=0.1)
(0.029643344144500712, 0.34909922671253196)
```
The prediction set is the closed interval whose boundaries are indicated by the output.

Alternative 2: Learn an initial training set offline, and predict e.g. only the last example
```py
cp = ConformalRidgeRegressor()
cp.learn_initial_training_set(X[:-1], Y[:-1])
cp.predict(X[-1], epsilon=0.1)
(0.8748194061248175, 1.3357383729107446)
```
Furhter examples can be found in the notebooks, e.g. [`example.ipynb`](example.ipynb). Current functionality includes
* Conformal regression
* Conformal classification
* Testing exchangeability through conformal test martignales

# References
The main reference for Conformal Prediction is the book

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World (2nd ed). Springer Nature, 2022.
