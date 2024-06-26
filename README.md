# OnlineConformalPrediction

This project is an implementation of Online Conformal Prediction.

For now, take a look at [`example.ipynb`](example.ipynb) to see how to use the library.


## Quick start

Import the library and create a regressor:

```py
from CRR import ConformalRidgeRegressor

cp = ConformalRidgeRegressor()
```

Predict the output for an object:

```py
>>> cp.predict(np.array([10,20]), epsilon=0.1, bounds='both')
(-inf, inf)
```

We know that the result value is between `-inf` and `inf` with `90%` certainty.

We learn the actual output for the above:

```py
>>> cp.learn_label(30)
```

We predict again:

```py
>>> cp.predict(np.array([20,10]), epsilon=0.1, bounds='both')
(-inf, inf)
```

We learn the actual output:

```py
>>> cp.learn_label(30)
```

...
After a while, you should start seeing useful predictions:

```py
>>> cp.predict(np.array([6,7]), epsilon=0.1, bounds='both')
(10, 14)
```
