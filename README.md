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


## Future considerations

### Properties of CPs?
* Should we keep track of errors internally in the parent class? 
* Should we store the average interval size?
* For classifiers; should we store the efficiency metrics?

### Linear regression
We will initally focus on regression, but online classification is actually easier. A simple class that uses e.g. scikit-learn classifiers to define nonconformity measure could be easily implemented. 

There are at least three commonly used regularisations used in linear regression, all of which are compatible with the kernel trick. 
* $L1$ (Lasso)
* $L2$ (Ridge)
* Linear combination of the above (Elastic net)

All of these can be conformalized, and at least Ridge can also be used in conformal predictive systems (CPS).

Another relatively simple regressor is the k-nearest neighbours algorithm, which is very flexible as it can use arbitrary distances. It is particularly interesting in the CPS setting. The distance can be measured in feature space as defined by a kernel.

Ridge and KNN are described in detail in Algorithmic Learning in a Random World. Lasso and Elastic net are conformalised in the paper Fast Exact Conformalization of Lasso using Piecewise Linear Homotopy, but I am unaware of any extention to CPS. 

### Teaching schedule
Section 3.3 in Algorithmic Learning in a Radnom World deals with, so called, weak teachers. In the pure online mode, labels arrive immediately after a predition is made. This makes little sense in practice. The notion of a teaching schedule formalises this, and makes the relevant validity guarantees clear. There are three types of validity; weak, strong, and iterated logartihm validity. 

There may be settings where the user wants to specify a teaching schedule beforehand, to guarantee some property of validity. It may also be the case that the teaching schedule is implied by the usage, and it would then be useful to know if the resulting prediciton sets are valid.

A teaching schedule also serves as documentation of what has been done, which could be useful in practice.

## Todo
* Should we add some scaler? Don't know if it is neccesary for Ridge
* Possibly add a class MimoConformalRidgeRegressor
* Add CPS version of ridge regressor?
* Possibly add a TeachingSchedule?
* Possibly add ACI, both for single, and MIMO CRR?
* Add references to papers and books to README
* Add k-NN regressor and CPS

# References
How to cite papers? I think I have seen it in some repos.