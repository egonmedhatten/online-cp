# online-cp -- Online Conformal Prediction

[![online-cp's Build Status][build-status]][build-log]
[![online-cp on PyPI][pypi-version]][online-cp-on-pypi]

This project is an implementation of Online Conformal Prediction.

For now, take a look at [`example.ipynb`][] to see how to use the library.


## Quick start

The `online-cp` package is [available on PyPI][online-cp-on-pypi], to install just:

```bash
pip install online-cp
```

Let's create a dataset with noisy evaluations of the function _f(x₁, x₂) = x₁ + x₂_.

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
cp = ConformalRidgeRegressor(epsilon=0.1)
```

To predict, simply do
```py
cp.predict(X[0])
(-inf, inf)
```
The output is non-informative since we have not learned anything yet. The parameter `epsilon` is the significance level.

Alternative 1: Learn the dataset sequentially online, and make predictions as we go. In order to output nontrivial prediction at significance level `epsilon=0.1`, we need to have learned at least 20 examples.

```py
for x, y in zip(X[-1], Y[-1]):
    print(f'Prediction set: {cp.predict(x)}')
    cp.learn_one(x, y)
```

In the online setting, we first observe the object _x_, which is used to make a prediction, only then to observe the label _y_. The output will be `(inf, inf)` for the first 19 predictions, after which we will typically see meaningful prediction sets. The snippet above learned all but the last example. To predict it, do (your output may not be exactly the same, as the dataset depends on the random seed).

```py
cp.predict(X[-1])
(0.029643344144500712, 0.34909922671253196)
```

The prediction set is the closed interval whose boundaries are indicated by the output.

Alternative 2: Learn an initial training set offline, and predict e.g. only the last example

```py
cp = ConformalRidgeRegressor()
cp.learn_initial_training_set(X[:-1], Y[:-1])
cp.predict(X[-1])
(0.8748194061248175, 1.3357383729107446)
```

Furhter examples can be found in the notebooks, e.g. [`example.ipynb`][]. Current functionality includes
* Conformal regression
* Conformal classification
* Testing exchangeability through conformal test martignales


## Links

* [online-cp on GitHub][online-cp-on-github]
* [online-cp on PyPI][online-cp-on-pypi]


## References

The main reference for Conformal Prediction is the book

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World (2nd ed). Springer Nature, 2022.


[`example.ipynb`]: https://github.com/egonmedhatten/online-cp/blob/main/notebooks/example.ipynb
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