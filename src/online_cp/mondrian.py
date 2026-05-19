"""Mondrian conformal prediction.

Provides ``MondrianWrapper`` — a wrapper that takes any conformal predictor
and a category function, maintaining a separate predictor instance per category
to achieve group-conditional coverage guarantees.

The category function is a callable ``(x) -> hashable`` that maps each
observation to a category label. Categories are discovered online as new
labels appear.

Example
-------
>>> from online_cp import ConformalRidgeRegressor
>>> from online_cp.mondrian import MondrianWrapper
>>>
>>> # Category based on first feature being positive/negative
>>> wrapper = MondrianWrapper(
...     base_model=ConformalRidgeRegressor(a=1.0),
...     category_fn=lambda x: "pos" if x[0] > 0 else "neg",
... )
>>> wrapper.learn_initial_training_set(X_train, y_train)
>>> interval = wrapper.predict(x_test, epsilon=0.1)
"""

import copy
from collections import defaultdict

import numpy as np

__all__ = ["MondrianWrapper"]


class MondrianWrapper:
    """Mondrian conformal predictor: group-conditional coverage via partitioning.

    Wraps any conformal predictor (regressor or classifier) and routes each
    observation to a category-specific copy. Each group maintains its own
    nonconformity scores and calibration data, yielding group-conditional
    validity.

    Parameters
    ----------
    base_model : conformal predictor
        A template model instance. Will be deep-copied for each new category.
        Must implement ``predict(x, ...)`` and ``learn_one(x, y)``.
    category_fn : callable
        A function ``(x) -> hashable`` that maps an input to its category.
        Categories are discovered online.
    min_group_size : int, optional
        Minimum number of training examples in a group before making
        group-conditional predictions. Below this, predictions use the
        pooled model (if available). Default: 1.

    Attributes
    ----------
    models_ : dict
        Mapping from category label to the per-group model instance.
    counts_ : dict
        Number of training observations in each group.
    """

    def __init__(self, base_model, category_fn, *, min_group_size=1):
        self.base_model = base_model
        self.category_fn = category_fn
        self.min_group_size = min_group_size
        self.models_ = {}
        self.counts_ = defaultdict(int)

    def _get_model(self, category):
        """Get or create the model for a category."""
        if category not in self.models_:
            self.models_[category] = copy.deepcopy(self.base_model)
        return self.models_[category]

    def learn_initial_training_set(self, X, y):
        """Route each training example to its category-specific model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Group examples by category
        groups = defaultdict(lambda: ([], []))
        for x_i, y_i in zip(X, y):
            cat = self.category_fn(x_i)
            groups[cat][0].append(x_i)
            groups[cat][1].append(y_i)

        # Initialize each group's model
        for cat, (X_group, y_group) in groups.items():
            model = self._get_model(cat)
            X_arr = np.array(X_group)
            y_arr = np.array(y_group)
            model.learn_initial_training_set(X_arr, y_arr)
            self.counts_[cat] = len(y_group)

    def predict(self, x, **kwargs):
        """Predict using the category-specific model.

        Parameters
        ----------
        x : array-like
            Input observation.
        **kwargs
            Passed to the underlying model's ``predict`` method
            (e.g. ``epsilon``, ``return_p_values``).

        Returns
        -------
        Prediction set/interval from the group-specific model.

        Raises
        ------
        KeyError
            If the category has not been seen during training and
            ``min_group_size > 0``.
        """
        cat = self.category_fn(x)
        if cat not in self.models_:
            raise KeyError(
                f"Category '{cat}' not seen during training. "
                f"Call learn_one() with examples from this category first."
            )
        model = self.models_[cat]
        if self.counts_[cat] < self.min_group_size:
            raise ValueError(
                f"Category '{cat}' has only {self.counts_[cat]} examples "
                f"(min_group_size={self.min_group_size})."
            )
        return model.predict(x, **kwargs)

    def learn_one(self, x, y, **kwargs):
        """Route a single example to its category-specific model.

        Parameters
        ----------
        x : array-like
            Input observation.
        y : scalar
            True label/response.
        **kwargs
            Passed to the underlying model's ``learn_one`` method.
        """
        cat = self.category_fn(x)
        model = self._get_model(cat)
        model.learn_one(x, y, **kwargs)
        self.counts_[cat] += 1

    @property
    def categories(self):
        """Return the set of discovered categories."""
        return set(self.models_.keys())

    def get_model(self, category):
        """Access the model for a specific category.

        Parameters
        ----------
        category : hashable
            Category label.

        Returns
        -------
        The conformal predictor instance for this category.
        """
        return self.models_[category]

    def __repr__(self):
        n_cats = len(self.models_)
        total = sum(self.counts_.values())
        return (
            f"MondrianWrapper(n_categories={n_cats}, "
            f"n_total={total}, "
            f"base={self.base_model.__class__.__name__})"
        )
