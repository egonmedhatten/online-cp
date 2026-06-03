"""Mondrian conformal prediction.

Provides ``MondrianConformalRegressor`` and ``MondrianConformalClassifier`` —
wrappers that provide **group-conditional** coverage guarantees by partitioning
the calibration step (not the model) by category.

A single underlying model is trained on ALL data (pooled). Only the calibration
(p-value / quantile computation) is restricted to examples from the same category
as the test point. This is the theoretically correct Mondrian CP construction
from Vovk et al. (2005).

The category function is a callable ``(x) -> hashable`` that maps each
observation to a category label.

Example
-------
>>> import numpy as np
>>> from online_cp import ConformalRidgeRegressor
>>> from online_cp.mondrian import MondrianConformalRegressor
>>>
>>> wrapper = MondrianConformalRegressor(
...     base_model=ConformalRidgeRegressor(a=1.0),
...     category_fn=lambda x: "pos" if x[0] > 0 else "neg",
... )
>>> wrapper.learn_initial_training_set(X_train, y_train)  # doctest: +SKIP
>>> interval = wrapper.predict(x_test, epsilon=0.1)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

import numpy as np
from numpy.typing import NDArray

from online_cp.regressors import (
    ConformalLassoRegressor,
    ConformalPredictionInterval,
    ConformalRegressor,
    ConformalRidgeRegressor,
    KernelConformalRidgeRegressor,
    MultiLevelPredictionInterval,
    _solve_lasso,
)

__all__ = ["MondrianConformalRegressor", "MondrianConformalClassifier"]


# ---------------------------------------------------------------------------
# Mondrian Conformal Regressor
# ---------------------------------------------------------------------------


class MondrianConformalRegressor:
    """Mondrian conformal regressor: group-conditional coverage.

    Wraps a conformal regressor (ridge, kernel ridge, or lasso). A single
    pooled model is trained on ALL data. At prediction time, the nonconformity
    scores are computed using the pooled model, but the calibration step
    (interval construction) only compares against scores from the SAME category.

    This yields valid group-conditional coverage:
        P(y in Gamma(x) | category(x) = k) >= 1 - epsilon  for all k.

    Parameters
    ----------
    base_model : ConformalRidgeRegressor, KernelConformalRidgeRegressor, or ConformalLassoRegressor
        The underlying conformal regressor. Trained on all data (pooled).
    category_fn : callable
        A function ``(x) -> hashable`` that assigns a category to each input.
    """

    def __init__(self, base_model: ConformalRegressor, category_fn: Callable[[NDArray[np.floating[Any]]], Hashable]) -> None:
        if not isinstance(
            base_model,
            (ConformalRidgeRegressor, KernelConformalRidgeRegressor, ConformalLassoRegressor),
        ):
            raise TypeError(
                f"base_model must be a ConformalRidgeRegressor, "
                f"KernelConformalRidgeRegressor, or ConformalLassoRegressor, "
                f"got {type(base_model).__name__}"
            )
        self.base_model = base_model
        self.category_fn = category_fn
        self.categories_ = []

    def learn_initial_training_set(self, X, y):
        """Train the pooled model on all data and record categories."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.base_model.learn_initial_training_set(X, y)
        self.categories_ = [self.category_fn(x_i) for x_i in X]

    def learn_one(self, x, y, **kwargs):
        """Learn a single example in the pooled model and record its category."""
        self.base_model.learn_one(x, y, **kwargs)
        self.categories_.append(self.category_fn(x))

    def predict(self, x: NDArray[np.floating[Any]], epsilon: float | NDArray[np.floating[Any]] | None = None, bounds: str = "both") -> ConformalPredictionInterval | MultiLevelPredictionInterval:
        """Compute the Mondrian conformal prediction interval.

        Uses the pooled model's A, B decomposition but calibrates only
        against same-category training examples.
        """
        if epsilon is None:
            epsilon = self.base_model.epsilon

        cat = self.category_fn(x)

        if isinstance(self.base_model, ConformalRidgeRegressor):
            return self._predict_ridge(x, epsilon, bounds, cat)
        elif isinstance(self.base_model, KernelConformalRidgeRegressor):
            return self._predict_kernel_ridge(x, epsilon, bounds, cat)
        else:
            return self._predict_lasso(x, epsilon, cat)

    def _predict_ridge(self, x, epsilon, bounds, cat):
        model = self.base_model
        if model._safe_size_check(model.X) == 0:
            return self._inf_interval(epsilon)

        X = np.append(model.X, x.reshape(1, -1), axis=0)
        XTXinv = model.XTXinv - (
            model.XTXinv @ np.outer(x, x) @ model.XTXinv
        ) / (1 + x.T @ model.XTXinv @ x)
        A, B = model.compute_A_and_B(X, XTXinv, model.y)

        cat_mask = np.array([c == cat for c in self.categories_] + [True])
        n_cat = int(cat_mask.sum())

        eps_check = max(epsilon) if hasattr(epsilon, "__iter__") else epsilon
        if bounds == "both" and not (eps_check >= 2 / n_cat):
            return self._inf_interval(epsilon)
        if bounds != "both" and not (eps_check >= 1 / n_cat):
            return self._inf_interval(epsilon)

        A_cat = A[cat_mask]
        B_cat = B[cat_mask]
        l_dic, u_dic = ConformalRegressor._vectorised_l_and_u(A_cat, B_cat)
        return self._build_interval(l_dic, u_dic, epsilon, n_cat, bounds)

    def _predict_kernel_ridge(self, x, epsilon, bounds, cat):
        model = self.base_model
        if model.X is None:
            return self._inf_interval(epsilon)

        k = model.kernel(model.X, x).reshape(-1, 1)
        kappa = model.kernel(x, x)
        K = KernelConformalRidgeRegressor._update_K(model.K, k, kappa)
        Kinv = KernelConformalRidgeRegressor._update_Kinv(model.Kinv, k, kappa + model.a)
        X = np.append(model.X, x.reshape(1, -1), axis=0)
        A, B = KernelConformalRidgeRegressor.compute_A_and_B(X, K, Kinv, model.y)

        cat_mask = np.array([c == cat for c in self.categories_] + [True])
        n_cat = int(cat_mask.sum())

        eps_check = max(epsilon) if hasattr(epsilon, "__iter__") else epsilon
        if bounds == "both" and not (eps_check >= 2 / n_cat):
            return self._inf_interval(epsilon)
        if bounds != "both" and not (eps_check >= 1 / n_cat):
            return self._inf_interval(epsilon)

        A_cat = A[cat_mask]
        B_cat = B[cat_mask]
        l_dic, u_dic = ConformalRegressor._vectorised_l_and_u(A_cat, B_cat)
        return self._build_interval(l_dic, u_dic, epsilon, n_cat, bounds)

    def _predict_lasso(self, x, epsilon, cat):
        model = self.base_model
        x = np.atleast_1d(x).ravel()
        if model.X is None or model.X.shape[0] < 2:
            return self._inf_interval(epsilon)
        if hasattr(epsilon, "__iter__"):
            predictions = {}
            for eps in epsilon:
                predictions[eps] = self._predict_lasso_single(x, eps, cat)
            return MultiLevelPredictionInterval(predictions)
        return self._predict_lasso_single(x, epsilon, cat)

    def _predict_lasso_single(self, x, epsilon, cat):
        model = self.base_model
        cat_mask_train = np.array([c == cat for c in self.categories_])
        n_cat = int(cat_mask_train.sum()) + 1  # +1 for test point
        if n_cat < 2:
            return ConformalPredictionInterval(-np.inf, np.inf, epsilon)

        threshold = int(np.ceil(n_cat * (1 - epsilon)))
        y0 = x @ model.beta
        y_range = model.y.max() - model.y.min()
        if y_range == 0:
            y_range = 1.0
        t_max = (model.y.max() - y0) + model.search_range_factor * y_range
        t_min = (model.y.min() - y0) - model.search_range_factor * y_range

        intervals_pos = self._run_homotopy_mondrian(x, +1, t_max, cat_mask_train, threshold)
        intervals_neg = self._run_homotopy_mondrian(x, -1, t_min, cat_mask_train, threshold)

        all_intervals = []
        for a, b in intervals_neg:
            all_intervals.append((y0 + a, y0 + b))
        if self._t_in_set(x, 0.0, cat_mask_train, threshold):
            all_intervals.append((y0, y0))
        for a, b in intervals_pos:
            all_intervals.append((y0 + a, y0 + b))

        merged = ConformalLassoRegressor._merge_intervals(all_intervals)
        if not merged:
            return ConformalPredictionInterval(np.nan, np.nan, epsilon)
        elif len(merged) == 1:
            return ConformalPredictionInterval(merged[0][0], merged[0][1], epsilon)
        else:
            return ConformalPredictionInterval(merged[0][0], merged[-1][1], epsilon)

    def _t_in_set(self, x_new, t, cat_mask_train, threshold):
        model = self.base_model
        X_aug = np.vstack([model.X, x_new.reshape(1, -1)])
        y_aug = np.append(model.y, x_new @ model.beta + t)
        beta_t = _solve_lasso(X_aug, y_aug, model.lam, rho=model.rho, warm_start=model.beta)
        abs_res = np.abs(y_aug - X_aug @ beta_t)
        rank = int(np.sum(abs_res[:-1][cat_mask_train] <= abs_res[-1])) + 1
        return rank <= threshold

    def _run_homotopy_mondrian(self, x_new, direction, t_bound, cat_mask_train, threshold):
        from online_cp.regressors import _compute_crossings

        model = self.base_model
        sign = 1 if direction > 0 else -1
        t_bound_abs = abs(t_bound)
        n = model.X.shape[0]
        p = model.X.shape[1]
        lam = model.lam
        beta_k = model.beta.copy()
        J_k = np.where(np.abs(beta_k) > 1e-12)[0]
        residuals = model.y - model.X @ beta_k
        v_full = model.X.T @ residuals
        J_c_k = np.setdiff1d(np.arange(p), J_k)
        v_inactive = v_full[J_c_k]
        r_train = residuals.copy()
        r_test = 0.0
        XtX_aug = model.X.T @ model.X + np.outer(x_new, x_new)
        t_accumulated = 0.0
        intervals_in_set = []

        for _ in range(model.max_homotopy_steps):
            if t_accumulated >= t_bound_abs:
                break
            if len(J_k) == 0:
                if len(J_c_k) == 0:
                    break
                gamma_k = sign * x_new[J_c_k]
                dt_dual = np.full(len(J_c_k), np.inf)
                for idx in range(len(J_c_k)):
                    g = gamma_k[idx]
                    if g > 1e-15:
                        dt_dual[idx] = (lam - sign * v_inactive[idx]) / g
                    elif g < -1e-15:
                        dt_dual[idx] = (-lam - sign * v_inactive[idx]) / g
                dt_dual = np.where(dt_dual > 1e-12, dt_dual, np.inf)
                dt_k = min(float(np.min(dt_dual)), t_bound_abs - t_accumulated)
                sub = self._find_intervals_mondrian(
                    r_train, r_test, np.zeros(n), sign * 1.0,
                    dt_k, t_accumulated, sign, threshold, cat_mask_train,
                )
                intervals_in_set.extend(sub)
                t_accumulated += dt_k
                r_test += sign * 1.0 * dt_k
                v_inactive += gamma_k * dt_k
                if dt_k < t_bound_abs - t_accumulated + 1e-12:
                    entering = J_c_k[np.argmin(dt_dual)]
                    J_k = np.append(J_k, entering)
                    J_c_k = np.setdiff1d(np.arange(p), J_k)
                    v_inactive = v_full[J_c_k]
                continue

            Sigma_J = XtX_aug[np.ix_(J_k, J_k)] + model.rho * np.eye(len(J_k))
            x_J = x_new[J_k]
            try:
                Sigma_J_inv = np.linalg.inv(Sigma_J)
            except np.linalg.LinAlgError:
                break
            Sigma_J_inv_x = Sigma_J_inv @ x_J
            eta_k = sign * Sigma_J_inv_x
            if len(J_c_k) > 0:
                gamma_k = sign * (x_new[J_c_k] - XtX_aug[np.ix_(J_c_k, J_k)] @ Sigma_J_inv_x)
            else:
                gamma_k = np.array([])
            slopes_train = -(model.X[:, J_k] @ eta_k)
            slope_test = sign * (1.0 - x_J @ Sigma_J_inv_x)
            beta_J = beta_k[J_k]
            dt_primal = np.full(len(J_k), np.inf)
            for idx in range(len(J_k)):
                if abs(eta_k[idx]) > 1e-15:
                    dt = -beta_J[idx] / eta_k[idx]
                    if dt > 1e-12:
                        dt_primal[idx] = dt
            dt_dual = np.full(len(J_c_k), np.inf)
            for idx in range(len(J_c_k)):
                g = gamma_k[idx]
                v_j = v_inactive[idx]
                if abs(g) > 1e-15:
                    dt1 = (lam - v_j) / g
                    dt2 = (-lam - v_j) / g
                    cands = []
                    if dt1 > 1e-12:
                        cands.append(dt1)
                    if dt2 > 1e-12:
                        cands.append(dt2)
                    if cands:
                        dt_dual[idx] = min(cands)
            dt_k = min(
                float(np.min(dt_primal)) if len(dt_primal) > 0 else np.inf,
                float(np.min(dt_dual)) if len(dt_dual) > 0 else np.inf,
            )
            dt_k = min(dt_k, t_bound_abs - t_accumulated)
            if dt_k <= 0 or not np.isfinite(dt_k):
                break
            sub = self._find_intervals_mondrian(
                r_train, r_test, slopes_train, slope_test,
                dt_k, t_accumulated, sign, threshold, cat_mask_train,
            )
            intervals_in_set.extend(sub)
            beta_k[J_k] += eta_k * dt_k
            r_train += slopes_train * dt_k
            r_test += slope_test * dt_k
            if len(J_c_k) > 0:
                v_inactive += gamma_k * dt_k
            t_accumulated += dt_k
            min_primal = float(np.min(dt_primal)) if len(dt_primal) > 0 else np.inf
            min_dual = float(np.min(dt_dual)) if len(dt_dual) > 0 else np.inf
            if dt_k >= t_bound_abs - (t_accumulated - dt_k):
                break
            if min_primal <= min_dual:
                leaving_idx = np.argmin(dt_primal)
                beta_k[J_k[leaving_idx]] = 0.0
                J_k = np.delete(J_k, leaving_idx)
            else:
                entering_var = J_c_k[np.argmin(dt_dual)]
                J_k = np.append(J_k, entering_var)
            J_c_k = np.setdiff1d(np.arange(p), J_k)
            v_full = model.X.T @ r_train + x_new * r_test
            v_inactive = v_full[J_c_k]

        return intervals_in_set

    @staticmethod
    def _find_intervals_mondrian(
        r_train, r_test, slopes_train, slope_test,
        dt_k, t_accumulated, sign, threshold, cat_mask_train,
    ):
        from online_cp.regressors import _compute_crossings

        n = len(r_train)
        raw_crossings = _compute_crossings(r_train, slopes_train, r_test, slope_test, n, dt_k)
        crossings = sorted(set([0.0] + list(raw_crossings) + [dt_k]))
        result_intervals = []
        for idx in range(len(crossings) - 1):
            d_start = crossings[idx]
            d_end = crossings[idx + 1]
            if d_end - d_start < 1e-14:
                continue
            d_mid = (d_start + d_end) / 2
            r_i_mid = r_train + slopes_train * d_mid
            r_test_mid = r_test + slope_test * d_mid
            # Only count same-category residuals for rank
            abs_r_cat = np.abs(r_i_mid[cat_mask_train])
            abs_r_test = np.abs(r_test_mid)
            rank = int(np.sum(abs_r_cat <= abs_r_test)) + 1
            if rank <= threshold:
                t_start = sign * (t_accumulated + d_start)
                t_end = sign * (t_accumulated + d_end)
                if t_start > t_end:
                    t_start, t_end = t_end, t_start
                result_intervals.append((t_start, t_end))
        return result_intervals

    def compute_p_value(self, x, y, **kwargs):
        """Compute the Mondrian conformal p-value for (x, y)."""
        cat = self.category_fn(x)
        model = self.base_model
        if isinstance(model, ConformalRidgeRegressor):
            return self._p_value_ridge(x, y, cat, **kwargs)
        elif isinstance(model, KernelConformalRidgeRegressor):
            return self._p_value_kernel_ridge(x, y, cat, **kwargs)
        else:
            return self._p_value_lasso(x, y, cat, **kwargs)

    def _p_value_ridge(self, x, y, cat, bounds="both", smoothed=True, tau=None):
        model = self.base_model
        if tau is None and smoothed:
            tau = model.rnd_gen.uniform(0, 1)
        if model.XTXinv is None:
            return tau if smoothed else 1
        X = np.append(model.X, x.reshape(1, -1), axis=0)
        XTXinv = model.XTXinv - (
            model.XTXinv @ np.outer(x, x) @ model.XTXinv
        ) / (1 + x.T @ model.XTXinv @ x)
        A, B = model.compute_A_and_B(X, XTXinv, model.y)
        cat_mask = np.array([c == cat for c in self.categories_] + [True])
        A_cat, B_cat = A[cat_mask], B[cat_mask]
        if bounds == "both":
            E = A_cat + y * B_cat
            Alpha = np.array([min((E >= e).sum(), (E <= e).sum()) for e in E])
            c_type = "conformity"
        elif bounds == "lower":
            Alpha = -(A_cat + y * B_cat)
            c_type = "nonconformity"
        else:
            Alpha = A_cat + y * B_cat
            c_type = "nonconformity"
        return ConformalRegressor._compute_p_value(Alpha, tau if smoothed else None, c_type=c_type)

    def _p_value_kernel_ridge(self, x, y, cat, bounds="both", smoothed=True, tau=None):
        model = self.base_model
        if tau is None and smoothed:
            tau = model.rnd_gen.uniform(0, 1)
        if model.Kinv is None:
            return tau if smoothed else 1
        k = model.kernel(model.X, x).reshape(-1, 1)
        kappa = model.kernel(x, x)
        K = KernelConformalRidgeRegressor._update_K(model.K, k, kappa)
        Kinv = KernelConformalRidgeRegressor._update_Kinv(model.Kinv, k, kappa + model.a)
        X = np.append(model.X, x.reshape(1, -1), axis=0)
        A, B = KernelConformalRidgeRegressor.compute_A_and_B(X, K, Kinv, model.y)
        cat_mask = np.array([c == cat for c in self.categories_] + [True])
        A_cat, B_cat = A[cat_mask], B[cat_mask]
        if bounds == "both":
            E = A_cat + y * B_cat
            Alpha = np.array([min((E >= e).sum(), (E <= e).sum()) for e in E])
            c_type = "conformity"
        elif bounds == "lower":
            Alpha = -(A_cat + y * B_cat)
            c_type = "nonconformity"
        else:
            Alpha = A_cat + y * B_cat
            c_type = "nonconformity"
        return ConformalRegressor._compute_p_value(Alpha, tau if smoothed else None, c_type=c_type)

    def _p_value_lasso(self, x, y, cat, smoothed=True, tau=None):
        model = self.base_model
        x = np.atleast_1d(x).ravel()
        if tau is None and smoothed:
            tau = model.rnd_gen.uniform()
        if model.X is None:
            return tau if smoothed else 1
        X_aug = np.vstack([model.X, x.reshape(1, -1)])
        y_aug = np.append(model.y, y)
        beta_aug = _solve_lasso(X_aug, y_aug, model.lam, rho=model.rho, warm_start=model.beta)
        residuals = np.abs(y_aug - X_aug @ beta_aug)
        cat_mask_train = np.array([c == cat for c in self.categories_])
        r_cat = residuals[:-1][cat_mask_train]
        r_test = residuals[-1]
        if smoothed and tau is not None:
            gt = np.sum(r_cat > r_test)
            eq = np.sum(r_cat == r_test) + 1
            p = (gt + tau * eq) / (len(r_cat) + 1)
        else:
            geq = np.sum(r_cat >= r_test) + 1
            p = geq / (len(r_cat) + 1)
        return float(p)

    @staticmethod
    def _build_interval(l_dic, u_dic, epsilon, n, bounds):
        if hasattr(epsilon, "__iter__"):
            predictions = {}
            for eps in epsilon:
                if bounds == "both":
                    lo = ConformalRegressor._get_lower(l_dic=l_dic, epsilon=eps / 2, n=n)
                    up = ConformalRegressor._get_upper(u_dic=u_dic, epsilon=eps / 2, n=n)
                elif bounds == "lower":
                    lo = ConformalRegressor._get_lower(l_dic=l_dic, epsilon=eps, n=n)
                    up = np.inf
                else:
                    lo = -np.inf
                    up = ConformalRegressor._get_upper(u_dic=u_dic, epsilon=eps, n=n)
                predictions[eps] = ConformalPredictionInterval(lo, up, eps)
            return MultiLevelPredictionInterval(predictions)
        if bounds == "both":
            lo = ConformalRegressor._get_lower(l_dic=l_dic, epsilon=epsilon / 2, n=n)
            up = ConformalRegressor._get_upper(u_dic=u_dic, epsilon=epsilon / 2, n=n)
        elif bounds == "lower":
            lo = ConformalRegressor._get_lower(l_dic=l_dic, epsilon=epsilon, n=n)
            up = np.inf
        else:
            lo = -np.inf
            up = ConformalRegressor._get_upper(u_dic=u_dic, epsilon=epsilon, n=n)
        return ConformalPredictionInterval(lo, up, epsilon)

    @staticmethod
    def _inf_interval(epsilon):
        if hasattr(epsilon, "__iter__"):
            return MultiLevelPredictionInterval(
                {eps: ConformalPredictionInterval(-np.inf, np.inf, eps) for eps in epsilon}
            )
        return ConformalPredictionInterval(-np.inf, np.inf, epsilon)

    @property
    def categories(self):
        """Return the set of discovered categories."""
        return set(self.categories_)

    def __repr__(self):
        return (
            f"MondrianConformalRegressor(n_categories={len(self.categories)}, "
            f"n_total={len(self.categories_)}, "
            f"base={self.base_model.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# Mondrian Conformal Classifier
# ---------------------------------------------------------------------------


class MondrianConformalClassifier:
    """Mondrian conformal classifier: group-conditional coverage.

    Wraps a conformal classifier (KNN or SVM). A single pooled model is trained
    on ALL data. At prediction time, nonconformity scores are computed using the
    pooled model, but the p-value computation only compares against scores from
    the SAME category.

    This yields valid group-conditional coverage:
        P(y in Gamma(x) | category(x) = k) >= 1 - epsilon  for all k.

    The most common special case is **label-conditional** conformal prediction
    (ALRW2 §4.6.7), where the category of an example is its label:
    κ(n, (x, y)) = y. Use ``category_fn="label"`` for this.

    Parameters
    ----------
    base_model : ConformalNearestNeighboursClassifier or ConformalSupportVectorMachine
        The underlying conformal classifier. Trained on all data (pooled).
    category_fn : str or callable
        Determines the Mondrian taxonomy:

        - ``"label"`` — label-conditional: category = label (most common).
        - A callable ``(x, y) -> hashable`` — general Mondrian taxonomy
          depending on both features and label.
        - A callable ``(x) -> hashable`` — object-conditional taxonomy
          depending only on features (legacy interface).
    """

    def __init__(self, base_model: Any, category_fn: str | Callable[..., Hashable]) -> None:
        self.base_model = base_model

        if category_fn == "label":
            self._category_fn = lambda x, y: y
            self._label_aware = True
        elif callable(category_fn):
            import inspect
            sig = inspect.signature(category_fn)
            n_params = sum(
                1 for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            )
            if n_params >= 2:
                self._category_fn = category_fn
                self._label_aware = True
            else:
                self._category_fn = category_fn
                self._label_aware = False
        else:
            raise TypeError(
                f"category_fn must be 'label' or a callable, got {type(category_fn).__name__}"
            )

        self.category_fn = category_fn
        self.categories_ = []

    def learn_initial_training_set(self, X: NDArray[np.floating[Any]], y: NDArray[Any]) -> None:
        """Train the pooled model on all data and record categories."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.base_model.learn_initial_training_set(X, y)
        if self._label_aware:
            self.categories_ = [self._category_fn(x_i, y_i) for x_i, y_i in zip(X, y)]
        else:
            self.categories_ = [self._category_fn(x_i) for x_i in X]

    def learn_one(self, x: NDArray[np.floating[Any]], y: Any, **kwargs: Any) -> None:
        """Learn a single example in the pooled model and record its category."""
        self.base_model.learn_one(x, y, **kwargs)
        if self._label_aware:
            self.categories_.append(self._category_fn(x, y))
        else:
            self.categories_.append(self._category_fn(x))

    def predict(self, x: NDArray[np.floating[Any]], epsilon: float | NDArray[np.floating[Any]] | None = None, return_p_values: bool = False) -> Any:
        """Compute the Mondrian conformal prediction set."""
        from online_cp.classifiers import (
            ConformalNearestNeighboursClassifier,
            ConformalSupportVectorMachine,
        )

        if epsilon is None:
            epsilon = self.base_model.epsilon
        model = self.base_model
        tau = model.rnd_gen.uniform(0, 1)

        if self._label_aware:
            # Label-aware: per-hypothesis category mask
            cat_masks = {}
            for label in model.label_space:
                cat = self._category_fn(x, label)
                cat_masks[label] = np.array([c == cat for c in self.categories_])
        else:
            # Object-conditional: one shared mask for all hypotheses
            cat = self._category_fn(x)
            shared_mask = np.array([c == cat for c in self.categories_])
            cat_masks = {label: shared_mask for label in model.label_space}

        if isinstance(model, ConformalNearestNeighboursClassifier):
            p_values = self._predict_knn(x, cat_masks, tau)
        elif isinstance(model, ConformalSupportVectorMachine):
            p_values = self._predict_svm(x, cat_masks, tau)
        else:
            raise TypeError(f"Unsupported classifier: {type(model).__name__}")

        Gamma = model._compute_Gamma(p_values, epsilon)
        if return_p_values:
            return Gamma, p_values
        return Gamma

    def _predict_knn(self, x, cat_masks, tau):
        model = self.base_model
        p_values = {}
        if model.y.shape[0] < 1:
            for label in model.label_space:
                p_values[label] = 1.0
            return p_values

        d = model.distance_func(model.X, x)
        D = model.update_distance_matrix(model.D, d)
        for label in model.label_space:
            y = np.append(model.y, label)
            same_d, diff_d = model._find_nearest_distances(D, y)
            Alpha_full = np.nan_to_num(same_d / diff_d, nan=np.inf)
            # Filter to same-category + test point (last element always included)
            cat_mask_aug = np.append(cat_masks[label], True)
            Alpha_cat = Alpha_full[cat_mask_aug]
            p_values[label] = self._compute_mondrian_p(Alpha_cat, tau)
        return p_values

    def _predict_svm(self, x, cat_masks, tau):
        from online_cp.classifiers import _smo_solve

        model = self.base_model
        p_values = {}
        x = np.atleast_1d(x).ravel()
        if model.X is None or model.y.shape[0] == 0:
            for label in model.label_space:
                p_values[label] = 1.0
            return p_values

        k_row = model._compute_kernel_row(model.X, x)
        kappa = model._kernel(x.reshape(1, -1))
        if np.ndim(kappa) > 0:
            kappa = kappa.item()
        n = model.K.shape[0]
        K_aug = np.empty((n + 1, n + 1))
        K_aug[:n, :n] = model.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        for label in model.label_space:
            cat_mask = cat_masks[label]
            y_aug = np.append(model.y, float(label))
            y_binary = np.where(y_aug == label, 1.0, -1.0)
            alpha, _ = _smo_solve(
                K_aug, y_binary, model.C, tol=model.smo_tol, max_iter=model.smo_max_iter
            )
            if len(model.label_space) > 2:
                # Multiclass: filter to same-class first, then apply cat_mask
                same_class_mask = y_binary == 1.0
                alpha_class = alpha[same_class_mask]
                cat_indices = np.where(same_class_mask)[0]
                cat_mask_subset = np.array(
                    [(cat_mask[i] if i < n else True) for i in cat_indices]
                )
                Alpha_cat = alpha_class[cat_mask_subset]
            else:
                Alpha_cat = alpha[np.append(cat_mask, True)]
            p_values[label] = self._compute_mondrian_p(Alpha_cat, tau)
        return p_values

    @staticmethod
    def _compute_mondrian_p(Alpha, tau):
        """Smoothed p-value from category-filtered scores (last = test)."""
        alpha_n = Alpha[-1]
        gt = np.sum(Alpha > alpha_n)
        eq = np.sum(Alpha == alpha_n)
        return float((gt + tau * eq) / Alpha.shape[0])

    @property
    def categories(self):
        """Return the set of discovered categories."""
        return set(self.categories_)

    def __repr__(self):
        return (
            f"MondrianConformalClassifier(n_categories={len(self.categories)}, "
            f"n_total={len(self.categories_)}, "
            f"base={self.base_model.__class__.__name__})"
        )
