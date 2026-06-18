"""Decision-making utilities for conformal and Venn predictive systems.

Implements the three-layer architecture:

1. **Utility function** — ``UtilityFunction(fn, decisions)`` bundles a utility
   callable ``U(x, y, d) → float`` with its decision space.
2. **Expected utility computation** — engines that compute expected utility
   tables from a predictive object (CPD or Venn multiprobability).
3. **Decision criteria** — generic selectors (maximize, maximin, Hurwicz,
   minimax regret) that operate on expectation tables.

References
----------
- Vovk, V. & Bendtsen, C. (2018). *Conformal predictive decision making*.
  Proceedings of Machine Learning Research, 91, 52–62.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Callable

try:
    from ._serialization import SerializableMixin, SerializationError, from_token, to_token
except ImportError:
    from _serialization import SerializableMixin, SerializationError, from_token, to_token

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "UtilityFunction",
    "cps_expected_utilities",
    "venn_expected_utilities",
    "alpha_utility",
    "alpha_regret",
    "cps_decision",
    "venn_decision",
    "ConformalPredictiveDecisionMaker",
]


# ---------------------------------------------------------------------------
# Layer 1: Utility Function
# ---------------------------------------------------------------------------


class UtilityFunction:
    """Utility function bundled with its decision space.

    Parameters
    ----------
    fn : callable
        A function ``(x, y, d) -> float`` that returns the utility of
        taking decision *d* when features are *x* and outcome is *y*.
    decisions : sequence
        The set of available decisions (the decision space D).

    Examples
    --------
    >>> utility = UtilityFunction(lambda x, y, d: -(y - d)**2,
    ...                           decisions=[0.0, 0.5, 1.0, 1.5, 2.0])
    >>> utility(None, 1.0, 0.5)
    -0.25
    """

    def __init__(self, fn: Callable[..., float], decisions: Sequence[Any]) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn
        self.decisions = list(decisions)
        if not self.decisions:
            raise ValueError("decisions must be non-empty")

    def __call__(self, x: Any, y: Any, d: Any) -> float:
        return self.fn(x, y, d)

    def __repr__(self) -> str:
        return f"UtilityFunction(|D|={len(self.decisions)})"


# ---------------------------------------------------------------------------
# Layer 2: Expected Utility Computation
# ---------------------------------------------------------------------------


def cps_expected_utilities(
    cpd,
    utility: UtilityFunction,
    x: Any,
    tau: float = 0.5,
) -> dict[Any, float]:
    r"""Compute expected utility under a CPD for each decision (Vovk & Bendtsen 2018).

    For each decision *d* in ``utility.decisions``, computes:

    $$
    \mathbb{E}_\tau[U(x, \cdot, d)] = \sum_j U(x, C_j, d) \, \Delta Q_\tau(C_j)
    $$

    where $\Delta Q_\tau(C_j)$ is the probability mass at critical point $C_j$
    under randomisation parameter $\tau$.

    Parameters
    ----------
    cpd : ConformalPredictiveDistributionFunction
        A conformal predictive distribution object with attributes ``Y``
        (sorted critical points), ``L``, and ``U`` arrays.
    utility : UtilityFunction
        Utility function with decision space.
    x : any
        Features of the test object (passed to utility).
    tau : float, default 0.5
        Randomisation parameter in [0, 1].

    Returns
    -------
    dict
        Mapping from each decision to its expected utility (float).
    """
    if not (0 <= tau <= 1):
        raise ValueError(f"tau must be in [0, 1], got {tau}")

    delta_Q = _cpd_masses(cpd, tau)
    # Pre-compute Y values where mass is non-zero AND finite
    finite_mask = np.isfinite(cpd.Y)
    mask = (delta_Q != 0) & finite_mask
    Y_nz = cpd.Y[mask]
    dQ_nz = delta_Q[mask]

    result = {}
    for d in utility.decisions:
        utilities = np.array([utility(x, y, d) for y in Y_nz])
        result[d] = float(utilities @ dQ_nz)
    return result


def venn_expected_utilities(
    venn_pred,
    utility: UtilityFunction,
    x: Any,
) -> dict[Any, NDArray[np.floating]]:
    r"""Compute expected utility under each Venn hypothesis for each decision.

    For each decision *d* and hypothesis *v*:

    $$
    \mathbb{E}_{P^v}[U(x, \cdot, d)] = \sum_j P^v(\text{label}_j) \, U(x, \text{label}_j, d)
    $$

    Parameters
    ----------
    venn_pred : VennPrediction
        Multiprobability prediction with ``probs`` matrix (|Y| × |Y|)
        and ``label_space`` array.
    utility : UtilityFunction
        Utility function with decision space.
    x : any
        Features of the test object (passed to utility).

    Returns
    -------
    dict
        Mapping from each decision to an ndarray of shape (|Y|,) giving
        the expected utility under each hypothesis.
    """
    labels = venn_pred.label_space
    probs = venn_pred.probs  # shape (|Y|, |Y|)

    result = {}
    for d in utility.decisions:
        # Utility vector: U(x, label_j, d) for each label j
        u_vec = np.array([utility(x, y, d) for y in labels])
        # Expected utility under each hypothesis v: probs[v, :] @ u_vec
        result[d] = probs @ u_vec
    return result


# ---------------------------------------------------------------------------
# Layer 3: Decision Criteria (α-utility and α-regret families)
# ---------------------------------------------------------------------------


def alpha_utility(expectations: dict[Any, float | NDArray | tuple], alpha: float = 0.0) -> Any:
    """α-utility criterion: Hurwicz-weighted expected utility.

    For each decision *d*, the score is:

    ``score(d) = α · upper(E[U|d]) + (1 − α) · lower(E[U|d])``

    Special cases:

    - α = 0: maximin (pessimistic — maximise worst-case utility)
    - α = 1: maximax (optimistic — maximise best-case utility)
    - α = 0.5: midpoint (average of best/worst case)

    When expectations are scalars (point case), all α values give the
    same result (maximize).

    Parameters
    ----------
    expectations : dict
        Mapping from decision to expected utility. Values can be:
        scalars (point), ndarrays (one per hypothesis), or tuples (lo, hi).
    alpha : float, default 0.0
        Optimism index in [0, 1] (0 = pessimistic, 1 = optimistic).

    Returns
    -------
    any
        The optimal decision.
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    def score(d):
        lo = _lower(expectations[d])
        hi = _upper(expectations[d])
        return alpha * hi + (1 - alpha) * lo

    return max(expectations, key=score)


def alpha_regret(expectations: dict[Any, NDArray | tuple], alpha: float = 0.0) -> Any:
    """α-regret criterion: Hurwicz-weighted regret minimisation.

    For each scenario/hypothesis *v*, the regret of decision *d* is
    ``max_{d'} E_v[U(d')] - E_v[U(d)]``. The score for each decision is:

    ``score(d) = α · min_v R_v(d) + (1 − α) · max_v R_v(d)``

    We select the decision minimising this score.

    Special cases:

    - α = 0: minimax regret (pessimistic — minimise worst-case regret)
    - α = 1: minimin regret (optimistic — minimise best-case regret)

    Parameters
    ----------
    expectations : dict
        Mapping from decision to imprecise expectations (ndarray or tuple).
    alpha : float, default 0.0
        Optimism index in [0, 1] (0 = pessimistic, 1 = optimistic).

    Returns
    -------
    any
        The decision minimising the α-regret score.
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    decisions = list(expectations.keys())
    cols = [_as_array(expectations[d]) for d in decisions]
    matrix = np.column_stack(cols)  # shape (n_scenarios, n_decisions)

    best_per_scenario = matrix.max(axis=1)
    regret = best_per_scenario[:, None] - matrix

    score = alpha * regret.min(axis=0) + (1 - alpha) * regret.max(axis=0)

    best_idx = int(np.argmin(score))
    return decisions[best_idx]


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def cps_decision(
    cpd,
    utility: UtilityFunction,
    x: Any,
    tau: float = 0.5,
) -> Any:
    """Select optimal decision by maximising expected utility under a label-CPD.

    This is the *naive* approach: a single CPD is trained on raw labels,
    and expected utility is computed by weighting U(x, C_j, d) by the CPD
    masses. Practical for large or continuous decision spaces.

    For the exact Vovk & Bendtsen (2018) Algorithm 1 (one CPD per decision,
    trained on utility-transformed labels), use
    :class:`ConformalPredictiveDecisionMaker`.

    Parameters
    ----------
    cpd : ConformalPredictiveDistributionFunction
        Conformal predictive distribution.
    utility : UtilityFunction
        Utility function with decision space.
    x : any
        Features of the test object.
    tau : float, default 0.5
        Randomisation parameter in [0, 1].

    Returns
    -------
    any
        The optimal decision.
    """
    exps = cps_expected_utilities(cpd, utility, x, tau)
    return max(exps, key=exps.get)


def venn_decision(
    venn_pred,
    utility: UtilityFunction,
    x: Any,
    criterion: str = "utility",
    alpha: float = 0.0,
) -> Any:
    """Select optimal decision under Venn multiprobability (Venn-PDMS).

    Computes expected utilities under each Venn hypothesis, then applies
    one of two decision criterion families:

    - ``"utility"`` (α-utility): score(d) = α·upper + (1−α)·lower.
      α=0 is maximin, α=1 is maximax, α=0.5 is midpoint.
    - ``"regret"`` (α-regret): minimise the α-weighted regret.
      α=0 is minimax regret, α=1 is minimin regret.

    Parameters
    ----------
    venn_pred : VennPrediction
        Multiprobability prediction.
    utility : UtilityFunction
        Utility function with decision space.
    x : any
        Features of the test object.
    criterion : str, default "utility"
        One of ``"utility"`` or ``"regret"``.
    alpha : float, default 0.0
        Optimism index in [0, 1]. α=0 is pessimistic (default).

    Returns
    -------
    any
        The optimal decision.
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    exps = venn_expected_utilities(venn_pred, utility, x)
    return _apply_criterion(exps, criterion, alpha)


# ---------------------------------------------------------------------------
# V&B 2018 Algorithm 1: Conformal Predictive Decision Making
# ---------------------------------------------------------------------------


class ConformalPredictiveDecisionMaker(SerializableMixin):
    """Conformal Predictive Decision Making (Vovk & Bendtsen 2018, Algorithm 1).

    Maintains one conformal predictive system per decision.
    Each model is trained on utility-transformed labels:
    ``(x_i, U(x_i, y_i, d))`` for the corresponding decision *d*.

    At prediction time, the conformal mean of each utility-CPD gives the
    expected utility of that decision; the decision with highest expected
    utility is returned.

    Parameters
    ----------
    utility : UtilityFunction
        Utility function bundled with its finite decision space.
    cps_class : type, optional
        The CPS class to use for each decision. Must implement
        ``learn_initial_training_set(X, y)``, ``predict_cpd(x)``, and
        ``learn_one(x, y)``. Defaults to
        :class:`~online_cp.CPS.RidgePredictionMachine`.
    **cps_kwargs
        Keyword arguments passed to each CPS instance (e.g., ``a=1.0``
        for ridge regularisation).

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp import UtilityFunction, ConformalPredictiveDecisionMaker
    >>> U = UtilityFunction(lambda x, y, d: [[0, -10, 3], [-3, 5, 1]][int(y)][int(d)],
    ...                     decisions=[0, 1, 2])
    >>> cdm = ConformalPredictiveDecisionMaker(U, a=1.0)
    >>> X = np.random.default_rng(0).normal(size=(40, 5))
    >>> y = (X[:, 0] > 0).astype(float)
    >>> cdm.learn_initial_training_set(X, y)
    >>> decision = cdm.predict(X[0])
    >>> decision in [0, 1, 2]
    True
    """

    def __init__(self, utility: UtilityFunction, cps_class=None, **cps_kwargs) -> None:
        if cps_class is None:
            from online_cp.CPS import RidgePredictionMachine
            cps_class = RidgePredictionMachine

        self.utility = utility
        self._cps_class = cps_class
        self._models: dict[Any, Any] = {}
        # Default to warnings=False if not explicitly set
        cps_kwargs.setdefault("warnings", False)
        for d in utility.decisions:
            self._models[d] = cps_class(**cps_kwargs)

    def learn_initial_training_set(
        self, X: NDArray, y: NDArray
    ) -> None:
        """Train all |D| internal CPS models on utility-transformed labels.

        For each decision *d*, the training labels are
        ``u_i(d) = U(x_i, y_i, d)`` for ``i = 1, ..., n``.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Training features.
        y : ndarray of shape (n,)
            Training labels (outcomes).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        for d, model in self._models.items():
            u = np.array([self.utility(X[i], y[i], d) for i in range(len(y))])
            model.learn_initial_training_set(X, u)

    def predict(self, x: Any, tau: float = 0.5) -> Any:
        """Return the decision with highest expected utility.

        For each decision *d*, computes the conformal mean of the
        utility-CPD $Q^*_d$ and returns the decision maximising it.

        Parameters
        ----------
        x : array-like of shape (p,)
            Test object features.
        tau : float, default 0.5
            Randomisation parameter in [0, 1].

        Returns
        -------
        any
            The optimal decision.
        """
        eus = self.predict_expected_utilities(x, tau)
        return max(eus, key=eus.get)

    def predict_expected_utilities(self, x: Any, tau: float = 0.5) -> dict[Any, float]:
        """Return expected utilities for all decisions (without selecting one).

        Parameters
        ----------
        x : array-like of shape (p,)
            Test object features.
        tau : float, default 0.5
            Randomisation parameter in [0, 1].

        Returns
        -------
        dict
            Mapping from each decision to its expected utility.
        """
        if not (0 <= tau <= 1):
            raise ValueError(f"tau must be in [0, 1], got {tau}")

        result = {}
        for d, model in self._models.items():
            cpd = model.predict_cpd(x)
            masses = _cpd_masses(cpd, tau)
            finite_mask = np.isfinite(cpd.Y)
            result[d] = float((cpd.Y[finite_mask] * masses[finite_mask]).sum())
        return result

    def learn_one(self, x: Any, y: Any) -> None:
        """Update all |D| internal models with a new observation.

        For each decision *d*, the model is updated with the pair
        ``(x, U(x, y, d))``.

        Parameters
        ----------
        x : array-like of shape (p,)
            Features of the new observation.
        y : any
            True outcome (label).
        """
        for d, model in self._models.items():
            u = self.utility(x, y, d)
            model.learn_one(x, u)

    def __repr__(self) -> str:
        cps_name = self._cps_class.__name__
        n_decisions = len(self.utility.decisions)
        return f"ConformalPredictiveDecisionMaker(|D|={n_decisions}, cps={cps_name})"

    def save(self, filepath: str | os.PathLike, *, compress: int = 3) -> None:
        """Save this decision maker to *filepath*.

        The utility function callable (``utility.fn``) must be registered via
        :func:`~online_cp.register_callable` or be a module-level named
        function; lambdas will raise :class:`~online_cp.SerializationError`.

        !!! warning
            Only load files from **trusted sources**. Deserialising a file from
            an untrusted source can execute arbitrary code.
        """
        import joblib

        try:
            from online_cp import __version__ as _lib_version
        except Exception:
            _lib_version = "unknown"

        try:
            envelope = {
                "format_version": 1,
                "library_version": _lib_version,
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "utility_fn_token": to_token(self.utility.fn),
                "utility_decisions": self.utility.decisions,
                "cps_class": self._cps_class,
                "models": self._models,
            }
            joblib.dump(envelope, filepath, compress=compress)
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Failed to save model to {filepath!r}: {exc}"
            ) from exc

    @classmethod
    def load(cls, filepath: str | os.PathLike) -> ConformalPredictiveDecisionMaker:
        """Load a decision maker from *filepath*.

        !!! warning
            Only load files from **trusted sources**. Deserialising a file from
            an untrusted source can execute arbitrary code.
        """
        import warnings as _warnings

        import joblib

        try:
            from online_cp import __version__ as _lib_version
        except Exception:
            _lib_version = "unknown"

        try:
            envelope = joblib.load(filepath)
        except Exception as exc:
            raise SerializationError(
                f"Failed to read model file {filepath!r}: {exc}"
            ) from exc

        fmt_ver = envelope.get("format_version", 0)
        if fmt_ver > 1:
            raise SerializationError(
                f"Unsupported format_version {fmt_ver}. Update online-cp to load this file."
            )
        lib_ver = envelope.get("library_version", "unknown")
        if lib_ver != _lib_version:
            _warnings.warn(
                f"Model was saved with online-cp {lib_ver!r}, "
                f"but you are using {_lib_version!r}. Predictions may differ.",
                UserWarning,
                stacklevel=2,
            )
        expected = f"{cls.__module__}.{cls.__qualname__}"
        if envelope.get("class", "") != expected:
            raise SerializationError(
                f"Class mismatch: file contains '{envelope.get('class')}', "
                f"expected '{expected}'."
            )

        utility_fn = from_token(envelope["utility_fn_token"])
        utility = UtilityFunction(fn=utility_fn, decisions=envelope["utility_decisions"])
        obj = cls.__new__(cls)
        obj.utility = utility
        obj._cps_class = envelope["cps_class"]
        obj._models = envelope["models"]
        return obj


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cpd_masses(cpd, tau: float) -> NDArray[np.floating]:
    """Compute probability masses ΔQ at each critical point of a CPD.

    For a CPD with sorted critical points Y and lower/upper CDFs L, U:
    ΔQ_τ(C_j) = Q_τ(C_j) - lim_{y↑C_j} Q_τ(y)

    For piecewise-constant CPDs (Ridge, NN, etc.), the mass at C_j is
    the jump at that point: (1-τ)*L[j] + τ*U[j] - (1-τ)*L[j-1] - τ*U[j-1]
    evaluated with appropriate boundary logic.
    """
    L = cpd.L
    U = cpd.U
    # CDF value at each critical point
    Q = (1 - tau) * L + tau * U
    # Mass = jump at each point
    delta_Q = np.diff(Q, prepend=0.0)
    return delta_Q


def _lower(val) -> float:
    """Extract lower bound from imprecise expectation."""
    if isinstance(val, tuple):
        return float(val[0])
    return float(np.min(val))


def _upper(val) -> float:
    """Extract upper bound from imprecise expectation."""
    if isinstance(val, tuple):
        return float(val[1])
    return float(np.max(val))


def _as_array(val) -> NDArray:
    """Convert imprecise expectation to array of scenarios."""
    if isinstance(val, tuple):
        return np.array([val[0], val[1]])
    return np.asarray(val)


def _apply_criterion(
    expectations: dict, criterion: str, alpha: float = 0.0
) -> Any:
    """Dispatch to the appropriate decision criterion."""
    if criterion == "utility":
        return alpha_utility(expectations, alpha)
    elif criterion == "regret":
        return alpha_regret(expectations, alpha)
    else:
        raise ValueError(
            f"Unknown criterion {criterion!r}. "
            f"Choose from: 'utility', 'regret'."
        )
