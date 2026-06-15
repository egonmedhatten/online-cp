"""Tests for decision-making utilities (decision.py)."""

import numpy as np
import pytest

from online_cp.CPS import RidgePredictionMachine
from online_cp.decision import (
    UtilityFunction,
    alpha_regret,
    alpha_utility,
    cps_decision,
    cps_expected_utilities,
    venn_decision,
    venn_expected_utilities,
)
from online_cp.venn import VennPrediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def squared_error_utility():
    """Squared error utility: U(x, y, d) = -(y - d)^2."""
    decisions = np.linspace(-3, 3, 13).tolist()
    return UtilityFunction(lambda x, y, d: -(y - d) ** 2, decisions)


@pytest.fixture
def zero_one_utility():
    """Zero-one utility for classification: U(x, y, d) = 1{y == d}."""
    decisions = [0, 1, 2]
    return UtilityFunction(lambda x, y, d: float(y == d), decisions)


@pytest.fixture
def ridge_cpd():
    """A Ridge CPD built from synthetic linear data."""
    rng = np.random.default_rng(123)
    N = 50
    X = rng.normal(size=(N, 2))
    beta = np.array([2.0, -1.0])
    Y = X @ beta + rng.normal(scale=0.3, size=N)
    cps = RidgePredictionMachine(a=1.0, warnings=False)
    cps.learn_initial_training_set(X[:40], Y[:40])
    return cps.predict_cpd(X[40])


@pytest.fixture
def binary_venn_pred():
    """A binary VennPrediction with known p0, p1."""
    return VennPrediction.binary(p0=0.3, p1=0.9)


@pytest.fixture
def multiclass_venn_pred():
    """A 3-class VennPrediction."""
    probs = np.array([
        [0.7, 0.2, 0.1],  # hypothesis y=0
        [0.1, 0.8, 0.1],  # hypothesis y=1
        [0.2, 0.1, 0.7],  # hypothesis y=2
    ])
    return VennPrediction(probs, np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# Layer 1: UtilityFunction
# ---------------------------------------------------------------------------


class TestUtilityFunction:
    def test_callable(self):
        u = UtilityFunction(lambda x, y, d: y * d, decisions=[1, 2, 3])
        assert u(None, 2.0, 3.0) == 6.0

    def test_decisions_stored(self):
        u = UtilityFunction(lambda x, y, d: 0, decisions=[1, 2, 3])
        assert u.decisions == [1, 2, 3]


# ---------------------------------------------------------------------------
# Layer 2: CPS expected utilities
# ---------------------------------------------------------------------------


class TestCPSExpectedUtilities:
    def test_returns_dict_of_floats(self, ridge_cpd, squared_error_utility):
        exps = cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=0.5)
        assert isinstance(exps, dict)
        for _d, val in exps.items():
            assert isinstance(val, float)
            assert np.isfinite(val)

    def test_optimal_near_data_mean(self, ridge_cpd, squared_error_utility):
        """For squared error utility, optimal decision ≈ CPD mean."""
        exps = cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=0.5)
        d_star = max(exps, key=lambda d: exps[d])
        # The CPD median should be near the optimal decision
        median = ridge_cpd.median(0.5)
        assert abs(d_star - median) < 1.0  # within 1 unit on coarse grid

    def test_all_utilities_nonpositive_squared_error(self, ridge_cpd, squared_error_utility):
        """Squared error utility is always ≤ 0."""
        exps = cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=0.5)
        for val in exps.values():
            assert val <= 1e-10

    def test_finite_mass_less_than_one(self, ridge_cpd):
        """Mass on finite critical points should be n/(n+1) < 1."""
        from online_cp.decision import _cpd_masses
        masses = _cpd_masses(ridge_cpd, tau=0.5)
        finite_mask = np.isfinite(ridge_cpd.Y)
        finite_total = masses[finite_mask].sum()
        assert 0.9 < finite_total < 1.0  # n/(n+1) for n=40 is 40/41 ≈ 0.976


# ---------------------------------------------------------------------------
# Layer 2: Venn expected utilities
# ---------------------------------------------------------------------------


class TestVennExpectedUtilities:
    def test_binary_returns_dict(self, binary_venn_pred, zero_one_utility):
        # Use only decisions 0 and 1 for binary
        utility = UtilityFunction(lambda x, y, d: float(y == d), decisions=[0, 1])
        exps = venn_expected_utilities(binary_venn_pred, utility, x=None)
        assert set(exps.keys()) == {0, 1}
        for val in exps.values():
            assert val.shape == (2,)  # 2 hypotheses

    def test_multiclass_shape(self, multiclass_venn_pred, zero_one_utility):
        exps = venn_expected_utilities(multiclass_venn_pred, zero_one_utility, x=None)
        assert set(exps.keys()) == {0, 1, 2}
        for val in exps.values():
            assert val.shape == (3,)  # 3 hypotheses

    def test_zero_one_diagonal(self, multiclass_venn_pred):
        """For 0-1 utility and diagonal-dominant probs, each hypothesis
        prefers the matching decision."""
        utility = UtilityFunction(lambda x, y, d: float(y == d), decisions=[0, 1, 2])
        exps = venn_expected_utilities(multiclass_venn_pred, utility, x=None)
        # Under hypothesis v=0, E[U(d=0)] = P^0(0) = 0.7 > P^0(1) = 0.2
        assert exps[0][0] > exps[1][0]
        assert exps[1][1] > exps[0][1]
        assert exps[2][2] > exps[0][2]


# ---------------------------------------------------------------------------
# Layer 3: Decision Criteria
# ---------------------------------------------------------------------------


class TestDecisionCriteria:
    def test_alpha_utility_point_case(self):
        """For scalar expectations, alpha doesn't matter."""
        exps = {0: 1.0, 1: 3.0, 2: 2.0}
        assert alpha_utility(exps, alpha=0.0) == 1
        assert alpha_utility(exps, alpha=0.5) == 1
        assert alpha_utility(exps, alpha=1.0) == 1

    def test_alpha_utility_pessimistic(self):
        """α=0 (maximin): select best worst-case."""
        exps = {
            "A": np.array([1.0, 5.0]),  # min = 1
            "B": np.array([2.0, 3.0]),  # min = 2
            "C": np.array([0.0, 10.0]),  # min = 0
        }
        assert alpha_utility(exps, alpha=0.0) == "B"

    def test_alpha_utility_optimistic(self):
        """α=1 (maximax): select best best-case."""
        exps = {
            "A": np.array([1.0, 5.0]),  # max = 5
            "B": np.array([3.0, 3.0]),  # max = 3
        }
        assert alpha_utility(exps, alpha=1.0) == "A"

    def test_alpha_utility_interpolates(self):
        exps = {
            "A": np.array([1.0, 5.0]),  # α=1 → 5, α=0 → 1
            "B": np.array([3.0, 3.0]),  # always 3
        }
        # Fully optimistic: A wins (5 > 3)
        assert alpha_utility(exps, alpha=1.0) == "A"
        # Fully pessimistic: B wins (3 > 1)
        assert alpha_utility(exps, alpha=0.0) == "B"

    def test_alpha_utility_with_tuples(self):
        exps = {"A": (1.0, 5.0), "B": (3.0, 3.0)}
        assert alpha_utility(exps, alpha=1.0) == "A"
        assert alpha_utility(exps, alpha=0.0) == "B"

    def test_alpha_regret_pessimistic(self):
        """α=0 (minimax regret)."""
        exps = {
            "A": np.array([4.0, 0.0]),
            "B": np.array([3.0, 3.0]),
        }
        # Scenario 0: best=A(4). Regret: A=0, B=1
        # Scenario 1: best=B(3). Regret: A=3, B=0
        # Max regret: A=3, B=1 → B wins
        assert alpha_regret(exps, alpha=0.0) == "B"

    def test_alpha_regret_with_tuples(self):
        exps = {"A": (0.0, 4.0), "B": (3.0, 3.0)}
        assert alpha_regret(exps, alpha=0.0) == "B"

    def test_alpha_regret_optimistic(self):
        """α=1 (minimin regret)."""
        exps = {
            "A": np.array([5.0, 0.0]),
            "B": np.array([3.0, 3.0]),
            "C": np.array([4.0, 2.0]),
        }
        # Scenario 0: best=A(5). Regret: A=0, B=2, C=1
        # Scenario 1: best=B(3). Regret: A=3, B=0, C=1
        # Min regret: A=0, B=0, C=1
        # α=1 (minimin): score = min regret → A or B (both 0)
        # α=0 (minimax): max regret: A=3, B=2, C=1 → C wins
        assert alpha_regret(exps, alpha=0.0) == "C"
        assert alpha_regret(exps, alpha=1.0) in ("A", "B")

    def test_alpha_regret_interpolates(self):
        """Intermediate α should interpolate between extremes."""
        exps = {
            "A": np.array([5.0, 0.0]),
            "B": np.array([3.0, 3.0]),
            "C": np.array([4.0, 2.0]),
        }
        # Scenario 0: best=A(5). Regret: A=0, B=2, C=1
        # Scenario 1: best=B(3). Regret: A=3, B=0, C=1
        # α=0.5: score = 0.5*min + 0.5*max
        #   A: 0.5*0 + 0.5*3 = 1.5
        #   B: 0.5*0 + 0.5*2 = 1.0
        #   C: 0.5*1 + 0.5*1 = 1.0
        # B or C win (both score 1.0)
        result = alpha_regret(exps, alpha=0.5)
        assert result in ("B", "C")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


class TestCPSDecision:
    def test_returns_valid_decision(self, ridge_cpd, squared_error_utility):
        d = cps_decision(ridge_cpd, squared_error_utility, x=None, tau=0.5)
        assert d in squared_error_utility.decisions


class TestVennDecision:
    def test_utility_pessimistic(self, multiclass_venn_pred, zero_one_utility):
        d = venn_decision(multiclass_venn_pred, zero_one_utility, x=None, criterion="utility", alpha=0.0)
        assert d in zero_one_utility.decisions

    def test_utility_midpoint(self, multiclass_venn_pred, zero_one_utility):
        """α=0.5 (midpoint) should pick a valid decision."""
        d = venn_decision(multiclass_venn_pred, zero_one_utility, x=None, criterion="utility", alpha=0.5)
        assert d in zero_one_utility.decisions

    def test_invalid_criterion_raises(self, multiclass_venn_pred, zero_one_utility):
        with pytest.raises(ValueError, match="Unknown criterion"):
            venn_decision(multiclass_venn_pred, zero_one_utility, x=None, criterion="bogus")


# ---------------------------------------------------------------------------
# ConformalPredictiveDecisionMaker (V&B 2018 Algorithm 1)
# ---------------------------------------------------------------------------


class TestConformalPredictiveDecisionMaker:
    """Tests for the exact V&B 2018 Algorithm 1 implementation."""

    @pytest.fixture
    def binary_utility(self):
        """3-decision utility for binary classification (treat/dismiss/defer)."""
        cost_matrix = [[+8, -10, +3], [-3, +5, +1]]  # [y][d]

        def u(x, y, d):
            return cost_matrix[int(y)][int(d)]

        return UtilityFunction(u, decisions=[0, 1, 2])

    @pytest.fixture
    def binary_data(self):
        """Simple binary classification data."""
        rng = np.random.default_rng(123)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        return X, y

    def test_init_creates_models(self, binary_utility):
        from online_cp.decision import ConformalPredictiveDecisionMaker

        cdm = ConformalPredictiveDecisionMaker(binary_utility, a=1.0)
        assert len(cdm._models) == 3

    def test_learn_predict_cycle(self, binary_utility, binary_data):
        from online_cp.decision import ConformalPredictiveDecisionMaker

        X, y = binary_data
        cdm = ConformalPredictiveDecisionMaker(binary_utility, a=1.0)
        cdm.learn_initial_training_set(X[:40], y[:40])
        d = cdm.predict(X[40])
        assert d in [0, 1, 2]

    def test_predict_expected_utilities(self, binary_utility, binary_data):
        from online_cp.decision import ConformalPredictiveDecisionMaker

        X, y = binary_data
        cdm = ConformalPredictiveDecisionMaker(binary_utility, a=1.0)
        cdm.learn_initial_training_set(X[:40], y[:40])
        eus = cdm.predict_expected_utilities(X[40])
        assert set(eus.keys()) == {0, 1, 2}
        assert all(isinstance(v, float) for v in eus.values())

    def test_learn_one_updates(self, binary_utility, binary_data):
        from online_cp.decision import ConformalPredictiveDecisionMaker

        X, y = binary_data
        cdm = ConformalPredictiveDecisionMaker(binary_utility, a=1.0)
        cdm.learn_initial_training_set(X[:40], y[:40])

        # Predict before and after learn_one
        d_before = cdm.predict_expected_utilities(X[41])
        cdm.learn_one(X[40], y[40])
        d_after = cdm.predict_expected_utilities(X[41])

        # At least one utility should change
        assert any(
            abs(d_before[d] - d_after[d]) > 1e-10
            for d in [0, 1, 2]
        )

    def test_online_loop_reasonable_utility(self, binary_utility, binary_data):
        """Run a short online loop and verify cumulative utility is positive."""
        from online_cp.decision import ConformalPredictiveDecisionMaker

        X, y = binary_data
        cdm = ConformalPredictiveDecisionMaker(binary_utility, a=1.0)
        cdm.learn_initial_training_set(X[:30], y[:30])

        total = 0.0
        for i in range(30, 60):
            d = cdm.predict(X[i], tau=0.5)
            total += binary_utility(X[i], y[i], d)
            cdm.learn_one(X[i], y[i])

        # With this utility matrix and reasonable classification,
        # cumulative utility should be positive
        assert total > 0


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """Ensure out-of-range alpha and tau raise ValueError."""

    def test_alpha_utility_out_of_range(self):
        exps = {"A": np.array([1.0, 5.0]), "B": np.array([3.0, 3.0])}
        with pytest.raises(ValueError, match="alpha must be in"):
            alpha_utility(exps, alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be in"):
            alpha_utility(exps, alpha=1.1)

    def test_alpha_regret_out_of_range(self):
        exps = {"A": np.array([4.0, 0.0]), "B": np.array([3.0, 3.0])}
        with pytest.raises(ValueError, match="alpha must be in"):
            alpha_regret(exps, alpha=-0.01)
        with pytest.raises(ValueError, match="alpha must be in"):
            alpha_regret(exps, alpha=2.0)

    def test_cps_expected_utilities_tau_out_of_range(self, ridge_cpd, squared_error_utility):
        with pytest.raises(ValueError, match="tau must be in"):
            cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=-0.5)
        with pytest.raises(ValueError, match="tau must be in"):
            cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=1.5)

    def test_venn_decision_alpha_out_of_range(self, multiclass_venn_pred, zero_one_utility):
        with pytest.raises(ValueError, match="alpha must be in"):
            venn_decision(multiclass_venn_pred, zero_one_utility, x=None, alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be in"):
            venn_decision(multiclass_venn_pred, zero_one_utility, x=None, alpha=1.5)

    def test_cpdm_predict_tau_out_of_range(self):
        from online_cp.decision import ConformalPredictiveDecisionMaker

        utility = UtilityFunction(lambda x, y, d: -(y - d) ** 2, decisions=[0, 1])
        cdm = ConformalPredictiveDecisionMaker(utility, a=1.0)
        X = np.random.default_rng(0).normal(size=(20, 2))
        y = X[:, 0]
        cdm.learn_initial_training_set(X, y)

        with pytest.raises(ValueError, match="tau must be in"):
            cdm.predict(X[0], tau=-0.1)
        with pytest.raises(ValueError, match="tau must be in"):
            cdm.predict_expected_utilities(X[0], tau=1.1)

    def test_alpha_utility_boundary_values(self):
        """alpha=0.0 and alpha=1.0 should work without error."""
        exps = {"A": np.array([1.0, 5.0]), "B": np.array([3.0, 3.0])}
        assert alpha_utility(exps, alpha=0.0) is not None
        assert alpha_utility(exps, alpha=1.0) is not None

    def test_tau_boundary_values(self, ridge_cpd, squared_error_utility):
        """tau=0.0 and tau=1.0 should work without error."""
        exps0 = cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=0.0)
        exps1 = cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=1.0)
        assert all(np.isfinite(v) for v in exps0.values())
        assert all(np.isfinite(v) for v in exps1.values())


class TestCPSDecisionMaximizes:
    """Verify cps_decision returns the decision with highest expected utility."""

    def test_returns_argmax(self, ridge_cpd, squared_error_utility):
        exps = cps_expected_utilities(ridge_cpd, squared_error_utility, x=None, tau=0.5)
        expected = max(exps, key=exps.get)
        actual = cps_decision(ridge_cpd, squared_error_utility, x=None, tau=0.5)
        assert actual == expected


class TestUtilityFunctionValidation:
    """UtilityFunction input validation."""

    def test_empty_decisions_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            UtilityFunction(lambda x, y, d: 0, decisions=[])

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            UtilityFunction("not a function", decisions=[0, 1])

    def test_repr(self):
        u = UtilityFunction(lambda x, y, d: 0, decisions=[0, 1, 2])
        assert repr(u) == "UtilityFunction(|D|=3)"
