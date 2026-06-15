"""Tests for ConformalSupportVectorMachine."""

import numpy as np
import pytest

from online_cp.classifiers import ConformalSupportVectorMachine, MultiLevelPredictionSet, _smo_solve
from online_cp.kernels import GaussianKernel, LinearKernel

# --- SMO solver tests ---


class TestSMOSolver:
    """Tests for the underlying SMO QP solver."""

    def test_linearly_separable(self):
        """On perfectly separable data, all alphas should be < C."""
        np.random.seed(0)
        n = 20
        X = np.vstack([np.random.normal(-2, 0.3, (n, 2)), np.random.normal(2, 0.3, (n, 2))])
        y = np.array([-1.0] * n + [1.0] * n)
        K = X @ X.T  # linear kernel
        C = 100.0
        alpha, b = _smo_solve(K, y, C)

        # Feasibility: 0 <= alpha <= C
        assert np.all(alpha >= -1e-10)
        assert np.all(alpha <= C + 1e-10)
        # Sum constraint: sum(alpha_i * y_i) ≈ 0
        assert abs(np.dot(alpha, y)) < 1e-5
        # Not all alphas at bound (separable)
        assert np.all(alpha < C - 1e-3)

    def test_soft_margin(self):
        """With overlapping data and small C, some alphas should hit C."""
        np.random.seed(1)
        n = 30
        X = np.vstack([np.random.normal(-0.5, 1.0, (n, 2)), np.random.normal(0.5, 1.0, (n, 2))])
        y = np.array([-1.0] * n + [1.0] * n)
        K = X @ X.T
        C = 0.5
        alpha, b = _smo_solve(K, y, C)

        assert np.all(alpha >= -1e-10)
        assert np.all(alpha <= C + 1e-10)
        assert abs(np.dot(alpha, y)) < 1e-4
        # At least some alphas at bound
        assert np.any(alpha > C - 1e-3)

    def test_warm_start(self):
        """Warm start should converge to similar solution."""
        np.random.seed(2)
        n = 20
        X = np.vstack([np.random.normal(-1, 0.5, (n, 2)), np.random.normal(1, 0.5, (n, 2))])
        y = np.array([-1.0] * n + [1.0] * n)
        K = X @ X.T
        C = 1.0

        alpha1, b1 = _smo_solve(K, y, C, max_iter=5000)
        # Use warm start close to solution
        warm = alpha1 * 0.9
        alpha2, b2 = _smo_solve(K, y, C, warm_start=warm, max_iter=5000)

        # Both should satisfy KKT and yield similar objective values
        obj1 = np.sum(alpha1) - 0.5 * (alpha1 * y) @ K @ (alpha1 * y)
        obj2 = np.sum(alpha2) - 0.5 * (alpha2 * y) @ K @ (alpha2 * y)
        assert abs(obj1 - obj2) < 0.1


# --- ConformalSupportVectorMachine tests ---


class TestConformalSVM:
    """Tests for the conformal SVM classifier."""

    @pytest.fixture
    def separable_data(self):
        """Well-separated two-class data."""
        np.random.seed(42)
        n = 30
        X = np.vstack([np.random.normal(-2, 0.5, (n, 2)), np.random.normal(2, 0.5, (n, 2))])
        y = np.array([-1] * n + [1] * n)
        return X, y

    @pytest.fixture
    def overlapping_data(self):
        """Overlapping two-class data."""
        np.random.seed(123)
        n = 40
        X = np.vstack([np.random.normal(-0.3, 1.0, (n, 2)), np.random.normal(0.3, 1.0, (n, 2))])
        y = np.array([-1] * n + [1] * n)
        return X, y

    def test_prediction_returns_set(self, separable_data):
        """predict() should return a ConformalPredictionSet."""
        X, y = separable_data
        from online_cp.classifiers import ConformalPredictionSet

        svm = ConformalSupportVectorMachine(kernel=LinearKernel(), C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma = svm.predict(X[40])
        assert isinstance(Gamma, ConformalPredictionSet)

    def test_correct_label_in_set(self, separable_data):
        """On well-separated data with small epsilon, errors should be rare."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel=GaussianKernel(sigma=1.0), C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        errors = 0
        for i in range(40, 60):
            Gamma = svm.predict(X[i], epsilon=0.05)
            if y[i] not in Gamma:
                errors += 1
        # With epsilon=0.05 over 20 predictions, expect ~1 error
        # Allow up to 4 (very generous for stability)
        assert errors <= 4, f"Too many errors: {errors}/20"

    def test_p_values_returned(self, separable_data):
        """return_p_values=True should return dict of p-values."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel=LinearKernel(), C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma, p_vals = svm.predict(X[40], return_p_values=True)
        assert isinstance(p_vals, dict)
        assert -1 in p_vals and 1 in p_vals
        # p-values in [0, 1]
        for v in p_vals.values():
            assert 0 <= v <= 1

    def test_coverage_statistical(self, overlapping_data):
        """Coverage should be approximately 1-epsilon over many predictions."""
        X, y = overlapping_data
        epsilon = 0.2
        svm = ConformalSupportVectorMachine(kernel=GaussianKernel(sigma=1.0), C=1.0, epsilon=epsilon, rnd_state=42)
        n_train = 30
        svm.learn_initial_training_set(X[:n_train], y[:n_train])

        correct = 0
        n_test = len(X) - n_train
        for i in range(n_train, len(X)):
            Gamma = svm.predict(X[i], epsilon=epsilon)
            if y[i] in Gamma:
                correct += 1
            svm.learn_one(X[i], y[i])

        coverage = correct / n_test
        # Conformal guarantee: coverage >= 1 - epsilon in expectation
        # Allow some slack for small sample
        assert coverage >= 0.5, f"Coverage too low: {coverage}"

    def test_kernel_string_rbf(self, separable_data):
        """String kernel 'rbf' should work."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel="rbf", sigma=1.0, C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma = svm.predict(X[40])
        assert y[40] in Gamma

    def test_kernel_string_linear(self, separable_data):
        """String kernel 'linear' should work."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel="linear", C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma = svm.predict(X[40])
        assert y[40] in Gamma

    def test_kernel_string_poly(self, separable_data):
        """String kernel 'poly' should work."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel="poly", degree=2, C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma = svm.predict(X[40])
        assert y[40] in Gamma

    def test_kernel_callable(self, separable_data):
        """Sklearn-style kernel callable should work."""
        X, y = separable_data

        def my_rbf(X, Y):
            from scipy.spatial.distance import cdist

            dists = cdist(X, Y, "sqeuclidean")
            return np.exp(-dists / 2.0)

        svm = ConformalSupportVectorMachine(kernel=my_rbf, C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma = svm.predict(X[40])
        assert y[40] in Gamma

    def test_kernel_native(self, separable_data):
        """Native online_cp kernel should work."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel=GaussianKernel(sigma=2.0), C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        Gamma = svm.predict(X[40])
        assert y[40] in Gamma

    def test_learn_one_extends_gram(self, separable_data):
        """learn_one should extend the cached Gram matrix correctly."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel=LinearKernel(), C=1.0, rnd_state=0)
        svm.learn_initial_training_set(X[:10], y[:10])
        assert svm.K.shape == (10, 10)
        svm.learn_one(X[10], y[10])
        assert svm.K.shape == (11, 11)
        # Check symmetry
        np.testing.assert_allclose(svm.K, svm.K.T)
        # Check against recomputed
        K_expected = svm._compute_gram(svm.X)
        np.testing.assert_allclose(svm.K, K_expected, atol=1e-12)

    def test_invalid_kernel_string_raises(self):
        """Invalid kernel string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel string"):
            ConformalSupportVectorMachine(kernel="invalid")

    def test_invalid_kernel_type_raises(self):
        """Invalid kernel type should raise TypeError."""
        with pytest.raises(TypeError):
            ConformalSupportVectorMachine(kernel=42)

    def test_empty_training_set(self):
        """Prediction with no training data should return all labels."""
        svm = ConformalSupportVectorMachine(kernel="linear", label_space=np.array([-1, 1]), rnd_state=0)
        Gamma = svm.predict(np.array([1.0, 2.0]))
        assert -1 in Gamma and 1 in Gamma

    def test_compute_p_value_empty_training_set(self):
        """compute_p_value should return tau with no training data."""
        svm = ConformalSupportVectorMachine(kernel="linear", label_space=np.array([-1, 1]), rnd_state=0)
        p = svm.compute_p_value(np.array([1.0, 2.0]), 1)
        assert 0 < p < 1  # tau ~ Uniform(0, 1)

    def test_compute_p_value_binary_in_unit_interval(self, separable_data):
        """Binary compute_p_value should produce a valid p-value."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel=LinearKernel(), C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])
        p_value = svm.compute_p_value(X[40], y[40])
        assert 0 <= p_value <= 1

    def test_compute_p_value_matches_predict_binary(self, separable_data):
        """compute_p_value should match the corresponding predict() p-value in binary problems."""
        X, y = separable_data

        svm_predict = ConformalSupportVectorMachine(kernel=LinearKernel(), C=10.0, rnd_state=123)
        svm_predict.learn_initial_training_set(X[:40], y[:40])
        _, p_values = svm_predict.predict(X[40], return_p_values=True)

        svm_single = ConformalSupportVectorMachine(kernel=LinearKernel(), C=10.0, rnd_state=123)
        svm_single.learn_initial_training_set(X[:40], y[:40])
        p_value = svm_single.compute_p_value(X[40], y[40])

        assert np.isclose(p_value, p_values[y[40]], atol=1e-12)

    def test_multi_level_epsilon(self, separable_data):
        """predict with multiple epsilon levels should return a MultiLevelPredictionSet."""
        X, y = separable_data
        svm = ConformalSupportVectorMachine(kernel=GaussianKernel(sigma=1.0), C=10.0, rnd_state=0)
        svm.learn_initial_training_set(X[:40], y[:40])

        epsilons = [0.01, 0.05, 0.1, 0.2]
        result = svm.predict(X[40], epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert result.levels == sorted(epsilons)
        for i in range(len(epsilons) - 1):
            assert len(result[epsilons[i]]) >= len(result[epsilons[i + 1]])


# --- Multi-class tests ---


class TestMultiClassSVM:
    """Tests for multi-class conformal SVM."""

    @pytest.fixture
    def three_class_data(self):
        """Three well-separated Gaussian blobs with interleaved indices."""
        np.random.seed(7)
        n = 30
        X0 = np.random.normal(loc=[0, 3], scale=0.5, size=(n, 2))
        X1 = np.random.normal(loc=[-3, -1], scale=0.5, size=(n, 2))
        X2 = np.random.normal(loc=[3, -1], scale=0.5, size=(n, 2))
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * n + [1] * n + [2] * n)
        # Shuffle so that training split includes all classes
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    @pytest.fixture
    def four_class_data(self):
        """Four classes with arbitrary label names, shuffled."""
        np.random.seed(11)
        n = 20
        X = np.vstack(
            [
                np.random.normal(loc=[2, 2], scale=0.4, size=(n, 2)),
                np.random.normal(loc=[-2, 2], scale=0.4, size=(n, 2)),
                np.random.normal(loc=[-2, -2], scale=0.4, size=(n, 2)),
                np.random.normal(loc=[2, -2], scale=0.4, size=(n, 2)),
            ]
        )
        y = np.array([10] * n + [20] * n + [30] * n + [40] * n)
        # Shuffle so training includes all classes
        rng = np.random.default_rng(11)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def test_multiclass_basic(self, three_class_data):
        """Three-class SVM should return a valid prediction set."""
        X, y = three_class_data
        from online_cp.classifiers import ConformalPredictionSet

        svm = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, label_space=np.array([0, 1, 2]), rnd_state=0
        )
        svm.learn_initial_training_set(X[:60], y[:60])
        Gamma = svm.predict(X[60])
        assert isinstance(Gamma, ConformalPredictionSet)
        # True label should be in set (well-separated, default epsilon=0.1)
        assert y[60] in Gamma

    def test_multiclass_p_values(self, three_class_data):
        """P-values should be returned for all candidate labels."""
        X, y = three_class_data
        svm = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, label_space=np.array([0, 1, 2]), rnd_state=0
        )
        svm.learn_initial_training_set(X[:60], y[:60])
        Gamma, p_vals = svm.predict(X[60], return_p_values=True)
        assert set(p_vals.keys()) == {0, 1, 2}
        for v in p_vals.values():
            assert 0 <= v <= 1
        # True label should have highest p-value (well-separated data)
        assert p_vals[y[60]] == max(p_vals.values())

    def test_multiclass_coverage(self, three_class_data):
        """Coverage should be approximately 1-epsilon for 3 classes."""
        X, y = three_class_data
        epsilon = 0.2
        svm = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=5.0, label_space=np.array([0, 1, 2]), epsilon=epsilon, rnd_state=42
        )
        n_train = 50
        svm.learn_initial_training_set(X[:n_train], y[:n_train])

        correct = 0
        n_test = len(X) - n_train
        for i in range(n_train, len(X)):
            Gamma = svm.predict(X[i], epsilon=epsilon)
            if y[i] in Gamma:
                correct += 1
            svm.learn_one(X[i], y[i])

        coverage = correct / n_test
        # Should cover at least 60% (generous for small sample with eps=0.2)
        assert coverage >= 0.6, f"Coverage too low: {coverage:.3f}"

    def test_multiclass_arbitrary_labels(self, four_class_data):
        """Arbitrary label values (10, 20, 30, 40) should work."""
        X, y = four_class_data
        svm = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, label_space=np.array([10, 20, 30, 40]), rnd_state=0
        )
        svm.learn_initial_training_set(X[:60], y[:60])
        Gamma, p_vals = svm.predict(X[60], return_p_values=True)
        assert set(p_vals.keys()) == {10, 20, 30, 40}
        assert y[60] in Gamma

    def test_compute_p_value_multiclass_in_unit_interval(self, three_class_data):
        """Multiclass compute_p_value should produce a valid p-value."""
        X, y = three_class_data
        svm = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, label_space=np.array([0, 1, 2]), rnd_state=0
        )
        svm.learn_initial_training_set(X[:60], y[:60])
        p_value = svm.compute_p_value(X[60], y[60])
        assert 0 <= p_value <= 1

    def test_compute_p_value_matches_predict_multiclass(self, three_class_data):
        """compute_p_value should match the corresponding predict() p-value in multiclass problems."""
        X, y = three_class_data

        svm_predict = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, label_space=np.array([0, 1, 2]), rnd_state=123
        )
        svm_predict.learn_initial_training_set(X[:60], y[:60])
        _, p_values = svm_predict.predict(X[60], return_p_values=True)

        svm_single = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, label_space=np.array([0, 1, 2]), rnd_state=123
        )
        svm_single.learn_initial_training_set(X[:60], y[:60])
        p_value = svm_single.compute_p_value(X[60], y[60])

        assert np.isclose(p_value, p_values[y[60]], atol=1e-12)

    def test_multiclass_positive_class_only_p_value_regression(self, three_class_data):
        """With alpha NCM, multiclass p-values must use positive-class alphas only (validity fix)."""
        X, y = three_class_data
        label_space = np.array([0, 1, 2])
        label = y[60]

        svm = ConformalSupportVectorMachine(
            kernel=GaussianKernel(sigma=1.0), C=10.0, nonconformity="alpha",
            label_space=label_space, rnd_state=123
        )
        svm.learn_initial_training_set(X[:60], y[:60])

        x = X[60]
        tau = np.random.default_rng(123).uniform()
        k_row = svm._compute_kernel_row(svm.X, x)
        kappa = svm._kernel(x.reshape(1, -1))
        if np.ndim(kappa) > 0:
            kappa = kappa.item()

        n = svm.K.shape[0]
        K_aug = np.empty((n + 1, n + 1))
        K_aug[:n, :n] = svm.K
        K_aug[:n, n] = k_row
        K_aug[n, :n] = k_row
        K_aug[n, n] = kappa

        y_aug = np.append(svm.y, float(label))
        y_binary = np.where(y_aug == label, 1.0, -1.0)
        alpha, _ = _smo_solve(K_aug, y_binary, svm.C, tol=svm.smo_tol, max_iter=svm.smo_max_iter)

        expected = svm._compute_p_value(alpha[y_binary == 1.0], tau, "nonconformity")
        full_alpha = svm._compute_p_value(alpha, tau, "nonconformity")
        observed = svm.compute_p_value(x, label)

        assert np.isclose(observed, expected, atol=1e-12)
        assert not np.isclose(expected, full_alpha, atol=1e-12)

    def test_multiclass_learn_one(self, three_class_data):
        """learn_one should work with multi-class labels."""
        X, y = three_class_data
        svm = ConformalSupportVectorMachine(kernel=LinearKernel(), C=1.0, label_space=np.array([0, 1, 2]), rnd_state=0)
        svm.learn_initial_training_set(X[:10], y[:10])
        svm.learn_one(X[10], y[10])
        assert svm.K.shape == (11, 11)
        assert svm.y[-1] == y[10]


# --- NCM parameter tests ---


class TestNonconformityParameter:
    """Tests for the nonconformity= parameter ('margin' vs 'alpha')."""

    @pytest.fixture
    def noisy_data(self):
        """Overlapping two-class data where efficiency differences are visible."""
        np.random.seed(5)
        n = 60
        X = np.vstack([np.random.normal(-0.5, 1.0, (n, 2)), np.random.normal(0.5, 1.0, (n, 2))])
        y = np.array([-1] * n + [1] * n)
        rng = np.random.default_rng(5)
        idx = rng.permutation(2 * n)
        return X[idx], y[idx]

    def test_default_is_margin(self):
        """Default nonconformity should be 'margin'."""
        svm = ConformalSupportVectorMachine()
        assert svm.nonconformity == "margin"

    def test_invalid_nonconformity_raises(self):
        """Invalid nonconformity string should raise ValueError."""
        with pytest.raises(ValueError, match="nonconformity must be"):
            ConformalSupportVectorMachine(nonconformity="decision_function")

    def test_alpha_ncm_valid(self, noisy_data):
        """alpha NCM should still produce valid coverage."""
        X, y = noisy_data
        epsilon = 0.2
        svm = ConformalSupportVectorMachine(
            kernel="rbf", sigma=1.0, C=1.0, nonconformity="alpha",
            label_space=np.array([-1, 1]), epsilon=epsilon, rnd_state=0,
        )
        n_train = 40
        svm.learn_initial_training_set(X[:n_train], y[:n_train])
        correct = sum(
            y[i] in svm.predict(X[i], epsilon=epsilon)
            for i in range(n_train, len(X))
        )
        coverage = correct / (len(X) - n_train)
        assert coverage >= 0.5, f"alpha NCM coverage too low: {coverage:.3f}"

    def test_margin_ncm_valid(self, noisy_data):
        """margin NCM (default) should produce valid coverage."""
        X, y = noisy_data
        epsilon = 0.2
        svm = ConformalSupportVectorMachine(
            kernel="rbf", sigma=1.0, C=1.0, nonconformity="margin",
            label_space=np.array([-1, 1]), epsilon=epsilon, rnd_state=0,
        )
        n_train = 40
        svm.learn_initial_training_set(X[:n_train], y[:n_train])
        correct = sum(
            y[i] in svm.predict(X[i], epsilon=epsilon)
            for i in range(n_train, len(X))
        )
        coverage = correct / (len(X) - n_train)
        assert coverage >= 0.5, f"margin NCM coverage too low: {coverage:.3f}"

    def test_margin_more_efficient_than_alpha_on_noisy(self, noisy_data):
        """On noisy data, margin NCM should produce smaller or equal avg set size than alpha NCM."""
        X, y = noisy_data
        epsilon = 0.1
        n_train = 40
        n_test = len(X) - n_train

        def avg_size(ncm):
            svm = ConformalSupportVectorMachine(
                kernel="rbf", sigma=1.0, C=1.0, nonconformity=ncm,
                label_space=np.array([-1, 1]), epsilon=epsilon, rnd_state=42,
            )
            svm.learn_initial_training_set(X[:n_train], y[:n_train])
            total = sum(len(svm.predict(X[i], epsilon=epsilon)) for i in range(n_train, len(X)))
            return total / n_test

        size_margin = avg_size("margin")
        size_alpha = avg_size("alpha")
        # margin should produce fewer or equal total included labels on noisy data
        assert size_margin <= size_alpha + 0.2, (
            f"margin avg_size={size_margin:.3f} not better than alpha avg_size={size_alpha:.3f}"
        )

    def test_p_values_differ_between_ncms(self, noisy_data):
        """The two NCMs should generally produce different p-values (scores differ)."""
        X, y = noisy_data
        n_train = 40
        svm_m = ConformalSupportVectorMachine(
            kernel="rbf", sigma=1.0, C=1.0, nonconformity="margin",
            label_space=np.array([-1, 1]), rnd_state=99,
        )
        svm_a = ConformalSupportVectorMachine(
            kernel="rbf", sigma=1.0, C=1.0, nonconformity="alpha",
            label_space=np.array([-1, 1]), rnd_state=99,
        )
        svm_m.learn_initial_training_set(X[:n_train], y[:n_train])
        svm_a.learn_initial_training_set(X[:n_train], y[:n_train])
        _, pv_m = svm_m.predict(X[n_train], return_p_values=True)
        _, pv_a = svm_a.predict(X[n_train], return_p_values=True)
        # At least one label should have different p-values
        assert any(abs(pv_m[lbl] - pv_a[lbl]) > 1e-6 for lbl in [-1, 1])
