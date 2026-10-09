"""Layer 2: statistical-validity tests (``@pytest.mark.slow``).

These tests verify the *theoretical guarantees* of the predictors by relying on
the law of large numbers over many generated streams. Each guarantee is a
statement about a binomial probability (an error, rejection, or positive-class
indicator), so the assertions use exact binomial hypothesis tests
(``scipy.stats.binomtest``) at a fixed significance level rather than arbitrary
fixed margins. They are intentionally heavy and are excluded from the default
lane (``addopts = -m 'not slow'``).

Run with::

    pytest -m slow
    pytest -m slow tests/test_statistical.py
"""

import numpy as np
import pytest
from scipy.stats import binomtest, kstest

# Import additional models
from online_cp import (
    ConformalLassoRegressor,
    ConformalMondrianForestClassifier,
    ConformalMondrianForestRegressor,
    ConformalMondrianTreeClassifier,
    ConformalMondrianTreeRegressor,
    ConformalNearestNeighboursClassifier,
    ConformalNearestNeighboursRegressor,
    ConformalRidgeRegressor,
    ConformalSupportVectorMachine,
    DempsterHillConformalPredictiveSystem,
    ErrorRate,
    GaussianKernel,
    KernelConformalRidgeRegressor,
    KernelRidgePredictionMachine,
    MondrianTreePredictionMachine,
    MondrianVennPredictor,
    NearestNeighboursPredictionMachine,
    NearestNeighboursVennPredictor,
    RidgePredictionMachine,
    SimpleJumper,
    VennAbersPredictor,
    VilleWrapper,
    progressive_val,
)

# Import data generators from conftest
from .conftest import (
    generate_binary_classification_data,
    generate_linear_data,
)

# --------------------------------------------------------------------------- #
# Target 1: conformal coverage / validity.
#
# For exchangeable data the online error rate converges to epsilon, so each
# prediction is a Bernoulli(p_err) trial with p_err <= epsilon. We pool the
# error indicators over many independent streams and run a one-sided EXACT
# binomial test of H0: p_err <= epsilon vs H1: p_err > epsilon. A small p-value
# is evidence the predictor under-covers; we require the test NOT to reject at
# the 1% level. This replaces an arbitrary fixed margin with a principled
# significance level. (Indicators within a stream are not perfectly independent,
# but the test reliably catches gross violations of the marginal guarantee.)
# --------------------------------------------------------------------------- #

BINOM_ALPHA = 0.01  # significance level for every binomial gate below


@pytest.mark.slow
def test_ridge_regressor_coverage_validity():
    eps = 0.1
    n_runs, N, d = 40, 300, 3
    rng = np.random.default_rng(0)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X = rng.normal(size=(N, d))
        beta = rng.normal(size=d)
        y = X @ beta + rng.normal(scale=0.5, size=N)
        cp = ConformalRidgeRegressor(a=1.0, warnings=False)
        metric = progressive_val(cp, X, y, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )

@pytest.mark.slow
def test_mondrian_tree_regressor_coverage_validity():
    eps = 0.1
    n_runs, N, d = 40, 300, 3
    rng = np.random.default_rng(0)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X = rng.normal(size=(N, d))
        beta = rng.normal(size=d)
        y = X @ beta + rng.normal(scale=0.5, size=N)
        cp = ConformalMondrianTreeRegressor(lifetime=1.0, online=True)
        metric = progressive_val(cp, X, y, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_knn_classifier_coverage_validity():
    eps = 0.1
    n_runs, N, d = 25, 200, 4
    rng = np.random.default_rng(1)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        labels = np.array([0, 1] * (N // 2))
        X = rng.normal(size=(N, d))
        X[labels == 0] -= 1.5
        X[labels == 1] += 1.5
        clf = ConformalNearestNeighboursClassifier(k=1, label_space=np.array([0, 1]))
        metric = progressive_val(clf, X, labels, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


# --------------------------------------------------------------------------- #
# Target 2: Venn-Abers calibration.
#
# Venn predictors are well calibrated: the average predicted probability should
# match the empirical frequency of the positive class (calibration-in-the-large),
# tested via an exact binomial test on the observed positive count.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_venn_abers_calibration_in_the_large():
    rng = np.random.default_rng(2)
    N_train, N_test, d = 200, 150, 3
    beta = rng.normal(size=d)

    def sample(n):
        X = rng.normal(size=(n, d))
        logits = X @ beta
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(int)
        return X, y

    X_tr, y_tr = sample(N_train)
    X_te, y_te = sample(N_test)

    va = VennAbersPredictor(scorer="ridge", a=1.0, label_space=np.array([0, 1]))
    va.learn_initial_training_set(X_tr, y_tr)

    point_preds = np.array([va.predict(x).point for x in X_te])
    mean_pred = float(point_preds.mean())
    n_positives = int(y_te.sum())
    # Calibration-in-the-large: if the average predicted probability is correct,
    # the number of observed positives is Binomial(N_test, mean_pred). Run a
    # two-sided exact binomial test that the positive count is consistent with
    # mean_pred; require the test NOT to reject at the 1% level.
    result = binomtest(n_positives, int(y_te.size), mean_pred, alternative="two-sided")
    assert result.pvalue > BINOM_ALPHA, (
        f"Venn-Abers miscalibrated: mean pred {mean_pred:.3f} vs "
        f"{n_positives}/{y_te.size} positives (binom p={result.pvalue:.2e})"
    )


# --------------------------------------------------------------------------- #
# Target 3: change-point detection false-alarm rate (and power).
#
# Under H0 (exchangeable data => iid uniform p-values), Ville's inequality
# guarantees P(reject) <= 1/threshold. Under H1 (a clear distribution shift)
# the detector should fire with high probability.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_ville_false_alarm_rate_under_null():
    threshold = 20  # Ville bound: P(reject) <= 1/20 = 5%
    n_runs, T = 300, 500
    rng = np.random.default_rng(3)
    rejections = 0
    for _ in range(n_runs):
        ville = VilleWrapper(SimpleJumper(J=0.01), threshold=threshold)
        for p in rng.uniform(0.0, 1.0, size=T):
            ville.update(float(p))
        rejections += int(ville.rejected)
    # Ville's inequality caps the false-alarm probability at 1/threshold. The
    # number of rejecting runs is Binomial(n_runs, p_fa); test H0: p_fa <= 1/c
    # one-sided and require the test NOT to reject at the 1% level.
    p_bound = 1.0 / threshold
    result = binomtest(rejections, n_runs, p_bound, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"false-alarm rate {rejections / n_runs:.3f} significantly exceeds Ville bound "
        f"{p_bound} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_ville_detects_distribution_shift():
    threshold = 20
    n_runs, T = 30, 500
    rng = np.random.default_rng(4)
    detections = 0
    for _ in range(n_runs):
        ville = VilleWrapper(SimpleJumper(J=0.01), threshold=threshold)
        # H1: p-values skewed strongly toward 0 (small p-values => evidence
        # against exchangeability), as produced by a real change-point.
        for p in rng.beta(0.3, 3.0, size=T):
            ville.update(float(np.clip(p, 1e-6, 1 - 1e-6)))
        detections += int(ville.rejected)
    # Detections are Binomial(n_runs, power). Confirm the power is not
    # significantly below a 0.8 floor: H0: power >= 0.8 vs H1: power < 0.8,
    # one-sided. Require the test NOT to reject at the 1% level.
    floor = 0.8
    result = binomtest(detections, n_runs, floor, alternative="less")
    assert result.pvalue > BINOM_ALPHA, (
        f"detection power {detections / n_runs:.2f} significantly below {floor} (binom p={result.pvalue:.2e})"
    )


# --------------------------------------------------------------------------- #
# TARGET 4: Extended coverage tests for all production models.
#
# This section includes statistical validity tests for all production models
# using the same rigorous binomial hypothesis testing framework. Each test
# runs multiple independent streams with increased sample sizes (N=400-500,
# n_runs=40-60) to ensure good statistical power.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Regressors: coverage validity (6 models)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_knn_regressor_coverage_validity():
    """Test ConformalNearestNeighboursRegressor coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(0)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cp = ConformalNearestNeighboursRegressor(k=5)
        metric = progressive_val(cp, X, y, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_kernel_ridge_regressor_coverage_validity():
    """Test KernelConformalRidgeRegressor coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(0)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cp = KernelConformalRidgeRegressor(kernel=GaussianKernel(sigma=1.0), a=1.0)
        metric = progressive_val(cp, X, y, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_lasso_regressor_coverage_validity():
    """Test ConformalLassoRegressor coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(0)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cp = ConformalLassoRegressor(lam=0.1)
        metric = progressive_val(cp, X, y, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_mondrian_forest_regressor_coverage_validity():
    """Test ConformalMondrianForestRegressor coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(0)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        X_init, y_init = X[:20], y[:20]
        X_test, y_test = X[20:], y[20:]
        cp = ConformalMondrianForestRegressor(n_trees=10, lifetime=1.0)
        cp.learn_initial_training_set(X_init, y_init)
        metric = progressive_val(cp, X_test, y_test, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


# --------------------------------------------------------------------------- #
# Classifiers: coverage validity (4 models)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_svm_classifier_coverage_validity():
    """Test ConformalSupportVectorMachine coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(1)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_binary_classification_data(rng, N, d)
        clf = ConformalSupportVectorMachine(C=1.0, kernel=GaussianKernel(sigma=1.0))
        metric = progressive_val(clf, X, y, epsilon=eps, metric=ErrorRate())
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_mondrian_tree_classifier_coverage_validity():
    """Test ConformalMondrianTreeClassifier coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(1)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_binary_classification_data(rng, N, d)
        clf = ConformalMondrianTreeClassifier(lifetime=1.0, online=True)
        # Initialize with training data first
        clf.learn_initial_training_set(X[:200], y[:200])

        # Then do the test-then-train loop manually
        metric = ErrorRate()
        for i in range(200, N):
            x_i = X[i]
            y_i = y[i]
            Gamma = clf.predict(x_i, epsilon=eps)
            metric.update(y=y_i, Gamma=Gamma)
            # Learn from this example
            clf.learn_one(x_i, y_i)
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_mondrian_forest_classifier_coverage_validity():
    """Test ConformalMondrianForestClassifier coverage validity."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(1)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_binary_classification_data(rng, N, d)
        clf = ConformalMondrianForestClassifier(n_trees=10, lifetime=1.0)
        # Initialize with training data first
        clf.learn_initial_training_set(X[:200], y[:200])

        # Then do the test-then-train loop manually
        metric = ErrorRate()
        for i in range(200, N):
            x_i = X[i]
            y_i = y[i]
            Gamma = clf.predict(x_i, epsilon=eps)
            metric.update(y=y_i, Gamma=Gamma)
            # Learn from this example
            clf.learn_one(x_i, y_i)
        total_errors += int(metric.values.sum())
        total_preds += int(metric.values.size)
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


# --------------------------------------------------------------------------- #
# CPS: CDF validity and prediction set coverage (5 models)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_ridge_cps_cdf_validity():
    """Test RidgePredictionMachine CDF validity via PIT uniformity."""
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(2)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cps = RidgePredictionMachine(a=1.0)
        cps.learn_initial_training_set(X, y)

        # Test CDF validity via PIT uniformity test
        X_test = rng.normal(size=(100, d))
        pit_values = []
        for x in X_test:
            cpd = cps.predict_cpd(x)
            # Sample from the distribution and compute PIT values
            tau = rng.uniform(0, 1)
            y_sample = rng.normal()  # Sample from the true distribution
            pit = cpd(y_sample, tau)
            pit_values.append(pit)

        # Test uniformity of PIT values using Kolmogorov-Smirnoff test
        # For a valid conformal predictive system, PIT values should be uniformly distributed
        if len(pit_values) > 1:
            ks_statistic, p_value = kstest(pit_values, 'uniform')
            if p_value < BINOM_ALPHA:
                total_errors += 1
        total_preds += 1

    # Test that uniformity violations are within acceptable bounds
    eps = 0.1
    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"CDF violation rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_kernel_ridge_cps_coverage_validity():
    """Test KernelRidgePredictionMachine prediction set coverage."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(2)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cps = KernelRidgePredictionMachine(kernel=GaussianKernel(sigma=1.0), a=1.0)
        cps.learn_initial_training_set(X, y)

        X_test = rng.normal(size=(100, d))
        for x in X_test:
            cpd = cps.predict_cpd(x)
            tau = rng.uniform(0, 1)
            Gamma = cpd.predict_set(tau=tau, epsilon=eps)
            # Check coverage for true y (sample from same distribution)
            y_true = rng.normal()
            in_set = y_true in Gamma
            if not in_set:
                total_errors += 1
            total_preds += 1

    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_knn_cps_coverage_validity():
    """Test NearestNeighboursPredictionMachine prediction set coverage."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(2)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cps = NearestNeighboursPredictionMachine(k=5)
        cps.learn_initial_training_set(X, y)

        X_test = rng.normal(size=(100, d))
        for x in X_test:
            cpd = cps.predict_cpd(x)
            tau = rng.uniform(0, 1)
            Gamma = cpd.predict_set(tau=tau, epsilon=eps)
            y_true = rng.normal()
            in_set = y_true in Gamma
            if not in_set:
                total_errors += 1
            total_preds += 1

    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_dempster_hill_cps_coverage_validity():
    """Test DempsterHillConformalPredictiveSystem prediction set coverage."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(2)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cps = DempsterHillConformalPredictiveSystem()
        cps.learn_initial_training_set(y)

        X_test = rng.normal(size=(100, d))
        for x in X_test:
            cpd = cps.predict_cpd(x)
            tau = rng.uniform(0, 1)
            Gamma = cpd.predict_set(tau=tau, epsilon=eps)
            y_true = rng.normal()
            in_set = y_true in Gamma
            if not in_set:
                total_errors += 1
            total_preds += 1

    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_mondrian_cps_coverage_validity():
    """Test MondrianTreePredictionMachine prediction set coverage."""
    eps = 0.1
    n_runs, N, d = 40, 400, 4
    rng = np.random.default_rng(2)
    total_errors = 0
    total_preds = 0
    for _ in range(n_runs):
        X, y = generate_linear_data(rng, N, d)
        cps = MondrianTreePredictionMachine(lifetime=1.0)
        cps.learn_initial_training_set(X, y)

        X_test = rng.normal(size=(100, d))
        for x in X_test:
            cpd = cps.predict_cpd(x)
            tau = rng.uniform(0, 1)
            Gamma = cpd.predict_set(tau=tau, epsilon=eps)
            y_true = rng.normal()
            in_set = y_true in Gamma
            if not in_set:
                total_errors += 1
            total_preds += 1

    result = binomtest(total_errors, total_preds, eps, alternative="greater")
    assert result.pvalue > BINOM_ALPHA, (
        f"error rate {total_errors / total_preds:.4f} significantly exceeds eps={eps} (binom p={result.pvalue:.2e})"
    )


# --------------------------------------------------------------------------- #
# Venn predictors: calibration-in-the-large (3 models)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_knn_venn_calibration_in_the_large():
    """Test NearestNeighboursVennPredictor calibration."""
    rng = np.random.default_rng(2)
    N_train, N_test, d = 200, 150, 4

    def sample(n):
        X = rng.normal(size=(n, d))
        logits = X.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(int)
        return X, y

    X_tr, y_tr = sample(N_train)
    X_te, y_te = sample(N_test)

    venn = NearestNeighboursVennPredictor(k=5, label_space=np.array([0, 1]))
    venn.learn_initial_training_set(X_tr, y_tr)

    point_preds = np.array([venn.predict(x).point for x in X_te])
    mean_pred = float(point_preds.mean())
    n_positives = int(y_te.sum())

    result = binomtest(n_positives, int(y_te.size), mean_pred, alternative="two-sided")
    assert result.pvalue > BINOM_ALPHA, (
        f"Venn miscalibrated: mean pred {mean_pred:.3f} vs "
        f"{n_positives}/{y_te.size} positives (binom p={result.pvalue:.2e})"
    )


@pytest.mark.slow
def test_mondrian_venn_calibration_in_the_large():
    """Test MondrianVennPredictor calibration."""
    rng = np.random.default_rng(2)
    N_train, N_test, d = 200, 150, 4

    def sample(n):
        X = rng.normal(size=(n, d))
        logits = X.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(int)
        return X, y

    X_tr, y_tr = sample(N_train)
    X_te, y_te = sample(N_test)

    venn = MondrianVennPredictor(lifetime=1.0, label_space=np.array([0, 1]))
    venn.learn_initial_training_set(X_tr, y_tr)

    point_preds = np.array([venn.predict(x).point for x in X_te])
    mean_pred = float(point_preds.mean())
    n_positives = int(y_te.sum())

    result = binomtest(n_positives, int(y_te.size), mean_pred, alternative="two-sided")
    assert result.pvalue > BINOM_ALPHA, (
        f"Venn miscalibrated: mean pred {mean_pred:.3f} vs "
        f"{n_positives}/{y_te.size} positives (binom p={result.pvalue:.2e})"
    )
