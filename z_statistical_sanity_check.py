#!/usr/bin/env python3
"""
Statistical sanity check for all conformal prediction methods.

This script runs each method online and checks that:
  1. Empirical coverage >= 1 - epsilon (via binomial confidence interval)
  2. p-values are approximately uniform (Kolmogorov-Smirnov test)

Not a formal test -- just a quick check that the guarantees hold.

Usage:
    python3 z_statistical_sanity_check.py
    python3 z_statistical_sanity_check.py --verbose
    python3 z_statistical_sanity_check.py --seed 42
"""
import argparse
import numpy as np
from scipy import stats

from online_cp import (
    ConformalRidgeRegressor,
    KernelConformalRidgeRegressor,
    ConformalLassoRegressor,
    ConformalNearestNeighboursClassifier,
    ConformalSupportVectorMachine,
    RidgePredictionMachine,
    KernelRidgePredictionMachine,
    NearestNeighboursPredictionMachine,
    GaussianKernel,
)


# ============================================================================
# Helpers
# ============================================================================

def binomial_ci(k, n, alpha=0.05):
    """Wilson score interval for coverage proportion."""
    p_hat = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return centre - half_width, centre + half_width


def check_coverage(name, errors, n_total, epsilon, verbose=False):
    """Check if empirical coverage is consistent with 1 - epsilon."""
    n_errors = sum(errors)
    coverage = 1 - n_errors / n_total
    ci_lo, ci_hi = binomial_ci(n_total - n_errors, n_total)
    target = 1 - epsilon
    # Pass if target is within CI, or coverage exceeds target (conservative is fine)
    passed = ci_lo <= target <= ci_hi or coverage >= target - 0.01

    status = "\033[32mPASS\033[0m" if passed else "\033[33mWARN\033[0m"
    print(f"  [{status}] {name}: coverage={coverage:.4f} (target>={target:.2f}, "
          f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], n={n_total})")
    return passed


def check_p_values(name, p_values, verbose=False):
    """Check if p-values are approximately U[0,1] via KS test."""
    p_arr = np.array(p_values)
    ks_stat, ks_pval = stats.kstest(p_arr, 'uniform')
    passed = ks_pval > 0.01  # reject uniformity only at 1% level

    status = "\033[32mPASS\033[0m" if passed else "\033[33mWARN\033[0m"
    print(f"  [{status}] {name} p-values: KS={ks_stat:.4f}, p={ks_pval:.4f} "
          f"(reject uniformity if p<0.01)")
    return passed


# ============================================================================
# Data generators
# ============================================================================

def generate_regression_data(rng, n, p=5, noise=0.5):
    """Linear regression data with sparse true signal."""
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[:3] = [2.0, -1.5, 1.0]
    y = X @ beta_true + rng.normal(scale=noise, size=n)
    return X, y


def generate_classification_data(rng, n, p=5, n_classes=3):
    """Multi-class classification data (shifted clusters)."""
    X = np.empty((n, p))
    y = np.empty(n, dtype=int)
    per_class = n // n_classes
    for c in range(n_classes):
        start = c * per_class
        end = start + per_class if c < n_classes - 1 else n
        count = end - start
        X[start:end] = rng.normal(loc=c * 2, scale=1.0, size=(count, p))
        y[start:end] = c
    perm = rng.permutation(n)
    return X[perm], y[perm]


# ============================================================================
# Method tests
# ============================================================================

def test_conformal_ridge(rng, epsilon=0.1, n_train=50, n_test=200, verbose=False):
    """Test ConformalRidgeRegressor."""
    X, y = generate_regression_data(rng, n_train + n_test)

    cp = ConformalRidgeRegressor(a=1.0, epsilon=epsilon)
    cp.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        interval = cp.predict(X[i], epsilon=epsilon)
        errors.append(y[i] not in interval)
        p = cp.compute_p_value(X[i], y[i])
        p_values.append(p)
        cp.learn_one(X[i], y[i])

    ok1 = check_coverage("ConformalRidgeRegressor", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("ConformalRidgeRegressor", p_values, verbose)
    return ok1 and ok2


def test_kernel_conformal_ridge(rng, epsilon=0.1, n_train=50, n_test=150, verbose=False):
    """Test KernelConformalRidgeRegressor."""
    X, y = generate_regression_data(rng, n_train + n_test, p=3, noise=0.3)

    kernel = GaussianKernel(sigma=1.0)
    cp = KernelConformalRidgeRegressor(kernel=kernel, a=1.0, epsilon=epsilon)
    cp.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        interval = cp.predict(X[i], epsilon=epsilon)
        errors.append(y[i] not in interval)
        p = cp.compute_p_value(X[i], y[i])
        p_values.append(p)
        cp.learn_one(X[i], y[i])

    ok1 = check_coverage("KernelConformalRidgeRegressor", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("KernelConformalRidgeRegressor", p_values, verbose)
    return ok1 and ok2


def test_conformal_lasso(rng, epsilon=0.1, n_train=40, n_test=150, verbose=False):
    """Test ConformalLassoRegressor (pure Lasso)."""
    X, y = generate_regression_data(rng, n_train + n_test, p=10, noise=0.5)

    cp = ConformalLassoRegressor(lam=0.3, epsilon=epsilon, rnd_state=int(rng.integers(1e6)))
    cp.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        interval = cp.predict(X[i], epsilon=epsilon)
        errors.append(y[i] not in interval)
        p = cp.compute_p_value(X[i], y[i])
        p_values.append(p)
        cp.learn_one(X[i], y[i])

    ok1 = check_coverage("ConformalLassoRegressor", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("ConformalLassoRegressor", p_values, verbose)
    return ok1 and ok2


def test_conformal_elastic_net(rng, epsilon=0.1, n_train=40, n_test=150, verbose=False):
    """Test ConformalLassoRegressor with elastic net (rho > 0)."""
    X, y = generate_regression_data(rng, n_train + n_test, p=10, noise=0.5)

    cp = ConformalLassoRegressor(lam=0.3, rho=1.0, epsilon=epsilon,
                                 rnd_state=int(rng.integers(1e6)))
    cp.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        interval = cp.predict(X[i], epsilon=epsilon)
        errors.append(y[i] not in interval)
        p = cp.compute_p_value(X[i], y[i])
        p_values.append(p)
        cp.learn_one(X[i], y[i])

    ok1 = check_coverage("ConformalLassoRegressor (elastic net)", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("ConformalLassoRegressor (elastic net)", p_values, verbose)
    return ok1 and ok2


def test_conformal_knn_classifier(rng, epsilon=0.1, n_train=60, n_test=200, verbose=False):
    """Test ConformalNearestNeighboursClassifier."""
    n_classes = 3
    label_space = np.arange(n_classes)
    X, y = generate_classification_data(rng, n_train + n_test, p=5, n_classes=n_classes)

    cp = ConformalNearestNeighboursClassifier(
        k=5, label_space=label_space, epsilon=epsilon,
        rnd_state=int(rng.integers(1e6))
    )
    cp.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    for i in range(n_train, n_train + n_test):
        Gamma = cp.predict(X[i], epsilon=epsilon)
        errors.append(y[i] not in Gamma)
        cp.learn_one(X[i], y[i])

    ok = check_coverage("ConformalNearestNeighboursClassifier", errors, n_test, epsilon, verbose)
    return ok


def test_conformal_svm(rng, epsilon=0.1, n_train=60, n_test=150, verbose=False):
    """Test ConformalSupportVectorMachine."""
    n_classes = 3
    label_space = np.arange(n_classes)
    X, y = generate_classification_data(rng, n_train + n_test, p=5, n_classes=n_classes)

    cp = ConformalSupportVectorMachine(
        kernel='rbf', sigma=1.0, C=10.0,
        label_space=label_space, epsilon=epsilon,
        rnd_state=int(rng.integers(1e6))
    )
    cp.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    for i in range(n_train, n_train + n_test):
        Gamma = cp.predict(X[i], epsilon=epsilon)
        errors.append(y[i] not in Gamma)
        cp.learn_one(X[i], y[i])

    ok = check_coverage("ConformalSupportVectorMachine", errors, n_test, epsilon, verbose)
    return ok


def test_ridge_cps(rng, epsilon=0.1, n_train=50, n_test=200, verbose=False):
    """Test RidgePredictionMachine (conformal predictive system)."""
    X, y = generate_regression_data(rng, n_train + n_test)

    cps = RidgePredictionMachine(a=1.0)
    cps.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        tau = rng.uniform()
        cpd = cps.predict_cpd(X[i])
        interval = cpd.predict_set(tau, epsilon=epsilon, bounds='both')
        errors.append(y[i] not in interval)
        # Smoothed p-value from CPD
        p = cpd(y[i], tau)
        p_values.append(p)
        cps.learn_one(X[i], y[i])

    ok1 = check_coverage("RidgePredictionMachine (CPS)", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("RidgePredictionMachine (CPS)", p_values, verbose)
    return ok1 and ok2


def test_kernel_ridge_cps(rng, epsilon=0.1, n_train=50, n_test=100, verbose=False):
    """Test KernelRidgePredictionMachine (conformal predictive system)."""
    X, y = generate_regression_data(rng, n_train + n_test, p=3, noise=0.3)

    kernel = GaussianKernel(sigma=1.0)
    cps = KernelRidgePredictionMachine(kernel=kernel, a=1.0)
    cps.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        tau = rng.uniform()
        cpd = cps.predict_cpd(X[i])
        interval = cpd.predict_set(tau, epsilon=epsilon, bounds='both')
        errors.append(y[i] not in interval)
        p = cpd(y[i], tau)
        p_values.append(p)
        cps.learn_one(X[i], y[i])

    ok1 = check_coverage("KernelRidgePredictionMachine (CPS)", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("KernelRidgePredictionMachine (CPS)", p_values, verbose)
    return ok1 and ok2


def test_knn_cps(rng, epsilon=0.1, n_train=50, n_test=150, verbose=False):
    """Test NearestNeighboursPredictionMachine (conformal predictive system)."""
    X, y = generate_regression_data(rng, n_train + n_test, p=3, noise=0.5)

    cps = NearestNeighboursPredictionMachine(k=5)
    cps.learn_initial_training_set(X[:n_train], y[:n_train])

    errors = []
    p_values = []
    for i in range(n_train, n_train + n_test):
        tau = rng.uniform()
        cpd = cps.predict_cpd(X[i])
        interval = cpd.predict_set(tau, epsilon=epsilon, bounds='both')
        errors.append(y[i] not in interval)
        p = cpd(y[i], tau)
        p_values.append(p)
        cps.learn_one(X[i], y[i])

    ok1 = check_coverage("NearestNeighboursPredictionMachine (CPS)", errors, n_test, epsilon, verbose)
    ok2 = check_p_values("NearestNeighboursPredictionMachine (CPS)", p_values, verbose)
    return ok1 and ok2


# ============================================================================
# Multiple epsilon levels
# ============================================================================

def test_multiple_epsilons(rng, n_train=50, n_test=300, verbose=False):
    """Test ConformalRidgeRegressor at multiple significance levels."""
    print("\n--- Multiple epsilon levels (ConformalRidgeRegressor) ---")
    X, y = generate_regression_data(rng, n_train + n_test)

    all_ok = True
    for epsilon in [0.01, 0.05, 0.1, 0.2, 0.3]:
        cp = ConformalRidgeRegressor(a=1.0, epsilon=epsilon)
        cp.learn_initial_training_set(X[:n_train], y[:n_train])

        errors = []
        for i in range(n_train, n_train + n_test):
            interval = cp.predict(X[i], epsilon=epsilon)
            errors.append(y[i] not in interval)
            cp.learn_one(X[i], y[i])

        ok = check_coverage(f"epsilon={epsilon:.2f}", errors, n_test, epsilon, verbose)
        all_ok = all_ok and ok

    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical sanity check for conformal methods")
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--n-train', type=int, default=None,
                        help='Initial training set size (overrides per-test defaults)')
    parser.add_argument('--n-test', type=int, default=None,
                        help='Number of online test points (overrides per-test defaults)')
    parser.add_argument('--verbose', action='store_true', help='Print extra details')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []

    # Build size kwargs (only override if user specified)
    size_kw = {}
    if args.n_train is not None:
        size_kw['n_train'] = args.n_train
    if args.n_test is not None:
        size_kw['n_test'] = args.n_test

    print("=" * 70)
    print("STATISTICAL SANITY CHECK -- Online Conformal Prediction")
    print(f"Seed: {args.seed}")
    if size_kw:
        print(f"Size overrides: {size_kw}")
    print("=" * 70)

    # --- Regressors ---
    print("\n--- Conformal Regressors (coverage + p-value uniformity) ---")
    results.append(test_conformal_ridge(rng, verbose=args.verbose, **size_kw))
    results.append(test_kernel_conformal_ridge(rng, verbose=args.verbose, **size_kw))
    results.append(test_conformal_lasso(rng, verbose=args.verbose, **size_kw))
    results.append(test_conformal_elastic_net(rng, verbose=args.verbose, **size_kw))

    # --- Classifiers ---
    print("\n--- Conformal Classifiers (coverage) ---")
    results.append(test_conformal_knn_classifier(rng, verbose=args.verbose, **size_kw))
    results.append(test_conformal_svm(rng, verbose=args.verbose, **size_kw))

    # --- CPS ---
    print("\n--- Conformal Predictive Systems (coverage + p-value uniformity) ---")
    results.append(test_ridge_cps(rng, verbose=args.verbose, **size_kw))
    results.append(test_kernel_ridge_cps(rng, verbose=args.verbose, **size_kw))
    results.append(test_knn_cps(rng, verbose=args.verbose, **size_kw))

    # --- Multi-epsilon ---
    results.append(test_multiple_epsilons(rng, verbose=args.verbose, **size_kw))

    # --- Summary ---
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 70)
    if n_pass == n_total:
        print(f"\033[32mALL CHECKS PASSED ({n_pass}/{n_total})\033[0m")
    else:
        print(f"\033[33mWARNINGS: {n_total - n_pass}/{n_total} checks had issues\033[0m")
        print("(This may be due to randomness -- try a different seed)")
    print("=" * 70)


if __name__ == '__main__':
    main()
