#!/usr/bin/env python3
"""
Reproduce Table 1 (Simulation 1-I) from the conformalized Lasso paper
(Lei & Fithian) to verify the implementation.

Paper setup — Model I (Standard Gaussian linear model, low-dimensional):
  - n = 100 training points, 100 test points
  - p = 10 features
  - β ∈ {-1, +1}^p  (random signs)
  - X ~ N(0, I_p)
  - ε ~ N(0, 1)
  - Target coverage: 90%  (α = 0.1)
  - λ chosen by cross-validation
  - Search range: [y_min - 0.25*range, y_max + 0.25*range]

Paper results (exact method, Table 1):
  Coverage:  0.905  (SE 0.004)
  Length:    3.51   (SE 0.03)

We also compare the homotopy prediction set against brute-force grid evaluation
on a few examples to verify the homotopy logic itself.

Usage:
    python3 z_reproduce_lasso_paper.py
    python3 z_reproduce_lasso_paper.py --n-reps 100 --seed 42
    python3 z_reproduce_lasso_paper.py --quick   # fewer reps for a fast check
"""
import argparse
import time

import numpy as np
from scipy import stats

from online_cp import ConformalLassoRegressor
from online_cp.regressors import _solve_lasso


# ============================================================================
# Data generation (Model I from the paper)
# ============================================================================

def generate_model_I(rng, n, p=10):
    """Standard Gaussian linear model (Model I from the paper).

    Y = X'β + ε,  β ∈ {-1,+1}^p,  X ~ N(0,I_p),  ε ~ N(0,1).
    """
    beta = rng.choice([-1.0, 1.0], size=p)
    X = rng.normal(size=(n, p))
    y = X @ beta + rng.normal(size=n)
    return X, y, beta


# ============================================================================
# Pick lambda by cross-validation (as in the paper)
# ============================================================================

def cv_lambda(X, y, rng, n_folds=5, n_grid=50):
    """Choose λ by K-fold CV, returning the λ that minimises mean squared error."""
    n, p = X.shape
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lam_max = np.max(np.abs(X.T @ y)) / n
    lam_grid = np.geomspace(lam_max, lam_max * 1e-3, num=n_grid)

    best_lam, best_mse = lam_grid[0], np.inf
    for lam in lam_grid:
        mse = 0.0
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            beta_k = _solve_lasso(X[train_idx], y[train_idx], lam)
            mse += np.mean((y[val_idx] - X[val_idx] @ beta_k) ** 2)
        mse /= n_folds
        if mse < best_mse:
            best_mse = mse
            best_lam = lam
    return best_lam


# ============================================================================
# Brute-force conformal prediction (for validation)
# ============================================================================

def brute_force_conformal_lasso(X_train, y_train, x_test, lam, epsilon=0.1,
                                 n_grid=500, margin=0.25):
    """Compute conformal prediction set by grid evaluation (brute force).

    For each candidate y on a grid, augment the data with (x_test, y),
    refit Lasso, compute absolute residuals, and check if |r_{n+1}| has
    low enough rank to be included in the prediction set.
    """
    n = len(y_train)
    threshold = int(np.ceil((n + 1) * (1 - epsilon)))

    # Determine grid range
    y_min, y_max = y_train.min(), y_train.max()
    y_range = y_max - y_min
    grid_lo = y_min - margin * y_range
    grid_hi = y_max + margin * y_range
    grid = np.linspace(grid_lo, grid_hi, n_grid)

    # Warm-start: fit on training data
    beta0 = _solve_lasso(X_train, y_train, lam)
    X_aug = np.vstack([X_train, x_test.reshape(1, -1)])

    in_set = []
    for y_trial in grid:
        y_aug = np.append(y_train, y_trial)
        beta_aug = _solve_lasso(X_aug, y_aug, lam, warm_start=beta0)
        abs_res = np.abs(y_aug - X_aug @ beta_aug)
        rank = np.sum(abs_res <= abs_res[-1])
        if rank <= threshold:
            in_set.append(y_trial)

    if not in_set:
        return np.nan, np.nan
    return min(in_set), max(in_set)


# ============================================================================
# Main experiment
# ============================================================================

def run_one_replication(rng, n_train=100, n_test=100, p=10, epsilon=0.1):
    """Run one replication of Model I and return coverage and average length."""
    X, y, beta_true = generate_model_I(rng, n_train + n_test, p=p)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Pick lambda by CV
    lam = cv_lambda(X_train, y_train, rng)

    # Build conformal regressor
    cp = ConformalLassoRegressor(
        lam=lam,
        epsilon=epsilon,
        search_range_factor=0.25,
        rnd_state=int(rng.integers(1e9)),
    )
    cp.learn_initial_training_set(X_train, y_train)

    covered = 0
    total_length = 0.0
    for i in range(n_test):
        interval = cp.predict(X_test[i], epsilon=epsilon)
        if y_test[i] in interval:
            covered += 1
        total_length += interval.width()

    coverage = covered / n_test
    avg_length = total_length / n_test
    return coverage, avg_length, lam


def run_homotopy_vs_brute_force(rng, n_checks=10, verbose=True):
    """Compare homotopy and brute-force prediction sets on small problems."""
    print("\n" + "=" * 70)
    print("HOMOTOPY vs BRUTE-FORCE COMPARISON")
    print("=" * 70)

    n_train, p = 30, 10
    epsilon = 0.1
    n_agree = 0
    n_total = 0

    X, y, _ = generate_model_I(rng, n_train + n_checks, p=p)
    X_train, y_train = X[:n_train], y[:n_train]
    lam = cv_lambda(X_train, y_train, rng)

    cp = ConformalLassoRegressor(
        lam=lam,
        epsilon=epsilon,
        search_range_factor=0.25,
        rnd_state=int(rng.integers(1e9)),
    )
    cp.learn_initial_training_set(X_train, y_train)

    for i in range(n_checks):
        x_test = X[n_train + i]
        y_test = y[n_train + i]

        # Homotopy
        interval_hom = cp.predict(x_test, epsilon=epsilon)

        # Brute force (dense grid)
        lo_bf, hi_bf = brute_force_conformal_lasso(
            X_train, y_train, x_test, lam, epsilon=epsilon, n_grid=1000
        )

        # Compare: does the true label get the same inclusion decision?
        in_hom = y_test in interval_hom
        in_bf = (lo_bf <= y_test <= hi_bf) if not np.isnan(lo_bf) else False

        agree = in_hom == in_bf
        n_agree += agree
        n_total += 1

        if verbose:
            status = "\033[32mAGREE\033[0m" if agree else "\033[31mDISAGREE\033[0m"
            print(
                f"  [{status}] Test {i+1}: y={y_test:.3f}  "
                f"homotopy=[{interval_hom.lower:.3f}, {interval_hom.upper:.3f}]  "
                f"brute-force=[{lo_bf:.3f}, {hi_bf:.3f}]  "
                f"width_hom={interval_hom.width():.3f}  width_bf={hi_bf - lo_bf:.3f}"
            )

    pct = 100 * n_agree / n_total
    print(f"\nAgreement: {n_agree}/{n_total} ({pct:.0f}%)")
    return n_agree == n_total


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1 (Model I) from the conformal Lasso paper"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-reps", type=int, default=20,
                        help="Number of replications (paper uses 100)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer reps and test points")
    parser.add_argument("--skip-brute-force", action="store_true",
                        help="Skip the slow brute-force comparison")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.quick:
        n_reps = 5
        n_test = 20
    else:
        n_reps = args.n_reps
        n_test = 100

    print("=" * 70)
    print("REPRODUCING TABLE 1 (Model I) — Conformal Lasso Paper")
    print(f"  n_train=100, n_test={n_test}, p=10, ε=0.1, replications={n_reps}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)
    print()
    print("Paper reference (exact method):")
    print("  Coverage: 0.905 (SE 0.004)")
    print("  Length:   3.51  (SE 0.03)")
    print()

    coverages = []
    lengths = []
    lambdas = []

    for rep in range(n_reps):
        t0 = time.time()
        cov, avg_len, lam = run_one_replication(
            rng, n_train=100, n_test=n_test, p=10, epsilon=0.1
        )
        elapsed = time.time() - t0
        coverages.append(cov)
        lengths.append(avg_len)
        lambdas.append(lam)
        print(
            f"  Rep {rep+1:3d}/{n_reps}: coverage={cov:.3f}, "
            f"length={avg_len:.2f}, λ={lam:.4f}, time={elapsed:.1f}s"
        )

    coverages = np.array(coverages)
    lengths = np.array(lengths)

    print()
    print("-" * 70)
    print("RESULTS (our implementation):")
    print(f"  Coverage: {coverages.mean():.3f} (SE {coverages.std() / np.sqrt(n_reps):.3f})")
    print(f"  Length:   {lengths.mean():.2f}  (SE {lengths.std() / np.sqrt(n_reps):.2f})")
    print(f"  Lambda:   {np.median(lambdas):.4f} (median)")
    print()
    print("Paper reference (exact method):")
    print("  Coverage: 0.905 (SE 0.004)")
    print("  Length:   3.51  (SE 0.03)")
    print("-" * 70)

    # Basic checks
    target_coverage = 0.9
    ci_lo, ci_hi = (
        coverages.mean() - 2 * coverages.std() / np.sqrt(n_reps),
        coverages.mean() + 2 * coverages.std() / np.sqrt(n_reps),
    )
    cov_ok = ci_lo <= 0.92 and ci_hi >= 0.87  # generous check
    len_ok = 1.5 < lengths.mean() < 7.0  # paper gets ~3.51

    if cov_ok:
        print("\033[32mCoverage looks consistent with the paper.\033[0m")
    else:
        print(f"\033[31mCoverage may differ from paper: {coverages.mean():.3f} "
              f"(95% CI [{ci_lo:.3f}, {ci_hi:.3f}])\033[0m")

    if len_ok:
        print(f"\033[32mInterval length ({lengths.mean():.2f}) is in a reasonable range.\033[0m")
    else:
        print(f"\033[31mInterval length ({lengths.mean():.2f}) seems off "
              f"(paper reports ~3.51).\033[0m")

    # Homotopy vs brute-force
    if not args.skip_brute_force:
        bf_ok = run_homotopy_vs_brute_force(rng, n_checks=10, verbose=True)
    else:
        bf_ok = True

    print()
    print("=" * 70)
    all_ok = cov_ok and len_ok and bf_ok
    if all_ok:
        print("\033[32mALL CHECKS PASSED\033[0m")
    else:
        print("\033[31mSOME CHECKS FAILED — investigate the implementation\033[0m")
    print("=" * 70)


if __name__ == "__main__":
    main()
