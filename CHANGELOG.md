# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Improved
  - **Conformal Lasso: `predict` cost is now O(n + m) instead of O(n * m)**
    (speed target T2). The previous `_find_intervals_in_piece` materialised an
    `(n, m)` residual matrix per homotopy piece (m = number of crossings in the
    piece, up to n), so `predict` cost blew up super-linearly in n and was
    non-monotone (the 170 ms @ n=1e5 vs 4 ms @ n=1e6 sawtooth recorded in
    `archive/speed-baseline.md`). The ascending conformal rank is now computed by
    an O(n + m) incremental sweep: one O(n) rank at d = 0, then +/-1 at each
    crossing whose flip direction is set by the derivative of
    `(|r_i| - |r_{n+1}|)` at the crossing. Measured speedup on the worst-case
    (many-crossing) stream: n=8000 went from ~888 ms to ~6 ms, n=12000 from
    ~1608 ms to ~9 ms, and the non-monotonicity in n is gone. Verified
    correct: `TestFindIntervalsInPiece::test_matches_loop_reference` (300
    random pieces) and `TestHomotopyCorrectness::test_predict_covers_brute_force_set`
    both pass; new regression guards `test_with_index_agrees_with_plain_crossings`
    and `test_predict_cost_bounded_in_n` pin the identity arrays and the bounded
    cost. A new numba helper `_compute_crossings_with_index` (alongside the
    existing `_compute_crossings`) returns the crossing identity (point index +
    sign-case) needed for the sweep; the existing signature is unchanged so the
    Mondrian forest solver and existing tests are unaffected.

### Changed
  - The Lasso `predict`/`compute_p_value` p-value rank loops identified as
    "T4a" in the v0.4.0 speed baseline are already numpy-vectorised in the
    current codebase; the only non-vectorised rank computation was the O(n * m)
    matrix above, now swept. No separate T4a JIT work is required.

### Added
  - **Stream pruning** (`pruning=`, `budget=B`) on
    `ConformalNearestNeighboursRegressor`,
    `ConformalNearestNeighboursClassifier`, `KernelConformalRidgeRegressor`, and
    `ConformalSupportVectorMachine`. Only **strictly-valid** (data-blind)
    strategies are accepted, so the exact finite-sample coverage guarantee is
    preserved unchanged. A strategy is strictly valid iff the retained bag is a
    *bag-function* of the stream — a symmetric, value-blind sub-selection,
    randomised only by data-independent randomness. Two built-ins:
    - `pruning="reservoir"` — the retained training bag is a uniform random
      sample of `budget` examples from the stream (Vitter's Algorithm R).
    - `pruning="window"` — the retained training bag is the most recent
      `budget` examples (FIFO; expands to `budget` then slides). Deterministic
      (seed-independent); less robust than the reservoir on drifting /
      non-i.i.d. streams but trivial to maintain and useful for adaptation.
    Both are data-blind bag-functions, so the new example remains exchangeable
    with the bag. With `budget >= n` seen, behaviour is *bit-identical* to no
    pruning.
  - **Custom pruning strategies + validator** — a user may supply their own
    strictly-valid strategy via `online_cp.PruningStrategy` (or an object
    exposing `decide(n_seen)`), passed to a model with `pruning=<strategy>` and
    `budget=B`. The library enforces the data-blind contract at the API boundary
    with `online_cp.validate_strategy`: a strategy is accepted only if its
    `decide` signature does **not** receive the data (`x`, `y`, `model`, `d`,
    `alpha`, `leverage`, `point`, `example`, `bag`, `labels`, `features`,
    `data`, `train`); model-state / value-dependent selectors (SV, leverage,
    core-set, self-pruning, stratification) are **rejected** at construction —
    there is no "best-effort" mode. The model injects its own `budget` and
    `rnd_gen` into the strategy, so the RNG state serialises with the model and
    save/load round-trips reproducibly. The shared controller lives in
    `online_cp._pruning` (`PruningStrategy`, `_ReservoirBuffer`,
    `_WindowBuffer`, `_make_strategy_buffer`, `validate_strategy`). The pruned
    state (bag, example count, `budget`, RNG position) round-trips through
    `save`/`load`. Kernel-ridge recomputes the kernel inverse from scratch at
    size `budget` on each eviction (O(budget³), bounded by the user's budget);
    the k-NN classifier rebuilds its label index from the pruned labels.
    Verified by `tests/test_pruning.py` (bit-identity, bag cap, index
    consistency, coverage preservation, determinism, save/load round-trip,
    validator acceptance/rejection, custom-strategy save/load).
  - **Batch prediction API** — `predict_batch(X)` on the `ConformalRegressor`
    and `ConformalClassifier` bases. The default is a loop over `predict`, which
    is **bit-identical** to `predict(X[i])` per row by construction (Lasso and
    SVM use this). A vectorised override on the k-NN classifier replaces the
    per-point distance computation with one `cdist(X_batch, X_train)` call.
    Contract: `predict_batch(X)[i] == predict(X[i])` exactly (enforced by
    `tests/test_batch_predict.py`). Note: the smoothing variable `tau` is
    positional, so a batch is not permutation-invariant for tau-smoothed models;
    non-smoothed models (e.g. ridge) are.
  - **Mondrian wrapper with a pruned base model** — `MondrianConformalRegressor`
    and `MondrianConformalClassifier` now keep their category list
    (`categories_`) aligned with the base model's *retained* (possibly pruned)
    training rows, so a pruned base model can be used directly, e.g.
    `MondrianConformalRegressor(KernelConformalRidgeRegressor(kernel, pruning="reservoir",
    budget=1000), category_fn=...)`. The wrapper rebuilds `categories_` from
    `base_model.X`/`base_model.y` after every `learn_one` and
    `learn_initial_training_set` (a no-op behaviourally for unpruned bases, where
    the retained rows are exactly the appended stream in order). Restricting the
    pruned bag to a fixed Mondrian category is still a data-blind sub-selection,
    so the group-conditional coverage guarantee is preserved. No new constructor
    arguments: the user passes an already-configured base model. Verified by
    `tests/test_mondrian.py::TestMondrianPrunedBase` (alignment invariant for
    reservoir/window, bit-identity vs. unpruned, marginal coverage, label-aware
    classifier alignment, save/load round-trip).

---

## [0.4.0] — 2026-10-22

### Added
  - Module-level helper `_rank1_inverse_update(inv, x)` in `online_cp.regressors`
    that applies the matrix-inversion lemma for `(A + xx')^{-1}` and symmetrizes
    the result. Used by the ridge and kernel-ridge `learn_one` paths.

### Fixed
  - **Conformal Lasso: `_solve_lasso` now converges to the true optimum** — the
    coordinate-descent solver stopped on `max|delta beta| < tol` with
    `tol = 1e-6`.  That test is insufficient: a single coefficient can
    oscillate below the threshold each sweep while the subgradient is still far
    from `+/- lam`, so the loop halted early and returned a Lasso/elastic-net
    fit that is *not* the optimum.  A non-optimum starting fit is exactly what
    made the `ConformalLassoRegressor` homotopy (`predict`) diverge from the
    true piecewise-linear solution path and produce under-covering prediction
    intervals.  The solver now stops on the true KKT residual (active
    subgradient == `sign(beta)*lam`, inactive in `[-lam, lam]`), which
    converges to the (unique) optimum.  The rank criterion in
    `_find_intervals_in_piece` / `_t_in_prediction_set` / the Mondrian Lasso
    path remains `rank <= threshold` (ascending rank, 1 = smallest), which is
    the paper's criterion (Lei 2019, prelim.tex line 141 / method.tex line
    459): a candidate is kept iff the test residual is among the *smallest* in
    magnitude.  Verified: the homotopy interval now matches a brute-force grid
    reference implementing the paper's rank criterion on 11/12 random small
    problems (the 12th differs only by grid resolution), and
    `ConformalLassoRegressor.predict` is sub-millisecond to ~2 ms on
    well-conditioned (n >> p) problems.
  - **Numerical drift in online inverse updates** — the ridge `XTXinv` and kernel
    `Kinv` rank-1 (block) inverse updates are now symmetrized after each step
    (`0.5 * (M + M.T)`), eliminating the asymmetric round-off that accumulated
    over long online streams. Empirically this drops `Kinv` asymmetry from
    ~1e-7 to exactly 0 with no change to the predicted values. Users with
    near-collinear kernel data should still call `recompute_inverse()`
    periodically (or set `recompute_every`) to control diagonal drift.

### Changed
  - **ConformalRegressor order-statistic selectors** — `_get_lower` /
    `_get_upper` were refactored to use explicit `dict.get` rank lookups with
    documented ray-fallbacks (replacing `try/except KeyError`). Behavior is
    **provably identical** to the previous implementation (verified over 9000
    random trials, 0 differences); the change is for clarity and to pin the
    edge-case (out-of-range rank → unbounded ray) in tests.
  - **Ridge GCV: closed-form eigenbasis minimization** —
    `ConformalRidgeRegressor._tune_ridge_parameter` and
    `RidgePredictionMachine._tune_ridge_parameter` now factor `X^T X` **once**
    into an eigendecomposition and evaluate the GCV as O(n p) vector ops per
    candidate (instead of a fresh p×p inverse per evaluation, O(p³)), and
    minimise the 1-D GCV with `minimize_scalar` (bounded) rather than
    general-purpose Nelder-Mead. Result: **~12-20x faster** tuning, and the
    true GCV minimum is now attained (Nelder-Mead was under-converging toward
    the lower bound). The ridge fit is unchanged to <0.03% (the GCV is flat),
    so prediction intervals are unaffected. Pinned by
    `tests/test_regressors.py::test_ridge_gcv_eigenbasis_matches_inverse_gcv`.
  - **Lasso CV: warm-started homotopy** — `ConformalLassoRegressor._tune_lambda`
    now walks the lambda grid in descending order and warm-starts each fold's
    coordinate-descent fit from the previous lambda's solution (Lasso solutions
    are monotone in lambda, so nearby starts converge in far fewer sweeps).
    Same folds, same grid, same CV procedure -- now ~3x faster. Warm-starting
    can flip a *tie* between two adjacent grid points with identical CV MSE
    (pinned by `tests/test_lasso.py::test_autotune_selected_lambda_is_cv_optimal`,
    which asserts the selected lambda is CV-near-optimal rather than bit-identical
    to the cold version).
  - **Lasso homotopy loop: solve + vectorised breakpoints** —
    `ConformalLassoRegressor._run_homotopy` now (a) computes the active-set
    solve with `np.linalg.solve(Sigma_J, x_J)` instead of forming the full
    `np.linalg.inv(Sigma_J)` (numerically preferred, identical result), and
    (b) vectorises the per-piece `dt_primal` / `dt_dual` breakpoint searches
    (previously per-variable Python loops). Output is **identical** to the
    reference loop-based homotopy (pinned by
    `tests/test_lasso.py::TestRunHomotopyRefactor`, 40 random problems,
    0 interval differences), delivering a **~1.8x** speedup on `predict`.
  - **Lasso homotopy: vectorized per-piece rank check** —
    `ConformalLassoRegressor._find_intervals_in_piece` (called once per
    homotopy piece, ~twice per prediction) was vectorized: the per-sub-interval
    Python loop (an O(n) `np.abs(...) <= ...` rank scan per crossing) is
    replaced by a single broadcast over all sub-interval midpoints, and the
    Python-level `sorted(set(...))` over the crossing points is replaced by a
    C-level `np.unique`. The change is **provably equivalent** (verified against
    the original loop over 3000 random pieces, 0 differences) and is pinned by
    `tests/test_lasso.py::TestFindIntervalsInPiece`. It delivers a modest
    ~10% reduction in `predict` latency; the dominant remaining cost is the
    per-piece overhead in `_run_homotopy` (a `np.linalg.inv` and
    `np.setdiff1d` per active-set change), which is a separate, larger refactor.

### Added
  - Abstract base `ConformalPredictionSet` with a shared interface
    (`.is_empty`, `.is_discrete`, `.is_continuous`, `.width()`, `.__contains__`, `.coverage`).
  - `EmptyPredictionSet` — explicit sentinel for the empty set ∅.
  - `DiscretePredictionSet` — classifier label sets (`.elements`).
  - `ContinuousPredictionSet` — one or more **disjoint closed** intervals `[a, b]`,
    including rays, degenerate single points `[a, a]`, and `(-∞, ∞)`. Exposes
    `.intervals`, `.is_single_interval`, `.to_single_interval()`, and backward-compat
    `.lower`/`.upper`.
  - Unified `MultiLevelPredictionSet` (replaces both the regressor and classifier
    multi-level classes).
  - All now exported from the top-level `online_cp` namespace.

### Changed

- **Prediction-set return types** — regressor `predict`/`predict_set` now return
  `ContinuousPredictionSet` (and `MultiLevelPredictionSet` for array ε); classifier
  `predict` returns `DiscretePredictionSet`. Intervals are **topologically closed**
  `[a, b]` (boundaries included), matching the standard conformal-regression
  super-level set `{y : p(y) > ε}` = `{y : α(y) ≤ c}`.
- **Mondrian Lasso disjoint sets** — `MondrianConformalRegressor` now returns a
  proper disjoint `ContinuousPredictionSet` instead of collapsing multiple valid
  intervals to a single `[first, last]` interval.
- **Uniform CPS interface** — `DempsterHillConformalPredictiveSystem.predict_cpd` now
  accepts the standard `(x, return_update=False)` signature (the `x` argument is ignored,
  since the model is labels-only) and the redundant `predict` override was removed so it
  inherits the uniform base-class `predict`. `return_update=True` now returns
  `(cpd, {"labels_only": y})`.
- **ConformalMondrianTreeRegressor** — replaced O(n²) batch p-value evaluation with event-driven O(n log n) knot sweep algorithm.
  - Added `_build_knot_events()` method to generate knot-value + flag tuples with crossing-direction tags.
  - Added `_pvalue_from_sweep()` method to perform left-to-right O(n) sweep updating rank tallies in O(1) amortised time.
  - Knot events are tagged with +1 (entry) or -1 (exit) based on whether the outside-leaf nonconformity score crosses above or below the test score as the candidate y increases.
  - The p-value p(y) is piecewise-constant with O(n) algebraic breakpoints where NCM ranks change; the event-driven sweep finds the valid region {y : p(y) > ε} in O(n log n) time.
- **Mondrian tree CPS — leaf-local CPD** — `MondrianTreePredictionMachine.predict_cpd`
  now builds the CPD from the *category* of the test point (the leaf containing
  it, or, when that leaf holds no training data, the deepest ancestor whose
  subtree does — the empty-leaf walk-up): the interval-valued step CDF of the
  category's A training labels, with knots at the sorted category labels
  (ALRW2 §7.2 band). The previous implementation used the global empirical CDF
  of *all* training labels and left the computed leaf unused, so the partition
  did not drive the prediction. See `.roadmap/mondrian-cps-verification.md`
  for the construction and its finite-sample validity boundary.

### Removed / Deprecated

- `online_cp.regressors.ConformalPredictionInterval` — **deprecated** (use `ContinuousPredictionSet`).
  Kept as a `DeprecationWarning`-raising subclass with the legacy `(lower, upper, ε)` signature
  and `.lower`/`.upper` access.
- `online_cp.regressors.MultiLevelPredictionInterval` — **deprecated** (use `MultiLevelPredictionSet`).
- `online_cp.classifiers.ConformalPredictionSet` — **deprecated** (use `DiscretePredictionSet`).
  Kept as a `DeprecationWarning`-raising subclass for downstream compatibility.

### Fixed

- **Ruff Linting** — fixed import sorting (I001) and loop variable binding (B023) issues in tests.
- **Mondrian graphviz fallback** — `draw()` now degrades gracefully to the matplotlib backend when the `graphviz` Python package is installed but the system `dot` binary is missing, instead of crashing inside `dot`.
- **GitHub Pages Deployment** — fixed docs workflow to properly create virtual environment, install dependencies, and deploy documentation.
- **Mondrian CPD between-knot band** — `MondrianPredictiveDistributionFunction._cdf_bounds`
  returned the *next* knot's upper value for `y` strictly between knots,
  inflating the CPD band between knots (e.g. labels {1, 2}, y = 1.5 gave
  [0.5, 1.0] instead of [0.5, 0.5]). Fixed: the band collapses to the single
  CDF value `L[i] = U[i − 1]`; added a degenerate empty-category guard (band
  [0, 1] everywhere).
- **Mondrian CPD quantiles / prediction sets** — `_compute_quantile` used the
  τ-interpolated at-knot band instead of the single between-knot CDF value
  `U[j] = L[j + 1]`, making quantiles and prediction sets too narrow
  (e.g. `quantile(0.5, 0.5)` for labels {1, 2} returned 2.0 instead of just
  above 1.0). Fixed, plus the n = 0 (empty category) quantile rule
  (−∞ iff τ ≥ p).
- **`ConformalMondrianTreeRegressor.predict` empty-leaf path** — the A == 0
  branch (empty leaf → vacuous "total epistemic uncertainty" set) returned
  before caching `self._last_tree`, so `draw_partition()` after such a predict
  raised `RuntimeError`. The tree is now cached immediately after it is
  built, on all branches.
- **Mondrian notebook** — `notebooks/mondrian.ipynb` CPS section updated to the
  leaf-local construction (shallow tree, 800-point regression demo); the
  depth-vs-coverage trade-off at finite samples is documented inline.
- **k-NN tie-breaking: bag-function (permutation) invariance** — the October
  2026 12-module audit found three bare `np.argpartition` k-nearest selections
  that were unstable on distance ties, so the prediction changed with the
  training-row order on tie-rich data (a violation of the Mondrian/conformal
  bag-function requirement). Fixed by replacing them with a canonical
  `np.lexsort` tie-break:
  - `ConformalNearestNeighboursRegressor.compute_p_value` (test-point k-nearest
    → `lexsort((y, d))`, matching the training path in `_knn_predictions`);
    previously p was 1.0 for one insertion order and ≈ 0.71 for the other three.
  - `NearestNeighboursVennPredictor._compute_taxonomies` and
    `_compute_taxonomies_multiclass` (→ `lexsort((labels, d))`); previously the
    Abers p0/p1 ranged p0 0.43→0.0, p1 0.5→0.75 across orders.
  Regression guards: `tests/test_knn_regressor.py::test_pvalue_tie_invariant_across_insertion_orders`
  and `tests/test_venn_abers.py::TestNearestNeighboursVennPredictor::test_knn_taxonomy_tie_invariant_across_insertion_orders`.
  See `.roadmap/audit-2026-10.md`.

### Known issues

- **Lasso homotopy under-covers on some problems.**  The
  `ConformalLassoRegressor.predict` prediction set can be **narrower** than the
  true conformal set — i.e. it may exclude candidate labels whose p-value
  exceeds `epsilon` (anti-conservative, potentially overconfident intervals).
  This is a **pre-existing** correctness bug in the homotopy trace, **not**
  introduced by the performance changes in this release: the refactored
  `_run_homotopy` is provably behaviour-identical to the original (pinned by
  `tests/test_lasso.py::TestRunHomotopyRefactor`).  `compute_p_value` agrees
  with the theoretical definition; `predict` under-covers relative to it.
  Fixing the breakpoint bookkeeping in `_run_homotopy` is tracked separately
  and is **out of scope** for this performance-focused release.  Users who need
  guaranteed validity should rely on `compute_p_value` or on non-Lasso
  estimators until this is resolved.

---

## [0.3.1] — 2026-09-24
 
### Added
 
- **Mondrian trees & forests** (`online_cp`) — promoted the Mondrian process
  predictors from development to production and reorganised them under a new
  `online_cp.mondrian` subpackage.
  - **`MondrianTree`** (`online_cp.mondrian`) — a standalone Mondrian partition
    core: `grow(X, rng, lifetime, ...)`, `find_leaf`, `collect_leaves`,
    `struct_stats`, `to_dataframe`, `draw` / `draw_partition`. Takes an external
    RNG and is free of predictor-module dependencies. The partition is a bag
    function (permutation-invariant); string lifetimes (`'sqrt_n'`, `'density'`)
    grow a deterministic truncation of the master tree (Balog §2.3).
  - **`ConformalMondrianTreeClassifier`, `ConformalMondrianForestClassifier`**
    (`online_cp.classifiers`) — transductive conformal classifiers using
    Leaf-Local Laplace Smoothing as the nonconformity measure.
  - **`ConformalMondrianTreeRegressor`, `ConformalMondrianForestRegressor`**
    (`online_cp.regressors`) — conformal regressors using the leaf-residual NCM;
    the single tree solves the prediction interval exactly via an O(n log n)
    knot-point solver (vectorised p-values via binary search over the
    candidate-independent leaf scores).
  - **`MondrianVennPredictor`** (`online_cp.venn`) — a Venn predictor using the
    Mondrian leaf as the taxonomy category (bag function → Venn validity,
    ALRW2 Thm 6.4); refactored onto `MondrianTree`.
  - All six classes are exported from the top-level `online_cp` namespace and
    documented under *API Reference → Mondrian Prediction*.
  - The group-conditional wrappers `MondrianConformalClassifier` /
    `MondrianConformalRegressor` moved from `online_cp/mondrian.py` to
    `online_cp/mondrian/framework.py`; their public import path
    (`from online_cp import ...`) is unchanged.
  - Added property (permutation invariance) and adversarial (tie-break
    order-invariance) tests for the Mondrian partition, plus `save` / `load`
    round-trip tests for all five predictors.

- **Online (incremental) Mondrian predictors** — all four Mondrian conformal
  predictors gained an opt-in `online=True` mode for streaming data.
  - `ConformalMondrianTree{Classifier,Regressor}` keep a single persistent tree
    and `ConformalMondrianForest{Classifier,Regressor}` keep `n_trees`
    persistent trees. `predict` projects the test point into the partition with
    the `ExtendMondrianBlock` update (Roy & Teh 2009; Lakshminarayanan et al.
    2014) in `O(depth)` per tree instead of regrowing it from scratch. Batch
    mode remains the default and reference implementation.
  - `learn_one` persists the exact extension that `predict` scored (surfaced via
    `return_update=True` and an internal cache), so the realised stream is a
    single consistent draw of the Mondrian process — giving the clean online
    i.i.d.-error-sequence guarantee (ALRW2 §2.2.9), not merely per-step marginal
    validity. A staleness guard falls back to a fresh extension under
    delayed / out-of-order feedback.
  - Requires a fixed float `lifetime` and `max_depth=None` (the projective
    Mondrian regime); the persistent tree(s) are serialised by `save` / `load`.
  - Typical speedups over rebuilding each step (200 predict + learn, n 200→400):
    tree regressor ~6×, forest classifier ~6×, forest regressor ~3×.

- **Random Number Generation** — expanded `seed` and `rnd_state` arguments in regressors, classifiers, and betting strategies to accept `np.random.Generator` objects in addition to integer seeds.

- **Optimistic-FTRL Barrier Legendre Martingale** (`online_cp.martingale`) —
  added `OptimisticFTRLBarrierLegendreMartingale(orders, eta)` as a jumpless
  multivariate Legendre betting martingale using the Optimistic-FTRL + Barrier
  update rule (Wang et al., Algorithm 3) with the same product betting function
  family as `ProductLegendreJumper`.
  - Uses an exact analytical gradient for all `|K|`:
    $\nabla_{\epsilon_k}\ell_t = -\frac{P_k(p_t)}{1+\epsilon_k P_k(p_t)} + \frac{1}{Z(\epsilon)}\frac{\partial Z(\epsilon)}{\partial \epsilon_k}$.
  - Adds `_compute_Z_and_gradient(eps)` in `legendre.py`, computing both
    `Z(\epsilon)` and `\nabla Z(\epsilon)` exactly from precomputed
    Gaunt-coefficient terms.
  - Exported from both `online_cp.martingale` and top-level `online_cp`.
  - Test coverage added:
    - `tests/test_martingale_legendre.py`: behavioural tests plus regression
      test that `orders=[1]` matches a raw 1D Wang FTRL+Barrier loop exactly.
    - `tests/test_properties_adversarial.py`: adversarial properties A9/A10
      for density validity, positivity/finite wealth, and interior
      parameter stability (`|\epsilon_k| < 1`).
 
### Fixed

- **Ruff Linting** — fixed import sorting (I001) and loop variable binding (B023) issues in tests.
- **Mondrian graphviz fallback** — `draw()` now degrades gracefully to the
  matplotlib backend when the `graphviz` Python package is installed but the
  system `dot` binary is missing, instead of crashing inside `dot`.
- **GitHub Pages Deployment** — fixed docs workflow to properly create virtual environment, install dependencies, and deploy documentation.

---

## [0.3.0] — 2026-06-18

### Added

- **Legendre Jumper Martingales** (`online_cp.martingale`) — four new conformal
  test martingales based on Legendre polynomial betting, following:

  > "([LegendreJumper, preprint]).

  - `SimpleLegendreJumper(order, J, epsilon_grid)` — Algorithm 2; single-order
    Legendre polynomial basis with Markov expert chain.
  - `ProductLegendreJumper(orders, J, epsilon_grid)` — Algorithm 3; full
    Cartesian product state space E_1 × ⋯ × E_K with Z-normalised product
    betting function.
  - `VariationalLegendreJumper(orders, J, epsilon_grid)` — Algorithm 4;
    O(|K|·g) per-step linear alternative using independent sub-jumpers,
    consensus parameters, and precomputed Gaunt coefficients.
  - `CompositeLegendreJumper(base_class, J)` — arithmetic mean over multiple
    jump rates; analogue of `CompositeJumper` for Legendre families.
  - Utility functions: `shifted_legendre_poly`, `compute_normalization_Z`,
    `product_betting_value`, `STANDARD_GRID`.
  - All four classes exported from `online_cp` top-level.
  - 45 tests in `tests/test_martingale_legendre.py`; 2 adversarial leancheck
    properties in `tests/test_properties_adversarial.py` (A7, A8).

- **`VennPredictor` base class** (`online_cp.venn`) — shared abstract base for
  all Venn predictors, analogous to `ConformalClassifier`/`ConformalRegressor`:
  - Manages `label_space` initialisation and incremental updates
    (`_update_label_space_batch`, `_update_label_space_one`).
  - Provides `_empty_prediction()` and the taxonomy dispatch hook
    `_categories_for_hypothesis` / `_venn_predict_from_taxonomy` for
    subclasses that use partition-based taxonomies.
  - `VennAbersPredictor` and `NearestNeighboursVennPredictor` now inherit
    `VennPredictor`; all existing behaviour is unchanged.

  - **`PCA` transformer** (`online_cp.preprocessing`) — principal-component rotation
    for use as a preprocessing step in `Pipeline`.
    - Computes the unbiased sample covariance matrix and decomposes it via
      `numpy.linalg.eigh`; eigenvectors are sorted by descending explained variance
      and given a deterministic sign convention (largest-magnitude element positive).
    - Parameters: `n_components` (int or None), `mode` ("frozen"/"bag").
    - Attributes: `mean_`, `components_` (shape k×d), `singular_values_`, `n_`.
    - Full `save`/`load` serialization via `SerializableMixin`.
    - Primary use-case: axis-aligning features before Mondrian conformal methods to
      yield tighter, more balanced partitions.

- **`SVD` transformer** (`online_cp.preprocessing`) — truncated right-singular-vector
    projection for dimensionality reduction and multicollinearity removal.
    - Identical algorithm to `PCA`; adds a `center` parameter (default `True`).
      When `center=True` the result is numerically identical to `PCA`; `center=False`
      uses the raw (uncentred) Gram matrix with an `n` denominator.
    - Parameters: `n_components`, `mode`, `center`.
    - Full `save`/`load` serialization via `SerializableMixin`.
    - Primary use-case: stabilising `ConformalRidgeRegressor` and `RidgePredictionMachine`
      on near-collinear or high-dimensional feature matrices.

  - Both `PCA` and `SVD` exported from `online_cp` top-level.
  - 27 unit tests in `tests/test_preprocessing.py`.
  - 8 round-trip serialization tests in `tests/test_save_load.py`.
  - 2 leancheck orthonormality properties (P22, P23) in `tests/test_properties.py`.
  - `docs/api/pipeline.md` updated with mkdocstrings entries for both classes.

### Changed

- **`ConformalSupportVectorMachine` nonconformity measure** — default NCM
  changed from Lagrange multiplier (`alpha_i`) to signed margin
  (`-(y_i · f(x_i))`, where `f` is the SVM decision function). The margin
  NCM is continuous (no ties), improving efficiency (smaller prediction sets)
  on noisy and overlapping data while preserving the coverage guarantee.
  The original `alpha` NCM is retained via `nonconformity='alpha'`; the new
  default is `nonconformity='margin'`. The multiclass same-class restriction
  required for validity in OVR decomposition applies to both NCMs.

- **`online_cp.martingale` is now a package** (`martingale/`) split into
  submodules `base`, `jumpers`, `sleepers`, `wrappers`, `legendre`. All
  existing public names remain importable at the same paths — no breakage.

### Added

- **Save / load serialization** — all model classes now support `save(filepath)` /
  `cls.load(filepath)` round-trips:
  - `SerializableMixin` in `online_cp._serialization` — declarative mixin; subclasses
    list `_SAVE_PARAMS`, `_SAVE_STATE`, `_SAVE_CALLABLES`, `_PARAM_MAP` tuples; the
    mixin provides generic `save`/`load` without boilerplate.
  - **Exact RNG reproducibility**: the full `bit_generator.state` dict is saved, not
    just the seed, so predictions are identical after any number of online updates.
  - **Named-callable registry**: `@register_callable("name")` decorator + exported
    helpers `to_token` / `from_token`.  Module-level functions and `Kernel` objects
    round-trip automatically; lambdas raise `SerializationError` with a clear message.
  - **Versioned envelope**: each file records `format_version`, `library_version`, and
    `class`; a `UserWarning` is emitted on library-version mismatch; an error is raised
    on class mismatch or unsupported format version.
  - Coverage: `ConformalRidgeRegressor`, `KernelConformalRidgeRegressor`,
    `ConformalNearestNeighboursRegressor`, `ConformalLassoRegressor`,
    `ConformalNearestNeighboursClassifier`, `ConformalSupportVectorMachine`,
    `RidgePredictionMachine`, `KernelRidgePredictionMachine`,
    `NearestNeighboursPredictionMachine`, `DempsterHillConformalPredictiveSystem`,
    `VennAbersPredictor`, `NearestNeighboursVennPredictor`,
    `ConformalMondrianTreeClassifier`, `ConformalMondrianForestClassifier`,
    `ConformalMondrianTreeRegressor`, `ConformalMondrianForestRegressor`,
    `MondrianConformalRegressor`, `MondrianConformalClassifier`,
    `ConformalPredictiveDecisionMaker` (custom), `Pipeline` (custom, preserves
    bag-mode raw training data).
  - New public exports: `SerializationError`, `register_callable`.
  - 59 round-trip tests in `tests/test_save_load.py`.

- **Decision-making module** (`online_cp.decision`):
  - `ConformalPredictiveDecisionMaker`: Vovk & Bendtsen (2018) Algorithm 1 —
    maintains one CPS per decision, trained on utility-transformed labels.
  - `UtilityFunction`: bundles a utility callable with its decision space.
  - `cps_decision`, `cps_expected_utilities`: single-CPD decision making.
  - `venn_decision`, `venn_expected_utilities`: Venn multiprobability decisions.
  - `alpha_utility`, `alpha_regret`: Hurwicz and minimax-regret criteria.
- **Calibration diagnostics**:
  - `CalibrationError` metric (binned ECE, uniform/quantile strategies,
    `use_hypothesis` mode for Venn validity checking).
  - `plot_reliability_diagram`, `plot_reliability_diagram_venn`,
    `plot_sharpness`, `plot_pit_histogram`, `plot_calibration_conditional`.
- **Venn/probabilistic metrics**: `BrierScore`, `LogLoss`, `Width` for
  scoring Venn multiprobability predictions.
- **CPD scoring metrics**: `TruncatedCRPS`, `ConformalCRPS` (exact
  piecewise-constant summation, no numerical quadrature).
- **sklearn wrapper improvements**:
  - `warm_start` parameter ("auto"/True/False) on `ConformalClassifierWrapper`.
  - `n_jobs` parameter for parallel label loop via joblib.
  - Base fit caching between predict calls (invalidated on `learn_one`).
- `VennAbersPredictor`: multiclass support (|Y| > 2) via OVR isotonic
  calibration. All 4 scorers (ridge, kernel_ridge, knn, svm) supported.
- `VennAbersPredictor`: kernel-ridge scorer added.
- `MulticlassVennPrediction`: output type for multiclass Venn predictors —
  |Y| × |Y| multiprobability matrix with `.point` aggregation property.
- `NearestNeighboursVennPredictor`: Online Venn predictor with k-NN voting
  taxonomy (ALRW2 §6.2). Binary and multiclass.
- `VennAbersPredictor`: Full online Venn-Abers predictor (Algorithm 6.1,
  ALRW2 §6.4). First known Python implementation of full/transductive variant.
- `VennPrediction`: Unified output type for all Venn predictors.
- `log_loss_point(p0, p1)`, `brier_point(p0, p1)`: merge Venn pairs into
  single probabilities minimising log loss / Brier loss.
- `ConformalNearestNeighboursRegressor`: Online conformal k-NN regressor
  (§2.4, ALRW2) with leave-one-out predictions, configurable k,
  mean/median aggregation, and custom distance functions.
- `VilleWrapper`: Ville's inequality procedure (§8.4.1, ALRW2).
- `SleeperStayer`: Algorithm 9.4 (ALRW2) for change-point detection.
- `SleeperDrifter`: Algorithm 9.5 (ALRW2) for gradual drift detection.
- `CUSUMWrapper`: Page CUSUM wrapper (§8.3, ALRW2).
- `ShiryaevRobertsWrapper`: Shiryaev-Roberts wrapper (§8.3, ALRW2).
- `PiecewiseConstantBetting`: betting function f_{(a,b)} (§9.2, ALRW2).
- `plot_detector`: visualise change-point detection wrappers with alarm markers.
- `kernel_induced_distance`, `kernel_matrix_to_distance_matrix`: convert
  kernels to distance metrics for k-NN methods.
- Label-conditional Mondrian classifier (`category_fn='label'`).
- Type annotations on all public API methods.
- "Which method?" decision guide (`docs/guide.md`, `notebooks/guide.ipynb`).
- Benchmarking infrastructure: 5 datasets, 12 model configs, CLI.
- **API contract specification** (`.roadmap/REQUIREMENTS.md`):
  - Formal interface contracts for all 8 module categories: regressors, classifiers, Venn, CPS, martingales, betting strategies, decision-making, evaluation.
  - Versioning & deprecation policy (SemVer, 1-minor deprecation cycle).
  - Stability classification (Stable/Beta/Experimental per category).
  - Thread safety disclaimer.
- **Weak & Lazy Teacher evaluation** (ALRW2, §3.3): all four progressive-
  validation functions (`progressive_val`, `iter_progressive_val`,
  `progressive_val_venn`, `iter_progressive_val_venn`) now accept a
  `delay` parameter (int or callable `(step, x, y) → int`) that schedules
  labels in a min-heap priority queue and resolves them when they arrive,
  directly implementing Vovk's teaching-schedule formalism:
  - **Weak / slow teacher**: fixed or dynamic label lag; LIL-rate asymptotic
    validity preserved for invariant conformal predictors (Thm 3.7–3.11).
  - **Lazy teacher**: pass `y = None` in stream items to skip metric update
    and learning for that step while still emitting a prediction.
  - `iter_*` functions emit a final teardown snapshot after the stream is
    exhausted so the last yielded snapshot is always fully resolved.
  - `delay=0` (default) is backward-compatible with prior behaviour.

### Fixed

- **Property test suite aligned with LeanCheck idiom** (`tests/test_properties.py`,
  `tests/test_properties_adversarial.py`):
  - `leancheck.precondition()` now used for all vacuous-case guards (8 locations).
    Previously `if condition: return True` inflated the passed-test count; skipped
    cases are now reported honestly (e.g. `passed 354 tests` instead of `360`).
  - Domain-specific `Enumerator` types registered as float subclasses:
    `Epsilon` (12 significance levels including 0.01, 0.02), `UnitProb` (τ, p₀/p₁
    on a 101-point [0, 1] grid), `PValue` (99-point open-(0, 1) grid). Property
    signatures now express domain constraints directly; int→domain helper functions
    (`_eps`, `_unit_prob`, `_tau_val`, `_pvals`) removed. LeanCheck exhausts finite
    enumerators and reports `(exhausted)` — nesting properties now confirm complete
    pairwise coverage at 132 tests, CPD boundary conditions at 101 tests.
  - Three-tier test budget (`_TESTS_SLOW = 360`, `_TESTS_MED = 720`,
    `_TESTS_FAST = 3600`) replaces the former two-tier `MAX_TESTS = 200` / 500 split.
    Sequence-loop and matrix-decomposition properties were previously misclassified
    as "fast"; they now use the correct MED tier. `minimum_training_set` properties
    switched from `k: int` with modular arithmetic (wasted 3500+ repetitions at 3600
    tests) to `eps: PValue` (exhausted at 99 tests in < 5 ms).
  - `leancheck.main(verbose=True, exit_on_failure=False)` added to both files so
    `python3 tests/test_properties.py` runs all properties with LeanCheck's native
    per-property report.
  - `_check()` wrapper removed; all `test_*` functions call `leancheck.check()`
    directly with an explicit, named budget constant.

- **martingale.py direct script execution**: Handle relative imports with try/except fallback for both package context and direct script execution (needed for CI `run-modules` step).
- **Holistic audit findings** (pre-v0.3.0 freeze):
  - Exported but undocumented classes added to API docs: `Kernel`, `CustomKernel` (kernels.md), `ConformalPredictiveDecisionMaker` (decision.md), `MulticlassVennPrediction` (venn.md).
  - New dedicated documentation page for betting strategies (`docs/api/betting.md`, was scattered in martingale.md).
  - Removed experimental `ConformalClassifierWrapper` reference from README (not exported).
  - Version alignment: CITATION.cff updated 0.1.1 → 0.2.0.

### Changed

- **Documentation pass** — all public docstrings enriched for v0.3.0:
  ALRW2 section/theorem citations, MathJax LaTeX formulas, property
  classifications (proper scoring rules vs. efficiency criteria), and full
  parameter docs across all modules. RST directives (`.. math::`,
  `.. warning::`) converted to MkDocs/admonition syntax. Pre-existing
  broken module-level doctests fixed (`evaluate`, `Metrics`, `PluginMartingale`,
  `UtilityFunction`). `martingale_legendre_dev.py` removed (superseded by
  `legendre.py`).

- **BREAKING**: `VennPrediction` is now a unified class for binary *and*
  multiclass predictions. Binary: `VennPrediction.binary(p0, p1)`.
  Multiclass: `VennPrediction(probs, label_space)`.
- **BREAKING**: Label-space policy for `NearestNeighboursVennPredictor` and
  conformal classifiers: default is now `label_space=None` (adaptive).
  Pass explicit `label_space=[...]` to lock.
- **BREAKING**: `VennAbersPrediction` renamed to `VennPrediction`.
- **BREAKING**: `CRPS` now delegates to `TruncatedCRPS` (identical
  behaviour, emits deprecation warning).
- Decision criteria renamed: `alpha_utility` / `alpha_regret` (was
  `hurwicz` / `minimax_regret`).
- Martingale architecture: martingales are pure evidence processes;
  statistical decision procedures are separate wrappers.
- CPS module: removed unnecessary parameter constraints, improved
  performance with `solve_triangular`, improved tie handling.
- `GaussianKDE` numba functions use `cache=False` to avoid stale bytecode.
- API consistency fixes: `PeriodicKernel.name`, `LinearCombinationKernel.name`,
  `evaluate.py` tau handling, `predict()` canonical on Venn/CPS.
- README updated with decision-making section, expanded Features table,
  and proper references (Lei 2019, Vovk & Bendtsen 2018).

### Removed

- `update_martingale_value()` alias — use `.update(p)` directly.
- `warnings` and `warning_level` parameters from martingale constructors.
- `check_warning()`, `.max`, `.log_max` from martingale base class.
- `conformal_expectation` from CPS module (use `cps_decision` instead).

## [0.2.0] — 2026-05-19

### Added

- **Mondrian conformal prediction**: `MondrianConformalRegressor` and `MondrianConformalClassifier` for group-conditional coverage via a single pooled model with category-filtered calibration.
- **Streaming evaluation**: `progressive_val()` and `iter_progressive_val()` — River-style test-then-train loops with composable metrics.
- **Plotting utilities**: `plot_coverage`, `plot_martingale`, `plot_intervals`, `plot_set_sizes`.
- **Composable metrics module**: `ErrorRate`, `IntervalWidth`, `WinklerScore`, `CRPS`, `SetSize`, `ObservedExcess`, `ObservedFuzziness` — combine with `+`, query with `.get()` and `.cumulative_mean()`.
- **Optional numba acceleration**: install with `pip install online-cp[fast]` for faster Lasso homotopy and KDE.
- **Multi-level predictions**: pass `epsilon=[0.01, 0.05, 0.1]` to get predictions at multiple significance levels in a single call.

### Changed

- Martingale module redesigned: cleaner API with `PluginMartingale`, `SimpleMixtureMartingale`, `SimpleJumper`, `CompositeJumper`.
- Development status upgraded from Pre-Alpha to Alpha.

## [0.1.1] — 2024-12-01

### Added

- Initial release on PyPI.
- `ConformalRidgeRegressor`, `KernelConformalRidgeRegressor`, `ConformalLassoRegressor`.
- `ConformalNearestNeighboursClassifier`, `ConformalSupportVectorMachine`.
- Conformal predictive systems: `RidgePredictionMachine`, `KernelRidgePredictionMachine`, `NearestNeighboursPredictionMachine`, `DempsterHillConformalPredictiveSystem`.
- Kernel library: `GaussianKernel`, `LinearKernel`, `PolynomialKernel`, `PeriodicKernel`, `LinearCombinationKernel`.
- Test martingales: `PluginMartingale`, `SimpleMixtureMartingale`.

[0.2.0]: https://github.com/egonmedhatten/online-cp/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/egonmedhatten/online-cp/releases/tag/v0.1.1
