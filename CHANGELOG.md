# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — targeting 0.3.1

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
