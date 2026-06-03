# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

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
