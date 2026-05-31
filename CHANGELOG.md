# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `VennAbersPredictor`: Full online Venn-Abers predictor (Algorithm 6.1, ALRW2 §6.4) producing calibrated multi-probability predictions for binary classification. Supports ridge regression and k-NN scoring functions. First known Python implementation of the full/transductive variant.
- `VennAbersPrediction`: Output type for Venn-Abers predictions — the multiprobability pair (p0, p1).
- `log_loss_point(p0, p1)`: Merge a Venn-Abers pair into a single probability minimising log loss (ALRW2 §6.4).
- `brier_point(p0, p1)`: Merge a Venn-Abers pair into a single probability minimising Brier loss.
- `ConformalNearestNeighboursRegressor`: Online conformal k-NN regressor (§2.4, ALRW2) with leave-one-out k-NN predictions, configurable k, mean/median aggregation, and custom distance functions.
- `VilleWrapper`: Ville's inequality wrapper for any conformal test martingale — rejects when running maximum exceeds threshold (§8.4.1, ALRW2).
- `SleeperStayer`: Sleeper/Stayer martingale (Algorithm 9.4, ALRW2) for change-point detection.
- `SleeperDrifter`: Sleeper/Drifter martingale (Algorithm 9.5, ALRW2) for gradual drift detection.
- `CUSUMWrapper`: Page CUSUM wrapper for any conformal test martingale (§8.3, ALRW2).
- `ShiryaevRobertsWrapper`: Shiryaev-Roberts wrapper for any conformal test martingale (§8.3, ALRW2).
- `PiecewiseConstantBetting`: Piecewise-constant betting function f_{(a,b)} (§9.2, ALRW2).

### Changed

- Martingale architecture redesigned: martingales are now pure evidence processes. Statistical decision procedures (`VilleWrapper`, `CUSUMWrapper`, `ShiryaevRobertsWrapper`) are separate wrappers.
- CPS module refactored: removed unnecessary parameter constraints, improved performance with `solve_triangular`, improved tie handling in `NearestNeighboursPredictionMachine`.
- `GaussianKDE` numba functions use `cache=False` to avoid stale bytecode cache issues.

### Removed

- `update_martingale_value()` alias — use `.update(p)` directly.
- `warnings` and `warning_level` parameters removed from all martingale constructors.
- `check_warning()`, `.max`, `.log_max` removed from martingale base class (use `VilleWrapper` instead).

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
