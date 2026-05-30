# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SleeperStayer`: Sleeper/Stayer martingale (Algorithm 9.4, ALRW2) for change-point detection.
- `SleeperDrifter`: Sleeper/Drifter martingale (Algorithm 9.5, ALRW2) for gradual drift detection.
- `CUSUMWrapper`: Page CUSUM wrapper for any conformal test martingale (§8.3, ALRW2).
- `ShiryaevRobertsWrapper`: Shiryaev-Roberts wrapper for any conformal test martingale (§8.3, ALRW2).
- `PiecewiseConstantBetting`: Piecewise-constant betting function f_{(a,b)} (§9.2, ALRW2).

### Changed

- CPS module refactored: removed unnecessary parameter constraints, improved performance with `solve_triangular`, improved tie handling in `NearestNeighboursPredictionMachine`.
- `GaussianKDE` numba functions use `cache=False` to avoid stale bytecode cache issues.

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
