# Stream Pruning

Budget-bounded pruning for the online models that store a growing training bag
(`ConformalNearestNeighboursRegressor`, `ConformalNearestNeighboursClassifier`,
`KernelConformalRidgeRegressor`, `ConformalSupportVectorMachine`). Pass
`pruning="reservoir"` or `pruning="window"` plus `budget=B` to cap the retained
bag at `B` examples. Only **strictly-valid** (data-blind) strategies are
accepted, so the exact finite-sample coverage guarantee is preserved unchanged.

A strategy is valid iff the retained bag is a *bag-function* of the stream — a
symmetric, value-blind sub-selection, randomised only by data-independent
randomness. Model-state / value-dependent selectors (support-vector, leverage,
core-set, self-pruning, stratification) are **rejected** at construction.

```python
import online_cp as ocp

# Built-in: uniform random sample of `budget` examples (Vitter's Algorithm R)
model = ocp.ConformalNearestNeighboursRegressor(k=5, pruning="reservoir", budget=1000)

# Built-in: the most recent `budget` examples (FIFO window)
model = ocp.KernelConformalRidgeRegressor(kernel, pruning="window", budget=1000)

# Custom: subclass PruningStrategy (must pass validate_strategy)
model = ocp.ConformalNearestNeighboursClassifier(k=5, pruning=my_strategy, budget=1000)
```

## PruningStrategy

::: online_cp._pruning.PruningStrategy

## validate_strategy

::: online_cp._pruning.validate_strategy
