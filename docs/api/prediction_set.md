# Prediction Sets

The unified conformal prediction-set hierarchy. Every conformal regressor and
classifier in `online-cp` returns one of these objects (or a
`MultiLevelPredictionSet` when `epsilon` is an array-like).

The hierarchy:

```text
ConformalPredictionSet (abstract)
├── EmptyPredictionSet              # ∅
├── DiscretePredictionSet           # {0, 1, 2} — classifiers
└── ContinuousPredictionSet         # 1 to N closed intervals [a, b]
```

- `ConformalPredictionSet` — abstract base with the shared interface
- `EmptyPredictionSet` — the empty set ∅
- `DiscretePredictionSet` — classifier label sets
- `ContinuousPredictionSet` — one or more disjoint **closed** intervals `[a, b]`
- `MultiLevelPredictionSet` — prediction sets at multiple significance levels

::: online_cp.prediction_set.ConformalPredictionSet

::: online_cp.prediction_set.EmptyPredictionSet

::: online_cp.prediction_set.DiscretePredictionSet

::: online_cp.prediction_set.ContinuousPredictionSet

::: online_cp.prediction_set.MultiLevelPredictionSet

## Interval notation

All numeric bounds are **closed** `[a, b]` — the inclusion criterion is the
super-level set `{y : p(y) > ε}`, which for a continuous nonconformity measure
is algebraically `{y : α(y) ≤ c}`, a closed set. Infinity bounds stay
parenthesised because ∞ is not a real number.

| Type | Notation | Python representation |
|------|----------|-----------------------|
| Single / Disjoint | `[a, b]` | `ContinuousPredictionSet([(a, b)], epsilon)` |
| Ray (bounded below) | `[a, ∞)` | `ContinuousPredictionSet([(a, np.inf)], epsilon)` |
| Ray (bounded above) | `(-∞, b]` | `ContinuousPredictionSet([(-np.inf, b)], epsilon)` |
| All reals | `(-∞, ∞)` | `ContinuousPredictionSet([(-np.inf, np.inf)], epsilon)` |
| Single point | `[a, a]` | `ContinuousPredictionSet([(a, a)], epsilon)` |
| Empty | `∅` | `EmptyPredictionSet()` |
