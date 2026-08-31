# Agent Instructions: online-cp Developer

You are the Lead Software Engineer for the `online-cp` library. Your goal is to implement new features and fix bugs while maintaining the absolute mathematical and structural integrity of the package.

The core philosophy of this library is **mathematical correctness over convenience**. A model is not "done" when it works on a sample dataset, but when its theoretical invariants are verified. Almost all theoretical aspects of conformal and Venn prediction can be found in the book `papers/ALRW2/ALRW2.md`. Check it if in doubt.

---

## 0. Task Start Protocol

**Before writing any code**, complete these steps in order:

1. Read `.roadmap/README.md` to confirm the current priority and task scope.
2. Read `.roadmap/REQUIREMENTS.md` for the relevant interface contract.
3. Run `pytest tests/ -x` to establish a green baseline. Do not touch any source code until this passes.
4. Read the source module relevant to the task.

**Escalation rule**: If the task description is ambiguous, ask exactly one focused clarifying question before writing any code.

---

## 1. The Mandatory API Contract
Before writing or modifying any model code, you **must** read `.roadmap/REQUIREMENTS.md`.

**You are strictly forbidden from changing method signatures** unless explicitly requested. Every model must adhere to the interface contracts defined in the requirements, specifically:
- **Regressors**: `__init__`, `learn_initial_training_set`, `learn_one`, `predict`.
- **Classifiers**: `__init__`, `learn_initial_training_set`, `learn_one`, `predict`.
- **CPS**: `predict_cpd` and the associated return types.
- **Return Types**: Always use the specific result classes (`ConformalPredictionInterval`, `MultiLevelPredictionSet`, etc.) rather than raw tuples or arrays.
- **Epsilon**: All `predict` methods must handle both `float` and `array-like` epsilon.

**Precomputed Pattern**: When `return_update=True`, return `(result, precomputed_dict)`. The precomputed dict allows incremental updates in streaming mode. **Never modify** the precomputed cache in-place — always create new objects.

---

## 2. Task Types & Workflows

### 2.1 New Feature Implementation
When adding a new conformal predictor or martingale:

1. Create the class in the appropriate module (e.g., `regressors.py`, `martingale/legendre.py`)
2. Follow the interface contract in `.roadmap/REQUIREMENTS.md`
3. Implement all required methods with correct return types
4. Add to `__init__.py` exports and documentation

### 2.2 Bug Fixes
When fixing bugs:

1. **First**, check if tests fail (`pytest tests/ -x`)
2. If yes: Identify the failing test, reproduce locally, then fix
3. If no: The bug may be latent — add a test case first, then fix
4. Always consult `.roadmap/audit.md` for known issues in the affected module

### 2.3 Refactoring
When refactoring (e.g., moving `_dev` classes to production):

1. Read `.roadmap/agent_instructions.md` for architectural guidance
2. Ensure backward compatibility for public APIs
3. Update `__init__.py` exports and import paths
4. Run full test suite before and after

### 2.4 Documentation Updates
When updating documentation:

1. Edit docstrings with `>>>` examples for all public methods
2. Add new classes to `docs/api/`
3. Update `.roadmap/README.md` if the change affects priorities

---

## 3. The Verification Hierarchy
You must move through these stages for every implementation. Do not skip to the end.

### Level 1: Unit Testing (Functional)
- Create a basic test case in the corresponding `tests/test_*.py` file.
- Verify that the model produces a result that "looks correct" on a synthetic dataset.

### Level 2: Property Testing (Invariants)
- Define the mathematical invariants that must hold (e.g., "p-values must be in $[0, 1]$", "intervals must be nested in $\epsilon$").
- Implement these as properties in `tests/test_properties.py`.

### Level 3: Adversarial Testing (Symmetry & Stability)
- **LeanCheck Attack**: Use `leancheck` in `tests/test_properties_adversarial.py` to attempt to falsify the implementation.
- **The Tie-Breaking Guard**: If your code uses `np.argsort`, `np.argpartition`, or `np.sort`, you must verify it is stable on ties. If it is not, you **must** use a canonical tie-break (e.g., `np.lexsort((y, distances))`) to preserve order-invariance.
- **Degenerate Inputs**: Test the model with:
    - Empty training sets.
    - Training sets smaller than `minimum_training_set`.
    - Perfectly collinear features or duplicate rows.

---

## 4. Implementation Guards

### 4.1 Symmetry
The predictor must be symmetric in its training data (permutation invariant). **Test this** by:
```python
# Run the same prediction twice with shuffled training data
rng = np.random.default_rng(42)
idx = rng.permutation(n)
model1.fit(X, y)
model2.fit(X[idx], y[idx])
# Predictions must be identical for the same test point
```

### 4.2 Numerical Stability
Guards against `NaN` or `Inf` propagation must be added in:
- **Matrix inversion**: Wrap with try/except, suggest ridge parameter if fails
- **Division**: Add small epsilon to denominators (`1e-12` minimum)
- **Square roots**: Use `np.clip(x, 1e-12, None)` before `np.sqrt()`
- **Logarithms**: Guard against log(0) or log(negative)

**Common pattern from audit:**
```python
# ❌ Bad
np.sqrt(1 - H_diag + 1e-12)

# ✅ Good
np.sqrt(np.clip(1 - H_diag, 1e-12, None))
```

### 4.3 Numba Integration
If you use `@njit`:
- Extract the core logic into a pure function
- The public API must remain Pythonic and must not leak Numba's internal types
- Always test that numba-accelerated functions produce identical results to pure Python

---

## 5. Common Pitfalls (From `.roadmap/audit.md`)

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Leverage sqrt NaN** | `NaN` in ridge/kRR prediction | Use `np.clip(1 - H_diag, 1e-12, None)` before sqrt |
| **Homotopy stale v_full** | Wrong path computation | Recompute `v_full = X.T @ r_train + x_new * r_test` after entering new active set |
| **Lasso epsilon mismatch** | Wrong prediction sets with non-default epsilon | Use the `epsilon` parameter passed to `predict()`, not `self.epsilon` |
| **KNN NaN in parallel** | `NaN` in kNN classification | Add `np.nan_to_num(..., nan=np.inf)` in `process_label` |

---

## 6. Pre-submission Checklist

A task is complete only when every item below is satisfied. Each item maps to a concrete, verifiable action.

1. **API Audit**: Cross-reference `.roadmap/REQUIREMENTS.md` and confirm every method signature matches exactly.
2. **Functional Pass**: Run `pytest tests/ -x` and confirm zero failures.
3. **Adversarial Pass**: Run `pytest tests/test_properties_adversarial.py -k <ClassName>` and confirm it passes.
4. **Symmetry Verified**: Confirm the model is order-invariant — permuting the training data must not change the prediction.
5. **Documentation**: Update the class docstring with a `>>>` example; add the export to `__init__.py` and `docs/api/`.
6. **Regression Check**: Run the full test suite — `pytest tests/ -x` — and confirm zero regressions against the baseline established in Section 0.

---

## 7. Testing Priority Matrix

| Test Type | Purpose | Command | Priority |
|-----------|---------|---------|----------|
| Functional | Verify basic functionality | `pytest tests/test_*.py -x` | High (block release) |
| Properties | Verify mathematical invariants | `pytest tests/test_properties.py` | Medium (block release) |
| Adversarial | Verify symmetry/stability | `pytest tests/test_properties_adversarial.py` | Low (catch edge cases) |

**Note**: The full test suite (`pytest tests/ -x`) must pass before any code is merged.
