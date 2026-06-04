# Repo Sanity Check — Audit Log

Module-by-module findings. Items marked [FIX] need action before v0.3.0.
Items marked [OK] have been verified or are acceptable.

---

## regressors.py

**Status:** Audited (7 passes + final scan). Clean.

### Pass 1 changes (earlier sessions):
1. Wrapped all `np.linalg.inv` calls with try/except → clear `ValueError` suggesting ridge parameter
2. Homotopy `v_full` staleness fix: recompute `v_full = X.T @ r_train + x_new * r_test` in empty-J_k entering branch
3. Homotopy search range guard: `if t_max > 0` / `if t_min < 0` before calling homotopy
4. Lasso `compute_p_value`: added `if self.X is None` guard

### Pass 5 changes (9fe7e36):
1. [FIX] Studentised leverage sqrt: `np.sqrt(1 - H_diag + 1e-12)` → `np.sqrt(np.clip(1 - H_diag, 1e-12, None))` — prevents NaN when H_diag > 1+1e-12 due to Sherman-Morrison drift

### Pass 6 changes (134bc5b):
1. [FIX] Lasso `_run_homotopy` was using `self.epsilon` instead of the `epsilon` parameter passed to `predict()` — caused wrong prediction sets with non-default epsilon

### Pass 6b changes (493c796):
1. [PERF] KernelRidge `predict`: deferred O(n²) kernel matrix update until after epsilon feasibility check — only compute when needed or when `return_update=True`

### Final pass findings (no changes needed):
- [OK] `_vectorised_l_and_u` sort logic: correctly removes self-comparison entry
- [OK] `_get_lower`/`_get_upper` 1-indexed dict with edge-case KeyError → ±inf
- [OK] KernelRidge `_update_Kinv` Schur complement formula (a>0 required by design)
- [OK] kNN uses training-only residuals (exchangeability correct)
- [OK] Lasso homotopy rank computation and interval merging

### Theory verifications (all confirmed correct against ALRW2):
- [OK] Ridge A/B formula — matches ALRW2 Algorithms 2.2–2.5
- [OK] Studentised residuals — matches ALRW2 §7.4, eqs 7.11–7.17
- [OK] KernelRidge Sherman-Morrison — matches ALRW2 eqs 7.31–7.42, Proposition 7.8
- [OK] kNN conformity scores — matches ALRW2 eqs 2.41–2.50
- [OK] `_get_lower`/`_get_upper` quantile indexing — correct per Algorithm 2.4

**Priority:** Low — unusual to call p-value without any training.

---

## classifiers.py

**Status:** Audited (4 passes). Clean.

### Pass 1 changes:
1. [FIX] Critical parallel NaN bug in KNN — added `np.nan_to_num(..., nan=np.inf)` in `process_label`
2. Removed broken `reset()` method (only passed 2/9 params)
3. Removed deprecated `D` parameter from `learn_one` signature + deprecation warning
4. Removed stale TODO comments
5. Removed `__main__` doctest block
6. Cleaned verbose double-compute (dead debug prints that overwrote p-values)

### Pass 2 changes:
1. Removed vestigial `D: Any = None` from `ConformalClassifierWrapper.learn_one`
2. Fixed SVM `learn_one` triple kernel call → now matches predict/compute_p_value pattern

### Pass 3 changes:
1. [FIX] Fallback p-values: was `1.0`, now correctly `τ` (SVM predict, SVM compute_p_value, Wrapper _fallback_prediction)
2. Added `label_space=None` guards in KNN, SVM, and Wrapper `predict()` — returns empty set instead of `TypeError`
3. Added `k >= 1` validation in KNN `__init__`
4. Renamed `_validate_probabilities` → `_validate_scores`, `_align_probabilities` → `_align_scores`, `Prob` → `S`
5. Updated docstrings: "probabilities" → "scores" where referring to predict_proba output

### Pass 4 changes:
1. [FIX] `_compute_p_value` raises `ValueError` on invalid `score_type` (was `NameError` from unbound locals)
2. Fixed SVM docstring: `label_space` "Default [-1, 1]" → "If None, inferred from the first training data"
3. Removed unused `X` parameter from `_ensure_base_fit` (only used `self.X`)
4. Removed extra blank line between `_compute_Gamma` and KNN class

### Theory verifications (all confirmed correct):
- [OK] SMO I_up/I_low sets match Fan-Chen-Lin 2005
- [OK] SMO gradient update formula matches libsvm convention
- [OK] KNN NCM same_d/diff_d matches ALRW2 §2.3
- [OK] `_compute_Gamma` uses strict `>` (correct per theory)
- [OK] Multiclass SVM positive-class-only p-value (OVR exchangeability justified)
- [OK] Wrapper conformity scores from predict_proba (higher score = more conforming → large lt → high p)
- [OK] WSS3 second-order working set selection (correct gain formula)
- [OK] SMO L/H bounds for same-sign and opposite-sign pairs
- [OK] Boundary snapping tolerance (`tol * 1e-2`)
- [OK] Label-space auto-expansion vs fixed validation (consistent across all 3 classifiers)

## CPS.py

**Status:** Audited (7 passes). Clean.

### Pass 1 changes:
1. Removed dead `if hasattr(self, "h")` branch in `RidgePredictionMachine.learn_one` — `self.h` never set, code would crash if reached
2. Removed `if __name__ == "__main__"` doctest block
3. Removed TODO comments (lines 529, 555) about parallel processing
4. Removed "TODO: Write tests" and NOTE comments from `NearestNeighboursPredictiveDistributionFunction`
5. Removed design musing docstring from `NearestNeighboursPredictionMachine.__init__`
6. `raise Exception("Training set is too small...")` → `raise ValueError(...)` in NN predict_cpd
7. Bare `raise Exception` → `raise ValueError('bounds must be "both", "lower", or "upper"')` in predict_set
8. `raise Exception('bounds must be "both"...')` → `raise ValueError(...)` in predict_set minimise_width path
9. [FIX] DempsterHill quantile bug: was only checking at-knot CDF levels, missing between-knot levels. Could overshoot quantile by one knot → slight under-coverage (≤ 1/(n+1)). Fixed with Ridge-style two-case check (j_at vs j_between).
10. Ridge/NN quantile: replaced manual `abs(C[j]) * eps_machine` with `np.nextafter(C[j], np.inf)` — fixes edge case where C[j]==0 produced eps=0.

### Pass 2 changes:
1. [FIX] Replaced timing infrastructure (`import time`, `self.timer`, `self.timers`, `self.last_time`) with clean code — timing was dead/unused across all CPS classes
2. Added module-level docstring

### Pass 3 changes:
1. [FIX] `predict_set` HPD sweep: replaced brute-force grid search with exhaustive CDF-level sweep — correct for all CPD types (Ridge/NN/DH)
2. Refactored `predict_set` multi-epsilon path to use lazy import of `MultiLevelPredictionInterval`

### Pass 4 changes:
1. [FIX] DempsterHill `_compute_quantile`: IndexError when `j_at` sentinel used as index (no at-breakpoint satisfies `p>=1`). Added `j_at < len(self.Y)` guard.
2. [FIX] `find_smallest_epsilon`: infinite loop when no finite prediction set exists at any epsilon. Added `epsilon <= 1.0` cap, returns `None`.

### Pass 5 changes (dead code removal):
1. Removed `rnd_state`/`rnd_gen` from Ridge, KernelRidge, NN — never used (noise was never added)
2. Removed `warnings` parameter from KernelRidge and NN — never read
3. Removed `verbose` parameter from NN and DH — never read
4. Corrected stale "unique labels" docstring in Ridge (claimed noise was added; it was not)

### Pass 6 changes:
1. [FIX] Ridge `_compute_quantile`: returned `+inf` at `p=0, tau=1` because `ceil(-1)=-1`, neither condition fired. Added `p <= 0` early return (`-inf`) and `max(0, ...)` clamping on `j_between`.
2. Removed redundant double computation in KernelRidge `learn_initial_training_set` when `autotune=True` (was computing K, Kinv, H, h_diag, Hy, then immediately recomputing via `change_ridge_parameter`)

### Pass 7 findings (no changes needed):
- All edge cases verified clean: NN quantile at p=0/1, DH with ties, base `_cdf_bounds` boundary
- All imports used (including `warnings` — used in Ridge class)
- NN `_cdf_bounds` override correctly handles `len(L) == len(Y) - 1` convention
- HPD sweep branch selector (`len(self.L) == len(self.Y)`) correctly distinguishes Ridge/DH from NN

### Theory verifications (all confirmed correct against ALRW2):
- [OK] Ridge A/B/C formula — matches ALRW2 Eq 2.47-2.48 (studentized residuals via Sherman-Morrison)
- [OK] KernelRidge h_diag update — block matrix inversion: h_new[i] = h_old[i] - a·d·v_i², h_last = 1 - a·d
- [OK] KernelRidge h_last_row = a·d·v (from H = I - a·(K+aI)^{-1})
- [OK] KernelRidge Hy update: Hy_new = Hy_old - a·d·v·(v^T·y)
- [OK] NN full/single/semi neighbour classification — matches ALRW2 §7 Alg 7.4
- [OK] NN conformity score α_i = |{j ∈ N_k(i): y_j ≤ y_i}| — correct
- [OK] Ridge L/U = (j-1)/n, (j+1)/n — matches ALRW2 Eq 13.14
- [OK] DempsterHill L/U handles ties via searchsorted — consistent with __call__
- [OK] HPD two-pointer sweep — conservative by at most 1/n (acceptable)
- [OK] Ridge leverage < 1 guaranteed for a > 0 (eigenvalues λ/(λ+a))
- [OK] NN boundary: self.X.shape[0] == k works correctly (all points become neighbours)
- [OK] KernelRidge precomputed path consistency (verified by test_precomputed_matches_normal)

## venn.py

**Status:** Audited (4 passes). Clean.

### Pass 1 changes (c741548):
1. Removed dead code: `HAS_NUMBA` flag, unused `searchsorted` in `_isotonic_calibrate`, `__main__` block
2. [FIX] Ridge `learn_initial_training_set` / `learn_one` cold-start: added try/except on `np.linalg.inv` calls → clear `LinAlgError` fallback
3. [FIX] Binary dispatch: non-{0,1} 2-class labels incorrectly entered binary path that hardcodes hypothesis labels 0/1 → now routes to multiclass

### Pass 2 changes (d5c9061):
1. [FIX] `_compute_knn_scores_binary`: empty class-1 asymmetry — `d_to_1 = 0.0` when class 1 empty, but `d_to_0 = np.inf` when class 0 empty. Fixed: both now use `np.inf` for empty class.
2. [FIX] `NearestNeighboursVennPredictor._predict_binary`: hardcodes `for v in (0, 1)` and `np.sum(labels)` — non-{0,1} 2-class now routes to multiclass path
3. [FIX] Single-class `label_space = [1]`: added `len(label_space) <= 1` guard → enters binary path (both VennAbersPredictor and NearestNeighboursVennPredictor)

### Pass 3 changes (3682d1e):
1. Removed dead state: `self.p`, `self.Id`, `self._kernel_spec` — replaced with local `Id` variables
2. Fixed misleading `_compute_knn_scores` docstring (title said "d_same - d_diff", body said "d_diff - d_same"; actual formula is `d_to_class0 - d_to_class1`)

### Pass 3 findings (verified correct, no changes needed):
- [OK] Sherman-Morrison hat-matrix trick (ridge scorer)
- [OK] Kernel ridge hat matrix and `_augment_kernel_state` block inverse
- [OK] OVR multiclass: all off-diagonal hypotheses share `p_off` (correct — new point's label only affects its own class score)
- [OK] Precomputed kernel state timing (augment before score computation)
- [OK] Single-class edge cases in NN Venn (hypothesise both 0 and 1)

### Design notes (deferred):
- [NOTE] `learn_one` with `precomputed=True`: caller provides kernel row/col but there is no check that the supplied `x` matches. If caller passes wrong `(x, y)` pair, state silently corrupts. This pattern is shared across regressors.py, classifiers.py, CPS.py — any precomputed path trusts the caller. **Possible future fix:** store `x` in precomputed state and raise on mismatch. Low priority — user responsibility, but easy to misuse.

### Pass 4 findings (final sweep, no changes needed):
- [OK] `_compute_knn_scores` empty same-class `d_same = 0.0` (binary path): semantically "I'm in my own class" — data-dependent score rather than -inf; doesn't affect PAVA output when entire class is empty (constant anyway)
- [OK] `_compute_knn_scores_binary` empty-class `inf` (OVR path): symmetric treatment; also irrelevant when entire binarized class is empty
- [OK] `VennPrediction.point` normalization: redundant (rows sum to 1 after multiclass normalization) but harmless safety
- [OK] Row normalization all-zero guard (`row_sums == 0 → 1.0`) — correct fallback to uniform
- [OK] Binary/multiclass taxonomy methods differ (sum-of-labels vs same-class-count): intentional, different taxonomies
- [OK] `k_eff = min(self.k, n)` consistent with `_compute_taxonomies` `if k >= n-1` check
- [OK] All imports used (`warnings` for deprecation, `Any`/`NDArray` for type hints, `cdist`/`pdist`/`squareform`)
- [OK] No remaining dead code or unused variables

## martingale.py

*(not yet audited)*

## mondrian.py

**Status:** Audited (4 passes). Clean.

### Pass 1 changes (committed):
1. Removed `if __name__ != "__main__"` import guard → standard top-level imports
2. Changed final `elif` to `else` in `predict()` and `compute_p_value()` (eliminate implicit None)
3. Removed `__main__` doctest runner at bottom
4. Added 5 edge-case tests (tiny category, no training, p-value before training,
   single category ≈ non-Mondrian, invalid category_fn TypeError)

### Pass 2 findings (no changes needed):
- [OK] `isinstance` dispatch: all three regressor types are siblings, no overlap
- [OK] Multi-epsilon threshold: `max(epsilon) < 2/n_cat` is correct guard
- [OK] `_compute_mondrian_p`: matches ALRW2 Eq. 4.31 exactly
- [OK] Lasso rank computation in `_find_intervals_mondrian`
- [OK] SVM multiclass: same-class filter before category filter (OVR justified)
- [OK] `_p_value_lasso` eq counting includes test point correctly
- [NOTE] Homotopy `v_full` staleness — inherited from base regressors.py (see above)

### Pass 3 findings (no changes needed):
- [OK] `compute_A_and_B` shape contract (X is n+1, y is n-1)
- [OK] Sherman-Morrison update formula
- [OK] Arity detection via `inspect.signature` (tested edge cases)
- [OK] Test point always last in `Alpha` array (all paths verified)
- [OK] `bounds` silently ignored for lasso (matches base class)

### Pass 4 changes:
1. Added `if model.X is None` guard in `_p_value_lasso` (crash before training)
2. Added `model.X is None` guard in `_predict_lasso` (crash before training)
3. Added test `test_lasso_p_value_before_training`

### Pass 4 findings (no changes needed):
- [OK] Binary SVM: no same-class filter (all alphas exchangeable, matches base)
- [OK] Empty category → p = τ (correct)
- [OK] State sync: learn_initial replaces, learn_one appends atomically
- [NOTE] Homotopy search range issue — inherited from base regressors.py (see above)
- [NOTE] Base lasso compute_p_value no guard — inherited (see above)

### Post-audit revisit:
Once `regressors.py` fixes are applied (v_full staleness, search range guard,
compute_p_value guard), return to mondrian.py to verify the Mondrian homotopy
path still passes and benefits from the base-class fixes.

## decision.py

**Status:** Audited. Clean (committed).

## evaluate.py

**Status:** Audited. Clean (committed).

## metrics.py

**Status:** Audited. Clean (committed).

## kernels.py

**Status:** Audited. Clean (committed).

## plotting.py

**Status:** Audited (4 passes). Clean (committed).

# Post audit check all modules holistically
Once the full audit is completed, check all modules once more. Go though the entire package, including docs, README, scripts, and notebooks (excluding those that start with "z_").
Also go through everything and check that it agrees with the book and/or paper(s), so that
everything is theoretically sound and rigorous. Finally, verify that the code is soundly structured and modular enough to be
developed and maintained with relative ease.
This is the final pre-flight check. After the full audit, the package should be ready to release version 0.3.0.