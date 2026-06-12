# Evaluation

The evaluation module provides River-style test-then-train loops for online
conformal predictors. All four functions accept a `delay` parameter that
implements Vovk's **Weak Teacher** and **Lazy Teacher** protocols from
*Algorithmic Learning in a Random World* (2nd ed., §3.3).

## Teaching-schedule semantics

A *teaching schedule* $\mathcal{L} : N \to \mathbb{N}$ describes when the
predictor receives the true label $y_n$. Three canonical cases are supported:

| Mode | How to invoke | ALRW2 example |
|------|--------------|---------------|
| **Ideal teacher** | `delay=0` (default) | $\mathcal{L}(n)=n$ |
| **Slow / weak teacher** | `delay=l` (int) or `delay=callable(step,x,y)→int` | Fixed lag $l$, or variable lag |
| **Lazy teacher** | Yield `y = None` in the stream | $N \subsetneq \mathbb{N}$, immediate feedback for $n \in N$ |

## Validity guarantees

Asymptotic validity is preserved for any *invariant* conformal predictor
(predictions are independent of the order of training examples — every
predictor in this package satisfies this):

| Theorem | Condition | Guarantee |
|---------|-----------|-----------|
| Thm 3.7 / Cor 3.8 | $\lim_k n_k/n_{k-1} = 1$ (sub-exponential feedback gaps) | $\mathrm{Err}_n^\varepsilon/n \to \varepsilon$ in probability |
| Thm 3.9 / Cor 3.10 | $\sum_k (n_k/n_{k-1}-1)^2 < \infty$ | a.s. convergence |
| Thm 3.11 | $n_k = O(k)$ (equally spaced, e.g. fixed lag) | $\lvert\mathrm{Err}_n^\varepsilon/n - \varepsilon\rvert = O\!\left(\sqrt{\ln\ln n/n}\right)$ a.s. |

A fixed `delay=l` gives $n_k = O(k)$ and therefore satisfies the strongest
(LIL) guarantee. A lazy teacher is valid whenever feedback arrives at more
than a logarithmic fraction of trials.

---

::: online_cp.evaluate.progressive_val

::: online_cp.evaluate.iter_progressive_val

## Venn Predictor Evaluation

::: online_cp.evaluate.progressive_val_venn

::: online_cp.evaluate.iter_progressive_val_venn
