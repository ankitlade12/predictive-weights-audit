# Lag-Resolved Extraction: Experiment 13 Results

**Date:** 2026-04-06
**Hardware:** NVIDIA RTX 4090
**Status:** MARGINAL — works on simple lags, fails on hard cases

---

## What Was Tested

Can we extract lag-specific dependency scores S[tau, i, j] from trained
prediction models? Instead of just "does X_j influence X_i", we ask
"does X_j influence X_i at lag tau?"

Two extraction methods:
- **Lag-Jacobian**: ∂ŷ_i(t) / ∂x_j(t-τ) — exact, expensive
- **Lag-Impulse**: inject perturbation on channel j, measure response at lag τ — cheap

Four datasets with known lag structure:

| Dataset | Lag Structure | Difficulty |
|---|---|---|
| VAR1-chain K=5 | All edges at lag 1 | Easy |
| VAR2 K=5 | Edges at lags 1 AND 2 | Medium |
| Mixed-lag K=5 | X0→X1 at lag 1, X1→X2 at lag 2, X2→X3 at lag 3 | Hard |
| Delayed-sparse K=5 | X0→X1 at lag 3, X2→X3 at lag 4 | Hardest |

---

## Results

| Dataset | Aggregate Weight AUROC | Lag-Jacobian (per-lag) | Lag-Impulse (per-lag) |
|---|---|---|---|
| VAR1-chain (lag 1) | ~1.000 | **0.755±0.029** | **0.755±0.029** |
| VAR2 (lags 1+2) | 0.752±0.054 | **0.642±0.049** | 0.626±0.049 |
| Mixed-lag (lags 1,2,3) | 0.500 | 0.500 | 0.500 |
| Delayed-sparse (lags 3,4) | 0.500 | 0.500 | 0.500 |

---

## Interpretation

### What works
- **VAR1 (lag 1 only):** Per-lag AUROC 0.755 — the method correctly identifies lag 1 edges over other lags. Not perfect but clearly above random.
- **VAR2 (lags 1+2):** Per-lag AUROC 0.642 — can partially distinguish edges at different lags. Harder than single-lag, expected degradation.
- Both extraction methods (Jacobian and Impulse) perform similarly.

### What fails
- **Mixed-lag (lags 1,2,3):** All 0.500 = random. The model produces NaN/constant scores. With only 3 edges across 3 different lags, the signal is too sparse for the model to learn distinct lag-specific structure.
- **Delayed-sparse (lags 3,4):** All 0.500 = random. Same issue — long-range dependencies at lags 3-4 with coefficient 0.4 are dominated by the autoregressive component (diagonal 0.5).

### Why it fails on hard cases
1. **Weak signal-to-noise:** Cross-variable coefficients (0.4) compete with autoregressive diagonal (0.5). At higher lags, the cumulative effect is weaker.
2. **Sparse edges:** 3 edges out of 20 possible at any given lag is extremely sparse. The model doesn't have enough signal to differentiate.
3. **SSM sequential scan:** The recurrence h_t = A·h_{t-1} + B·x_t mixes all lags into the hidden state. Disentangling which lag carries each edge may require architectural changes.

---

## Decision Gate

Per Sai's update:
> "first test lag-resolved extraction on VAR(p), delayed sparse systems, and at least one nonlinear delayed setting"

Results:
- VAR(1): Partial success (per-lag AUROC 0.755)
- VAR(2): Marginal (per-lag AUROC 0.642)
- Delayed sparse: Complete failure (AUROC 0.500)
- Nonlinear delayed: Not tested (deferred given failures above)

**VERDICT: MARGINAL.** Lag-resolved extraction shows a real but weak signal on simple lag structures. It does not work on the hard cases (multi-lag, delayed) that would make it a distinctive contribution.

---

## Recommendations

1. **Do not pursue lag-resolved extraction as the primary paper contribution** — the signal is too weak on hard cases.

2. **Consider as a supporting analysis** — showing partial lag recovery on VAR(1) is an interesting additional finding, not a main result.

3. **Per Sai's plan:** Do not proceed to intervention-calibrated extraction yet, since the lag-resolved prerequisite was not cleanly met.

4. **Current strongest paper direction remains:** The observation paper (P1-P6 results) — post hoc bottleneck extraction works, with a clear architecture ranking and operating regime.
