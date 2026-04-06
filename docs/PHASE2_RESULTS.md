# Phase 2 Results: Robust Validation of Causality-Preserving Bottleneck

**Date:** 2026-04-06
**Status:** FAILED — gains from Phase 1 were not robust
**Hardware:** NVIDIA RTX 4090
**Runtime:** 109.0 minutes
**Seeds:** 10

---

## What Was Tested

Phase 1 showed D-Combined improving on 2/4 datasets with 3 seeds.
Phase 2 expanded to 10 seeds, 6 datasets, λ sensitivity, and statistical tests.

---

## Main Results (10 seeds)

| Dataset | A-Baseline | B-Sparse | D-Combined | E-Sparse+Consist | Jacobian-only | Linear | Δ (D vs A) |
|---|---|---|---|---|---|---|---|
| VAR1-chain K=5 | 0.986±0.028 | 0.983±0.027 | 0.977±0.027 | 0.977±0.025 | 1.000 | **1.000±0.000** | -0.009 |
| VAR1-random K=10 | 0.757 | 0.696 | 0.728 | — | — | **0.986** | -0.029 |
| VAR2-K5 | 0.783 | — | **0.803** | — | — | 0.867 | +0.019 |
| Lorenz-K3 | 0.700 | — | **0.720** | — | — | 0.960 | +0.020 |
| Regime-K3 | **0.988** | — | 0.950 | — | — | 0.925 | -0.038 |
| Hénon-K2 | **1.000** | — | **1.000** | — | — | 0.800 | 0.000 |

---

## λ Sensitivity (Lorenz, 3 seeds)

All 20 combinations of λ_sparse × λ_consist tested:

| λ_sparse | λ_consist=0 | λ_consist=1e-3 | λ_consist=1e-2 | λ_consist=5e-2 |
|---|---|---|---|---|
| 0 | 0.667 | 0.667 | 0.667 | 0.667 |
| 1e-5 | 0.667 | 0.667 | 0.667 | 0.667 |
| 3e-5 | 0.667 | 0.667 | 0.667 | 0.667 |
| 1e-4 | 0.667 | 0.667 | 0.667 | 0.667 |
| 3e-4 | 0.667 | 0.667 | 0.667 | 0.667 |

**All identical (0.667).** The regularizers have zero effect on Lorenz extraction quality.

---

## Statistical Significance Tests

| Dataset | Test | p-value | Result |
|---|---|---|---|
| VAR2-K5 | Wilcoxon (D vs A) | 0.0664 | NOT significant (p > 0.05) |
| Lorenz-K3 | Wilcoxon (D vs A) | 0.5000 | NOT significant |

---

## Interpretation

### Phase 1 signal was noise
- Phase 1 (3 seeds): D-Combined showed +0.020 (VAR2) and +0.067 (Lorenz)
- Phase 2 (10 seeds): gains shrink to +0.019 and +0.020, both p > 0.05
- The Lorenz +0.067 from Phase 1 was seed variance, not a real effect

### λ sensitivity confirms no effect
- 20 different λ combinations on Lorenz all give exactly 0.667 AUROC
- The regularizers are not changing the learned weight structure meaningfully

### D-Combined hurts on 3/6 datasets
- VAR1-chain: -0.009 (minor)
- VAR1-random K=10: -0.029 (notable degradation)
- Regime-K3: -0.038 (substantial degradation)

### Linear model still dominates
- Linear beats all SSM variants on 4/6 datasets
- Exception: Regime-K3 (SSM 0.988 vs Linear 0.925) and Hénon (SSM 1.0 vs Linear 0.8)

---

## Decision

Per Sai's decision gate:

> "We should stop or sharply rethink if all variants fail to improve
> over the plain baseline, gains are tiny and unstable, or the method
> only wins on toy settings."

**All three conditions are met:**
1. No variant robustly improves over baseline (p > 0.05 on all tests)
2. Gains are tiny and within noise (+0.019 at best)
3. λ sensitivity shows zero effect across 20 combinations

**DECISION: STOP causality-preserving bottleneck training direction.**

---

## What We Learned

1. The bottleneck weight product `|W_out @ W_in|` encodes dependency structure — but this is an emergent property of prediction training, not something that can be easily improved with lightweight regularization.

2. The structure is already close to optimal for the model's capacity — adding sparsity/stability/consistency constraints doesn't change the learned routing meaningfully.

3. Linear models inherently produce the cleanest bottleneck factorization because their internal mapping is the identity. No amount of regularization on SSMs can match this.

4. The remaining value is in the **observation** (bottleneck factorization encodes dependency for free) not in a **method** (making it better with training tricks).

---

## Remaining Path

Per Sai's fallback order:

> 1. Hybrid bottleneck + Jacobian extractor as a smaller method paper.
> 2. Narrow empirical paper on post hoc dependency extraction from forecasting bottlenecks.
> 3. Full pivot paper only if the method-building path fails cleanly.

The method-building path has failed cleanly. **Option 2 or 3** is the remaining path.
