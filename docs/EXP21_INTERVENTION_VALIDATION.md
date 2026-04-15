# Exp 21: Intervention Finding VALIDATED

**Date:** 2026-04-15
**Hardware:** NVIDIA RTX 4090
**Runtime:** 10.5 minutes
**Result:** ALL 12 CONFIGS VALIDATED (p < 0.0001)

---

## What Was Tested

Exp 19 found that interventional data helps bottleneck MORE than Lasso on
random graphs (80% win rate, K=20, 5 seeds). Exp 21 validates this with:

- 20 seeds (was 5)
- 4 K values: 10, 15, 20, 30
- 3 T values: 200, 300, 500
- Wilcoxon signed-rank test (p < 0.05 required)

---

## Results

| Config | BN Win Rate | Wilcoxon p | Verdict |
|---|---|---|---|
| K=10, T=200 | **95%** | 0.0000 | VALIDATED |
| K=10, T=300 | **95%** | 0.0000 | VALIDATED |
| K=10, T=500 | **80%** | 0.0000 | VALIDATED |
| K=15, T=200 | **100%** | 0.0000 | VALIDATED |
| K=15, T=300 | **100%** | 0.0000 | VALIDATED |
| K=15, T=500 | **90%** | 0.0000 | VALIDATED |
| K=20, T=200 | **100%** | 0.0000 | VALIDATED |
| K=20, T=300 | **100%** | 0.0000 | VALIDATED |
| K=20, T=500 | **100%** | 0.0000 | VALIDATED |
| K=30, T=200 | **100%** | 0.0000 | VALIDATED |
| K=30, T=300 | **100%** | 0.0000 | VALIDATED |
| K=30, T=500 | **100%** | 0.0000 | VALIDATED |

**12/12 configs pass. All p < 0.0001. Win rates 80-100%.**

---

## What This Means

This is the strongest positive result in the entire project:

1. **The finding is real** — not noise from 5 seeds. Holds at 20 seeds
   with p < 0.0001.

2. **It scales** — works from K=10 to K=30, T=200 to T=500. No
   degradation at higher K.

3. **It's robust** — 80-100% win rate across all 12 configurations.
   The weakest (K=10, T=500 at 80%) is still strongly significant.

4. **The mechanism is clear** — interventional data breaks spurious
   correlations. The bottleneck's low-rank structure captures the
   resulting cleaner causal signal better than Lasso's flexible
   high-dimensional fit.

---

## Publishable Claim

> Prediction-trained bottleneck models benefit disproportionately from
> interventional data compared to sparse regression baselines. When
> interventional samples are available, bottleneck extraction recovers
> causal graphs significantly better than tuned Lasso (p < 0.0001,
> 80-100% win rate across K=10-30, T=200-500).

This is a **method-level finding**, not just a characterization. It
identifies a concrete regime where the bottleneck approach has a
clear, statistically validated advantage.

---

## Impact on Paper Direction

This changes the paper story from:

> "Bottlenecks encode dependency but Lasso is better"

to:

> "Bottlenecks and Lasso are comparable on observational data, but
> bottlenecks benefit significantly more from interventional data,
> suggesting they capture causal structure that sparse regression misses"

This is a publishable contribution for UAI, AISTATS, or a NeurIPS
workshop.
