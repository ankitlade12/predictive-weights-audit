# Exp 24 + 25: Real Data Benchmark and Causal Baselines

**Date:** 2026-04-16
**Hardware:** NVIDIA RTX 4090
**Purpose:** Fill the two gaps Sai identified before ICBINB submission —
(1) real observational data, (2) proper causal discovery baselines.

---

## Exp 24: Real Data Benchmark

**Real datasets used (no synthetic):**
- **NOAA Climate Indices** — 757 months (1962–2024) of actual monthly
  ENSO, NAO, PDO, AMO, SOI, PNA indices from NOAA PSL.
  Ground truth: 6 literature-derived teleconnection edges (Runge et al. 2019).
- **Sector ETFs** — 1,893 trading days (2018–2025) of real log returns
  for 10 SPDR sector ETFs (XLE, XLB, XLI, XLY, XLP, XLF, XLK, XLC, XLU, XLRE)
  downloaded from Yahoo Finance.
  Ground truth: 6 sector lead-lag relationships from finance literature.

### Results

| Method | Climate (K=6, T=757) | Finance (K=10, T=1893) |
|---|---|---|
| **Granger** | **0.819** | 0.498 |
| **Lagged-Bottleneck** | 0.813 | 0.552 |
| Tuned Lasso | 0.799 | 0.500 |
| Weight-Proj (SSM) | 0.792 | 0.631 |
| PCMCI | 0.771 | 0.510 |
| Tuned Ridge | 0.729 | **0.696** |

### Interpretation

**Climate:** All methods cluster tightly (0.73–0.82 AUROC). Granger and
Lagged-Bottleneck tie at the top. No method clearly dominates —
real observational climate data admits many valid recovery strategies.

**Finance:** All methods perform weakly (0.50–0.70 AUROC). Real stock
returns are noisier and harder to attribute. Ridge does best; Weight-Proj
(SSM) beats Lasso, PCMCI, and Granger; Granger fails (0.50 = random).

**Headline finding:** On real data, the learned bottleneck is **competitive
but not superior** to classical causal baselines (PCMCI, Granger). It
neither dominates nor collapses — it sits in the same performance band
as Lasso/Ridge/PCMCI. This aligns with the full characterization story
that emerged from Exps 18–23.

---

## Exp 25: Causal Baselines on Intervention Control Setup

Mirrors Exp 22's size-matched intervention setup exactly (3 arms:
`X_obs_small`, `X_obs_big`, `X_combined`), adds PCMCI and Granger as
actual causal discovery baselines alongside Bottleneck and Lasso.

### Size-Matched Intervention Gap (Wilcoxon p-values)

| Config | Bottleneck | Lasso | PCMCI | Granger |
|---|---|---|---|---|
| K=10, T=300 | **+0.026 (p=9.8e-04)** ✓ | −0.139 (p=0.81) | +0.025 (p=0.42) | +0.021 (p=0.05) |
| K=20, T=300 | **+0.027 (p=9.8e-04)** ✓ | −0.012 (p=0.99) | +0.007 (p=0.42) | **+0.040 (p=9.8e-04)** ✓ |
| K=30, T=300 | **+0.054 (p=9.8e-04)** ✓ | −0.001 (p=0.78) | +0.022 (p=0.01) ✓ | **+0.095 (p=9.8e-04)** ✓ |

(Columns show size-matched gap = AUROC(combined) − AUROC(obs_big))

### Interpretation

1. **Bottleneck DOES show a genuine size-matched intervention gain** on
   this data distribution (+0.026 to +0.054 AUROC, p < 0.001 on all
   three K values). Sai's earlier Exp 22 reported ~+0.03–0.05; this
   replicates that.

2. **But Granger shows an EVEN LARGER effect** at K=20 (+0.040) and
   K=30 (+0.095, also p<0.001). **The "intervention advantage" is not
   unique to the bottleneck.** Bivariate Granger — a method that makes
   no low-rank bottleneck assumption — benefits equally or more.

3. **Lasso uniquely does NOT benefit** from the random-forcing
   intervention. At K=10 it actually hurts (−0.139). This is consistent
   with Sai's target-corruption hypothesis: per-step random forcing
   clamps the target row, which breaks Lasso's per-equation regression
   but helps methods that aggregate information across variables
   (Bottleneck, PCMCI, Granger).

4. **PCMCI is very weak on this data** (near-random on all arms). Its
   raw AUROC hovers around 0.50, so small gains are noisy and mostly
   non-significant.

### What This Confirms

Sai's Exp 22–23 conclusion is right and now has stronger evidence:

- The Exp 21 "bottleneck benefits uniquely from interventions" claim is
  **partially dead**: the bottleneck's size-matched gap is real and
  replicable, but so is Granger's, and Granger's is usually larger.
- The residual effect (after controlling for sample size) is
  **method-agnostic for methods that aggregate across variables**, and
  is consistent with a target-corruption robustness mechanism rather
  than a specific causal-structure signal.
- Lasso's per-equation fit is the one method that is genuinely hurt by
  random forcing.

---

## Combined Verdict for ICBINB

This is a clean ICBINB story. The full arc:

1. **Initial excitement:** `|W_out @ W_in|` recovers dependency
   structure without causal training (P2 emergence result, Exp 21
   headline numbers).
2. **Falsification chain:**
   - Linear baseline beats SSM at same extraction (P1, go/no-go).
   - Tuned Lasso beats Bottleneck on graph recovery (Exp 15–17).
   - Exp 21's intervention finding was ~60% sample-size confound (Exp 22).
   - Residual effect is specific to random-forcing interventions, not
     proper do(X=c) interventions (Exp 23).
   - Granger benefits equally or more from the same intervention scheme
     (Exp 25). The effect is not bottleneck-specific.
3. **On real data:** Bottleneck is competitive but not superior to
   classical baselines on both NOAA climate and Yahoo Finance sector
   ETFs (Exp 24).
4. **What survives as characterization:**
   - Bottlenecks encode readable dependency structure.
   - Narrow advantage in mild-nonlinearity regime (Exp 18).
   - Sample-efficient relative to Lasso as T grows (Exp 22).
   - Robust to target-corruption patterns (Exp 23, 25).

Paper framing: *"We wanted a free causal discovery method from
prediction training. We got a method-agnostic robustness-to-corruption
effect. Here is what we actually learned."*
