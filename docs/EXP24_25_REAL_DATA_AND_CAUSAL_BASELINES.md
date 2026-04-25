# Exp 24 + 25: Real Data Benchmark and Causal Baselines

**Dates:** 2026-04-16 (initial), 2026-04-17 (updated with revised ground truth and Lorenz-96)
**Purpose:** Fill the two gaps Sai identified before workshop/journal
submission — (1) real observational data with clean ground truth,
(2) proper causal discovery baselines.

---

## Revision note (2026-04-17)

The 2026-04-16 version of this document reported climate results that
included the ENSO$\leftrightarrow$SOI pair in ground truth. Those two
indices are the same signal with opposite sign (the Southern Oscillation
Index is a surface-pressure analog of the Niño 3.4 SST anomaly). Counting
them as causal edges inflates every method's AUROC on an
instantaneous-coupling artifact. The revised ground truth keeps three
unambiguous literature edges: ENSO→PNA, ENSO→PDO, NAO→AMO.

The revision **flipped the climate ranking**: bivariate Granger went from
the top (0.819) to the bottom (0.605) once the ENSO$\leftrightarrow$SOI
edges were removed. Several other methods moved too. This is itself a
useful methodological finding we report in the paper.

We also added Lorenz-96 ($K=10$, $F=10$, RK4 integration) as a third
dataset. Lorenz-96 has unambiguous ground truth (each variable is driven
by exactly three neighbors) and is the standard causal-discovery
benchmark used by Tank et al.~(2022) and Nauta et al.~(2019).

---

## Exp 24: Real Data Benchmark (revised)

**Datasets used:**

- **NOAA Climate Indices** — 757 months (1962–2024) of real monthly
  ENSO, NAO, PDO, AMO, SOI, PNA indices from NOAA PSL. Ground truth:
  3 unambiguous literature-derived edges (ENSO→PNA, ENSO→PDO,
  NAO→AMO), ENSO$\leftrightarrow$SOI excluded as definitional.
- **Sector ETFs** — 1,893 trading days (2018-06-19 to 2025-12-30) of
  real log returns for 10 SPDR sector ETFs downloaded via yfinance.
  Ground truth: 6 sector lead-lag edges from finance commentary
  (treated as soft).
- **Lorenz-96** — $K=10$, $F=10$, RK4 at $dt=0.01$, subsampled every
  10 steps, $T=1500$. Ground truth: each node $i$ driven by $\{i-2, i-1,
  i+1\}$ (mod $K$), 30 edges total.

### Results (AUROC, 40-epoch SSM, deterministic non-SSM methods)

| Method | Climate (K=6, T=757, 3 edges) | Finance (K=10, T=1893, 6 edges) | Lorenz-96 (K=10, T=1500, 30 edges) |
|---|---|---|---|
| Weight-Proj (SSM) | **0.864** | 0.438 | 0.802 |
| PCMCI | 0.704 | 0.510 | **0.986** |
| Tuned Lasso | 0.630 | 0.500 | 0.974 |
| Lagged Bottleneck | 0.617 | 0.514 | 0.920 |
| Tuned Ridge | 0.617 | **0.696** | 0.974 |
| Bivariate Granger | 0.605 | 0.498 | 0.979 |

### Interpretation

1. **Climate with honest ground truth**: all methods sit in a narrow
   band (0.60-0.86 AUROC) and rankings are unstable because there
   are only 3 positive edges over $K=6$. The SSM bottleneck tops the
   table but we do not read this as a causal-discovery claim; it is
   likely sensitive to the particular three edges we chose.
2. **Finance**: Ridge wins (0.696); most methods are at or below
   chance. ETF daily returns admit no clean lagged causal structure,
   consistent with efficient-markets behavior at the daily scale.
3. **Lorenz-96 with clean ground truth**: classical causal methods
   (PCMCI 0.986, Granger 0.979) dominate; tuned Lasso/Ridge nearly
   match them (0.974); the lagged bottleneck trails at 0.920 and the
   SSM bottleneck is worst at 0.802. **On the only dataset where
   ground truth is unambiguous, the bottleneck is not competitive.**

### Effect of the 2026-04-16 → 2026-04-17 ground-truth correction on climate

| Method | 2026-04-16 (6 edges incl. ENSO$\leftrightarrow$SOI) | 2026-04-17 (3 edges, honest) |
|---|---|---|
| Bivariate Granger | 0.819 (top) | 0.605 (bottom) |
| Lagged Bottleneck | 0.813 | 0.617 |
| Tuned Lasso | 0.799 | 0.630 |
| Weight-Proj (SSM) | 0.792 | 0.864 (top) |
| PCMCI | 0.771 | 0.704 |
| Tuned Ridge | 0.729 | 0.617 |

The ordering changed substantially. Granger dropped from first to
last, which tells us Granger was largely picking up the instantaneous
ENSO/SOI coupling. The SSM rose to first, but with only three positive
edges this is a fragile ranking. The methodological takeaway is that
literature ground-truth choices on soft-ground-truth datasets can
reverse conclusions — a reviewer-relevant point the paper now makes
directly.

---

## Exp 25: Causal Baselines on Intervention Control Setup

Mirrors Exp 22's size-matched intervention setup exactly (3 arms:
`X_obs_small`, `X_obs_big`, `X_combined`), adds PCMCI and Granger as
actual causal discovery baselines alongside Bottleneck and Lasso.

### Size-Matched Intervention Gap (Wilcoxon p-values)

| Config | Bottleneck | Lasso | PCMCI | Granger |
|---|---|---|---|---|
| K=10, T=300 | **+0.026 (p=9.8e-04)** | −0.139 (p=0.81) | +0.025 (p=0.42) | +0.021 (p=0.05) |
| K=20, T=300 | **+0.027 (p=9.8e-04)** | −0.012 (p=0.99) | +0.007 (p=0.42) | **+0.040 (p=9.8e-04)** |
| K=30, T=300 | **+0.054 (p=9.8e-04)** | −0.001 (p=0.78) | +0.022 (p=0.01) | **+0.095 (p=9.8e-04)** |

(Columns show size-matched gap = AUROC(combined) − AUROC(obs_big))

### Interpretation

1. **Bottleneck DOES show a genuine size-matched intervention gain**
   on this data distribution (+0.026 to +0.054 AUROC, p < 0.001 on
   all three K values). Sai's Exp 22 reported ~+0.03–0.05; this
   replicates that.
2. **But Granger shows an EVEN LARGER effect** at K=20 (+0.040) and
   K=30 (+0.095, also p<0.001). **The "intervention advantage" is
   not unique to the bottleneck.** Bivariate Granger — a method that
   makes no low-rank bottleneck assumption — benefits equally or more.
3. **Lasso uniquely does NOT benefit** from the random-forcing
   intervention. At K=10 it actually hurts (−0.139). This is consistent
   with the target-corruption hypothesis: per-step random forcing
   clamps the target row, breaking Lasso's per-equation regression
   but helping methods that aggregate information across variables
   (Bottleneck, PCMCI, Granger).
4. **PCMCI is weak on this data** (near-random on all arms). Its raw
   AUROC hovers around 0.50, so small gains are noisy and mostly
   non-significant.

### What This Confirms

- The Exp 21 "bottleneck benefits uniquely from interventions" claim
  is dead: the bottleneck's size-matched gap is real and replicable,
  but so is Granger's, and Granger's is usually larger.
- The residual effect (after controlling for sample size) is
  method-agnostic for methods that aggregate across variables, and is
  consistent with a target-corruption robustness mechanism rather
  than a specific causal-structure signal.
- Lasso's per-equation fit is the one method that is genuinely hurt
  by random forcing.

---

## Combined Verdict for the Paper

The updated evidence strengthens rather than weakens the negative-results
story. The full arc:

1. **Initial excitement:** `|W_out @ W_in|` recovers dependency
   structure without causal training.
2. **Falsification chain:**
   - Linear baseline beats SSM at same extraction.
   - Tuned Lasso beats Bottleneck on graph recovery (synthetic
     stress tests).
   - On Lorenz-96 (clean ground truth), classical causal methods
     (PCMCI, Granger) dominate; Bottleneck trails.
   - Exp 21's intervention finding was ~60% sample-size confound.
   - Residual effect is specific to random-forcing interventions,
     not proper do(X=c).
   - Granger benefits equally or more from the same intervention
     scheme. The effect is not bottleneck-specific.
   - Climate ranking is unstable under literature ground-truth
     choices, and soft ground truth can reverse conclusions.
3. **What survives as characterization:**
   - Bottlenecks encode readable dependency structure.
   - Narrow advantage in mild-nonlinearity regime.
   - Sample-efficient relative to Lasso as T grows.
   - Robust to target-corruption patterns.

Paper framing: *"We wanted a free causal discovery method from
prediction training. We got a method-agnostic robustness-to-corruption
effect and a rankings-sensitivity-to-ground-truth lesson. Here is
what we actually learned."*
