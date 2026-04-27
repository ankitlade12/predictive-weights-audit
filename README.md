# SSM-Causal: A Falsification of Prediction Bottlenecks for Causal Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Paper:** *Prediction Bottlenecks Don't Discover Causal Structure (But Here's What They Actually Do)*
> **arXiv:** `<arXiv-ID-pending>`
> **Code:** [github.com/ankitlade12/ssm-causal](https://github.com/ankitlade12/ssm-causal)

**Authors:** Ankit Hemant Lade, Sai Krishna Jasti, Indar Kumar, Aman Chadha

## Table of Contents

1. [Overview](#1-overview)
2. [The Hypothesis Being Falsified](#2-the-hypothesis-being-falsified)
3. [The Five Falsifications](#3-the-five-falsifications)
4. [Method: The Extraction Under Test](#4-method-the-extraction-under-test)
5. [Baselines](#5-baselines)
6. [Datasets](#6-datasets)
7. [Experimental Design](#7-experimental-design)
8. [Results](#8-results)
9. [What Survives](#9-what-survives)
10. [Methodological Lessons](#10-methodological-lessons)
11. [Reproduction](#11-reproduction)
12. [Project Structure](#12-project-structure)
13. [References](#13-references)

---

## 1. Overview

A natural hope in time-series modelling is that a forecaster trained only for next-step prediction will, as a byproduct, expose causal or dependency structure through its learned weights. For a Mamba selective state-space model trained on next-step MSE, the product `S = |W_out · W_in|` initially appeared to recover Granger-causal graphs with high reliability — across architectures, with `p < 10⁻⁵` significance under interventional data.

**This paper takes that hope through a five-stage falsification pipeline. The method-level claim does not survive.** The contribution is primarily *negative*: we identify five subtly different ways the candidate fails, document the controls that caused each failure, and report the narrow characterization findings that remain valid.

**Headline result (5 falsification stages, 3 real datasets, 6 baselines, multi-seed):**

- **Architecture is irrelevant.** A plain linear bottleneck matches or beats the SSM under matched capacity and seeds.
- **Tuned Lasso wins on synthetic graph recovery.** The bottleneck wins only **12% (6/48)** of cells against tuned `{OLS, Ridge, Lasso, RRR}`.
- **Classical methods dominate the only clean real benchmark.** On Lorenz-96, PCMCI (**0.986**) and bivariate Granger (**0.979**) lead; the SSM bottleneck trails at **0.722 ± 0.026**.
- **The headline intervention advantage is a confound.** Roughly 60% data-size, ~40% target-corruption artifact specific to a non-standard random-forcing scheme. Under proper `do(X=c)` interventions, the effect collapses.
- **The residual is method-agnostic.** Bivariate Granger reproduces the same effect with *larger* effect sizes at high K (+0.095 at K=30 vs +0.054 for the bottleneck).

**What survives** is a narrow characterization result, not a method claim: a mild-nonlinearity configuration, a sample-efficiency advantage, and target-corruption robustness — released alongside a reusable falsification scaffold.

---

## 2. The Hypothesis Being Falsified

For a `K`-variable multivariate time series `X ∈ ℝ^{T×K}`, train a model with input projection `W_in ∈ ℝ^{d×K}` and output projection `W_out ∈ ℝ^{K×d}` on next-step MSE. **No causal-specific loss.** After training, extract:

```
S = |W_out · W_in| ∈ ℝ^{K×K}
S[i,j] = strength of edge j → i
```

Zero the diagonal, normalize, threshold. The original claim was that **this single matrix product recovers Granger-causal structure across architectures, regimes, and intervention conditions** — for free, as a byproduct of strong forecasting.

This is the entire method candidate. Each of the five sections below disproves a different component of that claim.

---

## 3. The Five Falsifications

| # | Claim falsified | Control | Status |
|---|---|---|---|
| **F1** | The effect is specific to selective state-space models. | 10-seed matched-capacity sweep, 5 architectures × 6 datasets. | **Falsified.** Linear bottleneck matches or beats SSM on every dataset. |
| **F2** | The bottleneck is competitive with sparse regression. | Tuned `{OLS, Ridge, Lasso, RRR}` on 48-cell stress grid + tigramite CauseMe-style processes. | **Falsified.** Bottleneck wins 12% of graph runs and 0% of MSE runs. |
| **F3** | The bottleneck recovers structure on real data. | NOAA climate, sector ETFs, Lorenz-96 with literature-based ground truth. | **Falsified.** Worst on Lorenz-96 (the only benchmark with unambiguous ground truth). |
| **F4** | Interventional data amplifies the bottleneck's advantage. | Size-matched control + proper `do(X=c)` interventions. | **Falsified.** Effect is ~60% sample-size confound, ~40% random-forcing artifact. |
| **F5** | The residual intervention effect is unique to neural bottlenecks. | Add classical PCMCI and bivariate Granger to the same control protocol. | **Falsified.** Granger shows the same effect at *larger* size. |

The five stages are sequential — each control was added in response to a previous result that initially looked positive.

---

## 4. Method: The Extraction Under Test

### 4.1 The Extraction

```
Input: X ∈ ℝ^{T×K}  (multivariate time series)
  │
  ▼
Train Mamba on next-step prediction: min ||M(X_t) - X_{t+1}||²
  │
  ▼
Extract W_in ∈ ℝ^{d×K} and W_out ∈ ℝ^{K×d}
  │
  ▼
Compute S = |W_out · W_in| ∈ ℝ^{K×K}  ← single matrix multiply
  │
  ▼
Output: Causal score matrix Ĝ (threshold S)
```

For architectures with explicit lagged input (the lagged bottleneck variant), we extend to `S^(τ) = |W_out · W_in^(τ)|` and evaluate via flat-lag AUROC against ground-truth adjacency.

### 4.2 Architectures Compared (F1)

| Architecture | Notes |
|---|---|
| **Linear** | Plain `ŷ = W_out · W_in · x`. Provides the same `S` extraction by construction. |
| **Mamba SSM** | Selective state-space, 4 layers, `d_model=64`, `d_state=16`. |
| **Transformer** | Single block, `d_model=64`, 4 heads. |
| **LSTM** | Single layer, hidden=64. |
| **MLP** | 2 hidden layers, width=128. |

All five share input/output projection geometry, are matched in parameter budget where feasible, and are trained with the same optimizer, schedule, and seeds.

---

## 5. Baselines

The baselines change across the five falsifications because each stage tests a different competitor.

### 5.1 F2 — Sparse Regression (synthetic)

| Method | Implementation | Tuned? |
|---|---|---|
| OLS (lagged regression) | `numpy.linalg.lstsq` per equation | n/a |
| Ridge | `sklearn.linear_model.Ridge`, `α` via 5-fold CV | yes |
| Lasso | `sklearn.linear_model.Lasso`, `α` via 5-fold CV | yes |
| Reduced-Rank Regression (RRR) | analytic, rank via held-out validation | yes |

### 5.2 F3 — Real-Data Causal Discovery

| Method | Implementation | Notes |
|---|---|---|
| Bivariate Granger | `statsmodels.tsa.stattools.grangercausalitytests` | Pairwise lag-`L` F-test |
| PCMCI | `tigramite.PCMCI` with `ParCorr` test | Conditional independence testing |
| Tuned Lasso | as F2 | Group-Lasso variant for multi-lag |
| Tuned Ridge | as F2 | |
| Lagged Bottleneck | our method, lag-indexed `S^(τ)` | Single seed (deterministic except training) |
| Weight-Proj (SSM) | our method, Mamba | 5 seeds, mean ± std reported |

### 5.3 F5 — Intervention Control Suite

Same six methods as F3 (PCMCI, Granger, Lasso, Ridge, Bottleneck variants), evaluated on the size-matched intervention protocol from F4 to test method-agnosticity.

---

## 6. Datasets

### 6.1 Synthetic (F1, F2, F4, F5)

| Generator | Params | Purpose |
|---|---|---|
| **VAR(1) chain** | `K=5`, sparse linear | F1: clean signal |
| **VAR(1) random** | `K=10`, dense linear | F1: capacity test |
| **Regime-switch** | `K=3`, 2-state Markov | F1: non-stationarity |
| **Lorenz-3** | `K=3`, RK4 integration | F1: nonlinear chaos |
| **VAR(p) stress grid** | `K∈{10,20}`, `T∈{150,300}`, max-lag 8 | F2: 48-cell sweep |
| **Tigramite CauseMe-style** | `K∈{10,20,30}`, structural processes | F2: realistic synthetic |
| **VAR with intervention arms** | configurable size + intervention type | F4, F5: control suite |

### 6.2 Real (F3)

| Dataset | K | T | Source | Ground Truth |
|---|---:|---:|---|---|
| **NOAA climate indices** | 6 | 757 | [NOAA PSL](https://psl.noaa.gov/), monthly 1962–2024 | 3 teleconnection edges from climate-dynamics literature |
| **SPDR sector ETFs** | 10 | 1893 | yfinance, daily log returns 2018-06–2025-12 | 6 sector lead-lag edges (soft labels) |
| **Lorenz-96** | 10 | 1500 | RK4 integration, `F=10` | Each variable driven by 2 left + 1 right neighbour (mathematically exact) |

Climate indices: ENSO (Niño 3.4), NAO, PDO, AMO, SOI, PNA. Edges:
- ENSO → PNA (Trenberth & Hurrell, 1994)
- ENSO → PDO (Newman et al., 2016)
- NAO → AMO (Delworth & Mann, 2000)

We **deliberately exclude** the ENSO↔SOI pair — SOI and Niño 3.4 SST anomaly are the same physical signal with opposite sign, not a causal edge. Including them inflates every method's AUROC and reorders the entire ranking (see Table 3 below).

---

## 7. Experimental Design

### 7.1 Scale

| Stage | Models × seeds × configs | Total runs |
|---|---|---|
| F1 architecture sweep | 5 archs × 10 seeds × 6 datasets | 300 |
| F2 stress grid | 5 methods × 4 (K,T) × 12 hyperparam cells | 240 |
| F2 CauseMe-style | 5 methods × 3 K × 5 seeds × 4 graph types | 300 |
| F3 real data | 6 methods × 3 datasets × {1, 5} seeds | ~50 |
| F4 intervention control | 2 methods × 3 K × 12 seeds × 3 intervention types | 216 |
| F5 method-agnosticity | 5 methods × 3 K × 10 seeds × size-matched arms | 450 |

### 7.2 Fairness Guarantees

| Guarantee | Mechanism |
|---|---|
| Same data | Identical train/test splits, identical synthetic seeds |
| Same evaluation | Flat-lag AUROC against ground-truth adjacency tensor |
| Same intervention semantics | Three explicitly-distinguished variants — `do(X=c)`, soft (added-noise), random-forcing — evaluated on the same data and same control arms (see Lessons §10.2 for why we report three variants rather than one) |
| Matched capacity | Architecture-specific parameter counts logged; F1 reports matched runs |
| Multi-seed reporting | All headline numbers report mean ± std where stochastic |

### 7.3 Metrics

- **Flat-lag AUROC** — primary metric for graph recovery
- **Prediction MSE** — secondary metric for forecasting quality
- **Win rate** — fraction of (K,T,seed) cells where method-A beats method-B
- **Wilcoxon p-value** — paired one-sided test for intervention effects
- **Size-matched gap** — `AUROC(combined) − AUROC(obs,big)` to remove sample-size confound

### 7.4 Experiments

| # | Experiment | Script | Falsification stage |
|---|---|---|---|
| 7 | Architecture comparison | `experiments/exp7_architecture_comparison.py` | **F1** |
| 15 | Lagged bottleneck stress grid | `experiments/exp15_lagged_bottleneck_stress_grid.py` | **F2** |
| 16 | Real-benchmark prelim | `experiments/exp16_real_benchmark.py` | F2/F3 |
| 17 | CauseMe lag gate | `experiments/exp17_causeme_lag_gate.py` | **F2** |
| 18 | Mechanistic ablation | `experiments/exp18_mechanistic_ablation.py` | "what survives" |
| 19 | Initial intervention | `experiments/exp19_intervention.py` | (looked positive) |
| 21 | Intervention validation (20 seeds) | `experiments/exp21_intervention_validation.py` | (still looked positive) |
| 22 | Intervention size-matched control | `experiments/exp22_intervention_control.py` | **F4 (~60% confound)** |
| 23 | Intervention type sweep | `experiments/exp23_intervention_sweep.py` | **F4 (~40% artifact)** |
| 24 | Real-data benchmark | `experiments/exp24_real_data_benchmark.py` | **F3** |
| 25 | Causal baselines on intervention | `experiments/exp25_causal_baselines.py` | **F5** |
| 26 | SSM multi-seed real benchmark | `experiments/exp26_ssm_multiseed.py` | F3 (mean ± std) |

---

## 8. Results

### 8.1 F1 — Architecture Does Not Matter

10-seed, matched-capacity sweep. Mean ± std AUROC:

| Dataset | SSM (Mamba) | Linear | LSTM | Transformer | MLP |
|---|---:|---:|---:|---:|---:|
| VAR(1)-chain K=5 | 1.00 ± 0.00 | **1.00 ± 0.00** | — | — | — |
| VAR(1)-random K=10 | 0.93 ± 0.08 | **0.99 ± 0.02** | — | — | — |
| Regime-switch K=3 | 0.98 ± 0.08 | **1.00 ± 0.00** | — | — | — |
| Lorenz K=3 | 0.52 ± 0.24 | **0.94 ± 0.09** | — | — | — |

**Reading:** the linear bottleneck matches or beats the SSM on every dataset. The "emergence" of structure in the SSM is the linear bottleneck doing low-rank lag-1 regression — the SSM achieves the same estimator at higher parameter cost.

### 8.2 F2 — Tuned Lasso Beats the Bottleneck

Focused stress grid: `K ∈ {10, 20}`, `T ∈ {150, 300}`, max-lag 8 (48 cells total).

| Bottleneck vs `{OLS, Ridge, Lasso, RRR}` | Mean delta | Win rate |
|---|---:|---:|
| Flat-lag AUROC | −0.050 | 12% (6/48) |
| Prediction MSE | +0.286 | 0% (0/48) |

Best baseline by AUROC: **Lasso (38/48), RRR (10/48)**.
Best baseline by MSE: **Lasso (48/48)**.

On tigramite CauseMe-style processes the gap **widens** with K: tuned Lasso reaches AUROC **0.98** at K=20 versus **0.73** for the bottleneck. The bottleneck is not the better forecaster either, which qualifies the original "causal discovery as a byproduct of strong forecasting" framing.

### 8.3 F3 — Real Data: Not Competitive on the Clean Benchmark

AUROC on three datasets of progressively cleaner ground truth. Weight-Proj (SSM) reports mean ± std over 5 training seeds; the Lagged Bottleneck row uses a single seed; the four classical methods are deterministic.

| Method | Climate (NOAA, K=6, T=757) | Finance (ETFs, K=10, T=1893) | Lorenz-96 (K=10, T=1500) |
|---|---:|---:|---:|
| **Weight-Proj (SSM, 5 seeds)** | **0.827 ± 0.015** | 0.444 ± 0.046 | 0.722 ± 0.026 |
| PCMCI | 0.704 | 0.510 | **0.986** |
| Tuned Lasso | 0.630 | 0.500 | 0.974 |
| Lagged Bottleneck | 0.617 | 0.514 | 0.920 |
| Tuned Ridge | 0.617 | **0.696** | 0.974 |
| Bivariate Granger | 0.605 | 0.498 | 0.979 |

**Reading:** the bottleneck tops only the climate column, where ground truth is weak (3 edges, K=6). On the clean Lorenz-96 benchmark — the only one with mathematically unambiguous ground truth — classical PCMCI and Granger lead a tight cluster in which the bottleneck **trails by ~0.25 AUROC**.

### 8.4 F3 sensitivity — Ground-Truth Choices Reverse the Climate Ranking

Climate AUROC before and after excluding the definitional ENSO↔SOI edges (the two are the same physical signal with opposite sign, not a causal edge):

| Method | 6 edges (incl. ENSO↔SOI) | 3 edges (honest) |
|---|---:|---:|
| Bivariate Granger | **0.819** | 0.605 |
| Lagged Bottleneck | 0.813 | 0.617 |
| Tuned Lasso | 0.799 | 0.630 |
| Weight-Proj (SSM) | 0.792 | **0.864** |
| PCMCI | 0.771 | 0.704 |
| Tuned Ridge | 0.729 | 0.617 |

**Reading:** removing two definitional edges drops Granger from 1st to last and promotes the SSM bottleneck from middle to 1st. The "win" on climate is a sensitivity artefact, not a causal-discovery claim.

### 8.5 F4 — Intervention Type Sweep

Original Exp 19/21 reported 12/12 configs significant at `p < 10⁻⁵`. Two sequential controls dissolve the result.

**Size-match (Exp 22):** the original comparison conflated more data with intervention content. Adding a fresh observational arm of equal total size shrinks the gap to roughly **40% of the raw value** (mean +0.03 to +0.05 AUROC at `K∈{10,20,30}`, n=15 seeds).

**Intervention type (Exp 23):** the original "intervention" was per-step random forcing, not a standard `do`-intervention. Replacing it with proper `do(X=c)` clamps:

| Intervention type | Significant (out of 12) | Mean gap (AUROC) |
|---|---:|---:|
| Random forcing (Exp 19/21 scheme) | 12/12 | +0.116 |
| Constant `do(X=c)` | 3/12 | +0.002 |
| Soft (added-noise) | 6/12 | +0.008 |

**Reading:** only the original random-forcing scheme produces a stable bottleneck gain, and that gain is best explained as robustness to corrupted target rows rather than causal-content extraction.

### 8.6 F5 — Method-Agnosticity (the strongest single result)

Adding PCMCI and bivariate Granger to the same size-matched protocol. Significant size-matched gains at `n=10` seeds, `p < 10⁻³`:

| Method | K=10 | K=20 | K=30 |
|---|---:|---:|---:|
| **Bottleneck** | +0.026 | +0.040 | +0.054 |
| **Bivariate Granger** | +0.013 | **+0.040** | **+0.095** |
| Lasso | (negative) | (negative) | (negative) |
| PCMCI | ≈ 0 | ≈ 0 | ≈ 0 |

**Reading:** bivariate Granger shows the same effect at the same significance level, with **larger** effect sizes at higher K. Lasso uniquely fails (per-equation regression breaks under target corruption). The intervention effect, insofar as it survives controls, is method-agnostic and matches a target-corruption-robustness story — *and a classical method shows it more strongly than the learned bottleneck.*

---

## 9. What Survives

After five falsifications, three narrow positives remain. **These are characterization claims, not method claims. We do not propose the bottleneck as a causal discovery method.**

### 9.1 Mild-Nonlinearity Configuration

At a single `(K, T) = (20, 300)` point with `nonlinear=0.3` in the data-generating process, the bottleneck beats the best tuned baseline (Lasso/RRR) on **87% of nine `(d_bottleneck, λ_sparse)` cells over 10 seeds**, with mean AUROC improvement **+0.121**. At stronger nonlinearity (`=0.6`, `=1.0`) both methods fail equally; at zero nonlinearity Lasso is preferred. We label this a *configuration*, not a regime — we have not swept K or T for it.

### 9.2 Sample Efficiency

Under size-matched control arms, bottleneck gains from additional observational data exceed Lasso's by **~0.07 AUROC** at `K ∈ {20, 30}`. Modest but useful where collecting more observations is cheap.

### 9.3 Target-Corruption Robustness

The bottleneck's shared `W_out` absorbs per-step random forcing better than Lasso's per-equation fits. This explains the residual intervention effect (F5), but is a **reliability** result, not a causal-discovery one.

---

## 10. Methodological Lessons

The falsification chain produced four lessons that generalize beyond this specific paper.

### 10.1 Size-Matched Controls Beat Seed Counts

Exp 19 had 5 seeds. Exp 21 had 20 seeds and reported `p < 10⁻⁵` across 12 configs. Both found the same effect; neither would have flagged it as a sample-size confound. **Adding seeds tightens a noisy measurement; it cannot diagnose a missing structural control.**

### 10.2 Pre-Register the Intervention Scheme

The gap between "random forcing" and proper `do(X=c)` is the difference between a significant effect and a null, and it is not obvious from the code. Future "intervention experiments" should specify which intervention semantics count as "interventional data" *before* running the experiment.

### 10.3 Prediction Baselines Are Not Causal Baselines

Our intervention experiments compared the bottleneck only against prediction-fit baselines (OLS, Ridge, Lasso, RRR). It was natural to miss that classical *causal* methods (Granger, PCMCI) would benefit from the same intervention scheme by the same mechanism. **Any future "neural causal discovery" claim should include at least one classical causal baseline in the intervention control arm.**

### 10.4 Audit Ground Truth for Definitional Couplings

On soft observational benchmarks with few positive edges, rankings are sensitive to which edges are included in ground truth. Our climate comparison flipped Granger from first to last by removing two edges that are definitionally the same signal. Before claiming a method "wins" on such a benchmark, authors should state explicitly which edges are included, whether any pair is tautologically coupled, and what the ranking looks like with the tautological edges removed.

---

## 11. Reproduction

### Installation

```bash
git clone https://github.com/ankitlade12/ssm-causal.git
cd ssm-causal

# With uv (recommended)
uv sync
uv sync --extra baselines  # for PCMCI / tigramite
uv sync --extra dev        # for tests

# Or with pip
pip install -e .
pip install -e ".[baselines,dev]"
```

### Quick Start (Makefile)

```bash
make help            # List available targets
make test            # Run test suite
make experiments     # Run the full experiment grid (long; see reproduce_all.sh)
make paper           # Compile the arXiv draft
```

### Reproduce a Specific Falsification

```bash
# F1: Architecture sweep (10 seeds, 5 archs, 6 datasets)
uv run python experiments/exp7_architecture_comparison.py

# F2: Lagged bottleneck stress grid (48 cells)
uv run python experiments/exp15_lagged_bottleneck_stress_grid.py

# F3: Real-data benchmark (NOAA, ETFs, Lorenz-96)
uv run python experiments/exp24_real_data_benchmark.py
uv run python experiments/exp26_ssm_multiseed.py   # multi-seed SSM

# F4: Intervention size-matched control
uv run python experiments/exp22_intervention_control.py
uv run python experiments/exp23_intervention_sweep.py

# F5: Add classical causal baselines to the intervention protocol
uv run python experiments/exp25_causal_baselines.py
```

### One-Command Full Reproduction

```bash
chmod +x reproduce_all.sh
./reproduce_all.sh
```

### Compile the Paper

```bash
cd paper/draft
pdflatex main_arxiv.tex
bibtex main_arxiv
pdflatex main_arxiv.tex
pdflatex main_arxiv.tex
# Same recipe for main_4page.tex (anonymous workshop version)
```

### Use the Extraction as a Library

```python
import numpy as np
import torch
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer

# Multivariate time series (T timesteps, K variables)
X = np.random.randn(2000, 5)

# Train Mamba on next-step prediction
model = MambaForTimeSeries(n_variables=5, d_model=64, d_state=16, n_layers=4)
trainer = SSMTrainer(model, lr=1e-3)
trainer.train(X, n_epochs=150)

# Extract the score matrix — the extraction this paper falsifies
W_in = model.input_proj.weight.detach()    # [d_model, K]
W_out = model.output_proj.weight.detach()  # [K, d_model]
S = (W_out @ W_in).abs()                   # [K, K]

# Per this paper, S is NOT a reliable causal-discovery signal.
# Use tuned Lasso or PCMCI for synthetic / real benchmarks instead.
```

---

## 12. Project Structure

```
ssm_causal/
├── ssm_causal/                       # Core library
│   ├── mamba_model.py                #   Mamba SSM (no CUDA required)
│   ├── extractor.py                  #   B, C matrix extraction with hooks
│   ├── bottleneck_training.py        #   Linear / Mamba bottleneck training
│   ├── lag_bottleneck.py             #   Lag-indexed S^(τ) extraction
│   ├── causeme_benchmarks.py         #   Tigramite CauseMe-style runners
│   └── improved_score.py             #   Score normalization + thresholding
│
├── baselines/                        # Comparison methods
│   ├── granger.py                    #   Bivariate Granger (statsmodels)
│   ├── pcmci_wrapper.py              #   PCMCI (tigramite)
│   └── neural_granger.py             #   cLSTM (Tank et al. 2021)
│
├── data/                             # Generators & loaders
│   ├── synthetic.py                  #   VAR(p), Lorenz, regime-switching
│   ├── climate.py                    #   NOAA climate index loaders
│   ├── finance.py                    #   ETF log-return loaders
│   ├── real_loaders.py               #   Unified real-data interface
│   └── real/                         #   Cached climate index .data files
│
├── experiments/                      # 25+ experiments (one per result)
│   ├── exp7_architecture_comparison.py        # F1
│   ├── exp15_lagged_bottleneck_stress_grid.py # F2
│   ├── exp17_causeme_lag_gate.py              # F2
│   ├── exp22_intervention_control.py          # F4 size-match
│   ├── exp23_intervention_sweep.py            # F4 type sweep
│   ├── exp24_real_data_benchmark.py           # F3
│   ├── exp25_causal_baselines.py              # F5
│   └── exp26_ssm_multiseed.py                 # F3 multi-seed
│
├── paper/
│   ├── draft/
│   │   ├── main.tex                  # Long-form draft
│   │   ├── main_arxiv.tex            # Non-anonymous arXiv version
│   │   ├── main_4page.tex            # 4-page anonymous workshop version
│   │   └── README.md                 # Build + submission notes
│   ├── references.bib                # Bibliography
│   └── neurips_2026.sty              # NeurIPS style file
│
├── docs/                             # Per-experiment writeups
│   ├── EXP18_19_20_RESULTS.md
│   ├── EXP21_INTERVENTION_VALIDATION.md
│   ├── EXP24_25_REAL_DATA_AND_CAUSAL_BASELINES.md
│   └── RESEARCH_WRAP_UP_2026-04-16.md
│
├── tests/                            # Test suite
│   ├── test_core.py
│   ├── test_bottleneck_training.py
│   ├── test_lag_bottleneck.py
│   └── test_causeme_benchmarks.py
│
├── scripts/
│   ├── download_etfs.py              # Fetch sector ETF returns
│   └── run_exp18_19_20.sh
│
├── reproduce_all.sh                  # One-command full reproduction
├── Makefile                          # make test / experiments / paper
├── pyproject.toml                    # Dependencies (uv/pip)
├── CITATION.cff
├── LICENSE
└── README.md                         # This file
```

---

## 13. References

### Method papers being tested
- Gu & Dao (2023). *Mamba: Linear-time sequence modeling with selective state spaces.* arXiv:2312.00752.
- Tank, Covert, Foti, Shojaie & Fox (2021). *Neural Granger causality.* IEEE TPAMI 44(8).
- Nauta, Bucur & Seifert (2019). *Causal discovery with attention-based convolutional neural networks.* MAKE.

### Classical causal-discovery baselines
- Granger (1969). *Investigating causal relations by econometric models and cross-spectral methods.* Econometrica 37(3).
- Runge, Bathiany, Bollt et al. (2019). *Detecting and quantifying causal associations in large nonlinear time series datasets.* Science Advances 5(11). (PCMCI)
- Pamfil, Sriwattanaworachai, Schölkopf & Bauer (2020). *DYNOTEARS: Structure learning from time-series data.* AISTATS.

### Climate / real-data ground truth
- Trenberth & Hurrell (1994). *Decadal atmosphere–ocean variations in the Pacific.* Climate Dynamics 9(6).
- Newman, Alexander, Ault et al. (2016). *The Pacific Decadal Oscillation, revisited.* J. Climate 29(12).
- Delworth & Mann (2000). *Observed and simulated multidecadal variability in the Northern Hemisphere.* Climate Dynamics 16(9).
- Lorenz (1996). *Predictability: a problem partly solved.* Proc. ECMWF Seminar on Predictability.

### Sparse-regression baselines
- Tibshirani (1996). *Regression shrinkage and selection via the Lasso.* J. R. Stat. Soc. B.
- Reinsel & Velu (1998). *Multivariate Reduced-Rank Regression.* Springer.

---

## License

MIT — see [LICENSE](LICENSE).
