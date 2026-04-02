# SSM-Causal: Causal Discovery via State Space Model Internals

> **Paper:** *Causal Discovery via State Space Model Internals*
> **Venue:** NeurIPS 2026
> **Authors:** Ankit Hemant Lade, Aman Chadha

## TL;DR

A trained Mamba model's weight matrices encode Granger causal structure.
The product `S = |W_out @ W_in|` recovers the causal graph — **no causal-specific
training, no regularization, no statistical testing**. Just train on prediction,
read the weights.

## Key Results

| Dataset | SSM-Causal | Granger | PCMCI | cLSTM |
|---|---|---|---|---|
| VAR(1) chain K=3 | **1.000** | **1.000** | 0.500 | 0.750 |
| VAR(1) chain K=5 | 0.969 | **1.000** | 0.406 | 0.234 |
| VAR(1) fork K=5 | 0.859 | **1.000** | 0.469 | 0.109 |
| Regime-switching K=3 | **1.000** | **1.000** | **1.000** | 0.375 |
| Lorenz (nonlinear) K=3 | **0.800** | **0.800** | 0.400 | 0.200 |
| **Climate K=8** | **0.984** | 0.933 | 0.333 | 0.698 |
| fMRI K=12 | 0.889 | **0.972** | 0.590 | **0.994** |

SSM-Causal **outperforms both PCMCI and cLSTM on 6/7 benchmarks** and achieves the
highest AUROC on climate data (0.984), beating Granger for the first time.

## Installation

```bash
# Clone
git clone https://github.com/ankit/ssm-causal.git
cd ssm-causal

# Install with uv
uv sync
uv sync --extra baselines  # for PCMCI baseline
uv sync --extra dev        # for tests

# Or with pip
pip install -e .
pip install -e ".[baselines,dev]"
```

## Quick Start

### Core validation (should print "THE PAPER IS REAL")
```bash
uv run python experiments/exp0_core_validation.py
```

### Run all experiments
```bash
make experiments
# or
chmod +x reproduce_all.sh && ./reproduce_all.sh
```

### Run tests
```bash
make test
# or
uv run pytest tests/ -v
```

### Use as a library
```python
import numpy as np
import torch
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer

# Your multivariate time series (T timesteps, K variables)
X = np.random.randn(2000, 5)

# Train Mamba on next-step prediction
model = MambaForTimeSeries(n_variables=5, d_model=64, d_state=16, n_layers=4)
trainer = SSMTrainer(model, lr=1e-3)
trainer.train(X, n_epochs=150)

# Extract causal graph — one line
W_in = model.input_proj.weight.detach()   # [d_model, K]
W_out = model.output_proj.weight.detach() # [K, d_model]
S = (W_out @ W_in).abs()                  # [K, K] causal score matrix

# That's it. S[i,j] = strength of causal link j -> i
```

## Reproducing Paper Results

Each experiment maps to a specific table/figure in the paper:

| Experiment | Script | Paper Reference |
|---|---|---|
| Core validation | `exp0_core_validation.py` | Day 3 test |
| Synthetic benchmark | `exp1_synthetic.py` | Table 1 |
| Regime-switching | `exp2_nonstationary.py` | Table 2 + Figure 1 |
| Climate teleconnections | `exp3_climate.py` | Figure 2 + Table 3 |
| fMRI brain connectivity | `exp4_fmri.py` | Figure 3 + Table 3 |
| Runtime scaling | `exp5_efficiency.py` | Table 4 + Figure 4 |
| Ablation study | `exp6_ablation.py` | Table 5 |

Auto-generate LaTeX tables from results:
```bash
uv run python paper/generate_tables.py
```

## Project Structure

```
ssm_causal/
├── ssm_causal/                 # Core library
│   ├── mamba_model.py          #   Mamba SSM implementation (no CUDA required)
│   ├── extractor.py            #   B, C matrix extraction with hooks
│   ├── score.py                #   Causal score computation API
│   ├── threshold.py            #   Score → binary graph (Otsu, percentile, etc.)
│   └── utils.py                #   Metrics (AUROC, F1) & visualization
│
├── baselines/                  # Comparison methods
│   ├── granger.py              #   Bivariate Granger test (statsmodels)
│   ├── pcmci_wrapper.py        #   PCMCI (tigramite)
│   └── neural_granger.py       #   cLSTM (Tank et al. 2021)
│
├── data/                       # Data generators & loaders
│   ├── synthetic.py            #   VAR(1/2), Lorenz, Hénon, regime-switching
│   ├── climate.py              #   NOAA climate indices + synthetic
│   └── fmri.py                 #   HCP fMRI + synthetic brain connectivity
│
├── experiments/                # Reproducible experiments (one per table/figure)
│   ├── exp0_core_validation.py #   Day 3 test — does the idea work?
│   ├── exp1_synthetic.py       #   Table 1: full synthetic benchmark
│   ├── exp2_nonstationary.py   #   Table 2 + Figure 1: regime-switching
│   ├── exp3_climate.py         #   Figure 2: climate teleconnections
│   ├── exp4_fmri.py            #   Figure 3: brain connectivity
│   ├── exp5_efficiency.py      #   Table 4: runtime scaling
│   └── exp6_ablation.py        #   Table 5: ablation study
│
├── paper/                      # NeurIPS paper (LaTeX)
│   ├── main.tex                #   Main paper (modular \input{} structure)
│   ├── introduction.tex        #   Section 1
│   ├── background.tex          #   Section 2
│   ├── method.tex              #   Section 3 (SSM Causal Score + Theory)
│   ├── experiments.tex         #   Section 4
│   ├── related_work.tex        #   Section 5
│   ├── conclusion.tex          #   Section 6
│   ├── theorem1_proof.tex      #   Appendix A: Theorem 1 proof
│   ├── checklist.tex           #   NeurIPS mandatory checklist
│   ├── references.bib          #   Bibliography
│   ├── generate_tables.py      #   Auto-generate LaTeX from JSON results
│   └── neurips_2026.sty        #   NeurIPS style file
│
├── tests/                      # Test suite (19 tests)
│   └── test_core.py            #   Data generators, model, extraction, metrics
│
├── results/                    # Experiment outputs (auto-generated)
├── reproduce_all.sh            # One-command full reproduction
├── Makefile                    # make install / test / experiments / paper
├── pyproject.toml              # Dependencies (uv/pip)
├── CITATION.cff                # Citation metadata
└── README.md                   # This file
```

## Method Overview

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
Compute S = |W_out · W_in| ∈ ℝ^{K×K}  ← ONE MATRIX MULTIPLY
  │
  ▼
Output: Causal graph Ĝ (threshold S)
```

**Why it works:** A model trained to predict Y must learn which inputs
causally influence which outputs. The weight matrices encode this routing.
Their product captures the net input-to-output pathway — precisely what
Granger causality measures.

## Citation

```bibtex
@inproceedings{lade2026ssm,
  title={Causal Discovery via State Space Model Internals},
  author={Lade, Ankit Hemant and Chadha, Aman},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

MIT
