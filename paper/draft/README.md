# Paper Drafts — Prediction Bottlenecks Don't Discover Causal Structure

**Title:** Prediction Bottlenecks Don't Discover Causal Structure
(But Here's What They Actually Do)

**Authors:** Ankit Hemant Lade, Sai Krishna Jasti, Indar Kumar, Aman Chadha

## Overview

A staged falsification of the claim that prediction-trained bottleneck
models encode causal structure in `|W_out @ W_in|`. The method-level
claim does not survive. What survives is a narrow characterization
result.

## Files

- `main.tex` — full long-form draft, source for the trims below
- `main_arxiv.tex` — non-anonymous arXiv version
- `main_4page.tex` — 4-page anonymous workshop version
- References: `../references.bib`

## Key Findings

1. **Architecture doesn't matter** — Linear matches/beats SSM at the
   same extraction (Exp 7 / P1).
2. **Tuned Lasso beats the bottleneck on synthetic graphs** — Exp 15-17.
3. **Real data**: bottleneck not the best on the only benchmark with
   clean ground truth (Lorenz-96); rankings on climate are fragile to
   ground-truth choices (Exp 24).
4. **Intervention effect is a confound** — Exp 19 → 21 → 22 → 23 →
   25, size-matched controls + proper do(X=c) kill the headline
   effect; the residual reproduces (and is larger) under classical
   Granger.
5. **What survives** — mild-nonlinearity regime, sample efficiency
   under growing T, target-corruption robustness.

## Building

```bash
cd paper/draft
pdflatex main_arxiv.tex
bibtex main_arxiv
pdflatex main_arxiv.tex
pdflatex main_arxiv.tex
```

(Same recipe for `main_4page.tex`.)

## Submission targets

- **arXiv** (non-anonymous, `main_arxiv.tex`) — post first to establish
  priority.
- **TMLR** (archival journal, accepts negative results, no novelty
  requirement) — primary archival home.
- **NeurIPS 2026 workshop** (non-archival, 4-page version,
  `main_4page.tex`) — to be picked once the workshop list is announced
  ~July 11, 2026; paper deadlines typically late August.

The TMLR submission must precede or run in parallel with any workshop
submission, since TMLR does not accept "expanded versions of conference
papers."

## Related documents

- `docs/EXP18_19_20_RESULTS.md` — initial intervention experiments
- `docs/EXP21_INTERVENTION_VALIDATION.md` — Exp 21 (the finding that
  looked strong before the controls)
- `docs/RESEARCH_WRAP_UP_2026-04-16.md` — final summary after Exps 22-23
  killed the headline claim
- `docs/EXP24_25_REAL_DATA_AND_CAUSAL_BASELINES.md` — real-data
  benchmark + causal baselines (the evidence behind §F3 and §F5)
