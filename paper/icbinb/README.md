# ICBINB @ NeurIPS 2026 Submission

**Title:** Prediction Bottlenecks Don't Discover Causal Structure
(But Here's What They Actually Do)

**Authors:** Ankit Hemant Lade, Sai Krishna, Aman Chadha

## Overview

This paper reports a 25-experiment falsification chain on the claim
that prediction-trained bottleneck models encode causal structure in
`|W_out @ W_in|`. The method-level claim does not survive. What
survives is a narrow characterization result.

## Structure

- `main.tex` — full paper (currently one-file)
- References pulled from `../references.bib`

## Key Findings to Cite

1. **Architecture doesn't matter** — Linear beats SSM (Exp 7 / P1)
2. **Tuned Lasso beats Bottleneck** — Exp 15-17, on CauseMe-style
   structural processes
3. **Real data: competitive, not superior** — Exp 24, NOAA climate
   + sector ETFs
4. **Intervention effect is a confound** — Exp 19 → 21 → 22 → 23
   → 25, size-matched controls + proper do(X=c) kill the effect
5. **What survives:** mild-nonlinearity regime, sample efficiency
   under growing T, target-corruption robustness

## Building

```bash
cd paper/icbinb
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission Target

ICBINB @ NeurIPS 2026, deadline typically mid-September 2026.
4-8 page workshop paper.

## Related Documents

- `docs/EXP18_19_20_RESULTS.md` — Aman's feedback experiments
- `docs/EXP21_INTERVENTION_VALIDATION.md` — Exp 21 (the finding
  that looked strong)
- `docs/RESEARCH_WRAP_UP_2026-04-16.md` — Sai's final summary
  after Exps 22-23 killed the main claim
- `docs/EXP24_25_REAL_DATA_AND_CAUSAL_BASELINES.md` — Real data +
  causal baselines (this paper's new evidence)
