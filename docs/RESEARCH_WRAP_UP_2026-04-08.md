# Research Wrap-Up

Date: 2026-04-08

Purpose: record the final state of the project after the full sequence of
method-building attempts, benchmark corrections, and final lag-specific
semi-real evaluation.

## Final Decision

Stop new method development on this project line.

The project produced a real empirical observation, a substantial amount of
careful falsification, and several reusable experiment modules. But it no
longer supports a strong new-method paper claim.

## What We Tested

1. **Original SSM-internals-first thesis**

   We tested whether selective SSM internals supported a distinctive and cheap
   post hoc causal discovery mechanism. The main SSM-specific claim weakened
   once stronger architecture comparisons showed that the endpoint bottleneck
   signal is not uniquely tied to Mamba/SSMs.

2. **Broader bottleneck pivot**

   We kept the empirical insight that `|W_out @ W_in|` can encode dependency
   structure after prediction training. This remains the strongest surviving
   phenomenon from the project.

3. **Causality-preserving bottleneck training (CPB)**

   Sparse/stability/consistency regularizers did not robustly improve over the
   plain post hoc bottleneck baseline, and Phase 2 validation did not justify
   continuing CPB as the main direction.

4. **Lag probing of existing SSMs**

   Jacobian/impulse lag probing showed limited signal and was partly confounded
   by synthetic-generator instability on the harder cases.

5. **Explicit lag-resolved bottleneck extraction**

   We then tested the cleaner explicit lagged model:

   `S_lag[tau] = |W_out @ W_in_tau|`

   This passed synthetic feasibility and was technically valid.

6. **Stronger tuned baselines**

   Once validation-tuned `Lasso` and `RRR` were added, the lagged bottleneck no
   longer had a clean advantage on stress tests.

7. **CauseMe-style semi-real gates**

   We added semi-real structural-process benchmarks and then a corrected
   lag-specific gate. The final lag-specific test still did not show a robust
   bottleneck win over tuned classical baselines.

## What Survived

These claims still look real:

- prediction-trained bottleneck models often encode readable dependency
  structure,
- `|W_out @ W_in|` is a meaningful empirical extraction signal,
- linear predictors frequently preserve that signal as well as or better than
  the more complex neural alternatives,
- and the project now has a good negative-result benchmark scaffold.

## What Did Not Survive

These claims do not currently hold strongly enough:

- that the effect is SSM-specific,
- that CPB creates a better causal extractor than the plain bottleneck score,
- that explicit lagged bottlenecks beat tuned sparse or low-rank lagged
  regression,
- or that this line currently supports a top-tier methods-paper contribution.

## Honest Publication Outlook

This project no longer looks like a strong NeurIPS/ICML methods paper.

It could still support a smaller output if desired:

- a workshop paper,
- a characterization study,
- a negative-results paper,
- or a clean technical report / preprint.

The honest story would be:

> Forecasting bottlenecks often encode dependency structure, but tuned
> classical sparse/low-rank baselines remain stronger for graph recovery.

## Recommended Repository State

Keep:

- the bottleneck extraction implementation,
- the lagged bottleneck implementation,
- the tuned baseline comparisons,
- and the result summaries documenting the falsification path.

Do not keep treating any of the following as the active next method:

- CPB regularization,
- lag probing of existing SSMs,
- explicit lagged bottleneck extraction,
- or "one more tuning sweep" to rescue the method claim.

## If We Ever Revisit This

Only revisit as one of these:

- a characterization/benchmark paper,
- a small real-data external-validity study,
- or a new idea that is genuinely outside the current bottleneck-method family.

Do not revisit by making another small variation of the same bottleneck method
and hoping it will separate from tuned sparse regression.
