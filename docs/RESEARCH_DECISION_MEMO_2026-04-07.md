# Research Decision Memo

Date: 2026-04-07

Purpose: record the current decision after CPB, lag probing, explicit lagged
bottleneck, and tuned-baseline stress tests.

## Addendum: 2026-04-08

This memo should now be read as an intermediate decision point, not the final
state.

The next gate it recommended was completed:

- Exp 16 aggregate CauseMe-style benchmarking was directionally negative, but
  its initial implementation used a fallback local VAR path because of a
  Tigramite API mismatch.
- Exp 17 corrected that issue and ran the fairer lag-specific gate on
  CauseMe-style structural causal processes with tuned `OLS`, `Ridge`, `Lasso`,
  and `RRR`.

Final status after Exp 17:

- the lag-resolved bottleneck idea is technically valid,
- but it still does not show a robust advantage over tuned classical baselines,
- so method-building on this branch should be considered closed.

See `EXP17_CAUSEME_LAG_GATE_RESULTS.md` and
`RESEARCH_WRAP_UP_2026-04-08.md` for the authoritative final wrap-up.

## Current Decision

Do **not** continue the current explicit linear lagged bottleneck as the primary
paper method.

The idea is technically viable:

- explicit lagged bottlenecks can recover lag-specific structure on controlled
  synthetic tests,
- the implementation is clean enough to keep as a reusable module,
- and the results are useful supporting evidence for the broader bottleneck
  extraction phenomenon.

But the stronger method claim does not survive tuned classical baselines:

- validation-tuned `Lasso` wins held-out prediction MSE in the focused Exp 15
  K=20 grid,
- `Lasso` or reduced-rank regression (`RRR`) usually wins flat lag AUROC,
- and the bottleneck does not show a consistent graph-recovery advantage.

Current conclusion:

> Lag-resolved bottleneck extraction is a useful probe and ablation, but not yet
> a standalone causal discovery method contribution.

## Evidence Chain

1. **Original SSM-internals thesis weakened.**

   Architecture comparisons showed that the endpoint weight-product signal is
   not uniquely SSM-specific. Linear predictors often preserve the bottleneck
   signal better than Mamba.

2. **Causality-preserving bottleneck training failed Phase 2.**

   Sparse/stability/consistency regularizers did not robustly improve over the
   plain post hoc endpoint score, and lambda sensitivity suggested the losses
   were not meaningfully changing the learned routing.

3. **Exp 13 lag probing was marginal and confounded.**

   Jacobian/impulse lag probing of the existing SSM showed weak signal on simple
   lag settings, but its hard-case failures were confounded by unstable synthetic
   generators.

4. **Exp 14 explicit lagged bottleneck passed clean synthetic feasibility.**

   Splitting `W_in` by lag made lag-specific extraction work on stable small
   synthetic systems:

   `S_lag[tau] = |W_out @ W_in_tau|`

5. **Exp 15 stronger baseline check narrowed the claim.**

   Against fixed OLS/Ridge/Lasso baselines, the bottleneck looked somewhat
   promising in stressed regimes. After adding validation-tuned `Lasso` and
   `RRR`, that advantage mostly disappeared.

## What This Means

The current strongest paper shape is no longer:

> We introduce a new lag-resolved bottleneck causal discovery method.

The more defensible shape is:

> Forecasting bottlenecks often contain readable dependency structure, but this
> phenomenon has clear limits and overlaps substantially with classical sparse
> lagged regression.

That can still be scientifically useful, but it is more of a careful empirical
and mechanistic study than a top-tier method paper unless we find a stronger
separation regime.

## Real-Data Benchmarking

We should add real or semi-real datasets, but with disciplined framing.

The project has relied heavily on synthetic controlled systems. That is useful
for debugging and causal ground truth, but it is not enough for a strong paper
claim. Real data should be used as an **external-validity and stress-test gate**,
not as a proof of causal correctness.

Important caution:

- real observational time series rarely have clean causal ground truth,
- expert/literature-derived graphs are often incomplete,
- interventional datasets may not match our lagged time-series setting,
- and real-data failures are hard to interpret because preprocessing and
  latent confounding can dominate the result.

Candidate benchmark tiers:

1. **CauseMe / Causality for Climate**

   Good fit for time-series causal discovery benchmarking. The Causality for
   Climate competition was hosted on the CauseMe platform and includes
   climate-model-based time-series datasets with known causal ground truth.

   Use for: time-series external validation, especially if lagged ground truth
   is available in an accessible format.

   Source: https://proceedings.mlr.press/v123/runge20a.html

2. **ESS industrial time-series benchmark**

   A real industrial subsystem benchmark with an expert-constructed causal
   graph. This is attractive because it directly targets real multivariate time
   series and known/expert structure.

   Use for: real-system stress test and scaling/robustness analysis.

   Source: https://proceedings.mlr.press/v236/mogensen24a/mogensen24a.pdf

3. **DREAM5 gene regulatory networks**

   Includes real biological network inference tasks such as E. coli and yeast
   expression data with benchmark reference networks. This is less directly
   aligned with lagged forecasting unless time-ordering is available, but it is
   a credible real/semi-real network-inference benchmark family.

   Use for: non-time-series or weakly temporal network extraction comparison,
   only if the method is adapted carefully.

   Reference example: https://pmc.ncbi.nlm.nih.gov/articles/PMC3965471/

4. **Sachs protein signaling**

   A classic interventional causal discovery dataset from single-cell protein
   signaling. It is not a lagged time-series benchmark, so it should not be used
   to validate lag-resolved extraction directly.

   Use for: possible static/non-temporal bottleneck extraction sanity check,
   not as main evidence for this project.

   Source: https://www.science.org/doi/10.1126/science.1105809

Datasets to treat carefully:

- fMRI, finance, and observational climate records can be useful qualitative
  case studies, but they should not be called ground-truth causal benchmarks
  unless the ground truth is clearly defined and externally justified.

## Recommended Next Step

Do not run more GPU-scale synthetic grids yet.

Next concrete step:

1. Commit the current Exp 14/15 implementation and results docs.
2. Add one small real/semi-real benchmark feasibility script, preferably for
   CauseMe/C4C or the ESS industrial benchmark.
3. Treat that script as a data-ingestion and protocol test first:
   fixed extractor, fixed baselines, no tuning on graph labels.
4. Continue method development only if the real/semi-real benchmark reveals a
   regime where bottleneck extraction has a meaningful advantage that tuned
   sparse lagged regression does not already explain.

Decision gate:

- If tuned sparse/low-rank baselines still match or beat the bottleneck on the
  real/semi-real benchmark, stop method-building and write the project as a
  careful characterization/negative-results story.
- If the bottleneck separates on a real/semi-real benchmark, investigate the
  mechanism before scaling experiments.

## Current Position

We should keep the lagged bottleneck code, but treat it as evidence and tooling,
not as the final paper method.

The project is not dead, but the honest next move is to test whether the
bottleneck observation has external validity beyond controlled synthetic
systems.
