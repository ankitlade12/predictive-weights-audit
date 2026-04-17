# Research Wrap-Up (Follow-Up)

Date: 2026-04-16

Purpose: close out the project after the post-wrap-up follow-ups that were
motivated by advisor feedback on the 2026-04-08 wrap-up. This document
supersedes that one as the final status.

## Context for This Follow-Up

The 2026-04-08 wrap-up already recommended stopping new method development.
Advisor feedback on that wrap-up suggested one concrete direction before
walking away: test whether interventional or distribution-shifted data amplify
the gap between learned bottlenecks and linear/sparse baselines. Exps 18-21
were run to chase that suggestion. Exps 22-23 were added to properly control
the intervention claim after it looked stronger than it was.

## What Was Run After 2026-04-08

1. **Exp 18 — Mechanistic ablation across regimes.**
   Swept noise, density, nonlinearity. Bottleneck only has a clean win in the
   mild-nonlinearity regime (nonlinear=0.3, 78% win rate). Ties or loses
   elsewhere. Clean characterization finding.

2. **Exp 19 — Observational vs interventional vs combined.**
   First pass at the advisor's suggestion. Reported "BN benefits more from
   interventional data on random/fork graphs, 60-80% win rate."

3. **Exp 20 — Distribution shift robustness.**
   No robustness advantage for bottleneck over Lasso. Clean negative.

4. **Exp 21 — Validation of the Exp 19 intervention finding.**
   20 seeds, Wilcoxon, K×T grid. Reported 12/12 configs significant, p < 1e-5,
   80-100% win rate. Claimed this was "the strongest positive result in the
   entire project."

5. **Exp 22 — Size-matched observational control (added after review).**
   Exp 21 compared `AUROC(X_obs + X_int)` against `AUROC(X_obs)` without
   matching sample size: `X_combined` has `T + K*T_int_per_target` rows,
   `X_obs` has `T`. The gain metric therefore conflated "intervention content"
   with "more training data." Exp 22 added a third arm: a fresh observational
   draw of the same size as `X_combined`. Intervention-specific gap =
   `AUROC(combined) − AUROC(obs_big)`.

   Result: the gap shrank from Exp 21's ~+0.13 AUROC to a size-matched
   +0.03-0.05 AUROC. Still statistically significant (p ≈ 3e-5, n=15), but
   about 60% of Exp 21's headline was the data-size confound.

6. **Exp 23 — Intervention type × strength × baseline sweep.**
   Exp 22's residual effect used Exp 21's non-standard "random forcing"
   intervention (per-step clamp to `strength × random sign`). Exp 23 added
   proper do(X=c) constant clamps and soft interventions, plus Ridge and RRR
   baselines.

## Final Result Table (Exp 23)

Intervention-specific gap, mean over K∈{20,30}, 12 seeds each:

| Intervention type | Significant configs | Mean gap (AUROC) |
|---|---|---|
| `random` (Exp 21 scheme) | 12/12 | +0.116 |
| `constant` (do(X=c)) | 3/12 | +0.002 |
| `soft` (exogenous noise) | 6/12 | +0.008 |

| Baseline | Significant configs | Mean gap |
|---|---|---|
| Lasso | 9/12 | +0.032 |
| Ridge | 7/12 | +0.013 |
| RRR | 5/12 | +0.080 (RRR itself weak here — ~0.87 AUROC vs Lasso ~0.99) |

At K=30 with `constant` strength=1, Ridge actually beats the bottleneck
(gap −0.021, p=1.00).

## What This Means

The Exp 21 claim as written — that bottlenecks benefit disproportionately
from interventional data — does not survive either control.

- The size-matching control explained ~60% of the headline gap as a
  data-size confound.
- The intervention-type sweep showed the residual is specific to the
  random-forcing scheme and vanishes under standard do(X=c) or soft
  interventions.

The mechanism is almost certainly robustness-to-corrupted-targets rather
than causal-signal extraction: per-step random forcing clamps the target
variable to noise, which breaks Lasso's per-equation regression for that
row, while the bottleneck's shared `W_out` absorbs the damage. That is a
different phenomenon from "interventions expose causal structure to
low-rank learners."

## What Is Worth Keeping (Updated)

From the full project, the following claims still look honest:

- Prediction-trained bottlenecks encode readable dependency structure via
  `|W_out @ W_in|`.
- In the mild-nonlinearity regime (nonlinear≈0.3), bottlenecks outperform
  tuned Lasso in graph recovery (Exp 18).
- Bottlenecks appear more sample-efficient than sparse linear baselines
  under fixed K as T grows — BN gain from extra observational data
  exceeds Lasso's by ~+0.07 AUROC at K=20-30 (Exp 22).
- Bottlenecks are more robust to target-corruption patterns such as the
  random-forcing scheme tested in Exp 21-23 (Exp 23).

These remain characterization results, not causal-discovery method results.

## What Does Not Survive (Updated)

- The claim that bottlenecks carry a specific advantage for interventional
  data over tuned linear baselines.
- Any framing that leans on the Exp 21 headline numbers without Exp 22's
  size-matched control and Exp 23's intervention-type sweep.

## Final Publication Outlook

Still not a methods paper at NeurIPS / ICML / UAI / AISTATS level.

Could still support, if any of us wants to write it up:

- a short workshop paper combining Exp 18 (mild-nonlinearity regime),
  Exp 22's sample-efficiency finding, and Exp 23's null on proper
  interventions, framed as "when learned bottlenecks align with and
  diverge from sparse regression,"
- or a technical report / preprint with the same scope.

The honest title would be something like:

> Learned bottlenecks and tuned sparse regression behave similarly on
> lagged causal-graph recovery, with narrow bottleneck advantages in
> mild-nonlinearity and target-corruption regimes.

## Closing

This is the final close on the method-building track. The intervention
direction was worth chasing once — the advisor was right that it might
amplify differences — but the amplified difference turned out to be a
sample-size confound plus a target-corruption artifact, not a causal
intervention signal. No further one-more-experiment is likely to rescue
this line.

## Artifacts Added in This Round

- `experiments/exp18_mechanistic_ablation.py`
- `experiments/exp19_intervention.py`
- `experiments/exp20_distribution_shift.py`
- `experiments/exp21_intervention_validation.py`
- `experiments/exp22_intervention_control.py`
- `experiments/exp23_intervention_sweep.py`
- `docs/EXP18_19_20_RESULTS.md`
- `docs/EXP21_INTERVENTION_VALIDATION.md`
- `results/intervention_control/intervention_control_results.json`
- `results/intervention_sweep/intervention_sweep_results.json`
