# Exp 17: CauseMe-Style Lag Gate Results

Date: 2026-04-08

Purpose: run the fair final gate for the lag-resolved bottleneck idea on
CauseMe-style structural causal processes, using lag-specific evaluation and the
same tuned lagged baselines that already challenged the synthetic stress-grid
results.

## Why Exp 17 Was Needed

Exp 16 was useful directionally, but it was not the last word on the lagged
bottleneck claim:

- it compared mostly aggregate graph recovery,
- it did not score the lag tensor directly,
- and the initial implementation silently fell back to the local VAR generator
  because of a Tigramite API mismatch.

Exp 17 corrects that by:

- using the intended Tigramite structural causal process generator,
- evaluating lag-specific scores with `flat_lag_auroc` and
  `edge_lag_top1_acc`,
- comparing against tuned `OLS`, `Ridge`, `Lasso`, and `RRR`,
- and still retaining aggregate graph metrics plus held-out prediction MSE.

## Protocol

Scenarios:

- `CauseMe-linear-K5`
- `CauseMe-linear-K10`
- `CauseMe-linear-K20`
- `CauseMe-nonlinear-K5`
- `CauseMe-nonlinear-K10`

Settings:

- 3 seeds per scenario
- 80 epochs
- bottleneck width `d_bottleneck = 16`
- train/test split from the same generated series
- baseline tuning by validation prediction MSE only

Metrics:

- `flat_lag_auroc`
- `edge_lag_top1_acc`
- `aggregate_auroc`
- held-out prediction MSE

## Overall Result

Over 15 runs total:

- flat lag win rate: `13%`
- top-1 lag win rate: `20%`
- aggregate win rate: `7%`
- MSE win rate: `7%`
- mean flat lag AUROC delta vs best baseline: `-0.027`
- mean top-1 lag delta vs best baseline: `-0.087`
- mean aggregate AUROC delta vs best baseline: `-0.021`
- mean test MSE delta vs best baseline: `+0.006`

Best baseline counts:

- best flat lag AUROC: `Lasso` 5, `OLS` 5, `Ridge` 3, `RRR` 2
- best top-1 lag accuracy: `OLS` 15
- best aggregate AUROC: `Lasso` 5, `OLS` 4, `Ridge` 4, `RRR` 2
- best held-out MSE: `Lasso` 13, `OLS` 2

## By Scenario

### Linear

`CauseMe-linear-K5`

- bottleneck underperforms clearly on lag recovery
- mean `flat_lag_auroc`: bottleneck `0.866`, Lasso `0.967`
- mean `edge_lag_top1_acc`: bottleneck `0.722`, baselines near `1.000`

`CauseMe-linear-K10`

- all methods are very strong, but tuned Lasso still slightly leads
- bottleneck is close, not superior

`CauseMe-linear-K20`

- bottleneck roughly ties on flat lag AUROC in some seeds
- bottleneck still loses on lag-top1 and held-out MSE

### Nonlinear

`CauseMe-nonlinear-K5`

- no clear bottleneck win
- `Ridge` / `RRR` lead on average lag and aggregate recovery

`CauseMe-nonlinear-K10`

- this is the only mildly interesting regime
- bottleneck improves top-1 lag identification on average
- but it still does not win cleanly on flat lag AUROC, aggregate AUROC, or MSE

## Interpretation

Exp 17 is the fairest test we ran for the lag-resolved bottleneck hypothesis.

It supports the following conclusion:

> Explicit lag-resolved bottleneck extraction is technically valid, but it does
> not consistently beat tuned sparse or low-rank lagged baselines even when we
> evaluate the lag tensor directly.

This matters because one possible objection to Exp 15 and Exp 16 was:

> maybe the bottleneck only looks weak because we are collapsing lag structure
> into an aggregate graph.

Exp 17 addresses that objection and still does not produce a robust bottleneck
advantage.

## Final Decision From Exp 17

Do not continue lag-resolved bottleneck extraction as an active method-building
direction for this project.

Keep it as:

- a useful implementation artifact,
- a clean falsification result,
- and supporting evidence that bottleneck models can encode real lag structure
  even when classical methods remain stronger.
