# Exp 15: Lagged Bottleneck Stress Grid

Date: 2026-04-07

Status: **preliminary local CPU result**

Update: after adding validation-tuned `Ridge`, `Lasso`, and reduced-rank regression
(`RRR`) baselines, the bottleneck result is weaker than the first fixed-baseline
stress test. Tuned `Lasso` is the strongest baseline in the focused K=20 grid.

## Question

Does explicit lagged bottleneck extraction separate from ordinary lagged linear baselines in sample-limited or over-specified-lag regimes?

The tested claim is narrower than "lag extraction works":

- easy VAR settings are already solved by OLS/Ridge/Lasso,
- the possible bottleneck advantage should come from low-rank regularization,
- so the key regime is higher-dimensional, lower-sample, and over-specified lag order.

## Implementation

New stress script:

```bash
experiments/exp15_lagged_bottleneck_stress_grid.py
```

It compares:

- `LaggedBottleneck`
- `OLS`
- `Ridge`
- `Lasso`
- `RRR`

Main metrics:

- flat lag AUROC across all `(lag, target, source)` entries,
- edge lag top-1 accuracy,
- held-out prediction MSE,
- deltas against the best baseline.

## Local Run

### Fixed-Baseline Pass

Command:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/exp15_lagged_bottleneck_stress_grid.py \
  --device cpu \
  --datasets VAR2-random Block-lowrank Delayed-sparse \
  --K-values 10 20 \
  --T-values 150 300 \
  --max-lag-values 8 \
  --d-bottleneck-values 8 16 \
  --seeds 2 \
  --epochs 80 \
  --batch-size 64 \
  --save-dir /tmp/exp15_lagged_bottleneck_local_grid
```

Summary over 48 runs:

| Metric | Value |
|---|---:|
| Mean flat lag AUROC delta vs best baseline | -0.015 |
| Median flat lag AUROC delta vs best baseline | -0.004 |
| Graph win rate | 0.29 |
| Mean held-out MSE delta vs best baseline | -1.256 |
| Held-out MSE win rate | 1.00 |

By dataset:

| Dataset | Mean flat delta | Graph win rate | Mean MSE delta | MSE win rate |
|---|---:|---:|---:|---:|
| VAR2-random | -0.014 | 0.38 | -1.189 | 1.00 |
| Block-lowrank | -0.003 | 0.44 | -1.222 | 1.00 |
| Delayed-sparse | -0.028 | 0.06 | -1.358 | 1.00 |

By regime:

| Regime | Mean flat delta | Graph win rate | Mean MSE delta |
|---|---:|---:|---:|
| K=10 | -0.032 | 0.21 | -0.906 |
| K=20 | +0.002 | 0.38 | -1.606 |
| T=150 | -0.057 | 0.12 | -1.718 |
| T=300 | +0.027 | 0.46 | -0.794 |

Most promising cells:

| Cell | Mean flat delta | Graph win rate | Mean MSE delta |
|---|---:|---:|---:|
| VAR2-random, K=20, T=300 | +0.064 | 1.00 | -1.297 |
| Block-lowrank, K=20, T=300 | +0.125 | 1.00 | -1.267 |

## Interpretation

This does not yet prove a clean graph-recovery advantage. The bottleneck wins held-out prediction MSE consistently, but graph AUROC only improves in a minority of runs.

The emerging hypothesis is:

> Explicit lagged bottlenecks act as useful low-rank regularizers for over-specified lagged prediction, but graph recovery only improves when there is enough sample signal to identify the lag structure.

This is different from the original pivot and more plausible than the broad SSM claim, but it still needs stronger validation.

## Original Next Gate

Before the tuned-baseline follow-up, the proposed next gate was to run a larger
grid around the cells where the signal appeared:

- `K=20, 50`
- `T=300, 600`
- `max_lag=8, 12, 16`
- `d_bottleneck=8, 16, 32`
- datasets: `VAR2-random`, `Block-lowrank`, and harder nonlinear delayed variants
- seeds: at least 5

Continue only if the bottleneck shows both:

- consistent flat lag AUROC gains over OLS/Ridge/Lasso,
- and held-out prediction MSE that remains better or comparable.

This gate is now superseded by the tuned-baseline follow-up below.

## Stronger Tuned-Baseline Follow-Up

Added baselines:

- validation-tuned `Ridge`,
- validation-tuned `Lasso`,
- validation-tuned reduced-rank regression (`RRR`).

Tuning used held-out prediction MSE on a validation slice from the training
segment. Graph labels were not used for tuning.

Focused local command:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/exp15_lagged_bottleneck_stress_grid.py \
  --device cpu \
  --datasets VAR2-random Block-lowrank \
  --K-values 20 \
  --T-values 300 600 \
  --max-lag-values 8 12 \
  --d-bottleneck-values 16 32 \
  --seeds 3 \
  --epochs 100 \
  --batch-size 64 \
  --baseline-methods ols ridge lasso rrr \
  --baseline-alphas 1e-4 1e-3 1e-2 1e-1 1.0 \
  --rrr-ranks 4 8 16 32 \
  --save-dir /tmp/exp15_tuned_baselines_focused_k20
```

Summary over 48 runs:

| Metric | Value |
|---|---:|
| Mean flat lag AUROC delta vs best tuned baseline | -0.050 |
| Median flat lag AUROC delta vs best tuned baseline | -0.041 |
| Graph win rate | 0.12 |
| Mean held-out MSE delta vs best tuned baseline | +0.286 |
| Held-out MSE win rate | 0.00 |

By dataset:

| Dataset | Mean flat delta | Graph win rate | Mean MSE delta | MSE win rate |
|---|---:|---:|---:|---:|
| VAR2-random | -0.100 | 0.00 | +0.293 | 0.00 |
| Block-lowrank | +0.000 | 0.25 | +0.280 | 0.00 |

Baseline winners:

| Winner type | Count |
|---|---:|
| Best flat lag AUROC: `Lasso` | 38 / 48 |
| Best flat lag AUROC: `RRR` | 10 / 48 |
| Best held-out MSE: `Lasso` | 48 / 48 |

## Revised Interpretation

The earlier fixed-baseline stress result overstated the bottleneck advantage.
Once `Lasso` and `RRR` are tuned, the explicit lagged bottleneck does not show a
clean advantage on this focused grid.

Current conclusion:

> Lag-resolved bottleneck extraction is technically viable, but it is not yet a
> strong standalone paper direction because tuned sparse lagged regression
> baselines match or beat it.

The most defensible role for this result right now is supporting evidence for
the broader bottleneck-extraction story, not the primary method claim.

If we keep exploring, the next attempt should change the mechanism rather than
only scale the same grid:

- tune bottleneck sparsity directly for the explicit lagged model,
- add nonlinear lagged bottlenecks where linear `Lasso` should be misspecified,
- or test architectures that share parameters across lags/sources in a way
  classical sparse VAR baselines do not.

## Current Next Gate

Do not run a large rental-GPU grid for the current explicit linear lagged
bottleneck as-is.

Continue only if one of the following produces a new separation from tuned
`Lasso` and `RRR`:

- explicit bottleneck sparsity tuning improves flat lag AUROC without hurting
  held-out prediction,
- nonlinear lagged bottlenecks beat tuned linear sparse baselines on nonlinear
  delayed systems,
- or a parameter-sharing lagged architecture creates a regime that classical
  sparse VAR baselines cannot match.
