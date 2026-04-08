# Exp 15 + Exp 16 GPU Results

**Date:** 2026-04-08
**Hardware:** NVIDIA RTX 4090

---

## Exp 15: Lagged Bottleneck Stress Grid (GPU, with tuned baselines)

72 runs: 3 datasets × 2 K values × 2 T values × 2 d_bottleneck × 3 seeds

Baselines: OLS, Ridge, Lasso (tuned), RRR (tuned)

| Metric | Value |
|---|---|
| Mean flat lag AUROC delta vs best baseline | **-0.071** |
| Graph win rate | **12%** |
| Mean held-out MSE delta | +0.073 |
| MSE win rate | 11% |

By dataset:

| Dataset | Graph Win Rate | Mean Flat Delta |
|---|---|---|
| VAR2-random | 4% | -0.114 |
| Block-lowrank | 21% | -0.077 |
| Delayed-sparse | 12% | -0.022 |

**Verdict:** Tuned baselines (Lasso, RRR) dominate. Bottleneck wins only 12% of runs.

---

## Exp 16: CauseMe-Style Benchmark Gate

3 scenarios, 3 seeds each. CauseMe-style data via tigramite var_process.

| Scenario | Weight-Proj (SSM) | Tuned-Lasso | Granger | Delta (WP-Lasso) |
|---|---|---|---|---|
| K=5, T=500 | 0.817 | **0.917** | ~1.000 | -0.099 |
| K=10, T=800 | 0.781 | **0.984** | ~1.000 | -0.203 |
| K=20, T=1000 | 0.731 | **0.981** | ~1.000 | -0.250 |

**Verdict:** Lasso dominates all scenarios. Gap widens with K.

Correction note:

- the original Exp 16 summary was directionally useful, but the initial
  implementation silently fell back to the local VAR generator because of a
  Tigramite API mismatch,
- and it scored mainly aggregate graph recovery rather than the lag tensor
  directly.

This means Exp 16 should be treated as an intermediate negative signal, not the
final authoritative test of the lag-resolved bottleneck idea.

The corrected final gate is Exp 17 in `EXP17_CAUSEME_LAG_GATE_RESULTS.md`.

---

## Combined Decision

Per Sai's decision memo:

> If tuned sparse/low-rank baselines still match or beat the bottleneck on
> the real/semi-real benchmark, stop method-building and write the project
> as a careful characterization/negative-results story.

Exp 15 plus the corrected Exp 17 gate together support the same higher-level
decision: tuned classical baselines beat or match the bottleneck direction in
the regimes we tested.

**DECISION: Stop method-building. Write characterization paper.**

---

## What the paper should be

The project discovered something real but limited:

1. **Prediction-trained bottleneck models encode dependency structure** — this is a genuine, novel observation
2. **Linear models do it best** (identity internal mapping preserves sparsity)
3. **SSM converges fastest** to dependency structure during training
4. **But tuned sparse regression (Lasso) beats everything** for actual graph recovery

The honest paper is a **characterization study**: "We investigated whether prediction-trained bottleneck models encode Granger structure. They do, but not better than classical methods."

This is still publishable as a workshop paper or empirical study, just not as a methods paper.
