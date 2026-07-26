# Exp 24 + 25: Corrected Benchmark and Intervention Baselines

**Original dates:** 2026-04-16 to 2026-04-17
**Corrected local rerun:** 2026-07-25

## Status

This document supersedes the earlier Exp 24/25 tables. In particular:

- Tigramite PCMCI and PCMCI+ matrices are now converted from
  `[source, target, lag]` to the project convention `[target, source]`.
- intervention comparisons are segment-safe and target-aware;
- all methods use the same size-matched estimand;
- climate and finance labels are explicitly treated as soft evidence;
- Lorenz-96 is identified as a simulation with an exact 30-edge graph.

The corrected artifacts are:

- `results/corrected_workshop/real_data/real_data_benchmark_results.json`
- `results/corrected_workshop/real_data/edge_provenance_cards.json`
- `results/corrected_workshop/intervention_methods/causal_baselines_results.json`

## Exp 24: Corrected Benchmark

### Dataset and Label Scope

| Dataset | Kind | Label status | Permitted interpretation |
|---|---|---|---|
| NOAA climate | Real observational, `K=6`, `T=757` | 3 literature-derived soft teleconnection labels | Label-sensitivity audit only |
| Sector ETFs | Real observational, `K=10`, `T=1893` | 6 heuristic soft lead-lag labels | Heuristic sensitivity analysis only |
| Lorenz-96 | Simulated, `K=10`, `T=1500` | Exact 30-edge generating interaction graph | Controlled graph-recovery benchmark |

Climate labels retain ENSO to PNA, ENSO to PDO, and NAO to AMO. The
definitionally coupled ENSO/SOI pair is excluded. Finance labels are not causal
ground truth; they encode a heuristic sector graph and must not be used for a
causal-discovery leaderboard.

### Corrected Results

AUROC values below come directly from the corrected local artifact. This rerun
includes Lasso, Ridge, PCMCI, PCMCI+, and bivariate Granger under documented
default/tuned protocols, plus a three-model-seed Lagged-Bottleneck run
(`42,43,44`, 80 epochs). Every method uses lag horizon `3`.

| Method | Climate | Finance | Lorenz-96 |
|---|---:|---:|---:|
| Lasso | **0.617** | 0.500 | 0.973 |
| Ridge | **0.617** | **0.597** | 0.973 |
| Lagged-Bottleneck (3 model seeds) | **0.626 +/- 0.021** | 0.564 +/- 0.033 | 0.881 +/- 0.018 |
| PCMCI | 0.407 | **0.609** | 0.839 |
| PCMCI+ | 0.605 | 0.579 | 0.875 |
| Bivariate Granger | 0.605 | 0.512 | **0.989** |

The former Lorenz-96 PCMCI value `0.986` was an orientation error and is
superseded by `0.839` under the matched lag horizon. The corrected ordering
still supports the substantive
negative conclusion without claiming that PCMCI dominates: on the exact
simulated graph, simple Granger and tuned linear regression outperform PCMCI
and PCMCI+ in this configuration.

The Lagged-Bottleneck AUROC of `0.881 +/- 0.018` on Lorenz-96 demonstrates a
readable signal on the exact generating graph, but it remains below Granger,
Lasso, and Ridge (`0.989/0.973/0.973`). The reported spread is across model
initialization seeds, with the data held fixed. It is descriptive
initialization stability, not independent uncertainty and not evidence for a
competitiveness claim.

Climate and finance rankings are descriptive only. With three
literature-derived climate labels and six heuristic finance labels, small label
choices can change rankings; the provenance cards are therefore part of the
result, not supplementary decoration.

## Exp 25: Target-Aware Method Comparison

The corrected intervention protocol uses random forcing with:

- 10 independent data-generating seeds;
- `T=300`, `K in {10,20}`;
- equal usable rows in the size-matched observational (`big`) and
  observational-plus-intervention (`combined`) arms;
- independent trajectory segments with no cross-boundary lag windows;
- explicit intervention-target metadata;
- exclusion of the intervened target outcome from each intervention segment;
- aggregate edge AUROC as the primary metric;
- paired bootstrap 95% intervals and Holm-adjusted one-sided Wilcoxon tests.

The primary estimand for each method is:

`AUROC(combined) - AUROC(big)`

### Corrected Size-Matched Gaps

| Method | `K=10, T=300` | `K=20, T=300` |
|---|---:|---:|
| Bottleneck | `+0.057` `[0.036, 0.079]` | `+0.050` `[0.038, 0.061]` |
| Lasso | `+0.033` `[0.017, 0.050]` | `+0.014` `[0.008, 0.021]` |
| PCMCI | `+0.041` `[0.022, 0.060]` | `+0.022` `[0.014, 0.031]` |
| Bivariate Granger | **`+0.068`** `[0.044, 0.094]` | **`+0.080`** `[0.057, 0.106]` |

All eight within-method comparisons reject after Holm correction
(`p_Holm=0.0078125`). The important result is not that interventions validate
the bottleneck: all four methods gain under the same target-aware random-
forcing protocol. A direct exploratory paired comparison of
Granger-minus-Bottleneck gains is not significant at `K=10`
(`p_Holm=0.278`), but is positive at `K=20`
(`+0.030`, `p_Holm=0.027`). The evidence therefore rejects both
bottleneck-specificity and method invariance.

This corrected run also rejects the old target-corruption explanation. Lasso
does not uniquely fail after forced target outcomes are excluded; it gains at
both `K` values. The present evidence does not identify a mechanism for the
common gain.

## Relation to Exp 23

Exp 23 varies intervention semantics and measures the bottleneck's relative
intervention-specific gap against Lasso and Ridge. Positive differences are
small (`0.012`-`0.021` AUROC), and constant clamps yield no bottleneck relative
advantage. Together, Exp 23 and Exp 25 show:

1. intervention semantics materially alter the apparent result;
2. the bottleneck has no advantage under constant clamps;
3. random and soft forcing can produce small relative differences in selected
   cells;
4. random forcing improves all tested methods under the target-aware,
   size-matched estimand;
5. no bottleneck-specific causal extraction claim survives.

DYNOTEARS and VAR-LiNGAM are absent from the corrected empirical table because
their optional dependencies were unavailable locally. Their orientation and
lag adapters are covered by fixture-based tests, but this artifact makes no
empirical claim about either method.

## Workshop Interpretation

The workshop contribution is an audit protocol for interpretability-based
scientific discovery claims:

1. declare matrix direction and lag conventions;
2. attach provenance and claim scope to every graph label;
3. separate real observational soft labels from simulated exact graphs;
4. preserve trajectory boundaries;
5. match usable data budgets;
6. encode intervention targets in fitting and evaluation;
7. include sparse and classical causal baselines;
8. use independent seeds, paired intervals, and multiplicity correction;
9. retain contradictory and null results;
10. state explicitly that readable predictive dependency is not causal
    knowledge.

These experiments are not evidence for deploying bottleneck scores, Granger,
PCMCI, or any other benchmarked method as an automated causal decision system.
Domain validation and an intervention design matched to the scientific
estimand remain necessary.
