# Exp 18-20: Exploratory Results and Corrected Interpretation

**Original date:** 2026-04-15
**Corrected interpretation:** 2026-07-25

## Status

Exp 18-20 were useful hypothesis-generating experiments, but their original
mechanistic conclusions are not workshop evidence. Later target-aware controls
and the pre-specified Exp 27 confirmation supersede the claims that bottlenecks
have a mild-nonlinearity regime, a special intervention advantage, or
distribution-shift robustness.

The historical numbers below explain why the follow-up experiments were run.
They should not be cited without the corrected interpretation.

## Historical Exp 18: Mechanistic Ablation

The original sweep compared bottleneck variants with Lasso at one principal
setting, `K=20, T=300`. It reported a positive difference near
`nonlinear=0.3` and interpreted this as a mild-nonlinearity advantage.

That interpretation does not survive Exp 27. The confirmatory experiment fixed
the Linear-BN configuration in advance, used independent data seeds, and swept
`K in {10, 20}`, `T in {300, 600}`, and nonlinearity mixes
`{0.0, 0.2, 0.3, 0.4, 0.6, 1.0}`.

| Setting | Corrected observation |
|---|---|
| `K=10, T=300` | Linear-BN minus Lasso AUROC is `+0.05` at every mix; no cell passes the pre-specified gate. |
| `K=20, T=300` | The difference is `+0.50` at every mix from `0.0` through `1.0`. |
| `K in {10,20}, T=600` | Both methods reach AUROC `1.00`; the difference is zero at every mix. |

Exp 27 records the overall status as `isolated_positive_exploratory`. Because
the `K=20, T=300` effect is exactly invariant to nonlinearity and disappears
when `T` doubles, it does not identify a mild-nonlinearity regime. The supported
interpretation is that the result is consistent with a finite-sample,
dimensional regularization effect in this controlled generator.

Artifact:
`results/corrected_workshop/nonlinearity_confirmation.json`

## Historical Exp 19: Intervention Comparison

Exp 19 initially compared observational data with a larger combined
observational-plus-interventional dataset and reported graph-dependent
bottleneck gains. That comparison did not isolate intervention content from
sample count, and the original implementation did not exclude the intervened
target equation from method fitting.

The corrected protocol uses three matched arms:

- `small`: the original observational trajectory;
- `big`: observational segments with the same usable row count as `combined`;
- `combined`: the small trajectory plus one intervention segment per target.

All lag windows respect segment boundaries. For every intervention segment,
rows whose outcome is the intervened target are excluded. This target-aware
estimand asks whether the remaining equations benefit from intervention
content; it does not score the deterministic or externally forced target
equation as ordinary observational evidence.

Corrected Exp 23 reports the bottleneck's intervention-specific gap relative to
Lasso or Ridge over 10 independent seeds:

| `K` | Intervention | BN relative to Lasso | BN relative to Ridge |
|---:|---|---:|---:|
| 10 | random forcing | `+0.0126` | `-0.0016` |
| 10 | constant clamp | `-0.0131` | `-0.0203` |
| 10 | soft forcing | `+0.0125` | `-0.0017` |
| 20 | random forcing | `+0.0208` | `+0.0122` |
| 20 | constant clamp | `-0.0000` | `-0.0082` |
| 20 | soft forcing | `+0.0209` | `+0.0125` |

Holm-adjusted significance occurs in 6 of 12 cells, but the positive
differences are small (`0.012`-`0.021` AUROC). Constant clamps show no
bottleneck relative advantage. These results do not support a bottleneck-
specific intervention mechanism.

Artifact:
`results/corrected_workshop/intervention_semantics/intervention_sweep_results.json`

## Exp 20: Graph-Label Sensitivity, Not Distribution Robustness

The original Exp 20 held learned score matrices fixed while changing the target
graph labels. It therefore measured score sensitivity to graph changes, not
model robustness under a distribution shift. Noise and coefficient changes
that leave graph labels unchanged necessarily produce zero measured change
under that design.

No distribution-shift robustness claim is retained. A valid robustness study
would require retraining or reevaluating predictions and extracted scores under
explicit source and target distributions.

## Exp 9 Historical Context

Earlier extractor ablations suggested that Jacobian-weighted scores could help
on one small Lorenz example. Those runs remain exploratory and are not part of
the corrected workshop claim set. They motivated auditing richer
interpretability readouts, but do not establish nonlinear causal discovery.

## Corrected Takeaway

The surviving contribution is methodological:

1. Treat promising interpretability signals as hypotheses, not causal
   discoveries.
2. Fix model variants before using graph labels for evaluation.
3. Match usable sample counts and preserve segment boundaries.
4. Encode intervention targets and exclude their forced outcome equations.
5. Compare against classical and sparse baselines under the same estimand.
6. Report independent seeds, paired uncertainty, and multiple-testing
   correction.

The original positive regime and mechanism narratives are superseded. The
corrected evidence supports a reusable falsification and audit protocol for
interpreting dependency structure in predictive models.
