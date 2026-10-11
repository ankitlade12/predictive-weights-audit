# When Predictive Weights Look Causal: An Audit Protocol for Interpretability-Based Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository studies a tempting interpretability claim: whether the endpoint
weights of a next-step predictor reveal causal structure. For a model with
input and output projections, the candidate score is

```text
S = abs(W_out @ W_in)
S[target, source] = proposed strength of source -> target
```

The score can contain readable temporal dependency information, but the
corrected evidence does not support interpreting it as causal knowledge or as
an SSM-specific discovery mechanism.

## Workshop Contribution

Accepted at the NeurIPS 2026 Workshop on Interpretability for Discovery
(non-archival). The camera-ready revision is `paper/draft/main_camera_ready.tex`;
build it with `make paper-camera-ready`, or build both PDFs and the checksummed
release archive with `make release`. The original four-page manuscript is
retained separately. See [the revision record](docs/CAMERA_READY_REVISION.md)
for every reviewer concern, follow-up commands, and remaining limits.

Post-review analyses in `results/reviewer_followup/` add ten independent
Lorenz-96 trajectories, actual DYNOTEARS and VAR-LiNGAM runs, and a diagnostic
reproducing all 240 original Lasso fits. These are exploratory additions;
they do not broaden the evaluated graph family or interpretability signal.

The Interp4Discovery-facing contribution is a reusable falsification and audit
protocol for interpretability-based scientific discovery claims. Before
elevating an interpretation to a discovery claim:

1. Test direction, lag, and diagonal conventions on planted examples.
2. Match data budgets, preprocessing, trajectory boundaries, and seeds.
3. Encode intervention targets and state intervention semantics.
4. Compare against sparse and classical baselines under the same estimand, and
   verify that each comparator gives a non-degenerate ranking (no all-tied
   scores, no selected penalty at a grid boundary).
5. Distinguish exact graphs from soft labels.
6. Fix variants before evaluation, report multiplicity-adjusted effects, and
   retain failures.

This is a negative/failure study, not a causal-discovery competition. Earlier
positive results are retained as historical context but are superseded where
the corrected local artifacts disagree.

## Corrected Evidence

### 1. Real-data labels and simulated ground truth

The corrected Exp 24 artifact separates label scope before reporting scores:

| Dataset | Kind | Labels | Claim scope |
|---|---|---|---|
| NOAA climate | Real observational | 3 literature-derived soft labels | Sensitivity audit |
| Sector ETFs | Real observational | 6 heuristic soft labels | Sensitivity audit only |
| Lorenz-96 | Simulated | Exact 30-edge generating graph | Controlled graph recovery |

Corrected AUROC:

| Method | Climate | Finance | Lorenz-96 |
|---|---:|---:|---:|
| Lasso | 0.617 | 0.500 | 0.973 |
| Ridge | 0.617 | 0.597 | 0.973 |
| Lagged-Bottleneck (3 model seeds) | 0.626 +/- 0.021 | 0.596 +/- 0.051 | 0.881 +/- 0.018 |
| PCMCI | 0.407 | 0.609 | 0.839 |
| PCMCI+ | 0.605 | 0.579 | 0.875 |
| Bivariate Granger | 0.605 | 0.512 | 0.989 |

The Lorenz-96 PCMCI AUROC of `0.986` in our earlier preprint
(arXiv:2605.09169) came from a transposed score matrix at lag horizon `5`.
At the correct orientation, with every method evaluated at lag horizon `3`,
the corrected value is `0.839`. Finance Lasso selects the largest grid penalty
and returns all-zero cross-variable scores, so its `0.500` is a tie, not a
ranking. The exact
simulated benchmark still separates readable predictive structure from reliable
graph recovery, but it does not support the old claim that PCMCI dominates.
The Lagged-Bottleneck's `0.881 +/- 0.018` on the exact graph is readable
dependency signal, but Granger, Lasso, and Ridge reach
`0.989/0.973/0.973`. Its spread over model seeds 42, 43, and 44 describes
initialization stability only; these are not independent data replicates and
do not support uncertainty or competitiveness inference.

Artifact:
`results/corrected_workshop/real_data/real_data_benchmark_results.json`

### 2. Intervention semantics

The corrected intervention protocol:

- uses equal usable rows in `big` observational and `combined` arms;
- treats each trajectory as a separate segment;
- prevents lag windows from crossing segment boundaries;
- records the target of each intervention segment;
- excludes the intervened target outcome from fitting on that segment;
- pairs all arms within independent data-generating seeds.

Exp 23 measures the bottleneck's intervention-specific gap relative to Lasso or
Ridge. At `K in {10,20}` and `T=300`, Holm-significant positive effects are
small (`0.012` to `0.021` AUROC). Constant clamps show no bottleneck relative
advantage. Random and soft forcing produce selected positive cells, but do not
establish a bottleneck-specific causal mechanism.

Artifact:
`results/corrected_workshop/intervention_semantics/intervention_sweep_results.json`

### 3. Shared but unequal gains under random forcing

Exp 25 applies the same target-aware, size-matched random-forcing protocol to
four methods. Values are mean
`AUROC(combined) - AUROC(size-matched observational)` with paired bootstrap
95% intervals:

| Method | `K=10, T=300` | `K=20, T=300` |
|---|---:|---:|
| Bottleneck | `+0.057` `[0.036, 0.079]` | `+0.050` `[0.038, 0.061]` |
| Lasso | `+0.033` `[0.017, 0.050]` | `+0.014` `[0.008, 0.021]` |
| PCMCI | `+0.041` `[0.022, 0.060]` | `+0.022` `[0.014, 0.031]` |
| Bivariate Granger | **`+0.068`** `[0.044, 0.094]` | **`+0.080`** `[0.057, 0.106]` |

All eight within-method gains reject after Holm correction
(`p_Holm=0.0078125`), so the effect is not unique to the bottleneck. A direct
exploratory Granger-minus-Bottleneck paired contrast is not significant at
`K=10` (`p_Holm=0.278`), but is positive at `K=20`
(`+0.030`, `p_Holm=0.027`). This rejects a method-invariance claim while still
showing that random forcing can improve several distinct estimators. The
corrected evidence also does not support the older target-corruption
mechanism: target-aware Lasso gains.

Artifact:
`results/corrected_workshop/intervention_methods/causal_baselines_results.json`

### 4. Nonlinearity confirmation

Exp 27 fixed the Linear-BN configuration before evaluation and swept
`K in {10,20}`, `T in {300,600}`, and nonlinearity mixes from `0.0` to `1.0`.
No nonlinearity strength passes the pre-specified regime rule; the six
passing cells all sit at one `(K, T)` setting.

- At `K=20, T=300`, Linear-BN minus Lasso AUROC is exactly `+0.50` at
  every nonlinearity mix, including `0.0` and `1.0`.
- At `T=600`, both methods reach `1.00` AUROC and the difference vanishes.
- At `K=10, T=300`, the difference is `+0.05` at every mix, but no cell
  passes the pre-specified gate.

The invariance to nonlinearity and disappearance with more samples reject the
old mild-nonlinearity interpretation. The post-review diagnostic
(`results/reviewer_followup/lasso_diagnostic.json`) locates the cause in the
comparator: Linear-BN reaches `1.00` AUROC in all 240 fits, while every
`K=20, T=300` Lasso fit selects the largest grid penalty (`alpha=1`) and
returns all-zero cross-variable scores.

Artifact:
`results/corrected_workshop/nonlinearity_confirmation.json`

## What Is Superseded

The following statements should not be used as current conclusions:

- PCMCI achieves `0.986` AUROC on Lorenz-96 (transposed scores, lag horizon 5).
- DYNOTEARS and VAR-LiNGAM achieve `0.983` and `0.968` on Lorenz-96 (these
  included contemporaneous coefficients; strictly lagged values are `0.833`
  and `0.896`).
- Lorenz-96 is a real dataset or has 90 evaluated cross-variable edges.
- Bottlenecks have a confirmed mild-nonlinearity regime.
- The current experiments establish sample efficiency.
- Random-forcing gains arise from target-corruption robustness.
- Exp 20 establishes distribution-shift robustness.
- Intervention data uniquely benefits a bottleneck or neural method.

## Responsible Interpretation

Readable weights are evidence about a trained predictor, not automatically
about the data-generating process. Climate and finance scores must not be used
to recommend interventions, trading decisions, or scientific mechanisms.
Lorenz-96 results are controlled simulation evidence only. Any downstream
causal claim requires domain validation, explicit assumptions, target-aware
intervention design, and comparison with methods fit to the same estimand.

## Reproduction

The checked-in `uv.lock` is the dependency authority. Install the exact
core, baseline, and test environment:

```bash
uv sync --frozen --extra baselines --extra dev
```

Run the full test suite:

```bash
make test
```

Run the native bounded, CPU-friendly reproduction path:

```bash
make smoke
```

Run the corrected experiments:

```bash
make workshop-evidence
```

The exact commands and claim scope for each artifact are recorded in
`results/corrected_workshop/experiment_manifest.json`.

Build the camera-ready paper, the preprint, and the release archive with the
official NeurIPS 2026 style:

```bash
# Debian/Ubuntu native build dependency
sudo apt-get install texlive-latex-base texlive-latex-extra \
  texlive-fonts-recommended texlive-bibtex-extra lmodern
make release
```

The PDFs are written to `paper/draft/main_camera_ready.pdf` and
`paper/draft/main_arxiv.pdf`; the archive is written to
`dist/camera_ready_release.zip`.

The historical double-blind submission and its review supplement remain
reproducible; they are not the final paper. The submission builds with
`make paper-workshop` (written to `paper/draft/main_4page.pdf`).

Build the deterministic, identity-scanned review supplement:

```bash
make anonymous-supplement
```

## Repository Map

```text
ssm_causal/       model, extraction, intervention, and statistical utilities
                  (package name kept from the project's original SSM focus)
baselines/        Granger, PCMCI/PCMCI+, DYNOTEARS, and VAR-LiNGAM wrappers
data/             synthetic generators and observational data loaders
experiments/      executable experiment protocols
results/          local experiment artifacts
tests/            direction, lag, segmentation, seed, and statistics checks
docs/             research history, design decisions, and corrected summaries
paper/            workshop and arXiv paper sources
```

## Historical Context

The project began by asking whether selective SSM internals expose causal
structure. Architecture controls showed that a linear bottleneck can match or
beat the SSM, and sparse regression remains a strong baseline. A later
causality-preserving bottleneck design explored whether training objectives
could make the endpoint score more readable. The corrected workshop result is
not a new extraction method; it is the audit procedure that prevented readable
predictive structure from being overstated as causal discovery.

## License

MIT. See [LICENSE](LICENSE).
