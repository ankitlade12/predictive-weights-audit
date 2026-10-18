# When Predictive Weights Look Causal: An Audit Protocol for Interpretability-Based Discovery

Code and experimental evidence for the accepted NeurIPS 2026 Workshop on
Interpretability for Discovery study (non-archival). Manuscripts, PDFs,
LaTeX sources, and paper-build tooling are maintained outside this repository.

[Citation](CITATION.cff) · [MIT license](LICENSE)

| Author | Affiliation |
|---|---|
| Ankit Hemant Lade | Ascentt Systems Inc |
| Sai Krishna Jasti | Nextgen Healthcare |
| Aman Chadha | Apple |
| Indar Kumar | Ascentt Systems Inc |

## What the paper establishes

We audit the endpoint-product score of a prediction-trained linear factorized
model:

```text
S_lag = abs(W_out @ W_in_lag)
S_lag[target, source] = candidate strength of source -> target
```

Readable predictive weights can suggest hypotheses without identifying causal
structure. The contribution is a worked falsification protocol validated on
**one score family**, with controls for orientation, lag horizon, label
provenance, data budgets, intervention targets, and baseline comparisons.
Transfer to other interpretability signals remains untested.

## Quick start

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and Make.
Run from the repository root:

```bash
uv sync --frozen --extra baselines --extra dev
make test
make summary
make release
```

`make summary` prints paired follow-up metrics from saved results.
`make release` creates `dist/predictive_weights_audit.zip` and its SHA-256
checksum. The ZIP contains code, data, evidence, tests, and an internal
`SHA256SUMS` manifest, using an explicit file allowlist. Git history,
local workspace files, and manuscript assets are excluded. No TeX installation
is required. Building the archive does not rerun experiments.

## Corrected evidence

### Observational labels and benchmark orientation

Climate uses six NOAA indices and three literature-derived soft labels;
finance uses ten sector ETFs and six heuristic soft labels. Both are
sensitivity analyses, not causal leaderboards. Lorenz-96 uses an exact
30-edge continuous-time generating graph at `K=10, T=1500`; subsampling does
not guarantee the same discrete-time conditional-independence graph.

Corrected AUROC at maximum lag 3 on the original fixed datasets:

| Method | Climate | Finance | Lorenz-96 |
|---|---:|---:|---:|
| Lagged-BN | 0.626 ± 0.021 | 0.596 ± 0.051 | 0.881 ± 0.018 |
| Lasso | 0.617 | 0.500 | 0.973 |
| Ridge | 0.617 | 0.597 | 0.973 |
| PCMCI | 0.407 | 0.609 | 0.839 |
| PCMCI+ | 0.605 | 0.579 | 0.875 |
| Bivariate Granger | 0.605 | 0.512 | 0.989 |

Lagged-BN reports mean ± standard deviation across three initializations;
other entries are deterministic fits. This table is descriptive, not an
estimate of uncertainty across datasets. Finance Lasso returns all-zero
cross-variable scores. Ridge also selects grid-boundary penalties on climate
and finance, but its scores are untied.

The earlier preprint's PCMCI value of `0.986` used a transposed score matrix
and lag horizon 5; correcting orientation and matching the lag horizon gives
`0.839`. Earlier DYNOTEARS/VAR-LiNGAM values included contemporaneous
coefficients; their strictly lagged seed-0 AUROCs are `0.833` and `0.896`.

Evidence: [original benchmark](results/corrected_workshop/real_data/real_data_benchmark_results.json)
and [label provenance](results/corrected_workshop/real_data/edge_provenance_cards.json).

### Reviewer follow-up: ten independent trajectories

All eight methods receive identical Lorenz-96 trajectories. Lagged-BN averages
three initialization metrics within each trajectory. Entries below are means
with marginal 95% trajectory-bootstrap intervals; the last column is the
paired difference **method minus Lagged-BN**. These analyses are exploratory,
with no multiplicity-adjusted superiority claim.

| Method | AUROC [95% CI] | AUPRC [95% CI] | Paired Δ AUROC [95% CI] |
|---|---|---|---|
| Lagged-BN | 0.872 [0.854, 0.890] | 0.765 [0.731, 0.799] | — |
| Lasso | 0.955 [0.942, 0.968] | 0.933 [0.916, 0.949] | 0.083 [0.064, 0.100] |
| Ridge | 0.954 [0.941, 0.967] | 0.931 [0.914, 0.948] | 0.082 [0.063, 0.100] |
| PCMCI | 0.855 [0.841, 0.870] | 0.831 [0.820, 0.844] | -0.017 [-0.043, 0.008] |
| PCMCI+ | 0.844 [0.800, 0.885] | 0.836 [0.798, 0.874] | -0.028 [-0.065, 0.007] |
| Granger | 0.977 [0.963, 0.988] | 0.964 [0.944, 0.980] | 0.105 [0.094, 0.116] |
| DYNOTEARS | 0.800 [0.783, 0.816] | 0.851 [0.833, 0.866] | -0.073 [-0.094, -0.047] |
| VAR-LiNGAM | 0.900 [0.880, 0.917] | 0.845 [0.818, 0.868] | 0.027 [0.013, 0.042] |

Replication improves uncertainty reporting, but uses the same graph and
sampling configuration. It does not provide a harder benchmark. DYNOTEARS
and VAR-LiNGAM were evaluated only on Lorenz-96, with documented defaults;
this does not establish their causal validity on that system.

Evidence: [follow-up values](results/reviewer_followup/paper_values.json),
[primary runs](results/reviewer_followup/trajectory_replication.json), and
[optional baseline runs](results/reviewer_followup/optional_baselines.json).

### Interventions and nonlinearity

- Target-aware, size-matched random forcing improves all four tested methods
  at both dimensions. The gains are not bottleneck-specific or necessarily
  equal. Combined-arm AUROC is at least `.99`, so gains are bounded by each
  method's observational shortfall.
- The six-gate nonlinearity rule is not satisfied across dimensions at a
  common sample size. All six passing cells occur at `K=20, T=300`, with the
  same `+0.50` AUROC difference at every nonlinearity strength.
- Linear-BN attains AUROC/AUPRC `1.00` in all 240 fits. The apparent advantage
  comes from prediction-tuned Lasso returning all-zero cross-variable scores.
  At `T=600`, both methods attain AUROC `1.00`. This design cannot resolve a
  nonlinearity effect in either direction.

Evidence: [intervention semantics](results/corrected_workshop/intervention_semantics/intervention_sweep_results.json),
[intervention methods](results/corrected_workshop/intervention_methods/causal_baselines_results.json),
[nonlinearity cells](results/corrected_workshop/nonlinearity_confirmation.json),
and [Lasso diagnostic](results/reviewer_followup/lasso_diagnostic.json).
The experiment implementations define the complete gates, metrics, splits, and settings.

## Reproduction commands

The checked-in `uv.lock` pins the main environment. Optional DYNOTEARS and
VAR-LiNGAM runs use a separate Python 3.10.19 environment pinned by
`results/reviewer_followup/legacy-requirements.txt`.

| Command | Purpose |
|---|---|
| `make test` | Full test suite, including saved-evidence integrity checks |
| `make smoke` | Focused protocol, baseline, and release tests |
| `make summary` | Print paired follow-up metrics and bootstrap intervals |
| `make release` | Package code and evidence into a checksummed ZIP |
| `make workshop-evidence` | Rerun the original corrected experiment protocols |
| `make reviewer-evidence` | Rerun trajectory comparisons and the Lasso diagnostic |
| `make reviewer-optional` | Fit DYNOTEARS/VAR-LiNGAM to exported trajectory arrays |
| `bash reproduce_all.sh` | Check saved evidence, rerun all retained experiments, and build the release |

Run `reviewer-optional` after `reviewer-evidence`: it reads the exact exported
arrays. Experiment commands overwrite results, so use a separate checkout to
compare reruns with committed evidence. Integrity tests check the original
saved artifacts and should be run before overwriting them. Numeric or runtime
variation can change artifact hashes on a rerun. The original experiment
manifest remains historical provenance and is not relabeled as a new run.

An optional CPU Docker build is provided in [Dockerfile](Dockerfile). Its
`make all` command runs the tests and packages the code and evidence. Docker and a fresh
dependency installation were not exercised in the latest local verification.

## Repository contents and validation

- `experiments/`: eight retained experiment/helper modules and package init.
- `results/`: original corrected artifacts and exploratory reviewer follow-ups.
- `data/`: fixed input snapshots, loaders, and synthetic generators.
- `ssm_causal/`, `baselines/`: model, metric, protocol, and baseline dependencies.
- `tests/`: implementation, statistical, provenance, and release checks.
- `scripts/`: numerical summaries and code/evidence packaging.

The legacy package name `ssm_causal` is retained for compatibility. Older
modules needed by imports or recorded source hashes remain unchanged; obsolete
experiment entry points and separate anonymous/preprint builds were removed.

The original corrected and follow-up experimental artifacts are retained
unchanged, including their historical provenance and numerical summaries.
Some artifact paths retain `paper_values.json` as a historical filename;
these are machine-readable results, not manuscript files.
Full experimental reruns are separate from the repository tests.

Verified on 2026-10-17 from a clean Git snapshot: all 229 tests pass, every
tracked file is included in the code release, and all eight experiment CLIs
load successfully. The paired numerical summary exactly matches the saved
reference. Verification used the installed dependency environment; full
experiment reruns, a fresh dependency installation, and Docker were not run.

## Scope and release status

Only one interpretation score family and limited dimensions, sample sizes,
and graph settings were evaluated. Climate/finance labels are incomplete.
Passing the checklist does not establish causal identification; predictive
scores must not directly prescribe scientific interventions, financial trades,
or policy decisions.

Repository: `https://github.com/ankitlade12/predictive-weights-audit`.
Public release is pending; the repository rename and local cleanup do not
remove Git history. The manuscript and submission metadata are managed separately.

## Citation and license

Use [CITATION.cff](CITATION.cff) for the author list and workshop citation.
This is a non-archival workshop paper, not a main-conference proceedings paper.
Code is distributed under the [MIT license](LICENSE).
