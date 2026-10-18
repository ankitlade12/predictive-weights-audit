# When Predictive Weights Look Causal: An Audit Protocol for Interpretability-Based Discovery

Code and experimental evidence for the accepted NeurIPS 2026 Workshop on
Interpretability for Discovery study (non-archival).

[Citation](CITATION.cff) · [MIT license](LICENSE)

## Paper identity

- **Title:** When Predictive Weights Look Causal: An Audit Protocol for Interpretability-Based Discovery
- **Venue:** NeurIPS 2026 Workshop on Interpretability for Discovery (Interp4Discovery).
- **Status:** Accepted; non-archival workshop contribution.
- **Contribution:** Negative results and a worked audit of causal interpretations of predictive weights.

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

`Lagged-BN` and `Linear-BN` denote the linear factorized predictor, with no
hidden activation. The historical term “bottleneck” does not imply a strict
rank restriction: width is 16 in the main comparisons and 32 in the
nonlinearity study, and can exceed the variable count. Self-edges are
excluded. Observational scores aggregate positive lags by their maximum;
the nonlinearity experiment instead retains lag identity.

The reusable audit checks are:

1. Verify direction, lag, and diagonal conventions on planted examples.
2. Match usable observations, preprocessing, trajectory boundaries, and seeds.
3. Encode intervention targets and distinguish intervention semantics.
4. Compare classical baselines under the same estimand; check for all-tied
   scores and report penalties selected at grid boundaries.
5. Separate exact simulation graphs from incomplete observational labels.
6. Fix evaluation rules before graph evaluation, account for multiple testing,
   and retain failures and contradictory outcomes.

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

Climate uses six NOAA indices (`T=757`) and three literature-derived soft labels;
finance uses ten sector ETFs (`T=1893`) and six heuristic soft labels. Both are
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

### Trajectory replication: ten independent trajectories

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

Evidence: [follow-up values](results/followup/paper_values.json),
[primary runs](results/followup/trajectory_replication.json), and
[optional baseline runs](results/followup/optional_baselines.json).

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
and [Lasso diagnostic](results/followup/lasso_diagnostic.json).
The experiment implementations define the complete gates, metrics, splits, and settings.

### Statistical and experimental details

The intervention studies use `K ∈ {10,20}`, `T=300`, and ten independent data
seeds. Forced target outcomes are excluded from fitting on their intervention
segments, and lag windows never cross segment boundaries. The semantics
comparison uses lag-resolved AUROC; the four-method comparison aggregates lags.

The nonlinearity grid is `K ∈ {10,20}`, `T ∈ {300,600}`, and
`s ∈ {0,.2,.3,.4,.6,1}`: 24 cells with ten paired data seeds each. Its two
planted cross-variable edges are at lags 3 and 4; evaluation uses horizon 5.
Data are split chronologically 80/20, with standardization fitted on training
data only. Lasso selects its penalty using prediction validation, not graph
labels. Without outcome masks it is MultiTaskLasso; target-aware intervention
fits use separate outcome regressions.

A positive cell must meet all six original gates:

1. Mean paired AUROC difference ≥ `.03`.
2. Paired bootstrap 95% lower bound > `0`.
3. One-sided Wilcoxon Holm-adjusted `p < .05`, across all 24 cells.
4. Mean bottleneck AUPRC greater than Lasso AUPRC.
5. Mean bottleneck AUPRC greater than edge prevalence.
6. Mean bottleneck test MSE ≤ `1.10 ×` mean Lasso test MSE.

A regime requires the **same two or more** strengths from `{.2,.3,.4}` to
pass at both dimensions at one common `T`. None satisfies this rule. AUPRC
is trapezoidal precision–recall area, not average precision; the prevalence
gate is a heuristic floor, not a calibrated null test. All-zero Wilcoxon
contrasts are unevaluable, enter Holm as `p=1`, and cannot pass.

The follow-up uses data seeds `81000–81009`, model seeds `42–44`, and 10,000
bootstrap resamples with seed `82000`. Independent trajectories—not model
initializations—are the inference units. Its confidence intervals are
marginal, and its analyses were exploratory and not pre-registered.

## Reproduction commands

The checked-in `uv.lock` pins the main environment. Optional DYNOTEARS and
VAR-LiNGAM runs use a separate Python 3.10.19 environment pinned by
`results/followup/legacy-requirements.txt`.

| Command | Purpose |
|---|---|
| `make test` | Full test suite, including saved-evidence integrity checks |
| `make smoke` | Focused protocol, baseline, and release tests |
| `make summary` | Print paired follow-up metrics and bootstrap intervals |
| `make release` | Package code and evidence into a checksummed ZIP |
| `make workshop-evidence` | Rerun the original corrected experiment protocols |
| `make followup-evidence` | Rerun trajectory comparisons and the Lasso diagnostic |
| `make followup-optional` | Fit DYNOTEARS/VAR-LiNGAM to exported trajectory arrays |
| `bash reproduce_all.sh` | Check saved evidence, rerun all retained experiments, and build the release |

Run `followup-optional` after `followup-evidence`: it reads the exact exported
arrays. Experiment commands overwrite results, so use a separate checkout to
compare reruns with committed evidence. Integrity tests check the original
saved artifacts and should be run before overwriting them. Numeric or runtime
variation can change artifact hashes on a rerun. The original experiment
manifest remains historical provenance and is not relabeled as a new run.

An optional CPU Docker build is provided in [Dockerfile](Dockerfile). Its
`make all` command runs the tests and packages the code and evidence. Docker and a fresh
dependency installation were not exercised in the latest local verification.

## Citation and license

Ankit Hemant Lade, Sai Krishna Jasti, Aman Chadha, and Indar Kumar (2026).
“When Predictive Weights Look Causal: An Audit Protocol for
Interpretability-Based Discovery.” NeurIPS 2026 Workshop on Interpretability
for Discovery (Interp4Discovery).

Use [CITATION.cff](CITATION.cff) for the author list and workshop citation.
This is a non-archival workshop paper, not a main-conference proceedings paper.
Code is distributed under the [MIT license](LICENSE).
