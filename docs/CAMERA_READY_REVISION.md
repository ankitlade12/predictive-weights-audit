# Camera-ready revision record — 2026-10-10

Paper: **When Predictive Weights Look Causal: An Audit Protocol for
Interpretability-Based Discovery**. Accepted at the non-archival NeurIPS 2026
Workshop on Interpretability for Discovery. The supplied decision contains
no explicit revision conditions. This record maps the reviews to changes;
it is not a response posted to OpenReview.

## Deliverables and scope

- Final workshop source: `paper/draft/main_camera_ready.tex`.
- Final PDF: `paper/draft/main_camera_ready.pdf`.
- Shared revised text: `camera_ready_content.tex` and `camera_ready_appendix.tex`.
- Preprint: `paper/draft/main_arxiv.tex`, sharing the revised content.
- Historical anonymous source: `paper/draft/main_4page.tex`; existing local
  figure edits were preserved. It is not the revised final PDF.
- Submitted experiment artifacts remain in `results/corrected_workshop/`.
- Post-review artifacts are separate in `results/reviewer_followup/`.
- Release ZIP: `dist/camera_ready_release.zip`, with per-file SHA-256 checksums.

Author order and affiliations follow the accepted OpenReview entry, as
confirmed by the corresponding user: Ankit Hemant Lade (University of Central
Missouri), Sai Krishna Jasti (Ascentt Systems Inc), Aman Chadha (Google),
Indar Kumar (Ascentt Systems Inc). The outdated citation metadata was corrected
to name all four authors and the workshop, without claiming a main-conference
proceedings publication.

## Every reviewer concern

| Reviewer | Concern | Revision and evidence | Remaining limit |
|---|---|---|---|
| AH3Q, CqQD | General protocol claim exceeds one score/model family | Abstract, introduction, limitations, conclusion explicitly separate reusable recommendations from validation on one endpoint-product score family | No second interpretation signal tested |
| AH3Q | No working discovery method | Preserve explicit statement that this is a negative-results audit, not a proposed discovery estimator | No new positive discovery method claimed |
| AH3Q, CqQD | Small dimensions, sparse soft labels, few benchmarks | Correct scope: interventions use one T, nonlinearity uses two; disclose two planted cross-variable edges and retain soft-label restrictions | No larger K, harder graph family, or stronger observational labels |
| AH3Q, CqQD | DYNOTEARS and VAR-LiNGAM only fixture-tested | Both actually run on original Lorenz-96 data plus ten new trajectories in isolated Python 3.10; settings and limitations in Appendix B | Neither added baseline run on climate, finance, interventions, or nonlinearity; defaults are not equal tuning budgets |
| CqQD | Missing attention/saliency and interpretability literature | Add Jain & Wallace, Adebayo et al., Wiegreffe & Pinter, Rudin; explain causal-discovery-specific contribution in related work | No claim to originate the general readability warning |
| CqQD | Near-ceiling Granger makes graph too easy | Explicitly acknowledge this; limit rankings to this configuration; report independent trajectories and AUPRC | Replication does not make the graph harder; a harder graph remains future work |
| CqQD | Nonlinearity rules hard to reconstruct | Main text gives six gates, 24-test family, same-strength/common-T regime rule; Appendix A gives all 24 rows, split, prevalence, MSE definition and failed gates | Gates are those originally fixed, not retuned after review |
| CqQD | Initialization spread versus deterministic points | Table 1 explicitly descriptive; new Table 2 supplies trajectory-level CIs and paired differences for all eight methods | Ten trajectories of the same graph; intervals are marginal, not simultaneous |

## Additional issues found and addressed

1. Section 2's general max-over-lags description obscured the nonlinearity
   experiment's lag-resolved primary metric. Both universes are now explicit.
2. All 240 Lasso fits were reproduced, matching penalty, AUROC and test MSE.
   All 60 K=20,T=300 fits have zero off-diagonal lag scores at alpha=1.
   At T=600 all 120 fits choose alpha=.1 and reach AUROC=1. This supports a
   concrete diagnostic, not a universal explanation or a license to retune
   against graph labels. The counts concern cross-variable scores, not self
   coefficients.
3. The linear model's width 32 exceeds K in the nonlinearity experiment; it
   is factorized but not strictly output-rank limited. The revised text
   avoids attributing the result specifically to low rank.
4. Lasso is MultiTaskLasso without intervention masks and separate outcome
   regressions with masks. The implementation is now disclosed.
5. AUPRC is trapezoidal PR area, not average precision. Tied-score interpolation
   can differ from prevalence; the prevalence gate is a heuristic, not a
   calibrated null test.
6. All-zero Wilcoxon differences are unevaluable in the saved results, enter
   Holm conservatively as p=1, and cannot pass the gate. Appendix tables
   preserve their missing p-values rather than inventing a measured p-value.
7. The Lorenz-96 reference is the continuous-time generating interaction graph;
   it need not equal a discrete-time conditional-independence graph after
   subsampling.
8. Replaced “null” with “does not reject” for the K=10 intervention contrast.
   Gains across four methods are not evidence of equality or universality.
9. Passing an audit checklist does not establish causal identification.

## Post-review experiment protocol

These are exploratory reviewer follow-ups, not pre-registered confirmations.
Seeds and estimator settings were fixed in the follow-up script before its
first execution. No original gate or graph-label selection rule was changed.

Ten Lorenz-96 trajectories use data seeds 81000–81009, K=10, T=1500,
F=10, integration step .01, subsampling 10, and the original full-series
standardization. The bottleneck averages model seeds 42–44 inside each
trajectory. Every method receives the identical exported arrays; hashes are
checked before constructing the comparison. The original seed-0 array is
also regenerated and checked against the submitted data/graph hashes.

The original six methods use the existing locked environment. CausalNex
0.12.1 and LiNGAM 1.11.0 use a separate Python 3.10.19 environment, whose full
installed versions are recorded in `legacy-requirements.txt`. Defaults are
documented in Appendix B, including VAR-LiNGAM's BIC selection within maximum
lag 3. No causal validity is assumed for these linear models on Lorenz-96.

All 22 optional-baseline fits completed without captured warnings. The actual
installed adapters also passed a planted one-way lagged-edge check; all
returned diagonals were zero. Existing adapter fixture tests remain part of
the repository suite.

Table 2 is generated from paired trajectories, using 10,000 percentile
bootstrap resamples with seed 82000. Means and paired contrasts are reported
with marginal 95% intervals; no adjusted superiority tests are claimed.

## Reproduction

From the repository root (or extracted release archive):

```sh
uv sync --frozen --extra baselines --extra dev
uv run --frozen --extra baselines --extra dev python experiments/exp28_reviewer_followup.py primary
uv run --frozen --extra baselines --extra dev python experiments/exp28_reviewer_followup.py diagnose
uv run --no-project --python 3.10.19 --with-requirements results/reviewer_followup/legacy-requirements.txt python experiments/exp28_reviewer_followup.py optional
make paper-camera-ready
make paper-arxiv
make test
python3 scripts/build_camera_ready_release.py
```

The optional stage must follow the primary stage because it reads the exact
exported arrays. Building the PDFs uses saved artifacts and does not require
re-running experiments or installing the legacy environment. TeX requires
pdflatex, bibtex, and the packages listed in the existing paper-build guide.

Generated table code rejects missing, duplicated, mismatched-input, or failed
replicates. Regression tests exercise these failure modes and verify joining
by data seed rather than list order. Source hashes are recorded in the new
experiment artifacts; archive checksums cover all packaged files.

## Completed validation

- Full repository suite: **244 passed** (seven existing dependency warnings).
- After the final archive-builder adjustment: **15 focused release tests passed**.
- The same **15 release tests passed from the extracted ZIP** without Git
  metadata. All 157 packaged-file checksums verified after extraction.
- The extracted release rebuilt the final PDF and regenerated byte-identical
  table sources using the existing locked runtime and TeX installation.
- Ruff checks pass for the new experiment, table generator, release builder,
  updated anonymous builder, and new tests; `git diff --check` passes.
- Workshop and preprint PDFs both compile: **five main-text pages, one
  reference page, two appendix pages**. No undefined references/citations or
  overfull boxes. The benign underfull-page warning concerns appendix layout.
- Rendered pages inspected, including both result tables and final author metadata.
- All seven submitted artifact hashes still match their original manifest.
- All source hashes recorded by the three post-review experiment stages
  match the delivered source. The seed-0 data and graph match the original
  benchmark hashes; optional and primary runs use matching exported arrays.

## Publication logistics still requiring the authors' attention

The [official CFP](https://interpretability4discovery.github.io/cfp.html),
checked 2026-10-10, allows six main-text pages for accepted papers; references
and appendices are excluded. Accepted papers are presented as posters, with
a subset selected for talks. In-person attendance is encouraged, not required.

The public CFP does **not** list a camera-ready deadline. Check the acceptance
email/OpenReview author task for that deadline, the actual upload field, and
any later organizer instructions. Poster dimensions, upload deadlines, and
whether this paper received a spotlight were not provided. No upload,
submission, organizer message, public release, or repository push was made.

The broader benchmark and interpretation-signal limitations are addressed
through bounded claims, not reported as experimentally resolved.
