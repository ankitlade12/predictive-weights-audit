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

Author order follows the accepted OpenReview entry: Ankit Hemant Lade
(University of Central Missouri), Sai Krishna Jasti (Independent Researcher),
Aman Chadha (Google), Indar Kumar (Ascentt Systems Inc). The outdated citation metadata was corrected
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

## Completed validation (second round)

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

## Venue requirements

The [official CFP](https://interpretability4discovery.github.io/cfp.html) and
the acceptance notice allow six main-text pages for the camera-ready version;
references and appendices are excluded. Accepted papers are presented as
posters (34 x 34 inches recommended), with a subset selected for spotlight
talks.

The broader benchmark and interpretation-signal limitations are addressed
through bounded claims, not reported as experimentally resolved.

## Second audit

The final audit checks a clean Git snapshot as well as the packaged archive.
It found that the finance CSV was present in the release archive but ignored
by Git. The exact benchmark snapshot is now versioned, so a fresh checkout
does not need a potentially changed download to rebuild the release.
Regression checks verify the loaded finance data and graph against the
submitted artifact hashes, and verify exported trajectory hashes against
both follow-up stages. Two minor prose errors were corrected. Main-text
methods now explicitly distinguish the original Holm-adjusted tests from
exploratory follow-up intervals, and clarify VAR-LiNGAM's lag selection.
The final paper links directly to the code and artifact repository.

All 24 original nonlinearity cell summaries (including bootstrap intervals,
Holm adjustment, gates, and the regime outcome) were recomputed exactly from
saved replicates. All 22 optional-baseline AUROC/AUPRC pairs were recomputed
from saved score matrices. No numerical changes were needed. A clean Git
snapshot built both PDFs and the release archive; 23 focused reproduction
and release tests passed there. Original artifact and follow-up source hashes
remain valid.

## Third revision round

A further audit against the verbatim reviews (AH3Q, CqQD) and the earlier
preprint found these issues; all are fixed in the camera-ready sources.

Corrections to the record above:

- The repository was private when this record called it public. The paper
  now links `https://github.com/ankitlade12/predictive-weights-audit`, so the
  repository must be renamed and made public before upload.
- The finance array-hash test failed on some x86-64 machines: float32 log
  returns differ in the last ULP. The test now pins the CSV file bytes and
  checks the graph hash; every finance AUROC was reproduced exactly.
- The release builder failed on a fresh clone because the figure is
  generated. `make release` now builds both PDFs before packaging.

Paper changes:

- Bibliography: corrected the author lists of `runge2019pcmci` and
  `pamfil2020dynotears`; added PCMCI+ (Runge 2020), which appeared in both
  tables uncited; updated Tank et al. (2022) and Mamba (COLM 2024); fixed
  entry types and capitalization. The self-citation lives in
  `paper/camera_ready_references.bib` so the anonymous archive stays
  identity-free.
- The earlier preprint (arXiv:2605.09169) is now cited wherever its values
  appear. Its PCMCI 0.986 combined a transposed score matrix with lag horizon
  5, so the abstract no longer attributes the change to a transpose alone.
  Its DYNOTEARS (0.983) and VAR-LiNGAM (0.968) values included contemporaneous
  coefficients; Appendix B states that the strictly lagged values supersede
  them.
- Nonlinearity: Linear-BN attains AUROC and AUPRC 1.00 in all 240 fits, so
  the gate-passing +0.50 comes entirely from the prediction-tuned Lasso
  selecting the largest grid penalty and returning all-zero cross scores.
  The abstract and Section 3.3 now say so. The undefined term "isolated
  positive" is removed. The tested hypothesis (the preprint's mild-nonlinearity
  finding) is stated.
- Checklist item (iv) adds a comparator non-degeneracy check (no all-tied
  scores, no selected penalty at a grid boundary). Finance Lasso in Table 1
  is flagged as the same degenerate case.
- Section 3.2 describes the VAR(3) intervention generator and defines the
  gain before the semantics audit uses it. The numeric intervention table
  dropped after submission is restored as Appendix C.
- Section 2 states that the predictor is linear with width 16 (32 in the
  nonlinearity study) and unifies the bottleneck labels. Table 2 states the
  sign of its difference column, and the text summarizes it, including that
  Lagged-BN has the lowest AUPRC of all eight methods.
- Author block with emails, Acknowledgments and Disclosure of Funding, PDF
  metadata, Latin Modern typewriter font (no Type 3 bitmap fonts), larger
  Figure 1 text, and a deterministic figure timestamp.

- A final independent review added: disclosure that Ridge also selects
  grid-boundary penalties on climate and finance (checklist item (iv) now asks
  to report such penalties rather than forbid them); a note that combined-arm
  AUROC is at least .99, so intervention gains are bounded by observational
  shortfall; the definition of sigma in the nonlinearity generator; the exact
  per-lag edge probability of the intervention generator; consistent
  Lagged-BN labels in Figure 1, Table 4 and the text; and a finance test that
  checks the loaded array against a float64 recomputation.

- Repository cleanup: superseded research notes and planning documents, the
  original state-space-model paper sources and figures, the legacy table and
  figure generators with their Makefile targets, the Colab notebook, and the
  AWS/Drive helper scripts were removed from the tree (they remain in Git
  history). `LICENSE` now names all four authors.

Validation after this round: the full suite passes (246 tests). Both PDFs
build to six main-text pages, two reference pages, and three appendix pages,
with no undefined references, overfull boxes, bibliography warnings, or Type 3
fonts. A fresh-clone simulation of the change passes `make release`, the
release tests from the extracted archive, and the anonymous-supplement
identity audit; all recorded follow-up source hashes still match.

Still requiring author confirmation: Aman Chadha's affiliation line, the
funding/competing-interest statement, and the repository rename.

OpenReview abstract field (plain text, matching the PDF):

> Readable model weights are tempting evidence about the mechanisms that
> generated scientific data. We audit one such candidate: the endpoint-product
> score of a next-step time-series predictor. A sequence of initially
> plausible architecture, benchmark, intervention, and nonlinearity stories
> does not survive controls for direction conventions, lag horizons, label
> provenance, data budgets, intervention targets, and classical baselines.
> Correcting a source/target transpose and matching lag horizons changes
> Lorenz-96 PCMCI AUROC from 0.986 in our earlier preprint to 0.839. On
> target-aware, size-matched random forcing, all four tested methods improve,
> so the effect is not bottleneck-specific. In a fixed nonlinearity
> confirmation, the only gate-passing bottleneck advantage is identical at
> every nonlinearity strength, vanishes when the sample size doubles, and
> arises because the prediction-tuned Lasso comparator returns all-zero
> cross-variable scores. We demonstrate a reproducible falsification protocol
> on one endpoint-product score family, with reporting and responsible-use
> boundaries; transfer to other interpretability signals remains untested.
> This negative study illustrates how a readable predictive representation
> can remain useful for hypothesis generation without constituting causal
> knowledge.
