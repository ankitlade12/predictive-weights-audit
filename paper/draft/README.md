# Paper Builds

The current paper is an audit of causal interpretations of predictive
bottleneck weights. Its contribution is the falsification protocol and the
corrected evidence boundary, not a new causal-discovery method.

## Files

- `main_camera_ready.tex` — accepted workshop revision; author order follows
  the accepted OpenReview entry; final workshop template
- `camera_ready_content.tex` — revised scientific content shared by the final
  workshop version and preprint
- `camera_ready_appendix.tex` — all 24 nonlinearity cells, inference rules,
  added-baseline configuration, intervention gains, and evidence boundaries
- `generated_reviewer_results.tex` — post-review artifact-derived tables

- `main_4page.tex` — anonymous Interpretability for Discovery submission
- `main.tex` — superseded long-form draft of the earlier preprint; not built
- `main_arxiv.tex` — preprint sharing the camera-ready content and appendices
- `generated_workshop_results.tex` — artifact-derived values; do not edit manually
- `../neurips_2026.sty` — official NeurIPS 2026 workshop style
- `../references.bib` — shared bibliography
- `../camera_ready_references.bib` — self-citation of the earlier preprint,
  kept out of the anonymous archive

The historical four-page workshop source uses `dblblindworkshop`, declares
`\workshoptitle{Interpretability for Discovery}`, and keeps the main text to
four pages; references follow on a fifth page.

## Evidence Boundary

- Climate and finance have sparse soft labels and are sensitivity analyses.
- Lorenz-96 has an exact simulated graph; corrected linear and Granger
  baselines outperform the predictive bottleneck in this configuration.
- Target-aware random forcing improves all four tested estimators, so the
  effect is not bottleneck-specific.
- The fixed nonlinearity gate does not identify a common mild-nonlinearity
  regime; its only passing cells come from a prediction-tuned Lasso that
  returns all-zero cross-variable scores at `K=20, T=300`.
- Readable endpoint weights are suitable for hypothesis generation, not
  intervention, trading, or mechanism claims without further validation.

Submitted-experiment values are generated from `results/corrected_workshop/`
by `paper/generate_workshop_results.py`; post-review values (Table 2 and the
appendix gate table) come from `results/reviewer_followup/` via
`paper/generate_reviewer_results.py`.

## Building

For a native Debian/Ubuntu build, install the required LaTeX packages once:

```bash
sudo apt-get install texlive-latex-base texlive-latex-extra \
  texlive-fonts-recommended texlive-bibtex-extra lmodern
```

From the repository root:

```bash
make paper-camera-ready
make paper-arxiv
make release
make paper-workshop
make anonymous-supplement
make submission-audit
```

`make release` writes both final PDFs and
`../../dist/camera_ready_release.zip`. The historical anonymous workshop PDF is
`main_4page.pdf`, with the identity-scanned review archive
`../../dist/anonymous_workshop_supplement.zip`.

The accepted revision builds to `main_camera_ready.pdf`. The workshop CFP
allows six main-text pages for accepted papers, excluding references and
appendices: https://interpretability4discovery.github.io/cfp.html (checked
2026-10-10). The final submission deadline is not listed there. See
`../../docs/CAMERA_READY_REVISION.md` for the post-review evidence and release
checks. Do not upload the historical anonymous PDF as the final revision.
