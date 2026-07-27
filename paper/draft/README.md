# Paper Builds

The current paper is an audit of causal interpretations of predictive
bottleneck weights. Its contribution is the falsification protocol and the
corrected evidence boundary, not a new causal-discovery method.

## Files

- `main_4page.tex` — anonymous Interpretability for Discovery submission
- `main_arxiv.tex` — author-identified preprint with identical scientific claims
- `generated_workshop_results.tex` — artifact-derived values; do not edit manually
- `../neurips_2026.sty` — official NeurIPS 2026 workshop style
- `../references.bib` — shared bibliography

The workshop source uses `dblblindworkshop`, declares
`\workshoptitle{Interpretability for Discovery}`, and keeps the main text to
four pages; references follow on a fifth page.

## Evidence Boundary

- Climate and finance have sparse soft labels and are sensitivity analyses.
- Lorenz-96 has an exact simulated graph; corrected linear and Granger
  baselines outperform the predictive bottleneck in this configuration.
- Target-aware random forcing improves all four tested estimators, so the
  effect is not bottleneck-specific.
- The fixed nonlinearity gate does not identify a common mild-nonlinearity
  regime.
- Readable endpoint weights are suitable for hypothesis generation, not
  intervention, trading, or mechanism claims without further validation.

All reported current values are generated from
`results/corrected_workshop/` by `paper/generate_workshop_results.py`.

## Building

From the repository root:

```bash
make paper-workshop
make paper-arxiv
make anonymous-supplement
make submission-audit
```

The workshop PDF is `main_4page.pdf`. The deterministic, identity-scanned
review archive is `../../dist/anonymous_workshop_supplement.zip`. The public
repository and the author-identified preprint must not be linked in the
double-blind submission.
