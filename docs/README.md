# Docs

Current results are in the camera-ready paper
(`../paper/draft/main_camera_ready.pdf`) and these documents:

1. `CAMERA_READY_REVISION.md`
   Maps every reviewer concern to a change, documents the post-review
   evidence, and records how to reproduce and validate the release.

2. `EXP24_25_REAL_DATA_AND_CAUSAL_BASELINES.md`
   Corrected real-data benchmark (soft climate and finance labels, the exact
   Lorenz-96 graph) and the intervention method contrasts.

3. `EXP18_19_20_RESULTS.md`
   Exploratory Exp 18-20 results and their corrected interpretation, including
   the nonlinearity confirmation.

Earlier research notes and the original state-space-model paper drafts were
removed when the camera-ready version was prepared; they remain in Git
history. Where an older statement disagrees with the camera-ready paper, the
paper is authoritative.

Benchmark artifacts:

- Edge provenance records live in `data/real_loaders.py`.
- Provenance-card and leave-one-group-out utilities live in
  `ssm_causal/benchmark_metadata.py`.
- The submitted provenance cards are in
  `results/corrected_workshop/real_data/edge_provenance_cards.json`.
- `make smoke` runs bounded CPU-friendly empirical verification.
- `make workshop-evidence` reproduces the corrected workshop artifact set;
  `reproduce_all.sh` and `make falsification` run the broader pipeline.
