# Docs

This folder is the home for the project's research-planning and strategy documents.

Suggested reading order:

1. `EXP24_25_REAL_DATA_AND_CAUSAL_BASELINES.md`
   Current real-data benchmark and causal-baseline summary. Read this first
   for the reviewer-facing falsification story: corrected NOAA ground truth,
   Lorenz-96, PCMCI/Granger, and intervention method contrasts.

2. `RESEARCH_WRAP_UP_2026-04-16.md`
   Final follow-up wrap-up after the intervention controls and real-data
   causal-baseline additions.

3. `RESEARCH_WRAP_UP_2026-04-08.md`
   Final project wrap-up: what was tested, what survived, what failed, and why
   method-building is now closed.

4. `EXP17_CAUSEME_LAG_GATE_RESULTS.md`
   Final lag-specific semi-real benchmark gate for explicit lag-resolved
   bottleneck extraction.

5. `neurips_gap_analysis.md`
   Broad audit of the original project, including novelty, reviewer risks,
   and why the initial thesis was not yet NeurIPS-ready.

6. `RESEARCH_STATUS.md`
   Short honest snapshot of where the project stands after the first round
   of experiments and scientific corrections.

7. `RESULTS.md`
   Written summary of the main experiment outcomes, especially the go/no-go
   findings that weakened the original SSM-specific thesis.

8. `OPTION_A_PIVOT.md`
   The proposed broader pivot toward bottleneck prediction models.

9. `RESEARCH_RESET_PLAN.md`
   Historical reset document plus status updates showing how the later method
   branches were tested and ultimately closed.

10. `PHASE2_RESULTS.md`
   Robust validation showing why causality-preserving bottleneck training was
   stopped as the active method-building direction.

11. `PIVOT_EXPERIMENTS_RESULTS.md`
   Pivot experiment summary showing the broader bottleneck extraction signal
   and architecture ranking.

12. `LAG_RESOLVED_RESULTS.md`
   Exp 13 results for Jacobian/impulse lag probing of existing SSM models.

13. `EXP15_LAGGED_BOTTLENECK_STRESS_RESULTS.md`
   Exp 15 results for explicit lagged bottleneck extraction against tuned
   lagged linear baselines.

14. `RESEARCH_DECISION_MEMO_2026-04-07.md`
    Intermediate decision memo before the final Exp 17 gate; now preserved as
    part of the reasoning chain rather than the final status.

15. `EXP15_EXP16_GPU_RESULTS.md`
    GPU stress-grid summary plus the intermediate semi-real aggregate gate.
    Read with the correction note pointing to Exp 17.

Current benchmark-hardening artifacts:

- Edge provenance records live in `data/real_loaders.py`.
- Provenance-card and leave-one-group-out utilities live in
  `ssm_causal/benchmark_metadata.py`.
- Exp 24 writes `results/real_data_benchmark/edge_provenance_cards.json`.
- `make smoke` runs bounded CPU-friendly empirical verification.
- `make workshop-evidence` reproduces the compact corrected workshop artifact
  set; `reproduce_all.sh` and `make falsification` run the broader F1-F6
  pipeline.
