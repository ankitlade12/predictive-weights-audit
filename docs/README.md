# Docs

This folder is the home for the project's research-planning and strategy documents.

Suggested reading order:

1. `neurips_gap_analysis.md`
   Broad audit of the original project, including novelty, reviewer risks,
   and why the initial thesis was not yet NeurIPS-ready.

2. `RESEARCH_STATUS.md`
   Short honest snapshot of where the project stands after the first round
   of experiments and scientific corrections.

3. `RESULTS.md`
   Written summary of the main experiment outcomes, especially the go/no-go
   findings that weakened the original SSM-specific thesis.

4. `OPTION_A_PIVOT.md`
   The proposed broader pivot toward bottleneck prediction models.

5. `RESEARCH_RESET_PLAN.md`
   The current reset document: reflection on what happened so far plus the
   next planned research direction centered on a stronger method contribution.

6. `PHASE2_RESULTS.md`
   Robust validation showing why causality-preserving bottleneck training was
   stopped as the active method-building direction.

7. `PIVOT_EXPERIMENTS_RESULTS.md`
   Pivot experiment summary showing the broader bottleneck extraction signal
   and architecture ranking.

8. `LAG_RESOLVED_RESULTS.md`
   Exp 13 results for Jacobian/impulse lag probing of existing SSM models.

9. `EXP15_LAGGED_BOTTLENECK_STRESS_RESULTS.md`
   Exp 15 results for explicit lagged bottleneck extraction against tuned
   lagged linear baselines.

10. `RESEARCH_DECISION_MEMO_2026-04-07.md`
    Current decision memo: lagged bottleneck is useful tooling/supporting
    evidence, but not a standalone method claim yet; next gate is real/semi-real
    benchmarking.

Planned additions:

- implementation notes and experiment logs for the reset-phase work.
