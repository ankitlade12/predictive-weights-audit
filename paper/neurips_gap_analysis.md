# NeurIPS 2026 Gap Analysis for SSM-Causal

Date: 2026-04-02

Scope: honest internal assessment of the current repo, the scientific idea, the likely novelty, and what would need to become true for this project to be competitive for a NeurIPS 2026 submission.

## Executive Summary

Short version:

- The project has a real research signal.
- The strongest current idea is not "SSMs solve causal discovery" in general.
- The potentially interesting claim is much narrower: a prediction-trained Mamba or selective SSM may encode useful lagged causal structure in a way that can be extracted post hoc, possibly with a very cheap score such as `|W_out W_in|`.
- In its current state, this repo is not NeurIPS-ready.
- The main blockers are not polish. They are scientific validity, claim discipline, and benchmark quality.

My current recommendation:

- Continue the project.
- Do not frame it as submission-ready yet.
- Treat the next phase as a falsification campaign.
- Only keep the NeurIPS target if the next experiments show that the effect is real, robust, fair, and meaningfully specific to SSMs.

Current confidence:

- Confidence that there is a publishable idea somewhere here: medium.
- Confidence that the current version is strong enough for NeurIPS: low.
- Confidence that this could become a strong workshop paper or focused methods paper with moderate additional work: medium to high.

## One-Sentence Honest Assessment

This is a promising post hoc causal screening idea built around selective state space models, but the current evidence base is too synthetic, the theory is too weak, and the evaluation protocol has too many reviewer attack surfaces for a top-tier NeurIPS submission.

## What Seems Potentially New

The likely new part is not "Mamba for causality" by itself. That space is already active. The potentially new part is the following combination:

1. Train a standard sequence model on next-step prediction only.
2. Do not use a causal objective, sparsity penalty, or explicit graph learning.
3. Extract a variable-level causal score from the trained model after training.
4. Show that a selective SSM, and maybe specifically Mamba-style routing, makes that post hoc extraction unusually effective.

If the endpoint projection score

`S = |W_out W_in|`

really works reliably and better than matched non-SSM architectures, that is interesting.

If the only thing that works is a generic Jacobian saliency score, then the novelty drops a lot, because Jacobian-based influence analysis is already a known direction in neural Granger work.

## What Is Already Strong in the Repo

The repo is not empty or hand-wavy. It already has:

- A real modeling implementation in `ssm_causal/mamba_model.py`.
- A concrete core extraction story in `experiments/exp0_core_validation.py`.
- A paper draft with method, related work, theory, and experiments.
- Multiple experiment tracks: synthetic, non-stationary, climate, fMRI, efficiency, architecture comparison, ablation.
- Baseline wrappers for Granger, PCMCI, and cLSTM.
- An internal understanding that the cleanest theoretical object (`B/C` extraction) is not yet the strongest empirical object.

These are real assets. This is enough to justify spending more time.

## What Would Get Rejected Right Now

If submitted in its current scientific form, I expect major reviewer pushback on the following:

1. Claim inflation relative to the evidence.
   The paper and README imply that causal structure is read directly from SSM internals with a single clean principle. In practice, the best-performing extractors currently appear to be endpoint projection and Jacobian variants, not a clean state-space-internal object.

2. Real-world evidence is not yet strong enough.
   The climate and fMRI experiments are currently synthetic by default, and when real data is used the "ground truth" is a sparse literature-derived edge list rather than benchmark-standard causal ground truth.

3. Oracle model/variant selection.
   The climate and fMRI scripts currently choose the better of Jacobian and weight-projection after looking at AUROC. That is not acceptable for a main result.

4. The theory is not rigorous enough for the strength of the theorem statement.
   The proof relies on a genericity argument that effectively bypasses the complicated part of the model. A skeptical reviewer will say the theorem is about a bottleneck factorization, not about Mamba.

5. Baseline fairness is vulnerable.
   The Granger baseline is bivariate rather than multivariate. PCMCI is run in a simple linear mode. The cLSTM baseline is described in the code as simplified rather than exact. That weakens every comparison.

6. Architecture specificity is unproven.
   A reviewer will immediately ask: why should this work because of SSMs rather than because any predictor with an input and output bottleneck has to route predictive information?

7. Causal language is too strong for observational predictive extraction.
   Even if the method recovers Granger structure, strong wording about "causal discovery" on real observational systems will attract criticism around confounding, latent variables, and contemporaneous effects.

## Most Important Scientific Question

The single most important question for the project is this:

Does the extraction work because selective SSM dynamics genuinely preserve causal routing better than other predictors, or does it work because any competent forecasting model has to encode predictive dependencies in its endpoint weights or Jacobian?

If the answer is "any model does this," the project becomes much less interesting.

If the answer is "SSMs do this more cleanly, more robustly, or more efficiently than matched architectures," the project becomes much more interesting.

## Current Repo Risks by Area

### 1. Core Claim Risk

Current draft story:

- SSM internals encode causal structure.
- A single matrix product can recover the graph.

Current repo reality:

- Weight projection appears to be the cleanest simple score.
- Jacobian extraction is treated as a strong practical method.
- `B/C`-based extraction is not yet convincingly established as the true core mechanism.

Interpretation:

- The project may be drifting from "state-space internals explain causality" toward "a trained predictor contains usable predictive routing signals."
- That weaker claim is still publishable, but it is not the same paper.

### 2. Theory Risk

The theorem currently makes a very strong statement:

- if the model reaches optimal prediction on VAR(`p`), then `|W_out W_in|` recovers Granger sparsity.

The problem is that the proof does not deeply analyze the actual trained Mamba architecture. It leans on:

- sufficient capacity,
- generic full-rank internal mapping,
- absence of cancellation,
- routing intuition.

That may be acceptable as intuition or proposition-level discussion, but not as the backbone of a top-tier theorem claim.

### 3. Evaluation Risk

The experimental setup is still too easy to attack:

- many benchmarks are synthetic,
- some "real-world" ground truths are literature-derived,
- the baselines are not yet maximally strong,
- the selection protocol for SSM variant is not clean.

For NeurIPS, this matters as much as the method itself.

### 4. Positioning Risk

The repo currently tries to sell several stories at once:

- SSM internals are causal,
- training-free extraction works,
- the method is theoretically justified,
- it beats baselines,
- it helps on climate and neuroscience,
- it is efficient,
- it is architecture-specific.

That is too many fronts unless the evidence is unusually strong.

The project needs one primary claim and two supporting claims, not six equal-weight claims.

## Best Possible Paper Version

If everything goes well, the best version of the paper probably looks like this:

Title-level idea:

"Post Hoc Granger Graph Extraction from Selective State Space Forecasting Models"

Primary claim:

- selective SSMs trained only for forecasting encode lagged predictive structure that can be extracted after training with a simple score.

Secondary claims:

- this extraction is unusually effective for selective SSMs compared with matched MLP, LSTM, and linear predictors,
- the method is computationally cheap and useful as a hypothesis-generation tool,
- a limited theoretical result explains why endpoint projection can recover sparsity in linear settings.

This is a more defensible paper than:

"Causal discovery via SSM internals" with broad scientific claims.

## Exact Experiments to Run Next

These are the six experiments that matter most. They should be treated as go/no-go experiments, not just additional figures.

### Experiment 1: Architecture Specificity Test

Question:

- Does `|W_out W_in|` work better for Mamba than for matched MLP, LSTM, and linear predictors?

Design:

- Use the existing architecture comparison script as the starting point.
- Match parameter count as closely as possible.
- Match training budget and data exactly.
- Evaluate on VAR(1), VAR(2), regime-switching, Lorenz, and at least one sparse random graph setting.
- Report mean and standard deviation over at least 10 seeds.

Success criterion:

- Mamba should consistently outperform the non-SSM models on the same extraction rule, especially on non-stationary and long-memory settings.

Failure mode:

- If MLP or LSTM performs similarly, the SSM-specific story weakens substantially.

### Experiment 2: Pre-Registered Extractor Comparison

Question:

- Which extractor actually works best when chosen fairly?

Design:

- Fix a protocol before running:
  - either declare a single default extractor,
  - or choose the extractor on a validation split that is separate from the reported test split.
- Compare:
  - weight projection,
  - Jacobian,
  - any repaired `B/C` score,
  - layer-aggregated versions.

Success criterion:

- One extractor wins robustly under a fair selection protocol.

Failure mode:

- If the story depends on choosing whichever extractor wins after seeing test AUROC, the paper will not hold up.

### Experiment 3: Stronger Baseline Audit

Question:

- Do the gains remain once the baselines are made hard to dismiss?

Design:

- Add stronger or more faithful baselines where feasible:
  - multivariate VAR-based Granger baseline,
  - stronger PCMCI settings including nonlinear CI tests when appropriate,
  - a faithful external cLSTM/cMLP reproduction if possible,
  - TCDF-style or similar temporal neural causal discovery baseline if practical.
- Tune baselines with a documented validation protocol instead of fixed arbitrary defaults.

Success criterion:

- The SSM method still wins on at least a subset of settings for reasons that align with the paper's claim.

Failure mode:

- If the gains disappear under stronger baselines, the current result is mostly a comparison artifact.

### Experiment 4: Real Data with Defensible Framing

Question:

- Does the method remain useful when synthetic ground truth is no longer available?

Design:

- Stop treating literature edges as exact ground truth in a strong sense.
- Reframe real-data experiments as:
  - recovery of accepted candidate edges,
  - ranking of plausible mechanisms,
  - robustness across splits, windows, and subjects,
  - consistency with prior knowledge rather than exact graph accuracy.
- Climate:
  - use actual NOAA or ERA5-derived indices,
  - evaluate stability across eras and lags,
  - compare edge ranking consistency.
- fMRI:
  - use real preprocessed data if possible,
  - avoid strong causal wording,
  - evaluate reproducibility across subjects or sessions.

Success criterion:

- The method produces stable and interpretable rankings that align with prior domain knowledge better than weak baselines.

Failure mode:

- If results are unstable or only look good on synthetic proxies, the real-world story should be removed from the main paper.

### Experiment 5: Scaling and Regime Advantage

Question:

- Is there a genuine practical regime where this method is especially useful?

Design:

- Build a clean benchmark around:
  - long sequences,
  - non-stationarity,
  - limited retraining budget,
  - medium variable counts.
- Compare runtime and quality jointly.
- Measure not only extraction cost but total usable pipeline cost.

Success criterion:

- The method should show a real tradeoff advantage, such as:
  - comparable quality with much lower post hoc extraction cost,
  - better robustness to regime changes than classical baselines,
  - better scaling when a predictor must already be trained anyway.

Failure mode:

- If the only efficiency win is "matrix multiply after expensive training," reviewers will say the method is not truly efficient for causal discovery.

### Experiment 6: Falsification Sweep

Question:

- When does the method fail?

Design:

- Systematically vary:
  - graph density,
  - edge strength,
  - hidden confounding in synthetic settings,
  - contemporaneous effects,
  - noise level,
  - model under-capacity,
  - sample size,
  - lag order mismatch.
- Publish the failures.

Success criterion:

- You discover a coherent operating regime and can state it precisely.

Failure mode:

- If performance is inconsistent and there is no clear regime where the method is reliable, the project should be downgraded.

## Go/No-Go Criteria

Keep the NeurIPS target only if all of the following become true:

1. A fair extractor protocol is established with no oracle test-time selection.
2. The architecture comparison shows a meaningful SSM-specific advantage.
3. Stronger baselines do not erase the gains.
4. The paper is reframed around defensible claims rather than broad causal promises.
5. Real-data results are reframed conservatively and remain useful.
6. The theorem is weakened or improved enough that it no longer looks overstated.

If only some of these happen, the project may still be good, but probably not as a NeurIPS main-track paper.

## Recommended Paper Positioning Options

### Option A: Keep NeurIPS Ambition

Use this option only if the next round of experiments is unusually strong.

Paper story:

- selective SSMs uniquely support cheap, post hoc Granger graph extraction from forecasting models.

Pros:

- highest upside.

Cons:

- requires the strongest evidence and the tightest claim discipline.

### Option B: Workshop or Focused Methods Paper

Paper story:

- forecasting-trained selective SSMs contain interpretable lagged dependency structure that can be extracted post hoc for hypothesis generation.

Pros:

- much easier to defend,
- still scientifically useful,
- lower theory burden.

Cons:

- lower prestige ceiling.

### Option C: Tool Paper / Application-Led Paper

Paper story:

- a practical post hoc dependency-screening tool for long multivariate sequences, demonstrated on climate or neuroscience.

Pros:

- strong societal relevance if framed honestly.

Cons:

- needs stronger real data and domain evaluation.

## Concrete Repo Changes Recommended Before More Hype

These are not cosmetic. They improve scientific credibility.

1. Stop calling synthetic climate/fMRI results "real-world" in the main pitch.
2. Remove any wording that implies exact causal recovery from literature-derived edge lists.
3. Eliminate oracle choice between Jacobian and weight projection on test results.
4. Decide whether the paper is fundamentally about:
   - weight projection,
   - Jacobian extraction,
   - or a repaired SSM-internal score.
   Right now the paper mixes all three.
5. Strengthen or qualify the theorem.
6. Make the architecture comparison a centerpiece, not an appendix curiosity.
7. Add negative results and failure modes to the main experimental narrative.

## Compute Strategy

The next phase does not require huge-scale training, but it does require disciplined sweeps.

Recommended practical setup:

- Use a single reproducible GPU environment on RunPod, Clore.ai, or a similar service.
- Prefer one container image with pinned dependencies and fixed seeds.
- Save all raw JSON outputs, not just summary tables.
- Budget for repeated medium-scale runs rather than one giant expensive benchmark.

Suggested priority order for compute spend:

1. architecture comparison,
2. fair extractor comparison,
3. baseline strengthening,
4. falsification sweep,
5. real-data conservative analysis.

## Suggested 3-Week Tactical Plan

### Week 1

- Clean evaluation protocol.
- Lock extractor-selection policy.
- Run architecture comparison over multiple seeds.
- Run one strong synthetic benchmark suite with clean logging.

Decision point:

- If SSM is not meaningfully better than MLP/LSTM under the same extraction rule, reconsider the main thesis immediately.

### Week 2

- Upgrade baselines.
- Run non-stationary and scaling benchmarks.
- Build a failure-mode table.

Decision point:

- If gains vanish with stronger baselines, narrow the claim and downgrade venue ambition.

### Week 3

- Run conservative climate and fMRI analyses on actual data if available.
- Rewrite introduction, contributions, and method claims around what truly survived.

Decision point:

- If the real-data story is weak or unstable, keep it as supporting evidence rather than as a main contribution.

## Final Recommendation

This project is worth continuing, but only in a skeptical mode.

The right mindset is not:

"We already have a NeurIPS paper and need more polish."

The right mindset is:

"We may have found a genuinely interesting effect. Now we need to stress-test whether it is real, specific to SSMs, and strong enough to justify a top-tier claim."

If the next experiments go well, this could become a strong and distinctive paper.
If they do not, the project can still yield a valuable and honest publication with a narrower claim.

That is still a very good outcome.

## Audit Notes

This memo is based on:

- static inspection of the repo,
- inspection of the draft paper and experiments,
- a literature scan focused on adjacent work in neural Granger, temporal causal discovery, interpretable Mamba internals, and SSM-based causal modeling,
- but not a full local rerun of experiments in this environment, because the current sandbox lacks installed Python dependencies and cannot fetch packages from the network.
