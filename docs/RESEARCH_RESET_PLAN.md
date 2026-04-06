# Research Reset Plan

Date: 2026-04-05

Purpose: preserve the full reasoning chain behind the current reset, record what we learned from the first thesis, and define the next research direction in a way that can survive context resets and team handoffs.

## Executive Decision

We are intentionally moving away from the original paper thesis:

- not because the project was useless,
- but because the original claim is no longer the best scientific story,
- and because we believe the underlying insight can support a stronger and more original contribution.

Current decision:

- Drop the old SSM-internals-first thesis as the main paper direction.
- Keep the core empirical insight: prediction bottlenecks can encode dependency structure post hoc.
- Build a stronger method on top of that insight.
- Primary new direction: **causality-preserving bottleneck training**.
- Supporting direction: **hybrid bottleneck + Jacobian extraction**.

This document is both:

- a reflection on what happened so far, and
- a concrete research plan for what we do next.

## Part I: Reflection

### 1. What the Original Idea Was

The original project thesis was:

- train a Mamba / selective SSM on next-step prediction,
- do not use any causal-specific training objective,
- extract a lagged causal or dependency graph from the trained model,
- use the endpoint projection product
  `S = |W_out @ W_in|`
  as the main score.

The ambitious version of the story was:

- the trained SSM internally encodes Granger structure,
- and this structure can be recovered cheaply after training,
- which would make causal discovery almost a free byproduct of forecasting.

### 2. What Looked Good Early On

The early project had real strengths:

- the core idea was crisp and easy to explain,
- the extraction rule was strikingly simple,
- some synthetic results were genuinely promising,
- the repo already had a real implementation, baselines, and paper draft,
- there was a plausible theory story in linear VAR settings.

This was enough to justify taking the project seriously.

### 3. What Started To Break

As the project matured, several issues became clear.

#### A. The elegant "SSM internals" story was not the strongest working mechanism

There were multiple extraction candidates:

- endpoint weight projection,
- Jacobian extraction,
- `B/C`-style internal pathway extraction.

But the cleanest theoretical SSM-internal object did not emerge as the strongest practical method.

That created a mismatch:

- what sounded most novel in the paper,
- versus what actually worked best in code.

#### B. The evaluation setup was vulnerable

The biggest issues were:

- synthetic or planted-structure domain benchmarks were too close to toy settings,
- some experiments selected the best extractor after looking at test AUROC,
- the "real-world" wording was stronger than the actual evidence,
- the baselines were not strong enough to eliminate reviewer skepticism.

#### C. The key unresolved question became impossible to ignore

The central question became:

Does the extraction work because of selective SSMs specifically,
or because any predictor with a bottleneck has to route predictive information through `W_in` and `W_out`?

That question turned out to be more important than all the paper polishing.

### 4. What the Gap Analysis Said

The gap analysis concluded:

- there is a real phenomenon here,
- but the repo was not NeurIPS-ready in its current scientific form,
- and the main weaknesses were scientific, not cosmetic.

The key gaps were:

1. The theory was too strong relative to the actual evidence.
2. The SSM-specific claim was not yet demonstrated.
3. The real-data story was too weak.
4. The baseline comparisons were still attackable.
5. The extractor protocol was not locked cleanly enough.
6. The paper was trying to sell too many stories at once.

### 5. What the Go / No-Go Experiments Changed

Your teammate then ran the go/no-go experiments and summarized them in the current repo docs.

The most important outcome was:

- the original strong SSM-specific thesis weakened substantially.

In particular, the architecture comparison suggested:

- the endpoint weight-product effect is not uniquely an SSM phenomenon,
- and linear models may preserve this bottleneck dependency signal even better than Mamba in several settings.

This is the result that forced the pivot.

### 6. Why the Pivot Happened

The pivot logic was:

- if the effect is not SSM-specific,
- then the paper should not pretend it is,
- and the broader bottleneck phenomenon becomes the more honest story.

That proposed pivot was:

**"Post Hoc Granger Graph Extraction from Bottleneck Prediction Models"**

This was a meaningful improvement in honesty and scientific alignment.

### 7. Why We Are Not Stopping at the Pivot

Even though the pivot is more honest than the original paper thesis, it is still not the highest-upside final destination.

Why not:

- it risks becoming mostly a characterization paper,
- it is more "we discovered a phenomenon" than "we built a strong new method",
- and that can easily land in medium-novelty territory.

So our conclusion is:

- the original thesis should be retired,
- the pivot insight should be kept,
- but the best next move is to build a stronger method from that insight.

## Part II: What We Now Believe

### 1. What We Are Confident About

We are reasonably confident that:

- trained prediction bottlenecks can contain recoverable dependency structure,
- `|W_out @ W_in|` is not a meaningless trick,
- the effect is real enough to have motivated multiple experiments and a pivot,
- and this insight is worth building on.

### 2. What We Are Not Confident About

We are not yet confident that:

- the broad bottleneck claim is fully settled,
- the current pivot alone is strong enough for a top-tier paper,
- or the existing manuscript can be saved by rewriting alone.

### 3. Our New Standard

The next project should satisfy all three:

1. It should produce a real method, not only an observation.
2. It should preserve the strongest part of what we learned.
3. It should have a clearer path to a distinctive contribution.

## Part III: Chosen New Direction

## Primary Direction

### Causality-Preserving Bottleneck Training

Core idea:

Do not merely observe that a bottleneck predictor sometimes becomes post hoc readable.
Instead, train the predictor so that its bottleneck representation becomes deliberately more faithful, sparse, stable, and interpretable as a dependency graph.

This gives us a stronger claim:

- not "surprisingly, the weights already encode something,"
- but "we design training so the endpoint bottleneck becomes a reliable dependency extractor while preserving forecasting performance."

This is much closer to a genuine method paper.

## Supporting Direction

### Hybrid Bottleneck + Jacobian Extraction

Core idea:

The current project already discovered a useful pattern:

- weight projection works well in cleaner linear settings,
- Jacobian information can help in more nonlinear settings.

Rather than choosing one ad hoc, we treat this as supporting evidence for a hybrid estimator or consistency constraint.

This direction supports the primary method rather than replacing it.

## Part IV: Core Research Hypothesis

### Main Hypothesis

A forecasting model trained with a causality-preserving bottleneck objective can produce a post hoc dependency score that is:

- more sparse,
- more stable,
- more faithful to true lagged structure,
- and more robust than plain `|W_out @ W_in|`,

without requiring full graph-learning or heavy causal-specific supervision.

### Supporting Hypothesis

Agreement between endpoint bottleneck scores and local Jacobian scores can act as a useful training signal or regularizer:

- when they agree, the model's dependency structure is cleaner,
- when they disagree, the bottleneck is likely entangled or noisy.

## Part V: Concrete Method Direction

We should prototype a model that still predicts the next step, but adds lightweight structure-promoting objectives.

### Base Model

Start from the current time-series predictor family already in the repo:

- Mamba / selective SSM,
- and optionally a simpler linear or shallow baseline for ablation.

### Primary Extraction Score

Keep:

`S_weight = |W_out @ W_in|`

This remains important because:

- it is the original insight,
- it is computationally cheap,
- and it provides a clean target to improve.

### Candidate New Loss Terms

We should start with a small family of losses rather than one overcomplicated idea.

#### 1. Sparsity loss on endpoint projections

Goal:

- encourage a cleaner, more graph-like endpoint factorization.

Candidates:

- L1 penalty on `W_in` and `W_out`,
- row/column group sparsity,
- or sparsity directly on the product `W_out @ W_in`.

#### 2. Stability loss across subsequences

Goal:

- the extracted graph should not change wildly across nearby windows or random subsequences.

Idea:

- sample two subsequences from the same process,
- compute extracted scores,
- penalize disagreement.

#### 3. Weight-Jacobian consistency loss

Goal:

- align the cheap endpoint score with the richer local sensitivity signal.

Idea:

- compute a low-cost Jacobian approximation on small windows,
- encourage rank or correlation agreement between `S_weight` and `S_jac`.

#### 4. Optional asymmetry / anti-dense regularization

Goal:

- avoid trivial dense solutions.

Idea:

- penalize entropy or density of the extracted off-diagonal score matrix,
- or use threshold-free sparsity-promoting objectives.

### Minimal First Method Family

Start with three variants only:

1. `Forecast + sparsity`
2. `Forecast + stability`
3. `Forecast + sparsity + weight/Jacobian consistency`

This keeps the search space disciplined.

## Part VI: Research Questions

The next project should answer the following questions clearly.

### Q1. Can we improve bottleneck faithfulness without ruining forecasting?

This is the main question.

### Q2. Does the new objective outperform plain post hoc extraction?

If the answer is no, the method is not worth the added complexity.

### Q3. Does the method help especially on the currently weak settings?

Examples:

- random sparse graphs,
- multi-lag settings,
- higher-dimensional synthetic benchmarks,
- noisy, domain-inspired settings like fMRI-like data.

### Q4. Is the gain bigger for SSMs than for simpler bottleneck models?

This is no longer the starting assumption, but it remains a valuable empirical question.

## Part VII: Experiment Plan

## Phase 1: Fast Feasibility

Goal:

- determine whether the new losses can improve endpoint extraction at all.

### Datasets

Use a small but representative set:

- VAR(1) chain K=5,
- VAR(1) random K=10,
- VAR(2) K=5,
- Lorenz K=3.

### Methods

- Original weight projection baseline,
- Forecast + sparsity,
- Forecast + stability,
- Forecast + sparsity + consistency.

### Metrics

- AUROC,
- AUPRC,
- F1,
- sparsity of extracted graph,
- score stability across seeds/subsequences,
- forecasting MSE.

### Decision Gate

Continue only if at least one new method:

- improves graph recovery on at least two hard datasets,
- while not materially hurting prediction quality.

## Phase 2: Stronger Validation

Goal:

- test whether the improvement is real and not a lucky artifact.

### Additions

- 5-10 seeds,
- stronger baselines where feasible,
- compare against pure Jacobian and hybrid extraction,
- include at least one linear bottleneck baseline trained with analogous regularization.

### Decision Gate

Continue only if the new method beats the plain post hoc baseline robustly.

## Phase 3: Domain-Inspired Stress Test

Goal:

- see whether the improved method matters on more realistic synthetic settings.

Use:

- climate-like synthetic,
- finance-like synthetic,
- EEG-like synthetic,
- fMRI-like synthetic.

Interpretation:

- not as "real-world causal discovery,"
- but as structured, harder, domain-motivated testbeds.

## Phase 4: Paper Positioning Test

Ask:

- does the story now sound like a real algorithmic contribution?

If yes:

- write the paper around the new method.

If no:

- downgrade to a more careful empirical paper or rethink again.

## Part VIII: Success Criteria

The new direction is worth pursuing only if we can eventually say something like:

"We introduce a training scheme that makes forecasting models causally readable through their bottleneck projections, improving post hoc lagged dependency recovery without requiring full causal supervision."

In practice, that means:

1. A method clearly better than plain `|W_out @ W_in|`.
2. A clean ablation showing which added loss matters.
3. A forecasting-vs-recovery tradeoff that still looks favorable.
4. A paper story that is clearly stronger than the current pivot.

## Part IX: Risks

### Risk 1: The new regularizers hurt forecasting and do not help graph recovery

Mitigation:

- start simple,
- keep losses weak,
- monitor prediction quality from day one.

### Risk 2: Gains are tiny and only on easy synthetic tasks

Mitigation:

- prioritize currently weak datasets early,
- do not spend weeks polishing if Phase 1 gains are marginal.

### Risk 3: The method only reproduces what linear models already do

Mitigation:

- compare directly against simple bottleneck baselines,
- do not assume SSM-specific gains.

### Risk 4: We accidentally drift into another medium-novelty study

Mitigation:

- keep the method contribution central,
- treat characterization as support, not as the main output.

## Part X: Fallback Paths

If the primary direction does not separate enough, the fallback order should be:

1. Hybrid bottleneck + Jacobian extractor as a smaller method paper.
2. Narrow empirical paper on post hoc dependency extraction from forecasting bottlenecks.
3. Full pivot paper only if the method-building path fails cleanly.

This means we do not lose the work.

## Part XI: Immediate Next Steps

### This Week

- Define the minimal method variants.
- Implement endpoint sparsity regularization.
- Implement subsequence stability regularization.
- Implement a lightweight weight/Jacobian consistency term.
- Run Phase 1 on 4 datasets.

### After First Results

- Decide whether any method variant is meaningfully better.
- If yes, formalize the best variant and expand experiments.
- If no, stop early and reconsider before spending more time.

## Part XII: What We Are Explicitly Not Doing

To keep the direction focused, we are not currently prioritizing:

- time-varying / regime-centric causal graph work in this project,
- broad manuscript rewriting for the current pivoted paper before proving the stronger method,
- presenting synthetic domain-inspired datasets as real observational validation,
- or spending weeks polishing the old paper framing.

## Final Position

We are not abandoning the project because it failed.

We are resetting because the first thesis taught us something important:

- the real insight is probably not "SSM internals uniquely reveal causality,"
- but rather that forecasting bottlenecks can become readable,
- and the stronger research question is whether we can make them readable by design.

That is the contribution worth chasing next.
