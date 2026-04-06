# Causality-Preserving Bottleneck Training

Date: 2026-04-05

Status: design document for the first prototype

Purpose: define the first new method direction after the research reset in a way that is concrete enough to implement, test, and kill quickly if it fails.

## 1. Problem Statement

The current project discovered a useful empirical fact:

- the endpoint bottleneck score
  `S_weight = |W_out @ W_in|`
  can recover lagged dependency structure surprisingly well in some settings.

But the current weakness is also clear:

- plain post hoc extraction is an observation, not a strong new method,
- and the existing effect is not uniquely SSM-specific.

The new question is:

**Can we train a forecasting model so that its endpoint bottleneck becomes more causally readable by design, while preserving forecasting quality?**

This document proposes the first method family for that question.

## 2. Core Idea

We keep next-step forecasting as the main task, but augment training with lightweight structure-promoting objectives so that the extracted bottleneck score becomes:

- sparser,
- more stable,
- more aligned with local sensitivity,
- and more faithful to true lagged graph structure.

The goal is not to train a full graph-learning model.

The goal is to shape a standard predictor so that post hoc extraction works better.

## 3. High-Level Method

Base predictor:

- same forecasting architecture family as the current repo,
- starting with Mamba / selective SSM,
- possibly compared against a linear bottleneck baseline later.

Training objective:

`L_total = L_pred + lambda_sparse * L_sparse + lambda_stable * L_stable + lambda_consist * L_consist`

where:

- `L_pred` is next-step prediction loss,
- `L_sparse` encourages a graph-like bottleneck,
- `L_stable` encourages extraction stability across subsequences,
- `L_consist` encourages agreement between bottleneck and Jacobian signals.

We will not start with all terms at once.

The first prototype will test them incrementally.

## 4. Primary Extraction Objects

### 4.1 Endpoint Weight Score

Main score:

`S_weight = abs(W_out @ W_in)`

where:

- `W_in` has shape `[d_model, K]`,
- `W_out` has shape `[K, d_model]`,
- `S_weight` has shape `[K, K]`.

Post-processing:

- zero diagonal,
- normalize by max entry when nonzero.

This remains the main target because:

- it is the most practical readout,
- it is cheap,
- and it is the part of the original project that seems empirically real.

### 4.2 Local Jacobian Score

Support score:

`S_jac[i,j] = mean_t | d y_i(t) / d x_j(t) |`

This is not the default deployment score.

Instead, it is used as:

- a richer local influence signal,
- a support signal for nonlinear settings,
- and a possible training-time consistency target.

## 5. Loss Design

We should test the losses in increasing order of complexity.

## 5.1 Prediction Loss

Base objective:

`L_pred = MSE( y_hat_{t+1}, y_{t+1} )`

This remains the dominant loss.

The new method should not be allowed to become a graph learner that no longer forecasts well.

## 5.2 Sparsity Loss

Goal:

- make the extracted bottleneck score cleaner and more graph-like.

### Option A: L1 on endpoint score

`L_sparse_score = || offdiag(S_weight_raw) ||_1`

where:

- `S_weight_raw = W_out @ W_in`,
- `offdiag` removes the diagonal.

Pros:

- directly acts on the extracted object.

Cons:

- may be harder to optimize cleanly because it depends on the matrix product.

### Option B: L1 on endpoint weights

`L_sparse_weights = ||W_in||_1 + ||W_out||_1`

Pros:

- simple and stable.

Cons:

- does not directly target sparsity of the product.

### Option C: Group sparsity by variable pathway

Possible later extension:

- penalize column norms of `W_in`,
- penalize row norms of `W_out`,
- or penalize pairwise pathway energy.

### Initial Recommendation

Start with:

- `L_sparse_weights`

and then test:

- `L_sparse_score`

only if the simple version is not enough.

## 5.3 Stability Loss

Goal:

- the extracted dependency score should be stable across nearby subsequences from the same process.

Construction:

1. Sample two subsequences from the same time series:
   - `X_a`
   - `X_b`
2. Run both through the same model training batch or alternating updates.
3. Compute extracted scores:
   - `S_a`
   - `S_b`
4. Penalize disagreement.

Candidate loss:

`L_stable = || normalize(offdiag(S_a)) - normalize(offdiag(S_b)) ||_F^2`

Possible variants:

- cosine similarity loss,
- Spearman-rank consistency approximation,
- top-k overlap penalty.

### Initial Recommendation

Use simple Frobenius disagreement on normalized off-diagonal scores.

This is easy to implement and interpret.

## 5.4 Weight-Jacobian Consistency Loss

Goal:

- encourage the cheap global bottleneck signal to align with the richer local sensitivity signal.

Motivation:

- when the weight score and Jacobian disagree strongly, the bottleneck may be entangled or misleading.

Construction:

1. On a small window or minibatch, compute:
   - `S_weight`
   - `S_jac`
2. Normalize both.
3. Penalize disagreement.

Candidate forms:

### Option A: L2 score agreement

`L_consist_l2 = || normalize(S_weight) - normalize(S_jac) ||_F^2`

### Option B: Rank correlation alignment

Optimize for agreement in ordering rather than exact magnitudes.

This is likely more correct conceptually, but harder to optimize.

### Option C: Top-k overlap consistency

Encourage overlap of the strongest predicted edges.

### Initial Recommendation

Start with:

- `L_consist_l2`

on small windows only, to keep cost manageable.

## 6. Minimal Prototype Family

We do not want to explore an unbounded design space.

The first prototype family should be exactly these four:

### Model A: Baseline

`L = L_pred`

This is the current baseline.

### Model B: Sparse Bottleneck

`L = L_pred + lambda_sparse * L_sparse_weights`

### Model C: Stable Bottleneck

`L = L_pred + lambda_stable * L_stable`

### Model D: Sparse + Stable + Consistent

`L = L_pred + lambda_sparse * L_sparse_weights + lambda_stable * L_stable + lambda_consist * L_consist_l2`

This gives us one simple regularizer, one temporal regularizer, and one full combined model.

## 7. Initial Hyperparameter Philosophy

We should not do a huge hyperparameter search in the first round.

Start with small values that let prediction remain primary.

Suggested initial ranges:

- `lambda_sparse in {1e-5, 1e-4, 1e-3}`
- `lambda_stable in {1e-3, 1e-2}`
- `lambda_consist in {1e-3, 1e-2}`

If prediction collapses, scale down immediately.

## 8. Implementation Plan

## 8.1 New Module

Create a new module, likely:

- `ssm_causal/bottleneck_training.py`

Responsibilities:

- extraction helpers usable during training,
- regularizer implementations,
- trainer class or training utility for the new loss variants.

## 8.2 Reuse Existing Pieces

Reuse:

- current model in `ssm_causal/mamba_model.py`,
- weight extractor logic from `experiments/exp0_core_validation.py`,
- any reusable Jacobian helper,
- current metrics from `ssm_causal/utils.py`.

Avoid:

- scattering the method logic across experiment scripts.

## 8.3 Trainer Structure

Recommended design:

- keep `SSMTrainer` as baseline trainer,
- add either:
  - `CausalBottleneckTrainer`, or
  - a new configurable trainer that wraps the model and optional losses.

The key is that the new method should be callable from experiments without rewriting training code every time.

## 8.4 Computational Strategy

The Jacobian term is the main cost risk.

To control cost:

- compute Jacobian consistency on short windows only,
- maybe every N steps rather than every step,
- or on a detached small batch separate from the main forecasting update.

Prediction quality should not be sacrificed for a fancy but too-slow consistency term.

## 9. Experiment Plan

## 9.1 Phase 1: Fast Feasibility

Purpose:

- test whether the new losses improve graph recovery at all.

Datasets:

- `VAR1-chain-K5`
- `VAR1-random-K10`
- `VAR2-K5`
- `Lorenz-K3`

Methods:

- Baseline
- Sparse Bottleneck
- Stable Bottleneck
- Combined Model

Metrics:

- AUROC
- AUPRC
- F1
- prediction MSE
- graph density of extracted score
- stability across seeds

Seeds:

- start with 3 seeds for fast feedback.

### Phase 1 Success Criterion

Continue only if at least one new method:

- improves recovery on at least 2 of the harder datasets,
- and does not clearly worsen prediction quality.

If not, stop early and rethink.

## 9.2 Phase 2: Robust Validation

Purpose:

- determine whether gains are real.

Expand to:

- 5 to 10 seeds,
- more careful variance reporting,
- compare against weight-only and Jacobian-only extraction.

This phase should answer:

- does the new training actually beat plain post hoc extraction?

## 9.3 Phase 3: Domain-Inspired Stress Test

Use:

- synthetic climate-like,
- synthetic finance-like,
- synthetic EEG-like,
- synthetic fMRI-like.

Interpretation:

- this is stress-testing,
- not strong real-world causal validation.

## 10. Evaluation Rules

To avoid repeating old mistakes:

### Rule 1: Lock the default extractor

Primary reported extractor for the new method:

- `S_weight`

Any hybrid or Jacobian analysis should be clearly labeled as ablation/supporting analysis.

### Rule 2: Do not choose method variants after seeing test results

Selection of:

- loss variant,
- hyperparameters,
- or extractor

must be based on a validation protocol or an explicitly declared default.

### Rule 3: Always report prediction quality

This project is about improving post hoc readability of forecasting models.

If forecasting gets much worse, the method is less interesting.

### Rule 4: Always compare to the plain baseline

If the new training does not beat plain `|W_out @ W_in|`, we should not overclaim.

## 11. Scientific Questions This Prototype Must Answer

1. Can endpoint sparsity be improved without destroying forecasting?
2. Can subsequence stability improve graph recovery?
3. Does Jacobian agreement help in nonlinear settings?
4. Is the combined model actually better than each individual regularizer?
5. Does the method help on the current weak spots of the repo?

## 12. Likely Failure Modes

### Failure Mode A: Regularization shrinks everything

Symptom:

- scores become sparse but useless.

Response:

- lower `lambda_sparse`,
- check whether rank/order information survives,
- prefer stability over raw shrinkage if needed.

### Failure Mode B: Stability regularization forces oversmoothing

Symptom:

- extracted graphs become stable but too bland or dense.

Response:

- reduce `lambda_stable`,
- test rank consistency instead of full-matrix agreement.

### Failure Mode C: Jacobian consistency is too expensive

Symptom:

- training becomes impractical.

Response:

- compute consistency less often,
- use smaller windows,
- keep it as an ablation-only term if needed.

### Failure Mode D: Gains only happen on easy chain graphs

Symptom:

- no improvement on random/multi-lag/nonlinear settings.

Response:

- consider early termination of this line,
- or shift emphasis to a smaller hybrid-extractor paper.

## 13. What Would Make This NeurIPS-Worthy

This direction becomes strong if we can eventually claim:

- we discovered that post hoc bottleneck extraction works,
- and we introduced a training method that makes it substantially more faithful,
- while preserving forecasting and keeping extraction cheap.

That would be a real method contribution.

The minimum evidence for that kind of claim would be:

1. consistent gains over the plain post hoc baseline,
2. clear ablations,
3. a readable mechanism story,
4. honest scope and limitations.

## 14. What Would Make Us Stop

We should stop or sharply rethink if:

- all variants fail to improve over the plain baseline,
- gains are tiny and unstable,
- forecasting degrades too much,
- or the method only wins on toy settings.

This is a prototype intended to fail fast if the core idea is weak.

## 15. Immediate Build Order

Recommended exact build order:

1. Implement endpoint sparsity regularization.
2. Implement subsequence stability regularization.
3. Add a minimal configurable trainer.
4. Run 3-seed feasibility on the four starter datasets.
5. Only then add Jacobian consistency.

Reason:

- sparsity and stability are cheaper,
- more aligned with the main bottleneck story,
- and easier to debug than Jacobian-heavy training.

## 16. Deliverables

The first milestone should produce:

- code for the new trainer and losses,
- one experiment script for feasibility,
- raw JSON outputs saved under `results/`,
- a short markdown note summarizing whether the method cleared Phase 1.

That is the point where we decide whether to invest further.
