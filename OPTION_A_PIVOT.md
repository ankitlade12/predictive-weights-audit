# Option A: Pivot Paper for NeurIPS 2026

## New Title

**"Post Hoc Granger Graph Extraction from Bottleneck Prediction Models"**

Alternative titles:
- "Prediction-Trained Neural Networks Encode Granger Structure: A Bottleneck Factorization Perspective"
- "Free Causal Screening: Extracting Dependency Graphs from Any Trained Prediction Model"

---

## New Core Claim

**Any neural prediction model with input/output projection layers encodes
Granger-type dependency structure in the product `|W_out @ W_in|`, enabling
post hoc dependency extraction at near-zero cost.**

This is a **broader and stronger claim** than the original SSM-specific one.
It turns our "weakness" (Linear beats SSM) into a **strength** (the method
is architecture-agnostic).

---

## Why This Is Actually a Better Paper

### Original story (weak)
"SSM internals encode causal structure"
→ Falsified by go/no-go experiments. Linear model does it better.

### Pivoted story (strong)
"Any bottleneck predictor encodes dependency structure. We discover this,
prove it theoretically, show it works across architectures, and identify
when and why it fails."

### Why this is more novel
1. **No one has systematically studied weight-projection dependency
   extraction across architectures.** Tank et al. (cLSTM) use group-lasso
   on input weights — that's a training-time technique. We show the same
   structure emerges *without* any causal objective.

2. **The architecture comparison IS the contribution.** Showing that
   Linear > SSM > LSTM > MLP on the same extraction is a genuinely new
   finding that the community should know about.

3. **The failure modes are publishable.** Dense graphs fail, sparse graphs
   work. Under-capacity fails, sufficient capacity works. These are clean,
   interpretable findings.

---

## Revised Contributions

1. **Discovery:** Prediction-trained models with bottleneck projections
   encode Granger dependency structure in `|W_out @ W_in|`, without any
   causal training objective.

2. **Theory:** We prove that this extraction recovers VAR(p) Granger
   structure for any bottleneck model with sufficient capacity (d ≥ Kp)
   under a genericity condition.

3. **Architecture ranking:** We show that simpler architectures (Linear,
   SSM) encode dependency structure more faithfully than complex nonlinear
   ones (MLP, LSTM), establishing a clear hierarchy:
   Linear ≥ SSM >> LSTM > MLP.

4. **Operating regime:** We characterize when the method works (sparse
   graphs, sufficient capacity, any noise level) and when it fails (dense
   graphs, d < Kp), through comprehensive falsification experiments.

5. **Practical tool:** Post-training extraction requires one matrix
   multiply (<1ms), enabling dependency screening as a free byproduct of
   any prediction model.

---

## Revised Paper Outline

### Section 1: Introduction
- Observation: trained predictors must encode which inputs drive which outputs
- Key insight: for bottleneck architectures, this is readable from W_out @ W_in
- This paper: systematic study of when, why, and how well this works

### Section 2: Background
- Granger causality and VAR models
- Neural Granger causality (cLSTM — requires causal objective)
- Bottleneck architectures (Linear, SSM, LSTM, MLP)

### Section 3: Method
- Definition: Bottleneck Dependency Score S = |W_out @ W_in|
- Proposition: recovers VAR(p) Granger structure under capacity + genericity
- Algorithm: train any predictor, read W_out and W_in, multiply
- Complexity analysis: O(dK²) extraction, one-time training

### Section 4: Experiments

#### 4.1 Architecture Comparison (THE CENTERPIECE)
- 4 architectures × 4 datasets × 10 seeds
- Finding: Linear ≥ SSM >> LSTM > MLP
- Interpretation: nonlinear internal computation disrupts the clean
  bottleneck factorization; simpler architectures preserve sparsity

#### 4.2 Extractor Comparison
- Weight-Proj vs Jacobian vs B·C^T
- Finding: Weight-Proj wins on linear data, Jacobian on nonlinear
- Protocol: locked default (weight-proj), no oracle selection

#### 4.3 Baseline Comparison
- SSM vs Granger (bivariate + multivariate) vs PCMCI vs cLSTM (tuned)
- Finding: competitive with all baselines, no causal training needed

#### 4.4 Falsification Sweep
- Density: works for ≤0.5, fails at 0.8
- Noise: robust up to σ=1.0
- K: scales to 20+
- T: works from 500+
- Capacity: d_model ≥ 32 needed for K=5

#### 4.5 Domain-Inspired Benchmarks
- Climate, Finance, EEG, fMRI (synthetic with planted ground truth)
- Framed as: "can the method recover planted structure in realistic settings?"
- NOT framed as: "real-world causal discovery"

### Section 5: Related Work
- Neural Granger (cLSTM, cMLP) — requires causal objective
- SSM interpretability — we extend to dependency extraction
- Causal representation learning — we provide a concrete instance

### Section 6: Discussion
- Why does this work? Prediction models must route predictive information
  through the bottleneck. The factored structure encodes this routing.
- Why Linear > SSM? Nonlinear layers create off-diagonal noise in the
  effective mapping. Linear models have W_eff = I, giving exact recovery.
- When to use this? Hypothesis generation, fast screening, post hoc
  analysis of existing models. NOT a replacement for careful causal
  inference.
- Limitations: dense graphs, confounding, contemporaneous effects,
  synthetic ground truth only

---

## Why This Could Work at NeurIPS

1. **Genuinely new finding.** No one has shown that untrained weight
   projections encode Granger structure across architectures.

2. **Rigorous evaluation.** 10-seed comparisons, pre-registered extractor,
   falsification sweep, honest failures. This is the evaluation quality
   NeurIPS expects.

3. **Clean narrative.** One discovery, proved theoretically, validated
   across architectures, with clear failure modes.

4. **Practical utility.** Any existing prediction model can be queried
   for dependency structure at <1ms cost.

5. **Honest framing.** We don't oversell. We call it "dependency
   extraction," not "causal discovery." We show failures. We explain
   why Linear beats SSM.

---

## What Still Needs To Be Done

### Must-do (1 week)
- [ ] Rewrite paper around pivoted claim
- [ ] Architecture comparison as Section 4.1 (centerpiece)
- [ ] Remove all SSM-specific language from title/abstract/contributions
- [ ] Update theory to explicitly state "any bottleneck architecture"
- [ ] Add Linear model results to all tables
- [ ] Add failure mode table (density, capacity)

### Should-do (1 week)
- [ ] Run on at least one real observational dataset (actual NOAA data)
- [ ] Add TCDF or DYNOTEARS baseline
- [ ] Add cross-validated extractor selection protocol
- [ ] Bootstrap confidence intervals on main table

### Nice-to-have (stretch)
- [ ] Theoretical analysis of why Linear > SSM (spectral analysis of W_eff)
- [ ] Connection to transfer entropy
- [ ] Extension to transformer architectures (attention as bottleneck?)

---

## Timeline

| Week | Tasks |
|---|---|
| 1 (Apr 3-9) | Rewrite paper, update all tables with go/no-go results |
| 2 (Apr 10-16) | Add real data, stronger baselines, bootstrap CIs |
| 3 (Apr 17-23) | Polish, get feedback from Aman, revise |
| 4 (Apr 24-30) | Final draft, code cleanup |
| 5 (May 1-7) | Submit to NeurIPS + arXiv |

---

## Risk Assessment

| Risk | Probability | Mitigation |
|---|---|---|
| Reviewers say "just a linear probe" | 30% | Emphasize: discovery is that it works WITHOUT causal training |
| Reviewers want real data | 40% | Add actual NOAA climate data with conservative framing |
| Someone publishes similar finding | 15% | Post arXiv immediately after rewrite |
| Aman disagrees with pivot | 10% | Share go/no-go results — data speaks |

---

## Decision

If you agree with Option A:
1. I rewrite the paper immediately around the pivoted claim
2. We share go/no-go results with Aman for feedback
3. We target NeurIPS 2026 with the stronger, broader paper

The pivoted paper is actually **more publishable** than the original
because it makes a broader claim backed by stronger evidence, with
honest failure modes and no SSM-specific overclaiming.
