# Research Status — Honest Assessment

**Date:** 2026-04-03
**Status:** Active research, NOT submission-ready

## What We Have

A potentially interesting observation: trained Mamba models encode lagged
predictive structure in their weight matrices that can be extracted post hoc
via `S = |W_out @ W_in|`. This works reasonably well on synthetic benchmarks.

## What We Don't Have Yet

1. **Proof that this is SSM-specific.** Exp7 shows MLP/Linear match or beat
   Mamba on Lorenz. SSM advantage is clear only on regime-switching.

2. **Fair evaluation protocol.** Current scripts pick the best extractor
   after seeing test AUROC. Must lock a single default before evaluation.

3. **Strong baselines.** Bivariate Granger, default PCMCI, simplified cLSTM.
   All vulnerable to "you didn't tune the baseline" criticism.

4. **Real data.** All "real-world" experiments use synthetic data with planted
   ground truth. Not real-world.

5. **Rigorous theory.** Theorem 1 is about bottleneck factorization, not about
   SSMs specifically. Any model with W_in/W_out satisfies the same argument.

## Go/No-Go Criteria for NeurIPS

Keep NeurIPS target ONLY if ALL of these become true:
- [ ] SSM consistently outperforms matched MLP/LSTM under same extraction (10+ seeds)
- [ ] Single locked extractor (weight projection) works without oracle selection
- [ ] Gains survive with stronger baselines (multivariate Granger, tuned PCMCI)
- [ ] Paper reframed around defensible narrow claim
- [ ] Real data experiments reframed as "hypothesis generation" not "causal discovery"
- [ ] Theorem weakened to proposition or removed

## Recommended Narrow Claim

"Post hoc Granger graph extraction from selective SSM forecasting models"

NOT: "Causal discovery via SSM internals"

## Next Steps

1. Lock weight projection as default extractor — no oracle selection
2. Run architecture comparison with 10 seeds, matched parameter counts
3. Upgrade baselines (multivariate Granger, tuned PCMCI, faithful cLSTM)
4. Falsification sweep (when does the method fail?)
5. Decide go/no-go based on results
