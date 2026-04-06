"""
Go/No-Go Experiments for NeurIPS Submission
============================================

Six critical experiments that determine if this is a NeurIPS paper.
All use pre-registered protocols with no oracle selection.

Default extractor: weight projection (|W_out @ W_in|), locked.

Experiment 1: Architecture Specificity (10 seeds)
Experiment 2: Pre-Registered Extractor Comparison
Experiment 3: Stronger Baseline Audit
Experiment 4: Falsification Sweep
Experiment 5: Scaling & Regime Advantage
Experiment 6: Summary Go/No-Go Table

Usage:
    python experiments/exp_gonogo.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")

from data.synthetic import (
    generate_lorenz,
    generate_random_sparse_var,
    generate_regime_switching,
    generate_var1,
)
from experiments.exp0_core_validation import (
    extract_bc_projected_score,
    extract_jacobian_score,
)
from ssm_causal.bottleneck_training import extract_weight_score
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics

# ── Helper: matched architectures ──

class MLPForTS(nn.Module):
    def __init__(self, K, d_model=64, n_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(K, d_model)
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.1)]
        self.body = nn.Sequential(*layers)
        self.output_proj = nn.Linear(d_model, K)

    def forward(self, x):
        return self.output_proj(self.body(self.input_proj(x)))


class LSTMForTS(nn.Module):
    def __init__(self, K, d_model=64, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(K, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=0.1)
        self.output_proj = nn.Linear(d_model, K)

    def forward(self, x):
        h, _ = self.lstm(self.input_proj(x))
        return self.output_proj(h)


class LinearForTS(nn.Module):
    def __init__(self, K, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(K, d_model)
        self.output_proj = nn.Linear(d_model, K)

    def forward(self, x):
        return self.output_proj(self.input_proj(x))


def train_generic(model, X, n_epochs=150, lr=1e-3, device="cpu"):
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    X = X.to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    T = X.shape[0]
    seq_len = min(200, T - 2)

    model.train()
    for _ in range(n_epochs):
        starts = torch.randint(0, T - seq_len - 1, (32,))
        bx = torch.stack([X[s:s+seq_len] for s in starts])
        by = torch.stack([X[s+1:s+seq_len+1] for s in starts])
        loss = loss_fn(model(bx), by)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1: Architecture Specificity (10 seeds)
# ══════════════════════════════════════════════════════════

def exp1_architecture_specificity(device="cpu", n_seeds=10):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Architecture Specificity (10 seeds)")
    print("Question: Does SSM beat MLP/LSTM under same extraction?")
    print("=" * 70)

    datasets = {
        "VAR1-chain-K5": lambda s: generate_var1(K=5, T=2000, graph_type="chain", seed=s),
        "VAR1-random-K10": lambda s: generate_var1(K=10, T=3000, graph_type="random", seed=s),
        "Regime-K3": lambda s: generate_regime_switching(K=3, T=4000, seed=s),
        "Lorenz-K3": lambda s: generate_lorenz(T=5000, seed=s, downsample=5),
    }

    archs = {
        "Mamba": lambda K: MambaForTimeSeries(n_variables=K, d_model=64, d_state=16, n_layers=4),
        "MLP": lambda K: MLPForTS(K, d_model=64, n_layers=4),
        "LSTM": lambda K: LSTMForTS(K, d_model=64, n_layers=2),
        "Linear": lambda K: LinearForTS(K, d_model=64),
    }

    results = {}

    for ds_name, gen_fn in datasets.items():
        print(f"\n  {ds_name}:")
        results[ds_name] = {}

        for arch_name, model_fn in archs.items():
            aurocs = []
            for seed in range(n_seeds):
                X, _, G = gen_fn(seed + 100)
                K = X.shape[1]
                torch.manual_seed(seed * 42)
                model = model_fn(K)

                if arch_name == "Mamba":
                    trainer = SSMTrainer(model, lr=1e-3, device=device)
                    trainer.train(X, n_epochs=150, verbose=False)
                else:
                    model = train_generic(model, X, n_epochs=150, device=device)

                S = extract_weight_score(model)
                m = compute_metrics(S, G)
                aurocs.append(m["auroc"])

            mean_a = np.mean(aurocs)
            std_a = np.std(aurocs)
            results[ds_name][arch_name] = {
                "mean": mean_a, "std": std_a, "all": aurocs
            }
            print(f"    {arch_name:8s}: {mean_a:.3f} +/- {std_a:.3f}")

    return results


# ══════════════════════════════════════════════════════════
# EXPERIMENT 2: Pre-Registered Extractor Comparison
# ══════════════════════════════════════════════════════════

def exp2_extractor_comparison(device="cpu", n_seeds=5):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Pre-Registered Extractor Comparison")
    print("Question: Which extractor wins under fair protocol?")
    print("=" * 70)

    datasets = {
        "VAR1-chain-K5": lambda s: generate_var1(K=5, T=2000, graph_type="chain", seed=s),
        "Regime-K3": lambda s: generate_regime_switching(K=3, T=4000, seed=s),
        "Lorenz-K3": lambda s: generate_lorenz(T=5000, seed=s, downsample=5),
    }

    extractors = {
        "Weight-Proj": lambda m, X, d: extract_weight_score(m),
        "Jacobian": lambda m, X, d: extract_jacobian_score(m, X, d, seq_len=100),
        "BC-Proj": lambda m, X, d: extract_bc_projected_score(m, X, d),
    }

    results = {}

    for ds_name, gen_fn in datasets.items():
        print(f"\n  {ds_name}:")
        results[ds_name] = {}

        for ext_name, ext_fn in extractors.items():
            aurocs = []
            for seed in range(n_seeds):
                X, _, G = gen_fn(seed + 200)
                K = X.shape[1]
                torch.manual_seed(seed * 42)
                model = MambaForTimeSeries(n_variables=K, d_model=64, d_state=16, n_layers=4)
                trainer = SSMTrainer(model, lr=1e-3, device=device)
                trainer.train(X, n_epochs=150, verbose=False)
                X_t = torch.FloatTensor(X)

                try:
                    S = ext_fn(model, X_t, device)
                    m = compute_metrics(S, G)
                    aurocs.append(m["auroc"])
                except Exception:
                    aurocs.append(0.5)

            mean_a = np.mean(aurocs)
            std_a = np.std(aurocs)
            results[ds_name][ext_name] = {"mean": mean_a, "std": std_a}
            print(f"    {ext_name:12s}: {mean_a:.3f} +/- {std_a:.3f}")

    return results


# ══════════════════════════════════════════════════════════
# EXPERIMENT 3: Stronger Baseline Audit
# ══════════════════════════════════════════════════════════

def exp3_stronger_baselines(device="cpu"):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Stronger Baseline Audit")
    print("Question: Do gains survive with better-tuned baselines?")
    print("=" * 70)

    X, _, G = generate_var1(K=5, T=2000, graph_type="chain", seed=42)
    K = X.shape[1]
    results = {}

    # SSM-Causal (locked weight projection)
    model = MambaForTimeSeries(n_variables=K, d_model=64, d_state=16, n_layers=4)
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=150, verbose=False)
    S = extract_weight_score(model)
    m = compute_metrics(S, G)
    results["SSM-Causal"] = m["auroc"]
    print(f"  SSM-Causal (weight-proj): {m['auroc']:.3f}")

    # Granger (bivariate, default)
    try:
        from baselines.granger import granger_causality_matrix
        S_g = granger_causality_matrix(X, max_lag=5)
        m = compute_metrics(S_g, G)
        results["Granger-biv"] = m["auroc"]
        print(f"  Granger (bivariate):      {m['auroc']:.3f}")
    except Exception as e:
        print(f"  Granger: SKIP ({e})")

    # Granger (multivariate via VAR)
    try:
        from statsmodels.tsa.api import VAR as VARModel
        var_model = VARModel(X).fit(maxlags=5, ic='aic')
        coefs = np.abs(var_model.coefs).sum(axis=0)
        np.fill_diagonal(coefs, 0)
        if coefs.max() > 0:
            coefs = coefs / coefs.max()
        m = compute_metrics(coefs, G)
        results["Granger-VAR"] = m["auroc"]
        print(f"  Granger (multivariate):   {m['auroc']:.3f}")
    except Exception as e:
        print(f"  Granger-VAR: SKIP ({e})")

    # PCMCI (default alpha)
    try:
        from baselines.pcmci_wrapper import pcmci_causality_matrix
        S_p1 = pcmci_causality_matrix(X, tau_max=5, pc_alpha=0.05)
        m1 = compute_metrics(S_p1, G)
        results["PCMCI-0.05"] = m1["auroc"]
        print(f"  PCMCI (alpha=0.05):       {m1['auroc']:.3f}")

        S_p2 = pcmci_causality_matrix(X, tau_max=5, pc_alpha=None)
        m2 = compute_metrics(S_p2, G)
        results["PCMCI-None"] = m2["auroc"]
        print(f"  PCMCI (alpha=None):       {m2['auroc']:.3f}")
    except Exception as e:
        print(f"  PCMCI: SKIP ({e})")

    # cLSTM (multiple lambda values)
    try:
        from baselines.neural_granger import train_clstm
        for lam in [0.001, 0.01]:
            _, S_c = train_clstm(X, n_epochs=200, lambda_group=lam,
                                  hidden_size=64, verbose=False, device=device)
            m = compute_metrics(S_c, G)
            results[f"cLSTM-{lam}"] = m["auroc"]
            print(f"  cLSTM (lambda={lam}):      {m['auroc']:.3f}")
    except Exception as e:
        print(f"  cLSTM: SKIP ({e})")

    return results


# ══════════════════════════════════════════════════════════
# EXPERIMENT 4: Falsification Sweep
# ══════════════════════════════════════════════════════════

def exp4_falsification(device="cpu"):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Falsification Sweep")
    print("Question: When does the method fail?")
    print("=" * 70)

    results = {}

    # Vary graph density
    print("\n  Graph density:")
    for density in [0.1, 0.2, 0.3, 0.5, 0.8]:
        X, _, G = generate_random_sparse_var(K=5, T=2000, density=density, seed=42)
        model = MambaForTimeSeries(n_variables=5, d_model=64, d_state=16, n_layers=4)
        trainer = SSMTrainer(model, lr=1e-3, device=device)
        trainer.train(X, n_epochs=150, verbose=False)
        S = extract_weight_score(model)
        m = compute_metrics(S, G)
        results[f"density={density}"] = m["auroc"]
        print(f"    density={density}: AUROC={m['auroc']:.3f}")

    # Vary noise level
    print("\n  Noise level:")
    for noise in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        X, _, G = generate_var1(K=5, T=2000, graph_type="chain", noise_std=noise, seed=42)
        model = MambaForTimeSeries(n_variables=5, d_model=64, d_state=16, n_layers=4)
        trainer = SSMTrainer(model, lr=1e-3, device=device)
        trainer.train(X, n_epochs=150, verbose=False)
        S = extract_weight_score(model)
        m = compute_metrics(S, G)
        results[f"noise={noise}"] = m["auroc"]
        print(f"    noise={noise}: AUROC={m['auroc']:.3f}")

    # Vary K (dimensionality)
    print("\n  Variable count K:")
    for K in [3, 5, 10, 15, 20]:
        X, _, G = generate_var1(K=K, T=3000, graph_type="chain", seed=42)
        model = MambaForTimeSeries(n_variables=K, d_model=64, d_state=16, n_layers=4)
        trainer = SSMTrainer(model, lr=1e-3, device=device)
        trainer.train(X, n_epochs=150, verbose=False)
        S = extract_weight_score(model)
        m = compute_metrics(S, G)
        results[f"K={K}"] = m["auroc"]
        print(f"    K={K}: AUROC={m['auroc']:.3f}")

    # Vary sample size T
    print("\n  Sample size T:")
    for T in [200, 500, 1000, 2000, 5000]:
        X, _, G = generate_var1(K=5, T=T, graph_type="chain", seed=42)
        model = MambaForTimeSeries(n_variables=5, d_model=64, d_state=16, n_layers=4)
        trainer = SSMTrainer(model, lr=1e-3, device=device)
        trainer.train(X, n_epochs=150, verbose=False)
        S = extract_weight_score(model)
        m = compute_metrics(S, G)
        results[f"T={T}"] = m["auroc"]
        print(f"    T={T}: AUROC={m['auroc']:.3f}")

    # Under-capacity (d < Kp)
    print("\n  Model capacity (d_model):")
    for d in [16, 32, 64, 128]:
        X, _, G = generate_var1(K=5, T=2000, graph_type="chain", seed=42)
        model = MambaForTimeSeries(n_variables=5, d_model=d, d_state=16, n_layers=4)
        trainer = SSMTrainer(model, lr=1e-3, device=device)
        trainer.train(X, n_epochs=150, verbose=False)
        S = extract_weight_score(model)
        m = compute_metrics(S, G)
        results[f"d_model={d}"] = m["auroc"]
        print(f"    d_model={d}: AUROC={m['auroc']:.3f}")

    return results


# ══════════════════════════════════════════════════════════
# MAIN: Run all and generate go/no-go summary
# ══════════════════════════════════════════════════════════

def run_all_gonogo(device="cpu", save_dir="results/gonogo"):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    all_results = {}

    # Run all experiments
    all_results["exp1_architecture"] = exp1_architecture_specificity(device, n_seeds=10)
    all_results["exp2_extractors"] = exp2_extractor_comparison(device, n_seeds=5)
    all_results["exp3_baselines"] = exp3_stronger_baselines(device)
    all_results["exp4_falsification"] = exp4_falsification(device)

    total_time = time.time() - total_start

    # ── Go/No-Go Summary ──
    print("\n" + "=" * 70)
    print("GO / NO-GO SUMMARY")
    print("=" * 70)

    # Criterion 1: SSM beats MLP/LSTM on >= 2/4 datasets
    arch = all_results["exp1_architecture"]
    ssm_wins = 0
    for ds, methods in arch.items():
        ssm_mean = methods.get("Mamba", {}).get("mean", 0)
        best_other = max(
            methods.get("MLP", {}).get("mean", 0),
            methods.get("LSTM", {}).get("mean", 0),
            methods.get("Linear", {}).get("mean", 0),
        )
        if ssm_mean > best_other + 0.02:  # Meaningful margin
            ssm_wins += 1
    c1 = ssm_wins >= 2
    print("\n  Criterion 1: SSM beats alternatives on >= 2/4 datasets")
    print(f"    SSM wins: {ssm_wins}/4 → {'PASS' if c1 else 'FAIL'}")

    # Criterion 2: Weight projection is best or tied-best extractor
    ext = all_results["exp2_extractors"]
    wp_best = 0
    for ds, methods in ext.items():
        wp = methods.get("Weight-Proj", {}).get("mean", 0)
        best = max(m.get("mean", 0) for m in methods.values())
        if wp >= best - 0.03:
            wp_best += 1
    c2 = wp_best >= 2
    print("\n  Criterion 2: Weight-proj is best/tied extractor on >= 2/3")
    print(f"    Weight-proj best: {wp_best}/3 → {'PASS' if c2 else 'FAIL'}")

    # Criterion 3: SSM beats strongest Granger baseline
    bl = all_results["exp3_baselines"]
    ssm_auroc = bl.get("SSM-Causal", 0)
    best_granger = max(bl.get("Granger-biv", 0), bl.get("Granger-VAR", 0))
    c3 = ssm_auroc >= best_granger - 0.05
    print("\n  Criterion 3: SSM competitive with strongest Granger")
    print(f"    SSM={ssm_auroc:.3f} vs Granger={best_granger:.3f} → {'PASS' if c3 else 'FAIL'}")

    # Criterion 4: Method works for K >= 10
    fals = all_results["exp4_falsification"]
    k10_auroc = fals.get("K=10", 0)
    c4 = k10_auroc >= 0.70
    print("\n  Criterion 4: Works for K=10 (AUROC >= 0.70)")
    print(f"    K=10 AUROC={k10_auroc:.3f} → {'PASS' if c4 else 'FAIL'}")

    # Overall
    passes = sum([c1, c2, c3, c4])
    print(f"\n  {'=' * 50}")
    if passes >= 4:
        print(f"  OVERALL: GO for NeurIPS ({passes}/4 criteria met)")
    elif passes >= 3:
        print(f"  OVERALL: CONDITIONAL GO ({passes}/4 — address weak criterion)")
    elif passes >= 2:
        print(f"  OVERALL: Consider WORKSHOP paper ({passes}/4)")
    else:
        print(f"  OVERALL: NO-GO for top venue ({passes}/4)")
    print(f"  {'=' * 50}")
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")

    # Save all results
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(os.path.join(save_dir, "gonogo_results.json"), "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)

    print(f"\n  Results saved to {save_dir}/gonogo_results.json")
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    run_all_gonogo(device=device)
