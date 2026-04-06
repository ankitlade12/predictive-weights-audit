"""
Phase 2: Robust Validation of Causality-Preserving Bottleneck
==============================================================

Follows Phase 1 success. Tests whether D-Combined gains are real.

Changes from Phase 1:
- 10 seeds (was 3)
- Added datasets: Regime-switching, Hénon
- Added comparisons: Linear bottleneck, Jacobian-only, Sparse+Consist (no stable)
- λ sensitivity sweep on best dataset (Lorenz)
- d_model=128 test for K=10

Decision gate: D-Combined must beat baseline robustly (p < 0.05).

Usage:
    python experiments/exp12_phase2_validation.py
    python experiments/exp12_phase2_validation.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from data.synthetic import (
    generate_var1, generate_var2, generate_lorenz,
    generate_regime_switching, generate_henon,
)
from ssm_causal.bottleneck_training import (
    BottleneckTrainingConfig,
    CausalBottleneckTrainer,
    evaluate_bottleneck_model,
    extract_weight_score,
)
from ssm_causal.mamba_model import MambaForTimeSeries
from ssm_causal.utils import compute_metrics
from experiments.exp0_core_validation import extract_jacobian_score


# ── Linear bottleneck for comparison ──
class LinearBottleneck(nn.Module):
    def __init__(self, K, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(K, d_model)
        self.output_proj = nn.Linear(d_model, K)
    def forward(self, x):
        return self.output_proj(self.input_proj(x))


DATASETS = {
    "VAR1-chain-K5": lambda s: generate_var1(K=5, T=1200, graph_type="chain", seed=s),
    "VAR1-random-K10": lambda s: generate_var1(K=10, T=1600, graph_type="random", seed=s),
    "VAR2-K5": lambda s: generate_var2(K=5, T=1600, seed=s),
    "Lorenz-K3": lambda s: generate_lorenz(T=5000, seed=s, downsample=5),
    "Regime-K3": lambda s: generate_regime_switching(K=3, T=4000, seed=s),
    "Henon-K2": lambda s: generate_henon(T=3000, seed=s),
}

METHODS = {
    "A-Baseline": {"lambda_sparse": 0.0, "lambda_stable": 0.0, "lambda_consist": 0.0},
    "B-Sparse": {"lambda_sparse": 3e-5, "lambda_stable": 0.0, "lambda_consist": 0.0},
    "D-Combined": {"lambda_sparse": 3e-5, "lambda_stable": 1e-2, "lambda_consist": 1e-2},
    "E-Sparse+Consist": {"lambda_sparse": 3e-5, "lambda_stable": 0.0, "lambda_consist": 1e-2},
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval(model, X, G_true, config, device):
    split = int(len(X) * 0.8)
    X_train, X_eval = X[:split], X[split - 1:]
    trainer = CausalBottleneckTrainer(model, config=config, device=device)
    trainer.train(X_train, verbose=False)
    return evaluate_bottleneck_model(model, X_eval, G_true, device=device)


def run_phase2(
    n_seeds=10,
    epochs=80,
    d_model=64,
    d_state=16,
    n_layers=4,
    device="cpu",
    save_dir="results/phase2_validation",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("PHASE 2: Robust Validation (10 seeds)")
    print(f"Device: {device}")
    print("=" * 72)

    all_results = {}

    # ── Part 1: Main comparison (10 seeds) ──
    print("\n╔══ Part 1: Main Methods (10 seeds) ══╗")

    for ds_name, gen_fn in DATASETS.items():
        X_sample, _, _ = gen_fn(0)
        K = X_sample.shape[1]
        dm = 128 if K >= 10 else d_model  # Adaptive for K=10

        print(f"\n  {ds_name} (K={K}, d_model={dm}):")
        ds_results = {}

        for method_name, lambdas in METHODS.items():
            aurocs = []
            pred_mses = []
            for seed in range(n_seeds):
                set_seed(seed * 42 + 7)
                X, _, G = gen_fn(100 + seed)
                model = MambaForTimeSeries(
                    n_variables=K, d_model=dm, d_state=d_state, n_layers=n_layers, dropout=0.05,
                )
                config = BottleneckTrainingConfig(
                    n_epochs=epochs, batch_size=16, seq_len=100,
                    lambda_sparse=lambdas["lambda_sparse"],
                    lambda_stable=lambdas["lambda_stable"],
                    lambda_consist=lambdas["lambda_consist"],
                    consist_every_n=5, print_every=epochs + 1,
                )
                metrics = train_and_eval(model, X, G, config, device)
                aurocs.append(metrics["auroc"])
                pred_mses.append(metrics["prediction_mse"])

            ds_results[method_name] = {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "auroc_all": [float(a) for a in aurocs],
                "pred_mse_mean": float(np.mean(pred_mses)),
            }
            print(f"    {method_name:18s}: AUROC={np.mean(aurocs):.3f}±{np.std(aurocs):.3f} | MSE={np.mean(pred_mses):.5f}")

        # Jacobian-only extraction (from baseline model)
        set_seed(42)
        X, _, G = gen_fn(100)
        model = MambaForTimeSeries(n_variables=K, d_model=dm, d_state=d_state, n_layers=n_layers)
        config = BottleneckTrainingConfig(n_epochs=epochs, batch_size=16, seq_len=100, print_every=epochs+1)
        split = int(len(X) * 0.8)
        trainer = CausalBottleneckTrainer(model, config=config, device=device)
        trainer.train(X[:split], verbose=False)
        X_tensor = torch.FloatTensor(X)
        S_jac = extract_jacobian_score(model, X_tensor, device, seq_len=100)
        m_jac = compute_metrics(S_jac, G)
        ds_results["Jacobian-only"] = {"auroc_mean": float(m_jac["auroc"]), "auroc_std": 0.0}
        print(f"    {'Jacobian-only':18s}: AUROC={m_jac['auroc']:.3f}")

        # Linear bottleneck baseline
        lin_aurocs = []
        for seed in range(min(5, n_seeds)):
            set_seed(seed * 42 + 7)
            X, _, G = gen_fn(100 + seed)
            lin_model = LinearBottleneck(K, dm)
            lin_model = lin_model.to(device)
            X_t = torch.FloatTensor(X).to(device)
            opt = torch.optim.AdamW(lin_model.parameters(), lr=1e-3, weight_decay=1e-4)
            T_len = X_t.shape[0]
            sl = min(100, T_len - 2)
            lin_model.train()
            for _ in range(epochs):
                starts = torch.randint(0, T_len - sl - 1, (16,))
                bx = torch.stack([X_t[s:s+sl] for s in starts])
                by = torch.stack([X_t[s+1:s+sl+1] for s in starts])
                loss = nn.MSELoss()(lin_model(bx), by)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(lin_model.parameters(), 1.0)
                opt.step()
            S_lin = extract_weight_score(lin_model)
            m_lin = compute_metrics(S_lin, G)
            lin_aurocs.append(m_lin["auroc"])

        ds_results["Linear-baseline"] = {
            "auroc_mean": float(np.mean(lin_aurocs)),
            "auroc_std": float(np.std(lin_aurocs)),
        }
        print(f"    {'Linear-baseline':18s}: AUROC={np.mean(lin_aurocs):.3f}±{np.std(lin_aurocs):.3f}")

        all_results[ds_name] = ds_results

    # ── Part 2: λ sensitivity on Lorenz ──
    print("\n╔══ Part 2: λ Sensitivity (Lorenz) ══╗")
    lambda_results = {}

    for ls in [0, 1e-5, 3e-5, 1e-4, 3e-4]:
        for lc in [0, 1e-3, 1e-2, 5e-2]:
            label = f"sparse={ls:.0e},consist={lc:.0e}"
            aurocs = []
            for seed in range(3):
                set_seed(seed * 42 + 7)
                X, _, G = generate_lorenz(T=5000, seed=100 + seed, downsample=5)
                model = MambaForTimeSeries(n_variables=3, d_model=64, d_state=16, n_layers=4)
                config = BottleneckTrainingConfig(
                    n_epochs=epochs, batch_size=16, seq_len=100,
                    lambda_sparse=ls, lambda_stable=0.0, lambda_consist=lc,
                    consist_every_n=5, print_every=epochs+1,
                )
                metrics = train_and_eval(model, X, G, config, device)
                aurocs.append(metrics["auroc"])
            lambda_results[label] = {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))}
            print(f"  {label:35s}: AUROC={np.mean(aurocs):.3f}±{np.std(aurocs):.3f}")

    all_results["lambda_sensitivity"] = lambda_results

    # ── Part 3: Statistical significance ──
    print("\n╔══ Part 3: Statistical Tests ══╗")
    from scipy.stats import wilcoxon

    for ds_name in ["VAR2-K5", "Lorenz-K3"]:
        if ds_name not in all_results:
            continue
        baseline_aurocs = all_results[ds_name]["A-Baseline"].get("auroc_all", [])
        combined_aurocs = all_results[ds_name]["D-Combined"].get("auroc_all", [])
        if len(baseline_aurocs) >= 5 and len(combined_aurocs) >= 5:
            try:
                stat, p_val = wilcoxon(combined_aurocs, baseline_aurocs, alternative="greater")
                sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                print(f"  {ds_name}: D-Combined vs Baseline Wilcoxon p={p_val:.4f} ({sig})")
                all_results[ds_name]["significance_test"] = {"statistic": float(stat), "p_value": float(p_val)}
            except Exception as e:
                print(f"  {ds_name}: test failed ({e})")

    total_time = time.time() - total_start

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print("PHASE 2 SUMMARY")
    print(f"{'=' * 72}")

    for ds_name in DATASETS:
        if ds_name not in all_results:
            continue
        ds = all_results[ds_name]
        baseline = ds["A-Baseline"]["auroc_mean"]
        combined = ds["D-Combined"]["auroc_mean"]
        linear = ds.get("Linear-baseline", {}).get("auroc_mean", 0)
        delta = combined - baseline
        print(f"  {ds_name:20s}: Baseline={baseline:.3f} D-Combined={combined:.3f} Linear={linear:.3f} Δ={delta:+.3f}")

    print(f"\n  Total runtime: {total_time / 60:.1f} minutes")

    # Save
    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [serialize(v) for v in obj]
        return obj

    out_path = Path(save_dir) / "phase2_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(all_results), f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/phase2_validation")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_phase2(n_seeds=2, epochs=15, d_model=32, d_state=8, n_layers=2,
                   device=args.device, save_dir=args.save_dir)
    else:
        run_phase2(n_seeds=args.seeds, epochs=args.epochs,
                   device=args.device, save_dir=args.save_dir)
