"""
Experiment 9: Improved SSM-Causal vs Original
===============================================

Tests whether the improved extraction methods close the gaps:
1. full_path: Traces B·C^T through the complete SSM pathway
2. gradient_weighted: Geometric mean of Jacobian + Weight scores
3. ensemble: Rank-based ensemble of all three methods
4. Adaptive model sizing: d_model scaled with K
5. Sparsity-aware training: L1 on projections

Compares against original method on the weakest benchmarks:
- VAR1-random K=10 (original: 0.700)
- VAR2 K=5 (original: 0.798)
- fMRI K=12 (original: 0.778)

Usage:
    python experiments/exp9_improved.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from data.synthetic import generate_var1, generate_var2, generate_lorenz, generate_regime_switching
from data.fmri import generate_synthetic_fmri
from data.climate import generate_synthetic_climate
from data.finance import generate_synthetic_finance, generate_synthetic_eeg
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.improved_score import (
    ssm_causal_improved,
    extract_full_path_score,
    extract_gradient_weighted_score,
    adaptive_model_config,
    train_with_sparsity,
)
from ssm_causal.utils import compute_metrics, print_metrics_table
from experiments.exp0_core_validation import extract_jacobian_score, extract_weight_score


def run_original(X, G_true, device="cpu"):
    """Run original SSM-Causal (d_model=64, weight projection)."""
    K = X.shape[1]
    model = MambaForTimeSeries(n_variables=K, d_model=64, d_state=16, n_layers=4)
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    t0 = time.time()
    trainer.train(X, n_epochs=150, verbose=False)
    X_tensor = torch.FloatTensor(X)
    S = extract_weight_score(model)
    elapsed = time.time() - t0
    m = compute_metrics(S, G_true)
    m["runtime_s"] = elapsed
    return m, "Weight-Proj d=64"


def run_improved(X, G_true, method, device="cpu"):
    """Run improved SSM-Causal."""
    t0 = time.time()
    S_mean, S_std = ssm_causal_improved(
        X, n_runs=3, device=device, method=method,
        use_sparsity=True, verbose=False,
    )
    elapsed = time.time() - t0
    m = compute_metrics(S_mean, G_true)
    m["runtime_s"] = elapsed
    return m, f"Improved ({method})"


def run_improved_experiment(device="cpu", save_dir="results/exp9_improved"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment 9: Improved SSM-Causal")
    print("Testing full-path, gradient-weighted, and ensemble methods")
    print("=" * 60)

    # All datasets
    datasets = {}

    X, _, G = generate_var1(K=5, T=2000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K5"] = (X, G)

    X, _, G = generate_var1(K=10, T=3000, graph_type="random", seed=42)
    datasets["VAR1-random-K10"] = (X, G)

    X, _, G = generate_var2(K=5, T=3000, seed=42)
    datasets["VAR2-K5"] = (X, G)

    X, _, G = generate_lorenz(T=5000, seed=42, downsample=5)
    datasets["Lorenz-K3"] = (X, G)

    X, _, G = generate_regime_switching(K=3, T=4000, seed=42)
    datasets["Regime-K3"] = (X, G)

    X, G, _ = generate_synthetic_climate(n_indices=8, T=720)
    datasets["Climate-K8"] = (X, G)

    X, G, _ = generate_synthetic_fmri(n_rois=12, T=1200)
    datasets["fMRI-K12"] = (X, G)

    X, G, _ = generate_synthetic_finance(n_sectors=10, T=2500)
    datasets["Finance-K10"] = (X, G)

    X, G, _ = generate_synthetic_eeg(n_channels=8, T=5000)
    datasets["EEG-K8"] = (X, G)

    all_results = {}
    methods_to_test = ["full_path", "gradient_weighted", "ensemble"]

    for ds_name, (X, G_true) in datasets.items():
        K = X.shape[1]
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name} (K={K}, T={X.shape[0]})")
        print(f"{'='*50}")

        ds_results = {}

        # Original
        print(f"  Original (d=64, weight-proj)...", end=" ", flush=True)
        m, label = run_original(X, G_true, device)
        ds_results["Original"] = m
        print(f"AUROC={m['auroc']:.3f}")

        # Improved methods
        for method in methods_to_test:
            print(f"  Improved ({method})...", end=" ", flush=True)
            try:
                m, label = run_improved(X, G_true, method, device)
                ds_results[f"Improved-{method}"] = m
                print(f"AUROC={m['auroc']:.3f}")
            except Exception as e:
                print(f"FAILED: {e}")

        # Granger baseline for comparison
        try:
            from baselines.granger import granger_causality_matrix
            t0 = time.time()
            S_g = granger_causality_matrix(X, max_lag=5)
            m = compute_metrics(S_g, G_true)
            m["runtime_s"] = time.time() - t0
            ds_results["Granger"] = m
            print(f"  Granger: AUROC={m['auroc']:.3f}")
        except Exception:
            pass

        all_results[ds_name] = ds_results
        print_metrics_table(ds_results, dataset_name=ds_name)

    # Save
    results_json = {}
    for ds, methods in all_results.items():
        results_json[ds] = {}
        for method, metrics in methods.items():
            results_json[ds][method] = {
                k: float(v) if not isinstance(v, str) else v
                for k, v in metrics.items()
            }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Original vs Best Improved")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Original':>10} {'Best Improved':>15} {'Granger':>10} {'Delta':>8}")
    print("-" * 63)
    for ds_name, ds_results in all_results.items():
        orig = ds_results.get("Original", {}).get("auroc", 0)
        improved_aurocs = {k: v["auroc"] for k, v in ds_results.items() if k.startswith("Improved")}
        best_improved = max(improved_aurocs.values()) if improved_aurocs else 0
        best_method = max(improved_aurocs, key=improved_aurocs.get) if improved_aurocs else "N/A"
        granger = ds_results.get("Granger", {}).get("auroc", 0)
        delta = best_improved - orig
        print(f"{ds_name:<20} {orig:>10.3f} {best_improved:>15.3f} {granger:>10.3f} {delta:>+8.3f}")

    print(f"\nResults saved to {save_dir}/results.json")
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_improved_experiment(device=device)
