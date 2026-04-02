"""
Experiment 1: Full Synthetic Benchmark
========================================

Reproduces Table 1 of the paper.

Compares SSM-Causal against all baselines on 5 synthetic datasets:
    1. VAR(1) chain K=5
    2. VAR(1) fork K=5
    3. VAR(1) random K=10
    4. VAR(2) K=5
    5. Nonlinear Lorenz K=3

Reports: AUROC, AUPRC, F1, Precision, Recall, Runtime for each.

Usage:
    python experiments/exp1_synthetic.py
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

from data.synthetic import generate_var1, generate_var2, generate_lorenz
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table, plot_roc_curves
from experiments.exp0_core_validation import extract_causal_score


def run_benchmark(device="cpu", save_dir="results/exp1_synthetic"):
    os.makedirs(save_dir, exist_ok=True)

    # ── Define datasets ──
    datasets = {}

    # VAR(1) chain K=5
    X, A, G = generate_var1(K=5, T=2000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K5"] = (X, G)

    # VAR(1) fork K=5
    X, A, G = generate_var1(K=5, T=2000, graph_type="fork", seed=42)
    datasets["VAR1-fork-K5"] = (X, G)

    # VAR(1) random K=10
    X, A, G = generate_var1(K=10, T=3000, graph_type="random", seed=42)
    datasets["VAR1-random-K10"] = (X, G)

    # VAR(2) K=5
    X, A_dict, G = generate_var2(K=5, T=3000, seed=42)
    datasets["VAR2-K5"] = (X, G)

    # Lorenz K=3
    X, _, G = generate_lorenz(T=5000, seed=42, downsample=5)
    datasets["Lorenz-K3"] = (X, G)

    # ── Define methods ──
    all_results = {}

    for ds_name, (X, G_true) in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} (T={X.shape[0]}, K={X.shape[1]})")
        print(f"{'=' * 60}")

        K = X.shape[1]
        ds_results = {}

        # --- SSM-Causal ---
        print("  Training SSM-Causal...")
        model = MambaForTimeSeries(
            n_variables=K, d_model=64, d_state=16, n_layers=4
        )
        trainer = SSMTrainer(model, lr=1e-3, device=device)

        t0 = time.time()
        trainer.train(X, n_epochs=150, verbose=False)
        X_tensor = torch.FloatTensor(X)
        S_jac, S_bc = extract_causal_score(model, X_tensor, device)
        if S_jac.max() > 0:
            S_jac = S_jac / S_jac.max()
        ssm_time = time.time() - t0

        metrics = compute_metrics(S_jac, G_true)
        metrics["runtime_s"] = ssm_time
        ds_results["SSM-Causal"] = metrics
        print(f"    AUROC: {metrics['auroc']:.3f} ({ssm_time:.1f}s)")

        # --- Granger ---
        try:
            from baselines.granger import granger_causality_matrix
            t0 = time.time()
            S_g = granger_causality_matrix(X, max_lag=5)
            g_time = time.time() - t0
            m = compute_metrics(S_g, G_true)
            m["runtime_s"] = g_time
            ds_results["Granger"] = m
            print(f"    Granger AUROC: {m['auroc']:.3f} ({g_time:.1f}s)")
        except ImportError:
            print("    [SKIP] Granger")

        # --- PCMCI ---
        try:
            from baselines.pcmci_wrapper import pcmci_causality_matrix
            t0 = time.time()
            S_p = pcmci_causality_matrix(X, tau_max=5)
            p_time = time.time() - t0
            m = compute_metrics(S_p, G_true)
            m["runtime_s"] = p_time
            ds_results["PCMCI"] = m
            print(f"    PCMCI AUROC: {m['auroc']:.3f} ({p_time:.1f}s)")
        except ImportError:
            print("    [SKIP] PCMCI")

        # --- cLSTM ---
        try:
            from baselines.neural_granger import train_clstm
            t0 = time.time()
            _, S_c = train_clstm(X, n_epochs=200, verbose=False, device=device)
            c_time = time.time() - t0
            m = compute_metrics(S_c, G_true)
            m["runtime_s"] = c_time
            ds_results["cLSTM"] = m
            print(f"    cLSTM AUROC: {m['auroc']:.3f} ({c_time:.1f}s)")
        except Exception as e:
            print(f"    [SKIP] cLSTM: {e}")

        # --- Correlation ---
        S_corr = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(S_corr, 0)
        if S_corr.max() > 0:
            S_corr = S_corr / S_corr.max()
        m = compute_metrics(S_corr, G_true)
        m["runtime_s"] = 0.001
        ds_results["Correlation"] = m

        all_results[ds_name] = ds_results

        # Print table for this dataset
        print_metrics_table(ds_results, dataset_name=ds_name)

    # ── Save results ──
    # Convert to serializable format
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

    print(f"\nResults saved to {save_dir}/results.json")
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_benchmark(device=device)
