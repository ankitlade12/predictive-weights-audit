"""
Week 2: Causal Discovery Baselines
====================================

Understand Granger causality and PCMCI well enough to know exactly
what we're comparing against.

Tests:
    1. Granger test from scratch on VAR(1) — verify it works
    2. PCMCI on VAR(1) — verify it works
    3. Both methods on 5 synthetic datasets — calibrate AUROC ranges
    4. SSM-Causal vs both baselines — first real comparison

Goal: Confirm SSM method is at least competitive on linear data.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.synthetic import (
    generate_var1, generate_var2, generate_lorenz, generate_regime_switching,
)
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table, plot_roc_curves
from experiments.exp0_core_validation import extract_jacobian_score, extract_weight_score


def train_and_extract(X, device="cpu", d_model=32, n_layers=2, n_epochs=80):
    """Train SSM and extract causal scores."""
    K = X.shape[1]
    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=16, n_layers=n_layers, dropout=0.05,
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=n_epochs, batch_size=16, seq_len=100, verbose=False)

    X_tensor = torch.FloatTensor(X)
    S_jac = extract_jacobian_score(model, X_tensor, device, seq_len=100)
    S_wt = extract_weight_score(model)
    return S_jac, S_wt


def run_week2(device="cpu", save_dir="results/week2_baselines"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("WEEK 2: Granger + PCMCI Baselines on Synthetic Data")
    print("=" * 70)

    # ── Define datasets ──
    datasets = {}

    X, _, G = generate_var1(K=3, T=1000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K3"] = (X, G)

    X, _, G = generate_var1(K=5, T=1000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K5"] = (X, G)

    X, _, G = generate_var1(K=5, T=1000, graph_type="fork", seed=42)
    datasets["VAR1-fork-K5"] = (X, G)

    X, _, G = generate_var1(K=5, T=1500, graph_type="random", seed=42)
    datasets["VAR1-random-K5"] = (X, G)

    X, _, G = generate_var2(K=5, T=1500, seed=42)
    datasets["VAR2-K5"] = (X, G)

    # ── Run all methods on all datasets ──
    all_results = {}

    for ds_name, (X, G_true) in datasets.items():
        K = X.shape[1]
        T = X.shape[0]
        n_edges = int(G_true.sum())
        n_possible = K * (K - 1)
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name} | T={T}, K={K} | Edges: {n_edges}/{n_possible}")
        print(f"{'─' * 60}")

        ds_results = {}
        ds_scores = {}

        # ── SSM-Causal ──
        t0 = time.time()
        S_jac, S_wt = train_and_extract(X, device)
        ssm_time = time.time() - t0

        m = compute_metrics(S_jac, G_true)
        m["runtime_s"] = ssm_time
        ds_results["SSM-Jacobian"] = m
        ds_scores["SSM-Jacobian"] = S_jac
        print(f"  SSM-Jacobian  AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}  ({ssm_time:.1f}s)")

        m = compute_metrics(S_wt, G_true)
        m["runtime_s"] = ssm_time
        ds_results["SSM-Weight"] = m
        ds_scores["SSM-Weight"] = S_wt
        print(f"  SSM-Weight    AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}")

        # ── Granger ──
        from baselines.granger import granger_causality_matrix
        t0 = time.time()
        S_g = granger_causality_matrix(X, max_lag=5)
        g_time = time.time() - t0
        m = compute_metrics(S_g, G_true)
        m["runtime_s"] = g_time
        ds_results["Granger"] = m
        ds_scores["Granger"] = S_g
        print(f"  Granger       AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}  ({g_time:.1f}s)")

        # ── PCMCI ──
        try:
            from baselines.pcmci_wrapper import pcmci_causality_matrix
            t0 = time.time()
            S_p = pcmci_causality_matrix(X, tau_max=5)
            p_time = time.time() - t0
            m = compute_metrics(S_p, G_true)
            m["runtime_s"] = p_time
            ds_results["PCMCI"] = m
            ds_scores["PCMCI"] = S_p
            print(f"  PCMCI         AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}  ({p_time:.1f}s)")
        except Exception as e:
            print(f"  PCMCI         [ERROR] {e}")

        # ── Correlation (sanity) ──
        S_corr = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(S_corr, 0)
        if S_corr.max() > 0:
            S_corr = S_corr / S_corr.max()
        m = compute_metrics(S_corr, G_true)
        ds_results["Correlation"] = m
        ds_scores["Correlation"] = S_corr
        print(f"  Correlation   AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}")

        all_results[ds_name] = ds_results

        # ── ROC curves per dataset ──
        try:
            fig = plot_roc_curves(
                ds_scores, G_true,
                save_path=os.path.join(save_dir, f"roc_{ds_name}.png"),
            )
            plt.close(fig)
        except Exception:
            pass

    # ── Summary table ──
    print("\n\n" + "=" * 70)
    print("WEEK 2 SUMMARY — AUROC across all datasets")
    print("=" * 70)

    methods = ["SSM-Jacobian", "SSM-Weight", "Granger", "PCMCI", "Correlation"]
    header = f"{'Dataset':<20}" + "".join(f"{m:>15}" for m in methods)
    print(header)
    print("-" * len(header))

    for ds_name, ds_results in all_results.items():
        row = f"{ds_name:<20}"
        for method in methods:
            if method in ds_results:
                auroc = ds_results[method]["auroc"]
                row += f"{auroc:>15.3f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    # ── Win/loss analysis ──
    print("\n\nWin/Loss Analysis (SSM-Jacobian vs baselines):")
    ssm_wins = 0
    total = 0
    for ds_name, ds_results in all_results.items():
        ssm_auroc = ds_results.get("SSM-Jacobian", {}).get("auroc", 0)
        granger_auroc = ds_results.get("Granger", {}).get("auroc", 0)
        pcmci_auroc = ds_results.get("PCMCI", {}).get("auroc", 0)
        best_baseline = max(granger_auroc, pcmci_auroc)

        if ssm_auroc >= best_baseline:
            ssm_wins += 1
            status = "WIN"
        else:
            status = f"LOSE (gap={best_baseline - ssm_auroc:.3f})"
        total += 1
        print(f"  {ds_name:<20} SSM={ssm_auroc:.3f} vs best_baseline={best_baseline:.3f} -> {status}")

    print(f"\n  SSM wins {ssm_wins}/{total} datasets")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    run_week2(device=device)
