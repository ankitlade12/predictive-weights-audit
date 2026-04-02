"""
Week 3: Full Benchmark — All Methods on All Datasets
======================================================

Goals:
    1. Add cLSTM (Neural Granger) baseline
    2. Add regime-switching + Lorenz datasets
    3. Full comparison table: SSM vs Granger vs PCMCI vs cLSTM vs Correlation
    4. Identify where SSM wins and loses

Hypotheses to test:
    - SSM wins on non-stationary data (regime-switching)
    - SSM wins on long sequences (efficiency)
    - SSM loses on sparse graphs with weak edges
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
import matplotlib.pyplot as plt

from data.synthetic import (
    generate_var1, generate_var2, generate_lorenz,
    generate_regime_switching, generate_henon,
)
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table, plot_roc_curves
from baselines.granger import granger_causality_matrix
from baselines.neural_granger import train_clstm
from experiments.exp0_core_validation import extract_jacobian_score, extract_weight_score


def train_ssm_and_extract(X, device="cpu"):
    K = X.shape[1]
    model = MambaForTimeSeries(
        n_variables=K, d_model=32, d_state=16, n_layers=2, dropout=0.05,
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=80, batch_size=16, seq_len=100, verbose=False)
    X_t = torch.FloatTensor(X)
    S_jac = extract_jacobian_score(model, X_t, device, seq_len=100)
    S_wt = extract_weight_score(model)
    return S_jac, S_wt


def run_method(name, func, X, G_true):
    """Run a single method with timing."""
    t0 = time.time()
    try:
        S = func(X)
        runtime = time.time() - t0
        m = compute_metrics(S, G_true)
        m["runtime_s"] = runtime
        return m, S
    except Exception as e:
        print(f"    [{name}] ERROR: {e}")
        return None, None


def run_week3(device="cpu", save_dir="results/week3_benchmark"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("WEEK 3: Full Benchmark — All Methods × All Datasets")
    print("=" * 70)

    # ── Datasets ──
    datasets = {}

    X, _, G = generate_var1(K=3, T=1000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K3"] = (X, G)

    X, _, G = generate_var1(K=5, T=1000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K5"] = (X, G)

    X, _, G = generate_var1(K=5, T=1000, graph_type="fork", seed=42)
    datasets["VAR1-fork-K5"] = (X, G)

    X, _, G = generate_regime_switching(K=3, T=2000, seed=42)
    datasets["Regime-switch-K3"] = (X, G)

    X, _, G = generate_lorenz(T=3000, seed=42, downsample=3)
    datasets["Lorenz-K3"] = (X, G)

    # ── Methods ──
    all_results = {}
    summary_rows = []

    for ds_name, (X, G_true) in datasets.items():
        K = X.shape[1]
        T = X.shape[0]
        n_edges = int(G_true.sum())
        print(f"\n{'─' * 60}")
        print(f"{ds_name} | T={T}, K={K} | Edges: {n_edges}/{K*(K-1)}")
        print(f"{'─' * 60}")

        ds_results = {}
        ds_scores = {}

        # SSM-Causal
        t0 = time.time()
        S_jac, S_wt = train_ssm_and_extract(X, device)
        ssm_time = time.time() - t0

        m = compute_metrics(S_jac, G_true)
        m["runtime_s"] = ssm_time
        ds_results["SSM-Causal"] = m
        ds_scores["SSM-Causal"] = S_jac
        print(f"  SSM-Causal    AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}  ({ssm_time:.1f}s)")

        # Granger
        t0 = time.time()
        S_g = granger_causality_matrix(X, max_lag=5)
        g_time = time.time() - t0
        m = compute_metrics(S_g, G_true)
        m["runtime_s"] = g_time
        ds_results["Granger"] = m
        ds_scores["Granger"] = S_g
        print(f"  Granger       AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}  ({g_time:.1f}s)")

        # PCMCI
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

        # cLSTM
        t0 = time.time()
        _, S_c = train_clstm(X, n_epochs=150, hidden_size=32, verbose=False, device=device)
        c_time = time.time() - t0
        m = compute_metrics(S_c, G_true)
        m["runtime_s"] = c_time
        ds_results["cLSTM"] = m
        ds_scores["cLSTM"] = S_c
        print(f"  cLSTM         AUROC={m['auroc']:.3f}  F1={m['best_f1']:.3f}  ({c_time:.1f}s)")

        # Correlation
        S_corr = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(S_corr, 0)
        if S_corr.max() > 0:
            S_corr = S_corr / S_corr.max()
        m = compute_metrics(S_corr, G_true)
        m["runtime_s"] = 0.0
        ds_results["Correlation"] = m
        ds_scores["Correlation"] = S_corr

        all_results[ds_name] = ds_results

        # ROC figure
        try:
            fig = plot_roc_curves(ds_scores, G_true,
                                  save_path=os.path.join(save_dir, f"roc_{ds_name}.png"))
            plt.close(fig)
        except Exception:
            pass

        # Collect for summary
        summary_rows.append({
            "dataset": ds_name,
            **{f"{method}_auroc": ds_results.get(method, {}).get("auroc", float("nan"))
               for method in ["SSM-Causal", "Granger", "PCMCI", "cLSTM", "Correlation"]},
            **{f"{method}_time": ds_results.get(method, {}).get("runtime_s", float("nan"))
               for method in ["SSM-Causal", "Granger", "PCMCI", "cLSTM"]},
        })

    # ── Summary ──
    print("\n\n" + "=" * 80)
    print("WEEK 3 SUMMARY — AUROC Table")
    print("=" * 80)
    methods = ["SSM-Causal", "Granger", "PCMCI", "cLSTM", "Correlation"]
    header = f"{'Dataset':<22}" + "".join(f"{m:>13}" for m in methods)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = f"{row['dataset']:<22}"
        for method in methods:
            auroc = row.get(f"{method}_auroc", float("nan"))
            line += f"{auroc:>13.3f}"
        print(line)

    print("\n\nRuntime Table (seconds)")
    print("-" * 70)
    header = f"{'Dataset':<22}" + "".join(f"{m:>13}" for m in ["SSM-Causal", "Granger", "PCMCI", "cLSTM"])
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = f"{row['dataset']:<22}"
        for method in ["SSM-Causal", "Granger", "PCMCI", "cLSTM"]:
            t = row.get(f"{method}_time", float("nan"))
            line += f"{t:>13.1f}"
        print(line)

    # ── Key analysis ──
    print("\n\nKEY FINDINGS:")
    for row in summary_rows:
        ds = row["dataset"]
        ssm = row.get("SSM-Causal_auroc", 0)
        granger = row.get("Granger_auroc", 0)
        pcmci = row.get("PCMCI_auroc", 0)
        clstm = row.get("cLSTM_auroc", 0)
        best_baseline = max(granger, pcmci, clstm)
        if ssm >= best_baseline:
            print(f"  {ds}: SSM WINS ({ssm:.3f} >= {best_baseline:.3f})")
        else:
            gap = best_baseline - ssm
            winner = "Granger" if granger == best_baseline else ("PCMCI" if pcmci == best_baseline else "cLSTM")
            print(f"  {ds}: {winner} wins (gap={gap:.3f})")

    # Save JSON
    results_json = {}
    for ds, methods_dict in all_results.items():
        results_json[ds] = {}
        for method, metrics in methods_dict.items():
            results_json[ds][method] = {k: float(v) for k, v in metrics.items()}

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {save_dir}/")
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    run_week3(device=device)
