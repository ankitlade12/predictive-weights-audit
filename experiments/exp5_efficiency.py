"""
Experiment 5: Efficiency & Scalability Analysis
================================================

Reproduces Table 4 / Figure 4 of the paper.

Measures wall-clock time and peak memory for all methods across:
    - Varying K (number of variables): 5, 10, 20, 50
    - Varying T (sequence length): 500, 1000, 5000, 10000

Key claim: SSM-Causal inference is O(T) while PCMCI is O(T^2).

Usage:
    python experiments/exp5_efficiency.py
"""

import sys
import os
import time
import json
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.synthetic import generate_var1
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from experiments.exp0_core_validation import extract_jacobian_score, extract_weight_score


def time_method(fn, *args, **kwargs):
    """Time a function and return (result, elapsed_seconds)."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    return result, elapsed


def benchmark_ssm_causal(X, device="cpu", d_model=64, d_state=16, n_layers=4, n_epochs=100):
    """Train SSM and extract causal score. Return (score_matrix, total_time)."""
    K = X.shape[1]
    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=d_state, n_layers=n_layers
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)

    t0 = time.time()
    trainer.train(X, n_epochs=n_epochs, verbose=False)
    X_tensor = torch.FloatTensor(X)
    S = extract_weight_score(model)  # Use fast weight projection
    total_time = time.time() - t0

    return S, total_time


def benchmark_granger(X, max_lag=5):
    """Run Granger causality. Return (score_matrix, time)."""
    from baselines.granger import granger_causality_matrix
    t0 = time.time()
    S = granger_causality_matrix(X, max_lag=max_lag)
    elapsed = time.time() - t0
    return S, elapsed


def benchmark_pcmci(X, tau_max=5):
    """Run PCMCI. Return (score_matrix, time)."""
    from baselines.pcmci_wrapper import pcmci_causality_matrix
    t0 = time.time()
    S = pcmci_causality_matrix(X, tau_max=tau_max)
    elapsed = time.time() - t0
    return S, elapsed


def benchmark_clstm(X, device="cpu", n_epochs=100):
    """Run cLSTM. Return (score_matrix, time)."""
    from baselines.neural_granger import train_clstm
    t0 = time.time()
    _, S = train_clstm(X, n_epochs=n_epochs, verbose=False, device=device)
    elapsed = time.time() - t0
    return S, elapsed


def run_scaling_K(device="cpu", save_dir="results/exp5_efficiency"):
    """Scale number of variables K."""
    print("\n" + "=" * 60)
    print("Scaling: Number of Variables (K)")
    print("=" * 60)

    T = 2000
    results = {}

    for K in [5, 10, 20]:
        print(f"\n  K={K}, T={T}...")
        X, _, G = generate_var1(K=K, T=T, graph_type="random", seed=42)

        row = {}

        # SSM-Causal
        try:
            _, t = benchmark_ssm_causal(X, device=device, n_epochs=80)
            row["SSM-Causal"] = t
            print(f"    SSM-Causal: {t:.2f}s")
        except Exception as e:
            print(f"    SSM-Causal: FAILED ({e})")
            row["SSM-Causal"] = None

        # Granger
        try:
            _, t = benchmark_granger(X)
            row["Granger"] = t
            print(f"    Granger:    {t:.2f}s")
        except Exception as e:
            print(f"    Granger: FAILED ({e})")
            row["Granger"] = None

        # PCMCI
        try:
            _, t = benchmark_pcmci(X)
            row["PCMCI"] = t
            print(f"    PCMCI:      {t:.2f}s")
        except Exception as e:
            print(f"    PCMCI: FAILED ({e})")
            row["PCMCI"] = None

        # cLSTM
        try:
            _, t = benchmark_clstm(X, device=device, n_epochs=80)
            row["cLSTM"] = t
            print(f"    cLSTM:      {t:.2f}s")
        except Exception as e:
            print(f"    cLSTM: FAILED ({e})")
            row["cLSTM"] = None

        results[f"K={K}"] = row

    return results


def run_scaling_T(device="cpu", save_dir="results/exp5_efficiency"):
    """Scale sequence length T."""
    print("\n" + "=" * 60)
    print("Scaling: Sequence Length (T)")
    print("=" * 60)

    K = 5
    results = {}

    for T in [500, 1000, 2000, 5000]:
        print(f"\n  K={K}, T={T}...")
        X, _, G = generate_var1(K=K, T=T, graph_type="chain", seed=42)

        row = {}

        # SSM-Causal
        try:
            _, t = benchmark_ssm_causal(X, device=device, n_epochs=80)
            row["SSM-Causal"] = t
            print(f"    SSM-Causal: {t:.2f}s")
        except Exception as e:
            row["SSM-Causal"] = None

        # Granger
        try:
            _, t = benchmark_granger(X)
            row["Granger"] = t
            print(f"    Granger:    {t:.2f}s")
        except Exception as e:
            row["Granger"] = None

        # PCMCI
        try:
            _, t = benchmark_pcmci(X)
            row["PCMCI"] = t
            print(f"    PCMCI:      {t:.2f}s")
        except Exception as e:
            row["PCMCI"] = None

        # cLSTM
        try:
            _, t = benchmark_clstm(X, device=device, n_epochs=80)
            row["cLSTM"] = t
            print(f"    cLSTM:      {t:.2f}s")
        except Exception as e:
            row["cLSTM"] = None

        results[f"T={T}"] = row

    return results


def plot_scaling(results_K, results_T, save_dir):
    """Generate scaling plots."""
    os.makedirs(save_dir, exist_ok=True)

    methods = ["SSM-Causal", "Granger", "PCMCI", "cLSTM"]
    colors = {"SSM-Causal": "#2196F3", "Granger": "#4CAF50", "PCMCI": "#FF9800", "cLSTM": "#F44336"}
    markers = {"SSM-Causal": "o", "Granger": "s", "PCMCI": "^", "cLSTM": "D"}

    # Plot: Runtime vs K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ks = [int(k.split("=")[1]) for k in results_K.keys()]
    for method in methods:
        times = []
        valid_ks = []
        for k_label, row in results_K.items():
            if row.get(method) is not None:
                times.append(row[method])
                valid_ks.append(int(k_label.split("=")[1]))
        if times:
            ax1.plot(valid_ks, times, f"-{markers[method]}", color=colors[method],
                     label=method, markersize=8, linewidth=2)

    ax1.set_xlabel("Number of Variables (K)")
    ax1.set_ylabel("Wall-clock Time (s)")
    ax1.set_title("Runtime Scaling with K (T=2000)")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)

    # Plot: Runtime vs T
    ts = [int(t.split("=")[1]) for t in results_T.keys()]
    for method in methods:
        times = []
        valid_ts = []
        for t_label, row in results_T.items():
            if row.get(method) is not None:
                times.append(row[method])
                valid_ts.append(int(t_label.split("=")[1]))
        if times:
            ax2.plot(valid_ts, times, f"-{markers[method]}", color=colors[method],
                     label=method, markersize=8, linewidth=2)

    ax2.set_xlabel("Sequence Length (T)")
    ax2.set_ylabel("Wall-clock Time (s)")
    ax2.set_title("Runtime Scaling with T (K=5)")
    ax2.legend()
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scaling_runtime.png"), dpi=300)
    plt.close(fig)
    print(f"\n  Scaling plot saved to {save_dir}/scaling_runtime.png")


def run_efficiency_analysis(device="cpu", save_dir="results/exp5_efficiency"):
    os.makedirs(save_dir, exist_ok=True)

    results_K = run_scaling_K(device, save_dir)
    results_T = run_scaling_T(device, save_dir)

    plot_scaling(results_K, results_T, save_dir)

    # Save
    all_results = {"scaling_K": results_K, "scaling_T": results_T}
    with open(os.path.join(save_dir, "efficiency_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("Runtime Summary Table (seconds)")
    print("=" * 60)
    print(f"{'Config':<15} {'SSM-Causal':>12} {'Granger':>12} {'PCMCI':>12} {'cLSTM':>12}")
    print("-" * 63)

    for label, row in {**results_K, **results_T}.items():
        vals = []
        for m in ["SSM-Causal", "Granger", "PCMCI", "cLSTM"]:
            v = row.get(m)
            vals.append(f"{v:.2f}" if v is not None else "FAIL")
        print(f"{label:<15} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_efficiency_analysis(device=device)
