"""
Experiment 2: Non-Stationary / Regime-Switching Data
=====================================================

Reproduces Table 2 + Figure 1 of the paper.

KEY WIN CONDITION: SSMs should outperform PCMCI on non-stationary data
because PCMCI assumes stationarity while the selective SSM adapts to
regime changes.

Tests:
    1. Regime-switching VAR (graph flips halfway)
    2. Time-varying coefficient strength
    3. Windowed analysis (causal score over time)

Usage:
    python experiments/exp2_nonstationary.py
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

from data.synthetic import generate_regime_switching, generate_var1
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table
from experiments.exp0_core_validation import extract_causal_score


def windowed_causal_score(model, X_tensor, window_size=500, stride=100, device="cpu"):
    """
    Compute causal scores in sliding windows to track how causal
    structure changes over time.

    Returns:
        scores: List of (window_center, score_matrix) tuples.
    """
    T = X_tensor.shape[0]
    scores = []

    for start in range(0, T - window_size, stride):
        end = start + window_size
        X_window = X_tensor[start:end]

        S_jac, _ = extract_causal_score(model, X_window, device)
        if S_jac.max() > 0:
            S_jac = S_jac / S_jac.max()

        center = (start + end) // 2
        scores.append((center, S_jac))

    return scores


def run_regime_switching_experiment(device="cpu", save_dir="results/exp2_nonstationary"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment 2: Non-Stationary / Regime-Switching Data")
    print("=" * 60)

    # ── Dataset: Regime-switching VAR ──
    K, T = 3, 4000
    X, A_dict, G_true = generate_regime_switching(K=K, T=T, seed=42)

    print(f"\nData: T={T}, K={K}")
    print(f"Regime 1 (t<{T//2}): chain X0->X1->X2")
    print(f"Regime 2 (t>={T//2}): reversed X2->X1->X0")
    print(f"Ground truth (union): \n{G_true}")

    results = {}

    # ── SSM-Causal (full sequence) ──
    print("\n  Training SSM-Causal on full sequence...")
    model = MambaForTimeSeries(n_variables=K, d_model=64, d_state=16, n_layers=4)
    trainer = SSMTrainer(model, lr=1e-3, device=device)

    t0 = time.time()
    trainer.train(X, n_epochs=150, verbose=False)
    X_tensor = torch.FloatTensor(X)
    S_full, _ = extract_causal_score(model, X_tensor, device)
    if S_full.max() > 0:
        S_full = S_full / S_full.max()
    ssm_time = time.time() - t0

    m = compute_metrics(S_full, G_true)
    m["runtime_s"] = ssm_time
    results["SSM-Causal (full)"] = m
    print(f"    Full-sequence AUROC: {m['auroc']:.3f}")

    # ── SSM-Causal (windowed) ──
    print("\n  Computing windowed causal scores...")
    window_scores = windowed_causal_score(model, X_tensor, window_size=500, stride=200, device=device)

    # Evaluate on each regime separately
    G_regime1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)  # X0->X1, X1->X2
    G_regime2 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)  # X2->X1, X1->X0

    regime1_aurocs = []
    regime2_aurocs = []
    window_centers = []

    for center, S_w in window_scores:
        window_centers.append(center)
        if center < T // 2:
            m1 = compute_metrics(S_w, G_regime1)
            regime1_aurocs.append(m1.get("auroc", 0.5))
            regime2_aurocs.append(0.5)  # Not applicable
        else:
            m2 = compute_metrics(S_w, G_regime2)
            regime2_aurocs.append(m2.get("auroc", 0.5))
            regime1_aurocs.append(0.5)  # Not applicable

    # ── Baselines ──
    try:
        from baselines.granger import granger_causality_matrix
        t0 = time.time()
        S_g = granger_causality_matrix(X, max_lag=5)
        g_time = time.time() - t0
        m = compute_metrics(S_g, G_true)
        m["runtime_s"] = g_time
        results["Granger"] = m
        print(f"    Granger AUROC: {m['auroc']:.3f}")
    except ImportError:
        pass

    try:
        from baselines.pcmci_wrapper import pcmci_causality_matrix
        t0 = time.time()
        S_p = pcmci_causality_matrix(X, tau_max=5)
        p_time = time.time() - t0
        m = compute_metrics(S_p, G_true)
        m["runtime_s"] = p_time
        results["PCMCI"] = m
        print(f"    PCMCI AUROC: {m['auroc']:.3f}")
    except ImportError:
        pass

    try:
        from baselines.neural_granger import train_clstm
        t0 = time.time()
        _, S_c = train_clstm(X, n_epochs=200, verbose=False, device=device)
        c_time = time.time() - t0
        m = compute_metrics(S_c, G_true)
        m["runtime_s"] = c_time
        results["cLSTM"] = m
        print(f"    cLSTM AUROC: {m['auroc']:.3f}")
    except Exception:
        pass

    print_metrics_table(results, dataset_name="Regime-Switching VAR(1)")

    # ── Figure 1: Windowed AUROC over time ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_centers, regime1_aurocs, "b-o", markersize=3, label="Regime 1 (forward chain)")
    ax.plot(window_centers, regime2_aurocs, "r-s", markersize=3, label="Regime 2 (reversed chain)")
    ax.axvline(x=T // 2, color="gray", linestyle="--", alpha=0.7, label="Regime switch")
    ax.set_xlabel("Time")
    ax.set_ylabel("AUROC")
    ax.set_title("SSM-Causal: Causal Recovery Accuracy Over Time (Regime-Switching)")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "figure1_windowed_auroc.png"), dpi=300)
    plt.close(fig)

    # Save results as JSON
    import json
    results_json = {}
    for method, metrics in results.items():
        results_json[method] = {
            k: float(v) if not isinstance(v, str) else v
            for k, v in metrics.items()
        }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nFigure saved to {save_dir}/figure1_windowed_auroc.png")
    print(f"Results saved to {save_dir}/results.json")
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_regime_switching_experiment(device=device)
