"""
Experiment 3: Climate Teleconnection Recovery
==============================================

Reproduces Figure 2 + Table 3 (climate portion) of the paper.

Tests whether SSM-Causal can recover known climate teleconnections
(ENSO -> IOD, ENSO -> PNA, PDO -> ENSO, etc.) from multivariate
climate index time series.

Uses synthetic climate data with known ground truth for reproducibility,
with option to run on real NOAA data when available.

Usage:
    python experiments/exp3_climate.py
    python experiments/exp3_climate.py --real-data data/climate_raw/
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.climate import generate_synthetic_climate, load_noaa_indices, get_climate_ground_truth
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table
from experiments.exp0_core_validation import extract_jacobian_score, extract_weight_score


def run_climate_experiment(
    use_real_data: bool = False,
    data_dir: str = "data/climate_raw",
    device: str = "cpu",
    save_dir: str = "results/exp3_climate",
):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment 3: Climate Teleconnection Recovery")
    print("=" * 60)

    # ── Load data ──
    if use_real_data:
        print("\n  Loading real NOAA climate indices...")
        X, index_names = load_noaa_indices(data_dir)
        if X is None:
            print("  Real data not found. Falling back to synthetic.")
            X, G_true, index_names = generate_synthetic_climate()
        else:
            G_true = get_climate_ground_truth(index_names)
    else:
        print("\n  Using synthetic climate data (known ground truth)...")
        X, G_true, index_names = generate_synthetic_climate(n_indices=8, T=720)

    K = X.shape[1]
    T = X.shape[0]
    print(f"  Variables ({K}): {', '.join(index_names)}")
    print(f"  Timesteps: {T} (monthly)")
    print(f"  Ground truth edges: {int(G_true.sum())}")

    results = {}

    # ── SSM-Causal ──
    print("\n  Training SSM-Causal...")
    model = MambaForTimeSeries(
        n_variables=K, d_model=64, d_state=16, n_layers=4
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)

    t0 = time.time()
    trainer.train(X, n_epochs=200, verbose=False)
    X_tensor = torch.FloatTensor(X)

    # Default extractor: weight projection (locked, no oracle selection)
    S_ssm = extract_weight_score(model)
    ssm_time = time.time() - t0

    m = compute_metrics(S_ssm, G_true)
    variant_used = "Weight-Proj"

    m["runtime_s"] = ssm_time
    results["SSM-Causal"] = m
    print(f"    AUROC: {m['auroc']:.3f} (variant: {variant_used}, {ssm_time:.1f}s)")

    # ── Baselines ──
    try:
        from baselines.granger import granger_causality_matrix
        t0 = time.time()
        S_g = granger_causality_matrix(X, max_lag=12)
        g_time = time.time() - t0
        m = compute_metrics(S_g, G_true)
        m["runtime_s"] = g_time
        results["Granger"] = m
        print(f"    Granger AUROC: {m['auroc']:.3f} ({g_time:.1f}s)")
    except Exception as e:
        print(f"    [SKIP] Granger: {e}")

    try:
        from baselines.pcmci_wrapper import pcmci_causality_matrix
        t0 = time.time()
        S_p = pcmci_causality_matrix(X, tau_max=12)
        p_time = time.time() - t0
        m = compute_metrics(S_p, G_true)
        m["runtime_s"] = p_time
        results["PCMCI"] = m
        print(f"    PCMCI AUROC: {m['auroc']:.3f} ({p_time:.1f}s)")
    except Exception as e:
        print(f"    [SKIP] PCMCI: {e}")

    try:
        from baselines.neural_granger import train_clstm
        t0 = time.time()
        _, S_c = train_clstm(X, n_epochs=200, verbose=False, device=device)
        c_time = time.time() - t0
        m = compute_metrics(S_c, G_true)
        m["runtime_s"] = c_time
        results["cLSTM"] = m
        print(f"    cLSTM AUROC: {m['auroc']:.3f} ({c_time:.1f}s)")
    except Exception as e:
        print(f"    [SKIP] cLSTM: {e}")

    # Correlation
    S_corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(S_corr, 0)
    if S_corr.max() > 0:
        S_corr = S_corr / S_corr.max()
    m = compute_metrics(S_corr, G_true)
    results["Correlation"] = m

    print_metrics_table(results, dataset_name="Climate Teleconnections")

    # ── Figure 2: Climate causal graph ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    im0 = axes[0].imshow(G_true, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth\n(Known Teleconnections)")
    axes[0].set_xticks(range(K))
    axes[0].set_yticks(range(K))
    axes[0].set_xticklabels(index_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(index_names, fontsize=8)

    # SSM-Causal
    im1 = axes[1].imshow(S_ssm, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_title(f"SSM-Causal\n(AUROC={results['SSM-Causal']['auroc']:.3f})")
    axes[1].set_xticks(range(K))
    axes[1].set_yticks(range(K))
    axes[1].set_xticklabels(index_names, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(index_names, fontsize=8)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Correlation (for comparison)
    im2 = axes[2].imshow(S_corr, cmap="RdBu_r", vmin=0, vmax=1)
    axes[2].set_title(f"Correlation\n(AUROC={results['Correlation']['auroc']:.3f})")
    axes[2].set_xticks(range(K))
    axes[2].set_yticks(range(K))
    axes[2].set_xticklabels(index_names, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticklabels(index_names, fontsize=8)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle("Climate Teleconnection Recovery", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "figure2_climate.png"), dpi=300)
    plt.close(fig)

    # Save results
    results_json = {}
    for method, metrics in results.items():
        results_json[method] = {
            k: float(v) if not isinstance(v, str) else v
            for k, v in metrics.items()
        }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nFigure saved to {save_dir}/figure2_climate.png")
    print(f"Results saved to {save_dir}/results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=str, default=None,
                        help="Path to real NOAA climate data directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_climate_experiment(
        use_real_data=args.real_data is not None,
        data_dir=args.real_data or "data/climate_raw",
        device=device,
    )
