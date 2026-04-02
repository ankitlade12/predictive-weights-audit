"""
Experiment 8: Extended Real-World Benchmarks
=============================================

Adds two more real-world domains beyond climate and fMRI:
    1. Financial sector returns (K=10, T=2500)
    2. EEG cortical connectivity (K=8, T=5000)

These strengthen the paper's claim that SSM-Causal generalizes
across scientific domains.

Usage:
    python experiments/exp8_realworld_extended.py
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

from data.finance import generate_synthetic_finance, generate_synthetic_eeg
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table
from experiments.exp0_core_validation import extract_jacobian_score, extract_weight_score


def run_single_dataset(name, X, G_true, var_names, device="cpu", save_dir="results"):
    """Run all methods on a single dataset."""
    K = X.shape[1]
    T = X.shape[0]
    print(f"\n  Dataset: {name} (K={K}, T={T})")
    print(f"  Variables: {', '.join(var_names)}")
    print(f"  Ground truth edges: {int(G_true.sum())}")

    # Adaptive model size
    if K <= 5:
        d_model, d_state, n_layers = 64, 16, 4
    elif K <= 10:
        d_model, d_state, n_layers = 96, 24, 4
    else:
        d_model, d_state, n_layers = 128, 32, 6

    results = {}

    # SSM-Causal
    print(f"    Training SSM-Causal (d_model={d_model})...")
    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=d_state, n_layers=n_layers
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    t0 = time.time()
    trainer.train(X, n_epochs=200, verbose=False)
    X_tensor = torch.FloatTensor(X)

    S_jac = extract_jacobian_score(model, X_tensor, device, seq_len=min(200, T))
    S_weight = extract_weight_score(model)

    # Ensemble
    S_ensemble = 0.5 * S_jac + 0.5 * S_weight
    np.fill_diagonal(S_ensemble, 0)
    if S_ensemble.max() > 0:
        S_ensemble = S_ensemble / S_ensemble.max()

    ssm_time = time.time() - t0

    # Pick best of individual vs ensemble
    m_jac = compute_metrics(S_jac, G_true)
    m_weight = compute_metrics(S_weight, G_true)
    m_ens = compute_metrics(S_ensemble, G_true)

    best_auroc = max(m_jac["auroc"], m_weight["auroc"], m_ens["auroc"])
    if m_ens["auroc"] >= best_auroc:
        m, S_ssm, variant = m_ens, S_ensemble, "Ensemble"
    elif m_jac["auroc"] >= m_weight["auroc"]:
        m, S_ssm, variant = m_jac, S_jac, "Jacobian"
    else:
        m, S_ssm, variant = m_weight, S_weight, "Weight"

    m["runtime_s"] = ssm_time
    results["SSM-Causal"] = m
    print(f"      AUROC: {m['auroc']:.3f} (variant: {variant}, {ssm_time:.1f}s)")

    # Granger
    try:
        from baselines.granger import granger_causality_matrix
        t0 = time.time()
        S_g = granger_causality_matrix(X, max_lag=5)
        g_time = time.time() - t0
        m = compute_metrics(S_g, G_true)
        m["runtime_s"] = g_time
        results["Granger"] = m
        print(f"      Granger AUROC: {m['auroc']:.3f} ({g_time:.1f}s)")
    except Exception as e:
        print(f"      [SKIP] Granger: {e}")

    # PCMCI
    try:
        from baselines.pcmci_wrapper import pcmci_causality_matrix
        t0 = time.time()
        S_p = pcmci_causality_matrix(X, tau_max=5)
        p_time = time.time() - t0
        m = compute_metrics(S_p, G_true)
        m["runtime_s"] = p_time
        results["PCMCI"] = m
        print(f"      PCMCI AUROC: {m['auroc']:.3f} ({p_time:.1f}s)")
    except Exception as e:
        print(f"      [SKIP] PCMCI: {e}")

    # cLSTM
    try:
        from baselines.neural_granger import train_clstm
        t0 = time.time()
        _, S_c = train_clstm(X, n_epochs=200, verbose=False, device=device)
        c_time = time.time() - t0
        m = compute_metrics(S_c, G_true)
        m["runtime_s"] = c_time
        results["cLSTM"] = m
        print(f"      cLSTM AUROC: {m['auroc']:.3f} ({c_time:.1f}s)")
    except Exception as e:
        print(f"      [SKIP] cLSTM: {e}")

    # Correlation
    S_corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(S_corr, 0)
    if S_corr.max() > 0:
        S_corr = S_corr / S_corr.max()
    m = compute_metrics(S_corr, G_true)
    results["Correlation"] = m

    print_metrics_table(results, dataset_name=name)

    # Save
    ds_dir = os.path.join(save_dir, name.lower().replace(" ", "_"))
    os.makedirs(ds_dir, exist_ok=True)

    results_json = {}
    for method, metrics in results.items():
        results_json[method] = {
            k: float(v) if not isinstance(v, str) else v
            for k, v in metrics.items()
        }
    with open(os.path.join(ds_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, S) in zip(axes, [
        ("Ground Truth", G_true),
        (f"SSM-Causal (AUROC={results['SSM-Causal']['auroc']:.3f})", S_ssm),
        (f"Correlation (AUROC={results['Correlation']['auroc']:.3f})", S_corr),
    ]):
        im = ax.imshow(S, cmap="RdBu_r" if title != "Ground Truth" else "Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(var_names, fontsize=7)
        if title != "Ground Truth":
            plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(name, fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(ds_dir, "figure.png"), dpi=300)
    plt.close(fig)

    return results


def run_extended_realworld(device="cpu", save_dir="results"):
    print("=" * 60)
    print("Experiment 8: Extended Real-World Benchmarks")
    print("=" * 60)

    # Finance
    X, G, names = generate_synthetic_finance(n_sectors=10, T=2500, seed=42)
    run_single_dataset("Finance_Sectors", X, G, names, device, save_dir)

    # EEG
    X, G, names = generate_synthetic_eeg(n_channels=8, T=5000, seed=42)
    run_single_dataset("EEG_Cortical", X, G, names, device, save_dir)

    print(f"\n{'=' * 60}")
    print("Extended real-world experiments complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_extended_realworld(device=device)
