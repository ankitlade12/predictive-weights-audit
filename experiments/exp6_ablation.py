"""
Experiment 6: Ablation Study
=============================

Reproduces Table 5 of the paper.

Ablates every design choice:
    1. Score variant: time_avg vs gate_weighted vs layer_agg vs jacobian vs weight_proj
    2. Model size: d_model in {32, 64, 128}
    3. State dimension: d_state in {8, 16, 32}
    4. Number of layers: n_layers in {2, 4, 6}
    5. Training length effect: T in {500, 1000, 2000, 5000}

Usage:
    python experiments/exp6_ablation.py
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

from data.synthetic import generate_var1, generate_lorenz, generate_regime_switching
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics
from experiments.exp0_core_validation import (
    extract_jacobian_score,
    extract_weight_score,
    extract_bc_projected_score,
)


def run_single_experiment(
    X, G_true, d_model=64, d_state=16, n_layers=4, n_epochs=150,
    device="cpu", seed=42,
):
    """Train model and extract scores with all variants."""
    torch.manual_seed(seed)
    K = X.shape[1]

    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=d_state, n_layers=n_layers
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)

    t0 = time.time()
    trainer.train(X, n_epochs=n_epochs, verbose=False)
    train_time = time.time() - t0

    X_tensor = torch.FloatTensor(X)

    results = {}

    # Jacobian score
    t0 = time.time()
    S_jac = extract_jacobian_score(model, X_tensor, device, seq_len=100)
    jac_time = time.time() - t0
    m = compute_metrics(S_jac, G_true)
    m["runtime_s"] = train_time + jac_time
    results["Jacobian"] = m

    # Weight projection score
    t0 = time.time()
    S_weight = extract_weight_score(model)
    w_time = time.time() - t0
    m = compute_metrics(S_weight, G_true)
    m["runtime_s"] = train_time + w_time
    results["Weight-Proj"] = m

    # B·C^T projected score
    t0 = time.time()
    S_bc = extract_bc_projected_score(model, X_tensor, device)
    bc_time = time.time() - t0
    m = compute_metrics(S_bc, G_true)
    m["runtime_s"] = train_time + bc_time
    results["BC-Proj"] = m

    return results, train_time


def ablation_score_variant(device="cpu"):
    """Ablation 1: Compare score extraction variants across datasets."""
    print("\n" + "=" * 60)
    print("Ablation 1: Score Extraction Variant")
    print("=" * 60)

    datasets = {}
    X, _, G = generate_var1(K=3, T=2000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K3"] = (X, G)
    X, _, G = generate_var1(K=5, T=2000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K5"] = (X, G)
    X, _, G = generate_regime_switching(K=3, T=4000, seed=42)
    datasets["Regime-K3"] = (X, G)
    X, _, G = generate_lorenz(T=5000, seed=42, downsample=5)
    datasets["Lorenz-K3"] = (X, G)

    all_results = {}
    for ds_name, (X, G_true) in datasets.items():
        print(f"\n  {ds_name}...")
        results, _ = run_single_experiment(X, G_true, device=device)
        all_results[ds_name] = results

        for method, m in results.items():
            print(f"    {method:15s} AUROC={m['auroc']:.3f}  F1={m['f1']:.3f}")

    return all_results


def ablation_model_size(device="cpu"):
    """Ablation 2: Effect of d_model."""
    print("\n" + "=" * 60)
    print("Ablation 2: Model Size (d_model)")
    print("=" * 60)

    X, _, G_true = generate_var1(K=5, T=2000, graph_type="chain", seed=42)

    results = {}
    for d_model in [32, 64, 128]:
        print(f"\n  d_model={d_model}...")
        r, train_time = run_single_experiment(
            X, G_true, d_model=d_model, d_state=16, n_layers=4, device=device
        )
        results[f"d_model={d_model}"] = {
            "auroc_jac": r["Jacobian"]["auroc"],
            "auroc_weight": r["Weight-Proj"]["auroc"],
            "f1_jac": r["Jacobian"]["f1"],
            "train_time": train_time,
        }
        print(f"    Jacobian AUROC={r['Jacobian']['auroc']:.3f}  "
              f"Weight AUROC={r['Weight-Proj']['auroc']:.3f}  "
              f"Time={train_time:.1f}s")

    return results


def ablation_state_dim(device="cpu"):
    """Ablation 3: Effect of d_state."""
    print("\n" + "=" * 60)
    print("Ablation 3: State Dimension (d_state)")
    print("=" * 60)

    X, _, G_true = generate_var1(K=5, T=2000, graph_type="chain", seed=42)

    results = {}
    for d_state in [8, 16, 32]:
        print(f"\n  d_state={d_state}...")
        r, train_time = run_single_experiment(
            X, G_true, d_model=64, d_state=d_state, n_layers=4, device=device
        )
        results[f"d_state={d_state}"] = {
            "auroc_jac": r["Jacobian"]["auroc"],
            "auroc_weight": r["Weight-Proj"]["auroc"],
            "f1_jac": r["Jacobian"]["f1"],
            "train_time": train_time,
        }
        print(f"    Jacobian AUROC={r['Jacobian']['auroc']:.3f}  "
              f"Weight AUROC={r['Weight-Proj']['auroc']:.3f}  "
              f"Time={train_time:.1f}s")

    return results


def ablation_n_layers(device="cpu"):
    """Ablation 4: Effect of number of layers."""
    print("\n" + "=" * 60)
    print("Ablation 4: Number of Layers (n_layers)")
    print("=" * 60)

    X, _, G_true = generate_var1(K=5, T=2000, graph_type="chain", seed=42)

    results = {}
    for n_layers in [2, 4, 6]:
        print(f"\n  n_layers={n_layers}...")
        r, train_time = run_single_experiment(
            X, G_true, d_model=64, d_state=16, n_layers=n_layers, device=device
        )
        results[f"n_layers={n_layers}"] = {
            "auroc_jac": r["Jacobian"]["auroc"],
            "auroc_weight": r["Weight-Proj"]["auroc"],
            "f1_jac": r["Jacobian"]["f1"],
            "train_time": train_time,
        }
        print(f"    Jacobian AUROC={r['Jacobian']['auroc']:.3f}  "
              f"Weight AUROC={r['Weight-Proj']['auroc']:.3f}  "
              f"Time={train_time:.1f}s")

    return results


def ablation_data_length(device="cpu"):
    """Ablation 5: Effect of training data length T."""
    print("\n" + "=" * 60)
    print("Ablation 5: Training Data Length (T)")
    print("=" * 60)

    results = {}
    for T in [500, 1000, 2000, 5000]:
        print(f"\n  T={T}...")
        X, _, G_true = generate_var1(K=5, T=T, graph_type="chain", seed=42)
        r, train_time = run_single_experiment(
            X, G_true, d_model=64, d_state=16, n_layers=4,
            n_epochs=150, device=device,
        )
        results[f"T={T}"] = {
            "auroc_jac": r["Jacobian"]["auroc"],
            "auroc_weight": r["Weight-Proj"]["auroc"],
            "f1_jac": r["Jacobian"]["f1"],
            "train_time": train_time,
        }
        print(f"    Jacobian AUROC={r['Jacobian']['auroc']:.3f}  "
              f"Weight AUROC={r['Weight-Proj']['auroc']:.3f}  "
              f"Time={train_time:.1f}s")

    return results


def ablation_bootstrap_confidence(device="cpu", n_seeds=10):
    """Ablation 6: Bootstrap confidence intervals for main results."""
    print("\n" + "=" * 60)
    print(f"Ablation 6: Bootstrap Confidence Intervals ({n_seeds} seeds)")
    print("=" * 60)

    datasets = {
        "VAR1-chain-K3": lambda s: generate_var1(K=3, T=2000, graph_type="chain", seed=s),
        "VAR1-chain-K5": lambda s: generate_var1(K=5, T=2000, graph_type="chain", seed=s),
        "Lorenz-K3": lambda s: generate_lorenz(T=5000, seed=s, downsample=5),
    }

    results = {}
    for ds_name, gen_fn in datasets.items():
        print(f"\n  {ds_name}...")
        aurocs = []
        for seed in range(n_seeds):
            X, _, G_true = gen_fn(seed + 100)
            r, _ = run_single_experiment(
                X, G_true, d_model=64, d_state=16, n_layers=4,
                n_epochs=100, device=device, seed=seed,
            )
            aurocs.append(r["Jacobian"]["auroc"])

        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        ci_low = np.percentile(aurocs, 2.5)
        ci_high = np.percentile(aurocs, 97.5)

        results[ds_name] = {
            "mean": mean_auroc,
            "std": std_auroc,
            "ci_95": [ci_low, ci_high],
            "all_aurocs": aurocs,
        }
        print(f"    AUROC = {mean_auroc:.3f} +/- {std_auroc:.3f}  "
              f"(95% CI: [{ci_low:.3f}, {ci_high:.3f}])")

    return results


def plot_ablation_results(all_results, save_dir):
    """Generate publication-quality ablation figures."""
    os.makedirs(save_dir, exist_ok=True)

    # --- Figure: Score variant comparison ---
    if "score_variant" in all_results:
        sv = all_results["score_variant"]
        datasets = list(sv.keys())
        methods = ["Jacobian", "Weight-Proj", "BC-Proj"]
        colors = ["#2196F3", "#4CAF50", "#FF9800"]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(datasets))
        width = 0.25

        for i, method in enumerate(methods):
            aurocs = [sv[ds][method]["auroc"] for ds in datasets]
            ax.bar(x + i * width, aurocs, width, label=method, color=colors[i])

        ax.set_ylabel("AUROC")
        ax.set_title("Score Extraction Variant Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "ablation_score_variant.png"), dpi=300)
        plt.close(fig)

    # --- Figure: Hyperparameter sensitivity ---
    for param_name, key in [("d_model", "model_size"), ("d_state", "state_dim"),
                             ("n_layers", "n_layers"), ("T", "data_length")]:
        if key not in all_results:
            continue

        data = all_results[key]
        labels = list(data.keys())
        jac_aurocs = [data[l]["auroc_jac"] for l in labels]
        weight_aurocs = [data[l]["auroc_weight"] for l in labels]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, jac_aurocs, width, label="Jacobian", color="#2196F3")
        ax.bar(x + width / 2, weight_aurocs, width, label="Weight-Proj", color="#4CAF50")

        ax.set_ylabel("AUROC")
        ax.set_title(f"Effect of {param_name} on Causal Recovery (VAR1-chain-K5)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"ablation_{key}.png"), dpi=300)
        plt.close(fig)

    # --- Figure: Bootstrap confidence intervals ---
    if "bootstrap" in all_results:
        boot = all_results["bootstrap"]
        datasets = list(boot.keys())
        means = [boot[ds]["mean"] for ds in datasets]
        stds = [boot[ds]["std"] for ds in datasets]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(datasets))
        ax.bar(x, means, yerr=stds, capsize=5, color="#2196F3", alpha=0.8)
        ax.set_ylabel("AUROC")
        ax.set_title("SSM-Causal AUROC with 95% Confidence Intervals")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "ablation_bootstrap.png"), dpi=300)
        plt.close(fig)


def run_all_ablations(device="cpu", save_dir="results/ablation"):
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    # Run all ablations
    all_results["score_variant"] = ablation_score_variant(device)
    all_results["model_size"] = ablation_model_size(device)
    all_results["state_dim"] = ablation_state_dim(device)
    all_results["n_layers"] = ablation_n_layers(device)
    all_results["data_length"] = ablation_data_length(device)
    all_results["bootstrap"] = ablation_bootstrap_confidence(device, n_seeds=10)

    # Generate plots
    plot_ablation_results(all_results, save_dir)

    # Save results (convert numpy to lists for JSON serialization)
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(os.path.join(save_dir, "ablation_results.json"), "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All ablation results saved to {save_dir}/")
    print(f"{'=' * 60}")

    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_all_ablations(device=device)
