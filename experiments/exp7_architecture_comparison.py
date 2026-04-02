"""
Experiment 7: Architecture Comparison (SSM vs MLP vs LSTM)
===========================================================

KEY EXPERIMENT for NeurIPS reviewers.

Tests: Does the SSM architecture specifically help, or would any
model with input/output projections work?

Compares weight-projection causal score from:
    1. Mamba SSM (our method)
    2. Simple MLP (same d_model, same projections)
    3. LSTM (same d_model, same projections)
    4. Linear model (no hidden layers)

If SSM outperforms MLP/LSTM on the same extraction method,
it proves the SSM architecture encodes causal structure
better — not just any bottleneck model.

Usage:
    python experiments/exp7_architecture_comparison.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.synthetic import generate_var1, generate_lorenz, generate_regime_switching
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, print_metrics_table


class MLPForTimeSeries(nn.Module):
    """Simple MLP with same input/output projections as Mamba."""

    def __init__(self, n_variables, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_variables, d_model)
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)
        self.output_proj = nn.Linear(d_model, n_variables)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.layers(h)
        return self.output_proj(h)


class LSTMForTimeSeries(nn.Module):
    """LSTM with same input/output projections as Mamba."""

    def __init__(self, n_variables, d_model=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_variables, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(d_model, n_variables)

    def forward(self, x):
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.output_proj(h)


class LinearForTimeSeries(nn.Module):
    """Linear model: direct input-to-output (no hidden layers)."""

    def __init__(self, n_variables, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(n_variables, d_model)
        self.output_proj = nn.Linear(d_model, n_variables)

    def forward(self, x):
        return self.output_proj(self.input_proj(x))


def extract_weight_score(model):
    """Extract causal score from any model with input_proj and output_proj."""
    W_in = model.input_proj.weight.detach().cpu()
    W_out = model.output_proj.weight.detach().cpu()
    S = (W_out @ W_in).abs().numpy()
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def train_model(model, X, n_epochs=150, lr=1e-3, device="cpu"):
    """Generic training loop for any model."""
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    X = X.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    T = X.shape[0]
    seq_len = 200
    batch_size = 32

    model.train()
    for epoch in range(n_epochs):
        starts = torch.randint(0, T - seq_len - 1, (batch_size,))
        batch_x = torch.stack([X[s:s + seq_len] for s in starts])
        batch_y = torch.stack([X[s + 1:s + seq_len + 1] for s in starts])

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model


def run_architecture_comparison(device="cpu", save_dir="results/exp7_architecture"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment 7: Architecture Comparison")
    print("Why SSM specifically encodes causal structure")
    print("=" * 60)

    datasets = {}
    X, _, G = generate_var1(K=5, T=2000, graph_type="chain", seed=42)
    datasets["VAR1-chain-K5"] = (X, G)
    X, _, G = generate_regime_switching(K=3, T=4000, seed=42)
    datasets["Regime-switch-K3"] = (X, G)
    X, _, G = generate_lorenz(T=5000, seed=42, downsample=5)
    datasets["Lorenz-K3"] = (X, G)

    architectures = {
        "Mamba (SSM)": lambda K: MambaForTimeSeries(
            n_variables=K, d_model=64, d_state=16, n_layers=4),
        "MLP": lambda K: MLPForTimeSeries(
            n_variables=K, d_model=64, n_layers=4),
        "LSTM": lambda K: LSTMForTimeSeries(
            n_variables=K, d_model=64, n_layers=2),
        "Linear": lambda K: LinearForTimeSeries(
            n_variables=K, d_model=64),
    }

    all_results = {}

    for ds_name, (X, G_true) in datasets.items():
        K = X.shape[1]
        print(f"\n  Dataset: {ds_name} (K={K})")
        ds_results = {}

        for arch_name, model_fn in architectures.items():
            print(f"    {arch_name}...", end=" ", flush=True)
            model = model_fn(K)

            t0 = time.time()
            if arch_name == "Mamba (SSM)":
                trainer = SSMTrainer(model, lr=1e-3, device=device)
                trainer.train(X, n_epochs=150, verbose=False)
            else:
                model = train_model(model, X, n_epochs=150, device=device)

            S = extract_weight_score(model)
            elapsed = time.time() - t0

            m = compute_metrics(S, G_true)
            m["runtime_s"] = elapsed
            ds_results[arch_name] = m
            print(f"AUROC={m['auroc']:.3f} ({elapsed:.1f}s)")

        all_results[ds_name] = ds_results
        print_metrics_table(ds_results, dataset_name=ds_name)

    # Save results
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

    # Figure: Architecture comparison
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    colors = ["#0072B2", "#E69F00", "#CC79A7", "#999999"]

    for ax, (ds_name, ds_results) in zip(axes, all_results.items()):
        methods = list(ds_results.keys())
        aurocs = [ds_results[m]["auroc"] for m in methods]
        bars = ax.bar(range(len(methods)), aurocs, color=colors[:len(methods)])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_title(ds_name)
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Weight-Projection Causal Score: SSM vs Other Architectures",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "architecture_comparison.png"), dpi=300)
    plt.close(fig)

    print(f"\nFigure saved to {save_dir}/architecture_comparison.png")
    print(f"Results saved to {save_dir}/results.json")
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_architecture_comparison(device=device)
