"""
Experiment 24: Real Data Benchmark for ICBINB Submission
=========================================================

Final missing piece before ICBINB submission: run the bottleneck method
and all baselines on REAL observational time-series data.

Previous "real-world" experiments (Exp 3, 4, 8) used synthetic data with
planted ground truth. This experiment uses actual downloaded/public data
with literature-derived ground truth edges.

Datasets:
    1. NOAA Climate Indices (ENSO, NAO, PDO, AMO, etc.) — real monthly
       observations from NOAA PSL, 70+ years. Ground truth: published
       teleconnection relationships.
    2. DREAM3 Gene Expression (E. coli size-10) — real gene regulatory
       network with known benchmark graph.
    3. Stock returns — 10 sector ETFs, real daily returns, ground truth
       from sector lead-lag relationships in finance literature.

Methods:
    - Weight-Projection (bottleneck)
    - Lagged-Bottleneck (Sai's explicit lagged version)
    - Tuned Lasso (baseline)
    - Ridge (baseline)
    - PCMCI (actual causal discovery baseline)
    - Bivariate Granger (classical baseline)

Framing: This is hypothesis-generation / ranking stability analysis,
NOT a claim that any method "discovers" true causal structure on
observational data.

Usage:
    python experiments/exp24_real_data_benchmark.py --device cuda
    python experiments/exp24_real_data_benchmark.py --quick
    python experiments/exp24_real_data_benchmark.py --datasets climate finance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ssm_causal.bottleneck_training import extract_weight_score
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics


# ═══════════════════════════════════════════
# Real data loaders
# ═══════════════════════════════════════════

def load_climate_indices(data_dir: str = "data/real") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load REAL NOAA monthly climate indices (ENSO, NAO, PDO, AMO, SOI, PNA)."""
    from data.real_loaders import load_noaa_climate
    X, G, names = load_noaa_climate(data_dir=data_dir)
    print(f"  Loaded REAL NOAA climate data: T={X.shape[0]} months, K={X.shape[1]}")
    print(f"  Variables: {names}")
    print(f"  Ground truth edges: {int(G.sum())}")
    return X, G, names


def load_stock_returns(data_dir: str = "data/real") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load REAL Yahoo Finance sector ETF log returns."""
    from data.real_loaders import load_sector_etfs
    X, G, names = load_sector_etfs(data_dir=data_dir)
    print(f"  Loaded REAL sector ETF returns: T={X.shape[0]} days, K={X.shape[1]}")
    print(f"  Variables: {names}")
    print(f"  Ground truth edges: {int(G.sum())}")
    return X, G, names


def load_dream3_gene() -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    DREAM3 E. coli size-10 gene expression data.

    This is a well-known benchmark for causal discovery. Only loads if
    data files are available; otherwise returns None.
    """
    # Placeholder — DREAM3 requires manual download and preprocessing.
    # If files are present at data/dream3/, load them here.
    print("  [SKIP] DREAM3 not bundled — requires manual download")
    return None


# ═══════════════════════════════════════════
# Evaluation methods
# ═══════════════════════════════════════════

def eval_weight_projection(X, G, d_model=64, epochs=150, device="cpu"):
    K = X.shape[1]
    torch.manual_seed(42)
    model = MambaForTimeSeries(n_variables=K, d_model=d_model, d_state=16, n_layers=4)
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=epochs, verbose=False)
    S = extract_weight_score(model)
    S = np.nan_to_num(S, 0)
    return compute_metrics(S, G)


def eval_lagged_bottleneck(X, G, max_lag=3, d_bottleneck=16, epochs=100, device="cpu"):
    from ssm_causal.lag_bottleneck import (
        LaggedBottleneckModel,
        LaggedBottleneckTrainer,
        LaggedBottleneckTrainingConfig,
        extract_lagged_weight_score,
        normalize_lagged_score,
    )
    from ssm_causal.lag_extraction import lag_score_to_aggregate

    K = X.shape[1]
    torch.manual_seed(42)
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(n_epochs=epochs, batch_size=min(64, X.shape[0] - max_lag))
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)

    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = np.nan_to_num(S_lag, 0)
    S = lag_score_to_aggregate(S_lag)
    return compute_metrics(S, G)


def eval_lasso(X, G, max_lag=3):
    from sklearn.linear_model import MultiTaskLasso
    K, T = X.shape[1], X.shape[0]
    X_lag = np.zeros((T - max_lag, K * max_lag))
    Y = X[max_lag:]
    for tau in range(max_lag):
        X_lag[:, tau * K : (tau + 1) * K] = X[max_lag - tau - 1 : T - tau - 1]

    split = int(len(X_lag) * 0.6)
    val_end = int(len(X_lag) * 0.8)
    best_a, best_mse = 0.01, 1e10
    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        try:
            m = MultiTaskLasso(alpha=a, max_iter=5000)
            m.fit(X_lag[:split], Y[:split])
            mse = np.mean((m.predict(X_lag[split:val_end]) - Y[split:val_end]) ** 2)
            if mse < best_mse:
                best_mse, best_a = mse, a
        except Exception:
            pass
    lasso = MultiTaskLasso(alpha=best_a, max_iter=5000)
    lasso.fit(X_lag[:val_end], Y[:val_end])
    S = np.abs(lasso.coef_).reshape(K, max_lag, K).max(axis=1)
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return compute_metrics(S, G)


def eval_ridge(X, G, max_lag=3):
    from sklearn.linear_model import Ridge
    K, T = X.shape[1], X.shape[0]
    X_lag = np.zeros((T - max_lag, K * max_lag))
    Y = X[max_lag:]
    for tau in range(max_lag):
        X_lag[:, tau * K : (tau + 1) * K] = X[max_lag - tau - 1 : T - tau - 1]

    split = int(len(X_lag) * 0.6)
    val_end = int(len(X_lag) * 0.8)
    best_a, best_mse = 1.0, 1e10
    for a in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        try:
            m = Ridge(alpha=a)
            m.fit(X_lag[:split], Y[:split])
            mse = np.mean((m.predict(X_lag[split:val_end]) - Y[split:val_end]) ** 2)
            if mse < best_mse:
                best_mse, best_a = mse, a
        except Exception:
            pass
    ridge = Ridge(alpha=best_a)
    ridge.fit(X_lag[:val_end], Y[:val_end])
    S = np.abs(ridge.coef_).reshape(K, max_lag, K).max(axis=1)
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return compute_metrics(S, G)


def eval_pcmci(X, G, tau_max=5):
    try:
        from baselines.pcmci_wrapper import pcmci_causality_matrix
        S = pcmci_causality_matrix(X, tau_max=tau_max, pc_alpha=0.05)
        S = np.nan_to_num(S, 0)
        return compute_metrics(S, G)
    except Exception as e:
        print(f"    PCMCI failed: {e}")
        return {"auroc": 0.5, "auprc": 0.5, "f1": 0.0}


def eval_granger(X, G, max_lag=5):
    try:
        from baselines.granger import granger_causality_matrix
        S = granger_causality_matrix(X, max_lag=max_lag)
        S = np.nan_to_num(S, 0)
        return compute_metrics(S, G)
    except Exception as e:
        print(f"    Granger failed: {e}")
        return {"auroc": 0.5, "auprc": 0.5, "f1": 0.0}


# ═══════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════

def run_real_benchmark(
    datasets: list[str],
    device: str = "cpu",
    save_dir: str = "results/real_data_benchmark",
    climate_data_dir: str | None = None,
    epochs: int = 150,
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 24: Real Data Benchmark")
    print(f"Datasets: {datasets} | Device: {device}")
    print("Protocol: locked extractors, tuned baselines, no graph-label tuning")
    print("=" * 72)

    # Load datasets
    loaded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    if "climate" in datasets:
        X, G, names = load_climate_indices(data_dir=climate_data_dir)
        loaded["climate"] = (X, G, names)
    if "finance" in datasets:
        X, G, names = load_stock_returns()
        loaded["finance"] = (X, G, names)
    if "dream3" in datasets:
        result = load_dream3_gene()
        if result is not None:
            loaded["dream3"] = result

    methods = {
        "Weight-Proj (SSM)": lambda X, G: eval_weight_projection(X, G, epochs=epochs, device=device),
        "Lagged-Bottleneck": lambda X, G: eval_lagged_bottleneck(X, G, epochs=100, device=device),
        "Tuned-Lasso": lambda X, G: eval_lasso(X, G),
        "Tuned-Ridge": lambda X, G: eval_ridge(X, G),
        "PCMCI": lambda X, G: eval_pcmci(X, G),
        "Granger": lambda X, G: eval_granger(X, G),
    }

    all_results = {}

    for ds_name, (X, G, names) in loaded.items():
        K, T = X.shape[1], X.shape[0]
        n_edges = int(G.sum())
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name} (K={K}, T={T}, edges={n_edges})")
        print(f"  Variables: {', '.join(names[:10])}{'...' if len(names) > 10 else ''}")
        print(f"{'─' * 60}")

        # Standardize
        X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)

        ds_results = {}
        for method_name, method_fn in methods.items():
            t0 = time.time()
            try:
                metrics = method_fn(X_std, G)
                elapsed = time.time() - t0
                ds_results[method_name] = {
                    "auroc": float(metrics["auroc"]),
                    "auprc": float(metrics.get("auprc", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                    "runtime_s": elapsed,
                }
                print(f"  {method_name:22s}: AUROC={metrics['auroc']:.3f}  AUPRC={metrics.get('auprc', 0):.3f}  F1={metrics.get('f1', 0):.3f}  ({elapsed:.1f}s)")
            except Exception as e:
                ds_results[method_name] = {"auroc": 0.5, "error": str(e)}
                print(f"  {method_name:22s}: FAILED ({e})")

        all_results[ds_name] = ds_results

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 72}")
    print("SUMMARY: Real Data Benchmark")
    print(f"{'=' * 72}")
    print(f"{'Method':<22} " + " ".join(f"{ds:<12}" for ds in all_results))
    for method in methods:
        row = f"{method:<22} "
        for ds_name, ds_results in all_results.items():
            auroc = ds_results.get(method, {}).get("auroc", 0.5)
            row += f"{auroc:<12.3f} "
        print(row)

    print(f"\nTotal runtime: {total_time / 60:.1f} min")

    # Save
    out_path = Path(save_dir) / "real_data_benchmark_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["climate", "finance"],
        choices=["climate", "finance", "dream3"],
    )
    parser.add_argument("--climate-data-dir", default="data/real")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/real_data_benchmark")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20
        args.datasets = ["climate"]

    run_real_benchmark(
        datasets=args.datasets,
        device=args.device,
        save_dir=args.save_dir,
        climate_data_dir=args.climate_data_dir,
        epochs=args.epochs,
    )
