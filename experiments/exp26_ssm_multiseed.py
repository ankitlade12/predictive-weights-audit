"""
Experiment 26: Multi-seed SSM weight-projection on the real-data benchmark.

Hardens the SSM row of Table 2 in the paper. Exp 24 reported a single seed
per dataset. This script reruns the SSM weight-projection extractor across
5 seeds on each of NOAA climate, sector ETFs, and Lorenz-96 and reports
mean +/- std.

Usage:
    ./.venv/bin/python -u experiments/exp26_ssm_multiseed.py --seeds 5 --epochs 40
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


def eval_weight_projection_seeded(X, G, seed, d_model=64, epochs=40, device="cpu"):
    K = X.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=16, n_layers=4,
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=epochs, verbose=False)
    S = extract_weight_score(model)
    S = np.nan_to_num(S, 0)
    return compute_metrics(S, G)


def load_dataset(name):
    if name == "climate":
        from data.real_loaders import load_noaa_climate
        return load_noaa_climate()
    if name == "finance":
        from data.real_loaders import load_sector_etfs
        return load_sector_etfs()
    if name == "lorenz96":
        from data.real_loaders import load_lorenz96
        return load_lorenz96(K=10, T=1500, F=10.0, dt=0.01, subsample=10, seed=0)
    raise ValueError(f"unknown dataset: {name}")


def standardize_X(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["climate", "finance", "lorenz96"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/ssm_multiseed")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    started = time.time()

    print("=" * 72)
    print("EXP 26: Multi-seed SSM weight-projection")
    print(f"Datasets: {args.datasets} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 72)

    all_results = {}

    for dname in args.datasets:
        X, G, names = load_dataset(dname)
        X_std = standardize_X(X.astype(np.float32))
        K, T = X_std.shape[1], X_std.shape[0]
        n_edges = int(G.sum())
        print(f"\n--- {dname} (K={K}, T={T}, edges={n_edges}) ---")

        per_seed = []
        for s in range(args.seeds):
            t0 = time.time()
            metrics = eval_weight_projection_seeded(
                X_std, G, seed=s, epochs=args.epochs, device=args.device,
            )
            elapsed = time.time() - t0
            per_seed.append({
                "seed": s,
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics.get("auprc", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "runtime_s": elapsed,
            })
            print(
                f"  seed={s} AUROC={metrics['auroc']:.3f} "
                f"AUPRC={metrics.get('auprc', 0):.3f} "
                f"F1={metrics.get('f1', 0):.3f} "
                f"({elapsed:.1f}s)"
            )

        aurocs = np.array([r["auroc"] for r in per_seed])
        auprcs = np.array([r["auprc"] for r in per_seed])
        agg = {
            "auroc_mean": float(aurocs.mean()),
            "auroc_std": float(aurocs.std(ddof=1)) if len(aurocs) > 1 else 0.0,
            "auroc_min": float(aurocs.min()),
            "auroc_max": float(aurocs.max()),
            "auprc_mean": float(auprcs.mean()),
            "auprc_std": float(auprcs.std(ddof=1)) if len(auprcs) > 1 else 0.0,
            "n_seeds": args.seeds,
            "per_seed": per_seed,
        }
        all_results[dname] = agg
        print(
            f"  -> AUROC {aurocs.mean():.3f} +/- {aurocs.std(ddof=1):.3f}"
            f" (range {aurocs.min():.3f}-{aurocs.max():.3f})"
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for dname, agg in all_results.items():
        print(
            f"  {dname:10s} AUROC = {agg['auroc_mean']:.3f} +/- {agg['auroc_std']:.3f}"
            f"  (n={agg['n_seeds']} seeds, range {agg['auroc_min']:.3f}-{agg['auroc_max']:.3f})"
        )

    out_path = Path(args.save_dir) / "ssm_multiseed_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRuntime: {(time.time() - started) / 60:.1f} min")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
