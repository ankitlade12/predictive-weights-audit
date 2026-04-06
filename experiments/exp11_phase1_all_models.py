"""
Phase 1 Feasibility: All Four Causality-Preserving Models
==========================================================

Tests the complete method family from the research reset plan:

    Model A: L = L_pred                           (baseline)
    Model B: L = L_pred + λ_sparse * L_sparse     (sparse bottleneck)
    Model C: L = L_pred + λ_stable * L_stable     (stable bottleneck)
    Model D: L = L_pred + all three losses         (full combined)

Datasets: VAR1-chain-K5, VAR1-random-K10, VAR2-K5, Lorenz-K3

Decision gate: Continue only if at least one new method improves
recovery on at least 2 hard datasets without ruining prediction.

Usage:
    python experiments/exp11_phase1_all_models.py
    python experiments/exp11_phase1_all_models.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.synthetic import generate_var1, generate_var2, generate_lorenz
from ssm_causal.bottleneck_training import (
    BottleneckTrainingConfig,
    CausalBottleneckTrainer,
    evaluate_bottleneck_model,
)
from ssm_causal.mamba_model import MambaForTimeSeries


DATASETS = {
    "VAR1-chain-K5": lambda s: generate_var1(K=5, T=1200, graph_type="chain", seed=s),
    "VAR1-random-K10": lambda s: generate_var1(K=10, T=1600, graph_type="random", seed=s),
    "VAR2-K5": lambda s: generate_var2(K=5, T=1600, seed=s),
    "Lorenz-K3": lambda s: generate_lorenz(T=5000, seed=s, downsample=5),
}

# The four model configs from the design doc
MODEL_CONFIGS = {
    "A-Baseline": {"lambda_sparse": 0.0, "lambda_stable": 0.0, "lambda_consist": 0.0},
    "B-Sparse": {"lambda_sparse": 3e-5, "lambda_stable": 0.0, "lambda_consist": 0.0},
    "C-Stable": {"lambda_sparse": 0.0, "lambda_stable": 1e-2, "lambda_consist": 0.0},
    "D-Combined": {"lambda_sparse": 3e-5, "lambda_stable": 1e-2, "lambda_consist": 1e-2},
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_phase1(
    n_seeds: int = 3,
    epochs: int = 80,
    d_model: int = 64,
    d_state: int = 16,
    n_layers: int = 4,
    device: str = "cpu",
    save_dir: str = "results/phase1_all_models",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("PHASE 1: Causality-Preserving Bottleneck Training")
    print("Models: A (baseline), B (sparse), C (stable), D (combined)")
    print(f"Seeds: {n_seeds} | Epochs: {epochs} | Device: {device}")
    print("=" * 72)

    all_results = {}

    for ds_name, gen_fn in DATASETS.items():
        X_sample, _, _ = gen_fn(0)
        K = X_sample.shape[1]
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name} (K={K})")
        print(f"{'─' * 60}")

        ds_results = {}

        for model_name, lambdas in MODEL_CONFIGS.items():
            seed_metrics = []

            for seed_idx in range(n_seeds):
                set_seed(seed_idx * 42 + 7)
                X, _, G_true = gen_fn(100 + seed_idx)

                # Train/eval split
                split = int(len(X) * 0.8)
                X_train, X_eval = X[:split], X[split - 1:]

                model = MambaForTimeSeries(
                    n_variables=K, d_model=d_model,
                    d_state=d_state, n_layers=n_layers, dropout=0.05,
                )

                config = BottleneckTrainingConfig(
                    n_epochs=epochs,
                    batch_size=16,
                    seq_len=100,
                    lambda_sparse=lambdas["lambda_sparse"],
                    lambda_stable=lambdas["lambda_stable"],
                    lambda_consist=lambdas["lambda_consist"],
                    consist_every_n=5,
                    print_every=epochs + 1,  # silent
                )

                trainer = CausalBottleneckTrainer(model, config=config, device=device)
                history = trainer.train(X_train, verbose=False)

                metrics = evaluate_bottleneck_model(model, X_eval, G_true, device=device)
                metrics["final_pred_loss"] = float(history["prediction_loss"][-1])
                seed_metrics.append(metrics)

            # Aggregate across seeds
            aurocs = [m["auroc"] for m in seed_metrics]
            pred_mses = [m["prediction_mse"] for m in seed_metrics]
            densities = [m["graph_density"] for m in seed_metrics]
            f1s = [m["f1"] for m in seed_metrics]

            summary = {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "pred_mse_mean": float(np.mean(pred_mses)),
                "f1_mean": float(np.mean(f1s)),
                "density_mean": float(np.mean(densities)),
                "runs": seed_metrics,
            }
            ds_results[model_name] = summary

            print(
                f"  {model_name:12s}: AUROC={summary['auroc_mean']:.3f}±{summary['auroc_std']:.3f}"
                f" | F1={summary['f1_mean']:.3f}"
                f" | PredMSE={summary['pred_mse_mean']:.5f}"
                f" | Density={summary['density_mean']:.3f}"
            )

        all_results[ds_name] = ds_results

    total_time = time.time() - total_start

    # ── Decision Gate ──
    print(f"\n{'=' * 72}")
    print("PHASE 1 DECISION GATE")
    print(f"{'=' * 72}")

    baseline_aurocs = {}
    improvements = {}

    for ds_name, ds_results in all_results.items():
        baseline = ds_results["A-Baseline"]["auroc_mean"]
        baseline_aurocs[ds_name] = baseline
        best_new = 0.0
        best_model = ""
        for model_name in ["B-Sparse", "C-Stable", "D-Combined"]:
            auroc = ds_results[model_name]["auroc_mean"]
            pred_ok = ds_results[model_name]["pred_mse_mean"] < ds_results["A-Baseline"]["pred_mse_mean"] * 2.0
            if auroc > best_new and pred_ok:
                best_new = auroc
                best_model = model_name
        delta = best_new - baseline
        improvements[ds_name] = {"delta": delta, "best_model": best_model, "baseline": baseline, "best": best_new}
        status = "IMPROVED" if delta > 0.02 else ("SAME" if delta > -0.02 else "WORSE")
        print(f"  {ds_name:20s}: Baseline={baseline:.3f} → Best={best_new:.3f} ({best_model}) Δ={delta:+.3f} [{status}]")

    improved_count = sum(1 for v in improvements.values() if v["delta"] > 0.02)
    print(f"\n  Datasets improved (>2% AUROC): {improved_count}/{len(DATASETS)}")

    if improved_count >= 2:
        print("  ✓ DECISION: CONTINUE to Phase 2 — method shows promise")
    elif improved_count >= 1:
        print("  ~ DECISION: MARGINAL — investigate which loss helps and why")
    else:
        print("  ✗ DECISION: STOP — new losses do not improve over baseline")

    print(f"\n  Total runtime: {total_time / 60:.1f} minutes")

    # Save
    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    out_path = Path(save_dir) / "phase1_all_models_results.json"
    with out_path.open("w") as f:
        json.dump(serialize({"results": all_results, "improvements": improvements}), f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/phase1_all_models")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_phase1(n_seeds=1, epochs=10, d_model=16, d_state=8, n_layers=1,
                   device=args.device, save_dir=args.save_dir)
    else:
        run_phase1(n_seeds=args.seeds, epochs=args.epochs,
                   device=args.device, save_dir=args.save_dir)
