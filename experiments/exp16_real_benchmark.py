"""
Experiment 16: Semi-real benchmark gate on CauseMe-style synthetic processes.

This experiment is an external-validity stress test beyond our local hand-built
VAR generators, but it is still not a true real-data benchmark. It uses
Tigramite structural causal processes when available and falls back to a local
synthetic VAR otherwise.

Decision gate per the research memo:
- If tuned Lasso still beats bottleneck -> write characterization paper
- If bottleneck separates -> investigate mechanism
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
from ssm_causal.causeme_benchmarks import default_causeme_scenarios
from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    extract_lagged_weight_score,
    make_lagged_dataset,
    normalize_lagged_score,
)
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics

# ═══════════════════════════════════════════
# Evaluation methods
# ═══════════════════════════════════════════

def eval_weight_projection(X, G_agg, d_model=64, n_epochs=150, device="cpu"):
    """Our method: train predictor, extract |W_out @ W_in|."""
    K = X.shape[1]
    model = MambaForTimeSeries(n_variables=K, d_model=d_model, d_state=16, n_layers=4)
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=n_epochs, verbose=False)
    S = extract_weight_score(model)
    S = np.nan_to_num(S, 0)
    return compute_metrics(S, G_agg)


def eval_lagged_bottleneck(X, G_agg, G_lag, max_lag=3, d_bottleneck=16, n_epochs=100, device="cpu"):
    """Sai's lagged bottleneck."""
    K = X.shape[1]
    X_lagged, Y_lagged = make_lagged_dataset(X, max_lag=max_lag, device=device)

    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(n_epochs=n_epochs, batch_size=min(64, len(X_lagged)))
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)

    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = np.nan_to_num(S_lag, 0)

    # Aggregate for comparison
    from ssm_causal.lag_extraction import lag_score_to_aggregate
    S_agg = lag_score_to_aggregate(S_lag)

    return compute_metrics(S_agg, G_agg)


def eval_tuned_lasso(X, G_agg, max_lag=3):
    """Validation-tuned Lasso baseline."""
    K = X.shape[1]
    T = X.shape[0]
    split = int(T * 0.6)
    val_end = int(T * 0.8)

    X_lagged_np, Y_np = _make_lagged_numpy(X, max_lag)
    X_train = X_lagged_np[:split - max_lag]
    Y_train = Y_np[:split - max_lag]
    X_val = X_lagged_np[split - max_lag:val_end - max_lag]
    Y_val = Y_np[split - max_lag:val_end - max_lag]

    from sklearn.linear_model import MultiTaskLasso

    best_alpha = 0.01
    best_mse = float("inf")
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        try:
            lasso = MultiTaskLasso(alpha=alpha, max_iter=5000)
            lasso.fit(X_train, Y_train)
            mse = np.mean((lasso.predict(X_val) - Y_val) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except Exception:
            continue

    lasso = MultiTaskLasso(alpha=best_alpha, max_iter=5000)
    lasso.fit(np.vstack([X_train, X_val]), np.vstack([Y_train, Y_val]))

    coefs = lasso.coef_  # [K, K*max_lag]
    S = np.abs(coefs).reshape(K, max_lag, K).max(axis=1)  # Aggregate over lags
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()

    return compute_metrics(S, G_agg)


def eval_granger(X, G_agg, max_lag=5):
    """Granger baseline."""
    from baselines.granger import granger_causality_matrix
    S = granger_causality_matrix(X, max_lag=max_lag)
    return compute_metrics(S, G_agg)


def _make_lagged_numpy(X, max_lag):
    T, K = X.shape
    X_lagged = np.zeros((T - max_lag, K * max_lag))
    Y = X[max_lag:]
    for tau in range(max_lag):
        X_lagged[:, tau * K:(tau + 1) * K] = X[max_lag - tau - 1:T - tau - 1]
    return X_lagged, Y


# ═══════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════

def run_benchmark(
    n_seeds=3,
    device="cpu",
    save_dir="results/real_benchmark",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 16: Real/Semi-Real Benchmark Gate")
    print(f"Seeds: {n_seeds} | Device: {device}")
    print("Protocol: locked extractor, no tuning on graph labels")
    print("=" * 72)

    scenarios = default_causeme_scenarios(include_nonlinear=True)

    methods = {
        "Weight-Proj (SSM)": lambda X, Ga, Gl: eval_weight_projection(X, Ga, device=device),
        "Lagged-Bottleneck": lambda X, Ga, Gl: eval_lagged_bottleneck(X, Ga, Gl, device=device),
        "Tuned-Lasso": lambda X, Ga, Gl: eval_tuned_lasso(X, Ga),
        "Granger": lambda X, Ga, Gl: eval_granger(X, Ga),
    }

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'─' * 60}")
        print(f"Scenario: {scenario.name}")
        print(f"{'─' * 60}")

        scenario_results = {}

        for method_name, method_fn in methods.items():
            aurocs = []
            for seed in range(n_seeds):
                try:
                    X, G_agg, G_lag, _ = scenario.generate(100 + seed)
                    X = np.nan_to_num(X, 0.0)

                    m = method_fn(X, G_agg, G_lag)
                    aurocs.append(m["auroc"])
                except Exception as e:
                    print(f"    {method_name} seed {seed}: FAILED ({e})")
                    aurocs.append(0.5)

            mean_a = np.mean(aurocs)
            std_a = np.std(aurocs)
            scenario_results[method_name] = {
                "auroc_mean": float(mean_a),
                "auroc_std": float(std_a),
                "all": [float(a) for a in aurocs],
            }
            print(f"  {method_name:25s}: AUROC={mean_a:.3f}±{std_a:.3f}")

        all_results[scenario.name] = scenario_results

    total_time = time.time() - total_start

    # ── Decision Gate ──
    print(f"\n{'=' * 72}")
    print("DECISION GATE: Bottleneck vs Tuned Lasso on Semi-Real Data")
    print(f"{'=' * 72}")

    bottleneck_wins = 0
    for scenario, results in all_results.items():
        wp = results.get("Weight-Proj (SSM)", {}).get("auroc_mean", 0)
        lb = results.get("Lagged-Bottleneck", {}).get("auroc_mean", 0)
        lasso = results.get("Tuned-Lasso", {}).get("auroc_mean", 0)
        best_bottleneck = max(wp, lb)
        delta = best_bottleneck - lasso

        if delta > 0.02:
            bottleneck_wins += 1
            print(
                f"  {scenario:30s}: Bottleneck={best_bottleneck:.3f} "
                f"Lasso={lasso:.3f} Δ={delta:+.3f} ← BOTTLENECK WINS"
            )
        else:
            print(
                f"  {scenario:30s}: Bottleneck={best_bottleneck:.3f} "
                f"Lasso={lasso:.3f} Δ={delta:+.3f}"
            )

    print(f"\n  Bottleneck wins: {bottleneck_wins}/{len(scenarios)}")

    if bottleneck_wins >= 2:
        print("  ✓ DECISION: Bottleneck separates on semi-real data. Investigate mechanism.")
    elif bottleneck_wins >= 1:
        print("  ~ DECISION: Marginal. Bottleneck has niche advantage.")
    else:
        print("  ✗ DECISION: Lasso dominates. Write characterization paper.")

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

    out_path = Path(save_dir) / "real_benchmark_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(all_results), f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/real_benchmark")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_benchmark(n_seeds=1, device=args.device, save_dir=args.save_dir)
    else:
        run_benchmark(n_seeds=args.seeds, device=args.device, save_dir=args.save_dir)
