"""
Experiment 13: Lag-Resolved Bottleneck Extraction
==================================================

Tests Sai's Priority 1 direction: can we extract lag-specific
dependency scores S[tau, i, j] from trained prediction models?

Datasets designed to have known lag structure:
    1. VAR(1) K=5: all edges at lag 1
    2. VAR(2) K=5: edges at lags 1 AND 2
    3. Mixed-lag K=5: different edges at different lags
    4. Delayed-sparse K=5: some edges at lag 3-4 (hard)
    5. Lorenz K=3: nonlinear, continuous lags

Two extraction methods:
    - Lag-Jacobian: exact but expensive
    - Lag-Impulse: cheap approximation

Decision gate: the method must correctly identify WHICH LAG
each edge operates at, not just whether the edge exists.

Usage:
    python experiments/exp13_lag_resolved.py
    python experiments/exp13_lag_resolved.py --quick
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

from data.synthetic import generate_var1, generate_var2, generate_lorenz
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.lag_extraction import (
    extract_lag_jacobian_score,
    extract_lag_impulse_score,
    lag_score_to_aggregate,
    evaluate_lag_recovery,
)
from ssm_causal.bottleneck_training import extract_weight_score
from ssm_causal.utils import compute_metrics


# ═══════════════════════════════════════════
# Dataset generators with lag ground truth
# ═══════════════════════════════════════════

def generate_var1_with_lag_gt(K=5, T=2000, seed=42):
    """VAR(1): all edges at lag 1."""
    X, A, G = generate_var1(K=K, T=T, graph_type="chain", seed=seed)
    max_lag = 5
    G_lag = np.zeros((max_lag, K, K))
    G_lag[0] = G  # All edges at lag 1 (index 0)
    return X, G_lag, "VAR1-chain-K5 (all edges at lag 1)"


def generate_var2_with_lag_gt(K=5, T=2000, seed=42):
    """VAR(2): edges at lags 1 AND 2."""
    X, A_dict, G = generate_var2(K=K, T=T, seed=seed)
    max_lag = 5
    G_lag = np.zeros((max_lag, K, K))
    A1 = A_dict["A1"]
    A2 = A_dict["A2"]
    G_lag[0] = (np.abs(A1) > 1e-10).astype(float)
    G_lag[1] = (np.abs(A2) > 1e-10).astype(float)
    np.fill_diagonal(G_lag[0], 0)
    np.fill_diagonal(G_lag[1], 0)
    return X, G_lag, "VAR2-K5 (edges at lags 1 and 2)"


def generate_mixed_lag(K=5, T=2000, seed=42):
    """Mixed lags: X0->X1 at lag 1, X1->X2 at lag 2, X2->X3 at lag 3."""
    rng = np.random.RandomState(seed)
    max_lag = 5

    # Build multi-lag coefficient matrices
    A = [np.diag([0.5] * K) for _ in range(max_lag)]
    # Lag 1: X0 -> X1
    A[0][1, 0] = 0.4
    # Lag 2: X1 -> X2
    A[1][2, 1] = 0.4
    # Lag 3: X2 -> X3
    A[2][3, 2] = 0.4

    # Ensure stability (check companion matrix spectral radius)
    for lag_idx in range(max_lag):
        eigvals = np.linalg.eigvals(A[lag_idx])
        sr = np.max(np.abs(eigvals))
        if sr >= 0.95:
            A[lag_idx] = A[lag_idx] * (0.94 / sr)

    X = np.zeros((T, K))
    for t in range(max_lag, T):
        for lag_idx in range(max_lag):
            X[t] += A[lag_idx] @ X[t - lag_idx - 1]
        X[t] += rng.randn(K) * 0.1

    G_lag = np.zeros((max_lag, K, K))
    for lag_idx in range(max_lag):
        G_lag[lag_idx] = (np.abs(A[lag_idx]) > 1e-10).astype(float)
        np.fill_diagonal(G_lag[lag_idx], 0)

    return X, G_lag, "Mixed-lag-K5 (edges at lags 1, 2, 3)"


def generate_delayed_sparse(K=5, T=2000, seed=42):
    """Delayed edges: X0->X1 at lag 3, X2->X3 at lag 4."""
    rng = np.random.RandomState(seed)
    max_lag = 5

    A = [np.diag([0.5] * K) for _ in range(max_lag)]
    # Lag 3: X0 -> X1
    A[2][1, 0] = 0.4
    # Lag 4: X2 -> X3
    A[3][3, 2] = 0.4

    for lag_idx in range(max_lag):
        eigvals = np.linalg.eigvals(A[lag_idx])
        sr = np.max(np.abs(eigvals))
        if sr >= 0.95:
            A[lag_idx] = A[lag_idx] * (0.94 / sr)

    X = np.zeros((T, K))
    for t in range(max_lag, T):
        for lag_idx in range(max_lag):
            X[t] += A[lag_idx] @ X[t - lag_idx - 1]
        X[t] += rng.randn(K) * 0.1

    G_lag = np.zeros((max_lag, K, K))
    for lag_idx in range(max_lag):
        G_lag[lag_idx] = (np.abs(A[lag_idx]) > 1e-10).astype(float)
        np.fill_diagonal(G_lag[lag_idx], 0)

    return X, G_lag, "Delayed-sparse-K5 (edges at lags 3, 4)"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════

def run_lag_experiment(
    n_seeds=3,
    epochs=150,
    d_model=64,
    d_state=16,
    n_layers=4,
    max_lag=5,
    device="cpu",
    save_dir="results/lag_resolved",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 13: Lag-Resolved Bottleneck Extraction")
    print(f"Seeds: {n_seeds} | Epochs: {epochs} | Max lag: {max_lag} | Device: {device}")
    print("=" * 72)

    datasets = {
        "VAR1-chain": generate_var1_with_lag_gt,
        "VAR2": generate_var2_with_lag_gt,
        "Mixed-lag": generate_mixed_lag,
        "Delayed-sparse": generate_delayed_sparse,
    }

    all_results = {}

    for ds_name, gen_fn in datasets.items():
        X_sample, G_lag_sample, desc = gen_fn(seed=0)
        K = X_sample.shape[1]
        n_edges_per_lag = [int(G_lag_sample[tau].sum()) for tau in range(max_lag)]

        print(f"\n{'─' * 60}")
        print(f"Dataset: {desc}")
        print(f"  Edges per lag: {n_edges_per_lag}")
        print(f"{'─' * 60}")

        ds_results = {}

        for seed in range(n_seeds):
            set_seed(seed * 42 + 7)
            X, G_lag, _ = gen_fn(K=5, T=2000, seed=100 + seed) if "VAR1" in ds_name or "VAR2" in ds_name else gen_fn(seed=100 + seed)

            # Train model
            model = MambaForTimeSeries(
                n_variables=K, d_model=d_model, d_state=d_state, n_layers=n_layers,
            )
            trainer = SSMTrainer(model, lr=1e-3, device=device)
            trainer.train(X, n_epochs=epochs, verbose=False)
            X_tensor = torch.FloatTensor(X)

            # 1. Aggregate weight score (baseline)
            S_weight = extract_weight_score(model)
            G_agg = (G_lag.sum(axis=0) > 0).astype(float)
            np.fill_diagonal(G_agg, 0)
            m_agg = compute_metrics(S_weight, G_agg)

            # 2. Lag-Jacobian extraction
            S_jac_lag = extract_lag_jacobian_score(
                model, X_tensor, max_lag=max_lag, device=device, seq_len=min(80, len(X) - max_lag - 1),
            )
            m_jac_lag = evaluate_lag_recovery(S_jac_lag, G_lag)

            # 3. Lag-Impulse extraction
            S_imp_lag = extract_lag_impulse_score(
                model, X_tensor, max_lag=max_lag, device=device, n_samples=10,
            )
            m_imp_lag = evaluate_lag_recovery(S_imp_lag, G_lag)

            # Store
            seed_key = f"seed_{seed}"
            ds_results[seed_key] = {
                "aggregate_weight": {"auroc": float(m_agg["auroc"]), "f1": float(m_agg["f1"])},
                "lag_jacobian": m_jac_lag,
                "lag_impulse": m_imp_lag,
            }

        # Summarize across seeds
        agg_aurocs = [ds_results[f"seed_{s}"]["aggregate_weight"]["auroc"] for s in range(n_seeds)]
        jac_agg_aurocs = [ds_results[f"seed_{s}"]["lag_jacobian"]["aggregate_auroc"] for s in range(n_seeds)]
        jac_lag_aurocs = [ds_results[f"seed_{s}"]["lag_jacobian"]["per_lag_auroc_mean"] for s in range(n_seeds)]
        imp_agg_aurocs = [ds_results[f"seed_{s}"]["lag_impulse"]["aggregate_auroc"] for s in range(n_seeds)]
        imp_lag_aurocs = [ds_results[f"seed_{s}"]["lag_impulse"]["per_lag_auroc_mean"] for s in range(n_seeds)]

        summary = {
            "aggregate_weight_auroc": f"{np.mean(agg_aurocs):.3f}±{np.std(agg_aurocs):.3f}",
            "lag_jacobian_agg_auroc": f"{np.mean(jac_agg_aurocs):.3f}±{np.std(jac_agg_aurocs):.3f}",
            "lag_jacobian_per_lag_auroc": f"{np.mean(jac_lag_aurocs):.3f}±{np.std(jac_lag_aurocs):.3f}",
            "lag_impulse_agg_auroc": f"{np.mean(imp_agg_aurocs):.3f}±{np.std(imp_agg_aurocs):.3f}",
            "lag_impulse_per_lag_auroc": f"{np.mean(imp_lag_aurocs):.3f}±{np.std(imp_lag_aurocs):.3f}",
        }

        print(f"\n  Results ({n_seeds} seeds):")
        print(f"    Aggregate Weight:        AUROC={summary['aggregate_weight_auroc']}")
        print(f"    Lag-Jacobian (aggregate): AUROC={summary['lag_jacobian_agg_auroc']}")
        print(f"    Lag-Jacobian (per-lag):   AUROC={summary['lag_jacobian_per_lag_auroc']}")
        print(f"    Lag-Impulse (aggregate):  AUROC={summary['lag_impulse_agg_auroc']}")
        print(f"    Lag-Impulse (per-lag):    AUROC={summary['lag_impulse_per_lag_auroc']}")

        ds_results["summary"] = summary
        all_results[ds_name] = ds_results

    total_time = time.time() - total_start

    # ── Decision Gate ──
    print(f"\n{'=' * 72}")
    print("DECISION GATE: Lag-Resolved Extraction")
    print(f"{'=' * 72}")

    lag_wins = 0
    for ds_name, ds_results in all_results.items():
        s = ds_results["summary"]
        agg = float(s["aggregate_weight_auroc"].split("±")[0])
        jac_lag = float(s["lag_jacobian_per_lag_auroc"].split("±")[0])
        imp_lag = float(s["lag_impulse_per_lag_auroc"].split("±")[0])
        best_lag = max(jac_lag, imp_lag)

        if best_lag > 0.6:
            lag_wins += 1
            print(f"  {ds_name:20s}: per-lag AUROC={best_lag:.3f} — LAG RECOVERY WORKS")
        else:
            print(f"  {ds_name:20s}: per-lag AUROC={best_lag:.3f} — lag recovery weak")

    print(f"\n  Datasets with lag recovery (AUROC > 0.6): {lag_wins}/{len(datasets)}")
    if lag_wins >= 2:
        print("  ✓ DECISION: Lag-resolved extraction shows promise. Continue.")
    else:
        print("  ✗ DECISION: Lag-resolved extraction does not work well enough.")

    print(f"\n  Total runtime: {total_time / 60:.1f} minutes")

    # Save
    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [serialize(v) for v in obj]
        return obj

    out_path = Path(save_dir) / "lag_resolved_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(all_results), f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/lag_resolved")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_lag_experiment(n_seeds=1, epochs=30, d_model=32, d_state=8, n_layers=2,
                          max_lag=3, device=args.device, save_dir=args.save_dir)
    else:
        run_lag_experiment(n_seeds=args.seeds, epochs=args.epochs, max_lag=args.max_lag,
                          device=args.device, save_dir=args.save_dir)
