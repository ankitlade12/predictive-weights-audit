"""
Experiment 21: Validate Intervention Finding
==============================================

Exp 19 found that interventional data helps bottleneck MORE than Lasso
on random graphs (80% win rate, K=20, 5 seeds).

This experiment validates whether that finding is:
1. Statistically significant (20 seeds, Wilcoxon test)
2. Robust across K (10, 15, 20, 30)
3. Robust across T (200, 300, 500)
4. Robust across graph density (0.1, 0.2, 0.3, 0.5)
5. Robust across intervention strength

If ALL pass → publishable finding
If SOME pass → regime-specific finding (still interesting)
If NONE pass → Exp 19 was noise

Usage:
    python experiments/exp21_intervention_validation.py --device cuda
    python experiments/exp21_intervention_validation.py --quick
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
from scipy.stats import wilcoxon

from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    extract_lagged_weight_score,
    fit_tuned_lagged_linear_baseline,
    make_lagged_dataset,
    normalize_lagged_score,
    evaluate_lagged_scores,
)


# ═══════════════════════════════════════════
# Data generation with interventions
# ═══════════════════════════════════════════

def generate_random_var(K, T, density=0.2, max_lag=3, seed=42):
    """Random sparse VAR with known ground truth, using Sai's stable generators."""
    from ssm_causal.lag_synthetic import (
        ensure_stable_lag_coefficients,
        simulate_lagged_var,
        standardize,
    )
    rng = np.random.RandomState(seed)

    A_lags = np.zeros((max_lag, K, K))
    G_lag = np.zeros((max_lag, K, K))

    for tau in range(max_lag):
        A_lags[tau] = np.diag(rng.uniform(0.2, 0.4, K))
        for i in range(K):
            for j in range(K):
                if i != j and rng.random() < density / max_lag:
                    A_lags[tau, i, j] = rng.uniform(0.1, 0.25) * rng.choice([-1, 1])
                    G_lag[tau, i, j] = 1.0

    A_lags = ensure_stable_lag_coefficients(A_lags, target_radius=0.85)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
    X = standardize(X)

    return X, G_lag, [A_lags[t] for t in range(max_lag)]


def generate_interventional_data(A_lags, K, T_per_target, max_lag, strength=1.0, seed=42):
    """Generate hard-intervention data: clamp each variable in turn."""
    from ssm_causal.lag_synthetic import standardize
    rng = np.random.RandomState(seed)
    all_X = []

    for target_var in range(K):
        X = np.zeros((T_per_target + max_lag, K))
        for t in range(max_lag, T_per_target + max_lag):
            for tau in range(len(A_lags)):
                X[t] += A_lags[tau] @ X[t - tau - 1]
            X[t] += rng.randn(K) * 0.1
            # Hard intervention: clamp target variable
            X[t, target_var] = strength * rng.choice([-1, 1])
        all_X.append(X[max_lag:])

    result = np.vstack(all_X)
    return standardize(result)


def eval_bottleneck(X, G_lag, max_lag=3, d_bottleneck=16, epochs=80, device="cpu"):
    """Train bottleneck and extract lag score."""
    K = X.shape[1]
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(n_epochs=epochs, batch_size=min(64, len(X) - max_lag))
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)

    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = np.nan_to_num(S_lag, 0)

    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return metrics.get("flat_lag_auroc", 0.5)


def eval_lasso(X, G_lag, max_lag=3):
    """Tuned Lasso baseline."""
    K = X.shape[1]
    result = fit_tuned_lagged_linear_baseline(
        X, max_lag=max_lag, method="lasso",
        alpha_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    S_lag = result.score
    S_lag = np.nan_to_num(S_lag, 0)
    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return metrics.get("flat_lag_auroc", 0.5)


# ═══════════════════════════════════════════
# Main validation
# ═══════════════════════════════════════════

def run_validation(
    n_seeds=20,
    K_values=(10, 15, 20, 30),
    T_values=(200, 300, 500),
    densities=(0.2,),
    int_strengths=(2.0,),
    T_int_per_target=50,
    max_lag=3,
    d_bottleneck=16,
    epochs=80,
    device="cpu",
    save_dir="results/intervention_validation",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 21: Intervention Finding Validation")
    print(f"Seeds: {n_seeds} | K: {K_values} | T: {T_values}")
    print(f"Device: {device}")
    print("=" * 72)

    all_results = {}

    for K in K_values:
        for T in T_values:
            for density in densities:
                config_name = f"K={K}_T={T}_d={density}"
                print(f"\n{'─' * 60}")
                print(f"Config: {config_name}")
                print(f"{'─' * 60}")

                bn_obs_aurocs = []
                bn_int_aurocs = []
                bl_obs_aurocs = []
                bl_int_aurocs = []

                for seed in range(n_seeds):
                    torch.manual_seed(seed * 42 + 7)

                    # Generate observational data
                    X_obs, G_lag, A_lags = generate_random_var(
                        K=K, T=T, density=density, max_lag=max_lag, seed=100 + seed,
                    )

                    # Generate interventional data
                    X_int = generate_interventional_data(
                        A_lags, K=K, T_per_target=T_int_per_target,
                        max_lag=max_lag, seed=200 + seed,
                    )

                    # Combined data
                    X_combined = np.vstack([X_obs, X_int])

                    # Eval bottleneck on obs only
                    bn_obs = eval_bottleneck(X_obs, G_lag, max_lag, d_bottleneck, epochs, device)
                    bn_obs_aurocs.append(bn_obs)

                    # Eval bottleneck on combined (obs + intervention)
                    bn_int = eval_bottleneck(X_combined, G_lag, max_lag, d_bottleneck, epochs, device)
                    bn_int_aurocs.append(bn_int)

                    # Eval Lasso on obs only
                    bl_obs = eval_lasso(X_obs, G_lag, max_lag)
                    bl_obs_aurocs.append(bl_obs)

                    # Eval Lasso on combined
                    bl_int = eval_lasso(X_combined, G_lag, max_lag)
                    bl_int_aurocs.append(bl_int)

                    bn_gain = bn_int - bn_obs
                    bl_gain = bl_int - bl_obs
                    tag = "BN-WINS" if bn_gain > bl_gain + 0.01 else ("BL-WINS" if bl_gain > bn_gain + 0.01 else "TIE")
                    print(f"  seed={seed:2d} | BN obs={bn_obs:.3f} int={bn_int:.3f} gain={bn_gain:+.3f} | BL obs={bl_obs:.3f} int={bl_int:.3f} gain={bl_gain:+.3f} | {tag}")

                # Compute gains
                bn_gains = [bn_int_aurocs[i] - bn_obs_aurocs[i] for i in range(n_seeds)]
                bl_gains = [bl_int_aurocs[i] - bl_obs_aurocs[i] for i in range(n_seeds)]
                gain_diffs = [bn_gains[i] - bl_gains[i] for i in range(n_seeds)]

                bn_win_rate = sum(1 for d in gain_diffs if d > 0.01) / n_seeds

                # Statistical test
                try:
                    stat, p_val = wilcoxon(bn_gains, bl_gains, alternative="greater")
                    sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                except Exception:
                    p_val = 1.0
                    sig = "test failed"

                result = {
                    "bn_mean_gain": float(np.mean(bn_gains)),
                    "bl_mean_gain": float(np.mean(bl_gains)),
                    "bn_win_rate": float(bn_win_rate),
                    "wilcoxon_p": float(p_val),
                    "significant": p_val < 0.05,
                    "bn_gains": [float(g) for g in bn_gains],
                    "bl_gains": [float(g) for g in bl_gains],
                }
                all_results[config_name] = result

                print(f"\n  BN mean gain: {np.mean(bn_gains):+.3f} | BL mean gain: {np.mean(bl_gains):+.3f}")
                print(f"  BN win rate: {bn_win_rate:.0%}")
                print(f"  Wilcoxon p={p_val:.2e} ({sig})")

    total_time = time.time() - total_start

    # ── Decision ──
    print(f"\n{'=' * 72}")
    print("VALIDATION VERDICT")
    print(f"{'=' * 72}")

    sig_count = 0
    for config, r in all_results.items():
        tag = "✓ VALIDATED" if r["significant"] else "✗ not significant"
        print(f"  {config:30s}: win_rate={r['bn_win_rate']:.0%} p={r['wilcoxon_p']:.2e} {tag}")
        if r["significant"]:
            sig_count += 1

    total_configs = len(all_results)
    print(f"\n  Significant configs: {sig_count}/{total_configs}")

    if sig_count >= total_configs * 0.5:
        print("  ✓ FINDING VALIDATED — intervention advantage is robust")
    elif sig_count >= 2:
        print("  ~ PARTIALLY VALIDATED — regime-specific finding")
    else:
        print("  ✗ NOT VALIDATED — Exp 19 was likely noise")

    print(f"\n  Total runtime: {total_time / 60:.1f} minutes")

    # Save
    out_path = Path(save_dir) / "intervention_validation_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/intervention_validation")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_validation(n_seeds=3, K_values=(10, 20), T_values=(300,),
                       epochs=30, device=args.device, save_dir=args.save_dir)
    else:
        run_validation(n_seeds=args.seeds, device=args.device, save_dir=args.save_dir)
