"""
Experiment 25: Causal Baselines (PCMCI + Granger) on Exp 22/23 Control Setup
============================================================================

Reviewers at ICBINB will ask: "Did you compare against actual causal discovery
methods, or just prediction baselines?" Exps 22 and 23 compared bottleneck
against Lasso/Ridge/RRR — all prediction baselines. This experiment adds
PCMCI and bivariate Granger to the same size-matched intervention-control
setup, so our falsification story includes proper causal discovery methods.

Setup mirrors Exp 22 exactly:
    A: X_obs_small       (T samples)
    B: X_obs_big         (T + K*T_int_per_target samples, all observational)
    C: X_combined        (T observational + K*T_int_per_target intervention)

Methods evaluated:
    - Bottleneck (for reference)
    - Tuned Lasso (for reference)
    - PCMCI (actual causal discovery)
    - Bivariate Granger (classical causal discovery)

Key question: Is the residual size-matched intervention effect unique to
the bottleneck? Expected answer: no. Granger reproduces the effect, often
with larger gains, so the residual is method-agnostic.

Usage:
    python experiments/exp25_causal_baselines.py --device cuda
    python experiments/exp25_causal_baselines.py --quick
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
    evaluate_lagged_scores,
    extract_lagged_weight_score,
    fit_tuned_lagged_linear_baseline,
    normalize_lagged_score,
)

# ═══════════════════════════════════════════
# Data generation (mirrors Exp 22)
# ═══════════════════════════════════════════

def generate_random_var(K, T, density=0.2, max_lag=3, seed=42):
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


def generate_observational_from_coef(A_lags_list, K, T, max_lag, seed):
    from ssm_causal.lag_synthetic import simulate_lagged_var, standardize
    A_lags = np.stack(A_lags_list, axis=0)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
    return standardize(X)


def generate_interventional_data(A_lags_list, K, T_per_target, max_lag, strength=1.0, seed=42):
    """Exp 21 scheme: random-forcing (per-step clamp)."""
    from ssm_causal.lag_synthetic import standardize
    rng = np.random.RandomState(seed)
    all_X = []
    for target_var in range(K):
        X = np.zeros((T_per_target + max_lag, K))
        for t in range(max_lag, T_per_target + max_lag):
            for tau in range(len(A_lags_list)):
                X[t] += A_lags_list[tau] @ X[t - tau - 1]
            X[t] += rng.randn(K) * 0.1
            X[t, target_var] = strength * rng.choice([-1, 1])
        all_X.append(X[max_lag:])
    result = np.vstack(all_X)
    return standardize(result)


# ═══════════════════════════════════════════
# Evaluators (reference + new causal baselines)
# ═══════════════════════════════════════════

def eval_bottleneck(X, G_lag, max_lag=3, d_bottleneck=16, epochs=80, device="cpu"):
    K = X.shape[1]
    torch.manual_seed(42)
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(
        n_epochs=epochs, batch_size=min(64, len(X) - max_lag)
    )
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)

    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = np.nan_to_num(S_lag, 0)
    return evaluate_lagged_scores(S_lag, G_lag).get("flat_lag_auroc", 0.5)


def eval_lasso(X, G_lag, max_lag=3):
    result = fit_tuned_lagged_linear_baseline(
        X, max_lag=max_lag, method="lasso",
        alpha_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    S_lag = np.nan_to_num(result.score, 0)
    return evaluate_lagged_scores(S_lag, G_lag).get("flat_lag_auroc", 0.5)


def eval_pcmci(X, G_lag, max_lag=3):
    """
    PCMCI returns an aggregate causal matrix. To evaluate against the lag
    tensor, we broadcast the aggregate score across lags (since PCMCI
    without --tau-specific scoring gives a collapsed graph).
    """
    try:
        from baselines.pcmci_wrapper import pcmci_with_details
        details = pcmci_with_details(X, tau_max=max_lag, pc_alpha=0.05)
        val = details.get("val_matrix")  # shape (K, K, tau_max + 1)
        if val is None:
            # Fall back to aggregate
            from baselines.pcmci_wrapper import pcmci_causality_matrix
            S_agg = pcmci_causality_matrix(X, tau_max=max_lag)
            S_lag = np.broadcast_to(S_agg, (max_lag, *S_agg.shape)).copy()
        else:
            # val has shape (K, K, tau_max+1) — take lags 1..tau_max
            val = np.abs(val)
            S_lag = np.zeros((max_lag, val.shape[0], val.shape[1]))
            for tau in range(max_lag):
                if tau + 1 < val.shape[2]:
                    S_lag[tau] = val[:, :, tau + 1]
            # Normalize per-lag
            for tau in range(max_lag):
                np.fill_diagonal(S_lag[tau], 0)
                if S_lag[tau].max() > 0:
                    S_lag[tau] = S_lag[tau] / S_lag[tau].max()

        S_lag = np.nan_to_num(S_lag, 0)
        return evaluate_lagged_scores(S_lag, G_lag).get("flat_lag_auroc", 0.5)
    except Exception as e:
        print(f"      PCMCI error: {e}")
        return 0.5


def eval_granger(X, G_lag, max_lag=3):
    """
    Bivariate Granger — returns aggregate causal strength. Broadcast across lags.
    """
    try:
        from baselines.granger import granger_causality_matrix
        S_agg = granger_causality_matrix(X, max_lag=max_lag)
        S_agg = np.nan_to_num(S_agg, 0)
        # Broadcast across lags (Granger aggregates by construction)
        S_lag = np.broadcast_to(S_agg, (max_lag, *S_agg.shape)).copy()
        for tau in range(max_lag):
            np.fill_diagonal(S_lag[tau], 0)
            if S_lag[tau].max() > 0:
                S_lag[tau] = S_lag[tau] / S_lag[tau].max()
        return evaluate_lagged_scores(S_lag, G_lag).get("flat_lag_auroc", 0.5)
    except Exception as e:
        print(f"      Granger error: {e}")
        return 0.5


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def run_causal_baselines(
    n_seeds=10,
    K_values=(10, 20, 30),
    T_values=(300,),
    density=0.2,
    max_lag=3,
    T_int_per_target=50,
    intervention_strength=1.0,
    d_bottleneck=16,
    epochs=80,
    device="cpu",
    save_dir="results/causal_baselines",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 25: Causal Baselines (PCMCI + Granger)")
    print(f"Seeds: {n_seeds} | K: {K_values} | T: {T_values}")
    print(f"Device: {device}")
    print("Protocol: mirror Exp 22 size-matched setup, add PCMCI and Granger")
    print("=" * 72)

    methods = {
        "Bottleneck": lambda X, G: eval_bottleneck(X, G, max_lag, d_bottleneck, epochs, device),
        "Lasso": lambda X, G: eval_lasso(X, G, max_lag),
        "PCMCI": lambda X, G: eval_pcmci(X, G, max_lag),
        "Granger": lambda X, G: eval_granger(X, G, max_lag),
    }

    all_results = {}

    for K in K_values:
        for T in T_values:
            cfg = f"K={K}_T={T}"
            print(f"\n{'─' * 60}\nConfig: {cfg}\n{'─' * 60}")

            # Per-method arrays
            scores = {m: {"small": [], "big": [], "combined": []} for m in methods}

            for seed in range(n_seeds):
                torch.manual_seed(seed * 42 + 7)

                # Generate coefficient matrices + small observational draw
                X_small, G_lag, A_lags = generate_random_var(
                    K=K, T=T, density=density, max_lag=max_lag, seed=100 + seed,
                )

                # Big observational draw from same coefficients
                T_big = T + K * T_int_per_target
                X_big = generate_observational_from_coef(
                    A_lags, K=K, T=T_big, max_lag=max_lag, seed=500 + seed,
                )

                # Interventional (Exp 21 random-forcing scheme)
                X_int = generate_interventional_data(
                    A_lags, K=K, T_per_target=T_int_per_target,
                    max_lag=max_lag, strength=intervention_strength, seed=200 + seed,
                )
                X_combined = np.vstack([X_small, X_int])

                for m_name, m_fn in methods.items():
                    try:
                        s_small = m_fn(X_small, G_lag)
                        s_big = m_fn(X_big, G_lag)
                        s_comb = m_fn(X_combined, G_lag)
                    except Exception as e:
                        print(f"      [WARN] {m_name} seed={seed}: {e}")
                        s_small = s_big = s_comb = 0.5

                    scores[m_name]["small"].append(s_small)
                    scores[m_name]["big"].append(s_big)
                    scores[m_name]["combined"].append(s_comb)

                print(f"  seed={seed:2d} | " + " | ".join(
                    f"{m}: comb={scores[m]['combined'][-1]:.3f} big={scores[m]['big'][-1]:.3f}"
                    for m in ("Bottleneck", "Lasso", "PCMCI", "Granger")
                ))

            # Aggregate
            cfg_result = {}
            for m in methods:
                small_arr = np.array(scores[m]["small"])
                big_arr = np.array(scores[m]["big"])
                comb_arr = np.array(scores[m]["combined"])

                # Size-matched intervention gap
                gap = comb_arr - big_arr
                # Raw (unmatched) intervention gain
                raw_gain = comb_arr - small_arr

                try:
                    stat, p_val = wilcoxon(gap, alternative="greater")
                except Exception:
                    p_val = 1.0

                cfg_result[m] = {
                    "small_mean": float(small_arr.mean()),
                    "big_mean": float(big_arr.mean()),
                    "combined_mean": float(comb_arr.mean()),
                    "raw_gain_mean": float(raw_gain.mean()),
                    "size_matched_gap_mean": float(gap.mean()),
                    "wilcoxon_p": float(p_val),
                    "significant": bool(p_val < 0.05),
                    "small_all": small_arr.tolist(),
                    "big_all": big_arr.tolist(),
                    "combined_all": comb_arr.tolist(),
                }

            all_results[cfg] = cfg_result

            print(f"\n  Summary for {cfg}:")
            print(
                f"  {'Method':<12} {'small':>8} {'big':>8} {'comb':>8} "
                f"{'raw_gain':>10} {'matched':>10} {'p':>10}"
            )
            for m in methods:
                r = cfg_result[m]
                tag = "✓" if r["significant"] else " "
                print(
                    f"  {m:<12} "
                    f"{r['small_mean']:>8.3f} {r['big_mean']:>8.3f} {r['combined_mean']:>8.3f} "
                    f"{r['raw_gain_mean']:>+10.3f} {r['size_matched_gap_mean']:>+10.3f} "
                    f"{r['wilcoxon_p']:>10.2e} {tag}"
                )

    total_time = time.time() - total_start
    print(f"\n{'=' * 72}\nFINAL VERDICT\n{'=' * 72}")
    print(f"{'Config':<15} " + " ".join(f"{m:<12}" for m in methods))
    for cfg, res in all_results.items():
        row = f"{cfg:<15} "
        for m in methods:
            gap = res[m]["size_matched_gap_mean"]
            sig = "*" if res[m]["significant"] else ""
            row += f"{gap:+.3f}{sig:<7} "
        print(row)
    print(f"\nTotal runtime: {total_time / 60:.1f} min")

    out_path = Path(save_dir) / "causal_baselines_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--K-values", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--T-values", type=int, nargs="+", default=[300])
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--T-int-per-target", type=int, default=50)
    parser.add_argument("--intervention-strength", type=float, default=1.0)
    parser.add_argument("--d-bottleneck", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/causal_baselines")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 3
        args.K_values = [10]
        args.epochs = 20

    run_causal_baselines(
        n_seeds=args.seeds,
        K_values=tuple(args.K_values),
        T_values=tuple(args.T_values),
        density=args.density,
        max_lag=args.max_lag,
        T_int_per_target=args.T_int_per_target,
        intervention_strength=args.intervention_strength,
        d_bottleneck=args.d_bottleneck,
        epochs=args.epochs,
        device=args.device,
        save_dir=args.save_dir,
    )
