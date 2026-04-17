"""
Experiment 22: Intervention vs Size-Matched Observational Control
==================================================================

Exp 21 claimed bottleneck benefits more from interventional data than Lasso
(p < 1e-5, 80-100% win rate). But the "combined" arm has T + K*T_int_per_target
samples while the "observational" arm has only T samples. The gain metric
therefore conflates two things:

    gain(combined) = gain(more data) + gain(intervention content)

This experiment separates them by adding a third arm, X_obs_big, which has
the *same total sample count* as X_combined but is drawn entirely from the
observational distribution. The intervention-specific effect is then:

    intervention_specific_gain = AUROC(combined) - AUROC(obs_big)

Null hypothesis: once we match sample size, the bottleneck's "intervention
advantage" vanishes — i.e., Exp 21's effect is a data-size confound that
scales with K.

We test both bottleneck and Lasso on:
    A: X_obs_small            (T samples)
    B: X_obs_big              (T + K*T_int_per_target samples, all obs)
    C: X_combined             (T obs + K*T_int_per_target intervention)

The headline number is the Wilcoxon test on
    (bn_combined - bn_obs_big) - (bl_combined - bl_obs_big)

If p < 0.05 and positive: intervention content truly helps BN more.
If not: Exp 21's signal was sample size all along.

Usage:
    uv run python experiments/exp22_intervention_control.py --quick
    uv run python experiments/exp22_intervention_control.py
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
    normalize_lagged_score,
    evaluate_lagged_scores,
)


# ═══════════════════════════════════════════
# Data generation — mirror Exp 21 scheme exactly so the comparison is fair
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
    """Draw fresh observational data from the same VAR coefficients."""
    from ssm_causal.lag_synthetic import simulate_lagged_var, standardize
    A_lags = np.stack(A_lags_list, axis=0)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
    return standardize(X)


def generate_interventional_data(A_lags_list, K, T_per_target, max_lag, strength=1.0, seed=42):
    """Mirror Exp 21: hard clamp per step to strength * random sign."""
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
# Evaluators
# ═══════════════════════════════════════════

def eval_bottleneck(X, G_lag, max_lag=3, d_bottleneck=16, epochs=80, device="cpu"):
    K = X.shape[1]
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(
        n_epochs=epochs, batch_size=min(64, len(X) - max_lag)
    )
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)

    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = np.nan_to_num(S_lag, 0)

    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return metrics.get("flat_lag_auroc", 0.5)


def eval_lasso(X, G_lag, max_lag=3):
    result = fit_tuned_lagged_linear_baseline(
        X, max_lag=max_lag, method="lasso",
        alpha_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    S_lag = np.nan_to_num(result.score, 0)
    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return metrics.get("flat_lag_auroc", 0.5)


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def run_control(
    n_seeds=10,
    K_values=(10, 20, 30),
    T_values=(300,),
    density=0.2,
    int_strength=1.0,
    T_int_per_target=50,
    max_lag=3,
    d_bottleneck=16,
    epochs=80,
    device="cpu",
    save_dir="results/intervention_control",
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 22: Intervention vs Size-Matched Obs Control")
    print(f"Seeds: {n_seeds} | K: {K_values} | T: {T_values}")
    print(f"Device: {device}")
    print("=" * 72)

    all_results = {}

    for K in K_values:
        for T in T_values:
            config_name = f"K={K}_T={T}_d={density}"
            T_int_total = K * T_int_per_target
            T_big = T + T_int_total

            print(f"\n{'─' * 60}")
            print(f"Config: {config_name}  (T_big={T_big}, T_int_total={T_int_total})")
            print(f"{'─' * 60}")

            per_seed = []

            for seed in range(n_seeds):
                torch.manual_seed(seed * 42 + 7)

                # Observational data from same DGP
                X_obs_small, G_lag, A_lags = generate_random_var(
                    K=K, T=T, density=density, max_lag=max_lag, seed=100 + seed,
                )

                # Size-matched fresh observational draw (the CONTROL)
                X_obs_big = generate_observational_from_coef(
                    A_lags, K=K, T=T_big, max_lag=max_lag, seed=300 + seed,
                )

                # Interventional data (friend's scheme, mirrors Exp 21)
                X_int = generate_interventional_data(
                    A_lags, K=K, T_per_target=T_int_per_target,
                    max_lag=max_lag, strength=int_strength, seed=200 + seed,
                )
                X_combined = np.vstack([X_obs_small, X_int])

                # Sanity: combined and obs_big should have same length
                assert len(X_combined) == len(X_obs_big), (
                    f"size mismatch: combined={len(X_combined)} big={len(X_obs_big)}"
                )

                # Evaluate all three arms for both methods
                bn_small = eval_bottleneck(X_obs_small, G_lag, max_lag, d_bottleneck, epochs, device)
                bn_big = eval_bottleneck(X_obs_big, G_lag, max_lag, d_bottleneck, epochs, device)
                bn_comb = eval_bottleneck(X_combined, G_lag, max_lag, d_bottleneck, epochs, device)

                bl_small = eval_lasso(X_obs_small, G_lag, max_lag)
                bl_big = eval_lasso(X_obs_big, G_lag, max_lag)
                bl_comb = eval_lasso(X_combined, G_lag, max_lag)

                # Key metric: intervention-specific benefit (combined vs size-matched obs)
                bn_int_specific = bn_comb - bn_big
                bl_int_specific = bl_comb - bl_big

                # Exp 21 metric: combined vs small (confounded with data size)
                bn_exp21 = bn_comb - bn_small
                bl_exp21 = bl_comb - bl_small

                # Pure data-size effect
                bn_data_only = bn_big - bn_small
                bl_data_only = bl_big - bl_small

                per_seed.append({
                    "seed": seed,
                    "bn_small": bn_small, "bn_big": bn_big, "bn_comb": bn_comb,
                    "bl_small": bl_small, "bl_big": bl_big, "bl_comb": bl_comb,
                    "bn_int_specific": bn_int_specific,
                    "bl_int_specific": bl_int_specific,
                    "bn_exp21_style": bn_exp21,
                    "bl_exp21_style": bl_exp21,
                    "bn_data_only": bn_data_only,
                    "bl_data_only": bl_data_only,
                })

                print(
                    f"  seed={seed:2d} | BN s={bn_small:.3f} b={bn_big:.3f} c={bn_comb:.3f} "
                    f"(int-spec={bn_int_specific:+.3f}) | "
                    f"BL s={bl_small:.3f} b={bl_big:.3f} c={bl_comb:.3f} "
                    f"(int-spec={bl_int_specific:+.3f})"
                )

            # Aggregate
            bn_is = np.array([r["bn_int_specific"] for r in per_seed])
            bl_is = np.array([r["bl_int_specific"] for r in per_seed])
            bn_e21 = np.array([r["bn_exp21_style"] for r in per_seed])
            bl_e21 = np.array([r["bl_exp21_style"] for r in per_seed])
            bn_do = np.array([r["bn_data_only"] for r in per_seed])
            bl_do = np.array([r["bl_data_only"] for r in per_seed])

            # Wilcoxon: does BN benefit from intervention content MORE than Lasso, after size matching?
            try:
                _, p_int_specific = wilcoxon(bn_is, bl_is, alternative="greater")
            except Exception:
                p_int_specific = 1.0

            # Reproduce Exp 21's original test for calibration
            try:
                _, p_exp21 = wilcoxon(bn_e21, bl_e21, alternative="greater")
            except Exception:
                p_exp21 = 1.0

            result = {
                "K": K, "T": T, "T_big": T_big, "T_int_total": T_int_total,
                "n_seeds": n_seeds,
                "mean_bn_int_specific": float(bn_is.mean()),
                "mean_bl_int_specific": float(bl_is.mean()),
                "bn_int_specific_wins_rate": float(np.mean((bn_is - bl_is) > 0.01)),
                "p_int_specific": float(p_int_specific),
                "mean_bn_exp21_style": float(bn_e21.mean()),
                "mean_bl_exp21_style": float(bl_e21.mean()),
                "p_exp21_style": float(p_exp21),
                "mean_bn_data_only": float(bn_do.mean()),
                "mean_bl_data_only": float(bl_do.mean()),
                "per_seed": [
                    {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                     for k, v in r.items()}
                    for r in per_seed
                ],
            }
            all_results[config_name] = result

            print(
                f"\n  [EXP21 STYLE]  BN={bn_e21.mean():+.3f}  BL={bl_e21.mean():+.3f}  "
                f"p={p_exp21:.2e}  "
                f"{'replicates' if p_exp21 < 0.05 else 'no-repl'}"
            )
            print(
                f"  [SIZE-MATCHED] BN={bn_is.mean():+.3f}  BL={bl_is.mean():+.3f}  "
                f"p={p_int_specific:.2e}  "
                f"{'SURVIVES' if p_int_specific < 0.05 else 'CONFOUNDED'}"
            )
            print(
                f"  [DATA-ONLY]    BN={bn_do.mean():+.3f}  BL={bl_do.mean():+.3f}  "
                f"(extra obs-only data benefit)"
            )

    # ── Verdict ──
    print(f"\n{'=' * 72}")
    print("CONTROL-EXPERIMENT VERDICT")
    print(f"{'=' * 72}")

    surv = 0
    for config, r in all_results.items():
        tag_21 = "repl" if r["p_exp21_style"] < 0.05 else "noop"
        tag_sm = "SURVIVES" if r["p_int_specific"] < 0.05 else "CONFOUND"
        print(
            f"  {config:22s} | "
            f"exp21-style p={r['p_exp21_style']:.1e} ({tag_21})  "
            f"size-matched p={r['p_int_specific']:.1e} ({tag_sm})"
        )
        if r["p_int_specific"] < 0.05:
            surv += 1

    total = len(all_results)
    print(f"\n  Size-matched significant: {surv}/{total}")
    if surv >= total * 0.5:
        print("  → Intervention content does carry signal beyond sample size.")
    elif surv >= 1:
        print("  → Regime-specific intervention signal; mostly a data-size effect.")
    else:
        print("  → Exp 21 was a sample-size confound. No intervention-specific advantage.")

    print(f"\n  Runtime: {(time.time() - total_start) / 60:.1f} min")

    out_path = Path(save_dir) / "intervention_control_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--K", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--T", type=int, nargs="+", default=[300])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/intervention_control")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_control(
            n_seeds=3, K_values=(10, 20), T_values=(300,),
            epochs=30, device=args.device, save_dir=args.save_dir,
        )
    else:
        run_control(
            n_seeds=args.seeds, K_values=tuple(args.K), T_values=tuple(args.T),
            epochs=args.epochs, device=args.device, save_dir=args.save_dir,
        )
