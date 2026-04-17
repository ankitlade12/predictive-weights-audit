"""
Experiment 23: Intervention type × strength × baseline sweep
============================================================

Exp 22 showed the intervention-specific advantage (after matching sample
size) survives at p ≈ 3e-5 but the effect size is ~+0.03 to +0.05 AUROC —
much smaller than Exp 21's headline. Before calling this a workshop paper,
we need to know:

    (a) Does the effect survive under a proper do(X=c) intervention, or is
        it an artifact of Exp 21's random-forcing scheme?
    (b) Does the bottleneck still beat the *best* linear baseline (adding
        Ridge and RRR to the pool, not just Lasso)?
    (c) Is there a strength/type where the gap widens or collapses?

Arms per (K, seed, int_type, strength):
    S = X_obs_small       (T)
    B = X_obs_big         (T + K*T_int_per_target, same DGP)  ← control
    C = X_combined        (T obs + K*T_int_per_target int)

Interventions:
    random   — X[t,i] = strength * sign_t  (Exp 21/22 scheme, per-step jitter)
    constant — X[t,i] = strength           (standard do(X=c) clamp)
    soft     — X[t,i] = normal mechanism + strength * noise (doesn't sever)

Baselines: Lasso, Ridge, RRR (each tuned via validation).

Key metric for each (int_type, strength, K, baseline):
    int_specific_gap = (bn_C - bn_B) - (bl_C - bl_B)

If this gap is positive and significant for at least some intervention
types across at least some baselines, the workshop story holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.linalg import LinAlgWarning
from scipy.stats import ConstantInputWarning, wilcoxon
from sklearn.exceptions import ConvergenceWarning

from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    extract_lagged_weight_score,
    fit_tuned_lagged_linear_baseline,
    normalize_lagged_score,
    evaluate_lagged_scores,
)
from ssm_causal.lag_synthetic import (
    ensure_stable_lag_coefficients,
    simulate_lagged_var,
    standardize,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ═══════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════

def generate_random_var(K, T, density=0.2, max_lag=3, seed=42):
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
    return standardize(X), G_lag, [A_lags[t] for t in range(max_lag)]


def observational_from_coef(A_lags_list, T, seed):
    A_lags = np.stack(A_lags_list, axis=0)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
    return standardize(X)


def interventional_data(
    A_lags_list, K, T_per_target, max_lag, int_type, strength, seed,
):
    """
    int_type:
      "random"   — per-step clamp to strength * random sign
      "constant" — clamp to a fixed +strength for entire segment
                   (alternated sign across segments to get both directions)
      "soft"     — target follows normal mechanism plus strength-scaled noise
                   (does NOT sever incoming edges)
    """
    rng = np.random.RandomState(seed)
    all_X = []
    for target_var in range(K):
        X = np.zeros((T_per_target + max_lag, K))
        # alternate constant clamp sign across targets so we sample both ends
        const_val = strength * (1 if target_var % 2 == 0 else -1)
        for t in range(max_lag, T_per_target + max_lag):
            for tau in range(len(A_lags_list)):
                X[t] += A_lags_list[tau] @ X[t - tau - 1]
            X[t] += rng.randn(K) * 0.1
            if int_type == "random":
                X[t, target_var] = strength * rng.choice([-1, 1])
            elif int_type == "constant":
                X[t, target_var] = const_val
            elif int_type == "soft":
                X[t, target_var] += strength * rng.randn()
            else:
                raise ValueError(f"unknown int_type: {int_type}")
        all_X.append(X[max_lag:])
    return standardize(np.vstack(all_X))


# ═══════════════════════════════════════════
# Evaluators
# ═══════════════════════════════════════════

def eval_bottleneck(X, G_lag, max_lag, d_bottleneck, epochs, device):
    K = X.shape[1]
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(
        n_epochs=epochs, batch_size=min(64, len(X) - max_lag),
    )
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)
    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = np.nan_to_num(S_lag, 0)
    return evaluate_lagged_scores(S_lag, G_lag).get("flat_lag_auroc", 0.5)


def eval_baseline(X, G_lag, max_lag, method):
    kwargs = dict(X_train=X, max_lag=max_lag, method=method)
    if method in ("lasso", "ridge"):
        kwargs["alpha_values"] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    if method == "rrr":
        kwargs["alpha_values"] = [1e-3, 1e-2, 1e-1]
        kwargs["rank_values"] = [2, 4, 8, 16]
    result = fit_tuned_lagged_linear_baseline(**kwargs)
    S_lag = np.nan_to_num(result.score, 0)
    return evaluate_lagged_scores(S_lag, G_lag).get("flat_lag_auroc", 0.5)


# ═══════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════

def run_sweep(
    n_seeds,
    K_values,
    T,
    density,
    int_types,
    int_strengths,
    T_int_per_target,
    max_lag,
    d_bottleneck,
    epochs,
    baselines,
    device,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    total_start = time.time()

    print("=" * 80)
    print("EXP 23: Intervention sweep (type × strength × K × baseline)")
    print(
        f"seeds={n_seeds}  K={K_values}  T={T}  density={density}\n"
        f"int_types={int_types}  strengths={int_strengths}  baselines={baselines}"
    )
    print("=" * 80)

    # Results structure: results[K][int_type][strength][baseline] → list of per-seed dicts
    results: dict = {}

    for K in K_values:
        results[K] = {}
        T_int_total = K * T_int_per_target
        T_big = T + T_int_total
        print(f"\n─── K={K}  T_big={T_big}  T_int_total={T_int_total} ───")

        for int_type in int_types:
            results[K][int_type] = {}
            for strength in int_strengths:
                results[K][int_type][strength] = {b: [] for b in baselines}

        for seed in range(n_seeds):
            torch.manual_seed(seed * 42 + 7)
            # Fixed DGP per seed
            X_small, G_lag, A_lags = generate_random_var(
                K=K, T=T, density=density, max_lag=max_lag, seed=100 + seed,
            )
            X_big = observational_from_coef(A_lags, T=T_big, seed=300 + seed)

            # BN + all baselines on S and B (reused across all int configs)
            bn_S = eval_bottleneck(X_small, G_lag, max_lag, d_bottleneck, epochs, device)
            bn_B = eval_bottleneck(X_big, G_lag, max_lag, d_bottleneck, epochs, device)
            bl_S = {b: eval_baseline(X_small, G_lag, max_lag, b) for b in baselines}
            bl_B = {b: eval_baseline(X_big, G_lag, max_lag, b) for b in baselines}

            for int_type in int_types:
                for strength in int_strengths:
                    X_int = interventional_data(
                        A_lags, K=K, T_per_target=T_int_per_target,
                        max_lag=max_lag, int_type=int_type,
                        strength=strength, seed=200 + seed,
                    )
                    X_comb = np.vstack([X_small, X_int])

                    bn_C = eval_bottleneck(
                        X_comb, G_lag, max_lag, d_bottleneck, epochs, device,
                    )
                    bl_C = {b: eval_baseline(X_comb, G_lag, max_lag, b) for b in baselines}

                    for b in baselines:
                        rec = {
                            "seed": seed,
                            "bn_S": bn_S, "bn_B": bn_B, "bn_C": bn_C,
                            "bl_S": bl_S[b], "bl_B": bl_B[b], "bl_C": bl_C[b],
                            "bn_int_specific": bn_C - bn_B,
                            "bl_int_specific": bl_C[b] - bl_B[b],
                            "gap_int_specific": (bn_C - bn_B) - (bl_C[b] - bl_B[b]),
                            "gap_exp21_style": (bn_C - bn_S) - (bl_C[b] - bl_S[b]),
                        }
                        results[K][int_type][strength][b].append(rec)

            # Short progress print
            print(
                f"  seed={seed:2d}  bn_S={bn_S:.3f} bn_B={bn_B:.3f}  "
                + " ".join(f"{b}_B={bl_B[b]:.3f}" for b in baselines)
            )

    # ═══════════════════════════════════════════
    # Aggregate + test
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("RESULTS TABLE: intervention-specific gap (BN minus baseline), after size match")
    print("=" * 80)
    header = (
        f"{'K':>3} {'int_type':>9} {'strength':>8} {'baseline':>9} "
        f"{'gap_mean':>10} {'gap_med':>8} {'win_rt':>7} {'p_gt_0':>9} "
        f"{'exp21_gap':>10}"
    )
    print(header)
    print("-" * len(header))

    flat: list[dict] = []
    for K in K_values:
        for int_type in int_types:
            for strength in int_strengths:
                for b in baselines:
                    recs = results[K][int_type][strength][b]
                    gaps_is = np.array([r["gap_int_specific"] for r in recs])
                    gaps_e21 = np.array([r["gap_exp21_style"] for r in recs])
                    try:
                        _, p = wilcoxon(gaps_is, alternative="greater")
                    except Exception:
                        p = 1.0
                    win_rate = float(np.mean(gaps_is > 0.01))
                    row = {
                        "K": K, "int_type": int_type, "strength": strength,
                        "baseline": b,
                        "gap_int_specific_mean": float(gaps_is.mean()),
                        "gap_int_specific_median": float(np.median(gaps_is)),
                        "gap_int_specific_wins_rate": win_rate,
                        "p_gap_int_specific_gt_0": float(p),
                        "gap_exp21_style_mean": float(gaps_e21.mean()),
                        "n_seeds": len(recs),
                        "per_seed": recs,
                    }
                    flat.append(row)
                    print(
                        f"{K:>3} {int_type:>9} {strength:>8.2f} {b:>9} "
                        f"{gaps_is.mean():>+10.3f} {np.median(gaps_is):>+8.3f} "
                        f"{win_rate:>7.2f} {p:>9.1e} "
                        f"{gaps_e21.mean():>+10.3f}"
                    )

    # Survivability summary
    sig_rows = [r for r in flat if r["p_gap_int_specific_gt_0"] < 0.05 and r["gap_int_specific_mean"] > 0]
    print(f"\nCONFIGS where BN beats baseline on intervention-specific gap (p<0.05): "
          f"{len(sig_rows)}/{len(flat)}")

    # Per baseline
    print("\nBy baseline:")
    for b in baselines:
        brows = [r for r in flat if r["baseline"] == b]
        sig = [r for r in brows if r["p_gap_int_specific_gt_0"] < 0.05 and r["gap_int_specific_mean"] > 0]
        mean_gap = np.mean([r["gap_int_specific_mean"] for r in brows])
        print(f"  {b:>8s}: sig configs {len(sig)}/{len(brows)}  mean_gap={mean_gap:+.3f}")

    # Per intervention type
    print("\nBy intervention type:")
    for it in int_types:
        trows = [r for r in flat if r["int_type"] == it]
        sig = [r for r in trows if r["p_gap_int_specific_gt_0"] < 0.05 and r["gap_int_specific_mean"] > 0]
        mean_gap = np.mean([r["gap_int_specific_mean"] for r in trows])
        print(f"  {it:>9s}: sig configs {len(sig)}/{len(trows)}  mean_gap={mean_gap:+.3f}")

    runtime = (time.time() - total_start) / 60
    print(f"\nRuntime: {runtime:.1f} min")

    out_path = Path(save_dir) / "intervention_sweep_results.json"
    serialized = []
    for r in flat:
        r2 = {k: v for k, v in r.items() if k != "per_seed"}
        r2["per_seed"] = [
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in rec.items()}
            for rec in r["per_seed"]
        ]
        serialized.append(r2)
    with out_path.open("w") as f:
        json.dump(serialized, f, indent=2)
    print(f"Saved to {out_path}")

    return flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--K", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--int-types", nargs="+",
                        default=["random", "constant", "soft"])
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[1.0, 2.0])
    parser.add_argument("--T-int-per-target", type=int, default=50)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--d-bottleneck", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--baselines", nargs="+",
                        default=["lasso", "ridge", "rrr"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/intervention_sweep")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_sweep(
            n_seeds=3, K_values=(20,), T=300, density=0.2,
            int_types=("random", "constant"), int_strengths=(1.0, 2.0),
            T_int_per_target=50, max_lag=3, d_bottleneck=16, epochs=30,
            baselines=("lasso", "ridge"),
            device=args.device, save_dir=args.save_dir,
        )
    else:
        run_sweep(
            n_seeds=args.seeds, K_values=tuple(args.K), T=args.T,
            density=args.density,
            int_types=tuple(args.int_types),
            int_strengths=tuple(args.strengths),
            T_int_per_target=args.T_int_per_target, max_lag=args.max_lag,
            d_bottleneck=args.d_bottleneck, epochs=args.epochs,
            baselines=tuple(args.baselines),
            device=args.device, save_dir=args.save_dir,
        )
