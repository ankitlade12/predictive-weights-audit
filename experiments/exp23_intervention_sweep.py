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
    constant — X[t,i] = ±strength          (alternating signed do(X=c) clamps)
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
from scipy.stats import ConstantInputWarning
from sklearn.exceptions import ConvergenceWarning

from ssm_causal.intervention_protocol import (
    SegmentedSeries,
    apply_holm_family,
    build_raw_intervention_arms,
    paired_effect_inference,
    preprocess_intervention_arms,
    require_finite_array,
    require_finite_metric,
)
from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    evaluate_lagged_scores,
    extract_lagged_weight_score,
    fit_tuned_lagged_linear_baseline,
    normalize_lagged_score,
)
from ssm_causal.lag_synthetic import (
    ensure_stable_lag_coefficients,
    simulate_lagged_var,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ═══════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════


def _generate_random_var_raw(
    K,
    T,
    density=0.2,
    max_lag=3,
    seed=42,
    noise_std=0.1,
    burnin=300,
):
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
    X = simulate_lagged_var(
        A_lags,
        T=T,
        noise_std=noise_std,
        seed=seed,
        burnin=burnin,
        standardize_output=False,
    )
    return X, G_lag, [A_lags[t] for t in range(max_lag)]


# ═══════════════════════════════════════════
# Evaluators
# ═══════════════════════════════════════════


def eval_bottleneck(
    X,
    G_lag,
    max_lag,
    d_bottleneck,
    epochs,
    device,
    segment_lengths=None,
    model_seed=42,
    target_valid_mask=None,
    evaluation_context=None,
):
    K = X.shape[1]
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    usable_rows = (
        len(X) - max_lag
        if segment_lengths is None
        else sum(length - max_lag for length in segment_lengths)
    )
    config = LaggedBottleneckTrainingConfig(
        n_epochs=epochs,
        batch_size=min(64, usable_rows),
    )
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(
        X,
        segment_lengths=segment_lengths,
        target_valid_mask=target_valid_mask,
    )
    method_context = "Bottleneck"
    if evaluation_context is not None:
        method_context += f" ({evaluation_context})"
    S_lag = require_finite_array(
        extract_lagged_weight_score(model),
        context=method_context,
    )
    S_lag = normalize_lagged_score(S_lag, mode="global")
    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return require_finite_metric(
        metrics,
        "flat_lag_auroc",
        context=method_context,
    )


def eval_baseline(
    X,
    G_lag,
    max_lag,
    method,
    segment_lengths=None,
    model_seed=None,
    target_valid_mask=None,
    evaluation_context=None,
):
    del model_seed
    kwargs = dict(
        X_train=X,
        max_lag=max_lag,
        method=method,
        segment_lengths=segment_lengths,
        target_valid_mask=target_valid_mask,
    )
    if method in ("lasso", "ridge"):
        kwargs["alpha_values"] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    if method == "rrr":
        kwargs["alpha_values"] = [1e-3, 1e-2, 1e-1]
        kwargs["rank_values"] = [2, 4, 8, 16]
    result = fit_tuned_lagged_linear_baseline(**kwargs)
    method_context = method
    if evaluation_context is not None:
        method_context += f" ({evaluation_context})"
    S_lag = require_finite_array(result.score, context=method_context)
    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return require_finite_metric(
        metrics,
        "flat_lag_auroc",
        context=method_context,
    )


def _require_finite_score(value, *, method, arm, config, seed):
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(
            f"{method} returned a non-finite score for arm={arm}, config={config}, seed={seed}"
        )
    return score


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
    noise_std=0.1,
    burnin=300,
    scale_floor=1e-8,
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
    if not int_types or not int_strengths:
        raise ValueError("int_types and int_strengths must be non-empty.")
    if "rrr" in baselines:
        raise ValueError("RRR does not support intervention target masks; use lasso and/or ridge.")

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
            model_seed = seed * 42 + 7
            data_seed = 100 + seed
            # Fixed DGP per seed
            X_small_raw, G_lag, A_lags = _generate_random_var_raw(
                K=K,
                T=T,
                density=density,
                max_lag=max_lag,
                seed=data_seed,
                noise_std=noise_std,
                burnin=burnin,
            )
            small_series = SegmentedSeries(X_small_raw, (T,))

            def make_arms(current_type, current_strength):
                raw_arms = build_raw_intervention_arms(
                    np.stack(A_lags, axis=0),
                    small_observational=small_series,
                    T_int_per_target=T_int_per_target,
                    intervention_type=current_type,
                    strength=current_strength,
                    root_seed=200 + seed,
                    noise_std=noise_std,
                    burnin=burnin,
                )
                return preprocess_intervention_arms(
                    raw_arms,
                    scale_floor=scale_floor,
                )[0]

            for int_type in int_types:
                control_arms = make_arms(int_type, int_strengths[0])
                X_small = control_arms.small.values.copy()
                X_big = control_arms.big.values.copy()
                small_lengths = control_arms.small.segment_lengths
                big_lengths = control_arms.big.segment_lengths
                small_target_mask = control_arms.target_valid_mask("small")
                big_target_mask = control_arms.target_valid_mask("big")

                # S/B share preprocessing within an intervention type, not across types.
                bn_S = eval_bottleneck(
                    X_small,
                    G_lag,
                    max_lag,
                    d_bottleneck,
                    epochs,
                    device,
                    segment_lengths=small_lengths,
                    model_seed=model_seed,
                    target_valid_mask=small_target_mask,
                    evaluation_context=(f"arm=small, K={K}, type={int_type}, seed={seed}"),
                )
                bn_B = eval_bottleneck(
                    X_big,
                    G_lag,
                    max_lag,
                    d_bottleneck,
                    epochs,
                    device,
                    segment_lengths=big_lengths,
                    model_seed=model_seed,
                    target_valid_mask=big_target_mask,
                    evaluation_context=(f"arm=big, K={K}, type={int_type}, seed={seed}"),
                )
                bl_S = {
                    baseline: eval_baseline(
                        X_small,
                        G_lag,
                        max_lag,
                        baseline,
                        segment_lengths=small_lengths,
                        model_seed=model_seed,
                        target_valid_mask=small_target_mask,
                        evaluation_context=(f"arm=small, K={K}, type={int_type}, seed={seed}"),
                    )
                    for baseline in baselines
                }
                bl_B = {
                    baseline: eval_baseline(
                        X_big,
                        G_lag,
                        max_lag,
                        baseline,
                        segment_lengths=big_lengths,
                        model_seed=model_seed,
                        target_valid_mask=big_target_mask,
                        evaluation_context=(f"arm=big, K={K}, type={int_type}, seed={seed}"),
                    )
                    for baseline in baselines
                }

                for strength in int_strengths:
                    arms = (
                        control_arms
                        if strength == int_strengths[0]
                        else make_arms(int_type, strength)
                    )
                    X_comb = arms.combined.values.copy()
                    combined_lengths = arms.combined.segment_lengths
                    combined_target_mask = arms.target_valid_mask("combined")

                    bn_C = eval_bottleneck(
                        X_comb,
                        G_lag,
                        max_lag,
                        d_bottleneck,
                        epochs,
                        device,
                        segment_lengths=combined_lengths,
                        model_seed=model_seed,
                        target_valid_mask=combined_target_mask,
                        evaluation_context=(
                            f"arm=combined, K={K}, type={int_type}, "
                            f"strength={strength}, seed={seed}"
                        ),
                    )
                    bl_C = {
                        baseline: eval_baseline(
                            X_comb,
                            G_lag,
                            max_lag,
                            baseline,
                            segment_lengths=combined_lengths,
                            model_seed=model_seed,
                            target_valid_mask=combined_target_mask,
                            evaluation_context=(
                                f"arm=combined, K={K}, type={int_type}, "
                                f"strength={strength}, seed={seed}"
                            ),
                        )
                        for baseline in baselines
                    }
                    protocol_metadata = arms.metadata(max_lag)
                    protocol_metadata["data_seed"] = data_seed
                    config_context = f"K={K}, type={int_type}, strength={strength}"
                    bn_S = _require_finite_score(
                        bn_S,
                        method="Bottleneck",
                        arm="small",
                        config=config_context,
                        seed=seed,
                    )
                    bn_B = _require_finite_score(
                        bn_B,
                        method="Bottleneck",
                        arm="big",
                        config=config_context,
                        seed=seed,
                    )
                    bn_C = _require_finite_score(
                        bn_C,
                        method="Bottleneck",
                        arm="combined",
                        config=config_context,
                        seed=seed,
                    )
                    for baseline in baselines:
                        bl_S[baseline] = _require_finite_score(
                            bl_S[baseline],
                            method=baseline,
                            arm="small",
                            config=config_context,
                            seed=seed,
                        )
                        bl_B[baseline] = _require_finite_score(
                            bl_B[baseline],
                            method=baseline,
                            arm="big",
                            config=config_context,
                            seed=seed,
                        )
                        bl_C[baseline] = _require_finite_score(
                            bl_C[baseline],
                            method=baseline,
                            arm="combined",
                            config=config_context,
                            seed=seed,
                        )

                    for b in baselines:
                        rec = {
                            "seed": seed,
                            "model_seed": model_seed,
                            "segment_lengths": {
                                "small": list(small_lengths),
                                "big": list(big_lengths),
                                "combined": list(combined_lengths),
                            },
                            "protocol": protocol_metadata,
                            "bn_S": bn_S,
                            "bn_B": bn_B,
                            "bn_C": bn_C,
                            "bl_S": bl_S[b],
                            "bl_B": bl_B[b],
                            "bl_C": bl_C[b],
                            "bn_int_specific": bn_C - bn_B,
                            "bl_int_specific": bl_C[b] - bl_B[b],
                            "gap_int_specific": (bn_C - bn_B) - (bl_C[b] - bl_B[b]),
                            "gap_exp21_style": (bn_C - bn_S) - (bl_C[b] - bl_S[b]),
                        }
                        results[K][int_type][strength][b].append(rec)

                print(
                    f"  seed={seed:2d} type={int_type:>8s} "
                    f"bn_S={bn_S:.3f} bn_B={bn_B:.3f}  "
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
        f"{'gap_mean':>10} {'gap_med':>8} {'win_rt':>7} {'holm_p':>9} "
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
                    win_rate = float(np.mean(gaps_is > 0.01))
                    inference = paired_effect_inference(
                        gaps_is,
                        np.zeros_like(gaps_is),
                        bootstrap_seed=(
                            10_000
                            + K * 100
                            + int_types.index(int_type) * 10
                            + int_strengths.index(strength)
                        ),
                    )
                    row = {
                        "K": K,
                        "int_type": int_type,
                        "strength": strength,
                        "baseline": b,
                        "gap_int_specific_mean": float(gaps_is.mean()),
                        "gap_int_specific_median": float(np.median(gaps_is)),
                        "gap_int_specific_wins_rate": win_rate,
                        "primary_inference": inference,
                        "gap_exp21_style_mean": float(gaps_e21.mean()),
                        "n_seeds": len(recs),
                        "per_seed": recs,
                    }
                    flat.append(row)
    primary_family = "intervention_specific_gap_across_sweep"
    apply_holm_family(
        [row["primary_inference"] for row in flat],
        family_name=primary_family,
    )
    for row in flat:
        inference = row["primary_inference"]
        adjusted_p = inference["holm_adjusted_p"]
        p_text = "null" if adjusted_p is None else f"{adjusted_p:.1e}"
        print(
            f"{row['K']:>3} {row['int_type']:>9} "
            f"{row['strength']:>8.2f} {row['baseline']:>9} "
            f"{row['gap_int_specific_mean']:>+10.3f} "
            f"{row['gap_int_specific_median']:>+8.3f} "
            f"{row['gap_int_specific_wins_rate']:>7.2f} "
            f"{p_text:>9} {row['gap_exp21_style_mean']:>+10.3f}"
        )

    # Survivability summary
    sig_rows = [
        r for r in flat if r["primary_inference"]["reject_holm"] and r["gap_int_specific_mean"] > 0
    ]
    print(
        f"\nCONFIGS where BN beats baseline after Holm adjustment (p<0.05): "
        f"{len(sig_rows)}/{len(flat)}"
    )

    # Per baseline
    print("\nBy baseline:")
    for b in baselines:
        brows = [r for r in flat if r["baseline"] == b]
        sig = [
            r
            for r in brows
            if r["primary_inference"]["reject_holm"] and r["gap_int_specific_mean"] > 0
        ]
        mean_gap = np.mean([r["gap_int_specific_mean"] for r in brows])
        print(f"  {b:>8s}: sig configs {len(sig)}/{len(brows)}  mean_gap={mean_gap:+.3f}")

    # Per intervention type
    print("\nBy intervention type:")
    for it in int_types:
        trows = [r for r in flat if r["int_type"] == it]
        sig = [
            r
            for r in trows
            if r["primary_inference"]["reject_holm"] and r["gap_int_specific_mean"] > 0
        ]
        mean_gap = np.mean([r["gap_int_specific_mean"] for r in trows])
        print(f"  {it:>9s}: sig configs {len(sig)}/{len(trows)}  mean_gap={mean_gap:+.3f}")

    runtime = (time.time() - total_start) / 60
    print(f"\nRuntime: {runtime:.1f} min")

    out_path = Path(save_dir) / "intervention_sweep_results.json"
    serialized = []
    for r in flat:
        r2 = {k: v for k, v in r.items() if k != "per_seed"}
        r2["per_seed"] = [
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in rec.items()}
            for rec in r["per_seed"]
        ]
        serialized.append(r2)
    payload = {
        "metadata": {
            "experiment": "exp23_intervention_sweep",
            "generation": {
                "n_seeds": n_seeds,
                "replicate_indices": list(range(n_seeds)),
                "data_seeds": [100 + seed for seed in range(n_seeds)],
                "protocol_root_seeds": [200 + seed for seed in range(n_seeds)],
                "K_values": list(K_values),
                "T_values": [T],
                "density": float(density),
                "intervention_types": list(int_types),
                "intervention_strengths": [float(strength) for strength in int_strengths],
                "T_int_per_target": T_int_per_target,
                "max_lag": max_lag,
                "noise_std": float(noise_std),
                "burnin": burnin,
                "scale_floor": float(scale_floor),
            },
            "training": {
                "model_family": "LaggedBottleneckModel",
                "d_bottleneck": d_bottleneck,
                "epochs": epochs,
                "batch_size": "min(64, usable_rows)",
                "optimizer": "AdamW",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "model_seeds": [seed * 42 + 7 for seed in range(n_seeds)],
                "device": device,
            },
            "evaluation": {
                "methods": ["Bottleneck", *list(baselines)],
                "metric": "flat_lag_auroc",
                "estimand": "intervention_target_aware",
                "baselines": list(baselines),
                "validation_fraction": 0.25,
                "alpha_values": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            },
            "statistics": {
                "primary_family": primary_family,
                "multiple_testing": "Holm",
                "alpha": 0.05,
                "paired_test": "Wilcoxon greater",
                "effect": "paired_mean_difference",
                "bootstrap_confidence": 0.95,
                "bootstrap_resamples": 10_000,
            },
        },
        "results": serialized,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"Saved to {out_path}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--K", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--int-types", nargs="+", default=["random", "constant", "soft"])
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--T-int-per-target", type=int, default=50)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--d-bottleneck", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--baselines", nargs="+", default=["lasso", "ridge"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/intervention_sweep")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--scale-floor", type=float, default=1e-8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_sweep(
            n_seeds=3,
            K_values=(20,),
            T=300,
            density=0.2,
            int_types=("random", "constant"),
            int_strengths=(1.0, 2.0),
            T_int_per_target=50,
            max_lag=3,
            d_bottleneck=16,
            epochs=30,
            baselines=("lasso", "ridge"),
            device=args.device,
            save_dir=args.save_dir,
            noise_std=args.noise_std,
            burnin=args.burnin,
            scale_floor=args.scale_floor,
        )
    else:
        run_sweep(
            n_seeds=args.seeds,
            K_values=tuple(args.K),
            T=args.T,
            density=args.density,
            int_types=tuple(args.int_types),
            int_strengths=tuple(args.strengths),
            T_int_per_target=args.T_int_per_target,
            max_lag=args.max_lag,
            d_bottleneck=args.d_bottleneck,
            epochs=args.epochs,
            baselines=tuple(args.baselines),
            device=args.device,
            save_dir=args.save_dir,
            noise_std=args.noise_std,
            burnin=args.burnin,
            scale_floor=args.scale_floor,
        )
