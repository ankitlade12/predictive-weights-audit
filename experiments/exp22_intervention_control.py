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

The primary null is that, after matching sample size, the bottleneck has no
relative difference-in-differences advantage over Lasso. This comparison does
not test whether intervention data directly helps either method.

We test both bottleneck and Lasso on:
    A: X_obs_small            (T samples)
    B: X_obs_big              (T + K*T_int_per_target samples, all obs)
    C: X_combined             (T obs + K*T_int_per_target intervention)

The primary estimand is the relative difference-in-differences
    (bn_combined - bn_obs_big) - (bl_combined - bl_obs_big)

A positive Holm-adjusted result indicates a bottleneck advantage relative to
Lasso for this contrast. It does not establish that interventions directly
improve either method in isolation.

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

# ═══════════════════════════════════════════
# Data generation — mirror Exp 21 scheme exactly so the comparison is fair
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
    from ssm_causal.lag_synthetic import (
        ensure_stable_lag_coefficients,
        simulate_lagged_var,
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
    max_lag=3,
    d_bottleneck=16,
    epochs=80,
    device="cpu",
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


def eval_lasso(
    X,
    G_lag,
    max_lag=3,
    segment_lengths=None,
    model_seed=None,
    target_valid_mask=None,
    evaluation_context=None,
):
    del model_seed
    result = fit_tuned_lagged_linear_baseline(
        X,
        max_lag=max_lag,
        method="lasso",
        alpha_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        segment_lengths=segment_lengths,
        target_valid_mask=target_valid_mask,
    )
    method_context = "Lasso"
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


def _relative_advantage_verdict(result):
    inference = result["primary_inference"]
    if (
        inference["status"] != "ok"
        or inference["holm_adjusted_p"] is None
        or inference["reject_holm"] is None
    ):
        return "INDETERMINATE"
    if result["mean_relative_effect"] > 0 and inference["reject_holm"] is True:
        return "RELATIVE_ADVANTAGE"
    return "NO_RELATIVE_ADVANTAGE"


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
    noise_std=0.1,
    burnin=300,
    scale_floor=1e-8,
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
                model_seed = seed * 42 + 7
                data_seed = 100 + seed

                X_obs_small_raw, G_lag, A_lags = _generate_random_var_raw(
                    K=K,
                    T=T,
                    density=density,
                    max_lag=max_lag,
                    seed=data_seed,
                    noise_std=noise_std,
                    burnin=burnin,
                )
                raw_arms = build_raw_intervention_arms(
                    np.stack(A_lags, axis=0),
                    small_observational=SegmentedSeries(
                        X_obs_small_raw,
                        (T,),
                    ),
                    T_int_per_target=T_int_per_target,
                    intervention_type="random",
                    strength=int_strength,
                    root_seed=200 + seed,
                    noise_std=noise_std,
                    burnin=burnin,
                )
                arms, _ = preprocess_intervention_arms(
                    raw_arms,
                    scale_floor=scale_floor,
                )
                X_obs_small = arms.small.values.copy()
                X_obs_big = arms.big.values.copy()
                X_combined = arms.combined.values.copy()
                segment_lengths = {
                    "small": arms.small.segment_lengths,
                    "big": arms.big.segment_lengths,
                    "combined": arms.combined.segment_lengths,
                }
                target_valid_masks = {
                    name: arms.target_valid_mask(name) for name in ("small", "big", "combined")
                }
                protocol_metadata = arms.metadata(max_lag)
                protocol_metadata["data_seed"] = data_seed

                # Evaluate all three arms for both methods
                bn_small = eval_bottleneck(
                    X_obs_small,
                    G_lag,
                    max_lag,
                    d_bottleneck,
                    epochs,
                    device,
                    segment_lengths=segment_lengths["small"],
                    model_seed=model_seed,
                    target_valid_mask=target_valid_masks["small"],
                    evaluation_context=(f"arm=small, config={config_name}, seed={seed}"),
                )
                bn_big = eval_bottleneck(
                    X_obs_big,
                    G_lag,
                    max_lag,
                    d_bottleneck,
                    epochs,
                    device,
                    segment_lengths=segment_lengths["big"],
                    model_seed=model_seed,
                    target_valid_mask=target_valid_masks["big"],
                    evaluation_context=(f"arm=big, config={config_name}, seed={seed}"),
                )
                bn_comb = eval_bottleneck(
                    X_combined,
                    G_lag,
                    max_lag,
                    d_bottleneck,
                    epochs,
                    device,
                    segment_lengths=segment_lengths["combined"],
                    model_seed=model_seed,
                    target_valid_mask=target_valid_masks["combined"],
                    evaluation_context=(f"arm=combined, config={config_name}, seed={seed}"),
                )

                bl_small = eval_lasso(
                    X_obs_small,
                    G_lag,
                    max_lag,
                    segment_lengths=segment_lengths["small"],
                    model_seed=model_seed,
                    target_valid_mask=target_valid_masks["small"],
                    evaluation_context=(f"arm=small, config={config_name}, seed={seed}"),
                )
                bl_big = eval_lasso(
                    X_obs_big,
                    G_lag,
                    max_lag,
                    segment_lengths=segment_lengths["big"],
                    model_seed=model_seed,
                    target_valid_mask=target_valid_masks["big"],
                    evaluation_context=(f"arm=big, config={config_name}, seed={seed}"),
                )
                bl_comb = eval_lasso(
                    X_combined,
                    G_lag,
                    max_lag,
                    segment_lengths=segment_lengths["combined"],
                    model_seed=model_seed,
                    target_valid_mask=target_valid_masks["combined"],
                    evaluation_context=(f"arm=combined, config={config_name}, seed={seed}"),
                )
                score_values = {
                    "bn_small": bn_small,
                    "bn_big": bn_big,
                    "bn_combined": bn_comb,
                    "lasso_small": bl_small,
                    "lasso_big": bl_big,
                    "lasso_combined": bl_comb,
                }
                for score_name, score_value in score_values.items():
                    method, arm = score_name.split("_", maxsplit=1)
                    score_values[score_name] = _require_finite_score(
                        score_value,
                        method=method,
                        arm=arm,
                        config=config_name,
                        seed=seed,
                    )
                bn_small = score_values["bn_small"]
                bn_big = score_values["bn_big"]
                bn_comb = score_values["bn_combined"]
                bl_small = score_values["lasso_small"]
                bl_big = score_values["lasso_big"]
                bl_comb = score_values["lasso_combined"]

                # Key metric: intervention-specific benefit (combined vs size-matched obs)
                bn_int_specific = bn_comb - bn_big
                bl_int_specific = bl_comb - bl_big

                # Exp 21 metric: combined vs small (confounded with data size)
                bn_exp21 = bn_comb - bn_small
                bl_exp21 = bl_comb - bl_small

                # Pure data-size effect
                bn_data_only = bn_big - bn_small
                bl_data_only = bl_big - bl_small

                per_seed.append(
                    {
                        "seed": seed,
                        "model_seed": model_seed,
                        "segment_lengths": {
                            name: list(lengths) for name, lengths in segment_lengths.items()
                        },
                        "protocol": protocol_metadata,
                        "bn_small": bn_small,
                        "bn_big": bn_big,
                        "bn_comb": bn_comb,
                        "bl_small": bl_small,
                        "bl_big": bl_big,
                        "bl_comb": bl_comb,
                        "bn_int_specific": bn_int_specific,
                        "bl_int_specific": bl_int_specific,
                        "bn_exp21_style": bn_exp21,
                        "bl_exp21_style": bl_exp21,
                        "bn_data_only": bn_data_only,
                        "bl_data_only": bl_data_only,
                    }
                )

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

            primary_inference = paired_effect_inference(
                bn_is,
                bl_is,
                bootstrap_seed=10_000 + K * 100 + T,
            )
            exp21_inference = paired_effect_inference(
                bn_e21,
                bl_e21,
                bootstrap_seed=20_000 + K * 100 + T,
            )

            result = {
                "K": K,
                "T": T,
                "T_big": T_big,
                "T_int_total": T_int_total,
                "n_seeds": n_seeds,
                "mean_bn_int_specific": float(bn_is.mean()),
                "mean_bl_int_specific": float(bl_is.mean()),
                "mean_relative_effect": float((bn_is - bl_is).mean()),
                "bn_int_specific_wins_rate": float(np.mean((bn_is - bl_is) > 0.01)),
                "mean_bn_exp21_style": float(bn_e21.mean()),
                "mean_bl_exp21_style": float(bl_e21.mean()),
                "mean_bn_data_only": float(bn_do.mean()),
                "mean_bl_data_only": float(bl_do.mean()),
                "primary_inference": primary_inference,
                "secondary_exp21_inference": exp21_inference,
                "per_seed": [
                    {
                        k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in r.items()
                    }
                    for r in per_seed
                ],
            }
            all_results[config_name] = result

            print(
                f"\n  [EXP21 STYLE]  BN={bn_e21.mean():+.3f}  BL={bl_e21.mean():+.3f}  "
                f"status={exp21_inference['status']}"
            )
            print(
                f"  [SIZE-MATCHED] BN={bn_is.mean():+.3f}  BL={bl_is.mean():+.3f}  "
                f"status={primary_inference['status']}"
            )
            print(
                f"  [DATA-ONLY]    BN={bn_do.mean():+.3f}  BL={bl_do.mean():+.3f}  "
                f"(extra obs-only data benefit)"
            )

    primary_family = "size_matched_bottleneck_vs_lasso"
    apply_holm_family(
        [result["primary_inference"] for result in all_results.values()],
        family_name=primary_family,
    )

    # ── Verdict ──
    print(f"\n{'=' * 72}")
    print("RELATIVE DIFFERENCE-IN-DIFFERENCES VERDICT")
    print(f"{'=' * 72}")

    relative_advantage_count = 0
    indeterminate_configs = []
    for config, r in all_results.items():
        inference = r["primary_inference"]
        adjusted_p = inference["holm_adjusted_p"]
        verdict = _relative_advantage_verdict(r)
        if verdict == "INDETERMINATE":
            indeterminate_configs.append(config)
        r["verdict"] = verdict
        p_text = "null" if adjusted_p is None else f"{adjusted_p:.1e}"
        print(f"  {config:22s} | size-matched Holm p={p_text} ({verdict})")
        if verdict == "RELATIVE_ADVANTAGE":
            relative_advantage_count += 1

    total = len(all_results)
    print(f"\n  Positive adjusted relative advantages: {relative_advantage_count}/{total}")
    if indeterminate_configs:
        overall_verdict = "INDETERMINATE"
        print("  → INDETERMINATE: one or more primary comparisons were not estimable.")
    elif relative_advantage_count >= total * 0.5:
        overall_verdict = "RELATIVE_ADVANTAGE"
        print(
            "  → Bottleneck has a positive adjusted relative advantage "
            "over Lasso for the primary contrast."
        )
    else:
        overall_verdict = "NO_RELATIVE_ADVANTAGE"
        print(
            "  → No positive adjusted bottleneck advantage relative to "
            "Lasso for the primary contrast."
        )

    print(f"\n  Runtime: {(time.time() - total_start) / 60:.1f} min")
    summary = {
        "overall_verdict": overall_verdict,
        "relative_advantage_configs": [
            config
            for config, result in all_results.items()
            if result["verdict"] == "RELATIVE_ADVANTAGE"
        ],
        "no_relative_advantage_configs": [
            config
            for config, result in all_results.items()
            if result["verdict"] == "NO_RELATIVE_ADVANTAGE"
        ],
        "indeterminate_configs": indeterminate_configs,
    }

    out_path = Path(save_dir) / "intervention_control_results.json"
    payload = {
        "metadata": {
            "experiment": "exp22_intervention_control",
            "generation": {
                "n_seeds": n_seeds,
                "replicate_indices": list(range(n_seeds)),
                "data_seeds": [100 + seed for seed in range(n_seeds)],
                "protocol_root_seeds": [200 + seed for seed in range(n_seeds)],
                "K_values": list(K_values),
                "T_values": list(T_values),
                "density": float(density),
                "intervention_types": ["random"],
                "intervention_strengths": [float(int_strength)],
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
                "methods": ["Bottleneck", "Lasso"],
                "metric": "flat_lag_auroc",
                "estimand": "intervention_target_aware",
                "lasso_alpha_values": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            },
            "statistics": {
                "primary_family": primary_family,
                "primary_estimand": (
                    "(bottleneck_combined-bottleneck_big)-(lasso_combined-lasso_big)"
                ),
                "interpretation": ("relative_method_advantage_not_direct_intervention_utility"),
                "verdict_labels": [
                    "RELATIVE_ADVANTAGE",
                    "NO_RELATIVE_ADVANTAGE",
                    "INDETERMINATE",
                ],
                "multiple_testing": "Holm",
                "alpha": 0.05,
                "paired_test": "Wilcoxon greater",
                "effect": "paired_mean_difference",
                "bootstrap_confidence": 0.95,
                "bootstrap_resamples": 10_000,
            },
        },
        "summary": summary,
        "results": all_results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"  Results saved to {out_path}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--K", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--T", type=int, nargs="+", default=[300])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/intervention_control")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--scale-floor", type=float, default=1e-8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_control(
            n_seeds=3,
            K_values=(10, 20),
            T_values=(300,),
            epochs=30,
            device=args.device,
            save_dir=args.save_dir,
            noise_std=args.noise_std,
            burnin=args.burnin,
            scale_floor=args.scale_floor,
        )
    else:
        run_control(
            n_seeds=args.seeds,
            K_values=tuple(args.K),
            T_values=tuple(args.T),
            epochs=args.epochs,
            device=args.device,
            save_dir=args.save_dir,
            noise_std=args.noise_std,
            burnin=args.burnin,
            scale_floor=args.scale_floor,
        )
