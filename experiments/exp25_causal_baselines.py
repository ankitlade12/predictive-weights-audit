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
the bottleneck? The primary tests quantify each method's gain. A separate
exploratory paired contrast tests whether Granger's gain exceeds the
bottleneck's rather than inferring equivalence from parallel significance.

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
from ssm_causal.utils import compute_metrics

# ═══════════════════════════════════════════
# Data generation (mirrors Exp 22)
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
# Evaluators (reference + new causal baselines)
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
        "aggregate_auroc",
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
        "aggregate_auroc",
        context=method_context,
    )


def eval_pcmci(
    X,
    G_lag,
    max_lag=3,
    segment_lengths=None,
    model_seed=None,
    target_valid_mask=None,
    evaluation_context=None,
):
    """Evaluate PCMCI on the aggregate edge graph used by every Exp25 method."""
    del model_seed
    from baselines.pcmci_wrapper import pcmci_with_details

    kwargs = {}
    if segment_lengths is not None:
        kwargs["segment_lengths"] = segment_lengths
    if target_valid_mask is not None:
        kwargs["target_valid_mask"] = target_valid_mask
    details = pcmci_with_details(
        X,
        tau_max=max_lag,
        pc_alpha=0.05,
        **kwargs,
    )
    method_context = "PCMCI"
    if evaluation_context is not None:
        method_context += f" ({evaluation_context})"
    val = details.get("val_matrix")  # Tigramite: (source, target, lag)
    if val is None:
        from baselines.pcmci_wrapper import pcmci_causality_matrix

        S_agg = pcmci_causality_matrix(
            X,
            tau_max=max_lag,
            **kwargs,
        )
        return _evaluate_aggregate_score(
            S_agg,
            G_lag,
            context=method_context,
        )
    else:
        # Convert Tigramite [source, target] to project [target, source].
        val = np.abs(require_finite_array(val, context=method_context))
        S_lag = np.zeros((max_lag, val.shape[0], val.shape[1]))
        for tau in range(max_lag):
            if tau + 1 < val.shape[2]:
                S_lag[tau] = val[:, :, tau + 1].T
        for tau in range(max_lag):
            np.fill_diagonal(S_lag[tau], 0)
        global_max = S_lag.max()
        if global_max > 0:
            S_lag = S_lag / global_max

    metrics = evaluate_lagged_scores(S_lag, G_lag)
    return require_finite_metric(
        metrics,
        "aggregate_auroc",
        context=method_context,
    )


def eval_granger(
    X,
    G_lag,
    max_lag=3,
    segment_lengths=None,
    model_seed=None,
    target_valid_mask=None,
    evaluation_context=None,
):
    """Evaluate aggregate Granger strength against aggregate ground truth."""
    del model_seed
    from baselines.granger import granger_causality_matrix

    S_agg = granger_causality_matrix(
        X,
        max_lag=max_lag,
        segment_lengths=segment_lengths,
        target_valid_mask=target_valid_mask,
    )
    method_context = "Granger"
    if evaluation_context is not None:
        method_context += f" ({evaluation_context})"
    return _evaluate_aggregate_score(
        S_agg,
        G_lag,
        context=method_context,
    )


def _evaluate_aggregate_score(S_agg, G_lag, *, context):
    score = require_finite_array(S_agg, context=context).astype(float, copy=True)
    np.fill_diagonal(score, 0.0)
    graph = (np.asarray(G_lag).sum(axis=0) > 0).astype(float)
    np.fill_diagonal(graph, 0.0)
    metrics = compute_metrics(score, graph)
    return require_finite_metric(metrics, "auroc", context=context)


def _require_finite_score(value, *, method, arm, config, seed):
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(
            f"{method} returned a non-finite score for arm={arm}, config={config}, seed={seed}"
        )
    return score


def eval_pcmci_plus(
    X,
    G_lag,
    max_lag=3,
    segment_lengths=None,
    model_seed=None,
    target_valid_mask=None,
    evaluation_context=None,
):
    del model_seed
    from baselines.pcmci_wrapper import pcmci_plus_causality_matrix

    S_agg = pcmci_plus_causality_matrix(
        X,
        tau_max=max_lag,
        pc_alpha=0.05,
        segment_lengths=segment_lengths,
        target_valid_mask=target_valid_mask,
    )
    method_context = "PCMCI+"
    if evaluation_context is not None:
        method_context += f" ({evaluation_context})"
    return _evaluate_aggregate_score(
        S_agg,
        G_lag,
        context=method_context,
    )


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
    include_modern_baselines=False,
    noise_std=0.1,
    burnin=300,
    scale_floor=1e-8,
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
        "Bottleneck": lambda X, G, lengths, seed, mask, context: eval_bottleneck(
            X,
            G,
            max_lag,
            d_bottleneck,
            epochs,
            device,
            segment_lengths=lengths,
            model_seed=seed,
            target_valid_mask=mask,
            evaluation_context=context,
        ),
        "Lasso": lambda X, G, lengths, seed, mask, context: eval_lasso(
            X,
            G,
            max_lag,
            segment_lengths=lengths,
            model_seed=seed,
            target_valid_mask=mask,
            evaluation_context=context,
        ),
        "PCMCI": lambda X, G, lengths, seed, mask, context: eval_pcmci(
            X,
            G,
            max_lag,
            segment_lengths=lengths,
            model_seed=seed,
            target_valid_mask=mask,
            evaluation_context=context,
        ),
        "Granger": lambda X, G, lengths, seed, mask, context: eval_granger(
            X,
            G,
            max_lag,
            segment_lengths=lengths,
            model_seed=seed,
            target_valid_mask=mask,
            evaluation_context=context,
        ),
    }
    if include_modern_baselines:
        methods["PCMCI+"] = lambda X, G, lengths, seed, mask, context: eval_pcmci_plus(
            X,
            G,
            max_lag,
            segment_lengths=lengths,
            model_seed=seed,
            target_valid_mask=mask,
            evaluation_context=context,
        )
        print("Skipping DYNOTEARS: it does not support multiple independent segments.")
        print("Skipping VAR-LiNGAM: it does not support multiple independent segments.")

    all_results = {}

    for K in K_values:
        for T in T_values:
            cfg = f"K={K}_T={T}"
            print(f"\n{'─' * 60}\nConfig: {cfg}\n{'─' * 60}")

            # Per-method arrays
            scores = {m: {"small": [], "big": [], "combined": []} for m in methods}
            per_seed_protocol = []

            for seed in range(n_seeds):
                model_seed = seed * 42 + 7
                data_seed = 100 + seed

                # Generate coefficient matrices + small observational draw
                X_small_raw, G_lag, A_lags = _generate_random_var_raw(
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
                    small_observational=SegmentedSeries(X_small_raw, (T,)),
                    T_int_per_target=T_int_per_target,
                    intervention_type="random",
                    strength=intervention_strength,
                    root_seed=200 + seed,
                    noise_std=noise_std,
                    burnin=burnin,
                )
                arms, _ = preprocess_intervention_arms(
                    raw_arms,
                    scale_floor=scale_floor,
                )
                X_small = arms.small.values.copy()
                X_big = arms.big.values.copy()
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
                per_seed_protocol.append(
                    {
                        "seed": seed,
                        "model_seed": model_seed,
                        "segment_lengths": {
                            name: list(lengths) for name, lengths in segment_lengths.items()
                        },
                        "protocol": protocol_metadata,
                    }
                )

                for m_name, m_fn in methods.items():
                    arm_values = {
                        "small": X_small,
                        "big": X_big,
                        "combined": X_combined,
                    }
                    arm_scores = {}
                    for arm_name, values in arm_values.items():
                        try:
                            evaluated = m_fn(
                                values,
                                G_lag,
                                segment_lengths[arm_name],
                                model_seed,
                                target_valid_masks[arm_name],
                                (f"arm={arm_name}, config={cfg}, seed={seed}"),
                            )
                            arm_scores[arm_name] = _require_finite_score(
                                evaluated,
                                method=m_name,
                                arm=arm_name,
                                config=cfg,
                                seed=seed,
                            )
                        except Exception as error:
                            raise RuntimeError(
                                f"{m_name} failed for K={K}, T={T}, seed={seed}, arm={arm_name}"
                            ) from error
                    s_small = arm_scores["small"]
                    s_big = arm_scores["big"]
                    s_comb = arm_scores["combined"]

                    scores[m_name]["small"].append(s_small)
                    scores[m_name]["big"].append(s_big)
                    scores[m_name]["combined"].append(s_comb)

                print(
                    f"  seed={seed:2d} | "
                    + " | ".join(
                        f"{m}: comb={scores[m]['combined'][-1]:.3f} big={scores[m]['big'][-1]:.3f}"
                        for m in ("Bottleneck", "Lasso", "PCMCI", "Granger")
                    )
                )

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

                inference = paired_effect_inference(
                    gap,
                    np.zeros_like(gap),
                    bootstrap_seed=10_000 + K * 100 + T,
                )

                cfg_result[m] = {
                    "small_mean": float(small_arr.mean()),
                    "big_mean": float(big_arr.mean()),
                    "combined_mean": float(comb_arr.mean()),
                    "raw_gain_mean": float(raw_gain.mean()),
                    "size_matched_gap_mean": float(gap.mean()),
                    "primary_inference": inference,
                    "small_all": small_arr.tolist(),
                    "big_all": big_arr.tolist(),
                    "combined_all": comb_arr.tolist(),
                    "per_seed": [
                        {
                            **metadata,
                            "small": float(scores[m]["small"][index]),
                            "big": float(scores[m]["big"][index]),
                            "combined": float(scores[m]["combined"][index]),
                        }
                        for index, metadata in enumerate(per_seed_protocol)
                    ],
                }

            all_results[cfg] = cfg_result

    primary_family = "size_matched_gap_across_methods_and_configs"
    apply_holm_family(
        [
            method_result["primary_inference"]
            for config_result in all_results.values()
            for method_result in config_result.values()
        ],
        family_name=primary_family,
    )

    contrast_family = "granger_minus_bottleneck_across_configs"
    method_contrasts = {"Granger-minus-Bottleneck": {}}
    for contrast_index, (cfg, config_result) in enumerate(all_results.items()):
        bottleneck = config_result["Bottleneck"]
        granger = config_result["Granger"]
        bottleneck_gap = np.asarray(bottleneck["combined_all"]) - np.asarray(bottleneck["big_all"])
        granger_gap = np.asarray(granger["combined_all"]) - np.asarray(granger["big_all"])
        difference = granger_gap - bottleneck_gap
        inference = paired_effect_inference(
            granger_gap,
            bottleneck_gap,
            bootstrap_seed=20_000 + contrast_index,
        )
        method_contrasts["Granger-minus-Bottleneck"][cfg] = {
            "estimand": "granger_gap_minus_bottleneck_gap",
            "interpretation": "exploratory_paired_method_contrast",
            "mean_difference": float(difference.mean()),
            "per_seed_difference": difference.tolist(),
            "inference": inference,
        }
    apply_holm_family(
        [result["inference"] for result in method_contrasts["Granger-minus-Bottleneck"].values()],
        family_name=contrast_family,
    )

    total_time = time.time() - total_start
    print(f"\n{'=' * 72}\nFINAL VERDICT\n{'=' * 72}")
    print(f"{'Config':<15} " + " ".join(f"{m:<12}" for m in methods))
    for cfg, res in all_results.items():
        row = f"{cfg:<15} "
        for m in methods:
            gap = res[m]["size_matched_gap_mean"]
            sig = "*" if res[m]["primary_inference"]["reject_holm"] else ""
            row += f"{gap:+.3f}{sig:<7} "
        print(row)
    print(f"\nTotal runtime: {total_time / 60:.1f} min")

    out_path = Path(save_dir) / "causal_baselines_results.json"
    payload = {
        "metadata": {
            "experiment": "exp25_causal_baselines",
            "generation": {
                "n_seeds": n_seeds,
                "replicate_indices": list(range(n_seeds)),
                "data_seeds": [100 + seed for seed in range(n_seeds)],
                "protocol_root_seeds": [200 + seed for seed in range(n_seeds)],
                "K_values": list(K_values),
                "T_values": list(T_values),
                "density": float(density),
                "intervention_types": ["random"],
                "intervention_strengths": [float(intervention_strength)],
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
                "methods": list(methods),
                "metric": "aggregate_auroc",
                "metric_scope": "aggregate_edge",
                "estimand": "intervention_target_aware",
                "lasso_alpha_values": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "pcmci_pc_alpha": 0.05,
                "granger_score_type": "neg_log_pvalue",
                "include_modern_baselines": bool(include_modern_baselines),
                "skipped_methods": (
                    ["DYNOTEARS", "VAR-LiNGAM"] if include_modern_baselines else []
                ),
            },
            "statistics": {
                "primary_family": primary_family,
                "exploratory_method_contrast_family": contrast_family,
                "multiple_testing": "Holm",
                "alpha": 0.05,
                "paired_test": "Wilcoxon greater",
                "effect": "paired_mean_difference",
                "bootstrap_confidence": 0.95,
                "bootstrap_resamples": 10_000,
            },
        },
        "results": all_results,
        "method_contrasts": method_contrasts,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"Saved to {out_path}")

    return payload


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
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--scale-floor", type=float, default=1e-8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--modern-baselines", action="store_true")
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
        include_modern_baselines=args.modern_baselines,
        noise_std=args.noise_std,
        burnin=args.burnin,
        scale_floor=args.scale_floor,
    )
