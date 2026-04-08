"""
Experiment 17: lag-specific CauseMe-style benchmark gate.

This is the fairer follow-up to Exp 16 for the lag-resolved bottleneck idea.
Instead of collapsing every method to an aggregate graph only, we compare the
full lag score tensor against tuned lagged baselines on the same semi-real
structural process generators.
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

from ssm_causal.causeme_benchmarks import CauseMeScenario, default_causeme_scenarios
from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    evaluate_lagged_scores,
    extract_lagged_weight_score,
    fit_tuned_lagged_linear_baseline,
)
from ssm_causal.lag_synthetic import train_test_split_time_series

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize(obj):
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    if isinstance(obj, tuple):
        return [serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        value = obj.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if hasattr(obj, "__dict__"):
        return serialize(vars(obj))
    return obj


def _score_or_default(value: float, default: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else default


def evaluate_lagged_bottleneck(
    X_train: np.ndarray,
    X_test: np.ndarray,
    G_lag: np.ndarray,
    *,
    max_lag: int,
    d_bottleneck: int,
    epochs: int,
    batch_size: int,
    lambda_sparse: float,
    device: str,
) -> dict[str, float | dict]:
    K = X_train.shape[1]
    model = LaggedBottleneckModel(
        n_variables=K,
        max_lag=max_lag,
        d_bottleneck=d_bottleneck,
    )
    config = LaggedBottleneckTrainingConfig(
        n_epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_sparse=lambda_sparse,
        print_every=max(1, epochs // 2),
    )
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    history = trainer.train(X_train, verbose=False)

    S_lag = extract_lagged_weight_score(model, normalize="global")
    metrics = evaluate_lagged_scores(S_lag, G_lag)
    metrics["train_prediction_mse"] = trainer.prediction_mse(X_train)
    metrics["test_prediction_mse"] = trainer.prediction_mse(X_test)
    metrics["params"] = {
        "d_bottleneck": d_bottleneck,
        "epochs": epochs,
        "lambda_sparse": lambda_sparse,
        "final_prediction_loss": history["prediction_loss"][-1],
    }
    return metrics


def evaluate_tuned_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    G_lag: np.ndarray,
    *,
    max_lag: int,
    method: str,
    alpha_values: list[float],
    rank_values: list[int],
    validation_fraction: float,
) -> dict[str, float | dict]:
    baseline = fit_tuned_lagged_linear_baseline(
        X_train=X_train,
        max_lag=max_lag,
        method=method,
        alpha_values=alpha_values,
        rank_values=rank_values,
        validation_fraction=validation_fraction,
        X_test=X_test,
    )
    metrics = evaluate_lagged_scores(baseline.score, G_lag)
    metrics["train_prediction_mse"] = baseline.train_prediction_mse
    metrics["test_prediction_mse"] = baseline.test_prediction_mse
    metrics["params"] = baseline.params or {}
    return metrics


def compare_to_baselines(
    methods: dict[str, dict[str, float | dict]],
) -> dict[str, float | str | bool]:
    bottleneck = methods["LaggedBottleneck"]
    baselines = {
        name: metrics
        for name, metrics in methods.items()
        if name != "LaggedBottleneck"
    }

    best_flat_name, best_flat = max(
        baselines.items(),
        key=lambda item: _score_or_default(item[1]["flat_lag_auroc"], -np.inf),
    )
    best_top1_name, best_top1 = max(
        baselines.items(),
        key=lambda item: _score_or_default(item[1]["edge_lag_top1_acc"], -np.inf),
    )
    best_agg_name, best_agg = max(
        baselines.items(),
        key=lambda item: _score_or_default(item[1]["aggregate_auroc"], -np.inf),
    )
    best_mse_name, best_mse = min(
        baselines.items(),
        key=lambda item: _score_or_default(item[1]["test_prediction_mse"], np.inf),
    )

    flat_delta = _score_or_default(bottleneck["flat_lag_auroc"], -np.inf) - _score_or_default(
        best_flat["flat_lag_auroc"],
        -np.inf,
    )
    top1_delta = _score_or_default(
        bottleneck["edge_lag_top1_acc"],
        -np.inf,
    ) - _score_or_default(best_top1["edge_lag_top1_acc"], -np.inf)
    aggregate_delta = _score_or_default(
        bottleneck["aggregate_auroc"],
        -np.inf,
    ) - _score_or_default(best_agg["aggregate_auroc"], -np.inf)
    mse_delta = _score_or_default(
        bottleneck["test_prediction_mse"],
        np.inf,
    ) - _score_or_default(best_mse["test_prediction_mse"], np.inf)

    return {
        "best_flat_baseline": best_flat_name,
        "best_top1_baseline": best_top1_name,
        "best_aggregate_baseline": best_agg_name,
        "best_mse_baseline": best_mse_name,
        "flat_lag_auroc_delta_vs_best_baseline": float(flat_delta),
        "edge_lag_top1_delta_vs_best_baseline": float(top1_delta),
        "aggregate_auroc_delta_vs_best_baseline": float(aggregate_delta),
        "test_mse_delta_vs_best_baseline": float(mse_delta),
        "bottleneck_flat_win": bool(flat_delta > 0.02),
        "bottleneck_top1_win": bool(top1_delta > 0.02),
        "bottleneck_aggregate_win": bool(aggregate_delta > 0.02),
        "bottleneck_mse_win": bool(mse_delta < -0.005),
    }


def summarize_runs(runs: list[dict]) -> dict[str, dict]:
    def _summarize(items: list[dict]) -> dict[str, float]:
        flat_deltas = [
            item["comparison"]["flat_lag_auroc_delta_vs_best_baseline"]
            for item in items
        ]
        top1_deltas = [
            item["comparison"]["edge_lag_top1_delta_vs_best_baseline"]
            for item in items
        ]
        aggregate_deltas = [
            item["comparison"]["aggregate_auroc_delta_vs_best_baseline"]
            for item in items
        ]
        mse_deltas = [
            item["comparison"]["test_mse_delta_vs_best_baseline"]
            for item in items
        ]

        return {
            "runs": len(items),
            "flat_graph_win_rate": float(
                np.mean([item["comparison"]["bottleneck_flat_win"] for item in items])
            ),
            "top1_win_rate": float(
                np.mean([item["comparison"]["bottleneck_top1_win"] for item in items])
            ),
            "aggregate_win_rate": float(
                np.mean([item["comparison"]["bottleneck_aggregate_win"] for item in items])
            ),
            "mse_win_rate": float(
                np.mean([item["comparison"]["bottleneck_mse_win"] for item in items])
            ),
            "mean_flat_delta": float(np.mean(flat_deltas)),
            "mean_top1_delta": float(np.mean(top1_deltas)),
            "mean_aggregate_delta": float(np.mean(aggregate_deltas)),
            "mean_mse_delta": float(np.mean(mse_deltas)),
        }

    by_scenario = {}
    scenario_names = sorted({run["scenario"] for run in runs})
    for scenario_name in scenario_names:
        by_scenario[scenario_name] = _summarize(
            [run for run in runs if run["scenario"] == scenario_name]
        )
    return {"overall": _summarize(runs), "by_scenario": by_scenario}


def format_summary(summary: dict[str, float]) -> str:
    return (
        f"runs={summary['runs']} "
        f"flat_win={summary['flat_graph_win_rate']:.2f} "
        f"top1_win={summary['top1_win_rate']:.2f} "
        f"agg_win={summary['aggregate_win_rate']:.2f} "
        f"mse_win={summary['mse_win_rate']:.2f} "
        f"flat_delta={summary['mean_flat_delta']:+.3f} "
        f"top1_delta={summary['mean_top1_delta']:+.3f} "
        f"agg_delta={summary['mean_aggregate_delta']:+.3f} "
        f"mse_delta={summary['mean_mse_delta']:+.3f}"
    )


def run_benchmark(
    scenarios: list[CauseMeScenario],
    *,
    seeds: int,
    epochs: int,
    batch_size: int,
    d_bottleneck: int,
    train_fraction: float,
    lambda_sparse: float,
    baseline_methods: list[str],
    alpha_values: list[float],
    rank_values: list[int],
    baseline_validation_fraction: float,
    device: str,
    save_dir: str,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    runs: list[dict] = []

    print("=" * 80)
    print("EXPERIMENT 17: CauseMe-style Lag-Specific Gate")
    print(
        f"Scenarios: {len(scenarios)} | Seeds: {seeds} | Epochs: {epochs} | Device: {device}"
    )
    print("=" * 80)

    for scenario in scenarios:
        print(f"\n{'─' * 80}")
        print(f"{scenario.name} | K={scenario.K} T={scenario.T} lag={scenario.max_lag}")
        print(f"{'─' * 80}")

        for seed_idx in range(seeds):
            seed = 30_000 + seed_idx
            set_seed(seed)
            X, _, G_lag, description = scenario.generate(seed=seed)
            X = np.nan_to_num(X, 0.0)
            X_train, X_test = train_test_split_time_series(
                X,
                train_fraction=train_fraction,
                max_lag=scenario.max_lag,
            )

            methods: dict[str, dict[str, float | dict]] = {
                "LaggedBottleneck": evaluate_lagged_bottleneck(
                    X_train,
                    X_test,
                    G_lag,
                    max_lag=scenario.max_lag,
                    d_bottleneck=d_bottleneck,
                    epochs=epochs,
                    batch_size=batch_size,
                    lambda_sparse=lambda_sparse,
                    device=device,
                )
            }
            for method in baseline_methods:
                methods[method.upper()] = evaluate_tuned_baseline(
                    X_train,
                    X_test,
                    G_lag,
                    max_lag=scenario.max_lag,
                    method=method,
                    alpha_values=alpha_values,
                    rank_values=rank_values,
                    validation_fraction=baseline_validation_fraction,
                )

            comparison = compare_to_baselines(methods)
            run = {
                "scenario": scenario.name,
                "description": description,
                "seed": seed_idx,
                "config": serialize(scenario),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "methods": methods,
                "comparison": comparison,
            }
            runs.append(run)

            print(
                f"seed={seed_idx} | "
                f"flat_delta={comparison['flat_lag_auroc_delta_vs_best_baseline']:+.3f} "
                f"top1_delta={comparison['edge_lag_top1_delta_vs_best_baseline']:+.3f} "
                f"agg_delta={comparison['aggregate_auroc_delta_vs_best_baseline']:+.3f} "
                f"mse_delta={comparison['test_mse_delta_vs_best_baseline']:+.3f} "
                f"| best_flat={comparison['best_flat_baseline']}"
            )

    summary = summarize_runs(runs)
    results = {
        "metadata": {
            "scenarios": [serialize(scenario) for scenario in scenarios],
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "d_bottleneck": d_bottleneck,
            "train_fraction": train_fraction,
            "lambda_sparse": lambda_sparse,
            "baseline_methods": baseline_methods,
            "alpha_values": alpha_values,
            "rank_values": rank_values,
            "baseline_validation_fraction": baseline_validation_fraction,
            "device": device,
            "runtime_seconds": time.time() - start_time,
        },
        "summary": summary,
        "runs": runs,
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(format_summary(summary["overall"]))
    for scenario_name, scenario_summary in summary["by_scenario"].items():
        print(f"{scenario_name:24s} {format_summary(scenario_summary)}")

    out_path = Path(save_dir) / "causeme_lag_gate_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-bottleneck", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--lambda-sparse", type=float, default=0.0)
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["ols", "ridge", "lasso", "rrr"],
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1],
    )
    parser.add_argument("--rrr-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--baseline-validation-fraction", type=float, default=0.25)
    parser.add_argument("--save-dir", default="results/causeme_lag_gate")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    scenarios = default_causeme_scenarios(include_nonlinear=True)
    if args.scenarios:
        requested = set(args.scenarios)
        scenarios = [scenario for scenario in scenarios if scenario.name in requested]

    if args.quick:
        quick_names = {"CauseMe-linear-K5", "CauseMe-nonlinear-K5"}
        scenarios = [scenario for scenario in scenarios if scenario.name in quick_names]
        args.seeds = min(args.seeds, 1)
        args.epochs = min(args.epochs, 30)

    if not scenarios:
        raise ValueError("No scenarios selected.")

    run_benchmark(
        scenarios=scenarios,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_bottleneck=args.d_bottleneck,
        train_fraction=args.train_fraction,
        lambda_sparse=args.lambda_sparse,
        baseline_methods=args.baseline_methods,
        alpha_values=args.alpha_grid,
        rank_values=args.rrr_ranks,
        baseline_validation_fraction=args.baseline_validation_fraction,
        device=choose_device(args.device),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
