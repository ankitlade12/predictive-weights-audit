"""
Experiment 15: Stress grid for explicit lagged bottleneck extraction.

The question is no longer "can lag extraction work on clean VAR data?"
It can. The sharper question is whether a low-rank bottleneck separates from
ordinary lagged linear baselines when the problem is sample-limited or the
model lag order is over-specified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.linalg import LinAlgWarning
from scipy.stats import ConstantInputWarning
from sklearn.exceptions import ConvergenceWarning

from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    evaluate_lagged_bottleneck_model,
    evaluate_lagged_scores,
    fit_tuned_lagged_linear_baseline,
)
from ssm_causal.lag_synthetic import (
    STRESS_DATASETS,
    pad_lag_graph,
    train_test_split_time_series,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


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
    return obj


def train_and_evaluate_bottleneck(
    X_train: np.ndarray,
    X_test: np.ndarray,
    G_lag: np.ndarray,
    K: int,
    max_lag: int,
    d_bottleneck: int,
    epochs: int,
    batch_size: int,
    lambda_sparse: float,
    device: str,
) -> dict:
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
    metrics = evaluate_lagged_bottleneck_model(model, X_train, G_lag, device=device)
    metrics["train_prediction_mse"] = metrics.pop("prediction_mse")
    metrics["test_prediction_mse"] = trainer.prediction_mse(X_test)
    metrics["final_prediction_loss"] = history["prediction_loss"][-1]
    return metrics


def train_and_evaluate_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    G_lag: np.ndarray,
    max_lag: int,
    baseline_methods: list[str],
    baseline_alphas: list[float],
    rrr_ranks: list[int],
    baseline_validation_fraction: float,
) -> dict[str, dict[str, float]]:
    baseline_results = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        for method in baseline_methods:
            baseline = fit_tuned_lagged_linear_baseline(
                X_train=X_train,
                max_lag=max_lag,
                method=method,
                alpha_values=baseline_alphas,
                rank_values=rrr_ranks,
                validation_fraction=baseline_validation_fraction,
                X_test=X_test,
            )
            metrics = evaluate_lagged_scores(baseline.score, G_lag)
            metrics["train_prediction_mse"] = baseline.train_prediction_mse
            metrics["test_prediction_mse"] = baseline.test_prediction_mse
            metrics["params"] = baseline.params or {}
            baseline_results[method.upper()] = metrics
    return baseline_results


def compare_to_baselines(methods: dict[str, dict[str, float]]) -> dict[str, float | str]:
    bottleneck = methods["LaggedBottleneck"]
    baselines = {
        name: metrics
        for name, metrics in methods.items()
        if name != "LaggedBottleneck"
    }

    best_flat_name, best_flat = max(
        baselines.items(),
        key=lambda item: item[1]["flat_lag_auroc"],
    )
    best_mse_name, best_mse = min(
        baselines.items(),
        key=lambda item: item[1]["test_prediction_mse"],
    )

    flat_delta = bottleneck["flat_lag_auroc"] - best_flat["flat_lag_auroc"]
    top1_delta = bottleneck["edge_lag_top1_acc"] - best_flat["edge_lag_top1_acc"]
    mse_delta = bottleneck["test_prediction_mse"] - best_mse["test_prediction_mse"]

    return {
        "best_flat_baseline": best_flat_name,
        "best_mse_baseline": best_mse_name,
        "flat_lag_auroc_delta_vs_best_baseline": float(flat_delta),
        "edge_lag_top1_delta_vs_best_flat_baseline": float(top1_delta),
        "test_mse_delta_vs_best_baseline": float(mse_delta),
        "bottleneck_graph_win": bool(flat_delta > 0.005),
        "bottleneck_mse_win": bool(mse_delta < -0.005),
    }


def run_stress_grid(
    datasets: list[str],
    K_values: list[int],
    T_values: list[int],
    max_lag_values: list[int],
    d_bottleneck_values: list[int],
    seeds: int,
    epochs: int,
    batch_size: int,
    train_fraction: float,
    lambda_sparse: float,
    baseline_methods: list[str],
    baseline_alphas: list[float],
    rrr_ranks: list[int],
    baseline_validation_fraction: float,
    device: str,
    save_dir: str,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    runs = []

    grid = list(product(datasets, K_values, T_values, max_lag_values, d_bottleneck_values))
    print("=" * 80)
    print("EXPERIMENT 15: Lagged Bottleneck Stress Grid")
    print(
        f"Grid configs: {len(grid)} | Seeds: {seeds} | Epochs: {epochs} | "
        f"Device: {device}"
    )
    print("=" * 80)

    for dataset, K, T, max_lag, d_bottleneck in grid:
        generator = STRESS_DATASETS[dataset]
        for seed_idx in range(seeds):
            seed = 20_000 + seed_idx
            set_seed(seed)
            X, G_lag, desc = generator(K=K, T=T, max_lag=max_lag, seed=seed)
            G_lag = pad_lag_graph(G_lag, max_lag)
            X_train, X_test = train_test_split_time_series(
                X,
                train_fraction=train_fraction,
                max_lag=max_lag,
            )

            methods = {
                "LaggedBottleneck": train_and_evaluate_bottleneck(
                    X_train=X_train,
                    X_test=X_test,
                    G_lag=G_lag,
                    K=K,
                    max_lag=max_lag,
                    d_bottleneck=d_bottleneck,
                    epochs=epochs,
                    batch_size=batch_size,
                    lambda_sparse=lambda_sparse,
                    device=device,
                )
            }
            methods.update(
                train_and_evaluate_baselines(
                    X_train=X_train,
                    X_test=X_test,
                    G_lag=G_lag,
                    max_lag=max_lag,
                    baseline_methods=baseline_methods,
                    baseline_alphas=baseline_alphas,
                    rrr_ranks=rrr_ranks,
                    baseline_validation_fraction=baseline_validation_fraction,
                )
            )

            comparison = compare_to_baselines(methods)
            run = {
                "dataset": dataset,
                "description": desc,
                "K": K,
                "T": T,
                "max_lag": max_lag,
                "d_bottleneck": d_bottleneck,
                "seed": seed_idx,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "true_edges_per_lag": G_lag.sum(axis=(1, 2)).astype(int).tolist(),
                "methods": methods,
                "comparison": comparison,
            }
            runs.append(run)

            print(
                f"{dataset:18s} K={K:<3d} T={T:<4d} lag={max_lag:<2d} "
                f"d={d_bottleneck:<2d} seed={seed_idx} | "
                f"flat_delta={comparison['flat_lag_auroc_delta_vs_best_baseline']:+.3f} "
                f"mse_delta={comparison['test_mse_delta_vs_best_baseline']:+.3f}"
            )

    summary = summarize_runs(runs)
    results = {
        "metadata": {
            "datasets": datasets,
            "K_values": K_values,
            "T_values": T_values,
            "max_lag_values": max_lag_values,
            "d_bottleneck_values": d_bottleneck_values,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_fraction": train_fraction,
            "lambda_sparse": lambda_sparse,
            "baseline_methods": baseline_methods,
            "baseline_alphas": baseline_alphas,
            "rrr_ranks": rrr_ranks,
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
    print(format_overall_summary(summary["overall"]))
    for dataset, dataset_summary in summary["by_dataset"].items():
        print(f"{dataset:18s} {format_overall_summary(dataset_summary)}")

    out_path = Path(save_dir) / "lagged_bottleneck_stress_grid_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def summarize_runs(runs: list[dict]) -> dict:
    def _summarize(items: list[dict]) -> dict:
        flat_deltas = [
            item["comparison"]["flat_lag_auroc_delta_vs_best_baseline"]
            for item in items
        ]
        mse_deltas = [
            item["comparison"]["test_mse_delta_vs_best_baseline"]
            for item in items
        ]
        graph_wins = [item["comparison"]["bottleneck_graph_win"] for item in items]
        mse_wins = [item["comparison"]["bottleneck_mse_win"] for item in items]
        return {
            "n_runs": len(items),
            "mean_flat_delta": float(np.mean(flat_deltas)) if items else float("nan"),
            "median_flat_delta": float(np.median(flat_deltas)) if items else float("nan"),
            "mean_test_mse_delta": float(np.mean(mse_deltas)) if items else float("nan"),
            "median_test_mse_delta": float(np.median(mse_deltas)) if items else float("nan"),
            "graph_win_rate": float(np.mean(graph_wins)) if items else float("nan"),
            "mse_win_rate": float(np.mean(mse_wins)) if items else float("nan"),
        }

    by_dataset = {
        dataset: _summarize([run for run in runs if run["dataset"] == dataset])
        for dataset in sorted({run["dataset"] for run in runs})
    }
    by_K = {
        str(K): _summarize([run for run in runs if run["K"] == K])
        for K in sorted({run["K"] for run in runs})
    }
    by_T = {
        str(T): _summarize([run for run in runs if run["T"] == T])
        for T in sorted({run["T"] for run in runs})
    }
    by_lag = {
        str(max_lag): _summarize([run for run in runs if run["max_lag"] == max_lag])
        for max_lag in sorted({run["max_lag"] for run in runs})
    }
    by_d_bottleneck = {
        str(d_bottleneck): _summarize(
            [run for run in runs if run["d_bottleneck"] == d_bottleneck]
        )
        for d_bottleneck in sorted({run["d_bottleneck"] for run in runs})
    }
    return {
        "overall": _summarize(runs),
        "by_dataset": by_dataset,
        "by_K": by_K,
        "by_T": by_T,
        "by_max_lag": by_lag,
        "by_d_bottleneck": by_d_bottleneck,
    }


def format_overall_summary(summary: dict) -> str:
    return (
        f"n={summary['n_runs']} | "
        f"mean_flat_delta={summary['mean_flat_delta']:+.3f} | "
        f"graph_win_rate={summary['graph_win_rate']:.2f} | "
        f"mean_mse_delta={summary['mean_test_mse_delta']:+.3f} | "
        f"mse_win_rate={summary['mse_win_rate']:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["VAR2-random", "Block-lowrank", "Delayed-sparse"],
    )
    parser.add_argument("--K-values", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--T-values", nargs="+", type=int, default=[150, 300])
    parser.add_argument("--max-lag-values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--d-bottleneck-values", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-sparse", type=float, default=0.0)
    parser.add_argument("--baseline-methods", nargs="+", default=["ols", "ridge", "lasso", "rrr"])
    parser.add_argument(
        "--baseline-alphas",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    parser.add_argument("--rrr-ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--baseline-validation-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="results/lagged_bottleneck_stress_grid")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)
    if args.quick:
        args.datasets = ["VAR2-random", "Block-lowrank"]
        args.K_values = [10]
        args.T_values = [150]
        args.max_lag_values = [8]
        args.d_bottleneck_values = [8]
        args.seeds = 1
        args.epochs = 40

    run_stress_grid(
        datasets=args.datasets,
        K_values=args.K_values,
        T_values=args.T_values,
        max_lag_values=args.max_lag_values,
        d_bottleneck_values=args.d_bottleneck_values,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        lambda_sparse=args.lambda_sparse,
        baseline_methods=args.baseline_methods,
        baseline_alphas=args.baseline_alphas,
        rrr_ranks=args.rrr_ranks,
        baseline_validation_fraction=args.baseline_validation_fraction,
        device=device,
        save_dir=args.save_dir,
    )
