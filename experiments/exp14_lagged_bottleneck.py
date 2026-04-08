"""
Experiment 14: Explicit Lagged Bottleneck Extraction.

This tests the lag-resolved bottleneck hypothesis directly:

    train [x[t-1], ..., x[t-p]] -> bottleneck -> x[t]
    extract S_lag[p, target, source] = |W_out @ W_in_p|

Unlike Exp 13, this does not ask an SSM hidden state to disentangle lags
after the fact. The lag split is explicit in the input projection.
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

from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    evaluate_lagged_bottleneck_model,
    evaluate_lagged_scores,
    extract_lagged_weight_score,
    fit_lagged_linear_baseline,
)
from ssm_causal.lag_synthetic import (
    DATASETS,
    pad_lag_graph,
    train_test_split_time_series,
)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
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


def run_experiment(
    datasets: list[str],
    seeds: int,
    epochs: int,
    T: int,
    K: int,
    max_lag: int,
    d_bottleneck: int,
    batch_size: int,
    lambda_sparse: float,
    train_fraction: float,
    baseline_methods: list[str],
    baseline_alpha: float,
    device: str,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    results = {}

    print("=" * 72)
    print("EXPERIMENT 14: Explicit Lagged Bottleneck Extraction")
    print(f"Device: {device} | Seeds: {seeds} | Epochs: {epochs} | Max lag: {max_lag}")
    print("=" * 72)

    for dataset_name in datasets:
        generator = DATASETS[dataset_name]
        dataset_results = {}
        print(f"\n{'-' * 72}")
        print(f"Dataset: {dataset_name}")
        print(f"{'-' * 72}")

        for seed_idx in range(seeds):
            seed = 10_000 + seed_idx
            set_seed(seed)
            X, G_lag, desc = generator(K=K, T=T, max_lag=max_lag, seed=seed)
            G_lag = pad_lag_graph(G_lag, max_lag)

            X_train, X_test = train_test_split_time_series(
                X,
                train_fraction=train_fraction,
                max_lag=max_lag,
            )

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
            bottleneck_metrics = evaluate_lagged_bottleneck_model(
                model,
                X_train,
                G_lag,
                device=device,
            )
            bottleneck_metrics["train_prediction_mse"] = bottleneck_metrics.pop(
                "prediction_mse"
            )
            bottleneck_metrics["test_prediction_mse"] = trainer.prediction_mse(X_test)
            S_lag = extract_lagged_weight_score(model, normalize="global")

            method_results = {
                "LaggedBottleneck": {
                    "metrics": bottleneck_metrics,
                    "score_mass_per_lag": np.abs(S_lag).sum(axis=(1, 2)).tolist(),
                }
            }

            for method in baseline_methods:
                baseline = fit_lagged_linear_baseline(
                    X_train=X_train,
                    max_lag=max_lag,
                    method=method,
                    alpha=baseline_alpha,
                    X_test=X_test,
                )
                baseline_metrics = evaluate_lagged_scores(baseline.score, G_lag)
                baseline_metrics["train_prediction_mse"] = baseline.train_prediction_mse
                baseline_metrics["test_prediction_mse"] = baseline.test_prediction_mse
                method_results[method.upper()] = {
                    "metrics": baseline_metrics,
                    "score_mass_per_lag": np.abs(baseline.score).sum(axis=(1, 2)).tolist(),
                }

            dataset_results[f"seed_{seed_idx}"] = {
                "description": desc,
                "methods": method_results,
                "final_prediction_loss": history["prediction_loss"][-1],
                "max_abs_data": float(np.abs(X).max()),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "true_edges_per_lag": G_lag.sum(axis=(1, 2)).astype(int).tolist(),
            }

            print(
                f"  seed={seed_idx} "
                f"bottleneck_flat={bottleneck_metrics['flat_lag_auroc']:.3f} "
                f"bottleneck_top1={bottleneck_metrics['edge_lag_top1_acc']:.3f} "
                f"bottleneck_test_mse={bottleneck_metrics['test_prediction_mse']:.4f}"
            )

        summary = summarize_dataset(dataset_results)
        dataset_results["summary"] = summary
        results[dataset_name] = dataset_results
        for method_name, method_summary in summary.items():
            print(f"  {method_name:16s}", format_summary(method_summary))

    results["metadata"] = {
        "seeds": seeds,
        "epochs": epochs,
        "T": T,
        "K": K,
        "max_lag": max_lag,
        "d_bottleneck": d_bottleneck,
        "batch_size": batch_size,
        "lambda_sparse": lambda_sparse,
        "train_fraction": train_fraction,
        "baseline_methods": baseline_methods,
        "baseline_alpha": baseline_alpha,
        "device": device,
        "runtime_seconds": time.time() - start_time,
    }

    out_path = Path(save_dir) / "lagged_bottleneck_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def summarize_dataset(dataset_results: dict) -> dict[str, dict[str, dict[str, float]]]:
    metric_names = [
        "aggregate_auroc",
        "flat_lag_auroc",
        "flat_lag_auprc",
        "per_lag_auroc_mean",
        "edge_lag_top1_acc",
        "train_prediction_mse",
        "test_prediction_mse",
    ]
    summary = {}
    seed_items = [value for key, value in dataset_results.items() if key.startswith("seed_")]
    method_names = sorted(
        {
            method_name
            for item in seed_items
            for method_name in item["methods"]
        }
    )

    for method_name in method_names:
        method_summary = {}
        for metric_name in metric_names:
            values = [
                item["methods"][method_name]["metrics"][metric_name]
                for item in seed_items
                if method_name in item["methods"]
                and item["methods"][method_name]["metrics"][metric_name] is not None
                and np.isfinite(item["methods"][method_name]["metrics"][metric_name])
            ]
            method_summary[metric_name] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
            }
        summary[method_name] = method_summary
    return summary


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    top1_mean = summary["edge_lag_top1_acc"]["mean"]
    top1_std = summary["edge_lag_top1_acc"]["std"]
    return (
        f"agg={summary['aggregate_auroc']['mean']:.3f}+/-{summary['aggregate_auroc']['std']:.3f} | "
        f"flat={summary['flat_lag_auroc']['mean']:.3f}+/-{summary['flat_lag_auroc']['std']:.3f} | "
        f"top1={top1_mean:.3f}+/-{top1_std:.3f} | "
        f"test_mse={summary['test_prediction_mse']['mean']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--d-bottleneck", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-sparse", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--baseline-methods", nargs="+", default=["ols", "ridge", "lasso"])
    parser.add_argument("--baseline-alpha", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="results/lagged_bottleneck")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)
    if args.quick:
        args.datasets = ["VAR1-chain", "VAR2-random", "Mixed-lag", "Delayed-sparse"]
        args.seeds = 1
        args.epochs = 30
        args.T = 600
        args.d_bottleneck = 24
        args.batch_size = 128
    run_experiment(
        datasets=args.datasets,
        seeds=args.seeds,
        epochs=args.epochs,
        T=args.T,
        K=args.K,
        max_lag=args.max_lag,
        d_bottleneck=args.d_bottleneck,
        batch_size=args.batch_size,
        lambda_sparse=args.lambda_sparse,
        train_fraction=args.train_fraction,
        baseline_methods=args.baseline_methods,
        baseline_alpha=args.baseline_alpha,
        device=device,
        save_dir=args.save_dir,
    )
