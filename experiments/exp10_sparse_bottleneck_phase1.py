"""
Phase 1 feasibility for Prototype Model B (Sparse Bottleneck).

Compares:
    - Baseline forecasting training
    - Sparse bottleneck training with endpoint L1 regularization

Primary readout is still the locked default extractor:
    S_weight = |W_out @ W_in|

Usage:
    python experiments/exp10_sparse_bottleneck_phase1.py
    python experiments/exp10_sparse_bottleneck_phase1.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.synthetic import generate_var1, generate_var2
from ssm_causal.bottleneck_training import (
    BottleneckTrainingConfig,
    CausalBottleneckTrainer,
    evaluate_bottleneck_model,
)
from ssm_causal.mamba_model import MambaForTimeSeries

DatasetFactory = Callable[[int], tuple[np.ndarray, object, np.ndarray]]


DATASETS: Dict[str, DatasetFactory] = {
    "VAR1-chain-K5": lambda seed: generate_var1(
        K=5,
        T=1200,
        graph_type="chain",
        seed=seed,
    ),
    "VAR1-random-K10": lambda seed: generate_var1(
        K=10,
        T=1600,
        graph_type="random",
        seed=seed,
    ),
    "VAR2-K5": lambda seed: generate_var2(
        K=5,
        T=1600,
        seed=seed,
    ),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_eval(
    X: np.ndarray,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential split for a quick forecast/readout sanity check."""
    split_idx = int(len(X) * train_ratio)
    split_idx = max(2, min(len(X) - 2, split_idx))
    X_train = X[:split_idx]
    X_eval = X[split_idx - 1 :]
    return X_train, X_eval


def make_model(
    n_variables: int,
    d_model: int,
    d_state: int,
    n_layers: int,
) -> MambaForTimeSeries:
    return MambaForTimeSeries(
        n_variables=n_variables,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        dropout=0.05,
    )


def run_method(
    *,
    method_name: str,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    G_true: np.ndarray,
    d_model: int,
    d_state: int,
    n_layers: int,
    config: BottleneckTrainingConfig,
    model_seed: int,
    device: str,
) -> Dict[str, float]:
    set_seed(model_seed)
    model = make_model(
        n_variables=X_train.shape[1],
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
    )

    trainer = CausalBottleneckTrainer(model, config=config, device=device)
    history = trainer.train(X_train, verbose=False)
    metrics = evaluate_bottleneck_model(model, X_eval, G_true, device=device)
    metrics["final_total_loss"] = float(history["loss"][-1])
    metrics["final_prediction_loss"] = float(history["prediction_loss"][-1])
    metrics["final_sparse_loss"] = float(history["sparse_loss"][-1])
    metrics["lambda_sparse"] = float(config.lambda_sparse)
    metrics["method"] = method_name
    return metrics


def summarize_runs(run_metrics: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    metric_keys = [
        "auroc",
        "auprc",
        "f1",
        "prediction_mse",
        "graph_density",
        "mean_abs_score",
        "final_total_loss",
        "final_prediction_loss",
        "final_sparse_loss",
    ]

    for key in metric_keys:
        values = [float(run[key]) for run in run_metrics if key in run and np.isfinite(run[key])]
        if not values:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return summary


def print_dataset_summary(dataset_name: str, results: Dict[str, Dict[str, object]]) -> None:
    print(f"\n{'=' * 72}")
    print(f"{dataset_name}")
    print(f"{'=' * 72}")
    for method_name, payload in results.items():
        summary = payload["summary"]
        print(
            f"{method_name:18s}"
            f" AUROC {summary['auroc']['mean']:.3f} +/- {summary['auroc']['std']:.3f}"
            f" | AUPRC {summary['auprc']['mean']:.3f}"
            f" | F1 {summary['f1']['mean']:.3f}"
            f" | Pred MSE {summary['prediction_mse']['mean']:.5f}"
            f" | Density {summary['graph_density']['mean']:.3f}"
        )


def to_serializable(obj: object) -> object:
    """Convert numpy scalars and nested containers for JSON output."""
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_phase1(
    *,
    dataset_names: list[str],
    seeds: int,
    d_model: int,
    d_state: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    seq_len: int,
    lambda_sparse: float,
    lambda_grid: Optional[list[float]],
    train_ratio: float,
    device: str,
    save_dir: str,
) -> Dict[str, Dict[str, object]]:
    os.makedirs(save_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, object]] = {}
    sparse_values = lambda_grid or [lambda_sparse]
    sparse_values = [float(value) for value in sparse_values if float(value) > 0.0]
    sparse_values = list(dict.fromkeys(sparse_values))

    for dataset_name in dataset_names:
        generator = DATASETS[dataset_name]
        dataset_results: Dict[str, Dict[str, object]] = {"Baseline": {"runs": []}}

        print(f"\nRunning {dataset_name}...")
        for seed_idx in range(seeds):
            data_seed = 100 + seed_idx
            X, _, G_true = generator(data_seed)
            X_train, X_eval = split_train_eval(X, train_ratio=train_ratio)

            baseline_config = BottleneckTrainingConfig(
                n_epochs=epochs,
                batch_size=batch_size,
                seq_len=seq_len,
                lambda_sparse=0.0,
            )
            method_configs: Dict[str, BottleneckTrainingConfig] = {"Baseline": baseline_config}

            for value in sparse_values:
                method_name = f"Sparse({value:.0e})"
                dataset_results.setdefault(method_name, {"runs": []})
                method_configs[method_name] = BottleneckTrainingConfig(
                    n_epochs=epochs,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    lambda_sparse=value,
                )

            for method_name, config in method_configs.items():
                metrics = run_method(
                    method_name=method_name,
                    X_train=X_train,
                    X_eval=X_eval,
                    G_true=G_true,
                    d_model=d_model,
                    d_state=d_state,
                    n_layers=n_layers,
                    config=config,
                    model_seed=1000 + seed_idx,
                    device=device,
                )
                dataset_results[method_name]["runs"].append(metrics)

        for method_name in dataset_results:
            dataset_results[method_name]["summary"] = summarize_runs(
                dataset_results[method_name]["runs"]
            )

        all_results[dataset_name] = dataset_results
        print_dataset_summary(dataset_name, dataset_results)

    payload = {
        "config": {
            "datasets": dataset_names,
            "seeds": seeds,
            "d_model": d_model,
            "d_state": d_state,
            "n_layers": n_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lambda_sparse": lambda_sparse,
            "lambda_grid": sparse_values,
            "train_ratio": train_ratio,
            "device": device,
        },
        "results": all_results,
    }

    output_path = Path(save_dir) / "phase1_sparse_bottleneck_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)

    print(f"\nSaved results to {output_path}")
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["VAR1-chain-K5", "VAR1-random-K10", "VAR2-K5"],
        choices=sorted(DATASETS.keys()),
        help="Datasets to include in the Phase 1 feasibility run.",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=100, help="Training subsequence length.")
    parser.add_argument(
        "--lambda-sparse",
        type=float,
        default=3e-5,
        help="Sparse bottleneck regularization strength.",
    )
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        help="Optional sparse bottleneck sweep values. Baseline is always included.",
    )
    parser.add_argument("--d-model", type=int, default=64, help="Model width.")
    parser.add_argument("--d-state", type=int, default=16, help="SSM state size.")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of SSM layers.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/eval split ratio.")
    parser.add_argument(
        "--save-dir",
        default="results/phase1_sparse_bottleneck",
        help="Directory for JSON outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny smoke version on one dataset and one seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick:
        args.datasets = ["VAR1-chain-K5"]
        args.seeds = 1
        args.epochs = 5
        args.batch_size = 4
        args.seq_len = 40
        args.d_model = 16
        args.d_state = 8
        args.n_layers = 1

    run_phase1(
        dataset_names=args.datasets,
        seeds=args.seeds,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lambda_sparse=args.lambda_sparse,
        lambda_grid=args.lambda_grid,
        train_ratio=args.train_ratio,
        device=args.device,
        save_dir=args.save_dir,
    )
