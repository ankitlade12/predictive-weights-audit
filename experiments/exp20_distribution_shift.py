"""
Experiment 20: Fixed-score graph-label sensitivity.

Each method is trained once on source data. Its extracted score matrix is then
evaluated against reference graph labels and, when rewiring changes the labels,
against comparison graph labels. This measures sensitivity of a fixed score
matrix to changed labels; it does not measure adaptation or robustness under a
target distribution.

One explicit identity-label control verifies that unchanged labels are reported
as not applicable. Structural comparisons rewire labels without retraining.

Usage:
    python experiments/exp20_distribution_shift.py
    python experiments/exp20_distribution_shift.py --quick
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
import torch.nn as nn
from scipy.linalg import LinAlgWarning
from scipy.stats import ConstantInputWarning
from sklearn.exceptions import ConvergenceWarning

from ssm_causal.lag_bottleneck import (
    evaluate_lagged_scores,
    fit_tuned_lagged_linear_baseline,
    make_lagged_dataset,
    normalize_lagged_score,
)
from ssm_causal.lag_synthetic import (
    ensure_stable_lag_coefficients,
    lag_graph_from_coefficients,
    pad_lag_graph,
    simulate_lagged_var,
    standardize,
    train_test_split_time_series,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

BASE_RNG_SEED = 50_000
SOURCE_NOISE_STD = 0.1
SOURCE_TRAIN_FRACTION = 0.8


# ── Comparison generators ────────────────────────────────────────────────────


def make_base_coefficients(
    K: int,
    max_lag: int,
) -> np.ndarray:
    """Chain-plus-delayed graph used as the base causal structure."""
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.30] * K)
    for i in range(1, K):
        A_lags[0, i, i - 1] = 0.45
    if max_lag > 2 and K > 2:
        A_lags[2, 2, 0] = 0.40
    if max_lag > 3 and K > 4:
        A_lags[3, 4, 1] = 0.35
    return ensure_stable_lag_coefficients(A_lags)


def make_same_graph_label_control(
    A_lags: np.ndarray,
    T: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Create source data with an explicit unchanged-label comparison."""
    X_source = standardize(
        simulate_lagged_var(
            A_lags,
            T=T,
            noise_std=SOURCE_NOISE_STD,
            seed=seed,
        )
    )
    G_lag = lag_graph_from_coefficients(A_lags)
    return X_source, G_lag, G_lag.copy(), "same-graph-label-control"


def apply_structural_rewiring(
    A_lags: np.ndarray,
    n_edges_to_flip: int,
    T: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Create source data and a rewired comparison graph."""
    rng = np.random.RandomState(seed + 9000)
    max_lag, K, _ = A_lags.shape

    X_source = standardize(
        simulate_lagged_var(
            A_lags,
            T=T,
            noise_std=SOURCE_NOISE_STD,
            seed=seed,
        )
    )

    A_shifted = A_lags.copy()
    pairs = [(lag, i, j) for lag in range(max_lag) for i in range(K) for j in range(K) if i != j]
    rng.shuffle(pairs)

    flipped = 0
    for lag, i, j in pairs:
        if flipped >= n_edges_to_flip:
            break
        if abs(A_shifted[lag, i, j]) > 1e-10:
            # Remove edge
            A_shifted[lag, i, j] = 0.0
        else:
            # Add edge
            A_shifted[lag, i, j] = rng.uniform(0.25, 0.50) * rng.choice([-1, 1])
        flipped += 1

    A_shifted = ensure_stable_lag_coefficients(A_shifted)
    G_source = lag_graph_from_coefficients(A_lags)
    G_comparison = lag_graph_from_coefficients(A_shifted)
    desc = f"structural-rewiring(flipped={n_edges_to_flip})"
    return X_source, G_source, G_comparison, desc


# ── Models ───────────────────────────────────────────────────────────────────


class LinearBottleneck(nn.Module):
    def __init__(self, n_variables: int, max_lag: int, d_bottleneck: int):
        super().__init__()
        self.n_variables = n_variables
        self.max_lag = max_lag
        self.d_bottleneck = d_bottleneck
        self.input_proj = nn.Linear(n_variables * max_lag, d_bottleneck)
        self.output_proj = nn.Linear(d_bottleneck, n_variables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.input_proj(x))


class NonlinearBottleneck(nn.Module):
    def __init__(self, n_variables: int, max_lag: int, d_bottleneck: int):
        super().__init__()
        self.n_variables = n_variables
        self.max_lag = max_lag
        self.d_bottleneck = d_bottleneck
        self.input_proj = nn.Linear(n_variables * max_lag, d_bottleneck)
        self.act = nn.ReLU()
        self.output_proj = nn.Linear(d_bottleneck, n_variables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.act(self.input_proj(x)))


# ── Helpers ──────────────────────────────────────────────────────────────────


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
        return None if isinstance(value, float) and not np.isfinite(value) else value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def extract_endpoint_score(model: nn.Module) -> np.ndarray:
    W_in = model.input_proj.weight.detach()
    W_out = model.output_proj.weight.detach()
    K = model.n_variables
    max_lag = model.max_lag
    scores = []
    for tau in range(max_lag):
        start = tau * K
        stop = start + K
        scores.append((W_out @ W_in[:, start:stop]).abs().cpu().numpy())
    return normalize_lagged_score(np.stack(scores, axis=0), mode="global")


def evaluate_fixed_score_graph_labels(
    score: np.ndarray,
    reference_graph: np.ndarray,
    comparison_graph: np.ndarray,
) -> dict:
    """Evaluate one unchanged score matrix against two graph label sets."""
    reference_metrics = evaluate_lagged_scores(score, reference_graph)
    comparison_metrics = evaluate_lagged_scores(score, comparison_graph)
    graph_labels_changed = not np.array_equal(reference_graph, comparison_graph)
    sensitivity = None
    comparison_status = "not_applicable_same_graph_labels"
    if graph_labels_changed:
        sensitivity = reference_metrics["flat_lag_auroc"] - comparison_metrics["flat_lag_auroc"]
        comparison_status = "fixed_score_graph_label_sensitivity"

    return {
        "reference_flat_lag_auroc": reference_metrics["flat_lag_auroc"],
        "reference_aggregate_auroc": reference_metrics["aggregate_auroc"],
        "comparison_flat_lag_auroc": comparison_metrics["flat_lag_auroc"],
        "comparison_aggregate_auroc": comparison_metrics["aggregate_auroc"],
        "graph_labels_changed": graph_labels_changed,
        "comparison_status": comparison_status,
        "graph_label_sensitivity": sensitivity,
        "score_matrix_reused": True,
    }


def train_bottleneck_and_score_graph_labels(
    model: nn.Module,
    X_source: np.ndarray,
    reference_graph: np.ndarray,
    comparison_graph: np.ndarray,
    max_lag: int,
    epochs: int,
    batch_size: int,
    lambda_sparse: float,
    device: str,
) -> dict:
    """Train on source and evaluate one score matrix against both label sets."""
    model = model.to(device)
    features, targets = make_lagged_dataset(X_source, max_lag, device=device)
    n_examples = features.shape[0]
    steps_per_epoch = max(1, n_examples // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, n_examples, (batch_size,), device=device)
            pred = model(features[idx])
            pred_loss = loss_fn(pred, targets[idx])
            sparse_loss = model.input_proj.weight.abs().sum() + model.output_proj.weight.abs().sum()
            loss = pred_loss + lambda_sparse * sparse_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    S_lag = extract_endpoint_score(model)
    return evaluate_fixed_score_graph_labels(
        S_lag,
        reference_graph,
        comparison_graph,
    )


# ── Main grid ────────────────────────────────────────────────────────────────


COMPARISON_CONFIGS = {
    "same-graph-label-control": ("control", {}),
    "struct-small": ("structural", {"n_edges_to_flip": 2}),
    "struct-large": ("structural", {"n_edges_to_flip": 5}),
}


def run_fixed_score_graph_label_sensitivity_experiment(
    comparison_names: list[str],
    K: int,
    T: int,
    max_lag: int,
    d_bottleneck: int,
    lambda_sparse: float,
    seeds: int,
    epochs: int,
    batch_size: int,
    baseline_methods: list[str],
    baseline_alphas: list[float],
    rrr_ranks: list[int],
    device: str,
    save_dir: str,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    runs = []
    actual_rng_seeds = [BASE_RNG_SEED + seed_idx for seed_idx in range(seeds)]
    method_names = [
        "Linear-BN",
        "Nonlinear-BN",
        *(method.upper() for method in baseline_methods),
    ]

    print("=" * 80)
    print("EXPERIMENT 20: Fixed-Score Graph-Label Sensitivity")
    print(f"Comparisons: {comparison_names} | K={K} T={T} | Seeds: {seeds}")
    print("=" * 80)

    for comparison_name in comparison_names:
        comparison_type, comparison_params = COMPARISON_CONFIGS[comparison_name]

        for seed_idx in range(seeds):
            seed = actual_rng_seeds[seed_idx]
            set_seed(seed)

            A_lags = make_base_coefficients(K, max_lag)

            # Generate source data and the graph-label comparison.
            if comparison_type == "control":
                X_source, G_source, G_comparison, desc = make_same_graph_label_control(
                    A_lags, T=T, seed=seed
                )
            elif comparison_type == "structural":
                X_source, G_source, G_comparison, desc = apply_structural_rewiring(
                    A_lags,
                    T=T,
                    seed=seed,
                    **comparison_params,
                )
            else:
                raise ValueError(f"Unknown comparison_type: {comparison_type}")

            G_source = pad_lag_graph(G_source, max_lag)
            G_comparison = pad_lag_graph(G_comparison, max_lag)
            graph_label_difference_count = int(np.count_nonzero(G_source != G_comparison))
            X_source_train, _ = train_test_split_time_series(
                X_source,
                SOURCE_TRAIN_FRACTION,
                max_lag,
            )

            methods: dict[str, dict] = {}

            # Linear bottleneck
            set_seed(seed)
            model_lin = LinearBottleneck(K, max_lag, d_bottleneck)
            methods["Linear-BN"] = train_bottleneck_and_score_graph_labels(
                model_lin,
                X_source_train,
                G_source,
                G_comparison,
                max_lag,
                epochs,
                batch_size,
                lambda_sparse,
                device,
            )

            # Nonlinear bottleneck
            set_seed(seed)
            model_nl = NonlinearBottleneck(K, max_lag, d_bottleneck)
            methods["Nonlinear-BN"] = train_bottleneck_and_score_graph_labels(
                model_nl,
                X_source_train,
                G_source,
                G_comparison,
                max_lag,
                epochs,
                batch_size,
                lambda_sparse,
                device,
            )

            # Classical baselines: train once, then reuse the same score matrix.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for bmethod in baseline_methods:
                    baseline = fit_tuned_lagged_linear_baseline(
                        X_source_train,
                        max_lag,
                        method=bmethod,
                        alpha_values=baseline_alphas,
                        rank_values=rrr_ranks,
                        validation_fraction=0.25,
                    )
                    methods[bmethod.upper()] = evaluate_fixed_score_graph_labels(
                        baseline.score,
                        G_source,
                        G_comparison,
                    )

            comparison_status = methods["Linear-BN"]["comparison_status"]

            run = {
                "comparison_name": comparison_name,
                "comparison_type": comparison_type,
                "comparison_parameters": dict(comparison_params),
                "description": desc,
                "K": K,
                "T": T,
                "max_lag": max_lag,
                "seed_index": seed_idx,
                "rng_seed": seed,
                "source_noise_std": SOURCE_NOISE_STD,
                "source_total_length": len(X_source),
                "source_train_length": len(X_source_train),
                "comparison_status": comparison_status,
                "graph_labels_changed": bool(methods["Linear-BN"]["graph_labels_changed"]),
                "graph_label_difference_count": graph_label_difference_count,
                "methods": methods,
            }
            runs.append(run)

            print(
                f"{comparison_name:24s} seed={seed_idx} "
                f"rng_seed={seed} | completed {len(methods)} methods"
            )

    summary = _summarize_comparisons(runs)
    results = {
        "metadata": {
            "objective": "fixed_score_graph_label_sensitivity",
            "interpretation": (
                "A source-trained score matrix is reused across graph labels; "
                "this does not measure target adaptation or shift robustness."
            ),
            "comparison_names": comparison_names,
            "actual_rng_seeds": actual_rng_seeds,
            "run_config": {
                "K": K,
                "T": T,
                "max_lag": max_lag,
                "d_bottleneck": d_bottleneck,
                "lambda_sparse": lambda_sparse,
                "n_seeds": seeds,
                "epochs": epochs,
                "batch_size": batch_size,
                "device": device,
                "methods": method_names,
                "baseline_methods": list(baseline_methods),
                "baseline_alpha_grid": list(baseline_alphas),
                "rrr_rank_grid": list(rrr_ranks),
                "source_noise_std": SOURCE_NOISE_STD,
                "source_train_fraction": SOURCE_TRAIN_FRACTION,
                "comparison_configs": {
                    name: {
                        "comparison_type": COMPARISON_CONFIGS[name][0],
                        "parameters": dict(COMPARISON_CONFIGS[name][1]),
                    }
                    for name in comparison_names
                },
            },
            "runtime_seconds": time.time() - start_time,
        },
        "summary": summary,
        "runs": runs,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: Fixed-Score Graph-Label Sensitivity")
    print("=" * 80)
    for comparison_name, comparison_summary in summary["by_comparison"].items():
        print(f"  {comparison_name}")
        for method_name, method_summary in comparison_summary["methods"].items():
            sensitivity = method_summary["graph_label_sensitivity_mean"]
            sensitivity_std = method_summary["graph_label_sensitivity_std"]
            sensitivity_text = (
                "N/A (labels unchanged)"
                if sensitivity is None
                else f"{sensitivity:+.3f} +/- {sensitivity_std:.3f}"
            )
            print(
                f"    {method_name:16s} | "
                f"reference_auroc="
                f"{method_summary['reference_flat_lag_auroc_mean']:.3f} "
                f"comparison_auroc="
                f"{method_summary['comparison_flat_lag_auroc_mean']:.3f} "
                f"label_sensitivity={sensitivity_text}"
            )

    out_path = Path(save_dir) / "fixed_score_graph_label_sensitivity_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def _summarize_comparisons(runs: list[dict]) -> dict:
    def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    def _method_stats(method_results: list[dict]) -> dict:
        reference_values = [result["reference_flat_lag_auroc"] for result in method_results]
        comparison_values = [result["comparison_flat_lag_auroc"] for result in method_results]
        sensitivity_values = [
            result["graph_label_sensitivity"]
            for result in method_results
            if result["graph_label_sensitivity"] is not None
        ]
        reference_mean, reference_std = _mean_std(reference_values)
        comparison_mean, comparison_std = _mean_std(comparison_values)
        sensitivity_mean, sensitivity_std = _mean_std(sensitivity_values)
        return {
            "n_seeds": len(method_results),
            "reference_flat_lag_auroc_mean": reference_mean,
            "reference_flat_lag_auroc_std": reference_std,
            "comparison_flat_lag_auroc_mean": comparison_mean,
            "comparison_flat_lag_auroc_std": comparison_std,
            "graph_label_sensitivity_mean": sensitivity_mean,
            "graph_label_sensitivity_std": sensitivity_std,
        }

    by_comparison = {}
    for comparison_name in sorted({r["comparison_name"] for r in runs}):
        comparison_runs = [run for run in runs if run["comparison_name"] == comparison_name]
        method_names = sorted(
            {method_name for run in comparison_runs for method_name in run["methods"]}
        )
        by_comparison[comparison_name] = {
            "n_seeds": len(comparison_runs),
            "methods": {
                method_name: _method_stats(
                    [
                        run["methods"][method_name]
                        for run in comparison_runs
                        if method_name in run["methods"]
                    ]
                )
                for method_name in method_names
            },
        }
    return {"by_comparison": by_comparison}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-names",
        nargs="+",
        default=list(COMPARISON_CONFIGS.keys()),
    )
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--T", type=int, default=800)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--d-bottleneck", type=int, default=12)
    parser.add_argument("--lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-methods", nargs="+", default=["lasso", "rrr"])
    parser.add_argument(
        "--baseline-alphas",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    parser.add_argument("--rrr-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-dir",
        default="results/fixed_score_graph_label_sensitivity",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)

    if args.quick:
        args.comparison_names = ["same-graph-label-control", "struct-small"]
        args.K = 4
        args.T = 400
        args.seeds = 2
        args.epochs = 30

    run_fixed_score_graph_label_sensitivity_experiment(
        comparison_names=args.comparison_names,
        K=args.K,
        T=args.T,
        max_lag=args.max_lag,
        d_bottleneck=args.d_bottleneck,
        lambda_sparse=args.lambda_sparse,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        baseline_methods=args.baseline_methods,
        baseline_alphas=args.baseline_alphas,
        rrr_ranks=args.rrr_ranks,
        device=device,
        save_dir=args.save_dir,
    )
