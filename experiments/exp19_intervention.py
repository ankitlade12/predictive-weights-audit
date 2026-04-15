"""
Experiment 19: Interventional data — does do(X=x) amplify SSM vs linear gap?

Aman's hypothesis: observational data alone may not separate bottleneck
extraction from linear baselines, but interventional or do-calculus data
might reveal a genuine advantage.

The experiment generates paired datasets for each causal graph:
    1. Observational: standard VAR simulation.
    2. Interventional: for each target variable we clamp it to a constant
       (hard intervention do(X_i = c)) and concatenate the resulting data.

We train the same bottleneck models (linear + nonlinear) and baselines
(Lasso, RRR) on each data condition and compare graph recovery metrics.

Key question: is the delta(AUROC_interventional - AUROC_observational)
larger for bottleneck models than for classical baselines?

Usage:
    python experiments/exp19_intervention.py
    python experiments/exp19_intervention.py --quick
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
import torch.nn as nn
import torch.nn.functional as F
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
    standardize,
    train_test_split_time_series,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ── Interventional data generation ───────────────────────────────────────────


def simulate_observational_var(
    A_lags: np.ndarray,
    T: int,
    noise_std: float,
    seed: int,
    burnin: int = 300,
) -> np.ndarray:
    """Standard VAR simulation (observational regime)."""
    rng = np.random.RandomState(seed)
    max_lag, K, _ = A_lags.shape
    X = np.zeros((T + burnin, K))
    X[:max_lag] = rng.randn(max_lag, K) * noise_std

    for t in range(max_lag, T + burnin):
        value = np.zeros(K)
        for lag in range(max_lag):
            value += A_lags[lag] @ X[t - lag - 1]
        X[t] = value + rng.randn(K) * noise_std

    return X[burnin:]


def simulate_interventional_var(
    A_lags: np.ndarray,
    T_per_target: int,
    noise_std: float,
    seed: int,
    intervention_value: float = 0.0,
    burnin: int = 100,
) -> np.ndarray:
    """
    Generate interventional data: for each variable i, simulate under do(X_i = c).

    Under do(X_i = c) we clamp X_i to intervention_value at every step and
    sever all incoming edges to i. The resulting data is concatenated.
    """
    rng = np.random.RandomState(seed)
    max_lag, K, _ = A_lags.shape
    segments = []

    for target_i in range(K):
        X = np.zeros((T_per_target + burnin, K))
        X[:max_lag] = rng.randn(max_lag, K) * noise_std
        X[:max_lag, target_i] = intervention_value

        for t in range(max_lag, T_per_target + burnin):
            value = np.zeros(K)
            for lag in range(max_lag):
                value += A_lags[lag] @ X[t - lag - 1]
            value += rng.randn(K) * noise_std
            # Hard intervention: clamp target_i
            value[target_i] = intervention_value
            X[t] = value

        segments.append(X[burnin:])

    return np.concatenate(segments, axis=0)


def make_var_coefficients(
    K: int,
    max_lag: int,
    graph_type: str,
    seed: int,
) -> np.ndarray:
    """Generate stable VAR lag coefficients for a given graph type."""
    rng = np.random.RandomState(seed)
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.30] * K)

    if graph_type == "chain":
        for i in range(1, K):
            A_lags[0, i, i - 1] = 0.45
        if max_lag > 2:
            A_lags[2, min(K - 1, 2), 0] = 0.40

    elif graph_type == "fork":
        for i in range(1, K):
            A_lags[0, i, 0] = 0.35 + rng.uniform(-0.05, 0.05)

    elif graph_type == "random":
        n_off = K * (K - 1)
        n_edges = max(1, int(0.25 * n_off))
        pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
        chosen = rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False)
        for idx in chosen:
            i, j = pairs[idx]
            lag = rng.choice(min(3, max_lag))
            A_lags[lag, i, j] = rng.uniform(0.25, 0.55) * rng.choice([-1, 1])

    elif graph_type == "confounded":
        # Hidden common cause: var 0 drives vars 1 and 2 at different lags
        A_lags[0, 1, 0] = 0.50
        if max_lag > 1:
            A_lags[1, 2, 0] = 0.50
        # Spurious correlation between 1 and 2 (no direct edge)
        if K > 3:
            A_lags[0, 3, 1] = 0.35

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return ensure_stable_lag_coefficients(A_lags)


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


def train_bottleneck(
    model: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    G_lag: np.ndarray,
    max_lag: int,
    epochs: int,
    batch_size: int,
    lambda_sparse: float,
    device: str,
) -> dict:
    model = model.to(device)
    features, targets = make_lagged_dataset(X_train, max_lag, device=device)
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
            sparse_loss = (
                model.input_proj.weight.abs().sum()
                + model.output_proj.weight.abs().sum()
            )
            loss = pred_loss + lambda_sparse * sparse_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    S_lag = extract_endpoint_score(model)
    metrics = evaluate_lagged_scores(S_lag, G_lag)

    test_feat, test_tgt = make_lagged_dataset(X_test, max_lag, device=device)
    model.eval()
    with torch.no_grad():
        metrics["test_prediction_mse"] = float(
            F.mse_loss(model(test_feat), test_tgt).item()
        )
    return metrics


# ── Main grid ────────────────────────────────────────────────────────────────


def run_intervention_experiment(
    graph_types: list[str],
    K: int,
    T_obs: int,
    T_int_per_target: int,
    max_lag: int,
    d_bottleneck: int,
    lambda_sparse: float,
    seeds: int,
    epochs: int,
    batch_size: int,
    noise_std: float,
    baseline_methods: list[str],
    baseline_alphas: list[float],
    rrr_ranks: list[int],
    device: str,
    save_dir: str,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    runs = []

    conditions = ["observational", "interventional", "combined"]

    print("=" * 80)
    print("EXPERIMENT 19: Intervention vs Observational Data")
    print(
        f"Graph types: {graph_types} | K={K} T_obs={T_obs} "
        f"T_int/target={T_int_per_target} | Seeds: {seeds}"
    )
    print("=" * 80)

    for graph_type in graph_types:
        for seed_idx in range(seeds):
            seed = 40_000 + seed_idx
            set_seed(seed)

            A_lags = make_var_coefficients(K, max_lag, graph_type, seed)
            G_lag = lag_graph_from_coefficients(A_lags)
            G_lag = pad_lag_graph(G_lag, max_lag)

            # Generate data for each condition
            X_obs = standardize(
                simulate_observational_var(A_lags, T_obs, noise_std, seed)
            )
            X_int = standardize(
                simulate_interventional_var(
                    A_lags, T_int_per_target, noise_std, seed + 5000,
                )
            )
            X_combined = standardize(np.concatenate([X_obs, X_int], axis=0))

            data_by_condition = {
                "observational": X_obs,
                "interventional": X_int,
                "combined": X_combined,
            }

            condition_results: dict[str, dict[str, dict]] = {}

            for condition in conditions:
                X = data_by_condition[condition]
                X_train, X_test = train_test_split_time_series(X, 0.8, max_lag)

                methods: dict[str, dict] = {}

                # Linear bottleneck
                set_seed(seed)
                model_lin = LinearBottleneck(K, max_lag, d_bottleneck)
                methods["Linear-BN"] = train_bottleneck(
                    model_lin, X_train, X_test, G_lag,
                    max_lag, epochs, batch_size, lambda_sparse, device,
                )

                # Nonlinear bottleneck
                set_seed(seed)
                model_nl = NonlinearBottleneck(K, max_lag, d_bottleneck)
                methods["Nonlinear-BN"] = train_bottleneck(
                    model_nl, X_train, X_test, G_lag,
                    max_lag, epochs, batch_size, lambda_sparse, device,
                )

                # Classical baselines
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    for bmethod in baseline_methods:
                        baseline = fit_tuned_lagged_linear_baseline(
                            X_train, max_lag, method=bmethod,
                            alpha_values=baseline_alphas,
                            rank_values=rrr_ranks,
                            validation_fraction=0.25,
                            X_test=X_test,
                        )
                        bl_metrics = evaluate_lagged_scores(baseline.score, G_lag)
                        bl_metrics["test_prediction_mse"] = baseline.test_prediction_mse
                        bl_metrics["params"] = baseline.params or {}
                        methods[bmethod.upper()] = bl_metrics

                condition_results[condition] = methods

            # Compute deltas: interventional - observational
            deltas: dict[str, dict[str, float]] = {}
            for method_name in condition_results["observational"]:
                obs_auroc = condition_results["observational"][method_name]["flat_lag_auroc"]
                int_auroc = condition_results["interventional"][method_name]["flat_lag_auroc"]
                comb_auroc = condition_results["combined"][method_name]["flat_lag_auroc"]
                deltas[method_name] = {
                    "obs_auroc": float(obs_auroc),
                    "int_auroc": float(int_auroc),
                    "comb_auroc": float(comb_auroc),
                    "int_minus_obs": float(int_auroc - obs_auroc),
                    "comb_minus_obs": float(comb_auroc - obs_auroc),
                }

            # Does intervention help bottleneck more than baselines?
            bn_methods = [k for k in deltas if k.endswith("-BN")]
            bl_methods_list = [k for k in deltas if not k.endswith("-BN")]
            bn_int_gain = np.mean([deltas[k]["int_minus_obs"] for k in bn_methods])
            bl_int_gain = np.mean([deltas[k]["int_minus_obs"] for k in bl_methods_list])

            run = {
                "graph_type": graph_type,
                "K": K,
                "T_obs": T_obs,
                "T_int_per_target": T_int_per_target,
                "max_lag": max_lag,
                "seed": seed_idx,
                "conditions": condition_results,
                "deltas": deltas,
                "bn_mean_int_gain": float(bn_int_gain),
                "bl_mean_int_gain": float(bl_int_gain),
                "intervention_helps_bn_more": bool(bn_int_gain > bl_int_gain + 0.005),
            }
            runs.append(run)

            print(
                f"{graph_type:12s} seed={seed_idx} | "
                f"BN int_gain={bn_int_gain:+.3f} "
                f"BL int_gain={bl_int_gain:+.3f} "
                f"{'BN-WINS' if bn_int_gain > bl_int_gain + 0.005 else 'NO-DIFF'}"
            )

    summary = _summarize_intervention(runs)
    results = {
        "metadata": {
            "graph_types": graph_types,
            "K": K,
            "T_obs": T_obs,
            "T_int_per_target": T_int_per_target,
            "max_lag": max_lag,
            "d_bottleneck": d_bottleneck,
            "lambda_sparse": lambda_sparse,
            "noise_std": noise_std,
            "seeds": seeds,
            "epochs": epochs,
            "device": device,
            "runtime_seconds": time.time() - start_time,
        },
        "summary": summary,
        "runs": runs,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: Does interventional data amplify bottleneck advantage?")
    print("=" * 80)
    for graph_type, gsummary in summary["by_graph_type"].items():
        print(
            f"  {graph_type:12s} | "
            f"bn_int_gain={gsummary['mean_bn_int_gain']:+.3f} "
            f"bl_int_gain={gsummary['mean_bl_int_gain']:+.3f} "
            f"bn_wins_rate={gsummary['bn_wins_rate']:.2f}"
        )

    overall = summary["overall"]
    print(
        f"\n  OVERALL: bn_int_gain={overall['mean_bn_int_gain']:+.3f} "
        f"bl_int_gain={overall['mean_bl_int_gain']:+.3f} "
        f"bn_wins_rate={overall['bn_wins_rate']:.2f}"
    )

    out_path = Path(save_dir) / "intervention_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def _summarize_intervention(runs: list[dict]) -> dict:
    def _stats(items: list[dict]) -> dict:
        bn_gains = [r["bn_mean_int_gain"] for r in items]
        bl_gains = [r["bl_mean_int_gain"] for r in items]
        bn_wins = [r["intervention_helps_bn_more"] for r in items]
        return {
            "n_runs": len(items),
            "mean_bn_int_gain": float(np.mean(bn_gains)) if items else float("nan"),
            "mean_bl_int_gain": float(np.mean(bl_gains)) if items else float("nan"),
            "bn_wins_rate": float(np.mean(bn_wins)) if items else float("nan"),
        }

    by_graph_type = {
        gt: _stats([r for r in runs if r["graph_type"] == gt])
        for gt in sorted({r["graph_type"] for r in runs})
    }
    return {
        "overall": _stats(runs),
        "by_graph_type": by_graph_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-types", nargs="+",
        default=["chain", "fork", "random", "confounded"],
    )
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--T-obs", type=int, default=800)
    parser.add_argument("--T-int-per-target", type=int, default=200)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--d-bottleneck", type=int, default=12)
    parser.add_argument("--lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--baseline-methods", nargs="+", default=["lasso", "rrr"])
    parser.add_argument(
        "--baseline-alphas", nargs="+", type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    parser.add_argument("--rrr-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="results/intervention")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)

    if args.quick:
        args.graph_types = ["chain", "confounded"]
        args.K = 4
        args.T_obs = 400
        args.T_int_per_target = 100
        args.seeds = 2
        args.epochs = 30

    run_intervention_experiment(
        graph_types=args.graph_types,
        K=args.K,
        T_obs=args.T_obs,
        T_int_per_target=args.T_int_per_target,
        max_lag=args.max_lag,
        d_bottleneck=args.d_bottleneck,
        lambda_sparse=args.lambda_sparse,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        baseline_methods=args.baseline_methods,
        baseline_alphas=args.baseline_alphas,
        rrr_ranks=args.rrr_ranks,
        device=device,
        save_dir=args.save_dir,
    )
