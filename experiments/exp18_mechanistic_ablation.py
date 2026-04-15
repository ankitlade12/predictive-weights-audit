"""
Experiment 18: Mechanistic ablation — *why* do bottleneck regimes fail?

Prior experiments (exp_gonogo, exp15) documented *when* the bottleneck breaks
(dense graphs, under-capacity, extreme noise). This experiment asks *why* by
ablating the architectural components that differ between the bottleneck and
linear baselines:

    1. Nonlinearity:   Does adding a ReLU/tanh hidden layer help or hurt?
    2. Bottleneck rank: Over- vs under-parameterized d relative to K*p.
    3. Sparsity prior:  Does L1 on endpoints help only in specific regimes?

For each ablation the experiment sweeps across regimes (noise, density, K, T)
and compares against tuned Lasso / RRR baselines, exactly as in exp15.

The goal is a table that tells us: for each failure mode, which architectural
knob is responsible, and whether any knob rescues the method.

Usage:
    python experiments/exp18_mechanistic_ablation.py
    python experiments/exp18_mechanistic_ablation.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
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
    LaggedBottleneckTrainingConfig,
    evaluate_lagged_scores,
    extract_lagged_weight_score,
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


# ── Model variants for ablation ──────────────────────────────────────────────


class LinearBottleneck(nn.Module):
    """Baseline: linear bottleneck (same as LaggedBottleneckModel)."""

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
    """Ablation: add a nonlinearity between input_proj and output_proj."""

    def __init__(
        self,
        n_variables: int,
        max_lag: int,
        d_bottleneck: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.n_variables = n_variables
        self.max_lag = max_lag
        self.d_bottleneck = d_bottleneck
        self.input_proj = nn.Linear(n_variables * max_lag, d_bottleneck)
        self.activation_name = activation
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.output_proj = nn.Linear(d_bottleneck, n_variables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.act(self.input_proj(x)))


class DeepBottleneck(nn.Module):
    """Ablation: two hidden layers between endpoints."""

    def __init__(self, n_variables: int, max_lag: int, d_bottleneck: int):
        super().__init__()
        self.n_variables = n_variables
        self.max_lag = max_lag
        self.d_bottleneck = d_bottleneck
        self.input_proj = nn.Linear(n_variables * max_lag, d_bottleneck)
        self.hidden = nn.Linear(d_bottleneck, d_bottleneck)
        self.output_proj = nn.Linear(d_bottleneck, n_variables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.input_proj(x))
        z = F.relu(self.hidden(z))
        return self.output_proj(z)


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
    """Extract |W_out @ W_in_tau| for any model with input_proj/output_proj."""
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


# ── Regime data generators ───────────────────────────────────────────────────


def generate_regime_data(
    regime: str,
    K: int,
    T: int,
    max_lag: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate data for a named regime condition."""
    rng = np.random.RandomState(seed)

    if regime.startswith("noise="):
        noise_std = float(regime.split("=")[1])
        A_lags = _make_chain_lags(K, max_lag, rng)
        A_lags = ensure_stable_lag_coefficients(A_lags)
        X = simulate_lagged_var(A_lags, T=T, noise_std=noise_std, seed=seed)
        return X, lag_graph_from_coefficients(A_lags), f"chain-noise{noise_std}"

    if regime.startswith("density="):
        density = float(regime.split("=")[1])
        A_lags = _make_random_lags(K, max_lag, density, rng)
        A_lags = ensure_stable_lag_coefficients(A_lags)
        X = simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
        return X, lag_graph_from_coefficients(A_lags), f"random-density{density}"

    if regime.startswith("nonlinear="):
        strength = float(regime.split("=")[1])
        return _generate_nonlinear_regime(K, T, max_lag, strength, seed)

    raise ValueError(f"Unknown regime: {regime}")


def _make_chain_lags(K: int, max_lag: int, rng: np.random.RandomState) -> np.ndarray:
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.35] * K)
    for i in range(1, K):
        A_lags[0, i, i - 1] = 0.45
    if max_lag > 2:
        A_lags[2, min(K - 1, 2), 0] = 0.40
    return A_lags


def _make_random_lags(
    K: int, max_lag: int, density: float, rng: np.random.RandomState,
) -> np.ndarray:
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.30] * K)
    n_off = K * (K - 1)
    n_edges = max(1, int(density * n_off))
    pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
    chosen = rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False)
    active_lags = [0] if max_lag == 1 else [0, min(2, max_lag - 1)]
    for idx in chosen:
        i, j = pairs[idx]
        lag = rng.choice(active_lags)
        A_lags[lag, i, j] = rng.uniform(0.25, 0.55) * rng.choice([-1, 1])
    return A_lags


def _generate_nonlinear_regime(
    K: int, T: int, max_lag: int, strength: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.RandomState(seed)
    X = np.zeros((T + 300, K))
    X[:max_lag] = rng.randn(max_lag, K) * 0.1

    for t in range(max_lag, T + 300):
        value = 0.35 * X[t - 1]
        value[1 % K] += strength * np.tanh(X[t - min(3, max_lag), 0])
        if K > 3:
            value[3 % K] += strength * np.tanh(X[t - min(4, max_lag), 2 % K])
        X[t] = value + rng.randn(K) * 0.08

    G_lag = np.zeros((max_lag, K, K))
    if max_lag > 2:
        G_lag[min(2, max_lag - 1), 1 % K, 0] = 1.0
    if K > 3 and max_lag > 3:
        G_lag[min(3, max_lag - 1), 3 % K, 2 % K] = 1.0
    return standardize(X[300:]), G_lag, f"nonlinear-strength{strength}"


# ── Training loop ────────────────────────────────────────────────────────────


def train_and_evaluate_variant(
    variant_name: str,
    model: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    G_lag: np.ndarray,
    max_lag: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lambda_sparse: float,
    device: str,
) -> dict:
    model = model.to(device)
    features, targets = make_lagged_dataset(X_train, max_lag, device=device)
    n_examples = features.shape[0]
    steps_per_epoch = max(1, n_examples // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    # Prediction MSE on test
    test_feat, test_tgt = make_lagged_dataset(X_test, max_lag, device=device)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_feat)
        metrics["test_prediction_mse"] = float(F.mse_loss(test_pred, test_tgt).item())

    metrics["variant"] = variant_name
    return metrics


# ── Main grid ────────────────────────────────────────────────────────────────


def run_mechanistic_ablation(
    regimes: list[str],
    K: int,
    T: int,
    max_lag: int,
    d_bottleneck_values: list[int],
    lambda_sparse_values: list[float],
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

    grid = list(product(regimes, d_bottleneck_values, lambda_sparse_values))
    print("=" * 80)
    print("EXPERIMENT 18: Mechanistic Ablation")
    print(
        f"Regimes: {len(regimes)} | d_bottleneck: {d_bottleneck_values} | "
        f"lambda_sparse: {lambda_sparse_values} | Seeds: {seeds}"
    )
    print("=" * 80)

    for regime, d_bn, lam_sp in grid:
        for seed_idx in range(seeds):
            seed = 30_000 + seed_idx
            set_seed(seed)

            X, G_lag, desc = generate_regime_data(regime, K, T, max_lag, seed)
            G_lag = pad_lag_graph(G_lag, max_lag)
            X_train, X_test = train_test_split_time_series(X, 0.8, max_lag)

            methods: dict[str, dict] = {}

            # Variant 1: Linear bottleneck (baseline architecture)
            set_seed(seed)
            model_lin = LinearBottleneck(K, max_lag, d_bn)
            methods["Linear-BN"] = train_and_evaluate_variant(
                "Linear-BN", model_lin, X_train, X_test, G_lag,
                max_lag, epochs, batch_size, 1e-3, 1e-4, lam_sp, device,
            )

            # Variant 2: ReLU bottleneck
            set_seed(seed)
            model_relu = NonlinearBottleneck(K, max_lag, d_bn, activation="relu")
            methods["ReLU-BN"] = train_and_evaluate_variant(
                "ReLU-BN", model_relu, X_train, X_test, G_lag,
                max_lag, epochs, batch_size, 1e-3, 1e-4, lam_sp, device,
            )

            # Variant 3: Tanh bottleneck
            set_seed(seed)
            model_tanh = NonlinearBottleneck(K, max_lag, d_bn, activation="tanh")
            methods["Tanh-BN"] = train_and_evaluate_variant(
                "Tanh-BN", model_tanh, X_train, X_test, G_lag,
                max_lag, epochs, batch_size, 1e-3, 1e-4, lam_sp, device,
            )

            # Variant 4: Deep bottleneck (2-layer nonlinear)
            set_seed(seed)
            model_deep = DeepBottleneck(K, max_lag, d_bn)
            methods["Deep-BN"] = train_and_evaluate_variant(
                "Deep-BN", model_deep, X_train, X_test, G_lag,
                max_lag, epochs, batch_size, 1e-3, 1e-4, lam_sp, device,
            )

            # Classical baselines (Lasso, RRR)
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

            # Pick best bottleneck variant and best baseline
            bn_variants = {k: v for k, v in methods.items() if k.endswith("-BN")}
            bl_variants = {k: v for k, v in methods.items() if not k.endswith("-BN")}
            best_bn_name = max(bn_variants, key=lambda k: bn_variants[k]["flat_lag_auroc"])
            best_bl_name = max(bl_variants, key=lambda k: bl_variants[k]["flat_lag_auroc"])

            delta = methods[best_bn_name]["flat_lag_auroc"] - methods[best_bl_name]["flat_lag_auroc"]

            run = {
                "regime": regime,
                "description": desc,
                "K": K,
                "T": T,
                "max_lag": max_lag,
                "d_bottleneck": d_bn,
                "lambda_sparse": lam_sp,
                "seed": seed_idx,
                "methods": methods,
                "best_bottleneck_variant": best_bn_name,
                "best_baseline": best_bl_name,
                "flat_auroc_delta": float(delta),
            }
            runs.append(run)

            print(
                f"{regime:22s} d={d_bn:<3d} lam={lam_sp:.0e} seed={seed_idx} | "
                f"best_bn={best_bn_name:8s} auroc={methods[best_bn_name]['flat_lag_auroc']:.3f} "
                f"best_bl={best_bl_name:5s} auroc={methods[best_bl_name]['flat_lag_auroc']:.3f} "
                f"delta={delta:+.3f}"
            )

    summary = _summarize(runs)
    results = {
        "metadata": {
            "regimes": regimes,
            "K": K,
            "T": T,
            "max_lag": max_lag,
            "d_bottleneck_values": d_bottleneck_values,
            "lambda_sparse_values": lambda_sparse_values,
            "seeds": seeds,
            "epochs": epochs,
            "device": device,
            "runtime_seconds": time.time() - start_time,
        },
        "summary": summary,
        "runs": runs,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: Which architectural knobs matter per regime?")
    print("=" * 80)
    for regime, rsummary in summary["by_regime"].items():
        print(
            f"  {regime:22s} | "
            f"mean_delta={rsummary['mean_flat_delta']:+.3f} | "
            f"best_variant_mode={rsummary['most_common_best_variant']} | "
            f"bn_win_rate={rsummary['bn_win_rate']:.2f}"
        )

    out_path = Path(save_dir) / "mechanistic_ablation_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def _summarize(runs: list[dict]) -> dict:
    def _stats(items: list[dict]) -> dict:
        deltas = [r["flat_auroc_delta"] for r in items]
        bn_wins = [r["flat_auroc_delta"] > 0.005 for r in items]
        variants = [r["best_bottleneck_variant"] for r in items]
        from collections import Counter
        variant_counts = Counter(variants)
        most_common = variant_counts.most_common(1)[0][0] if variant_counts else "N/A"
        return {
            "n_runs": len(items),
            "mean_flat_delta": float(np.mean(deltas)) if items else float("nan"),
            "bn_win_rate": float(np.mean(bn_wins)) if items else float("nan"),
            "most_common_best_variant": most_common,
            "variant_counts": dict(variant_counts),
        }

    by_regime = {
        regime: _stats([r for r in runs if r["regime"] == regime])
        for regime in sorted({r["regime"] for r in runs})
    }
    by_d = {
        str(d): _stats([r for r in runs if r["d_bottleneck"] == d])
        for d in sorted({r["d_bottleneck"] for r in runs})
    }
    by_lambda = {
        str(lam): _stats([r for r in runs if r["lambda_sparse"] == lam])
        for lam in sorted({r["lambda_sparse"] for r in runs})
    }
    return {
        "overall": _stats(runs),
        "by_regime": by_regime,
        "by_d_bottleneck": by_d,
        "by_lambda_sparse": by_lambda,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=[
            "noise=0.05", "noise=0.2", "noise=0.5", "noise=1.0",
            "density=0.1", "density=0.3", "density=0.6",
            "nonlinear=0.3", "nonlinear=0.6", "nonlinear=1.0",
        ],
    )
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--T", type=int, default=800)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--d-bottleneck-values", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--lambda-sparse-values", nargs="+", type=float, default=[0.0, 1e-4, 1e-3])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-methods", nargs="+", default=["lasso", "rrr"])
    parser.add_argument(
        "--baseline-alphas", nargs="+", type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    parser.add_argument("--rrr-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="results/mechanistic_ablation")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)

    if args.quick:
        args.regimes = ["noise=0.2", "density=0.3", "nonlinear=0.6"]
        args.d_bottleneck_values = [8]
        args.lambda_sparse_values = [0.0]
        args.seeds = 1
        args.epochs = 30
        args.K = 5
        args.T = 400

    run_mechanistic_ablation(
        regimes=args.regimes,
        K=args.K,
        T=args.T,
        max_lag=args.max_lag,
        d_bottleneck_values=args.d_bottleneck_values,
        lambda_sparse_values=args.lambda_sparse_values,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        baseline_methods=args.baseline_methods,
        baseline_alphas=args.baseline_alphas,
        rrr_ranks=args.rrr_ranks,
        device=device,
        save_dir=args.save_dir,
    )
