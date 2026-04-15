"""
Experiment 20: Distribution shift — does shift amplify SSM vs linear gap?

Aman's suggestion: if observational data yields parity between bottleneck
and linear baselines, distribution shifts may break that symmetry.

The experiment trains each method on a *source* distribution and evaluates
graph recovery on a *target* distribution that has shifted in one of:

    1. Noise shift:      noise_std changes between train and test.
    2. Coefficient shift: causal strengths scale up or down.
    3. Structural shift:  edges are added or removed between train and test.

For each shift type we measure how much graph-recovery degrades and whether
the bottleneck or linear baselines degrade less (i.e., are more robust).

If bottleneck models are more robust under shift, this supports the
"representation bottleneck" framing: the learned low-rank projection
generalises better than coefficient-fitting.

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
    simulate_lagged_var,
    standardize,
    train_test_split_time_series,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ── Shift generators ─────────────────────────────────────────────────────────


def make_base_coefficients(
    K: int, max_lag: int, seed: int,
) -> np.ndarray:
    """Chain-plus-delayed graph used as the base causal structure."""
    rng = np.random.RandomState(seed)
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.30] * K)
    for i in range(1, K):
        A_lags[0, i, i - 1] = 0.45
    if max_lag > 2 and K > 2:
        A_lags[2, 2, 0] = 0.40
    if max_lag > 3 and K > 4:
        A_lags[3, 4, 1] = 0.35
    return ensure_stable_lag_coefficients(A_lags)


def apply_noise_shift(
    A_lags: np.ndarray,
    source_noise: float,
    target_noise: float,
    K: int,
    T: int,
    max_lag: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Same graph, different noise levels."""
    X_source = standardize(
        simulate_lagged_var(A_lags, T=T, noise_std=source_noise, seed=seed)
    )
    X_target = standardize(
        simulate_lagged_var(A_lags, T=T, noise_std=target_noise, seed=seed + 7000)
    )
    G_lag = lag_graph_from_coefficients(A_lags)
    desc = f"noise-shift({source_noise}->{target_noise})"
    return X_source, X_target, G_lag, G_lag, desc


def apply_coefficient_shift(
    A_lags: np.ndarray,
    scale_factor: float,
    K: int,
    T: int,
    max_lag: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Same graph topology, but off-diagonal strengths are scaled."""
    X_source = standardize(
        simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
    )

    A_shifted = A_lags.copy()
    for lag in range(max_lag):
        diag = np.diag(A_shifted[lag]).copy()
        A_shifted[lag] *= scale_factor
        np.fill_diagonal(A_shifted[lag], diag)  # keep autoregressive part
    A_shifted = ensure_stable_lag_coefficients(A_shifted)

    X_target = standardize(
        simulate_lagged_var(A_shifted, T=T, noise_std=0.1, seed=seed + 7000)
    )
    G_source = lag_graph_from_coefficients(A_lags)
    G_target = lag_graph_from_coefficients(A_shifted)
    desc = f"coeff-shift(scale={scale_factor:.1f})"
    return X_source, X_target, G_source, G_target, desc


def apply_structural_shift(
    A_lags: np.ndarray,
    n_edges_to_flip: int,
    K: int,
    T: int,
    max_lag: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Add or remove edges between source and target."""
    rng = np.random.RandomState(seed + 9000)

    X_source = standardize(
        simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
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
    X_target = standardize(
        simulate_lagged_var(A_shifted, T=T, noise_std=0.1, seed=seed + 7000)
    )
    G_source = lag_graph_from_coefficients(A_lags)
    G_target = lag_graph_from_coefficients(A_shifted)
    desc = f"struct-shift(flipped={n_edges_to_flip})"
    return X_source, X_target, G_source, G_target, desc


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


def train_bottleneck_on_source_eval_on_target(
    model: nn.Module,
    X_source: np.ndarray,
    X_target: np.ndarray,
    G_lag_source: np.ndarray,
    G_lag_target: np.ndarray,
    max_lag: int,
    epochs: int,
    batch_size: int,
    lambda_sparse: float,
    device: str,
) -> dict:
    """Train on source, extract scores, evaluate graph recovery on both."""
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
    source_metrics = evaluate_lagged_scores(S_lag, G_lag_source)
    target_metrics = evaluate_lagged_scores(S_lag, G_lag_target)

    # Prediction MSE on target
    tgt_feat, tgt_tgt = make_lagged_dataset(X_target, max_lag, device=device)
    model.eval()
    with torch.no_grad():
        target_metrics["target_prediction_mse"] = float(
            F.mse_loss(model(tgt_feat), tgt_tgt).item()
        )

    return {
        "source_flat_lag_auroc": source_metrics["flat_lag_auroc"],
        "source_aggregate_auroc": source_metrics["aggregate_auroc"],
        "target_flat_lag_auroc": target_metrics["flat_lag_auroc"],
        "target_aggregate_auroc": target_metrics["aggregate_auroc"],
        "target_prediction_mse": target_metrics["target_prediction_mse"],
        "degradation": source_metrics["flat_lag_auroc"] - target_metrics["flat_lag_auroc"],
    }


# ── Main grid ────────────────────────────────────────────────────────────────


SHIFT_CONFIGS = {
    "noise-low": ("noise", {"source_noise": 0.1, "target_noise": 0.3}),
    "noise-high": ("noise", {"source_noise": 0.1, "target_noise": 0.8}),
    "coeff-weaken": ("coefficient", {"scale_factor": 0.5}),
    "coeff-strengthen": ("coefficient", {"scale_factor": 1.8}),
    "struct-small": ("structural", {"n_edges_to_flip": 2}),
    "struct-large": ("structural", {"n_edges_to_flip": 5}),
}


def run_distribution_shift_experiment(
    shift_names: list[str],
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

    print("=" * 80)
    print("EXPERIMENT 20: Distribution Shift")
    print(
        f"Shifts: {shift_names} | K={K} T={T} | Seeds: {seeds}"
    )
    print("=" * 80)

    for shift_name in shift_names:
        shift_type, shift_params = SHIFT_CONFIGS[shift_name]

        for seed_idx in range(seeds):
            seed = 50_000 + seed_idx
            set_seed(seed)

            A_lags = make_base_coefficients(K, max_lag, seed)

            # Generate source/target data for this shift
            if shift_type == "noise":
                X_source, X_target, G_source, G_target, desc = apply_noise_shift(
                    A_lags, K=K, T=T, max_lag=max_lag, seed=seed, **shift_params,
                )
            elif shift_type == "coefficient":
                X_source, X_target, G_source, G_target, desc = apply_coefficient_shift(
                    A_lags, K=K, T=T, max_lag=max_lag, seed=seed, **shift_params,
                )
            elif shift_type == "structural":
                X_source, X_target, G_source, G_target, desc = apply_structural_shift(
                    A_lags, K=K, T=T, max_lag=max_lag, seed=seed, **shift_params,
                )
            else:
                raise ValueError(f"Unknown shift_type: {shift_type}")

            G_source = pad_lag_graph(G_source, max_lag)
            G_target = pad_lag_graph(G_target, max_lag)

            methods: dict[str, dict] = {}

            # Linear bottleneck
            set_seed(seed)
            model_lin = LinearBottleneck(K, max_lag, d_bottleneck)
            methods["Linear-BN"] = train_bottleneck_on_source_eval_on_target(
                model_lin, X_source, X_target, G_source, G_target,
                max_lag, epochs, batch_size, lambda_sparse, device,
            )

            # Nonlinear bottleneck
            set_seed(seed)
            model_nl = NonlinearBottleneck(K, max_lag, d_bottleneck)
            methods["Nonlinear-BN"] = train_bottleneck_on_source_eval_on_target(
                model_nl, X_source, X_target, G_source, G_target,
                max_lag, epochs, batch_size, lambda_sparse, device,
            )

            # Classical baselines — train on source, evaluate on both
            X_src_train, _ = train_test_split_time_series(X_source, 0.8, max_lag)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for bmethod in baseline_methods:
                    baseline = fit_tuned_lagged_linear_baseline(
                        X_src_train, max_lag, method=bmethod,
                        alpha_values=baseline_alphas,
                        rank_values=rrr_ranks,
                        validation_fraction=0.25,
                    )
                    source_m = evaluate_lagged_scores(baseline.score, G_source)
                    target_m = evaluate_lagged_scores(baseline.score, G_target)
                    methods[bmethod.upper()] = {
                        "source_flat_lag_auroc": source_m["flat_lag_auroc"],
                        "source_aggregate_auroc": source_m["aggregate_auroc"],
                        "target_flat_lag_auroc": target_m["flat_lag_auroc"],
                        "target_aggregate_auroc": target_m["aggregate_auroc"],
                        "degradation": source_m["flat_lag_auroc"] - target_m["flat_lag_auroc"],
                    }

            # Who degrades less?
            bn_degs = [methods[k]["degradation"] for k in methods if k.endswith("-BN")]
            bl_degs = [methods[k]["degradation"] for k in methods if not k.endswith("-BN")]
            mean_bn_deg = float(np.mean(bn_degs))
            mean_bl_deg = float(np.mean(bl_degs))

            run = {
                "shift_name": shift_name,
                "shift_type": shift_type,
                "description": desc,
                "K": K,
                "T": T,
                "max_lag": max_lag,
                "seed": seed_idx,
                "methods": methods,
                "mean_bn_degradation": mean_bn_deg,
                "mean_bl_degradation": mean_bl_deg,
                "bn_more_robust": bool(mean_bn_deg < mean_bl_deg - 0.005),
            }
            runs.append(run)

            print(
                f"{shift_name:18s} seed={seed_idx} | "
                f"BN_degrade={mean_bn_deg:+.3f} "
                f"BL_degrade={mean_bl_deg:+.3f} "
                f"{'BN-ROBUST' if mean_bn_deg < mean_bl_deg - 0.005 else 'NO-DIFF'}"
            )

    summary = _summarize_shift(runs)
    results = {
        "metadata": {
            "shift_names": shift_names,
            "K": K,
            "T": T,
            "max_lag": max_lag,
            "d_bottleneck": d_bottleneck,
            "lambda_sparse": lambda_sparse,
            "seeds": seeds,
            "epochs": epochs,
            "device": device,
            "runtime_seconds": time.time() - start_time,
        },
        "summary": summary,
        "runs": runs,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: Robustness under distribution shift")
    print("=" * 80)
    for shift_name, ssummary in summary["by_shift"].items():
        print(
            f"  {shift_name:18s} | "
            f"bn_degrade={ssummary['mean_bn_degradation']:+.3f} "
            f"bl_degrade={ssummary['mean_bl_degradation']:+.3f} "
            f"bn_robust_rate={ssummary['bn_robust_rate']:.2f}"
        )

    overall = summary["overall"]
    print(
        f"\n  OVERALL: bn_degrade={overall['mean_bn_degradation']:+.3f} "
        f"bl_degrade={overall['mean_bl_degradation']:+.3f} "
        f"bn_robust_rate={overall['bn_robust_rate']:.2f}"
    )

    verdict = (
        "Bottleneck is more robust under shift"
        if overall["bn_robust_rate"] > 0.5
        else "No clear robustness advantage for bottleneck"
    )
    print(f"\n  VERDICT: {verdict}")

    out_path = Path(save_dir) / "distribution_shift_results.json"
    with out_path.open("w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


def _summarize_shift(runs: list[dict]) -> dict:
    def _stats(items: list[dict]) -> dict:
        bn_degs = [r["mean_bn_degradation"] for r in items]
        bl_degs = [r["mean_bl_degradation"] for r in items]
        bn_robust = [r["bn_more_robust"] for r in items]
        return {
            "n_runs": len(items),
            "mean_bn_degradation": float(np.mean(bn_degs)) if items else float("nan"),
            "mean_bl_degradation": float(np.mean(bl_degs)) if items else float("nan"),
            "bn_robust_rate": float(np.mean(bn_robust)) if items else float("nan"),
        }

    by_shift = {
        sn: _stats([r for r in runs if r["shift_name"] == sn])
        for sn in sorted({r["shift_name"] for r in runs})
    }
    by_type = {
        st: _stats([r for r in runs if r["shift_type"] == st])
        for st in sorted({r["shift_type"] for r in runs})
    }
    return {
        "overall": _stats(runs),
        "by_shift": by_shift,
        "by_shift_type": by_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shift-names", nargs="+",
        default=list(SHIFT_CONFIGS.keys()),
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
        "--baseline-alphas", nargs="+", type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    parser.add_argument("--rrr-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="results/distribution_shift")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)

    if args.quick:
        args.shift_names = ["noise-high", "coeff-weaken", "struct-small"]
        args.K = 4
        args.T = 400
        args.seeds = 2
        args.epochs = 30

    run_distribution_shift_experiment(
        shift_names=args.shift_names,
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
