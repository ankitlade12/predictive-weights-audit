"""
Experiment 24: Observational and Simulated Benchmark
=====================================================

Run selected bottleneck and baseline methods on observational datasets
with soft reference labels and on Lorenz-96 with an exact simulated graph.

Datasets:
    1. NOAA Climate Indices (ENSO, NAO, PDO, AMO, etc.) — real monthly
       observations from NOAA PSL, 70+ years. Ground truth: published
       teleconnection relationships.
    2. DREAM3 Gene Expression (E. coli size-10) — real gene regulatory
       network with known benchmark graph.
    3. Stock returns — 10 sector ETFs, real daily returns, evaluated as a
       heuristic soft-label sensitivity analysis without a ground-truth claim.
    4. Lorenz-96 — simulated nonlinear dynamics with an exact generating graph.

Methods:
    - Weight-Projection (bottleneck)
    - Lagged-Bottleneck (Sai's explicit lagged version)
    - Tuned Lasso (baseline)
    - Ridge (baseline)
    - PCMCI (actual causal discovery baseline)
    - Bivariate Granger (classical baseline)

Framing: This is hypothesis-generation / ranking stability analysis,
NOT a claim that any method "discovers" true causal structure on
observational data.

Usage:
    python experiments/exp24_real_data_benchmark.py --device cuda
    python experiments/exp24_real_data_benchmark.py --quick
    python experiments/exp24_real_data_benchmark.py --datasets climate finance
    python experiments/exp24_real_data_benchmark.py --methods PCMCI PCMCI+ Granger
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ssm_causal.bottleneck_training import extract_weight_score
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics

MetricEvaluator = Callable[[np.ndarray, np.ndarray, int | None], dict[str, object]]

METHOD_NAMES = (
    "SSM",
    "Lagged-Bottleneck",
    "Lasso",
    "Ridge",
    "PCMCI",
    "PCMCI+",
    "Granger",
    "DYNOTEARS",
    "VAR-LiNGAM",
)
DEFAULT_METHODS = ("SSM", "Lagged-Bottleneck", "Lasso", "Ridge", "PCMCI", "Granger")
DEFAULT_MODEL_SEEDS = (101, 102, 103)

DATASET_METADATA = {
    "climate": {
        "kind": "real_observational",
        "label_type": "soft_literature_labels",
        "provenance": (
            "NOAA monthly climate indices with literature-derived teleconnection labels"
        ),
    },
    "finance": {
        "kind": "real_observational",
        "label_type": "heuristic_soft_label_sensitivity",
        "provenance": (
            "Sector ETF daily log returns with a heuristic sector lead-lag sensitivity graph"
        ),
        "claim_scope": "sensitivity_analysis_only",
    },
    "dream3": {
        "kind": "real_observational",
        "label_type": "benchmark_graph",
        "provenance": "DREAM3 E. coli gene-expression benchmark",
    },
    "lorenz96": {
        "kind": "simulated",
        "label_type": "exact_graph",
        "provenance": "Lorenz-96 simulation with the exact generating interaction graph",
    },
}


@dataclass(frozen=True)
class MethodSpec:
    evaluator: MetricEvaluator
    hyperparameters: dict[str, object]
    replication_unit: str
    device: str
    stochastic: bool = False
    optional: bool = False


# Real data loaders


def load_climate_indices(data_dir: str = "data/real") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load REAL NOAA monthly climate indices (ENSO, NAO, PDO, AMO, SOI, PNA)."""
    from data.real_loaders import load_noaa_climate

    X, G, names = load_noaa_climate(data_dir=data_dir)
    print(f"  Loaded REAL NOAA climate data: T={X.shape[0]} months, K={X.shape[1]}")
    print(f"  Variables: {names}")
    print(f"  Ground truth edges: {int(G.sum())}")
    return X, G, names


def load_stock_returns(data_dir: str = "data/real") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load REAL Yahoo Finance sector ETF log returns."""
    from data.real_loaders import load_sector_etfs

    X, G, names = load_sector_etfs(data_dir=data_dir)
    print(f"  Loaded REAL sector ETF returns: T={X.shape[0]} days, K={X.shape[1]}")
    print(f"  Variables: {names}")
    print(f"  Ground truth edges: {int(G.sum())}")
    return X, G, names


def load_dream3_gene() -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    DREAM3 E. coli size-10 gene expression data.

    This is a well-known benchmark for causal discovery. Only loads if
    data files are available; otherwise returns None.
    """
    # Placeholder — DREAM3 requires manual download and preprocessing.
    # If files are present at data/dream3/, load them here.
    print("  [SKIP] DREAM3 not bundled — requires manual download")
    return None


def load_lorenz96_bench(K: int = 10, T: int = 1500) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Lorenz-96 with unambiguous ground truth (Tank et al., 2022)."""
    from data.real_loaders import load_lorenz96

    X, G, names = load_lorenz96(K=K, T=T, F=10.0, dt=0.01, subsample=10, seed=0)
    print(f"  Loaded Lorenz-96: T={X.shape[0]} steps, K={X.shape[1]}, F=10 (chaotic)")
    print(f"  Ground truth edges: {int(G.sum())} (each node driven by 3 neighbors)")
    return X, G, names


# Evaluation methods


def _validated_score_matrix(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError("non-finite score matrix")
    return score


def eval_weight_projection(X, G, d_model=64, epochs=150, device="cpu", seed=42):
    K = X.shape[1]
    torch.manual_seed(seed)
    model = MambaForTimeSeries(n_variables=K, d_model=d_model, d_state=16, n_layers=4)
    trainer = SSMTrainer(model, lr=1e-3, device=device)
    trainer.train(X, n_epochs=epochs, verbose=False)
    S = _validated_score_matrix(extract_weight_score(model))
    return compute_metrics(S, G)


def eval_lagged_bottleneck(
    X,
    G,
    max_lag=3,
    d_bottleneck=16,
    epochs=100,
    device="cpu",
    seed=42,
):
    from ssm_causal.lag_bottleneck import (
        LaggedBottleneckModel,
        LaggedBottleneckTrainer,
        LaggedBottleneckTrainingConfig,
        extract_lagged_weight_score,
        normalize_lagged_score,
    )
    from ssm_causal.lag_extraction import lag_score_to_aggregate

    K = X.shape[1]
    torch.manual_seed(seed)
    model = LaggedBottleneckModel(n_variables=K, max_lag=max_lag, d_bottleneck=d_bottleneck)
    config = LaggedBottleneckTrainingConfig(
        n_epochs=epochs,
        batch_size=min(64, X.shape[0] - max_lag),
    )
    trainer = LaggedBottleneckTrainer(model, config=config, device=device)
    trainer.train(X)

    S_lag = extract_lagged_weight_score(model)
    S_lag = normalize_lagged_score(S_lag, mode="global")
    S_lag = _validated_score_matrix(S_lag)
    S = _validated_score_matrix(lag_score_to_aggregate(S_lag))
    return compute_metrics(S, G)


def eval_lasso(X, G, max_lag=3):
    from sklearn.linear_model import MultiTaskLasso

    K, T = X.shape[1], X.shape[0]
    X_lag = np.zeros((T - max_lag, K * max_lag))
    Y = X[max_lag:]
    for tau in range(max_lag):
        X_lag[:, tau * K : (tau + 1) * K] = X[max_lag - tau - 1 : T - tau - 1]

    split = int(len(X_lag) * 0.6)
    val_end = int(len(X_lag) * 0.8)
    best_a, best_mse = 0.01, 1e10
    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        try:
            m = MultiTaskLasso(alpha=a, max_iter=5000)
            m.fit(X_lag[:split], Y[:split])
            mse = np.mean((m.predict(X_lag[split:val_end]) - Y[split:val_end]) ** 2)
            if mse < best_mse:
                best_mse, best_a = mse, a
        except Exception:
            pass
    lasso = MultiTaskLasso(alpha=best_a, max_iter=5000)
    lasso.fit(X_lag, Y)
    S = _validated_score_matrix(np.abs(lasso.coef_).reshape(K, max_lag, K).max(axis=1))
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    metrics = compute_metrics(S, G)
    metrics["selected_hyperparameters"] = {"alpha": best_a}
    return metrics


def eval_ridge(X, G, max_lag=3):
    from sklearn.linear_model import Ridge

    K, T = X.shape[1], X.shape[0]
    X_lag = np.zeros((T - max_lag, K * max_lag))
    Y = X[max_lag:]
    for tau in range(max_lag):
        X_lag[:, tau * K : (tau + 1) * K] = X[max_lag - tau - 1 : T - tau - 1]

    split = int(len(X_lag) * 0.6)
    val_end = int(len(X_lag) * 0.8)
    best_a, best_mse = 1.0, 1e10
    for a in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        try:
            m = Ridge(alpha=a)
            m.fit(X_lag[:split], Y[:split])
            mse = np.mean((m.predict(X_lag[split:val_end]) - Y[split:val_end]) ** 2)
            if mse < best_mse:
                best_mse, best_a = mse, a
        except Exception:
            pass
    ridge = Ridge(alpha=best_a)
    ridge.fit(X_lag, Y)
    S = _validated_score_matrix(np.abs(ridge.coef_).reshape(K, max_lag, K).max(axis=1))
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    metrics = compute_metrics(S, G)
    metrics["selected_hyperparameters"] = {"alpha": best_a}
    return metrics


def eval_pcmci(X, G, tau_max=5):
    from baselines.pcmci_wrapper import pcmci_causality_matrix

    S = _validated_score_matrix(pcmci_causality_matrix(X, tau_max=tau_max, pc_alpha=0.05))
    return compute_metrics(S, G)


def eval_pcmci_plus(X, G, tau_max=5):
    from baselines.pcmci_wrapper import pcmci_plus_causality_matrix

    S = _validated_score_matrix(pcmci_plus_causality_matrix(X, tau_max=tau_max, pc_alpha=0.05))
    return compute_metrics(S, G)


def eval_granger(X, G, max_lag=5):
    from baselines.granger import granger_causality_matrix

    S = _validated_score_matrix(granger_causality_matrix(X, max_lag=max_lag))
    return compute_metrics(S, G)


def eval_dynotears(X, G, max_lag=3):
    from baselines.dynotears import dynotears_causality_matrix

    S = _validated_score_matrix(dynotears_causality_matrix(X, max_lag=max_lag))
    return compute_metrics(S, G)


def eval_var_lingam(X, G, max_lag=3):
    from baselines.lingam_wrapper import var_lingam_causality_matrix

    S = _validated_score_matrix(var_lingam_causality_matrix(X, max_lag=max_lag))
    return compute_metrics(S, G)


def build_method_specs(device: str, epochs: int) -> dict[str, MethodSpec]:
    return {
        "SSM": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_weight_projection(
                X,
                G,
                d_model=64,
                epochs=epochs,
                device=device,
                seed=DEFAULT_MODEL_SEEDS[0] if seed is None else seed,
            ),
            hyperparameters={
                "d_model": 64,
                "d_state": 16,
                "n_layers": 4,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "epochs": epochs,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "extractor": "absolute_output_projection_times_input_projection",
            },
            replication_unit="model_initialization_seed",
            device=device,
            stochastic=True,
        ),
        "Lagged-Bottleneck": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_lagged_bottleneck(
                X,
                G,
                max_lag=3,
                d_bottleneck=16,
                epochs=epochs,
                device=device,
                seed=DEFAULT_MODEL_SEEDS[0] if seed is None else seed,
            ),
            hyperparameters={
                "max_lag": 3,
                "d_bottleneck": 16,
                "epochs": epochs,
                "batch_size": "min(64, T - max_lag)",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "lambda_sparse": 0.0,
                "score_normalization": "global",
            },
            replication_unit="model_initialization_seed",
            device=device,
            stochastic=True,
        ),
        "Lasso": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_lasso(X, G, max_lag=3),
            hyperparameters={
                "max_lag": 3,
                "alpha_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "max_iter": 5000,
                "train_fraction": 0.6,
                "validation_end_fraction": 0.8,
                "selection_metric": "validation_prediction_mse",
                "refit_scope": "all_lagged_rows",
            },
            replication_unit="deterministic_run",
            device="cpu",
        ),
        "Ridge": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_ridge(X, G, max_lag=3),
            hyperparameters={
                "max_lag": 3,
                "alpha_grid": [1e-2, 1e-1, 1.0, 10.0, 100.0],
                "train_fraction": 0.6,
                "validation_end_fraction": 0.8,
                "selection_metric": "validation_prediction_mse",
                "refit_scope": "all_lagged_rows",
            },
            replication_unit="deterministic_run",
            device="cpu",
        ),
        "PCMCI": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_pcmci(X, G, tau_max=3),
            hyperparameters={
                "tau_max": 3,
                "pc_alpha": 0.05,
                "independence_test": "parcorr",
                "significance": "analytic",
                "lag_aggregation": "maximum_absolute_test_statistic",
            },
            replication_unit="deterministic_run",
            device="cpu",
        ),
        "PCMCI+": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_pcmci_plus(X, G, tau_max=3),
            hyperparameters={
                "tau_min": 1,
                "tau_max": 3,
                "pc_alpha": 0.05,
                "independence_test": "parcorr",
                "significance": "analytic",
                "lag_aggregation": "maximum_absolute_test_statistic",
            },
            replication_unit="deterministic_run",
            device="cpu",
        ),
        "Granger": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_granger(X, G, max_lag=3),
            hyperparameters={
                "max_lag": 3,
                "significance": 0.05,
                "score_type": "neg_log_pvalue",
                "test": "bivariate_ssr_ftest",
            },
            replication_unit="deterministic_run",
            device="cpu",
        ),
        "DYNOTEARS": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_dynotears(X, G, max_lag=3),
            hyperparameters={
                "max_lag": 3,
                "lag_aggregation": "maximum_absolute_weight",
            },
            replication_unit="deterministic_run",
            device="cpu",
            optional=True,
        ),
        "VAR-LiNGAM": MethodSpec(
            evaluator=lambda X, G, seed=None: eval_var_lingam(X, G, max_lag=3),
            hyperparameters={"max_lag": 3, "random_state": 0},
            replication_unit="deterministic_run",
            device="cpu",
            optional=True,
        ),
    }


def build_method_registry(device: str, epochs: int, include_modern_baselines: bool = False):
    specs = build_method_specs(device=device, epochs=epochs)
    registry = {
        "Weight-Proj (SSM)": specs["SSM"].evaluator,
        "Lagged-Bottleneck": specs["Lagged-Bottleneck"].evaluator,
        "Tuned-Lasso": specs["Lasso"].evaluator,
        "Tuned-Ridge": specs["Ridge"].evaluator,
        "PCMCI": specs["PCMCI"].evaluator,
        "Granger": specs["Granger"].evaluator,
    }
    if include_modern_baselines:
        registry.update(
            {name: specs[name].evaluator for name in ("PCMCI+", "DYNOTEARS", "VAR-LiNGAM")}
        )
    return registry


def optional_method_unavailability(method_name: str) -> str | None:
    if method_name == "DYNOTEARS":
        if sys.version_info >= (3, 12):
            return (
                "DYNOTEARS requires CausalNex 0.12.1 in a separate legacy "
                "Python 3.10 environment; it is unsupported in the primary Python 3.12 environment"
            )
        try:
            available = importlib.util.find_spec("causalnex.structure.dynotears")
        except (ImportError, ModuleNotFoundError):
            available = None
        if available is None:
            return "DYNOTEARS requires the optional causalnex dependency"
    if method_name == "VAR-LiNGAM" and importlib.util.find_spec("lingam") is None:
        return "VAR-LiNGAM requires the optional lingam dependency"
    return None


def collect_edge_cards(datasets: list[str]) -> dict[str, dict[str, list[dict[str, str]]]]:
    cards = {}
    if "climate" in datasets:
        from data.real_loaders import load_noaa_edge_cards

        cards["climate"] = load_noaa_edge_cards(include_definitional=True)
    if "finance" in datasets:
        from data.real_loaders import load_sector_edge_cards

        cards["finance"] = load_sector_edge_cards()
    return cards


# Main experiment


def _git_metadata() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError:
        revision_bytes = b""
        diff_bytes = b""
        status_bytes = b""
    else:
        revision_bytes = revision.stdout if revision.returncode == 0 else b""
        diff_bytes = diff.stdout if diff.returncode == 0 else b""
        status_bytes = status.stdout if status.returncode == 0 else diff_bytes

    return {
        "code_revision": revision_bytes.decode().strip() or None,
        "worktree_dirty": bool(status_bytes),
        "worktree_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    package_names = (
        "numpy",
        "torch",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tigramite",
        "pandas",
        "causalnex",
        "lingam",
    )
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _source_tree_sha256(root: Path | None = None) -> str:
    source_root = root or Path(__file__).resolve().parents[1]
    excluded_parts = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        "build",
        "dist",
    }
    source_paths = sorted(
        path
        for path in source_root.rglob("*.py")
        if not excluded_parts.intersection(path.relative_to(source_root).parts)
    )

    digest = hashlib.sha256()
    for source_path in source_paths:
        relative_path = source_path.relative_to(source_root).as_posix().encode("utf-8")
        contents = source_path.read_bytes()
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(str(len(contents)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contents)
    return digest.hexdigest()


def _load_requested_datasets(
    datasets: list[str],
    climate_data_dir: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    loaded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    if "climate" in datasets:
        loaded["climate"] = load_climate_indices(data_dir=climate_data_dir)
    if "finance" in datasets:
        loaded["finance"] = load_stock_returns()
    if "dream3" in datasets:
        dream3 = load_dream3_gene()
        if dream3 is None:
            raise RuntimeError(
                "Requested dataset 'dream3' is unavailable; install and preprocess "
                "the DREAM3 benchmark files before running Exp24"
            )
        loaded["dream3"] = dream3
    if "lorenz96" in datasets:
        loaded["lorenz96"] = load_lorenz96_bench(K=10, T=1500)
    return loaded


def _selected_method_names(
    methods: list[str] | None,
    include_modern_baselines: bool,
) -> list[str]:
    if methods is None:
        selected = list(DEFAULT_METHODS)
        if include_modern_baselines:
            selected.extend(["PCMCI+", "DYNOTEARS", "VAR-LiNGAM"])
    else:
        selected = list(dict.fromkeys(methods))

    unknown = [name for name in selected if name not in METHOD_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown methods: {', '.join(unknown)}. Choose from: {', '.join(METHOD_NAMES)}"
        )
    return selected


def _validated_model_seeds(model_seeds: list[int] | None) -> list[int]:
    seeds = list(DEFAULT_MODEL_SEEDS if model_seeds is None else model_seeds)
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds
    ):
        raise ValueError("model_seeds must be a non-empty list of integer seeds")
    normalized = [int(seed) for seed in seeds]
    if len(set(normalized)) != len(normalized):
        raise ValueError("model_seeds must contain unique seeds")
    return normalized


def _null_metrics() -> dict[str, None]:
    return {"auroc": None, "auprc": None, "oracle_f1": None}


def _validated_metrics(metrics: dict[str, object]) -> dict[str, float]:
    selected = {
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "oracle_f1": float(metrics["f1"]),
    }
    non_finite = [name for name, value in selected.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(f"non-finite metrics: {', '.join(non_finite)}")
    return selected


def _skipped_record(spec: MethodSpec, reason: str) -> dict[str, object]:
    return {
        "status": "skipped",
        "error": reason,
        "metrics": _null_metrics(),
        "hyperparameters": spec.hyperparameters,
        "device": spec.device,
        "runtime_s": 0.0,
        "replication_unit": spec.replication_unit,
    }


def _resolved_hyperparameters(
    spec: MethodSpec,
    X: np.ndarray,
    evaluator_result: dict[str, object] | None = None,
) -> dict[str, object]:
    hyperparameters = dict(spec.hyperparameters)
    if hyperparameters.get("batch_size") == "min(64, T - max_lag)":
        hyperparameters["batch_size"] = min(
            64,
            X.shape[0] - int(hyperparameters["max_lag"]),
        )
    if evaluator_result is not None:
        selected_hyperparameters = evaluator_result.get("selected_hyperparameters")
        if selected_hyperparameters is not None:
            hyperparameters["selected"] = selected_hyperparameters
    return hyperparameters


def _evaluate_once(
    dataset_name: str,
    method_name: str,
    spec: MethodSpec,
    X: np.ndarray,
    G: np.ndarray,
    model_seed: int | None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    try:
        evaluator_result = spec.evaluator(X, G, model_seed)
        metrics = _validated_metrics(evaluator_result)
    except Exception as error:
        seed_context = "" if model_seed is None else f" at model seed {model_seed}"
        raise RuntimeError(
            f"{dataset_name} method {method_name} failed{seed_context}: {error}"
        ) from error

    return {
        "metrics": metrics,
        "hyperparameters": _resolved_hyperparameters(spec, X, evaluator_result),
        "runtime_s": time.perf_counter() - started_at,
    }


def _evaluate_method(
    dataset_name: str,
    method_name: str,
    spec: MethodSpec,
    X: np.ndarray,
    G: np.ndarray,
    model_seeds: list[int],
) -> dict[str, object]:
    if spec.optional:
        unavailable_reason = optional_method_unavailability(method_name)
        if unavailable_reason is not None:
            return _skipped_record(spec, unavailable_reason)

    if not spec.stochastic:
        evaluation = _evaluate_once(
            dataset_name,
            method_name,
            spec,
            X,
            G,
            model_seed=None,
        )
        return {
            "status": "ok",
            "error": None,
            **evaluation,
            "device": spec.device,
            "replication_unit": spec.replication_unit,
        }

    seed_evaluations = [
        (
            model_seed,
            _evaluate_once(
                dataset_name,
                method_name,
                spec,
                X,
                G,
                model_seed=model_seed,
            ),
        )
        for model_seed in model_seeds
    ]
    aggregate_metrics = {}
    for metric_name in ("auroc", "auprc", "oracle_f1"):
        values = np.asarray(
            [evaluation["metrics"][metric_name] for _, evaluation in seed_evaluations],
            dtype=float,
        )
        aggregate_metrics[f"{metric_name}_mean"] = float(values.mean())
        aggregate_metrics[f"{metric_name}_std"] = float(values.std(ddof=0))

    claims_note = (
        "Results are descriptive initialization stability only; model seeds "
        "cannot support competitiveness claims and are not independent "
        "uncertainty replicates."
    )
    hyperparameters = _resolved_hyperparameters(spec, X)
    hyperparameters["model_seeds"] = list(model_seeds)
    return {
        "status": "ok",
        "error": None,
        "metrics": aggregate_metrics,
        "per_seed": [
            {
                "model_seed": model_seed,
                "status": "ok",
                "error": None,
                "metrics": evaluation["metrics"],
                "runtime_s": evaluation["runtime_s"],
            }
            for model_seed, evaluation in seed_evaluations
        ],
        "hyperparameters": hyperparameters,
        "device": spec.device,
        "runtime_s": sum(evaluation["runtime_s"] for _, evaluation in seed_evaluations),
        "replication_unit": spec.replication_unit,
        "aggregation": {
            "unit": "model_initialization_seed",
            "std_ddof": 0,
            "interpretation": "descriptive_initialization_stability_only",
            "independent_uncertainty": False,
        },
        "competitiveness_claims_permitted": False,
        "competitiveness_claims_note": claims_note,
    }


def _dataset_payload(
    dataset_name: str,
    X: np.ndarray,
    G: np.ndarray,
    names: list[str],
    method_names: list[str],
    specs: dict[str, MethodSpec],
    model_seeds: list[int],
) -> dict[str, object]:
    K, T = X.shape[1], X.shape[0]
    metadata = DATASET_METADATA[dataset_name]
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    method_results: dict[str, dict[str, object]] = {}
    for method_name in method_names:
        record = _evaluate_method(
            dataset_name,
            method_name,
            specs[method_name],
            X_std,
            G,
            model_seeds,
        )
        method_results[method_name] = record
        if record["status"] == "ok":
            metrics = record["metrics"]
            if specs[method_name].stochastic:
                print(
                    f"  {method_name:22s}: AUROC={metrics['auroc_mean']:.3f}"
                    f"+/-{metrics['auroc_std']:.3f}  "
                    f"AUPRC={metrics['auprc_mean']:.3f}  "
                    f"Oracle-F1={metrics['oracle_f1_mean']:.3f}  "
                    f"({record['runtime_s']:.1f}s)"
                )
            else:
                print(
                    f"  {method_name:22s}: AUROC={metrics['auroc']:.3f}  "
                    f"AUPRC={metrics['auprc']:.3f}  "
                    f"Oracle-F1={metrics['oracle_f1']:.3f}  "
                    f"({record['runtime_s']:.1f}s)"
                )
        else:
            print(f"  {method_name:22s}: SKIPPED ({record['error']})")

    return {
        **metadata,
        "K": int(K),
        "T": int(T),
        "n_edges": int(G.sum()),
        "names": list(names),
        "data_sha256": _array_sha256(X),
        "graph_sha256": _array_sha256(G),
        "standardization": "per-variable mean zero and unit variance on full series",
        "methods": method_results,
    }


def _build_rankings(
    dataset_results: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    rankings = {}
    for dataset_name, dataset in dataset_results.items():
        rows = [
            {
                "method": method_name,
                "auroc": (
                    record["metrics"]["auroc_mean"]
                    if record["replication_unit"] == "model_initialization_seed"
                    else record["metrics"]["auroc"]
                ),
                "aggregation": (
                    "mean_across_model_seeds"
                    if record["replication_unit"] == "model_initialization_seed"
                    else "deterministic_run"
                ),
            }
            for method_name, record in dataset["methods"].items()
            if record["status"] == "ok"
        ]
        rankings[dataset_name] = sorted(rows, key=lambda row: row["auroc"], reverse=True)
    return rankings


def run_real_benchmark(
    datasets: list[str],
    device: str = "cpu",
    save_dir: str = "results/real_data_benchmark",
    climate_data_dir: str = "data/real",
    epochs: int = 150,
    include_modern_baselines: bool = False,
    methods: list[str] | None = None,
    model_seeds: list[int] | None = None,
):
    total_start = time.perf_counter()
    selected_methods = _selected_method_names(methods, include_modern_baselines)
    selected_model_seeds = _validated_model_seeds(model_seeds)
    specs = build_method_specs(device=device, epochs=epochs)

    print("=" * 72)
    print("EXPERIMENT 24: Observational and Simulated Benchmark")
    print(f"Datasets: {datasets} | Device: {device}")
    print(f"Methods: {selected_methods}")
    if any(specs[name].stochastic for name in selected_methods):
        print(f"Model seeds: {selected_model_seeds}")
    print("Protocol: locked extractors, tuned baselines, no graph-label tuning")
    print("=" * 72)

    loaded = _load_requested_datasets(datasets, climate_data_dir)
    dataset_results: dict[str, dict[str, object]] = {}

    for ds_name, (X, G, names) in loaded.items():
        K, T = X.shape[1], X.shape[0]
        n_edges = int(G.sum())
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name} (K={K}, T={T}, edges={n_edges})")
        print(f"  Variables: {', '.join(names[:10])}{'...' if len(names) > 10 else ''}")
        print(f"{'─' * 60}")

        dataset_results[ds_name] = _dataset_payload(
            ds_name,
            X,
            G,
            names,
            selected_methods,
            specs,
            selected_model_seeds,
        )

    total_time = time.perf_counter() - total_start
    rankings = _build_rankings(dataset_results)
    print(f"\n{'=' * 72}")
    print("SUMMARY: Observational and Simulated Benchmark")
    print(f"{'=' * 72}")
    print(f"{'Method':<22} " + " ".join(f"{ds:<12}" for ds in dataset_results))
    for method_name in selected_methods:
        row = f"{method_name:<22} "
        for dataset in dataset_results.values():
            record = dataset["methods"][method_name]
            if record["status"] == "ok":
                auroc = (
                    record["metrics"]["auroc_mean"]
                    if record["replication_unit"] == "model_initialization_seed"
                    else record["metrics"]["auroc"]
                )
                row += f"{auroc:<12.3f} "
            else:
                row += f"{'SKIPPED':<12} "
        print(row)

    print(f"\nTotal runtime: {total_time / 60:.1f} min")

    created_at_utc = datetime.now(timezone.utc).isoformat()
    argv = list(sys.argv)
    payload = {
        "schema_version": "1.0",
        "experiment": "exp24_real_data_benchmark",
        **_git_metadata(),
        "source_tree_sha256": _source_tree_sha256(),
        "command": shlex.join(argv),
        "argv": argv,
        "created_at": created_at_utc,
        "created_at_utc": created_at_utc,
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "device": device,
        "epochs": epochs,
        "runtime_s": total_time,
        "metric_definitions": {
            "auroc": {
                "description": "Area under the receiver operating characteristic curve",
                "oracle_tuned": False,
            },
            "auprc": {
                "description": "Area under the precision-recall curve",
                "oracle_tuned": False,
            },
            "oracle_f1": {
                "description": (
                    "Best F1 after selecting the score threshold against all graph labels"
                ),
                "oracle_tuned": True,
                "claim_restriction": "Descriptive only; not a deployable or validation metric",
            },
        },
        "claim_policy": {
            "single_seed_competitiveness_claims_permitted": False,
            "model_seed_interpretation": ("descriptive_initialization_stability_only"),
            "model_seeds_are_independent_replications": False,
            "model_seed_competitiveness_claims_permitted": False,
            "rankings_are_descriptive_only": True,
        },
        "stochastic_competitiveness_claims_permitted": False,
        "config": {
            "datasets": list(datasets),
            "methods": selected_methods,
            "climate_data_dir": climate_data_dir,
            "include_modern_baselines": include_modern_baselines,
            "model_seeds": selected_model_seeds,
        },
        "datasets": dataset_results,
        "rankings": rankings,
    }
    serialized_payload = json.dumps(payload, indent=2, allow_nan=False)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / "real_data_benchmark_results.json"
    out_path.write_text(serialized_payload + "\n")
    print(f"Saved to {out_path}")

    cards = collect_edge_cards(datasets)
    if cards:
        cards_path = Path(save_dir) / "edge_provenance_cards.json"
        cards_path.write_text(json.dumps(cards, indent=2, allow_nan=False) + "\n")
        print(f"Saved edge provenance cards to {cards_path}")

    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["climate", "finance", "lorenz96"],
        choices=["climate", "finance", "dream3", "lorenz96"],
    )
    parser.add_argument("--climate-data-dir", default="data/real")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="results/real_data_benchmark")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--modern-baselines", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES)
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_MODEL_SEEDS),
    )
    parser.add_argument("--quick", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()

    if args.quick:
        args.epochs = 20
        args.datasets = ["climate"]

    run_real_benchmark(
        datasets=args.datasets,
        device=args.device,
        save_dir=args.save_dir,
        climate_data_dir=args.climate_data_dir,
        epochs=args.epochs,
        include_modern_baselines=args.modern_baselines,
        methods=args.methods,
        model_seeds=args.model_seeds,
    )
