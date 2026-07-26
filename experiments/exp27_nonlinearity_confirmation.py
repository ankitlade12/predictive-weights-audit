"""Pre-specified confirmation of bottleneck recovery under mild nonlinearity."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import wilcoxon

from experiments.exp18_mechanistic_ablation import (
    LinearBottleneck,
    set_seed,
    train_and_evaluate_variant,
)
from ssm_causal.lag_bottleneck import (
    evaluate_lagged_scores,
    fit_tuned_lagged_linear_baseline,
)
from ssm_causal.statistics import holm_correction, paired_bootstrap_ci

PRIMARY_MODEL_NAME = "Linear-BN"
PRIMARY_COMPARATOR_NAME = "Lasso"
D_BOTTLENECK = 32
LAMBDA_SPARSE = 1e-3
LASSO_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MILD_STRENGTHS = (0.2, 0.3, 0.4)
EDGE_COEFFICIENT = 0.55
SELF_COEFFICIENT = 0.35
NOISE_STD = 0.08
GENERATOR_BURNIN = 300
PREPROCESSING_SCALE_FLOOR = 1e-8
GAUSS_HERMITE_ORDER = 64

_REQUIRED_METRICS = (
    "flat_lag_auroc",
    "flat_lag_auprc",
    "aggregate_auroc",
    "aggregate_auprc",
    "prediction_mse",
)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        return "mps"
    return "cpu"


def calibrated_nonlinear_transform(
    values: np.ndarray | float,
    nonlinearity_mix: float,
) -> tuple[np.ndarray, dict]:
    """Calibrate transform variance under the stationary Gaussian AR source."""
    if not 0.0 <= nonlinearity_mix <= 1.0:
        raise ValueError("nonlinearity_mix must lie in [0, 1].")

    stationary_variance = NOISE_STD**2 / (1.0 - SELF_COEFFICIENT**2)
    stationary_std = float(np.sqrt(stationary_variance))
    nodes, weights = np.polynomial.hermite.hermgauss(GAUSS_HERMITE_ORDER)
    quadrature_values = np.sqrt(2.0) * stationary_std * nodes
    unscaled_quadrature = (
        1.0 - nonlinearity_mix
    ) * quadrature_values + nonlinearity_mix * stationary_std * np.tanh(
        quadrature_values / stationary_std
    )
    normalized_weights = weights / np.sqrt(np.pi)
    unscaled_mean = float(np.sum(normalized_weights * unscaled_quadrature))
    unscaled_variance = float(
        np.sum(normalized_weights * (unscaled_quadrature - unscaled_mean) ** 2)
    )
    unscaled_std = float(np.sqrt(unscaled_variance))
    scale = stationary_std / unscaled_std

    input_values = np.asarray(values, dtype=float)
    unscaled_values = (
        1.0 - nonlinearity_mix
    ) * input_values + nonlinearity_mix * stationary_std * np.tanh(input_values / stationary_std)
    transformed = scale * unscaled_values
    calibration = {
        "distribution": "stationary_gaussian_ar1",
        "method": "gauss_hermite_quadrature",
        "quadrature_order": GAUSS_HERMITE_ORDER,
        "stationary_mean": 0.0,
        "stationary_variance": stationary_variance,
        "stationary_std": stationary_std,
        "innovation_std": NOISE_STD,
        "ar_coefficient": SELF_COEFFICIENT,
        "target_std": stationary_std,
        "unscaled_std": unscaled_std,
        "scale": scale,
        "calibrated_std": scale * unscaled_std,
        "independent_of_experiment_data_and_seeds": True,
    }
    return transformed, calibration


def generate_controlled_nonlinear_data(
    *,
    K: int,
    T: int,
    max_lag: int,
    nonlinearity_mix: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate raw data while varying only the planted edges' functional form."""
    if K < 2:
        raise ValueError("K must be at least 2.")
    if max_lag < 1 or T <= max_lag:
        raise ValueError("T must be greater than a positive max_lag.")
    if not 0.0 <= nonlinearity_mix <= 1.0:
        raise ValueError("nonlinearity_mix must lie in [0, 1].")

    planted_edges = [
        {
            "source": 0,
            "target": 1,
            "lag": min(3, max_lag),
            "coefficient": EDGE_COEFFICIENT,
        }
    ]
    if K >= 4:
        planted_edges.append(
            {
                "source": 2,
                "target": 3,
                "lag": min(4, max_lag),
                "coefficient": EDGE_COEFFICIENT,
            }
        )

    rng = np.random.RandomState(seed)
    _, amplitude_calibration = calibrated_nonlinear_transform(
        np.asarray([0.0]),
        nonlinearity_mix,
    )
    transform_scale = amplitude_calibration["scale"]
    stationary_std = amplitude_calibration["stationary_std"]
    total_length = T + GENERATOR_BURNIN
    X = np.zeros((total_length, K), dtype=float)
    X[:max_lag] = rng.normal(scale=NOISE_STD, size=(max_lag, K))
    for time_index in range(max_lag, total_length):
        value = SELF_COEFFICIENT * X[time_index - 1]
        for edge in planted_edges:
            source_value = X[
                time_index - edge["lag"],
                edge["source"],
            ]
            transformed = transform_scale * (
                (1.0 - nonlinearity_mix) * source_value
                + nonlinearity_mix * stationary_std * np.tanh(source_value / stationary_std)
            )
            value[edge["target"]] += edge["coefficient"] * transformed
        X[time_index] = value + rng.normal(scale=NOISE_STD, size=K)

    graph = np.zeros((max_lag, K, K), dtype=float)
    for edge in planted_edges:
        graph[edge["lag"] - 1, edge["target"], edge["source"]] = 1.0
    mechanism = {
        "name": "fixed_coefficient_functional_form_interpolation",
        "base_functional_form": "(1-s)*x + s*sigma*tanh(x/sigma)",
        "functional_form": ("a(s) * ((1-s)*x + s*sigma*tanh(x/sigma))"),
        "nonlinearity_mix": float(nonlinearity_mix),
        "amplitude_calibration": amplitude_calibration,
        "edge_coefficient": EDGE_COEFFICIENT,
        "self_coefficient": SELF_COEFFICIENT,
        "noise_std": NOISE_STD,
        "burnin": GENERATOR_BURNIN,
        "planted_edges": planted_edges,
    }
    return X[GENERATOR_BURNIN:], graph, mechanism


def split_and_standardize_training_only(
    X: np.ndarray,
    *,
    max_lag: int,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split raw time series, then fit and apply scaling from training rows only."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("X must have shape [T, K].")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1).")

    minimum_rows = max_lag + 2
    split_index = int(len(values) * train_fraction)
    split_index = min(
        max(split_index, minimum_rows),
        len(values) - minimum_rows,
    )
    raw_train = values[:split_index]
    raw_test = values[split_index:]
    training_mean = raw_train.mean(axis=0, keepdims=True)
    training_std = raw_train.std(axis=0, keepdims=True)
    training_scale = np.maximum(training_std, PREPROCESSING_SCALE_FLOOR)
    train = (raw_train - training_mean) / training_scale
    test = (raw_test - training_mean) / training_scale
    protocol = {
        "name": "training_only_per_variable_standardization",
        "fit_scope": "training_split_only",
        "train_fraction": train_fraction,
        "split_index": split_index,
        "training_rows": int(len(raw_train)),
        "test_rows": int(len(raw_test)),
        "training_mean": training_mean.reshape(-1).tolist(),
        "training_scale": training_scale.reshape(-1).tolist(),
        "scale_floor": PREPROCESSING_SCALE_FLOOR,
    }
    return train, test, protocol


def _method_failure(start_time: float, error: Exception) -> dict:
    return {
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "runtime_seconds": time.perf_counter() - start_time,
    }


def _validated_metrics(metrics: dict, prediction_mse: float) -> dict:
    values = {
        "flat_lag_auroc": metrics["flat_lag_auroc"],
        "flat_lag_auprc": metrics["flat_lag_auprc"],
        "aggregate_auroc": metrics["aggregate_auroc"],
        "aggregate_auprc": metrics["aggregate_auprc"],
        "prediction_mse": prediction_mse,
    }
    if not all(np.isfinite(float(value)) for value in values.values()):
        raise ValueError("Primary metrics must all be finite.")
    return {key: float(value) for key, value in values.items()}


def _run_replicate(
    *,
    K: int,
    T: int,
    nonlinearity_mix: float,
    data_seed: int,
    model_seed: int,
    max_lag: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict:
    replicate_start = time.perf_counter()
    replicate_id = f"K{K}-T{T}-nonlinearity{nonlinearity_mix:g}-seed{data_seed}"
    base = {
        "replicate_id": replicate_id,
        "K": K,
        "T": T,
        "nonlinearity_mix": float(nonlinearity_mix),
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
    }

    try:
        raw_X, G_lag, mechanism = generate_controlled_nonlinear_data(
            K=K,
            T=T,
            max_lag=max_lag,
            nonlinearity_mix=nonlinearity_mix,
            seed=data_seed,
        )
        X_train, X_test, preprocessing = split_and_standardize_training_only(
            raw_X,
            max_lag=max_lag,
        )
        off_diagonal = ~np.eye(K, dtype=bool)
        edge_prevalence = float(np.mean(G_lag[:, off_diagonal] > 0))
    except Exception as error:
        failure = _method_failure(replicate_start, error)
        return {
            **base,
            "mechanism": None,
            "preprocessing": None,
            "edge_prevalence": None,
            "status": "failed",
            "error": failure["error"],
            "runtime_seconds": failure["runtime_seconds"],
            "methods": {
                PRIMARY_MODEL_NAME: dict(failure),
                PRIMARY_COMPARATOR_NAME: dict(failure),
            },
        }

    methods = {}

    method_start = time.perf_counter()
    try:
        set_seed(model_seed)
        model = LinearBottleneck(K, max_lag, D_BOTTLENECK)
        metrics = train_and_evaluate_variant(
            PRIMARY_MODEL_NAME,
            model,
            X_train,
            X_test,
            G_lag,
            max_lag,
            epochs,
            batch_size,
            1e-3,
            1e-4,
            LAMBDA_SPARSE,
            device,
        )
        prediction_mse = metrics.get(
            "test_prediction_mse",
            metrics.get("prediction_mse"),
        )
        methods[PRIMARY_MODEL_NAME] = {
            "status": "success",
            "error": None,
            **_validated_metrics(metrics, prediction_mse),
            "runtime_seconds": time.perf_counter() - method_start,
        }
    except Exception as error:
        methods[PRIMARY_MODEL_NAME] = _method_failure(method_start, error)

    method_start = time.perf_counter()
    try:
        lasso = fit_tuned_lagged_linear_baseline(
            X_train,
            max_lag=max_lag,
            method="lasso",
            alpha_values=LASSO_ALPHA_GRID,
            validation_fraction=0.25,
            X_test=X_test,
            normalize="global",
        )
        metrics = evaluate_lagged_scores(lasso.score, G_lag)
        params = lasso.params or {}
        methods[PRIMARY_COMPARATOR_NAME] = {
            "status": "success",
            "error": None,
            **_validated_metrics(metrics, lasso.test_prediction_mse),
            "selected_alpha": float(params["alpha"]),
            "validation_prediction_mse": float(params["validation_mse"]),
            "runtime_seconds": time.perf_counter() - method_start,
        }
    except Exception as error:
        methods[PRIMARY_COMPARATOR_NAME] = _method_failure(method_start, error)

    successful = all(
        methods[name]["status"] == "success"
        for name in (PRIMARY_MODEL_NAME, PRIMARY_COMPARATOR_NAME)
    )
    errors = [
        f"{name}: {methods[name]['error']}"
        for name in (PRIMARY_MODEL_NAME, PRIMARY_COMPARATOR_NAME)
        if methods[name]["status"] != "success"
    ]
    return {
        **base,
        "mechanism": mechanism,
        "preprocessing": preprocessing,
        "edge_prevalence": edge_prevalence,
        "status": "success" if successful else "failed",
        "error": None if successful else "; ".join(errors),
        "runtime_seconds": time.perf_counter() - replicate_start,
        "methods": methods,
    }


def _one_sided_wilcoxon(differences: Sequence[float]) -> dict:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2:
        return {
            "status": "invalid_insufficient_pairs",
            "raw_p_value": None,
            "statistic": None,
        }
    if not np.all(np.isfinite(values)):
        return {
            "status": "invalid_nonfinite_differences",
            "raw_p_value": None,
            "statistic": None,
        }
    if np.all(values == 0.0):
        return {
            "status": "invalid_all_zero_differences",
            "raw_p_value": None,
            "statistic": None,
        }

    try:
        result = wilcoxon(values, alternative="greater", method="auto")
    except (ValueError, RuntimeError) as error:
        return {
            "status": "invalid_wilcoxon_error",
            "raw_p_value": None,
            "statistic": None,
            "error": f"{type(error).__name__}: {error}",
        }
    if not np.isfinite(result.pvalue) or not np.isfinite(result.statistic):
        return {
            "status": "invalid_nonfinite_wilcoxon_result",
            "raw_p_value": None,
            "statistic": None,
        }
    return {
        "status": "valid",
        "raw_p_value": float(result.pvalue),
        "statistic": float(result.statistic),
    }


def _paired_values(replicates: Sequence[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    bottleneck_values = []
    lasso_values = []
    for replicate in replicates:
        if replicate.get("status") != "success":
            continue
        bottleneck = replicate["methods"][PRIMARY_MODEL_NAME]
        lasso = replicate["methods"][PRIMARY_COMPARATOR_NAME]
        if bottleneck.get("status") != "success" or lasso.get("status") != "success":
            continue
        pair = (bottleneck.get(metric), lasso.get(metric))
        if all(value is not None and np.isfinite(float(value)) for value in pair):
            bottleneck_values.append(float(pair[0]))
            lasso_values.append(float(pair[1]))
    return np.asarray(bottleneck_values), np.asarray(lasso_values)


def _cell_statistics(
    replicates: Sequence[dict],
    *,
    planned_data_seeds: Sequence[int],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    if not replicates:
        raise ValueError("A cell must contain at least one replicate.")
    first = replicates[0]
    cell = {
        "K": int(first["K"]),
        "T": int(first["T"]),
        "nonlinearity_mix": float(first["nonlinearity_mix"]),
        "n_planned_replicates": len(planned_data_seeds),
        "n_observed_replicates": len(replicates),
    }

    planned_seeds = [int(seed) for seed in planned_data_seeds]
    observed_seeds = [int(replicate["data_seed"]) for replicate in replicates]
    duplicate_seeds = sorted({seed for seed in observed_seeds if observed_seeds.count(seed) > 1})
    missing_seeds = sorted(set(planned_seeds) - set(observed_seeds))
    unexpected_seeds = sorted(set(observed_seeds) - set(planned_seeds))
    failed_seeds = sorted(
        {
            int(replicate["data_seed"])
            for replicate in replicates
            if replicate.get("status") != "success"
            or any(
                replicate.get("methods", {}).get(method, {}).get("status") != "success"
                for method in (PRIMARY_MODEL_NAME, PRIMARY_COMPARATOR_NAME)
            )
        }
    )
    replication_complete = not (
        missing_seeds
        or unexpected_seeds
        or duplicate_seeds
        or failed_seeds
        or len(replicates) != len(planned_seeds)
    )
    cell["replication"] = {
        "status": "complete" if replication_complete else "incomplete",
        "planned_data_seeds": planned_seeds,
        "observed_data_seeds": observed_seeds,
        "missing_data_seeds": missing_seeds,
        "unexpected_data_seeds": unexpected_seeds,
        "duplicate_data_seeds": duplicate_seeds,
        "failed_data_seeds": failed_seeds,
    }
    edge_prevalences = [
        float(replicate["edge_prevalence"])
        for replicate in replicates
        if replicate.get("edge_prevalence") is not None
        and np.isfinite(float(replicate["edge_prevalence"]))
    ]
    cell["edge_prevalence"] = float(np.mean(edge_prevalences)) if edge_prevalences else None

    bn_flat, lasso_flat = _paired_values(replicates, "flat_lag_auroc")
    cell["n_complete_pairs"] = int(bn_flat.size)
    if bn_flat.size == 0:
        cell.update(
            {
                "means": None,
                "paired_effects": None,
                "inference": {
                    "status": "invalid_no_complete_pairs",
                    "statistic": None,
                    "raw_p_value": None,
                    "holm_adjusted_p_value": None,
                    "holm_rejected": False,
                },
            }
        )
        return cell

    means = {}
    paired_effects = {}
    for metric in _REQUIRED_METRICS:
        bn_values, lasso_values = _paired_values(replicates, metric)
        if bn_values.size != bn_flat.size:
            raise ValueError(f"Incomplete paired metric values for {metric}.")
        means[metric] = {
            PRIMARY_MODEL_NAME: float(np.mean(bn_values)),
            PRIMARY_COMPARATOR_NAME: float(np.mean(lasso_values)),
        }
        paired_effects[metric] = float(np.mean(bn_values - lasso_values))

    bootstrap = paired_bootstrap_ci(
        bn_flat,
        lasso_flat,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    inference = _one_sided_wilcoxon(bn_flat - lasso_flat)
    inference.update(
        {
            "holm_adjusted_p_value": None,
            "holm_rejected": False,
        }
    )
    paired_effects["flat_lag_auroc_bootstrap_ci"] = {
        "confidence": 0.95,
        "low": bootstrap.ci_low,
        "high": bootstrap.ci_high,
    }
    cell.update(
        {
            "means": means,
            "paired_effects": paired_effects,
            "inference": inference,
        }
    )
    return cell


def _gate_cell(cell: dict) -> dict:
    failed = []
    effects = cell.get("paired_effects")
    means = cell.get("means")
    inference = cell["inference"]
    replication_complete = cell["replication"]["status"] == "complete"
    if not replication_complete:
        failed.append("complete_planned_replication")
    if effects is None or means is None:
        failed.append("complete_paired_metrics")
    else:
        if effects["flat_lag_auroc"] < 0.03:
            failed.append("mean_flat_auroc_delta")
        if effects["flat_lag_auroc_bootstrap_ci"]["low"] <= 0.0:
            failed.append("bootstrap_lower_ci")
        if not (
            inference["holm_adjusted_p_value"] is not None
            and inference["holm_adjusted_p_value"] < 0.05
        ):
            failed.append("holm_adjusted_p_value")
        flat_auprc = means["flat_lag_auprc"]
        if flat_auprc[PRIMARY_MODEL_NAME] <= flat_auprc[PRIMARY_COMPARATOR_NAME]:
            failed.append("flat_auprc_superiority")
        if (
            cell["edge_prevalence"] is None
            or flat_auprc[PRIMARY_MODEL_NAME] <= cell["edge_prevalence"]
        ):
            failed.append("flat_auprc_above_edge_prevalence")
        prediction_mse = means["prediction_mse"]
        if prediction_mse[PRIMARY_MODEL_NAME] > 1.10 * prediction_mse[PRIMARY_COMPARATOR_NAME]:
            failed.append("prediction_mse_margin")
    return {
        "status": ("eligible" if replication_complete else "ineligible_incomplete_replication"),
        "supported_positive": not failed,
        "failed_criteria": failed,
        "criteria": {
            "mean_flat_auroc_delta_at_least": 0.03,
            "bootstrap_lower_ci_above": 0.0,
            "holm_adjusted_p_below": 0.05,
            "flat_auprc": "Linear-BN > Lasso",
            "flat_auprc_vs_prevalence": "Linear-BN > edge prevalence",
            "prediction_mse": "Linear-BN <= 1.10 * Lasso",
        },
    }


def apply_holm_and_gates(cells: Sequence[dict]) -> list[dict]:
    if not cells:
        return []
    adjusted_cells = copy.deepcopy(list(cells))
    family_p_values = [
        (float(cell["inference"]["raw_p_value"]) if cell["inference"]["status"] == "valid" else 1.0)
        for cell in adjusted_cells
    ]
    adjusted_p_values, rejected = holm_correction(family_p_values)
    for index, cell in enumerate(adjusted_cells):
        if cell["inference"]["status"] == "valid":
            cell["inference"]["holm_adjusted_p_value"] = float(adjusted_p_values[index])
            cell["inference"]["holm_rejected"] = bool(rejected[index])
        cell["gate"] = _gate_cell(cell)
    return adjusted_cells


def classify_mild_nonlinearity_regime(
    cells: Sequence[dict],
    *,
    k_values: Sequence[int],
    t_values: Sequence[int],
) -> dict:
    requested_k_values = set(k_values)
    if not requested_k_values:
        return {
            "status": "not_evaluable_empty_requested_K_values",
            "rule": (
                "At least two of strengths 0.2, 0.3, and 0.4 must pass for "
                "every requested K at the same T."
            ),
            "supporting_T_values": [],
            "supporting_strengths_by_T": {},
        }

    supporting_strengths_by_T = {}
    for T in t_values:
        supported_strengths = []
        for strength in MILD_STRENGTHS:
            matching = [
                cell
                for cell in cells
                if cell["T"] == T
                and cell["K"] in requested_k_values
                and np.isclose(cell["nonlinearity_mix"], strength)
            ]
            if len(matching) == len(requested_k_values) and all(
                cell["gate"]["supported_positive"] for cell in matching
            ):
                supported_strengths.append(strength)
        if len(supported_strengths) >= 2:
            supporting_strengths_by_T[str(T)] = supported_strengths

    if supporting_strengths_by_T:
        status = "supported_mild_nonlinearity_regime"
    elif any(cell["gate"]["supported_positive"] for cell in cells):
        status = "isolated_positive_exploratory"
    else:
        status = "not_supported"
    return {
        "status": status,
        "rule": (
            "At least two of strengths 0.2, 0.3, and 0.4 must pass for every "
            "requested K at the same T."
        ),
        "supporting_T_values": [int(value) for value in supporting_strengths_by_T],
        "supporting_strengths_by_T": supporting_strengths_by_T,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_strict_json(path: str | Path, artifact: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(artifact), handle, indent=2, allow_nan=False)


def _package_versions() -> dict:
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), **packages}


def run_nonlinearity_confirmation(
    *,
    k_values: Sequence[int],
    t_values: Sequence[int],
    strengths: Sequence[float],
    data_seeds: Sequence[int],
    max_lag: int = 5,
    epochs: int = 80,
    batch_size: int = 64,
    device: str = "cpu",
    output: str | Path = "results/nonlinearity_confirmation.json",
    bootstrap_resamples: int = 10_000,
    model_seed_start: int = 70_000,
) -> dict:
    start_time = time.perf_counter()
    resolved_device = choose_device(device)
    data_seeds = [int(seed) for seed in data_seeds]
    if not data_seeds or len(data_seeds) != len(set(data_seeds)):
        raise ValueError("data_seeds must be nonempty and unique.")
    model_seeds = [int(model_seed_start + index) for index in range(len(data_seeds))]

    replicates = []
    for K in k_values:
        for T in t_values:
            for strength in strengths:
                for seed_index, data_seed in enumerate(data_seeds):
                    replicates.append(
                        _run_replicate(
                            K=int(K),
                            T=int(T),
                            nonlinearity_mix=float(strength),
                            data_seed=data_seed,
                            model_seed=model_seeds[seed_index],
                            max_lag=max_lag,
                            epochs=epochs,
                            batch_size=batch_size,
                            device=resolved_device,
                        )
                    )

    cells = []
    cell_index = 0
    for K in k_values:
        for T in t_values:
            for strength in strengths:
                matching = [
                    replicate
                    for replicate in replicates
                    if replicate["K"] == K
                    and replicate["T"] == T
                    and np.isclose(
                        replicate["nonlinearity_mix"],
                        strength,
                    )
                ]
                cells.append(
                    _cell_statistics(
                        matching,
                        planned_data_seeds=data_seeds,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_seed=80_000 + cell_index,
                    )
                )
                cell_index += 1
    cells = apply_holm_and_gates(cells)
    regime = classify_mild_nonlinearity_regime(
        cells,
        k_values=k_values,
        t_values=t_values,
    )

    runtime_seconds = time.perf_counter() - start_time
    artifact = {
        "metadata": {
            "schema_version": "1.0",
            "experiment": "exp27_nonlinearity_confirmation",
            "hypothesis": (
                "A pilot-fixed linear bottleneck has a reproducible positive "
                "dependency-recovery effect over validation-tuned Lasso in a "
                "mild-nonlinearity regime."
            ),
            "fixed_primary_model": {
                "name": PRIMARY_MODEL_NAME,
                "d_bottleneck": D_BOTTLENECK,
                "lambda_sparse": LAMBDA_SPARSE,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
            "fixed_primary_comparator": {
                "name": PRIMARY_COMPARATOR_NAME,
                "alpha_grid": list(LASSO_ALPHA_GRID),
                "validation_fraction": 0.25,
            },
            "selection_signal": "validation_prediction_mse",
            "graph_labels_used_for_selection": False,
            "replication_unit": "independent_data_seed",
            "multiple_testing_family": (
                "all_primary_flat_lag_auroc_tests_across_K_T_nonlinearity_mix"
            ),
            "dataset_generator": (
                "experiments.exp27_nonlinearity_confirmation.generate_controlled_nonlinear_data"
            ),
            "controlled_mechanism": {
                "base_functional_form": ("(1-s)*x + s*sigma*tanh(x/sigma)"),
                "functional_form": ("a(s) * ((1-s)*x + s*sigma*tanh(x/sigma))"),
                "edge_coefficient": EDGE_COEFFICIENT,
                "self_coefficient": SELF_COEFFICIENT,
                "noise_std": NOISE_STD,
                "burnin": GENERATOR_BURNIN,
                "ground_truth_constant_across_mixes": True,
                "amplitude_calibration": {
                    "distribution": "stationary_gaussian_ar1",
                    "method": "gauss_hermite_quadrature",
                    "quadrature_order": GAUSS_HERMITE_ORDER,
                    "stationary_variance": (NOISE_STD**2 / (1.0 - SELF_COEFFICIENT**2)),
                    "target": "stationary_source_standard_deviation",
                    "independent_of_experiment_data_and_seeds": True,
                },
            },
            "preprocessing": {
                "name": "training_only_per_variable_standardization",
                "fit_scope": "training_split_only",
                "train_fraction": 0.8,
                "scale_floor": PREPROCESSING_SCALE_FLOOR,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_versions": _package_versions(),
            "device": resolved_device,
            "runtime_seconds": runtime_seconds,
            "data_seeds": data_seeds,
            "model_seeds": model_seeds,
            "config": {
                "K_values": [int(value) for value in k_values],
                "T_values": [int(value) for value in t_values],
                "nonlinearity_mixes": [float(value) for value in strengths],
                "max_lag": max_lag,
                "epochs": epochs,
                "batch_size": batch_size,
                "bootstrap_resamples": bootstrap_resamples,
            },
        },
        "pre_specified_gate": {
            "positive_cell": (
                "delta >= 0.03, bootstrap lower CI > 0, Holm p < 0.05, "
                "BN AUPRC > Lasso and edge prevalence, and "
                "BN MSE <= 1.10 * Lasso"
            ),
            "mild_regime": (
                "At least two of 0.2/0.3/0.4 pass across every requested K at at least one T."
            ),
        },
        "replicates": replicates,
        "aggregates": cells,
        "regime_outcome": regime,
    }
    write_strict_json(output, artifact)
    return _json_safe(artifact)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--t-values", nargs="+", type=int, default=[300, 600])
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.3, 0.4, 0.6, 1.0],
    )
    parser.add_argument("--data-seeds", nargs="+", type=int)
    parser.add_argument("--data-seed-start", type=int, default=60_000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--model-seed-start", type=int, default=70_000)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="results/nonlinearity_confirmation.json",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    data_seeds = args.data_seeds
    if data_seeds is None:
        data_seeds = list(range(args.data_seed_start, args.data_seed_start + args.n_seeds))
    return run_nonlinearity_confirmation(
        k_values=args.k_values,
        t_values=args.t_values,
        strengths=args.strengths,
        data_seeds=data_seeds,
        max_lag=args.max_lag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output=args.output,
        bootstrap_resamples=args.bootstrap_resamples,
        model_seed_start=args.model_seed_start,
    )


if __name__ == "__main__":
    main()
