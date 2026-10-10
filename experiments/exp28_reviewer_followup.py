"""Post-review follow-up; preserves the submitted artifacts and fixed gates.

Run `primary` in the locked project environment, then `optional` in the
isolated Python 3.10 environment. Both stages use identical exported arrays.
These analyses are exploratory, not pre-registered confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "results" / "reviewer_followup"
DATA_SEEDS = tuple(range(81000, 81010))
METHODS = ("Lagged-Bottleneck", "Lasso", "Ridge", "PCMCI", "PCMCI+", "Granger")


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def environment(packages):
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}


def source_hashes():
    paths = [Path(__file__), ROOT / "data/real_loaders.py"]
    paths += [
        ROOT / f"experiments/{name}.py"
        for name in (
            "exp18_mechanistic_ablation",
            "exp24_real_data_benchmark",
            "exp27_nonlinearity_confirmation",
        )
    ]
    paths += list((ROOT / "ssm_causal").glob("*.py"))
    paths += list((ROOT / "baselines").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def primary():
    import torch

    from data.real_loaders import load_lorenz96
    from experiments.exp24_real_data_benchmark import _array_sha256, build_method_specs

    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = build_method_specs("cpu", 80)
    payload = {
        "status": "running",
        "source_sha256": source_hashes(),
        "analysis": "post_review_exploratory",
        "environment": environment(["numpy", "scipy", "scikit-learn", "torch", "tigramite"]),
        "protocol": {
            "K": 10,
            "T": 1500,
            "F": 10.0,
            "dt": 0.01,
            "subsample": 10,
            "lag_horizon": 3,
            "data_seeds": DATA_SEEDS,
            "model_seeds": [42, 43, 44],
            "replication_unit": "independent_simulated_trajectory",
            "aggregation": "mean_over_model_initializations_within_each_trajectory",
            "preprocessing": "full_series_standardization_matching_submitted_Table_1",
            "scope": "same_graph_and_sampling_configuration_not_harder_graph_validation",
        },
        "replicates": [],
    }
    # Seed 0 reproduces the submitted benchmark for the optional baselines only.
    for seed in (0, *DATA_SEEDS):
        X, G, _ = load_lorenz96(seed=seed)
        if seed == 0:
            original = json.loads(
                (
                    ROOT / "results/corrected_workshop/real_data/real_data_benchmark_results.json"
                ).read_text()
            )
            assert _array_sha256(X) == original["datasets"]["lorenz96"]["data_sha256"]
            assert _array_sha256(G) == original["datasets"]["lorenz96"]["graph_sha256"]
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        array_path = OUTPUT / f"lorenz96_seed{seed}.npz"
        np.savez_compressed(array_path, X=X, G=G)
        record = {
            "data_seed": seed,
            "input_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "methods": {},
        }
        if seed == 0:
            continue
        for method in METHODS:
            started = time.perf_counter()
            model_seeds = [42, 43, 44] if specs[method].stochastic else [None]
            runs = [specs[method].evaluator(X, G, model_seed) for model_seed in model_seeds]
            record["methods"][method] = {
                "auroc": float(np.mean([r["auroc"] for r in runs])),
                "auprc": float(np.mean([r["auprc"] for r in runs])),
                "per_initialization": runs if specs[method].stochastic else None,
                "runtime_seconds": time.perf_counter() - started,
            }
            print(seed, method, record["methods"][method]["auroc"], flush=True)
        payload["replicates"].append(record)
        save(OUTPUT / "trajectory_replication.json", payload)
    payload["status"] = "complete"
    save(OUTPUT / "trajectory_replication.json", payload)


def optional():
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    from baselines.dynotears import dynotears_causality_matrix
    from baselines.lingam_wrapper import var_lingam_causality_matrix

    payload = {
        "status": "running",
        "source_sha256": source_hashes(),
        "analysis": "post_review_exploratory",
        "environment": environment(["causalnex", "lingam", "numpy", "scipy", "scikit-learn"]),
        "protocol": {
            "lag_horizon": 3,
            "VAR-LiNGAM_random_state": 0,
            "VAR-LiNGAM_criterion": "bic",
            "VAR-LiNGAM_prune": True,
            "DYNOTEARS": {
                "lambda_w": 0.1,
                "lambda_a": 0.1,
                "max_iter": 100,
                "h_tol": 1e-8,
                "w_threshold": 0,
            },
            "tuning": "package_defaults_no_graph_label_selection",
            "scores": "absolute_lagged_coefficients_max_over_lags_no_self_edges",
            "scope": "observational_Lorenz96_only_no_intervention_claim",
        },
        "replicates": [],
    }
    for seed in (0, *DATA_SEEDS):
        path = OUTPUT / f"lorenz96_seed{seed}.npz"
        arrays = np.load(path)
        X, G = arrays["X"], arrays["G"]
        mask = ~np.eye(len(G), dtype=bool)
        record = {
            "data_seed": seed,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "methods": {},
        }
        for name, runner in [
            ("DYNOTEARS", dynotears_causality_matrix),
            ("VAR-LiNGAM", var_lingam_causality_matrix),
        ]:
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    score = runner(X, max_lag=3)
                assert score.shape == G.shape and np.isfinite(score).all()
                precision, recall, _ = precision_recall_curve(G[mask], score[mask])
                result = {
                    "status": "ok",
                    "auroc": float(roc_auc_score(G[mask], score[mask])),
                    "auprc": float(auc(recall, precision)),
                    "warnings": sorted({str(w.message) for w in captured}),
                }
                np.save(OUTPUT / f"{name}_seed{seed}_scores.npy", score)
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            result["runtime_seconds"] = time.perf_counter() - started
            record["methods"][name] = result
            print(seed, name, result, flush=True)
        payload["replicates"].append(record)
        save(OUTPUT / "optional_baselines.json", payload)
    payload["status"] = "complete"
    save(OUTPUT / "optional_baselines.json", payload)


def diagnose():
    from experiments.exp27_nonlinearity_confirmation import (
        LASSO_ALPHA_GRID,
        generate_controlled_nonlinear_data,
        split_and_standardize_training_only,
    )
    from ssm_causal.lag_bottleneck import evaluate_lagged_scores, fit_tuned_lagged_linear_baseline

    original_path = ROOT / "results/corrected_workshop/nonlinearity_confirmation.json"
    original = json.loads(original_path.read_text())
    records = []
    for saved in original["replicates"]:
        X, G, _ = generate_controlled_nonlinear_data(
            K=saved["K"],
            T=saved["T"],
            max_lag=5,
            nonlinearity_mix=saved["nonlinearity_mix"],
            seed=saved["data_seed"],
        )
        train, test, _ = split_and_standardize_training_only(X, max_lag=5)
        fit = fit_tuned_lagged_linear_baseline(
            train,
            max_lag=5,
            method="lasso",
            alpha_values=LASSO_ALPHA_GRID,
            validation_fraction=0.25,
            X_test=test,
            normalize="none",
        )
        metrics = evaluate_lagged_scores(fit.score, G)
        old = saved["methods"]["Lasso"]
        assert fit.params["alpha"] == old["selected_alpha"], saved["replicate_id"]
        assert np.isclose(metrics["flat_lag_auroc"], old["flat_lag_auroc"]), saved["replicate_id"]
        assert np.isclose(fit.test_prediction_mse, old["prediction_mse"], rtol=1e-5), saved[
            "replicate_id"
        ]
        records.append(
            {
                "replicate_id": saved["replicate_id"],
                "K": saved["K"],
                "T": saved["T"],
                "s": saved["nonlinearity_mix"],
                "alpha": fit.params["alpha"],
                "nonzero_off_diagonal_lag_scores": int(np.count_nonzero(fit.score)),
                "auroc": metrics["flat_lag_auroc"],
            }
        )
    save(
        OUTPUT / "lasso_diagnostic.json",
        {
            "analysis": "post_review_diagnostic",
            "source_sha256": source_hashes(),
            "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "environment": environment(["numpy", "scikit-learn", "torch"]),
            "replicates": records,
        },
    )
    print("Verified all", len(records), "saved Lasso runs and recorded off-diagonal sparsity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["primary", "optional", "diagnose"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    {"primary": primary, "optional": optional, "diagnose": diagnose}[args.stage]()
