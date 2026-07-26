import hashlib
import json
from datetime import datetime

import numpy as np
import pytest

from baselines import lingam_wrapper, pcmci_wrapper
from experiments import exp24_real_data_benchmark as exp24


def _tiny_dataset():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 2))
    G = np.array([[0.0, 0.0], [1.0, 0.0]])
    return X, G, ["source", "target"]


def _metrics(auroc=0.8):
    return {"auroc": auroc, "auprc": 0.7, "f1": 0.6}


def _reported_metrics(auroc=0.8):
    return {"auroc": auroc, "auprc": 0.7, "oracle_f1": 0.6}


def _patch_climate(monkeypatch):
    monkeypatch.setattr(exp24, "load_climate_indices", lambda data_dir=None: _tiny_dataset())
    monkeypatch.setattr(exp24, "collect_edge_cards", lambda datasets: {})


def test_core_method_failure_raises_with_context_and_writes_no_result(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)

    def fail_pcmci(X, G, tau_max=5):
        raise RuntimeError("estimator exploded")

    monkeypatch.setattr(exp24, "eval_pcmci", fail_pcmci)

    with pytest.raises(RuntimeError, match=r"climate.*PCMCI.*estimator exploded"):
        exp24.run_real_benchmark(
            datasets=["climate"],
            methods=["PCMCI"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_pcmci_plus_failure_is_core_contextual_and_writes_no_result(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)

    def fail_pcmci_plus(X, G, tau_max=5):
        raise RuntimeError("pcmciplus exploded")

    monkeypatch.setattr(exp24, "eval_pcmci_plus", fail_pcmci_plus)

    with pytest.raises(RuntimeError, match=r"climate.*PCMCI\+.*pcmciplus exploded"):
        exp24.run_real_benchmark(
            datasets=["climate"],
            methods=["PCMCI+"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_optional_missing_method_is_explicitly_skipped_with_null_metrics(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(
        exp24,
        "optional_method_unavailability",
        lambda method_name: "requires a legacy Python 3.10 environment",
    )

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["DYNOTEARS"],
        save_dir=str(tmp_path),
    )

    record = payload["datasets"]["climate"]["methods"]["DYNOTEARS"]
    assert record["status"] == "skipped"
    assert record["error"] == "requires a legacy Python 3.10 environment"
    assert record["metrics"] == {"auroc": None, "auprc": None, "oracle_f1": None}
    assert payload["rankings"]["climate"] == []

    saved = json.loads((tmp_path / "real_data_benchmark_results.json").read_text())
    assert saved["datasets"]["climate"]["methods"]["DYNOTEARS"] == record


def test_var_lingam_preflight_unavailability_is_skipped_with_null_metrics(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(
        exp24,
        "optional_method_unavailability",
        lambda method_name: "lingam is not installed",
    )

    def unexpected_var_lingam(X, G, max_lag=3):
        raise AssertionError("preflight skip must not execute the evaluator")

    monkeypatch.setattr(exp24, "eval_var_lingam", unexpected_var_lingam)

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["VAR-LiNGAM"],
        save_dir=str(tmp_path),
    )

    record = payload["datasets"]["climate"]["methods"]["VAR-LiNGAM"]
    assert record["status"] == "skipped"
    assert record["error"] == "lingam is not installed"
    assert record["metrics"] == {"auroc": None, "auprc": None, "oracle_f1": None}
    assert payload["rankings"]["climate"] == []


def test_optional_runtime_import_error_raises_contextually(monkeypatch, tmp_path):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(exp24, "optional_method_unavailability", lambda method_name: None)

    def broken_var_lingam(X, G, max_lag=3):
        raise ImportError("installed backend failed during fit")

    monkeypatch.setattr(exp24, "eval_var_lingam", broken_var_lingam)

    with pytest.raises(
        RuntimeError,
        match=r"climate.*VAR-LiNGAM.*installed backend failed during fit",
    ):
        exp24.run_real_benchmark(
            datasets=["climate"],
            methods=["VAR-LiNGAM"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_raw_non_finite_score_matrix_raises_contextually(monkeypatch, tmp_path):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(
        pcmci_wrapper,
        "pcmci_causality_matrix",
        lambda X, tau_max, pc_alpha: np.full((2, 2), np.nan),
    )

    with pytest.raises(
        RuntimeError,
        match=r"climate.*PCMCI.*non-finite score matrix",
    ):
        exp24.run_real_benchmark(
            datasets=["climate"],
            methods=["PCMCI"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_var_lingam_non_finite_wrapper_failure_is_contextual(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(
        exp24,
        "optional_method_unavailability",
        lambda method_name: None,
    )
    monkeypatch.setattr(
        lingam_wrapper,
        "var_lingam_causality_matrix",
        lambda X, max_lag: (_ for _ in ()).throw(
            ValueError("VAR-LiNGAM returned non-finite adjacency_matrices_")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"climate.*VAR-LiNGAM.*non-finite adjacency_matrices_",
    ):
        exp24.run_real_benchmark(
            datasets=["climate"],
            methods=["VAR-LiNGAM"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_git_metadata_hashes_exact_binary_diff_deterministically(monkeypatch):
    diff_bytes = b"diff --git a/file b/file\nliteral 3\nabc\n"

    class GitResult:
        def __init__(self, stdout):
            self.stdout = stdout
            self.returncode = 0

    def fake_git_run(command, **kwargs):
        git_args = command[1:]
        if git_args == ["rev-parse", "HEAD"]:
            return GitResult(b"abc123\n")
        if git_args == ["diff", "--binary", "HEAD"]:
            return GitResult(diff_bytes)
        if git_args == ["status", "--porcelain=v1", "--untracked-files=normal"]:
            return GitResult(b" M experiments/exp24_real_data_benchmark.py\n")
        raise AssertionError(f"Unexpected git command: {command}")

    monkeypatch.setattr(exp24.subprocess, "run", fake_git_run)

    first = exp24._git_metadata()
    second = exp24._git_metadata()

    expected_hash = hashlib.sha256(diff_bytes).hexdigest()
    assert first == second
    assert first == {
        "code_revision": "abc123",
        "worktree_dirty": True,
        "worktree_diff_sha256": expected_hash,
    }


def test_source_tree_hash_includes_untracked_python_content(tmp_path):
    (tmp_path / "tracked.py").write_text("VALUE = 1\n")
    untracked_source = tmp_path / "new_untracked.py"
    untracked_source.write_text("VALUE = 2\n")

    first = exp24._source_tree_sha256(tmp_path)
    second = exp24._source_tree_sha256(tmp_path)
    untracked_source.write_text("VALUE = 3\n")
    changed = exp24._source_tree_sha256(tmp_path)

    assert first == second
    assert changed != first


def test_lorenz96_is_simulated_with_exact_graph_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(exp24, "load_lorenz96_bench", lambda K=10, T=1500: _tiny_dataset())
    monkeypatch.setattr(exp24, "eval_ridge", lambda X, G, max_lag=3: _metrics())
    monkeypatch.setattr(exp24, "collect_edge_cards", lambda datasets: {})

    payload = exp24.run_real_benchmark(
        datasets=["lorenz96"],
        methods=["Ridge"],
        save_dir=str(tmp_path),
    )

    dataset = payload["datasets"]["lorenz96"]
    assert dataset["kind"] == "simulated"
    assert dataset["label_type"] == "exact_graph"
    assert "real" not in dataset["provenance"].lower()


def test_method_selection_excludes_ssm(monkeypatch, tmp_path):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(exp24, "eval_lasso", lambda X, G, max_lag=3: _metrics())

    def unexpected_ssm(*args, **kwargs):
        raise AssertionError("SSM must not be trained")

    monkeypatch.setattr(exp24, "eval_weight_projection", unexpected_ssm)

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["Lasso"],
        save_dir=str(tmp_path),
    )

    assert list(payload["datasets"]["climate"]["methods"]) == ["Lasso"]
    assert payload["config"]["methods"] == ["Lasso"]


def test_programmatic_default_uses_real_climate_data_directory(monkeypatch, tmp_path):
    received_data_dirs = []

    def fake_loader(data_dir):
        received_data_dirs.append(data_dir)
        return _tiny_dataset()

    monkeypatch.setattr(exp24, "load_climate_indices", fake_loader)
    monkeypatch.setattr(exp24, "eval_lasso", lambda X, G, max_lag=3: _metrics())
    monkeypatch.setattr(exp24, "collect_edge_cards", lambda datasets: {})

    exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["Lasso"],
        save_dir=str(tmp_path),
    )

    assert received_data_dirs == ["data/real"]


def test_requested_unavailable_dream3_raises_dataset_error(monkeypatch, tmp_path):
    monkeypatch.setattr(exp24, "load_dream3_gene", lambda: None)
    monkeypatch.setattr(exp24, "collect_edge_cards", lambda datasets: {})

    with pytest.raises(RuntimeError, match=r"Requested dataset.*dream3.*unavailable"):
        exp24.run_real_benchmark(
            datasets=["dream3"],
            methods=["Lasso"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_result_payload_contains_reproducibility_metadata(monkeypatch, tmp_path):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(exp24, "eval_lasso", lambda X, G, max_lag=3: _metrics())
    monkeypatch.setattr(
        exp24,
        "_git_metadata",
        lambda: {
            "code_revision": "abc123",
            "worktree_dirty": False,
            "worktree_diff_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    monkeypatch.setattr(
        exp24,
        "_package_versions",
        lambda: {
            "numpy": "test-numpy",
            "torch": "test-torch",
            "scipy": "test-scipy",
            "scikit-learn": "test-sklearn",
            "statsmodels": "test-statsmodels",
            "tigramite": "test-tigramite",
        },
    )
    monkeypatch.setattr(exp24, "_source_tree_sha256", lambda root=None: "source-tree-hash")
    monkeypatch.setattr(
        exp24.sys,
        "argv",
        ["experiments/exp24_real_data_benchmark.py", "--methods", "Lasso"],
    )

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["Lasso"],
        device="cpu",
        epochs=3,
        save_dir=str(tmp_path),
    )

    assert payload["schema_version"]
    assert payload["experiment"] == "exp24_real_data_benchmark"
    assert payload["code_revision"] == "abc123"
    assert payload["worktree_dirty"] is False
    assert payload["worktree_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert payload["source_tree_sha256"] == "source-tree-hash"
    assert payload["argv"] == [
        "experiments/exp24_real_data_benchmark.py",
        "--methods",
        "Lasso",
    ]
    assert payload["command"] == ("experiments/exp24_real_data_benchmark.py --methods Lasso")
    created_at = datetime.fromisoformat(payload["created_at_utc"])
    assert created_at.utcoffset().total_seconds() == 0
    assert payload["python_version"]
    assert payload["package_versions"]["scikit-learn"] == "test-sklearn"
    assert payload["package_versions"]["tigramite"] == "test-tigramite"
    assert payload["device"] == "cpu"
    assert payload["epochs"] == 3
    assert payload["runtime_s"] >= 0
    assert payload["metric_definitions"]["oracle_f1"]["oracle_tuned"] is True
    assert payload["claim_policy"]["single_seed_competitiveness_claims_permitted"] is False

    dataset = payload["datasets"]["climate"]
    assert dataset["kind"] == "real_observational"
    assert dataset["label_type"] == "soft_literature_labels"
    assert "NOAA" in dataset["provenance"]
    assert dataset["K"] == 2
    assert dataset["T"] == 24
    assert dataset["n_edges"] == 1
    assert dataset["names"] == ["source", "target"]
    X, G, _ = _tiny_dataset()
    assert dataset["data_sha256"] == exp24._array_sha256(X)
    assert dataset["graph_sha256"] == exp24._array_sha256(G)
    assert dataset["data_sha256"] == exp24._array_sha256(X.copy())
    assert dataset["graph_sha256"] == exp24._array_sha256(G.copy())

    record = dataset["methods"]["Lasso"]
    assert record["status"] == "ok"
    assert record["error"] is None
    assert record["device"] == "cpu"
    assert record["runtime_s"] >= 0
    assert record["replication_unit"] == "deterministic_run"
    assert record["hyperparameters"]["alpha_grid"]
    assert record["hyperparameters"]["refit_scope"] == "all_lagged_rows"
    assert record["metrics"] == _reported_metrics()
    assert "f1" not in record["metrics"]


def test_package_versions_include_optional_causal_packages(monkeypatch):
    monkeypatch.setattr(
        exp24,
        "version",
        lambda package_name: f"{package_name}-installed",
    )

    versions = exp24._package_versions()

    assert versions["causalnex"] == "causalnex-installed"
    assert versions["lingam"] == "lingam-installed"


def test_finance_metadata_is_heuristic_soft_label_sensitivity():
    metadata = exp24.DATASET_METADATA["finance"]

    assert metadata["label_type"] == "heuristic_soft_label_sensitivity"
    assert metadata["claim_scope"] == "sensitivity_analysis_only"
    assert "heuristic" in metadata["provenance"].lower()
    assert "literature" not in metadata["provenance"].lower()


def test_benchmark_method_specs_use_one_lag_horizon():
    specs = exp24.build_method_specs(device="cpu", epochs=1)

    assert specs["Lagged-Bottleneck"].hyperparameters["max_lag"] == 3
    assert specs["Lasso"].hyperparameters["max_lag"] == 3
    assert specs["Ridge"].hyperparameters["max_lag"] == 3
    assert specs["PCMCI"].hyperparameters["tau_max"] == 3
    assert specs["PCMCI+"].hyperparameters["tau_max"] == 3
    assert specs["Granger"].hyperparameters["max_lag"] == 3


def test_result_records_selected_and_resolved_hyperparameters(monkeypatch, tmp_path):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(
        exp24,
        "eval_lasso",
        lambda X, G, max_lag=3: {
            **_metrics(),
            "selected_hyperparameters": {"alpha": 0.01},
        },
    )
    monkeypatch.setattr(
        exp24,
        "eval_lagged_bottleneck",
        lambda X, G, max_lag=3, d_bottleneck=16, epochs=100, device="cpu", seed=42: _metrics(),
    )

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["Lasso", "Lagged-Bottleneck"],
        save_dir=str(tmp_path),
    )

    records = payload["datasets"]["climate"]["methods"]
    assert records["Lasso"]["hyperparameters"]["selected"] == {"alpha": 0.01}
    assert records["Lagged-Bottleneck"]["hyperparameters"]["batch_size"] == 21


@pytest.mark.parametrize(
    ("evaluator_name", "estimator_name"),
    [
        ("eval_lasso", "MultiTaskLasso"),
        ("eval_ridge", "Ridge"),
    ],
)
def test_linear_baseline_refits_selected_alpha_on_all_lagged_rows(
    monkeypatch,
    evaluator_name,
    estimator_name,
):
    from sklearn import linear_model

    fit_row_counts = []

    class FakeRegressor:
        def __init__(self, alpha, **kwargs):
            self.alpha = alpha

        def fit(self, features, targets):
            fit_row_counts.append(len(features))
            self.coef_ = np.zeros((targets.shape[1], features.shape[1]))
            return self

        def predict(self, features):
            return np.zeros((len(features), 2))

    monkeypatch.setattr(linear_model, estimator_name, FakeRegressor)
    monkeypatch.setattr(exp24, "compute_metrics", lambda score, graph: _metrics())
    X, G, _ = _tiny_dataset()

    getattr(exp24, evaluator_name)(X, G, max_lag=3)

    n_lagged_rows = len(X) - 3
    validation_train_rows = int(n_lagged_rows * 0.6)
    assert fit_row_counts[-1] == n_lagged_rows
    assert all(rows == validation_train_rows for rows in fit_row_counts[:-1])


def test_stochastic_methods_store_per_seed_records_and_rank_by_mean(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)
    seed_scores = {11: 0.9, 12: 0.1, 13: 0.9}

    def fake_ssm(X, G, d_model=64, epochs=150, device="cpu", seed=42):
        return _metrics(seed_scores[seed])

    monkeypatch.setattr(exp24, "eval_weight_projection", fake_ssm)
    monkeypatch.setattr(exp24, "eval_lasso", lambda X, G, max_lag=3: _metrics(0.75))

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["SSM", "Lasso"],
        model_seeds=[11, 12, 13],
        save_dir=str(tmp_path),
    )

    ssm = payload["datasets"]["climate"]["methods"]["SSM"]
    expected_mean = np.mean(list(seed_scores.values()))
    assert [row["model_seed"] for row in ssm["per_seed"]] == [11, 12, 13]
    assert ssm["metrics"]["auroc_mean"] == pytest.approx(expected_mean)
    assert ssm["metrics"]["auroc_std"] == pytest.approx(np.std(list(seed_scores.values()), ddof=0))
    assert ssm["aggregation"] == {
        "unit": "model_initialization_seed",
        "std_ddof": 0,
        "interpretation": "descriptive_initialization_stability_only",
        "independent_uncertainty": False,
    }
    assert all(row["status"] == "ok" and row["error"] is None for row in ssm["per_seed"])
    assert ssm["competitiveness_claims_permitted"] is False
    assert "cannot support competitiveness claims" in (ssm["competitiveness_claims_note"])
    assert "not independent uncertainty" in ssm["competitiveness_claims_note"]
    assert payload["stochastic_competitiveness_claims_permitted"] is False
    assert payload["claim_policy"]["model_seed_interpretation"] == (
        "descriptive_initialization_stability_only"
    )
    assert payload["claim_policy"]["model_seeds_are_independent_replications"] is False
    assert payload["rankings"]["climate"][0]["method"] == "Lasso"
    ssm_ranking = next(row for row in payload["rankings"]["climate"] if row["method"] == "SSM")
    assert ssm_ranking["auroc"] == pytest.approx(expected_mean)
    assert ssm_ranking["aggregation"] == "mean_across_model_seeds"


def test_lagged_bottleneck_uses_configured_model_seed_list(monkeypatch, tmp_path):
    _patch_climate(monkeypatch)
    received_seeds = []

    def fake_lagged(
        X,
        G,
        max_lag=3,
        d_bottleneck=16,
        epochs=100,
        device="cpu",
        seed=42,
    ):
        received_seeds.append(seed)
        return _metrics()

    monkeypatch.setattr(exp24, "eval_lagged_bottleneck", fake_lagged)

    exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["Lagged-Bottleneck"],
        model_seeds=[21, 22, 23],
        save_dir=str(tmp_path),
    )

    assert received_seeds == [21, 22, 23]


def test_single_stochastic_seed_prohibits_competitiveness_claims(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(
        exp24,
        "eval_weight_projection",
        lambda X, G, d_model=64, epochs=150, device="cpu", seed=42: _metrics(),
    )

    payload = exp24.run_real_benchmark(
        datasets=["climate"],
        methods=["SSM"],
        model_seeds=[7],
        save_dir=str(tmp_path),
    )

    record = payload["datasets"]["climate"]["methods"]["SSM"]
    assert record["competitiveness_claims_permitted"] is False
    assert "descriptive initialization stability only" in (record["competitiveness_claims_note"])
    assert "cannot support competitiveness claims" in (record["competitiveness_claims_note"])
    assert payload["stochastic_competitiveness_claims_permitted"] is False


def test_non_finite_core_metric_is_rejected_before_json_is_written(
    monkeypatch,
    tmp_path,
):
    _patch_climate(monkeypatch)
    monkeypatch.setattr(exp24, "eval_lasso", lambda X, G, max_lag=3: _metrics(np.nan))

    with pytest.raises(RuntimeError, match=r"climate.*Lasso.*non-finite"):
        exp24.run_real_benchmark(
            datasets=["climate"],
            methods=["Lasso"],
            save_dir=str(tmp_path),
        )

    assert not (tmp_path / "real_data_benchmark_results.json").exists()


def test_cli_accepts_explicit_method_selection():
    args = exp24.build_argument_parser().parse_args(
        ["--datasets", "climate", "--methods", "PCMCI", "PCMCI+", "Granger"]
    )

    assert args.methods == ["PCMCI", "PCMCI+", "Granger"]
    assert len(args.model_seeds) >= 3


def test_legacy_method_registry_keeps_existing_display_names():
    registry = exp24.build_method_registry(device="cpu", epochs=1)

    assert "Weight-Proj (SSM)" in registry
    assert "Tuned-Lasso" in registry
    assert "Tuned-Ridge" in registry
