import json
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.exp27_nonlinearity_confirmation as exp27


def _metric_record(
    *,
    flat_auroc,
    flat_auprc,
    prediction_mse,
    aggregate_auroc=0.7,
    aggregate_auprc=0.2,
):
    return {
        "flat_lag_auroc": flat_auroc,
        "flat_lag_auprc": flat_auprc,
        "aggregate_auroc": aggregate_auroc,
        "aggregate_auprc": aggregate_auprc,
        "prediction_mse": prediction_mse,
    }


def _replicate(
    *,
    K,
    T,
    strength,
    seed,
    bn_flat,
    lasso_flat,
    bn_auprc=0.4,
    lasso_auprc=0.3,
    bn_mse=1.0,
    lasso_mse=1.0,
    edge_prevalence=0.01,
):
    return {
        "replicate_id": f"K{K}-T{T}-s{strength}-seed{seed}",
        "K": K,
        "T": T,
        "nonlinearity_mix": strength,
        "data_seed": seed,
        "model_seed": 70_000 + seed,
        "edge_prevalence": edge_prevalence,
        "status": "success",
        "methods": {
            "Linear-BN": {
                "status": "success",
                **_metric_record(
                    flat_auroc=bn_flat,
                    flat_auprc=bn_auprc,
                    prediction_mse=bn_mse,
                ),
            },
            "Lasso": {
                "status": "success",
                **_metric_record(
                    flat_auroc=lasso_flat,
                    flat_auprc=lasso_auprc,
                    prediction_mse=lasso_mse,
                ),
                "selected_alpha": 0.01,
                "validation_prediction_mse": 0.9,
            },
        },
    }


def test_controlled_generator_keeps_planted_edges_at_zero_mix():
    X, graph, mechanism = exp27.generate_controlled_nonlinear_data(
        K=5,
        T=1_000,
        max_lag=5,
        nonlinearity_mix=0.0,
        seed=60_000,
    )

    assert X.shape == (1_000, 5)
    assert mechanism["edge_coefficient"] == pytest.approx(0.55)
    assert mechanism["base_functional_form"] == ("(1-s)*x + s*sigma*tanh(x/sigma)")
    assert mechanism["functional_form"] == ("a(s) * ((1-s)*x + s*sigma*tanh(x/sigma))")
    planted_edges = mechanism["planted_edges"]
    assert planted_edges
    assert all(edge["coefficient"] == pytest.approx(0.55) for edge in planted_edges)
    for edge in planted_edges:
        assert graph[edge["lag"] - 1, edge["target"], edge["source"]] == 1.0

    first_edge = planted_edges[0]
    lag = first_edge["lag"]
    target = first_edge["target"]
    source = first_edge["source"]
    rows = np.arange(max(5, lag), len(X))
    design = np.column_stack(
        [
            np.ones(rows.size),
            X[rows - 1, target],
            X[rows - lag, source],
        ]
    )
    coefficients = np.linalg.lstsq(design, X[rows, target], rcond=None)[0]
    assert coefficients[2] == pytest.approx(0.55, abs=0.05)


def test_controlled_generator_changes_only_functional_form_across_mix():
    linear_data, linear_graph, linear_mechanism = exp27.generate_controlled_nonlinear_data(
        K=5,
        T=100,
        max_lag=5,
        nonlinearity_mix=0.0,
        seed=60_000,
    )
    _, nonlinear_graph, nonlinear_mechanism = exp27.generate_controlled_nonlinear_data(
        K=5,
        T=100,
        max_lag=5,
        nonlinearity_mix=1.0,
        seed=60_000,
    )

    np.testing.assert_array_equal(linear_graph, nonlinear_graph)
    assert linear_mechanism["edge_coefficient"] == (nonlinear_mechanism["edge_coefficient"])
    assert linear_mechanism["nonlinearity_mix"] == 0.0
    assert nonlinear_mechanism["nonlinearity_mix"] == 1.0
    source_values = linear_data[:, 0]
    nonlinear_values, _ = exp27.calibrated_nonlinear_transform(
        source_values,
        1.0,
    )
    relative_shape_change = np.mean(np.abs(source_values - nonlinear_values)) / np.std(
        source_values
    )
    assert relative_shape_change > 0.05


def test_gauss_hermite_calibration_matches_independent_stationary_ar_simulation():
    stationary_variance = exp27.NOISE_STD**2 / (1.0 - exp27.SELF_COEFFICIENT**2)
    stationary_std = np.sqrt(stationary_variance)
    rng = np.random.default_rng(91_337)
    source = np.empty(200_000)
    source[0] = rng.normal(scale=stationary_std)
    innovations = rng.normal(scale=exp27.NOISE_STD, size=len(source) - 1)
    for index, innovation in enumerate(innovations, start=1):
        source[index] = exp27.SELF_COEFFICIENT * source[index - 1] + innovation

    amplitude_ratios = []
    for nonlinearity_mix in [0.0, 0.2, 0.3, 0.4, 0.6, 1.0]:
        transformed, calibration = exp27.calibrated_nonlinear_transform(
            source,
            nonlinearity_mix,
        )
        amplitude_ratios.append(float(np.std(transformed) / np.std(source)))
        assert calibration["distribution"] == "stationary_gaussian_ar1"
        assert calibration["method"] == "gauss_hermite_quadrature"
        assert calibration["stationary_variance"] == pytest.approx(stationary_variance)
        assert calibration["independent_of_experiment_data_and_seeds"] is True

    np.testing.assert_allclose(
        amplitude_ratios,
        np.ones(len(amplitude_ratios)),
        rtol=0.0,
        atol=0.01,
    )


def test_training_only_preprocessing_ignores_held_out_outlier():
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(100, 4))
    with_outlier = raw.copy()
    with_outlier[-1] = 1e9

    clean_train, _, clean_protocol = exp27.split_and_standardize_training_only(
        raw,
        max_lag=3,
    )
    outlier_train, outlier_test, outlier_protocol = exp27.split_and_standardize_training_only(
        with_outlier,
        max_lag=3,
    )

    np.testing.assert_allclose(clean_train, outlier_train)
    np.testing.assert_allclose(
        clean_protocol["training_mean"],
        outlier_protocol["training_mean"],
    )
    np.testing.assert_allclose(
        clean_protocol["training_scale"],
        outlier_protocol["training_scale"],
    )
    assert np.max(np.abs(outlier_test[-1])) > 1e8
    assert outlier_protocol["fit_scope"] == "training_split_only"


def test_runner_forwards_only_fixed_primary_configs(monkeypatch, tmp_path):
    calls = {"generator": [], "bn": [], "lasso": []}

    def fake_generator(K, T, max_lag, nonlinearity_mix, seed):
        calls["generator"].append((K, T, max_lag, nonlinearity_mix, seed))
        X = np.arange(T * K, dtype=float).reshape(T, K)
        G_lag = np.zeros((max_lag, K, K))
        G_lag[0, 1, 0] = 1.0
        return (
            X,
            G_lag,
            {
                "nonlinearity_mix": nonlinearity_mix,
                "edge_coefficient": 0.55,
                "planted_edges": [{"source": 0, "target": 1, "lag": 1, "coefficient": 0.55}],
            },
        )

    def fake_bn(
        variant_name,
        model,
        X_train,
        X_test,
        G_lag,
        max_lag,
        epochs,
        batch_size,
        lr,
        weight_decay,
        lambda_sparse,
        device,
    ):
        calls["bn"].append(
            {
                "variant": variant_name,
                "d_bottleneck": model.d_bottleneck,
                "lambda_sparse": lambda_sparse,
                "epochs": epochs,
                "device": device,
            }
        )
        return {
            **_metric_record(
                flat_auroc=0.7,
                flat_auprc=0.4,
                prediction_mse=0.8,
            ),
            "test_prediction_mse": 0.8,
        }

    def fake_lasso(X_train, max_lag, **kwargs):
        calls["lasso"].append({"max_lag": max_lag, **kwargs})
        return SimpleNamespace(
            score=np.ones((max_lag, X_train.shape[1], X_train.shape[1])),
            test_prediction_mse=1.0,
            params={"alpha": 0.01, "validation_mse": 0.9},
        )

    monkeypatch.setattr(exp27, "generate_controlled_nonlinear_data", fake_generator)
    monkeypatch.setattr(exp27, "train_and_evaluate_variant", fake_bn)
    monkeypatch.setattr(exp27, "fit_tuned_lagged_linear_baseline", fake_lasso)
    monkeypatch.setattr(
        exp27,
        "evaluate_lagged_scores",
        lambda score, graph: _metric_record(
            flat_auroc=0.6,
            flat_auprc=0.3,
            prediction_mse=1.0,
        ),
    )

    artifact = exp27.run_nonlinearity_confirmation(
        k_values=[4],
        t_values=[60],
        strengths=[0.2],
        data_seeds=[60_000],
        max_lag=3,
        epochs=2,
        batch_size=8,
        device="cpu",
        output=tmp_path / "confirmation.json",
        bootstrap_resamples=100,
    )

    assert calls["generator"] == [(4, 60, 3, 0.2, 60_000)]
    assert calls["bn"] == [
        {
            "variant": "Linear-BN",
            "d_bottleneck": 32,
            "lambda_sparse": 1e-3,
            "epochs": 2,
            "device": "cpu",
        }
    ]
    assert calls["lasso"][0]["method"] == "lasso"
    assert calls["lasso"][0]["alpha_values"] == exp27.LASSO_ALPHA_GRID
    assert calls["lasso"][0]["validation_fraction"] == pytest.approx(0.25)
    assert artifact["replicates"][0]["methods"]["Lasso"]["selected_alpha"] == 0.01


def test_artifact_has_no_best_cell_or_graph_label_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        exp27,
        "_run_replicate",
        lambda **kwargs: _replicate(
            K=kwargs["K"],
            T=kwargs["T"],
            strength=kwargs["nonlinearity_mix"],
            seed=kwargs["data_seed"],
            bn_flat=0.4,
            lasso_flat=0.9,
        ),
    )

    artifact = exp27.run_nonlinearity_confirmation(
        k_values=[10],
        t_values=[300],
        strengths=[0.2, 0.3],
        data_seeds=[60_000, 60_001],
        epochs=1,
        device="cpu",
        output=tmp_path / "confirmation.json",
        bootstrap_resamples=100,
    )
    all_keys = json.dumps(artifact).lower()

    assert "best_" not in all_keys
    assert artifact["metadata"]["selection_signal"] == "validation_prediction_mse"
    assert artifact["metadata"]["graph_labels_used_for_selection"] is False
    assert artifact["metadata"]["fixed_primary_model"]["d_bottleneck"] == 32
    assert artifact["metadata"]["fixed_primary_comparator"]["alpha_grid"] == [
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ]


def test_holm_adjustment_can_block_raw_significance_from_gate(monkeypatch):
    raw_p_values = iter([0.02, 0.5, 0.5])
    monkeypatch.setattr(
        exp27,
        "_one_sided_wilcoxon",
        lambda differences: {
            "status": "valid",
            "raw_p_value": next(raw_p_values),
            "statistic": 10.0,
        },
    )
    cells = [
        exp27._cell_statistics(
            [
                _replicate(
                    K=10,
                    T=300,
                    strength=strength,
                    seed=seed,
                    bn_flat=0.8,
                    lasso_flat=0.7,
                )
                for seed in range(10)
            ],
            planned_data_seeds=list(range(10)),
            bootstrap_resamples=100,
            bootstrap_seed=index,
        )
        for index, strength in enumerate([0.2, 0.3, 0.4])
    ]

    adjusted = exp27.apply_holm_and_gates(cells)

    assert adjusted[0]["inference"]["raw_p_value"] == pytest.approx(0.02)
    assert adjusted[0]["inference"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted[0]["gate"]["supported_positive"] is False
    assert "holm_adjusted_p_value" in adjusted[0]["gate"]["failed_criteria"]


def test_regime_requires_two_mild_strengths_at_both_k_for_same_t():
    cells = []
    for T in [300, 600]:
        for K in [10, 20]:
            for strength in [0.2, 0.3, 0.4, 0.6]:
                supported = T == 600 and strength in {0.2, 0.3}
                cells.append(
                    {
                        "K": K,
                        "T": T,
                        "nonlinearity_mix": strength,
                        "gate": {"supported_positive": supported},
                    }
                )

    supported = exp27.classify_mild_nonlinearity_regime(
        cells,
        k_values=[10, 20],
        t_values=[300, 600],
    )
    broken = [
        {
            **cell,
            "gate": {
                "supported_positive": (
                    cell["gate"]["supported_positive"]
                    and not (cell["K"] == 20 and cell["nonlinearity_mix"] == 0.3)
                )
            },
        }
        for cell in cells
    ]
    isolated = exp27.classify_mild_nonlinearity_regime(
        broken,
        k_values=[10, 20],
        t_values=[300, 600],
    )

    assert supported["status"] == "supported_mild_nonlinearity_regime"
    assert supported["supporting_T_values"] == [600]
    assert supported["supporting_strengths_by_T"]["600"] == [0.2, 0.3]
    assert isolated["status"] == "isolated_positive_exploratory"


def test_regime_uses_the_exact_requested_k_set():
    cells = [
        {
            "K": 10,
            "T": 300,
            "nonlinearity_mix": strength,
            "gate": {"supported_positive": True},
        }
        for strength in [0.2, 0.3]
    ]

    outcome = exp27.classify_mild_nonlinearity_regime(
        cells,
        k_values=[10],
        t_values=[300],
    )

    assert outcome["status"] == "supported_mild_nonlinearity_regime"


def test_requested_k30_failure_blocks_regime_support():
    cells = []
    for K in [10, 20, 30]:
        for nonlinearity_mix in [0.2, 0.3]:
            cells.append(
                {
                    "K": K,
                    "T": 300,
                    "nonlinearity_mix": nonlinearity_mix,
                    "gate": {"supported_positive": not (K == 30 and nonlinearity_mix == 0.3)},
                }
            )

    outcome = exp27.classify_mild_nonlinearity_regime(
        cells,
        k_values=[10, 20, 30],
        t_values=[300],
    )

    assert outcome["status"] == "isolated_positive_exploratory"
    assert outcome["supporting_T_values"] == []


def test_invalid_wilcoxon_is_explicit_and_not_encoded_as_p_one():
    replicates = [
        _replicate(
            K=10,
            T=300,
            strength=0.2,
            seed=seed,
            bn_flat=0.7,
            lasso_flat=0.7,
        )
        for seed in range(4)
    ]

    cell = exp27._cell_statistics(
        replicates,
        planned_data_seeds=list(range(4)),
        bootstrap_resamples=100,
        bootstrap_seed=4,
    )
    adjusted = exp27.apply_holm_and_gates([cell])[0]

    assert adjusted["inference"]["status"] == "invalid_all_zero_differences"
    assert adjusted["inference"]["raw_p_value"] is None
    assert adjusted["inference"]["holm_adjusted_p_value"] is None
    assert adjusted["gate"]["supported_positive"] is False


def test_gate_requires_auprc_and_prediction_parity(monkeypatch):
    monkeypatch.setattr(
        exp27,
        "_one_sided_wilcoxon",
        lambda differences: {
            "status": "valid",
            "raw_p_value": 0.001,
            "statistic": 10.0,
        },
    )
    replicates = [
        _replicate(
            K=10,
            T=300,
            strength=0.2,
            seed=seed,
            bn_flat=0.8,
            lasso_flat=0.7,
            bn_auprc=0.2,
            lasso_auprc=0.3,
            bn_mse=1.2,
            lasso_mse=1.0,
        )
        for seed in range(10)
    ]

    cell = exp27._cell_statistics(
        replicates,
        planned_data_seeds=list(range(10)),
        bootstrap_resamples=100,
        bootstrap_seed=3,
    )
    adjusted = exp27.apply_holm_and_gates([cell])[0]

    assert adjusted["gate"]["supported_positive"] is False
    assert set(adjusted["gate"]["failed_criteria"]) >= {
        "flat_auprc_superiority",
        "prediction_mse_margin",
    }


def test_gate_requires_bn_auprc_above_edge_prevalence(monkeypatch):
    monkeypatch.setattr(
        exp27,
        "_one_sided_wilcoxon",
        lambda differences: {
            "status": "valid",
            "raw_p_value": 0.001,
            "statistic": 10.0,
        },
    )
    replicates = [
        _replicate(
            K=10,
            T=300,
            strength=0.2,
            seed=seed,
            bn_flat=0.8,
            lasso_flat=0.7,
            bn_auprc=0.2,
            lasso_auprc=0.1,
            edge_prevalence=0.25,
        )
        for seed in range(10)
    ]

    cell = exp27._cell_statistics(
        replicates,
        planned_data_seeds=list(range(10)),
        bootstrap_resamples=100,
        bootstrap_seed=3,
    )
    adjusted = exp27.apply_holm_and_gates([cell])[0]

    assert adjusted["edge_prevalence"] == pytest.approx(0.25)
    assert adjusted["gate"]["supported_positive"] is False
    assert "flat_auprc_above_edge_prevalence" in adjusted["gate"]["failed_criteria"]


@pytest.mark.parametrize("incomplete_kind", ["failed", "missing"])
def test_gate_is_ineligible_unless_every_planned_seed_succeeds(
    monkeypatch,
    incomplete_kind,
):
    monkeypatch.setattr(
        exp27,
        "_one_sided_wilcoxon",
        lambda differences: {
            "status": "valid",
            "raw_p_value": 0.001,
            "statistic": 10.0,
        },
    )
    replicates = [
        _replicate(
            K=10,
            T=300,
            strength=0.2,
            seed=seed,
            bn_flat=0.8,
            lasso_flat=0.7,
        )
        for seed in range(10)
    ]
    if incomplete_kind == "failed":
        replicates[-1]["status"] = "failed"
        replicates[-1]["error"] = "synthetic failure"
    else:
        replicates.pop()

    cell = exp27._cell_statistics(
        replicates,
        planned_data_seeds=list(range(10)),
        bootstrap_resamples=100,
        bootstrap_seed=3,
    )
    adjusted = exp27.apply_holm_and_gates([cell])[0]

    assert adjusted["replication"]["status"] == "incomplete"
    assert adjusted["gate"]["status"] == "ineligible_incomplete_replication"
    assert adjusted["gate"]["supported_positive"] is False
    assert "complete_planned_replication" in adjusted["gate"]["failed_criteria"]


def test_strict_json_replaces_nonfinite_values_and_records_failures(tmp_path):
    output = tmp_path / "strict.json"
    exp27.write_strict_json(
        output,
        {
            "finite": 1.0,
            "nonfinite": float("nan"),
            "failure": {"status": "failed", "error": "synthetic"},
        },
    )
    text = output.read_text()
    parsed = json.loads(text, parse_constant=lambda value: pytest.fail(value))

    assert "NaN" not in text
    assert parsed["nonfinite"] is None
    assert parsed["failure"]["status"] == "failed"


def test_defaults_and_metadata_encode_confirmatory_protocol(monkeypatch, tmp_path):
    args = exp27.parse_args([])
    assert args.k_values == [10, 20]
    assert args.t_values == [300, 600]
    assert args.strengths == [0.0, 0.2, 0.3, 0.4, 0.6, 1.0]
    assert args.data_seed_start == 60_000
    assert args.n_seeds == 10
    assert args.max_lag == 5
    assert args.epochs == 80

    monkeypatch.setattr(
        exp27,
        "_run_replicate",
        lambda **kwargs: _replicate(
            K=kwargs["K"],
            T=kwargs["T"],
            strength=kwargs["nonlinearity_mix"],
            seed=kwargs["data_seed"],
            bn_flat=0.8,
            lasso_flat=0.7,
        ),
    )
    artifact = exp27.run_nonlinearity_confirmation(
        k_values=[10],
        t_values=[300],
        strengths=[0.2],
        data_seeds=[60_000, 60_001],
        epochs=1,
        device="cpu",
        output=tmp_path / "confirmation.json",
        bootstrap_resamples=100,
    )
    metadata = artifact["metadata"]

    assert metadata["schema_version"] == "1.0"
    assert metadata["hypothesis"]
    assert metadata["replication_unit"] == "independent_data_seed"
    assert metadata["multiple_testing_family"] == (
        "all_primary_flat_lag_auroc_tests_across_K_T_nonlinearity_mix"
    )
    assert metadata["dataset_generator"] == (
        "experiments.exp27_nonlinearity_confirmation.generate_controlled_nonlinear_data"
    )
    assert metadata["preprocessing"]["fit_scope"] == "training_split_only"
    assert metadata["controlled_mechanism"]["edge_coefficient"] == 0.55
    calibration = metadata["controlled_mechanism"]["amplitude_calibration"]
    assert calibration["distribution"] == "stationary_gaussian_ar1"
    assert calibration["method"] == "gauss_hermite_quadrature"
    assert calibration["quadrature_order"] == 64
    assert calibration["stationary_variance"] == pytest.approx(
        exp27.NOISE_STD**2 / (1.0 - exp27.SELF_COEFFICIENT**2)
    )
    assert "every requested K" in artifact["pre_specified_gate"]["mild_regime"]
    assert "both K" not in artifact["pre_specified_gate"]["mild_regime"]
    assert metadata["data_seeds"] == [60_000, 60_001]
    assert metadata["model_seeds"] == [70_000, 70_001]
    assert metadata["device"] == "cpu"
    assert metadata["runtime_seconds"] >= 0.0
    assert "python" in metadata["package_versions"]
    assert len(artifact["replicates"]) == 2
