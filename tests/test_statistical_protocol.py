import copy
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.exp20_distribution_shift as exp20
import ssm_causal.statistics as statistics_module
from experiments.exp20_distribution_shift import (
    COMPARISON_CONFIGS,
    _summarize_comparisons,
    evaluate_fixed_score_graph_labels,
    run_fixed_score_graph_label_sensitivity_experiment,
)
from ssm_causal.statistics import holm_correction, paired_bootstrap_ci


def test_holm_correction_matches_hand_worked_example():
    adjusted, rejected = holm_correction([0.01, 0.04, 0.03, 0.002], alpha=0.05)

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.008])
    np.testing.assert_array_equal(rejected, [True, False, False, True])


@pytest.mark.parametrize(
    ("p_values", "alpha"),
    [
        ([], 0.05),
        ([[0.01, 0.02]], 0.05),
        ([0.01, np.nan], 0.05),
        ([0.01, np.inf], 0.05),
        ([-0.01, 0.02], 0.05),
        ([0.01, 1.01], 0.05),
        ([0.01], 0.0),
        ([0.01], 1.01),
    ],
)
def test_holm_correction_rejects_invalid_inputs(p_values, alpha):
    with pytest.raises(ValueError):
        holm_correction(p_values, alpha=alpha)


def test_paired_bootstrap_ci_is_deterministic_and_uses_mean_difference():
    first = np.array([3.0, 5.0, 8.0, 11.0])
    second = np.array([1.0, 4.0, 4.0, 9.0])

    result_a = paired_bootstrap_ci(
        first,
        second,
        confidence=0.95,
        n_resamples=2_000,
        seed=17,
    )
    result_b = paired_bootstrap_ci(
        first,
        second,
        confidence=0.95,
        n_resamples=2_000,
        seed=17,
    )

    assert result_a == result_b
    assert result_a.effect == pytest.approx(np.mean(first - second))
    assert result_a.ci_low <= result_a.effect <= result_a.ci_high


def test_paired_bootstrap_ci_bounds_each_index_draw(monkeypatch):
    real_rng = np.random.default_rng(23)
    requested_shapes = []

    class RecordingGenerator:
        def integers(self, low, high=None, size=None):
            requested_shapes.append(size)
            if np.prod(size) > 1_000_000:
                raise AssertionError("bootstrap requested a giant index array")
            return real_rng.integers(low, high=high, size=size)

    monkeypatch.setattr(
        statistics_module.np.random,
        "default_rng",
        lambda seed: RecordingGenerator(),
    )

    first = np.linspace(0.0, 1.0, 2_000)
    result = paired_bootstrap_ci(
        first,
        np.zeros_like(first),
        n_resamples=1_200,
        seed=23,
    )

    assert result.effect == pytest.approx(0.5)
    assert len(requested_shapes) >= 2
    assert all(np.prod(shape) <= 1_000_000 for shape in requested_shapes)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([[1.0]], [[1.0]]),
        ([1.0, np.nan], [0.0, 1.0]),
        ([1.0, 2.0], [0.0, np.inf]),
    ],
)
def test_paired_bootstrap_ci_rejects_invalid_pairs(first, second):
    with pytest.raises(ValueError):
        paired_bootstrap_ci(first, second)


@pytest.mark.parametrize(
    ("confidence", "n_resamples"),
    [(0.0, 100), (1.0, 100), (0.95, 0)],
)
def test_paired_bootstrap_ci_rejects_invalid_configuration(
    confidence,
    n_resamples,
):
    with pytest.raises(ValueError):
        paired_bootstrap_ci(
            [1.0, 2.0],
            [0.0, 1.0],
            confidence=confidence,
            n_resamples=n_resamples,
        )


def test_unchanged_graph_labels_are_not_reported_as_robustness():
    score = np.array([[[0.0, 0.1], [0.9, 0.0]]])
    graph = np.array([[[0, 0], [1, 0]]])

    result = evaluate_fixed_score_graph_labels(score, graph, graph.copy())

    assert result["comparison_status"] == "not_applicable_same_graph_labels"
    assert result["graph_label_sensitivity"] is None
    assert result["score_matrix_reused"] is True
    assert "degradation" not in result
    assert not any("robust" in key for key in result)


def test_rewired_graph_labels_are_fixed_score_sensitivity_not_adaptation():
    score = np.array([[[0.0, 0.1], [0.9, 0.0]]])
    reference_graph = np.array([[[0, 0], [1, 0]]])
    rewired_graph = np.array([[[0, 1], [0, 0]]])

    result = evaluate_fixed_score_graph_labels(
        score,
        reference_graph,
        rewired_graph,
    )

    assert result["comparison_status"] == "fixed_score_graph_label_sensitivity"
    assert result["graph_label_sensitivity"] > 0.0
    assert result["score_matrix_reused"] is True
    assert "target_prediction_mse" not in result
    assert "adaptation" not in result
    assert not any("robust" in key for key in result)


def test_exp20_has_one_explicit_same_graph_control_and_no_noise_pseudoshifts():
    control_names = [
        name
        for name, (comparison_type, _) in COMPARISON_CONFIGS.items()
        if comparison_type == "control"
    ]

    assert control_names == ["same-graph-label-control"]
    assert "noise-low" not in COMPARISON_CONFIGS
    assert "noise-high" not in COMPARISON_CONFIGS
    assert all("target_noise" not in parameters for _, parameters in COMPARISON_CONFIGS.values())


def _method_result(sensitivity):
    return {
        "reference_flat_lag_auroc": 0.8,
        "reference_aggregate_auroc": 0.75,
        "comparison_flat_lag_auroc": 0.8 - (sensitivity or 0.0),
        "comparison_aggregate_auroc": 0.72,
        "graph_labels_changed": sensitivity is not None,
        "comparison_status": (
            "fixed_score_graph_label_sensitivity"
            if sensitivity is not None
            else "not_applicable_same_graph_labels"
        ),
        "graph_label_sensitivity": sensitivity,
        "score_matrix_reused": True,
    }


def test_method_summary_is_independent_of_other_methods():
    runs = [
        {
            "comparison_name": "struct-small",
            "methods": {"Linear-BN": _method_result(0.1)},
        },
        {
            "comparison_name": "struct-small",
            "methods": {"Linear-BN": _method_result(0.2)},
        },
    ]
    runs_with_baseline = copy.deepcopy(runs)
    for run, sensitivity in zip(runs_with_baseline, [0.4, 0.5], strict=True):
        run["methods"]["LASSO"] = _method_result(sensitivity)

    original = _summarize_comparisons(runs)
    expanded = _summarize_comparisons(runs_with_baseline)

    original_linear = original["by_comparison"]["struct-small"]["methods"]["Linear-BN"]
    expanded_linear = expanded["by_comparison"]["struct-small"]["methods"]["Linear-BN"]
    assert original_linear == expanded_linear
    assert "LASSO" not in original["by_comparison"]["struct-small"]["methods"]
    assert "LASSO" in expanded["by_comparison"]["struct-small"]["methods"]


def test_exp20_trains_all_methods_on_identical_source_slice(monkeypatch, tmp_path):
    training_inputs = []

    def fake_train(
        model,
        source_train,
        reference_graph,
        comparison_graph,
        *args,
    ):
        training_inputs.append(source_train.copy())
        score = np.arange(reference_graph.size).reshape(reference_graph.shape)
        return evaluate_fixed_score_graph_labels(
            score,
            reference_graph,
            comparison_graph,
        )

    def fake_fit(source_train, max_lag, **kwargs):
        training_inputs.append(source_train.copy())
        score = np.arange(max_lag * 4 * 4).reshape(max_lag, 4, 4)
        return SimpleNamespace(score=score)

    monkeypatch.setattr(
        exp20,
        "train_bottleneck_and_score_graph_labels",
        fake_train,
    )
    monkeypatch.setattr(
        exp20,
        "fit_tuned_lagged_linear_baseline",
        fake_fit,
    )

    run_fixed_score_graph_label_sensitivity_experiment(
        comparison_names=["same-graph-label-control"],
        K=4,
        T=80,
        max_lag=3,
        d_bottleneck=4,
        lambda_sparse=1e-4,
        seeds=1,
        epochs=1,
        batch_size=16,
        baseline_methods=["lasso"],
        baseline_alphas=[1e-3],
        rrr_ranks=[2],
        device="cpu",
        save_dir=tmp_path,
    )

    assert len(training_inputs) == 3
    assert len(training_inputs[0]) < 80
    for method_input in training_inputs[1:]:
        np.testing.assert_array_equal(method_input, training_inputs[0])


def test_exp20_output_marks_same_labels_not_applicable_and_rewiring_as_sensitivity(
    tmp_path,
):
    results = run_fixed_score_graph_label_sensitivity_experiment(
        comparison_names=["same-graph-label-control", "struct-small"],
        K=4,
        T=80,
        max_lag=3,
        d_bottleneck=4,
        lambda_sparse=1e-4,
        seeds=1,
        epochs=1,
        batch_size=16,
        baseline_methods=[],
        baseline_alphas=[1e-3],
        rrr_ranks=[2],
        device="cpu",
        save_dir=tmp_path,
    )

    runs = {run["comparison_name"]: run for run in results["runs"]}
    assert results["metadata"]["objective"] == "fixed_score_graph_label_sensitivity"
    assert results["metadata"]["comparison_names"] == [
        "same-graph-label-control",
        "struct-small",
    ]
    run_config = results["metadata"]["run_config"]
    assert results["metadata"]["actual_rng_seeds"] == [50_000]
    assert run_config["batch_size"] == 16
    assert run_config["methods"] == ["Linear-BN", "Nonlinear-BN"]
    assert run_config["baseline_alpha_grid"] == [1e-3]
    assert run_config["rrr_rank_grid"] == [2]
    assert run_config["source_noise_std"] == 0.1
    assert run_config["comparison_configs"]["struct-small"] == {
        "comparison_type": "structural",
        "parameters": {"n_edges_to_flip": 2},
    }
    control = runs["same-graph-label-control"]
    assert control["comparison_status"] == "not_applicable_same_graph_labels"
    assert control["seed_index"] == 0
    assert control["rng_seed"] == 50_000
    assert control["graph_label_difference_count"] == 0
    assert control["comparison_parameters"] == {}
    assert runs["struct-small"]["comparison_status"] == ("fixed_score_graph_label_sensitivity")
    assert runs["struct-small"]["graph_label_difference_count"] > 0
    assert runs["struct-small"]["comparison_parameters"] == {
        "n_edges_to_flip": 2,
    }
    summary = results["summary"]["by_comparison"]["struct-small"]
    assert set(summary["methods"]) == {"Linear-BN", "Nonlinear-BN"}
    assert summary["methods"]["Linear-BN"]["n_seeds"] == 1
    assert "bn_more_robust" not in runs["struct-small"]
    assert "mean_bn_graph_label_sensitivity" not in runs["struct-small"]
    assert "mean_bl_graph_label_sensitivity" not in runs["struct-small"]
    assert not any("degradation" in key for key in runs["struct-small"])
    assert (tmp_path / "fixed_score_graph_label_sensitivity_results.json").exists()
