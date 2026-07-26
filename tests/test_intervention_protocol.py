import importlib
import json as standard_json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import exp22_intervention_control as exp22
from experiments import exp23_intervention_sweep as exp23
from experiments import exp25_causal_baselines as exp25
from ssm_causal.lag_bottleneck import make_lagged_dataset
from ssm_causal.lag_synthetic import simulate_lagged_var, standardize


def _protocol():
    return importlib.import_module("ssm_causal.intervention_protocol")


def _stable_coefficients():
    return np.array(
        [
            [
                [0.25, 0.0, 0.0],
                [0.3, 0.2, 0.0],
                [0.0, -0.2, 0.2],
            ],
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
        ]
    )


def _raw_arms(intervention_type="random"):
    protocol = _protocol()
    A_lags = _stable_coefficients()
    small = protocol.generate_observational_segments(
        A_lags,
        lengths=(18,),
        seeds=(101,),
        noise_std=0.1,
    )
    return protocol.build_raw_intervention_arms(
        A_lags,
        small_observational=small,
        T_int_per_target=8,
        intervention_type=intervention_type,
        strength=1.5,
        root_seed=201,
        noise_std=0.1,
    )


def test_simulate_lagged_var_raw_flag_preserves_default_behavior():
    A_lags = np.array([[[0.3, 0.0], [0.2, 0.25]]])

    default = simulate_lagged_var(
        A_lags,
        T=24,
        noise_std=0.1,
        seed=11,
    )
    raw = simulate_lagged_var(
        A_lags,
        T=24,
        noise_std=0.1,
        seed=11,
        standardize_output=False,
    )

    np.testing.assert_array_equal(default, standardize(raw))
    assert not np.allclose(raw.std(axis=0), np.ones(2))


def test_segmented_series_owns_writable_values_and_freezes_boundaries():
    protocol = _protocol()
    first = np.arange(12, dtype=float).reshape(6, 2)
    second = np.arange(100, 114, dtype=float).reshape(7, 2)

    series = protocol.SegmentedSeries.from_segments((first, second))
    first[0, 0] = -100.0

    assert series.segment_lengths == (6, 7)
    assert series.values.flags.writeable
    assert series.values[0, 0] == 0.0
    series.values[0, 0] = 50.0
    assert series.values[0, 0] == 50.0
    with pytest.raises(FrozenInstanceError):
        series.segment_lengths = (13,)
    with pytest.raises(ValueError, match="sum"):
        protocol.SegmentedSeries(np.zeros((8, 2)), (3, 4))
    with pytest.raises(ValueError, match="variables"):
        protocol.SegmentedSeries.from_segments((np.zeros((4, 2)), np.zeros((4, 3))))


def test_raw_generators_are_exact_and_deterministic_for_all_semantics():
    protocol = _protocol()
    A_lags = _stable_coefficients()

    first = protocol.generate_observational_segments(
        A_lags,
        lengths=(11, 9, 10),
        seeds=(31, 32, 33),
        noise_std=0.1,
    )
    second = protocol.generate_observational_segments(
        A_lags,
        lengths=(11, 9, 10),
        seeds=(31, 32, 33),
        noise_std=0.1,
    )

    assert first.segment_lengths == (11, 9, 10)
    np.testing.assert_array_equal(first.values, second.values)
    assert not np.array_equal(first.iter_segments()[0], first.iter_segments()[1])

    for intervention_type in ("random", "constant", "soft"):
        generated = protocol.generate_intervention_segments(
            A_lags,
            T_per_target=9,
            intervention_type=intervention_type,
            strength=1.25,
            seeds=(41, 42, 43),
            noise_std=0.1,
        )
        repeated = protocol.generate_intervention_segments(
            A_lags,
            T_per_target=9,
            intervention_type=intervention_type,
            strength=1.25,
            seeds=(41, 42, 43),
            noise_std=0.1,
        )
        assert generated.segment_lengths == (9, 9, 9)
        np.testing.assert_array_equal(generated.values, repeated.values)


def test_seed_streams_are_unique_reproducible_and_disjoint_across_replicates():
    first = _raw_arms()
    repeated = _raw_arms()
    protocol = _protocol()
    A_lags = _stable_coefficients()
    small = protocol.generate_observational_segments(
        A_lags,
        lengths=(18,),
        seeds=(101,),
        noise_std=0.1,
    )
    adjacent = protocol.build_raw_intervention_arms(
        A_lags,
        small_observational=small,
        T_int_per_target=8,
        intervention_type="random",
        strength=1.5,
        root_seed=202,
        noise_std=0.1,
    )

    first_seeds = set(first.observational_seeds + first.intervention_seeds)
    adjacent_seeds = set(adjacent.observational_seeds + adjacent.intervention_seeds)
    assert len(first_seeds) == len(first.observational_seeds) + len(first.intervention_seeds)
    assert first_seeds.isdisjoint(adjacent_seeds)
    assert first.observational_seeds == repeated.observational_seeds
    assert first.intervention_seeds == repeated.intervention_seeds
    np.testing.assert_array_equal(first.big.values, repeated.big.values)
    np.testing.assert_array_equal(first.combined.values, repeated.combined.values)


def test_raw_arms_have_exact_matching_and_segment_safe_usable_rows():
    raw = _raw_arms()
    max_lag = 2
    expected_small = (18,)
    expected_matched = (18, 8, 8, 8)

    assert raw.small.segment_lengths == expected_small
    assert raw.big.segment_lengths == expected_matched
    assert raw.combined.segment_lengths == expected_matched
    assert len(raw.big.values) == len(raw.combined.values) == sum(expected_matched)
    np.testing.assert_array_equal(
        raw.combined.iter_segments()[0],
        raw.small.values,
    )
    np.testing.assert_array_equal(
        raw.big.iter_segments()[0],
        raw.small.values,
    )

    for arm in (raw.big, raw.combined):
        features, targets = make_lagged_dataset(
            arm.values,
            max_lag=max_lag,
            segment_lengths=arm.segment_lengths,
        )
        expected_rows = sum(length - max_lag for length in expected_matched)
        assert len(features) == len(targets) == expected_rows
        independent = [
            make_lagged_dataset(segment, max_lag=max_lag) for segment in arm.iter_segments()
        ]
        torch.testing.assert_close(
            features,
            torch.cat([rows[0] for rows in independent]),
        )
        torch.testing.assert_close(
            targets,
            torch.cat([rows[1] for rows in independent]),
        )


def test_intervention_arms_expose_target_valid_masks_and_signed_clamps():
    raw = _raw_arms("constant")

    small_mask = raw.target_valid_mask("small")
    big_mask = raw.target_valid_mask("big")
    combined_mask = raw.target_valid_mask("combined")

    assert small_mask.shape == raw.small.values.shape
    assert big_mask.shape == raw.big.values.shape
    assert combined_mask.shape == raw.combined.values.shape
    assert small_mask.all()
    assert big_mask.all()
    boundaries = np.cumsum((0, *raw.combined.segment_lengths))
    assert raw.combined_intervention_targets == (None, 0, 1, 2)
    for segment_index, target in enumerate(
        raw.combined_intervention_targets[1:],
        start=1,
    ):
        segment_mask = combined_mask[boundaries[segment_index] : boundaries[segment_index + 1]]
        assert not segment_mask[:, target].any()
        assert segment_mask[:, np.arange(3) != target].all()

    metadata = raw.metadata(max_lag=2)
    assert metadata["intervention_targets"]["combined"] == [None, 0, 1, 2]
    assert metadata["signed_clamp_values"]["combined"] == [
        None,
        1.5,
        -1.5,
        1.5,
    ]


@pytest.mark.parametrize("intervention_type", ["random", "soft"])
def test_preprocessing_centers_segments_with_one_small_reference_scale(
    intervention_type,
):
    protocol = _protocol()
    raw = _raw_arms(intervention_type)

    transformed, reference_scale = protocol.preprocess_intervention_arms(raw)

    expected_scale = raw.small.values.std(axis=0)
    np.testing.assert_allclose(reference_scale, expected_scale)
    for raw_arm, transformed_arm in zip(
        (raw.small, raw.big, raw.combined),
        (transformed.small, transformed.big, transformed.combined),
    ):
        assert transformed_arm.segment_lengths == raw_arm.segment_lengths
        for raw_segment, transformed_segment in zip(
            raw_arm.iter_segments(),
            transformed_arm.iter_segments(),
        ):
            np.testing.assert_allclose(
                transformed_segment.mean(axis=0),
                0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                transformed_segment.std(axis=0),
                raw_segment.std(axis=0) / reference_scale,
                atol=1e-12,
            )

    intervention_stds = np.stack(
        [segment.std(axis=0) for segment in transformed.combined.iter_segments()[1:]]
    )
    assert not np.allclose(intervention_stds, np.ones_like(intervention_stds))
    metadata = transformed.metadata(max_lag=2)
    assert metadata["preprocessing"] == ("per_segment_mean_with_small_observational_scale")


def test_constant_preprocessing_preserves_signed_clamp_levels():
    protocol = _protocol()
    raw = _raw_arms("constant")

    transformed, reference_scale = protocol.preprocess_intervention_arms(raw)

    reference_mean = raw.small.values.mean(axis=0)
    intervention_segments = transformed.combined.iter_segments()[1:]
    for target, segment in enumerate(intervention_segments):
        signed_strength = 1.5 if target % 2 == 0 else -1.5
        expected_level = (signed_strength - reference_mean[target]) / reference_scale[target]
        np.testing.assert_allclose(segment[:, target], expected_level, atol=1e-12)
        assert abs(segment[:, target].mean()) > 1.0

    metadata = transformed.metadata(max_lag=2)
    assert metadata["preprocessing"] == ("small_observational_mean_and_scale_for_all_segments")
    np.testing.assert_allclose(metadata["reference_mean"], reference_mean)
    np.testing.assert_allclose(metadata["reference_scale"], reference_scale)
    assert metadata["noise_std"] == 0.1
    assert metadata["burnin"] == 300
    assert metadata["scale_floor"] == 1e-8


def test_holm_family_preserves_invalid_hypotheses_in_family_size():
    protocol = _protocol()
    valid = {
        "status": "ok",
        "raw_p": 0.04,
        "holm_adjusted_p": None,
        "reject_holm": None,
    }
    invalid = {
        "status": "degenerate_all_zero",
        "raw_p": None,
        "holm_adjusted_p": None,
        "reject_holm": None,
    }

    protocol.apply_holm_family(
        [valid, invalid],
        family_name="pre_specified_family",
    )

    assert valid["holm_adjusted_p"] == pytest.approx(0.08)
    assert valid["reject_holm"] is False
    assert invalid["raw_p"] is None
    assert invalid["holm_adjusted_p"] is None
    assert invalid["reject_holm"] is False


def test_same_model_seed_repeats_one_epoch_parameters_and_new_seed_changes_them(
    monkeypatch,
):
    X = np.random.default_rng(7).normal(size=(20, 3))
    G_lag = np.zeros((1, 3, 3))
    G_lag[0, 1, 0] = 1.0
    captured_parameters = []
    original_extract = exp22.extract_lagged_weight_score

    def capture_parameters(model):
        captured_parameters.append(
            np.concatenate(
                [parameter.detach().cpu().numpy().ravel() for parameter in model.parameters()]
            )
        )
        return original_extract(model)

    monkeypatch.setattr(exp22, "extract_lagged_weight_score", capture_parameters)
    for model_seed in (17, 17, 18):
        exp22.eval_bottleneck(
            X,
            G_lag,
            max_lag=1,
            d_bottleneck=4,
            epochs=1,
            device="cpu",
            segment_lengths=(20,),
            model_seed=model_seed,
        )

    np.testing.assert_array_equal(captured_parameters[0], captured_parameters[1])
    assert not np.array_equal(captured_parameters[0], captured_parameters[2])


def _recording_evaluator(calls, name, score):
    def evaluator(
        X,
        G_lag,
        *args,
        segment_lengths=None,
        model_seed=None,
        target_valid_mask=None,
        **kwargs,
    ):
        calls.append(
            {
                "name": name,
                "length": len(X),
                "segment_lengths": tuple(segment_lengths),
                "model_seed": model_seed,
                "target_valid_mask": np.asarray(target_valid_mask).copy(),
            }
        )
        return score

    return evaluator


def _record_strict_json_dump(monkeypatch, experiment_module):
    calls = []

    def recording_dump(payload, stream, **kwargs):
        calls.append(kwargs.get("allow_nan"))
        return standard_json.dump(payload, stream, **kwargs)

    monkeypatch.setattr(
        experiment_module,
        "json",
        SimpleNamespace(dump=recording_dump),
    )
    return calls


def _assert_matched_forwarding(calls, T, T_int_per_target, K):
    expected = [
        (T,),
        (T, *((T_int_per_target,) * K)),
        (T, *((T_int_per_target,) * K)),
    ]
    by_name = {}
    for call in calls:
        by_name.setdefault(call["name"], []).append(call)
    for method_calls in by_name.values():
        assert [call["segment_lengths"] for call in method_calls] == expected
        assert len({call["model_seed"] for call in method_calls}) == 1
        assert [call["length"] for call in method_calls] == [sum(lengths) for lengths in expected]
        assert method_calls[0]["target_valid_mask"].all()
        assert method_calls[1]["target_valid_mask"].all()
        assert not method_calls[2]["target_valid_mask"].all()


def test_exp22_forwards_matched_segments_and_model_seed(monkeypatch, tmp_path):
    calls = []
    json_allow_nan = _record_strict_json_dump(monkeypatch, exp22)
    monkeypatch.setattr(
        exp22,
        "eval_bottleneck",
        _recording_evaluator(calls, "bottleneck", 0.6),
    )
    monkeypatch.setattr(
        exp22,
        "eval_lasso",
        _recording_evaluator(calls, "lasso", 0.5),
    )

    results = exp22.run_control(
        n_seeds=1,
        K_values=(3,),
        T_values=(12,),
        density=0.5,
        T_int_per_target=7,
        max_lag=1,
        epochs=1,
        save_dir=tmp_path,
    )

    _assert_matched_forwarding(calls, T=12, T_int_per_target=7, K=3)
    assert results["metadata"]["generation"]["density"] == 0.5
    assert results["metadata"]["training"]["epochs"] == 1
    assert results["metadata"]["training"]["d_bottleneck"] == 16
    assert results["metadata"]["evaluation"]["methods"] == [
        "Bottleneck",
        "Lasso",
    ]
    assert results["metadata"]["evaluation"]["estimand"] == ("intervention_target_aware")
    assert results["metadata"]["statistics"]["primary_family"] == (
        "size_matched_bottleneck_vs_lasso"
    )
    record = results["results"]["K=3_T=12_d=0.5"]["per_seed"][0]
    assert record["model_seed"] == 7
    assert record["segment_lengths"] == {
        "small": [12],
        "big": [12, 7, 7, 7],
        "combined": [12, 7, 7, 7],
    }
    assert record["protocol"]["reference_scale_source"] == "small_observational"
    assert json_allow_nan == [False]


def test_exp23_forwards_matched_segments_and_model_seed(monkeypatch, tmp_path):
    calls = []
    json_allow_nan = _record_strict_json_dump(monkeypatch, exp23)
    monkeypatch.setattr(
        exp23,
        "eval_bottleneck",
        _recording_evaluator(calls, "bottleneck", 0.6),
    )
    monkeypatch.setattr(
        exp23,
        "eval_baseline",
        _recording_evaluator(calls, "ridge", 0.5),
    )

    results = exp23.run_sweep(
        n_seeds=1,
        K_values=(3,),
        T=12,
        density=0.5,
        int_types=("random",),
        int_strengths=(1.0,),
        T_int_per_target=7,
        max_lag=1,
        d_bottleneck=4,
        epochs=1,
        baselines=("ridge",),
        device="cpu",
        save_dir=tmp_path,
    )

    _assert_matched_forwarding(calls, T=12, T_int_per_target=7, K=3)
    assert results["metadata"]["generation"]["K_values"] == [3]
    assert results["metadata"]["generation"]["intervention_types"] == ["random"]
    assert results["metadata"]["generation"]["intervention_strengths"] == [1.0]
    assert results["metadata"]["training"]["epochs"] == 1
    assert results["metadata"]["evaluation"]["estimand"] == ("intervention_target_aware")
    assert results["metadata"]["statistics"]["primary_family"] == (
        "intervention_specific_gap_across_sweep"
    )
    record = results["results"][0]["per_seed"][0]
    assert record["model_seed"] == 7
    assert record["segment_lengths"]["big"] == [12, 7, 7, 7]
    assert record["protocol"]["intervention_type"] == "random"
    assert json_allow_nan == [False]


def test_exp23_uses_type_matched_preprocessing_for_big_and_combined(
    monkeypatch,
    tmp_path,
):
    bottleneck_calls = []
    baseline_calls = []

    def make_recorder(calls):
        def record_preprocessing(
            X,
            G_lag,
            *args,
            segment_lengths=None,
            model_seed=None,
            **kwargs,
        ):
            boundaries = np.cumsum((0, *segment_lengths))
            calls.append(
                [
                    X[boundaries[index] : boundaries[index + 1]].mean(axis=0)
                    for index in range(len(segment_lengths))
                ]
            )
            return 0.5

        return record_preprocessing

    monkeypatch.setattr(
        exp23,
        "eval_bottleneck",
        make_recorder(bottleneck_calls),
    )
    monkeypatch.setattr(
        exp23,
        "eval_baseline",
        make_recorder(baseline_calls),
    )

    payload = exp23.run_sweep(
        n_seeds=1,
        K_values=(3,),
        T=12,
        density=0.5,
        int_types=("random", "constant"),
        int_strengths=(1.0,),
        T_int_per_target=7,
        max_lag=1,
        d_bottleneck=4,
        epochs=1,
        baselines=("ridge",),
        device="cpu",
        save_dir=tmp_path,
    )

    rows = {row["int_type"]: row["per_seed"][0] for row in payload["results"]}
    assert rows["random"]["protocol"]["preprocessing"] == (
        "per_segment_mean_with_small_observational_scale"
    )
    assert rows["constant"]["protocol"]["preprocessing"] == (
        "small_observational_mean_and_scale_for_all_segments"
    )

    random_small, random_big, random_combined = bottleneck_calls[:3]
    constant_small, constant_big, constant_combined = bottleneck_calls[3:]
    for arm_means in (random_small, random_big, random_combined):
        np.testing.assert_allclose(arm_means, 0.0, atol=1e-12)
    np.testing.assert_allclose(constant_small, 0.0, atol=1e-12)
    assert np.max(np.abs(constant_big[1:])) > 0.1
    assert np.max(np.abs(constant_combined[1:])) > 1.0


def test_exp25_forwards_segments_and_skips_unsupported_modern_methods(
    monkeypatch,
    tmp_path,
    capsys,
):
    calls = []
    json_allow_nan = _record_strict_json_dump(monkeypatch, exp25)
    for name in ("bottleneck", "lasso", "pcmci", "granger", "pcmci_plus"):
        monkeypatch.setattr(
            exp25,
            f"eval_{name}",
            _recording_evaluator(calls, name, 0.55),
        )

    results = exp25.run_causal_baselines(
        n_seeds=1,
        K_values=(3,),
        T_values=(12,),
        density=0.5,
        max_lag=1,
        T_int_per_target=7,
        epochs=1,
        device="cpu",
        save_dir=tmp_path,
        include_modern_baselines=True,
    )

    _assert_matched_forwarding(calls, T=12, T_int_per_target=7, K=3)
    assert set(results["results"]["K=3_T=12"]) == {
        "Bottleneck",
        "Lasso",
        "PCMCI",
        "Granger",
        "PCMCI+",
    }
    assert results["metadata"]["generation"]["intervention_types"] == ["random"]
    assert results["metadata"]["generation"]["density"] == 0.5
    assert results["metadata"]["training"]["epochs"] == 1
    assert results["metadata"]["evaluation"]["methods"] == [
        "Bottleneck",
        "Lasso",
        "PCMCI",
        "Granger",
        "PCMCI+",
    ]
    assert results["metadata"]["evaluation"]["metric"] == "aggregate_auroc"
    assert results["metadata"]["evaluation"]["estimand"] == ("intervention_target_aware")
    assert results["metadata"]["statistics"]["primary_family"] == (
        "size_matched_gap_across_methods_and_configs"
    )
    contrast = results["method_contrasts"]["Granger-minus-Bottleneck"]["K=3_T=12"]
    assert contrast["estimand"] == "granger_gap_minus_bottleneck_gap"
    assert contrast["inference"]["primary_family"] == ("granger_minus_bottleneck_across_configs")
    assert len(contrast["per_seed_difference"]) == 1
    record = results["results"]["K=3_T=12"]["Bottleneck"]["per_seed"][0]
    assert record["model_seed"] == 7
    assert record["segment_lengths"]["combined"] == [12, 7, 7, 7]
    output = capsys.readouterr().out
    assert "Skipping DYNOTEARS" in output
    assert "Skipping VAR-LiNGAM" in output
    assert "multiple independent segments" in output
    assert json_allow_nan == [False]


@pytest.mark.parametrize("experiment_module", [exp22, exp23, exp25])
def test_bottleneck_evaluators_reject_raw_non_finite_score_tensors(
    monkeypatch,
    experiment_module,
):
    score = np.zeros((1, 3, 3))
    score[0, 1, 0] = np.nan
    monkeypatch.setattr(
        experiment_module.LaggedBottleneckTrainer,
        "train",
        lambda self, X, **kwargs: {},
    )
    monkeypatch.setattr(
        experiment_module,
        "extract_lagged_weight_score",
        lambda model: score,
    )
    graph = np.zeros_like(score)
    graph[0, 1, 0] = 1.0

    with pytest.raises(ValueError, match="Bottleneck.*non-finite.*score tensor"):
        experiment_module.eval_bottleneck(
            np.zeros((12, 3)),
            graph,
            max_lag=1,
            d_bottleneck=4,
            epochs=1,
            device="cpu",
            segment_lengths=(12,),
            model_seed=7,
        )


@pytest.mark.parametrize("experiment_module", [exp22, exp23, exp25])
def test_bottleneck_evaluators_reject_non_finite_metrics(
    monkeypatch,
    experiment_module,
):
    monkeypatch.setattr(
        experiment_module.LaggedBottleneckTrainer,
        "train",
        lambda self, X, **kwargs: {},
    )
    monkeypatch.setattr(
        experiment_module,
        "extract_lagged_weight_score",
        lambda model: np.zeros((1, 3, 3)),
    )
    metric_name = "aggregate_auroc" if experiment_module is exp25 else "flat_lag_auroc"
    monkeypatch.setattr(
        experiment_module,
        "evaluate_lagged_scores",
        lambda score, graph: {metric_name: np.nan},
    )

    with pytest.raises(ValueError, match="Bottleneck.*non-finite.*metric"):
        experiment_module.eval_bottleneck(
            np.zeros((12, 3)),
            np.zeros((1, 3, 3)),
            max_lag=1,
            d_bottleneck=4,
            epochs=1,
            device="cpu",
            segment_lengths=(12,),
            model_seed=7,
        )


def test_exp22_degenerate_primary_test_is_indeterminate(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        exp22,
        "eval_bottleneck",
        _recording_evaluator([], "bottleneck", 0.5),
    )
    monkeypatch.setattr(
        exp22,
        "eval_lasso",
        _recording_evaluator([], "lasso", 0.5),
    )

    payload = exp22.run_control(
        n_seeds=1,
        K_values=(3,),
        T_values=(12,),
        density=0.5,
        T_int_per_target=7,
        max_lag=1,
        epochs=1,
        save_dir=tmp_path,
    )

    config = payload["results"]["K=3_T=12_d=0.5"]
    assert config["primary_inference"]["status"] == "degenerate_all_zero"
    assert config["verdict"] == "INDETERMINATE"
    assert payload["summary"]["overall_verdict"] == "INDETERMINATE"
    assert payload["summary"]["indeterminate_configs"] == ["K=3_T=12_d=0.5"]
    assert payload["metadata"]["statistics"]["primary_estimand"] == (
        "(bottleneck_combined-bottleneck_big)-(lasso_combined-lasso_big)"
    )
    output = capsys.readouterr().out
    assert "INDETERMINATE" in output
    assert "CONFOUND" not in output
    assert "Intervention content does carry signal" not in output


@pytest.mark.parametrize(
    ("mean_relative_effect", "reject_holm", "expected"),
    [
        (0.1, True, "RELATIVE_ADVANTAGE"),
        (-0.1, True, "NO_RELATIVE_ADVANTAGE"),
        (0.1, False, "NO_RELATIVE_ADVANTAGE"),
    ],
)
def test_exp22_relative_verdict_requires_positive_effect_and_rejection(
    mean_relative_effect,
    reject_holm,
    expected,
):
    result = {
        "mean_relative_effect": mean_relative_effect,
        "primary_inference": {
            "status": "ok",
            "holm_adjusted_p": 0.01,
            "reject_holm": reject_holm,
        },
    }

    assert exp22._relative_advantage_verdict(result) == expected


def test_exp22_reports_relative_advantage_without_intervention_signal_claim(
    monkeypatch,
    tmp_path,
    capsys,
):
    def evaluator(big_score, combined_score):
        def evaluate(
            X,
            G_lag,
            *args,
            target_valid_mask=None,
            **kwargs,
        ):
            del X, G_lag, args, kwargs
            return combined_score if not target_valid_mask.all() else big_score

        return evaluate

    def fake_inference(first, second, **kwargs):
        del kwargs
        effect = float(np.mean(np.asarray(first) - np.asarray(second)))
        return {
            "status": "ok",
            "raw_p": 0.01,
            "paired_mean_effect": effect,
            "bootstrap_ci": {
                "confidence": 0.95,
                "low": effect,
                "high": effect,
                "n_resamples": 10_000,
                "seed": 0,
            },
            "holm_adjusted_p": None,
            "reject_holm": None,
        }

    def fake_holm(inferences, **kwargs):
        del kwargs
        for inference in inferences:
            inference["holm_adjusted_p"] = 0.01
            inference["reject_holm"] = True

    monkeypatch.setattr(exp22, "eval_bottleneck", evaluator(0.6, 0.8))
    monkeypatch.setattr(exp22, "eval_lasso", evaluator(0.5, 0.55))
    monkeypatch.setattr(exp22, "paired_effect_inference", fake_inference)
    monkeypatch.setattr(exp22, "apply_holm_family", fake_holm)

    payload = exp22.run_control(
        n_seeds=1,
        K_values=(3,),
        T_values=(12,),
        density=0.5,
        T_int_per_target=7,
        max_lag=1,
        epochs=1,
        save_dir=tmp_path,
    )

    result = payload["results"]["K=3_T=12_d=0.5"]
    assert result["mean_relative_effect"] == pytest.approx(0.15)
    assert result["verdict"] == "RELATIVE_ADVANTAGE"
    assert payload["summary"]["overall_verdict"] == "RELATIVE_ADVANTAGE"
    output = capsys.readouterr().out
    assert "RELATIVE_ADVANTAGE" in output
    assert "Intervention content does carry signal" not in output


def test_exp25_core_evaluator_failure_aborts_without_artifact(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        exp25,
        "eval_bottleneck",
        _recording_evaluator([], "bottleneck", 0.6),
    )
    monkeypatch.setattr(
        exp25,
        "eval_lasso",
        _recording_evaluator([], "lasso", 0.5),
    )

    def fail_pcmci(*args, **kwargs):
        raise ImportError("tigramite unavailable")

    monkeypatch.setattr(exp25, "eval_pcmci", fail_pcmci)

    with pytest.raises(
        RuntimeError,
        match=r"PCMCI failed for K=3, T=12, seed=0",
    ):
        exp25.run_causal_baselines(
            n_seeds=1,
            K_values=(3,),
            T_values=(12,),
            density=0.5,
            max_lag=1,
            T_int_per_target=7,
            epochs=1,
            device="cpu",
            save_dir=tmp_path,
        )

    assert not (tmp_path / "causal_baselines_results.json").exists()


def test_exp23_tiny_cpu_smoke(tmp_path):
    results = exp23.run_sweep(
        n_seeds=1,
        K_values=(3,),
        T=18,
        density=0.8,
        int_types=("soft",),
        int_strengths=(1.0,),
        T_int_per_target=7,
        max_lag=1,
        d_bottleneck=4,
        epochs=1,
        baselines=("ridge",),
        device="cpu",
        save_dir=tmp_path,
    )

    assert len(results["results"]) == 1
    assert (tmp_path / "intervention_sweep_results.json").exists()


def test_exp25_tiny_cpu_smoke(tmp_path):
    results = exp25.run_causal_baselines(
        n_seeds=1,
        K_values=(3,),
        T_values=(18,),
        density=0.8,
        max_lag=1,
        T_int_per_target=7,
        d_bottleneck=4,
        epochs=1,
        device="cpu",
        save_dir=tmp_path,
    )

    assert "K=3_T=18" in results["results"]
    assert (tmp_path / "causal_baselines_results.json").exists()
