import os
import sys

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssm_causal.lag_bottleneck as lag_bottleneck_module
from data.synthetic import generate_var1
from ssm_causal.lag_bottleneck import (
    LaggedBottleneckModel,
    LaggedBottleneckTrainer,
    LaggedBottleneckTrainingConfig,
    evaluate_lagged_bottleneck_model,
    extract_lagged_weight_score,
    fit_lagged_linear_baseline,
    fit_tuned_lagged_linear_baseline,
    make_lagged_dataset,
)
from ssm_causal.lag_synthetic import generate_block_lowrank_lagged


def test_make_lagged_dataset_uses_lag_major_order():
    X = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ],
        dtype=np.float32,
    )

    features, targets = make_lagged_dataset(X, max_lag=2)

    expected_features = torch.tensor(
        [
            [2.0, 3.0, 0.0, 1.0],
            [4.0, 5.0, 2.0, 3.0],
        ]
    )
    expected_targets = torch.tensor([[4.0, 5.0], [6.0, 7.0]])

    torch.testing.assert_close(features, expected_features)
    torch.testing.assert_close(targets, expected_targets)


@pytest.mark.parametrize("use_torch", [False, True])
def test_make_lagged_dataset_keeps_segment_windows_independent(use_torch):
    first_segment = np.arange(8, dtype=np.float32).reshape(4, 2)
    second_segment = np.arange(100, 110, dtype=np.float32).reshape(5, 2)
    X = np.concatenate([first_segment, second_segment])
    if use_torch:
        X = torch.from_numpy(X)

    features, targets = make_lagged_dataset(
        X,
        max_lag=2,
        segment_lengths=[len(first_segment), len(second_segment)],
    )
    first_features, first_targets = make_lagged_dataset(first_segment, max_lag=2)
    second_features, second_targets = make_lagged_dataset(second_segment, max_lag=2)

    torch.testing.assert_close(features, torch.cat([first_features, second_features]))
    torch.testing.assert_close(targets, torch.cat([first_targets, second_targets]))
    assert features.shape[0] == sum(length - 2 for length in [4, 5])
    assert not torch.any((features < 50).any(dim=1) & (features > 50).any(dim=1))


@pytest.mark.parametrize(
    ("segment_lengths", "message"),
    [
        ([3, 0, 3], "positive"),
        ([3, 4], "sum to T"),
        ([2, 4], "more rows than max_lag"),
        ([[3, 3]], "one-dimensional"),
        ([2.5, 3.5], "integer-like"),
    ],
)
def test_make_lagged_dataset_validates_segment_lengths(segment_lengths, message):
    X = np.arange(12, dtype=np.float32).reshape(6, 2)

    with pytest.raises(ValueError, match=message):
        make_lagged_dataset(X, max_lag=2, segment_lengths=segment_lengths)


def test_lagged_target_mask_excludes_outcome_but_keeps_value_in_later_features():
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [50.0, 1.0], [3.0, 100.0]],
        dtype=np.float32,
    )
    valid = np.ones_like(X, dtype=bool)
    valid[2, 0] = False

    features, _ = make_lagged_dataset(X, max_lag=1)
    lagged_valid = lag_bottleneck_module.make_lagged_target_mask(
        valid,
        max_lag=1,
    )

    assert not lagged_valid[1, 0]
    assert lagged_valid[2, 1]
    assert features[2, 0] == 50.0


def test_masked_neural_outcome_value_does_not_change_one_epoch_parameters():
    X = np.random.default_rng(14).normal(size=(12, 2)).astype(np.float32)
    changed = X.copy()
    changed[-1, 0] = 1_000_000.0
    valid = np.ones_like(X, dtype=bool)
    valid[-1, 0] = False
    parameters = []

    for values in (X, changed):
        torch.manual_seed(31)
        model = LaggedBottleneckModel(
            n_variables=2,
            max_lag=1,
            d_bottleneck=3,
        )
        trainer = LaggedBottleneckTrainer(
            model,
            config=LaggedBottleneckTrainingConfig(n_epochs=1, batch_size=8),
            device="cpu",
        )
        torch.manual_seed(31)
        trainer.train(
            values,
            verbose=False,
            target_valid_mask=valid,
        )
        parameters.append(
            torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        )

    torch.testing.assert_close(parameters[0], parameters[1], rtol=0.0, atol=0.0)


def test_extract_lagged_weight_score_splits_input_projection_by_lag():
    model = LaggedBottleneckModel(n_variables=2, max_lag=2, d_bottleneck=2, bias=False)

    with torch.no_grad():
        model.input_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            )
        )
        model.output_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, -1.0]]))

    score = extract_lagged_weight_score(model, normalize="none")

    expected_lag1 = np.abs(np.array([[1.0, 2.0], [-5.0, -6.0]]))
    expected_lag2 = np.abs(np.array([[3.0, 4.0], [-7.0, -8.0]]))
    np.fill_diagonal(expected_lag1, 0.0)
    np.fill_diagonal(expected_lag2, 0.0)
    expected = np.stack([expected_lag1, expected_lag2], axis=0)

    np.testing.assert_allclose(score, expected)


def test_lagged_bottleneck_trainer_smoke_reports_metrics():
    X, A, _ = generate_var1(K=3, T=120, graph_type="chain", seed=7)
    G_lag = np.zeros((2, 3, 3))
    G_lag[0] = (np.abs(A) > 1e-10).astype(float)
    np.fill_diagonal(G_lag[0], 0.0)

    model = LaggedBottleneckModel(n_variables=3, max_lag=2, d_bottleneck=8)
    config = LaggedBottleneckTrainingConfig(n_epochs=2, batch_size=16)
    trainer = LaggedBottleneckTrainer(model, config=config, device="cpu")
    history = trainer.train(X, verbose=False)
    metrics = evaluate_lagged_bottleneck_model(model, X, G_lag, device="cpu")

    assert len(history["loss"]) == 2
    assert np.isfinite(history["loss"][-1])
    for key in ["aggregate_auroc", "flat_lag_auroc", "prediction_mse"]:
        assert key in metrics
    assert np.isfinite(metrics["prediction_mse"])


def test_trainer_and_evaluator_use_segment_safe_example_counts(monkeypatch):
    segment_lengths = [5, 6]
    X = np.arange(22, dtype=np.float32).reshape(11, 2)
    G_lag = np.zeros((2, 2, 2))
    G_lag[0, 0, 1] = 1.0
    calls = []
    original_make_lagged_dataset = lag_bottleneck_module.make_lagged_dataset

    def spy_make_lagged_dataset(X, max_lag, device="cpu", segment_lengths=None):
        features, targets = original_make_lagged_dataset(
            X,
            max_lag,
            device=device,
            segment_lengths=segment_lengths,
        )
        calls.append((segment_lengths, features.shape[0]))
        return features, targets

    monkeypatch.setattr(
        lag_bottleneck_module,
        "make_lagged_dataset",
        spy_make_lagged_dataset,
    )
    model = LaggedBottleneckModel(n_variables=2, max_lag=2, d_bottleneck=4)
    trainer = LaggedBottleneckTrainer(
        model,
        config=LaggedBottleneckTrainingConfig(n_epochs=1, batch_size=4),
        device="cpu",
    )

    trainer.train(X, verbose=False, segment_lengths=segment_lengths)
    trainer.prediction_mse(X, segment_lengths=segment_lengths)
    evaluate_lagged_bottleneck_model(
        model,
        X,
        G_lag,
        device="cpu",
        segment_lengths=segment_lengths,
    )

    assert calls == [(segment_lengths, 7)] * 3


def test_fit_lagged_linear_baseline_reports_lag_score_and_mse():
    X, _, _ = generate_var1(K=3, T=160, graph_type="chain", seed=11)
    result = fit_lagged_linear_baseline(
        X_train=X[:120],
        max_lag=2,
        method="ridge",
        alpha=1e-3,
        X_test=X[120:],
    )

    assert result.score.shape == (2, 3, 3)
    assert np.isfinite(result.train_prediction_mse)
    assert result.test_prediction_mse is not None
    assert np.isfinite(result.test_prediction_mse)


@pytest.mark.parametrize("method", ["ols", "ridge", "lasso"])
def test_masked_linear_outcome_value_does_not_change_coefficients(method):
    X = np.random.default_rng(19).normal(size=(30, 2)).astype(np.float32)
    changed = X.copy()
    changed[-1, 0] = 1_000_000.0
    valid = np.ones_like(X, dtype=bool)
    valid[-1, 0] = False

    first = fit_lagged_linear_baseline(
        X,
        max_lag=1,
        method=method,
        alpha=1e-3,
        normalize="none",
        train_target_valid_mask=valid,
    )
    second = fit_lagged_linear_baseline(
        changed,
        max_lag=1,
        method=method,
        alpha=1e-3,
        normalize="none",
        train_target_valid_mask=valid,
    )

    np.testing.assert_allclose(first.score, second.score, rtol=0.0, atol=0.0)


def test_masked_linear_retains_intervention_value_as_downstream_predictor():
    X = np.zeros((10, 2), dtype=np.float32)
    X[4, 0] = 25.0
    X[5, 1] = 50.0
    valid = np.ones_like(X, dtype=bool)
    valid[4, 0] = False

    result = fit_lagged_linear_baseline(
        X,
        max_lag=1,
        method="ols",
        normalize="none",
        train_target_valid_mask=valid,
    )

    assert result.score[0, 1, 0] == pytest.approx(2.0)


def test_masked_rrr_is_rejected():
    X = np.random.default_rng(2).normal(size=(20, 3))
    valid = np.ones_like(X, dtype=bool)
    valid[10:, 0] = False

    with pytest.raises(ValueError, match="RRR.*target-valid"):
        fit_lagged_linear_baseline(
            X,
            max_lag=1,
            method="rrr",
            train_target_valid_mask=valid,
        )


def test_fit_lagged_linear_baseline_uses_segment_safe_example_counts(monkeypatch):
    train_segment_lengths = [5, 6]
    test_segment_lengths = [4, 5]
    calls = []
    original_make_lagged_dataset = lag_bottleneck_module.make_lagged_dataset

    def spy_make_lagged_dataset(X, max_lag, device="cpu", segment_lengths=None):
        features, targets = original_make_lagged_dataset(
            X,
            max_lag,
            device=device,
            segment_lengths=segment_lengths,
        )
        calls.append((segment_lengths, features.shape[0]))
        return features, targets

    monkeypatch.setattr(
        lag_bottleneck_module,
        "make_lagged_dataset",
        spy_make_lagged_dataset,
    )

    fit_lagged_linear_baseline(
        X_train=np.arange(22, dtype=np.float32).reshape(11, 2),
        max_lag=2,
        method="ols",
        X_test=np.arange(18, dtype=np.float32).reshape(9, 2),
        train_segment_lengths=train_segment_lengths,
        test_segment_lengths=test_segment_lengths,
    )

    assert calls == [
        (train_segment_lengths, 7),
        (test_segment_lengths, 5),
    ]


def test_segmented_linear_fit_matches_independently_constructed_design_rows():
    rng = np.random.default_rng(23)

    def make_segment(length, offset):
        source = rng.normal(size=length).astype(np.float32) + offset
        target = np.empty(length, dtype=np.float32)
        target[0] = -10_000.0 * offset
        target[1:] = 3.0 * source[:-1] + 2.0
        return np.column_stack([source, target])

    segments = [make_segment(20, 0.0), make_segment(22, 100.0)]
    X = np.concatenate(segments)
    independent_rows = [make_lagged_dataset(segment, max_lag=1) for segment in segments]
    features = torch.cat([rows[0] for rows in independent_rows]).numpy()
    targets = torch.cat([rows[1] for rows in independent_rows]).numpy()
    expected_estimator = LinearRegression().fit(features, targets)
    expected_prediction_mse = float(np.mean((expected_estimator.predict(features) - targets) ** 2))

    result = fit_lagged_linear_baseline(
        X_train=X,
        max_lag=1,
        method="ols",
        normalize="none",
        train_segment_lengths=[len(segment) for segment in segments],
    )

    expected_score = np.abs(expected_estimator.coef_)
    np.fill_diagonal(expected_score, 0.0)
    np.testing.assert_allclose(
        result.score[0],
        expected_score,
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.train_prediction_mse == pytest.approx(
        expected_prediction_mse,
        rel=1e-6,
        abs=1e-6,
    )


def test_fit_reduced_rank_baseline_reports_params():
    X, _, _ = generate_var1(K=4, T=180, graph_type="chain", seed=13)
    result = fit_lagged_linear_baseline(
        X_train=X[:130],
        max_lag=3,
        method="rrr",
        alpha=1e-3,
        rank=2,
        X_test=X[130:],
    )

    assert result.score.shape == (3, 4, 4)
    assert result.params is not None
    assert result.params["rank"] == 2
    assert np.isfinite(result.train_prediction_mse)
    assert result.test_prediction_mse is not None
    assert np.isfinite(result.test_prediction_mse)


def test_fit_tuned_baseline_records_validation_mse():
    X, _, _ = generate_var1(K=4, T=220, graph_type="chain", seed=17)
    result = fit_tuned_lagged_linear_baseline(
        X_train=X[:170],
        max_lag=3,
        method="rrr",
        alpha_values=[1e-3, 1e-2],
        rank_values=[1, 2],
        X_test=X[170:],
    )

    assert result.score.shape == (3, 4, 4)
    assert result.params is not None
    assert "validation_mse" in result.params
    assert np.isfinite(result.params["validation_mse"])


def test_tuned_baseline_splits_each_segment_and_refits_without_crossing_boundaries(
    monkeypatch,
):
    segment_lengths = [12, 14]
    test_segment_lengths = [5, 6]
    X_train = np.random.default_rng(29).normal(size=(26, 2)).astype(np.float32)
    X_test = np.random.default_rng(31).normal(size=(11, 2)).astype(np.float32)
    first_segment = X_train[:12]
    second_segment = X_train[12:]
    expected_fit = np.concatenate([first_segment[:8], second_segment[:10]])
    expected_validation = np.concatenate([first_segment[8:], second_segment[10:]])
    expected_fit_lengths = [8, 10]
    expected_validation_lengths = [4, 4]
    calls = []
    original_fit_lagged_linear_baseline = lag_bottleneck_module.fit_lagged_linear_baseline

    def spy_fit_lagged_linear_baseline(X_train, *args, **kwargs):
        X_test_arg = kwargs.get("X_test")
        calls.append(
            {
                "X_train": np.asarray(X_train).copy(),
                "X_test": (None if X_test_arg is None else np.asarray(X_test_arg).copy()),
                "train_segment_lengths": kwargs.get("train_segment_lengths"),
                "test_segment_lengths": kwargs.get("test_segment_lengths"),
            }
        )
        return original_fit_lagged_linear_baseline(X_train, *args, **kwargs)

    monkeypatch.setattr(
        lag_bottleneck_module,
        "fit_lagged_linear_baseline",
        spy_fit_lagged_linear_baseline,
    )

    fit_tuned_lagged_linear_baseline(
        X_train=X_train,
        max_lag=2,
        method="ridge",
        alpha_values=[1e-3, 1e-2],
        validation_fraction=0.25,
        X_test=X_test,
        segment_lengths=segment_lengths,
        test_segment_lengths=test_segment_lengths,
    )

    assert len(calls) == 3
    for candidate_call in calls[:2]:
        np.testing.assert_array_equal(candidate_call["X_train"], expected_fit)
        np.testing.assert_array_equal(candidate_call["X_test"], expected_validation)
        assert candidate_call["train_segment_lengths"] == expected_fit_lengths
        assert candidate_call["test_segment_lengths"] == expected_validation_lengths

    refit_call = calls[2]
    np.testing.assert_array_equal(refit_call["X_train"], X_train)
    np.testing.assert_array_equal(refit_call["X_test"], X_test)
    assert refit_call["train_segment_lengths"] == segment_lengths
    assert refit_call["test_segment_lengths"] == test_segment_lengths


def test_tuned_baseline_short_segment_falls_back_to_first_candidate(monkeypatch):
    segment_lengths = [7, 12]
    test_segment_lengths = [5, 6]
    calls = []
    original_make_lagged_dataset = lag_bottleneck_module.make_lagged_dataset

    def spy_make_lagged_dataset(X, max_lag, device="cpu", segment_lengths=None):
        features, targets = original_make_lagged_dataset(
            X,
            max_lag,
            device=device,
            segment_lengths=segment_lengths,
        )
        calls.append((segment_lengths, features.shape[0]))
        return features, targets

    monkeypatch.setattr(
        lag_bottleneck_module,
        "make_lagged_dataset",
        spy_make_lagged_dataset,
    )

    result = fit_tuned_lagged_linear_baseline(
        X_train=np.arange(38, dtype=np.float32).reshape(19, 2),
        max_lag=2,
        method="rrr",
        alpha_values=[0.25, 0.5],
        rank_values=[1, 2],
        X_test=np.arange(22, dtype=np.float32).reshape(11, 2),
        segment_lengths=segment_lengths,
        test_segment_lengths=test_segment_lengths,
    )

    assert result.params is not None
    assert result.params["alpha"] == 0.25
    assert result.params["rank"] == 1
    assert "validation_mse" not in result.params
    assert calls == [
        (segment_lengths, 15),
        (test_segment_lengths, 7),
    ]


def test_tuned_baseline_rejects_segment_too_short_for_full_refit():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)

    with pytest.raises(ValueError, match="more rows than max_lag"):
        fit_tuned_lagged_linear_baseline(
            X_train=X,
            max_lag=2,
            method="ridge",
            alpha_values=[0.1],
            segment_lengths=[2, 8],
        )


def test_block_lowrank_generator_is_finite_and_has_lag_edges():
    X, G_lag, _ = generate_block_lowrank_lagged(K=8, T=120, max_lag=5, seed=9)

    assert X.shape == (120, 8)
    assert G_lag.shape == (5, 8, 8)
    assert np.isfinite(X).all()
    assert np.abs(X).max() < 10.0
    assert G_lag.sum() > 0
