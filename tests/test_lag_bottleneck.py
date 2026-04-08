import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_block_lowrank_generator_is_finite_and_has_lag_edges():
    X, G_lag, _ = generate_block_lowrank_lagged(K=8, T=120, max_lag=5, seed=9)

    assert X.shape == (120, 8)
    assert G_lag.shape == (5, 8, 8)
    assert np.isfinite(X).all()
    assert np.abs(X).max() < 10.0
    assert G_lag.sum() > 0
