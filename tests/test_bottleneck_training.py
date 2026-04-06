import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import generate_var1
from ssm_causal.bottleneck_training import (
    BottleneckTrainingConfig,
    CausalBottleneckTrainer,
    endpoint_weight_l1_penalty,
    evaluate_bottleneck_model,
    extract_weight_score,
)
from ssm_causal.mamba_model import MambaForTimeSeries


def test_extract_weight_score_matches_endpoint_product():
    model = MambaForTimeSeries(n_variables=2, d_model=2, d_state=4, n_layers=1)

    with torch.no_grad():
        model.input_proj.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.output_proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, -1.0]]))

    score = extract_weight_score(model, normalize=False)
    expected = np.abs(np.array([[2.0, 4.0], [-3.0, -4.0]]))
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(score, expected)


@pytest.fixture
def tiny_var1_data():
    X, _, G = generate_var1(K=3, T=120, graph_type="chain", seed=42)
    return X, G


def test_sparse_bottleneck_trainer_records_losses(tiny_var1_data):
    X, _ = tiny_var1_data
    model = MambaForTimeSeries(n_variables=3, d_model=16, d_state=8, n_layers=1)
    config = BottleneckTrainingConfig(
        n_epochs=2,
        batch_size=4,
        seq_len=20,
        lambda_sparse=1e-4,
    )

    trainer = CausalBottleneckTrainer(model, config=config, device="cpu")
    history = trainer.train(X, verbose=False)

    assert len(history["loss"]) == 2
    assert len(history["prediction_loss"]) == 2
    assert len(history["sparse_loss"]) == 2
    assert history["sparse_loss"][-1] > 0.0
    assert np.isfinite(history["loss"][-1])
    assert endpoint_weight_l1_penalty(model).item() > 0.0


def test_evaluate_bottleneck_model_reports_prediction_metrics(tiny_var1_data):
    X, G_true = tiny_var1_data
    model = MambaForTimeSeries(n_variables=3, d_model=16, d_state=8, n_layers=1)
    config = BottleneckTrainingConfig(
        n_epochs=2,
        batch_size=4,
        seq_len=20,
        lambda_sparse=0.0,
    )

    trainer = CausalBottleneckTrainer(model, config=config, device="cpu")
    trainer.train(X, verbose=False)
    metrics = evaluate_bottleneck_model(model, X, G_true, device="cpu")

    for key in ["auroc", "auprc", "f1", "prediction_mse", "graph_density", "mean_abs_score"]:
        assert key in metrics
    assert np.isfinite(metrics["prediction_mse"])
    assert 0.0 <= metrics["graph_density"] <= 1.0
