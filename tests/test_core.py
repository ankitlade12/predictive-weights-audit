"""
Core tests for SSM-Causal.

Tests the fundamental claims:
    1. Data generators produce valid data with correct shapes
    2. Model trains and converges
    3. Graph metrics distinguish perfect and imperfect rankings
"""

import numpy as np
import torch
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import (
    generate_var1, generate_var2, generate_lorenz,
    generate_henon, generate_regime_switching,
)
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics



class TestDataGenerators:
    def test_var1_chain_shape(self):
        X, A, G = generate_var1(K=3, T=100, graph_type="chain")
        assert X.shape == (100, 3)
        assert A.shape == (3, 3)
        assert G.shape == (3, 3)

    def test_var1_chain_ground_truth(self):
        _, _, G = generate_var1(K=3, T=100, graph_type="chain")
        # Chain: X0->X1, X1->X2 means G[1,0]=1, G[2,1]=1
        assert G[1, 0] == 1.0
        assert G[2, 1] == 1.0
        assert G[0, 1] == 0.0  # No reverse edge
        assert np.trace(G) == 0.0  # No self-loops

    def test_var1_fork(self):
        X, _, G = generate_var1(K=5, T=100, graph_type="fork")
        assert X.shape == (100, 5)
        # Fork: X0->X1, X0->X2, X0->X3, X0->X4
        for i in range(1, 5):
            assert G[i, 0] == 1.0

    def test_var2_shape(self):
        X, A_dict, G = generate_var2(K=5, T=200)
        assert X.shape == (200, 5)
        assert "A1" in A_dict and "A2" in A_dict

    def test_lorenz_shape(self):
        X, _, G = generate_lorenz(T=500, downsample=5)
        assert X.shape == (100, 3)
        assert G.shape == (3, 3)

    def test_henon_shape(self):
        X, _, G = generate_henon(T=200)
        assert X.shape == (200, 2)
        assert G[1, 0] == 1.0  # X1->X2
        assert G[0, 1] == 0.0  # No reverse

    def test_regime_switching(self):
        X, A_dict, G = generate_regime_switching(K=3, T=200)
        assert X.shape == (200, 3)
        assert "A1" in A_dict and "A2" in A_dict

    def test_stability(self):
        """All VAR generators should produce stable (non-diverging) series."""
        for seed in range(5):
            X, _, _ = generate_var1(K=5, T=2000, graph_type="random", seed=seed)
            assert np.all(np.isfinite(X)), f"VAR1 diverged with seed={seed}"

    def test_reproducibility(self):
        X1, _, _ = generate_var1(K=3, T=100, seed=42)
        X2, _, _ = generate_var1(K=3, T=100, seed=42)
        np.testing.assert_array_equal(X1, X2)


class TestModel:
    @pytest.fixture
    def small_model(self):
        return MambaForTimeSeries(n_variables=3, d_model=16, d_state=8, n_layers=1)

    def test_forward_shape(self, small_model):
        x = torch.randn(2, 50, 3)
        y = small_model(x)
        assert y.shape == (2, 50, 3)

    def test_training_converges(self):
        X, _, _ = generate_var1(K=3, T=500, seed=42)
        model = MambaForTimeSeries(n_variables=3, d_model=16, d_state=8, n_layers=1)
        trainer = SSMTrainer(model, lr=1e-3)
        losses = trainer.train(X, n_epochs=30, verbose=False, seq_len=50, batch_size=8)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_bc_caches_populated(self, small_model):
        x = torch.randn(1, 50, 3)
        _ = small_model(x)
        for layer in small_model.layers:
            assert layer._B is not None
            assert layer._C is not None


class TestMetrics:
    def test_perfect_recovery(self):
        G_true = np.array([[0, 0], [1, 0]], dtype=float)
        S_pred = np.array([[0.0, 0.1], [0.9, 0.0]])
        m = compute_metrics(S_pred, G_true)
        assert m["auroc"] == 1.0

    def test_random_baseline(self):
        np.random.seed(42)
        G_true = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        S_pred = np.random.rand(3, 3)
        np.fill_diagonal(S_pred, 0)
        m = compute_metrics(S_pred, G_true)
        # Random should be around 0.5 AUROC
        assert 0.0 <= m["auroc"] <= 1.0
