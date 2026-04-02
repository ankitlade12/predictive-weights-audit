"""
High-level API for computing SSM causal scores.

This module provides a clean interface that wraps the SSMCausalExtractor
for common use cases.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


def compute_causal_score(
    model: nn.Module,
    X: Union[torch.Tensor, np.ndarray],
    variant: str = "time_avg",
    normalize: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the SSM causal score matrix from a trained model and input data.

    This is the main entry point for causal discovery.

    Args:
        model: Trained Mamba/SSM model.
        X: Input time series, shape (T, K) or (batch, T, K).
        variant: 'time_avg', 'gate_weighted', or 'layer_agg'.
        normalize: Normalize scores to [0, 1].
        device: Device for computation.

    Returns:
        S: Causal score matrix (K, K) as numpy array.
           S[i, j] > threshold suggests j Granger-causes i.
    """
    from .extractor import SSMCausalExtractor

    # Convert numpy to tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)

    # Add batch dimension if needed
    if X.dim() == 2:
        X = X.unsqueeze(0)  # [1, T, K]

    X = X.to(device)
    model = model.to(device)
    model.eval()

    extractor = SSMCausalExtractor(model)
    S = extractor.compute_score(X, variant=variant, normalize=normalize)

    return S.numpy()


def compute_causal_score_from_bc(
    B: np.ndarray,
    C: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute causal score directly from pre-extracted B and C matrices.

    Useful when B and C have already been extracted and saved,
    or when working with custom SSM implementations.

    Args:
        B: Input-to-state matrix, shape (state_dim, K).
        C: State-to-output matrix, shape (K, state_dim).
        normalize: Normalize to [0, 1].

    Returns:
        S: Causal score matrix (K, K).
    """
    S = np.abs(C @ B.T)

    if normalize and S.max() > 0:
        S = S / S.max()

    return S


def multi_run_causal_score(
    model: nn.Module,
    X: Union[torch.Tensor, np.ndarray],
    n_runs: int = 5,
    variant: str = "time_avg",
    device: str = "cpu",
) -> tuple:
    """
    Compute causal score across multiple forward passes and return
    mean + standard deviation (for confidence estimation).

    Useful because B and C are data-dependent in selective SSMs,
    so different subsequences may yield slightly different scores.

    Args:
        model: Trained model.
        X: Input time series (T, K) or (batch, T, K).
        n_runs: Number of forward passes with different subsequences.
        variant: Score variant.
        device: Compute device.

    Returns:
        S_mean: Mean causal score matrix (K, K).
        S_std: Standard deviation of scores (K, K).
    """
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)

    if X.dim() == 2:
        X = X.unsqueeze(0)

    T = X.shape[1]
    scores = []

    for i in range(n_runs):
        # Use different contiguous subsequences
        start = (i * T) // (n_runs + 1)
        end = start + T // 2
        if end > T:
            end = T
            start = max(0, end - T // 2)

        X_sub = X[:, start:end, :].to(device)
        S = compute_causal_score(model, X_sub, variant=variant, device=device)
        scores.append(S)

    scores = np.stack(scores)
    return scores.mean(axis=0), scores.std(axis=0)
