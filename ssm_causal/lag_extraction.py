"""
Lag-Resolved Bottleneck Extraction.

Instead of a single aggregate score S[i,j], extracts a lag-specific
score S[tau, i, j] that identifies who influences whom at which lag.

Core idea:
    For a trained prediction model, perturb input variable j at time
    t-tau and measure the effect on output i at time t. This gives a
    lag-specific causal influence score.

Two extraction methods:

1. Lag-Jacobian: ∂ŷ_i(t) / ∂x_j(t-τ) — exact but expensive
2. Lag-Impulse: inject unit impulse on channel j at time t0,
   measure response on channel i at times t0+1, t0+2, ...

Both produce S[tau, i, j] — a 3D tensor of lag-specific scores.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .mamba_model import MambaForTimeSeries


def extract_lag_jacobian_score(
    model: nn.Module,
    X_tensor: torch.Tensor,
    max_lag: int = 5,
    device: str = "cpu",
    seq_len: int = 100,
) -> np.ndarray:
    """
    Lag-resolved Jacobian: ∂ŷ_i(t) / ∂x_j(t-τ) for each lag τ.

    Returns S[tau, i, j] where:
        tau = 0..max_lag-1  (lag index, tau=0 means lag 1)
        i = target variable
        j = source variable

    This is the gold standard lag-resolved extraction.
    Cost: O(K * max_lag) backward passes.
    """
    model.eval()
    K = X_tensor.shape[1]
    effective_len = min(seq_len, X_tensor.shape[0] - max_lag - 1)

    # Use a window starting after max_lag so all lags are available
    start = max_lag
    X_window = X_tensor[start - max_lag:start + effective_len].unsqueeze(0).to(device)
    X_window = X_window.requires_grad_(True)

    Y = model(X_window)  # [1, max_lag + effective_len, K]

    S = np.zeros((max_lag, K, K))

    for tau in range(1, max_lag + 1):
        for i in range(K):
            model.zero_grad()
            if X_window.grad is not None:
                X_window.grad.zero_()

            # Sum output i over the prediction window (after max_lag offset)
            target = Y[0, max_lag:, i].sum()
            target.backward(retain_graph=True)

            # Read gradient at the lagged input positions
            # grad shape: [1, max_lag + effective_len, K]
            # We want the gradient w.r.t. inputs at positions [0:effective_len]
            # which correspond to times t-tau for the outputs at positions [max_lag:]
            grad = X_window.grad[0]  # [max_lag + effective_len, K]

            # Average gradient magnitude at the tau-offset positions
            # Output at time t (index max_lag + s) depends on input at time t-tau (index max_lag + s - tau)
            lag_grads = []
            for s in range(effective_len):
                input_idx = max_lag + s - tau
                if 0 <= input_idx < grad.shape[0]:
                    lag_grads.append(grad[input_idx].abs())

            if lag_grads:
                S[tau - 1, i, :] = torch.stack(lag_grads).mean(dim=0).detach().cpu().numpy()

    # Normalize per-lag
    for tau in range(max_lag):
        np.fill_diagonal(S[tau], 0)
        if S[tau].max() > 0:
            S[tau] = S[tau] / S[tau].max()

    return S


def extract_lag_impulse_score(
    model: nn.Module,
    X_tensor: torch.Tensor,
    max_lag: int = 5,
    device: str = "cpu",
    impulse_strength: float = 1.0,
    n_samples: int = 10,
) -> np.ndarray:
    """
    Lag-resolved impulse response: inject impulse on channel j,
    measure effect on channel i at each subsequent lag.

    Cheaper than Jacobian — requires only forward passes.
    More interpretable — directly measures causal effect.

    Returns S[tau, i, j] — impulse response at each lag.
    """
    model.eval()
    K = X_tensor.shape[1]
    T = X_tensor.shape[0]
    seq_len = min(50, T - max_lag - 5)

    S = np.zeros((max_lag, K, K))

    with torch.no_grad():
        for sample in range(n_samples):
            # Random starting point
            start = torch.randint(0, T - seq_len - max_lag - 1, (1,)).item()
            X_base = X_tensor[start:start + seq_len + max_lag].unsqueeze(0).to(device)

            # Baseline prediction
            Y_base = model(X_base)[0]  # [seq_len + max_lag, K]

            for j in range(K):
                # Inject impulse on channel j at a fixed position
                impulse_pos = seq_len // 2
                X_perturbed = X_base.clone()
                X_perturbed[0, impulse_pos, j] += impulse_strength

                Y_perturbed = model(X_perturbed)[0]

                # Measure response at each lag
                for tau in range(1, max_lag + 1):
                    response_pos = impulse_pos + tau
                    if response_pos < Y_perturbed.shape[0]:
                        diff = (Y_perturbed[response_pos] - Y_base[response_pos]).abs()
                        S[tau - 1, :, j] += diff.cpu().numpy()

        # Average over samples
        S = S / n_samples

    # Normalize per-lag
    for tau in range(max_lag):
        np.fill_diagonal(S[tau], 0)
        if S[tau].max() > 0:
            S[tau] = S[tau] / S[tau].max()

    return S


def lag_score_to_aggregate(S_lag: np.ndarray) -> np.ndarray:
    """Convert lag-resolved S[tau, i, j] to aggregate S[i, j] by taking max over lags."""
    S_agg = np.abs(S_lag).max(axis=0)
    np.fill_diagonal(S_agg, 0)
    if S_agg.max() > 0:
        S_agg = S_agg / S_agg.max()
    return S_agg


def lag_score_to_lag_graph(S_lag: np.ndarray, threshold: float = 0.3) -> Dict:
    """
    Convert lag-resolved scores to a lag-annotated causal graph.

    Returns dict of edges: {(source, target): [lag1, lag2, ...]}
    """
    max_lag, K, _ = S_lag.shape
    edges = {}
    for tau in range(max_lag):
        for i in range(K):
            for j in range(K):
                if i != j and S_lag[tau, i, j] > threshold:
                    key = (j, i)  # source -> target
                    if key not in edges:
                        edges[key] = []
                    edges[key].append(tau + 1)
    return edges


def evaluate_lag_recovery(
    S_lag: np.ndarray,
    G_true_lag: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate lag-resolved recovery.

    Args:
        S_lag: Predicted [max_lag, K, K]
        G_true_lag: Ground truth [max_lag, K, K] (binary)

    Returns metrics for both:
        - aggregate graph recovery (AUROC on union graph)
        - per-lag recovery (average AUROC across lags)
    """
    from .utils import compute_metrics

    # Aggregate evaluation
    S_agg = lag_score_to_aggregate(S_lag)
    G_agg = (G_true_lag.sum(axis=0) > 0).astype(float)
    np.fill_diagonal(G_agg, 0)
    agg_metrics = compute_metrics(S_agg, G_agg)

    # Per-lag evaluation
    max_lag = min(S_lag.shape[0], G_true_lag.shape[0])
    lag_aurocs = []
    for tau in range(max_lag):
        G_tau = G_true_lag[tau]
        if G_tau.sum() > 0 and G_tau.sum() < G_tau.size:
            S_tau = S_lag[tau]
            np.fill_diagonal(S_tau, 0)
            if S_tau.max() > 0:
                S_tau = S_tau / S_tau.max()
            m = compute_metrics(S_tau, G_tau)
            lag_aurocs.append(m["auroc"])

    result = {
        "aggregate_auroc": agg_metrics["auroc"],
        "aggregate_f1": agg_metrics["f1"],
        "per_lag_auroc_mean": float(np.mean(lag_aurocs)) if lag_aurocs else 0.0,
        "per_lag_auroc_std": float(np.std(lag_aurocs)) if lag_aurocs else 0.0,
        "n_lags_evaluated": len(lag_aurocs),
    }
    return result
