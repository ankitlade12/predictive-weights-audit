"""
Improved causal score extraction methods.

Addresses weaknesses identified in initial experiments:
1. Ensemble: Combine Jacobian + Weight projection scores
2. Multi-run: Average across multiple training seeds
3. Adaptive model sizing: Scale d_model with K
4. Sparsity-aware training: Light L1 on projections to sharpen signal
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .mamba_model import MambaForTimeSeries, SSMTrainer


def extract_jacobian_score(model, X_tensor, device="cpu", seq_len=100):
    """Jacobian-based causal score (from exp0)."""
    model.eval()
    K = X_tensor.shape[1]
    X_sub = X_tensor[:seq_len].unsqueeze(0).to(device).requires_grad_(True)
    Y = model(X_sub)

    S = torch.zeros(K, K)
    for i in range(K):
        model.zero_grad()
        if X_sub.grad is not None:
            X_sub.grad.zero_()
        Y[0, :, i].sum().backward(retain_graph=(i < K - 1))
        S[i, :] = X_sub.grad[0].abs().mean(dim=0)

    S = S.detach().numpy()
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def extract_weight_score(model):
    """Weight-projection causal score."""
    W_in = model.input_proj.weight.detach().cpu()
    W_out = model.output_proj.weight.detach().cpu()
    S = (W_out @ W_in).abs().numpy()
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def extract_ensemble_score(model, X_tensor, device="cpu", seq_len=100, alpha=0.5):
    """
    Ensemble of Jacobian + Weight scores.

    Combines the structural signal from weight projections with the
    dynamic signal from the Jacobian. Alpha controls the blend:
        S = alpha * S_weight + (1 - alpha) * S_jacobian
    """
    S_jac = extract_jacobian_score(model, X_tensor, device, seq_len)
    S_weight = extract_weight_score(model)
    S = alpha * S_weight + (1 - alpha) * S_jac
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def adaptive_model_size(K):
    """Scale model dimensions with K to ensure d >= Kp capacity."""
    if K <= 5:
        return {"d_model": 64, "d_state": 16, "n_layers": 4}
    elif K <= 10:
        return {"d_model": 96, "d_state": 24, "n_layers": 4}
    else:
        return {"d_model": 128, "d_state": 32, "n_layers": 6}


def train_with_sparsity(
    X,
    d_model=64,
    d_state=16,
    n_layers=4,
    n_epochs=200,
    lr=1e-3,
    l1_lambda=0.001,
    device="cpu",
):
    """
    Train Mamba with light L1 penalty on input/output projections.

    This encourages the weight matrices to be sparse, making the
    causal score matrix cleaner — especially for larger K where the
    score can become noisy.
    """
    if isinstance(X, np.ndarray):
        K = X.shape[1]
    else:
        K = X.shape[1]

    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=d_state, n_layers=n_layers
    )
    model = model.to(device)

    if isinstance(X, np.ndarray):
        X_t = torch.FloatTensor(X).to(device)
    else:
        X_t = X.float().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    T = X_t.shape[0]
    seq_len = min(200, T - 2)
    batch_size = 32

    model.train()
    for epoch in range(n_epochs):
        starts = torch.randint(0, T - seq_len - 1, (batch_size,))
        batch_x = torch.stack([X_t[s:s + seq_len] for s in starts])
        batch_y = torch.stack([X_t[s + 1:s + seq_len + 1] for s in starts])

        pred = model(batch_x)
        mse_loss = loss_fn(pred, batch_y)

        # L1 on projection weights to encourage sparsity
        l1_loss = (
            model.input_proj.weight.abs().sum() +
            model.output_proj.weight.abs().sum()
        ) * l1_lambda

        loss = mse_loss + l1_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model


def improved_causal_score(
    X,
    n_runs=3,
    device="cpu",
    use_sparsity=True,
    use_ensemble=True,
    adaptive_size=True,
):
    """
    Full improved pipeline: adaptive sizing + sparsity + ensemble + multi-run.

    Args:
        X: Time series (T, K) as numpy array.
        n_runs: Number of training runs to average over.
        device: 'cpu' or 'cuda'.
        use_sparsity: Add L1 penalty on projection weights.
        use_ensemble: Blend Jacobian + Weight scores.
        adaptive_size: Scale model size with K.

    Returns:
        S: Causal score matrix (K, K).
        S_std: Standard deviation across runs (K, K).
    """
    if isinstance(X, np.ndarray):
        K = X.shape[1]
    else:
        K = X.shape[1]

    # Adaptive model size
    if adaptive_size:
        params = adaptive_model_size(K)
    else:
        params = {"d_model": 64, "d_state": 16, "n_layers": 4}

    all_scores = []

    for run in range(n_runs):
        torch.manual_seed(run * 42)

        if use_sparsity:
            model = train_with_sparsity(
                X,
                d_model=params["d_model"],
                d_state=params["d_state"],
                n_layers=params["n_layers"],
                n_epochs=200,
                l1_lambda=0.001,
                device=device,
            )
        else:
            model = MambaForTimeSeries(
                n_variables=K,
                d_model=params["d_model"],
                d_state=params["d_state"],
                n_layers=params["n_layers"],
            )
            trainer = SSMTrainer(model, lr=1e-3, device=device)
            trainer.train(X, n_epochs=200, verbose=False)

        X_tensor = torch.FloatTensor(X) if isinstance(X, np.ndarray) else X

        if use_ensemble:
            S = extract_ensemble_score(model, X_tensor, device, alpha=0.5)
        else:
            S = extract_weight_score(model)

        all_scores.append(S)

    S_mean = np.mean(all_scores, axis=0)
    S_std = np.std(all_scores, axis=0)

    np.fill_diagonal(S_mean, 0)
    if S_mean.max() > 0:
        S_mean = S_mean / S_mean.max()

    return S_mean, S_std
