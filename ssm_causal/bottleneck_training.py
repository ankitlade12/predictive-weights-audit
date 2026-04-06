"""
Utilities for causality-preserving bottleneck training.

This module implements the smallest prototype from the reset plan:
Model B (Sparse Bottleneck), which adds endpoint weight sparsity to the
standard forecasting objective while keeping the cheap weight-product
extractor as the default graph readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_model import MambaForTimeSeries
from .utils import compute_metrics

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_float_tensor(X: ArrayLike, device: str) -> torch.Tensor:
    """Move an array-like time series to the requested device."""
    if isinstance(X, torch.Tensor):
        tensor = X.float()
    else:
        tensor = torch.as_tensor(X, dtype=torch.float32)
    return tensor.to(device)


def extract_weight_score_tensor(
    model: nn.Module,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extract the endpoint bottleneck score as a torch tensor.

    S_weight = |W_out @ W_in|
    """
    W_in = model.input_proj.weight
    W_out = model.output_proj.weight

    score = (W_out @ W_in).abs()
    score = score.clone()
    score.fill_diagonal_(0.0)

    if normalize:
        max_value = score.max()
        if torch.isfinite(max_value).item() and max_value.item() > 0:
            score = score / max_value

    return score


def extract_weight_score(
    model: nn.Module,
    normalize: bool = True,
) -> np.ndarray:
    """Numpy wrapper for the endpoint bottleneck score."""
    return extract_weight_score_tensor(model, normalize=normalize).detach().cpu().numpy()


def endpoint_weight_l1_penalty(model: nn.Module) -> torch.Tensor:
    """Prototype Model B penalty: L1 on the endpoint projections."""
    return model.input_proj.weight.abs().sum() + model.output_proj.weight.abs().sum()


def score_density(
    score: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """Fraction of off-diagonal entries above a fixed normalized threshold."""
    mask = ~np.eye(score.shape[0], dtype=bool)
    return float((np.abs(score[mask]) > threshold).mean())


def compute_prediction_mse(
    model: nn.Module,
    X: ArrayLike,
    device: str = "cpu",
) -> float:
    """
    Compute one-step forecasting MSE on a contiguous evaluation segment.
    """
    X_tensor = _as_float_tensor(X, device)
    if X_tensor.ndim != 2 or X_tensor.shape[0] < 2:
        raise ValueError("X must have shape [T, K] with T >= 2.")

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = X_tensor[:-1].unsqueeze(0)
        targets = X_tensor[1:].unsqueeze(0)
        predictions = model(inputs)
        return float(F.mse_loss(predictions, targets).item())


def evaluate_bottleneck_model(
    model: nn.Module,
    X_eval: ArrayLike,
    G_true: np.ndarray,
    device: str = "cpu",
    density_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Evaluate the default bottleneck extractor plus prediction quality.
    """
    score = extract_weight_score(model)
    metrics = compute_metrics(score, G_true)
    metrics["prediction_mse"] = compute_prediction_mse(model, X_eval, device=device)
    metrics["graph_density"] = score_density(score, threshold=density_threshold)
    metrics["mean_abs_score"] = float(
        np.abs(score[~np.eye(score.shape[0], dtype=bool)]).mean()
    )
    return metrics


def subsequence_stability_loss(
    model: nn.Module,
    X: torch.Tensor,
    seq_len: int = 100,
) -> torch.Tensor:
    """
    Model C loss: penalize disagreement between extracted scores
    from two different subsequences of the same time series.

    If the model encodes stable dependency structure, the extracted
    graph should not change across different windows.
    """
    T = X.shape[0]
    effective_len = min(seq_len, T // 2 - 1)
    if effective_len < 2:
        return torch.tensor(0.0, device=X.device)

    # Sample two non-overlapping subsequences
    mid = T // 2
    start_a = torch.randint(0, mid - effective_len, (1,)).item()
    start_b = torch.randint(mid, T - effective_len, (1,)).item()

    sub_a = X[start_a:start_a + effective_len].unsqueeze(0)
    sub_b = X[start_b:start_b + effective_len].unsqueeze(0)

    # Forward pass on both (populates internal caches)
    with torch.no_grad():
        _ = model(sub_a)
        _ = model(sub_b)

    # Extract scores (same weights, different data context doesn't
    # affect W_in/W_out — but it may affect B/C caches if we use them later)
    S = extract_weight_score_tensor(model, normalize=True)

    # For weight projection, the score is deterministic given weights,
    # so stability across subsequences is automatic.
    # The real value of this loss is when combined with Jacobian:
    # compute Jacobian on two subsequences and penalize disagreement.

    # Jacobian-based stability: compute sensitivity on both windows
    K = X.shape[1]
    model.eval()

    def _jacobian_score(x_sub):
        x_sub = x_sub.requires_grad_(True)
        y = model(x_sub)
        S_j = torch.zeros(K, K, device=X.device)
        for i in range(K):
            model.zero_grad()
            if x_sub.grad is not None:
                x_sub.grad.zero_()
            y[0, :, i].sum().backward(retain_graph=(i < K - 1))
            S_j[i, :] = x_sub.grad[0].abs().mean(dim=0)
        return S_j

    S_a = _jacobian_score(sub_a.detach().clone())
    S_b = _jacobian_score(sub_b.detach().clone())

    # Normalize
    if S_a.max() > 0:
        S_a = S_a / S_a.max()
    if S_b.max() > 0:
        S_b = S_b / S_b.max()

    # Frobenius disagreement on off-diagonal
    mask = ~torch.eye(K, dtype=torch.bool, device=X.device)
    loss = ((S_a[mask] - S_b[mask]) ** 2).mean()

    model.train()
    return loss


def weight_jacobian_consistency_loss(
    model: nn.Module,
    X: torch.Tensor,
    seq_len: int = 50,
) -> torch.Tensor:
    """
    Model D loss: encourage agreement between the cheap weight-projection
    score and the richer Jacobian score.

    When they agree, the bottleneck is clean.
    When they disagree, the bottleneck may be entangled.
    """
    K = X.shape[1]
    effective_len = min(seq_len, X.shape[0] - 1)
    if effective_len < 2:
        return torch.tensor(0.0, device=X.device)

    # Weight score (cheap, deterministic)
    S_w = extract_weight_score_tensor(model, normalize=True)

    # Jacobian score on a small window (expensive but informative)
    start = torch.randint(0, X.shape[0] - effective_len, (1,)).item()
    x_sub = X[start:start + effective_len].unsqueeze(0).requires_grad_(True)

    model.eval()
    y = model(x_sub)

    S_j = torch.zeros(K, K, device=X.device)
    for i in range(K):
        model.zero_grad()
        if x_sub.grad is not None:
            x_sub.grad.zero_()
        y[0, :, i].sum().backward(retain_graph=(i < K - 1))
        S_j[i, :] = x_sub.grad[0].abs().mean(dim=0)

    S_j = S_j.detach()
    S_j.fill_diagonal_(0.0)
    if S_j.max() > 0:
        S_j = S_j / S_j.max()

    model.train()

    # L2 disagreement on off-diagonal entries
    mask = ~torch.eye(K, dtype=torch.bool, device=X.device)
    loss = ((S_w[mask] - S_j[mask]) ** 2).mean()
    return loss


@dataclass
class BottleneckTrainingConfig:
    """
    Configuration for causality-preserving bottleneck training.

    Supports four model variants:
        Model A: L = L_pred                                    (lambda_sparse=0, lambda_stable=0, lambda_consist=0)
        Model B: L = L_pred + lambda_sparse * L_sparse         (lambda_stable=0, lambda_consist=0)
        Model C: L = L_pred + lambda_stable * L_stable          (lambda_sparse=0, lambda_consist=0)
        Model D: L = L_pred + all three regularizers
    """

    n_epochs: int = 100
    batch_size: int = 32
    seq_len: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_sparse: float = 0.0
    lambda_stable: float = 0.0
    lambda_consist: float = 0.0
    consist_every_n: int = 5  # Compute consistency loss every N epochs (expensive)
    print_every: int = 10


class CausalBottleneckTrainer:
    """
    Configurable trainer for all four causality-preserving variants.

        Model A: L = L_pred
        Model B: L = L_pred + lambda_sparse * L_sparse_weights
        Model C: L = L_pred + lambda_stable * L_stable
        Model D: L = L_pred + lambda_sparse * L_sparse + lambda_stable * L_stable + lambda_consist * L_consist
    """

    def __init__(
        self,
        model: MambaForTimeSeries,
        config: BottleneckTrainingConfig | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or BottleneckTrainingConfig()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.config.n_epochs, 1),
            eta_min=self.config.lr * 0.01,
        )
        self.loss_fn = nn.MSELoss()
        self.history = {
            "loss": [],
            "prediction_loss": [],
            "sparse_loss": [],
            "stable_loss": [],
            "consist_loss": [],
        }

    def _sample_batch(
        self,
        X: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random contiguous subsequences for next-step prediction."""
        T = X.shape[0]
        effective_seq_len = min(seq_len, T - 1)
        if effective_seq_len < 1:
            raise ValueError("Need at least 2 timesteps for next-step prediction.")

        max_start = T - effective_seq_len - 1
        if max_start <= 0:
            starts = [0] * batch_size
        else:
            starts = torch.randint(0, max_start + 1, (batch_size,)).tolist()

        batch_x = torch.stack([X[s : s + effective_seq_len] for s in starts])
        batch_y = torch.stack([X[s + 1 : s + effective_seq_len + 1] for s in starts])
        return batch_x, batch_y

    def train(
        self,
        X: ArrayLike,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train with optional causality-preserving regularization.
        """
        X_tensor = _as_float_tensor(X, self.device)
        T = X_tensor.shape[0]
        seq_len = min(self.config.seq_len, T - 1)
        steps_per_epoch = max(1, T // max(self.config.batch_size * max(seq_len, 1), 1))

        self.history = {k: [] for k in self.history}
        use_stable = self.config.lambda_stable > 0
        use_consist = self.config.lambda_consist > 0

        self.model.train()
        for epoch in range(self.config.n_epochs):
            ep_total = 0.0
            ep_pred = 0.0
            ep_sparse = 0.0
            ep_stable = 0.0
            ep_consist = 0.0

            for _ in range(steps_per_epoch):
                batch_x, batch_y = self._sample_batch(
                    X_tensor, self.config.batch_size, seq_len,
                )

                predictions = self.model(batch_x)
                pred_loss = self.loss_fn(predictions, batch_y)

                # Sparsity (Model B, D)
                sparse_loss = endpoint_weight_l1_penalty(self.model)

                # Stability (Model C, D) — every epoch
                stable_loss = torch.tensor(0.0, device=self.device)
                if use_stable:
                    stable_loss = subsequence_stability_loss(
                        self.model, X_tensor, seq_len=min(50, seq_len),
                    )

                # Consistency (Model D) — every N epochs (expensive)
                consist_loss = torch.tensor(0.0, device=self.device)
                if use_consist and (epoch + 1) % self.config.consist_every_n == 0:
                    consist_loss = weight_jacobian_consistency_loss(
                        self.model, X_tensor, seq_len=min(30, seq_len),
                    )

                total_loss = (
                    pred_loss
                    + self.config.lambda_sparse * sparse_loss
                    + self.config.lambda_stable * stable_loss
                    + self.config.lambda_consist * consist_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                ep_total += float(total_loss.item())
                ep_pred += float(pred_loss.item())
                ep_sparse += float(sparse_loss.item())
                ep_stable += float(stable_loss.item())
                ep_consist += float(consist_loss.item())

            steps = float(steps_per_epoch)
            self.history["loss"].append(ep_total / steps)
            self.history["prediction_loss"].append(ep_pred / steps)
            self.history["sparse_loss"].append(ep_sparse / steps)
            self.history["stable_loss"].append(ep_stable / steps)
            self.history["consist_loss"].append(ep_consist / steps)
            self.scheduler.step()

            if verbose and (epoch + 1) % self.config.print_every == 0:
                parts = [
                    f"Epoch {epoch + 1:4d}/{self.config.n_epochs}",
                    f"Total: {self.history['loss'][-1]:.6f}",
                    f"Pred: {self.history['prediction_loss'][-1]:.6f}",
                ]
                if self.config.lambda_sparse > 0:
                    parts.append(f"Sparse: {self.history['sparse_loss'][-1]:.3f}")
                if use_stable:
                    parts.append(f"Stable: {self.history['stable_loss'][-1]:.4f}")
                if use_consist:
                    parts.append(f"Consist: {self.history['consist_loss'][-1]:.4f}")
                print(" | ".join(parts))

        return self.history
