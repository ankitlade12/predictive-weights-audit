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


@dataclass
class BottleneckTrainingConfig:
    """
    Lightweight configuration for causality-preserving bottleneck training.
    """

    n_epochs: int = 100
    batch_size: int = 32
    seq_len: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_sparse: float = 0.0
    print_every: int = 10


class CausalBottleneckTrainer:
    """
    Configurable trainer for baseline and sparse-bottleneck variants.

    The current implementation covers:
        Model A: L = L_pred
        Model B: L = L_pred + lambda_sparse * L_sparse_weights
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
        Train on next-step forecasting with optional sparse bottleneck regularization.
        """
        X_tensor = _as_float_tensor(X, self.device)
        T = X_tensor.shape[0]
        seq_len = min(self.config.seq_len, T - 1)
        steps_per_epoch = max(1, T // max(self.config.batch_size * max(seq_len, 1), 1))
        self.history = {
            "loss": [],
            "prediction_loss": [],
            "sparse_loss": [],
        }

        self.model.train()
        for epoch in range(self.config.n_epochs):
            loss_total = 0.0
            pred_total = 0.0
            sparse_total = 0.0

            for _ in range(steps_per_epoch):
                batch_x, batch_y = self._sample_batch(
                    X_tensor,
                    batch_size=self.config.batch_size,
                    seq_len=seq_len,
                )

                predictions = self.model(batch_x)
                pred_loss = self.loss_fn(predictions, batch_y)
                sparse_loss = endpoint_weight_l1_penalty(self.model)
                total_loss = pred_loss + self.config.lambda_sparse * sparse_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                loss_total += float(total_loss.item())
                pred_total += float(pred_loss.item())
                sparse_total += float(sparse_loss.item())

            steps = float(steps_per_epoch)
            self.history["loss"].append(loss_total / steps)
            self.history["prediction_loss"].append(pred_total / steps)
            self.history["sparse_loss"].append(sparse_total / steps)
            self.scheduler.step()

            if verbose and (epoch + 1) % self.config.print_every == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{self.config.n_epochs}"
                    f" | Total: {self.history['loss'][-1]:.6f}"
                    f" | Pred: {self.history['prediction_loss'][-1]:.6f}"
                    f" | Sparse: {self.history['sparse_loss'][-1]:.3f}"
                )

        return self.history
