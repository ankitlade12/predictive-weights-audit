"""
Explicit lag-resolved bottleneck models.

This module tests a narrower hypothesis than Exp 13:

    If the input projection is split by lag, can the endpoint bottleneck
    product recover a lag-specific dependency tensor?

The model predicts x[t] from [x[t-1], ..., x[t-p]] and extracts:

    S_lag[tau] = |W_out @ W_in_tau|

where tau=0 corresponds to lag 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLasso, Ridge
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from .lag_extraction import lag_score_to_aggregate
from .utils import compute_metrics

ArrayLike = np.ndarray | torch.Tensor
NormalizeMode = Literal["global", "per_lag", "none"]
LaggedLinearMethod = Literal["ols", "ridge", "lasso", "rrr"]


def _as_float_tensor(X: ArrayLike, device: str) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        tensor = X.float()
    else:
        tensor = torch.as_tensor(X, dtype=torch.float32)
    return tensor.to(device)


def _validate_segment_lengths(
    segment_lengths: Sequence[int],
    total_length: int,
    max_lag: int,
) -> list[int]:
    lengths_array = np.asarray(segment_lengths)
    if lengths_array.ndim != 1:
        raise ValueError("segment_lengths must be one-dimensional.")
    if (
        np.issubdtype(lengths_array.dtype, np.bool_)
        or not np.issubdtype(lengths_array.dtype, np.number)
        or np.issubdtype(lengths_array.dtype, np.complexfloating)
        or not np.all(np.isfinite(lengths_array))
        or not np.all(lengths_array == np.floor(lengths_array))
    ):
        raise ValueError("segment_lengths must contain integer-like values.")

    lengths = lengths_array.astype(np.int64).tolist()
    if any(length <= 0 for length in lengths):
        raise ValueError("segment_lengths must contain positive lengths.")
    if sum(lengths) != total_length:
        raise ValueError("segment_lengths must sum to T.")
    if any(length <= max_lag for length in lengths):
        raise ValueError("Each segment must have more rows than max_lag.")
    return lengths


def make_lagged_dataset(
    X: ArrayLike,
    max_lag: int,
    device: str = "cpu",
    segment_lengths: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a time series into supervised lagged examples.

    Feature order is lag-major:
        [x[t-1], x[t-2], ..., x[t-max_lag]]

    When segment_lengths is provided, each segment is windowed independently.

    Returns:
        features: [N, max_lag * K]
        targets: [N, K]
        where N is T - max_lag for one trajectory, or the sum of
        (segment_length - max_lag) across segments.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")

    X_tensor = _as_float_tensor(X, device)
    if X_tensor.ndim != 2:
        raise ValueError("X must have shape [T, K].")

    T = X_tensor.shape[0]
    if T <= max_lag:
        raise ValueError("Need more timesteps than max_lag.")

    if segment_lengths is None:
        lengths = [T]
    else:
        lengths = _validate_segment_lengths(segment_lengths, T, max_lag)

    feature_chunks = []
    target_chunks = []
    start = 0
    for length in lengths:
        segment = X_tensor[start : start + length]
        lagged_blocks = [segment[max_lag - lag - 1 : length - lag - 1] for lag in range(max_lag)]
        feature_chunks.append(torch.cat(lagged_blocks, dim=1))
        target_chunks.append(segment[max_lag:length])
        start += length

    return torch.cat(feature_chunks, dim=0), torch.cat(target_chunks, dim=0)


def _validate_target_valid_mask(
    target_valid_mask: ArrayLike,
    total_length: int,
    n_variables: int,
) -> np.ndarray:
    mask = np.asarray(target_valid_mask)
    if mask.dtype != np.bool_ or mask.shape != (total_length, n_variables):
        raise ValueError("target_valid_mask must be boolean with shape [T, K].")
    if not np.all(mask.any(axis=0)):
        raise ValueError("Each target must have at least one valid outcome.")
    return mask


def make_lagged_target_mask(
    target_valid_mask: ArrayLike,
    max_lag: int,
    device: str = "cpu",
    segment_lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """Align a full-length target-valid mask to segment-safe prediction rows."""
    mask = np.asarray(target_valid_mask)
    if mask.ndim != 2:
        raise ValueError("target_valid_mask must have shape [T, K].")
    validated = _validate_target_valid_mask(mask, mask.shape[0], mask.shape[1])
    lengths = (
        [len(mask)]
        if segment_lengths is None
        else _validate_segment_lengths(segment_lengths, len(mask), max_lag)
    )
    chunks = []
    start = 0
    for length in lengths:
        chunks.append(validated[start + max_lag : start + length])
        start += length
    return torch.as_tensor(
        np.concatenate(chunks),
        dtype=torch.bool,
        device=device,
    )


def normalize_lagged_score(
    S_lag: np.ndarray,
    mode: NormalizeMode = "global",
) -> np.ndarray:
    """Normalize a lag score tensor after zeroing same-variable entries."""
    S_lag = np.asarray(S_lag, dtype=float).copy()
    for tau in range(S_lag.shape[0]):
        np.fill_diagonal(S_lag[tau], 0.0)

    if mode == "global":
        max_value = np.nanmax(S_lag)
        if np.isfinite(max_value) and max_value > 0:
            S_lag = S_lag / max_value
    elif mode == "per_lag":
        for tau in range(S_lag.shape[0]):
            max_value = np.nanmax(S_lag[tau])
            if np.isfinite(max_value) and max_value > 0:
                S_lag[tau] = S_lag[tau] / max_value
    elif mode != "none":
        raise ValueError(f"Unknown normalize mode: {mode}")

    return S_lag


class LaggedBottleneckModel(nn.Module):
    """Low-rank lagged predictor with lag-split endpoint projections."""

    def __init__(
        self,
        n_variables: int,
        max_lag: int,
        d_bottleneck: int = 32,
        bias: bool = True,
    ):
        super().__init__()
        if n_variables < 1:
            raise ValueError("n_variables must be >= 1.")
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1.")

        self.n_variables = n_variables
        self.max_lag = max_lag
        self.d_bottleneck = d_bottleneck
        self.input_proj = nn.Linear(n_variables * max_lag, d_bottleneck, bias=bias)
        self.output_proj = nn.Linear(d_bottleneck, n_variables, bias=bias)

    def forward(self, lagged_x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(lagged_x)
        return self.output_proj(z)


def extract_lagged_weight_score(
    model: LaggedBottleneckModel,
    normalize: NormalizeMode = "global",
) -> np.ndarray:
    """
    Extract S_lag[tau, target, source] = |W_out @ W_in_tau|.

    tau=0 means lag 1.
    """
    W_in = model.input_proj.weight.detach()
    W_out = model.output_proj.weight.detach()
    K = model.n_variables

    scores = []
    for tau in range(model.max_lag):
        start = tau * K
        stop = start + K
        scores.append((W_out @ W_in[:, start:stop]).abs().cpu().numpy())

    return normalize_lagged_score(np.stack(scores, axis=0), mode=normalize)


def _coefficient_matrix_to_lag_score(
    coefficients: np.ndarray,
    max_lag: int,
    n_variables: int,
    normalize: NormalizeMode = "global",
) -> np.ndarray:
    """Convert [target, lag*source] coefficients into S_lag[tau, target, source]."""
    S_lag = np.zeros((max_lag, n_variables, n_variables))
    for tau in range(max_lag):
        start = tau * n_variables
        stop = start + n_variables
        S_lag[tau] = np.abs(coefficients[:, start:stop])
    return normalize_lagged_score(S_lag, mode=normalize)


@dataclass
class LaggedLinearBaselineResult:
    method: str
    score: np.ndarray
    train_prediction_mse: float
    test_prediction_mse: float | None
    params: dict[str, float | int | str] | None = None


def _fit_regularized_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit multi-output ridge coefficients as [features, targets] plus intercept."""
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    X_centered = X - x_mean
    y_centered = y - y_mean

    gram = X_centered.T @ X_centered
    if alpha > 0:
        gram = gram + alpha * np.eye(gram.shape[0])
        coefficients = np.linalg.solve(gram, X_centered.T @ y_centered)
    else:
        coefficients = np.linalg.pinv(X_centered) @ y_centered

    intercept = y_mean.reshape(-1) - x_mean.reshape(-1) @ coefficients
    return coefficients, intercept


def _fit_reduced_rank_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    rank: int,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit reduced-rank regression coefficients as [features, targets]."""
    full_coefficients, _ = _fit_regularized_coefficients(X, y, alpha=alpha)
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    X_centered = X - x_mean

    y_hat = X_centered @ full_coefficients
    max_rank = min(y_hat.shape)
    rank = min(max(int(rank), 1), max_rank)

    _, _, vt = np.linalg.svd(y_hat, full_matrices=False)
    target_basis = vt[:rank].T
    coefficients = full_coefficients @ target_basis @ target_basis.T
    intercept = y_mean.reshape(-1) - x_mean.reshape(-1) @ coefficients
    return coefficients, intercept


def _predict_from_coefficients(
    X: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    return X @ coefficients + intercept


def fit_lagged_linear_baseline(
    X_train: ArrayLike,
    max_lag: int,
    method: LaggedLinearMethod = "ols",
    alpha: float = 1e-3,
    rank: int | None = None,
    X_test: ArrayLike | None = None,
    normalize: NormalizeMode = "global",
    train_segment_lengths: Sequence[int] | None = None,
    test_segment_lengths: Sequence[int] | None = None,
    train_target_valid_mask: ArrayLike | None = None,
    test_target_valid_mask: ArrayLike | None = None,
) -> LaggedLinearBaselineResult:
    """
    Fit a standard lagged linear baseline and extract lag-specific coefficients.

    This is the key sanity baseline for the explicit lagged bottleneck: if OLS or
    Ridge produce the same score tensor, the method claim has to be framed as a
    bottleneck factorization/low-rank variant rather than a new estimator.
    """
    train_features, train_targets = make_lagged_dataset(
        X_train,
        max_lag=max_lag,
        device="cpu",
        segment_lengths=train_segment_lengths,
    )
    X_np = train_features.detach().cpu().numpy()
    y_np = train_targets.detach().cpu().numpy()
    lagged_train_mask = None
    if train_target_valid_mask is not None:
        if method == "rrr":
            raise ValueError("RRR does not support target-valid masks.")
        lagged_train_mask = make_lagged_target_mask(
            train_target_valid_mask,
            max_lag=max_lag,
            segment_lengths=train_segment_lengths,
        ).numpy()

    params: dict[str, float | int | str] = {"method": method}

    if method == "rrr":
        effective_rank = rank or min(y_np.shape[1], max(1, y_np.shape[0] // 4))
        coefficients, intercept = _fit_reduced_rank_coefficients(
            X_np,
            y_np,
            rank=effective_rank,
            alpha=alpha,
        )
        coef_for_score = coefficients.T
        train_pred = _predict_from_coefficients(X_np, coefficients, intercept)
        params.update({"alpha": alpha, "rank": effective_rank})
    elif method in {"ols", "ridge", "lasso"} and lagged_train_mask is None:
        if method == "ols":
            estimator = LinearRegression()
            params["alpha"] = 0.0
        elif method == "ridge":
            estimator = Ridge(alpha=alpha)
            params["alpha"] = alpha
        else:
            estimator = MultiTaskLasso(alpha=alpha, max_iter=10_000, random_state=0)
            params["alpha"] = alpha

        estimator.fit(X_np, y_np)
        coef_for_score = estimator.coef_
        train_pred = estimator.predict(X_np)
    elif method in {"ols", "ridge", "lasso"}:
        estimators = []
        coefficient_rows = []
        train_pred = np.empty_like(y_np)
        for target in range(y_np.shape[1]):
            valid_rows = lagged_train_mask[:, target]
            if method == "ols":
                target_estimator = LinearRegression()
                params["alpha"] = 0.0
            elif method == "ridge":
                target_estimator = Ridge(alpha=alpha)
                params["alpha"] = alpha
            else:
                target_estimator = Lasso(
                    alpha=alpha,
                    max_iter=10_000,
                    random_state=0,
                )
                params["alpha"] = alpha
            target_estimator.fit(
                X_np[valid_rows],
                y_np[valid_rows, target],
            )
            estimators.append(target_estimator)
            coefficient_rows.append(target_estimator.coef_)
            train_pred[:, target] = target_estimator.predict(X_np)
        coef_for_score = np.stack(coefficient_rows)
    else:
        raise ValueError(f"Unknown lagged linear baseline: {method}")

    squared_train_error = (train_pred - y_np) ** 2
    train_mse = float(
        squared_train_error.mean()
        if lagged_train_mask is None
        else squared_train_error[lagged_train_mask].mean()
    )

    test_mse = None
    if X_test is not None:
        test_features, test_targets = make_lagged_dataset(
            X_test,
            max_lag=max_lag,
            device="cpu",
            segment_lengths=test_segment_lengths,
        )
        X_test_np = test_features.detach().cpu().numpy()
        y_test_np = test_targets.detach().cpu().numpy()
        lagged_test_mask = None
        if test_target_valid_mask is not None:
            lagged_test_mask = make_lagged_target_mask(
                test_target_valid_mask,
                max_lag=max_lag,
                segment_lengths=test_segment_lengths,
            ).numpy()
        if method == "rrr":
            test_pred = _predict_from_coefficients(X_test_np, coefficients, intercept)
        elif lagged_train_mask is not None:
            test_pred = np.column_stack(
                [target_estimator.predict(X_test_np) for target_estimator in estimators]
            )
        else:
            test_pred = estimator.predict(X_test_np)
        squared_test_error = (test_pred - y_test_np) ** 2
        test_mse = float(
            squared_test_error.mean()
            if lagged_test_mask is None
            else squared_test_error[lagged_test_mask].mean()
        )

    score = _coefficient_matrix_to_lag_score(
        coef_for_score,
        max_lag=max_lag,
        n_variables=y_np.shape[1],
        normalize=normalize,
    )
    return LaggedLinearBaselineResult(
        method=method,
        score=score,
        train_prediction_mse=train_mse,
        test_prediction_mse=test_mse,
        params=params,
    )


def fit_tuned_lagged_linear_baseline(
    X_train: ArrayLike,
    max_lag: int,
    method: LaggedLinearMethod,
    alpha_values: list[float] | tuple[float, ...] = (1e-3,),
    rank_values: list[int] | tuple[int, ...] = (4, 8, 16),
    validation_fraction: float = 0.25,
    X_test: ArrayLike | None = None,
    normalize: NormalizeMode = "global",
    segment_lengths: Sequence[int] | None = None,
    test_segment_lengths: Sequence[int] | None = None,
    target_valid_mask: ArrayLike | None = None,
    test_target_valid_mask: ArrayLike | None = None,
) -> LaggedLinearBaselineResult:
    """
    Tune a lagged linear baseline by validation prediction MSE, then refit.

    The tuning signal is prediction quality only; graph labels are not used.
    """
    X_array = np.asarray(X_train, dtype=np.float32)
    min_len = max_lag + 2
    validated_segment_lengths = None
    if segment_lengths is not None:
        validated_segment_lengths = _validate_segment_lengths(
            segment_lengths,
            X_array.shape[0],
            max_lag,
        )
    validated_target_mask = None
    if target_valid_mask is not None:
        validated_target_mask = _validate_target_valid_mask(
            target_valid_mask,
            X_array.shape[0],
            X_array.shape[1],
        )
        if method == "rrr":
            raise ValueError("RRR does not support target-valid masks.")

    if (validated_segment_lengths is None and X_array.shape[0] < 2 * min_len) or (
        validated_segment_lengths is not None
        and any(length < 2 * min_len for length in validated_segment_lengths)
    ):
        if method == "rrr":
            return fit_lagged_linear_baseline(
                X_train,
                max_lag=max_lag,
                method=method,
                alpha=alpha_values[0],
                rank=rank_values[0],
                X_test=X_test,
                normalize=normalize,
                train_segment_lengths=validated_segment_lengths,
                test_segment_lengths=test_segment_lengths,
                train_target_valid_mask=validated_target_mask,
                test_target_valid_mask=test_target_valid_mask,
            )
        return fit_lagged_linear_baseline(
            X_train,
            max_lag=max_lag,
            method=method,
            alpha=alpha_values[0],
            X_test=X_test,
            normalize=normalize,
            train_segment_lengths=validated_segment_lengths,
            test_segment_lengths=test_segment_lengths,
            train_target_valid_mask=validated_target_mask,
            test_target_valid_mask=test_target_valid_mask,
        )

    fit_segment_lengths = None
    validation_segment_lengths = None
    if validated_segment_lengths is None:
        split = int(X_array.shape[0] * (1.0 - validation_fraction))
        split = min(max(split, min_len), X_array.shape[0] - min_len)
        fit_segment = X_array[:split]
        validation_segment = X_array[split:]
        fit_target_mask = None if validated_target_mask is None else validated_target_mask[:split]
        validation_target_mask = (
            None if validated_target_mask is None else validated_target_mask[split:]
        )
    else:
        fit_chunks = []
        validation_chunks = []
        fit_segment_lengths = []
        validation_segment_lengths = []
        fit_mask_chunks = []
        validation_mask_chunks = []
        start = 0
        for length in validated_segment_lengths:
            split = int(length * (1.0 - validation_fraction))
            split = min(max(split, min_len), length - min_len)
            segment = X_array[start : start + length]
            fit_chunks.append(segment[:split])
            validation_chunks.append(segment[split:])
            fit_segment_lengths.append(split)
            validation_segment_lengths.append(length - split)
            if validated_target_mask is not None:
                segment_mask = validated_target_mask[start : start + length]
                fit_mask_chunks.append(segment_mask[:split])
                validation_mask_chunks.append(segment_mask[split:])
            start += length
        fit_segment = np.concatenate(fit_chunks, axis=0)
        validation_segment = np.concatenate(validation_chunks, axis=0)
        fit_target_mask = None if validated_target_mask is None else np.concatenate(fit_mask_chunks)
        validation_target_mask = (
            None if validated_target_mask is None else np.concatenate(validation_mask_chunks)
        )

    candidates = [(0.0, None)] if method == "ols" else []
    if method in {"ridge", "lasso"}:
        candidates = [(float(alpha), None) for alpha in alpha_values]
    elif method == "rrr":
        candidates = [(float(alpha), int(rank)) for alpha in alpha_values for rank in rank_values]
    elif method != "ols":
        raise ValueError(f"Unknown lagged linear baseline: {method}")

    best_result = None
    best_validation_mse = float("inf")
    for alpha, rank in candidates:
        result = fit_lagged_linear_baseline(
            fit_segment,
            max_lag=max_lag,
            method=method,
            alpha=alpha,
            rank=rank,
            X_test=validation_segment,
            normalize=normalize,
            train_segment_lengths=fit_segment_lengths,
            test_segment_lengths=validation_segment_lengths,
            train_target_valid_mask=fit_target_mask,
            test_target_valid_mask=validation_target_mask,
        )
        validation_mse = result.test_prediction_mse
        if validation_mse is not None and validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_result = result

    if best_result is None:
        raise RuntimeError(f"No valid tuning candidate found for baseline {method}.")

    best_alpha = float(best_result.params.get("alpha", 0.0)) if best_result.params else 0.0
    best_rank = None
    if best_result.params and "rank" in best_result.params:
        best_rank = int(best_result.params["rank"])

    refit_result = fit_lagged_linear_baseline(
        X_train,
        max_lag=max_lag,
        method=method,
        alpha=best_alpha,
        rank=best_rank,
        X_test=X_test,
        normalize=normalize,
        train_segment_lengths=validated_segment_lengths,
        test_segment_lengths=test_segment_lengths,
        train_target_valid_mask=validated_target_mask,
        test_target_valid_mask=test_target_valid_mask,
    )
    refit_result.params = dict(refit_result.params or {})
    refit_result.params["validation_mse"] = best_validation_mse
    return refit_result


def _flatten_lag_off_diagonal(
    S_lag: np.ndarray,
    G_lag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_lag = min(S_lag.shape[0], G_lag.shape[0])
    K = S_lag.shape[1]
    mask = ~np.eye(K, dtype=bool)

    scores = []
    labels = []
    for tau in range(max_lag):
        scores.append(np.abs(S_lag[tau][mask]).reshape(-1))
        labels.append(G_lag[tau][mask].reshape(-1))
    return np.concatenate(scores), np.concatenate(labels)


def _binary_auc_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(auc(recall, precision)),
    }


def evaluate_lagged_scores(
    S_lag: np.ndarray,
    G_lag: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate lag-resolved recovery with both aggregate and all-lag metrics.

    The flat all-lag metric is stricter than averaging only non-empty lags:
    it penalizes assigning a true edge to the wrong lag.
    """
    max_lag = min(S_lag.shape[0], G_lag.shape[0])
    S_lag = np.asarray(S_lag[:max_lag], dtype=float).copy()
    G_lag = np.asarray(G_lag[:max_lag], dtype=float).copy()
    K = S_lag.shape[1]

    for tau in range(max_lag):
        np.fill_diagonal(S_lag[tau], 0.0)
        np.fill_diagonal(G_lag[tau], 0.0)

    S_agg = lag_score_to_aggregate(S_lag)
    G_agg = (G_lag.sum(axis=0) > 0).astype(float)
    np.fill_diagonal(G_agg, 0.0)
    aggregate_metrics = compute_metrics(S_agg, G_agg)

    flat_scores, flat_labels = _flatten_lag_off_diagonal(S_lag, G_lag)
    flat_metrics = _binary_auc_metrics(flat_scores, flat_labels)

    per_lag_aurocs = []
    for tau in range(max_lag):
        labels_tau = G_lag[tau][~np.eye(K, dtype=bool)]
        if len(np.unique(labels_tau)) > 1:
            metrics_tau = compute_metrics(S_lag[tau], G_lag[tau])
            per_lag_aurocs.append(metrics_tau["auroc"])

    true_edge_lag_hits = []
    for target in range(K):
        for source in range(K):
            if target == source or G_lag[:, target, source].sum() == 0:
                continue
            pred_lag = int(np.argmax(np.abs(S_lag[:, target, source])))
            true_lags = np.flatnonzero(G_lag[:, target, source] > 0)
            true_edge_lag_hits.append(float(pred_lag in true_lags))

    return {
        "aggregate_auroc": float(aggregate_metrics["auroc"]),
        "aggregate_auprc": float(aggregate_metrics["auprc"]),
        "aggregate_f1": float(aggregate_metrics["f1"]),
        "flat_lag_auroc": flat_metrics["auroc"],
        "flat_lag_auprc": flat_metrics["auprc"],
        "per_lag_auroc_mean": (float(np.mean(per_lag_aurocs)) if per_lag_aurocs else float("nan")),
        "per_lag_auroc_std": (float(np.std(per_lag_aurocs)) if per_lag_aurocs else float("nan")),
        "edge_lag_top1_acc": (
            float(np.mean(true_edge_lag_hits)) if true_edge_lag_hits else float("nan")
        ),
        "n_true_lag_edges": int(G_lag[:, ~np.eye(K, dtype=bool)].sum()),
    }


@dataclass
class LaggedBottleneckTrainingConfig:
    n_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_sparse: float = 0.0
    print_every: int = 25


class LaggedBottleneckTrainer:
    """Trainer for explicit lagged bottleneck predictors."""

    def __init__(
        self,
        model: LaggedBottleneckModel,
        config: LaggedBottleneckTrainingConfig | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config or LaggedBottleneckTrainingConfig()
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        self.history: dict[str, list[float]] = {
            "loss": [],
            "prediction_loss": [],
            "sparse_loss": [],
        }

    def train(
        self,
        X: ArrayLike,
        verbose: bool = True,
        segment_lengths: Sequence[int] | None = None,
        target_valid_mask: ArrayLike | None = None,
    ) -> dict[str, list[float]]:
        features, targets = make_lagged_dataset(
            X,
            self.model.max_lag,
            device=self.device,
            segment_lengths=segment_lengths,
        )
        lagged_target_mask = None
        if target_valid_mask is not None:
            lagged_target_mask = make_lagged_target_mask(
                target_valid_mask,
                self.model.max_lag,
                device=self.device,
                segment_lengths=segment_lengths,
            )
        n_examples = features.shape[0]
        steps_per_epoch = max(1, n_examples // self.config.batch_size)

        self.history = {key: [] for key in self.history}
        self.model.train()

        for epoch in range(self.config.n_epochs):
            ep_loss = 0.0
            ep_pred = 0.0
            ep_sparse = 0.0

            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n_examples, (self.config.batch_size,), device=self.device)
                batch_x = features[idx]
                batch_y = targets[idx]

                pred = self.model(batch_x)
                if lagged_target_mask is None:
                    pred_loss = self.loss_fn(pred, batch_y)
                else:
                    batch_mask = lagged_target_mask[idx]
                    squared_error = (pred - batch_y).square()
                    pred_loss = (squared_error * batch_mask).sum() / batch_mask.sum().clamp_min(1)
                sparse_loss = (
                    self.model.input_proj.weight.abs().sum()
                    + self.model.output_proj.weight.abs().sum()
                )
                loss = pred_loss + self.config.lambda_sparse * sparse_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                ep_loss += float(loss.item())
                ep_pred += float(pred_loss.item())
                ep_sparse += float(sparse_loss.item())

            denom = float(steps_per_epoch)
            self.history["loss"].append(ep_loss / denom)
            self.history["prediction_loss"].append(ep_pred / denom)
            self.history["sparse_loss"].append(ep_sparse / denom)

            if verbose and (epoch + 1) % self.config.print_every == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{self.config.n_epochs} | "
                    f"Loss: {self.history['loss'][-1]:.6f} | "
                    f"Pred: {self.history['prediction_loss'][-1]:.6f}"
                )

        return self.history

    def prediction_mse(
        self,
        X: ArrayLike,
        segment_lengths: Sequence[int] | None = None,
        target_valid_mask: ArrayLike | None = None,
    ) -> float:
        features, targets = make_lagged_dataset(
            X,
            self.model.max_lag,
            device=self.device,
            segment_lengths=segment_lengths,
        )
        lagged_target_mask = None
        if target_valid_mask is not None:
            lagged_target_mask = make_lagged_target_mask(
                target_valid_mask,
                self.model.max_lag,
                device=self.device,
                segment_lengths=segment_lengths,
            )
        self.model.eval()
        with torch.no_grad():
            pred = self.model(features)
            if lagged_target_mask is None:
                return float(F.mse_loss(pred, targets).item())
            squared_error = (pred - targets).square()
            return float(
                ((squared_error * lagged_target_mask).sum() / lagged_target_mask.sum()).item()
            )


def evaluate_lagged_bottleneck_model(
    model: LaggedBottleneckModel,
    X: ArrayLike,
    G_lag: np.ndarray,
    device: str = "cpu",
    segment_lengths: Sequence[int] | None = None,
    target_valid_mask: ArrayLike | None = None,
) -> dict[str, float]:
    """Extract lag scores and report lag-recovery plus prediction metrics."""
    S_lag = extract_lagged_weight_score(model, normalize="global")
    metrics = evaluate_lagged_scores(S_lag, G_lag)

    features, targets = make_lagged_dataset(
        X,
        model.max_lag,
        device=device,
        segment_lengths=segment_lengths,
    )
    lagged_target_mask = None
    if target_valid_mask is not None:
        lagged_target_mask = make_lagged_target_mask(
            target_valid_mask,
            model.max_lag,
            device=device,
            segment_lengths=segment_lengths,
        )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(features)
        if lagged_target_mask is None:
            metrics["prediction_mse"] = float(F.mse_loss(pred, targets).item())
        else:
            squared_error = (pred - targets).square()
            metrics["prediction_mse"] = float(
                ((squared_error * lagged_target_mask).sum() / lagged_target_mask.sum()).item()
            )

    return metrics
