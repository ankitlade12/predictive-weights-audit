"""Stable synthetic generators for lag-resolved dependency experiments."""

from __future__ import annotations

import numpy as np

from data.synthetic import generate_var1, generate_var2


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / np.maximum(std, 1e-8)


def companion_spectral_radius(A_lags: np.ndarray) -> float:
    max_lag, K, _ = A_lags.shape
    companion = np.zeros((K * max_lag, K * max_lag))
    companion[:K, : K * max_lag] = np.concatenate(A_lags, axis=1)
    if max_lag > 1:
        companion[K:, : K * (max_lag - 1)] = np.eye(K * (max_lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def ensure_stable_lag_coefficients(
    A_lags: np.ndarray,
    target_radius: float = 0.85,
) -> np.ndarray:
    radius = companion_spectral_radius(A_lags)
    if radius > target_radius:
        A_lags = A_lags * (target_radius / radius)
    return A_lags


def simulate_lagged_var(
    A_lags: np.ndarray,
    T: int,
    noise_std: float,
    seed: int,
    burnin: int = 300,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    max_lag, K, _ = A_lags.shape
    X = np.zeros((T + burnin, K))
    X[:max_lag] = rng.randn(max_lag, K) * noise_std

    for t in range(max_lag, T + burnin):
        value = np.zeros(K)
        for lag in range(max_lag):
            value += A_lags[lag] @ X[t - lag - 1]
        X[t] = value + rng.randn(K) * noise_std

    return standardize(X[burnin:])


def lag_graph_from_coefficients(A_lags: np.ndarray) -> np.ndarray:
    G_lag = (np.abs(A_lags) > 1e-10).astype(float)
    for lag in range(G_lag.shape[0]):
        np.fill_diagonal(G_lag[lag], 0.0)
    return G_lag


def pad_lag_graph(G_lag: np.ndarray, max_lag: int) -> np.ndarray:
    if G_lag.shape[0] >= max_lag:
        return G_lag[:max_lag].copy()
    padded = np.zeros((max_lag, G_lag.shape[1], G_lag.shape[2]))
    padded[: G_lag.shape[0]] = G_lag
    return padded


def train_test_split_time_series(
    X: np.ndarray,
    train_fraction: float,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    split = int(len(X) * train_fraction)
    min_len = max_lag + 2
    split = min(max(split, min_len), len(X) - min_len)
    return X[:split], X[split:]


def generate_var1_lagged(K: int, T: int, max_lag: int, seed: int):
    X, A, _ = generate_var1(K=K, T=T, graph_type="chain", seed=seed)
    G_lag = np.zeros((max_lag, K, K))
    G_lag[0] = (np.abs(A) > 1e-10).astype(float)
    np.fill_diagonal(G_lag[0], 0.0)
    return standardize(X), G_lag, "VAR1-chain"


def generate_var2_lagged(K: int, T: int, max_lag: int, seed: int):
    X, A_dict, _ = generate_var2(K=K, T=T, seed=seed)
    G_lag = np.zeros((max_lag, K, K))
    G_lag[0] = (np.abs(A_dict["A1"]) > 1e-10).astype(float)
    G_lag[1] = (np.abs(A_dict["A2"]) > 1e-10).astype(float)
    np.fill_diagonal(G_lag[0], 0.0)
    np.fill_diagonal(G_lag[1], 0.0)
    return standardize(X), G_lag, "VAR2-random"


def generate_mixed_lagged(K: int, T: int, max_lag: int, seed: int):
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.35] * K)
    A_lags[0, 1, 0] = 0.55
    A_lags[1, 2, 1] = 0.55
    A_lags[2, 3, 2] = 0.55
    A_lags = ensure_stable_lag_coefficients(A_lags)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.08, seed=seed)
    return X, lag_graph_from_coefficients(A_lags), "Mixed-lag-stable"


def generate_delayed_sparse_lagged(K: int, T: int, max_lag: int, seed: int):
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.35] * K)
    A_lags[2, 1, 0] = 0.65
    A_lags[3, 3, 2] = 0.65
    A_lags = ensure_stable_lag_coefficients(A_lags)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.08, seed=seed)
    return X, lag_graph_from_coefficients(A_lags), "Delayed-sparse-stable"


def generate_nonlinear_delayed_lagged(K: int, T: int, max_lag: int, seed: int):
    rng = np.random.RandomState(seed)
    X = np.zeros((T + 300, K))
    X[:max_lag] = rng.randn(max_lag, K) * 0.1

    for t in range(max_lag, T + 300):
        value = 0.35 * X[t - 1]
        value[1] += 0.60 * np.tanh(X[t - 3, 0])
        value[3] += 0.60 * np.tanh(X[t - 4, 2])
        X[t] = value + rng.randn(K) * 0.08

    G_lag = np.zeros((max_lag, K, K))
    G_lag[2, 1, 0] = 1.0
    G_lag[3, 3, 2] = 1.0
    return standardize(X[300:]), G_lag, "Nonlinear-delayed"


def generate_block_lowrank_lagged(K: int, T: int, max_lag: int, seed: int):
    """Block-sparse low-rank lagged VAR designed for sample-limited stress tests."""
    rng = np.random.RandomState(seed)
    A_lags = np.zeros((max_lag, K, K))
    A_lags[0] += np.diag([0.25] * K)

    rank = max(1, min(3, K // 3))
    support_size = max(2, K // 4)
    active_lags = sorted(set([0, min(2, max_lag - 1), min(4, max_lag - 1)]))

    for component in range(rank):
        lag = active_lags[component % len(active_lags)]
        sources = rng.choice(K, size=support_size, replace=False)
        targets = rng.choice(K, size=support_size, replace=False)

        source_vec = np.zeros(K)
        target_vec = np.zeros(K)
        source_vec[sources] = rng.uniform(0.6, 1.0, size=support_size) * rng.choice(
            [-1, 1], size=support_size
        )
        target_vec[targets] = rng.uniform(0.6, 1.0, size=support_size) * rng.choice(
            [-1, 1], size=support_size
        )
        A_lags[lag] += 0.45 * np.outer(target_vec, source_vec)

    for lag in range(max_lag):
        np.fill_diagonal(A_lags[lag], 0.25 if lag == 0 else 0.0)

    A_lags = ensure_stable_lag_coefficients(A_lags)
    X = simulate_lagged_var(A_lags, T=T, noise_std=0.1, seed=seed)
    return X, lag_graph_from_coefficients(A_lags), "Block-lowrank-lagged"


DATASETS = {
    "VAR1-chain": generate_var1_lagged,
    "VAR2-random": generate_var2_lagged,
    "Mixed-lag": generate_mixed_lagged,
    "Delayed-sparse": generate_delayed_sparse_lagged,
    "Nonlinear-delayed": generate_nonlinear_delayed_lagged,
}

STRESS_DATASETS = {
    **DATASETS,
    "Block-lowrank": generate_block_lowrank_lagged,
}
