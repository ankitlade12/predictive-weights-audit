"""
Synthetic data generators for causal discovery benchmarks.

Each generator returns:
    X: np.ndarray of shape (T, K) — the multivariate time series
    A_true: np.ndarray or dict — the ground truth causal structure
    graph: np.ndarray of shape (K, K) — binary adjacency matrix (j->i means graph[i,j]=1)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def generate_var1(
    K: int = 3,
    T: int = 2000,
    noise_std: float = 0.1,
    graph_type: str = "chain",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a VAR(1) process: X[t] = A @ X[t-1] + noise.

    Args:
        K: Number of variables.
        T: Number of timesteps.
        noise_std: Standard deviation of Gaussian noise.
        graph_type: "chain" (X0->X1->...->XK) or "fork" (X0->all others)
                    or "random" (random sparse graph).
        seed: Random seed for reproducibility.

    Returns:
        X: Time series of shape (T, K).
        A_true: VAR coefficient matrix of shape (K, K).
        graph: Binary adjacency matrix (K, K), graph[i,j]=1 means j->i.
    """
    rng = np.random.RandomState(seed)

    if graph_type == "chain":
        A_true = np.diag([0.5] * K)
        for i in range(1, K):
            A_true[i, i - 1] = 0.4  # X_{i-1} -> X_i
    elif graph_type == "fork":
        A_true = np.diag([0.5] * K)
        for i in range(1, K):
            A_true[i, 0] = 0.3 + rng.uniform(-0.1, 0.1)  # X_0 -> X_i
    elif graph_type == "random":
        A_true = _random_sparse_var_matrix(K, density=0.3, rng=rng)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Ensure stability: spectral radius < 1
    A_true = _ensure_stable(A_true)

    X = np.zeros((T, K))
    for t in range(1, T):
        X[t] = A_true @ X[t - 1] + rng.randn(K) * noise_std

    # Ground truth adjacency: off-diagonal non-zero entries
    graph = (np.abs(A_true) > 1e-10).astype(float)
    np.fill_diagonal(graph, 0)  # Self-loops are not causal edges

    return X, A_true, graph


def generate_var2(
    K: int = 5,
    T: int = 3000,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Generate a VAR(2) process: X[t] = A1 @ X[t-1] + A2 @ X[t-2] + noise.

    Returns:
        X: Time series (T, K).
        A_dict: {"A1": ..., "A2": ...} coefficient matrices.
        graph: Binary adjacency (union of A1 and A2 off-diagonal non-zeros).
    """
    rng = np.random.RandomState(seed)

    A1 = _random_sparse_var_matrix(K, density=0.3, rng=rng) * 0.6
    A2 = _random_sparse_var_matrix(K, density=0.2, rng=rng) * 0.3

    # Ensure combined stability (approximate check via companion form)
    A1 = _ensure_stable(A1, target_radius=0.7)
    A2 = _ensure_stable(A2, target_radius=0.3)

    X = np.zeros((T, K))
    for t in range(2, T):
        X[t] = A1 @ X[t - 1] + A2 @ X[t - 2] + rng.randn(K) * noise_std

    graph1 = (np.abs(A1) > 1e-10).astype(float)
    graph2 = (np.abs(A2) > 1e-10).astype(float)
    graph = np.clip(graph1 + graph2, 0, 1)
    np.fill_diagonal(graph, 0)

    return X, {"A1": A1, "A2": A2}, graph


def generate_lorenz(
    T: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
    seed: int = 42,
    downsample: int = 1,
) -> Tuple[np.ndarray, None, np.ndarray]:
    """
    Generate the Lorenz attractor system.

    Causal structure:
        - X <-> Y (bidirectional coupling via sigma*(Y-X) and X*(rho-Z)-Y)
        - X -> Z (via X*Y term in dZ/dt)
        - Y -> Z (via X*Y term in dZ/dt)

    The ground truth graph captures the dominant causal directions:
        X -> Y, Y -> X, X -> Z, Y -> Z

    Returns:
        X: Time series (T//downsample, 3).
        None: No coefficient matrix for nonlinear systems.
        graph: Binary adjacency (3, 3).
    """
    rng = np.random.RandomState(seed)

    raw = np.zeros((T, 3))
    raw[0] = [1.0, 1.0, 1.0] + rng.randn(3) * 0.01

    for t in range(1, T):
        x, y, z = raw[t - 1]
        raw[t, 0] = x + dt * sigma * (y - x)
        raw[t, 1] = y + dt * (x * (rho - z) - y)
        raw[t, 2] = z + dt * (x * y - beta * z)

    X = raw[::downsample]

    # Ground truth: X->Y, Y->X, X->Z, Y->Z
    # (Z does feed back into Y via x*(rho-Z), so Z->Y is also present)
    graph = np.array([
        [0, 1, 0],  # X is caused by Y (sigma*(Y-X))
        [1, 0, 1],  # Y is caused by X and Z (x*(rho-z)-y)
        [1, 1, 0],  # Z is caused by X and Y (x*y)
    ], dtype=float)

    return X, None, graph


def generate_henon(
    T: int = 3000,
    a: float = 1.4,
    b: float = 0.3,
    coupling: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, None, np.ndarray]:
    """
    Generate coupled Hénon maps: a nonlinear discrete-time system.

    System:
        X1[t] = 1 - a*X1[t-1]^2 + b*X1[t-2]
        X2[t] = 1 - a*X2[t-1]^2 + b*X2[t-2] + coupling*X1[t-1]^2

    Causal structure: X1 -> X2 (unidirectional).
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((T, 2))
    X[0] = rng.randn(2) * 0.1
    X[1] = rng.randn(2) * 0.1

    for t in range(2, T):
        X[t, 0] = 1 - a * X[t - 1, 0] ** 2 + b * X[t - 2, 0]
        X[t, 1] = (
            1 - a * X[t - 1, 1] ** 2 + b * X[t - 2, 1]
            + coupling * X[t - 1, 0] ** 2
        )

    # Clip to avoid divergence
    X = np.clip(X, -10, 10)

    graph = np.array([
        [0, 0],  # X1 not caused by X2
        [1, 0],  # X2 caused by X1
    ], dtype=float)

    return X, None, graph


def generate_regime_switching(
    K: int = 3,
    T: int = 4000,
    noise_std: float = 0.1,
    n_regimes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Generate a regime-switching VAR(1) process.

    Regime 1 (t < T//2): Chain X0 -> X1 -> X2
    Regime 2 (t >= T//2): Reversed chain X2 -> X1 -> X0

    This is the key benchmark where SSMs should outperform PCMCI,
    since PCMCI assumes stationarity.

    Returns:
        X: Time series (T, K).
        A_dict: {"A1": regime-1 matrix, "A2": regime-2 matrix}.
        graph: Union of both regimes' adjacency matrices.
    """
    rng = np.random.RandomState(seed)

    # Regime 1: forward chain
    A1 = np.diag([0.5] * K)
    for i in range(1, K):
        A1[i, i - 1] = 0.4

    # Regime 2: backward chain
    A2 = np.diag([0.5] * K)
    for i in range(K - 1):
        A2[i, i + 1] = 0.4

    X = np.zeros((T, K))
    switch_point = T // n_regimes

    for t in range(1, T):
        regime_idx = min(t // switch_point, n_regimes - 1)
        A = A1 if regime_idx % 2 == 0 else A2
        X[t] = A @ X[t - 1] + rng.randn(K) * noise_std

    # Union graph for evaluation
    g1 = (np.abs(A1) > 1e-10).astype(float)
    g2 = (np.abs(A2) > 1e-10).astype(float)
    graph = np.clip(g1 + g2, 0, 1)
    np.fill_diagonal(graph, 0)

    return X, {"A1": A1, "A2": A2}, graph


def generate_random_sparse_var(
    K: int = 10,
    T: int = 2000,
    density: float = 0.2,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a VAR(1) with a random sparse causal graph.

    Args:
        K: Number of variables.
        T: Number of timesteps.
        density: Fraction of off-diagonal entries that are non-zero.
        noise_std: Noise standard deviation.
        seed: Random seed.
    """
    rng = np.random.RandomState(seed)
    A_true = _random_sparse_var_matrix(K, density=density, rng=rng)
    A_true = _ensure_stable(A_true)

    X = np.zeros((T, K))
    for t in range(1, T):
        X[t] = A_true @ X[t - 1] + rng.randn(K) * noise_std

    graph = (np.abs(A_true) > 1e-10).astype(float)
    np.fill_diagonal(graph, 0)

    return X, A_true, graph


# ─── Internal helpers ───────────────────────────────────────────────────────


def _random_sparse_var_matrix(
    K: int, density: float = 0.3, rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Create a random sparse VAR coefficient matrix."""
    if rng is None:
        rng = np.random.RandomState()

    A = np.diag(rng.uniform(0.3, 0.6, size=K))

    # Add off-diagonal entries with given density
    n_off_diag = K * (K - 1)
    n_edges = max(1, int(density * n_off_diag))

    off_diag_indices = [(i, j) for i in range(K) for j in range(K) if i != j]
    chosen = rng.choice(len(off_diag_indices), size=n_edges, replace=False)

    for idx in chosen:
        i, j = off_diag_indices[idx]
        A[i, j] = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])

    return A


def _ensure_stable(A: np.ndarray, target_radius: float = 0.95) -> np.ndarray:
    """Scale matrix so that its spectral radius is below target_radius."""
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    if spectral_radius > target_radius:
        A = A * (target_radius / spectral_radius)
    return A
