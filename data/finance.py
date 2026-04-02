"""
Financial time series data for causal discovery.

Tests causal relationships between stock sector indices.
Ground truth: well-known sector lead-lag relationships from
financial economics literature.

Key relationships:
    - Energy → Materials (oil prices drive input costs)
    - Tech → Communication Services (tech leads telecom)
    - Financials → Real Estate (interest rates drive housing)
    - Consumer Discretionary → Consumer Staples (spending shifts)
    - Utilities ← Financials (rate sensitivity)
"""

import numpy as np
from typing import Tuple, List


# Known financial sector causal relationships (from literature)
SECTOR_RELATIONSHIPS = [
    ("Energy", "Materials", 0.35),        # Oil → input costs
    ("Tech", "CommServices", 0.30),       # Tech leads telecom
    ("Financials", "RealEstate", 0.40),   # Rates → housing
    ("ConsumerDisc", "ConsumerStap", 0.25),  # Spending patterns
    ("Financials", "Utilities", 0.30),    # Rate sensitivity
    ("Energy", "Industrials", 0.20),      # Energy → manufacturing
]


def generate_synthetic_finance(
    n_sectors: int = 10,
    T: int = 2500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic financial sector return data with known causal structure.

    Mimics properties of real stock returns:
    - Low autocorrelation (efficient markets)
    - Cross-sector lead-lag relationships
    - Volatility clustering (GARCH-like)
    - Fat tails

    Args:
        n_sectors: Number of sector indices (max 10).
        T: Number of daily timesteps (2500 ≈ 10 years).
        seed: Random seed.

    Returns:
        X: Returns time series (T, n_sectors).
        G_true: Ground truth adjacency matrix.
        sector_names: List of sector names.
    """
    rng = np.random.RandomState(seed)

    all_sectors = [
        "Energy", "Materials", "Industrials", "ConsumerDisc",
        "ConsumerStap", "Financials", "Tech", "CommServices",
        "Utilities", "RealEstate",
    ]
    sector_names = all_sectors[:n_sectors]
    K = len(sector_names)

    name_to_idx = {name: i for i, name in enumerate(sector_names)}

    # Build VAR coefficient matrix
    # Low diagonal (weak autocorrelation — efficient markets)
    A = np.diag([0.15] * K)

    G_true = np.zeros((K, K))
    for src, tgt, strength in SECTOR_RELATIONSHIPS:
        if src in name_to_idx and tgt in name_to_idx:
            i, j = name_to_idx[tgt], name_to_idx[src]
            A[i, j] = strength
            G_true[i, j] = 1.0

    # Ensure stability
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        A = A * (0.94 / max_eig)

    # Generate with volatility clustering (simplified GARCH)
    X = np.zeros((T, K))
    vol = np.ones(K) * 0.01  # Initial volatility

    for t in range(1, T):
        # Update volatility (GARCH-like)
        vol = 0.9 * vol + 0.1 * np.abs(X[t - 1]) + 0.005

        # VAR + heteroskedastic noise
        noise = rng.randn(K) * vol
        X[t] = A @ X[t - 1] + noise

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, G_true, sector_names


def generate_synthetic_eeg(
    n_channels: int = 8,
    T: int = 5000,
    fs: float = 256.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic EEG-like data with known cortical connectivity.

    Mimics properties of real EEG:
    - Oscillatory components (alpha, beta bands)
    - Volume conduction effects (spatial correlation)
    - Known directed connectivity between cortical areas

    Ground truth from known cortical connectivity:
        - Frontal → Parietal (top-down attention)
        - Occipital → Temporal (visual processing stream)
        - Motor → Sensory (efference copy)

    Args:
        n_channels: Number of EEG channels (max 8).
        T: Number of timepoints.
        fs: Sampling frequency in Hz.
        seed: Random seed.

    Returns:
        X: EEG time series (T, n_channels).
        G_true: Ground truth adjacency matrix.
        channel_names: List of channel names.
    """
    rng = np.random.RandomState(seed)

    all_channels = ["Fz", "Cz", "Pz", "Oz", "F3", "F4", "C3", "C4"]
    channel_names = all_channels[:n_channels]
    K = len(channel_names)

    name_to_idx = {name: i for i, name in enumerate(channel_names)}

    # Causal connections
    connections = [
        ("Fz", "Pz", 0.35),   # Frontal → Parietal (attention)
        ("Fz", "Cz", 0.30),   # Frontal → Central (motor planning)
        ("Oz", "Pz", 0.25),   # Occipital → Parietal (visual)
        ("Cz", "C3", 0.20),   # Central → Left motor
        ("Cz", "C4", 0.20),   # Central → Right motor
        ("F3", "C3", 0.30),   # Left frontal → Left motor
    ]

    A = np.diag([0.6] * K)  # Strong autocorrelation (oscillatory)
    G_true = np.zeros((K, K))

    for src, tgt, strength in connections:
        if src in name_to_idx and tgt in name_to_idx:
            i, j = name_to_idx[tgt], name_to_idx[src]
            A[i, j] = strength
            G_true[i, j] = 1.0

    # Ensure stability
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        A = A * (0.94 / max_eig)

    # Generate with oscillatory components
    X = np.zeros((T, K))
    t_vec = np.arange(T) / fs

    for t in range(1, T):
        X[t] = A @ X[t - 1] + rng.randn(K) * 0.2

    # Add alpha oscillation (10 Hz) to occipital/parietal
    for ch in ["Oz", "Pz"]:
        if ch in name_to_idx:
            idx = name_to_idx[ch]
            X[:, idx] += 0.5 * np.sin(2 * np.pi * 10 * t_vec)

    # Add beta oscillation (20 Hz) to motor areas
    for ch in ["Cz", "C3", "C4"]:
        if ch in name_to_idx:
            idx = name_to_idx[ch]
            X[:, idx] += 0.3 * np.sin(2 * np.pi * 20 * t_vec)

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, G_true, channel_names
