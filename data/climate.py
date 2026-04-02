"""
Climate teleconnection data loader.

Downloads and preprocesses NOAA/ERA5 climate reanalysis data for
causal discovery benchmarks.

Key climate indices (teleconnection nodes):
    - ENSO (El Nino Southern Oscillation): Nino 3.4 SST index
    - NAO (North Atlantic Oscillation)
    - PDO (Pacific Decadal Oscillation)
    - AMO (Atlantic Multidecadal Oscillation)
    - IOD (Indian Ocean Dipole)
    - PNA (Pacific North American Pattern)
    - SOI (Southern Oscillation Index)
    - SAM (Southern Annular Mode)

Ground truth: Well-established teleconnection relationships from
decades of climate science literature.

Data sources:
    - NOAA PSL: https://psl.noaa.gov/data/climateindices/list/
    - NOAA ESRL: https://www.esrl.noaa.gov/psd/data/
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional


# Known causal relationships from climate literature
# Format: (source_index, target_index, lag_months, reference)
KNOWN_TELECONNECTIONS = {
    "ENSO -> IOD": {"source": "ENSO", "target": "IOD", "lag": 1,
                    "ref": "Saji et al. (1999)"},
    "ENSO -> PNA": {"source": "ENSO", "target": "PNA", "lag": 0,
                    "ref": "Wallace & Gutzler (1981)"},
    "ENSO -> SOI": {"source": "ENSO", "target": "SOI", "lag": 0,
                    "ref": "By definition (anticorrelated)"},
    "NAO -> AMO": {"source": "NAO", "target": "AMO", "lag": 12,
                   "ref": "Li et al. (2013)"},
    "PDO -> ENSO": {"source": "PDO", "target": "ENSO", "lag": 6,
                    "ref": "Newman et al. (2016)"},
}


def generate_synthetic_climate(
    n_indices: int = 8,
    T: int = 720,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate synthetic climate-like data with known teleconnection structure.

    Mimics the statistical properties of real climate indices
    (autocorrelation, seasonal cycles, teleconnection lags) while
    providing exact ground truth for evaluation.

    Args:
        n_indices: Number of climate index variables.
        T: Number of monthly timesteps (720 = 60 years).
        seed: Random seed.

    Returns:
        X: Time series (T, n_indices).
        G_true: Ground truth adjacency matrix.
        index_names: List of variable names.
    """
    rng = np.random.RandomState(seed)

    index_names = ["ENSO", "NAO", "PDO", "AMO", "IOD", "PNA", "SOI", "SAM"][:n_indices]
    K = len(index_names)

    # Build causal coefficient matrix with known structure
    # Each variable has autoregressive component + causal links
    A = np.diag([0.7] * K)  # Strong autocorrelation (climate persistence)

    # Known causal links (month-scale lags approximated as lag-1)
    name_to_idx = {name: i for i, name in enumerate(index_names)}

    causal_edges = [
        ("ENSO", "IOD", 0.3),   # ENSO drives Indian Ocean Dipole
        ("ENSO", "PNA", 0.35),  # ENSO drives Pacific-North American pattern
        ("ENSO", "SOI", -0.4),  # ENSO and SOI are anticorrelated
        ("PDO", "ENSO", 0.2),   # PDO modulates ENSO
        ("NAO", "AMO", 0.15),   # NAO influences Atlantic temps (slow)
    ]

    G_true = np.zeros((K, K))
    for src, tgt, strength in causal_edges:
        if src in name_to_idx and tgt in name_to_idx:
            i, j = name_to_idx[tgt], name_to_idx[src]
            A[i, j] = strength
            G_true[i, j] = 1.0

    # Ensure stability
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        A = A * (0.94 / max_eig)

    # Generate with seasonal component
    X = np.zeros((T, K))
    for t in range(1, T):
        # Autoregressive + causal
        X[t] = A @ X[t - 1] + rng.randn(K) * 0.15
        # Add seasonal cycle (12-month period)
        month = t % 12
        X[t, 0] += 0.3 * np.sin(2 * np.pi * month / 12)  # ENSO seasonal

    return X, G_true, index_names


def load_noaa_indices(
    data_dir: str = "data/climate_raw",
    start_year: int = 1950,
    end_year: int = 2023,
) -> Tuple[Optional[np.ndarray], list]:
    """
    Load real NOAA climate index data from CSV files.

    Expected CSV format per index: columns [year, month, value].
    Download from: https://psl.noaa.gov/data/climateindices/list/

    Args:
        data_dir: Directory containing CSV files named by index (e.g., nino34.csv).
        start_year: Start year for data range.
        end_year: End year for data range.

    Returns:
        X: Time series (T, K) or None if files not found.
        index_names: List of index names loaded.
    """
    index_files = {
        "ENSO": "nino34.csv",
        "NAO": "nao.csv",
        "PDO": "pdo.csv",
        "AMO": "amo.csv",
        "SOI": "soi.csv",
        "PNA": "pna.csv",
    }

    loaded = {}
    for name, filename in index_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Standardize column names
                df.columns = [c.strip().lower() for c in df.columns]
                if "value" in df.columns:
                    loaded[name] = df["value"].values
                elif len(df.columns) >= 3:
                    loaded[name] = df.iloc[:, 2].values
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

    if not loaded:
        print(f"No climate data found in {data_dir}. Using synthetic data.")
        return None, []

    # Align all series to same length
    min_len = min(len(v) for v in loaded.values())
    index_names = list(loaded.keys())
    X = np.column_stack([loaded[name][:min_len] for name in index_names])

    # Standardize (zero mean, unit variance)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, index_names


def get_climate_ground_truth(index_names: list) -> np.ndarray:
    """
    Build ground truth adjacency matrix from known teleconnections.

    Args:
        index_names: List of climate index names.

    Returns:
        G_true: Binary adjacency matrix (K, K).
    """
    K = len(index_names)
    G_true = np.zeros((K, K))

    name_to_idx = {name: i for i, name in enumerate(index_names)}

    for link_name, link_info in KNOWN_TELECONNECTIONS.items():
        src = link_info["source"]
        tgt = link_info["target"]
        if src in name_to_idx and tgt in name_to_idx:
            G_true[name_to_idx[tgt], name_to_idx[src]] = 1.0

    return G_true
