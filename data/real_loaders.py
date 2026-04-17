"""
Real data loaders for ICBINB benchmark.

Actual observational time series with literature-derived ground truth:
    1. NOAA monthly climate indices (1948-present, real)
    2. Yahoo Finance sector ETFs (daily returns, real)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


NOAA_URLS = {
    "ENSO": "https://psl.noaa.gov/data/correlation/nina34.data",
    "NAO": "https://psl.noaa.gov/data/correlation/nao.data",
    "PDO": "https://psl.noaa.gov/data/correlation/pdo.data",
    "AMO": "https://psl.noaa.gov/data/correlation/amon.us.data",
    "SOI": "https://psl.noaa.gov/data/correlation/soi.data",
    "PNA": "https://psl.noaa.gov/data/correlation/pna.data",
}

# Ground truth teleconnections from Runge et al. (2019) and climate literature.
# Format: graph[target, source] = 1 means source Granger-causes target.
NOAA_GROUND_TRUTH_EDGES = [
    ("ENSO", "SOI"),      # ENSO and SOI are definitionally coupled (opposite sign)
    ("SOI", "ENSO"),
    ("ENSO", "PNA"),      # ENSO drives PNA pattern
    ("ENSO", "PDO"),      # Known coupling
    ("PDO", "ENSO"),
    ("NAO", "AMO"),       # NAO influences Atlantic SST
]


def _parse_noaa_file(path: str) -> pd.Series:
    """
    NOAA .data format:
        line 1:   start_year  end_year
        line 2+:  year v_jan v_feb ... v_dec
        footer:   missing value (e.g., -99.99) and optional metadata

    Returns a monthly pandas Series indexed by datetime.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # First line is year range
    parts = lines[0].split()
    start_year, end_year = int(parts[0]), int(parts[1])

    # Find missing value marker (usually right after data rows)
    missing = -99.99
    records = []

    for line in lines[1:]:
        tokens = line.split()
        # Skip non-data lines (metadata)
        try:
            year = int(tokens[0])
        except (ValueError, IndexError):
            continue
        if year < start_year or year > end_year:
            continue
        if len(tokens) < 13:
            continue
        values = []
        for t in tokens[1:13]:
            try:
                v = float(t)
                if abs(v - missing) < 0.01 or abs(v + 99.99) < 0.01 or abs(v + 999.9) < 0.01:
                    v = np.nan
                values.append(v)
            except ValueError:
                values.append(np.nan)
        records.append((year, values))

    rows = []
    for year, vals in records:
        for month, v in enumerate(vals, start=1):
            rows.append({"date": pd.Timestamp(year=year, month=month, day=1), "value": v})

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df["value"]


def load_noaa_climate(
    data_dir: str = "data/real",
    start_date: str = "1960-01-01",
    end_date: str = "2024-12-31",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load real NOAA monthly climate indices.

    Returns:
        X: time series (T, K), standardized
        G: binary ground-truth adjacency (K, K), literature-derived
        names: variable names
    """
    files = {
        "ENSO": "nino34.txt",
        "NAO": "nao.txt",
        "PDO": "pdo.txt",
        "AMO": "amo.txt",
        "SOI": "soi.txt",
        "PNA": "pna.txt",
    }

    series_dict = {}
    for name, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing NOAA file: {path}")
        series_dict[name] = _parse_noaa_file(path)

    df = pd.DataFrame(series_dict)
    df = df.loc[start_date:end_date]
    df = df.dropna()  # drop months with any missing

    names = list(df.columns)
    X = df.values.astype(np.float32)

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Build ground-truth adjacency
    K = len(names)
    name_idx = {n: i for i, n in enumerate(names)}
    G = np.zeros((K, K), dtype=np.float32)
    for target, source in NOAA_GROUND_TRUTH_EDGES:
        if target in name_idx and source in name_idx:
            G[name_idx[target], name_idx[source]] = 1.0

    return X, G, names


# ─── Finance ───

SECTOR_GROUND_TRUTH = [
    # Source -> Target, based on sector lead-lag literature
    ("XLE", "XLB"),    # Energy drives Materials input costs
    ("XLE", "XLI"),    # Energy drives Industrial costs
    ("XLF", "XLRE"),   # Financials drive Real Estate (rates)
    ("XLF", "XLU"),    # Financials drive Utilities (rates)
    ("XLK", "XLC"),    # Tech leads Communication Services
    ("XLY", "XLP"),    # Consumer Discretionary -> Staples (spending shifts)
]


def load_sector_etfs(
    data_dir: str = "data/real",
    compute_returns: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load real sector ETF prices from Yahoo Finance download.

    Returns:
        X: log returns time series (T, K), standardized
        G: binary ground-truth adjacency (K, K), finance literature
        names: ETF ticker names
    """
    path = os.path.join(data_dir, "sector_etfs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ETF file: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna()

    names = list(df.columns)
    X_prices = df.values.astype(np.float32)

    if compute_returns:
        X = np.log(X_prices[1:] / X_prices[:-1])
    else:
        X = X_prices

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    K = len(names)
    name_idx = {n: i for i, n in enumerate(names)}
    G = np.zeros((K, K), dtype=np.float32)
    for source, target in SECTOR_GROUND_TRUTH:
        if source in name_idx and target in name_idx:
            G[name_idx[target], name_idx[source]] = 1.0

    return X, G, names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=== Testing real data loaders ===\n")

    X, G, names = load_noaa_climate()
    print(f"NOAA Climate: T={X.shape[0]}, K={X.shape[1]}")
    print(f"  Variables: {names}")
    print(f"  Ground truth edges: {int(G.sum())}")
    print(f"  Date range: {X.shape[0]} months")
    print()

    X, G, names = load_sector_etfs()
    print(f"Sector ETFs: T={X.shape[0]}, K={X.shape[1]}")
    print(f"  Variables: {names}")
    print(f"  Ground truth edges: {int(G.sum())}")
