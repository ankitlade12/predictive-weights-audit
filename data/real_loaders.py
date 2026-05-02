"""
Data loaders for the ICBINB benchmark.

    1. NOAA monthly climate indices (real, 1960-2024) — literature-derived
       ground truth with ENSO↔SOI excluded as definitionally coupled.
    2. Yahoo Finance sector ETFs (real, 2018-2025) — finance-literature
       lead-lag edges; weak ground truth, used as a null-check.
    3. Lorenz-96 (synthetic, but with unambiguous ground truth) — standard
       causal-discovery benchmark (Tank et al. 2022, Nauta et al. 2019).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ssm_causal.benchmark_metadata import (
    EdgeProvenance,
    build_adjacency_from_edges,
    edge_provenance_cards,
)

NOAA_URLS = {
    "ENSO": "https://psl.noaa.gov/data/correlation/nina34.data",
    "NAO": "https://psl.noaa.gov/data/correlation/nao.data",
    "PDO": "https://psl.noaa.gov/data/correlation/pdo.data",
    "AMO": "https://psl.noaa.gov/data/correlation/amon.us.data",
    "SOI": "https://psl.noaa.gov/data/correlation/soi.data",
    "PNA": "https://psl.noaa.gov/data/correlation/pna.data",
}

# Ground truth teleconnections from the climate dynamics literature.
# Format: (target, source) — meaning source Granger-causes target.
#
# We deliberately exclude ENSO↔SOI because those two indices are
# definitionally the same signal with opposite sign (Southern Oscillation
# Index is a surface-pressure analog of the Niño 3.4 SST index). Including
# them inflates every method's AUROC on what is really a near-instantaneous
# coupling, not a causal edge.
NOAA_EDGE_PROVENANCE = [
    EdgeProvenance(
        target="PNA",
        source="ENSO",
        label="causal",
        rationale="ENSO teleconnection drives PNA variability.",
        reference="Trenberth & Hurrell 1994",
        group="teleconnection",
    ),
    EdgeProvenance(
        target="PDO",
        source="ENSO",
        label="causal",
        rationale="ENSO modulates PDO phase transitions.",
        reference="Newman et al. 2016",
        group="teleconnection",
    ),
    EdgeProvenance(
        target="AMO",
        source="NAO",
        label="causal",
        rationale="NAO drives Atlantic SST patterns.",
        reference="Delworth & Mann 2000",
        group="teleconnection",
    ),
    EdgeProvenance(
        target="SOI",
        source="ENSO",
        label="definitional",
        rationale="SOI and Nino 3.4 index the same ENSO signal with opposite sign.",
        group="enso_soi",
    ),
    EdgeProvenance(
        target="ENSO",
        source="SOI",
        label="definitional",
        rationale="SOI and Nino 3.4 index the same ENSO signal with opposite sign.",
        group="enso_soi",
    ),
]

NOAA_GROUND_TRUTH_EDGES = [
    (edge.target, edge.source)
    for edge in NOAA_EDGE_PROVENANCE
    if edge.label == "causal"
]


def load_noaa_edge_provenance(include_definitional: bool = False) -> list[EdgeProvenance]:
    """Return NOAA edge provenance records used for benchmark audits."""
    if include_definitional:
        return list(NOAA_EDGE_PROVENANCE)
    return [edge for edge in NOAA_EDGE_PROVENANCE if edge.label == "causal"]


def load_noaa_edge_cards(include_definitional: bool = False) -> dict[str, list[dict[str, str]]]:
    """Return NOAA edge provenance cards grouped by provenance label."""
    return edge_provenance_cards(load_noaa_edge_provenance(include_definitional))


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
        "ENSO": "nina34.data",
        "NAO": "nao.data",
        "PDO": "pdo.data",
        "AMO": "amon.us.data",
        "SOI": "soi.data",
        "PNA": "pna.data",
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

    G = build_adjacency_from_edges(
        names,
        load_noaa_edge_provenance(include_definitional=False),
        include_labels={"causal"},
    )

    return X, G, names


# ─── Finance ───

SECTOR_EDGE_PROVENANCE = [
    EdgeProvenance(
        "XLB", "XLE", "soft", "Energy affects materials input costs.", group="sector_lead_lag"
    ),
    EdgeProvenance(
        "XLI", "XLE", "soft", "Energy affects industrial input costs.", group="sector_lead_lag"
    ),
    EdgeProvenance(
        "XLRE",
        "XLF",
        "soft",
        "Financials proxy rate conditions for real estate.",
        group="sector_lead_lag",
    ),
    EdgeProvenance(
        "XLU",
        "XLF",
        "soft",
        "Financials proxy rate conditions for utilities.",
        group="sector_lead_lag",
    ),
    EdgeProvenance(
        "XLC", "XLK", "soft", "Technology can lead communication services.", group="sector_lead_lag"
    ),
    EdgeProvenance(
        "XLP",
        "XLY",
        "soft",
        "Discretionary spending shifts can lead staples.",
        group="sector_lead_lag",
    ),
]

SECTOR_GROUND_TRUTH = [
    (edge.source, edge.target)
    for edge in SECTOR_EDGE_PROVENANCE
]


def load_sector_edge_cards() -> dict[str, list[dict[str, str]]]:
    """Return sector ETF edge provenance cards grouped by provenance label."""
    return edge_provenance_cards(SECTOR_EDGE_PROVENANCE)


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

    G = build_adjacency_from_edges(names, SECTOR_EDGE_PROVENANCE, include_labels={"soft"})

    return X, G, names


# ─── Lorenz-96 (standard causal-discovery benchmark) ───

def load_lorenz96(
    K: int = 10,
    T: int = 1500,
    F: float = 10.0,
    dt: float = 0.01,
    subsample: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Discrete-time Lorenz-96 dataset with unambiguous Granger ground truth.

    System:  dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F   (cyclic i)

    Following Tank et al. (2022, Neural Granger Causality) and Nauta et al.
    (2019, TCDF), ground truth edges are the four variables appearing on the
    RHS of the equation for each x_i: itself, and x_{i-2}, x_{i-1}, x_{i+1}.
    We exclude the self-loop when scoring (diagonal is zeroed by convention).

    Args:
        K: number of oscillators
        T: number of kept timesteps after subsampling
        F: forcing term (8 = chaotic; 10 = strongly chaotic, Tank et al.'s setting)
        dt: integration step
        subsample: keep every nth integration step
        seed: RNG seed for initialization

    Returns:
        X: (T, K) standardized time series
        G: (K, K) binary adjacency with G[i, j]=1 iff j ∈ {i-2, i-1, i+1}
        names: generic ["x0", "x1", ...]
    """
    rng = np.random.default_rng(seed)
    x = F * np.ones(K) + rng.standard_normal(K) * 0.01

    def deriv(xp: np.ndarray) -> np.ndarray:
        return (np.roll(xp, -1) - np.roll(xp, 2)) * np.roll(xp, 1) - xp + F

    n_steps = T * subsample + 500
    traj = np.zeros((n_steps, K), dtype=np.float64)
    traj[0] = x
    for t in range(1, n_steps):
        xp = traj[t - 1]
        k1 = deriv(xp)
        k2 = deriv(xp + 0.5 * dt * k1)
        k3 = deriv(xp + 0.5 * dt * k2)
        k4 = deriv(xp + dt * k3)
        traj[t] = xp + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    traj = traj[500::subsample][:T]
    X = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-8)

    G = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for offset in (-2, -1, 1):
            j = (i + offset) % K
            G[i, j] = 1.0

    names = [f"x{i}" for i in range(K)]
    return X.astype(np.float32), G, names


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
    print()

    X, G, names = load_lorenz96()
    print(f"Lorenz-96: T={X.shape[0]}, K={X.shape[1]}")
    print(f"  Ground truth edges: {int(G.sum())} (exactly 3*K expected)")
