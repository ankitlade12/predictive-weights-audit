"""Optional VAR-LiNGAM baseline wrapper."""

from __future__ import annotations

import numpy as np


def var_lingam_causality_matrix(
    X: np.ndarray,
    max_lag: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """
    Run VAR-LiNGAM and return an aggregate lagged causal score matrix.

    The contemporaneous ``B0`` matrix is excluded. Lagged ``B1..Blag``
    matrices already use the project orientation ``[target, source]``.

    Requires the optional lingam package.
    """
    try:
        from lingam import VARLiNGAM
    except ImportError:
        raise ImportError(
            "lingam is required for the VAR-LiNGAM baseline. "
            "Install the optional modern-baseline dependencies before running it."
        )

    model = VARLiNGAM(lags=max_lag, random_state=random_state)
    model.fit(X)

    adjacency = getattr(model, "adjacency_matrices_", None)
    if adjacency is None:
        raise RuntimeError("VARLiNGAM did not expose adjacency_matrices_.")

    adjacency = np.asarray(adjacency, dtype=float)
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("VAR-LiNGAM returned non-finite adjacency_matrices_.")
    if adjacency.ndim == 2 or adjacency.shape[0] <= 1:
        S = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    else:
        lagged = adjacency[1:]
        S = np.abs(lagged).max(axis=0)

    np.fill_diagonal(S, 0.0)
    if S.max() > 0:
        S = S / S.max()
    return S
