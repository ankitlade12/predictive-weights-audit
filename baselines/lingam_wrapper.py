"""Optional VAR-LiNGAM baseline wrapper."""

from __future__ import annotations

import numpy as np


def var_lingam_causality_matrix(X: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """
    Run VAR-LiNGAM and return an aggregate lagged causal score matrix.

    Requires the optional lingam package.
    """
    try:
        from lingam import VARLiNGAM
    except ImportError:
        raise ImportError(
            "lingam is required for the VAR-LiNGAM baseline. "
            "Install the optional modern-baseline dependencies before running it."
        )

    model = VARLiNGAM(lags=max_lag)
    model.fit(X)

    adjacency = getattr(model, "adjacency_matrices_", None)
    if adjacency is None:
        raise RuntimeError("VARLiNGAM did not expose adjacency_matrices_.")

    S_lag = np.abs(np.asarray(adjacency, dtype=float))
    if S_lag.ndim == 2:
        S = S_lag
    else:
        S = S_lag.max(axis=0)

    np.fill_diagonal(S, 0.0)
    if S.max() > 0:
        S = S / S.max()
    return S
