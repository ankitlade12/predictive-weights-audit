"""Optional DYNOTEARS baseline wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dynotears_causality_matrix(X: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """
    Run DYNOTEARS and return an aggregate lagged causal score matrix.

    Requires causalnex. The wrapper is optional so the core benchmark remains
    usable without the heavier modern-baseline dependency set.
    """
    try:
        from causalnex.structure.dynotears import from_pandas_dynamic
    except ImportError:
        raise ImportError(
            "causalnex is required for the DYNOTEARS baseline. "
            "Install the optional modern-baseline dependencies before running it."
        )

    K = X.shape[1]
    columns = [f"x{i}" for i in range(K)]
    graph = from_pandas_dynamic(pd.DataFrame(X, columns=columns), p=max_lag)
    S = np.zeros((K, K), dtype=float)

    for source, target, data in graph.edges(data=True):
        source_name = str(source).split("_lag")[0]
        target_name = str(target).split("_lag")[0]
        if source_name not in columns or target_name not in columns:
            continue
        source_idx = columns.index(source_name)
        target_idx = columns.index(target_name)
        if source_idx == target_idx:
            continue
        weight = abs(float(data.get("weight", 1.0)))
        S[target_idx, source_idx] = max(S[target_idx, source_idx], weight)

    if S.max() > 0:
        S = S / S.max()
    return S
