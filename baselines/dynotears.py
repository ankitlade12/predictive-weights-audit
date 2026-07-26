"""Optional DYNOTEARS baseline wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_dynamic_node(
    node: object,
    column_indices: dict[str, int],
) -> tuple[int, int] | None:
    """Parse CausalNex ``<column>_lag<integer>`` nodes without raising."""
    column, separator, lag_text = str(node).rpartition("_lag")
    if not separator or column not in column_indices or not lag_text.isdigit():
        return None
    return column_indices[column], int(lag_text)


def dynotears_causality_matrix(X: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """
    Run DYNOTEARS and return an aggregate lagged causal score matrix.

    Only positive-lag source nodes pointing to lag-0 targets are aggregated,
    with output orientation ``S[target, source]``. Non-finite weights are
    ignored.

    CausalNex 0.12.1 requires a separate legacy Python 3.10 environment and is
    not part of the primary reproducible environment.
    """
    try:
        from causalnex.structure.dynotears import from_pandas_dynamic
    except ImportError:
        raise ImportError(
            "DYNOTEARS requires causalnex 0.12.1, which does not support the "
            "project's primary Python 3.12 runtime. Use a separate legacy "
            "Python 3.10 environment with causalnex 0.12.1; DYNOTEARS is not "
            "part of the primary reproducible environment."
        )

    K = X.shape[1]
    columns = [f"x{i}" for i in range(K)]
    column_indices = {column: index for index, column in enumerate(columns)}
    graph = from_pandas_dynamic(pd.DataFrame(X, columns=columns), p=max_lag)
    S = np.zeros((K, K), dtype=float)

    for source, target, data in graph.edges(data=True):
        source_node = _parse_dynamic_node(source, column_indices)
        target_node = _parse_dynamic_node(target, column_indices)
        if source_node is None or target_node is None:
            continue
        source_idx, source_lag = source_node
        target_idx, target_lag = target_node
        if source_lag <= 0 or target_lag != 0 or source_idx == target_idx:
            continue
        weight = abs(float(data.get("weight", 1.0)))
        if not np.isfinite(weight):
            continue
        S[target_idx, source_idx] = max(S[target_idx, source_idx], weight)

    if S.max() > 0:
        S = S / S.max()
    return S
