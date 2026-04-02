"""
Thresholding methods to convert continuous causal score matrices
into binary causal graphs.
"""

import numpy as np
from typing import Tuple, Optional


def threshold_graph(
    S: np.ndarray,
    threshold: float = 0.5,
    remove_diagonal: bool = True,
) -> np.ndarray:
    """
    Apply a fixed threshold to the causal score matrix.

    Args:
        S: Causal score matrix (K, K), values in [0, 1].
        threshold: Edges with S[i,j] > threshold are kept.
        remove_diagonal: If True, zero out diagonal (no self-causation).

    Returns:
        G: Binary adjacency matrix (K, K).
    """
    G = (np.abs(S) > threshold).astype(float)
    if remove_diagonal:
        np.fill_diagonal(G, 0)
    return G


def adaptive_threshold(
    S: np.ndarray,
    method: str = "otsu",
    remove_diagonal: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Automatically determine the best threshold.

    Args:
        S: Causal score matrix (K, K), values in [0, 1].
        method: Thresholding method:
            'otsu' — Otsu's method (maximize inter-class variance).
            'percentile_90' — Top 10% of scores become edges.
            'gap' — Find the largest gap in sorted scores.
            'knee' — Find the knee point in sorted scores.
        remove_diagonal: Zero out diagonal.

    Returns:
        G: Binary adjacency matrix (K, K).
        threshold: The threshold value used.
    """
    scores = np.abs(S).copy()
    if remove_diagonal:
        np.fill_diagonal(scores, 0)

    # Get off-diagonal values for thresholding
    mask = ~np.eye(S.shape[0], dtype=bool)
    values = scores[mask].flatten()

    if method == "otsu":
        threshold = _otsu_threshold(values)
    elif method == "percentile_90":
        threshold = np.percentile(values, 90)
    elif method == "gap":
        threshold = _gap_threshold(values)
    elif method == "knee":
        threshold = _knee_threshold(values)
    else:
        raise ValueError(f"Unknown method: {method}")

    G = (scores > threshold).astype(float)
    if remove_diagonal:
        np.fill_diagonal(G, 0)

    return G, threshold


def best_f1_threshold(
    S: np.ndarray,
    G_true: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, float, float]:
    """
    Find the threshold that maximizes F1 score against ground truth.
    (Used for evaluation, not as a method — requires ground truth.)

    Args:
        S: Causal score matrix (K, K).
        G_true: Ground truth binary adjacency (K, K).
        n_thresholds: Number of thresholds to try.

    Returns:
        G_best: Best binary graph.
        best_threshold: Optimal threshold.
        best_f1: F1 score at optimal threshold.
    """
    mask = ~np.eye(S.shape[0], dtype=bool)
    thresholds = np.linspace(0, np.abs(S[mask]).max() + 1e-6, n_thresholds)

    best_f1 = -1
    best_threshold = 0
    best_G = np.zeros_like(S)

    true_flat = G_true[mask].flatten()

    for t in thresholds:
        G = (np.abs(S) > t).astype(float)
        np.fill_diagonal(G, 0)
        pred_flat = G[mask].flatten()

        tp = np.sum((pred_flat == 1) & (true_flat == 1))
        fp = np.sum((pred_flat == 1) & (true_flat == 0))
        fn = np.sum((pred_flat == 0) & (true_flat == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_G = G.copy()

    return best_G, best_threshold, best_f1


# ─── Internal threshold methods ────────────────────────────────────────────


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method: maximize inter-class variance."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    if n == 0:
        return 0.0

    best_thresh = sorted_vals[0]
    best_variance = 0

    for i in range(1, n):
        w0 = i / n
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue

        mean0 = sorted_vals[:i].mean()
        mean1 = sorted_vals[i:].mean()

        variance = w0 * w1 * (mean0 - mean1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = sorted_vals[i]

    return best_thresh


def _gap_threshold(values: np.ndarray) -> float:
    """Find the largest gap in sorted values."""
    sorted_vals = np.sort(values)
    if len(sorted_vals) < 2:
        return 0.0

    diffs = np.diff(sorted_vals)
    gap_idx = np.argmax(diffs)
    return (sorted_vals[gap_idx] + sorted_vals[gap_idx + 1]) / 2


def _knee_threshold(values: np.ndarray) -> float:
    """Find the knee/elbow point in the sorted score curve."""
    sorted_vals = np.sort(values)[::-1]  # descending
    n = len(sorted_vals)
    if n < 3:
        return sorted_vals[0] if n > 0 else 0.0

    # Line from first to last point
    x = np.arange(n)
    p1 = np.array([0, sorted_vals[0]])
    p2 = np.array([n - 1, sorted_vals[-1]])

    # Distance from each point to the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return sorted_vals[0]

    distances = []
    for i in range(n):
        point = np.array([i, sorted_vals[i]])
        d = np.abs(np.cross(line_vec, p1 - point)) / line_len
        distances.append(d)

    knee_idx = np.argmax(distances)
    return sorted_vals[knee_idx]
