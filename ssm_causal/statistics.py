"""Statistical helpers for replicate-level experiment comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_MAX_BOOTSTRAP_INDEX_VALUES = 1_000_000


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Mean paired difference and its percentile bootstrap interval."""

    effect: float
    ci_low: float
    ci_high: float


def holm_correction(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm-adjusted p-values and rejection decisions in input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("p_values must contain only finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in (0, 1]")

    order = np.argsort(values)
    sorted_values = values[order]
    multipliers = values.size - np.arange(values.size)
    adjusted_sorted = np.maximum.accumulate(sorted_values * multipliers)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha


def paired_bootstrap_ci(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Bootstrap the replicate-level mean of paired ``first - second`` effects."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if (
        first_values.ndim != 1
        or second_values.ndim != 1
        or first_values.size == 0
        or first_values.shape != second_values.shape
    ):
        raise ValueError("paired samples must be nonempty one-dimensional arrays of equal size")
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        raise ValueError("paired samples must contain only finite values")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and lie in (0, 1)")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")

    differences = first_values - second_values
    rng = np.random.default_rng(seed)
    bootstrap_effects = np.empty(int(n_resamples), dtype=float)
    rows_per_chunk = max(
        1,
        _MAX_BOOTSTRAP_INDEX_VALUES // differences.size,
    )
    for start in range(0, int(n_resamples), rows_per_chunk):
        stop = min(start + rows_per_chunk, int(n_resamples))
        indices = rng.integers(
            0,
            differences.size,
            size=(stop - start, differences.size),
        )
        bootstrap_effects[start:stop] = differences[indices].mean(axis=1)
    tail_probability = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(
        bootstrap_effects,
        [tail_probability, 1.0 - tail_probability],
    )
    return PairedBootstrapResult(
        effect=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )
