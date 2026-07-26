"""Segment-safe data generation for matched intervention experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.stats import wilcoxon

from .lag_synthetic import simulate_lagged_var
from .statistics import holm_correction, paired_bootstrap_ci

InterventionType = Literal["random", "constant", "soft"]


def require_finite_array(
    values: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Return an array only when every raw score value is finite."""
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains a non-finite score tensor.")
    return array


def require_finite_metric(
    metrics: dict[str, float],
    metric_name: str,
    *,
    context: str,
) -> float:
    """Return one evaluator metric only when it exists and is finite."""
    if metric_name not in metrics:
        raise ValueError(f"{context} did not return the {metric_name} metric.")
    metric = float(metrics[metric_name])
    if not np.isfinite(metric):
        raise ValueError(f"{context} returned a non-finite {metric_name} metric.")
    return metric


def paired_effect_inference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    bootstrap_seed: int,
    alternative: Literal["two-sided", "greater", "less"] = "greater",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
) -> dict[str, object]:
    """Summarize one paired contrast without fabricating invalid p-values."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if (
        first_values.ndim != 1
        or first_values.size == 0
        or first_values.shape != second_values.shape
        or not np.all(np.isfinite(first_values))
        or not np.all(np.isfinite(second_values))
    ):
        raise ValueError("Paired inference requires finite, equal-length vectors.")

    bootstrap = paired_bootstrap_ci(
        first_values,
        second_values,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
    )
    differences = first_values - second_values
    raw_p = None
    status = "ok"
    if np.all(differences == 0.0):
        status = "degenerate_all_zero"
    else:
        try:
            result = wilcoxon(
                first_values,
                second_values,
                alternative=alternative,
            )
            if np.isfinite(result.pvalue):
                raw_p = float(result.pvalue)
            else:
                status = "non_estimable"
        except (ValueError, FloatingPointError):
            status = "non_estimable"

    return {
        "status": status,
        "raw_p": raw_p,
        "paired_mean_effect": float(bootstrap.effect),
        "bootstrap_ci": {
            "confidence": float(confidence),
            "low": float(bootstrap.ci_low),
            "high": float(bootstrap.ci_high),
            "n_resamples": int(n_resamples),
            "seed": int(bootstrap_seed),
        },
        "holm_adjusted_p": None,
        "reject_holm": None,
    }


def apply_holm_family(
    inferences: Sequence[dict[str, object]],
    *,
    family_name: str,
    alpha: float = 0.05,
) -> None:
    """Apply Holm over the full pre-specified family."""
    if inferences:
        adjustment_p_values = [
            (1.0 if inference["raw_p"] is None else float(inference["raw_p"]))
            for inference in inferences
        ]
        adjusted, rejected = holm_correction(
            adjustment_p_values,
            alpha=alpha,
        )
        for inference, adjusted_p, reject in zip(
            inferences,
            adjusted,
            rejected,
        ):
            if inference["raw_p"] is None:
                inference["holm_adjusted_p"] = None
                inference["reject_holm"] = False
            else:
                inference["holm_adjusted_p"] = float(adjusted_p)
                inference["reject_holm"] = bool(reject)
    for inference in inferences:
        inference["primary_family"] = family_name
        inference["holm_alpha"] = float(alpha)


def _validated_lengths(
    segment_lengths: Sequence[int],
    total_length: int,
) -> tuple[int, ...]:
    lengths_array = np.asarray(segment_lengths)
    if lengths_array.ndim != 1 or lengths_array.size == 0:
        raise ValueError("segment_lengths must be a non-empty one-dimensional sequence.")
    if (
        np.issubdtype(lengths_array.dtype, np.bool_)
        or not np.issubdtype(lengths_array.dtype, np.number)
        or np.issubdtype(lengths_array.dtype, np.complexfloating)
        or not np.all(np.isfinite(lengths_array))
        or not np.all(lengths_array == np.floor(lengths_array))
    ):
        raise ValueError("segment_lengths must contain integer-like values.")

    lengths = tuple(int(length) for length in lengths_array)
    if any(length <= 0 for length in lengths):
        raise ValueError("segment_lengths must contain positive values.")
    if sum(lengths) != total_length:
        raise ValueError("segment_lengths must sum to the number of value rows.")
    return lengths


@dataclass(frozen=True)
class SegmentedSeries:
    """Owned values plus frozen boundaries for independent trajectories."""

    values: np.ndarray
    segment_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("values must have shape [T, variables].")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite.")
        lengths = _validated_lengths(self.segment_lengths, len(values))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "segment_lengths", lengths)

    @classmethod
    def from_segments(cls, segments: Sequence[np.ndarray]) -> SegmentedSeries:
        arrays = tuple(np.asarray(segment, dtype=float) for segment in segments)
        if not arrays:
            raise ValueError("segments must be non-empty.")
        if any(array.ndim != 2 or len(array) == 0 for array in arrays):
            raise ValueError("Each segment must have shape [T, variables] and be non-empty.")
        n_variables = arrays[0].shape[1]
        if n_variables == 0 or any(array.shape[1] != n_variables for array in arrays):
            raise ValueError("All segments must contain the same variables.")
        return cls(
            values=np.concatenate(arrays, axis=0),
            segment_lengths=tuple(len(array) for array in arrays),
        )

    def iter_segments(self) -> tuple[np.ndarray, ...]:
        boundaries = np.cumsum((0, *self.segment_lengths))
        return tuple(
            self.values[boundaries[index] : boundaries[index + 1]]
            for index in range(len(self.segment_lengths))
        )

    def usable_rows(self, max_lag: int) -> int:
        if max_lag < 1 or any(length <= max_lag for length in self.segment_lengths):
            raise ValueError("Every segment must have more rows than max_lag.")
        return sum(length - max_lag for length in self.segment_lengths)


@dataclass(frozen=True)
class InterventionArms:
    """Small, size-matched observational, and combined intervention arms."""

    small: SegmentedSeries
    big: SegmentedSeries
    combined: SegmentedSeries
    intervention_type: InterventionType
    strength: float
    observational_seeds: tuple[int, ...]
    intervention_seeds: tuple[int, ...]
    small_intervention_targets: tuple[int | None, ...]
    big_intervention_targets: tuple[int | None, ...]
    combined_intervention_targets: tuple[int | None, ...]
    preprocessing: str = "raw"
    noise_std: float = 0.1
    burnin: int = 300
    reference_scale: np.ndarray | None = None
    reference_mean: np.ndarray | None = None
    scale_floor: float | None = None

    def __post_init__(self) -> None:
        if len(self.small.segment_lengths) != 1:
            raise ValueError("small must contain exactly one observational segment.")
        if self.big.segment_lengths != self.combined.segment_lengths:
            raise ValueError("big and combined must have identical segment lengths.")
        if self.big.segment_lengths[0] != self.small.segment_lengths[0]:
            raise ValueError("Matched arms must start with the small segment length.")
        if self.intervention_type not in {"random", "constant", "soft"}:
            raise ValueError(f"Unknown intervention_type: {self.intervention_type}")
        if len(self.observational_seeds) != len(self.big.segment_lengths) - 1:
            raise ValueError("observational_seeds must identify supplemental big-arm segments.")
        if len(self.intervention_seeds) != len(self.combined.segment_lengths) - 1:
            raise ValueError("intervention_seeds must identify every intervention segment.")
        arm_targets = (
            ("small", self.small, self.small_intervention_targets),
            ("big", self.big, self.big_intervention_targets),
            ("combined", self.combined, self.combined_intervention_targets),
        )
        n_variables = self.small.values.shape[1]
        for arm_name, series, targets in arm_targets:
            if len(targets) != len(series.segment_lengths):
                raise ValueError(f"{arm_name}_intervention_targets must match segment count.")
            if any(
                target is not None
                and (
                    isinstance(target, (bool, np.bool_))
                    or not isinstance(target, (int, np.integer))
                    or not 0 <= int(target) < n_variables
                )
                for target in targets
            ):
                raise ValueError(f"{arm_name}_intervention_targets contains an invalid target.")
        if any(target is not None for target in self.small_intervention_targets):
            raise ValueError("small segments must be observational.")
        if any(target is not None for target in self.big_intervention_targets):
            raise ValueError("big segments must be observational.")
        if self.reference_scale is not None:
            scale = np.array(self.reference_scale, dtype=float, copy=True)
            if scale.shape != (self.small.values.shape[1],):
                raise ValueError("reference_scale must have one value per variable.")
            object.__setattr__(self, "reference_scale", scale)
        if self.reference_mean is not None:
            mean = np.array(self.reference_mean, dtype=float, copy=True)
            if mean.shape != (self.small.values.shape[1],):
                raise ValueError("reference_mean must have one value per variable.")
            object.__setattr__(self, "reference_mean", mean)

    def metadata(self, max_lag: int) -> dict[str, object]:
        metadata = {
            "intervention_type": self.intervention_type,
            "strength": float(self.strength),
            "preprocessing": self.preprocessing,
            "reference_scale_source": "small_observational",
            "noise_std": float(self.noise_std),
            "burnin": int(self.burnin),
            "scale_floor": (None if self.scale_floor is None else float(self.scale_floor)),
            "observational_seeds": list(self.observational_seeds),
            "intervention_seeds": list(self.intervention_seeds),
            "intervention_targets": {
                "small": list(self.small_intervention_targets),
                "big": list(self.big_intervention_targets),
                "combined": list(self.combined_intervention_targets),
            },
            "signed_clamp_values": {
                "small": [None] * len(self.small_intervention_targets),
                "big": [None] * len(self.big_intervention_targets),
                "combined": self._combined_signed_clamp_values(),
            },
            "segment_lengths": {
                "small": list(self.small.segment_lengths),
                "big": list(self.big.segment_lengths),
                "combined": list(self.combined.segment_lengths),
            },
            "usable_rows": {
                "small": self.small.usable_rows(max_lag),
                "big": self.big.usable_rows(max_lag),
                "combined": self.combined.usable_rows(max_lag),
            },
        }
        if self.reference_scale is not None:
            metadata["reference_scale"] = self.reference_scale.tolist()
        if self.reference_mean is not None:
            metadata["reference_mean"] = self.reference_mean.tolist()
        return metadata

    def _combined_signed_clamp_values(self) -> list[float | None]:
        if self.intervention_type != "constant":
            return [None] * len(self.combined_intervention_targets)
        return [
            None if target is None else float(self.strength * (-1) ** target)
            for target in self.combined_intervention_targets
        ]

    def target_valid_mask(self, arm: Literal["small", "big", "combined"]) -> np.ndarray:
        """Return [T, K] validity for target outcomes in one experiment arm."""
        if arm not in {"small", "big", "combined"}:
            raise ValueError(f"Unknown arm: {arm}")
        series = getattr(self, arm)
        targets = getattr(self, f"{arm}_intervention_targets")
        valid = np.ones(series.values.shape, dtype=bool)
        start = 0
        for length, target in zip(series.segment_lengths, targets):
            if target is not None:
                valid[start : start + length, int(target)] = False
            start += length
        return valid


def _validate_seed_count(seeds: Sequence[int], count: int) -> tuple[int, ...]:
    validated = tuple(int(seed) for seed in seeds)
    if len(validated) != count:
        raise ValueError(f"Expected {count} independent seeds, got {len(validated)}.")
    return validated


def generate_observational_segments(
    A_lags: np.ndarray,
    lengths: Sequence[int],
    seeds: Sequence[int],
    noise_std: float = 0.1,
    burnin: int = 300,
) -> SegmentedSeries:
    """Generate independent raw observational trajectories with exact lengths."""
    requested_lengths = tuple(int(length) for length in lengths)
    if not requested_lengths or any(length <= 0 for length in requested_lengths):
        raise ValueError("lengths must contain positive values.")
    independent_seeds = _validate_seed_count(seeds, len(requested_lengths))
    segments = [
        simulate_lagged_var(
            A_lags,
            T=length,
            noise_std=noise_std,
            seed=seed,
            burnin=burnin,
            standardize_output=False,
        )
        for length, seed in zip(requested_lengths, independent_seeds)
    ]
    return SegmentedSeries.from_segments(segments)


def generate_intervention_segments(
    A_lags: np.ndarray,
    T_per_target: int,
    intervention_type: InterventionType,
    strength: float,
    seeds: Sequence[int],
    noise_std: float = 0.1,
    burnin: int = 300,
) -> SegmentedSeries:
    """Generate one raw intervention trajectory per target.

    Constant interventions alternate exact signed clamps by target:
    ``+strength, -strength, +strength, ...``.
    """
    A_lags = np.asarray(A_lags, dtype=float)
    if A_lags.ndim != 3 or A_lags.shape[1] != A_lags.shape[2]:
        raise ValueError("A_lags must have shape [max_lag, variables, variables].")
    if intervention_type not in {"random", "constant", "soft"}:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")
    if T_per_target <= 0:
        raise ValueError("T_per_target must be positive.")

    max_lag, n_variables, _ = A_lags.shape
    independent_seeds = _validate_seed_count(seeds, n_variables)
    segments = []
    for target, seed in enumerate(independent_seeds):
        rng = np.random.RandomState(seed)
        values = np.zeros((T_per_target + burnin, n_variables))
        values[:max_lag] = rng.randn(max_lag, n_variables) * noise_std
        constant_value = strength if target % 2 == 0 else -strength

        for time_index in range(max_lag, len(values)):
            value = np.zeros(n_variables)
            for lag in range(max_lag):
                value += A_lags[lag] @ values[time_index - lag - 1]
            value += rng.randn(n_variables) * noise_std
            if intervention_type == "random":
                value[target] = strength * rng.choice((-1.0, 1.0))
            elif intervention_type == "constant":
                value[target] = constant_value
            else:
                value[target] += strength * rng.randn()
            values[time_index] = value

        segments.append(values[burnin:])
    return SegmentedSeries.from_segments(segments)


def build_raw_intervention_arms(
    A_lags: np.ndarray,
    small_observational: SegmentedSeries,
    T_int_per_target: int,
    intervention_type: InterventionType,
    strength: float,
    root_seed: int,
    noise_std: float = 0.1,
    burnin: int = 300,
) -> InterventionArms:
    """Build raw S/B/C arms with exactly matched B/C segmentation."""
    if len(small_observational.segment_lengths) != 1:
        raise ValueError("small_observational must contain exactly one segment.")
    n_variables = small_observational.values.shape[1]
    if np.asarray(A_lags).shape[1:] != (n_variables, n_variables):
        raise ValueError("A_lags and small_observational variable counts must match.")

    observational_seeds = tuple(
        _derive_seed(root_seed, namespace=0x4F4253, index=index) for index in range(n_variables)
    )
    intervention_seeds = tuple(
        _derive_seed(root_seed, namespace=0x494E54, index=index) for index in range(n_variables)
    )
    supplemental_observations = generate_observational_segments(
        A_lags,
        lengths=(T_int_per_target,) * n_variables,
        seeds=observational_seeds,
        noise_std=noise_std,
        burnin=burnin,
    )
    big = SegmentedSeries.from_segments(
        (small_observational.values, *supplemental_observations.iter_segments())
    )
    interventions = generate_intervention_segments(
        A_lags,
        T_per_target=T_int_per_target,
        intervention_type=intervention_type,
        strength=strength,
        seeds=intervention_seeds,
        noise_std=noise_std,
        burnin=burnin,
    )
    combined = SegmentedSeries.from_segments(
        (small_observational.values, *interventions.iter_segments())
    )
    return InterventionArms(
        small=small_observational,
        big=big,
        combined=combined,
        intervention_type=intervention_type,
        strength=float(strength),
        observational_seeds=observational_seeds,
        intervention_seeds=intervention_seeds,
        small_intervention_targets=(None,),
        big_intervention_targets=(None,) * (n_variables + 1),
        combined_intervention_targets=(None, *range(n_variables)),
        noise_std=float(noise_std),
        burnin=int(burnin),
    )


def _derive_seed(root_seed: int, namespace: int, index: int) -> int:
    """Derive a stable uint32 seed from a root and stream identity."""
    seed_sequence = np.random.SeedSequence([int(root_seed), int(namespace), int(index)])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _transform_series_with_segment_means(
    series: SegmentedSeries,
    reference_scale: np.ndarray,
) -> SegmentedSeries:
    return SegmentedSeries.from_segments(
        tuple(
            (segment - segment.mean(axis=0, keepdims=True)) / reference_scale
            for segment in series.iter_segments()
        )
    )


def _transform_series_with_reference_mean(
    series: SegmentedSeries,
    reference_mean: np.ndarray,
    reference_scale: np.ndarray,
) -> SegmentedSeries:
    return SegmentedSeries.from_segments(
        tuple((segment - reference_mean) / reference_scale for segment in series.iter_segments())
    )


def preprocess_intervention_arms(
    raw_arms: InterventionArms,
    scale_floor: float = 1e-8,
) -> tuple[InterventionArms, np.ndarray]:
    """
    Apply intervention-aware centering and one small-observational scale.

    Random and soft environments are centered independently. Constant do(X=c)
    environments use the small observational mean so clamp and descendant level
    shifts remain. Variances are never fitted per arm or segment.
    """
    reference_mean = raw_arms.small.values.mean(axis=0)
    reference_scale = np.maximum(
        raw_arms.small.values.std(axis=0),
        scale_floor,
    )
    if raw_arms.intervention_type == "constant":
        transformed_small = _transform_series_with_reference_mean(
            raw_arms.small, reference_mean, reference_scale
        )
        transformed_big = _transform_series_with_reference_mean(
            raw_arms.big, reference_mean, reference_scale
        )
        transformed_combined = _transform_series_with_reference_mean(
            raw_arms.combined, reference_mean, reference_scale
        )
        preprocessing = "small_observational_mean_and_scale_for_all_segments"
    else:
        transformed_small = _transform_series_with_segment_means(raw_arms.small, reference_scale)
        transformed_big = _transform_series_with_segment_means(raw_arms.big, reference_scale)
        transformed_combined = _transform_series_with_segment_means(
            raw_arms.combined, reference_scale
        )
        preprocessing = "per_segment_mean_with_small_observational_scale"

    transformed = replace(
        raw_arms,
        small=transformed_small,
        big=transformed_big,
        combined=transformed_combined,
        preprocessing=preprocessing,
        reference_scale=reference_scale,
        reference_mean=reference_mean,
        scale_floor=float(scale_floor),
    )
    return transformed, transformed.reference_scale.copy()
