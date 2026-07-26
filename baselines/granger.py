"""
Bivariate Granger causality baseline.

Wraps statsmodels to produce a causal score matrix comparable
to our SSM-based method.
"""

from collections.abc import Sequence

import numpy as np


def _validate_segment_lengths(
    segment_lengths: Sequence[int] | None,
    n_samples: int,
    max_lag: int,
) -> list[int] | None:
    if segment_lengths is None:
        return None

    lengths = np.asarray(segment_lengths)
    is_numeric = np.issubdtype(lengths.dtype, np.integer) or np.issubdtype(
        lengths.dtype, np.floating
    )
    if lengths.ndim != 1 or lengths.size == 0 or not is_numeric:
        raise ValueError(
            "segment_lengths must be a non-empty 1D sequence of positive integer-like values"
        )
    if (
        not np.all(np.isfinite(lengths))
        or not np.all(lengths == np.floor(lengths))
        or np.any(lengths <= 0)
    ):
        raise ValueError(
            "segment_lengths must be a non-empty 1D sequence of positive integer-like values"
        )

    validated = [int(length) for length in lengths]
    if sum(validated) != n_samples:
        raise ValueError(
            f"segment_lengths must sum to X.shape[0] ({n_samples}), got {sum(validated)}"
        )
    if any(length <= max_lag for length in validated):
        raise ValueError(
            f"each segment length in segment_lengths must be greater than "
            f"the maximum lag ({max_lag})"
        )
    return validated


def _validate_target_valid_mask(target_valid_mask, X):
    if target_valid_mask is None:
        return None
    mask = np.asarray(target_valid_mask)
    if mask.dtype != np.bool_ or mask.shape != X.shape:
        raise ValueError("target_valid_mask must be boolean with the same shape as X")
    return mask


def _build_segmented_lag_rows(
    segments,
    target,
    source,
    lag,
    target_valid_masks=None,
):
    outcomes = []
    restricted_rows = []
    unrestricted_rows = []
    n_segments = len(segments)
    if target_valid_masks is None:
        target_valid_masks = [None] * n_segments
    if len(target_valid_masks) != n_segments:
        raise ValueError("target_valid_masks must match the segment count")
    for segment_id, segment in enumerate(segments):
        target_history = np.column_stack(
            [segment[lag - step : -step, target] for step in range(1, lag + 1)]
        )
        source_history = np.column_stack(
            [segment[lag - step : -step, source] for step in range(1, lag + 1)]
        )
        fixed_effects = np.zeros((len(segment) - lag, n_segments))
        fixed_effects[:, segment_id] = 1.0
        restricted = np.column_stack([fixed_effects, target_history])
        valid_rows = (
            np.ones(len(segment) - lag, dtype=bool)
            if target_valid_masks[segment_id] is None
            else target_valid_masks[segment_id][lag:, target]
        )
        outcomes.append(segment[lag:, target][valid_rows])
        restricted_rows.append(restricted[valid_rows])
        unrestricted_rows.append(np.column_stack([restricted, source_history])[valid_rows])

    return (
        np.concatenate(outcomes),
        np.concatenate(restricted_rows),
        np.concatenate(unrestricted_rows),
    )


def _residual_sum_squares(design, outcome):
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    return residuals @ residuals


def _scale_pooled_regression_data(
    outcome,
    restricted,
    unrestricted,
    n_segments,
):
    outcome_scale = np.max(np.abs(outcome))
    if outcome_scale == 0:
        outcome_scale = 1.0

    target_history = restricted[:, n_segments:]
    target_scales = np.max(np.abs(target_history), axis=0)
    target_scales[target_scales == 0] = 1.0

    source_history = unrestricted[:, restricted.shape[1] :]
    source_scales = np.max(np.abs(source_history), axis=0)
    source_scales[source_scales == 0] = 1.0

    fixed_effects = restricted[:, :n_segments]
    scaled_restricted = np.column_stack([fixed_effects, target_history / target_scales])
    scaled_unrestricted = np.column_stack([scaled_restricted, source_history / source_scales])
    return outcome / outcome_scale, scaled_restricted, scaled_unrestricted


def _nested_regression_f_test(outcome, restricted, unrestricted):
    from scipy.stats import f as f_distribution

    restricted_rank = np.linalg.matrix_rank(restricted)
    unrestricted_rank = np.linalg.matrix_rank(unrestricted)
    numerator_df = unrestricted_rank - restricted_rank
    denominator_df = len(outcome) - unrestricted_rank
    if numerator_df <= 0 or denominator_df <= 0:
        return None

    try:
        restricted_rss = _residual_sum_squares(restricted, outcome)
        unrestricted_rss = _residual_sum_squares(unrestricted, outcome)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(restricted_rss) or not np.isfinite(unrestricted_rss):
        return None

    eps = np.finfo(float).eps
    outcome_variation = float(outcome @ outcome)
    rss_floor = eps * len(outcome) * max(outcome_variation, 1.0)
    if unrestricted_rss <= rss_floor:
        return None

    rss_reduction = restricted_rss - unrestricted_rss
    reduction_tolerance = eps * len(outcome) * max(restricted_rss, unrestricted_rss, 1.0)
    if rss_reduction <= reduction_tolerance:
        return 0.0, 1.0

    f_stat = (rss_reduction / numerator_df) / (unrestricted_rss / denominator_df)
    p_value = f_distribution.sf(f_stat, numerator_df, denominator_df)
    if not np.isfinite(f_stat) or not np.isfinite(p_value):
        return None
    return f_stat, p_value


def _segmented_granger_causality_matrix(
    X,
    max_lag,
    significance,
    score_type,
    segment_lengths,
    target_valid_mask=None,
):
    boundaries = np.cumsum([0, *segment_lengths])
    segments = []
    segment_masks = []
    for index in range(len(segment_lengths)):
        start = boundaries[index]
        stop = boundaries[index + 1]
        segment = X[start:stop]
        relative = segment - segment[:1]
        segments.append(relative - relative.mean(axis=0, keepdims=True))
        if target_valid_mask is not None:
            segment_masks.append(target_valid_mask[start:stop])
    n_variables = X.shape[1]
    score = np.zeros((n_variables, n_variables))

    for target in range(n_variables):
        for source in range(n_variables):
            if target == source:
                continue

            p_values = []
            f_stats = []
            for lag in range(1, max_lag + 1):
                outcome, restricted, unrestricted = _build_segmented_lag_rows(
                    segments,
                    target,
                    source,
                    lag,
                    target_valid_masks=(segment_masks if target_valid_mask is not None else None),
                )
                if not (
                    np.all(np.isfinite(outcome))
                    and np.all(np.isfinite(restricted))
                    and np.all(np.isfinite(unrestricted))
                ):
                    continue

                outcome, restricted, unrestricted = _scale_pooled_regression_data(
                    outcome,
                    restricted,
                    unrestricted,
                    n_segments=len(segments),
                )
                test_result = _nested_regression_f_test(
                    outcome,
                    restricted,
                    unrestricted,
                )
                if test_result is None:
                    continue
                f_stat, p_value = test_result
                f_stats.append(f_stat)
                p_values.append(p_value)

            if not p_values:
                continue
            min_p = min(p_values)
            max_f = max(f_stats)
            if score_type == "neg_log_pvalue":
                score[target, source] = -np.log10(max(min_p, 1e-300))
            elif score_type == "binary":
                score[target, source] = 1.0 if min_p < significance else 0.0
            elif score_type == "f_stat":
                score[target, source] = max_f

    if score.max() > 0:
        score /= score.max()
    return score


def granger_causality_matrix(
    X: np.ndarray,
    max_lag: int = 5,
    significance: float = 0.05,
    score_type: str = "neg_log_pvalue",
    segment_lengths: Sequence[int] | None = None,
    target_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute pairwise Granger causality scores.

    Tests whether X_j Granger-causes X_i for all pairs (i, j).

    Args:
        X: Multivariate time series (T, K).
        max_lag: Maximum lag to test.
        significance: P-value threshold for binary decisions.
        score_type: How to convert p-values to scores:
            'neg_log_pvalue' — -log10(p), higher = more causal (default).
            'binary' — 1 if significant, 0 otherwise.
            'f_stat' — F-statistic value.
        segment_lengths: Optional lengths of independent trajectories.

    Returns:
        S: Causal score matrix (K, K).
           S[i, j] > 0 suggests j Granger-causes i.
    """
    if score_type not in {"neg_log_pvalue", "binary", "f_stat"}:
        raise ValueError(f"Unknown score_type: {score_type}")
    validated_lengths = _validate_segment_lengths(
        segment_lengths,
        X.shape[0],
        max_lag,
    )
    validated_target_mask = _validate_target_valid_mask(target_valid_mask, X)
    if validated_lengths is not None or validated_target_mask is not None:
        return _segmented_granger_causality_matrix(
            X,
            max_lag,
            significance,
            score_type,
            validated_lengths or [len(X)],
            validated_target_mask,
        )

    from statsmodels.tsa.stattools import grangercausalitytests

    K = X.shape[1]
    S = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                continue

            try:
                # grangercausalitytests expects [target, predictor]
                data = np.column_stack([X[:, i], X[:, j]])
                results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

                # Get the minimum p-value across all tested lags
                p_values = []
                f_stats = []
                for lag in range(1, max_lag + 1):
                    p_val = results[lag][0]["ssr_ftest"][1]
                    f_stat = results[lag][0]["ssr_ftest"][0]
                    p_values.append(p_val)
                    f_stats.append(f_stat)

                min_p = min(p_values)
                max_f = max(f_stats)

                if score_type == "neg_log_pvalue":
                    S[i, j] = -np.log10(max(min_p, 1e-300))
                elif score_type == "binary":
                    S[i, j] = 1.0 if min_p < significance else 0.0
                elif score_type == "f_stat":
                    S[i, j] = max_f
            except Exception as error:
                raise RuntimeError(
                    f"Granger failed for target={i}, source={j}, max_lag={max_lag}: {error}"
                ) from error

    # Normalize to [0, 1] for comparability
    if S.max() > 0:
        S = S / S.max()

    return S
