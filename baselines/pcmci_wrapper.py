"""
PCMCI baseline wrapper.

PCMCI (Runge et al., 2019) is the gold-standard method for causal discovery
from time series. It handles autocorrelation and is designed for climate data.
"""

from collections.abc import Sequence

import numpy as np


def _validate_segment_lengths(
    segment_lengths: Sequence[int] | None,
    n_samples: int,
    tau_max: int,
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
    minimum_length = 2 * tau_max
    if any(length <= minimum_length for length in validated):
        raise ValueError(
            f"each segment length in segment_lengths must be greater than "
            f"2 * tau_max ({minimum_length})"
        )
    return validated


def _validate_target_valid_mask(target_valid_mask, X):
    if target_valid_mask is None:
        return None
    mask = np.asarray(target_valid_mask)
    if mask.dtype != np.bool_ or mask.shape != X.shape:
        raise ValueError("target_valid_mask must be boolean with the same shape as X")
    return mask


def _make_tigramite_dataframe(
    pp,
    X,
    segment_lengths,
    target_valid_mask=None,
):
    """Build Tigramite data, removing segmented offset nuisances by mean centering."""
    if segment_lengths is None:
        kwargs = {} if target_valid_mask is None else {"mask": ~target_valid_mask}
        return pp.DataFrame(X, **kwargs)

    boundaries = np.cumsum([0, *segment_lengths])
    data_dict = {}
    mask_dict = {}
    for segment_id in range(len(segment_lengths)):
        start = boundaries[segment_id]
        stop = boundaries[segment_id + 1]
        segment = X[start:stop]
        data_dict[segment_id] = segment - segment.mean(axis=0, keepdims=True)
        if target_valid_mask is not None:
            mask_dict[segment_id] = ~target_valid_mask[start:stop]
    kwargs = {"analysis_mode": "multiple"}
    if target_valid_mask is not None:
        kwargs["mask"] = mask_dict
    return pp.DataFrame(data_dict, **kwargs)


def _validated_library_matrix(
    matrix: np.ndarray,
    *,
    method_name: str,
    output_name: str,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{method_name} returned non-finite {output_name}.")
    return values


def _aggregate_lagged_scores(
    val_matrix: np.ndarray,
    method_name: str = "PCMCI",
) -> np.ndarray:
    lag_values = _validated_library_matrix(
        val_matrix,
        method_name=method_name,
        output_name="val_matrix",
    )[:, :, 1:]
    S = np.abs(lag_values).max(axis=2).T
    np.fill_diagonal(S, 0.0)
    if S.max() > 0:
        S = S / S.max()
    return S


def pcmci_causality_matrix(
    X: np.ndarray,
    tau_max: int = 5,
    pc_alpha: float = 0.05,
    method: str = "parcorr",
    segment_lengths: Sequence[int] | None = None,
    target_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run PCMCI and return a causal score matrix.

    Args:
        X: Multivariate time series (T, K).
        tau_max: Maximum lag to consider.
        pc_alpha: Significance level for the PC condition-selection step.
        method: Independence test:
            'parcorr' — Partial correlation (linear, fast).
            'gpdc' — Gaussian Process Distance Correlation (nonlinear, slow).
        segment_lengths: Optional independent trajectory lengths, each greater
            than 2 * tau_max. Segments are mean-centered per variable, without
            variance scaling, to remove offset nuisances.

    Returns:
        S: Causal score matrix (K, K).
           S[i, j] > 0 suggests j causes i (aggregated across lags).
    """
    validated_lengths = _validate_segment_lengths(
        segment_lengths,
        X.shape[0],
        tau_max,
    )
    validated_target_mask = _validate_target_valid_mask(target_valid_mask, X)

    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        if method == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr

            cond_ind_test = ParCorr(
                significance="analytic",
                **({"mask_type": "y"} if validated_target_mask is not None else {}),
            )
        elif method == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC

            cond_ind_test = GPDC(
                significance="analytic",
                **({"mask_type": "y"} if validated_target_mask is not None else {}),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    except ImportError:
        raise ImportError(
            "tigramite is required for PCMCI baseline. Install it with: pip install tigramite"
        )

    dataframe = _make_tigramite_dataframe(
        pp,
        X,
        validated_lengths,
        validated_target_mask,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    val_matrix = results["val_matrix"]
    return _aggregate_lagged_scores(val_matrix, method_name="PCMCI")


def pcmci_plus_causality_matrix(
    X: np.ndarray,
    tau_max: int = 5,
    tau_min: int = 1,
    pc_alpha: float = 0.05,
    method: str = "parcorr",
    segment_lengths: Sequence[int] | None = None,
    target_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run PCMCI+ and return an aggregate lagged causal score matrix.

    PCMCI+ extends PCMCI with contemporaneous orientation rules. For this
    benchmark we keep the same lagged scoring convention as PCMCI: aggregate
    absolute test statistics over strictly positive lags. Independent segments
    are mean-centered per variable without variance scaling.
    """
    validated_lengths = _validate_segment_lengths(
        segment_lengths,
        X.shape[0],
        tau_max,
    )
    validated_target_mask = _validate_target_valid_mask(target_valid_mask, X)

    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        if method == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr

            cond_ind_test = ParCorr(
                significance="analytic",
                **({"mask_type": "y"} if validated_target_mask is not None else {}),
            )
        elif method == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC

            cond_ind_test = GPDC(
                significance="analytic",
                **({"mask_type": "y"} if validated_target_mask is not None else {}),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    except ImportError:
        raise ImportError(
            "tigramite is required for PCMCI+ baseline. Install it with: uv sync --extra baselines"
        )

    dataframe = _make_tigramite_dataframe(
        pp,
        X,
        validated_lengths,
        validated_target_mask,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    val_matrix = results["val_matrix"]

    return _aggregate_lagged_scores(val_matrix, method_name="PCMCI+")


def pcmci_with_details(
    X: np.ndarray,
    tau_max: int = 5,
    pc_alpha: float = 0.05,
    segment_lengths: Sequence[int] | None = None,
    target_valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Run PCMCI and return detailed results including per-lag scores
    and p-values (useful for analysis and figures). Independent segments are
    mean-centered per variable without variance scaling.

    Returns:
        Dict with project-oriented score_matrix[target, source] and raw
        Tigramite outputs. In particular, val_matrix retains Tigramite's
        native val_matrix[source, target, lag] orientation.
    """
    validated_lengths = _validate_segment_lengths(
        segment_lengths,
        X.shape[0],
        tau_max,
    )
    validated_target_mask = _validate_target_valid_mask(target_valid_mask, X)

    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError:
        raise ImportError("tigramite required. pip install tigramite")

    dataframe = _make_tigramite_dataframe(
        pp,
        X,
        validated_lengths,
        validated_target_mask,
    )
    parcorr = ParCorr(
        significance="analytic",
        **({"mask_type": "y"} if validated_target_mask is not None else {}),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    val_matrix = _validated_library_matrix(
        results["val_matrix"],
        method_name="PCMCI",
        output_name="val_matrix",
    )
    p_matrix = _validated_library_matrix(
        results["p_matrix"],
        method_name="PCMCI",
        output_name="p_matrix",
    )

    S = _aggregate_lagged_scores(val_matrix, method_name="PCMCI")

    return {
        "score_matrix": S,
        "p_matrix": p_matrix,
        "val_matrix": val_matrix,
        "graph": results.get("graph", None),
    }
