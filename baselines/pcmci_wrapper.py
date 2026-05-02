"""
PCMCI baseline wrapper.

PCMCI (Runge et al., 2019) is the gold-standard method for causal discovery
from time series. It handles autocorrelation and is designed for climate data.
"""


import numpy as np


def pcmci_causality_matrix(
    X: np.ndarray,
    tau_max: int = 5,
    pc_alpha: float = 0.05,
    method: str = "parcorr",
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

    Returns:
        S: Causal score matrix (K, K).
           S[i, j] > 0 suggests j causes i (aggregated across lags).
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        if method == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr
            cond_ind_test = ParCorr(significance="analytic")
        elif method == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC
            cond_ind_test = GPDC(significance="analytic")
        else:
            raise ValueError(f"Unknown method: {method}")

    except ImportError:
        raise ImportError(
            "tigramite is required for PCMCI baseline. "
            "Install it with: pip install tigramite"
        )

    T, K = X.shape

    dataframe = pp.DataFrame(X)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    # results['val_matrix'] has shape (K, K, tau_max+1)
    # Aggregate across lags: take absolute max across all lags > 0
    val_matrix = results["val_matrix"]

    # Exclude lag 0 (contemporaneous) and take abs max across lags
    S = np.abs(val_matrix[:, :, 1:]).max(axis=2)  # (K, K)

    # Normalize
    if S.max() > 0:
        S = S / S.max()

    return S


def pcmci_plus_causality_matrix(
    X: np.ndarray,
    tau_max: int = 5,
    tau_min: int = 1,
    pc_alpha: float = 0.05,
    method: str = "parcorr",
) -> np.ndarray:
    """
    Run PCMCI+ and return an aggregate lagged causal score matrix.

    PCMCI+ extends PCMCI with contemporaneous orientation rules. For this
    benchmark we keep the same lagged scoring convention as PCMCI: aggregate
    absolute test statistics over strictly positive lags.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        if method == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr
            cond_ind_test = ParCorr(significance="analytic")
        elif method == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC
            cond_ind_test = GPDC(significance="analytic")
        else:
            raise ValueError(f"Unknown method: {method}")

    except ImportError:
        raise ImportError(
            "tigramite is required for PCMCI+ baseline. "
            "Install it with: uv sync --extra baselines"
        )

    dataframe = pp.DataFrame(X)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    val_matrix = results["val_matrix"]

    S = np.abs(val_matrix[:, :, 1:]).max(axis=2)
    np.fill_diagonal(S, 0.0)
    if S.max() > 0:
        S = S / S.max()
    return S


def pcmci_with_details(
    X: np.ndarray,
    tau_max: int = 5,
    pc_alpha: float = 0.05,
) -> dict:
    """
    Run PCMCI and return detailed results including per-lag scores
    and p-values (useful for analysis and figures).

    Returns:
        Dict with keys: score_matrix, p_matrix, val_matrix, graph.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError:
        raise ImportError("tigramite required. pip install tigramite")

    T, K = X.shape

    dataframe = pp.DataFrame(X)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]

    # Aggregate score
    S = np.abs(val_matrix[:, :, 1:]).max(axis=2)
    if S.max() > 0:
        S = S / S.max()

    return {
        "score_matrix": S,
        "p_matrix": p_matrix,
        "val_matrix": val_matrix,
        "graph": results.get("graph", None),
    }
