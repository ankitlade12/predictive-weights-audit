"""
Bivariate Granger causality baseline.

Wraps statsmodels to produce a causal score matrix comparable
to our SSM-based method.
"""

import numpy as np
from typing import Optional


def granger_causality_matrix(
    X: np.ndarray,
    max_lag: int = 5,
    significance: float = 0.05,
    score_type: str = "neg_log_pvalue",
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

    Returns:
        S: Causal score matrix (K, K).
           S[i, j] > 0 suggests j Granger-causes i.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    T, K = X.shape
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
                else:
                    raise ValueError(f"Unknown score_type: {score_type}")

            except Exception:
                S[i, j] = 0.0

    # Normalize to [0, 1] for comparability
    if S.max() > 0:
        S = S / S.max()

    return S
