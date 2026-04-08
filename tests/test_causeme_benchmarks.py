import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm_causal.causeme_benchmarks import (
    default_causeme_scenarios,
    generate_causeme_linear,
    generate_causeme_nonlinear,
)


def test_default_causeme_scenarios_include_expected_names():
    scenarios = default_causeme_scenarios(include_nonlinear=True)
    names = [scenario.name for scenario in scenarios]

    assert "CauseMe-linear-K5" in names
    assert "CauseMe-linear-K20" in names
    assert "CauseMe-nonlinear-K10" in names


def test_generate_causeme_linear_returns_finite_shapes():
    X, G_agg, G_lag, desc = generate_causeme_linear(
        K=4,
        T=60,
        max_lag=2,
        density=0.25,
        seed=3,
    )

    assert X.shape == (60, 4)
    assert G_agg.shape == (4, 4)
    assert G_lag.shape == (2, 4, 4)
    assert np.isfinite(X).all()
    assert np.isfinite(G_lag).all()
    assert "K=4" in desc
    assert np.allclose(np.diag(G_agg), 0.0)


def test_generate_causeme_nonlinear_returns_finite_shapes():
    X, G_agg, G_lag, desc = generate_causeme_nonlinear(
        K=4,
        T=60,
        max_lag=2,
        density=0.25,
        seed=5,
    )

    assert X.shape == (60, 4)
    assert G_agg.shape == (4, 4)
    assert G_lag.shape == (2, 4, 4)
    assert np.isfinite(X).all()
    assert np.isfinite(G_lag).all()
    assert "K=4" in desc
    assert np.allclose(np.diag(G_agg), 0.0)
