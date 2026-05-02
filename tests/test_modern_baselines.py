import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.dynotears import dynotears_causality_matrix
from baselines.lingam_wrapper import var_lingam_causality_matrix
from baselines.pcmci_wrapper import pcmci_plus_causality_matrix
from experiments.exp24_real_data_benchmark import build_method_registry


def test_pcmci_plus_returns_finite_score_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))

    S = pcmci_plus_causality_matrix(X, tau_max=2, pc_alpha=0.2)

    assert S.shape == (3, 3)
    assert np.all(np.isfinite(S))
    assert np.allclose(np.diag(S), 0.0)


def test_dynotears_reports_missing_optional_dependency():
    with pytest.raises(ImportError, match="causalnex"):
        dynotears_causality_matrix(np.zeros((20, 3)), max_lag=2)


def test_var_lingam_reports_missing_optional_dependency():
    with pytest.raises(ImportError, match="lingam"):
        var_lingam_causality_matrix(np.zeros((20, 3)), max_lag=2)


def test_real_data_registry_can_include_modern_baselines():
    methods = build_method_registry(device="cpu", epochs=1, include_modern_baselines=True)

    assert "PCMCI+" in methods
    assert "DYNOTEARS" in methods
    assert "VAR-LiNGAM" in methods
