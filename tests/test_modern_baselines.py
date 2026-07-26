import os
import sys
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import pcmci_wrapper
from baselines.dynotears import dynotears_causality_matrix
from baselines.lingam_wrapper import var_lingam_causality_matrix
from baselines.pcmci_wrapper import (
    _aggregate_lagged_scores,
    pcmci_causality_matrix,
    pcmci_plus_causality_matrix,
    pcmci_with_details,
)
from experiments import exp25_causal_baselines as exp25
from experiments.exp24_real_data_benchmark import build_method_registry
from experiments.exp25_causal_baselines import eval_pcmci


def _planted_lagged_process(n_samples=160):
    rng = np.random.default_rng(17)
    X = np.zeros((n_samples, 2))
    X[:, 0] = rng.normal(size=n_samples)
    X[0, 1] = rng.normal(scale=0.1)
    X[1:, 1] = 1.5 * X[:-1, 0] + rng.normal(scale=0.1, size=n_samples - 1)
    return X


def test_pcmci_reports_planted_direction_in_project_orientation():
    X = _planted_lagged_process()

    S = pcmci_causality_matrix(X, tau_max=1, pc_alpha=0.05)

    assert S[1, 0] > S[0, 1]


def test_pcmci_plus_reports_planted_direction_in_project_orientation():
    X = _planted_lagged_process()

    S = pcmci_plus_causality_matrix(X, tau_max=1, pc_alpha=0.05)

    assert S[1, 0] > S[0, 1]


def test_pcmci_details_orients_score_but_preserves_native_val_matrix():
    X = _planted_lagged_process()

    details = pcmci_with_details(X, tau_max=1, pc_alpha=0.05)

    assert details["score_matrix"][1, 0] > details["score_matrix"][0, 1]
    assert abs(details["val_matrix"][0, 1, 1]) > abs(details["val_matrix"][1, 0, 1])


def test_exp25_pcmci_transposes_each_native_tigramite_lag(monkeypatch):
    val_matrix = np.zeros((2, 2, 2))
    val_matrix[0, 1, 1] = 1.0
    monkeypatch.setattr(
        pcmci_wrapper,
        "pcmci_with_details",
        lambda X, tau_max, pc_alpha: {"val_matrix": val_matrix},
    )
    G_lag = np.zeros((1, 2, 2))
    G_lag[0, 1, 0] = 1.0

    score = eval_pcmci(np.zeros((10, 2)), G_lag, max_lag=1)

    assert score == pytest.approx(1.0)


def test_exp25_pcmci_uses_aggregate_auroc_as_primary_metric(monkeypatch):
    val_matrix = np.zeros((2, 2, 2))
    val_matrix[0, 1, 1] = 1.0
    monkeypatch.setattr(
        pcmci_wrapper,
        "pcmci_with_details",
        lambda X, tau_max, pc_alpha: {"val_matrix": val_matrix},
    )
    monkeypatch.setattr(
        exp25,
        "evaluate_lagged_scores",
        lambda score, graph: {
            "aggregate_auroc": 0.875,
            "flat_lag_auroc": 0.125,
        },
    )

    score = eval_pcmci(
        np.zeros((10, 2)),
        np.zeros((1, 2, 2)),
        max_lag=1,
    )

    assert score == pytest.approx(0.875)


def test_exp25_pcmci_rejects_non_finite_native_score_tensor(monkeypatch):
    val_matrix = np.zeros((2, 2, 2))
    val_matrix[0, 1, 1] = np.nan
    monkeypatch.setattr(
        pcmci_wrapper,
        "pcmci_with_details",
        lambda X, tau_max, pc_alpha: {"val_matrix": val_matrix},
    )

    with pytest.raises(ValueError, match="PCMCI.*non-finite.*score tensor"):
        eval_pcmci(
            np.zeros((10, 2)),
            np.zeros((1, 2, 2)),
            max_lag=1,
        )


def test_exp25_pcmci_preserves_canonical_global_lag_ranking(monkeypatch):
    val_matrix = np.zeros((3, 3, 3))
    val_matrix[0, 1, 1] = 100.0
    val_matrix[2, 0, 2] = 2.0
    captured_scores = []
    monkeypatch.setattr(
        pcmci_wrapper,
        "pcmci_with_details",
        lambda X, tau_max, pc_alpha: {"val_matrix": val_matrix},
    )

    def capture_score(score, graph):
        del graph
        captured_scores.append(score.copy())
        return {"aggregate_auroc": 0.5}

    monkeypatch.setattr(exp25, "evaluate_lagged_scores", capture_score)

    eval_pcmci(
        np.zeros((10, 3)),
        np.zeros((2, 3, 3)),
        max_lag=2,
    )

    aggregate_score = np.max(captured_scores[0], axis=0)
    canonical_score = _aggregate_lagged_scores(val_matrix)
    np.testing.assert_allclose(aggregate_score, canonical_score)
    assert aggregate_score[1, 0] > aggregate_score[0, 2]


@pytest.mark.parametrize("non_finite_value", [np.nan, np.inf, -np.inf])
def test_pcmci_aggregation_rejects_any_non_finite_library_value(
    non_finite_value,
):
    val_matrix = np.zeros((2, 2, 3))
    val_matrix[0, 1, 1:] = [0.5, non_finite_value]
    val_matrix[1, 0, 1] = 0.25

    with pytest.raises(ValueError, match="PCMCI.*non-finite.*val_matrix"):
        _aggregate_lagged_scores(val_matrix)


def test_pcmci_aggregation_rejects_all_nan_library_output():
    with pytest.raises(ValueError, match="PCMCI.*non-finite.*val_matrix"):
        _aggregate_lagged_scores(np.full((2, 2, 3), np.nan))


@pytest.mark.parametrize(
    "score_fn",
    [pcmci_causality_matrix, pcmci_plus_causality_matrix],
)
def test_pcmci_wrappers_return_finite_score_matrix(score_fn):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))

    S = score_fn(X, tau_max=2, pc_alpha=0.2)

    assert S.shape == (3, 3)
    assert np.all(np.isfinite(S))
    assert np.allclose(np.diag(S), 0.0)


def test_dynotears_uses_only_positive_lag_sources_to_lag_zero_targets(
    monkeypatch,
):
    class FakeGraph:
        def edges(self, data):
            assert data is True
            return [
                ("x0_lag0", "x1_lag0", {"weight": 100.0}),
                ("x0_lag1", "x1_lag0", {"weight": -0.5}),
                ("x1_lag2", "x0_lag0", {"weight": 1.0}),
                ("x0_lag2", "x1_lag1", {"weight": 50.0}),
                ("x0", "x1_lag0", {"weight": 40.0}),
                ("x0_lagbad", "x1_lag0", {"weight": 30.0}),
                ("x0_lag1_extra", "x1_lag0", {"weight": 20.0}),
                ("x0_lag1", "x1_bad", {"weight": 10.0}),
                ("x9_lag1", "x1_lag0", {"weight": 10.0}),
            ]

    dynotears_module = ModuleType("causalnex.structure.dynotears")
    dynotears_module.from_pandas_dynamic = lambda dataframe, p: FakeGraph()
    structure_module = ModuleType("causalnex.structure")
    structure_module.dynotears = dynotears_module
    causalnex_module = ModuleType("causalnex")
    causalnex_module.structure = structure_module
    monkeypatch.setitem(sys.modules, "causalnex", causalnex_module)
    monkeypatch.setitem(sys.modules, "causalnex.structure", structure_module)
    monkeypatch.setitem(
        sys.modules,
        "causalnex.structure.dynotears",
        dynotears_module,
    )

    S = dynotears_causality_matrix(np.zeros((20, 2)), max_lag=2)

    np.testing.assert_allclose(S, [[0.0, 1.0], [0.5, 0.0]])


@pytest.mark.parametrize("non_finite_value", [np.nan, np.inf, -np.inf])
def test_dynotears_ignores_non_finite_weights_without_erasing_valid_edges(
    monkeypatch,
    non_finite_value,
):
    class FakeGraph:
        def edges(self, data):
            assert data is True
            return [
                ("x0_lag2", "x1_lag0", {"weight": non_finite_value}),
                ("x0_lag1", "x1_lag0", {"weight": 0.5}),
                ("x1_lag1", "x0_lag0", {"weight": 1.0}),
            ]

    dynotears_module = ModuleType("causalnex.structure.dynotears")
    dynotears_module.from_pandas_dynamic = lambda dataframe, p: FakeGraph()
    structure_module = ModuleType("causalnex.structure")
    structure_module.dynotears = dynotears_module
    causalnex_module = ModuleType("causalnex")
    causalnex_module.structure = structure_module
    monkeypatch.setitem(sys.modules, "causalnex", causalnex_module)
    monkeypatch.setitem(sys.modules, "causalnex.structure", structure_module)
    monkeypatch.setitem(
        sys.modules,
        "causalnex.structure.dynotears",
        dynotears_module,
    )

    S = dynotears_causality_matrix(np.zeros((20, 2)), max_lag=2)

    assert np.all(np.isfinite(S))
    np.testing.assert_allclose(S, [[0.0, 1.0], [0.5, 0.0]])


def test_var_lingam_excludes_b0_and_aggregates_lagged_matrices(monkeypatch):
    constructor_kwargs = {}

    class FakeVARLiNGAM:
        def __init__(self, **kwargs):
            constructor_kwargs.update(kwargs)

        def fit(self, X):
            self.adjacency_matrices_ = np.array(
                [
                    [[0.0, 0.0], [100.0, 0.0]],
                    [[0.0, 1.0], [0.5, 0.0]],
                    [[0.0, 0.25], [0.125, 0.0]],
                ]
            )

    lingam_module = ModuleType("lingam")
    lingam_module.VARLiNGAM = FakeVARLiNGAM
    monkeypatch.setitem(sys.modules, "lingam", lingam_module)

    S = var_lingam_causality_matrix(
        np.zeros((20, 2)),
        max_lag=2,
    )

    assert constructor_kwargs == {"lags": 2, "random_state": 0}
    np.testing.assert_allclose(S, [[0.0, 1.0], [0.5, 0.0]])


@pytest.mark.parametrize("non_finite_value", [np.nan, np.inf, -np.inf])
def test_var_lingam_rejects_non_finite_adjacency_output(
    monkeypatch,
    non_finite_value,
):
    class FakeVARLiNGAM:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            self.adjacency_matrices_ = np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [non_finite_value, 0.0]],
                ]
            )

    lingam_module = ModuleType("lingam")
    lingam_module.VARLiNGAM = FakeVARLiNGAM
    monkeypatch.setitem(sys.modules, "lingam", lingam_module)

    with pytest.raises(
        ValueError,
        match="VAR-LiNGAM.*non-finite.*adjacency_matrices_",
    ):
        var_lingam_causality_matrix(np.zeros((20, 2)), max_lag=1)


def test_dynotears_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "causalnex.structure.dynotears", None)

    with pytest.raises(ImportError) as exc_info:
        dynotears_causality_matrix(np.zeros((20, 3)), max_lag=2)

    message = str(exc_info.value)
    assert "causalnex 0.12.1" in message
    assert "separate legacy Python 3.10 environment" in message
    assert "not part of the primary reproducible environment" in message


def test_var_lingam_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "lingam", None)

    with pytest.raises(ImportError, match="lingam"):
        var_lingam_causality_matrix(np.zeros((20, 3)), max_lag=2)


def test_real_data_registry_can_include_modern_baselines():
    methods = build_method_registry(device="cpu", epochs=1, include_modern_baselines=True)

    assert "PCMCI+" in methods
    assert "DYNOTEARS" in methods
    assert "VAR-LiNGAM" in methods
