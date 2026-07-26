import sys
from types import ModuleType

import numpy as np
import pytest
from scipy import stats

from baselines import granger as granger_module
from baselines.granger import granger_causality_matrix
from baselines.pcmci_wrapper import (
    pcmci_causality_matrix,
    pcmci_plus_causality_matrix,
    pcmci_with_details,
)


def _install_fake_tigramite(monkeypatch, val_matrix_override=None):
    dataframe_calls = []
    runner_calls = []
    cond_test_calls = []

    class FakeDataFrame:
        def __init__(self, data, **kwargs):
            dataframe_calls.append((data, kwargs))
            self.data = data

    class FakeParCorr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            cond_test_calls.append(kwargs)

    class FakeGPDC(FakeParCorr):
        pass

    class FakePCMCI:
        def __init__(self, dataframe, cond_ind_test, verbosity):
            self.dataframe = dataframe

        def _results(self, runner):
            runner_calls.append(runner)
            data = self.dataframe.data
            first_segment = next(iter(data.values())) if isinstance(data, dict) else data
            n_variables = first_segment.shape[1]
            if val_matrix_override is None:
                val_matrix = np.zeros((n_variables, n_variables, 3))
                val_matrix[0, 1, 1] = 1.0
            else:
                val_matrix = np.asarray(val_matrix_override).copy()
            return {
                "val_matrix": val_matrix,
                "p_matrix": np.ones_like(val_matrix),
            }

        def run_pcmci(self, **kwargs):
            return self._results("pcmci")

        def run_pcmciplus(self, **kwargs):
            return self._results("pcmci_plus")

    data_processing_module = ModuleType("tigramite.data_processing")
    data_processing_module.DataFrame = FakeDataFrame
    pcmci_module = ModuleType("tigramite.pcmci")
    pcmci_module.PCMCI = FakePCMCI
    parcorr_module = ModuleType("tigramite.independence_tests.parcorr")
    parcorr_module.ParCorr = FakeParCorr
    gpdc_module = ModuleType("tigramite.independence_tests.gpdc")
    gpdc_module.GPDC = FakeGPDC
    independence_tests_module = ModuleType("tigramite.independence_tests")
    tigramite_module = ModuleType("tigramite")
    tigramite_module.data_processing = data_processing_module

    monkeypatch.setitem(sys.modules, "tigramite", tigramite_module)
    monkeypatch.setitem(sys.modules, "tigramite.data_processing", data_processing_module)
    monkeypatch.setitem(sys.modules, "tigramite.pcmci", pcmci_module)
    monkeypatch.setitem(
        sys.modules,
        "tigramite.independence_tests",
        independence_tests_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "tigramite.independence_tests.parcorr",
        parcorr_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "tigramite.independence_tests.gpdc",
        gpdc_module,
    )
    return dataframe_calls, runner_calls, cond_test_calls


@pytest.mark.parametrize(
    ("score_fn", "expected_runner"),
    [
        (pcmci_causality_matrix, "pcmci"),
        (pcmci_plus_causality_matrix, "pcmci_plus"),
    ],
)
def test_segmented_pcmci_uses_exact_multiple_mode_arrays(
    monkeypatch,
    score_fn,
    expected_runner,
):
    dataframe_calls, runner_calls, _ = _install_fake_tigramite(monkeypatch)
    X = np.arange(22, dtype=float).reshape(11, 2)
    original = X.copy()

    score_fn(
        X,
        tau_max=2,
        pc_alpha=0.05,
        segment_lengths=[np.int64(5), 6.0],
    )

    data, kwargs = dataframe_calls[0]
    assert kwargs == {"analysis_mode": "multiple"}
    assert list(data) == [0, 1]
    np.testing.assert_allclose(data[0], X[:5] - X[:5].mean(axis=0))
    np.testing.assert_allclose(data[1], X[5:] - X[5:].mean(axis=0))
    np.testing.assert_array_equal(X, original)
    assert runner_calls == [expected_runner]


def test_unsegmented_pcmci_preserves_single_dataframe_mode(monkeypatch):
    dataframe_calls, _, _ = _install_fake_tigramite(monkeypatch)
    X = np.arange(16, dtype=float).reshape(8, 2)

    pcmci_causality_matrix(X, 2, 0.05, "parcorr", None)

    data, kwargs = dataframe_calls[0]
    assert data is X
    assert kwargs == {}


def test_segmented_pcmci_details_uses_multiple_mode(monkeypatch):
    dataframe_calls, _, _ = _install_fake_tigramite(monkeypatch)
    X = np.arange(22, dtype=float).reshape(11, 2)

    pcmci_with_details(X, tau_max=2, segment_lengths=[5, 6])

    data, kwargs = dataframe_calls[0]
    assert kwargs == {"analysis_mode": "multiple"}
    np.testing.assert_allclose(data[0], X[:5] - X[:5].mean(axis=0))
    np.testing.assert_allclose(data[1], X[5:] - X[5:].mean(axis=0))


@pytest.mark.parametrize(
    "score_fn",
    [pcmci_causality_matrix, pcmci_plus_causality_matrix, pcmci_with_details],
)
def test_pcmci_target_valid_mask_is_attached_as_y_only_invalid_mask(
    monkeypatch,
    score_fn,
):
    dataframe_calls, _, cond_test_calls = _install_fake_tigramite(monkeypatch)
    X = np.arange(22, dtype=float).reshape(11, 2)
    valid = np.ones_like(X, dtype=bool)
    valid[6:9, 0] = False

    score_fn(
        X,
        tau_max=2,
        segment_lengths=[5, 6],
        target_valid_mask=valid,
    )

    data, kwargs = dataframe_calls[0]
    assert kwargs["analysis_mode"] == "multiple"
    assert list(kwargs["mask"]) == [0, 1]
    np.testing.assert_array_equal(kwargs["mask"][0], ~valid[:5])
    np.testing.assert_array_equal(kwargs["mask"][1], ~valid[5:])
    assert cond_test_calls == [{"significance": "analytic", "mask_type": "y"}]
    assert data[1][1, 0] == X[6, 0] - X[5:].mean(axis=0)[0]


@pytest.mark.parametrize(
    ("score_fn", "method_name"),
    [
        (pcmci_causality_matrix, "PCMCI"),
        (pcmci_plus_causality_matrix, "PCMCI\\+"),
        (pcmci_with_details, "PCMCI"),
    ],
)
def test_pcmci_wrappers_reject_all_nan_library_output(
    monkeypatch,
    score_fn,
    method_name,
):
    _install_fake_tigramite(
        monkeypatch,
        val_matrix_override=np.full((2, 2, 3), np.nan),
    )

    with pytest.raises(ValueError, match=rf"{method_name}.*non-finite.*val_matrix"):
        score_fn(np.zeros((12, 2)), tau_max=2)


def _planted_segmented_process(segment_lengths):
    rng = np.random.default_rng(712)
    segments = []
    for length in segment_lengths:
        segment = np.zeros((length, 2))
        segment[:, 0] = rng.normal(size=length)
        segment[0, 1] = rng.normal(scale=0.1)
        segment[1:, 1] = 1.5 * segment[:-1, 0] + rng.normal(scale=0.1, size=length - 1)
        segments.append(segment)
    return np.concatenate(segments)


def test_segmented_pcmci_reports_planted_direction():
    segment_lengths = [80, 90]
    X = _planted_segmented_process(segment_lengths)

    score = pcmci_causality_matrix(
        X,
        tau_max=1,
        pc_alpha=0.05,
        segment_lengths=segment_lengths,
    )

    assert score[1, 0] > score[0, 1]


@pytest.mark.parametrize(
    "segment_lengths",
    [
        [[4, 4]],
        [4, 0, 4],
        [3.5, 4.5],
        [3, 4],
        [2, 6],
    ],
)
@pytest.mark.parametrize(
    "score_fn",
    [
        pcmci_causality_matrix,
        pcmci_plus_causality_matrix,
        pcmci_with_details,
    ],
)
def test_pcmci_rejects_invalid_segment_lengths(score_fn, segment_lengths):
    with pytest.raises(ValueError, match="segment_lengths"):
        score_fn(
            np.zeros((8, 2)),
            tau_max=2,
            segment_lengths=segment_lengths,
        )


def _discontinuous_segmented_process(segment_lengths):
    rng = np.random.default_rng(91)
    segments = []
    for offset, length in zip([0.0, 1_000_000.0], segment_lengths):
        segment = np.zeros((length, 2))
        segment[:, 0] = offset + rng.normal(scale=1_000.0, size=length)
        segment[0, 1] = 1.4 * offset + rng.normal(scale=100.0)
        segment[1:, 1] = 1.4 * segment[:-1, 0] + rng.normal(scale=100.0, size=length - 1)
        segments.append(segment)
    return np.concatenate(segments)


def test_segmented_granger_reports_planted_direction_across_discontinuity():
    segment_lengths = [90, 100]
    max_lag = 2
    X = _discontinuous_segmented_process(segment_lengths)

    score = granger_causality_matrix(
        X,
        max_lag=max_lag,
        score_type="f_stat",
        segment_lengths=segment_lengths,
    )

    assert abs(X[segment_lengths[0], 0] - X[segment_lengths[0] - 1, 0]) > 100_000
    assert np.all(np.isfinite(score))
    assert score[1, 0] > score[0, 1]


def test_segmented_granger_is_invariant_to_large_segment_offsets():
    segment_lengths = [90, 100]
    X = _planted_segmented_process(segment_lengths)
    shifted = X.copy()
    shifted[segment_lengths[0] :] += 1_000_000.0

    score = granger_causality_matrix(
        X,
        max_lag=2,
        score_type="f_stat",
        segment_lengths=segment_lengths,
    )
    shifted_score = granger_causality_matrix(
        shifted,
        max_lag=2,
        score_type="f_stat",
        segment_lengths=segment_lengths,
    )

    np.testing.assert_allclose(shifted_score, score, rtol=1e-9, atol=1e-12)
    assert score[1, 0] > score[0, 1]
    assert shifted_score[1, 0] > shifted_score[0, 1]


def test_segmented_granger_rows_have_fixed_effects_and_exclude_boundaries():
    segment_lengths = [6, 7]
    max_lag = 2
    X = np.arange(sum(segment_lengths) * 2, dtype=float).reshape(-1, 2)
    segments = [X[:6], X[6:]]

    outcome, restricted, unrestricted = granger_module._build_segmented_lag_rows(
        segments,
        target=1,
        source=0,
        lag=max_lag,
    )

    expected_rows = sum(length - max_lag for length in segment_lengths)
    assert len(outcome) == expected_rows
    assert restricted.shape == (expected_rows, len(segment_lengths) + max_lag)
    assert unrestricted.shape == (
        expected_rows,
        len(segment_lengths) + 2 * max_lag,
    )
    assert expected_rows != len(X) - max_lag
    np.testing.assert_array_equal(outcome, np.concatenate([X[2:6, 1], X[8:, 1]]))
    np.testing.assert_array_equal(
        restricted[:, :2],
        np.vstack(
            [
                np.tile([1.0, 0.0], (4, 1)),
                np.tile([0.0, 1.0], (5, 1)),
            ]
        ),
    )
    np.testing.assert_array_equal(
        restricted[:, 2:],
        np.vstack(
            [
                np.column_stack([X[1:5, 1], X[:4, 1]]),
                np.column_stack([X[7:12, 1], X[6:11, 1]]),
            ]
        ),
    )
    np.testing.assert_array_equal(
        unrestricted[:, -2:],
        np.vstack(
            [
                np.column_stack([X[1:5, 0], X[:4, 0]]),
                np.column_stack([X[7:12, 0], X[6:11, 0]]),
            ]
        ),
    )


def test_segmented_granger_masks_only_target_outcomes_and_keeps_source_history():
    X = np.arange(16, dtype=float).reshape(8, 2)
    segment = X.copy()
    valid = np.ones_like(X, dtype=bool)
    valid[4, 0] = False

    outcome_0, _, _ = granger_module._build_segmented_lag_rows(
        [segment],
        target=0,
        source=1,
        lag=1,
        target_valid_masks=[valid],
    )
    outcome_1, _, unrestricted_1 = granger_module._build_segmented_lag_rows(
        [segment],
        target=1,
        source=0,
        lag=1,
        target_valid_masks=[valid],
    )

    assert X[4, 0] not in outcome_0
    np.testing.assert_array_equal(outcome_1, X[1:, 1])
    assert X[4, 0] in unrestricted_1[:, -1]


def test_offset_only_segments_do_not_create_granger_edges():
    alternating = np.tile([1.0, -1.0], 20)
    first = np.column_stack([alternating, np.zeros(40)])
    second = np.column_stack([1_000_000.0 + alternating, np.full(40, 1_000_000.0)])

    score = granger_causality_matrix(
        np.concatenate([first, second]),
        max_lag=2,
        score_type="f_stat",
        segment_lengths=[40, 40],
    )

    np.testing.assert_array_equal(score, np.zeros((2, 2)))


def test_duplicate_source_histories_are_non_estimable():
    rng = np.random.default_rng(2)
    values = rng.normal(size=100)

    score = granger_causality_matrix(
        np.column_stack([values, values]),
        max_lag=2,
        score_type="f_stat",
        segment_lengths=[50, 50],
    )

    np.testing.assert_array_equal(score, np.zeros((2, 2)))


def test_pooled_unit_scaling_preserves_nested_f_stat_and_fixed_effects():
    rng = np.random.default_rng(19)
    segments = [rng.normal(size=(30, 2)), rng.normal(size=(35, 2))]
    outcome, restricted, unrestricted = granger_module._build_segmented_lag_rows(
        segments,
        target=1,
        source=0,
        lag=2,
    )
    raw_result = granger_module._nested_regression_f_test(
        outcome,
        restricted,
        unrestricted,
    )

    scaled = granger_module._scale_pooled_regression_data(
        outcome,
        restricted,
        unrestricted,
        n_segments=2,
    )
    scaled_outcome, scaled_restricted, scaled_unrestricted = scaled
    scaled_result = granger_module._nested_regression_f_test(*scaled)

    assert raw_result is not None
    assert scaled_result is not None
    assert scaled_result[0] == pytest.approx(raw_result[0], rel=1e-10)
    np.testing.assert_array_equal(scaled_restricted[:, :2], restricted[:, :2])
    np.testing.assert_array_equal(scaled_unrestricted[:, :2], unrestricted[:, :2])
    assert np.max(np.abs(scaled_outcome)) == pytest.approx(1.0)
    assert np.all(np.max(np.abs(scaled_restricted[:, 2:]), axis=0) <= 1.0)
    assert np.all(np.max(np.abs(scaled_unrestricted[:, 2:]), axis=0) <= 1.0)


def test_nested_granger_test_uses_effective_rank_degrees_of_freedom(monkeypatch):
    captured_degrees = []

    def fake_sf(f_stat, numerator_df, denominator_df):
        captured_degrees.append((numerator_df, denominator_df))
        return 0.25

    monkeypatch.setattr(stats.f, "sf", fake_sf)
    x = np.arange(8, dtype=float)
    z = np.array([0.0, 1.0, 0.0, -1.0, 1.0, -1.0, 0.5, -0.5])
    outcome = 0.5 * x + 2.0 * z + np.tile([0.1, -0.1], 4)
    restricted = np.column_stack([np.ones(8), x])
    unrestricted = np.column_stack([restricted, z, 2.0 * z])

    result = granger_module._nested_regression_f_test(
        outcome,
        restricted,
        unrestricted,
    )

    assert result is not None
    assert result[0] > 0
    assert result[1] == 0.25
    assert captured_degrees == [(1, 5)]


@pytest.mark.parametrize(
    "segment_lengths",
    [
        [[4, 4]],
        [4, 0, 4],
        [3.5, 4.5],
        [3, 4],
        [2, 6],
    ],
)
def test_granger_rejects_invalid_segment_lengths(segment_lengths):
    with pytest.raises(ValueError, match="segment_lengths"):
        granger_causality_matrix(
            np.zeros((8, 2)),
            max_lag=2,
            segment_lengths=segment_lengths,
        )


@pytest.mark.parametrize(
    "score_fn",
    [
        pcmci_causality_matrix,
        pcmci_plus_causality_matrix,
        pcmci_with_details,
    ],
)
def test_pcmci_rejects_segments_too_short_for_default_cutoff(score_fn):
    with pytest.raises(ValueError, match=r"2 \* tau_max \(4\)"):
        score_fn(
            np.zeros((10, 2)),
            tau_max=2,
            segment_lengths=[4, 6],
        )


def test_segmented_granger_guards_invalid_denominator_degrees_of_freedom():
    score = granger_causality_matrix(
        np.arange(8, dtype=float).reshape(4, 2),
        max_lag=1,
        segment_lengths=[2, 2],
    )

    np.testing.assert_array_equal(score, np.zeros((2, 2)))


def test_unsegmented_granger_api_still_runs():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 2))

    score = granger_causality_matrix(X, 2, 0.05, "neg_log_pvalue", None)

    assert score.shape == (2, 2)
    assert np.all(np.isfinite(score))


def test_unsegmented_granger_raises_context_for_pairwise_failure(monkeypatch):
    from statsmodels.tsa import stattools

    def fail_pair(*args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(stattools, "grangercausalitytests", fail_pair)

    with pytest.raises(
        RuntimeError,
        match=r"Granger failed for target=0, source=1, max_lag=2.*backend exploded",
    ):
        granger_causality_matrix(
            np.zeros((20, 2)),
            max_lag=2,
        )
