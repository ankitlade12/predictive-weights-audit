"""Release checks for paired post-review evidence and generated tables."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reviewer_tables", ROOT / "paper/generate_reviewer_results.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def artifacts():
    directory = ROOT / "results/reviewer_followup"
    return tuple(
        json.loads((directory / name).read_text())
        for name in ("trajectory_replication.json", "optional_baselines.json")
    )


def test_saved_followup_has_ten_paired_trajectories_and_zero_self_contrast():
    primary, optional = artifacts()
    summary = generator.summarize(primary, optional)
    assert len(primary["replicates"]) == 10
    assert len(summary) == 8
    assert summary["Lagged-Bottleneck"]["delta_auroc"] == {"mean": 0, "low": 0, "high": 0}
    reordered = copy.deepcopy(optional)
    reordered["replicates"].reverse()
    assert generator.summarize(primary, reordered) == summary


@pytest.mark.parametrize("damage", ["duplicate", "missing", "mismatch", "failure"])
def test_followup_refuses_unpaired_or_failed_results(damage):
    primary, optional = artifacts()
    if damage == "duplicate":
        optional["replicates"].append(optional["replicates"][-1])
    elif damage == "missing":
        optional["replicates"].pop()
    elif damage == "mismatch":
        optional["replicates"][-1]["input_sha256"] = "different-data"
    else:
        optional["replicates"][-1]["methods"]["DYNOTEARS"]["status"] = "failed"
    with pytest.raises(ValueError):
        generator.summarize(primary, optional)
