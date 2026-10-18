"""Summarize paired follow-up results without modifying experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_causal.statistics import paired_bootstrap_ci  # noqa: E402

RESULTS = ROOT / "results/followup"


def summarize(primary, optional):
    """Join by data seed, never treating neural initializations as replicates."""
    expected = set(primary["protocol"]["data_seeds"])
    if primary["status"] != "complete" or optional["status"] != "complete":
        raise ValueError("Both follow-up stages must be complete")
    rows = {r["data_seed"]: r for r in primary["replicates"]}
    extra = {r["data_seed"]: r for r in optional["replicates"]}
    if len(rows) != len(primary["replicates"]) or set(rows) != expected:
        raise ValueError("Missing or duplicate primary data seeds")
    if len(extra) != len(optional["replicates"]) or set(extra) != expected | {0}:
        raise ValueError("Missing or duplicate optional data seeds")
    methods = tuple(rows[min(expected)]["methods"]) + ("DYNOTEARS", "VAR-LiNGAM")
    samples = {method: {metric: [] for metric in ("auroc", "auprc")} for method in methods}
    for seed in sorted(expected):
        if rows[seed]["input_sha256"] != extra[seed]["input_sha256"]:
            raise ValueError(f"Input mismatch for seed {seed}")
        for method in methods:
            result = (rows[seed] if method in rows[seed]["methods"] else extra[seed])["methods"][
                method
            ]
            if result.get("status", "ok") != "ok":
                raise ValueError(f"Failed method {method}, seed {seed}")
            for metric in ("auroc", "auprc"):
                samples[method][metric].append(result[metric])
    summary = {}
    reference = samples["Lagged-Bottleneck"]["auroc"]
    for method in methods:
        summary[method] = {}
        for metric in ("auroc", "auprc", "delta_auroc"):
            first = samples[method]["auroc" if metric == "delta_auroc" else metric]
            second = reference if metric == "delta_auroc" else np.zeros(len(first))
            ci = paired_bootstrap_ci(first, second, n_resamples=10000, seed=82000)
            summary[method][metric] = {"mean": ci.effect, "low": ci.ci_low, "high": ci.ci_high}
    return summary


if __name__ == "__main__":
    primary = json.loads((RESULTS / "trajectory_replication.json").read_text())
    optional = json.loads((RESULTS / "optional_baselines.json").read_text())
    print(json.dumps(summarize(primary, optional), indent=2, allow_nan=False))
