"""Build a clean, checksum-verified manifest for workshop evidence artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "corrected_workshop" / "experiment_manifest.json"

SOURCE_PREFIXES = (
    "baselines/",
    "data/",
    "experiments/",
    "paper/generate_workshop_results.py",
    "scripts/download_etfs.py",
    "scripts/generate_workshop_manifest.py",
    "ssm_causal/",
    "tests/",
)
SOURCE_FILES = {
    "Dockerfile",
    "Makefile",
    "pyproject.toml",
    "reproduce_all.sh",
    "uv.lock",
}

EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "exp20",
        "artifact": (
            "results/corrected_workshop/graph_label_sensitivity_smoke/"
            "fixed_score_graph_label_sensitivity_results.json"
        ),
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "experiments/exp20_distribution_shift.py --quick "
            "--save-dir results/corrected_workshop/graph_label_sensitivity_smoke"
        ),
        "scope": "CPU smoke validation",
        "claim": (
            "The protocol measures fixed-score graph-label sensitivity, not "
            "adaptation or distribution-shift robustness."
        ),
    },
    {
        "id": "exp22",
        "artifact": (
            "results/corrected_workshop/intervention_control/"
            "intervention_control_results.json"
        ),
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "experiments/exp22_intervention_control.py --seeds 10 --K 10 20 "
            "--T 300 --epochs 80 --device cpu "
            "--save-dir results/corrected_workshop/intervention_control"
        ),
        "scope": "10 independent data seeds; target-aware, size-matched random forcing",
        "claim": (
            "The bottleneck has small relative advantages over Lasso in these "
            "two configurations; this contrast does not establish a direct "
            "causal-information benefit."
        ),
    },
    {
        "id": "exp23",
        "artifact": (
            "results/corrected_workshop/intervention_semantics/"
            "intervention_sweep_results.json"
        ),
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "experiments/exp23_intervention_sweep.py --seeds 10 --K 10 20 "
            "--T 300 --int-types random constant soft --strengths 1 "
            "--baselines lasso ridge --epochs 80 --device cpu "
            "--save-dir results/corrected_workshop/intervention_semantics"
        ),
        "scope": "10 independent data seeds; target-aware intervention-semantics comparison",
        "claim": (
            "Constant clamps show no bottleneck relative advantage; selected "
            "random and soft effects are small and do not identify a "
            "bottleneck-specific mechanism."
        ),
    },
    {
        "id": "exp24",
        "artifact": (
            "results/corrected_workshop/real_data/"
            "real_data_benchmark_results.json"
        ),
        "provenance_artifact": (
            "results/corrected_workshop/real_data/edge_provenance_cards.json"
        ),
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "experiments/exp24_real_data_benchmark.py --datasets climate finance "
            "lorenz96 --methods Lagged-Bottleneck Lasso Ridge PCMCI PCMCI+ "
            "Granger --model-seeds 42 43 44 --epochs 80 --device cpu "
            "--save-dir results/corrected_workshop/real_data"
        ),
        "scope": "Climate and finance soft-label sensitivity; exact simulated Lorenz-96 graph",
        "claim": (
            "Lagged bottleneck weights encode readable Lorenz-96 dependency "
            "signal, but Granger and tuned linear baselines recover the exact "
            "graph better in this configuration."
        ),
    },
    {
        "id": "exp25",
        "artifact": (
            "results/corrected_workshop/intervention_methods/"
            "causal_baselines_results.json"
        ),
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "experiments/exp25_causal_baselines.py --seeds 10 --K-values 10 20 "
            "--T-values 300 --epochs 80 --device cpu "
            "--save-dir results/corrected_workshop/intervention_methods"
        ),
        "scope": "10 independent data seeds; target-aware random forcing",
        "claim": (
            "All four tested methods have positive within-method gains. A direct "
            "exploratory Granger-minus-Bottleneck contrast is null at K=10 and "
            "positive at K=20 after Holm correction."
        ),
    },
    {
        "id": "exp27",
        "artifact": "results/corrected_workshop/nonlinearity_confirmation.json",
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "experiments/exp27_nonlinearity_confirmation.py "
            "--output results/corrected_workshop/nonlinearity_confirmation.json"
        ),
        "scope": (
            "10 independent data seeds; K in {10,20}; T in {300,600}; "
            "nonlinearity mix in {0,0.2,0.3,0.4,0.6,1}"
        ),
        "claim": (
            "No common-T mild-nonlinearity regime passes the fixed gate; the "
            "isolated K=20,T=300 effect is invariant to nonlinearity and "
            "disappears at T=600."
        ),
    },
    {
        "id": "paper-values",
        "artifact": "results/corrected_workshop/paper_values.json",
        "command": (
            "uv run --frozen --extra baselines --extra dev python "
            "paper/generate_workshop_results.py"
        ),
        "scope": "Mechanical derivation of manuscript values from corrected artifacts",
        "claim": "The manuscript tables and headline values are artifact-derived.",
    },
)


def _run_git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _tracked_source_paths() -> list[Path]:
    tracked = _run_git("ls-files", "-z").split(b"\0")
    relative_paths = sorted(
        item.decode()
        for item in tracked
        if item
        and (
            item.decode() in SOURCE_FILES
            or item.decode().endswith(".py")
            and item.decode().startswith(SOURCE_PREFIXES)
        )
    )
    return [ROOT / relative_path for relative_path in relative_paths]


def _source_tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode()
        contents = path.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(len(contents)).encode())
        digest.update(b"\0")
        digest.update(contents)
    return digest.hexdigest()


def _tracked_source_diff(paths: list[Path]) -> bytes:
    relative_paths = [path.relative_to(ROOT).as_posix() for path in paths]
    unstaged = _run_git("diff", "--binary", "HEAD", "--", *relative_paths)
    staged = _run_git("diff", "--binary", "--cached", "HEAD", "--", *relative_paths)
    return staged + unstaged


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path.relative_to(ROOT)}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "tigramite",
        "torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest() -> dict[str, Any]:
    source_paths = _tracked_source_paths()
    source_diff = _tracked_source_diff(source_paths)
    revision = _run_git("rev-parse", "HEAD").decode().strip()

    experiments = []
    for definition in EXPERIMENTS:
        record = dict(definition)
        artifact_path = ROOT / record["artifact"]
        record["artifact_sha256"] = _sha256(artifact_path)
        if "provenance_artifact" in record:
            record["provenance_artifact_sha256"] = _sha256(
                ROOT / record["provenance_artifact"]
            )
        experiments.append(record)

    return {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": "local_cpu",
        "code_revision": revision,
        "tracked_source_worktree_dirty": bool(source_diff),
        "tracked_source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "tracked_source_tree_sha256": _source_tree_sha256(source_paths),
        "tracked_source_file_count": len(source_paths),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "replication_policy": {
            "unit": (
                "independent data-generating seed unless an artifact explicitly "
                "states otherwise"
            ),
            "model_seed_spread": "descriptive initialization stability only",
            "multiplicity": "Holm correction within each declared primary family",
            "uncertainty": "paired bootstrap intervals for paired effects",
        },
        "experiments": experiments,
        "not_rerun": [
            {
                "method": "SSM real-data benchmark",
                "reason": (
                    "Outside the corrected CPU evidence slice; the workshop "
                    "manuscript does not make a corrected empirical claim for it."
                ),
            },
            {
                "method": "DYNOTEARS",
                "reason": "Optional causalnex dependency is unavailable.",
            },
            {
                "method": "VAR-LiNGAM",
                "reason": "Optional lingam dependency is unavailable.",
            },
        ],
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    if manifest["tracked_source_worktree_dirty"]:
        raise SystemExit("Refusing a release manifest with dirty tracked source files.")


if __name__ == "__main__":
    main()
