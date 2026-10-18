"""Package code and experimental evidence using an explicit file allowlist."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FILES = (
    ".dockerignore",
    ".gitignore",
    "CITATION.cff",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README.md",
    "baselines/__init__.py",
    "baselines/dynotears.py",
    "baselines/granger.py",
    "baselines/lingam_wrapper.py",
    "baselines/neural_granger.py",
    "baselines/pcmci_wrapper.py",
    "data/__init__.py",
    "data/real/amon.us.data",
    "data/real/nao.data",
    "data/real/nina34.data",
    "data/real/pdo.data",
    "data/real/pna.data",
    "data/real/sector_etfs.csv",
    "data/real/soi.data",
    "data/real_loaders.py",
    "data/synthetic.py",
    "experiments/__init__.py",
    "experiments/exp18_mechanistic_ablation.py",
    "experiments/exp20_distribution_shift.py",
    "experiments/exp22_intervention_control.py",
    "experiments/exp23_intervention_sweep.py",
    "experiments/exp24_real_data_benchmark.py",
    "experiments/exp25_causal_baselines.py",
    "experiments/exp27_nonlinearity_confirmation.py",
    "experiments/exp28_followup.py",
    "pyproject.toml",
    "reproduce_all.sh",
    "results/corrected_workshop/experiment_manifest.json",
    "results/corrected_workshop/graph_label_sensitivity_smoke/fixed_score_graph_label_sensitivity_results.json",
    "results/corrected_workshop/intervention_control/intervention_control_results.json",
    "results/corrected_workshop/intervention_methods/causal_baselines_results.json",
    "results/corrected_workshop/intervention_semantics/intervention_sweep_results.json",
    "results/corrected_workshop/nonlinearity_confirmation.json",
    "results/corrected_workshop/paper_values.json",
    "results/corrected_workshop/real_data/edge_provenance_cards.json",
    "results/corrected_workshop/real_data/real_data_benchmark_results.json",
    "results/followup/DYNOTEARS_seed0_scores.npy",
    "results/followup/DYNOTEARS_seed81000_scores.npy",
    "results/followup/DYNOTEARS_seed81001_scores.npy",
    "results/followup/DYNOTEARS_seed81002_scores.npy",
    "results/followup/DYNOTEARS_seed81003_scores.npy",
    "results/followup/DYNOTEARS_seed81004_scores.npy",
    "results/followup/DYNOTEARS_seed81005_scores.npy",
    "results/followup/DYNOTEARS_seed81006_scores.npy",
    "results/followup/DYNOTEARS_seed81007_scores.npy",
    "results/followup/DYNOTEARS_seed81008_scores.npy",
    "results/followup/DYNOTEARS_seed81009_scores.npy",
    "results/followup/VAR-LiNGAM_seed0_scores.npy",
    "results/followup/VAR-LiNGAM_seed81000_scores.npy",
    "results/followup/VAR-LiNGAM_seed81001_scores.npy",
    "results/followup/VAR-LiNGAM_seed81002_scores.npy",
    "results/followup/VAR-LiNGAM_seed81003_scores.npy",
    "results/followup/VAR-LiNGAM_seed81004_scores.npy",
    "results/followup/VAR-LiNGAM_seed81005_scores.npy",
    "results/followup/VAR-LiNGAM_seed81006_scores.npy",
    "results/followup/VAR-LiNGAM_seed81007_scores.npy",
    "results/followup/VAR-LiNGAM_seed81008_scores.npy",
    "results/followup/VAR-LiNGAM_seed81009_scores.npy",
    "results/followup/lasso_diagnostic.json",
    "results/followup/legacy-requirements.txt",
    "results/followup/lorenz96_seed0.npz",
    "results/followup/lorenz96_seed81000.npz",
    "results/followup/lorenz96_seed81001.npz",
    "results/followup/lorenz96_seed81002.npz",
    "results/followup/lorenz96_seed81003.npz",
    "results/followup/lorenz96_seed81004.npz",
    "results/followup/lorenz96_seed81005.npz",
    "results/followup/lorenz96_seed81006.npz",
    "results/followup/lorenz96_seed81007.npz",
    "results/followup/lorenz96_seed81008.npz",
    "results/followup/lorenz96_seed81009.npz",
    "results/followup/optional_baselines.json",
    "results/followup/paper_values.json",
    "results/followup/trajectory_replication.json",
    "scripts/build_release.py",
    "scripts/summarize_results.py",
    "ssm_causal/__init__.py",
    "ssm_causal/benchmark_metadata.py",
    "ssm_causal/bottleneck_training.py",
    "ssm_causal/causeme_benchmarks.py",
    "ssm_causal/extractor.py",
    "ssm_causal/improved_score.py",
    "ssm_causal/intervention_protocol.py",
    "ssm_causal/lag_bottleneck.py",
    "ssm_causal/lag_extraction.py",
    "ssm_causal/lag_synthetic.py",
    "ssm_causal/mamba_model.py",
    "ssm_causal/score.py",
    "ssm_causal/statistics.py",
    "ssm_causal/threshold.py",
    "ssm_causal/utils.py",
    "tests/test_benchmark_metadata.py",
    "tests/test_bottleneck_training.py",
    "tests/test_causeme_benchmarks.py",
    "tests/test_core.py",
    "tests/test_intervention_protocol.py",
    "tests/test_lag_bottleneck.py",
    "tests/test_modern_baselines.py",
    "tests/test_nonlinearity_confirmation.py",
    "tests/test_real_data_benchmark.py",
    "tests/test_reproducibility_wiring.py",
    "tests/test_followup.py",
    "tests/test_segmented_causal_baselines.py",
    "tests/test_statistical_protocol.py",
    "tests/test_release.py",
    "uv.lock",
)


def build(output: Path | None = None):
    files = [ROOT / name for name in FILES]
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    checksums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(contents.items())
    )
    contents["SHA256SUMS"] = checksums.encode()
    output = output or ROOT / "dist/predictive_weights_audit.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name == "reproduce_all.sh" else 0o644) << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, data in contents.items():
            assert archive.read(name) == data
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Wrote {output} ({len(contents)} files); SHA-256 {digest}")
    return output


if __name__ == "__main__":
    build()
