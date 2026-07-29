import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_NEURIPS_2026_SHA256 = (
    "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a"
)


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workshop_uses_official_double_blind_style():
    style = ROOT / "paper" / "neurips_2026.sty"
    manuscript = (ROOT / "paper" / "draft" / "main_4page.tex").read_text()

    assert hashlib.sha256(style.read_bytes()).hexdigest() == OFFICIAL_NEURIPS_2026_SHA256
    assert r"\usepackage[dblblindworkshop]{neurips_2026}" in manuscript
    assert r"\workshoptitle{Interpretability for Discovery}" in manuscript
    assert r"\bibliographystyle{plainnat}" in manuscript
    assert "Ankit Hemant" not in manuscript
    assert "@gmail.com" not in manuscript


def test_generated_values_cover_every_number_repeated_in_prose():
    generator = _load_script(
        "generate_workshop_results",
        "paper/generate_workshop_results.py",
    )
    rendered = generator.render_tex(generator.build_values())

    for macro in (
        "CorrectedPCMCI",
        "CorrectedLaggedBNMean",
        "CorrectedLaggedBNStd",
        "CorrectedGranger",
        "CorrectedLasso",
        "CorrectedRidge",
        "WithinMethodHolmP",
        "SemanticsPositiveRange",
        "GrangerBNKTenP",
        "GrangerBNKTwentyDelta",
        "GrangerBNKTwentyP",
        "NonlinearityKTenDelta",
        "NonlinearityKTwentyDelta",
        "NonlinearityTSixHundredAUROC",
    ):
        assert rf"\newcommand{{\{macro}}}" in rendered


def test_anonymous_makefile_has_self_contained_default_workflow():
    builder = _load_script(
        "build_anonymous_supplement",
        "scripts/build_anonymous_supplement.py",
    )
    patched = builder._patch_anonymous_makefile((ROOT / "Makefile").read_text())

    assert "all: install test workshop-evidence paper-workshop\n" in patched
    assert (
        "all: install test workshop-evidence paper-workshop anonymous-supplement"
        not in patched
    )


def test_lockfile_is_present_and_docker_uses_frozen_extras():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert (ROOT / "uv.lock").is_file()
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "uv sync --frozen --extra baselines --extra dev" in dockerfile


def test_anonymous_supplement_supports_output_outside_repository(tmp_path):
    builder = _load_script(
        "build_anonymous_supplement_external_output",
        "scripts/build_anonymous_supplement.py",
    )
    output = tmp_path / "anonymous_workshop_supplement.zip"

    digest = builder.build(output)

    assert output.is_file()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest


def test_anonymous_supplement_scrubs_revisions_and_rehashes_artifacts(tmp_path):
    builder = _load_script(
        "build_anonymous_supplement_sanitized",
        "scripts/build_anonymous_supplement.py",
    )
    output = tmp_path / "anonymous_workshop_supplement.zip"
    builder.build(output)

    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("results/corrected_workshop/experiment_manifest.json"))
        real_data_bytes = archive.read(
            "results/corrected_workshop/real_data/real_data_benchmark_results.json"
        )
        real_data = json.loads(real_data_bytes)
        paper_values_bytes = archive.read("results/corrected_workshop/paper_values.json")
        paper_values = json.loads(paper_values_bytes)

    assert "code_revision" not in manifest
    assert "code_revision" not in real_data
    exp24 = next(record for record in manifest["experiments"] if record["id"] == "exp24")
    paper_record = next(
        record for record in manifest["experiments"] if record["id"] == "paper-values"
    )
    assert exp24["artifact_sha256"] == hashlib.sha256(real_data_bytes).hexdigest()
    assert paper_record["artifact_sha256"] == hashlib.sha256(paper_values_bytes).hexdigest()
    assert (
        paper_values["source_artifacts"][
            "results/corrected_workshop/real_data/real_data_benchmark_results.json"
        ]
        == hashlib.sha256(real_data_bytes).hexdigest()
    )


def test_anonymous_supplement_only_advertises_available_commands(tmp_path):
    builder = _load_script(
        "build_anonymous_supplement_commands",
        "scripts/build_anonymous_supplement.py",
    )
    output = tmp_path / "anonymous_workshop_supplement.zip"
    builder.build(output)

    with zipfile.ZipFile(output) as archive:
        makefile = archive.read("Makefile").decode()
        readme = archive.read("README.md").decode()

    assert "anonymous-supplement:" not in makefile
    assert "scripts/build_anonymous_supplement.py" not in makefile
    assert "make anonymous-supplement" not in readme
    assert "submission-audit: test paper-workshop workshop-manifest" in makefile


def test_extracted_supplement_regenerates_manifest_without_git(tmp_path):
    builder = _load_script(
        "build_anonymous_supplement_git_free",
        "scripts/build_anonymous_supplement.py",
    )
    output = tmp_path / "anonymous_workshop_supplement.zip"
    extracted = tmp_path / "extracted"
    builder.build(output)
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extracted)

    result = subprocess.run(
        [sys.executable, "scripts/generate_workshop_manifest.py"],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (extracted / "results/corrected_workshop/experiment_manifest.json").read_text()
    )
    assert manifest["code_revision"] is None
    assert manifest["provenance_mode"] == "anonymous_snapshot_filesystem"


def test_archive_builder_enumerates_files_without_git(monkeypatch):
    builder = _load_script(
        "build_anonymous_supplement_no_git",
        "scripts/build_anonymous_supplement.py",
    )

    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(builder.subprocess, "run", fail_git)

    files = builder._tracked_files()

    assert "Makefile" in files
    assert "results/corrected_workshop/experiment_manifest.json" in files
    assert "tests/test_workshop_submission.py" not in files
