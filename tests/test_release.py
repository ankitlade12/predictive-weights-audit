"""Evidence provenance and code-only release isolation checks."""

import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_followup_source_and_original_artifact_hashes_remain_valid():
    for name in ("trajectory_replication", "optional_baselines", "lasso_diagnostic"):
        payload = json.loads((ROOT / f"results/reviewer_followup/{name}.json").read_text())
        for relative, digest in payload["source_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    manifest = json.loads(
        (ROOT / "results/corrected_workshop/experiment_manifest.json").read_text()
    )
    for entry in manifest["experiments"]:
        assert (
            hashlib.sha256((ROOT / entry["artifact"]).read_bytes()).hexdigest()
            == entry["artifact_sha256"]
        )


def test_release_is_allowlisted_checksummed_and_builds_without_git(tmp_path, monkeypatch):
    builder = _load_script("code_release", "scripts/build_release.py")
    clean = tmp_path / "snapshot"
    for relative in builder.FILES:
        source = ROOT / relative
        destination = clean / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # Clutter in otherwise permitted directories must never leak into the release.
    for relative in (
        "experiments/local_credentials.py",
        "results/reviewer_followup/private.txt",
        "ssm_causal/paper/old_draft.tex",
        ".git/config",
    ):
        path = clean / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local-only")
    monkeypatch.setattr(builder, "ROOT", clean)
    output = builder.build(tmp_path / "release.zip")
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(builder.FILES) | {"SHA256SUMS"}
        for line in archive.read("SHA256SUMS").decode().splitlines():
            digest, relative = line.split("  ", 1)
            assert hashlib.sha256(archive.read(relative)).hexdigest() == digest
        assert not any(
            Path(name).suffix in {".tex", ".bib", ".sty", ".pdf"} or name.startswith("paper/")
            for name in archive.namelist()
        )
        assert (archive.getinfo("reproduce_all.sh").external_attr >> 16) & 0o111
    assert (
        output.with_suffix(".zip.sha256").read_text().split()[0]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
