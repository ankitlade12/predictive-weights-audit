"""Package the accepted revision and its evidence, excluding local workspace clutter."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = {
        ROOT / name
        for name in (
            ".gitignore",
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
            "Dockerfile",
            "reproduce_all.sh",
            "docs/CAMERA_READY_REVISION.md",
            "paper/neurips_2026.sty",
            "paper/references.bib",
            "paper/camera_ready_references.bib",
            "paper/generate_workshop_results.py",
            "paper/generate_reviewer_results.py",
            "paper/draft/README.md",
            "paper/draft/main_4page.tex",
            "paper/draft/main_camera_ready.tex",
            "paper/draft/main_arxiv.tex",
            "paper/draft/camera_ready_content.tex",
            "paper/draft/camera_ready_appendix.tex",
            "paper/draft/generated_workshop_results.tex",
            "paper/draft/generated_reviewer_results.tex",
            "paper/draft/workshop_evidence_summary.pdf",
            "paper/draft/main_camera_ready.pdf",
            "paper/draft/main_arxiv.pdf",
        )
    }
    # Only immediate Python sources: the workspace also contains an unrelated nested checkout.
    for directory in ("baselines", "data", "experiments", "ssm_causal", "tests", "scripts"):
        files.update((ROOT / directory).glob("*.py"))
    files.update((ROOT / "data/real").glob("*.data"))
    files.add(ROOT / "data/real/sector_etfs.csv")
    for directory in ("results/corrected_workshop", "results/reviewer_followup"):
        files.update(p for p in (ROOT / directory).rglob("*") if p.is_file())
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    checksums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(contents.items())
    )
    contents["SHA256SUMS"] = checksums.encode()
    output = ROOT / "dist/camera_ready_release.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, data in contents.items():
            assert archive.read(name) == data
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Wrote {output} ({len(contents)} files); SHA-256 {digest}")


if __name__ == "__main__":
    build()
