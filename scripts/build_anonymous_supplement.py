"""Build and audit a deterministic, identity-free workshop supplement ZIP."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "anonymous_workshop_supplement.zip"

TOP_LEVEL_FILES = {
    ".gitignore",
    "Dockerfile",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "reproduce_all.sh",
    "uv.lock",
}
INCLUDED_PREFIXES = (
    "baselines/",
    "data/",
    "experiments/",
    "results/corrected_workshop/",
    "ssm_causal/",
    "tests/",
)
INCLUDED_FILES = {
    "paper/draft/generated_workshop_results.tex",
    "paper/draft/main_4page.tex",
    "paper/generate_workshop_results.py",
    "paper/neurips_2026.sty",
    "paper/references.bib",
    "scripts/download_etfs.py",
    "scripts/generate_workshop_manifest.py",
}
EXCLUDED_FILES = {
    # This local release test intentionally contains identity-denial fixtures.
    "tests/test_workshop_submission.py",
}
DENIED_IDENTITY_TOKENS = (
    "ankit hemant",
    "ankitlade12",
    "sai krishna jasti",
    "sai-krishna99",
    "jsaikrishna379",
    "indar kumar",
    "indarkarhana",
    "aman chadha",
    "hi@aman.ai",
    "@gmail.com",
    "google deepmind",
)
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".cfg",
    ".csv",
    ".data",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

ANONYMOUS_LICENSE = """MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

ANONYMITY_NOTE = """# Anonymous Workshop Supplement

This snapshot contains the code, tests, corrected result artifacts, data
loaders, and anonymous manuscript required to inspect and reproduce the
submission. It intentionally excludes Git history, author metadata,
acknowledgments, non-anonymous manuscripts, and identity-bearing project
files. The public repository should not be linked during double-blind review.
"""


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return sorted(item.decode() for item in result.stdout.split(b"\0") if item)


def _is_included(relative_path: str) -> bool:
    if relative_path in EXCLUDED_FILES:
        return False
    return (
        relative_path in TOP_LEVEL_FILES
        or relative_path in INCLUDED_FILES
        or relative_path.startswith(INCLUDED_PREFIXES)
    )


def _patch_anonymous_makefile(contents: str) -> str:
    contents = contents.replace(
        "paper: paper-arxiv",
        "paper: paper-workshop",
    )
    contents = contents.replace(
        "all: install test falsification paper-arxiv",
        "all: install test workshop-evidence paper-workshop",
    )
    contents = contents.replace(
        "all: install test workshop-evidence paper-workshop anonymous-supplement",
        "all: install test workshop-evidence paper-workshop",
    )
    return contents


def _scan_identity(relative_path: str, contents: bytes) -> list[str]:
    if Path(relative_path).suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        decoded = contents.decode("utf-8")
    except UnicodeDecodeError:
        return []
    lowered = decoded.lower()
    return [token for token in DENIED_IDENTITY_TOKENS if token in lowered]


def _zip_info(relative_path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative_path)
    info.date_time = (2026, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build(output: Path) -> str:
    included = [path for path in _tracked_files() if _is_included(path)]
    required = TOP_LEVEL_FILES | INCLUDED_FILES
    missing = sorted(required.difference(included))
    if missing:
        raise FileNotFoundError(f"Required tracked supplement files are missing: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    identity_findings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="anonymous-supplement-"):
        with zipfile.ZipFile(output, "w") as archive:
            for relative_path in included:
                contents = (ROOT / relative_path).read_bytes()
                if relative_path == "Makefile":
                    contents = _patch_anonymous_makefile(contents.decode()).encode()
                hits = _scan_identity(relative_path, contents)
                if hits:
                    identity_findings.append(f"{relative_path}: {', '.join(hits)}")
                archive.writestr(_zip_info(relative_path), contents)

            for relative_path, text in (
                ("LICENSE", ANONYMOUS_LICENSE),
                ("ANONYMITY.md", ANONYMITY_NOTE),
            ):
                archive.writestr(_zip_info(relative_path), text.encode())

    if identity_findings:
        output.unlink(missing_ok=True)
        joined = "\n".join(identity_findings)
        raise RuntimeError(f"Identity audit failed:\n{joined}")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"SHA256 {digest}")
    print(f"Files {len(included) + 2}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
