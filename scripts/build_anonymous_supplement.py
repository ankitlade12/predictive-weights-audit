"""Build and audit a deterministic, identity-free workshop supplement ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

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
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and _is_included(path.relative_to(ROOT).as_posix())
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
    contents = contents.replace(
        "submission-audit: test paper-workshop workshop-manifest anonymous-supplement",
        "submission-audit: test paper-workshop workshop-manifest",
    )
    contents = contents.replace(
        "\nanonymous-supplement:\n\t$(UV_RUN) python scripts/build_anonymous_supplement.py\n",
        "",
    )
    contents = contents.replace(
        " scripts/build_anonymous_supplement.py",
        "",
    )
    contents = contents.replace(
        '\t@echo "  make anonymous-supplement - Build identity-scanned review ZIP"\n',
        "",
    )
    contents = contents.replace(" anonymous-supplement", "")
    return contents


def _patch_anonymous_readme(contents: str) -> str:
    return contents.replace(
        "Build the deterministic, identity-scanned review supplement:\n\n"
        "```bash\n"
        "make anonymous-supplement\n"
        "```\n",
        "This directory is already the deterministic, identity-scanned review supplement.\n",
    )


def _remove_code_revisions(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_code_revisions(item)
            for key, item in value.items()
            if key != "code_revision"
        }
    if isinstance(value, list):
        return [_remove_code_revisions(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _sanitize_anonymous_contents(contents: dict[str, bytes]) -> dict[str, bytes]:
    sanitized = dict(contents)
    result_prefix = "results/corrected_workshop/"
    for relative_path, raw_contents in list(sanitized.items()):
        if relative_path.startswith(result_prefix) and relative_path.endswith(".json"):
            sanitized[relative_path] = _json_bytes(_remove_code_revisions(json.loads(raw_contents)))

    paper_values_path = "results/corrected_workshop/paper_values.json"
    paper_values = json.loads(sanitized[paper_values_path])
    for artifact_path in paper_values.get("source_artifacts", {}):
        paper_values["source_artifacts"][artifact_path] = hashlib.sha256(
            sanitized[artifact_path]
        ).hexdigest()
    sanitized[paper_values_path] = _json_bytes(paper_values)

    manifest_path = "results/corrected_workshop/experiment_manifest.json"
    manifest = json.loads(sanitized[manifest_path])
    manifest["provenance_mode"] = "anonymous_snapshot_filesystem"
    for record in manifest["experiments"]:
        record["artifact_sha256"] = hashlib.sha256(sanitized[record["artifact"]]).hexdigest()
        provenance_path = record.get("provenance_artifact")
        if provenance_path is not None:
            record["provenance_artifact_sha256"] = hashlib.sha256(
                sanitized[provenance_path]
            ).hexdigest()
    sanitized[manifest_path] = _json_bytes(manifest)
    return sanitized


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
    contents_by_path = {
        relative_path: (ROOT / relative_path).read_bytes() for relative_path in included
    }
    contents_by_path["Makefile"] = _patch_anonymous_makefile(
        contents_by_path["Makefile"].decode()
    ).encode()
    contents_by_path["README.md"] = _patch_anonymous_readme(
        contents_by_path["README.md"].decode()
    ).encode()
    contents_by_path = _sanitize_anonymous_contents(contents_by_path)

    identity_findings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="anonymous-supplement-"):
        with zipfile.ZipFile(output, "w") as archive:
            for relative_path in included:
                contents = contents_by_path[relative_path]
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
    display_path = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"Wrote {display_path}")
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
