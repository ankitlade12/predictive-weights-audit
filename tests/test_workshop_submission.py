import hashlib
import importlib.util
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
