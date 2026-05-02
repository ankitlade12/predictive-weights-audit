import os
import subprocess
import sys
from pathlib import Path


def test_reproduce_all_runs_current_falsification_pipeline():
    script = Path("reproduce_all.sh").read_text()

    for exp in (
        "exp7_architecture_comparison.py",
        "exp15_lagged_bottleneck_stress_grid.py",
        "exp22_intervention_control.py",
        "exp23_intervention_sweep.py",
        "exp24_real_data_benchmark.py",
        "exp25_causal_baselines.py",
        "exp26_ssm_multiseed.py",
    ):
        assert exp in script

    assert "exp3_climate.py" not in script
    assert "exp4_fmri.py" not in script


def test_makefile_exposes_help_and_paper_pipeline_targets():
    makefile = Path("Makefile").read_text()

    assert "help:" in makefile
    assert "smoke:" in makefile
    assert "MPLBACKEND := Agg" in makefile
    assert "falsification:" in makefile
    assert "paper-arxiv:" in makefile
    assert "main_arxiv.tex" in makefile
    assert "docker-build" not in makefile


def test_reproduction_does_not_require_docker():
    readme = Path("README.md").read_text()

    assert not Path("Dockerfile").exists()
    assert "docker build" not in readme
    assert "make smoke" in readme


def test_package_import_tolerates_notebook_matplotlib_backend():
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

    subprocess.run(
        [sys.executable, "-c", "import ssm_causal.utils"],
        check=True,
        env=env,
    )
