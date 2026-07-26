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
        "exp27_nonlinearity_confirmation.py",
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
    assert "exp27:" in makefile
    assert "exp27_nonlinearity_confirmation.py" in makefile
    assert "workshop-evidence:" in makefile
    assert "--model-seeds 42 43 44" in makefile
    assert "--methods Lagged-Bottleneck Lasso Ridge PCMCI PCMCI+ Granger" in makefile
    assert "docker-build" not in makefile


def test_reproduction_supports_but_does_not_require_docker():
    readme = Path("README.md").read_text()
    makefile = Path("Makefile").read_text()

    assert Path("Dockerfile").exists()
    assert "docker-build" not in makefile
    assert "make smoke" in readme


def test_full_real_data_targets_download_missing_finance_data():
    makefile = Path("Makefile").read_text()

    assert "data/real/sector_etfs.csv:" in makefile
    assert "uv run --with yfinance python scripts/download_etfs.py" in makefile
    assert "exp24: data/real/sector_etfs.csv" in makefile
    assert "exp26: data/real/sector_etfs.csv" in makefile


def test_exp26_uses_cuda_when_available_by_default():
    script = Path("experiments/exp26_ssm_multiseed.py").read_text()

    assert 'default="cuda" if torch.cuda.is_available() else "cpu"' in script
    assert 'print(f"Device: {args.device}")' in script


def test_package_import_tolerates_notebook_matplotlib_backend():
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

    subprocess.run(
        [sys.executable, "-c", "import ssm_causal.utils"],
        check=True,
        env=env,
    )
