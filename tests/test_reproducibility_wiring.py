import os
import subprocess
import sys
from pathlib import Path


def test_reproduction_stages_exist_and_optional_stage_follows_primary():
    script = Path("reproduce_all.sh").read_text()
    makefile = Path("Makefile").read_text()
    targets = ["test", "workshop-evidence", "followup-evidence", "followup-optional", "release"]
    positions = [script.index(f"make {target}") for target in targets]
    assert positions == sorted(positions)
    for target in targets:
        result = subprocess.run(["make", "-n", target], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert f"{target}:" in makefile
    for line in makefile.splitlines():
        for word in line.split():
            if word.startswith(("experiments/", "scripts/", "paper/")) and word.endswith(".py"):
                assert Path(word).is_file(), word


def test_package_import_tolerates_notebook_matplotlib_backend():
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    subprocess.run([sys.executable, "-c", "import ssm_causal.utils"], check=True, env=env)
