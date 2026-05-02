# SSM-Causal Makefile

UV_CACHE_DIR ?= /tmp/uv-cache
MPLCONFIGDIR ?= /tmp/matplotlib-cache
MPLBACKEND := Agg
export UV_CACHE_DIR
export MPLCONFIGDIR
export MPLBACKEND

.PHONY: help install test smoke exp7 exp15 exp17 exp22 exp23 exp24 exp25 exp26 falsification experiments tables figures paper paper-arxiv paper-workshop all clean

help:
	@echo "Targets:"
	@echo "  install          Install core, baseline, and dev dependencies with uv"
	@echo "  test             Run the test suite"
	@echo "  smoke            Run bounded CPU-friendly empirical verification"
	@echo "  falsification    Run the current F1-F5 paper experiment pipeline"
	@echo "  experiments      Alias for falsification"
	@echo "  paper-arxiv      Compile paper/draft/main_arxiv.tex"
	@echo "  paper-workshop   Compile paper/draft/main_4page.tex"
	@echo "  paper            Alias for paper-arxiv"

install:
	uv sync
	uv sync --extra baselines --extra dev

test:
	uv run pytest tests/ -v

smoke:
	uv run pytest tests/test_benchmark_metadata.py tests/test_modern_baselines.py tests/test_reproducibility_wiring.py -q
	uv run python experiments/exp15_lagged_bottleneck_stress_grid.py --quick --save-dir /tmp/ssm-causal-exp15-quick
	uv run python experiments/exp17_causeme_lag_gate.py --quick --save-dir /tmp/ssm-causal-exp17-quick
	uv run python experiments/exp22_intervention_control.py --quick --save-dir /tmp/ssm-causal-exp22-quick
	uv run python experiments/exp23_intervention_sweep.py --quick --save-dir /tmp/ssm-causal-exp23-quick
	uv run python experiments/exp24_real_data_benchmark.py --datasets climate --epochs 1 --modern-baselines --save-dir /tmp/ssm-causal-exp24-smoke
	uv run python experiments/exp25_causal_baselines.py --quick --save-dir /tmp/ssm-causal-exp25-quick
	uv run python experiments/exp26_ssm_multiseed.py --datasets climate --seeds 1 --epochs 1 --save-dir /tmp/ssm-causal-exp26-smoke

exp7:
	uv run python experiments/exp7_architecture_comparison.py

exp15:
	uv run python experiments/exp15_lagged_bottleneck_stress_grid.py

exp17:
	uv run python experiments/exp17_causeme_lag_gate.py

exp22:
	uv run python experiments/exp22_intervention_control.py

exp23:
	uv run python experiments/exp23_intervention_sweep.py

exp24:
	uv run python experiments/exp24_real_data_benchmark.py

exp25:
	uv run python experiments/exp25_causal_baselines.py

exp26:
	uv run python experiments/exp26_ssm_multiseed.py

falsification: exp7 exp15 exp17 exp22 exp23 exp24 exp25 exp26

experiments: falsification

tables:
	uv run python paper/generate_tables.py

figures:
	uv run python paper/generate_figures.py

paper-arxiv:
	cd paper/draft && pdflatex main_arxiv.tex && bibtex main_arxiv && pdflatex main_arxiv.tex && pdflatex main_arxiv.tex

paper-workshop:
	cd paper/draft && pdflatex main_4page.tex && bibtex main_4page && pdflatex main_4page.tex && pdflatex main_4page.tex

paper: paper-arxiv

all: install test falsification paper-arxiv

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	cd paper/draft && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz 2>/dev/null || true
