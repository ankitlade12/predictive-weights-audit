# SSM-Causal Makefile

UV_CACHE_DIR ?= /tmp/uv-cache
MPLCONFIGDIR ?= /tmp/matplotlib-cache
MPLBACKEND := Agg
export UV_CACHE_DIR
export MPLCONFIGDIR
export MPLBACKEND

UV_RUN := uv run --frozen --extra baselines --extra dev

.PHONY: help install test smoke workshop-evidence workshop-tables workshop-manifest anonymous-supplement submission-audit exp7 exp15 exp17 exp22 exp23 exp24 exp25 exp26 exp27 falsification experiments tables figures paper paper-arxiv paper-workshop all clean

help:
	@echo "Targets:"
	@echo "  install          Install core, baseline, and dev dependencies with uv"
	@echo "  test             Run the test suite"
	@echo "  smoke            Run bounded CPU-friendly empirical verification"
	@echo "  workshop-evidence Reproduce the corrected CPU workshop artifacts"
	@echo "  falsification    Run the current F1-F6 paper experiment pipeline"
	@echo "  exp27            Run the nonlinear-regime confirmation"
	@echo "  experiments      Alias for falsification"
	@echo "  paper-arxiv      Compile paper/draft/main_arxiv.tex"
	@echo "  paper-workshop   Compile paper/draft/main_4page.tex"
	@echo "  anonymous-supplement Build and audit the double-blind code supplement"
	@echo "  submission-audit Run tests, paper build, manifest, and anonymity checks"
	@echo "  paper            Alias for paper-arxiv"

install:
	uv sync --frozen --extra baselines --extra dev

data/real/sector_etfs.csv:
	uv run --frozen --with yfinance python scripts/download_etfs.py

test:
	$(UV_RUN) pytest tests/ -v

smoke:
	$(UV_RUN) pytest tests/test_benchmark_metadata.py tests/test_modern_baselines.py tests/test_reproducibility_wiring.py -q
	$(UV_RUN) python experiments/exp15_lagged_bottleneck_stress_grid.py --quick --save-dir /tmp/ssm-causal-exp15-quick
	$(UV_RUN) python experiments/exp17_causeme_lag_gate.py --quick --save-dir /tmp/ssm-causal-exp17-quick
	$(UV_RUN) python experiments/exp22_intervention_control.py --quick --save-dir /tmp/ssm-causal-exp22-quick
	$(UV_RUN) python experiments/exp23_intervention_sweep.py --quick --save-dir /tmp/ssm-causal-exp23-quick
	$(UV_RUN) python experiments/exp24_real_data_benchmark.py --datasets climate --epochs 1 --modern-baselines --save-dir /tmp/ssm-causal-exp24-smoke
	$(UV_RUN) python experiments/exp25_causal_baselines.py --quick --save-dir /tmp/ssm-causal-exp25-quick
	$(UV_RUN) python experiments/exp26_ssm_multiseed.py --datasets climate --seeds 1 --epochs 1 --save-dir /tmp/ssm-causal-exp26-smoke
	$(UV_RUN) python experiments/exp27_nonlinearity_confirmation.py --k-values 10 --t-values 150 --strengths 0.3 --n-seeds 2 --epochs 2 --bootstrap-resamples 100 --output /tmp/ssm-causal-exp27-smoke.json

workshop-evidence: data/real/sector_etfs.csv
	$(UV_RUN) python experiments/exp20_distribution_shift.py --quick --save-dir results/corrected_workshop/graph_label_sensitivity_smoke
	$(UV_RUN) python experiments/exp22_intervention_control.py --seeds 10 --K 10 20 --T 300 --epochs 80 --device cpu --save-dir results/corrected_workshop/intervention_control
	$(UV_RUN) python experiments/exp23_intervention_sweep.py --seeds 10 --K 10 20 --T 300 --int-types random constant soft --strengths 1 --baselines lasso ridge --epochs 80 --device cpu --save-dir results/corrected_workshop/intervention_semantics
	$(UV_RUN) python experiments/exp24_real_data_benchmark.py --datasets climate finance lorenz96 --methods Lagged-Bottleneck Lasso Ridge PCMCI PCMCI+ Granger --model-seeds 42 43 44 --epochs 80 --device cpu --save-dir results/corrected_workshop/real_data
	$(UV_RUN) python experiments/exp25_causal_baselines.py --seeds 10 --K-values 10 20 --T-values 300 --epochs 80 --device cpu --save-dir results/corrected_workshop/intervention_methods
	$(UV_RUN) python experiments/exp27_nonlinearity_confirmation.py --output results/corrected_workshop/nonlinearity_confirmation.json
	$(MAKE) workshop-tables
	$(MAKE) workshop-manifest

workshop-tables:
	$(UV_RUN) python paper/generate_workshop_results.py

workshop-manifest:
	$(UV_RUN) python scripts/generate_workshop_manifest.py

anonymous-supplement:
	$(UV_RUN) python scripts/build_anonymous_supplement.py

submission-audit: test paper-workshop workshop-manifest anonymous-supplement
	$(UV_RUN) ruff check paper/generate_workshop_results.py scripts/generate_workshop_manifest.py scripts/build_anonymous_supplement.py

exp7:
	$(UV_RUN) python experiments/exp7_architecture_comparison.py

exp15:
	$(UV_RUN) python experiments/exp15_lagged_bottleneck_stress_grid.py

exp17:
	$(UV_RUN) python experiments/exp17_causeme_lag_gate.py

exp22:
	$(UV_RUN) python experiments/exp22_intervention_control.py

exp23:
	$(UV_RUN) python experiments/exp23_intervention_sweep.py

exp24: data/real/sector_etfs.csv
	$(UV_RUN) python experiments/exp24_real_data_benchmark.py

exp25:
	$(UV_RUN) python experiments/exp25_causal_baselines.py

exp26: data/real/sector_etfs.csv
	$(UV_RUN) python experiments/exp26_ssm_multiseed.py

exp27:
	$(UV_RUN) python experiments/exp27_nonlinearity_confirmation.py

falsification: exp7 exp15 exp17 exp22 exp23 exp24 exp25 exp26 exp27

experiments: falsification

tables:
	$(UV_RUN) python paper/generate_tables.py

figures:
	$(UV_RUN) python paper/generate_figures.py

paper-arxiv:
	cd paper/draft && TEXINPUTS=..: pdflatex main_arxiv.tex && bibtex main_arxiv && TEXINPUTS=..: pdflatex main_arxiv.tex && TEXINPUTS=..: pdflatex main_arxiv.tex

paper-workshop:
	$(MAKE) workshop-tables
	cd paper/draft && TEXINPUTS=..: pdflatex main_4page.tex && bibtex main_4page && TEXINPUTS=..: pdflatex main_4page.tex && TEXINPUTS=..: pdflatex main_4page.tex

paper: paper-arxiv

all: install test workshop-evidence paper-workshop anonymous-supplement

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	cd paper/draft && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz 2>/dev/null || true
