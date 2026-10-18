# Code and reproducible experimental evidence.
UV_CACHE_DIR ?= /tmp/uv-cache
MPLCONFIGDIR ?= /tmp/matplotlib-cache
MPLBACKEND := Agg
export UV_CACHE_DIR MPLCONFIGDIR MPLBACKEND
UV_RUN := uv run --frozen --extra baselines --extra dev
.DEFAULT_GOAL := help

.PHONY: summary help install test smoke workshop-evidence reviewer-evidence reviewer-optional release all

help:
	@echo "Targets: install test smoke workshop-evidence reviewer-evidence reviewer-optional summary release"

install:
	uv sync --frozen --extra baselines --extra dev

test:
	$(UV_RUN) pytest tests/ -q

smoke:
	$(UV_RUN) pytest tests/test_reviewer_followup.py tests/test_statistical_protocol.py tests/test_nonlinearity_confirmation.py tests/test_modern_baselines.py tests/test_release.py -q

workshop-evidence:
	$(UV_RUN) python experiments/exp20_distribution_shift.py --quick --save-dir results/corrected_workshop/graph_label_sensitivity_smoke
	$(UV_RUN) python experiments/exp22_intervention_control.py --seeds 10 --K 10 20 --T 300 --epochs 80 --device cpu --save-dir results/corrected_workshop/intervention_control
	$(UV_RUN) python experiments/exp23_intervention_sweep.py --seeds 10 --K 10 20 --T 300 --int-types random constant soft --strengths 1 --baselines lasso ridge --epochs 80 --device cpu --save-dir results/corrected_workshop/intervention_semantics
	$(UV_RUN) python experiments/exp24_real_data_benchmark.py --datasets climate finance lorenz96 --methods Lagged-Bottleneck Lasso Ridge PCMCI PCMCI+ Granger --model-seeds 42 43 44 --epochs 80 --device cpu --save-dir results/corrected_workshop/real_data
	$(UV_RUN) python experiments/exp25_causal_baselines.py --seeds 10 --K-values 10 20 --T-values 300 --epochs 80 --device cpu --save-dir results/corrected_workshop/intervention_methods
	$(UV_RUN) python experiments/exp27_nonlinearity_confirmation.py --output results/corrected_workshop/nonlinearity_confirmation.json


reviewer-evidence:
	$(UV_RUN) python experiments/exp28_reviewer_followup.py primary
	$(UV_RUN) python experiments/exp28_reviewer_followup.py diagnose

# Run after reviewer-evidence; reads the exact exported trajectory arrays.
reviewer-optional:
	uv run --no-project --python 3.10.19 --with-requirements results/reviewer_followup/legacy-requirements.txt python experiments/exp28_reviewer_followup.py optional

summary:
	$(UV_RUN) python scripts/summarize_results.py

release:
	$(UV_RUN) python scripts/build_release.py

all: test release
