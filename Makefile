# SSM-Causal Makefile
# ====================

.PHONY: install test experiments tables paper clean all

# ── Setup ──
install:
	uv sync
	uv sync --extra baselines --extra dev

# ── Tests ──
test:
	uv run pytest tests/ -v

# ── Experiments ──
exp0:
	uv run python experiments/exp0_core_validation.py

exp1:
	uv run python experiments/exp1_synthetic.py

exp2:
	uv run python experiments/exp2_nonstationary.py

exp3:
	uv run python experiments/exp3_climate.py

exp4:
	uv run python experiments/exp4_fmri.py

exp5:
	uv run python experiments/exp5_efficiency.py

exp6:
	uv run python experiments/exp6_ablation.py

experiments: exp0 exp1 exp2 exp3 exp4 exp5 exp6

# ── Paper ──
tables:
	uv run python paper/generate_tables.py

figures:
	uv run python paper/generate_figures.py

paper: tables figures
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# ── Full pipeline ──
all: install test experiments tables figures

# ── Cleanup ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	cd paper && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz 2>/dev/null || true
