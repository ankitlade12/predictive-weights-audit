#!/bin/bash
# ============================================================
# Reproduce the F1-F6 predictive-bottleneck audit pipeline.
# ============================================================
#
# Prerequisites:
#   uv sync
#   uv sync --extra baselines --extra dev
#
# Usage:
#   ./reproduce_all.sh
#
# Results are saved under results/.
# ============================================================

set -euo pipefail

echo "=============================================="
echo " SSM-Causal: F1-F6 Audit Reproduction"
echo "=============================================="

mkdir -p results

echo ""
echo "[Audit] Fixed-score graph-label sensitivity smoke"
echo "----------------------------------------------"
uv run python experiments/exp20_distribution_shift.py --quick \
    --save-dir results/corrected_workshop/graph_label_sensitivity_smoke

echo ""
echo "[F1] Architecture comparison"
echo "----------------------------------------------"
uv run python experiments/exp7_architecture_comparison.py

echo ""
echo "[F2] Lagged bottleneck stress grid"
echo "----------------------------------------------"
uv run python experiments/exp15_lagged_bottleneck_stress_grid.py

echo ""
echo "[F2] CauseMe lag gate"
echo "----------------------------------------------"
uv run python experiments/exp17_causeme_lag_gate.py

echo ""
echo "[F3] Real-data benchmark"
echo "----------------------------------------------"
uv run python experiments/exp24_real_data_benchmark.py \
    --datasets climate finance lorenz96 \
    --methods Lagged-Bottleneck Lasso Ridge PCMCI PCMCI+ Granger \
    --model-seeds 42 43 44 --epochs 80 --device cpu \
    --save-dir results/corrected_workshop/real_data

echo ""
echo "[F3] SSM multi-seed real benchmark"
echo "----------------------------------------------"
uv run python experiments/exp26_ssm_multiseed.py

echo ""
echo "[F4] Size-matched intervention control"
echo "----------------------------------------------"
uv run python experiments/exp22_intervention_control.py \
    --seeds 10 --K 10 20 --T 300 --epochs 80 --device cpu \
    --save-dir results/corrected_workshop/intervention_control

echo ""
echo "[F4] Intervention type sweep"
echo "----------------------------------------------"
uv run python experiments/exp23_intervention_sweep.py \
    --seeds 10 --K 10 20 --T 300 \
    --int-types random constant soft --strengths 1 \
    --baselines lasso ridge --epochs 80 --device cpu \
    --save-dir results/corrected_workshop/intervention_semantics

echo ""
echo "[F5] Classical causal baselines on intervention protocol"
echo "----------------------------------------------"
uv run python experiments/exp25_causal_baselines.py \
    --seeds 10 --K-values 10 20 --T-values 300 --epochs 80 --device cpu \
    --save-dir results/corrected_workshop/intervention_methods

echo ""
echo "[F6] Confirmatory nonlinearity sweep"
echo "----------------------------------------------"
uv run python experiments/exp27_nonlinearity_confirmation.py \
    --output results/corrected_workshop/nonlinearity_confirmation.json

echo ""
echo "=============================================="
echo " F1-F6 reproduction complete"
echo " Results saved to results/"
echo "=============================================="
