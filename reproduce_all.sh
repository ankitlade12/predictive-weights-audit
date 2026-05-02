#!/bin/bash
# ============================================================
# Reproduce the F1-F5 falsification pipeline for:
# "Prediction Bottlenecks Don't Discover Causal Structure"
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
echo " SSM-Causal: F1-F5 Falsification Reproduction"
echo "=============================================="

mkdir -p results

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
uv run python experiments/exp24_real_data_benchmark.py

echo ""
echo "[F3] SSM multi-seed real benchmark"
echo "----------------------------------------------"
uv run python experiments/exp26_ssm_multiseed.py

echo ""
echo "[F4] Size-matched intervention control"
echo "----------------------------------------------"
uv run python experiments/exp22_intervention_control.py

echo ""
echo "[F4] Intervention type sweep"
echo "----------------------------------------------"
uv run python experiments/exp23_intervention_sweep.py

echo ""
echo "[F5] Classical causal baselines on intervention protocol"
echo "----------------------------------------------"
uv run python experiments/exp25_causal_baselines.py

echo ""
echo "=============================================="
echo " F1-F5 reproduction complete"
echo " Results saved to results/"
echo "=============================================="
