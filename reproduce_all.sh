#!/bin/bash
# ============================================================
# Reproduce All Experiments
# Paper: "Causal Discovery via State Space Model Internals"
# ============================================================
#
# Prerequisites:
#   uv sync
#   uv sync --extra baselines   # for PCMCI
#
# Usage:
#   chmod +x reproduce_all.sh
#   ./reproduce_all.sh
#
# Results are saved to results/ directory.
# ============================================================

set -e

echo "=============================================="
echo " SSM-Causal: Full Experiment Reproduction"
echo "=============================================="

# Create results directory
mkdir -p results

# Experiment 0: Core validation (MUST PASS)
echo ""
echo "[Exp 0] Core Validation — Day 3 Test"
echo "----------------------------------------------"
uv run python experiments/exp0_core_validation.py
echo ""

# Experiment 1: Full synthetic benchmark (Table 1)
echo "[Exp 1] Synthetic Benchmark — Table 1"
echo "----------------------------------------------"
uv run python experiments/exp1_synthetic.py
echo ""

# Experiment 2: Non-stationary data (Table 2 + Figure 1)
echo "[Exp 2] Regime-Switching — Table 2 + Figure 1"
echo "----------------------------------------------"
uv run python experiments/exp2_nonstationary.py
echo ""

# Experiment 3: Climate teleconnections (Figure 2 + Table 3)
echo "[Exp 3] Climate Teleconnections — Figure 2"
echo "----------------------------------------------"
uv run python experiments/exp3_climate.py
echo ""

# Experiment 4: fMRI brain connectivity (Figure 3 + Table 3)
echo "[Exp 4] fMRI Brain Connectivity — Figure 3"
echo "----------------------------------------------"
uv run python experiments/exp4_fmri.py
echo ""

# Experiment 5: Efficiency & scalability (Table 4 + Figure 4)
echo "[Exp 5] Efficiency & Scalability — Table 4"
echo "----------------------------------------------"
uv run python experiments/exp5_efficiency.py
echo ""

# Experiment 6: Ablation study (Table 5)
echo "[Exp 6] Ablation Study — Table 5"
echo "----------------------------------------------"
uv run python experiments/exp6_ablation.py
echo ""

# Experiment 7: Architecture comparison (SSM vs MLP vs LSTM)
echo "[Exp 7] Architecture Comparison — Table 6"
echo "----------------------------------------------"
uv run python experiments/exp7_architecture_comparison.py
echo ""

# Experiment 8: Extended real-world (Finance + EEG)
echo "[Exp 8] Extended Real-World — Table 3 (extended)"
echo "----------------------------------------------"
uv run python experiments/exp8_realworld_extended.py
echo ""

echo "=============================================="
echo " All experiments complete!"
echo " Results saved to results/"
echo "=============================================="
