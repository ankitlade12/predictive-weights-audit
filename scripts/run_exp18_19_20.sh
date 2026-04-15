#!/bin/bash
# Run experiments 18, 19, 20 on RunPod GPU
# Paste this into the RunPod Web Terminal

set -e

echo "=== Setting up repo ==="
cd /workspace
git clone https://github.com/ankitlade12/ssm-causal.git 2>/dev/null || (cd ssm-causal && git pull)
cd ssm-causal

echo "=== Installing dependencies ==="
pip install -q torch numpy scipy scikit-learn matplotlib statsmodels networkx pandas seaborn

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "  EXPERIMENT 18: Mechanistic Ablation"
echo "=========================================="
python experiments/exp18_mechanistic_ablation.py --device cuda

echo ""
echo "=========================================="
echo "  EXPERIMENT 19: Intervention"
echo "=========================================="
python experiments/exp19_intervention.py --device cuda

echo ""
echo "=========================================="
echo "  EXPERIMENT 20: Distribution Shift"
echo "=========================================="
python experiments/exp20_distribution_shift.py --device cuda

echo ""
echo "=========================================="
echo "  ALL DONE — Results saved to:"
echo "=========================================="
ls -lh results/mechanistic_ablation/*.json 2>/dev/null
ls -lh results/intervention/*.json 2>/dev/null
ls -lh results/distribution_shift/*.json 2>/dev/null

echo ""
echo "To download results:"
echo "  cat results/mechanistic_ablation/mechanistic_ablation_results.json"
echo "  cat results/intervention/intervention_results.json"
echo "  cat results/distribution_shift/distribution_shift_results.json"
