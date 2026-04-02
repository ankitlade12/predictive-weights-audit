#!/bin/bash
# ============================================================
# AWS GPU Instance Setup for SSM-Causal Experiments
# ============================================================
#
# STEP 1: Launch an EC2 instance
#   - Recommended: g5.xlarge (A10G 24GB, ~$1.01/hr on-demand)
#   - Alternative: g4dn.xlarge (T4 16GB, ~$0.53/hr, slower)
#   - AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
#   - Storage: 50GB gp3
#
# STEP 2: SSH into the instance and run this script:
#   chmod +x scripts/aws_setup.sh && ./scripts/aws_setup.sh
#
# STEP 3: Run experiments:
#   ./reproduce_all.sh
#
# Estimated cost: $10-15 total (~10 GPU hours)
# ============================================================

set -e

echo "=============================================="
echo " SSM-Causal: AWS GPU Setup"
echo "=============================================="

# ── System packages ──
echo "[1/5] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git curl

# ── Install uv ──
echo "[2/5] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Clone repo (or use existing) ──
echo "[3/5] Setting up project..."
if [ ! -d "ssm_causal" ]; then
    echo "  No project found. Please clone your repo or scp the code."
    echo "  Example: scp -r ssm_causal/ ubuntu@<your-ip>:~/"
    exit 1
fi

cd ssm_causal

# ── Install dependencies ──
echo "[4/5] Installing Python dependencies..."
uv sync
uv sync --extra baselines --extra dev

# ── Verify GPU ──
echo "[5/5] Verifying GPU..."
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected! Falling back to CPU.')
"

echo ""
echo "=============================================="
echo " Setup complete!"
echo ""
echo " Run experiments:"
echo "   ./reproduce_all.sh"
echo ""
echo " Or run individual experiments:"
echo "   uv run python experiments/exp0_core_validation.py"
echo "   uv run python experiments/exp6_ablation.py"
echo "=============================================="
