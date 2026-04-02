#!/bin/bash
# ============================================================
# Full AWS GPU Pipeline: Launch → Run → Download → Terminate
# ============================================================
#
# Prerequisites:
#   1. AWS CLI configured: aws configure
#   2. Key pair created (or will be created automatically)
#
# Usage:
#   chmod +x scripts/run_on_aws.sh
#   ./scripts/run_on_aws.sh
#
# This script will:
#   1. Create a key pair (if needed)
#   2. Create a security group (if needed)
#   3. Launch a g4dn.xlarge GPU instance
#   4. Upload the code
#   5. Run ALL experiments
#   6. Download results
#   7. TERMINATE the instance (no lingering bills)
#
# Estimated: ~$5-8 total (~2-3 GPU hours on g4dn.xlarge)
# ============================================================

set -e

# ── Configuration ──
INSTANCE_TYPE="g4dn.xlarge"     # T4 16GB, $0.53/hr (cheapest GPU)
REGION="us-east-1"              # Change if needed
KEY_NAME="ssm-causal-key"
SG_NAME="ssm-causal-sg"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_LOCAL="$PROJECT_DIR/results"

echo "=============================================="
echo " SSM-Causal: AWS GPU Auto-Pilot"
echo "=============================================="
echo " Instance: $INSTANCE_TYPE"
echo " Region:   $REGION"
echo " Project:  $PROJECT_DIR"
echo "=============================================="

# ── Step 1: Key pair ──
echo ""
echo "[1/7] Setting up SSH key pair..."
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
if [ ! -f "$KEY_FILE" ]; then
    # Delete existing key pair on AWS if any
    aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION" 2>/dev/null || true
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "  Created: $KEY_FILE"
else
    echo "  Using existing: $KEY_FILE"
fi

# ── Step 2: Security group ──
echo ""
echo "[2/7] Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "SSM-Causal experiments" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region "$REGION"
    echo "  Created: $SG_ID"
else
    echo "  Using existing: $SG_ID"
fi

# ── Step 3: Find AMI ──
echo ""
echo "[3/7] Finding Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text 2>/dev/null)

if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
    # Fallback: Ubuntu 22.04 with CUDA
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)
    echo "  Using Ubuntu 22.04: $AMI_ID (will need manual CUDA setup)"
else
    echo "  Using Deep Learning AMI: $AMI_ID"
fi

# ── Step 4: Launch instance ──
echo ""
echo "[4/7] Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ssm-causal-gpu}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance: $INSTANCE_ID"
echo "  Waiting for instance to start..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "  IP: $PUBLIC_IP"
echo "  Waiting for SSH to become available..."
sleep 30

# Wait for SSH
for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" "echo ok" 2>/dev/null; then
        break
    fi
    echo "  Retrying SSH... ($i/30)"
    sleep 10
done

# ── Step 5: Upload code and setup ──
echo ""
echo "[5/7] Uploading code and installing dependencies..."

# Create tar of the project (excluding .venv, __pycache__, results)
cd "$PROJECT_DIR"
tar czf /tmp/ssm_causal.tar.gz \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='results' \
    --exclude='*.pyc' \
    .

scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    /tmp/ssm_causal.tar.gz ubuntu@"$PUBLIC_IP":~/

ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" << 'REMOTE_SETUP'
set -e
mkdir -p ~/ssm_causal && cd ~/ssm_causal
tar xzf ~/ssm_causal.tar.gz

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
cd ~/ssm_causal
uv sync
uv sync --extra baselines --extra dev

# Verify GPU
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
REMOTE_SETUP

echo "  Setup complete!"

# ── Step 6: Run ALL experiments ──
echo ""
echo "[6/7] Running ALL experiments on GPU..."
echo "  This will take ~30-60 minutes on GPU"

ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" << 'REMOTE_RUN'
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/ssm_causal

echo "Starting experiments at $(date)"

# Run all experiments
chmod +x reproduce_all.sh
./reproduce_all.sh 2>&1 | tee ~/experiment_log.txt

# Also run exp4 (fMRI) which isn't in reproduce_all yet
uv run python experiments/exp4_fmri.py 2>&1 | tee -a ~/experiment_log.txt

# Generate tables
uv run python paper/generate_tables.py 2>&1 | tee -a ~/experiment_log.txt

echo ""
echo "All experiments complete at $(date)"
REMOTE_RUN

# ── Step 7: Download results and TERMINATE ──
echo ""
echo "[7/7] Downloading results and terminating instance..."

# Download results
mkdir -p "$RESULTS_LOCAL"
scp -o StrictHostKeyChecking=no -r -i "$KEY_FILE" \
    ubuntu@"$PUBLIC_IP":~/ssm_causal/results/ "$RESULTS_LOCAL/../results_gpu/"

# Download generated tables
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    ubuntu@"$PUBLIC_IP":~/ssm_causal/paper/table*.tex "$PROJECT_DIR/paper/" 2>/dev/null || true

# Download log
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    ubuntu@"$PUBLIC_IP":~/experiment_log.txt "$PROJECT_DIR/" 2>/dev/null || true

echo "  Results downloaded to: $RESULTS_LOCAL/../results_gpu/"

# TERMINATE the instance
echo "  Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" > /dev/null

echo ""
echo "=============================================="
echo " DONE!"
echo " Instance $INSTANCE_ID has been TERMINATED."
echo " No further charges will be incurred."
echo ""
echo " Results: $RESULTS_LOCAL/../results_gpu/"
echo " Tables:  $PROJECT_DIR/paper/table*.tex"
echo " Log:     $PROJECT_DIR/experiment_log.txt"
echo "=============================================="
