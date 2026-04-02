#!/bin/bash
# ============================================================
# Upload SSM-Causal project to Google Drive for Colab
# ============================================================
#
# This creates a zip of the project and opens Google Drive
# for manual upload. After upload, the Colab notebook can
# access the files.
#
# Usage:
#   chmod +x scripts/upload_to_drive.sh
#   ./scripts/upload_to_drive.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_FILE="/tmp/ssm_causal_project.zip"

echo "=============================================="
echo " Packaging SSM-Causal for Google Colab"
echo "=============================================="

cd "$PROJECT_DIR"

# Create zip excluding unnecessary files
zip -r "$ZIP_FILE" \
    ssm_causal/ \
    baselines/ \
    data/ \
    experiments/ \
    paper/generate_tables.py \
    -x "**/__pycache__/*" "**/.venv/*" "**/results/*" "**/*.pyc"

echo ""
echo "Created: $ZIP_FILE ($(du -h "$ZIP_FILE" | cut -f1))"
echo ""
echo "Next steps:"
echo "  1. Go to: https://drive.google.com"
echo "  2. Upload $ZIP_FILE to 'My Drive'"
echo "  3. In Colab, mount Drive and unzip:"
echo "     !unzip /content/drive/MyDrive/ssm_causal_project.zip -d /content/"
echo ""
echo "Or just open the notebook and it will guide you."
echo ""

# Open Google Drive in browser
open "https://drive.google.com" 2>/dev/null || xdg-open "https://drive.google.com" 2>/dev/null || true
