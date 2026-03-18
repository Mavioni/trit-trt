#!/usr/bin/env bash
# TRIT-TRT Setup Script
# Yunis AI — Sovereign Ternary Inference
# ──────────────────────────────────────
# This script sets up the full TRIT-TRT environment:
# 1. Creates conda environment
# 2. Installs Python dependencies
# 3. Clones and builds BitNet
# 4. Downloads the default model
#
# Usage:
#   chmod +x scripts/setup_model.sh
#   ./scripts/setup_model.sh [MODEL_ID]

set -euo pipefail

export PYTHONIOENCODING=utf-8

MODEL_ID="${1:-microsoft/BitNet-b1.58-2B-4T}"
QUANT_TYPE="${2:-i2_s}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "════════════════════════════════════════════════════════"
echo "  TRIT-TRT Setup — Yunis AI"
echo "  Model: $MODEL_ID"
echo "  Quant: $QUANT_TYPE"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Python environment ──────────────────────────────
echo ""
echo "[1/4] Setting up Python environment..."

if command -v conda &>/dev/null; then
    if ! conda env list | grep -q "trit-trt"; then
        echo "  Creating conda env: trit-trt (Python 3.10)"
        conda create -n trit-trt python=3.10 -y
    fi
    echo "  Activating trit-trt environment"
    eval "$(conda shell.bash hook)"
    conda activate trit-trt
else
    echo "  conda not found — using system Python"
    echo "  (Recommended: install miniconda for isolation)"
fi

# ── Step 2: Install dependencies ────────────────────────────
echo ""
echo "[2/4] Installing Python dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
pip install -e "$PROJECT_DIR" --quiet
echo "  Done."

# ── Step 3: Clone and build BitNet ──────────────────────────
echo ""
echo "[3/4] Setting up BitNet..."

BITNET_DIR="$HOME/BitNet"
if [ -d "$BITNET_DIR" ]; then
    echo "  BitNet already cloned at $BITNET_DIR"
else
    echo "  Cloning microsoft/BitNet..."
    git clone --recursive https://github.com/microsoft/BitNet.git "$BITNET_DIR"
fi

# Install BitNet requirements
pip install -r "$BITNET_DIR/requirements.txt" --quiet

# ── Step 4: Download and quantize model ─────────────────────
echo ""
echo "[4/4] Downloading model: $MODEL_ID..."

MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')
MODEL_DIR="$BITNET_DIR/models/$MODEL_SHORT"

# Download GGUF version if available
GGUF_REPO="${MODEL_ID}-gguf"
if hf repo info "$GGUF_REPO" &>/dev/null 2>&1; then
    echo "  Found GGUF repo: $GGUF_REPO"
    hf download "$GGUF_REPO" --local-dir "$MODEL_DIR"
else
    echo "  No GGUF repo found, downloading safetensors and converting..."
    hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

# Build and quantize
echo "  Building bitnet.cpp and quantizing to $QUANT_TYPE..."
cd "$BITNET_DIR"
python setup_env.py -md "$MODEL_DIR" -q "$QUANT_TYPE"

# ── Done ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Quick test:"
echo "    python -c \"from trit_trt import TritTRT; print('Ready!')\""
echo ""
echo "  Run inference:"
echo "    python -c \""
echo "      from trit_trt import TritTRT"
echo "      engine = TritTRT('$MODEL_ID')"
echo "      result = engine.generate('Hello, who are you?')"
echo "      print(result.text)"
echo "    \""
echo ""
echo "  Run benchmark:"
echo "    python scripts/benchmark.py --model $MODEL_ID"
echo "════════════════════════════════════════════════════════"
