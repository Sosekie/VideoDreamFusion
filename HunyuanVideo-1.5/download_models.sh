#!/bin/bash
# Download script for HunyuanVideo-1.5 models
# Usage: bash download_models.sh [optional: hf_token]

set -e

CKPT_DIR="./ckpts"
HF_TOKEN="${1:-}"

echo "=========================================="
echo "HunyuanVideo-1.5 Model Download"
echo "=========================================="

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

# Step 1: Download main DiT and VAE checkpoints
echo ""
echo "[1/3] Downloading main DiT and VAE checkpoints..."
echo "This may take a while (~50GB)..."
hf download tencent/HunyuanVideo-1.5 --local-dir "$CKPT_DIR"

# Step 2: Download text encoders
echo ""
echo "[2/3] Downloading text encoders..."
echo "  - Downloading Qwen2.5-VL-7B-Instruct (MLLM)..."
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$CKPT_DIR/text_encoder/llm"

echo "  - Downloading byT5 encoder..."
hf download google/byt5-small --local-dir "$CKPT_DIR/text_encoder/byt5-small"

echo "  - Downloading Glyph-SDXL-v2..."
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "$CKPT_DIR/text_encoder/Glyph-SDXL-v2"

# Step 3: Download vision encoder (requires HF token and access permission)
echo ""
echo "[3/3] Downloading vision encoder (Siglip)..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not provided"
    echo "To download vision encoder, you need to:"
    echo "  1. Request access to: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
    echo "  2. Generate a Hugging Face access token"
    echo "  3. Run: bash download_models.sh <your_hf_token>"
    echo ""
    echo "For now, skipping vision encoder download..."
else
    echo "  - Downloading FLUX.1-Redux-dev (contains Siglip)..."
    hf download black-forest-labs/FLUX.1-Redux-dev --local-dir "$CKPT_DIR/vision_encoder/siglip" --token "$HF_TOKEN"
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Model directory structure:"
tree -L 2 "$CKPT_DIR" 2>/dev/null || find "$CKPT_DIR" -maxdepth 2 -type d | sort

echo ""
echo "Next steps:"
echo "1. If you skipped the vision encoder, request access and download later"
echo "2. Run: conda activate HunyuanVideoThreestudio"
echo "3. Test inference: python generate.py --help"
