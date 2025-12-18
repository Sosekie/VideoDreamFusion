#!/bin/bash
# Download HunyuanVideo-1.5 720p I2V model (fp8)

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./ckpts}"
HF_TOKEN="${1:-}"
ENABLE_SR="${ENABLE_SR:-true}"

echo "=========================================="
echo "Downloading HunyuanVideo-1.5 720p I2V checkpoints"
echo "=========================================="

mkdir -p "$CKPT_DIR"

echo "Downloading core checkpoints from tencent/HunyuanVideo-1.5..."
hf download tencent/HunyuanVideo-1.5 \
  --local-dir "$CKPT_DIR" \
  --include "transformer/720p_i2v/**" \
  --include "scheduler/**" \
  --include "vae/**"

if [[ "$ENABLE_SR" != "false" ]]; then
  echo ""
  echo "Downloading super-resolution checkpoints (required when --sr true)..."
  hf download tencent/HunyuanVideo-1.5 \
    --local-dir "$CKPT_DIR" \
    --include "transformer/1080p_sr_distilled/**" \
    --include "upsampler/1080p_sr_distilled/**"
fi

echo ""
echo "Downloading text encoders..."
echo "  - MLLM (Qwen2.5-VL-7B-Instruct)..."
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$CKPT_DIR/text_encoder/llm"

echo "  - byT5..."
hf download google/byt5-small --local-dir "$CKPT_DIR/text_encoder/byt5-small"

echo "  - Glyph-SDXL-v2..."
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "$CKPT_DIR/text_encoder/Glyph-SDXL-v2"

echo ""
echo "Downloading vision encoder (Siglip)..."
if [[ -z "$HF_TOKEN" ]]; then
  echo "⚠️  Skipping vision encoder download (HF token not provided)."
  echo "    Inference will fail until you download it:"
  echo "      1) Request access: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
  echo "      2) Generate token: https://huggingface.co/settings/tokens"
  echo "      3) Run: bash download_720p_i2v.sh <your_hf_token>"
else
  hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir "$CKPT_DIR/vision_encoder/siglip" \
    --token "$HF_TOKEN"
fi

echo ""
echo "Verifying required directories..."
missing=()
for p in \
  "$CKPT_DIR/transformer/720p_i2v" \
  "$CKPT_DIR/vae" \
  "$CKPT_DIR/scheduler" \
  "$CKPT_DIR/text_encoder/llm" \
; do
  [[ -e "$p" ]] || missing+=("$p")
done

if [[ "$ENABLE_SR" != "false" ]]; then
  for p in \
    "$CKPT_DIR/transformer/1080p_sr_distilled" \
    "$CKPT_DIR/upsampler/1080p_sr_distilled" \
  ; do
    [[ -e "$p" ]] || missing+=("$p")
  done
fi

if [[ ${#missing[@]} -ne 0 ]]; then
  echo "❌ Missing required checkpoint paths:"
  for p in "${missing[@]}"; do
    echo "  - $p"
  done
  echo ""
  echo "You can always download everything with:"
  echo "  hf download tencent/HunyuanVideo-1.5 --local-dir \"$CKPT_DIR\""
  exit 1
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Model location: $CKPT_DIR"
echo ""
echo "To use the 720p I2V model, run:"
echo "  python generate.py --image_path <image.png> --resolution 720p --model_path ./ckpts"
