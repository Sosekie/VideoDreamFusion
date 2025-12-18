#!/bin/bash
# Simple inference script for HunyuanVideo-1.5 I2V

# Configuration
IMAGE_PATH="${1:-dog.png}"
PROMPT="${2:-a 3d model of dog, 360 degree turntable rotation}"
OUTPUT_PATH="${3:-./outputs/output_i2v_$(echo $PROMPT | tr ' ' '_').mp4}"
MODEL_PATH="./ckpts"
RESOLUTION="480p"
NUM_STEPS=12  # Use 8 or 12 for step-distilled model (recommended: 12)
SEED=42
ENABLE_STEP_DISTILL=true  # Enable step-distilled model for faster inference

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Set to 1 to force PyTorch SDPA instead of Flash Attention (slower but more stable)
# Useful if you encounter segmentation faults with flash attention
export FORCE_TORCH_ATTN="${FORCE_TORCH_ATTN:-0}"

# Set to 1 to force Flash Attention 2 instead of Flash Attention 3 (beta)
# Flash Attention 3 may have compatibility issues on some GPUs
export FORCE_FLASH2=1

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     HunyuanVideo-1.5 Image-to-Video Generation            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Error: Image not found at $IMAGE_PATH"
    echo "   Current directory: $(pwd)"
    echo "   Looking for: $(cd $(dirname $IMAGE_PATH) 2>/dev/null && pwd)/$(basename $IMAGE_PATH)"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path not found at $MODEL_PATH"
    echo "   Please download models first:"
    echo "   bash download_720p_i2v.sh"
    exit 1
fi

# Check required subfolders for 480p I2V
missing=()
for p in \
    "$MODEL_PATH/transformer/480p_i2v" \
    "$MODEL_PATH/scheduler" \
    "$MODEL_PATH/vae" \
    "$MODEL_PATH/text_encoder/llm" \
    "$MODEL_PATH/vision_encoder/siglip" \
; do
    [ -e "$p" ] || missing+=("$p")
done

if [ ${#missing[@]} -ne 0 ]; then
    echo "❌ Error: Missing required checkpoint paths:"
    for p in "${missing[@]}"; do
        echo "   - $p"
    done
    echo ""
    echo "Download 480p I2V model:"
    echo "  hf download tencent/HunyuanVideo-1.5 --include 'transformer/480p_i2v/*' --local-dir ./ckpts"
    echo ""
    echo "Or download everything (recommended):"
    echo "  bash download_models.sh <your_hf_token>"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "📋 Configuration:"
echo "  Image:        $(realpath $IMAGE_PATH)"
echo "  Prompt:       $PROMPT"
echo "  Resolution:   $RESOLUTION"
echo "  Steps:        $NUM_STEPS"
echo "  Step Distill: $ENABLE_STEP_DISTILL"
echo "  Output:       $(realpath $OUTPUT_PATH)"
echo "  Force Torch Attn: $FORCE_TORCH_ATTN"
echo "  Force Flash2:     $FORCE_FLASH2"
echo ""

# Run generation with minimal flags for stability
echo "🎬 Starting video generation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python generate.py \
    --image_path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --resolution "$RESOLUTION" \
    --num_inference_steps $NUM_STEPS \
    --seed $SEED \
    --output_path "$OUTPUT_PATH" \
    --enable_step_distill $ENABLE_STEP_DISTILL \
    --sr false \
    --rewrite false
EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ "${EXIT_CODE:-}" == "0" ]]; then
    echo "✓ Video generation completed successfully!"
    echo "✓ Output saved to: $(realpath $OUTPUT_PATH)"
    echo ""
    echo "Video details:"
    if command -v ffprobe &> /dev/null; then
        ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=p=0 "$OUTPUT_PATH" | while IFS=',' read -r width height duration; do
            echo "  Resolution: ${width}x${height}"
            echo "  Duration:   ${duration}s"
        done
    fi
else
    echo "❌ Video generation failed (exit code: ${EXIT_CODE:-unknown})"
    exit 1
fi

echo ""
