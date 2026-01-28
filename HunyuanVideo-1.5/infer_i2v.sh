#!/bin/bash
# Simple inference script for HunyuanVideo-1.5 I2V

# Configuration
# Text-to-video cfg-distill (480p_t2v_distilled); image optional but unused.
MODEL_TYPE="HunyuanVideo-1.5-480P-T2V-cfg-distill"
IMAGE_PATH="${1:-}"
PROMPT="${2:-A cinematic video of a hamburger. The vertical camera angle is fixed at 0 degrees (eye-level), while the horizontal camera rotates clockwise a full 360 degrees around the hamburger. The hamburger remains centered in the frame throughout the rotation, with smooth, continuous motion and high visual detail.}"
safe_name=$(echo "$PROMPT" | tr ' ' '_' | tr -cd '[:alnum:]_.' )
safe_hash=$(echo -n "$PROMPT" | md5sum | cut -c1-8)
OUTPUT_PATH="${3:-./outputs/output_i2v_${safe_name:0:80}-${safe_hash}.mp4}"
MODEL_PATH="./ckpts"
RESOLUTION="480p"
NUM_STEPS=50  # cfg-distill; guidance_scale=1 typically
VIDEO_LENGTH=1
ASPECT_RATIO="1:1"
SEED=42
ENABLE_STEP_DISTILL=false  # cfg-distill model (not step-distilled)
CFG_DISTILLED=true

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
if [ -n "$IMAGE_PATH" ] && [ ! -f "$IMAGE_PATH" ]; then
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

# Check required subfolders for 480p t2v cfg-distilled
missing=()
for p in \
    "$MODEL_PATH/transformer/480p_t2v_distilled" \
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
    echo "Download 480p T2V cfg-distilled model:"
    echo "  hf download tencent/HunyuanVideo-1.5 --include 'transformer/480p_t2v_distilled/*' --local-dir ./ckpts"
    echo ""
    echo "Or download everything (recommended):"
    echo "  bash download_models.sh <your_hf_token>"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "📋 Configuration:"
echo "  Image:        ${IMAGE_PATH:+$(realpath $IMAGE_PATH)}"
echo "  Prompt:       $PROMPT"
echo "  Resolution:   $RESOLUTION"
echo "  Steps:        $NUM_STEPS"
echo "  Video Length: $VIDEO_LENGTH"
echo "  Step Distill: $ENABLE_STEP_DISTILL"
echo "  Model Type:   $MODEL_TYPE"
echo "  CFG Distill:  $CFG_DISTILLED"
echo "  Aspect Ratio: ${ASPECT_RATIO:-16:9}"
echo "  Output:       $(realpath $OUTPUT_PATH)"
echo "  Force Torch Attn: $FORCE_TORCH_ATTN"
echo "  Force Flash2:     $FORCE_FLASH2"
echo ""

# Run generation with minimal flags for stability
echo "🎬 Starting video generation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python generate.py \
    ${IMAGE_PATH:+--image_path "$IMAGE_PATH"} \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --resolution "$RESOLUTION" \
    ${ASPECT_RATIO:+--aspect_ratio "$ASPECT_RATIO"} \
    --num_inference_steps $NUM_STEPS \
    --video_length $VIDEO_LENGTH \
    --seed $SEED \
    --output_path "$OUTPUT_PATH" \
    --enable_step_distill $ENABLE_STEP_DISTILL \
    --cfg_distilled $CFG_DISTILLED \
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
