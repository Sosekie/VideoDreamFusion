#!/bin/bash
# Quick status check for HunyuanVideo-1.5 setup

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     HunyuanVideo-1.5 Setup Status Check                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Source environment
. setup_env.sh 2>/dev/null

echo "📊 Configuration Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1️⃣  Conda Environment"
if conda activate HunyuanVideoThreestudio 2>/dev/null; then
    echo "   ✓ Environment: HunyuanVideoThreestudio"
else
    echo "   ✗ Environment not found"
fi
echo ""

echo "2️⃣  Python & PyTorch"
python -c "
import sys, torch
print(f'   ✓ Python: {sys.version_info.major}.{sys.version_info.minor}')
print(f'   ✓ PyTorch: {torch.__version__}')
print(f'   ✓ CUDA: {torch.version.cuda}')
" 2>/dev/null
echo ""

echo "3️⃣  GPU Setup"
python -c "
import torch
if torch.cuda.is_available():
    print(f'   ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'   ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'   ✓ Compute Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('   ✗ GPU not available')
" 2>/dev/null
echo ""

echo "4️⃣  Model Files"
if [ -d "ckpts" ]; then
    echo "   ✓ ckpts directory exists"
    if [ -f "ckpts/transformer/config.json" ]; then
        echo "   ✓ Transformer model found"
    else
        echo "   ⚠ Transformer model NOT downloaded"
    fi
    if [ -d "ckpts/text_encoder" ]; then
        echo "   ✓ Text encoders found"
    else
        echo "   ⚠ Text encoders NOT downloaded"
    fi
    
    # Calculate total size
    if [ -d "ckpts" ]; then
        SIZE=$(du -sh ckpts 2>/dev/null | awk '{print $1}')
        echo "   📦 Total size: $SIZE"
    fi
else
    echo "   ⚠ ckpts directory NOT created"
fi
echo ""

echo "5️⃣  Scripts"
echo "   ✓ setup_env.sh (environment setup)"
echo "   ✓ download_720p_i2v.sh (model download)"
echo "   ✓ test_installation.sh (verification)"
echo "   ✓ quick_start.sh (quick start guide)"
echo "   ✓ SETUP_GUIDE.md (detailed guide)"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Quick Commands                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Download models:"
echo "  bash download_720p_i2v.sh"
echo ""
echo "Test installation:"
echo "  bash test_installation.sh"
echo ""
echo "Generate video (T2V):"
echo "  python generate.py --prompt 'Your prompt' --resolution 720p --model_path ./ckpts"
echo ""
echo "Generate video (I2V):"
echo "  python generate.py --image_path input.png --resolution 720p --model_path ./ckpts"
echo ""
echo "For more info:"
echo "  cat SETUP_GUIDE.md"
echo ""
