#!/bin/bash
# Setup script for HunyuanVideo-1.5 environment
# Usage: source setup_env.sh

# 1) Load CUDA module (Ubelix specific)
if command -v module &>/dev/null; then
  module purge
  module load CUDA/12.6.0
  module load cuDNN/9.5.0.50-CUDA-12.6.0
fi

# 2) Auto-detect CUDA_HOME
if command -v nvcc &>/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
elif [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
else
  echo "[ERR] nvcc not found. Please run: module load CUDA/12.6.0"
  return 1 2>/dev/null || exit 1
fi

# 3) Clean up conflicting CUDA paths (keep only 12.6)
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/cuda-11\.' | paste -sd: -)"
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/cuda-11\.' | paste -sd: -)"

# 4) Inject CUDA 12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# 5) Set compilation architecture based on GPU type
# For H100: SM 9.0, For 4090: SM 8.9, For A100: SM 8.0
export TORCH_CUDA_ARCH_LIST="9.0"  # Change to 8.9 for 4090, 8.0 for A100
export TCNN_CUDA_ARCHITECTURES="90"  # Change to 89 for 4090, 80 for A100
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# 6) Load Anaconda
module load Anaconda3/2024.02-1 2>/dev/null || true
eval "$(/software.9/software/Anaconda3/2024.02-1/bin/conda shell.bash hook)"

# 7) Activate the environment
conda activate ThreestudioWithHunyuanVideo

# 7.5) Add PyTorch lib to LD_LIBRARY_PATH (fix for Flash Attention and other compiled extensions)
export TORCH_LIB=$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# 8) Verification
echo "=========================================="
echo "Environment Setup Complete"
echo "=========================================="
python - <<'PY'
import os, shutil, torch, importlib.util

print("torch version:", torch.__version__)
print("CUDA version (torch):", torch.version.cuda)
print("nvcc:", shutil.which("nvcc"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("TORCH_LIB:", os.environ.get("TORCH_LIB"))
print("LD_LIBRARY_PATH contains TORCH_LIB:", os.environ.get("TORCH_LIB","") in os.environ.get("LD_LIBRARY_PATH",""))
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))

# Ensure FA2 extension is discoverable and loadable even without prior imports
spec = importlib.util.find_spec("flash_attn_2_cuda")
print("flash_attn_2_cuda spec:", spec is not None)
if spec is not None:
    import flash_attn_2_cuda
    print("flash_attn_2_cuda import: OK")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Capability:", torch.cuda.get_device_capability(0))
print("==========================================")
PY


# === 复制 HunyuanVideo checkpoints 到本地 scratch 以加速加载 ===
CKPTS_LINK="HunyuanVideo-1.5/ckpts"
CKPTS_BACKUP="HunyuanVideo-1.5/ckpts.homefs"
TARGET="${TMPDIR:-/tmp}/HunyuanVideo-1.5/ckpts"

# 检查符号链接是否存在且有效（目标目录可访问）
if [ -L "$CKPTS_LINK" ] && [ -d "$CKPTS_LINK" ]; then
    echo "[ckpts] 符号链接有效，跳过复制: $(readlink "$CKPTS_LINK")"
else
    echo "[ckpts] 需要重新设置..."
    
    # 确保备份目录存在
    if [ -d "$CKPTS_LINK" ] && [ ! -L "$CKPTS_LINK" ]; then
        # 原始目录存在（不是符号链接），移动为备份
        mv "$CKPTS_LINK" "$CKPTS_BACKUP"
        echo "[ckpts] 已备份原目录到 $CKPTS_BACKUP"
    fi
    
    # 删除无效的符号链接
    [ -L "$CKPTS_LINK" ] && rm -f "$CKPTS_LINK"
    
    # 删除备份目录内可能存在的错误符号链接
    [ -L "$CKPTS_BACKUP/ckpts" ] && rm -f "$CKPTS_BACKUP/ckpts"
    
    # 复制到 scratch
    echo "[ckpts] 复制到 $TARGET ..."
    mkdir -p "$TARGET"
    rsync -a "$CKPTS_BACKUP/" "$TARGET"/
    
    # 创建符号链接
    ln -s "$TARGET" "$CKPTS_LINK"
    echo "[ckpts] 完成！"
fi

ls "$CKPTS_LINK"