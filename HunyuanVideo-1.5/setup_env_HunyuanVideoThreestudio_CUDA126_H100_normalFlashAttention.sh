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
conda activate HunyuanVideoThreestudio_CUDA126_H100_normalFlashAttention

# 8) Verification
echo "=========================================="
echo "Environment Setup Complete"
echo "=========================================="
python - <<'PY'
import os, shutil, torch

print("torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("nvcc:", shutil.which("nvcc"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Capability:", torch.cuda.get_device_capability(0))
print("==========================================")
PY
