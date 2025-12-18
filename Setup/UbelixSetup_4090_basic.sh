# Setup/UbelixSetup_4090.sh
# 用法：source Setup/UbelixSetup_4090.sh

# 1) 选定 CUDA 11.8（Ubelix 推荐）
if command -v module &>/dev/null; then
  module purge
  module load CUDA/11.8.0
fi

# 2) 自动推导 CUDA_HOME（若没 nvcc，则回退到 /usr/local/cuda-11.8）
if command -v nvcc &>/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
elif [ -d /usr/local/cuda-11.8 ]; then
  export CUDA_HOME=/usr/local/cuda-11.8
else
  echo "[ERR] nvcc 未找到，且 /usr/local/cuda-11.8 不存在。请先 `module load CUDA/11.8.0`。"
  return 1 2>/dev/null || exit 1
fi

# 3) 清理 PATH/LD_LIBRARY_PATH 中可能残留的 CUDA 12.x 路径，避免混淆
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -vE '/cuda-1[2-9]\.|/cuda-12\.' | paste -sd: -)"
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '/cuda-1[2-9]\.|/cuda-12\.' | paste -sd: -)"

# 4) 注入当前 CUDA 11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# 5) 设定 4090 的编译架构（SM 8.9）
export TORCH_CUDA_ARCH_LIST="8.9"
export TCNN_CUDA_ARCHITECTURES="89"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# 6) 进入 conda 环境
module load Anaconda3/2024.02-1 2>/dev/null || true
eval "$(/software.9/software/Anaconda3/2024.02-1/bin/conda shell.bash hook)"
conda activate threestudio_4090_basic

# 7) 自检并修复 PyTorch 版本
python - <<'PY'
import os, shutil, torch, sys

print("=" * 60)
print("Environment Check")
print("=" * 60)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("nvcc:", shutil.which("nvcc"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("TCNN_CUDA_ARCHITECTURES:", os.environ.get("TCNN_CUDA_ARCHITECTURES"))

# Check if PyTorch CUDA version matches
if torch.version.cuda != "11.8":
    print("\n" + "=" * 60)
    print("⚠️  WARNING: PyTorch CUDA version mismatch!")
    print("=" * 60)
    print(f"Expected: CUDA 11.8")
    print(f"Got:      CUDA {torch.version.cuda}")
    print("\nTo fix, run these commands:")
    print("  pip uninstall torch torchvision torchaudio -y")
    print("  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118")
    print("  pip uninstall tinycudann -y")
    print("  pip install --no-cache-dir --force-reinstall git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
    print("=" * 60)
    sys.exit(1)
else:
    print("\n✓ PyTorch CUDA version is correct (11.8)")
PY
