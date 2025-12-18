# HunyuanVideo-1.5 配置完成指南

## ✅ 完成状态

### 已完成
- ✓ Clone HunyuanVideo-1.5仓库到 `./HunyuanVideo-1.5`
- ✓ 创建conda环境：`HunyuanVideoThreestudio`
- ✓ 安装基础依赖 (requirements.txt)
- ✓ 安装Flash Attention优化库
- ✓ 修复PyTorch版本 (2.5.1 + CUDA 11.8)
- ✓ 验证GPU环境 (H100 NVL, 99.9GB)

### 待完成
- ⏳ 下载预训练模型 (~50GB+)
- ⏳ 测试视频生成

---

## 📁 项目结构

```
/storage/homefs/cf23h027/VideoDreamFusion/HunyuanVideo-1.5/
├── setup_env.sh                 # 环境加载脚本
├── download_720p_i2v.sh        # 模型下载脚本
├── test_installation.sh         # 安装验证脚本
├── quick_start.sh              # 快速启动指南
├── generate.py                 # 主生成脚本
├── requirements.txt            # Python依赖
├── ckpts/                       # 模型权重目录（需下载）
│   ├── transformer/
│   ├── vae/
│   └── text_encoder/
└── ...
```

---

## 🚀 快速开始

### 1. 激活环境
```bash
cd /storage/homefs/cf23h027/VideoDreamFusion/HunyuanVideo-1.5
source setup_env.sh
```

### 2. 下载模型
首次使用需要下载模型（约50-60GB）：
```bash
bash download_720p_i2v.sh
```

或者手动下载指定部分：
```bash
# 只下载720p I2V模型
hf download tencent/HunyuanVideo-1.5 \
  --local-dir ./ckpts \
  --include "transformer/*720p*" \
  --include "vae/*"
```

### 3. 验证安装
```bash
bash test_installation.sh
```

### 4. 生成视频

**文本生成视频（T2V）**
```bash
python generate.py \
  --prompt "A beautiful sunset over mountains" \
  --resolution 720p \
  --model_path ./ckpts \
  --output_path ./output.mp4
```

**图像生成视频（I2V）**
```bash
python generate.py \
  --image_path ./input.png \
  --prompt "Camera slowly zooms in on the image" \
  --resolution 720p \
  --model_path ./ckpts \
  --output_path ./output.mp4
```

**快速生成（480p, 8步）**
```bash
python generate.py \
  --image_path ./input.png \
  --resolution 480p \
  --enable_step_distill \
  --num_inference_steps 8 \
  --model_path ./ckpts
```

---

## 🔧 环境信息

| 项目 | 配置 |
|------|------|
| **Python** | 3.10.19 |
| **PyTorch** | 2.5.1 + CUDA 11.8 |
| **GPU** | NVIDIA H100 NVL (99.9 GB) |
| **Flash Attention** | 2.8.3 ✓ |
| **Conda环境** | `HunyuanVideoThreestudio` |

---

## 📋 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt` | 文本提示 (T2V) | 必需 |
| `--image_path` | 输入图像 (I2V) | None |
| `--resolution` | 分辨率: 480p/720p | 必需 |
| `--num_inference_steps` | 推理步数 | 50 |
| `--seed` | 随机种子 | 123 |
| `--output_path` | 输出视频路径 | ./outputs/output_{timestamp}.mp4 |
| `--enable_step_distill` | 启用步长蒸馏 (480p I2V) | false |
| `--cfg_distilled` | 启用CFG蒸馏 (~2x加速) | false |
| `--sr` | 启用超分 | true |
| `--rewrite` | 启用提示词重写 | true |

---

## 💾 模型大小参考

| 模型 | 大小 | 说明 |
|------|------|------|
| DiT (720p) | ~30GB | 主变压器模型 |
| VAE | ~5GB | 视频编码器 |
| MLLM (Qwen2.5-VL) | ~15GB | 文本编码器 |
| byT5 | ~1GB | 字符级文本编码 |
| Glyph-SDXL | ~3GB | 字形编码 |
| **总计** | **~50GB** | - |

---

## ⚡ 性能提示

1. **启用步长蒸馏** (480p I2V)
   ```bash
   --enable_step_distill --num_inference_steps 8
   ```
   速度提升：75% ⚡

2. **启用CFG蒸馏**
   ```bash
   --cfg_distilled
   ```
   速度提升：2x ⚡

3. **禁用offloading** (GPU内存充足时)
   ```bash
   --offloading false
   ```
   速度提升：显著 ⚡

4. **启用缓存加速**
   ```bash
   --enable_cache --cache_type deepcache
   ```

---

## 🔗 有用链接

- **官方仓库**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- **模型下载**: https://huggingface.co/tencent/HunyuanVideo-1.5
- **提示词指南**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md

---

## 📝 下一步

1. **下载模型** (必需)
   ```bash
   bash download_720p_i2v.sh
   ```

2. **测试生成** (可选)
   ```bash
   bash test_installation.sh
   ```

3. **开始创建** 
   ```bash
   python generate.py --help
   ```

---

## 💡 常见问题

**Q: 模型下载太慢？**
A: 使用HF镜像加速：
```bash
HF_ENDPOINT=https://hf-mirror.com bash download_720p_i2v.sh
```

**Q: 显存不足？**
A: 启用模型offloading:
```bash
--offloading true --group_offloading true
```

**Q: 需要提示词优化？**
A: 配置vLLM服务器:
```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server>"
export T2V_REWRITE_MODEL_NAME="Qwen3-235B-A22B-Thinking-2507"
```

---

更新时间: 2025-12-16
