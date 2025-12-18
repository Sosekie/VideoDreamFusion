# 🚀 Ubelix集群GPU使用完整教程

## 📋 目录
- [GPU资源申请方式](#gpu资源申请方式)
- [交互式会话使用](#交互式会话使用)
- [批处理作业提交](#批处理作业提交)
- [任务状态监控](#任务状态监控)
- [日志查看和调试](#日志查看和调试)
- [资源管理和优化](#资源管理和优化)
- [常见问题解决](#常见问题解决)
- [最佳实践建议](#最佳实践建议)

---

## GPU资源申请方式

### 🎯 两种主要方式

#### 1. 交互式会话 (Interactive Session)
**特点：** 直接在GPU节点上获得shell访问，适合开发、调试、测试

#### 2. 批处理作业 (Batch Job)
**特点：** 提交脚本到队列，后台运行，适合长时间训练

---

## 交互式会话使用

### 📁 快速启动脚本

本项目提供了三种预配置的交互式会话脚本：

```bash
# 智能选择脚本（推荐新用户）
./Setup/H100.sh

# 直接启动特定QoS会话
./Setup/interactive_job_interactive.sh          # 标准交互式
./Setup/interactive_job_gpu_preemptable.sh      # 推荐：成功率最高
./Setup/interactive_job_gpu_debug.sh            # 快速调试（20分钟限制）
```

### 🔧 手动命令格式

```bash
srun --partition=<分区> --qos=<QoS> --gpus=<GPU类型>:<数量> \
     --cpus-per-task=<CPU数> --mem=<内存> --time=<时间限制> --pty bash
```

### 📊 QoS策略对比

| QoS类型 | 优先级 | 时间限制 | 分区 | 特点 | 推荐场景 |
|---------|--------|----------|------|------|----------|
| `job_interactive` | 50 | 8小时 | gpu | 标准交互式 | 日常开发 |
| `job_gpu_preemptable` | 0 | 1天 | gpu-invest | 可抢占，成功率高 | **推荐使用** |
| `job_gpu_debug` | 50 | 20分钟 | gpu | 快速获得资源 | 快速测试 |

### ✅ 交互式会话成功后

```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA版本
nvcc --version

# 激活conda环境
conda activate your_env

# 运行Python脚本
python your_script.py

# 实时监控GPU使用
watch -n 1 nvidia-smi
```

---

## 批处理作业提交

### 📄 SLURM脚本示例

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my_training
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 激活环境
conda activate your_env

# 运行训练脚本
python train.py --epochs 100 --batch-size 32
```

### 🚀 提交和管理

```bash
# 提交作业
sbatch your_script.sbatch

# 取消作业
scancel <JOB_ID>

# 查看作业详情
scontrol show job <JOB_ID>

# 查看作业历史
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
```

---

## 任务状态监控

### 🔍 基础查看命令

```bash
# 查看自己的任务
squeue -u $USER

# 查看详细信息
squeue -u $USER --format="%.10i %.12P %.20j %.10u %.2t %.10M %.5D %.20R"

# 查看特定分区的任务
squeue -p gpu-invest

# 实时监控
watch -n 5 'squeue -u $USER'
```

### 📊 任务状态含义

| 状态 | 含义 | 说明 |
|------|------|------|
| `PD` | Pending | 等待资源分配 |
| `R` | Running | 正在运行 |
| `CG` | Completing | 即将完成 |
| `CD` | Completed | 已完成 |
| `F` | Failed | 失败 |
| `CA` | Cancelled | 已取消 |

### 🚨 常见等待原因

| 原因 | 解释 | 解决方案 |
|------|------|----------|
| `(QOSGrpGRES)` | QoS资源限制 | 尝试其他QoS或减少资源请求 |
| `(Resources)` | 资源不足 | 等待或减少资源需求 |
| `(Priority)` | 优先级低 | 使用更高优先级的QoS |
| `(QOSMaxGRESPerUser)` | 用户资源限制 | 等待其他任务完成 |

---

## 日志查看和调试

### 📝 日志文件位置

```bash
# SLURM自动生成的日志
logs/your_job_name-<JOB_ID>.out    # 标准输出
logs/your_job_name-<JOB_ID>.err    # 错误输出

# 自定义日志目录
logs_training/
tensorboard_logs/
checkpoints/
```

### 🔧 实时日志监控

```bash
# 实时查看输出日志
tail -f logs/your_job-12345.out

# 实时查看错误日志
tail -f logs/your_job-12345.err

# 查看最近的日志内容
tail -100 logs/your_job-12345.out

# 搜索日志中的关键词
grep -n "error\|Error\|ERROR" logs/your_job-12345.err
grep -n "loss\|accuracy" logs/your_job-12345.out
```

### 📊 TensorBoard监控（如果适用）

```bash
# 在交互式会话中启动TensorBoard
tensorboard --logdir=./logs_training --host=0.0.0.0 --port=6006

# 通过SSH隧道访问
ssh -L 6006:localhost:6006 user@submit03.unibe.ch
# 然后在浏览器中访问 http://localhost:6006
```

---

## 资源管理和优化

### 💾 存储管理

```bash
# 检查磁盘使用
du -sh * | sort -hr

# 清理临时文件
find . -name "*.tmp" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 压缩大文件
tar -czf archive_name.tar.gz folder_to_compress/
```

### 🔋 GPU资源查看

```bash
# 当前GPU使用情况
gpu-usage

# 查看可用GPU类型
sinfo -p gpu --Format=partition,avail,nodes,gres

# 检查GPU节点状态
sinfo -p gpu-invest -N -o "%N %G %C %m %e %T"
```

### ⚡ 性能监控

```bash
# GPU监控脚本（保存为monitor_gpu.sh）
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    echo ""
    sleep 10
done
```

---

## 常见问题解决

### ❓ Q1: 任务一直在等待队列中
```bash
# 检查任务等待原因
squeue -j <JOB_ID> --format="%.18i %.20P %.8j %.8u %.8T %.19S %.6D %.20R %Q"

# 解决方案
1. 尝试其他QoS: job_gpu_preemptable
2. 减少资源请求（内存、GPU数量）
3. 选择不同时间段提交
4. 使用gpu-invest分区
```

### ❓ Q2: 交互式会话连接断开
```bash
# 使用screen或tmux保持会话
screen -S gpu_session
# 或
tmux new-session -s gpu_session

# 断开连接后重新连接
screen -r gpu_session
# 或
tmux attach -t gpu_session
```

### ❓ Q3: 内存不足错误
```bash
# 检查内存使用
free -h
top -u $USER

# 解决方案
1. 减少batch_size
2. 使用gradient_accumulation
3. 请求更多内存（--mem=180G）
4. 使用混合精度训练（fp16）
```

### ❓ Q4: CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi  # Driver CUDA Version
nvcc --version  # Runtime CUDA Version

# 解决方案
module load CUDA/11.8
# 或重新安装匹配的PyTorch版本
```

---

## 最佳实践建议

### 🎯 资源申请策略

1. **开发调试阶段**
   ```bash
   ./Setup/interactive_job_gpu_debug.sh  # 快速获得20分钟
   ```

2. **长时间开发**
   ```bash
   ./Setup/interactive_job_gpu_preemptable.sh  # 推荐
   ```

3. **正式训练**
   ```bash
   sbatch your_training_script.sbatch  # 批处理作业
   ```

### 📁 项目组织建议

```
your_project/
├── data/                    # 数据文件
├── src/                     # 源代码
├── configs/                 # 配置文件
├── scripts/                 # 训练脚本
│   ├── train.py
│   └── train.sbatch
├── logs/                    # SLURM日志
├── checkpoints/            # 模型检查点
├── results/                # 实验结果
└── Setup/                  # 环境配置（本教程提供的脚本）
    ├── H100.sh
    ├── interactive_job_gpu_preemptable.sh
    └── GPU_Tutorial.md     # 本教程
```

### ⚡ 性能优化建议

1. **训练优化**
   ```python
   # 使用混合精度
   from torch.cuda.amp import autocast, GradScaler
   
   # 启用XFormers（如适用）
   model.enable_xformers_memory_efficient_attention()
   
   # 优化DataLoader
   DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

2. **资源监控**
   ```bash
   # 定期检查GPU使用率
   nvidia-smi dmon -s puc
   
   # 监控训练进度
   tail -f logs/training.log | grep -E "(loss|accuracy|step)"
   ```

3. **检查点管理**
   ```python
   # 定期保存检查点
   if step % 500 == 0:
       torch.save(model.state_dict(), f'checkpoint-{step}.pt')
   
   # 限制检查点数量
   checkpoints_total_limit = 3
   ```

### 🔒 安全和备份

```bash
# 定期备份重要文件
rsync -av --progress checkpoints/ backup/checkpoints/
rsync -av --progress results/ backup/results/

# 使用版本控制
git add -A && git commit -m "Experiment checkpoint"
git push origin main
```

---

## 📞 获取帮助

### 🆘 紧急问题

1. **任务异常终止**
   ```bash
   # 查看详细错误信息
   sacct -j <JOB_ID> --format=JobID,State,ExitCode,DerivedExitCode
   
   # 查看完整日志
   cat logs/your_job-<JOB_ID>.err
   ```

2. **资源使用异常**
   ```bash
   # 联系管理员前收集信息
   scontrol show job <JOB_ID>
   sstat -j <JOB_ID> --format=JobID,MaxRSS,MaxVMSize,NTasks
   ```

### 📚 更多资源

- [Ubelix官方文档](https://hpc-unibe-ch.github.io/user-guide/)
- [SLURM用户指南](https://slurm.schedmd.com/documentation.html)
- [集群使用政策](https://www.id.unibe.ch/hpc)

---

## 📈 更新日志

- **2025-09-24**: 初始版本，包含完整的GPU使用教程
- 涵盖交互式会话、批处理作业、监控和调试
- 提供三种QoS策略的对比和使用建议

---

**💡 提示**: 建议将此教程加入书签，并定期查看更新。如有问题或建议，请联系项目维护者。