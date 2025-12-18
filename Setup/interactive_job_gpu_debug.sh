#!/bin/bash
# 快速调试会话 - job_gpu_debug (20分钟限制)
echo "⚡ 启动快速调试会话 (job_gpu_debug)"
echo "优点: 优先级高，快速获得资源"
echo "缺点: 时间限制20分钟"
echo "适用于: 快速测试、调试、验证"
echo "正在分配资源..."

srun --partition=gpu --qos=job_gpu_debug --gpus=h100:1 --cpus-per-task=4 --mem=90G --time=00:20:00 --pty bash