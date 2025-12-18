#!/bin/bash
# 默认交互式会话 - job_interactive
echo "🔄 启动默认交互式会话 (job_interactive)"
echo "优点: 标准交互式QoS，优先级50"
echo "缺点: 资源紧张时等待时间长"
echo "正在分配资源..."

srun --partition=gpu --qos=job_interactive --gpus=h100:1 --cpus-per-task=4 --mem=90G --time=08:00:00 --pty bash