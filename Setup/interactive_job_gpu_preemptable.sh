#!/bin/bash
# 推荐交互式会话 - gpu-invest + preemptable
echo "🚀 启动推荐交互式会话 (gpu-invest + preemptable)"
echo "优点: 成功率高，利用投资分区资源"
echo "缺点: 可能被高优先级任务抢占"
echo "正在分配资源..."

srun --partition=gpu-invest --qos=job_gpu_preemptable --gpus=h100:1 --cpus-per-task=4 --mem=90G --time=08:00:00 --pty bash