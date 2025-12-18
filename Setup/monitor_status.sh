#!/bin/bash

# GPU和任务状态监控脚本
# 使用方法: ./monitor_status.sh

clear
echo "🚀 Ubelix GPU 状态监控面板"
echo "==========================================="
echo "按 Ctrl+C 退出监控"
echo ""

while true; do
    # 获取当前时间
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 清屏并显示标题
    clear
    echo "🚀 Ubelix GPU 状态监控面板 - $current_time"
    echo "==========================================="
    
    # 显示用户的任务状态
    echo ""
    echo "📊 您的任务状态:"
    echo "-------------------------------------------"
    my_jobs=$(squeue -u $USER --noheader 2>/dev/null | wc -l)
    if [ $my_jobs -eq 0 ]; then
        echo "   ✅ 当前没有运行中的任务"
    else
        squeue -u $USER --format="%.8i %.12P %.15j %.8T %.10M %.6D %R" | head -10
        if [ $my_jobs -gt 10 ]; then
            echo "   ... 还有 $((my_jobs - 10)) 个任务未显示"
        fi
    fi
    
    # 显示GPU分区资源状态
    echo ""
    echo "🔋 GPU资源状态:"
    echo "-------------------------------------------"
    # GPU分区状态
    gpu_total=$(sinfo -p gpu -h -o "%D")
    gpu_idle=$(sinfo -p gpu -t idle -h -o "%D" | awk '{sum+=$1} END {print sum+0}')
    gpu_alloc=$(sinfo -p gpu -t allocated -h -o "%D" | awk '{sum+=$1} END {print sum+0}')
    
    echo "   GPU分区:     总计:$gpu_total  空闲:$gpu_idle  已分配:$gpu_alloc"
    
    # GPU-invest分区状态
    gpu_inv_total=$(sinfo -p gpu-invest -h -o "%D" 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    gpu_inv_idle=$(sinfo -p gpu-invest -t idle -h -o "%D" 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    gpu_inv_alloc=$(sinfo -p gpu-invest -t allocated -h -o "%D" 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    
    echo "   GPU-invest:  总计:$gpu_inv_total  空闲:$gpu_inv_idle  已分配:$gpu_inv_alloc"
    
    # 显示排队情况
    echo ""
    echo "⏳ 队列状态:"
    echo "-------------------------------------------"
    
    # GPU分区队列
    gpu_pending=$(squeue -p gpu -t pending --noheader 2>/dev/null | wc -l)
    echo "   GPU分区排队:        $gpu_pending 个任务"
    
    # GPU-invest分区队列
    gpu_inv_pending=$(squeue -p gpu-invest -t pending --noheader 2>/dev/null | wc -l)
    echo "   GPU-invest分区排队: $gpu_inv_pending 个任务"
    
    # 显示最新的任务完成情况
    echo ""
    echo "📈 最近完成的任务 (最近1小时):"
    echo "-------------------------------------------"
    recent_jobs=$(sacct -S now-1hour -u $USER --format=JobID,JobName,State,End --noheader 2>/dev/null | head -5)
    if [ -z "$recent_jobs" ]; then
        echo "   📭 最近1小时内没有完成的任务"
    else
        echo "$recent_jobs"
    fi
    
    # 如果当前在GPU节点，显示GPU使用情况
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        echo "🎮 当前节点GPU状态:"
        echo "-------------------------------------------"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk -F, '{printf "   GPU%d: %s | 使用率:%s%% | 显存:%s/%sMB | 温度:%s°C\n", $1, $2, $3, $4, $5, $6}'
    fi
    
    # 显示推荐操作
    echo ""
    echo "💡 快速操作:"
    echo "-------------------------------------------"
    echo "   获取交互式GPU: ./Setup/interactive_job_gpu_preemptable.sh"
    echo "   查看详细队列:   squeue -u $USER"
    echo "   提交训练任务:   sbatch your_script.sbatch"
    echo "   取消任务:       scancel <JOB_ID>"
    
    echo ""
    echo "🔄 30秒后自动刷新... (Ctrl+C 退出)"
    
    # 等待30秒
    sleep 30
done