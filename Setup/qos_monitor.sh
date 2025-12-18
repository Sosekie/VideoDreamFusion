#!/bin/bash

# QOS配额实时监控脚本
# 使用方法: ./qos_monitor.sh

clear
echo "🎯 QOS配额状态监控"
echo "=================="

while true; do
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    clear
    
    echo "🎯 QOS配额状态监控 - $current_time"
    echo "=========================================="
    
    # QOS配额使用情况
    echo ""
    echo "📊 各QOS组运行任务数:"
    echo "----------------------------------------"
    interactive_running=$(squeue --qos=job_interactive -t running --noheader 2>/dev/null | wc -l)
    preemptable_running=$(squeue --qos=job_gpu_preemptable -t running --noheader 2>/dev/null | wc -l)
    debug_running=$(squeue --qos=job_gpu_debug -t running --noheader 2>/dev/null | wc -l)
    
    echo "   🔴 Interactive组:   $interactive_running 个任务运行中"
    echo "   🟡 Preemptable组:   $preemptable_running 个任务运行中"
    echo "   🟢 Debug组:         $debug_running 个任务运行中"
    
    # QOS排队情况
    echo ""
    echo "⏳ 各QOS组排队情况:"
    echo "----------------------------------------"
    interactive_pending=$(squeue --qos=job_interactive -t pending --noheader 2>/dev/null | wc -l)
    preemptable_pending=$(squeue --qos=job_gpu_preemptable -t pending --noheader 2>/dev/null | wc -l)
    debug_pending=$(squeue --qos=job_gpu_debug -t pending --noheader 2>/dev/null | wc -l)
    
    echo "   🔴 Interactive组:   $interactive_pending 个任务排队"
    echo "   🟡 Preemptable组:   $preemptable_pending 个任务排队"  
    echo "   🟢 Debug组:         $debug_pending 个任务排队"
    
    # 计算竞争强度
    echo ""
    echo "🏆 QOS竞争强度分析:"
    echo "----------------------------------------"
    
    # Interactive组分析
    if [ $interactive_pending -eq 0 ]; then
        interactive_status="✅ 空闲 - 可立即申请"
    elif [ $interactive_pending -lt 10 ]; then
        interactive_status="⚠️  轻度排队 - 预计等待10-30分钟"
    elif [ $interactive_pending -lt 30 ]; then
        interactive_status="🔶 中度排队 - 预计等待1-3小时"
    else
        interactive_status="🔴 严重拥堵 - 预计等待3+小时"
    fi
    echo "   Interactive: $interactive_status"
    
    # Preemptable组分析
    if [ $preemptable_pending -eq 0 ]; then
        preemptable_status="✅ 空闲 - 可立即申请"
    elif [ $preemptable_pending -lt 50 ]; then
        preemptable_status="⚠️  轻度排队 - 预计等待15-45分钟"
    elif [ $preemptable_pending -lt 200 ]; then
        preemptable_status="🔶 中度排队 - 预计等待1-2小时"
    else
        preemptable_status="🔴 严重拥堵 - 预计等待2+小时"
    fi
    echo "   Preemptable: $preemptable_status"
    
    # Debug组分析
    if [ $debug_pending -eq 0 ]; then
        debug_status="✅ 空闲 - 立即可用"
    elif [ $debug_pending -lt 5 ]; then
        debug_status="⚠️  轻微排队 - 预计等待2-10分钟"
    else
        debug_status="🔶 排队较多 - 预计等待10-30分钟"
    fi
    echo "   Debug:       $debug_status"
    
    # 推荐策略
    echo ""
    echo "💡 当前推荐策略:"
    echo "----------------------------------------"
    
    if [ $debug_pending -lt 3 ]; then
        echo "   🎯 首选: job_gpu_debug (快速测试 ≤20分钟)"
        echo "      命令: ./Setup/interactive_job_gpu_debug.sh"
    fi
    
    if [ $preemptable_pending -lt 100 ]; then
        echo "   🎯 推荐: job_gpu_preemptable (长时间作业)"
        echo "      命令: ./Setup/interactive_job_gpu_preemptable.sh"
    fi
    
    if [ $interactive_pending -lt 5 ] && [ $preemptable_pending -gt 200 ]; then
        echo "   🎯 备选: job_interactive (交互式开发)"
        echo "      命令: ./Setup/interactive_job_interactive.sh"
    fi
    
    if [ $interactive_pending -gt 20 ] && [ $preemptable_pending -gt 200 ] && [ $debug_pending -gt 5 ]; then
        echo "   ⏰ 建议: 等待非高峰期 (晚上/周末) 再申请"
    fi
    
    # 物理资源状态
    echo ""
    echo "🔋 物理资源状态:"
    echo "----------------------------------------"
    gpu_idle=$(sinfo -p gpu -t idle --noheader 2>/dev/null | wc -l)
    gpu_invest_idle=$(sinfo -p gpu-invest -t idle --noheader 2>/dev/null | wc -l)
    
    echo "   GPU分区空闲节点:        $gpu_idle 个"
    echo "   GPU-invest分区空闲节点: $gpu_invest_idle 个"
    
    if [ $gpu_idle -gt 0 ] || [ $gpu_invest_idle -gt 0 ]; then
        echo "   ✅ 有物理资源可用 - 瓶颈在QOS配额"
    else
        echo "   ⚠️  物理资源全部使用中"
    fi
    
    echo ""
    echo "🔄 30秒后自动刷新... (Ctrl+C 退出)"
    sleep 30
done