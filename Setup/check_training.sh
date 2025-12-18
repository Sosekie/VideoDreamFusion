#!/bin/bash

# 训练日志监控脚本
# 使用方法: ./check_training.sh [job_id]

# 如果没有提供job_id，尝试获取最新的任务
JOB_ID=${1:-$(squeue -u $USER --noheader --format="%i" | head -1)}

if [ -z "$JOB_ID" ]; then
    echo "❌ 没有找到运行中的任务"
    echo "使用方法: $0 [job_id]"
    exit 1
fi

echo "🔍 检查任务 $JOB_ID 的训练状态"
echo "==========================================="

# 检查任务状态
echo ""
echo "📊 任务信息:"
squeue --job=$JOB_ID --format="%.8i %.12P %.15j %.8T %.10M %.6D %R %E" 2>/dev/null || echo "任务 $JOB_ID 可能已完成或不存在"

# 查找日志文件
echo ""
echo "📝 日志文件检查:"
echo "-------------------------------------------"

# 检查当前目录的日志
if [ -f "svd_xtend-${JOB_ID}.out" ]; then
    echo "✅ 输出日志: svd_xtend-${JOB_ID}.out"
    OUT_LOG="svd_xtend-${JOB_ID}.out"
elif [ -f "logs/svd_xtend-${JOB_ID}.out" ]; then
    echo "✅ 输出日志: logs/svd_xtend-${JOB_ID}.out"
    OUT_LOG="logs/svd_xtend-${JOB_ID}.out"
else
    echo "❓ 未找到输出日志 svd_xtend-${JOB_ID}.out"
    OUT_LOG=""
fi

if [ -f "svd_xtend-${JOB_ID}.err" ]; then
    echo "✅ 错误日志: svd_xtend-${JOB_ID}.err"
    ERR_LOG="svd_xtend-${JOB_ID}.err"
elif [ -f "logs/svd_xtend-${JOB_ID}.err" ]; then
    echo "✅ 错误日志: logs/svd_xtend-${JOB_ID}.err"
    ERR_LOG="logs/svd_xtend-${JOB_ID}.err"
else
    echo "❓ 未找到错误日志 svd_xtend-${JOB_ID}.err"
    ERR_LOG=""
fi

# 显示最新的训练进度
if [ -n "$OUT_LOG" ] && [ -f "$OUT_LOG" ]; then
    echo ""
    echo "🚀 最新训练进度 (最后10行):"
    echo "-------------------------------------------"
    tail -10 "$OUT_LOG"
    
    echo ""
    echo "📈 训练统计:"
    echo "-------------------------------------------"
    
    # 提取训练步数
    current_step=$(grep -o "step [0-9]\+" "$OUT_LOG" | tail -1 | grep -o "[0-9]\+")
    if [ -n "$current_step" ]; then
        echo "   当前步数: $current_step"
        total_steps=100000
        progress=$(echo "scale=2; $current_step * 100 / $total_steps" | bc -l)
        echo "   训练进度: ${progress}% (${current_step}/${total_steps})"
    fi
    
    # 查找loss信息
    latest_loss=$(grep "loss:" "$OUT_LOG" | tail -1 | grep -o "loss: [0-9.]*" | grep -o "[0-9.]*")
    if [ -n "$latest_loss" ]; then
        echo "   最新损失: $latest_loss"
    fi
    
    # 查找学习率
    latest_lr=$(grep "lr:" "$OUT_LOG" | tail -1 | grep -o "lr: [0-9.e-]*" | grep -o "[0-9.e-]*")
    if [ -n "$latest_lr" ]; then
        echo "   学习率: $latest_lr"
    fi
fi

# 检查错误日志
if [ -n "$ERR_LOG" ] && [ -f "$ERR_LOG" ] && [ -s "$ERR_LOG" ]; then
    echo ""
    echo "⚠️  错误日志内容:"
    echo "-------------------------------------------"
    tail -10 "$ERR_LOG"
fi

# 检查checkpoint目录
echo ""
echo "💾 模型检查点:"
echo "-------------------------------------------"
for checkpoint_dir in outputs_seedling_continue outputs_seedling_full; do
    if [ -d "$checkpoint_dir" ]; then
        echo "📁 $checkpoint_dir/:"
        ls -lt "$checkpoint_dir"/checkpoint-* 2>/dev/null | head -5 | while read line; do
            echo "   $line"
        done
    fi
done

echo ""
echo "🔧 有用的命令:"
echo "-------------------------------------------"
echo "   实时监控输出: tail -f $OUT_LOG"
echo "   查看完整日志: less $OUT_LOG"
echo "   取消任务:     scancel $JOB_ID"
echo "   任务详情:     scontrol show job $JOB_ID"