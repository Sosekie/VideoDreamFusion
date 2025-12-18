#!/bin/bash
# ==========================================
# RTX 4090 交互式 GPU 申请脚本
# 功能与 H100.sh 相同，仅 GPU 型号和文案替换
# 若集群中 4090 设备资源标识不是 rtx4090:1，请通过环境变量 GPU_TYPE=xxx 指定
# 例如: GPU_TYPE=4090 ./Setup/4090.sh
# ==========================================

GPU_TYPE=${GPU_TYPE:-rtx4090}
MAIL_ADDR=${MAIL_ADDR:-chenrui_fan@outlook.com}
ACCOUNT=${ACCOUNT:-gratis}  # 默认账户: gratis

# GPU资源查看命令 (允许缺失)
gpu-usage 2>/dev/null || echo "(提示) gpu-usage 命令不可用，忽略"
squeue -u $USER

echo "=========================================="
echo "选择GPU交互式会话方案： (目标: RTX 4090)"
echo "=========================================="
echo "📧 邮件通知: 申请成功后将发送通知到 ${MAIL_ADDR}"
echo "👤 计费账户: ${ACCOUNT} (可通过 export ACCOUNT=xxx 修改)"
echo ""
echo "1. job_gratis (gpu分区) - 标准交互式 (原 job_interactive)"
echo "2. job_gpu_preemptable (gpu分区) - 抢占式 (资源更多但易中断)" 
echo "3. job_debug (gpu分区) - 快速调试(30分钟)"
echo "4. 查看当前资源状况后选择"
echo ""
echo "对应的快速启动脚本："
echo "  ./Setup/interactive_job_interactive.sh"
echo "  ./Setup/interactive_job_gpu_preemptable.sh"
echo "  ./Setup/interactive_job_gpu_debug.sh"
echo "=========================================="

read -p "请选择方案 (1-4, 默认1): " choice
choice=${choice:-1}

case $choice in
  1) DEFAULT_HOURS=24; MAX_HOURS=24 ;;      # job_gratis 通常限制 24h
  2) DEFAULT_HOURS=24; MAX_HOURS=24 ;;      # job_gpu_preemptable 限制 24h (UBELIX Update)
  3) DEFAULT_HOURS=0.5; MAX_HOURS=0.5 ;;   # Debug 30min
  4) DEFAULT_HOURS=24; MAX_HOURS=24 ;;      # 备选
  *) DEFAULT_HOURS=24; MAX_HOURS=24 ;;
 esac

if [ "$choice" != "3" ]; then
  echo ""
  read -p "请输入申请时长 (小时，默认${DEFAULT_HOURS}小时，最大${MAX_HOURS}小时): " hours
  hours=${hours:-$DEFAULT_HOURS}
  if ! [[ "$hours" =~ ^[0-9]+$ ]] || [ "$hours" -lt 1 ] || [ "$hours" -gt $MAX_HOURS ]; then
    echo "⚠️ 无效的时长，使用默认值: ${DEFAULT_HOURS}小时"
    hours=$DEFAULT_HOURS
  fi
  TIME_FORMAT=$(printf "%02d:00:00" $hours)
  echo "✅ 设置申请时长为: ${hours}小时 (${TIME_FORMAT})"
else
  TIME_FORMAT="00:30:00"
  hours="0.5"
  echo "✅ Debug模式固定时长: 30分钟"
fi

echo ""

# 通用申请函数
do_interactive() {
  local PARTITION=$1
  local QOS=$2
  local QOS_TAG=$3  # 用于邮件主题
  local HOURS_STR=$4
  local TIMEFMT=$5
  local MEM_SIZE=$6
  local EXTRA_NOTE=$7

  echo "正在启动交互式会话 (partition=${PARTITION}, qos=${QOS}, account=${ACCOUNT}, mem=${MEM_SIZE})..."
  # 🔥 关键修改: 添加 --account 和 --nodes=1
  srun --partition=${PARTITION} --qos=${QOS} --account=${ACCOUNT} --nodes=1 --gpus=${GPU_TYPE}:1 --cpus-per-task=4 --mem=${MEM_SIZE} --time=${TIMEFMT} --pty bash -c '
    send_notice() {
      local subject="$1"; shift
      local body="$1"; shift
      if command -v mailx >/dev/null 2>&1; then
        printf "%s" "$body" | mailx -a "Content-Type: text/plain; charset=UTF-8" -s "$subject" ${MAIL_ADDR} || echo "⚠️ 邮件可能未发送成功 (mailx)"
      elif command -v mail >/dev/null 2>&1; then
        printf "%s" "$body" | mail -s "GPU 4090 申请成功" ${MAIL_ADDR} || echo "⚠️ 邮件可能未发送成功 (mail)"
      else
        echo "⚠️ 系统未安装 mail/mailx，跳过发信"
      fi
    }

    echo "🎉 GPU RTX 4090 资源申请成功！"
    echo "📧 正在发送邮件通知..."

    send_notice "🎉 GPU RTX 4090申请成功 - ${QOS_TAG}" "GPU RTX 4090 资源申请成功！\n\n详细信息：\n• 申请时间：$(date "+%Y-%m-%d %H:%M:%S")\n• 申请时长：${HOURS_STR}小时\n• QoS类型：${QOS_TAG}\n• 账户：${ACCOUNT}\n• 分区：${PARTITION}\n• 分配节点：$(hostname)\n• 用户：$USER\n• GPU类型：RTX 4090\n${EXTRA_NOTE}\n\n你可以开始使用GPU资源进行训练/调试了！\n记得在使用完毕后及时释放资源。\n\n祝你顺利！🚀"

    echo "✅ 进入GPU节点成功！现在可以开始你的工作了。"
    exec bash
  '
  echo "✅ GPU会话结束"
}

case $choice in
  1)
    echo "🔄 使用 job_gratis QoS"
    echo "优点: 标准免费交互式QoS"
    echo "缺点: 资源可能受限"
    echo "申请时长: ${hours}小时"
    do_interactive gpu job_gratis job_gratis ${hours} ${TIME_FORMAT} "32G" ""
    ;;
  2)
    echo "🚀 使用 job_gpu_preemptable QoS"
    echo "优点: 抢占式资源，排队成功率高"
    echo "缺点: 可能被抢占"
    echo "申请时长: ${hours}小时"
    do_interactive gpu-invest job_gpu_preemptable job_gpu_preemptable ${hours} ${TIME_FORMAT} "64G" "• 注意：该 QoS 可能被高优先级任务抢占"
    ;;
  3)
    echo "⚡ 使用 job_debug QoS"
    echo "优点: 快速获得资源"
    echo "缺点: 时长限制 30 分钟"
    echo "申请时长: 30分钟 (固定)"
    do_interactive gpu job_debug job_debug ${hours} ${TIME_FORMAT} "32G" "• Debug 模式：时间限制 30 分钟"
    ;;
  4)
    echo "📊 当前资源状况："
    echo "----------------------------------------"
    echo "GPU分区队列状态："
    squeue -p gpu --format="%.8i %.10u %.2t %R" | head -10
    echo "----------------------------------------"
    echo "可用QoS选项 (Gratis账户)："
    echo "- job_gratis (标准)"
    echo "- job_gpu_preemptable (抢占式)"
    echo "- job_debug (30分钟)"
    read -p "是否直接使用推荐的 job_gpu_preemptable QoS? (y/n): " confirm
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
      do_interactive gpu-invest job_gpu_preemptable job_gpu_preemptable ${hours} ${TIME_FORMAT} "64G" "• 注意：该 QoS 可能被高优先级任务抢占"
    else
      echo "请重新运行脚本选择其他方案"
    fi
    ;;
  *)
    echo "❌ 无效选择，使用默认方案 job_gratis"
    do_interactive gpu job_gratis job_gratis ${hours} ${TIME_FORMAT} "32G" ""
    ;;
 esac

echo ""
echo "=========================================="
echo "交互式会话说明："
echo "=========================================="
echo "✅ 成功进入节点后可以："
echo "   - 执行命令 (nvidia-smi, python 等)"
echo "   - 运行训练脚本"
echo "   - 实时调试/采样"
echo "   - 查看日志输出"
echo ""
echo "⚠️  注意事项："
echo "   - SSH 断开会中断会话 (建议使用 tmux / screen)"
echo "   - 长时间训练建议使用 sbatch 批处理提交"
echo "   - Debug QoS 仅适合快速测试"
echo "=========================================="