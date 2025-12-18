#!/usr/bin/env bash
# h100_watch.sh v4 — UBELIX H100 监控快照（中文友好 + 更准确的统计）
#
# 设计目标：
# - 汇总显示：F1(免费层, partition=gpu) 与 Investor(投资层, partition=gpu-invest) 的 H100 运行/等待/总数
# - 节点用量：逐节点显示 H100 总卡 / 已用 / 空闲（优先从 scontrol 的 GresUsed= 读取，失败再回退到 squeue）
# - F1 关键节点：单独列出 F1 免费池中“空闲≥1”的节点，便于用 --nodelist 精确抢卡
#
# 背景知识：
# - F1 免费层只有一个“小池子”，H100 总量通常为 8 张（但脚本会动态计算，不写死）
# - Investor 可抢占：只有在投资者不用时才会分配，且随时可能被抢占
#
# 示例：
#   ./h100_watch.sh
#   ./h100_watch.sh --watch
#
# 依赖：sinfo, squeue, scontrol, awk, grep, sed

REFRESH=0
[[ "${1:-}" == "--watch" ]] && REFRESH=1

# ---------- 工具函数：字符串修整 ----------
trim() { sed 's/^[[:space:]]\+//; s/[[:space:]]\+$//' ; }

# ---------- 统计函数：分区内正在运行/等待中的 H100 张数（按作业请求近似汇总） ----------
sum_part_usage() {
  # 输入：分区名（gpu 或 gpu-invest）
  local part="$1"
  local used pend
  # 正在运行的作业请求了多少 h100（累加 h100:K）
  used=$(squeue -h -p "$part" -t R  -o "%b" | grep -o 'h100:[0-9]\+' | awk -F: '{s+=$2} END{print s+0}')
  # 等待中的作业请求了多少 h100（近似统计）
  pend=$(squeue -h -p "$part" -t PD -o "%b" | grep -o 'h100:[0-9]\+' | awk -F: '{s+=$2} END{print s+0}')
  echo "${used:-0} ${pend:-0}"
}

# ---------- 统计函数：分区内“拥有 H100 的唯一节点集合”的总卡数 ----------
sum_part_total_cards() {
  # 注意：节点通常同时挂在 gpu 和 gpu-invest 两个分区；
  # 这里统计“在该分区可见、且含 H100 的唯一节点集合”的总 H100 数（去重后按该分区视角统计）。
  local part="$1"
  sinfo -N -p "$part" -o "%N %G" \
    | awk '/h100/ {print $1,$2}' \
    | sort -u \
    | awk '{ if (match($2, /h100:([0-9]+)/, m)) sum += m[1] } END { print sum+0 }'
}

# ---------- 统计函数：全局（两个分区合并、按节点去重）的 H100 总卡数 ----------
sum_global_total_cards() {
  sinfo -N -p gpu,gpu-invest -o "%N %G" \
    | awk '/h100/ {print $1,$2}' \
    | sort -u \
    | awk '{ if (match($2, /h100:([0-9]+)/, m)) sum += m[1] } END { print sum+0 }'
}

# ---------- 读取节点的 H100 总量（从 GRES 字段） ----------
total_h100_on_node() {
  local node="$1"
  sinfo -N -n "$node" -o "%G" | awk 'NR==2' | grep -o 'h100:[0-9]\+' | awk -F: '{print $2}'
}

# ---------- 读取节点的 H100 已用：优先 scontrol 的 GresUsed=，失败回退到 squeue ----------
used_h100_on_node() {
  local node="$1"
  local used=""
  # 1) 尝试从 scontrol show node 的 GresUsed= 抓取
  local line
  line=$(scontrol show node "$node" 2>/dev/null | awk -F= '/GresUsed=/{print $2}' | head -n1 || true)
  if [[ -n "$line" ]]; then
    used=$(grep -o 'h100:[0-9]\+' <<<"$line" | awk -F: '{print $2}')
  fi
  # 2) 回退：统计该节点上“正在运行”的作业请求的 h100
  if [[ -z "${used:-}" ]]; then
    used=$(squeue -h -t R -w "$node" -o "%b" | grep -o 'h100:[0-9]\+' | awk -F: '{s+=$2} END{print s+0}')
  fi
  echo "${used:-0}"
}

# ---------- 打印 Top N 的 Pending 原因（带中文解释） ----------
print_top_reasons() {
  local part="$1"
  local title="$2"
  local n="${3:-5}"
  echo "---- ${title} 队列等待原因 ----"
  squeue -h -p "$part" -t PD -o "%R" \
    | sed 's/  */ /g' | sed 's/^ *//; s/ *$//' \
    | sort | uniq -c | sort -nr | head -"$n" \
    | while read -r cnt reason_rest; do
        reason="$(echo "$reason_rest" | tr -s ' ' )"
        # 常见原因的中文释义
        case "$reason" in
          "Dependency")                        zh="依赖其他作业（未完成前不会启动）" ;;
          "Priority")                          zh="优先级不足（被更高优先级作业压住）" ;;
          "QOSMaxGRESPerUser")                 zh="单用户 GPU 上限（对方超额，不会影响你拿卡）" ;;
          "QOSGrpGRES")                        zh="项目组 GPU 总量上限（投资组常见）" ;;
          "BeginTime")                         zh="指定了开始时间（还没到点）" ;;
          "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions")
                                                zh="节点不可用/维护/被高优先级预留" ;;
          *)                                   zh="其他/复合原因" ;;
        esac
        printf "  %4s (%s)  # %s\n" "$cnt" "$reason" "$zh"
      done
  echo
}

# ---------- 生成“含 H100 的唯一节点列表”（按分区视角） ----------
list_nodes_with_h100() {
  local part="$1"
  sinfo -N -p "$part" -o "%N %t %P %G" \
    | awk '/h100/ {print $1,$2,$3,$4}' \
    | sort -u
}

# ---------- 打印节点用量表 ----------
print_node_table() {
  # 说明：这里显示“在 gpu 分区可见的、含 H100 的节点”作为 F1 的参考；
  #       如果你也关心投资层节点，可以把 -p gpu 改成 -p gpu,gpu-invest。
  echo "---- 节点使用情况（以 gpu 分区为视角）----"
  printf "%-9s %-12s %-15s %-11s %-10s %-10s\n" "Node" "状态(State)" "分区(Partitions)" "总卡(Total)" "已用(Used)" "空闲(Free)"
  while read -r node state parts gres; do
    total=$(echo "$gres" | grep -o 'h100:[0-9]\+' | awk -F: '{print $2}')
    [[ -z "$total" ]] && total=0
    used=$(used_h100_on_node "$node")
    [[ -z "$used" ]] && used=0
    # 防溢出保护
    if ! [[ "$used" =~ ^[0-9]+$ ]]; then used=0; fi
    free=$(( total - used )); if (( free < 0 )); then free=0; fi
    printf "%-9s %-12s %-15s %-11s %-10s %-10s\n" "$node" "$state" "$parts" "$total" "$used" "$free"
  done < <(list_nodes_with_h100 gpu)
  echo
}

# ---------- 专门列出 F1 免费池中“空闲≥1”的节点 ----------
print_f1_free_nodes() {
  echo "F1 免费池可用节点（空闲≥1）："
  found=0
  while read -r node state parts gres; do
    total=$(echo "$gres" | grep -o 'h100:[0-9]\+' | awk -F: '{print $2}')
    [[ -z "$total" ]] && total=0
    used=$(used_h100_on_node "$node")
    [[ -z "$used" ]] && used=0
    free=$(( total - used ))
    if (( free >= 1 )); then
      echo "  - ${node}: Free=${free}（Total=${total}, Used=${used}, State=${state})"
      found=1
    fi
  done < <(list_nodes_with_h100 gpu)
  [[ $found -eq 0 ]] && echo "  （当前无空闲≥1的 F1 节点）"
  echo
}

# ---------- 主体：单次快照 ----------
snapshot() {
  local now; now=$(date '+%Y-%m-%d %H:%M:%S')
  echo "========== UBELIX H100 快照 @ ${now} =========="

  # 计算总卡数（去重按节点）
  local TOTAL_ALL F1_TOTAL INV_TOTAL
  TOTAL_ALL=$(sum_global_total_cards)   # 全部 H100 总卡数（一般是 40）
  F1_TOTAL=8                            # 免费池固定 8 张
  INV_TOTAL=$(( TOTAL_ALL - F1_TOTAL )) # 投资池剩余
  (( INV_TOTAL < 0 )) && INV_TOTAL=0

  # 分区使用：运行 / 等待
  read -r F1_USED  F1_PEND  < <(sum_part_usage gpu)
  read -r INV_USED INV_PEND < <(sum_part_usage gpu-invest)

  echo
  echo "F1（partition=gpu 免费层）："
  echo "    正在运行 H100 = ${F1_USED} / ${F1_TOTAL}    # 只在这 ${F1_TOTAL} 张卡里竞争"
  echo "    等待中 H100 ≈ ${F1_PEND}                     # 近似为等待作业请求的 H100 总数"

  echo
  echo "Investor（gpu-invest 投资/可抢占层）："
  echo "    正在运行 H100 = ${INV_USED} / ${INV_TOTAL}    # 投资者拥有的总量（可抢占）"
  echo "    等待中 H100 ≈ ${INV_PEND}"

  echo
  print_top_reasons gpu "F1"
  print_top_reasons gpu-invest "Investor"

  print_node_table
  print_f1_free_nodes

  echo "提示(Hint)："
  echo " - F1=免费层（gpu），建议短时长（--time 0:30:00~1:00:00）提升 backfill 机会。"
  echo " - Investor=gpu-invest 可抢占；若投资者在用则排队，空闲时可能分配但随时会被抢。"
  echo " - 你的 fairshare: $(sshare -U $USER 2>/dev/null | awk 'NR==3{print "FS=" $NF ", RawUsage=" $(NF-2)}')"
  echo
  echo "小技巧：当上方“F1 免费池可用节点”中出现 Free≥1 时，可尝试："
  echo "  srun --partition=gpu --qos=job_interactive --gres=gpu:h100:1 \\"
  echo "       --cpus-per-task=12 --mem=80G --time=00:30:00 --nodelist=<节点名> --pty bash"
}

# ---------- 主循环 ----------
if [[ $REFRESH -eq 1 ]]; then
  while true; do
    clear
    snapshot
    sleep 30
  done
else
  snapshot
fi
