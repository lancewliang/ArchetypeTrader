#!/usr/bin/env bash
# ============================================================
# ArchetypeTrader 全流程训练 + 评估脚本
# Phase I → Phase II → Phase III → Evaluate
# 日志同时输出到终端和文件
# 任一阶段失败则立即终止
# ============================================================
set -euo pipefail

# --- 设置 PYTHONPATH 为项目根目录 ---
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")" && pwd)"

# --- 配置 ---
PAIR="${1:-AL}"  
BATCH_ID="${2:-batch_001}"                        # 默认 ETH，可通过第一个参数指定
LOG_DIR="logs/${PAIR}/${BATCH_ID}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${PAIR}_pipeline_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

# 将所有输出同时写入终端和日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================"
echo " ArchetypeTrader Pipeline — ${PAIR}"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 日志文件: ${LOG_FILE}"
echo "========================================================"

# 传递额外参数（跳过 PAIR 和 BATCH_ID 两个位置参数）
EXTRA_ARGS=("${@:3}")

has_extra_arg() {
  local flag="$1"
  local arg
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

PHASE1_ARGS=()
if ! has_extra_arg "--train-file"; then
  PHASE1_ARGS+=(--train-file "data/${PAIR}/train.feather")
fi
if ! has_extra_arg "--val-file"; then
  PHASE1_ARGS+=(--val-file "data/${PAIR}/val.feather")
fi
if ! has_extra_arg "--test-file"; then
  PHASE1_ARGS+=(--test-file "data/${PAIR}/test.feather")
fi

# --- Phase I: Archetype Discovery ---
echo ""
echo ">>> [Phase I] 开始训练 — $(date '+%H:%M:%S')"
python scripts/train_phase1.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" "${PHASE1_ARGS[@]}" "${EXTRA_ARGS[@]}"
echo ">>> [Phase I] 完成 — $(date '+%H:%M:%S')"

# --- Phase II: Archetype Selection ---
echo ""
echo ">>> [Phase II] 开始训练 — $(date '+%H:%M:%S')"
python scripts/train_phase2.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" "${EXTRA_ARGS[@]}"
echo ">>> [Phase II] 完成 — $(date '+%H:%M:%S')"

# --- Phase II 后评估: val + test（无 Phase III，用 base actions 直接执行）---
echo ""
echo ">>> [Evaluate after Phase II] val + test — $(date '+%H:%M:%S')"
python scripts/evaluate.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" --split val test --stage-label phase2_eval --with-dp "${EXTRA_ARGS[@]}"
echo ">>> [Evaluate after Phase II] 完成 — $(date '+%H:%M:%S')"

# --- Phase III: Archetype Refinement ---
# echo ""
# echo ">>> [Phase III] 开始训练 — $(date '+%H:%M:%S')"
# python scripts/train_phase3.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" "${EXTRA_ARGS[@]}"
# echo ">>> [Phase III] 完成 — $(date '+%H:%M:%S')"

# --- Evaluate ---
# echo ""
# echo ">>> [Evaluate after Phase III] val + test — $(date '+%H:%M:%S')"
# python scripts/evaluate.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" --split val test --stage-label phase3_eval "${EXTRA_ARGS[@]}"
# echo ">>> [Evaluate] 完成 — $(date '+%H:%M:%S')"

echo ""
echo "========================================================"
echo " 全流程完成: ${PAIR}"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 日志文件: ${LOG_FILE}"
echo "========================================================"
