#!/bin/bash
# SWAT RevIN 重训：3 种子顺序跑，占 GPU 0+1（DataParallel，复刻生产 2GPU 配方）
set -x
export PATH=/data/zenghaoyang/miniconda3/envs/tranad/bin:$PATH
cd /data/modeluse/TS/ISTAD || exit 1
for s in 48 89 2021; do
  echo "===== SWAT revin seed $s start $(date) ====="
  bash scripts/anomaly_detection/SWAT/ISTAD_revin_s${s}.sh
  echo "===== SWAT revin seed $s done $(date) exit=$? ====="
done
