#!/bin/bash
# SMD RevIN 重训：4 种子顺序跑，占 GPU 2（单卡，复刻生产配方）
set -x
export PATH=/data/zenghaoyang/miniconda3/envs/tranad/bin:$PATH
cd /data/modeluse/TS/ISTAD || exit 1
for s in 48 2021 2022 2025; do
  echo "===== SMD revin seed $s start $(date) ====="
  bash scripts/anomaly_detection/SMD/ISTAD_revin_s${s}.sh
  echo "===== SMD revin seed $s done $(date) exit=$? ====="
done
