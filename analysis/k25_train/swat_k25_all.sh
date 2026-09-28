#!/bin/bash
# SWAT k25 三种子顺序驱动（预注册 vus_diag/prereg_swat_width.md）
cd /data/modeluse/TS/ISTAD
mkdir -p /data/modeluse/TS/vus_diag/k25_train
for s in 48 89 2021; do
  echo "=== k25 s$s start $(date '+%F %T') ==="
  bash scripts/anomaly_detection/SWAT/ISTAD_k25_s${s}.sh \
    > /data/modeluse/TS/vus_diag/k25_train/s${s}.log 2>&1
  echo "=== k25 s$s exit $? $(date '+%F %T') ==="
done
echo "ALL_DONE $(date '+%F %T')"
