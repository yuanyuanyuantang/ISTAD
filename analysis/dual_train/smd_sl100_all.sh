#!/bin/bash
# SMD dual sl100 驱动（prereg_smd_sl100.md）：4 run 顺序，GPU1
cd /data/modeluse/TS/ISTAD || exit 1
for s in 48 2021 2022 2025; do
  echo "===== START smd_dual_sl100_s${s} $(date) ====="
  conda run --no-capture-output -n tslib \
    bash scripts/anomaly_detection/SMD/ISTAD_dual_sl100_s${s}.sh \
    > /data/modeluse/TS/vus_diag/dual_train/smd_sl100_s${s}.log 2>&1
  echo "===== END smd_dual_sl100_s${s} exit=$? $(date) ====="
done
echo "ALL SMD SL100 RUNS DONE $(date)"
