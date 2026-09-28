#!/bin/bash
# SMD dual sl100 逐特征误差转储（prereg_smd_sl100.md）：4 npz，GPU2，顺序
cd /data/modeluse/TS/vus_diag || exit 1
for s in 48 2021 2022 2025; do
  echo "===== dump SMD dual_sl100_s${s} $(date) ====="
  conda run --no-capture-output -n tslib \
    python -u dump_istad_perfeat.py SMD ${s} 2 dual_sl100 \
    >> /data/modeluse/TS/vus_diag/dual_train/dump_smd_sl100_s${s}.log 2>&1
  echo "===== dump SMD dual_sl100_s${s} exit=$? $(date) ====="
done
echo "ALL SMD SL100 DUMPS DONE $(date)"
