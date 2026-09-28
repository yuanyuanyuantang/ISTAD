#!/bin/bash
# PSM dual λ 扫描驱动（prereg_psm_lambda.md）：8 run 顺序，GPU0，按种子交替两臂
cd /data/modeluse/TS/ISTAD || exit 1
for s in 87 90 98 2021; do
  for arm in l05 l20; do
    echo "===== START dual_${arm}_s${s} $(date) ====="
    conda run --no-capture-output -n tslib \
      bash scripts/anomaly_detection/PSM/ISTAD_dual_${arm}_s${s}.sh \
      > /data/modeluse/TS/vus_diag/dual_train/psm_lambda_${arm}_s${s}.log 2>&1
    echo "===== END dual_${arm}_s${s} exit=$? $(date) ====="
  done
done
echo "ALL PSM LAMBDA RUNS DONE $(date)"
