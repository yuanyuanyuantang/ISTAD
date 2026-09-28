#!/bin/bash
# PSM λ 臂逐特征误差转储（prereg_psm_lambda.md）：8 npz，GPU2，顺序
cd /data/modeluse/TS/vus_diag || exit 1
for arm in l05 l20; do
  for s in 87 90 98 2021; do
    echo "===== dump PSM dual_${arm}_s${s} $(date) ====="
    conda run --no-capture-output -n tslib \
      python -u dump_istad_perfeat.py PSM ${s} 2 dual_${arm} \
      >> /data/modeluse/TS/vus_diag/dual_train/dump_psm_lambda_${arm}_s${s}.log 2>&1
    echo "===== dump PSM dual_${arm}_s${s} exit=$? $(date) ====="
  done
done
echo "ALL PSM LAMBDA DUMPS DONE $(date)"
