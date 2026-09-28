#!/bin/bash
# RevIN 变体分数导出（训练完成后运行）：3 GPU 并行，每 GPU 顺序处理各自 (ds,seed)。
# full = 全长 score/label/train_score（stride-1）；perfeat = 逐特征误差（诊断/聚合用）。
PY=/data/zenghaoyang/miniconda3/envs/tranad/bin/python
cd /data/modeluse/TS/vus_diag || exit 1

run_pair() {  # $1=ds $2=seed $3=gpu
  $PY dump_istad_full.py "$1" "$2" "$3" revin || exit 1
  $PY dump_istad_perfeat.py "$1" "$2" "$3" revin || exit 1
}

( run_pair SMD 48 0;    run_pair SMD 2021 0;  run_pair PSM 98 0 )  > revin_train/dump_g0.log 2>&1 &
( run_pair SMD 2022 1;  run_pair SMD 2025 1 )  > revin_train/dump_g1.log 2>&1 &
( run_pair SWAT 48 2;   run_pair SWAT 89 2;   run_pair SWAT 2021 2 ) > revin_train/dump_g2.log 2>&1 &
wait
echo "===== all dumps done $(date) ====="
ls -la istad_scores/*_revin.npz istad_perfeat/*_revin.npz
