#!/bin/bash
# seq_len 试点双轨驱动（prereg_sl100.md）
# Track 1: SWAT sl100 ×3 种子，GPU 0,1（双卡 bs128），顺序执行
# Track 2: PSM dual sl100 ×4 种子，GPU 2（单卡 bs64），顺序执行
# 两轨并行，wait 汇合后统一退出。
cd /data/modeluse/TS/ISTAD
LOG=/data/modeluse/TS/vus_diag/sl100_train

track_swat() {
  for s in 48 89 2021; do
    echo "===== SWAT sl100 s${s} start $(date +%T) ====="
    bash scripts/anomaly_detection/SWAT/ISTAD_sl100_s${s}.sh > ${LOG}/swat_sl100_s${s}.log 2>&1 \
      && echo "===== SWAT sl100 s${s} exit 0 $(date +%T) =====" \
      || echo "===== SWAT sl100 s${s} EXIT $? $(date +%T) ====="
  done
  echo "SWAT_TRACK_DONE"
}

track_psm() {
  for s in 87 90 98 2021; do
    echo "===== PSM dual_sl100 s${s} start $(date +%T) ====="
    bash scripts/anomaly_detection/PSM/ISTAD_dual_sl100_s${s}.sh > ${LOG}/psm_sl100_s${s}.log 2>&1 \
      && echo "===== PSM dual_sl100 s${s} exit 0 $(date +%T) =====" \
      || echo "===== PSM dual_sl100 s${s} EXIT $? $(date +%T) ====="
  done
  echo "PSM_TRACK_DONE"
}

track_swat &
P1=$!
track_psm &
P2=$!
wait $P1; S1=$?
wait $P2; S2=$?
echo "ALL_DONE swat=$S1 psm=$S2"
