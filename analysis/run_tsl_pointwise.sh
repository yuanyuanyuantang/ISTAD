#!/bin/bash
# TSLib 3 模型 × 4 数据集 × 3 seeds = 36 run 逐点分数补导，三卡并行（12/卡），已存在则跳过
P=/data/zenghaoyang/miniconda3/envs/tslib/bin/python
D=/data/modeluse/TS/vus_diag
LOG=$D/tsl_pointwise_dump.log

run_one() {
  local ds=$1 model=$2 seed=$3 gpu=$4
  local out=$D/tsl_pointwise/${model}_${ds}_s${seed}.npz
  if [ -f "$out" ]; then echo "[skip] $model $ds s$seed" >> $LOG; return; fi
  $P -u $D/dump_tsl_pointwise.py "$ds" "$model" "$seed" "$gpu" >> $LOG 2>&1
  echo "[done rc=$?] $model $ds s$seed GPU$gpu" >> $LOG
}

SEEDS="48 2021 2022"

# GPU0: Exathlon×9 + PSM TimesNet×3
(
  for m in TimesNet DLinear KANAD; do for s in $SEEDS; do run_one Exathlon $m $s 0; done; done
  for s in $SEEDS; do run_one PSM TimesNet $s 0; done
) &

# GPU1: PSM DLinear/KANAD×6 + SMD TimesNet/DLinear×6
(
  for m in DLinear KANAD; do for s in $SEEDS; do run_one PSM $m $s 1; done; done
  for m in TimesNet DLinear; do for s in $SEEDS; do run_one SMD $m $s 1; done; done
) &

# GPU2: SMD KANAD×3 + SWAT×9
(
  for s in $SEEDS; do run_one SMD KANAD $s 2; done
  for m in TimesNet DLinear KANAD; do for s in $SEEDS; do run_one SWAT $m $s 2; done; done
) &

wait
echo ALL_DONE >> $LOG
