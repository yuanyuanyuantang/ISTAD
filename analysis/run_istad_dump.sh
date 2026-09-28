#!/bin/bash
# 13 个 ISTAD 全长分数导出，GPU1/GPU2 两路并行（各路串行）
P=/data/zenghaoyang/miniconda3/envs/tslib/bin/python
D=/data/modeluse/TS/vus_diag
run_one() {
  local ds=$1 seed=$2 gpu=$3
  $P $D/dump_istad_full.py "$ds" "$seed" "$gpu" >> $D/istad_dump.log 2>&1
}
(
  run_one PSM 87 1
  run_one PSM 90 1
  run_one PSM 98 1
  run_one PSM 2021 1
  run_one EXA 48 1
  run_one EXA 89 1
  run_one EXA 2021 1
) &
(
  run_one SMD 2021 2
  run_one SMD 2022 2
  run_one SMD 2025 2
  run_one SWAT 89 2
  run_one SWAT 48 2
  run_one SWAT 2021 2
) &
wait
echo ALL_DONE >> $D/istad_dump.log
