#!/bin/bash
# TSLib 35 个 train 分数补导（SMD TimesNet s48 已完成），GPU1/GPU2 两路
P=/data/zenghaoyang/miniconda3/envs/tslib/bin/python
D=/data/modeluse/TS/vus_diag
dump_tsl() {
  $P $D/dump_tslib_train.py "$1" "$2" "$3" "$4" >> $D/tsl_train_dump.log 2>&1
}
(
  for ds in SMD PSM SWAT Exathlon; do
    for m in TimesNet DLinear KANAD; do
      for s in 48 2021 2022; do
        [ "$ds" = "SMD" ] && [ "$m" = "TimesNet" ] && [ "$s" = "48" ] && continue
        dump_tsl "$ds" "$m" "$s" 1
      done
    done
  done
) &
(
  $P $D/dump_istad_full.py SMD 48 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py SMD 2021 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py SMD 2022 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py SMD 2025 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py PSM 87 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py PSM 90 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py PSM 98 2 >> $D/istad_dump.log 2>&1
) &
(
  $P $D/dump_istad_full.py PSM 2021 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py EXA 48 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py EXA 89 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py EXA 2021 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py SWAT 89 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py SWAT 48 2 >> $D/istad_dump.log 2>&1
  $P $D/dump_istad_full.py SWAT 2021 2 >> $D/istad_dump.log 2>&1
) &
wait
echo ALL_DONE >> $D/tsl_train_dump.log
