#!/bin/bash
# 剩余 9 个重活重新分摊到 3 GPU（GPU0 与冒烟并行，内存充足）
P=/data/zenghaoyang/miniconda3/envs/tranad/bin/python
D=/data/modeluse/TS/vus_diag
dump_c() {
  local out=$D/classic_scores/$1_$2.npz
  if [ -f "$out" ]; then echo "[skip] $1 $2 (exists)" >> $D/classic_dump.log; return; fi
  $P -u $D/dump_classic.py "$1" "$2" "$3" >> $D/classic_dump.log 2>&1
  echo "[done rc=$?] $1 $2 GPU$3" >> $D/classic_dump.log
}
( for spec in "MTAD_GAT PSM" "GDN SMD" "DAGMM SMD" "MAD_GAN SMD"; do
    set -- $spec; dump_c "$1" "$2" 1; done ) &
( for spec in "MTAD_GAT Exathlon" "GDN SWaT" "DAGMM PSM" "TranAD SWaT"; do
    set -- $spec; dump_c "$1" "$2" 2; done ) &
( dump_c "MTAD_GAT" "SWaT" 0 ) &
wait
echo ALL_DONE2 >> $D/classic_dump.log
