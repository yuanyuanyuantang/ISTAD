#!/bin/bash
# classic-7 全 28 run 分数补导（MTAD_GAT SMD 冒烟已单跑，GPU0 链等它结束后接力），三路并行
P=/data/zenghaoyang/miniconda3/envs/tranad/bin/python
D=/data/modeluse/TS/vus_diag
SMOKE_PID=${1:-111860}
dump_c() {
  $P -u $D/dump_classic.py "$1" "$2" "$3" >> $D/classic_dump.log 2>&1
  echo "[done rc=$?] $1 $2 GPU$3" >> $D/classic_dump.log
}

# GPU0：等冒烟进程结束后接力（重活：MTAD_GAT×3 + GDN×2）
(
  while kill -0 "$SMOKE_PID" 2>/dev/null; do sleep 20; done
  for spec in "MTAD_GAT PSM" "MTAD_GAT SWaT" "MTAD_GAT Exathlon" "GDN SMD" "GDN SWaT" \
              "DAGMM SMD" "DAGMM PSM" "MAD_GAN SMD" "TranAD SWaT"; do
    set -- $spec; dump_c "$1" "$2" 0
  done
) &

# GPU1：LSTM_AD×4 + OmniAnomaly×4 + TranAD SMD（循环型模型）
(
  for m in LSTM_AD OmniAnomaly; do
    for ds in SMD PSM SWaT Exathlon; do dump_c "$m" "$ds" 1; done
  done
  dump_c TranAD SMD 1
) &

# GPU2：GDN×2 + MAD_GAN×3 + DAGMM×2 + TranAD×2
(
  for spec in "GDN PSM" "GDN Exathlon" "MAD_GAN PSM" "MAD_GAN SWaT" "MAD_GAN Exathlon" \
              "DAGMM SWaT" "DAGMM Exathlon" "TranAD PSM" "TranAD Exathlon"; do
    set -- $spec; dump_c "$1" "$2" 2
  done
) &

wait
echo ALL_DONE >> $D/classic_dump.log
