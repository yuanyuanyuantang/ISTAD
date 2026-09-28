#!/bin/bash
set -e
cd /data/modeluse/TS/ISTAD
for s in 48 89 2021; do
  echo "===== SWAT dual s$s START $(date) ====="
  bash scripts/anomaly_detection/SWAT/ISTAD_dual_s$s.sh
  echo "===== SWAT dual s$s DONE $(date) ====="
done
