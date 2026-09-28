#!/bin/bash
# 每个 GPU 一条队列
cd /data/modeluse/TS/vus_diag
PY=/data/zenghaoyang/miniconda3/envs/tranad/bin/python
$PY dump_istad_perfeat.py SMD 48 0    > perfeat_smd48.log 2>&1
$PY dump_istad_perfeat.py EXA 48 0    > perfeat_exa48.log 2>&1
$PY dump_istad_perfeat.py SWAT 48 0   > perfeat_swat48.log 2>&1
$PY dump_istad_perfeat.py PSM 98 0    > perfeat_psm98.log 2>&1
/data/zenghaoyang/miniconda3/envs/tranad/bin/python dump_istad_perfeat.py SMD 2025 0  > perfeat_smd2025.log 2>&1
