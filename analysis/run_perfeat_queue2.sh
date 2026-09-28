#!/bin/bash
cd /data/modeluse/TS/vus_diag
PY=/data/zenghaoyang/miniconda3/envs/tranad/bin/python
$PY dump_istad_perfeat.py SMD 2021 1  > perfeat_smd2021.log 2>&1
$PY dump_istad_perfeat.py EXA 89 1    > perfeat_exa89.log 2>&1
$PY dump_istad_perfeat.py SWAT 89 1   > perfeat_swat89.log 2>&1
