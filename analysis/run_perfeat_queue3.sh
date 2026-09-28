#!/bin/bash
cd /data/modeluse/TS/vus_diag
PY=/data/zenghaoyang/miniconda3/envs/tranad/bin/python
$PY dump_istad_perfeat.py SMD 2022 2  > perfeat_smd2022.log 2>&1
$PY dump_istad_perfeat.py EXA 2021 2  > perfeat_exa2021.log 2>&1
$PY dump_istad_perfeat.py SWAT 2021 2 > perfeat_swat2021.log 2>&1
