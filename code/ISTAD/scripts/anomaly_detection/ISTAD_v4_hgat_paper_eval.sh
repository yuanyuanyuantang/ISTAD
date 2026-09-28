#!/usr/bin/env bash
set -euo pipefail

# Regenerate float64 score artifacts from frozen checkpoints, then build all
# paper-facing metrics. No training and no hyperparameter selection occurs.
gpu="${1:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"

for seed in 87 90 98; do
  for dataset in EXATHLON PSM SMD SWAT; do
    bash "${script_dir}/ISTAD_v4_hgat_integrated.sh" "${dataset}" "${gpu}" "${seed}" eval
  done
done

cd "${repo_dir}"
python analysis/evaluate_v4_hgat_integrated.py
python analysis/benchmark_v4_hgat_efficiency.py --device "cuda:${gpu}" --warmup 30 --repeats 100
python analysis/evaluate_v4_hgat_paper.py --random-tie-repeats 100
