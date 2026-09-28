#!/usr/bin/env bash
set -euo pipefail

analysis_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "${analysis_dir}/.." && pwd)"
launcher="${release_root}/code/ISTAD/scripts/anomaly_detection/ISTAD_v4_hgat_e1.sh"
gpu="${1:-0}"

export ISTAD_E1_RANDOM_REPEATS="${ISTAD_E1_RANDOM_REPEATS:-100}"
for dataset in EXATHLON PSM SMD SWAT; do
  for seed in 87 90 98; do
    echo "=== E1 ${dataset} seed ${seed} ==="
    bash "${launcher}" "${dataset}" "${gpu}" "${seed}"
  done
done

python "${analysis_dir}/summarize_v4_hgat_e1.py"
final_dir="${analysis_dir}/e1_incidence_counterfactual/final"
archive="${analysis_dir}/ISTAD_E1_results_20260911.tar.gz"
tar -czf "${archive}" -C "${final_dir}" .
echo "E1 result package: ${archive}"

