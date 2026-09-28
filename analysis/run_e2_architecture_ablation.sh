#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash analysis/run_e2_architecture_ablation.sh smoke 0
#   bash analysis/run_e2_architecture_ablation.sh full 0

scope="${1:-smoke}"
gpu="${2:-0}"
analysis_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "${analysis_dir}/.." && pwd)"
launcher="${release_root}/code/ISTAD/scripts/anomaly_detection/ISTAD_e2_architecture_ablation.sh"
result_root="${ISTAD_E2_RESULT_ROOT:-${analysis_dir}/e2_architecture_ablation/runtime}"
checkpoint_root="${ISTAD_E2_CHECKPOINT_ROOT:-${release_root}/checkpoints/ISTAD_e2_architecture_ablation}"
force_rerun="${ISTAD_E2_FORCE_RERUN:-0}"
arms=(learned_dynamic learned_static fixed_random no_message)

case "${scope}" in
  smoke)
    datasets=(PSM)
    seeds=(87)
    ;;
  full)
    datasets=(EXATHLON PSM SMD SWAT)
    seeds=(87 90 98)
    ;;
  *)
    echo "Unsupported scope: ${scope}. Use smoke or full." >&2
    exit 2
    ;;
esac

mkdir -p "${result_root}" "${checkpoint_root}"
export ISTAD_E2_RESULT_ROOT="${result_root}"
export ISTAD_E2_CHECKPOINT_ROOT="${checkpoint_root}"

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for arm in "${arms[@]}"; do
      score_file="$(find "${result_root}" -type f \
        -path "*/anomaly_detection_${dataset}_*_e2_${arm}_s${seed}_0/point_scores.npz" \
        -print -quit)"
      checkpoint_file="$(find "${checkpoint_root}" -type f \
        -path "*/anomaly_detection_${dataset}_*_e2_${arm}_s${seed}_0/checkpoint.pth" \
        -print -quit)"
      if [[ "${force_rerun}" != "1" && -n "${score_file}" && -n "${checkpoint_file}" ]]; then
        echo "=== E2 skip completed ${dataset} seed ${seed} ${arm} ==="
        continue
      fi
      echo "=== E2 ${dataset} seed ${seed} ${arm} ==="
      bash "${launcher}" "${dataset}" "${gpu}" "${seed}" "${arm}" train
    done
  done
done

output_dir="${analysis_dir}/e2_architecture_ablation/final_${scope}"
python "${analysis_dir}/summarize_e2_architecture_ablation.py" \
  --result-root "${result_root}" \
  --checkpoint-root "${checkpoint_root}" \
  --output-dir "${output_dir}" \
  --scope "${scope}"

archive="${analysis_dir}/ISTAD_E2_${scope}_results_20260912.tar.gz"
tar -czf "${archive}" -C "${output_dir}" .
echo "E2 ${scope} result package: ${archive}"
