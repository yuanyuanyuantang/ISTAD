#!/usr/bin/env bash
set -euo pipefail

# E2 audits the neural reconstruction path with one common reconstruction score.
# Usage:
#   bash scripts/anomaly_detection/ISTAD_e2_architecture_ablation.sh \
#     PSM 0 87 learned_dynamic train

dataset="${1:-PSM}"
gpu="${2:-0}"
seed="${3:-87}"
arm="${4:-learned_dynamic}"
run_mode="${5:-train}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${arm}" in
  learned_dynamic) relation_mode=dynamic ;;
  learned_static) relation_mode=static ;;
  fixed_random) relation_mode=fixed_random ;;
  no_message) relation_mode=no_message ;;
  *)
    echo "Unsupported E2 arm: ${arm}" >&2
    echo "Use learned_dynamic, learned_static, fixed_random, or no_message." >&2
    exit 2
    ;;
esac

case "${run_mode}" in
  train) is_training=1 ;;
  eval) is_training=0 ;;
  *)
    echo "Unsupported run mode: ${run_mode}. Use train or eval." >&2
    exit 2
    ;;
esac

case "${dataset}" in
  SMD)
    model_id=SMD; data_name=SMD; data_dir=SMD; seq_len=96; channels=38
    batch_size=128; epochs=20; patience=5; learning_rate=0.001; lradj=cosine
    dataset_args=(--istad_entity_aware 1)
    ;;
  PSM)
    model_id=PSM; data_name=PSM; data_dir=PSM; seq_len=64; channels=25
    batch_size=128; epochs=3; patience=3; learning_rate=0.001; lradj=type1
    dataset_args=()
    ;;
  SWAT)
    model_id=SWAT; data_name=SWAT; data_dir=SWAT; seq_len=96; channels=51
    batch_size=64; epochs=12; patience=4; learning_rate=0.001; lradj=cosine
    dataset_args=(--istad_kernel_size 15 --istad_n_hyperedges 20 --istad_k_top 10)
    ;;
  EXATHLON)
    model_id=EXATHLON; data_name=EXATHLON; data_dir=EXATHLON; seq_len=100; channels=19
    batch_size=128; epochs=20; patience=5; learning_rate=0.001; lradj=cosine
    dataset_args=()
    ;;
  *)
    echo "Unsupported dataset: ${dataset}. Use SMD, PSM, SWAT, or EXATHLON." >&2
    exit 2
    ;;
esac

project_dir="$(cd "${script_dir}/../.." && pwd)"
release_root="$(cd "${project_dir}/../.." && pwd)"
workspace_dir="$(cd "${project_dir}/../../.." && pwd)"
data_root="${ISTAD_DATA_ROOT:-${workspace_dir}/ISTAD/dataset}"
checkpoint_root="${ISTAD_E2_CHECKPOINT_ROOT:-${release_root}/checkpoints/ISTAD_e2_architecture_ablation}"
result_root="${ISTAD_E2_RESULT_ROOT:-${release_root}/analysis/e2_architecture_ablation/runtime}"

mkdir -p "${checkpoint_root}" "${result_root}"
export ISTAD_RESULT_ROOT="${result_root}"

cd "${project_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" python -u run.py \
  --task_name anomaly_detection --is_training "${is_training}" \
  --root_path "${data_root}/${data_dir}" --checkpoints "${checkpoint_root}" \
  --model_id "${model_id}" --model ISTAD --data "${data_name}" --features M \
  --seq_len "${seq_len}" --enc_in "${channels}" --c_out "${channels}" --gpu 0 \
  --istad_recon_type pointwise --istad_recon_hid_dim 32 \
  --istad_branch_mode hgat --istad_spatial_type lite --istad_hgat_rank 8 \
  --istad_hgat_projection linear --istad_hgat_relation_mode "${relation_mode}" \
  --istad_denoise 1 --istad_clean_lambda 1.0 --istad_denoise_lambda 0.5 \
  --istad_frequency_lambda 0.0 --istad_corrupt_modes 4 \
  --istad_evidence_head 0 --istad_score_mode base_mean \
  --istad_corrupt_prob 0.8 --istad_corrupt_time_ratio 0.15 \
  --istad_corrupt_channel_ratio 0.2 --istad_corrupt_scale 1.5 \
  --istad_holdout_val 1 --istad_eval_step "${seq_len}" \
  --istad_no_test_during_train 1 --istad_threshold_source train \
  --istad_dump_scores 1 --istad_enable_explain 0 --use_bestf1_threshold 1 \
  --bestf1_search_mode adaptive --bestf1_coarse_step_num 200 \
  --bestf1_fine_step_num 500 --bestf1_use_adjustment 0 \
  --anomaly_ratio 1 --learning_rate "${learning_rate}" --lradj "${lradj}" \
  --batch_size "${batch_size}" --num_workers 4 --patience "${patience}" \
  --train_epochs "${epochs}" --seed "${seed}" --des "e2_${arm}_s${seed}" \
  "${dataset_args[@]}"
