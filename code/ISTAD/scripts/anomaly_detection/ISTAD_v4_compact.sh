#!/usr/bin/env bash
set -euo pipefail

# Confirmation candidate, frozen after the 2026-09-09 exploratory run.
# Usage: bash scripts/anomaly_detection/ISTAD_v4_compact.sh SMD 0 2021
# Run multiple seeds; do not report only the best seed.

dataset="${1:-SMD}"
gpu="${2:-0}"
seed="${3:-2021}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_dir="$(cd "${project_dir}/../../.." && pwd)"
data_root="${ISTAD_DATA_ROOT:-${workspace_dir}/ISTAD/dataset}"
checkpoint_root="${ISTAD_CHECKPOINT_ROOT:-${project_dir}/../../checkpoints/ISTAD_v4_compact}"

case "${dataset}" in
  SMD)
    model_id=SMD; data_name=SMD; data_dir=SMD; seq_len=96; channels=38
    batch_size=64; epochs=40; patience=6; learning_rate=0.01; lradj=type1
    dataset_args=(--istad_kanad_order 4 --istad_dropout 0.2 --istad_entity_aware 1)
    ;;
  PSM)
    model_id=PSM; data_name=PSM; data_dir=PSM; seq_len=64; channels=25
    batch_size=64; epochs=3; patience=3; learning_rate=0.01; lradj=type1
    dataset_args=(--istad_kanad_order 6 --istad_dropout 0.3 --istad_dual 1
      --istad_dual_lambda 1.0)
    ;;
  SWAT)
    model_id=SWAT; data_name=SWAT; data_dir=SWAT; seq_len=96; channels=51
    batch_size=32; epochs=20; patience=6; learning_rate=0.001; lradj=cosine
    dataset_args=(--istad_kanad_order 6 --istad_dropout 0.1 --istad_dual 1
      --istad_dual_lambda 1.0 --istad_kernel_size 15)
    ;;
  EXATHLON)
    model_id=EXATHLON; data_name=EXATHLON; data_dir=EXATHLON; seq_len=100; channels=19
    batch_size=64; epochs=40; patience=5; learning_rate=0.0001; lradj=cosine
    dataset_args=(--istad_kanad_order 4 --istad_dropout 0.1 --istad_dual 1
      --istad_dual_lambda 1.0)
    ;;
  *)
    echo "Unsupported dataset: ${dataset}. Use SMD, PSM, SWAT, or EXATHLON." >&2
    exit 2
    ;;
esac

cd "${project_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" python -u run.py \
  --task_name anomaly_detection --is_training 1 \
  --root_path "${data_root}/${data_dir}" --checkpoints "${checkpoint_root}" \
  --model_id "${model_id}" --model ISTAD --data "${data_name}" --features M \
  --seq_len "${seq_len}" --enc_in "${channels}" --c_out "${channels}" --gpu 0 \
  --istad_recon_type kanad --istad_branch_mode kan_tcn \
  --istad_kan_grid_size 10 --istad_kan_spline_order 3 \
  --istad_denoise 1 --istad_clean_lambda 1.0 --istad_denoise_lambda 0.5 \
  --istad_frequency_lambda 0.0 --istad_corrupt_modes 4 \
  --istad_evidence_head 1 --istad_evidence_lambda 0.2 \
  --istad_point_evidence_lambda 0.0 --istad_score_mode innovation_fused \
  --istad_score_aggregate calibrated --istad_score_topk 3 \
  --istad_evidence_transform prob --istad_innovation_lag 1 \
  --istad_innovation_ridge 0.01 --istad_innovation_pool auto \
  --istad_innovation_degenerate_cutoff 0.10 \
  --istad_innovation_scale_floor 0.10 --istad_innovation_fusion_weight 0.001 \
  --istad_corrupt_prob 0.8 --istad_corrupt_time_ratio 0.15 \
  --istad_corrupt_channel_ratio 0.2 --istad_corrupt_scale 1.5 \
  --istad_holdout_val 1 --istad_eval_step "${seq_len}" \
  --istad_no_test_during_train 1 --istad_threshold_source train \
  --istad_dump_scores 1 --istad_enable_explain 0 --use_bestf1_threshold 1 \
  --bestf1_search_mode adaptive --bestf1_coarse_step_num 200 \
  --bestf1_fine_step_num 500 --bestf1_use_adjustment 1 \
  --anomaly_ratio 1 --learning_rate "${learning_rate}" --lradj "${lradj}" \
  --batch_size "${batch_size}" --num_workers 4 --patience "${patience}" \
  --train_epochs "${epochs}" --seed "${seed}" --des "v4compact_s${seed}" \
  "${dataset_args[@]}"
