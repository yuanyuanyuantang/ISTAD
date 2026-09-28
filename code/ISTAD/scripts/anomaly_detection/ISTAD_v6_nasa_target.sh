#!/usr/bin/env bash
set -euo pipefail

# Frozen seed-87 target-aware NASA development diagnostic.
# Usage: bash scripts/anomaly_detection/ISTAD_v6_nasa_target.sh DATASET ARM GPU [SEED] [train|eval]
# DATASET: MSL | SMAP; ARM: trecon | forecast

dataset="${1:?choose MSL or SMAP}"
arm="${2:?choose trecon or forecast}"
gpu="${3:-0}"
seed="${4:-87}"
run_mode="${5:-train}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_dir="$(cd "${project_dir}/../../.." && pwd)"
data_root="${ISTAD_DATA_ROOT:-${workspace_dir}/ISTAD/dataset}"
checkpoint_root="${ISTAD_CHECKPOINT_ROOT:-${project_dir}/../../checkpoints/ISTAD_v6_nasa_target}"

case "${dataset}" in
  MSL) channels=55 ;;
  SMAP) channels=25 ;;
  *)
    echo "Unsupported dataset: ${dataset}. Use MSL or SMAP." >&2
    exit 2
    ;;
esac

case "${arm}" in
  trecon) objective=target_reconstruct ;;
  forecast) objective=target_forecast ;;
  *)
    echo "Unsupported arm: ${arm}. Use trecon or forecast." >&2
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

cd "${project_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" python -u run.py \
  --task_name anomaly_detection --is_training "${is_training}" \
  --root_path "${data_root}/${dataset}" --checkpoints "${checkpoint_root}" \
  --model_id "${dataset}" --model ISTAD --data "${dataset}" --features M \
  --seq_len 100 --enc_in "${channels}" --c_out 1 --gpu 0 \
  --istad_branch_mode kan_tcn --istad_spatial_type legacy \
  --istad_recon_type kanad --istad_kanad_order 4 --istad_dropout 0.2 \
  --istad_kan_grid_size 10 --istad_kan_spline_order 3 \
  --istad_objective "${objective}" --istad_target_features 0 \
  --istad_forecast_lag 1 --istad_corrupt_target_only 1 \
  --istad_denoise 1 --istad_clean_lambda 1.0 --istad_denoise_lambda 0.5 \
  --istad_frequency_lambda 0.0 --istad_corrupt_modes 4 \
  --istad_evidence_head 0 --istad_evidence_lambda 0.0 \
  --istad_point_evidence_lambda 0.0 --istad_score_mode base_mean \
  --istad_score_aggregate mean --istad_score_topk 1 \
  --istad_corrupt_prob 0.8 --istad_corrupt_time_ratio 0.15 \
  --istad_corrupt_channel_ratio 1.0 --istad_corrupt_scale 1.5 \
  --istad_holdout_val 1 --istad_entity_aware 1 --istad_eval_step 100 \
  --istad_no_test_during_train 1 --istad_threshold_source train \
  --istad_dump_scores 1 --istad_enable_explain 0 \
  --use_bestf1_threshold 1 --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 200 --bestf1_fine_step_num 500 \
  --bestf1_use_adjustment 1 --anomaly_ratio 1 \
  --learning_rate 0.01 --lradj type1 --batch_size 128 --num_workers 4 \
  --patience 3 --train_epochs 3 --seed "${seed}" \
  --des "v6nasa_${arm}_s${seed}"
