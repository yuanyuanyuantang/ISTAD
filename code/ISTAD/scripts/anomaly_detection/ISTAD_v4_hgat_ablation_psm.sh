#!/usr/bin/env bash
set -euo pipefail

# Frozen PSM three-arm HGAT ablation.
# Usage: bash scripts/anomaly_detection/ISTAD_v4_hgat_ablation_psm.sh ARM GPU SEED [train|eval] [RECALIBRATE]
# ARM: legacy | lite | none

arm="${1:?choose legacy, lite, or none}"
gpu="${2:-0}"
seed="${3:-90}"
run_mode="${4:-train}"
recalibrate="${5:-0}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_dir="$(cd "${project_dir}/../../.." && pwd)"
data_root="${ISTAD_DATA_ROOT:-${workspace_dir}/ISTAD/dataset}"
checkpoint_root="${ISTAD_CHECKPOINT_ROOT:-${project_dir}/../../checkpoints/ISTAD_v4_hgat_ablation}"

case "${run_mode}" in
  train) is_training=1 ;;
  eval) is_training=0 ;;
  *)
    echo "Unsupported run mode: ${run_mode}. Use train or eval." >&2
    exit 2
    ;;
esac
if [[ "${recalibrate}" != "0" && "${recalibrate}" != "1" ]]; then
  echo "RECALIBRATE must be 0 or 1." >&2
  exit 2
fi

case "${arm}" in
  legacy)
    architecture_args=(--istad_branch_mode hgat_kan_tcn --istad_spatial_type legacy)
    ;;
  lite)
    architecture_args=(--istad_branch_mode hgat_kan_tcn --istad_spatial_type lite
      --istad_hgat_rank 16)
    ;;
  none)
    architecture_args=(--istad_branch_mode kan_tcn --istad_spatial_type legacy)
    ;;
  *)
    echo "Unsupported arm: ${arm}. Use legacy, lite, or none." >&2
    exit 2
    ;;
esac

cd "${project_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" python -u run.py \
  --task_name anomaly_detection --is_training "${is_training}" \
  --root_path "${data_root}/PSM" --checkpoints "${checkpoint_root}" \
  --model_id PSM --model ISTAD --data PSM --features M \
  --seq_len 64 --enc_in 25 --c_out 25 --gpu 0 \
  --istad_recon_type kanad --istad_kanad_order 6 --istad_dropout 0.3 \
  --istad_dual 1 --istad_dual_lambda 1.0 \
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
  --istad_fusion_recalibrate "${recalibrate}" \
  --istad_corrupt_prob 0.8 --istad_corrupt_time_ratio 0.15 \
  --istad_corrupt_channel_ratio 0.2 --istad_corrupt_scale 1.5 \
  --istad_holdout_val 1 --istad_eval_step 64 --istad_no_test_during_train 1 \
  --istad_threshold_source train --istad_dump_scores 1 --istad_enable_explain 0 \
  --use_bestf1_threshold 1 --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 200 --bestf1_fine_step_num 500 \
  --bestf1_use_adjustment 1 --anomaly_ratio 1 \
  --learning_rate 0.01 --lradj type1 --batch_size 64 --num_workers 4 \
  --patience 3 --train_epochs 3 --seed "${seed}" --des "hgatabl_${arm}_s${seed}" \
  "${architecture_args[@]}"
