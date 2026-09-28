#!/usr/bin/env bash
set -euo pipefail

# ISTAD V7.1 nested residual causal hypergraph.
# Usage: bash scripts/anomaly_detection/ISTAD_v71_residual_hypergraph.sh DATASET ARM GPU [SEED] [train|eval]
# ARM: full | no_prior | temporal_only

dataset="${1:?choose EXATHLON, PSM, SMD, SWAT, MSL, or SMAP}"
arm="${2:?choose full, no_prior, or temporal_only}"
gpu="${3:-0}"
seed="${4:-87}"
run_mode="${5:-train}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_dir="$(cd "${project_dir}/../../.." && pwd)"
data_root="${ISTAD_DATA_ROOT:-${workspace_dir}/ISTAD/dataset}"
checkpoint_root="${ISTAD_CHECKPOINT_ROOT:-${project_dir}/../../checkpoints/ISTAD_v71_residual_hypergraph}"

entity_args=()
case "${dataset}" in
  EXATHLON) enc_in=19; target_count=19; seq_len=100 ;;
  PSM) enc_in=25; target_count=25; seq_len=64 ;;
  SMD)
    enc_in=66; target_count=38; seq_len=96
    entity_args+=(--istad_entity_aware 1 --istad_entity_context 1)
    ;;
  SWAT) enc_in=51; target_count=51; seq_len=96 ;;
  MSL)
    enc_in=82; target_count=1; seq_len=100
    entity_args+=(--istad_entity_aware 1 --istad_entity_context 1)
    ;;
  SMAP)
    enc_in=78; target_count=1; seq_len=100
    entity_args+=(--istad_entity_aware 1 --istad_entity_context 1)
    ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac

case "${arm}" in
  full) use_hypergraph=1; prior_strength=1.0; prior_lambda=0.05 ;;
  no_prior) use_hypergraph=1; prior_strength=0.0; prior_lambda=0.0 ;;
  temporal_only) use_hypergraph=0; prior_strength=0.0; prior_lambda=0.0 ;;
  *) echo "Unsupported arm: ${arm}" >&2; exit 2 ;;
esac

case "${run_mode}" in
  train) is_training=1 ;;
  eval) is_training=0 ;;
  *) echo "Unsupported run mode: ${run_mode}. Use train or eval." >&2; exit 2 ;;
esac

targets="$(seq -s, 0 "$((target_count - 1))")"

cd "${project_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" python -u run.py \
  --task_name anomaly_detection --is_training "${is_training}" \
  --root_path "${data_root}/${dataset}" --checkpoints "${checkpoint_root}" \
  --model_id "${dataset}" --model ISTAD --data "${dataset}" --features M \
  --seq_len "${seq_len}" --enc_in "${enc_in}" --c_out "${target_count}" --gpu 0 \
  --istad_arch v71 --istad_branch_mode tcn --istad_tcn_type standard \
  --istad_recon_type pointwise --istad_recon_hid_dim 32 --istad_dropout 0.1 \
  --istad_objective target_forecast --istad_target_features "${targets}" \
  --istad_forecast_lag 1 --istad_denoise 0 --istad_evidence_head 0 \
  --istad_v7_relation_dim 16 --istad_v7_use_hypergraph "${use_hypergraph}" \
  --istad_v7_prior_ridge 0.01 \
  --istad_v7_prior_topk -1 --istad_v7_prior_strength "${prior_strength}" \
  --istad_v7_prior_floor 0.01 --istad_v7_prior_lambda "${prior_lambda}" \
  --istad_v7_relation_score_weight 0 \
  --istad_v71_graph_gate_init 0.05 --istad_v71_prior_gate_init 0.5 \
  --istad_v7_entity_calibration 1 "${entity_args[@]}" \
  --istad_score_mode v7_fused --istad_holdout_val 1 --istad_eval_step "${seq_len}" \
  --istad_no_test_during_train 1 --istad_threshold_source train \
  --istad_dump_scores 1 --istad_enable_explain 0 \
  --use_bestf1_threshold 1 --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 200 --bestf1_fine_step_num 500 \
  --bestf1_use_adjustment 1 --anomaly_ratio 1 \
  --learning_rate 0.005 --lradj type1 --batch_size 128 --num_workers 4 \
  --patience 3 --train_epochs 5 --seed "${seed}" \
  --des "v71_${arm}_s${seed}"

