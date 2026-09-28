export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model ISTAD \
  --data SMD \
  --features M \
  --seq_len 96 \
  --enc_in 38 \
  --c_out 38 \
  --istad_recon_type kanad \
  --istad_branch_mode hgat_kan_tcn \
  --use_bestf1_threshold 1 \
  --bestf1_coarse_step_num 200 \
  --bestf1_fine_step_num 500 \
  --bestf1_use_adjustment 1 \
  --istad_dropout 0.2 \
  --istad_kanad_order 4 \
  --anomaly_ratio 1 \
  --learning_rate 0.01 \
  --batch_size 64 \
  --num_workers 4 \
  --patience 5 \
  --train_epochs 100 \
  --seed 48  

#48：
# 最终最佳阈值: 2.239964
# 最佳 F1: 0.8698
# Precision: 0.8913
# Recall: 0.8494

