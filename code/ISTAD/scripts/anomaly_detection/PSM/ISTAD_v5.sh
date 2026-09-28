export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM_v5 \
  --model ISTAD \
  --data PSM \
  --features M \
  --seq_len 64 \
  --enc_in 25 \
  --c_out 25 \
  --gpu 0 \
  --istad_recon_type kanad \
  --istad_branch_mode hgat_kan_tcn \
  --istad_kanad_order 6 \
  --anomaly_ratio 1 \
  --use_bestf1_threshold 1 \
  --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 200 \
  --bestf1_fine_step_num 500 \
  --bestf1_use_adjustment 1 \
  --istad_dropout 0.3 \
  --learning_rate 0.01 \
  --batch_size 64 \
  --num_workers 4 \
  --patience 5 \
  --train_epochs 5 \
  --seed 87
