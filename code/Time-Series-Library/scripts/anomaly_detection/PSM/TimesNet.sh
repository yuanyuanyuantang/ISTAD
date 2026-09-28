export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --use_bestf1_threshold 0 \
  --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 50 \
  --bestf1_fine_step_num 100 \
  --bestf1_use_adjustment 1 \
  --batch_size 128 \
  --train_epochs 3