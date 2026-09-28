export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model TimesNet \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 16 \
  --e_layers 1 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --use_bestf1_threshold 1 \
  --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 50 \
  --bestf1_fine_step_num 100 \
  --bestf1_use_adjustment 1 \
  --batch_size 128 \
  --train_epochs 1