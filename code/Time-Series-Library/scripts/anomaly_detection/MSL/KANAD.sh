export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model KANAD \
  --data MSL \
  --features M \
  --seq_len 64 \
  --d_model 3 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --use_bestf1_threshold 1 \
  --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 50 \
  --bestf1_fine_step_num 100 \
  --bestf1_use_adjustment 1 \
  --learning_rate 0.01 \
  --batch_size 128 \
  --num_workers 4 \
  --patience 5 \
  --train_epochs 100