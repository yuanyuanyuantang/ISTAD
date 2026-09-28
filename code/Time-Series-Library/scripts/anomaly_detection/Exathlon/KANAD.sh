export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/Exathlon \
  --model_id Exathlon \
  --model KANAD \
  --data EXATHLON \
  --features M \
  --seq_len 96 \
  --d_model 4 \
  --enc_in 19 \
  --c_out 19 \
  --anomaly_ratio 1 \
  --use_bestf1_threshold 0 \
  --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 100 \
  --bestf1_fine_step_num 150 \
  --bestf1_use_adjustment 1 \
  --learning_rate 0.01 \
  --batch_size 128 \
  --num_workers 4 \
  --patience 5 \
  --train_epochs 100