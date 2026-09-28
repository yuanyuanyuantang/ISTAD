#!/usr/bin/env python3
"""TSLib 基线：为已有 test_scores.npy 的 run 补导训练集分数 train_scores.npy
（POT 需要训练集分数定 init 阈值）。加载 checkpoint → forward train_loader → 保存。
用法: python dump_tslib_train.py <DS> <MODEL> <SEED> <GPU>
"""
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

DS, MODEL, SEED, GPU = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU)

TSL = '/data/modeluse/TS/Time-Series-Library'
os.chdir(TSL)
sys.path.insert(0, TSL)

from exp.exp_anomaly_detection import Exp_Anomaly_Detection  # noqa: E402
from utils.print_args import print_args  # noqa: E402

# run.py 默认值（.sh 未写明的键）
DEFAULTS = dict(label_len=48, pred_len=96, d_model=512, n_heads=8, e_layers=2,
                d_layers=1, d_ff=2048, expand=2, d_conv=4, factor=1,
                embed='timeF', distil='True', top_k=5, num_kernels=6,
                freq='h', dropout=0.1, activation='gelu', moving_avg=25,
                decomp_method='moving_avg', use_norm=1, channel_independence=1,
                individual=False, seg_len=96, patch_len=16, pos=1,
                p_hidden_dims=[128, 128], p_hidden_layers=2,
                down_sampling_layers=0, down_sampling_window=1,
                down_sampling_method=None, augmentation_ratio=0)
DS_ROOT = {'Exathlon': './dataset/EXATHLON', 'PSM': './dataset/PSM',
           'SMD': './dataset/SMD', 'SWAT': './dataset/SWAT'}
DS_DATA = {'Exathlon': 'EXATHLON', 'PSM': 'PSM', 'SMD': 'SMD', 'SWAT': 'SWAT'}


def sh_args(model, ds):
    """从 .sh 提取键值（排除 CUDA export 与 python -u run.py 行）。"""
    p = f'{TSL}/scripts/anomaly_detection/{ds}/{model}.sh'
    out = {}
    for line in open(p):
        m = re.match(r'\s*--(\w+)\s+(.+?)\s*\\\s*$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def main():
    ds_tsl = {'Exathlon': 'Exathlon', 'PSM': 'PSM', 'SMD': 'SMD', 'SWAT': 'SWAT'}[DS]
    script = sh_args(MODEL, ds_tsl)
    model_id = script.get('model_id', DS)
    data = script.get('data', DS_DATA[DS])
    sl = int(script.get('seq_len', 96))

    # 定位现存 setting 目录（_ms{seed}_0）；setting 顺序 = task_{model_id}_{model}_{data}
    pats = [s for s in os.listdir(f'{TSL}/test_results')
            if re.fullmatch(rf'anomaly_detection_{model_id}_{MODEL}_{data}_ftM_sl{sl}_.*_ms{SEED}_0', s)]
    assert len(pats) == 1, f'{DS} {MODEL} {SEED}: {pats}'
    setting = pats[0]

    args = type('A', (), {})()
    for k, v in DEFAULTS.items():
        setattr(args, k, v)
    for k, v in script.items():
        if k in ('export',):
            continue
        cur = getattr(args, k, None)
        if isinstance(cur, int):
            setattr(args, k, int(v))
        elif isinstance(cur, float):
            setattr(args, k, float(v))
        else:
            setattr(args, k, v)
    args.task_name = 'anomaly_detection'
    args.model = MODEL
    args.model_id = model_id
    args.data = data
    args.root_path = script.get('root_path', DS_ROOT[DS])
    args.features = 'M'
    args.seq_len = sl
    args.enc_in = int(script['enc_in'])
    args.c_out = int(script['c_out'])
    args.batch_size = int(script.get('batch_size', 128))
    args.anomaly_ratio = float(script.get('anomaly_ratio', 1))
    args.use_gpu = True
    args.gpu = 0
    args.gpu_type = 'cuda'
    args.use_multi_gpu = False
    args.devices = '0'
    args.device_ids = [0]
    args.num_workers = 2
    args.checkpoints = './checkpoints/'
    args.device = torch.device('cuda:0')

    ckpt = f'./checkpoints/{setting}/checkpoint.pth'
    assert os.path.exists(ckpt), f'no ckpt: {ckpt}'

    exp = Exp_Anomaly_Detection(vars(args) if not hasattr(args, '__dict__') else args)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    train_set, train_loader = exp._get_data(flag='train')
    criterion = nn.MSELoss(reduce=False)
    scores = []
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.float().to(exp.device)
            outputs = exp.model(batch_x, None, None, None)
            scores.append(torch.mean(criterion(batch_x, outputs), dim=-1).cpu().numpy())
    sc = np.concatenate(scores, axis=0).reshape(-1)
    out = f'{TSL}/test_results/{setting}/train_scores.npy'
    np.save(out, sc)
    print(f'[train-dump] {DS} {MODEL} s{SEED}  n={sc.shape[0]}  -> {out}', flush=True)


if __name__ == '__main__':
    main()
