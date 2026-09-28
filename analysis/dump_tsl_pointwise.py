#!/usr/bin/env python3
"""TSLib 三基线逐点分数补导（统一口径修复）：

背景：Time-Series-Library 的 loader 对 PSM/MSL/SMAP 硬编码 step=1，对 SMD/EXA/SWAT
step=100——导致 PSM 的 test_scores.npy 是 stride-1 全窗口展开（每点被打 ~win 次分，
序列长 n×win），KANAD（sl≠100）则是抽稀/有洞序列。与 ISTAD/classic 的逐点分数
（每点恰好一次）不可比。本脚本统一用 step=seq_len 平铺窗口（每点恰好一次，尾部
不足一窗舍弃，与现有 SMD/SWAT TimesNet/DLinear 的 step=100 npy 同规则）重导
test/train 逐点分数。

回归校验：SMD/SWAT 的 TimesNet/DLinear（原本就是 step=100=sl 平铺）新导出应与
已有 test_scores.npy 逐位一致。

用法: python dump_tsl_pointwise.py <DS> <MODEL> <SEED> <GPU>
DS ∈ {Exathlon, PSM, SMD, SWAT}; MODEL ∈ {TimesNet, DLinear, KANAD}
输出: vus_diag/tsl_pointwise/{MODEL}_{DS}_s{SEED}.npz (score/label/train_score)
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
import data_provider.data_loader as DL  # noqa: E402

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
LOADER = {'Exathlon': 'EXATHLONSegLoader', 'PSM': 'PSMSegLoader',
          'SMD': 'SMDSegLoader', 'SWAT': 'SWATSegLoader'}


def sh_args(model, ds):
    p = f'{TSL}/scripts/anomaly_detection/{ds}/{model}.sh'
    out = {}
    for line in open(p):
        m = re.match(r'\s*--(\w+)\s+(.+?)\s*\\\s*$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def main():
    script = sh_args(MODEL, DS)
    model_id = script.get('model_id', DS)
    data = script.get('data', DS_DATA[DS])
    sl = int(script.get('seq_len', 96))

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

    exp = Exp_Anomaly_Detection(args)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    Cls = getattr(DL, LOADER[DS])

    def tiled_scores(flag):
        # step=seq_len：窗口平铺，每点恰好一次（尾部不足一窗舍弃）
        ds = Cls(args, args.root_path, win_size=sl, step=sl, flag=flag)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        criterion = nn.MSELoss(reduce=False)
        scores, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].float().to(exp.device)
                outputs = exp.model(batch_x, None, None, None)
                scores.append(torch.mean(criterion(batch_x, outputs), dim=-1).cpu().numpy())
                if flag == 'test':
                    labels.append(batch[1].numpy())
        sc = np.concatenate(scores, axis=0).reshape(-1)
        if flag == 'test':
            return sc, np.concatenate(labels, axis=0).reshape(-1)
        return sc

    sc, lb = tiled_scores('test')
    tr = tiled_scores('train')

    out_dir = '/data/modeluse/TS/vus_diag/tsl_pointwise'
    os.makedirs(out_dir, exist_ok=True)
    out = f'{out_dir}/{MODEL}_{DS}_s{SEED}.npz'
    np.savez(out, score=sc.astype(np.float64), label=lb.astype(int), train_score=tr.astype(np.float64))

    # 回归：原本就是 step=100=sl 平铺的 run（SMD/SWAT × TimesNet/DLinear），应与旧 npy 一致
    old = f'{TSL}/test_results/{setting}/test_scores.npy'
    if os.path.exists(old) and MODEL != 'KANAD' and sl == 100:
        old_sc = np.load(old).reshape(-1)
        delta = float(np.max(np.abs(old_sc - sc))) if old_sc.shape == sc.shape else float('nan')
        reg = f'REG shape={old_sc.shape} maxdelta={delta:.3e}'
    else:
        reg = f'old_shape={np.load(old).shape if os.path.exists(old) else None} (surface changed)'
    print(f'[pw-dump] {MODEL:9s} {DS:9s} s{SEED:>5} n_test={sc.shape[0]} n_train={tr.shape[0]} '
          f'old_n={np.load(old).shape[0] if os.path.exists(old) else None} {reg} -> {out}', flush=True)


if __name__ == '__main__':
    main()
