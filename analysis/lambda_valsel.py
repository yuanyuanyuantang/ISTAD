#!/usr/bin/env python3
"""PSM dual λ 扫描选择规则（prereg_psm_lambda.md 修订版）：base 路验证重建 MSE。

背景：Wave 2 原声明"验证集 bf-F1"不可行——PSM vali split（train 后 20%）无标签；
且 dual 训练 vali_loss = base + λ·MSE(revin) 联合损失跨臂不可比。
本脚本：8 个 λ 臂 checkpoint + 4 个 λ=1.0 现行 checkpoint（参照），各在 vali split
（stride-1 窗口，与训练 early-stopping 同一数据面）前向，base 半区对输入的 MSE。
零测试泄漏（只用 vali split，无标签需求）。

判定：两臂 vali base-MSE 较小者入选；相对差 <1% → KEEP λ=1.0（不采纳任何臂）。
测试集只评入选臂一次（lambda_tables.py 出表，但采纳权在 valsel）。

用法: conda run -n tslib python lambda_valsel.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # GPU0/GPU1 被训练占用

sys.path.insert(0, '/data/modeluse/TS/ISTAD')
os.chdir('/data/modeluse/TS/ISTAD')

from exp.exp_anomaly_detection import Exp_Anomaly_Detection  # noqa: E402
from utils.tools import dotdict  # noqa: E402

BASE = '/data/modeluse/TS/ISTAD/checkpoints'
SETTINGS = {}
for s in ('87', '90', '98', '2021'):
    SETTINGS[f'l05_s{s}'] = (f'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_'
                             f'PSM_g10_dual_l05_s{s}_0', '0.5')
    SETTINGS[f'l10_s{s}'] = (f'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_'
                             f'PSM_g10_dual_s{s}_0', '1.0')
    SETTINGS[f'l20_s{s}'] = (f'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_'
                             f'PSM_g10_dual_l20_s{s}_0', '2.0')


def build_args(setting):
    base = dotdict(
        task_name='anomaly_detection', is_training=0, model='ISTAD',
        features='M', checkpoint_path='',
        data='PSM', root_path='./dataset/PSM', model_id='PSM',
        seq_len=64, enc_in=25, c_out=25, batch_size=64,
        istad_kanad_order=6, istad_dropout=0.3,
        istad_kernel_size=7, istad_feat_gat_embed_dim=-1,
        istad_gru_n_layers=1, istad_gru_hid_dim=150,
        istad_recon_n_layers=1, istad_recon_hid_dim=150,
        istad_recon_type='kanad', istad_alpha=0.2,
        istad_n_hyperedges=-1, istad_k_top=-1,
        istad_h_param_init='normal',
        istad_branch_mode='hgat_kan_tcn', istad_tcn_type='kan',
        istad_kan_grid_size=10, istad_kan_spline_order=3,
        istad_arch='legacy',
        hgst_mod_scale=0.05, hgst_gate=1, hgst_use_relation=1,
        hgst_act='spline', hgst_rank=2, hgst_incidence='kan',
        hgst_s_grid_size=5, hgst_s_spline_order=3,
        hgst_fusion_norm='ln', hgst_gate_bias=0.0, hgst_decoder_input='fused',
        hgst_residual='linear', hgst_input_conv=0,
        use_gpu=1, gpu=0, gpu_type='cuda',
        use_multi_gpu=False, devices='0', device_ids=[0],
        num_workers=2, itr=1, anomaly_ratio=1.0,
        istad_enable_explain=0, use_bestf1_threshold=0,
        istad_dual=1, istad_dual_lambda=1.0,
        seed=42,
    )
    base.setting = setting
    base.checkpoint_path = os.path.join(BASE, setting, 'checkpoint.pth')
    return base


def vali_base_mse(setting):
    args = build_args(setting)
    assert os.path.exists(args.checkpoint_path), f'checkpoint 不存在: {args.checkpoint_path}'
    exp = Exp_Anomaly_Detection(args)
    exp._load_checkpoint(args.checkpoint_path)
    exp.model.eval()
    _, vali_loader = exp._get_data(flag='val')
    criterion = nn.MSELoss(reduce=False)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for batch_x, _ in vali_loader:
            batch_x = batch_x.float().to(exp.device)
            outputs = exp.model(batch_x, None, None, None)
            C = batch_x.shape[-1]
            err = criterion(batch_x, outputs[..., :C])  # base 半区
            tot += err.sum().item()
            cnt += err.numel()
    del exp
    torch.cuda.empty_cache()
    return tot / cnt


def main():
    results = {}
    for name, (setting, lam) in SETTINGS.items():
        mse = vali_base_mse(setting)
        results[name] = mse
        print(f'{name:10s} λ={lam}  base-vali MSE = {mse:.6f}', flush=True)

    print('\n===== 臂均值（4 种子）=====')
    arm_mean = {}
    for lam, tag in (('0.5', 'l05'), ('1.0', 'l10'), ('2.0', 'l20')):
        vals = [v for k, v in results.items() if k.startswith(f'{tag}_')]
        arm_mean[lam] = float(np.mean(vals))
        print(f'λ={lam}: {arm_mean[lam]:.6f}')

    m05, m20 = arm_mean['0.5'], arm_mean['2.0']
    rel = abs(m05 - m20) / min(m05, m20)
    print(f'\n相对差 |Δ|/min = {rel:.4%}（门槛 1%）')
    if rel < 0.01:
        pick = 'KEEP λ=1.0（平手）'
    else:
        pick = 'λ=0.5' if m05 < m20 else 'λ=2.0'
    print(f'>>> 选择规则判定：{pick}（测试集只评入选臂，lambda_tables.py 出表）')


if __name__ == '__main__':
    main()
