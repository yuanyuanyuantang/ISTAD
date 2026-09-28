#!/usr/bin/env python3
"""ISTAD 全长测试分数导出（绕过 run.py，直接构建模型 + forward）。

背景：test_results/*/feature_localization_timeseries.csv 里的 anomaly_score/label_true
是 _to_window_last_step_series 按 seq_len 抽稀的可视化序列，不能用于逐点指标。
本脚本按与 exp.test() 完全相同的公式（mean over vars 的重建 MSE）导出全长
score/label 到 /data/modeluse/TS/vus_diag/istad_scores/{ds}_s{seed}.npz。

用法: python dump_istad_full.py <DS> <SEED> <GPU>
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

DS, SEED, GPU = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else ''  # 'revin' = RevIN 重训变体
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU)

sys.path.insert(0, '/data/modeluse/TS/ISTAD')
os.chdir('/data/modeluse/TS/ISTAD')

from exp.exp_anomaly_detection import Exp_Anomaly_Detection  # noqa: E402
from utils.tools import dotdict  # noqa: E402

CKPT = {
    ('PSM', '87', 'revin'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_revin_s87_0',
    ('PSM', '90', 'revin'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_revin_s90_0',
    ('PSM', '2021', 'revin'): 'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_revin_s2021_0',
    ('PSM', '98', 'revin'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_revin_s98_0',
    ('SMD', '48', 'revin'):    'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_revin_s48_0',
    ('SMD', '2021', 'revin'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_revin_s2021_0',
    ('SMD', '2022', 'revin'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_revin_s2022_0',
    ('SMD', '2025', 'revin'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_revin_s2025_0',
    ('SWAT', '48', 'revin'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_revin_s48_0',
    ('SWAT', '89', 'revin'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_revin_s89_0',
    ('SWAT', '2021', 'revin'): 'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_revin_s2021_0',
    ('SMD', '48'):    'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s48_0',
    ('SMD', '2021'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s2021_0',
    ('SMD', '2022'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s2022_0',
    ('SMD', '2025'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s2025_0',
    ('PSM', '87'):    'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s87_0',
    ('PSM', '90'):    'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s90_0',
    ('PSM', '98'):    'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s98_0',
    ('PSM', '2021'):  'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s2021_0',
    ('EXA', '48'):    'anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_g10_s48_0',
    ('EXA', '89'):    'anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_g10_s89_0',
    ('EXA', '2021'):  'anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_g10_s2021_0',
    ('SWAT', '89'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_test_0',
    ('SWAT', '48'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_swats48_0',
    ('SWAT', '2021'): 'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_swats2021_0',
}

# 每数据集训练时覆盖 run.py 默认值的参数（其余全部用 run.py 默认，与原始 run 一致）
DS_ARGS = {
    'SMD':  dict(data='SMD', root_path='./dataset/SMD', model_id='SMD',
                 seq_len=96, enc_in=38, c_out=38, batch_size=64,
                 istad_kanad_order=4, istad_dropout=0.2),
    'PSM':  dict(data='PSM', root_path='./dataset/PSM', model_id='PSM',
                 seq_len=64, enc_in=25, c_out=25, batch_size=64,
                 istad_kanad_order=6, istad_dropout=0.3),
    'EXA':  dict(data='EXATHLON', root_path='./dataset/EXATHLON', model_id='EXATHLON',
                 seq_len=100, enc_in=19, c_out=19, batch_size=64,
                 istad_kanad_order=4, istad_dropout=0.1),
    'SWAT': dict(data='SWAT', root_path='./dataset/SWAT', model_id='SWAT_BestF1_Adaptive_2GPU',
                 seq_len=96, enc_in=51, c_out=51, batch_size=32,
                 istad_kanad_order=6, istad_dropout=0.1,
                 istad_feat_gat_embed_dim=256, istad_gru_hid_dim=256,
                 istad_recon_n_layers=2, istad_recon_hid_dim=256,
                 istad_n_hyperedges=20, istad_k_top=10, istad_kernel_size=15),
}


def build_args():
    base = dotdict(
        task_name='anomaly_detection', is_training=0, model='ISTAD',
        features='M', checkpoint_path='',
        # run.py 默认值（与训练一致的部分）
        istad_kernel_size=7, istad_feat_gat_embed_dim=-1,
        istad_gru_n_layers=1, istad_gru_hid_dim=150,
        istad_recon_n_layers=1, istad_recon_hid_dim=150,
        istad_recon_type='kanad', istad_kanad_order=4,
        istad_dropout=0.2, istad_alpha=0.2,
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
    )
    base.update(DS_ARGS[DS])
    if MODE == 'revin':
        base.istad_revin = 1
    base.seed = int(SEED)
    key = (DS, SEED, MODE) if (DS, SEED, MODE) in CKPT else (DS, SEED)
    base.setting = CKPT[key]
    base.checkpoint_path = os.path.join('./checkpoints', base.setting, 'checkpoint.pth')
    return base


def main():
    args = build_args()
    assert os.path.exists(args.checkpoint_path), f'checkpoint 不存在: {args.checkpoint_path}'
    exp = Exp_Anomaly_Detection(args)
    exp._load_checkpoint(args.checkpoint_path)
    exp.model.eval()

    _, test_loader = exp._get_data(flag='TEST')
    _, train_loader = exp._get_data(flag='train')
    criterion = nn.MSELoss(reduce=False)

    def forward_scores(loader, with_label):
        scores, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].float().to(exp.device)
                outputs = exp.model(batch_x, None, None, None)
                score = torch.mean(criterion(batch_x, outputs), dim=-1)
                scores.append(score.detach().cpu().numpy())
                if with_label:
                    labels.append(batch[1])
        sc = np.concatenate(scores, axis=0).reshape(-1)
        if with_label:
            lb = np.concatenate(labels, axis=0).reshape(-1)
            return sc, lb
        return sc

    sc, lb = forward_scores(test_loader, True)
    assert sc.shape == lb.shape, f'{sc.shape} vs {lb.shape}'
    tr = forward_scores(train_loader, False)

    from sklearn.metrics import roc_auc_score
    roc = roc_auc_score(lb, sc) if 0 < lb.sum() < len(lb) else float('nan')
    out_dir = '/data/modeluse/TS/vus_diag/istad_scores'
    os.makedirs(out_dir, exist_ok=True)
    suffix = f'_{MODE}' if MODE else ''
    out = os.path.join(out_dir, f'{DS}_s{SEED}{suffix}.npz')
    np.savez(out, score=sc, label=lb, train_score=tr)
    print(f'[dump] {DS} s{SEED}  n={sc.shape[0]}  anom_frac={lb.mean():.4f}  '
          f'ROC={roc:.4f}  -> {out}', flush=True)


if __name__ == '__main__':
    main()
