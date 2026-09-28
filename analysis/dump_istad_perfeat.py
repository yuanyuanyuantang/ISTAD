#!/usr/bin/env python3
"""ISTAD 逐特征重建误差导出（诊断用）：存 (n, C) 的每点每特征 MSE，
用于定位"正常重尾"集中在哪些特征、检验鲁棒聚合/逐特征标准化的效果。
测试集用 TEST 平铺（逐点），训练集也用 step=win_size 平铺（统计量来源，避免 stride-1 冗余）。

用法: python dump_istad_perfeat.py <DS> <SEED> <GPU>
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

DS, SEED, GPU = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else ''  # 'revin' = RevIN 重训变体
os.environ['CUDA_VISIBLE_DEVICES'] = GPU  # 强制：setdefault 曾被外部环境污染致落错卡

sys.path.insert(0, '/data/modeluse/TS/ISTAD')
os.chdir('/data/modeluse/TS/ISTAD')

from exp.exp_anomaly_detection import Exp_Anomaly_Detection  # noqa: E402
from utils.tools import dotdict  # noqa: E402
from data_provider.data_loader import (  # noqa: E402
    PSMSegLoader, SMDSegLoader, SWATSegLoader, EXATHLONSegLoader,
)

CKPT = {
    ('SWAT', '48', 'dual'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_dual_s48_0',
    ('SWAT', '89', 'dual'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_dual_s89_0',
    ('SWAT', '2021', 'dual'): 'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_dual_s2021_0',
    ('PSM', '87', 'dual'):    'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_s87_0',
    ('PSM', '90', 'dual'):    'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_s90_0',
    ('PSM', '98', 'dual'):    'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_s98_0',
    ('PSM', '2021', 'dual'):  'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_s2021_0',
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
    ('SMD', '48', 'dual'):    'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_dual_s48_0',
    ('SMD', '2021', 'dual'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_dual_s2021_0',
    ('SMD', '2022', 'dual'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_dual_s2022_0',
    ('SMD', '2025', 'dual'):  'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_dual_s2025_0',
    ('EXA', '48', 'dual'):    'anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_dual_s48_0',
    ('EXA', '89', 'dual'):    'anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_dual_s89_0',
    ('EXA', '2021', 'dual'):  'anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_dual_s2021_0',
    ('SWAT', '89'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_test_0',
    ('SWAT', '48'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_swats48_0',
    ('SWAT', '2021'): 'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_swats2021_0',
    ('SWAT', '48', 'k25'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_k25_s48_0',
    ('SWAT', '89', 'k25'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_k25_s89_0',
    ('SWAT', '2021', 'k25'): 'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_SWAT_k25_s2021_0',
    # seq_len 试点（prereg_sl100.md）：sl100 训练的新 checkpoint
    ('PSM', '87', 'dual_sl100'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl100_bmhgat_kan_tcn_bs64_PSM_g10_dual_sl100_s87_0',
    ('PSM', '90', 'dual_sl100'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl100_bmhgat_kan_tcn_bs64_PSM_g10_dual_sl100_s90_0',
    ('PSM', '98', 'dual_sl100'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl100_bmhgat_kan_tcn_bs64_PSM_g10_dual_sl100_s98_0',
    ('PSM', '2021', 'dual_sl100'): 'anomaly_detection_PSM_ISTAD_PSM_ftM_sl100_bmhgat_kan_tcn_bs64_PSM_g10_dual_sl100_s2021_0',
    ('SWAT', '48', 'sl100'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl100_bmhgat_kan_tcn_bs128_SWAT_sl100_s48_0',
    ('SWAT', '89', 'sl100'):   'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl100_bmhgat_kan_tcn_bs128_SWAT_sl100_s89_0',
    ('SWAT', '2021', 'sl100'): 'anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl100_bmhgat_kan_tcn_bs128_SWAT_sl100_s2021_0',
    # λ 扫描（prereg_psm_lambda.md）：dual 联合训练 λ∈{0.5,2.0}
    ('PSM', '87', 'dual_l05'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l05_s87_0',
    ('PSM', '90', 'dual_l05'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l05_s90_0',
    ('PSM', '98', 'dual_l05'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l05_s98_0',
    ('PSM', '2021', 'dual_l05'): 'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l05_s2021_0',
    ('PSM', '87', 'dual_l20'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l20_s87_0',
    ('PSM', '90', 'dual_l20'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l20_s90_0',
    ('PSM', '98', 'dual_l20'):   'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l20_s98_0',
    ('PSM', '2021', 'dual_l20'): 'anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_dual_l20_s2021_0',
    # SMD sl100（prereg_smd_sl100.md）：dual λ=1.0，仅 seq_len 96→100
    ('SMD', '48', 'dual_sl100'):   'anomaly_detection_SMD_ISTAD_SMD_ftM_sl100_bmhgat_kan_tcn_bs64_SMD_dual_sl100_s48_0',
    ('SMD', '2021', 'dual_sl100'): 'anomaly_detection_SMD_ISTAD_SMD_ftM_sl100_bmhgat_kan_tcn_bs64_SMD_dual_sl100_s2021_0',
    ('SMD', '2022', 'dual_sl100'): 'anomaly_detection_SMD_ISTAD_SMD_ftM_sl100_bmhgat_kan_tcn_bs64_SMD_dual_sl100_s2022_0',
    ('SMD', '2025', 'dual_sl100'): 'anomaly_detection_SMD_ISTAD_SMD_ftM_sl100_bmhgat_kan_tcn_bs64_SMD_dual_sl100_s2025_0',
}

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
LOADER = {'PSM': PSMSegLoader, 'SMD': SMDSegLoader,
          'SWAT': SWATSegLoader, 'EXA': EXATHLONSegLoader}
# dump 用单卡前向（无 DataParallel 分摊），dual 双路激活更大；
# SWAT hgat(embed256, 20 超边) 在训练侧 bs128 的测试阶段已实测 OOM → 降批
DUMP_BS = {'SWAT': 32}


def build_args():
    base = dotdict(
        task_name='anomaly_detection', is_training=0, model='ISTAD',
        features='M', checkpoint_path='',
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
    if DS in DUMP_BS:
        base.batch_size = DUMP_BS[DS]
    if MODE == 'revin':
        base.istad_revin = 1
        base.istad_revin_eps = 1e-5
    if MODE.startswith('dual'):
        base.istad_dual = 1
        base.istad_dual_lambda = 1.0  # λ 仅训练期生效，推理只要求 dual 前向输出 2C
    if MODE == 'k25':
        # SWAT 宽度试点（prereg_swat_width.md）：base 前向，仅覆写卷积核宽度
        base.istad_kernel_size = 25
    if MODE in ('dual_sl100', 'sl100'):
        # seq_len 试点（prereg_sl100.md）：sl100 训练，前向窗口须与训练一致
        base.seq_len = 100
    base.seed = int(SEED)
    if MODE:
        # 变体模型严禁静默回退到 base checkpoint（曾险酿错模型 dump）
        assert (DS, SEED, MODE) in CKPT, f'MISSING CKPT entry for {(DS, SEED, MODE)}'
        key = (DS, SEED, MODE)
    else:
        key = (DS, SEED)
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
    # 训练集平铺加载器（统计量来源；避免 stride-1 的 8.4M 冗余）
    train_ds = LOADER[DS](args, args.root_path, win_size=args.seq_len,
                          step=args.seq_len, flag='train')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=2)
    criterion = nn.MSELoss(reduce=False)

    def forward_errors(loader):
        errs, errs_rev = [], []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].float().to(exp.device)
                outputs = exp.model(batch_x, None, None, None)
                if MODE.startswith('dual'):
                    C = batch_x.shape[-1]
                    errs.append(criterion(batch_x, outputs[..., :C]).detach().cpu().numpy())
                    errs_rev.append(criterion(batch_x, outputs[..., C:2 * C]).detach().cpu().numpy())
                else:
                    errs.append(criterion(batch_x, outputs).detach().cpu().numpy())
        if MODE.startswith('dual'):
            return np.concatenate(errs, axis=0), np.concatenate(errs_rev, axis=0)
        return np.concatenate(errs, axis=0)  # (n_win, W, C)

    if MODE.startswith('dual'):
        te, te_rev = forward_errors(test_loader)
        n_win, W, C = te.shape
        # sl64 PSM 历史硬编码 87808；sl100 试点按实际平铺长度（label 按 base 文件截断）
        L = n_win * W if MODE == 'dual_sl100' else (87808 if DS == 'PSM' else n_win * W)
        test_err = te.reshape(n_win * W, C)[:L]
        test_err_revin = te_rev.reshape(n_win * W, C)[:L]
        print(f'test errors: {test_err.shape} (base + revin)', flush=True)
        tr, tr_rev = forward_errors(train_loader)
        train_err = tr.reshape(-1, C)
        train_err_revin = tr_rev.reshape(-1, C)
        print(f'train errors: {train_err.shape} (base + revin)', flush=True)
    else:
        te = forward_errors(test_loader)
        n_win, W, C = te.shape
        test_err = te.reshape(n_win * W, C)[:n_win * W if MODE == 'sl100'
                                            else (87808 if DS == 'PSM' else n_win * W)]
        print(f'test errors: {test_err.shape}', flush=True)
        tr = forward_errors(train_loader)
        train_err = tr.reshape(-1, C)
        print(f'train errors: {train_err.shape}', flush=True)

    # label 恒取自 base 分数文件（同一测试集）；回归校验对同 MODE 分数
    suffix = f'_{MODE}' if MODE else ''
    base_path = f'/data/modeluse/TS/vus_diag/istad_scores/{DS}_s{SEED}.npz'
    label_out = np.load(base_path)['label']
    if len(label_out) < len(test_err):
        if DS == 'SMD':
            # SMD sl100（prereg_smd_sl100.md）：TEST 平铺为 stride=W 非重叠、前缀逐点
            # 对齐 → 截断到现行评估面（=base label 长 708384），与 v2.1 完全同面
            n = len(label_out)
            test_err = test_err[:n]
            if 'test_err_revin' in locals():
                test_err_revin = test_err_revin[:n]
            print(f'test errors 截断到现行评估面: {n}', flush=True)
        else:
            # sl100 平铺长度可超过 base（sl96/sl64）平铺（尾部对齐差异）→ 从原始测试 csv 补足
            import pandas as pd
            raw_path = {'SWAT': './dataset/SWAT/swat2.csv',
                        'PSM': './dataset/PSM/test_label.csv'}[DS]
            df = pd.read_csv(raw_path)
            raw = df.values[:, -1] if DS == 'SWAT' else df.values[:, 1]
            assert np.array_equal(label_out.astype(int), raw[:len(label_out)].astype(int)), \
                'base label 与原始 csv 前缀不一致，拒绝补足'
            label_out = raw[:len(test_err)]
            print(f'label 从原始 csv 补足: {len(label_out)}', flush=True)
    else:
        label_out = label_out[:len(test_err)]

    ref_path = f'/data/modeluse/TS/vus_diag/istad_scores/{DS}_s{SEED}{suffix}.npz'
    if MODE.startswith('dual'):
        print('[dual] 联合训练新模型，无同权重参考，跳过回归校验', flush=True)
    elif os.path.exists(ref_path):
        agg = test_err.mean(axis=1)
        ref = np.load(ref_path)['score'].astype(float)[:len(agg)]
        md = np.max(np.abs(agg - ref))
        print(f'regression maxdelta vs istad_scores: {md:.6f}', flush=True)
    else:
        print(f'[warn] {ref_path} 不存在，跳过回归校验', flush=True)

    out = f'/data/modeluse/TS/vus_diag/istad_perfeat/{DS}_s{SEED}{suffix}.npz'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if MODE.startswith('dual'):
        np.savez(out, test_err=test_err.astype(np.float32),
                 train_err=train_err.astype(np.float32),
                 test_err_revin=test_err_revin.astype(np.float32),
                 train_err_revin=train_err_revin.astype(np.float32),
                 label=label_out)
    else:
        np.savez(out, test_err=test_err.astype(np.float32),
                 train_err=train_err.astype(np.float32),
                 label=label_out)
    print(f'[dump] -> {out}', flush=True)


if __name__ == '__main__':
    main()
