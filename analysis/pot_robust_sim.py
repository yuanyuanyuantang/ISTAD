#!/usr/bin/env python3
"""POT 稳健化离线仿真（零训练）：统一更换阈值标定协议，11 模型同口径重算 C(POT+PA)。
动机：v2.1 下 C 表唯一"健康参照真缺口"是 EXA（ISTAD 0.9434 vs DLinear 0.9556 /
KANAD 0.9577）；SWAT C 的 TimesNet 0.8215 与 D 0.9064 同为极性反转展品。
变体（对全部模型一视同仁，禁逐模型挑选）：
  V0 lm_table   现行协议（LM0 表，TranAD 用 LM1）——回归对照
  V1 mult1      保留逐数据集 SPOT level，去掉手工乘子（lm[1]=1.0）
  V2 flat98     完全统一 level=0.98 / mult=1.0（无逐数据集手调）
  V3 flat99     level=0.99 / mult=1.0
  V4 pctl_rate  抛弃 EVT：threshold = percentile(train+test, 100-100*rate)，
                rate = 数据集异常率（数据集级、模型无关，TSLib anomaly_ratio 同款）
判定纪律：任何变体须全模型统一重算整张 C 表；只有 ISTAD 相对位置改善（EXA 超过
DLinear/KANAD 且其余数据集不丢第一/打平带）才谈采纳，否则原样收回。

用法: python pot_robust_sim.py EXA [PSM SMD SWAT]   （先打靶数据集，再全网格）
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np

VUS = '/data/modeluse/TS/vus_diag'
TSL = '/data/modeluse/TS/Time-Series-Library'
PF = f'{VUS}/istad_perfeat'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fm = _load(f'{VUS}/final_matrix.py', 'fm')
adjust_predicts, point2point, pot_threshold = fm.adjust_predicts, fm.point2point, fm.pot_threshold
LM0 = {'SMD': (0.988, 1.00), 'SWAT': (0.9999, 1.2), 'PSM': (0.98, 0.9), 'EXA': (0.99, 1.0)}
LM1 = {'SMD': (0.997, 1.06), 'SWAT': (0.9999, 1.28), 'PSM': (0.98, 0.9), 'EXA': (0.99, 1.0)}
DS_CLASSIC = {'SMD': 'SMD', 'PSM': 'PSM', 'SWAT': 'SWaT', 'EXA': 'Exathlon'}
DS_DATA = {'EXA': 'Exathlon', 'PSM': 'PSM', 'SMD': 'SMD', 'SWAT': 'SWAT'}
TSL_CFG = {  # (ds_data, seq_len, model_id) —— 抄 assemble_unified.py
    ('EXA', 'TimesNet'): ('Exathlon', 96, 'EXATHLON'), ('EXA', 'DLinear'): ('Exathlon', 96, 'EXATHLON'),
    ('EXA', 'KANAD'): ('Exathlon', 96, 'EXATHLON'),
    ('PSM', 'TimesNet'): ('PSM', 100, 'PSM'), ('PSM', 'DLinear'): ('PSM', 100, 'PSM'),
    ('PSM', 'KANAD'): ('PSM', 64, 'PSM'),
    ('SMD', 'TimesNet'): ('SMD', 100, 'SMD'), ('SMD', 'DLinear'): ('SMD', 100, 'SMD'),
    ('SMD', 'KANAD'): ('SMD', 96, 'SMD'),
    ('SWAT', 'TimesNet'): ('SWAT', 100, 'SWAT'), ('SWAT', 'DLinear'): ('SWAT', 100, 'SWAT'),
    ('SWAT', 'KANAD'): ('SWAT', 80, 'SWAT'),
}
TSL_SEEDS = [48, 2021, 2022]
IST_SEEDS = {'SMD': ['48', '2021', '2022', '2025'], 'PSM': ['87', '90', '98', '2021'],
             'EXA': ['48', '89', '2021'], 'SWAT': ['48', '89', '2021']}
CLASSIC = ['DAGMM', 'GDN', 'LSTM_AD', 'MAD_GAN', 'MTAD_GAT', 'OmniAnomaly', 'TranAD']
ORDER = ['ISTAD', 'TimesNet', 'DLinear', 'KANAD'] + CLASSIC
VARIANTS = ['V0_lm_table', 'V1_mult1', 'V2_flat98', 'V3_flat99', 'V4_pctl_rate']


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def load_istad(ds, seed):
    """v2.1 最终分数：EXA/SWAT=gated；PSM/SMD=dual min_z（与 dual_tables* 逐位一致）"""
    if ds in ('EXA', 'SWAT'):
        zd = np.load(f'{PF}/{ds}_s{seed}.npz')
        te, tr = zd['test_err'].astype(float), zd['train_err'].astype(float)
        g_te, _ = fm.gated_score(te, tr)
        g_tr, _ = fm.gated_score(tr, tr)
        lb = zd['label'].astype(int)
        return g_te, g_tr, lb
    zd = np.load(f'{PF}/{ds}_s{seed}_dual.npz')
    te, tr = zd['test_err'].astype(float), zd['train_err'].astype(float)
    g_te, _ = fm.gated_score(te, tr)
    g_tr, _ = fm.gated_score(tr, tr)
    m_te = zd['test_err_revin'].astype(float).mean(axis=1)
    m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
    n = min(len(g_tr), len(m_tr))
    sc = np.minimum(z(g_te), z(m_te))
    sc_tr = np.minimum(z(g_tr[:n]), z(m_tr[:n]))
    lb = zd['label'].astype(int)[:len(sc)]
    return sc, sc_tr, lb


def load_tslib(ds, model, seed):
    ds_data, sl, mid = TSL_CFG[(ds, model)]
    pats = [s for s in os.listdir(f'{TSL}/test_results')
            if re.fullmatch(rf'anomaly_detection_{ds_data}_{model}_{mid}_ftM_sl{sl}_.*_ms{seed}_0', s)]
    assert len(pats) == 1, (ds, model, seed, pats)
    d = f'{TSL}/test_results/{pats[0]}'
    sc = np.load(f'{d}/test_scores.npy').astype(float).reshape(-1)
    tr = np.load(f'{d}/train_scores.npy').astype(float).reshape(-1)
    lb = np.load(f'{d}/test_labels.npy').astype(int).reshape(-1)
    n = min(len(sc), len(lb))
    return sc[:n], tr, lb[:n]


def load_classic(ds, model):
    zd = np.load(f'{VUS}/classic_scores/{model}_{DS_CLASSIC[ds]}.npz')
    return (zd['score'].astype(float), zd['train_score'].astype(float),
            zd['label'].astype(int))


def threshold_of(variant, test_sc, train_sc, lb, ds, model):
    lm_base = LM1[ds] if model == 'TranAD' else LM0[ds]
    if variant == 'V0_lm_table':
        return pot_threshold(train_sc, test_sc, lm_base)
    if variant == 'V1_mult1':
        return pot_threshold(train_sc, test_sc, (lm_base[0], 1.0))
    if variant == 'V2_flat98':
        return pot_threshold(train_sc, test_sc, (0.98, 1.0))
    if variant == 'V3_flat99':
        return pot_threshold(train_sc, test_sc, (0.99, 1.0))
    if variant == 'V4_pctl_rate':
        rate = float(lb.mean())
        return float(np.percentile(np.concatenate([train_sc, test_sc]), 100 - 100 * rate))
    raise ValueError(variant)


def run_ds(ds):
    print(f'\n########## {ds} ##########', flush=True)
    jobs = [('ISTAD', s, 'istad') for s in IST_SEEDS[ds]]
    jobs += [(m, s, 'tslib') for m in ['TimesNet', 'DLinear', 'KANAD'] for s in TSL_SEEDS]
    jobs += [(m, None, 'classic') for m in CLASSIC]

    acc = {v: {m: [] for m in ORDER} for v in VARIANTS}
    accA = {v: {m: [] for m in ORDER} for v in VARIANTS}
    for model, seed, kind in jobs:
        if kind == 'istad':
            sc, tr, lb = load_istad(ds, seed)
        elif kind == 'tslib':
            sc, tr, lb = load_tslib(ds, model, seed)
        else:
            sc, tr, lb = load_classic(ds, model)
        lb01 = lb > 0.1
        line = [f'{model:11s} {str(seed or "s"):>5s}']
        for v in VARIANTS:
            th = threshold_of(v, sc, tr, lb, ds, model)
            pred = (sc > th).astype(int)
            c = point2point(adjust_predicts(sc, lb01, th), lb01)[0]
            a = point2point(pred, lb01)[0]
            acc[v][model].append(c)
            accA[v][model].append(a)
            line.append(f'{v.split("_")[0]}={c:.4f}/A={a:.4f}')
        print('  '.join(line), flush=True)

    print(f'\n===== {ds} C（POT+PA）按变体 mean±std =====', flush=True)
    for v in VARIANTS:
        cells = []
        for m in ORDER:
            vals = acc[v][m]
            cells.append(f'{m}={np.mean(vals):.4f}' + (f'±{np.std(vals):.4f}' if len(vals) > 1 else ''))
        print(f'[{v}] ' + ' | '.join(cells), flush=True)
    best = {v: max(ORDER, key=lambda m: np.mean(acc[v][m])) for v in VARIANTS}
    ist = {v: sorted([np.mean(acc[v][m]) for m in ORDER], reverse=True)
           for v in VARIANTS}
    for v in VARIANTS:
        rank = 1 + sum(1 for m in ORDER if np.mean(acc[v][m]) > np.mean(acc[v]['ISTAD']) + 1e-12)
        print(f'[{v}] ISTAD 排位 #{rank} / best={best[v]}', flush=True)
    return {v: {m: acc[v][m] for m in ORDER} for v in VARIANTS}


def main():
    dss = sys.argv[1:] or ['EXA']
    out = {}
    for ds in dss:
        out[ds] = run_ds(ds)
    with open(f'{VUS}/pot_robust_sim.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('\n[done] -> pot_robust_sim.json', flush=True)


if __name__ == '__main__':
    main()
