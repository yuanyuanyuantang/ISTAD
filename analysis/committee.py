#!/usr/bin/env python3
"""跨种子分数委员会（零重训）：inference 时 M 个种子模型的 z 分数取均值。
检验能否靠降种子方差闭合 SMD D 缺口（0.8361 → ≥0.8514），
顺带看 PSM min_z 委员会能否再逼近 TimesNet 0.9732。
全协议：A/B/C/D。训练侧同样委员会化供 POT。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SC = '/data/modeluse/TS/vus_diag/istad_scores'
SL = {'PSM': 64, 'SMD': 96}


def smd_committee(seeds):
    tes, trs, lbs = [], [], []
    for s in seeds:
        zb = np.load(f'{PF}/SMD_s{s}.npz')
        g_te, _ = gated_score(zb['test_err'].astype(float), zb['train_err'].astype(float))
        g_tr, _ = gated_score(zb['train_err'].astype(float), zb['train_err'].astype(float))
        tes.append(z(g_te)); trs.append(z(g_tr)); lbs.append(zb['label'].astype(int))
    n = min(len(t) for t in tes)
    sc_te = np.mean([t[:n] for t in tes], axis=0)
    lb = lbs[0][:n]
    assert all((x[:n] == lb).all() for x in lbs)
    # 训练侧：各模型 train 分长度可能不同，取 min 后平均
    m = min(len(t) for t in trs)
    sc_tr = np.mean([t[:m] for t in trs], axis=0)
    return sc_te, sc_tr, lb


def psm_committee(seeds):
    sl = SL['PSM']
    tes, trs, lbs = [], [], []
    for s in seeds:
        zb = np.load(f'{PF}/PSM_s{s}.npz')
        zr = np.load(f'{PF}/PSM_s{s}_revin.npz')
        te, tr = zb['test_err'].astype(float), zb['train_err'].astype(float)
        g_te, _ = gated_score(te, tr); g_tr, _ = gated_score(tr, tr)
        rte = zr['test_err'].astype(float).mean(axis=1)
        rtr = np.load(f'{SC}/PSM_s{s}_revin.npz')['train_score'].astype(float)[::sl]
        mm = min(len(rtr), len(g_tr))
        tes.append(z(np.minimum(z(g_te), z(rte))))
        trs.append(z(np.minimum(z(g_tr[:mm]), z(rtr[:mm]))))
        lbs.append(zb['label'].astype(int))
    n = min(len(t) for t in tes)
    sc_te = np.mean([t[:n] for t in tes], axis=0)
    lb = lbs[0][:n]
    assert all((x[:n] == lb).all() for x in lbs)
    m = min(len(t) for t in trs)
    sc_tr = np.mean([t[:m] for t in trs], axis=0)
    return sc_te, sc_tr, lb


def report(ds, sc_te, sc_tr, lb):
    th = pot_threshold(sc_tr, sc_te, LM[ds])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)
    print(f'{ds} committee[{len_seeds}] A={a:.4f} C={c:.4f} D={d["f1"]:.4f} '
          f'(hit {d["n_hit"]}/{d["n_seg"]} FP={d["fp"]}) '
          f'ROC={roc_auc_score(lb, sc_te):.4f} AP={average_precision_score(lb, sc_te):.4f} '
          f'POT_th={th:.4g}')


len_seeds = '?'
if __name__ == '__main__':
    import sys
    for k in (2, 3, 4):
        seeds = ['48', '2021', '2022', '2025'][:k]
        len_seeds = k
        sc_te, sc_tr, lb = smd_committee(seeds)
        report('SMD', sc_te, sc_tr, lb)
    print('TimesNet SMD D=0.8514 / C=0.7726；TranAD C=0.7825')
    for k in (2, 3, 4):
        seeds = ['87', '90', '98', '2021'][:k]
        len_seeds = k
        sc_te, sc_tr, lb = psm_committee(seeds)
        report('PSM', sc_te, sc_tr, lb)
    print('TimesNet PSM D=0.9732 / C=0.9672')
