#!/usr/bin/env python3
"""混合打分第二遍：min_z / wmean_03 / mean_z 的全协议（C=POT+PA, A=POT raw, D, ROC, AP）。
训练侧分数：gated(tr) 与 revin train score（stride-1 抽稀对齐）按同式组合，z 化用各自 split 统计。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import (adjust_predicts, point2point, pot_threshold,
                          gated_score, LM)

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SC = '/data/modeluse/TS/vus_diag/istad_scores'
SEEDS = ['87', '90', '2021']


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def build(mode, gz, rz):
    if mode == 'min_z':
        return np.minimum(z(gz), z(rz))
    if mode == 'mean_z':
        return (z(gz) + z(rz)) / 2
    if mode == 'wmean_03':
        return z(rz) + 0.3 * z(gz)
    raise ValueError(mode)


def fmt(v):
    return f'{np.mean(v):.4f}±{np.std(v):.4f}'


def main():
    modes = ['min_z', 'mean_z', 'wmean_03']
    tables = {m: {k: [] for k in ['A', 'AP', 'ROC', 'C', 'D']} for m in modes}
    for seed in SEEDS:
        zb = np.load(f'{PF}/PSM_s{seed}.npz')
        zr_npz = np.load(f'{PF}/PSM_s{seed}_revin.npz')
        te, tr = zb['test_err'].astype(float), zb['train_err'].astype(float)
        lb = zb['label'].astype(int)
        rte = zr_npz['test_err'].astype(float).mean(axis=1)
        rtr = np.load(f'{SC}/PSM_s{seed}_revin.npz')['train_score'].astype(float)[::64]
        g_te, _ = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        m = min(len(rtr), len(g_tr))
        rtr, g_tr_a = rtr[:m], g_tr[:m]
        for mode in modes:
            sc_te = build(mode, g_te, rte)
            sc_tr = build(mode, g_tr_a, rtr)
            th = pot_threshold(sc_tr, sc_te, LM['PSM'])
            a = point2point(sc_te > th, lb > 0.1)[0]
            c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
            d = bf_pa(sc_te, lb)['f1']
            ap = average_precision_score(lb, sc_te)
            roc = roc_auc_score(lb, sc_te)
            for key, v in zip(['A', 'AP', 'ROC', 'C', 'D'], [a, ap, roc, c, d]):
                tables[mode][key].append(v)
            print(f's{seed} [{mode}] A={a:.4f} C={c:.4f} D={d:.4f} ROC={roc:.4f} '
                  f'AP={ap:.4f} POT_th={th:.4g}', flush=True)
    for mode in modes:
        print(f'\n=== {mode} PSM seed-mean ===')
        print('  ' + '  '.join(f'{k}={fmt(v)}' for k, v in tables[mode].items()))


if __name__ == '__main__':
    main()
