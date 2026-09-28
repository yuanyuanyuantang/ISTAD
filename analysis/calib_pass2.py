#!/usr/bin/env python3
"""自校准打分第二遍：对第一遍赢家做全协议（C=POT+PA, A=POT raw, D=bf+PA, ROC, AP）。
POT 的 init 用训练集自校准分数（各 split 用自身分位数校准，部署语义=按近期基线校准）。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import (adjust_predicts, point2point, pot_threshold, LM)

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = {'PSM': ['87', '90', '98', '2021'], 'SMD': ['48', '2021', '2022', '2025'],
         'EXA': ['48', '89', '2021'], 'SWAT': ['48', '89', '2021']}


def calib(e, q):
    return np.clip(e - q[None, :], 0.0, None)


def selfcalib(e, q_level=50):
    return calib(e, np.percentile(e, q_level, axis=0))


def topk(e, k):
    return np.sort(e, axis=1)[:, -k:].mean(axis=1)


def trim1(e):
    idx = np.argmax(e, axis=1)
    return (e.sum(axis=1) - e[np.arange(len(e)), idx]) / (e.shape[1] - 1)


def agg(e, mode):
    return {'mean': e.mean(axis=1), 'top2': topk(e, 2), 'top3': topk(e, 3),
            'trim1': trim1(e)}[mode]


def fmt(v):
    return f'{np.mean(v):.4f}±{np.std(v):.4f}'


def main():
    modes = ['mean', 'top2', 'top3', 'trim1']
    tables = {m: {k: {} for k in ['A', 'B_ap', 'B_roc', 'C', 'D']} for m in modes}
    for ds, seeds in SEEDS.items():
        rows = {m: [] for m in modes}
        for seed in seeds:
            z = np.load(f'{PF}/{ds}_s{seed}.npz')
            te, tr = z['test_err'].astype(float), z['train_err'].astype(float)
            lb = z['label'].astype(int)
            ce, ctr = selfcalib(te), selfcalib(tr)
            for m in modes:
                sc_te, sc_tr = agg(ce, m), agg(ctr, m)
                th = pot_threshold(sc_tr, sc_te, LM[ds])
                a = point2point(sc_te > th, lb > 0.1)[0]
                c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
                d = bf_pa(sc_te, lb)['f1']
                ap = average_precision_score(lb, sc_te)
                roc = roc_auc_score(lb, sc_te)
                rows[m].append((a, ap, roc, c, d))
                print(f'{ds} s{seed} [{m}] A={a:.4f} C={c:.4f} D={d:.4f} '
                      f'ROC={roc:.4f} AP={ap:.4f} th={th:.4g}', flush=True)
        for m in modes:
            for i, key in [(0, 'A'), (1, 'B_ap'), (2, 'B_roc'), (3, 'C'), (4, 'D')]:
                tables[m][key][ds] = fmt([r[i] for r in rows[m]])
    for m in modes:
        print(f'\n=== 自校准 c50+{m}（mean±std） ===')
        for key, row in tables[m].items():
            print(f'{key:6s} | ' + ' | '.join(f'{ds} {v}' for ds, v in row.items()))


if __name__ == '__main__':
    main()
