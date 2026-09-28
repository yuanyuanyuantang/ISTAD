#!/usr/bin/env python3
"""gated 打分刷新：按统一表种子口径（含 PSM s98）重算 ISTAD 四表行。
与 final_matrix.py 同一套函数；仅输出 mean±std 供论文素材更新。"""
import numpy as np

from final_matrix import (adjust_predicts, point2point, pot_threshold,
                          gated_score, LM)
from agg_simulate import bf_pa
from sklearn.metrics import roc_auc_score, average_precision_score

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = {'PSM': ['87', '90', '98', '2021'], 'SMD': ['48', '2021', '2022', '2025'],
         'EXA': ['48', '89', '2021'], 'SWAT': ['48', '89', '2021']}


def fmt(v):
    return f'{np.mean(v):.4f}±{np.std(v):.4f}'


def main():
    tables = {k: {} for k in ['A', 'B_ap', 'B_roc', 'C', 'D']}
    for ds, seeds in SEEDS.items():
        rows = []
        for seed in seeds:
            z = np.load(f'{PF}/{ds}_s{seed}.npz')
            te, tr = z['test_err'].astype(float), z['train_err'].astype(float)
            lb = z['label'].astype(int)
            g_te, gr = gated_score(te, tr)
            g_tr, _ = gated_score(tr, tr)
            th = pot_threshold(g_tr, g_te, LM[ds])
            a = point2point(g_te > th, lb > 0.1)[0]
            c = point2point(adjust_predicts(g_te, lb, th), lb > 0.1)[0]
            d = bf_pa(g_te, lb)['f1']
            ap = average_precision_score(lb, g_te)
            roc = roc_auc_score(lb, g_te)
            rows.append((seed, a, ap, roc, c, d, gr, th))
            print(f'{ds} s{seed}: A={a:.4f} AP={ap:.4f} ROC={roc:.4f} '
                  f'C={c:.4f} D={d:.4f} gate={gr:.3f} th={th:.4g}', flush=True)
        for i, key in [(1, 'A'), (2, 'B_ap'), (3, 'B_roc'), (4, 'C'), (5, 'D')]:
            tables[key][ds] = fmt([r[i] for r in rows])
    print('\n=== ISTAD gated 行（mean±std） ===')
    for key, row in tables.items():
        print(f'{key:6s} | ' + ' | '.join(f'{ds} {v}' for ds, v in row.items()))


if __name__ == '__main__':
    main()
