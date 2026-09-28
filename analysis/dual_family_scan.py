#!/usr/bin/env python3
"""双头联合模型融合家族 × 4 种子扫描（探索性）：
看是否存在比 min_z 均值更高的融合（D 为主，ROC≥0.70 护栏）。
注：同种子选融合有多重比较成分，结果仅作是否续线的决策参考。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
DS = 'PSM'
SEEDS = ['87', '90', '98', '2021']


def proto(sc_te, sc_tr, lb):
    th = pot_threshold(sc_tr, sc_te, LM[DS])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    return dict(A=a, C=c, D=d, ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te))


def softmin(a, b, t):
    return -t * np.log(np.exp(-a / t) + np.exp(-b / t))


def main():
    # 预载
    views = {}
    for s in SEEDS:
        zd = np.load(f'{PF}/{DS}_s{s}_dual.npz')
        lb = zd['label'].astype(int)
        g_te, _ = gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))
        g_tr, _ = gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))
        m_te = zd['test_err_revin'].astype(float).mean(axis=1)
        m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
        views[s] = (g_te, g_tr, m_te, m_tr, lb)

    fams = {
        'min_z': lambda a, b: np.minimum(a, b),
        'mean_z': lambda a, b: (a + b) / 2,
        'wmean03': lambda a, b: 0.3 * a + 0.7 * b,
        'wmean05': lambda a, b: 0.5 * a + 0.5 * b,
        'wmean07': lambda a, b: 0.7 * a + 0.3 * b,
        'softmin_t05': lambda a, b: softmin(a, b, 0.5),
        'softmin_t1': lambda a, b: softmin(a, b, 1.0),
        'softmin_t2': lambda a, b: softmin(a, b, 2.0),
    }
    print(f'{"fusion":14s} {"A":>12s} {"C":>12s} {"D":>12s} {"ROC":>12s} {"AP":>12s}')
    for name, f in fams.items():
        agg = {k: [] for k in ['A', 'C', 'D', 'ROC', 'AP']}
        for s in SEEDS:
            g_te, g_tr, m_te, m_tr, lb = views[s]
            n = min(len(g_tr), len(m_tr))
            sc_te = f(z(g_te), z(m_te))
            sc_tr = f(z(g_tr[:n]), z(m_tr[:n]))
            r = proto(sc_te, sc_tr, lb)
            for k in agg:
                agg[k].append(r[k])
        print(f'{name:14s} ' + '  '.join(
            f'{np.mean(agg[k]):.4f}±{np.std(agg[k]):.4f}' for k in ['A', 'C', 'D', 'ROC', 'AP']))
    print('\nTimesNet 靶: C=0.9672±0.0008  D=0.9732±0.0008')


if __name__ == '__main__':
    main()
