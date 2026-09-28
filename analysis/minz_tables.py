#!/usr/bin/env python3
"""min_z 双视角打分终表（PSM / SMD / SWAT）：score = min(z(gated_base), z(revin_mean))，
z 化按各自 split。训练侧同式组合供 POT init（revin train 按 seq_len 抽稀对齐
perfeat 的 step=seq_len 平铺）。EXA 无 RevIN 重训，维持 gated（见 gated_refresh）。
全协议：A = POT raw F1；C = POT+PA；D = bf+PA；ROC / AP。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import (adjust_predicts, point2point, pot_threshold,
                          gated_score, LM)

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SC = '/data/modeluse/TS/vus_diag/istad_scores'
SEEDS = {'PSM': ['87', '90', '98', '2021'],
         'SMD': ['48', '2021', '2022', '2025'],
         'SWAT': ['48', '89', '2021']}
SEQLEN = {'PSM': 64, 'SMD': 96, 'SWAT': 96}


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def min_z(g, r):
    return np.minimum(z(g), z(r))


def fmt(v):
    return f'{np.mean(v):.4f}±{np.std(v):.4f}'


def main():
    tables = {ds: {k: [] for k in ['A', 'AP', 'ROC', 'C', 'D']} for ds in SEEDS}
    for ds, seeds in SEEDS.items():
        sl = SEQLEN[ds]
        for seed in seeds:
            zb = np.load(f'{PF}/{ds}_s{seed}.npz')
            zr = np.load(f'{PF}/{ds}_s{seed}_revin.npz')
            te, tr = zb['test_err'].astype(float), zb['train_err'].astype(float)
            lb = zb['label'].astype(int)
            rte = zr['test_err'].astype(float).mean(axis=1)
            rtr = np.load(f'{SC}/{ds}_s{seed}_revin.npz')['train_score'].astype(float)[::sl]
            g_te, gate_rate = gated_score(te, tr)
            g_tr, _ = gated_score(tr, tr)
            m = min(len(rtr), len(g_tr))
            sc_te = min_z(g_te, rte)
            sc_tr = min_z(g_tr[:m], rtr[:m])
            th = pot_threshold(sc_tr, sc_te, LM[ds])
            a = point2point(sc_te > th, lb > 0.1)[0]
            c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
            d = bf_pa(sc_te, lb)['f1']
            ap = average_precision_score(lb, sc_te)
            roc = roc_auc_score(lb, sc_te)
            for key, v in zip(['A', 'AP', 'ROC', 'C', 'D'], [a, ap, roc, c, d]):
                tables[ds][key].append(v)
            print(f'{ds} s{seed} [min_z] A={a:.4f} C={c:.4f} D={d:.4f} '
                  f'ROC={roc:.4f} AP={ap:.4f} POT_th={th:.4g} gate={gate_rate:.3f}',
                  flush=True)
    print('\n===== ISTAD min_z seed-mean（mean±std） =====')
    for ds, t in tables.items():
        print(f'{ds:5s} ' + '  '.join(f'{k}={fmt(v)}' for k, v in t.items()))


if __name__ == '__main__':
    main()
