#!/usr/bin/env python3
"""双头联合训练 4 种子终表（PSM）：fusion = min_z(gated_base, revin_mean)，
与 minz_tables.py 完全同协议（z 按各自 split、POT LM、bf_pa）。
对照列：联合模型 base 单视角（gated） vs 分开训练 base（istad_perfeat 原 dump）。
SOTA 靶: TimesNet C 0.9672±0.0008 / D 0.9732±0.0008。"""
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
    roc = roc_auc_score(lb, sc_te)
    ap = average_precision_score(lb, sc_te)
    return dict(A=a, C=c, D=d, ROC=roc, AP=ap)


def fmt(vals):
    return f'{np.mean(vals):.4f}±{np.std(vals):.4f}'


def main():
    tabs = {k: {'dual_minz': [], 'dual_base': [], 'sep_base': []}
            for k in ['A', 'C', 'D', 'ROC', 'AP']}
    for s in SEEDS:
        zd = np.load(f'{PF}/{DS}_s{s}_dual.npz')
        lb = zd['label'].astype(int)
        g_te, gate = gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))
        g_tr, _ = gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))
        m_te = zd['test_err_revin'].astype(float).mean(axis=1)
        m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
        n = min(len(g_tr), len(m_tr))
        sc_te = np.minimum(z(g_te), z(m_te))
        sc_tr = np.minimum(z(g_tr[:n]), z(m_tr[:n]))
        r_f = proto(sc_te, sc_tr, lb)
        r_b = proto(g_te, g_tr, lb)
        # 分开训练 base 对照
        zb = np.load(f'{PF}/{DS}_s{s}.npz')
        gs_te, _ = gated_score(zb['test_err'].astype(float), zb['train_err'].astype(float))
        gs_tr, _ = gated_score(zb['train_err'].astype(float), zb['train_err'].astype(float))
        r_s = proto(gs_te, gs_tr, lb)
        for k in tabs:
            tabs[k]['dual_minz'].append(r_f[k])
            tabs[k]['dual_base'].append(r_b[k])
            tabs[k]['sep_base'].append(r_s[k])
        print(f's{s:4s} gate={gate:.3f} | min_z: A={r_f["A"]:.4f} C={r_f["C"]:.4f} '
              f'D={r_f["D"]:.4f} ROC={r_f["ROC"]:.4f} AP={r_f["AP"]:.4f} | '
              f'base视角: D={r_b["D"]:.4f} ROC={r_b["ROC"]:.4f} | '
              f'[sep]base: D={r_s["D"]:.4f} ROC={r_s["ROC"]:.4f}', flush=True)

    print('\n===== PSM 双头联合 4 种子 mean±std =====')
    for k in ['A', 'C', 'D', 'ROC', 'AP']:
        t = tabs[k]
        print(f'{k:4s} dual_min_z={fmt(t["dual_minz"])}  '
              f'dual_base={fmt(t["dual_base"])}  sep_base={fmt(t["sep_base"])}')
    print('\n旧采纳(分开训练 min_z): A=0.1300±0.0134 C=0.9575±0.0065 '
          'D=0.9674±0.0045 ROC=0.7310±0.0323 AP=0.4765±0.0233')
    print('SOTA 靶 TimesNet:      C=0.9672±0.0008 D=0.9732±0.0008 '
          '(DLinear D 0.9715)')
    d = tabs['D']['dual_minz']
    c = tabs['C']['dual_minz']
    print(f'\n判定: D mean {np.mean(d):.4f} {"≥" if np.mean(d) >= 0.9732 else "<"} '
          f'0.9732 | C mean {np.mean(c):.4f} {"≥" if np.mean(c) >= 0.9672 else "<"} 0.9672')


if __name__ == '__main__':
    main()
