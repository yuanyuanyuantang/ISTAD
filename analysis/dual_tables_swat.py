#!/usr/bin/env python3
"""SWAT 双头联合 3 种子判定表：fusion = min_z(gated_base, revin_mean)，
与 dual_tables.py / minz_tables.py 完全同协议（z 按各自 split、POT LM、bf_pa）。

预注册采纳判据（三者同时满足才采纳，否则按无增益回退、维持 gated）：
  AP >= 0.70 且 ROC >= 0.82 且 D >= 0.869
  （当前 gated 采纳值：AP 0.7317、ROC 0.8413、D 0.8689）
诊断判据（§3.0f 讨论点）：revin 第二视角单独 ROC >= 0.7 且正相关才有互补；
分开训练时 SWAT revin 视角 ROC≈0.24（反相关），预期 dual 也救不回。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
DS = 'SWAT'
SEEDS = ['48', '89', '2021']
CRIT = dict(AP=0.70, ROC=0.82, D=0.869)          # 预注册采纳线
GATED = dict(AP=0.7317, ROC=0.8413, D=0.8689)    # 当前采纳（gated）


def proto(sc_te, sc_tr, lb):
    th = pot_threshold(sc_tr, sc_te, LM[DS])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    return dict(A=a, C=c, D=d, ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te))


def fmt(vals):
    return f'{np.mean(vals):.4f}±{np.std(vals):.4f}'


def main():
    tabs = {k: {'dual_minz': [], 'dual_base': [], 'sep_base': []}
            for k in ['A', 'C', 'D', 'ROC', 'AP']}
    revin_rocs, revin_aps = [], []
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
        revin_rocs.append(roc_auc_score(lb, m_te))
        revin_aps.append(average_precision_score(lb, m_te))
        for k in tabs:
            tabs[k]['dual_minz'].append(r_f[k])
            tabs[k]['dual_base'].append(r_b[k])
            tabs[k]['sep_base'].append(r_s[k])
        print(f's{s:4s} gate={gate:.3f} revinROC={revin_rocs[-1]:.4f} | '
              f'min_z: A={r_f["A"]:.4f} C={r_f["C"]:.4f} D={r_f["D"]:.4f} '
              f'ROC={r_f["ROC"]:.4f} AP={r_f["AP"]:.4f} | '
              f'base视角: D={r_b["D"]:.4f} ROC={r_b["ROC"]:.4f} | '
              f'[sep]base: D={r_s["D"]:.4f} ROC={r_s["ROC"]:.4f}', flush=True)

    print('\n===== SWAT 双头联合 3 种子 mean±std =====')
    for k in ['A', 'C', 'D', 'ROC', 'AP']:
        t = tabs[k]
        print(f'{k:4s} dual_min_z={fmt(t["dual_minz"])}  '
              f'dual_base={fmt(t["dual_base"])}  sep_base={fmt(t["sep_base"])}')
    print(f'revin 第二视角单独: ROC={np.mean(revin_rocs):.4f} AP={np.mean(revin_aps):.4f}'
          f'（分开训练 ~0.24/~0.09，诊断判据要求 >=0.7）')
    print(f'当前 gated 采纳: AP={GATED["AP"]:.4f} ROC={GATED["ROC"]:.4f} D={GATED["D"]:.4f}')

    m = {k: float(np.mean(tabs[k]['dual_minz'])) for k in ['D', 'ROC', 'AP']}
    ok = m['AP'] >= CRIT['AP'] and m['ROC'] >= CRIT['ROC'] and m['D'] >= CRIT['D']
    print('\n=== 预注册判定 ===')
    print(f'AP {m["AP"]:.4f} {">=" if m["AP"] >= CRIT["AP"] else "<"} {CRIT["AP"]} | '
          f'ROC {m["ROC"]:.4f} {">=" if m["ROC"] >= CRIT["ROC"] else "<"} {CRIT["ROC"]} | '
          f'D {m["D"]:.4f} {">=" if m["D"] >= CRIT["D"] else "<"} {CRIT["D"]}')
    if ok:
        print('>>> ADOPT：dual min_z 三项全过，SWAT 采纳双头（更新最终表）')
    else:
        print('>>> KEEP GATED：未全过预注册线（预期内——水平型异常被窗口归一化抹掉，'
              '第二视角反相关），SWAT 维持 gated，双头线在 SWAT 终止')


if __name__ == '__main__':
    main()
