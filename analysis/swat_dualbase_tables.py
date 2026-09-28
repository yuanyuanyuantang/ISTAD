#!/usr/bin/env python3
"""SWAT dual_base 官方采纳表（2026-09-03，事后采纳，用户决策）。

dual_base = dual 联合模型 base 视角 + gated 聚合（与 §3.0g(4) 侧观察同源）。
本脚本按 dual_tables_swat.py 同款协议产出逐种子 + mean±std 官方行，
并回归校验（gated 单模型行必须逐位复现 v2.1 定稿行）。

协议：POT 逐数据集 LM 表（SWAT: 0.9999/1.2）、bf_pa 200/500+adjust、
adjust_predicts 段级 PA。零训练：SWAT_s{48,89,2021}_dual.npz 已有转储。
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from agg_simulate import bf_pa
from final_matrix import LM, gated_score, point2point, pot_threshold
from sota_push_scan import adjust_predicts_fast, segments_of

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = ['48', '89', '2021']
DS = 'SWAT'
REF_GATED = dict(A=0.2583, C=0.8125, D=0.8689, ROC=0.8413, AP=0.7317)


def proto(sc_te, sc_tr, lb):
    segs = segments_of(lb)
    th = pot_threshold(sc_tr, sc_te, LM[DS])
    a = point2point(sc_te > th, lb > 0.1)[0]
    pred = adjust_predicts_fast(sc_te, lb, th, segs)
    c = point2point(pred, lb > 0.1)[0]
    d = bf_pa(sc_te, lb)
    return dict(A=a, C=c, D=d['f1'], ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te),
                hit=f'{d["n_hit"]}/{d["n_seg"]}')


def fmt(vals):
    return f'{np.mean(vals):.4f}±{np.std(vals):.4f}'


def main():
    keys = ['A', 'C', 'D', 'ROC', 'AP']
    tabs = {'gated': {k: [] for k in keys}, 'dual_base': {k: [] for k in keys}}
    # 严格两源：base npz = 分开训练模型（gated = v2.1 定稿行）
    #           dual npz = 联合训练模型 base 视角（gated 聚合后 = dual_base 采纳行）
    print(f'===== {DS} 严格两源对照（base npz = 分开训练 / dual npz = 联合训练）=====',
          flush=True)
    for s in SEEDS:
        zb = np.load(f'{PF}/{DS}_s{s}.npz')
        te, tr, lb = (zb['test_err'].astype(float), zb['train_err'].astype(float),
                      zb['label'].astype(int))
        g_te, gate_b = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        r_g = proto(g_te, g_tr, lb)
        zd = np.load(f'{PF}/{DS}_s{s}_dual.npz')
        te2, tr2, lb2 = (zd['test_err'].astype(float), zd['train_err'].astype(float),
                         zd['label'].astype(int))
        d_te, gate_d = gated_score(te2, tr2)
        d_tr, _ = gated_score(tr2, tr2)
        r_d = proto(d_te, d_tr, lb2)
        for k in keys:
            tabs['gated'][k].append(r_g[k])
            tabs['dual_base'][k].append(r_d[k])
        print(f's{s:4s} gate(base)={gate_b:.3f} gate(dual)={gate_d:.3f} | '
              f'gated: A={r_g["A"]:.4f} C={r_g["C"]:.4f} D={r_g["D"]:.4f} '
              f'ROC={r_g["ROC"]:.4f} AP={r_g["AP"]:.4f} hit={r_g["hit"]} | '
              f'dual_base: A={r_d["A"]:.4f} C={r_d["C"]:.4f} D={r_d["D"]:.4f} '
              f'ROC={r_d["ROC"]:.4f} AP={r_d["AP"]:.4f} hit={r_d["hit"]}', flush=True)

    print(f'\n===== {DS} mean±std =====', flush=True)
    for name in ('gated', 'dual_base'):
        print(f'{name:10s} ' + '  '.join(f'{k}={fmt(tabs[name][k])}' for k in keys),
              flush=True)

    print('\n===== gated 回归校验（v2.1 定稿行）=====', flush=True)
    allok = True
    for k, v in REF_GATED.items():
        got = float(np.mean(tabs['gated'][k]))
        ok = abs(got - v) < 0.003
        allok &= ok
        print(f'  ref {k}={v:.4f} got {got:.4f} [{"OK" if ok else "MISMATCH"}]', flush=True)
    print(f'>>> 回归 {"通过" if allok else "失败"}；'
          f'dual_base 官方行 = ' +
          '  '.join(f'{k}={fmt(tabs["dual_base"][k])}' for k in keys), flush=True)


if __name__ == '__main__':
    main()
