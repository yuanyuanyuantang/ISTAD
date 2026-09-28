#!/usr/bin/env python3
"""SWAT k25 宽度试点判定（预注册 vus_diag/prereg_swat_width.md）：
k25（--istad_kernel_size 25，329,340 params）vs k15 现行采纳（303,330 params），
同种子同管线配对（dump 逐特征误差 → gated 聚合 → 全协议 A/C/D/ROC/AP）。
门槛（3 种子 mean，ddof=0，四条全过才采纳）：
  D≥0.8764（DLinear 诚实参照）| AP≥0.73 | ROC≥0.84 | C≥0.8075（保 SWAT C 诚实第一）
任一不过 → KEEP k15；k25+其它打分变体不构成本预注册采纳理由。

用法: python swat_k25_tables.py   （依赖 k25 dump: istad_perfeat/SWAT_s*_k25.npz）
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
DS = 'SWAT'
SEEDS = ['48', '89', '2021']

# 预注册门槛
GATES = {'D': 0.8764, 'AP': 0.73, 'ROC': 0.84, 'C': 0.8075}
# k15 现行采纳行（统一表口径，ddof=1 in 素材；此处脚本 ddof=0 回归参照）
K15_REF = {'D': 0.8689, 'C': 0.8125, 'ROC': 0.8413, 'AP': 0.7317, 'A': 0.2583}


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


def gated_row(npz):
    zd = np.load(npz)
    lb = zd['label'].astype(int)
    te = zd['test_err'].astype(float)
    tr = zd['train_err'].astype(float)
    g_te, gate = gated_score(te, tr)
    g_tr, _ = gated_score(tr, tr)
    return g_te, g_tr, lb, gate


def main():
    tabs = {k: {'k25': [], 'k15': []} for k in ['A', 'C', 'D', 'ROC', 'AP']}
    for s in SEEDS:
        for tag in ('k25', 'k15'):
            fn = f'{PF}/{DS}_s{s}.npz' if tag == 'k15' else f'{PF}/{DS}_s{s}_{tag}.npz'
            g_te, g_tr, lb, gate = gated_row(fn)
            r = proto(g_te, g_tr, lb)
            for k in tabs:
                tabs[k][tag].append(r[k])
            print(f's{s:4s} [{tag}] gate={gate:.3f} A={r["A"]:.4f} C={r["C"]:.4f} '
                  f'D={r["D"]:.4f} ROC={r["ROC"]:.4f} AP={r["AP"]:.4f}', flush=True)

    print('\n===== SWAT k25 vs k15（gated，同管线配对）3 种子 mean±std =====')
    for k in ['A', 'C', 'D', 'ROC', 'AP']:
        print(f'{k:4s} k25={fmt(tabs[k]["k25"])}  k15={fmt(tabs[k]["k15"])}')

    print(f'\n[k15 回归校验] 现行采纳行 K15_REF={K15_REF}')
    for k, ref in K15_REF.items():
        m = np.mean(tabs[k]['k15'])
        flag = 'OK' if abs(m - ref) < 0.003 else 'DRIFT!'
        print(f'  {k:4s} {m:.4f} vs ref {ref:.4f}  {flag}')

    print('\n===== 预注册判定（k25 3 种子 mean，四条全过才采纳）=====')
    all_pass = True
    for k, gate in GATES.items():
        m = np.mean(tabs[k]['k25'])
        ok = m >= gate
        all_pass &= ok
        print(f'  {k:4s} {m:.4f} {"≥" if ok else "<"} {gate}  {"✓" if ok else "✗"}')
    print('\n判定: ' + ('ADOPT k25（SWAT 最终行换 k25 gated，需全表刷新）' if all_pass
                      else 'KEEP k15（v2.1 维持；k25 作负结果存档）'))


if __name__ == '__main__':
    main()
