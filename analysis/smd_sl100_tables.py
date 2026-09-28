#!/usr/bin/env python3
"""SMD dual sl100 判定表（prereg_smd_sl100.md）。

臂：SMD dual 现行采纳配方（λ=1.0 dual min_z）seq_len 96→100 × 4 种子（dump MODE='dual_sl100'）。
打分：dual min_z（与现行采纳同族，协议同 dual_tables_smd_exa.py）。
label 对齐：TEST 平铺 stride=W 前缀逐点对齐，dump 阶段已截断到现行评估面 708384（与 v2.1 同面）。
回归纪律：先用现存 SMD dual npz（sl96）复现 v2.2 SMD 行（5 指标），管线精确才信新数。
门槛（4 种子 mean，ddof=0）：主门槛 D≥0.8515；护栏 ROC≥0.7431 / AP≥0.1570 / C≥0.8094；
A 不设判据。

用法: conda run -n tslib python smd_sl100_tables.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
DS = 'SMD'
SEEDS = ['48', '2021', '2022', '2025']
KEYS = ['A', 'C', 'D', 'ROC', 'AP']
# v2.2 SMD 行（dual min_z 4 种子，dual_tables_smd_exa.py / 论文素材 §3.0g(7)）
REF_V22 = dict(A=0.0863, C=0.8194, D=0.8480, ROC=0.7481, AP=0.1620)
GATES = dict(D=0.8515, ROC=0.7431, AP=0.1570, C=0.8094)


def minz_scores(zd):
    lb = zd['label'].astype(int)
    g_te, gate = gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))
    g_tr, _ = gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))
    m_te = zd['test_err_revin'].astype(float).mean(axis=1)
    m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
    n = min(len(g_tr), len(m_tr))
    sc_te = np.minimum(z(g_te), z(m_te))
    sc_tr = np.minimum(z(g_tr[:n]), z(m_tr[:n]))
    return sc_te, sc_tr, lb, gate


def proto(sc_te, sc_tr, lb):
    th = pot_threshold(sc_tr, sc_te, LM[DS])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    return dict(A=a, C=c, D=d, ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te))


def fmt(vals):
    return f'{np.mean(vals):.4f}±{np.std(vals):.4f}'


def arm_table(suffix, label):
    tabs = {k: [] for k in KEYS}
    for s in SEEDS:
        zd = np.load(f'{PF}/{DS}_s{s}{suffix}.npz')
        sc_te, sc_tr, lb, gate = minz_scores(zd)
        r = proto(sc_te, sc_tr, lb)
        for k in KEYS:
            tabs[k].append(r[k])
        print(f'{DS}{label} s{s:4s} gate={gate:.3f} | A={r["A"]:.4f} C={r["C"]:.4f} '
              f'D={r["D"]:.4f} ROC={r["ROC"]:.4f} AP={r["AP"]:.4f}', flush=True)
    return tabs


def main():
    print('===== 1) 回归校验：SMD dual npz（sl96）复现 v2.2 SMD 行 =====', flush=True)
    tabs_ref = arm_table('_dual', ' sl96')
    allok = True
    for k in KEYS:
        got = float(np.mean(tabs_ref[k]))
        ok = abs(got - REF_V22[k]) < 0.003
        allok &= ok
        print(f'  ref {k}={REF_V22[k]:.4f} got {got:.4f} [{"OK" if ok else "MISMATCH"}]',
              flush=True)
    if not allok:
        print('>>> 回归失败：管线漂移，禁止采信新数字', flush=True)
        return

    print(f'\n===== 2) 臂 sl100（{len(SEEDS)} 种子 mean±std）=====', flush=True)
    tabs = arm_table('_dual_sl100', ' sl100')
    m = {k: float(np.mean(tabs[k])) for k in KEYS}
    print(f'sl100  ' + '  '.join(f'{k}={fmt(tabs[k])}' for k in KEYS), flush=True)
    print('门槛: ' + '  '.join(
        f'{k} {m[k]:.4f} {"≥" if m[k] >= v else "<"} {v:.4f}'
        for k, v in GATES.items()), flush=True)
    passed = all(m[k] >= v for k, v in GATES.items())
    print(f'\n>>> {"ADOPT sl100（SMD 最终配置更换，v2.3 刷新）" if passed else "KEEP sl96（未全过门槛，seq_len 轴全域关闭）"}')
    print('v2.2 现行 sl96: D=0.8480 / C=0.8194 / ROC=0.7481 / AP=0.1620')
    print('SOTA 靶 TimesNet: D=0.8514±0.0042（护栏 ROC/AP = v2.2 行 −0.005，C −0.010）')


if __name__ == '__main__':
    main()
