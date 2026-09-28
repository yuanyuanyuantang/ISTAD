#!/usr/bin/env python3
"""PSM dual λ 扫描判定表（prereg_psm_lambda.md）。

臂：λ∈{0.5, 2.0} × 4 种子（dump MODE='dual_l05'/'dual_l20'）。
打分：dual min_z（min(z(gated_base), z(mean_revin))，与现行采纳同族，协议同 dual_tables.py）。
回归纪律：先用现存 λ=1.0 dual npz 复现 v2.2 PSM 行（5 指标），管线精确才信新数。
门槛（4 种子 mean，ddof=0）：主门槛 D≥0.9733；护栏 ROC≥0.7412 / AP≥0.4824 / C≥0.9464；
A 不设判据。
**采纳权在 lambda_valsel.py（base 路验证重建 MSE 选臂）**：仅入选臂可采纳；
未入选臂测试数字仅日志留档，若入选臂未过门槛 → KEEP λ=1.0（无论另一臂如何）。

用法: conda run -n tslib python lambda_tables.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
DS = 'PSM'
SEEDS = ['87', '90', '98', '2021']
KEYS = ['A', 'C', 'D', 'ROC', 'AP']
# v2.2 PSM 行（dual min_z 4 种子，dual_tables.py / 论文素材 §3.0g）
REF_V22 = dict(A=0.0996, C=0.9564, D=0.9696, ROC=0.7462, AP=0.4874)
GATES = dict(D=0.9733, ROC=0.7412, AP=0.4824, C=0.9464)


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
        zd = np.load(f'{PF}/PSM_s{s}{suffix}.npz')
        sc_te, sc_tr, lb, gate = minz_scores(zd)
        r = proto(sc_te, sc_tr, lb)
        for k in KEYS:
            tabs[k].append(r[k])
        print(f'PSM{label} s{s:4s} gate={gate:.3f} | A={r["A"]:.4f} C={r["C"]:.4f} '
              f'D={r["D"]:.4f} ROC={r["ROC"]:.4f} AP={r["AP"]:.4f}', flush=True)
    return tabs


def main():
    print('===== 1) 回归校验：λ=1.0 现行 dual npz 复现 v2.2 PSM 行 =====', flush=True)
    tabs_ref = arm_table('_dual', ' λ1.0')
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

    arms = {}
    for suffix, label in (('_dual_l05', 'λ=0.5'), ('_dual_l20', 'λ=2.0')):
        print(f'\n===== 2) 臂 {label}（{len(SEEDS)} 种子 mean±std）=====', flush=True)
        tabs = arm_table(suffix, f' {label}')
        arms[label] = {k: float(np.mean(tabs[k])) for k in KEYS}
        print(f'{label}  ' + '  '.join(f'{k}={fmt(tabs[k])}' for k in KEYS), flush=True)
        print('门槛: ' + '  '.join(
            f'{k} {arms[label][k]:.4f} {"≥" if arms[label][k] >= v else "<"} {v:.4f}'
            for k, v in GATES.items()), flush=True)

    print('\n===== 汇总（选择规则见 lambda_valsel.py；仅入选臂可采纳）=====')
    print('v2.2 现行 λ=1.0: D=0.9696 / C=0.9564 / ROC=0.7462 / AP=0.4874')
    for label, m in arms.items():
        passed = all(m[k] >= v for k, v in GATES.items())
        print(f'{label}: D={m["D"]:.4f} C={m["C"]:.4f} ROC={m["ROC"]:.4f} AP={m["AP"]:.4f} '
              f'A={m["A"]:.4f}  门槛{"全过 ✓" if passed else "未全过 ✗"}')
    print('SOTA 靶 TimesNet: D=0.9732±0.0008（护栏 ROC/AP = v2.2 行 −0.005，C −0.010）')


if __name__ == '__main__':
    main()
