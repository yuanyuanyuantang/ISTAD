#!/usr/bin/env python3
"""双头联合训练 pilot 评估（PSM s87）：
同一模型出两路逐特征误差（base / revin），离线试融合家族，全协议 A/C/D/ROC/AP。
对照（同种子、分开训练）：base gated、revin 单视角、min_z（已采纳配方）。

预注册判据：
  PASS : 存在融合 D ≥ 0.973 且 ROC ≥ 0.72（追平/超 TimesNet 0.9732）
  GRAY : D ≥ 0.968 且 ROC ≥ 0.72 但 <0.973 —— 若联合模型 revin 单视角 ROC ≥ 0.70
         （双视角准则成立，分开训练时仅 ~0.55）→ 继续 SWAT pilot；否则终止
  KILL : 无任何融合满足 D ≥ 0.968 且 ROC ≥ 0.72
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SC = '/data/modeluse/TS/vus_diag/istad_scores'
DS, SEED = 'PSM', '87'
SL = 64


def full_proto(sc_te, sc_tr, lb, ds='PSM'):
    th = pot_threshold(sc_tr, sc_te, LM[ds])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    return dict(A=a, C=c, D=d, ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te), th=th)


def line(name, r):
    print(f'{name:24s} A={r["A"]:.4f} C={r["C"]:.4f} D={r["D"]:.4f} '
          f'ROC={r["ROC"]:.4f} AP={r["AP"]:.4f} POT_th={r["th"]:.4g}')


def main():
    zd = np.load(f'{PF}/{DS}_s{SEED}_dual.npz')
    lb = zd['label'].astype(int)
    b_te, b_tr = zd['test_err'].astype(float), zd['train_err'].astype(float)
    r_te, r_tr = zd['test_err_revin'].astype(float), zd['train_err_revin'].astype(float)

    # 两路各自的聚合
    gb_te, gate = gated_score(b_te, b_tr)
    gb_tr, _ = gated_score(b_tr, b_tr)
    gr_te, _ = gated_score(r_te, r_tr)
    gr_tr, _ = gated_score(r_tr, r_tr)
    m_te = r_te.mean(axis=1)   # revin 视角传统口径（mean）
    m_tr = r_tr.mean(axis=1)[::1]

    print('=== 单视角（联合模型） ===')
    line('base gated', full_proto(gb_te, gb_tr, lb))
    line('revin mean', full_proto(m_te, m_tr, lb))
    line('revin gated', full_proto(gr_te, gr_tr, lb))

    print('\n=== 融合家族（z 化按各自 split） ===')
    # revin train 侧抽稀对齐（与 minz_tables 相同处理）
    results = {}

    def fused(name, mk):
        # mk(zb_te, zr_te) -> sc_te; 训练侧同式
        # 训练侧长度对齐：gated train 与 revin-mean train 长度可能不同
        n = min(len(gb_tr), len(m_tr))
        sc_te = mk(z(gb_te), z(m_te))
        sc_tr = mk(z(gb_tr[:n]), z(m_tr[:n]))
        r = full_proto(sc_te, sc_tr, lb)
        results[name] = r
        line(name, r)

    fused('min_z(base,revin_mean)', lambda a, b: np.minimum(a, b))
    fused('mean_z', lambda a, b: (a + b) / 2)
    for w in (0.3, 0.5, 0.7):
        fused(f'wmean base*{w:.1f}', lambda a, b, w=w: w * a + (1 - w) * b)
    # gated-revin 第二视角的 min_z
    n2 = min(len(gb_tr), len(gr_tr))
    sc_te = np.minimum(z(gb_te), z(gr_te))
    sc_tr = np.minimum(z(gb_tr[:n2]), z(gr_tr[:n2]))
    r = full_proto(sc_te, sc_tr, lb)
    results['min_z(base,revin_gated)'] = r
    line('min_z(base,revin_gated)', r)

    print('\n=== 对照：同种子分开训练 ===')
    zb = np.load(f'{PF}/{DS}_s{SEED}.npz')          # 分开训练 base
    zr = np.load(f'{PF}/{DS}_s{SEED}_revin.npz')    # 分开训练 revin
    g_te, _ = gated_score(zb['test_err'].astype(float), zb['train_err'].astype(float))
    g_tr, _ = gated_score(zb['train_err'].astype(float), zb['train_err'].astype(float))
    line('[sep] base gated', full_proto(g_te, g_tr, lb))
    rm_te = zr['test_err'].astype(float).mean(axis=1)
    rm_tr = np.load(f'{SC}/{DS}_s{SEED}_revin.npz')['train_score'].astype(float)[::SL]
    line('[sep] revin mean', full_proto(rm_te, rm_tr, lb))
    n3 = min(len(g_tr), len(rm_tr))
    sc_te = np.minimum(z(g_te), z(rm_te))
    sc_tr = np.minimum(z(g_tr[:n3]), z(rm_tr[:n3]))
    line('[sep] min_z', full_proto(sc_te, sc_tr, lb))

    print('\n=== kill-switch 判定 ===')
    best = max(results.items(), key=lambda kv: (kv[1]['D'] if kv[1]['ROC'] >= 0.72 else -1))
    name, r = best
    revin_roc = roc_auc_score(lb, m_te)
    print(f'联合模型 revin 单视角 ROC = {revin_roc:.4f}（分开训练 ~0.55）')
    print(f'满足 ROC≥0.72 的最优融合: {name} D={r["D"]:.4f} ROC={r["ROC"]:.4f}')
    if r['D'] >= 0.973 and r['ROC'] >= 0.72:
        print('>>> PASS：追平/超 TimesNet D 0.9732，扩展到全种子')
    elif r['D'] >= 0.968 and r['ROC'] >= 0.72:
        verdict = '继续 SWAT pilot（双视角准则成立）' if revin_roc >= 0.70 else '机制未修复 → 终止线'
        print(f'>>> GRAY：{verdict}')
    else:
        print('>>> KILL：无融合满足 D≥0.968 且 ROC≥0.72，双头联合线终止')


if __name__ == '__main__':
    main()
