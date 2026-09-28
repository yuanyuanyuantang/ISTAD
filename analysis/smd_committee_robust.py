#!/usr/bin/env python3
"""SMD 委员会诚实性检查（零训练）：
存档 committee[3]（seeds 48/2021/2022，恰为列表前三、剔除弱种子 s2025）
达 C=0.7907/D=0.8577 双超 SOTA——但成员选择有 picked 嫌疑。
本脚本回答：不挑成员能否同样过线？
  1) 全体 4 种子的稳健聚合：median / trimmed-mean(去最高最低取中间2) / mean
  2) 全部 4 个 3-种子子集（子集扫描 = 成员敏感性）
  3) dual min_z 分数上的 committee（median/mean）
目标 D >= 0.8515 (TimesNet) 且 C >= 0.7826 (TranAD，保第一)。
"""
import itertools

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from agg_simulate import bf_pa
from final_matrix import LM, gated_score, point2point, pot_threshold
from minz_tables import z
from sota_push_scan import evaluate, segments_of

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = ['48', '2021', '2022', '2025']


def agg_scores(zs, how):
    zs = np.stack(zs)                      # (M, n)
    if how == 'mean':
        return zs.mean(axis=0)
    if how == 'median':
        return np.median(zs, axis=0)
    if how == 'trimmed':                   # 去每点最高最低，取中间均值（M>=3）
        s = np.sort(zs, axis=0)
        return s[1:-1].mean(axis=0)
    raise ValueError(how)


def report(tag, sc_te, sc_tr, lb):
    segs = segments_of(lb)
    r = evaluate('SMD', sc_te, sc_tr, lb, segs)
    ok = r['D'] >= 0.8515 and r['C'] >= 0.7826
    print(f'{tag:28s} A={r["A"]:.4f} C={r["C"]:.4f} D={r["D"]:.4f} '
          f'ROC={r["ROC"]:.4f} AP={r["AP"]:.4f}  '
          f'{"*** PASS C+D ***" if ok else ""}', flush=True)
    return r


def main():
    # sep-trained base dumps + gated（与存档 committee.py 同源）
    g = {}
    for s in SEEDS:
        zd = np.load(f'{PF}/SMD_s{s}.npz')
        g[s] = (gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))[0],
                gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))[0],
                zd['label'].astype(int))

    print('===== 1) 全体 4 种子稳健聚合（sep-base gated z）=====', flush=True)
    n = min(len(g[s][0]) for s in SEEDS)
    lb = g[SEEDS[0]][2][:n]
    zt = [z(g[s][0])[:n] for s in SEEDS]
    zr = [z(g[s][1])[:min(len(g[s][1]) for s in SEEDS)] for s in SEEDS]
    for how in ('mean', 'median', 'trimmed'):
        sc_te = agg_scores(zt, how)
        m = min(len(t) for t in zr)
        sc_tr = agg_scores([t[:m] for t in zr], how)
        report(f'committee4[{how}]', sc_te, sc_tr, lb)

    print('\n===== 2) 全部 4 个 3-种子子集（成员敏感性）=====', flush=True)
    for sub in itertools.combinations(SEEDS, 3):
        zt3 = [z(g[s][0])[:n] for s in sub]
        sc_te = np.mean(zt3, axis=0)
        trs = [z(g[s][1]) for s in sub]
        m = min(len(t) for t in trs)
        sc_tr = np.mean([t[:m] for t in trs], axis=0)
        report(f'committee3{{{",".join(sub)}}}', sc_te, sc_tr, lb)

    print('\n===== 3) dual min_z 分数上的 committee =====', flush=True)
    mz = {}
    for s in SEEDS:
        zd = np.load(f'{PF}/SMD_s{s}_dual.npz')
        te, tr = zd['test_err'].astype(float), zd['train_err'].astype(float)
        g_te, _ = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        m_te = zd['test_err_revin'].astype(float).mean(axis=1)
        m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
        nn = min(len(g_tr), len(m_tr))
        mz[s] = (np.minimum(z(g_te), z(m_te)),
                 np.minimum(z(g_tr[:nn]), z(m_tr[:nn])),
                 zd['label'].astype(int))
    n4 = min(len(mz[s][0]) for s in SEEDS)
    lb4 = mz[SEEDS[0]][2][:n4]
    zt4 = [z(mz[s][0])[:n4] for s in SEEDS]
    for how in ('mean', 'median', 'trimmed'):
        sc_te = agg_scores(zt4, how)
        trs = [z(mz[s][1]) for s in SEEDS]
        m = min(len(t) for t in trs)
        sc_tr = agg_scores([t[:m] for t in trs], how)
        report(f'mz-committee4[{how}]', sc_te, sc_tr, lb4)
    for sub in itertools.combinations(SEEDS, 3):
        zt3 = [z(mz[s][0])[:n4] for s in sub]
        sc_te = np.mean(zt3, axis=0)
        trs = [z(mz[s][1]) for s in sub]
        m = min(len(t) for t in trs)
        sc_tr = np.mean([t[:m] for t in trs], axis=0)
        report(f'mz-committee3{{{",".join(sub)}}}', sc_te, sc_tr, lb4)

    print('\n参照: 现行 v2.1 min_z 单模型 A=0.0863 C=0.8194 D=0.8480 ROC=0.7481 AP=0.1620')
    print('目标: D>=0.8515 (TimesNet) 且 C>=0.7826 (TranAD, 保第一)')
    print('\n[done]', flush=True)


if __name__ == '__main__':
    main()
