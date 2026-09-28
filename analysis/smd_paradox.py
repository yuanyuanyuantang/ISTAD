#!/usr/bin/env python3
"""SMD 悖论诊断：ISTAD ROC/AP 赢 TimesNet，但 D (bf+PA) 输。
对比分数形态：正常尾、段命中、near-miss、missed 段画像。
对每个共有的种子 (48/2021/2022) 配对比较。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa, segments_of
from final_matrix import gated_score

ISTAD_SC = '/data/modeluse/TS/vus_diag/istad_scores'
ISTAD_PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
TSL = '/data/modeluse/TS/vus_diag/tsl_pointwise'
SEEDS = ['48', '2021', '2022']


def tail_profile(score, label):
    nm = score[label == 0]
    return {k: float(np.percentile(nm, q)) for k, q in
            [('p90', 90), ('p99', 99), ('p999', 99.9), ('max', 100)]}


def seg_profile(score, label):
    segs = segments_of(label)
    segmax = np.array([score[s:e].max() for s, e in segs])
    seglen = np.array([e - s for s, e in segs])
    return segs, segmax, seglen


def bf_detail(score, label):
    """复现 bf_pa 搜索，并返回命中细节。"""
    from agg_simulate import bf_pa as _b
    res = _b(score, label)
    th = res['th']
    segs, segmax, seglen = seg_profile(score, label)
    hit = segmax > th
    nm = score[label == 0]
    fp = int((nm > th).sum())
    return {'f1': res['f1'], 'th': th, 'n_hit': int(hit.sum()),
            'n_seg': len(segs), 'fp': fp, 'hit': hit,
            'missed_len': int(seglen[~hit].sum()),
            'total_anom': int(seglen.sum())}


def near_miss(segmax, th, frac=0.5):
    """未命中段中，segmax 在 (frac*th, th] 的比例——差一口气的段。"""
    missed = segmax[segmax <= th]
    if len(missed) == 0:
        return 0, 0
    near = ((missed > frac * th)).sum()
    return int(near), len(missed)


def report(name, score, label):
    tp = tail_profile(score, label)
    d = bf_detail(score, label)
    segs, segmax, seglen = seg_profile(score, label)
    near, nmiss = near_miss(segmax, d['th'])
    hit_segmax = segmax[d['hit']]
    print(f'  [{name}]')
    print(f'    normal tail: p90={tp["p90"]:.4g} p99={tp["p99"]:.4g} '
          f'p999={tp["p999"]:.4g} max={tp["max"]:.4g}')
    print(f'    segmax     : p10={np.percentile(segmax,10):.4g} '
          f'p50={np.percentile(segmax,50):.4g} min={segmax.min():.4g}')
    print(f'    bf+PA      : F1={d["f1"]:.4f} th={d["th"]:.4g} '
          f'seg-hit {d["n_hit"]}/{d["n_seg"]} '
          f'recall_pts={(d["total_anom"]-d["missed_len"])/d["total_anom"]:.3f} '
          f'FP={d["fp"]}')
    print(f'    missed segs: {nmiss} 个，其中 segmax>0.5*th 的 near-miss {near} 个')
    return d, segs, segmax, seglen


def main():
    for seed in SEEDS:
        print(f'===== SMD seed {seed} =====')
        # ISTAD gated
        zb = np.load(f'{ISTAD_PF}/SMD_s{seed}.npz')
        g_te, gr = gated_score(zb['test_err'].astype(float),
                               zb['train_err'].astype(float))
        lb_i = zb['label'].astype(int)
        # TimesNet / DLinear
        tn = np.load(f'{TSL}/TimesNet_SMD_s{seed}.npz')
        dl = np.load(f'{TSL}/DLinear_SMD_s{seed}.npz')
        # TSLib sl=100 平铺 (708400) vs ISTAD sl=96 平铺 (708384)：截尾对齐
        m_len = min(len(tn['score']), len(dl['score']), len(g_te))
        tn_sc, tn_lb = tn['score'][:m_len].astype(float), tn['label'][:m_len].astype(int)
        dl_sc, dl_lb = dl['score'][:m_len].astype(float), dl['label'][:m_len].astype(int)
        g_te, lb_i = g_te[:m_len], lb_i[:m_len]
        assert (tn_lb == lb_i).all() and (dl_lb == lb_i).all()
        print(f'  n={len(g_te)} anom_frac={lb_i.mean():.4f} n_seg={len(segments_of(lb_i))} '
              f'gate={gr:.3f}')
        print(f'  ROC/AP: ISTAD {roc_auc_score(lb_i,g_te):.4f}/'
              f'{average_precision_score(lb_i,g_te):.4f} | '
              f'TimesNet {roc_auc_score(lb_i,tn_sc):.4f}/'
              f'{average_precision_score(lb_i,tn_sc):.4f} | '
              f'DLinear {roc_auc_score(lb_i,dl_sc):.4f}/'
              f'{average_precision_score(lb_i,dl_sc):.4f}')
        di, segs, smax_i, slen = report('ISTAD gated', g_te, lb_i)
        dt, _, smax_t, _ = report('TimesNet', tn_sc, lb_i)
        report('DLinear', dl_sc, lb_i)
        # 交叉分析：TimesNet 命中而 ISTAD miss 的段（反之亦然）
        both_hit = di['hit'] & dt['hit']
        t_only = dt['hit'] & ~di['hit']
        i_only = di['hit'] & ~dt['hit']
        print(f'    段交集: 共命中 {both_hit.sum()} | 仅 TimesNet 命中 {t_only.sum()} '
              f'({slen[t_only].sum()} pts) | 仅 ISTAD 命中 {i_only.sum()} '
              f'({slen[i_only].sum()} pts)')
        # near-miss 对比：ISTAD miss 段里，segmax 相对各自阈值的差距
        m = ~di['hit']
        if m.sum():
            ratio_i = (smax_i[m] / di['th'])
            ratio_t_of_same = (smax_t[m] / dt['th'])
            print(f'    ISTAD 的 {m.sum()} 个 miss 段: 自身 segmax/th 中位 '
                  f'{np.median(ratio_i):.3f}；同段在 TimesNet 的 segmax/th 中位 '
                  f'{np.median(ratio_t_of_same):.3f}')


if __name__ == '__main__':
    main()
