#!/usr/bin/env python3
"""SMD 悖论诊断 II：
(1) 阈值无关的 FP-budget 曲线——阈值 = 第 k 大正常分，数段命中；
    回答"段分离度是否本质劣于 TimesNet"。
(2) 零重训聚合家族在 SMD 上的 bf+PA / ROC / AP：mean(现 gated)、top-k z、
    L2、max、gated 变体——寻找无需重训即可闭合 D 缺口的聚合。
(3) miss 段画像：长度、TimesNet 命中而 ISTAD miss 的段在各聚合下的命中情况。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa, segments_of
from final_matrix import gated_score

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
TSL = '/data/modeluse/TS/vus_diag/tsl_pointwise'
SEEDS = ['48', '2021', '2022']
FP_BUDGETS = [500, 1000, 3000, 10000]


def fp_budget_hits(score, label, budgets):
    nm = np.sort(score[label == 0])[::-1]          # 正常分降序
    segs = segments_of(label)
    segmax = np.array([score[s:e].max() for s, e in segs])
    seglen = np.array([e - s for s, e in segs])
    out = {}
    for k in budgets:
        th = nm[min(k, len(nm) - 1)]
        hit = segmax > th
        tp = seglen[hit].sum()
        prec = tp / (tp + k)
        rec = tp / seglen.sum()
        out[k] = (hit.sum(), prec, rec)
    return out


def aggregations(te, tr):
    """零重训聚合家族：在 (n, C) 逐特征误差上。"""
    mu, sd = tr.mean(axis=0), tr.std(axis=0) + 1e-8
    z = np.clip((te - mu) / sd, 0, None)           # 逐特征训练标准化截负
    C = te.shape[1]
    order_z = np.argsort(z, axis=1)
    fam = {'gated': gated_score(te, tr)[0],
           'mean': te.mean(axis=1),
           'zmean': z.mean(axis=1),
           'ztop1': z[np.arange(len(z)), order_z[:, -1]],
           'ztop3': np.take_along_axis(z, order_z[:, -3:], 1).mean(axis=1),
           'ztop5': np.take_along_axis(z, order_z[:, -5:], 1).mean(axis=1),
           'zl2': np.sqrt((z ** 2).sum(axis=1)),
           'rawtop3': np.sort(te, axis=1)[:, -3:].mean(axis=1)}
    return fam


def main():
    for seed in SEEDS:
        print(f'===== SMD seed {seed} =====')
        zb = np.load(f'{PF}/SMD_s{seed}.npz')
        te, tr = zb['test_err'].astype(float), zb['train_err'].astype(float)
        lb = zb['label'].astype(int)
        m_len = len(lb)
        tn = np.load(f'{TSL}/TimesNet_SMD_s{seed}.npz')
        tn_sc = tn['score'][:m_len].astype(float)
        assert (tn['label'][:m_len].astype(int) == lb).all()

        # (1) FP-budget 曲线
        print('  FP-budget 段命中（阈值=第k大正常分）:')
        bi = fp_budget_hits(gated_score(te, tr)[0], lb, FP_BUDGETS)
        bt = fp_budget_hits(tn_sc, lb, FP_BUDGETS)
        for k in FP_BUDGETS:
            hi, pi, ri = bi[k]; ht, pt, rt = bt[k]
            print(f'    FP≤{k:6d}: ISTAD hit={hi:3d} P={pi:.3f} R={ri:.3f} | '
                  f'TimesNet hit={ht:3d} P={pt:.3f} R={rt:.3f}')

        # (2) 聚合家族
        fam = aggregations(te, tr)
        print('  聚合家族（bf+PA / ROC / AP）:')
        rows = {}
        for name, sc in fam.items():
            d = bf_pa(sc, lb)
            roc = roc_auc_score(lb, sc); ap = average_precision_score(lb, sc)
            rows[name] = d
            print(f'    {name:8s} D={d["f1"]:.4f} th={d["th"]:.4g} '
                  f'hit={d["n_hit"]}/{d["n_seg"]} FP={d["fp"]} '
                  f'ROC={roc:.4f} AP={ap:.4f}')

        # (3) miss 段画像（以最优聚合为准）
        best_name = max(rows, key=lambda n: rows[n]['f1'])
        sc = fam[best_name]
        segs = segments_of(lb)
        segmax_i = np.array([sc[s:e].max() for s, e in segs])
        segmax_t = np.array([tn_sc[s:e].max() for s, e in segs])
        seglen = np.array([e - s for s, e in segs])
        th_i = rows[best_name]['th']
        d_t = bf_pa(tn_sc, lb)
        miss_i = segmax_i <= th_i
        hit_t = segmax_t > d_t['th']
        print(f'  [{best_name}] miss {miss_i.sum()} 段（{seglen[miss_i].sum()} pts）；'
              f'其中 TimesNet 能命中 {(miss_i & hit_t).sum()} 段；'
              f'miss 段长度中位 {np.median(seglen[miss_i]):.0f} '
              f'(全体 {np.median(seglen):.0f})')


if __name__ == '__main__':
    main()
