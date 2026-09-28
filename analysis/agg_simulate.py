#!/usr/bin/env python3
"""离线聚合实验：在逐特征重建误差 (n, C) 上尝试不同聚合方式，
用与 bf_search_adaptive 完全一致的向量化 bf+PA 作为门控指标，
ROC/AP 作为回归保护。

用法: python agg_simulate.py <npz> [--ds PSM]
npz 需含 test_err (n,C), train_err (m,C), label (n,)
"""
import argparse
import bisect

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def segments_of(label):
    """返回异常段 [(start, end_exclusive), ...]"""
    label = np.asarray(label).reshape(-1)
    d = np.diff(np.concatenate([[0], label, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def bf_pa(score, label, coarse_step=200, fine_step=500):
    """完全复现 bf_search_adaptive(use_adjustment=True)：
    搜索范围 [max(min, p1*0.5), min(max, p99*2.0)]，粗 200 + 细 500（±10% 范围）。
    PA：段内任一点超阈值 → 整段命中。"""
    score = np.asarray(score, float).reshape(-1)
    label = np.asarray(label).reshape(-1)
    segs = segments_of(label)
    seg_starts = np.array([s for s, e in segs])
    seg_lens = np.array([e - s for s, e in segs], dtype=np.int64)
    segmax = np.array([score[s:e].max() for s, e in segs])
    total_anom = int(seg_lens.sum())
    normal_mask = label == 0
    sorted_normal = np.sort(score[normal_mask])
    n_normal = len(sorted_normal)

    def eval_grid(th):
        # TP/FN：段命中
        hit = segmax[None, :] > th[:, None]          # (T, S)
        tp = hit @ seg_lens                          # (T,)
        fn = total_anom - tp
        # FP：正常点超阈值
        fp = n_normal - np.searchsorted(sorted_normal, th, side='right')
        prec = tp / (tp + fp + 1e-5)
        rec = tp / (tp + fn + 1e-5)
        f1 = 2 * prec * rec / (prec + rec + 1e-5)
        return f1, prec, rec, tp, fp

    s_min, s_max = float(score.min()), float(score.max())
    p1, p99 = np.percentile(score, [1, 99])
    start = max(s_min, float(p1) * 0.5)
    end = min(s_max, float(p99) * 2.0)

    def search(lo, hi, steps):
        th = lo + np.arange(1, steps + 1) * (hi - lo) / steps
        f1, prec, rec, tp, fp = eval_grid(th)
        i = int(np.argmax(f1))  # argmax 取首个最大 → 与 `if f1 > best` 一致
        return dict(f1=float(f1[i]), prec=float(prec[i]), rec=float(rec[i]),
                    th=float(th[i]), tp=int(tp[i]), fp=int(fp[i]),
                    n_hit=int((segmax > th[i]).sum()), n_seg=len(segs))

    coarse = search(start, end, coarse_step)
    rng = end - start
    flo = max(start, coarse['th'] - 0.1 * rng)
    fhi = min(end, coarse['th'] + 0.1 * rng)
    fine = search(flo, fhi, fine_step)
    best = fine if fine['f1'] > coarse['f1'] else coarse
    best['stage'] = 'fine' if fine['f1'] > coarse['f1'] else 'coarse'
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--zone-feat', type=int, default=3, help='PSM 重尾特征（err>3 定义 zone）')
    ap.add_argument('--zone-th', type=float, default=3.0)
    args = ap.parse_args()

    z = np.load(args.npz)
    te, tr, lb = z['test_err'], z['train_err'], z['label']
    n, C = te.shape
    print(f'loaded {args.npz}: test_err {te.shape}, train_err {tr.shape}, '
          f'anom_frac={lb.mean():.4f}', flush=True)

    # zone（重尾区）
    zone = te[:, args.zone_feat] > args.zone_th
    print(f'zone(f{args.zone_feat}>{args.zone_th:g}): {zone.sum()} pts, '
          f'normal {((zone) & (lb == 0)).sum()}, anom {((zone) & (lb == 1)).sum()}')

    mu = tr.mean(axis=0)
    sd = tr.std(axis=0) + 1e-8
    zsc = np.clip((te - mu) / sd, 0, None)             # 逐特征训练标准化，截负
    zsc_nozf = zsc.copy(); zsc_nozf[:, args.zone_feat] = 0.0

    order = np.argsort(te, axis=1)  # 升序
    def trim(k):
        m = np.ones((n, C), bool)
        rows = np.arange(n)[:, None]
        m[rows, order[:, -k:]] = False
        return te[m].reshape(n, C - k).mean(axis=1)

    variants = {
        'mean': te.mean(axis=1),
        'median': np.median(te, axis=1),
        'trim1': trim(1),
        'trim2': trim(2),
        'zmean': zsc.mean(axis=1),
        'zmean_nozf': zsc_nozf.mean(axis=1),
        'zmedian': np.median(zsc, axis=1),
        'ztop5': np.sort(zsc, axis=1)[:, -5:].mean(axis=1),
    }

    print(f'\n{"variant":14s} {"bf+PA":>7s} {"P":>6s} {"R":>6s} {"th":>9s} '
          f'{"hit/seg":>8s} {"ROC":>6s} {"AP":>6s} {"np90":>7s} {"np99":>7s}')
    results = {}
    for name, sc in variants.items():
        nm = sc[lb == 0]
        r = bf_pa(sc, lb)
        roc = roc_auc_score(lb, sc)
        avgp = average_precision_score(lb, sc)
        results[name] = dict(f1=r['f1'], prec=r['prec'], rec=r['rec'], th=r['th'],
                             n_hit=r['n_hit'], n_seg=r['n_seg'], roc=roc, ap=avgp,
                             np90=float(np.percentile(nm, 90)),
                             np99=float(np.percentile(nm, 99)))
        print(f'{name:14s} {r["f1"]:7.4f} {r["prec"]:6.4f} {r["rec"]:6.4f} '
              f'{r["th"]:9.4f} {r["n_hit"]:>3d}/{r["n_seg"]:<4d} '
              f'{roc:6.4f} {avgp:6.4f} {results[name]["np90"]:7.3f} '
              f'{results[name]["np99"]:7.3f}', flush=True)

    # zone 内分离度：各变体在 zone 点上的 AUROC
    if zone.sum() > 0 and 0 < lb[zone].sum() < zone.sum():
        print('\nzone 内 AUROC（zone 点上的排序能力）:')
        for name, sc in variants.items():
            print(f'  {name:14s} {roc_auc_score(lb[zone], sc[zone]):.4f}')

    # 与 istad_scores 的回归校验（mean 应复现）
    try:
        import os
        s = args.npz.split('/')[-1].replace('.npz', '')
        ref = np.load(f'/data/modeluse/TS/vus_diag/istad_scores/{s}.npz')
        md = float(np.max(np.abs(variants['mean'] - ref['score'][:n].astype(float))))
        print(f'\n[regression] mean vs istad_scores max|Δ|={md:.2e}')
    except Exception as e:
        print(f'[regression] skip: {e}')


if __name__ == '__main__':
    main()
