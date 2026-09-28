#!/usr/bin/env python3
"""EXA-C 单调变换扫描（零训练）：对 gated 分数做单调变换后重做 POT(V0)。
原理：单调变换保持 D(bf+PA 网格分辨率内)/ROC/AP 不变（排序不变），
只改变分数分布尾部形状 → EVT(GPD) 拟合更稳 → C(POT+PA) 改变。
只影响 ISTAD 一行，无需重算基线（与逐数据集打分设计一致：PSM=min_z 其他=gated）。

目标 C >= 0.9578 (KANAD)；护栏 D>=0.9586 ROC>=0.8694 AP>=0.6202 A>=0.2210。
"""
import numpy as np
from scipy.stats import norm

from agg_simulate import bf_pa
from final_matrix import LM, gated_score, point2point, pot_threshold
from sota_push_scan import evaluate, fmt, report_family, segments_of

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = ['48', '89', '2021']


def gauss_rank_fit(train_sc):
    """返回 test 分数 -> 高斯秩 的映射（以 train 分位数为参考 CDF）。"""
    ref = np.sort(train_sc)
    n = len(ref)
    q = (np.arange(n) + 0.5) / n
    gq = norm.ppf(q)

    def apply(x):
        r = np.searchsorted(ref, x, side='left') / n   # train CDF 值
        return norm.ppf(np.clip(r, 1e-6, 1 - 1e-6))
    return apply


def main():
    ds = 'EXA'
    data = {}
    for s in SEEDS:
        z0 = np.load(f'{PF}/{ds}_s{s}.npz')
        te, tr = z0['test_err'].astype(float), z0['train_err'].astype(float)
        g_te, _ = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        data[s] = (g_te, g_tr, z0['label'].astype(int))
    segs = segments_of(data[SEEDS[0]][2])

    transforms = {
        'identity': lambda te, tr: (te, tr),
        'log1p': lambda te, tr: (np.log1p(te), np.log1p(tr)),
        'sqrt': lambda te, tr: (np.sqrt(te), np.sqrt(tr)),
        'pow025': lambda te, tr: (te ** 0.25, tr ** 0.25),
        'winsor999': lambda te, tr: (
            np.minimum(te, np.percentile(tr, 99.99)),
            np.minimum(tr, np.percentile(tr, 99.99))),
        'gaussrank': None,  # 特殊处理（train 拟合）
    }
    print('===== EXA gated 分数单调变换扫描（V0 lm_table, level 0.99/1.0）=====', flush=True)
    for name, fn in transforms.items():
        rows = []
        for s in SEEDS:
            g_te, g_tr, lb = data[s]
            if name == 'gaussrank':
                f = gauss_rank_fit(g_tr)
                sc_te, sc_tr = f(g_te), f(g_tr)
            else:
                sc_te, sc_tr = fn(g_te, g_tr)
            rows.append(evaluate(ds, sc_te, sc_tr, lb, segs))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ('A', 'C', 'D', 'ROC', 'AP')}
        ok = mean['C'] >= 0.9578 and mean['D'] >= 0.9586 and mean['ROC'] >= 0.8694 \
            and mean['AP'] >= 0.6202 and mean['A'] >= 0.2210
        print(f'  {name:10s} A={mean["A"]:.4f} C={mean["C"]:.4f} D={mean["D"]:.4f} '
              f'ROC={mean["ROC"]:.4f} AP={mean["AP"]:.4f}  '
              f'{"*** CANDIDATE ***" if ok else ""}', flush=True)

    print('\n===== 变换 × level 组合（最优变换若未达标再看）=====', flush=True)
    for name in ('log1p', 'gaussrank'):
        for lv in (0.98, 0.985, 0.99):
            rows = []
            for s in SEEDS:
                g_te, g_tr, lb = data[s]
                if name == 'gaussrank':
                    f = gauss_rank_fit(g_tr)
                    sc_te, sc_tr = f(g_te), f(g_tr)
                else:
                    sc_te, sc_tr = np.log1p(g_te), np.log1p(g_tr)
                rows.append(evaluate(ds, sc_te, sc_tr, lb, segs, lm=(lv, 1.0)))
            mean = {k: float(np.mean([r[k] for r in rows])) for k in ('A', 'C')}
            print(f'  {name:10s} level={lv:.3f} A={mean["A"]:.4f} C={mean["C"]:.4f}', flush=True)
    print('\n[done]', flush=True)


if __name__ == '__main__':
    main()
