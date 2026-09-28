#!/usr/bin/env python3
"""混合打分实验（PSM，离线）：base 家族捕获水平异常但带重尾，RevIN 家族对漂移不变
但丢水平信息。逐点混合策略：
- max_z / min_z / mean_z：z 化后逐点 max / min / mean
- rank_mean：秩平均
- gate_swap：gate 区（任一特征 > 训练 p99.9）用 revin 分数，区外用 gated 分数
  （漂移区信任平移不变分支，区外信任保排序分支）
第一遍：D / ROC / AP + bf 最优阈值下的段命中统计。"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import gated_score

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = ['87', '90', '2021']


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def main():
    acc = {}
    for seed in SEEDS:
        zb_npz = np.load(f'{PF}/PSM_s{seed}.npz')
        zr_npz = np.load(f'{PF}/PSM_s{seed}_revin.npz')
        te, tr = zb_npz['test_err'].astype(float), zb_npz['train_err'].astype(float)
        lb = zb_npz['label'].astype(int)
        rte = zr_npz['test_err'].astype(float).mean(axis=1)
        zg, gate = gated_score(te, tr)
        zr = rte
        gz, rz = z(zg), z(zr)
        gr = np.percentile(tr, 99.9, axis=0)
        variants = {
            'gated_ref': zg,
            'revin_ref': zr,
            'max_z': np.maximum(gz, rz),
            'min_z': np.minimum(gz, rz),
            'mean_z': (gz + rz) / 2,
            'wmean_03': rz + 0.3 * gz,
            'rank_mean': (rankdata(gz) + rankdata(rz)) / 2,
            'gate_swap': np.where(gate, rz, gz),
            'gate_or': np.where(gate, np.maximum(gz, rz), gz),
        }
        print(f'===== seed {seed} (gate={gate.mean():.3f}) =====', flush=True)
        for name, sc in variants.items():
            r = bf_pa(sc, lb)
            roc = roc_auc_score(lb, sc)
            ap = average_precision_score(lb, sc)
            nm = sc[lb == 0]
            acc.setdefault(name, []).append((r['f1'], roc, ap, r['n_hit'], r['n_seg'], r['th'],
                                             float(np.percentile(nm, 99))))
            print(f'  {name:10s} D={r["f1"]:.4f} ROC={roc:.4f} AP={ap:.4f} '
                  f'hit={r["n_hit"]}/{r["n_seg"]} th={r["th"]:.4g} np99={np.percentile(nm, 99):.4g}',
                  flush=True)
    print('\n===== PSM seed-mean =====', flush=True)
    for name, rows in acc.items():
        m = np.mean(rows, axis=0)
        print(f'  {name:10s} D={m[0]:.4f} ROC={m[1]:.4f} AP={m[2]:.4f} '
              f'hit={m[3]:.1f}/{m[4]:.0f} np99={m[6]:.4g}')


if __name__ == '__main__':
    main()
