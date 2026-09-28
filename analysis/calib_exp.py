#!/usr/bin/env python3
"""逐特征测试期自校准打分实验：ec = clip(e - q_c, 0)，q_c 取该特征测试集分位数
（中位数为主），剥离所有点共担的漂移成本、保留点内对比。
与 RevIN 的区别：不动输入、不抹全部幅度信息，只去公共水平。
第一遍：bf+PA / ROC / AP（快）；赢家再做 POT+PA(C) 与 raw F1(A)。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import gated_score

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SEEDS = {'PSM': ['87', '90', '98', '2021'], 'SMD': ['48', '2021', '2022', '2025'],
         'EXA': ['48', '89', '2021'], 'SWAT': ['48', '89', '2021']}


def calib(e, q):
    return np.clip(e - q[None, :], 0.0, None)


def trim1(e):
    idx = np.argmax(e, axis=1)
    return (e.sum(axis=1) - e[np.arange(len(e)), idx]) / (e.shape[1] - 1)


def topk(e, k):
    return np.sort(e, axis=1)[:, -k:].mean(axis=1)


def main():
    summary = {}
    for ds, seeds in SEEDS.items():
        print(f'\n===== {ds} =====', flush=True)
        acc = {}
        for seed in seeds:
            z = np.load(f'{PF}/{ds}_s{seed}.npz')
            te, tr = z['test_err'].astype(float), z['train_err'].astype(float)
            lb = z['label'].astype(int)
            q50 = np.percentile(te, 50, axis=0)
            q25 = np.percentile(te, 25, axis=0)
            q40 = np.percentile(te, 40, axis=0)
            em, e25, e40 = calib(te, q50), calib(te, q25), calib(te, q40)
            g_te, _ = gated_score(te, tr)
            variants = {
                'base_mean': te.mean(axis=1),
                'gated': g_te,
                'c50_mean': em.mean(axis=1),
                'c50_trim1': trim1(em),
                'c50_top2': topk(em, 2),
                'c50_top3': topk(em, 3),
                'c25_mean': e25.mean(axis=1),
                'c40_mean': e40.mean(axis=1),
            }
            print(f'-- seed {seed} --', flush=True)
            for name, sc in variants.items():
                d = bf_pa(sc, lb)['f1']
                roc = roc_auc_score(lb, sc)
                ap = average_precision_score(lb, sc)
                acc.setdefault(name, []).append((d, roc, ap))
                print(f'  {name:11s} D={d:.4f} ROC={roc:.4f} AP={ap:.4f}', flush=True)
        print(f'-- {ds} seed-mean --', flush=True)
        for name, rows in acc.items():
            ms = np.mean(rows, axis=0)
            summary[(ds, name)] = ms
            print(f'  {name:11s} D={ms[0]:.4f} ROC={ms[1]:.4f} AP={ms[2]:.4f}', flush=True)

    print('\n===== 汇总 =====', flush=True)
    for (ds, name), ms in summary.items():
        print(f'{ds:5s} {name:11s} D={ms[0]:.4f} ROC={ms[1]:.4f} AP={ms[2]:.4f}')


if __name__ == '__main__':
    main()
