#!/usr/bin/env python3
"""Route B 终局决策矩阵：在逐特征误差上重算 base/trim1/gated(/revin/fusion) 的
C(POT+PA) D(bf+PA) A(POT 阈值、无 PA 的 raw F1) ROC AP，全数据集。
gated 规则：任一特征误差 > 该特征训练 p99.9 → trim1，否则 mean。
"""
import importlib.util

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa

spec = importlib.util.spec_from_file_location(
    'spot_mod', '/data/modeluse/TS/TranAD_improve/src/spot.py')
spot_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spot_mod)
SPOT = spot_mod.SPOT

LM = {'SMD': (0.988, 1.00), 'SWAT': (0.9999, 1.2), 'PSM': (0.98, 0.9), 'EXA': (0.99, 1.0)}
Q = 1e-5
SEEDS = {'PSM': ['87', '90', '2021'], 'SMD': ['48', '2021', '2022', '2025'],
         'EXA': ['48', '89', '2021'], 'SWAT': ['48', '89', '2021']}
PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
SC = '/data/modeluse/TS/vus_diag/istad_scores'


def adjust_predicts(score, label, threshold):
    score = np.asarray(score)
    label = np.asarray(label)
    predict = score > threshold
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict


def point2point(predict, actual):
    TP = np.sum(predict * actual)
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return f1, precision, recall


def pot_threshold(init_score, score, lm):
    lms = lm[0]
    while True:
        try:
            s = SPOT(Q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except Exception:
            lms = lms * 0.999
        else:
            break
    ret = s.run(dynamic=False)
    return float(np.mean(ret['thresholds']) * lm[1])


def trim1_score(err):
    n, C = err.shape
    mx = err.max(axis=1, keepdims=True)
    # 去掉每点最大特征（若并列则去第一个最大值）
    idx = np.argmax(err, axis=1)
    total = err.sum(axis=1) - err[np.arange(n), idx]
    return total / (C - 1)


def gated_score(te, tr):
    p999 = np.percentile(tr, 99.9, axis=0)
    gate = (te > p999[None, :]).any(axis=1)
    mean_sc = te.mean(axis=1)
    t1 = trim1_score(te)
    return np.where(gate, t1, mean_sc), gate.mean()


def metrics(name, test_sc, train_sc, lb, lm):
    d = bf_pa(test_sc, lb)['f1']
    roc = roc_auc_score(lb, test_sc)
    ap = average_precision_score(lb, test_sc)
    th = pot_threshold(train_sc, test_sc, lm)
    c = point2point(adjust_predicts(test_sc, lb, th), lb > 0.1)[0]
    a = point2point(test_sc > th, lb > 0.1)[0]
    print(f'  {name:10s} C={c:.4f}  D={d:.4f}  A={a:.4f}  ROC={roc:.4f}  '
          f'AP={ap:.4f}  POT_th={th:.4g}', flush=True)
    return dict(C=c, D=d, A=a, ROC=roc, AP=ap)


def main():
    summary = {}
    for ds, seeds in SEEDS.items():
        print(f'\n===== {ds} =====', flush=True)
        acc = {}
        for seed in seeds:
            z = np.load(f'{PF}/{ds}_s{seed}.npz')
            te, tr, lb = z['test_err'].astype(float), z['train_err'].astype(float), z['label'].astype(int)
            print(f'-- seed {seed} (test {te.shape}) --', flush=True)
            variants = {
                'base': (te.mean(axis=1), tr.mean(axis=1)),
                'trim1': (trim1_score(te), trim1_score(tr)),
            }
            g_te, gate_rate = gated_score(te, tr)
            g_tr, _ = gated_score(tr, tr)
            variants['gated'] = (g_te, g_tr)
            print(f'  [gate fires test: {gate_rate:.3f}]', flush=True)

            if ds == 'PSM':
                zr = np.load(f'{PF}/{ds}_s{seed}_revin.npz')
                revin_te = zr['test_err'].astype(float).mean(axis=1)
                revin_tr = np.load(f'{SC}/{ds}_s{seed}_revin.npz')['train_score'].astype(float)
                variants['revin'] = (revin_te, revin_tr)
                # fusion: z(revin) + 0.3*z(base)，各自 split 内 z 化
                def zfit(x):
                    return (x - x.mean()) / (x.std() + 1e-12)
                fus_te = zfit(revin_te) + 0.3 * zfit(te.mean(axis=1))
                # revin train 分数是 stride-1 平铺（8.4M），base train 是 step=64（132k）；
                # SPOT init 只需样本分布，按 64 抽稀对齐长度即可
                revin_tr_s = revin_tr[::64]
                m = min(len(revin_tr_s), len(tr))
                fus_tr = zfit(revin_tr_s[:m]) + 0.3 * zfit(tr[:m].mean(axis=1))
                variants['fusion'] = (fus_te, fus_tr)
                # zone 分解：base f3 err<=3 的区外点上比较 base vs revin 的排序
                zone = te[:, 3] > 3.0
                out = ~zone
                if 0 < lb[out].sum() < out.sum():
                    r_base_out = roc_auc_score(lb[out], te[out].mean(axis=1))
                    r_rev_out = roc_auc_score(lb[out], revin_te[out])
                    r_base_in = roc_auc_score(lb[zone], te[zone].mean(axis=1)) \
                        if 0 < lb[zone].sum() < zone.sum() else float('nan')
                    r_rev_in = roc_auc_score(lb[zone], revin_te[zone]) \
                        if 0 < lb[zone].sum() < zone.sum() else float('nan')
                    print(f'  [zone f3>3: n={zone.sum()}]  区外 ROC base={r_base_out:.4f} '
                          f'revin={r_rev_out:.4f} | 区内 ROC base={r_base_in:.4f} '
                          f'revin={r_rev_in:.4f}', flush=True)

            for name, (test_sc, train_sc) in variants.items():
                r = metrics(name, test_sc, train_sc, lb, LM[ds])
                acc.setdefault(name, []).append(r)

        print(f'-- {ds} seed-mean --', flush=True)
        for name, rows in acc.items():
            ms = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            summary[(ds, name)] = ms
            print(f'  {name:10s} C={ms["C"]:.4f}  D={ms["D"]:.4f}  A={ms["A"]:.4f}  '
                  f'ROC={ms["ROC"]:.4f}  AP={ms["AP"]:.4f}', flush=True)

    print('\n===== 汇总 =====', flush=True)
    for (ds, name), ms in summary.items():
        print(f'{ds:5s} {name:8s} C={ms["C"]:.4f} D={ms["D"]:.4f} A={ms["A"]:.4f} '
              f'ROC={ms["ROC"]:.4f} AP={ms["AP"]:.4f}')


if __name__ == '__main__':
    main()
