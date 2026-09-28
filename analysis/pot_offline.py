#!/usr/bin/env python3
"""离线 POT+adjust：对已 dump 的全长 train/test 分数统一计算 POT F1。

协议与 TranAD_improve 表 1 严格一致：
  - SPOT (EVT) init 阈值来自训练集分数, level=lm[0], q=1e-5
  - pot_th = mean(thresholds) * lm[1]
  - adjust_predicts (OmniAnomaly 式 point-adjust) + point2point
  - lm 按数据集查表: SMD (0.988,1.00) / SWaT (0.9999,1.2) / PSM (0.98,0.9) / Exathlon (0.99,1)
"""
import json
import os
import sys
import importlib.util
import multiprocessing as mp

import numpy as np

TSL = '/data/modeluse/TS/Time-Series-Library'
IST = '/data/modeluse/TS/ISTAD'
TRI = '/data/modeluse/TS/TranAD_improve'
SC = '/data/modeluse/TS/vus_diag/istad_scores'

spec = importlib.util.spec_from_file_location('spot_mod', f'{TRI}/src/spot.py')
spot_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spot_mod)
SPOT = spot_mod.SPOT

LM = {'SMD': (0.988, 1.00), 'SWAT': (0.9999, 1.2), 'PSM': (0.98, 0.9), 'EXA': (0.99, 1.0)}
Q = 1e-5


def adjust_predicts(score, label, threshold):
    """OmniAnomaly 式 point-adjust（TranAD_improve/src/pot.py 同款）。"""
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
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall


def pot_eval(init_score, score, label, lm):
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
    pot_th = np.mean(ret['thresholds']) * lm[1]
    pred = adjust_predicts(score, label, pot_th)
    f1, p, r = point2point(pred, np.asarray(label) > 0.1)
    return {'f1': float(f1), 'precision': float(p), 'recall': float(r),
            'threshold': float(pot_th)}


def tslib_setting(ds_data, model, model_id, sl, seed):
    import re
    # setting 顺序 = task_{model_id}_{model}_{data}
    pats = [s for s in os.listdir(f'{TSL}/test_results')
            if re.fullmatch(rf'anomaly_detection_{model_id}_{model}_{ds_data}_ftM_sl{sl}_.*_ms{seed}_0', s)]
    assert len(pats) == 1, (ds_data, model, seed, pats)
    return pats[0]


TSL_CFG = {  # (ds, model) -> (model_id, seq_len, ds_data)；从实际目录核实
    ('EXA', 'TimesNet'): ('Exathlon', 96, 'EXATHLON'), ('EXA', 'DLinear'): ('Exathlon', 96, 'EXATHLON'),
    ('EXA', 'KANAD'): ('Exathlon', 96, 'EXATHLON'),
    ('PSM', 'TimesNet'): ('PSM', 100, 'PSM'), ('PSM', 'DLinear'): ('PSM', 100, 'PSM'),
    ('PSM', 'KANAD'): ('PSM', 64, 'PSM'),
    ('SMD', 'TimesNet'): ('SMD', 100, 'SMD'), ('SMD', 'DLinear'): ('SMD', 100, 'SMD'),
    ('SMD', 'KANAD'): ('SMD', 96, 'SMD'),
    ('SWAT', 'TimesNet'): ('SWAT', 100, 'SWAT'), ('SWAT', 'DLinear'): ('SWAT', 100, 'SWAT'),
    ('SWAT', 'KANAD'): ('SWAT', 80, 'SWAT'),
}

JOBS = []
for ds in ['EXA', 'PSM', 'SMD', 'SWAT']:
    for model in ['TimesNet', 'DLinear', 'KANAD']:
        for seed in [48, 2021, 2022]:
            JOBS.append(('tsl', model, ds, seed))
for ds in ['SMD', 'PSM', 'EXA', 'SWAT']:
    seeds = {'SMD': [48, 2021, 2022, 2025], 'PSM': [87, 90, 98, 2021],
             'EXA': [48, 89, 2021], 'SWAT': [89, 48, 2021]}[ds]
    for seed in seeds:
        JOBS.append(('ist', 'ISTAD', ds, seed))


def process_one(job):
    kind, model, ds, seed = job
    if kind == 'tsl':
        model_id, sl, ds_data = TSL_CFG[(ds, model)]
        setting = tslib_setting(ds_data, model, model_id, sl, seed)
        d = f'{TSL}/test_results/{setting}'
        tr = np.load(f'{d}/train_scores.npy')
        sc = np.load(f'{d}/test_scores.npy')
        lb = np.load(f'{d}/test_labels.npy')
    else:
        z = np.load(f'{SC}/{ds}_s{seed}.npz')
        tr, sc, lb = z['train_score'].astype(float), z['score'].astype(float), z['label'].astype(int)
    try:
        r = pot_eval(tr, sc, lb, LM[ds])
        ok = 'OK'
    except Exception as e:
        r = {'f1': float('nan'), 'precision': float('nan'), 'recall': float('nan')}
        ok = f'FAIL {type(e).__name__}: {e}'
    tag = f'{model:9s} {ds:9s} s{seed:>5}  POT-F1={r["f1"]:.4f}  P={r["precision"]:.4f} R={r["recall"]:.4f}  {ok}'
    return (model, ds, seed, r, tag)


if __name__ == '__main__':
    print(f'=== POT on {len(JOBS)} runs ===', flush=True)
    with mp.Pool(10) as pool:
        results = []
        for r in pool.imap_unordered(process_one, JOBS):
            results.append(r)
            print(r[-1], flush=True)

    import statistics as st
    print('\n=== POT aggregate mean±std ===', flush=True)
    agg = {}
    for model, ds, seed, r, _ in results:
        agg.setdefault((model, ds), []).append(r['f1'])
    for k in sorted(agg):
        v = agg[k]
        print(f'{k[0]:9s} {k[1]:9s} N={len(v)}  POT-F1 {st.mean(v):.4f}±{(st.stdev(v) if len(v) > 1 else 0):.4f}', flush=True)

    with open('/data/modeluse/TS/vus_diag/pot_results.json', 'w') as f:
        json.dump([{'model': m, 'ds': d, 'seed': s, **r} for m, d, s, r, _ in results], f, indent=1)
