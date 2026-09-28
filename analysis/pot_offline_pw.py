#!/usr/bin/env python3
"""POT+adjust（逐点口径）：TSLib 行改用 vus_diag/tsl_pointwise/*.npz（每点恰好一次），
ISTAD 行不变（istad_scores）。协议与 pot_offline.py 完全相同（同 SPOT、q=1e-5、lm 表）。
输出 pot_results_pw.json。
"""
import json
import importlib.util
import multiprocessing as mp

import numpy as np

TRI = '/data/modeluse/TS/TranAD_improve'
PW = '/data/modeluse/TS/vus_diag/tsl_pointwise'
SC = '/data/modeluse/TS/vus_diag/istad_scores'

spec = importlib.util.spec_from_file_location('spot_mod', f'{TRI}/src/spot.py')
spot_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spot_mod)
SPOT = spot_mod.SPOT

LM = {'SMD': (0.988, 1.00), 'SWAT': (0.9999, 1.2), 'PSM': (0.98, 0.9), 'EXA': (0.99, 1.0)}
Q = 1e-5
DS_NAME = {'SMD': 'SMD', 'PSM': 'PSM', 'SWAT': 'SWAT', 'EXA': 'Exathlon'}


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
        z = np.load(f'{PW}/{model}_{DS_NAME[ds]}_s{seed}.npz')
    else:
        z = np.load(f'{SC}/{ds}_s{seed}.npz')
    tr, sc, lb = z['train_score'].astype(float), z['score'].astype(float), z['label'].astype(int)
    try:
        r = pot_eval(tr, sc, lb, LM[ds])
        ok = 'OK'
    except Exception as e:
        r = {'f1': float('nan'), 'precision': float('nan'), 'recall': float('nan'),
             'threshold': float('nan')}
        ok = f'FAIL {type(e).__name__}: {e}'
    tag = f'{model:9s} {ds:9s} s{seed:>5}  POT-F1={r["f1"]:.4f}  th={r.get("threshold", float("nan")):.4f}  {ok}'
    return (model, ds, seed, r, tag)


if __name__ == '__main__':
    print(f'=== POT(pointwise) on {len(JOBS)} runs ===', flush=True)
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

    with open('/data/modeluse/TS/vus_diag/pot_results_pw.json', 'w') as f:
        json.dump([{'model': m, 'ds': d, 'seed': s, **r} for m, d, s, r, _ in results], f, indent=1)
