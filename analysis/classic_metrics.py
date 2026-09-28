#!/usr/bin/env python3
"""classic-7 统一指标 + 回归校验：
1) 回归：从 dump 分数重算 POT+PA，对拍 results/*/final_result.json（f1/threshold）
2) 新口径：POT 阈值 + 裸 F1（无 PA）、AP、ROC、Best-F1+PA（bf_search_adaptive，与 TSLib 表同款）
lm 表与 constants.py 完全一致（TranAD 走第二组）。
"""
import importlib.util
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TRI = '/data/modeluse/TS/TranAD_improve'
TSL = '/data/modeluse/TS/Time-Series-Library'
SC = '/data/modeluse/TS/vus_diag/classic_scores'

spec = importlib.util.spec_from_file_location('spot_mod', f'{TRI}/src/spot.py')
spot_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spot_mod)
SPOT = spot_mod.SPOT

spec = importlib.util.spec_from_file_location('tools_tsl', f'{TSL}/utils/tools.py')
tools_tsl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools_tsl)
bf_search_adaptive = tools_tsl.bf_search_adaptive

LM0 = {'SMD': (0.988, 1.00), 'SWaT': (0.9999, 1.2), 'PSM': (0.98, 0.9), 'Exathlon': (0.99, 1.0)}
LM1 = {'SMD': (0.997, 1.06), 'SWaT': (0.9999, 1.28), 'PSM': (0.98, 0.9), 'Exathlon': (0.99, 1.0)}
Q = 1e-5

MODELS = ['DAGMM', 'GDN', 'LSTM_AD', 'MAD_GAN', 'MTAD_GAT', 'OmniAnomaly', 'TranAD']
DATASETS = ['SMD', 'PSM', 'SWaT', 'Exathlon']


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


def process_one(job):
    model, ds = job
    path = f'{SC}/{model}_{ds}.npz'
    z = np.load(path)
    sc, lb, tr = z['score'].astype(float), z['label'].astype(int), z['train_score'].astype(float)
    stored = json.load(open(f'{TRI}/results/{model}_{ds}/final_result.json'))
    lm = LM1[ds] if 'TranAD' in model else LM0[ds]
    th = pot_threshold(tr, sc, lm)
    pred_pa = adjust_predicts(sc, lb, th)
    f1_pa, p_pa, r_pa = point2point(pred_pa, lb > 0.1)
    f1_raw, p_raw, r_raw = point2point((sc > th).astype(int), lb > 0.1)
    ap = average_precision_score(lb, sc)
    roc = roc_auc_score(lb, sc)
    bfr = bf_search_adaptive(score=sc, label=lb, coarse_step_num=200,
                             fine_step_num=500, verbose=False, use_adjustment=True)
    reg_f1 = abs(f1_pa - stored['f1']) < 1e-6
    reg_th = abs(th - stored['threshold']) < 1e-9
    row = dict(model=model, ds=ds, n=int(len(sc)),
               th=th, f1_pa=f1_pa, f1_raw=f1_raw, p_raw=p_raw, r_raw=r_raw,
               ap=ap, roc=roc, bf=bfr['f1'], bf_th=bfr['threshold'],
               stored_f1=stored['f1'], stored_th=stored['threshold'],
               reg_f1=bool(reg_f1), reg_th=bool(reg_th))
    tag = (f'{model:12s} {ds:9s} th={th:.6f} POT+PA={f1_pa:.4f}(存{stored["f1"]:.4f} '
           f'{"OK" if reg_f1 else "**DIFF**"}) raw={f1_raw:.4f} AP={ap:.4f} ROC={roc:.4f} '
           f'BF1={bfr["f1"]:.4f}')
    return row, tag


def main():
    import multiprocessing as mp
    jobs = [(m, d) for m in MODELS for d in DATASETS
            if os.path.exists(f'{SC}/{m}_{d}.npz')]
    missing = [(m, d) for m in MODELS for d in DATASETS
               if not os.path.exists(f'{SC}/{m}_{d}.npz')]
    for m, d in missing:
        print(f'MISSING {m} {d}', flush=True)
    print(f'=== {len(jobs)} runs, pool=8 ===', flush=True)
    rows = []
    with mp.Pool(8) as pool:
        for row, tag in pool.imap_unordered(process_one, jobs):
            rows.append(row)
            print(tag, flush=True)

    rows.sort(key=lambda r: (MODELS.index(r['model']), DATASETS.index(r['ds'])))
    with open('/data/modeluse/TS/vus_diag/classic_metrics.json', 'w') as f:
        json.dump(rows, f, indent=1)
    n_bad = sum(1 for r in rows if not r['reg_f1'])
    print(f'\nregression mismatches: {n_bad}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
