#!/usr/bin/env python3
"""统一 11 模型 × 4 数据集四张表——逐点口径版（每点恰好一次，全模型一致）：
  A  POT 阈值 + 裸 F1（无 PA）★主表        B  AUC-PR / AUC-ROC
  C  POT + point-adjust（文献口径）        D  Best-F1 + point-adjust（oracle 阈值）
与 assemble_unified.py 唯一差别：TSLib 三模型改用 vus_diag/tsl_pointwise/*.npz
（修复 PSM step=1 展开 / KANAD 抽稀的口径混杂），POT 阈值用 pot_results_pw.json。
ISTAD/classic 分数不变（本来就是逐点）。
"""
import importlib.util
import json
import statistics as st

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TSL = '/data/modeluse/TS/Time-Series-Library'
IST = '/data/modeluse/TS/ISTAD'
VUS = '/data/modeluse/TS/vus_diag'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf_tsl = _load(f'{TSL}/utils/tools.py', 'tools_tsl').bf_search_adaptive
bf_ist = _load(f'{IST}/utils/tools.py', 'tools_ist').bf_search_adaptive

IST_SEEDS = {'SMD': [48, 2021, 2022, 2025], 'PSM': [87, 90, 98, 2021],
             'EXA': [48, 89, 2021], 'SWAT': [89, 48, 2021]}
TSL_SEEDS = [48, 2021, 2022]
CLASSIC = ['DAGMM', 'GDN', 'LSTM_AD', 'MAD_GAN', 'MTAD_GAT', 'OmniAnomaly', 'TranAD']
DS_CLASSIC = {'SMD': 'SMD', 'PSM': 'PSM', 'SWAT': 'SWaT', 'EXA': 'Exathlon'}
DS_NAME = {'SMD': 'SMD', 'PSM': 'PSM', 'SWAT': 'SWAT', 'EXA': 'Exathlon'}
ORDER = ['ISTAD', 'TimesNet', 'DLinear', 'KANAD', 'DAGMM', 'GDN', 'LSTM_AD',
         'MAD_GAN', 'MTAD_GAT', 'OmniAnomaly', 'TranAD']
DS_ORDER = ['EXA', 'PSM', 'SMD', 'SWAT']


def point2point(predict, actual):
    TP = np.sum(predict * actual)
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    p = TP / (TP + FP + 1e-5)
    r = TP / (TP + FN + 1e-5)
    return 2 * p * r / (p + r + 1e-5)


def build_jobs():
    pot = {(r['model'], r['ds'], str(r['seed'])): r['threshold']
           for r in json.load(open(f'{VUS}/pot_results_pw.json'))}
    jobs = []
    for ds in DS_ORDER:
        for model in ['TimesNet', 'DLinear', 'KANAD']:
            for seed in TSL_SEEDS:
                jobs.append((model, ds, str(seed), 'tsl',
                             f'{VUS}/tsl_pointwise/{model}_{DS_NAME[ds]}_s{seed}.npz',
                             pot[(model, ds, str(seed))]))
        for seed in IST_SEEDS[ds]:
            jobs.append(('ISTAD', ds, str(seed), 'ist',
                         f'{VUS}/istad_scores/{ds}_s{seed}.npz', pot[('ISTAD', ds, str(seed))]))
    cl = {(r['model'], r['ds']): r for r in json.load(open(f'{VUS}/classic_metrics.json'))}
    for ds in DS_ORDER:
        for model in CLASSIC:
            jobs.append((model, ds, 'single', 'classic',
                         f'{VUS}/classic_scores/{model}_{DS_CLASSIC[ds]}.npz',
                         cl[(model, DS_CLASSIC[ds])]['th']))
    return jobs


def process_one(job):
    model, ds, seed, kind, path, th = job
    z = np.load(path)
    sc, lb = z['score'].astype(float), z['label'].astype(int)
    ap = average_precision_score(lb, sc)
    roc = roc_auc_score(lb, sc)
    raw = point2point((sc > th).astype(int), lb > 0.1)
    bfun = bf_ist if kind == 'ist' else bf_tsl
    bf = bfun(score=sc, label=lb, coarse_step_num=200, fine_step_num=500,
              verbose=False, use_adjustment=True)['f1']
    return model, ds, seed, dict(raw=float(raw), ap=float(ap), roc=float(roc), bf=float(bf))


def ms(x):
    return f'{st.mean(x):.4f}' + (f'±{st.stdev(x):.4f}' if len(x) > 1 else '')


def main():
    import multiprocessing as mp
    jobs = build_jobs()
    print(f'=== {len(jobs)} runs (pointwise), pool=10 ===', flush=True)
    agg = {}
    with mp.Pool(10) as pool:
        for model, ds, seed, m in pool.imap_unordered(process_one, jobs):
            a = agg.setdefault((model, ds), {'raw': [], 'ap': [], 'roc': [], 'bf': []})
            for k in a:
                a[k].append(m[k])
            print(f'{model:12s} {ds} s{seed}  raw={m["raw"]:.4f} AP={m["ap"]:.4f} '
                  f'ROC={m["roc"]:.4f} BF={m["bf"]:.4f}', flush=True)

    cl = {(r['model'], r['ds']): r for r in json.load(open(f'{VUS}/classic_metrics.json'))}
    inv = {v: k for k, v in DS_CLASSIC.items()}
    pot = json.load(open(f'{VUS}/pot_results_pw.json'))
    potpa = {}
    for r in pot:
        potpa.setdefault((r['model'], r['ds']), []).append(r['f1'])
    for (model, ds_t), r in cl.items():
        potpa.setdefault((model, inv[ds_t]), []).append(r['f1_pa'])

    out = {}
    for title, key in [('A: POT阈值+裸F1（主部署表，无PA）', 'raw'),
                       ('B1: AUC-PR', 'ap'), ('B2: AUC-ROC', 'roc'),
                       ('D: Best-F1+PA（oracle阈值）', 'bf')]:
        print(f'\n=== {title} ===')
        print('| Model | EXATHLON | PSM | SMD | SWAT |')
        print('|---|---|---|---|---|')
        tbl = {}
        for model in ORDER:
            cells = []
            for ds in DS_ORDER:
                v = agg[(model, ds)][key]
                cells.append(ms(v))
                tbl[ds] = ms(v)
            out.setdefault(model, {})[title] = tbl
            print(f'| {model} | ' + ' | '.join(cells) + ' |')
    print('\n=== C: POT+PA（文献口径）===')
    print('| Model | EXATHLON | PSM | SMD | SWAT |')
    print('|---|---|---|---|---|')
    for model in ORDER:
        cells = []
        for ds in DS_ORDER:
            cells.append(ms(potpa[(model, ds)]))
        print(f'| {model} | ' + ' | '.join(cells) + ' |')

    with open(f'{VUS}/unified_tables_pw.json', 'w') as f:
        json.dump({'agg': {f'{m}|{d}': v for (m, d), v in agg.items()},
                   'potpa': {f'{m}|{d}': v for (m, d), v in potpa.items()}}, f, indent=1)


if __name__ == '__main__':
    main()
