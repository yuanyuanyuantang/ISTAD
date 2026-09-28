#!/usr/bin/env python3
"""PA-free metric diagnostic: AUC-PR / AUC-ROC / VUS-PR on dumped anomaly scores.
0) Regression check: recomputed Best-F1(+adjust) from dumped scores must match stored F1.
1) Metrics: sklearn AUC-PR/AUC-ROC (exact, point-wise, no point-adjustment, no threshold oracle);
   VUS-PR: range-based PR (existence*overlap*size weights) integrated over thresholds,
   averaged over buffers {0,25,50,100} (hand-rolled, eboniol/vus-style, for sensitivity only).
Scores来源: TSLib test_results/*/test_scores.npy (36 个 ms checkpoint dump) +
ISTAD csv (4 SMD + PSM s87 原有, 8 个 shadow __vusdump 重导出, SWAT s89 原始 run)。"""
import os, re, json, glob, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TSL = '/data/modeluse/TS/Time-Series-Library'
IST = '/data/modeluse/TS/ISTAD'

import importlib.util


def _load_tools(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf_tsl = _load_tools(f'{TSL}/utils/tools.py', 'tools_tsl').bf_search_adaptive
bf_ist = _load_tools(f'{IST}/utils/tools.py', 'tools_ist').bf_search_adaptive

SEEDS = [48, 2021, 2022]
MODELS = ['TimesNet', 'DLinear', 'KANAD']
DS_MAP = {'Exathlon': 'EXA', 'PSM': 'PSM', 'SMD': 'SMD', 'SWAT': 'SWAT'}


def tslib_settings(model, ds):
    pats = glob.glob(f'{TSL}/test_results/anomaly_detection_{ds}_{model}_*_ms*_0')
    return {re.search(r'_ms(\d+)_0$', s).group(1): os.path.basename(s) for s in pats}


def istad_source(ds, seed):
    """return (score, label, f1_stored) — 全长分数来自 dump_istad_full.py 导出的 npz"""
    SC = '/data/modeluse/TS/vus_diag/istad_scores'
    z = np.load(f'{SC}/{ds}_s{seed}.npz')
    stored = {
        ('SMD', 48):   f'{IST}/test_results/anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s48_0',
        ('SMD', 2021): f'{IST}/test_results/anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s2021_0',
        ('SMD', 2022): f'{IST}/test_results/anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s2022_0',
        ('SMD', 2025): f'{IST}/test_results/anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_leg_grid10_s2025_0',
        ('PSM', 87):   f'{IST}/test_results/anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s87_0',
        ('PSM', 90):   f'{IST}/test_results/anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s90_0',
        ('PSM', 98):   f'{IST}/test_results/anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s98_0',
        ('PSM', 2021): f'{IST}/test_results/anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_kan_tcn_bs64_PSM_g10_s2021_0',
        ('EXA', 48):   f'{IST}/test_results/anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_g10_s48_0',
        ('EXA', 89):   f'{IST}/test_results/anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_g10_s89_0',
        ('EXA', 2021): f'{IST}/test_results/anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_g10_s2021_0',
        ('SWAT', 48):  f'{IST}/test_results/anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_swats48_0',
        ('SWAT', 2021): f'{IST}/test_results/anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_swats2021_0',
        ('SWAT', 89):  f'{IST}/results_ours_adjust_best_f1/kanad/hgat_kan_tcn/anomaly_detection_SWAT_BestF1_Adaptive_2GPU_ISTAD_SWAT_ftM_sl96_bmhgat_kan_tcn_bs128_test_0',
    }[(ds, seed)]
    f1 = json.load(open(stored + '/bestf1_threshold_results.json'))['f1']
    return z['score'].astype(float), z['label'].astype(int), f1


# ---------- range-based PR (VUS-style), vectorized via coverage prefix sums ----------
def _ranges(b):
    """(starts, ends) of True-runs in boolean array."""
    x = np.r_[False, b, False].astype(np.int8)
    d = np.diff(x)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _cov_setup(s, e, beta):
    """merge buffered intervals, return (starts, ends, prefix_cover_len) for O(log n) overlap queries."""
    if len(s) == 0:
        return None
    bs, be = s - beta, e + beta
    # merge overlapping/adjacent
    order = np.argsort(bs)
    bs, be = bs[order], be[order]
    ms, me = [bs[0]], [be[0]]
    for a, b in zip(bs[1:], be[1:]):
        if a <= me[-1]:
            me[-1] = max(me[-1], b)
        else:
            ms.append(a); me.append(b)
    ms, me = np.asarray(ms, np.int64), np.asarray(me, np.int64)
    lens = me - ms
    prefix = np.r_[0, np.cumsum(lens)]
    return ms, me, prefix


def _overlap(setup, a, b):
    """total covered length of [a,b) by merged intervals, vectorized."""
    ms, me, prefix = setup
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    # 负索引回绕 bug 修复：查询点在首个区间之前时 searchsorted-1 == -1，
    # 必须 clamp 到 0 并显式置 0（否则 prefix[-1] 取到全长 → overlap 负值/爆炸）
    i = np.maximum(np.searchsorted(ms, a, side='right') - 1, 0)
    g_a = prefix[i] + np.clip(a - ms[i], 0, None)
    g_a = np.where(a < ms[i], 0.0, g_a)
    g_a = np.where(a >= me[i], prefix[i + 1], g_a)
    j = np.maximum(np.searchsorted(ms, b, side='right') - 1, 0)
    g_b = prefix[j] + np.clip(b - ms[j], 0, None)
    g_b = np.where(b < ms[j], 0.0, g_b)
    g_b = np.where(b >= me[j], prefix[j + 1], g_b)
    return g_b - g_a


BUFFERS = [0, 25, 50, 100]


def vus_pr(score, gt, n_thr=120):
    gt_b = gt.astype(bool)
    gs, ge = _ranges(gt_b)
    if len(gs) == 0:
        return np.nan
    gt_setup = {beta: _cov_setup(gs, ge, beta) for beta in BUFFERS}
    thr = np.unique(np.quantile(score, np.linspace(0.0001, 0.9999, n_thr)))
    auc = {b: 0.0 for b in BUFFERS}
    prev_r = {b: 0.0 for b in BUFFERS}
    for t in thr:
        ps, pe = _ranges(score >= t)
        if len(ps) == 0:
            continue
        glen = (ge - gs).astype(float)
        for b in BUFFERS:
            p_setup = _cov_setup(ps, pe, b)
            rec = np.minimum(_overlap(p_setup, gs, ge) / glen, 1.0).mean()
            prec = (_overlap(gt_setup[b], ps, pe) / ((pe - ps) + 2 * b)).mean()
            auc[b] += max(0.0, rec - prev_r[b]) * prec
            prev_r[b] = rec
    for b in BUFFERS:  # endpoint: predict everything
        auc[b] += max(0.0, 1.0 - prev_r[b]) * (gt_b.mean())
    return float(np.mean([auc[b] for b in BUFFERS]))


# ---------- main (process pool: bf search + metrics per run in parallel) ----------
import multiprocessing as mp

JOBS = []
for ds_tsl, ds in DS_MAP.items():
    for model in MODELS:
        for seed, setting in sorted(tslib_settings(model, ds_tsl).items(), key=lambda x: int(x[0])):
            JOBS.append(('tsl', model, ds, seed, setting))
for ds in ['SMD', 'PSM', 'EXA', 'SWAT']:
    seeds = {'SMD': [48, 2021, 2022, 2025], 'PSM': [87, 90, 98, 2021],
             'EXA': [48, 89, 2021], 'SWAT': [89, 48, 2021]}[ds]
    for seed in seeds:
        JOBS.append(('ist', 'ISTAD', ds, seed, (ds, seed)))


FAST = len(sys.argv) > 1 and sys.argv[1] == '--fast'  # 基线 bf 已精确验证过，跳过省时


def process_one(job):
    kind, model, ds, seed, ref = job
    if kind == 'tsl':
        sc = np.load(f'{TSL}/test_results/{ref}/test_scores.npy')
        lb = np.load(f'{TSL}/test_results/{ref}/test_labels.npy')
        txt = open(f'{TSL}/test_results/{ref}/result_anomaly_detection.txt').read()
        m = re.findall(r'Best F1 Threshold Method:\s*\nThreshold: [\d.]+\s*\nF1: ([\d.]+)', txt)
        stored, tol = float(m[-1]), 5e-4
        if FAST:
            res = {'f1': float('nan')}
        else:
            res = bf_tsl(score=sc, label=lb, coarse_step_num=200,
                         fine_step_num=500, verbose=False, use_adjustment=True)
    else:
        sc, lb, stored = istad_source(ds, seed)
        tol = 5e-3
        res = bf_ist(score=sc, label=lb, coarse_step_num=200,
                     fine_step_num=500, verbose=False, use_adjustment=True)
    ok = True if FAST else abs(res['f1'] - stored) < tol
    ap = average_precision_score(lb, sc)
    ra = roc_auc_score(lb, sc)
    vp = vus_pr(sc, lb)
    tag = f'{model:9s} {ds:9s} s{seed:>5}'
    return (model, ds, f's{seed}', stored, res['f1'], ok, ap, ra, vp,
            f"{tag}  F1 {stored:.4f}->{res['f1']:.4f} {'OK' if ok else '**MISMATCH**'} AP={ap:.4f} ROC={ra:.4f} VUS-PR={vp:.4f}")


if __name__ == '__main__':
    print(f'=== {len(JOBS)} runs, process pool ===', flush=True)
    with mp.Pool(10) as pool:
        results = []
        for r in pool.imap_unordered(process_one, JOBS):
            results.append(r)
            print(r[-1], flush=True)

    print('\n=== aggregate mean±std (F1-stored | AUC-PR | AUC-ROC | VUS-PR) ===', flush=True)
    import statistics as st
    agg = {}
    for r in results:
        agg.setdefault((r[0], r[1]), []).append((r[3], r[6], r[7], r[8]))
    for k in sorted(agg):
        v = agg[k]
        def ms(i):
            x = [e[i] for e in v]
            return f'{st.mean(x):.4f}±{(st.stdev(x) if len(x) > 1 else 0):.4f}'
        print(f'{k[0]:9s} {k[1]:9s} N={len(v)}  F1 {ms(0)}  AUC-PR {ms(1)}  AUC-ROC {ms(2)}  VUS-PR {ms(3)}', flush=True)

    mism = [r for r in results if not r[5]]
    print(f'\nmismatches: {len(mism)}', flush=True)
    for r in mism:
        print('  MISMATCH:', r[-1], flush=True)
