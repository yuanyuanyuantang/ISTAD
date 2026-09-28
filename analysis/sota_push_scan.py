#!/usr/bin/env python3
"""SOTA 收尾攻坚（零训练，2026-09-03）：三个定稿表缺口的离线扫描。

目标（用户 09-03 指定）：
  EXA  C(POT+PA) : 0.9434 -> >= 0.9578 (KANAD)      路线: 聚合家族扫描 + POT level 曲线
  SWAT D(bf+PA)  : 0.8689 -> >= 0.8722 (KANAD 健康参照; DLinear 0.8764 AP=0.163 疑展品)
  SMD  D(bf+PA)  : 0.8480 -> >= 0.8515 (TimesNet)   路线: dual min_z 跨种子委员会 + 家族

判定纪律（诚实性）：
  - 回归校验：gated/min_z/dual_base 必须先复现定稿行，再信任何新数字。
  - 本扫描为事后探索（多家族多比较），任何采纳 = 事后采纳，须如实标注
    （先例：prereg_smd_minz.md）；不作预测性预注册宣称。
  - 每数据集独立护栏（保护既有第一/打平带），全过 target+护栏才算候选。

用法: python sota_push_scan.py EXA|SWAT|SMD   （可多个，顺序执行）
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import (LM, adjust_predicts, gated_score, point2point,
                          pot_threshold, trim1_score)
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
VUS = '/data/modeluse/TS/vus_diag'

FAMILIES = ['gated', 'base', 'trim1', 'trim2', 'median', 'rawtop3',
            'zmean', 'zl2', 'ztop1', 'ztop3', 'ztop5', 'max']

# 定稿参照行（论文素材.md §3.0g v2.1 最终行 / §3.0g(5) dual_base）
REF = {
    'EXA': dict(A=0.2410, C=0.9434, D=0.9606, ROC=0.8744, AP=0.6252),
    'SWAT': dict(A=0.2583, C=0.8125, D=0.8689, ROC=0.8413, AP=0.7317),
    'SMD': dict(A=0.0863, C=0.8194, D=0.8480, ROC=0.7481, AP=0.1620),
}
# 目标与护栏（每数据集独立）
TARGET = {
    'EXA': dict(metric='C', th=0.9578,
                guard=dict(D=0.9606 - 0.002, ROC=0.8744 - 0.005,
                           AP=0.6252 - 0.005, A=0.2410 - 0.02)),
    'SWAT': dict(metric='D', th=0.8722,
                 guard=dict(C=0.8108, ROC=0.8413 - 0.005,
                            AP=0.7317 - 0.005, A=0.20)),
    'SMD': dict(metric='D', th=0.8515,
                guard=dict(C=0.7826, ROC=0.7481 - 0.005,
                           AP=0.1620 - 0.005, A=0.05)),
}


def topk_mean(mat, k):
    return np.sort(mat, axis=1)[:, -k:].mean(axis=1)


def trimk(mat, k):
    n, C = mat.shape
    order = np.argsort(mat, axis=1)
    mask = np.ones((n, C), bool)
    rows = np.arange(n)[:, None]
    mask[rows, order[:, -k:]] = False
    return mat[mask].reshape(n, C - k).mean(axis=1)


def fam_score(name, te, tr):
    """返回 (test_score, train_score)。z 族用 train 统计拟合（无泄漏），截负。"""
    if name == 'gated':
        return gated_score(te, tr)[0], gated_score(tr, tr)[0]
    if name == 'base':
        return te.mean(axis=1), tr.mean(axis=1)
    if name == 'max':
        return te.max(axis=1), tr.max(axis=1)
    if name == 'median':
        return np.median(te, axis=1), np.median(tr, axis=1)
    if name in ('trim1', 'trim2'):
        k = int(name[-1])
        return trimk(te, k), trimk(tr, k)
    if name == 'rawtop3':
        return topk_mean(te, 3), topk_mean(tr, 3)
    mu, sd = tr.mean(axis=0), tr.std(axis=0) + 1e-12
    zt = np.clip((te - mu) / sd, 0, None)
    zr = np.clip((tr - mu) / sd, 0, None)
    if name == 'zmean':
        return zt.mean(axis=1), zr.mean(axis=1)
    if name == 'zl2':
        return np.sqrt((zt ** 2).sum(axis=1)), np.sqrt((zr ** 2).sum(axis=1))
    if name.startswith('ztop'):
        k = int(name[-1])
        return topk_mean(zt, k), topk_mean(zr, k)
    raise ValueError(name)


def segments_of(label):
    label = (np.asarray(label).reshape(-1) > 0.1).astype(int)
    d = np.diff(np.concatenate([[0], label, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends


def adjust_predicts_fast(score, label, threshold, segs):
    """adjust_predicts 的向量化等价实现（段内任一点命中 -> 整段 True）。"""
    starts, ends = segs
    segmax = np.array([score[s:e].max() for s, e in zip(starts, ends)])
    hit = segmax > threshold
    seg_id = np.full(len(label), -1, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg_id[s:e] = i
    pred = score > threshold
    pred[seg_id >= 0] |= hit[seg_id[seg_id >= 0]]
    return pred


def evaluate(ds, sc_te, sc_tr, lb, segs, lm=None, tag=''):
    lm = lm or LM[ds]
    th = pot_threshold(sc_tr, sc_te, lm)
    a = point2point(sc_te > th, lb > 0.1)[0]
    pred = adjust_predicts_fast(sc_te, lb, th, segs)
    c = point2point(pred, lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    return dict(A=a, C=c, D=d, ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te), th=th)


def fmt(vals):
    return f'{np.mean(vals):.4f}' + (f'±{np.std(vals):.4f}' if len(vals) > 1 else '')


def report_family(ds, name, rows, segs, lm=None):
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ('A', 'C', 'D', 'ROC', 'AP')}
    t = TARGET[ds]
    m = t['metric']
    ok_target = mean[m] >= t['th']
    ok_guard = all(mean[k] >= v for k, v in t['guard'].items())
    verdict = ('*** CANDIDATE ***' if (ok_target and ok_guard) else
               f'target{"✓" if ok_target else "✗"}/guard{"" if ok_guard else "✗ " + str([k for k, v in t["guard"].items() if mean[k] < v])}')
    print(f'  {name:9s} A={fmt([r["A"] for r in rows])} C={fmt([r["C"] for r in rows])} '
          f'D={fmt([r["D"] for r in rows])} ROC={fmt([r["ROC"] for r in rows])} '
          f'AP={fmt([r["AP"] for r in rows])}  {verdict}', flush=True)
    return mean, ok_target and ok_guard


# ---------------------------------------------------------------- EXA
def run_exa():
    ds = 'EXA'
    seeds = ['48', '89', '2021']
    segs_by_seed, data = {}, {}
    for s in seeds:
        z0 = np.load(f'{PF}/{ds}_s{s}.npz')
        data[s] = (z0['test_err'].astype(float), z0['train_err'].astype(float),
                   z0['label'].astype(int))
        segs_by_seed[s] = segments_of(data[s][2])
    print('===== 回归校验：gated（v2.1 定稿 EXA 行）=====', flush=True)
    ref_rows = []
    for s in seeds:
        te, tr, lb = data[s]
        g_te, _ = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        ref_rows.append(evaluate(ds, g_te, g_tr, lb, segs_by_seed[s]))
    report_family(ds, 'gated', ref_rows, segs_by_seed[s])
    for k, v in REF[ds].items():
        got = float(np.mean([r[k] for r in ref_rows]))
        flag = 'OK' if abs(got - v) < 0.003 else f'MISMATCH (Δ={got - v:+.4f})'
        print(f'    ref {k}={v:.4f} got {got:.4f} [{flag}]', flush=True)

    print('\n===== EXA 聚合家族扫描（V0 lm_table）=====', flush=True)
    cands = []
    for name in FAMILIES:
        if name == 'gated':
            continue
        rows = []
        for s in seeds:
            te, tr, lb = data[s]
            sc_te, sc_tr = fam_score(name, te, tr)
            rows.append(evaluate(ds, sc_te, sc_tr, lb, segs_by_seed[s]))
        _, ok = report_family(ds, name, rows, segs_by_seed[s])
        if ok:
            cands.append(name)

    print('\n===== EXA POT level 曲线（gated 分数，mult=1.0）=====', flush=True)
    for lv in (0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99):
        rows = []
        for s in seeds:
            te, tr, lb = data[s]
            g_te, _ = gated_score(te, tr)
            g_tr, _ = gated_score(tr, tr)
            rows.append(evaluate(ds, g_te, g_tr, lb, segs_by_seed[s], lm=(lv, 1.0)))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ('A', 'C', 'D', 'ROC', 'AP')}
        print(f'  level={lv:.3f} A={mean["A"]:.4f} C={mean["C"]:.4f} '
              f'(D/ROC/AP 与 level 无关: {mean["D"]:.4f}/{mean["ROC"]:.4f}/{mean["AP"]:.4f})', flush=True)

    print(f'\n[EXA] CANDIDATE 家族: {cands or "无"}', flush=True)


# ---------------------------------------------------------------- SWAT
def run_swat():
    ds = 'SWAT'
    seeds = ['48', '89', '2021']
    print('===== 回归校验 1：base npz gated（v2.1 定稿 SWAT 行）=====', flush=True)
    data, segs_by_seed = {}, {}
    for s in seeds:
        z0 = np.load(f'{PF}/{ds}_s{s}.npz')
        data[s] = (z0['test_err'].astype(float), z0['train_err'].astype(float),
                   z0['label'].astype(int))
        segs_by_seed[s] = segments_of(data[s][2])
    rows = []
    for s in seeds:
        te, tr, lb = data[s]
        g_te, _ = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        rows.append(evaluate(ds, g_te, g_tr, lb, segs_by_seed[s]))
    report_family(ds, 'gated', rows, segs_by_seed[s])
    for k, v in REF[ds].items():
        got = float(np.mean([r[k] for r in rows]))
        flag = 'OK' if abs(got - v) < 0.003 else f'MISMATCH (Δ={got - v:+.4f})'
        print(f'    ref {k}={v:.4f} got {got:.4f} [{flag}]', flush=True)

    print('\n===== 回归校验 2：dual npz dual_base（§3.0g(4) 侧观察 0.8715±0.0072）=====', flush=True)
    ddata = {}
    for s in seeds:
        zd = np.load(f'{PF}/{ds}_s{s}_dual.npz')
        ddata[s] = (zd['test_err'].astype(float), zd['train_err'].astype(float),
                    zd['label'].astype(int))
    rows = []
    for s in seeds:
        te, tr, lb = ddata[s]
        g_te, _ = gated_score(te, tr)
        g_tr, _ = gated_score(tr, tr)
        rows.append(evaluate(ds, g_te, g_tr, lb, segs_by_seed[s]))
    report_family(ds, 'dual_base', rows, segs_by_seed[s])

    print('\n===== SWAT 聚合家族扫描（base npz，V0）=====', flush=True)
    cands = []
    for name in FAMILIES:
        if name == 'gated':
            continue
        rows = []
        for s in seeds:
            te, tr, lb = data[s]
            sc_te, sc_tr = fam_score(name, te, tr)
            rows.append(evaluate(ds, sc_te, sc_tr, lb, segs_by_seed[s]))
        _, ok = report_family(ds, name, rows, segs_by_seed[s])
        if ok:
            cands.append(('base', name))

    print('\n===== SWAT 聚合家族扫描（dual npz base 视角，V0）=====', flush=True)
    for name in FAMILIES:
        rows = []
        for s in seeds:
            te, tr, lb = ddata[s]
            sc_te, sc_tr = fam_score(name, te, tr)
            rows.append(evaluate(ds, sc_te, sc_tr, lb, segs_by_seed[s]))
        _, ok = report_family(ds, 'dual:' + name, rows, segs_by_seed[s])
        if ok:
            cands.append(('dual', name))

    print('\n===== SWAT D-vs-AP 展品表（11 模型同口径重算）=====', flush=True)
    exhibit_table(ds)
    print(f'\n[SWAT] CANDIDATE: {cands or "无"}', flush=True)


def exhibit_table(ds):
    """每模型 D(bf+PA) vs AP/ROC —— 识别 PA 展品。"""
    import re
    TSL = '/data/modeluse/TS/Time-Series-Library'
    TSL_CFG = {'TimesNet': ('SWAT', 100, 'SWAT'), 'DLinear': ('SWAT', 100, 'SWAT'),
               'KANAD': ('SWAT', 80, 'SWAT')}
    DS_CLASSIC = {'SWAT': 'SWaT'}
    CLASSIC = ['DAGMM', 'GDN', 'LSTM_AD', 'MAD_GAN', 'MTAD_GAT', 'OmniAnomaly', 'TranAD']
    rows = []
    seeds = ['48', '89', '2021']
    for s in seeds:
        z0 = np.load(f'{PF}/{ds}_s{s}.npz')
        te, tr, lb = (z0['test_err'].astype(float), z0['train_err'].astype(float),
                      z0['label'].astype(int))
        sc_te, _ = gated_score(te, tr)
        rows.append(('ISTAD(gated,v2.1)', s, bf_pa(sc_te, lb)['f1'],
                     average_precision_score(lb, sc_te), roc_auc_score(lb, sc_te)))
    for model, (ds_data, sl, mid) in TSL_CFG.items():
        for ms in (48, 2021, 2022):
            pats = [p for p in os.listdir(f'{TSL}/test_results')
                    if re.fullmatch(rf'anomaly_detection_{ds_data}_{model}_{mid}_ftM_sl{sl}_.*_ms{ms}_0', p)]
            if not pats:
                continue
            d = f'{TSL}/test_results/{pats[0]}'
            sc = np.load(f'{d}/test_scores.npy').astype(float).reshape(-1)
            lb = np.load(f'{d}/test_labels.npy').astype(int).reshape(-1)
            n = min(len(sc), len(lb))
            rows.append((model, ms, bf_pa(sc[:n], lb[:n])['f1'],
                         average_precision_score(lb[:n], sc[:n]),
                         roc_auc_score(lb[:n], sc[:n])))
    for model in CLASSIC:
        zd = np.load(f'{VUS}/classic_scores/{model}_{DS_CLASSIC[ds]}.npz')
        sc, lb = zd['score'].astype(float), zd['label'].astype(int)
        rows.append((model + '*', 's', bf_pa(sc, lb)['f1'],
                     average_precision_score(lb, sc), roc_auc_score(lb, sc)))
    print(f'  {"model":18s} {"seed":>5s} {"D(bf+PA)":>9s} {"AP":>7s} {"ROC":>7s}  note')
    for m, s, d, ap, roc in sorted(rows, key=lambda r: -r[2]):
        note = 'ARTIFACT: AP<0.4' if ap < 0.4 else ('anti-corr ROC<0.5' if roc < 0.5 else '')
        print(f'  {m:18s} {str(s):>5s} {d:9.4f} {ap:7.4f} {roc:7.4f}  {note}', flush=True)


# ---------------------------------------------------------------- SMD
def smd_minz(seed):
    zd = np.load(f'{PF}/SMD_s{seed}_dual.npz')
    te, tr = zd['test_err'].astype(float), zd['train_err'].astype(float)
    g_te, _ = gated_score(te, tr)
    g_tr, _ = gated_score(tr, tr)
    m_te = zd['test_err_revin'].astype(float).mean(axis=1)
    m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
    n = min(len(g_tr), len(m_tr))
    sc = np.minimum(z(g_te), z(m_te))
    sc_tr = np.minimum(z(g_tr[:n]), z(m_tr[:n]))
    return sc, sc_tr, zd['label'].astype(int)


def run_smd():
    ds = 'SMD'
    seeds = ['48', '2021', '2022', '2025']
    print('===== 回归校验：min_z（v2.1 定稿 SMD 行 0.8480±0.0063 等）=====', flush=True)
    minz = {}
    for s in seeds:
        minz[s] = smd_minz(s)
    segs = segments_of(minz[seeds[0]][2])
    rows = []
    for s in seeds:
        sc, sc_tr, lb = minz[s]
        rows.append(evaluate(ds, sc, sc_tr, lb, segs))
    report_family(ds, 'min_z', rows, segs)
    for k, v in REF[ds].items():
        got = float(np.mean([r[k] for r in rows]))
        flag = 'OK' if abs(got - v) < 0.003 else f'MISMATCH (Δ={got - v:+.4f})'
        print(f'    ref {k}={v:.4f} got {got:.4f} [{flag}]', flush=True)

    print('\n===== SMD 跨种子委员会（min_z 分数，2/3/4 模型）=====', flush=True)
    for k in (2, 3, 4):
        ss = seeds[:k]
        tes = [z(minz[s][0]) for s in ss]
        trs = [z(minz[s][1]) for s in ss]
        n = min(len(t) for t in tes)
        sc_te = np.mean([t[:n] for t in tes], axis=0)
        m = min(len(t) for t in trs)
        sc_tr = np.mean([t[:m] for t in trs], axis=0)
        lb = minz[ss[0]][2][:n]
        assert all((minz[s][2][:n] == lb).all() for s in ss)
        rows = [evaluate(ds, sc_te, sc_tr, lb, segments_of(lb))]
        report_family(ds, f'committee[{k}]', rows, segs)

    print('\n===== SMD 委员会（gated 分数，复算 §3.0g(2) 0.7907/0.8577）=====', flush=True)
    g = {}
    for s in seeds:
        zd = np.load(f'{PF}/SMD_s{s}_dual.npz')
        g[s] = (gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))[0],
                gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))[0],
                zd['label'].astype(int))
    for k in (2, 3, 4):
        ss = seeds[:k]
        tes = [z(g[s][0]) for s in ss]
        trs = [z(g[s][1]) for s in ss]
        n = min(len(t) for t in tes)
        sc_te = np.mean([t[:n] for t in tes], axis=0)
        m = min(len(t) for t in trs)
        sc_tr = np.mean([t[:m] for t in trs], axis=0)
        lb = g[ss[0]][2][:n]
        rows = [evaluate(ds, sc_te, sc_tr, lb, segments_of(lb))]
        report_family(ds, f'g-committee[{k}]', rows, segs)

    print('\n===== SMD min_z 融合家族（dual npz，V0）=====', flush=True)
    for name in ['min_z', 'mean_z', 'softmin', 'wmean03', 'revin_only', 'gated_only']:
        rows = []
        for s in seeds:
            zd = np.load(f'{PF}/SMD_s{s}_dual.npz')
            te, tr = zd['test_err'].astype(float), zd['train_err'].astype(float)
            g_te, _ = gated_score(te, tr)
            g_tr, _ = gated_score(tr, tr)
            m_te = zd['test_err_revin'].astype(float).mean(axis=1)
            m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
            n = min(len(g_tr), len(m_tr))
            if name == 'min_z':
                sc_te, sc_tr = np.minimum(z(g_te), z(m_te)), np.minimum(z(g_tr[:n]), z(m_tr[:n]))
            elif name == 'mean_z':
                sc_te, sc_tr = z(g_te) + z(m_te), z(g_tr[:n]) + z(m_tr[:n])
            elif name == 'softmin':
                sc_te = -np.logaddexp(-5 * z(g_te), -5 * z(m_te)) / 5
                sc_tr = -np.logaddexp(-5 * z(g_tr[:n]), -5 * z(m_tr[:n])) / 5
            elif name == 'wmean03':
                sc_te, sc_tr = z(m_te) + 0.3 * z(g_te), z(m_tr[:n]) + 0.3 * z(g_tr[:n])
            elif name == 'revin_only':
                sc_te, sc_tr = m_te, m_tr
            else:
                sc_te, sc_tr = g_te, g_tr
            lb = zd['label'].astype(int)
            rows.append(evaluate(ds, sc_te, sc_tr, lb, segs))
        report_family(ds, name, rows, segs)


if __name__ == '__main__':
    for ds in (sys.argv[1:] or ['EXA']):
        {'EXA': run_exa, 'SWAT': run_swat, 'SMD': run_smd}[ds]()
    print('\n[done]', flush=True)
