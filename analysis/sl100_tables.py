#!/usr/bin/env python3
"""seq_len 试点判定（预注册 vus_diag/prereg_sl100.md）：
PSM  dual sl64→sl100（4 种子，dual min_z 融合，与 dual_tables.py 同协议）
SWAT gated sl96→sl100（3 种子，gated 聚合，与 swat_k25_tables.py 同协议）
同种子同管线配对，dump 逐特征误差 → 全协议 A/C/D/ROC/AP。

门槛（mean，ddof=0；两数据集独立判定）：
  PSM  主门槛（二选一即算超 SOTA）：C ≥ 0.9673（TimesNet 0.9672）或 D ≥ 0.9733（0.9732）
       护栏：ROC ≥ 0.7412、AP ≥ 0.4824
  SWAT 四条全过：D ≥ 0.8764 且 C ≥ 0.8075 且 AP ≥ 0.73 且 ROC ≥ 0.84
任一不过 → KEEP 现行 sl，seq_len 轴对该数据集关闭；禁止事后换打分变体。

用法: python sl100_tables.py
依赖 dump: istad_perfeat/PSM_s*_dual_sl100.npz、istad_perfeat/SWAT_s*_sl100.npz
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'

# 预注册门槛
PSM_GATES = {'C_or_D_C': 0.9673, 'C_or_D_D': 0.9733, 'ROC': 0.7412, 'AP': 0.4824}
SWAT_GATES = {'D': 0.8764, 'C': 0.8075, 'AP': 0.73, 'ROC': 0.84}
# 现行采纳行（v2.1，回归参照）
PSM_REF = {'A': 0.0996, 'C': 0.9564, 'D': 0.9696, 'ROC': 0.7462, 'AP': 0.4874}
SWAT_REF = {'A': 0.2583, 'C': 0.8125, 'D': 0.8689, 'ROC': 0.8413, 'AP': 0.7317}


def proto_full(sc_te, sc_tr, lb, ds):
    th = pot_threshold(sc_tr, sc_te, LM[ds])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    roc = roc_auc_score(lb, sc_te)
    ap = average_precision_score(lb, sc_te)
    return dict(A=a, C=c, D=d, ROC=roc, AP=ap)


def fmt(vals):
    return f'{np.mean(vals):.4f}±{np.std(vals):.4f}'


def psm_block():
    keys = ['A', 'C', 'D', 'ROC', 'AP']
    tabs = {k: {'sl100': [], 'sl64': []} for k in keys}
    for s in ['87', '90', '98', '2021']:
        # ---- sl100（试点）：dual min_z ----
        zd = np.load(f'{PF}/PSM_s{s}_dual_sl100.npz')
        lb = zd['label'].astype(int)
        g_te, gate = gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))
        g_tr, _ = gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))
        m_te = zd['test_err_revin'].astype(float).mean(axis=1)
        m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
        n = min(len(g_tr), len(m_tr))
        sc_te = np.minimum(z(g_te), z(m_te))
        sc_tr = np.minimum(z(g_tr[:n]), z(m_tr[:n]))
        r_n = proto_full(sc_te, sc_tr, lb, 'PSM')
        # ---- sl64（现行采纳，同管线回归对照）----
        zb = np.load(f'{PF}/PSM_s{s}_dual.npz')
        lb_b = zb['label'].astype(int)
        gb_te, _ = gated_score(zb['test_err'].astype(float), zb['train_err'].astype(float))
        gb_tr, _ = gated_score(zb['train_err'].astype(float), zb['train_err'].astype(float))
        mb_te = zb['test_err_revin'].astype(float).mean(axis=1)
        mb_tr = zb['train_err_revin'].astype(float).mean(axis=1)
        nb = min(len(gb_tr), len(mb_tr))
        sb_te = np.minimum(z(gb_te), z(mb_te))
        sb_tr = np.minimum(z(gb_tr[:nb]), z(mb_tr[:nb]))
        r_o = proto_full(sb_te, sb_tr, lb_b, 'PSM')
        for k in keys:
            tabs[k]['sl100'].append(r_n[k])
            tabs[k]['sl64'].append(r_o[k])
        print(f's{s:4s} gate={gate:.3f} [sl100] A={r_n["A"]:.4f} C={r_n["C"]:.4f} D={r_n["D"]:.4f} '
              f'ROC={r_n["ROC"]:.4f} AP={r_n["AP"]:.4f} | [sl64] C={r_o["C"]:.4f} D={r_o["D"]:.4f} '
              f'ROC={r_o["ROC"]:.4f} AP={r_o["AP"]:.4f}', flush=True)

    print('\n===== PSM dual min_z：sl100 vs sl64（同管线配对）4 种子 mean±std =====')
    for k in keys:
        print(f'{k:4s} sl100={fmt(tabs[k]["sl100"])}  sl64={fmt(tabs[k]["sl64"])}')

    print(f'\n[sl64 回归校验] 现行采纳行 PSM_REF={PSM_REF}')
    ok = True
    for k, ref in PSM_REF.items():
        m = np.mean(tabs[k]['sl64'])
        flag = 'OK' if abs(m - ref) < 0.003 else 'DRIFT!'
        ok &= flag == 'OK'
        print(f'  {k:4s} {m:.4f} vs ref {ref:.4f}  {flag}')

    print('\n===== PSM 预注册判定（sl100 4 种子 mean）=====')
    c_m = np.mean(tabs['C']['sl100'])
    d_m = np.mean(tabs['D']['sl100'])
    roc_m = np.mean(tabs['ROC']['sl100'])
    ap_m = np.mean(tabs['AP']['sl100'])
    main = c_m >= PSM_GATES['C_or_D_C'] or d_m >= PSM_GATES['C_or_D_D']
    roc_ok = roc_m >= PSM_GATES['ROC']
    ap_ok = ap_m >= PSM_GATES['AP']
    print(f'  主门槛(二选一): C {c_m:.4f} {"≥" if c_m >= PSM_GATES["C_or_D_C"] else "<"} 0.9673 | '
          f'D {d_m:.4f} {"≥" if d_m >= PSM_GATES["C_or_D_D"] else "<"} 0.9733 → '
          f'{"✓" if main else "✗"}')
    print(f'  护栏 ROC {roc_m:.4f} {"≥" if roc_ok else "<"} 0.7412  {"✓" if roc_ok else "✗"}')
    print(f'  护栏 AP  {ap_m:.4f} {"≥" if ap_ok else "<"} 0.4824  {"✓" if ap_ok else "✗"}')
    verdict = main and roc_ok and ap_ok
    print(f'\nPSM 判定: ' + ('ADOPT sl100（PSM 最终配置换 dual sl100，五表刷新）' if verdict
                           else 'KEEP sl64（PSM seq_len 轴关闭；负结果入素材）'))
    return verdict


def swat_block():
    keys = ['A', 'C', 'D', 'ROC', 'AP']
    tabs = {k: {'sl100': [], 'sl96': []} for k in keys}
    for s in ['48', '89', '2021']:
        # ---- sl100（试点）：gated ----
        zd = np.load(f'{PF}/SWAT_s{s}_sl100.npz')
        lb = zd['label'].astype(int)
        g_te, gate = gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))
        g_tr, _ = gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))
        r_n = proto_full(g_te, g_tr, lb, 'SWAT')
        # ---- sl96（现行采纳，同管线回归对照）----
        zb = np.load(f'{PF}/SWAT_s{s}.npz')
        lb_b = zb['label'].astype(int)
        gb_te, _ = gated_score(zb['test_err'].astype(float), zb['train_err'].astype(float))
        gb_tr, _ = gated_score(zb['train_err'].astype(float), zb['train_err'].astype(float))
        r_o = proto_full(gb_te, gb_tr, lb_b, 'SWAT')
        for k in keys:
            tabs[k]['sl100'].append(r_n[k])
            tabs[k]['sl96'].append(r_o[k])
        print(f's{s:4s} gate={gate:.3f} [sl100] A={r_n["A"]:.4f} C={r_n["C"]:.4f} D={r_n["D"]:.4f} '
              f'ROC={r_n["ROC"]:.4f} AP={r_n["AP"]:.4f} | [sl96] C={r_o["C"]:.4f} D={r_o["D"]:.4f} '
              f'ROC={r_o["ROC"]:.4f} AP={r_o["AP"]:.4f}', flush=True)

    print('\n===== SWAT gated：sl100 vs sl96（同管线配对）3 种子 mean±std =====')
    for k in keys:
        print(f'{k:4s} sl100={fmt(tabs[k]["sl100"])}  sl96={fmt(tabs[k]["sl96"])}')

    print(f'\n[sl96 回归校验] 现行采纳行 SWAT_REF={SWAT_REF}')
    for k, ref in SWAT_REF.items():
        m = np.mean(tabs[k]['sl96'])
        flag = 'OK' if abs(m - ref) < 0.003 else 'DRIFT!'
        print(f'  {k:4s} {m:.4f} vs ref {ref:.4f}  {flag}')

    print('\n===== SWAT 预注册判定（sl100 3 种子 mean，四条全过才采纳）=====')
    all_pass = True
    for k in ['D', 'C', 'AP', 'ROC']:
        m = np.mean(tabs[k]['sl100'])
        gate = SWAT_GATES[k]
        ok = m >= gate
        all_pass &= ok
        print(f'  {k:4s} {m:.4f} {"≥" if ok else "<"} {gate}  {"✓" if ok else "✗"}')
    print(f'\nSWAT 判定: ' + ('ADOPT sl100（SWAT 最终配置换 gated sl100，五表刷新）' if all_pass
                            else 'KEEP sl96（SWAT seq_len 轴关闭；负结果入素材）'))
    return all_pass


if __name__ == '__main__':
    print('################ PSM (dual min_z, 4 seeds) ################', flush=True)
    psm_block()
    print('\n################ SWAT (gated, 3 seeds) ################', flush=True)
    swat_block()
