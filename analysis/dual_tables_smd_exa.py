#!/usr/bin/env python3
"""SMD/EXA 双头扩展战役判定表：dual_base = dual 联合模型 base 视角 + gated 聚合。
协议与 dual_tables.py / dual_tables_swat.py 完全一致
（gated_score、z 按各自 split、POT 逐数据集 LM 表、bf_pa 200/500+adjust）。

预注册判据见 prereg_smd_exa.md（训练启动前落盘），每数据集独立，四条全过才采纳：
  D  >= gated D  + 0.002
  ROC >= gated ROC - 0.005
  AP  >= gated AP  - 0.005
  A   >= gated A   - 0.02
附带记录（不参与判定）：revin 第二视角 ROC/AP、min_z 变体全指标。

用法: conda run -n tslib python dual_tables_smd_exa.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from agg_simulate import bf_pa
from final_matrix import adjust_predicts, point2point, pot_threshold, gated_score, LM
from minz_tables import z

PF = '/data/modeluse/TS/vus_diag/istad_perfeat'
CFG = {
    'SMD': dict(seeds=['48', '2021', '2022', '2025'],
                ref=dict(A=0.2208, AP=0.1750, ROC=0.7700, C=0.7331, D=0.8361)),
    'EXA': dict(seeds=['48', '89', '2021'],
                ref=dict(A=0.2410, AP=0.6252, ROC=0.8744, C=0.9434, D=0.9606)),
}
EPS = dict(D=0.002, ROC=-0.005, AP=-0.005, A=-0.02)   # 判据容差（预注册）


def proto(ds, sc_te, sc_tr, lb):
    th = pot_threshold(sc_tr, sc_te, LM[ds])
    a = point2point(sc_te > th, lb > 0.1)[0]
    c = point2point(adjust_predicts(sc_te, lb, th), lb > 0.1)[0]
    d = bf_pa(sc_te, lb)['f1']
    return dict(A=a, C=c, D=d, ROC=roc_auc_score(lb, sc_te),
                AP=average_precision_score(lb, sc_te))


def fmt(vals):
    return f'{np.mean(vals):.4f}±{np.std(vals):.4f}'


def main():
    for DS in ['SMD', 'EXA']:
        cfg = CFG[DS]
        ref = cfg['ref']
        keys = ['A', 'C', 'D', 'ROC', 'AP']
        tabs = {k: {'dual_base': [], 'min_z': []} for k in keys}
        revin_rocs, revin_aps = [], []
        for s in cfg['seeds']:
            zd = np.load(f'{PF}/{DS}_s{s}_dual.npz')
            lb = zd['label'].astype(int)
            g_te, gate = gated_score(zd['test_err'].astype(float), zd['train_err'].astype(float))
            g_tr, _ = gated_score(zd['train_err'].astype(float), zd['train_err'].astype(float))
            m_te = zd['test_err_revin'].astype(float).mean(axis=1)
            m_tr = zd['train_err_revin'].astype(float).mean(axis=1)
            n = min(len(g_tr), len(m_tr))
            sc_mz = np.minimum(z(g_te), z(m_te))
            sc_mz_tr = np.minimum(z(g_tr[:n]), z(m_tr[:n]))
            r_b = proto(DS, g_te, g_tr, lb)
            r_m = proto(DS, sc_mz, sc_mz_tr, lb)
            revin_rocs.append(roc_auc_score(lb, m_te))
            revin_aps.append(average_precision_score(lb, m_te))
            for k in keys:
                tabs[k]['dual_base'].append(r_b[k])
                tabs[k]['min_z'].append(r_m[k])
            print(f'{DS} s{s:4s} gate={gate:.3f} | dual_base: '
                  f'A={r_b["A"]:.4f} C={r_b["C"]:.4f} D={r_b["D"]:.4f} '
                  f'ROC={r_b["ROC"]:.4f} AP={r_b["AP"]:.4f} | '
                  f'min_z: D={r_m["D"]:.4f} ROC={r_m["ROC"]:.4f} AP={r_m["AP"]:.4f}',
                  flush=True)

        print(f'\n===== {DS} dual_base {len(cfg["seeds"])} 种子 mean±std =====')
        mean = {}
        for k in keys:
            t = tabs[k]['dual_base']
            mean[k] = float(np.mean(t))
            print(f'{k:4s} dual_base={fmt(t)}  min_z={fmt(tabs[k]["min_z"])}  '
                  f'[gated 参照 {ref[k]:.4f}]')
        print(f'revin 第二视角单独: ROC={np.mean(revin_rocs):.4f} '
              f'AP={np.mean(revin_aps):.4f}')

        ok = (mean['D'] >= ref['D'] + EPS['D']
              and mean['ROC'] >= ref['ROC'] + EPS['ROC']
              and mean['AP'] >= ref['AP'] + EPS['AP']
              and mean['A'] >= ref['A'] + EPS['A'])
        print('=== 预注册判定 ===')
        print(f'D   {mean["D"]:.4f} vs {ref["D"] + EPS["D"]:.4f} '
              f'{">=" if mean["D"] >= ref["D"] + EPS["D"] else "<"} | '
              f'ROC {mean["ROC"]:.4f} vs {ref["ROC"] + EPS["ROC"]:.4f} '
              f'{">=" if mean["ROC"] >= ref["ROC"] + EPS["ROC"] else "<"}')
        print(f'AP  {mean["AP"]:.4f} vs {ref["AP"] + EPS["AP"]:.4f} '
              f'{">=" if mean["AP"] >= ref["AP"] + EPS["AP"] else "<"} | '
              f'A   {mean["A"]:.4f} vs {ref["A"] + EPS["A"]:.4f} '
              f'{">=" if mean["A"] >= ref["A"] + EPS["A"] else "<"}')
        print(f'>>> {"ADOPT dual_base（SMD/EXA 采纳双头 base 视角）" if ok else "KEEP GATED（未全过预注册线，维持 gated）"}\n')


if __name__ == '__main__':
    main()
