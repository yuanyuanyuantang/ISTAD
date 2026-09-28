#!/usr/bin/env python3
"""SWAT 机制检查：训练/测试分布偏移（PSM feature-3 类比的排查）。
用 StandardScaler(拟合训练集) 后的 z 空间，逐特征看测试集超出训练范围的程度，
以及超范围点中标签为异常的比例。"""
import numpy as np
import pandas as pd

root = '/data/modeluse/TS/ISTAD/dataset/SWAT'
tr = pd.read_csv(f'{root}/swat_train2.csv').values
te = pd.read_csv(f'{root}/swat2.csv').values
lb = te[:, -1].astype(int)
tr, te = tr[:, :-1], te[:, :-1]

mu, sd = tr.mean(axis=0), tr.std(axis=0) + 1e-8
trz, tez = (tr - mu) / sd, (te - mu) / sd
C = tr.shape[1]

print(f'train {tr.shape}  test {te.shape}  anom_frac={lb.mean():.4f}')
print(f'{"feat":>4s} {"tr_min":>7s} {"tr_max":>7s} {"te_min":>8s} {"te_max":>8s} '
      f'{"%OOD":>7s} {"%OOD_anom":>9s} {"shift_dir":>9s}')
rows = []
for c in range(C):
    lo, hi = trz[:, c].min(), trz[:, c].max()
    ood = (tez[:, c] > hi) | (tez[:, c] < lo)
    n_ood = int(ood.sum())
    pct = 100.0 * n_ood / len(te)
    pct_anom = 100.0 * lb[ood].mean() if n_ood else float('nan')
    # 偏移方向：测试中值相对训练中值
    shift = np.median(tez[:, c]) - np.median(trz[:, c])
    rows.append((c, lo, hi, tez[:, c].min(), tez[:, c].max(), pct, pct_anom, shift))
    if pct > 1.0:  # 只打印显著偏移的特征
        print(f'{c:>4d} {lo:7.2f} {hi:7.2f} {tez[:, c].min():8.2f} {tez[:, c].max():8.2f} '
              f'{pct:6.2f}% {pct_anom:8.2f}% {shift:+9.2f}')

rows.sort(key=lambda r: -r[5])
print('\nTop-8 按 %OOD:')
for c, lo, hi, tmin, tmax, pct, pct_anom, shift in rows[:8]:
    print(f'  f{c}: OOD {pct:6.2f}% (异常占 {pct_anom:5.1f}%)  '
          f'train[{lo:.2f},{hi:.2f}] test[{tmin:.2f},{tmax:.2f}] shift={shift:+.2f}')

# 全局：任意特征 OOD 的测试点
any_ood = np.zeros(len(te), bool)
for c in range(C):
    lo, hi = trz[:, c].min(), trz[:, c].max()
    any_ood |= (tez[:, c] > hi) | (tez[:, c] < lo)
print(f'\n任意特征 OOD 的测试点: {any_ood.sum()} ({100*any_ood.mean():.2f}%), '
      f'其中异常 {100*lb[any_ood].mean():.2f}%（整体异常率 {100*lb.mean():.2f}%）')
