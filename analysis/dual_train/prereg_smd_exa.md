# 预注册：SMD/EXA 双头扩展战役（2026-09-02，训练启动前落盘）

## 动机

双头联合训练（--istad_dual 1 --istad_dual_lambda 1.0）的 **base 视角**（gated 聚合）
在已完成的两个数据集上一致强于分开训练的 base（§3.0g 侧观察，当时因预注册纪律未采纳）：

- PSM：dual_base D 0.9266 vs sep 0.9061（+0.020）、ROC 0.7515 vs 0.7335、AP 0.5320 vs 0.5016
- SWAT：dual_base D 0.8715 vs sep 0.8689、ROC 0.8501 vs 0.8413、AP 0.7484 vs 0.7317、
  A 0.7037 vs 0.2583（修复 POT 过冲崩塌）

本战役在 SMD/EXA 上以相同配方 + dual 标志重训，按下列判据判定是否将 base 视角
（gated）采纳为最终配置。**目标：表 D 全表 SOTA-or-tied；顺带检查 A 表稳健性。**

## 训练配置（忠实复刻各自生产配方，仅加 dual 标志）

- SMD s48/2021/2022/2025：= SMD revin 模板配方（= leg_grid10 生产配方）
  sl96 enc38 grid10 spline3 order4 dropout0.2 lr0.01(type1) bs64 patience5 epochs100
- EXA s48/89/2021：= EXA_g10 生产配方（Exathlon/ISTAD.sh + 显式 grid10/spline3；
  legacy 默认 grid5 必须覆盖）
  sl100 enc19 grid10 spline3 order4 dropout0.1 lr1e-4(cosine) bs64 patience5 epochs40

checkpoint 目录名（dump 用）：
- anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_SMD_dual_s{seed}_0
- anomaly_detection_EXATHLON_ISTAD_EXATHLON_ftM_sl100_bmhgat_kan_tcn_bs64_EXA_dual_s{seed}_0

## 判定对象

**dual_base** = dual 联合模型的 base 视角 + shift-robust gated 聚合
（与 dual_tables_swat.py 完全同协议：gated_score + z 按 split + POT 逐数据集 lm 表 +
bf_pa 200/500+adjust；评估在逐特征误差转储上离线做，dump 脚本 MODE='dual'）。

参照值（当前采纳的 gated，4/3 种子均值，来源 gated_refresh.log）：

| 数据集 | A | AP | ROC | C | D | 种子 |
|---|---|---|---|---|---|---|
| SMD | 0.2208 | 0.1750 | 0.7700 | 0.7331 | 0.8361 | 48/2021/2022/2025 |
| EXA | 0.2410 | 0.6252 | 0.8744 | 0.9434 | 0.9606 | 48/89/2021 |

## 采纳判据（每数据集独立，四条全部满足才采纳，否则维持 gated）

1. **D 均值 ≥ gated D 均值 + 0.002**（对目标表有实质增益）
2. ROC 均值 ≥ gated ROC 均值 − 0.005（排序指标不回退）
3. AP 均值 ≥ gated AP 均值 − 0.005
4. A 均值 ≥ gated A 均值 − 0.02（可部署性不崩）

EXA 处于饱和带（D σ≈0.0005），判据 1 预期难以触发——EXA 未过线即维持 gated，
属预期结果，不算失败。

## 附带记录（不参与判定）

- revin 第二视角单独 ROC/AP（机制检查：SMD revin 分开训练时 ROC 0.733–0.769 正相关
  但弱于 gated base；EXA revin 首次训练）
- min_z 融合变体（诊断用；SMD 分开训练时已按无增益否决）
- dual 训练的 base 视角 A 值是否如 SWAT 一样修复 POT 过冲

## 机制与算力 caveat（写入论文，无论采纳与否）

双头收益来自联合标定 + 辅助 RevIN 损失正则，**不是**修复 RevIN 视角的排序能力
（PSM revin ROC 0.55→0.55、SWAT 0.24→0.239，联合训练前后不变）。训练 2× 前向；
推理只用 base 半区、成本不变。等算力对照（单视角 2× epochs）未做，论文注明。

## 教训绑定

dump 脚本 CKPT 映射必须加 SMD/EXA dual 显式键，任何非空 MODE 必须命中显式键、
禁止回退（3.0g 教训：错模型静默落盘）。
