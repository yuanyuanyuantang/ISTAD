# prereg_swat_dualbase.md — SWAT 采纳 dual_base（**事后采纳记录，非预测性预注册**）

日期：2026-09-03。性质：**事后采纳**（post-hoc）。候选配置作为侧观察首次记录于
论文素材.md §3.0g(4)（2026-09-02，SWAT 双头预注册判定 min_z 未全过 → KEEP GATED 时，
dual_base 仅记录不判定）；本日用户在 SOTA 收尾攻坚中决策采纳。数字在写下述门槛之前
已经全部算出（sota_push_scan_SWAT.log），故本文件不构成预测性预注册，仅作决策留痕，
与先例 `dual_train/prereg_smd_minz.md`（SMD min_z 事后采纳）同性质。

## 采纳配置

**SWAT dual_base** = dual 联合训练模型（共享骨干，raw+RevIN 双重建损失 λ=1，配方忠实
复刻 SWAT g10：双卡 bs128 cosine lr0.001 epochs30，种子 48/89/2021，checkpoint 已存在于
`ISTAD/checkpoints/*_dual_*`）+ base 视角逐特征误差的 gated 聚合。
零训练：`vus_diag/istad_perfeat/SWAT_s{48,89,2021}_dual.npz` 转储已有。

## 判定门槛与结果（swat_dualbase_tables.log，回归校验 gated 行 5/5 逐位复现 v2.1 定稿行）

| 指标 | 门槛 | gated（v2.1 现行） | dual_base | 判定 |
|---|---|---|---|---|
| D (bf+PA) | ≥ 0.8701（KANAD 0.8721 − 0.002，健康参照打平带） | 0.8689±0.0034 | **0.8715±0.0072** | ✓ |
| C (POT+PA) | ≥ 0.8108（DLinear 诚实参照，保第一） | 0.8125±0.0054 | **0.8137±0.0095** | ✓ |
| ROC | ≥ 0.8413（不低于 gated） | 0.8413±0.0081 | **0.8501±0.0067** | ✓ |
| AP | ≥ 0.7317（不低于 gated） | 0.7317±0.0048 | **0.7484±0.0069** | ✓ |
| A (POT+裸F1) | ≥ 0.2583（不低于 gated） | 0.2583±0.3516 | **0.7037±0.0680** | ✓ |

五条全过 → **ADOPT dual_base，SWAT 行切换**。

## 同日否决记录（SOTA 收尾攻坚，全部零训练）

1. **EXA C 攻坚（目标 ≥0.9578）**：聚合家族 11 族全灭；单调变换（log1p/sqrt/pow0.25/
   winsor/gaussrank）如理论预期严格保持 D/ROC/AP 而对 C 无效（~0.943）；唯一有效路线
   为 EXA POT level 0.99→0.98（C 0.9594/A 0.4145 双第一，基线列 V2 数字已有），
   **用户决策不采纳**（事后标定刀锋 + 不愿再动协议），EXA C −0.012 作为唯一健康参照
   缺口收尾，维持 §3.0f(8) 结论。
2. **SMD D 攻坚（目标 ≥0.8515）**：存档 committee[3]{48,2021,2022}（C 0.7907/D 0.8577
   双超）经成员敏感性检查**否决**——4 个 3-种子子集仅该子集过线（其余 0.818–0.830），
   第一完全依赖剔除弱种子 s2025，属幸运种子谬误。全体 4 种子稳健聚合（sep-base
   median/trimmed）D≤0.833 不过；min_z committee4-median D=0.8596 过但 C 崩 0.7162
   （第一让给 TranAD 0.7825）。**用户决策维持 min_z 单模型**（C 第一 + D σ内打平
   优于 D 第一 + C 掉第四）。
3. **SWAT D 聚合家族扫描**：base/dual 双源 11 族全灭（最高 dual:trim1 0.8698 <
   dual_base 0.8715），无更优聚合。

## 新增写作资产：SWAT D 列展品表（sota_push_scan_SWAT.log，11 模型逐种子同口径）

- TimesNet@SWAT 三种子 D=0.9038/0.9064/0.9091，全部伴随 AP≈0.090、ROC≈0.24
  （**分数与标签反相关**）——极性反转展品逐种子实锤。
- DLinear@SWAT 三种子 D=0.8760/0.8764/0.8769，全部伴随 AP 0.153–0.175、ROC 0.57–0.64
  （排序接近随机）——**第二族展品**：oracle+PA 可把烂排序抬进 0.87 带。
- 健康带（AP>0.4）：KANAD D 0.8697–0.8734（AP≈0.717）、ISTAD dual_base 0.8715
  （AP 0.748、ROC 0.850 —— 健康带内 ROC/AP 双第一）。
- 写法：SWAT D 列声明限定为"健康分数带内与最强基线打平、排序指标第一；
  更高数值属 PA 病理（附展品表）"。

## 方法一致性说明（论文表述）

采纳后打分流程：PSM/SMD = dual min_z；SWAT = dual_base（联合训练 + gated 聚合 base 视角）；
EXA = 分开训练 + gated。SWAT 采纳 dual 的机制与 PSM dual_base 侧观察一致：
辅助 RevIN 损失对 base 路有正则收益（SWAT 上五指标全部受益、A 修复 POT 过冲）；
不宣称"联合训练修复了 RevIN 视角"（SWAT revin 视角 ROC 0.239 反相关未变）。
算力注记：dual 每步 2× 前向，参数量不变（303,330），推理 2× 前向须脚注。

## 数据/脚本

`vus_diag/{sota_push_scan.py, exa_transform_scan.py, smd_committee_robust.py,
swat_dualbase_tables.py}` + 日志 `sota_push_scan_{EXA,SMD,SWAT}.log`、
`exa_transform_scan.log`、`smd_committee_robust.log`、`swat_dualbase_tables.log`。
重算：`python swat_dualbase_tables.py`（官方行）+ `python sota_push_scan.py SWAT`（展品表）。
