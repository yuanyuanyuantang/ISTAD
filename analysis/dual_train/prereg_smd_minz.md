# SMD dual min_z 采纳决策记录（prereg_smd_minz.md，09-02）

**性质：事后采纳决策（post-hoc adoption decision），非预测性预注册，如实标注。**
背景：prereg_smd_exa.md（预测性预注册）判据对象为 dual_base，四条未过 → KEEP GATED 成立；
min_z 当时仅为附带记录（不参与判定）。用户在"C/D 表冲 SOTA 路线"决策中选定 min_z 路线。
本文件在锁定配置前落盘采纳门槛；全部数字已在 dual_tables_smd_exa.log 披露。
与 PSM dual min_z 采纳流程同性质：用户决策 + 全数字披露（PSM 的 min_z 是预注册对象，
SMD 的不是——差别如实记录，不混称）。

## 采纳门槛（四条全过才采纳；复算：python dual_tables_smd_exa.py，取 min_z 列）

数据：SMD dual checkpoint ×4 种子（48/2021/2022/2025），
融合 = min(z(gated_base), z(mean_revin))，z 按 split（与 PSM 采纳融合同族）。

| # | 判据 | 门槛线 | 实测（ddof=0） | 结果 |
|---|---|---|---|---|
| 1 | C（POT+PA）≥ TranAD 0.7825（成 SMD C 第一） | 0.7825 | **0.8194±0.0165** | ✓ |
| 2 | D（bf+PA）≥ TimesNet 0.8514−0.005（打平带） | 0.8464 | **0.8480±0.0063** | ✓ |
| 3 | ROC ≥ TimesNet 0.7507−0.005（comparable 护栏） | 0.7457 | **0.7481±0.0219** | ✓ |
| 4 | AP ≥ KANAD 0.1551−0.005（第一护栏） | 0.1501 | **0.1620±0.0189** | ✓ |

A（POT+裸 F1）不设判据：A 牺牲叙事已在 PSM 确认并接受，0.2208→0.0863 如实入表。
std 为 ddof=0（脚本输出原样）；论文统一口径时换算 ddof=1（×sqrt(4/3)）。

## 决策：ADOPT —— SMD 最终行由 gated 换为 dual min_z（同 4 dual checkpoint，零重训）

全局影响（ddof=0）：
- C 表 SMD 0.7331→**0.8194**（超 TranAD 0.7825，成 SMD C 第一）
- D 表 SMD 0.8361→**0.8480**（与 TimesNet 0.8514 σ内打平）
- ROC SMD 0.7700→**0.7481**（第一→第二，差 −0.0026 ≪ σ，comparable）
- AP SMD 0.1750→**0.1620**（名义第一保持，vs KANAD 0.1551，σ内）
- A SMD 0.2208→0.0863（本来已输 TranAD 0.2239，进一步远离，如实入表）

**最终配置（v2.1）：EXA=gated；PSM=dual min_z；SMD=dual min_z；SWAT=gated。**
