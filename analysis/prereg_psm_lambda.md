# prereg_psm_lambda.md — PSM dual λ 扫描（Wave 2，2026-09-03，训练启动前落盘）

**性质：预测性预注册。** 承接 prereg_sl100.md 的 Wave 2 依赖声明（基线配置已锁定 = sl64 现行）。
用户在"SOTA 收官战役"中选定"两个都跑（PSM λ + SMD sl100）"。

## 动机

D-PSM 现行 0.9696±0.0105 vs TimesNet 0.9732±0.0008（−0.0036，统计上打平但名义未超）。
PSM 双头的收益已被证明来自辅助 RevIN 损失的正则（dual_base > sep-base 侧观察、联合训练
不修复视角排序），λ 是该正则的唯一强度旋钮——现行 λ=1.0 从未扫描。机制先验方向不明：
λ>1 = 更强辅助正则（可能更强正则收益）；λ<1 = 辅助让位主任务（base 容量更自由）。实测裁决。

## 臂设计（唯一改动 = λ）

λ ∈ {0.5, 2.0} × 种子 {87, 90, 98, 2021} = 8 run，其余一切忠实复刻 PSM g10 dual 现行配方
（sl64、bs64、lr0.01、3 epochs、patience 5、kanad_order 6、dropout 0.3、grid10、dual 开）。
脚本 `ISTAD/scripts/anomaly_detection/PSM/ISTAD_dual_l{05,20}_s{seed}.sh`，
日志 `vus_diag/dual_train/psm_lambda_l{05,20}_s{seed}.log`，GPU0 顺序执行。

## 选择规则（**修订自 Wave 2 声明，修订理由如下**）

Wave 2 原声明"验证集 bf-F1 选择"。开训前核实发现不可行，理由：
1. PSM 的 vali split = train 后 20% 且 **无标签**（`PSMSegLoader` L69：`self.val =
   self.train[int(data_len*0.8):]`，train 无 label 文件）→ 验证集 bf-F1 不可算；
2. dual 模型的训练 vali_loss = base + λ·MSE(revin) 联合损失（`exp.vali` → `_dual_loss`），
   **跨 λ 臂不可比**（λ=2 臂天然多计一项）。

**修订后选择规则：base 路验证重建 MSE。** 8 个 checkpoint 训完后，各在 vali split
（stride-1 窗口，与训练 early-stopping 同一数据面）前向，取 base 半区对输入的 MSE；
MSE 较小者入选。平手（两臂相对差 < 1%）→ **KEEP λ=1.0（不采纳任何臂）**。
该规则零测试泄漏、λ 无关可比，且 base 路正是打分所用的路径。

测试集只评入选臂一次（4 种子全表）；未入选臂的测试数字仅日志留档透明化，**不可采纳**
（若入选臂未过门槛 → KEEP λ=1.0，无论另一臂测试数字如何）。

## 判定门槛（入选臂 4 种子 mean，ddof=0；现行参照 = v2.2 PSM 行）

| 指标 | 门槛 | 现行参照 |
|---|---|---|
| D (bf+PA) | ≥ **0.9733**（TimesNet 0.9732，名义超） | 0.9696±0.0105 |
| ROC | ≥ 0.7412（护栏 ref−0.005） | 0.7462±0.0211 |
| AP | ≥ 0.4824（护栏 ref−0.005） | 0.4874±0.0222 |
| C (POT+PA) | ≥ 0.9464（护栏 ref−0.010） | 0.9564±0.0086 |
| A | 不设判据（牺牲叙事延续） | 0.0996±0.0557 |

## 决策规则

- 全过 → ADOPT 入选 λ 为 PSM 最终配置（v2.3 刷新五表）。
- 任一不过 → KEEP λ=1.0，λ 轴关闭，负结果如实入素材。
- **禁止事后换打分变体**（min_z 家族冻结）；8 run 全训全报，禁止弃种子。
- 单种子 pilot 不设（PSM 4 种子方差证据充分，pilot 的种子运气已两次教训）。

## 评估管线（锁定）

dump：`dump_istad_perfeat.py` 新 MODE 'dual_l05'/'dual_l20'（CKPT 显式键 + 硬断言，
沿用 dual 双前向分支；λ 仅训练期生效，推理只需 istad_dual=1）。
打分：dual min_z（min(z(gated_base), z(mean_revin))，与现行采纳同族）；
协议 = final_matrix/dual_tables 同源（POT lm、bf_pa 两段式 + PA、AP/ROC）。
回归纪律：判定前先用现存 PSM dual npz 复现 v2.2 行（5 指标），管线精确才信新数。
