# prereg_smd_sl100.md — SMD seq_len 96→100（2026-09-03，训练启动前落盘）

**性质：预测性预注册。** seq_len 轴的 SMD 段是最后一个未测的文档级基线错配
（素材 §3.3.5：SMD 基线 sl100 vs 我方 sl96；PSM/SWAT 已按 prereg_sl100.md Wave 1 判定
KEEP——PSM sl100 D 过主门槛但 ROC/AP 护栏双破，SWAT 全指标惰性）。用户选定"两个都跑"。

## 动机与先验（诚实版）

机制先验"更长窗口→每分数点更多上下文→段级 touching↑→PA 段填充（D 直指）"在 PSM 成立
（D +1.73pp）但伴随排序护栏击穿，在 SWAT 惰性（异常段长，sl96 已饱和覆盖）。SMD 为短段
困难集，方向不明——护栏就是为"PA 虚胖、排序变差"形态设计的，实测裁决。
SMD 另一动机：现行 D 缺口 −0.0034（0.8480 vs TimesNet 0.8514）且逐种子解剖显示缺口主要
由弱种子 s2025 拉大（s2021 上 ISTAD 0.8583 反胜 TimesNet 0.8509）——长窗若系统性抬 D，
4 种子均值有望过线。

## 臂设计（唯一改动 = seq_len）

SMD dual 现行采纳配方（λ=1.0 dual min_z）sl96→**sl100** × 4 种子 {48, 2021, 2022, 2025}，
其余一切忠实复刻（bs64、lr0.01、kanad_order 4、spline_order 3、dropout 0.2、grid10、
patience 5、epochs 100 早停）。脚本 `ISTAD/scripts/anomaly_detection/SMD/ISTAD_dual_sl100_s{seed}.sh`，
日志 `vus_diag/dual_train/smd_sl100_s{seed}.log`，GPU1 顺序执行。

## 评估管线（锁定）

dump：`dump_istad_perfeat.py` MODE='dual_sl100'（DS='SMD' 新增 CKPT 显式键 + 硬断言）。
TEST 平铺为 stride=W 非重叠（data_provider 大写 flag → loader 走 win_size 步长分支），
前缀逐点对齐：sl100 平铺 = 708,400 点 > 现行 708,384 → **截断到 708,384（=现行 label 长）**，
评估面与 v2.1 完全一致（尾部 16 点差异归零）。
打分：dual min_z（与现行采纳同族）；协议 = final_matrix/dual_tables_smd_exa 同源。
**回归纪律：判定前先用现存 SMD dual npz 复现 v2.2 行（5 指标），管线精确才信新数。**

## 判定门槛（4 种子 mean，ddof=0；现行参照 = v2.2 SMD 行）

| 指标 | 门槛 | 现行参照 |
|---|---|---|
| D (bf+PA) | ≥ **0.8515**（TimesNet 0.8514，名义超） | 0.8480±0.0063 |
| ROC | ≥ 0.7431（护栏 ref−0.005） | 0.7481±0.0219 |
| AP | ≥ 0.1570（护栏 ref−0.005） | 0.1620±0.0189 |
| C (POT+PA) | ≥ 0.8094（护栏 ref−0.010，SMD 的 C 第一不得明显倒退） | 0.8194±0.0165 |
| A | 不设判据（牺牲叙事延续） | 0.0863±0.0080 |

## 决策规则

- 全过 → ADOPT sl100 为 SMD 最终配置（v2.3 刷新五表；窗口 caveat 脚注消掉）。
- 任一不过 → KEEP sl96，**seq_len 轴全域关闭**（四数据集全部实测完毕），负结果如实入素材。
- 禁止事后换打分变体或种子；4 run 全训全报；单种子 pilot 不设。
