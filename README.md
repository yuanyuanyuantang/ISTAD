# ISTAD：模型、实验、论文与复现总说明

更新日期：2026-09-11

本文件是项目唯一的总说明，也是当前模型结论、实验口径和复现入口的统一来源。

论文正文方法稿位于 `paper/ISTAD_KBS_draft.md`。正文统一称 **ISTAD**；开发版本号仅在
本 README 的审计记录中保留，不进入论文方法、实验表标题或结论。

> 当前建议论文架构为 **ISTAD V4-HG rank-safe**。它在 V4 因果创新分数上加入受
> 数学保序约束的 HGAT-Lite 超边细化；三种子 `POT + point-adjust` 与
> `Best-F1 + point-adjust` Macro 为 **0.902270±0 / 0.945107±0**，与 V4 完全一致，
> 12/12 运行无不同 V4 排名反转。float64 严格复算中四集 ROC 均极小上升，但 SWaT
> AUC-PR 下降 0.003281；因此 HGAT 的排序证据是混合的，不能描述成全面精度提升。
> 整个神经分支仅 4,771–44,867 参数。V4-HG 在项目存档的 11 模型对比中，
> `POT + point-adjust` 与
> `Best-F1 + point-adjust` 均取得四个数据集的**探索性名义第一**。V4 及其共享 POT
> 安全系数是在观察测试结果后形成的，尚不能作为无偏的已确认 SOTA 结论。2026-09-10
> 最新基线审计进一步确认：MtsCID 等 2025–2026 方法已报告更高的重叠数据集 PA-F1，
> 因此这里的“第一”只限存档基线，不能外推成当前领域 SOTA。
>
> 最新开发分支 V7 已实现“正常训练因果先验约束的有向动态超图预测”。它在 PSM
> seeds 87/90/98 上通过内部机制门槛，但随后保持配置不变的五数据集 seed-87 冻结筛查
> 中没有一个数据集通过全部门槛：固定关系偏离分数在 5/5 数据集损害至少一个 ROC/AP
> 指标。已按预注册停止 seeds 90/98。V7.1 随后采用可退化图残差和可学习先验门，六数据集
> 三臂筛查仍仅 MSL 通过（1/6，多目标 0/4），再次停止多种子。因此 V7/V7.1 只保留为
> 开发诊断，**跨数据集机制主张失败，不替换 V4，也不构成 SOTA**。
>
> V8 shared-KAN HGAT 已完成最小实现和冻结 seed-87 四数据集筛查。它只把 HGAT-Lite 的
> `Linear(1, rank)` 换成共享 `SiLU + B-spline` 边函数，每个模型增加 72 个参数；EXA/SMD
> 的 HGAT-only AP 分别提高 0.0700/0.0264，但 SWaT 下降 0.2827，四集宏观下降 0.0473，
> 且 PSM 触发训练侧可靠性回退。故预注册门槛失败，不运行 seeds 90/98，不替换 V4-HG。
> V9 随后改用精确嵌套的 bounded-residual KAN：保留原线性投影，只加幅度受限且零初始化
> 的样条残差。它将宏观 AP 回退缩小到 0.001794，并使宏观 ROC 上升 0.000185，但仍只有
> PSM 的 AP 改善，未通过冻结门槛。因此 V9 同样不替换 V4-HG。
>
> V10 将关系学习从原始值域迁移到有符号、训练集标准化的 VAR 新息域，并启用已有的
> training-only 可靠性融合。冻结 seed-87 筛查中，四集 AP 全部提高，Macro AP/ROC
> 相对 V4-HG 提高 0.018838/0.007024，Best-F1+PA Macro 提高 0.001546；但 POT+PA
> Macro 下降 0.003366，超过预注册的 0.001 护栏。因此 V10 证明了新息域关系建模的潜力，
> 但整体继续门失败，不运行 seeds 90/98，也不替换 V4-HG。

## 0. KBS 投稿实验冻结（2026-09-11）

从本节起不再用已揭盲的 EXA、PSM、SMD、SWaT、MSL 或 SMAP 测试标签选择模型、阈值、
种子或融合权重。论文中各版本的角色固定如下：

| 角色 | 固定版本 | 证据强度 | 论文位置 |
|---|---|---|---|
| 主模型 | V4-HG rank-safe | 四数据集、三种子冻结结果 | 主方法与主实验 |
| 机制扩展 | V10 innovation-guided BR-KAN-HGAT | 四数据集、seed 87；整体门失败 | 消融/讨论，不进主 SOTA 行 |
| 投影证据 | V10 Linear vs BR-KAN | 等条件 seed-87；BR-KAN 通过保留规则 | 机制消融 |
| 负结果 | V5–V9、V7.1、MSL/SMAP | 按各自冻结规则停止 | 附录或局限性 |
| 历史基线 | 存档 11 模型 | 协议未与最新方法完全统一 | 文献兼容表，限定为存档比较 |

论文实验表按以下顺序组织，数字直接取自本文件后续对应小节：

1. 主表：V4-HG 的 ROC-AUC、AUC-PR、POT 裸 F1、Event-F1、POT+PA 与 Best-F1+PA；
2. 文献兼容表：存档 11 基线的两张 PA 表，标题必须包含“stored baselines”；
3. 组件消融：V4、HGAT-only、V4-HG，并单列 V10 Linear/BR-KAN 投影消融；
4. 效率表：参数量、HGAT 参数、延迟、显存和 checkpoint 大小；
5. 附录：V5–V9、V7/V7.1 和 MSL/SMAP 的停止结果，不能只选择有利数据集。

统计口径同时冻结：V4-HG 仅对三个固定种子报告 mean±population-std；接近零的方差源于
闭式 V4 主分数和受限并列细化，不得表述成普遍训练稳定性。V10 只有一个开发种子，必须
逐值报告，不能补写 `mean±std`、显著性或跨种子稳定性。四数据集 Macro 只是描述性算术
平均，时间点存在强自相关，禁止把时间点当独立样本做普通 t 检验。Best-F1+PA 和 Best
裸 F1 均为测试标签 oracle 上限。归档 POT 不读取测试标签，但会接收无标签测试分数流，
应视为转导式兼容协议；严格部署结论需另用只由正常训练段确定的固定阈值。

可用于摘要/结论的创新概括固定为：**训练侧因果新息建模、低秩动态超图关系汇聚、具有
保序保证的受控融合，以及面向 PA 膨胀的协议透明评估**。BR-KAN 只能描述为新息域关系
映射的支持性扩展。当前不能使用“当前 SOTA”“显著优于最新方法”或“HGAT 提高 PA-F1”。

由于当前目录没有 Git 元数据，关键实现、入口和结果文件由 `analysis/KBS_FREEZE.sha256`
锁定。修改论文代码前先复制为新版本；当前冻结状态可在项目根目录验证：

```bash
sha256sum -c analysis/KBS_FREEZE.sha256
```

2026-09-15 补充（post-freeze supplementary）：为论文 Section 6（Interpretability
Analysis）新增预注册可解释性实验包（真实案例 / 注入式变量定位 / 跨种子稳定性），
入口 `python analysis/evaluate_v4hg_interpretability.py --stage all`，协议、结果与
判定见 `analysis/v4hg_interpretability/`。该包只读冻结 checkpoint 与数据，不重训、
不选阈值、不参与任何模型选择；其新文件与相关代码变更已并入冻结清单重建。

## 1. 项目概述

ISTAD（Interpretable Spatial-Temporal Anomaly Detection）面向多变量时间序列异常检测。
项目包含九条有明确版本边界的能力：

1. KAN-TCN、HGAT、KANAD 重建及证据头，用于学习时空表征和逐特征解释；
2. V4 training-only causal innovation branch，用正常训练序列闭式拟合轻量异常分数；
3. **V4-HG rank-safe**，用简化 HGAT 的动态 incidence 聚合 V4 逐变量创新，并只在
   小于一个经验排名步长的范围内细化并列分数；
4. V6b target-forecast diagnostic，面向 MSL/SMAP 只预测官方目标列，并严格隔离实体边界；
5. V7 causal-prior directed hypergraph，用正常训练关系先验约束动态超图，并同时参与预测和打分；
6. V7.1 residual causal hypergraph，将图关系改为可退化预测残差并学习 graph/prior 门。
7. V8 shared-KAN HGAT，用共享样条边函数替换 HGAT-Lite 标量投影，作为冻结负消融保留。
8. V9 bounded-residual KAN-HGAT，以零初始化有界样条残差精确嵌套线性 HGAT。
9. V10 innovation-guided BR-KAN-HGAT，在 VAR 新息域学习动态关系并进行训练侧可靠性融合。

最新建议 PA 行使用第三条 V4-HG；其 F1 与第二条纯 V4 一致，但 HGAT 确实负责超边级
证据与并列排序。若只追求最小部署，可退化成纯 V4，仅保存系数矩阵和训练 ECDF 参考。
第四至九条分支目前都是揭盲数据上的开发诊断，不能作为独立 SOTA 证据。

### 论文可用的核心表述

ISTAD V4-HG 使用正常训练数据拟合因果创新残差，以极轻量动态超图汇聚变量间高阶关系，
并通过保序分数约束避免较弱图分支覆盖 V4 主排序。它在
Exathlon、PSM、SMD 和 SWaT 上取得有竞争力的 point-adjusted F1。与此同时，实验揭示
了 point adjustment 对稀疏尖峰检测器的偏好，因此 PA 结果必须与无 PA 的 AP、ROC 和
裸 F1 同时报告。不能把 V4-HG 与 V4 相同的 PA-F1 表述成 HGAT 带来了 F1 增益；HGAT
的实证作用是可复现的并列细化和高阶关系解释。V7 只能作为失败的开发诊断，不能表述为 SOTA。

## 2. V4 方法

给定标准化后的多变量序列 \(\mathbf{x}_t\)，在严格正常训练段上拟合一阶 ridge VAR：

\[
\hat{\mathbf{x}}_t=\mathbf{x}_{t-1}\mathbf{W},\qquad
\mathbf{W}=(X^\top X+10^{-2}I)^{-1}X^\top Y.
\]

异常分数管线如下：

1. 计算一步预测创新残差；
2. 根据正常训练残差中的近常量通道比例自动选择池化：低于 10% 使用逐特征标准化最大值
   `sparse`，否则使用全通道均方值 `dense`；
3. 仅使用正常训练创新分数建立 ECDF，将不同数据集统一到相同尺度；
4. POT 前使用单调映射 \(-\log(1-u)\)；
5. 在历史固定 POT 参数后施加所有数据集共享的 1.04 尾部安全系数，减少极端尾部估计
   偏低产生的孤立误报。

四个数据集自动选择为 EXA/PSM/SMD=`sparse`、SWaT=`dense`。SMD 的拟合、窗口和 PA
均隔离 28 台机器的实体边界。

该创新分支不读取测试标签来拟合 VAR、尺度、池化方式或 ECDF。需要注意，1.04 安全系数
是在开发过程中观察测试结果后选定的，因此整个 V4 仍属于探索候选。

### 2.1 V4-HG：保序超图细化（当前建议）

V4-HG 保留上面的 V4 分支，并增加一条极简关系路径：

```text
X -> causal Conv -> rank-8 HGAT-Lite -> dynamic sparse incidence H(t)
V4 feature innovation e(t,f) --H(t)^T--> hyperedge evidence g(t)
```

HGAT 仅用正常训练的重构与合成去噪任务学习。移除旧模型中的 TCN/KAN-TCN、GRU、
KANAD、RevIN、dual 和 evidence head。超边证据按 V4 已由训练集选出的 sparse/dense
规则池化，再用正常训练 ECDF 校准。最终分数为：

\[
s_t=\frac{q_t+\epsilon g_t}{1+\epsilon},\qquad
\epsilon=\frac{0.5}{n_{train}+1}.
\]

相邻不同 V4 经验排名相差 `1/(n_train+1)`，而 `0<=g<=1`，所以该修正严格小于一个
排名步长：HGAT 可以打破 V4 同分（尤其尾部饱和分数），但不能反转不同的 V4 训练排名。
另有只读正常训练尾稳定 kill-switch，失败时精确返回 V4。

| 数据集 | 整个神经分支参数 | 其中 HGAT 参数 | rank-safe epsilon |
|---|---:|---:|---:|
| Exathlon | 4,771 | 274 | 7.092e-6 |
| PSM | 7,303 | 346 | 4.718e-6 |
| SMD | 14,522 | 506 | 8.840e-7 |
| SWaT | 44,867 | 618 | 1.263e-6 |

三种子 12/12 运行中，不同 V4 排名反转数均为 0；每次都有数万级并列组被 HGAT 细化。
POT+PA 与 Best-F1+PA 和 V4 逐位一致。统计分数改用 float64 保存后，四集 ROC 都有
极小正向细化，但 SWaT AUC-PR 平均下降 0.003281；随机并列对照还显示前三集的 AP
优势并不显著高于任意并列打破。该结果证明图分支非空执行和保序约束成立，却不能支持
HGAT 带来稳定精度提升。完整无 PA、事件级、随机对照与效率结果见
`analysis/v4_hgat_paper/`，负融合实验见 `analysis/v4_hgat_integrated/RESULTS.md`。

#### 2.1.1 V8 shared-KAN HGAT（冻结负消融）

V8 保留 V4-HG 的全部数据流、训练和 rank-safe 评分，只把每个节点共享的标量线性投影
替换为 `SiLU` 残差加三次 B 样条：

\[
\phi_r(x)=a_r\operatorname{SiLU}(x)+\sum_k c_{r,k}B_k(x)+b_r.
\]

复用项目已有 `BSplineBasis`，不增加第三方 KAN 依赖；rank=8、grid=5、order=3 时投影
从 8 个参数增至 80 个参数，即整个模型固定增加 72 个参数。冻结 seed-87 结果为：

| 数据集 | 线性 HGAT AP | KAN HGAT AP | 差值 | POT+PA | Best-F1+PA |
|---|---:|---:|---:|---:|---:|
| Exathlon | 0.498319 | 0.568303 | +0.069984 | 0.959325 | 0.962799 |
| PSM | 0.412556 | 0.409627 | -0.002928 | 0.969713 | 0.982243 |
| SMD | 0.120110 | 0.146495 | +0.026385 | 0.847780 | 0.882938 |
| SWaT | 0.557061 | 0.274347 | -0.282714 | 0.832263 | 0.952449 |

KAN 的 HGAT-only 宏观 AP 差值为 -0.047318，未通过冻结诊断；PA 两表不变来自保序保护，
不是 KAN 的收益。完整预注册、指标、哈希和判定见 `analysis/v8_kan_hgat/`。该分支可复现，
但论文只能作为负消融，不能包装成主方法。

#### 2.1.2 V9 bounded-residual KAN-HGAT（稳定但未晋级）

V9 不再替换线性投影，而是采用精确可退化的任务专用关系函数：

\[
\phi(x)=Wx+0.1\tanh\!\left(\sum_k c_k B_k(x)\right).
\]

样条系数零初始化，因此同一 seed 下初始参数、随机数状态和输出与线性 V4-HG 逐位一致；
每个关系维的非线性修正幅度不超过 0.1。rank=8 时仅增加 64 个参数。seed-87 筛查中，
四集样条 L2 范数均非零、rank inversion 均为 0、PA 两表与 V4 完全一致。HGAT-only AP
差值为 EXA -0.007538、PSM +0.003143、SMD -0.001783、SWaT -0.000997，宏观
-0.001794；ROC 宏观 +0.000185。稳定性明显好于 V8，但未满足宏观 AP 非负和至少两集
改善的冻结门槛，故不运行 seeds 90/98。详见 `analysis/v9_brkan_hgat/`。

#### 2.1.3 V10 innovation-guided BR-KAN-HGAT（排序改善，整体门未通过）

V10 保持 V4 分数不变，但将神经关系分支的输入替换为训练集拟合的有符号标准化新息：

```text
X -> ridge-VAR(1) -> signed standardized innovation Z(t)
Z(t) -> causal Conv -> BR-KAN -> HGAT-Lite -> dynamic incidence H(t)
V4 magnitude evidence e(t,f) --H(t)^T--> graph evidence g(t)
q(t), g(t) -> normal-tail reliability mix -> train-only ECDF -> s(t)
```

VAR 系数、残差尺度、ECDF 和融合可靠性均只读取正常训练数据。BR-KAN 仍只增加 64 个参数。
冻结 seed-87 结果如下：

| 数据集 | V4-HG AP | V10 AP | ΔAP | V4-HG POT+PA | V10 POT+PA | V4-HG Best-F1+PA | V10 Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exathlon | 0.426348 | 0.454850 | +0.028501 | 0.959325 | 0.961231 | 0.962799 | 0.963512 |
| PSM | 0.427683 | 0.428805 | +0.001122 | 0.969713 | 0.966600 | 0.982243 | 0.979426 |
| SMD | 0.117178 | 0.127282 | +0.010104 | 0.847780 | 0.840440 | 0.882938 | 0.890056 |
| SWaT | 0.528790 | 0.564416 | +0.035626 | 0.832263 | 0.827348 | 0.952449 | 0.953618 |
| **Macro** | **0.375000** | **0.393838** | **+0.018838** | **0.902270** | **0.898905** | **0.945107** | **0.946653** |

四集 AP 均为正增益，说明新息域关系建模比 V9 的原始值域关系映射更合理；但 POT+PA
Macro 回退 0.003366，违反冻结护栏。按预注册停止多种子，不允许再根据这四集标签调融合。
完整协议与机器可读结果见 `analysis/v10_innovation_brkan_hgat/`。

#### 2.1.4 V10 等条件投影消融（BR-KAN 通过保留规则）

为区分 V10 的收益来自新息输入还是 BR-KAN，在 V10 已停止后补做 seed-87 解释性消融：
新息输入、HGAT-Lite、训练目标、可靠性融合及所有超参数保持不变，只将 BR-KAN 投影替换为
线性投影。该实验在运行前写明保留规则，但由于使用已揭盲测试集，不能用于晋级 V10。

| 数据集 | Linear AP | BR-KAN AP | ΔAP | Linear ROC | BR-KAN ROC | ΔROC |
|---|---:|---:|---:|---:|---:|---:|
| Exathlon | 0.453607 | 0.454850 | +0.001243 | 0.836411 | 0.836895 | +0.000484 |
| PSM | 0.432003 | 0.428805 | -0.003198 | 0.660782 | 0.658022 | -0.002761 |
| SMD | 0.125330 | 0.127282 | +0.001952 | 0.688875 | 0.689605 | +0.000730 |
| SWaT | 0.532461 | 0.564416 | +0.031955 | 0.847506 | 0.857717 | +0.010210 |
| **Macro** | **0.385850** | **0.393838** | **+0.007988** | **0.758394** | **0.760559** | **+0.002166** |

BR-KAN 的 AP 在 3/4 数据集非负，Macro AP、ROC、POT+PA 和 Best-F1+PA 分别相对线性
投影提高 0.007988、0.002166、0.000426 和 0.000944，仅增加 64 个参数。因此它满足
“Macro AP 更高且至少 2/4 数据集不下降”的冻结保留规则。论文可据此主张受限残差样条对
新息域关系映射有支持性贡献；不能写成独立确认、显著性结果或 V10 已超过冻结整体门槛。

### 2.2 历史 HGAT-Lite 消融与多种子融合确认

旧 HGAT 将每个传感器标量先投影到最高 256 维，执行两次方阵变换，并显式构造
`[batch, node, hyperedge, 2D]` 注意力张量。HGAT-Lite 改为：

1. 在 16 维关系空间融合当前传感器值与传感器身份嵌入；
2. 通过低秩节点/超边嵌入动态生成 incidence；
3. 保留每条超边的 Top-K 节点并修复孤立节点；
4. 使用同一 incidence 完成归一化的 node→edge→node 传播；
5. 通过可学习标量门控输出空间关系修正量。

旧 HGAT 仍为默认值并保留 checkpoint 兼容；新模块通过
`--istad_spatial_type lite --istad_hgat_rank 16` 显式启用。

| 数据集 | 旧空间分支参数 | HGAT-Lite 参数 | 降幅 |
|---|---:|---:|---:|
| Exathlon | 2,677 | 546 | 79.6% |
| PSM | 2,905 | 690 | 76.2% |
| SMD | 3,558 | 1,010 | 71.6% |
| SWaT | 138,513 | 1,234 | **99.1%** |

SWaT 整模型参数由 303,330 降至 166,051（−45.3%）；A6000、batch=8 的单次先导
测量中，峰值显存由约 1,650.9 MiB 降至 27.6 MiB，前向延迟由 9.756 ms/batch 降至
9.381 ms/batch。显存与延迟需在冻结环境中重复测量后才能作为论文正式数字。

PSM seed 87 的同配方单种子先导对比：

| 空间模块 | AUC-ROC | AUC-PR | Best-F1+PA | Train-p99 裸 F1 |
|---|---:|---:|---:|---:|
| 旧 HGAT | 0.7269 | 0.5127 | 0.9546 | 0.3498 |
| **HGAT-Lite** | **0.7560** | **0.5303** | **0.9586** | 0.3274 |

HGAT-Lite 的先导排序指标和 PA 上限更好，但裸 F1 下降 0.0224。该结果仅有一个种子，且
来自已观察过的 PSM 测试集，因此随后冻结配置，以 seed 90/98 进行三臂确认。下表为两次
确认运行的神经 `model_score` 均值（不混入创新分数）：

| 空间模块 | AUC-ROC | AUC-PR | Train-p99 裸 F1 | Best-F1+PA |
|---|---:|---:|---:|---:|
| 旧 HGAT | 0.7504 | 0.5196 | **0.4008** | 0.9553 |
| HGAT-Lite | 0.7419 | 0.5669 | 0.3384 | 0.9623 |
| **无空间分支** | **0.7623** | **0.5828** | 0.2666 | **0.9688** |

Lite 相对旧 HGAT 的 AUC-PR 提高 0.0473、Best-F1+PA 提高 0.0071，但 AUC-ROC 降低
0.0084、Train-p99 裸 F1 降低 0.0624；相对无空间分支的 AUC-PR、AUC-ROC 和
Best-F1+PA 也都更低。预注册五项门槛中第 2、3、5 项失败，因此**不将 HGAT-Lite
升为默认结构**。旧 HGAT 继续用于 checkpoint 兼容和原有解释性实验；V4 精简部署移除
空间分支。完整逐种子结果见
`analysis/istad_v4_lite_hgat/PSM_confirmation_s90_s98.json`。

同一批运行的 `innovation_fused(weight=0.001)` 文献兼容 PA 指标几乎不受空间结构影响：
Lite 的 PSM 两种子均值为 POT+PA 0.9694、Best-F1+PA 0.9824；纯创新 V4 分别为
0.9697、0.9822。这说明原先的存档表高分基本保持，但贡献来自 V4 因果创新分支，不能
表述成 HGAT-Lite 带来的性能提升。

#### 四数据集完整运行（seed 87）

在相同 HGAT-Lite 配置下完成 Exathlon、PSM、SMD、SWaT 全部训练；PSM 复用配置完全
相同的 seed-87 先导产物。下表比较 `innovation_fused(weight=0.001)` 与正式 V4：

| 数据集 | V4 POT+PA | Lite POT+PA | 差值 | V4 Best-F1+PA | Lite Best-F1+PA | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| Exathlon | 0.9593 | **0.9593** | +0.0000 | 0.9628 | **0.9632** | +0.0004 |
| PSM | **0.9697** | 0.9695 | −0.0003 | 0.9822 | **0.9824** | +0.0001 |
| SMD | 0.8478 | **0.8487** | +0.0009 | 0.8829 | **0.8853** | +0.0023 |
| SWaT | **0.8323** | 0.8201 | −0.0121 | 0.9524 | **0.9525** | +0.0001 |
| Macro-F1 | **0.9023** | 0.8994 | −0.0029 | 0.9451 | **0.9458** | +0.0007 |

HGAT-Lite 的 Best-F1+PA 四列均小幅提高，但 POT+PA 在 SWaT 上降至 0.8201，低于正式
V4 的 0.8323，也略低于存档 TimesNet 的 0.8215。因此完整运行仍不支持用 HGAT-Lite
替换正式 V4；它适合作为参数效率消融。

针对融合后分布不再校准的问题，进一步冻结并测试了训练-only ECDF：保持融合权重
0.001 和全部 POT 参数不变，只用正常训练融合分数建立经验分布，再映射测试分数。结果为：

| 数据集 | Train-ECDF POT+PA | 最强外部基线 | 领先 | Train-ECDF Best-F1+PA | 最强外部基线 | 领先 |
|---|---:|---:|---:|---:|---:|---:|
| Exathlon | **0.9593** | 0.9577 | +0.0016 | **0.9631** | 0.9612 | +0.0019 |
| PSM | **0.9695** | 0.9672 | +0.0023 | **0.9824** | 0.9732 | +0.0092 |
| SMD | **0.8477** | 0.8194 | +0.0283 | **0.8851** | 0.8514 | +0.0337 |
| SWaT | **0.8323** | 0.8215 | +0.0108 | **0.9524** | 0.9064 | +0.0460 |
| Macro-F1 | **0.9022** | 0.8786 | +0.0236 | **0.9457** | 0.9218 | +0.0239 |

该候选相对外部存档基线在两种 PA 协议均为 4/4 名义第一。内部冻结门槛仍未完全通过：
SMD 相对未重校准 Lite 下降 0.001056，略超 0.001 护栏；POT Macro 也比正式 V4 的
0.902270 低 0.000107。因此当前标记为**双 PA 存档基线领先候选**，尚不替换正式 V4。

#### 冻结的三种子确认（seed 87/90/98）

在 seed 87 开发结果之后先写入确认计划，再保持架构、融合权重、训练-only ECDF、数据集
配方、POT 参数与 PA 实现全部不变，运行 seed 90/98。seed 87 明确属于开发种子，90/98
属于确认种子；以下 `±` 为三种子的总体标准差，不把它表述成独立测试集验证。

| 数据集 | POT+PA（mean±std） | 最强外部基线 | 领先 | 胜出种子数 |
|---|---:|---:|---:|---:|
| Exathlon | **0.9592±0.0001** | 0.9577 | +0.0015 | 3/3 |
| PSM | **0.9695±0.0000** | 0.9672 | +0.0023 | 3/3 |
| SMD | **0.8472±0.0005** | 0.8194 | +0.0278 | 3/3 |
| SWaT | **0.8282±0.0057** | 0.8215 | +0.0067 | 2/3 |
| Macro-F1 | **0.9010±0.0015** | 0.8786 | +0.0224 | — |

| 数据集 | Best-F1+PA（mean±std） | 最强外部基线 | 领先 | 胜出种子数 |
|---|---:|---:|---:|---:|
| Exathlon | **0.9631±0.0000** | 0.9612 | +0.0019 | 3/3 |
| PSM | **0.9825±0.0000** | 0.9732 | +0.0093 | 3/3 |
| SMD | **0.8847±0.0009** | 0.8514 | +0.0333 | 3/3 |
| SWaT | **0.9524±0.0000** | 0.9064 | +0.0460 | 3/3 |
| Macro-F1 | **0.9456±0.0002** | 0.9218 | +0.0238 | — |

逐种子 F1 如下，便于定位方差来源：

| 数据集 | POT s87 | POT s90 | POT s98 | Best s87 | Best s90 | Best s98 |
|---|---:|---:|---:|---:|---:|---:|
| Exathlon | 0.9593 | 0.9591 | 0.9591 | 0.9631 | 0.9631 | 0.9632 |
| PSM | 0.9695 | 0.9695 | 0.9695 | 0.9824 | 0.9825 | 0.9824 |
| SMD | 0.8477 | 0.8466 | 0.8473 | 0.8851 | 0.8855 | 0.8834 |
| SWaT | 0.8323 | 0.8201 | 0.8323 | 0.9524 | 0.9523 | 0.9524 |
| Macro-F1 | 0.9022 | 0.8988 | 0.9020 | 0.9457 | 0.9458 | 0.9453 |

预注册的外部基线均值门槛、2/3 稳定性门槛、运行完整性和 SMD 实体隔离均通过；唯一失败
项是三种子 POT+PA Macro `0.901005` 未达到正式 V4 的 `0.902270`。因此最终决定是：
**将该配置保留为多种子存档基线领先候选，不替换正式 V4**。其中 SWaT POT 对神经
种子仍敏感（std 0.0057），论文必须展示方差；Best-F1+PA 使用测试标签搜索，只能称为
oracle 上限。精确逐种子指标、哈希和门槛判定见
`analysis/istad_v4_lite_hgat/multiseed_confirmation_s87_s90_s98.json`。

HGAT-Lite 神经分支自身的三种子无 PA 指标如下，证明融合 PA 分数与模型表征应分开报告：

| 数据集 | AUC-ROC（mean±std） | AUC-PR（mean±std） | Train-p99 裸 F1（mean±std） |
|---|---:|---:|---:|
| Exathlon | 0.8721±0.0016 | 0.6189±0.0041 | 0.3849±0.0007 |
| PSM | 0.7466±0.0112 | 0.5547±0.0180 | 0.3347±0.0198 |
| SMD | 0.7879±0.0104 | 0.2134±0.0231 | 0.2771±0.0146 |
| SWaT | 0.8424±0.0085 | 0.7063±0.0190 | 0.2912±0.0058 |

这些无 PA 数字并未形成四集统一 SOTA，尤其固定训练阈值裸 F1 明显低于 PA-F1；论文中
应把它们放在 PA 表旁边，而不是用 point adjustment 掩盖固定阈值表现。

## 3. 数据集与评估协议

| 数据集 | 特征数 | 窗口 | V4 评估点数 | 备注 |
|---|---:|---:|---:|---|
| Exathlon | 19 | 100 | 52,800 | 应用性能监控 |
| PSM | 25 | 64 | 87,808 | 服务器指标 |
| SMD | 38 | 96 | 706,560 | 28 个实体，边界感知 |
| SWaT | 51 | 96 | 449,856 | 工业控制系统 |

数据目录传给评估脚本的 `--data-root` 后应具有以下结构：

```text
<data-root>/
├── EXATHLON/
├── PSM/
├── SMD/
└── SWAT/
```

公开来源：SMD 来自 OmniAnomaly 仓库，PSM 来自 eBay PSM，Exathlon 来自 Exathlon
benchmark，SWaT 需向 SUTD iTrust 申请。使用或再分发前应分别检查数据许可证。
MSL/SMAP 的实体和列语义以 [Telemanom 原始实现](https://github.com/khundman/telemanom)
及其[论文](https://arxiv.org/abs/1802.04431)为依据：每个航天器通道独立建模，第 0 列为
目标遥测，其余列提供命令上下文。

### 指标定义

| 协议 | 阈值 | Point adjustment | 用途 |
|---|---|---|---|
| POT + PA | SPOT，`q=1e-5` | 是 | 文献兼容的无监督阈值结果 |
| Best-F1 + PA | 测试标签上 200 次粗搜 + 500 次细搜 | 是 | Oracle 上限，不可部署 |
| AUC-PR / AUC-ROC | 无阈值 | 否 | 衡量整体排序质量 |
| 裸 F1 | 训练集阈值 | 否 | 更接近实际部署表现 |

V4 的 POT 设置：

| 数据集 | Level | 历史乘子 | 共享安全系数 | 有效乘子 |
|---|---:|---:|---:|---:|
| Exathlon | 0.9900 | 1.00 | 1.04 | 1.040 |
| PSM | 0.9800 | 0.90 | 1.04 | 0.936 |
| SMD | 0.9880 | 1.00 | 1.04 | 1.040 |
| SWaT | 0.9999 | 1.20 | 1.04 | 1.248 |

## 4. 最新结果

以下为项目当前存档基线下的比较。粗体表示该列最高值；Macro-F1 是四个数据集的算术平均。

### 4.1 POT + point-adjust

| 模型 | Exathlon | PSM | SMD | SWaT | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| **ISTAD V4-HG (rank-safe)** | **0.9593** | **0.9697** | **0.8478** | **0.8323** | **0.9023** |
| ISTAD V4（无 HGAT） | 0.9593 | 0.9697 | 0.8478 | 0.8323 | 0.9023 |
| ISTAD v2.2 | 0.9434 | 0.9564 | 0.8194 | 0.8137 | 0.8832 |
| TimesNet | 0.9530 | 0.9672 | 0.7726 | 0.8215 | 0.8786 |
| DLinear | 0.9556 | 0.9663 | 0.7617 | 0.8108 | 0.8736 |
| KANAD | 0.9577 | 0.9299 | 0.7195 | 0.0000 | 0.6518 |
| DAGMM | 0.9510 | 0.7863 | 0.6544 | 0.8025 | 0.7986 |
| GDN | 0.8916 | 0.6999 | 0.3944 | 0.2987 | 0.5712 |
| LSTM-AD | 0.9171 | 0.6985 | 0.0432 | 0.0000 | 0.4147 |
| MAD-GAN | 0.9516 | 0.7386 | 0.4249 | 0.5579 | 0.6683 |
| MTAD-GAT | 0.7924 | 0.9188 | 0.1284 | 0.3028 | 0.5356 |
| OmniAnomaly | 0.9394 | 0.7478 | 0.5149 | 0.0000 | 0.5505 |
| TranAD | 0.9481 | 0.7775 | 0.7825 | 0.8057 | 0.8285 |

V4 相对各列原最强结果分别领先：Exathlon +0.0016、PSM +0.0025、SMD +0.0284、
SWaT +0.0108。

### 4.2 Best-F1 + point-adjust

| 模型 | Exathlon | PSM | SMD | SWaT | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| **ISTAD V4-HG (rank-safe)** | **0.9628** | **0.9822** | **0.8829** | **0.9524** | **0.9451** |
| ISTAD V4（无 HGAT） | 0.9628 | 0.9822 | 0.8829 | 0.9524 | 0.9451 |
| ISTAD v2.2 | 0.9606 | 0.9696 | 0.8480 | 0.8715 | 0.9124 |
| TimesNet | 0.9563 | 0.9732 | 0.8514 | 0.9064 | 0.9218 |
| DLinear | 0.9601 | 0.9715 | 0.8440 | 0.8764 | 0.9130 |
| KANAD | 0.9612 | 0.9503 | 0.8386 | 0.8721 | 0.9056 |
| DAGMM | 0.9542 | 0.9296 | 0.6773 | 0.8060 | 0.8418 |
| GDN | 0.9177 | 0.7360 | 0.4237 | 0.3050 | 0.5956 |
| LSTM-AD | 0.9357 | 0.8296 | 0.1648 | 0.8027 | 0.6832 |
| MAD-GAN | 0.9553 | 0.9095 | 0.4517 | 0.7943 | 0.7777 |
| MTAD-GAT | 0.8690 | 0.9673 | 0.3854 | 0.4282 | 0.6625 |
| OmniAnomaly | 0.9479 | 0.8756 | 0.5636 | 0.7996 | 0.7967 |
| TranAD | 0.9502 | 0.9469 | 0.7921 | 0.8091 | 0.8746 |

V4-HG 的 Macro-F1 为 0.9451，比第二名 TimesNet 高 0.0233；它和无 HGAT 的 V4 相同，
因此这张表支持“保留 V4 指标”，不支持“HGAT 提高 PA-F1”。

### 4.3 无 PA 排序指标

V4-HG 的三种子平均排序表现如下。这组数字明显弱于项目的 V3 深层证据分数，因此不能把
PA 表的提升解释为检测质量全面提高。

| 数据集 | V4-HG AUC-ROC | V4-HG AUC-PR | V4 AUC-ROC / AUC-PR | V3 严格证据 AUC-ROC / AUC-PR |
|---|---:|---:|---:|---:|
| Exathlon | 0.827546 | 0.426345 | 0.827541 / 0.426263 | 0.8749 / 0.6259 |
| PSM | 0.655103 | 0.427682 | 0.655103 / 0.427663 | 0.7269 / 0.5127 |
| SMD | 0.681471 | 0.117172 | 0.681471 / 0.117130 | 0.7688 / 0.2039 |
| SWaT | 0.850022 | 0.529264 | 0.849935 / 0.532545 | 0.8528 / 0.7456 |

解释：创新分数容易在异常段内产生至少一个高峰，PA 会把一次命中扩展到整个异常段；
事件内其他点和正常点的整体排序并没有同步改善。论文主证据应优先采用严格协议下的
AUC-PR、AUC-ROC 和无 PA 指标，PA 两表作为文献兼容结果。

### 4.3.1 无 PA、事件级与效率补充

统一评估器在完全相同的 12 份冻结产物上计算转导式 POT 裸 F1、重叠 Event-F1
和无 PA 的 test-label oracle Best-F1。这里的 SPOT 由正常训练分数初始化，但归档实现
会接收无标签测试分数流；Event-F1 把实体内连续异常段视为事件，只判断预测段与真实段
是否重叠，不执行 point adjustment。

| 数据集 | POT 裸 F1 | POT Event-F1 | Best 裸 F1（oracle） |
|---|---:|---:|---:|
| Exathlon | 0.1028 | 0.6835 | 0.5438 |
| PSM | 0.1286 | 0.6801 | 0.4735 |
| SMD | 0.1242 | 0.2927 | 0.1677 |
| SWaT | 0.0011 | 0.3121 | 0.7356 |
| Macro | 0.0892 | 0.4921 | 0.4802 |

这些结果量化了 PA 膨胀：POT+PA Macro 0.9023 与同一阈值的裸 Macro-F1 0.0892 相差
很大。它们不是可以删掉的不利数字，而是论文评估设计必须解释的核心限制。

RTX A6000 上完成 30 次预热、100 次测量的模型前向基准；生产 batch 的延迟为
3.150–7.027 ms/batch，约 24.61–70.83 µs/window，峰值已分配显存 62.7–198.1 MiB。
该范围包含动态 incidence 返回，不含数据加载和闭式 VAR 拟合。逐数据集表、设备和
PyTorch 版本见 `analysis/v4_hgat_paper/EFFICIENCY.md`。

### 4.4 MSL/SMAP 预注册外部扩展：负结果

在读取 MSL/SMAP 标签前冻结纯 V4 配置；随后从旧项目归档恢复六个数组，并验证转换后
`.npy` 的 SHA-256 与 THUML `Time-Series-Library` 官方页面逐个完全一致。冻结主结果为：

| 数据集 | AUC-ROC | AUC-PR | 异常比例 | Train-p99 裸 F1 | POT+PA | Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|
| MSL | 0.5024 | 0.1177 | 0.1054 | 0.0324 | 0.8857 | 0.8863 |
| SMAP | 0.5235 | 0.1245 | 0.1279 | 0.0110 | 0.7697 | 0.7705 |

完整性门槛通过，但 SMAP AUC-PR 低于异常比例，预注册排序门槛失败；两集的 oracle
Best-F1+PA 也分别低于 MtsCID 报告的 0.9513/0.9732。因此这是明确的**负确认**：
纯 V4 创新分数不能从原四集结论外推成当前领域 SOTA，MSL/SMAP 测试标签也不能再用于
回调阈值或结构。

NASA 数组实际由多实体拼接。主结果完成后又进行了非确认性的实体边界敏感性分析：
MSL/SMAP 的 ROC 为 0.5012/0.5259、AP 为 0.1157/0.1248、POT+PA 为
0.8823/0.6987、Best-F1+PA 为 0.8839/0.7680，负结论不变。论文中应将此作为失败的
外部泛化实验，而不是隐藏或用新的后处理覆盖。

### 4.5 V5 实体感知神经分支诊断：停止当前配方

在揭盲后的 MSL/SMAP 上进行了一次**开发诊断**，不能视为新的独立确认。代码现已按
NASA 官方 27/53 个实体分别做前 80% 训练、后 20% 验证；标准化、窗口、滞后对、评分和
PA 均不跨实体。训练前冻结 seed 87、三轮训练和两臂对照：`none` 为 KAN-TCN + 直连，
`lite` 仅增加 rank-16 HGAT-Lite。神经 `model_score` 结果如下：

| 数据集 | 空间分支 | AUC-ROC | AUC-PR | 异常比例 | Train-p99 裸 F1 | Best 裸 F1 | Best-F1+PA |
|---|---|---:|---:|---:|---:|---:|---:|
| MSL | **无空间** | **0.6216** | **0.1544** | 0.1032 | **0.1089** | **0.2325** | 0.7832 |
| MSL | HGAT-Lite | 0.5492 | 0.1299 | 0.1032 | 0.0578 | 0.1986 | **0.8033** |
| SMAP | **无空间** | **0.4816** | **0.1202** | 0.1277 | 0.0117 | **0.2296** | 0.6874 |
| SMAP | HGAT-Lite | 0.3872 | 0.1083 | 0.1277 | **0.0272** | 0.2262 | **0.6907** |

HGAT-Lite 相对无空间分支的 ROC/AP 在 MSL 下降 0.0725/0.0245，在 SMAP 下降
0.0944/0.0119。两臂在 SMAP 的 AP 都低于异常比例，且神经分支没有在两集同时超过
纯 innovation；因此两条预注册门槛均失败，**不运行 seeds 90/98，不采纳该 V5 配方**。

兼容融合的高 PA 数字也不能改变判定：MSL 的融合 Best-F1+PA 为 0.8839/0.8848
（none/lite），SMAP 为 0.7689/0.7691；它们几乎等于纯 innovation 的 0.8847/0.7680，
而融合 ROC/AP 仍只有约 0.501/0.115 和 0.526/0.125。也就是说，高分来自 PA 对稀疏
innovation 峰值的扩展，不是神经表征或 HGAT-Lite 的提升。

### 4.6 V6/V6b 目标预测诊断：因果修正与多种子停止决定

数据语义核对发现，Telemanom 版 MSL/SMAP 每个实体数组的第 0 列是要预测的遥测值，
其余 54/24 列主要是 one-hot 命令上下文；把所有列当作同质传感器共同重构并不合理。
因此新增三种训练目标：全量重构、目标列重构、目标列一步预测。目标预测只解码、训练、
扰动和评分第 0 列，当前目标值替换为一步滞后值，当前命令上下文仍可见；每个窗口首点
固定为无效分数，所有滞后对、窗口和 PA 均不跨实体。

初版 V6 使用 KAN-TCN + KANAD。运行后代码审计发现 KAN-TCN 的 InstanceNorm 和
KANAD 的 `Linear(W,W)` 会读取整窗，因此该实验只能算“目标遮蔽诊断”，不能称为
因果预测。修正后的 V6b 使用：

```text
输入 [B,W,F]
  → 左填充 causal Conv1d
  → 标准 dilated causal TCN
  → 拼接 TCN 表征与前端直连
  → 每个时刻独立的 LayerNorm(feature)-MLP
  → 仅输出目标列 [B,W,1]
  → 目标一步预测平方误差
```

自动“未来扰动不改变过去输出”测试在训练前通过。V6b 不使用 HGAT、KAN、KANAD、
RevIN、双视角、证据头或 innovation 融合。MSL/SMAP 总参数为 134,231/74,411，较
V5 无空间对照减少 11.8%/15.9%；解码器仅 3,805/1,765 参数，减少 77.3%/85.2%。

seed 87 的冻结两臂开发诊断如下。这里的 `train-p99+PA` 是训练分位数阈值，**不是
POT**；`Best+PA` 是读取测试标签的 oracle 上限。

| 数据集 | 目标 | ROC-AUC | PR-AUC | Train-p99 裸 F1 | Train-p99+PA | Best 裸 F1 | Best+PA |
|---|---|---:|---:|---:|---:|---:|---:|
| MSL | 重构 | 0.6263 | 0.2102 | 0.1843 | 0.8483 | 0.2559 | 0.8886 |
| MSL | **一步预测** | 0.6258 | 0.2098 | 0.1356 | **0.9033** | 0.2543 | **0.9083** |
| SMAP | 重构 | 0.5230 | **0.1378** | **0.0397** | **0.8399** | 0.2365 | **0.8532** |
| SMAP | **一步预测** | **0.5436** | 0.1343 | 0.0304 | 0.8255 | **0.2396** | 0.8445 |

seed-87 延续门槛通过后，按事先冻结的计划完整运行 seed 90/98；没有丢弃或替换不利
种子。一步预测三种子结果为：

| 数据集 | ROC-AUC | PR-AUC | Train-p99 裸 F1 | Train-p99+PA | Best 裸 F1 | Best+PA |
|---|---:|---:|---:|---:|---:|---:|
| MSL | 0.6508±0.0210 | 0.2186±0.0092 | 0.1369±0.0060 | 0.8875±0.0129 | 0.2684±0.0147 | 0.9080±0.0002 |
| SMAP | 0.5191±0.0487 | 0.1324±0.0097 | 0.0272±0.0023 | 0.7902±0.0378 | 0.2271±0.0196 | 0.8456±0.0009 |

MSL 的三个种子均高于 V5 AP 参考，且均值明显提高；但 SMAP seed 98 只有
ROC/AP=0.4512/0.1197，低于冻结的 ROC>0.5 与 V5 AP=0.1202 门槛，SMAP ROC 标准差
0.0487 也超过 0.03。故稳定性判定为**失败并停止该路线**。文献上下文目标同样没有
达到：MSL 参考 0.7314/0.2186/0.9513，SMAP 参考 0.6224/0.1912/0.9732
（ROC/AP/Best+PA）。这些是揭盲后的开发结果，不能改写为独立确认或 SOTA。

### 4.7 V7 因果先验有向超图：PSM 局部通过、跨数据集失败

V7 的目标不是把 HGAT 挂在 V4 的纯创新分数旁边，而是让同一个关系结构同时影响预测和
异常分数。完整数据流为：

```text
正常训练序列
  → signed ridge causal prior（每个预测目标对应一条有向超边）
输入窗口（当前目标替换为一步滞后目标，当前外生上下文保留）
  ├→ standard dilated causal TCN ───────────────────────────┐
  └→ [当前值, 一阶差分, 特征身份] → 低秩动态 incidence       │
       + log-prior bias → Top-K → signed node→edge 聚合 ─────┤
  → 逐时刻解码器([滞后目标, TCN 表征, 关系表征]) → 目标预测 ──┘
  → [预测误差, 动态 incidence 与正常先验的 JS 偏离]
  → 仅用正常训练 ECDF 分别校准 → 固定权重 0.75/0.25 融合
```

所有操作严格按时间因果：不使用整窗 InstanceNorm、`Linear(W,W)`、双向递归、RevIN 或
测试时序归一化；实体数据上的先验拟合、窗口、校准和 PA 均不得跨实体。MSL/SMAP/SMD
还支持缩放后附加实体 one-hot 上下文节点，它们只提供上下文，不作为预测目标。

V7-full 在 PSM 上的三种子冻结结果如下。`±` 是 seeds 87/90/98 的总体标准差；主判据是
无 PA ROC-AUC/PR-AUC，PA 两列仅作描述。

| 配置 | ROC-AUC | PR-AUC | Train-p99 裸 F1 | Train-p99+PA | Best 裸 F1 | Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|
| **V7 full** | **0.6568±0.0255** | **0.4448±0.0323** | **0.0693±0.0183** | 0.9673±0.0099 | **0.4703±0.0149** | 0.9804±0.0017 |
| 无因果先验 | 0.5965±0.0107 | 0.3638±0.0048 | 0.0328±0.0052 | 0.9474±0.0144 | 0.4446±0.0059 | 0.9713±0.0096 |
| 纯时序 | 0.6071±0.0237 | 0.3794±0.0138 | 0.0511±0.0033 | 0.9756±0.0070 | 0.4496±0.0139 | 0.9835±0.0008 |
| 无关系偏离分数¹ | 0.6208±0.0274 | 0.3947±0.0240 | 0.0554±0.0065 | **0.9840±0.0010** | 0.4549±0.0153 | **0.9850±0.0004** |

¹ 直接从 full 产物把关系权重置零复算，是不重复训练的精确 scoring ablation；seed 87 与
单独运行的对应臂逐数组完全一致。

| Full 相对对照 | Δ ROC-AUC | Δ PR-AUC | 配对胜出 |
|---|---:|---:|---:|
| 纯时序 | +0.0497 | +0.0654 | 两指标均 3/3 |
| 无因果先验 | +0.0603 | +0.0810 | 两指标均 3/3 |
| 无关系偏离分数 | +0.0360 | +0.0501 | 两指标均 3/3 |

七项冻结确认门槛全部通过，说明因果先验、超图预测路径和关系评分在 PSM 上都有一致的
无 PA 排序贡献。full 共 70,200 个可训练参数；纯时序有效路径 69,271 个，超图机制仅增加
929 个参数。但 V7 仍低于 HGAT-Lite 的 PSM 三种子 ROC/AP 0.7466/0.5547，而且 PSM
标签早已揭盲；因此当前结论是**机制成立但性能未达到论文定稿线**。PA 指标还出现纯时序
或无关系分数更高的排序反转，恰好再次证明不能用 PA 表为结构贡献背书。

随后按 `preregistered_cross_dataset_seed87.md` 保持架构和超参完全不变，在 EXATHLON、
SMD、SWAT、MSL、SMAP 上运行 full、无先验和纯时序三臂。15/15 产物均完整、有限且长度
正确。下表的 prediction-only 是直接从同一个 full 产物移除关系分数复算，不重复训练：

| 数据集 | Full ROC | Full AP | Prediction-only ROC | Prediction-only AP | Full train-p99+PA | Full Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|
| EXATHLON | 0.8576 | 0.4716 | **0.8633** | **0.4959** | 0.0656 | 0.9490 |
| SMD | 0.7172 | 0.1326 | **0.7179** | **0.1684** | 0.7156 | 0.7724 |
| SWAT | 0.6242 | 0.1492 | **0.7702** | **0.3137** | 0.4123 | 0.9222 |
| MSL | 0.4505 | 0.0897 | **0.5161** | **0.1182** | 0.4415 | 0.4919 |
| SMAP | 0.6376 | **0.3208** | **0.6687** | 0.2727 | 0.8545 | 0.8562 |

固定 0.25 关系偏离分数在 5/5 数据集降低至少一个主排序指标；超图机制门槛仅 EXATHLON
通过，因果先验门槛仅 EXATHLON/SMD/SMAP 通过。最终 0/5 数据集通过全部五项门槛，低于
预注册的 3/5 总体与 2/3 多目标要求，故 **不运行 seeds 90/98**。这不是实验缺失，而是
冻结停止决定。V7 当前规格的跨数据集机制主张被证伪；该失败模式直接促成下节 V7.1：
取消固定加性关系分数，并使图修正成为纯时序预测器的可退化残差分支。精确指标与哈希见
`analysis/v7_causal_hypergraph/cross_dataset_seed87.json`。

### 4.8 V7.1 可退化图残差：仅 MSL 通过，停止

针对 V7 的普遍失败模式，V7.1 在训练前冻结了两项结构修正：把预测写成
`lagged target + temporal residual + sigmoid(g) * graph residual`，并用可学习门在动态
incidence 与训练侧因果先验间插值。时序和图解码器完全分离，图门趋零时严格退化为同权重
纯时序模型；关系 JS 偏离继续转储供解释，但检测权重固定为 0。40 项回归测试中的嵌套性
测试已验证这一退化性质。

| 数据集 | Full ROC | Full AP | Train-p99+PA | Best-F1+PA | Δ vs temporal ROC/AP | Δ vs no-prior ROC/AP | 通过 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| EXATHLON | 0.8190 | 0.4243 | 0.9274 | 0.9540 | +0.0010 / +0.0024 | -0.0045 / -0.0042 | 否 |
| PSM | 0.5689 | 0.3559 | 0.9846 | 0.9850 | -0.0125 / -0.0128 | -0.0098 / -0.0077 | 否 |
| SMD | 0.7171 | 0.1624 | 0.8177 | 0.8781 | +0.0112 / +0.0028 | -0.0030 / -0.0011 | 否 |
| SWAT | 0.6335 | 0.1518 | 0.9407 | 0.9555 | +0.0665 / +0.0014 | -0.0761 / -0.0470 | 否 |
| MSL | **0.6307** | **0.1447** | 0.5960 | 0.6838 | +0.1422 / +0.0471 | +0.0594 / +0.0166 | **是** |
| SMAP | 0.6510 | 0.2866 | 0.7844 | 0.8602 | +0.1732 / +0.1251 | -0.0219 / -0.0490 | 否 |

18/18 产物完整且 learned gate 有效，但只有 MSL 通过全部门槛，低于冻结的 4/6 总体和
3/4 多目标继续条件，故不运行 seeds 90/98。图残差相对纯时序在若干数据集有增益，但
因果先验相对 no-prior 只在 MSL 全面胜出；这不足以支持 KBS 的跨数据集机制主张。精确
结果与哈希在 `analysis/v7_causal_hypergraph/v71_seed87.json`。由于六集标签均已揭盲，
继续在相同测试集迭代会变成 benchmark fitting，V7/V7.1 战役至此关闭。

## 5. 规模与精简

| 数据集 | V3 完整主干参数 | 去 HGAT 后参数 | V4 创新系数 | 主干参数降幅 |
|---|---:|---:|---:|---:|
| Exathlon | 82,715 | 79,677 | 361 | 3.7% |
| PSM | 86,665 | 83,135 | 625 | 4.1% |
| SMD | 116,286 | 111,284 | 1,444 | 4.3% |
| SWaT | 303,535 | 162,421 | 2,601 | 46.5% |

V4 的创新系数数量为 \(C^2\)，通过闭式 ridge 求解，不参与梯度训练。PA 表中的纯创新
检测不依赖 HGAT，因此可以把复杂空间分支留给解释性实验，而在轻量部署中移除。

V7 在 PSM 上共有 70,200 个可训练参数，其中超出纯时序有效路径的关系参数仅 929 个；
它比 V4 纯闭式检测器复杂，但把可学习的有向关系明确接入了预测与异常评分。

## 6. 代码与结果索引

| 内容 | 路径 |
|---|---|
| ISTAD 主代码 | `code/ISTAD/` |
| KBS 论文方法稿 | `paper/ISTAD_KBS_draft.md` |
| V4 创新分数实现 | `code/ISTAD/utils/innovation.py` |
| HGAT-Lite 实现 | `code/ISTAD/models/istad_layers/hypergraph_attention.py` |
| V8 KAN-HGAT 训练入口 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v8_kan_hgat.sh` |
| V8 冻结评估器 | `analysis/evaluate_v8_kan_hgat.py` |
| V8 预注册与结果 | `analysis/v8_kan_hgat/` |
| V9 BR-KAN 训练入口 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v9_brkan_hgat.sh` |
| V9 冻结评估器 | `analysis/evaluate_v9_brkan_hgat.py` |
| V9 预注册与结果 | `analysis/v9_brkan_hgat/` |
| V10 新息域 BR-KAN-HGAT 训练入口 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v10_innovation_brkan_hgat.sh` |
| V10 新息域 Linear-HGAT 消融入口 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v10_innovation_linear_hgat.sh` |
| V10 冻结评估器 | `analysis/evaluate_v10_innovation_brkan_hgat.py` |
| V10 投影消融评估器 | `analysis/evaluate_v10_projection_ablation.py` |
| V10 预注册与结果 | `analysis/v10_innovation_brkan_hgat/` |
| KBS 投稿代码与结果冻结清单 | `analysis/KBS_FREEZE.sha256` |
| 可解释性包预注册协议（2026-09-15） | `analysis/v4hg_interpretability/PREREGISTERED_PROTOCOL.md` |
| 可解释性包评估器（案例/定位/稳定性） | `analysis/evaluate_v4hg_interpretability.py` |
| 可解释性包结果与判定 | `analysis/v4hg_interpretability/RESULTS.md` |
| 模型评估接入 | `code/ISTAD/exp/exp_anomaly_detection.py` |
| V4 两张 PA 表复算 | `analysis/evaluate_istad_innovation_pa.py` |
| V4 精简训练脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v4_compact.sh` |
| HGAT-Lite 消融脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v4_lite_hgat.sh` |
| HGAT 三臂确认脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v4_hgat_ablation_psm.sh` |
| HGAT 神经分支复算 | `analysis/evaluate_hgat_ablation.py` |
| HGAT-Lite 三种子统一复算 | `analysis/evaluate_hgat_lite_multiseed.py` |
| HGAT 确认结果 | `analysis/istad_v4_lite_hgat/PSM_confirmation_s90_s98.json` |
| HGAT-Lite 四数据集结果 | `analysis/istad_v4_lite_hgat/all_datasets_s87.json` |
| HGAT-Lite 训练-ECDF 候选 | `analysis/istad_v4_lite_hgat/all_datasets_s87_train_ecdf.json` |
| HGAT-Lite 三种子确认结果 | `analysis/istad_v4_lite_hgat/multiseed_confirmation_s87_s90_s98.json` |
| 2025–2026 最新基线与协议审计 | `analysis/latest_baseline_audit_2026-09-10.md` |
| MSL/SMAP 预注册 | `analysis/independent_confirmation/preregistered_msl_smap.md` |
| MSL/SMAP 官方数组哈希清单 | `analysis/independent_confirmation/dataset_conversion_manifest.json` |
| MSL/SMAP 冻结主结果 | `analysis/independent_confirmation/msl_smap_results.json` |
| NASA 实体边界敏感性结果 | `analysis/independent_confirmation/msl_smap_entity_sensitivity.json` |
| V5 NASA 开发诊断预注册及结论 | `analysis/v5_nasa_diagnostic/preregistered_seed87.md` |
| V5 NASA seed-87 完整结果 | `analysis/v5_nasa_diagnostic/seed87.json` |
| V6 目标遮蔽诊断及因果审计 | `analysis/v6_nasa_target/preregistered_seed87.md` |
| V6 目标遮蔽完整结果 | `analysis/v6_nasa_target/seed87.json` |
| V6b 严格因果预注册及 seed-87 结论 | `analysis/v6b_nasa_causal/preregistered_seed87.md` |
| V6b 多种子冻结计划及停止决定 | `analysis/v6b_nasa_causal/confirmation_plan_s90_s98.md` |
| V6b 多种子完整结果 | `analysis/v6b_nasa_causal/multiseed_s87_s90_s98.json` |
| V7 冻结架构合同 | `analysis/v7_causal_hypergraph/preregistered_design.md` |
| V7 seed-87 四臂预注册/结果 | `analysis/v7_causal_hypergraph/preregistered_psm_seed87.md` / `psm_seed87.json` |
| V7 多种子确认计划/结果 | `analysis/v7_causal_hypergraph/confirmation_plan_psm_s90_s98.md` / `psm_multiseed_confirmation.json` |
| V7 五数据集冻结计划/停止结果 | `analysis/v7_causal_hypergraph/preregistered_cross_dataset_seed87.md` / `cross_dataset_seed87.json` |
| V7.1 残差修正计划/停止结果 | `analysis/v7_causal_hypergraph/preregistered_v71_residual_seed87.md` / `v71_seed87.json` |
| V7 结果摘要 | `analysis/v7_causal_hypergraph/RESULTS.md` |
| V7 因果先验/超图实现 | `code/ISTAD/utils/causal_prior.py` / `models/istad_layers/causal_hypergraph.py` |
| V7 训练脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh` |
| V7.1 训练脚本/评测器 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v71_residual_hypergraph.sh` / `analysis/evaluate_v71_seed87.py` |
| 本地归档转官方 NumPy 工具 | `analysis/convert_archived_msl_smap.py` |
| MSL/SMAP 冻结评测器 | `analysis/evaluate_independent_confirmation.py` |
| NASA 实体边界敏感性评测器 | `analysis/evaluate_nasa_entity_sensitivity.py` |
| V5 NASA 训练脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v5_nasa_diagnostic.sh` |
| V5 NASA 统一评测器 | `analysis/evaluate_v5_nasa_diagnostic.py` |
| V6b 严格因果训练脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh` |
| V6b seed-87 / 多种子评测器 | `analysis/evaluate_v6b_nasa_causal.py` / `analysis/evaluate_v6b_nasa_multiseed.py` |
| V3 严格协议脚本 | `code/ISTAD/scripts/anomaly_detection/ISTAD_v3_strict.sh` |
| V4 机器可读结果 | `analysis/istad_v4_innovation/pa_results.json` |
| V4 分数归档 | `analysis/istad_v4_innovation/*_innovation_scores.npz` |
| 统一基线表 | `analysis/unified_tables.json` |
| TSLib 基线 | `code/Time-Series-Library/` |
| Classic 基线 | `code/TranAD_improve/` |

## 7. 复现

### 7.1 环境

建议使用 Python 3.11、PyTorch 2.5、NumPy、pandas 和 scikit-learn。安装主模型依赖：

```bash
cd /data/modeluse/TS/ISTAD-release
pip install -r code/ISTAD/requirements.txt
```

### 7.2 零 GPU 复算 V4 两张 PA 表

默认数据位置是 `/data/modeluse/TS/ISTAD/dataset`，也可以显式指定：

```bash
cd /data/modeluse/TS/ISTAD-release
python analysis/evaluate_istad_innovation_pa.py \
  --data-root /data/modeluse/TS/ISTAD/dataset \
  --pot-tail-margin 1.04
```

输出会写入 `analysis/istad_v4_innovation/pa_results.json` 和四个压缩分数文件。

### 7.3 复算 HGAT-Lite 三种子确认

已有 12 个训练分数文件时，可在 CPU 上统一复算 POT、排序指标、固定阈值裸 F1 和门槛：

```bash
cd /data/modeluse/TS/ISTAD-release
python analysis/evaluate_hgat_lite_multiseed.py \
  --output analysis/istad_v4_lite_hgat/multiseed_confirmation_s87_s90_s98.json
```

### 7.4 复算 MSL/SMAP 负确认

MSL/SMAP 负确认可由本地官方哈希匹配归档复现，无需联网：

```bash
cd /data/modeluse/TS/ISTAD-release
python analysis/convert_archived_msl_smap.py
python analysis/evaluate_independent_confirmation.py
python analysis/evaluate_nasa_entity_sensitivity.py
```

### 7.5 复算 V5 NASA 开发诊断

四个训练产物已存在时，以下命令只做 CPU 统一评估并重写机器可读结论：

```bash
cd /data/modeluse/TS/ISTAD-release
python analysis/evaluate_v5_nasa_diagnostic.py
```

如需从头训练，按预注册文件中的四条命令运行。MSL/SMAP 已经揭盲，这组实验只能作为
失败机理诊断，不能通过重复种子或挑选分支升级为独立测试结论。

### 7.6 复算 V6b NASA 严格因果诊断

已有六个训练分数文件时，以下命令会重算完整指标、哈希和冻结门槛：

```bash
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v6b_nasa_causal.py
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v6b_nasa_multiseed.py
```

从头训练的命令见两个预注册文件。该分支没有运行 POT；文档中的 train-p99+PA 不得
改名为 POT+PA。

### 7.7 复现 V7 PSM 因果超图确认

以下命令使用已有产物重算全部逐种子指标、哈希、配对差值和冻结门槛：

```bash
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v7_psm_seed87.py
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v7_psm_multiseed.py
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v7_cross_dataset_seed87.py
```

从头训练命令见 `analysis/v7_causal_hypergraph/confirmation_plan_psm_s90_s98.md`。CPU
合成烟雾测试为 `python analysis/smoke_v7_causal_hypergraph.py`。PSM 已经揭盲，这些运行
只能验证代码和机制稳定性，不能包装成独立测试集确认。

### 7.8 复算 V7.1 六数据集停止结果

```bash
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v71_seed87.py
```

从头训练命令及三臂定义见 `preregistered_v71_residual_seed87.md` 和
`ISTAD_v71_residual_hypergraph.sh`。六个数据集均已揭盲，该结果只属于开发诊断。

### 7.9 复现 V4-HG rank-safe

从头训练或使用同设置 checkpoint 评估：

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
bash scripts/anomaly_detection/ISTAD_v4_hgat_integrated.sh PSM 0 87 train
bash scripts/anomaly_detection/ISTAD_v4_hgat_integrated.sh SMD 0 87 eval
```

数据集参数可换成 `EXATHLON / PSM / SMD / SWAT`，种子使用 `87 / 90 / 98`。从全部
已有产物重算 POT、读取 Best-F1、核查排序不反转并生成哈希：

```bash
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v4_hgat_integrated.py
```

完整重导出 float64 分数、无 PA/事件级/随机并列对照和效率表：

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
bash scripts/anomaly_detection/ISTAD_v4_hgat_paper_eval.sh 0
```

若已有 12 份 float64 产物，可从仓库根目录只生成指标：

```bash
python analysis/evaluate_v4_hgat_paper.py --random-tie-repeats 100
python analysis/benchmark_v4_hgat_efficiency.py --device cuda:0
```

协议和失败的 0.20 融合均保存在 `analysis/v4_hgat_integrated/`，不得删除负结果。

V8 负消融可用同一接口复现；`train` 可改为 `eval` 读取已有 checkpoint：

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
bash scripts/anomaly_detection/ISTAD_v8_kan_hgat.sh PSM 0 87 train
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v8_kan_hgat.py
```

该筛查已经揭盲且未晋级，不应继续补跑种子或据此调参。

V9 使用同一命令格式，将脚本名换为 `ISTAD_v9_brkan_hgat.sh`；结果通过
`python analysis/evaluate_v9_brkan_hgat.py` 复算。

V10 seed-87 冻结筛查可复算为：

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
bash scripts/anomaly_detection/ISTAD_v10_innovation_brkan_hgat.sh PSM 0 87 train
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v10_innovation_brkan_hgat.py
```

四集筛查已经完成且整体继续门失败，不应补跑 seeds 90/98。

V10 的 Linear/BR-KAN 等条件投影消融可复算为：

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
bash scripts/anomaly_detection/ISTAD_v10_innovation_linear_hgat.sh PSM 0 87 train
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD:analysis python analysis/evaluate_v10_projection_ablation.py
```

四个数据集均已运行；上面以 PSM 展示命令格式。该消融支持保留 BR-KAN，但属于 V10
揭盲后的机制解释，不改变“不补跑 seeds 90/98、不晋级主模型”的决定。

### 7.10 测试

```bash
cd /data/modeluse/TS/ISTAD-release
PYTHONPATH=code/ISTAD python -m unittest discover \
  -s code/ISTAD/tests -p 'test_*.py'
```

当前回归结果为 50 项测试全部通过，包括旧 checkpoint 严格加载、training-only ECDF、
MSL/SMAP 冻结评测器的合成端到端验证、MSL/SMAP 实体隔离窗口测试，以及 HGAT-Lite
的形状、稀疏覆盖、归一化、梯度和参数量测试；还覆盖目标列输出、一步滞后输入及
未来扰动不改变过去输出的严格因果不变性。V7 新增覆盖先验恢复与实体隔离、实体级
ECDF、预测与 incidence 的未来不变性、动态关系梯度、checkpoint buffer 和实体上下文；
V7.1 还覆盖图门趋零时严格退化为同权重纯时序模型，以及 graph/prior 门的有限梯度；
V4-HG 新增覆盖逐变量证据精确还原 V4、超边池化、正常尾 kill-switch、动态 incidence
导出和保序修正不反转不同 V4 排名；V8 新增覆盖共享 KAN 投影的参数量、梯度、形状和
incidence 归一化；V9 还验证 BR-KAN 与线性投影的精确初始嵌套及样条梯度；V10 新增验证
有符号新息方向、训练尺度与实体边界；事件评估还
覆盖实体边界和预测碎片惩罚。

### 7.11 严格协议训练示例

从 `code/ISTAD` 运行：

```bash
bash scripts/anomaly_detection/ISTAD_v3_strict.sh SMD 0 48
bash scripts/anomaly_detection/ISTAD_v3_strict.sh PSM 1 87
bash scripts/anomaly_detection/ISTAD_v3_strict.sh SWAT 2 48
bash scripts/anomaly_detection/ISTAD_v3_strict.sh EXATHLON 0 89
```

训练/验证必须按时间分离，标准化器只拟合训练段，SMD 窗口不得跨实体边界。所有方法应
使用相同的切分、窗口、评分和 PA 实现。

## 8. 投稿前检查清单

1. 将 V4-HG 保留为已经冻结的四数据集探索架构，不再用 EXA/PSM/SMD/SWaT 或已经揭盲的
   MSL/SMAP 测试标签继续调阈值；
2. MSL/SMAP 负确认、V5 诊断和 V6b 多种子停止结果共同表明：不得继续在两集上调整
   PA/POT、窗口、融合权重或筛选种子。V7 已实现实体上下文与实体级校准；
3. V7 的五数据集冻结筛查和 V7.1 六数据集筛查均已失败，不得丢弃不利数据集或补跑种子
   包装成确认，也不得继续用这六集测试标签设计 V7.2；
4. V4-HG 的 AUC-PR、AUC-ROC、无 PA 裸 F1、事件级指标和效率已经生成，论文必须与
   PA-F1 同时报告，不能只展示有利的 PA 表；
5. Best-F1+PA 必须明确标注为测试标签 oracle 上限，不得描述成可部署性能；
6. 当前不得写“达到 SOTA”；可以写“在存档 11 基线和探索协议下取得四列名义第一”，
   并把 MSL/SMAP 负确认及当前基线差距一并报告。
7. V8/V9 KAN-HGAT 的 seed-87 冻结筛查均已失败；不得在同四个已揭盲测试集上继续调
   grid、order、幅度、初始化或融合权重后再包装为确认实验。
8. V10 虽在四集取得一致 AP/ROC 改善和更高 Best-F1+PA Macro，但 POT+PA 护栏失败；应
   保留为正向机制消融，不得补跑种子或根据这些测试标签调出一个事后 V10.1。等条件
   投影消融支持 BR-KAN 相对 Linear 的 Macro AP 增益，但它是事后解释，不重开整体门。
9. 投稿范围固定为现有数据集和存档基线，不新增 MtsCID、MSTDF-AD 等外部复现；代价是
   所有领先表述必须限定在本文存档基线与实验口径内。后续工作转入框架图和论文写作。

## 9. 当前结论

ISTAD V4-HG 已在项目当前存档表中保持两种 PA 协议的 4/4 名义第一，并用保序细化把
HGAT-Lite 真实接入 V4；三种子结果与 V4 完全一致，网络显著精简，排序指标只获得极小
ROC 正向并列细化，但 SWaT AUC-PR 回退 0.003281。不能把它写成显著或全面精度提升。
早期 HGAT-Lite + train-ECDF 和
0.20 关系融合均未通过门槛，继续作为负结果。PA 优势与无 PA 排序质量并不等价。最稳妥的论文叙事
是：HGAT-Lite 提供低秩动态关系建模和参数效率，V4 因果创新分支提供轻量检测与 PA
兼容；融合结果作为支持性实验，不把高 PA-F1 错归因于神经空间分支。最新基线审计已经
否决当前领域 SOTA 表述：WWW 2025 MtsCID 在 SMD、PSM、SWaT 报告的 PA-F1 均高于本项目
Best-F1+PA；预注册的 MSL/SMAP 也在排序指标和两种 PA 指标上给出负确认。实体边界
敏感性没有逆转结论，随后实体感知神经训练也证明 HGAT-Lite 在 NASA 两集降低 ROC/AP，
因此该 V5 配方已经停止。V6b 又修正了 NASA 的目标列语义和网络因果性，在 MSL 上取得
更好的三种子 ROC/AP，但 SMAP seed 98 使预注册稳定性门槛失败；V6b 同样停止，且不
替换 V4。V7 已进一步将训练侧有向因果先验、动态超图预测和关系偏离评分统一起来；其
PSM 三种子消融是局部正例，但五数据集冻结筛查 0/5 通过，证明固定关系分数和当前耦合
方式不能泛化。V7.1 已采用可退化图残差并把关系偏离降为解释性输出，但仍仅 MSL 通过
（1/6，多目标 0/4），所以同样停止，不能作为 KBS 主模型。真正的下一步不是在这六集上
继续调 V7.2；V8 的共享 KAN 投影因 SWaT 大幅回退而停止，V9 虽通过嵌套残差修复稳定性，
仍未证明原始值域 AP 增益。V10 改在 VAR 新息域学习关系后首次得到四集一致 AP 增益；
等条件消融进一步支持 BR-KAN 对该增益有贡献，但 POT+PA 护栏失败，尚不足以晋升主模型。
项目已停止新增模型、基线和数据集，下一步是按冻结证据完成方法、实验、局限性和结论写作。
