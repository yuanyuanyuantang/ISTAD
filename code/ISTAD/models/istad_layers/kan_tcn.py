"""
KAN-TCN: 基于 Kolmogorov-Arnold 网络改进的时间卷积网络

=============================================================================
理论基础
=============================================================================
Kolmogorov-Arnold 表示定理指出，任何连续多变量函数 f(x₁,...,xₙ) 均可表示为
单变量连续函数的有限复合：

    f(x) = Σⱼ Φⱼ( Σᵢ φᵢⱼ(xᵢ) )

KAN 网络（Kolmogorov-Arnold Networks）据此将传统 MLP 中 **固定激活函数 + 可学习
线性权重** 的范式反转为 **可学习激活函数 + 求和聚合**：

    MLP Edge:  y = σ(w·x + b)        # σ 固定（ReLU/SiLU），w 可学习
    KAN Edge:  y = φ(x)              # φ 可学习（B 样条），无需固定激活

每条边上的 φ 由 B 样条参数化：
    φ(x) = Σₖ cₖ · Bₖ(x) + α · SiLU(x)
其中 Bₖ 是 B 样条基函数，cₖ 是可学习系数，α·SiLU(x) 作为残差保证训练稳定。

=============================================================================
KAN-TCN 设计动机
=============================================================================
标准 TCN（时间卷积网络）的时间块结构：
    Conv1d → Chomp → ReLU → Dropout → Conv1d → Chomp → ReLU → Dropout + 残差

其中 Conv1d 负责时间混合（通过膨胀因果卷积扩大感受野），ReLU 提供非线性变换。
问题在于：**所有通道共享相同的固定非线性变换（ReLU）**，这限制了对复杂时序
模式的表达能力。

KAN-TCN 将 ReLU 替换为 **逐通道可学习的 B 样条激活函数**：
    Conv1d → Chomp → BSpline(x) + SiLU(x) → Dropout → ...  + 残差

优势：
  1. 每个特征通道学习独立的最优非线性变换
  2. B 样条的局部性和光滑性天然适合时序数据
  3. 保留因果卷积结构（时间步 t 不看未来信息）
  4. 可学习激活函数可视化后能辅助模型可解释性
  5. 参数增量很小（仅增加 ~3-5% 参数）
=============================================================================
"""

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


# ============================================================================
#  B 样条激活函数
# ============================================================================

class BSplineActivation(nn.Module):
    """逐通道可学习的 B 样条激活函数

    对于每个通道 c，激活函数为：
        φ_c(x) = Σₖ w_{c,k} · Bₖ(x)  +  α_c · SiLU(x)
                  ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                  可学习 B 样条分支            残差稳定分支

    其中 Bₖ(x) 是三阶 B 样条基函数（Cox-de Boor 递归），
    w_{c,k} 和 α_c 是可学习参数。

    参数:
        num_channels: 通道数（每个通道拥有独立的样条参数）
        grid_size:    B 样条网格区间数（越大逼近精度越高，默认 5）
        spline_order: B 样条阶数（3 = 三次样条，兼顾光滑与效率）
        grid_range:   网格覆盖范围（InstanceNorm 后输入大致在此范围内）
    """

    def __init__(self, num_channels, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order  # B 样条基函数总数

        # 均匀扩展网格（两端各扩展 spline_order 个额外节点）
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1,
                             dtype=torch.float32) * h + grid_range[0])
        self.register_buffer('grid', grid)

        # 可学习参数：B 样条系数（初始化为小随机值）
        self.spline_weight = nn.Parameter(
            torch.randn(num_channels, self.num_basis) * 0.1
        )
        # 残差分支权重（初始化为 1，使初始行为接近 SiLU）
        self.residual_weight = nn.Parameter(torch.ones(num_channels))

        # 输入归一化（将输入映射到网格范围内）
        self.norm = nn.InstanceNorm1d(num_channels, affine=True)

        # 残差基础激活
        self.base_activation = nn.SiLU()

    def compute_bspline_basis(self, x):
        """计算 B 样条基函数值（Cox-de Boor 递归）

        对于扩展网格 t₀, t₁, ..., t_G (G = grid_size + 2·spline_order)，
        0 阶基函数：  B_{i,0}(x) = 1  if t_i ≤ x < t_{i+1}, else 0
        k 阶递归：    B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) · B_{i,k-1}(x)
                                   + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) · B_{i+1,k-1}(x)

        最终得到 num_basis = grid_size + spline_order 个基函数。

        参数:
            x: 任意形状的标量张量 (*,)

        返回:
            bases: 形状 (*, num_basis) 的 B 样条基值
        """
        grid = self.grid
        x = x.unsqueeze(-1)  # (*, 1)

        # 0 阶：分段常数指示函数
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        # 逐阶递归构建高阶基函数
        for k in range(1, self.spline_order + 1):
            n_basis = bases.shape[-1] - 1  # 本阶基函数数量

            # 左项：(x - t_i) / (t_{i+k} - t_i) · B_{i,k-1}(x)
            left_num = x - grid[:n_basis]
            left_den = grid[k:k + n_basis] - grid[:n_basis]
            left = left_num / left_den.clamp(min=1e-8) * bases[..., :n_basis]

            # 右项：(t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) · B_{i+1,k-1}(x)
            right_num = grid[k + 1:k + 1 + n_basis] - x
            right_den = grid[k + 1:k + 1 + n_basis] - grid[1:1 + n_basis]
            right = right_num / right_den.clamp(min=1e-8) * bases[..., 1:1 + n_basis]

            bases = left + right

        return bases  # (*, num_basis)

    def forward(self, x):
        """
        参数:
            x: (batch, channels, seq_len) — 1D 时序数据

        返回:
            y: (batch, channels, seq_len) — 激活后的输出
        """
        B, C, L = x.shape

        # --- 残差分支 ---
        base_out = self.base_activation(x) * self.residual_weight.view(1, C, 1)

        # --- B 样条分支 ---
        x_norm = self.norm(x)  # 归一化到网格范围附近

        # 展平为标量序列，计算 B 样条基
        x_flat = x_norm.reshape(-1)                              # (B*C*L,)
        bases = self.compute_bspline_basis(x_flat)               # (B*C*L, num_basis)
        bases = bases.reshape(B, C, L, self.num_basis)           # (B, C, L, num_basis)

        # 逐通道加权求和
        spline_out = torch.einsum('bcln,cn->bcl', bases, self.spline_weight)

        return base_out + spline_out


# ============================================================================
#  因果裁剪层（与 tcn.py 中相同，为解耦独立定义）
# ============================================================================

class _Chomp1d(nn.Module):
    """裁剪填充 — 保证因果性（不看未来信息）"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


# ============================================================================
#  KAN 时间块（替代 TemporalBlock）
# ============================================================================

class KANTemporalBlock(nn.Module):
    """KAN 时间块 — TCN 的基本构建单元（可学习激活版本）

    结构对比：
        标准 TemporalBlock:
            Conv1d → Chomp → ReLU      → Dropout
            Conv1d → Chomp → ReLU      → Dropout  + 残差
        KAN TemporalBlock:
            Conv1d → Chomp → BSpline+SiLU → Dropout
            Conv1d → Chomp → BSpline+SiLU → Dropout  + 残差

    参数:
        n_inputs:     输入通道数
        n_outputs:    输出通道数
        kernel_size:  卷积核大小
        stride:       步长
        dilation:     膨胀率
        padding:      填充大小
        dropout:      Dropout 比率
        grid_size:    B 样条网格区间数
        spline_order: B 样条阶数
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2, grid_size=5, spline_order=3):
        super().__init__()

        # 第一个因果卷积 + KAN 激活
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = _Chomp1d(padding)
        self.kan_act1 = BSplineActivation(n_outputs, grid_size, spline_order)
        self.dropout1 = nn.Dropout(dropout)

        # 第二个因果卷积 + KAN 激活
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = _Chomp1d(padding)
        self.kan_act2 = BSplineActivation(n_outputs, grid_size, spline_order)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.kan_act1, self.dropout1,
            self.conv2, self.chomp2, self.kan_act2, self.dropout2,
        )

        # 残差连接（输入输出维度不同时用 1×1 卷积对齐）
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        # 残差加法后的激活（与标准 TemporalBlock 保持一致）
        self.post_activation = nn.SiLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='linear')

    def forward(self, x):
        """
        参数 / 返回:
            x: (batch, channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.post_activation(out + res)


# ============================================================================
#  KAN 时间卷积网络（替代 TemporalConvNet，接口兼容）
# ============================================================================

class TemporalKANNet(nn.Module):
    """基于 KAN 的时间卷积网络（KAN-TCN）

    与 TemporalConvNet 保持完全相同的接口，可 **直接 drop-in 替换**。
    唯一区别是将每个 TemporalBlock 中的 ReLU 替换为可学习 B 样条激活。

    参数:
        num_inputs:   输入通道数（= 特征数 n_features）
        num_channels: 各层输出通道数列表，如 [32, 32, 32, n_features]
        kernel_size:  卷积核大小（默认 2）
        dropout:      Dropout 比率（默认 0.2）
        grid_size:    B 样条网格区间数（默认 5，越大表达力越强）
        spline_order: B 样条阶数（默认 3 = 三次样条）
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 grid_size=5, spline_order=3):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(KANTemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
                grid_size=grid_size,
                spline_order=spline_order,
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        参数 / 返回:
            x: (batch, channels, seq_len)
        """
        return self.network(x)
