"""HGST v2 统一样条算子

三个可复用组件（规格：ISTAD_HGST_模型与实验完整修改方案.md 第 3/4.2 节）：
- BSplineBasis               共享 Cox-de Boor B 样条基（网格 buffer + 平滑有界越界映射）
- SplineIncidenceGenerator   KAN 风格边函数 φ^S(u)=a·SiLU(u)+Σ_k c_k·B_k(u) 生成动态超边归属
- ConditionalSplineActivation 超图关系状态低秩调制的时间样条激活 φ^T

设计要点：
- 越界输入使用 grid_max·tanh(x/grid_max) 平滑有界映射，不用硬 clamp（避免边界梯度为 0）
- Cox-de Boor 分母 clamp_min(1e-8)；基计算强制 float32，完成后再转回输入 dtype
- 关系调制采用低秩 einsum（不显式构造 (B,C,L,K) 的 Δw 大张量）
- 不引入任何第三方 KAN 包
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineBasis(nn.Module):
    """Cox-de Boor B 样条基函数（共享工具）

    输入任意形状 (*,)，输出 (*, num_basis)，num_basis = grid_size + spline_order。
    网格为均匀扩展网格（两端各延 spline_order 个节点）。
    """

    def __init__(self, grid_size=5, spline_order=3, grid_range=(-4.0, 4.0)):
        super().__init__()
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.num_basis = self.grid_size + self.spline_order
        h = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1,
                            dtype=torch.float32) * h + grid_range[0]
        self.register_buffer('grid', grid)
        self.grid_max = float(grid_range[1])

    def forward(self, x):
        """x: 任意形状 → (..., num_basis)"""
        # 平滑有界映射：|x|>grid_max 的部分被平滑压回网格，near 0 为恒等
        x = self.grid_max * torch.tanh(x / self.grid_max)
        dtype = x.dtype
        bases = self.compute_basis(x.float().reshape(-1))
        bases = bases.reshape(*x.shape, self.num_basis)
        return bases.to(dtype)

    def compute_basis(self, x):
        """x: (P,) float32 → (P, num_basis)，Cox-de Boor 递归"""
        grid = self.grid
        x = x.unsqueeze(-1)  # (P, 1)

        # 0 阶：分段常数指示函数
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            n_basis = bases.shape[-1] - 1

            # 左项：(x - t_i) / (t_{i+k} - t_i) · B_{i,k-1}(x)
            left_num = x - grid[:n_basis]
            left_den = grid[k:k + n_basis] - grid[:n_basis]
            left = left_num / left_den.clamp(min=1e-8) * bases[..., :n_basis]

            # 右项：(t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) · B_{i+1,k-1}(x)
            right_num = grid[k + 1:k + 1 + n_basis] - x
            right_den = grid[k + 1:k + 1 + n_basis] - grid[1:1 + n_basis]
            right = right_num / right_den.clamp(min=1e-8) * bases[..., 1:1 + n_basis]

            bases = left + right

        return bases


class SplineIncidenceGenerator(nn.Module):
    """KAN 风格动态超边归属生成器（空间样条）

    对每个 (变量 i, 超边 m, 描述符 q) 使用独立可学习边函数：
        φ^S_{i,m,q}(u) = a_{i,m,q}·SiLU(u) + Σ_k c_{i,m,q,k}·B_k(u)
    归属 logit：
        ℓ_{b,t,i,m} = p_{i,m} + b_m + Σ_q φ^S_{i,m,q}(Z_{b,t,i,q})
    归属分布（按节点维 softmax，带温度 τ_H）：
        H_{b,t,:,m} = softmax_i(ℓ_{b,t,:,m} / τ_H)

    输入 descriptors: (B, L, N, Q_in)，默认 Q_in=2（当前值 + 因果差分）
    输出 incidence:   (B, L, N, M)
    """

    def __init__(self, n_features, n_hyperedges, q_in=2,
                 grid_size=5, spline_order=3, grid_range=(-4.0, 4.0),
                 temperature=1.0):
        super().__init__()
        self.n_features = n_features
        self.n_hyperedges = n_hyperedges
        self.q_in = q_in
        self.temperature = float(temperature)

        basis = BSplineBasis(grid_size, spline_order, grid_range)
        self.basis = basis
        K = basis.num_basis

        # 边函数参数（变量×超边×描述符独立）
        self.a_param = nn.Parameter(torch.zeros(n_features, n_hyperedges, q_in))
        self.c_param = nn.Parameter(torch.zeros(n_features, n_hyperedges, q_in, K))
        # 静态先验 p_{i,m} 与超边偏置 b_m
        self.prior = nn.Parameter(torch.zeros(n_features, n_hyperedges))
        self.edge_bias = nn.Parameter(torch.zeros(n_hyperedges))

        nn.init.normal_(self.a_param, std=0.1)
        nn.init.normal_(self.c_param, std=0.1)
        nn.init.normal_(self.prior, std=0.1)

    def forward(self, descriptors, return_logits=False):
        """descriptors: (B, L, N, Q_in) → incidence (B, L, N, M)"""
        B, L, N, Q = descriptors.shape
        if N != self.n_features or Q != self.q_in:
            raise ValueError(
                f"描述符形状不匹配: 期望 (B,L,{self.n_features},{self.q_in})，"
                f"得到 (B,L,{N},{Q})")

        x = descriptors.float().reshape(-1)                       # (B*L*N*Q,)
        bases = self.basis.compute_basis(x).reshape(B, L, N, Q, -1)  # (B,L,N,Q,K)

        # φ^S = a·SiLU(u) + Σ_k c_k·B_k(u)
        silu_part = F.silu(descriptors).unsqueeze(-2) * self.a_param  # (B,L,N,1,Q)×(N,M,Q)→(B,L,N,M,Q)
        spline_part = torch.einsum('blnqk,nmqk->blnm', bases, self.c_param)
        logits = silu_part.sum(dim=-1) + spline_part                 # 描述符维求和后 (B,L,N,M)
        logits = logits + self.prior.unsqueeze(0).unsqueeze(0) + self.edge_bias

        H = F.softmax(logits / self.temperature, dim=2)              # 节点维归一化
        if return_logits:
            return H, logits
        return H


class ConditionalSplineActivation(nn.Module):
    """超图关系条件化的时间样条激活（低秩调制）

    对时间路径预激活 u (B, C, L)（C = N·r_t，通道 c 属于变量 c // r_t），
    由关系状态 R (B, L, N, d_s) 生成低秩调制幅度并调制样条系数：

        A_{b,t,i,c',q} = tanh(g_θ(R_{b,t,i})_{c',q})        c' ∈ [0, r_t)
        Δw_{b,t,i,c',k} = η·Σ_q A_{b,t,i,c',q}·V_{i,c',q,k}
        φ^T_{b,t,i,c'}(u) = γ·SiLU(u) + Σ_k (w + Δw)·B_k(u)

    稳定性：g_θ 末层零初始化（初始 Δw≡0）、η 固定超参、γ 初始 1、基础系数小随机。
    η=0 或 relation_state=None 时退化为固定样条（数值等价）。
    """

    def __init__(self, n_features, temporal_width, relation_dim,
                 modulation_rank=2, modulation_scale=0.05,
                 grid_size=5, spline_order=3, grid_range=(-4.0, 4.0),
                 ablation="full"):
        super().__init__()
        self.n_features = n_features
        self.temporal_width = temporal_width
        C = n_features * temporal_width
        self.C = C
        self.relation_dim = relation_dim
        self.rank = int(modulation_rank)
        self.scale = float(modulation_scale)
        self.ablation = str(ablation)

        self.basis = BSplineBasis(grid_size, spline_order, grid_range)
        K = self.basis.num_basis

        # 基础样条系数 w(C,K)：小随机（std 0.05），初始行为≈SiLU
        self.base_weight = nn.Parameter(torch.randn(C, K) * 0.05)
        # SiLU 残差分支 γ：初始 1
        self.residual_weight = nn.Parameter(torch.ones(C))
        # 低秩方向 V(C,Q,K)
        self.rank_basis = nn.Parameter(torch.randn(C, self.rank, K) * 0.1)

        if self.ablation == "full":
            # 关系→调制幅度生成器 g_θ：d_s → (r_t·Q)，末层零初始化（初始 Δw≡0）
            self.coeff_gen = nn.Linear(relation_dim, temporal_width * self.rank)
            nn.init.zeros_(self.coeff_gen.weight)
            nn.init.zeros_(self.coeff_gen.bias)
        else:
            self.coeff_gen = None

        self.act = nn.SiLU()

    def forward(self, u, relation_state=None, return_delta_norm=False):
        """
        u:              (B, C, L) 时间预激活
        relation_state: (B, L, N, d_s) 或 None
        返回:           (B, C, L)（return_delta_norm=True 时附 |Δw| 均值标量）
        """
        B, C, L = u.shape
        if C != self.C:
            raise ValueError(f"通道数不匹配: 期望 {self.C}，得到 {C}")

        if self.ablation == "silu_temporal":
            out = self.act(u) * self.residual_weight.view(1, C, 1)
            return (out, u.new_zeros(())) if return_delta_norm else out

        # 样条基 (B,C,L,K)
        x = u.float().transpose(1, 2).reshape(-1)            # (B*L*C,)
        bases = self.basis.compute_basis(x).reshape(B, L, C, -1)

        fixed = torch.einsum('blck,ck->bcl', bases, self.base_weight.float())

        delta_norm = None
        if self.coeff_gen is not None and relation_state is not None and self.scale > 0:
            Bl, N = relation_state.shape[1], relation_state.shape[2]
            # A: (B,L,N,r_t,Q) → (B,L,C,Q)，强制 float32 与 bases 对齐（AMP 安全）
            a_amp = torch.tanh(self.coeff_gen(relation_state)).float()  # (B,L,N,r_t*Q)
            a_amp = a_amp.reshape(B, Bl, N, self.temporal_width, self.rank)
            a_amp = a_amp.reshape(B, Bl, C, self.rank)

            # basis_rank: (B,L,C,Q) = Σ_k bases·V
            br = torch.einsum('blck,cqk->blcq', bases, self.rank_basis.float())
            # Δw 生成的低秩路径：dynamic_{b,c,l} = η·Σ_q A·basis_rank（不显式构造 Δw 大张量）
            dynamic = self.scale * torch.einsum('blcq,blcq->bcl', a_amp, br)
            delta_norm = dynamic.detach().abs().mean()
            out = self.act(u) * self.residual_weight.view(1, C, 1) + fixed + dynamic
        else:
            out = self.act(u) * self.residual_weight.view(1, C, 1) + fixed
            if return_delta_norm:
                delta_norm = u.new_zeros(())

        out = out.to(u.dtype)
        if return_delta_norm:
            return out, delta_norm
        return out
