"""超图注意力层 - 用于捕获特征间的高阶空间依赖关系"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spline_ops import BSplineBasis


class SharedKANProjection(nn.Module):
    """Shared scalar KAN edge functions for the HGAT relation space."""

    def __init__(self, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.out_features = int(out_features)
        self.basis = BSplineBasis(grid_size=grid_size, spline_order=spline_order)
        self.spline_weight = nn.Parameter(torch.empty(
            self.out_features, self.basis.num_basis
        ))
        self.residual_weight = nn.Parameter(torch.ones(self.out_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        nn.init.normal_(self.spline_weight, std=0.02)

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError(f"expected (batch,nodes,1), got {tuple(x.shape)}")
        scalar = x.squeeze(-1)
        spline = torch.einsum(
            "bnk,rk->bnr", self.basis(scalar), self.spline_weight
        )
        residual = F.silu(scalar).unsqueeze(-1) * self.residual_weight
        return spline + residual + self.bias


class BoundedResidualKANProjection(nn.Module):
    """Linear relation projection plus a small, zero-start KAN correction."""

    def __init__(self, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.linear = nn.Linear(1, int(out_features), bias=False)
        self.basis = BSplineBasis(grid_size=grid_size, spline_order=spline_order)
        self.spline_weight = nn.Parameter(torch.zeros(
            int(out_features), self.basis.num_basis
        ))
        self.correction_scale = 0.1

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError(f"expected (batch,nodes,1), got {tuple(x.shape)}")
        spline = torch.einsum(
            "bnk,rk->bnr", self.basis(x.squeeze(-1)), self.spline_weight
        )
        return self.linear(x) + self.correction_scale * torch.tanh(spline)


class LightweightHypergraphMixer(nn.Module):
    """Low-rank dynamic hypergraph propagation for scalar sensor nodes.

    The legacy HGAT first expands every scalar sensor reading to a large hidden
    vector and then materializes a ``(B, N, M, 2D)`` attention tensor.  Here the
    graph is built directly in a small relation space and propagated with a
    normalized node->edge->node operator.  Sensor identity embeddings preserve
    variable semantics, while the projected current value makes the incidence
    sample-dependent.

    The layer accepts and returns ``(batch, nodes, 1)`` so it can replace the
    old project-in/HGAT/project-out stack without changing the reconstruction
    interface.  ``return_extras=True`` keeps the incidence matrices available
    for relation explanations.
    """

    def __init__(
        self,
        n_nodes,
        n_hyperedges,
        rank=16,
        projection_type="linear",
        kan_grid_size=5,
        kan_spline_order=3,
        use_topk=True,
        k_top=5,
        dropout=0.1,
        tau=1e-6,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_hyperedges = int(n_hyperedges)
        self.rank = int(rank)
        self.projection_type = str(projection_type).lower()
        self.use_topk = bool(use_topk)
        self.k_top = int(k_top)
        self.dropout = float(dropout)
        self.tau = float(tau)
        if self.n_nodes < 1 or self.n_hyperedges < 1:
            raise ValueError("n_nodes and n_hyperedges must be positive")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.projection_type not in {"linear", "kan", "brkan"}:
            raise ValueError("projection_type must be 'linear', 'kan', or 'brkan'")

        if self.projection_type == "kan":
            self.value_projection = SharedKANProjection(
                self.rank,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
            )
        elif self.projection_type == "brkan":
            self.value_projection = BoundedResidualKANProjection(
                self.rank,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
            )
        else:
            self.value_projection = nn.Linear(1, self.rank, bias=False)
        self.node_embedding = nn.Parameter(torch.empty(self.n_nodes, self.rank))
        self.edge_embedding = nn.Parameter(
            torch.empty(self.n_hyperedges, self.rank)
        )
        self.state_norm = nn.LayerNorm(self.rank)
        self.edge_norm = nn.LayerNorm(self.rank)
        self.output_projection = nn.Linear(self.rank, 1)
        # Start as a modest relational correction; the direct Conv branch is
        # concatenated separately by ISTAD and remains an unconditional path.
        self.message_gate_logit = nn.Parameter(torch.tensor(-1.0))

        if self.projection_type == "linear":
            nn.init.xavier_uniform_(self.value_projection.weight)
        elif self.projection_type == "brkan":
            nn.init.xavier_uniform_(self.value_projection.linear.weight)
        nn.init.normal_(self.node_embedding, std=0.02)
        nn.init.normal_(self.edge_embedding, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _apply_topk_sparsification(self, incidence):
        """Keep top-k nodes per edge and guarantee that no node is isolated."""
        k = max(1, min(self.k_top, self.n_nodes))
        indices = torch.topk(incidence, k=k, dim=1).indices
        mask = torch.zeros_like(incidence, dtype=torch.bool).scatter_(
            1, indices, True
        )
        uncovered = ~mask.any(dim=2)
        if uncovered.any():
            best_edge = incidence.argmax(dim=2, keepdim=True)
            fallback = torch.zeros_like(mask).scatter_(2, best_edge, True)
            mask = mask | (uncovered.unsqueeze(2) & fallback)

        sparse = incidence * mask.to(incidence.dtype)
        return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(self.tau)

    def forward(self, x, return_extras=False):
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, nodes, 1), got {x.shape}")
        batch_size, n_nodes, n_features = x.shape
        if n_nodes != self.n_nodes or n_features != 1:
            raise ValueError(
                f"expected (*, {self.n_nodes}, 1), got {tuple(x.shape)}"
            )

        state = self.value_projection(x)
        state = self.state_norm(state + self.node_embedding.unsqueeze(0))
        logits = torch.einsum(
            "bnr,mr->bnm", state, self.edge_embedding
        ) / math.sqrt(self.rank)

        # Each edge aggregates a normalized distribution over its member nodes.
        incidence = F.softmax(logits, dim=1)
        if self.use_topk:
            incidence = self._apply_topk_sparsification(incidence)

        edge_state = torch.einsum("bnm,bnr->bmr", incidence, state)
        edge_state = self.edge_norm(edge_state)

        # Normalize the same incidence in the reverse direction.  Reusing one
        # relation matrix is cheaper and easier to explain than a second dense
        # node-edge attention mechanism.
        reverse = incidence / incidence.sum(dim=2, keepdim=True).clamp_min(
            self.tau
        )
        message = torch.einsum("bnm,bmr->bnr", reverse, edge_state)
        message = F.dropout(message, self.dropout, training=self.training)
        delta = self.output_projection(F.gelu(message))
        output = torch.sigmoid(self.message_gate_logit) * delta

        if return_extras:
            return output, reverse, incidence
        return output


class HypergraphAttentionLayer(nn.Module):
    """超图注意力层，用于建模特征之间的空间依赖关系
    
    使用超边来捕获特征（节点）之间的高阶相关性。
    传统图只能建模两两关系，超图可以同时关联多个特征。
    包含 Top-K 稀疏化机制以提高可解释性。
    
    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度（嵌入维度）
        n_nodes: 节点数量（特征数量）
        n_hyperedges: 超边数量（潜在模式数）
        use_topk: 是否使用 Top-K 稀疏化
        k_top: 每个超边保留的 top 节点数
        dropout: Dropout 比率
        alpha: LeakyReLU 的负斜率
        tau: 数值稳定性参数
        h_param_init: 超图参数初始化方式 (normal, xavier, uniform)
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        n_nodes,
        n_hyperedges=21,
        use_topk=True,
        k_top=5,
        dropout=0.6,
        alpha=0.2,
        tau=1e-6,
        h_param_init="normal"
    ):
        super(HypergraphAttentionLayer, self).__init__()
        
        self.n_features = in_features
        self.embed_dim = out_features
        self.n_nodes = n_nodes
        self.n_hyperedges = n_hyperedges
        self.dropout = dropout
        self.alpha = alpha
        self.k_top = k_top
        self.tau = tau
        self.use_topk = use_topk

        # 节点特征变换
        self.node_transform = nn.Linear(in_features, out_features, bias=False)
        
        # 超边特征变换
        self.hyperedge_transform = nn.Linear(out_features, out_features, bias=False)

        # 注意力权重参数
        self.attention_weight = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.attention_weight.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)

        # 超边分布参数 H: (节点数, 超边数) — 作为可学习先验
        self.H_param = nn.Parameter(torch.empty(n_nodes, n_hyperedges))
        self._init_h_param(h_param_init)

        # 数据驱动的超图生成器：根据输入动态生成超图结构
        # H(X) = softmax(f(X) + H_param)，不同样本产生不同的超边激活模式
        self.h_generator = nn.Linear(in_features, n_hyperedges)
        # 初始化为近零，使初期行为接近静态先验，随训练逐渐学习动态结构
        nn.init.normal_(self.h_generator.weight, std=0.01)
        nn.init.zeros_(self.h_generator.bias)
    
    def _init_h_param(self, init_method):
        """初始化超图参数 H_param"""
        if init_method == "normal":
            nn.init.normal_(self.H_param, mean=0.0, std=0.1)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.H_param, gain=1.0)
        elif init_method == "uniform":
            nn.init.uniform_(self.H_param, a=-0.1, b=0.1)
        else:
            nn.init.normal_(self.H_param, mean=0.0, std=0.1)

    def forward(self, X, return_extras=False):
        """
        前向传播
        
        参数:
            X: 输入张量，形状为 (Batch, Nodes, Features_in)
            return_extras: 如果为 True，返回注意力权重和 H 矩阵用于可视化
            
        返回:
            h: 输出张量，形状为 (Batch, Nodes, Features_out)
            如果 return_extras=True，还返回 (att, H)
        """
        B, N, F_in = X.shape
        
        # 输入验证
        if N != self.n_nodes:
            raise ValueError(
                f"节点数不匹配: 期望 {self.n_nodes}，得到 {N}"
            )
        if F_in != self.n_features:
            raise ValueError(
                f"输入特征维度不匹配: 期望 {self.n_features}，得到 {F_in}"
            )

        # 1) 数据驱动的超边分布：根据输入动态生成 + 可学习先验
        # H(X) = softmax(f(X) + H_param)，每个样本有不同的超图结构
        H_logits = self.h_generator(X)  # (B, N, M)
        H_logits = H_logits + self.H_param.unsqueeze(0)  # 加上可学习先验
        H_soft = F.softmax(H_logits, dim=1)  # 按节点维度归一化 (B, N, M)
        H = H_soft

        # 2) Top-K 稀疏化（可选，支持批次维度）
        if self.use_topk:
            H = self._apply_topk_sparsification(H_soft)

        # 3) 节点特征变换
        Xn = self.node_transform(X)  # (B, N, D)

        # 4) 超边特征聚合（动态超图，每个样本不同）
        E = torch.einsum('bnm,bnd->bmd', H, Xn)  # (B, M, D)
        E = self.hyperedge_transform(E)

        # 5) 节点-超边注意力
        att = self._compute_attention(Xn, E, H)

        # 6) 将超边信息聚合回节点
        h = torch.matmul(att, E)  # (B, N, D)

        if return_extras:
            return h, att, H
        return h

    def _apply_topk_sparsification(self, H_soft):
        """应用 Top-K 稀疏化机制（支持批次维度）
        
        参数:
            H_soft: (B, N, M) 批次化的超边分布
        返回:
            H: (B, N, M) 稀疏化后的超边分布
        """
        B, N, M = H_soft.shape
        k = max(1, min(self.k_top, N))
        
        # 选择每个超边的 top-k 节点（在节点维度 dim=1 上）
        _, idx = torch.topk(H_soft, k=k, dim=1)  # idx: (B, k, M)
        
        # 创建稀疏掩码
        mask = torch.zeros_like(H_soft, dtype=torch.bool).scatter_(1, idx, True)
        
        # 孤立节点修复：将未被任何超边覆盖的节点分配给其最佳超边
        row_sum = mask.sum(dim=2)  # (B, N)
        uncovered = (row_sum == 0)  # (B, N)
        if uncovered.any():
            best_e = torch.argmax(H_soft, dim=2)  # (B, N) 每个节点的最佳超边
            best_e_onehot = torch.zeros_like(mask).scatter_(2, best_e.unsqueeze(2), True)
            mask = mask | (uncovered.unsqueeze(2) & best_e_onehot)

        # 应用掩码并按节点维度重新归一化
        H = H_soft * mask.float()
        den = H.sum(dim=1, keepdim=True).clamp_min(self.tau)  # (B, 1, M)
        H = H / den
        
        return H

    def _compute_attention(self, Xn, E, H):
        """计算节点-超边注意力权重"""
        N = Xn.shape[1]
        M = E.shape[1]
        
        # 扩展维度以进行拼接
        N_exp = Xn.unsqueeze(2)  # (B, N, 1, D)
        E_exp = E.unsqueeze(1)   # (B, 1, M, D)
        
        # 拼接节点和超边特征
        att_in = torch.cat([
            N_exp.expand(-1, -1, M, -1),
            E_exp.expand(-1, N, -1, -1)
        ], dim=-1)  # (B, N, M, 2D)

        # 计算注意力分数
        e = self.leakyrelu(
            torch.matmul(att_in, self.attention_weight).squeeze(-1)
        )  # (B, N, M)

        # 使用超边结构进行掩码（H 已是 (B, N, M)，每个样本有不同结构）
        if self.use_topk:
            # 使用 log(H + tau) 进行掩码以强制结构
            e = e + torch.log(H + self.tau)
            
            # 创建掩码
            mask_h = (H > 0)  # (B, N, M)
            att = F.softmax(e, dim=-1)
            att = att * mask_h
            att = att / (att.sum(dim=-1, keepdim=True) + self.tau)
        else:
            e = e + torch.log(H + self.tau)
            att = F.softmax(e, dim=-1)

        # 应用 Dropout
        att = F.dropout(att, self.dropout, training=self.training)
        
        return att
