"""HGST v2 核心块：Hypergraph-Guided Spline Temporal Block

数据流（规格：ISTAD_HGST_模型与实验完整修改方案.md 第 2.2/3 节）：
    X (B,L,N)
    ├─ 空间路径: Z=[x,Δx] 描述符 → KAN 样条边函数 → 动态超图 → HGAT → 关系状态 R
    │            (R 一路供时间样条条件化，一路 readout 为 Hs)
    └─ 时间路径: grouped 1×1 提升 → 多层 [grouped 膨胀因果卷积 → 关系条件化样条激活
                 → dropout → 残差] → grouped 1×1 投影回 N 维
    → 门控融合 H = LN(H_T + G⊙H_S) → 统一表示 (B,L,N)

因果性：除左侧 padding 的分组因果卷积外全部为逐时间点算子，无跨窗口归一化。
消融（--istad_ablation）：full / fixed_temporal_spline / silu_temporal /
linear_incidence / no_spatial（kill-switch 阶段子集，其余按方案 Phase 8 再补）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spline_ops import SplineIncidenceGenerator, ConditionalSplineActivation


class HypergraphAttentionCore(nn.Module):
    """纯传播超图注意力核：incidence 生成（样条）与聚合解耦

    输入 node_features (Bflat, N, d_s)、incidence (Bflat, N, M)（已 softmax 归一化），
    执行 Top-K 稀疏化（含孤立节点修复）→ 超边聚合 → 节点-超边注意力 → 残差 + LayerNorm。
    逻辑延续 legacy HypergraphAttentionLayer，但不再内部生成 incidence。
    """

    def __init__(self, d_s, n_nodes, n_hyperedges, k_top=None,
                 dropout=0.1, alpha=0.2, tau=1e-6):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_hyperedges = n_hyperedges
        self.k_top = k_top if k_top is not None else max(3, math.ceil(0.2 * n_nodes))
        self.tau = tau

        self.node_transform = nn.Linear(d_s, d_s, bias=False)
        self.hyperedge_transform = nn.Linear(d_s, d_s, bias=False)
        self.attention_weight = nn.Parameter(torch.empty(2 * d_s, 1))
        nn.init.xavier_uniform_(self.attention_weight.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

        # 残差投影 + LayerNorm（spec 4.3）
        self.residual_proj = nn.Linear(d_s, d_s)
        self.norm = nn.LayerNorm(d_s)
        self.dropout = dropout

    def _apply_topk(self, H_soft):
        """H_soft: (Bflat, N, M) → Top-K 稀疏 + 按节点维重归一化（沿用 legacy 实现）"""
        B, N, M = H_soft.shape
        k = max(1, min(self.k_top, N))
        _, idx = torch.topk(H_soft, k=k, dim=1)
        mask = torch.zeros_like(H_soft, dtype=torch.bool).scatter_(1, idx, True)

        # 孤立节点修复：未被任何超边覆盖的节点加入其概率最大的超边
        row_sum = mask.sum(dim=2)
        uncovered = row_sum == 0
        if uncovered.any():
            best_e = torch.argmax(H_soft, dim=2)
            best_e_onehot = torch.zeros_like(mask).scatter_(2, best_e.unsqueeze(2), True)
            mask = mask | (uncovered.unsqueeze(2) & best_e_onehot)

        H = H_soft * mask.float()
        den = H.sum(dim=1, keepdim=True).clamp_min(self.tau)
        return H / den

    def _compute_attention(self, Xn, E, H):
        N, M = Xn.shape[1], E.shape[1]
        N_exp = Xn.unsqueeze(2)
        E_exp = E.unsqueeze(1)
        att_in = torch.cat([
            N_exp.expand(-1, -1, M, -1),
            E_exp.expand(-1, N, -1, -1)
        ], dim=-1)
        e = self.leakyrelu(torch.matmul(att_in, self.attention_weight).squeeze(-1))
        e = e + torch.log(H + self.tau)
        mask_h = H > 0
        att = F.softmax(e, dim=-1)
        att = att * mask_h
        att = att / (att.sum(dim=-1, keepdim=True) + self.tau)
        att = F.dropout(att, self.dropout, training=self.training)
        return att

    def forward(self, node_features, incidence):
        """
        node_features: (Bflat, N, d_s)
        incidence:     (Bflat, N, M)，已按节点维 softmax
        返回:          relation state (Bflat, N, d_s)
        """
        N = node_features.shape[1]
        if N != self.n_nodes:
            raise ValueError(f"节点数不匹配: 期望 {self.n_nodes}，得到 {N}")

        H = self._apply_topk(incidence)
        Xn = self.node_transform(node_features)
        E = torch.einsum('bnm,bnd->bmd', H, Xn)
        E = self.hyperedge_transform(E)
        att = self._compute_attention(Xn, E, H)
        h = torch.matmul(att, E)
        return self.norm(self.residual_proj(node_features) + h)


class _Chomp1d(nn.Module):
    """因果裁剪：只保留左侧 padding 后的合法感受野"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        return x


class ConditionalTemporalBlock(nn.Module):
    """单层时间块：grouped 膨胀因果卷积 → 关系条件化样条激活 → dropout → 残差

    输入/输出: (B, C, L)，C = N·r_t（通道 c 属于变量 c // r_t）
    """

    def __init__(self, n_features, temporal_width, relation_dim, kernel_size,
                 dilation, dropout, spline_grid_size, spline_order,
                 modulation_rank, modulation_scale, ablation="full"):
        super().__init__()
        C = n_features * temporal_width
        self.C = C
        self.pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(C, C, kernel_size, dilation=dilation, padding=self.pad,
                              groups=n_features)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

        self.chomp = _Chomp1d(self.pad)
        # 激活级消融映射：只有 silu_temporal/fixed_temporal_spline 影响激活器；
        # linear_incidence / no_spatial 只作用于空间路径，时间条件化保持 full
        act_ablation = ablation if ablation in ("silu_temporal", "fixed_temporal_spline") else "full"
        self.act = ConditionalSplineActivation(
            n_features=n_features,
            temporal_width=temporal_width,
            relation_dim=relation_dim,
            modulation_rank=modulation_rank,
            modulation_scale=modulation_scale,
            grid_size=spline_grid_size,
            spline_order=spline_order,
            ablation=act_ablation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, relation_state, return_delta_norm=False):
        # 因果卷积：conv 对称 padding 后 chomp 裁掉右侧受未来污染的尾部（等效左侧因果 padding）
        u = self.chomp(self.conv(t))
        if return_delta_norm:
            out, dn = self.act(u, relation_state, return_delta_norm=True)
        else:
            out, dn = self.act(u, relation_state), None
        out = t + self.dropout(out)
        return (out, dn) if return_delta_norm else (out, None)


class ConditionalTemporalNet(nn.Module):
    """多层时间网络：dilation 自动取到感受野 ≥ seq_len，最多 6 层

    RF = 1 + (k-1)·Σ_{l=0}^{J-1} 2^l；实现断言 RF >= seq_len。
    """

    def __init__(self, n_features, seq_len, temporal_width, relation_dim,
                 kernel_size=3, max_levels=6, dropout=0.1,
                 spline_grid_size=5, spline_order=3,
                 modulation_rank=2, modulation_scale=0.05, ablation="full"):
        super().__init__()
        C = n_features * temporal_width
        self.C = C

        # 自动层数：RF = 1 + (k-1)(2^J - 1) >= seq_len
        need = (seq_len - 1) / (kernel_size - 1) + 1
        levels = max(1, math.ceil(math.log2(need)))
        self.levels = min(levels, max_levels)
        self.receptive_field = 1 + (kernel_size - 1) * (2 ** self.levels - 1)
        if self.receptive_field < seq_len:
            raise ValueError(
                f"感受野不足: {self.receptive_field} < seq_len {seq_len}（层数已达上限 {max_levels}）")

        self.blocks = nn.ModuleList([
            ConditionalTemporalBlock(
                n_features, temporal_width, relation_dim, kernel_size,
                dilation=2 ** l, dropout=dropout,
                spline_grid_size=spline_grid_size, spline_order=spline_order,
                modulation_rank=modulation_rank, modulation_scale=modulation_scale,
                ablation=ablation)
            for l in range(self.levels)
        ])

    def forward(self, t, relation_state, return_delta_norm=False):
        """t: (B, C, L)；relation_state: (B, L, N, d_s)"""
        dns = []
        for blk in self.blocks:
            if return_delta_norm:
                t, dn = blk(t, relation_state, return_delta_norm=True)
                dns.append(dn)
            else:
                t, _ = blk(t, relation_state)
        if return_delta_norm:
            return t, torch.stack(dns)
        return t


class HGST2Block(nn.Module):
    """HGST v2 统一时空块：X (B,L,N) → 统一表示 (B,L,N)"""

    def __init__(
        self,
        n_features,
        seq_len,
        relation_dim=32,
        n_hyperedges=None,
        k_top=None,
        spatial_grid_size=5,
        spatial_spline_order=3,
        hypergraph_temperature=1.0,
        temporal_width=2,
        temporal_kernel_size=3,
        temporal_levels_max=6,
        temporal_grid_size=5,
        temporal_spline_order=3,
        modulation_rank=2,
        modulation_scale=0.05,
        dropout=0.1,
        ablation="full",
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.ablation = str(ablation)

        if n_hyperedges is None:
            n_hyperedges = max(4, math.ceil(n_features / 2))
        if k_top is None:
            k_top = max(3, math.ceil(0.2 * n_features))

        # --- 空间路径 ---
        self.relation_dim = relation_dim
        # 变量描述符编码: E0 = W_z·[x, Δx] + e_i（变量身份嵌入，含偏置作用）
        self.desc_proj = nn.Linear(2, relation_dim)
        self.var_embed = nn.Embedding(n_features, relation_dim)
        nn.init.xavier_uniform_(self.desc_proj.weight)
        nn.init.zeros_(self.desc_proj.bias)
        nn.init.normal_(self.var_embed.weight, std=0.02)

        if self.ablation == "linear_incidence":
            # 消融：线性边函数替代样条边函数（同先验/偏置结构）
            self.linear_incidence = nn.Linear(2, n_hyperedges)
            nn.init.normal_(self.linear_incidence.weight, std=0.01)
            nn.init.zeros_(self.linear_incidence.bias)
            self.incidence_gen = None
        else:
            self.incidence_gen = SplineIncidenceGenerator(
                n_features, n_hyperedges, q_in=2,
                grid_size=spatial_grid_size, spline_order=spatial_spline_order,
                temperature=hypergraph_temperature)
            self.linear_incidence = None

        self.hgat_core = HypergraphAttentionCore(
            d_s=relation_dim, n_nodes=n_features, n_hyperedges=n_hyperedges,
            k_top=k_top, dropout=dropout)
        self.spatial_readout = nn.Linear(relation_dim, 1)

        # --- 时间路径 ---
        C = n_features * temporal_width
        self.temporal_width = temporal_width
        self.temporal_in = nn.Conv1d(n_features, C, kernel_size=1, groups=n_features)
        nn.init.kaiming_normal_(self.temporal_in.weight, nonlinearity='relu')
        nn.init.zeros_(self.temporal_in.bias)

        self.temporal_net = ConditionalTemporalNet(
            n_features=n_features, seq_len=seq_len, temporal_width=temporal_width,
            relation_dim=relation_dim, kernel_size=temporal_kernel_size,
            max_levels=temporal_levels_max, dropout=dropout,
            spline_grid_size=temporal_grid_size, spline_order=temporal_spline_order,
            modulation_rank=modulation_rank, modulation_scale=modulation_scale,
            ablation=ablation)

        self.temporal_out = nn.Conv1d(C, n_features, kernel_size=1, groups=n_features)
        nn.init.kaiming_normal_(self.temporal_out.weight, nonlinearity='relu')
        nn.init.zeros_(self.temporal_out.bias)

        # --- 门控融合 ---
        self.use_spatial = self.ablation != "no_spatial"
        if self.use_spatial:
            self.gate = nn.Linear(2 * n_features, n_features)
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)  # 初始 gate=0.5
        self.fusion_norm = nn.LayerNorm(n_features)

    def forward(self, x, return_aux=False):
        """x: (B, L, N) → (B, L, N)"""
        B, L, N = x.shape
        if N != self.n_features:
            raise ValueError(f"特征数不匹配: 期望 {self.n_features}，得到 {N}")

        # 描述符 Z=[x, Δx]，Δx_0=0（因果差分）
        dx = torch.zeros_like(x)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]
        Z = torch.stack([x, dx], dim=-1)                      # (B,L,N,2)

        # --- 空间路径 ---
        E0 = self.desc_proj(Z) + self.var_embed.weight.unsqueeze(0).unsqueeze(0)
        E0f = E0.reshape(B * L, N, self.relation_dim)

        if self.incidence_gen is not None:
            inc = self.incidence_gen(Z)                       # (B,L,N,M)
        else:
            inc = F.softmax(self.linear_incidence(Z), dim=2)  # linear_incidence 消融
        incf = inc.reshape(B * L, N, -1)

        Rf = self.hgat_core(E0f, incf)                        # (B*L,N,d_s)
        R = Rf.reshape(B, L, N, self.relation_dim)
        Hs = self.spatial_readout(R).squeeze(-1)              # (B,L,N)

        # --- 时间路径 ---
        xt = x.permute(0, 2, 1)                               # (B,N,L)
        t0 = self.temporal_in(xt)                             # (B,C,L)
        if return_aux:
            ht, dns = self.temporal_net(t0, R, return_delta_norm=True)
        else:
            ht = self.temporal_net(t0, R)
        HT = self.temporal_out(ht).permute(0, 2, 1)           # (B,L,N)

        # --- 门控融合 ---
        if self.use_spatial:
            G = torch.sigmoid(self.gate(torch.cat([HT, Hs], dim=-1)))
            H = self.fusion_norm(HT + G * Hs)
        else:
            H = self.fusion_norm(HT)

        if return_aux:
            aux = {
                'incidence': inc.detach(),
                'relation_state': R.detach(),
                'spatial': Hs.detach(),
                'temporal': HT.detach(),
                'delta_norm': dns.detach() if torch.is_tensor(dns) else dns,
            }
            if self.use_spatial:
                aux['gate'] = G.detach()
            return H, aux
        return H


class ISTADHGST2(nn.Module):
    """HGST v2 版 ISTAD：HGST2Block + 逐时间点 MLP 重构 decoder"""

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        relation_dim=32,
        n_hyperedges=None,
        k_top=None,
        spatial_grid_size=5,
        spatial_spline_order=3,
        hypergraph_temperature=1.0,
        temporal_width=2,
        temporal_kernel_size=3,
        temporal_levels_max=6,
        temporal_grid_size=5,
        temporal_spline_order=3,
        modulation_rank=2,
        modulation_scale=0.05,
        dropout=0.1,
        decoder_hidden=None,
        ablation="full",
    ):
        super().__init__()
        self.window_size = window_size
        self.out_dim = out_dim
        self.hgst = HGST2Block(
            n_features=n_features, seq_len=window_size,
            relation_dim=relation_dim, n_hyperedges=n_hyperedges, k_top=k_top,
            spatial_grid_size=spatial_grid_size,
            spatial_spline_order=spatial_spline_order,
            hypergraph_temperature=hypergraph_temperature,
            temporal_width=temporal_width,
            temporal_kernel_size=temporal_kernel_size,
            temporal_levels_max=temporal_levels_max,
            temporal_grid_size=temporal_grid_size,
            temporal_spline_order=temporal_spline_order,
            modulation_rank=modulation_rank, modulation_scale=modulation_scale,
            dropout=dropout, ablation=ablation)

        if decoder_hidden is None or decoder_hidden <= 0:
            decoder_hidden = n_features
        self.decoder = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, out_dim),
        )

    def forward(self, x):
        """x: (B, L, N) → 重构 (B, L, out_dim)"""
        if x.shape[1] != self.window_size:
            raise ValueError(
                f"窗口长度不匹配: {x.shape[1]} != {self.window_size}")
        h = self.hgst(x)
        return self.decoder(h)
