"""普通重构 decoder：逐时间点共享 MLP（规格 4.5 节）

仅负责从统一时空表示恢复观测，不作为论文贡献单独命名。
输出层不加激活（标准化输入可为任意实数）。
"""

import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """h (B,L,N) → 重构 (B,L,N)；LayerNorm → Linear → GELU → Dropout → Linear"""

    def __init__(self, n_features, hidden_dim=None, dropout=0.1, out_dim=None):
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            hidden_dim = n_features
        out_dim = n_features if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)
