"""GRU层 - 用于时序编码和解码"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULayer(nn.Module):
    """门控循环单元（GRU）层
    
    用于编码时间序列的时序依赖关系。
    
    参数:
        in_dim: 输入特征维度
        hid_dim: 隐藏层维度
        n_layers: GRU 层数
        dropout: Dropout 比率（仅在多层时使用）
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.in_dim = in_dim
        
        # 创建多层 GRU cells
        self.cells = nn.ModuleList([
            nn.GRUCell(in_dim if i == 0 else hid_dim, hid_dim)
            for i in range(n_layers)
        ])

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, in_dim)
            
        返回:
            out: 最后时间步的输出，形状为 (batch_size, hid_dim)
            h_final: 最后一层的隐藏状态，形状为 (batch_size, hid_dim)
        """
        if x.size(2) != self.in_dim:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.in_dim}，得到 {x.size(2)}"
            )
        
        b, seq, feat = x.shape
        
        # 初始化隐藏状态
        h = [
            torch.zeros(b, self.hid_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.n_layers)
        ]
        
        # 转换为 (seq, batch, feature) 以便逐时间步处理
        x_t = x.permute(1, 0, 2)
        
        outputs = []
        for t in range(seq):
            inp = x_t[t]
            for layer in range(self.n_layers):
                h_new = self.cells[layer](inp, h[layer])
                
                # 在多层 GRU 之间应用 dropout（最后一层除外）
                if layer < self.n_layers - 1 and self.dropout > 0:
                    h_new = F.dropout(h_new, self.dropout, training=self.training)
                
                h[layer] = h_new
                inp = h_new
            
            outputs.append(inp)
        
        # 返回完整的时序输出和最后一层的隐藏状态
        all_outputs = torch.stack(outputs, dim=1)  # (batch, seq, hid_dim)
        h_final = h[-1]                            # (batch, hid_dim)
        
        return all_outputs, h_final


class RNNDecoder(nn.Module):
    """基于 GRU 的解码器网络
    
    将潜在向量转换为输出序列，用于重建任务。
    
    参数:
        in_dim: 输入特征维度
        hid_dim: 隐藏层维度
        n_layers: GRU 层数
        dropout: Dropout 比率
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        
        # 如果输入维度与隐藏维度不同，使用投影层
        if self.in_dim != self.hid_dim:
            self.input_proj = nn.Linear(in_dim, hid_dim)
            first_cell_input = hid_dim
        else:
            self.input_proj = None
            first_cell_input = in_dim
        
        # 创建多层 GRU cells
        self.cells = nn.ModuleList([
            nn.GRUCell(first_cell_input if i == 0 else hid_dim, hid_dim)
            for i in range(n_layers)
        ])

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, in_dim)
            
        返回:
            decoder_out: 解码输出，形状为 (batch_size, seq_len, hid_dim)
        """
        if x.size(2) != self.in_dim:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.in_dim}，得到 {x.size(2)}"
            )
        
        b, seq, feat = x.shape
        
        # 初始化隐藏状态
        h = [
            torch.zeros(b, self.hid_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.n_layers)
        ]
        
        # 转换为 (seq, batch, feature)
        x_t = x.permute(1, 0, 2)
        
        outputs = []
        for t in range(seq):
            inp = x_t[t]
            
            # 应用输入投影（如果需要）
            if self.input_proj is not None:
                inp = self.input_proj(inp)
            
            for layer in range(self.n_layers):
                h_new = self.cells[layer](inp, h[layer])
                
                # 在多层 GRU 之间应用 dropout（最后一层除外）
                if layer < self.n_layers - 1 and self.dropout > 0:
                    h_new = F.dropout(h_new, self.dropout, training=self.training)
                
                h[layer] = h_new
                inp = h_new
            
            outputs.append(inp)
        
        # 转换回 (batch, seq, hid_dim)
        decoder_out = torch.stack(outputs, dim=0).permute(1, 0, 2)
        
        return decoder_out
