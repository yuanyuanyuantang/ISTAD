"""一维卷积层 - 用于提取时间序列的局部特征"""

import torch.nn as nn


class ConvLayer(nn.Module):
    """一维卷积层，用于提取每个时间序列输入的高级特征
    
    使用因果卷积确保不泄露未来信息，适用于时间序列预测任务。
    
    参数:
        n_features: 输入特征/节点数量
        kernel_size: 卷积操作中使用的卷积核大小
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.n_features = n_features
        self.kernel_size = kernel_size
        
        # 使用因果卷积 padding：只在左侧填充，确保不泄露未来信息
        # padding 大小为 kernel_size - 1，这样输出长度与输入长度相同
        self.padding = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, window_size, n_features)
            
        返回:
            输出张量，形状为 (batch_size, window_size, n_features)
        """
        if x.size(2) != self.n_features:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.n_features}，得到 {x.size(2)}"
            )
        
        # 转换为 (batch, features, time) 以适配 Conv1d
        x = x.permute(0, 2, 1)
        
        # 应用因果填充和卷积
        x = self.padding(x)
        x = self.relu(self.conv(x))
        
        # 转换回 (batch, time, features)
        return x.permute(0, 2, 1)
