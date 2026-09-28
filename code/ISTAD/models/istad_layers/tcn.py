"""时间卷积网络（TCN）- 用于捕获长期时间依赖"""

import torch.nn as nn
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """裁剪层 - 用于因果卷积"""
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """裁剪填充部分以保持因果性"""
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """时间块 - TCN 的基本构建单元
    
    等价于残差块，包含两个卷积层和残差连接。
    
    参数:
        n_inputs: 输入通道数
        n_outputs: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        dilation: 膨胀率
        padding: 填充大小
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 组合网络
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # 残差连接的下采样层（如果输入输出维度不同）
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0, std=0.01)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch, channels, seq_len)
            
        返回:
            输出张量，形状为 (batch, channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """时间卷积网络（TCN）
    
    通过堆叠多个时间块来捕获长期时间依赖。
    使用膨胀卷积扩大感受野。
    
    参数:
        num_inputs: 输入通道数（特征数）
        num_channels: 每层的输出通道数列表
        kernel_size: 卷积核大小
        dropout: Dropout 比率
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            # 膨胀率呈指数增长
            dilation_size = 2 ** i
            
            # 确定输入输出通道数
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            # 添加时间块
            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch, channels, seq_len)
            
        返回:
            输出张量，形状为 (batch, channels, seq_len)
        """
        return self.network(x)
