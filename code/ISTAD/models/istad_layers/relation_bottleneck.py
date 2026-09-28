"""Isolated layers for the opt-in masked-channel relation bottleneck."""

import torch
import torch.nn as nn


class DepthwiseCausalEncoder(nn.Module):
    """Encode each channel's causal history without cross-channel mixing."""

    def __init__(self, n_features, kernel_size=7):
        super().__init__()
        self.n_features = int(n_features)
        self.kernel_size = int(kernel_size)
        self.padding = nn.ConstantPad1d((self.kernel_size - 1, 0), 0.0)
        self.conv = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=self.n_features,
            kernel_size=self.kernel_size,
            groups=self.n_features,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.n_features:
            raise ValueError(
                f"Expected (B,W,{self.n_features}) input, got {tuple(x.shape)}"
            )
        encoded = self.conv(self.padding(x.permute(0, 2, 1)))
        return self.activation(encoded).permute(0, 2, 1)


class ChannelwiseRelationPredictionModel(nn.Module):
    """Decode each channel from its own causal state and HGAT message only."""

    def __init__(self, n_channels, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.n_channels = int(n_channels)
        self.hidden_dim = max(1, int(hidden_dim))
        self.input_projection = nn.Conv1d(
            2 * self.n_channels,
            self.hidden_dim * self.n_channels,
            kernel_size=1,
            groups=self.n_channels,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))
        self.output_projection = nn.Conv1d(
            self.hidden_dim * self.n_channels,
            self.n_channels,
            kernel_size=1,
            groups=self.n_channels,
        )

    def forward(self, local_state, relation_message):
        if local_state.shape != relation_message.shape or local_state.ndim != 3:
            raise ValueError(
                "local_state and relation_message must have matching (B,W,C) shapes"
            )
        if local_state.size(-1) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {local_state.size(-1)}"
            )
        paired = torch.stack((local_state, relation_message), dim=2)
        paired = paired.permute(0, 3, 2, 1).contiguous().view(
            local_state.size(0), 2 * self.n_channels, local_state.size(1)
        )
        hidden = self.dropout(self.activation(self.input_projection(paired)))
        return self.output_projection(hidden).permute(0, 2, 1).contiguous()
