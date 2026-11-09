"""从左参考二维特征生成 GRU 的初始隐藏状态和固定上下文。"""

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.relu(self.norm1(self.conv1(x)))
        residual = self.norm2(self.conv2(residual))
        return self.relu(x + residual)


class ContextNetwork(nn.Module):
    def __init__(self, hidden_dim=64, context_dim=32):
        super().__init__()
        # 单尺度处理，不改变 H/W；GroupNorm 不依赖 batch 的大小。
        self.stem = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
        )
        self.residual_blocks = nn.Sequential(ResidualBlock2D(32), ResidualBlock2D(32))
        self.hidden_head = nn.Conv2d(32, hidden_dim, 3, padding=1)
        self.context_head = nn.Conv2d(32, context_dim, 3, padding=1)

    def forward(self, reference_feature):
        # 输入 [B,32,H,W]，来自左 Reference Volume 的任一深度切片。
        feature = self.residual_blocks(self.stem(reference_feature))
        hidden_init = torch.tanh(self.hidden_head(feature))  # [B,64,H,W]
        context_feature = torch.relu(self.context_head(feature))  # [B,32,H,W]
        return hidden_init, context_feature
