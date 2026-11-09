"""直接沿用 module/mrt_layers.py 中的 CNNEncoder，不引入 MRT。"""

# 原实现的来源声明：
# Adapted from the official S2M2 implementation (with modifications):
#   "S2M2: Scalable Stereo Matching Model for Reliable Depth Estimation", ICCV 2025
#   Junhong Min, Youngpil Jeon, Jimin Kim, Minyong Choi
#   https://github.com/junhong-3dv/s2m2 (CC BY-NC 4.0, non-commercial use only)

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(CNNEncoder, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.GELU(), nn.Conv2d(16, 16, kernel_size=1))

        self.conv1_down = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1),
        )

        self.norm1 = nn.GroupNorm(8, output_dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
        )

        self.conv2_down = nn.Sequential(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1))

    def forward(self, x: torch.Tensor):
        # 输入 [B,3,H0,W0]，先提取 16 通道特征，再下采样到约 1/2 分辨率。
        # C=output_dim（baseline 为 32）；Hf=ceil(H0/2)，Wf=ceil(W0/2)。
        # 以 512x512 输入为例：Hf=Wf=256，1/4 尺度为 128x128。
        x = self.conv0(x)  # [B,16,H0,W0]；例 [B,16,512,512]
        x_2x = self.norm1(self.conv1_down(x))  # [B,C,Hf,Wf]；例 [B,32,256,256]
        # 1/2 尺度上的残差细化；baseline 将这份特征送入球面扫描。
        x_2x = self.conv2(x_2x) + x_2x  # [B,C,Hf,Wf]；例 [B,32,256,256]
        x_4x = self.conv2_down(x_2x)  # [B,C,ceil(Hf/2),ceil(Wf/2)]；例 [B,32,128,128]
        # 保留原版两个输出：[B,C,ceil(H0/4),ceil(W0/4)]、[B,C,ceil(H0/2),ceil(W0/2)]。
        return x_4x, x_2x  # (1/4 特征, 1/2 特征)，通道均为 C
