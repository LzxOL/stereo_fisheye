import math
import torch
import torch.nn as nn


# Fig. 2 Geometry Embedding：球面方向 [B,3,H,W] -> 上下文通道 [B,C,H,W]。
# 和 PDD 不同，GE 是可学习特征补充；PDD 是 DAA 使用的单通道固定几何先验。
class GE(nn.Module):
    def __init__(self, input_dim=3, mapping_size=256, scale=10.0, out_dim=128, learnable_B=False):
        super().__init__()
        self.out_dim = out_dim

        # 随机 Fourier 投影矩阵 [3,M]；默认注册为 buffer，随 checkpoint 保存但不训练。
        B_mat = torch.randn((input_dim, mapping_size)) * scale
        if learnable_B:
            self.B = nn.Parameter(B_mat)
        else:
            self.register_buffer("B", B_mat)

        self.mlp = nn.Sequential(
            nn.Linear(mapping_size * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # 通道移到最后，以每个 ERP 像素的 xyz 作为一个独立的 3 维输入。
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"GE expects x with shape [B, 3, H, W], got {tuple(x.shape)}")

        x = x.permute(0, 2, 3, 1)

        B_mat = self.B
        if B_mat.dtype != x.dtype:
            B_mat = B_mat.to(dtype=x.dtype)
        # gamma(x)=[sin(2*pi*xB),cos(2*pi*xB)]，形状 [B,H,W,2M]。
        proj = (2.0 * math.pi) * torch.matmul(x, B_mat)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # MLP 压到 C 通道，再还原 BCHW，与参考体采样得到的上下文逐元素相加。
        out = self.mlp(fourier)
        return out.permute(0, 3, 1, 2)
