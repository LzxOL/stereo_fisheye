# Adapted from the official S2M2 implementation (with modifications):
#   "S2M2: Scalable Stereo Matching Model for Reliable Depth Estimation", ICCV 2025
#   Junhong Min, Youngpil Jeon, Jimin Kim, Minyong Choi
#   https://github.com/junhong-3dv/s2m2 (CC BY-NC 4.0, non-commercial use only)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor


# token 自注意力：[Btokens,N,C] -> 同形状；多头内部维度为 dim_expansion*C。
class SelfAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool,
    ):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.use_pe = use_pe
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)

        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)
        if self.use_pe:
            self.pe_proj = nn.Linear(32, self.head_dim)

    def forward(self, x: torch.Tensor, pe: torch.Tensor = None) -> torch.Tensor:
        # q/k/v=[Btokens,heads,N,head_dim]，标准注意力为 softmax(QK^T/sqrt(head_dim))*V。
        B, N, C = x.shape
        scale = self.scale

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_pe:
            score = torch.einsum("...ic, ...jc -> ...ij", scale * q, k)
            attn = score.reshape(B, self.num_heads, N, N).softmax(dim=-1)
            out = torch.einsum("...ij, ...jc -> ...ic", attn, v)
            # 相对位置编码 [N,N,32] 按注意力对 key 求和，再投影加入每个 query 的输出。
            pe_sum = torch.einsum("...nij, ijc -> ...nic", attn, pe)
            out = out + self.pe_proj(pe_sum)
        else:
            out = cp.checkpoint(F.scaled_dot_product_attention, q, k, v, use_reentrant=False)

        out = self.proj(out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))

        return out


# 双向跨视图注意力：Qx 对 Ky/Vy 取信息，同时 Qy 对 Kx/Vx 取信息。
# 输入两组同尺寸 token [Btokens,N,C]，返回两组更新量；外层 block 再加残差。
class CrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
    ):
        super(CrossAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        B, _, C = y.shape

        qx = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        ky = self.k(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vy = self.v(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x_out = F.scaled_dot_product_attention(qx, ky, vy)

        kx = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        qy = self.q(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vx = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        y_out = F.scaled_dot_product_attention(qy, kx, vx)

        x_out = self.proj(x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))
        y_out = self.proj(y_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))

        return x_out, y_out


# 将每一行作为独立序列 [B*H,W,C]，只在行内做自注意力，降低细尺度开销。
class SelfAttnBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool,
    ):
        super(SelfAttnBlock1D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn = SelfAttn(dim=self.dim, num_heads=self.num_heads, dim_expansion=dim_expansion, use_pe=use_pe)
        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor, pe: torch.Tensor = None):
        B, H, W, C = z.shape
        z = z.reshape(B * H, W, C)

        z_norm = self.norm_pre(z)
        z = self.attn(z_norm, pe) + z

        z = z.reshape(B, H, W, C)
        return z


# 将批次两半视作两组待匹配视图，并在每个相同行号内做跨视图注意力。
# 发生在鱼眼平面，尚未做球面投影，不能把这种逐行交换理解为严格的球面极线匹配。
class CrossAttnBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
    ):
        super(CrossAttnBlock1D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn = CrossAttn(self.dim, self.num_heads, dim_expansion=dim_expansion)

        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor):
        z_norm = self.norm_pre(z)
        # [前B,右B,后B,左B] -> x=[前B,右B]、y=[后B,左B]。
        # 对应 batch 元素配对，故实际交互为前<->后、右<->左，而非全部六种相机对。
        x, y = z_norm.chunk(2, dim=0)

        B, H, W, C = x.shape
        x, y = x.reshape(B * H, W, C), y.reshape(B * H, W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        z = torch.cat([x, y], dim=0) + z

        return z


# [B,H,W,C] 展平成 [B,H*W,C]，在最粗层做覆盖全图的自注意力。
class SelfAttnBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool,
    ):
        super(SelfAttnBlock2D, self).__init__()

        self.dim = dim
        self.attn = SelfAttn(dim=dim, num_heads=num_heads, dim_expansion=dim_expansion, use_pe=use_pe)
        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor, pe: torch.Tensor = None):
        B, H, W, C = z.shape
        z = z.reshape(B, H * W, C).contiguous()
        z_norm = self.norm_pre(z)
        z = self.attn(z_norm, pe) + z
        z = z.reshape(B, H, W, C).contiguous()

        return z


# 同样按批次两半配对，但每个 query 可访问相反视图的整张特征图。
class CrossAttnBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
    ):
        super(CrossAttnBlock2D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn = CrossAttn(self.dim, self.num_heads, dim_expansion=dim_expansion)
        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor):
        z_norm = self.norm_pre(z)
        x, y = z_norm.chunk(2, dim=0)

        B, H, W, C = x.shape
        x, y = x.reshape(B, H * W, C), y.reshape(B, H * W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        z = torch.cat([x, y], dim=0) + z

        return z


# Transformer 的逐 token 通道映射：LayerNorm -> 扩展 -> GELU -> 压回 -> 残差。
class FFN(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_expansion: int,
    ):
        super(FFN, self).__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, dim_expansion * self.dim),
            nn.GELU(),
            nn.Linear(dim_expansion * self.dim, self.dim),
        )

        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor):
        z_norm = self.norm_pre(z)
        z = self.ffn(z_norm) + z

        return z


# U-Net 的卷积块：3x3 空间分支与 1x1 通道分支相加，输入输出通道/尺寸一致。
class ConvBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dim_expansion: int,
    ):
        super(ConvBlock2D, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.convs = nn.Sequential(
            nn.Conv2d(self.dim, dim_expansion * self.dim, self.kernel_size, padding=self.kernel_size // 2),
            nn.GELU(),
            nn.Conv2d(dim_expansion * self.dim, self.dim, self.kernel_size, padding=self.kernel_size // 2),
        )

        self.convs_1x = nn.Sequential(
            nn.Conv2d(self.dim, dim_expansion * self.dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim_expansion * self.dim, self.dim, 1),
        )

    def forward(self, z: torch.Tensor):
        out = self.convs(z) + self.convs_1x(z)

        return out


# 最粗尺度的组合块：可选全图跨视图注意力 -> 全图自注意力，各自配 FFN。
class GlobalAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_cross_attn: bool = False,
        use_pe: bool = False,
    ):
        super(GlobalAttnBlock, self).__init__()

        self.self_attn = SelfAttnBlock2D(dim=dim, num_heads=num_heads, dim_expansion=dim_expansion, use_pe=use_pe)
        if use_cross_attn:
            self.cross_attn = CrossAttnBlock2D(dim=dim, num_heads=num_heads, dim_expansion=dim_expansion)
            self.ffn_c = FFN(dim=dim, dim_expansion=dim_expansion)
        else:
            self.cross_attn = None
        self.ffn = FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: torch.Tensor, pe: torch.Tensor = None):
        z = z.permute(0, 2, 3, 1)
        if self.cross_attn is not None:
            z = self.cross_attn(z)
            z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        z = z.permute(0, 3, 1, 2)

        return z.contiguous()


# 细尺度的组合块：逐行跨视图注意力 -> FFN -> 逐行自注意力 -> FFN。
# 与 GlobalAttnBlock 的区别是序列长度为 W，而非 H*W。
class BasicAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool = False,
    ):
        super(BasicAttnBlock, self).__init__()

        self.cross_attn = CrossAttnBlock1D(dim=dim, num_heads=num_heads, dim_expansion=dim_expansion)

        self.self_attn = SelfAttnBlock1D(dim=dim, num_heads=num_heads, dim_expansion=dim_expansion, use_pe=use_pe)
        self.ffn_c = FFN(dim=dim, dim_expansion=dim_expansion)
        self.ffn = FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: torch.Tensor, pe: torch.Tensor = None):
        z = z.permute(0, 2, 3, 1)
        z = self.cross_attn(z)
        z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        z = z.permute(0, 3, 1, 2)
        return z


# 融合两个同尺度 [B,C,H,W] 特征：学习残差 + w*z0+(1-w)*z1。
# 用在尺度间跳连，不是 volume_generator 中四相机投到 ERP 后的融合层。
class FeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        use_gate=True,
    ):
        super(FeatureFusion, self).__init__()

        pad = kernel_size // 2
        self.use_gate = use_gate
        if use_gate:
            self.feature_gate = nn.Sequential(
                nn.Conv2d(2 * dim, dim, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )

    def forward(self, z0: Tensor, z1: Tensor):
        # Align feature map sizes for odd-sized inputs (e.g., 1532 height).
        # Upsample/downsample chains can differ by 1 px at skip connections.
        if z0.shape[-2:] != z1.shape[-2:]:
            h = min(z0.shape[-2], z1.shape[-2])
            w = min(z0.shape[-1], z1.shape[-1])
            z0 = z0[..., (z0.shape[-2] - h) // 2:(z0.shape[-2] - h) // 2 + h,
                    (z0.shape[-1] - w) // 2:(z0.shape[-1] - w) // 2 + w]
            z1 = z1[..., (z1.shape[-2] - h) // 2:(z1.shape[-2] - h) // 2 + h,
                    (z1.shape[-1] - w) // 2:(z1.shape[-1] - w) // 2 + w]

        z = torch.cat([z0, z1], dim=1)
        if self.use_gate:
            eps = 0.01
            # w=[B,C,H,W] 为逐通道门控；限制到 [0.01,0.99] 使两条尺度路径都保留信息。
            w = self.feature_gate(z).clamp(min=eps, max=1 - eps)
            z_out = self.feature_fusion(z) + (w) * z0 + (1 - w) * z1
        else:
            z_out = self.feature_fusion(z)

        return z_out


# 共享二维 CNN，输入 [4B,3,Hi,Wi]；返回 (1/4 特征,1/2 特征)，两者通道均为 C。
# 更粗的 1/8、1/16 特征由 Unet 的池化路径产生，而不是此类直接返回四层。
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
        x = self.conv0(x)
        x_2x = self.norm1(self.conv1_down(x))
        x_2x = self.conv2(x_2x) + x_2x
        x_4x = self.conv2_down(x_2x)
        return x_4x, x_2x
