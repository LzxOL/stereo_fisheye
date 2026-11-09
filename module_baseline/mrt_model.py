# Adapted from the official S2M2 implementation (with modifications):
#   "S2M2: Scalable Stereo Matching Model for Reliable Depth Estimation", ICCV 2025
#   Junhong Min, Youngpil Jeon, Jimin Kim, Minyong Choi
#   https://github.com/junhong-3dv/s2m2 (CC BY-NC 4.0, non-commercial use only)

import torch
import torch.nn as nn
from torch import Tensor

from .mrt_layers import BasicAttnBlock, ConvBlock2D, FeatureFusion, GlobalAttnBlock
from .mrt_utils import get_pe


# 在单视图特征上建立四尺度金字塔，并在最粗尺度做全局自注意力。
# 输入 [4B,C,Hf,Wf]；返回 z0/z1/z2/z3，空间尺寸每级减半（边缘尺寸依实际运算）。
class Unet(nn.Module):
    def __init__(
        self,
        dims: list,
        dim_expansion: int,
        use_pe: bool,
        pe_mode: str = "grid",
        n_attn: int = 1,
        use_gate_fusion: bool = True,
    ):
        super(Unet, self).__init__()
        self.dims = dims
        self.use_pe = use_pe
        self.pe_mode = pe_mode

        self.down_conv0 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[0], dims[1], kernel_size=1))
        self.down_conv1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[1], dims[2], kernel_size=1))
        self.down_conv2 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.up_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[1], dims[0], kernel_size=1),
        )
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[2], dims[1], kernel_size=1),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[2], dims[2], kernel_size=1),
        )

        self.concat_conv0 = FeatureFusion(dims[0], kernel_size=1, use_gate=use_gate_fusion)
        self.concat_conv1 = FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.concat_conv2 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.enc0 = ConvBlock2D(dim=dims[0], kernel_size=3, dim_expansion=dim_expansion)
        self.enc1 = ConvBlock2D(dim=dims[1], kernel_size=3, dim_expansion=dim_expansion)
        self.enc2 = ConvBlock2D(dim=dims[2], kernel_size=3, dim_expansion=dim_expansion)
        self.enc3s = nn.ModuleList()
        for i in range(n_attn):
            self.enc3s.append(
                GlobalAttnBlock(
                    dim=dims[2],
                    num_heads=8,
                    dim_expansion=dim_expansion,
                    use_cross_attn=False,
                    use_pe=use_pe,
                )
            )

        self.dec0 = ConvBlock2D(dim=dims[0], kernel_size=3, dim_expansion=dim_expansion)
        self.dec1 = ConvBlock2D(dim=dims[1], kernel_size=3, dim_expansion=dim_expansion)
        self.dec2 = ConvBlock2D(dim=dims[2], kernel_size=3, dim_expansion=dim_expansion)
        self.dec3s = nn.ModuleList()
        for i in range(n_attn):
            self.dec3s.append(
                GlobalAttnBlock(
                    dim=dims[2],
                    num_heads=8,
                    dim_expansion=dim_expansion,
                    use_cross_attn=False,
                    use_pe=False,
                )
            )

    def forward(self, z: Tensor):
        # 位置编码是最粗层任意 token 对的相对坐标嵌入 [N,N,32]，N=(Hf/8)*(Wf/8)。
        # 这是鱼眼特征平面 PE，不是后续 ERP 的球面 GE 或 PDD。
        if self.use_pe:
            H, W = z.shape[-2:]
            pe = get_pe(H // 8, W // 8, 32, z.dtype, z.device, mode=self.pe_mode)
        else:
            pe = None

        z0 = self.enc0(z)
        z1 = self.down_conv0(z0)

        z1 = self.enc1(z1)
        z2 = self.down_conv1(z1)

        z2 = self.enc2(z2)
        z3 = self.down_conv2(z2)

        for block in self.enc3s:
            z3 = block(z3, pe)

        for block in self.dec3s:
            z3 = block(z3, pe)
        z3_new = z3

        # 解码器逐级恢复分辨率；FeatureFusion 结合同级编码特征与粗尺度上下文。
        z2_new = self.up_conv2(z3_new)
        z2_new = self.concat_conv2(z2, z2_new)
        z2_new = self.dec2(z2_new)

        z1_new = self.up_conv1(z2_new)
        z1_new = self.concat_conv1(z1, z1_new)
        z1_new = self.dec1(z1_new)

        z0_new = self.up_conv0(z1_new)
        z0_new = self.concat_conv0(z0, z0_new)
        z0_new = self.dec0(z0_new)

        return z0_new, z1_new, z2_new, z3_new


# 一个多分辨率 Transformer：细尺度使用逐行注意力，最粗尺度使用全图注意力。
# 每个尺度都包含相反视图的 cross-attention 与自身的 self-attention；
# 尺度间靠上下采样和 FeatureFusion 联通，而非直接将不同尺度 token 做 cross-attention。
class MRT(nn.Module):
    def __init__(
        self,
        dims: list,
        num_heads: int,
        dim_expansion: int,
        use_gate_fusion: bool,
    ):
        super(MRT, self).__init__()
        self.num_heads = num_heads
        self.dims = dims

        self.down_conv0 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[0], dims[1], kernel_size=1))
        self.down_conv1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[1], dims[2], kernel_size=1))
        self.down_conv2 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.up_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[1], dims[0], kernel_size=1),
        )
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[2], dims[1], kernel_size=1),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[2], dims[2], kernel_size=1),
        )

        self.down_concat1 = FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.down_concat2 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)
        self.down_concat3 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.up_concat0 = FeatureFusion(dims[0], kernel_size=1, use_gate=use_gate_fusion)
        self.up_concat1 = FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.up_concat2 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.enc_attn0 = BasicAttnBlock(dim=dims[0], num_heads=1 * num_heads, dim_expansion=dim_expansion, use_pe=False)
        self.enc_attn1 = BasicAttnBlock(dim=dims[1], num_heads=2 * num_heads, dim_expansion=dim_expansion, use_pe=False)
        self.enc_attn2 = BasicAttnBlock(dim=dims[2], num_heads=4 * num_heads, dim_expansion=dim_expansion, use_pe=False)
        self.enc_attn3s = nn.ModuleList()
        for i in range(2):
            self.enc_attn3s.append(
                GlobalAttnBlock(
                    dim=dims[2],
                    num_heads=8 * num_heads,
                    dim_expansion=dim_expansion,
                    use_cross_attn=True,
                    use_pe=False,
                )
            )

        self.dec_attn0 = BasicAttnBlock(dim=dims[0], num_heads=1 * num_heads, dim_expansion=dim_expansion, use_pe=False)

        self.dec_attn1 = BasicAttnBlock(dim=dims[1], num_heads=2 * num_heads, dim_expansion=dim_expansion, use_pe=False)

        self.dec_attn2 = BasicAttnBlock(dim=dims[2], num_heads=4 * num_heads, dim_expansion=dim_expansion, use_pe=False)

        self.dec_attn3s = nn.ModuleList()
        for i in range(2):
            self.dec_attn3s.append(
                GlobalAttnBlock(
                    dim=dims[2],
                    num_heads=8 * num_heads,
                    dim_expansion=dim_expansion,
                    use_cross_attn=True,
                    use_pe=False,
                )
            )

    def forward(
        self,
        z0: Tensor,
        z1: Tensor,
        z2: Tensor,
        z3: Tensor,
    ):
        # z0..z3 共享批次布局 [4B,...]；在向下路径把细尺度证据不断融入粗尺度。
        z0 = self.enc_attn0(z0)
        z1 = self.down_concat1(z1, self.down_conv0(z0))
        z1 = self.enc_attn1(z1)
        z2 = self.down_concat2(z2, self.down_conv1(z1))

        z2 = self.enc_attn2(z2)
        z3 = self.down_concat3(z3, self.down_conv2(z2))

        for block in self.enc_attn3s:
            z3 = block(z3)

        for block in self.dec_attn3s:
            z3 = block(z3)

        # 向上路径把已交换视图信息的全局上下文逐级传回最高分辨率 z0。
        z3_up = self.up_conv2(z3)
        z2 = self.up_concat2(z2, z3_up)
        z2 = self.dec_attn2(z2)

        z2_up = self.up_conv1(z2)
        z1 = self.up_concat1(z1, z2_up)
        z1 = self.dec_attn1(z1)

        z1_up = self.up_conv0(z1)
        z0 = self.up_concat0(z0, z1_up)
        z0 = self.dec_attn0(z0)

        return z0, z1, z2, z3


# 默认串联 3 个 MRT；中间保留四尺度，最后仅输出最高分辨率 z0=[4B,C,Hf,Wf]。
class StackedMRT(nn.Module):
    def __init__(
        self,
        num_transformer: int,
        dims: list,
        num_heads: int,
        dim_expansion: int,
        use_gate_fusion: bool,
    ):
        super(StackedMRT, self).__init__()

        uformer_list = nn.ModuleList()
        for ii in range(num_transformer):
            uformer_list.append(
                MRT(dims=dims, num_heads=num_heads, dim_expansion=dim_expansion, use_gate_fusion=use_gate_fusion)
            )
        self.uformer_list = uformer_list

    def forward(
        self,
        z0: Tensor,
        z1: Tensor,
        z2: Tensor,
        z3: Tensor,
    ):
        for l, uformer in enumerate(self.uformer_list):
            z0, z1, z2, z3 = uformer(z0, z1, z2, z3)

        return z0.contiguous()
