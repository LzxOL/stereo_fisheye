# The multi-resolution transformer (MRT) components used by this encoder are
# adapted from the official S2M2 implementation (see module/mrt_*.py):
#   https://github.com/junhong-3dv/s2m2 (CC BY-NC 4.0, non-commercial use only)

import torch
import torch.nn as nn

from .mrt_layers import CNNEncoder
from .mrt_model import StackedMRT, Unet


# Fig. 2 特征提取：共享 CNN -> 四尺度 U-Net -> 多层 MRT -> 每相机一张特征图。
# 原始图像尚未变成 ERP；此文件的 H/W 始终是鱼眼图像平面的空间尺寸。
class PDFOmniEncoder(nn.Module):
    def __init__(
        self,
        base_channel: int = 32,
        dim_expansion: int = 2,
        num_transformer: int = 3,
        downsample_twice: bool = False,
        pe_mode: str = "grid",
    ):
        super().__init__()
        self.downsample_twice = downsample_twice

        self.cnn_backbone = CNNEncoder(output_dim=base_channel)

        self.feat_pyramid = Unet(
            dims=[base_channel, base_channel, 2 * base_channel],
            dim_expansion=dim_expansion,
            use_gate_fusion=True,
            use_pe=True,
            pe_mode=pe_mode,
            n_attn=num_transformer * 2,
        )

        self.transformer = StackedMRT(
            num_transformer=num_transformer,
            dims=[base_channel, base_channel, 2 * base_channel],
            num_heads=1,
            dim_expansion=dim_expansion,
            use_gate_fusion=True,
        )

    def forward(self, imgs):
        # 输入四项 [B,Cin,Hi,Wi]，沿批次拼为 [4B,Cin,Hi,Wi] 共享权重提取特征。
        # 保持整组相机顺序 [前的B张,右的B张,后的B张,左的B张]，不是交错排列样本。
        is_list = isinstance(imgs, (list, tuple))
        if is_list:
            batch_size = imgs[0].shape[0]
            x = torch.cat(imgs, dim=0)
        else:
            x = imgs

        # 默认数据为灰度；复制三份只是适配 CNN 的 3 通道输入，不产生额外颜色信息。
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat_4x, feat_2x = self.cnn_backbone(x)

        # CNN 返回 1/4、1/2 分辨率，默认使用后者进入 U-Net。
        # 默认四尺度相对原图为 1/2、1/4、1/8、1/16，通道分别 C/C/2C/2C。
        feat_in = feat_4x if self.downsample_twice else feat_2x
        z0, z1, z2, z3 = self.feat_pyramid(feat_in)

        # MRT 的跨视图注意力按批次两半配对，因此前/后、右/左会互相交换信息。
        # 空间尺度间的信息通过下采样、上采样与门控跳连传递。
        feat_out = self.transformer(z0, z1, z2, z3)

        if is_list:
            # [4B,C,Hf,Wf] 按原相机块拆回四项，供各自标定网格进行 spherical_sweep。
            feats = torch.split(feat_out, batch_size, dim=0)
            return list(feats)

        return feat_out
