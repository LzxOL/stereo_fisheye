"""两张鱼眼图像 -> 扫描体 -> ACV -> 几何聚合 -> 物理逆深度/深度。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CNNEncoder
from .sweeping import SphericalSweeping
from .volume import StereoVolumeGenerator


def inverse_depth_to_depth(inverse_depth, valid_mask):
    # 输入/输出同为 [B,1,h,w]；h/w 可为工作尺寸 H/W 或输出尺寸 Ho/Wo。
    # 无效处先用 1 作为分母，再将结果置零，避免 1/0 及其反传产生 Inf/NaN。
    safe_inverse = torch.where(valid_mask, inverse_depth, torch.ones_like(inverse_depth))  # [B,1,h,w]
    return torch.where(valid_mask, safe_inverse.reciprocal(), 0.0)  # [B,1,h,w]


class StereoBaseline(nn.Module):
    def __init__(self, num_depth=96, sweep_size=(64, 128), output_size=(128, 256),
                 min_depth=1.0, max_depth=50.0, share_encoder=False):
        super().__init__()

        # 常用尺寸直接放在类里：96 个候选，64x128 工作网格，128x256 输出。
        self.num_depth = num_depth
        self.sweep_height, self.sweep_width = sweep_size
        self.output_size = output_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        # 3D 沙漏需要 D/H/W 能被 8 整除。
        assert num_depth > 0 and num_depth % 8 == 0, "num_depth must be a positive multiple of 8"
        assert all(n > 0 and n % 8 == 0 for n in sweep_size), "sweep_size must be positive multiples of 8"
        assert 0 < min_depth < max_depth < float("inf"), "Require 0 < min_depth < max_depth < inf"

        # 直接调用原版结构的 CNNEncoder；左右默认各自一套参数。
        self.encoder_left = CNNEncoder(output_dim=32)
        self.encoder_right = self.encoder_left if share_encoder else CNNEncoder(output_dim=32)
        self.sweeping = SphericalSweeping()
        self.volume_generator = StereoVolumeGenerator()
        # 直接使用 D 个物理逆深度值；不经过“完整索引 192 -> 工作索引 96”的换算。
        self.register_buffer("inv_depth_values", torch.linspace(
            1.0 / self.max_depth, 1.0 / self.min_depth, self.num_depth))  # [D]；默认 [96]

    def forward(self, left_image, right_image, left_grid, right_grid,
                valid_mask=None, return_intermediates=False):
        """输入 grid 的候选顺序必须与 self.inv_depth_values 一致。

        left_grid: [B,H,W,2]，right_grid: [B,H,W,D,2]，坐标归一化到 [-1,1]。
        valid_mask: 可选 bool [B,1,D,H,W]，进一步排除无效候选。
        输出是字典：高/低分辨率逆深度、深度、有效掩码、候选概率和匹配注意力。
        """
        # 以下标注：B=批大小，D=候选数，H/W=扫描网格，Ho/Wo=输出尺寸。
        # 默认 D=96、H=64、W=128、Ho=128、Wo=256；CNN 的 Hf/Wf 取决于各自图像尺寸。
        batch = left_image.shape[0]
        assert left_grid.shape == (batch, self.sweep_height, self.sweep_width, 2), "left_grid shape mismatch"
        assert right_grid.shape == (batch, self.sweep_height, self.sweep_width, self.num_depth, 2), "right_grid shape mismatch"

        # 1. 原版 CNN 返回 (x_4x, x_2x)，这里取第二个，即 32 通道的 1/2 特征。
        # 保留原版 1/4 分支，但其输出不参与当前匹配，因此该分支参数没有梯度。
        _, left_feature = self.encoder_left(left_image)  # [B,32,HfL,WfL]；512x512 输入时 [B,32,256,256]
        _, right_feature = self.encoder_right(right_image)  # [B,32,HfR,WfR]；512x512 输入时 [B,32,256,256]

        # 2. 球面采样：各得到 [B,32,96,64,128] 特征体。
        left_volume, right_volume, candidate_valid = self.sweeping(
            left_feature, right_feature, left_grid, right_grid, valid_mask)  # 两体各 [B,32,D,H,W]；mask [B,1,D,H,W]

        # 3. GWC -> Attention -> ACV -> 3D 聚合 -> 候选概率。
        probability, attention, intermediates = self.volume_generator(
            left_volume, right_volume, candidate_valid, return_intermediates)  # 前两项各 [B,1,D,H,W]；第三项为 dict/None

        # 4. 沿候选轴加权求和，得到 [B,1,64,128] 物理逆深度。
        # probability=[B,1,D,H,W]，values=[1,1,D,1,1]。
        values = self.inv_depth_values.float().view(1, 1, self.num_depth, 1, 1)  # [1,1,D,1,1]；默认 [1,1,96,1,1]
        inv_depth_low = (probability * values).sum(dim=2)  # [B,1,H,W]；默认 [B,1,64,128]
        valid_low = candidate_valid.any(dim=2)  # bool [B,1,H,W]；默认 [B,1,64,128]

        # 5. 上采样到 128x256。只改变空间尺寸，不对逆深度乘 2。
        # 用有效性权重归一化，避免无效零值拉低边界预测。
        valid_weight = F.interpolate(valid_low.float(), size=self.output_size, mode="bilinear", align_corners=False)  # [B,1,Ho,Wo]；默认 [B,1,128,256]
        inv_depth = F.interpolate(inv_depth_low, size=self.output_size, mode="bilinear", align_corners=False)  # [B,1,Ho,Wo]
        inv_depth = inv_depth / valid_weight.clamp_min(1e-8)  # [B,1,Ho,Wo]
        valid = F.interpolate(valid_low.float(), size=self.output_size, mode="nearest").bool()  # bool [B,1,Ho,Wo]
        inv_depth = torch.where(valid, inv_depth, 0.0)  # [B,1,Ho,Wo]

        result = {
            "inv_depth": inv_depth,  # [B,1,Ho,Wo]；默认 [B,1,128,256]
            "depth": inverse_depth_to_depth(inv_depth, valid),  # [B,1,Ho,Wo]
            "valid_mask": valid,  # bool [B,1,Ho,Wo]
            "inv_depth_low": inv_depth_low,  # [B,1,H,W]；默认 [B,1,64,128]
            "depth_low": inverse_depth_to_depth(inv_depth_low, valid_low),  # [B,1,H,W]
            "valid_mask_low": valid_low,  # bool [B,1,H,W]
            "probability": probability.squeeze(1),  # [B,D,H,W]；默认 [B,96,64,128]
            "attention": attention,  # [B,1,D,H,W]；默认 [B,1,96,64,128]
        }
        if return_intermediates:
            intermediates["left_feature"] = left_feature  # [B,32,HfL,WfL]
            intermediates["right_feature"] = right_feature  # [B,32,HfR,WfR]
            result["intermediates"] = intermediates  # dict：各阶段张量见 volume.py
        return result  # dict：张量形状见上方各字段
