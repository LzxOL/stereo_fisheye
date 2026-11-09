"""Split-Volume：GEV 用于 Context，CV 单独查询每轮匹配证据。"""

import torch
import torch.nn.functional as F


def probability_uncertainty(init_prob, candidate_valid, eps=1e-8):
    probability = torch.where(candidate_valid[:, 0], init_prob.float(), 0.0)
    entropy = -(probability * torch.log(probability + eps)).sum(dim=1, keepdim=True)
    count = candidate_valid.sum(dim=2)
    uncertainty = entropy / count.clamp_min(2).float().log()
    uncertainty = torch.where(count > 1, uncertainty, 0.0).clamp(0.0, 1.0)
    return uncertainty


def aggregate_geometry(geometry_volume, init_prob, candidate_valid):
    # 保持 OT 的 init_prob 不变，仅排除无效候选；使用 float32，保留两侧梯度。
    probability = torch.where(candidate_valid[:, 0], init_prob.float(), 0.0)
    geometry = torch.where(candidate_valid, geometry_volume.float(), 0.0)
    geometry_2d = (geometry * probability[:, None]).sum(dim=2)  # [B,8,H,W]
    return geometry_2d, probability_uncertainty(init_prob, candidate_valid)


class CorrelationVolumeSampler:
    def __init__(self, correlation_volume, num_levels=2, radius=4):
        B, C, D, H, W = correlation_volume.shape
        if C != 1 or D < 2 ** (num_levels - 1):
            raise ValueError("CV must have one channel and enough depth layers for its pyramid")
        self.B, self.H, self.W = B, H, W
        self.radius = radius
        # 与 CombinedGeoEncodingSampler 一致：每个像素独立沿候选轴建金字塔。
        volume = correlation_volume.permute(0, 3, 4, 1, 2).reshape(B * H * W, 1, 1, D)
        self.pyramid = [volume]
        for _ in range(1, num_levels):
            volume = F.avg_pool2d(volume, [1, 2], stride=[1, 2])
            self.pyramid.append(volume)

    def __call__(self, depth_index):
        B, _, H, W = depth_index.shape
        assert (B, H, W) == (self.B, self.H, self.W)
        index = depth_index.reshape(B * H * W, 1, 1, 1)
        offsets = torch.arange(-self.radius, self.radius + 1, device=index.device, dtype=index.dtype)
        offsets = offsets.view(1, 1, -1, 1)
        samples = []
        for level, volume in enumerate(self.pyramid):
            position = index / (2 ** level) + offsets
            x = 2.0 * position / max(1, volume.shape[-1] - 1) - 1.0
            grid = torch.cat([x, torch.zeros_like(x)], dim=-1).to(volume.dtype)
            # 连续索引线性插值，越界零填充；没有传统水平视差的平移或 all-pairs 相关。
            feature = F.grid_sample(volume, grid, align_corners=True, padding_mode="zeros")
            feature = feature.view(B, H, W, 2 * self.radius + 1).permute(0, 3, 1, 2)
            samples.append(feature)
        return torch.cat(samples, dim=1).float()  # 默认 [B,18,H,W]
