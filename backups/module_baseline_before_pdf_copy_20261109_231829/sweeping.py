"""按给定网格采样左右特征；不在此生成标定参数或 ERP 图像。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalSweeping(nn.Module):
    def forward(self, left_feature, right_feature, left_grid, right_grid, valid_mask=None):
        # left_grid=[B,H,W,2]：左参考网格在左特征图中的归一化 (u,v)。
        # right_grid=[B,H,W,D,2]：相同参考位置、不同候选在右特征图中的 (u,v)。
        # 这里的 H/W 是参考网格尺寸，未要求它等于鱼眼特征图的 Hf/Wf。
        # B=批大小，C=32；默认 D=96、H=64、W=128，左右输入特征各为 [B,C,Hf,Wf]。
        D = right_grid.shape[3]

        # 坐标有限且落在 [-1,1] 只是基本有效性；外部 mask 可进一步排除不可用观测。
        # 在采样前替换 NaN/Inf，避免先产生 NaN 后再乘零仍得到 NaN。
        left_valid = torch.isfinite(left_grid).all(dim=-1) & (left_grid.abs() <= 1).all(dim=-1)  # bool [B,H,W]
        right_valid = torch.isfinite(right_grid).all(dim=-1) & (right_grid.abs() <= 1).all(dim=-1)  # bool [B,H,W,D]
        left_safe = torch.where(left_valid[..., None], left_grid, 2.0).to(left_feature.dtype)  # [B,H,W,2]
        right_safe = torch.where(right_valid[..., None], right_grid, 2.0).to(right_feature.dtype)  # [B,H,W,D,2]

        # 左相机是参考，采样位置不随候选变化：只采一次，再沿 D 广播。
        # 沿用 PDFOmni.spherical_sweep 的 5D grid_sample 写法：
        # 特征 [B,C,Hf,Wf,1] 被解释为 [N,C,Din,Hin,Win]，
        # 因而网格末维要由 (u,v) 补成 (0,u,v)：0 对应长度为 1 的轴，
        # u/v 分别索引鱼眼特征图的列/行。输出 H/W 由参考射线网格决定，并不要求 ERP。
        left_grid_pad = torch.cat([torch.zeros_like(left_safe[..., :1]), left_safe], dim=-1)  # [B,H,W,3]
        left_sample = F.grid_sample(left_feature[..., None], left_grid_pad.unsqueeze(3), align_corners=True)  # [B,C,H,W,1]
        left_sample = left_sample[..., 0]  # [B,C,H,W,1] -> [B,C,H,W]
        left_volume = left_sample.unsqueeze(2).expand(-1, -1, D, -1, -1)  # [B,C,D,H,W]；默认 [B,32,96,64,128]

        # 右侧对每条左参考射线的 D 个候选点采样，一次得到原版排列 [B,C,H,W,D]。
        # 这里只执行网格采样；射线、候选半径和相机投影已编码在外部传入的 right_grid 中。
        right_grid_pad = torch.cat([torch.zeros_like(right_safe[..., :1]), right_safe], dim=-1)  # [B,H,W,D,3]
        right_sample = F.grid_sample(right_feature[..., None], right_grid_pad, align_corners=True)  # [B,C,H,W,D]
        right_volume = right_sample.permute(0, 1, 4, 2, 3)  # -> [B,C,D,H,W]

        valid = left_valid[:, None, None] & right_valid.permute(0, 3, 1, 2)[:, None]  # bool [B,1,D,H,W]
        if valid_mask is not None:
            valid = valid & valid_mask  # bool [B,1,D,H,W]
        # 对外统一 [B,C,D,H,W]；无效候选在左右体中都置零。
        left_volume = torch.where(valid, left_volume, 0.0)  # [B,C,D,H,W]
        right_volume = torch.where(valid, right_volume, 0.0)  # [B,C,D,H,W]；默认 [B,32,96,64,128]
        return left_volume, right_volume, valid  # 两体各 [B,C,D,H,W]；mask [B,1,D,H,W]
