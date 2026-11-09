from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import Edict
from .mrt_layers import CNNEncoder
from .volume_generator import Generator
from .context import ContextNetwork
from .update import UpdateBlock
from .volume_lookup import aggregate_geometry, probability_uncertainty, CorrelationVolumeSampler
from .volume_generator import CombinedGeoEncodingSampler
from .config import load_config
from torch.utils.checkpoint import checkpoint

autocast = partial(torch.amp.autocast, 'cuda')


# 双目鱼眼 -> 球面体 / 初始索引 -> 2D Context Network -> 原 UpdateBlock 迭代优化。
# 所有索引输出保留扫描网格尺寸；不进行物理深度转换或索引单位缩放。
class StereoBaseline(nn.Module):
    def __init__(self, num_depth=None, sweep_size=None, share_encoder=False, mixed_precision=False,
                 volume_update_mode="split", corr_levels=2, corr_radius=4, ot_iter=3,
                 sweep_precision="float32", gradient_checkpointing=False,
                 gru_small_kernel=1, gru_large_kernel=5, gru_large_mode="sep"):
        super().__init__()
        defaults = load_config()["data"]
        num_depth = defaults["num_depth"] if num_depth is None else num_depth
        sweep_size = defaults["sweep_size"] if sweep_size is None else sweep_size
        self.num_depth = num_depth
        self.gradient_checkpointing = gradient_checkpointing
        if volume_update_mode not in ("joint", "split"):
            raise ValueError("volume_update_mode must be joint or split")
        if sweep_precision not in ("float32", "amp"):
            raise ValueError("sweep_precision must be float32 or amp")
        self.volume_update_mode = volume_update_mode
        self.sweep_precision = sweep_precision
        self.sweep_height, self.sweep_width = sweep_size
        assert num_depth > 0 and num_depth % 8 == 0, "num_depth must be a positive multiple of 8"
        assert all(n > 0 and n % 8 == 0 for n in sweep_size), "sweep_size must be positive multiples of 8"
        if corr_levels < 1 or corr_radius < 0 or num_depth < 2 ** (corr_levels - 1) or ot_iter < 1:
            raise ValueError("Invalid correlation pyramid/radius or OT iteration count")

        # 沿用原 Generator 的 opts 接口；这里只列本阶段需要指定的参数。
        self.opts = Edict()
        self.opts.base_channel = 32
        self.opts.mixed_precision = mixed_precision
        self.opts.use_geo_acv = False  # 原版完整 ERP 的经纬度不适用于任意左参考网格。
        self.opts.gwc_groups = 8
        self.opts.geo_ch = 8
        self.opts.corr_levels = corr_levels
        self.opts.corr_radius = corr_radius
        self.opts.ot_iter = ot_iter
        self.opts.gru_small_kernel = gru_small_kernel
        self.opts.gru_large_kernel = gru_large_kernel
        self.opts.gru_large_mode = gru_large_mode
        self.opts.corr_channels = 1 if volume_update_mode == "split" else self.opts.geo_ch + 1
        self.opts.num_downsample = 0  # UpdateBlock 的掩码头配置；本阶段不调用其上采样头。

        # CNNEncoder 就是复制来的 mrt_layers.py 中的原类，没有另外包装 CNN。
        # 左右结构一致，默认参数独立；不使用原 backbone.py 的 U-Net / MRT。
        self.encoder_left = CNNEncoder(output_dim=32)
        self.encoder_right = self.encoder_left if share_encoder else CNNEncoder(output_dim=32)
        self.volume_gen = Generator(self.opts)

        # 沿用原 UpdateBlock 配置：Hidden=2C=64，Context=C=32。
        context_channels = 41 if volume_update_mode == "split" else 32
        self.context_net = ContextNetwork(hidden_dim=64, context_dim=32, input_dim=context_channels)
        self.update_block = UpdateBlock(self.opts, hidden_dim=64, input_dim=32)

    def _run(self, block, *args, **kwargs):
        # 只重算 CNN / Context / GRU；它们没有 BatchNorm running stats。
        # 不重算 3D BatchNorm，避免 checkpoint 导致同一批统计量被更新两次。
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False, **kwargs)
        return block(*args, **kwargs)

    def spherical_sweep(self, fisheye_feats, grids, valid_mask=None):
        # fisheye_feats=[左特征,右特征]，各 [B,32,Hf,Wf]，左右 Hf/Wf 可不同。
        # grids=[left_grid,right_grid]：分别为 [B,H,W,2]、[B,H,W,D,2]，末维是归一化 (u,v)。
        # 与原版不同：参考原点在左相机，左投影不随候选半径改变；只有右网格含 D 个投影。
        # 网格由调用方提供，本函数只执行采样，不生成标定或射线；不要求 ERP / 360°。
        left_feat, right_feat = fisheye_feats
        left_grid, right_grid = grids
        B = left_feat.shape[0]
        assert left_grid.shape == (B, self.sweep_height, self.sweep_width, 2), "left_grid shape mismatch"
        assert right_grid.shape == (B, self.sweep_height, self.sweep_width, self.num_depth, 2), "right_grid shape mismatch"

        left_valid = torch.isfinite(left_grid).all(dim=-1) & (left_grid.abs() <= 1).all(dim=-1)
        right_valid = torch.isfinite(right_grid).all(dim=-1) & (right_grid.abs() <= 1).all(dim=-1)
        # 网格精度保持 float32；可选 AMP 仅交给 CUDA autocast 决定特征采样的实际 dtype。
        left_grid = torch.where(left_valid[..., None], left_grid, 2.0).float()
        right_grid = torch.where(right_valid[..., None], right_grid, 2.0).float()

        # 原版 5D grid_sample：输入 [B,C,Hf,Wf,1]，坐标补成 (0,u,v)，0 索引长度为 1 的轴。
        left_grid_pad = torch.cat([torch.zeros_like(left_grid[..., :1]), left_grid], dim=-1)
        reference_feat = F.grid_sample(left_feat[..., None], left_grid_pad.unsqueeze(3), align_corners=True)
        reference_feat = torch.where(left_valid[:, None, :, :, None], reference_feat, 0.0)
        reference_feat = reference_feat.expand(-1, -1, -1, -1, self.num_depth)  # [B,C,H,W,D]

        right_grid_pad = torch.cat([torch.zeros_like(right_grid[..., :1]), right_grid], dim=-1)
        target_feat = F.grid_sample(right_feat[..., None], right_grid_pad, align_corners=True)
        target_feat = torch.where(right_valid[:, None], target_feat, 0.0)

        candidate_valid = left_valid[:, None, None] & right_valid.permute(0, 3, 1, 2)[:, None]  # [B,1,D,H,W]
        if valid_mask is not None:
            assert valid_mask.shape == candidate_valid.shape and valid_mask.dtype == torch.bool
            candidate_valid = candidate_valid & valid_mask
        # 不附加四路原版的 G0..G3，因为这里没有相机间的融合 MLP。
        # Reference 保留左图观测，右图某一候选不可见不应将左图上下文也清空；匹配时另用联合 mask。
        return [reference_feat, target_feat], candidate_valid

    def forward(self, imgs, grids, valid_mask=None, iters=12, return_intermediates=False):
        # 沿用原版 imgs/grids 列表接口，现均为两项：[左,右]。
        # imgs 各 [B,3,H0,W0]；H/W 是扫描网格尺寸，与原图、CNN 特征图尺寸分开。
        assert len(imgs) == 2 and len(grids) == 2, "Expected two fisheye images and two sampling grids"
        assert isinstance(iters, int) and iters >= 0, "iters must be a non-negative integer"
        # 两路 CNN 均取 1/2 分辨率特征 [B,32,Hf,Wf]。
        with autocast(enabled=self.opts.mixed_precision):
            _, left_feat = self._run(self.encoder_left, imgs[0])
            _, right_feat = self._run(self.encoder_right, imgs[1])

        # 默认保留已有 float32 扫描路径；AMP 扫描为显式实验选项，实际 dtype 由 profiling 记录。
        use_sweep_amp = self.opts.mixed_precision and self.sweep_precision == "amp"
        fisheye_feats = [left_feat, right_feat] if use_sweep_amp else [left_feat.float(), right_feat.float()]
        with autocast(enabled=use_sweep_amp):
            spherical_feats, candidate_valid = self.spherical_sweep(fisheye_feats, grids, valid_mask)

        with autocast(enabled=self.opts.mixed_precision):
            volumes, init_disp, context_feat_volume, init_prob = self.volume_gen(
                spherical_feats, candidate_valid, return_volumes=True)
        geometry_volume, correlation_volume = volumes
        if self.volume_update_mode == "split":
            geometry_2d, uncertainty = aggregate_geometry(geometry_volume, init_prob, candidate_valid)
            geo_sampler = CorrelationVolumeSampler(correlation_volume, self.opts.corr_levels, self.opts.corr_radius)
        else:
            geometry_2d = None
            uncertainty = probability_uncertainty(init_prob, candidate_valid)
            geo_sampler = CombinedGeoEncodingSampler(geometry_volume, correlation_volume,
                                                    self.opts.corr_levels, self.opts.corr_radius)

        # init_disp 是 [0,D-1] 内的连续逆深度候选索引，不是物理深度或水平像素视差。
        # 左体沿 D 广播，第 0 层即左参考二维特征；右侧候选 0 不可见不会清空它。
        context_feat = context_feat_volume[..., 0]  # [B,32,H,W]
        if self.volume_update_mode == "split":
            context_feat = torch.cat([context_feat.float(), geometry_2d, uncertainty], dim=1)  # [B,41,H,W]
        with autocast(enabled=self.opts.mixed_precision):
            hidden_init, context_feature = self._run(self.context_net, context_feat)

        net = hidden_init
        invdepth_idx = init_disp
        valid_pixels = candidate_valid.any(dim=2)
        disp_predictions = []
        for _ in range(iters):
            # 与原版相同，仅切断上轮“索引”的梯度；隐藏状态仍沿迭代反传。
            # VE/VC 没有 detach，因此查表特征仍可训练两路 CNN 和 3D 匹配分支。
            invdepth_idx = invdepth_idx.detach()
            geo_feat = geo_sampler(invdepth_idx)  # split:18 通道；joint:162 通道
            with autocast(enabled=self.opts.mixed_precision):
                net, delta_invdepth_idx, _ = self._run(self.update_block,
                    net, context_feature, geo_feat, invdepth_idx, no_upsample=True)

            # 增量和初值均为当前 D 个逆深度候选的索引，不乘空间缩放因子。
            invdepth_idx = (invdepth_idx + delta_invdepth_idx.float()).clamp(0, self.num_depth - 1)
            invdepth_idx = torch.where(valid_pixels, invdepth_idx, 0.0)
            disp_predictions.append(invdepth_idx)

        result = {
            "init_disp": init_disp,
            "init_prob": init_prob,
            "disp_predictions": disp_predictions,
            "disp": invdepth_idx,  # [B,1,H,W]；iters=0 时为初值
            "valid_mask": valid_pixels,
            "uncertainty": uncertainty,
        }
        if return_intermediates:
            result["intermediates"] = dict(reference=spherical_feats[0], target=spherical_feats[1],
                                           gev=geometry_volume, cv=correlation_volume, geometry_2d=geometry_2d,
                                           context_input=context_feat, hidden_init=hidden_init,
                                           context_feature=context_feature, candidate_valid=candidate_valid)
        return result
