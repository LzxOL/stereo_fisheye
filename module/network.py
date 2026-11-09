import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.featurelayer import Conv2D
from module.backbone import PDFOmniEncoder
from module.volume_generator import Generator
from module.ge import GE
from module.update import UpdateBlock, DiGRU
from utils.common import *

autocast = partial(torch.amp.autocast, 'cuda')


# 论文 Fig. 2 的总入口：四路鱼眼特征 -> 球面特征体 -> 匹配体/初值 -> 迭代细化。
# 全程回归的是“逆深度假设索引”，不是米制深度；物理量换算见 Dataset.indexToInvdepth。
# 记号：B=批大小，C=base_channel，H/W=低分辨率 ERP，D=低分辨率逆深度候选数。
class PDFOmni(torch.nn.Module):

    def __init__(self, varargin=None):
        super(PDFOmni, self).__init__()
        opts = Edict()
        opts.use_rgb = False
        self.opts = argparse(opts, varargin)

        self.use_ge = getattr(self.opts, "use_ge", False)
        self.use_digru = getattr(self.opts, "use_digru", False)
        self.distortion_mode = getattr(self.opts, "distortion_mode", "hyperbolic_dual_disk")

        self.encoder = PDFOmniEncoder(
            base_channel=self.opts.base_channel,
            dim_expansion=getattr(self.opts, "dim_expansion", 2),
            num_transformer=getattr(self.opts, "num_transformer", 3),
            downsample_twice=getattr(self.opts, "encoder_downsample_twice", False),
            pe_mode=getattr(self.opts, "pe_mode", "grid"),
        )
        context_dim = self.opts.base_channel
        hidden_dim  = self.opts.base_channel * 2

        self.volume_gen = Generator(self.opts)

        self.state_conv = Conv2D(context_dim, hidden_dim, 1, pad=0, relu=False)

        # GE 是球面坐标的 Fourier 嵌入；PDD 是标量畸变先验，两者用途不同。
        # 完整论文路径需要启用 use_ge/use_digru（训练命令中的两个开关）。
        if self.use_ge:
            ge_scale = getattr(self.opts, "ge_scale", 10.0)
            ge_map_size = getattr(self.opts, "ge_map_size", 256)
            self.ge = GE(input_dim=3, mapping_size=ge_map_size, scale=ge_scale, out_dim=context_dim)
        else:
            self.ge = None

        if self.use_digru:
            self.update_block = DiGRU(self.opts, hidden_dim=hidden_dim, input_dim=context_dim)
        else:
            self.update_block = UpdateBlock(self.opts, hidden_dim=hidden_dim, input_dim=context_dim)

    def freeze_bn(self):
        # 将当前 BatchNorm2d 切到评估态；后续 model.train() 会再次改变其状态。
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # 1. buildLookupTable 计算采样坐标：ERP像素->射线->候选采样点->投影到四台相机
    # 2. spherical_sweep 根据坐标提取特征：四张鱼眼相机的特征 + grids -> grid_sample
    
    def spherical_sweep(self, fisheye_feats, grids):
        # 输入：四个 [B,C,Hf,Wf] 特征，四个 [H,W,D,2] 查表网格。
        # grids[i][y,x,d] 是 ERP 射线在第 d 个半径球面上的点投回相机 i 的位置。
        # 深度与标定已在 Dataset.buildLookupTable 中编码，此处只做可微反向采样。
        bs = fisheye_feats[0].shape[0]
        # 使用 5D grid_sample 一次取出所有候选深度：输入 feat[...,None] 为
        # [B,C,Hf,Wf,1]，被解释为 [N,C,Din,Hin,Win]。因此采样坐标必须是
        # (0,u,v)：0 对应末尾长度为 1 的轴，u/v 分别寻址鱼眼的列/行。
        # 输出轴虽然名为 Dout/Hout/Wout，实际含义仍是 ERP 的 H/W/候选 D。
        grids_pad = [torch.cat([torch.zeros_like(grid[..., :1]), grid], dim=-1) for grid in grids]
        sph_feats = []
        for feat, grid in zip(fisheye_feats, grids_pad):
            sph_feat = F.grid_sample(feat[..., None], grid.repeat(bs, 1, 1, 1, 1), align_corners=True)
            sph_feats.append(sph_feat)

        # 默认 padding_mode='zeros'：视场外网格产生零特征；没有在这里做可见性重加权。
        # 返回的列表长 8：[F0,F1,F2,F3,G0,G1,G2,G3]。
        # Fi=[B,C,H,W,D]，Gi=[B,2,H,W,D]；附带坐标供后续融合 MLP 判断采样位置。
        sph_feats = sph_feats + [grid.permute(-1, 0, 1, 2).repeat(bs, 1, 1, 1, 1) for grid in grids]
        return sph_feats

    @staticmethod
    def _build_unit_sphere_grid(h, w, device, dtype):
        # 输出 [1,3,H,W] 单位球方向供 GE 使用，不含深度或相机外参。
        # 注意这套角度为 phi∈[-pi/2,pi/2]、lam∈[-pi,pi]（含两端点）；
        # 并非直接复用 Dataset.makeSphericalRays 的裁剪纬度和 +pi/2 经度偏移。
        phi = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device, dtype=dtype)
        lam = torch.linspace(-math.pi, math.pi, w, device=device, dtype=dtype)
        grid_phi, grid_lam = torch.meshgrid(phi, lam, indexing="ij")
        x = -torch.cos(grid_phi) * torch.cos(grid_lam)
        y = torch.sin(grid_phi)
        z = torch.cos(grid_phi) * torch.sin(grid_lam)
        return torch.stack([x, y, z], dim=0).unsqueeze(0)

    @staticmethod
    def _build_distortion_map(h, w, device, dtype, mode="hyperbolic_dual_disk"):
        # 论文 §3.2、式 (2)-(7)：按 ERP 位置生成 [1,1,H,W] 的 PDD 畸变场。
        # 此先验仅由分辨率决定，不读取图像、真实镜头内参或本轮深度估计。
        if mode != "hyperbolic_dual_disk":
            raise ValueError(f"Unknown distortion mode: {mode}")

        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # 圆盘边界的分母接近零；半精度下先转 float32，避免距离计算溢出。
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        grid_x_f = grid_x.to(compute_dtype)
        grid_y_f = grid_y.to(compute_dtype)
        eps_h = torch.as_tensor(1e-6, device=device, dtype=compute_dtype)

        def _acosh(x):
            if hasattr(torch, "acosh"):
                return torch.acosh(x)
            return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=0.0)))

        def get_hyperbolic_distance_from_origin(u_norm, v_norm):
            # [0,1]² -> [-1,1]² -> 单位圆盘内部 -> 到圆心的双曲测地距离。
            x = 2.0 * u_norm - 1.0
            y = 2.0 * v_norm - 1.0

            r = torch.sqrt(x * x + y * y).clamp_min(eps_h)
            # p'=(1-eps)*p/max(1,||p||)：圆外点沿半径缩回边界内。
            # 与论文式 (3) 的分段写法略有区别：圆内点也乘了 (1-eps)。
            scale = (1.0 - eps_h) / torch.maximum(torch.ones_like(r), r)
            px = x * scale
            py = y * scale

            p2 = px * px + py * py
            denom = (1.0 - p2).clamp_min(eps_h)
            # 式 (2)：d(0,p')=acosh(1+2||p'||²/(1-||p'||²))。
            z = 1.0 + 2.0 * p2 / denom
            z = z.clamp_min(1.0 + eps_h)
            return _acosh(z)

        u = (grid_x_f + 1.0) / 2.0
        v = (grid_y_f + 1.0) / 2.0

        df = get_hyperbolic_distance_from_origin(u, v)
        # 后相机圆盘沿水平轴平移半个周期；取模保留 ERP 的周期关系。
        db = get_hyperbolic_distance_from_origin((u + 0.5) % 1.0, v)

        # 式 (5)-(7)：两相机距离越接近，balance 越大，表示视图归属越模糊。
        # min_norm 衡量较优相机仍有多大径向畸变；这里 alpha 固定为 0.5。
        balance = 1.0 - torch.abs(df - db) / (df + db + eps_h)
        min_norm = torch.tanh(torch.min(df, db))
        distortion = torch.tanh(min_norm * (1.0 + 0.5 * balance))

        return distortion.unsqueeze(0).unsqueeze(0)

    def upsample_invdepth_idx(self, invdepth, mask):
        # RAFT 式可学习凸上采样：[B,1,H,W] -> [B,1,sH,sW]，s=2^K。
        # mask=[B,9*s*s,H,W] 为每个输出子像素预测 3x3 邻域的 9 个权重。
        bs, ch, h, w = invdepth.shape
        factor = 2 ** self.opts.num_downsample
        mask = mask.view(bs, 1, 9, factor, factor, h, w)
        mask = torch.softmax(mask, dim=2)

        # 乘 s 同时把低分辨率候选索引换回完整索引单位；并非将米制深度放大。
        up_invdepth = F.unfold(factor * invdepth, [3, 3], padding=1)
        up_invdepth = up_invdepth.view(bs, ch, 9, 1, 1, h, w)

        up_invdepth = torch.sum(mask * up_invdepth, dim=2)
        # [B,ch,s,s,H,W] -> [B,ch,H,s,W,s]，把子像素交织回 ERP 空间位置。
        up_invdepth = up_invdepth.permute(0, 1, 4, 2, 5, 3)
        return up_invdepth.reshape(bs, ch, factor*h, factor*w)

    def volume_sample(self, feat_volume, invdepth_idx):
        # 在上下文体 [B,C,H,W,D] 的候选轴，按 [B,1,H,W] 连续索引线性采样。
        # 每个 ERP 像素只查自己的 D 轴，不在邻近 ERP 像素间取样。
        bs, ch, h, w, n_invd = feat_volume.shape

        invdepth_idx_floor = torch.floor(invdepth_idx)
        invdepth_idx_ceil  = invdepth_idx_floor + 1
        invdepth_idx_floor = torch.clamp(invdepth_idx_floor, 0, n_invd - 1)
        invdepth_idx_ceil  = torch.clamp(invdepth_idx_ceil , 0, n_invd - 1)
        invdepth_idx       = torch.clamp(invdepth_idx      , 0, n_invd - 1)

        # 内部区间：f(d)=(ceil(d)-d)*f_floor+(d-floor(d))*f_ceil。
        # 注意原实现先截断端点：当 d=D-1 且 D>1 时两权重均为 0，输出零上下文。
        # 这里保留其实际边界行为，不将它误解成端点复制，也不修改计算。
        weight_floor = (invdepth_idx_ceil - invdepth_idx)
        weight_floor[weight_floor == n_invd - 1] = 1.0
        weight_ceil  = (invdepth_idx - invdepth_idx_floor)
        weight_ceil[invdepth_idx_ceil == 0] = 1.0

        invdepth_idx_floor = invdepth_idx_floor.long()
        invdepth_idx_ceil  = invdepth_idx_ceil.long()

        # gather 的 dim=4 就是 D；将同一索引复制到 C 个通道，得到 [B,C,H,W]。
        feat_floor = torch.gather(
            feat_volume, 4, invdepth_idx_floor.repeat(1, ch, 1, 1).unsqueeze(-1)
        )[..., 0]
        feat_ceil = torch.gather(
            feat_volume, 4, invdepth_idx_ceil.repeat(1, ch, 1, 1).unsqueeze(-1)
        )[..., 0]

        return weight_ceil * feat_ceil + weight_floor * feat_floor

    def forward(self, imgs, grids, iters=12, test_mode=False):
        # imgs 顺序由数据集约定为 [前,右,后,左]，每项 [B,1或3,Hi,Wi]。
        # train 返回 (T 个全分辨率索引预测, 初始概率体)；test_mode 仅返回最终索引。
        with autocast(enabled=self.opts.mixed_precision):
            fisheye_feats = self.encoder(imgs)

        fisheye_feats   = [feat.float() for feat in fisheye_feats]
        spherical_feats = self.spherical_sweep(fisheye_feats, grids)

        with autocast(enabled=self.opts.mixed_precision):
            geo_sampler, init_disp, context_feat_volume, init_prob = self.volume_gen(spherical_feats)

        # 实际上下文路径：首轮直接用参考体第 0 个候选切片，未沿 D 做卷积/平均聚合。
        # 后续轮次在当前估计处 volume_sample；这是 Fig. 2 的 Context Feature 在
        # 本仓库中的具体实现，与 §3.1 的“沿逆深度轴聚合”概括应区别阅读。
        context_feat = context_feat_volume[..., 0]

        grid_feat = None
        if self.use_ge:
            grid_xyz = self._build_unit_sphere_grid(
                context_feat.shape[-2], context_feat.shape[-1],
                device=context_feat.device, dtype=context_feat.dtype
            ).expand(context_feat.shape[0], -1, -1, -1)
            with autocast(enabled=self.opts.mixed_precision):
                grid_feat = self.ge(grid_xyz)
                context_feat = context_feat + grid_feat

        with autocast(enabled=self.opts.mixed_precision):
            inp = torch.relu(context_feat)
            net = torch.tanh(self.state_conv(context_feat))  # hidden state
 
        # init_disp、invdepth_idx 都是低分辨率候选轴的连续索引（默认 D=96）。
        Nd = context_feat_volume.shape[-1]
        invdepth_idx = torch.clamp(init_disp, 0, Nd - 1)

        invdepth_idx_predictions = []
        distortion_map = None
        if self.use_digru:
            distortion_map = self._build_distortion_map(
                context_feat.shape[-2], context_feat.shape[-1],
                device=context_feat.device, dtype=context_feat.dtype,
                mode=self.distortion_mode
            ).expand(context_feat.shape[0], -1, -1, -1)
        # PDD 在本次 forward 内只生成一次并复用；跨 forward 未按分辨率缓存。
        for itr in range(iters):
            # 切断经“上一轮索引”回传的梯度；隐藏状态 net 的循环连接仍保留梯度。
            invdepth_idx = invdepth_idx.detach()

            invdepth_idx_safe = torch.nan_to_num(
                invdepth_idx, nan=0.0, posinf=float(Nd - 1), neginf=0.0
            ).clamp(0.0, float(Nd - 1))

            # 从 VE/VC 金字塔各层取当前索引附近的窗口，默认输出 [B,324,H,W]。
            geo_feat = geo_sampler(invdepth_idx_safe)

            if itr > 0:
                context_feat = self.volume_sample(context_feat_volume, invdepth_idx_safe)
                if self.use_ge and grid_feat is not None:
                    context_feat = context_feat + grid_feat
                inp = torch.relu(context_feat)

            with autocast(enabled=self.opts.mixed_precision):
                if self.use_digru:
                    net, delta_invdepth_idx, up_mask = self.update_block(
                        net, inp, geo_feat, invdepth_idx_safe, distortion_map=distortion_map,
                        no_upsample=(test_mode and itr < iters - 1)
                    )
                else:
                    net, delta_invdepth_idx, up_mask = self.update_block(
                        net, inp, geo_feat, invdepth_idx_safe,
                        no_upsample=(test_mode and itr < iters - 1)
                    )

            # 论文式 (9) 后的残差更新：d_t=d_{t-1}+Delta d_t。
            # safe 版本只供查表/更新块输入；此处累加仍使用原来的 invdepth_idx。
            invdepth_idx = invdepth_idx + delta_invdepth_idx

            if up_mask is not None:
                invdepth_idx_up = self.upsample_invdepth_idx(invdepth_idx, up_mask)
                invdepth_idx_predictions.append(invdepth_idx_up)

        # 训练时每轮都上采样以进行 Sequence Loss；测试时只在最后一轮生成掩码。
        # 输出裁剪到完整索引 [0,N-1]，再由数据集换算 rho 与 depth=1/rho。
        if test_mode:
            return torch.clamp(invdepth_idx_predictions[-1], 0, self.opts.num_invdepth - 1)

        if self.training:
            return invdepth_idx_predictions, init_prob
        return invdepth_idx_predictions
