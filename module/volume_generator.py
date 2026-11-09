# The attention-volume construction and 3D hourglass aggregation are adapted
# from ACVNet (MIT): https://github.com/gangweiX/ACVNet
# The geometry encoding volume lookup follows IGEV-Stereo (MIT).

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 本文件对外的 Generator 对应 Fig. 2 的 Volume Generation。
# 注意两种布局：球面/上下文体为 [B,C,H,W,D]；Conv3d 的体为 [B,C,D,H,W]。
class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, deconv=False, bn=True, relu=True):
        super().__init__()
        self.relu = relu
        self.use_bn = bn
        if deconv:
            self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch) if bn else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.leaky_relu(x, inplace=False)
        return x


# 在 D/H/W 三轴上下采样三次并通过跳连恢复原分辨率的 3D 沙漏。
# D 表示逆深度假设；普通情况下三轴尺寸需能匹配三次 stride=2 的跳连。
class Hourglass3D(nn.Module):
    def __init__(self, in_ch=8, base=8, out_ch=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv3d(in_ch, base*2, 3, 2, 1, deconv=False, bn=True, relu=True),
            BasicConv3d(base*2, base*2, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.conv2 = nn.Sequential(
            BasicConv3d(base*2, base*4, 3, 2, 1, deconv=False, bn=True, relu=True),
            BasicConv3d(base*4, base*4, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.conv3 = nn.Sequential(
            BasicConv3d(base*4, base*6, 3, 2, 1, deconv=False, bn=True, relu=True),
            BasicConv3d(base*6, base*6, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.up3 = BasicConv3d(base*6, base*4, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), deconv=True, bn=True, relu=True)
        self.fuse2 = nn.Sequential(
            BasicConv3d(base*8, base*4, 1, 1, 0, deconv=False, bn=True, relu=True),
            BasicConv3d(base*4, base*4, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.up2 = BasicConv3d(base*4, base*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), deconv=True, bn=True, relu=True)
        self.fuse1 = nn.Sequential(
            BasicConv3d(base*4, base*2, 1, 1, 0, deconv=False, bn=True, relu=True),
            BasicConv3d(base*2, base*2, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.up1 = BasicConv3d(base*2, out_ch, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), deconv=True, bn=False, relu=False)

    def forward(self, x):
        # 输入 [B,in_ch,D,H,W]；K=构造参数 base，O=out_ch；以下假设 D/H/W 均为 8 的倍数。
        # baseline 的两个调用均为 K=8、输入 [B,32,96,64,128]；Attention 的 O=32，聚合的 O=8。
        c1 = self.conv1(x)  # [B,2K,D/2,H/2,W/2]；baseline [B,16,48,32,64]
        c2 = self.conv2(c1)  # [B,4K,D/4,H/4,W/4]；baseline [B,32,24,16,32]
        c3 = self.conv3(c2)  # [B,6K,D/8,H/8,W/8]；baseline [B,48,12,8,16]

        u3 = self.up3(c3)  # [B,4K,D/4,H/4,W/4]；baseline [B,32,24,16,32]
        f2 = torch.cat([u3, c2], dim=1)  # [B,8K,D/4,H/4,W/4]；baseline [B,64,24,16,32]
        f2 = self.fuse2(f2)  # [B,4K,D/4,H/4,W/4]；baseline [B,32,24,16,32]
        u2 = self.up2(f2)  # [B,2K,D/2,H/2,W/2]；baseline [B,16,48,32,64]
        f1 = torch.cat([u2, c1], dim=1)  # [B,4K,D/2,H/2,W/2]；baseline [B,32,48,32,64]
        f1 = self.fuse1(f1)  # [B,2K,D/2,H/2,W/2]；baseline [B,16,48,32,64]
        out = self.up1(f1)  # [B,O,D,H,W]；baseline [B,32或8,96,64,128]，取决于调用方
        return out  # [B,O,D,H,W]


def groupwise_corr_5d(ref_vol, tgt_vol, num_groups=8):
    # [B,C,H,W,D] -> [B,G,D,H,W]：C 通道分为 G 组，各组逐元素乘后取均值。
    # 同一个 (y,x,d) 的两特征对应同一个假设三维点，因此无需额外做水平平移。
    # 这里沿通道归约，保留全部 D 候选，并非常规双目全像素 all-pairs 相关。
    B, C, H, W, D = ref_vol.shape
    assert C % num_groups == 0, "channels must be divisible by num_groups"
    g = num_groups
    cpg = C // g

    ref_g = ref_vol.view(B, g, cpg, H, W, D)  # [B,G,C/G,H,W,D]；baseline [B,8,4,64,128,96]
    tgt_g = tgt_vol.view(B, g, cpg, H, W, D)  # [B,G,C/G,H,W,D]
    cost = (ref_g * tgt_g).mean(dim=2)  # [B,G,H,W,D]；baseline [B,8,64,128,96]
    return cost.permute(0, 1, 4, 2, 3).contiguous()  # [B,G,D,H,W]；baseline [B,8,96,64,128]


def channel_corr_5d(ref_vol, tgt_vol):
    # Fig. 2 的 VC：[B,1,D,H,W]。对所有 C 通道做点积求和（不除以 C）。
    cost = (ref_vol * tgt_vol).sum(dim=1, keepdim=True)
    return cost.permute(0, 1, 4, 2, 3).contiguous()


# 编码体构建中的匹配注意力，与 Fig. 4 的畸变注意力 DAA 是两个模块。
# 输入 [B,G,D,H,W]；输出 [B,1,D,H,W] 匹配 logits，尚未做候选轴 softmax。
class ACVAttention(nn.Module):
    def __init__(self, num_groups: int, base_ch: int = 32, att_heads: int = None, use_geo_acv: bool = False):
        super().__init__()
        self.use_geo_acv = use_geo_acv
        G = num_groups + 2 if use_geo_acv else num_groups
        s1 = max(1, G // 4)
        s2 = max(1, G // 2)
        s3 = max(1, G - s1 - s2)
        if s1 + s2 + s3 != G:
            s3 = G - s1 - s2

        self.slices = (s1, s2, s3)

        # (1,3,3) 先只聚合 ERP 邻域；分组通道再使用不同 dilation 扩大空间上下文。
        self.patch_all = nn.Conv3d(G, G, kernel_size=(1,3,3), padding=(0,1,1), groups=G, bias=False)
        self.patch_l1  = nn.Conv3d(s1, s1, kernel_size=(1,3,3), padding=(0,1,1), dilation=(1,1,1), groups=s1, bias=False)
        self.patch_l2  = nn.Conv3d(s2, s2, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=s2, bias=False)
        self.patch_l3  = nn.Conv3d(s3, s3, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=s3, bias=False)

        self.dres1_att = nn.Sequential(
            BasicConv3d(G, base_ch, 3, 1, 1),
            BasicConv3d(base_ch, base_ch, 3, 1, 1)
        )
        self.dres2_att = Hourglass3D(in_ch=base_ch, base=8, out_ch=base_ch)
        self.classif_att = nn.Sequential(
            BasicConv3d(base_ch, base_ch, 3, 1, 1),
            BasicConv3d(base_ch, 1, 3, 1, 1, bn=False, relu=False)
        )

    def forward(self, gwc):
        # baseline 输入 [B,8,96,64,128]；s1/s2/s3=2/4/2，base_ch=32。
        B, G, D, H, W = gwc.shape
        s1, s2, s3 = self.slices

        gwc_s = self.patch_all(gwc)  # [B,G,D,H,W]；baseline [B,8,96,64,128]

        xs = []
        p = 0
        if s1 > 0:
            xs.append(self.patch_l1(gwc_s[:, p:p+s1]))  # 新增 [B,s1,D,H,W]；baseline [B,2,96,64,128]
            p += s1
        if s2 > 0:
            xs.append(self.patch_l2(gwc_s[:, p:p+s2]))  # 新增 [B,s2,D,H,W]；baseline [B,4,96,64,128]
            p += s2
        if s3 > 0:
            xs.append(self.patch_l3(gwc_s[:, p:p+s3]))  # 新增 [B,s3,D,H,W]；baseline [B,2,96,64,128]

        patch_vol = torch.cat(xs, dim=1)  # [B,G,D,H,W]；baseline [B,8,96,64,128]
        feat = self.dres1_att(patch_vol)  # [B,base_ch,D,H,W]；baseline [B,32,96,64,128]
        feat = self.dres2_att(feat)  # [B,base_ch,D,H,W]；内部各层尺寸见 Hourglass3D.forward
        att_logits = self.classif_att(feat)  # [B,1,D,H,W]；baseline [B,1,96,64,128]
        return att_logits  # [B,1,D,H,W]


# Fig. 2 的 Volume Lookup：预建 VE/VC 的 D 轴金字塔，再以当前深度索引查局部窗口。
# 无可训练参数；ERP 空间不降采样，每个像素保留一条独立的候选曲线。
class CombinedGeoEncodingSampler:
    def __init__(self, geo_encoding_volume, init_corr_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        B, Cg, D, H, W = geo_encoding_volume.shape
        self.B, self.Cg, self.D, self.H, self.W = B, Cg, D, H, W

        self.geo_pyrs = []
        self.init_pyrs = []

        # 将 B*H*W 个像素当作 grid_sample 的批次；高度恒为 1，宽度代表 D。
        geo = geo_encoding_volume.permute(0, 3, 4, 1, 2).reshape(B*H*W, Cg, 1, D)
        init = init_corr_volume.permute(0, 3, 4, 1, 2).reshape(B*H*W, 1 , 1, D)

        self.geo_pyrs.append(geo)
        self.init_pyrs.append(init)

        for _ in range(1, num_levels):
            geo = F.avg_pool2d(geo, kernel_size=[1,2], stride=[1,2])
            init = F.avg_pool2d(init, kernel_size=[1,2], stride=[1,2])
            self.geo_pyrs.append(geo)
            self.init_pyrs.append(init)

    def __call__(self, disp_idx, coords=None):
        # disp_idx=[B,1,H,W]；coords 参数在当前实现中未使用。
        # 输出通道数 L*(Cg+1)*(2r+1)：每层拼 VE 的 Cg 通道和 VC 的 1 通道。
        r = self.radius
        B, _, H, W = disp_idx.shape
        assert B == self.B and H == self.H and W == self.W

        flat = disp_idx.reshape(B*H*W, 1, 1, 1)
        windows = torch.linspace(-r, r, 2*r+1, device=disp_idx.device, dtype=disp_idx.dtype).view(1, 1, 2*r+1, 1)

        out_feats = []

        for lvl in range(self.num_levels):
            geo  = self.geo_pyrs[lvl]
            init = self.init_pyrs[lvl]

            # 第 lvl 层候选间隔变为原来的 2^lvl，故中心索引也需除以 2^lvl。
            x0 = (flat / (2**lvl)) + windows

            denom = max(1, geo.shape[-1] - 1)
            xn = 2.0 * x0 / float(denom) - 1.0
            yn = torch.zeros_like(xn)

            base_grid = torch.cat([xn, yn], dim=-1)

            grid_geo  = base_grid.to(dtype=geo.dtype)
            grid_init = base_grid.to(dtype=init.dtype)

            # 将连续索引映射到 [-1,1]，双线性采样在高度 1 时就是 D 轴线性插值。
            # 窗口可越出候选范围；这里使用默认零填充，而非边界值复制。
            geo_samp  = F.grid_sample(geo,  grid_geo,  mode='bilinear', align_corners=True)
            init_samp = F.grid_sample(init, grid_init, mode='bilinear', align_corners=True)

            geo_samp  = geo_samp.view(B, H, W, self.Cg*(2*r+1)).contiguous()
            init_samp = init_samp.view(B, H, W,       (2*r+1)).contiguous()

            feat_lvl = torch.cat([geo_samp, init_samp], dim=-1)
            out_feats.append(feat_lvl)

        out = torch.cat(out_feats, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


# 两层 1x1x1 Conv 实现逐体素 MLP；不混合邻近 ERP 像素或候选深度。
# 对融合层，输入 2C+4 通道，输出 1 个 sigmoid 权重，所有特征通道共享该权重。
class MLP(nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out=1):
        super().__init__()
        self.linear1 = nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv3d(ch_hid, ch_out, (1, 1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.out_act(x)


# [B,C,H,W,D] -> [B,Cout,D,H,W]；用 1x1x1 卷积压缩两侧特征以构建拼接体。
class Compress3D(nn.Module):
    def __init__(self, ch_in, ch_mid=32, ch_out=16):
        super().__init__()
        self.conv1 = nn.Conv3d(ch_in, ch_mid, kernel_size=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm3d(ch_mid)
        self.conv2 = nn.Conv3d(ch_mid, ch_out, kernel_size=1, padding=0, bias=False)
        self.bn2   = nn.BatchNorm3d(ch_out)

    def forward(self, x5d):
        # 输入 [B,ch_in,H,W,D]；baseline ch_in/ch_mid/ch_out=32/32/16。
        x = x5d.permute(0,1,4,2,3).contiguous()  # [B,ch_in,D,H,W]；baseline [B,32,96,64,128]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # [B,ch_mid,D,H,W]；baseline [B,32,96,64,128]
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)  # [B,ch_out,D,H,W]；baseline [B,16,96,64,128]
        return x  # [B,ch_out,D,H,W]


# 先做相反朝向的两两融合，再从 Vref/Vtgt 分出三条路径：
# (1) 匹配注意力加权拼接体 -> VE -> OT 初值；(2) 点积 -> VC；(3) Vref -> context。
class Generator(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        ch_in = opts.base_channel
        self.use_geo_acv = getattr(opts, "use_geo_acv", False)
        self.num_groups = getattr(opts, "gwc_groups", getattr(opts, "num_groups", 8))
        self.acv_concat_ch = getattr(opts, "acv_concat_ch", 16)
        self.geo_ch = getattr(opts, "geo_ch", 8)
        self.initcorr_ch = 1

        self.reference_mapping = MLP(2*ch_in+4, ch_in)
        self.target_mapping    = MLP(2*ch_in+4, ch_in)

        self.att_branch = ACVAttention(num_groups=self.num_groups, base_ch=32, use_geo_acv=self.use_geo_acv)
        self.ref_compress = Compress3D(ch_in, ch_mid=max(16, self.acv_concat_ch*2), ch_out=self.acv_concat_ch)
        self.tgt_compress = Compress3D(ch_in, ch_mid=max(16, self.acv_concat_ch*2), ch_out=self.acv_concat_ch)

        acv_in_ch = 2 * self.acv_concat_ch
        self.cost_agg = Hourglass3D(in_ch=acv_in_ch, base=8, out_ch=self.geo_ch)
        self.classifier = nn.Conv3d(self.geo_ch, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.ot_iter = getattr(opts, "ot_iter", 3)

    @staticmethod
    def _concat_every_other(feats):
        # 偶/奇索引分组的辅助函数；forward 目前直接写切片，没有调用本函数。
        return torch.cat(feats[0::2], dim=1), torch.cat(feats[1::2], dim=1)

    @staticmethod
    def _logsumexp_stable(x: torch.Tensor, dim: int, keepdim: bool = False, eps: float = 1e-30) -> torch.Tensor:
        # log(sum(exp(x))) 先减最大值，避免 Sinkhorn 中指数运算溢出。
        m, _ = x.max(dim=dim, keepdim=True)
        y = (x - m).exp().sum(dim=dim, keepdim=True)
        y = m + torch.log(torch.clamp(y, min=eps))
        return y if keepdim else y.squeeze(dim)

    def _sinkhorn(self, logits: torch.Tensor, iters: int) -> torch.Tensor:
        # [B,D,H,W] -> [B,D,H*W]：每张 ERP 的深度候选与所有像素组成分配矩阵。
        # 交替缩放行/列，使候选边缘质量约为 1/D，像素边缘质量约为 1/(H*W)。
        # 这不是每个像素独立的 softmax；u 的更新使用全图像素的分布信息。
        # 返回 log 域运输质量；当前实现没有单独的温度/熵正则系数参数。
        B, D, H, W = logits.shape
        N = H * W

        log_q = logits.reshape(B, D, N)
        log_mu = torch.full((B, D, 1), -math.log(max(D, 1)), device=logits.device, dtype=logits.dtype)
        log_nu = torch.full((B, 1, N), -math.log(max(N, 1)), device=logits.device, dtype=logits.dtype)

        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(max(int(iters), 1)):
            u = log_mu - self._logsumexp_stable(log_q + v, dim=2, keepdim=True)
            v = log_nu - self._logsumexp_stable(log_q + u, dim=1, keepdim=True)

        return (log_q + u + v).reshape(B, D, H, W)

    def forward(self, spherical_feats):
        # 列表严格为 [F前,F右,F后,F左,G前,G右,G后,G左]。
        # 0::2 取 [F前,F后,G前,G后]，1::2 取 [F右,F左,G右,G左]；
        # 每个拼接张量为 [B,2C+4,H,W,D]，坐标 G 是归一化鱼眼采样位置。
        front_cat = torch.cat(spherical_feats[0::2], dim=1)
        right_cat = torch.cat(spherical_feats[1::2], dim=1)  # 1,3,5,...

        # 分别学习 a/b=[B,1,H,W,D]，每个位置、每个深度假设都有自己的权重。
        # Vref=a*F前+(1-a)*F后；Vtgt=b*F右+(1-b)*F左。
        # 这不是将四相机直接平均，也不是选一张 ERP 图作为固定参考图。
        # 重叠区两相机均有特征，MLP 结合特征与坐标决定比例；视场外 Fi 通常为 0，
        # 但没有显式 mask 强制 a=0/1，也没有按可见相机数重新归一化。
        # 因而只有一侧可见时仍可能被权重衰减；两侧均越界为零时融合结果也为零。
        # 这里未根据预测深度显式检测遮挡；视场可见性不等于遮挡可见性。

        # 球面扫描得到的（fea+几何）输入 MLP 得到融合权重。
        front_weight = self.reference_mapping(front_cat)
        reference_feat = front_weight * spherical_feats[0] + (1.0 - front_weight) * spherical_feats[2]

        right_weight  = self.target_mapping(right_cat)
        target_feat   = right_weight * spherical_feats[1] + (1.0 - right_weight) * spherical_feats[3]

        # 保留所有 D 候选的前后融合体供迭代查上下文，不在此压缩候选轴。
        context_feat_volume = reference_feat

        gwc = groupwise_corr_5d(reference_feat, target_feat, num_groups=self.num_groups)
        # 可选附加 ERP 经纬角坐标；默认训练命令不启用，它既不是 Gi，也不是 GE/PDD。
        if self.use_geo_acv:
            _, _, D, H, W = gwc.shape
            phi = torch.linspace(-math.pi / 2, math.pi / 2, H, device=gwc.device, dtype=gwc.dtype)
            lam = torch.linspace(-math.pi, math.pi, W, device=gwc.device, dtype=gwc.dtype)
            grid_phi, grid_lam = torch.meshgrid(phi, lam, indexing="ij")
            coord = torch.stack([grid_phi, grid_lam], dim=0).unsqueeze(0).unsqueeze(2)
            coord = coord.expand(gwc.shape[0], -1, D, -1, -1)
            gwc_att = torch.cat([gwc, coord], dim=1)
        else:
            gwc_att = gwc

        att_logits = self.att_branch(gwc_att)
        # dim=2 为 D：每个 ERP 像素的候选注意力和为 1。
        att_prob   = F.softmax(att_logits, dim=2)

        ref_c = self.ref_compress(reference_feat)
        tgt_c = self.tgt_compress(target_feat)
        concat_vol = torch.cat([ref_c, tgt_c], dim=1)

        # [B,1,D,H,W] 广播到 [B,2*acv_concat_ch,D,H,W]，突出可信深度的两侧特征。
        # 真正送入 cost_agg 的是“加权拼接体”，并非直接将 GWC 当作 VE。
        acv = att_prob * concat_vol

        geo_encoding_volume = self.cost_agg(acv)

        # VE=[B,geo_ch,D,H,W]，3D 分类头得到每个候选的分数。
        # OT 得到全图耦合质量，再按 D 归一化成每像素概率，供初值回归及 LEC。
        logits = self.classifier(geo_encoding_volume).squeeze(1)
        ot_log_prob = self._sinkhorn(logits, self.ot_iter)
        init_prob = torch.exp(ot_log_prob)
        init_prob = init_prob / (init_prob.sum(dim=1, keepdim=True) + 1e-9)

        Nd = init_prob.shape[1]
        disp_values = torch.arange(0, Nd, device=init_prob.device, dtype=init_prob.dtype).view(1, Nd, 1, 1)
        # softargmax/期望：d0=sum_d P(d)*d，输出 [B,1,H,W] 连续候选索引。
        # 此处回归索引而非 sum P(d)/rho_d，后续才能沿等间距逆深度轴取窗口。
        init_disp = (init_prob * disp_values).sum(dim=1, keepdim=True)

        init_corr = channel_corr_5d(reference_feat, target_feat)

        geo_sampler = CombinedGeoEncodingSampler(
            geo_encoding_volume=geo_encoding_volume,
            init_corr_volume=init_corr,
            num_levels=getattr(self.opts, "corr_levels", 2),
            radius=getattr(self.opts, "corr_radius", 4),
        )

        # 四项分别交给：循环查匹配证据、初始化索引、动态上下文采样、训练 LEC。
        return geo_sampler, init_disp, context_feat_volume, init_prob
