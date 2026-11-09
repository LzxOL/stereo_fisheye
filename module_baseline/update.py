# The selective recurrent unit (small/large-kernel GRU blending) and the
# channel/spatial attention modules are adapted from Selective-Stereo (MIT):
#   https://github.com/Windsrain/Selective-Stereo
# The recurrent update structure follows RomniStereo / RAFT-Stereo (MIT).

import torch
import torch.nn as nn
import torch.nn.functional as F


# 将隐藏状态 [B,2C,H,W] 回归为 [B,1,H,W] 的“低分辨率逆深度索引增量”。
class DepthHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


# Fig. 4 的通道注意力：空间平均/最大池化 -> 共享两层映射 -> 相加 -> sigmoid。
# 输入 [B,C,H,W]，输出 [B,C,1,1]；用来重标定上下文通道，不直接决定 GRU 核大小。
class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 不启用 DiGRU 时使用的空间注意力；只有通道均值/最大值，没有 PDD 几何输入。
class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)


# Fig. 4、式 (8) 的空间 DAA：输入通道重标定后的上下文和 PDD，输出 [B,1,H,W]。
# 通道注意力由外部 DiGRU.cam 执行；本类仅实现后面的空间注意力部分。
class DistortionAwareAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.samconv = nn.Conv2d(3, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, distortion_map):
        # 在通道维求 avg/max，保留每个 ERP 像素的位置，各得到 [B,1,H,W]。
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        if distortion_map is None:
            distortion_map = torch.zeros_like(avg_out)
        elif distortion_map.shape[-2:] != x.shape[-2:]:
            distortion_map = F.interpolate(distortion_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        distortion_map = distortion_map.clamp(0.0, 1.0)

        # 三个单通道拼为 [B,3,H,W]，7x7 卷积+sigmoid 学习大核分支的混合系数 A。
        # A 受图像特征与 PDD 共同决定；它不是直接将 D 当作权重，也不保证 A 随 D 单调。
        combined = torch.cat([avg_out, max_out, distortion_map], dim=1)
        return self.sigmoid(self.samconv(combined))


# 标准卷积 GRU；小核分支默认 1x1，空间信息主要由输入特征自身携带。
class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) == 1:
                kernel_size = int(kernel_size[0])
            else:
                kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        if isinstance(kernel_size, (list, tuple)):
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            pad = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, h, x):
        # h=[B,hidden_dim,H,W]，x=[B,input_dim,H,W]；输出隐藏状态同 h 形状。
        # z 更新门、r 重置门、q 候选状态；h'=(1-z)*h+z*q。
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        return (1 - z) * h + z * q


# 大核分支先做 1xK 的完整 GRU 更新，再以新 h 做 Kx1 更新（训练入口默认 K=7）。
# 它包含两组 z/r/q 门，不是将两个线性卷积简单相加，也不是普通 KxK GRU。
class RaftSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=5):
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, k), padding=(0, pad))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, k), padding=(0, pad))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, k), padding=(0, pad))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (k, 1), padding=(pad, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (k, 1), padding=(pad, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (k, 1), padding=(pad, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


# Fig. 4、式 (9)：对相同旧状态 h 并行算小/大核候选，再按像素注意力混合。
class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256,
                 small_kernel_size=1, large_kernel_size=5, large_mode="sep"):
        super().__init__()
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        if large_mode == "sep":
            self.large_gru = RaftSepConvGRU(hidden_dim, input_dim, large_kernel_size)
        else:
            self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, x):
        # att=[B,1,H,W] 广播到所有隐藏通道；大核权重 A，小核权重 1-A。
        # 两分支都会执行；注意力调节混合比例，并没有跳过低权重分支的计算。
        h_small = self.small_gru(h, x)
        h_large = self.large_gru(h, x)
        return h_small * (1 - att) + h_large * att


# 将局部匹配证据与当前索引编码成更新特征 mt=[B,128,H,W]。
# geo_feat 不是 xyz 坐标，而是 VE/VC 各层查表窗口的拼接；invdepth 是索引而非物理深度。
class MotionEncoder(nn.Module):
    def __init__(self, cor_planes, c1_planes=64, c2_planes=64, d1_planes=64, d2_planes=64, out_planes=128):
        super(MotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, c1_planes, kernel_size=1, padding=0, bias=True)
        self.convc2 = nn.Conv2d(c1_planes, c2_planes, kernel_size=3, padding=1, bias=True)

        self.convd1 = nn.Conv2d(1, d1_planes, kernel_size=7, padding=3, bias=True)
        self.convd2 = nn.Conv2d(d1_planes, d2_planes, kernel_size=3, padding=1, bias=True)

        self.conv = nn.Conv2d(c2_planes + d2_planes, out_planes - 1, kernel_size=3, padding=1, bias=True)

    def forward(self, geo_feat, invdepth):
        cor = F.relu(self.convc1(geo_feat))
        cor = F.relu(self.convc2(cor))

        dep = F.relu(self.convd1(invdepth))
        dep = F.relu(self.convd2(dep))

        cor_dep = torch.cat([cor, dep], dim=1)
        out = F.relu(self.conv(cor_dep))
        # 保留最后一个原始索引通道：127 个学习通道 + 1 个当前索引，共 128 通道。
        return torch.cat([out, invdepth], dim=1)


# 未启用 use_digru 的更新块：仍包含小/大核混合，仅用外观空间注意力取代 DAA。
# 因而不能将这个代码分支直接等同于论文消融表中的“单个普通 GRU”。
class UpdateBlock(nn.Module):
    def __init__(self, opts, hidden_dim, input_dim):
        super(UpdateBlock, self).__init__()
        self.opts = opts

        L = opts.corr_levels
        r = opts.corr_radius
        # split 为 CV 的 1 通道；joint 为 GEV + CV 的 8+1 通道，与 GWC groups 无关。
        corr_channels = opts.corr_channels
        cor_planes = L * (2 * r + 1) * corr_channels

        self.encoder = MotionEncoder(cor_planes)
        encoder_output_dim = 128

        self.cam = ChannelAttentionEnhancement(input_dim)
        self.sam = SpatialAttentionExtractor(kernel_size=7)

        small_k = getattr(opts, "gru_small_kernel", 1)
        large_k = getattr(opts, "gru_large_kernel", 5)
        large_mode = getattr(opts, "gru_large_mode", "sep")
        self.gru = SelectiveConvGRU(
            hidden_dim,
            encoder_output_dim + input_dim,
            small_kernel_size=small_k,
            large_kernel_size=large_k,
            large_mode=large_mode
        )

        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2 ** self.opts.num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0)
        )

    def forward(self, net, inp, corr=None, inv_depth=None, no_upsample=False):
        geo_feat = corr

        motion_feat = self.encoder(geo_feat, inv_depth)

        cam_w = self.cam(inp)
        inp_cam = inp * cam_w

        att = self.sam(inp_cam)

        gru_inp = torch.cat([inp_cam, motion_feat], dim=1)
        net = self.gru(att, net, gru_inp)

        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        mask = 0.25 * self.mask(net)
        return net, delta_inv_depth, mask


# Fig. 4 的完整更新块：MotionEncoder + 通道注意力 + DAA + 双分支 GRU + 两个输出头。
# 输入 net=[B,2C,H,W]、inp=[B,C,H,W]、corr=[B,L*(2r+1)*(geo_ch+1),H,W]、
# inv_depth=[B,1,H,W]、distortion_map=[B,1,H,W]。
# 返回新 net、索引增量、凸上采样掩码 [B,9*s²,H,W]（或 None）。
class DiGRU(nn.Module):
    def __init__(self, opts, hidden_dim, input_dim):
        super(DiGRU, self).__init__()
        self.opts = opts

        L = opts.corr_levels
        r = opts.corr_radius
        G = getattr(opts, "gwc_groups", 8)
        # 此处同样隐含 geo_ch==gwc_groups；默认 L=4、r=4、G=8，得到 324 通道。
        cor_planes = L * (2 * r + 1) * (G + 1)

        self.encoder = MotionEncoder(cor_planes)
        encoder_output_dim = 128

        self.cam = ChannelAttentionEnhancement(input_dim)
        self.da_att = DistortionAwareAttention(kernel_size=7)

        small_k = getattr(opts, "gru_small_kernel", 1)
        large_k = getattr(opts, "gru_large_kernel", 5)
        large_mode = getattr(opts, "gru_large_mode", "sep")
        self.gru = SelectiveConvGRU(
            hidden_dim,
            encoder_output_dim + input_dim,
            small_kernel_size=small_k,
            large_kernel_size=large_k,
            large_mode=large_mode,
        )

        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2 ** self.opts.num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr=None, inv_depth=None, distortion_map=None, no_upsample=False):
        # 每轮先由当前索引和匹配窗口得到 mt，再重新计算上下文的通道/空间注意力。
        # D 在循环中固定，但 inp 会随深度查表改变，因此注意力 A 也可以逐轮变化。
        geo_feat = corr

        motion_feat = self.encoder(geo_feat, inv_depth)

        cam_w = self.cam(inp)
        inp_cam = inp * cam_w

        # A 只用于混合隐藏状态，不在此直接乘到索引增量或 PDD 本身。
        att = self.da_att(inp_cam, distortion_map)

        # 式 (9) 的 xt=[f_geo重标定,mt]，默认 C=32 时共有 160 个输入通道。
        gru_inp = torch.cat([inp_cam, motion_feat], dim=1)
        net = self.gru(att, net, gru_inp)

        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        mask = 0.25 * self.mask(net)
        return net, delta_inv_depth, mask
