"""GWC 引导的 Attention Concatenation Volume 与 3D 聚合。"""

import torch
import torch.nn as nn

from module.volume_generator import ACVAttention, Compress3D, Hourglass3D, groupwise_corr_5d


def masked_softmax(logits, valid_mask, dim=2):
    # 本网络调用时 logits/mask=[B,1,D,H,W]；dim=2 为候选轴。
    # 全无效像素不能直接对一排 -inf 做 softmax（会产生 NaN）。
    # 临时改为全零 logits，归一化后再将所有无效候选置零，得到零概率与零梯度。
    logits = logits.float()  # [B,1,D,H,W]，float32
    any_valid = valid_mask.any(dim=dim, keepdim=True)  # bool [B,1,1,H,W]（dim=2）
    masked = logits.masked_fill(~valid_mask, -torch.inf)  # [B,1,D,H,W]
    masked = torch.where(any_valid, masked, torch.zeros_like(masked))  # [B,1,D,H,W]
    probability = torch.softmax(masked, dim=dim)  # [B,1,D,H,W]
    return torch.where(valid_mask, probability, 0.0)  # [B,1,D,H,W]


class StereoVolumeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # C=32，GWC 分 8 组；左右各压到 16 通道，拼接后为 32 通道。
        self.num_groups = 8
        self.attention = ACVAttention(num_groups=8, base_ch=32, use_geo_acv=False)
        self.left_compress = Compress3D(ch_in=32, ch_mid=32, ch_out=16)
        self.right_compress = Compress3D(ch_in=32, ch_mid=32, ch_out=16)
        self.aggregation = Hourglass3D(in_ch=32, base=8, out_ch=8)
        self.classifier = nn.Conv3d(8, 1, 3, padding=1, bias=False)

    def forward(self, left_volume, right_volume, valid_mask, return_intermediates=False):
        # B=批大小；默认 C=32、D=96、H=64、W=128；mask=[B,1,D,H,W]。
        # 本模块统一接收 [B,C,D,H,W]。旧版 GWC/Compress3D 接收 [B,C,H,W,D]，
        # 因此仅在调用这些已有函数的边界转换维度；不修改原有组件。
        left_hwd = left_volume.permute(0, 1, 3, 4, 2)  # [B,32,H,W,D]；默认 [B,32,64,128,96]
        right_hwd = right_volume.permute(0, 1, 3, 4, 2)  # [B,32,H,W,D]
        # 1. 分组相关性：[B,8,D,H,W]。
        gwc = groupwise_corr_5d(left_hwd, right_hwd, self.num_groups)  # [B,8,D,H,W]；默认 [B,8,96,64,128]
        gwc = torch.where(valid_mask, gwc, 0.0)  # [B,8,D,H,W]，mask 沿 8 组广播

        # 2. 匹配分数沿候选轴归一化，得到 A=[B,1,D,H,W]。
        att_logits = self.attention(gwc)  # [B,1,D,H,W]；默认 [B,1,96,64,128]
        att_prob = masked_softmax(att_logits, valid_mask)  # [B,1,D,H,W]

        # 3. 左右各压到 16 通道，再拼接为 [B,32,D,H,W]。
        left_c = self.left_compress(left_hwd)  # [B,16,D,H,W]；默认 [B,16,96,64,128]
        right_c = self.right_compress(right_hwd)  # [B,16,D,H,W]
        concat_vol = torch.cat([left_c, right_c], dim=1)  # [B,32,D,H,W]；默认 [B,32,96,64,128]
        concat_vol = torch.where(valid_mask, concat_vol, 0.0)  # [B,32,D,H,W]

        # 4. A 对拼接体加权，随后 3D 沙漏输出 8 通道 GEV。
        acv = att_prob * concat_vol  # [B,32,D,H,W]，注意力沿 32 通道广播
        geo_encoding_volume = self.aggregation(acv)  # [B,8,D,H,W]；默认 [B,8,96,64,128]

        # 5. 分类头输出代价：代价越小，概率越高。
        cost = self.classifier(geo_encoding_volume)  # [B,1,D,H,W]；默认 [B,1,96,64,128]
        probability = masked_softmax(-cost, valid_mask)  # [B,1,D,H,W]

        # 与原 Generator 一样直接返回所需张量；调试时再附带中间结果。
        intermediates = None
        if return_intermediates:
            intermediates = {
                "left_volume": left_volume, "right_volume": right_volume,  # 各 [B,32,D,H,W]
                "gwc": gwc, "attention_logits": att_logits,  # [B,8,D,H,W]、[B,1,D,H,W]
                "left_compressed": left_c, "right_compressed": right_c,  # 各 [B,16,D,H,W]
                "concat_volume": concat_vol, "acv": acv,  # 各 [B,32,D,H,W]
                "geometry_volume": geo_encoding_volume, "cost": cost,  # [B,8,D,H,W]、[B,1,D,H,W]
                "candidate_valid_mask": valid_mask,  # bool [B,1,D,H,W]
            }
        return probability, att_prob, intermediates  # 前两项各 [B,1,D,H,W]；第三项为 dict/None
