# Adapted from the official S2M2 implementation (with modifications):
#   "S2M2: Scalable Stereo Matching Model for Reliable Depth Estimation", ICCV 2025
#   Junhong Min, Youngpil Jeon, Jimin Kim, Minyong Choi
#   https://github.com/junhong-3dv/s2m2 (CC BY-NC 4.0, non-commercial use only)

import math
import torch
import torch.nn.functional as F


def custom_sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        (torch.sin(3.1415 * x) / (3.1415 * x)).to(x.dtype),
    )


def get_pe(h: int, w: int, pe_dim: int, dtype, device, mode: str = "grid"):
    # 鱼眼特征的相对位置 PE：[h*w,h*w,pe_dim]，前两维对应 query/key token。
    # 默认 grid 模式分别编码相对行、列位移；radial 是另一个可选坐标模式。
    # 不参与鱼眼到 ERP 的几何投影，投影由标定 LUT 独立处理。
    if mode not in ("grid", "radial"):
        raise ValueError(f"Unknown PE mode: {mode}")
    with torch.no_grad():
        if mode == "radial":
            y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
            x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

            r = torch.sqrt(grid_x**2 + grid_y**2)
            theta = torch.atan2(grid_y, grid_x)

            pos_x = r * (w / 2)
            pos_y = (theta / math.pi) * (h / 2)
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h - 1, h, device=device, dtype=dtype),
                torch.linspace(0, w - 1, w, device=device, dtype=dtype),
                indexing="ij",
            )
            pos_x = grid_x
            pos_y = grid_y

        # 枚举所有 token 对的位移，而不是只为每个 token 存一个绝对位置向量。
        rel_x_pos = pos_x.reshape(-1, 1) - pos_x.reshape(1, -1)
        rel_y_pos = pos_y.reshape(-1, 1) - pos_y.reshape(1, -1)
        if mode == "radial":
            rel_x_pos = rel_x_pos.round().clamp(min=-(w - 1), max=(w - 1)).long()
            rel_y_pos = rel_y_pos.round().clamp(min=-(h - 1), max=(h - 1)).long()
        else:
            rel_x_pos = rel_x_pos.long()
            rel_y_pos = rel_y_pos.long()

        L = 2 * w + 1
        sig = 5 / pe_dim
        x_pos = torch.linspace(-3, 3, L).to(device).to(dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim // 2).to(device).to(dtype)
        pe_x = custom_sinc((dim_t[None, :] - x_pos[:, None]) / sig)

        pe_x = F.normalize(pe_x, p=2, dim=-1)
        rel_pe_x = pe_x[rel_x_pos + w - 1].reshape(h * w, h * w, pe_dim // 2).to(dtype)

        L = 2 * h + 1
        sig = 5 / pe_dim
        y_pos = torch.linspace(-3, 3, L).to(device).to(dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim // 2).to(device).to(dtype)
        pe_y = custom_sinc((dim_t[None, :] - y_pos[:, None]) / sig)

        pe_y = F.normalize(pe_y, p=2, dim=-1)
        rel_pe_y = pe_y[rel_y_pos + h - 1].reshape(h * w, h * w, pe_dim // 2).to(dtype)

        # 将水平/垂直的 sinc 基函数嵌入拼接，交给 SelfAttn 按注意力权重聚合。
        pe = 0.5 * torch.cat([rel_pe_x, rel_pe_y], dim=2)

    return pe.clone()
