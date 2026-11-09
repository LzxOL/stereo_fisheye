from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _ensure_bool_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    assert mask.shape == like.shape, f"mask shape {mask.shape} != target {like.shape}"
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return mask


def _finite_only(x: torch.Tensor, where: torch.Tensor) -> None:
    if torch.isinf(x[where]).any() or torch.isnan(x[where]).any():
        raise ValueError("Non-finite values found inside valid region.")


def sequence_loss(preds, gt, valid, loss_gamma: float = 0.9, use_smooth_l1: bool = False) -> torch.Tensor:
    # 论文 §3.4、式 (15)：监督每轮全分辨率预测，越靠后权重越大。
    # preds 是 T 个 [B,1,Hout,Wout] 索引图；gt/valid 同形状，GT 也必须是完整轴索引。
    n_predictions = len(preds)
    assert n_predictions >= 1, "At least one prediction required"

    # 实际折扣 gamma_eff=gamma^(15/T)，并非式 (15) 字面上的固定 gamma。
    # 默认 T=12 时 gamma_eff=0.9^1.25；用于让不同循环长度的总折扣跨度接近。
    adjusted = loss_gamma ** (15.0 / n_predictions)

    loss = 0.0
    mask = _ensure_bool_mask(valid, gt)
    _finite_only(gt, mask)

    for i in range(n_predictions):
        pi = preds[i]
        assert pi.shape == gt.shape, f"pred[{i}] shape {pi.shape} != gt {gt.shape}"
        _finite_only(pi, mask)

        # 仅在有效 GT 像素求平均，默认 L1；smooth_l1 是接口保留的可选路径。
        w = adjusted ** (n_predictions - i - 1)
        if use_smooth_l1:
            li = F.smooth_l1_loss(pi[mask], gt[mask], reduction="mean")
        else:
            li = (pi - gt).abs()[mask].mean()
        loss = loss + w * li

    return loss


def lec_loss(
    prob_volume: torch.Tensor,
    gt_disp: torch.Tensor,
    img: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
    detach_entropy: bool = True,
    full_num_invdepth=None,
    return_stats: bool = False,
) -> torch.Tensor:

    # 论文式 (10)-(14)：直接监督 OT 初始化概率体，鼓励概率落在 GT 的自适应邻域。
    # prob_volume=[B,D,H,W]；gt_disp/valid_mask 为完整 ERP 尺寸，可带或不带通道轴。
    # img 参数在当前实现中未使用；本损失不包含光度一致性或 RGB 边缘项。
    # return_stats=True 时另返回熵/半径/邻域质量等统计，供训练日志显示。
    if gt_disp.ndim == 3:
        gt_disp = gt_disp.unsqueeze(1)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)

    B, Nd, H, W = prob_volume.shape

    prob = torch.clamp(prob_volume, min=eps)
    prob = prob / (prob.sum(dim=1, keepdim=True) + eps)
    # 每个像素的候选分布熵除以最大熵 log(D)，得到 [0,1] 的不确定性度量。
    entropy = -(prob * torch.log(prob + eps)).sum(dim=1, keepdim=True)
    h_norm = entropy / max(math.log(Nd), eps)
    h_norm = torch.clamp(h_norm, 0.0, 1.0)
    # 半径路径停止梯度，避免把增大不确定性作为放宽监督的可学习捷径。
    if detach_entropy:
        h_norm = h_norm.detach()

    # 式 (11)：r=max(1,ceil(H_norm*(D-1)))。高熵时邻域更宽；最低半径仍为 1。
    # 均匀分布时 r=D-1，整个候选轴都属于邻域，LEC 的约束会相应变弱。
    radius = torch.ceil(h_norm * float(Nd - 1)).to(dtype=torch.long)
    radius = torch.clamp(radius, min=1)

    factor_h = max(1, gt_disp.shape[-2] // H)
    factor_w = max(1, gt_disp.shape[-1] // W)
    # GT 按空间块平均降采样；valid 用 min pooling（-max(-mask)）降采样，
    # 一个块只要含任意无效 GT，该低分辨率像素就不参与 LEC。
    gt_down = F.avg_pool2d(gt_disp, kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w))
    mask_down = -F.max_pool2d(
        -valid_mask.float(), kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w)
    )

    if full_num_invdepth is None:
        full_num_invdepth = Nd
    gt_to_volume_scale = float(full_num_invdepth) / float(Nd)
    # 先从完整索引单位除以 N/D（默认 192/96=2），再四舍五入为候选体内的 GT 中心。
    # 这一步是候选轴缩放，与前面的空间池化是两个独立操作。
    gt_idx = torch.round(gt_down / gt_to_volume_scale).clamp(0, Nd - 1).to(dtype=torch.long)
    d_indices = torch.arange(Nd, device=prob_volume.device, dtype=torch.long).view(1, Nd, 1, 1)
    dist = (d_indices - gt_idx).abs()
    neigh = (dist <= radius).to(dtype=prob_volume.dtype)

    # 式 (12)-(14)：neigh 是 |d-d_gt|<=r 的布尔邻域；mass 为其中概率和，
    # 损失 max(0,1-mass) 在有效像素求均值。不是交叉熵 -log P(d_gt)。
    mass = (prob * neigh).sum(dim=1, keepdim=True)
    loss_map = F.relu(1.0 - mass)

    denom = mask_down.sum() + 1e-6
    loss = (loss_map * mask_down).sum() / denom
    if not return_stats:
        return loss

    with torch.no_grad():
        valid_down = mask_down > 0.5
        if valid_down.any():
            entropy_mean = entropy[valid_down].mean()
            radius_mean = radius.float()[valid_down].mean()
            mass_mean = mass[valid_down].mean()
        else:
            entropy_mean = entropy.new_tensor(0.0)
            radius_mean = entropy.new_tensor(0.0)
            mass_mean = entropy.new_tensor(0.0)
        stats = {
            "entropy": entropy_mean.detach(),
            "radius": radius_mean.detach(),
            "mass": mass_mean.detach(),
            "gt_to_volume_scale": entropy.new_tensor(gt_to_volume_scale),
        }
    return loss, stats
