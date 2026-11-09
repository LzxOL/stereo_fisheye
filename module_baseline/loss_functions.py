from __future__ import annotations
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
    # 只做数值回归，越靠后的预测权重越大，不监督概率分布。
    # preds 是若干 [B,1,H,W] 预测，gt/valid 同形状；预测与真值必须使用同一种单位。
    # 当前网络输出工作网格的候选索引；若监督物理深度，调用前需先将预测转换为深度。
    n_predictions = len(preds)
    assert n_predictions >= 1, "At least one prediction required"

    # 实际折扣 gamma_eff=gamma^(15/T)，并非式 (15) 字面上的固定 gamma。
    # 默认 T=12 时 gamma_eff=0.9^1.25；用于让不同循环长度的总折扣跨度接近。
    adjusted = loss_gamma ** (15.0 / n_predictions)

    loss = 0.0
    mask = _ensure_bool_mask(valid, gt)
    _finite_only(gt, mask)
    target = gt[mask].float()
    num_valid = mask.sum().clamp_min(1)

    for i in range(n_predictions):
        pi = preds[i]
        assert pi.shape == gt.shape, f"pred[{i}] shape {pi.shape} != gt {gt.shape}"
        _finite_only(pi, mask)

        # 仅在有效 GT 像素求平均，默认 L1；smooth_l1 是接口保留的可选路径。
        w = adjusted ** (n_predictions - i - 1)
        prediction = pi[mask].float()
        if use_smooth_l1:
            li = F.smooth_l1_loss(prediction, target, reduction="sum") / num_valid
        else:
            li = (prediction - target).abs().sum() / num_valid
        # 先选有效值再做减法，屏蔽无效 GT 的 NaN；全无效时得到可反传的零损失。
        loss = loss + w * li

    return loss


def stereo_loss(result, gt_index, gt_valid, lec_weight=.1, loss_gamma=.9, supervise_init=False):
    """使用本工作轴单位监督，不做 192/96 的隐式换算。"""
    valid = gt_valid.bool() & result["valid_mask"]
    predictions = result["disp_predictions"]
    if supervise_init or not predictions:
        predictions = [result["init_disp"], *predictions]
    loss_seq = sequence_loss(predictions, gt_index, valid, loss_gamma=loss_gamma)
    loss_lec = result["init_prob"].sum() * 0.0
    if lec_weight:
        # 直接复用原版 LEC；GT 已是 D 层索引、同空间尺寸，所以 full_num_invdepth=D。
        # 先清理无效 GT，避免原版池化/round 路径遇到 NaN。
        from module.loss_functions import lec_loss
        safe_gt = torch.where(valid, gt_index, 0.0)
        loss_lec = lec_loss(result["init_prob"].float(), safe_gt, None, valid,
                            full_num_invdepth=result["init_prob"].shape[1])
    return loss_seq + lec_weight * loss_lec, {"sequence_loss": loss_seq, "lec_loss": loss_lec}
