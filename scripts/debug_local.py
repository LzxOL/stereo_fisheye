"""本地小尺寸前向检查：随机输入/人工网格，无需数据集或预训练权重。

用于学习模块连接与张量形状；输出不具有真实深度意义。
默认 CPU，可显式指定 --device mps 检查 Apple GPU 的算子兼容性。
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from easydict import EasyDict

from module.network import PDFOmni


def describe(value):
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(describe(v) for v in value) + "]"
    return type(value).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters 必须至少为 1")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("当前 Python/PyTorch 环境不能使用 MPS，请选择 --device cpu")

    torch.manual_seed(0)
    if args.device == "cpu":
        torch.set_num_threads(4)
    device = torch.device(args.device)
    # 保持论文完整模块和通道数，只缩小空间/候选轴并减少迭代次数。
    # s=2，完整 ERP=32x64、N=32 -> 内部 H=16,W=32,D=16。
    opts = EasyDict(
        base_channel=32, num_invdepth=32, num_downsample=1,
        corr_levels=4, corr_radius=4, mixed_precision=False,
        use_ge=True, use_digru=True, ge_scale=1.0, ge_map_size=256,
        gwc_groups=8, geo_ch=8, acv_concat_ch=16, ot_iter=3,
        gru_small_kernel=1, gru_large_kernel=7, gru_large_mode="sep",
    )
    model = PDFOmni(opts).to(device).eval()
    imgs = [torch.randn(1, 1, 64, 64, device=device) for _ in range(4)]
    yy, xx = torch.meshgrid(
        torch.linspace(-0.8, 0.8, 16, device=device),
        torch.linspace(-0.8, 0.8, 32, device=device), indexing="ij",
    )
    # 人工构造的归一化采样坐标，明确不使用真实相机标定。
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(2).repeat(1, 1, 16, 1)
    grids = [grid.clone() for _ in range(4)]

    def show(name):
        def hook(module, inputs, output):
            print(f"{name}: {describe(output)}", flush=True)
        return hook

    # 钩子只观察输出，不改变模型计算。可直接在原模块源码中设置断点。
    modules = {
        "encoder": model.encoder,
        "front/back weight": model.volume_gen.reference_mapping,
        "right/left weight": model.volume_gen.target_mapping,
        "VE": model.volume_gen.cost_agg,
        "volume generator": model.volume_gen,
        "GE": model.ge,
        "DAA": model.update_block.da_att,
        "DiGRU": model.update_block,
    }
    handles = [module.register_forward_hook(show(name)) for name, module in modules.items()]
    print(f"PyTorch {torch.__version__}; device={device}; synthetic input only", flush=True)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            result = model(imgs, grids, iters=args.iters, test_mode=True)
        if device.type == "mps":
            torch.mps.synchronize()
        assert result.shape == (1, 1, 32, 64), result.shape
        assert torch.isfinite(result).all().item(), "输出含非有限值"
        assert result.min().item() >= 0 and result.max().item() <= 31
        print(f"PASS: output={tuple(result.shape)}, elapsed={time.perf_counter()-started:.2f}s")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
