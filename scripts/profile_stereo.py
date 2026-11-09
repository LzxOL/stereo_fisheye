#!/usr/bin/env python3
"""测量真实模型在人工网格上的计算/显存；不代表真实数据几何或深度精度。"""

import argparse
import gc
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module_baseline import StereoBaseline
from module_baseline.config import load_config
from module_baseline.debug import ShapeDebug
from module_baseline.loss_functions import stereo_loss
from module_baseline.training import check_device, write_json


def benchmark(args, size, mode, precision, use_checkpoint):
    device = check_device(args.device)
    model = StereoBaseline(num_depth=args.num_depth, sweep_size=size, volume_update_mode=mode,
                           mixed_precision=args.mixed_precision and device.type == "cuda",
                           sweep_precision=precision, gradient_checkpointing=use_checkpoint).to(device).train()
    height, width = size
    y, x = torch.meshgrid(torch.linspace(-.8, .8, height, device=device),
                          torch.linspace(-.8, .8, width, device=device), indexing="ij")
    left = torch.stack([x, y], dim=-1)[None]
    right = left.unsqueeze(3).expand(-1, -1, -1, args.num_depth, -1).contiguous()
    images = [torch.randn(1, 3, *args.image_size, device=device) for _ in range(2)]
    valid = torch.ones(1, 1, args.num_depth, height, width, device=device, dtype=torch.bool)
    target = torch.full((1, 1, height, width), (args.num_depth - 1) / 2, device=device)
    if args.shape_debug:
        trace = ShapeDebug(model)
        try:
            with torch.no_grad():
                out = model(images, [left, right], valid, iters=args.iters, return_intermediates=True)
                trace.record_output(out)
                write_json(Path(args.output).with_suffix("").parent /
                           f"shapes_{height}x{width}_{mode}_{precision}_ckpt{use_checkpoint}.json", trace.records)
                del out
        finally:
            trace.close()
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    forwards, backwards = [], []
    forward_peaks, backward_peaks = [], []
    actual_dtype = None
    dtype_hook = model.volume_gen.register_forward_pre_hook(
        lambda module, inputs: setattr(module, "profile_sweep_dtype", str(inputs[0][0].dtype)))
    try:
        for repeat in range(args.warmup + args.repeats):
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            out = model(images, [left, right], valid, iters=args.iters)
            loss, _ = stereo_loss(out, target, valid.any(2), args.lec_weight)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_time = time.perf_counter() - start
            f_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            scaler.scale(loss).backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_time = time.perf_counter() - start
            b_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            actual_dtype = model.volume_gen.profile_sweep_dtype
            if repeat >= args.warmup:
                forwards.append(forward_time)
                backwards.append(backward_time)
                forward_peaks.append(f_peak)
                backward_peaks.append(b_peak)
            del out, loss
    finally:
        dtype_hook.remove()
    return dict(size=list(size), depth=args.num_depth, image_size=args.image_size, batch_size=1,
                mode=mode, sweep_precision=precision, actual_sweep_dtype=actual_dtype, grid_dtype=str(left.dtype),
                amp=args.mixed_precision and device.type == "cuda", gradient_checkpointing=use_checkpoint,
                forward_seconds=sum(forwards) / len(forwards), backward_seconds=sum(backwards) / len(backwards),
                forward_peak_allocated_mib=max(forward_peaks) if device.type == "cuda" else None,
                backward_peak_allocated_mib=max(backward_peaks) if device.type == "cuda" else None,
                backward_peak_includes_live_forward_graph=True)


def main():
    config = load_config()
    formal = config["data"]["sweep_size"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["64x128", "64x256", "80x320", f"{formal[0]}x{formal[1]}"])
    parser.add_argument("--num_depth", type=int, default=config["data"]["num_depth"])
    parser.add_argument("--image_size", type=int, nargs=2, default=config["data"]["image_size"])
    parser.add_argument("--modes", nargs="+", choices=["joint", "split"], default=["split"])
    parser.add_argument("--sweep_precisions", nargs="+", choices=["float32", "amp"], default=["float32", "amp"])
    parser.add_argument("--compare_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iters", type=int, default=config["train"]["train_iters"])
    parser.add_argument("--lec_weight", type=float, default=config["train"]["lec_weight"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shape_debug", action="store_true")
    parser.add_argument("--output", default="results/stereo_profile.json")
    args = parser.parse_args()
    try:
        check_device(args.device)
        if args.repeats < 1 or args.warmup < 0:
            raise ValueError("repeats must be positive and warmup non-negative")
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(input_kind="synthetic shape/profiling grid, not calibrated OmniThings data",
                  torch_version=torch.__version__, device=args.device, results=[])
    for text in args.sizes:
        size = tuple(map(int, text.lower().split("x")))
        for mode in args.modes:
            for precision in args.sweep_precisions:
                for use_checkpoint in ([False, True] if args.compare_checkpointing else [config["model"]["gradient_checkpointing"]]):
                    try:
                        record = benchmark(args, size, mode, precision, use_checkpoint)
                    except torch.OutOfMemoryError as exc:
                        record = dict(size=list(size), mode=mode, sweep_precision=precision,
                                      gradient_checkpointing=use_checkpoint, error="out_of_memory", detail=str(exc))
                    report["results"].append(record)
                    write_json(args.output, report)
                    print(record)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
