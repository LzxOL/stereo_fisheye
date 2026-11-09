"""独立双目训练/评估的公共参数和小型工具，不改四目训练入口。"""

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import DEFAULT_CONFIG, load_config
from .network import StereoBaseline
from .omnithings_stereo_dataset import OmniThingsStereoDataset


def training_parser():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = preliminary.parse_known_args()
    config = load_config(known.config)
    data, model, train = config["data"], config["model"], config["train"]
    parser = argparse.ArgumentParser(description="OmniThings stereo split-volume pretraining", parents=[preliminary])
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--root", default=data["root"], help="Direct OmniThings dataset directory")
    roots.add_argument("--db_root", help="PDF-Omni parent dataset directory; appends /omnithings")
    parser.add_argument("--ref_cam", choices=["cam1", "cam2", "cam3", "cam4"])
    parser.add_argument("--target_cam", choices=["cam1", "cam2", "cam3", "cam4"])
    parser.add_argument("--num_depth", type=int, default=data["num_depth"])
    parser.add_argument("--sweep_size", type=int, nargs=2, default=data["sweep_size"])
    parser.add_argument("--image_size", type=int, nargs=2, default=data["image_size"])
    for key in ("sweep_azimuth_center_deg", "sweep_azimuth_span_deg", "sweep_elevation_min_deg", "sweep_elevation_max_deg"):
        parser.add_argument("--" + key, type=float, default=data[key])
    parser.add_argument("--gt_width", type=int)
    parser.add_argument("--gt_phi", type=float)
    parser.add_argument("--check_occlusion", action=argparse.BooleanOptionalAction, default=data["check_occlusion"])
    parser.add_argument("--occlusion_abs", type=float, default=.01)
    parser.add_argument("--occlusion_rel", type=float, default=.01)
    parser.add_argument("--val_count", type=int, default=data["val_count"])
    parser.add_argument("--use_rgb", action=argparse.BooleanOptionalAction, default=data["use_rgb"])
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=data["augment"],
                        help="Reuse PDF-Omni photo_aug with shared parameters for the two views")
    parser.add_argument("--volume_update_mode", choices=["joint", "split"], default=model["volume_update_mode"])
    for key in ("corr_levels", "corr_radius", "ot_iter"):
        parser.add_argument("--" + key, type=int, default=model[key])
    for key in ("gru_small_kernel", "gru_large_kernel"):
        parser.add_argument("--" + key, type=int, default=model[key])
    parser.add_argument("--gru_large_mode", choices=["sep", "conv"], default=model["gru_large_mode"])
    for key in ("share_encoder", "gradient_checkpointing"):
        parser.add_argument("--" + key, action=argparse.BooleanOptionalAction, default=model[key])
    parser.add_argument("--sweep_precision", choices=["float32", "amp"], default=model["sweep_precision"])
    parser.add_argument("--epochs", "--total_epochs", dest="epochs", type=int, default=train["epochs"])
    for key in ("batch_size", "train_iters", "valid_iters", "num_workers", "seed"):
        parser.add_argument("--" + key, type=int, default=train[key])
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=train["learning_rate"])
    parser.add_argument("--weight_decay", "--wdecay", dest="weight_decay", type=float, default=train["weight_decay"])
    for key in ("lec_weight", "grad_clip", "loss_gamma"):
        parser.add_argument("--" + key, type=float, default=train[key])
    parser.add_argument("--optimizer", choices=["adamw"], default=train["optimizer"])
    parser.add_argument("--scheduler", choices=["onecycle"], default="onecycle")
    parser.add_argument("--supervise_init", action=argparse.BooleanOptionalAction, default=train["supervise_init"])
    parser.add_argument("--fix_bn", action="store_true")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=train["mixed_precision"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="checkpoints/stereo_omnithings")
    parser.add_argument("--name", default="StereoBaseline")
    parser.add_argument("--restore_ckpt")
    parser.add_argument("--pretrain_ckpt", help="Weights-only initialization; requires a compatible StereoBaseline checkpoint")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means no explicit step limit")
    parser.add_argument("--smoke_test", action="store_true", help="Only one real-data optimizer step, no validation")
    parser.add_argument("--shape_debug", action="store_true")
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(args, split):
    angles = [args.sweep_azimuth_center_deg, args.sweep_azimuth_span_deg,
              args.sweep_elevation_min_deg, args.sweep_elevation_max_deg]
    return OmniThingsStereoDataset(args.root, args.ref_cam, args.target_cam, split=split,
                                   num_depth=args.num_depth, sweep_size=args.sweep_size, angles=angles,
                                   image_size=args.image_size, val_count=args.val_count, seed=args.seed,
                                   use_rgb=args.use_rgb, augment=args.augment, gt_width=args.gt_width,
                                   gt_phi=args.gt_phi, check_occlusion=args.check_occlusion,
                                   occlusion_abs=args.occlusion_abs, occlusion_rel=args.occlusion_rel)


def model_parameters(args, device):
    return dict(num_depth=args.num_depth, sweep_size=tuple(args.sweep_size), share_encoder=args.share_encoder,
                mixed_precision=args.mixed_precision and device.type == "cuda",
                volume_update_mode=args.volume_update_mode, corr_levels=args.corr_levels,
                corr_radius=args.corr_radius, ot_iter=args.ot_iter, sweep_precision=args.sweep_precision,
                gradient_checkpointing=args.gradient_checkpointing, gru_small_kernel=args.gru_small_kernel,
                gru_large_kernel=args.gru_large_kernel, gru_large_mode=args.gru_large_mode)


def make_loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      drop_last=dataset.split == "train", num_workers=args.num_workers,
                      pin_memory=str(args.device).startswith("cuda"))


def move_batch(batch, device):
    return {name: ([item.to(device, non_blocking=True) for item in value] if isinstance(value, list)
                   else value.to(device, non_blocking=True)) for name, value in batch.items()}


def check_device(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use CPU or CUDA: the pipeline requires 5D grid_sample")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def batch_metrics(result, batch):
    mask = batch["gt_valid"] & result["valid_mask"]
    count = int(mask.sum())
    metrics = dict(valid_pixels=count, valid_fraction=float(mask.float().mean()))
    if count:
        target = batch["gt_index"][mask]
        metrics["init_index_mae"] = float((result["init_disp"][mask] - target).abs().mean().detach())
        metrics["iteration_index_mae"] = [float((pred[mask] - target).abs().mean().detach())
                                           for pred in result["disp_predictions"]]
    else:
        metrics.update(init_index_mae=None, iteration_index_mae=[])
    active = result["valid_mask"]
    probability = result["init_prob"].detach().float()
    raw_entropy = -(probability * (probability + 1e-8).log()).sum(dim=1, keepdim=True)
    metrics["mean_initial_entropy"] = float(raw_entropy[active].mean()) if active.any() else 0.
    metrics["mean_initial_normalized_entropy"] = float(result["uncertainty"][active].mean().detach()) if active.any() else 0.
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, axis, iters, max_batches=0):
    model.eval()
    totals = dict(valid_pixels=0, absolute_index_error=0., absolute_relative_depth_error=0., squared_depth_error=0.)
    per_iteration = np.zeros(iters, dtype=np.float64)
    initial_error = 0.
    samples = 0
    for i, raw in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch(raw, device)
        out = model(batch["imgs"], batch["grids"], batch["valid_mask"], iters=iters)
        mask = batch["gt_valid"] & out["valid_mask"]
        n = int(mask.sum())
        samples += batch["gt_index"].shape[0]
        if not n:
            continue
        pred, gt = out["disp"][mask], batch["gt_index"][mask]
        rho = float(axis[0]) + pred * float(axis[1] - axis[0])
        depth = rho.reciprocal()
        gt_depth = batch["gt_depth"][mask]
        totals["valid_pixels"] += n
        totals["absolute_index_error"] += float((pred - gt).abs().sum())
        totals["absolute_relative_depth_error"] += float(((depth - gt_depth).abs() / gt_depth).sum())
        totals["squared_depth_error"] += float((depth - gt_depth).square().sum())
        initial_error += float((out["init_disp"][mask] - gt).abs().sum())
        for t, value in enumerate(out["disp_predictions"]):
            per_iteration[t] += float((value[mask] - gt).abs().sum())
    n = totals["valid_pixels"]
    return dict(samples=samples, valid_pixels=n, index_mae=totals["absolute_index_error"] / n if n else None,
                init_index_mae=initial_error / n if n else None,
                iteration_index_mae=(per_iteration / n).tolist() if n else [],
                depth_abs_rel=totals["absolute_relative_depth_error"] / n if n else None,
                depth_rmse=(totals["squared_depth_error"] / n)**.5 if n else None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))
