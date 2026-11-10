#!/usr/bin/env python3
"""固定checkpoint的只读诊断；所有扰动仅发生在独立进程，不修改正在训练的模型。"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module_baseline import StereoBaseline
from module_baseline.training import make_dataset, check_device, write_json
from module_baseline.volume_lookup import CorrelationVolumeSampler, aggregate_geometry
from module_baseline.network import autocast


def index_mae(prediction, target, valid):
    return float((prediction[valid] - target[valid]).abs().mean())


def replay(model, hidden, context, initial, cv, valid, iters, zero_lookup=False):
    # 重用已有Context/UpdateBlock，隔离CV变化；不重训网络、不改变初始Context。
    sampler = CorrelationVolumeSampler(cv, model.opts.corr_levels, model.opts.corr_radius)
    state, index = hidden, initial
    for _ in range(iters):
        feature = sampler(index)
        if zero_lookup:
            feature = torch.zeros_like(feature)
        with autocast(enabled=model.opts.mixed_precision):
            state, delta, _ = model.update_block(state, context, feature, index, no_upsample=True)
        index = torch.where(valid, (index + delta.float()).clamp(0, model.num_depth - 1), 0.)
    return index


def frame_diagnostics(model, data, item, wrong_item, device, iters, finite_limit):
    sample = data[item]
    images = [image[None].to(device) for image in sample["imgs"]]
    grids = [grid[None].to(device) for grid in sample["grids"]]
    candidates = sample["valid_mask"][None].to(device)
    gt = sample["gt_index"][None].to(device)
    depth = sample["gt_depth"][None].to(device)
    logits = {}
    handle = model.volume_gen.classifier.register_forward_hook(
        lambda module, args, out: logits.update(value=out[:, 0].float()))
    try:
        out = model(images, grids, candidates, iters=iters, return_intermediates=True)
    finally:
        handle.remove()
    valid = sample["gt_valid"][None].to(device) & out["valid_mask"]
    if not valid.any():
        return dict(frame=sample["frame"], valid_pixels=0)
    detail = out["intermediates"]
    count = int(valid.sum())
    values = torch.arange(model.num_depth, device=device).view(1, -1, 1, 1)
    probability = out["init_prob"]
    soft = model.volume_gen._masked_softmax(logits["value"][:, None], candidates)[:, 0]
    soft_index = (soft * values).sum(1, keepdim=True)
    cv = detail["cv"]
    cv_peak = cv[:, 0].masked_fill(~candidates[:, 0], -torch.inf).argmax(1, keepdim=True).float()
    ref, target = detail["reference"].float(), detail["target"].float()
    cosine = (ref * target).sum(1) / (ref.square().sum(1).sqrt() * target.square().sum(1).sqrt()).clamp_min(1e-8)
    cosine_peak = cosine.permute(0, 3, 1, 2).masked_fill(~candidates[:, 0], -torch.inf).argmax(1, keepdim=True).float()
    report = dict(frame=sample["frame"], valid_pixels=count,
                  gt_index_mean=float(gt[valid].mean()), init_index_mean_gt=float(out["init_disp"][valid].mean()),
                  init_index_mean_all=float(out["init_disp"][out["valid_mask"]].mean()),
                  gt_fraction_of_active_pixels=float(valid.sum() / out["valid_mask"].sum()),
                  init_mae=index_mae(out["init_disp"], gt, valid), raw_softmax_init_mae=index_mae(soft_index, gt, valid),
                  ot_argmax_mae=index_mae(probability.argmax(1, keepdim=True).float(), gt, valid),
                  cv_peak_mae=index_mae(cv_peak, gt, valid), cosine_peak_mae=index_mae(cosine_peak, gt, valid),
                  final_mae=index_mae(out["disp"], gt, valid),
                  normalized_entropy=float(out["uncertainty"][valid].mean()),
                  iteration_mae=[index_mae(pred, gt, valid) for pred in out["disp_predictions"]])
    radius = model.opts.corr_radius * 2 ** (model.opts.corr_levels - 1)
    report["gt_within_nominal_initial_lookup_fraction"] = float(((out["init_disp"] - gt).abs()[valid] <= radius).float().mean())
    report["cv_peak_within_2_indices_fraction"] = float(((cv_peak - gt).abs()[valid] <= 2).float().mean())
    # 同一像素集合上的直方图，防止用全FOV预测均值与局部GT均值混淆。
    report["gt_histogram"] = torch.bincount(gt[valid].round().long(), minlength=model.num_depth).tolist()
    report["initial_probability_mass_gt_pixels"] = probability.permute(0, 2, 3, 1)[valid[:, 0]].sum(0).cpu().tolist()
    active = out["valid_mask"][:, 0]
    report["initial_probability_mass_all_pixels"] = probability.permute(0, 2, 3, 1)[active].sum(0).cpu().tolist()

    common = dict(model=model, hidden=detail["hidden_init"], context=detail["context_feature"],
                  initial=out["init_disp"], valid=out["valid_mask"], iters=iters)
    reproduced = replay(cv=cv, **common)
    torch.testing.assert_close(reproduced, out["disp"], rtol=1e-4, atol=1e-3)
    no_cv = replay(cv=cv, zero_lookup=True, **common)
    permutation = torch.randperm(cv.shape[-2] * cv.shape[-1], device=device)
    shuffled_cv = cv.flatten(3)[..., permutation].reshape_as(cv)
    bad_cv = replay(cv=shuffled_cv, **common)
    report["zero_lookup_mae"] = index_mae(no_cv, gt, valid)
    report["shuffled_cv_mae"] = index_mae(bad_cv, gt, valid)
    report["zero_lookup_output_change"] = index_mae(no_cv, out["disp"], valid)

    # 同权重的后验反事实，不等价于“重新训练一个无OT模型”。
    if model.volume_update_mode == "split":
        g2d, entropy = aggregate_geometry(detail["gev"], soft, candidates)
        context_input = torch.cat([detail["reference"][..., 0].float(), g2d, entropy], 1)
        with autocast(enabled=model.opts.mixed_precision):
            hidden, context = model.context_net(context_input)
        soft_final = replay(model, hidden, context, soft_index, cv, out["valid_mask"], iters)
        report["posthoc_no_ot_final_mae"] = index_mae(soft_final, gt, valid)

    near = valid & (depth <= finite_limit)
    rho = float(data.axis[0]) + out["disp"] * float(data.axis[1] - data.axis[0])
    predicted_depth = rho.reciprocal()
    report["finite_gt_fraction"] = float(near.sum() / valid.sum())
    report["predicted_zero_index_fraction"] = float((out["disp"][valid] == 0).float().mean())
    report["predicted_zero_on_finite_gt_fraction"] = float((out["disp"][near] == 0).float().mean()) if near.any() else None
    report["finite_gt_index_mae"] = index_mae(out["disp"], gt, near) if near.any() else None
    # 同时报告截断阈值和饱和比例，不用裁剪后的数值冒充原始米制指标。
    report["finite_gt_capped_depth_abs_rel"] = float(((predicted_depth[near].clamp_max(finite_limit) - depth[near]).abs() / depth[near]).mean()) if near.any() else None
    recovered = (float(data.axis[0]) + gt * float(data.axis[1] - data.axis[0])).reciprocal()
    report["gt_axis_roundtrip_max_rel_error"] = float(((recovered[valid] - depth[valid]).abs() / depth[valid]).max())

    # 错场景右图会同时改变GEV Context和CV；仅用于检测依赖性，不是有效立体数据。
    del detail, ref, target, cosine, logits, reproduced, no_cv, bad_cv, shuffled_cv
    wrong_image = data[wrong_item]["imgs"][1][None].to(device)
    wrong_out = model([images[0], wrong_image], grids, candidates, iters=iters)
    report["wrong_right_frame"] = data.frames[wrong_item]
    report["wrong_right_final_mae"] = index_mae(wrong_out["disp"], gt, valid)
    report["wrong_right_output_change"] = index_mae(wrong_out["disp"], out["disp"], valid)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--finite_depth_limit", type=float, default=50.)
    parser.add_argument("--min_free_gib", type=float, default=6.)
    parser.add_argument("--output", default="results/stereo_diagnostics.json")
    cli = parser.parse_args()
    device = check_device(cli.device)
    if cli.samples < 2 or cli.iters < 1:
        parser.error("Use at least two samples and one GRU iteration")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    saved = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**saved["training_args"])
    args.augment = False
    args.root = cli.root or args.root
    (args.sweep_azimuth_center_deg, args.sweep_azimuth_span_deg,
     args.sweep_elevation_min_deg, args.sweep_elevation_max_deg) = saved["geometry"]["angles"]
    data = make_dataset(args, "val")
    if saved["geometry"] != data.metadata:
        raise ValueError("Diagnostic data geometry differs from checkpoint")
    config = dict(saved["model_config"])
    config["mixed_precision"] &= device.type == "cuda"
    model = StereoBaseline(**config).to(device).eval()
    state = saved.get("net_state_dict", saved.get("model"))
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    selected = np.linspace(0, len(data) - 1, min(cli.samples, len(data)), dtype=int).tolist()
    report = dict(checkpoint=cli.checkpoint, checkpoint_epoch=saved["epoch"] + 1,
                  geometry=data.metadata, finite_depth_limit=cli.finite_depth_limit,
                  nominal_lookup_radius=model.opts.corr_radius * 2 ** (model.opts.corr_levels - 1), frames=[])
    with torch.no_grad():
        for i, item in enumerate(selected):
            if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] < cli.min_free_gib * 2**30:
                raise RuntimeError("Insufficient headroom; stop diagnostics without modifying baseline training")
            record = frame_diagnostics(model, data, item, selected[(i + 1) % len(selected)], device, cli.iters, cli.finite_depth_limit)
            report["frames"].append(record)
            write_json(cli.output, report)
            print(json.dumps({k: v for k, v in record.items() if not isinstance(v, list)}), flush=True)
    good = [r for r in report["frames"] if r["valid_pixels"]]
    summary = {}
    for key, value in good[0].items():
        if key not in ("frame", "wrong_right_frame", "valid_pixels") and isinstance(value, (int, float)):
            use = [r for r in good if isinstance(r.get(key), (int, float))]
            summary[key] = sum(r[key] * r["valid_pixels"] for r in use) / sum(r["valid_pixels"] for r in use)
    report["summary"] = summary
    write_json(cli.output, report)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
