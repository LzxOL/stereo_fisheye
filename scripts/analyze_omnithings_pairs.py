#!/usr/bin/env python3
"""根据真实 config.yaml 排序有向相机对；不假设 cam1/cam2 最优。"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.geometry import applyTransform, getRot, getTr
from module_baseline.stereo_geometry import (load_calibration, inverse_depth_axis, local_rays,
                                             project_camera, estimate_angles, read_full_gt)
from module_baseline.config import load_config


def analyze(root, num_depth):
    opts, cameras, raw = load_calibration(root)
    axis = inverse_depth_axis(opts, num_depth)
    rays = local_rays((91, 181), (0, 359, -89, 89))
    area = np.sqrt(np.maximum(0, 1 - rays[1] ** 2))
    camera_info = []
    for i, camera in enumerate(cameras):
        camera_info.append(dict(directory=f"cam{i+1}", cam_id=str(camera.id),
                                image_size=[camera.height, camera.width], max_fov_deg=float(np.rad2deg(camera.max_theta) * 2),
                                center_rig=getTr(camera.cam2rig).ravel().tolist(),
                                optical_axis_rig=(getRot(camera.cam2rig) @ np.array([0, 0, 1.])).tolist()))
    pairs = []
    depth_ids = sorted(set([0, num_depth // 4, num_depth // 2, num_depth - 1]))
    for i, reference in enumerate(cameras):
        _, ref_valid = project_camera(reference, rays)
        denominator = area[ref_valid].sum()
        for j, target in enumerate(cameras):
            if i == j:
                continue
            baseline = np.linalg.norm(getTr(reference.cam2rig) - getTr(target.cam2rig))
            axes = [np.array(camera_info[k]["optical_axis_rig"]) for k in (i, j)]
            angle = np.rad2deg(np.arccos(np.clip(np.dot(*axes), -1, 1)))
            infinity_rays = getRot(target.rig2cam) @ getRot(reference.cam2rig) @ rays
            _, infinity_valid = project_camera(target, infinity_rays)
            common_fraction = float(area[ref_valid & infinity_valid].sum() / max(denominator, 1e-12))
            depths = []
            score = 0.
            for d in depth_ids:
                rig_points = applyTransform(reference.cam2rig, rays / axis[d])
                target_points = applyTransform(target.rig2cam, rig_points)
                _, target_valid = project_camera(target, target_points)
                valid = ref_valid & target_valid
                coverage = float(area[valid].sum() / max(denominator, 1e-12))
                if valid.any():
                    ref_direction = getRot(reference.cam2rig) @ rays[:, valid]
                    tgt_direction = rig_points[:, valid] - getTr(target.cam2rig)
                    tgt_direction /= np.linalg.norm(tgt_direction, axis=0, keepdims=True)
                    parallax = np.arccos(np.clip((ref_direction * tgt_direction).sum(0), -1, 1))
                    mean_angle = float(np.rad2deg(np.average(parallax, weights=area[valid])))
                    score += coverage * float(np.average(np.sin(parallax), weights=area[valid]))
                else:
                    mean_angle = None
                depths.append(dict(candidate=d, inverse_depth=float(axis[d]), depth=float(1 / axis[d]),
                                   stereo_coverage=coverage, mean_triangulation_angle_deg=mean_angle))
            pair = dict(ref_cam=f"cam{i+1}", target_cam=f"cam{j+1}", baseline=float(baseline),
                        optical_axis_angle_deg=float(angle), common_fov_fraction=common_fraction,
                        by_depth=depths, score=score / len(depth_ids))
            try:
                pair["suggested_angles"] = estimate_angles(reference, target, axis)
            except ValueError as exc:
                pair["unusable_reason"] = str(exc)
            pairs.append(pair)
    pairs.sort(key=lambda p: p["score"], reverse=True)
    report = dict(root=str(Path(root).resolve()), cameras=camera_info, ranking=pairs,
                  fraction_definition="solid-angle-weighted fraction of valid reference-camera directions",
                  ranking_definition="mean(coverage * mean(sin(triangulation angle))) across sampled inverse depths",
                  limitation="FOV geometry only, not scene occlusion or learned matching quality")
    try:
        _, report["raw_gt_sample"] = read_full_gt(root, opts.train_idx[0], opts)
        report["raw_gt_sample"]["gt_phi"] = opts.gt_phi
    except (FileNotFoundError, ValueError) as exc:
        report["raw_gt_unavailable"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="../omnidata/omnithings")
    parser.add_argument("--num_depth", type=int, default=load_config()["data"]["num_depth"])
    parser.add_argument("--output", default="results/omnithings_pairs.json")
    args = parser.parse_args()
    try:
        report = analyze(args.root, args.num_depth)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
