"""沿用工程 OCam 定义的双目扫描几何与 Rig GT 重投影；长度单位沿用数据标定。"""

from pathlib import Path
import numpy as np
import yaml
from utils.common import Edict
from utils.dbhelper import loadDBConfigs
from utils.geometry import applyTransform, getRot, getTr
from utils.image import readImageFloat, pixelToGrid


def load_calibration(root):
    root = Path(root)
    if not (root / "config.yaml").is_file():
        raise FileNotFoundError(f"OmniThings config not found: {root / 'config.yaml'}")
    raw = yaml.safe_load((root / "config.yaml").read_text())
    defaults = Edict(img_fmt="cam%d/%04d.png", gt_depth_fmt="omnidepth_gt_%d/%05d.tiff",
                     equirect_size=[160, 640], num_invdepth=192, min_depth=.5,
                     max_depth=1 / np.finfo(np.float64).eps)
    opts, cameras = loadDBConfigs("omnithings", str(root), defaults)
    if len(cameras) != 4 or len(opts.train_idx) != 9216 or len(opts.test_idx) != 1024:
        raise ValueError("Expected the existing OmniThings four-camera / 9216+1024 split definition")
    for camera in cameras:
        if camera.invalid_mask.shape != (camera.height, camera.width):
            raise ValueError(f"Invalid mask size for camera {camera.id}")
    return opts, cameras, raw


def select_camera(cameras, name):
    # 与原 Dataset.loadImages 一致：camN 图像目录对应 YAML cameras[N-1]。
    # 分析报告同时打印 cam_id，避免将标定 id 与目录位置静默混淆。
    index = int(str(name).lower().replace("cam", "")) - 1
    if not 0 <= index < len(cameras):
        raise ValueError("Camera must be cam1, cam2, cam3 or cam4")
    return index, cameras[index]


def inverse_depth_axis(opts, num_depth):
    full_count = int(opts.num_invdepth)
    if num_depth < 2 or full_count < 2 or full_count % num_depth:
        raise ValueError("num_depth must divide the calibration's full num_invdepth for stride sampling")
    rho_min = float(opts.get("min_invdepth", 1 / opts.max_depth))
    rho_max = float(opts.get("max_invdepth", 1 / opts.min_depth))
    step = float(opts.get("sample_step_invdepth", (rho_max - rho_min) / (full_count - 1)))
    full_axis = rho_min + np.arange(full_count, dtype=np.float64) * step
    axis = full_axis[::full_count // num_depth]
    if not np.isfinite(axis).all() or axis[0] <= 0 or not (np.diff(axis) > 0).all():
        raise ValueError("Inverse-depth candidates must be finite, positive and strictly increasing")
    return axis  # 默认原 192 轴的 [0,2,...,190]，不是重新 linspace 到第 191 层。


def local_rays(size, angles):
    h, w = size
    center, span, emin, emax = angles
    if min(h, w) <= 1 or not (0 < span < 360 and -90 <= emin < emax <= 90):
        raise ValueError("Invalid local spherical grid size or angular bounds")
    az = np.deg2rad(np.linspace(center - span / 2, center + span / 2, w))
    el = np.deg2rad(np.linspace(emax, emin, h))
    az, el = np.meshgrid(az, el)
    # 参考相机 +Z 为光轴，az 朝 +X，正 elevation 朝 -Y（图像上方）。
    rays = np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
    return rays.reshape(3, -1)


def project_camera(camera, points):
    pixels = camera.rayToPixel(points)
    u, v = pixels
    valid = np.isfinite(points).all(0) & (np.linalg.norm(points, axis=0) > 0)
    valid &= np.isfinite(pixels).all(0) & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)
    # 掩码查表时允许安全限幅索引，但限幅永远不会把视场外点变成有效点。
    safe_u = np.nan_to_num(u, nan=0, posinf=0, neginf=0).clip(0, camera.width - 1)
    safe_v = np.nan_to_num(v, nan=0, posinf=0, neginf=0).clip(0, camera.height - 1)
    x0, x1 = np.floor(safe_u).astype(int), np.ceil(safe_u).astype(int)
    y0, y1 = np.floor(safe_v).astype(int), np.ceil(safe_v).astype(int)
    mask = camera.invalid_mask
    valid &= ~(mask[y0, x0] | mask[y0, x1] | mask[y1, x0] | mask[y1, x1])
    return pixels, valid


def feature_grid(camera, pixels, valid):
    # 原图不缩放。CNN 的 stride=2 卷积中心为原图 2*j；不是将 W-1 线性缩到 Wf-1。
    hf, wf = (camera.height + 1) // 2, (camera.width + 1) // 2
    if min(hf, wf) <= 1:
        raise ValueError("Fisheye images are too small for the CNN sampling grid")
    grid = pixelToGrid(pixels / 2, (1, pixels.shape[1]), (hf, wf)).reshape(-1, 2)
    valid = valid & np.isfinite(grid).all(1) & (np.abs(grid) <= 1).all(1)
    return np.where(valid[:, None], grid, 2.0).astype(np.float32), valid


def estimate_angles(reference, target, axis):
    # 从真实标定投影得到共同视场的包围窗口；窗口内空洞仍由逐候选 mask 排除。
    size = (91, 181)
    base_angles = (0., 359., -89., 89.)
    rays = local_rays(size, base_angles)
    _, ref_valid = project_camera(reference, rays)
    common = np.zeros(rays.shape[1], dtype=bool)
    for rho in axis[[0, len(axis) // 2, -1]]:
        points = applyTransform(target.rig2cam, applyTransform(reference.cam2rig, rays / rho))
        _, target_valid = project_camera(target, points)
        common |= ref_valid & target_valid
    if not common.any():
        raise ValueError("No co-visible rays found for this calibrated camera pair")
    az = np.rad2deg(np.arctan2(rays[0, common], rays[2, common]))
    center = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(az)).mean()))
    offset = (az - center + 180) % 360 - 180
    el = np.rad2deg(np.arcsin(-rays[1, common]))
    lo, hi = offset.min(), offset.max()
    return [float(center + (lo + hi) / 2), float(min(359., max(2., hi - lo + 2))),
            float(max(-89., el.min() - 1)), float(min(89., el.max() + 1))]


def make_sweep_grids(reference, target, axis, size, angles):
    if any(n <= 0 or n % 8 for n in (*size, len(axis))):
        raise ValueError("D/H/W must be positive multiples of 8 for the 3D hourglass")
    h, w = size
    rays = local_rays(size, angles)
    ref_pixels, ref_valid = project_camera(reference, rays)
    left, ref_valid = feature_grid(reference, ref_pixels, ref_valid)
    right = np.empty((h, w, len(axis), 2), dtype=np.float32)
    target_valid = np.zeros((len(axis), h, w), dtype=bool)
    for d, rho in enumerate(axis):
        rig_points = applyTransform(reference.cam2rig, rays / rho)
        target_points = applyTransform(target.rig2cam, rig_points)
        pixels, valid = project_camera(target, target_points)
        grid, valid = feature_grid(target, pixels, valid)
        right[:, :, d] = grid.reshape(h, w, 2)
        target_valid[d] = valid.reshape(h, w)
    left_valid = ref_valid.reshape(h, w)
    candidate_valid = left_valid[None] & target_valid
    return left.reshape(h, w, 2), right, candidate_valid[None], left_valid, target_valid


def zbuffer_indices(pixel_ids, distances, size):
    """每格保留最小径向距离，返回原点序号；空格为 -1，不插值填洞。"""
    result = np.full(size, -1, dtype=np.int64)
    if len(pixel_ids):
        order = np.argsort(distances, kind="stable")
        unique, first = np.unique(pixel_ids[order], return_index=True)
        result[unique] = order[first]
    return result


def cloud_visibility(camera, points, abs_tolerance=.01, rel_tolerance=.01):
    pixels, visible = project_camera(camera, points)
    distance = np.linalg.norm(points, axis=0)
    ids = np.flatnonzero(visible)
    u = np.rint(pixels[0, ids]).astype(int)
    v = np.rint(pixels[1, ids]).astype(int)
    pixel_ids = v * camera.width + u
    nearest = np.full(camera.height * camera.width, np.inf)
    np.minimum.at(nearest, pixel_ids, distance[ids])
    # 这只检查当前 Rig GT 点云已观测到的遮挡；不能恢复 Rig 中心不可见的表面。
    visible[ids] &= distance[ids] <= nearest[pixel_ids] * (1 + rel_tolerance) + abs_tolerance
    return visible


def rig_gt_to_reference(invdepth, gt_phi, reference, target, axis, size, angles,
                        check_occlusion=True, occlusion_abs=.01, occlusion_rel=.01):
    from dataset import makeSphericalRays
    if not isinstance(invdepth, np.ndarray) or invdepth.ndim != 2:
        raise ValueError("GT TIFF must contain a complete 2D inverse-depth image")
    raw_valid = np.isfinite(invdepth) & (invdepth > 0)
    rays = makeSphericalRays(invdepth.shape, gt_phi)
    rig_points = rays[:, raw_valid.ravel()] / invdepth[raw_valid].astype(np.float64)[None]
    ref_points = applyTransform(reference.rig2cam, rig_points)
    target_points = applyTransform(target.rig2cam, rig_points)
    distances = np.linalg.norm(ref_points, axis=0)
    ref_pixels, ref_visible = project_camera(reference, ref_points)
    _, ref_visible = feature_grid(reference, ref_pixels, ref_visible)
    target_pixels, target_visible = project_camera(target, target_points)
    _, target_visible = feature_grid(target, target_pixels, target_visible)
    if check_occlusion:
        ref_visible &= cloud_visibility(reference, ref_points, occlusion_abs, occlusion_rel)
        target_visible &= cloud_visibility(target, target_points, occlusion_abs, occlusion_rel)

    center, span, emin, emax = angles
    safe_distance = np.maximum(distances, np.finfo(np.float64).tiny)
    az = np.rad2deg(np.arctan2(ref_points[0], ref_points[2]))
    az = (az - center + 180) % 360 - 180
    el = np.rad2deg(np.arcsin(np.clip(-ref_points[1] / safe_distance, -1, 1)))
    h, w = size
    x = (az / span + .5) * (w - 1)
    y = (emax - el) / (emax - emin) * (h - 1)
    inside = ref_visible & np.isfinite(distances) & (distances > 0)
    inside &= (np.abs(az) <= span / 2) & (el >= emin) & (el <= emax)
    candidates = np.flatnonzero(inside)
    pixel_ids = np.rint(y[candidates]).astype(int) * w + np.rint(x[candidates]).astype(int)
    winner = zbuffer_indices(pixel_ids, distances[candidates], h * w)
    cells = np.flatnonzero(winner >= 0)
    point_ids = candidates[winner[cells]]
    index = (1 / distances[point_ids] - axis[0]) / (axis[1] - axis[0])
    good = target_visible[point_ids] & np.isfinite(index) & (index >= 0) & (index <= len(axis) - 1)
    gt = np.zeros(h * w, dtype=np.float32)
    valid = np.zeros(h * w, dtype=bool)
    depth = np.zeros(h * w, dtype=np.float32)
    gt[cells[good]] = index[good]
    valid[cells[good]] = True
    depth[cells[good]] = distances[point_ids[good]]
    return gt.reshape(1, h, w), valid.reshape(1, h, w), depth.reshape(1, h, w)


def read_full_gt(root, frame, opts, gt_width=None):
    root = Path(root)
    if gt_width is not None:
        path = root / (opts.gt_depth_fmt % (gt_width, frame))
    else:
        matches = sorted(root.glob(f"omnidepth_gt_*/{frame:05d}.tiff"))
        if len(matches) == 1:
            path = matches[0]
        else:
            path = root / (opts.gt_depth_fmt % (opts.equirect_size[1], frame))
            if not path.is_file():
                raise FileNotFoundError(f"Cannot uniquely locate full GT for frame {frame}; set --gt_width")
    if not path.is_file():
        raise FileNotFoundError(path)
    image = readImageFloat(str(path), read_or_die=False)
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError(f"GT is not a 2D inverse-depth TIFF: {path}")
    valid = np.isfinite(image) & (image > 0)
    stats = dict(path=str(path), shape=list(image.shape), dtype=str(image.dtype), quantity="inverse_depth",
                 valid_fraction=float(valid.mean()), min=float(image[valid].min()) if valid.any() else None,
                 max=float(image[valid].max()) if valid.any() else None)
    return image, stats
