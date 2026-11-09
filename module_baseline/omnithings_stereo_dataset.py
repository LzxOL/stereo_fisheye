"""OmniThings 双目样本：原始图像 + 左参考扫描网格 + 重投影后的径向 GT。"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.image import readImage, rgb2gray, normalizeImage
from .stereo_geometry import (load_calibration, select_camera, inverse_depth_axis, estimate_angles,
                              make_sweep_grids, read_full_gt, rig_gt_to_reference)
from .config import load_config
import hashlib


class OmniThingsStereoDataset(Dataset):
    def __init__(self, root="../omnidata/omnithings", ref_cam=None, target_cam=None,
                 split="train", num_depth=None, sweep_size=None, angles=None,
                 val_count=512, seed=0, use_rgb=False, augment=False, gt_width=None, gt_phi=None,
                 check_occlusion=True, occlusion_abs=.01, occlusion_rel=.01, image_size=None):
        defaults = load_config()["data"]
        num_depth = defaults["num_depth"] if num_depth is None else num_depth
        sweep_size = defaults["sweep_size"] if sweep_size is None else sweep_size
        image_size = defaults["image_size"] if image_size is None else image_size
        if angles is None:
            angles = [defaults[k] for k in ("sweep_azimuth_center_deg", "sweep_azimuth_span_deg",
                                           "sweep_elevation_min_deg", "sweep_elevation_max_deg")]
        self.root = Path(root)
        self.split = split
        self.opts, cameras, self.calibration = load_calibration(root)
        if ref_cam is None or target_cam is None or str(ref_cam) == str(target_cam):
            raise ValueError("Specify distinct --ref_cam and --target_cam after analyzing the actual calibration")
        self.ref_index, self.reference = select_camera(cameras, ref_cam)
        self.target_index, self.target = select_camera(cameras, target_cam)
        for camera in (self.reference, self.target):
            if (camera.height, camera.width) != tuple(image_size):
                raise ValueError(f"Camera {camera.id} calibration is {camera.height}x{camera.width}, expected {image_size}; do not resize implicitly")
        if self.ref_index == self.target_index:
            raise ValueError("Reference and target cameras must differ")
        self.axis = inverse_depth_axis(self.opts, num_depth)
        self.sweep_size = tuple(sweep_size)
        inferred = estimate_angles(self.reference, self.target, self.axis) if any(a is None for a in angles) else angles
        self.angles = [float(inferred[i] if value is None else value) for i, value in enumerate(angles)]
        left, right, candidate, self.left_valid, self.target_valid = make_sweep_grids(
            self.reference, self.target, self.axis, self.sweep_size, self.angles)
        self.grids = [torch.from_numpy(left), torch.from_numpy(right)]
        self.candidate_valid = torch.from_numpy(candidate)
        if not candidate.any():
            raise ValueError("The selected angular window has no valid stereo candidates")
        self.gt_width, self.gt_phi = gt_width, float(self.opts.gt_phi if gt_phi is None else gt_phi)
        if not 0 < self.gt_phi <= 90:
            raise ValueError("gt_phi must be in (0,90] degrees")
        self.use_rgb, self.augment = use_rgb, augment and split == "train"
        if self.augment:
            from dataset import Augmentor
            self.photo_aug = Augmentor().photo_aug
        self.check_occlusion = check_occlusion
        self.occlusion_abs, self.occlusion_rel = occlusion_abs, occlusion_rel
        official_train = list(self.opts.train_idx)
        if not 0 <= val_count < len(official_train):
            raise ValueError("val_count must leave at least one official training sample")
        shuffled = np.random.default_rng(seed).permutation(official_train).tolist()
        validation = set(shuffled[:val_count])
        splits = dict(train=[f for f in official_train if f not in validation],
                      val=sorted(validation), test=list(self.opts.test_idx))
        if split not in splits:
            raise ValueError("split must be train, val or test")
        self.frames = splits[split]
        self.metadata = dict(ref_cam=f"cam{self.ref_index+1}", target_cam=f"cam{self.target_index+1}",
                             camera_ids=[str(self.reference.id), str(self.target.id)], calibration=self.calibration,
                             image_size=list(image_size),
                             invalid_mask_hashes=[hashlib.sha256(c.invalid_mask.tobytes()).hexdigest()
                                                  for c in (self.reference, self.target)],
                             inverse_depth_values=self.axis.tolist(), num_depth=len(self.axis),
                             inverse_depth_min=float(self.axis[0]), inverse_depth_max=float(self.axis[-1]),
                             inverse_depth_step=float(self.axis[1] - self.axis[0]),
                             full_num_invdepth=int(self.opts.num_invdepth),
                             candidate_stride=int(self.opts.num_invdepth) // len(self.axis),
                             sweep_size=list(self.sweep_size), angles=self.angles, gt_phi=self.gt_phi,
                             gt_width=gt_width, check_occlusion=check_occlusion,
                             occlusion_abs=occlusion_abs, occlusion_rel=occlusion_rel,
                             use_rgb=use_rgb, val_count=val_count, split_seed=seed,
                             official_train_count=len(official_train), official_test_count=len(self.opts.test_idx),
                             validation_frames=sorted(validation), grid_convention="cnn_stride2_pixel_centers")

    def __len__(self):
        return len(self.frames)

    def inspect_gt(self, frame=None):
        _, stats = read_full_gt(self.root, self.frames[0] if frame is None else frame, self.opts, self.gt_width)
        return dict(stats, gt_phi=self.gt_phi, reference_centered=False)

    def __getitem__(self, item):
        frame = self.frames[item]
        raw_images = []
        for index, camera in ((self.ref_index, self.reference), (self.target_index, self.target)):
            path = self.root / (self.opts.img_fmt % (index + 1, frame))
            if not path.is_file():
                raise FileNotFoundError(path)
            image = readImage(str(path))
            if image.shape[:2] != (camera.height, camera.width):
                raise ValueError(f"Image/calibration size mismatch: {path}")
            if image.ndim == 3:
                image = image[..., :3]
            raw_images.append(image)
        # 复用原 Augmentor.photo_aug；将原四路 split(...,4) 改为两路共享同一变换。
        # 不调用原来按相机 1/3 索引擦除的四路 eraser，也不改变图像几何。
        if self.augment:
            stacked = np.concatenate(raw_images, axis=0)
            stacked = np.array(self.photo_aug(Image.fromarray(stacked)), dtype=np.uint8)
            raw_images = list(np.split(stacked, 2, axis=0))
        images = []
        for image, camera in zip(raw_images, (self.reference, self.target)):
            if not self.use_rgb and image.ndim == 3:
                image = rgb2gray(image, channel_wise_mean=True)
            image = normalizeImage(image, camera.invalid_mask)
            if image.ndim == 2:
                image = np.repeat(image[None], 3, axis=0)
            else:
                image = image.transpose(2, 0, 1)
            if not np.isfinite(image).all():
                raise ValueError(f"Non-finite normalized image: frame {frame}, camera {camera.id}")
            images.append(torch.from_numpy(image.astype(np.float32)))

        raw_gt, _ = read_full_gt(self.root, frame, self.opts, self.gt_width)
        gt, valid, depth = rig_gt_to_reference(raw_gt, self.gt_phi, self.reference, self.target,
                                             self.axis, self.sweep_size, self.angles, self.check_occlusion,
                                             self.occlusion_abs, self.occlusion_rel)
        # GT 落点与目标栅格中心存在离散误差，监督再要求其相邻候选 LUT 均有效。
        floor = np.floor(gt).astype(int).clip(0, len(self.axis) - 1)
        ceil = np.ceil(gt).astype(int).clip(0, len(self.axis) - 1)
        candidates = self.candidate_valid.numpy()[0]
        valid &= np.take_along_axis(candidates, floor, axis=0) & np.take_along_axis(candidates, ceil, axis=0)
        gt = np.where(valid, gt, 0).astype(np.float32)
        depth = np.where(valid, depth, 0).astype(np.float32)
        return dict(imgs=images, grids=self.grids, valid_mask=self.candidate_valid,
                    gt_index=torch.from_numpy(gt), gt_valid=torch.from_numpy(valid),
                    gt_depth=torch.from_numpy(depth), frame=frame)
