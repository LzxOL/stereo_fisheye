#!/usr/bin/env bash
# 相机对先经 analyze_omnithings_pairs.py 选择；所有正式默认值来自 YAML。
# 用法：bash scripts/train_stereo_omnithings.sh --ref_cam camX --target_cam camY [其他覆盖参数]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python train_stereo.py --config module_baseline/config/stereo_omnithings.yaml "$@"
