#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${DL_IMAGE:-dl-pdf-omni:torch2.8.0-cu128}
container=${CONTAINER_NAME:-pdf-omni-dev}
host_home=/homes/zhaoxian
cache_dir=${CACHE_DIR:-${host_home}/.cache/pdf-omni-docker}
mkdir -p "$cache_dir"
args=(--name "$container" --user "$(id -u):$(id -g)" --shm-size 16g
      --workdir /workspace/PDF-Omni
      --mount "type=bind,src=${project_root},dst=/workspace/PDF-Omni"
      --mount "type=bind,src=${host_home},dst=${host_home}"
      --mount "type=bind,src=${cache_dir},dst=/cache"
      --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly
      --mount type=bind,src=/etc/group,dst=/etc/group,readonly
      --env "HOME=${host_home}" --env "USER=$(id -un)" --env "LOGNAME=$(id -un)"
      --env TZ=Asia/Shanghai --env PYTHONPATH=/workspace/PDF-Omni
      --env CUDA_DEVICE_ORDER=PCI_BUS_ID
      --env "http_proxy=${CONTAINER_PROXY:-}" --env "https_proxy=${CONTAINER_PROXY:-}" --env all_proxy=
      --env "HTTP_PROXY=${CONTAINER_PROXY:-}" --env "HTTPS_PROXY=${CONTAINER_PROXY:-}" --env ALL_PROXY=)
if [[ ${GPU_MODE:-manual} == none ]]; then
    :
else
    args+=(--device /dev/nvidiactl --device /dev/nvidia-uvm)
    [[ ! -c /dev/nvidia-uvm-tools ]] || args+=(--device /dev/nvidia-uvm-tools)
    gpu_indices=${GPUS:-1}
    if [[ "$gpu_indices" == all ]]; then
        selected=()
        for path in /dev/nvidia[0-9]*; do selected+=("${path#/dev/nvidia}"); done
    else
        IFS=, read -r -a selected <<< "$gpu_indices"
    fi
    for gpu in "${selected[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ && -c /dev/nvidia${gpu} ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
        args+=(--device "/dev/nvidia${gpu}")
    done
    driver_dir=/lib/x86_64-linux-gnu
    for path in "$driver_dir"/libcuda.so* "$driver_dir"/libnvidia-ml.so* \
                "$driver_dir"/libnvidia-ptxjitcompiler.so* "$driver_dir"/libnvidia-gpucomp.so* \
                "$driver_dir"/libnvidia-nvvm.so* "$driver_dir"/libnvidia-fatbinaryloader.so*; do
        [[ -f "$path" ]] || continue
        args+=(--mount "type=bind,src=${path},dst=/usr/local/lib/host-nvidia/$(basename "$path"),readonly")
    done
    args+=(--env LD_LIBRARY_PATH=/usr/local/lib/host-nvidia:/usr/local/cuda/lib64
           --mount type=bind,src=/usr/bin/nvidia-smi,dst=/usr/local/bin/nvidia-smi,readonly)
fi
if [[ ${DETACH:-1} == 1 ]]; then
    args+=(-d)
    [[ $# -gt 0 ]] || set -- sleep infinity
else
    [[ ! -t 0 ]] || args+=(-i)
    [[ ! -t 1 ]] || args+=(-t)
    [[ $# -gt 0 ]] || set -- bash
fi
exec docker run "${args[@]}" "$image" "$@"
