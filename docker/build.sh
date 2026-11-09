#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
# Docker's saved loopback proxy is currently unavailable; host networking can use direct HTTPS.
exec docker build --network=host \
  --build-arg http_proxy= --build-arg https_proxy= \
  --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= \
  --build-arg all_proxy= --build-arg ALL_PROXY= \
  --build-arg "BASE_IMAGE=${BASE_IMAGE:-dl-ftfoot:torch2.8.0-cu128}" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}" \
  -t "${DL_IMAGE:-dl-pdf-omni:torch2.8.0-cu128}" -f docker/Dockerfile .
