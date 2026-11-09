ARG BASE_IMAGE=dl-ftfoot:torch2.8.0-cu128
FROM ${BASE_IMAGE}
USER root
LABEL org.opencontainers.image.title="PDF-Omni on reusable PyTorch CUDA" \
      org.opencontainers.image.description="PDF-Omni isolated dependencies; PyTorch 2.8 / CUDA 12.8 / RTX 5090"
RUN python -m venv --system-site-packages /opt/pdf-omni-venv
ENV PATH=/opt/pdf-omni-venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/pdf-omni-venv \
    PYTHONPATH=/workspace/PDF-Omni \
    PYTHONNOUSERSITE=1 \
    TZ=Asia/Shanghai
COPY requirements.txt /opt/environment/pdf-omni-requirements.txt
ARG PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN PIP_CACHE_DIR=/root/.cache/pip python -m pip install --index-url "${PIP_INDEX_URL}" --retries 3 --timeout 45 \
      -c /opt/environment/torch-constraints.txt -r /opt/environment/pdf-omni-requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/environment/pdf-omni-pip-freeze.txt \
    && python -c "import torch, numpy, scipy, skimage, PIL, matplotlib, imageio, tifffile, yaml, easydict, open3d; print(torch.__version__, numpy.__version__)"
RUN mkdir -p /workspace/PDF-Omni && chmod -R a+rX /opt/pdf-omni-venv
USER dl
WORKDIR /workspace/PDF-Omni
CMD ["bash"]
