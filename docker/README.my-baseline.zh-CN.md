# my-baseline 镜像

构建于 2026-11-06，服务器为 `zhaoxian@10.104.4.15`。

- 镜像名：`my-baseline:latest`
- 平台：`linux/amd64`
- 镜像 ID：`sha256:53afbf4be7a3c4c672257be572b6e1f7b9bf17e1f4cd6c3c1cc7cd3c2cd3f5be`
- 基础镜像：`dl-pdf-omni:torch2.8.0-cu128`
- 本次基础镜像 ID：`sha256:7a007e3a1a0ddb7bba05f15f52f43bb50489a230084c6643a9585ddac7a811c5`

新镜像继承原 PDF-Omni 镜像的全部文件系统层、环境变量、用户 `dl`、工作目录 `/workspace/PDF-Omni` 和默认命令 `bash`，仅更新镜像说明标签。无需重复安装依赖；两镜像共享原有环境层。

## 环境

Python 3.11.13、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、CUDA 12.8。项目依赖沿用 requirements.txt：NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.9、Pillow 12.2.0、ImageIO 2.37.4、scikit-image 0.25.2、tifffile 2025.5.10、easydict 1.13、PyYAML 6.0.3、TensorBoard 2.21.0、Open3D 0.19.0。

Python 环境位于 `/opt/pdf-omni-venv`，已加入 PATH，进入后直接使用 `python`。

## 构建

在上述服务器执行，要求基础镜像已存在：

```bash
cd /homes/zhaoxian/lzx_ws/project/PDF-Omni
docker build --pull=false --network=none -t my-baseline:latest - < docker/Dockerfile.my-baseline
```

## 创建开发容器

沿用项目已有的 GPU 设备与目录挂载配置：

```bash
cd /homes/zhaoxian/lzx_ws/project/PDF-Omni
DL_IMAGE=my-baseline:latest CONTAINER_NAME=my-baseline-dev GPUS=1 bash docker/run.sh
docker exec -it my-baseline-dev bash
```

项目源码由宿主目录挂载到 `/workspace/PDF-Omni`。上述命令用于首次创建容器；已有同名容器时使用 `docker start my-baseline-dev`，再执行 `docker exec`。GPU 1 为此次验证使用的设备，正式运行时按服务器空闲情况选择。

## 本次验证

- 新旧镜像的文件系统层与运行环境变量一致，镜像 ID 不同。
- 新镜像 `python -m pip check` 通过。
- 在 GPU 1（RTX 5090）开启 GE、DiGRU，完成四路 64×64 合成输入、16 个内部候选、两轮 FP32 前向；输出 `[1,1,16,16]` 且数值有限。PyTorch 记录的峰值已分配显存为 38.9 MiB，不包含 CUDA 上下文等全部进程开销。
- 该检查验证环境与模型可执行性，未评估真实数据精度。
- 临时验证容器已清理；本次只创建镜像，未保留开发容器。
