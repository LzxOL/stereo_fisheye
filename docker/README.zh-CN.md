# PDF-Omni：复用现有 PyTorch/CUDA 镜像

服务器项目：`/homes/zhaoxian/lzx_ws/project/PDF-Omni`。

镜像：`dl-pdf-omni:torch2.8.0-cu128`，基于服务器现有 `dl-ftfoot:torch2.8.0-cu128` 构建，复用其 CUDA/PyTorch 开发环境。项目依赖装在镜像的 `/opt/pdf-omni-venv`，保留基础环境；进入容器后默认的 `python` 已使用这个环境，无需 conda activate，也不需要再执行原 scripts/setup_env.sh。

## 启动、进入

已创建并启动容器 `pdf-omni-dev`，映射宿主 GPU 1。容器只读挂载宿主的 `/etc/passwd` 和 `/etc/group`，因此容器以宿主 UID/GID 运行时，用户名与组名也能正确显示。容器内部通过 `cuda:0` 使用这张卡。

```bash
# 在服务器上执行；首次创建或更新容器配置时执行
cd /homes/zhaoxian/lzx_ws/project/PDF-Omni
docker rm -f pdf-omni-dev  # 仅重建容器；项目、环境和数据保存在挂载目录/镜像中
GPUS=1 bash docker/run.sh

# 后续启动时
docker start pdf-omni-dev
docker exec -it pdf-omni-dev bash
```

进入容器后：

```bash
cd /workspace/PDF-Omni
python train.py --help
python eval.py --help
```

`/workspace/PDF-Omni` 对应宿主项目目录。源码、checkpoints、runs 等文件都保存在宿主机；`/homes/zhaoxian` 也映射到容器内相同路径。

## 第一次创建、重建镜像

只有需要创建新容器时才运行 run.sh；同名容器已存在时使用 docker start/exec。

```bash
cd /homes/zhaoxian/lzx_ws/project/PDF-Omni
bash docker/build.sh
# 创建一个持续运行、可反复进入的容器
GPUS=1 bash docker/run.sh
```

选择其他 GPU 时用不同容器名，例如：

```bash
CONTAINER_NAME=pdf-omni-gpu0 GPUS=0 bash docker/run.sh
```

启动脚本默认 GPU 1，支持 GPUS=0,1 或 GPUS=all。更改 GPU 需要重新创建容器；对已运行容器设置 GPUS 不会改变设备映射。脚本按当前服务器缺少 NVIDIA Container Toolkit 的情况，显式挂载选定 GPU 设备和只读驱动库，不使用 privileged。

## 已验证环境

- Python 3.11.13，PyTorch 2.8.0+cu128，torchvision 0.23.0+cu128。
- CUDA 12.8，RTX 5090，compute capability 12.0。
- NumPy 2.2.6，SciPy 1.15.3，Matplotlib 3.10.9，Pillow 12.2.0，TensorBoard 2.21.0，Open3D 0.19.0。
- requirements.txt 的固定依赖全部安装，pip check 通过。
- train.py --help、eval.py --help 通过。
- GE 和 DiGRU 开启的小尺寸合成输入 FP32 前向、反向通过，1094 个梯度张量均有限；AMP 推理通过；NumPy/Torch 相互转换通过。
- 合成检查使用 4 张 64×64 输入、16 个逆深度平面、2 次迭代；输出 1×1×16×16，峰值显存约 334 MiB。这不是完整分辨率训练的显存或性能数据。
- 未修改模型结构、损失或数据加载器。

项目 README 测试环境是 Python 3.10、PyTorch 2.11；这里复用 3.11/2.8，满足其声明的 PyTorch >=2.3，并通过上述检查。尚未进行真实数据/预训练权重精度验证。

完整版本清单在镜像 `/opt/environment/pdf-omni-pip-freeze.txt`，宿主备份在 `docker/environment-freeze.txt`。环境检查日志在 `docker/environment-check.log`。

## 数据和后续训练

检查时 `/homes/zhaoxian/lzx_ws/datasets/Omnidirectional_Stereo_Dataset/sunny` 是空目录，其余训练数据及 checkpoints 尚未准备，不能据此直接开始完整训练。准备好官方数据和权重后使用原项目入口，并显式指定数据目录，例如：

```bash
python train.py --dbname omnithings \
  --db_root /homes/zhaoxian/lzx_ws/datasets/Omnidirectional_Stereo_Dataset \
  --base_channel 32 --mixed_precision --ot_iter 3 --lec_weight 0.1 \
  --use_ge --use_digru --ge_scale 1 --total_epochs 30
```

## 代理

检查时 Docker 的 ~/.docker/config.json 自动注入 127.0.0.1:7894。bridge 容器应通过宿主网桥 `172.17.0.1:7894` 访问宿主代理；127.0.0.1 指容器自身。

实测 bridge 容器通过 172.17.0.1:7894 能访问清华镜像源（HTTP 200），但访问 PyPI 和 Docker Hub 仍在 TLS 握手阶段报 unexpected EOF。这说明正确代理地址可达，但部分目标站点的代理链路尚未恢复；不是登录错误，也没有关闭 TLS 校验。

当前构建使用 host 网络、清空代理，并通过清华镜像源下载。run.sh 默认也清空代理，可选择在创建容器时设置：

```bash
CONTAINER_PROXY=http://172.17.0.1:7894 \
  CONTAINER_NAME=pdf-omni-proxy GPUS=1 bash docker/run.sh
```

已运行容器可临时设置：

```bash
export http_proxy=http://172.17.0.1:7894
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
unset all_proxy ALL_PROXY
```

如果代理节点未恢复，访问部分站点仍可能失败。当前安装环境已完成，不依赖继续运行该代理。
