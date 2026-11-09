# macOS 本地环境与调试

本项目的独立 Conda 环境名为 `pdf_omni`，使用 Python 3.10、PyTorch 2.11.0、torchvision 0.26.0，以及原 `requirements.txt` 指定的依赖。Apple Silicon 使用 macOS ARM64 安装包，不安装 CUDA。

## 进入环境

```bash
conda activate pdf_omni
cd /Users/linzhaoxian/lzx-ws/project/PDF-Omni
python --version
python -c 'import torch; print(torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

本机解释器路径：`/Users/linzhaoxian/anaconda3/envs/pdf_omni/bin/python`。在编辑器中选择这一解释器，即可给原模型源码设置断点。

这个环境单独配置了清华 PyPI 镜像，以改善本机下载速度；配置位于该环境自己的 `pip.conf`，未改用户全局 pip 设置。需要使用官方 PyPI 时，可在某次命令中追加 `--index-url https://pypi.org/simple`。

## 不需要数据和权重的小尺寸前向

```bash
python scripts/debug_local.py
```

该入口使用 CPU、随机四路灰度图、人工采样网格、随机初始化权重，开启 GE 和 DiGRU，前向两轮。默认输出各模块的张量形状，最后检查输出为 `[1,1,32,64]`、数值有限、索引范围正确。

它沿用原模型模块和通道数，仅缩小输入/ERP/候选轴，减少迭代次数。它不会更改原模型计算，不需要调用训练脚本，也不保存训练结果。它验证模型连接和运行环境，输出没有真实深度意义；真实投影、遮挡和深度精度需要图像、标定和训练权重另行验证。

命令行断点调试：

```bash
python -m pdb scripts/debug_local.py
```

例如进入调试器后可输入 `b module/volume_generator.py:313`，再输入 `c`，停在 `Generator.forward`；使用 `n` 逐句执行、`p front_cat.shape` 查看张量。行号以源码当前位置为准。

## Apple GPU 兼容性检查

```bash
python scripts/debug_local.py --device mps
```

这一命令直接测试本机和当前 PyTorch 的 MPS 支持，不隐式切回 CPU。`torch.backends.mps.is_available()` 为真只表示后端可用，不表示该模型每个算子都一定兼容。若有未实现算子，可另试官方 CPU 回退方式：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/debug_local.py --device mps
```

回退变量必须在启动 Python 前设置；回退成功也属于 CPU/GPU 混合执行，不代表全程 GPU 加速。没有必要为了学习数据流先解决所有 GPU 兼容性问题，CPU 入口可以独立使用。

## 依赖检查与重建

```bash
python -m pip check
```

仓库根目录的 `environment.macos.yml` 记录了 macOS 环境的直接依赖。需要在另一台兼容 Mac 上重建时，在尚无同名环境的前提下执行：

```bash
conda env create -f environment.macos.yml
```

该文件仅使用 conda-forge 创建 Python 环境，再通过 pip 安装指定版本的项目依赖，不依赖 Anaconda 默认软件源的条款确认，也不改变用户全局软件源设置。

`requirements.macos.lock.txt` 保存本次环境的完整 pip 包版本快照（包括间接依赖），用于追踪和复现；原 `requirements.txt` 保持不变。

原 `scripts/setup_env.sh` 使用 CUDA wheel 地址，原 `train.py` / `eval.py` 也包含 `.cuda()`。安装 macOS 环境不会自动改变这些入口；本地环境验证请使用 `scripts/debug_local.py`。当前文档不宣称原 CUDA 训练/评估脚本已完成 macOS 移植。

## 本机实测记录（2026-10-30）

- macOS ARM64，Python 3.10.22，PyTorch 2.11.0，torchvision 0.26.0，Open3D 0.19.0。
- `pip check` 通过；原 requirements 的全部 11 个依赖版本逐项核对通过。
- Open3D、Dataset、PDFOmni 以及相关科学计算库导入通过。
- CPU 两轮前向通过，输出 `[1,1,32,64]`，有限值/索引范围检查通过。
- 原生 MPS 两轮前向通过，未设置 CPU fallback，输出与范围检查通过。
- 该结果仅验证小尺寸合成输入的环境和算子兼容性，没有测试真实数据、权重、训练反向传播或完整分辨率的性能/精度。

PyTorch 官方资料：[macOS 安装](https://docs.pytorch.org/get-started/locally/)、[MPS 环境变量与 CPU 回退](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)。
