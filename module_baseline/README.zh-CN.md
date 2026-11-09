# module_baseline：在 PDF-Omni 原 module 上修改的双目 pipeline

本目录由当前工程的整个 `module/` 源码复制而来，再修改双目相关部分。原 `module/` 保持不变。

此前独立搭建、会返回物理深度的版本已备份到 [backups/module_baseline_before_pdf_copy_20261109_231829](../backups/module_baseline_before_pdf_copy_20261109_231829/)。本目录不再混用旧版的 `encoder.py`、`sweeping.py`、`volume.py`。

主网络在原有初始 `init_disp` 后接入轻量 2D Context Network 和已有 `UpdateBlock`，返回初值、迭代结果及有效掩码。输入仍为 `imgs`、`grids` 和可选 `valid_mask`，新增可选 `iters`，默认迭代 12 次。所有输出索引保留工作网格尺寸与原候选单位，尚不转换为物理深度或上采样。

## 当前数据流

```text
imgs = [左鱼眼, 右鱼眼]
  -> 原 CNNEncoder 两个实例（默认权重独立）
  -> 取 x_2x，不使用 U-Net / MRT
  -> 左参考 Spherical Sweeping（给定 grids）
       左体 = Reference，右体 = Target
       不进行原版四路 MLP Fusion，不附加 G0..G3
  -> Generator
       Reference -> context_feat_volume
       Reference / Target -> GWC -> Attention -> masked softmax_D
       Reference / Target -> 各自 Compress3D -> 拼接
       Attention * 拼接体 -> ACV -> Hourglass3D -> Geometry Encoding Volume
       GEV -> classifier logits -> Sinkhorn OT -> 概率归一化 -> init_disp
       Reference / Target -> channel_corr -> Correlation Volume
       GEV / Correlation Volume -> 构建原版 geo_sampler
  -> context_feat = context_feat_volume[..., 0]
  -> Context Network：Conv2D + GroupNorm + ReLU + 两个 Residual Blocks
       Hidden Head：Conv2D + Tanh -> hidden_init
       Context Head：Conv2D + ReLU -> context_feature（各轮固定）
  -> net = hidden_init，idx = init_disp
  -> 重复 iters 次
       geo_sampler(idx) -> 局部 VE/VC 匹配特征
       UpdateBlock(net, context_feature, 匹配特征, idx)
         -> MotionEncoder -> SelectiveConvGRU -> 新 net、delta_idx
       idx = clamp(idx + delta_idx, 0, D-1)
       全无效像素置零，记录本轮 idx
  -> 返回初始索引、初始概率、各轮索引、最终索引、有效掩码
```

## 推荐阅读顺序

| 顺序 | 文件与入口 | 重点 |
|---|---|---|
| 1 | `network.py::StereoBaseline.__init__` | 默认 D=96、工作网格 64x128；左右 CNN、Generator、Context Network 和 UpdateBlock |
| 2 | `mrt_layers.py::CNNEncoder` | 原版卷积、GroupNorm、残差连接；返回 `(x_4x, x_2x)` |
| 3 | `network.py::StereoBaseline.forward` | 两路输入到初值，再接 Context / GRU；仅在关键边界标注形状 |
| 4 | `network.py::StereoBaseline.spherical_sweep` | 原版 5D 采样；左侧采一次再扩展 D，右侧按候选逐层取样 |
| 5 | `volume_generator.py::Generator.forward` | 双目 Reference/Target -> GWC / ACV / 初始索引 / context |
| 6 | 同文件 `groupwise_corr_5d`、`ACVAttention`、`Compress3D`、`Hourglass3D` | 保留原版组件，以模块输入输出说明尺度 |
| 7 | 同文件 `Generator._sinkhorn` | 原版 OT 加上无效候选支持，随后计算连续候选索引 |
| 8 | 同文件 `channel_corr_5d`、`CombinedGeoEncodingSampler` | VC 和当前索引附近的多层窗口查表 |
| 9 | `context.py::ContextNetwork` | 左参考二维特征 -> Hidden / Context 两个独立 Head |
| 10 | `update.py::UpdateBlock`、`MotionEncoder`、`SelectiveConvGRU` | 直接复用已有卷积 GRU，融合 Context 与 Motion，回归索引增量 |

这些步骤沿 Fig. 2 的特征、特征体和迭代路径组织。Context Network 是当前 baseline 新增的设计；本阶段复用普通 `UpdateBlock` 分支，没有接入 Fig. 4 的 DiGRU / PDD。已有 UpdateBlock 自带通道、空间注意力和小/大核 SelectiveConvGRU，不能将它等同于单个普通 3x3 ConvGRU。

## 与原版、上一版 baseline 的差别

| 部分 | 原版 PDF-Omni | 当前双目版 |
|---|---|---|
| 输入 | 四路 `[前,右,后,左]` | 两路 `[左,右]` |
| 特征提取 | 共享 CNN + U-Net + MRT | 原 `CNNEncoder`，默认左右参数独立，可设 `share_encoder=True` |
| CNN 输出 | CNN 后还经过 U-Net / MRT | 直接使用 `x_2x`，保留但不使用 `x_4x` |
| 扫描网格 | 围绕 Rig 参考原点的四路 ERP LUT | 左相机为参考，两路给定网格；不要求 ERP 或覆盖 360° |
| 特征体布局 | `[B,C,H,W,D]` | 跟随原版；3D 卷积分支为 `[B,C,D,H,W]` |
| Reference / Target | 前后、右左分别经 MLP 融合 | 左体、右体直接对应，取消融合 MLP |
| 附加相机坐标 Gi | 提供给四路融合 MLP | 不再需要 |
| GWC、压缩、ACV、沙漏 | 原版各组件 | 原样沿用 |
| `use_geo_acv` | 可附加完整 ERP 的经纬度 | 固定关闭，任意左参考网格不能套用该经纬度 |
| 初始 `disp` | classifier 正向 logits -> OT -> 归一化 -> 索引期望 | 沿用；候选无效时增加 mask，OT 与回归使用 float32 |
| 不可见区域 | 原版没有显式候选 mask | 从坐标与外部 mask 得到联合有效性，限制匹配注意力、ACV、VC 和 OT |
| 上下文 | Reference 切片直接初始化，后续按预测索引重查 | 左 Reference 切片先经新增 2D CNN，固定 Context 用于所有轮次 |
| 迭代 | UpdateBlock 或 DiGRU，按配置选择 | 复用已有 UpdateBlock，Hidden=64、Context=32 |
| GE / PDD / DiGRU / 上采样 | 按配置执行 | 源码保留，本阶段不启用 |
| 物理深度换算 | 后续经数据集完成 | 当前没有此步骤，也不做上一版 baseline 的物理逆深度期望和上采样 |

`update.py` 的类与计算原样复用；新阶段没有修改 Spherical Sweeping、GWC、ACV 或 Hourglass3D 的计算。`volume_generator.py` 只更新了两处过时注释。`backbone.py`、`featurelayer.py`、`ge.py`、`loss_functions.py`、`mrt_model.py`、`mrt_utils.py` 仍保留，尚未接入训练与损失入口。

## Context Network 到 GRU

左 Reference Volume 的深度轴是同一份左图特征的广播，所以 `[...,0]` 就是需要的二维特征。右相机候选 0 不可见不会将该左参考切片清空；左图自身无效时，输入对应位置已由球面扫描置零。

Context Network 全程保持 H/W：32 通道卷积 Stem、两个 32 通道残差块，再分两个独立 Head。Hidden Head 使用 Tanh 产生 64 通道 `hidden_init`，只作为第 0 轮隐藏状态；Context Head 使用 ReLU 产生 32 通道 `context_feature`，同一份张量用于所有轮次，不再按预测深度重新采样上下文。

`corr_levels=2`、`corr_radius=4`、`geo_ch=8`，因此 `geo_sampler` 输出通道为 `2*(8+1)*(2*4+1)=162`。已有 MotionEncoder 将这份几何特征与当前索引编码为 128 通道 Motion Feature，与 32 通道 Context 一起组成 160 通道 GRU 输入；GRU 更新 64 通道 Hidden，DepthHead 输出单通道增量。

沿用原版的迭代梯度方式：每轮切断上一轮索引的梯度，但保留 Hidden 的循环连接，以及查表到 VE/VC 的梯度。初始概率 / classifier 分支可通过返回的 `init_prob` 或 `init_disp` 单独监督；只监督最终输出时，不会经 detached 初值反传到 classifier。

本阶段调用 `no_upsample=True`：UpdateBlock 的上采样掩码头不执行，不放大索引单位。该头和 CNN 原有的 1/4 分支仍注册在模型中，但未参与当前预测，参数没有梯度。每轮结果限制在 `[0,D-1]`，全无效像素置零并返回 bool 有效掩码；有部分候选有效的像素仍允许回归连续索引。

## 初始 disp 的含义

```python
logits = self.classifier(geo_encoding_volume).squeeze(1)  # [B,D,H,W]
ot_log_prob = self._sinkhorn(logits, self.ot_iter, ...)
init_prob = exp(ot_log_prob) / (sum_D(exp(ot_log_prob)) + 1e-9)
init_disp = sum_D(init_prob * arange(D))                 # [B,1,H,W]
```

上面是流程示意，实际代码见 Generator。**此处沿用原版正向 logits，不取负号；回归的是索引期望（soft-argmax）。** 默认索引范围为 `[0,95]`，不是水平像素位移，也不是以米为单位的深度。网格第 k 层需要对应调用方第 k 个逆深度假设；当前未增加物理深度范围参数或转换步骤。

OT 沿用原版默认 3 次交替归一化，分数受同一张图中所有有效像素影响。mask 扩展会排除无效连接和完全无效的行/列，属于双目版相对原版的算法适配；全有效时与原版结果做过数值对照。有限次迭代不表示严格达到全部目标边缘分布。完全无效的像素 `init_prob=0`、`init_disp=0`，必须结合有效掩码解释。

## 网格与可见性约定

- `imgs=[left_image,right_image]`：各为浮点 RGB `[B,3,H0,W0]`，左右尺寸可不同，预处理由调用方保持一致。
- `grids=[left_grid,right_grid]`：分别为 `[B,H,W,2]`、`[B,H,W,D,2]`，末维为归一化 `(u,v)`，`align_corners=True`。
- 左参考射线上的点随半径改变时，投回左图的位置不变，因此左网格不含 D 轴；右投影随候选变化。
- 左/右网格均需有限且处于 `[-1,1]`；可额外传 bool `valid_mask=[B,1,D,H,W]`。候选有效性由这三者共同决定，坐标范围有效不代表已检测遮挡。
- 左参考体保留左图观测，不因右图某一候选无效就清空上下文；联合 mask 用于匹配分支。
- 当前只接收给定网格，不生成相机标定或三维投影网格。人工重复网格只能用来检查 pipeline，不能代替真实投影。

## 默认关键尺度

| 阶段 | 默认形状（两图均为 512x512 的示例） |
|---|---|
| 左右 CNN 的 x_2x | 各 `[B,32,256,256]` |
| Reference / Target / Context Volume | 各 `[B,32,64,128,96]` |
| GWC | `[B,8,96,64,128]` |
| Attention logits / A | `[B,1,96,64,128]` |
| 左右压缩体 | 各 `[B,16,96,64,128]` |
| Concat / ACV | `[B,32,96,64,128]` |
| GEV | `[B,8,96,64,128]` |
| classifier 去除单通道 / init_prob | 各 `[B,96,64,128]` |
| init_disp | `[B,1,64,128]` |
| Correlation Volume | `[B,1,96,64,128]` |
| context_feat | `[B,32,64,128]` |
| hidden_init / 每轮 Hidden | 各 `[B,64,64,128]` |
| context_feature | `[B,32,64,128]` |
| geo_sampler 查询结果 | `[B,162,64,128]` |
| Motion Feature / GRU 拼接输入 | `[B,128,64,128]` / `[B,160,64,128]` |
| 每轮索引增量 / 优化后索引 | 各 `[B,1,64,128]` |

D/H/W 需要是 8 的倍数，以匹配原沙漏的三次下采样。训练态最粗层 BatchNorm 还需要至少两个统计元素。

## 小尺寸运行与断点

```python
import torch
from module_baseline import StereoBaseline

model = StereoBaseline(num_depth=16, sweep_size=(8,16)).eval()
imgs = [torch.randn(1,3,64,64), torch.randn(1,3,64,64)]
y, x = torch.meshgrid(torch.linspace(-.8,.8,8), torch.linspace(-.8,.8,16), indexing="ij")
left_grid = torch.stack([x,y], dim=-1)[None]
right_grid = left_grid.unsqueeze(3).repeat(1,1,1,16,1)  # 仅用于接口调试的人工网格
with torch.no_grad():
    result = model(imgs, [left_grid, right_grid], iters=3)
print(result["disp"].shape)  # [1,1,8,16]，当前候选索引单位
```

输出字典：

- `init_disp`：初始索引 `[B,1,H,W]`。
- `init_prob`：初始概率 `[B,D,H,W]`，可用于初值监督。
- `disp_predictions`：长度为 iters 的列表，每项 `[B,1,H,W]`。
- `disp`：最后一轮索引 `[B,1,H,W]`；`iters=0` 时返回初值，列表为空。
- `valid_mask`：像素有效性 bool `[B,1,H,W]`，由任一候选是否有效决定。

可以在 `context_feat = ...`、`self.context_net(...)` 和 `self.update_block(...)` 处设断点，依次查看左参考特征、两个 Head、查表窗口和隐藏状态。训练可直接对 `disp_predictions` 设置多轮监督，并另外监督初值；当前未新增训练脚本或损失函数。

```bash
python -m unittest discover -s module_baseline/tests -v
```

检查覆盖原 CNN / 球面采样一致性、同权重下原 Generator 的 ACV/OT/查表一致性，以及新 Context 两个 Head、固定上下文与跨轮 Hidden 梯度、完整 CPU 前向/反向、索引增量单位、上下界、全无效区域和默认尺寸 meta 形状。meta 检查不代表默认大尺寸的实际推理或显存评估；本次没有进行训练收敛或深度精度实验。
