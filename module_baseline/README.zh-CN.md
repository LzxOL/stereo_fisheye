# module_baseline：在 PDF-Omni 原 module 上修改的双目 pipeline

本目录由当前工程的整个 `module/` 源码复制而来，再修改双目相关部分。原 `module/` 保持不变。

此前独立搭建、会返回物理深度的版本已备份到 [backups/module_baseline_before_pdf_copy_20261109_231829](../backups/module_baseline_before_pdf_copy_20261109_231829/)。本目录不再混用旧版的 `encoder.py`、`sweeping.py`、`volume.py`。

**当前主网络只写到 `context_feat = context_feat_volume[..., 0]`。`forward` 没有预测返回，调用结果为 `None`。** 初始 `init_disp` 和 `init_prob` 已在内部计算，可以在这行断点查看；尚未生成最终深度图。

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
  -> 当前阶段到此结束
```

## 推荐阅读顺序

| 顺序 | 文件与入口 | 重点 |
|---|---|---|
| 1 | `network.py::StereoBaseline.__init__` | 默认 D=96、工作网格 64x128；左右 CNN 和原 Generator |
| 2 | `mrt_layers.py::CNNEncoder` | 原版卷积、GroupNorm、残差连接；返回 `(x_4x, x_2x)` |
| 3 | `network.py::StereoBaseline.forward` | 两路输入到 context 的主干，仅在模块边界和关键维度变化处标注形状 |
| 4 | `network.py::StereoBaseline.spherical_sweep` | 原版 5D 采样；左侧采一次再扩展 D，右侧按候选逐层取样 |
| 5 | `volume_generator.py::Generator.forward` | 双目 Reference/Target -> GWC / ACV / 初始索引 / context |
| 6 | 同文件 `groupwise_corr_5d`、`ACVAttention`、`Compress3D`、`Hourglass3D` | 保留原版组件，以模块输入输出说明尺度 |
| 7 | 同文件 `Generator._sinkhorn` | 原版 OT 加上无效候选支持，随后计算连续候选索引 |
| 8 | 同文件 `channel_corr_5d`、`CombinedGeoEncodingSampler` | 原版 VC 和查表对象；当前尚未用它做迭代更新 |

这些步骤对应论文 Fig. 2 的 Feature Extraction、Volume Generation，以及进入 Iterative Refinement 前的初始化。Fig. 4 的 DiGRU、DAA / PDD 尚未接入当前主干。

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
| 上下文 | 融合后的 Reference 第 0 个候选切片 | 左 Reference 第 0 个候选切片 |
| 后续 GE / PDD / DiGRU / 上采样 | 主网络继续执行 | 源码保留，当前主网络不构建、不执行 |
| 物理深度换算 | 后续经数据集完成 | 当前没有此步骤，也不做上一版 baseline 的物理逆深度期望和上采样 |

`backbone.py`、`featurelayer.py`、`ge.py`、`loss_functions.py`、`mrt_model.py`、`mrt_utils.py`、`update.py` 均完整保留。它们出现在目录里不代表已接入双目主网络。`mrt_layers.py` 只为原 CNN 增加尺度注释；实际算法修改集中在 `network.py` 和 `volume_generator.py::Generator`。

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
    model(imgs, [left_grid, right_grid])  # 无预测返回
```

在 `network.py` 最后 `context_feat = ...` 一行设断点，查看 `init_disp`、`init_prob`、`context_feat_volume`，单步后查看 `context_feat`。测试通过 hook 获取 Generator 的内部返回，不为调试新增主网络输出接口。

```bash
python -m unittest discover -s module_baseline/tests -v
```

检查覆盖原 CNN / 球面采样一致性、同权重下原 Generator 的 ACV/OT/查表一致性、候选索引回归、无效候选与空 OT 行/列、CPU 前向/反向，以及默认尺寸 meta 形状。meta 检查不代表已完成默认大尺寸的实际推理或显存评估。
