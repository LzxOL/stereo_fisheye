# module_baseline：双目鱼眼网络 V1

本目录提供独立的 `StereoBaseline`，默认两路 CNN 参数不共享，不使用 Descriptor Head、MRT、左右 MLP Fusion、OT、GE 或 GRU。现阶段只实现网络 Forward，不接数据集、训练入口或标定生成流程。

```text
左右 RGB 图像
  -> 两路 CNNEncoder 的 1/2 分支
  -> 给定采样网格上的 Spherical Sweeping
  -> 左右特征体 [B,C,D,H,W]
       -> GWC -> 3D Attention -> 沿 D 的 masked softmax -> A
       -> 左右各自 Compress3D -> 拼接体 Vcat
  -> A * Vcat -> Hourglass3D -> Geometry Encoding Volume
  -> Classifier cost -> masked softmax(-cost)
  -> 物理逆深度期望 -> 空间上采样 -> 取倒数得到深度
```

## 文件与阅读顺序

1. `network.py`：先看 `StereoBaseline.__init__` 里的尺寸和候选范围，再沿 Forward 中的 1~5 步读完整流程。
2. `encoder.py`：直接放入原 `module/mrt_layers.py` 的 `CNNEncoder`，层结构和计算不变，没有额外 CNN 包装类。主网络直接创建左右两个实例，默认独立，取原返回值中的 `x_2x`。
3. `sweeping.py`：沿用原 `PDFOmni.spherical_sweep` 的 5D `grid_sample` 采样方法；左侧只采一次，再沿候选轴广播。
4. `volume.py`：通道数直接写在类的 `__init__` 中；沿 1~5 步组织 GWC、ACV、GEV 和候选概率。
5. `tests/test_baseline.py`：数值采样、无效掩码、默认形状与小尺寸梯度检查。

原模块使用 `[B,C,H,W,D]` 的接口只在调用边界转换；本网络对外的特征体统一为 `[B,C,D,H,W]`。原 PDFOmni 的源码及入口不需要修改。

代码沿用 PDF-Omni 的直接写法：`__init__` 定义网络层，`forward` 按计算顺序展开；体构建内部直接返回 `probability, attention, intermediates`，不通过多层字典传递张量。网络最终输出字典的接口保持一致，便于查看深度和中间结果。

## 类中直接定义的参数

不再使用单独的配置类。常用尺寸放在 `StereoBaseline.__init__`，固定的网络通道数直接写在对应层的构造处。

| 参数 | 默认值 |
|---|---|
| CNN 输出通道 | 32，定义在主网络的两个 `CNNEncoder` 构造处 |
| `share_encoder` | False |
| `num_depth` | 96，直接使用 96 个候选 |
| `sweep_size` | `(64,128)`，与原鱼眼特征图尺寸分开 |
| `output_size` | `(128,256)` |
| GWC 分组数 | 8，定义在 `StereoVolumeGenerator` 中 |
| Attention 通道 | 32 |
| 压缩通道 | 每侧 16，拼接后 32 |
| GEV 通道 | 8 |
| Hourglass base | 8 |
| `min_depth` / `max_depth` | 1 / 50 米，仅作可配置的初始搜索范围 |

`D、H、W` 必须是 8 的倍数，以匹配原 3D 沙漏三次下采样。训练时还需保证最粗层 BatchNorm 有至少两个统计元素；例如测试使用 `B=1,D=16,H=8,W=16`，而非 `B=1,D=H=W=8`。

## 与原版 CNN 和球面扫描的关系

`CNNEncoder` 原样保留了 `conv0 -> conv1_down -> norm1 -> conv2 残差 -> conv2_down`，返回 `(x_4x, x_2x)`。主网络直接写 `_, left_feature = self.encoder_left(left_image)`，右侧同理，使用 1/2 分辨率特征构体。1/4 分支仍会计算，但输出未接入本网络的预测，因此该分支参数没有梯度。

球面扫描的几何含义是：对于左相机的一条参考射线 `r(y,x)`，按候选逆深度 `rho[k]` 取三维点 `P_left = r(y,x) / rho[k]`（`r` 为单位方向，深度表示沿射线的距离），再将它变换、投影到右相机，得到 `right_grid[y,x,k]`。同一左射线上的点投回左相机的位置不随半径变化，所以左侧可以只采一次。当前模块接收这些已经算好的网格，不在内部生成射线或相机投影。

ERP 是球面方向的一种二维排列方式，并非球面扫描的必要条件。这里按左参考射线网格的 H/W 排列，只需在左右均有观测的区域匹配，不要求相机覆盖 360°；同样也没有把任意输入网格自动变成球面投影。若以后传入 ERP 排列的射线网格，输出体的 H/W 才具有 ERP 含义。

采样核心与原版相同：把 `(u,v)` 补成 `(0,u,v)`，对 `[B,C,Hf,Wf,1]` 调用 5D `grid_sample`，得到 `[B,C,H,W,D]` 后调整为本网络使用的 `[B,C,D,H,W]`。与四路原版相比，这里只保留左右两路，支持每个样本不同的网格，并增加无效候选 mask；不进行前后/左右 MLP 融合。原 PDFOmni 文件和入口保持不变。

## Forward 接口

```python
from module_baseline import StereoBaseline

model = StereoBaseline()
result = model(
    left_image, right_image,
    left_grid, right_grid,
    valid_mask=valid_mask,          # 可省略
    return_intermediates=True,     # 学习/调试时查看各阶段特征体
)
```

输入约定：

- 两图各为浮点 RGB `[B,3,H0,W0]`，图像值的预处理由调用方保持一致。CNN 输出约为输入的 1/2 分辨率，左右输入尺寸可不同。
- `left_grid` 为 `[B,H,W,2]`，`right_grid` 为 `[B,H,W,D,2]`。末维均为归一化 `(u,v)`，指定从对应鱼眼特征图哪里取值，使用 `align_corners=True`。没有批次维的旧 LUT 需先显式加批次维。
- 左参考采样网格不随候选变化；右网格第 k 层必须对应 `model.inv_depth_values[k]`。代码不会把任意网格自动变成正确的球面投影网格。
- 参考网格按左鱼眼位置排列，不在模块内建立 ERP 表示。测试网格只验证计算连接，不代表真实投影。
- 可选 `valid_mask` 为 bool `[B,1,D,H,W]`。网络还会自动排除网格中的 NaN、Inf 和越界坐标；坐标范围有效不等于真实观测一定有效，调用方可以用该 mask 继续约束。
- 图像、网格、掩码和模型应位于同一设备。可在外部使用 CUDA autocast；网络的概率归一化及回归使用 float32。

候选值在模型中按以下顺序生成，并作为 buffer 随模型保存：

```text
inv_depth_values = linspace(1/max_depth, 1/min_depth, D)
```

候选索引越大表示距离越近。范围变更后，调用方必须提供对应范围、相同顺序的右采样网格。

## 默认张量形状

下表以 `B=1`、两图输入 `512×512` 为例。512×512 是形状示例，并非真实图像缩放要求。

| 中间量 | 形状 |
|---|---|
| 左右 CNN 特征 | 各 `[1,32,256,256]` |
| 左右采样体 | 各 `[1,32,96,64,128]` |
| GWC | `[1,8,96,64,128]` |
| Attention logits / A | `[1,1,96,64,128]` |
| 左右压缩体 | 各 `[1,16,96,64,128]` |
| Vcat / ACV | `[1,32,96,64,128]` |
| GEV | `[1,8,96,64,128]` |
| cost | `[1,1,96,64,128]` |
| 候选概率 | `[1,96,64,128]` |
| 低分辨率逆深度 / 深度 | `[1,1,64,128]` |
| 输出逆深度 / 深度 | `[1,1,128,256]` |

输出字段：

- `inv_depth`、`depth`、`valid_mask`：输出尺寸下的逆深度、深度与 bool 有效性。
- `inv_depth_low`、`depth_low`、`valid_mask_low`：工作网格尺寸下的对应结果。
- `probability`：`[B,D,H,W]`，有效像素的候选概率和为 1，全无效像素为 0。
- `attention`：`[B,1,D,H,W]` 的匹配注意力，与候选概率不是同一个分支。
- 开启 `return_intermediates` 后额外返回 `intermediates` 字典，包含上表中的特征/体及 `candidate_valid_mask`。默认不保留这些大型调试输出。

逆深度是 `sum(P_k * inv_depth_values[k])`，深度为其倒数。上采样仅改变空间尺寸，不对逆深度乘 2。无效位置输出 0，并通过有效掩码区分；边界插值用有效性权重归一化，避免被无效零值衰减。

## 小尺寸 CPU 调试

在仓库根目录、已安装 PyTorch 的环境中执行：

```python
import torch
from module_baseline import StereoBaseline

torch.set_num_threads(2)
model = StereoBaseline(
    num_depth=16,
    sweep_size=(8,16),
    output_size=(16,32),
).eval()
left = torch.randn(1,3,64,64)
right = torch.randn(1,3,64,64)
yy, xx = torch.meshgrid(
    torch.linspace(-0.8,0.8,8), torch.linspace(-0.8,0.8,16), indexing="ij")
left_grid = torch.stack([xx,yy], dim=-1)[None]
# 人工网格用于演示接口，实际输入需提供与候选值对应的网格。
right_grid = left_grid.unsqueeze(3).repeat(1,1,1,16,1)

with torch.no_grad():
    result = model(left, right, left_grid, right_grid, return_intermediates=True)
for name, value in result["intermediates"].items():
    print(name, tuple(value.shape))
print("depth", tuple(result["depth"].shape))  # (1,1,16,32)
```

检查命令：

```bash
python -m unittest discover -s module_baseline/tests -v
```

检查包括默认 96×64×128 工作体的 meta 形状验证，以及小尺寸 CPU 的真实前向/反向。meta 只验证形状，不代表完整默认尺寸已做数值推理或显存评估。
