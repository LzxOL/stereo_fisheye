# 双目 Baseline：PDF-Omni 上的 Split / Joint 增量适配

模型源自当前工程的 `module/` 副本。原 `module/`、`train.py`、四目训练脚本和通用 OCam 工具保持不变。旧版独立实现仍在 `backups/module_baseline_before_pdf_copy_20261109_231829/`。

代码阶段后，按新增的上传和服务器训练目标进行了发布检查：13项既有测试再次通过；Split/Joint在小尺寸人工网格上的前后向、checkpoint开关及GEV/CV梯度检查通过；正式尺寸完成meta形状检查。真实OmniThings smoke和CUDA显存观察仍需服务器连接与实际数据，不能用这些本地检查代替。

## 集中配置

正式默认值在 `config/stereo_omnithings.yaml`，参数优先级为 YAML 默认值 -> 显式 CLI 覆盖。训练参数会保存到 checkpoint 和 `resolved_config.json`。

- 输入鱼眼：768x800，默认灰度归一化后复制为 3 通道；不隐式缩放图像或标定。
- 工作体：D=96，H=160，W=240；三次下采样后为 12x20x30。
- 角度暂定水平 130°，垂直 [-45°,45°]；水平中心由真实标定共同视场推导，所有候选逐点检查 FOV 和 invalid_mask。
- Batch size=1，CUDA AMP 开启；扫描默认 float32，几何网格始终 float32。
- 默认开启 CNN / Context / GRU activation checkpointing，不重算 3D BatchNorm。

构造 `StereoBaseline()` 时，D/H/W 从同一配置读取。训练使用的 AMP、checkpointing 等参数由入口显式传入；小尺寸调试可以显式覆盖 D/H/W。

## 网络数据流

```text
左右 CNNEncoder（原类，独立参数）
  -> x_2x -> 左光心 Spherical Sweeping
  -> 左 Reference / 右 Target，保留 [B,C,H,W,D] 布局
  -> 原 GWC / ACV Attention / 两侧压缩 / 拼接 / 3D Hourglass
  -> GEV + CV + 原 OT 概率 P0 + 原 soft-argmax 初始索引

split（默认）
  GEV * P0 -> 沿 D 求和 -> G2D（8 通道）
  P0 + 有效候选数 -> 归一化熵 U（1 通道）
  Concat(左二维特征32, G2D8, U1) -> Context Network（输入41）
  CV -> 深度金字塔 -> 每轮 Lookup（18 通道）

joint（消融）
  左二维特征32 -> 原 Context Network（输入32）
  原 CombinedGeoEncodingSampler(GEV, CV) -> Lookup（162 通道）

两模式共用原 MotionEncoder（输出128）和 UpdateBlock：
  Context32 + Motion128 -> SelectiveConvGRU 输入160
  Hidden64 -> DepthHead -> delta_idx
  idx = clamp(idx + delta_idx, 0, D-1)，全无效像素输出0
```

G2D / 熵使用 float32，没有 detach GEV 或 P0，也不重定义 OT 概率。熵只统计有效候选，数量 <=1 时输出0。每轮仍 detach 旧索引，Hidden 保留循环梯度。GE/PDD/DiGRU、空间上采样未接入。

## 阅读顺序

| 文件 | 内容 |
|---|---|
| `network.py` | 两模式连接、固定 Context、索引迭代；默认12轮 |
| `mrt_layers.py::CNNEncoder` | 原 CNN，返回1/4和1/2特征，主网络取后者 |
| `volume_generator.py` | 原 GWC、ACV、沙漏、OT；只新增可选的原始 Volume 返回 |
| `volume_lookup.py` | GEV 概率聚合、熵和 CV-only Lookup；原 Combined Sampler 保留 |
| `context.py` | 同一轻量网络，只将 Split 输入改为41通道；两个残差块和两个 Head |
| `update.py::UpdateBlock` | 只适配显式 corr_channels；GRU门、注意力、大小核分支不重写 |
| `stereo_geometry.py` | 复用 OCam / 刚体变换 / 原全景射线定义；生成新网格、GT重投影和Z-buffer |
| `omnithings_stereo_dataset.py` | 原 config.yaml / 官方分割 / 图像预处理的双目适配 |
| `../train_stereo.py` | 直接由原 train.py 修改的独立训练入口 |
| `training.py` | YAML/CLI映射、双目批次与评估工具 |

## 正式配置下的尺度（已通过meta形状检查，尚非CUDA实测）

| 张量 | B=1 的形状 |
|---|---|
| CNN x_2x | 两路各 `[1,32,384,400]` |
| 左网格 / 右网格 | `[1,160,240,2]` / `[1,160,240,96,2]` |
| 左右特征体 | 各 `[1,32,160,240,96]` |
| 候选 Mask | `[1,1,96,160,240]` |
| GWC / Attention | `[1,8,96,160,240]` / `[1,1,96,160,240]` |
| 两侧压缩体 | 各 `[1,16,96,160,240]` |
| Concat / ACV | 各 `[1,32,96,160,240]` |
| GEV / CV | `[1,8,96,160,240]` / `[1,1,96,160,240]` |
| P0 / 初始索引 | `[1,96,160,240]` / `[1,1,160,240]` |
| G2D / U / Split Context 输入 | `[1,8,160,240]` / `[1,1,160,240]` / `[1,41,160,240]` |
| CV Lookup / Motion / GRU 输入 | `[1,18,160,240]` / `[1,128,160,240]` / `[1,160,160,240]` |
| Hidden / Context | `[1,64,160,240]` / `[1,32,160,240]` |
| GT / 每轮索引 / 像素 Mask | 各 `[1,1,160,240]` |

## GT 和索引单位

完整 TIFF 按原 Dataset.readInvdepth 定义读取为逆深度；按其实际高宽和 OmniThings gt_phi=90 生成原 Rig 射线。先恢复 Rig 点云，再用参考相机 rig2cam 变换，计算径向距离并投到本地球面。每格保留最近点，不做插值填洞。

默认工作候选严格取完整192轴的 `[0,2,...,190]`，不是重新 linspace 到第191层。GT索引为 `(rho_ref-rho_work_min)/rho_work_step`，最终范围 `[0,95]`；网格、GT、OT、GRU、深度评估共用这条轴。范围外GT标无效，不夹到近/远端制造标签。

投影调用现有 OcamModel，图像目录 camN 对应原 YAML cameras[N-1]。源像素坐标除以2后，通过现有 pixelToGrid 归一化到 CNN 的 stride=2 特征中心，align_corners=True。不会用裁剪坐标把视场外点变为有效点。

监督同时要求参考/目标投影有效。可选的点云 range Z-buffer 检查两相机的已观测遮挡；Rig GT 不包含新视角下的所有可见表面，因此不能视为完整无遮挡保证。局部栅格的最近点散射也有离散化误差，孔洞保持无效。

## 损失与输出

按本次预训练要求，默认使用 `Sequence + 0.1 * LEC`；`--lec_weight 0` 可关闭LEC。Sequence保持原逐轮折扣公式，只默认监督迭代结果；已有空掩码/NaN防护保留。`--supervise_init` 可额外将初始索引加入同一种回归监督。

LEC 直接调用原 module/loss_functions.py，传入 `full_num_invdepth=D`；GT已经是工作轴索引，不能再除以2。原文件不修改。

网络输入保留 `imgs, grids, valid_mask`，可选 `iters`。输出字典保留 `init_disp`、`init_prob`、`disp_predictions`、`disp`、`valid_mask`，新增 `uncertainty`。`return_intermediates=True` 只用于形状/梯度调试。输出是连续候选索引，不是水平像素视差或米制深度。

## 使用入口

先运行 `scripts/analyze_omnithings_pairs.py` 分析真实标定，再显式指定 ref_cam / target_cam。没有数据时不能给出实际相机对排名。

相机对分析、smoke、训练、恢复、评估与profiling命令，以及与原训练入口逐项对齐的说明，见 [STEREO_TRAINING_ADAPTATION.zh-CN.md](../docs/STEREO_TRAINING_ADAPTATION.zh-CN.md)。
