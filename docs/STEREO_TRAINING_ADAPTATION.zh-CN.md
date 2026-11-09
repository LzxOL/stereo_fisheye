# OmniThings 双目训练：原 PDF-Omni 源码对照与增量适配

## 当前完成范围

最初按要求先完成代码。之后用户新增了上传GitHub、同步服务器和启动训练的目标，因此在发布前补做本地检查：开始改造前和发布前的**13项既有测试均通过**；小尺寸D=16,H=8,W=16的Split/Joint分别在checkpoint关闭/开启时完成前后向，左右CNN及GEV/CV梯度非零。真实数据smoke和CUDA观察需连接服务器后完成，不以本地检查冒充。

开始时已有未提交的 baseline 损失、测试和文档修改；它们已备份在 `.git/task-backups/20261110_013440/`，增量修改时保留了已有 Sequence Loss 的空掩码和无效 NaN 防护。原 `module/`、`train.py`、`scripts/train_omnithings.sh`、`dataset.py` 以及通用几何工具均不修改。

本地未找到 `../omnidata/omnithings/config.yaml` 或其他实际 OmniThings 配置。当前 Mac 无 CUDA。实际相机对、完整 TIFF 的尺寸/范围、真实有效监督比例及峰值显存均待挂载数据和 GPU 后验证，本文不提供伪造的相机参数或显存数字。

## 训练入口直接从原 train.py 修改

`train_stereo.py` 已直接复制原 `train.py` 再适配，并非另起一套优化框架。`count_parameters`、`fetch_optimizer`、`require_checkpoint` 三个函数与原函数 AST 一致。其余训练循环保留原执行顺序，只有模型、批次、标签和记录内容按双目任务调整。

| 项目 | 原 PDF-Omni 源码 | 当前适配 |
|---|---|---|
| 模型 | `module.network.PDFOmni` | 换成当前 `StereoBaseline`，不修改原类 |
| 优化器 | AdamW，lr=0.0005，weight_decay=0.00001，eps=1e-5 | `fetch_optimizer` 原样沿用 |
| Scheduler | OneCycleLR，total_steps=num_steps+100，pct_start=0.01，cycle_momentum=False，cos退火 | 原样沿用；num_steps由当前训练子集长度计算 |
| AMP | GradScaler('cuda')，unscale后裁剪，再step/update | 相同；若溢出跳过optimizer，也跳过scheduler |
| 梯度裁剪 | 默认1.0 | 相同，0可关闭 |
| Epoch / batch / iters | 正式脚本30 / 1 / 12 | YAML默认相同，CLI可覆盖 |
| mixed precision / OT / LEC | 原脚本开启AMP、ot_iter=3、lec_weight=0.1 | 正式默认相同；lec_weight=0关闭 |
| Sequence Loss | gamma_eff=0.9^(15/T)，只监督迭代输出 | 保持权重公式及默认监督范围；已有空掩码安全处理保留 |
| LEC | 原 `module/loss_functions.py::lec_loss` | 直接调用该函数；传 full_num_invdepth=D，局部GT不再除2 |
| 参数名 | --lr / --wdecay / --total_epochs | 保留这些别名，另外支持更直观名称 |
| --db_root | 父数据目录，再拼接dbname | 保留父目录语义并拼接omnithings；--root为直接目录，两者互斥 |
| 数据解析 | utils.dbhelper.loadDBConfigs | 直接复用，读取实际config和OcamModel |
| 官方划分 | train=1..4096及5121..10240，test=4097..5120 | 直接复用，共9216/1024；从train再划默认512个val，test不用于逐epoch验证 |
| 图像 | Augmentor.photo_aug、rgb2gray、normalizeImage | 复用；两路共享photo_aug，取消原四路特定的eraser索引 |
| CNN | 原CNNEncoder，随后U-Net/MRT | 原CNNEncoder不改；保留已确定的两路独立参数和x_2x输出 |
| Lookup | GEV+CV | joint保留；split仅CV，GEV用于Context |
| 恢复 | --restore_ckpt，恢复model/optimizer/scheduler/epoch | 同义，另恢复scaler/RNG并核对完整双目几何 |
| 预训练初始化 | --pretrain_ckpt，仅加载权重 | 同义；要求结构兼容的StereoBaseline，不猜测映射四目MRT权重 |
| checkpoint字段 | net_state_dict/net_opts/epoch/optimizer/scheduler/epoch_loss | 保留原字段，补充geometry/model_config/training_args/scaler/RNG |
| CUDA | DataParallel，脚本默认CUDA_VISIBLE_DEVICES=0 | 相同；增加CPU小尺寸调试分支，MPS不用于本5D采样训练 |
| 日志 | TensorBoard和epoch记录 | 保留标量日志，增加逐轮索引误差、熵、有效比例、显存、JSONL；四目专用拼图不照搬 |

**显式保留的 Baseline 差异：** 当前 corr_levels=2，GRU大核=5；原 train.py 的CLI会设为4和7。这里遵循已确定的双目结构，不悄悄改大核或金字塔层数；均已提供CLI参数。GE/PDD/DiGRU不启用。默认训练仍使用既有UpdateBlock，而不是重写一个GRU。

## 参数来源与优先级

1. `module_baseline/config/stereo_omnithings.yaml` 定义正式网络空间尺度与训练默认值。
2. CLI显式值覆盖对应默认值。
3. 相机标定、原始深度范围、完整GT角度定义由真实数据config和原loader解析；它们不能被网络分辨率反推。
4. 记录解析后的实际候选轴、角度、相机对、标定、invalid_mask哈希、图像尺寸到checkpoint。
5. 恢复训练时比较这些几何项；评估从checkpoint恢复模型和实际角度，不按新默认值重解释旧权重。

H/W/D不再在训练入口和数据模块各自硬编码。正式值为 D=96、H=160、W=240。原始输入保持768x800，CNN x_2x为384x400。沙漏沿D/H/W依次变为48/80/120、24/40/60、12/20/30，然后对称恢复；均能整除8。

## Split与Joint

原GWC、ACV、3D沙漏、OT及Soft-argmax继续共用。Generator新增可选的原Volume返回，但原CombinedGeoEncodingSampler和默认内部返回保留。

- split：`sum_D(P0*GEV)`得到8通道G2D；P0计算按有效候选数归一化的1通道熵；与左参考32通道拼成41通道Context输入。CV-only查询18通道。
- joint：保留32通道Context和162通道联合Lookup。不会为了记录熵而额外聚合无用的GEV。
- MotionEncoder的输入由显式corr_channels推导；GWC groups不再代表Lookup通道。Motion输出128、GRU输入160、Hidden64、Context32不变。
- GEV、P0、CV没有被无意detach；只保留原每轮索引detach策略。
- 初始概率的OT归一化不重写。聚合与熵使用float32，全无效/单候选熵为0。

正式配置的meta shape记录见 `docs/stereo_formal_shapes.meta.json`，其device明确为meta，不能解释为显存实测。真实训练的shape/dtype由 `--shape_debug` 写入 `shapes.json`。

## 数据几何

### 相机对

camN对应原Dataset读图约定：图像camN目录与config.cameras[N-1]匹配，报告另列cam_id。相机光轴使用OCam的+Z方向，经cam2rig旋转到Rig。分析脚本输出光心、基线、轴夹角、共同FOV、分深度覆盖和三角测量夹角，排序是明确标注的几何启发式，不代替真实训练效果。

没有标定不能指定最佳相机对。分析默认在参考相机可见方向上按立体角权重统计，而不是只在裁剪后的重叠窗口内报一个夸大的比例。

### 扫描网格

参考相机光心为原点，局部射线采用 az朝+X、elevation朝-Y（上方）、+Z为光轴。每个深度候选点依次通过reference.cam2rig和target.rig2cam，再调用现有OCam投影。左投影与深度无关，右投影随深度变化。

正式水平span=130°、elevation=[-45°,45°]暂定，中心从实际共同FOV估计。实际有效性始终由逐候选投影、图像边界、invalid_mask和CNN采样边界决定，不能把窗口内所有位置视为可见。

图像不缩放。特征坐标使用原像素/2，再调用现有pixelToGrid归一化到stride=2卷积中心，align_corners=True。这是相对原LUT直接复用图像归一化端点的一项明确坐标适配，不是重新实现OCam。

### GT与遮挡

读取完整原始TIFF，遵循原readInvdepth的逆深度定义；不用loadGTInvdepthIndex的裁剪结果。高度来自文件实际shape，原OmniThings loader定义gt_phi=90；不同版本可显式提供--gt_phi / --gt_width，须先核实。

流程：原makeSphericalRays恢复Rig点云 -> ref.rig2cam -> 径向距离与方向 -> 局部160x240栅格 -> range Z-buffer最近表面 -> 逆深度 -> 工作候选索引。无效和孔洞保持无效，不做resize填充。还检查GT点在左右相机中均可投影，并可启用两相机原始像素点云Z-buffer遮挡检查。

限制：仅有Rig中心观测到的点云，无法恢复新视角的全部表面；点云密度和局部栅格散射产生离散误差。Z-buffer不能被解释成完整场景无遮挡保证。

候选轴直接读取原配置深度范围，以原完整N轴每隔N/D取样。默认N=192、D=96，使用原层0,2,...,190，而不是0到191重新linspace。所有GT/输出都是工作轴0..95，不再做额外乘2或除2。

## 挂载数据后的命令（尚未执行）

先分析：

```bash
python scripts/analyze_omnithings_pairs.py --root /data/omnithings \
  --output results/omnithings_pairs.json
```

从报告选择真实相机对后设置REF_CAM和TARGET_CAM。以下变量不能在没有标定分析时随意代填。

先做单批次：

```bash
python train_stereo.py --root /data/omnithings \
  --ref_cam "$REF_CAM" --target_cam "$TARGET_CAM" \
  --smoke_test --shape_debug --num_workers 0 --output checkpoints/stereo_smoke
```

正式训练，不会自动启动：

```bash
bash scripts/train_stereo_omnithings.sh --root /data/omnithings \
  --ref_cam "$REF_CAM" --target_cam "$TARGET_CAM" \
  --output checkpoints/stereo_split
```

对应原参数写法也可使用 `--db_root /data`（实际读取/data/omnithings）、`--lr 0.0005 --wdecay 0.00001 --total_epochs 30`。

- Joint对照：同一数据与角度，增加`--volume_update_mode joint`，输出到新目录。
- 只用回归：`--lec_weight 0 --supervise_init`。默认预训练按当前任务使用0.1 LEC。
- 恢复：相同相机、配置与规划参数，增加`--restore_ckpt checkpoints/stereo_split/last.pt`。
- 权重初始化：`--pretrain_ckpt`只初始化兼容的双目权重，不恢复优化器；四目checkpoint不直接兼容。
- 不应在训练中每个epoch评估官方test。最终测试需显式运行下述test命令。

```bash
python eval_stereo.py --checkpoint checkpoints/stereo_split/last.pt --root /data/omnithings --split val
python eval_stereo.py --checkpoint checkpoints/stereo_split/last.pt --root /data/omnithings --split test
python scripts/profile_stereo.py --device cuda --compare_checkpointing --shape_debug
```

Profiling默认比较64x128、64x256、80x320和配置中的160x240，并比较扫描float32/amp。人工网格只用于计算成本测量，不冒充真实数据。脚本记录实际扫描dtype和网格dtype，不预先宣称CUDA grid_sample一定输出半精度。

## 显存与验证状态

单个实存的32通道、96x160x240、FP32特征体理论大小450 MiB；FP16为225 MiB。这只是元素存储量，不是训练峰值，左体还可能是D轴广播视图。完整峰值必须计入梯度、3D卷积、迭代和临时张量后实际测量。

默认CUDA AMP及batch=1；checkpointing仅作用于无BatchNorm运行统计的CNN/Context/GRU，避免重算改变3D BatchNorm状态。是否足以适配目标GPU仍待profiling，不能在当前Mac上给出CUDA峰值。

发布检查已完成源码静态对照、语法检查、既有13项测试、小尺寸Split/Joint及checkpoint前后向检查、正式尺寸meta检查和四个新CLI的help入口检查。旧测试明确固定到旧joint调试尺寸以保留对照。仍待真实标定几何数值核验、真实TIFF检查、真实单批次optimizer更新、checkpoint恢复验证及GPU profiling。网络不可达时不会声称训练已启动或没有OOM。
