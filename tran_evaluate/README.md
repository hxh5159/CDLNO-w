# CDLNO：ShapeNet-Car 与 Navier–Stokes 训练/评估

这两个脚本调用原任务入口，保留原数据、损失、优化器、时间循环和评价处理。本轮只准备脚本和做无数据验收，没有启动真实训练或评估。

**默认采用已确认的 CDLNO 架构，训练协议对齐原 Transolver。** CDLNO 与 Transolver 的模型超参数并非全部相同，区别如下。依据为 Transolver [论文 v2](https://arxiv.org/pdf/2402.02366v2) 附录 B.1/B.3、Table 8，以及原仓库 commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。

| 配置 | Car 原 Transolver | Car 本脚本 CDLNO | NS 原 Transolver | NS 本脚本 CDLNO |
|---|---|---|---|---|
| d / heads / L | 256 / 8 / 8 | 相同 | 256 / 8 / 8 | 相同 |
| 模型结构 | 8层 Transolver | 2完整LRSA + bridge + entry CDPA + 6后段 + readout | 8层 Transolver | 同左侧 CDLNO |
| M | 32 | **64（已确认计划）** | 32 | **64（已确认计划）** |
| 原MLP / CDLNO前段与点FFN ratio | 2 | 2 | 1（原exp默认） | **2（已确认计划）** |
| CDLNO后段 GEGLU ratio | 不适用 | 2 | 不适用 | 2 |
| epochs / batch | 200 / 1 | 相同 | 500 / 2 | 相同 |
| optimizer / max_lr | Adam / 0.001 | 相同 | AdamW / 0.001 | 相同 |
| weight_decay | Adam默认0 | 相同 | 1e-5 | 相同 |
| 梯度裁剪 | 无 | 无 | 无 | **无** |
| scheduler | 原OneCycleLR | 相同 | 原OneCycleLR | 相同 |
| 条件 | x7，原placeholder | 相同语义 | reference8×8 + fx10 | 相同语义，stem74 |

NS 之前的 CDLNO JSON/脚本额外设置了 `max_grad_norm=0.1`，与原 `Transolver_NS.sh → exp_ns.py` 的 `None` 不一致。本次将 **CDLNO NS** 默认纠正为 `null/None`，移除其启动脚本的显式0.1；原exp与Transolver脚本未改。需要自选裁剪时仍可追加 `--max_grad_norm 0.1`，但那属于改变默认训练配置。**不要用0表示关闭裁剪，0会清零有限梯度。**

## 使用已有环境

沿用你已成功运行 Car 的 Python3.10 / torch2.11 / CUDA12.8 / PyG 环境，不重新安装旧 requirements，也不升级/降级依赖。Python3.11仍属于未在远端验证的候选。NS还使用 NumPy、SciPy、Matplotlib、einops、tqdm；Car完整评价还需要原 VTK、SciPy、PyYAML 等依赖。

激活环境后，脚本默认调用 `python`。也可选择现有解释器：

```bash
export CDLNO_PYTHON=/absolute/path/to/existing/environment/bin/python
```

脚本自动加入仓库根目录到 `PYTHONPATH`，并切到原任务工作目录，所以无须为这些命令重装共享包。默认 `MPLBACKEND=Agg` 用于远端无显示绘图；不改变计算精度，不自动开启AMP/TF32/compile。`--gpu` 原样转发：Car按设备索引选择，NS沿用原入口的 `CUDA_VISIBLE_DEVICES` 设置。

## 四条启动命令

在远端仓库根目录执行。数据路径按实际情况修改；训练目录必须尚不存在，评估使用同一个已有目录。下面 Car 数据路径同时满足原完整阻力评价的固定路径要求。

```bash
cd /home/hwz/CDLNO

# 1. Car训练：一个run、fold0、200epochs、batch1。
bash tran_evaluate/car.sh train \
  --data_dir /data/PDE_data/mlcfd_data/training_data \
  --save_dir /data/PDE_data/mlcfd_data/preprocessed_data \
  --run_dir /home/hwz/CDLNO/runs/CDLNO/car_entry_fold0 --gpu 0

# 2. Car完整评价：训练完成后单独执行。
bash tran_evaluate/car.sh eval \
  --data_dir /data/PDE_data/mlcfd_data/training_data \
  --save_dir /data/PDE_data/mlcfd_data/preprocessed_data \
  --run_dir /home/hwz/CDLNO/runs/CDLNO/car_entry_fold0 --gpu 0

# 3. NS训练：500epochs、batch2、10→10。
bash tran_evaluate/ns.sh train \
  --data_path /data/fno \
  --cdlno-run-dir /home/hwz/CDLNO/runs/CDLNO/ns_entry --gpu 0

# 4. NS评价：训练完成后单独执行。
bash tran_evaluate/ns.sh eval \
  --data_path /data/fno \
  --cdlno-run-dir /home/hwz/CDLNO/runs/CDLNO/ns_entry --gpu 0
```

任何一条追加 `--dry-run` 只显示完整命令，**不会调用Python、读取数据、创建训练目录或加载模型**。`bash tran_evaluate/car.sh help` / `bash tran_evaluate/ns.sh help` 显示简要说明。支持从其他目录调用绝对脚本路径；数据/运行目录的相对路径按调用时工作目录解释，含空格的路径要加引号。

## 数据和评价合同

**Car：** `--data_dir` 是包含 `param0`…`param8` 的原始数据根目录；即使使用缓存，原loader仍要遍历这些目录。`--save_dir` 是**预处理数据目录**，不是checkpoint目录。默认 `--preprocessed 1` 使用你已有的缓存；只有实际没有缓存、且你要运行原预处理时才在训练命令追加 `--preprocessed 0`。评价仍沿用原入口 `preprocessed=True`。不要混用不同数据版本/归一化缓存。

按完整官方数据，fold0为789训练、100测试；实际数量由原目录内容决定，本次未查验真实文件。x7（xyz3、sdf1、normal3）→ `[N,velocity3+pressure1]`，保持表面mask与节点顺序。反向目标是 `MSE(全部节点速度3) + 0.5*MSE(表面压力)`；每图更新一次，无多图batch。每10epoch及最后一轮验证。OneCycle保留原 `total_steps=(len(train_dataset)//batch_size+1)*epochs`、`final_div_factor=1000`，不擅自修正其步数。lr0.001是OneCycle的max_lr，不是全程恒定学习率。

**Car原代码有两项已确认的限制，原文件保持：**

1. `utils/drag_coefficient.py::cal_coefficient` 固定读取 `/data/PDE_data/mlcfd_data/training_data/param0/<sample>/...vtk`。单改 `--data_dir` **不能**重定向阻力几何；完整评价必须用fold0，原始数据必须在该固定路径可访问，且与传入 `--data_dir` 是同一份数据。已存在的同数据目录符号链接可用，脚本不自动建立链接或搬移数据。`check_car_eval.py` 在调用入口前检查此限制，避免混用几何算错指标；此检查不替代真实文件完整性校验。其他fold可以训练，**当前完整阻力评估脚本明确拒绝其他fold**。
2. `train.py` 返回压力/速度损失，但外层按速度/压力接收，导致进度条和最终JSON的 `train_loss/val_loss` 汇总实际为 `pressure + 0.5*velocity`。内部 `total_loss.backward()` 正确使用 `velocity + 0.5*pressure`，训练目标没有颠倒。不要把日志汇总当成实际优化目标；评价指标由原 `main_evaluation.py` 单独计算。此继承问题本轮没有改动。

Car评价保留反归一化、表面压力与体积速度相对L2/RMSE、阻力误差及Spearman；保留原系数实现与计时。原模型forward计时未做CUDA同步，不能把它当作严格GPU性能基准。训练/评价仍会构造原图/geom，不能将模型不用它解释为可以删除数据处理。

**NS：** `--data_path /data/fno` 对应文件：

```text
/data/fno/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat
```

MAT字段 `u`，原协议 `[1200,64,64,20]`，前1000训练、后200测试，前10帧输入、后10帧标签。每次模型调用 `x[B,4096,2], fx[B,4096,10] → [B,4096,1]`；规则索引reference64替换xy再拼fx10，stem74，不新增时间编码或normalizer。

训练每步回填真值，10个单步relative L2累加后一次backward/optimizer/scheduler更新；每epoch验证和独立eval均回填预测。每个物理时间步重新执行完整模型和建立局部历史，无跨时间latent缓存。独立eval输出全10帧relative L2。保持 `downsample=1`；原评价绘图硬编码64×64，自选下采样不能声称完整绘图仍兼容。

## checkpoint、覆盖参数与验收范围

Car在最后一轮保存整模型 `run/model_200.pth` 和 `log_200.json`；非默认epochs对应文件名改变。NS按原每100epoch（含ep0）及末轮频率写 `run/model.pt`，最终文件为最后保存的权重，不自动选“best”。两者都有 `architecture.json`；不是包含optimizer状态的resume协议。

训练拒绝复用已有run目录，避免覆盖。eval先读已有sidecar再比较，随后严格加载，结果写入新的 `eval_*` 子目录，不覆盖sidecar/权重。Car整模型仅使用项目已有的局部可信 `weights_only=False` 加载；必须来自你信任的训练产物。

用户参数位于默认参数后。改变d/h/L/F/M/FFN/mode时在新run训练，评价重复相同架构；Car还必须匹配fold/epochs/weight/r/cfd_mesh。`chunk`属于运行参数，可在同一checkpoint上改变；CDPA off/entry/every_block属于架构配置，不能只在eval临时换模式。

```bash
# 例：只核查chunk=1的加载命令，不实际评价。
bash tran_evaluate/car.sh eval --run_dir runs/CDLNO/car_entry_fold0 \
  --cdpa_source_chunk_size 1 --dry-run
bash tran_evaluate/ns.sh eval --cdlno-run-dir runs/CDLNO/ns_entry \
  --cdpa-source-chunk-size 1 --dry-run
```

本轮已运行的无数据回归、脚本实参检查和冻结证据见 [训练启动审查报告](../docs/CDLNO_TRAINING_LAUNCH_REVIEW.md)。合成前反向、真实PyG对象接口、checkpoint及本地小张量GPU成功，不证明真实文件读取完整、远端新模型环境兼容、收敛、论文精度或真实epoch耗时；这些需要你接下来运行实际实验得到。
