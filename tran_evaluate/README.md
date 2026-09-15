# CDLNO 远端八任务训练与评估

更新（2026-09-15，统一实验记录）：新训练默认 `output/<数据集>/<UTC时间戳>/`，启动数据读取前创建 `config.json` 与日志；真实模型构造后补全参数量。训练/评估分别记入结果 JSON，复评使用独立子目录。八任务和三种前段模式均适用。详见 [目录、命令与边界](../docs/CDLNO_EXPERIMENT_OUTPUTS.md)。下述历史 A1/A2 目录规则由本次规则替代。

更新（补充A2）：八任务已接入 `--front-latent-mode full|no_sa|identity`。不传时新训练仍full；评估可从显式已有run的sidecar恢复mode，显式冲突拒绝。当前统一按时间戳创建目录；旧实验通过显式路径加载。`train_eval.sh TASK --front-latent-mode no_sa` 支持同配置训练后评估，余参仍最后覆盖。见 [八任务三模式命令](../docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md) 和 [A2报告](../docs/CDLNO_FRONT_ABLATION_A2.md)。以下A1及更早“尚未接入”状态保留为历史，当前以本段为准。

更新（补充A1）：共享核心已实现full/no_sa/identity，但八任务wrapper、CLI/任务JSON及本目录脚本尚未接入front_latent_mode，仍运行默认full。此时不要向启动脚本添加`--front-latent-mode`；等待用户指定后续接入阶段。见 [A1报告](../docs/CDLNO_FRONT_ABLATION_A1.md)。以下保留已有启动说明及此前状态记录。

所有脚本从根目录 `path.sh` 读取路径，转到各任务原工作目录执行原入口。一个脚本对应一个数据集，`train` / `eval` 分开执行；不会自动训练所有任务。前段默认完整 LRSA，也支持 `--front-latent-mode no_sa|identity`。

## 一条命令顺序完成训练和评估

新增 `train_eval.sh` 复用上述任务脚本：训练退出成功后才启动评估；训练失败立即停止并保留退出码。每次只选择一个数据集，不并行跑八个任务，也不改变原任务配置、损失或循环。

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w

# 先预览两条完整命令，不运行Python/数据/训练：
bash tran_evaluate/train_eval.sh darcy --gpu 0 --dry-run

# 以下每行独立，选择需要的数据集：
bash tran_evaluate/train_eval.sh darcy --gpu 0
bash tran_evaluate/train_eval.sh elasticity --gpu 0
bash tran_evaluate/train_eval.sh airfoil --gpu 0
bash tran_evaluate/train_eval.sh pipe --gpu 0
bash tran_evaluate/train_eval.sh ns --gpu 0
bash tran_evaluate/train_eval.sh plasticity --gpu 0
bash tran_evaluate/train_eval.sh car --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/train_eval.sh airfrans
```

保持下方任务配置表中的默认值：F2/L8、full前段、entry CDPA；Pipe M32，其余M64。两步共用命令中的模型参数。未设置 `CDLNO_RUN_TAG` 时，组合脚本生成一个含纳秒的UTC时间戳tag，两步使用同一个tag；如果已设置该变量，则沿用。实际run路径由两条原任务命令打印。tag不设置随机种子。显式run目录仍优先，必须尚不存在；评估已有训练请继续使用原 `TASK.sh eval`，不要重跑组合脚本。

```bash
# 显式保留可记忆的实验目录；训练评估共同使用F=3、L=8：
bash tran_evaluate/train_eval.sh darcy --gpu 0 --front-blocks 3 \
    --cdlno-run-dir "$PWD/output/darcy/F3_L8_run1"

# 关闭CDPA，仍是两个完整LRSA前段；两步自动共享该配置：
bash tran_evaluate/train_eval.sh ns --gpu 0 --cdpa-mode off \
    --cdlno-run-dir "$PWD/output/ns/F2_L8_off_run1"

# Car只供训练使用的参数，放在--train-args后，不传给eval parser：
bash tran_evaluate/train_eval.sh car --gpu 0 \
    --run_dir "$PWD/output/car/fold0_run1" \
    --train-args --preprocessed 1 --batch_size 1 --lr 0.001

# AirfRANS推荐统一配置Dataset，原脚本自动区分训练/评估路径：
CDLNO_AIRFRANS_DATASET=/actual/AirfRANS/Dataset CUDA_VISIBLE_DEVICES=0 \
    bash tran_evaluate/train_eval.sh airfrans
# 如要手动分别传路径：
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/train_eval.sh airfrans \
    --run_dir "$PWD/output/airfrans/full_run1" \
    --train-args --my_path /actual/AirfRANS/Dataset \
    --eval-args --my_path /actual/AirfRANS
```

共同参数放在 `--train-args` / `--eval-args` 标记之前；这两个标记后分别只传给训练/评估入口，`--dry-run` 无论放在哪都预览两步。run目录只能放共同参数中，避免加载其他实验。模型架构、Car fold/epochs、Air task/nmodel/训练预算等需要校验的参数应放共同区；不匹配仍由原sidecar检查拒绝，不绕过严格加载。

**Car 的完整阻力评估仍要求 fold0 和原固定路径 `/data/PDE_data/mlcfd_data/training_data/param0` 可访问，且与配置的raw目录是同一份数据。** 组合脚本保留这个检查；路径不满足时可能训练完成但评估被拒绝，不代表训练权重丢失。先确认下方Car限制。AirfRANS保持训练Dataset本身、评估父目录的区别；组合脚本拒绝把同一个 `--my_path` 直接共用。

三种前段模式已接入；未启动真实数据训练或评估。

## 远端路径与环境

本次填写的路径为：

```text
仓库：/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w
数据：/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data
六PDE数据公共目录：数据根/fno
```

`CDLNO_REPO_ROOT` 默认取 `path.sh` 实际所在目录，因此把仓库同步到上面的位置后会自动正确定位，不会继续使用 `/home/hwz/CDLNO`。数据默认使用你明确提供的远端位置。`path.sh` 在本轮开始时是空文件，现已填入可覆盖的环境变量；仅配置路径，不安装依赖、不创建目录或链接。

远端如果仍使用旧版 `path.sh`，它可能只导出 `REPO`、`DATA`、`FNO_DIR`、`CAR_DATA_DIR`、`CAR_SAVE_DIR`、`AIRFRANS_DATASET` 等名称。启动脚本和 `inspect_data.sh` 会兼容这些别名，并在未设置 `CDLNO_PYTHON` 时回退到当前环境的 `python`；旧版 `REPO` 指向其他Transolver副本时不会劫持当前 `CDLNO-w` 的入口路径。

保留你已工作的 Python3.10 / torch2.11 / CUDA12.8 / PyG 环境。脚本默认调用当前 `python`，可设置 `CDLNO_PYTHON=/absolute/path/to/existing/python`。脚本自动设置根目录 `PYTHONPATH`；不重装旧 requirements，不自动启用AMP/TF32/compile。NS等标准任务需要原有SciPy、NumPy、Matplotlib、einops、tqdm；工业任务仍需原图处理/VTK/PyVista等依赖。

## 先检查远端数据的实际格式

远端各文件尚未由本机实际读取。除用户确认的 `fno/elasticity` 外，下表更细的文件位置均依据当前loader构造；两个工业目录只是**待核实候选**。先在远端执行：

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w
bash tran_evaluate/inspect_data.sh > remote_data_inventory.json
```

检查程序只读取目录、文件头和manifest，报告：NPY shape/dtype/Fortran顺序、MAT字段及shape、AirfRANS split数量/样本文件、Car各fold原文件和缓存缺失数量。默认最多搜索6层、约20000目录项，列出实际文件的候选位置；不修改数据，不解压，不计算归一化，不运行task loader、PyG图构建或训练。不读取NPY数值、不反序列化pickle/checkpoint。MAT检查使用已有SciPy；检测到HDF5/v7.3则用已有h5py读取元数据并标注原scipy.loadmat不兼容，缺依赖只报告，不安装。

```bash
# 只检查path.sh配置的位置，省略递归搜索：
bash tran_evaluate/inspect_data.sh --no-search > remote_data_inventory.json
# 如数据套了更多目录，可显式扩大有限搜索：
bash tran_evaluate/inspect_data.sh --max-depth 8 --max-entries 40000 > remote_data_inventory.json
```

`exists=true` 或成功读取header不等于完整数据/标签/点序验收。远端文件未解压、字段名/轴顺序不同、MAT是v7.3时，先根据报告确认；脚本不会自动换格式、转置或使用另一个文件。`--data-root` 仅改变候选搜索根；改变各任务默认路径请修改 `path.sh` 或在启动时覆盖相应环境变量/CLI。

## 每个入口需要什么路径

下面以 `FNO=数据根/fno` 表示公共目录；环境变量都是 `path.sh` 中的实际变量名。

| 数据集 | 原入口的路径参数 / 默认值 | 入口实际追加或读取的文件 |
|---|---|---|
| Darcy | `--data_path $CDLNO_DARCY_ROOT`，默认FNO | `piececonst_r421_N1024_smooth1.mat`、`smooth2.mat`；coeff/sol `[S,421,421]` |
| Elasticity | `--data_path $CDLNO_ELASTICITY_ROOT`，默认FNO | `elasticity/Meshes/Random_UnitCell_XY_10.npy` `[972,2,S]`；`Random_UnitCell_sigma_10.npy` `[972,S]` |
| Airfoil | `--data_path $CDLNO_AIRFOIL_ROOT`，默认FNO/airfoil/naca | `NACA_Cylinder_X.npy`、`Y.npy`、`Q.npy`；Q[:,4]；221×51 |
| Pipe | `--data_path $CDLNO_PIPE_ROOT`，默认FNO/pipe | `Pipe_X.npy`、`Y.npy`、`Q.npy`；Q[:,0]；129×129 |
| NS | `--data_path $CDLNO_NS_ROOT`，默认FNO | `NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat`；u `[S,64,64,20]` |
| Plasticity | `--data_path $CDLNO_PLASTICITY_FILE`，默认FNO/plas_N987_T20.mat | **直接传MAT文件，不是目录**；input `[S,101]`，output `[S,101,31,20,4]` |
| Car | `--data_dir $CDLNO_CAR_RAW_ROOT`，候选data/mlcfd_data/training_data；`--save_dir $CDLNO_CAR_CACHE_ROOT`，候选data/mlcfd_data/preprocessed_data | 原始root/param0…8/sample/*.vtk；缓存root/param*/sample/{x,y,pos,surf,edge_index}.npy |
| AirfRANS | `$CDLNO_AIRFRANS_DATASET`，候选data/AirfRANS/Dataset | manifest.json；sample/sample_internal.vtu 和 sample_aerofoil.vtp |

**Elasticity虽存放在 `fno/elasticity`，参数必须传 `fno`，否则入口会重复拼接 `elasticity/elasticity`。** NS同样由入口追加完整子目录与MAT名称，不能直接传MAT文件。Airfoil与AirfRANS是两个不同任务。

例如，若检查发现Airfoil文件直接在 `fno/airfoil`，可以只覆盖该任务：

```bash
bash tran_evaluate/airfoil.sh train --data_path /actual/fno/airfoil --dry-run
```

所有用户参数放在默认值后，支持 `--flag value` 和 `--flag=value`。数据/run的相对路径按调用时工作目录解析，含空格时加引号。环境变量可在执行脚本前设置，例如 `CDLNO_FNO_ROOT=/actual/fno bash tran_evaluate/elasticity.sh train --dry-run`；若此前已source过path.sh并导出了各任务变量，修改父变量后也要同步更新子变量。

## 配置与16个启动命令

统一：CDLNO `L=8/F=2/P=6`、前段及点FFN ratio2、后段GEGLU ratio2、CDPA entry、chunk0、dropout0。默认M保持整个模型一致。是已确认的CDLNO设置，不声称与Transolver的M/heads/FFN所有模型超参数完全相同。

| 脚本 | d / h / M | epochs / batch | 位置及点FFN | 原训练配置 |
|---|---|---|---|---|
| darcy.sh | 128/8/64 | 500/4 | reference8，dense ConvFFN，downsample5 | AdamW/OneCycle，clip0.1，rL2+0.1梯度项 |
| elasticity.sh | 128/8/64 | 500/1 | xy，point FFN，972点 | AdamW/CosineAnnealing，clip0.1 |
| airfoil.sh | 128/4/64 | 500/4 | xy，dense ConvFFN，221×51 | AdamW/OneCycle，clip0.1 |
| pipe.sh | 128/4/32 | 500/8 | 原归一化xy，dense ConvFFN，129×129 | AdamW/OneCycle，clip0.1 |
| ns.sh | 256/8/64 | 500/2 | reference8+fx10，dense ConvFFN | AdamW/OneCycle，**无clip** |
| plasticity.sh | 128/8/64 | 500/8 | xy+fx1+原time embedding，dense ConvFFN | AdamW/原OneCycle节奏，clip0.1 |
| car.sh | 256/8/64 | 200/1 | x7/placeholder，point FFN | Adam/OneCycle，reg0.5，fold0 |
| airfrans.sh | 256/8/64 | **398/1** | x7+reference64，point FFN | Adam/OneCycle，原抽样32000、r0.05、max_neighbors64 |

lr参数均为0.001；六PDE的weight_decay均1e-5。OneCycle中的lr为max_lr，不是全程恒定；Elasticity保留CosineAnnealing按epoch step。配置分别与已确认JSON、现有CDLNO脚本和原入口核对；NS关闭裁剪沿用上一轮已完成的修正，其余配置本轮不改。

以下每行是独立命令，只选择本次要运行的一条。训练完成后再执行对应eval；所有命令均可追加 `--dry-run` 先看最终实参，dry-run不执行Python/数据/训练或创建run目录。

```bash
# 六个PDE benchmark
bash tran_evaluate/darcy.sh train --gpu 0
bash tran_evaluate/darcy.sh eval --gpu 0

bash tran_evaluate/elasticity.sh train --gpu 0
bash tran_evaluate/elasticity.sh eval --gpu 0

bash tran_evaluate/airfoil.sh train --gpu 0
bash tran_evaluate/airfoil.sh eval --gpu 0

bash tran_evaluate/pipe.sh train --gpu 0
bash tran_evaluate/pipe.sh eval --gpu 0

bash tran_evaluate/ns.sh train --gpu 0
bash tran_evaluate/ns.sh eval --gpu 0

bash tran_evaluate/plasticity.sh train --gpu 0
bash tran_evaluate/plasticity.sh eval --gpu 0

# ShapeNet-Car：先确认原始/缓存路径。
bash tran_evaluate/car.sh train --gpu 0
# 仅满足下文原阻力评估的固定路径/fold0限制后执行：
bash tran_evaluate/car.sh eval --gpu 0

# AirfRANS：原入口没有--gpu，按原cuda:0语义通过环境选择GPU。
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/airfrans.sh train
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/airfrans.sh eval
```

AirfRANS训练的 `--my_path` 是 **Dataset本身**；评价的 `--my_path` 是 **Dataset的父目录**，评价入口会追加 `/Dataset`。脚本根据同一 `CDLNO_AIRFRANS_DATASET` 自动分别传参。如手动覆盖CLI，必须保持两者语义。默认full/nmodel1/weight1，训练不自动启动完整score（score0）；其余split可用 `--task scarce/reynolds/aoa`，训练和评价同时传，并为该实验指定独立run。

## 输出目录与checkpoint

默认 `CDLNO_RUNS_ROOT=仓库根/output`，未指定tag时每次训练使用新的UTC时间戳；独立评估明确指定旧run：

- 六标准任务：`output/<task>/<timestamp>/`，checkpoint为 `model.pt`。
- Car：`output/car/<timestamp>/`，末轮整模型 `model_200.pth`。
- AirfRANS：`output/airfrans/<timestamp>/`，成员 `member_000/model`，模型列表 `CDLNO`。

train要求run尚不存在；eval用同一已有目录，校验原architecture.json后严格加载，写独立eval子目录，不覆盖sidecar/权重。不会自动寻找latest/best，不提供优化器状态resume。改变tag可开始另一次实验，训练和评价使用相同tag：

```bash
CDLNO_RUN_TAG=entry_seed2 bash tran_evaluate/ns.sh train --gpu 0
CDLNO_RUN_TAG=entry_seed2 bash tran_evaluate/ns.sh eval --gpu 0
```

tag只是目录名，**不会设置随机种子**。也可用六标准的 `--cdlno-run-dir /absolute/run` 或工业的 `--run_dir /absolute/run` 覆盖。切换CDPA模式、L/F/M/d/h/FFN等架构必须新run并在评价重复配置；同架构只改chunk允许加载原权重。Car需同时匹配fold/epochs/weight/r/cfd_mesh；Air需匹配task/nmodel/训练与抽样参数。原checkpoint保存频率不变；可信整模型加载仍只在已有局部位置使用weights_only=False，不全局放开。

## 保留的协议和实际限制

**NS** 保持黏度1e-5、前1000/后200、输入10帧输出10帧。每次forward产生1帧并重新建立latent/history；训练每步真值回填，10步loss累加后一次backward/optimizer/scheduler；测试预测回填，整10帧relativeL2。没有10→20/40或跨时间缓存。

**Plasticity** 保持空间101×31、fx1、T[B,1]、每次4通道；标签原轴序/随机时间排列保持，每batch20个时间点各自前向和参数更新，scheduler保持原节奏，无预测反馈。规则任务原可视化含硬编码尺寸，默认下采样不变，不承诺自选网格仍可直接绘图。

**Car** 默认缓存preprocessed1，`--save_dir` 是缓存目录而非checkpoint目录。即使使用缓存，原loader仍遍历raw的param0…8；原代码会跳过缺失样本/缓存，因此检查脚本会列出缺失数量，不能在缺样本时声称仍是论文789/100划分。输出velocity3/pressure1，原backward为全部节点速度MSE+0.5表面压力MSE，原surf/图处理/归一化不变。

Car原 `utils/drag_coefficient.py` 固定读取 `/data/PDE_data/mlcfd_data/training_data/param0/...vtk`。**你提供的 `/inspire/.../data` 路径不会自动改写这个常量，所以当前完整阻力评价不能仅靠新 `--data_dir` 就保证跑通。** 新eval脚本保留已有guard：要求fold0，且固定root与配置raw root指向同一份原始数据（可识别已存在的同目录符号链接）。本轮不创建链接、不改旧metrics；若远端没有此路径映射，脚本会明确报错，须以后单独处理路径兼容后再跑完整阻力指标。其他fold训练可显式指定，但完整评估不支持任意fold。

Car原train/test返回pressure、velocity，外层变量名反置，导致日志汇总实际为pressure+0.5velocity；内部真实backward目标正确。原文件保持，不把该日志当实际优化目标。原评价反归一化、压力/速度L2/RMSE、阻力误差/Spearman和原未同步CUDA的计时均保持。

**AirfRANS** 数据目录名/manifest与VTK场尚未在远端核实；保留每epoch抽样、验证重复抽样、图构造、散射平均、mask、边界后处理。完整评价依赖原PyVista/VTK及邻居图组件；以前Car训练成功不能代替这个工作流验收。

本轮仅验收启动参数、路径、检查工具和冻结证据，见[远端脚本报告](../docs/CDLNO_REMOTE_LAUNCHERS.md)。既有模型120项无数据回归证据见[此前审查](../docs/CDLNO_TRAINING_LAUNCH_REVIEW.md)，不冒充本轮重跑或远端真实训练结果。真实数据读取完整性、收敛、精度、显存和epoch效率仍待实际实验。
