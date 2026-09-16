# MSAR-LNO：八任务训练与评估（M9交付）

完整结构、参数/计算成本、有限GPU测量和验证边界见[最终报告](../../docs/MSAR_LNO_IMPLEMENTATION_REPORT.md)；[M8覆盖表](../../docs/MSAR_LNO_M8_ACCEPTANCE.md)区分静态、合成、原loss、PyG、checkpoint及GPU。以下训练命令仅交付，不会自动执行。

可用脚本：`darcy.sh`、`elasticity.sh`、`airfoil.sh`、`pipe.sh`、`ns.sh`、`plasticity.sh`、`car.sh`、`airfrans.sh`。调用形式是 `bash SCRIPT train|eval [原任务选项和MSAR选项]`，追加 `--dry-run` 只打印命令，不启动Python或访问数据。工业任务用法与限制见下方M7段落。

脚本定位其所在checkout并复用 `tran_evaluate/_common.sh`、根 `path.sh` 和原 `exp_*.py`；远端无需使用本机`/home/hwz/CDLNO`路径。原数据路径可通过对应环境变量或最后的`--data_path`覆盖。

| 脚本 | 原入口 | 路径变量 / 原始数据位置 |
|---|---|---|
| darcy.sh | exp_darcy.py | `CDLNO_DARCY_ROOT`：包含piececonst两个MAT的目录，默认`CDLNO_FNO_ROOT` |
| elasticity.sh | exp_elas.py | `CDLNO_ELASTICITY_ROOT`：Elasticity原npy文件目录，默认`CDLNO_FNO_ROOT` |
| airfoil.sh | exp_airfoil.py | `CDLNO_AIRFOIL_ROOT`：NACA_Cylinder_X/Y/Q.npy目录，默认fno/airfoil/naca |
| pipe.sh | exp_pipe.py | `CDLNO_PIPE_ROOT`：Pipe_X/Y/Q.npy目录，默认fno/pipe |
| ns.sh | exp_ns.py | `CDLNO_NS_ROOT`：含NavierStokes_V1e-5_N1200_T20子目录的目录，默认fno |
| plasticity.sh | exp_plas.py | `CDLNO_PLASTICITY_FILE`：完整plas_N987_T20.mat文件路径 |

默认Light、coverage floor/.01/.2。Full仅改变新profile结构，保留原任务训练预设：500epochs，batch分别4/1/4/8，lr.001，AdamW及原scheduler/clip等。Light/Full不按N裁剪M。`--coverage-mode off --coverage-weight 0`关闭正则；`--coverage-weight 0`也走无A的off路径。结构、训练目标默认还记录在`PDE-Solving-StandardBenchmark/configs/msar_lno/*.json`中；显式CLI最后覆盖。

M6新增NS batch2、默认不clip；Plasticity batch8、clip0.1，均保留500epochs/lr.001/AdamW/OneCycle。NS一次训练batch内10次forward、一次optimizer/scheduler；Plasticity20个时间点分别optimizer更新，batch末仅一次scheduler。新family没有卷积，也不改变空间点序和时间条件。

例如在**实际checkout根目录**：

```bash
bash tran_evaluate/msar_lno/darcy.sh train --gpu 0 --dry-run
bash tran_evaluate/msar_lno/elasticity.sh train --profile full --gpu 0 --dry-run
```

下面函数是可自行执行的训练成功后评估模板。每次创建独立目录，不搜索latest，不覆盖已有run；函数内没有缩小正式profile或epoch。将`TASK`替换成六个脚本名之一，即覆盖相应数据集。

```bash
# 在实际checkout根目录设置；DATA路径可先在path.sh或环境变量中配置。
msar_pair() {
    local task="$1" profile="$2" mode="$3" weight="$4"
    local run="$PWD/output/$task/msar_lno/$profile/coverage_$mode/$(date -u +%Y%m%dT%H%M%S%NZ)"
    bash "tran_evaluate/msar_lno/$task.sh" train \
        --gpu 0 --profile "$profile" --coverage-mode "$mode" --coverage-weight "$weight" \
        --msar-run-dir "$run" &&
    bash "tran_evaluate/msar_lno/$task.sh" eval --gpu 0 --msar-run-dir "$run"
}
```

六任务×四种设置的调用模板（按需选择，本文不会执行训练）：

| 任务 | Light coverage-on | Light off | Full coverage-on | Full off |
|---|---|---|---|---|
| Darcy | `msar_pair darcy light floor 0.01` | `msar_pair darcy light off 0` | `msar_pair darcy full floor 0.01` | `msar_pair darcy full off 0` |
| Elasticity | `msar_pair elasticity light floor 0.01` | `msar_pair elasticity light off 0` | `msar_pair elasticity full floor 0.01` | `msar_pair elasticity full off 0` |
| Airfoil | `msar_pair airfoil light floor 0.01` | `msar_pair airfoil light off 0` | `msar_pair airfoil full floor 0.01` | `msar_pair airfoil full off 0` |
| Pipe | `msar_pair pipe light floor 0.01` | `msar_pair pipe light off 0` | `msar_pair pipe full floor 0.01` | `msar_pair pipe full off 0` |
| NS | `msar_pair ns light floor 0.01` | `msar_pair ns light off 0` | `msar_pair ns full floor 0.01` | `msar_pair ns full off 0` |
| Plasticity | `msar_pair plasticity light floor 0.01` | `msar_pair plasticity light off 0` | `msar_pair plasticity full floor 0.01` | `msar_pair plasticity full off 0` |

也可独立评估：

```bash
bash tran_evaluate/msar_lno/pipe.sh eval --gpu 0 --msar-run-dir /absolute/path/to/existing/run
```

评估从run的`architecture.json`和`task.json`恢复保存的结构、训练目标与坐标/ref/downsample/ntrain合同；当前CLI默认Light不会覆盖Full权重。显式d/M/heads或输入合同冲突拒绝；显式coverage差异可用于纯eval，不写回sidecar。不传运行目录的训练自动创建`output/<task>/msar_lno/<profile>/coverage_<floor|off>/<UTC+uuid>`；显式`--save_name`作为该层级下的独占leaf，显式`--msar-run-dir`保持用户指定路径。

输出沿原统一记录协议：数据读取前创建目录/config.json/train.log/train_results.json，参数量在模型构造后实测；每epoch写train_history.jsonl；eval每次新建evaluations/目录。原`train_loss`仍保留任务定义；新日志`objective_pde/coverage_raw/coverage_weighted/objective_total`是**每优化步均值**，同时记录coverage_mode。Darcy的objective_pde包含原`.1*derivative_loss`，其旧train_loss仍仅field relative-L2。

周期可视化沿现有50完成epoch/final观察逻辑，预测调用不计算coverage。保存仍是原六PDE的`ep % 100 == 0`及final裸`model.pt`，不是完整optimizer/RNG断点续训档案，本阶段不改保存频率。原eval绘图的固定网格限制仍在：本阶段没有修改原数据/绘图代码来支持非默认downsample的所有导出图。5×7用于模型合同的合成验证，不是更改正式数据网格。

无数据验证及范围见 [M5报告](../../docs/MSAR_LNO_M5_STATIC_TASKS.md)。真实数据读取、真实训练/收敛、远端torch2.11/cu128未执行。

M6时间语义及证据见 [M6报告](../../docs/MSAR_LNO_M6_TEMPORAL_TASKS.md)。NS原PDE loss仍为10个逐帧relative-L2之和；每次forward先对四层coverage取均值，再对本次更新内的实际forward取均值，仅加一次`weight * mean_coverage`。因此不会额外乘rollout长度。Plasticity每个时间点的loss只加入该次四层coverage均值；不除以20、不把20次optimizer合并。日志中的四项目标以**优化步**为分母；原`train_step_loss`仍按样本数和时间点数归一，不应与`objective_pde`直接混同。

NS保持现有10帧输入→每次1帧→10帧预测回填评估；现有源码没有可用的20/40步CLI或对应长序列数据读取。本轮不添加不存在的`--rollout`选项或长时实验。CDLNO/KCDNO旧NS命令保留。MSAR时间wrapper沿既有固定网格合同：NS 64×64、Plasticity 101×31；不能把时间拼入N，非默认NS downsample不作为新支持项。

纯eval/周期图始终用预测Tensor，不计算coverage或diagnostics。需要模型级诊断时，显式`model.eval(); model(x, fx, T=..., return_aux=True, training_config=replace(model.training_config, diagnostics=True))`（`replace`来自`dataclasses`）；只返回该次小型no-grad统计，预测不变。训练CLI有`--diagnostics/--no-diagnostics`，默认关闭；现有任务eval循环不会因保存的训练开关自动收集aux，也不输出完整attention矩阵。

## M7：ShapeNet-Car / AirfRANS

工业脚本同样是 `train|eval`，复用现有Python环境、原main入口和数据路径；用户余参在最后覆盖。新增profile仅决定MSAR结构，不改原训练预设。Car默认200epochs、batch1、lr.001、weight.5、val_iter10；AirfRANS默认398epochs、batch1、lr.001、weight1、subsampling32000、r.05、max_neighbors64。两个任务保留Adam和原OneCycleLR，不混合多图，不使用卷积，也不删除图构造。

| 脚本 | 项目 / 真实参数 | 数据路径 |
|---|---|---|
| car.sh | Car-Design-ShapeNetCar/main.py 和 main_evaluation.py；`--cfd_model msar_lno`、`--gpu`、`--fold_id` | `CDLNO_CAR_RAW_ROOT` / `--data_dir` 原始数据，`CDLNO_CAR_CACHE_ROOT` / `--save_dir` 预处理缓存 |
| airfrans.sh | Airfoil-Design-AirfRANS/main.py 和 main_evaluation.py；`--model msar_lno`、`--task full|scarce|reynolds|aoa`、`--nmodel` | `CDLNO_AIRFRANS_DATASET` 指向 **Dataset**；train的`--my_path`是Dataset，eval的`--my_path`是Dataset父目录 |

在实际checkout根目录，先用 `--dry-run` 预览。例如：

```bash
bash tran_evaluate/msar_lno/car.sh train --profile light --gpu 0 --dry-run
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/msar_lno/airfrans.sh train --profile full --dry-run
```

AirfRANS原CLI没有`--gpu`，使用 `CUDA_VISIBLE_DEVICES`。不要给它传Car的选项。以下函数供用户按需运行；训练失败时不会启动评估，eval始终指向同一个确切run。此文档不会执行训练。

```bash
msar_car_pair() {
    local profile="$1" mode="$2" weight="$3"
    local run="$PWD/output/car/msar_lno/$profile/coverage_$mode/$(date -u +%Y%m%dT%H%M%S%NZ)"
    bash tran_evaluate/msar_lno/car.sh train --gpu 0 --fold_id 0 \
        --profile "$profile" --coverage-mode "$mode" --coverage-weight "$weight" \
        --msar-run-dir "$run" &&
    bash tran_evaluate/msar_lno/car.sh eval --gpu 0 --msar-run-dir "$run"
}
msar_airfrans_pair() {
    local profile="$1" mode="$2" weight="$3"
    local run="$PWD/output/airfrans/msar_lno/$profile/coverage_$mode/$(date -u +%Y%m%dT%H%M%S%NZ)"
    CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/msar_lno/airfrans.sh train \
        --task full --nmodel 1 --profile "$profile" \
        --coverage-mode "$mode" --coverage-weight "$weight" --msar-run-dir "$run" &&
    CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/msar_lno/airfrans.sh eval --msar-run-dir "$run"
}
```

| 任务 | Light coverage-on | Light off | Full coverage-on | Full off |
|---|---|---|---|---|
| ShapeNet-Car | `msar_car_pair light floor 0.01` | `msar_car_pair light off 0` | `msar_car_pair full floor 0.01` | `msar_car_pair full off 0` |
| AirfRANS | `msar_airfrans_pair light floor 0.01` | `msar_airfrans_pair light off 0` | `msar_airfrans_pair full floor 0.01` | `msar_airfrans_pair full off 0` |

Car完整drag评价仍有原仓库限制：fold0，`/data/PDE_data/mlcfd_data/training_data/param0`可读，且`--data_dir`与该固定根目录指向同一数据。新脚本复用原预检拒绝不一致路径；没有修改drag代码、创建软链接或假设远端路径已经满足。Car纯模型/表面mask评估的合成验证通过，不等于真实drag指标通过。

保存协议保持：Car整对象`model_<nb_epochs>.pth`；AirfRANS各成员整对象`member_NNN/model`及根目录模型列表`msar_lno`。新稳定类路径为`cdlno.msar_lno.industrial.CarModel`和`AirfRANSModel`。`architecture.json`保存完整family、resolved架构/版本/训练目标，`task.json`保存字段和任务合同；eval先读它们再加载可信本地pickle，严格验证类/结构/key/shape，coverage on/off的显式eval请求不改写训练sidecar。**没有新增resume，整模型文件不是完整optimizer/scheduler/RNG续训归档。**

Car的LPDE是全点速度MSE + weight×表面压力MSE；AirfRANS实际训练LPDE是体积四通道MSE + weight×表面四通道MSE。每个单图forward只加一次`coverage_weight * mean(四层raw coverage)`，epoch日志按optimizer step平均四项目标，surface mask不作为coverage source mask。无面积/体积权重时mu对图内所有源点均匀；latent层对其所有slot均匀。原Car外层日志的压力/速度变量交换保持，判断真实优化目标请看新`objective_pde/coverage_raw/coverage_weighted/objective_total`。

[八任务覆盖与M7实际证据](../../docs/MSAR_LNO_M7_INDUSTRIAL_TASKS.md)。本机真实PyG已用于合成单图、原loss、更新、eval和checkpoint；缺`torch_cluster`时AirfRANS完整radius_graph抽样epoch/评价未运行，不能由已通过的原训练函数和scatter片段推断完整链路成功。真实数据、收敛、物理系数准确率和远端环境仍未验证。
