# 训练与评估结果的离线可视化、报告和打包

这三个入口只读取已经保存的实验结果；不运行训练/评估、不读取原始数据集、不加载checkpoint，不需要GPU/PyG/Torch。运行环境需要已有的Python3.10+、NumPy及Matplotlib（沿用训练环境即可，不安装依赖）。三脚本共用本目录 `_report.py`，同步远端时请同步整个 `tran_evaluate/show/`。

## 远端直接使用

在**远端实际仓库根目录**执行：

```bash
bash tran_evaluate/show/Car-Design-ShapeNetCar.sh
bash tran_evaluate/show/Airfoil-Design-AirfRANS.sh
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh
```

三个命令分别生成三个项目各自的报告文件夹和ZIP；PDE脚本一次整理六个标准任务。脚本根据自身位置找到仓库，既适用于远端 `.../transolver/KCDLNO`，也适用于重命名后的checkout；**没有写死 `/home/hwz/CDLNO` 或旧Transolver_re路径**。从任意cwd也可以用上述脚本的绝对路径调用。不source可能过期的 `path.sh`，不读取config内旧绝对路径作为当前文件定位依据。

```text
<实际仓库>/Car-Design-ShapeNetCar/result_visualizations/<UTC时间戳_随机ID>/
<实际仓库>/Airfoil-Design-AirfRANS/result_visualizations/<UTC时间戳_随机ID>/
<实际仓库>/PDE-Solving-StandardBenchmark/result_visualizations/<UTC时间戳_随机ID>/
```

同级还会生成 `<UTC时间戳_随机ID>.zip`。文件夹含一个可离线打开的 `index.html`；下载ZIP、解压后直接打开即可浏览。每次使用新目录，不覆盖之前的报告；`--output-dir`若已存在也明确拒绝。

默认扫描当前仓库 `output/`、旧 `runs/`、对应子项目 `metrics/` 和 `scores/`。若已显式导出 `CDLNO_RUNS_ROOT` 则用它替代默认 `output/`。**每个数据集仅选择最后一次记录的验证损失最小的一个run/seed**，然后整理该run的训练曲线、评估结果和已有场图。PDE六个子任务分别选优，并生成一张六子图训练汇总和一张六子图评估汇总。未选run只列在筛选依据中，不进入绘图/字段复制流程。

**所有源实验（包括未选种子）、checkpoint和旧报告保持原样**。本工具没有删除、移动或回写源结果的流程，每次只创建新的报告目录和ZIP。

## 选取依据

先应用`--model/--seed/--task/--run-dir/--runs-root`筛选，再对每个task单独选一个run。使用最后一个有验证记录的epoch，**不是历次epoch中的最小值，也不使用独立评估结果选seed**。例如seed0前期0.001、最后0.03，seed1前期0.1、最后0.02，选择seed1。

| 数据集 | 比较的最后验证损失 |
|---|---|
| Darcy / Elasticity / Airfoil / Pipe | `validation_relative_l2` |
| NS / Plasticity | `test_full_loss`，即原训练循环的held-out监控值；保留其上游`test_*`命名，非独立验证集的新声明 |
| Car | `validation_velocity_mse + weight × validation_pressure_mse`，必须有保存的weight |
| AirfRANS | `validation_volume_mse + weight × validation_surface_mse`，使用保存的reg/weight，对应当前weighted MSE入口 |

Car/Air的旧`upstream_validation_log_value`分别受原外层分项交换、拼写分支影响，**不用于选优**；这里使用已正确记录的分项派生目标，原训练代码和结果不改写。Air若训练398epoch、最后验证在390epoch，则选优依据明确记为390，后续纯训练epoch不补验证值。周期场图仍取选中run已存的最后场图，不冒称是该验证epoch重算出的结果。

- 已记录为running/failed/interrupted/incomplete的run不参选；缺损/非有限/负损失、缺权重、损坏JSONL、缺成员验证不参选，也不退回更早的好数值。没有可参选run时明确留空，不从测试指标或checkpoint捏造依据。
- 旧run没有`train_results.json`、但有可解析最后验证记录时可参选，明确标注“完成状态未核实”。seed缺失保留unrecorded，不推断为0。
- 同一run若有多个Air成员，使用**各成员最后验证损失的算术均值**选择run，不挑其中最好的member、不跨seed平均；入选后各成员曲线仍分别保留。候选run之间成员数不同会提示配置差异。
- 完全同值时按明确记录的整数seed升序、再按绝对路径字典序选一个；缺seed排在已有seed之后，不按文件修改时间选取。
- 默认在本次纳入的run中选优；若混有不同模型/架构/训练设置/fold/split会提示，结果不代表受控的单纯seed比较。可用`--model`及重复`--run-dir`限定要比较的一组同配置实验。

`selection.json`/`selection.csv`记录**所有候选**、选择/排除原因、最后验证轮次、分项权重、成员loss、seed及路径。控制台及`--dry-run`也显示同一结果。`runs.csv`、曲线/评价CSV、HTML图片和复制场图仅包含获选run；其全部已存评估尝试保留，不再挑一次最好的测试结果。

## 常用选择

```bash
# 先检查每个数据集会选哪个seed、最后loss/epoch和排除原因；不创建输出。
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh --dry-run

# 在KCDNO实验中，各任务自动选择最后验证损失最小的seed。
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh --model kcdno

# 只整理KCDNO、seed0的六个PDE任务。
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh --model kcdno --seed 0

# 只整理Darcy（其余合法任务为airfoil/plasticity/elasticity/ns/pipe）。
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh --task darcy

# 精确指定一个或多个已有run，不猜最近一次。
bash tran_evaluate/show/Car-Design-ShapeNetCar.sh \
  --run-dir "$PWD/output/car/kcdno/实际run目录"

bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh \
  --run-dir "$PWD/output/darcy/kcdno/实际run目录1" \
  --run-dir "$PWD/output/ns/kcdno/实际run目录2"

# 结果不在默认位置：指定扫描根（可重复），覆盖默认搜索根。
bash tran_evaluate/show/Airfoil-Design-AirfRANS.sh --runs-root /实际远端磁盘/实验输出

# 复制获选run的所有已存周期场图及NPZ；不会重新纳入其他seed。
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh --fields all --include-arrays

# 只出曲线、评估图表，不复制场图；指定一个尚不存在的输出目录。
bash tran_evaluate/show/PDE-Solving-StandardBenchmark.sh \
  --fields none --output-dir /实际远端磁盘/新的报告目录

# 使用现有环境的具体解释器，或用300dpi减小PNG体积；PDF文字仍为矢量。
CDLNO_PYTHON=/已有环境/bin/python bash tran_evaluate/show/Car-Design-ShapeNetCar.sh --dpi 300
```

`--repo-root`可显式指定另一个真实checkout。`--model`按保存的family精确筛选：`kcdno`、`CDLNO`、`lrsa_matched`。`--seed`只筛选明确记录的seed；旧记录没seed时显示“unrecorded”，不会猜成0，也不从目录名臆测种子。无结果时仍输出索引与缺项说明，绝不生成虚构曲线。程序不会改变正在运行的训练；报告读取期间源文件若改变，会记录提示。这是逐文件读取，不承诺对活动训练目录作事务快照。

## 包内内容与指标口径

```text
报告目录/
  index.html                         # 总图册，支持离线浏览
  selection.json / selection.csv     # 所有候选的最终验证值/epoch及选取理由
  pde_selected_training.pdf / .png   # PDE专有，六任务各一个seed，2×3子图
  pde_selected_evaluation.pdf / .png # PDE专有，同样六任务，选中run的独立评价
  runs.csv                           # 仅选中run的模型、seed、fold/split、参数、状态
  training_history.csv               # 保留task/run/member/epoch/metric轴
  evaluation_metrics.csv             # 保留每次evaluation与status，不平均
  manifest.json                      # 来源绝对路径、SHA256、环境、配置、缺项
  WARNINGS.txt
  <task>/<run名_路径hash>/
    index.html
    run_summary.json
    config.json / architecture.json / train_results.json ...
    curves/member_0/                  # PDF、PNG、英文TXT/LaTeX图注
    evaluation/<评估ID>/              # 指标图、metrics.csv/tex、保存结果JSON
    field_snapshots/member_000/epoch_.../  # 原场图与图注/metadata，按需含NPZ
    legacy_figures/                   # 已存在的原绘图，保留原格式/分辨率
```

新曲线不平滑、不插值补数据点；稀疏验证使用**实际记录epoch**，连线只是视觉辅助。所有有限值都为正时使用对数纵轴，否则线性；获选run的中间非有限记录保留在CSV且图上为缺口，不替换为0。新图使用STIX/数学STIX，标题9pt、轴8pt、刻度/图例7pt；单栏3.5英寸，双面板/六子图汇总宽7.16英寸，默认600dpi PNG，PDF嵌入TrueType字体。图注标注模型、seed、模式、已记录架构宽深/latent数、fold/split和评价状态；最终投稿请按期刊栏宽排版。

PDE汇总按Darcy、Airfoil、Plasticity、Elasticity、NS、Pipe排列；图内标注每个获选seed、末次held-out loss及真实验证epoch。NS训练真值回填与held-out预测回填分别标注；Plasticity训练per-time和held-out full-field不冒称同一聚合量。没有合格run或完成的独立评价时，对应面板显示缺项，其他task照常绘制。独立评价汇总保留选中run的所有完成评价（E1/E2映射在图注），不选最小测试值。

| 项目 | 新生成的图表 | 依据/区别 |
|---|---|---|
| Darcy | 训练/held-out相对L2曲线、梯度正则项曲线、独立评估相对L2图 | `train_loss`是原field relative-L2；`derivative_regularizer`是未乘0.1的梯度项，不能称为总backward loss |
| Elasticity/Airfoil/Pipe | 训练/held-out相对L2曲线、独立评估图 | 原`train_loss`/`validation_relative_l2`/`relative_l2` |
| NS | 逐步误差、整段误差曲线及独立评估图 | 训练真值回填与评估预测回填是不同协议；保留原step/full指标，不改10→10 |
| Plasticity | 逐时间点及整段误差、独立评估图 | 原20个独立T；不存在的train_full曲线不补造 |
| ShapeNet-Car | 压力/速度MSE曲线、原目标分项可得时的目标曲线、压力/速度relative-L2与RMSE、阻力误差、Spearman | 内部正确命名pressure/velocity；旧外层交换名称的upstream日志只保留原数值，不误当真实训练目标；RMSE是原归一化输出上的指标 |
| AirfRANS | 加权训练loss；表面/体积均值及vx/vy/p/nut分项；独立评价MSE、气动力误差和Spearman | 按原`MSE_weighted`入口及原`score.json`，不把MSE称为relative-L2；验证不填满未执行epoch |
| AirfRANS配对数组存在时 | 升力/阻力预测—真值散点及相等线、Cp/Cf原表面散点 | 原`true_coefs[N,2]`、`pred_coefs_mean[N,group,2]`轴；误差棒为保存的成员std，不是置信区间。Cp倒置纵轴；不跨翼型上下表面连接点 |

AirfRANS `score.json` 的mean/std是原成员统计，不在这里重新跨seed平均。选中run的所有评估尝试都进表；failed/running/legacy_unverified指标不画成“完成评估”的柱状图。已有配对数组仍可生成带状态说明的散点。单例周期场图属于诊断，不能替代正式指标；复制既有图不重新裁剪色标/选择低误差案例。`--fields latest`只作用于选中run的周期场图，不丢弃该run的历史loss或较早评估记录。

Car原评价的`*_pred.npy/*_gt.npy`没有坐标和surf mask，且没有保存每案例阻力系数配对；脚本不会从它们捏造表面场图、拖曳系数散点或重算一个不同区域的误差。可使用之前已保存、带几何/mask的周期场图。AirfRANS边界层等其他原数组不新增物理重算；数值表和已有图完整保留其实际可用范围。

## 旧实验和缺项

首选结构化JSONL/JSON；六PDE若只有已有`train.log`，可按原`Epoch ... Train loss`、`Reg`、`train_step_loss/test_full_loss`和`rel_err`格式提取，源码0基epoch转为完成轮数。此路径明确注明控制台精度截断，不能冒充高精度原数值。旧Car/AirfRANS最终log不能恢复整条曲线；缺可判定最后验证值的run只列入筛选记录，不从PNG/OCR、最终测试结果反推验证值。旧Transolver是否能整理取决于其实际保留了哪些日志/图表，没有保证它自动具备新记录格式。

没有独立评估则显示缺项，不把训练中的验证冒称独立测试。新图的数值表和caption随包提供；原图保留原字体和分辨率，不宣称已把150dpi旧PNG升级成论文原图。模型权重不进入ZIP，也不反序列化；`.npy/.npz`只用`allow_pickle=False`，object/ragged数组会明确跳过，不扩大加载权限。

本机仅用合成**结果记录夹具**验证脚本，没有运行实际数据训练或重新评价。报告与证据：[离线结果整理交付](../../docs/CDLNO_RESULT_REPORTS.md)。
