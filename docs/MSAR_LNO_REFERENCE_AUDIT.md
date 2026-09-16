# MSAR-LNO M0：现有仓库审计与回归依据

2026-09-16。**本轮是 M1 之后补做的 M0**，仅审计、独立夹具和文档。已有 M1 配置实现被保留，没有重新实施 M1，也没有执行 M2。下文把真正的 pre-M1 证据与本轮新捕获的 post-M1 证据分开，不倒填阶段顺序。

结论：当前仓库可作为后续 MSAR-LNO 的集成基础；未发现已实现 M1 与本轮源码合同之间的阻断冲突。复用边界明确：**不能直接复用完整旧 block、最终 readout、CDPA 或 KCDNO history；Down 的 mask/权重输出和两来源融合需要新 family 的独立组件。** 此结论不代表 MSAR 数学、训练或模型 checkpoint 已实现。

## A. 审计范围、材料、版本与已有修改

- 仓库 `/home/hwz/CDLNO`；分支 `main`；HEAD `9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`。本轮开始已有 **23 个 tracked 文件修改**及未跟踪的 seed/visualization/show/MSAR 计划与 M1 文件；不是干净 HEAD。完整状态、原 diff、暂存区 diff 分别保存于 [status-before](msar_lno_audit/m0/status-before.txt)、[preexisting.diff](msar_lno_audit/m0/preexisting.diff)、[staged-before.diff](msar_lno_audit/m0/staged-before.diff)。暂存区为空。
- 适用指令为根 `AGENTS.md`；没有用旧阶段的历史停止文字覆盖本轮授权，也没有修改 AGENTS。保留已有 seed 接入、周期可视化、离线报告和 M1 工作。
- `rg --files --hidden` 与 `git ls-files` 合并清单共 **653 文件**，其中 **558 tracked**；**628 文本**、**25 排除的二进制**。157 Python 文件全部解析到函数/类/调用合同，70 shell 全文读取并 `bash -n`；67 Markdown/说明文档按段落读取和索引。源文件、配置、脚本、测试、数据工具、观察/性能工具都在清单内，不只包含预计修改的模型。
- 清单、每文件 SHA256/行数/符号、精确重复组见 [inventory](msar_lno_audit/m0/inventory.json)；[阅读清单与分组说明](msar_lno_audit/m0/READING_LEDGER.md) 列出实际材料及审查结论。[source-contracts](msar_lno_audit/m0/source-contracts.json) 保存每个 Python 函数 AST hash 和 CLI/加载/loss/step 等调用；[test-contracts](msar_lno_audit/m0/test-contracts.json) 保存测试断言合同；[configs-shells](msar_lno_audit/m0/configs-shells.json) 留存正式配置和所有 shell 全文。
- 分组仅在 **全文 SHA256 相同**时合并，10 组主要是重复历史回放证据；相似的 Transolver 模型、六个 exp、train/eval shell **没有按名称当作相同实现**。分别检查其坐标、通道、loss、归一化、时间与保存差异。全文读取/AST 索引不是数学正确性的证明；下文的源码判断另有真实模型回放与既有测试支持。
- 排除：`.git` 内部、`__pycache__`/测试缓存、构建产物、外部真实数据和实验目录遍历；25 个明确二进制（PNG/PDF/NPZ）仍保存文件 hash，逐项见清单。旧会话 PDF 本轮未重新 OCR，使用已有文本审查记录，不宣称本轮重读 PDF。Notebook 的文本/code cells纳入只读清单，未执行数据统计单元。许可证保留。

实际读取的设计与历史材料包括：`PLAN_MSAR_LNO/MRSA_LNO_arch.md`、`MSAR_LNO_Codex_Staged_Prompts.md`，CDLNO v1.2/v1.1、参考审计、阶段2—9、总报告/STATUS/需求矩阵、front A0—A4、输出/启动/可视化与 V1 文档；KCDNO 规格、staged prompts、K0—K10/总报告/STATUS/需求矩阵、seed 文档；根/三个任务 README、各 launcher README、memory 及现有 M1 报告。完整路径及阅读 hash 在阅读清单中。MSAR 架构文档包含早期对话候选，采用用户最新总控和最终四级/两来源/coverage-floor定义；不把早期 HistoryCross 当作批准设计。

源码快照（位于仓库外，不提交二进制）：

```text
/home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/source
```

它描述**补审开始时**的真实工作区。真正的 pre-M1 源码与6个核心夹具仍在 `msar-m1-before-n7y0eg1j/`，K0/pre-A1/A4/V1等旧产物均保留。M1 报告中“当时缺 M0”仍是准确历史，不删改成“当时已审计”。

## B. 实际结构与调用链

```text
仓库根
├── cdlno/                 CDLNO配置、公共原语、核心、wrapper、记录与独立归档工具
│   ├── kcdno/             kcdno / lrsa_matched配置、数学、wrapper、元数据与Run
│   └── msar_lno/          M1配置/profile/解析/metadata；没有模型类
├── PDE-Solving-StandardBenchmark/
│   ├── exp_{darcy,elas,airfoil,pipe,ns,plas}.py
│   ├── model_dict.py → model/*.py → 旧Transolver或共享wrapper
│   ├── cdlno_entry.py / kcdno_entry.py
│   └── configs/、scripts/、utils/{normalizer,testloss}.py
├── Car-Design-ShapeNetCar/
│   ├── main.py / main_evaluation.py / train.py
│   ├── models/{Transolver,CDLNO,KCDNO,cdlno_run}.py
│   └── dataset/、utils/drag_coefficient.py、configs/、scripts/
├── Airfoil-Design-AirfRANS/
│   ├── main.py / main_evaluation.py / train.py / params.yaml
│   ├── models/、dataset/、utils/metrics*.py
│   └── cdlno_entry.py / kcdno_entry.py / scripts/
├── tran_evaluate/         CDLNO调度；kcdno实际启动；kcdlno薄别名/seed；show离线报告
├── tools/cdlno_perf/      合成性能、完整矩阵成本及内存观察
├── tests/                数学reference、模型/任务/loss/加载/冻结/输出测试
└── docs/、PLAN_*/、memory/  规格、阶段报告与证据
```

共享包实际由根 `pyproject.toml` 的 `include=["cdlno*"]`发现，distribution为`cdlno`，目标 Python>=3.10,<3.12；没有另一个独立安装的 KCDLNO 包。本机没有安装 `cdlno` distribution；根启动器设置 checkout 根 `PYTHONPATH`，然后切到三个原子项目 cwd。夹具也按此方式起新进程。不能把当前源码导入成功写成 editable 安装或远端环境验收。

配置流并不统一成单个 trainer：

| 项目 | CLI → 参数/模型选择 → 保存/评估 |
|---|---|
| 六 PDE | exp 的真实 argparse声明 → `cdlno_entry.parse_args`，新家族委托 `cdlno.kcdno.entry.resolve_args` → `model_dict.get_model(args).Model` → `StaticRun` 或 `StandardRun`；`--eval`在同一个exp内选分支 |
| Car | main/main_evaluation parser → `models.cdlno_run.parse_args`，委托 `industrial_entry.resolve_car` → `--cfd_model`显式分支 → 原 `train.main`/独立main_evaluation；`CarRun`处理元数据/可信整对象 |
| Air | main/main_evaluation parser → 本地 `cdlno_entry.parse_args`/`air_entry.resolve_air` → `params.yaml`训练参数 + `--model`分支 → 原 `train.main`/`Results_test`；`AirRun`保存成员/列表 |

**导入风险：** 当前六个 `exp_*.py` 的真实数组/MAT读取位于 `main()`，由末尾 guard 调用；不能再把它们笼统说成“import即读数据”。但其顶层仍会解析 argv、设置设备并可能开始实验记录，所以仍禁止为审计直接 import。四个工业 `main.py/main_evaluation.py` 有顶层数据/manifest读取及流程执行，明确不能 import。测试只提取 parser/训练区段 AST，或导入经审查安全的模型、loss、normalizer、train函数；没有执行 exp/main 或其 `--help`。

## C. 八任务接口、原损失与评价

以下是**当前 CDLNO/KCDNO/matched wrapper** 的真实任务合同；旧 Transolver 的相应入口字段保持。`N` 为一次空间场点数，工业数据没有通用的多图 `[B,N,C]`合同。

| 任务 / 入口 | 数据与模型实参 | lift / 坐标 / 时间 | 输出与原训练 loss | 原 eval / normalizer |
|---|---|---|---|---|
| Darcy / `exp_darcy.py` | 两MAT `coeff/sol`；smooth1前1000训练，smooth2前200测试；421²通常downsample5→85²；`model(pos[B,N,2],fx[B,N,1])` | 规则索引网格的ref8²距离**替换xy**，再拼coeff；stem65，无placeholder/time | `[B,N,1]`；decode后 relative L2 +0.1×两个方向导数relative L2；预测边界先裁/补0后中心差分 | `UnitTransformer`输入/目标；eval原目标尺度field relative L2及原绘图；导数项不偷换成新物理loss |
| Elasticity / `exp_elas.py` | npy stress/xy原转置；1000/末200，N972；`model(pos,None)` | xy2→MLP+placeholder，point路径 | `[B,N,1]`stress；decode后 `TestLoss` relative L2 | y normalizer；点顺序不变；原散点评价 |
| Airfoil / `exp_airfoil.py` | X/Y网格和 `Q[:,4]`；1000/接着200，221×51；`model(pos,None)` | 物理xy2→MLP+placeholder | `[B,N,1]`Mach；relative L2 | 无normalizer；原物理网格/场输出与relative L2 |
| Pipe / `exp_pipe.py` | X/Y和 `Q[:,0]`；前1000/末200（使用1200样本），129²；`model(pos,None)` | 原入口先normalize xy；xy2→MLP+placeholder | `[B,N,1]`velocity；decode后relative L2 | x/y normalizer；原绘图保留decode坐标 |
| NS / `exp_ns.py` | MAT `u`，前1000/末200，64²；`model(pos,window[B,4096,10])` | ref64替换xy+10帧→stem74；无time_fc/placeholder | 每次 `[B,4096,1]`；10次逐帧relative L2求和，一次backward/optimizer/scheduler；训练回填真值 | 10次预测回填；step及整段trajectory loss；无跨forward latent/history缓存；不新增10→20/40 |
| Plasticity / `exp_plas.py` | `input/output` MAT；900/末80，空间101×31；`model(pos,fx[B,3131,1],T[B,1])` | 原meshgrid点序；fx normalizer；xy+fx→stem3；sin/cos时间嵌入→SiLU time_fc后相加 | 每次 `[B,3131,4]`；20时间点分别loss/backward/optimizer，共20次；batch后scheduler一次 | 20个独立T条件、输出拼回 `[B,N,4,20]`；不把T合到N；原 `random_collate_fn`按样本置换时间 |
| Car / `main.py`，`main_evaluation.py` | `GraphDataset`返回 `(cfd_data,geom_data)`；cfd有x7/pos3/y4/surf/edge/batch/ptr；可变N，正式batch1 | x7=xyz3+sdf1+normal3→MLP+placeholder；当前默认不加reference；wrapper不读y、不使用geom，但不删loader图/geom工作 | `[N,4]`velocity3/pressure1；velocity MSE（所有点）+reg×surface pressure MSE；reg默认0.5 | 训练归一化系数恢复；原vol/surface指标、drag/Spearman；fold0默认，raw阻力硬路径限制见下 |
| AirfRANS / `main.py`，`main_evaluation.py` | manifest full/scarce/reynolds/aoa；Data x7/pos2/y4/surf，训练抽样32000；`model(data)` | x7=xy2+inlet2+sdf1+normal2，append原pos在[-2,4]×[-1.5,1.5]域ref64→stem71+placeholder | `[N,4]`vx/vy/p/nut；真实入口传`MSE_weighted`，即volume四通道MSE+weight×surface四通道MSE，默认weight1 | 重复采样/scatter均值/边界归零/VTK物理指标；场指标为MSE，非relative L2；force/Cp/Cf/边界层/Spearman按原代码 |

参数/协议不可从旧 parser 裸默认猜测：

| 任务 | CDLNO与KCDNO主预设 d/h/M | epochs / batch / lr | 原 optimizer / scheduler | 保存与运行位置 |
|---|---|---|---|---|
| Darcy | 128/8/64 | 500 /4 /.001 | AdamW / OneCycle，每batch | PDE协议，见下 |
| Elasticity | 128/8/64 | 500 /1 /.001 | AdamW / **CosineAnnealingLR每epoch** | PDE协议 |
| Airfoil | 128/4/64 | 500 /4 /.001 | AdamW / OneCycle | PDE协议 |
| Pipe | 128/4/32 | 500 /8 /.001 | AdamW / OneCycle | PDE协议 |
| NS | 256/8/64 | 500 /2 /.001 | AdamW / OneCycle；默认无clip | PDE协议 |
| Plasticity | 128/8/64 | 500 /8 /.001 | AdamW / OneCycle；20次optimizer/一次scheduler | PDE协议 |
| Car | 256/8/64 | 200 /1 /.001 | Adam / OneCycle | whole object `model_<nb_epochs>.pth` |
| AirfRANS | 256/8/64 | **398** /1 /.001 | Adam / OneCycle；val_iter10，20重复抽样 | 成员`member_000/model`整对象、run根`CDLNO`/`kcdno`/`lrsa_matched`列表 |

CDLNO L8/F2/P6、front默认full；KCDNO L8/r16/history all；matched L8/full。规则点域ConvFFN适用于Darcy/Airfoil/Pipe/NS/Plasticity，Elasticity/Car/Air为PointFFN。`transolver_shape_match`只影响新K家族：Airfoil h8、Pipe h8/M64、NS/Car/Air M32。所有数值来自当前JSON/YAML/profile/launcher，不为 MSAR 重用任务特定d/M。

**checkpoint 与路径具体区别：**

- 六PDE新家族在run内保存纯`model.pt` state_dict，严格加载；`architecture.json`及K家族`task.json`约束wrapper与结构。原Transolver保留 `checkpoints/<save_name>.pt`/`results/<save_name>`，部分旧eval用`strict=False`，Pipe旧eval还有resave；这些是原路径，不是新family可照搬的兼容策略。
- CDLNO默认 `output/<task>/<UTC timestamp>`；KCDNO直接入口 `output/<task>/<family>/<profile>/<timestamp+结构digest>`；`kcdno/train_eval.sh`/seed队列显式传独立run时允许 `output/<task>/<family>/<tag>`，profile仍在元数据中。不是所有已有run路径都有profile这一层。显式旧路径仍可用，不猜latest、不覆盖训练配置；每次eval写 `evaluations/<timestamp>/results.json`并更新索引。
- Car旧分支保存整模型，旧稳定路径 `models.Transolver.Model`/`models.CDLNO.Model`/`models.KCDNO.Model`；Air旧类 `models.Transolver.Transolver`，CDLNO/K家族稳定共享类 `cdlno.airfrans.AirfRANSModel` / `cdlno.kcdno.airfrans.AirfRANSModel`。本地`models/*.py`alias并不改变真实pickle类名。Air还保留成员与列表两级保存，不统一成state_dict。
- 新CDLNO/K家族eval先读sidecar/任务合同再验证模型；工业局部`weights_only=False`仅加载明确可信本地对象，进一步核对类、配置、非权重行为与严格state。缺family旧文件仍按旧规则，不能猜成MSAR；CDLNO缺front字段只允许已识别的pre-A1 full。
- **现有任务保存不是完整断点续训。** `TrainingArchive/EpochObserver` V1工具有optimizer/scheduler/RNG/normalizer归档能力，但任务尚未接其resume；工具还绑定旧CDLNO sidecar。六PDE旧 `ep %100==0` 是零基epoch，完成1/101/201/301/401轮及final保存当前权重；工业沿用final整对象/列表。不能把它宣传成每100完成epoch的精确续训checkpoint。
- 周期可视化**现在已经接入**新模型家族：每50完成epoch及final，两固定holdout case、RNG隔离、独立输出；旧AGENTS/V1中“尚未接周期图”是历史。它没有顺带接resume。`tran_evaluate/show/_report.py`只读取结果，每子任务选最后记录验证loss最小的run/seed，PDE六面板汇总；不删除其他seed，不用正式test选seed，Air398的最后验证390如实保留。

## D. 实际模型与可复用部件

| 实际注册/名称 | 家族/配置与真实类 | forward / 保存边界 |
|---|---|---|
| `Transolver_Irregular_Mesh`、`Transolver_Structured_Mesh_2D`、`Transolver_Structured_Mesh_3D` | PDE `model_dict.get_model` →同名模块`.Model`；3D存在但不是八任务默认 | `forward(x,fx,T=None)`；原Physics-Attention slice/softmax/deslice；纯state_dict |
| 工业`Transolver` | Car `models.Transolver.Model`；Air `models.Transolver.Transolver`；main选择 | tuple/Data→N4；整对象/列表；不能假设原模型已校验多图隔离 |
| `CDLNO` | `CDLNOArchitectureConfig`→`cdlno.core.CDLNO`；PDE三个薄`model/CDLNO_*`、Car独立Model、Air共享Model | full/no_sa/identity；CDPA off/entry/every_block是另一独立轴；F/L→P、Bridge/rear/readout保留；按任务原保存格式 |
| `kcdno`（用户脚本目录写kcdlno） | `KCDNOArchitectureConfig`→`cdlno.kcdno.core.KCDNO`；PDE `StandardModel`、Car`models.KCDNO.Model`、Air共享类 | L个点域block，FFN1→kernel history→FFN2；all/off，最后自身Up后LN/head；按任务原格式 |
| 生产`lrsa_matched` | `MatchedLRSAConfig`→`cdlno.kcdno.matched.MatchedLRSA`；使用K任务wrapper/family分流 | 完整L层LRSA，**固定full**；八任务已接入；不是官方LRSA训练复现 |
| 性能`lrsa_matched` | `tools.cdlno_perf.models.LRSAMatched`，不同于上行生产类 | 只用于性能；生产对照在工具中叫`lrsa_matched_trainable`；两者都锁full |
| Air `MLP`、`PointNet`、`GraphSAGE`、`GUNet` | `params.yaml`+main；MLP实际选择`models.NN.NN`，其余同名类；encoder/decoder用`models.MLP.MLP` | 真实Data forward；GraphSAGE用edge，PointNet用batch全局池化，GUNet多尺度图/nearest；整对象/列表 |
| `msar_lno` | **仅M1配置注册协议** `cdlno.msar_lno.registry.family_for_model_key` | 无core、无任务模型factory构造器、无可运行训练入口，不可标作已训练支持 |

Transolver++、Transolver-3、LinearNO只在论文/说明中出现，不是当前八任务可选的实现。根 `Physics_Attention.py`是参考，三个项目使用本地源码；当前原Transolver各副本均是显式QK/softmax/AV，不能因新模块使用SDPA而声称全部旧模型也用SDPA。

| MSAR部件 | 现有文件/类/方法 | 复用判定与差异 |
|---|---|---|
| input lift | `cdlno.standard._axis/_grid`、`StaticStandardModel`/`TemporalStandardModel.__init__/forward`；K `StandardModel.__init__/forward`；Car/Air `preprocess` | **复用语义，需新wrapper**。MLP/reference距离/time嵌入实际内联在构造器和forward中，不存在独立`_mlp`或`_time_embedding` helper。xy/reference/fx/placeholder/time保持；不能把旧structured wrapper直接改point模块，它校验ConvFFN合同。每个模型独立实例化 |
| learned-query Down | `cdlno.modules._DownAttention.__init__/forward` | **需独立包装/新类**。Q为正交参数[M,h,dh]，无额外Wq/无query residual；当前仅SDPA输出，没有mask/returnA。MSAR floor要A[B,h,M,Nsrc]，off不物化A；不得为MSAR破坏旧调用/keys |
| 两FFN+SA | `PlainFFN`、`_SelfAttention`、`LRSAFrontBlock` latent子层 | **原语可用，完整block不可用**。MSAR独立latent-only FFN1→SA→FFN2，不能带Down/Up/点残差，也不能用rear GEGLU或no_sa开关 |
| norm | `RMSNorm`、`make_norm`，per-headQK RMS | latent/attention可用；RMSNorm有可训练scale。**PairwiseAttnRes只允许w、总w参数3d时，评分必须使用无额外可训练scale的RMS规则**，不可盲目实例化旧affine RMSNorm增加3d参数 |
| UpCross | `_UpAttention`继承`_ProjectedAttention.forward(q_input,kv_input)` | branch-only、无hidden query residual/FFN，可包装规范化；receiver heads独立配置；Q必须来自对应E，不能使用无关learned query。旧outer block负责额外点残差，不能一起复用 |
| point MLP/head | wrapper的`preprocess` Sequential、`_init_linear`；K wrapper `output_norm/output`，旧readout的LN/Linear | 逐点lift/head构造和初始化规则可用；**不能复用整个`LRSAFeatureReadout`**，其HF残差+point模块不属于MSAR final |
| 模型输出 | 现有wrapper预测Tensor | MSAR默认也Tensor；训练显式return_aux/专用adapter可新增；不能使用`last_loss`跨forward状态或改变旧返回类型 |
| Pairwise fusion | 旧 `CDPA._depth_fusion`/kernel Reader仅供比较 | **不能直接用**：旧来源集合、norm参数、无固定2或有gamma不同。新三处两source轴[B,M,2]、raw E/U、w0、固定2；无CDPA/history |
| coverage | 无现有attention coverage原语 | **新增**，仅四Down；heads均值→M均值p；source项求和，batch/层均值；mu有效点均匀，FP32，off/weight0不请求A；不是改原PDE loss |
| metadata/CLI | K `options.explicit_arguments`；新M1 `options/metadata/registry` | 显式识别可用；MSAR解析/结构+目标分离已有；后续补真实wrapper/task合同和Run，不能用K的config类型冒充MSAR |
| checkpoint/输出 | 三类Run、`experiment.Experiment`、V1Archive | 复用顺序/记录/排他目录协议；需MSAR专属family、任务验证和稳定类路径。Experiment当前family/run_key及绘图名称分支显式支持旧家族，不能不接线就声称可用；V1Archive不作为自动新增resume依据 |
| 性能 | `tools/cdlno_perf/{models,costs,measure}.py` | 复用同步、warmup、median/p90、完整步/显存；M9新增多尺度成本分类及floor显式A成本，不改变旧类别或以参数比例替代计时 |

### no_sa 与参考 LRSA 的边界

现有 `LRSAFrontBlock.forward` full 是 `S=Down(H)`、`A=S+FFN1(N1(S))`、`B=A+SA(Nsa(A))`、`T=B+FFN2(N2(B))`，然后Up/点更新。no_sa **仍有两个独立FFN**，identity `T=S`原对象。旧T都在Up专属norm之前，有梯度；KCDNO reader插在两FFN之间，writer使用FFN2后的raw T。本轮不移动这些边界。

实际参考 `/home/hwz/LRSA-Operator` commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`，`src/perceiverforpde/modeling/layers/attn.py`的约605—615、825—830行：`disable_interleaved_blocks`分支内同时包含SA和`channel_mixing_2`；打开它同时删除FFN2，**不等价于本仓库no_sa，更不等价于MSAR完整latent block**。参考测试在本轮完整套件中实际运行；没有复制参考训练框架或安装可选attention依赖。

### 初始化、mask、batch、dtype与无卷积边界

公共Linears显式trunc_normal std0.02、bias0；norm scale1/bias0；learned Down query正交（M>d时是列正交，不宣称M行互相正交）。Q/K/V无bias，O有bias；per-head QK RMS+标准head_dim^-1/2；构造独立ModuleList，不重复参数对象，不由wrapper末尾递归apply重置。ConvFFN是dense groups1 3×3+内部LN+fc1无bias/fc2有bias，保留原生Conv初始化；它在旧structured block **Up之后**，不是latent卷积。原Transolver结构分支另在slice前投影使用Conv2d/3d。

MSAR新core/lift/head不实例化这些卷积，不修改旧conv类。新latent块直接组合两PlainFFN/SA/norm，新的Down/Up围绕其实现；不能用“旧block设no_sa”或“旧readout去head”近似MSAR图。

旧公共Cross要求query/context batch相同，拒绝广播；point wrapper可变N，structured wrapper显式H/W且拒绝N!=H*W。没有latent padding/valid-mask接口，不能把surf（loss/显示mask）当作无效点mask。Car新家族只接受单图，全0batch、ptr[0,N]；Air允许采样后ptr[0,原N]且原N>=当前N，但仍必须单图。旧工业Transolver会把点合成一个场，不能据其能接PyG Batch推断多图安全。MSAR沿用已批准单图合同，不扩展图batch方式。

旧RMSNorm对half/bfloat16用FP32归约并返回原dtype，double保留；Down Q对齐投影K的dtype。CDPA深度融合、K kernel phi/summary/除法/来源融合真正关闭autocast用FP32；独立reference保留double。MSAR未来coverage要明确FP32，不能复用kernel强制float函数作double oracle。生产任务没有因本轮改变AMP/backend；没有以float32归约就宣称任意极端有限输入一定不溢出。

## E. 修改前依据、实际验证与环境

完整索引：[fixture-index.json](msar_lno_audit/m0/fixture-index.json)。所有比较加载**同一份保存权重和固定输入**，不以重新随机初始化互比。外部二进制不自动提交。

| 证据 | 来源/时间 | 本轮结果与边界 |
|---|---|---|
| 41个旧Transolver/CDLNO夹具 | 真正旧K0；原Transolver八任务8、CDLNO八任务×三front24、core三front×三CDPA9 | **41/41精确回放**；原输出/诊断梯度、strict权重与真实任务Run检查；CPU CDLNO、GPU原Transolver；不是原任务完整训练 |
| 6个KCDNO/matched core夹具 | **真正pre-M1**：all/off/matched×point/5×7conv | **6/6精确回放**，CPU FP32/math，权重键/shape/config/input/output保留 |
| 24个KCDNO/matched任务wrapper夹具 | 本轮post-M1、pre-MSAR数学；八任务×all/off/matched | capture与独立新进程replay **24/24**；PDE真实factory，工业真实PyG、稳定pickle；state_dict/Car whole/Air whole+list；无原loss训练声明 |
| 其他真实旧注册模型 | 本轮post-M1：Air MLP/PointNet/GraphSAGE、PDE structured3D | **4/4同权重精确回放**；真实构造/输出/strict state，Air原列表pickle；小型CPU输入。GUNet图链因扩展缺失未运行，代码/config/factory冻结 |
| 当前完整既有测试 | 未修改任何现有测试逻辑 | `Ran 316 tests in 576.961s`，**OK，0 failure/error，2 skip记录**；含动态生成的任务测试，故不同于AST计数274个命名test方法 |
| shell静态 | 全部70个shell | `bash -n`通过；未启动训练 |
| 源文件冻结 | M0开始653文件 | 除STATUS/记忆的3份增量文档外，其余650份字节不变；另与真正pre-M1快照比较的220个生产/入口/工具/测试/启动源码全部不变；见[freeze](msar_lno_audit/m0/freeze.json) |

回放固定 `eval/dropout=0`，同设备同后端 `atol=rtol=0`，输出差0。41个K0还核对诊断梯度；新24+4只作模型接口/权重/输出参考，不声称新训练步梯度夹具。24个wrapper中静态缩小5×7/B2，Elasticity N11/B2，NS **4096**/B1，Plasticity **3131**/B1且T独立，工业N13/单图；d8/h2/M4/L2/r3只是夹具尺寸，正式预设未改。其他模型用原Air YAML配置与真实合成边，未伪造radius_graph；3D是3×5×2/B2，不属于八任务训练支持声明。

完整套件实际复用原loss/normalizer/mask、NS10步/Plasticity20次更新、core/history/AMP/工业pickle、旧AST冻结、记录/周期图/show和M1配置测试；两次skip是Air完整抽样epoch和周期图radius_graph，不能由真实Data对象测试升级成这些链路通过。详见 [regression.log](msar_lno_audit/m0/regression.log)。未改旧测试使其迎合本阶段。

环境：[environment.json](msar_lno_audit/m0/environment.json)。本机Python3.13.9、torch2.13.0+cu130、CUDA13.0、RTX5090 Laptop、PyG2.3.1、NumPy2.2.6、SciPy1.16.3；pyvista0.48.4/vtk9.6.2可导入。`torch_cluster`/`pyg_lib`/`torch_scatter`缺失；真实三点radius_graph探针失败，未伪造替代模块。真实PyG Data/Batch、已有小GPU/AMP回归能执行。**目标远端Python3.10/torch2.11/CUDA12.8尚未执行**，本机超出pyproject的Python范围，不安装、不改范围，不把本机通过写成远端验收。

实际命令（均无真实数据；日志/JSON在audit目录）：

```bash
python -B docs/msar_lno_audit/m0/inventory.py
python -B docs/msar_lno_audit/m0/review_index.py
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result docs/msar_lno_audit/m0/k0-replay.json
python -B docs/msar_lno_audit/m1/replay_current.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/current-fixtures \
  --result docs/msar_lno_audit/m0/current-core-replay.json
python -B docs/msar_lno_audit/m0/wrapper_fixtures.py capture \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers \
  --result docs/msar_lno_audit/m0/wrappers-capture.json
python -B docs/msar_lno_audit/m0/wrapper_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers \
  --result docs/msar_lno_audit/m0/wrappers-replay.json
python -B docs/msar_lno_audit/m0/other_model_fixtures.py capture \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/other-models \
  --result docs/msar_lno_audit/m0/other-capture.json
python -B docs/msar_lno_audit/m0/other_model_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/other-models \
  --result docs/msar_lno_audit/m0/other-replay.json
python -B docs/msar_lno_audit/m0/finalize_evidence.py
```

最后的只读核查实际验证182个夹具文件hash（含原K0的136个），75个模型记录，无文件损坏或生产改动；新审计脚本按Python3.10语法解析，`git diff --check`及新文档链接检查通过。[结果摘要](msar_lno_audit/m0/summary.json)集中列出通过与未执行。审计辅助脚本`review_index.py`首次处理二进制记录时出现`KeyError: text_read`，已仅在该新增脚本改为读取可选字段，随后全清单执行成功；不是模型或原测试故障。

capture拒绝已有夹具目录；`review_index.py`也以exclusive方式保留已写证据。后续回归只使用replay，不能重新执行inventory/capture来覆盖本次冻结起点或构造所谓“旧基准”；新阶段应建立独立目录。远端须迁移夹具并替换`--artifacts`/结果路径。无历史训练checkpoint时这些即时模型是有效同权重回归参考，但**不能证明所有历史pickle或真实训练权重兼容**。旧pre-A1真实历史对象证据继续保留，按受影响范围使用。

远端已有环境可执行 `python -B tools/cdlno_environment_preflight.py`及上述unittest；LRSA参考路径需设为实际checkout，缺失会明确skip。没有真实数据读取、训练、收敛、准确率、真实epoch性能验收，也没有MSAR合成模型/GPU/参数量证据，因为数学模型尚不存在。

## F. 与先前 M1 的适配核对及后续最小范围

M1源码及`tests/test_msar_config.py`在本轮保持字节不变；13个配置测试包含在实际316项套件内，全部通过。

| M1条款 | 当前实现与M0核对结果 |
|---|---|
| 独立family/导入 | `msar_lno → cdlno.msar_lno`仅配置协议；根包发现可用；旧factory未接新构造器，没有占位模型破坏旧import |
| Light/Full | M分别[512,256,128,64]/[1024,512,256,128]、d96/192，heads[4,4,8,8]，双depths[3,1,1,1]，ratio2/GELU；不同于K任务d/h/M，无隐式继承 |
| 校验/扩张 | 四元素列表、正整数/整除、固定depth、M递减；N不裁剪M；`input_layout`可记录Full1024>N972。生产日志尚未接入，不能夸称已运行 |
| 目标/结构分离 | floor/.01/.2，off/0、weight0有效off；coverage字段仅training，eval目标不同不拒绝相同结构；**off不物化A的数学路径仍待M2/M3** |
| 解析 | 显式CLI>profile>family defaults，复用K explicit-only解析；新parser副本不修改旧defaults；d/n_hidden别名冲突拒绝；旧L/F/P/M/CDPA/kernel参数显式不适用 |
| metadata | outer family明确，旧缺family回旧规则；完整resolved结构严格比较、先读再比较、不写回；三实际保存格式只作为元数据约束，尚无MSAR对象重建 |
| 目录 | output/task/msar_lno/profile/coverage_floor或off/unique或save_name；exclusive，旧目录不变；需要后续真实Run/Experiment接线 |
| 无卷积/final读出 | `point_module=pointwise_mlp`只描述逐点lift/head，**不授权额外N点point残差块或final E0 skip**；新core必须按总控执行 |
| 融合norm参数 | M1 `norm=rmsnorm`不强制PairwiseFusion有scale；后续按M2“每尺度只有w”及M9“w仅3d”使用无新增scale的评分RMS。无须改M1来引入额外参数 |

这些是配置与集成位置的适配结论，不能将JSON往返冒充完整模型checkpoint通过。没有需要本轮修改M1生产代码的已发现阻断问题。

| 阶段 | 基于当前实际代码的最小变更范围（未来拟新增路径明确标“新”） | 主要验收 |
|---|---|---|
| M1 | 已有`cdlno/msar_lno/{config,profiles,registry,options,metadata}.py`，本轮仅核查 | 保持profile/优先级/metadata；不重做旧计划 |
| M2 | **新**`cdlno/msar_lno/modules.py`及独立reference/新测试 | Down可选mask/A，off无A；latent双FFN+SA；纯Up；w-only两源固定2；floor source求和；不改旧modules |
| M3 | **新**`cdlno/msar_lno/core.py`/core测试 | 四Down/四Up/12latent块/24FFN/3fusion；D4无skip、final无E0skip；默认Tensor，显式aux；无Conv/history |
| M4 | 新family Run/loss adapter；按项目分布最薄选择；必要时扩展`experiment.py`的新family记录分支 | 只新family取aux，保留PDE loss；严格sidecar/task/对象校验；暂不声称八任务完成；不新增task resume |
| M5 | **新**standard wrapper+`model/MSAR_LNO.py`；`model_dict.py`、`cdlno_entry.py`四静态薄分支；四exp构造/loss/保存调用；新配置/薄脚本 | 原lift/normalizer/loss/pointorder，MSAR无Conv；每任务原loss合成步，旧同权重回归 |
| M6 | 扩展同一新wrapper及`exp_ns.py/exp_plas.py`有限选择/aux连接 | NS原10步loss/teacher forcing，coverage按实际forward与原聚合尺度处理；Plasticity20次update/一次scheduler；无时间cache |
| M7 | **新**Car/Air稳定wrapper、模型选择/Run/params.yaml新key；原train函数仅新family aux适配分支 | 单图/变N/原字段/weighted mask loss；原graph工作保留；Carwhole/Airlist+member；3cwd加载 |
| M8 | 新MSAR任务/综合测试和覆盖表；复用原AST冻结与本索引 | Light off/floor、Full真实shape+小N，checkpoint与原loss/GPU/PyG分别标注；所有受影响旧夹具同权重 |
| M9 | 原`tools/cdlno_perf`最小扩展、最终README/命令/报告 | 完整NM1/各尺度SA/双FFN/Up/head成本，coverage训练A单列；同条件有限GPU，不宣称真实精度/epoch收益 |

以上是后续被点名时的实施边界，不是本轮授权。M2的“coverage对有效source归一”依总控明确公式解释为**source求和、batch/层均值**，不能再除N。源有效mask在softmax前；四层只有第一层使用原输入有效mask，其余latent全部有效，不能把原N-mask直接套到更小M。两源fusion使用无参数RMS规则的依据如上；不把当前旧affineNorm当作无法解决的架构冲突。

## G. 原有问题、未验证项与自审

| 已有事实/限制 | 影响与本轮处理 |
|---|---|
| 部分旧PDE parser默认 `Transolver_1D/2D`不在真实factory | 官方脚本显式给有效键；记录，不借M0修旧默认 |
| Car外层日志接收velocity/pressure返回值顺序有互换；内部backward正确 | 新aux将来必须连接实际loss，而非从错误日志反推；不改原loss/日志。show已按正确命名分项重建展示目标 |
| Car `cal_coefficient`硬编码 `/data/PDE_data/mlcfd_data/training_data/param0` | full drag评价受fold0/raw路径约束；不能从合成Data通过推断物理评价通过 |
| Air `if criterion == 'MSE' or 'MSE_weighted'`恒真、验证加权条件拼写 `MSE_weigthed` | 旧MAE/验证加权语义缺陷；当前实际训练入口是MSE_weighted。冻结原行为，MSAR后续需明确LPDE连接的是原backward，不在本轮修复 |
| 早期A3文档把Air函数默认MSE当入口协议 | 后续weighted测试/报告已纠正；本审计使用真实main调用，不能升级早期证据 |
| 早期报告路径/阶段支持有时滞 | 当前matched已接八任务、周期图已接、任务resume未接；以当前源码/新状态为准，保留历史原文 |
| 无积分权重数据合同 | Air cell Area用于原采样、Car面积用于指标，均未作为模型输入测度；MSAR coverage使用有效点均匀mu，不凭空估计面积 |
| 缺torch_cluster/远端环境/真实数据 | GUNet图链及Air实际radius采样未执行；无真实完整读取/收敛/精度/真实epoch依据；不能把这些边界写成模型实现失败 |

交付前已自审五点：①M0补做的时间顺序及pre/post-M1夹具没有混写；②Down权重/mask、Up残差、两FFN和norm参数通过实际forward确认；③八任务loss/时间/单图/保存差异未按名称统一；④M1仅配置的支持范围和后续文件边界清楚；⑤旧模型同权重回放、源码freeze和316项测试均有真实结果。未发现需要用户先改变MSAR公式的阻断冲突。上述既有缺陷和未验证环境仍保留，不作“整个仓库无缺陷”的保证。

本轮diff仅新增本审计、审计脚本/文本证据/外部独立小夹具，以及MSAR STATUS、旧总STATUS和memory增量指针；没有改模型、factory、配置实现、入口、数据、依赖或现有测试逻辑，没有commit/push/PR/真实训练。

本M阶段结束，未执行下一阶段。
