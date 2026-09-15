# KCDNO 最终实施与综合审查报告

2026-09-16。用户已明确将原逐阶段等待改为连续执行K4–K10，每阶段完成后自行审查再继续。本轮从已接受的K1配置、K2独立数学模块和K3核心开始，完成K4–K10。没有启动真实数据训练、下载数据、安装依赖、commit/push、PR、reset或修改AGENTS。

## A. 架构结果

已按确认的v1规格实现独立`family=kcdno`，八任务可选择主版`history_mode=all`或对照`off`。新增可训练`family=lrsa_matched`，保持完整LRSA processor作为结构对照。旧Transolver、CDLNO full/no_sa/identity、CDPA off/entry/every_block和chunk规则保留。

```text
任务原输入/位置/时间合同 → lift → X0[B,N,d]
for l=1..L:
  H    = RMS_point(X)
  S    = Down(H)                       # learned projected Q，softmax沿N
  U    = S + FFN1(RMS1(S))
  Uhat = Reader(U, tuple(past))         # 第一层/历史off直接U
  T    = Uhat + FFN2(RMS2(Uhat))         # raw history，Up norm之前
  V    = X + Up(H, RMS_up(T))           # Up latent norm一次，softmax沿M
  X    = V + PointModule(RMS_out(V))
  if l<L and history=all: past.append(Writer(T))
任务 LN/head(X) → 原任务输出
```

Writer源层拥有RMS_k/Wk；`K=clamp(ELU(RMS_k(T)Wk)+1,1e-6)`，`memory=KᵀT[B,r,d]`、`mass=sum_token K[B,r]`。每个源摘要在一次forward只写一次，不detach。Reader接收层一次产生Q，对每份历史独立计算`R=(Q memory)/(Q mass+1e-6)`。当前候选为RAW U，按`w·RMS_depth(R)`沿来源softmax，每样本/当前token一套权重；RAW候选加权为C，再`Uhat=U+gamma(C−U)`。w0、gamma为无约束标量初值.1；新Q/K biasFalse/Xavier gain1。关键核计算关闭autocast并用FP32，返回U.dtype，独立oracle保留double。

默认L8没有latent SA/Bridge/persistent processor/额外final Up：Down8/Up8/两latent FFN16/PointModule8，all7写/7Q/28逻辑读取、7次Reader API；首层无Reader、末层无Writer，L1/off无历史参数。Down/Up仍是标准多头SDPA，不能称整个模型无attention。point任务保留GELU FFN，规则任务保留dense3×3ConvFFN、内部LN/外部RMS和原网格点序。

逐条代码、公式轴与测试映射见 [需求矩阵](KCDNO_REQUIREMENTS_MATRIX.md)。矩阵覆盖规格§§1–13与用户总控，数学附件没有被解释成添加结构或物理损失的授权。

## B. 真实文件、接入和diff

| 文件/符号 | 本轮变更及作用 |
|---|---|
| `cdlno/kcdno/standard.py:StandardModel` | 六标准任务共用lift/坐标/时间/输出wrapper，core只接已提升特征 |
| `cdlno/kcdno/airfrans.py:AirfRANSModel` | 稳定shared pickle类，复用旧reference/single-graph/字段校验，独立core/head |
| `Car-Design-ShapeNetCar/models/KCDNO.py:Model` | 原tuple/x7/单图合同的新家族wrapper，稳定`models.KCDNO.Model` |
| `entry.py/industrial_entry.py/air_entry.py` | 各实际parser风格、profile优先级、独立目录、完整core+task sidecar和原保存协议 |
| `families.py/matched_config.py/matched.py` | 独立matched-full配置/元数据与L个原LRSAFrontBlock组合，共用八任务wrapper |
| `loading.py:validate_whole_model` | 可信对象加载内检查类/模块、影响forward的非权重属性、固定reference buffer，随后strict state keys |
| 六`exp_*.py`、model_dict、两工业main/evaluation | 只增加新模型构造、记录/加载有限分支；旧数据/loss/时间/指标正文保持 |
| 标准/Car新JSON、Air `params.yaml` | 前两只新增新任务配置，Air仅追加kcdno和lrsa_matched key，旧key字节内容保留 |
| `tran_evaluate/kcdno/` | 八任务train/eval脚本、公共路径转发、Car只读评价guard及顺序`train_eval.sh` |
| `cdlno/experiment.py` | 现有记录器仅新增可选family/run_dir选择；旧默认仍CDLNO，loss/optimizer不变 |
| `tools/cdlno_benchmark.py`与`cdlno_perf/` | 复用计时，增加生产模型选择、真实kernel完整MAC/来源/内存统计和独立闭式核对 |
| 新`test_kcdno_*`、已有冻结投影少量更新 | 数学/接口/strict加载/模式对照与原AST投影；没有为了import改变训练器 |

本轮开始commit：`222647fef343e9fe929e65412f5c7449ced5517b`（main），已有未提交K1–K3与用户提示词文件保留。K0固定模型夹具来自`9f72946e0adbfe27ba0b75b1646fa697ae16d3df`；原Transolver基线`75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。本报告不创建新commit。

本轮K4前快照：`/home/hwz/CDLNO-artifacts/k4-before-lxbxc_wh/source`；各阶段还有独立before.json和patch。最终 [累计K4–K10 diff](kcdno_audit/k10/integration-changes.patch) 包含新文件，避免只看git diff遗漏未跟踪源码。全部当前修改见 [git status](kcdno_audit/k10/git-status.txt)，已有K1–K3不会误报为本轮新写。

阶段报告：[K4](KCDNO_K4_STATIC.md)、[K5](KCDNO_K5_TEMPORAL.md)、[K6](KCDNO_K6_CAR.md)、[K7](KCDNO_K7_AIRFRANS.md)、[K8](KCDNO_K8_MATCHED.md)、[K9](KCDNO_K9_PERFORMANCE.md)。旧CDLNO/A阶段文档与夹具保留。

## C. 八任务×三个计算图覆盖

表中“三图”=kcdno all / kcdno off / lrsa_matched full。**S**为静态/真实parser；**M**为真实wrapper合成forward、backward、optimizer step；**E**为eval输出；**C**为同架构严格保存/加载和错误架构拒绝。M均为合成数据，不是实际数据集。原loss列区分直接原函数/原AST与诊断loss。

| 任务 | 三图S/M/E/C | 原loss及条件连接的实际证据 | PyG | checkpoint | 新模型GPU范围 |
|---|---|---|---|---|---|
| Darcy | 全部通过 | all/off原TestLoss；原normalizer/decode与central_diff梯度项AST直接连接 | 不适用 | strict model.pt | L8/d128/M64/r16/N7225/B1三图FP32计时/step |
| Elasticity | 全部通过 | all/off原TestLoss/normalizer；972点PointFFN | 不适用 | strict model.pt | L8/d128/M64/r16/N972/B1三图FP32计时/step |
| Airfoil | 全部通过 | all/off原loss/坐标；真实N11271减宽与5×7 | 不适用 | strict model.pt | 任务专属GPU未执行；共享core FP32/AMP已测 |
| Pipe | 全部通过 | all/off原loss/decode；真实N16641减宽，主M32 | 不适用 | strict model.pt | 任务专属GPU未执行 |
| NS | 全部通过 | all直接执行原10步真值回填/一次backward+step/scheduler；eval预测回填；off/matched另有诊断step | 不适用 | strict model.pt | 新时间任务GPU未执行 |
| Plasticity | 全部通过 | all直接执行原20时间前向/backward/step，batch末scheduler1次；T梯度/输出影响；off/matched诊断step | 不适用 | strict model.pt | 新时间任务GPU未执行 |
| Car | 全部通过 | all/off直接原train.train/test，velocity全点与surface pressure加权loss；matched诊断step复用相同冻结入口 | 真实Data/Batch通过 | 整模型 model_epoch.pth | 新工业任务GPU未执行 |
| AirfRANS | 全部通过 | all/off直接原train.train/test MSE_weighted surface/volume；matched诊断step复用同入口 | 真实Data/Batch通过 | 列表+member整对象 | 新工业任务GPU未执行 |

八任务32组family/profile默认解析、24组三图真实脚本train/eval预览及实际parser、启动记录/参数量/sidecar只读通过。六标准任务有唯一共享wrapper数学，无八份新训练器。Industrial支持原single-graph/N可变/全0batch，明确拒绝多图；Air原sampled ptr规则保留。没有执行完整radius_graph/scatter物理评价；该代码通过静态冻结核查，不能把字段接口检查等同完整评价。

原时间循环测试明确保留NS10→10，训练每步真值反馈、10步loss累加后一次更新；Plasticity N3131、标签[B,N,4,20]，20次独立时间条件更新，随后原一次scheduler，无预测反馈。不同真实时间每次重新执行L层core/history。

checkpoint原格式未统一：标准state_dict使用weights_only=True/strict=True；工业在已批准的局部可信范围用weights_only=False加载原整对象/列表，保留稳定类导入路径。所有新路径先读完整已有配置再比较显式覆盖，缺family不猜成新模型；旧CDLNO缺front字段的历史兼容仍走旧逻辑。同family的L/d/h/M/r/history/activation/adapter冲突拒绝，不实现跨模式自动权重迁移。

## D. 实际命令、环境与结果

实际完整命令见 [commands.txt](kcdno_audit/k10/commands.txt)。本轮最终套件：

```bash
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator PYTHONPATH=tests:. python -B -m unittest test_kcdno_config test_kernel_history test_kcdno_core test_kcdno_tasks test_kcdno_temporal test_kcdno_car test_kcdno_airfrans test_kcdno_matched test_kcdno_performance test_kcdno_delivery test_modules test_cdpa test_core_config test_core test_front_ablation test_front_task_modes test_front_training test_static_standard test_temporal_standard test_shapenet_car test_airfrans test_performance test_experiment_records test_lrsa_reference -v
PYTHONPATH=. python -B docs/kcdno_audit/make_regression_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/kcdno_audit/k10/old-replay.json
python -B docs/kcdno_audit/final_static.py
```

- 最终套件**257项运行，OK(skipped=1)，263.562s**。唯一skip是原AirfRANS完整抽样epoch观察器测试需要torch_cluster，本地缺失；未安装或伪造模块。日志末LRSA可选xformers/liger ImportError是参考源码自身打印并走原fallback，实际point/conv参考前向与梯度误差均0；不是新模型依赖失败。
- 最终覆盖表复核发现NS/Plasticity off之前只有forward/checkpoint，随后仅补一个针对性合成optimizer step测试，两任务通过，不重跑无关全矩阵。此项1测试通过0.525s，命令/日志另列`temporal-off-step.txt`；原257项日志保留，当前同一套测试枚举会多此1项。
- K0固定输入/同一权重**41/41回放精确一致**：33旧CDLNO CPU与8原Transolver GPU，含九个旧core front×CDPA组合、24个八任务三front模式及原工业pickle/list。输出/输入梯度/参数梯度atol=rtol=0，None梯度参与也比较。不是重新随机初始化作比较。
- [冻结证据](kcdno_audit/k10/freeze.json)：119个旧生产/数据/模型/脚本/依赖文件字节相同；15个入口、参数辅助文件、registry或YAML移除明确新增分支后完整AST/内容一致。旧modules/core/CDPA/checkpoint数学不变。
- 本机GPU：K2 FP32和FP16/BF16 AMP核子图实际autocast-off/梯度，K3 point/conv FP32及AMP前反向，本次最终套件已重跑；K9两个有限完整任务形状×4模型FP32实际性能。旧A阶段GPU任务回归属于旧模型证据，未算作新八任务GPU矩阵。
- 三个原工作目录的新进程共享配置/权重加载通过。标准state_dict、Car整对象、Air整对象+列表，matched工业加载也在两个原cwd新进程完成。D另在`python -I`进程验证旧parser不需要新shared包；没有import顶层读数据的exp/main。

环境：Python3.13.9、Torch2.13.0+cu130/CUDA13.0、PyG2.3.1、RTX5090 Laptop/driver591.86、cuDNN92000。用户目标Python3.10/Torch2.11/cu128及已可训练Car环境保留，尚无远端新模型验收。本机通过PYTHONPATH加载源码，未强行在pyproject指定Python>=3.10,<3.12范围外editable安装；无依赖安装/替换。

远端可先运行不依赖本机历史夹具的无数据核心套件（命令供用户执行，本轮未在远端运行）：

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_config test_kernel_history test_kcdno_core test_kcdno_performance -v
```

完整冻结/旧回放及部分任务测试需要保留本机K0/K4–K7与pre-A1外部源码/数值夹具；这些文件没有冒充仓库内正式数据，也没有自动提交大二进制。远端没有对应夹具时，应同步审计产物并配置实际路径，或只运行上面独立核心套件及K9工具，不能重建随机权重后声称旧基准回放。

K9有限FP32/TF32off/compileoff/autoSDPA对照，warmup3/10次同步计时、报告median/p90和peak allocated、AdamW state已初始化。点任务all较matched更慢；Darcy也未形成明显提速。完整MAC只下降约0.196%（Darcy核心主项），大量N点投影/卷积保留；实际全表、方差与热点见K9报告。不能用MAC/参数比例推断真实epoch或其他GPU速度。

## E. 最终自审与必要缺陷修复

已独立检查以下重点并完成相应检查，没有未裁定的结构冲突：

1. 两FFN间Reader、raw T/一次Up norm、7写7Q28读与首末/L1/off参数：源码、独立double oracle、手工core reference和live计数互相一致。
2. 配置/选择/旧默认：显式CLI>profile>家族默认；两family strict隔离；full对照固定full，KCDNO没有接入旧front/CDPA。最后修复旧入口无条件导入新增共享工具的依赖回归：新parser选项留在轻量本地helper，新包仅在选中family后导入。三个原cwd隔离旧parser检查通过。
3. 工业pickle的非权重结构：strict state_dict无法发现被修改的attention heads、norm eps或nonpersistent reference。新增局部行为/buffer校验，篡改heads/eps/reference/激活均明确拒绝；已学习gamma正常加载且不重置。
4. 时间及数据边界：Plasticity原time embedding需要d>=2，新增明确构造错误，避免合法core小d1落入任务维度错误；保持公式和正式d128。Air实际parse_args(argv=None)在K8修复为sys.argv解析，增加真实调用方式测试。所有数据/loss/时间/原指标区段通过冻结投影。
5. 完整成本/归因/命令：补计K2 F.linear绕过Linear hook的Q/K成本，分子/分母/score/stack/保存张量均不漏报；脚本只提供profile默认、不固定覆盖尺寸。GPU慢于预期时只报告热点，没有减少来源/共享参数/detach/改变公式。

初次测试命令类名错误、测试输出投影路径错误与近似数值舍入错误均保留失败日志并修正测试；不标成生产结构缺陷。K9初轮point计时与CPU测试重叠，明确排除并对相同有限配置单独复测；正式表只使用复测JSON与随后Darcy单独运行结果。

旧缺项继续独立记录：Car原拖曳评价固定raw param0路径/fold0、旧外层压力/速度日志标签交换、Air旧MAE条件分支及原baseline宽松加载等，不趁本轮重构旧分支。V1归档/可视化基础已存在，但任务V2–V5完整resume尚未接入；本次保留模型checkpoint协议，**没有优化器/scheduler/RNG状态的精确断点续训承诺**。

## F. 交付、命令和未验证项

[八任务三图训练/评价命令](KCDNO_COMMANDS.md)均使用实际`tran_evaluate/kcdno/*.sh`。`train_eval.sh TASK all|off|lrsa_matched`只在用户调用后顺序执行，训练失败不会开始评价；加`--dry-run`只预览。每次训练使用新run目录，已有路径拒绝覆盖；eval指定原run。工业路径语义与原物理指标限制在命令文档明确说明。

本轮交付的是源码可执行性、合成训练/数学与接口/加载正确性，以及有限本机性能证据。**仍未验证**：真实文件完整读取和split内容、真实工业抽样/图/scatter全链物理评价、实际数据集训练、收敛/泛化/精度、长轨迹误差、达到同误差的时间、完整任务epoch效率、远端2.11cu128和新八任务全GPU矩阵。没有从随机张量推断真实准确率，也不把两配置性能外推为整体架构优势。

本轮K4–K10均完成并自审，交付后停止。没有启动真实训练或额外模型实验。
