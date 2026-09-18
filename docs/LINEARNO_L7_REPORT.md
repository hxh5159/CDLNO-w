# LinearNO L7 — ShapeNet-Car 模型与评价组件

## 最终协议复核（2026-09-18）

用户已确认 Car 默认沿用官方标准化目标：全点三速度 normalized MSE + `0.5×` 表面压力 normalized MSE。生产 profile、metadata-first strict load、native train→checkpoint→resume→新进程 eval 和旧模型回归已在后续证据中完成，L7 最终状态为 PASS。下方早期 BLOCKED 段落保留为历史记录，真实数据、完整训练、真实 Cd/Spearman 和远端环境仍为 NOT RUN；最终结论见 [LINEARNO_IMPLEMENTATION_REPORT](LINEARNO_IMPLEMENTATION_REPORT.md)。

## 2026-09-18 — L7 Car 官方训练接线完成，论文目标仍待决定

**阶段状态：BLOCKED。** 连续执行 L7–L10 的授权有效。本轮完成所有不依赖 Car 论文训练解释的接线；没有进入 L8。唯一尚待决定的是 Car paper objective 的 `space` 与速度三通道 `reduction`。AirfRANS 的 MSE 决定不自动扩展到 Car。默认 paper profile 当前在创建运行目录/读数据前明确报未决错误；`official_release` 可执行。以下历史段落保留，但本段取代其中“Car 完全未接线”和“Air 仍未接线”的现状描述。

实际新增 `cdlno/linearno/car_entry.py`、`tran_evaluate/linearno/car_train.sh` / `car_eval.sh`、Car 原生合成闭环 worker/测试；Car `main.py` / `main_evaluation.py` / `models/cdlno_run.py` / `train.main` 仅增加 LinearNO 显式分支。`profiles.py` 的 `car_l7` 合同只明确实际 surface-velocity/sample-path 评价输入，不偷偷决定 paper 训练。模型数学、dataset、原 loss/optimizer/时间协议、旧 launcher 和其他 family 未改。

- `--cfd_model LinearNO` → `models.cdlno_run.parse_args` → `car_entry.parse_args/run_cli` → `ShapeNetLinearNO`。稳定整对象路径仍是 `cdlno.linearno.shapenet.ShapeNetLinearNO`，正式参数 **3,852,420**。
- 新 family 经实际 `train.main(..., linearno_run=run)` 调用独立训练 adapter；官方目标严格为全点三速度 normalized-MSE + 0.5×surface-pressure normalized-MSE。保留 Adam、每步 OneCycle、每 epoch 重建 shuffle/drop_last DataLoader、原 validation cadence。原模型路径的整段 AST 与本轮前一致。
- OneCycle `total_steps=(ntrain//batch+1)*epochs` / `final_div_factor=1000` 保留，metadata同时记录实际 updates/epoch。合成3图×3epoch是9次更新、total_steps12，未偷偷修成9。
- 新运行保存 hxh `config.json` / `train.log` / epoch记录；每 epoch 有严格 metadata/state pair、latest/final指针与 `model.pt`，最终仍保存本地 `model_<epochs>.pth`。eval/resume先读并验证 root/pair metadata，然后构造；禁止缺metadata时回退 Transolver。eval不改训练sidecar。
- normalizer以命名数值状态保存，eval只加载 heldout、不重拟合；resume使用存档coef加载训练集。原 loader 未改。只读preflight核对原 `os.listdir`顺序、9个fold、全部raw/preprocessed文件与hash，缺文件直接报错，防止原loader静默跳过导致sample路径错位。
- eval执行全点预测、按原node顺序导出prediction/target，分别报告physical rL2和normalized MSE；Cd适配只传surface速度及真实fold/sample路径。force metric与field metric轴分开；`--linearno-field-metric`可显式标记主指标。没有启用 released-bug force 诊断路径。
- launcher保留旧 `cfd_mesh=False` 默认，不绕过原radius-graph构造；合成验证显式传现有 `--cfd_mesh`，真实PyG图保留edges，且N=17/23/8205包含原Python random geometry采样。完整默认radius-graph路径因缺torch_cluster **NOT RUN**。

实际验证：初次综合命令 **102 passed / 2 failed**，两失败是此前L6的源码审计遗漏（旧清单不认识已修改Air测试、Car跨项目审计未剥离Air新分支），不是模型数值失败。修正为精确源码投影后，受影响集合 **34 passed / 0 failed，80.98秒**。包括新增4项Car测试、L1旧Transolver四项、Car/Air原套件各13项。其余已通过的L1–L6/模型数学测试未无意义重跑，合计105个不同方法有通过证据，不能把这个合计写成单次105全通过。详见 [实际命令](linearno_audit/l7/integration/commands.json)、[初次综合日志](linearno_audit/l7/integration/full-regression-initial.txt)、[最终受影响日志](linearno_audit/l7/integration/affected-final.txt)。

Car连续训练与第1epoch中断→新进程resume→独立eval，权重、预测、batch/geometry采样序列、全部RNG/optimizer/scheduler存档hash完全一致；最终本地产生whole-object也在新进程同权重加载。错误profile/M/d/fold/family拒绝；缺预处理文件和eval不重新拟合另有测试。测试替换的仅是原入口的数据获取边界与force执行开关，使用真实PyG和实际 `run_cli/train.main`，不创建同名假数据、不运行main顶层。

冻结：[baseline](linearno_audit/l7/integration/baseline.json)保留1,494文件现状，外部快照 `/home/hwz/CDLNO-artifacts/linearno-l7-integration-hgm5qckz/source`。所有原Transolver模型/Physics-Attention/Embedding相对L0内容hash不变，`LINEARNO/`仍不存在。L0广义manifest本轮仅新增两项预期差异：Car非数学CLI helper、Car旧源码测试；分别有完整AST和精确测试改动核对。L0至L6既有缓存/测试/记录器差异单列，不冒称广义manifest零改动。旧测试产生的8个小sidecar目录保存并迁移到外部快照；pytest两个索引仅在hash精确匹配原基线后恢复，生成版本也保留；没有删除用户产物。见 [freeze](linearno_audit/l7/integration/freeze.json)。

当前可执行官方链（模板，不曾启动真实训练）：

```bash
cd /home/hwz/CDLNO
CAR_RAW=/absolute/path/to/mlcfd_data/training_data
CAR_CACHE=/absolute/path/to/mlcfd_data/preprocessed_data
CAR_RUN="$PWD/output/car/linearno/official_release/shapenet_M32/seed0_example"
bash tran_evaluate/linearno/car_train.sh --linearno-profile official_release \
  --data_dir "$CAR_RAW" --save_dir "$CAR_CACHE" --experiment-dir "$CAR_RUN" --gpu 0 --seed 0
bash tran_evaluate/linearno/car_train.sh --resume \
  --data_dir "$CAR_RAW" --save_dir "$CAR_CACHE" --experiment-dir "$CAR_RUN" --gpu 0
bash tran_evaluate/linearno/car_eval.sh \
  --data_dir "$CAR_RAW" --save_dir "$CAR_CACHE" --experiment-dir "$CAR_RUN" --gpu 0
```

resume只用于中断运行；默认latest，默认eval为final。新训练拒绝覆盖已有run；eval各自创建唯一结果目录。论文训练命令在科学定义确认之前不列为可用。当前没有 `evaluate_checkpoint.py` 入口，实际评估为 `main_evaluation.py`；未虚构额外脚本。

已自审五点：actual M及正式参数、原region/reduction与真实scheduler步数、metadata-first/normalizer不重拟合、force surface点和真实路径、旧模型完整AST与hash。只剩一个科学决定：是否把 Car paper loss 明确为反归一化后的 surrounding velocity三通道联合rL2＋surface pressure rL2、逐样本平均（标明本地解释），或像Air一样默认沿用官方MSE并暂不启用该paper解释。尚未收到答复，不能按等待时间代替决定。

真实数据读取、完整训练/收敛/精度、真实Cd/Spearman、远端Torch2.11/CUDA12.8、L8外部checkpoint、L9综合效率、L10终审均未执行。**本L7阶段在Car论文目标处暂停，未执行L8–L10；无需重新授权阶段。**

**阶段状态：BLOCKED。** 本次用户明确授权继续 L7–L10；阶段许可不缺。已推进不依赖工业训练目标选择的 L7 工作。但 L6 仍未完成生产接线，且 L0-C08 对 AirfRANS 和 Car 的默认 paper 训练定义仍为 `UNRESOLVED`。本轮没有得到该定义的具体选择，不能将授权继续阶段解释为选择某个 loss。L8 要求两工业原生闭环先完成，当前不满足；未执行 L8–L10。

## A. 范围、基线和文件

目标 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`。开始时的1,392个文件、tracked/untracked/ignored分类与hash记录在 [baseline.json](linearno_audit/l7/baseline.json)，源码快照 `/home/hwz/CDLNO-artifacts/linearno-l7-before-q_o1nixy/source` 保留。本轮 `RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`；没有数据下载、真实训练、依赖安装、commit/push/reset/clean。

| 新增文件 | 内容及边界 |
|---|---|
| `cdlno/linearno/shapenet.py` | 独立 `ShapeNetLinearNO`、`ShapeNetBlock`、point MLP及单图检查；组合L2纯attention |
| `Car-Design-ShapeNetCar/models/LinearNO.py` | 本地稳定 `Model` 导出，完整对象类路径为 `cdlno.linearno.shapenet.ShapeNetLinearNO` |
| `cdlno/linearno/car_metrics.py` | 已明确的physical field rL2、normalized MSE、真实样本路径和surface velocity的drag适配器；不决定paper训练loss |
| `tests/linearno/test_shapenet_model.py` | 固定官方完整模型对照、独立double oracle、真实PyG、原训练函数、strict权重及本地whole-object新进程检查 |
| `tests/linearno/test_car_metrics.py` | 独立NumPy公式、mask/channel/零分母校验、真实VTK原语合成数值和路径/长度检查 |
| 本报告、STATUS/REPRODUCTION_MATRIX增量、`docs/linearno_audit/l7/` | 实际命令/结果/冻结和未完成项 |

Car/AirfRANS/Standard现有main、factory/parser、训练、数据、metrics、checkpoint helper、launcher、旧模型与已有测试均未修改。新模型尚未接入Car生产CLI。新评价模块是独立可调用组件，现有eval尚未调用它；不能把组件通过当成任务完成。

## B. 公式与官方实现映射

来源为固定 `HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` 的 `ShapeNetCar/models/LinearAttnNeuralOperator.py` 与对应 `main.py`。加载完整类AST前检查L0记录的官方SHA-256。没有执行官方入口或读取真实数据。

| 条款/官方符号 | 实际符号和行为 | 独立证据 |
|---|---|---|
| tuple `(cfd_data,geom)`；input7/output4 | `ShapeNetLinearNO.forward`；Data.x[N,7]→[N,4]；geom不读取 | 真实Data/Batch、改变geom预测不变、N1/13/29、输入张量不原地修改 |
| 原始构造 `space_dim=3,fun_dim=4` | 构造器保存3+4合同；wrapper实际只读取已有7维x | 官方main已核对，不能因fx=None误删4个输入通道；位置/label/edge均不额外拼入 |
| d256/L8/h8/ratio2，key_ratio1 | `linearno_rank=32`是actual M，M/(d/h)=1才是原key_ratio | 正式实际参数 **3,852,420**；完整key/shape/value和初始化RNG逐位一致 |
| Eq8/9，Q沿M/K沿N、KᵀV后Q乘 | 复用`LinearNOAttention(variant='shapenet')` | 独立exp/sum归一化attention reference和完整函数式网络oracle，不调用被测forward |
| 两个独立温度0.5、clamp[.1,2] | `blocks.i.Attn.tempreature_q/k`保留官方拼写 | 全参数梯度、同权重optimizer一步、原L2边界回归；actual M不是1 |
| Linear input，Linear-GELU-Linear-Dropout output | Car专属attention分支 | 官方逐block及最终输出对照；dropout=.2时恢复同一CPU/CUDA RNG |
| 8个完整pre-LN attention/FFN残差，末层LN/head | `ShapeNetBlock.forward` | 8次hook，所有参数参与梯度；无Conv2d/旧slice SA |
| 默认unified off；官方备用位置replacement | `get_grid`与非持久`pos` buffer | 非方形3×5/ref8，CPU参考仅临时适配官方.cuda，原生CUDA对照；不增加state key |
| 初始化构造→一次apply→placeholder | `__init__/initialize_weights` | 正式配置完整state和最终RNG完全一致，temperature未被重置 |
| 单图全点，不混合不同物理域 | `single_graph`检查batch和ptr=[0,N] | 真实两图Batch拒绝；单图Batch通过；点式路径置换等变 |

原官方Car的`unified_pos=True`会替换x，但wrapper固定fx=None；若仍fun_dim=4，preprocess输入维数不符。新模型对这个无效组合明确报错。数值对照使用有效备用配置`space_dim7/fun_dim0`，默认正式配置仍3+4/unified off。`Time_Input`正式为false；官方可选time_fc在tuple路径没有T输入，不宣称时间条件能力。

## C. 数值、原loss和checkpoint组件验证

详细误差：[model.json](linearno_audit/l7/model.json)；日志：[model-tests.txt](linearno_audit/l7/model-tests.txt)。新增模型6方法通过，unittest **24.198秒**，零失败/错误/跳过。

| 验证 | 实际结果 |
|---|---|
| 正式构造/完整参数keys与初始化 | 3,852,420；与固定官方全部参数值/RNG完全相同 |
| CPU FP64与FP32，默认点式路径 | 逐层/final/loss/input及每个参数gradient/Adam一步/strict往返：max/mean abs和relative均0；阈值1e-12/1e-10、1e-6/1e-5 |
| 原生CUDA FP32，默认与unified备用路径 | 上述同权重检查误差0，atol1e-5/rtol1e-4；官方类未改写 |
| CPU unified备用路径 | get_grid/逐层/final/梯度/step一致；仅测试官方参考端临时将Tensor.cuda置identity，不能称原生官方CPU支持 |
| 独立double完整网络oracle | forward max abs **1.3877787807814457e-17**；全部梯度max abs **4.440892098500626e-16** |
| 真实PyG/几何不参与/点置换/可变N | N1/13/29；输出[N,4]，梯度finite，输入不变，多图拒绝 |
| 原train函数AST一步 | 原normalized所有点velocity3 MSE + .5×surface pressure MSE，与独立同权重手算/Adam/OneCycle一步一致；保留原return pressure,velocity次序 |
| 本地整模型新进程 | Car实际cwd→`models.LinearNO.Model`，本轮生成且可信的whole object读取，输出逐位相同；没有加载外部官方pickle |

正式大配置只做实际构造/初始化/参数量；数值对照保留8层，缩小为d16/h4/M8/N17，unified为N15。不是32,186点真实训练/性能结果。已验证的模型保存加载不等于metadata-first原生train→checkpoint→resume→eval闭环。

## D. 已明确评价语义的独立验证

`car_metrics.field_metrics`同时返回physical surrounding-velocity3联合rL2、physical surface-pressure rL2、normalized region/channel MSE，名称分开。按原eval使用已保存输出mean/std解码，保持其`x*std+mean`行为；不重拟合、不自动加epsilon。零rL2分母和空region明确拒绝。

`drag_pair`只把surface压力和surface速度送给`coefficient`；后者使用调用方提供的真实sample目录，不写死param0。它组合原 `utils/drag_coefficient.py` 的VTK低层原语，保留面积、法向、速度梯度、压力转cell和Cd系数的原公式；原helper和旧入口完全不变。没有实现released-bug分支，也没有宣称该诊断动作已可用。

新增4方法通过，unittest **4.231秒**：

- NumPy独立手算4项field指标、无grad；修改volume pressure不影响正式评价项。
- 分母/mask/normalizer错误检查；不悄悄用epsilon掩盖无效样本。
- 一个非零fold路径的纯路由检查，明确3个surface点/4个volume点，捕获传入的正确surface数组。
- 真实内存VTK四边形与原数学helper，新的路径适配与旧cal_coefficient结果完全相同；点数和通道不符拒绝。

最后两项仅在测试里隔离文件存在性/读取调用，用内存合成VTK对象。没有创建同名假数据、没有读取真实raw/preprocessed文件；不能据此声称真实mesh/文件完整性通过。完整eval聚合、Spearman、散射/导出和训练入口仍未接线。

## E. 回归、冻结与环境

[regression-results.json](linearno_audit/l7/regression-results.json)记录实际运行的旧Transolver四方法、原Car套件、已接受Air模型和L2全部attention检查；[commands.json](linearno_audit/l7/commands.json)给出重跑命令。其他L1–L6证据在对应报告保留，未重复执行的六Standard原生闭环仍引用L6结果，不能称本轮全部重新运行。

本轮合计 **48个不同测试方法通过**：新模型6、评价4、上述回归38，零失败/错误/跳过。逐模块unittest与进程耗时见 [results-summary.json](linearno_audit/l7/results-summary.json)。

本机版本见 [environment.json](linearno_audit/l7/environment.json)：Python3.13.9/Torch2.13+cu130/PyG2.3.1、RTX5090 Laptop；已安装真实VTK用于合成数学检查。远端Py3.10/Torch2.11/cu128、真实数据、完整采样/训练、精度/收敛、实际epoch耗时均NOT RUN；没有安装缺失torch_cluster/pyg-lib。

冻结检查见 [delivery-check.json](linearno_audit/l7/delivery-check.json)：以L7分类/hash和L0冻结双重核对，既存源码与已有测试字节不变；只在STATUS和REPRODUCTION_MATRIX前插本阶段记录。L0广义冻结的L4/L5已有三个例外单列，绝不算成本轮改动。所有新测试运行产物指定仓库外路径；`LINEARNO/**`与monitor仍不存在，N/A，未创建替代包。

## F. 阶段条款状态与未完成项

| L7条款 | 状态 |
|---|---|
| 1–2：任务模型、M、温度/拼写、tuple输入 | 已实现并独立验证 |
| 3：生产main/eval/factory/checkpoint/launcher选择、resolved OneCycle | **未接入**；原train函数的一步检查不是该项通过 |
| 4：official完整objective和默认paper objective/profile/正交evaluation_spec | 原release loss已独立验证；paper空间/reduction仍为C08未决；生产profile尚未接线 |
| 5：正式参数/官方完整数值对照、PyG和模型级保存加载 | 已通过；原生任务存档/续跑仍不能据此标通过 |
| 6：preprocessed preflight、原生checkpoint/resume/eval和fallback | **未完成**；真实路径/surface force独立组件已验证，尚未接实际eval |
| 7：回归与八任务矩阵 | 所列针对性回归已执行；工业接线和八任务完整生产矩阵未完成 |

阻塞来源可执行核验在 [prerequisites.json](linearno_audit/l7/prerequisites.json)：`require_resolved_objective(resolve_config('car'))`仍拒绝space/reduction；AirfRANS另缺volume/surface训练通道决定。L0 C08不能以`UNRESOLVED`进入默认训练，也不能自行改为某种rL2。

用户需要确定的协议已再次异步提出：两工业任务训练rL2使用physical空间的相关通道联合范数，还是normalized空间逐通道范数比后平均；Air的训练通道需同时明确。无论选择如何，已要求的两个surface系数、region和独立official normalized-MSE版本保留。具体选择将作为本地复现解释写入台账，不冒充作者已披露细节。

L8明确要求L4–L7原生闭环已经完成，遗漏必须回到对应阶段，不能在L8临时另造路径。当前Car和Air生产接线都未完成，因此没有把L8/L9/L10标成PASS或写最终复现完成报告。收到C08决定后，仍使用已有自动阶段授权继续，无需再次申请L7–L10许可。

## G. 已自审的五点

1. 官方3+4构造元数据与实际7维x、geom无效合同一致。
2. actual M与原乘数的映射、两个温度及拼写、两层output投影均由原源码/独立oracle验证。
3. 全部8层、初始化时序、placeholder、正常单图与备用unified位置路径均有独立证据。
4. paper field评价与release MSE明确区分，force使用正确区域/真实路径，未动旧函数。
5. 新whole-object/state往返与尚未完成的原生metadata/resume闭环严格区分；冻结和旧模型回归不冒充生产接线验收。

**L7在C08及前置工业接线处暂停，尚未完成；未执行L8–L10。**
