# KCDNO 实施状态

## 2026-09-16：离线报告按每task最后验证损失选择单个seed完成

三个show入口现在每个数据集仅展示最后一次记录的验证损失最小的run/seed；PDE六任务独立选取，一张2×3训练汇总、一张2×3独立评价汇总。selection.json/CSV记录所有候选的seed、loss、真实验证epoch、权重/成员、选择或排除理由；曲线/复制场图/评价仅使用获选run。绝不取历史最好epoch或用正式测试值选seed。Car/Air按正确分项和保存权重派生目标；Air398轮最后验证390如实记录；缺损/非有限/未完成run不退回更早好值，legacy缺状态明确标注。

[报告](CDLNO_RESULT_REPORTS.md)、[用法](../tran_evaluate/show/README.md)、[证据](show_audit/selection/)。14/14定向测试通过，6task×3seed记录夹具、工业加权/多成员/稀疏验证、失败/NaN/缺项、同值规则、dry-run、只复制获选run、源文件/旧报告hash与ZIP检查；实际渲染并查看两张六子图。149个既有生产/启动文件和3个show shell与本轮快照字节一致。没有模型/训练/数据/checkpoint/依赖改动，未访问远端实际结果，未训练或commit/push。历史交付原样保留。本阶段结束，未执行下一阶段。

## 2026-09-16：三子项目离线结果报告脚本完成

新增tran_evaluate/show下Car-Design-ShapeNetCar.sh、Airfoil-Design-AirfRANS.sh、PDE-Solving-StandardBenchmark.sh，复用_report.py。读取原训练JSONL/独立评估JSON/数值数组及已存场图，在实际远端子项目result_visualizations下创建独立时间戳文件夹+ZIP：loss/分项/step-full曲线、测试指标、Air升阻力配对与Cp/Cf、原场图、PDF/600dpiPNG/图注/CSV/LaTeX表/离线HTML。按run/member/seed/fold/评估次数隔离，不混平均、不平滑、不把验证当测试。

报告[CDLNO_RESULT_REPORTS.md](CDLNO_RESULT_REPORTS.md)，用法[show/README](../tran_evaluate/show/README.md)。8项定向测试通过，模拟带空格远端目录/外部cwd、三实际shell、八任务结果夹具、原文件hash、稀疏epoch、失败/缺项、数组轴、真实图/PDF字体与ZIP均检查；实际查看合成格式示例。149个原生产/启动文件byte不变。新工具不导入Torch/模型/exp、不读取真实数据/checkpoint、不安装依赖；源文件和已有修改保留。远端真实训练产物未访问，缺记录不能从权重恢复。无真实训练或commit/push。本阶段结束，未执行下一阶段。


## 2026-09-16：八任务每50轮可视化完成

当前明确请求授权可视化接入，见[交付/使用/论文依据](CDLNO_PERIODIC_VISUALIZATION.md)和[证据](periodic_visualization_audit/)。CDLNO所有既有模式、KCDNO all/off、lrsa_matched的八任务在完成50/100/...及最后epoch自动输出固定前两个held-out案例；PDF/600dpi PNG、原始NPZ、固定色标、英文/LaTeX caption及单例诊断。NS预测回填10步、Plasticity20个独立T；Car surf压力/volume薄层速度；Air原数量固定抽样+真实radius_graph，明确不是正式scatter平均评价。原Transolver分支保持现状。

完整287tests/319.519s无fail/error，3个skip（2个Air torch_cluster依赖项+外部LRSA未指定）；LRSA之后实际补跑2/2精确通过。最终绘图11tests/17.178s仅1个Air建图skip；Car原train.main合成完整epoch实际出图且权重/RNG精确，Air真实PyG已抽样输入/原weighted loss片段通过但完整采样链未运行。CPU观察器前后及下一步optimizer/scheduler/RNG精确；有限GPU FP32/math/TF32off、确定性测试设置下下一步精确。初次GPU未固定确定性零容差失败约1.9e-9及旧Car pos2测试夹具问题如实记录，没有改生产后端或模型。

161个保护生产文件byte相同、9个完整入口/训练AST剥离精确新增观察调用后相同；快照`/home/hwz/CDLNO-artifacts/visualization-before-133lyhbo/source`保留。模型/数据/loss/采样/时间循环/optimizer/scheduler/原checkpoint保存协议未变，既有seed修复保留。当前只完成可视化，不将旧V1归档基础宣传为八任务resume完成。无真实数据/训练/远端2.11cu128或真实论文案例验收，无commit/push/安装依赖。原历史状态如下保留。

本阶段结束，未执行下一阶段。


## 远端seed解析失败的交付核查（2026-09-16）

用户远端run_seed已传seed，但旧parser拒绝`--seed 0`。本地用HEAD旧parser复现exit2，
当前parser接受`--seed 0 --gpu 1`；不导入exp或读取数据。提供仅含标准parser、
新家族seed helper、experiment seed记录三个文件的[最小补丁](kcdno_audit/seed_suite/seed-support.patch)，
独立副本apply-check/apply后3文件与当前实现逐字节相同。未修改模型/数据/循环，
未新增研究功能或运行训练；远端尚需同步，不宣称已修复远端文件。
具体上传/只读核查方法见[KCDNO_SEED_SUITE.md](KCDNO_SEED_SUITE.md)。

## 六标准任务单seed队列完成（2026-09-16）

新增[run_seed.sh](../tran_evaluate/kcdlno/run_seed.sh)，选择0/1/2之一，严格按
darcy→airfoil→plasticity→elasticity→ns→pipe，各任务train→eval→即时输出带seed结果，
然后才进入下一任务。每任务独立带seed时间戳目录，seed_summary和队列summary增量保存；
失败停止并保留既有结果。可选seed在标准新家族helper真实设置Python/NumPy/torch
CPU/CUDA RNG，子进程PYTHONHASHSEED也设置；记录进入原初始化元数据、配置与结果。
省略seed的旧命令不重置RNG；旧模型和六exp/数据/时间循环/预设/原启动脚本未改。

定向8/8（6.169s）及旧记录2/2（10.901s）通过：六任务三seed重复初始化精确一致、
不同seed权重不同、strict/checkpoint/sidecar、逐任务报告、故障停止、60条命令预览。
195个已有源文件192未变，仅3个seed/记录helper有授权差异。初次一项测试误用35点
配5×5模型，改为实际H×W后通过，未放松生产合同。
无真实训练/数据读取/GPU重复性或远端验收；不新增resume/确定性后端设置。
详见[报告及命令](KCDNO_SEED_SUITE.md)。本阶段结束，未执行下一阶段。

## KCDNO 八任务启动包装（2026-09-16）

新增 `tran_evaluate/kcdlno/`，为 Darcy、Elasticity、Airfoil、Pipe、Navier–Stokes、
Plasticity、ShapeNet-Car 和 AirfRANS 各提供一个薄脚本。脚本只转发到已验证的
`tran_evaluate/kcdno/` 入口，实际模型注册名仍为 `kcdno`，默认 profile 为
`kcdno_v1`，不复制模型、数据读取或训练循环。每个脚本支持 `train`、`eval` 和
`train_eval`；顺序模式在训练成功后才评价同一 run，并保留 `all`、`off`、
`lrsa_matched` 选择。最终9个shell语法及120个分发/帮助/负向检查通过，共112条底层
命令预览；使用禁止Python调用的临时替身执行，因此最终检查没有数据入口访问。
含空格路径、参数末尾覆盖、同run及Air两种my_path语义通过；173个已有生产/脚本
文件hash不变。本轮只增启动目录及增量状态文档，未变更模型或任务协议。

初次帮助检查发现 `--help` 被错误消费导致调用真实Darcy入口；因缺本地远端文件，
在数据读取处失败、未执行训练，其失败目录已清理。已修复帮助和无参数/选项解析，
最终保护性检查全部通过。未执行真实训练/评价、新GPU/模型回归或远端数据验收。
原工业路径/fold限制及任务级完整resume尚未接入的边界保留。使用说明、实际命令和
自审证据见 [README](../tran_evaluate/kcdlno/README.md) 及
[verification.json](../tran_evaluate/kcdlno/verification.json)。

本阶段结束，未执行下一阶段。

## K4–K10 全部完成并自审（2026-09-16，当前状态）

最新用户授权为连续执行K4至K10，各阶段自审后继续；已完成最终交付，现在停止。下列旧“等待单独授权/待审查”等状态为当时历史，不代表当前进度。

- 八任务均接入KCDNO all/off及可训练matched LRSA-full，旧Transolver/CDLNO三front模式和CDPA规则保留。
- 最终257项检查运行，OK(skipped=1)，263.562s；另补NS/Plasticity off合成优化器步1测试通过0.525s。唯一未执行子项为缺torch_cluster的Air完整抽样epoch观察器，真实PyG接口/原mask loss及加载已执行。
- K0固定同权重41/41精确回放，atol=rtol=0；119旧文件字节相同、15完整入口/helper/registry/YAML的冻结投影相同。三个原cwd新进程strict加载通过。
- K2/K3本机GPU FP32/AMP重新验收；K9两代表配置×四模型FP32同步性能已测。all未显示普遍提速，不将参数/MAC下降外推epoch；远端2.11/cu128、真实数据读取/抽样评价/训练/收敛/精度未验证。
- K10修复新入口不应使旧parser提前依赖shared包、工业pickle非权重属性/固定reference校验、Plasticity d1时间边界；模型核心数学与旧冻结数据协议未改。K8修复新Air argv=None；失败与通过日志均保留。
- 保存协议仍为标准state_dict/Car整对象/Air列表和整对象，不新增resume；旧V2–V5任务级完整优化器/RNG续训仍未接入。

交付：[最终报告](KCDNO_IMPLEMENTATION_REPORT.md)、[需求矩阵](KCDNO_REQUIREMENTS_MATRIX.md)、[八任务三图命令](KCDNO_COMMANDS.md)、[性能结果](KCDNO_K9_PERFORMANCE.md)、[累计diff](kcdno_audit/k10/integration-changes.patch)、[最终日志](kcdno_audit/k10/final-suite.txt)。保留K0–K3及全部旧CDLNO/A/V证据、用户已有修改；未commit/push/PR/安装依赖/下载或启动真实训练。

| 当前阶段 | 状态 |
|---|---|
| K0–K3 | 用户此前已接受，保留同权重回归证据 |
| K4–K7 | 八任务主模型/历史off接入完成并自审 |
| K8 | 可训练matched-full独立配置/模型/八任务加载完成并自审 |
| K9 | 完整参数/MAC、有限CPU/GPU性能核查完成 |
| K10 | 逐项需求反查、必要接入缺陷修复、最终回归与交付完成 |

本K阶段结束，未执行下一阶段。


## K9 完成并自审，继续 K10（2026-09-16）

完整MAC/实际参数/执行计数核对、15/15 CPU检查通过；两代表配置×4模型本机GPU FP32计时完成，all未表现为稳定提速，数据及热点如实见 [K9报告](KCDNO_K9_PERFORMANCE.md)。不外推真实epoch/精度/远端环境。旧工具默认模型列表保留，未改模型数学。

## K8 完成并自审，继续 K9（2026-09-16）

八任务可选择kcdno all/off与独立family lrsa_matched；完整LRSA固定full，共享新任务wrapper，不触及旧模型。18/18有限回归通过，matched公共权重数学等价、八任务strict往返和工业新进程加载通过。详见 [K8报告](KCDNO_K8_MATCHED.md) 与 [命令](KCDNO_COMMANDS.md)。Air实际argv=None接入缺陷已修复并验证。无真实训练。

## K7 完成并自审，继续 K8（2026-09-16）

八任务KCDNO all/off接口、训练/评价选择和实际checkpoint格式已接入。Air真实PyG原损失合成步、列表/整模型新进程加载、5测试和4旧同权重回放通过；原21冻结文件不变。详见 [K7报告](KCDNO_K7_AIRFRANS.md)。未执行真实抽样评价或数据训练。

## K6 完成并自审，继续 K7（2026-09-16）

Car新模型、原整对象协议和原mask训练步已接通；5/5检查、4/4 K0原权重回放通过。真实PyG单图/多图拒绝、旧工业冻结、新进程加载已核查。详见 [K6报告](KCDNO_K6_CAR.md)。无真实训练/拖曳评价，原路径/fold限制保留。

## K5 完成并自审，继续 K6（2026-09-15）

NS/Plasticity已接入新family，原10→10/逐时间20次更新协议保留。12/12检查、8/8旧同权重时间任务回放通过；原时间loop语句直接合成执行，NS optimizer/scheduler各1次、Plasticity optimizer20/scheduler1次，cache逐forward隔离。详见 [K5报告](KCDNO_K5_TEMPORAL.md)。六标准任务已有接口/启动/strict加载；工业任务随后实施。真实训练/远端环境未执行。


## K4 完成并自审，继续 K5（2026-09-15）

用户最新授权连续 K4–K10；各阶段自审后继续，不再套用旧逐阶段停顿。四静态任务新family/factory/wrapper/profile/脚本/strict state_dict接入完成；6项新检查、16份K0旧四任务同权重回放和完整冻结AST通过。原loss/decode语句、真实N减宽、5×7、标准cwd新进程加载通过。未运行新任务GPU/远端/真实数据训练。详见 [K4报告](KCDNO_K4_STATIC.md) 和 kcdno_audit/k4。旧记录保留。


## K2已审查通过；K3完成，待审查（2026-09-15）

本轮仅 K3：新增 `cdlno/kcdno/core.py` 的 `KCDNOBlock` 与 `KCDNO`。真实计算为 point RMS→Down→FFN1残差→K2历史Reader→FFN2残差→一次Up latent RMS→Up/点残差→point FFN或dense ConvFFN残差。raw T严格取FFN2后/Up norm前，非末层Writer(T)；L个独立完整点域block，无SA/Bridge/persistent/额外final Up。core接收及返回 `[B,N,d]` 特征，任务lift/LN_out/head留待后续接入。

每次forward新建局部history，tuple快照先读再写；首层无reader、末层无writer、L1/off无历史参数。现有K1配置不改；K3固定FFN hidden2d，其他已解析hidden在core构造时明确拒绝。旧block/CDPA和生产任务入口不改。

交付：[K3报告、公式/参数/时序](KCDNO_K3_CORE.md)、[最终测试](kcdno_audit/k3/final-tests.txt)、[实际默认计数/参数](kcdno_audit/k3/core-evidence.json)、[9旧core回放](kcdno_audit/k3/old-core-replay.json)、[冻结](kcdno_audit/k3/freeze.json)、[完整阶段diff](kcdno_audit/k3/stage-changes.patch)。57/57联合测试9.912s（新K3 10、K2 16、K1配置13、旧公共模块18）；9/9 K0旧core同权重/输入/输出/梯度完全一致，atol=rtol=0。

L1/2/4/8/12×all/off小配置、point/5×7conv、非法值、参数/storage独立、特殊初始化一次、raw T与一次Up norm、tuple因果/跨forward隔离、gamma0同权重off等价及strict state_dict往返通过。手工原语+K2 double显式核参考：输出最大误差3.652e-7、所有输入/参数梯度最大1.959e-6，容差2e-6/3e-5和2e-5/3e-4。本机GPU point/conv共2个FP32及4个AMP前反向通过，K2原FP32/autocast关闭/梯度验证重跑通过；非远端环境验收。

默认L8/d128/h8/M64/r16，Down8/Up8/latentSA0/latentFFN16/PointModule8；all写7/Q7/逻辑读28，Reader API7，numerator/denominator einsum各7，source softmax7，Down/Up SDPA16；off历史计数0。实际参数（不含任务lift/head）：point all2,605,063/off2,572,800；conv all3,785,735/off3,753,472。未作性能计时或速度推断。

起点main/222647f，保留已有未提交K1/K2。422文件快照 `/home/hwz/CDLNO-artifacts/k3-before-bd2pluhq/source`；420文件不变，仅本STATUS/memory增量更新。公共modules/旧核心/CDPA、K1/K2、全部任务数据/训练/脚本/依赖/已有测试保留。Python3.13.9/torch2.13+cu130/RTX5090 Laptop；未安装依赖或运行真实数据/训练。当前core没有工业整对象/任务checkpoint或resume接入，原V2–V5等历史未完成项保持。

| 阶段 | 最新状态 |
|---|---|
| K0/K1/K2 | 用户已审查通过 |
| K3 | 独立block/core、有限数值/梯度/计数/GPU与旧core回归完成，待审查 |
| K4–K10 | 未执行；等待单独授权 |

自审未发现待裁定结构冲突或未修复问题。后续边界：八任务wrapper/lift/head、真实loss/normalizer/时间协议绑定、PyG新模型/工业pickle/list、sidecar+实际权重校验、远端2.11/cu128、性能及所有真实数据实验尚未验收。历史记录如下保留。

本K阶段结束，未执行下一阶段。

## K1已审查通过；K2完成，待审查（2026-09-15）

本轮仅独立KernelHistoryWriter/Reader与reference。新 `cdlno/kcdno/history.py`：源层RMSk/Wk→phi→FP32 sum memory/mass；接收层一次RMSq/Wq→按来源stack摘要→独立核归一化→统一source softmax→RAW候选融合→无约束gamma残差。biasFalse/Xavier gain1/norm1/w0/gamma.1；空history同一U对象。无Wv/Wo、token softmax、额外缩放、raw-T rekey、mask或持久缓存；旧CDPA不改。

交付：[K2报告/公式代码测试表](KCDNO_K2_KERNEL_HISTORY.md)、[最终日志](kcdno_audit/k2/final-tests.txt)、[汇总](kcdno_audit/k2/summary.json)、[diff](kcdno_audit/k2/code-changes.patch)、[冻结](kcdno_audit/k2/freeze.json)。最终68/68通过4.169s（新16/K1配置13/旧模块CDPA39）；7/7原冻结检查0.444s；K0旧同权重41/41回放。double缓存/显式12组合最大梯度误差2.66e-15，2个double gradcheck；FP32/double24组合最大1.43e-6（atol5e-6/rtol1e-4）。CPU BF16及本机GPU FP16/BF16 AMP实际算子FP32/autocast关闭、前反向通过；极端缩放到1e15、小分母clamp通过，不承诺任意有限值无溢出。

起点main/222647f，已有K1未提交成果；403文件快照 `/home/hwz/CDLNO-artifacts/k2-before-98ze33eu/source`。401已有文件字节不变，仅本STATUS/memory增量更新；K1源码/证据、旧模型/数据/训练入口/依赖/脚本/测试/旧CDPA保留。实际Python3.13.9/torch2.13+cu130/CUDA13/RTX5090 Laptop；远端2.11/cu128未执行或重装。真实数据/训练、新KCDNO完整模型/任务loss/checkpoint/性能未运行。

| 阶段 | 最新状态 |
|---|---|
| K0/K1 | 用户已审查通过 |
| K2 | 独立核历史模块、reference及数值/梯度/AMP验证完成，待审查 |
| K3–K10 | 未执行；按用户逐阶段授权 |

K3仍需组装两FFN间reader、pre-Up T writer、局部tuple时序、首层无reader/末层无writer/L1-off无闲置参数及L8的7写7Q28读。独立模块空history恒等不代表这些整网要求已经验收。旧A/V及K0/K1记录继续保留如下。

本K阶段结束，未执行下一阶段。

## K0已审查通过；K1完成，待审查（2026-09-15）

本轮仅K1：独立 `cdlno/kcdno/` 配置/profile/显式参数/metadata基础，正式family=kcdno，无旧F/P/front/CDPA字段。主profile与transolver_shape_match八任务结构数据完成；train显式CLI>profile>家族默认，eval先读已有resolved配置再校验显式字段，初始化起点/运行字段分离，旧缺family交旧规则。严格新JSON、独立目录建议，不注册占位模型，不接八任务入口，不新增resume。

交付：[K1报告](KCDNO_K1_CONFIGURATION.md)、[实际命令](kcdno_audit/k1/commands.txt)、[最终26项](kcdno_audit/k1/final-tests.txt)、[代码diff](kcdno_audit/k1/code-changes.patch)、[冻结证据](kcdno_audit/k1/freeze.json)。新13项配置测试及旧6配置+7冻结共26/26通过；K0同权重41份在修改前/后均回放通过（33旧CDLNO CPU、8旧Transolver GPU，含真实PyG旧工业整模型/列表）。这不是KCDNO模型/核读取或新任务训练检查。

起点main/`222647fef343e9fe929e65412f5c7449ced5517b`干净；371文件快照在`/home/hwz/CDLNO-artifacts/k1-before-khe93ugp/source`。K0的41份/136文件hash全在，无需补造基准；原夹具不改。旧生产/训练/数据/依赖/测试/脚本与K0材料保持字节一致，仅本STATUS和memory增量更新。实际Python3.13.9/torch2.13+cu130/PyG2.3.1；远端Python3.10/3.11、torch2.11/cu128未执行，没有安装依赖。新子包在现有cdlno*包发现内，三个原cwd纯配置加载通过；无实际editable安装。

| 阶段 | 最新状态 |
|---|---|
| K0 | 用户已审查通过；保留全部原审计/夹具 |
| K1 | 配置/profile/family/metadata基础完成，等待用户审查 |
| K2–K10 | 未执行；须逐阶段另行授权 |

后续须补新wrapper的reference/grid/time/output合同与真实权重绑定，不能以本轮core JSON宣称新模型checkpoint可训练/恢复。V1公共归档已有、V2–V5任务resume未接入的旧边界保持。无核读取/整模型/训练入口/真实数据训练/commit/push。下列K0原文保留历史。

本K阶段结束，未执行下一阶段。

## K0完成，待审查（2026-09-15）

当前授权仅K0，未执行K1及以后阶段。当前旧仓库HEAD为`9f72946e0adbfe27ba0b75b1646fa697ae16d3df`（main）；起点工作树干净。KCDNO规格文件存在且已读，**尚无KCDNO生产配置/模型/factory/任务入口**。

交付：[参考审计](KCDNO_REFERENCE_AUDIT.md)、[回归夹具索引](kcdno_audit/fixture_index.json)、[冻结基线](kcdno_audit/baseline.json)、[实际命令](kcdno_audit/commands.txt)。本轮只新增这两个文档与`docs/kcdno_audit/`内审计材料；旧CDLNO/A/V阶段文档、AGENTS、memory和全部生产/测试文件保留。

### A. 范围与结论

已审查KCDNO规格、CDLNO v1.2与当前状态、front A0–A4、最新输出/V1边界；建立部件复用映射、两FFN插入点、八任务字段/损失/入口/目录/保存合同及K1–K10最小建议范围。K1–K10独立执行指南当前未找到，分工是建议，后续用户明确点名内容优先。

现有Down/Up、RMSNorm、PlainFFN、dense ConvFFN可供新类组合。新history必须位于FFN1之后/FFN2之前；不调用整个旧no_sa block后补history。旧CDLNO继续F/P、bridge、persistent rear、CDPA和HF readout；新KCDNO独立L个点域block，不修改旧family或类路径。

### B. 新增文件与差异

| 文件/产物 | 用途 |
|---|---|
| `docs/KCDNO_REFERENCE_AUDIT.md` | 完整A–F审计、版本/来源/许可、代码映射、八任务接口、未完成项和后续建议 |
| 本文件 | 独立K阶段状态，不覆盖旧CDLNO/A/V记录 |
| `docs/kcdno_audit/make_regression_fixtures.py` | 独立capture/replay脚本；实际wrapper、原模型、三原cwd新进程、同权重/梯度/加载 |
| `docs/kcdno_audit/audit_static.py` | parser AST、安全CLI/注册/预设/函数摘要和冻结hash核查 |
| `baseline/environment/presets/entry-contracts/cli-*/capture*/replay*/fixture_index/freeze`等JSON/log | 真实命令/数值/版本/源码证据与失败记录 |
| 外部`/home/hwz/CDLNO-artifacts/k0-before-zf3l4cve` | 302文本源码快照、完整入口AST、41份权重/输入/输出/诊断梯度与工业对象/list，约16.26MiB最终夹具；不进入git |

无生产修改，无新kernel/history数学实现，无训练/数据/依赖/既有测试修改。git可审查diff全部为新增审计材料。全328个原tracked文件保持SHA256一致，见[freeze.json](kcdno_audit/freeze.json)。

### C. 公式/轴与参数归属

新KCDNO待实现：每层`point RMS→Down→U=S+FFN1(RMS1(S))→Uhat=history(U,tuple(past))→T=Uhat+FFN2(RMS2(Uhat))→Up→点残差/PointModule`，最终LN/head。S/U/T均[B,M,d]。

源s拥有Wk_s，缓存`K_s^T T_s [B,r,d]`和`sum_token(K_s) [B,r]`，raw T作value；接收l拥有Wq_l，Q[B,M,r]一次读取全部先前来源；来源softmax权重[B,M,l]，w0、gamma.1，`Uhat=U+gamma*(Σalpha*rawR-U)`。无M×M token softmax/current self-read/Wv/Wo。L8主版应7写7Q28读。以上是后续验收合同，K0不标为实现通过。

### D. 实际验证

| 项目 | K0结果/范围 |
|---|---|
| 捕获/回放 | **41/41通过**：24任务×front模式、9核心front×CDPA组合、8原Transolver模型；严格同权重/输入，输出/诊断梯度atol=rtol=0 |
| CPU合成 | 33份CDLNO：实际wrapper/core、FP32输出+诊断backward；无optimizer/原loss训练 |
| GPU实测 | 8份原Transolver模型，直接原源码CUDA路径，FP32；这是旧模型基准，KCDNO/GPU新模型检查未运行 |
| 真PyG对象 | Car/Air分别三CDLNO+原模型，单图N11/19；真实Data/Batch、输出/诊断梯度/整对象及Air list；不含radius_graph/真实数据 |
| checkpoint | 24份CDLNO实际StaticRun/CarRun/AirRun；same mode、chunk0→1、eval省略mode恢复、显式冲突拒绝、sidecar字节不变，全通过 |
| CLI/元数据 | 3/3子项目进程，八任务默认/老注册、24组三模式train/eval解析，8 JSON+Air YAML留存 |
| 旧冻结回归 | 既有7项AST/hash检查**7/7通过，1.221s**；未修改其逻辑，未重跑无限扫描 |
| 导入方式 | 三cwd显式PYTHONPATH均通过；去掉PYTHONPATH均ModuleNotFoundError，当前未editable安装；如实记录为当前本地安装限制 |
| 全部现有文件 | 328份tracked文件字节相同；所有新增只在K0 docs区域 |

[环境](kcdno_audit/environment.json)：Python3.13.9、torch2.13.0+cu130/CUDA13.0、RTX5090 Laptop、PyG2.3.1；无torch_cluster/pyg-lib。SDPA=MATH，FP32、单CPU线程、TF32/AMP/compile关，cuDNN benchmark关/deterministic开；只在审计子进程设置，未修改生产默认。目标仍为远端Python3.10/Torch2.11/cu128（3.11候选），未远端验证或安装任何依赖。

实际主命令：

```bash
python -B docs/kcdno_audit/make_regression_fixtures.py capture --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/kcdno_audit/capture.json
python -B docs/kcdno_audit/make_regression_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/kcdno_audit/replay.json
python -B docs/kcdno_audit/audit_static.py --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures
python -B docs/kcdno_audit/audit_static.py --freeze
git diff --check
```

capture已完成，现有目录拒绝覆盖。后续只能重放同一份旧权重/输入；三进程完整命令和7项既有冻结测试命令见commands.txt。初版审计脚本错误已在本轮自身修正，最初22份不完整捕获不算最终验收；记录见execution-notes.json。

### E. 冻结及旧缺项

本轮未修改原Transolver/CDLNO核心、三mode/CDPA历史/初始化、八任务数据/采样/normalizer/loss/时间循环/optimizer/scheduler/指标、checkpoint实现、旧脚本或依赖。7项既有审查恢复原Transolver基线AST/字节合同；K0的328文件hash作后续新family接入的直接前基准。

实际A3/A4已存在且可用，不能照用户粘贴AGENTS旧段误写未实现。另行V1完整归档/绘图基础已存在，V2–V5任务resume/周期图未接入；普通model.pt/整模型文件不含完整optimizer/RNG续训状态。K0不自动补这些旧缺项。Car日志变量名/阻力路径、Air旧MAE分支及旧parser别名问题分别记录，未归为KCDNO回归或擅自修复。

### F. 已自审和未验证

已自行复核：两FFN与Up norm位置、writer/reader参数归属和旧CDPA隔离、真实git/阶段/目录状态、同权重回放与pickle稳定路径、源码冻结与验证边界。当前没有需要先重写旧baseline的实质冲突。

未验证：KCDNO模型与数学（尚未实现）、KCDNO八任务入口、远端Torch2.11/cu128、新模型AMP/性能；真实文件完整读取、采样/物理指标、训练、收敛、准确率和epoch效率。K0诊断loss不是原任务训练链路；当前合成夹具不证明全部历史checkpoint兼容，保留pre-A1真实旧夹具供后续。

| K阶段 | 状态 |
|---|---|
| K0 | 审计/独立夹具/CLI与冻结证据完成，待审查 |
| K1–K10 | 未实施；后续必须按用户单独点名的阶段执行 |

本K阶段结束，未执行下一阶段。
