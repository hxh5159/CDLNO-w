# LinearNO 实施状态

## 2026-09-18 - L10 最终交付

**PASS（实现和无数据验收完成；真实数据、完整训练、远端环境 NOT RUN）。** L6 的 AirfRANS 目标已按用户决定固定为标准化四通道 `volume MSE + 1×surface MSE`；L7 的 Car 目标已按用户决定固定为全点三速度 normalized MSE + `0.5×` surface pressure MSE。两项均不把论文 physical rL2 文字描述改名为训练实现，rL2/drag/Spearman 保持独立评价轴。

L10 反向源码审查确认 Q/K softmax 轴、实际 M、任务专属温度/初始化/位置语义、8 个完整 block、输出投影、Air/Car wrapper、strict metadata/checkpoint 和旧 Transolver 分支均与固定源码或已确认 profile 一致。实际命令：抽查 `28 passed, 46.91s`；全 `tests/linearno` `84 passed, 2 warnings, 394.61s`；profile/schema/converter `13 passed, 13.97s`；`bash -n tran_evaluate/linearno/*.sh`、compileall 和 `git diff --check` 通过。L9 的综合 `118 passed` 与有限 RTX5090 FP32 smoke 证据保留在 [L9报告](LINEARNO_L9_REPORT.md)。

完整参数量为 Darcy 1,766,145、Elasticity 585,217、Airfoil/Pipe 1,765,889、NS 3,377,921、Plasticity 1,799,428、AirfRANS 3,358,788、ShapeNet-Car 3,852,420。固定的实现/测试交付见 [最终报告](LINEARNO_IMPLEMENTATION_REPORT.md)。没有下载数据、读取真实数据、执行长训练、安装依赖、commit 或 push。`LINEARNO/PropagationMonitor` 在当前 checkout 不存在，记为 N/A。

阶段统一状态：L0 PASS，L1 PASS，L2 PASS，L3 PASS，L4 PASS，L5 PASS，L6 PASS，L7 PASS，L8 PASS，L9 PASS，L10 PASS。历史段落中 L6/L7 的 BLOCKED 记录对应用户决定前的状态；以本段和最终报告为准。

**本L阶段结束，未执行下一阶段。**

## 2026-09-18 - L9 综合验证

**PASS（真实数据/完整训练/远端环境 NOT RUN）。** L8 转换器、profile 和 metadata 汇总已完成；
L9 最终综合命令 `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/linearno tests/test_airfrans.py tests/test_shapenet_car.py tests/test_front_task_modes.py tests/msar_entry_projection.py` 返回 **118 passed, 2 warnings, 419.44s**。`bash -n tran_evaluate/linearno/*.sh`、13 项 profile/schema/converter 检查和有限 RTX5090 FP32 smoke 也通过。第一次综合中的复现矩阵冻结后缀误报已恢复 L1 原文并单独重跑通过；没有放宽 strict、删除测试或改变模型。

正式参数量：Darcy 1,766,145；Elasticity 585,217；Airfoil/Pipe 1,765,889；NS 3,377,921；Plasticity 1,799,428；AirfRANS 3,358,788；ShapeNet-Car 3,852,420。L9 证据见 [报告](LINEARNO_L9_REPORT.md)、[性能记录](linearno_audit/l9/performance.json)。当前 checkout 没有 PropagationMonitor，记为 N/A；既有 Transolver 冻结 hash 与旧 checkpoint/CLI 回归证据保持。**L9 阶段结束，未执行下一阶段。**

## 2026-09-18 — L6 重新审查（取代下文旧验收结论）

**PASS；真实数据/真实训练/VTK全量指标/远端环境 NOT RUN。** 用户再次确认默认采用官方 Transolver/LinearNO 的标准化四通道 `MSE(~surf) + 1*MSE(surf)`。原接入已使用此目标，无需切换 loss。此前提出的物理空间联合 rL2 未采用；论文训练细节仍不冒充已公开。

本次修复实际入口测试揭示的工程遗漏：恢复保存的seed/task、设置原生记录器所需resume checkpoint、严格核对profile→constructor、验证pointer hash、单图batch拒绝、多成员顺序续训（包括后续成员尚未开始）、将曲线历史绑定到epoch存档、恢复非validation epoch所需的val_surf。数据checksum现在包含实际VTU/VTP文件，normalizer绑定聚合checksum；provenance使用实际源码和AST规范化patch。运行目录先于loader预留；默认目录含evaluation hash；显式CLI数据/相对运行路径正确转发。新family的周期场图复用hxh渲染，仅在独立AirFields中按原均匀采样提取点，无需图kernel。没有修改旧共享模型、数据loader或旧训练分支。

`tests/linearno/air_entry_worker.py` 现在执行真实parser AST→run_cli→train.main→原记录器→每成员pair→新进程resume/eval。仅以内存真实PyG替代数据边界，VTK正式指标边界在本闭环中替换为shape/选择验证；不能称真实数据或完整物理指标验收。保留原20次采样validation、每batch Adam/OneCycle。3合成epoch、9训练图、每图13–15点采样11点，d8/L2，正式结构未改。单成员及双成员连续/中断后的weights、全部resume_state hash、训练采样顺序、最终预测逐位一致；10项非法参数/结构/pointer检查及final场图通过。评估未改architecture.json/config.json且恢复saved normalizer，不重拟合。

[实际命令与结果](linearno_audit/l6/review/results.json)：31 passed，157.67s；含此前Air全模型官方parity（本机CPU及可用CUDA）、L1旧八任务Transolver回归及旧Air13项。[冻结核验](linearno_audit/l6/review/freeze.json)：旧Air完整训练AST相等，13个Transolver/PhysicsAttention/Embedding hash一致，309个既有ignored文件一致，既有LINEARNO不存在（N/A）。初次复核中的schema checksum和测试hook序列化失败已修复并重跑；不隐藏历史失败日志。

L6自审通过：目标/评价分离、原生续训与随机流、多成员顺序、元数据严格性、旧模型冻结。既往L6 PASS未覆盖真实run_cli恢复缺口，以本次扩展证据为准。按用户已有连续授权进入L7，未启动真实训练。

## 2026-09-18 — L7 Car 接线推进（当前状态）

**L7：BLOCKED，等待 Car paper objective 的计算空间与通道聚合决定。** 连续 L7–L10 授权仍有效。`official_release` 已接真实 Car parser/main/train.main/eval；metadata-first、严格权重、数值normalizer、native train→checkpoint→resume→新进程eval及真实PyG合成通过。默认paper仍明确拒绝未决目标；未进入L8/L9/L10。

[本轮报告及可执行官方命令](LINEARNO_L7_REPORT.md)、[机器记录](linearno_audit/l7/integration/results-summary.json)、[冻结核验](linearno_audit/l7/integration/freeze.json)。初次综合102通过/2审计失败，精确投影修正后受影响34项全通过；合计105个不同方法已有通过证据（不是一次全量执行）。旧Transolver数学hash不变，Car旧分支完整AST一致。正式Car参数3,852,420；未运行真实数据/default radius graph/真实训练/远端环境。此前“Car未接线”的段落为历史，以下原文保留。

## 2026-09-18 — L6 AirfRANS 接入完成

**阶段状态：PASS（资源门控项 NOT RUN）。** AirfRANS 的 `LinearNO` 已接入实际 `main.py`、`main_evaluation.py` 和 `cdlno_entry.py` 选择路径。默认目标采用已审查的标准化四通道 volume MSE + 1×surface MSE；训练目录/metadata记录完整 profile、objective、数据 checksum、normalizer 数值状态和 provenance。评估只读取已保存 normalizer，严格构造并加载 LinearNO state pair；resume 默认从 `latest`，eval 默认 `final`。两成员 ensemble 的严格 state pair 和 manifest order 也已验证。

验证：`python -m pytest -q tests/linearno` 为 **75 passed**；`python -m pytest -q tests/test_airfrans.py` 为 **13 passed**；两个 Air launcher `bash -n` 和 eval `--dry-run --my_path ... --experiment-dir ...` 通过。模型正式参数量 **3,358,788**。旧 AirfRANS/Transolver 回归通过，旧分支只在新 family 显式选择时绕过；旧数据采样、指标和保存路径逻辑未改。

NOT RUN：真实 manifest/VTK 数据、32k采样训练、400 epoch、真实 force/pressure 指标、远端 Python3.10/Torch2.11/CUDA12.8、AMP/compile/性能。**本 L6 阶段结束，未执行下一阶段。**

## 2026-09-18 — AirfRANS C08 用户决定已落实

**本次协议补充：PASS；L6 接入阶段仍为 PARTIAL，尚未完成。** 用户明确选择新 LinearNO 的 AirfRANS 默认训练沿用官方 Transolver/LinearNO：标准化四通道、体积 MSE＋1×表面 MSE。此前“反归一化后四通道联合 rL2”是助手对论文未公开细节的提议，未被采纳；只有 rL2 与0.5系数来自论文。详见 [协议决定](LINEARNO_AIRFRANS_OBJECTIVE_DECISION.md)。本段取代下文历史记录中 AirfRANS C08 仍待决定的状态，保留所有历史原文。

- `resolve_config('airfrans')` 默认应用 `airfrans_transolver_mse_v1`；三profile都记录完整MSE objective与 `integration_contract` 字段来源。模型/训练预算/评价轴不混改；其他七任务默认配置不变。
- `contract=None` 仅保留原L0来源审计及旧metadata语义；旧paper UNRESOLVED继续拒绝运行，不自动迁移或冒充新决定。L0 catalog/清单不改写。
- 20项检查通过：新协议4＋既有profile/schema11（1.727秒）；旧Transolver四项（含八任务同权重、CLI、checkpoint）＋Air原MSE训练一步/新进程对象加载1项（28.130秒）。未运行真实数据或训练。
- 本轮修改只在独立profile、相应测试、文档/清单。生产模型、八任务入口/训练/eval、数据/依赖/launcher和checkpoint helper均未修改；基线与冻结见 [证据](linearno_audit/air-objective-decision/)。
- L6的main/YAML/CLI及原生train→checkpoint→resume→eval仍未完成，不能因目标已确认而标阶段PASS。Car空间/reduction尚未决定，不把本次Air选择扩展到Car；L7仍BLOCKED，L8–L10未执行。此前连续阶段授权仍有效。

## 2026-09-18 — L7（独立组件完成，工业训练协议仍阻塞）

**阶段状态：BLOCKED。** 用户已明确授权顺序完成L7–L10；本轮推进了不依赖C08的Car模型/评价组件。L6仍未完成生产接线，C08尚未收到具体选择，故不能把默认paper loss变成任意本地解释，也不满足L8要求的两个工业原生闭环。未执行L8–L10。

- [L7报告](LINEARNO_L7_REPORT.md)、[证据](linearno_audit/l7/)。新增`cdlno.linearno.shapenet.ShapeNetLinearNO`及Car本地`models.LinearNO.Model`薄导出；Data.x7、tuple输入、geom不参与、d256/L8/h8/M32/ratio2/out4，两个tempreature参数0.5/clamp[.1,2]。正式参数**3,852,420**，与固定官方完整state和初始化RNG逐位一致。
- 新模型6方法通过24.198s：默认CPU double/FP32、原生CUDA FP32及有效备用unified路径，逐block/final/loss/全梯度/Adam一步/strict checkpoint误差0。独立double oracle输出max1.39e-17、梯度max4.44e-16；真实PyG可变N/单图Batch/拒绝多图/输入不变/geom无效/置换等变通过。原train AST归一化MSE一步、本地可信整模型Car新进程加载通过；不等于原生metadata/resume闭环。
- 新评价4方法通过4.231s：physical rL2与normalized MSE明确分开、mask/通道/分母校验；独立drag适配接收surface速度和真实非零fold路径，真实内存VTK计算与旧公式完全一致。原drag helper不改；尚未接实际eval，未读取真实VTK或造假数据。
- 旧Transolver、Car套件、L6 Air模型和L2 attention的38项针对性回归通过，加新模型6/评价4，合计48个不同方法零失败/错误/跳过。L6已通过的六Standard原生合成闭环保留引用，不虚称本轮重跑全部八任务。无真实训练/下载/依赖安装/commit/push。
- 全部既存模型/生产入口/parser/数据/train/eval/launcher/checkpoint/profile及旧测试不改；L7基线1,392文件快照`/home/hwz/CDLNO-artifacts/linearno-l7-before-q_o1nixy/source`保留，旧源码与L0冻结复核。新代码尚未注册Car CLI。
- 尚未完成：Car生产profile/原生train→checkpoint→resume→eval、preprocessed前置校验/launcher、Air多成员原生闭环。`require_resolved_objective`仍明确拒绝两工业paper未决配置；已再次异步询问C08。远端Torch2.11/cu128、真实数据/收敛/论文精度/性能均NOT RUN。

**L7在C08及前置工业接线处暂停，尚未完成；未执行L8–L10。** 原有自动阶段授权继续有效，缺少的是训练协议决定。

## 2026-09-17 — L6（阻塞点记录）

**阶段状态：BLOCKED。** 当前用户已授权依次完成L6–L10；阶段授权充分，但L0 C08尚未决定AirfRANS paper训练rL2的通道/计算空间/reduction，L6第4项禁止自行选择。已提出缺失协议问题，尚未收到决定；未进入L7–L10。L0–L5历史PASS保留。

- [L6报告](LINEARNO_L6_REPORT.md)、[证据](linearno_audit/l6/)。新增完整`cdlno.linearno.airfrans.AirfRANSLinearNO`和本地`models.LinearNO.Model`薄导出；组合L2纯attention。正式d256/L8/h8/M32/ratio2/out4，7维原输入拼接64维[-2,4]×[-1.5,1.5]参考距离，参数**3,358,788**，所有初始化值/key/RNG与官方一致。
- 新7方法通过12.086s。CPU double/FP32和本机原生CUDA FP32逐block/final/梯度/Adam一步对照误差0；CPU官方参考只适配其硬编码`.cuda()`。独立double oracle输出max4.16e-17、梯度max2.66e-15。真实PyG可变N/输入不改/单图拒绝混图/dead temperature无grad与不影响输出通过；单成员strict state和本地可信整对象新进程往返通过。
- 原train函数AST的normalized四通道volume+surface MSE合成一步通过，**不等于原生生产CLI、resume或ensemble闭环**。旧Transolver4、Air13和原weighted-loss记录片段1通过；L1–L5另48方法通过（九进程wall577.955s），包含六Standard原生合成存档/resume/eval重跑。合计73个不同方法零失败/错误/跳过，另一次文档保留检查也通过。无真实数据、完整训练、外部官方pickle、依赖安装、commit/push。
- 所有AirfRANS既存生产文件和已有测试不改；main/params/eval/profile/checkpoint/launcher接线、paper loss与正式metrics、nmodel>1原生闭环尚未完成。提示词声称既有的两个ensemble命名产物实际不存在，已按源码记录差异，不伪称已有。
- 冻结以本轮1,356文件快照`/home/hwz/CDLNO-artifacts/linearno-l6-before-ryvj5v4k/source`和L0分类/hash核对；仅新增模型/测试/证据及两份状态文档增量。旧Transolver核心不变，tracked diff保持L6开始状态；L0广义冻结的L4/L5既有三项例外单列，未新增ignored。
- NOT RUN：Air原生训练→成员checkpoint→resume→新进程eval、多成员ensemble、完整radius-graph采样（缺torch_cluster）、真实数据/收敛/精度/耗时、远端Python3.10/Torch2.11/cu128。monitor自L0不存在，N/A。模型合成CUDA通过不替代这些项目。

唯一待决定的科学问题见报告C08候选表；不是再次申请阶段许可。**本L阶段在C08处暂停，L6尚未完成；未执行L7–L10。**

## 2026-09-17 — L5

**阶段状态：PASS。** 本轮只接入NS与Plasticity，六Standard现已接入；未执行L6/工业接线。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`。

- [L5报告](LINEARNO_L5_REPORT.md)、[六题两profile命令](../tran_evaluate/linearno/README.md)、[证据](linearno_audit/l5/)。复用原Standard LinearNO模型和factory；NS plain/d256/L8/h8/M32/ratio2/ref10/unified on，参数3,377,921；Plasticity conv/d128/L8/h8/M64/ratio1/H101/W31/Time_Input/out4，参数1,799,428，两profile正式构造均通过。
- 只在两个exp增加显式family guard；时间循环/loss完整保留：NS十步真值回填累加后一次更新、eval十步预测回填；Plasticity二十次独立更新后一次scheduler。L0 C23已确认原collate是逐样本torch.randperm，非提示词中的NumPy；原函数与点序不改，Torch/NumPy/Python RNG全部恢复。时间任务维持batch sum rL2，不套L4静态batch mean。
- 存档复用L4 strict=True/metadata-first/完整状态/normalizer恢复；Plasticity global_step按optimizer计数，是scheduler的20倍；原保存时机保持。没有改checkpoint.py/schema、模型数学、四静态exp或工业生产文件。
- L5新增4静态方法（1.662s）及1原生闭环方法（143.287s）通过。真实空间N4096/3131、B2、d8/L2/h2/M4测试override，每题3合成epoch/6outer batches；NS60forward/6backward及optimizer/scheduler，Plasticity120forward/backward及optimizer、6scheduler。连续与1+2新进程续跑所有权重/optimizer/scheduler/RNG/batch/T序列逐位相同；独立新进程eval预测及完整时间指标一致，normalizer不refit、训练metadata/config不变；两题周期场图completed。14严格负向检查通过。
- 四静态L4 suite全部7方法（189.012s）重跑通过。L1 16、L2 14、L3 10、旧静态2和时间10通过；记录9方法中一个方法的6子案例初因测试查找最外层CDLNO分支失败，仅修为ast.walk后受影响10方法全通过。初轮日志保留。一个既有Air完整采样epoch子案例缺torch_cluster跳过；Car实际PyG合成与Air weighted-loss记录片段通过。monitor从L0起不存在，N/A。
- 冻结核验含L5前1,323文件快照`/home/hwz/CDLNO-artifacts/linearno-l5-before-uh7r5hxj`及L0分类/hash；所有模型数学、旧Transolver核心/attention/embedding不改。两exp旧完整AST一致。L0广义冻结累计只有L4 experiment/test-projection及本轮旧测试AST locator适配，未伪称全部字节相同，无新增ignored。
- NOT RUN：真实loader/数据/训练/精度/收敛/性能、时间任务GPU/AMP/远端Torch2.11/cu128闭环、外部权重转换。当地既有CUDA原子parity回归不替代真实任务验收。

无未解决失败或新架构决策。**本L阶段结束，未执行下一阶段。**

## 2026-09-17 — L4

**阶段状态：PASS。** L3已审查通过；本轮只接入Airfoil、Darcy、Elasticity、Pipe。NS/Plasticity/两工业生产入口未改，未执行L5。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`；无真实数据或训练。

- [L4报告](LINEARNO_L4_REPORT.md)、[四题两profile训练/评估/续跑命令](../tran_evaluate/linearno/README.md)、[证据](linearno_audit/l4/)。新factory key为`LinearNO_Structured_Mesh_2D`/`LinearNO_Irregular_Mesh`，共同返回已接受的`model.LinearNO.Model`。旧key/default/save_name/strict=False路径保留；新family用独立variant/rank，strict=True。
- 四真实exp仅新增明确guard的接入；新增`standard_entry.py`/`checkpoint.py`提供metadata-first构造、保存normalizer数值、数据checksum、完整optimizer/scheduler/RNG存档和epoch边界resume，复用hxh实验记录/可视化与atomic/RNG原语。eval不重新fit、不重写训练config/sidecar。原保存时机ep%100==0+final保持。
- 正式mapping：Airfoil conv_temp/M64/221×51；Darcy conv_temp/M64/85×85/unified off，decode/0.1导数/zero-padding/dx1/85/仅导数预测边界置零；Elasticity temp/M64/fx=None；Pipe conv_temp/M64/ratio1/batch4/129×129正方形。
- 最新L4明确要求batch mean rL2，无epsilon；两个接入profile均记录`standard_static_l4`合同，与发布batch sum梯度缩放差异明确，不冒充release-exact。L1无contract的事实配置保持。Airfoil原每epoch验证、Pipe原first1200切分保持。Darcy official_release拒绝非500epochs；paper使用resolved epochs。OneCycle epochs/steps_per_epoch/total_steps均保存。
- L4集成7方法最终通过（193.116s），4题真实N的原main AST各完成3个合成epoch/6更新、连续与1+2新进程续跑及独立eval。weights/optimizer/scheduler/RNG/batch顺序逐位一致，eval预测hash一致；normalizer禁止refit断言、训练文件hash保持；四题final PeriodicFields图均completed。d8/h2/L2/M4仅合成测试override，不是正式宽度性能或真实数据。
- 28项针对真实新存档的负向检查通过（profile/variant/rank/family/缺键/额外键/shape）。L1全部16、L2全部14、L3全部10回归通过；另旧静态2、记录9执行，1个测试投影失败修正后最终静态6方法重跑全部通过（15.309s）。保留初轮失败与修正日志；1个旧Air完整采样epoch子案例缺torch_cluster跳过，Car实际PyG合成epoch及Air weighted-loss片段通过。
- 16新launcher预览通过，原81个shell字节不改且bash -n通过。四exp/parser/factory/experiment完整AST去除新guard后与L4前完全一致。L0原Transolver核心/attention/embedding字节不改；广义freeze中仅允许的experiment和测试投影helper更新，详见delivery-check，不将授权接线伪装成整个manifest字节不变。无新增ignored；快照`/home/hwz/CDLNO-artifacts/linearno-l4-before-ti1ycp32`及新闭环存档保留。
- NOT RUN：真实loader/数据完整性/精度/收敛/500epoch/实际耗时、四任务GPU/AMP/远端Torch2.11/cu128闭环、官方外部权重转换；本机L2/L3合成CUDA回归不能替代四任务真实GPU验收。

无未解决parity失败或新增架构决策；工业未决协议仍保留。**本L阶段结束，未执行下一阶段。**

## 2026-09-17 — L3

**阶段状态：PASS。** L2已审查通过，本轮仅组装六Standard共用的完整LinearNO Model并完成独立验证；未改factory、六exp、生产launcher/训练/评估/checkpoint helper，未执行L4。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`；CUDA只做明确授权的合成官方parity及L2回归。

- 交付：[L3报告/公式→类→测试映射/参数表](LINEARNO_L3_REPORT.md)、[完整证据](linearno_audit/l3/)。真实稳定类为`PDE-Solving-StandardBenchmark/model/LinearNO.py`的`model.LinearNO.Model`；组合原L2 attention，公开`Model(x,fx,T=None)`。旧registry尚未注册linearno，不提供生产训练命令。
- 保留preprocess、可选time_fc、n_layers完整block和末层LN/head；独立linearno_variant/linearno_rank，不读取args.model。unified位置替换x，非持久buffer支持设备迁移；官方位置值和state_dict key集合一致。顺序严格为默认构造→一次全模型apply→最后placeholder，所有初始化值/RNG逐位一致。
- 正式六配置真实参数量：Darcy1,766,145；Elasticity585,217；Airfoil/Pipe各1,765,889；NS3,377,921；Plasticity1,799,428。固定官方模型先测量，目标key/shape/参数值/闭式计数均一致；没有用缩小模型冒充正式参数量。
- L3新增10方法通过（9.594s）：6正式构造、8初始化、18完整官方parity、8独立double oracle子案例；小模型CPU与原生CUDA unified路径、逐block/输出/loss/输入和所有参数梯度/AdamW一步/strict checkpoint通过。CUDA最终输出max8.38e-9、参数grad2.98e-8、step1.42e-7；double独立oracle所有比较max1.14e-13。阈值未放宽，完整max/mean/relative结果保留。
- 第八层实际执行attention+FFN+LN/head；四变体各8次完整attention/FFN，无N×N或slice SA；合法M×dh context不误判。覆盖B1/2、fx/T有无、unified on/off、非方形3×5/5×7、点式可变N、batch隔离、storage独立、输入不改、第二次backward无缓存、真实cwd导入。
- 重跑L1全部16方法（45.859s）、L2全部14方法（8.600s）及既存存档/输出/冻结28方法（68.302s）。合计68方法0失败/错误；1个旧Air完整采样epoch子案例因torch_cluster缺失跳过，Car真实PyG合成epoch和Air原weighted-loss记录片段通过。原monitor不存在，N/A。
- L3起点1252文件快照`/home/hwz/CDLNO-artifacts/linearno-l3-before-7d5dtz83`；仅本STATUS增量，其余1251文件不变。L0全部1161原文件、221重点冻结文件分类/size/hash均不变；L1/L2新源码不变，无新增ignored。没有依赖/数据/旧模型改动、下载、commit/push或真实训练。
- NOT RUN：六正式尺寸完整forward/训练、真实数据/loader/指标/精度/耗时、生产接线、完整Model AMP/compile、远端Py3.10/Torch2.11/cu128、工业LinearNO完整模型/官方pickle转换。本机Py3.13.9/Torch2.13+cu130/RTX5090 Laptop不能替代远端验收。double+T设备/dtype扩展由独立公式验证，不冒充原官方double+T原生通过。

初始化、位置替换、第八层、strict keys与旧冻结五点已自审，无未解释parity失败。L0工业协议未决项保留至对应后续阶段。**本L阶段结束，未执行下一阶段。**

## 2026-09-17 — L2

**阶段状态：PASS。** L1已审查通过，本轮仅完成纯LinearNO attention原语与六变体包装、独立oracle及验收；没有八层完整模型/任务wrapper、factory/CLI/训练/eval接线，没有执行L3。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`；GPU仅按本轮L2第6项做合成attention FP32/AMP检查。

- 交付：[L2报告及公式→类→测试映射](LINEARNO_L2_REPORT.md)、[逐参数误差/回归/冻结证据](linearno_audit/l2/)。新共有原语`cdlno/linearno/attention.py`，三benchmark各新增`LinearNO_Attention.py`；原模型、L1配置/schema、parser/factory/训练/eval/checkpoint、launcher及已有测试不变。
- Standard plain/temp/conv/conv_temp、AirfRANS、ShapeNet六种attention独立包装。单输入投影，Q/K/V跨head共享小Linear；Q沿M/K沿N，KᵀV后Q乘，无scale/注意力dropout/slice normalization/SA。conv单套denseConv2d；Air保留dead temperature，Car保留tempreature_q/k与[.1,2]clamp。默认forward只返回Tensor，无常驻Q/K/C或CPU同步。
- L2新14方法通过（8.980s）。72数值记录：6初始化、24CPU oracle、42官方同权重parity。CPU double/FP32及CUDA FP32/FP16/BF16已实际执行；所有官方forward/input&parameter梯度/一步AdamW/strict checkpoint对比误差0。double oracle最大abs8.88e-16，FP32最大abs7.15e-7；阈值未放宽，完整max/mean/relative逐参数记录在parity.json。
- 首轮发现torch2.13与官方timm截断正态初始化RNG算法不同；仅修新callback为已有timm函数。修复后六类构造与外层apply两处state/RNG均逐位一致。原子构造不提前apply；后续完整模型必须外层仅初始化一次，placeholder时序留待L3，不冒充已验收。
- L1全部16方法重跑通过（47.961s），八旧Transolver同权重/小checkpoint/随机流不变。另重跑L1引用的旧基础和源码28方法（66.440s），0失败/错误，1个Air完整采样epoch子案例因缺torch_cluster跳过。三原子项目cwd新进程导入/strict state往返通过。
- L2起点1223文件快照`/home/hwz/CDLNO-artifacts/linearno-l2-before-2cz69qfr`；仅STATUS增量，其余1222不变；L0原1161文件及重点221冻结文件hash不变，无新增ignored产物。未改其他模型/数据/依赖，未安装、下载、commit/push或真实训练。
- NOT RUN：完整LinearNO任务模型/论文参数量/完整网络parity、真实loader/数据/指标/训练/精度、AMP scaler/实际epoch性能、官方pickle转换、远端Python3.10/Torch2.11/cu128。当前本机Py3.13.9/Torch2.13+cu130/RTX5090 Laptop，不能替代远端验收。不存在原monitor，N/A。

独立oracle、固定源码、执行shape trace及冻结已自审。无未解释parity失败；L0工业objective/force/部分数据协议待定仍留给对应后续阶段，没有在本轮自行作出选择。

**本L阶段结束，未执行下一阶段。**

## 2026-09-17 — L1

**阶段状态：PASS。** L0已审查通过，本轮仅完成L1：独立LinearNO schema/profile解析，以及现有Transolver与用户产物/随机流增强的修改前回归。未实现LinearNO算子，未接入任何生产parser/factory/训练/eval/checkpoint helper，未执行L2。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`。

- 交付：[L1报告](LINEARNO_L1_REPORT.md)、[证据](linearno_audit/l1/)、`cdlno/linearno/{profiles,schema,_profile_data}.py`、`tests/linearno/`。默认`paper_table8_on_release_model`，另有`official_release/transolver_matched`，8×3配置已逐项对照L0。CLI解析器完全独立；字段来源可追踪，旧默认不覆盖新profile。
- schema分离model/profile/data/objective/evaluation/provenance/normalizer/resume/ensemble；numerical状态可逆、hash严格，旧无family仍归旧路径。构造器完整signature验证需后续loader传入真实类；本轮没有假LinearNO类、模型成功加载声明或外部pickle转换。
- 新16测试方法通过（46.241s）；最终自审后受影响11配置/schema方法重跑通过。8个原Transolver CPU小夹具同权重strict roundtrip误差0，真实PyG Data、Car本地整对象/Air列表往返；10个旧parser AST默认及隔离检查。原`.cuda()`位置仅测试子进程临时适配CPU，非原生CPU/GPU验收。
- 续跑/观察：OneCycle与Cosine均8个合成更新，连续与4+4新进程续跑、启停真实绘图与保存，batch/参数/optimizer/scheduler/RNG全部CPU逐位一致。Plasticity实际Torch逐样本时间排列，Air实际Python抽样表达式，另测NumPy流；不是完整任务训练或任务原生resume。
- 既存CPU存档/产物22方法重跑（67.785s），0失败/错误，1个Air完整采样epoch子案例因torch_cluster缺失跳过；Car实际合成epoch、Air原weighted-loss/记录片段通过。另旧入口AST/文件6方法通过（0.514s）。原monitor不存在，N/A。
- L1起点快照`/home/hwz/CDLNO-artifacts/linearno-l1-before-znfjyu2s`，1192文件：只对本STATUS增量，其他1191不变；L0重点冻结221及既存1161文件不变。用户PLAN/L0证据/旧生产文件/已有测试保留，详见delivery-check。
- NOT RUN：LinearNO官方parity/模型重建（未实现）、GPU/AMP、真实数据/mini-run/精度、远端Torch2.11/cu128、Air完整图采样（缺依赖）、八任务完整resume（当前未接线）。没有安装、生产改动、commit/push或真实训练。

L0 C08工业paper目标定义等未决项继续保留；schema显式列UNRESOLVED，并拒绝声称已解析成可训练objective。Car官方备用位置描述在新schema按二维replacement记录；未动旧Transolver三维位置实现。自审及转换清单见报告。

**本L阶段结束，未执行下一阶段。**

## 2026-09-17 — L0

**阶段状态：PASS。** 判定仅针对本轮只读审计：三方固定来源/论文v3可访问，清单与公式/任务/变体/profile/冲突映射已完成，源码检查通过，没有生产改动。此状态不表示LinearNO已实现、官方数值parity通过或论文结果复现；工业paper训练细节仍为显式未决，必须在受影响接入阶段前审查。

**授权范围**：仅L0；`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`。L1—L10均未执行。

### A. 完成范围与产物

- [REFERENCE_AUDIT](LINEARNO_REFERENCE_AUDIT.md)：实际仓库/固定来源、全清单阅读口径、原Transolver比较、现有模型注册、八任务输入/loss/循环/产物、副作用、复用/冻结、环境、实际命令及阶段预测。
- [REPRODUCTION_MATRIX](LINEARNO_REPRODUCTION_MATRIX.md)：全部编号公式、Table8字段、七变体、8×3profile记录、两工业objective/evaluation正交合同、30项冲突、发布真实argv与未来接线边界、checkpoint协议建议。
- [L0只读证据](linearno_audit/l0/)：基线/全部文件manifest、重点freeze、三树reading/AST/CLI清单、普通双树diff、paper公式、配置/profile、已有回归索引、环境、实际检查日志及交付核验。

L0清单与报告为新增文件；既存报告、PLAN、AGENTS、memory、README未改。目标源码仍为 `main@bb73b3099d3b8ce45bd939156b737453b9ca5454`，remote `hxh5159/CDLNO-w`，不是提示词的旧`66bc489`目标。外部快照 `/home/hwz/CDLNO-artifacts/linearno-l0-before-65a2wvms` 保留。

### B. 已执行验证

| 检查 | 结果/边界 |
|---|---|
| 全文件阅读/结构覆盖 | target910、thuml71、LinearNO61 tracked；可读文本分别885/885、59/59、61/61，完全相同文件按SHA分组；非数学/执行通过率 |
| 固定thuml→实际target | 55相同blob、16改动、839仅target、0原文件缺失；13原模型/attention/embedding文件字节一致 |
| Python AST/shell语法 | 384/384成功；未执行launcher或任务顶层代码 |
| 既存旧源码冻结方法 | 6/6通过，unittest0.483秒（进程3.944秒）；八任务入口/训练/factory完整AST投影及文件哈希 |
| 官方六Standard shell真实parser解析 | 6/6通过；全部eval1；逐项保留parser默认与shell覆盖来源 |
| L0文件保护 | 既存910 tracked+2 untracked+249 ignored逐字节保护；重点冻结221文件；最终结果见delivery-check.json |
| Python/torch/PyG/GPU | 仅只读版本/import/设备查询；Py3.13.9/Torch2.13+cu130/PyG2.3.1，RTX5090 Laptop×1；缺torch_cluster/scatter/pyg-lib；未安装依赖 |
| 官方→移植数值parity/新模型构造/新checkpoint | NOT RUN：L0未实现；不把既有历史夹具通过当本轮新结果 |
| GPU模型/AMP/工业全采样/真实数据/mini-run/长训练 | NOT RUN：未授权、无数据；远端Py3.10/Torch2.11/cu128也未验证 |

### C. 已决定的复现边界

默认`paper_table8_on_release_model`，另有可执行`official_release`的后续实现目标；保留release专属forward、逐block卷积、跨head共享投影、温度/初始化/key。模型family拟`linearno`；不放入冻结`LINEARNO/**`，也不借用CDLNO/KCDNO/MSAR机制。`transolver_matched`只作可选超参对照，和已有`lrsa_matched`不同。

训练/评价指标不得混合：Air release为normalized MSE，paper表2为physical pressure rL2；Car正式release eval已有physical rL2及RMSE，训练则是全点velocity MSE+0.5surface pressure。checkpoint最终角色/三seed全部报告，不用test挑最好权重或展示工具挑最好seed。

### D. 自审与仍需审查事项

已自审原始模型字节、旧入口AST、Q/K与M语义、工业mask/force路径、阶段边界；检查证据在审计报告。保留以下实际待定/计划修正，不把已通过项目伪装成待用户解决的问题：

1. **C08：工业paper训练rL2**的通道聚合和normalized/physical空间尚无唯一论文定义；报告列出两种选择及影响，L6/L7前需明确。不会静默只把MSE权重改掉。
2. **C12：Car force路径**：surface velocity已正确，helper仍旧root/param0。建议后续新family adapter接受实际样本路径；旧helper不动。是否保留发布bug仅作明确诊断，必须与正常指标分开。
3. **C22/C23：阶段提示词需按当前仓库解释**：无既存LINEARNO/monitor、无任务级完整resume；Plasticity时间排列为Torch而非NumPy。L1只做独立schema和已有基础可执行测试；任务resume在L4—L7新增接线，不宣称旧入口原生支持。
4. **C25—C27：release与current任务协议小差异**：Airfoil每10epoch/最后测试且B1、Pipe去掉[:1200]前截断、Air训练LinearNO免radius_graph。已列两种处理和推荐，后续仅新family显式实现，不全局修改旧任务。

L0前旧文件和全部dirty状态保留；没有生产源码/数据/依赖/launcher/已有测试变更，没有commit/push/真实训练。**本L阶段结束，未执行下一阶段。**
