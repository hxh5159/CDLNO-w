# CDLNO 实施状态

## 2026-09-15：临时插入的 Darcy 消融启动脚本请求完成

新增 `tran_evaluate/ablation/no_sa/` 与 `identity/`，各含 train/eval/train_eval 三个薄包装。只向已有 Darcy 启动器传入对应 front_latent_mode，余参最后覆盖；F2/L8/P6、entry CDPA、Darcy 模型/训练预设与 `output/darcy/<timestamp>` 规则保持。组合脚本复用现有训练成功才评估及同目录逻辑；独立 eval 须指定已有目录。新增说明与 [实际核查结果](../tran_evaluate/ablation/verification.json)。

验证：六个脚本 `bash -n` 通过；10次 `--dry-run` 产生16条命令，经原 Darcy parser AST及安全 `cdlno_entry.parse_args` 校验通过（eval parser仅使用临时架构sidecar）；两次缺少eval目录负向检查通过。外部含空格cwd、用户余参覆盖、两步同目录均核对。没有导入exp或运行模型/数据/训练/GPU；本次不作新的权重加载或模型验收声明。现有模型、配置、数据、训练/评估入口、依赖及旧启动器未改；既有K0文档与夹具保留，不推进K1。新增6个shell及README/核查JSON，现有文档仅增量记录。

## 2026-09-15：统一实验目录与记录完成，待审查

用户批准的新范围已完成：八任务CDLNO默认 `output/<dataset>/<UTC timestamp>`，有效参数解析后/数据加载前预留目录写config/log；真实model后补total/trainable/architecture/adapter/optimizer和实际协议。训练history/results、每次eval独立子目录+根索引；old runs显式加载、sidecar/config不改写，strict和原checkpoint格式不变。全模式适用，模型/数据/优化/时间/指标计算冻结。见 [报告](CDLNO_EXPERIMENT_OUTPUTS_REPORT.md)、[命令](CDLNO_EXPERIMENT_OUTPUTS.md)、[证据](output_audit/)。

Final212tests/142.652s，0fail/error，1Air sampled-epoch子项因本机torch_cluster缺失skip；新9test methods/12.872s，24任务模式合成记录/step/checkpoint+复评、三原cwd新进程、Car完整合成epoch有无记录权重/RNG精确、Air原weighted loss/梯度+真实记录AST片段通过。既有实际GPU套件重跑通过；未运行真实数据/远端目标环境/完整Air抽样评价。86已有核心/model/preset/data/utils/script/tool文件hash相同，12entry/train完整AST仅移除精确记录语句后与本轮312文件快照相同：`/home/hwz/CDLNO-artifacts/output-before-kql2cnpe`。先前未提交A1–A4/V1等修改保留。

原模型checkpoint保存频率/格式保留；本轮不是V2–V5任务resume/周期可视化接入，不能宣称已有任务权重能完整恢复optimizer/RNG。无commit/push/真实训练。阶段结束，未执行下一阶段。旧状态如下保留为历史。

更新日期：2026-09-15。

## 最新：A3已批准，A4计数/有限性能/文档交付完成，待审查

本轮只执行补充A4；[报告](CDLNO_FRONT_ABLATION_A4.md)、[命令](CDLNO_FRONT_ABLATION_A2_COMMANDS.md)、[本轮diff](front_ablation_audit/a4/a4-changes.patch)、[累计前段消融运行diff](front_ablation_audit/a4/front-ablation-runtime.patch)。full仍默认，八任务已接入三front模式。性能工具现按实际SA/FFN调用统计，matched LRSA锁full；模型/数据/训练入口/预设/脚本/依赖与V1组件未改，不推进V2–V5。

| 补充阶段 | 当前状态 |
|---|---|
| A0/A1/A2/A3 | 用户已审查通过 |
| A4 | 计数、完整成本、有限GPU实测与三模式命令/兼容交付完成，待审查 |
| 另行V1 | 归档/绘图公共组件完成，原范围保持 |
| 另行V2–V5 | 尚未执行；正式任务还未接入新resume/周期图 |

默认2+6实测full/no_sa/identity的latentSA为8/6/6，前段FFN4/4/0；后段SA/GEGLU各6、Down/Up/规则ConvFFN各3；entry2历史/1调用，every27/6。13/13定向通过（14.496s），最终**203/203通过（126.687s，无失败/错误/跳过）**。33行CPU完整成本与chunk0/1/2等价、10行有限GPU性能、24个正式模型参数统计、24组顺序预览/48次真实parser均通过。修改前真实full性能基准五模型的初始化hash/参数/成本/原计数精确保持。

本机RTX5090 Laptop/Python3.13.9/torch2.13cu130。两组有限GPU为原Elasticity N972/B1及缩小Airfoil17×23/B2，统一FP32/mathSDPA/TF32关/compile关，warmup5/measure20、同步/median/p90/峰值显存/已初始化AdamW。固定entry三mode的结果详见报告；不能由MAC/参数降幅推定延迟或真实epoch效率，不能证明CDPA替代前段子层，后段SA仍在。没有真实训练、远端torch2.11cu128或新模式全GPU/AMP/compile性能矩阵。

284份本轮起点保留于`/home/hwz/CDLNO-artifacts/front-a4-before-ftvcu781/source`；128份[冻结文件](front_ablation_audit/a4/freeze.json)不变，40份shell语法通过；累计pre-A1原CDPA/core历史/rear/decoder及数据训练协议AST/hash也通过。实际Air main使用MSE_weighted，前A3默认MSE测试不升级为正式weighted三模式链路验收；现存旧Car/Air问题继续分开报告。无commit/push/下载/依赖改动或真实训练。**本补充阶段结束，未执行下一阶段。** 以下为历史状态。

## 最新：可视化/续训计划已批准，V1公共组件完成，待审查

本轮按逐阶段约束只执行V1，详见[V1报告](CDLNO_VISUALIZATION_RESUME_V1.md)和[增量diff](viz_resume_audit/v1/v1-changes.patch)。新增完整checkpoint/配对纯权重、严格恢复、RNG隔离、epoch周期和输出场绘图；**八任务训练/评估/脚本尚未接入本次新功能，不能使用新的`--resume`。** V2–V5及A4未执行。

| 阶段 | 范围 | 当前状态 |
|---|---|---|
| V0 | 论文可视化/源码审查与计划 | 用户已批准实施计划 |
| V1 | 完整归档/恢复/周期/RNG隔离/绘图公共组件 | 完成，待审查 |
| V2 | Darcy/Elasticity/Airfoil/Pipe | 未执行 |
| V3 | NS/Plasticity时间任务 | 未执行 |
| V4 | Car/AirfRANS工业/成员恢复 | 未执行 |
| V5 | 八任务综合验收/使用交付 | 未执行 |

本轮最终201/201回归通过（122.573s，无失败/错误/跳过）；新增14项归档/观察器+3项绘图检查。CPU三front模式×两scheduler共6组跨新进程恢复与连续训练逐位相等，包括权重、AdamW矩/step、scheduler、每步loss/lr、采样及RNG。有限本机GPU identity/OneCycle FP32/mathSDPA/TF32关闭对照通过；不是远端torch2.11/cu128验收。新的恢复对照采用B2/5×7/d8/h2/M4小核心及明确标注的合成MSE，不是NS原时间训练链路。原损失/PyG旧回归仍通过，但新八任务续训/周期图尚未验证。

实际验证两份文件后才发布manifest/latest；补强并测试manifest已提交而latest失败时的防回退；错误模式/协议/权重/优化器/调度器/RNG、旧纯权重及跨run拒绝。真实推理/绘图开关不改训练权重/梯度/随机状态，异常渲染仍保存到期checkpoint。纯权重只能在配套完整归档存在并通过摘要校验时通过本接口用于续训。

270份起点snapshot保留于`/home/hwz/CDLNO-artifacts/viz-v1-before-04ww1akt/source`；[冻结证据](viz_resume_audit/v1/freeze.json)确认128份原生产/任务/脚本/工具文件字节不变，包括模型结构、数据、损失、训练评估与依赖。未下载数据、安装依赖、真实训练或commit/push。新功能的真实数据/原时间采样协议/工业整对象及list续训、AMP、远端文件系统断电恢复、收敛/精度/epoch效率未验证。默认final-only四任务仍无中途恢复点。**本阶段结束，未执行下一阶段。** 以下审批前及A3状态保留为历史。

## 最新请求：八任务可视化/周期保存/断点续训，计划待审批

用户要求先阅读相关论文并提交修改计划，再由用户决定是否修改。本轮仅完成 [可视化与续训计划](CDLNO_VISUALIZATION_RESUME_PLAN.md)、[五篇论文来源/图示记录](viz_resume_audit/paper_sources.json)、源码静态核对和状态更新。**没有实现新的checkpoint/resume/绘图功能，没有新增模型验收结果。** A4和性能工具阶段未执行；后续V1–V5每阶段均需用户明确指定。

计划：八任务每50个已完成epoch可视化；NS/Car/AirfRANS/Airfoil每100轮及结束保存完整续训checkpoint+独立权重，其他四任务仅实际结束保存。当前PDE500/Car200/Air398默认不改；建议最终补图，固定2个案例。保存optimizer/scheduler/随机数/normalizer/数据顺序及Air成员进度，epoch边界恢复；保留原最终eval格式。只有最终保存的四任务没有中途恢复点。模型/训练协议不变，仅拟增加运行管理与epoch边界回调，尚未获实施授权。

本轮源码发现并更正历史表述：AirfRANS实际`main.py`显式选择MSE_weighted（volume+weight×surface），并非train函数默认MSE。A3的新三模式合成检查跑过函数默认MSE，旧测试另有weighted分支，但不能将其描述为完整三模式正式入口weighted验收；后续新测试应按实际调用补足。Plasticity原x_normalizer实际作用fx，random_collate_fn随机排列时间；Car原目录顺序及geom抽样、Air原反复抽样均要纳入续训协议。没有因此改loss或数据读取。

阅读范围是Transolver/Transolver++/Transolver-3/LRSA/LinearNO相关可视化正文/附录/图注与选定图片；未声称本轮重新完整数学审查。计划前源码hash及git状态见 [planning_start.json](viz_resume_audit/planning_start.json)。以下A3等报告按当时范围保留；最新更正与未实现状态以本节及计划为准。

## 最新：补充 A3 验证完成，待审查；A4 未执行

A2已由用户审查通过。本轮按最新A3指令完成八任务×三种front模式的原接口、合成优化器更新、原损失连接、eval/checkpoint验证；**不是旧建议中的性能工具阶段**。只新增/调整测试和审查文档，没有修改生产模型、配置、训练入口/数据协议、脚本、依赖或性能工具。253份起点源码保存于 `/home/hwz/CDLNO-artifacts/front-a3-before-l9s4pe2c/source`；128份生产/任务/脚本/工具文件hash不变，用户与A1/A2已有修改完整保留。

完整回归 **184/184通过，75.836s，0失败/错误/跳过**。新增43项中：24个CPU任务×模式格，6个Airfoil/Pipe的5×7格，12个Darcy/NS/Plasticity/Car的GPU格，以及1项禁止import实验入口检查。所有42个计算格实际完成优化器更新、原损失、eval及checkpoint；只修正结果JSON的真实N标签后定向重跑43/43通过（13.469s）。新checkpoint改变chunk后的最大绝对输出误差6.147e-8，容差沿用atol1e-5/rtol3e-4。真正pre-A1的12+2+2份full基准（含整对象/list）再次原零容差通过。

六标准任务以真实PDE N和小d8/h2/M4测试；NS B2保持10输入/10步真值回填训练、1次backward/optimizer/scheduler及10步预测回填eval；Plasticity B2/N3131/T[B,1]，20次独立forward/backward/update与1次scheduler，时间梯度通过。真实PyG工业原train/test、mask/默认MSE、变量N、多图拒绝、标签独立、输入不变、Air抽样后ptr合同通过；工业N32186/32000仅单独layout forward。三原cwd新进程导入/同模式加载、显式错误模式与坏权重拒绝、eval不覆盖sidecar、旧full兼容通过。CDPA off/entry/every和三front模式有限组合、真实T与梯度/历史隔离复用原核心回归并通过。

| 阶段 | 最新状态 |
|---|---|
| A0/A1/A2 | 用户已审查通过 |
| A3 | 八任务三模式合成训练/eval/checkpoint验证完成，待审查 |
| A4 | 未执行，等待明确指示 |
| 性能工具新模式支持 | 本轮未授权实施，保持冻结，不按旧阶段建议自动推进 |

本地Python3.13.9/torch2.13+cu130/PyG2.3.1/RTX5090 Laptop。GPU新增12格为FP32/math SDPA、TF32关；不是全八任务GPU/AMP或远端torch2.11/cu128验收。真实数据完整读取/抽样/全物理指标、训练/收敛/准确率/真实epoch效率全部未执行。原Car日志字段交换/drag路径fold限制、Air旧MAE条件缺陷仍属原有问题，本轮无生产修复。

交付：[A3报告与八任务覆盖表](CDLNO_FRONT_ABLATION_A3.md)、[逐格结果](front_ablation_audit/a3/training-cases.json)、[实际原batch片段](front_ablation_audit/a3/executed-standard-fragments.json)、[增量diff](front_ablation_audit/a3/a3-changes.patch)、[冻结证据](front_ablation_audit/a3/freeze.json)。自审未发现本轮新增模型/训练协议回归；没有commit/push/下载/真实训练。**本补充阶段结束，未执行下一阶段。** 以下状态保留为历史，以本节为准。

## 最新：补充 A2 完成，待审查；A3/A4 未执行

A1已由用户审查通过。本轮A2完成八任务 `front_latent_mode` 的CLI/JSON→wrapper/core传递、eval先读sidecar恢复未指定mode、显式冲突拒绝和工业整对象/list的实际mode检查。默认full、旧full路径、所有任务模型/训练预设不变；缺少其它架构字段拒绝补默认，仅已识别的旧完整full允许缺mode。root任务脚本消融默认目录追加模式，显式run仍优先且新训练拒绝已有目录。共享数学、数据/训练/评价循环、原模型、依赖及性能工具未改。

最终 **141/141回归通过，62.591s，0失败/错误/跳过**。含8项A2检查、24任务模式实例、真实PyG工业接口及三个原cwd新进程保存/加载。48条真实parser root命令及24组顺序命令预览通过；12+2+2真正修改前full基准再次零容差通过。首次测试脚本常量/后端及3项旧参数捕获占位目录问题如实记录并修正，没有关闭实际加载校验。新增配置完整对象负向项包含在最终套件。

本地Python3.13.9/torch2.13+cu130，已有PyG/GPU；套件既有full GPU检查通过，A2新模式八任务GPU矩阵未运行。远端torch2.11/cu128、真实数据读取/轨迹/训练/收敛/精度/epoch效率均未验证。没有下载数据、重装依赖或commit/push。

| 阶段 | 当前状态 |
|---|---|
| A0/A1 | 已由用户审查通过 |
| A2 | 八任务三模式接口、配置/checkpoint、必要目录及命令完成，待审查 |
| A3 | 未执行；性能工具新模式/成本统计等仍待用户明确指定 |
| A4 | 未执行；补充最终综合交付仍待授权 |

交付：[A2报告及任务×模式覆盖表](CDLNO_FRONT_ABLATION_A2.md)、[八任务训练/评估命令](CDLNO_FRONT_ABLATION_A2_COMMANDS.md)、[本轮diff](front_ablation_audit/a2/a2-changes.patch)、[冻结证据](front_ablation_audit/a2/freeze.json)。已完成重点自审，A2无剩余已知实现缺陷。**本补充阶段结束，未执行下一阶段。** 下方旧状态保留为历史，以本节为准。

## 最新：补充 A1 完成，待审查；A2–A4 未执行

用户明确恢复A1后，仅修改共享 `cdlno/config.py`、`modules.py`、`core.py` 及针对性测试/说明。新增架构字段 `front_latent_mode`，默认full；no_sa只移除SA完整残差子层及其norm，identity直接T=S且不注册latent FFN/SA/norm。保留Down/Up/点FFN、bridge/rear/readout、CDPA两级融合及原历史时序。配置规则、旧full兼容和完整计算图见 [A1报告](CDLNO_FRONT_ABLATION_A1.md)。

修改前真实基准保存在 `/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0`：12个front/core、2个Car、2个Air旧对象及list重放均以原atol=rtol=0通过键/权重/初始化/输出/梯度检查。Car早期捕获的import失败已记录，并在生产编辑前修正独立进程PYTHONPATH后成功捕获；未重造已完成基准。已知旧JSON和18槽配置pickle按历史full恢复；sidecar只读校验，显式新模式/错误权重拒绝，未使用strict=False。

最终 **133/133测试通过，0失败/错误/跳过**，含13项新检查、36个三前段模式×深度×CDPA配置，F0同权重精确等价及旧完整边界；本地GPU三新模式FP32 parity通过。首次综合测试的2项旧冻结清单错误已保留日志并仅在测试中修正，未改任务实现。实际环境为本地Python3.13.9/torch2.13+cu130/RTX5090 Laptop；未替换依赖，未运行用户远端Python3.10/torch2.11/cu128的新模式或新模式AMP。

三任务项目、共享wrapper、CDPA/checkpoint、任务配置/所有启动shell和依赖均与恢复A1时一致；八任务数据/训练/评估协议保持，新模式尚未接入其CLI。完整证据、初次/最终日志及可审查源码patch在 [a1审查目录](front_ablation_audit/a1/)。没有真实数据下载/训练/评价、commit/push或后续阶段。

| 补充阶段 | 当前状态 |
|---|---|
| A0 | 已由用户审查通过，原审查全文保留 |
| A1 | 共享配置、三模式前段、核心及兼容回归完成，待用户审查 |
| A2 | 未执行：八任务参数/运行及新模式checkpoint接入 |
| A3 | 未执行：新模式启动配置/性能统计等后续范围仍以用户明确指令为准 |
| A4 | 未执行：补充最终综合回归/交付仍需明确授权 |

已自行复核full基准、整子层删除、T/history梯度、strict兼容及冻结证据，A1范围无剩余已知架构缺陷。**本补充阶段结束，未执行下一阶段。** 以下暂停/A0及早期状态保留为历史记录，以本节为当前状态。

## 最新：A1暂停；八任务训练后评估组合脚本完成

用户已审查通过A0，随后授权A1，但在生产实现修改前明确暂停，另行要求训练/评估顺序脚本。当前A1未完成，A2–A4未执行。

新增 `tran_evaluate/train_eval.sh TASK`，顺序复用原任务训练和评估脚本；同配置同run，训练失败不评估，未设置tag时生成一对命令共用的新tag。八任务默认+覆盖共32条实际parser命令及纯shell顺序/失败/引用检查通过；没有真实训练/评价。只改组合脚本及文档记录，原任务脚本、模型/配置/入口/数据/损失/时间循环/依赖均未改。Car完整阻力评价的固定raw路径/param0/fold0限制保留，AirfRANS两步my_path语义保留。见 [顺序启动报告](CDLNO_SEQUENTIAL_LAUNCH.md)和 [命令](../tran_evaluate/README.md)。

A1暂停时，原源码/基准存于 `/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0`。12个front/core、2个Air旧对象/list基准确定性重复通过；Car捕获因独立进程缺少models搜索路径失败，待恢复A1后先处理，未伪报通过。尚未增加front_latent_mode字段或三种前段模式；不能把本轮脚本通过当作A1模型验收。旧阶段记录保留如下。

## 补充 A0：前段 latent processor 只读审查（最新）

**A0 完成，待用户审查；A1–A4 未授权、未执行。** 以下旧阶段记录原文保留，涉及“尚未指定A阶段”的历史描述由本段更新。详细交付：[CDLNO_FRONT_ABLATION.md](CDLNO_FRONT_ABLATION.md)；起点/静态证据：[front_ablation_audit](front_ablation_audit/)。

| 补充阶段 | 执行状态 | 范围 |
|---|---|---|
| A0 | 完成，待审查 | 源码/参考开关、三模式差异、八入口/保存协议、兼容及修改前基准方案；仅文档 |
| A1 | 未授权、未执行 | 建议先捕获真实旧full基准，再实施核心/配置兼容；以用户后续指令为准 |
| A2 | 未授权、未执行 | 建议八任务mode参数/checkpoint接入；不改训练数据协议 |
| A3 | 未授权、未执行 | 建议脚本/目录及合成性能计数接入 |
| A4 | 未授权、未执行 | 建议综合回归与文档交付，不含真实训练 |

- **A 范围**：从实际代码反查，当前full严格符合FFN1→SA→FFN2三次pre-norm残差；T在Up专属norm前返回，不detach。无前段CDPA/相邻frontCDPA/后段Kimi AttnRes，无需裁定当前baseline冲突。八任务原CDLNO入口已完成，新增三模式字段/CLI/执行均未实现。
- **B diff**：新增审查文档、起点SHA清单和只读审查文本/结果；本文件、AGENTS、current-state仅增量记录，保留原文及既有远端启动修改。实际main/HEAD为`769fa333742f73c132868cf560bce5ec21529362`，原阶段0commit不是当前版本。
- **C 合同**：no_sa只移除SA完整残差层及其pre-norm/QK norms/投影；identity移除两个latent FFN及SA全部子层/norm，T=S；Down/Up/点更新和bridge/rear/CDPA/readout保留。LRSA固定快照的disable_interleaved_blocks同时跳过SA和FFN2，不等于no_sa。未来默认2+6的SA计数应为8/6/6，down/up/Conv各3及CDPA来源数不变。
- **D 实际检查**：`python -B docs/front_ablation_audit/static_check.txt`的13组源码/AST/JSON/hash检查通过，0失败；`git diff --check`通过。Python3.13.9仅作标准库审查执行器，没有检查/变更训练环境。模型前反向、旧pickle兼容、三目录新进程模型加载、PyG、GPU、真实数据/训练均未运行；既有120项回归是历史结果。
- **E 冻结证据**：起点203文件中仅AGENTS/STATUS/current-state三文档增加内容，其他200文件SHA相同。三个项目、core、配置、工具/测试、脚本、依赖本轮未变；对阶段0的65原文件，53字节一致，12差异仍仅此前已接入的入口/config，且这些12文件本轮未变。原exp冻结AST验收未重跑，不把静态不变写成真实数据通过。
- **F 自审与后续**：公式/T、前段CDPA缺席、参考开关、八任务加载链、冻结和基准边界已自审。新增模式需要受控兼容旧JSON、18槽frozen配置pickle和缺mode属性的旧front对象；FFN2-specific history_rule需准确迁移，不能strict=False或忽略版本。A0只有源码指纹，A1首次实现修改前必须生成真正旧full配置/权重/输出及Car/Air成员/list基准，不能改后伪造。性能计数目前仅适用于full，后续需跟随模式更新。没有剩余当前full架构缺陷需要用户裁定。

**本补充阶段结束，未执行下一阶段。**

**远端路径补充已完成：** 依据用户`/inspire/.../CDLNO-w`与`/inspire/.../data/fno`，填写原空path.sh，提供八任务train/eval脚本和只读数据格式检查器。16条默认命令+8组override实际parser检查通过，原模型/入口/数据/训练/配置均未改；没有重跑模型/GPU或真实训练。工业子目录尚未实测，Car阻力评价固定root/param0仍受限。见[远端脚本报告](CDLNO_REMOTE_LAUNCHERS.md)。针对远端旧版path.sh缺少`CDLNO_PYTHON`及任务变量的问题，现已增加旧变量派生回退；临时旧版path.sh下inspect及16/16 train/eval dry-run通过。前段full/no_sa/identity补充约束已记录，尚未收到具体A阶段实施授权，本轮未实现或接入这些参数。

**最新工作：用户要求的Car/NS最终逻辑核查和`tran_evaluate/`启动准备已完成，未启动真实训练/评价。最终120项回归通过（0失败/错误/跳过），四条脚本命令经真实parser AST解析检查通过。保持CDLNO M64/ratio2/2+6，纠正新NS preset额外clip0.1为原入口None。原数据/训练/模型/评价文件均未改。Car完整阻力评价固定路径/param0及原日志汇总问题已在[启动审查](CDLNO_TRAINING_LAUNCH_REVIEW.md)和[脚本说明](../tran_evaluate/README.md)记录；不能将既有AST冻结等同于原先所有生效默认值一致。真实数据、远端新模型兼容、精度、收敛与epoch效率仍未验收。**

以下保留阶段0–10的历史记录；关于NS clip的最终状态以上述启动核查为准。

**当前状态：阶段0–9已获用户审查通过；阶段10最终综合审查与交付已完成，等待用户审查。最终119项回归通过，无失败/错误/跳过；本轮仅补时间验收测试及文档，生产代码与依赖未改。真实数据训练/完整抽样评价与远端新模型环境仍未验证。**

## 持续有效的执行规则

1. 每轮只执行用户明确指定的一个阶段；阶段内自行完成必要修改和验证，不逐文件询问。完成后停止，等待用户审查并明确指示下一阶段。拥有完整计划不构成后续阶段授权；本规则替代 v1.2 §11 的一次完成主计划方式。
2. 设计优先级：**用户当前及后续明确确认的决定 → 聊天记录中较晚确认的决定 → v1.2 完整实施计划 → v1.1 历史补充**。数学附件用于解释、核对，不授权新增结构、损失或研究功能。历史讨论、放弃方案与最终决定分别记录。
3. 正式模型名及核心类为 **CDLNO — Cross-Depth Latent Neural Operator**，共享包名为 `cdlno`；机制仍为 CDPA。命名映射见下表，不全局替换机制名称。
4. 默认计算图：原任务输入提升 → 2 个完整 LRSA block → IPOT 式 bridge → 入口 CDPA → 6 个独立 persistent latent block → LRSA feature decoder。最终 query 和点残差均来自 H_F；规则网格用 LRSA 后置 dense 3×3 ConvFFN，潜空间无卷积。norm、QK norm、FFN、bias、初始化执行 v1.2 的完整规定。
5. 冻结数据读取、字段、划分、采样、点序、归一化、标签通道、损失、时间循环、优化器/调度器、评价处理。只在指定阶段修改新模型及必要的模型选择、参数转发、独立输出路径和 checkpoint 接入；原 Transolver 模型与脚本继续可用。
6. 必须支持 L/F 配置、P=L−F，默认 2+6，覆盖 F=0…6 和扩展 L；CDPA 支持 off/entry/every_block 及数学等价的来源批处理/分块，entry 是默认主设置。全阶段 M 相同，任务初值按 v1.2，不被旧 parser 默认值静默覆盖。上述能力按用户分配的相应阶段实现，不在本轮自动创建。
7. 当前无真实数据：不下载、不启动真实训练，不 import 顶层读数据的 exp 脚本。阶段验收按范围完成静态审查、合成张量、原损失连接、前反向与 checkpoint 检查；分别记录实际通过、失败、未运行。理论或环境检查不是模型验收。
8. 目标环境是用户已有 CUDA12.8、torch2.11，原 Transolver ShapeNet-Car 已有训练成功证据。先检查并保留，不盲目重装/升级/降级。无 GPU 访问则 GPU 项未运行；新核心用 PyTorch SDPA，不引入自定义 CUDA/Triton kernel 或整套参考项目训练框架。
9. 排除稀疏 Darcy、坐标 query decoder、递增 M、CDPA-Slice、旧校正层、新物理损失、latent 卷积、跨真实时间 latent 缓存、缺少相应长轨迹标签的 NS 10→20/40。不得以预留扩展为由添加这些代码。
10. 读取适用 AGENTS.md，保护已有用户修改。不自动 commit/push、建 PR、训练、reset 或无关重构。实质架构/数据协议冲突须列出依据和建议，不自行改方案。

每个阶段结束后更新本文件及项目记忆，按 A–F 报告，并明确写出 **“本阶段结束，未执行下一阶段”**。

补充持续约束：每阶段交付前，先自行审查拟列出的3–5项重点，核对源码、确认规格和测试证据；通过项记录在报告，向用户仅提出自审后仍存在的缺陷、未确定项或需要其判断的决定。不能把尚未自行检查的清单直接交回用户；阶段之间仍须等待明确授权。

## 阶段状态

| 阶段 | 状态 | 说明 |
|---:|---|---|
| 0 | **完成，已审查通过** | 已阅读材料并完成 Transolver/LRSA/IPOT、接口、许可和冲突审查 |
| 1 | **完成，已审查通过** | 共享 `cdlno` 包、配置/sidecar基础协议、数据无关环境预检；未实现 block/训练接入 |
| 2 | **完成，已审查通过** | 基础层、完整LRSA前段、bridge、独立后段、feature readout；18项模块测试+2项原LRSA同配置对照通过 |
| 3 | **完成，已审查通过** | 独立CDPA、显式数学reference、21项测试通过；包含GPU和基础sidecar回归 |
| 4 | **完成，已审查通过** | CDLNO核心、三模式历史与73项全回归中的既有核心验证；后续发现的runtime chunk类型缺口于阶段5修复 |
| 5 | **完成，已审查通过** | 仅四静态标准任务wrapper/注册/有限入口/checkpoint/独立脚本；13项新测试通过，自审完成 |
| 6 | **完成，已审查通过** | NS/Plasticity时间wrapper、循环语义、配置、脚本、严格checkpoint；8项定向测试，81项全回归 |
| 7 | **完成，已审查通过** | 仅ShapeNet-Car；PyG合成接口、mask损失、checkpoint/sidecar通过；未运行真实数据训练 |
| 8 | **完成，已审查通过** | AirfRANS接口/reference71/YAML398/整模型与列表加载；13项定向、107项全回归通过，实际抽样评价未运行 |
| 9 | **完成，已审查通过** | v1.2 §9性能工具/薄matched LRSA；118项回归、3组配置27行GPU结果通过；仅合成模型性能 |
| 10 | **完成，待用户审查** | 最终需求矩阵、三仓/冻结核查、README/报告；补齐时间测试，119项通过；无真实训练 |

详细报告见 [CDLNO_REFERENCE_AUDIT.md](CDLNO_REFERENCE_AUDIT.md)。用户已补充远端实证：Python 3.10 环境按 cu128 对应的 PyTorch 2.11 安装方式部署后，`Car-Design-ShapeNetCar` 原训练程序已经跑通。该远端环境作为当前最可靠的兼容性依据；Python 3.11 作为同一依赖族的待确认兼容目标。本地工具环境只作为非权威观察，不用于兼容性决策。

## 正式命名映射

| 对象 | 旧工程占位 | 当前确认名称 / 处理 |
|---|---|---|
| 完整模型 | CDPA 模型（语境上指整网） | CDLNO，Cross-Depth Latent Neural Operator |
| 共享 Python 包 | `cdpa_operator` | `cdlno` |
| 共享核心类 | `CDPAOperator` | `CDLNO` |
| 新整网的模型注册名 | `CDPA` | `CDLNO` |
| 原实验入口的 wrapper 导出 | `Model` | 可继续导出 `Model`，内部调用 CDLNO |
| 跨深度物理注意力机制 | CDPA / 早期 CDPA-Cross | 保留 CDPA，Cross-Depth Physics Attention |
| 机制配置或内部模块 | `cdpa_mode`、`cdpa_source_chunk_size` 等 | 保持机制语义，不因整网改名而全局替换 |

已建立 `cdlno` 共享包、阶段2基本模块、阶段3独立机制类 `cdlno.cdpa.CDPA` 和阶段4核心类 `cdlno.core.CDLNO`；六标准任务和两个工业任务已通过CDLNO注册项和wrapper接入。八任务JSON与16份启动脚本见 [CDLNO_TASK_LAUNCHERS.md](CDLNO_TASK_LAUNCHERS.md)。旧计划文件保留原文，后续代码按已确认命名实施。

## 阅读依据与基线

- [AGENTS.md](../AGENTS.md)、[当前记忆](../memory/current-state.md)。
- [v1.2 完整计划](../PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md)、[v1.1 历史补充](../PLAN_CDLNO/CDPA_v1_1/CDPA_Transolver_Implementation_Plan_v1_1.md)。
- [151 页聊天导出](../PLAN_CDLNO/比较模型架构.pdf)已完整通读；[页码索引与演变记录](../memory/2026-09-13-exported-conversation-review.md)区分废弃、确认与建议。
- [参考仓库审计](../memory/2026-09-13-cdpa-plan-and-reference-audit.md)：LRSA `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`；IPOT `18c177846267505ee9503445a146dfd7dee34c41`。
- Transolver 基线：`main`，`75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。本轮开始时 `git status --short` 为未跟踪的 `AGENTS.md`、`PLAN_CDLNO/`、`memory/`，已跟踪文件没有 diff。
- 数学附件位置待核实：用户说明已经提供，但本轮仓库文件名检索未定位 `CDPA_Mathematical_Foundations.md`、`CDPA_Theory_Manuscript.tex/.pdf`、`check_theory_identities.py`、`theory_identity_checks.json`。已读导出中的理论摘要不能代替附件全文；不假定附件未在其他路径提供，不伪报已读或已复现。

## 阶段 0 交付记录（2026-09-13）

- **A 完成范围**：完成材料、基线、LRSA/IPOT源码、许可、八任务接口、冻结区段和计划冲突审查。
- **B 文件**：新增 `docs/CDLNO_REFERENCE_AUDIT.md`；本状态文件同步阶段表和报告入口；未改模型、训练、数据或依赖。
- **C 公式/形状**：记录 `H:[B,N,d]`、`T/Z:[B,M,d]`、独立历史 Cross、`R0=Z`、depth softmax 和 `H_F` feature decoder；尚无实现代码。

### 阶段 0 复核补充：Kimi K3 与最终 decoder

- 已核对 Kimi K3 技术报告 §2.2、式 (8)–(10)：AttnRes 直接提供 learned pseudo-query 的跨深度来源评分/聚合；CDLNO 的每历史独立 Cross 是面向 latent token 对齐的明确适配，不是逐式复制。
- `Z0→T1`、`Z0→T2` 必须是两个独立 `M×M` Cross；之后 `[R0=Z0,R1,R2]` 在每个 token 上做来源 softmax。不得使用 `2M` 单次 Cross，也不得在融合外再加一次 `Z0`。
- LRSA 代码核对确认 up 是点 query、latent K/V，外层再加点残差并做 point FFN。最终 CDLNO decoder 按 v1.2 使用 `H_F` 作为 query 来源和原值残差，`Z_P` 只作 K/V，并保留一次 point FFN/ConvFFN 与输出 head。
- 以上只更新审查记录；模型、训练、数据和依赖配置仍未修改，阶段 1 及后续阶段未执行。
- **D 验证**：静态源码和 Git 审查完成；远端环境、forward/backward、checkpoint、真实数据训练均未运行。
- **E 冻结证据**：未修改原 Transolver 模型、入口、数据、loss、时间循环、优化器、scheduler、评价或依赖；本地环境不作为远端兼容依据。
- **F 待审查**：数学附件路径、LRSA许可证、旧 parser 与新任务默认冲突、阶段1范围。

**本阶段结束，未执行下一阶段。**

## 阶段 1 交付记录（2026-09-13）

### A. 完成范围

建立了可 editable 安装的根级 `cdlno` 包；实现架构/运行配置分离、`P=L-F` 派生和合法性检查；建立 sidecar `architecture.json` 的序列化、读取、架构比较与不可覆盖策略；新增只读、无数据环境预检。没有实现任何 LRSA/IPOT/CDPA block，也没有修改训练、数据或模型选择入口。

### B. 修改文件、理由和 diff 摘要

- `pyproject.toml`：最小 setuptools editable 构建配置，Python 3.10–3.11 约束，运行依赖留空以避免阶段 1 改动远端依赖。
- `cdlno/__init__.py`、`cdlno/config.py`：共享命名导出与配置对象。`CDLNOArchitectureConfig` 只接受 `L/F`，`P` 为只读派生属性；没有独立 rear-depth 字段。`CDLNORuntimeConfig` 保存 chunk/device/dtype/SDPA 等运行字段。
- `cdlno/checkpoint.py`：sidecar 写入/读取/比较/校验；校验顺序固定为先读取已有 sidecar、再比较架构；运行字段差异不参与不兼容判断；默认拒绝覆盖已有 sidecar。
- `tools/cdlno_environment_preflight.py`：只读检查 Python、torch、SDPA、numpy、yaml、einops；不 import `exp_*.py`，不安装依赖、不下载数据、不训练。
- 本状态与 `memory/current-state.md`：记录阶段 1 协议、验证和边界。

### C. 关键协议与形状对应

- 架构配置序列化保存 `L,F`，并额外写入信息性 `derived.P=L-F`；加载时重新计算并拒绝不一致值。默认 `L=8,F=2,P=6`，但合法性只要求 `0≤F<L`，未硬编码 `F≤6`。
- `architecture` 字段代表 checkpoint 结构语义；`runtime` 字段代表执行路径。`compare_architecture()` 只比较前者，因此改变 `source_chunk_size/device/dtype/sdpa_backend` 不会拒绝相同权重。
- 当前没有模型 tensor；后续模型应使用这些协议保存 `[B,N,d]` 点状态和 `[B,M,d]` latent 状态的架构元数据。

### D. 实际验证命令、环境、通过/失败/未运行

| 检查 | 状态 |
|---|---|
| 配置往返、`0≤F<L` 非法值、拒绝 `rear_depth`、`P` 派生一致性 | **通过**（临时 Python 合成检查） |
| sidecar 写入、读取优先、架构 mismatch 拒绝、runtime 差异放行、已有文件拒绝覆盖 | **通过** |
| 三个原任务工作目录新进程 `import cdlno` | **通过**；均解析 `/home/hwz/CDLNO/cdlno/__init__.py` |
| `tools/cdlno_environment_preflight.py` | **通过执行**；仅完成观察，不代表远端兼容验收 |
| 本地观察值 | Python 3.13.9、torch 2.13.0+cu130、CUDA runtime 13.0、CUDA 可用；与目标 Python 3.10/3.11、torch 2.11、CUDA 12.8 不同，标为**非权威环境观察** |
| 依赖安装/替换、远端 CUDA12.8/torch2.11 检查 | **本地未运行**；用户报告远端 Python 3.10 + torch 2.11/cu128 + PyG/pyg-lib 可用，且 ShapeNet-Car 原训练已通过 |
| 模型 forward/backward、真实数据、训练入口、checkpoint tensor round-trip | **未运行**；本阶段明确不实现 |

### E. 冻结区域变化证据

`git status --short` 显示新增仅限 `cdlno/`、`pyproject.toml`、`tools/cdlno_environment_preflight.py` 及既有审查文档/计划/记忆；`PDE-Solving-StandardBenchmark/`、`Car-Design-ShapeNetCar/`、`Airfoil-Design-AirfRANS/` 的数据、训练、模型入口和依赖文件没有 diff。未执行安装、下载、训练、reset、commit 或 push。

### F. 未解决问题与优先审查点

1. 当前没有执行 `pip install -e .`，因为本阶段约束禁止安装/替换依赖；pyproject 已可供用户远端环境显式 editable 安装。
2. sidecar 目前是基础协议，尚未接入实际 checkpoint 保存/加载路径；这是后续阶段任务。
3. 远端 Python 3.10 + torch 2.11/cu128 的基础可用性已有 ShapeNet-Car 原训练成功证据；Python 3.11、六个标准 PDE 任务和新 CDLNO 路径仍未验证。
4. 用户给出的安装命令先指定 `torch_geometric==2.4.0`，随后又执行 `pip install -U torch_geometric`；最终 PyG 版本应以后一次命令在远端实际解析出的版本为准，不能仅凭命令文本把 2.4.0 写死为最终版本。`pyg-lib` wheel 必须与 `torch-2.11.0+cu128` 页面及实际 Python ABI 匹配。
5. 用户消息中的示例 URL 显示为 `cu121`，并附有按驱动选择 `cu121/cu124/cu126/cu128` 的说明；因此不能仅凭命令文本断言远端最终 CUDA runtime 是 12.8。后续若需精确记录，应以远端实际 `torch.__version__`、`torch.version.cuda`、`torch_geometric.__version__` 和 `pyg_lib` 版本为准。

建议优先审查：架构字段是否足够覆盖 v1.2 的最终模型语义；sidecar 读取优先且禁止覆盖的行为；`P` 是否始终只由 `L-F` 派生；三个工作目录是否接受根级 editable 包路径。

**本阶段结束，未执行下一阶段。**

## 阶段2交付记录（2026-09-13；后续已获用户审查通过）

详细公式、源码差异、执行命令及初次失败记录见 [CDLNO_PHASE2_MODULES.md](CDLNO_PHASE2_MODULES.md)。

- **A 范围**：实现 norm、SDPA、plain/GEGLU/dense ConvFFN、完整前段、bridge、独立后段、H_F feature readout。各模块独立验证，未实现CDPA或整网。
- **B 文件**：新增 `cdlno/modules.py`、`tests/test_modules.py`、`tests/test_lrsa_reference.py` 和阶段2报告；只更新包说明与审查/状态/AGENTS/记忆。阶段1配置、checkpoint及工程依赖文件未动。
- **C 公式/形状**：前段 `[B,N,d]→(H_next[B,N,d],T[B,M,d])`，两次latent FFN且T在up前返回；bridge `[B,N,d]→[B,M,d]` 有query residual；后段 `[B,M,d]→[B,M,d]` 的Q/K/V同源LN；readout `(H_F,Z_P)→[B,N,C_out]`，输出head按§2.7用LayerNorm，分支外层按§4.3用RMSNorm。
- **D 已通过**：`python -B -m unittest discover -s tests -p test_modules.py -v` 为18/18；`CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p test_lrsa_reference.py -v` 为2/2。LRSA点/结构版同配置同权重FP64输出、T和对应梯度本次误差均0。5×7非方形点序、初始化、参数独立性通过。实际GPU RTX5090 Laptop 的FP32及FP16/BF16 autocast小模块前反向通过。
- **D 初次失败与限制**：本地CPU oneDNN不支持BF16卷积反向（2个错误），改为测试内关闭MKLDNN的CPU数学路径后通过；GPU结构版CPU对照首次约4.77e-6差异（1个断言），测试关闭TF32后在原容差下通过。模块不设置全局backend。执行机torch2.13/cu130；用户远端torch2.11/cu128、新整模型、任务损失及真实训练未运行。
- **E 冻结证据**：75个原始/阶段1基础文件逐项SHA256核对均不变，`git diff --name-only HEAD`为空。4个新增/更新Python文件通过3.10语法解析（非3.10运行验收），全部本轮文件直接检查空白与本地文档链接通过。原任务模型、入口、数据、损失、时间循环、优化器/调度器、评价及依赖文件不变。
- **F 优先审查**：T取点和完整前段；down/bridge残差区别；后段同源LN和独立参数；readout的H_F/RMS/LN_out；dense ConvFFN点序/bias/初始化。配置绑定与CDPA/整网/任务接入等待下一阶段明确授权。

**本阶段结束，未执行下一阶段**

## 阶段3交付记录（2026-09-13）

完整公式、API、容差、测试范围及 **§8.3全部14条逐项覆盖表** 见 [CDLNO_PHASE3_CDPA.md](CDLNO_PHASE3_CDPA.md)。

- **A 范围**：仅实现单次CDPA：共享每来源Cross对齐、R0恒等、逐token RAW候选融合、FP32 depth及chunk0/1/k等价执行；未组装模型或接入数据/训练。
- **B 文件**：新增 `cdlno/cdpa.py`、`tests/cdpa_reference.py`、`tests/test_cdpa.py`、阶段3报告；更新本状态、AGENTS和当前记忆。既有配置/sidecar、阶段2模块/测试和依赖文件均未修改。
- **C 公式/形状**：`Z,T_s:[B,M,d]`；每来源token softmax在M轴，chunk折叠为 `[B*k,h,M,d_h]`，Q只投影一次。完整O+b后得到R_s，R0=Z，depth权重 `[B,M,S+1]`；w=0给候选均值，无外加Z或gate。depth禁用autocast并FP32计算后转回Z.dtype。
- **D 实际通过**：`python -B -m unittest discover -s tests -p test_cdpa.py -v`：21/21，0失败/错误/跳过，3.141s。CPU48例reference矩阵比较全部输入/参数梯度；实际RTX5090 Laptop GPU执行FP32/FP16/BF16。执行机Python3.13.9、torch2.13/cu130，非远端目标验收。新文件3.10语法解析通过；无依赖改动。
- **D 边界**：§8.3第1条单调用恒等通过、F0/entry参数省略待核心；第9条融合梯度通过、整网梯度待核心；第11条逐层历史ID/时序待核心；第14条同权重跨chunk及现有sidecar运行/架构边界通过，实际模型/任务checkpoint接入待后续。远端torch2.11/cu128、原损失连接、真实训练和性能未运行。
- **E 冻结证据**：阶段开始86项文件SHA256均不变，`git diff --name-only HEAD`为空；对未tracked的本轮文件另做语法、空白/换行和文档路径检查。三任务模型、入口、数据、loss、时间循环、优化器/调度器、评价及依赖不变。清单路径及SHA256见阶段3报告。
- **F 优先审查**：O+b/RAW/恒等候选；来源与batch布局及两轴softmax；FP32子图和零w梯度；独立reference与容差/混合dtype；第1/9/11/14条剩余核心/入口验收边界。本阶段范围内无实质架构冲突。

**本阶段结束，未执行下一阶段**

## 阶段4交付记录（2026-09-13）

完整架构张量流、模式历史表、实际调用计数、逐条§8.3边界和失败修正记录见 [CDLNO_PHASE4_CORE.md](CDLNO_PHASE4_CORE.md)。

- **A 范围**：新增 `cdlno.core.CDLNO`，接收已提升 `H_0[B,N,d]`，执行F个完整前段、bridge、三种CDPA模式、P个独立后段和 `H_F` feature readout，仅返回 `[B,N,C_out]` Tensor。默认L8/F2/P6；未接八任务wrapper、数据或训练。
- **B 文件**：新增 `cdlno/core.py`、`tests/test_core.py`、`tests/test_core_config.py` 和阶段4报告；更新 `cdlno/config.py`（核心版本/历史规则、独立latent ratio、网格与合法性检查）、`cdlno/__init__.py`（懒加载CDLNO）、本状态、AGENTS和记忆。阶段2/3实现、checkpoint.py、依赖和原任务目录未改。
- **C 公式/形状**：前段 `[B,N,d]→(H_i[B,N,d],T_i[B,M,d])`；bridge→raw `Z_0[B,M,d]`；后段第j层当前 `Z_(j−1)`，every历史 `T_1…T_F,Z_0…Z_(j−2)`；readout `(H_F,Z_P)→[B,N,C_out]`。历史仅本次forward存活并传tuple快照，无detach或跨位置K/V缓存。模块调用down/bridge=F+1、up/readout=F+1、latent SA=L（仅full；no_sa/identity为P）、结构ConvFFN=F+1。默认chunk0实际测得：off 0份历史/0次历史SDPA/14次总SDPA；entry 2/1/15；every 27/6/20。
- **D 实际通过**：最终 `CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v` 为 **60/60通过，0失败/错误/跳过，12.912秒**，日志 `/tmp/cdlno-phase4-final-tests.log`。其中阶段4为13项核心+6项配置测试：42组L8/F0…6/三模式/点与5×7网格；L12/F2、L16/F6、L1/F0及F7/F8；非法配置/输入/chunk；完整raw历史ID和两forward图隔离；同权重F0等价；参数/storage独立、所有活跃路径梯度；真实chunk切换的state_dict、sidecar不覆盖、新进程权重恢复与三个任务cwd共享导入。核心数学reference/跨chunk容差atol1e-5/rtol3e-4。
- **D 环境/失败/未运行**：本地Python3.13.9、torch2.13.0+cu130、RTX5090；核心18组GPU FP32/FP16 AMP/BF16 AMP前反向有限，已有阶段2/3精度检查通过。早期测试计数断言及冻结slots只读属性的异常类型期望曾失败，现已修正；另外补强了实际SDPA、完整历史ID及跨chunk证据，没有改变核心数学或阶段2/3实现。远端Python3.10/torch2.11/cu128、任务loss/入口checkpoint、真实训练和性能未运行。未实施未经确认的自动dtype转换。
- **E 冻结证据**：阶段开始保存98文件清单及5个允许更新文件副本；93个非目标文件SHA256全部不变。71个原tracked文件无diff。原八任务模型、入口、数据、loss、时间、优化器/调度器、评价和依赖未编辑；无安装、数据下载、训练、commit或push。新Python文件和修改后的配置/导出另做Python3.10语法解析；不声称在3.10解释器实际运行。
- **F 优先审查**：every raw历史时序与F0空位置；同权重F0 entry/off等价、位置独立与单次初始化；3/3/8/3模块计数及2/1、27/6来源/调用区别；ratio/grid/version的架构sidecar与运行字段分离；HF query/残差、Tensor返回和后续wrapper边界。当前范围内无待改架构冲突，任务接入等待下一阶段明确授权。

**本阶段结束，未执行下一阶段**


## 阶段4补充自审（2026-09-13）

按用户要求已自行审查此前4项重点，详见 [阶段4报告补充自审](CDLNO_PHASE4_CORE.md#补充自审2026-09-13)。15项定向CPU测试全部通过（4.991秒），但原测试没有覆盖运行配置的非法chunk类型；额外复现确认 `True/1.5/NaN/Inf` 能通过 `CDLNORuntimeConfig.validate` 及sidecar保存/加载/比较，而核心正确拒绝。先前“无待修复问题”结论需据此收窄。

raw Z0及完整历史时序、同权重F0等价、独立初始化、实际来源/SDPA计数、HF query与残差、正常架构/运行字段区分及sidecar不覆盖均未发现错误。唯一待处理项位于 `cdlno/config.py:124`：运行字段缺少非负整数类型校验。拟仅补此校验（不引入torch依赖）并加入运行配置/sidecar回归；不改变模型计算、架构兼容策略或已有文件。按此前“先报告计划、由用户决定修改”的要求，本轮尚未执行修复。

本轮只更新AGENTS、STATUS、phase4报告、current-state共4份Markdown。审查前102文件快照 `/tmp/cdlno-phase4-self-review-6ac3ba44/hashes.json` 用于核对其余98文件不变；未改实现/测试/数据/训练/依赖。远端和GPU本轮未重跑，沿用前次明确标注的历史证据；未进入阶段5。

**本阶段结束，未执行下一阶段**


## 阶段5交付记录（2026-09-13）

完整接口、文件清单、启动方式、冻结证明、自审结论见 [CDLNO_PHASE5_STATIC_TASKS.md](CDLNO_PHASE5_STATIC_TASKS.md)。

- **A 范围**：只完成Darcy、Elasticity、Airfoil、Pipe四静态任务；新增共享输入适配、两个Model wrapper、CDLNO工厂分派、四exp新参数/构造/严格checkpoint有限分支、4份JSON预设及8份独立训练/评估脚本。NS/Plasticity/工业任务未接入。
- **B 差异**：新增 `cdlno/standard.py`、标准项目 `cdlno_entry.py`、两个wrapper、预设、脚本、`tests/test_static_standard.py` 和阶段报告；修改四exp/model_dict，以及runtime chunk整数校验、AGENTS/STATUS/记忆。旧模型/脚本、核心及阶段2/3数学模块、依赖均未改。
- **C 合同**：Darcy `[B,7225,2]+fx1 → 固定索引reference64+fx1 → stem65`，无placeholder；Elasticity972/Airfoil11271/Pipe16641点，fx=None，保留坐标/placeholder。Pipe坐标normalizer仍在原exp。静态wrapper无time_fc，输出均 `[B,N,1]`；只在wrapper局部初始化stem，不递归覆盖核心。默认d128；h为8/8/4/4，M为64/64/64/32，L8/F2/P6，epochs500、batch4/1/4/8。
- **D 通过**：`python -B -m unittest discover -s tests -p test_static_standard.py -v` 13/13；最终 `CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v` **73/73，0失败/错误/跳过，26.554秒**。CPU真实N减小d/M前反向、Darcy原normalizer/decode/零边界/梯度项loss、原stem对照、strict权重及新进程恢复、sidecar/输出隔离、脚本覆盖和完整旧AST还原通过。本地GPU12组wrapper精度/前反向有限检查通过。
- **D 失败/未运行**：首次测试把Darcy s×s误认成5×7，修正测试H/W后通过；一次编辑生成的缩进错误由落盘前AST检查截获并修正。最终无失败。实际环境Python3.13.9/torch2.13+cu130/RTX5090，不代表远端3.10/torch2.11/cu128验收。未下载数据、import exp、启动真实训练/评价或安装依赖。
- **E 冻结**：102文件基线 `/tmp/cdlno-static-baseline-0a0j551e/hashes.json`；93个非目标文件哈希不变。原71 tracked文件仅四exp和工厂有diff（239增加/119删除）；限定投影后四exp及model_dict完整AST等于原commit。数据、归一化、loss、优化器/调度器、循环、旧checkpoint分支和评价数组处理均保留。
- **F 自审**：已自行核对输入合同/placeholder/time、初始化/共享、默认/旧分支、sidecar/目录隔离、证据范围，未发现剩余新增实现问题。上轮chunk缺口已补校验并验证非法值不会写入或覆盖sidecar。原评估绘图固定网格尺寸未改，非默认下采样可视化未验收；远端与真实数据效果仍待实际运行。

**本阶段结束，未执行下一阶段**


## 阶段6交付记录（2026-09-13）

详见 [CDLNO_PHASE6_TEMPORAL_TASKS.md](CDLNO_PHASE6_TEMPORAL_TASKS.md)。

- **A 范围**：仅接入Navier–Stokes与Plasticity，复用共享核心和标准入口协议；工业任务及后续阶段未执行。
- **B 文件**：新增`TemporalStandardModel`、时间wrapper、NS/Plasticity JSON预设、4个脚本和`tests/test_temporal_standard.py`；修改`model_dict.py`、`exp_ns.py`、`exp_plas.py`、AGENTS/STATUS/记忆。未复制数学核心。
- **C 合同**：NS输入`x[B,N,2],fx[B,N,10]`，固定64×64时N=4096，stem=74，输出`[B,N,1]`；Plasticity空间101×31=3131，`fx[B,N,1],T[B,1]`，stem=3，输出`[B,N,4]`。NS每次forward重建core；Plasticity每个T独立调用，time_fc仅Time_Input=True注册。
- **D 通过**：`python -B -m unittest discover -s tests -p test_temporal_standard.py -v`为8/8；最终`CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v`为**81/81通过，0失败/错误/跳过**（日志`/tmp/cdlno-phase6-full-tests.log`）。覆盖B>1真值/预测窗口回填、不同T与时间梯度、Plasticity 20次独立更新、原循环AST合同、strict checkpoint和新脚本语法。本地Python3.13.9/torch2.13+cu130/RTX5090；未运行远端3.10/torch2.11/cu128、真实轨迹训练或评估。
- **E 冻结**：NS/Plasticity原时间循环仅增加新模型/sidecar/run目录条件；原读取、标签轴、loss、optimizer/scheduler、teacher forcing/autoregressive逻辑保留。未修改数据、normalizer、loss工具或依赖。
- **F 自审**：已核对NS truth/prediction反馈位置、每次forward无跨时间latent、Plasticity T形状/梯度及20步更新、输出通道和checkpoint sidecar。未发现新增实现缺陷；真实数据和远端环境仍未运行。

**本阶段结束，未执行下一阶段**


## 阶段7交付记录（2026-09-14）

完整报告：[CDLNO_PHASE7_SHAPENET_CAR.md](CDLNO_PHASE7_SHAPENET_CAR.md)。阶段6已获用户审查通过，本阶段只接入ShapeNet-Car。

- **A 范围**：稳定Model wrapper、模型选择/参数转发、任务JSON、独立脚本和run/eval路径、原整模型checkpoint及sidecar接入。AirfRANS未修改。
- **B 文件**：新增Car models/CDLNO.py、models/cdlno_run.py、配置及2脚本、tests/test_shapenet_car.py；修改Car main.py/main_evaluation.py和记录文档。core、六标准任务、train.py、dataset、原模型/脚本及依赖未改。
- **C 合同**：x[N,7]→原stem+placeholder→H0[1,N,d]→共享point核心→[N,4] velocity3/pressure1；忽略geom/y，保持节点序和surf外部mask。batch/ptr明确拒绝多图；单图和可变N合法；无time_fc/geom encoder。
- **D 实际验证**：`python -B -m unittest discover -s tests -p test_shapenet_car.py -v` **13/13通过**；`CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v` **94/94通过，0失败/错误/跳过，57.568秒**，日志`/tmp/cdlno-phase7-final-tests.log`。原train.train/test真PyG合成批次、全部梯度和optimizer/scheduler步、32186点减小d/M、双独立Car进程整模型保存/加载、sidecar不覆盖和fold/架构拒绝均通过。另行GPU真PyG FP32三模式小张量通过。
- **D 环境/边界**：本地Python3.13.9、torch2.13/cu130、PyG2.3.1、RTX5090；无安装。未执行远端3.10/torch2.11/cu128、真实数据/loader/drag评价或训练，未执行九折。初版3项测试未涵盖交付合同，续做修正实现及测试后由13/94最终结果替代。
- **E 冻结**：两个完整入口AST去除有限新分支/路径别名及授权兼容改动后与基线一致；其他37个原Car/AirfRANS文件逐字节一致。仅原Car eval的epoch类型float→int和两处局部可信整模型load兼容属于原行为例外；原保存协议/损失/循环未改。
- **F 自审**：已复核placeholder/初始化、graph元数据、mask差异、sidecar先读/strict整模型/输出隔离和冻结证据；未发现剩余实现缺陷。远端及真实数据是待运行边界。

**本阶段结束，未执行下一阶段**


## 阶段8交付记录（2026-09-14）

完整报告：[CDLNO_PHASE8_AIRFRANS.md](CDLNO_PHASE8_AIRFRANS.md)；配置脚本清单：[CDLNO_TASK_LAUNCHERS.md](CDLNO_TASK_LAUNCHERS.md)。

- **A 范围**：仅AirfRANS输入wrapper、双入口模型选择/参数/独立路径、YAML新key、JSON及训练/评估脚本、可信整模型/模型列表和sidecar。八任务均已具备模型接口及启动配置。
- **B 文件**：新增cdlno/airfrans.py、Air models/CDLNO.py、cdlno_entry.py、配置/2脚本、test_airfrans及报告/清单；修改Air main/main_evaluation/params（+49/−18行）、阶段7冻结测试清单和AGENTS/STATUS/memory。原Air train/dataset/metrics/模型/旧脚本及共享数学未改。
- **C 合同**：原x[N,7]追加pos[N,2]在[-2,4]×[-1.5,1.5]域的64维reference距离→stem71+placeholder→共享core→[N,4] vx/vy/p/nut。支持变N单图，兼容原Infer_test保留的单图旧ptr；多图batch/ptr报错。输出节点序和外部mask处理不变。
- **D 验证**：`python -B -m unittest discover -s tests -p test_airfrans.py -v` **13/13通过，8.123秒**；`CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v` **107/107通过，0失败/错误/跳过，76.872秒**，日志`/tmp/cdlno-phase8-final-tests.log`。真PyG合成接口/原loss全部梯度、原reference精确对照、原工作目录两个新进程的真实构造/加载分支和完整模型/list往返、sidecar读取优先/不覆盖及输出隔离通过。另行CUDAFP32三模式PyG小张量前反向通过。
- **D 环境/未运行**：本地Python3.13.9、torch2.13/cu130、PyG2.3.1、RTX5090；无安装。未执行远端3.10/3.11/torch2.11/cu128、真实数据/采样/radius_graph/idx平均/边界处理/指标或训练；源码冻结证据不等于这些路径已运行。
- **E 冻结**：开始Air无用户已有diff；Transolver预算398保持，params仅追加CDLNO且原key一致。两个完整入口AST去除授权模型/路径/load差异后等于基线，21个其余原Air文件逐字节不变。阶段7测试仅将授权的3Air文件交由阶段8核查，原Car和其测试继续回归。
- **F 自审**：已核对71追加/域/placeholder、抽样ptr、多模型目录与可信list加载、398继承/显式覆盖、原图处理冻结及my_path区别。范围内未发现剩余实现缺陷；真实数据及远端效果仍待独立验证。

**本阶段结束，未执行下一阶段**


## 阶段9交付（2026-09-14）

A. 完成独立benchmark、八任务合成工厂、原Transolver/lrsa_matched/CDLNO三模式对照、task/matched分组、chunk同权重比较和同步性能/显存/成本记录；没有接入额外训练架构。详细报告：[CDLNO_PHASE9_PERFORMANCE.md](CDLNO_PHASE9_PERFORMANCE.md)，命令：[CDLNO_PERFORMANCE_TOOLS.md](CDLNO_PERFORMANCE_TOOLS.md)。

B. 新增tools/cdlno_benchmark.py、tools/cdlno_perf四文件、tests/test_performance.py及报告/原始证据；既有文件仅更新AGENTS/STATUS/memory。核心、任务、依赖保持。

C. L个完整LRSA block只接LN/head，末层已up；2+6实际down/up/SA/Conv=3/3/8/3，entry2来源/1历史SDPA、every27/6（chunk0）。所有N投影、FFN、denseConv、CDPA Q/K/V/O均计矩阵MAC；bias/norm/depth/softmax/临时复制和存储另列，profiler漏计SDPA不当0；2FLOPs/MAC不冒充精确全算子FLOPs。

D. 完整回归118/118通过59.769秒，无失败/错误/跳过。三组顺序执行GPU配置共27行通过：Elasticity matchedN972、Airfoil task221×51/B4、Airfoil matched17×23/B2。原始结果在[performance/phase9](performance/phase9/elasticity-matched.json)。同权重chunk最大输出差1.79e-7/参数梯度差1.44e-6；5warmup/20samples、同步median/p90、AdamW状态已初始化。实际环境Python3.13.9/torch2.13+cu130/PyG2.3.1/RTX5090Laptop，FP32/TF32off/AMPoff/compileoff、实际efficient SDPA。远端目标、其他精度/backend/compile、真实训练/epoch及完整网格矩阵未运行。旧/tmp初测记录在续接时不可用，最终验收重新留存持久化结果，不使用丢失记录。

E. 152文件续接hash manifest最终核对：149文件相同，仅AGENTS/STATUS/memory变化，意外改动0；既有AST/字节冻结回归保护先前改动；本阶段没有编辑三个任务子项目或cdlno/、依赖。无exp/main导入、真实数据/训练、下载、安装、commit/push/PR。

F. 五项优先自审完成，没有遗留架构/接口缺陷。效率并非一致提升：Elasticity entry0.627GMAC却训练34.96ms，高于原Transolver24.30ms；小算子/launch/AdamWtensor开销有诊断证据，未改模型。Airfoil原任务preset中entry77.64ms对原Transolver135.96ms，但heads4/8不同，不能混成同结构比较或epoch收益。运行波动/远端外推边界保留在报告。

本阶段结束，未执行下一阶段。


## 阶段10：最终综合审查与交付

- 用户明确批准阶段9并授权最终审查。交付 [最终报告](CDLNO_IMPLEMENTATION_REPORT.md)、[逐项需求矩阵](CDLNO_REQUIREMENTS_MATRIX.md)、[来源许可](CDLNO_THIRD_PARTY_NOTICES.md)、README使用说明及更新的八任务清单；证据与可审查patch保存在 [final_audit](final_audit/)。
- 需求反查发现测试证据缺口：既有NS合成只跑3步，既有时间AST仅查关键字符串。仅修改test_temporal_standard.py，补齐两路10步窗口/单次backward与step、Plasticity同batch不同T/time_fc梯度、两份完整exp AST基线比较。没有发现或修改生产架构/数据/训练缺陷。
- 修改前118/118（58.082s），补充定向9/9（10.114s），最终119/119（49.871s）全通过，无失败/错误/skip。实际命令CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v；最终日志docs/final_audit/regression.log。
- CPU包括point/grid各21配置、扩展L/非法值、原N布局、原loss/时间与checkpoint；真实PyG Data/Batch两工业集成通过；三原cwd独立进程import与严格load通过；六个本地GPU测试方法实际执行。环境Python3.13.9/torch2.13+cu130/PyG2.3.1/RTX5090 Laptop。目标Python3.10/3.11+torch2.11/cu128及editable安装未运行，未更换依赖。
- 阶段0全部71 tracked files核查：58字节不变，12既有授权模型分支/YAML，本轮仅README追加。阶段10起点162文件：156不变，6个test/docs变化，无意外删除/修改。全时间/静态/工业入口AST恢复基线；原Transolver模型/脚本、data/train/normalizer/metrics/requirements字节不变。
- 三仓快照与许可已重新核对：Transolver根MIT、IPOT MIT，Air ODbL保留；LRSA无明确许可文件/metadata，不作MIT假定。数学附件独立文件仍未定位，全文/证明/8项理论检查未复验，不增加任何结构或损失。
- 五项重点已自行复查，未留下新的已知生产缺陷。剩余未验证是：真实数据读取完整性、图构建/采样/全评价、真实训练收敛与精度、epoch效率、远端环境、全默认GPU任务矩阵及理论附件。未commit/push、安装、下载数据或启动训练。

**本阶段结束，未执行下一阶段**。

## 独立KCDNO交付关联（2026-09-16）

独立K4–K10按用户连续授权完成，详见[KCDNO最终报告](KCDNO_IMPLEMENTATION_REPORT.md)。旧CDLNO/A阶段图与checkpoint规则保留，41份K0原权重回放精确一致；新家族接入不改变原数据/loss/时间循环。旧V2–V5任务resume未因此补齐，未启动真实训练。

用户另行要求的 `tran_evaluate/kcdlno/` 八数据集脚本已补充，复用现有KCDNO入口。
9语法+120分发检查、112条预览通过，173已有生产/脚本hash不变。帮助解析初测问题
已修复并清理失败目录，未执行真实训练；详见[KCDNO状态](KCDNO_IMPLEMENTATION_STATUS.md)
及[使用说明](../tran_evaluate/kcdlno/README.md)。旧CDLNO模型与启动脚本保持不变。
