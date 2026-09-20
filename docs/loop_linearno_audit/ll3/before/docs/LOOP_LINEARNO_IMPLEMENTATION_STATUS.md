# Looped LinearNO 实施状态

## LL2 — 2026-09-19

**唯一状态：PASS（点域AttnRes与原block body适配完成）。LL3–LL10 未执行。**

- A 范围：先写独立FP64/FP32 oracle并通过手算，再新增 `cdlno/linearno_loop/attnres.py` 的 `PointDepthAttnRes`、`body.py` 的 `LinearNOBlockBody`；没有完整loop模型、任务CLI/factory或checkpoint接线。[完整报告/公式映射](LOOP_LINEARNO_PRIMITIVES.md)。
- B 改动：新增3个原语包文件、独立oracle与2个测试文件、LL2报告/证据；仅更新已有研究状态及LL1“无完整模型”测试边界（允许原语目录存在）。顶层纯schema和所有旧生产/回归/launcher/monitor均不改。[源码增量](loop_linearno_audit/ll2/source.diff)。
- C 数学：单pseudo-query、eps1e-6、query0/scale1，每receiver恰2H参数；source softmax和raw value，无projection/来源数缩放/额外1/R。body仅调用旧ln/Attn/MLP，native/scaled/raw为内部语义；head显式finalize。不存在旧latent A/K或历史缓存。
- D 验证：**LL1+LL2共41/41通过，5.856s，无失败/错误/跳过**。AttnRes独立oracle/VJP/finite-difference、CPU FP64/FP32、本机CUDA FP32通过；60组block同权重forward/梯度/AdamW一步误差全0，dropout RNG精确相等。[数值汇总](loop_linearno_audit/ll2/numeric-summary.json)、[测试日志](loop_linearno_audit/ll2/loop-tests.log)。新首轮置换测试的逐位相等断言已按既定FP64阈值纠正，保留原日志，未改数学或放宽既定门槛。
- E 兼容：重跑LL0原153方法：147通过、2个历史文档冻结失败方法/4断言、4 CUDA skip、0error；24模块结果及失败文本/hash与已审查LL0精确一致，无新增失败。没有修改旧golden。[回归对照](loop_linearno_audit/ll2/regression-summary.json)。全量起点2005文件仅2项授权修改，其他2003项内容/分类不变，详见[冻结](loop_linearno_audit/ll2/end-freeze.json)。
- F 环境/边界：Python3.13.9、torch2.13+cu130、RTX5090 Laptop，小型合成CPU/CUDA检查；未访问真实数据，未训练任务或更改依赖。完整loop、任务接线、loop strict archive/resume、远端、性能/精度均未执行。五项自审通过，无新增需裁定冲突。

**本 LL2 阶段结束，未执行下一阶段。**

以下为LL1与LL0历史记录，其未执行阶段表述对应当时状态。

## LL1 — 2026-09-19

**唯一状态：PASS（独立配置/schema/metadata 合同与测试完成）。LL2–LL10 未执行。**

- A 范围：新增顶层 `linearno_loop/` 纯配置包；未实现 torch 模型，未接八任务 parser/factory/train/eval/launcher。完整交付见 [CONFIGURATION](LOOP_LINEARNO_CONFIGURATION.md)。
- B 改动：6个纯包文件、`tests/loop_linearno/` 的6个测试/支持文件、LL1文档与证据；只增量更新本研究状态，保留LL0内容。起点 [manifest](loop_linearno_audit/ll1/start-manifest.json)，末次 [freeze](loop_linearno_audit/ll1/end-freeze.json)。
- C 合同：family=`linearno_loop`，extension=`loop_linearno_v1`；P/C/R/S唯一拓扑，两个preset执行8次/存5或6套；三残差模式；AR点域eps1e-6/query0/scale1；RB receivers=2CR+1、LB=R。task base M从旧profile读取，默认×2、支持×1；显式actual M与multiplier互斥。旧A/K字段即使false也禁止混用。
- D 实测：**新24/24通过（3.490s），旧33/33通过（23.008s），0失败/错误/跳过**。新进程禁torch/NumPy/模型import仍通过metadata/配置/矩阵校验；Python/NumPy/Torch CPU全局RNG精确不变。日志与命令见[LL1证据](loop_linearno_audit/ll1/)。
- E 兼容：24原profile与LL0快照全等；旧parser默认、八任务Transolver同权重输出/keys/参数/checkpoint、LinearNO/history schema和monitor通过。全量冻结检查保护1975个起点文件（包括untracked/ignored），仅本状态允许变更；所有旧生产/测试/配置/launcher/monitor文件不变。LL0的2个历史文档断言失败方法仍是已解释历史事项，本轮没有修改golden，也未声称全仓测试全绿。
- F 机器交付：144主矩阵+144 rank×1+24 custom配置预览，paired本地seed0/1/2；[合同目录](loop_linearno_audit/ll1/contract-catalog.json)、[矩阵](loop_linearno_audit/ll1/configuration-matrix.json)、[合成metadata](loop_linearno_audit/ll1/synthetic-metadata.json)。class_path只是计划合同，无占位模型；所有预览不可用于训练。
- G 边界：meta先读、完整构造kwargs、结构不匹配逐字段报错、strict=True已固化并纯schema验证；实际torch加载、optimizer/backend RNG恢复、新模型共享/梯度/性能/训练全部NOT RUN，须后续授权实施。未改依赖，未运行真实数据/GPU/远端/训练，未commit/push。五项交付自审见配置文档，无新增待裁定冲突。

**本 LL1 阶段结束，未执行下一阶段。**

以下为已审查通过的 LL0 历史记录；其中“LL1未执行”描述当时状态。

## LL0 — 2026-09-19

**唯一状态：PASS（只读审计与冻结完成）。LL1–LL10 未执行。**

- 范围：当前 `main@5b991226c5354af3332b2f7306b370aef0950c79`，纯LinearNO/history/Transolver兼容基线、八任务调用链、固定论文/源码、三残差公式、最小后续接线与风险。详见 [REFERENCE_AUDIT](LOOP_LINEARNO_REFERENCE_AUDIT.md)。
- 文件：仅新增本状态、审计报告及 [LL0证据目录](loop_linearno_audit/ll0/)。生产/测试/golden/launcher/依赖/AGENTS/旧报告均未编辑。
- 初始冻结：1549 tracked + 2用户untracked + 361 ignored，全量size/SHA-256/分类；tracked diff为空。最终复核见 [end-freeze.json](loop_linearno_audit/ll0/end-freeze.json)。
- 实际测试：153方法，**147通过、2失败方法（4条历史冻结断言）、4 CUDA skip、0error**。两失败方法涉及README、path.sh、复现矩阵的已存在文档/注释增量，已保存完整差异并对照历史R9登记；未改测试/golden/容差。不是全绿声明。[逐模块命令与结果](loop_linearno_audit/ll0/regression-results.json)。
- 数值：六attention变体与官方CPU FP32/FP64误差0；Standard整模型FP32误差0，FP64最大1.88e-16以内；Air/Car整模型CPU官方误差0。独立oracle、输入/参数梯度、一步optimizer、strict往返通过；旧Transolver八任务、pure原生合成train/resume/eval与history核心59方法、monitor10方法通过。[数值证据](loop_linearno_audit/ll0/parity-summary.json)。
- 历史兼容：100个受保护文件hash不变；8处已有路由投影全等；pure来源hash保留。suffix>=1可隔离单次最终ln_3/mlp2。loop新增family需要显式适配记录器、精确provenance投影与逻辑执行级监测，不能只加factory。
- 设计真值：family=linearno_loop，extension=loop_linearno_v1；P/C/R/S唯一拓扑；两个preset均执行8次、存5/6套block；三mode仅sr_1_over_r / rb_attnres / lb_attnres_1_over_r；M按任务base×2，支持×1；无旧A/K。均为设计审计，尚无新模型实现。
- 环境：本机Python3.13.9、torch2.13.0+cu130、PyG2.3.1；本轮显式CPU。未运行真实数据、GPU、远端、长训练、精度或性能；未安装依赖、commit/push。

LL0 PASS 表示现状、公式、路由和冻结边界可追溯且无未解释的模型语义冲突，不表示研究效果、全仓测试全绿或后续实施已获授权。

**本 LL0 阶段结束，未执行下一阶段。**
