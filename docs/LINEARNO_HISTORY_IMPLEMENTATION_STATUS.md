# 当前交付：R6–R10 已完成

**PASS（2026-09-18，授权范围内）**。最终[实现报告](LINEARNO_HISTORY_IMPLEMENTATION_REPORT.md)、[远端八任务命令](LINEARNO_HISTORY_COMMANDS.md)、[可选诊断/性能](LINEARNO_HISTORY_DIAGNOSTICS.md)。R9返回R6修正旧parser独立导入回归并通过完整复验；未修改旧测试容差/golden。R10关键63方法62通过/1历史方法3断言失败；官方与RNG23方法19通过/4CUDA未运行。全库最终537通过/6历史失败方法11断言/35skip/0error，非全绿。八任务32原生合成闭环、20核心配置、160参数/成本项通过。真实数据/GPU/完整3seed/精度/epoch速度/SOTA未验证，无commit/push。已停止，没有后续阶段执行。

---

# LinearNO history 扩展实施状态

最新阶段：**R5 PASS（2026-09-18），完成 A/K 联合研究模型、内部 config/factory/checkpoint 与合成 20 配置闭环；未接八任务生产 launcher，未执行 R6。** 详见 [R5 报告](LINEARNO_HISTORY_R5_REPORT.md) 和 [验收证据](linearno_history_audit/r5/)。下列 R0–R4 记录保留为历史阶段证据。

## R0（2026-09-18）

**PARTIAL：只读审计和新增机制前冻结已完成；不进入 R1。**

已完成：

- 当前 checkout、branch/HEAD/tree/remote、用户 dirty/ignored 分类和全量 hash manifest；
- 当前 Standard、AirfRANS、ShapeNet-Car LinearNO 类、factory/parser/checkpoint/eval/launcher 调用链定位；
- 固定 Transolver、LinearNO、论文 v3、Attention Residuals v1、Kimi K3 来源及 blob/tree/hash 核对；
- 来源台账：A 不是 CDPA 或 Kimi 官方 AttnRes，K 没有官方实现，Transolver 核曲线不是 LinearNO 实验证据；
- 现有 CPU 无数据回归和 monitor 测试；
- 八任务缩小 synthetic fixture：逐 block 输出、最终输出、输入/参数梯度、AdamW 一步、strict round-trip、state_dict 和 RNG；双独立进程 JSON 一致；
- 新增研究证据目录 [linearno_history_audit/r0](linearno_history_audit/r0/)。

实际结果：纯 attention/Standard/Air/Car 数值 parity 与独立 oracle 通过；84 个 LinearNO 回归方法中 78 passed、4 GPU skipped、4 个历史冻结断言失败。monitor 测试 3 passed。失败详情和完整命令见 [REFERENCE_AUDIT](LINEARNO_HISTORY_REFERENCE_AUDIT.md)。

未完成且明确不在 R0：A/K 生产模块、配置/factory/CLI、训练/评估接入、诊断 effective-factor、checkpoint 扩展、真实数据、GPU、长训练、性能和论文精度。

## 阶段门

| 阶段 | 状态 | 说明 |
|---|---|---|
| R0 | PARTIAL | 基线关键数学 parity 通过；非数学冻结断言与 GPU 按授权边界未全绿 |
| R1 | NOT STARTED | 需要先审查 R0，冻结 A/K schema、配置和公平比较矩阵 |
| R2–R10 | NOT STARTED | 未执行 |

R0 没有修复测试失败、没有改变旧模型默认、没有注册研究 family。A0K0 仍是当前纯 LinearNO；后续任何新 family 必须保持旧类路径和 strict checkpoint 可用。

**本 R0 阶段结束，未执行 R1 或后续阶段。**

## R1（2026-09-18）

**PASS：研究规格、配置/metadata 合同、测试矩阵和公平初始化协议已冻结；未实现 A/K 数学模块，未接 launcher，未读取真实数据。**

先冻结 `history_conditioned_k_v1` 的三个具体选择：

1. 当前层基础 `to_k.weight` 的 M 行作为 K-conditioning 槽 query；
2. 全网络、所有 receiver/source/head 共享、且与 A 完全分离的无 bias `Uq_K/Uk_K/Uv_K`；
3. receiver layer × head 独立的 `tanh` gate，raw gate 严格零初始化。

本阶段没有收到改变这三项的决定，因此它们进入 R1 schema 和后续测试契约。

新增独立 `linearno_history/` 纯 schema 包。它不导入 torch、模型、factory、任务入口或 checkpoint loader，提供：唯一 A/K 字段解析、派生 `A0K0/A1K0/A0K1/A1K1` 签名、研究 metadata 校验、legacy checkpoint 拒绝猜测、strict load 规则、结构字段先验比较、运行目录 ID 和独立公平 seed 派生。没有把文件放进 `cdlno/linearno/*.py`，所以不改变既有 Standard resume 的源码哈希范围。

机器可读合同与 fixture 位于 [R1 evidence](linearno_history_audit/r1/)：JSON Schema、20 项核心矩阵、32 项八任务 wrapper/生产闭环矩阵、负面案例、公平初始化协议和受保护生产文件 hash。研究 metadata 只允许 A1K0/A0K1/A1K1，并要求 `architecture_extension=linearno_history_v1`；A0K0 仍使用旧 LinearNO schema/loader，旧 checkpoint 没有 `innovation_spec` 时只允许纯基线。研究 class path 是未来实现的稳定合同字段，当前不可导入，不作为占位模型。

新增 `tests/linearno/test_history_schema.py`，10 项通过；联合旧 `linearno.test_schema`、`linearno.test_profiles` 共 21 项通过。测试未构造任何 A/K 模型，未调用 factory、launcher 或真实数据。R1 受保护的 31 个生产文件与 HEAD 内容 hash 全部一致。

R1 的核心矩阵状态均为 `NOT_STARTED`，后续每个 L4–L8、A/K 组合必须完成构造、forward、backward 和 strict checkpoint round-trip 后才能更新。八任务真实生产闭环全部保持 `NOT_STARTED`。

**本 R1 阶段结束，未执行 R2 或后续阶段。**

## R2（2026-09-18）

**PASS：研究 core 与 forward-local raw-history context 已建立；A/K 数学机制、factory/CLI、benchmark launcher 和真实数据均未接入。**

### A. 范围与判定

新增的 `LinearNOHistoryCore` 只包装已经由纯 LinearNO 父类构造并初始化的独立 block 对象，因此不改变参数对象、`blocks.<index>` state-dict 键或初始化 RNG。每次调用在局部创建不可变 `RawHistoryContext`，只按深度保存 `C_raw=K^T V`，并在完整 block（输出投影、点残差、FFN、末层 head）完成后写入。R2 的研究 wrapper 仅供后续阶段内部使用；纯 A0K0 仍由既有类和既有 factory 返回。

### B. 文件与关键改动

- `cdlno/linearno_history/context.py`：不可变 tuple context，禁止 detach、跨调用保存和 pickle；校验 B/H/d_h、dtype/device。
- `cdlno/linearno_history/core.py`：显式 `AttentionFactors(Z, base_q_logits, base_k_logits, Q, K, V, C_raw, QC_raw)`、逐层 trace observer、原六种 attention 变体的等价计算和原残差顺序。
- `PDE-Solving-StandardBenchmark/model/LinearNO_History.py`：Standard 内部 wrapper。
- `cdlno/linearno_history/models.py`：AirfRANS 与 ShapeNet-Car 内部 wrapper，保留真实 Data/tuple 合同和单图边界。
- `tests/linearno/test_history_core.py`、`fixtures/history_r2_temporal.json`：R2 合成验收。

原纯 LinearNO、任务 factory、parser、训练/评估、数据 loader、checkpoint helper 和 monitor 均未修改；新增文件仍是工作树 untracked，未 commit/push。

### C. 公式、时序与代码映射

对每个 block，`attention_factors()` 复用其现有 `in_project_x`、`to_q/to_k/to_v`、任务温度 clamp 和 softmax 轴（Q 沿 rank，K 沿点 N），得到 `C_raw = einsum(K,V)` 与 `QC_raw = einsum(Q,C_raw)`。`LinearNOHistoryCore.forward()` 的 `context.before_block(index)` 只能读之前的 tuple；`context.after_block(index, factors.C_raw)` 位于 FFN/末层 head 之后。没有 M×M/N×N attention、history dropout、K 修正或 gate。

### D. 实际验证

核心命令（CPU、无数据）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LINEARNO_R2_REPORT=/home/hwz/CDLNO-artifacts/linearno-history-r2-before-13jm89as/core-results.json \
  python -B -m unittest linearno.test_history_core -v
```

结果为 **11/11 passed，37.074 s**。六种变体（plain/temp/conv/conv_temp/AirfRANS/ShapeNet）、float64/float32、train/eval、L=4 与 L=5–8 均覆盖同权重逐 block/final 输出、MSE、输入及全部主干梯度、AdamW 一步和 strict state-dict round-trip。独立 oracle 记录了 Q/K/V/C_raw/QC_raw 和输出的最大/平均误差，CPU double 门限 `1e-12/1e-10`；相同运算顺序的 no-op parity 为 `atol=rtol=0`。

同一命令还验证：真实 `torch_geometric.data.Data` 的可变 N、AirfRANS/Car 多图拒绝、三个任务工作目录的新 Python 进程 strict 恢复；温度边界与 AirfRANS dead temperature；observer 异常、连续 backward、不同 batch/N、嵌套重入、context 不 detach/不序列化；NS 10 次 teacher-forcing/prediction-feedback 和 Plasticity 20 次 optimizer/1 次 scheduler 的时间 fixture。改变 NS 未来真值不改变 prediction，证明没有未来标签泄漏。

旧回归命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_history_schema linearno.test_schema \
    linearno.test_profiles linearno.test_attention_parity \
    linearno.test_attention_structure linearno.test_standard_model \
    linearno.test_standard_structure linearno.test_airfrans_model \
    linearno.test_shapenet_model linearno.test_legacy linearno.test_rng \
    monitor.test_monitor -v
```

结果为 **66 tests：59 passed，4 CUDA skipped，3 个已知 R0 冻结断言失败**（README、path.sh、旧 reproduction-matrix 前缀/hash）。其余旧 LinearNO 数值、结构、八任务 checkpoint/选择、随机流及 monitor 方法通过；完整输出和 JSON 位于 `docs/linearno_history_audit/r2/` 的外部归档 `/home/hwz/CDLNO-artifacts/linearno-history-r2-before-13jm89as/`。本机 CUDA 测试按授权未运行。

### E. 兼容性证据

纯模型 state-dict key/shape/count、初始化 RNG、forward RNG、输入/参数梯度和 AdamW 更新均与同权重纯模型一致；研究 wrapper 的参数对象与纯模型独立但逐键严格相同。A0K0 factory 选择仍为纯 `Model`，旧模型和 monitor 文件 hash 未因 R2 改动。新研究 wrapper 未注册为生产 family，不能被误认为已完成 A/K 训练接入。

### F. 未运行与风险

未运行真实数据、真实训练、GPU/AMP、A/K 实际历史读取、R3–R10、benchmark CLI、研究 checkpoint metadata loader。observer 是显式测试接口；默认 forward 不保存 trace。后续生产接入必须先实例化 A/K 开关并保持本 R2 的 raw 时序，不得把当前 no-op wrapper 当作研究模型完成。

### G. 阶段状态

**PASS**。本 R2 阶段结束，未执行 R3 或后续阶段。

## R3（2026-09-18）

**PASS：latent-summary AttnRes A-only 内部模块与三套研究 wrapper 已实现和验证。K-conditioning 保持不存在，生产任务 CLI/factory/launcher 未接线。**

新增 `cdlno/linearno_history/attnres.py`，包含全网络共享、跨 head 共享的无 bias Wk/Wv，以及 receiver 独立的 Q/O、last-dimension RMSNorm、零 w、零标量 gamma。严格逐历史 token-softmax，再对真实来源+零 null 做 source-softmax，输出 `C_raw + gamma*H`。p=.1 固定，训练 sample×source mask 在 Cross/评分后应用；singleton 保留、eval 无采样、无 inverted scaling、全 mask 返回原 C。

原纯 LinearNO 全部保持原样；A-only 类在原主干构造结束后用独立 feature_seed 初始化 A，不推进主干 RNG。研究 core 将 A 插入 raw 与原 Q 重建之间，缓存仍仅 raw，并在完整 block 后写入。所有 baseline key 保持，新增 A key 只存在于内部 A-only 类。没有生产 A0K0 包装替换，也未生成研究 resume/eval 协议。

验收：**13/13 新测试通过，9.513 s**。包括独立 oracle FP64/FP32、全部梯度/AdamW一步、有限差分、手算、三种置换、B/head隔离、mask时序/广播、两步隔离VJP、raw生命周期、六变体×L4/L8完整模型零门严格parity、三个原工作目录新进程strict权重恢复。最大 oracle 误差：FP64 所有指标4.44e-16；FP32 1.79e-7；零门完整模型输出/主干梯度/更新误差0。

旧回归含 R2：**77 个方法：72通过、4 GPU跳过、1既有失败方法中的3个subtest断言，0 errors，79.383 s**。失败仅为 R0 已知 README/path.sh/旧 reproduction matrix 冻结快照；纯模型数学、官方parity、八任务旧Transolver fixture、随机流和monitor测试均通过。按方法与subtest区分计数后，R2历史日志的正确计数是61通过/4跳过/1失败方法（3断言），不是早先状态文字中的59通过。

R3 初始1625文件包含1242 tracked、53 untracked、330 ignored；原tracked和ignored均不变，原untracked只有四个研究源码及两个研究文档获本阶段增量更新。详情见 [冻结检查](linearno_history_audit/r3/freeze.json)。未改变其他模型、数据、配置、schema、旧测试、任务训练/评估、checkpoint helper、monitor 或依赖。

未执行：GPU/AMP/compile、远端Torch2.11/cu128、真实数据和训练、A生产checkpoint metadata/CLI、K、联合模式、研究monitor接入、R4–R10。R1 20组合及八任务生产闭环矩阵仍未升级。完整 A–G 报告、实际命令与数值见 [R3 report](LINEARNO_HISTORY_R3_REPORT.md) 和 [R3 evidence](linearno_history_audit/r3/README.md)。

**本 R3 阶段结束，未执行 R4 或后续阶段。**

## R4（2026-09-18）

**PASS：history_conditioned_k_v1 已独立实现和验证，不依赖 A，未接 benchmark launcher。**

新增 `cdlno/linearno_history/history_k.py`：无参数LN0、全网络共享无bias Uq/Uk/Uv、真实基础to_k.weight槽行query、全部旧raw token bank一次softmax、Z×G双线性修正并沿N中心化、receiver×head零初始化tanh门。第一层无gate且完全bypass；先修正raw K logits再用原温度/softmax_N，Q/V/重建保留。

研究core/三套wrapper加入内部K-only路径，pure类和生产factory不改。K-only不构造/调用/import A模块；A+K同时请求明确拒绝，留给R5。R3 A模块、R2 context、R1 schema和既有测试保持原样。

新增独立oracle、13项测试及R4证据；**13/13通过，11.026 s**。FP64全前向/梯度/step最大误差2.11e-15，FP32 9.54e-7；六变体L4 eval/train及L8 eval共18例zero gate逐层输出、loss、输入/主干梯度、主干AdamW一步与RNG完全一致。非零门下隔离eta×Delta对Z/基础slot weight/Uq/Uk/Uv/旧raw的VJP全部finite非零；当前/未来C不参与。实际query对象、单次因果更新、token置换、B/head隔离、M1/32/64含M>N、连续forward/异常/重入、三项目fresh-process strict checkpoint且不import A均通过。

首轮仅有测试采集器误报：共享query广播矩阵被PyTorch分派为mm，而旧采集器只捕获bmm。补齐mm+bmm和真实softmax/Delta形状检查后通过；模型公式及容差未改。R4交互为M×S和N×M，无N×N；常规投影/聚合的M×d_h、S×d_h亦按真实情况记录。

旧回归含R2/R3：**90方法：85通过、4 GPU跳过、1历史失败方法含3个subtest断言，0 errors，92.242 s**。失败仍是R0已知README/path.sh/旧reproduction-matrix快照，不是新增回归。原1242 tracked与330 ignored全部不变；研究增量相对本轮1640文件快照见 [freeze](linearno_history_audit/r4/freeze.json)。

未执行：R5–R10、A+K联合、生产config/factory/metadata-first研究resume、八任务CLI/loader闭环、研究monitor、真实数据/训练、GPU/AMP/compile、远端环境、性能/精度。没有新架构决定待裁定；R1核心20配置和八任务生产矩阵仍未升级为完成。

完整A–G报告、命令和数值：[R4 report](LINEARNO_HISTORY_R4_REPORT.md)、[R4 evidence](linearno_history_audit/r4/README.md)。

**本 R4 阶段结束，未执行 R5 或后续阶段。**


## R5（2026-09-18）

**PASS：仅内部联合集成完成，未接任务生产 CLI/launcher。**

- 顺序严格为旧 raw → K correction → 当前 `C_raw` → A 读取同一旧 raw tuple → `C_tilde` → 原 Q 重建 → 原 projection/point residual/FFN/final LN/head → append 当前 raw 一次。
- 三套 wrapper 均有独立 joint 类；内部 config/factory 对四组合只构造相应原类或真实 A/K 模块。A0K0 仍为原 pure 类，没有新增参数、context、feature 属性或 checkpoint metadata；其原 helper 和 schema 原样委托。
- 研究 checkpoint 使用完整 innovation/config/profile/data/objective/evaluation/provenance/normalizer/resume/ensemble metadata，`architecture_extension=linearno_history_v1`，`implementation_version=r5-internal-v1`。metadata-first、逐字段结构拒绝、全 key/shape/dtype 预检、`strict=True`；不加载外部 whole-object pickle，不实现 baseline→research resume。
- 新增 11/11 测试通过；20 个 A/K×L4–8 配置的完整训练一步/save/独立新进程 reload/eval 通过，末层均完整执行。六变体 18 个门分解、12 个 omitted/explicit A0K0 train/eval 精确回归、四组合下一步恢复均 exact。
- 两机制数学源码/context 未改；联合 oracle max abs：K `2.776e-17`，A `4.337e-19`（CPU float64 门限 `1e-12/1e-10`）。train 的 A dropout 固定 .1，明确会消耗额外 RNG；仅 A0K0 在 train/eval 都维持旧 RNG。
- 完整回归 114 方法：109 passed、4 GPU skipped、1 个历史 freeze 方法的 3 个既有断言失败、0 errors，114.623 s。失败集合与已审查 R4 相同，没有新增失败或容差放宽。
- 全部 1242 tracked 文件未变；原 LinearNO/Transolver/其他模型、八任务入口/factory/launcher/数据/旧 checkpoint、LINEARNO/monitor、330 个既有 ignored 文件未变。研究变更均逐个相对真实 R5 前快照校验。

[报告](LINEARNO_HISTORY_R5_REPORT.md)含实际改动、内部接口、完整证据、已审查的冻结例外与未验证边界。研究 32 项八任务生产 wrapper/闭环仍 NOT_STARTED；真实数据、GPU/AMP/远端、精度/收敛/epoch 效率、R6 及以后均未执行。

**本 R5 阶段结束，未执行 R6 或后续阶段。**


## R6（2026-09-18）

**PASS**。四静态任务研究parser/factory/原生train-resume-eval及可视化接入完成；4/4新测试、80 preset计数、16真实空间尺寸合成闭环均通过（356.901s），连续/恢复权重与RNG/batch/预测exact。旧核心114方法109通过/4GPU跳过/1历史失败方法3断言；静态旧7方法6通过/1历史path.sh失败；其他模型69方法61通过/6资源跳过/2历史AST方法5断言，已在修改前快照复现。六exp、所有模型数学及monitor不变；共享路由精确投影证明旧源码指纹相同。详见[报告](LINEARNO_HISTORY_R6_REPORT.md)、[命令](LINEARNO_HISTORY_COMMANDS.md)及r6证据。真实数据/GPU/完整实验未运行。按用户连续授权继续R7，无需再次授权。


## R7（2026-09-18）

**PASS**。NS/Plasticity生产接入，40真实preset构造/计数、8合成原生闭环；3/3新测试通过282.733s。mask、batch、查询、所有状态、连续/恢复/新进程eval exact。NS60forward→6更新，Plasticity120forward→120更新/6scheduler；逐forward历史0..L-1。原L5 5/5通过63.932s；R6四题4/4及16闭环通过346.428s。六exp和模型数学保持相同；公平运行仅隔离原Plasticity collate数据随机流。真实数据/GPU未运行。见[报告](LINEARNO_HISTORY_R7_REPORT.md)和r7证据；连续授权进入R8。


## R8（2026-09-18）

**PASS**。两工业任务四组合接入；40真实preset计数、8单成员合成PyG闭环、4组合双成员ensemble两类恢复边界通过。4/4方法654.323s；额外metadata提前拒绝1/1方法6.477s。旧工业17/17通过200.609s，Standard预检4/4通过5.809s。纯类/原objective/force/sampling语义不变；两类旧源码指纹和共享helper核对通过。真实VTK/力系数、GPU、真实训练均未运行。见[报告](LINEARNO_HISTORY_R8_REPORT.md)。按连续授权进入R9。


## R9 检出 R6 导入边界回归，返回 R6 返修

综合回归发现旧 Standard parser 在隔离环境下被研究包提前导入破坏。已在 R6 前快照证明是新增问题；已将导入移入显式LinearNO分支。返修报告 `LINEARNO_HISTORY_R6_CORRECTION.md`，等待完整16闭环回归；R9尚未验收。KCDNO Air旧AST断言在同一pre-R6快照失败，属历史漂移。


## R6 返修完成，恢复 R9

**PASS**，8/8方法381.743s；旧parser独立导入恢复，80正式构造/16完整合成闭环再次通过。详见`LINEARNO_HISTORY_R6_CORRECTION.md`。未改测试或模型数学。R9初次 sweep 中新增的该失败保留原日志，并以本返修结果替代最终判定。


## R9（2026-09-18）

**PASS（本阶段）**。578方法综合扫描，返回R6修复新增导入回归并通过8/8复验；最终537方法通过、6历史方法11断言失败、35资源/授权跳过、0错误。32八任务四组合原生合成闭环、核心20、160正式计数；48诊断无扰动、24理论/实际MAC、50性能/对照行、480命令预览。GPU/数据/完整精度仍NOT RUN。详见[报告](LINEARNO_HISTORY_R9_REPORT.md)、诊断文档及r9证据。按连续授权进入R10，只读终审。

## A1K0 无 history dropout 增量（2026-09-18）

**PASS（增量配置；未运行真实数据）**。保留原 A1K0 默认 `p=0.1`，新增显式
`linearno_latent_attnres=1 + linearno_history_k_conditioning=0 +
linearno_attnres_history_dropout_p=0`。该配置仍构造完整 A-only AttnRes，
只跳过训练时的 history-source mask；A0K0、A0K1、A1K1(p=.1)及纯
LinearNO 路径未改变。A1K1(p=0)和所有非 A 配置的 dropout 参数仍在解析阶段拒绝。

修改范围为 history schema/config/factory、Standard 与工业 A/joint wrapper、
AttnRes dropout 参数及针对性测试；没有改动任务数据、损失、训练循环、旧
factory 或 launcher。p=0 的 `model_spec.constructor_kwargs` 显式保存
`attnres_history_dropout_p=0.0`，运行签名含 `__nodrop__`，研究 checkpoint
继续 metadata-first、逐字段校验和 `strict=True`。

验证：编译检查通过；A1K0 p=0 的 Standard/AirfRANS/ShapeNet 构造和
`dropout_p` 传递通过；A1K0 p=0 的 strict checkpoint round-trip、p=.1
checkpoint 互拒、A1K1 p=0 拒绝、训练 RNG 不额外消耗通过；新增/相关测试通过。
未运行真实数据、GPU、远端环境或完整训练评估。完整命令见
[LINEARNO_HISTORY_NO_DROPOUT.md](LINEARNO_HISTORY_NO_DROPOUT.md)。
