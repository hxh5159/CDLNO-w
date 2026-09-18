# LinearNO history 扩展研究矩阵（R0 基线）

最新更新：**R5 内部联合集成 PASS**；当前完成范围见文末 R5 表与独立证据。R0–R4 的逐阶段文字保留为历史记录。

本表把需求、当前代码和独立证据分开。R0 只读；“未实现”是预期阶段边界，不是缺陷修复任务。

| 规格/问题 | 当前文件与符号 | 独立证据/来源 | R0 状态 |
|---|---|---|---|
| Standard `Q/K/V`, `K^T V`, block loop | `cdlno/linearno/attention.py:LinearNOAttention`; `PDE.../model/LinearNO.py:LinearNOBlock/Model` | `attention-parity.json`, `standard-parity.json`, `standard-structure.json` | PASS |
| plain/temp/conv/conv_temp | `LinearNOAttention.variant`; Standard `Model` 选择 `linearno_variant` | L2/L3 inherited tests and source AST | PASS |
| Air absolute M/dead temperature | `cdlno/linearno/airfrans.py:AirfRANSLinearNO` | `air-parity.json`, Air source parity | PASS |
| Car `M=key_ratio*d_h`/拼写温度键 | `cdlno/linearno/shapenet.py:ShapeNetLinearNO` | `car-parity.json`, PyG synthetic tests | PASS |
| 八个完整 block、末层 head | Standard/Air/Car `blocks` and output code | fixture hooks and structure report | PASS |
| 纯模型 checkpoint strict round-trip | `cdlno/linearno/checkpoint.py`, `schema.py` | L1/L4/L8 inherited tests; R0 fixture replay | PASS |
| 旧 Transolver 选择/默认/存档不变 | `model_dict.py`, legacy task branches | old-l0-freeze-comparison + legacy suite | PASS（数学/路径）；冻结断言 PARTIAL |
| A0K0 不创建新参数/缓存 | 当前没有 history family | source search and state_dict fixture | PASS（基线现状） |
| A latent-summary AttnRes | 尚不存在；未来研究模块 | AttnRes v1 Eq/README only | NOT IMPLEMENTED（R0 边界） |
| K history-conditioned compression | 尚不存在 | Fixed LinearNO/AttnRes/Kimi trees have no implementation | NOT IMPLEMENTED（原创设计） |
| raw cache pre-AttnRes、局部 tuple、可反传 | 未来 R2/R3/R5 合同 | staged prompt §3.5 | UNVERIFIED |
| A 每份历史内部 cross-attention + source softmax | 未来 R3 | staged prompt §3.2 | UNVERIFIED |
| K 全旧 raw token bank、LN0、Uq/Uk/Uv | 未来 R4 | staged prompt §3.3 | UNVERIFIED |
| K `Delta[N,M]` 点×槽位并沿 N 中心化 | 未来 R4 | staged prompt §3.3；需独立反例 | UNVERIFIED |
| 两机制顺序 | 未来 R5 | staged prompt §3.4 | UNVERIFIED |
| 四开关 A0K0/A1K0/A0K1/A1K1 | 当前 parser 无新字段 | staged prompt §1 | NOT WIRED |
| monitor 基础 `rho(P_i,P_j)` | `monitor/runtime.py`, `monitor/kernels.py` | monitor 3 tests passed | PASS（基础诊断） |
| monitor 对 K 后有效因子 | 当前 runtime 只重建基础 Q/K | source review | NOT SUPPORTED |
| Standard resume source hash | `standard_entry.py:provenance/StandardRun.prepare` | source audit | RISK TO RESOLVE |
| 不新增旧 family metadata | old schema and current paths | schema source audit | PASS（现状） |

## 任务入口与数据合同

| 任务 | 真实入口/模型 | 输入→输出 | 当前主要协议 |
|---|---|---|---|
| Airfoil | `exp_airfoil.py` → structured LinearNO | `[B,N,2]`, fx channel → `[B,N,1]` | `H=221,W=51`, conv_temp, 原 normalizer/loss/OneCycle |
| Darcy | `exp_darcy.py` → structured LinearNO | grid+fx → field；decode 后导数分支 | `H=W=85` 常用 downsample；zero-padding derivative |
| Elasticity | `exp_elas.py` → irregular LinearNO | point xy、`fx=None` → stress | `N=972`, temp，Cosine |
| Pipe | `exp_pipe.py` → structured LinearNO | `[B,N,2]`/field → output | `H=W=129`, conv_temp，OneCycle |
| Navier–Stokes | `exp_ns.py` → plain LinearNO | past 10 frames → 1 frame；eval 回填 | 每 forward 重建网络，不缓存时间 latent |
| Plasticity | `exp_plas.py` → conv LinearNO | field + `T[B,1]` → 4 channels | 每 batch 20 次时间查询/optimizer，scheduler 原节奏 |
| AirfRANS | `main.py/main_evaluation.py` → `AirfRANSLinearNO` | PyG Data `x[7],pos[2]` → `[N,4]` | 单图/可变 N，保留采样与 scatter |
| ShapeNet-Car | `main.py/main_evaluation.py` → `ShapeNetLinearNO` | `(Data,None)`, `x[7]` → `[N,4]` | 单图/可变 N，图字段给旧 loader/eval 保留 |

任务命令的共同链为 `tran_evaluate/linearno/*` → `_common.sh` → 各子项目入口 → 新 parser/factory → LinearNO wrapper → 原 loss/eval → schema/checkpoint。R0 未导入有顶层数据副作用的 exp/main，只按 AST、入口路由和合成边界检查。

## 来源与设计决策矩阵

| 机制 | LinearNO 官方 | AttnRes/Kimi | 本项目研究设计 | 处理 |
|---|---|---|---|---|
| 点域线性注意力 | 有 | 无关 | 基线 | 保持旧实现 |
| depth source softmax | 无 | AttnRes/Kimi 有 | A 的灵感来源 | 不能称官方 LinearNO |
| latent summary cross-attention | 无 | 固定材料没有 | A 原创适配 | R3 单独实现/验证 |
| raw summary 改变当前 K logits | 无 | 无 | K 原创 | R4 单独实现/验证 |
| `gamma=0`、`w=0`、history dropout | LinearNO 无 | AttnRes 有零 pseudo-query，但语义不同 | A 的适配选择 | 需 R1 冻结 |
| `tanh` per-head K gate | 无 | Kimi 没有 | K 原创选择 | 需 R1 冻结 |
| P 核相似度 | 无 LinearNO 实验证据 | 无关 | 本项目诊断 | 不作性能/塌缩结论 |

## R0→R1 风险转交

1. **基线测试不是全绿**：失败是历史文档/path 快照断言，不是数值 parity；R1 需决定是否更新测试基线，R0 不自行处理。
2. **Standard 源码哈希**：`standard_entry.py` 将多个 `cdlno/linearno/*.py` 纳入 `source_sha256`，新增共享研究文件可能使旧 resume 拒绝。后续应将 family/source scope 分离，而不是放宽 strict 或删哈希。
3. **监测接口**：当前 monitor 从输入重建 base Q/K；K 开启后需要显式 effective Q/K/factor hook，不能复用旧重建逻辑。
4. **A/K 尚未冻结实现细节**：R1 必须复述 `to_k.weight` 槽 query、共享 `Uq/Uk/Uv`、per-head `tanh` 零门等选择；R0 不替用户变更。

**本 R0 阶段结束，未执行 R1 或后续阶段。**

## R1 冻结合同

R1 已将以下字段确定为唯一 feature 真值：`linearno_latent_attnres`、`linearno_history_k_conditioning`（均为严格 JSON bool）和 `linearno_attnres_history_dropout_p`（默认 null；A 开启时解析为 0.1；生产模式只允许 0.1，p=0 仅供内部 oracle/parity）。`feature_signature` 只能由两个 bool 派生，旧命令缺省等于显式 A0K0。

`history_conditioned_k_v1` 的首版低层选择已经冻结：基础 `to_k.weight` 行作槽 query；全网络共享且与 A 分离的 `Uq_K/Uk_K/Uv_K`；按 receiver layer × head 的 `tanh` 零门。R1 没有收到用户改变选择的决定，因此状态为 PASS。

研究 metadata 的 stable extension 是 `family=linearno_history`、`architecture_extension=linearno_history_v1`、`innovation_spec.schema_version=1`，仅用于 A1K0/A0K1/A1K1。它记录研究 class path、基础 variant、task variant、L/H/d_h/M、温度语义、raw-cache timing/storage、A 的 per-head Cross/d_m/source-softmax/null/gamma/dropout、K 的 row-query/token-bank/point×slot/centering/tanh gate、constructor hyperparameters 和 schema versions。A0K0 不写这些字段。

研究加载协议为 metadata-first、逐结构字段比较、`strict=True`；A/K、L、variant、M、head_dim、schema、class path 或 constructor kwargs 不匹配时，构造和权重加载前失败。旧 LinearNO metadata 缺 `innovation_spec` 时只能走 legacy 基线；研究 metadata 缺 spec 不猜。baseline→research 不是 resume，只能由未来单独授权的 `init_from_baseline` 流程处理。

运行目录格式固定为 `{task}__{protocol_profile}__L{layers}__A{0|1}K{0|1}__seed{seed}`。官方 profile 的数据、objective 和 metric 语义没有被改变；研究 checkpoint 才记录 extension，A0K0 签名仅进入外部目录/manifest。

机器可读文件和测试映射见 [R1 evidence](linearno_history_audit/r1/README.md)。

## R2 evidence: research core and local raw context

| R2 requirement | File/symbol | Independent check | Status |
|---|---|---|---|
| Preserve pure class and initialization | `PDE.../model/LinearNO.py:Model`; `LinearNO_History.Model` | same-weight state/RNG and optimizer parity | PASS |
| Expose Z and base Q/K logits | `cdlno.linearno_history.core:attention_factors` | independent reference and projection assertions | PASS |
| Expose Q/K/V, C_raw, QC_raw | `AttentionFactors` | double oracle, normalization and factor-shape checks | PASS |
| Local authoritative cache only C_raw | `RawHistoryContext.raw` | identity, no-detach, pickle and weakref checks | PASS |
| Read prior depth and append after block | `before_block`/`after_block` | FFN/head→append events; sizes 0,1,2,3 | PASS |
| No M×M/N×N attention | `attention_factors` einsum path | TorchDispatch shapes for N=15, M=8, d_h=4 | PASS |
| Standard/AirfRANS/ShapeNet wrappers | `LinearNO_History.py`, `models.py` | real contracts and fresh-process strict load | PASS |
| NS/Plasticity temporal isolation | `history_r2_temporal.json` | 10/20 call fixtures and future-label counterfactual | PASS (synthetic only) |
| A/K operators, dropout, gates | intentionally absent in R2 | source inspection and invalid-flag tests | NOT IMPLEMENTED (scope) |
| Production factory/CLI/checkpoint | intentionally unchanged | legacy parser/factory/checkpoint regressions | NOT WIRED (scope) |
| Real data/GPU/AMP | not authorized | no data or CUDA execution | NOT RUN |

The R2 implementation is a mathematical no-op research core. It is evidence for a later A/K insertion point, not evidence that either research mechanism or benchmark training is implemented.

## R3：创新 A 的实现与独立证据

| 规格 | 文件/符号 | 真实证据（test_latent_attnres） | 状态 |
|---|---|---|---|
| head内独立Cross，d_m=d_h，K/V全局共享 | `LatentSummaryAttnRes.to_k/to_v`，`receivers.*.to_q/to_o` | 独立 oracle/gradcheck；B/head隔离；L4–8共享对象与参数数目 | PASS |
| 每份历史内部单独token-softmax | `_evaluate` 的逐history循环 | 不等长历史oracle、手算、token置换；L4执行6次Cross softmax | PASS |
| RMSNorm末维/eps1e-6/scale-only，w=0 | `SummaryRMSNorm`、`AttnResReceiver.w` | 独立逐元素oracle、参数清单、两阶段scale梯度 | PASS |
| sources仅历史+固定零null | `_evaluate` 的 scores/null/alpha | 手算3与7的均值、alpha轴、全mask、source+mask置换 | PASS |
| C_raw+gamma H，原Q重建 | `AttnResReceiver.gamma`、`core.forward` | 手算20/3；零门逐层/梯度parity；to_out前hook核对Q C_tilde | PASS |
| p=.1，sample×source，singleton/eval/全mask | `_evaluate` dropout region | 采样时机/次数/阈值/广播、null=1、无inverted scaling | PASS |
| raw pre-A权威缓存，不detach | 未改的 `RawHistoryContext`、core.after_block | raw identity≠fused、连续backward/B变化/异常恢复；隔离raw VJP | PASS |
| 主干与feature初始化隔离 | A构造的CPU fork_rng | 同seed主干/RNG完全一致；A seed重现与变化 | PASS |
| A-only三套内部wrapper及strict state | `AttnResModel`、`AirfRANSAttnResModel`、`ShapeNetAttnResModel` | 六变体×L4/L8，三个fresh-cwd进程加载 | PASS（合成） |
| pure LinearNO/Transolver/其他旧模型保护 | 原文件未改；pure factory仍旧类 | 1242 tracked hash冻结；旧数学/CLI/fixed-weight/monitor回归 | PASS（本轮无变化）；历史文档冻结失败保留 |
| K/联合、两bool生产选择、metadata-first研究resume | 未实现/未接线 | R1矩阵和schema未变 | NOT IMPLEMENTED / NOT WIRED |
| 真实数据、GPU/AMP、远端、性能、精度 | 未运行 | 授权范围为本地CPU合成R3 | NOT RUN |

R3参数增量为 `2*d_h²+(L-1)*(2*d_h²+2*d_h+1)`；不按历史边创建参数。不采用历史投影缓存，每次由raw和共享投影重算。数学及最终证据见 [R3 report](LINEARNO_HISTORY_R3_REPORT.md)。不将R3 A-only内部验收标作R1全20配置或八任务生产闭环完成。

## R4：独立 K-conditioning

| 规格 | 文件/符号 | 独立证据（test_history_k） | 状态 |
|---|---|---|---|
| LN0末维keepdim/unbiased=False/eps1e-6/无参数 | `history_k.ln0` | 显式平方差oracle、常数/单通道有限、有限差分 | PASS |
| query=本层基础to_k.weight行 | `HistoryConditionedK.forward:E`、`core.attention_factors` | 实际参数对象hook、base weight隔离VJP；无slot参数 | PASS |
| Uq/Uk/Uv全局共享/无bias/不依赖A | `uq/uk/uv` | 参数key/计数、各receiver hook对象相同、fresh-process无A import | PASS |
| all-old-raw bank，总S轴一次softmax | `bank/attention/G` | 不等长历史oracle、跨来源联合token置换、bank长度8/16/24 | PASS |
| Delta=Z G^T，减mean_N | `uncentered/delta` | n/m方差、mean_N≈0、slot-bias严格无效反例、隔离Z/raw梯度 | PASS |
| per-receiver/head tanh零门，温度前相加 | `raw_gates`、`combined_k_logits` | 六变体三温度边界oracle、Q/V不变、Air dead无梯度 | PASS |
| first layer完全bypass | `forward:index==0` | 同一base对象、0投影/matmul/softmax/RNG、无gate0 | PASS |
| raw因果/不detach/不跨调用 | 原 `RawHistoryContext`、core loop | 指定旧raw VJP、前置history tensor identity、连续backward/B/N/异常/重入 | PASS |
| 六变体K-only内部模型 | 三套 `*HistoryKModel` | 18项zero gate模型案例、M1/32/64、strict往返、三cwd新进程 | PASS（CPU合成） |
| 交互形状无N×N | K模块matrix/softmax | mm+bmm dispatch，M×S、N×M；合法M×d_h聚合单列 | PASS |
| R3 A、纯LinearNO和旧模型 | A文件/原1242 tracked不变 | R2/R3与旧parity/Transolver/monitor回归；历史3断言保留 | PASS（本轮无新增回归） |
| A+K、内部factory/metadata和benchmark CLI | R4拒绝combined；未接生产 | R1配置/生产矩阵未改 | NOT IMPLEMENTED / NOT WIRED |
| GPU/AMP/compile、真实数据、远端、性能/精度 | 本轮未运行 | 明确CPU合成范围 | NOT RUN |

K参数增量为 `3*d_h²+(L-1)*H`，无per-edge/slot参数。当前没有历史投影缓存，全部由forward-local raw重算。K-only不会读取/修改A参数、cache或mask。完整数值与冻结见 [R4 report](LINEARNO_HISTORY_R4_REPORT.md)；本轮不把独立K结果称作A+K联合或八任务生产验收。


## R5 联合集成与内部闭环

| R5 要求 | 实现/测试 | 实际结果 |
|---|---|---|
| K-before-raw-before-A，raw only、block 后写一次 | core + 六变体 `test_joint_order_raw_once_residual_once_independent_oracles` | PASS；共享旧 tuple identity、顺序/残差/head 与独立 oracle |
| 四开关与原类隔离 | 内部 config/factory allowlist | PASS；A0K0 原类；A-only/K-only 无对方模块；其他 family fail fast |
| 完整研究 metadata、signature、strict 重建 | 新 `checkpoint.py` + on-disk negative checks | PASS；metadata 先校验后构造/权重读取；旧 schema/loader 原样委托 |
| 20 配置：4组合×L4–8 | `test_20_config_complete_block_matrix_fresh_process_reload` | PASS；全部构造/forward/loss/backward/optimizer/save/独立进程 strict reload/eval |
| 完整末层 | 所有配置 attention/to_out、FFN、final norm/head hooks | PASS；L/L/1/1 |
| 旧命令缺省与显式 A0K0 | 六变体×train/eval，原 helper 读写/metadata/manifest/RNG | PASS；class/键/shape/value/输出/梯度/optimizer/RNG 逐值一致 |
| 联合门分解 | 六变体×baseline/A-only/K-only，显式逐键复制 | PASS；eval exact，未互载不同结构 checkpoint |
| 两门同时非零的梯度 | 六变体 joint forward/backward | PASS；两套新增参数梯度全 finite，参数对象不共享 |
| train dropout 随机流边界 | A/K/AK 与 baseline RNG 对比 | PASS；A 的 .1 dropout 消耗 RNG，K-only 无 A 调用 |
| 恢复训练下一步 | 四组合含原输出 dropout + A history dropout | PASS；state/loss/lr/optimizer/scheduler/Python/NumPy/Torch/独立 generator exact |
| 错误 checkpoint/state 身份 | 跨组合12对、L/M/variant/heads/schema及完整metadata异常 | PASS；权重读取/`load_state_dict` 前拒绝 |
| 八任务研究生产接线 | 原 parser/factory/entry/launcher 未变 | NOT_STARTED，留到 R6–R8；32 行矩阵没有升级 |
| 原模型、监测和用户文件 | snapshot 分类/hash、旧回归 | 1242 tracked/330 ignored 未变；无新增失败，保留三个历史文档/path断言 |

当前机器可读核心验收矩阵为 [r5/configuration_matrix.json](linearno_history_audit/r5/configuration_matrix.json)，独立于 R1 的原计划 fixture。完整报告为 [R5 REPORT](LINEARNO_HISTORY_R5_REPORT.md)。所有数据为小型 CPU 合成，不代表真实训练/评估、GPU、远端或改进精度证据。

**本 R5 阶段结束，未执行 R6 或后续阶段。**


## R6 增量验收

Airfoil/Darcy/Elasticity/Pipe × A0K0/A1K0/A0K1/A1K1：L4–8共80个真实preset构造/计数PASS；L4缩宽但真实N的16个原生训练/严格存档/续训/独立进程评估闭环PASS。证据 `linearno_history_audit/r6/final-results.json`。其余四任务本阶段未启用；全部真实数据项NOT RUN。


## R7 增量验收

NS/Plasticity ×四组合：L4–8共40真实构造/计数PASS；L4真实N/时间维的8原生合成闭环PASS。六Standard累计120构造+24闭环；原生mask/查询连续性见 `linearno_history_audit/r7/results.json`；全八任务分级矩阵见 `r7/task-matrix.json`。工业尚未接入，真实数据项全部NOT RUN。


### R8 八任务×四组合生产验收

| 任务 | A0K0 | A1K0 | A0K1 | A1K1 | 证据 |
|---|---|---|---|---|---|
| Airfoil | PASS | PASS | PASS | PASS | r6 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| Darcy | PASS | PASS | PASS | PASS | r6 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| Elasticity | PASS | PASS | PASS | PASS | r6 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| Pipe | PASS | PASS | PASS | PASS | r6 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| NS | PASS | PASS | PASS | PASS | r7 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| Plasticity | PASS | PASS | PASS | PASS | r7 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| AirfRANS | PASS | PASS | PASS | PASS | r8 results.json；CPU合成原生train/checkpoint/resume/独立eval |
| ShapeNet-Car | PASS | PASS | PASS | PASS | r8 results.json；CPU合成原生train/checkpoint/resume/独立eval |

所有任务L4–8四组合正式preset构造计数PASS；真实loader、mini-run、GPU、完整多seed结果仍NOT RUN。Air另有四模式双成员、两类中断边界exact；eval先读各成员完整spec。


### R9/R10 最终验收索引

32八任务四模式原生闭环PASS；20核心depth/mode全步骤PASS；160正式preset构造计数PASS；48诊断无扰动PASS；24理论/实算MAC精确PASS；50 CPU效率/强对照行已测；480远端命令shell预览PASS。详见`linearno_history_audit/r9/verification-matrix.json`逐task/mode/depth/device/dtype记录，R10最终报告。R10只读关键检查62通过/1历史方法3断言失败、另19官方/RNG通过4CUDA跳过。所有真实loader/mini-run/GPU/完整多seed和SOTA均NOT RUN；不能从合成检查推断论文指标已复现。R6导入回归已通过单独返修证据闭合。
