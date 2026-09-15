# CDLNO 前段 latent processor 消融审查

**最新（2026-09-15）：A3已审查通过；A4计数、有限性能及文档交付完成，待审查。** 见[A4报告](CDLNO_FRONT_ABLATION_A4.md)、[八任务命令](CDLNO_FRONT_ABLATION_A2_COMMANDS.md)、[实际结果](front_ablation_audit/a4/summary.json)。性能工具支持三front模式、matched LRSA显式full；默认SA8/6/6、前段FFN4/4/0实际核对，完整成本/实际参数、33行CPU/10行GPU通过；203/203回归通过。生产数学/八任务/预设/训练数据未改，128份本轮冻结文件不变；累计pre-A1核心历史/CDPA/rear/readout及原协议核查通过。没有真实训练或自动后续矩阵；V2–V5续训可视化接入仍未执行。下方状态均为历史。

**最新（2026-09-15）：A2已由用户审查通过；A3八任务三模式合成训练/eval/checkpoint验证完成，待审查。** 见 [A3报告与覆盖表](CDLNO_FRONT_ABLATION_A3.md)。184/184完整回归通过，最后43/43定向检查通过；真实PyG、有限12格新模式GPU及真正修改前full16份基准均实际通过。只改测试/文档，128份生产/任务/脚本/工具文件hash不变；无真实数据训练。用户本轮A3范围替代旧建议“性能工具A3”，性能工具仍冻结，A4未执行。下方A0–A2及阶段建议保留为历史，以本段与A3报告为准。

**最新：A1已由用户审查通过；本轮A2接入八任务模式参数、配置/加载校验和必要目录/命令。** 见 [A2报告](CDLNO_FRONT_ABLATION_A2.md) 与 [命令](CDLNO_FRONT_ABLATION_A2_COMMANDS.md)。共享数学未改；A3性能工具等及A4未执行。以下A0/A1记录保留为历史。

**最新（2026-09-14）：A0已审查通过；A1共享配置/前段三模式/核心定向验证已完成，待审查。** 见 [A1实现与验收报告](CDLNO_FRONT_ABLATION_A1.md)。当前共享API已有front_latent_mode；八任务wrapper/CLI与性能工具尚未接入该字段，仍默认full；A2–A4未执行。以下A0原文及“尚未实现”等状态保留为当时审查快照。

更新：2026-09-14。**补充阶段 A0 已完成，待用户审查；A1–A4 未授权、未实施。**

当前生产模型的 full 路径与用户这次给出的三行公式一致，未发现需要裁定的 baseline 架构冲突。当前没有前段 CDPA，也没有 `front_latent_mode` 配置或两种新消融的实现。本轮只读取源码、执行静态审查并增量写文档；没有改模型、配置实现、任务入口、脚本或依赖，没有运行模型或训练。

## A. 依据、版本与范围

本轮依据按优先级为：用户最新补充定义及 A0 指令 → 后来确认的原阶段决定 → [v1.2](../PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md) → v1.1 历史说明。参考了 [AGENTS](../AGENTS.md)、[STATUS](CDLNO_IMPLEMENTATION_STATUS.md)、[当前记忆](../memory/current-state.md)、[对话演变复核](../memory/2026-09-13-exported-conversation-review.md)、[最终实施报告](CDLNO_IMPLEMENTATION_REPORT.md)、[需求矩阵](CDLNO_REQUIREMENTS_MATRIX.md)及各阶段模块/接口/性能记录，并从当前源码反查下面的结论。历史报告中的计数和测试结果不是本轮重新执行的验收。

仅新增两个前段 processor 例外，统一架构字段为 `front_latent_mode ∈ {full, no_sa, identity}`，默认 full、全部前段一致。数学附件只解释/核对，不授权额外结构、损失或实验。历史记录中的独立数学附件未定位状态不因本次审查升级；本轮不声称重新阅读了附件全文或执行了其理论验证。

| 项目 | A0 实际状态 |
|---|---|
| 工作区 | `/home/hwz/CDLNO` |
| 分支 / 当前 HEAD | `main` / `769fa333742f73c132868cf560bce5ec21529362` |
| 阶段0原 Transolver 基线 | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`；不能把该旧 commit 当作目前 CDLNO 版本 |
| 已实现类/包 | `cdlno.core.CDLNO`，共享包 `cdlno`；机制类 `cdlno.cdpa.CDPA` |
| 架构/sidecar 版本 | `model_version=cdlno-core-v1`；`history_rule=front-t-after-ffn2-before-up-v1`；sidecar `schema_version=1` |
| 默认 | L8/F2/P6，entry，chunk0；Pipe M32，其余任务 M64；前段始终执行 full |
| 当前暂存区 | 无 staged diff |
| A0 起点文件证据 | [a0-start.json](front_ablation_audit/a0-start.json)：203 个 tracked/非 ignored untracked 文件 SHA256、分支、HEAD、原始 status |

起点已有修改必须保留，不归为 A0 修改：

```text
 M AGENTS.md
 M docs/CDLNO_IMPLEMENTATION_STATUS.md
 M docs/CDLNO_TASK_LAUNCHERS.md
 M memory/current-state.md
 M tran_evaluate/README.md
 M tran_evaluate/_common.sh
 M tran_evaluate/car.sh
 M tran_evaluate/ns.sh
?? docs/CDLNO_REMOTE_LAUNCHERS.md
?? docs/remote_launch_audit/                 # 7 个已有审查附件
?? path.sh
?? tran_evaluate/_standard.sh
?? tran_evaluate/airfoil.sh
?? tran_evaluate/airfrans.sh
?? tran_evaluate/darcy.sh
?? tran_evaluate/elasticity.sh
?? tran_evaluate/inspect_data.py
?? tran_evaluate/inspect_data.sh
?? tran_evaluate/pipe.sh
?? tran_evaluate/plasticity.sh
```

这些已有 diff 属于远端路径/启动准备及记录。A0 起点 `cdlno/`、三个任务项目、`tools/`、`tests/`、`pyproject.toml` 与 HEAD 无 diff。当前根目录没有计划占位的 `configs/`；真实任务 JSON 位于各子项目的 `configs/CDLNO/`。不使用旧 `cdpa_operator` / `CDPAOperator` / 整模型注册名 `CDPA` 作为当前实现路径。

## B. 本阶段文件与冻结证据

| 本轮增量 | 理由 |
|---|---|
| 本文 | 公式与源码映射、模式差异、八任务链路、兼容方案、A1–A4 建议及修改前基准要求 |
| [STATUS](CDLNO_IMPLEMENTATION_STATUS.md) | 前置 A0 状态，保留原阶段全文 |
| [AGENTS](../AGENTS.md)、[memory/current-state.md](../memory/current-state.md) | 前置最新授权边界与审查结论，保留原记录；不将 A0 完成解释为 A1 授权 |
| [front_ablation_audit/](front_ablation_audit/) | 起点哈希、只读静态检查文本及结果；不是生产模块或新增训练/模型测试实现 |

A0 最终对起点 203 文件核对：仅上表的 AGENTS/STATUS/current-state 三个既有文档增加内容，其余 **200 文件字节不变**；新增文件均在 `docs/`。原模型、配置、数据、采样/点序、normalizer、标签、loss、optimizer/scheduler、时间循环、图处理、评价、脚本和依赖均未修改。

另外对阶段0基线的三个任务目录及根 `Physics_Attention.py` 共 65 个原文件作字节比较：53 个完全一致；12 个差异文件恰为此前已接入的六 exp、`model_dict.py`、两工业各自 main/main_evaluation、AirfRANS params.yaml。本轮这些 12 文件又均与当前 HEAD/起点一致。原模型/原脚本、工业 train/dataset/metrics 包含在 53 个一致文件内；六 exp 内冻结区域的完整 AST 等价证据沿用原阶段测试，本轮没有重新执行那套测试。不能把“12 个已有集成差异”写成 A0 新增修改，也不能把源码不变当成真实数据路径已经运行。

## C. 公式与真正执行的代码

以下行号对应上述 A0 HEAD。`B` 为 batch，`N` 为点数，`M` 为 latent 数，`d` 为宽度，`h` 为头数，`d_h=d/h`。下文公式的中间量 B 与 batch 符号按上下文区分。

### C1. 当前 full 的准确顺序

实际入口是 [cdlno/modules.py](../cdlno/modules.py) 的 `LRSAFrontBlock.__init__`（296）及 `forward`（328），由 [core.py](../cdlno/core.py) 的 `CDLNO.__init__`（57）独立创建 F 次，`forward`（90）逐层调用。

```text
H = 输入点特征 [B,N,d]
Hn = point_norm(H)
S = down(Hn)                          [B,M,d]
A = S + latent_ffn_1(latent_norm_1(S))
B = A + latent_sa(latent_norm_sa(A))
T = B + latent_ffn_2(latent_norm_2(B)) [B,M,d]
U = H + up(Hn, up_latent_norm(T))     [B,N,d]
H_next = U + point_ffn(point_ffn_norm(U))
return H_next, T
```

这就是用户确认的 `S → A → B → T` 三行公式；当前变量名复用 `z`，不改变计算顺序。T 是完整 residual 更新后的表示，不是 FFN2 分支的裸输出，不是 `LN(T)`。Up 的专属 norm 仅用于 Up 输入，返回的仍是原 T，未 detach。规则网格 point_ffn 另接 `(H,W)`；T 不受该网格 reshape。

| 对象 | 当前文件 / 类或属性 / 函数 | norm、参数和残差 |
|---|---|---|
| 点输入 norm | `modules.py:312`，`LRSAFrontBlock.point_norm` | RMSNorm(d)，scale1，无 bias，eps1e-6；其输出同时用于 Down K/V 和 Up query |
| Down | `modules.py:235,264`，`_DownAttention.forward` | learned Q `[M,h,d_h]`，K/V/O 投影；没有额外 WQ、没有 query residual；独立 per-head Q/K RMSNorm；含完整 O+b |
| latent FFN1 | `modules.py:314`，`latent_norm_1` + `latent_ffn_1:PlainFFN` | RMS pre-norm；两 Linear 含 bias，d→r d→d，GELU；外层加 S |
| latent SA | `modules.py:316`，`latent_norm_sa` + `latent_sa:_SelfAttention` | RMS pre-norm；同一个归一化 A 作为 Q/K/V 输入，QKV 无 bias、O 有 bias，内部独立 Q/K RMSNorm(d_h)；外层加 A |
| latent FFN2 | `modules.py:318`，`latent_norm_2` + `latent_ffn_2:PlainFFN` | 同 FFN1 形式、参数独立；外层加 B 后得到 T |
| Up | `modules.py:320`，`up_latent_norm` + `up:_UpAttention` | T 的专属 RMSNorm(d)保留；Up query 是 Hn；独立于 Down 的 Q/K/V/O、Q/K RMSNorm；Up 结果加原 H |
| 点 FFN | `modules.py:325`，`point_ffn_norm` + `point_ffn` | 外部 RMSNorm；不规则用 PlainFFN；规则用 ConvFFN；加 U 的点残差 |
| ConvFFN | `modules.py:123`，`ConvFFN.forward` | `[B,N,d]→[B,d,H,W]`，dense groups=1 3×3卷积→内部 affine LN→无bias Linear→GELU→有bias Linear→原序展开；非 depthwise |
| 共用 SDPA 实现 | `modules.py:150,185`，`_ProjectedAttention.forward`；Down 独立实现 | Down 注意力 `[B,h,M,N]`、SA `[B,h,M,M]`、Up `[B,h,N,M]`；实际用 PyTorch SDPA，不在正常返回值中保存注意力矩阵 |
| T 的返回 | `modules.py:340,347` | `t = ...ffn2...` 后只作为 Up 输入及第二返回值，历史仍有梯度 |
| T 的收集 | `core.py:87–94`，`CDLNO.forward` | 有活跃 CDPA 才收集；每次调用新建局部列表，不缓存跨 forward/真实时间状态 |
| CDPA 消费 | `core.py:96–109`、`cdpa.py:CDPA.forward` | 先 bridge，再在后段位置传 `previous, tuple(history)`；不在 front 内消费 |

初始化基线保持：普通 Linear 由子模块初始化一次 trunc_normal(std=.02)、bias0；RMS scale1；Down query orthogonal；bridge query normal(.02)；dense Conv 原生初始化；CDPA scorer 零初始化。后续 full 模式必须保留原模块命名、创建顺序和随机数消耗顺序，不在 wrapper 递归 apply 重置核心。删除分支的模式不注册其闲置参数。

### C2. 前段 CDPA 与冻结边界

**没有找到已落地的前段 CDPA。** 证据不仅是关键词检索：`LRSAFrontBlock` 实例化的模块和 forward 调用如上；`CDLNO.cdpa_at` 只以后段索引 j 建立；前段循环只调用 block/append T，bridge 后的后段循环才调用 CDPA。当前没有“Down→CDPA→Up”、相邻 front CDPA 或后段 Kimi AttnRes，也没有可见的单独批准并已实现这些结构的依据。

因此未来 identity 的 `T=S` 与当前已接受 baseline 没有冲突。不能把后段入口 CDPA 移到 front 来解释 identity，不能因增设 identity 而关闭后段既有 CDPA。若 A1 开始前用户又修改了源码，须重新核对此结论，不能仅依赖 A0 文档。

冻结的整网顺序仍为：原 stem → F 个 front → bridge → 按 off/entry/every_block 执行 CDPA 和 P 个后段 → final feature readout。Bridge 仍是 learned query 加 Cross，无 encoder FFN；后段仍 SA 残差再 GEGLU 残差；readout 仍用 H_F 作 query 及点残差。各层 M 一致，P 仅由 L−F 派生，F0/扩展 L 保留。

entry 只读 T1…TF；every 在第 j 个后段前读 T1…TF、raw Z0…Z(j−2)，当前为 Z(j−1)，不重复当前。各位置参数独立，不共享跨位置投影缓存，不 detach；CDPA 两级 softmax、R0 恒等、完整 O+b、RAW 融合、零 scorer、FP32 depth 和 chunk 公式全部冻结。

### C3. 三种模式的保留/移除清单（A0时为规格；A1–A4已实现并核验）

| 子层、norm、残差、参数 | full | no_sa | identity |
|---|---|---|---|
| point_norm、完整 Down、learned queries、Down K/V/O+b、Down Q/K norms | 保留 | 保留 | 保留 |
| N1 / FFN1 两层 Linear+b / `S + FFN1(N1(S))` | 保留 | 保留 | 整段移除 |
| Nsa / SA Q/K/V/O+b / SA 内 Q/K norms / `A + SA(Nsa(A))` | 保留 | 整段移除 | 整段移除 |
| N2 / FFN2 两层 Linear+b / 相应外层 residual | 保留 | 保留；输入改为 A | 整段移除 |
| 历史 T 的值 | `B+FFN2(N2(B))` | `A+FFN2(N2(A))` | 原 S 本身 |
| 历史 T 的位置 | FFN2 residual 后、Up norm 前 | FFN2 residual 后、Up norm 前 | Down 完整输出后、Up norm 前 |
| up_latent_norm、完整 Up 投影/QK norms、`H + Up(...)` | 保留 | 保留 | 保留 |
| point_ffn_norm、point FFN/ConvFFN（包括内部 LN）、点 residual | 保留 | 保留 | 保留 |
| Bridge / 后段 / CDPA / final readout | 保留 | 保留 | 保留 |

no_sa 不能写成 `A + Identity(A)`；identity 不能写成 `S + Identity(S)`、`T=LN(S)` 或保留闲置 Q/K/V 给其他机制。identity 只是 **latent processor** 恒等，整个前段 block 仍然执行可学习的 Down/Up/点更新，H_next 不等于 H。

参数差异可由当前层定义推导。每个前段、宽 d、FFN ratio r、头宽 d_h=d/h：

```text
SA 加专属 Nsa：      4 d² + 2 d + 2 d_h
每个 FFN 加专属 norm：2 r d² + (r+2)d
no_sa 比 full 少：    4 d² + 2 d + 2 d_h
identity 比 full 少： (4+4r)d² + (2r+6)d + 2 d_h
r=2 时 identity 少：  12 d² + 10 d + 2 d_h
```

两个 d_h 来自 SA 内 Q/K RMS 的两个 scale 向量；不是每个 head 另乘 h。整网差额乘 F；F0 为0。以上是代数计数，不是本轮参数量实测。后续须以 named_parameters/实际调用核对。

| L8/F2/P6 的操作（CDPA模式相同） | full | no_sa | identity |
|---|---:|---:|---:|
| Down/bridge | 3 | 3 | 3 |
| Up/readout | 3 | 3 | 3 |
| latent SA | 8 | 6 | 6 |
| 前段 latent FFN | 4 | 4 | 0 |
| 规则任务点 ConvFFN | 3 | 3 | 3 |
| entry 历史份数 / chunk0 历史 SDPA 调用 | 2 / 1 | 2 / 1 | 2 / 1 |
| every 历史份数 / chunk0 历史 SDPA 调用 | 27 / 6 | 27 / 6 | 27 / 6 |

一般 SA 数为 full=L、其余=P；历史来源总数仍是 entry F、every PF+P(P−1)/2。历史数相同不意味着 T 的内容相同。L 仍是总 block 数，不能将 no_sa/identity 的 L8 报告成8次 SA。

### C4. 参考 LRSA 消融开关

实际本地参考 `/home/hwz/LRSA-Operator`，HEAD `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`、工作区干净，来源 [Adversarr/LRSA-Operator](https://github.com/Adversarr/LRSA-Operator/tree/47b03f8c8c8da30bbcc0737b008dc4548f9cb98e)。依据 `src/perceiverforpde/modeling/layers/attn.py:PerceiverAttention.forward` 的 598–617 行，而非论文名称或开关名称。

```text
slice = down_project(...)
if not disable_interleaved_channel_mixing:
    slice = slice + channel_mixing_1(channel_norm_1(slice))
if not disable_interleaved_blocks:
    sn = sm_norm(slice)
    slice = slice + latents_attention(sn, sn, sn)
    slice = slice + channel_mixing_2(channel_norm_2(slice))
o = up_project(..., slice, ...)
```

| disable_interleaved_blocks | disable_interleaved_channel_mixing | 实际 latent processor |
|---|---|---|
| False | False | FFN1→SA→FFN2，完整 full |
| True | False | 仅 FFN1；**不是本次 no_sa** |
| False | True | SA→FFN2；也不是本次 no_sa |
| True | True | 不执行三个 latent 残差子层，latent 计算恒等；不据此复制参考的两布尔接口/参数注册策略 |

本次 no_sa 是用户明确选择的 FFN1→FFN2，不能用参考单一 `disable_interleaved_blocks=True` 冒充。已有 [test_lrsa_reference.py](../tests/test_lrsa_reference.py) 的 full 参数映射可复用，但新 no_sa 应按独立显式公式验证。

本地参考不属于生产 import 路径。本工程仍按确认公式实现，不移植参考训练器/RoPE/gate。许可沿用 [来源说明](CDLNO_THIRD_PARTY_NOTICES.md)：Transolver MIT、IPOT MIT；LRSA 此快照未发现明确 LICENSE，不能推断 MIT。本轮未 vendor 参考代码或获取新远端版本。

## D. 八任务配置、构造、保存与加载

### D1. 当前接口与默认配置

所有行都是当前 full。统一 L8/F2、两个 FFN ratios2、entry/chunk0/dropout0；新增 mode 不应改变下表。N 为原任务布局，工业支持可变 N 单图。

| 任务 / 配置文件名 | 真实 wrapper 输入 → 输出 | 位置/时间与 stem | d/h/M；batch/epochs；点 FFN |
|---|---|---|---|
| Darcy / darcy.json | x[B,N,2], fx[B,N,1] → [B,N,1]，N=85×85 默认 | 原规则索引网格64维reference距离替换xy，再拼fx；stem65；无placeholder/time参数 | 128/8/64；4/500；Conv |
| Elasticity / elasticity.json | x[B,972,2], fx=None → [B,972,1] | 原xy，stem2 + 活跃placeholder；无time | 128/8/64；1/500；point |
| Airfoil / airfoil.json | x[B,221×51,2], fx=None → [B,N,1] | 保留弯曲物理坐标及原索引；stem2+placeholder | 128/4/64；4/500；Conv |
| Pipe / pipe.json | x[B,129×129,2], fx=None → [B,N,1] | 原入口归一化的坐标及原索引；stem2+placeholder | 128/4/32；8/500；Conv |
| NS / ns.json | x[B,4096,2], fx[B,4096,10] → [B,4096,1] | reference64替xy，再拼窗口10，stem74；无time投影参数 | 256/8/64；2/500；Conv |
| Plasticity / plasticity.json | x[B,3131,2], fx[B,3131,1], T[B,1] → [B,3131,4] | 101×31空间；stem3；保留原time embedding+time_fc，不展平时间 | 128/8/64；8/500；Conv |
| ShapeNet-Car / shapenet_car.json | (cfd_data,geom_data) → [N,4] | 只读cfd.x7及单图元数据，stem7+placeholder；velocity3/pressure1；不读y/geom编码 | 256/8/64；1/200；point |
| AirfRANS / airfrans.json | data → [N,4] | x7追加原pos到64 reference的距离，域x[-2,4]/y[-1.5,1.5]；stem71+placeholder；vx/vy/p/nut | 256/8/64；1/398；point |

六标准 JSON 的 `training` 会实际注入 parser 默认值；NS 的 max_grad_norm 为 null/None，其余五为0.1，保持后来已确认修正。NS 每次 forward 重建 history，训练10步真值回填并累加 loss 后一次更新，测试预测回填；Plasticity 每 batch 20 个 T 各自 forward/backward/step，scheduler 节奏保持原入口。不得为消融变更这些语义。

### D2. CLI / JSON / YAML 到核心的实际路径

| 入口组 | 当前参数和模型构造链 | 最小新增 mode 传递位置（后续建议） |
|---|---|---|
| 六标准 | 六 `exp_*.py` → `PDE-Solving-StandardBenchmark/cdlno_entry.py:parse_args(18)` → 本目录 `configs/CDLNO/<task>.json`，显式CLI优先 → `model_dict.py:get_model` → 薄 `model/CDLNO_*.Model` → `cdlno.standard.StaticStandardModel`/`TemporalStandardModel` → config → core | 公用 parser + `model_kwargs(51)`、两个共享 wrapper ctor/config、core front 构造；六exp已有 `**cdlno_model_kwargs(args)`，无需为本字段逐个改exp |
| Car | `main.py` 的 cfd_model=CDLNO → `models/cdlno_run.py:parse_args(21)` 读取 `configs/CDLNO/shapenet_car.json` 的 model → `model_kwargs(56)` → `models/CDLNO.py:Model(37)` / `architecture(13)` → config → core | 新参数加入公用 parser、kwargs、architecture与Model；main已有kwargs调用 |
| AirfRANS | `main.py` / `main_evaluation.py` → `cdlno_entry.py:parse_args(20)` 读取 `configs/CDLNO/airfrans.json` 的 model → `model_kwargs(56)` → `models/CDLNO.py` 导出的 `cdlno.airfrans.AirfRANSModel` / `architecture` → config → core | parser/kwargs及共享Air wrapper；main和eval已有CDLNO分支 |

薄标准模块实际为 `CDLNO_Irregular_Mesh.py`、`CDLNO_Structured_Mesh_2D.py`、`CDLNO_Temporal_Structured_Mesh_2D.py`；不需要复制三套数学模块或为消融再注册三种整模型名。配置字段属于 architecture，不属于 runtime；CLI 可统一 `--front-latent-mode`，工业可保留项目惯例的同字段下划线别名，仍只有一个值、没有两组布尔开关。

Car JSON 的 training 只是记录，预算实际来自原 CLI/main；不要误以为改 JSON 即改变训练。Air JSON 的 initial_training 也只是记录；实际使用 `params.yaml:CDLNO`，`resolve_hparams(64)` 只应用显式 nb_epochs/batch_size/lr 覆盖。该 YAML 的六个训练/采样字段目前与 Transolver 对应值一致（398epochs/batch1/lr.001/subsampling32000/r.05/max_neighbors64）。新增 mode 应进 model JSON/架构参数，不塞入训练 hparams 导致旧 run_contract 被无关改变；本次不需要改 YAML。

### D3. 保存协议和 eval 顺序

| 项目 | 实际保存对象 / 路径 / 类名 | train / eval 与校验 |
|---|---|---|
| 六标准 | `StaticRun.save`：`torch.save(model.state_dict(), run/model.pt)` | 同一exp先构造 wrapper；`StaticRun.__init__(70)` eval先 `_validate`；`load(123)` 再validate→`torch.load(weights_only=True)`→`load_state_dict(strict=True)`；原保存频率保留 |
| Car | 原 `train.py:112` 保存整模型 `run/model_<nb_epochs>.pth`；稳定 `models.CDLNO.Model` | train由main构造；eval `main_evaluation.py:36`→`CarRun`；`load(148)`先validate sidecar→局部可信 `weights_only=False`→检查wrapper/core配置、类型→新expected实例strict校验state→返回加载的原对象 |
| AirfRANS 单成员 | 原 `train.py:271` 保存 `run/member_000/model` 等整模型；稳定 `cdlno.airfrans.AirfRANSModel` | `AirRun.load(member=...)`支持成员加载；同样先sidecar后局部可信对象加载/strict校验 |
| AirfRANS 总列表 | `main.py:106`附近保存 `run/CDLNO` 为模型对象列表 | 正式eval `main_evaluation.py:55`→`AirRun.load()`；检查list类型/nmodel长度及每个成员类、wrapper/core配置、严格state，再交给原评价流程 |

不是所有项目都存 state_dict；不能为新 mode 强行统一保存协议。`weights_only=False` 仅限项目已有的可信整模型加载位置，不修改全局权限。Car/Air当前校验用临时 expected 模型加载 state 验证后，返回的仍是原 pickle 对象；这正是旧对象属性兼容也必须处理的原因。

所有项目在 `architecture.json` 保存 core 架构，运行 chunk/device/dtype 等另放 runtime；标准还校验 `metadata.wrapper_architecture`，Car另有fold/epochs/weight/cfd_mesh/r，Air另有task/nmodel/weight/hparams run_contract。不能只比较 M/d/h 而丢掉任务/位置/时间/模式信息。

现有 `save_sidecar` 拒绝覆盖；eval 经 `load_sidecar` 读取和 canonicalize 后比较，不能先写入“当前配置”覆盖旧文件。新建训练目录用 `mkdir(exist_ok=False)`；eval结果另用唯一子目录。six standard旧Transolver分支已有某些 `strict=False` 加载语句，这是冻结的原行为，不是 CDLNO 加载策略；本消融不得借用它们规避差异。

### D4. 启动路径与尚未完成的入口

目前存在八 JSON、子项目16个 CDLNO train/eval 脚本，根 `tran_evaluate/{darcy,elasticity,airfoil,pipe,ns,plasticity,car,airfrans}.sh` 八个双模式脚本。完整清单见 [任务清单](CDLNO_TASK_LAUNCHERS.md)及 [远端启动说明](../tran_evaluate/README.md)。

根脚本通过 `_common.sh`/`_standard.sh`、`path.sh` 保留用户远端路径及命令尾参数覆盖；标准默认 run 为 `CDLNO_RUNS_ROOT/<task>/CDLNO_RUN_TAG`（默认entry），Car为car/fold0_<tag>，Air为airfrans/full_<tag>。不传显式目录的子项目入口另外生成包含L/F/M/CDPA、时间戳/uuid的目录。Air train `--my_path` 是 Dataset 本身，eval 是其父目录；Car `--save_dir` 是预处理数据目录，不是权重输出目录。

**没有仍未接入的原八任务 CDLNO 入口；本次新增模式在所有入口都尚未实现。** 原报告中“工业待完成”等是阶段历史，不是当前状态。下面是尚未运行/继承限制，而不是“模型选择入口没写完”：

- 八任务 CDLNO 真实数据完整性、训练收敛、精度和真实epoch效率仍未验收。用户原Car远端训练成功、Darcy MAT存在及启动日志，不等于新增模型全程成功。
- Car完整阻力评价仍有原固定 `/data/PDE_data/mlcfd_data/training_data/param0` 路径及fold0限制；原外层pressure/velocity日志命名交叉不影响内部正确backward。均不在本消融中顺便修改。
- Air真实采样/图构造/idx scatter平均/边界处理未端到端实测，不能删除这些步骤获取速度收益。
- 新模式的 CPU合成、PyG、旧pickle、新checkpoint及GPU验收全部属于后续阶段；本轮未跑。远端 Python3.10/torch2.11/cu128 是用户报告的兼容基准，Python3.11和远端新模式运行未核实，不重装或替换依赖。

## E. 最小兼容方案（仅建议，A0 未实施）

### E1. 架构字段、旧 JSON 与 history_rule

`CDLNOArchitectureConfig` 是 frozen+slots dataclass，现有18字段，P是派生property。新增 mode 必须严格枚举校验，进入 `to_dict/from_dict/compare_architecture`；chunk/device/dtype仍不参与权重结构兼容。

只在字段真正缺失且版本属于已知历史模型时，将旧配置解释为 full；显式 `null`、空串、未知值或其他类型应报错，不能用 truthy/falsy 回退。旧full sidecar可匹配新显式full；旧full不能匹配显式no_sa/identity。F0即使三种前段模式数学相同，仍建议严格记录/比较显式架构选择，不吞掉用户配置冲突。

**需处理的文字合同：** 当前 `history_rule=front-t-after-ffn2-before-up-v1` 在 config.validate 和 core 中都是强校验。它准确描述当前full；未来identity没有FFN2，不能仍声称历史取于FFN2后。这是新增消融的兼容工作，不是当前baseline错误。

建议采用一个准确的中性 canonical 历史规则（例如 `front-t-after-selected-processor-before-up-v1`），其含义由唯一 mode 决定，三模式T位置见C3。旧 `cdlno-core-v1` + 旧rule + 缺mode/显式full可在读取时映射为中性rule/full；不得把旧rule与显式identity/no_sa冲突组合静默改写。新构造自动使用中性规则，不要求用户再指定第二个history开关。未知model_version/schema/history_rule继续拒绝，不能全局忽略版本/rule比较。新增字段已能区分子结构，不必为了消融重命名类或所有state键；具体迁移实现须在A1通过旧样本验证。

canonicalize只改内存中的解释，**eval不重写旧 architecture.json**。metadata的wrapper输入/位置/时间版本继续严格检查，不因补mode而全部升级或清空；这些语义没有变化。

### E2. 整模型 pickle 不等同于 JSON

旧pickle有两层兼容点：

1. slotted配置：frozen dataclass的默认pickle状态按声明顺序列出字段值，恢复按字段顺序zip。**把新字段插入中间会使旧值错位；仅追加到末尾仍可能留下未赋值的新slot**，使 `asdict/validate/compare` 访问失败。应保留旧18字段顺序，在末尾追加mode，并对已知旧18值布局实施受限恢复/补full与rule规范化，或等价的显式受控序列化方案；其他长度/不合法值拒绝。不能仅在JSON `from_dict` 中setdefault就声称整模型兼容。
2. 旧nn.Module对象：`torch.load` 恢复对象时不重新跑新的 `__init__`。旧front block没有mode属性；未来forward要能识别“缺属性的旧full”（例如受限的 `getattr(..., 'full')` 路径或受控加载迁移）。只能补语义元数据，不能重建/随机初始化旧权重，不能detach或改residual。显式存在但无效的属性应报错。

项目加载时须一起核对 sidecar、wrapper.config、core.config、每个front的实际模式/模块集合和F数量。旧缺属性full应具备原FFN1/SA/FFN2及norms；新no_sa/identity应没有被删子层的参数。不能只改配置标签，把仍含full模块但走另一路径的对象当作正确消融。

每次仍用 `strict=True` 检查权重key/shape；保留 Car `models.CDLNO.Model` 和 Air `cdlno.airfrans.AirfRANSModel` 路径，分别测试单对象和列表。要验“直接载入真正旧对象后forward”及“通过CarRun/AirRun加载”两条路径。已有整模型可信加载边界不扩大。

本次不授权full→no_sa/identity的训练权重迁移/热启动功能。测试中为核对固定子模块公式而显式拷贝共同参数，不等于允许eval用缺权重或忽略意外key的方式跨模式加载。

### E3. 模式冲突与实验隔离

| 情况 | 后续应有结果 |
|---|---|
| 历史缺mode配置/真实旧对象，请求full | 仅已知旧布局按full解释；保留全部权重；同环境输出回归通过 |
| 历史full，请求no_sa/identity | 在sidecar比较处拒绝，不能先载权重再删层 |
| 新配置同mode、只换chunk/device/dtype | 架构比较允许；设备/精度实际能力另验 |
| 显式不同mode、非法值、版本未知、缺少/多余权重 | 清晰报错；sidecar不改写；不使用strict=False |
| sidecar、wrapper/core配置或block属性互相矛盾 | 拒绝，即使某些state形状恰好一致或F=0 |
| 新train目录已存在 | 继续拒绝，不覆盖、清空或自动当resume |
| eval | 明确指定已有run和匹配mode；先读后验；新结果子目录 |

建议 full 保留既有默认路径语义；新no_sa/identity自动生成带mode的独立默认目录，或脚本显式传不同run/tag。目录名必须使用**实际解析的有效参数**，包含用户末尾覆盖，而不是只读默认值；显式自定义路径仍优先且必须是新目录。不能自动把旧entry目录当no_sa实验。根脚本本身传了显式run目录，所以仅改底层run-helper的自动名称不足以覆盖所有启动方式；A3须同时核对根脚本与子项目脚本。A0未增加任何当前parser不支持的flag。

### E4. 性能计数兼容

`tools/cdlno_perf/costs.py:audit.pre`（99）目前对 `LRSAFrontBlock` 和 `PersistentLatentBlock` 一律增加一次latent_sa，**当前full下正确**，新增模式后应按实际活跃SA子模块调用计数。`Case`/`build` 和benchmark CLI/结果也需显式携带mode，保留原参数与数学MAC口径；不能漏算未被删除的Down/Up/Conv、CDPA、stack或history存储。

`lrsa_matched` 仍是连续L个完整LRSA block的既定结构对照，默认full，不随新CDLNO mode偷偷删层；原Transolver对照不接收新kwargs。需分别记录原任务配置和同配置结构比较。仅计数/合成性能不能推出新模式精度、收敛或真实epoch提升。

## F. A1–A4 最小变更与验收建议

下表是建议分工，不是实施授权；用户随后明确点名的阶段范围优先。保持单共享数学实现，不复制三份CDLNO，不新增研究分支。

| 阶段建议 | 最小候选变更文件 | 必要验收/停止点 |
|---|---|---|
| A1：先留旧full基准，再做核心/config兼容 | `cdlno/config.py`、`checkpoint.py`、`modules.py`、`core.py`；相关 `tests/test_modules.py`、`test_core.py`、`test_core_config.py`，必要时独立消融reference测试；下节的基准捕获工具/产物 | 首次实现编辑前完成真实旧基准；full严格state键/数值回归；no_sa/identity显式数学reference、T在Up norm前/不detach、删除参数及norm；F0/扩展L/非法mode；JSON/旧slots pickle/显式冲突/sidecar不覆盖；不接训练入口 |
| A2：八任务参数与checkpoint接入 | `cdlno/standard.py`、`cdlno/airfrans.py`、`Car-Design-ShapeNetCar/models/CDLNO.py`；标准`cdlno_entry.py`、Car`models/cdlno_run.py`、Air`cdlno_entry.py`；八个现有JSON；四任务测试文件`test_static_standard.py`、`test_temporal_standard.py`、`test_shapenet_car.py`、`test_airfrans.py` | 三组真实parser AST提取检查、显式CLI优先/旧模型无新kwargs；八wrapper输出与loss合成前反向；时间语义；两工业真PyG单图/变N/多图拒绝；三个原cwd新进程full旧对象及三新mode保存/加载；不改原保存协议 |
| A3：启动脚本、目录与成本工具 | 子项目16份`CDLNO*.sh`（标准12/Car2/Air2）；根`tran_evaluate/_standard.sh`、`car.sh`、`airfrans.sh`及必要的`_common.sh`目录解析；`tools/cdlno_perf/models.py`、`costs.py`、`tools/cdlno_benchmark.py`、`tests/test_performance.py`；现有使用/清单文档 | 默认full命令不漂移、mode显式转发/末尾覆盖、train/eval同配置、三mode目录不覆盖；实际计数C3与参数代数吻合；同权重chunk0/1/小块；matched LRSA仍full；性能比较条件一致。GPU仅可用时做有限合成，不自动扫描 |
| A4：综合回归与交付 | 定向补足已有tests与必要消融验收；本文/STATUS/README/launcher使用/性能及最终报告 | 全任务新模式接口；full历史基准；F0..6×CDPA三模式与扩展L的必要门槛；history/reset/梯度；旧/新单对象及列表；frozen源码diff/AST；一次最终套件，CPU/PyG/GPU/真实实验分开报告；不顺修旧分支 |

不预设需要修改 `cdlno/cdpa.py`、三个薄标准wrapper文件、`model_dict.py`、六exp、两工业main/main_evaluation/train、Air `params.yaml`、`path.sh`、原Transolver模型/脚本、数据/metrics/依赖。这些现有调用路径已足以复用；若后续出现必须触及的具体缺口，须列证据并限于授权阶段处理，不能为方便import重构训练器。根六标准小脚本只是调用统一helper，是否需要逐个改以有效参数流为准，不为凑文件数改它们。

### F1. A1 修改前 full 数值/权重/配置基准

**本轮仅留存源码指纹，没有生成数值基准或旧pickle。A1必须在第一次编辑模型、配置或加载实现之前完成下列步骤。** 不允许修改后才从新实现生成“旧full”；从新对象删属性也不是真正旧pickle证据。

1. 重读状态/检查diff，将当时生产文件哈希与 [A0快照](front_ablation_audit/a0-start.json) 对照。如用户在A0后改过实现，先记录新基线并核对兼容影响。以**该工作树实际内容**复制一份隔离只读baseline（不reset/checkout覆盖工作区）；保留import所需的共享包、三个项目模型/helpers/configs及源路径清单、HEAD和未提交diff。不能只保存旧上游Transolver commit。
2. 在仍使用原实现的独立进程生成固定seed合成输入，记录Python/torch/device/dtype/SDPA backend、TF32/AMP/matmul精度/线程等实际条件。主要回归固定CPU FP32、相同受控SDPA条件；full构造前重设seed并保存初始RNG状态。基准工具不能import会读数据的exp/main，不安装依赖、不创建伪数据集、不启动训练循环。
3. 留存原始配置JSON、architecture sidecar字节（**字段确实缺mode**）、constructor kwargs、state_dict及key/shape/dtype/逐tensor校验、named_buffers（包括nonpersistent reference/pos）、全部inputs/targets、H_next/T、core输出和wrapper输出，以及明确损失下输入/关键参数梯度。只做合成前反向；基准本身无需optimizer更新。初始零scorer与一份受控非零scorer权重都保存，避免仅均匀融合掩盖历史错误；标记二者不同fixture，不改变生产初始化。
4. 最小代表覆盖：独立front的point及5×7非方形网格、B>1；core默认F2/L8下off/entry/every、F0，以及用户常用F3/F4的小宽配置；固定M/d/h。同权重F0 entry/off需显式复制相同state。八wrapper保存各自输入/输出/配置，NS保留64²、Plasticity保留101×31而减小d/M；时间条件用同batch不同T，工业用真实PyG单图两种N，保留其实际字段。
5. 在三个**原工作目录**的新进程分别保存原六标准state_dict、真正旧 `models.CDLNO.Model` 整对象、真正旧 `cdlno.airfrans.AirfRANSModel` 成员及模型列表，连同匹配sidecar；进程通过明确baseline代码路径加载，核对 `cdlno.__file__` 及模型类文件，不让editable安装意外指向修改后的工作树。原18槽config、缺新属性的真实block由旧源码自然生成；A1后再由新代码加载这些固定文件。
6. fixture放独立、不可覆盖的审查产物目录（例如用户可保存的 `artifacts/front_ablation/full-before-a1/<timestamp>/`，大tensor不加入Git）。另留可追溯manifest：生成脚本内容/hash、源码hash、原命令、配置、所有产物SHA256、时间和运行结果；至少保留到A4交付。缺依赖导致某类旧fixture无法生成时先写明缺口；不能把其改后补造的对象算作通过，不能为了生成而擅装环境。
7. 修改前先把fixture重新加载验证自洽，再开始A1实现。修改后full用严格同权重、同输入和同精度对比，而非只用相同seed推定相同权重。原/新full同seed新构造的state键和值也要相等以排查初始化顺序变化；保存既有权重与nonpersistent buffer独立对照。结构/state必须精确一致，受控CPU同算子输出/梯度优先要求逐元素一致；如原环境本身存在可量化数值波动，**修改前**用重复baseline确定并记录容差，不能在看到修改后误差后放宽。跨设备/AMP验证另报容差，不冒充这一真实full回归。

### F2. 交付前自审与剩余边界

| 自审重点 | 本轮核查结果 |
|---|---|
| full公式、T位置、是否已有前段CDPA | 源码逐行及AST序列通过；无需用户裁定baseline差异；新模式尚未实现 |
| 模块/norm/residual删除范围与参考开关 | 已核对真实PerceiverAttention.forward；单个disable_interleaved_blocks不是no_sa；Up norm属于保留项 |
| 旧JSON、slotted配置、整对象/列表与strict校验 | 已定位各加载点和旧布局风险；给出受限兼容建议；实际旧pickle兼容须后续实现并验证，A0不能判通过 |
| 八任务链路/目录/冻结区 | 8JSON、16脚本、三条wrapper/loader链已核；保留既有远端修改；8入口无缺接入，新增mode全未接入 |
| full基准与性能统计口径 | 已规定改前真实fixture流程；当前SA计数仅适用于full，后续须跟随消融更新；本轮未生成数值/性能结果 |

需要后续解决的是新增字段迁移、旧pickle实证、前段参数删除与成本计数接入；不是已发现当前full公式有误。没有要求用户在A0裁定的实质baseline冲突。模型数值、PyG、GPU、真实数据/训练/精度等未运行项如D4，不以本轮静态检查代替。

## G. 实际命令与结果

本轮只使用Git、文件读取、标准库AST/JSON/hash及文档编辑工具。用于运行审查文本的本地解释器为Python3.13.9；这不是远端训练环境检查，也没有据此调整依赖。未import torch/PyG/exp/main进行模型运行。

```bash
git status --short
git diff --stat
git diff --cached --stat
git branch --show-current
git rev-parse HEAD
git diff HEAD --name-only -- cdlno PDE-Solving-StandardBenchmark Car-Design-ShapeNetCar Airfoil-Design-AirfRANS tools tests pyproject.toml
git -C /home/hwz/LRSA-Operator status --short
git -C /home/hwz/LRSA-Operator rev-parse HEAD
python -B docs/front_ablation_audit/static_check.txt > docs/front_ablation_audit/static-results.json
git diff --check
```

源码审查另用 `rg`/`sed`/`nl`；[检查文本](front_ablation_audit/static_check.txt)包含实际AST断言、8JSON/脚本存在性、三保存/加载链、基线字节比较及起点hash核对，可重复执行。起点快照从本轮开始时 `/tmp/cdlno-a0-readonly-3wbzvyi4/start.json` 原样保留，没有根据修改后的文档反向生成。

结果：[static-results.json](front_ablation_audit/static-results.json)，**13组静态检查通过、0失败**；`git diff --check`通过。文档链接/起点原文保留另作交付核查。此前119/120项模型回归及原阶段GPU结果只作历史证据，本轮未重跑，不能记为本次消融通过。

**实际未运行：** no_sa/identity任何模型前反向；full数值基准捕获与旧pickle迁移；三目录新进程模型加载；PyG/GPU/远端环境检查；真实数据训练、采样评价、收敛、精度和epoch效率。本轮没有模型测试失败记录，因为没有执行这些测试。

**本补充阶段结束，未执行下一阶段。**
