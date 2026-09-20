# 面向 `hxh5159/CDLNO-w` 的 Looped LinearNO 分阶段 Codex 修改提示词

> 目标仓库：<https://github.com/hxh5159/CDLNO-w>
>
> 本文针对 **LinearNO 的循环化改造**。本轮不修改 Transolver，也不把此前的 latent-summary AttnRes、history-conditioned K、HCC、LSAR 等机制带入新模型。
>
> 使用方式：第一次把“总控提示词”与 `LL0` 一起交给 Codex。审查每一阶段的报告后，再单独发送下一阶段。不要一次性授权 Codex 连续执行全部阶段。

---

## 0. 生成本提示词时核对到的仓库状态

我在 2026-09-19 重新克隆并只读检查了远端 `main`。当时可见的 HEAD 为：

```text
5b991226c5354af3332b2f7306b370aef0950c79
```

这只是编写提示词时的观察值。Codex 实施时必须以用户实际 checkout 的当前 HEAD、dirty 状态和适用的 `AGENTS.md` 为准，不得因为 SHA 不同而覆盖、回退或清理用户工作。

当前仓库已经包含完整的八任务 LinearNO，而不是仍待移植的旧仓库：

- 六个 Standard benchmark 的主体模型位于 `PDE-Solving-StandardBenchmark/model/LinearNO.py`；
- 共用 LinearNO 原语、profile、schema、checkpoint 和任务适配位于 `cdlno/linearno/`；
- AirfRANS 主体模型位于 `cdlno/linearno/airfrans.py`，任务本地导出位于 `Airfoil-Design-AirfRANS/models/LinearNO.py`；
- ShapeNet-Car 主体模型位于 `cdlno/linearno/shapenet.py`；
- 既有另一条研究线位于 `cdlno/linearno_history/`、顶层 `linearno_history/` 和 `tran_evaluate/linearno_history/`；
- 纯 LinearNO 的入口与命令位于 `tran_evaluate/linearno/`；
- 仓库已有严格的训练记录、checkpoint/resume、可视化、监测和大量回归测试；
- `AGENTS.md` 很长，并记录了当前阶段、冻结区和既有验收事实，必须完整阅读。

当前正式 profile 的基础结构为：

| 任务 | 基础深度 | 隐藏宽度 H | heads | 基础每头 M | variant | 本研究主配置 M |
|---|---:|---:|---:|---:|---|---:|
| Airfoil | 8 | 128 | 8 | 64 | `conv_temp` | 128 |
| Darcy | 8 | 128 | 8 | 64 | `conv_temp` | 128 |
| Elasticity | 8 | 128 | 8 | 64 | `temp` | 128 |
| Pipe | 8 | 128 | 8 | 64 | `conv_temp` | 128 |
| Navier–Stokes | 8 | 256 | 8 | 32 | `plain` | 64 |
| Plasticity | 8 | 128 | 8 | 64 | `conv` | 128 |
| ShapeNet-Car | 8 | 256 | 8 | 32 | `shapenet` | 64 |
| AirfRANS | 8 | 256 | 8 | 32 | `airfrans` | 64 |

上表中的 `M` 是每个 attention head 的实际 latent token 数或线性注意力秩。实施时必须由当前 profile 动态解析基础值，不能只把这张表硬编码进模型。

当前纯 LinearNO block 还有一个关键实现事实：最后一个 block 同时持有 hidden-to-hidden body 与 `ln_3 + mlp2` 输出头。因此，循环核心不得包含 `last_layer=True` 的 block。第一版通用拓扑应要求 `suffix_blocks >= 1`，只让最后一个独立 suffix block 持有输出头。

---

# 一、冻结的模型规格

## 1. LinearNO 主干保持不变

输入点状态记为

\[
X\in\mathbb{R}^{B\times N\times H},\qquad d_h=H/h.
\]

每次执行一个 LinearNO operator 时，必须重新由当前点状态计算

\[
Q\in\mathbb{R}^{B\times h\times N\times M},\quad
K\in\mathbb{R}^{B\times h\times N\times M},\quad
V\in\mathbb{R}^{B\times h\times N\times d_h},
\]

其中 \(Q\) 沿 \(M\) 归一化，\(K\) 沿 \(N\) 归一化。压缩与重建为

\[
C=K^\top V\in\mathbb{R}^{B\times h\times M\times d_h},
\qquad O=QC\in\mathbb{R}^{B\times h\times N\times d_h}.
\]

循环意味着复用 block 参数，不意味着复用上一次的 \(Q,K,V,C\)。每次 visit 都必须根据新的点状态重新压缩和重建。禁止增加 \(N\times N\) attention，也禁止增加 latent token 间的 \(M\times M\) self-attention。

本研究保持每个任务原有 \(H\)、head 数、MLP ratio、variant、位置特征、时间输入、输出通道、初始化、数据协议与损失，只在主实验配置中将实际 \(M\) 解析为基础值的 2 倍。

## 2. 通用循环拓扑

使用四个结构字段：

```text
P = prefix_blocks
C = recurrent_core_blocks
R = loop_repeats
S = suffix_blocks
```

模型执行顺序为

\[
\operatorname{Stem}
\rightarrow P_1,\ldots,P_P
\rightarrow (C_1,\ldots,C_C)^{\times R}
\rightarrow S_1,\ldots,S_S
\rightarrow \operatorname{Output}.
\]

派生量：

\[
D_{\rm unique}=P+C+S,
\qquad
D_{\rm exec}=P+CR+S.
\]

约束：

- `prefix_blocks >= 0`；
- `recurrent_core_blocks >= 1`；
- `loop_repeats >= 1`；正式主实验使用 2；
- `suffix_blocks >= 1`，以保证输出头不进入循环核心；
- 四个字段是拓扑唯一真值，`unique_depth` 与 `executed_depth` 只能派生；
- 若兼容旧 `n_layers` 字段，它只能作为记录或一致性断言，不能再实例化另一套 block；
- prefix、core、suffix 的物理 block 彼此独立；core 内的同一个物理 block 在不同 round 严格共享全部参数；
- 原任务已有的物理时间/查询时间条件（例如记作 \(T\) 的时间嵌入）必须原样保留；不得把 logical loop index 当成新的时间步特征注入；
- 每次调用 dropout module 可自然重新采样 mask，不要求不同 round 共用随机 mask；
- 共享必须表现为同一个 module 被反复调用，不能为每一轮 `deepcopy` 一份权重，也不能只在初始化时把两份权重设成相等。

必须内置两个命名 preset：

| preset | P | C | R | S | 独立 block 数 | 实际执行次数 |
|---|---:|---:|---:|---:|---:|---:|
| `p1_c3_r2_s1` | 1 | 3 | 2 | 1 | 5 | 8 |
| `p2_c2_r2_s2` | 2 | 2 | 2 | 2 | 6 | 8 |

同时支持 `custom`，由用户显式给出四个字段。preset 与 custom 字段不得混用或静默覆盖。

## 3. Rank 规则

新 family 的主配置保持 \(H\) 不变，并令

\[
M_{\rm loop}=2M_{\rm base}.
\]

实现必须记录 `base_rank`、`rank_multiplier`、`resolved_rank` 及字段来源。训练新 run 时：

- 默认 `rank_multiplier=2`；
- 允许显式 `rank_multiplier=1` 作为公平控制；
- 允许显式 actual `linearno_rank`，但不得同时显式给 multiplier；
- ShapeNet-Car 的 actual \(M\) 仍须满足其与 \(d_h\) 的整数倍率约束；
- 所有 prefix/core/suffix block 使用同一 resolved \(M\)，不同 round 不能拥有不同 M；
- 不调整 H，不加入逐层 rank、温度或粒度搜索。

## 4. 三种且仅三种残差模式

公共规则：三种模式只改变共享循环核心的残差组织；prefix 和 suffix 继续使用原 LinearNO 的普通未缩放残差。三种模式必须互斥，通过一个稳定枚举选择：

```text
sr_1_over_r
rb_attnres
lb_attnres_1_over_r
```

不得悄悄增加 `sr_unscaled`、`lb_attnres_unscaled` 等第四、第五种正式模式。

### 4.1 `sr_1_over_r`

对循环核心内每一个 operator residual 与 MLP residual 都应用 \(1/R\)：

\[
X^{\rm op}=X+\frac{1}{R}\mathcal{A}_k(\operatorname{LN}_{k,1}X),
\]

\[
X^+=X^{\rm op}+\frac{1}{R}\mathcal{F}_k(\operatorname{LN}_{k,2}X^{\rm op}).
\]

identity path 不缩放。这里的 \(R\) 是循环次数，不是总执行深度，也不是 unique core block 数。

### 4.2 `rb_attnres`

这是 **Kimi-faithful Block AttnRes 在点域上的适配**。一次完整的 recurrent-core pass 视为一个 AttnRes block；若 core 有 \(C\) 个 LinearNO block，则一轮包含

\[
J=2C
\]

个 residual sublayer，即每个 LinearNO block 的 operator 与 MLP 各一个。

定义循环入口锚点

\[
b_0=a\in\mathbb{R}^{B\times N\times H}.
\]

第 \(r\) 轮内部的 raw partial sum 为

\[
p_{r,0}=0,
\qquad p_{r,j}=\sum_{\ell=1}^{j}u_{r,\ell}.
\]

在第一个 sublayer 前，来源为

\[
\mathcal{V}_{r,1}=[b_0,b_1,\ldots,b_{r-1}].
\]

其余 sublayer 前，来源为

\[
\mathcal{V}_{r,j}=[b_0,b_1,\ldots,b_{r-1},p_{r,j-1}],\quad j\ge 2.
\]

当前 sublayer 的输入为

\[
h_{r,j}=\operatorname{AR}_{r,j}(\mathcal{V}_{r,j}),
\]

然后才计算 raw branch output：

\[
u_{r,2k-1}=\mathcal{A}_{k}(\operatorname{LN}_{k,1}h_{r,2k-1}),
\]

\[
u_{r,2k}=\mathcal{F}_{k}(\operatorname{LN}_{k,2}h_{r,2k}).
\]

一轮结束时

\[
b_r=p_{r,J}.
\]

循环核心最终输出为

\[
z_{\rm core}=\operatorname{AR}_{\rm out}([b_0,b_1,\ldots,b_R]).
\]

此模式用 AttnRes 替换循环核心内部的普通 residual accumulation：

- 不执行 `h + u`；
- partial 只累加 raw branch output；
- 每个 operator 和 MLP 前都重新做一次 AttnRes；
- 不是只在每轮边界做一次 AttnRes；
- 不在任何位置使用 \(1/R\)；
- 不访问 latent token，所有来源均为对齐的点域 `[B,N,H]` 状态。

### 4.3 `lb_attnres_1_over_r`

轮内使用 `sr_1_over_r`，只在轮次边界与循环出口使用 AttnRes。令 \(\Phi_{1/R}\) 表示共享核心完整执行一轮：

\[
H_1=a.
\]

对 \(r=1,\ldots,R\)：

\[
Y_r=\Phi_{1/R}(H_r),
\qquad
\Delta_r=Y_r-H_r.
\]

若 \(r<R\)，下一轮入口为

\[
H_{r+1}=\operatorname{AR}_{r+1}([a,\Delta_1,\ldots,\Delta_r]).
\]

最终输出为

\[
z_{\rm core}=\operatorname{AR}_{\rm out}([a,\Delta_1,\ldots,\Delta_R]).
\]

注意：\(\Delta_r\) 已经由内部每条 residual branch 的 \(1/R\) 形成，AttnRes 本身不得再次乘 \(1/R\)，也不得乘来源数量。

## 5. 点域 AttnRes 原语

对同一 receiver 的来源

\[
V_s\in\mathbb{R}^{B\times N\times H},\quad s=1,\ldots,S_r,
\]

定义

\[
K_s=\operatorname{RMSNorm}_{g_r}(V_s),
\qquad
\ell_s=\langle w_r,K_s\rangle_H,
\]

\[
\alpha_s=\operatorname{softmax}_s(\ell_s),
\qquad
\operatorname{AR}_r(V_1,\ldots,V_{S_r})=
\sum_s\alpha_sV_s.
\]

冻结规则：

- softmax 只沿来源轴；每个 batch、物理点独立选择深度来源；
- key 使用 RMSNorm，value 使用 raw tensor；
- 不做 value projection；
- 不除以 \(\sqrt H\)；
- 不乘来源数；
- 不额外使用 \(1/R\)；
- 每个 receiver 有独立的单个 pseudo-query \(w_r\in\mathbb{R}^H\) 与 RMSNorm scale \(g_r\in\mathbb{R}^H\)；
- `w_r` 严格初始化为 0，`g_r` 初始化为 1；
- 本项目合同固定 RMSNorm `eps=1e-6`，并在 metadata 中记录；
- 零 query 导致初始来源权重均匀，这是 Kimi AttnRes 的明确初始化行为，不得通过来源数缩放或偏置把它改成旧 residual；
- receiver 参数按执行位置独立，因此 RB/LB 的残差路由器具有执行位置身份；这不等于向点特征注入 timestep embedding，但论文和文档也不能宣称“全模型没有任何 loop-position-specific 参数”；
- 不做 depth multi-head AttnRes；LinearNO 自身的 h 个 operator heads 不改变上述单 query 深度聚合定义。

参数增量应能由实现与解析式互相验证：

- RB receiver 数为 \(2CR+1\)，新增参数约为 \(2H(2CR+1)\)；
- LB receiver 数为 \(R\)，新增参数约为 \(2HR\)；
- SR 不增加残差路由参数。

第一个 RB receiver 只有一个来源，因此它虽按规格注册 query 与 norm scale，前向必为严格恒等映射，其路由参数通常没有有效梯度；测试和报告必须把这一点作为预期结构性质，而不是误判为断梯度缺陷。

若把一次 receiver 对一个来源的访问记为一次 source visit，则：

\[
V_{\rm RB}=\frac{(2C)R(R+3)}{2}+1,
\qquad
V_{\rm LB}=\frac{R(R+3)}{2}.
\]

因而 `p1_c3_r2_s1` 的 RB/LB 分别为 31/5 次，`p2_c2_r2_s2` 为 21/5 次。每次 source visit 的 query dot 与 value mix 约为 \(2BNH\) MAC，此外还有 RMSNorm 与 softmax；这只是残差路由开销，不能替代全模型实测。

## 6. 明确排除的内容

第一版不实现：

- Transolver loop 化；
- 旧 `linearno_history` 中的 latent-summary AttnRes、history-conditioned K、history dropout、关系感知评分；
- HCC、CKCR、LSAR 或其他旧跨层机制；
- 显式或隐式 feature timestep encoding；
- ACT、动态停机、测试时改变循环次数、elastic depth；
- MoE、LoRA、逐轮 adapter；
- 不同层不同 rank、不同温度、不同粒度；
- 多样性损失或强迫层间核差异；
- 改变 benchmark 数据、split、normalizer、loss、metric 或训练预算；
- 为了“跑通”而用 `strict=False`、忽略未知键、随机补权重或复用不匹配 checkpoint。

## 7. 论文和性能声明边界

两个默认 preset 都实际执行 8 次 block body，但只存储 5 或 6 个独立 block。因此第一版可以研究参数共享、参数存储和优化器状态，不得预先宣称减少了 block FLOPs 或顺序延迟。

同时，主配置把 \(M\) 翻倍。LinearNO 的 \(M\)-相关计算会增加，因此该模型不是 SMELT 式 compute-matched 模型。实施与论文中必须分别报告：

- 独立参数量；
- 实际 block 调用次数；
- MACs/FLOPs；
- 训练吞吐；
- 推理延迟；
- 峰值显存；
- rank 与 residual 的控制实验。

至少保留以下控制能力：原 8-block LinearNO + M、原 8-block LinearNO + 2M、loop + M、loop + 2M。新实现不必增加第四种残差模式，但必须允许 `rank_multiplier=1/2`，以免性能提升全部被错误归因于 loop。

三种模式改变的是循环核心的整套残差组织，其中 RB 同时改变了历史来源拓扑并取消 \(1/R\) branch scaling；所以它们是三种完整残差架构的比较，不能在论文中伪称为严格的单因素消融。需要通过上述 rank/非循环控制组和诊断量拆分可能的增益来源。

---

# 二、总控提示词

```text
你将在用户当前的 hxh5159/CDLNO-w checkout 中实施一个全新的 Looped LinearNO family。开始任何修改前，完整阅读仓库根 AGENTS.md 及其要求优先阅读的状态/报告/证据；再阅读当前纯 LinearNO、linearno_history、八任务入口、checkpoint/resume、launcher、monitor 和测试。当前代码是实施真值，不能假设旧提示词中的路径仍成立。

安全与范围：
1. 先输出 repo 根目录、remote、branch、HEAD、git status --short、tracked/untracked/ignored 摘要。禁止 reset、clean、stash、checkout 覆盖、rebase、commit、push；不得删除或格式化用户已有改动。
2. 只实现 LinearNO loop family。Transolver、CDLNO、KCDNO、MSAR-LNO、纯 LinearNO 和既有 linearno_history 的数学、默认 CLI、state_dict、checkpoint 与结果路径默认冻结。
3. 新代码使用独立 family `linearno_loop`，优先放入 `cdlno/linearno_loop/`，纯 schema 可按当前仓库惯例放入顶层 `linearno_loop/`。新命令放入 `tran_evaluate/linearno_loop/`。若实际审计表明需要不同位置，先在 LL0 报告中说明，不要擅自改旧包语义。
4. 不把旧 latent-history 模块拿来充当本任务 AttnRes。新 AttnRes 只处理点域 [B,N,H]，公式、来源时序和初始化必须符合本提示词。
5. `linearno_loop` 与旧 A/K flags 互斥；任何混用必须在构造模型、读取权重和创建运行目录之前报错。没有 loop flags 的旧命令必须保持原行为。
6. 使用共享物理 module 反复调用，而非每轮复制权重。每轮重新计算 LinearNO 的 Q/K/V/K^T V/QC。
7. 新模型主配置保持 H 不变、M=2×当前任务基础 M；同时保留 M×1 控制。不得把倍增后的 M 硬编码为全任务同一数字。
8. 只支持三种 residual mode：sr_1_over_r、rb_attnres、lb_attnres_1_over_r。三者可以在相同拓扑与训练协议下独立 train/resume/eval。
9. 两个正式 preset 为 P1-C3-R2-S1 与 P2-C2-R2-S2；同时实现完整 custom P/C/R/S。第一版 suffix>=1，循环核心绝不持有最终输出头。
10. 不安装或降级 torch/CUDA/PyG，不访问真实数据或启动长训练，除非当前阶段和用户另行明确授权。单元测试、合成前向/反向、短 synthetic 闭环、dry-run 和只读 profiling 可以执行。
11. 实现必须先有独立数学 oracle，再做生产模块；不能用被测函数生成 expected。所有新 checkpoint 必须 metadata-first、构造参数全量校验、strict=True。
12. 每阶段只做该阶段内容，更新 docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md，给出 PASS/PARTIAL/BLOCKED、实际文件、diff、命令和结果、未运行项、冻结区检查、我最应复核的 3–5 点。最后写“本 LLx 阶段结束，未执行下一阶段”，然后停止。

架构合同以随本提示词提供的“冻结的模型规格”为准。若仓库事实与规格发生会改变模型语义的冲突，完成不受影响的只读工作，列出文件/行、两个可选处理及影响，停止等待用户决定，不能自行猜测。
```

---

# 三、分阶段提示词

## LL0：只读审计与冻结

```text
现在只执行 LL0。禁止修改任何生产模型、parser、factory、训练/评估、checkpoint、launcher、测试 golden 或依赖。只允许新增 docs/LOOP_LINEARNO_REFERENCE_AUDIT.md、docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md 和 docs/loop_linearno_audit/ll0/ 下的只读证据。

1. 完整读取根 AGENTS.md 及其指定的当前状态报告。记录实际 repo、remote、branch、HEAD、tree、git status；建立 tracked/untracked/ignored 清单，不重置或清理。
2. 建立现有源码冻结 manifest，至少覆盖：
   - cdlno/linearno/**
   - cdlno/linearno_history/** 与顶层 linearno_history/**
   - 三个 benchmark 中所有 LinearNO/Transolver 模型文件
   - 八任务真实 parser/factory/train/eval/data/metric 入口
   - cdlno/checkpoint、training_state、experiment、visualization、monitor
   - tran_evaluate/linearno/** 与 tran_evaluate/linearno_history/**
   - 全部现有测试和状态文档
   保存文件分类、size、SHA-256、当前 dirty diff；后续相对 LL0 基线检查，不能把用户原有 dirty 项算成本任务改动。
3. 用 rg --files、rg、AST/源码阅读定位八任务的：模型 class path、constructor、stem、blocks、最后输出头、forward 输入输出、actual M、H、heads、variant、位置/time 语义、loss、metric、checkpoint、resume、eval、launcher。
4. 确认纯 LinearNO 当前每个 block 的真实 operator/MLP 顺序、最后 block 的 ln_3/mlp2、初始化时序、placeholder 时序和 state_dict 键。确认每个任务当前正式 profile 的 base M，而不是只读 argparse fallback。
5. 审计既有 linearno_history 路由，明确哪些入口已经被 intercept。提出 loop family 与 pure/history 共存的最小路由方案，且 loop 与 A/K flags 必须互斥。不要在 LL0 实现。
6. 固定外部来源并保存版本/hash/关键公式摘要：
   - LinearNO paper v3 与 HiPRL/LinearNO 固定实现；
   - Attention Residuals arXiv:2603.15031，重点是 Eq.2–6、Figure 2、zero pseudo-query；
   - Kimi K3 report 中 AttnRes 的使用边界；
   - On the Residual Scaling of Looped Transformers arXiv:2606.18524，重点只作为 1/R 动机，不将其 theorem 冒充 LinearNO 定理。
   外部文件放在 repo 外或明确 ignored 的证据缓存，不整树 vendor。
7. 形成公式到现有类的映射，并逐项检查本文冻结规格是否可在不改变纯模型的前提下实现。特别确认 suffix>=1 是否足以隔离输出头；若不够，列出原因并停止。
8. 给出拟新增文件与最小修改文件清单、阶段边界、风险。明确 P/C/R/S、两 preset、custom、三 residual、M×2、checkpoint metadata 和命令字段的唯一真值。
9. 运行当前能运行的旧 LinearNO、history、monitor 核心回归，记录既有失败/skip；不得通过删测试或放宽容差制造绿色基线。

LL0 PASS 的条件是：仓库现状、现有三条模型线、八任务接线、公式和冻结区均有可追溯证据，且没有未解释的模型语义冲突。完成后停止。
```

## LL1：配置、schema、metadata 与实验矩阵

```text
LL0 已由用户审查通过。现在只执行 LL1：建立 loop family 的纯配置/schema/metadata 合同与测试，不实现 torch 模型，不修改八任务生产 parser/factory/train/eval。

1. 按 LL0 决定建立独立 schema/config 包。schema 层不得 import torch、任务入口或构造模型。
2. family 固定为 `linearno_loop`，architecture_extension 固定为 `loop_linearno_v1`。定义并严格验证：
   - topology_preset: p1_c3_r2_s1 | p2_c2_r2_s2 | custom
   - prefix_blocks, recurrent_core_blocks, loop_repeats, suffix_blocks
   - unique_depth=P+C+S、executed_depth=P+C*R+S，只能派生
   - residual_mode: sr_1_over_r | rb_attnres | lb_attnres_1_over_r
   - base_rank、rank_multiplier、resolved_rank
   - hidden、heads、head_dim、variant、任务与 profile
   - point_domain_attnres=true、feature_timestep_encoding=false
   - AttnRes eps、query/norm初始化、receiver 数量和公式版本
3. preset 与 custom 字段互斥；custom 必须四字段齐全。P>=0、C>=1、R>=1、S>=1。拒绝 bool 冒充 int、未知字段、NaN、字符串数字及不一致派生量。
4. 解析 rank：loop train 默认 multiplier=2；允许1；显式 actual rank 与显式 multiplier 互斥。由当前八任务 profile 计算 base/resolved M，并验证 ShapeNet 倍率约束。旧纯 LinearNO profile 本身不得被改写。
5. 定义计划中的公共 CLI 名称，但本阶段不接 parser：
   --linearno-loop 1
   --linearno-loop-topology PRESET
   --linearno-loop-prefix-blocks P
   --linearno-loop-core-blocks C
   --linearno-loop-repeats R
   --linearno-loop-suffix-blocks S
   --linearno-loop-residual-mode MODE
   --linearno-loop-rank-multiplier 1|2
   继续允许既有 --linearno-rank 作为 actual M 的显式控制，但制定冲突规则。
6. 定义 loop_spec/model_spec/profile_spec/data_spec/objective_spec/evaluation_spec/provenance_spec/normalizer_spec/resume_state。loop_spec 必须保存完整拓扑、残差、rank policy、参数共享规则、AttnRes合同和 schema version。
7. eval/resume 必须从 architecture/checkpoint metadata 恢复 family、拓扑、residual、rank 和构造参数；用户显式字段只作一致性断言。任何冲突在模型构造和 torch.load 权重之前失败。
8. loop family 与旧 linearno_history flags 的任意混用均为非法。省略全部 loop 字段时不得改变纯 LinearNO/history 的解析结果、目录名、RNG 或 schema。
9. 预声明正式矩阵：八任务 × 两 topology preset × 三 residual mode × paired seeds；另有 custom topology 和 rank×1 控制。只生成配置和预览，不训练。
10. 增加 schema/config 的正负测试、序列化/hash 往返、字段来源和旧配置无变化测试。更新 docs/LOOP_LINEARNO_CONFIGURATION.md 与状态后停止。
```

## LL2：点域 AttnRes 原语与 block body 适配层

```text
LL1 已审查通过。现在只执行 LL2：实现独立、任务无关的点域 AttnRes 原语和 LinearNO block body 适配层；不组装完整 loop 模型，不接生产任务。

1. 先写完全独立的 FP64/FP32 oracle：stack来源、RMSNorm key、单 pseudo-query 点积、source softmax、raw value加权。oracle不能调用生产 AttnRes forward。
2. 实现 receiver-specific PointDepthAttnRes：输入若干同 shape [B,N,H] tensor，输出 [B,N,H]。严格执行 eps=1e-6、query=0初始化、norm scale=1、无bias/无value projection/无sqrt(H)/无来源数乘法/无1/R。
3. 验证 source 轴、batch、点与channel，不允许对N做attention或混淆来源轴。检查 device/dtype/shape、空来源、B/N/H不一致并给出明确错误。
4. 实现只调用现有纯 LinearNO block 子模块的 body adapter：
   - operator raw branch = block.Attn(block.ln_1(x))；
   - MLP raw branch = block.mlp(block.ln_2(x))；
   - 能按 native residual、scaled residual、raw branch 三种内部调用语义复用；
   - 输出头只通过显式 finalize 调用最后 suffix block 的 ln_3/mlp2；
   - 不改 block 参数、不复制、不重初始化、不注册缓存历史。
   这些是内部原语，不是新增正式 residual mode。
5. 对 Standard plain/temp/conv/conv_temp、AirfRANS、ShapeNet 的现有 block 做 body 级对照。native adapter 与原 block.forward 在非末层必须达到既定容差；末层在 body+finalize 后对照原 forward。
6. AttnRes 测试覆盖：零 query 均匀权重、非零手算、来源置换同步等变、value保持raw、RMSNorm幅值抑制、输入/参数梯度、B>1/N变化、CPU double/float、可用时CUDA FP32。
7. 参数测试确认每个 receiver 只有H维query和H维norm scale；不出现Q/K/V projection或多头深度参数。
8. 重跑 LL0 旧回归与冻结检查。LL2 不修改任何任务入口、旧模型数学、launcher或checkpoint。更新公式→实现→测试映射后停止。
```

## LL3：通用拓扑与 `sr_1_over_r`

```text
LL2 已审查通过。现在只执行 LL3：组装通用 P/C/R/S loop core，并完成 sr_1_over_r；暂不实现两种 AttnRes residual，不接生产任务 CLI。

1. 用现有纯 LinearNO block 构造 U=P+C+S 个物理 block。只实例化一次；prefix/core/suffix分别注册。最后一个 suffix 是唯一 last_layer=True；core 全部 last_layer=False。
2. forward 顺序严格为 prefix一次、core按相同module顺序调用R轮、suffix一次。每个visit重新执行Attention和MLP，因此重新计算Q/K/V/C/O。
3. prefix/suffix直接保持原 native residual。core的operator与MLP分别使用1/R scaled residual；identity不缩放。
4. 支持两个preset及custom。训练结构字段必须来自LL1 config，模型内部再次断言。公开属性/diagnostic只保存整数和字符串，不保存forward tensor或跨batch history。
5. 为 Standard、Air、Car 建仅用于合成验证的薄wrapper，保持各自输入输出合同；尚不接正式parser/factory。
6. 公平初始化：同task/topology/rank/seed的三残差模式公共backbone必须能得到相同键和值。共享core只初始化一次。不要构造8层后再删除三层来消耗不同RNG，也不要全模型二次apply。
7. 测试：
   - preset A：unique5、executed8、调用顺序P1/C1C2C3/C1C2C3/S1；
   - preset B：unique6、executed8；
   - 至少一个custom，例如P0/C2/R3/S1；
   - module/parameter id证明round间共享，state_dict中无round2复制键；
   - hook证明每个core block调用R次且Q/K/V随当前输入重新计算；
   - 输出头只调用一次；
   - 手算小网络验证两条branch都乘1/R；
   - second-round loss能向同一core参数和first-round计算图反传；
   - strict state_dict roundtrip；
   - invalid S=0、C=0、R=0、派生量伪造拒绝。
8. 参数/MAC计数分别报告unique与executed，不把参数减少写成FLOPs减少。更新状态后停止。
```

## LL4：Kimi-faithful `rb_attnres`

```text
LL3 已审查通过。现在只执行 LL4：在同一通用loop主干实现 rb_attnres，不接生产任务。

1. 以 Attention Residuals Eq.2–6/Figure2 和冻结规格为真值。一次完整core pass是一整个AttnRes block；每个LinearNO block拆成operator、MLP两个residual sublayer。
2. core入口a作为b0。每轮开始没有当前partial；第一个sublayer读取[b0..b_{r-1}]。从第二个sublayer起额外读取当前raw partial。每个sublayer前调用独立receiver。
3. raw partial只累加当前轮raw branch output。不得把AttnRes输入h、anchor或普通residual隐式加进partial。轮末b_r=partial。
4. operator与MLP都以各自AttnRes输出经原LN后计算raw output。不要先执行原block.forward，也不要在其外再套AttnRes。
5. 最终独立receiver读取[b0..b_R]，其输出才交给suffix。全模式无1/R，无额外source-count scale。
6. receiver按(round, sublayer)及output独立注册，共2*C*R+1个。query/norm不跨执行位置共享；core算子参数仍跨round共享。文档准确区分这两种共享。
7. 历史只存在于单次forward局部变量：不parameter/buffer、不断图、不detach、不跨batch、不跨NS rollout调用、不跨Plasticity时间查询、不跨ensemble成员。
8. 独立oracle覆盖完整小core，不只测AttnRes原语。逐步比较每个source列表、权重、h、raw u、partial、b_r和final。
9. 结构/负向测试：
   - C3/R2 receiver=13、参数=26H；C2/R2 receiver=9、参数=18H；
   - C3/R2 的逐 receiver 来源数必须是 `[1,2,2,2,2,2 | 2,3,3,3,3,3 | 3]`，C2/R2 必须是 `[1,2,2,2 | 2,3,3,3 | 3]`；竖线仅表示 round/final 分界；
   - 第一个receiver只有anchor，输出精确为anchor；
   - 零query在多来源时产生均匀平均；
   - query非零后每点权重可不同；
   - 不产生N×N或M×M attention；
   - 跨层latent token从未进入来源；
   - operator/MLP前调用次数精确；
   - 除“单来源且恒等”的第一个 receiver 外，所有活动 router 参数在合适的非零 query 测试状态下有 finite gradient；第一个 receiver 的 query/norm 无有效梯度是预期行为；
   - 异常后不遗留history。
10. sr_1_over_r回归及全部旧模型冻结通过后，更新报告并停止。
```

## LL5：`lb_attnres_1_over_r` 与三模式统一

```text
LL4 已审查通过。现在只执行 LL5：实现lb_attnres_1_over_r，并把三种残差统一为互斥、可构造、可序列化的正式loop core模式；仍不接任务生产CLI。

1. 实现Phi_1/R：一轮内每个core block的operator和MLP residual均乘1/R。
2. 记录每轮入口H_r、输出Y_r和Delta_r=Y_r-H_r。Delta不能从raw branches重新近似，必须由这两个实际点状态相减得到。
3. 第r轮后若还有下一轮，独立boundary receiver读取[a,Delta_1,...,Delta_r]。最终output receiver读取[a,Delta_1,...,Delta_R]。
4. boundary/output AttnRes沿用LL2原语；不再次缩放Delta，不乘1/R，不乘来源数。不得在轮内operator/MLP前调用AttnRes。
5. receiver数量固定为R-1个boundary加1个output，即R；参数量2HR。R=1时只有output receiver，语义仍必须明确和可测，但正式实验R=2。
6. 三种mode通过单一枚举分派。未知mode失败；不得通过多个bool形成非法联合。非当前模式的router不得注册参数或污染state_dict。
7. 对R2小模型逐张量oracle：Y1/Delta1/H2/Y2/Delta2/final。验证zero-query就是来源均匀平均，不加隐藏补偿因子；此时必须精确满足 H2=(a+Delta1)/2，final=(a+Delta1+Delta2)/3。
8. 验证三模式公共backbone在相同seed/topology/rank下键和值一致；RB/LB只多各自router键。三模式均完成forward/backward/optimizer一步/strict roundtrip。
9. 给出默认两拓扑的实际新增参数和理论式一致性；测量而非猜测额外MAC。更新docs/LOOP_LINEARNO_CORE_REPORT.md与状态后停止。
```

## LL6：六个 Standard benchmark 生产接入

```text
LL5 已审查通过。现在只执行 LL6：把linearno_loop接入Airfoil、Darcy、Elasticity、Pipe、NS、Plasticity；不接AirfRANS/Car，不执行真实数据长训练。

1. 复用当前纯LinearNO profile、数据、normalizer、objective、metric、optimizer/scheduler、时间循环、可视化和记录器。新family只覆盖模型构造、loop metadata与strict checkpoint，不复制或改写任务科学协议。
2. 按LL0最小路由方案接入：只有显式loop train或metadata family=linearno_loop的eval/resume进入新adapter。省略loop字段的旧pure/history路径和model_dict结果必须完全不变。
3. 新Standard模型保持现有Model(x,fx,T=None)输入输出与各variant语义。构造U个物理block，默认resolved M=2×base M；不要先构造8个独立block再丢弃。
4. parser接入LL1公共字段。新train必须显式给topology与residual mode；eval/resume从run metadata恢复。preset和custom、actual rank和multiplier、loop与A/K冲突均在构造/torch.load前拒绝。
5. checkpoint沿用仓库已验证的pair/manifest/resume/RNG协议，但family/schema/loop_spec/class_path必须独立。state_dict strict=True；共享core权重只保存一次。resume恢复optimizer state时验证参数组、shape、拓扑和router键。
6. 目录名至少包含task/profile/P-C-R-S/residual/resolvedM/seed；同一目录不可覆盖或跨残差resume。eval不得猜最近run、不得改训练sidecar或重拟合normalizer。
7. 六任务原生合成闭环覆盖2 preset×3 residual：真实parser→factory→原train/eval主干→checkpoint→新进程resume/eval。NS十步调用与Plasticity二十查询每次forward都重新建立loop局部状态。
8. 回归：纯LinearNO、linearno_history、旧Transolver的入口、模型输出/state键和既有测试。对被改路由文件做去除新guard后的AST等价或更强证据。
9. 不生成虚构SOTA结果。真实loader、500epoch、远端GPU均NOT RUN并列出将来准确命令。更新LL6报告后停止。
```

## LL7：AirfRANS 与 ShapeNet-Car 生产接入

```text
LL6 已审查通过。现在只执行 LL7：接入AirfRANS和ShapeNet-Car；不改两任务现有数据、loss、metric、sampling、fold或ensemble协议，不跑真实训练。

1. AirfRANS保持forward(Data)->[N,4]、单图、可变N、原位置reference-distance拼接、dead temperature键和多成员ensemble。ShapeNet保持forward((cfd_data,geom))->[N,4]、单图、原7通道输入、tempreature拼写与fold/drag协议。
2. 两个wrapper只把纯模型的物理blocks替换为相同初始化合同下的P/C/S loop组织；stem、placeholder、reference/pos buffer、任务输出不改变。
3. resolved rank默认AirfRANS 32→64、Car 32→64，但仍由profile解析。Car验证M/head_dim整数倍率。
4. main/eval入口只在显式loop或saved family匹配时路由。旧pure/history/Transolver whole-object与state_dict路径不受影响。loop checkpoint使用新安全schema和strict state_dict，不从不可信whole-object猜结构。
5. Air每个ensemble成员独立模型、optimizer、router和局部history；不能跨成员共享core实例或缓存。resume覆盖成员内与成员间中断边界。
6. 用真实PyG对象的内存合成测试覆盖两preset×三residual、train一步、pair checkpoint、新进程resume/eval、Air多成员、Car非零fold与surface/drag接口边界。明确合成不等于真实VTK指标。
7. B/N、batch/ptr、device/dtype、custom topology、错family/residual/topology/rank/router键均在正确层级拒绝。
8. 重跑LL6六题及旧Air/Car/pure/history/Transolver回归。更新LL7报告后停止。
```

## LL8：统一 launcher、train/resume/eval 命令与公平运行

```text
LL7 已审查通过。现在只执行 LL8：提供八任务统一launcher、命令文档、公平初始化/数据流协议和dry-run矩阵；不执行真实训练。

1. 新增tran_evaluate/linearno_loop/<task>.sh，接口统一为train|resume|eval。复用path.sh和当前任务原launcher，不复制训练逻辑。
2. train示例合同：
   bash tran_evaluate/linearno_loop/airfoil.sh train \
     --linearno-loop 1 \
     --linearno-loop-topology p1_c3_r2_s1 \
     --linearno-loop-residual-mode sr_1_over_r \
     --linearno-loop-rank-multiplier 2 \
     --linearno-profile paper_table8_on_release_model \
     --seed 17 --gpu 0 --experiment-dir RUN
   另给preset B与custom四字段示例。eval/resume只需GPU与确切RUN；若显式重复结构字段，只作冲突断言。
3. 三残差比较必须共享同task/topology/rank/profile/seed的公共backbone初始化与DataLoader generator序列。router初始化不应推进或改变公共backbone RNG。记录backbone初始hash与数据顺序seed。
4. 不把两种topology称为参数量相同：A有5个独立block，B有6个。运行manifest保存unique/executed depth、actual call schedule、参数量和router参数量。
5. dry-run矩阵至少覆盖8任务×2preset×3mode×train/eval，并覆盖resume、custom、rank×1及所有结构冲突。每个run路径唯一，不覆盖旧pure/history结果。
6. 文档给出未来三paired seeds的完整循环示例、每任务数据路径含义、预期产物和未执行声明。不得选择test最优seed/checkpoint。
7. shell语法、引用、带空格路径、GPU映射、错误传播和train成功后才允许eval的顺序测试通过。更新docs/LOOP_LINEARNO_COMMANDS.md与状态后停止。
```

## LL9：综合数值验证、成本核算与可选诊断

```text
LL8 已审查通过。现在只执行 LL9：做完整无数据综合验证、参数/MAC/FLOPs/延迟/显存工具和可选loop诊断；不得宣称SOTA或真实epoch效率。

1. 建立独立解析式与实际module统计：每任务、两preset、三mode、rank×1/2的参数量；分stem、prefix、shared core、suffix/head、router。验证RB 2H(2CR+1)、LB 2HR。
2. MAC/FLOPs同时报告unique参数与executed调用。LinearNO operator按实际N/H/h/M/variant统计K^T V与QC；不得把M翻倍模型标成compute-matched。标出不含softmax/RMSNorm/GELU等scalar ops的口径。
3. CPU固定环境测forward、forward+backward/optimizer的median/p90；若本机有CUDA且当前规则允许，只做合成FP32/AMP smoke与peak memory，记录硬件和同步方法。不得外推真实数据epoch。
4. 综合矩阵：八任务×两preset×三mode的小模型/正式shape可承受forward，至少完成配置、构造、state、调用计数、forward/backward、checkpoint。Standard与工业生产合成闭环引用LL6/LL7证据并重跑关键项。
5. 可选诊断保持默认关闭，若实现则记录logical round、physical core index、sublayer、source权重/熵、state norm、Delta norm、Q/K路由统计。不得修改旧monitor定义或把Transolver核图冒充LinearNO loop证据。
   RB 的凸来源混合并不自动继承 1/R residual scaling 的稳定性结论；诊断还应能记录每次共享 core 访问的 update norm、相邻更新 cosine、入口/出口 RMS、共享参数 gradient norm 与 NaN/Inf。诊断只观测，不得据此偷偷加入 gate、缩放或裁剪。
6. 验证没有N×N、M×M attention，没有跨forward history，没有输出头多次调用，没有round-specific core权重副本。
7. 完整重跑现有pure/history/Transolver/CDLNO相关回归；保留所有既有失败/skip的来源，0个新失败才可PASS。
8. 输出docs/LOOP_LINEARNO_PERFORMANCE.md、机器可读JSON和状态。明确参数减少、M翻倍、router开销和实际测量边界后停止。
```

## LL10：反向源码审计与最终交付

```text
LL9 已审查通过。现在只执行 LL10：不增加新功能，逐条从冻结公式反查实现、完成最终回归与交付报告。

1. 从模型forward反向追踪到：Stem→prefix→R轮共享core→suffix→单次head。逐task核对P/C/R/S、共享对象、执行次数、M、H、variant和输出合同。
2. 分别逐行验证：
   - SR的operator/MLP均1/R、identity/prefix/suffix不缩放；
   - RB每sublayer前AR、raw partial、round summary、final AR、无1/R；
   - LB轮内1/R、Delta实际相减、boundary/final来源、AR不二次缩放；
   - AR的RMSNorm key/raw value/source softmax/zero query/one norm scale，无sqrt、无投影、无N²/M²。
3. 检查初始化/RNG：公共backbone同seed跨mode相同；共享block只初始化一次；query0/norm1；placeholder与纯任务原时序合理且有证据。
4. 检查state/checkpoint：family/loop_spec/rank/topology/residual全量保存，metadata-first，strict=True，optimizer/RNG/DataLoader resume完整，共享参数只存一次；错配置在权重加载前拒绝。
5. 检查兼容：旧pure LinearNO、linearno_history、Transolver及其命令/checkpoint/结果路径未被新默认污染；冻结manifest除LL阶段已批准的最小接线外无漂移。
6. 执行最终目标测试、全相关回归、compile/shell/diff检查。对每条命令记录实际结果、耗时、环境、失败/skip；禁止只写“全部通过”。
7. 生成docs/LOOP_LINEARNO_IMPLEMENTATION_REPORT.md：来源版本、架构公式、文件映射、八任务配置、两preset、三mode、命令、参数/成本、测试证据、已知限制、真实实验待办。
8. 给出最终PASS/PARTIAL/BLOCKED。真实数据、三seed、完整epoch、SOTA、远端GPU若未运行必须明确NOT RUN，不能影响“实现范围PASS”的定义，也不能被描述成已验证性能。

完成后停止，不启动任何真实训练，不commit/push。最后写“本 LL10 阶段结束，未执行真实实验”。
```

---

# 四、建议的最终公开命名

为避免和旧 latent-history 项目混淆，建议代码与实验统一采用：

- 模型族：`Looped LinearNO`；
- 内部 family：`linearno_loop`；
- 标准缩放残差：`SR-1/R`；
- 轮级 Kimi Block AttnRes：`RB-AttnRes`；
- 混合边界残差：`LB-AttnRes-1/R`；
- 拓扑：`P1-C3-R2-S1`、`P2-C2-R2-S2`。

不要把 `RB-AttnRes` 写成“整个模型等同 Kimi K3”。准确说法是：循环核心采用 Kimi-faithful Block AttnRes residual topology，LinearNO operator、任务协议与输出结构仍来自本项目。

---

# 五、参考来源

- 当前目标仓库：<https://github.com/hxh5159/CDLNO-w>
- LinearNO paper v3：<https://arxiv.org/abs/2511.06294v3>
- LinearNO 官方仓库：<https://github.com/HiPRL/LinearNO>
- Attention Residuals：<https://arxiv.org/abs/2603.15031>
- Kimi K3 report：<https://arxiv.org/abs/2607.24653>
- On the Residual Scaling of Looped Transformers：<https://arxiv.org/abs/2606.18524>
