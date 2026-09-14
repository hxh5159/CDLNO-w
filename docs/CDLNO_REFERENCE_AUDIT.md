# CDLNO 阶段 0：材料与源码审查

更新日期：2026-09-13。阶段：**0（仅阅读、核对和文档记录）**。

主体保留阶段0审查记录；阶段2实际实现与差异复核见文末§11及 [CDLNO_PHASE2_MODULES.md](CDLNO_PHASE2_MODULES.md)。当前执行状态以 [CDLNO_IMPLEMENTATION_STATUS.md](CDLNO_IMPLEMENTATION_STATUS.md) 为准。

本阶段没有修改模型、训练、数据、评价或依赖配置，没有下载数据或启动训练。用户指定的阶段边界优先于 v1.2 §11 的一次性实施指令。

## 1. 审查范围与证据规则

已审查：

- `PLAN_CDLNO/比较模型架构.pdf`：151 页聊天导出，已逐页通读；pp.82、103–104、111–124、128–150 是设计核对重点。
- `PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md`：当前主要实施规格。
- `PLAN_CDLNO/CDPA_v1_1/CDPA_Transolver_Implementation_Plan_v1_1.md`：历史补充、环境候选和验收框架。
- 数学附件：本工作区未找到计划点名的 `CDPA_Mathematical_Foundations.md`、`CDPA_Theory_Manuscript.tex/.pdf`、`check_theory_identities.py`、`theory_identity_checks.json`。PDF pp.147–150 有理论摘要，但不能替代附件全文。用户说明附件已提供；实际路径需后续补充，不能据此断言附件不存在或声称本地已读附件全文。
- 当前 Transolver 工作区及三个实验项目；未 import 会在顶层读取数据的 `exp_*.py`。
- LRSA-Operator：`https://github.com/Adversarr/LRSA-Operator.git`，commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`，提交 `Publish.`（2026-05-02，作者 Zherui Yang）。该快照没有发现根目录 `LICENSE`/`COPYING` 文件；复制代码前需向上游确认许可，或按审查后的公式重写。
- IPOT：`https://github.com/7tl7qns7ch/IPOT.git`，commit `18c177846267505ee9503445a146dfd7dee34c41`，提交 `Update README.md`（2024-07-31，作者 Seungjun Lee）；根目录为 MIT License（Copyright 2023 Seungjun Lee）。

历史助手在 PDF 中报告的论文阅读、理论证明、NumPy 检查或远端 GPU 结果，除非本阶段实际复现，否则只记录为历史报告，不标作本地验证。

## 2. 当前仓库基线

| 项目 | 实际状态 |
|---|---|
| 分支 | `main` |
| HEAD | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` |
| 远程 | `origin git@github.com:thuml/Transolver.git` |
| HEAD 提交 | Merge pull request #25 from `jaeminoh/main`，2026-02-26 |
| `git status --short` | 仅有未跟踪 `AGENTS.md`、`PLAN_CDLNO/`、`docs/`、`memory/`；没有已跟踪代码 diff |
| 用户已有代码修改 | 未发现；未执行 reset、checkout、commit、push |
| 数据/训练 | 本阶段未下载数据、未启动训练 |

仓库保留 Transolver 原始三套入口：`PDE-Solving-StandardBenchmark/`、`Car-Design-ShapeNetCar/`、`Airfoil-Design-AirfRANS/`，以及根目录 `Physics_Attention.py`。项目内有局部 attention 副本，不能假定根目录文件就是所有入口实际导入的实现。

## 3. 我理解的最终 CDLNO 架构

### 3.1 整体目标

CDLNO（Cross-Depth Latent Neural Operator）以 Transolver 的数据和任务入口为基线，在模型内部把少量完整 LRSA 风格的点域压缩—潜空间处理—重建 block、一次 IPOT 风格 bridge、CDPA 对不同深度 latent 历史的读取与融合、多个只在 latent 中运行的 processor block，以及一个面向输入输出同一节点集合的 LRSA 风格最终读出连接起来。目标是减少昂贵的 N 点规模往返和点域处理，同时保留足够表达能力；这不是无损保存历史信息的定理，也不是已证明的加速保证。

### 3.2 默认计算图

默认 `L=8`、`F=2`、`P=L-F=6`：

```text
原任务输入提升 → H0 [B,N,d]
  → LRSA block 1（完整 down/latent FFN1/SA/FFN2/up/点残差/点FFN）
       保存 T1 [B,M,d]（latent FFN2 后、up 前）并得到 H1 [B,N,d]
  → LRSA block 2（同样完整）
       保存 T2 [B,M,d] 并得到 HF=H2 [B,N,d]
  → IPOT bridge：learned query residual + H_F cross-attention
       得到原始 Z0 [B,M,d]
  → 入口 CDPA：Z0 分别查询 T1、T2，逐 token 做跨来源 softmax
  → 6 个独立 persistent latent block：pre-LN self-attention + GEGLU FFN
       得到 ZP [B,M,d]
  → LRSA feature-conditioned decoder：H_F 查询 ZP 的 K/V
       H_F 残差 + 点 FFN/ConvFFN + 输出 LN/head
  → 原任务要求的输出形状
```

`L` 是包含 latent self-attention 的处理 block 总数，不把 bridge、CDPA 和最终 readout 偷算进 8 层。`F` 在 `0..L-1` 范围内配置，`P=L-F` 且至少为 1；当 `L=8` 时必须覆盖 `F=0..6`，扩大 `L` 时不能把前段上限硬编码为 6。各阶段使用同一个标量 `M`。

### 3.3 前段完整 LRSA block

每个前段 block 参数独立，顺序为：点域 pre-norm并保留入口 `H`；独立 learned latent queries 将 `[B,N,d]` 压到 `[B,M,d]`，点特征产生 K/V；latent FFN1 残差；latent self-attention 残差；latent FFN2 残差得到完整 `T_i`；保存 `T_i`（不 detach、不保存 attention 矩阵）；用入口点特征作 query、`T_i` 作 K/V 执行 up-attention；加回点入口残差，再执行点 FFN。规则网格点 FFN 使用 LRSA 风格普通 `Conv2d(d,d,3,padding=1,groups=1)` ConvFFN，非结构化点使用逐点 FFN。保存 `T_i` 不得删掉本层 up、点残差或点 FFN。

### 3.4 Bridge 与 persistent latent

Bridge 使用独立 `Q_0∈R^{M×d}`：

`Z0 = Q0 + CrossAttn(LN(Q0), LN(HF), LN(HF))`。

保留 IPOT learned-query residual；不注册 IPOT 源码中 forward 未调用的 `encoder_ff`。Bridge 对 `F=0` 仍工作，此时读取 `H0`。每个 persistent block 独立实例化，采用 pre-LN self-attention + GEGLU FFN residual；GEGLU 首版 ratio=2。后段不重新读取 N 个点，不执行 down/up，不在 latent 编号上加卷积，不跨真实物理时间缓存状态。

### 3.5 CDPA 的确定计算

CDPA 是用户确认的机制名，等同于此前的 CDPA-Cross；CDPA-Slice 不在首版范围。

对每个历史 `T_s` 独立执行当前到历史的 Cross：

```text
Q   = LNq(Z) WQ
Ks  = LNk_v(Ts) WK
Vs  = LNk_v(Ts) WV
Rs  = MHA(Q, Ks, Vs) WO + bO
R0  = Z
es[m] = wᵀ RMSNormdepth(Rs[m])
alpha[m] = softmax_source(es[m])
Zfused[m] = sum_s alpha_s[m] Rs[m]
```

同一个 CDPA 位置内各来源共享投影和归一化；不同 CDPA 位置使用独立参数。每个历史内部的 softmax 只沿其 M 个 token，不能把多个来源拼成长度 `S*M` 后统一 softmax。来源可折叠到 batch 或分块执行，但数学和梯度必须与逐来源路径一致。`R0=Z` 走恒等路径，不经过 Cross 或 `WO`。`w` 全零初始化，故首步是所有 RAW 候选均值；两份历史时为 `(Z+R1+R2)/3`，不是 identity 初始化，也不能再做外部 `Z+Zfused`。评分、softmax、加权归约使用 FP32 并保留梯度，返回前恢复输入 dtype。

默认 `entry` 只在第一个 persistent block 前执行一次。`off` 不收集历史；无历史时严格为 identity 且不注册无用 CDPA 参数。可选 `every_block` 的第 `j` 个后段 block 前，当前 identity 是 `Z_(j-1)`，Cross 历史是 `[T1,...,TF,Z0,...,Z_(j-2)]`；当前状态不重复加入，不加入未来状态、SA/FFN 中间结果或跨 forward 历史。entry 是默认效率主设置，every_block 是补充消融。

### 3.6 最终 decoder

八个当前任务采用 feature-conditioned 读出：

```text
Q = Norm(HF) WQ_up
K = Norm(ZP) WK_up
V = Norm(ZP) WV_up
DeltaH = CrossAttn(Q,K,V)
HD = HF + DeltaH
Hout = HD + PointFFN/ConvFFN(Norm(HD))
yhat = Linear_out(LN_out(Hout))
```

最终 query 和点残差来自 `H_F`。这只适用于输入输出节点存在明确对应关系的当前八项任务；不实现坐标 query decoder 或稀疏 Darcy。相同 N 数量本身不足以证明对应关系。

## 4. 已放弃方案、实现默认与理论边界

### 已放弃或当前禁止导入

- PDF p.82 已放弃 Gram/统计/EMA 校正、逆矩阵、正交或校正缓存路线。
- p.103–104 的点域只读物理记忆和点侧 K/V 缓存不是最终方案；历史来自 LRSA 中间 latent。
- 早期坐标 query decoder 退出八项当前任务首版；只在未来稀疏 Darcy 记录中保留。
- 不采用 CDPA-Slice、latent 卷积、跨层参数共享、额外 PDE/守恒损失、局部图网络、自动 sweep、NS 10→20/40（没有同条件长轨迹标签）。
- 不把 Kimi AttnRes 原始 residual replacement、Slot Attention GRU/竞争迭代或 OmniNet 逐通道 max pooling 当作本模型实现。

### 计划明确的实现级默认

- `L=8,F=2,P=6`，`cdpa_mode=entry`，`dropout=0`，后段 GEGLU ratio=2。
- 全阶段一个 `M`；v1.2 起始表：Darcy 64、Elasticity 64、Airfoil 64、Pipe 32、NS 64、Plasticity 64、ShapeNet-Car 64、AirfRANS 64；宽度/头数按 v1.2 表，不用旧 parser 默认值替代。
- 前段和最终读出按任务使用 LRSA 后置 dense ConvFFN；潜空间无卷积。
- `off/entry/every_block`、F/L、来源 chunk size 是新模型参数；不能影响旧 Transolver 分支默认行为。
- 正式命名 `cdlno` / `CDLNO`；旧计划中的 `cdpa_operator` / `CDPAOperator` / 整网注册 `CDPA` 仅是旧占位映射。机制配置仍称 CDPA，原 wrapper 可以继续导出 `Model`。

### 理论边界

- 相邻 LRSA block 结构容量相同，但输入点状态、参数、压缩/写回方向不同；冻结线性路由的残差秩分析可说明不同方向累积，不能把完整动态 attention block 的 Jacobian 直接写成 rank≤M。
- 某些 PDE 的谱衰减、POD 或解流形可压缩性支持低维/潜空间近似；不能声称所有 PDE 天然低秩、低秩等于低频，或固定 M 自动普适。
- CDPA 只有在历史保留目标方向、Cross 未消除、来源融合未抵消且后段/decoder 能利用该方向时，才可能补充 off 的局部 H_F 与 bridge 无法访问的全局信息。
- 最终 decoder 有 `H_F` 局部点/3×3 邻域旁路，因此 CDLNO 不是纯最终 latent 瓶颈。不能宣称无损压缩、重训后的 off 函数类严格包含、所有任务误差必降、PDE 守恒或 GPU 必然加速。

## 5. 八任务接口与训练连接

| 任务 | 模型输入与形状 | 输出 | 位置/时间条件 | 模型构造、损失与保存 |
|---|---|---|---|---|
| Elasticity | `pos [B,972,2]`；`fx=None` | `[B,972,1]` | 位置直接输入提升；无时间 | `space_dim=2,fun_dim=0,out_dim=1`；`TestLoss`；原脚本保存 `./checkpoints/<name>.pt` state_dict |
| Darcy | `pos [B,s²,2]`；系数 `fx [B,s²,1]` | `[B,s²,1]`，脚本 squeeze | 规则网格；无时间 | `fun_dim=1,out_dim=1`；`TestLoss + 0.1` 中心差分梯度损失；原 eval 常用 `strict=False`，新模型用独立 sidecar 严格校验 |
| Airfoil | 规则索引网格 `pos [B,H·W,2]`，通常 221×51；`fx=None` | `[B,H·W,1]` | 原入口的位置/任务条件；无时间 | `Transolver_2D`、`TestLoss`；保持 checkpoint/result 语义，CDLNO 使用独立输出 |
| Pipe | 规则网格 `pos [B,H·W,2]`，通常 129×129；`fx=None` | `[B,H·W,1]` | 位置；无时间 | `Transolver_2D`、`TestLoss`；实际 batch=8 |
| Navier–Stokes | `pos [B,4096,2]`；历史 `fx [B,4096,10]` | 每次 `[B,4096,1]` | 10 帧输入；训练真实帧回填、测试预测帧回填；当前 10→10 | `fun_dim=10,out_dim=1`；每步 `TestLoss`，10 步累计后一次 backward/update；不改时间协议 |
| Plasticity | `pos [B,101·31,2]`；`fx [B,101·31,1]` | 每次 `[B,101·31,4]` | `T [B,1]`，20 个时间查询 | `Time_Input=True,fun_dim=1,out_dim=4`；逐时间 forward，保留原逐时间 optimizer 更新节奏 |
| ShapeNet-Car | PyG `cfd_data/geom_data`，单图变 N，节点特征/位置 | `[N,4]`：速度3+压力1 | 图位置、表面 mask 和 geometry 对象 | 原 train：速度 MSE + `reg*` 表面压力 MSE；batch=1；当前保存完整模型 `metrics/.../model_*.pth`；多图需明确拒绝 |
| AirfRANS | PyG `Data(x,pos,y,surf)`，单图变 N，通常约 32000 | `[N,4]`：vx、vy、p、nut | signed distance、来流/法向、surface mask；无时间 | 原 train：volume MSE + `reg*` surface MSE；采样、半径图、后处理冻结；当前实际 398 epochs；保存 `metrics/.../model` |

标准入口 `exp_*.py` 在模块级解析参数并访问数据路径，不能在测试中直接 import。模型适配应保持 `Model(x,fx,T=None)->[B,N,C]`；工业 wrapper 保持原 PyG 调用和节点顺序。Plasticity 的实际 optimizer 更新位置和旧脚本细节仍需在实现阶段逐行保护，不能按 IPOT 语义改写。

## 6. 参考源码核对与计划冲突

### LRSA 实际代码

关键文件：`src/perceiverforpde/modeling/layers/attn.py`、`perceiver.py`、`perceiver_structured.py`、`layers/mlp.py`。

- 完整顺序是 down attention → latent FFN/channel mixing 1 → latent self-attention → latent FFN/channel mixing 2 → up attention。
- Down 使用 learned latent query、点 K/V；up 使用点 query、latent K/V。
- Structured `ConvNextConv` 的 `dwconv` 实际是普通 `Conv2d(d,d,3,padding=1)`，无 `groups=d`；后接 channels-last norm 和 pointwise linear/激活/linear。
- LRSA 提供 RoPE、attention gate、mass、xFormers/Liger 等可选路径；v1.2 首版不全部移植。
- 快照未找到独立许可证文件；必须保留来源记录，不能直接把整段代码视作已有可复制许可。

### IPOT 实际代码

关键文件：`models/ipot/ipot_encoder.py`、`ipot_processor.py`、`ipot_decoder.py`、`layers.py`。

- Encoder 使用 learned latents、pre-norm cross-attention 和 query residual；定义的 `encoder_ff` 未在 `forward` 调用。CDLNO bridge 保留 residual，去除无效 FFN。
- Processor 是 pre-norm self-attention + FeedForward residual；该快照即使 `weight_tie_layers=False` 也反复放入同一 attention/FFN 对象，会造成参数共享。CDLNO 后段必须逐层新建独立模块。
- FeedForward 是 GEGLU：第一层输出 `2*mult*d`，分 value/gate，经 GELU 相乘后映射回 d。
- Decoder 支持任意输出 query；CDLNO 八任务首版采用 H_F feature decoder，坐标 query 只记录给未来稀疏 Darcy。

### Transolver 实际代码与 v1.2 冲突及处理

1. 旧 `exp_*.py` 常见默认 `n_layers=3,n_hidden=64,n_heads=4,slice_num=32`，而 CDLNO 默认 L=8/F=2/P=6 且 M/宽度/头数按 v1.2。**处理：** 新模型使用独立显式参数和任务默认；旧 Transolver 分支默认不改，不能全局改 parser。
2. 旧 block 是 slice→slice-token attention→deslice+点 MLP，不是 LRSA。**处理：** CDLNO 另设核心模块，原模型继续可用。
3. Transolver 结构化 attention 的卷积式投影在 slice 前；v1.2 要求 LRSA 重建后 ConvFFN。**处理：** 不复制 Transolver 前置卷积，只在新模型指定点 FFN 位置使用 LRSA ConvFFN。
4. 旧最后 block 直接投影输出；CDLNO 需要 H_F、bridge、persistent latent 和 feature decoder。**处理：** 只改新 wrapper 的模型组织，最终输出形状不变。
5. 旧脚本有 `strict=False` 或直接保存完整模型；CDLNO 权重不能与 Transolver 混用。**处理：** 新输出目录、架构 sidecar，先读 sidecar 再验证架构字段。
6. 旧 requirements 固定 `torch==1.10.1`，与用户远端目标 Python 3.10/3.11、PyTorch 2.11、CUDA 12.8 不同。**处理：** 本阶段不安装或替换依赖；后续以用户远端环境和 v1.1/v1.2 cu128 候选为依据，不用本地版本替代。

## 7. 允许修改与必须冻结

### 允许修改（需用户逐阶段授权）

- 新建共享 `cdlno/` 包及 CDLNO/CDPA 核心模块。
- 必要的模型注册、参数转发、新模型专用输出目录和 checkpoint sidecar。
- 不触碰数据格式的 task wrapper，保持原入口调用合同。
- 无数据合成测试、数学 reference 对照、前反向、checkpoint round-trip 和独立性能计数工具。

### 必须冻结

- 数据下载、预处理、字段选择、样本划分、采样、点顺序、网格索引、surface mask、normalizer、目标通道。
- Loss、梯度损失、surface/volume 权重、指标、反归一化和物理后处理。
- NS 的 10 帧输入、真实/预测回填语义和现有 10 步评估。
- Plasticity 的时间条件、20 个查询和逐时间更新节奏。
- Adam/AdamW、学习率、scheduler、epoch、batch、clip 及原脚本行为；新模型只增加必要参数而不改变旧模型路径。
- 原 Transolver 模型、脚本、数据入口和许可文件。

## 8. Kimi K3 AttnRes 与最终 decoder 的再次核对

已阅读工作区外可定位到的 Kimi K3 技术报告 `/home/hwz/LaKDA/references/kimi k3.pdf`，重点核对 §2.2、式 (8)–(10)。原文 Full AttnRes 对每个深度位置使用 learned pseudo-query `q_l=w_l`，把 embedding 和之前层输出作为 key/value，在深度来源轴做 softmax 加权和；Block AttnRes 则先形成块内部分和，再对块级表示做跨深度读取。原文没有把当前 latent 作为 query，逐个历史 latent 做 `M→M` Cross。

因此，用户所说的入口 CDPA

```text
Z0 分别查询 T1、T2
每个历史独立做 Cross attention
加入 R0=Z0
对历史来源做逐 token softmax 融合
```

应精确定义为：同一 `Z0[B,M,d]` 只生成一次 Q；对 `T1[B,M,d]`、`T2[B,M,d]` 分别执行两次独立的 `Q(M)-K/V(M)` attention，得到 `R1`、`R2`；再把 `[R0=Z0,R1,R2]` 在每个当前 token `m` 上沿来源轴做 softmax 融合。历史 token 轴的 attention softmax 和来源/深度轴的 softmax 是两个不同归一化，不能把 `T1,T2` 拼成长度 `2M` 的一个 Cross，也不能先各自做来源融合再平均。`R0` 不经过 Cross 或 `W_O`，且不再执行外部 `Z0+fused`。

这保留了 Kimi AttnRes 的“跨深度、按 token 选择历史表示”思想，同时增加了适应 CDLNO latent token 语义所需的 Cross 对齐。它不是 Kimi Full/Block AttnRes 的逐式复制：Kimi 的 pseudo-query 是学习参数，CDLNO 的 Q 来自当前 `Z0`；Kimi 的历史通常是同 token 身份的层输出或块表示，CDLNO 的 `T_i` 是 LRSA down/latent FFN2 后保存的压缩表示；CDLNO 还明确保留 identity 候选和每历史独立 Cross。

再次核对 LRSA-Operator 的实际 `PerceiverUpProject`：point features 产生 Q，latent features 产生 K/V，attention 输出由外层 block 加到原 point features，再执行 point FFN/ConvFFN。v1.2 的最终 decoder 与此方向一致，具体为：

```text
Q = Norm(H_F) WQ_up       # H_F 是 query 的来源
K,V = Norm_z(Z_P) WK/WV   # latent 只提供 K/V
DeltaH = Up(Q,K,V)
H_D = H_F + DeltaH        # H_F 原值是点残差旁路
H_out = H_D + PointFFN/ConvFFN(Norm(H_D))
output = Linear_out(LN_out(H_out))
```

这里的 `Norm(H_F)` 是 v1.2 对最终读出的明确规范；LRSA 参考实现的 up 层内部还可有独立 q/k RMSNorm，不能因此删掉计划要求的外层 norm。最终 decoder 只有一次 latent-to-point up、一次 `H_F` 点残差和一次点 FFN/ConvFFN；不重新 down、不做 latent self-attention、不使用坐标 query。规则网格的 ConvFFN 必须使用原任务已知的 `(H,W)` 重排，不能用 `sqrt(N)` 猜形状，也不能改变节点顺序。`H_F` 旁路只提供对应点（规则网格时最后一次 `3×3` ConvFFN 的局部邻域）信息，不能描述为绕过压缩后仍可直接访问全部输入。

结论：这两处当前设计与 v1.2 及参考源码语义一致；实现时要防止“历史拼接成 `2M` 做单次 Cross”“把来源 softmax 与历史 token softmax 混为一个轴”“把 `R0` 再加到融合结果外面”以及“把最终 query 错换成 `Z0`、`H_0`、坐标或 latent”。

## 9. 阶段 0—10 状态表

| 阶段 | 目标 | 当前状态 |
|---:|---|---|
| 0 | 材料阅读、源码审查、接口/许可/冲突记录 | **本阶段完成** |
| 1 | 新共享包骨架、命名与无数据导入边界 | 待用户明确授权 |
| 2 | LRSA 前段、bridge、persistent block、feature decoder | 待授权 |
| 3 | CDPA entry/off/every_block 与来源 batch/chunk 等价路径 | 待授权 |
| 4 | 标准六任务模型选择和参数转发 | 待授权 |
| 5 | ShapeNet-Car/AirfRANS wrapper 接入 | 待授权 |
| 6 | checkpoint sidecar、独立输出和加载校验 | 待授权 |
| 7 | 无数据合成合同、数学 parity、前反向和损失连接 | 待授权 |
| 8 | matched LRSA/CDLNO 合成性能计数与计时工具 | 待授权 |
| 9 | 用户远端环境检查与真实数据前运行准备 | 待授权；不在本地替代执行 |
| 10 | 用户授权后的真实数据训练、精度和端到端效率实验 | 待授权；当前禁止自动执行 |

## 10. 阶段 0 交付报告

### A. 完成范围

完成聊天导出、v1.1、v1.2、当前 Transolver 基线、LRSA/IPOT 快照、八任务入口和源码冲突审查；记录最终设计、废弃方案、实现默认、理论边界、许可和冻结/允许区段。用户随后明确的远端环境优先级已纳入，未把本地依赖作为项目依据。

### B. 修改文件、理由和 diff 摘要

- `docs/CDLNO_REFERENCE_AUDIT.md`：新增本审查报告。
- `docs/CDLNO_IMPLEMENTATION_STATUS.md`：更新为阶段 0 完成并链接本报告。
- `AGENTS.md`、`memory/current-state.md`、`memory/README.md`：同步阶段 0 边界、远端环境优先级和报告入口。

本阶段不修改模型、训练、数据、评价或依赖文件；不执行 git add/commit/push。

### C. 关键公式/张量形状与代码对应

阶段 0 没有 CDLNO 实现代码，因此公式只与 v1.2 设计和 LRSA/IPOT 实际来源映射：点状态 `H_i [B,N,d]`，历史/latent `T_i,Z_j [B,M,d]`，CDPA 每历史内部 token softmax 后跨来源 token-wise softmax，最终 decoder 使用 `H_F`。实际代码映射留到获准实施阶段，不把设计记录当作实现。

### D. 实际验证命令、环境、通过/失败/未运行

| 检查 | 状态 |
|---|---|
| `git branch --show-current`、`git rev-parse HEAD`、`git remote -v`、`git status --short`、`git diff --stat` | 已运行并记录；无已跟踪代码 diff |
| 计划、聊天 PDF、Transolver、LRSA、IPOT 源码静态读取 | 已完成 |
| LRSA/IPOT commit、remote、文件和 license 检查 | 已完成；LRSA 未发现独立 LICENSE，IPOT MIT |
| 数据入口、训练、损失、保存/加载静态审查 | 已完成；未 import `exp_*.py` |
| 用户目标远端 Python 3.10/3.11、PyTorch 2.11、CUDA 12.8 | **未在本地替代检查**；以后只在用户授权的远端环境阶段检查 |
| 本地 Python/torch 版本 | 曾被工具读取，仅为工作机观察，**不作为项目兼容性依据** |
| CDLNO forward/backward、checkpoint、真实数据/训练/性能 | 未运行；尚无实现且当前禁止 |

### E. 冻结区域变化证据

本阶段只新增/更新文档。模型、数据读取/预处理、字段、划分、采样、点序、归一化、标签、损失、训练循环、优化器/调度器、评价及原脚本未修改。结束时以 `git diff --check`、`git status --short` 和逐文件哈希审查确认；不执行 git add/commit/push。

### F. 未解决问题与优先审查点

1. 独立数学附件实际路径仍需用户提供或后续授权阶段定位；当前只使用 PDF 理论摘要和计划公式边界。
2. LRSA 快照没有独立许可证，后续应采用公式指导下重写或先确认许可，不能直接复制整段源码。
3. 旧 parser 默认与 CDLNO 任务默认冲突，后续必须使用新模型显式配置，不能全局改默认。
4. 阶段 1 具体范围尚未授权；阶段表不构成自动授权。

建议优先审查：T_i 是否确实在 FFN2 后、up 前保存；bridge 是否无效 FFN；CDPA 是否为独立 Cross、`R0=Z`、FP32 depth softmax 且无外部 `Z+fused`；decoder 是否使用 `H_F`；数据、loss、时间循环和远端依赖是否冻结。

**本阶段结束，未执行下一阶段。**

## 11. 阶段2实际模块与参考差异复核

用户已确认阶段0、1通过，本次仅实现独立模块。源代码为 `cdlno/modules.py`，逐项形状、初始化与测试记录见 [阶段2报告](CDLNO_PHASE2_MODULES.md)。没有复制参考项目的数据/训练框架或新增依赖。

| 对照点 | 本次采用的实现/验证 |
|---|---|
| LRSA down→FFN1→SA→FFN2→up | 完整保留；返回的T为up前完整latent且仍有梯度。原 `SinglePerceiverBlock` / `StructuredPerceiverBlock` 在相同FFN/QK norm/bias/权重下完成输出、T和梯度对照 |
| LRSA参考位置 | `/home/hwz/LRSA-Operator`，commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`；`src/perceiverforpde/modeling/layers/attn.py`、`perceiver.py`、`perceiver_structured.py`、`layers/mlp.py` |
| Norm/FFN/config | 显式选plain GELU、ratio2、RMS外层/per-head QK、QKV无bias/O有bias；不导入YAML或隐藏宽度roundup规则。默认参数不冒充论文训练preset |
| Dense ConvFFN | 普通groups=1的3×3卷积，up和点残差后才执行，内部affine LN；非对称核手算验证5×7点序、跨通道混合和边界 |
| LRSA初始化/可选kernel | 普通Linear统一trunc_normal(std=.02)，特殊query单独初始化；Conv2d原生默认。未用上游父模块递归reset/xformers/Liger/RoPE/gate/mass |
| IPOT源码 | `/tmp/IPOT-remote`，commit `18c177846267505ee9503445a146dfd7dee34c41`，`models/ipot/ipot_encoder.py`、`ipot_processor.py`、`layers.py` |
| IPOT encoder_ff | 不定义无消费者FFN，bridge公式仍是query residual加一次Cross；零化Cross输出后严格返回learned Q0 |
| IPOT共享/PreNorm差异 | 6个独立构造的后段对象及storage均不同；Q/K/V确认为同一个LN(Z)对象，与未修正源码的raw context有明确差异。采用§2.6显式公式核对输出和梯度 |
| 最终读出 | H_F生成query并作为原值点残差，Z_P仅作K/V。分支外层RMSNorm；输出§2.7的LN_out使用LayerNorm。一次up加一次点FFN，不新增down/SA或坐标decoder |

同配置LRSA点/结构版两项FP64对照通过，该样例点输出、T及所有对应梯度误差均为0。另18项模块测试通过，包括实际本地GPU FP32/FP16/BF16 autocast；执行机为torch2.13/cu130，不能当作用户远端torch2.11/cu128验收。初次oneDNN BF16反向失败、GPU TF32精度差异和仅在测试内调整backend的处理已保存在阶段2报告。

本次未发现需要改变已确认架构或任务协议的源码冲突。LRSA未找到独立许可证的原记录继续有效，核心按计划公式重写，测试只读取既有参考checkout。整模型、CDPA、任务损失、checkpoint接入和真实训练均未在阶段2执行。

**本阶段结束，未执行下一阶段**
