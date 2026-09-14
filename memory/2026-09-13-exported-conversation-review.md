# 《比较模型架构》完整对话阅读与 v1.2 对照记录

日期：2026-09-13。当前阶段：阅读、理解、准备后续修改；只维护 `AGENTS.md` 和 `memory/`，不修改代码、计划、依赖或数据。

## 来源、覆盖范围与证据等级

- 原文：[比较模型架构.pdf](../PLAN_CDLNO/比较模型架构.pdf)，用户确认它是分享链接 `https://chatgpt.com/share/6aa60a96-cfb4-83ea-a6a9-de8ca19e55fb` 的完整导出。
- 共 **151 页，已从第 1 页到第 151 页逐页通读**；页码均指 PDF 页码。用 PyMuPDF 提取逐页正文，补读被工具输出截断的段落，并视觉核对第 1、121、141、148、150 页。公式、表格以原 PDF 为准，文本提取的公式顺序有时落在段落后。
- PDF 大小 2,630,420 字节；SHA256：`bf6dd80940df07df92e160f6eac00a1a0634fe7e1e0c1e5e65295b8ce23b33de`。
- 最终实现规格：[CDPA_Transolver_Implementation_Plan_v1_2.md](../PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md)。既有来源核查：[LRSA/IPOT 审计](2026-09-13-cdpa-plan-and-reference-audit.md)。
- 本记录区分：**用户明确选择、历史助手建议、已核查源码事实、最终计划规定、尚未验证的研究假设**。PDF 内的“已检查论文/已编译/已运行脚本”是历史助手的报告，不自动成为本工作区的验证结果。
- 对话内的旧实施请求不是当前执行授权；用户当前明确要求不要编辑代码。v1.2 是最终工程规格，导出提供细节与生成过程，不应把每一阶段提案累加到最终模型。

## 设计演变与废弃分支

| 页码 | 对话内容及转折 | 后续工作应如何使用 |
|---|---|---|
| 1–38 | 比较 Transolver、LinearNO、LRSA、LNO、FLARE 等的压缩、混合、重建、残差与路由 | 形成问题背景；保留随后对早期说法的纠正 |
| 38–48 | 探索表示秩/交互秩、Gram 条件、区域秩、细节路径、双空间、signed/differential attention 等 | 研究候选，均不自动增加到主模型 |
| 48–82 | Gram 校正、统计/EMA 校正、浅层替代与成本分析 | **第 82 页用户明确暂时放弃校正方向**；不增加逆矩阵、校正层、正交损失或校正缓存 |
| 82–92 | LRSA 重复点域读写与 IPOT 持续 latent；卷积位置与效率核算纠正 | 效率目标是减少昂贵的 N 规模读写与点域处理；不等同于无条件削减深度 |
| 92–98 | 数个完整 LRSA block 后接 IPOT 风格后段；初期使用输出坐标 decoder | 完整前段后的独立 bridge 必须保留；坐标 decoder 后来退出首版 |
| 98–103 | 助手把 AttnRes 延伸为点域只读记忆，提出共享点 K/V | **不是用户最终意图**，不应移植其 N 规模后段读取或成本公式 |
| 103–110 | 用户明确要连接前段中间 latent 与后段编码；解释与原 AttnRes 的区别 | 复用既有压缩结果；不是再次读取点；直接逐编号融合尚有对齐问题 |
| 111–116 | 用户提出不断增长的历史，以及 Cross / Slice 两种 latent 对齐 | 形成“先对齐 token、再选择深度”的 CDPA；不能假设不同层相同 slot 编号天然对应 |
| 116–119 | 比较 entry 与 every_block 的成本/调用次数 | entry 成为效率优先方案；早期估算仍含旧卷积、旧 decoder 与跨层 KV 缓存假设 |
| 119–124 | 用户先选 CDPA-Cross；确定 source 内 Cross 与逐 token 深度 softmax、RAW values、w=0 | 最终 CDPA 核心公式来源；第 124 页用户同意该建议 |
| 125–128 | 助手推荐 LRSA 作基仓库；用户在第 126 页明确选择 Transolver，以保留方便的数据流程 | **Transolver 是基仓库**；只适配架构，不能导入 LRSA 的数据格式/训练框架 |
| 128–130 | 用户改选 LRSA 的卷积方式；讨论 latent 的索引不构成物理邻接 | 首版使用重建后 ConvFFN；不继承更早的压缩前卷积，不加 latent 卷积 |
| 130–136 | 各论文主配置、消融、效率、泛化中的 M 及其歧义 | 超参数依据；不可把头数×M 或层数×M 当成单层全宽 latent 长度 |
| 136–143 | 八任务、NS 时间协议、2+6、相同 M、feature decoder 与 H_F 旁路 | 首版以 H_F 构造最终 query 并保留点残差；仅 N_in=N_out 不足以保证节点对应 |
| 143–144 | 用户明确未来稀疏 Darcy 无卷积前段、IPOT 坐标 decoder，并将 CDPA-Cross 简称 CDPA | 稀疏任务设计仅记录；CDPA-Slice 等待用户再次提出 |
| 144–145 | v1 计划固定可调 F/L、entry/off/every_block、八任务合同与无数据验收 | 默认 2+6；F 至少覆盖 0–6、P=L−F；Pipe batch 与 Plasticity 更新节奏以核实结果为准 |
| 145–147 | 用户补充 torch 2.11/cu128；v1.1 补环境、来源批处理、独立位置投影和公平计时 | **取消跨 CDPA 位置缓存投影 KV**；批处理仅改变执行组织，不改变数学计算量 |
| 147–150 | 四级理论依据、局部读出访问限制、条件方向恢复；生成 v1.2 | 理论不改变架构；不能把条件定理扩大成无条件精度/模型类优势 |
| 151 | 助手推荐完整模型名称 CDLNO，机制名仍为 CDPA | 补充命名来源；不代表代码已经改名或收到新的实施授权 |

## 当前模型为何如此设计

研究目标是：以少量完整点域更新产生可复用的 latent 统计量，通过 CDPA 对齐并融合历史，让后段持续在小规模 latent 中计算，检验能否以较少点域处理保持或改善精度，并取得实际延迟收益。

“重复压缩重建有用”不能直接推出“只有重复压缩才能产生表达能力”。新 LRSA 层会面对已更新的点状态，可能提取旧历史未保留的信息；历史复用能够利用已保存的统计量，无法保证替代任意后续点域读取。后段 self-attention 并不必然逐层丢信息，残差的存在也不等于可逆或无损。

最终计算图按 v1.2 §§2–4 固定：

```text
原任务 stem → H_0
F 个完整 LRSA block：H_(i−1) → down → latent FFN1 → SA → latent FFN2 → T_i
                                                          T_i → up → 点残差/FFN → H_i
H_F → 独立 bridge → 原始 Z_0
(Z_0, T_1…T_F) → entry CDPA → P 个独立 latent SA+GEGLU block → Z_P
(H_F, Z_P) → LRSA feature-conditioned up → H_F 残差 → 点 FFN/ConvFFN 残差 → 输出 LN/head
```

- **T_i 取点**：latent FFN2 后、up 前；它保存完整 latent 状态，不是 residual 分支增量。保存它不能删掉后续 up、点 FFN 或 bridge。历史保留梯度。
- **Bridge**：独立 learned query residual 加 pre-LN cross-attention；不照抄 IPOT 未参与 forward 的 encoder FFN。前段 LRSA down 则不额外加 learned-query residual，两者有意不同。
- **后段**：每层独立 pre-LN SA + GEGLU，ratio=2；不复制 IPOT 重复追加同一模块造成的权重共享，不读取 N 点，不跨真实时间保留状态。
- **卷积**：五个结构化任务（Darcy、Airfoil、Pipe、NS、Plasticity）在前段和最终 readout 使用 LRSA 式 dense 3×3 ConvFFN，groups=1。Elasticity 和两工业任务用点 FFN。此统一选择是新模型设计，不是 LRSA 所有任务原脚本都启用卷积。
- **信息进入卷积的时机**：T_1 尚不含本层后置卷积，T_2 可含第一次卷积影响，Z_0 可含两次影响。不能把 T_i 说成本层 ConvFFN 后的压缩。
- **最终 decoder**：query 来自 H_F，latent 提供 K/V；H_F 同时有真正的隐藏特征残差旁路。位置 query 与 feature query 都可受输入影响，但后者查询端额外直接依赖节点条件。精度归因必须把 decoder 旁路与 CDPA 分开。
- **节点合同**：主版要求输入与输出节点一一对应，含材料参考点身份；不是仅要求相同数量。保留原索引邻接，不对 Airfoil/Pipe 按坐标重排。F=0 时直接桥接 H_0，但最终结构化 ConvFFN 仍有一次，不能称作原始 IPOT。

## CDPA 的两级计算与 AttnRes 边界

对应 PDF pp.111–124、145–147 与 v1.2 §3。省略 batch/head 布局：

```text
Q = LN_q(Z) W_Q
K_s = LN_kv(T_s) W_K
V_s = LN_kv(T_s) W_V
R_s = ConcatHeads(softmax_history_token(Q K_sᵀ / sqrt(d_h)) V_s) W_O + b_O
R_0 = Z
e_s[m] = wᵀ RMSNorm_depth(R_s[m])
alpha_s[m] = softmax_source(e)[s,m]
fused[m] = sum_s alpha_s[m] R_s[m]
```

- 每份历史单独在自身 M 个 token 上归一化，再在 S+1 个候选间选择深度。历史都由同一个当前 query 读取，结果具有当前 token 的索引；不依赖历史原 slot 编号相同。
- 共享投影/norm 指同一次融合的不同来源。各目的 CDPA 位置独立：不得跨位置复用投影 K/V，连 learned LN 后的表示也不能直接缓存复用。PDF pp.116–118 的跨层共享缓存建议被 pp.145–147/v1.2 §3.6 取代。
- 来源折叠到 batch `[B*k,h,M,d_h]` 或分块不改变 key 长度 M；不能改成 kM token 的统一 softmax。各 chunk 的候选最后共同做深度 softmax，不能先分块融合后平均。
- 当前状态仅作为 RAW identity 候选，不做自对齐，不再额外加 `Z+fused`。每份历史结果也不额外加 Z 或单独接 FFN。
- 深度权重在合并多头及 O 投影后计算：每 token、每来源一个标量，作用于整个 d 维向量；不是全样本统一权重或每通道独立门控。
- `w=0`、RMSNorm scale=1，Cross 使用常规非零初始化。初始融合是 S+1 个 RAW 候选的均值，默认两历史为 1/3 各占一份；**不是 identity 初始化**。无历史时才严格为 identity，且不注册闲置 CDPA 参数。
- 深度 RMS、评分、softmax、加权归约使用 FP32，并显式关闭该子图 autocast；保持梯度，返回前转回输入 dtype。不能仅用 `.float()` 便认定后续 autocast 运算一定在 FP32。
- 来源 softmax、评分归一化与其凸组合界只说明当次融合的性质，不证明整网稳定、守恒、物理对齐正确或无损。当前没有绝对深度 embedding、source bias、Slot Attention 的竞争迭代/GRU。

原 AttnRes 聚合同一 token 在深度上的分支输出或块内分支和；CDPA 聚合独立压缩产生的完整 latent，经 Cross 对齐后才使用其评分范式。Kimi、Slot Attention、OmniNet 提供思想来源，不是已经提出并验证了本架构。Slot 的目标轴 softmax 与 CDPA-Cross 的历史 token 轴不同；OmniNet 的跨深度逐通道 max pooling 也不是这里的加权融合。

### 历史时序

| 模式 | 当前状态与读取历史 | 生命周期 |
|---|---|---|
| off | 不调用 CDPA | 不收集无用历史 |
| entry | 第一个后段前，Z_0 为 identity，Cross 读取 T_1…T_F | 只使用前段历史，不增长后段历史 |
| every_block，第 j 块前 | identity=Z_(j−1)；Cross=[T_1…T_F, Z_0…Z_(j−2)] | 保存已有完整 block 输出；不含未来、SA/FFN 中间量或额外融合结果 |

Z_0 特指未经 CDPA 覆盖的 bridge 输出。每个 block 完成后，把该 block 的原输入加入后续历史，当前输出成为下一 block 的 identity；避免当前重复计入。F=0/every_block 第一位置无 CDPA，第二位置开始可读 Z_0。所有历史只在当前 forward 内存在，不能跨样本、时间预测或 optimizer step 复用。未来使用 activation checkpoint 也应传不可变历史快照。

## 实验、容量与效率的最终口径

默认 L=8、F=2、P=L−F=6。至少支持 F=0…6，实际有效范围 0≤F<L、P≥1；L 增大时不把 F 上限写死为 6。L 统计含 latent SA 的处理 block，bridge/CDPA/readout 另计。

| 任务 | d | heads | 全阶段 M | 点域 FFN |
|---|---:|---:|---:|---|
| Darcy | 128 | 8 | 64 | ConvFFN |
| Elasticity | 128 | 8 | 64 | 逐点 |
| Airfoil | 128 | 4 | 64 | ConvFFN |
| Pipe | 128 | 4 | 32 | ConvFFN |
| NS | 256 | 8 | 64 | ConvFFN |
| Plasticity | 128 | 8 | 64 | ConvFFN |
| ShapeNet-Car | 256 | 8 | 64 | 逐点 |
| AirfRANS | 256 | 8 | 64 | 逐点 |

这是约定的首版起点，不是新模型已验证最优配置。PDF pp.130–136 的论文盘点提示：LRSA 的 NS 主配置表写 M=32，效率表明确用 64，部分误差表对应不清；IPOT 经常用更大 M，但其宽度、深度、训练协议不同。不得按相同 M 或相同“8 层”自动宣称公平。Transolver/LinearNO 的每头 M 不应乘头数后称作 M*h 个全宽 latent。

保留的任务协议：

- 六项应称“标准 PDE benchmarks”，不与专名 PDEBench 数据库混淆。沿用 Transolver 三套数据/训练入口，不迁移 LRSA HDF5 数据格式或 IPOT 训练器。
- NS 默认数据 T20，前 10 帧条件、后 10 帧真值；模型一次输出一帧。训练真实帧回填，十步损失累计后一次 batch backward/update；测试预测帧回填。10→20/40 需至少 30/50 帧同条件轨迹。IPOT 不同黏度的不同预测长度不能冒充同条件长时消融。
- Plasticity 每次处理 101×31 空间点，以 T 条件分别预测 20 个时刻，各时间查询各自更新参数；不改成 62620 个时空 token 或 NS 式反馈。
- 标准模型输出 `[B,N,C_out]`；Car 接收 `(cfd_data,geom_data)`，AirfRANS 接收 data，均保持 `[N,4]`、节点顺序、mask、采样、反归一化与物理量后处理。工业单图支持变 N，多 graph 显式拒绝，不能把多样本连成一个场。
- 训练划分、字段、normalizer、loss、metric、optimizer、scheduler 与原预算冻结；Pipe 实际 batch=8，AirfRANS 实际 398 epochs，不能沿用 PDF 早期示意表中的 4/400。
- 新 checkpoint 需严格架构 sidecar；先读旧配置再比较，不能先覆盖再验证。chunk/device/dtype/backend 是运行信息，变化本身不代表权重架构不兼容。

效率以 v1.2 §9 为准，旧 pp.117–118 的数字建立在尚未最终确定的配置上，不作为主版性能结论。令 S 为所有 CDPA 位置读取的非 identity 历史总数、A 为活跃位置数：

```text
S_off = 0
S_entry = F
S_every = P*F + P*(P−1)/2
C_matched_lrsa_attention = 2d [2LNM + LM²]
C_new_attention = 2d [2(F+1)NM + (L+S)M²]
C_cdpa_projection ≈ Md² (A+3S)
C_cdpa_attention = 2SM²d
```

均为单样本主矩阵 MAC；batch 再乘 B。默认 entry：S=2,A=1；every：S=27,A=6，投影分别约 7Md²/87Md²。早期跨层缓存假设下的更低投影成本不适用最终实现。

2+6 的 down/bridge 与 up/readout 各 3 次，latent SA 共 8 次，规则 ConvFFN 共 3 次；同设置八层 matched LRSA 的 down/up/ConvFFN 各 8 次。全来源批处理使 CDPA SDPA 调用为 entry 1 次、every 6 次，但数学上仍分别读取 2/27 份历史；API 调用不等于 GPU kernel 数。

attention 矩阵项的节省条件是 `2(P−1)N > S*M`；N≫M 时 entry 约为对照的 37.5%，**不能改写成整网速度比**。真实比较需计点投影、dense ConvFFN、所有 FFN、LN/QKVO、临时 stack/expand、副本与保留的 H_F/梯度图；统一 dtype、batch、TF32、SDPA backend、compile 设置及计时方法。matched LRSA 是合成结构性能对照，当前不自动扩展为额外真实训练体系。精度主 baseline 仍是原 Transolver。

## 数学支撑及可用于论文的边界

PDF pp.147–150 与 v1.2 §§2.8、14 给出主线：**局部读出和 bridge 的访问限制 → 历史保留目标信息 → Cross 对齐和深度融合有条件地引入该信息**。以下记录原文展示的关系和假设，不代替缺失附件的完整证明与独立复核。

### 相邻 LRSA 层的不同作用

相邻层结构容量相同，但参数、输入点状态、压缩方向与写回方向可不同。冻结路由的线性代理 `h_j=(I+R_j)h_(j−1)`，`R_j=B_j K_j C_j`，若 rank(R_j)≤r_j，则：

```text
rank((I+R_2)(I+R_1)−I) ≤ r_1+r_2
rank(R_2 R_1) ≤ min(rank(R_1), rank(R_2))
```

不同方向的两个 rank-1 残差更新可共同修改两个方向；没有点残差时纯乘积仍受较小秩限制。真实模型含输入条件路由、FFN、norm、卷积和点 query，其完整 Jacobian 不能直接套 rank≤M。压缩与重建解耦也不自动构成正交投影、双正交基或最优传输。

### PDE 的低秩与非线性潜表示

对正自伴 A、紧逆 A⁻¹、递增正特征值 λ_j，固定线性解算子谱截断满足最优 rank-m 算子范数误差 λ_(m+1)⁻¹；正时间热半群截断误差为 exp(−tλ_(m+1))。这为某些 PDE 的一次编码/潜计算/一次解码提供存在性依据。

系数到解 `a↦u(a)` 是非线性映射，某一个固定 A_a⁻¹ 的谱衰减不足以证明整个数据族共享低秩。仍需参数化解流形宽度、POD 平均误差等条件。移动阶跃族可以由一个非线性坐标描述，却未必有同样紧凑的共同线性空间；解码稳定性也需考虑。不能声称所有 PDE 天然低秩、低秩等于低频，或有限固定 M/d 自动具备普适性。

对话提到单次全局平均配合足够宽非线性编码/解码的算子逼近理论，以限制“只有反复 slice/deslice 才有表达能力”的说法；这里不凭对话摘要补写缺失的文献标题、完整假设或证明。

### CDPA 与最终局部读出的条件关系

冻结前端和 bridge，对输出点 i 定义 `V_i(x)=(H_F(x)|邻域_i,Z_0(x))`。不规则读出邻域为该点，结构化读出为最终一次 3×3 ConvFFN 邻域；H_F 本身已经可以含前段传播的全局信息。

off 的该点预测只能是 V_i 的函数。若 `V_i(x_+)=V_i(x_−)` 而目标不同，则同一冻结前端下任何 off 后段至少在一者上误差不小于目标差范数的一半。这个条件不要求**完整** H_F 相同，因此没有忽略真实存在的点旁路。

在可微处，设 `D V_i(x*) v=0`，CDPA scorer 为 w=0，则原文给出：

```text
D C(x*) v = (1/(S+1)) sum_s [D_(T_s) R_s] [D T_s(x*) v]
```

当前 Z_0 对 v 的导数为零，初始深度权重恒为均匀，因此该关系来自历史链路。要形成输出上的有效增益，历史需保留此方向、Cross 不能消掉、源间不可抵消、后续 processor/readout 必须能够利用它。历史 LN 会消掉 token 通道均值平移，说明“保存原历史”不保证读取保留所有变化。

PDF 报告了一个同冻结前端下 off 最优 MSE=1、CDPA 可为 0 的有限样本构造，但其完整附件在本地缺失；本次没有复现该构造或运行 NumPy 核查。也没有证明允许 bridge/front 重新训练后，off 仍不能解决该任务。

零化新增模块不是严格模型类包含证明：若历史 R_s=0、w=0，则 `C=Z/(S+1)`，并非 Z。不能为方便证明偷偷增加外部 residual/gate、正交约束、PDE loss 或删 norm。RAW 凸融合的逐点范数界不等于整网非扩张、能量稳定或误差必降。

四级参考框架是：基础谱/POD/宽度/条件期望/残差秩/Jacobian；LRSA/IPOT/LNO/LinearNO/Transolver；连续 attention、参数化椭圆解流形、全局平均算子逼近；相关 ICLR 工作。引用已有定理必须注明来源，改符号不构成新定理；ICLR 2025 抛物问题的复合误差组织方式也不意味着本模型就是其假设下的 Picard 迭代。

## 命名、未来分支与文件缺口

PDF 第 151 页最后推荐：完整模型 **CDLNO — Cross-Depth Latent Neural Operator（跨深度潜空间神经算子）**，仓库/包 `cdlno`，类 `CDLNO`；机制 **CDPA — Cross-Depth Physics Attention**。CDPA 简称是用户在 pp.143–144 明确指定；CDLNO 的具体类/包命名是末尾助手建议，PDF 没有后续用户确认。v1.2 §6.1 的 `CDPAOperator` / `cdpa_operator` 是先前工程占位名称；本轮仅记忆其先后关系，不编辑计划或重命名代码。

未来稀疏 Darcy 已在 pp.143–144 约定：前段 LRSA 去空间卷积，只在已观测点上完成 down/up，bridge 和 CDPA 保留，后段采用 IPOT 风格 latent，最终由完整输出坐标查询解场；不得用未观测真实系数造 query。**当前不实施**。递增 M、长 NS horizon、CDPA-Slice 同样不进入首版；Gram/统计校正路线已经放弃。

用户已报告远端 CUDA12.8 / torch2.11 / PyG+pyg-lib 可训练原 Transolver ShapeNet-Car；不能据此说八任务已验证，也不能重装旧 requirements 降级。v1.1 环境清单、constraints 与预检脚本已在 `PLAN_CDLNO/CDPA_v1_1/`，但本轮没有安装或执行环境探针。

仍缺：`CDPA_Mathematical_Foundations.md`、`CDPA_Theory_Manuscript.tex/.pdf`、`check_theory_identities.py`、`theory_identity_checks.json`。导出 pp.147–150 给出理论摘要、下载链接文字和“已做 8 项检查/9 页 PDF 编译”的历史报告，并没有嵌入完整证明文件或核查产物。记录为**摘要可读、附件不可用、未在本地复现**，不要生成假附件或沿用“已验证”状态。

本轮完成的是全量阅读、设计对照与文档记忆更新。没有实现新模型、适配器、测试或 benchmark；没有下载数据、训练、测精度/延迟或验证远端 GPU。后续实施的无数据验收仍按 v1.2 §§7–9 执行，不能把本轮阅读当成验收通过。
