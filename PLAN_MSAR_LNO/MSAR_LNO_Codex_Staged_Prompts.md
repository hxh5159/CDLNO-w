# MSAR-LNO：基于现有 KCDLNO 仓库的分阶段 Codex 实施提示词

版本：v1，2026-09-16。

本文用于指导 VS Code 中的 Codex，在已经实现 Transolver、CDLNO、CDLNO 前段消融和 KCDLNO 的现有仓库上，继续增加一个独立的多尺度 latent 神经算子。为使代码入口稳定，本文暂用：

- 模型显示名：`MSAR-LNO`；
- 代码 family：`msar_lno`；
- 含义：Multi-Scale Attention-Residual Latent Neural Operator。

这是代码阶段的工作标识，不强制作为最终论文名称。实现后不要自动重命名它，也不要改动已有模型的名字、导入路径或 checkpoint family。

## 使用方式

1. 先把“总控提示词”发送给 Codex，再单独发送 `M0`。
2. 每次只发送并授权一个阶段。Codex 完成后检查 diff、报告和实际命令，再决定接受、返修或进入下一阶段。
3. 本次阶段编号为 `M0—M9`，与原 CDLNO 阶段、前段消融 `A0—A4`、KCDLNO 的 `K0—K10` 区分。
4. 当前仓库的实际代码是唯一集成对象。本文给出的类名是语义名称；Codex 必须先按仓库现有命名、factory、配置和任务目录映射，不能假设文件路径。
5. 当前没有真实数据。所有阶段都应完成源码、合成输入、真实模型、checkpoint 和可用 PyG 层面的检查；不得伪造真实数据、收敛、精度或真实 epoch 时间。
6. 默认配置为 Light。Full 只需保证可构造、可保存/加载和能完成缩小输入的功能检查，不自动启动大规模训练。

| 阶段 | 工作范围 | 用户主要审查内容 |
|---|---|---|
| M0 | 完整仓库审计和修改前回归依据 | 是否真正理解现有仓库；是否识别用户已有修改 |
| M1 | 独立 family、配置、profile 和 checkpoint 元数据 | 新旧参数是否隔离；Light/Full 是否准确 |
| M2 | 独立 Down、latent block、Up、AttnRes、coverage 原语 | 公式、张量轴、初始化、可关闭损失 |
| M3 | 组装四级 encoder–decoder core | 计算顺序、同尺度 slot 对齐、模块计数 |
| M4 | 接入现有模型选择与可选训练损失 | 旧模型输出合同不变；off 模式只用 PDE loss |
| M5 | Darcy／Elasticity／Airfoil／Pipe | 四个静态任务的数据和训练语义不变 |
| M6 | Navier–Stokes／Plasticity | 时间循环、rollout、loss 聚合不变 |
| M7 | ShapeNet-Car／AirfRANS | PyG、可变 N、checkpoint 实际格式不变 |
| M8 | 无真实数据的综合回归和消融闭环 | 八任务、两 profile、loss on/off、旧模型回归 |
| M9 | 参数/计算/有限计时、独立复查与交付 | 不夸大效率和精度；文档与实际代码一致 |

## 总控提示词：首先发送

```text
现在继续修改当前已经实现 Transolver、CDLNO、CDLNO 前段消融和 KCDLNO 的现有仓库，新增一个独立模型 MSAR-LNO，代码 family 暂定为 msar_lno。本消息是持续总控约束，不授权一次完成全文；本轮只执行我随后明确点名的一个 M 阶段，完成后停止等待审查。

一、修改前必须完成的仓库理解
1. 在修改任何生产代码前，必须系统浏览完整仓库。先读取适用的 AGENTS.md、README、docs、git 分支/commit/status/diff，再用 rg --files 和针对性 rg 遍历所有受版本控制的源码、配置、训练/评估脚本、模型 factory、checkpoint 工具、测试及八个任务目录。二进制、缓存、真实数据目录和构建产物可排除，但不能只浏览预想中的几个模型文件就开始修改。
2. 必须先形成“仓库结构→现有模型→任务入口→配置传递→loss→checkpoint→测试”的实际映射，列出已有类、函数、注册名、参数合同和可复用模块。聊天记录、旧计划和本文用于定义目标，当前代码用于确定接入位置；不能凭提示词虚构类名或路径。
3. 先识别用户已有未提交修改和此前 Codex 已完成的 CDLNO/KCDLNO 工作。不得 reset、checkout 覆盖、删除、改写历史，不自动 commit/push、建 PR 或启动真实训练。发现旧实现存在问题，要区分“原有缺陷”和“本次回归”，不能借新增模型擅自重构旧模型。
4. M0 只允许审计、文档和独立回归夹具；在 M0 被我接受前不得修改生产模型、训练、评估、数据或依赖。后续每个阶段都必须引用 M0 的实际映射；仓库发生变化时先重新核查受影响部分。

二、旧模型和旧实验必须保持
5. 原 Transolver、CDLNO full/no_sa/identity、原 CDPA 的既有模式、KCDLNO all/off、已有 matched LRSA 和仓库中其他现有模型都必须继续按原命令训练、评估和加载。不要改变旧 family、旧默认模型、旧 CLI 默认值、旧 state_dict key、pickle 类路径、模型 forward 返回类型或输出目录语义。
6. 新模型以新增 family、独立配置和独立 checkpoint 元数据接入。优先组合已经验证的通用原语；复用代码不等于复用同一参数对象。若修改共享模块存在改变旧数值行为的风险，应增加新类或兼容分支，并用修改前同权重夹具验证旧输出。
7. 不使用 strict=False、丢弃不匹配 key、静默补参数或重新随机初始化来伪装 checkpoint 兼容。历史 checkpoint 缺少新 family 字段时仍按仓库原有逻辑识别，不能猜成 msar_lno。
8. 数据读取、split/fold、采样、点序、图构造、normalizer、标签通道、PDE loss、optimizer、scheduler、训练 epoch、时间 rollout 和评价指标保持。新增辅助损失只能作用于 msar_lno，不能进入旧模型路径。

三、MSAR-LNO 的唯一主结构
9. 使用四级 latent encoder–decoder。输入逐点提升得到 E0∈R[B,N,d]；新模型不使用卷积，包括规则网格任务。输入提升和输出 head 使用现有任务 wrapper 的字段和通道约定，但在新 family 内使用逐点线性/MLP。
10. 默认 Light：M=[512,256,128,64]，d=96，heads=[4,4,8,8]，encoder_depths=[3,1,1,1]，decoder_depths=[3,1,1,1]。Full：M=[1024,512,256,128]，d=192，其余 heads/depths 相同。默认训练 profile 为 Light；不得按 N 自动裁剪 M 或偷偷换任务特定 M。M1>N 的合法扩张情况允许并记录。
11. 第 l 级 Down 使用 LRSA 风格 learned-query cross-attention：可学习 P_l∈R[M_l,d] 提供 Q，E_{l-1} 提供 K/V，输出 S_l；不把 P_l 作为额外 query residual 加回。随后 E_l=EncoderStack_l(S_l)。四级参数独立，不跨层共享。
12. 每个尺度内的 latent block 固定为三个独立 pre-norm 残差子层：
    X1 = X + FFN1(Norm1(X))
    X2 = X1 + SA(NormSA(X1))
    Xout = X2 + FFN2(Norm2(X2))
    两个 FFN 均为 d→2d→d、GELU、参数独立。SA 为该尺度标准多头 self-attention。复用仓库已验证的 LRSA 原语和 norm 细节，但不能复用会删除第二个 FFN 的旧消融开关。
13. 最深层按 MoNo 拓扑：D4=DecoderStack4(E4)。这里没有横向 skip 或额外历史融合，也不增加第三套 bottleneck processor。
14. 对 l=3,2,1，先做 query 对齐的逐级解码：
    U_l = UpCross_l(Q=E_l, K=D_{l+1}, V=D_{l+1})。
    UpCross 只返回读取分支，不在内部自动残差加 E_l。输出投影回 d。receiver 使用当前尺度 heads。由于输出行由 E_l 的 query 定址，U_l[i] 与 E_l[i] 属于同一目标 slot；如果实现采用与 E_l 无关的 decoder query，则不符合本模型。
15. 在 MoNo 原本同尺度加法的位置使用两来源、尺度保持的 AttnRes-style fusion。每个尺度一个 w_l∈R[d]，初始化为0。对每个样本和当前 token：
    sE = w_l^T RMSNorm(E_l[i])
    sU = w_l^T RMSNorm(U_l[i])
    [alphaE,alphaU] = softmax([sE,sU], source_dim)
    Z_l[i] = 2*(alphaE*E_l[i] + alphaU*U_l[i])
    D_l = DecoderStack_l(Z_l)
    仅 key 用同一个 RMSNorm 规则评分，value 为 raw E/U；没有1/sqrt(d)、QKV投影、source bias、多头路由或额外 residual。固定系数2和w=0使初始化时Z=E+U。Decoder block内部普通残差仍保留。
16. 最后使用 U0=UpCross0(Q=E0,K=D1,V=D1)，再接逐点输出 head。最终 N 点处没有 E0 AttnRes skip、没有 N 点 self-attention、没有任意输出位置 decoder。
17. 旧版讨论过的多尺度 HistoryCross、一次读取所有更细编码尺度、CDPA、KCDLNO核化全历史、persistent latent、IPOT Bridge、CoTAP/Sinkhorn、卷积、CDPA-Slice、跨真实时间 cache、递增 M、稀疏 Darcy和其他候选都不属于本模型，不能被带入。

四、唯一默认辅助损失及可消融要求
18. 只为四个 Encoder Down 提供可选 coverage-floor 辅助损失，不实现 specialization/MI、严格JS/L2全边际或 decoder/AttnRes 平衡损失。令每头Down权重A^[h]∈R[B,H,M,Nsrc]，先按head平均，再令p_j=(1/M)Σ_i Abar_ij。有效源token目标测度mu默认是mask内均匀分布；若现有任务已经可靠提供面积/体积积分权重才允许显式传入，不能修改数据格式凭空估计。
19. 每层原始coverage损失：
    Lcover_l = Σ_j relu(kappa*mu_j-p_j)^2/(mu_j+eps)，再对batch和有效层取mean。
    总训练损失：Ltotal=LPDE+lambda_cover*mean_l(Lcover_l)。推荐研究profile为kappa=0.2、lambda_cover=1e-2；两者是训练配置，不改变推理结构。
20. 必须有明确的coverage_mode∈{off,floor}。off时总损失严格为原LPDE，不要求返回/保存attention map，不计算coverage，不保留诊断计算图；floor时才获取Down权重并计算FP32辅助损失。coverage_weight=0也按off优化路径处理。训练日志分别记录PDE、raw coverage、加权项和total。
21. coverage配置必须可以从CLI/配置直接消融，并写入新checkpoint的训练复现元数据；它不属于state_dict形状字段，纯评估时不应因coverage on/off而拒绝加载相同架构。不得改变旧模型loss接口。可按当前仓库最佳方式为新family增加return_aux、专用loss adapter或结构化输出，但默认推理和旧调用仍返回原预测张量。
22. CoverageRatio、DiversityRatio和AttnRes来源权重/熵只作为可关闭诊断，不进入默认损失，不在正常forward中CPU同步或长期持有大attention张量。

五、工程、环境和验证边界
23. 目标环境CUDA12.8、PyTorch2.11.0、现有torch_geometric/pyg-lib cu128。保留当前能训练Car的环境；不盲装旧requirements，不升级/降级torch，不新增xformers、liger、自定义CUDA或Triton。标准PyTorch/SDPA路径优先。
24. 当前没有真实数据。不下载、不创建同名假数据、不运行会在import时读取数据的main/exp脚本。通过源码审计、真实模型的合成张量、现有可安全导入的loss/normalizer、真实PyG Data、严格checkpoint往返和静态参数流完成范围内工作。
25. 所有新增模块需要明确shape检查和错误信息；mask应在softmax前屏蔽。不同图/样本不能因展平而互相attention。不得为了让测试通过改变正式profile、真实接口或数据语义。
26. 每阶段结束更新独立 docs/MSAR_LNO_IMPLEMENTATION_STATUS.md；M0另写 docs/MSAR_LNO_REFERENCE_AUDIT.md；最终另写 docs/MSAR_LNO_IMPLEMENTATION_REPORT.md。保留旧文档。报告必须包含：完成/未完成项、实际文件和符号、公式映射、真实命令和结果、旧模型回归、未验证环境、用户应审查的3—5点。结束时明确“本M阶段结束，未执行下一阶段”。
```

## M0：完整浏览仓库，建立修改前依据

```text
现在只执行M0：完整审计当前已改KCDLNO仓库并建立修改前依据。允许新增审计文档、只读分析脚本和独立小型回归夹具；禁止修改生产模型、factory、训练/评估、数据、依赖和已有测试逻辑。

1. 严格执行总控中的“修改前完整浏览”：读取适用AGENTS.md和所有现有实现/阶段文档；记录git branch、commit、status、diff、未跟踪文件。用rg --files形成仓库总清单，系统阅读所有受版本控制的Python源码、配置、shell脚本、测试、模型与任务README；对重复模板可按哈希/差异归组，但必须说明归组依据。列出排除的二进制、缓存和数据目录。不要只阅读你准备修改的文件。
2. 输出仓库结构图和八任务入口表：Darcy、Elasticity、Airfoil、Pipe、NS、Plasticity、ShapeNet-Car、AirfRANS各自的数据对象/张量、模型调用、输入输出通道、训练loss、eval路径、CLI/配置、checkpoint格式和运行目录。标注哪些脚本import即读数据，后续不能直接导入。
3. 建立现有模型清单：Transolver、CDLNO full/no_sa/identity、CDPA模式、KCDLNO all/off、matched LRSA及其他实际存在模型。逐项记录family、factory、类路径、forward合同、配置来源、state_dict/整对象/列表保存方式和训练/评估选择逻辑。不要根据文档假定已经实现。
4. 建“新模型部件→现有文件/类/方法→可直接复用/需包装/不能复用→理由”的映射：input lift、learned-query Down、两个FFN和latent SA、norm、UpCross、point MLP/head、模型输出、aux loss、metadata、checkpoint、性能工具。特别核查旧Down能否按需返回attention权重而不改变旧调用，旧Up是否含隐藏query残差，旧block开关是否连带删除FFN2。
5. 核对已有模块初始化、参数共享、mask、batch、AMP和dtype路径。确认规则网格卷积位于哪里，并说明新family如何绕过它而不改变旧模型。确认工业任务是否只支持batch=1及可变N，不能假设通用[B,N,C]已经成立。
6. 为所有受本次可能触及的旧模型保存修改前确定性回归依据：配置、固定合成输入、同一份权重、eval/dropout关闭后的输出、state_dict key/shape、factory选择和checkpoint往返。优先复用已有K0/A0夹具；证据仍有效则引用，不复制大二进制。没有历史checkpoint时可用当前代码即时构造，但注明证据边界。
7. 检查Python、torch/CUDA、PyG版本和可用设备，不安装依赖。列出在当前环境可以运行及不能运行的检查。缺GPU/PyG不妨碍完成其余审计。
8. 基于实际代码给出M1—M9的最小文件变更范围、可能冲突和复用策略。如果当前实现与总控公式存在无法在后续阶段安全解决的实质冲突，完成其他审计后单列问题，不自行改架构。

增量写入docs/MSAR_LNO_REFERENCE_AUDIT.md和docs/MSAR_LNO_IMPLEMENTATION_STATUS.md，附回归夹具索引。完成后报告并停止，不执行M1。
```

## M1：新增独立配置、profile、family 与元数据基础

```text
M0已审查通过。现在只执行M1：按照M0确认的实际注册和配置体系，为msar_lno建立独立family、Light/Full配置、解析校验和checkpoint元数据基础。暂不实现Down/Up/AttnRes/core，不接八任务生产训练。

1. family固定为msar_lno，不重命名或复用旧CDLNO/KCDLNO family。按当前仓库规范建立稳定的新模块路径和最薄注册元数据；尚无真实模型类时，不在旧import路径放会破坏导入的占位构造器。
2. 结构profile精确定义：
   Light默认：num_latents=[512,256,128,64]，d=96，heads=[4,4,8,8]，encoder_depths=[3,1,1,1]，decoder_depths=[3,1,1,1]。
   Full：num_latents=[1024,512,256,128]，d=192，其余heads/depths相同。
   两者latent_ffn_ratio=2、activation=GELU、四级、无卷积。不要从旧L/F/P/M字段隐式推导这些值。
3. 验证每个列表长度为4、M为正整数、d能被所有heads整除、depth非负且本版必须与确认配置一致。不要按输入N自动截断M；Full第一层M>N时允许构造，并在运行日志中明确这是latent扩张。
4. 训练目标字段独立于结构字段：coverage_mode∈{off,floor}、coverage_weight>=0、coverage_kappa∈[0,1]、coverage_eps>0、diagnostics布尔值。研究默认profile使用floor/1e-2/0.2；另提供明确的no_regularizer覆盖示例off/0，不能用多个冲突布尔开关。
5. 解析优先级沿用当前项目约定，但必须实现“显式CLI > 所选新profile > 新family默认”。旧parser中的n_layers/n_hidden/slice_num默认不能静默覆盖Light。只在选择msar_lno时解析新字段，不能把新kwargs传给旧模型。
6. checkpoint元数据区分：结构字段、训练目标字段、运行字段。结构不一致严格拒绝；coverage设置写入复现信息，但纯eval不因coverage on/off拒绝同一结构state_dict。缺family的旧checkpoint继续走旧规则。
7. 新实验目录至少包含family、Light/Full和coverage on/off标识，避免覆盖旧模型或自身消融。保留用户显式save_name及现有覆盖防护。
8. 完成配置合法/非法、CLI覆盖、profile往返、旧CLI不变、metadata读写和family冲突的测试。M0旧模型同权重回归仍应通过；不要为测试构造假模型成功路径。

更新STATUS，报告resolved Light/Full和实际注册映射后停止。
```

## M2：实现并验证独立数学原语

```text
M1已审查通过。现在只执行M2：在M0确认的合适新模块位置实现MSAR-LNO的原子组件及独立数学reference。暂不组装完整模型，不接训练入口，不修改旧共享模块的数值行为。

需要实现或安全包装的语义组件：LearnedQueryDown、LatentFFN-SA-FFNBlock、QueryAlignedUpCross、PairwiseAttnResFusion、CoverageFloorLoss。类名可按仓库规范调整，但报告必须给出映射。

1. LearnedQueryDown：每尺度独立P∈R[M,d]，P提供Q，source提供K/V，标准多头cross-attention，输出[B,M,d]；无P residual。支持valid mask。正常路径可使用现有SDPA；只有coverage开启且训练请求aux时才返回/计算A[B,H,M,N]。off/weight0不物化A。
2. Latent block严格执行：X1=X+FFN1(N1(X))；X2=X1+SA(Nsa(X1))；Xout=X2+FFN2(N2(X2))。两个FFN d→2d→d/GELU，三个norm和所有参数独立；不调用旧no_sa/identity开关。
3. QueryAlignedUpCross：Q=current_encoder_feature，K/V=deeper_decoder_feature；返回纯attention分支[B,Mcurrent,d]，无query residual、无隐藏FFN。使用receiver heads并有独立Q/K/V/O。验证对current query排列等变。
4. PairwiseAttnRes：输入E,U同shape。每尺度只有w∈R[d]且严格零初始化；score分别为w^T RMSNorm(E/U)，softmax只沿长度2的source轴；value为raw E/U；输出固定2*(alphaE*E+alphaU*U)。无sqrt(d)、额外投影、source bias、多头、可学习gamma或外层再加残差。w=0时输出与E+U在数值容差内一致，输入直接Jacobian系数为1，w有可学习梯度。
5. CoverageFloorLoss：A先在head平均；p=(1/M)sum_latent A；mu为每个样本有效source上的归一化测度，默认mask内均匀。FP32计算relu(kappa*mu-p)^2/(mu+eps)，对有效source、batch、层按规格归一。mask位置既不进入softmax也不进入mu/loss。此模块无可训练参数。
6. 不实现specialization/MI、JS/L2严格均衡、Sinkhorn、CoTAP、decoder balance或AttnRes entropy loss。CoverageRatio/DiversityRatio作为no-grad可选诊断，不能被默认forward长期保存。
7. 独立reference不能调用待测forward。验证B=1/2、不同N/M/d/head、非方形数量、mask、float32/float64 reference、finite forward/backward、无意外参数共享和输入不被原地修改。
8. 验证Down注意力行和为1；coverage低于floor时loss为正、满足floor时为0；kappa=0或mode off为0。验证coverage梯度能到Down query和Q/K投影，但不要求所有随机参数首步非零。
9. 验证Up query排列时输出同排列；单独打乱U再与E融合应改变结果，说明PairwiseAttnRes依赖slot对应。不同batch不得混合。
10. 复用旧模块时跑M0同权重回归；若为了返回A修改共享attention，必须证明旧默认调用、state_dict key和数值输出保持，否则改为新包装类。

交付公式→类/方法→测试映射，更新STATUS后停止。
```

## M3：组装四级 MSAR-LNO core

```text
M2已审查通过。现在只执行M3：用M2原语组装四级MSAR-LNO core和输入/输出无关的模型级验证。不接八任务生产脚本或真实数据。

1. core输入合同按M0现有共享方式接收已经提升或待提升的点特征；任务wrapper差异留给后续。构造E0后，四级encoder依次执行Down_l和EncoderStack_l，保存E1—E4。每级参数独立。
2. 按profile精确实例化：Light/Full的M、d、heads、encoder/decoder depths；总encoder latent blocks=6、decoder latent blocks=6、latent SA=12、latent FFN=24。禁止层对象重复放入ModuleList造成意外权重共享。
3. deepest：D4=DecoderStack4(E4)。不得在E4/D4间加入AttnRes、Up、history、bridge或额外processor。
4. 对l=3,2,1严格执行：U_l=UpCross_l(Q=E_l,K/V=D_{l+1})；Z_l=PairwiseAttnRes_l(E_l,U_l)；D_l=DecoderStack_l(Z_l)。顺序不能改成先Decoder再融合，也不能让Up内部加E后又被AttnRes重复使用。
5. final：U0=UpCross0(Q=E0,K/V=D1)，prediction=pointwise_head(U0)。无E0横向skip、无N点SA、无卷积。final Up heads沿第一尺度设置，具体投影按M0公共约定实现。
6. coverage只收集四个Encoder Down的raw loss；core在明确请求aux时返回可审查结构，至少含prediction、per_level raw coverage和mean raw coverage。默认forward/纯eval仍返回prediction张量，不能改变旧模型。
7. coverage off或weight0时不请求A、不构造aux计算图；floor时才计算。prediction数学路径不能因为coverage mode改变；同权重off/floor的prediction在相应backend容差内一致。
8. diagnostics按需no-grad返回每级coverage/diversity和三个AttnRes的alpha均值/熵；正常训练关闭时不得保留大tensor、CPU同步或跨batch状态。
9. 用可缩小测试profile覆盖完整拓扑，另构造真实Light并检查参数shape；Full做构造、state_dict和小N forward（允许M1>N）。检查输入N变化、batch1/2、非方形N、forward/backward、所有阶段梯度、w0融合、保存/加载等价。
10. 建模块计数断言：Down4、Up4、encoder blocks6、decoder blocks6、PairwiseAttnRes3；HistoryCross/CDPA/KCDLNO reader/writer/Conv数量均为0。核对state_dict无旧机制闲置参数。
11. 不把MSAR-LNO塞进旧CDLNO/KCDLNO类的复杂模式分支。优先新core组合公共原语；保留稳定的新类导入路径供工业整对象checkpoint使用。

更新模型公式、shape trace、参数归属和STATUS后停止。
```

## M4：接入 factory、训练目标和 checkpoint 公共流程

```text
M3已审查通过。现在只执行M4：把msar_lno接入M0确认的公共或各子项目模型选择、参数转发、训练loss适配和checkpoint元数据基础。暂不声称八任务全部完成，不修改数据读取。

1. 按实际factory/registry增加独立msar_lno分支，只向它传新配置。旧模型选择和默认值保持。若仓库没有统一factory，使用最小薄适配，不新建一套虚假的全仓库框架。
2. 保持旧调用返回预测张量。为msar_lno选择与现有训练器最兼容且线程/重入安全的aux合同；优先显式return_aux或新family专用adapter，避免用model.last_loss这样的隐式跨forward可变状态。说明实际方案。
3. 新family训练时：先按原任务得到LPDE；coverage=floor且weight>0时取mean raw coverage并计算Ltotal=LPDE+weight*Lcoverage；off/weight0时Ltotal与LPDE为同一个数学目标且不计算A。日志分别输出PDE/raw/weighted/total。旧模型训练分支完全不认识该aux。
4. eval、预测导出和物理指标始终只使用prediction；coverage设置不改变预测shape或decode路径。诊断默认关闭。
5. 新checkpoint严格保存resolved结构profile、family、版本和训练目标元数据；eval先读checkpoint结构再构造，再核对用户显式结构覆盖。coverage on/off差异不阻止纯eval，但resume训练若仓库已有resume功能，应报告并按当前规则核对训练配置。
6. 保留state_dict、整模型、列表等各任务原保存协议，不能为了统一新family改旧格式。新增类路径稳定；旧pickle加载不应import新模块失败。
7. 添加选择/参数/负向测试：旧模型不收到新kwargs；新family缺结构字段按profile解析；错误family/shape拒绝；same-structure checkpoint严格往返；coverage off/floor均可重建；eval不改写sidecar。
8. 给出基于实际入口的最小构造示例，但不启动真实任务训练。M0旧回归全部重跑受影响部分。

更新STATUS和公共接入表后停止。
```

## M5：接入 Darcy、Elasticity、Airfoil、Pipe

```text
M4已审查通过。现在只执行M5：将msar_lno接入Darcy、Elasticity、Airfoil、Pipe现有训练、评估和checkpoint流程。保持四个任务数据及训练语义不变，不下载真实数据。

1. 逐任务按M0实际入口接入，不假定存在统一train.py。保留原输入字段、坐标/条件拼接、输出通道、normalizer、loss、optimizer/scheduler、epoch、batch、split和评价。不要复制四套模型数学实现。
2. 新模型即使在规则网格也使用逐点lift/head和latent attention，不调用旧ConvFFN或任何Conv2d。不要删除数据loader中旧模型仍需要的网格信息。
3. 默认选择msar_lno时使用Light；允许显式Full。不要沿用KCDLNO的任务特定d/M默认覆盖Light，也不自动按N截断1024。
4. coverage on/off必须能从各任务真实CLI/配置选择，并传到同一个loss adapter。off日志仍明确记录coverage关闭；floor日志分开记录PDE/coverage/total。evaluation不计算coverage。
5. 基于真实wrapper合同构造合成输入，完成每任务的forward、原loss可安全复用部分、backward、optimizer step、eval和checkpoint往返。使用非方形小网格检查不依赖sqrt(N)。不能把合成输入称为真实数据。
6. 检查full profile M1>N的Elasticity等边界只要内存允许就能运行；正式profile不应因测试缩小而改变。大配置无法在当前设备运行时，用小配置做数值、真实配置做静态/构造检查并如实区分。
7. 提供四任务Light coverage-on、Light off、Full覆盖的实际训练/评估命令模板；不能给不存在的参数。输出目录不互相覆盖。
8. 重跑这些任务的旧Transolver/CDLNO/KCDLNO选择、固定权重回归和checkpoint检查，确认新增分支没有改变旧默认。

更新任务覆盖表和STATUS后停止。
```

## M6：接入 Navier–Stokes 与 Plasticity

```text
M5已审查通过。现在只执行M6：将msar_lno接入Navier–Stokes和Plasticity，严格保留现有时间建模、训练和评估协议。

1. 从M0接口表出发，保留NS真实输入帧数、单次输出帧数、teacher forcing/预测回填、rollout长度、normalizer、loss和指标。每次模型forward重新构建四级latent，不跨真实时间缓存E/D/attention或coverage状态。
2. 保留Plasticity的空间点组织、时间条件输入、输出通道、逐时间点/序列训练节奏和optimizer/scheduler调用。不得把时间维并入空间N或改变样本循环。
3. coverage按“实际发生的每次模型forward”产生raw值，并按照当前任务已有PDE loss聚合尺度一致地平均；不能因rollout步数增加而无意把coverage重复放大。报告实际聚合公式及代码位置。
4. coverage off/floor、Light/Full参数流、checkpoint和eval选择与M4一致。评估默认不计算aux；需要诊断时使用明确开关且不改变预测。
5. 用真实wrapper合同的合成时间数据完成forward、原loss可复用部分、backward/optimizer、短rollout和checkpoint往返。检查每个时间步状态清空、无未来真值泄漏、输出shape和回填窗口不变。
6. 保留NS长时rollout已有能力和命令，但不新增论文未有的rollout实验或启动真实训练。提供基于实际脚本的Light on/off和Full命令模板。
7. 重跑NS/Plasticity旧模型受影响回归，确认新family没有改变时间循环的旧分支。

更新STATUS后停止。
```

## M7：接入 ShapeNet-Car 与 AirfRANS

```text
M6已审查通过。现在只执行M7：将msar_lno接入ShapeNet-Car和AirfRANS现有训练、评估和checkpoint流程，保持PyG、图字段、可变N和工业任务后处理不变。

1. 严格使用M0记录的真实forward合同，例如Car的tuple/Data组合和AirfRANS的Data对象；不要凭旧提示词假设字段。保留坐标、几何条件、reference feature、surf/boundary mask、采样、标签通道、原loss和评价散射。
2. 新模型不使用图边和卷积，但不能删除、改写或绕开loader中旧模型/评价仍需要的图构造。只在新wrapper中提取现有点特征并调用同一MSAR-LNO core。
3. 保留当前单图batch=1和可变N合同。若仓库已经支持独立多图batch，必须保证每张图单独attention；不得把不同几何拼成同一物理域。没有可靠隔离时清晰拒绝batch>1。
4. 默认Light，Full显式可选。coverage的mu在没有现成面积/体积权重时仅对该图有效点均匀；不能从标签推导重要性，也不能修改数据文件补权重。
5. 保留工业任务实际state_dict、整模型对象或模型列表checkpoint格式和稳定类路径。train/eval先读元数据再构造；错误family/结构严格拒绝。旧可信pickle边界不扩大。
6. 在可用PyG环境中用真实torch_geometric.data.Data构造合成单图，完成forward、现有可安全复用的mask/loss、backward/optimizer、可变N、eval和save/load。缺PyG则完成其余工作并明确真实PyG项未执行，不能创建假torch_geometric包。
7. 检查coverage on/off不改变预测节点顺序，输出仍与原采样idx/完整场scatter一致。诊断不得保存整图attention到日志。
8. 提供两个任务Light on/off和Full的实际命令模板，保留原运行目录和依赖。重跑旧Transolver/CDLNO/KCDLNO工业入口回归。

更新八任务覆盖表和STATUS后停止。
```

## M8：无真实数据的综合验收与消融闭环

```text
M7已审查通过。现在只执行M8：在不下载真实数据、不启动真实训练的前提下，完成MSAR-LNO八任务、Light/Full、coverage on/off、checkpoint及旧模型的综合可执行验收。不增加新架构。

1. 建八任务覆盖矩阵，分别标记：静态参数流、真实模型合成forward、原loss连接、backward/optimizer、eval、checkpoint、真实PyG、GPU、真实数据。未运行必须标未执行，不能因一个小张量测试把整行写为支持完成。
2. 每个任务至少用Light完成coverage off和floor的小型训练步及eval；Full至少完成真实参数构造、严格state_dict往返和缩小N的forward/backward。复用M5—M7有效证据，不做无意义全笛卡尔积。
3. 用同一结构和同一权重核对：coverage off与floor的prediction一致于backend容差；off的total=PDE且不物化A；floor的raw loss有限，乘weight后正确进入total，梯度能到四级Down但不进入decoder/AttnRes额外平衡项。
4. 验证kappa=0、weight=0、mode=off、无mask、部分mask、M1>N、不同N、batch边界和非法配置。检查aux/diagnostics不会跨forward残留或造成第二次backward图引用。
5. 检查三个AttnRes：w0时严格为对应E+U；softmax轴为source=2；每token权重不同；打乱U token会改变融合；梯度到w、E、U；没有额外gamma、projection或历史来源。
6. 模块计数和执行hook应确认Down4/Up4/PairFusion3/latent blocks12；无HistoryCross/CDPA/kernel history/Conv/N点SA。检查各层参数对象独立。
7. 从各实际子项目工作目录启动新Python进程验证导入、factory、训练/评估模型选择和需要的工业对象加载。不能直接运行会顶层读真实数据的脚本；先静态检查并使用安全入口。
8. 使用M0相同权重和输入重新验证旧Transolver、CDLNO各模式、KCDLNO各模式及已有matched LRSA。旧输出、state_dict key、CLI默认、checkpoint导入路径和loss必须保持约定容差。
9. 汇总所有真实执行命令、版本、耗时和限制，给出远端CUDA12.8/Torch2.11的必要补充检查命令，但不声称尚未运行的GPU/数据项通过。

更新覆盖矩阵和STATUS后停止。
```

## M9：成本、有限性能、独立反查与最终交付

```text
M8已审查通过。现在只执行M9：核对MSAR-LNO完整参数与计算，进行环境允许的有限性能检查，从总控规格独立反查代码并完成交付。不得改变模型来追求更好数字，不启动真实数据训练。

1. 从实际模型统计Light/Full的参数量，分解stem/head、4 Down、12 latent blocks、4 Up、3 PairwiseAttnRes。检查两个FFN、所有norm和attention投影均计入；coverage无训练参数，w仅3d。不要引用MoNo论文参数量作为本模型参数量。
2. MAC/FLOP应包含第一层O(NM1d)、每层Q/K/V/O、latent SA O(M_l^2d)、两个FFN、Up和pointwise head。coverage训练开销单列，推理off不计；不能只报attention矩阵乘法或用参数减少推断延迟。
3. 比较对象至少包括：MSAR-LNO Light coverage off、同模型floor训练路径、Full（能运行时）、原Transolver、现有matched LRSA、KCDLNO。结构不等宽/不等参数时明确，不称公平速度或精度对照。
4. 有可用GPU时复用现有计时工具，固定device/batch/N/dtype/AMP/compile/backend，warmup并CUDA同步，报告forward median/p90、完整训练step、峰值显存。无GPU完成参数/MAC和CPU功能，不用CPU推断4090延迟。
5. 特别测coverage floor显式A相对off的训练显存/时间；正常推理两者应走同一prediction路径。大N无法测试时给出内存上界和远端命令，不修改kappa/M来隐藏成本。
6. 从需求逐条建立“规格→文件/符号→独立证据→状态”矩阵，重点反查：Light/Full、无卷积、Down query、FFN-SA-FFN、deepest拓扑、Up Q=E、纯Up分支、两源scale-preserving AttnRes、final无E0 skip、coverage只在Down且可关闭、旧模型不变。
7. 审查测试是否与实现共用同一错误公式；关键Down/Up/AttnRes/coverage使用独立reference或手算小例。发现本次范围内缺陷可修复并重跑受影响检查；需要改变已确认架构或数据协议的事项只报告，不自行改。
8. 交付docs/MSAR_LNO_IMPLEMENTATION_REPORT.md、README新增独立章节、八任务实际训练/评估命令、Light/Full与coverage on/off示例、checkpoint规则、无数据验收命令、有限性能命令、已验证和未验证边界。保留旧README和旧阶段文档。

最终明确回答：
A. 哪些架构条款已经实际实现；
B. 八任务哪些静态/合成/原loss/PyG/checkpoint/GPU检查真实通过；
C. coverage off是否真正移除辅助目标和权重物化；
D. 旧模型保留的证据；
E. 真实数据读取完整性、收敛、精度、实际epoch时长和SOTA中哪些仍未验证。

完成后停止，不自动commit/push或执行训练。
```

## 阶段间通用提示词

### 接受并进入下一阶段

```text
我已审查并接受M[N]。现在只执行M[N+1]，完整范围如下：
[粘贴下一阶段正文]

此前MSAR-LNO总控、旧模型保留、数据/训练冻结和无真实数据约束持续有效。完成本阶段后停止，不自动进入后续阶段。
```

### 阶段返修

```text
M[N]暂不通过，不进入下一阶段。请只修复以下问题：
[粘贴具体问题]

先定位原因、实际文件/符号和违反的规格，再做最小修复。只重跑受影响测试及必要旧模型回归，不借返修加入新机制、重构旧模型或改变数据/训练协议。更新阶段报告后停止。
```

### 独立只读审查

```text
请只读审查M[N]的实际diff、阶段报告、MSAR-LNO规格和测试证据，不修改文件、不进入下一阶段。

重点查找：只看局部文件而遗漏真实入口；公式与代码不符；Up不是Q=E；Up隐藏query residual；AttnRes source轴/固定2/w0错误；模块重复残差；coverage在off时仍物化A；辅助loss进入旧模型；卷积或旧History/CDPA被带回；参数意外共享；checkpoint用strict=False；合成检查被写成真实数据验证。

按严重程度给出文件/符号、证据、影响和最小修复建议。区分确定缺陷、待验证风险和已确认设计取舍。
```

### 缺数据/GPU时继续当前阶段

```text
当前仍没有真实数据，也不要求下载。请继续完成M[N]范围内所有能通过源码、真实模型合成输入、现有可用loss/normalizer、PyG对象和checkpoint完成的实现与验证。

不要只交付计划或空接口，不创建同名假数据。缺GPU/PyG的项目单独标未执行并给出基于仓库实际脚本的远端命令。完成当前阶段后停止。
```

### 会话恢复

```text
继续MSAR-LNO分阶段实施。先读取适用AGENTS.md、docs/MSAR_LNO_REFERENCE_AUDIT.md、docs/MSAR_LNO_IMPLEMENTATION_STATUS.md、当前git status/diff和已接受阶段证据，确认当前仓库状态。

原Transolver、CDLNO、其消融、KCDLNO和matched LRSA都是必须保留的既有模型，不重新实施旧计划。MSAR-LNO已接受到M[N]；本轮只授权：[当前返修或下一阶段完整正文]。复用仍有效证据，报告与代码不一致时先核查。完成后停止。
```

## 用户审查时优先检查

1. M0 是否真的遍历了整个仓库和所有任务入口，而不是先写代码后补审计文档。
2. 新模型是否是独立 family，旧 CDLNO/KCDLNO 类、参数名、默认命令和 checkpoint 是否保持。
3. UpCross 是否严格由对应 (E_l) 提供 query，并且没有在内部先加 (E_l)。
4. AttnRes 是否只有两个同尺度来源，是否为 `2 * softmax-weighted sum`，`w=0` 是否精确得到 `E+U`。
5. 是否彻底删除上一版多尺度 HistoryCross；是否没有 CDPA、KCDLNO history、CoTAP 或卷积混入。
6. coverage off 是否真正只优化原 PDE loss，并且不物化 attention weights；不是“权重设0但仍完整计算”。
7. coverage floor 是否只约束 Encoder Down，是否使用 mask 内均匀测度而非标签重要性。
8. Light/Full 的四级 token 数、宽度、heads 和 encoder/decoder depths 是否完全匹配规格。
9. 无数据验证是否使用真实模型和真实wrapper合同，是否如实区分合成、PyG、GPU和真实数据结果。
10. 性能报告是否计算了 (NM_1)、各级 (M_l^2)、双FFN和coverage训练开销，是否避免宣称未实测的SOTA或延迟优势。

