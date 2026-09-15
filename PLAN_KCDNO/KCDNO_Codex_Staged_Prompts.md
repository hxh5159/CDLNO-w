# KCDNO：基于现有 CDLNO 仓库的分阶段 Codex 实施提示词

版本：v1，2026-09-15。

这是交给用户另一 Codex 会话执行的提示词。它基于已经确认的 `KCDNO_Model_Specification_v1.md`，为现有 CDLNO 仓库增加独立 KCDNO 模型；不替换原 CDLNO 实现计划，也不覆盖已完成的两项前段消融。本文件生成时没有修改用户远端仓库，没有验证远端实际代码状态。

## 使用方式

1. 将本文件和 `KCDNO_Model_Specification_v1.md` 一起提供给正在修改项目的 Codex。此前 CDLNO v1_2、原分阶段提示词和 front 消融提示词如果已在该会话或仓库中，无需重新实施，作为保留旧行为的依据。
2. 先发送下面的“总控提示词”，再发送“K0”。全文是工作参考，只有你本轮点名的阶段可以执行。
3. 本次使用 **K0—K10** 编号，与原 CDLNO 的阶段0—10、消融 A0—A4 分开，避免误执行旧阶段。
4. 每次只发送一个阶段正文。检查 Codex 的 diff、公式映射和实际验证结果；接受后再发送下一阶段。发现问题使用文末返修模板。
5. 全部实施与无数据验收均不要求下载真实数据。真实数据训练、准确率和真实 epoch 时间验证留待另行授权。
6. 新架构默认从头训练，不把旧 CDLNO checkpoint 自动转换成 KCDNO。旧模型及旧权重继续按原入口使用。

| 阶段 | 本次范围 | 你主要审查什么 |
|---|---|---|
| K0 | 当前仓库审计、旧模型回归依据 | 是否真正理解已有实现、没有假定类名或功能 |
| K1 | 新配置与架构元数据基础 | 新旧参数隔离、默认值、checkpoint重建规则 |
| K2 | 独立核 writer／reader | 缓存公式、两类归一化、来源融合、梯度 |
| K3 | 新 block 与完整全历史 core | 插入位置、两 FFN、首末层、28份历史读取 |
| K4 | Darcy／Elasticity／Airfoil／Pipe | 静态任务接口、卷积、实际训练/评估选择 |
| K5 | NS／Plasticity | 原时间协议、缓存不跨真实时间 |
| K6 | ShapeNet-Car | tuple、PyG、surf mask、checkpoint |
| K7 | AirfRANS | graph接口、抽样与重建评价、模型列表加载 |
| K8 | 可训练的 matched LRSA 对照 | 公共部件一致、区别于旧 CDLNO 消融 |
| K9 | 完整成本与有限性能检查 | 参数、MAC、真实计时口径、不夸大加速 |
| K10 | 独立综合审查和交付 | 八任务闭环、旧版本回归、真实限制 |

## 总控提示词：首先发送

```text
现在基于这个已经实现 CDLNO 及前段 full/no_sa/identity 消融的现有仓库，增加一个独立模型 KCDNO（Kernelized Cross-Depth Neural Operator）。我已经确认 KCDNO_Model_Specification_v1.md 中的架构。

请继续采用分阶段工作方式。本消息是持续总控约束，不单独授权实施全文；本轮只执行我随后明确点名的 K 阶段。

一、授权和依据
1. 每次只实施我点名的一个阶段。在阶段范围内自主完成必要修改与验证，不逐文件询问；完成后停止，等我审查。持有全部提示词不等于获得后续阶段授权。
2. 优先级：我当前及以后明确确认的决定 > KCDNO_Model_Specification_v1.md > 本执行指南。旧 CDLNO v1_2、原阶段报告和 front 消融计划继续约束旧模型的行为；其中“前段不加新机制”等限制不阻止本次独立 KCDNO，但仍约束旧 CDLNO。
3. 不把此前讨论过的校正层、CDPA-Slice、persistent AttnRes、后段 MLP 等候选带回本模型。参考论文和数学附件不能授权新增结构或损失。
4. 读取适用 AGENTS.md，先核对实际分支、commit、git status、未提交修改和已有阶段报告。保留我的工作，不 reset、不覆盖、不自动 commit/push、建PR或启动真实训练。正常实施不改 AGENTS.md 来规避规则。
5. 当前实际仓库是集成对象。不要假定某个类、测试或模型已经存在，也不要从原 Transolver 重新搭建整套 CDLNO。发现已接受计划与当前代码存在实质差异，完成本阶段不受影响部分，再列出具体冲突及建议供我裁定，不自行修复成另一种研究架构。

二、新模型的唯一主结构
6. 新模型注册家族为 kcdno；按实际项目注册方式建立映射，checkpoint 内统一记录 family=kcdno。不重命名 CDLNO 项目，不修改旧 family、模型名、默认命令或已有类导入路径。
7. 默认 L=8 个 LRSA 风格完整点域 block。每层顺序是：point norm → Down → latent FFN1残差 → 全历史核读取与来源融合 → latent FFN2残差 → Up → 点残差 → point FFN/ConvFFN残差。
8. 删除原 latent SA 整个残差子层及其专属 norm/Q/K/V/O，不保留废弃计算。两个 latent FFN保留。Down/Up仍是原已验证的标准多头Cross，投影与norm保留。不能将 SA 模块换成Identity后继续外层加法，也不能调用整个旧no_sa block后再在外面补history。
9. 历史 T_l取 FFN2残差后、Up latent norm之前。每层读取所有 s<l 的历史。没有Bridge、persistent后段、IPOT processor或额外final feature decoder；最后一个block写回点域并完成point模块后，直接接任务输出norm/head。
10. 首版各层M一致、point/latent/attention宽度均为d，r为单组核特征的总维度，与Down/Up的heads独立。层数L是新模型自身的完整block数，不复用CDLNO的F/P派生规则。

三、新历史机制
11. 源层s拥有 Wk_s：K_s=phi(RMSNorm_k_s(T_s) Wk_s)，cache_s=(K_s^T T_s, K_s^T 1)。value是raw T_s，无额外Wv/Wo。每份摘要在一次forward中写一次。
12. 接收层l拥有 Wq_l：Q_l=phi(RMSNorm_q_l(U_l) Wq_l)。对每个历史分别读取 R_ls=Q_l cache_s.matrix/(Q_l cache_s.mass+eps)。不能使用receiver-owned K重算历史；不能预合并所有源摘要；不做M×M token softmax或current kernel self-read。
13. 当前候选R_l0=U_l。用接收层共享的w_l·RMSNorm_depth_l(R_ls)评分，沿来源维度统一softmax，raw候选加权得到C_l；最后Uhat_l=U_l+gamma_l*(C_l-U_l)。不重复加残差。
14. w_l=0初始化；gamma_l为每个接收层一个无约束可学习标量，初始化0.1，不改成sigmoid；norm scale=1。均匀的是初始深度权重，不代表核token权重均匀。这个gate只属于新KCDNO，不改变旧CDPA。
15. phi(t)=ELU(t)+1，FP32后clamp_min(1e-6)，denominator eps=1e-6；摘要使用sum，不改mean。Wq/Wk为d→r、bias=False、Xavier uniform gain1。其余公共模块初始化按已定front规范，不被新模块递归初始化覆盖。
16. 生产路径phi、缓存矩阵累加、分母/除法、来源评分/softmax/融合用禁用autocast的FP32计算，返回前转回U.dtype；独立数学reference保留float64，不能把强制float函数直接用于double gradcheck。
17. 第一层无历史reader/scorer/gate；最后一层无key writer；L=1无历史参数。history_mode=all为主版，off为新模型专属对照，off不注册闲置历史参数。默认L8应有7次写入、7次Q生成、28份历史读取。
18. 缓存是单次forward内的局部状态，不detach、不no_grad、不跨batch/样本/NS真实时间、不原地修改。分层历史使用tuple快照；不要在activation checkpoint中捕获还会增长的list。

四、保留旧模型和数据训练协议
19. 原Transolver、CDLNO full/no_sa/identity、原CDPA off/entry/every_block及已确认的执行选项全部保持。新增独立配置和checkpoint目录，旧state_dict键和pickle导入路径不变，不用strict=False隐藏错误。
20. 复用已经符合条件的Down/Up、两个FFN、norm、ConvFFN、输入提升/输出头和任务wrapper逻辑；复用代码不等于共享参数对象。优先新类组合原语，避免改变旧block的forward返回类型或旧T_i定义。
21. 八任务数据读取、字段、split/fold、采样、点序、normalizer、loss、optimizer/scheduler、训练/测试循环及评价处理保留。只改新模型/对照模型及必要选择、参数转发、配置和加载分支，不复制八套训练器，不删除模型未用的图处理来制造加速。
22. 工业任务保持当前已接受的单图/原batch=1、可变N合同；若当前已支持独立多图，沿用其隔离规则。不能把多张图拼成一个物理域，也不因本次任务扩展新多图训练方式。
23. 当前没有真实数据。不下载、不创建冒充真实数据的文件、不启动真实训练；不import会顶层读数据的exp/main。通过源码、真实模型合成输入、原损失可安全复用部分、真实PyG对象和checkpoint检查推进。缺真实数据不阻止可执行实现工作。
24. 目标远端环境CUDA12.8、Torch2.11；Car已有可训练PyG环境。先检查并保留，不盲装旧requirements，不主动升级降级torch/PyG，不引入xformers/liger或自定义CUDA/Triton。无GPU/PyG则如实列未验证，不能伪造环境或成功结果。
25. 不实现稀疏Darcy、任意输出位置decoder、递增M、邻接/窗口历史、current kernel SA、latent卷积、新物理损失或跨真实时间cache。新模型只能宣称移除了latent SA，不能宣称全模型没有attention。

五、逐阶段报告
每个阶段结束必须更新独立的 docs/KCDNO_IMPLEMENTATION_STATUS.md，保留旧CDLNO/A阶段记录。报告：
A. 本阶段完成范围和未完成项；
B. 实际新增/修改文件、符号、diff摘要及原因；
C. 规格条款、公式/张量轴/参数归属与代码对应；
D. 实际执行命令、版本、通过/失败/未运行及证据；
E. 旧模型回归、冻结的数据/训练区段是否改变；
F. 我应重点审查的3–5点和剩余依赖。
缺环境可完成本阶段其他工作并交付限制，不把“未运行”写成“通过”。完成后明确：“本K阶段结束，未执行下一阶段。”
```

## K0：核对现有仓库并保存改动前依据

```text
现在只执行K0：审计现有已改CDLNO仓库。允许新增审计文档、独立回归夹具和临时核查脚本，不修改生产模型、训练/评估、数据、依赖或现有测试逻辑。

1. 阅读当前可用的KCDNO规格、已接受的CDLNO计划/阶段状态和front消融记录；列出实际读到的材料。新规格若缺失，先完成其余只读审计，再说明缺少的关键文件；不要凭印象补齐。旧资料缺失但当前代码和新规格足够的部分继续核查，不要求重新实施旧计划。
2. 记录git分支、commit、status和用户已有diff/未跟踪文件。旧实现有问题与本次待改内容分开，不清理、不reset。确认当前三个子项目的模型入口及共享包实际安装方式。
3. 建立“规格部件→实际文件/类/方法→可否直接复用→差异”的表：输入提升、Down、FFN1、SA及norm、FFN2、T_i返回、Up latent norm、PointFFN/ConvFFN、输出头、CDPA、模型factory、metadata/checkpoint工具。不要仅凭no_sa名称认定结构匹配。
4. 确认旧no_sa内部两FFN的边界。新history必须能放在两者之间；给出新类组合已有原语的最小方案。核对LRSA参考disable_interleaved_blocks是否同时移除FFN2；不把官方开关直接等同新方案。
5. 建八任务接口表：数据字段→模型实参→输入提升/坐标时间处理→输出通道→原loss→train/eval选择→checkpoint类型→运行目录；标注会顶层读数据的脚本。工业任务特别记录单图规则、整模型/列表pickle与稳定类路径。
6. 保存“修改前”的旧Transolver和CDLNO full/no_sa/identity回归依据：实际配置、state_dict键/形状、固定合成输入及eval输出，并留存对应同一份权重。按已有wrapper覆盖点/卷积/时间/工业接口；缺PyG等实际无法运行的项明确标记。没有历史checkpoint时可用当前模型临时构造参考权重，但注明不能据此证明所有历史checkpoint兼容。夹具单独保存，不覆盖已有实验结果，不自动提交大二进制。
7. 保存旧CLI默认/模型注册、已确认训练配置和冻结区段的可比较依据。后续回归必须加载同一份权重，不拿重新随机初始化输出互比。记录数值容差和实际dtype/backend。
8. 只读检查Python、torch/CUDA、PyG等实际环境。不安装依赖，不运行真实训练，不为了取接口直接import顶层数据脚本；--help也先审查是否安全。
9. 确认后续K1—K10的最小变更范围。若旧CDLNO/A阶段尚有缺项，记录其影响，不借此自动完成未授权旧阶段。

写入 docs/KCDNO_REFERENCE_AUDIT.md、docs/KCDNO_IMPLEMENTATION_STATUS.md 和回归夹具索引。文件已存在则增量更新，保留历史。报告新/旧模型边界和实际可复用项后停止。
```

## K1：新家族配置和 checkpoint 元数据基础

```text
K0已审查通过。现在只执行K1：建立独立KCDNO配置、模型家族映射协议和元数据处理基础。复用现有共享包/安装结构，不重建项目；不实现核读取、完整模型或八任务训练接入。

1. 按K0实际目录新增独立配置类/模块，不强制另建顶层包。family=kcdno。旧CDLNO配置的F/P/front_latent_mode/CDPA mode不改变，也不强行继承为KCDNO字段。
2. 新结构字段至少包含：L、d、h、M、kernel_rank、history_mode(all/off)、两FFN hidden/activation、point模块类型、norm/QK norm、phi/clamp/eps、gate参数化。首版固定等宽，各层固定M；L>=1、d/h/M/r正整数、d可被h整除，非法值明确报错，不静默取整。
3. 默认L8、r16、history=all、latent/point FFN hidden2d plain GELU、norm eps1e-6、kernel phi ELU+1/clamp1e-6/deneps1e-6、无约束标量gamma初始化0.1、w0。新Q/K projection biasFalse Xavier gain1，其他公共参数初始化按规格。字段少而明确，未授权方案不加开关。
4. 建新任务profile（可先只提供配置数据）：
   Darcy       d128 h8 M64 ConvFFN
   Elasticity  d128 h8 M64 PointFFN
   Airfoil     d128 h4 M64 ConvFFN
   Pipe        d128 h4 M32 ConvFFN
   NS          d256 h8 M64 ConvFFN
   Plasticity  d128 h8 M64 ConvFFN
   Car         d256 h8 M64 PointFFN
   AirfRANS    d256 h8 M64 PointFFN
   各任务L8、r16、hidden2d。主profile=kcdno_v1。
   另记录transolver_shape_match：Airfoil h8；Pipe h8/M64；NS/Car/AirfRANS M32，其余同主profile。只作用于新家族，不覆盖旧defaults。
5. 固定解析优先级：训练用“显式CLI > 所选profile > 新家族默认”；原parser中未显式传入的默认n_layers/n_hidden/slice_num不得覆盖新profile。选一种可审查的显式参数识别方式，保持旧CLI行为；不要制造两个冲突的L/M来源。新薄脚本以profile提供默认结构，不把全部主profile值硬写为显式CLI，导致用户切换profile后仍被这些参数覆盖。KCDNO不支持的旧front/rear/CDPA选项若被用户显式传入，应清楚说明不适用，不能静默改变新结构。
6. eval先读checkpoint的已有架构信息重建，再核对用户显式架构参数；不能用当前CLI默认覆盖，也不能先写sidecar再校验。新checkpoint必须明确family，缺family的旧checkpoint继续走旧规则，不能猜成kcdno。
7. 区分结构/行为字段、复现字段和运行字段。结构/行为不一致需拒绝；初始化版本、gamma_init记录为训练起点，加载时不重新初始化覆盖已学习值。device/batch/backend等运行信息不作为参数形状冲突，但要记录可能数值差异。profile名本身不能代替resolved配置。
8. 保留各任务原state_dict、整模型、列表保存协议；新sidecar和输出目录独立。不能新增原项目没有的resume功能；若已有resume，接入时校验架构并沿用原优化器状态协议。不提供旧CDLNO到KCDNO的自动转换或strict=False宽松加载。
9. 实施必要配置测试：合法/非法、profile覆盖、显式CLI优先、配置往返、错误family/r/history/activation拒绝、eval不改写sidecar、旧路径缺新字段仍走旧规则。新类尚未实现时不要在旧import路径放会报错的占位构造器。

补齐可用环境下K0未保存且本阶段修改前必须留存的旧基准；缺环境如实记录。不要重装torch/PyG。更新STATUS和K1报告后停止。
```

## K2：独立实现核 writer／reader

```text
K1已审查通过。现在只执行K2：实现独立KernelHistoryWriter和KernelHistoryReader（类名可按项目规范映射），以及独立数学reference。暂不组装完整KCDNO，不接训练入口，不修改旧CDPA。

精确公式：
源s：
  K_s = phi(RMSNorm_k_s(T_s) @ Wk_s)        # [B,M,r]
  memory_s = K_s.transpose(-1,-2) @ T_s    # [B,r,d]
  mass_s = K_s.sum(dim=token)              # [B,r]
接收l：
  Q_l = phi(RMSNorm_q_l(U_l) @ Wq_l)        # [B,M,r]
  denominator = Q_l @ mass_s.unsqueeze(-1) # [B,M,1]
  R_ls = (Q_l @ memory_s) / (denominator + eps)
  candidates = [U_l, R_l1, ..., R_lS]
  score_s = w_l dot RMSNorm_depth_l(candidate_s)
  alpha = softmax(score, dim=source)
  C_l = sum_source(alpha_s * raw_candidate_s)
  Uhat_l = U_l + gamma_l * (C_l - U_l)

要求：
1. Writer属于源层；Reader属于接收层。Reader接收摘要，不接收T后重新投影K；Q在同一次融合只生成一次。新核是单组总rank r，与Down/Up头数无关。无Wv/Wo、无额外sqrt(d)/sqrt(r)缩放、无token softmax、无M×M矩阵生产路径。
2. value为raw T；score norm只用于评分。当前U不经过kernel self-read。空history直接返回U，不额外norm/投影/加残差。来源权重按每个样本、每个当前token计算，形状[B,M,S+1]。
3. phi=ELU+1并clamp_min1e-6，deneps1e-6，使用sum摘要。新Wq/Wk biasFalse、Xavier gain1；RMSNorm scale1；w0；gamma为无约束标量0.1。不要把当前来源加正bias或改变旧CDPA的初始化。
4. 生产FP32按规格真正关闭关键区域autocast，保留梯度；返回U.dtype。独立reference保留double，使用相同phi/clamp/eps。reference不得调用待测reader.forward来证明自己正确；double梯度核对避开clamp/ELU分段边界。
5. 默认批量stack各来源摘要计算，保留来源轴；同时提供逐来源reference验证数学等价。若memory为[B,S,r,d]、mass为[B,S,r]，可用einsum('bmr,bsrd->bsmd',Q,memory)和einsum('bmr,bsr->bsm',Q,mass)[...,None]明确batch/source轴。不能把[B,M,r]直接与[B,r]做普通@并期望逐batch向量乘法；不能预合并摘要、按chunk各做来源softmax再平均。首版不实现复杂流式softmax或窗口读取。

必要验证：
a. float64缓存形式与显式正值核矩阵QK^T形式的前向、输入及Q/K投影梯度一致；不是与softmax attention比较。
b. 生产FP32前向/梯度按相应容差对照reference；有可用GPU才测实际AMP，缺GPU列未执行。
c. B1/B2、S0/1/3、不同M/r；来源次序不变性（对应记录一起换序）、历史token同步置换不变、当前token置换等变、batch互不混合。
d. w0时alpha均匀；gamma0.1时当前系数0.9+0.1/(S+1)，每历史0.1/(S+1)。核token权重不必均匀。
e. 隔离reader的loss验证最早T及Wk有效有限梯度，不用整网point旁路掩盖detach。w0时depth norm首步零梯度正常，不能要求所有参数每步非零。
f. 输入/cache不被原地修改；参数未跨writer/reader意外共享。测试极端但有限幅值的NaN/Inf与分母。

本模型M个latent都有效，不主动新增latent padding方案。若现有接口确有mask需求，先说明合同；正值核mask应屏蔽phi后的贡献，不能把输入置零后过phi，因为phi(0)=1。此处不授权新增图batch方式。

交付清晰的公式→代码→测试表，核对FP32生产与double oracle边界，更新STATUS后停止。
```

## K3：组装新 block 与全历史 core

```text
K2已审查通过。现在只执行K3：复用已验证公共原语和K2模块，组装KCDNOBlock与L层KCDNO core。不接八任务生产训练入口，不改变旧block/CDPA定义。

每层唯一计算：
  H = PointNorm(X)
  S = Down(H)
  U = S + FFN1(Norm1(S))
  Uhat = Reader(U, tuple(history)) if history else U
  T = Uhat + FFN2(Norm2(Uhat))
  V = X + Up(H, T)                   # Up latent norm恰好一次
  Xnext = V + PointModule(NormOut(V))
  若本层非末层且history=all：缓存Writer(T)供后层读取

1. Down learned query直接作为投影后的Q参数；不额外Wq或query residual。Down/Up Q/K/V无bias、O有bias，per-head QK RMSNorm和标准d_h^-1/2缩放保留。新kernel读取不套此缩放。
2. 两latent FFN都是Linear(d,2d,bias=True)→GELU→Linear(2d,d,bias=True)，各自pre-RMSNorm及残差。旧SA及专属norm完全不注册，不在U和Uhat之间保留旧norm残余。
3. point版为plain GELU FFN hidden2d；conv版保持外层RMSNorm→denseConv2d(d,d,3,padding1,groups1)→内部LN→Linear(d,2d,biasFalse)→GELU→Linear(2d,d,biasTrue)，按原H/W和点序还原，无下采样。latent不卷积。
4. T严格是FFN2后、Up norm前；不要保存H或norm(T)替代。公共模块初始化按已定规范，learned Down query正交；新kernel Xavier/w0/gamma0.1在特殊初始化后不再被递归apply覆盖。复用类但每层独立参数。
5. core接收lift后的点特征，最后返回点特征供任务head；不添加Bridge/persistent/final Up。若项目接口合理地把head一起封装，计算上仍只保留最后一个block自己的Up。输入提升/时间注入留在已验证任务语义中。
6. 每次forward创建新的局部history；先读取旧tuple，再计算当前T并写入。第一层无reader，最后层无writer，L1和history=off都没有历史参数。off仍保留L次Down/Up和2L次latent FFN。
7. 默认L8实际计数：Down8、Up8、latent SA0、latent FFN16、PointModule8、writer7、query7、读取历史28。读取份数与批量API调用数分开报告，不能把一次batched matmul当一份历史。
8. 训练history不断梯度；不跨forward/NS时间；可选debug只在测试开启，正常forward不常驻attention矩阵、retain_grad、CPU同步统计或历史输出。不要返回改变任务合同的tuple。

验收：
- 小模型L1/2/4/8及一个扩展L（例如12），合法d/h/M/r组合、非法配置；不要求完整超参数笛卡尔积。
- 执行计数与模块参数存在性、层间参数/storage独立、首末层边界。
- 用独立手工组合公共原语和K2reference核对一层及两层前向；覆盖Up norm只一次、T抽取位置和残差。
- KCDNO(all,所有gamma=0)与KCDNO(off)显式复制公共权重后输出一致。只在测试用白名单映射，不在生产用strict=False。不拿旧CDLNO no_sa作全网络同构对照；all gamma0仍有历史计算，成本不等于off。
- eval A→B→A调用输出一致，历史无泄漏；隔离历史分支的梯度复用K2，再补核心因果链证据。
- 5×7非方形网格及N!=H*W拒绝，point版支持不同N；同配置strict state_dict往返。
- 重跑本次触及公共部件相关的旧模型基准，未触及的已有效证据保留，不随意扩大测试范围。

给出真实类/函数结构、参数表和历史时序，更新STATUS后停止。
```

## K4：接入四个静态标准任务

```text
K3已审查通过。现在只接入KCDNO在Darcy、Elasticity、Airfoil、Pipe上的训练/评估能力；时间和工业任务留待后续。

1. 复用当前CDLNO已有输入提升、输出头、任务wrapper结构与model factory约定；新增kcdno分支/薄wrapper，保持旧返回和模型注册可用。只修改必要模型选择、KCDNO参数转发、独立profile/脚本和metadata加载。数据、loss、normalizer、optimizer/scheduler及循环不改。
2. 合同：forward(x,fx,T=None)→[B,N,Cout]。Darcy fx1/输出1，reference64替换xy后拼fx，stem65；Elasticity/Airfoil/Pipe fx=None、输出1，坐标和placeholder沿用现有语义。不要给固定fx任务注册闲置placeholder，不在wrapper末尾重新初始化core。
3. 原典型布局：Darcy85×85，Elasticity972点，Airfoil221×51，Pipe129×129。Darcy/Airfoil/Pipe用ConvFFN，Elasticity用PointFFN；保持曲线网格原索引，不按物理坐标重新排序。
4. 新主配置按K1：Darcy128/8/64，Elas128/8/64，Airfoil128/4/64，Pipe128/4/32（d/h/M），L8、r16。训练默认500epochs；batch分别4/1/4/8，原AdamW、lr参数1e-3、原clip/wd/scheduler。若当前用户已有合法训练变更，按K0已记录协议沿用并明确报告，不覆盖旧值。
5. 为四任务提供真实存在的新训练/评估薄脚本，默认kcdno_v1，支持显式L/M/r/history/profile等覆盖；脚本用户余参放最后。检查shell参数能实际进入新配置，不被旧parser默认覆盖。不能给不存在的统一train.py命令。
6. 输出目录包含family/profile及关键resolved架构标识或配置hash，重复运行遵守现有覆盖防护。新实验不覆盖旧CDLNO/消融/Transolver结果。eval先读sidecar按架构重建，显式冲突拒绝；同配置strict加载，新模型默认从头训练。
7. 无数据核查：安全导入真实wrapper/factory，合成forward、原可复用loss连接、backward、一次optimizer step、eval和save/load；Darcy核查decode后的原相对L2及梯度项路径，不能另写近似loss冒充原loss。损失无法安全导入时明确覆盖限制，不能借机重构训练器。
8. 使用小d/M测试真实N布局，必要时缩小非方形网格测数值，分别记录；不改正式profile。静态核对CLI/exp冻结区段，不import顶层读数据脚本。旧四任务模型选择及full/no_sa/identity按K0权重回归。

交付四任务“参数→wrapper→core→输出→checkpoint→eval”映射和实际命令，明确真实数据训练未执行。更新STATUS后停止。
```

## K5：接入 NS 与 Plasticity

```text
K4已审查通过。现在只接入KCDNO的Navier–Stokes与Plasticity任务，复用共享core；不改变原时间语义，不接工业任务。

NS：
1. x[B,N,2]、fx[B,N,10]，每次输出[B,N,1]；原64维reference替换xy后拼10帧，stem74。典型N4096=64×64，ConvFFN。主配置L8/d256/h8/M64/r16，500epochs、batch2，沿用原优化器和调度。
2. 以源码核对并保留原10→10训练/评估：训练用真值回填窗口，原10步loss累加后一次backward/optimizer更新；评估预测回填。若当前已批准实现不同，按K0证据说明，不自行切换teacher forcing策略。
3. 每次model.forward都重建KCDNO历史。不能把网络深度history当真实时间state，不能只编码初态一次。原20帧数据不支持声称完成10→20/40真值评估，本轮不增加长时实验。

Plasticity：
4. x[B,N,2]、fx[B,N,1]、T[B,1]，输出[B,N,4]。空间N101×31=3131，不把20时间点拼进N；输入提升与原time embedding及注入位置保持。主配置L8/d128/h8/M64/r16，ConvFFN，500epochs、batch8。
5. 保留原每batch20个时间条件分别前向/backward/optimizer更新及原scheduler节奏，不合并成一次总loss更新，不改成自回归。

共同工作：
6. 接通新train/eval参数、profile、独立脚本/目录及checkpoint重建；旧参数和旧模型选择不受影响。
7. 合成检查NS真值/预测两种窗口回填、时间调用缓存隔离；Plasticity不同T的输入路径/梯度和原标签轴顺序。验证输出不读取未来标签或y，时间相关参数实际参与有效路径。
8. 复用K4加载/配置测试，补充时间条件checkpoint往返和两任务旧模型回归。按实际源码核对时间循环冻结区段，不能拿重写的小循环称原训练程序已跑通。

报告时间协议和optimizer/scheduler调用证据、合成检查结果及真实轨迹训练未执行。更新STATUS后停止。
```

## K6：接入 ShapeNet-Car

```text
K5已审查通过。现在只接入KCDNO的ShapeNet-Car训练/评估，不修改AirfRANS。

1. 保持forward((cfd_data,geom_data))→[N,4]；原x七通道、[velocity3,pressure1]顺序、surf mask、点序、几何输入合同、原loss和物理量评价保持。不要读y作为输入，不新增geom encoder，不删除模型不用的原数据处理。
2. 主配置L8/d256/h8/M64/r16、PointFFN；原200epochs、batch1、Adam/lr参数1e-3、reg/fold/采样和评估规则沿用当前已确认版本。一次run对应原来一个fold，不自动展开所有fold训练。
3. 复用当前CDLNO工业wrapper接入方式，新增新模型选择、参数转发、profile、独立脚本和checkpoint目录。保持单图可变N、原单图batch全0合法。沿用K0确认的图批处理合同：当前仅支持单图时多图明确拒绝；当前已支持各图隔离时保留该路径，不退化为拒绝。不能把图拼起来做全局attention，也不扩展新的批处理方式。
4. 核对实际checkpoint是state_dict还是整模型对象。保留原保存/可信加载边界；新类必须在稳定可导入模块中，不能定义在__main__。已有整对象反序列化不会调用新__init__，旧类路径不能重命名。不能全局更改torch.load策略或用strict=False掩盖错误。
5. 训练和对应评价入口都能选择/识别kcdno，eval先读架构信息，不用CLI默认覆盖checkpoint。保留原epoch路径和命名语义；如当前有影响新入口的既有bug，先记录，只有已批准的最小兼容修复可以执行。
6. 有真实PyG时用合成Data/Batch和原tuple调用完成forward、surf/volume mask原损失连接、backward、一步optimizer；检查不同N、改变y不影响eval输出、输入不被原地修改。仅支持单图的合同验证多图拒绝；已支持隔离多图的合同验证样本间隔离。禁止伪造torch_geometric模块来标注PyG通过。
7. 从Car原工作目录的新进程验证可导入、同架构save/load、可信整模型加载（若该任务实际使用）、显式错误架构拒绝；复核旧Transolver和CDLNO三模式的K0基准。

缺PyG/GPU时完成可执行的代码/纯模型项，明确真实PyG或CUDA未验证并给实际远端命令，不重装已可训练torch/PyG。更新STATUS后停止，不跑真实数据或fold训练。
```

## K7：接入 AirfRANS

```text
K6已审查通过。现在只接入KCDNO的AirfRANS，复用共享core和已验证加载工具，不改变原数据抽样、图、loss和评价流程。

1. 保持forward(data)→[N,4]，输入x7、输出[vx,vy,p,nut]。64维reference-distance追加到x7而非替换，stem71；reference域和placeholder语义按当前wrapper保持。
2. 主配置L8/d256/h8/M64/r16、PointFFN，沿用原398epochs、batch1、Adam及lr参数1e-3。实际用户已有经确认配置按K0记录保留。只在params/model配置中新增新family项，不修改旧key默认。
3. train/main_evaluation或当前实际独立评价入口都接通kcdno。保留每epoch抽样、验证反复抽样、原图构造、idx scatter/平均、surf/boundary mask及物理指标/后处理。不能因为模型不使用边就删掉图处理并归为架构加速。
4. 保留当前单图可变N规则，多图显式拒绝或沿已经确认的独立图路径处理，不新增batch方案。训练/评价my_path语义若不同，分别在脚本中说明，不假设相同目录结构。
5. 核对整模型/模型列表checkpoint的实际格式；稳定类路径、sidecar先读后验、新family识别、同架构重建与旧checkpoint加载规则保持。不得在eval覆盖已有配置或静默加载成旧CDLNO。
6. 用真实PyG合成单图检查输入七通道、参考追加、四输出、原mask/loss连接及一步backward/optimizer、可变N、字段不被改写和无y泄漏。缺真实PyG明确未执行，不创建假包。
7. 从AirfRANS工作目录新进程验证对应整对象/列表save/load与eval选择，检查输出对齐原抽样节点。原完整散射评价只做可安全复用的合成/静态验证，不能声称运行了真实数据评价。
8. 更新八任务训练/评价/加载覆盖表，补充AirfRANS旧Transolver/CDLNO三模式回归。确认新配置不会改变Car和六标准任务。

交付本任务实际命令与证据，更新STATUS后停止。
```

## K8：打通公平结构对照

```text
K7已审查通过。现在只执行K8：为已确认研究问题打通最薄的matched LRSA-full对照和KCDNO history=off对照，使它们能走同一八任务训练/评估流程。不新增研究机制，不启动真实训练。

1. 先审计原CDLNO阶段9是否已有lrsa_matched：核查是否真是L个完整LRSA block、相同stem/head/point模块、末层后没有额外Up；是否只在性能工具临时定义；是否已注册为正式可训练模型。能复用的复用，不能因名字相同就宣称完成。
2. 对照结构固定：
   matched LRSA-full：L次Down→FFN1→SA→FFN2→Up→PointModule，无history/Bridge/persistent/final Up。
   matched LRSA-noSA：同上删除SA，不加history；优先直接使用KCDNO(history_mode=off)，文档说明这是该计算图对照，不必重复实现第三套core。
   KCDNO主版：同样公共部件，两FFN间核化全历史读取，SA为0。
   原Transolver按原任务配置保留，旧CDLNO三模式也独立保留。
3. matched LRSA-full与KCDNO保持相同任务输入提升、输出head、L/d/h/M、两个FFN hidden/activation、点域FFN/ConvFFN、norm、公共部件初始化和训练协议。SA采用既有已验证LRSA原语；不要把front_latent_mode全局透传而把full对照也消融。
4. 新对照采用独立family/variant和metadata。若已有lrsa_matched命名可用就沿用并记录；不重命名旧pickle类。通过八任务相同wrapper/factory、参数转发、脚本和checkpoint接入，不能只在性能脚本有一个临时类却声称可训练。
5. 主实验采用kcdno_v1共同尺寸；另提供transolver_shape_match共同尺寸覆盖，明确仅匹配L/d/h/M，不等参数/等FLOPs。LRSA官方论文配置、当前YAML和这个受控对照不同，不能称完整官方复现。
6. 验证同尺寸公共权重的白名单复制只用于数学测试；正常训练各自从头初始化，正常加载只接受同变体。off不注册历史参数。gamma=0的all与off数值等价，但前者仍有历史计算，不能混用成本。
7. 对每个任务核对三个计算图的选择、训练/评估参数、元数据和save/load。复用K4—K7已覆盖合同，只补新增对照分支风险，不重跑所有大配置。

交付八任务可训练/评估对照命令、实际参数/公共部件差异表。不要自动训练三组对照或跑超参搜索。更新STATUS后停止。
```

## K9：核对完整成本并提供有限性能工具

```text
K8已审查通过。现在只执行K9：复用现有无数据性能工具，核对KCDNO及对照的参数、MAC和有限代表性性能。不修改数学设计、数据处理、默认精度或训练协议来追求好看结果。

1. 参数数来自实际构造模型，区分core与stem/head/整网，检查删去SA参数确实不存在，writer/readers无首末闲置参数。计入norm、bias、gate/scorer实际小项，不把公式近似称精确总参数。
2. 对等宽d、FFN hidden2d、raw values、单组r的新机制，用理论主项交叉核查：
   P_new = P_matchedLRSA - 4L*d^2 + 2(L-1)*d*r + O(Ld)
   history_MAC = B*M*d*r*(L-1)*(L+6)/2
   removed_SA_MAC = B*L*(4*M*d^2 + 2*M^2*d)
   point_LRSA_MAC = B*L*(8*N*d^2 + 15*M*d^2 + 4*N*M*d + 2*M^2*d)
   conv_LRSA_MAC = point_LRSA_MAC + 9*B*L*N*d^2
   new_MAC = baseline_MAC - removed_SA_MAC + history_MAC，另报逐元素等未计项。
3. 必须包含每层N规模Down/Up投影、attention、点域FFN/dense卷积，以及Q/K投影、摘要写入、全部历史读取、depth评分/softmax和临时stack。profiler漏SDPA等项不能计作0；若采用FLOPs≈2MAC，明确忽略了哪些逐元素操作。
4. 默认L8：Down/Up各8、SA0、latent FFN16、PointModule8、写7/query7/读28；all-depth历史计算随L有二次项，不能宣称对深度线性。
5. 参考核对L8/M64/d128/r16的主要删除与新增参数差为495616；PointFFN core主项约降16.09%，ConvFFN约11.63%。Darcy N7225的ConvFFN core主项MAC仅降约0.196%。实际实现含小项可有差异，但必须解释，不能把局部84.69%降幅写成整网加速。
6. 复用计时工具，比较matched LRSA-full、KCDNO off/all、原Transolver（原配置与shape-match分开）。用相同device、batch、输入、AMP/TF32/compile/backend策略，核机制规定FP32路径记录清楚，不能偷偷换低精度公式。先warmup，CUDA同步/events，报告median/p90及峰值allocated显存；训练步写明包含forward/loss/backward/optimizer和优化器状态是否已初始化。
7. 当有GPU时只执行有限代表性合成配置，不自动跑八任务全宽/多seed/多rank大矩阵。无GPU完成CPU功能和成本核对，给实际可执行远端命令，GPU项标未执行。不得由CPU或MAC推断4090实际延迟，不把合成step时间当真实epoch时间。
8. 推理摘要缓存大小B(L-1)(rd+r)，训练autograd保留额外激活；不能用推理cache小宣称训练显存一定更小。正常路径不为统计保留全部T/R/attention或触发每层CPU同步；诊断按需开启。
9. 若新模型更慢，报告热点和原因，不改成只读相邻历史、不预合并摘要、不增加共享参数或detach，不改变r或默认gamma来掩盖。后续优化另行讨论。

更新性能工具说明、实际命令、参数/成本范围和未实测项，停止。准确率实验、达到同误差所需时间及真实epoch训练不在本阶段执行。
```

## K10：最终独立审查与交付

```text
K9已审查通过。现在只执行K10：从已确认规格反查代码，完成最终综合审查、范围内缺陷修复和交付。不新增机制，不下载数据，不启动真实训练，不自动commit/push。

1. 建“需求→实际文件/符号→验证证据→状态”矩阵，逐条覆盖KCDNO规格及总控；独立检查diff，不能只拼接此前报告。若测试与实现共用同一错误公式，补充独立reference。
2. 重点反查：两FFN间history、T位置、Up norm一次、无SA/Bridge/persistent/额外finalUp、源层Wk/接收层Wq、raw value、sum摘要、phi/eps、单组r、深度softmax、gamma初始化/参数化、首末层参数和单次forward因果缓存。
3. 反查八任务实际train和eval入口，而不仅是core：CLI/profile显式覆盖、位置/时间条件、输出通道/点序、单图规则、normalizer/loss/optimizer/scheduler冻结、metadata先读后验、真实save/load格式及稳定import路径。检查KCDNO all/off和matched LRSA-full都可选择，旧家族不接新kwargs。
4. 用K0改动前同权重固定输入证据核查旧Transolver、CDLNO full/no_sa/identity。旧CDPA off/entry/every_block和已有执行选项的行为不应改变；只补本次触及路径的必要组合，不做无限笛卡尔积。旧缺陷与新回归分别说明。
5. 执行一次约定的必要最终验收套件：数学oracle/生产精度、history/off/L1、扩展L、因果/梯度、任务合成训练步、checkpoint/factory/脚本配置、真实可用PyG和GPU项。复用有效证据，不为“全部绿色”伪造缺失环境或跳过标记。
6. 从三个原任务工作目录的新进程核查包导入和需要的工业对象加载。脚本先静态语法/参数流检查；顶层读取数据的程序不能为了检查直接运行。不给不存在的选项或统一入口命令。
7. 交付独立KCDNO README段落/使用文档、模型架构与公式映射、八任务训练/评估/profile覆盖命令、all/off和matched LRSA对照命令、checkpoint规则、环境说明、无数据检查命令、有限性能命令以及未验证项。保留旧README使用说明，不覆盖此前计划与A阶段文档。
8. 更新docs/KCDNO_IMPLEMENTATION_REPORT.md和STATUS，记录实际commit/工作区修改、阶段结果与剩余限制。若确有未完成代码项，明确列出，不把“预留接口”写成已支持；在本阶段授权内修复新缺陷，若需改变模型或数据协议则先完成其他项并说明具体冲突。

最终分别回答：
A. 已确认架构哪些条款已实际实现；
B. 八任务哪些静态/合成模型步/原loss连接/PyG/checkpoint/GPU检查已运行并通过；
C. 旧CDLNO与消融保留的证据；
D. 真实数据读取完整性、收敛、准确率、实际GPU训练/推理及epoch时间中哪些仍未验证。

交付可审查diff、真实命令和报告后停止。只有我以后明确要求时，才执行真实数据训练或额外模型实验。
```

## 每个阶段之间可以直接使用的提示词

### 接受并进入下一阶段

```text
我已审查并接受K[N]阶段。现在只执行K[N+1]，范围与验收如下：
[粘贴下一阶段完整正文]

此前总控及旧模型/数据/训练冻结范围持续有效。完成后停止，不自动继续。
```

### 本阶段返修

```text
K[N]阶段暂不通过，不进入下一阶段。请只修复以下问题：
[列出具体问题]

先解释原因、涉及文件和规格条款，再完成本阶段范围内修复。只重跑受影响检查及必要旧模型回归，保留其他有效证据。不要借返修更改已确认的phi/gamma/历史范围/FFN/数据协议，也不实施后续阶段。更新报告后停止。
```

### 数学／接口独立只读审查

```text
请只读审查K[N]阶段的diff、KCDNO规格和实际验证证据，不修改文件、不进入下一阶段。

重点找：公式与代码不符、残差重复、FFN顺序/Up norm错误、source-owned缓存被改成receiver重算、来源softmax轴错误、double与FP32验证混淆、历史detach或未来泄漏、旧模型参数/默认值变化、训练/评估漏入口、测试通过但没调用真实模型。

按严重程度给出具体文件/符号、证据、影响和最小修复建议。区分明确缺陷、需要验证的风险和设计取舍；没发现问题也说明检查范围，不写“保证无bug”。
```

### 缺真实数据或 GPU 时继续完成当前阶段

```text
当前仍没有真实数据，也不要求下载。请继续完成K[N]已授权范围内所有能通过源码、合成输入、真实模型、现有可用PyG和checkpoint完成的实现与验证。

不要以缺数据为理由只交付计划或空接口，也不要生成同名假数据冒充真实bench。缺GPU/PyG的项单独列“未执行”，提供基于实际脚本的远端命令。CPU合成通过不代表CUDA/真实训练通过。完成当前阶段后停止。
```

### 重新打开会话后的恢复

```text
继续KCDNO分阶段实施。先读取适用AGENTS.md、KCDNO_Model_Specification_v1.md、KCDNO_REFERENCE_AUDIT、KCDNO_IMPLEMENTATION_STATUS及当前git status/diff，核对实际工作和已接受检查点。

原CDLNO和front full/no_sa/identity已经是需要保留的已有模型，不重新实施旧计划。KCDNO新线由我接受到K[N]；本轮只授权：[当前阶段返修，或下一K阶段及完整正文]。

复用仍有效的测试证据，不从头重写。报告与实际文件不一致先核查，不推断剩余全文已获授权。完成本轮范围后停止。
```

### 可选：用户审查后的本地 Git 检查点

```text
我已审查并接受K[N]。现在仅授权将本阶段已审查的变更创建本地Git commit，作为恢复检查点。

先核对staged/unstaged与K0用户已有改动，按明确文件/hunk选择，不盲目git add -A，不混入无关代码或大回归夹具，不reset/revert我的工作。只本地commit，不push、不建PR、不执行后续阶段。返回hash、提交内容及剩余工作区状态。
```

## 你审查时最值得看的内容

1. **旧模型有没有实际保住。** 看改动前夹具、同权重回归和旧CLI/类路径，而不是只看一句“未影响旧功能”。
2. **核缓存是不是真的复用。** 看源层只写一次、receiver只读摘要，以及L8的7写/7query/28读。
3. **数学位置有没有变。** history在FFN1与FFN2之间，T在FFN2后、Up norm前，最后没有额外decoder。
4. **八任务是否真的接到新core。** 训练能选、评估能重建、checkpoint能加载，不能只演示随机张量经过一个孤立模块。
5. **证据有没有越界。** 合成训练步可以没有数据完成；真实数据读取、精度、收敛和真实epoch时间仍需另做实验。

本提示词已经把当前阶段内部的常规实现与必要验证授权给Codex。阶段结束停止是你要求的审查节奏；不需要在每一次文件编辑前再询问你。
