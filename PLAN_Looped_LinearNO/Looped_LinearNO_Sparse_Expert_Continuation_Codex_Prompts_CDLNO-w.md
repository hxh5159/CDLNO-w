# 面向 `hxh5159/CDLNO-w` 的 Looped LinearNO 稀疏专家残差续接实施提示词

> 目标仓库：<https://github.com/hxh5159/CDLNO-w>
>
> 前置实现：按照 `Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md` 完成并审计过的 `linearno_loop` 模型族。
>
> 本文件是**续接阶段**，不是重新实现 Looped LinearNO。它只做两件事：
>
> 1. 将今后所有新建 Looped LinearNO 实验的默认 latent rank \(M\) 从此前的 \(2M_{\rm base}\) 修正为 Transolver benchmark 对应的 \(M_{\rm base}\)；
> 2. 在不破坏既有 dense-loop 配置的前提下，增加可选的“原始稠密 FFN + 零初始化稀疏专家修正 + Zero Expert”。

## 使用方式

1. 将本文件完整交给 Codex，但首次只授权“总控提示词 + `LSE0`”。
2. 每次只执行一个阶段；检查该阶段报告、diff 和测试后，再授权下一阶段。
3. 本文件假定用户实际 checkout 已经包含上一轮 LL0–LL10 及其后续修复。公开远端可能落后，实际工作树才是实施真值。
4. 不得因为公开远端没有 loop 代码而重新运行旧提示词、覆盖当前实现或建立第二套 loop family。
5. 下载数据、长时间真实训练、远端 GPU、commit、push、PR 均不在默认授权内。

---

# 一、必须先理解的兼容边界

## 1. 既有 Looped LinearNO 完全保留

已有模型的以下语义不得改变：

- family：`linearno_loop`；
- 两个 preset：`p1_c3_r2_s1` 与 `p2_c2_r2_s2`；
- custom `P/C/R/S`；
- 三种且仅三种残差系统：
  - `sr_1_over_r`；
  - `rb_attnres`；
  - `lb_attnres_1_over_r`；
- 每次 visit 都从当前点状态重新计算 LinearNO 的 \(Q,K,V,K^\top V,QC\)；
- prefix/core/suffix 的参数所有权与共享关系；
- 点域 AttnRes 的来源时序、RMSNorm-key/raw-value、source-softmax、零 query 初始化；
- 原八任务的输入输出、位置/时间条件、normalizer、loss、metric、checkpoint、resume、评估和可视化；
- 旧 pure LinearNO、`linearno_history`、Transolver、CDLNO/KCDNO/MSAR-LNO 的所有默认行为。

本轮不能借机调整 AttnRes、残差缩放、循环拓扑、输出头位置或原 LinearNO 算子数学。

## 2. 默认 rank 修正不是旧 checkpoint 迁移

此前提示词令新 loop run 默认使用

\[
M_{\rm resolved}=2M_{\rm base}.
\]

从本轮开始，新建 Looped LinearNO 训练在用户没有显式指定 actual rank 或 multiplier 时必须使用

\[
\boxed{M_{\rm resolved}=M_{\rm base},\qquad \text{rank\_multiplier}=1.}
\]

当前 benchmark 的目标默认值为：

| 任务 | \(M_{\rm base}\) | 新建 loop run 默认 \(M\) |
|---|---:|---:|
| Airfoil | 64 | 64 |
| Darcy | 64 | 64 |
| Elasticity | 64 | 64 |
| Pipe | 64 | 64 |
| Plasticity | 64 | 64 |
| Navier–Stokes | 32 | 32 |
| AirfRANS | 32 | 32 |
| ShapeNet-Car | 32 | 32 |

这些值必须从当前仓库的权威 task profile/launcher 动态解析，表格只用于核对，禁止在模型类里硬编码。

兼容规则：

- 已存在且 metadata 明确记录 `rank_multiplier=2` 或 `resolved_rank=2M` 的 run/checkpoint 必须继续按原值严格恢复；
- 不得重写旧 config、sidecar、checkpoint、目录名或 architecture signature；
- 显式 multiplier 2 可保留为历史复现或 rank 消融，但不再是默认主配置；
- 新 train 省略 rank 参数时使用 multiplier 1；resume/eval 始终以 checkpoint 中的 resolved rank 为真值；
- 旧纯 LinearNO profile 和非 loop 模型默认值不得改变。

---

# 二、冻结的新增模型结构

## 3. 作用位置

新增结构只作用于 Looped LinearNO **共享 recurrent core 中实际被调用的原始点域 FFN/MLP 分支**。

- prefix FFN 不变；
- suffix FFN 与最终输出头不变；
- LinearNO operator 分支不变；
- 每个独立 core block 位置拥有自己的 router、真实专家池和门控标量；
- 同一 core block 的这些新增参数在所有 loop visit 之间严格共享；
- 不同 core block 位置之间不共享专家池；
- 不使用 loop index、round embedding 或执行步编码作为 router 输入。

若 core block 的 FFN 输入为

\[
U\in\mathbb{R}^{B\times N\times H},
\]

其中 \(U\) 是原 block 已完成 `ln_2` 后、原稠密 FFN 实际接收的 tensor，则 router 和专家都读取同一个 \(U\)。不得绕过原 `ln_2`，也不得另加会改变 dense 路径的归一化。

## 4. 原始稠密 FFN 永远保留

记原 core block 的稠密 FFN 为

\[
F_j^{\rm dense}(U).
\]

它的模块对象、参数、初始化、激活和调用语义必须沿用已有 Looped LinearNO，不得替换为 MoE，也不得改变宽度。

Zero Expert 只表示不执行**额外修正**；任何点即使选择 Zero Expert，也仍然执行原始稠密 FFN。

## 5. Router、真实专家与 Zero Expert

第一版固定使用 \(E=4\) 个真实专家和 1 个 Zero Expert。对 core 位置 \(j\)：

\[
\ell_j(U)=W_j^{\rm router}U+b_j^{\rm router}
\in\mathbb{R}^{B\times N\times(E+1)},
\]

\[
p_j(U)=\operatorname{softmax}_{\rm FP32}(\ell_j(U)).
\]

Top-1 选择为

\[
k_j^*(b,n)=\arg\max_{k\in\{1,\ldots,E,0\}}p_{j,k}(U_{b,n}),
\]

其中 0 表示 Zero Expert。实现中建议把 Zero Expert 放在最后一个索引，以避免全相等 logits 时 tie-break 全部落到 Zero Expert；该索引约定必须写入 schema 和测试。

第 \(e\) 个真实专家为窄点域 MLP：

\[
\mathcal E_{j,e}(U)
=W^{(2)}_{j,e}\,\phi(W^{(1)}_{j,e}U+b^{(1)}_{j,e})+b^{(2)}_{j,e}.
\]

Zero Expert 固定为

\[
\mathcal E_{j,0}(U)=0.
\]

它没有参数，不得构造伪 Linear 层，也不得执行真实专家再乘零。

稀疏修正为

\[
S_j(U_{b,n})=
\begin{cases}
0, & k_j^*(b,n)=0,\\[2mm]
p_{j,k_j^*}(U_{b,n})\,\mathcal E_{j,k_j^*}(U_{b,n}), & k_j^*(b,n)>0.
\end{cases}
\]

被选概率使用在完整 \(E+1\) softmax 中的值；Top-1 后不得把它重新归一化成 1。

## 6. 零初始化修正系数

每个独立 core block 位置只有一个可学习标量 \(a_j\)：

\[
\gamma_j=\tanh(a_j),
\qquad a_j\big|_{\rm init}=0.
\]

增强后的 FFN raw branch 为

\[
\boxed{
F_j^{\rm aug}(U)
=F_j^{\rm dense}(U)+\gamma_j S_j(U).
}
\]

因此初始化时

\[
F_j^{\rm aug}(U)=F_j^{\rm dense}(U)
\]

逐元素成立。零初始化的是标量 \(a_j\)，不是所有专家权重。真实专家和 router 必须正常初始化，否则会出现对称性或无法启动的问题。

第一步的预期梯度结构必须写入测试：

- 若有点被路由到真实专家，\(a_j\) 可从任务损失获得梯度；
- 在 \(a_j=0\) 时，真实专家从任务损失得到的梯度为零，这是预期的分阶段启动；
- router 可从辅助路由损失获得梯度；
- 当 \(a_j\) 偏离零后，真实专家与 router 才从任务损失获得有效梯度。

## 7. 专家宽度与参数容量

令原稠密 FFN 的真实中间宽度为 \(D_{\rm ff}\)。第一版固定

\[
D_{\rm expert}=D_{\rm ff}/E,
\qquad E=4.
\]

必须从实际 block/MLP 实例解析 \(D_{\rm ff}\)，不能假定所有 variant 都等于 \(H\)。若不能整除，构造阶段 fail-fast，不得静默取整。

忽略 bias 时，每个 core 位置全部真实专家的存储参数约为

\[
E\cdot 2HD_{\rm expert}=2HD_{\rm ff},
\]

即约等于一个原稠密 FFN。该设计补回共享 block 所减少的 FFN 参数部分，不声称补回完整 operator block 的全部参数。

真实专家使用与原 FFN 相同类型的激活函数，但第一版专家内部 dropout 固定为 0，避免零门控分支额外消耗全局随机数并改变后续随机轨迹。不得因此修改原稠密 FFN 的 dropout。

## 8. 真正的稀疏执行

实现必须按每个真实专家的选中点索引进行 gather → expert → scatter：

- 每个点最多调用一个真实专家；
- 选择 Zero Expert 的点不调用任何真实专家；
- 某真实专家本轮没有点时不得调用该专家；
- 禁止计算全部 \(E\) 个专家后再用 one-hot mask 求和；
- 不设置 capacity factor，不丢点，不做 token dropping，不随机改派；
- 恢复输出时必须保持原 \([B,N,H]\) 点顺序；
- 若任务提供有效点 mask，padding/无效点必须产生零修正且不进入路由损失和利用率统计。

## 9. 与三种残差模式的精确组合

新增结构只替换 core 中“计算 FFN raw branch”的内部函数：

\[
F_j^{\rm dense}(U)\longrightarrow F_j^{\rm aug}(U).
\]

不得增加第四种 residual mode。

### 9.1 `sr_1_over_r`

完整增强 FFN 分支统一乘原来的 \(1/R\)：

\[
X^+=X^{\rm op}+\frac1R F_j^{\rm aug}(\operatorname{LN}_{j,2}X^{\rm op}).
\]

不得只缩放 dense 项或只缩放专家项。

### 9.2 `rb_attnres`

对应 MLP sublayer 的 raw branch output 改为

\[
u_{r,2j}=F_j^{\rm aug}(\operatorname{LN}_{j,2}h_{r,2j}).
\]

随后按原 RB 规则累加到 raw partial。仍然：

- 不做 `h+u`；
- 不乘 \(1/R\)；
- 不改变 source list、receiver、round summary 或 final AttnRes；
- 不把 router logits、专家输出或 Zero Expert 状态加入 AttnRes history。

### 9.3 `lb_attnres_1_over_r`

轮内 SR 的 FFN branch 使用完整 \(F_j^{\rm aug}\) 并乘 \(1/R\)。边界 \(\Delta_r=Y_r-H_r\)、boundary receiver 和 output receiver 的公式完全不变，专家修正不得再次单独缩放。

## 10. Router 辅助目标

Router 损失按每个独立 core 专家池分别计算，再对 core 位置与本次 forward 中的 visit 求平均；不同专家池的同一索引不能混为一个专家。

设 Zero Expert 概率为 \(p_{t,0}\)，第一版固定目标真实专家激活率

\[
\rho=0.5.
\]

预算损失为

\[
\mathcal L_{\rm budget}
=\left(\frac1T\sum_{t=1}^T(1-p_{t,0})-\rho\right)^2.
\]

真实专家均衡损失只在被路由到真实专家的有效点集合

\[
\mathcal A=\{t:k_t^*>0\}
\]

上计算。对真实专家 \(e=1,\ldots,E\)，定义

\[
f_e=\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}
\mathbf 1[k_t^*=e],
\]

\[
q_{t,e}=\frac{p_{t,e}}
{\sum_{j=1}^{E}p_{t,j}+\varepsilon},
\qquad
\bar q_e=\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}q_{t,e},
\]

其中固定 \(\varepsilon=10^{-9}\)。

\[
\mathcal L_{\rm bal}
=E\sum_{e=1}^{E}\operatorname{stopgrad}(f_e)\bar q_e.
\]

若 \(|\mathcal A|=0\)，定义 \(\mathcal L_{\rm bal}=0\)，不得产生 NaN。硬 dispatch fraction \(f_e\) 只作为 stop-gradient 路由占比，梯度通过 \(\bar q_e\) 进入 router。不得把 Zero Expert 与每个真实专家强制成相同使用率。实现前必须为该公式写独立 oracle。

总训练目标为

\[
\mathcal L
=\mathcal L_{\rm task}
+\lambda_{\rm bal}\mathcal L_{\rm bal}
+\lambda_{\rm budget}\mathcal L_{\rm budget}.
\]

第一版默认：

\[
\lambda_{\rm bal}=10^{-2},
\qquad
\lambda_{\rm budget}=10^{-2},
\qquad
\rho=0.5.
\]

要求：

- task metric、验证误差和测试误差只使用原任务定义，不混入 aux loss；
- 日志分别记录 task、balance、budget 和 total loss；
- dense 模式不计算、不注册也不返回路由损失；
- 多步 rollout/多查询任务必须汇总同一训练样本中所有实际 core visit 的 aux 项，不得只取最后一次 forward；
- 不允许通过 module 上长期保存带图 tensor 的方式跨 batch 累积损失。

## 11. 数值与 dtype 合同

- router logits 与 softmax 在 FP32 中计算；
- 选中概率在乘专家输出前转换到专家输出 dtype；
- \(\gamma_j\) 在使用时转换到分支输出 dtype；
- `dense + correction` 的结果必须与原 dense FFN raw output 具有相同 device、shape 和 dtype；
- 不得让 FP32 gate 意外把 RB raw partial 永久提升为 FP32；
- 必须复用当前仓库对 SR/RB/LB 已确定的 AMP 来源/累加 dtype 合同；
- CPU FP64/FP32、CUDA FP32、可用时 FP16/BF16 均需测试 finite forward/backward；特别覆盖 RB。

## 12. 初始化与随机性

- 先按原逻辑完整构造和初始化公共 loop backbone；
- 再在隔离 RNG 上下文中初始化 router/专家；恢复外部 CPU/CUDA RNG 状态；
- dense 与 sparse 配置在相同 task/topology/residual/rank/seed 下的公共 backbone 键和值必须完全一致；
- 三种 residual mode 的相同专家位置在 paired seed 下应使用可复现的专家初始化；
- router 使用正常的小随机权重与零 bias；不得正向偏置 Zero Expert；
- Zero Expert 是最后索引；\(a_j=0\)；
- 关闭扩展时不得为了“统一类结构”注册休眠的 router、专家或 gate。

## 13. 配置与 checkpoint 合同

新增一个互斥枚举，而不是多个布尔开关：

```text
ffn_mode = dense | sparse_residual
```

建议 CLI：

```text
--linearno-loop-ffn-mode dense|sparse_residual
--linearno-loop-num-real-experts 4
--linearno-loop-router-target-active-fraction 0.5
--linearno-loop-router-balance-loss-weight 0.01
--linearno-loop-router-budget-loss-weight 0.01
```

`top_k=1`、Zero Expert、专家宽度策略、placement 和跨 visit 共享是冻结结构字段，不开放为常规搜索开关，但必须写入 architecture metadata。

兼容规则：

- 旧命令省略 `ffn_mode` 时解析为 `dense`；
- 旧 loop checkpoint/schema 中不存在该字段时，只能正规化为 `dense`；
- dense 构造不注册新增参数，旧 state dict 仍用 `strict=True`；
- sparse checkpoint 必须记录完整 `sparse_ffn_spec`、architecture signature、loss 权重和实现版本；
- dense/sparse、专家数、专家宽度、目标激活率、placement 任一不符都必须在加载 tensor 前失败；
- 禁止用 `strict=False`、忽略 missing/unexpected keys 或随机补专家权重完成转换；
- optimizer/resume 必须核对新增参数组、shape、state 与 RNG；
- 结果目录必须编码 `ffn-dense` 或稀疏结构摘要，不能覆盖旧 dense-loop run。

## 14. 与 MoE++ 的关系边界

该设计只借鉴 MoE++ 的 Zero Expert 与条件计算思想。它不是原论文的直接复现：

- 原 MoE++ 以 Top-2 为主并替换/组织 MoE 层；这里原稠密 FFN 始终保留，额外分支使用 Top-1；
- 本设计不加入 Copy Expert、Constant Expert、gating residual、capacity dropping 或跨层 router history；
- 不能把 MoE++ 在语言模型上的吞吐或精度结果当作本神经算子实现已经验证的结论；
- 论文与报告应称为 “MoE++-inspired zero-expert sparse residual FFN”。

---

# 三、总控提示词

```text
你将在用户当前的 hxh5159/CDLNO-w checkout 中，续接已经完成的 linearno_loop 实现。不要重新实现旧 loop 模型。本任务先把新建 loop run 的默认 rank multiplier 从2纠正为1，同时保持所有旧2M checkpoint按metadata严格可恢复；然后增加可选的 sparse_residual FFN 扩展：原稠密FFN + tanh(零初始化标量) × (Top-1真实窄专家或Zero Expert)。

开始前：
1. 完整阅读根AGENTS.md及其要求的状态文档；重点阅读docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md、IMPLEMENTATION_REPORT、CONFIGURATION、COMMANDS、PERFORMANCE、loop audit、当前tests，以及用户后来为LL9/RB AMP问题加入的修复与报告。
2. 输出实际repo root、remote、branch、HEAD、git status --short、tracked/untracked/ignored摘要。当前工作树可能领先公开远端或包含用户改动；禁止reset、clean、stash、checkout覆盖、rebase、commit、push。
3. 以当前已经验收的linearno_loop代码为真值。若旧LL提示词与现代码冲突，先报告；不得从公开远端重建第二套实现。

冻结范围：
4. pure LinearNO、linearno_history、Transolver、CDLNO/KCDNO/MSAR-LNO以及旧数据/loss/metric/checkpoint/visualization默认冻结。
5. 既有P/C/R/S、SR/RB/LB、点域AttnRes、输出头、AMP合同和每visit重新计算Q/K/V的语义全部冻结。
6. ffn_mode=dense时不得注册新增参数，旧命令、旧state_dict、旧checkpoint、旧结果路径和数值必须保持兼容。
7. 新建dense-loop和sparse-loop run在省略rank字段时都默认M=task-profile base M。旧2M run仍由metadata恢复；显式2倍仅作为历史/消融能力。
8. sparse_residual只放在共享core中实际调用的原FFN旁路：dense FFN始终执行；每core位置4个窄专家+1个无参数Zero Expert；专家池跨visit共享；每点Top-1；gamma=tanh(alpha), alpha=0；prefix/suffix/operator不改。
9. 真实稀疏dispatch，不得全专家计算后mask；无capacity/token drop；不加入Copy/Constant Expert、router history、timestep、LoRA或新residual mode。
10. SR/LB对完整增强FFN应用原1/R；RB把增强FFN作为原raw MLP branch，不改变partial/source/receiver且不加1/R。
11. 必须先写独立oracle，再写生产实现。不能调用生产forward生成expected。
12. 所有新checkpoint metadata-first、architecture signature严格校验、state_dict strict=True。旧schema缺失sparse字段只可映射为dense。
13. 未经本阶段和用户明确授权，不访问真实数据、不启动长训练、不安装/降级torch/CUDA/PyG。
14. 每阶段只做本阶段内容，更新docs/LOOP_LINEARNO_SPARSE_IMPLEMENTATION_STATUS.md，给出PASS/PARTIAL/BLOCKED、真实diff、命令和结果、skip/NOT RUN、冻结区检查和最值得复核的3–5点；最后写“本LSE阶段结束，未执行下一阶段”，然后停止。

数学与配置合同以本文件“冻结的新增模型结构”为准。若实际代码缺少安全插入点或会改变旧模型语义，完成不受影响的只读工作，列出文件/行、至少两个方案及影响后停止等待用户，不得自行近似。
```

---

# 四、分阶段提示词

## LSE0：当前实现与兼容性只读审计

```text
现在只执行LSE0。禁止修改生产模型、schema、parser、factory、trainer、checkpoint、launcher、旧测试golden或依赖。只允许新增docs/LOOP_LINEARNO_SPARSE_REFERENCE_AUDIT.md、docs/LOOP_LINEARNO_SPARSE_IMPLEMENTATION_STATUS.md和docs/loop_linearno_sparse_audit/lse0/**。

1. 完整阅读AGENTS.md和当前loop状态/报告/测试；记录实际repo/HEAD/dirty与上一轮LL阶段之后的所有改动。特别确认LL9的RB AMP dtype冲突是否已经修复、修复边界及当前回归状态，不重做或绕过它。
2. 建立冻结manifest：cdlno/linearno_loop/**、顶层linearno_loop/**、八任务loop adapter/入口、tran_evaluate/linearno_loop/**、checkpoint/schema/experiment、相关tests/docs，以及pure/history/Transolver冻结文件。保存size、SHA-256和进入本任务前dirty diff。
3. 从生产代码反查八任务当前权威profile：base M、H、heads、实际FFN中间宽度/激活/dropout、variant、有效点mask、forward/rollout/query调用方式。核对表中的64/32，但以代码和launcher为真值。
4. 定位rank默认值为2的所有来源：schema resolver、argparse、launcher、配置生成器、测试、文档、目录名、性能脚本。区分“新train默认”“显式2倍”“checkpoint resolved值”和“历史文本”，给出最小修正白名单。
5. 逐variant定位core FFN真实调用点及SR/RB/LB调用栈。证明一个增强raw FFN函数可被三模式共享，而不改变operator、partial、Delta或AttnRes来源。
6. 审计checkpoint兼容：旧loop schema/version、architecture signature、旧2M run、old whole-object/state_dict路径、resume/eval构造顺序。提出缺失ffn字段→dense的单向兼容规则，不实现。
7. 审计八任务训练目标如何安全取得可微aux loss，尤其NS rollout、Plasticity多查询、AirfRANS ensemble和Car fold。禁止推荐跨batch保存带图tensor；给出显式return_aux/collector方案及最小入口改动。
8. 固定MoE++论文/官方仓库版本和本项目适配边界；不得宣称官方已验证PDE效果。
9. 输出拟新增文件、最小接线文件白名单、测试计划、参数/FLOP解析式、已知风险。若当前loop实现未达到可续接状态，标BLOCKED并停止。

LSE0结束后停止，不执行LSE1。
```

## LSE1：将新建 loop run 的默认 rank 修正为基础 M

```text
LSE0已审查通过。现在只执行LSE1：纠正已有dense-loop及未来sparse-loop的新训练默认rank；不实现任何专家模块。

1. 将linearno_loop新train在actual rank与multiplier均未显式给出时的默认由2改为1。权威值必须来自task profile base M。
2. 更新schema字段来源、launcher、配置预览、run matrix、目录摘要和文档。主配置不再自动传2；显式--linearno-loop-rank-multiplier 2仍可用于历史复现/消融。
3. resume/eval先读checkpoint metadata：旧resolved=2M必须原样构造并strict load；CLI省略rank不能用新默认覆盖checkpoint。显式CLI只作一致性断言。
4. 旧config/schema缺少字段时按其已有version和保存的resolved/base信息解析；不得把无法证明的旧checkpoint猜成M。歧义必须fail-fast。
5. 不修改pure LinearNO task profile、非loop parser fallback、已有run sidecar/checkpoint或历史实验目录。
6. 测试八任务省略rank得到64/32；显式1相同；显式2得到128/64；actual rank与multiplier冲突失败；旧2M checkpoint重建不变；新M run与旧2M目录不碰撞。
7. 重跑两preset×三mode的schema/construct/synthetic forward和现有loop/pure/history回归。证明除resolved M及其派生shape/参数外，拓扑、残差、H、初始化与训练配置无漂移。
8. 更新CONFIGURATION、COMMANDS、PERFORMANCE和状态文档中“主配置2M”的现行描述；历史报告数值不篡改，只加版本说明。

LSE1结束后停止，不执行LSE2。
```

## LSE2：稀疏 FFN schema、独立数学 oracle 与实验签名

```text
LSE1已审查通过。现在只执行LSE2：实现纯配置/schema、独立数学oracle和测试；不接生产LinearNO block或八任务trainer。

1. 在现有linearno_loop architecture spec上新增ffn_mode=dense|sparse_residual。省略字段和旧schema正规化为dense；未知值失败。dense spec不得带稀疏专属字段。
2. sparse_residual冻结字段：placement=core_ffn_only、share_across_visits=true、top_k=1、num_real_experts=4、zero_expert=true/last_index、expert_width_policy=dense_width_div_num_experts、selected_probability_weighting=true、gate=tanh_scalar、gate_init=0、expert_dropout=0、rho=.5、lambda_bal=.01、lambda_budget=.01、dispatch/aux/dtype版本。
3. 从task profile的真实D_ff派生D_expert=D_ff/4；非整除、非正数、字段伪造、bool冒充数、NaN/Inf、sparse字段出现在dense模式均在torch构造前失败。
4. architecture signature包含所有影响参数shape/forward的字段；run provenance与训练loss权重按现仓库分层保存。明确哪些loss字段改变训练但不改变state shape，并在resume中仍需一致性验证。
5. 写完全独立的FP64 oracle：router logits/softmax、Top-1、Zero分支、选中概率、逐专家gather/scatter、gamma、dense+correction、budget和real-expert balance。oracle不能import生产模块。
6. 预声明CLI并测试SUPPRESS/None来源、旧命令不增字段、checkpoint优先、dense/sparse冲突早失败。
7. 生成矩阵预览：8任务×2preset×3residual×2ffn_mode×paired seeds；主M均为base M。显式2M只作为另列消融，不进入默认矩阵。
8. 输出docs/LOOP_LINEARNO_SPARSE_CONFIGURATION.md与公式→字段→checkpoint映射后停止。
```

## LSE3：任务无关的真实稀疏专家残差原语

```text
LSE2已审查通过。现在只执行LSE3：在独立模块中实现router、真实专家、Zero Expert、零初始化gate和真实稀疏dispatch；不接完整loop core或生产trainer。

1. 实现输入[B,N,H]、输出[B,N,H]的SparseResidualFFN组件。它接收一个外部dense_fn或dense raw output，但不得复制/拥有第二份原dense FFN参数。
2. Router读取与dense FFN相同的预归一化U；Linear(H,E+1)，FP32 softmax，Top-1，Zero为最后索引。Top-1后保留原softmax选中概率。
3. 每个真实专家为H→D_expert→H窄MLP，激活由显式profile传入，dropout=0。Zero Expert无Module/Parameter。
4. 对各真实专家按索引gather，只有非空专家执行；scatter回原B,N顺序。禁止evaluate-all-then-mask；加入hooks证明Zero与未选专家无forward调用。
5. alpha标量为0，gamma=tanh(alpha)。正常初始化router/experts，但用隔离RNG，构造前后外部CPU/CUDA RNG及已构造backbone参数不变。
6. logits/softmax FP32；概率/gamma/correction回到dense raw输出dtype后相加。测试不得重现RB的混合来源dtype问题。
7. 单元测试对独立oracle：手算路由、Zero输出、概率不重归一化、点顺序、B/N变化、同步点排列等变、mask、空专家、全Zero、单专家、非法shape/device/dtype。
8. 梯度测试：alpha0下prediction与dense逐元素相等；alpha有finite梯度；expert任务梯度为0是预期；aux使router有梯度；手动alpha非零后selected expert/router任务梯度finite；未选专家无任务梯度。
9. 参数计数核对router+alpha+E个窄expert；Zero参数为0。CPU double/float、可用时CUDA FP32/FP16/BF16 forward/backward/optimizer step均finite。
10. 本阶段不修改旧loop controller、任务入口或checkpoint。输出模块报告后停止。
```

## LSE4：接入共享 core，并保持 SR/RB/LB 数学不变

```text
LSE3已审查通过。现在只执行LSE4：把稀疏组件接到已有linearno_loop core FFN raw branch；不接生产训练loss和launcher。

1. 每个物理core block位置注册一个独立专家组件；同一对象随core在R轮重复调用。prefix/suffix不注册。不得按round复制专家。
2. ffn_mode=dense必须走原调用路径且不注册任何新键。对进入本任务前冻结实现比较model class/factory结果、state_dict键和值、参数量、forward、输入梯度、参数梯度和strict roundtrip。
3. sparse模式只把现有mlp_raw(U)替换为dense_mlp(U)+gamma*S(U)。不改变Q/K/V、operator、LN、输出头、调用次数或last_layer规则。
4. SR：完整augmented FFN raw branch乘1/R。RB：它作为原MLP raw u进入partial，无h+u、无1/R、无新source。LB：轮内完整分支1/R，Delta/boundary/final不改。
5. 两preset、custom和R=1分别用独立逐张量oracle验证。hook证明每core位置router/expert module id跨round相同、不同位置不同；每visit路由从当前U重算，无跨forward缓存。
6. alpha0时，对每种topology×residual×task wrapper，sparse prediction/task loss必须与dense同seed同输入逐元素相等；公共backbonehash一致。aux loss可以非零，须单独比较。
7. alpha非零时验证专家修正只改变对应FFN raw branch；SR/LB缩放一次，RB不缩放；AttnRes source count/weight/partial/Delta形状和时序与旧实现一致。
8. 参数解析：拓扑A增加3个专家池，拓扑B增加2个；实测与解析式核对。不得把补回FFN参数写成补回完整block参数。
9. 重跑当前RB AMP、三残差、旧loop、pure/history回归；0个新增失败方可PASS。输出CORE_INTEGRATION报告后停止。
```

## LSE5：路由辅助目标与六个 Standard benchmark

```text
LSE4已审查通过。现在只执行LSE5：为六个Standard任务接入可微aux收集、训练目标、日志、checkpoint和train/resume/eval；不接AirfRANS/Car，不跑完整训练。

1. 使用显式forward(return_aux=True)或LSE0批准的无跨batch状态collector。默认forward仍返回原prediction。禁止module长期保存带图loss；一次训练调用结束后不得残留旧aux。
2. 按每个专家池聚合所有实际core visits的有效点，计算real-expert balance与budget，再跨池取平均。NS rollout所有step都纳入且只纳入一次；eval默认不构造aux图。
3. total loss=原task loss+.01*balance+.01*budget；原task metric/解码/可视化完全不变。日志和checkpoint history分别保存四项loss、hard/soft active率、每专家count、entropy、gamma、dense/correction norm。
4. dense模式trainer必须走原objective且不出现新增loss键或数值漂移；旧命令和旧checkpoint严格回归。
5. sparse train/resume/eval由metadata重建；architecture与training spec冲突在model/optimizer/load前失败。optimizer包含所有新增参数且无重复；按现有bias/norm/scalar decay策略处理，不创造新的学习率。
6. 六任务做真实parser→factory→tiny/synthetic forward/backward→optimizer→checkpoint→新进程resume下一步→eval闭环。Darcy derivative、NS rollout、Plasticity查询、原normalizer/metric/可视化语义保持。
7. 覆盖2preset×3residual×dense/sparse的构造和最小数值矩阵；至少对一个代表任务完成全部12个组合的optimizer一步，其他任务做覆盖实际入口的分层测试。
8. 输出STANDARD_INTEGRATION报告；真实数据和长训练NOT RUN后停止。
```

## LSE6：AirfRANS 与 ShapeNet-Car 接入

```text
LSE5已审查通过。现在只执行LSE6：接入AirfRANS和ShapeNet-Car，保持其原ensemble/fold/评估协议；不跑真实完整训练。

1. 先从当前Looped LinearNO实现确认两个任务实际执行的dense FFN，扩展只挂到该FFN；不得启用以前未执行的参数或改变输入/输出/placeholder/位置特征。
2. AirfRANS保持单图可变N、four split、sampling、normalizer、score和ensemble。每member独立aux统计；不得跨member共享forward状态。权威ensemble/member checkpoint保存并验证sparse spec。
3. ShapeNet-Car保持7维输入、4维输出、fold、pressure/velocity/Cd、raw/preprocessed路径和B=1语义。每sample路由，点顺序恢复；不跨fold共享专家状态。
4. 两任务的train/resume/eval旧dense-loop路径完全可用；sparse路径使用相同数据预算与评估。旧whole-module/legacy state路径按当前合同保留，不将其升级为新权威格式。
5. 做tiny/synthetic single-member/fold0闭环、strict checkpoint、新进程恢复、eval产物和负向结构冲突测试。其他split/fold做parser/factory/dry-run。
6. 检查可变N、mask、Zero全选、某专家空负载、AMP、异常后无aux/history残留。输出AIR_CAR_INTEGRATION报告后停止。
```

## LSE7：统一命令、旧配置兼容、性能与消融工具

```text
LSE6已审查通过。现在只执行LSE7：扩展tran_evaluate/linearno_loop/**与性能工具，形成可直接训练评估的dense/sparse矩阵；不得启动真实长训练。

1. 旧loop命令不带ffn字段时仍为dense。新train显式支持--linearno-loop-ffn-mode；稀疏专属字段在dense时若显式给出必须报错，而不是忽略。
2. 所有任务的新默认M为base M，launcher不再暗传2。旧2M checkpoint resume/eval仍由metadata恢复；给出显式2M历史/消融命令但不列为主配置。
3. 每任务给出dense与sparse的train|resume|eval完整命令；两preset×三residual×两ffn_mode×paired seeds可由manifest展开。exact experiment dir和signature防覆盖。
4. paired dense/sparse run必须共享公共backbone初始化hash、DataLoader seed、数据顺序、优化预算和评估协议；不得挑test最佳专家配置、seed或checkpoint。
5. profiler分别报告registered/trainable/首步task-gradient-active参数；统计router、专家、gate。测量hard active率下的实际专家MAC、router/gather/scatter开销、forward/train-step median/p90、吞吐、延迟和峰值显存。
6. 解析式检查：E=4,D_expert=D_ff/4时每core位置专家参数约一个dense FFN；rho=.5时单个增强FFN期望专家主MAC约dense FFN的rho/E。拓扑A/B分别按6/8和4/8次core执行折算，但报告使用实测active率，不把目标rho当实测值。
7. 诊断输出每core位置/round的soft/hard active、专家count、entropy、gamma和correction/dense norm；默认关闭，隔离目录，不改变训练结果。
8. dry-run矩阵覆盖8任务×2preset×3residual×2ffn_mode×train/eval，并覆盖resume、custom、显式2M、所有结构/loader冲突和带空格路径。shell错误必须传播。
9. 更新COMMANDS、PERFORMANCE和机器可读profile；不得生成虚构SOTA、速度或显存结果。

LSE7结束后停止，不执行LSE8。
```

## LSE8：最终反向审计与交付

```text
LSE7已审查通过。现在只执行LSE8：不增加新功能，从冻结公式反查实现并完成最终交付。

1. 对八任务逐一追踪Stem→prefix→R轮同一core→suffix→head；核对新默认M=base、旧2M metadata恢复、两preset/custom和三残差无漂移。
2. 对每个core FFN反查：原dense始终执行；router读同一U；Top-1；Zero无参数/无调用；只选一个窄专家；概率不重归一化；gamma=tanh(alpha)且alpha0；dense+correction dtype一致。
3. 逐模式核对SR/LB只缩放完整augmented branch一次；RB把augmented raw u放入原partial且不改source/receiver/1R。
4. 核对每core位置一个专家池、跨visit共享、不同位置独立、prefix/suffix无专家、无round/timestep输入、无跨forward状态、无all-expert计算、无token drop。
5. 核对aux公式、有效点、rollout/query/ensemble聚合、task metric隔离；第一步梯度结构与文档一致。
6. 核对初始化与RNG、公共backbone paired hash、AMP dtype、optimizer参数唯一性、strict checkpoint、旧schema→dense兼容及所有错配早失败。
7. 执行目标测试、当前loop全回归、pure/history/Transolver相关回归、compile/shell/diff白名单。逐项记录通过/失败/skip和既有失败；有新增失败不得标PASS。
8. 生成docs/LOOP_LINEARNO_SPARSE_IMPLEMENTATION_REPORT.md：来源边界、数学、文件映射、rank修正、八任务配置、命令、参数/FLOP、测试证据、限制和真实实验待办。
9. 最终状态PASS/PARTIAL/BLOCKED。真实数据、完整epoch、三seed、SOTA和远端GPU未执行时明确NOT RUN，不能表述为有效性已经验证。

完成后停止，不commit/push。写“本LSE8阶段结束，未执行真实完整实验”。
```

---

# 五、用户审查时最值得检查的十二项

1. 新训练省略 rank 后是否真的得到 base \(M\)，而旧 \(2M\) checkpoint 是否仍按 metadata 恢复。
2. `ffn_mode=dense` 是否完全不注册 router、专家和 gate。
3. 原始稠密 FFN 是否始终执行，Zero Expert 是否只跳过额外修正。
4. 是否只在 recurrent core 中加入专家，prefix/suffix 是否未改变。
5. 每个 core 位置是否只有一个专家池并跨 visit 共享，而不是按 round 复制。
6. 是否真正只执行一个被选真实专家，而不是全专家计算后 mask。
7. \(a=0\) 时 prediction 是否逐元素等于 dense-loop；专家权重是否正常初始化而非全零。
8. SR/LB 是否对完整增强 FFN 只缩放一次，RB 的 raw partial/source 是否完全未变。
9. Router loss 是否不把 Zero Expert 强制成与每个真实专家等频。
10. NS/Plasticity/AirfRANS/Car 的多次 forward aux 是否完整且不跨 batch 泄漏。
11. AMP 下新增分支是否保持原 FFN raw output dtype，并未重新引入 RB dtype 冲突。
12. 报告是否区分参数容量、理论 MAC、实测延迟与尚未验证的精度。

---

# 六、参考来源

- 当前目标仓库：<https://github.com/hxh5159/CDLNO-w>
- 前一轮 Looped LinearNO 提示词：`Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md`
- LinearNO：<https://arxiv.org/abs/2511.06294>
- Transolver：<https://proceedings.mlr.press/v235/wu24r.html>
- MoE++：<https://arxiv.org/abs/2410.07348>
- MoE++ 官方仓库：<https://github.com/SkyworkAI/MoE-plus-plus>
