# LinearNO：在 `hxh5159/Transolver` 上进行论文／官方代码一致性移植的分阶段 Codex 提示词

本文件用于指导 Codex 在用户实际仓库 `hxh5159/Transolver` 中新增一个**纯 LinearNO 基线**，并使其能够在该仓库已有的六个 Standard PDE、AirfRANS 与 ShapeNet-Car 流水线上训练、评估和复现论文配置。

这里的“完整 LinearNO”不是指把某一个 `LinearAttnNeuralOperator.py` 复制八次。LinearNO 官方仓库针对 Standard PDE、AirfRANS、ShapeNet-Car 分别保留了不同的输入投影、Q/K 维度语义、温度参数、输出投影和 checkpoint 结构；必须同时结合论文与各任务官方源码实现。当前阶段只建立忠实的纯 LinearNO 基线，**不加入 CDLNO、CDPA、KCDNO、跨深度历史、差异化正则或其他研究机制**。

## 使用方式

1. 把本文件完整提供给 Codex，但首次只授权“总控提示词＋L0”。
2. 每次只授权一个阶段。Codex 完成阶段报告后，先审查 diff、测试和未决冲突，再决定是否进入下一阶段。
3. 即使 Codex 已经看到全部阶段，也不能自动执行未被本轮明确点名的阶段。
4. L0 只允许审计和新增审计文档；L1 才允许建立回归夹具；L2 起才允许新增 LinearNO 模型代码。
5. L0—L10 的目标是完成可信实现、配置、测试与复现入口。下载数据、下载官方 checkpoint、长时间真实训练、提交 commit、push 或 PR 均需另行明确授权。
6. 若换会话，先发送文末“会话恢复提示词”，再明确当前获准阶段。

## 已核验的参考快照

以下 SHA 是编写本提示词时核验的固定参考，不代表 Codex 可以 reset、merge 或覆盖用户工作区。开始实施时仍需记录目标仓库实际 HEAD、分支和 dirty 状态。

| 角色 | 固定来源 | 已核验提交／版本 | 使用原则 |
|---|---|---|---|
| 目标开发仓库 | `https://github.com/hxh5159/Transolver.git` | `66bc489fa92e9ab1fddd63ee618e1b76e8623f6b` | 唯一可写基座；若实际 HEAD 已变化，保留现状并重新审计 |
| Transolver 只读参照 | `https://github.com/thuml/Transolver.git` | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` | 核对原 benchmark 与 Transolver 行为；不能 merge/rebase 到目标仓库 |
| LinearNO 只读参照 | `https://github.com/HiPRL/LinearNO.git` | `3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` | 核对各任务真实实现、初始化、脚本和 checkpoint |
| LinearNO 论文 | *Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention* | arXiv `2511.06294v3` | 数学定义、方法主张、表格配置与报告指标 |

固定链接：

- 论文：<https://arxiv.org/html/2511.06294v3>
- LinearNO：<https://github.com/HiPRL/LinearNO/tree/3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269>
- Transolver：<https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7>
- 目标仓库已核验快照：<https://github.com/hxh5159/Transolver/tree/66bc489fa92e9ab1fddd63ee618e1b76e8623f6b>

已知目标仓库不是 GitHub 记录意义上的 fork，且与 `thuml/Transolver` 没有共同 Git 祖先；不得使用 merge-base 或三点 diff 推断用户改动。应做固定 SHA 的 tree-to-tree、blob hash、普通 diff 和必要的 AST／语义比较。目标快照中的 Transolver 模型逻辑已核验为未发生实质修改，但训练／评估产物、seed、checkpoint/resume、可视化和部分实用任务路径已被增强；这些增强属于必须保留的用户工作。

在已核验快照中，官方 71 个 tracked 文件全部仍存在：48 个 blob 相同、23 个被修改、另有 81 个新增文件。该统计只用于 L0 复核，不得取代对实际 checkout 的重新审计。旧 Transolver 六任务参数量可作为冻结指纹：Airfoil `2,810,817`、Darcy `2,826,945`、Elasticity `713,665`、NS `11,232,321`、Pipe `3,073,985`、Plasticity `2,844,484`；实施环境中应重新实例化确认。

## 全程有效的模型真值

对每个 head，令点特征投影后为 `X_h ∈ R[B,N,d_h]`，实际低秩维度为 `M`。LinearNO 的共同核心必须是：

\[
Q=\operatorname{softmax}_{M}(X_hW_Q),\qquad
K=\operatorname{softmax}_{N}(X_hW_K),\qquad
V=X_hW_V,
\]

\[
C=K^{\mathsf T}V\in\mathbb R^{B\times h\times M\times d_h},
\qquad
Y=QC\in\mathbb R^{B\times h\times N\times d_h}.
\]

温度版本只是在 softmax 前分别除以任务实现规定的 `tau_q/tau_k`。这里：

- `Q` 沿最后的低秩轴 `M` softmax；
- `K` 沿点轴 `N` softmax；
- Q/K 是独立投影，不能复用同一组路由；
- 官方代码先拆分 heads，再用同一个 `Linear(d_h,M)` 分别作用于所有 head，因此 Q/K/V 的小投影参数在 heads 间共享；不能误写成每个 head 各有一套权重；
- 没有 `1/sqrt(d_h)` 缩放；
- 没有 `M×M` slice-token self-attention；
- 没有 attention probability dropout；
- Q/K/V 投影无 bias；
- 先算 `K^T V` 再算 `Q(K^T V)`，不能构造 `N×N` 矩阵；
- 完整 block 仍是 `x += Attn(LN(x))`，再 `x += FFN(LN(x))`；
- 最后一个 block 仍经过最后 LayerNorm 与线性输出层。

### 官方实现变体矩阵

| 范围 | 输入投影 | 实际 M 的语义 | 温度 | 输出投影 | 必须保留的兼容细节 |
|---|---|---|---|---|---|
| Standard `plain` | `Linear(d,d)` | `key_ratio` 是绝对 M | 无 | `Linear(d,d)+Dropout` | 官方 `no_temp` 字符串实际落入此分支 |
| Standard `temp` | `Linear(d,d)` | 绝对 M | Q/K 各自per-head 0.5，clamp `[0.01,1]` | `Linear(d,d)+Dropout` | Elasticity 使用 |
| Standard `conv` | `Conv2d(d,d,3,1,1)` | 绝对 M | 无 | `Linear-GELU-Linear-Dropout` | Plasticity 使用；要求 `N=H×W` |
| Standard `conv_temp` | 同上 | 绝对 M | Q/K 各自per-head 0.5，clamp `[0.01,1]` | `Linear-GELU-Linear-Dropout` | Airfoil、Darcy、Pipe 使用 |
| AirfRANS | `Linear(d,d)` | `slice_num` 是绝对 M | 注册per-head `[1,h,1,1]` 的 `temperature=0.5`，但 forward 不使用 | `Linear(d,d)+Dropout` | 未使用参数仍在 state dict／参数量中 |
| ShapeNet-Car | `Linear(d,d)` | `M=key_ratio×d_h` | Q/K 各自 0.5，clamp `[0.1,2]` | `Linear-GELU-Linear-Dropout` | 官方 checkpoint 键拼作 `tempreature_q/k` |

不要用一个“统一近似实现”抹平这些差异。可以复用内部数学函数，但每个官方变体的参数形状、初始化、forward、输出和 checkpoint 映射必须单独通过一致性测试。`Super-Resoltion-AppendixE` 必须在 L0 阅读，用于防止误用其乘数式 `key_ratio` 与 cross-LinearNO；但它不属于目标 Transolver 八任务，本轮不接入生产代码。

论文的算子收敛定理只讨论理想化单层积分核与采样收敛，不覆盖完整 residual/FFN、卷积、learned temperature、确定性结构网格或多层网络；不得以该定理为理由删改发布实现。

## 配置与分域真值规则

不能用一个全局“论文优先”或“代码优先”处理全部冲突；先判断决策域，再选真值：

1. 用户本轮和后续明确确认的决定始终最高。
2. 方法身份、Q/K归一化方向、generalization/simplification边界与论文声称的指标定义，以论文v3为真值。
3. `official_release` 和 `paper_table8_on_release_model` 的可执行模型拓扑、逐block卷积、跨head共享投影、learned temperature、初始化、参数/state_dict与真实forward，以固定提交的LinearNO**任务专属**源码为真值；不得用论文概述反推并删掉发布细节。
4. `paper_table8_on_release_model` 的训练超参数、loss类型/权重和报告指标，以论文Table 8及正文为真值；论文没说明的实现细节由任务源码补齐，但必须把本地解释标明。若要尝试逐字论文架构，另建 `paper_literal_experimental`，将欠规范处列为UNRESOLVED，且不得声称它应复现Table 1/2。
5. 当前 `hxh5159/Transolver` 主导数据读取、split、训练入口、artifact、checkpoint/resume、可视化和已修复的实用评估集成；若这些与论文/LinearNO发布协议影响可比性，必须建显式profile/protocol，不得静默覆盖。
6. 官方Transolver固定提交只用于核对基线；旧聊天、`LINEARNO/linearno__discuss.md`、`CODEX/`、旧zip和便捷脚本只能作线索。

论文与发布代码不能同时成立时，必须建立冲突台账，并提供至少两个明确命名的 profile：

- `paper_table8_on_release_model`：保留发布模型实现，但采用论文表格／正文明确的训练objective、超参数和指标；所有补充解释必须可追踪；
- `official_release`：忠实采用固定提交中真正生效的模型／训练配置。这个profile明确不包含“运行train还是eval”的控制流flag；六个Standard发布shell的`--eval 1`另保留为`released_eval_exact`命令，而可训练命令显式`eval off`并记录这一处launcher偏差。AirfRANS完整benchmark命令同样显式`debug=0`；逐字默认`debug=1`仅作为`released_cli_default_exact`诊断动作，不能代表完整论文训练；
- 可选 `transolver_matched`：仅用于同流水线公平比较，必须明确标记“不是论文复现配置”。

六个 Standard 任务中，论文主要超参数与发布代码基本可以联合解析，数值复现默认以 `official_release` 为第一基准，再核对 `paper_table8_on_release_model`；两个工业任务存在训练 loss 和指标口径冲突，必须同时报告两套口径，不能期待一个运行同时逐字满足论文表格和发布代码。工业评估不能压成一个含糊字符串，必须使用正交 `evaluation_spec`：`prediction_sampling_protocol`、`field_metric_protocol`、`force_metric_protocol`、`force_input_protocol`、`split_protocol`、`aggregation_protocol`。其中ShapeNet的field rL2口径与drag输入修复是两个独立维度，发布bug兼容只能作诊断。

任何报告、目录、checkpoint 元数据都不得只写含糊的 `official`。必须记录 profile 名称、论文版本、三方 SHA、解析后的完整配置和冲突决策。

## 六个 Standard PDE 的已核验 `official_release` 架构／训练基线

| 任务 | variant | d / L / h / M | FFN ratio | batch | 训练核心设置 | 位置／接口 |
|---|---|---:|---:|---:|---|---|
| Airfoil | `conv_temp` | 128 / 8 / 8 / 64 | 1 | 4 | 500 epoch，AdamW，lr `1e-3`，wd `1e-5`，OneCycle，clip 1 | 221×51，`fun_dim=0`，unified off |
| Darcy | `conv_temp` | 128 / 8 / 8 / 64 | 1 | 4 | 500，AdamW，lr `1e-3`，wd `1e-6`，OneCycle，clip 1；rel-L2 + 0.1 derivative | 85×85，`fun_dim=1`，unified off，downsample 5 |
| Elasticity | `temp` | 128 / 8 / 8 / 64 | 1 | 1 | 500，AdamW，lr `1e-3`，wd `1e-5`，CosineAnnealing，clip 1 | 不规则 972 点，`fun_dim=0` |
| Navier–Stokes | `plain` | 256 / 8 / 8 / 32 | 2 | 2 | 500，AdamW，lr `1e-3`，wd `1e-6`，OneCycle，无显式 clip | 64×64，10→10，unified on，ref 10 |
| Pipe | `conv_temp` | 128 / 8 / 8 / 64 | 1 | 4 | 500，AdamW，lr `1e-3`，wd `1e-5`，OneCycle，clip 1 | 129×129，`fun_dim=0`，unified off |
| Plasticity | `conv` | 128 / 8 / 8 / 64 | 1 | 8 | 500，AdamW，lr `1e-3`，wd `1e-6`，OneCycle，clip 1 | 101×31，T=20，`Time_Input=True`，out 4 |

用于架构回归的官方配置参数量：Airfoil/Pipe `1,765,889`，Darcy `1,766,145`，Elasticity `585,217`，NS `3,377,921`，Plasticity `1,799,428`。若移植版不同，必须先解释并证明原因，不能放宽测试掩盖。

注意这些是 LinearNO 发布代码的有效配置，不是目标仓库原 Transolver launcher 的配置。例如 LinearNO 的 Pipe 是 batch 4、FFN ratio 1，而目标 Transolver launcher 是 batch 8、ratio 2；LinearNO 的 NS 是 ref 10、ratio 2，而目标 Transolver 是 ref 8、ratio 1；LinearNO 的 Darcy unified position 为 off，而目标 Transolver 为 on。不得为了“少改参数”沿用 Transolver 值。

### 复现敏感的执行协议

- Standard任务的rL2是逐样本flatten后计算 `||pred-y||₂/||y||₂`（无epsilon），再对样本取平均；不是把整个测试集拼接后算一次。
- Standard多数OneCycle按 `epochs=args.epochs, steps_per_epoch=len(train_loader)` 构造；Darcy发布代码却给scheduler使用模块级常量`epochs=500`，而训练循环使用`args.epochs`，只有默认500时两者一致。profile必须保存实际scheduler参数与总step，不能只记“OneCycle”。
- Darcy梯度正则采用zero padding与 `dx=1/85`，并将预测边界置零；不得换成`1/84`或其他差分边界。
- Plasticity每个batch先用NumPy随机排列20个时间索引，执行20次forward/backward/optimizer step，但scheduler每个batch只step一次；resume必须恢复该NumPy RNG状态。
- AirfRANS OneCycle的 `total_steps=(len(train_dataset)//batch+1)*epochs`，使用PyTorch默认`final_div_factor=1e4`；ShapeNet使用同一total_steps公式但显式`final_div_factor=1000`，且训练DataLoader `drop_last=True`。短程测试若改变数据量，必须报告resolved总step，不得称与完整官方调度完全等价。
- ShapeNet发布parser把`batch_size/nb_epochs`声明为float，显式CLI会产生float及`200.0`路径问题；目标hxh的整数修复应保留，并作为有记录的工程修复，而非复制bug。

## 两个实用任务的已核验基线

| 任务 | 架构 | 训练／接口关键点 | 已知论文—代码冲突 |
|---|---|---|---|
| AirfRANS | d256/L8/h8/M32，ratio2，Linear 输入／`Linear+Dropout`输出，7 原始特征再拼 8×8 reference-distance，out4；reference grid固定x∈[-2,4]、y∈[-1.5,1.5]，从原始`data.pos`算欧氏距离 | batch1，400 epoch，Adam `1e-3`，OneCycle；每个模拟随机抽 32k 点；验证重复随机抽样 20 次；`forward(Data)->[N,4]` | 发布训练为归一化空间四通道volume MSE + 1×四通道surface MSE；论文Table 8写surrounding/surface physical-field rL2且surface权重0.5，Table 2的Volume/Surface评价聚焦压力；必须分 profile |
| ShapeNet-Car | d256/L8/h8，`key_ratio=1→M32`，ratio2，独立温度，out4 | batch1，200 epoch，Adam `1e-3`，OneCycle；数据有 `param0…param8` 九组可作留一组划分，但发布入口默认只跑 `fold_id=0`，即789 train/100 test；`forward((cfd_data,geom))->[N,4]` | 发布代码默认“所有点velocity normalized-MSE + 0.5×surface pressure normalized-MSE”，论文表格写“surrounding-region velocity rL2 + surface pressure rL2”；必须分 profile，不能只改权重 |

AirfRANS 官方的完整场评估是通过多次随机 32k 子集逐步覆盖并平均重复点预测；由于每次子集会改变全局上下文，不能擅自替换成一次全点前向后仍称为官方复现。ShapeNet-Car 官方温度键拼写错误和两任务的 whole-object checkpoint 都需要显式兼容方案。ShapeNet发布 evaluator 还把volume velocity误传给按surface点读取的drag helper，并硬编码 `param0`；hxh当前已改为surface velocity和真实样本路径。论文 `C_D/rho_D` 应采用并明确标注hxh的intended/fixed协议，发布bug兼容模式只能作诊断，不能声称复现Table 2。

论文结果只作为完整训练后的对照目标，不是单元测试阈值：

| 任务 | LinearNO 论文主结果 |
|---|---:|
| Airfoil relative L2 | 0.0049 |
| Pipe relative L2 | 0.0024 |
| Plasticity relative L2 | 0.0011 |
| Navier–Stokes relative L2 | 0.0699 |
| Darcy relative L2 | 0.0050 |
| Elasticity relative L2 | 0.0050 |

论文工业表中 LinearNO 为：AirfRANS `Volume 0.0011 / Surface 0.0077 / C_L 0.0491 / rho_L 0.9992`；ShapeNet-Car `Volume 0.0194 / Surface 0.0754 / C_D 0.0106 / rho_D 0.9925`。论文把AirfRANS Volume/Surface表述为周围区域压力/表面压力relative L2，但发布评估实际计算normalized-output MSE；因此最终报告必须并列“论文标签”和“代码实际计算”，不能只换标题。论文 Table 9 的 `Parameter (GB)` 与代码参数量一致时实际应按百万参数理解；记录这一疑点，不擅自改论文。

论文 Table 9 在 batch 1 下报告 LinearNO GFLOPs 为 Airfoil `21.34`、Pipe `31.51`、Plasticity `6.03`、NS `15.53`、Darcy `13.68`、Elasticity `0.69`。它们只能在明确相同输入 shape、计数口径与 profiler 后用于对照，不能作为跨工具的逐位单元测试。

---

## 总控提示词：首先发送

```text
你将在我当前 checkout 的 hxh5159/Transolver 仓库上新增“纯 LinearNO”模型族，使它在已有六个 Standard PDE、AirfRANS 和 ShapeNet-Car benchmark 中可训练、可评估，并能按论文 v3 与官方 LinearNO 固定源码的联合规格复现。必须按我逐轮授权的阶段实施。

一、阶段授权与工作边界
1. 本轮以及后续每轮，只执行我明确点名的一个 L 阶段。在阶段范围内可自主完成必要实现、测试和文档，但完成后必须停止，等待我审查。持有完整计划不等于获准提前执行后续阶段。
2. 当前任务只建立忠实的纯 LinearNO 基线。不要实现或接入 CDLNO、CDPA、KCDNO、MSAR-LNO、跨层历史、传播核多样性、浅层替代、额外物理损失或其他新研究机制。不要把旧聊天里的候选设计带进基线。
3. 不下载数据或官方 checkpoint，不启动完整真实训练，不自动 commit/push/rebase/merge/建 PR。每次阶段授权可带三个开关：`RUN_REAL_BATCH`、`RUN_MINIRUN`、`MAX_STEPS`；未写时前两者一律视为false、MAX_STEPS视为0。只有对应阶段、数据可用且开关明确开启时才能消耗真实数据/GPU；完整长训练始终需L10后的单独授权。

二、先完整理解三方仓库与论文
4. 读取适用的 AGENTS.md/CLAUDE.md，但它们只是仓库说明，不替代源码。系统阅读目标仓库、固定提交的官方 Transolver、固定提交的官方 LinearNO 以及论文 v3。必须覆盖所有 tracked 的文本源码、配置、shell、README、测试和相关 notebook code cell；对重复文件可用 blob/hash/AST 分组，但要留下覆盖清单、分组依据和未读理由。不能只读模型文件或 README。
5. 目标仓库与 thuml 官方仓库无共同 Git 祖先。不得 merge/rebase 官方 main，也不得用 merge-base 或三点 diff 推断用户改动。使用固定 SHA 的 tree-to-tree、blob hash、普通 diff 和 AST/语义比较。不要 reset、checkout 覆盖、清理或改写用户工作。
6. 若只读参考仓库不在本机，可在目标仓库之外克隆并 checkout 固定 SHA；保持只读。若论文或源码不可访问，明确列出缺口并停止受影响决策，不能凭记忆补写。

三、真值与冲突规则
7. 不使用单一全局证据优先级。按决策域定真值：方法身份、Q/K归一化与论文指标定义看论文v3；official_release及paper_table8_on_release_model的forward拓扑、逐block卷积、共享投影、temperature、初始化和state_dict看LinearNO任务专属固定源码；paper profile的训练objective/表格超参看论文；数据入口、artifact/checkpoint/resume/可视化与已修复的实用集成看当前hxh。官方Transolver只作基线。任何跨域冲突建立source-of-truth/冲突/决定/影响台账，不得静默混合。真正paper_literal只能建experimental profile，欠规范处标UNRESOLVED，不得宣称应复现Table 1/2。
8. 配置至少区分 paper_table8_on_release_model、official_release；需要公平对照时另建 transolver_matched。official_release只描述模型/训练配置，不把train/eval action flag算入profile；另保留released_eval_exact命令，并为实际训练显式eval off、记录偏差。名称必须进入运行目录、config和checkpoint元数据。工业任务把训练objective与evaluation_spec分别记录，后者至少拆成prediction sampling、field metric、force metric、force input、split和aggregation；不能把ShapeNet的paper field-rL2、hxh surface-velocity drag fix与released bug混成一个字符串。不能用一个含糊的official配置覆盖不同目标。
9. Super-Resolution Appendix-E 需要审阅以防误用其乘数式 key_ratio 和 cross-attention，但不属于当前 Transolver 八任务，不实施。官方 LinearNO 固定树未提供 LICENSE 文件；L0 记录来源与许可风险，生产实现优先依据公式重新实现并保留出处，不整树 vendoring。

四、LinearNO 数学与实现硬约束
10. 每头严格实现 Q=softmax_M(XWq)、K=softmax_N(XWk)、V=XWv、C=K^T V、Y=QC。Q/K 独立；Q沿M、K沿N；无sqrt缩放、无N×N矩阵、无M×M latent self-attention、无attention-weight dropout；Q/K/V无bias。官方实现是在拆head后用同一组Linear(d_h,·)跨head共享参数，不是per-head独立权重。温度只在对应官方变体中使用。
11. 保留 pre-LN attention residual、pre-LN两层GELU FFN residual、最后block的LN+Linear输出。8层就是8个完整block，不能把最后一层当纯head而减少一次attention/FFN。
12. 不得用一套实现近似全部任务。Standard 必须支持 plain/temp/conv/conv_temp；AirfRANS、ShapeNet-Car必须按各自官方文件处理实际M语义、温度、输入/输出投影、位置特征和state_dict。对外可统一叫 linearno_rank/qk_dim，但元数据必须保存实际M及原官方参数如何映射。
13. Standard 的 args.model 在目标仓库是模型注册键，而官方 LinearNO 错把同名字段用于内部 variant。保留旧 --model 语义，新增独立 --linearno_variant（或经审计确认的等价字段）；绝不能让注册名 LinearNO 静默落入 plain 分支。不要复用 --slice_num 隐藏 LinearNO rank。
14. 初始化也属于模型：Linear权重 trunc_normal std=.02、bias 0，LayerNorm weight1/bias0；Standard Conv2d Kaiming normal；placeholder 要保留官方初始化时序。ShapeNet官方 tempreature_q/k 拼写和AirfRANS未使用temperature等参数，若为官方checkpoint/参数量兼容需保留原键或提供显式、可逆、严格验证的转换。
15. 可以修复硬编码 .cuda()、非buffer position 等设备缺陷，但只能做语义保持的工程修复，例如 persistent=False buffer；必须同时证明有效官方配置在相同权重下数值一致、state_dict集合没有意外变化，并在差异台账标记，不得把修复后的模式冒充逐字官方代码。

五、保护当前 hxh 仓库
16. 保留所有现有 Transolver 模型key、类路径、CLI默认值、forward、state_dict、旧checkpoint加载和输出语义。保留用户新增的seed、可视化、checkpoint/resume、实验目录和评估工具。冻结目标仓库中实施前已经存在的整个 `LINEARNO/**`（其中 monitor 是可选的Transolver传播核诊断器，不是LinearNO模型），不得覆盖、改名或把新模型放进这里；新实现放进各 benchmark 的模型目录。L0必须对冻结区及既存Transolver核心模型/attention/embedding文件保存tracked/untracked状态与内容hash，此后都与L0基线比较，不能只看git diff。也冻结现有 `train_and_evaluate/**`、`CODEX/train_and_evaluate/**`、`evaluate/**` 中旧 Transolver 命令的既有语义；为 LinearNO 新增平行 launcher 或不改变旧默认的显式分支。
17. 根 Physics_Attention.py 不是八任务真实入口；不能只改根文件。六个标准题走 PDE-Solving-StandardBenchmark/model_dict.py→module.Model，Car和AirfRANS各有独立wrapper。逐入口接入并测试。
18. 数据文件、split/fold、样本顺序、下采样、归一化、标签通道、损失计算空间、optimizer/scheduler step节奏、NS teacher forcing/自由rollout、Plasticity时间循环、AirfRANS随机子采样和ShapeNet mask语义默认冻结。只有为实现 paper_table8_on_release_model/official_release 明确差异而新增的LinearNO专属profile可以改变对应超参，且不能污染Transolver分支。
19. 不把官方仓库的已知bug直接复制成全局行为，也不静默“修好”后声称exact reproduction。尤其记录：六个LinearNO Standard shell全是eval=1；官方脚本无seed；ShapeNet/AirfRANS whole-object pickle；两实用任务loss权重论文与代码冲突；OneCycle实际步数怪癖；官方日志字段误命名。
20. 当前 hxh 的新训练产物不再等价于旧 ./checkpoints 或 metrics 路径。新增LinearNO必须接入当前artifact/checkpoint框架，并单独验证train→checkpoint→eval round trip；不得照抄旧Evaluation.sh后假定能找到权重。运行产物、数据、checkpoint和大二进制不得提交。

六、环境、checkpoint与安全
21. 保留用户当前可工作的CUDA12.8/PyTorch2.11/PyG cu128环境。先只读记录环境；不要按旧requirements/environment.yml降级或重装torch。分别报告“官方历史环境”和“当前兼容环境”，不能把环境相同当作架构一致，也不能因版本不同跳过可完成的数学验证。
22. 新LinearNO checkpoint schema必须把构造参数与运行协议分开：`model_spec={class_path, constructor_kwargs}`，其中constructor_kwargs只含真实构造器字段；另设`profile_spec`、`data_spec`、`objective_spec`、正交字段组成的`evaluation_spec`与`provenance_spec`。`model_spec`至少覆盖variant、actual M及原始key_ratio语义、space/fun/out dim、d/L/heads/ratio、H/W/ref/grid范围、unified_pos、Time_Input、dropout、activation、projection/kernel/temperature；其他spec保存task/family/profile、split/checksum、训练/评价协议、config/schema version、三方SHA、base_commit/dirty及源码/patch hash。必须保存可数值恢复的`normalizer_spec`：命名后的input/output或任务专属normalizer/coef_norm状态、dtype/shape/hash、fit split及数据checksum，不能在评估时悄悄重拟合。`resume_state`必须含checkpoint_role、selection_split/selection_metric、epoch/global_step、optimizer、scheduler、Python/NumPy/Torch CPU/CUDA RNG，以及显式DataLoader generator/sampler状态（若使用）；ensemble另存member id/order→state_dict/path/hash manifest。评估先读spec、全量校验constructor_kwargs后构造模型，并对LinearNO strict=True加载；不能利用当前通用strict=False吞错键。旧Transolver checkpoint仍按原逻辑兼容。
23. 官方whole-object checkpoint只在用户明确提供且来源可信时加载；对PyTorch2.6+需要weights_only=False的情况使用最小、显式、局部转换工具并说明pickle风险。官方键转换必须列出所有改名、检查shape、拒绝未知/缺失键，转换后strict=True并比较输出。不能用丢键、补随机权重或strict=False伪装兼容。

七、验证与复现声明
24. 官方实现→移植实现一致性不能只测shape。用独立数学oracle和固定官方源码做：参数key/shape/count、逐层和最终forward、输入与每个参数gradient、一次optimizer step、strict checkpoint roundtrip。报告dtype/device/tolerance/max与mean误差。初始门槛建议CPU FP32 `atol=1e-6, rtol=1e-5`、CUDA FP32 `atol=1e-5, rtol=1e-4`；CPU float64 oracle另给更严门槛。若需放宽必须有数值证据并经审查，不能只为让测试通过。
25. 测试必须覆盖B>1不混样本、Q沿M和K沿N归一、四种Standard变体、温度clamp、非方形H/W、N≠H×W报错、fx有/无、unified position、T[B,1]、随机点置换等变性（仅线性点式变体）、无N×N注意力且无“两个轴都代表slice”的M×M self-attention/logits、所有活跃参数有合理梯度。注意合法的 `C=K^T V` 形状是 `[M,d_h]`；当 `M=d_h=32` 时它数值上也是32×32，不能因此误判为slice self-attention。
26. 八任务按“静态/合成通过→真实loader一批通过→短程overfit/mini-run→完整论文运行”分级。没有数据/GPU时继续完成所有能做的层级，并精确写未运行项；不能用随机合成结果声称复现论文精度。
27. 论文结果是实验复现目标，不是单元测试常数。`official_release`和默认paper profile固定评估final epoch/checkpoint，因为发布训练没有test选best协议；任何best checkpoint只能作为另标诊断，必须预先指定validation-only selection_split/metric，绝不可用test选择权重。完整运行记录数据checksum、split、GPU、软件版本、precision、seed列表、完整命令、解析后config、参数量、训练曲线、checkpoint_role/selection规则、原始metric与聚合。论文称每个模型训练3次但未公布seed和mean/std细节；预声明3个本地seed，逐次及mean±std都报告，并注明不是作者seed。不挑最好一次，不因一次未达数值就改模型。

八、每阶段交付格式
28. 每阶段更新 docs/LINEARNO_IMPLEMENTATION_STATUS.md。L0另建REFERENCE_AUDIT和REPRODUCTION_MATRIX，L10另建IMPLEMENTATION_REPORT。保留原文档，不覆盖其他模型报告。
29. 阶段结束必须给出唯一状态 `PASS / PARTIAL / BLOCKED`：PASS表示本阶段所有非资源门控验收通过，缺数据/GPU项可NOT RUN但须给原因和精确命令；PARTIAL表示在已有权限/资源下仍有未完成项或需用户明确豁免；BLOCKED表示真值冲突、实现错误或前置条件使下一阶段不应继续。只有前阶段PASS，或用户明确接受列明的PARTIAL项，才可进入下一阶段。报告还须包含：A范围/状态及判定理由；B实际改动文件与diff摘要；C论文公式/官方符号→代码映射；D实际命令、环境、通过/失败/未运行；E官方parity与旧Transolver回归证据；F冻结区相对L0是否改变；G未决冲突和我最应检查的3—5点。存在未解释的parity失败不得标为PASS。最后明确“本L阶段结束，未执行下一阶段”。
30. 若本阶段发现会改变架构、训练协议或论文复现含义的新冲突，先完成不受影响工作，把原文/文件/行、两个选择和影响写清后停止，等我决定；不要猜。
```

## L0：完整只读审计并建立真值矩阵

```text
现在只执行L0。允许新增或增量更新docs/LINEARNO_REFERENCE_AUDIT.md、docs/LINEARNO_REPRODUCTION_MATRIX.md、docs/LINEARNO_IMPLEMENTATION_STATUS.md和只读清单；禁止修改生产模型、factory、训练/评估、数据、依赖、launcher和已有测试。

1. 读取适用指令，记录目标仓库绝对路径、remote、branch、HEAD、git status、tracked/untracked diff；不reset/clean/stash。核验目标是否仍对应已知66bc489快照；若已变化，保存现状并以实际HEAD为基线。对既存 `LINEARNO/**`、三个benchmark内既存Transolver核心模型/Physics-Attention/Embedding文件建立冻结manifest：保存 `git status --porcelain=v1 --untracked-files=all -- <paths>`、文件清单、tracked/untracked/ignored分类、size与SHA-256。另定义本任务允许生成的ignored临时/运行目录清单；后续只能相对本L0 manifest判断新增变化，不能把用户原有dirty项误算成本任务改动，也不能让untracked文件逃过检查。
2. 在目标外定位或取得三份固定来源：hxh目标、thuml/Transolver@75e0f676、HiPRL/LinearNO@3f2b80d，以及论文v3。记录tree SHA、最后提交、许可证/NOTICE。特别记录LinearNO固定树无LICENSE这一事实及分发影响，不擅自作法律结论。
3. 用rg --files/git ls-files建立三仓覆盖清单。阅读全部tracked文本源码、配置、shell、README、测试和相关notebook code cell；二进制图片、zip、wheel、数据只记录路径/大小/hash，不解压后当当前源码。重复文件按blob/hash/AST分组，清单逐文件标明read/group/excluded/reason。最终报告阅读覆盖率，不能写笼统“已读仓库”。
4. 对hxh与thuml做tree-to-tree清单：相同blob、修改、仅一侧存在；对所有模型类和八任务入口做AST/语义比较。确认用户模型逻辑是否确实未改，并区分可视化/checkpoint/seed增强、数据路径修复及其他功能变化。禁止使用merge-base/三点diff。
5. 建八任务入口表：真实model import/factory、forward签名、输入输出shape/通道、H/W/N、fx/T/Data/tuple语义、数据文件、split/fold、normalizer、loss、optimizer/scheduler及step时机、metric、checkpoint、train→eval路径、import副作用。不要import会立即读数据/parse argv/训练的脚本。
6. 完整审阅LinearNO的Standard、AirfRANS、ShapeNetCar、Appendix-E模型及其训练/评估/配置。把论文每个公式和Table 8字段映射到官方文件、类、参数与实际有效命令。不能把argparse默认、shell值、paper值混为一谈。
7. 建变体表：plain/temp/conv/conv_temp/AirfRANS/ShapeNet/Appendix-E，记录input projection、q/k/v shape、actual M、softmax轴、温度初值/clamp/是否使用、output projection、block顺序、初始化、parameter/state_dict键、checkpoint格式。
8. 建冲突台账，至少核对：Standard六脚本eval=1；--model字段冲突；Standard key_ratio绝对值 vs ShapeNet/Appendix乘数；论文写结构化网格“一个卷积预处理层”而代码在每个attention block内卷积；论文投影表述 vs 代码跨head共享小投影；Darcy/NS/Pipe等与Transolver配置不同；AirfRANS与ShapeNet不仅loss权重相反，objective类型（rL2/MSE）、region/mask、通道、归一化空间和reduction也不同或未充分说明；industrial指标relative-L2标签 vs 实际MSE；ShapeNet默认fold0 vs九组可选留出；官方ShapeNet drag evaluator的volume/surface参数错误和param0硬编码 vs hxh修复；debug/preprocessed默认；whole-object checkpoint；位置特征替换vs拼接；官方无seed；OneCycle步数与日志命名问题。
9. 建三profile矩阵：paper_table8_on_release_model、official_release、可选transolver_matched；另把released_eval_exact与AirfRANS released_cli_default_exact列为诊断动作而非训练profile。每个任务写resolved model config、训练config、loss、完整正交evaluation_spec、数据协议、来源证据和未决项。paper/code冲突保持两行，不自行合并。
10. 只读检查当前Python/torch/CUDA/PyG/timm/einops等版本和GPU可用性；对照两官方环境文件，但不安装。列出哪些parity/集成测试当前可执行。
11. 给出L1—L10最小文件变更预测、共享原语与任务wrapper边界、旧模型冻结manifest、允许的ignored产物清单、官方checkpoint兼容策略和主要风险。若阶段结构需按实际代码微调，先提议，不提前实现。

REFERENCE_AUDIT必须附关键命令及其实际结果摘要；REPRODUCTION_MATRIX必须能从“论文字段→官方代码→目标参数→启动命令/元数据”逐列追踪。完成后停止。
```

## L1：建立修改前回归依据与配置／checkpoint协议

```text
L0已审查通过。现在只执行L1：为旧Transolver和用户增强建立修改前回归依据，并为新LinearNO定义独立配置/profile/checkpoint元数据协议；暂不实现LinearNO算子，不接生产训练。L1只允许新增测试、独立的schema/profile解析模块和文档；不得修改现有八任务parser、factory、训练、评估或checkpoint helper。生产接线分别在L4—L7完成。

1. 按L0确定的真实位置建立tests/linearno（或仓库已有等价位置）。固定seed、eval模式和合成输入，保存/计算六Standard、AirfRANS、ShapeNet现有Transolver的factory选择、forward输出摘要、state_dict key/shape、参数量、checkpoint roundtrip。测试应比较同一份权重与输入，而非两个随机实例。
2. 回归夹具尽量用可重算的seed+hash/小数值fixture，不提交大模型二进制。记录生成代码版本与容差。覆盖hxh当前artifact/checkpoint、resume和eval路径中能无数据验证的部分。
3. 建立“仪表化不扰动随机流”回归：同seed比较启/停可视化与保存；比较不中断K步与K1步保存→新进程resume→K2步的batch顺序、Plasticity NumPy时间排列、AirfRANS Python random子采样、Torch DataLoader shuffle、最终权重与scheduler/optimizer状态。若底层非确定性只能容差比较，必须先证明确切来源。
4. 只在独立schema/profile模块中定义linearno family、稳定命名、独立linearno_variant与actual linearno_rank/qk_dim语义，以及hidden%heads=0、M>0、结构化N=H×W等验证规则；L1不得把字段接入任何现有argparse/factory。生产CLI解析/转发只在对应L4—L7接线，旧--model始终不变。
5. 只在独立模块中把paper_table8_on_release_model、official_release、transolver_matched设计成显式、可序列化配置并测试解析规则；L1不得改现有parser。未来接线的优先级为显式CLI>所选profile>新family默认，LinearNO override参数须用 `default=None`、`argparse.SUPPRESS` 或等价argv来源追踪，先解析profile，再只应用用户显式值；resolved config逐字段记录 `cli_explicit / profile / family_default / legacy_default`。不得让旧parser的slice_num/mlp/batch默认静默覆盖LinearNO profile。
6. 在独立模块定义checkpoint schema：`model_spec={class_path, constructor_kwargs}`（constructor_kwargs只放真实构造器字段），以及分离的`profile_spec/data_spec/objective_spec/evaluation_spec/provenance_spec/normalizer_spec/resume_state`。normalizer_spec保存命名数值状态/coef_norm、dtype/shape/hash、fit split与数据checksum；resume_state保存checkpoint_role、selection_split/selection_metric、epoch/global_step、optimizer/scheduler、Python/NumPy/Torch CPU/CUDA RNG及显式DataLoader generator/sampler状态；ensemble manifest保存member id/order→state_dict/path/hash。schema/config version与hash、三方SHA、paper版本、base_commit、dirty、LinearNO源码/规范化patch hash必须齐全。结构字段不一致严格拒绝，运行字段变化单独报告；旧checkpoint缺新字段仍按旧路径识别，不能猜成LinearNO。
7. 定义官方checkpoint转换清单，但本阶段不加载外部pickle：裸state_dict、ShapeNet tempreature键、AirfRANS dead parameter、whole-object安全策略。任何转换后都必须strict=True。
8. 增加配置解析、非法配置、CLI隔离、metadata往返和旧CLI无变化的测试。不要放占位假模型让factory“成功”。
9. 运行L1夹具，确认仅新增测试/配置基础不改变旧模型输出、state_dict、随机流与已有monitor测试。更新STATUS后停止。
```

## L2：实现并验证 LinearNO 数学原语与六种官方变体

```text
L1已审查通过。现在只执行L2：实现纯LinearNO attention原语和Standard四变体、AirfRANS变体、ShapeNet变体的独立模块/包装，不组装八层完整任务模型，不接训练入口。

1. 先写独立的显式数学oracle，不能调用待测forward。oracle严格执行Q沿M、K沿N、K^T V再Q乘；温度、clamp和无scale按变体处理。
2. 实现Standard plain/temp/conv/conv_temp。conv仅有一套Conv2d输入投影到inner_dim，再按head切分；不要错误保留Transolver的x/fx两套卷积、slice normalization或slice self-attention。Q/K/V小投影必须按官方跨head共享。按官方区别实现单层或`Linear-GELU-Linear` output projection，并保留末尾Dropout模块，即使论文配置p=0。
3. 实现AirfRANS语义：absolute M=32、Linear input、`Linear+Dropout` output；官方dead temperature若为参数量/state_dict兼容需保留，但forward绝不能误用。实现ShapeNet语义：M=key_ratio*head_dim、独立温度、clamp[.1,2]、`Linear-GELU-Linear-Dropout` output，并处理官方拼写键。
4. 所有投影、bias、dropout、初始化、shape检查按总控。不要在生产forward返回完整Q/K/C，调试统计通过显式可关闭hook或测试接口，默认不保留计算图/CPU同步。
5. 对固定官方源码中的对应类建立同权重parity：严格对齐parameter key/shape/count；随机固定输入比较forward、input grad、每个参数grad和一次optimizer step。若涉及dropout或随机采样，两个实现每次forward前恢复完全相同的RNG状态，避免假失败。若类名/键不同，用显式映射并测试双向覆盖，不得strict=False。
6. 覆盖CPU float64 oracle、CPU float32；有CUDA再测CUDA float32/AMP兼容。报告每类atol/rtol、max/mean abs和relative error，不只写allclose。
7. 结构测试：Q.sum(M)=1、K.sum(N)=1；小N下factorized结果与显式`(Q@K^T)@V`一致；B>1样本隔离；Linear变体点置换等变；conv非方形H/W顺序；N错误明确报错；温度边界；无N×N注意力且无slice×slice self-attention/logits（不得把合法 `[M,d_h]` context在M恰等于d_h时误判）；Q/K参数对象独立但同类投影跨head共享；所有使用参数有finite grad，AirfRANS已知dead参数单独标记。
8. 跑L1全部旧Transolver回归。L2不改factory/exp/train。更新公式→类→测试映射和STATUS后停止。
```

## L3：组装完整 Standard LinearNO 模型并完成全模型 parity

```text
L2已审查通过。现在只执行L3：组装六个Standard任务共用但可配置的完整LinearNO Model；仍不修改六个exp或生产launcher。

1. 按官方Standard模型保留preprocess、placeholder、可选time embedding、n_layers个完整block和最后block输出。forward合同为Model(x,fx,T=None)->[B,N,out_dim]。
2. ordinary position与unified position语义必须逐任务一致。若修复官方构造时.cuda()，使用不进入state_dict的device-safe实现并证明相同位置数值；不能从“替换x”改成“拼接x”。
3. 初始化顺序逐项一致：先构造preprocess/time_fc/全部blocks（因此先消耗框架默认初始化RNG），再对完整已注册模块恰好一次 `self.apply(_init_weights)` 重置Linear/LN/Conv；temperature参数不受该apply影响；最后才创建placeholder，使其保持官方 `(1/d)*rand(d)` 且不被apply触及。不要因L2原语已有局部初始化而跳过或重复wrapper级官方apply；用同seed参数逐值/初始化RNG测试验证。
4. variant从独立字段获得，绝不读取架构注册用args.model决定。模块必须导出目标factory需要的Model类；稳定路径和命名在报告中列明。
5. 对官方六组有效配置至少做构造、参数量、state_dict key/shape断言；参数量应匹配已核验数值。再用缩小模型做B=1/2、fx有/无、T有/无、unified on/off、非方形网格前反向。
6. 在可运行环境中，把官方Standard完整模型的权重严格映射到移植模型，比较逐block输出、最终输出、loss、全梯度与一次optimizer step。官方硬编码设备使某项不能CPU执行时，如实移到CUDA或用等价reference，不伪造完成。
7. 证明最后第8层确实执行attention+FFN+LN/head；证明没有Transolver的slice-token M×M attention残留；检查正常forward峰值张量不含N×N。
8. 跑L1/L2及既有 `LINEARNO/**` 相关测试，确认旧模型和诊断器不变；用L0冻结manifest复算既有LINEARNO与Transolver核心文件的状态/内容hash，必须无本任务新增变化，并检查untracked与允许的ignored产物。更新STATUS后停止。
```

## L4：接入四个静态 Standard 任务

```text
L3已审查通过。现在只接入Airfoil、Darcy、Elasticity、Pipe；NS、Plasticity、AirfRANS、ShapeNet留待后续。

1. 在真实model_dict/factory新增LinearNO结构化/不规则key，旧key与默认保持。只在LinearNO分支传variant/rank等新字段。
2. 对四个exp做最小参数接入；数据加载、split、归一化、decode、loss、scheduler cadence、可视化和用户artifact/checkpoint逻辑保持。LinearNO特有profile显式覆盖其论文/官方配置，不改变Transolver分支。
3. 任务映射必须是：Airfoil conv_temp/M64/H221/W51；Darcy conv_temp/M64/H85/W85/unified off并保留0.1导数项、zero-padding差分、dx=1/85与预测边界置零；Elasticity temp/M64不规则；Pipe conv_temp/M64/ratio1/batch4/H129/W129。Standard rL2严格按逐样本flatten无epsilon后平均实现。OneCycle保存resolved epochs/steps_per_epoch/total_steps；特别测试Darcy发布scheduler固定500而训练循环读取args.epochs的差异，official_release默认500可一致，非500必须显式报偏差或拒绝，不能静默错步数。
4. 为每题提供明确分开的train与eval launcher/命令，且launcher显式传有效factory key；目标六个exp的历史默认Transolver_1D/2D并不存在于model_dict，不能依赖默认值。不能复制官方eval=1脚本充当训练；不能覆盖Transolver save_name。使用hxh当前experiment_dir/checkpoint约定，让eval读取同一运行的metadata与权重。
5. 配置和checkpoint记录resolved variant/actual M/profile。LinearNO路径strict=True；旧通用strict=False行为只保留给原模型兼容，不得用于新模型。
6. 始终用任务真实shape/接口的合成输入测试；只有本次授权同时写明 `RUN_REAL_BATCH=true` 且数据/GPU可用时，才运行loader一批前反向和eval，并受 `MAX_STEPS` 限制，绝不启动完整500 epoch。Darcy额外验证decode后边界/导数loss链，Elasticity验证fx=None，Airfoil验证221×51的非方形H/W交换与点序，Pipe验证129×129的官方flatten/reshape点序（不能把Pipe误称非方形）。
7. 本阶段必须完成四题各自的hxh原生train→checkpoint→resume→新进程eval闭环：评估先读metadata再构造正确family，LinearNO逐键strict=True，同config输出一致，错误profile/variant/M拒绝。不得把此闭环推迟到L8。运行所有旧Transolver回归和现有实验工具静态检查。
8. 逐段/AST证明冻结训练数据区域未变化，报告四题paper_table8_on_release_model与official_release命令及仍未运行的完整实验。停止。
```

## L5：接入 Navier–Stokes 与 Plasticity

```text
L4已审查通过。现在只接入NS与Plasticity，复用已验证Standard模型，不修改两个时间任务的语义。

1. NS使用plain无温度、d256/L8/h8/M32、ratio2、unified on/ref10、输入10帧每次输出1帧。保留训练10步teacher forcing、累加后一次backward/optimizer和测试10步预测回填；不要跨真实时间缓存latent。
2. Plasticity使用conv无温度、d128/L8/h8/M64、ratio1、H101/W31、Time_Input=True、out4。保持field/token与现有Conv2d reshape的历史点序，即使meshgrid看似相反也不能顺手修。保留每batch由NumPy RNG随机排列20个时间查询、各自forward/backward/optimizer共20次，但scheduler只step一次的原节奏；checkpoint/resume恢复NumPy状态。
3. 新参数、profile、checkpoint、train/eval launcher按L4规则接入；不让Transolver的NS ref8/ratio1或其他旧默认覆盖LinearNO。
4. 合成测试覆盖T形状[B,1]、B>1、不同T影响及time_fc梯度；NS真值/预测回填两条路径、窗口shape、10步loss；Plasticity 20次调用与optimizer/scheduler计数。
5. 真实数据测试仅在 `RUN_REAL_BATCH=true` 时做一批；短步仅在 `RUN_MINIRUN=true` 时做且不超过 `MAX_STEPS`，否则用合成闭环并标记NOT RUN。本阶段完成两题hxh原生train→checkpoint→resume→独立新进程eval，先读metadata、strict=True，并验证参数量NS 3,377,921与Plasticity 1,799,428；不得推迟到L8。
6. 跑六Standard LinearNO测试、全部旧Transolver回归和monitor测试；更新STATUS后停止。
```

## L6：接入 AirfRANS 的任务专属 LinearNO

```text
L5已审查通过。现在只接入AirfRANS，不修改ShapeNet或Standard。

1. 在Airfoil-Design-AirfRANS的稳定模块路径实现/接入官方任务专属模型：input7，unified reference-distance 64维是拼接，preprocess宽71；reference grid严格为 `x=linspace(-2,4,8)`、`y=linspace(-1.5,1.5,8)`，从官方数据中二维的原始`data.pos`计算欧氏距离并按官方flatten顺序拼接（先断言末维为2，不静默切片/换坐标）；d256/L8/h8/M32/ratio2/out4；Linear input、`Linear+Dropout` output；forward(Data)->[N,4]。用边界/中心固定坐标做64维距离golden test，禁止复用Standard的[0,1]^2网格。
2. 保留官方state_dict所需的未使用temperature等成员；测试它们确实不影响forward并明确无grad。不要错误套Standard conv_temp或ShapeNet温度。
3. 在当前main/params.yaml/main_evaluation及hxh artifact框架增加独立LinearNO选择。旧Transolver和其他模型选择不变；debug必须在复现命令中显式0。OneCycle按发布式 `total_steps=(len(train_dataset)//batch+1)*epochs`，保持默认`final_div_factor=1e4`并把resolved total_steps写入配置。
4. 建两个明确profile，不能只改surface权重：`official_release`逐字保留发布代码的归一化输出空间、volume/surface掩码、四通道MSE reduction与surface权重1；`paper_table8_on_release_model`按论文把 `L_v/L_s` 实现为逐样本relative L2，分别作用于surrounding region和surface，surface权重0.5，并把训练通道、mask、inverse-normalization时机与batch聚合写进resolved config；其Table 2评价的Volume/Surface按论文口径计算周围区域压力/表面压力rL2。rL2默认严格无epsilon并断言分母非零；若数据迫使加入epsilon，必须另命名本地偏差profile。论文未明确训练rL2究竟聚合哪些物理通道及归一化时机，必须遵循L0冲突台账中经审查的显式决定；若L0漏记则停止并返修L0，不能悄悄沿用normalized MSE。两profile共享模型权重shape但训练/评价配置不同，目录/metadata不能互相覆盖；正式报告同时输出代码原指标与论文口径的额外rL2，不得把MSE改名为rL2。
5. 为AirfRANS写完整正交evaluation_spec：prediction sampling保留每次随机32k子集、迭代覆盖全场并平均重复点；split为完整manifest test集；field metric分别登记发布normalized-MSE与论文physical-rL2，Table 2的Volume/Surface都只取输出 `[vx,vy,p,nut]` 的压力通道index 2及对应volume/surface mask；force metric按源码逐样本计算 `abs((C_true-C_pred)/C_true)` 后对完整test集平均，并在完整test集上算C_L Spearman，明确成员/ensemble/seed的聚合轴。`Results_test(..., n_test=3)` 的n_test只随机选择3个样本保存VTK/展示，绝不能限制正式指标样本数。保留最后10%验证、每模拟32k训练采样和validation重复20次。不要把正式eval改成一次全点前向；可另有明确标记的smoke sampling，不能用于论文指标。
6. 对固定官方AirfRANS完整模型与移植完整模型做key/shape/count、同权重逐block/final forward、input/parameter gradient和一次optimizer-step parity，覆盖preprocess、placeholder、8 blocks、输出head和reference grid。静态期望参数量为3,358,788（含dead temperature与placeholder），必须在实施环境实例化复核；不符先停止解释。再用真实PyG Data合成对象测试输入不原地修改、输出shape、mask/损失、backward、单图可变N。batch>1若当前语义会混图则明确拒绝，不静默拼成一个物理场。
7. 本阶段完整验证hxh原生train→每成员checkpoint→resume→新进程eval闭环。当前hxh还会产生 `ensemble_full.pth` whole-object与 `ensemble_state_dict.pth` 状态字典列表，且 `main_evaluation.py` 原先硬编码Transolver：新LinearNO评估必须优先从config/metadata加 `ensemble_state_dict` 或成员checkpoint重建，先读family和model kwargs再构造模型，每成员strict=True；验证 `nmodel>1` 的成员数量、顺序与metadata一致。保留旧Transolver whole-object路径；LinearNO缺metadata时必须报错，不得默认回退Transolver。若提供官方可信checkpoint，其外部转换另按L8验证。
8. 跑旧AirfRANS Transolver回归与此前全部测试，验证旧whole-object及可视化/产物目录语义不变。更新STATUS后停止。
```

## L7：接入 ShapeNet-Car 的任务专属 LinearNO

```text
L6已审查通过。现在只接入ShapeNet-Car。

1. 在Car-Design-ShapeNetCar稳定模块路径实现/接入官方任务专属模型：输入7=[xyz,sdf,normal]，d256/L8/h8/head_dim32，key_ratio1→actual M32，ratio2，Q/K独立温度0.5 clamp[.1,2]，两层GELU output，输出[velocity3,pressure1]；forward((cfd_data,geom))->[N,4]，geom按官方仍不参与模型。
2. 保留或显式映射官方tempreature_q/tempreature_k拼写，确保官方state_dict可严格转换。不得套用Standard absolute key_ratio=1导致M=1。
3. 在main、当前checkpoint/evaluate_checkpoint/main_evaluation和启动脚本中增加架构感知模型选择。评估必须先从metadata识别family/profile构造对应类；不再为新LinearNO硬编码Transolver，同时旧Transolver checkpoint仍可用。OneCycle按发布式 `total_steps=(len(train_dataset)//batch+1)*epochs` 且 `final_div_factor=1000`，训练DataLoader保持`drop_last=True`；hxh已修正的整数batch/epoch语义保留并记录，不复制官方float parser/path bug。
4. 建两个完整objective profile，不能只改权重：`official_release`保持发布代码“所有点三通道velocity normalized-MSE + 0.5×surface pressure normalized-MSE”的mask、通道与reduction；`paper_table8_on_release_model`按论文实现“surrounding-region velocity relative L2 + 1×surface pressure relative L2”，明确inverse-normalization与逐样本/跨样本聚合。rL2默认严格无epsilon并断言分母非零；需要epsilon只能另建本地偏差profile。保持数据中 `param0…param8` 的可选留一组机制，但默认复现固定 `fold_id=0`（789 train/100 test），不得宣称论文做了9-fold平均。evaluation_spec分开记录：full-point prediction sampling；field metric的paper physical-rL2或release normalized-MSE；force metric的Cd relative error与跨完整test集Spearman；force input的`hxh_fixed_surface_velocity+real_sample_path`或只作诊断的`released_bug_volume_velocity+param0`；fold0 split及sample/member/seed aggregation。不要用 `hxh_fixed/paper_metric` 斜杠别名绑死可正交组合，也不要纠正与核心无关的官方日志变量反名而改变现有结果。
5. 对固定官方ShapeNet完整模型与移植完整模型做key/shape/count、同权重逐block/final forward、input/parameter gradient和一次optimizer-step parity，覆盖preprocess、placeholder、8 blocks、温度与输出head。静态期望参数量为3,852,420（含temperature与placeholder），必须在实施环境实例化复核；不符先停止解释。再用真实PyG Data/Batch合成接口验证surf mask、可变N、输入不变、backward、strict state_dict和whole-model/当前checkpoint策略。batch1为正式合同；多图如不支持应清楚报错。
6. 验证preprocessed配置与数据存在性错误信息；不能生成同名假数据冒充预处理完成。本阶段完成hxh原生train→checkpoint→resume→新进程eval，metadata先于模型构造、LinearNO strict=True，且Cd评估入口不会误构造模型。对fixed force_input协议验证drag helper收到surface点速度、真实fold/sample路径和匹配长度；若保留released-bug force_input，隔离测试并加显著警告。metadata缺失时只有明确识别为旧Transolver checkpoint才允许legacy fallback；不得把未知或LinearNO权重默认当Transolver。不得把闭环推迟到L8。
7. 跑旧ShapeNet Transolver回归、八任务LinearNO静态/合成矩阵和此前测试。更新STATUS后停止。
```

## L8：官方外部 checkpoint 转换与跨任务复现配置汇总

```text
L7已审查通过。现在只完善官方外部checkpoint兼容、跨任务统一验证、三profile汇总与复现实验清单，不进行完整长训练。L4—L7已经分别完成hxh原生checkpoint/resume/eval闭环；不得把遗漏的任务接线推迟到本阶段。

1. 完成官方Standard裸state_dict、ShapeNet whole-object、AirfRANS whole-object/list的显式导入方案。AirfRANS与ShapeNet pickle可能记录相同限定名 `models.LinearAttnNeuralOperator.LinearAttentionNeuralOperator`、实际类却不同；导入CLI必须先显式指定source_task，不得靠反序列化后猜任务。只在对应固定官方checkout、正确cwd/sys.path的隔离子进程加载用户明确提供且可信的pickle，子进程只导出state_dict+metadata再退出；不得让两任务模块共存污染`sys.modules`。没有可信文件时只测试本地构造的等价格式，不下载、不执行未知pickle。
2. 转换器必须先识别来源/任务/官方配置，再生成完整key映射；ShapeNet拼写、可能的module.前缀、dead参数逐项处理。拒绝未知键、重复映射、shape不一致和缺键；最终strict=True，并比较原/新模型固定输入逐层输出。
3. 跨八任务复核已完成的新hxh checkpoint均能从model/profile/data/objective/evaluation/normalizer/provenance spec恢复模型与数值预处理，并从resume_state恢复optimizer、scheduler、epoch/global_step和全部RNG；metadata先于构造读取。核对base_commit/dirty/source或patch hash/config-schema version及Air ensemble manifest。发现L4—L7漏接时标为BLOCKED并退回对应阶段，不在L8临时另造路径。旧Transolver加载行为不变。
4. 将八任务paper_table8_on_release_model和official_release形成机器可读且人可读的resolved config/launcher。训练脚本必须eval off，评估脚本显式指定experiment_dir/checkpoint；另保留released_eval_exact及AirfRANS released_cli_default_exact命令作来源审计。每个运行目录含task/family/profile/evaluation-spec摘要或hash/variant/M/seed，完整evaluation_spec写入config/metadata。
5. 明确seed策略：官方发布代码无seed，不能编造作者seed。为可复现实验提供预声明seed列表，并在报告中称为本复现实验seed，不称官方seed。保留可选的官方随机行为profile说明。
6. 对每个任务统一重跑无数据的config→model→checkpoint→reload→eval闭环；真实数据/GPU一批只在RUN_REAL_BATCH=true时执行，短程只在RUN_MINIRUN=true且不超过MAX_STEPS时执行。错误family/profile/metadata必须fail fast。AirfRANS额外复核ensemble成员数量/顺序和每成员strict load；ShapeNet/AirfRANS的LinearNO缺metadata不得回退Transolver。
7. 生成数据准备/文件名/checksum脚本或说明，只读验证已有数据，不下载、不改数据。记录split、normalizer和metric聚合。
8. 更新REPRODUCTION_MATRIX，给出完整训练、resume、eval命令及预期产物；更新STATUS后停止。
```

## L9：综合验证、短程实验与效率检查

```text
L8已审查通过。现在只做完整静态/合成验证；真实一批仅在 `RUN_REAL_BATCH=true`，短程实验仅在 `RUN_MINIRUN=true` 且不超过 `MAX_STEPS` 时执行。开关缺省为false/0，仍不自动跑八任务完整论文训练。L9只允许执行现有测试/命令、生成运行产物并更新STATUS/报告；不得修改生产代码、测试或配置。任何失败标记并退回对应L2—L8等待另行返修授权。

1. 运行全部单元、parity、旧Transolver回归、monitor、factory/config/checkpoint测试，输出按任务/设备/dtype的矩阵。失败不能通过放宽strict、删测试或换配置隐藏。
2. 对开关授权的任务执行真实loader一批forward/backward/eval；在另有MINIRUN授权时再做小子集短程overfit或数个step，检查loss有限且总体下降、scheduler/optimizer计数、checkpoint恢复连续。未授权或没有资源的项目明确NOT RUN。
3. 检查参数量、理论复杂度和实际峰值内存/latency。profile必须区分warmup、同步、batch/N/dtype/device；不得仅凭O(NM)声称比Transolver实际更快。
4. 用profiler/语义化shape hook确认LinearNO正常路径不物化N×N attention，也不产生两个轴都代表slice的M×M self-attention/logits；不得把合法 `[M,d_h]` context在数值维度相等时误报。AirfRANS/ShapeNet大N smoke不能长期保存Q/K诊断张量。
5. 检查旧Transolver相同命令、模型选择、输出、state_dict和checkpoint仍通过L1基线；对用户可视化做产物级回归，确保旧目录结构、JSON字段、图像生成和resume后的epoch编号不变。LinearNO不支持的内部诊断应显式skip而非使训练失败。
6. 对L0冻结manifest复算tracked/untracked/ignored状态与内容hash，既有LINEARNO和Transolver核心文件必须无本任务新增变化；检查 `git status --porcelain=v1 --untracked-files=all -- LINEARNO/` 并与L0基线比较，ignored运行产物只能位于预先允许目录。安装既有PropagationMonitor后跑LinearNO forward，断言其发现的Physics-Attention层数为0；monitor开/关前后在相同RNG状态下LinearNO输出、梯度和RNG演化一致，测试产物只写临时目录，绝不写 `LINEARNO/monitor/output`。
7. 对paper_table8_on_release_model与official_release做最终配置diff，列出任何仍会使论文数值不可比的条件。给出完整训练所需时间/磁盘估算与建议顺序，但不启动。
8. 更新STATUS和测试报告后停止。
```

## L10：独立终审与最终交付

```text
L9已审查通过。现在只执行L10：做独立只读终审并交付可复现说明；不修改生产代码、测试或配置，不启动完整训练，不添加新特性。若发现问题，标记BLOCKED/PARTIAL并列入“本阶段返修”清单，等待另行授权。

1. 从论文公式和固定官方源码反向审查目标代码，而不是沿实现自证。逐项检查softmax轴、实际M、温度、input/output投影、block数、初始化、位置语义、任务wrapper和loss/profile。
2. 重跑官方→移植forward/gradient/optimizer parity、参数量、strict checkpoint、八任务接口、旧Transolver回归。随机抽查至少一个Standard conv_temp、一个plain、AirfRANS和ShapeNet逐层输出。
3. 审查git diff与tracked文件：确认任务diff只含必要源码、测试、配置、脚本和文档；不得包含数据、checkpoint、output、图像缓存、wheel/zip副本或临时clone。L10不执行清理；若以后授权清理，也只允许删除本任务明确创建且已列清单的临时文件，绝不碰用户原有tracked/untracked文件。
4. 审查所有训练/评估命令能从正确工作目录解析，train/eval路径相接，paper_table8_on_release_model/official_release不互相覆盖，checkpoint metadata足以在新进程重建模型。
5. 生成docs/LINEARNO_IMPLEMENTATION_REPORT.md，至少包括：三方版本；源码覆盖；公式→实现映射；变体/任务配置；论文—代码冲突决策；数据协议；训练/eval命令；checkpoint转换；实际测试结果；未运行的GPU/真实训练；论文目标指标记录模板。
6. 将各阶段最终状态统一标为PASS/PARTIAL/BLOCKED，并在测试矩阵里另标每项passed/failed/not-run；不能把未跑长训练写成复现成功。最终报告明确区分“实现一致性已证实”“真实一批已通过”“完整论文数值已/未复现”。
7. 不commit/push。报告用户最应人工审查的5项和下一步完整训练顺序后停止。
```

## 阶段间通用提示词

### L10 后另行授权完整论文复现

```text
L0—L10已经审查通过。现在只运行我明确列出的完整复现实验，不再改模型或协议：
- tasks: <明确任务列表>
- profile: <official_release 或 paper_table8_on_release_model；不得写含糊official>
- evaluation_specs: <逐任务指定prediction_sampling、field_metric、force_metric、force_input、split、aggregation；Standard不适用字段写N/A>
- seeds: <预先声明的三个本地seed>
- data roots: <已有数据路径>
- devices: <GPU列表>

开始前先做只读preflight：核对git SHA/dirty diff、数据文件与checksum、resolved config、checkpoint/output空间和每个任务上一阶段smoke证据。任何配置与REPRODUCTION_MATRIX不同先停止报告，不自行修模型。

按任务逐个训练、保存每个seed的原始曲线和final-epoch checkpoint，再用独立评估命令加载同一experiment_dir。除非本次预先明确授权validation-only诊断选择器，否则不生成/选用best；任何情况下都不得用test选权重。不得查看测试结果后调参、重跑挑最好seed或改变split/metric。报告每次原始结果、mean±std、相对论文目标的绝对/相对差值、参数量、FLOPs、峰值显存、耗时和失败/恢复记录。若某次中断，使用已验证resume，不覆盖已有run。完成所列任务后停止。
```

### 接受并进入下一阶段

```text
我已审查并接受 `L_PREV` 的报告与diff。现在只执行 `L_NEXT`：
- RUN_REAL_BATCH: false
- RUN_MINIRUN: false
- MAX_STEPS: 0

继续遵守总控与已确认冲突决策；只有我在本次明确改写上述开关时才运行对应真实/GPU工作。完成后停止，不执行再下一阶段。
```

### 本阶段返修

```text
不要进入下一阶段。仍只在L__范围内修复以下问题：
1. ...
2. ...

修复后重跑本阶段相关测试和所有受影响的旧Transolver回归，更新STATUS，报告新增diff与证据后停止。

本模板不得直接用于L10。L10发现的问题须回退到对应实施阶段另行授权，或由我明确发出post-L10修复授权；修复完成后必须重新执行L10只读终审。
```

### 独立只读数学／官方一致性审查

```text
不要新增功能，也不要进入下一阶段。对当前L__成果做一次从论文v3和HiPRL/LinearNO@3f2b80d反向出发的独立只读审查：重点检查Q/K softmax轴、actual M、温度、投影、初始化、最后block、任务专属差异、state_dict和profile来源。列出确切文件/符号、证据、严重级别与最小修复建议。除审查文档外不改代码。
```

### 缺真实数据或 GPU 时继续完成当前阶段

```text
当前缺少真实数据和/或GPU。不要下载、造同名假数据，也不要把合成测试写成真实复现。请继续完成本阶段全部源码审计、CPU数学oracle、合成前反向、配置、strict checkpoint、旧模型回归和静态检查；把只能在真实数据/GPU完成的项目列为NOT RUN，并给出以后可直接执行的精确命令。完成当前阶段后停止。
```

### 会话恢复

```text
先恢复上下文，不修改代码：读取本提示词、docs/LINEARNO_REFERENCE_AUDIT.md、docs/LINEARNO_REPRODUCTION_MATRIX.md、docs/LINEARNO_IMPLEMENTATION_STATUS.md和当前git status/diff，核对当前三方SHA、已完成阶段、未决冲突与冻结范围。简要复述当前状态并等待我明确授权某个L阶段；不要自动继续。
```

## 用户审查时最值得看的内容

1. `--model` 是否仍只表示架构注册，而内部变体使用独立字段；这是最容易让四个 Standard 任务悄悄变成 plain LinearNO 的错误。
2. Standard、AirfRANS、ShapeNet 是否真的是三套任务专属兼容实现，而不是只复制了某一份；尤其检查实际 M、温度和输出投影。
3. parity 是否包括同权重逐层输出、梯度和一次 optimizer step，checkpoint 是否严格加载；只有 shape 对并不足够。
4. Pipe、NS、Darcy 是否使用 LinearNO 的真实配置而不是沿用 Transolver launcher；两实用任务是否把 paper_table8_on_release_model 与 official_release 的目标类型、区域/mask、通道、归一化空间、reduction和权重全部分开，而不只是换一个loss权重。
5. 现有 Transolver、可视化、resume、train→eval、`LINEARNO/monitor` 是否有修改前后回归证据；是否无意提交了output、checkpoint或数据。
6. 最终“复现”声明是否分清实现一致性、真实一批、短程训练和完整多seed论文数值，是否避免用一次最佳结果代替mean±std。
