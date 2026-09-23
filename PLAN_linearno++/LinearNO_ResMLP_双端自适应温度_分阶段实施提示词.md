# LinearNO：Looped ResidualMLP + 双端自适应温度

## 交给 Codex 的分阶段实施提示词

> 目标仓库：<https://github.com/hxh5159/CDLNO-w.git>  
> 本文只规定工程实施流程。论文模型名可以以后再定；工程公开 selector 固定为 `architecture=resmlp_dual_temp_v4`，内部包可使用 `linearno_loop/v4`。

---

## 一、使用方法

1. 先把“总控提示词”完整发送给 Codex。
2. 接着只发送“阶段 0”。Codex 完成并报告后，先人工审查。
3. 审查通过后，再逐次发送阶段 1、2……7。不要一次发送全部阶段。
4. 某阶段有问题时，使用文末的“阶段返修提示词”；不要让 Codex带着已知问题进入下一阶段。
5. 每个阶段的后续文字只用于交代全局目标，不构成提前执行授权。

本次划分为 8 个阶段，依据是模型依赖关系与可审查边界，而不是照搬参考附件的阶段数：

| 阶段 | 主要工作 | 核心验收 |
|---:|---|---|
| 0 | 只读审计与冻结基线 | 找到完整纯 LinearNO、V1–V3、八任务入口、FLARE 准确定义 |
| 1 | 新版本合约、配置与 FLARE 风格 ResidualMLP | 深度语义、宽度、内部残差及配置封闭性 |
| 2 | 八逻辑块静态温度核心 | 8 套独立算子、5 个 ResMLP、共享映射、`1/sqrt(2)` |
| 3 | 两种双端自适应温度 | 形状、softmax 轴、零初始化、公平初始化和六类 attention 变体 |
| 4 | 工厂、版本分发、checkpoint 与通用 wrapper | 新旧版本隔离、严格恢复、三种温度模式可构造 |
| 5 | 六个标准 PDE 任务 | 原训练语义保持、两种动态温度可分别训练/评估/恢复 |
| 6 | AirfRANS、ShapeNet-Car 与完整启动脚本 | 工业任务边界、八任务显式运行入口 |
| 7 | 回归、成本、最终需求反查与使用文档 | 要求到代码和证据的闭环，不夸大未运行结果 |

---

## 二、已经复核的仓库事实

这些事实供 Codex 定位，但 Codex 在阶段 0 必须以实际工作树为准重新核验，不能把本文记录当作当前仓库状态的替代品。

- 复核时仓库默认分支为 `main`，HEAD 为 `36a2e0be9287949b06006606e4726aa77d4a2a66`（提交 `023`）。若实际 HEAD 已变化，只记录差异，不要擅自 checkout、reset 或覆盖用户改动。
- 仓库已经有完整纯 LinearNO，不需要重新发明基线：
  - `PDE-Solving-StandardBenchmark/model/LinearNO.py`
  - `cdlno/linearno/attention.py`
  - `cdlno/linearno/airfrans.py`
  - `cdlno/linearno/shapenet.py`
  - `cdlno/linearno/_profile_data.py`
  - `cdlno/linearno/profiles.py`
- 现有 loop 版本均不符合本次参数所有权：
  - V1 跨轮共享完整 block；
  - V2 共享 operator/LN1，而 FFN 逐轮独立，恰好接近本次要求的反方向；
  - V3 共享完整中间 block，并含 latent FFN、Q/K adapter，默认逻辑深度也不是本次固定的 8。
- 现有 V1–V3 的 SR 语义是 `1/R`；本次是新的 `1/sqrt(R)`。不得通过修改旧枚举或旧 forward 静默改变已有模型及 checkpoint 的含义。
- 公共分发层有多处 `is_v3`、`config_version == 3` 或直接导入 V3 checkpoint 的硬编码。新增模型类后还必须检查版本分发、训练/评估入口、记录器和恢复流程。

FLARE 参考必须以论文和官方实现为准：

- 论文：<https://arxiv.org/html/2508.12594v3>
- 官方实现固定版本：<https://github.com/vpuri3/FLARE.py/blob/4e053784fcb8b803c4459cba1dd2bd5566fe68a1/pdebench/models/flare.py>
- `ResidualMLP`：<https://github.com/vpuri3/FLARE.py/blob/4e053784fcb8b803c4459cba1dd2bd5566fe68a1/pdebench/models/flare.py#L24-L58>
- `FLAREBlock`：<https://github.com/vpuri3/FLARE.py/blob/4e053784fcb8b803c4459cba1dd2bd5566fe68a1/pdebench/models/flare.py#L161-L215>

双端温度的研究起点还包括 Transolver++：

- 论文：<https://arxiv.org/abs/2502.02414>
- 官方仓库：<https://github.com/thuml/Transolver_plus>
- 官方核心实现：<https://github.com/thuml/Transolver_plus/blob/main/models/Transolver_plus.py>

Transolver++ 官方代码只在 slice/compression 一侧使用逐点自适应温度，并结合 bias、clamp 和 Gumbel softmax。本模型把“输入相关温度”迁移到 LinearNO 解耦的 compression 与 reconstruction 两端，并规定下面两种 K 端粒度；同时本轮明确不使用 Gumbel 噪声。因此应核对其思想和真实实现，但不能逐行照搬后声称完成了本合同。

### 必须消除的 FLARE 命名歧义

FLARE 官方代码中的 `num_layers=L` 表示隐藏状态内部的残差 Linear 层数：

\[
h_0=\phi(W_{\mathrm{in}}x+b_{\mathrm{in}})+x,
\]

\[
h_\ell=h_{\ell-1}+\phi(W_\ell h_{\ell-1}+b_\ell),\qquad \ell=1,\ldots,L,
\]

\[
F(x)=W_{\mathrm{out}}h_L+b_{\mathrm{out}}+h_L.
\]

FLARE block 原设置取 \(C_{in}=C_h=C_{out}=H\)，所以输入和输出内部短接均启用。激活为 `nn.GELU(approximate="tanh")`，模块内部没有 LayerNorm、dropout 或输出激活。

- `num_layers=2` 实际共有 4 个 Linear：`fc1 + 2 个隐藏残差 Linear + fc2`。
- `num_layers=3` 实际共有 5 个 Linear：`fc1 + 3 个隐藏残差 Linear + fc2`。

FLARE 原论文把 `L=2` 用于模型外部输入/输出投影，而所有 FLARE block 的点域 ResMLP 为 `L=3`。本模型采用用户指定的 **FLARE 风格适配**：把逻辑第 1、8 个 LinearNO block 的点域 FFN 换成 `L=2` ResMLP，把中间 6 个逻辑 block 的点域 FFN 换成 `L=3` ResMLP。LinearNO 原有 lifting/preprocess、末端归一化和 prediction head 保持原任务实现，不改成额外 ResMLP。

为遵守此前“其他隐藏维数保持纯 LinearNO 基线”的约束，本模型令 \(C_{in}=C_{out}=H\)、\(C_h=H\times r_{\mathrm{task}}\)，其中 \(r_{\mathrm{task}}\) 沿用该任务纯 LinearNO 的 FFN ratio。FLARE 的短接条件原样保留：输入短接只在 `input_residual=True` 且 \(C_{in}=C_h\) 时启用，输出短接只在 `output_residual=True` 且 \(C_h=C_{out}\) 时启用；所有同宽隐藏层内部残差始终启用。因此 ratio=1 的任务具有首尾内部短接，ratio=2 的任务只有隐藏层内部残差。这一宽度保持是对 FLARE 拓扑的适配，不能写成 FLARE 全部超参数的逐项复刻。

---

# 三、总控提示词（先发送一次）

~~~~text
你将在仓库 https://github.com/hxh5159/CDLNO-w.git 中分阶段实现一个新的、与旧版本隔离的 LinearNO 架构。当前消息是持续有效的总控约束。之后我每次只授权一个阶段；你只能完成被明确授权的阶段，然后停止并提交可审查报告。即使你已经看到全部后续阶段，也不能提前执行。

一、工作原则

1. 先阅读仓库中的 AGENTS.md、当前 git 状态、有关源码、测试和文档。当前用户决定高于旧聊天记录、旧设计文档和旧实现。
2. 仓库已有完整纯 LinearNO。必须在它的真实任务接口、profile、attention 变体、数据流和 checkpoint 语义上增量实现，不要重新写一个脱离仓库的玩具模型。
3. 新模型公开 selector 固定为 `architecture=resmlp_dual_temp_v4`；`architecture_family="linearno_loop"`、`architecture_extension="resmlp_dual_temp"`、`architecture_version=4`；新 checkpoint 固定使用 `checkpoint_schema="resmlp_dual_temp_v4"`、`checkpoint_version=1`。内部 Python 包可命名为 `linearno_loop/v4`，但不得作为第二个公开 selector。它必须与纯 LinearNO、loop V1、V2、V3 共存，不能通过更改旧类、旧枚举值或旧 checkpoint 语义来实现。
4. 不得 reset、checkout 覆盖、删除或回滚用户已有修改；不得自动 commit、push、开 PR。除非阶段提示词明确授权，不下载真实数据，不启动真实长训练。
5. 如果缺少 GPU、真实数据或可选依赖，继续完成源码、静态检查、CPU 合成输入、可用框架对象和 checkpoint 验证；准确标注 NOT RUN。不得把随机前向称为真实训练或精度验证。
6. 不为了通过测试削弱断言、mock 掉被测数学、吞掉异常、静默 fallback 到旧模型，或复制实现逻辑作为测试 oracle。先证明测试能捕获典型错误，再证明实现通过。
7. 每阶段结束时更新 `docs/resmlp_dual_temp_v4/STATUS.md`，并新增或更新该阶段的证据文档。报告实际命令、环境、结果和未验证边界。

二、冻结的新模型合同

A. 基线与深度

- 基线是仓库中完整的纯 LinearNO，不是任何旧 loop LinearNO。
- 固定 8 个逻辑 block，执行次序为：
  `[first, A1, B1, C1, A2, B2, C2, last]`。
- 不实现可变 block 数，不继承旧 V3 的 D12、latent FFN、二次访问 adapter 或其他附加结构。
- 保持各任务原有 embedding width H、head 数、latent 数 M、输入 lifting、位置/时间特征、输出 head 和任务数据语义。

B. 参数所有权与 loop

- 8 个逻辑深度各自拥有独立的 LinearNO operator/attention、LN1 和 LN2；它们都不跨轮共享。
- 只注册 5 个点域 ResidualMLP 参数实体：`F_first, F_A, F_B, F_C, F_last`。
- FFN 调用映射严格为 `[F_first, F_A, F_B, F_C, F_A, F_B, F_C, F_last]`。
- A1/A2 只共享 F_A；B1/B2 只共享 F_B；C1/C2 只共享 F_C。A、B、C 相互独立，首尾相互独立。
- 不要把同一模块对象放进 ModuleList 的两个注册位置。由一个父模块只注册 5 个 owner，再通过普通不可注册的 key/index schedule 调用，确保 state_dict 和 optimizer 只看到每个参数一次。
- 动态温度预测器属于各自的 8 个 operator，不随着 A/B/C 的 ResidualMLP 共享。

C. FLARE 风格 ResidualMLP

- 第 1、8 个逻辑 block 的点域 ResidualMLP：`num_layers=2`。
- 中间第 2–7 个逻辑 block：`num_layers=3`；第二轮复用 A/B/C 的完整 ResMLP 参数。
- 执行深度为 `[2,3,3,3,3,3,3,2]`，唯一 owner 深度为 `[2,3,3,3,2]`。
- `num_layers` 是 FLARE 中内部同宽残差层的数量，因此 L=2/3 分别有 4/5 个 Linear，不能实现成总共 2/3 个 Linear。
- 所有新点域 ResMLP 固定 `C_in=C_out=H`、`C_hidden=H*原任务纯 LinearNO 的 ffn_mlp_ratio`。不得从旧 V3 的 matched/efficient profile 取 ratio，也不得为强行开启短接把 ratio=2 静默改成 1。
- `input_residual` 和 `output_residual` 开关保持开启，但必须沿用 FLARE 的维度条件：仅在对应维度相等时实际相加；每个 hidden Linear 的同宽内部残差始终启用。配置与成本报告要记录哪些 task 的端点短接实际生效。
- 激活固定为 `GELU(approximate="tanh")`；ResMLP 内无 LayerNorm、dropout、最终激活。
- 复用 FLARE 的拓扑、深度和激活，但保留仓库 LinearNO 的统一外层初始化协议；不要调用 FLARE 的全模型初始化去重置现有 LinearNO 主干，并在文档中说明这是适配。
- LinearNO 的 LN2 位于 ResMLP 外部。原 lifting/preprocess 和最终 LN/head 不因“首尾 L=2”而替换。

D. 外层缩放式加性残差

令 R=2。中间六个逻辑 block 的计算必须是：

`u_j = x_j + (1/sqrt(2)) * O_j(LN1_j(x_j))`

`x_{j+1} = u_j + (1/sqrt(2)) * F_{rho(j)}(LN2_j(u_j))`

首、尾逻辑 block 使用同样的加性结构，但两条分支系数均为 1。

- `1/sqrt(2)` 只乘 raw operator 分支和 raw outer ResMLP 分支，各一次。
- 不缩放 identity，不缩放整个 block 状态，不把系数乘进 ResMLP 的输入/隐藏/输出内部短接，不重复缩放。
- 新残差语义使用新的明确标识，例如 `sr_1_over_sqrt_r`。不得修改旧 `sr_1_over_r`。

E. 双端自适应温度

同一个新架构必须支持三个配置模式：

1. `base`：新八块/ResMLP/残差结构上的静态温度消融对照；不注册动态温度预测器。
2. `latent_k_point_q`：
   - compression/K 温度形状 `[B,Hd,1,M]`，每个 latent 一个样本与 head 相关的温度；
   - reconstruction/Q 温度形状 `[B,Hd,N,1]`，逐点温度。
3. `point_k_point_q`：
   - compression/K 温度形状 `[B,Hd,N,1]`，逐点温度；
   - reconstruction/Q 温度形状 `[B,Hd,N,1]`，逐点温度。

其中 `Hd` 表示 head 数，避免与 embedding width H 混淆。令 routing feature `S` 是每个 operator 内在 `in_project_x`/等价输入投影和 reshape 之后、`to_q/to_k/to_v` 之前的逐头特征，形状 `[B,Hd,N,d_h]`。若某个 attention 变体命名或布局不同，应在不改变其语义的前提下找到同一语义位置，并在代码注释和报告中说明。

- point predictor：`Linear(d_h,M) -> GELU(tanh) -> Linear(M,1)`，逐 head 输入但跨 head 共享 MLP 权重。
- latent-K predictor：先 `mean_N(S)` 得 `[B,Hd,d_h]`，再 `Linear(d_h,M) -> GELU(tanh) -> Linear(M,M)`，最后扩为 `[B,Hd,1,M]`。
- K 与 Q 使用两个不同预测器；8 个逻辑 operator 各自拥有不同的一对预测器。
- predictor hidden width 默认为 M，并封入配置与 checkpoint 元数据。
- 所有 predictor 的末层 weight 和 bias 在公共模型初始化完成后强制零初始化，使 delta=0。
- 构造动态模块不能扰动公共主干的随机初始化。完整生命周期固定为：先构造不含 predictor 的完整公共 wrapper（stem、8 operators/norms、5 RMLP、final norm/head）→按原 LinearNO 协议且仅一次执行全树初始化并保留特殊 placeholder 的原时序→在隔离 RNG 中显式调用 `install_temperature_predictors(...)`→把 predictor 末层 weight/bias 置零→再统一 `.to(device/dtype)`。安装前进入动态 forward 必须 fail-fast；安装后不得再次调用全树初始化。
- predictor seed 必须由稳定、与 Python `hash()` 无关的规则从 `(public_seed, logical_depth, branch, strategy)` 派生。Q 分支 seed 不含 strategy，使两种动态模式同一深度的同形 Q predictor 初始 tensor bitwise 相同；K 分支 seed 包含策略，因为输出结构不同。相同公开 seed 下三个模式的所有同名公共 tensor 必须 bitwise 相同。
- 温度使用：`tau = tau_base * exp(log(2) * tanh(delta))`，因此乘数位于 `(1/2,2)`。先按原变体得到合法 `tau_base`，不要再把动态后的总 tau clamp 回旧静态上界。
- logits 和 softmax 固定为：
  `K = softmax_N(L_K / tau_K)`，`Q = softmax_M(L_Q / tau_Q)`，`Z = K^T V`，`Y = Q Z`。
- `latent_k_point_q` 的 K 温度是每 latent 的空间分布温度；它不会改变某个固定 latent 内点 logits 的次序。
- `point_k_point_q` 的 K 调制作用在参与 N 维竞争的不同点上，可能改变点次序；文档中称为逐点异质温度/有界 logit 调制，不夸称为普通标量温度的完全等价物。
- plain/conv 路径的 base tau 为 1；temp/conv_temp 继续使用原 q/k 静态参数及原 clamp 生成 base；ShapeNet 保留现有 `tempreature_q/k` 拼写和 clamp 的兼容语义；AirfRANS 原声明温度目前未进入 forward，`base` 保持旧 forward，动态模式以 base=1 接入。阶段 0 如发现当前代码已变化，以实码为准并报告。
- 不构造 `[N,N]` 或 `[M,M]` 温度张量。

F. 明确排除项

本轮不实现 Gumbel 或其他随机路由噪声，不实现 AttnRes、LB/boundary residual、latent FFN、Q/K LoRA/低秩 adapter、history mixing、额外路由数量、可变逻辑深度、Triton 自定义核，也不改变数据、loss、optimizer、scheduler 或评估指标。

三、证据与报告格式

每阶段必须给出以下 A–F 六部分：

A. 本阶段授权范围与完成状态。
B. 修改/新增文件、原因和 diff 摘要。
C. 公式、张量形状、参数所有权/共享关系到具体代码位置的映射。
D. 实际运行的命令、环境，以及 PASS / FAIL / NOT RUN；失败不得隐去。
E. 冻结区域的未改动证据：纯 LinearNO、V1–V3、数据和训练语义、旧 checkpoint 等，按本阶段相关性列出。
F. 遗留问题，以及我最应人工检查的 3–5 个位置。

每阶段报告最后必须原样写：
“本阶段结束，未执行下一阶段。”
~~~~

---

# 四、阶段提示词

## 阶段 0：只读审计、参考核对与基线冻结

~~~~text
现在只执行阶段 0：只读审计、参考核对与基线冻结。遵守此前的总控提示词。本阶段禁止修改模型、配置、训练入口和测试逻辑；只允许新增审计/状态文档，以及为记录命令输出新增纯文本证据文件。

1. 仓库状态
   - 阅读全部适用的 AGENTS.md。
   - 报告当前 branch、完整 HEAD、git status、未跟踪文件和相关依赖环境。
   - 若与已知审计 HEAD `36a2e0be9287949b06006606e4726aa77d4a2a66` 不同，只分析变化；不得 checkout、reset、stash 或覆盖。

2. 完整纯 LinearNO
   - 从八任务入口追踪到实际 Model、block、attention、profile、初始化、训练、评估、保存与恢复代码。
   - 列出每个任务的 H、head 数、latent 数 M、原 FFN ratio、attention 变体、输入输出形状和特殊时间/网格逻辑。
   - 核对至少 plain、temp、conv、conv_temp、ShapeNet、AirfRANS 的 compression/reconstruction 公式、softmax 轴、base temperature、拼写兼容和输出投影差异。

3. 旧 loop 版本
   - 对 V1、V2、V3 分别画出真实参数所有权：operator、LN1、LN2、FFN、首尾、head、附加模块。
   - 明确说明为什么它们都不能直接作为本次核心；尤其核对 V2 是否为“共享 operator、逐轮 FFN 独立”，以及 V3 是否含 latent FFN/adapter 与旧深度假设。
   - 找出所有版本硬编码、配置/CLI selector、entry、checkpoint、provenance、记录器、launcher 和 accounting 接点。

4. FLARE 与 Transolver++ 精确核对
   - 阅读 FLARE v3 论文附录与官方 commit `4e053784fcb8b803c4459cba1dd2bd5566fe68a1` 中 ResidualMLP、FLAREBlock、模型 input/output projection 和初始化。
   - 用公式与源码行号证明：`num_layers=L` 是内部残差层数；L=2/3 分别有 4/5 个 Linear；激活、短接条件、norm/dropout 情况。
   - 明确区分 FLARE 原设置与本项目适配：用户要求逻辑 block 1/8 的点域 RMLP 取 L=2，中间 6 次取 L=3；LinearNO 外部 stem/head 不替换。
   - 核对 FLARE 的条件短接规则；读取每个任务纯 LinearNO 的真实 FFN ratio，并冻结为新 ResMLP 的 hidden ratio。明确 ratio=1 时输入/输出短接生效，ratio=2 时端点短接按维度条件关闭而 hidden residual 保留。
   - 阅读 Transolver++ 论文和官方 `thuml/Transolver_plus` 实现，记录其温度 predictor、bias/clamp、softmax 轴与 Gumbel 做法；逐项说明本模型为何只继承输入相关温度思想，而采用双端、两种 K 粒度、有界乘数、零初始化且无噪声的封闭合同。

5. 冻结证据
   - 为纯 LinearNO 和 V1–V3 选择小而有判别力的现有 smoke/config/state_dict 测试或只读快照；不要创建依赖真实数据的大型 golden 文件。
   - 记录现有测试的基线 PASS/FAIL。已有失败必须标记为 pre-existing，不得为让基线变绿而修改旧实现。

产出：
- `docs/resmlp_dual_temp_v4/REFERENCE_AUDIT.md`
- `docs/resmlp_dual_temp_v4/STATUS.md`
- 必要时 `docs/resmlp_dual_temp_v4/evidence/stage0/` 下的小型文本证据

`STATUS.md` 应含：当前 HEAD、工作树状态、阶段表、已通过阶段、模型冻结合同、待决问题、下一阶段入口条件。若实码与总控合同冲突，不自行改变合同；在报告中指出，并给出最小适配方案。

按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 1：新版本合约、配置与 FLARE 风格 ResidualMLP

~~~~text
阶段 0 已经由我审查通过。现在只执行阶段 1：建立隔离的新版本合约/配置，并实现可独立验证的 FLARE 风格 ResidualMLP。不要组装完整八块模型，不要接入动态温度 forward，不要修改八任务训练入口。

1. 版本与封闭合约
   - 新增独立 V4/extension 的 contracts、config、schema 或仓库现有架构要求的等价层。公开 selector 只能是 `architecture=resmlp_dual_temp_v4`；固定 family/extension/version/checkpoint schema 为总控中给出的值，内部 `linearno_loop/v4` 不是第二个 selector。
   - 固定逻辑拓扑 P1-C3-R2-S1，即 `[first,A,B,C,A,B,C,last]`；本轮不开放任意深度。
   - 固定 operator owner 数 8、LN1/LN2 owner 数各 8、RMLP owner 数 5、RMLP 调用数 8。
   - 固定 RMLP route `[0,1,2,3,1,2,3,4]`，执行 `num_layers=[2,3,3,3,3,3,3,2]`，owner 深度 `[2,3,3,3,2]`。
   - 每个任务的 `resmlp_hidden_ratio` 固定继承其纯 LinearNO `ffn_mlp_ratio`；固定 `GELU(approximate="tanh")`，并记录 input/output residual 开关及其维度条件。
   - 固定 residual formula/version 为新值 `sr_1_over_sqrt_r`，R=2，中间 alpha=1/sqrt(2)，首尾 alpha=1。
   - 温度模式枚举只允许 `base`、`latent_k_point_q`、`point_k_point_q`；记录 predictor hidden width=M、temperature multiplier bound=2、routing feature 语义。此阶段只定义配置，不接 forward。
   - 所有不合法组合应在模型构造前给出清晰错误。例如旧 residual enum、新 V3 附加模块、错误深度、错误共享映射、与该任务纯 LinearNO profile 不一致的 RMLP ratio、未知温度模式。
   - run id/config hash 必须区分三种温度模式，且 V4 不与 V1–V3 发生 selector 或 checkpoint extension 碰撞。

2. ResidualMLP 原语
   - 在新 V4 包内实现，避免直接改写纯 LinearNO 的 PointwiseMLP。
   - 精确实现：
     `h = GELU_tanh(fc1(x))`；若 `input_residual=True and C_in==C_hidden`，再令 `h=h+x`；
     对 l=1..L：`h = h + GELU_tanh(fc_l(h))`；
     `y = fc2(h)`；若 `output_residual=True and C_hidden==C_out`，再令 `y=y+h`。
   - `C_in=C_out=H`、`C_hidden=H*task_ffn_ratio`；input/output skip 只在维度相等时生效，hidden residual 始终生效。不含内部 norm、dropout、输出 activation。
   - 保留合理的通用形状检查；新模型打开 input/output residual 开关，但是否实际相加由维度条件决定。
   - 沿用仓库统一初始化流程；不能从模块构造器调用 FLARE 全模型初始化，不能构造后丢弃普通 FFN 再替换。

3. 独立验收
   - 使用独立手算 oracle 检查 L=2 和 L=3 forward；oracle 不得调用被测模块内部 helper。
   - 断言 L=2 有 4 个 Linear，L=3 有 5 个 Linear。
   - 用固定非零权重分别证明：维度相等时 input/output skip 生效、维度不等时按条件关闭、每个 hidden residual 始终生效。再用测试内独立错误实现、monkeypatch 或临时副本证明测试可捕获漏短接、总深度误解和错误 GELU；禁止直接改生产源码后再手工恢复。
   - 检查前向/反向、不同 batch/N、dtype 可用范围、state_dict 严格保存加载。
   - 配置测试不得为了读取元数据而强制导入 torch/GPU 或八任务数据。

4. 文档
   - 记录“FLARE 原始 input/output projection L=2、block L=3”与“本项目首尾逻辑 block L=2”的差异，不能写成逐字复刻 FLARE。
   - 更新 STATUS 和阶段 1 证据。

按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 2：八逻辑块核心与静态温度对照

~~~~text
阶段 1 已经由我审查通过。现在只执行阶段 2：组装任务无关的八逻辑块核心，先只使用 `temperature_mode=base`。不要实现动态温度，不要接八任务训练入口，不要改数据或 checkpoint 格式。

1. 直接构造最终所有权
   - 构造 8 套彼此独立的 operator/attention、8 套 LN1、8 套 LN2。
   - 只注册 5 个 RMLP owner：first、A、B、C、last；通过普通 route `[first,A,B,C,A,B,C,last]` 调用 8 次。
   - 不得先构造 8 个完整 block 再替换/删除 FFN；不得把同一 RMLP 重复注册在两个路径；不得从旧 full-block loop core 继承出隐式 operator 共享。
   - 末端 LN3 与 task head 在整个模型外只执行一次。本阶段核心提供清晰接口，不自行复制任务 head。

2. forward 数学
   - block 0 和 7：
     `u=x+O_j(LN1_j(x))`
     `y=u+F_j(LN2_j(u))`
   - block 1..6：
     `u=x+(1/sqrt(2))*O_j(LN1_j(x))`
     `y=u+(1/sqrt(2))*F_route(j)(LN2_j(u))`
   - 缩放只作用于 raw branch。RMLP 内部三个层级的 skip 保持系数 1。
   - base 模式保持对应 attention 变体原 compression/reconstruction 数学与静态温度语义。

3. 必须通过的结构验收
   - forward hook 证明执行次序恰为 `0,A1,B1,C1,A2,B2,C2,7`，operator 与 RMLP 调用各恰为 8 次；核心输出保持 `[B,N,H]`。末端 task LN/head 的注册与单次调用留到阶段 4 wrapper 验收。
   - `id(Attn_A1) != id(Attn_A2)`；对应 LN1、LN2 也不同。B、C 同理。
   - `id(RMLP_A1) == id(RMLP_A2)`；B、C 同理。A/B/C 互不相同；首尾互不相同。恰有 5 个唯一 RMLP owner。
   - optimizer 参数列表没有重复 object id；state_dict 中没有同一个共享 RMLP 的两套别名 key。
   - 共享模块在两次调用产生的梯度应累加。用“只保留第一轮 loss / 只保留第二轮 loss / 两轮共同 loss”验证，而不是只看 grad 非空。

4. 必须通过的数值验收
   - 用受控、互不相等的 identity/branch 输出手算 8 block oracle，能区分 `1/2`、`1/sqrt(2)`、缩放整个状态、漏缩放第二分支和重复缩放 RMLP 内部 skip。
   - L=2/3 实际 Linear 数与 route 对应正确。
   - 使用至少两个 N 和两种合成 profile 做前向/反向；检查有限值与输出形状。
   - strict state_dict 内存 round-trip 后，重新构造时先恢复 5-owner alias 结构再 load，输出一致。
   - 对纯 LinearNO、V1–V3 做相关 smoke 回归，证明没有改变旧 SR 与旧 selector。

5. 初始化可比性
   - 所有最终组件只构造一次，不构造后废弃。
   - 记录新 RMLP 替代普通 FFN后与纯 LinearNO不能整模型数值等价；不要把 base 模式误称为纯 LinearNO。

更新 STATUS 和阶段 2 证据，按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 3：两种双端自适应温度及全部 attention 变体

~~~~text
阶段 2 已经由我审查通过。现在只执行阶段 3：在新 V4 operator 中接入 `latent_k_point_q` 与 `point_k_point_q`，并保持 `base` 对照。不要接具体任务训练脚本或真实数据。

1. 公共 routing feature 与 predictor
   - 对每个逻辑 operator，取得 `S:[B,Hd,N,d_h]`：位于该变体的输入投影/逐头 reshape 之后、q/k/v 投影之前。
   - 两种模式的 Q predictor 都独立计算 `delta_Q=f_Q(S):[B,Hd,N,1]`。
   - `latent_k_point_q`：`g=mean_N(S):[B,Hd,d_h]`，`delta_K=f_K(g):[B,Hd,M]`，扩成 `[B,Hd,1,M]`。
   - `point_k_point_q`：`delta_K=f_K(S):[B,Hd,N,1]`。
   - f_K 与 f_Q 不共享；8 个 operator 也不共享 predictor。MLP 权重可跨 head 复用，但不同 head 输入产生不同温度。
   - point predictor 为 `d_h -> M -> 1`，latent predictor 为 `d_h -> M -> M`；中间激活 `GELU(approximate="tanh")`，末层后无激活。
   - 实现可延迟安装接口 `install_temperature_predictors(...)`（名称可按仓库规范等价调整）。动态模式在安装前 forward 必须清晰报错；重复安装必须拒绝，不能覆盖已训练 predictor。

2. 温度与路由
   - predictor 最后一层 weight/bias 在公共初始化之后置零。
   - `tau=tau_base*exp(log(2)*tanh(delta))`；保证正值、有限值和正确广播。
   - `K=softmax(L_K/tau_K, dim=N)`；`Q=softmax(L_Q/tau_Q, dim=M)`。
   - 后续保持 LinearNO 的 `Z=K^T V`、`Y=QZ` 及各变体原输出投影。
   - base 模式不注册 predictor，尽可能走未经改写的原 attention 数学。

3. 六类变体兼容
   - plain/conv：base tau=1。
   - temp/conv_temp：保留已有 learned q/k base temperature 和原 clamp，再乘动态倍率；不把总 tau 重新 clamp 到旧范围。
   - ShapeNet：保留 `tempreature_q/k` checkpoint key 拼写和 clamp 兼容。
   - AirfRANS：base 模式保持当前原 forward；如果审计确认旧温度声明未使用，动态模式 base=1，不借机改变旧类。
   - 保持 conv 网格 reshape、Air contiguous/采样路径、ShapeNet 输出投影等现有特性。
   - 最好在 V4 包中新增 adaptive wrapper/subclass/composition，不直接改变 `cdlno/linearno/attention.py` 的旧 forward。

4. 原语级公平初始化
   - 本阶段只验证 attention/core 级延迟安装。在最小测试 owner 中先执行一次公共初始化，再按稳定派生 seed 隔离安装 predictor；安装后末层 weight/bias 精确为零，安装前动态 forward fail-fast，重复安装拒绝。
   - 两种策略同一 logical depth 的 Q predictor 使用与 K strategy 无关的 seed，所以全部同形 Q tensor bitwise 相同；K predictor 按策略单独派生 seed。
   - common tensor 配对比较必须明确按 state key 和 shape 过滤，不以“输出接近”代替。
   - delta=0 时，两种动态模式的 K、Q、单 operator 输出和八块 core hidden output 应与同一 V4 `base` core 在相同公共参数下数值一致。完整 wrapper/head 的公共 tensor 与输出公平性留到阶段 4；不能声称与纯 LinearNO 整模型一致。

5. 数学与梯度验收
   - 断言 `sum_N K=1`、`sum_M Q=1`，并检查 tau 的精确形状、范围和广播。
   - 构造对抗 logits：证明 latent-K 对固定 latent 内所有点除以同一正标量，不会逆转该 latent 的点排序；证明 point-K 可逆转不同点的次序。
   - Q 逐点温度应改变每个点 latent 分布的锐度，但同一点内正标量不改变 logits 排序。
   - 首次 backward 时 predictor 最后一层必须有梯度；由于末层零初始化，前一层首次梯度可以为零。执行一次 optimizer update 后再前向/反向，验证前层开始获得学习信号。不要写成“首次所有 predictor 参数梯度均非零”。
   - 证明输入不同样本在更新后能得到不同 tau；两种策略得到不同 tau 形状与行为。
   - 检查可变 N、至少两个 M、所有 attention 变体、CPU；环境支持时检查 CUDA/AMP 有限值。不得产生 `[N,N]` 或 `[M,M]` 中间量，可用 shape hook/profiler/源码断言佐证。

6. 隔离回归
   - base 新模型保持阶段 2 输出。
   - 纯 LinearNO、V1–V3 的 state keys、selector 和相关 smoke 不变。

更新 STATUS 和阶段 3 证据，按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 4：工厂、版本分发、checkpoint 与任务无关 wrapper

~~~~text
阶段 3 已经由我审查通过。现在只执行阶段 4：把新核心接入独立的构造工厂、版本分发、checkpoint/provenance 和任务无关 wrapper。不要修改六个标准任务及两个工业任务的正式训练流程。

1. 构造与版本分发
   - 为 V4 新增清晰包结构和 factory；复用纯 LinearNO 已审计的 profile、attention 语义与任务接口，不复制数据/训练代码。
   - 更新所有必要的版本分发点，消除“只认识 V3”的硬编码，但旧默认解析必须保持不变。只有显式 selector 才构造 V4。
   - 支持 Standard、ShapeNet-Car、AirfRANS 三类任务 wrapper 的合成构造；保留各自 stem、位置/时间输入、末端 LN/head 和输出形状。
   - 三种 mode 均从同一 V4 config 构造：`base`、`latent_k_point_q`、`point_k_point_q`。

2. 初始化顺序
   - 严格执行总控规定的唯一生命周期：构造不含 predictor 的完整 wrapper → 按纯 LinearNO 协议仅一次初始化全树并保留 placeholder 的原时序 → 按稳定派生 seed 在隔离 RNG 中安装 predictor → 末层 weight/bias 置零 → `.to(device/dtype)`。不能构造普通 FFN 或旧 block 后丢弃，安装后不能再对整棵模型 `.apply` 初始化。
   - base 模式永不注册 predictor；动态模式各有 8 对。安装前、重复安装和安装后试图全树重初始化均应 fail-fast 或由构造 API 排除。
   - 三个模式同 public seed 的全部同名同形公共 tensor 必须 bitwise 一致；两种动态模式的同深度 Q predictor 也必须 bitwise 一致。delta=0 时完整 wrapper 输出对齐 base。
   - 输出一份参数 owner 清单：stem、8 operators、16 block norms、5 RMLP、0 或 8 对 predictor、final norm/head。

3. checkpoint 与 provenance
   - 新 checkpoint schema/version/extension 与 V1–V3 隔离，不隐式迁移旧 checkpoint。
   - metadata 至少封存：architecture selector、8-operator/5-RMLP route、执行深度与 owner 深度、任务 ResMLP ratio/activation/条件 internal skip、residual formula 和 R、temperature mode、predictor width、temperature bound/base policy、task/profile、H/Hd/M、attention variant、输入输出规格、代码/配置版本。
   - run id 必须区分两种动态模式和 base。
   - 加载顺序：先读并校验 metadata，再构造正确 alias 拓扑，再以 `strict=True` 加载 tensor。
   - 旧 V1–V3、错误 temperature mode、错误 route/depth/residual、错误 task/profile 的 checkpoint 必须在读取 tensor 前给出可解释拒绝；不能静默部分加载或 fallback。
   - 本阶段只完成 config/provenance 与严格 model-state 保存恢复；不新造通用 optimizer/scheduler/scaler/DataLoader/epoch resume。任务级完整 resume 留在阶段 5、6，沿真实入口实现。

4. 合成验收
   - Standard、Car、Air wrapper 在三种模式下前向/反向，形状符合原接口。
   - 每个完整 wrapper 的最终 LN/head 只注册一份、每次 forward 只调用一次；核心内部不得私藏第二份 head。
   - 每种模式保存到临时文件，在新的 Python 进程严格恢复，并验证参数 alias、metadata、输出一致。
   - 恶意修改 metadata 的 mode/route/residual/task，证明在 tensor load 前拒绝。
   - 同 seed 三模式 common tensor bitwise 对齐；零 delta 动态模式输出对齐 base。
   - 旧 checkpoint 的相关回归保持预期行为，已有 pre-existing failure 单列。

5. 记录和统计接口
   - 更新仓库现有 output recorder/accounting 的版本分发，使它能记录新版本；本阶段只接结构，不运行正式任务。
   - unique parameter owners 与 8 次实际执行分开统计，不因 tying 宣称计算量自动减少。

更新 STATUS 和阶段 4 证据，按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 5：六个标准 PDE 任务接入

~~~~text
阶段 4 已经由我审查通过。现在只执行阶段 5：把 V4 显式接入六个标准任务 Darcy、Elasticity、Airfoil、Pipe、Navier–Stokes、Plasticity。不要接 AirfRANS 或 ShapeNet-Car，不下载真实数据，不进行长训练。

1. 保持任务语义
   - 从阶段 0 的调用图接入现有统一工厂/entry。不要复制一套偏离原脚本的训练循环。
   - 除显式选择新架构外，保持原数据读取、normalizer、坐标/网格构造、输入输出通道、loss、optimizer、scheduler、batch 语义、seed、日志、checkpoint cadence 和评估指标。
   - Navier–Stokes 保持原 10 步 rollout/时间拼接语义；Plasticity 保持原 query/time 更新和输出逻辑；其他任务保持原 mask/网格/张量 layout。
   - 保持阶段 0 记录的 H、Hd、M 和纯 LinearNO task FFN ratio；RMLP 内部宽度为 `H*该 task ratio`。

2. 配置与命令
   - 对六个任务都提供显式 `architecture=resmlp_dual_temp_v4` 的 train、resume、eval/dry-run 路径。
   - `latent_k_point_q` 与 `point_k_point_q` 必须是两套可分别训练、评估、保存、恢复和记录的配置，run id/输出目录不能碰撞。
   - `base` 作为可选消融，不替代两种动态模式的命令覆盖。
   - 若仓库配置系统使用 Python/YAML/CLI 的其他命名，按现有惯例实现，但对用户暴露的选项必须唯一且文档一致。

3. 有意义的合成验收
   - 使用任务原有 model/loss/rollout 函数与符合真实 layout 的小型合成 batch，而不是只调用通用 `model(randn)`。
   - 六任务、两种动态模式均至少完成构造、一个 loss forward、backward 和有限值检查；资源允许时做 1–2 次 optimizer step。
   - 对 NS 覆盖完整 10 步 shape/control flow，并至少做一次该任务真实网格 N 的形状检查。对 Plasticity 使用真实 `101×31`、`N=3131` layout 与真实时间/query control flow；通过 batch=1、减少 optimizer step 数控制成本，不能为通过 conv reshape 擅自缩小生产 profile。若资源仍不足，可以另设显式标记的非生产 miniature test profile 补充快速反向，但必须另外完成真实 N 的 forward/shape 检查并把未完成项标为 NOT RUN。
   - 每任务至少选择一种动态模式做 checkpoint 新进程 strict resume；另一个模式可用共享的参数化测试覆盖，前提是 mode metadata 冲突测试明确。任务级 resume 必须沿现有真实入口恢复 optimizer、scheduler、scaler（若有）、epoch/step 和既有随机状态字段；不另造并行 checkpoint 协议。
   - 验证旧纯 LinearNO 与 V1–V3 的入口默认值、命令预览及相关 smoke 未改变。

4. 证据边界
   - 不得把合成 batch 的 loss 下降称为精度提升或收敛。
   - 若真实任务模块 import 会触发数据读取或训练，先最小重构为无副作用入口，并证明原 CLI 行为保持；不要靠 mock 隐去副作用。

更新 STATUS、六任务命令表和阶段 5 证据，按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 6：AirfRANS、ShapeNet-Car 与八任务启动脚本

~~~~text
阶段 5 已经由我审查通过。现在只执行阶段 6：接入 AirfRANS 与 ShapeNet-Car，并形成八任务可直接使用的训练/恢复/评估启动脚本。仍不下载数据、不进行真实长训练。

1. AirfRANS
   - 保持原数据采样、mask、坐标/物理特征、批处理、输出通道、loss、指标、ensemble/多模型隔离和保存恢复语义。
   - base 模式不得顺手激活旧实现中声明但未使用的温度。两个动态模式使用新 V4 operator 的明确 base=1 规则。
   - 用符合原对象/张量接口的合成样本完成 forward、loss、backward、短 optimizer step 和 strict resume。验证不同 ensemble member 的模型/optimizer/checkpoint 不串用。

2. ShapeNet-Car
   - 保持 PyG/Data 或当前真实图对象接口、点云 batch/fold/whole-object 边界、位置特征、输出反归一化和评估聚合。
   - 保留 `tempreature_q/k` 兼容 key 与原 clamp 生成 base 的语义；新 metadata 明确标识 V4，不能与旧 ShapeNet checkpoint 混用。
   - 环境有 PyG 时必须用真实 PyG Data/Batch 构造小型合成对象；没有时标记依赖缺失，仍完成不依赖 PyG 的组件与配置检查，不能用不等价字典假装完整通过。

3. 八任务 launchers
   - 不复用现有硬编码 D12、old V3 latent/adapter 的 launcher 冒充新模型。新增或彻底版本化隔离的 V4 启动层。
   - 对每个任务分别给出：
     a. `latent_k_point_q` train / resume / eval；
     b. `point_k_point_q` train / resume / eval；
     c. 可选 `base` ablation。
   - launcher 必须明确写出 `architecture=resmlp_dual_temp_v4`、温度模式和输出目录；两种动态模式不得共享 checkpoint 路径。
   - 支持 dry-run/print-config，只解析并打印最终封闭配置，不访问数据或启动训练。打印 ownership route、RMLP depths、residual formula、temperature mode 和 profile。

4. 验收
   - 两个工业任务的两种动态模式都完成配置、合成前向/反向和命令 dry-run。
   - 至少每个工业任务一种模式完成新进程 strict checkpoint round-trip；另一模式由参数化恢复测试覆盖。
   - 任务级 resume 沿 AirfRANS/Car 既有入口恢复各自 optimizer、scheduler/scaler（若有）、epoch/step、fold/member 与随机状态字段，不另造通用训练状态格式。
   - 所有八任务 launcher 的 train/resume/eval 命令构造通过，参数传递到实际 Model，而不是只停留在 parser。
   - 回归旧 launcher/selector；证明未把旧 V3 默认改为新版本。

更新 STATUS、完整八任务命令矩阵和阶段 6 证据，按 A–F 格式报告，然后停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

## 阶段 7：最终回归、成本分析、需求反查与交付文档

~~~~text
阶段 6 已经由我审查通过。现在只执行阶段 7：最终独立审计、回归、成本分析和交付文档。除修复本阶段发现的直接缺陷外，不新增模型机制，不运行真实长训练。

1. 从需求反查代码
   建立最终矩阵：
   `需求 -> 配置字段 -> 参数 owner -> forward 代码 -> 测试/命令 -> 实际结果 -> 未验证边界`。
   至少逐项覆盖：
   - 8 个独立 operator/LN；
   - 5 个唯一 RMLP 与 `[first,A,B,C,A,B,C,last]` 调用；
   - L=2/3 是内部层数，实际 4/5 个 Linear；
   - RMLP 沿用 task ratio、GELU tanh、条件 input/output skip 与始终启用的 hidden residual；
   - 中间两条 raw branch 各乘 `1/sqrt(2)`，首尾为 1；
   - 三个温度模式与两种动态形状；
   - 8 对独立 predictor、Q/K 不共享；
   - base temperature 六变体兼容；
   - 公平初始化、零 delta 退化和严格 checkpoint；
   - 八任务两种动态配置可分别 train/resume/eval。

2. 独立实现审计
   - 由未参与主要实现路径的测试或独立阅读方式检查 softmax 轴、广播、共享 object id、state_dict alias、初始化顺序和 metadata-first reject。
   - 对关键测试做临时 mutation：在测试内 monkeypatch、临时副本或独立小型错误实现中模拟把 `sqrt` 改成 R、共享一个 operator、复制 A 的 RMLP、交换 softmax 轴、改变 tau shape、去掉 predictor 零初始化，确认测试会失败。不要为此直接编辑生产源码；测试结束后工作树不得留下 mutation。
   - 检查没有 Gumbel、AttnRes/LB、latent FFN、旧 adapter、可变深度等越界功能。

3. 回归
   - 运行与改动相关的纯 LinearNO、V1、V2、V3 回归，并把新失败与阶段 0 的 pre-existing failure 分开。
   - 运行 V4 的组件、核心、三 wrapper、八任务配置/合成路径、strict checkpoint 测试。
   - 不为追求全仓绿色反复扩大无关测试；若全量测试代价合理可运行，否则列出选择依据和未覆盖区域。

4. 成本与性能
   - 分别报告纯 LinearNO、旧 loop 代表版本、V4-base、V4-latent-K、V4-point-K 的：总参数、trainable 参数、unique owner 数、8 次执行数、代表 shape 的 MAC/FLOPs、前向/前反向延迟和峰值显存。
   - 分开报告 RMLP tying 节省的“唯一参数所有权”和其实际 8 次执行成本；不要把参数共享写成 FLOPs 必然下降。
   - 分开报告两种 predictor 的参数/计算差异；latent-K 的 M 维输出通常比 point-K 的标量输出参数更多。
   - benchmark 先 warm-up、再同步、重复有限次数，记录硬件、dtype、batch/N/H/Hd/M。无 CUDA 时给 CPU 结果并标注 GPU NOT RUN。
   - 不根据合成 benchmark 宣称真实数据性能或精度提升。

5. 最终交付文档
   - 更新 `docs/resmlp_dual_temp_v4/STATUS.md` 为最终状态。
   - 新增或更新：架构公式、模块所有权图、配置字段、八任务命令矩阵、checkpoint 兼容表、测试证据、成本表、已知限制。
   - 明确写出真实数据训练/收敛/精度是否运行。若没有，统一标为 NOT RUN，不使用“完整验证”“性能提升”等表述。
   - 给出建议的下一步真实实验顺序：先 base，再 latent-K，后 point-K；同 seed/profile/训练预算比较，并保存温度统计，但不要在本阶段启动。

按 A–F 格式给出最终报告并停止。报告末尾写“本阶段结束，未执行下一阶段。”
~~~~

---

# 五、阶段返修与审查模板

## 1. 阶段返修提示词

~~~~text
不要进入下一阶段。只返修阶段 <N> 的以下问题：

1. <问题一：给出文件、行为或失败命令>
2. <问题二>
3. <问题三>

继续遵守总控提示词和阶段 <N> 的授权边界。先复现问题，再做最小根因修复，然后重跑能证明修复且能防回归的测试。不得用放宽断言、吞异常、fallback 或修改旧模型语义来绕过失败。

按 A–F 格式报告，特别列出返修前后的证据。末尾写“本阶段返修结束，未执行下一阶段。”
~~~~

## 2. 独立只读审查提示词

~~~~text
请只读审查当前阶段 <N> 的实现，不修改文件。重点检查：

- 实际参数 object id 和 state_dict 是否满足 8 套 operator/LN、5 个 RMLP 的所有权；
- `num_layers=2/3` 是否分别实现为 4/5 个 Linear；
- `1/sqrt(2)` 是否只乘两条 raw branch，且未进入 RMLP 内部 skip；
- tau 的输入特征、形状、广播、softmax 轴与 base temperature 是否正确；
- predictor 是否每深度、K/Q 独立，且初始化顺序没有扰动公共主干；
- 测试 oracle 是否独立，是否能捕获典型错误；
- 旧 LinearNO/V1–V3、数据和训练语义是否被意外改变。

输出按严重程度排序的问题，每项给出文件/符号、证据、影响和最小修复建议。若未发现问题，明确说明检查范围和仍未覆盖的风险。不要执行下一阶段。
~~~~

## 3. 批准进入下一阶段提示词

~~~~text
我已经审查并接受阶段 <N> 的实现与证据。现在按既定总控提示词只执行阶段 <N+1>。开始前先读取 `docs/resmlp_dual_temp_v4/STATUS.md`、当前 git diff 和阶段 <N> 的证据；若工作树状态与记录不一致，先报告并停止，不要猜测或覆盖。
~~~~

## 4. 上下文恢复提示词

~~~~text
先不要改代码。你正在继续一个分阶段任务。请读取：

1. 仓库适用的 AGENTS.md；
2. `docs/resmlp_dual_temp_v4/STATUS.md`；
3. `docs/resmlp_dual_temp_v4/REFERENCE_AUDIT.md`；
4. 已完成阶段的证据文档；
5. 当前 branch、HEAD、git status 和 git diff。

然后用不超过 20 条列出：已通过阶段、当前模型合同、当前改动、未解决问题、下一阶段授权前置条件。此轮只恢复上下文，不修改文件，不执行下一阶段。
~~~~

## 5. 可选本地提交提示词

仅当你明确希望 Codex 创建本地 commit 时再发送；默认总控不允许自动提交。

~~~~text
我已审查并接受阶段 <N>。现在只允许创建一个本地 git commit，不 push、不建 PR。提交前再次运行该阶段最关键的快速验收，并确认 git diff 只包含已审查内容。commit message 使用：

`<建议的提交信息>`

报告 commit SHA、包含文件、最终 git status 和测试结果，然后停止。
~~~~

---

# 六、人工审查时最关键的 12 个问题

1. Codex 是否确实复用了仓库完整 LinearNO，而不是从旧 V3 继续叠改？
2. 是否有 8 个不同的 operator、LN1、LN2 object？
3. 是否只有 5 个唯一 ResMLP，而 A/B/C 各执行两次？
4. 首尾 `num_layers=2` 是否为 4 个 Linear，中间 `num_layers=3` 是否为 5 个 Linear？
5. 是否保持 `C_in=C_out=H`、`C_hidden=H*task_ffn_ratio`，并按 FLARE 维度条件启用 input/output skip、始终启用 hidden residual？
6. `1/sqrt(2)` 是否只缩放中间 block 的两条 raw branch，首尾是否为 1？
7. 两种温度模式的 tau 形状、K 的 N 维 softmax、Q 的 M 维 softmax 是否完全正确？
8. 8 个 operator 的 predictor 是否独立，K/Q predictor 是否独立？
9. predictor 零初始化是否发生在公共初始化之后，三种模式的公共 tensor 是否同 seed bitwise 一致？
10. `base` 是否只是新架构的静态温度对照，而没有被错误宣称为纯 LinearNO 数值等价？
11. checkpoint 是否先验 metadata 校验、再重建 alias、最后 strict load，并拒绝错误模式/旧版本？
12. 八任务是否都有两种动态模式的独立 train/resume/eval 配置，且没有改变原数据、loss 与评估语义？
