# 交给 Codex 的分阶段实施提示词：独立压缩/重建、共享算子主干与特征门控稠密专家的 Loop LinearNO

> **使用方式**：将本文从“总任务”到“交付要求”一次性复制给在 `hxh5159/CDLNO-w` 实际工作树中运行的 Codex。按阶段连续实施，每阶段核对结果并记录；遇到证据不足的关键架构歧义，先从当前代码和本文求证，不能自行换成旧方案。无需在每个阶段要求我重新批准；遇到不可解决的阻碍或需要破坏既有用户数据时再说明。

## 0. 总任务、当前仓库与边界

你需要在 `https://github.com/hxh5159/CDLNO-w.git` **当前工作树**上，以已有的纯 LinearNO 和 Loop LinearNO 实现为参考，新增一个能够在该仓库全部八项任务上训练、续训、评估的**独立架构版本**。不要改造 Transolver 模型；不要把本模型实现为已有 loop 版本的行为变更。

我最后一次可核对的远端 `main` 为 `c721ed161f0b94e5293d7e43b7b55ef20ba48167`（2026-09-24）；**这不是假定你的本地工作树仍停留在该提交**。开始时先记录实际 HEAD、未提交改动、子模块和本地新增内容，再以实际代码决定接线；保留已有用户改动，不做重置或覆盖。仓库已有纯 LinearNO、Loop LinearNO v1/v2/v3，以及 `resmlp_dual_temp_v4`，还曾有 Top-1/Zero Expert 的其他设计；**它们都不是本次逐点 softmax、全专家并行的结构**。

本次冻结的目标是：在循环核心的同一**物理位置**上，两次逻辑访问共用算子中除压缩/重建之外的权重、共用每个点域专家；两次访问分别拥有压缩与重建投影、分别拥有点特征门控的打分参数。首、尾块独立。以节省的主干参数容纳多个**稠密计算**的点域 FFN 专家。新增模型应兼容八任务现有的数据处理、损失、训练日程、评估指标和输出（训练记录、图像、checkpoint、权重与原生任务产物）。

### 阶段 A：先阅读、盘点，不得直接套用旧模型

1. 阅读当前 `AGENTS.md` 和现存阶段报告；核对其时间和实际文件是否仍存在。重点浏览 `docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md`、`docs/LOOP_LINEARNO_FFN_IMPLEMENTATION_STATUS.md`、`docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md`、`docs/resmlp_dual_temp_v4/{STATUS,USAGE,CHECKPOINT_COMPATIBILITY,ARCHITECTURE}.md`。历史 `AGENTS.md` 若提到不存在的文件，记录“缺失”，不要创建虚假的历史报告，也不要让旧阶段冻结要求妨碍本次明确授权的隔离新增实现。
2. 画清楚当前实际结构和调用链：纯 LinearNO `cdlno/linearno/attention.py`、`cdlno/linearno/_profile_data.py`、六个 Standard 任务的 `PDE-Solving-StandardBenchmark/model/LinearNO.py`、AirfRANS/Car 各自的 LinearNO 模型；Loop v1/v2/v3/v4 的 core、wrapper、versioning、factory、入口、训练器和八任务启动脚本。核对 `cdlno/linearno_loop/standard_entry.py`、工业任务入口，以及 `linearno_loop/versioning.py` 和 `cdlno/linearno_loop/versioning.py` 的派发边界；文件如有移动，以当前工作树为准。
3. 检查现有任务配置、profile 解析、预处理/反归一化、最后输出层、NS 滚动预测、Plasticity 时间调用、Car/AirfRANS 原生评估与多成员逻辑；检查 `cdlno/experiment.py`、可视化、记录、checkpoint、strict 重载与 `tran_evaluate/linearno_loop_v4/` 的运行合同。区分纯 LinearNO、旧 loop 各版和本次新模型的输出/元数据。
4. 给出简短审计记录：实际 commit/脏文件；每个任务的 `C`、原 FFN 中间宽 `F_0=C×ffn_ratio`、原算子 head 数和 latent rank `M`，各任务算子变体与温度方式；版本选择和原生输出路径；此次将修改/新增的文件。若发现与下文规格有**实质冲突**，列出证据、在不修改已有模型的前提下做明确决策并在最终报告说明。不能因旧实现只支持某一种拓扑便删掉另一种。

请从此阶段起保持现有模型（纯 LinearNO、Transolver、Loop v1–v4）各自配置、权重键、CLI 默认值与训练评估结果合同不变。新增模型必须有自己的架构标识、版本化配置与存档识别；现有 `linearno_loop` family 内增加新 version 如与实际 version dispatcher 一致则可以，**不能**误称现有 v4 已是该设计。

## 1. 新模型的完整、不可偷换的数学规格

设隐藏维数为 `C`，每个 LinearNO 原有 attention head 的 latent rank 为 `M`，**门控专家数**为 `K`，每个专家的中间宽为 `F`。`M`、原算子 head 数、`K`、`F` 是四个不同的量；改变 `K/F` 绝不暗中改变 `M/heads/C`。

令 `p` 表示循环核心中一个物理块的位置，`r` 表示这个位置所在的第 `r` 次核心访问。每次访问的点表示 `X_{p,r} ∈ ℝ^{B×N×C}`，计算为：

```math
Z_{p,r}=X_{p,r}+\mathcal O_{p,r}\bigl(\mathrm{LN}_{1,p}(X_{p,r})\bigr),
\qquad U_{p,r}=\mathrm{LN}_{2,p}(Z_{p,r}),
```

```math
S_{p,r}=U_{p,r}W_g^{p,r}+b_g^{p,r}\in\mathbb R^{B\times N\times K},
\quad \pi_{p,r}[b,n,j]=\operatorname{softmax}_{j=1}^{K} S_{p,r}[b,n,j],
```

```math
G_{p,r}[b,n,:]=\sum_{j=1}^{K}\pi_{p,r}[b,n,j]\cdot E_{p,j}(U_{p,r}[b,n,:]),
\qquad X_{\mathrm{next}}=Z_{p,r}+\frac{1}{R}G_{p,r}.
```

两项固定实验均 `R=2`，所以**核心算子残差为 `+1×算子输出`，核心点域专家残差为 `+1/2×专家混合结果`**。为使可配置的非两轮拓扑有确定定义，本实施规格将 `R>2` 时专家分支推广为 `1/R`，`R=1` 时为 `1`；这是工程扩展选择，不声称已有针对其他 `R` 的性能证据。身份路径 `X` 和 `Z` 永不缩放；不能写成 `(Z+G)/2`，也不能套用 v1/v2 的“两支都乘 1/R”、v4 的“两支都乘 1/√R”。首尾非循环块两支残差系数均为 `1`，其点域分支同样替换为下面的特征门控专家，但这些块各有独立参数。块数增加不会额外引入第三条残差或修改最终任务输出层。

### 1.1 保留 LinearNO 算子的结构，明确什么独立、什么共享

- 在**同一物理位置 `p` 的不同访问 `r` 之间**：算子的 `to_q`、`to_k` **分别独立**，它们分别生成当前访问的分析/压缩权重与合成/重建权重。对有 Q/K 现有温度的任务，相应 `temperature_q/k`（ShapeNet-Car 中拼写为 `tempreature_q/k`）也应跟各自 Q/K 访问独立，沿用**原变体**的公式和 clamp；不得把它们强行共享导致两次路由仍被同一温度约束。AirfRANS 原生登记但 forward 不使用的 `temperature` 沿用其现有无作用语义，按位置共享、不擅自启用或删除。不得把 Q/K 分片当成转置互逆、做 token 编号对齐，或把独立 Q/K 误写成两套完整不共享的算子。
- **同一位置跨访问共用**：原生 `in_project_x`（含需要的 Conv 路径）、`to_v`、`to_out`、其他算子投影/非 QK 算子参数、`LN_1`、`LN_2`、下面完整的 `K` 个 FFN 专家。不同位置 `p≠p'` 的这些参数互不共享；首尾独立块也互不共享。由当前源码判断哪些注册对象属于此表，建立只有一个 owner 的参数注册结构，不要靠复制 state_dict 假装共享；优化器每个参数只出现一次。
- 保留原 LinearNO 的 **原有多头算子、各任务 `plain/temp/conv/conv_temp/airfrans/shapenet` 实现、`Q=softmax_M` 与 `K_{operator}=softmax_N`、Q/K/V/context、时间/空间编码及输出**。这里关于 `K_{operator}` 的字母只是原算子的 key，**不等于专家个数 `K`**。本次“门控不使用多头/温度”只针对新增门控，不能删掉原算子的 heads/QK 温度。不要引入 v4 的动态温度预测器或 v3 的 latent FFN/双侧 adapter。
- 由于内层共享的 `in_project_x` 和 `to_out` 等仍被重复调用，**不同访问并非整个算子独立**；独立 Q/K 可得到不同路由，但并不保证一定学到不同的路由。不要把此设计或性能提升写成已经证明的结论。

### 1.2 新的点域专家和门控

- 每个位置 `p` 拥有 `K≥1` 个独立参数的**完整 FFN**：`E_{p,j}: Linear(C,F) → GELU → Linear(F,C)`。保留纯 LinearNO 原点域 FFN 的线性层偏置、激活与 `LN_2` 放置方式；每个专家的**中间宽 `F`**一致（同次试验内），但可以通过新模型配置修改。共享仅指**同一 `p` 的该专家在不同 `r` 重复调用**，不同专家 `j` 不共享权重。并行指网络分支并行加权，**不是** Top-1 只运行某个分支；对每个点都计算全部 `K` 个专家。
- 本次门控在每次逻辑访问各有独立 `W_g^{p,r}∈ℝ^{C×K}`、`b_g^{p,r}∈ℝ^K`，输入是该访问的 `U=LN_2(Z)`；每个点分别在**专家轴 K 上做一次 softmax**。这是一层点特征 `Linear(C,K, bias=True)`，不用空间坐标单独驱动的 MLP；也**不沿点轴 N 归一化**。没有 Transolver 式门控多头、没有门控温度、没有 Top-k/Zero Expert、没有额外的二级门控。保留位置编码等原有 stem 对 `U` 的间接影响，但不把几何坐标接成新的显式门控输入。
- 这是受 GNOT 多专家加权思想启发、**将几何打分替换成点特征打分**的改法。文档中称“特征门控的稠密专家”，不要称为与 GNOT 相同的“几何门控”，也不要借已有稀疏专家计划实现成 Top-1 MoE。`K` 越大，实算 FFN 次数和耗时通常会上升：共享的是参数，不是免费计算。

**小张量验算**：令一轮的 `B=1,N=2,C=2,K=2`，`Z=[[5,1],[2,5]]`，`U` 取已知值（测试中可以通过固定 `LN_2`/独立纯函数注入），两个点的 router 权重分别 `[0.75,0.25]`、`[0.4,0.6]`；两个专家在该测试输入上的输出分别为 `E_1(U)=[[2,0],[1,0]]`、`E_2(U)=[[0,4],[0,2]]`。那么核心输出应为 `[[5.75,1.5],[2.2,5.6]]`：每点有自己的权重，专家全算，**最后只在专家混合上乘 1/2**。注意此例只用于验证混合/残差，不用它假装是某个随机初始化 FFN 的自然输出。

### 1.3 两种必需拓扑与自定义拓扑

定义 `P=首部独立块数`、`C_core=核心物理位置数`、`R=核心重复次数`、`S=尾部独立块数`，**逻辑执行块数 `L=P+C_core×R+S`**。所有顺序固定为先运行完整首部，接着顺序执行核心 `A,B,...` 共 `R` 轮，最后尾部；同一个核心位置的公共参数和专家跨轮共享，Q/K 与 router 各逻辑访问独立。尾部原任务 head/LN 只发生在真正最后逻辑块一次。

| 必做预设 | `P` | `C_core` | `R` | `S` | 逻辑执行顺序 | `L` |
|---|---:|---:|---:|---:|---|---:|
| `p1_c3_r2_s1` | 1 | 3 | 2 | 1 | `First, A, B, C, A, B, C, Last` | 8 |
| `p2_c2_r2_s2` | 2 | 2 | 2 | 2 | `First1, First2, A, B, A, B, Last1, Last2` | 8 |

支持通过新架构配置显式设置 `P≥1,C_core≥1,R≥1,S≥1` 的自定义拓扑，并正确更新逻辑层数、残差 `1/R`、head 所属层、记录和 checkpoint。至少包括两个预设和一个 `R≠2` 的最小前反向验证。禁止将一次共享循环轮理解成额外创建一套 FFN/`in_project_x`；禁止直接套 v4 固定 `p1_c3_r2_s1` 序列来冒充 p2。

## 2. 两个可独立调节的专家配置及任务默认值

在新架构独立提供、序列化、回显并校验 `expert_count=K`（正整数）和 `expert_width=F`（正整数）；CLI 建议明确命名为 `--linearno-loop-dense-expert-count`、`--linearno-loop-dense-expert-width`，与旧 Top-1 稀疏专家及 latent rank 的参数名区分。为方便**初次运行**，若目前工作树没有用户已经冻结的新模型默认 `K`，可取 `K=2` 作为明确声明的演示默认；它**不是性能最优结论**，脚本必须允许 `K=2,3,4,8` 与任意合法正整数。不要因改 `K` 自动重算 `F`，也不要因改 `F` 自动更改 `K/M/C/ffn_ratio`。除新模型的 `expert_width` 外，现有纯 LinearNO/旧 loop 的 `ffn_ratio` 语义不变。

当**未显式指定** `expert_width` 时，解析已选择的任务和 profile，设置 `F=C×该 profile 的原 LinearNO ffn_ratio`。例如 `paper_table8_on_release_model` 默认 profile：Airfoil、Darcy、Elasticity、Pipe、Plasticity 的 `C=128,F=128`；NS、AirfRANS、ShapeNet-Car 的 `C=256,F=512`。这只是该 profile 的实例；如选 `transolver_matched` 或明确调整模型配置，**以当次解析后的 profile 为准**，不能将该表写死成所有 profile 的值。新架构默认 `M` 使用**同一次纯 LinearNO profile**的原 latent rank（不要沿用旧 loop 曾经×2 的默认）；独立显式 override rank 须服从该任务原有的合法性约束，尤其 ShapeNet-Car 的 `M % (C/heads) == 0` 校验，不得无声放宽纯 LinearNO 的限制。

构建配置和入口需支持：同一拓扑下只改 K、只改 F、两者都改；在 Standard 六任务和 AirfRANS/Car 两任务均可通过新架构 CLI/配置完成 train、resume、eval，不需要改源代码。加载 checkpoint 时以存档中精确 `K,F,拓扑,rank,heads,variant,profile,版本,残差规则` 为准，显式请求冲突应在加载 state_dict 之前报清楚；旧存档缺少这些字段不能默认为新模型。

## 3. 逐阶段实现任务

### 阶段 B：建立隔离的 core、配置和共享归属

1. 在符合当前仓库版本策略的新目录/模块中建立新架构专用 config、version selector、operator/point block、wrapper 和参数所有权检查；给出架构名（建议 `partial_share_feature_gate_v5`，如 v5 已占用请换唯一名称），不要重写 v1–v4 的 forward、参数或公开 flag。`family` 是否继续 `linearno_loop` 由当前 version dispatcher 决定，不要凭名字擅自改变存档契约。
2. 按 §1 的 owner 图注册一次共享模块：位置级公共算子主干 + norm + `K` 个点域 FFN；访问级 Q/K 路由权重 + Q/K 现有温度 + `Linear(C,K)` router；首尾块只访问一次、所有参数自有。需要保留算子已有分支与 state 语义；若已有 `LinearNOAttention.forward` 不支持 Q/K 独立注入，优先在新版本创建**最小重组**的 operator 并用固定权重和 profile 与纯 LinearNO 算子逐项对齐，绝不修改纯算子在旧模型中的行为。
3. 按 §1.2 计算完整专家混合，保持梯度能到所有启用专家和不同访问的 scorer/QK；保持原 LinearNO 的全局 stem、placeholder、最终 `LN_3/head` 和任务 forward 签名。遵循仓库“整模型外层只初始化一次”的机制；不能在安装新模块后对共享旧主干全树二次 `.apply` 初始化或把优化器里注册同一参数两次。检查 Conv、Car、Air 的原生变体与 AMP dtype。
4. 实现 `p1_c3_r2_s1`、`p2_c2_r2_s2` 和自定义解析；采用单一真实调用 schedule 和参数 owner 映射进行检查，统计**唯一** trainable 参数，逐次调用数要等于逻辑 L，最后 head 只调用一次。

完成阶段 B 时给出两种拓扑的调用序列、所有权表（哪些对象按位置共享、哪些按访问独立）、参数计数及小形状前反向。重点验证从第二轮回传能更新首轮共用专家，也能单独更新第二轮 Q/K 与门控。

### 阶段 C：接到八项真实任务的已有训练评估路径

1. 阅读当前 Standard `cdlno_entry.py`、原生实验类和 LinearNO/loop factory；在模型选择里增添**显式的新架构分支**，保留旧 `--model`、旧 loop 模式、旧 profile 和训练器行为。六任务 Airfoil、Darcy、Elasticity、NS、Pipe、Plasticity 必须各自完成从命令→构造→loss/optimizer→checkpoint→eval 的路径。
2. 阅读 AirfRANS 和 ShapeNet-Car 的 `cdlno_entry`、训练/评估入口、fold/成员与保存约定；将新架构接入这些路径并保持原 task-specific 输出维数、单位、mask、可视化与物理评估。不能拿 Standard 的 forward 签名强套这两个任务。
3. 在仓库现有 launcher 风格下添加**新架构独立的**八个可运行脚本/等价命令，支持 `train`、`resume`、`eval`、`train_eval` 和原有 `--dry-run`/`--print-config`/`--experiment-dir` 约定（按当前入口实际能力调整），两预设、K 和 F 均可传入。旧 v4 launcher 的 plan/metadata 会强制 V4，不能只给它透传新的架构 flag；应建立新 launcher 或在公共入口做显式版本分流，保证 v4 原 launcher 仍只运行 v4。每个任务至少提供“选择纯 LinearNO 的同一 profile/rank + p1/p2 + K=2 + 默认 F”的确切调用实例，以及可独立覆盖 `K/F` 的实例；`--dry-run` 不应生成假 run、碰数据或加载权重。训练时原始数据不存在就准备可运行脚本并执行不依赖真实 benchmark 的最小闭环；不要造 SOTA 结果。

### 阶段 D：产物、续训与评估合同

1. 复用目前的运行目录及记录器：唯一 run 目录、解析后的有效配置和参数量、训练日志/每 epoch 结果、训练曲线/场图/数值产物、按任务的 checkpoint/纯模型权重以及独立的评估记录。原生 AirfRANS ensemble/member、Car 原生预测/Cd 及其历史模型导出按当前逻辑保留；这些历史便捷模型文件不要充当可恢复 optimizer 的完整档案。
2. 给新架构单独的 immutable architecture/config 元数据、原子 checkpoint/权重成对存档和严格恢复方案，或等效复用当前版本的可验证基础组件；序列化模型状态、optimizer、scheduler、RNG、任务需要的 sampler/normalizer 与 epoch，并记录 `P,C_core,R,S,L,K,F,M,heads,variant,profile,scale_op=1,scale_expert_core=1/R,gate_axis=K,gate_bias=true,gate_multihead=false,gate_temperature=none,dense_experts=true`。其中 `heads` **只表示原算子的头数**。若使用已有 `latest/final/epoch` 指针/哈希，应先校验元数据和指针再严格加载，并在 resume 时恢复完整训练状态；评估沿用当前仓库对显式 checkpoint selector 的规则，且每次 eval 不覆盖训练记录或更换 config。
3. 如现有 run manifest 记录参数归属/实际调用时序，扩展**新版本**的记录，使每次真实访问的 `p,r`、Q/K/scorer owner、共享专家 owner 可复核，同时不改动旧 v4 manifest 字段。视觉输出继续是目标值、预测值和误差场/任务现有图，额外门控统计只可选择性新增，不替代原图或改变性能指标。
4. AMP 检查要覆盖门控 logits、softmax、专家输出与分支相加的 dtype 及数值稳定性。区分 CPU/可用 GPU 的局部 autocast 验证和真正生产脚本的 AMP/Scaler+恢复验证；若生产路径本来未提供功能，不得声称已验证生产 AMP。既有 loop 历史曾出现混合精度来源类型不一致，不要引入类似问题或改变纯旧模型来掩盖它。

### 阶段 E：有辨识力的验收，不伪称实测结果

1. **结构**：对两预设、至少一个自定义拓扑核对执行顺序、唯一 owner 身份、按访问独立的 Q/K/门控、按位置共享的 `in_project_x/to_v/to_out/LN/专家`、首尾独立与最后 head 一次。检查 FFN 是 `K` 套带偏置 GELU 两层网络并**全部执行**；归一化仅在 K 轴，逐点可不同；门控无新增多头和温度，但原算子多头及原温度存在。
2. **计算**：固定小张量核对 §1.2 的数值结果以及核心两种不同残差系数；`K=1` 时 softmax 权重必为 1，前后两个现有 skip 位置都在；R=1/2/3 时总逻辑层数与 scale 正确。独立改 `K` 与 `F`，核对输出形状、参数计数、前反向/优化器更新，rank/head/旧 profile 不被联动篡改。对同权重算子做旧 LinearNO QKV 路径的针对性等价检查（operator branch 的算子方程不变），避免仅测“能跑通”。
3. **版本和真实入口**：跑八任务 print-config/dry-run 与各自可运行的最小 train→保存→严格重载→resume→eval 闭环（有数据时用原生少量数据；缺数据时用隔离的最小 mock 路径，说明覆盖范围）。检查每个任务的原生记录、visualizations、weight/checkpoint 不丢失。旧纯 LinearNO、Transolver、Loop v1–v4 的至少关键 selector 与已有代表性烟测应保持不变；将事前已知失败和新回归区分，不要改旧 golden 或静默删除历史测试。
4. **成本**：在相同任务/profile、逻辑 block 数、输入 `B,N` 下，将新模型（两拓扑，示例 K=2/3/4/8、默认 F 与另一个自定义 F）的**唯一参数量**、至少一次前向测得的 FLOPs/时延/峰值显存（能测的条件下）与纯 LinearNO 和已有 loop 分别列明。共享参数减少不代表全专家稠密计算便宜；不要宣称参数/FLOPs 已与原模型匹配或性能已胜出，除非实测数据支持。单个无数据 dry-run 不可用来声称训练收敛。

## 4. 交付要求

- 提交新增/修改文件清单，说明新架构名和配置/CLI、两个拓扑、`K` 与 `F` 的独立改法、任务默认 `F` 来源；写明一次真实 `train_eval`、`resume`、`eval` 以及 `--dry-run` 的**当前仓库中实际可执行**命令。至少演示一条 `K=3,F=显式非默认`、另一个 `K=2,F=profile 默认` 命令。
- 用表格给出两预设的唯一参数量和对比、八任务入口及产物、验收结果（通过/未运行/受环境阻挡），附实际复现命令、测试输出位置及当前 commit/diff。可以报告局限与风险，不得编造 benchmark 精度/时长或把历史失败写成新模型失败。
- 说明此次只通过**新增模型版本/适配层**增加能力，纯 LinearNO、Transolver、loop v1–v4 的代码、初始化、默认配置、已有 checkpoint 与历史训练产物未被更改。若因入口接线必须改共享文件，展示其只有选择新架构时才执行的新分支及旧模式的回归依据。

参考资料，以工作树实际实现为准：

- 当前工作仓库：<https://github.com/hxh5159/CDLNO-w>
- LinearNO 官方仓库：<https://github.com/HiPRL/LinearNO>
- Transolver 官方仓库：<https://github.com/thuml/Transolver>
- GNOT 论文（多专家门控的参考；本项目的打分输入和分享边界由以上规格决定）：<https://proceedings.mlr.press/v202/hao23c.html>
