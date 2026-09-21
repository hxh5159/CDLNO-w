# 面向 `hxh5159/CDLNO-w` 的 Looped LinearNO 后续分阶段 Codex 修改提示词

> 目标仓库：<https://github.com/hxh5159/CDLNO-w>
>
> 本文是在现有 Looped LinearNO 已完成实现之后继续扩展，不是重新实现 LinearNO 或重新实现三种循环残差。
>
> 最终只新增两个实验模式：
>
> 1. `round_specific`：共享 LinearNO operator，解除核心点域 FFN 的循环轮次共享；
> 2. `round_specific_latent`：在 `round_specific` 基础上，为每个共享核心 operator 增加一个跨轮共享的 latent FFN。
>
> 原先讨论过的“两子块 point-FFN stack/Hourglass 风格方案”已经取消，不得实现、保留占位配置或暗中混入本任务。

---

## 0. 使用方法

建议第一次把“总控提示词”和 `LF0` 一起交给 Codex。每个阶段完成后，先审查其报告和 diff，再单独发送下一阶段。不要一次授权真实数据训练、长训练或自动提交。

本文中的文件路径来自生成提示词时对远端仓库的只读核对。2026-09-20 可见远端 `main` 为：

```text
80ebe42d5755fc58ac6b41e2f6a0512d601ac8a8
```

这不是实施时应强制 checkout 的 SHA。Codex 必须以用户实际 checkout 的当前 HEAD、dirty 状态、当前 `AGENTS.md` 和现有实现为真值；不得 reset、clean、stash、rebase、覆盖或删除用户修改。

当前仓库已经包含：

- `cdlno/linearno_loop/` 中的完整 Looped LinearNO 模型与三种残差；
- 顶层 `linearno_loop/` 中的配置、schema、metadata 与恢复合同；
- 六个 Standard benchmark 以及 AirfRANS、ShapeNet-Car 的生产接入；
- `tran_evaluate/linearno_loop/` 启动器；
- checkpoint/resume/eval、记录、性能工具、AMP 修复及大量回归证据；
- `docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md` 中 LL0--LL10/LL9R 的历史状态。

因此本任务必须在现有实现上做版本化增量，不能照旧提示词重新覆盖 `LinearNOLoopCore`、三种残差或任务入口。

---

# 一、冻结后的科学与实现规格

## 1. 三种共存架构

仓库最终必须同时支持以下三种结构，并能分别 train/resume/eval：

| 模式 | 核心 operator | 核心 point FFN | latent FFN | 用途 |
|---|---|---|---|---|
| 既有 v1/legacy | 每个核心位置跨轮共享 | 随整个 block 跨轮共享 | 无 | 严格保留旧模型、旧命令和旧 checkpoint |
| `round_specific` | 每个核心位置跨轮共享 | 每个“核心位置 × 循环轮次”独立 | 无 | 新基础消融 |
| `round_specific_latent` | 每个核心位置跨轮共享 | 每个“核心位置 × 循环轮次”独立 | 每个核心位置一套，跨轮共享 | 最终方案一 |

建议新增稳定字段：

```text
core_ffn_mode = round_specific | round_specific_latent
```

建议 CLI：

```text
--linearno-loop-core-ffn-mode round_specific
--linearno-loop-core-ffn-mode round_specific_latent
```

旧命令未出现该字段时必须继续进入现有 v1 语义，不能被新默认值重新解释。若当前 schema 结构更适合另一名称，可在 `LF0` 报告中提出，但一个字段必须是唯一真值，不能同时存在多个别名或相互覆盖的开关。

新训练模式应使用版本化 architecture/config/metadata 合同，例如新的 extension/schema 版本。旧 v1 metadata 必须仍按其旧版本解析并 strict reload，不能通过修改 v1 默认值导致旧 config hash 或 checkpoint 失效。

## 2. 通用拓扑和共享关系

沿用现有唯一拓扑：

```text
P = prefix_blocks
C = recurrent_core_blocks
R = loop_repeats
S = suffix_blocks
```

执行顺序：

\[
\operatorname{Stem}
\rightarrow P_1,\ldots,P_P
\rightarrow (C_1,\ldots,C_C)^{\times R}
\rightarrow S_1,\ldots,S_S
\rightarrow \operatorname{Head}.
\]

继续支持：

- `p1_c3_r2_s1`；
- `p2_c2_r2_s2`；
- 完整自定义 P/C/R/S；
- `suffix_blocks >= 1`，最终 head 仍只在最后 suffix 执行一次。

对新两种模式，核心位置 \(p\) 与轮次 \(r\) 的共享规则为：

\[
\mathcal O_{p,1}=\cdots=\mathcal O_{p,R}=\mathcal O_p,
\qquad
\mathcal F_{p,r_1}\ne\mathcal F_{p,r_2}\quad(r_1\ne r_2).
\]

其中：

- 共享 operator \(\mathcal O_p\) 包含该位置的 `ln_1` 和 LinearNO attention/operator 参数；
- 轮次独立 point FFN \(\mathcal F_{p,r}\) 包含该逻辑位置自己的 `ln_2` 与 `mlp`；
- 不能只解除 MLP 权重共享而继续共享它的 `ln_2`；
- prefix 和 suffix 保持现有完整独立 block，不按轮次复制；
- 每个 visit 都重新由当前点状态计算 Q/K/V/context/readout；不缓存上轮 Q/K/V/context；
- 不跨 forward、物理时间查询、NS rollout 或 ensemble member 保存激活。

对 `P1-C3-R2-S1`：

- 3 套共享核心 operator；
- 6 套核心 point FFN；
- prefix/suffix 各一套原 point FFN；
- point FFN 总数与 8 次执行深度一致，但 operator 仍只存 5 个物理位置。

对 `P2-C2-R2-S2`：

- 2 套共享核心 operator；
- 4 套核心 point FFN；
- prefix/suffix 共 4 套原 point FFN；
- operator 仍只存 6 个物理位置。

`round_specific` 可以在文档中解释为“由固定逻辑位置硬路由的轮次特定 FFN”或“部分参数共享”，但不得称为稀疏 MoE：它没有输入条件路由、负载均衡或动态专家选择。

## 3. LinearNO 与 latent FFN 的准确计算

LinearNO operator 的核心仍为：

\[
Q\in\mathbb R^{B\times h\times N\times M},\quad
K\in\mathbb R^{B\times h\times N\times M},\quad
V\in\mathbb R^{B\times h\times N\times d_h},
\]

\[
C=K^\top V\in\mathbb R^{B\times h\times M\times d_h},
\qquad
Y=QC.
\]

其中 Q 沿 M softmax，K 沿 N softmax，\(H=h d_h\)。

只在 `round_specific_latent` 中，把当前核心位置的 context 合并 heads：

\[
Z=\operatorname{MergeHeads}(C)\in\mathbb R^{B\times M\times H},
\]

然后执行：

\[
\widetilde Z
=Z+W^{z}_{2,p}\operatorname{GELU}
\left(W^{z}_{1,p}\operatorname{LN}^{z}_{p}(Z)\right),
\]

再恢复为 `[B,h,M,d_h]`：

\[
\widetilde C=\operatorname{SplitHeads}(\widetilde Z),
\qquad
Y=Q\widetilde C.
\]

冻结要求：

1. latent FFN 严格位于 `K^T V` 之后、Q readout 之前；
2. 每个核心位置 \(p\) 有独立的 \(\mathcal G_p\)，不同位置不共享；
3. 同一 \(\mathcal G_p\) 在 R 个循环轮次之间共享；
4. prefix/suffix 不增加 latent FFN；
5. latent FFN 对每个 latent token 独立应用相同通道函数，不混合 M 轴；
6. 不增加 M×M latent attention，更不增加 N×N attention；
7. 不读取历史 latent token，不要求不同轮次或不同 block 的 token 索引语义对齐；
8. GELU；两个带 bias 的 Linear；无 dropout；独立 affine LayerNorm，`eps=1e-5`；
9. heads 合并/恢复必须保持确切 head 顺序并使用安全的 transpose/contiguous/reshape；
10. 输出 dtype/device 与输入 context 一致，AMP 下不得静默关闭整个 operator 的 autocast。

其复杂度增加约为每次核心 operator visit：

\[
2BMHD_z\ \text{MACs}
\]

另有 LayerNorm、GELU 与 residual add。总新增主矩阵 MAC 约为：

\[
2BCRMHD_z.
\]

它与 N 无关，不改变主要的 N×M 因子化读写结构；但是不能据此预先声称真实训练更快或精度一定提高。

## 4. latent FFN 的固定宽度

第一版按任务固定，不引入新的宽度搜索 CLI：

| 任务 | H | latent FFN |
|---|---:|---|
| Airfoil | 128 | `128 -> 572 -> 128` |
| Darcy | 128 | `128 -> 572 -> 128` |
| Pipe | 128 | `128 -> 572 -> 128` |
| Plasticity | 128 | `128 -> 572 -> 128` |
| Elasticity | 128 | `128 -> 128 -> 128` |
| Navier--Stokes | 256 | `256 -> 256 -> 256` |
| AirfRANS | 256 | `256 -> 256 -> 256` |
| ShapeNet-Car | 256 | `256 -> 256 -> 256` |

这些值应由版本化的新 profile/config 解析并完整写入 metadata，而不是散落在多个模型文件或启动脚本中。生产 CLI 第一版不提供任意 `latent_ffn_hidden` 搜索；测试可以通过内部小型构造器使用缩小维度。

## 5. 初始化合同

latent FFN 采用 identity-preserving 初始化：

- `LN.weight = 1`，`LN.bias = 0`；
- 第一 Linear 沿用 LinearNO release 的普通 Linear 初始化：`trunc_normal_(std=0.02)`，bias=0；
- 第二 Linear 的 weight 和 bias 在任何全树初始化之后严格置 0；
- 因而初始化时 \(\widetilde Z=Z\)，`round_specific_latent` 与同权重 `round_specific` 在 forward、输入梯度和公共参数梯度上应精确或达到既定严格容差；
- 初始第一步中，第二 Linear 可以获得梯度，而第一 Linear/LN 的梯度为零是预期的分阶段启动现象，必须测试和如实记录。

新增 latent 模块的构造和初始化不得改变公共主干、轮次独立 point FFN、placeholder 或数据 RNG。Codex 必须实现并验证隔离初始化，例如使用独立、记录在 metadata 中的派生 seed 和 `torch.random.fork_rng`，或其他能提供同等证据的做法。

同 task/profile/topology/rank/public seed 下至少满足：

- `round_specific` 与 `round_specific_latent` 的所有公共 state_dict tensor 逐位相同；
- 三种 residual mode 的公共 backbone/round-specific FFN tensor 逐位相同，差异仅为各自 router 及 latent 专属键；
- 构造前后的全局 Python/NumPy/Torch RNG 差异符合现有公平初始化合同；latent 专属初始化不能推进公共 RNG；
- placeholder 的初始化时序不得被改变。

不得为了制造“公平”把所有轮次 FFN 初始化成同一份权重。不同 \((p,r)\) 的 point FFN 是独立参数并按独立 RNG 抽样；只有公共部分需要跨实验模式配对。

## 6. rank/M 规则修正

新 `round_specific` 和 `round_specific_latent` 模式默认使用当前任务 LinearNO/Transolver profile 的原始 M：

\[
M_{\rm new}=M_{\rm base}.
\]

即新模式默认 `rank_multiplier=1`，不再默认翻倍。仍允许显式 multiplier=2 或显式 actual M 作为控制实验，但不能同时给 actual M 与 multiplier。

兼容要求：

- 不得直接把旧 v1 config 的默认 multiplier 从 2 改成 1，从而破坏旧 config hash、旧 metadata 或旧 checkpoint；
- 旧 v1 archive 继续按 v1 的冻结语义恢复；
- 新 version/mode 的默认值为 1；
- 新 run directory、sidecar 和 checkpoint 必须明确记录 base M、resolved M、policy、字段来源和版本；
- 省略新 `core_ffn_mode` 的旧命令保持现状，不被静默迁移。

## 7. 与三种现有残差的组合

新两种模式都必须兼容当前三种残差：

```text
sr_1_over_r
rb_attnres
lb_attnres_1_over_r
```

不得改写现有公式：

- SR：核心每次 operator raw branch 与所选轮次 point-FFN raw branch 各乘一次 `1/R`，identity 不缩放；
- RB：每个 `(round, operator/point-FFN sublayer)` receiver 的来源和 raw partial 时序保持现状；operator 使用共享 \(\mathcal O_p\)，FFN 使用对应 \(\mathcal F_{p,r}\)；不增加普通 residual，不使用 `1/R`；
- LB：轮内保持 SR `1/R`，Delta 仍由实际 `Y-H` 得到，boundary/output AttnRes 不二次缩放；
- 现有 RB AMP 的 receiver-boundary dtype 修复必须保留；不得把 local cast 扩散到 SR/LB 或权威 raw cache；
- latent FFN 是 operator 内部 context 映射，不成为新的点域 residual receiver，不新增 AttnRes 来源；
- latent FFN 内部 residual 不单独乘 `1/R`；SR/LB 只对整个 operator raw output 按现有位置缩放一次；
- receiver 数量、来源列表、调用顺序和 router 参数量不能因为新 FFN/latent 结构而变化。

## 8. 参数与计算核对

设原 point FFN 中间宽度为 \(D_0\)。一个包含 pre-LN 的 point/latent FFN 参数量为：

\[
F_H(D)=2HD+D+3H.
\]

新基础模式相对现有共享整 block 的 v1 增加：

\[
\Delta P_{\rm round\text{-}FFN}=C(R-1)F_H(D_0).
\]

该项只增加存储参数、优化器状态和 checkpoint 大小，不增加 point FFN 的实际调用次数或对应理论 FLOPs。

latent 模式再增加：

\[
\Delta P_{\rm latent}=C F_H(D_z),
\]

因为 latent FFN 按核心位置存储、跨轮共享，但在每个核心 visit 都执行。

实现必须对八任务、两个 preset、三残差、两个新模式、M×1/M×2逐项给出：

- 实测总参数；
- operator、point FFN、latent FFN、router、stem/head 分区参数；
- state_dict key/shape；
- unique/executed 调用次数；
- 主矩阵 MAC 与非矩阵操作清单；
- 与解析公式的逐项相等检查。

不能把参数量恢复写成计算量不变，也不能用 MAC 代替真实延迟、显存或 epoch 时间。

## 9. 明确排除

本任务不实现：

- 已取消的两子块 point-FFN stack 或 Hourglass FFN；
- SwiGLU、SiLU gate 或新的 activation 消融；
- 稀疏专家、Zero Expert、MoE、LoRA、adapter；
- Transolver loop 化；
- M 默认翻倍；
- 历史 latent AttnRes、history-conditioned compression、HCC、CKCR、LSAR；
- latent token 间 M×M self-attention；
- 显式 timestep embedding、ACT、动态停机或测试时任意改变 R；
- 每层不同 M、温度或粒度；
- 数据、split、normalizer、目标、loss、metric、训练预算或可视化语义改变；
- `strict=False`、忽略键、随机补权重、把旧 v1 checkpoint 猜测迁移到 v2；
- 自动 commit/push、真实数据下载或未经授权的长训练。

---

# 二、总控提示词

```text
你将在用户当前的 hxh5159/CDLNO-w checkout 中，为已经完成的 Looped LinearNO 增加两个版本化模式：

1. round_specific：核心 LinearNO operator 按物理核心位置跨轮共享，point FFN（包括 ln_2）按“核心位置 × 循环轮次”独立；
2. round_specific_latent：在 round_specific 基础上，每个共享核心 operator 增加一套位于 K^T V 与 Q readout 之间的 residual latent FFN，该 latent FFN仅按核心位置存储并跨轮共享。

先完整阅读仓库根 AGENTS.md、docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md、LL9R/LL10报告、原 Looped LinearNO 提示词与证据，再沿实际 import/forward/config/checkpoint/launcher 路径审计当前代码。当前代码是实施真值，不能假设旧提示词中的路径、HEAD或测试统计仍然成立。

硬性边界：
1. 禁止 reset、clean、stash、checkout 覆盖、rebase、commit、push；保护 tracked/untracked/ignored 用户文件和已有证据。
2. 既有 v1 Looped LinearNO、三种 residual、纯 LinearNO、linearno_history、Transolver、CDLNO/KCDNO/MSAR-LNO、任务数据与训练评估协议默认冻结。
3. 新模式必须版本化；省略新字段的旧命令继续使用旧 v1 语义，旧 config hash/metadata/checkpoint 必须 strict replay。不得修改 v1 默认 rank=2× 的历史含义；新模式默认 M=base，即 multiplier=1。
4. 不实现已取消的两子块 FFN stack/Hourglass 方案，不留下不可达配置或占位模块。
5. 不直接修改纯 LinearNO attention 数学来方便新功能，除非先证明所有旧路径 state_dict、输出、梯度、RNG逐位不变；优先在独立 v2 路径实现。
6. 两个新模式必须支持现有 P/C/R/S、两个 preset、custom 和 SR/RB/LB 三种残差。现有 receiver 数、来源、1/R位置及RB AMP修复不得改变。
7. round-specific FFN包括独立ln_2+MLP；共享operator包括ln_1+LinearNO operator。prefix/suffix保持现状。
8. latent FFN公式、位置、宽度、GELU、无dropout、LayerNorm eps1e-5、每位置跨轮共享及W2零初始化完全按本提示词冻结规格执行。
9. 新模块初始化必须隔离，确保同seed下两个新模式的公共tensor逐位一致，不推进公共RNG。不得让全树apply随后覆盖latent W2的零初始化。
10. 所有新 checkpoint 都必须 metadata-first、完整构造参数校验、strict=True；跨新模式、跨residual、跨topology/rank、v1→v2错误加载必须在torch.load或权重应用前明确拒绝。
11. 先写独立oracle和失败测试，再实现生产代码。oracle不能调用被测forward，也不能复制生产helper充当expected。
12. 不访问真实数据、不执行长训练、不安装/降级依赖。可运行静态检查、CPU/GPU合成forward/backward/AdamW、AMP、strict reload、原生小型闭环和dry-run。
13. 每阶段只执行该阶段，建立单独的 docs/loop_linearno_ffn_audit/lfX/ 证据，不覆盖LL0--LL10历史证据。报告明确PASS/PARTIAL/BLOCKED、修改文件、公式到代码、命令、passed/failed/skipped/not-run、冻结区和3--5个优先复核点。
14. 若仓库事实与冻结规格存在会改变模型语义的冲突，完成不受影响的只读工作后停止，列出源码位置、两种处理及影响，不得自行改变研究设计。

完成每个阶段后写“本 LFx 阶段结束，未执行下一阶段”，然后停止等待用户批准。
```

---

# 三、分阶段提示词

## LF0：最新仓库只读审计与冻结

```text
现在只执行 LF0。禁止修改任何生产代码、parser、schema、checkpoint、launcher、测试golden或依赖。只允许新增：

- docs/LOOP_LINEARNO_FFN_REFERENCE_AUDIT.md
- docs/LOOP_LINEARNO_FFN_IMPLEMENTATION_STATUS.md
- docs/loop_linearno_ffn_audit/lf0/**

任务：
1. 完整读取根AGENTS.md及其当前优先报告；读取现有Looped LinearNO旧提示词、配置、LL2--LL10、LL9R、性能和启动器报告。记录repo/remote/branch/HEAD/tree、git status、tracked/untracked/ignored及当前已有失败/skip，不重置。
2. 建立LF0起点manifest，至少覆盖：
   - cdlno/linearno_loop/**、linearno_loop/**、tests/loop_linearno/**；
   - cdlno/linearno/**、linearno_history相关代码；
   - 三项目的LinearNO/loop入口、parser/factory/train/eval；
   - checkpoint/training_state/experiment/record/visualization；
   - tran_evaluate/linearno_loop/**；
   - 当前loop性能工具、provenance projection及LL历史证据。
3. 沿实际调用链确认当前v1：
   - block/body/core的所有者关系；
   - ln_1/Attn/ln_2/mlp/head的state_dict路径；
   - SR/RB/LB真实forward与RB AMP local-cast；
   - Standard/Air/Car初始化与placeholder时序；
   - config/schema/metadata version、constructor严格校验、run id、checkpoint/resume/eval；
   - 八任务loop启动器的参数流和actual M默认值。
4. 逐一核对六类LinearNO operator（plain/temp/conv/conv_temp/airfrans/shapenet）如何取得features、Q/K/V、context、readout和to_out。给出在不修改纯模型state_dict/数学的情况下插入context processor的最小安全方案。
5. 设计v2 module ownership：共享ln_1+operator；round-specific ln_2+MLP；每核心位置共享latent FFN。明确怎样避免重复注册、参数alias、unused子模块、二次初始化和无用构造消耗RNG。
6. 审计旧v1默认M×2与新v2默认M×1的版本迁移方案。旧config必须原样重建；不能仅改一个默认常量。给出versioned config/class/metadata/CLI和run-directory方案。
7. 捕获修改前数值夹具：至少覆盖两个preset、三residual、六operator variant、CPU FP64/FP32及可用GPU/AMP。保存输入、输出、输入/参数梯度、optimizer一步、RNG、state keys/shape、参数量和调用序列。
8. 运行当前loop与关键旧回归，记录真实baseline。不得修复旧失败或刷新golden。若当前LL10报告与实际测试不同，分别记录历史结论和本次事实。
9. 输出拟新增/最小修改文件表、阶段风险和LF1--LF7验收矩阵。特别说明是否需要新v2类路径；若无法同时满足旧严格构造签名与新字段，优先采用新类/schema版本，不放宽旧strict规则。

LF0只有在没有未解释的参数所有权、初始化、context插入或旧archive兼容冲突时才可PASS。完成后停止。
```

## LF1：版本化配置、schema、metadata 与实验矩阵

```text
LF0已由用户审查通过。现在只执行LF1：实现纯配置/schema/metadata合同，不实现新torch模型，不修改生产任务路由。

1. 保留v1常量、默认M×2、config hash、metadata恢复和class path的历史语义。为新模式建立明确v2 extension/schema/config版本；旧validate/read/restore按保存版本分派，不能用新默认重建旧request。
2. 新增唯一字段core_ffn_mode，正式取值仅：
   - round_specific
   - round_specific_latent
   不加入已取消stacked/hourglass值。
3. 新v2默认rank_multiplier=1；允许显式1/2或actual M，冲突规则沿用。八任务base M从当前profile动态解析，不硬编码到模型。
4. loop_spec必须明确记录：
   - operator sharing scope；
   - point FFN与ln_2的round-specific scope；
   - point FFN实例数P+C*R+S；
   - latent enabled、位置、每核心位置共享、实例数C；
   - latent width、activation、norm/eps、dropout、bias、W2 zero init；
   - feature timestep embedding仍为false，但round-specific FFN属于执行位置专属参数，不得宣称全模型没有位置专属参数。
5. 新model_spec使用完整显式constructor字段，不用*args/**kwargs。若v1类签名无法兼容，新增v2 class path；不得放宽v1 validate_constructor。
6. 新CLI字段必须只有显式出现时才选择v2。旧命令namespace/default/目录保持原样。旧A/K/history字段继续与loop互斥。
7. 新run id必须包含core_ffn_mode、M、PCRS、residual、seed和config hash，避免两个新模式或v1目录碰撞。
8. 新metadata/checkpoint记录初始化子种子或等价可复算协议、完整state分区、latent width来源和版本。跨mode/topology/residual/rank/version冲突在模型构造或torch.load前拒绝。
9. 生成但不运行正式矩阵：8任务×2preset×3residual×2新mode×3paired seed，另M×2控制及至少一个custom topology。
10. 测试严格类型、未知字段、旧v1精确重放、新v2序列化/hash/restore、M默认、width解析、CLI显式来源、run-id隔离和所有非法组合。schema层保持无torch/任务入口导入。
11. 更新配置报告和状态，重跑LF0旧配置/metadata回归及冻结检查后停止。
```

## LF2：round-specific point FFN 原语与核心模式

```text
LF1已审查通过。现在只执行LF2：实现round_specific的任务无关原语与合成core，不接生产任务CLI，不实现latent FFN。

1. 保持现有v1 LinearNOLoopCore和LinearNOBlockBody的旧调用/keys/数学可严格重放。优先新增v2 owner/adapter，而不是在v1中加入大量条件分支。
2. v2核心每个物理核心位置只拥有一套共享ln_1+operator；每个(p,r)拥有独立ln_2+point MLP。prefix/suffix继续使用现有完整block，最终head不变。
3. 不允许：
   - deepcopy整block后只口头声称operator共享；
   - 不同round注册重复operator参数；
   - 多个state_dict路径指向同一FFN参数；
   - 为获得FFN而构造并丢弃会推进公共RNG的完整operator；
   - 只解共享MLP但共享ln_2。
4. 轮次FFN使用各任务原PointwiseMLP/PointMLP合同和GELU，保持原中间宽度H*mlp_ratio、bias与dropout语义。不要引入新的FFN实现差异。
5. 将round_specific接入SR/RB/LB：每次逻辑FFN visit准确选择F[p,r]；operator始终选择O[p]。RB raw partial/source时序、LB Delta和SR缩放位置保持不变。
6. 先写独立全core oracle，显式持有共享operator函数和二维[p][r] FFN函数；不能调用生产core。逐步比较每次operator输入/输出、FFN选择、receiver sources/raw partial/Delta/final。
7. 结构测试：
   - 两preset及custom R=3；
   - operator/ln_1参数ID跨round相同；ln_2/MLP参数ID跨round不同；
   - 不同核心位置全部不同；
   - state keys不存在round-specific operator副本；
   - point FFN数=P+C*R+S；
   - 调用hook证明每个F[p,r]恰调用一次而O[p]调用R次；
   - second-round loss向共享O[p]和相应F[p,r]反传。
8. 数值测试覆盖六variant、三residual、FP64/FP32、dropout/RNG、AdamW一步、strict state roundtrip、错误cross-mode load。可用时运行CUDA FP32/FP16 AMP/BF16 AMP，特别复核RB dtype边界。
9. 参数解析式与实测逐项一致；相同拓扑/输入下，round_specific与v1有相同operator/FFN调用次数和主干MAC，只增加参数存储与优化器状态。不得声称延迟必然相同，需后续实测。
10. 不修改八任务生产入口、旧launcher或旧checkpoint。更新LF2报告/状态/冻结证据后停止。
```

## LF3：共享 latent FFN 原语与 LinearNO context 适配

```text
LF2已审查通过。现在只执行LF3：实现并独立验证latent FFN和六variant的context插入，不接生产任务CLI。

1. 先写独立LinearNO公式oracle：由现有投影参数显式计算features、Q/K/V、K^T V、head merge、latent FFN、head split、Q readout和to_out。oracle不能调用生产enhanced operator forward。
2. 实现LatentContextFFN：输入[B,h,M,d_h]或经明确adapter得到[B,M,H]，计算LN→Linear→GELU→Linear并与Z相加；无M轴混合、无dropout、eps1e-5、bias按规格。
3. 每个核心位置一套latent FFN，跨round复用同一module；不同核心位置参数ID不同；prefix/suffix无该module。
4. 第二Linear的weight/bias在所有通用初始化后零初始化。建立专门初始化函数/阶段，防止wrapper的model.apply再次覆盖。验证W1/LayerNorm首步零梯度、W2非零梯度属于预期。
5. 对plain/temp/conv/conv_temp/airfrans/shapenet逐类验证context位置。保留每类temperature/clamp、structured reshape、contiguous、to_out和dead参数语义。不得用hook缓存或第二次重算Q/K/V形成双倍operator。
6. mode=off/没有latent的v2 operator必须调用现有原operator或达到输出、梯度、RNG严格等价；不能维护两套略有差异的公式而缺少对照证据。
7. 零初始化下，将同权重round_specific与round_specific_latent逐项比较：输出、loss、输入梯度、全部公共参数梯度、RNG、operator中间Q/K/V/context/readout。应精确或满足事先固定的严格浮点预算。
8. 将W2设为非零后，与独立oracle比较forward/VJP/finite-difference，证明latent路径实际生效；检查所有活动参数有限且有梯度。
9. 测试latent token置换等变：同步置换Q的M轴以及context的M轴时结果相应不变；证明模块按token共享而非假定跨轮slot编号对齐。测试不同N/M、B>1、h>1及非法shape。
10. 记录新增MAC、参数和激活形状；断言没有[B,N,N]或[B,M,M]张量。更新LF3报告/状态后停止。
```

## LF4：两个新模式的完整 v2 wrapper、初始化与 checkpoint 闭环

```text
LF3已审查通过。现在只执行LF4：组装完整v2 Standard/Air/Car合成wrapper，完成两个新模式×三residual，不接真实任务parser/launcher。

1. 按LF0决定的版本化类路径组装v2模型。stem、位置、物理时间、placeholder、输出head和外部forward签名完全复用现有纯/loop实现。
2. round_specific不注册任何latent模块/key；round_specific_latent只注册C套latent FFN。两模式都注册C*R套核心ln_2+point MLP。
3. 实现公平初始化：
   - 公共stem/prefix/shared operators/round-specific FFNs/suffix/head/placeholder跨两个新模式逐位相同；
   - 公共backbone跨SR/RB/LB相同；
   - extra FFN和latent使用记录过的隔离seed；
   - latent构造/初始化不推进公共RNG；
   - W2在最终构造完成后仍严格为0。
4. 真实支持两个preset和custom P/C/R/S；R不硬编码为2。custom R变化时创建准确C*R套FFN和C套latent。
5. 完成v2 metadata-first state_dict/optimizer/scheduler/RNG/generator保存恢复。严格拒绝：
   - v1 checkpoint载入v2；
   - round_specific与round_specific_latent互载；
   - SR/RB/LB互载；
   - PCRS/M/width/variant/version不一致；
   - missing/unexpected key和shape冲突。
6. 对8任务profile×2preset×3residual×2mode×M1/M2做构造、参数/state key解析。用缩小张量完成代表性全矩阵forward/backward/AdamW/strict fresh-process reload；六variant都必须覆盖。
7. CUDA可用时运行FP32、FP16 AMP、BF16 AMP。不得只测SR；RB/LB和三类wrapper必须覆盖，保留LL9R canonical-dtype合同。
8. 重放旧v1 SR/RB/LB archive和旧纯/history档案，证明新代码没有改变旧输出、梯度、RNG、state keys或恢复结果。旧失败按LF0 baseline比较，不改golden。
9. 输出完整解析/实测参数和MAC表，分别标注synthetic规模与全profile参数；不得把小型GPU验收称为全宽训练。
10. 更新LF4报告、状态和冻结证据后停止。
```

## LF5：六个 Standard benchmark 的生产接入

```text
LF4已审查通过。现在只执行LF5：接入Airfoil、Darcy、Elasticity、Pipe、NS、Plasticity，不修改其数据、loss、time loop、metric或旧模型路径。

1. 只有显式新core_ffn_mode的train，或saved metadata明确为v2的新resume/eval，才进入新路径。省略新字段的旧v1/pure/history命令保持原行为和懒加载边界。
2. 将新字段、v2默认M1、latent宽度和初始化协议从parser显式来源传到config/model/metadata/run id。旧argparse默认不得伪装为用户显式输入。
3. eval/resume必须先读取sidecar恢复v2结构，再构造模型和torch.load。显式CLI只作一致性断言；不得用当前默认覆盖saved结构。
4. 复用当前loop checkpoint、记录、可视化和输出体系；新增字段完整进入architecture metadata。训练不得覆盖旧run；eval使用独立结果目录；sidecar不重写。
5. 保留六任务当前profile的H、heads、base M、variant、mlp_ratio、位置、T、normalizer、目标、loss、optimizer、scheduler、epoch和评估逻辑。NS仍10步协议，Plasticity仍原时间查询/更新节奏。
6. 为每任务生成两个新mode×三residual×两个preset的dry-run/train/resume/eval命令预览；默认M必须等于base profile，并提供显式M×2示例。
7. 使用内存合成数据运行原生短闭环：train→中断checkpoint→fresh-process resume→eval；比较连续训练与恢复后的权重、optimizer/scheduler、RNG/generator、批次和输出。明确不等于真实数据训练。
8. 运行旧六任务v1/pure/history/Transolver关键回归、AST/字节投影和新模式CPU/可用GPU AMP。任何新增失败不得归为“旧问题”而无证据。
9. 更新LF5报告、命令文档、状态和冻结证据后停止。
```

## LF6：AirfRANS 与 ShapeNet-Car 的生产接入

```text
LF5已审查通过。现在只执行LF6：接入AirfRANS和ShapeNet-Car，不运行真实VTK/数据训练。

1. 沿当前loop工业adapter、metadata、member/fold、normalizer和state_dict保存方式做最小版本化扩展。不要恢复旧whole-object猜测加载。
2. AirfRANS保持x/pos/surf/batch、采样、成员ensemble、MSE_weighted训练入口与官方评估合同；Car保持单图、x7、velocity3+pressure1、fold、surface/drag接口。
3. 只有显式v2 train或saved v2 metadata才import/构造新类；缺共享包的旧parser仍可工作。错误结构在读取大权重和创建输出目录前拒绝。
4. 新模式默认M=base 32；显式M×2仍可控制。ShapeNet actual M继续满足其head_dim整数倍率映射。
5. 每任务运行两个mode×三residual×两个preset的内存PyG合成闭环；Air至少覆盖两个member与中断恢复，Car覆盖非默认fold。fresh-process strict reload并比较完整状态/RNG/输出。
6. 可用GPU时覆盖FP32/FP16 AMP/BF16 AMP，特别检查placeholder导致的anchor dtype及RB local source转换，没有权威raw cache突变。
7. 重放旧v1工业archive及旧pure/history archive，文件字节不改，结果按既有严格标准一致。
8. 更新LF6报告、命令、状态和冻结证据后停止。
```

## LF7：统一启动器、成本工具、最终回归与交付

```text
LF6已审查通过。现在只执行LF7：完成八任务统一命令、参数/计算工具和最终无数据验收；不得运行真实训练。

1. 扩展现有tran_evaluate/linearno_loop启动器，使两个新mode均可train/resume/eval/then-eval/dry-run；旧v1命令输出、目录和解析保持兼容。用户参数最后覆盖，但结构冲突必须fail-fast。
2. 提供八任务×两个preset×三residual×两个新mode×3seed的正式命令矩阵；主命令默认M1，另给M2控制。PREVIEW/dry-run默认安全，不自动启动144/288个训练。
3. 扩展成本工具，实测并解析：总参数、各分区参数、state keys、unique/executed调用、MAC、forward和train-step median/p90、峰值显存。结果分清CPU小合成、GPU小合成和未运行的真实epoch。
4. 可选诊断默认关闭，不能保存有梯度的跨forward tensor。若记录，至少包括(p,r) FFN调用、共享operator调用、latent context/update范数与梯度；关闭诊断时输出/梯度/RNG必须不变。
5. 最终矩阵至少覆盖：
   - 8任务×2preset×3residual×2mode的全profile构造/参数/strict reload；
   - 六variant；
   - CPU forward/backward/AdamW；
   - 可用GPU FP32/FP16 AMP/BF16 AMP；
   - 六Standard与两工业原生合成闭环；
   - 旧v1 archive与旧pure/history关键archive；
   - parser缺共享包隔离、run目录、metadata冲突、错误cross-mode load；
   - Python compile、shell bash -n、git diff --check。
6. 运行LF0登记的完整旧回归清单。逐项报告pass/fail/skip/error并与LF0 baseline比较；不得删除测试、放宽容差、刷新golden或把新失败归入历史。
7. 冻结检查覆盖生产数学、任务协议、旧模型、旧证据和用户文件。只允许本任务登记的增量；provenance兼容投影必须精确且mutation-sensitive，不能泛化忽略新diff。
8. 生成：
   - docs/LOOP_LINEARNO_FFN_IMPLEMENTATION_REPORT.md
   - docs/LOOP_LINEARNO_FFN_COMMANDS.md
   - docs/LOOP_LINEARNO_FFN_PERFORMANCE.md
   - 最终requirements matrix与审计索引
   - 更新的独立FFN研究状态文档
9. 最终状态只有在所有授权无数据验收通过且无新增未解释回归时才能PASS；否则如实标PARTIAL/BLOCKED。明确NOT RUN：真实数据、完整epoch、收敛、精度、SOTA、真实吞吐、远端torch2.11/cu128、distributed/compile（若未测）。
10. 完成3--5项源码优先自审，列出用户最应复核的位置。不得commit/push。写“本LF7阶段结束，未执行真实实验”，然后停止。
```

---

# 四、最终验收清单

Codex 完成全部授权阶段后，至少应能证明：

1. 旧 v1 Looped LinearNO 的命令、config、metadata、checkpoint 和数值没有被新默认值重解释；
2. `round_specific` 和 `round_specific_latent` 能在八任务上分别 train/resume/eval；
3. 两个新模式均支持两个 preset、custom 和 SR/RB/LB；
4. 新模式默认 M 与任务原 LinearNO profile 一致，M×2只作为显式控制；
5. operator/ln_1跨轮共享，point FFN/ln_2按(p,r)独立；
6. latent FFN按核心位置独立、跨轮共享，只位于K^T V与Q readout之间；
7. latent token之间没有M×M混合，也不要求跨轮token对齐；
8. latent W2零初始化后两个新模式初始函数等价，公共参数/RNG公平；
9. latent激活后有真实梯度和非平凡作用，而不是未调用的注册参数；
10. RB AMP修复、SR/LB公式、receiver数量与来源完全保留；
11. 参数和MAC解析式与实际模型逐项一致；
12. 跨mode/version/topology/residual/rank checkpoint错误严格拒绝；
13. 旧pure LinearNO、linearno_history、Transolver及八任务科学协议未被破坏；
14. 已取消的stacked/Hourglass FFN方案没有代码、配置、文档或测试残留；
15. 所有报告严格区分合成验证与真实训练，不提前声称性能提升或SOTA。

