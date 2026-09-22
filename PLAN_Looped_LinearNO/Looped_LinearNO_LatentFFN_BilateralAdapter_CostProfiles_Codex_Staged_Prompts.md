# 面向 `hxh5159/CDLNO-w` 的 Looped LinearNO：Latent FFN + 第二轮双端低秩适配分阶段 Codex 提示词

> 目标仓库：<https://github.com/hxh5159/CDLNO-w.git>
>
> 本文面向用户实际 checkout 中已经存在的 Looped LinearNO 实现继续开发。它不是重新实现 LinearNO，也不是覆盖现有 Looped LinearNO v1 或 Round-Specific-FFN/Latent-FFN v2。
>
> 生成本文时只读核对到的远端 `origin/main` 为 `c02e671506f706910e0a1d58f03c310abf188345`（commit `020`）。实施时必须以用户当前 checkout、当前 dirty 状态、当前 `AGENTS.md` 和实际源码为真值；该 SHA 只用于说明提示词形成时的审计背景，绝不能强制 checkout。

---

## 0. 如何使用本文

第一次交给 Codex 时，发送“总控提示词”和 `LAA0`。每完成一个阶段，审查报告、源码 diff 与证据后，再单独授权下一阶段。不要一次授权真实数据长训练，也不要让 Codex 自动 commit 或 push。

本文把新实验族暂称为：

```text
family                 = linearno_loop
architecture_extension = loop_linearno_latent_adapter_v3
```

如果 `LAA0` 发现仓库已有同名或更合适的版本字段，可以在不改变科学语义的前提下调整名称，但必须满足：

1. v1、v2、v3 可以无歧义并存；
2. 旧命令省略 v3 字段时绝不能被重新解释为 v3；
3. 旧 config hash、metadata、state-dict key 和 checkpoint 继续严格恢复；
4. v3 的 train/resume/eval 必须由显式配置或已保存的 v3 metadata 选择。

建议显式选择器使用类似：

```text
--linearno-loop-architecture operator_latent_adapter_v3
```

`--cost-profile` 不能单独充当模型版本选择器，因为它描述成本配置而不是 architecture。当前 version resolver 以是否存在 v2 的 `core_ffn_mode` 区分旧版本；v3 必须增加第三条显式且可由 metadata 恢复的分支。

本文后续使用阶段编号 `LAA0`--`LAA10`，以避免覆盖仓库已有的 `LL0`--`LL10` 和 `LF0`--`LF7` 证据。

文中的 `matched_v1` / `efficient_v1` 后缀只表示“成本配置表第 1 版”，**不是** Looped LinearNO architecture v1。实现中必须分别使用 `architecture_extension` 与 `cost_profile` 两个字段，不能据字符串 `v1` 猜模型族。

---

# 一、当前仓库事实与不可破坏边界

生成本文时，当前远端已经包含：

- `cdlno/linearno_loop/`：共享完整 block 的 Looped LinearNO v1，以及 `sr_1_over_r`、`rb_attnres`、`lb_attnres_1_over_r` 三种残差；
- `linearno_loop/`：v1 配置、schema、metadata、checkpoint 与状态合同；
- `cdlno/linearno_loop/v2/` 与 `linearno_loop/v2/`：已经完成的 Round-Specific Point-FFN / Latent-FFN v2；
- 六个 Standard benchmark、AirfRANS、ShapeNet-Car 的 loop train/resume/eval 路由；
- `tran_evaluate/linearno_loop/` 的统一启动、记录和 dry-run 设施；
- 可视化、训练记录、权重、完整恢复 checkpoint、严格 eval 目录和结果输出链路；
- RB AMP 接收器边界修复及当前三种残差的大量回归证据。

因此，本任务必须遵守：

1. **不得把 v3 写进 v1 或 v2 的旧语义中。** 新功能使用独立版本化配置、构造器和 checkpoint 路由。
2. **不得修改纯 LinearNO 数学作为实现捷径。** 优先在独立 v3 路径复用已经审计的原语；若必须抽取共享 helper，须先用修改前数值档案证明 v1/v2/纯 LinearNO 的 state keys、输出、梯度、初始化与 RNG 均未改变。
3. **不得撤销 v2。** v2 的 `round_specific` 与 `round_specific_latent` 仍能按原命令训练、恢复和评估。
4. **v3 不采用轮次 Point-FFN 解共享。** v3 的核心完整 block（`ln_1 + operator + ln_2 + point FFN`）按物理核心位置跨两个循环轮次共享。这一点与 v2 明确不同，也正是本文参数量与 FLOPs 计算的前提。
5. 原 Transolver、CDLNO、KCDNO、MSAR-LNO、纯 LinearNO、LinearNO-history、v1、v2、八任务数据与科学协议、旧输出目录及旧档案默认冻结。
6. 不准使用 `git reset/clean/stash/rebase`、覆盖用户修改、删除旧证据、刷新旧 golden、放宽旧容差或使用 `strict=False` 掩盖不兼容。

当前实现还有两个必须显式审计的接线点：`tran_evaluate/linearno_loop/recording.py` 的现有观察逻辑区分 v1/v2，不能让 v3 被误记成 v1；`tools/linearno_loop_accounting.py` 的现有成本分派也必须增加独立 v3 公式，而不是套用 v2 所有权。

当前六个 PDE 任务经 `PDE-Solving-StandardBenchmark/linearno_entry.py` 进入 `LoopStandardRun`，AirfRANS/Car 也由各自 LinearNO entry 延迟转入 loop Run；现有八个 task shell 只负责参数转发。v3 应最小化地扩展 versioned config/construction/checkpoint/dispatch/recording，而不是修改任务 `exp/main/train/data/loss/metric` 主体。

---

# 二、冻结后的 v3 科学规格

## 2.1 拓扑和 block 所有权

统一拓扑：

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

v3 的默认深度规则固定为：

```text
P = 2
S = 2
R = 2
C = (executed_depth - 4) / 2
```

例如：

```text
executed_depth = 12
P2 -> [C1,C2,C3,C4] x 2 -> S2
unique parameterized blocks = 2 + 4 + 2 = 8
executed blocks             = 2 + 4*2 + 2 = 12
```

必须把两个“深度”字段分开记录：

```text
unique_depth   = P + C + S
executed_depth = P + C*R + S
```

禁止用一个含混的 `depth` 字段同时指代二者。

第一版成本 profile 的规范对应关系为：

| 原始 LinearNO 比较深度 \(L_0\) | v3 executed depth | P | C | R | S | v3 unique depth |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12 | 2 | 4 | 2 | 2 | 8 |
| 12 | 20 | 2 | 8 | 2 | 2 | 12 |
| 16 | 28 | 2 | 12 | 2 | 2 | 16 |
| 32 | 60 | 2 | 28 | 2 | 2 | 32 |

所以在这些配对中，v3 与原始比较模型存储相同数量的完整物理 block，但核心 block 被多执行一次。

实现仍可保留显式 P/C/R/S 的 custom 构造能力，但：

- `matched_v1` 和 `efficient_v1` 只对上表四组规范拓扑有冻结配置；
- 非规范拓扑必须使用 `cost_profile=custom` 并显式给出 H 与 latent width；
- 不得把一个不在表中的深度静默四舍五入或映射到最近 profile；
- 双端适配第一版冻结为“第二轮专用”，因此 adapter 开启时要求 `R=2`；adapter 关闭时可保留已有合法 custom R。

按 executed depth 自动推导时还必须校验它是合法偶数，且 \(C=(D-4)/2\ge1\)。不得把 D=12/20/28/60 加入 v1/v2 的全局旧 preset 集合；这些 shorthand/profile 只属于 v3。

## 2.2 三种残差必须保持现有定义

v3 必须兼容：

```text
sr_1_over_r
rb_attnres
lb_attnres_1_over_r
```

不得改写现有公式、来源或时序：

- SR：核心 operator raw branch 与 point-FFN raw branch 各恰好乘一次 `1/R`，identity 不缩放；
- RB：每个 `(round, operator/point-FFN sublayer)` receiver 继续读取现有 anchor、completed raw round sums 和 current raw partial；没有普通 residual，没有 `1/R`；
- LB：轮内继续采用 SR 的 `1/R`，`Delta` 仍由实际 `Y-H` 计算，boundary/output AttnRes 不二次缩放；
- v3 的 latent FFN 是 operator 内部的 context 映射，不成为新的点域 AttnRes source；
- 双端适配是 Q/K logits 的第二轮增量，不成为新的 residual branch；
- latent FFN 内部 residual 与 adapter 均不再单独乘 `1/R`；SR/LB 只在现有位置缩放整个 operator raw output；
- 保留 RB 当前以 core-entry anchor dtype 在 receiver 调用边界构造本地同 dtype tuple 的 AMP 修复；不得改变 authoritative raw partial、autograd、SR 或 LB。

成本 profile 以 `sr_1_over_r` 为解析成本锚点。RB/LB 的 router 参数和 contraction 必须另行实测与报告，不能把 SR 匹配数字宣传为三种残差都严格相等。

## 2.3 LinearNO 基础算子

记号约定：本文件公式和成本表里的 \(H\) 都表示模型隐藏宽度，即代码中的 `n_hidden` / `hidden_width`；Standard benchmark 构造器里原有的空间网格 `H_grid/W_grid`（旧参数名可能恰好也是 `H/W`）是另一回事。实现、CLI、metadata 和测试中必须使用不含混的字段名，不得用成本表的隐藏宽度覆盖空间网格尺寸。

每次访问都从当前点状态重新计算：

\[
Q,K\in\mathbb R^{B\times h\times N\times M},\qquad
V\in\mathbb R^{B\times h\times N\times d_h},
\]

\[
C_{\rm raw}=K^\top V\in\mathbb R^{B\times h\times M\times d_h},
\qquad
Y=Q C.
\]

其中 \(H=h d_h\)。Q 沿 M 做 softmax，K 沿 N 做 softmax。不得缓存上一轮或上一物理时间调用的 Q/K/V/context；不得跨 NS rollout、Plasticity 时间查询、batch 或 ensemble member 保存激活。

## 2.4 跨轮共享的 latent FFN

对每个物理核心位置 \(p\)，注册一套 latent FFN \(\mathcal G_p\)，在两轮中共享，并在每次访问中执行：

\[
Z=\operatorname{MergeHeads}(C_{\rm raw})\in\mathbb R^{B\times M\times H},
\]

\[
\widetilde Z
=Z+W^{z}_{2,p}\operatorname{GELU}
\left(W^{z}_{1,p}\operatorname{LN}^{z}_{p}(Z)\right),
\]

\[
\widetilde C=\operatorname{SplitHeads}(\widetilde Z),
\qquad Y=Q\widetilde C.
\]

冻结细节：

1. 严格位于 `K^T V` 之后、Q readout 之前；
2. 每个核心位置一套，不同核心位置不共享，同一核心位置跨两个 visit 共享；
3. prefix/suffix 不增加 latent FFN；
4. 对 M 个 latent token 分别应用同一个通道 FFN，不混合 M 轴；
5. 不读取历史 latent，不假设两轮 latent token 按索引对齐；
6. `LayerNorm(H, affine=True, eps=1e-5) -> Linear(H,D_z,bias=True) -> GELU -> Linear(D_z,H,bias=True)`；
7. 无 dropout、无 M×M attention、无 N×N attention；
8. heads merge/split 必须保持现有 head 顺序并正确处理 contiguous/reshape；
9. 输入输出 shape、device、dtype 相同，不能在 AMP 下关闭整个 operator 的 autocast；
10. 第一 Linear 采用 release 普通 Linear 初始化，第二 Linear 的 weight 与 bias 在所有全树初始化之后归零。

必须提供独立开关：

```text
latent_ffn = on | off
```

`off` 时不注册 latent FFN 参数，也不执行其 FLOPs；不能只把输出乘 0 后仍保留模块。

## 2.5 第二轮专用的双端 Q/K 低秩适配

“双端”指 LinearNO 的分析端 K 与合成/重建端 Q 同时加入低秩 logits 增量，不是增加两个点域残差。

对核心位置 \(p\) 的第二轮 visit，令输入投影后的 per-head features 为：

\[
X\in\mathbb R^{B\times h\times N\times d_h}.
\]

分别注册可学习矩阵：

\[
A_{Q,p},A_{K,p}\in\mathbb R^{r_a\times d_h},\qquad
B_{Q,p},B_{K,p}\in\mathbb R^{M\times r_a}.
\]

这些适配矩阵在 head 之间共享，但 Q/K 两端及不同核心位置之间互不共享。第二轮 logits 为：

\[
L_Q=XW_Q^\top
+\lambda (XA_{Q,p}^\top)B_{Q,p}^\top,
\]

\[
L_K=XW_K^\top
+\lambda (XA_{K,p}^\top)B_{K,p}^\top,
\qquad
\lambda=\frac{\alpha}{r_a}.
\]

然后才按原 LinearNO 变体完整保留其温度、clamp 和 softmax 轴：

- `temp/conv_temp`：对合并后的 Q/K logits 使用原 `temperature_q/temperature_k`；
- `shapenet`：保留原拼写、clamp 范围和语义；
- `plain/conv/airfrans`：不得凭空新增有效温度；AirfRANS 的 release 惰性温度仍保持惰性；
- Q 仍沿 M softmax，K 仍沿 N softmax；
- V、`K^T V`、latent FFN、Q readout 和 `to_out` 的语义不变。

第一轮严格使用基础 Q/K：

\[
L_Q^{(1)}=XW_Q^\top,\qquad L_K^{(1)}=XW_K^\top.
\]

默认推荐配置：

```text
adapter_mode = bilateral_qk_lowrank_second_visit
adapter_rank = 4
adapter_alpha = 4
adapter_scale = alpha / rank = 1
```

初始化：

- A 使用独立、记录 seed 的 Kaiming/等价合理随机初始化；
- B 严格零初始化；
- 不额外加入可学习标量 gate，也不能同时把 A、B 都置零；
- B=0 保证初始化时 adapter-on 与 adapter-off 的函数等价；第一步 B 可获得梯度，而 A 的第一步梯度为零是预期的分阶段启动现象。
- AMP 下使用 autocast 兼容的线性代数；delta 在与 base logits 相加前必须具有兼容的 dtype/device。不得把整个 attention 或模型全局强转 FP32，也不得借此修改共享 `PointDepthAttnRes`。

必须提供：

```text
adapter_mode = none | bilateral_qk_lowrank_second_visit
```

`none` 时不能注册 A/B，也不能执行适配 FLOPs。latent FFN 与 adapter 必须独立开关，从而得到四个可训练/恢复/评估的消融：

| latent FFN | 双端适配 | 含义 |
|---|---|---|
| off | none | v3 窄宽度共享-block loop 基础消融 |
| on | none | latent-only |
| off | on | adapter-only |
| on | on | 完整模型；两套成本 profile 均以此为成本计算目标 |

## 2.6 M、heads 与 ShapeNet-Car

v3 默认保持当前 LinearNO/Transolver profile 的原始 M，不再翻倍：

| 任务 | M | heads |
|---|---:|---:|
| Airfoil | 64 | 8 |
| Darcy | 64 | 8 |
| Elasticity | 64 | 8 |
| Pipe | 64 | 8 |
| Plasticity | 64 | 8 |
| Navier--Stokes | 32 | 8 |
| AirfRANS | 32 | 8 |
| ShapeNet-Car | 32 | 8 |

旧 ShapeNet wrapper 为复现 release 的 `key_ratio` 语义，仍有 `M % d_h == 0` 的历史约束；这会拒绝 v3 的 H=208/200/192/... 配置。v3 必须把实际 M 作为独立的正整数传给已经支持任意 rank 的 LinearNO attention primitive，不能把 M 重新解释成 `key_ratio*d_h`。但严禁删除或放宽纯 LinearNO、v1、v2 的旧约束与旧 checkpoint 语义。

---

# 三、成本模型和两套冻结 profile

## 3.1 成本口径

下列数值是**矩阵 MAC/FLOPs 解析口径**：

- 1 MAC = 2 FLOPs，因此 FLOPs 比例与 MAC 比例相同；
- 包含 stem、time projection、完整 block 矩阵乘、`K^T V`、Q readout、latent FFN、第二轮适配与输出 head；
- 不包含 softmax、LayerNorm、GELU、逐元素 residual、索引/reshape 和 kernel launch；
- 以 SR 为锚点，不包含 RB/LB AttnRes router contraction；
- 不是实测延迟、epoch 训练时长、峰值显存或完整训练 FLOPs；这些必须独立测量；
- NS 数值是一次模型调用，不自动乘十步 rollout；Plasticity 不是二十次训练更新的总 epoch 成本；
- AirfRANS/ShapeNet-Car 的 N 是审计用代表点数，真实样本点数变化时应同时输出符号公式和实际 shape 计数。

设 latent FFN 开启，则每个物理核心位置增加：

\[
P_{\rm latent}=D_z(2H+1)+3H,
\]

在 R=2 时，全部核心的新增矩阵 MAC 为：

\[
\operatorname{MAC}_{\rm latent}=4BCMHD_z.
\]

双端适配开启且只在第二轮执行时，每个核心位置增加：

\[
P_{\rm adapter}=2r_a(d_h+M),
\]

全部核心的新增矩阵 MAC 为：

\[
\operatorname{MAC}_{\rm adapter}
=2BChNr_a(d_h+M).
\]

Codex 必须实现独立解析 oracle，并用实际实例参数分区、hooks 或 profiler 逐项核对。不得把本文表格直接复制成“实测结果”。

## 3.2 原始八任务成本锚点（原始 LinearNO，L=8）

| 任务 | 原 H | M | FFN ratio | 代表 N | 代表 B | 原参数量 | 矩阵 MAC/代表 batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Airfoil | 128 | 64 | 1 | 11271 | 4 | 1,765,889 | 90,883,573,248 |
| Darcy | 128 | 64 | 1 | 7225 | 4 | 1,766,145 | 58,266,099,200 |
| Elasticity | 128 | 64 | 1 | 972 | 1 | 585,217 | 812,809,728 |
| Pipe | 128 | 64 | 1 | 16641 | 4 | 1,765,889 | 134,184,503,808 |
| Plasticity | 128 | 64 | 1 | 3131 | 8 | 1,799,428 | 51,330,365,440 |
| Navier--Stokes | 256 | 32 | 2 | 4096 | 2 | 3,377,921 | 29,991,370,752 |
| AirfRANS | 256 | 32 | 2 | 32000 | 1 | 3,358,788 | 116,539,392,000 |
| ShapeNet-Car | 256 | 32 | 2 | 32186 | 1 | 3,852,420 | 133,036,839,936 |

这些数值已与当前仓库的 LinearNO 审计口径逐项复核相等。实施时仍须由当前源码重新实例化验证，不能假设未来 checkout 完全不变。

## 3.3 默认 profile：`matched_v1`

这是 train/eval 脚本的默认 profile。表中为完整模型 `latent=on + adapter=on(r=4,alpha=4)` 的 H 与 \(D_z\)：

| 比较深度 → v3执行深度 | Airfoil | Darcy | Elasticity | Pipe | Plasticity | NS | AirfRANS | ShapeNet-Car |
|---|---|---|---|---|---|---|---|---|
| 8 → 12 | 104/704 | 104/704 | 96/320 | 104/704 | 104/720 | 208/680 | 208/672 | 208/776 |
| 12 → 20 | 96/736 | 96/736 | 88/312 | 96/736 | 96/744 | 200/592 | 200/592 | 200/688 |
| 16 → 28 | 96/648 | 96/648 | 88/272 | 96/648 | 96/656 | 192/616 | 192/616 | 192/712 |
| 32 → 60 | 88/728 | 88/728 | 80/288 | 88/728 | 88/736 | 184/600 | 184/600 | 184/696 |

每格格式为 `H/D_z`。所有 H 都须能被 heads=8 整除。

对最主要的 8→12 设置，复核比例为：

| 任务 | 参数/原模型 | 矩阵 MAC/原模型 |
|---|---:|---:|
| Airfoil | 99.79% | 102.76% |
| Darcy | 99.79% | 102.94% |
| Elasticity | 99.92% | 98.28% |
| Pipe | 99.79% | 102.65% |
| Plasticity | 99.91% | 103.04% |
| Navier--Stokes | 100.14% | 100.09% |
| AirfRANS | 99.85% | 99.77% |
| ShapeNet-Car | 99.90% | 99.83% |

所以它应被准确称为“约匹配”，不能称为逐项严格相等。四个深度组的复核范围为：

| 原深度→v3执行深度 | 参数比范围 | 矩阵 MAC 比范围 |
|---|---:|---:|
| 8→12 | 99.79%–100.14% | 98.28%–103.04% |
| 12→20 | 99.78%–100.36% | 97.09%–103.65% |
| 16→28 | 99.75%–100.21% | **100.47%–104.99%** |
| 32→60 | 99.16%–100.12% | 95.79%–100.55% |

注意 16→28 的矩阵 MAC 下界是 100.47%，不是早期摘要中的 100.64%。

## 3.4 第二个必做 profile：`efficient_v1`

这是用户明确会执行的第二套配置；必须像 `matched_v1` 一样提供八任务 train/resume/eval 脚本，而不是只作为文档示例。

| 比较深度 → v3执行深度 | Airfoil | Darcy | Elasticity | Pipe | Plasticity | NS | AirfRANS | ShapeNet-Car |
|---|---|---|---|---|---|---|---|---|
| 8 → 12 | 96/512 | 96/512 | 88/256 | 96/512 | 96/512 | 192/512 | 192/512 | 192/512 |
| 12 → 20 | 88/512 | 88/512 | 80/256 | 88/512 | 88/512 | 184/512 | 184/512 | 184/512 |
| 16 → 28 | 88/512 | 88/512 | 80/224 | 88/512 | 88/512 | 176/512 | 176/512 | 176/512 |
| 32 → 60 | 80/512 | 80/512 | 72/256 | 80/512 | 80/512 | 168/512 | 168/512 | 168/512 |

每格仍为 `H/D_z`，完整模型仍默认 adapter rank=4、alpha=4。

最主要的 8→12 设置：

| 任务 | 参数/原模型 | 参数减少 | 矩阵 MAC/原模型 | 矩阵 FLOPs减少 |
|---|---:|---:|---:|---:|
| Airfoil | 79.13% | 20.87% | 88.76% | 11.24% |
| Darcy | 79.13% | 20.87% | 88.88% | 11.12% |
| Elasticity | 79.79% | 20.21% | 85.21% | 14.79% |
| Pipe | 79.13% | 20.87% | 88.69% | 11.31% |
| Plasticity | 78.71% | 21.29% | 88.80% | 11.20% |
| Navier--Stokes | 80.23% | 19.77% | 85.96% | 14.04% |
| AirfRANS | 80.26% | 19.74% | 85.73% | 14.27% |
| ShapeNet-Car | 77.03% | 22.97% | 85.64% | 14.36% |

即：

\[
\boxed{\text{参数量减少 }19.74\%\text{--}22.97\%}
\]

\[
\boxed{\text{矩阵 FLOPs 减少 }11.12\%\text{--}14.79\%}
\]

上述两个范围只对应 8→12 的完整模型，不能直接推广到更深设置。更深设置的复核范围为：

| 原深度→v3执行深度 | 参数减少范围 | 矩阵 FLOPs减少范围 |
|---|---:|---:|
| 12→20 | 17.29%–24.94% | 11.47%–16.63% |
| 16→28 | 19.08%–23.69% | 10.45%–14.88% |
| 32→60 | 19.04%–27.25% | 15.30%–19.23% |

## 3.5 `custom` profile

必须支持自定义实验，但不能让 profile 来源含混：

```text
cost_profile = matched_v1 | efficient_v1 | custom
```

- `matched_v1/efficient_v1`：H、\(D_z\)、M、heads、P/C/R/S 由版本化表解析；显式冲突值必须 fail-fast，不能静默覆盖；
- `custom`：用户显式给出至少 `H`、`latent_width`、P/C/R/S 或明确的 executed-depth 推导；heads、M、adapter rank/alpha 可显式调整并完整记录；
- eval/resume 先读已保存 metadata，再构造模型；不能用当前 profile 表重新猜测旧运行；
- 成本 profile 的 `matched/efficient` 只在 latent=on、adapter=on、rank=4、alpha=4、SR 解析口径下具有表中含义。做消融或换 residual 后必须输出实际参数与成本，不得继续贴“matched”结果数字。

---

# 四、初始化、公平性、checkpoint 与输出合同

## 4.1 公平初始化

构造顺序必须保证：

1. 公共 stem、prefix/core/suffix 完整 block、head 和 placeholder 按现有 release 顺序构造与初始化；
2. latent FFN 和 adapter 在公共树完成后，通过各自记录在 metadata 的派生 seed 和 `torch.random.fork_rng`（或有同等证据的方法）安装；
3. 新特性构造不推进全局 Python/NumPy/Torch/DataLoader RNG；
4. 同 task/profile/topology/H/M/public seed 下，四个消融的全部公共 state-dict tensor 逐位相同；
5. 三种 residual 的公共 backbone 逐位相同，差异只能来自现有 router 与所选新模块；
6. latent W2/B2 的 zero init 与 adapter B 的 zero init 不能被后续全树 `.apply(...)` 覆盖；
7. 不能为了配对实验而让不同物理核心 block 共享参数或同值克隆；它们仍是独立抽样、独立参数。

初始化等价测试至少覆盖：

- feature-off 与 v1 共享-block core 在相同 H/M/P/C/R/S/residual 下的输出、输入梯度、公共参数梯度；
- latent-on 初始化与 latent-off 的输出等价；
- adapter-on 初始化与 adapter-off 的输出等价；
- 完整 on/on 与 base off/off 的公共路径等价；
- 第一步 latent W2 与 adapter B 有有效梯度，latent W1/LN 与 adapter A 的首步零梯度被明确记录为预期，而不是误报缺梯度。

## 4.2 版本化配置与 strict checkpoint

v3 metadata 必须完整记录：

- family、architecture extension、schema/config/checkpoint version；
- task、原 profile、`cost_profile`、字段来源；
- comparator depth、P/C/R/S、unique depth、executed depth；
- H、heads、\(d_h\)、M、FFN ratio、variant、\(D_z\)；
- residual mode、latent on/off、adapter mode/rank/alpha/scale/apply visit；
- 公共 seed、latent seed、adapter seed、DataLoader generators；
- 模块所有权、参数分区、实际总参数和解析成本版本；
- 数据、normalizer、objective、evaluation、训练进度、optimizer/scheduler/scaler/RNG 和当前仓库已有的所有恢复字段。

加载规则：

1. metadata-first；在构造模型或 `torch.load` 大 tensor 前拒绝架构冲突；
2. 同一 v3 配置 `strict=True`；
3. v1↔v2↔v3、不同 cost profile、H、\(D_z\)、M、topology、residual、开关或 adapter rank 冲突必须明确拒绝；
4. 不使用键过滤、随机补权、`strict=False` 或把 v1/v2 猜测迁移为 v3；
5. 可另提供显式离线转换器的设计报告，但本任务不需要转换旧权重。

## 4.3 输出逻辑

v3 必须接入当前仓库已有的输出协议，而不是另建简化训练器：

- run directory 预留与防覆盖；
- 完整 resolved config、命令、环境、状态和 epoch JSONL；
- 当前各任务训练曲线、验证结果和最终评估格式；
- 当前 periodic/final visualization、PNG/PDF/NPZ 或任务既有产物；
- paired pure weights 与完整 resume archive；
- checkpoint cadence、final weights、eval 唯一子目录与索引；
- AirfRANS ensemble/member 隔离、ShapeNet-Car fold/whole-object 历史边界；
- resume 后 optimizer/scheduler/scaler/RNG/DataLoader 顺序连续；
- eval 不重写训练 sidecar、不重新拟合 normalizer、不猜 latest run。

run id 至少区分：task、v3、cost profile、P/C/R/S、residual、latent、adapter/rank、M、seed、config hash，避免两套必做配置或消融相互覆盖。

---

# 五、明确排除

本任务不实现：

- 修改或删除现有 v1/v2；
- v3 的 point FFN 轮次解共享；
- M 默认翻倍、每层不同 M、不同温度或不同粒度；
- 时间步 embedding、ACT、动态停机、任意测试时改变 R；
- history-conditioned compression、HCC、CKCR、LSAR、历史 latent bank；
- Q-only/K-only 生产模式、额外 gate 或关系损失；
- M×M latent self-attention、N×N attention；
- 稀疏专家、MoE++、Zero Expert、扩大点域 FFN、Hourglass FFN；
- Transolver loop 化；
- 改数据、split、采样、point order、normalizer、目标、loss、metric、训练预算或优化器以换取结果；
- 宣称通用加速、精度提升或 SOTA；
- 未经授权的真实数据下载、长训练、自动 sweep、commit 或 push。

---

# 六、总控提示词

```text
你将在用户当前的 hxh5159/CDLNO-w checkout 中，为已经完成的 Looped LinearNO 新增一个独立、版本化的 v3 实验族：共享完整核心 block + 跨轮共享 latent FFN + 第二轮专用 Q/K 双端低秩适配。

开始前必须完整阅读当前根 AGENTS.md、memory/current-state.md、docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md、docs/LOOP_LINEARNO_LL9R.md、docs/LOOP_LINEARNO_FFN_IMPLEMENTATION_STATUS.md、v1/v2实现与其报告、当前八任务入口/launcher/checkpoint/recording/visualization。沿真实 import、config、construction、forward、train、resume、eval 和输出链路审计。用户实际 checkout 是唯一真值，不得假定本文生成时的 SHA、文件路径或测试数量仍成立。

冻结研究规格：
1. 新 extension 使用独立 v3 schema/config/checkpoint/constructor 路由；v1/v2/纯LinearNO及其他模型不改语义。旧命令省略v3字段时保持原路由。
2. v3 的 P/C/R/S core 共享完整 block，包括 ln_1+LinearNO operator+ln_2+point FFN；不采用 v2 的 round-specific FFN。prefix/suffix保持完整独立block。
3. 默认 P=S=2,R=2；给定 executed depth，C=(D-4)/2。规范D=12/20/28/60分别比较原L=8/12/16/32。adapter开启要求R=2。
4. 三种残差 sr_1_over_r、rb_attnres、lb_attnres_1_over_r 的公式、来源、缩放、receiver时序及RB AMP边界完全保留。
5. 每个物理core位置一套跨轮共享latent FFN：KtV后、Q readout前，LN(H,eps1e-5)->Linear(H,Dz)->GELU->Linear(Dz,H)，tokenwise、不混M、W2/b2零初始化；开关off时不注册模块。
6. 第二轮专用双端适配：Q/K分别使用 A[r,d_h]、B[M,r]，delta=(X A^T)B^T*(alpha/r)，head间共享，先与base logits相加，再执行原温度和softmax；V不变。默认r=4,alpha=4，A随机、B零；off时不注册模块。
7. latent与adapter独立开关，四种消融均可train/resume/eval。两套成本profile的数字只针对on/on、r4/a4、SR矩阵口径。
8. M保持任务官方默认：五任务M64，NS/AirfRANS/Car M32。v3 ShapeNet把实际M与d_h解耦，但不得改变旧wrapper/checkpoint约束。
9. matched_v1是默认；efficient_v1也是用户确定会做的实验。两者必须按本文任务×深度H/Dz表解析，并准备八任务可直接训练、恢复、评估的脚本。custom允许显式调整但不得静默覆盖profile。
10. 新模块使用隔离seed；同配置四消融公共tensor逐位相同，公共RNG/DataLoader序列不受特性开关影响。零初始化必须在全树初始化后保持。
11. checkpoint必须metadata-first、完整冲突检查、strict=True。输出必须复用当前仓库的config/log/status/epoch记录、可视化、weights、完整resume archive、eval目录与结果逻辑。
12. 先写独立oracle和失败测试，再写生产实现。oracle不得调用被测forward或复制生产helper充当expected。
13. 解析成本分参数、矩阵MAC、非矩阵操作和router；1MAC=2FLOPs。真实延迟/训练时长/峰值显存独立测量，不得由MAC直接推断。
14. 禁止reset/clean/stash/rebase/commit/push，禁止覆盖用户文件、删除旧证据、修改旧golden/容差、strict=False、真实长训练或安装依赖。
15. 每个阶段建立 docs/loop_linearno_latent_adapter_audit/laaX/ 独立证据，更新独立状态文档，不覆盖LL/LF证据。报告必须区分PASS/PARTIAL/BLOCKED和passed/failed/skipped/not-run，列出修改文件、公式到代码、命令、数值误差、冻结区及3--5个优先复核点。
16. 若仓库事实与冻结规格冲突且会改变数学语义，完成不受影响的只读审计后停止，给出源码位置、两种选择和影响，不得自行改研究设计。
17. v3必须由显式architecture字段或saved v3 metadata选择；cost_profile本身不能选择模型版本。不得把D12/20/28/60写进v1/v2旧preset合同。

每个阶段结束后输出“本 LAAx 阶段结束，未执行下一阶段”，停止等待用户授权。
```

---

# 七、分阶段 Codex 提示词

## LAA0：当前仓库只读审计、数值基线与冻结

```text
执行 LAA0，只做审计、基线捕获和实施计划，不修改生产模型/入口/launcher/旧测试。

任务：
1. 记录当前HEAD、remote、branch、tracked/staged/untracked/ignored状态；建立包含内容hash和分类的起点manifest。不得清理或回滚任何文件。
2. 完整阅读根AGENTS、current-state、LL9R/LL10、LF0--LF7、v1/v2 prompt/report/evidence，生成reading ledger；明确历史陈述与当前事实的区别。
3. 沿八任务逐一画出当前纯LinearNO、v1、v2从parser/config到constructor/forward/checkpoint/resume/eval/visualization/output的真实调用链和稳定class path。
4. 审计v1共享完整block、v2共享operator/轮次FFN、三残差、RB AMP helper、latent primitive、rank/M处理、初始化顺序、placeholder、Air ensemble、Car fold/whole-object边界。
5. 捕获修改前数值档案：纯LinearNO、v1、v2；六attention variant；两个canonical topology+一个custom；三residual；CPU FP64/FP32，有GPU时再做有限FP32/FP16 AMP/BF16 AMP。保存state keys、参数hash、输出、输入/参数梯度、一步optimizer、RNG、strict reload。
6. 捕获当前八任务最小原生合成train→checkpoint→resume→eval→output/visualization接口基线；不得访问真实数据。若成本过大，使用仓库已有受控测试并明确范围。
7. 用独立公式复算本文全部baseline、matched_v1、efficient_v1表。必须至少逐项报告8→12八任务的精确参数和矩阵MAC，并检查更深表的H/Dz和范围；发现差异时停止在LAA0报告中，不得自行改表。
8. 提出最小v3文件/版本路由方案。优先复用审计过的LatentContextFFN、v1残差逻辑和输出设施，但不得让v1/v2导入v3或改变旧key。
9. 列出风险：ShapeNet实际M约束、温度顺序、AMP dtype、zero-init梯度启动、开关RNG公平、cost-profile冲突、metadata先读、变量N、RB/LB额外成本。
10. 特别核对当前versioning分派、Standard/Air/Car loop Run、`tran_evaluate/linearno_loop/{launch.py,entry.py,recording.py}`和成本工具。报告v3最小共享修改面；禁止通过编辑八个exp/main/train/data/loss/metric文件复制训练逻辑。

交付：
- docs/LOOP_LINEARNO_LATENT_ADAPTER_REFERENCE_AUDIT.md
- docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md（只写LAA0状态）
- docs/loop_linearno_latent_adapter_audit/laa0/ 下的manifest、reading-ledger、source-map、baseline-fixtures、cost-recalculation、test/regression日志和delivery-review
- 分阶段实际修改文件预算；不得提前创建空生产模块。

验收：生产源码、旧测试、旧golden、旧证据、任务协议字节不变；已解释所有现有失败/skip，不能把历史失败写成新回归，也不能虚报全绿。

本 LAA0 阶段结束，未执行下一阶段。
```

## LAA1：v3 纯配置、schema、cost profile 与解析 oracle

```text
仅执行LAA1。在LAA0审计基础上先实现无torch/无任务入口副作用的v3配置和metadata合同，不构造生产tensor模型。

要求：
1. 新建独立v3 config/schema/contracts命名空间；保留v1/v2所有默认、hash和导入行为。
2. 定义唯一显式v3 architecture选择字段；定义cost_profile=matched_v1|efficient_v1|custom、latent开关、adapter mode/rank/alpha、P/C/R/S、actual M、hidden_width、Dz及字段来源。cost_profile不得隐式选择v3。
3. matched/efficient严格写入本文四组深度×八任务H/Dz表；默认train profile为matched_v1，默认D=12、P2/C4/R2/S2、M=任务base、latent=on、adapter=bilateral_qk_lowrank_second_visit、r=4、alpha=4。
4. profile值不可被普通显式H/Dz静默覆盖。要覆盖必须选择custom；eval/resume由saved metadata恢复。规范profile只接受D12/20/28/60及对应P/C/R/S，非法组合fail-fast。
5. custom提供明确的H/Dz/topology/M/adapter配置；H%heads=0；adapter-on要求R=2；latent-off时Dz仍可记录为resolved profile事实但不得导致将来构造模块。
6. v3 ShapeNet配置允许实际M独立于d_h；这一规则只能存在于v3合同，旧v1/v2验证不变。
7. metadata完整记录第4.2节字段；run id包含所有会改变state_dict或数学的字段。
8. 实现独立参数/MAC oracle，分stem/time、prefix/core/suffix、point FFN、latent、adapter、head、router；同时列非矩阵操作。不得导入未来v3模型。
9. 生成机器可读配置矩阵：8任务×2 cost profiles×4规范深度×3residual×4消融×paired seeds；只preview，不训练。另生成必要custom负例。
10. 单元测试覆盖合法解析、所有冲突、JSON roundtrip、hash稳定、无torch导入、RNG零影响、v1/v2配置回归、本文表格逐项核对。

额外检查隐藏宽度字段与Standard原空间网格H/W不冲突；cost profile只解析`hidden_width`，原任务`grid_height/grid_width`保持profile事实。
不要扩展v1/v2全局旧topology preset表；v3在自己的版本空间解析executed-depth shorthand。resolved profile、model spec、constructor kwargs和日志里的有效hidden_width必须一致，不能只改constructor而让记录仍显示原128/256。

报告必须说明：cost匹配只对on/on+r4/a4+SR矩阵口径成立；RB/LB和关闭模块的实际成本另算；不能声称真实速度。

本 LAA1 阶段结束，未执行下一阶段。
```

## LAA2：独立 latent/adapter 原语与数学 oracle

```text
仅执行LAA2。先写完全独立的数学oracle和预期失败测试，再实现任务无关原语；不接生产wrapper/入口。

实现：
1. 可复用当前v2 LatentContextFFN的已审计实现；若直接复用，禁止修改其语义/key/init。若需独立v3 wrapper，必须证明数值等价。
2. 新增BilateralQKLowRankAdapter或等价原语：每core位置拥有Q/K各自A[r,d_h]和B[M,r]；head间共享；输出两个delta logits，shape[B,h,N,M]；无bias、无gate、无缓存。
3. 默认r4/alpha4；A合理随机初始化，B零；提供显式feature seed隔离构造。
4. 原语只计算delta，不自行执行温度、softmax、V/context、point residual或1/R，避免把variant逻辑复制进adapter。
5. AMP下输出与base logits可安全相加；不得把整个forward强制FP32。对不支持的dtype给明确错误或局部合理合同，不得静默detach。

独立oracle/测试：
- 小张量手算和einsum/matmul双重独立展开；
- Q/K两端、每core独立、head共享、batch/N隔离；
- forward、VJP、finite difference、输入/A/B梯度；
- B=0严格函数等价、B首步非零梯度、A首步零梯度、优化一步后A开始可学习；
- latent merge/split、token置换等变、无M轴混合、W2=0等价及首步梯度；
- 参数公式2*r*(d_h+M)及latent公式Dz(2H+1)+3H与实例逐项相等；
- CPU FP64/FP32、有限CUDA FP32/FP16/BF16（可用时），NaN/Inf和非法shape/rank/alpha；
- state_dict strict roundtrip、无跨forward状态、构造不推进公共RNG。

本 LAA2 阶段结束，未执行下一阶段。
```

## LAA3：v3 context-aware LinearNO attention

```text
仅执行LAA3。实现独立v3 attention路径，但不组装完整loop wrapper或任务入口。

要求：
1. 普通无特性路径必须直接委托或严格复用现有LinearNOAttention，不能维持第二套略有差异的“等价”公式。
2. active路径逐variant复用当前审计顺序：in_project_x→reshape/head layout→base q/k/v logits；第二轮且adapter-on时加入Q/K delta；再使用原variant温度/clamp；Q沿M、K沿N softmax；KtV→可选latent→Q readout→merge→to_out。
3. 第一轮绝不访问adapter；adapter-off不注册/不调用adapter；latent-off不注册/不调用processor。
4. latent在两个visit都执行；同core位置用同一实例；prefix/suffix调用纯native attention。
5. 保留conv/conv_temp的H*W检查、AirfRANS contiguous布局、ShapeNet温度拼写和所有现有dropout/惰性属性语义。
6. v3 actual M直接决定to_q/to_k输出；ShapeNet不再要求M%d_h==0，但只在v3构造器成立。
7. 不持久保存features/q/k/v/context；诊断若需要只能opt-in、detach并在forward后清除。

测试：
- 六variant、第一/第二轮、四消融的独立oracle对照；
- feature-off对现有attention输出/梯度/RNG严格等价；
- zero-init active对off初始等价；
- 非零B/W2后手算、梯度和每个模块影响均成立；
- 温度必须作用于base+delta整体，设计一个能区分“先温度后加delta”错误顺序的测试；
- Q/K softmax轴、M独立ShapeNet、conv reshape、AMP dtype、dropout RNG、strict reload；
- hook确认无N×N/M×M注意力和每visit恰一次Q/K/V/KtV/QC。

不得修改纯LinearNO/v1/v2 attention以使测试通过。

本 LAA3 阶段结束，未执行下一阶段。
```

## LAA4：v3 共享完整-block core 与三种残差

```text
仅执行LAA4。组装任务无关v3 loop core，不接八任务生产入口。

结构：
1. P与S使用现有完整native block；每个core位置只注册一套完整block并在R=2两轮复用，因此ln1/operator/ln2/point-FFN都共享。
2. 每core位置可注册一套跨轮共享latent FFN和一套第二轮adapter；两者通过LAA1开关决定是否真实存在。
3. core执行必须显式传递round_index：round0=base；round1按配置启用adapter。两轮均可启用同一latent processor。
4. 保留v1的SR/RB/LB公式和RB dtype helper；不得从v2复制round-specific FFN所有权。
5. 最后suffix head只执行一次；每visit重新计算Q/K/V/context；无activation cache。

测试/证据：
- module id/parameter id证明P+C+S完整block独立、core跨轮复用、latent/adapter按core共享且不同core独立；
- 两规范topology及custom（adapter-off可R3），unique/executed调用计数；
- 三residual完整独立oracle，不使用生产core/helper；逐round记录sources/raw partial/delta/scale/receiver；
- feature-off与同H/M v1 core数值、梯度、一步optimizer、dropout RNG等价；
- 四消融zero-init等价与非零后差异；所有公共keys逐位配对；
- adapter只在第二轮收到调用/梯度；latent两轮均调用；prefix/suffix零调用；
- RB FP32/FP16/BF16来源dtype和authoritative cache不变；SR/LB前后逐位或既定严格容差回归；
- 参数、state keys、解析MAC与hooks逐项一致；adapter/latent off无专属keys；
- strict reload与所有跨配置负例。

本 LAA4 阶段结束，未执行下一阶段。
```

## LAA5：三类 v3 wrapper、初始化与 checkpoint/output 基础接入

```text
仅执行LAA5。新增Standard、AirfRANS、ShapeNet-Car三类v3 wrapper及v3 checkpoint/output适配，但还不修改八任务真实生产选择分支。

要求：
1. wrapper完整复用当前纯LinearNO/v1已批准的输入、位置、reference distance、time embedding、placeholder、输出shape和单图校验；只把blocks容器换为v3 core。
2. Standard保留structured/irregular/temporal变体；AirfRANS保持x7/pos2、reference与ensemble接口；Car保持(data,geom)、x7、单图、输出[N,4]与fold语义。
3. H可按cost profile变窄；所有stem/time/head形状随resolved H正确构造。数据通道、目标和协议不变。
4. ShapeNet v3实际M独立于d_h；验证H=208,M=32等规范配置能构造。旧ShapeNet类和旧checkpoint限制不变。
5. 严格按第4.1节完成公共初始化、latent seed、adapter seed、placeholder顺序和四消融公共tensor配对。
6. v3 metadata/checkpoint必须先读sidecar再import/构造/加载；完整archive与纯weights配对、hash、optimizer groups、scheduler/scaler/RNG/normalizer/ensemble manifest校验复用现有设施。
7. 新输出适配使用当前记录器和可视化接口；不改任务绘图数学、不吞异常、不覆盖旧run。

生产接线优先沿当前`LoopStandardRun`、Air/Car loop Run与version dispatch扩展；不要创建平行训练框架。v3 checkpoint format必须独立于现有`linearno-loop-epoch-pair-v2`。

合成验收覆盖六attention variant、三wrapper、两cost profile的D12、三residual、四消融；执行forward/backward/AdamW、save→新进程strict resume/eval。旧v1/v2档案另做新进程replay，字节不改。

本 LAA5 阶段结束，未执行下一阶段。
```

## LAA6：六个 Standard benchmark 生产路由

```text
仅执行LAA6。把v3显式接入Airfoil、Darcy、Elasticity、Pipe、Navier--Stokes、Plasticity；不得改变原exp科学主体。

要求：
1. 只有显式v3 train或saved family/extension=v3的resume/eval才延迟import v3；旧parser在共享包不可用时仍能解析并运行旧模型。
2. profile解析从单一版本化表进入constructor，不在六个exp/脚本复制H/Dz。
3. 保留每任务实际数据路径、split、normalizer、batch、epochs、loss、gradient clip、optimizer/scheduler和eval metric；特别保留NS十步teacher-forced训练/预测反馈评估以及Plasticity二十个time-conditioned更新。
4. 默认v3为matched_v1+D12+P2C4R2S2+SR+latent on+adapter on/r4/a4；允许选择efficient_v1、其他规范深度、三residual和四消融。旧模型默认完全不变。
5. train创建唯一run并写完整resolved config；resume/eval先读saved config，显式冲突在权重加载前拒绝；输出接现有log/status/epoch/weights/archive/viz/eval逻辑。

测试每任务至少覆盖：两个cost profile的真实parser/dry-run；三residual和四消融配置；最小原生合成train→checkpoint→新进程resume→eval；任务输出/visualization hook；旧命令namespace与修改前AST projection。不得用简化MSE冒充任务原loss，也不得称合成输入为真实数据结果。

本 LAA6 阶段结束，未执行下一阶段。
```

## LAA7：AirfRANS 与 ShapeNet-Car 生产路由

```text
仅执行LAA7。把v3接入AirfRANS和ShapeNet-Car，保持工业任务原生训练/评估与输出边界。

AirfRANS：
- 保留实际weighted训练loss、原metric、sampling、radius graph、成员数/顺序和member独立checkpoint/RNG；
- 每个ensemble member独立v3模型、optimizer和resume状态；不得共享latent/adapter参数或激活；
- representative N=32000只用于成本报告，不改变真实采样。

ShapeNet-Car：
- 保留fold、单图、surface/drag接口、tuple输入与本地trusted whole-object历史边界；v3新档案仍优先使用版本化strict pair；
- v3 H与M解耦，验证matched/efficient各深度可构造；旧模型约束和档案不改；
- representative N=32186只用于成本报告。

两任务都必须完成两个cost profile、三residual、四消融的parser/dry-run，及受控PyG合成train/save/resume/eval。若本机缺torch_cluster，精确记录skip并运行不需要该依赖的边界测试，不能伪报通过。

本 LAA7 阶段结束，未执行下一阶段。
```

## LAA8：两套必做配置的八任务启动脚本与实验矩阵

```text
仅执行LAA8。交付用户可直接使用的两套八任务train/resume/eval脚本，不运行真实长训练。

脚本要求：
1. 为matched_v1和efficient_v1各提供八任务薄wrapper（共16个清晰入口）或功能等价且同样直接的目录布局；公共解析只写一处，不能复制16份易漂移命令。
2. 每个任务至少支持train、resume、eval、train_eval、dry-run/preview、print-run-dir；train成功后才允许then-eval同一run。
3. matched_v1是默认；两套脚本默认D12/P2C4R2S2、SR、M base、latent on、adapter on/r4/a4。用户可显式选择D20/28/60、RB/LB或四消融；custom必须显式H/Dz。
4. shell最后传用户override，但profile冲突由配置层fail-fast；路径含空格、GPU选择、seed、数据root、显式run和错误传播正确。
5. run目录和记录中明确cost profile、depth、residual、开关、adapter rank、M、seed和config hash。
6. 生成官方命令文档：八任务×两profile的train/resume/eval示例；adapter-off、latent-only、adapter-only、custom低成本示例；三paired seeds循环；远端路径覆盖。
7. 生成机器可读实验manifest，不自动启动矩阵。preview必须经过真实parser但不得读数据或权重tensor。
8. 扩展当前version dispatch和`recording.py`时必须显式识别v3 ownership/call schedule/state partition；不得让观察层把共享完整block误报成v2的round-specific FFN。旧v1/v2记录结果必须回归不变。

验收至少包括全部脚本bash -n、实际argv解析、运行目录唯一性、profile任务H/Dz逐项断言、saved-config恢复、冲突负例和旧launcher回归。

本 LAA8 阶段结束，未执行下一阶段。
```

## LAA9：成本、AMP、诊断和受控性能验收

```text
仅执行LAA9。不得访问真实数据或做完整epoch训练。

1. 对8任务×2profile×4depth×3residual×4消融实例化并核对解析/实测参数分区、state keys、unique/executed调用数和矩阵MAC；规模过大时可把纯解析全覆盖与tensor实例分层，但不得漏掉D12完整实例。
2. 对本文8→12表逐项给出精确计数和比例；复核matched/efficient更深范围。任何差异都标PARTIAL并定位，不得调整公式以迎合表格。
3. 成本报告明确SR锚点；RB/LB router参数/contraction另列；non-matrix操作另列；实际FLOPs profiler的不完整性另列。
4. 扩展`tools/linearno_loop_accounting.py`或其实际后继工具的v3独立分派，并用实例参数与shape trace交叉核对；旧v1/v2报告必须逐项保持。
5. 受控CPU/GPU小合成矩阵覆盖三residual、四消融、FP32/FP16 AMP/BF16 AMP、forward/backward/AdamW/strict reload。特别检查RB混合dtype和adapter logits dtype。
6. 有资源时，对少量canonical-N/full-width case测同步后的forward、train-step median/p90、peak allocated/reserved memory、checkpoint大小；不得据此推断真实epoch。
7. 可加默认关闭的detached诊断：latent update/base norm比、adapter delta/base-logit比、Q/K熵、各core/visit梯度、NaN/Inf。诊断开关不得改变forward/grad/RNG，退出后不保留图。
8. 新测试全通过后运行旧v1/v2/纯LinearNO/全仓回归；保留所有原始失败、skip和环境限制。不得改旧golden或把分模块重跑拼成“单次全绿”而不说明。

本 LAA9 阶段结束，未执行下一阶段。
```

## LAA10：最终源码审查、交付和边界声明

```text
仅执行LAA10，做最终审查与交付，不训练真实数据。

逐条检查：
1. v3确实共享完整core block，latent按core跨轮共享，adapter仅第二轮；没有意外round-specific point FFN、历史cache、M×M/N×N attention。
2. 三residual公式、RB AMP边界、温度/softmax轴、ShapeNet M独立、zero-init和RNG配对与冻结规格一致。
3. matched默认与efficient必做脚本在八任务可解析；D12/20/28/60拓扑和H/Dz正确；custom冲突严格。
4. 八任务原数据/训练/评估/可视化/checkpoint输出逻辑保持；新模型能train/resume/eval；旧模型与旧档案严格回放。
5. 汇总LAA0--LAA9的所有修改、测试、原始日志、失败/skip/not-run、环境和冻结证据。运行compile、shell syntax、git diff --check与最终回归。
6. 做一次逆向需求矩阵：本文每条冻结规格→源码→测试→证据；未验证项不得标PASS。
7. 写用户使用文档和最终实施报告，含两套profile命令、四消融、三residual、成本口径、checkpoint兼容表、已知限制和真实实验下一步。

最终状态只能是PASS、PARTIAL或BLOCKED之一。PASS只代表授权的无真实数据实现/验证完成，不代表收敛、精度、加速或SOTA。明确列出NOT RUN：真实数据、完整训练、三seed收敛、SOTA、真实epoch时长、远端栈、distributed/compile（若未跑）。

本 LAA10 阶段结束，未执行真实实验。
```

---

# 八、交付前的人工复核清单

把本文交给 Codex 前后，优先人工确认以下五点：

1. **模型族没有混淆：** v3 是共享完整核心 block；v2 才是轮次 Point-FFN 解共享。成本表不能套到 v2。
2. **适配顺序正确：** `base logits + delta logits` 之后才进入原温度和 softmax；adapter 只在第二轮。
3. **profile 含义准确：** matched 是约匹配；efficient 的 19.74%–22.97% 参数下降和 11.12%–14.79% 矩阵 FLOPs下降只对应 8→12、完整 on/on、r4/a4、SR解析口径。
4. **ShapeNet 没有破坏旧档案：** 仅 v3 将实际 M 与 `d_h` 解耦；旧纯模型/v1/v2约束仍在。
5. **输出和恢复不是附属功能：** 两套profile、四消融和三残差必须沿当前八任务同一记录、可视化、weights、完整checkpoint、resume/eval链路工作。
