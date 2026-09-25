# Loop LinearNO v5：核心 LayerNorm 按访问独立的增量修改提示词

## 任务目标

请在 `https://github.com/hxh5159/CDLNO-w.git` 的当前工作树上，对已经实现的新版 Loop LinearNO 做一次局部修改。

目标模型当前应为 `partial_share_feature_gate_v5`，但开始前必须先检查当前仓库、分支、未提交修改、模型版本和实际文件位置，不要仅根据旧提示词推断代码状态，也不要覆盖用户已有修改。

本次唯一的模型计算改动是：

> 循环核心中，同一物理 block 的不同访问／轮次使用各自独立的 `LN1` 和 `LN2` 可学习参数。

其余模型结构、参数所有权、残差、训练评估和输出逻辑全部保持不变。

## 1. 归一化所有权

设 `p` 为循环核心中的物理位置，`r` 为该位置的访问编号。核心计算应为：

```math
S_{p,r}=\operatorname{LN}_{1,p,r}(X_{p,r}),
```

```math
Z_{p,r}=X_{p,r}+\mathcal O_{p,r}(S_{p,r}),
```

```math
U_{p,r}=\operatorname{LN}_{2,p,r}(Z_{p,r}).
```

随后继续使用现有 router、共享稠密专家和残差：

```math
\pi_{p,r}=\operatorname{softmax}_{E}(U_{p,r}W_g^{p,r}+b_g^{p,r}),
```

```math
G_{p,r}=\sum_e\pi_{p,r,e}\mathcal E_{p,e}(U_{p,r}),
\qquad
X_{\mathrm{next}}=Z_{p,r}+\frac{1}{R}G_{p,r}.
```

最终所有权规则如下：

- Prefix：每个 prefix block 保持自己原有的 `LN1/LN2`，不同 block 不共享。
- Core：每个 `(p,r)` 拥有独立的 `LN1[p,r]` 和 `LN2[p,r]`。
- Suffix：每个 suffix block 保持自己原有的 `LN1/LN2`，不同 block 不共享。
- 最终 `LN3`：全模型只保留一份，只在最终输出头前调用一次。
- 不增加 Q/K norm、router norm、专家内部 norm、stem norm 或其他归一化。
- 不改变 LayerNorm 类型、`eps`、`normalized_shape` 和 affine 设置。

这里的“独立”只指 LayerNorm 的 `weight=gamma` 和 `bias=beta` 是不同 Parameter。均值和方差仍在每次调用时根据当前输入实时计算，不得建立跨访问统计量。

## 2. 保留共享模式

为目标 v5 模型增加一个小型配置开关：

```text
core_norm_mode = visit_independent | shared
```

- 默认值为 `visit_independent`，作为新模型的正式配置。
- `shared` 保留修改前的语义：同一核心物理位置的所有访问使用同一对 `LN1/LN2`。
- 该开关只允许影响核心 `LN1/LN2`；prefix、suffix、最终 `LN3` 和其他参数都不能变化。
- 旧模型不应暴露或接受该参数。
- 两种模式都应能够训练、评估、保存和加载，但不需要为 `shared` 模式准备论文实验脚本。

建议 CLI 名称为：

```bash
--linearno-loop-core-norm-mode visit_independent
--linearno-loop-core-norm-mode shared
```

CLI、resolved config、metadata、checkpoint 和运行目录标识中都应记录该选择，避免两种模式的结果目录冲突。

## 3. 其他参数所有权保持不变

循环核心同一物理位置跨访问继续共享：

```text
in_project_x
to_v
to_out
其他非 Q/K operator 参数
全部稠密专家参数
```

按访问继续独立：

```text
to_q
to_k
有效的原生 Q/K temperature
feature router
```

本次只在按访问独立集合中加入：

```text
LN1
LN2
```

不得修改 Q/K/V/O、slicing/deslicing、router、专家、Softmax 轴、专家宽度、专家数量或任何其他参数所有权。

## 4. 初始化与实现要求

所有访问专属 LayerNorm 使用相同初值：

```math
\gamma_{p,r}=1,\qquad \beta_{p,r}=0.
```

参数对象和 storage 必须独立。这样新模型初始化时与共享归一化版本前向等价，但训练后可以逐访问分化。

实现时可以保留当前 `ln_1/ln_2` 作为 visit 0，并为后续访问注册额外的 `ModuleList`，由 `visit_index` 选择对应归一化。不要把同一个 LayerNorm 对象多次注册，也不要通过复制完整 block 实现。

Prefix 和 suffix 的 `visit_count=1`，不应创建多余副本。

## 5. 配置、记录和 checkpoint

同步更新目标 v5 自己的：

- config 与严格枚举校验；
- constructor kwargs 和 CLI；
- `loop_spec.state_partition`；
- 参数量统计；
- recording/hook 中的归一化 owner；
- checkpoint/config/formula 版本；
- run directory ID。

`visit_independent` 会增加 state keys，因此不得用 `strict=False` 静默加载旧共享 checkpoint。新 checkpoint 必须记录 `core_norm_mode`，模式冲突应在加载权重前报错。

当前仓库若没有不可替代的真实 v5 训练 checkpoint，则升级格式并让旧格式明确拒绝加载即可，不必实现自动迁移。若确实存在必须保留的旧 checkpoint，先报告证据，再提供显式的 shared-to-visit 权重初始化：把旧 `LN1/LN2` 的 gamma/beta 复制到每个访问副本，但不得复制 optimizer state，也不得把它称为严格续训。

## 6. 参数量要求

`visit_independent` 相比 `shared` 只能增加：

```math
\Delta P=4C\,C_{\mathrm{core}}(R-1),
```

其中 `C` 是隐藏维数，`C_core` 是核心物理 block 数。

- `p1_c3_r2_s1`：增加 `12C`，即 `C=128` 时 1,536，`C=256` 时 3,072。
- `p2_c2_r2_s2`：增加 `8C`，即 `C=128` 时 1,024，`C=256` 时 2,048。

LayerNorm 调用次数不变，矩阵乘法 FLOPs 不变，最终 `LN3` 仍只调用一次。

## 7. 必要验证

完成以下有辨识力的测试即可，不要扩展为大规模重构或真实 benchmark 训练：

1. `visit_independent` 下，同一核心位置不同访问的 `LN1/LN2` Parameter 和 data pointer 均不同。
2. `shared` 下，同一核心位置不同访问确实选择同一对 `LN1/LN2`。
3. 将两种模式的全部参数设置为相同数值后，初始化前向输出一致。
4. 只修改某一个访问的 norm 参数时，另一访问的 norm 参数保持不变。
5. 每个实际访问的 norm 都能得到有限梯度，并且 optimizer 中只注册一次。
6. 参数增量严格符合 `4C*C_core*(R-1)`。
7. 覆盖两个预设拓扑以及一个 `R=3` 的小型 custom topology。
8. 两种模式分别完成 save、strict load 和输出一致性验证；模式不匹配时明确失败。
9. 对 Standard、AirfRANS、ShapeNet-Car wrapper 做现有条件下的最小前向/反向冒烟。
10. 确认纯 LinearNO、Transolver、Loop v1-v4 的代表性行为没有变化。

## 8. 禁止修改

不得修改：

- operator、Q/K/V/O 和 slicing/deslicing；
- Q/K temperature；
- router 和稠密专家；
- 两条残差及 `1/R` 系数；
- prefix/core/suffix 调度；
- hidden width、rank、heads、专家数和专家宽度；
- stem、时间编码、placeholder 和输出 head；
- 数据、损失、optimizer、scheduler、AMP；
- checkpoint 保存频率、可视化、评估指标和结果输出格式；
- 纯 LinearNO、Transolver 和旧 Loop 模型。

本次唯一允许发生的模型数学变化是：

```text
循环核心的 LN1/LN2
从默认按物理位置共享
改为默认按 (物理位置, visit) 独立。
```

## 9. 交付报告

完成后简要报告：

- 修改前审计结果；
- 实际修改文件；
- `core_norm_mode` 的默认值和 CLI；
- LayerNorm 的最终所有权；
- 参数增量；
- checkpoint 兼容规则；
- 测试命令与结果；
- 未运行项目和事前已有失败。

最后明确确认：默认模式为 `visit_independent`，`shared` 模式仍可训练和评估，且此次没有改变目标模型的其他计算逻辑。
