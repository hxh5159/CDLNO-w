# KCDNO：全 LRSA 骨架与可缓存核化跨深度读取

版本：v1，2026-09-15。本文是一份待实现的新模型规格，不表示已经修改或验证了用户远端的 CDLNO 仓库。

建议工程名称为 **KCDNO（Kernelized Cross-Depth Neural Operator，核化跨深度神经算子）**，模型注册键为 `kcdno`。名称用于区分实验，不要求重命名已有 CDLNO 项目，也不宣称该名称已经做过全球查重。

本文延续此前确认的“每层重新压缩、重建，以可缓存的全历史读取替代 latent 自注意力”的候选。对尚未明确的数值细节，本文提出明确的首版默认值，尤其是核特征维数、核特征初始化和历史注入系数；这些是本次的工程选择，不应倒写为原 CDLNO 已有设置。

## 1. 模型边界与研究问题

新模型的每层都采用：

> Down → latent FFN1 → 全历史核化读取与深度融合 → latent FFN2 → Up → 点域 FFN／ConvFFN。

总层数默认 L=8。每层都有自己的压缩和重建，每层读取此前所有 block 的 latent 历史。第一层没有历史，所以直接跳过历史读取。整个 latent SA 残差子层被移除，包括它专属的 norm 和 Q/K/V/O 参数。

两个 latent FFN 保留；Down 和 Up 仍是标准多头 cross-attention。因而“移除层内注意力”在本文中的准确含义是：**移除每层压缩后、重建前的 latent self-attention**，不是删除所有 attention，也不是将全模型改成逐点 MLP。

新模型没有 persistent latent 后段、Bridge、IPOT processor 或额外 final feature decoder。最后一层正常 Up 回点域并完成点域 FFN 后，接对应任务的输出 norm/head。每个 forward 共 L 次 Down 和 L 次 Up。

研究假设是：保留各层重新观察点域的能力，并让当前压缩表示检索过去不同深度形成的摘要，能否用较小的 latent 交互参数预算替代逐层 latent SA。准确率优势、收敛优势和延迟优势均需实验验证。

既有 CDLNO 的 `full`、`no_sa`、`identity`、原 CDPA 各模式与对应 checkpoint 保持原定义。本规格不是覆盖这些模型的补丁。

## 2. 符号与首版统一尺寸

| 符号 | 含义 | 首版约定 |
|---|---|---|
| B | batch size | 沿用任务训练设置 |
| N | 每个样本的空间输入点数 | 保持原任务格式 |
| L | 完整点域 block 数 | 8，可配置正整数 |
| d | 点特征及 latent 通道宽度 | 128 或 256 |
| M | 每层 latent token 数 | 同一模型各层一致，32 或 64 |
| h | Down／Up 的多头数 | 4 或 8 |
| d_h | Down／Up 每头宽度 | d/h |
| r | 新核读取的总特征维数 | 16，后续比较 32 |
| X_l | 第 l 层之后的点特征 | [B,N,d] |
| T_l | 第 l 层 FFN2 后、Up norm 前的 latent | [B,M,d] |

r 不是 token 数，不是 FFN expansion ratio，也不是每个 Down head 的 rank。首版核读取为单组核特征，总维数就是 r；Down／Up 继续使用原来的 h 个 heads。

当前主配置使用相等的点宽、latent 宽和 attention inner width。若以后做不等宽实验，应分别定义 D、C、A，不得通过广播或 reshape 隐式混淆。首版无需为了迁移所有 LRSA YAML 而增加不等宽实验。

## 3. 一个 block 的完整计算

第 l 层输入为 X_{l-1}。先保留点残差，并计算入口 norm：

\[
H_l=\operatorname{RMSNorm}_{p,l}(X_{l-1}).
\]

### 3.1 Down：当前点域重新压缩

每层独立学习查询 P_l，形状为 [M,h,d_h]，直接作为投影后的 Q 参数。K/V 来自 H_l 的独立投影，输出经过 W_O 映射回 d 通道：

\[
S_l=\operatorname{Down}_l(H_l;P_l)\in\mathbb R^{B\times M\times d}.
\]

Down 内部保留已确认 LRSA 原语的 per-head Q/K RMSNorm、scaled dot-product softmax 与输出投影。softmax 沿 N 个输入点计算。此处不额外增加 P_l 残差，不再给已经投影后的 learned query 额外增加 W_Q。

Down 与 Up 的参数解耦，不要求互为转置、互为逆或正交。每层 learned query、Down／Up 及 FFN 参数均独立。

### 3.2 第一次 latent FFN

\[
U_l=S_l+\operatorname{FFN}_{1,l}
\bigl(\operatorname{RMSNorm}_{1,l}(S_l)\bigr).
\]

默认普通 GELU FFN：Linear(d,2d) → GELU → Linear(2d,d)。它逐 token 处理通道，使当前压缩结果先经过非线性变换，再生成历史查询。

### 3.3 核化历史读取的位置

历史读取位于 FFN1 与 FFN2 之间，输出记为 \(\widehat U_l\)。完整公式见第 4 节。第一层严格令 \(\widehat U_1=U_1\)。

原来的 SA 残差子层整体删除。不能留下 `U + Identity(Norm(U))`，因为那会产生 `U + Norm(U)`。新历史模块有自己的 norm、query 和融合参数，不是继续计算已删除 SA 的 Q/K/V 后弃用它们。

### 3.4 第二次 latent FFN 与历史定义

\[
T_l=\widehat U_l+\operatorname{FFN}_{2,l}
\bigl(\operatorname{RMSNorm}_{2,l}(\widehat U_l)\bigr).
\]

FFN2 与 FFN1 同样采用 hidden=2d 的普通 GELU FFN，参数独立。FFN2 加工当前和历史融合后的信息。历史取这里的原始 T_l；不能改取 Down 输出、FFN1 输出、SA 输出、Up norm 后的 latent 或整层点特征。

### 3.5 Up 与点域更新

\[
\Delta X_l=\operatorname{Up}_l
\left(Q\text{ 来自 }H_l,\;K,V\text{ 来自 }\operatorname{RMSNorm}_{up,l}(T_l)\right),
\]

\[
V_l=X_{l-1}+\Delta X_l,
\qquad
X_l=V_l+\operatorname{PointModule}_l
\bigl(\operatorname{RMSNorm}_{out,l}(V_l)\bigr).
\]

Up 使用独立 Q/K/V/O 投影和 per-head Q/K norm，softmax 沿 M 个 latent 计算。Query 来自本层入口点特征，保持 LRSA 的输入特征条件重建。若现有 Up 内部已经对 latent 做 norm，外部不重复归一化。

规则网格 PointModule 为已确认的 LRSA 风格 ConvFFN：普通 dense Conv2d(d,d,3,stride=1,padding=1,groups=1) → channels-last LayerNorm → Linear(d,2d) → GELU → Linear(2d,d)。外部残差如上。内部 LayerNorm 与外层 RMSNorm 都保留。

Elasticity、ShapeNet-Car、AirfRANS 使用逐点 GELU FFN，hidden=2d。latent 上不做卷积。结构化任务保持原 H/W 和点序，不通过 sqrt(N) 猜网格，不对弯曲网格重新按物理坐标排序，不加入下采样或时空 3D 卷积。

## 4. 可缓存的全历史核化读取

### 4.1 历史写入：参数属于产生历史的层

每个源层 s<L 有自己的 key norm 和 W_{k,s}∈R^{d×r}：

\[
K_s=\phi\left(\operatorname{RMSNorm}_{k,s}(T_s)W_{k,s}\right)
\in\mathbb R^{B\times M\times r}.
\]

将其写为两个摘要：

\[
\boxed{\mathcal M_s=K_s^\top T_s\in\mathbb R^{B\times r\times d}},
\qquad
\boxed{b_s=K_s^\top\mathbf1\in\mathbb R^{B\times r}}.
\]

每个历史在当前 forward 中只写入一次。这里 value 就是原始 T_s，没有额外 W_V 或 W_O。

**可缓存的关键在参数归属：** W_{k,s} 属于源层 s，不随接收层 l 改变。所有接收层用自己的 query 读取同一份摘要。若把 W_K 改成每个接收层各自投影历史，便不再满足这里的跨接收层缓存与成本公式。

### 4.2 当前查询：参数属于读取历史的层

接收层 l>1 有自己的 query norm 和 W_{q,l}∈R^{d×r}：

\[
Q_l=\phi\left(\operatorname{RMSNorm}_{q,l}(U_l)W_{q,l}\right)
\in\mathbb R^{B\times M\times r}.
\]

Q_l 在本层只计算一次。对每个历史 s=1,…,l−1：

\[
\boxed{
R_{ls}=\frac{Q_l\mathcal M_s}{Q_lb_s+\varepsilon}
\in\mathbb R^{B\times M\times d}
}.
\]

分母为 [B,M,1] 并沿通道广播。对应的逐 token 权重由正值核 \(q_{lm}^\top k_{sn}\) 产生。这里没有对 M×M 的 logits 计算 softmax；它是归一化正值核注意力，不是标准 softmax attention 的无损加速实现。

即使各层 latent 的索引身份不同，当前 token 也能按内容读取源层任意 token。不要求第 m 个历史 token 必须与第 m 个当前 token 对齐。

### 4.3 按深度选择来源

当前表示作为额外恒等来源：

\[
R_{l0}=U_l.
\]

接收层 l 用一份评分向量 w_l∈R^d 和一份 depth RMSNorm，对全部候选评分：

\[
e_{l,b,m,s}=w_l^\top
\operatorname{RMSNorm}_{depth,l}(R_{ls,b,m}),
\]

\[
\alpha_{l,b,m,s}=\operatorname{softmax}_{s=0,\ldots,l-1}(e_{l,b,m,s}),
\]

\[
C_{l,b,m}=\sum_{s=0}^{l-1}\alpha_{l,b,m,s}R_{ls,b,m}.
\]

权重形状是 [B,M,l]；每个样本、每个当前 latent 分别选择来源。softmax 确实存在，但只沿来源深度计算。归一化仅用于评分，参与 value 融合的是原始候选 R。

不能将全部历史先求和成一份摘要，因为那会丢掉逐来源选择；也不能把不同 chunk 分别 softmax 后再平均。

### 4.4 首版采用小幅历史注入

\[
\boxed{\widehat U_l=U_l+\gamma_l(C_l-U_l)}.
\]

本次建议明确采用：每个接收层一个可学习标量 γ_l，初始化为 0.1；训练中不限制其取值。w_l 初始化为零，depth RMSNorm scale 初始化为 1。

因此初始化时 α=1/l，但实际融合后的当前来源系数为：

\[
1-0.1+\frac{0.1}{l}=0.9+\frac{0.1}{l},
\]

每份历史系数是 0.1/l。例如第 8 层当前系数为 0.9125，其余 7 份历史各 0.0125。这样避免随着历史增加，初始当前表示直接被稀释为 1/l。

这是显式融合系数，不是完整 Jacobian 系数；候选和 α 本身也依赖输入。它不是严格恒等初始化，也不是原 CDLNO 不带 gate 的融合。训练后无约束 γ 不保证凸组合。若需要固定凸组合约束，必须另做 sigmoid gate 实验，不能暗改当前定义。

不为每个 R_ls 再加 U_l，也不在上述公式之后再额外加一次当前残差。γ=0 可得到不使用历史的计算，但历史分支内部参数首步梯度会为零；所以默认采用小非零值。

### 4.5 张量例子：第 4 层

设 B=2、M=64、d=128、r=16。第 4 层拥有 3 份历史：

| 张量 | 形状 |
|---|---|
| 当前 U_4 | [2,64,128] |
| 每份缓存 M_s | [2,16,128] |
| 每份缓存 b_s | [2,16] |
| 当前核查询 Q_4 | [2,64,16] |
| 每份读取 numerator | [2,64,128] |
| 每份读取 denominator | [2,64,1] |
| 全历史读取 R_41,R_42,R_43 | [2,3,64,128] |
| 加入当前候选后的来源集合 | [2,4,64,128] |
| 来源分数／权重 | [2,64,4] |
| 融合输出 C_4、Uhat_4、FFN2 后 T_4 | [2,64,128] |

这里没有生成 [2,64,64] 的 token attention 矩阵。来源维度为 4，不是 3，因为当前表示也参与来源选择。

## 5. 因果顺序、缓存生命周期与执行方式

概念性伪代码如下；这是架构语义，不是对现有类名的断言：

```python
X = task_lift(task_inputs)
history = []                         # 本次 forward 的局部状态
for l, block in enumerate(blocks):
    H = block.point_norm(X)
    S = block.down(H)
    U = S + block.ffn1(block.norm1(S))

    if history_enabled and history:
        U_hat = block.reader(U, tuple(history))
    else:
        U_hat = U

    T = U_hat + block.ffn2(block.norm2(U_hat))
    V = X + block.up(H, T)           # latent norm 在 Up 内执行一次
    X = V + block.point_module(block.output_norm(V))

    if history_enabled and l + 1 < len(blocks):
        history.append(block.writer(T))

return task_head(X)
```

第 1 层不创建 reader 的 query/scorer/gate；第 L 层不创建无人读取的 key writer。L=1 时仍能执行 Down—FFN1—FFN2—Up—PointModule，但不创建任何历史参数。

当 L≥2 时，写摘要 L−1 次，生成当前 Q L−1 次，读取历史总次数 L(L−1)/2。L=8 对应 7 次写入、7 次 Q 生成、28 份历史读取。每层读取自身之前的全部 block，没有窗口、采样或相邻层限制。

缓存中的摘要保留计算图；训练不能 detach 或放在 no_grad 中。缓存不跨 batch、样本或 forward 保存，不注册为会被 checkpoint 持久保存的状态。不要原地改写已经被其他层读取的张量。activation checkpoint 应传入历史 tuple 快照，不捕获仍在增长的列表。

默认在一个接收层把各来源摘要 stack 后批量矩阵乘法，保留来源维度；这减少逐来源 Python 调度，不改变数学。可保留逐来源参考实现用于核对。若以后按来源分块执行，最终深度 softmax 仍必须全来源归一化。首版 L=8 不必先实现复杂的流式 softmax。

## 6. 数值、初始化与可训练性约定

| 项目 | 首版设置 |
|---|---|
| 核映射 | φ(t)=ELU(t)+1，FP32 后 clamp_min(1e−6) |
| 核分母 | 加 ε=1e−6 |
| query／key norm | 独立 RMSNorm，eps=1e−6，scale=1 |
| kernel query／key Linear | d→r，bias=False，Xavier uniform gain=1 |
| depth norm | RMSNorm，eps=1e−6，scale=1，无 bias |
| depth scorer | 每个接收层一个 w[d]，初始全零 |
| 历史注入 | 每个接收层一个可学习标量 γ，初始 0.1 |
| 新机制 dropout | 0 |
| value／output 投影 | 不额外添加 |
| latent／point FFN | plain GELU，hidden=2d |

训练／推理生产路径的核特征正值化、摘要乘法和累加、分母、除法、来源评分/softmax/融合使用 FP32，并避免 autocast 把这些关键乘法重新降精度。输出在进入 FFN2 前转回 U 的 dtype。数学参考 helper 允许保留 float64，用于前向及梯度等价核对；生产路径另按 FP32 容差验证，不将强制降精度的函数直接用于严格 double gradcheck。固定为 sum 摘要，不悄悄换成 mean 后继续使用相同 ε。

理想 ELU+1 对有限实数为正，但有限精度可能舍入到 0；clamp 和分母 ε 是明确的数值约定。这个实现没有将指数 softmax 核作无偏随机近似的保证。

旧的 Down／Up／FFN 初始化沿用已定 CDLNO front 规范：普通 Linear trunc_normal(std=0.02)，bias=0；Conv 采用已有 PyTorch 默认初始化；norm scale=1、bias=0；learned Down query reshape 为 [M,d] 后 orthogonal。新 Wq/Wk 的 Xavier 初始化是本次核模块特定设置，避免投影过小导致所有正值特征起初几乎相同。不要用 wrapper 的递归初始化覆盖新模块的 Wq/Wk、w 或 γ。

若用户远端已经确认过与文档不同的初始化，先输出差异；新模型与 matched LRSA 对照的公共部件应一致，而旧模型不得被自动重置。

新机制可仅用现有 PyTorch 算子实现，不需要引入 LRSA 包的 xformers／liger 依赖，也不需要引入 IPOT 官方 processor。继续兼容用户 CUDA 12.8／Torch 2.11 环境；本次不更换已可训练的 PyG 安装方案。是否通过该远端环境测试，需要实际环境证据。

## 7. 各任务首版超参数

主 profile 建议命名 `kcdno_v1`，继承此前 CDLNO v1_2 的任务宽度、head、M、输入提升和点域模块，统一改成 8 个本规格 block。这不是逐任务照搬 LRSA 当前 YAML。

| 数据集 | L | d | Down/Up h | M | kernel r | latent FFN hidden | 点域模块 |
|---|---:|---:|---:|---:|---:|---:|---|
| Darcy | 8 | 128 | 8 | 64 | 16 | 256 | ConvFFN |
| Elasticity | 8 | 128 | 8 | 64 | 16 | 256 | PointFFN |
| Airfoil | 8 | 128 | 4 | 64 | 16 | 256 | ConvFFN |
| Pipe | 8 | 128 | 4 | 32 | 16 | 256 | ConvFFN |
| Navier–Stokes | 8 | 256 | 8 | 64 | 16 | 512 | ConvFFN |
| Plasticity | 8 | 128 | 8 | 64 | 16 | 256 | ConvFFN |
| ShapeNet-Car | 8 | 256 | 8 | 64 | 16 | 512 | PointFFN |
| AirfRANS | 8 | 256 | 8 | 64 | 16 | 512 | PointFFN |

点域 FFN 的 hidden 也为 2d；所有 block 的 M 一致。RoPE、额外 attention gate、mass weight、额外 positional embedding 均维持既有主版关闭设置。这里采用五个结构化任务的 ConvFFN，是既有 CDLNO 选择的延续；不是声称 LRSA 官方在这五个任务中都启用了卷积。

为了与原 Transolver 更接近地匹配 L/d/h/M，另记录 `transolver_shape_match` profile，仅覆盖新模型对应项：Airfoil h=8；Pipe h=8、M=64；NS、ShapeNet-Car、AirfRANS M=32。其余保持上表，r=16。该 profile 不修改旧 CDLNO 默认配置。

即使匹配 L/d/h/M，两种模型的 FFN 数量、hidden、投影和卷积仍不同，不能称为等参数或等 FLOPs 比较。

配置依据的已核查快照：LRSA `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`；Transolver `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。LRSA 论文表 9 与当前官方训练链在 Darcy、NS、AirfRANS 等存在差异，而且有些任务 latent width 不等于 point width。因此不应直接把 parser 默认值或一个 YAML 当作全部官方实验设置。

下一步容量实验优先单独比较 r=16/32；再考虑改变 M 或 FFN 宽度。首版不要同时更换激活、拓宽全部通道并改变层数，否则无法分离跨深度机制的贡献。

## 8. 数据与训练协议保持

仅替换模型家族及其架构，保留每个任务已有数据加载、下采样、normalizer、loss、训练循环、评估、mask、物理量计算和 rollout 调用方式。以下维度来自原脚本的常用设置，是核查合同，不是在模型内部硬编码 N。

| 任务 | 单次模型输入约定 | 单次输出 | 空间组织 |
|---|---|---|---|
| Darcy | x[B,N,2]，fx[B,N,1] | [B,N,1] | N=7225，85×85 |
| Elasticity | x[B,N,2]，fx=None | [B,N,1] | N=972，不规则 |
| Airfoil | x[B,N,2]，fx=None | [B,N,1] | N=11271，221×51 |
| Pipe | x[B,N,2]，fx=None | [B,N,1] | N=16641，129×129 |
| NS | x[B,N,2]，fx[B,N,10] | [B,N,1] | N=4096，64×64 |
| Plasticity | x[B,N,2]，fx[B,N,1]，T[B,1] | [B,N,4] | N=3131，101×31 |
| ShapeNet-Car | 原 (cfd_data,geom_data) 调用及七通道输入 | [N,4] | 保留图、表面 mask、几何路径 |
| AirfRANS | 原 graph data 调用及七通道输入 | [N,4] | 保留采样、batch、边界信息 |

输入提升继续使用任务原语义。例如 Darcy/NS 的 reference-distance 替换坐标路径、AirfRANS 的 reference-distance 追加路径、fx=None 的 placeholder 和 Plasticity 时间条件不能被一个通用坐标编码器替换。工业图批处理要隔离不同样本，不能把一批图拼成一个可相互 attention 的物理域。

NS 原训练用 10 帧输入预测后续一步并按原训练协议更新输入窗口，原评估执行 autoregressive rollout。每次模型调用都重建层间历史；本机制不跨真实时间缓存。原 20 帧样本支持默认 10→10 评估；未来 20/40 帧误差需要更长真实标签，不能用补零、重复真值或仅预测无标签替代验证。

Plasticity 保留逐时间条件调用及原 optimizer/scheduler 节奏。不要把 20 个时间点并入空间 N，也不要把原每时间条件更新改成一次总更新。

| 任务 | epochs | batch | optimizer | 原 lr 参数 |
|---|---:|---:|---|---:|
| Darcy | 500 | 4 | AdamW | 1e−3 |
| Elasticity | 500 | 1 | AdamW | 1e−3 |
| Airfoil | 500 | 4 | AdamW | 1e−3 |
| Pipe | 500 | 8 | AdamW | 1e−3 |
| NS | 500 | 2 | AdamW | 1e−3 |
| Plasticity | 500 | 8 | AdamW | 1e−3 |
| ShapeNet-Car | 200 | 1 | Adam | 1e−3 |
| AirfRANS | 398 | 1 | Adam | 1e−3 |

标准任务沿用原 weight_decay、clip 与调度器：Elasticity 是按 epoch 的 cosine；其余标准任务按原 OneCycle 节奏。1e−3 在 OneCycle 中并非整段训练恒定学习率。工业任务的 fold、subsampling、loss 权重等也沿用原配置。若用户实际分支已有经确认的实验协议，应保留并明确记录，而非机械覆盖成上表。

Darcy 稀疏系数观测→完整解场的实验本次不实现；坐标-only decoder、稀疏前段和 IPOT 时间缓存均不属于本模型的主范围。

## 9. 数学依据与可成立的结论

### 9.1 缓存与显式核注意力等价

固定同一 φ、ε、Q、K、T，由矩阵乘法结合律：

\[
\frac{Q(K^\top T)}{Q(K^\top\mathbf1)+\varepsilon}
=\frac{(QK^\top)T}{(QK^\top)\mathbf1+\varepsilon}.
\]

因而摘要写入不额外损失这一个已选核读取所需的信息，数值上允许浮点重排误差。但它不保证摘要无损保存 T，也不等价于指数 softmax attention。这种重排是线性注意力的标准依据；跨深度源层缓存、当前层读取与来源选择是此处的具体组合设计。

### 9.2 不需要跨层 token 索引对齐

设 Π 为某历史层 token 的置换矩阵，同时将 K、T 变为 ΠK、ΠT，则：

\[
(\Pi K)^\top(\Pi T)=K^\top T,
\quad (\Pi K)^\top\mathbf1=K^\top\mathbf1.
\]

所以读取对历史 token 的一致重排不变。它按内容聚合，不依赖“第几个 slot”的跨层一致身份。该性质不自动保证不同层 value 的通道语义完全兼容；source key、receiver query、各 FFN 需通过端到端训练形成协作。

### 9.3 明确的低秩约束

历史读取矩阵为：

\[
A_{ls}=\operatorname{diag}(Q_lb_s+\varepsilon)^{-1}Q_lK_s^\top.
\]

有 rank(A_ls)≤r。当 r<M 时，这是比一般 softmax Cross 更严格的单来源匹配约束。正值核提供非负权重；ε>0 时行和为 z/(z+ε)<1，其中 z=Q_lb_s，偏离 1 的程度取决于 z 与 ε 的比例，不能严格称每个读取都是凸组合。ε=0 且分母正时才严格行随机。

这个秩界只约束单份历史读取矩阵，不能推导整个非线性 block、跨来源融合或整个网络的秩≤r。当前残差、两 FFN、Down/Up 和点域旁路仍保留其他表达路径。

### 9.4 相对 no-SA 对照的函数类包含

对于相同公共部件的 LRSA-noSA，对本模型所有 γ_l 取 0，即可使所有历史注入消失，逐层计算退化为 Down—FFN1—FFN2—Up—PointModule。因此在允许 γ=0 的当前参数化下，no-SA 对照的函数类包含于新模型函数类。

该结论不是相对 full LRSA 的函数类包含定理，也不能保证优化器找到更优解或测试误差下降。严格的“跨深度优于层内 SA”仍是需要实验检验的研究命题。

### 9.5 为什么保留两个 FFN

FFN1 位于检索前，非线性地构造查询特征；FFN2 位于检索后，加工融合结果。二者之间存在输入相关跨 token 读取，通常不能合并成一个相同宽度 FFN。第一层没有历史而连续执行两个 FFN，是为保持深度与容量对照；也不意味着两者数学等价于单个同宽 FFN。

每层 Down 本身聚合全 N 点，Up 将全 M latent 写回点域；下一层再次 Down。因此即使没有当前 latent SA，模型仍有全局信息交换。

### 9.6 表达能力边界

历史只包含过去计算形成的信息。当前压缩中新出现的 token 关系，未必能在历史中被读取；它可能需要下一轮 Up→Down 才进一步交换。低秩摘要和跨来源融合也可能丢失细节，FFN2 不保证逆转已经丢失的信息。

共享 raw value 通道、正值核的选择性、历史范数差异和逐层信息冗余都是需观察的瓶颈。建议记录 α 的熵、γ、各 R 的范数和 query/key 特征分布；不能仅看 α 判断历史贡献强弱。

“物理”来自被处理的 PDE 特征。这里不增加 PDE 残差损失、守恒约束或物理方程打分，也不声称具备这些性质。

## 10. 参数、计算与实际效率

以下是本规格等宽 d、普通 FFN hidden=2d、无新增 W_V/W_O 的主要矩阵项。忽略 bias、norm、gate/scorer 的 O(Ld) 参数、逐元素运算以及相同输入输出头；MAC 一次乘加计一次，不与 FLOPs 混用。

matched LRSA 点 FFN 核心：

\[
P_{base,point}=L(23d^2+Md).
\]

dense 3×3 ConvFFN 核心：

\[
P_{base,conv}=L(32d^2+Md).
\]

移除 SA 的主要参数为 4Ld²，新增源 key 与接收 query 为 2(L−1)dr：

\[
\boxed{P_{new}=P_{base}-4Ld^2+2(L-1)dr+O(Ld)}.
\]

全历史读写主要 MAC：

\[
\boxed{C_{history}=BMdr\frac{(L-1)(L+6)}2}.
\]

其中写 key 和摘要各 L−1 次，query 投影 L−1 次，读取 L(L−1)/2 次。不能因 token 轴采用线性注意力就声称对深度 L 也是线性复杂度。

移除的 SA MAC：

\[
C_{SA}=BL(4Md^2+2M^2d).
\]

整个原 matched LRSA 核心的主要 MAC：

\[
C_{base,point}=BL(8Nd^2+15Md^2+4NMd+2M^2d),
\]

\[
C_{base,conv}=C_{base,point}+9BLNd^2,
\qquad C_{new}=C_{base}-C_{SA}+C_{history}.
\]

以 B=1、L=8、M=64、d=128、r=16 为例：

| 项目 | matched LRSA | 新模型 | 主项下降 |
|---|---:|---:|---:|
| PointFFN 核心参数 | 3,080,192 | 2,584,576 | 16.09% |
| ConvFFN 核心参数 | 4,259,840 | 3,764,224 | 11.63% |
| 被替换的 SA／历史部分 MAC | 41,943,040 | 6,422,528 | 84.69% |
| Darcy N=7225、ConvFFN 核心 GMAC | 18.12713 | 18.09161 | 0.196% |

参数减少较明显而整网计算下降很小并不矛盾：SA 的参数作用于 M 个 latent；保留的 Down/Up 与卷积每层作用于 N 个点。**本模型不具有原 CDLNO 持续潜空间部分省去多轮点域处理的成本优势。**

推理缓存的长期摘要大小为 B(L−1)(rd+r) 个数；上述示例 FP32 约 0.0551 MiB。推理在自身 Up/写入完成后不必长期保存原 T；训练 autograd 仍需其计算图。多来源读取的中间激活可能达到 O(BL²Md)，不能把推理摘要大小当作训练显存。

原 SA 可使用 fused SDPA；新读取涉及小矩阵乘法、FP32 累加、除法和深度融合。实际推理可能更快，也可能更慢。首次论文结论应报告真实每步训练时间、推理时间、峰值显存，以及到达同等误差的时间；不能用 MAC 比例推算 wall-clock 提速。

## 11. 在已有 CDLNO 仓库中的独立接入

本轮无法读取用户远端已改仓库的具体状态，因此以下是模块契约，而不是断言某些类已存在。

1. 新增独立模型家族 `kcdno`、独立配置、启动入口与 checkpoint 目录。保留默认 `transolver`／`cdlno` 路径，旧脚本不增加参数也应保持旧行为。
2. 新核 writer／reader 独立于旧 softmax CDPA 类。复用 RMSNorm/FFN 等基础原语可以，但不能把旧 CDPA 的计算含义改掉。
3. 若现有 LRSA/no-SA block 已允许在 FFN1 和 FFN2 之间插入模块，可在新模型路径复用该接口；若只返回完整 block 输出，应在新类中组合已验证的 Down、FFN1、FFN2、Up、PointModule。不能先跑完整 no-SA block 再在它外部补历史读取。
4. 官方 LRSA `disable_interleaved_blocks` 会同时关闭 SA 与 FFN2，不符合这里的“两 FFN 保留”。不要直接用它替代新模式。
5. 复用代码不等于共享参数对象。各层应独立实例化，不能将同一个 block 重复放入 ModuleList。
6. 不修改旧 `front_latent_mode=full/no_sa/identity` 的含义，不重命名旧 state_dict 键，不用 strict=False 掩盖结构不兼容。
7. 新构造器参数只送进新模型分支。不要将 kernel_rank、history_gate 等未经支持的 kwargs 送入旧模型构造器。
8. 复用各任务已验证 wrapper 的数据输入、时间、坐标和输出逻辑；共享新 core，避免复制八套训练器或数据处理。
9. checkpoint 保存模型家族、profile、L/d/M/h/r、FFN/norm 类型、核映射/ε、历史模式和初始化版本。恢复时按元数据重建，不能把旧 checkpoint 猜成新模型。输出目录含模型家族和配置，避免覆盖旧结果。
10. 可加新模型专属 `history_mode=off` 作为机制对照；off 不创建永不使用的历史参数。它不是改写旧 CDLNO no-SA。当前主版只需 all/off，不主动扩展相邻、窗口、persistent 或 Slice 机制。

新模型本体不需要导入 IPOT processor，不涉及 IPOT 的 encoder FFN／GEGLU 差异；旧 CDLNO 中这些已记录设计照常保留。

## 12. 公平对照与无真实数据时的验收

必要比较至少包括：

| 对照 | 结构 | 回答的问题 |
|---|---|---|
| Transolver 原模型 | 原脚本配置与训练 | 与原基线性能/成本相比如何 |
| matched LRSA-full | 与新模型同输入头、L/d/M/h、两 FFN、卷积，保留 SA | 新机制能否替代 SA |
| matched LRSA-noSA | 同上，删 SA，不加历史 | 新机制是否真正带来收益 |
| KCDNO | 删 SA，两 FFN 间加全历史核读取 | 本模型 |

matched LRSA 是用于控制变量的 LRSA 风格对照，不称作完整官方训练复现。此前 CDLNO full/no_sa/identity 仍可额外比较，但它们含 persistent 后段，不能冒充上述全 LRSA 对照。

记录实际参数量、MAC/FLOPs 统计范围、任务误差、训练/推理时间与显存。相同层数不等于相同预算；尽量多随机种子，报告波动。新模型与 noSA 相比参数增加、与 full 相比参数减少，应将两组对照一起解释。

没有真实数据也能验证以下架构语义：

- float64 小张量比较缓存形式与显式核矩阵形式的前向及输入/投影梯度，使用相同 φ、ε；不是比较它与 softmax attention 完全相等。
- 历史 token 同步重排后的读取不变；batch 样本互相隔离。
- L=1/2/4 等配置验证空历史、读写次数和因果性；首层无 reader、末层无 writer；L=8 总读取 28 份历史。
- w=0 的来源均匀、γ=0.1 的融合系数符合公式；空历史输出严格等于 U。
- 在隔离历史读取的合成 loss 下，最早源层及 Wk 能收到有限非零梯度，排除 detach 被点域旁路掩盖。
- eval 模式 A→B→A 三次调用，相同 A 输出一致，缓存不残留。
- 八个任务的合成输入形状与现有模型调用一致；检查网格 H×W、NS 十帧通道、Plasticity 时间和四输出、工业图 mask/batch 路径。
- 旧 CDLNO 三种模式用相同 checkpoint 和固定输入，state_dict 键与输出保持；新参数不改变旧默认。
- 在实际可用设备检查 forward/backward 和 AMP 数值。CPU FP32 通过不能写成 CUDA 12.8 通过。

真实数据集训练效果、长时 rollout 精度和 GPU 延迟必须留到有真实条件时测量。合成检查足以发现许多架构/梯度/接口错误，但不证明数据语义和真实训练结果完全正确。

## 13. 参考依据与归属

- [LRSA 方法与实验配置](https://arxiv.org/html/2604.03582v1)：提供 Down–latent mixing–Up 的骨架与两 latent FFN 的来源。新模型删除其 SA，并非逐字复现。
- [LRSA 官方仓库](https://github.com/Adversarr/LRSA-Operator)：只读核对原语和配置链；实现时以既定新规格为准，不能盲拷构造/初始化问题。
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)：正值核注意力和结合律缓存的数学基础。本文使用跨网络深度的独立来源摘要，不是照搬语言模型的真实时间递归。
- [Attention Residuals](https://arxiv.org/abs/2603.15031)：按深度进行输入相关来源选择的启发。这里先用核读取解决跨层 latent 对应，再进行深度融合，且有独立 γ；不是原 AttnRes 的逐式复现。
- [Transolver 官方仓库](https://github.com/thuml/Transolver)：保持任务接口与基线训练评估协议。

本地既有约束依据为 CDLNO v1_2 计划、原分阶段实现提示词与 front latent 消融提示词。KCDNO 是这些既有模型之外的新入口；以后如修改本规格，应版本化记录，不回写改变旧实验定义。
