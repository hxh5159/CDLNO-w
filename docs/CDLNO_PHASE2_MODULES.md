# CDLNO 阶段2：基础模块实现与验证

日期：2026-09-13。依据：用户本轮阶段2授权、v1.2 §2/§4、阶段0确认的 LRSA/IPOT 源码差异。阶段0、1已获用户审查通过；阶段3未执行。

## A. 完成范围

实现 `cdlno.modules` 中的 norm、PyTorch SDPA 投影层、plain FFN、GEGLU、dense ConvFFN，以及 `LRSAFrontBlock`、`IPOTBridge`、`PersistentLatentBlock`、`LRSAFeatureReadout`。本阶段输入为已经提升的隐藏特征，不包含输入 stem、坐标/时间适配器。

没有实现 CDPA 融合、`CDLNO` 整模型类、模型注册、训练入口或任务 wrapper。测试独立运行各模块；六个后段实例仅用于验证参数独立性，没有组装 2+6 模型。

## B. 文件和改动理由

| 文件 | 内容和理由 |
|---|---|
| `cdlno/modules.py` | 所有本阶段数学模块的单一实现；只依赖 torch 和标准库 |
| `cdlno/__init__.py` | 更新包说明；保留原有配置导出，模块通过 `cdlno.modules` 显式导入，使配置/sidecar 导入不要求先加载 torch |
| `tests/test_modules.py` | 18 项模块合同测试：独立显式公式、梯度、历史取点、残差、初始化、5×7 点序、CPU/GPU 精度 |
| `tests/test_lrsa_reference.py` | 2 项可选的原 LRSA 同配置对照；读取既有参考 checkout，不复制上游源码或安装其训练框架 |
| 本报告、`CDLNO_IMPLEMENTATION_STATUS.md`、`CDLNO_REFERENCE_AUDIT.md`、`AGENTS.md`、`memory/current-state.md` | 更新实际完成、验证结果、数值约定和阶段边界；历史审查与当前实现状态分开 |

阶段1的 `config.py`、`checkpoint.py`、`pyproject.toml`、环境预检脚本未修改。各模块直接接收维度和 ratio；本阶段不把它们接入任务配置或重写 checkpoint schema。

## C. 计算顺序、公式与形状

记 `H:[B,N,d]`、latent `[B,M,d]`、`h` 为头数、`d_h=d/h`。默认 dropout=0、FFN ratio=2。测试主体采用 `B=2,N=35,d=16,h=4,M=5`；不同模块没有隐式共用参数。

| 模块/代码符号 | 计算顺序 | 输出形状 |
|---|---|---|
| `RMSNorm` | 沿最后一个维度 `x / sqrt(mean(x²)+1e-6) * scale`，scale 初值1、无 bias | 保持输入形状；Q/K 时最后一维为 `d_h` |
| `make_norm(..., 'layernorm')` / `nn.LayerNorm` | 沿最后一维去均值并归一化，eps=1e-6、scale1/bias0 | 保持形状 |
| `_ProjectedAttention` | Q/K/V 独立投影→分头→按模块规定做 QK norm→非因果 SDPA→合头→O 投影 | Q 长度为 S 时 `[B,S,d]` |
| `PlainFFN` | `Linear(d,2d,bias=True) → GELU → Linear(2d,d,bias=True)` | `[B,S,d]` |
| `GEGLUFFN` | `Linear(d,4d) → split(value,gate)` 得到两个2d → `value*GELU(gate)` → `Linear(2d,d)`，两个 Linear 有 bias | `[B,M,d]` |
| `ConvFFN` | `[B,N,d] → [B,d,H,W]` → 普通3×3卷积 → channels-last → 内层 LayerNorm → 无bias `Linear(d,2d)` → GELU → 有bias `Linear(2d,d)` → 原点序 | `[B,N,d]` |

所有 attention 使用标准 `d_h^(-1/2)` 缩放，softmax 沿各自 K/V 的 token 轴。down 的 Q/K/V 形状为 `[B,h,M,d_h]` / `[B,h,N,d_h]` / `[B,h,N,d_h]`；up 则为 `[B,h,N,d_h]` / `[B,h,M,d_h]` / `[B,h,M,d_h]`；latent SA 都是 `[B,h,M,d_h]`。生产模块不返回 attention 矩阵。

### 完整前段 `LRSAFrontBlock`

```text
Hn = RMSNorm_point(H)
Z = down(Hn; learned Q[M,h,d_h])       # 无 W_Q，无 query residual
Z = Z + PlainFFN1(RMSNorm1(Z))
Z = Z + latent_SA(RMSNorm_sa(Z))
T = Z + PlainFFN2(RMSNorm2(Z))         # 显式保留完整 T，不 detach
U = H + up(Hn, RMSNorm_up(T))         # down/up 参数独立
H_next = U + PointFFN/ConvFFN(RMSNorm_point_ffn(U))
return H_next, T
```

返回 `[B,N,d]` 和 `[B,M,d]`。测试用 hook 验证返回的 `T` 与 up 前归一化接收的是同一张量对象，并从仅依赖 `H_next` 的损失反传到 `T` 及两次 latent FFN。修改末尾点 FFN 只改变 `H_next`，不改变该次保存的 `T`。

### Bridge 与独立后段

```text
IPOTBridge(H):
    Q0 = learned_queries[M,d]，广播为[B,M,d]
    Z0 = Q0 + Cross(LN_q(Q0), LN_context(H))

PersistentLatentBlock(Z):
    normalized = LN1(Z)               # 只算一次
    A = Z + SA(normalized)             # Q/K/V 都来自同一个 normalized
    Z_next = A + GEGLU(LN2(A))
```

Bridge 的输出为 `[B,M,d]`，不注册 encoder FFN；后段输入输出都为 `[B,M,d]`。两者采用 LayerNorm，没有 per-head QK norm。零化 down 输出会得到零 latent；零化 bridge 的 Cross 输出则得到原始 learned Q0。后段不访问点域、没有卷积或跨 forward 状态。

### 最终 `LRSAFeatureReadout`

```text
DeltaH = up(RMSNorm_query(H_F), RMSNorm_latent(Z_P))
H_D = H_F + DeltaH
H_out = H_D + PointFFN/ConvFFN(RMSNorm_point_ffn(H_D))
y = Linear_out(LayerNorm_out(H_out))
```

`forward_features(H_F,Z_P)` 返回 `[B,N,d]`；`forward(H_F,Z_P)` 返回 `[B,N,C_out]`。Q 来源和原值点残差都是 `H_F`，latent 仅提供 K/V。读出内部没有 down 或 latent SA。输出 head 的 `LN_out` 按 §2.7 使用 **LayerNorm**；up 的外层、latent 及 point FFN 外层按 §4.3 使用 **RMSNorm**，二者分开记录。

### 初始化和数值政策

- 所有普通 Linear 使用一次显式 `trunc_normal_(std=0.02)`，bias0。没有父级递归 reset，也没有深度残差缩放。PyTorch 构造器自身的初始赋值不作为再次执行模型初始化。
- Conv2d 保留 PyTorch 默认 Kaiming uniform **及默认 bias 初始化**；不在父模块里归零或重置卷积。其配置为 `groups=1,kernel_size=3,stride=1,padding=1`，零 padding。
- 前段 query `[M,h,d_h]` 展为 `[M,d]`，orthogonal 初始化；M>d 时测试列 Gram 为单位阵，不声称行两两正交。Bridge query 使用 normal(std=0.02)。特殊 query 在所属 attention/bridge 普通子层初始化后设置，父模块不再覆盖。
- LRSA down/SA/up 的 Q/K 分头后分别做 RMSNorm(d_h)，参数互相独立；Q/K/V 无 bias，O 有 bias。Bridge/后段也遵循这些 projection bias，但不做 QK norm。
- RMSNorm 对 FP16/BF16 输入用 FP32 累计，保留 FP64 数学参考精度，返回输入 dtype；down 在 autocast 下把 learned Q 对齐到已投影 K 的 dtype。
- ratio 产生非整数隐藏宽度时明确报错，不采用参考库的按16向上取整。错误 `d/h/M`、batch 广播、空序列和 `N!=H*W` 会报错。结构 block 在 attention 前检查网格；不猜 sqrt(N) 或重排物理坐标。

## D. 实际验证

命令（仓库根目录执行，无数据、无依赖安装）：

```bash
python -B -m unittest discover -s tests -p test_modules.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p test_lrsa_reference.py -v
```

实际执行机：Python 3.13.9、torch 2.13.0+cu130、CUDA runtime 13.0、NVIDIA GeForce RTX 5090 Laptop GPU。这里只记录测试执行环境；目标仍是用户已运行原 ShapeNet-Car 的远端 Python3.10/torch2.11/cu128 依赖族，Python3.11 待验证。没有运行远端预检或更换依赖。

| 检查 | 实际结果 |
|---|---|
| `test_modules.py` | **18/18 通过**；包含 CPU FP32/FP64、CPU BF16 数学路径及实际 GPU 小模块测试 |
| 独立显式公式 | RMS/LN、plain/GEGLU、QK^T/softmax/AV、bridge、后段、读出输出和参数/输入梯度通过；CPU FP64 输出 atol=1e-10/rtol=1e-8，梯度 atol=2e-9/rtol=2e-7 |
| 原 LRSA 同配置 | **2/2 通过**：点版、5×7结构版；移入相同权重，点输出/T/所有对应梯度在该 FP64 样例上 max abs error 均为0；不作一般逐位一致承诺 |
| 历史、残差、参数 | T取点/活跃梯度通过；down无query残差、bridge有query残差；6个独立后段实例的参数对象和data_ptr均无共享 |
| 初始化 | 每个普通 Linear 的显式截断正态只调用一次，特殊query初始化一次；norm/bias符合策略；卷积与同seed原生Conv2d初始权重和bias相等 |
| 5×7 点序 | 使用不对称上邻/右邻卷积核和跨通道权重，以 `row*7+col` 逐点手算对照；边界、输入梯度、非连续输入均通过 |
| 本地 GPU | FP32、FP16 autocast、BF16 autocast 的各个小模块前向/反向均有限；CPU/GPU前向对照通过；未做真实配置、大N训练或速度测量 |
| Python3.10语法、冻结文件哈希、空白检查 | **通过**：4个Python文件以`ast.parse(feature_version=(3,10))`解析；75个冻结文件哈希不变；全部本轮文件直接检查无行尾空白且文档本地链接有效。语法解析不等于3.10运行验收 |

首次运行有 **1个数值断言失败、2个后端错误**，处理及边界如下：

1. 本地 oneDNN 报 `DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2`，出现在结构前段/readout 的 CPU BF16 卷积反向。便携测试局部关闭 MKLDNN 后前反向通过；没有改模块或全局运行政策，也不声称此环境默认 oneDNN BF16 卷积反向可用。上下文恢复时的 oneDNN/Intel GPU warning 如实保留在测试输出。
2. 结构模块的默认 GPU/CPU 对照曾有约 `4.77e-6` 差异，超出原 FP32 绝对容差 `2e-6`。测试中关闭 CUDA matmul/cuDNN TF32 并在结束后恢复原值后，在**原容差不变**下通过；没有修改生产精度设置。

原 LRSA import 会打印缺少可选 xformers/liger 的提示，随后正常使用 PyTorch 路径完成测试。没有为参考代码安装这些依赖。

未运行：远端 torch2.11/cu128 验收、整网前反向、原任务损失连接、真实 checkpoint 权重/任务加载、真实数据训练、误差或性能比较。这些不属于本阶段基本模块验收。

## E. 冻结区域证据

没有修改三个原任务目录、根 `Physics_Attention.py`、原依赖文件或计划材料。对71个已跟踪基线文件及4个阶段1基础协议文件共75项保存 SHA256 清单，并在结束时逐项对比。清单文件 `/tmp/cdlno-phase2-frozen-hashes.json` 的 SHA256 为 `a64ab0dbe768fd1dcb607834a19bffa1eabc4d1f1ca6b6b1089e748f601b1d1b`。

结束时 `git diff --name-only HEAD` 实际为空，75个冻结文件哈希均不变；新包和文档仍为未跟踪文件，因此还直接检查新文件空白及哈希，不把单独的 `git diff --check` 当作覆盖全部新文件的证据。参考LRSA/IPOT相关model源码对HEAD也没有diff。未 import exp/main/train/dataset 入口，没有下载数据、真实训练、安装/替换依赖、commit/push。

## F. 差异、未解决事项与优先审查点

参考依据为 LRSA commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e` 和 IPOT commit `18c177846267505ee9503445a146dfd7dee34c41`。LRSA 未发现独立许可证，核心按已确认公式重新编写；可选测试只读取用户已有参考 checkout。

- LRSA：采用相同完整结构和显式 plain FFN/32维hidden/QK RMSNorm/bias配置进行对照，不读训练 YAML；禁用 RoPE/gate/mass。使用 PyTorch SDPA，不引入 optional kernel。初始化统一为计划策略，不复用上游父构造器的多次 reset。
- IPOT：后段每次构造创建独立 SA/GEGLU；Q/K/V 同源 pre-LN，修正原 processor `context=z` 的未归一化 K/V 路径。Bridge 保留 query residual，省略源码定义但未调用的 encoder_ff；GEGLU ratio=2，LN eps=1e-6。由独立显式公式验证，不把与未修正源码的差异称为 parity 失败。
- 最终 decoder：使用 LRSA up/point FFN 的计算方向，H_F旁路和独立输出head按计划；没有实现 IPOT 坐标 decoder。

需要优先审查的5点：

1. `LRSAFrontBlock.forward` 中 T 的取点及两次 latent FFN 是否符合意图。
2. Down/bridge 不同的 query residual，以及后段 Q/K/V 共用同一 LN 输出。
3. readout 的 H_F query/原值残差、RMS外层与 `LayerNorm_out` 的区分。
4. dense ConvFFN 的点序、groups=1、内层LN与linear bias，以及卷积保留默认初始化。
5. 当前只有独立模块。阶段1 schema 与这些模块的最终配置绑定、CDPA、整网和任务接入仍需下一阶段明确授权；本地测试不代替远端新模型训练验收。

**本阶段结束，未执行下一阶段**
