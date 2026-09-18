# LinearNO L3 — Standard 完整 Model

**状态：PASS。** 本轮只完成六个 Standard PDE 可共用的独立完整 Model、数学/reference/官方 parity 与旧模型回归；非资源门控验收均通过。未修改或接入 model_dict、六个 exp、训练/评估、checkpoint helper 或生产 launcher，未执行 L4。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`；GPU 仅执行 L3 明确要求的合成官方模型检查及 L2 回归。

## A. 基线、文件与范围

目标 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`，remote `git@github.com:hxh5159/CDLNO-w.git`。此前 L0–L2/user untracked 文件全部保留。修改前保存 1,252 个文件（910 tracked、93 untracked、249 ignored）到 `/home/hwz/CDLNO-artifacts/linearno-l3-before-7d5dtz83/source`。见 [baseline](linearno_audit/l3/baseline.json)。没有 reset/clean/stash、安装、下载、commit/push 或真实训练。

| 文件 | 实际改动和作用 |
|---|---|
| `PDE-Solving-StandardBenchmark/model/LinearNO.py` | 新增完整 `Model`、`LinearNOBlock`、`PointwiseMLP`、位置函数；组合已接受 L2 原语 |
| `tests/linearno/standard_support.py` | 新增固定官方文件 SHA 校验、完整且未改写的 class/function AST 加载、构造字段显式映射；不导入任务入口 |
| `tests/linearno/standard_reference.py` | 新增独立完整公式：手写 LN moments/erf GELU、L2 数学 oracle，无待测 forward 调用 |
| `tests/linearno/test_standard_model.py` | 新增 5 方法：六真实配置、RNG/初始化、完整官方 forward/梯度/step/checkpoint、double oracle |
| `tests/linearno/test_standard_structure.py` | 新增 5 方法：第八层/shape trace、batch/置换/输入保护、位置/设备、非法合同、独立进程加载 |
| `tests/linearno/fixtures/official_standard_models.json` | 官方完整模型在生产实现前生成的小型 key/shape/count/初始化 hash 夹具，无权重二进制 |
| 本报告、`docs/linearno_audit/l3/*`、`LINEARNO_IMPLEMENTATION_STATUS.md` | 新证据及 STATUS 增量；旧报告/测试/计划不改 |

固定官方 `HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` 仍在目标外 `/home/hwz/LinearNO`。Standard 原文件 SHA256 `2019eacb9c6b61b3b7ede6a94e4ebfa3acb6ab95b25c1ab22299ab3dcc6c79d6`，Embedding SHA256 `e3a4f71c4378b7cce09a43b8c306d6d8a17edb4cd69db5eb069e55a57d2affe2`。未改参考仓库、未加载外部 pickle。按 L0 许可边界独立组合/实现，没有整树 vendoring；原固定树无 LICENSE 的事实未变。

## B. 公式、类和 state_dict

稳定类路径为 **`model.LinearNO.Model`**，文件为 `PDE-Solving-StandardBenchmark/model/LinearNO.py`。从该 benchmark cwd 导入 `from model.LinearNO import Model`；公开 `Model(x, fx, T=None) -> [B,N,out_dim]`。未来 factory 返回此模块即可；本阶段旧 registry 对 `linearno` 仍抛 KeyError，测试专门确认没有提前接线。

| 规格/官方符号 | 新实现 | 独立证据 |
|---|---|---|
| `preprocess: MLP(input,2d,d,n_layers=0)` | `Model.__init__` → `PointwiseMLP`，key 为 `linear_pre.0`/`linear_post`；保留空 `linears` | 六配置所有 key/shape/count；逐参数初始化比较 |
| ordinary x；有 fx 时 concat(x,fx) | `Model.forward`，输入 `[B,N,space_dim]`、`[B,N,fun_dim]` | fx 有/无、batch1/2、梯度与独立 oracle |
| unified 替换 x | `_unified_positions`，`pos:[1,H,W,ref²]`，`persistent=False`；lift 输入为 `ref²+fun_dim` | 改变 x 值输出逐位不变；x 无梯度；官方 CUDA pos 值；5×7 scalar geometry |
| 无 fx：lift(x)+placeholder | `Model.forward`；fun_dim=0；placeholder 最后注册 | 官方对应梯度；零通道 Tensor 与 None 的语义另测 |
| Time_Input：sin/cos → Linear/SiLU/Linear | 未改的 `model.Embedding.timestep_embedding` + `time_fc`；T `[B,1]` | T 有/无、原官方 T 梯度；double 扩展与独立公式 |
| Q=softmax_M(XWq)，K=softmax_N(XWk)，Y=Q(KᵀV) | L2 `LinearNOAttention.forward`，四个现有 Standard 包装 | L2 全部回归；完整模型 oracle/官方 parity/运行 shape trace |
| `z=Attn(LN(x))+x; y=MLP(LN(z))+z` | `LinearNOBlock.forward`；独立 ln_1/ln_2/Attn/mlp | 完整逐 block 输出、全部参数梯度 |
| 第八层仍是完整 block，再 LN/head | 最后 `LinearNOBlock` 另有 ln_3、mlp2 | 实际 hook 顺序 `0.Attn,0.mlp,…,7.Attn,7.mlp,7.ln_3,7.mlp2` |
| 官方 args.model 选变体 | 目标只读取独立 `linearno_variant`；M 为 `linearno_rank` 绝对值 | args/model/slice_num/key_ratio 构造 kwargs 均明确拒绝；旧 registry 未变 |
| 官方全模型 apply 和 placeholder 时序 | `initialize_weights` → 一次 `self.apply(_init_weights)` → 最后 rand(d)/d | before/after apply 同权重/RNG 均精确相同；每 module callback 恰好一次 |

所有正常 forward 只返回预测，不缓存 Q/K/C、无 CPU 同步或调试统计。卷积变体仍在**每个 attention block**做一套输入 Conv2d；没有 Transolver 的第二套 x/fx 卷积、slice normalization 或 slice self-attention。所有层参数对象和 storage 独立。

**输入约束与设备修复的明确差异：**

1. 原 unified 构造中的 `.cuda()` 去掉；保留 NumPy linspace→float32 的采样/舍入顺序和 `[H,W,ref,ref]` 展平顺序。位置改为非持久 buffer，`.to()` 可迁移，state_dict key 未新增。原官方 unified parity 在原生 CUDA 执行，无 `.cuda()` monkeypatch；实际 NS 64×64/ref10 和小网格位置值最大差为 0。
2. 原 sin/cos embedding 始终 FP32；新 Model 在进入 time_fc 前转到其权重 dtype，使显式 `.double()` 模型可运行。官方有效 FP32 路径是 no-op，原生 FP32 parity 通过。double+T 是独立公式/功能检查，**不冒充原官方 double+T 原生通过**。
3. 新入口对维度、N=H×W、fx/T、variant 给出明确错误，不补齐/截断。支持本阶段确认的 GELU profile；Time_Input 要求偶数 d，避免原函数奇数宽度的错误 padding。它们不改变六个已确认配置。无 fx 要求 fun_dim=0；传 `[B,N,0]` 的显式空 fx 则沿官方 concat 分支、不加 placeholder。

state_dict 与官方完整模型 **identity mapping**（无需改名），双向 `strict=True`；没有 strict=False、丢键或补权重。temperature 不经 Linear/LN/Conv 初始化 callback 改写，仍为 .5。有 fx 的 placeholder、T=None 的 time_fc 保留注册但无梯度，与官方条件路径一致。

## C. 六个发布配置的真实参数量

以下由**未改官方 Model 先构造测量**，随后由目标模型测量、全部 key/shape/同 seed 值与独立闭式计数三重核对。它们不是论文标称参数量，也不是缩小测试配置。都为 L8、heads8、GELU、dropout0。

| 任务 | variant | d | 每头 M | FFN ratio | H×W（模型用途） | fun/out | unified/ref | Time | 实际参数 |
|---|---|---:|---:|---:|---|---|---|---|---:|
| Darcy | conv_temp | 128 | 64 | 1 | 85×85 | 1/1 | off/8 | off | **1,766,145** |
| Elasticity | temp | 128 | 64 | 1 | 85×85（未使用；点式 N 可变） | 0/1 | off/8 | off | **585,217** |
| Airfoil | conv_temp | 128 | 64 | 1 | 221×51 | 0/1 | off/8 | off | **1,765,889** |
| Pipe | conv_temp | 128 | 64 | 1 | 129×129 | 0/1 | off/8 | off | **1,765,889** |
| NS | plain（发布名 no_temp） | 256 | 32 | 2 | 64×64（用于位置） | 10/1 | on/10 | off | **3,377,921** |
| Plasticity | conv | 128 | 64 | 1 | 101×31 | 1/4 | off/8 | on | **1,799,428** |

闭式计数包含 preprocess、placeholder、可选 time_fc、8 组 attention/FFN/两 LN 和末层额外 LN/head。令 `q=d/h`、`c=是否conv`、`t=是否有效temp`、`I=lift输入宽度`、`O=out_dim`、`r=FFN ratio`：

```text
preprocess = 2dI + 2d + 2d² + d
attention/block = (1+8c)d² + d + 2qM + q² + (1+c)(d²+d) + 2th
FFN+norm/block = 2rd² + (r+1)d + 4d
total = preprocess + L*(attention/block + FFN+norm/block)
        + (2d + dO + O) + d + [Time_Input ? 2(d²+d) : 0]
```

六个真实配置执行的是构造、参数、初始化、state_dict 检查；不声称六种原尺寸完整 forward 或数据协议已经接线。完整数值 forward/backward 使用 d12/h3/M4、3 层和两个额外 8 层案例，3×5/5×7 网格、B1/2；普通点式另测 N=1/7/19。

## D. 实际执行、误差与边界

环境：Python3.13.9，torch2.13.0+cu130，timm1.0.28，einops0.8.2，NumPy2.2.6；RTX5090 Laptop×1。PyG2.3.1 可用，缺 torch_cluster/torch_scatter/pyg-lib。没有安装依赖。官方 parity 设置 TF32 off、cuDNN deterministic/benchmark off、compile off，restore CPU/CUDA RNG 后分别执行非零 output dropout。见 [environment](linearno_audit/l3/environment.json)、[执行命令](linearno_audit/l3/commands.json)。

| 实际验收 | 结果 | 数值与证据 |
|---|---|---|
| L3 新测试 | 10 方法通过，9.594s | `regression-tests.txt`、`model-parity.json`、`structure.json` |
| 六个真实配置 + 8 个初始化子案例 | 全部通过 | 所有 key/shape/count、同 seed 参数、构造/apply 后 RNG 逐位一致；无大二进制 |
| 完整官方 CPU parity | 10 子案例；double/FP32 | 各 block/最终 forward 最大差 0；参数梯度最大 2.78e-17；step 最大 1.87e-16 |
| 完整官方原生 CUDA unified parity | 8 子案例；FP32 | final/block max abs 8.38e-9，loss 4.66e-10，全参数 grad 2.98e-8，AdamW step 1.42e-7 |
| 独立完整 double oracle | 8 子案例 | forward/梯度所有比较 max abs 1.14e-13；无调用待测 Model/block forward |
| strict state checkpoint | 上述 18 parity 案例 + 新 cwd | checkpoint 输出误差 0，位置不写入 state_dict；仅本地自建 pure-state，无外部 pickle |
| 第八层与内存形状 | 四变体通过 | 各 8 次完整 attention/FFN、16 次 softmax、16 次 bmm；N35 时无 N×N；合法 context 为 M4×dh4 |
| L1 全部回归 | 16 方法通过，45.859s | 八原 Transolver、旧 parser、artifact/schema、绘图/保存 RNG、fresh-process resume |
| L2 全部回归 | 14 方法通过，8.600s | 原子 oracle/官方 CPU/CUDA FP32/FP16/BF16、独立结构；新记录 `l2-parity.json` |
| 既存存档/输出/源冻结 | 28 方法，68.302s，0失败/错误 | 1 个 Air 完整采样 epoch 子案例因缺 torch_cluster 跳过；Car 真实 PyG 合成 epoch、Air weighted-loss/记录片段通过 |

合计 **68 测试方法，0 失败、0 错误，1 个旧 Air 子案例跳过**。主回归逐模块耗时见 `regression-results.json`；上述时间是测试时间，不是模型性能。`ShapeTrace` 记录的是每个实际算子输出 shape 及其中最大 tensor，**不是 CUDA allocator 峰值显存或运行速度测试**。

L1 的 fresh-process resume/随机流检查复用既有基础设施和合成更新序列，覆盖绘图/保存启停、DataLoader shuffle、Plasticity 的实际 Torch 时间排列以及 Python/NumPy 流；它不表示八任务已有完整原生 resume 接线。旧输出记录检查的既有数据与 loss 边界保持原报告定义。

阈值没有放宽：CPU double `atol=1e-12,rtol=1e-10`；CPU FP32 `1e-6,1e-5`；CUDA FP32 `1e-5,1e-4`。[完整误差](linearno_audit/l3/model-parity.json)逐输入/参数记录 max/mean abs、max/mean relative 和 relative L2（relative 分母下界1e-12）；[汇总](linearno_audit/l3/parity-summary.json)也保留这些字段。CUDA final mean abs 最大8.38e-10、relative L2最大1.10e-7；近零参数梯度的逐元素 relative 最大约0.00424，但绝对值及既定 atol+rtol 均通过。没有隐藏失败或通过放宽容差消除失败。

可重跑的无数据命令（repo root）：

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  linearno.test_standard_model linearno.test_standard_structure \
  linearno.test_profiles linearno.test_schema linearno.test_legacy linearno.test_rng \
  linearno.test_attention_parity linearno.test_attention_structure -v
```

设置 `LINEARNO_REFERENCE_ROOT` 可定位外部固定官方源码；源码 hash 不符会拒绝。完整原生 unified 官方检查必须有 CUDA，缺 GPU 时明确跳过该项。使用 `LINEARNO_L3_PARITY_REPORT`/`LINEARNO_L3_STRUCTURE_REPORT`/`LINEARNO_L2_PARITY_REPORT` 可将新结果写到新路径；不要覆盖已经接受的 L1/L2 证据。本轮真实执行的 unittest loader/结果记录脚本保存于 commands.json。

最小完整模型构造示例（**合成测试，不是生产训练命令**）：

```bash
cd PDE-Solving-StandardBenchmark
PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import torch
from model.LinearNO import Model
torch.set_num_threads(1)
m = Model(space_dim=2, fun_dim=1, out_dim=1, n_hidden=16, n_head=4,
          n_layers=8, H=5, W=7, linearno_variant='conv_temp', linearno_rank=8)
y = m(torch.randn(2, 35, 2), torch.randn(2, 35, 1))
assert y.shape == (2, 35, 1)
y.square().mean().backward()
PY
```

## E. 旧模型保护与自审

[delivery-check](linearno_audit/l3/delivery-check.json)复算 L0 manifest，而不是只检查 git diff：原 1,161 文件及 221 重点冻结文件的**分类、size、SHA256**全部保持，910 tracked 无 diff；L3 起点其余1,251文件不变，仅 STATUS 增量。L1/L2新增源文件 hash 也保持，preexisting untracked/ignored 没有逃过检查。没有新增 ignored 文件，未修改 `.gitignore`；只允许的运行路径仍遵循 L0 `ignored-output-policy.json`。当前 `LINEARNO/**` 和 monitor 从 L0 起就不存在，相关测试为 **N/A**，不是虚构“monitor通过”。

本阶段已自审以下五点，均有独立证据，不需要用户重新裁定：

1. **初始化顺序**：真实 pre-apply 参数和 RNG、post-apply 参数和 RNG、placeholder 最后一次随机 draw；原子源码保持。没有为了匹配输出跳过构造 RNG。
2. **位置和条件输入**：unified replacement、显式空 fx/None 差别、TNone 已注册但不执行的 time_fc、非持久 buffer、CPU↔CUDA 与 dtype 迁移。
3. **层数和纯 LinearNO 数学**：最后完整 block、Q/K 轴、跨head共享小投影、合法平方 context 与禁止 slice attention 的区分；双重依据为原源码和独立 oracle。
4. **兼容/冻结**：全部 key/shape/count、identity strict load；新 Model 无 args.model 注册语义复用；真实 cwd 与旧 factory 选择仍保持；所有旧文件 hash 未变。
5. **声明边界**：六个正式配置只构造/参数验证；小模型官方 forward、原生 GPU unified parity、原子 AMP 重跑与真实训练/远端验收明确分开。

## F. 未执行及后续边界

NOT RUN：真实数据/loader、生产 train→checkpoint→eval、论文指标/精度/收敛、实际 epoch 耗时；六个正式尺寸完整训练/forward；完整 Model AMP/compile、远端 Python3.10/Torch2.11/cu128；工业 LinearNO 完整模型/外部 whole-object 转换。本阶段没有声称这些事项通过。既有 L0 工业 objective/force/split 冲突仍留给被授权的相关阶段，本轮没有新增需要改变确认架构的冲突。

后续接线需显式把 L1 resolved 配置映射到这里真实的 Model 签名，保持 CLI/profile/registry 三者独立；本轮未执行。**本L阶段结束，未执行下一阶段。**
