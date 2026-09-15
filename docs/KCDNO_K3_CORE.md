# K3：独立 KCDNOBlock 与 L 层特征核心

2026-09-15。K2 已由用户审查通过，本次仅实施 K3。依据为当前 K3 逐条指令、`PLAN_KCDNO/KCDNO_Model_Specification_v1.md` §§3–8、已接受的 K1/K2 和 K0 公共原语映射；旧 CDLNO/A 阶段合同继续保留。

## A. 完成范围与接口

新增 [core.py](../cdlno/kcdno/core.py) 的 `KCDNOBlock` 与 `KCDNO`。直接组合现有 Down/Up、RMSNorm、PlainFFN、ConvFFN 与 K2 Writer/Reader，没有调用旧 `LRSAFrontBlock(no_sa)`，没有修改它或 CDPA。

```python
import torch
from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.kcdno.core import KCDNO

cfg = KCDNOArchitectureConfig(L=2, d=8, h=2, M=3, kernel_rank=5,
                              point_module="conv_ffn", history_mode="all")
core = KCDNO(cfg)
x = torch.randn(2, 35, 8)  # 已由任务提升的特征，不是原数据字段
features = core(x, grid_shape=(5, 7))  # [2,35,8]
features.square().mean().backward()   # 合成诊断损失，不是任务训练链路
```

`KCDNO()` 使用既有 K1 默认 L8/d128/h8/M64/r16/history=all/point_ffn。core 只接收、返回 `[B,N,d]`，没有输入提升、时间/位置注入、输出 LN/head、Bridge、persistent 后段或额外 final Up。K1 的 `output_norm=layernorm` 是后续任务 head 合同，本特征 core 不执行它。

Conv 模式每次显式传入正整数 `(H,W)`，要求 N=H×W；不猜 sqrt(N)、不排序、不重采样。point 模式支持不同 N，不接受被忽略的 grid_shape。M/r 与点数无关，r 不要求被 h 整除。

K1 配置可记录显式 hidden 宽度；本次 K3 指令明确三处 FFN hidden=2d，因此 core 对不是 2d 的已解析宽度清楚报错，不静默忽略或取整，不修改 K1 配置协议。

`KCDNOBlock.forward(X, tuple_of_caches, grid_shape=...)` 的**内部**返回值为 `(Xnext, cache_or_None)`，仅用于新 core 的局部传递；不返回 raw 历史供任务使用。`KCDNO.forward` 正式接口始终返回一个 Tensor。没有注册到八任务 factory，也没有新增任务脚本、加载器或 resume。

## B. 实际变更与边界

| 文件 | 新增/修改及原因 |
|---|---|
| `cdlno/kcdno/core.py` | 新增 `_check_config`、`_check_input`、`KCDNOBlock`、`KCDNO`；独立组装、边界验证和局部历史 |
| `tests/test_kcdno_core.py` | 新增 10 项针对性测试、独立公共原语组合、同权重白名单和实际调用计数 |
| `docs/kcdno_audit/k3/verify.py` | 独立证据脚本：实际默认参数/调用、限定 9 份旧 core 回放、源码 hash/Python 3.10 语法检查 |
| 本报告、`docs/kcdno_audit/k3/*` | 公式/类/参数/时序、日志、快照索引、diff 和验证证据 |
| `docs/KCDNO_IMPLEMENTATION_STATUS.md`、`memory/current-state.md` | 增量写入 K3 状态，保留历史原文 |

开始时 main/`222647fef343e9fe929e65412f5c7449ced5517b`，工作区已有未提交的 K1/K2 源码、测试、证据和 STATUS/memory 修改，全部保留。[before.json](kcdno_audit/k3/before.json) 记录当时 status、422 份文件 hash、环境与 `/home/hwz/CDLNO-artifacts/k3-before-bd2pluhq/source` 快照。

本阶段代码 diff 见 [code-changes.patch](kcdno_audit/k3/code-changes.patch)，完整阶段文本 diff 见 [stage-changes.patch](kcdno_audit/k3/stage-changes.patch)。不能把 git status 中已有的 K1/K2 未跟踪文件全部计作 K3 新实现。

## C. 公式、参数与执行时序

下表层编号采用论文 1…L；类的 `layer_index` 使用 0…L−1。

| K3 公式/部件 | 真实代码 | 形状、参数归属与验证 |
|---|---|---|
| H=PointNorm(X) | `block.point_norm` | 独立 RMSNorm，H `[B,N,d]`；同一个 H 送 Down 和 Up query |
| S=Down(H) | `block.down: modules._DownAttention` | S `[B,M,d]`；learned query `[M,h,d/h]` 是直接 Q，无额外 Wq/query residual |
| U=S+FFN1(Norm1(S)) | `latent_norm_1`、`latent_ffn_1` | 独立 RMS、两层 bias=True Linear/GELU，hidden2d |
| Uhat=Reader(U,past)，首层 Uhat=U | `reader: KernelHistoryReader` | 仅接收层 2…L/all 注册；没有 SA 或 SA norm 残余；source softmax `[B,M,l]` |
| T=Uhat+FFN2(Norm2(Uhat)) | `latent_norm_2`、`latent_ffn_2` | live T `[B,M,d]`，第二个独立 FFN；历史唯一抽取点 |
| V=X+Up(H,T) | `up_latent_norm(T)` → `up: modules._UpAttention` | latent pre-RMS **一次**；Up 原语本身只做投影后的 per-head Q/K norm |
| Xnext=V+PointModule(NormOut(V)) | `point_ffn_norm`、`point_ffn` | 外层 RMS。point 为 PlainFFN；conv 为 dense3×3/groups1→内部 LN→无偏置 fc1→GELU→有偏置 fc2 |
| 非末层写 Writer(T) | `writer: KernelHistoryWriter` | 仅源层 1…L−1/all 注册。Writer 输入与 Up pre-norm 输入为同一 raw T 对象，不写 H/norm(T) |
| 层间局部历史 | `KCDNO.forward` | 先构造 tuple 快照、执行本层、再 append 本层 cache；最终仅返回点特征 |

Down/Up 的 Q/K/V 投影无 bias，O 有 bias；Down 不存在 `to_q` Linear。两者各自 per-head RMSNorm，SDPA scale=(d/h)^−1/2。kernel 继续使用 K2 的单组 r、raw T value、sum 摘要及 FP32 核/来源计算，不套 head 缩放。每层独立参数对象和底层 storage。

初始化由每个原语自行完成：普通 Linear trunc_normal(.02)/bias0，Down Q 正交，Conv 原生初始化，norm scale1/bias0（适用处）；K2 Wq/Wk Xavier gain1、无偏置，w0、无约束标量 gamma=.1。父 block/core 不 `.apply()` 或重置这些值。初始化 spy 实测每层 13 次普通 Linear 初始化；all L4 有 4 次 Down 正交及 6 次 kernel Xavier，目标参数不重复或交叉覆盖。

默认 d128/h8/M64/r16 的**实际注册参数**（不含任务 lift/LN_out/head）：

| 部件 | 单层参数 | all 中数量 |
|---|---:|---:|
| PointNorm、Norm1、Norm2、Up latent norm、NormOut | 每个 128 | 各 L |
| Down（含 learned Q、per-head norms） | 57,504 | L |
| latent FFN1 | 65,920 | L |
| latent FFN2 | 65,920 | L |
| Up（含 per-head norms） | 65,696 | L |
| PointFFN | 65,920 | point 模式 L |
| ConvFFN（含 dense conv、内部 LN） | 213,504 | conv 模式 L |
| 源 Writer（Wk、key RMS） | 2,176 | L−1 |
| 接收 Reader（Wq、两 RMS、w、gamma） | 2,433 | L−1 |

| L8 核心 | history=all | history=off |
|---|---:|---:|
| point_ffn | 2,605,063 | 2,572,800 |
| conv_ffn | 3,785,735 | 3,753,472 |

参数均可训练，已删除的 SA 分支不注册任何参数。独立交叉核算：point 公共单层 `19d²+16d+4(d/h)+Md`；conv 公共单层 `28d²+17d+4(d/h)+Md`；all 历史增量 `(L−1)(2dr+4d+1)`。实测与这些总式一致。参数数量不代表延迟、FLOPs 或性能收益。

L8/all 的历史时序：

| 当前层 l | 读取旧源 | 逻辑份数 | 本层 Writer |
|---|---|---:|---|
| 1 | 空 | 0 | T1 → cache1 |
| 2 | cache1 | 1 | T2 → cache2 |
| 3 | cache1…2 | 2 | T3 → cache3 |
| 4 | cache1…3 | 3 | T4 → cache4 |
| 5 | cache1…4 | 4 | T5 → cache5 |
| 6 | cache1…5 | 5 | T6 → cache6 |
| 7 | cache1…6 | 6 | T7 → cache7 |
| 8 | cache1…7 | 7 | 无 writer |

| L8 实际操作 | all | off |
|---|---:|---:|
| Down / Up | 各 8 | 各 8 |
| latent SA | 0 | 0 |
| latent FFN | 16 | 16 |
| PointModule、Up latent pre-norm | 各 8 | 各 8 |
| Down/Up 的 SDPA API | 16 | 16 |
| Writer / Q projection / Reader API | 各 7 | 0 |
| 逻辑历史读取份数 | 28 | 0 |
| numerator einsum API / denominator einsum API | 各 7 | 0 |
| 统一 source softmax | 7 | 0 |

一次 batched Reader 同时处理多个来源；7 次 Reader 不等于只读 7 份历史。L1/all 与 off 都没有 Writer/Reader 参数。all 的 gamma 即使置零仍执行历史读写，其计算成本不等于 off。

## D. 实际验收、命令与数值边界

实际 Python3.13.9、torch2.13.0+cu130/CUDA13.0、RTX5090 Laptop GPU，保留现有依赖。远端 Python3.10/Torch2.11/cu128 未执行。没有安装或主动引入 attention 框架；初次 GPU 加载的环境头文件 `_POSIX_C_SOURCE` warning 已保留，测试没有失败。

```bash
# 仓库根目录，只运行合成/配置/公共原语检查
PYTHONPATH=tests:. python -B -m unittest test_kcdno_core test_kernel_history test_kcdno_config test_modules -v
python -B docs/kcdno_audit/k3/verify.py core
python -B docs/kcdno_audit/k3/verify.py old-core
python -B docs/kcdno_audit/k3/verify.py freeze
git diff --check
```

这些命令也可在同步后的远端 checkout 根目录运行。`old-core` 需要索引指向的真实 K0 本地权重夹具；未同步夹具时不能伪造通过，先只执行无夹具依赖的 unittest/core。远端检查不包含数据读取或训练，GPU 用例只在实际 CUDA 可用时运行。

| 检查 | 实际结果/证据 |
|---|---|
| K3 新测试 + K2 + K1 配置 + 旧公共模块 | **57/57 通过，9.912s**；10+16+13+18，[final-tests.txt](kcdno_audit/k3/final-tests.txt) |
| 深度与边界 | L1/2/4/8/12，all/off 共 10 个小配置；B2，M1/3/4/5、r2/3/5/7（独立于 h），point 与 5×7 conv；活跃参数梯度存在且有限 |
| 独立一层/两层公式 | point/conv 共 4 例；禁止调用 core/block/Writer/Reader.forward，手工公共原语+K2 double 显式正值核 reference；输出 maxabs **3.65225202e-7**，全部输入/参数梯度 maxabs **1.95877881e-6** |
| 容差 | 手工输出 atol2e-6/rtol3e-5；梯度 atol2e-5/rtol3e-4。参考始终 double；生产 kernel 即使 `.double()` 也仍 FP32，未把它当 double oracle |
| Up norm / T / 残差 | hooks 验证 pre-norm 一次、Up query 为 H、cache 输入为同一个 pre-Up T，T=Uhat+FFN2；reference 使用非平凡 norm/w/gamma/FFN2 bias，避免初值掩盖错误 |
| gamma0 与 off | 测试白名单显式复制公共权重，FP32/dropout0，point/conv L4 输出 **atol=rtol=0**；all 仍 3 写/6 读，off 为 0。生产没有迁移或 strict=False |
| 历史因果与梯度 | tuple 长度严格 0…L−1、逐对象核对过去摘要、无当前/未来；最早 T/Wk 对最后 Reader 的隔离损失及最早 Wk 对最终输出都有非零有限梯度；K2 隔离读取梯度用例同时重跑 |
| 初始化与参数 | 两 FFN/Down/Up/point 独立，首层无 reader、末层无 writer、L1/off 无历史参数；对象/storage 无共享；普通/特殊初始化目标分别只执行一次 |
| forward 隔离 | eval A(N11)→B(N19)→A 完全一致，摘要对象跨调用独立、旧摘要未改、两次 backward 图独立；没有 module cache、attention buffer 或调试 Tensor 常驻 |
| 合法/非法 | 非整数/空维/非法 history/不支持 hidden/错误层号/缺失或错误 tuple/非方形 grid 与 N 不匹配均拒绝 |
| 新 core strict 权重往返 | all/off × point/conv 4 例，保存 resolved config+state_dict，strict=True 输出完全一致；已学习负 gamma 保留，错误 history/r 的不匹配权重拒绝 |
| 本机 GPU 新 core | B2/N35/d8/L2/M3/r5，point/conv 各 FP32 CPU 对照及 GPU backward；另 FP16/BF16 AMP 各一，共 2 FP32+4 AMP 例通过。MATH SDPA，TF32关，compile关；AMP 与 FP32 输出 atol3e-3/rtol1e-2、梯度有限，不宣称完整 AMP 梯度逐参数等价 |
| K2 精度原检查 | double缓存/显式、2 gradcheck、CPU BF16、4 GPU AMP实际 FP32/autocast关闭路径及梯度、1e15 极端输入等重跑通过，原容差未改 |
| 旧 core 同权重回放 | **9/9** K0 `full/no_sa/identity × off/entry/every_block`，旧输出和输入/参数诊断梯度 **atol=rtol=0**，[old-core-replay.json](kcdno_audit/k3/old-core-replay.json) |
| 实际默认结构计数/参数 | 默认 d128/h8/M64/r16/L8，4 个 point/conv × all/off 实例实际计数；[core-evidence.json](kcdno_audit/k3/core-evidence.json) |

CPU 新数学测试使用现有 CPU SDPA 分派；默认参数计数/旧回放/GPU 对照明确强制 MATH。测试只临时设置单线程或精度控制并恢复，不更改生产全局默认。未计时性能，所以没有 warmup、median/p90 或训练速度结论。

新 core 的 strict state_dict 检查不能单独识别所有相同形状的架构差异（例如 h 或 L1 时 all/off）；实际任务加载仍需 K1 resolved metadata 先核对后 strict 权重加载。K3 不把独立内存往返测试宣称为八任务 checkpoint 已接入，也没有增加 resume。

## E. 旧模型与冻结区段

[freeze.json](kcdno_audit/k3/freeze.json) 对照本轮开始的 422 文件：420 文件字节完全相同，仅本阶段 STATUS 与 memory 增量更新；没有缺失或意外修改。K1/K2 原源码/测试/证据、`cdlno/modules.py`、旧核心/CDPA、三子项目全部模型/数据/训练/评估/指标、脚本、profiles、checkpoint 工具、依赖和 AGENTS 保留。

只新增一份生产 core 文件，没有修改旧测试逻辑。复用相关公共模块 18 项回归及 9 份旧 core 数值夹具已执行；未受触及的 K0/K1/K2 八任务、工业 pickle/PyG、三原 cwd 和旧 Transolver 既有结果保留，本轮没有重复整个任务矩阵，也不改写为 K3 新验收。

## F. 已完成自审与剩余依赖

交付前逐条自审：

1. **唯一计算图**：两个 FFN 之间仅 K2 Reader，无 SA/norm 残余、bridge/rear/head/final Up。公共模块未修改；独立两层参考前反向通过。
2. **raw T 与历史寿命**：一次 Up latent norm，writer 接收同一 raw T；首末层/L1/off 参数和 tuple 时序均由实际 hooks 验证；A→B→A 和隔离梯度通过。
3. **初始化与独立性**：无父级递归初始化；Xavier/orthogonal/trunc 分别一次；参数/storage 不共享，零 w 时 depth norm 首步零梯度没有误判图断开。
4. **计数与成本陈述**：28 份逻辑读取和 7 次 batched Reader 分开，Down/Up 仍是 attention；gamma0 不作为 off 成本证据；实际参数包含全部 core 部件。
5. **回归与范围**：420 文件 hash、9 旧同权重 core、57 联合检查均通过；任务 adapter/原损失/工业加载/远端训练不在本阶段成果内。

没有发现需要用户裁定的实质结构冲突或未修复缺陷。尚未验证/实施：KCDNO 八任务 wrapper 与 train/eval 选择、原任务 loss/normalizer/时间循环接入、工业 PyG/整对象/list checkpoint、新家族 task sidecar 与输出头绑定、远端 Torch2.11/cu128、性能测量、真实数据读取完整性、真实训练/收敛/准确率/epoch 效率。没有新增 activation checkpoint；当前 tuple 快照为后续实现保留正确的局部状态合同。

本K阶段结束，未执行下一阶段。
