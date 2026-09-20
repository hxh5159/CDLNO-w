# Looped LinearNO LL5：三种残差模式

**唯一状态：PASS。LL5 完成；未接生产任务，未执行 LL6–LL10。**

## A. 范围、来源与结论

当前 checkout `/home/hwz/CDLNO`，remote `https://github.com/hxh5159/CDLNO-w.git`，branch `main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，HEAD tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。日期 2026-09-20。本轮从用户已审查通过的 LL4 增量实现 LB，不重建旧基线。

读取根 AGENTS、LL0–LL4 报告/状态、现有 schema/三套 wrapper/core/LL2 body 与 AttnRes 原语/测试；核对 [冻结规格 §4.3 与 LL5](../PLAN_Looped_LinearNO/Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md) 及用户本轮九项要求。前一阶段文件 hash 复核见 [prior-stage-recheck.json](loop_linearno_audit/ll5/prior-stage-recheck.json)，没有 LL4 交付后的差异。未重新下载论文或来源。

设计优先级为本轮用户要求与冻结规格。历史讨论中曾建议通过累加 raw branches 获得 Delta；本轮明确要求实际 `Y−H`，实现和 oracle 均以此为准。点域 AR 原语来自 LL2；LB 的 anchor/轮增量来源和共享 LinearNO core 组合是本项目适配，不是 LinearNO/Kimi 官方发布的完整模型。未引入旧 A/K。

三个正式 mode 已统一使用既有 `RESIDUAL_MODES` 枚举：`sr_1_over_r`、`rb_attnres`、`lb_attnres_1_over_r`。未知 mode、bool、组合字符串与额外开关拒绝；LL1 schema/version/含义无需迁移。三套稳定类路径保持：

- `cdlno.linearno_loop.standard.LoopedStandardModel`
- `cdlno.linearno_loop.airfrans.LoopedAirfRANSModel`
- `cdlno.linearno_loop.shapenet.LoopedShapeNetModel`

## B. 文件与差异

| 文件 | 本轮改动 |
|---|---|
| `cdlno/linearno_loop/core.py` | 允许完整三枚举；LB 专属 boundary/output receiver；forward-local `_lb_forward`；router 数量、独立性、模式互斥检查 |
| `cdlno/linearno_loop/construction.py` | 仅更新公共初值函数文档，注明 RB/LB receiver 不消耗 RNG；实际 config 构造复用 core validator |
| `tests/loop_linearno/test_sr_core.py` | 把旧“LB 尚未实现”断言改为三 mode 可构造；旧 SR 数值/共享/计数/strict 断言不变 |
| `tests/loop_linearno/lb_oracle.py` | 在 LB 生产编辑前建立独立完整 PCRS 方程：LN、erf GELU、LinearNO 因子公式、轮内 residual、实际 Delta、point AR、suffix/head |
| `tests/loop_linearno/lb_support.py` | 仅测试使用的 live hooks、来源/权重/事件记录、非零 router fixture |
| `tests/loop_linearno/test_lb_core.py` | 6 方法：36 数学案例、逐轮/梯度/SGD、24 训练 dropout/AdamW、R1、来源时机、实际相减、局部历史 |
| `tests/loop_linearno/test_modes.py` | 4 方法：48 全 profile 构造、48 小 wrapper 更新、9 新进程 strict reload、模式负面矩阵、6 个计算量实测 |
| 本报告、`LOOP_LINEARNO_IMPLEMENTATION_STATUS.md`、`loop_linearno_audit/ll5/` | 阶段记录、原文件备份、命令与日志、错误定位、数值、diff、完整冻结、自审 |

[source.diff](loop_linearno_audit/ll5/source.diff) 包含相对本阶段开始快照的真实 diff，包括此前尚未 tracked 的 core 文件。三套 wrapper、LL2 原语、纯 schema、旧模型、八任务 parser/factory/train/eval/checkpoint/launcher/monitor 均未改。

调用链仍只用于内部合成验证：`LL1 resolve/validate_config → build_from_config → 稳定 wrapper → 原 stem/position/time → prefix → 共享 core 的单一 mode → suffix → 唯一末层 head`。本轮没有生产 CLI 或新的任务训练路径。

## C. 数学、状态与三模式合同

令 hidden width 为 H（不是 head 数或网格高），prefix 输出为 `a`，`H1=a`。LB 第 r 轮的每个物理 core block 都执行：

\[
x' = x + \frac{1}{R}\operatorname{Attn}(\operatorname{LN}_1(x)),\qquad
x''=x'+\frac{1}{R}\operatorname{MLP}(\operatorname{LN}_2(x')).
\]

一轮后保存真实入口、出口关系：

\[
Y_r=\Phi_{1/R}(H_r),\qquad \Delta_r=Y_r-H_r.
\]

若有下一轮，`H(r+1)=lb_boundaries[r−1]((a,Delta1,...,Deltar))`；最终 `z=lb_output((a,Delta1,...,DeltaR))` 才进入 suffix。每个 receiver 沿用 LL2 的 query=0、norm_scale=1、eps1e-6、source softmax、raw value。**没有轮内 AR、Delta 再缩放、来源数补偿或 anchor 额外残差。**

`entry=x`、`x`、`deltas`、`sources` 仅在 `_lb_forward` 内存在。历史值不 detach、不放 buffer/self，不跨 forward；测试 hook 用于记录 H/Y，正常 forward 不保存额外日志或计算图。Delta 由实际两个点状态相减，测试还拦截 `Tensor.__sub__` 验证传入/返回对象身份，避免数学上类似的 raw-sum 实现绕过要求。

R2、零 query 时严格是来源平均：

\[
H_2=(a+\Delta_1)/2,\qquad z=(a+\Delta_1+\Delta_2)/3.
\]

精确可表示的标量 fixture：`a=3`、operator=2x、MLP=2x、R2，得到 `Y1=12, Delta1=9, H2=6, Y2=24, Delta2=18, z=10`，两条等式逐位通过。一般浮点输入下乘 reciprocal 与求和后除法存在舍入差，按 FP64 容差验证，不更改 LL2 运算顺序。

R1 的语义仍包含 output AR，不退化为 SR：operator=2x、MLP=3x、anchor=1 时 `Y=12, Delta=11, z=(1+11)/2=6`；只注册一个 output receiver，无 boundary。正式 preset 仍为 R2。

| mode | 轮内 | 来源与路由时机 | receiver 数 | 新参数 |
|---|---|---|---:|---:|
| SR | 每条 branch 乘 1/R | 无 router | 0 | 0 |
| RB | raw branch；无 1/R | 每个 sublayer 前读 anchor/completed/raw partial，末尾 output | 2CR+1 | 2H(2CR+1) |
| LB | 每条 branch 乘 1/R | 仅轮间/output 读 anchor 与实际 Delta | R | 2HR |

LB 新键只有 `loop.lb_boundaries.<i>.query/norm_scale` 与 `loop.lb_output.query/norm_scale`。RB 只含原 `loop.rb_*`，SR 二者都没有。core 算子仍为 `loop.core.<physical_index>.block.*`，跨轮复用同一对象。receiver 按边界/output 独立，参数共享规则不混淆。构造仍只创建 U=P+C+S 个物理 block，最后 suffix 是唯一 head；无二次 apply、无删除多余 blocks、无新随机初始化消耗。

序列化复用 LL1：读取 metadata → 校验/恢复 config → 构造 → `torch.load(weights_only=True)` → `load_state_dict(strict=True)`。显式 mode 冲突在 import/读权重前报错；三 mode 错误 state 互载也分别 strict 拒绝。这里验证的是合成 schema + state_dict 存取，**不是已接入八任务的完整 optimizer/RNG resume**；测试 metadata 中的 resume/data/protocol 块仍明确为合成 fixture。

## D. 测试与数值证据

环境：CPU，Python3.13.9、torch distribution2.13.0、NumPy2.2.6、PyG2.3.1、timm1.0.28、einops0.8.2；完整信息见 [environment](loop_linearno_audit/ll5/environment.json)。所有测试设 `CUDA_VISIBLE_DEVICES=''`，不导入读取真实数据的 exp 入口。

最终 **74 方法：72 pass、2 CUDA skip、0 failure/error，33.399s**。10 项 LL5 新方法全部通过。见 [完整日志](loop_linearno_audit/ll5/loop-tests.log)、[LB 逐张量](loop_linearno_audit/ll5/lb-report.json)、[模式/参数/MAC](loop_linearno_audit/ll5/modes-report.json)、[自审](loop_linearno_audit/ll5/delivery-review.json)。

| 检查 | 实际覆盖/结果 |
|---|---|
| 独立全 core oracle | 六 attention variant × P1/P2/customP0C2R3S1 × FP64/FP32，共36案例；独立 oracle 不调用生产 forward |
| 逐张量 | H/Y/Delta、所有 source、权重、route、physical visit、最终输出、loss、输入和每参数梯度、SGD一步全部比较；Air dead temperature 保持无梯度 |
| 梯度/历史 | 非零 query 下所有 LB router 参数 finite/nonzero；隔离最终 route 对旧 Delta 和输入做 VJP；历史不断图；异常后、更换B/N后无旧输入依赖/缓存 |
| 时机/缩放 | hook 事件精确为每轮C次physical block再一次router；来源始终原anchor+实际Delta；R1、R2、R3及精确手算反例通过 |
| AdamW | 24案例：六任务变体×两preset×两dtype；手写 native 算子展开，原 attention dropout=.2；输出/梯度/权重/optimizer state/RNG逐位相同；未引入 history dropout |
| 公共初始化 | 八任务×两preset×三mode，48次实际 full-profile 构造，公共 state键/shape/value、参数共享和构造后RNG逐位一致 |
| 三mode一步/strict | 八任务×两preset×三mode，48个小shape wrapper，forward→合成平方loss→backward→AdamW→新对象strict reload→eval逐位相同 |
| 新进程 | Standard/Darcy、AirfRANS、Car × 三mode，9个 metadata 先读的 strict 重载，从临时cwd的新解释器输出逐位相同 |
| 负面 | mode未知/非字符串/非法多开关、不相符metadata、strict=False、六种错误跨模式state加载、receiver别名和非当前mode的router注入都拒绝 |

独立 oracle 的一步更新使用 SGD；AdamW 用独立控制流但相同 native 算子做逐位对照，沿用 LL3/LL4 对 AdamW 小梯度分母敏感性的验证方式，没有更改生产训练 optimizer。

| LB 数值项 | FP64 最大 abs | FP32 最大 abs |
|---|---:|---:|
| 所有对照张量 | 5.3290705e-15 | 7.3909760e-6 |
| 最终输出 | 4.1633363e-17 | 2.9802322e-8 |
| 梯度 | 1.0659876e-15 | 6.3562766e-7 |
| 传播后的 source weights | 3.5527137e-15 | 3.5762787e-6 |
| 各张量 mean abs 的最大值 | 3.6435969e-16 | 2.1860251e-7 |

FP64 使用 `atol=1e-12,rtol=1e-10`；FP32 输出/状态/梯度/更新使用 `1e-6,1e-5`。**只有独立全 core 传播后的 FP32 LB source weights 使用单列的 `1e-5,1e-4`**；实际相同输入来源的独立 AR 对照仍用原 LL2 容差。全部张量 max/mean abs、max/mean relative 见 JSON，relative 分母 floor1e-12。所有张量汇总的最大 relative 为 FP64 9.49e-9、FP32 6.19635（近零梯度）；这是诚实记录的比例误差，不能把它当整体输出相对误差，验收使用逐元素 `atol+rtol*abs(reference)`。

首轮 6 方法中有2个数值断言失败，原日志和报告保留为 [lb-first-tests.log](loop_linearno_audit/ll5/lb-first-tests.log)、[lb-first-report.json](loop_linearno_audit/ll5/lb-first-report.json)：

1. 精确均值 fixture 使用 /3 与逐项乘1/3，在 FP64 相差4.44e-16；正式浮点均值按既定 FP64 容差检验，另选可精确表示的手算例逐位证明两条 R2 等式。没有修改原语。
2. conv/customR3 的独立 FP32 oracle 在第二轮 source weights 相差3.57628e-6。真实 Delta 最大仅约0.002918，`Y−H` 的绝对差2.98e-8/5.96e-8经 RMS（eps1e-6，局部放大可达约1000）传播到权重。把**同一实际 source**交给独立 AR oracle，权重差降为2.98e-8；FP64 对应权重差9.99e-16。最终输出该例差1.86e-8。记录见 [lb-roundoff.json](loop_linearno_audit/ll5/lb-roundoff.json) 和可复算脚本；只为这类传播权重建立上述明确预算，未改模型公式或 LL2/SR/RB 容差。

## E. 参数与实际计算量

以下为真实 full-profile 构造的 `sum(p.numel())`，profile为默认 paper_table8_on_release_model，rank multiplier2。H128适用于Airfoil/Darcy/Elasticity/Pipe/Plasticity；H256适用于NS/AirfRANS/Car。

| preset | U / 执行depth | mode | receiver | H128新增参数 | H256新增参数 |
|---|---|---|---:|---:|---:|
| P1C3R2S1 | 5 / 8 | SR | 0 | 0 | 0 |
| P1C3R2S1 | 5 / 8 | RB | 13 | 3,328 | 6,656 |
| P1C3R2S1 | 5 / 8 | LB | 2 | 512 | 1,024 |
| P2C2R2S2 | 6 / 8 | SR | 0 | 0 | 0 |
| P2C2R2S2 | 6 / 8 | RB | 9 | 2,304 | 4,608 |
| P2C2R2S2 | 6 / 8 | LB | 2 | 512 | 1,024 |

| 任务 | H / actual M | P1 SR / RB / LB总参数 | P2 SR / RB / LB总参数 |
|---|---|---|---|
| Airfoil | 128 / 128 | 1,126,737 / 1,130,065 / 1,127,249 | 1,345,249 / 1,347,553 / 1,345,761 |
| Darcy | 128 / 128 | 1,126,993 / 1,130,321 / 1,127,505 | 1,345,505 / 1,347,809 / 1,346,017 |
| Elasticity | 128 / 128 | 388,817 / 392,145 / 389,329 | 459,745 / 462,049 / 460,257 |
| Pipe | 128 / 128 | 1,126,737 / 1,130,065 / 1,127,249 | 1,345,249 / 1,347,553 / 1,345,761 |
| NS | 256 / 64 | 2,192,385 / 2,199,041 / 2,193,409 | 2,593,025 / 2,597,633 / 2,594,049 |
| Plasticity | 128 / 128 | 1,160,324 / 1,163,652 / 1,160,836 | 1,378,820 / 1,381,124 / 1,379,332 |
| AirfRANS | 256 / 64 | 2,173,228 / 2,179,884 / 2,174,252 | 2,573,876 / 2,578,484 / 2,574,900 |
| Car | 256 / 64 | 2,469,460 / 2,476,116 / 2,470,484 | 2,935,908 / 2,940,516 / 2,936,932 |

计算量使用实际小型 Darcy wrapper：`B2,N15,H8,heads2,M4,grid3×5`，两preset、三mode，共6次forward。Linear/Conv hooks 和 TorchDispatch 的实际 KTV/QC bmm 计主干 MAC；receiver 内沿 channel/source 的两次 multiply-sum 记录真实输入shape/轴/调用次数，计 contraction MAC。singleton直接返回anchor，没有虚报AR开销。

| preset | mode | 主干MAC | 额外router contraction MAC | 两类MAC合计 |
|---|---|---:|---:|---:|
| P1 | SR | 243,600 | 0 | 243,600 |
| P1 | RB | 243,600 | 14,400 | 258,000 |
| P1 | LB | 243,600 | 2,400 | 246,000 |
| P2 | SR | 243,600 | 0 | 243,600 |
| P2 | RB | 243,600 | 9,600 | 253,200 |
| P2 | LB | 243,600 | 2,400 | 246,000 |

这些不是完整 FLOPs 或延迟。LB 实测来源数 `[2,3]`，两次score收缩+两次value收缩，等于 `2BNH(2+3)=2400` MAC；与理论 `BNH*R*(R+3)`一致。额外真实 `Y−H` 是2次各240元素相减，合计480标量减法，不计为MAC。

RMS/softmax/逐元素工作单列于 `router_operator_inventory`：LB实测2次square（1200元素）、2次mean（150输出）、2次eps add/rsqrt（各150输出）、8次mul（4800输出，其中包含收缩用乘法，不能再全量重复加计）、4次sum（630输出）、2次softmax（150输出）、2次stack。完整报告列出RB对应实测；不能把 `2×MAC` 当含这些工作的总 FLOPs。只计一次前向，没有测运行时间、内存峰值、反向完整FLOPs或真实N的epoch速度。共享物理参数5/6套都执行8次，不能由参数下降推断训练变快。

## F. 命令、冻结、未运行与自审

实际最终命令（仓库根目录）：

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LOOP_LL2_ATTNRES_REPORT=docs/loop_linearno_audit/ll5/attnres-report.json \
LOOP_LL2_BODY_REPORT=docs/loop_linearno_audit/ll5/body-report.json \
LOOP_LL3_PARITY_REPORT=docs/loop_linearno_audit/ll5/sr-parity.json \
LOOP_LL3_ACCOUNTING_REPORT=docs/loop_linearno_audit/ll5/accounting.json \
LOOP_LL4_REPORT=docs/loop_linearno_audit/ll5/rb-report.json \
LOOP_LL5_REPORT=docs/loop_linearno_audit/ll5/lb-report.json \
LOOP_LL5_MODES_REPORT=docs/loop_linearno_audit/ll5/modes-report.json \
python -B -m unittest discover -s tests/loop_linearno -p 'test_*.py' -v \
> docs/loop_linearno_audit/ll5/loop-tests.log 2>&1

PYTHONDONTWRITEBYTECODE=1 python -B docs/loop_linearno_audit/ll5/run_existing_regressions.py \
> docs/loop_linearno_audit/ll5/regression-progress.jsonl 2>&1

PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B docs/loop_linearno_audit/ll5/reproduce_roundoff.py

PYTHONDONTWRITEBYTECODE=1 python -B docs/loop_linearno_audit/ll5/finalize_evidence.py
```

旧 LL0 回归重跑24模块/153方法：**147pass、2个历史失败方法（4条断言）、4CUDA skip、0error**。历史失败仍仅 README/path.sh/旧REPRODUCTION_MATRIX静态冻结期望；逐方法结果及完整失败摘要与已审查 LL0一致，不改旧测试来掩盖。pure/legacy组78pass、history59pass、monitor10pass；本轮没有新增失败。[对照结果](loop_linearno_audit/ll5/regression-summary.json)。LL3 SR数值与accounting、LL4 RB数值JSON和本轮结果完整相等。

本轮开始清单2210文件（tracked1549、untracked193、ignored468），全部分类/大小/SHA256复算。既存仅允许上述 core、construction、SR测试、研究STATUS四文件变化，另2206文件内容与分类不变；包括三任务工程、旧生产数学/入口/配置/checkpoint/脚本、LINEARNO/monitor、LL2原语与LL1 schema。新增文件限本轮测试、报告和LL5审计目录；新增ignored仅该目录日志，无新缓存/运行目录。tracked/staged diff和HEAD保持本轮开始状态。见 [start-manifest](loop_linearno_audit/ll5/start-manifest.json)、[最终freeze](loop_linearno_audit/ll5/end-freeze.json)。

交付前自审五项均已检查并有测试/证据：①实际Delta与两条branch的1/R；②仅boundary/output的来源与均值，无补偿；③receiver独立、单枚举互斥及公平公共初值；④metadata先读、48次对象重载/9次新进程与跨mode拒绝；⑤SR/RB数值不变、旧回归无新增失败、全部tracked/untracked/ignored冻结。

未运行：GPU/AMP/compile、远端Python3.10环境、真实数据/完整训练/准确率、完整生产resume、生产parser/factory/launcher/monitor接线、LL6及后续。没有待决数学冲突；实际Delta相减对低精度小增量可能更敏感，未来GPU/AMP验收必须单独量化，本轮未用raw累加替换。上述CPU合成通过不宣称模型性能提升。

本 LL5 阶段结束，未执行下一阶段。
