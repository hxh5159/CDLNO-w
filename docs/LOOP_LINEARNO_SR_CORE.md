# Looped LinearNO LL3：共享 core 与 SR 1/R

**唯一阶段状态：PASS。仅完成 LL3；LL4–LL10 未执行。**

## A. 范围、真值和审计

2026-09-20，目标 `/home/hwz/CDLNO`，remote `https://github.com/hxh5159/CDLNO-w.git`，branch `main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。起点 tracked/staged diff 均为空，但存在用户计划、LL0–LL2 未跟踪文件和 ignored 产物，绝未把“未跟踪”视为可覆盖。

先读根 AGENTS、其引用状态/报告、LL0–LL2 报告/证据、用户冻结规格/LL3 提示词，核对实际三套纯模型、LL1 config/schema/class path、LL2 body/AttnRes 和现有测试。沿用已审查的 fixed-source/基线结论，不重新获取数据或改写旧报告。本阶段真值为用户 LL3 八条要求，以及 [冻结规格](../PLAN_Looped_LinearNO/Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md) §2、§4.1、LL3。

新增内容仅为独立 `linearno_loop` 内部 SR core、三套合成 wrapper、oracle/测试/计数及本报告。RB/LB 可执行模型、八任务 parser/factory/CLI/launcher、生产 checkpoint/resume、monitor/输出接线均未实施。`build_from_config` 是合成构造入口，不是旧 factory 的替换。

起点 [start-manifest](loop_linearno_audit/ll3/start-manifest.json) 保存 **2069 个** tracked/untracked/ignored 文件的分类、size、SHA-256；[status](loop_linearno_audit/ll3/start-status.txt)、[tracked diff](loop_linearno_audit/ll3/start-tracked.diff)、[staged diff](loop_linearno_audit/ll3/start-staged.diff) 保留原状。

## B. 实际改动与构造路径

| 文件 | 职责 |
|---|---|
| `cdlno/linearno_loop/core.py` | `PhysicalBlock` 注册一份原 block；`LinearNOLoopCore` 分开注册 prefix/core/suffix，检查唯一 head、参数不跨物理 block 别名共享，按 SR 执行 |
| `cdlno/linearno_loop/construction.py` | LL1 配置校验、真实 constructor 校验、隔离 CPU seed 的合成构造；三 mode 公共主干初值导出 |
| `cdlno/linearno_loop/standard.py` | `LoopedStandardModel`，原 Standard 输入/位置/time 合同 |
| `cdlno/linearno_loop/airfrans.py` | `LoopedAirfRANSModel`，原 `Data -> [N,4]`、位置距离拼接及 single-graph 校验 |
| `cdlno/linearno_loop/shapenet.py` | `LoopedShapeNetModel`，原 `(Data,geom) -> [N,4]`，位置替换和原字段合同 |
| `tests/loop_linearno/sr_oracle.py` | 先写的独立 SR 全 core 公式，不调用生产 core/body/attention forward |
| `tests/loop_linearno/sr_support.py` | 小型 LL1 config/输入、纯 U 层参数映射、解析计数 |
| `tests/loop_linearno/test_sr_core.py` | 11 项 core/oracle/共享/初始化/梯度/RNG/strict 测试 |
| `tests/loop_linearno/test_sr_accounting.py` | 3 项参数/MAC/非法 factory 测试 |
| 本文、状态、`docs/loop_linearno_audit/ll3/` | 来源、命令、数值、完整差异与冻结清单 |

已有文件仅增量改动：`cdlno/linearno_loop/__init__.py` 的说明（无 eager import）；`tests/loop_linearno/test_artifacts.py` 将“完整模型文件不存在”的 LL1 阶段断言更新为历史 fixture 仍是 `LL1-contract-only`，保留无生产 launcher 的断言；本研究 STATUS 追加 LL3。旧 LL1 合同/catalog/fixture 和顶层无 torch schema 原样保留。

稳定类路径与 LL1 `model_spec` 完全相同。真实流程为：

```text
LL1 resolve_config / validate_config
  -> build_from_config：完整结构/hash校验 -> require_sr -> validate_constructor
  -> 已记录的 CPU seed 下构造对应 Looped*Model
  -> 原 stem / 可选 time_fc -> U 个原 block -> 一次 native whole-tree apply
  -> 最后创建原 placeholder；保留原 reference/pos buffer 的时序
  -> 原模型 forward：输入/位置/时间处理 -> self.blocks 中唯一 loop -> 原输出形状
```

wrapper 不调用原完整模型 constructor，不先建 8 层再丢弃；只复用原 block、stem 类及完整原 `forward`。`LoopForwardView.blocks` 是返回 `(self.loop,)` 的非注册 property，避免重复 state 键。loop 真正注册的路径为 `loop.prefix.i.block.*`、`loop.core.i.block.*`、`loop.suffix.i.block.*`，与 stem/placeholder 一起形成 state_dict；没有 round2/round3 副本。

`PhysicalBlock.forward` 调用已验收 LL2 的非持有型 body adapter，每次重新调用原 LN/Attn/MLP。新 wrapper 模块本身可被 hook，底下的原 Attn、Q/K/V、MLP 也都按 visit 执行。原 block 的 full-forward 不在缩放 core 中使用，防止隐含未缩放 residual/head。

`common_initial_state` 对三种 mode **只返回公共主干 tensor 字典**。它不返回可执行 RB/LB 模型；这两种 mode 调用 `build_from_config` 会在 import/分配前明确报错。后续实际 router 仍须在公共主干初始化之后创建，并重新验证跨 mode 键/值公平性。

## C. 公式 → 实现 → 验证

令 (A_b(x)=\mathrm{Attn}_b(\mathrm{LN}_{1,b}(x)))，(F_b(x)=\mathrm{MLP}_b(\mathrm{LN}_{2,b}(x)))。

- prefix/suffix：(u=x+A_b(x),\ y=u+F_b(u))。
- core 每个 visit：(u=x+\frac1R A_b(x),\ y=u+\frac1R F_b(u))。identity 始终系数 1。
- 唯一最终 head：最后 suffix 完成两条 branch 后，执行 `mlp2(ln_3(y))`。
- (U=P+C+S\)，(E=P+CR+S\)。模块分配数由 U 决定，算子 visit 数由 E 决定。

| 合同 | 实现 | 验证证据 |
|---|---|---|
| P1/C3/R2/S1 | U5/E8 | hook 次序 `P1,C1,C2,C3,C1,C2,C3,S1` |
| P2/C2/R2/S2 | U6/E8 | hook 次序 `P1,P2,C1,C2,C1,C2,S1,S2` |
| custom P0/C2/R3/S1 | U3/E7 | `C1,C2` 重复 3 次，后接 S1；同时验证 R1 |
| 真正权重共享 | `ModuleList core` 被反复调用 | 同 module/native block/parameter id；state 无 round 副本；共享梯度等于独立展开各轮梯度之和 |
| 每次重算 Q/K/V/C/O | 原 Attn forward | Q/K/V hook 每轮值变化；每 visit `K^T V` 与 `QC` 各一次；24 个完整 wrapper 的 MAC hook 也计入重复计算 |
| 两分支都除 R | LL2 body.scaled | 独立公式 36 案例；手算 A(x)=2x、F(x)=3x、R2/C1，core=25x，native suffix 后=300x |
| 单次最终 LN/head | suffix[-1] finalize | ln_3/mlp2 各恰一次；core 没有 head；非法 factory 提前拒绝 |
| 跨轮 autograd | 不 detach、不保存 history | 仅从第二轮 raw Attn loss 对第一轮输出及同一 core.to_v VJP，均 finite/nonzero |
| native 初始化顺序 | 唯一完整 apply、placeholder 最后 | 6 类任务 × 3 拓扑 = 18 次，对同 seed 的原 U 层模型逐键值及构造后 RNG **完全一致** |
| 无跨 forward 状态 | forward 只用局部 x | 连续 forward、异常退出后 B/N 改变；模块属性集合不增加，没有新 tensor cache |
| strict state roundtrip | 先 config 后构造、weights_only/strict=True | Standard/Air/Car 三个独立进程和临时 cwd，输出完全相同；删键必失败 |
| 构造校验 | LL1 validate + core 再验 | C/R/S=0、bool/string 冒充 int、伪造 U/E/M、重复物理参数、非法 core head、宽度/rank 结构损坏拒绝 |

Standard 普通/统一位置、Plasticity 时间输入、Air 距离拼接、Car 原 tuple/通道、Air dead temperature、Car `tempreature` 键及各 variant 的温度/投影均来自原子模块。R1 下八任务完整 wrapper 与对应纯 U 层模型同权重输出逐位相等，且 `type(loop).forward is type(pure).forward`。Plasticity 不同 T 产生不同输出，time_fc 有非零梯度。NS/Plasticity 没有跨物理时间缓存；本阶段未执行真实时序训练循环。

## D. 命令、环境、结果和数值限度

本地 Python 3.13.9、torch 2.13.0+cu130、NumPy 2.2.6、timm 1.0.28、einops 0.8.2、PyG 2.3.1；本轮显式 `CUDA_VISIBLE_DEVICES=''`，全部 CPU。详见 [environment](loop_linearno_audit/ll3/environment.json)。没有安装/升级依赖。

在仓库根执行的最终 loop 测试命令：

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LOOP_LL2_ATTNRES_REPORT=docs/loop_linearno_audit/ll3/attnres-report.json \
LOOP_LL2_BODY_REPORT=docs/loop_linearno_audit/ll3/body-report.json \
LOOP_LL3_PARITY_REPORT=docs/loop_linearno_audit/ll3/sr-parity.json \
LOOP_LL3_ACCOUNTING_REPORT=docs/loop_linearno_audit/ll3/accounting.json \
python -B -m unittest discover -s tests/loop_linearno -p 'test_*.py' -v \
> docs/loop_linearno_audit/ll3/loop-tests.log 2>&1
```

**55 方法：53 通过、0 failure/error、2 CUDA skip，20.902s。** LL3 新增的 14 方法全部通过；2 skip 是本轮 CPU 隔离下的 LL2 CUDA 方法，不能将以前 LL2 CUDA 成功写成本轮 GPU 验收。详细 [log](loop_linearno_audit/ll3/loop-tests.log)、[逐张量数值](loop_linearno_audit/ll3/sr-parity.json)、[数值汇总](loop_linearno_audit/ll3/numeric-summary.json)。

| 对照 | 案例 | atol/rtol | 最大 abs | 各张量 mean abs 的最大值 | 最大 relative / mean relative |
|---|---:|---|---:|---:|---:|
| 独立公式 FP64（含 AdamW） | 18 | 1e-12 / 1e-10 | 8.57821e-16 | 2.22045e-16 | 5.08351e-11 / 1.69519e-12 |
| 独立公式 FP32（含 SGD） | 18 | 1e-6 / 1e-5 | 1.78814e-7 | 1.19209e-7 | 8.41266e-3 / 6.32470e-4 |
| 原子模块独立展开 FP64（含 AdamW） | 18 | 0 / 0 | 0 | 0 | 0 / 0 |
| 原子模块独立展开 FP32（含 AdamW） | 18 | 0 / 0 | 0 | 0 | 0 / 0 |

独立公式覆盖六 attention 变体 × 三拓扑 × 两 dtype 的每 visit、final、loss、输入及所有活动参数梯度。Air 已知 dead temperature 单独记录 grad=None。relative 定义为 `abs(error)/max(abs(reference),1e-12)`；近零梯度处 relative 比值会放大，验收仍使用原先声明的 atol+rtol，没有单独放宽 relative 门槛。原子模块参考独立书写循环而不调用被测 core/body，测试 train dropout=.2，逐位比较最终权重、AdamW step/exp_avg/exp_avg_sq 和随机流。

**保留的一次新测试失败及定位：** 首轮 10 方法有 1 个失败子例，Car/P2/FP32 的独立公式 AdamW 一步；forward 最大差 2.23517e-8，某个 MLP 梯度为 -4.07363e-9 与 -4.10819e-9，差仅 3.45608e-11，经过 `g/(abs(g)+eps)`（eps1e-8）变成参数差 1.74064e-6，超过既定容差。用标量首步公式重算得到**完全相同**的差值。原 [失败日志](loop_linearno_audit/ll3/sr-first-tests.log) 和 [诊断数值](loop_linearno_audit/ll3/adamw-oracle-roundoff.json) 均保留。

没有修改模型或放宽容差。最终独立 FP32 公式使用对梯度误差线性敏感的 SGD 更新；独立 FP64 仍测 AdamW；额外 36 例使用相同 native 数值核、独立控制流对照 AdamW，全部逐位相同。这是明确记录的参考方法拆分，**不宣称原 FP32 数学 oracle 的 AdamW 比较通过**，也没有更改任何训练 optimizer。可复现诊断：

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B docs/loop_linearno_audit/ll3/reproduce_adamw_roundoff.py
```

旧回归实跑命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B docs/loop_linearno_audit/ll3/run_existing_regressions.py \
> docs/loop_linearno_audit/ll3/regression-progress.jsonl 2>&1
python -B docs/loop_linearno_audit/ll3/finalize_evidence.py
```

24 个原模块共 153 方法，147 pass、2 历史失败方法（4 断言）、4 CUDA skip、0 error。pure/原 Transolver、history、monitor 结果与用户批准的 LL0 完全一致，失败 ID 和断言文本也精确对照。历史失败只涉及先前 README/path.sh/纯复现矩阵冻结预期，不改旧 golden；不能称“全仓测试全绿”。[逐模块结果](loop_linearno_audit/ll3/regression-results.json)、[历史结果比较](loop_linearno_audit/ll3/regression-summary.json)。

## E. 参数与 MAC：分开报告存储和执行

以下来自 `paper_table8_on_release_model` 的真实 d/head/ratio、M×2；B=1，N 取该任务标准点数。**全尺寸只构造/计参数，不执行 forward、不读数据**。32 配置（八任务 × 两 preset × rank1/2）解析参数量与实际 `.parameters().numel()` 全等；另外 24 个小型真实 wrapper forward，通过 Linear/Conv hook 和 KTV/QC 形状计数核验每个实际执行 MAC。全部原始行见 [accounting.json](loop_linearno_audit/ll3/accounting.json)。

| 任务 | actual M | P1 unique参数（5块） | P2 unique参数（6块） | 两preset executed参数使用数（8 visits） | P1物理块各一次 MAC | P2物理块各一次 MAC | 两preset实际执行 MAC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Airfoil | 128 | 1,126,737 | 1,345,249 | 1,782,273 | 16,188,402,048 | 19,350,774,144 | 25,675,518,336 |
| Darcy | 128 | 1,126,993 | 1,345,505 | 1,782,529 | 10,379,030,400 | 12,406,192,000 | 16,460,515,200 |
| Elasticity | 128 | 388,817 | 459,745 | 601,601 | 679,435,776 | 808,828,416 | 1,067,613,696 |
| Pipe | 128 | 1,126,737 | 1,345,249 | 1,782,273 | 23,901,268,608 | 28,570,333,824 | 37,908,464,256 |
| NS | 64 | 2,192,385 | 2,593,025 | 3,394,305 | 10,331,619,328 | 12,244,221,952 | 16,069,427,200 |
| Plasticity | 128 | 1,160,324 | 1,378,820 | 1,815,812 | 4,601,618,176 | 5,480,101,632 | 7,237,068,544 |
| AirfRANS | 64 | 2,173,228 | 2,573,876 | 3,375,172 | 80,101,376,000 | 95,043,584,000 | 124,928,000,000 |
| Car | 64 | 2,469,460 | 2,935,908 | 3,868,804 | 90,059,002,880 | 107,197,404,160 | 141,474,206,720 |

executed 参数使用数是按 block visit 多重计数（含 dead parameter）的展开等价值，**不是额外分配的参数量**。物理块各一次 MAC 只作组织对照，实际 forward 应看最右列。MAC 只含 dense Linear/Conv、K^T V、QC；不含 norm/GELU/softmax/位置与时间编码/加法/访存；1 MAC=2 dense FLOPs 也不代表完整 profiler FLOPs。NS 是单次模型调用，非 10 步 rollout；Plasticity 是单次 T 查询，非 20 次训练更新。不由参数减少推论算量下降或速度提升。

## F. 冻结、未运行项与交付自审

完整 [末次冻结](loop_linearno_audit/ll3/end-freeze.json) 对起点 2069 文件逐项比较内容和分类。仅上述 3 个已有研究文件修改；其他 **2066 文件完全不变**。所有旧模型/训练/CLI/factory/metadata/checkpoint/launcher/monitor、顶层纯 schema、LL2 原语、AGENTS 和既存 ignored/untracked 内容均保留。无新缓存/运行目录，新增 ignored 仅 `docs/loop_linearno_audit/ll3/*.log`。HEAD/tree 不变、tracked/staged diff 仍空；新文件和已有 untracked 文件差异另存 [source.diff](loop_linearno_audit/ll3/source.diff)，不让未跟踪文件逃过检查。

| 自审重点 | 已完成检查 |
|---|---|
| 是否真正共享且未误放 head | id、key、重复 call、ln_3/mlp2 单次、非法 factory、梯度累加证据一致 |
| 是否两 branch 缩放而 identity 不缩放 | 独立 36 例、300x 手算、R1 native 输出精确一致 |
| 是否初始化公平且未多构造/二次 apply | 18 次原 U 层完整初值/RNG 相等；三 mode 公共字典相等，unsupported 模式不伪装可执行 |
| 是否把数值测试失败隐藏为通过 | 首次 FP32 AdamW 失败、定位脚本/数值原样保留；数值核独立展开验证与数学 oracle 分别列明 |
| 是否破坏既有模型或夸大验收 | 原153回归精确对照、2069文件冻结；真值/数据/生产流程和远端明确 NOT RUN |

没有新增待用户裁定的架构冲突。未运行/未实现：RB/LB、生产任务与输出/monitor 接线、完整 loop optimizer/RNG resume archive、真实数据/训练、GPU/AMP/compile、远端 Python3.10/torch2.11 cu128、性能计时和精度。当前 strict round-trip 是合成 state_dict 新进程验证，不能称完整生产 train/resume/eval 已完成。

**本 LL3 阶段结束，未执行下一阶段。**
