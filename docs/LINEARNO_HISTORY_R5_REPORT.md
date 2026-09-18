# R5：联合研究模型、内部配置与 checkpoint

## A. 实际读取/审计、范围与 checkout

**唯一阶段状态：PASS。** 本阶段只将已审查的 A、K 集成到完整研究 LinearNO，并提供内部 config/factory/checkpoint 接口。八任务生产 parser、factory、训练/评估和 launcher 均未接入创新。阶段验收与完整回归数字见下文；R0–R4 的历史记录保留。

实施前快照：`/home/hwz/CDLNO-artifacts/linearno-history-r5-before-3h8t_bpd`；实际 `main`，HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`，tree `d74a1b07aa405009992879847aca0c48c75c31f0`。开始时 1242 tracked、83 untracked、330 ignored 文件；没有 stash/reset/checkout/clean/commit/push。新增研究前各阶段的未提交文件均被纳入快照。

依据用户本轮 R5 和阶段提示词 R5，核对了根 `AGENTS.md`、R1 schema/公平初始化 fixture、R2 context/core、R3 A、R4 K、三套 pure/research wrapper、旧 `cdlno/linearno/{profiles,schema,checkpoint,standard_entry}.py`、历史状态与回归测试。关键只读命令是 `git status --short --branch --untracked-files=all`、`git rev-parse HEAD/HEAD^{tree}`、`git ls-files` 三类文件清单、`rg` 符号搜索和普通内容 hash/diff；没有用 merge-base 或三点 diff。

## B. 本阶段决定及来源

- 用户 R5 规定唯一联合顺序和 A0K0 原类/原 schema；本轮直接复用已审查 R2–R4 路径，不新增研究机制。
- R1 固定 feature 真值、raw-cache、运行名、dropout .1 和公平 seed；内部 config 只派生这些字段，保留历史 R1 fixture。
- 现有 pure checkpoint/schema 是旧格式依据；A0K0 委托原 helper，研究 family 使用独立完整格式和 allowlist，不放宽旧校验。
- 用户仅授权内部集成与合成检查；八任务 parser/factory/launcher、真实数据、GPU 和后续阶段保持未执行。

## C. 修改文件、关键 diff 和计算映射

| 文件 | R5 改动及原因 |
|---|---|
| `cdlno/linearno_history/core.py` | 移除 R4 的联合调用拒绝；复用原 K-before-A 顺序与末尾 raw append |
| `PDE-Solving-StandardBenchmark/model/LinearNO_History.py` | 新增 `JointHistoryModel`，先构造纯主干，再注册独立 A、K |
| `cdlno/linearno_history/models.py` | 新增 `AirfRANSJointHistoryModel`、`ShapeNetJointHistoryModel`，继承原任务输入合同 |
| `cdlno/linearno_history/config.py`（新增） | 复用原 profile，独立解析唯一 A/K 字段；派生完整 constructor、signature、seed 和 innovation spec |
| `cdlno/linearno_history/factory.py`（新增） | allowlist class path；A0K0 直接构造原类，其余只构造启用模块；不修改生产 registry |
| `cdlno/linearno_history/checkpoint.py`（新增） | 完整研究 metadata、独立原子 epoch pair、metadata-first 重建、逐键预检及 strict load、内部 resume |
| `linearno_history/schema.py` | 保留 R1 合同；允许已实现的 `r5-internal-v1` 描述并严格检查 code schema 整数类型 |
| `cdlno/linearno_history/__init__.py` | 仅更新阶段边界说明，仍无 eager 模型导入 |
| `tests/linearno/test_history_integration.py`（新增） | 11 项 R5 实际验收，包括 20 配置与独立进程 |
| `tests/linearno/test_history_k.py` | 原“联合功能尚未实现”断言已过期；改为拒绝非 K 算子对象，其余 R4 数学测试未变 |
| 本报告、研究 STATUS/MATRIX、`docs/linearno_history_audit/r5/` | 保存结果、覆盖矩阵、环境、freeze 与审查证据 |

A/K 单机制源码 `attnres.py`、`history_k.py` 和 raw `context.py` 完全未改。没有改旧纯 LinearNO 文档或其他模型状态文件。

### 计算和配置映射

每层仅有以下顺序：

1. `context.before_block(l)` 取得进入当前层前的 tuple；A、K 接收同一 tuple 对象。
2. 原 `in_project_x` 得到 Z，原 `to_q/to_k/to_v` 得到 base logits/V。
3. K 用旧 raw 和本层 `to_k.weight` 修正 K logits；再应用该任务原温度与 `softmax_N`。
4. `C_raw = K^T V`；A 仅从旧 raw 读取，得到 `C_tilde=C_raw+gamma*H`。
5. 当前原 Q 重建 `Q C_tilde`，原 `to_out`、一次 attention 点残差、原 FFN 残差；最后完整层继续 LN/head。
6. 完整 block 后 `context.after_block(l,C_raw)`，每层一次；绝不把 `C_tilde` 入库。

`test_joint_order_raw_once_residual_once_independent_oracles` 对六变体检查上述调用顺序、tuple/张量 identity、raw 次数、温度前后 logits、Q/K/C，以及逐层显式残差/FFN/head 重算。K/A 数值还分别与 R3/R4 的独立 oracle 比较。

三个 task kind 的内部 factory 映射：

| 签名 | Standard | AirfRANS | ShapeNet-Car |
|---|---|---|---|
| A0K0 | `model.LinearNO.Model` | `cdlno.linearno.airfrans.AirfRANSLinearNO` | `cdlno.linearno.shapenet.ShapeNetLinearNO` |
| A1K0 | `model.LinearNO_History.AttnResModel` | `AirfRANSAttnResModel` | `ShapeNetAttnResModel` |
| A0K1 | `model.LinearNO_History.HistoryKModel` | `AirfRANSHistoryKModel` | `ShapeNetHistoryKModel` |
| A1K1 | `model.LinearNO_History.JointHistoryModel` | `AirfRANSJointHistoryModel` | `ShapeNetJointHistoryModel` |

工业研究类位于 `cdlno.linearno_history.models`。Standard 的 `model` 是原 namespace package；内部导入能解析实际目录，并拒绝另一个工程占用该 namespace。生产 factory 未注册这些研究类。

内部 API 是 `config.resolve_config(existing_profile, family='linearno_history', features={...})` → `factory.build_model(config)`。省略 flags 或显式 false/false 都解析为原 family/class。Transolver、CDLNO、KCDNO、MSAR 等不接受这些开关。新模型仍限制 L4–8；它不限制原入口支持的深度。配置只保存两个布尔真值及 A dropout（启用固定 .1），拒绝自填 `feature_signature`；所有派生字段、class path、构造 kwargs、resolved hash 都需重算匹配。

公共主干构造顺序未变；新增模块各用隔离的 CPU feature RNG。20 配置测试先生成一份真实公共主干，再逐键复制到四组合；没有新增 `init_from_baseline` 或跨结构 resume。DataLoader 独立 generator 的 seed 与状态另存。运行名为 `{task}__{protocol_profile}__L{layers}__A{0|1}K{0|1}__seed{seed}`。

## D. checkpoint 与实际测试

研究 metadata 包含原 profile/data/objective/evaluation/provenance/numeric-normalizer/resume/ensemble 各独立 section，再增加完整 `innovation_spec`、`architecture_extension=linearno_history_v1`、`resolved_config` 和签名。保留原 schema/config 版本，implementation version 为 `r5-internal-v1`；额外记录研究源码 hash。R1 的 `r1-contract-only` fixture 仅作历史合同，不能由实际 R5 loader 构造模型。

原 profile 数据、objective、metric 含义不变。研究校验器先完整验证实际 model/innovation/config，再用一个不落盘的基线结构视图复用旧 numeric normalizer/RNG/profile 等验证器；不会把研究存档伪装成 baseline。研究存储格式为 `linearno-history-epoch-pair-v1`，含完整 state+metadata 和配对纯权重，先提交 hash manifest，再发布 latest/final；拒绝改写已提交 epoch。读取只用 `weights_only=True`，构造前比较 A/K/L/M/heads/d_h/variant/profile/schema，全部 key/shape/dtype 再预检，最后 `strict=True`。

A0K0 直接委托现有 `cdlno.linearno.checkpoint.save_pair/inspect_checkpoint/read_pair`；不写 innovation spec、extension、签名或 resolved research config。测试比较了省略/显式 false 的原类、初始化 state、前反向、优化器一步、metadata/manifest 内容和 RNG，并从原 helper 再次读回权重。

内部 resume 先读 metadata、恢复模型，使用调用方显式 optimizer/scheduler factory 重建并恢复状态；检查 generator/sampler 名称集合，恢复这些状态后最后恢复 Python/NumPy/Torch RNG。未来任务接线负责提供与任务相符的 optimizer/scheduler factory；R5 不提供任务训练循环。eval 使用保存的协议，运行参数变化单独返回；resume 拒绝训练/objective/evaluation 协议变化。

环境：Python 3.13.9、torch 2.13.0（cu130 build）、NumPy 2.2.6、PyG 2.3.1、timm 1.0.28、einops 0.8.2。通过 `CUDA_VISIBLE_DEVICES=''` 限定 CPU，未安装依赖。

实际新验收命令：

```bash
cd /home/hwz/CDLNO
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_history_integration -v
```

**11/11 通过，独立最终运行 12.455 s。** 包含以下实际覆盖：

- 4 组合 × L4–8 全部构造、CPU FP32 合成 forward/MSE/backward/AdamW step/save；另外一个新进程从 `/tmp` 读取 20 个 native checkpoint，strict reload 后 eval/state 逐值一致。每例实际 attention/FFN 调用 L 次，最后 LN/head 各 1 次。
- 六变体 × train/eval 的 12 个 A0K0 严格基线比较，`atol=rtol=0`；保存调用确实委托旧 helper。
- 六变体 × 三种门分解的 18 个 CPU float64 比较，`atol=rtol=0`；两门同时非零时所有新增参数梯度存在且 finite，两套参数对象没有交集。
- 六变体的联合顺序和独立 oracle；固定 CPU float64 门限 `atol=1e-12, rtol=1e-10`。具体 max/mean absolute/relative 误差见 `numerical-summary.json`。
- 四组合训练恢复：下一步输出/loss/权重/AdamW/StepLR 状态，Python random、NumPy 20 项排列、Torch RNG 和独立 generator shuffle 均逐值一致。这里是**明确标注的合成 MSE 与短状态闭环**；不会将 StepLR fixture 或提前保存的 synthetic final 解释为 profile 的实际 OneCycle/完整 epoch 实验。
- 12 对跨 A0K0/A/K/AK 的 metadata 错配在 `torch.load` 前拒绝；相同 12 对裸 state 的 key 错配在 `load_state_dict` 前拒绝。depth/M/variant/heads、schema、缺 spec、未知 kwargs、R1 placeholder 合同、缺 RNG/normalizer 等均拒绝；错误磁盘 metadata 即使重算文件校验和仍被语义校验拒绝。
- AirfRANS/Car joint 使用真实 PyG Data/tuple 合同完成额外 native checkpoint 新对象闭环；不是其生产任务 loader/train/eval 验收。
- train 下 A-only/AK 的 .1 history dropout 推进 RNG，K-only 不调用 A、不采样 A mask。**没有声称研究零门 train RNG 等于 baseline。**

20 配置使用 Standard plain、B2/N15/H3/W5/d12/h3/M8、fx1/T[B,1]/out2，是小型合成 fixture。参数量如下，不能当作八任务正式 preset 参数量：

| L | A0K0 | A1K0 | A0K1 | A1K1 |
|---|---:|---:|---:|---:|
| 4 | 3778 | 3933 | 3835 | 3990 |
| 5 | 4530 | 4726 | 4590 | 4786 |
| 6 | 5282 | 5519 | 5345 | 5582 |
| 7 | 6034 | 6312 | 6100 | 6378 |
| 8 | 6786 | 7105 | 6855 | 7174 |

## E. 回归与保护证据

完整回归实际执行 R1 schema、R2 core、R3 A、R4 K、R5 integration、旧 profiles/schema、官方 attention/Standard/Air/Car parity、旧 Transolver 八模型真实 fixture、旧 CLI、随机流/真实可视化测试及 monitor。实际模块列表、命令边界、结果与 traceback 保存在 [R5 evidence](linearno_history_audit/r5/)。

`freeze.json` 基于修改前真实 manifest，覆盖 tracked/untracked/ignored，不只看 git diff。全部 1242 tracked 文件未变；纯 LinearNO 原模型/attention、八任务 parser/factory/entry/launcher、旧 checkpoint、其他研究模型、`LINEARNO/`、`monitor/` 均未改。本轮仅对已审查的 untracked 研究文件作上述增量修改。

## F. 自审与未验证边界

已逐项从代码和测试反查：

1. **顺序与缓存**：K-before-raw-before-A，raw 入库位于完整 block 后；A/K 共享同一旧 tuple，只有 raw 是 authoritative，无 fused/future/current 泄漏。
2. **基线隔离**：A0K0 原类/原 state/原 schema/原 loader；没有 dormant 参数、feature metadata 或额外 RNG。
3. **checkpoint 身份**：allowlist、完整 constructor/spec/profile 一致、先 metadata 后构造、先 key/shape 后 strict 权重加载；没有跨家族补键或 random fallback。
4. **公平性与随机流**：公共主干逐键复制，新增投影 seed 隔离；A train dropout 的额外随机流被明确区分；恢复的下一步包含其真实 mask 采样。
5. **冻结范围**：保留用户 untracked 文件，复算全部分类/hash；A/K 数学和原模型未改。

未运行：八任务生产研究 launcher/CLI、真实数据读取/训练/评估、真实任务 scheduler/optimizer/data-loader 完整研究闭环、有效传播核 monitor 接线、GPU/AMP/远端环境、收敛/精度/epoch 耗时。八任务 32 项研究生产闭环矩阵仍为 NOT_STARTED。本轮无新的架构或数据协议决定需要用户裁定。

R5 内部 checkpoint 验收限 CPU FP32 存档；float64 用于计算 oracle/门分解。分布式/多 worker/mid-epoch 恢复、工业 ensemble 的任务调度属于未接线边界。本轮不把内部合成 API 作为远端任务命令交付。

## G. 阶段结论

**PASS。** R5 新增 11 项验收、20 配置矩阵和所有新增结构/metadata/恢复训练要求均通过。完整联合回归 **114 个测试方法：109 passed、4 GPU skipped、1 个历史冻结方法含 3 个已知失败断言；0 errors，114.623 s**。失败集合与已审查的 R4 完全相同：`README.md`、`path.sh` 的旧 hash，以及 `docs/LINEARNO_REPRODUCTION_MATRIX.md` 的历史前缀断言。三文件本轮均未改变，未修改旧 fixture 来消除失败。PASS 指本 R5 范围验收通过，不宣称历史仓库全部测试全绿。

独立 oracle 的最大全集误差：K combined logits `max_abs=2.776e-17`、逐例最大 `mean_abs=3.857e-18`；A fused `max_abs=4.337e-19`、逐例最大 `mean_abs=2.259e-21`。baseline/门分解/新进程 eval/恢复下一步全部 exact；相对误差及 reference 近零处理见数值 JSON。

**本 R5 阶段结束，未执行 R6 或后续阶段。**
