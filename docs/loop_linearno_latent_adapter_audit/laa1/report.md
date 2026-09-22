# LAA1：V3 配置、metadata、成本与预览合同

**状态：PASS，限本阶段纯配置/schema/解析成本范围。未执行 LAA2。**

## A. 完成范围与来源

checkout 为 `/home/hwz/CDLNO`，`main @ c02e671506f706910e0a1d58f03c310abf188345`，origin 为 `git@github.com:hxh5159/CDLNO-w.git`。实施依据是用户本轮 LAA1、冻结提示词 §2–4/§LAA1、LAA0 reference audit 与当前源码；未假定旧文档的测试数量或远端环境有效。

新增独立 `linearno_loop.v3` 纯配置空间。没有创建 `cdlno.linearno_loop.v3` 生产模型，没有修改版本分派、八任务 parser/factory/train/eval、checkpoint backend、launcher、记录器或可视化。metadata 内的 class path 与 constructor signature 是后续实现合同，不代表对应模型已经存在。

`architecture=operator_latent_adapter_v3` 是唯一显式选择字段；未来 CLI 名称为 `--linearno-loop-architecture`。只给 cost_profile 或省略 architecture 会拒绝。architecture extension 为 `loop_linearno_latent_adapter_v3`；schema/config/checkpoint version 均为整数3；checkpoint format 为 `linearno-loop-epoch-pair-v3`。

默认 `matched_v1 / D12 / P2-C4-R2-S2 / SR / latent on / bilateral_qk_lowrank_second_visit / rank4 / alpha4`。五任务 M64，NS/Air/Car M32，heads8。V1 默认 M×2 和 V2 默认 M×1 完全保留。训练默认只在调用新纯解析 API 且明确选择 V3 时生效；现有任务命令尚不能使用这些 V3 字段。

## B. 文件与修改边界

| 文件 | 用途 |
|---|---|
| `linearno_loop/v3/__init__.py` | 显式 V3 纯 API；旧包不反向导入它 |
| `contracts.py` | 版本、选择器、字段名、strict JSON/hash、未来 CLI 名称 |
| `profiles.py` | 两 cost profiles × 四深度 × 八任务的64组 H/Dz |
| `config.py` | train 配置解析、saved-facts 校验、字段来源、seed派生、run id |
| `schema.py` | metadata-first 合同、signature 检查、显式冲突断言、create-only JSON |
| `costs.py` | 独立于 tensor 模型的参数/MAC/FLOPs分区与非矩阵操作清单 |
| `matrix.py` | 仅配置预览，明确标记尚不可作为生产命令运行 |
| `tests/loop_linearno_latent_adapter/{oracle,support,test_laa1_*}.py`、`__init__.py` | 独立 shape oracle、结构 fixture、配置/schema/成本/隔离测试 |
| 本目录 `run-checks.py`、`verify-preview.py`、`static-audit.py` 及 JSON/log/matrix | 本阶段可复查证据 |
| `docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md` | 只追加本阶段结果，保留 LAA0 原文 |

共新增7个纯实现文件、7个测试/fixture文件；另有本阶段证据脚本/文档。`source-diff.patch` 保存新实现与测试的完整增量。既有 tracked diff 仍仅为用户的 `check_checkpoints/check_pipe_loop_resume.sh`，内容未改。

V3 单向复用旧模块中已审计的纯 profile resolver、AttnRes 合同及 JSON 状态检查；不复制任务科学协议。新进程 import guard 实际阻断 torch、NumPy、模型包与任务模块，仍完成解析、metadata 恢复与成本计算。没有改旧模块的导入行为。

## C. 规格到实现/测试

| 规格 | 实现 | 验证 |
|---|---|---|
| 显式选 V3；cost label 不选版本 | `contracts.ARCHITECTURE_SELECTOR`、`config._request` | architecture 缺失/错误与旧历史字段负例 |
| 两表全部 H/Dz | `profiles.PROFILE_WIDTHS`、`_resolve_facts` | 从原提示词读取64格逐项对照；独立 LAA0 成本证据精确对照 |
| 规范 D12/20/28/60，P=S=2/R=2 | `_topology` | 非法深度、混用、bool/string/NaN、缺字段全部拒绝 |
| custom 显式 H/Dz/拓扑，M/head可独立 | `_request`、`_resolve_facts` | P0/C2/R3/S1、旧两preset及D6/10/16/40 shorthand；Car H200/h8/M31合法，旧v1/v2同M仍拒绝 |
| H与空间网格分离 | `_assemble` | loop/resolution/resolved profile/constructor/preview 的 `hidden_width` 全等；网格保持原 profile事实 |
| full core共享、特性所有权与三残差 | `loop_spec.sharing/residual_contract/attnres` | 闭合字段、伪造派生量拒绝；只是合同，无新forward |
| latent/adapter开关与初始化 | `latent_ffn/adapter/initialization` | off实例/参数为0，Dz仍保存；R≠2且adapter-on拒绝；四消融seed相同 |
| metadata-first/strict | `schema.validate_metadata/restore_config` | 跨family/version/模式/H/M/Dz/topology等断言失败；`strict=False`拒绝；metadata文件不覆盖 |
| 原 profile恢复不重解析 | `config.validate_config`、`schema.restore_config` | mock当前 resolver/宽度表为抛错，saved metadata仍能恢复 |
| 参数/MAC分区 | `costs.analytic_cost` | 独立 tensor-shape inventory、LAA0数值、2304条落盘矩阵三方核对 |

custom 必须明确给 `hidden_width`、`latent_width`、四个拓扑字段或合法 shorthand。M/heads 未显式给时按选定任务 profile 解析；adapter 未显式给时使用已记录的 V3 默认，并保存来源；不是不明来源的隐式覆盖。规范 profile 的 H/Dz/M/heads 允许显式相同值作为断言，不同值必须使用 custom。`explicit_profile_assertions` 保存这些相同值断言，参与 hash。

`base_profile_spec` 保存旧profile原始快照与hash；有效 `profile_spec.values.model` 使用 `hidden_width`、`grid_height/grid_width`，不残留旧的有效 hidden=128/256 字段。旧值只作为明确标识的 base snapshot 保留。cost_profile不覆盖空间网格、数据、objective或evaluation。

metadata保存 §4.2 的版本、task/profile/cost_profile、字段来源、comparator/unique/executed depth、P/C/R/S、H/h/dh/M/FFN ratio/variant/Dz、残差与特性合同、公共/特性/DataLoader seeds、所有权与成本分区，以及 data/normalizer/objective/evaluation/provenance/optimizer/scheduler/scaler/RNG/generator/sampler/ensemble状态。尚未构造模型时实际参数计数严格为 `pending_model_construction` 和 null；不能把 analytic count 填作实测。epoch/global_step推进后不允许 pending。

状态 fixture 是结构测试编码，不是可恢复训练的档案。真实 optimizer 参数对应、权重hash、后端 RNG 恢复和实际 parameter measurement 将在后续接线验收；本阶段没有 `torch.load` 或实际模型加载。hash用于完整性校验，不声称能防止攻击者同时替换metadata及其hash。

run id 的可读部分包含 task、V3、cost profile、P/C/R/S、residual、H/h/M/Dz、特性开关、rank/alpha、seed，尾部是完整 config SHA256；其余会改变数学、state或协议的字段同样进入hash。2304条唯一，最长176字节。此阶段只生成标识，不预留训练目录或覆盖已有运行。

## D. 成本公式与边界

设 `U=P+C+S`、`D=P+CR+S`、`dh=H/h`。block参数按U计，核心执行MAC按CR计，输出head计一次。参数互斥分区：stem/placeholder、time、prefix、shared_core、suffix、head、latent、adapter、router；block区内另分operator、两处norm、point FFN。

- latent参数：`C × [Dz(2H+1)+3H]`；MAC：`2 B C R M H Dz`。
- bilateral adapter参数：`2 C r (dh+M)`；仅第二轮MAC：`2 B C h N r (dh+M)`。
- RB router参数：`2H(2CR+1)`；LB：`2HR`；SR：0。router score/value contraction：`2BNH × sum(source_counts)`，单独报告。
- 每visit `KᵀV` 和 `QC` 各为 `BNHM`。矩阵 `FLOPs=2×MAC`。
- softmax、LayerNorm/RMSNorm、GELU、bias/residual/scale、dropout、温度/clamp、位置距离、sin/cos、索引与reshape另列。清单不假装是完整scalar FLOPs或耗时。

**冻结成本比较只适用于 on/on、r4、alpha4、SR、paper_table8_on_release_model 与代表B/N的矩阵口径。** matched是约匹配；efficient是另一套必做表。RB/LB加router，关闭模块和变更rank时重新算实际解析成本；不同原始profile或显式不同B/N不携带冻结表的匹配声明。只改profile标签不能使不同FFN ratio的计算量变为matched。

NS成本为一次forward，不含十步rollout总成本；Plasticity不含20次时间查询总成本；工业任务为代表N。实际模型参数、hooks、前后向耗时、显存、epoch效率均未测，不能据MAC推断真实速度。

## E. 实际执行与结果

本地 Python3.13.9/Anaconda，Linux WSL2；已安装 torch2.13.0、NumPy2.2.6、PyG2.3.1。纯解析不依赖这些tensor包。RNG零影响测试仅在独立子进程预先导入NumPy/Torch并比较状态，未构造V3模型。测试进程CUDA_VISIBLE_DEVICES为空、OMP/MKL线程数1；未安装或更改任何依赖。

| 命令 | 实际结果 | 耗时 |
|---|---|---:|
| `python -B docs/loop_linearno_latent_adapter_audit/laa1/run-checks.py targeted` | 30 passed / 0 failed / 0 skipped | unittest 3.193s；进程3.293s |
| `python -B docs/loop_linearno_latent_adapter_audit/laa1/run-checks.py legacy` | 49 passed / 0 failed / 0 skipped | unittest 2.025s；进程4.151s |
| `python -B docs/loop_linearno_latent_adapter_audit/laa1/run-checks.py matrix` | 2304条配置；3合法custom、9预期拒绝 | 6.734s |
| `PYTHONDONTWRITEBYTECODE=1 python -B docs/loop_linearno_latent_adapter_audit/laa1/verify-preview.py` | 2304条独立参数/MAC对照，最大整数误差0；192组backbone配对，24组task/seed数据序列 | 5.476s |
| `python -B docs/loop_linearno_latent_adapter_audit/laa1/static-audit.py` | 17 Python内存compile；130 shell `bash -n`；`git diff --check`通过 | 逐项见 `static-checks.json` |

runner的真实子命令、环境、路径、exit code及秒数见 `*-verified.json`；原日志为 `*-verified.log`。文件使用create-only，重跑请给runner第二个label参数并对矩阵另选新输出目录。完整矩阵在 `matrix/configuration-matrix.jsonl`（约52.5MB），摘要在 `matrix/matrix-summary.json`，所有argv明确标记 `SCHEMA_PREVIEW_NOT_RUNNABLE_LAA1`。这不是LAA8生产parser dry-run。

30个新测试含64格表核对、96个task/mode/ablation成本组合、严格类型/未知字段/来源/派生字段、metadata roundtrip/hash稳定、恢复不查当前profile、独立进程import guard、Python/NumPy/Torch CPU RNG不变及152份旧配置字节级回放。49个既有测试来自 pure profiles/schema、history schema、loop V1 config/schema、V2 LF1 config/schema；旧源码和测试未修改。

### 初次失败与修正记录

没有覆盖失败日志或放宽旧golden/容差：

1. 最初 `red-tests.log` 中是新配置草稿的SyntaxError，后来修正。它不是模型数学负例，不据此宣称完整oracle验收；首轮17项通过与5项隔离通过分别保留。
2. `contract-gap-red.log` 记录6项中2项失败：缺 `checkpoint_version`、缺显式profile断言来源。只补新V3合同，`contract-gap-green.log` 为6/6通过。
3. `cost-scope-red.log` 记录2项失败：显式null topology被当省略、修改基础profile仍标表格成本适用。修正新V3解析/成本标记后green通过，并纳入最终30项。
4. 新证据runner首轮根目录层级写错，命令cwd误为 `/home/hwz`，导致模块未找到；`targeted-final/legacy-final/matrix-final.{json,log}` 是这次**失败**，虽旧命名含final也不代表最终结果。修正runner root后使用新的 `*-verified.*` 记录成功，不覆盖原始失败文件。错误发生在导入之前，未启动任务或读数据。

## F. 冻结与交付自审

`end-freeze.json` 同时比较 LAA0 的3377个文件及 LAA1 的4083个文件：LAA0起点全部不变；LAA1起点只允许独立LAA状态文档追加，其他文件不变、无缺失。tracked/untracked/ignored一并核验，用户checkpoint检查脚本的原diff和staged状态保持。旧LL/LF证据、测试/golden、纯模型、V1/V2、Transolver及八任务链路不变。

| 优先复核点 | 自审结论与证据 |
|---|---|
| V3选择与版本隔离 | 显式architecture才解析；152旧配置exact；49旧合同回归通过；冻结清单 |
| H/Dz表与网格字段 | 64格完全一致；2304条effective/constructor/loop宽度与原grid一致 |
| 所有权/公平性/关闭模块 | 完整core共享合同；off实例参数0；192配对组及24数据seed组一致；实际tensor配对留后续阶段 |
| metadata恢复边界 | 不读取当前profile表、不load tensor，strict/冲突/create-only及恢复字段测试通过；真实backend未冒充已测 |
| 成本独立性和宣称范围 | 2304条独立shape oracle误差0；router/scalar另列；不同基础profile不冒用匹配结论；没有速度宣称 |

本阶段没有剩余需要用户裁决的数学规格冲突。旧全仓689项回归未在LAA1重跑；LAA0的644 passed/9历史failed/36 skipped仍保留，不能把本阶段49项定向回归写成全仓全绿。

NOT RUN：V3生产tensor模块/forward/backward、实际参数与hooks、生产parser/factory/checkpoint/resume/eval、GPU/AMP、真实数据/训练/完整epoch、三seed收敛、SOTA、实际延迟/显存/epoch、远端栈。paired seeds0/1/2仅生成配置；没有运行实验。未commit/push。

本 LAA1 阶段结束，未执行下一阶段。
