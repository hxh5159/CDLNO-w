# Looped LinearNO：LL1 配置与 metadata 合同

日期：2026-09-19。**阶段状态：PASS。仅配置/schema/测试；没有 loop torch 模型，没有任务接线。**

## A. 范围与实际基线

仓库 `/home/hwz/CDLNO`，origin `https://github.com/hxh5159/CDLNO-w.git`，分支 `main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，HEAD tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。本轮起点 tracked staged/unstaged diff 均为空；已有 LL0 文档、证据及用户两个设计文件未提交。逐文件起点与末次分类/size/SHA-256见 [start-manifest](loop_linearno_audit/ll1/start-manifest.json) 和 [end-freeze](loop_linearno_audit/ll1/end-freeze.json)。不能将这些已有 untracked/ignored 文件算成本轮新增。

本阶段依据：用户当前 LL1 十项要求、[冻结规格](../PLAN_Looped_LinearNO/Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md)、已批准 [LL0](LOOP_LINEARNO_REFERENCE_AUDIT.md)，以及实际 `cdlno/linearno/{profiles,_profile_data}.py` 和三套原模型构造签名。遵守根 AGENTS 的冻结要求，阶段范围以用户当前授权为准。未更新旧纯 LinearNO/history/CDLNO 报告、AGENTS 或生产文件。

## B. 新文件与职责

| 文件 | 实际作用 |
|---|---|
| `linearno_loop/__init__.py` | 独立顶层包，仅导出纯配置 resolver/validator |
| `linearno_loop/contracts.py` | family/version、拓扑与 CLI 名称、参数共享合同；严格 JSON/type/hash 与逐字段诊断 |
| `linearno_loop/config.py` | 从现有不可变 profile 解析 base M；派生拓扑/rank/AttnRes；计划构造器参数；来源标记、paired seeds、目录标识 |
| `linearno_loop/state.py` | 纯标准库检查数值/RNG/optimizer 编码；不恢复 tensor/backend 状态 |
| `linearno_loop/schema.py` | 完整 metadata、严格恢复断言、实际构造器签名检查接口；只读 JSON / create-only JSON 写入 |
| `linearno_loop/matrix.py` | 配置与计划 argv 预览；不建目录、不执行命令 |
| `tests/loop_linearno/` | 24 项测试；合成 metadata producer、负面测试、隔离进程、旧文件冻结与矩阵 fixture |
| `docs/loop_linearno_audit/ll1/` | 起点、机器合同/矩阵/metadata、测试日志、复算工具和最终证据 |
| 本文、`LOOP_LINEARNO_IMPLEMENTATION_STATUS.md` | LL1 合同与交付状态；保留 LL0 历史 |

没有 `cdlno/linearno_loop/` 生产模型包或 `tran_evaluate/linearno_loop/` launcher。以下 class_path 是 **后续真实实现应满足的命名合同**，当前不能 import，也没有占位模型：

- `cdlno.linearno_loop.standard.LoopedStandardModel`
- `cdlno.linearno_loop.airfrans.LoopedAirfRANSModel`
- `cdlno.linearno_loop.shapenet.LoopedShapeNetModel`

## C. 配置、公式和来源

唯一 family 为 `linearno_loop`，extension 为 `loop_linearno_v1`；config/schema version 均为整数 `1`，公式版本 `point-loop-residual-v1`。

```python
from linearno_loop import resolve_config, validate_config

config = resolve_config(
    'darcy', 'paper_table8_on_release_model',
    options={
        'topology_preset': 'p1_c3_r2_s1',
        'residual_mode': 'rb_attnres',
    },
    profile_overrides={'runtime.seed': 0},
)
validate_config(config)  # JSON-compatible dict; does not construct a model
```

该 API 只接受已定型的 Python/JSON 值。整数字段拒绝 bool、float、字符串数字、null；整个树拒绝 NaN/Inf、未知字段。修改字段后重算 hash 也不能绕过从 `request` 重建完整配置的校验。输入和返回均隔离 deepcopy。

### 拓扑

| preset | P | C | R | S | unique=P+C+S | executed=P+C×R+S |
|---|---:|---:|---:|---:|---:|---:|
| `p1_c3_r2_s1` | 1 | 3 | 2 | 1 | 5 | 8 |
| `p2_c2_r2_s2` | 2 | 2 | 2 | 2 | 6 | 8 |
| `custom` | ≥0 | ≥1 | ≥1 | ≥1 | 派生 | 派生 |

字段名依次为 `prefix_blocks/recurrent_core_blocks/loop_repeats/suffix_blocks`。preset 不得附带任何显式 P/C/R/S；custom 必须四项齐全。未指定拓扑或 residual mode 会报错；用户没有选定它们的默认值，本阶段没有代选。`unique_depth/executed_depth/head_dim` 只可派生，不能作为训练 options 输入。

`loop_spec.sharing` 固定：prefix/core/suffix 物理模块独立；core 每个物理模块只注册一次，跨 round 复用 operator/MLP/LN 全部参数；每 visit 重算 Q/K/V/context；只由最后 suffix 执行一次输出头。forward-local raw 点历史不 detach、不跨 forward/物理时间/member。此处只是合同，实际共享/张量行为在后续模型阶段验证。

### Rank 与原 profile

| task | variant | 主 profile hidden/heads | base M | 默认 ×2 | ×1 对照 |
|---|---|---:|---:|---:|---:|
| airfoil | conv_temp | 128/8 | 64 | 128 | 64 |
| darcy | conv_temp | 128/8 | 64 | 128 | 64 |
| elasticity | temp | 128/8 | 64 | 128 | 64 |
| pipe | conv_temp | 128/8 | 64 | 128 | 64 |
| ns | plain | 256/8 | 32 | 64 | 32 |
| plasticity | conv | 128/8 | 64 | 128 | 64 |
| car | shapenet | 256/8 | 32 | 64 | 32 |
| airfrans | airfrans | 256/8 | 32 | 64 | 32 |

表仅展示主 profile；实现每次调用当前 profile resolver，未硬编码这张表。全部八任务×三 profile 与 LL0 快照逐值一致。`transolver_matched` 的实际宽度等以 resolver 为准。

- 默认 `rank_policy=profile_multiplier, rank_multiplier=2`；显式允许倍率 1 或 2。
- 显式 `options={'linearno_rank':47, ...}` 表示 actual M=47，记录 `rank_policy=explicit_actual, rank_multiplier=null`。不伪造整数倍率。实际 M 与显式 multiplier 同时给出一律拒绝。
- ShapeNet 必须 `actual M % head_dim == 0`，例如 d_h=32 时 M96 合法、M48 非法。Standard 的原 key_ratio 是绝对 M，Air 的原 slice_num 是绝对 M，Car 的原 key_ratio 为 `M/d_h`；保存在 `operator_contract.rank_mapping`。
- `hidden % heads == 0`；time input 要求偶数 hidden。工业输入/输出通道、Car unified replacement 与 fun_dim、结构化 H/W 等沿用实际旧构造约束。
- 所有 block 与 round 使用同一 resolved M。`profile_spec` 完整保存原 `family=linearno` 的协议来源与基础 layers/rank；它不是待构造的 loop 结构。真正模型字段只取 `loop_spec` 和 `model_spec`。拒绝 profile override 中再指定 layers/rank/qk_dim。
- `profile_overrides` 只用于现有 model/training/runtime 的显式值；data/objective/evaluation 保持所选 profile 合同，拒绝另行覆盖。没有更改旧 profile resolver/CATALOG。

字段来源分为 `cli_explicit/profile/family_default/integration_contract`（原 profile 来源原样保留），拓扑增加 `preset`，派生项为 `derived`，固定项为 `frozen_contract`，显式 actual M 下 multiplier 为 `not_applicable`。例如默认 ×2 来源 family_default，显式倍率来源 cli_explicit，base M 来源 profile，hidden/head 来源对应原 profile 字段。字段来源本身也参与严格重建和 hash。

### 三 residual mode 与点域 AttnRes

下列 d 指隐藏宽度 hidden，不是 operator heads 数。

| mode | 核心合同 | receiver 数 | router 参数数 | source visits |
|---|---|---:|---:|---:|
| sr_1_over_r | 每条 operator/MLP branch ×1/R，identity 不缩放 | 0 | 0 | 0 |
| rb_attnres | 每子层前 AR；anchor+已完成 raw round sum+当前 raw partial；不做 h+u；出口 AR；无1/R | 2CR+1 | 2d(2CR+1) | (2C)R(R+3)/2+1 |
| lb_attnres_1_over_r | 轮内 SR1/R；Δ=Y−H；轮间/出口 AR(anchor,Δ历史)，不再缩放 | R | 2dR | R(R+3)/2 |

P1 的 RB/LB visits=31/5；P2=21/5。prefix/suffix 的原残差始终不缩放；head 只执行一次。`loop_spec.residual_contract`、`attnres`、`sharing` 共同固化这些含义。

`point_domain_attnres=true` 表示该 family 的 AR 域合同，SR 的 `attnres.enabled=false` 且不产生 receiver。AR 使用对齐 `[B,N,hidden]` raw value；key RMSNorm 最后一维 keepdim、eps1e-6、scale1、无bias；receiver query0，单来源维 softmax，无 value projection、sqrt(hidden)/来源数/额外1/R缩放、无 depth multihead。RB 首 receiver 单来源恒等且 query/norm 无有效梯度是预期。router 按逻辑位置独立，所以有位置专属参数；`feature_timestep_encoding=false` 只禁止额外 loop index 特征，保留 NS/Plasticity 原物理 T 语义。

### 计划 CLI（未接入 parser，不可用于训练）

| CLI | typed 字段 |
|---|---|
| `--linearno-loop 1` | `linearno_loop=True` |
| `--linearno-loop-topology PRESET` | topology_preset |
| `--linearno-loop-prefix-blocks P` | prefix_blocks |
| `--linearno-loop-core-blocks C` | recurrent_core_blocks |
| `--linearno-loop-repeats R` | loop_repeats |
| `--linearno-loop-suffix-blocks S` | suffix_blocks |
| `--linearno-loop-residual-mode MODE` | residual_mode |
| `--linearno-loop-rank-multiplier 1\|2` | rank_multiplier |
| 既有 `--linearno-rank M` | actual linearno_rank |

未来 parser 必须追踪 argv 显式来源，不能将旧 slice_num/n_layers 等默认值送入 loop options；0/1 wire 值只由 parser 转 bool，schema 不做隐式类型转换。`route_intent` 是纯 guard，未安装到任何入口：省略全部 loop 字段或只有旧 actual-M 字段时返回 None，让旧路径处理。显式 loop=false 单独出现也返回 None；与其它 loop 字段混用拒绝。

loop 与任意旧 A/K/history dropout 字段存在即冲突，**包括 false、0、None**；给其它 family 开 loop 也拒绝。必须在模型、读权重、建运行目录前调用此检查。

## D. Metadata、恢复与公平比较

完整 machine contract：[contract-catalog.json](loop_linearno_audit/ll1/contract-catalog.json)。这是可读取的合同目录；跨字段关系以 Python validator 为准，没有把它误标为通用 JSON Schema。

| section | 保存内容与校验 |
|---|---|
| loop_spec | 完整拓扑/派生量、mode、base/resolved/rank policy、共享、AR 初始化/receiver/公式与 schema/config version |
| model_spec | class_path + constructor_kwargs；沿用任务 stem/时间/位置字段，将 n_layers 替换成 P/C/R/S 和 mode，actual linearno_rank；不塞 profile、family 或派生量 |
| profile_spec | 原 profile 的数据/训练/评价/默认及字段来源；独立于实际 loop 结构 |
| data_spec | profile protocol、实际 split、sampling、命名数据 checksum、real/synthetic scope、运行数据描述 |
| objective_spec / evaluation_spec | 与所选当前 profile 严格一致，分别保存，不因 loop 改写；Air/Car 保持已确认官方 MSE 合同 |
| provenance_spec | 当前 target/base commit/dirty、固定 Transolver/LinearNO/AttnRes/K3 SHA、论文v3版本/hash、残差缩放来源版本、实际源清单hash/规范化patch hash、代码/schema版本、argv、环境 |
| normalizer_spec | none 或命名 input/output/coef_norm 等记录；algorithm、fit_split、数据checksum、命名数值dtype/shape/base64/hash；不允许静默重拟合 |
| resume_state | role/selection、epoch/global_step、optimizer/scheduler、Python/NumPy/Torch CPU/CUDA RNG、显式 DataLoader generator device/state、sampler state |
| ensemble_manifest | 有序 member id→相对 state_dict 路径/hash；拒绝重复id、乱序、路径逃逸、whole object |

top-level 还记录完整 resolved_config、config_hash、metadata_hash 和固定 load_policy。所有 section 必需；不使用旧家族 fallback。检查数值 bytes 长度/shape/dtype/hash，支持 torch bfloat16 与 NumPy endian 数值。Python RNG 用隔离对象验证，不推进 global RNG；NumPy/Torch state 仅纯字节/结构检查。**LL1 不宣称已验证真实 backend 能恢复该状态或 optimizer 与参数的对应关系；后续 loader 必须验证并恢复。**

`read_metadata` 只读取 JSON，拒绝重复键和非有限 JSON。`write_metadata` 使用 create-only，不建目录、不覆盖旧 sidecar。

eval/resume 计划流程：

1. 从明确给定运行的 architecture/checkpoint JSON 读 metadata，验证全部结构、hash、协议及 family；不能按当前 CLI 默认猜。
2. `restore_config(metadata, explicit=..., runtime=..., strict=True)` 取回保存结构；显式拓扑/mode/rank/hidden/head_dim/variant/schema 等只作一致性断言，多个不一致逐字段报告。显式 actual rank 与 multiplier 冲突仍拒绝。
3. 仅 device/experiment_dir 可作为 runtime 变化另行返回；不得覆盖保存的结构/训练协议。测试同时证明原 metadata 不变。
4. 后续引入真实 class 后用 `validate_constructor` 对真实 callable 完整显式签名核验全部 kwargs，不执行构造器；禁止 *args/**kwargs 遮蔽参数。
5. 上述通过后才构造模型/读权重；按 manifest 校验权重并 strict=True。禁止 baseline 作为 loop resume、错结构随机补键或忽略键。实际 torch checkpoint/optimizer/RNG 恢复不在 LL1。

[synthetic-metadata.json](loop_linearno_audit/ll1/synthetic-metadata.json) 是 schema 测试夹具，**无模型或权重**：小型数值 normalizer、隔离 RNG 和空 optimizer-state 编码。明确 synthetic scope；source hash 覆盖新纯包；规范化 patch 使用排序相对路径、before=null、after=LF文本的 `path-before-after-lf-v1` 编码。不是已训练归档。后续保存真实归档必须填真实数据、训练状态、完整实际实现源和 patch 证据。

公平比较预声明本地 seeds=0,1,2，不冒称作者 seed。同任务/profile/topology/actual M/seed 先产生唯一公共主干再复制到三个 mode；router 用独立记录 seed，DataLoader train/test 各有独立 generator seed。seed 派生纯 hash，不调用 RNG；构造顺序不得改变公共主干权重。不同 topology 不宣称参数匹配，两个正式 preset 不宣称 FLOPs 或延迟减半。

正式 [configuration-matrix.json](loop_linearno_audit/ll1/configuration-matrix.json)：paper profile 下 8任务×2presets×3modes×3paired seeds=144；另144组 rank×1 控制，24组 custom P0-C2-R3-S1 的配置检查。记录未来 construction/forward/backward/strict roundtrip/native resume-eval 为 NOT RUN。还列明纯8层 base M 和2M是后续比较对照。完整实验逐seed及mean/std报告，使用final checkpoint，禁止test挑权重。

新目录标识合同：

```text
{task}__linearno_loop__{profile}__P{P}-C{C}-R{R}-S{S}__{mode}__M{M}__seed{seed}__cfg{hash12}
```

后续输出 reservation 再加唯一 run id；LL1 只返回标识，不创建路径。矩阵中的 argv 均标记 `PLANNED_NOT_RUNNABLE_LL1`，不能作为远端训练命令；当前远端根仍是用户指定的 `.../transolver/LinearNO-monitor`，未修改 path.sh。

## E. 实际验证、兼容证据与未运行项

环境：本机 Python3.13.9；CPU测试，CUDA_VISIBLE_DEVICES为空。未安装依赖。确切包版本见 [environment](loop_linearno_audit/ll1/environment.json)。

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -B docs/loop_linearno_audit/ll1/build_contract_artifacts.py

PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -m unittest discover -s tests/loop_linearno -v

PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -m unittest linearno.test_profiles linearno.test_schema linearno.test_history_schema \
linearno.test_legacy.LegacyRegression.test_real_parser_AST_defaults_unchanged_and_new_flags_isolated \
linearno.test_legacy.LegacyRegression.test_eight_real_models_same_weights_and_checkpoints_fresh_workdirs \
monitor.test_monitor monitor.test_export_linearno_results -v
```

- 新24项：全部通过，3.490s，0 failure/error/skip。[日志](loop_linearno_audit/ll1/schema-tests.log)。包括24 profile快照、strict type、配置/metadata重哈希篡改、rank、混flag、恢复多字段诊断、normalizer/RNG/ensemble/JSON负例、完整机器矩阵复算。
- 旧33项：全部通过，23.008s，0 failure/error/skip。[日志](loop_linearno_audit/ll1/legacy-regression.log)。原 parser 默认值与隔离、纯LinearNO/history profile/schema、八任务Transolver参数/key/同权重输出和checkpoint、monitor与结果导出通过。Transolver输出沿用 atol1e-6/rtol1e-5，不放宽旧容差；配置/hash/RNG/文件检查为精确相等。
- 独立新进程禁止 torch/NumPy/任务/模型 import 后，仍能读metadata/restore/生成288矩阵，Python RNG精确不变。已有 tensor 环境中完整配置/夹具/校验调用前后 Python/NumPy/Torch CPU RNG编码精确相等。
- 全量冻结比对覆盖 LL1 起点1975文件（含 untracked/ignored），只允许更新本研究状态文档。旧生产、数据、配置、parser、factory、checkpoint、测试、monitor、launcher 逐字节不变；见末次证据。旧纯/history schema 和目录函数未被调用链改写。
- 未重跑 LL0 的全部153方法：其完整数值parity仍以 LL0证据为准；LL0 已解释的2个历史文档冻结失败方法/4条断言保持原状，没有调整golden让它们变绿。本轮33项通过不代表全仓全绿。
- **NOT RUN**：loop torch模型/公式oracle/真实共享/梯度/参数量/torch strict load/真实resume（LL2及以后）；八任务loop生产接线、launcher、远端；真实数据、GPU、短/长训练、性能和精度。它们不属于LL1，没有借合成metadata宣称完成。

## F. 交付前自审

| 优先复核点 | 已审查证据 |
|---|---|
| 是否改了旧默认/数学/输出路径 | 全量起点manifest复算、33旧回归；新增包无生产import |
| 是否有第二套深度/rank真值 | preset/custom互斥；derived输入拒绝；original profile仅来源；model_spec只使用PCRS与actual M；错误重哈希负测 |
| RB/LB是否误用旧latent-history或缩放 | 三mode合同、receiver/参数/source-visit解析值；新点域合同与旧A/K字段互斥；尚未实现张量模块 |
| metadata是否会静默换结构/宽松加载 | metadata-first、逐字段一致性、strict=True、独立构造签名guard、legacy拒绝；无torch.load |
| 是否混淆计划与执行/公平初始化 | 312行均配置预览；所有模型验收NOT RUN；配对seed与独立generator合同；不宣称实际模型已满足公平初始化 |

未发现需要用户裁定的新增架构冲突。下一阶段实现必须遵守此合同；若要变更合同，先明确更新版本和迁移规则，不得让已有metadata静默变义。

**本 LL1 阶段结束，未执行下一阶段。**
