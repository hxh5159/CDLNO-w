# MSAR-LNO M1：独立配置、family 与 checkpoint 元数据

2026-09-16。本轮仅实施用户点名的 **M1**。配置与元数据基础已完成；没有实现 MSAR 的 Down/Up/AttnRes/core、coverage 数学、任务 wrapper 或生产训练接入。以下配置预览不是训练命令，JSON 往返不是新模型权重验收。

## A. 依据、当前仓库与证据边界

工作区：`/home/hwz/CDLNO`，分支 `main`，HEAD `9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`。修改前已有 seed、周期可视化、离线报告等未提交改动，全部保留；[原 status](msar_lno_audit/m1/status-before.txt) 与 [原 diff](msar_lno_audit/m1/preexisting.diff) 已单独留存。

实际阅读材料包括根 `AGENTS.md`、README、`PLAN_MSAR_LNO/MRSA_LNO_arch.md` 全文、`PLAN_MSAR_LNO/MSAR_LNO_Codex_Staged_Prompts.md` 全文、现有 CDLNO/KCDNO 状态与 front 消融记录、KCDNO K0/K1 审计/配置文档和夹具索引；按实际源码核查三个子项目入口、factory、解析 helper、共享配置/metadata、输出工具及对应测试。新配置只采用最终四级结构、两来源融合与 coverage-floor 目标，没有采用架构讨论早期的 HistoryCross 等候选。

**缺失材料：** 用户已确认 M0 通过，但本工作区没有 `docs/MSAR_LNO_REFERENCE_AUDIT.md`、原 MSAR STATUS 或 M0 夹具索引；仓库、外部 artifacts 和临时目录中未找到。因此本报告不能引用一份实际不存在的 M0 映射，也不能声称重跑了该索引。本轮补做 M1 所需源码核查，并复用可用的 K0 夹具及本轮修改前真实捕获的旧模型基准。[inventory.json](msar_lno_audit/m1/inventory.json) 索引了 555 个文本文件和 Python AST，未发现语法问题；这只是文本/符号索引，不冒充完整 M0 语义审计。

修改前保存 614 个既有文件的 SHA256 与外部源码快照：

```text
/home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/source
```

[before.json](msar_lno_audit/m1/before.json) 和 [fixture-index.json](msar_lno_audit/m1/fixture-index.json) 提供可比较依据。旧 K0 的 41 个模型夹具在修改前先回放成功；另外在新增生产模块前，为现有 KCDNO all/off 与 trainable matched LRSA 的 point/conv core 保存 6 份真实配置、同一份 state_dict、键/形状、固定输入及输出。大二进制只放外部 artifacts，不放进仓库、不自动提交。这些即时构造基准不是历史训练 checkpoint，不能证明任意历史 checkpoint 都兼容。

## B. 真实接入映射与 diff 范围

| 当前职责 | 已有真实位置 | 本轮处理 |
|---|---|---|
| 共享包发现 | `pyproject.toml` 的 `include=["cdlno*"]` | 自动包含新增 `cdlno.msar_lno`；不改依赖或安装 |
| PDE 模型 factory | `PDE-Solving-StandardBenchmark/model_dict.py:get_model` | 原 Transolver/CDLNO/KCDNO/matched 映射不变，尚未注册 MSAR 构造器 |
| PDE CLI/配置/运行 | `PDE-Solving-StandardBenchmark/cdlno_entry.py:parse_args`，`cdlno/kcdno/entry.py:resolve_args,StandardRun` | 不改、不接新参数 |
| Car CLI/运行 | `Car-Design-ShapeNetCar/models/cdlno_run.py:parse_args`，`cdlno/kcdno/industrial_entry.py:resolve_car,CarRun` | 不改，整模型保存与稳定类路径不变 |
| AirfRANS CLI/运行 | `Airfoil-Design-AirfRANS/cdlno_entry.py:parse_args`，`cdlno/kcdno/air_entry.py:resolve_air,AirRun` | 不改，整对象/列表协议不变 |
| 旧 family 与 sidecar | `cdlno/kcdno/families.py`、`metadata.py`；`cdlno/checkpoint.py` | 不增加猜测旧 family 的规则 |
| 显式参数识别 | `cdlno/kcdno/options.py:explicit_arguments` | 新模块直接复用此纯解析函数，原文件不改 |
| 旧输出/记录 | `cdlno/experiment.py` 及现有 launchers | 原输出目录/覆盖防护不改 |

新增文件/符号：

| 文件 | 主要符号与用途 |
|---|---|
| `cdlno/msar_lno/config.py` | `MSARArchitectureConfig`、`MSARTrainingConfig`、`MSARRuntimeConfig`、`input_layout` |
| `cdlno/msar_lno/profiles.py` | `profile_values`、`resolve_profile`、`ResolvedMSARConfig`；八任务共用 Light/Full |
| `cdlno/msar_lno/registry.py` | `family_for_model_key`、`checkpoint_family`；仅 family 分流协议 |
| `cdlno/msar_lno/options.py` | `parser_for_family`、`parse_training_options`、`resolve_training`；新 family 独立解析 |
| `cdlno/msar_lno/metadata.py` | `MSARMetadata`、`resolve_evaluation`、exclusive JSON 写入/目录预留 |
| `cdlno/msar_lno/__init__.py` | 导出独立配置，不提供占位 Model/core |
| `cdlno/msar_lno/__main__.py` | 无数据、无模型、无实验目录创建的配置预览 |
| `tests/test_msar_config.py` | 13 个针对性测试方法，实际 parser 声明/元数据/配置检查 |
| `docs/msar_lno_audit/m1/replay_current.py` | 6 份既有真实 core 的独立同权重留存/回放 |
| 本报告、独立 STATUS、audit JSON/日志 | 可复查的命令、结果、冻结及夹具索引 |

实际新注册协议是 **`msar_lno → cdlno.msar_lno` 的配置/family 分流**，不是八任务 factory 中已经能构造模型。其他 model key 返回 `None` 交还旧选择器；checkpoint 缺少外层 `family` 也返回 `None`，即使其内部字典碰巧含新名字，也不推测为 MSAR。生产 `--model msar_lno` 要等后续明确授权接入，当前不可宣传为可训练。

既有生产文件、入口、依赖、原测试/启动器全部保持字节一致。仅旧 `docs/CDLNO_IMPLEMENTATION_STATUS.md` 与 `memory/current-state.md` 增量追加本阶段链接，历史原文保留。[freeze.json](msar_lno_audit/m1/freeze.json) 列出全部比较；[m1-source.patch](msar_lno_audit/m1/m1-source.patch) 包含新增源码/测试与文档的实际 patch，避免 `git diff` 漏掉未跟踪文件。

## C. Resolved 配置、公式合同与元数据

### 精确 profile

CLI 名称为小写 `light` / `full`，默认 `light`；论文显示名对应 Light / Full。

| profile | num_latents | d | heads | encoder_depths | decoder_depths | 每个 latent FFN hidden |
|---|---|---:|---|---|---|---:|
| Light（默认） | [512,256,128,64] | 96 | [4,4,8,8] | [3,1,1,1] | [3,1,1,1] | 192 |
| Full | [1024,512,256,128] | 192 | [4,4,8,8] | [3,1,1,1] | [3,1,1,1] | 384 |

两个 profile 在八任务上相同，不从旧 L/F/P/M 或任务默认值推导。结构字段固定记录 `family=msar_lno`、`architecture_version=msar-lno-core-v1`、ratio=2、GELU、pointwise MLP、RMSNorm/per-head QK RMSNorm、eps=1e-6、任务输出 LayerNorm、query-aligned two-source fusion 和固定倍率 2。Norm 标签沿用当前公共模块与任务 head 约定；本阶段只是配置描述，不执行模块。没有卷积/CDPA/kernel-history/front-mode 开关。

每个列表长度必须为 4；M/d/heads 为正整数，拒绝 bool/浮点冒充整数；d 必须被所有 heads 整除。各级 M 递减；两个 depths 必须非负且严格等于本版 `[3,1,1,1]`。数组入对象后变为不可变 tuple，JSON 导出为独立 list。hidden 由最终 d 唯一推导 2d，不再存一个可能冲突的宽度来源。改变 d/M/heads 的显式覆盖记录在 `profile_overrides`，profile 标签不能代替 resolved 配置。

N 不限制 M；Full 在 `N=972` 时预览明确输出 `first_down_expands=true` 和 token path `[972,1024,512,256,128]`，并注明合法 latent 扩张，不截断任何 M。`input_layout` 为后续日志提供该记录；**生产训练 logger 尚未接线，不能声称已经在真实训练日志输出。**

| 已确认计算定义 | 本阶段配置对应 | 执行边界 |
|---|---|---|
| 四级 learned-query Down → encoder；三级对齐 Up/fusion → decoder；最终点域 Up/head | num_latents、heads、两个 depths、point_module | 未实现 Down/Up/core |
| 每个 latent block 两个独立 d→2d→d GELU FFN + SA | ratio=2、activation、norm/QK norm | 没有参数对象/SA 计算，也不调用旧消融 |
| `Z=2*(alphaE*E+alphaU*U)` | fusion 标签、fusion_scale=2 | 未实现 scorer/softmax，不宣称初始化/梯度验收 |
| `LPDE + coverage_weight * mean(Lcover)`，floor 阈值 `kappa*mu` | 独立 `MSARTrainingConfig` | 未改 loss 或计算 coverage |

### 训练目标与运行记录

研究默认：`coverage_mode=floor`、`coverage_weight=0.01`、`coverage_kappa=0.2`、`coverage_eps=1e-6`、`diagnostics=false`。权重需有限且 >=0，kappa∈[0,1]，eps有限且 >0，diagnostics 必须为布尔值。

唯一开关是 coverage_mode；`coverage_enabled` 是只读派生值：floor 且 weight>0。off 或 weight=0 都记录有效模式 off。`no_regularizer` 示例是 `--coverage-mode off --coverage-weight 0`，不是第三个结构 profile，也没有另加冲突布尔开关。kappa=0 合法，不另造一个模式；将来计算的 penalty 可为零。此处仅决定配置语义，不声称已执行 off 的无 attention-map 路径。

`MSARRuntimeConfig` 独立记录 device/dtype/batch_size/SDPA backend/AMP/TF32/compile。这些是运行记录字段，不是八任务新训练默认值。纯 eval 比较时只报告差异，不按权重形状冲突拒绝；不承诺不同后端输出逐位相同。

### 解析优先级与旧命令隔离

训练：**显式 CLI > 所选 profile > 新 family 默认**。复用 argparse 深拷贝后清空默认值的识别方法，支持别名、`--flag=value` 与同一选项最后覆盖。只在 model_key 精确为 `msar_lno` 时创建解析副本；旧 family 不解析新参数，不修改旧 parser/choices/defaults、不生成新 kwargs。

新字段：`--d`、`--num-latents`、`--heads`、`--encoder-depths`、`--decoder-depths`（列表选项固定 4 个整数），以及 coverage/diagnostics。保留既有显式 `n_hidden→d` 别名；两种名称同时给不同值清楚拒绝。旧 parser 未显式传入的 `n_layers/n_hidden/slice_num` 等完全不影响 Light/Full。显式旧 `n_layers/slice_num/n_heads`、front/rear/CDPA、kernel/history、旧 mlp/dropout 等在新 family 报不适用，不折算为多尺度字段；旧 float ratio=2.0 只作为同一个固定 ratio=2 接受。

测试提取 10 个真实入口与对应公共 helper 的参数声明 AST，**不导入 exp/main、不执行其数据/模型分支**。AirfRANS eval 的 model choices 来自 helper；副本增加 msar_lno，原 choices 保持原样。这是未来接线协议的实际 parser 验证，不是生产入口已支持新 family。

### checkpoint 与输出基础

新 sidecar 完整记录 outer family、schema_version、task、profile、checkpoint_format，以及独立 architecture/training/runtime。读取要求全部字段齐全且没有未知字段；不为不完整保存记录补默认。结构包括行为字段，不只检查 tensor shape。错误 family/task/d/M/heads/depths/激活/norm 等拒绝。

eval 顺序为：**先读保存的 resolved 配置 → 核对用户显式结构覆盖 → 返回保存结构及训练/运行差异记录**。coverage off/floor 不拒绝同一结构；保存的训练设置仍保留，另返回 requested_training，文件字节不变。沿用当前 KCDNO 的约定，eval 的 profile 仅是来源标签，不能展开新默认覆盖 checkpoint；显式传 d/M 等字段才进入结构比较。

保存格式字段仅描述原协议：六标准任务 `state_dict`，Car `whole_model`，AirfRANS 顶层 `model_list` / 成员 `whole_model`。本阶段不调用 torch.load/save 保存 MSAR 权重、不增加 pickle 权限、不修改旧类路径、不实现 resume/迁移/模型重建；未来仍需结合真实 wrapper 与权重严格校验。

新目录协议：

```text
output/<dataset>/msar_lno/<light|full>/<coverage_floor|coverage_off>/<UTC时间戳_UUID或显式save_name>/
```

`new_run_path` 只提议路径；`reserve_run_directory` 用 `exist_ok=False` 预留，sidecar 用 `open('x')` 独占写入。显式 save_name 原样作为叶目录，沿用文件名 stem 校验；同名已有目录拒绝覆盖。effective off 包括 floor+weight0。旧实验目录与保存行为没有修改。

## D. 实际命令、环境与结果

本机 Python3.13.9、torch2.13.0+cu130、PyG2.3.1、RTX5090 Laptop GPU，见 [environment.json](msar_lno_audit/m1/environment.json)。没有安装或变更环境；pyproject 的目标 Python 范围仍为 >=3.10,<3.12，本机使用源码 PYTHONPATH，不声称完成 editable 安装或目标远端兼容验收。

```bash
cd /home/hwz/CDLNO
PYTHONPATH=tests:. python -B -m unittest test_msar_config test_kcdno_config test_core_config -v

python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result docs/msar_lno_audit/m1/old-after.json

python -B docs/msar_lno_audit/m1/replay_current.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/current-fixtures \
  --result docs/msar_lno_audit/m1/current-after.json
```

| 实际检查 | 结果与边界 |
|---|---|
| 新配置 13 + 旧 KCDNO 配置 13 + 旧 core 配置 6 | **32/32 通过，16.293s**，0 failure/error/skip；[日志](msar_lno_audit/m1/config-tests-final.txt) |
| Light/Full、8 task 配置/JSON、10 真实 parser 声明样式 | 合法/非法、显式优先、原 parser 不变通过；不是八任务模型验收 |
| metadata、结构冲突、coverage/runtime eval 差异、sidecar 不改写 | 通过；三类原保存格式只核对元数据，未伪造新模型权重 |
| 3 子项目 cwd 新进程导入、包发现、Python3.10 AST | 通过；纯配置不加载 torch；没有执行3.10运行时或安装 |
| 旧 K0 模型夹具 | 修改前/后均 **41/41**，输出最大绝对差0，原梯度/checkpoint比较通过，atol=rtol=0；[after](msar_lno_audit/m1/old-after.json) |
| 本轮修改前 KCDNO all/off、matched LRSA × point/conv | **6/6** strict 同权重输出精确，CPU FP32/math SDPA，B2/N35/d8/L2，conv 为5×7；[after](msar_lno_audit/m1/current-after.json) |
| 基准文件完整性 | 136 个 K0 artifact + 6 个新旧core夹具 SHA256 不变；[记录](msar_lno_audit/m1/fixture-hashes.json) |
| 真实数据、MSAR 合成训练/辅助损失/GPU/参数量、远端 torch2.11/cu128 | **未执行**，新模型尚不存在；不报告精度、收敛或速度 |

K0 replay 的旧 Transolver 实际使用本机 CUDA；旧 CDLNO 使用 CPU/真实 PyG 对象和保存协议，forced math SDPA、FP32、AMP/TF32/compile off。该有限 GPU 回放不能写成“MSAR GPU 通过”。41 条包括八任务基本接口及既有边界案例，不等于所有历史权重/真实数据协议全部被验证。6 个新增基准只是当前 core 输出/state 往返，未额外宣称训练步或工业 wrapper 验收。

首次新增配置测试有 1 项错误：仅提取了入口顶层 parser，遗漏 AirfRANS eval helper 补充的 model 参数。保留 [初次失败日志](msar_lno_audit/m1/config-tests-initial.txt)。修复为提取实际 helper 的声明段，并在新 family 副本内扩展受限 model choices；没有修改生产入口。最终32项全部通过。

可运行的配置预览（无训练、无模型构造、无目录创建）：

```bash
python -B -m cdlno.msar_lno --task darcy
python -B -m cdlno.msar_lno --task elasticity --profile full --input-tokens 972
python -B -m cdlno.msar_lno --task darcy --coverage-mode off --coverage-weight 0
```

三条实际输出已保存为 [Light](msar_lno_audit/m1/light-preview.json)、[Full/扩张](msar_lno_audit/m1/full-preview.json)、[no_regularizer](msar_lno_audit/m1/no-regularizer-preview.json)。从其他 cwd 使用前先进入实际 checkout，或将实际 checkout 根目录加入 PYTHONPATH；不要照抄本机路径到不同远端位置。

## E. 冻结范围与未完成项

614 个修改前文件中，612 个字节一致，仅旧 STATUS/memory 追加阶段记录；数据、loss、optimizer/scheduler、训练/eval 循环、原 CLI/factory、模型参数名/forward、已有 checkpoint/可视化/seed/show 脚本、旧测试和依赖未改。无 commit/push/PR/reset/训练/下载。

Down/Up/latent block/AttnRes、coverage 原语、完整 core、任务 factory/八任务 CLI/wrapper/loss/日志/权重接入仍是后续授权阶段的工作。当前配置只记录数学合同，不能把 off 优化路径、有效参数量、权重 strict 加载或原损失连接标为 MSAR 已通过。没有借本阶段补做此前 resume 等旧阶段缺项。

## F. 已完成自审与剩余依赖

1. **profile 与解析隔离**：两 profile 精确、depths 固定、M>N 不裁剪；旧 parser 默认未进入显式字典；10 声明样式/旧family返回路径通过。
2. **结构与目标分离**：coverage 不在 architecture，eval 允许目标/运行差异而拒绝结构冲突；完整配置字段和文件不可覆盖检查通过。
3. **family/包/目录**：只导出稳定配置路径，没有会破坏旧导入的占位模型；包发现与三 cwd 导入通过；family/profile/有效coverage三级隔离，同名拒绝覆盖。
4. **旧行为与同权重证据**：41+6 精确回放、夹具hash、既有源文件冻结通过；没有随机重建输出冒充旧基准，也没有宽松加载。
5. **报告边界**：没有把 JSON/AST 测试写成新模型/八任务训练成功，没有修旧数据/训练问题；没有执行 M2。

剩余证据依赖是本工作区缺失的 MSAR M0 报告/索引，以及目标远端运行环境未验证。没有发现需要改变本轮 Light/Full 规格的冲突。后续涉及数学与训练接入时，需继续以实际源码与可取得的 M0 记录核对，不能把本轮局部映射当成完整 M0 的替代。

本M阶段结束，未执行下一阶段。
