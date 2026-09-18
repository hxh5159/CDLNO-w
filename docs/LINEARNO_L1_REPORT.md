# LinearNO L1 — 配置协议与修改前回归依据

日期：2026-09-17。**阶段状态：PASS。** 仅指 L1 非资源门控验收；没有 LinearNO 算子、模型注册或八任务生产接线。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`。后续阶段未执行。

## A. 实际范围和基线

当前 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`，remote `hxh5159/CDLNO-w`。沿用已审查 [L0 实际映射](LINEARNO_REFERENCE_AUDIT.md) 和 [复现矩阵](LINEARNO_REPRODUCTION_MATRIX.md)。没有按旧提示词重建 `66bc489` 或假设存在 `LINEARNO/PropagationMonitor`。

修改前新快照：`/home/hwz/CDLNO-artifacts/linearno-l1-before-znfjyu2s/source/`；[baseline.json](linearno_audit/l1/baseline.json) 保存 1192 个既存 tracked/untracked/ignored 文件的分类、size、SHA-256，包含 L0 新增文件及用户两份 PLAN。L0 的外部快照继续保留。既存 STATUS 仅增量前置 L1 段落；其他旧文件逐字节保持。没有修改旧模型、八任务 parser/factory/训练/eval/checkpoint helper、launcher、依赖、已有测试、AGENTS、其他模型报告或 memory。

## B. 新文件及边界

| 文件/符号 | 职责 |
|---|---|
| `cdlno/linearno/__init__.py` | 隔离命名空间；不导入模型、不注册 factory；符合当前 `cdlno*` 安装范围 |
| `_profile_data.py:CATALOG` | 根据 L0 审计整理的独立配置数据；不在运行时读取 docs；不 vendoring 官方算子 |
| `profiles.py:resolve_config/parse_overrides/validate_resolved` | family=`linearno`，三个 profile、显式覆盖来源、rank 映射、结构校验和 hash |
| `profiles.py:require_resolved_objective` | 工业 paper objective 未定时明确拒绝声称已解析成可训练目标 |
| `schema.py:validate_model_spec` | 模型 class_path 与真正 constructor kwargs 分开；提供实际构造器时检查完整 signature，拒绝未知字段，即使类含 `**kwargs` |
| `schema.py:numerical_state/restore_numerical_state/pack_state/unpack_state` | 无 pickle 的数值编码；保留 dtype/shape/原始字节/hash、tuple、optimizer 整数 key |
| `schema.py:make_metadata/read_metadata/write_metadata/compare_metadata` | 独立元数据协议；只读校验，创建拒绝覆盖；结构严格冲突，运行/协议差异单列 |
| `tests/linearno/legacy_worker.py/capture_fixtures.py` | 原 Transolver 同权重小夹具；独立子进程、真实 factory/类、真实 PyG Data |
| `tests/linearno/rng_worker.py` | 原 Transolver 的测试专用续跑；调用已有 RNG 隔离/周期绘图基础、提取实际时间排列与抽样表达式 |
| `tests/linearno/test_{profiles,schema,legacy,rng}.py` | 16 个测试方法，内部覆盖八模型、24 配置、10 parser、两个 scheduler 等多子案例 |
| `tests/linearno/fixtures/transolver_cpu.json` | 约 144 KB 文本；seed/config、输入/权重哈希、完整输出、key/shape/count；不提交二进制模型 |
| `docs/linearno_audit/l1/` | 基线、夹具索引、24 resolved 配置、环境、转换清单、实际测试日志与交付核查 |

`parse_overrides` 是独立模块的演示解析器，**没有挂到任何任务 argparse**。下列是目前可执行的配置查看，不是训练命令：

```bash
python -B - <<'PY'
from cdlno.linearno.profiles import parse_overrides, resolve_config
profile, explicit = parse_overrides([
    '--linearno-profile', 'official_release', '--linearno-rank', '48'])
print(resolve_config('darcy', profile, explicit=explicit))
PY
```

不创建 LinearNO 假类/占位构造器让 factory 成功。`model_spec` 的 metadata-only 测试使用 L0 记录的真实官方 class path/构造器字段作为描述；未传实际构造器时只允许审计过的官方签名，拒绝未知类、缺失/额外字段及与profile不一致的结构值，不导入或构造官方模型；signature 校验单测使用真实 `torch.nn.Linear`，不是假 LinearNO。后续 loader 必须显式传实际任务构造器完成 signature 校验，再构造并 `strict=True` 加载；L1 元数据通过不代表已完成模型重建。

## C. 公式/字段 → 代码 → 证据

| 已确认规格 | L1 映射 | 可执行证据 |
|---|---|---|
| Q沿M、K沿N；小投影跨head共享且Q/K独立 | `derived_model` 仅记录轴、投影/bias/温度/初始化约定 | `test_all_24_match_independent_L0_inventory_and_roundtrip`；**未实现/未做算子 parity** |
| Standard `M=key_ratio`；Air `M=slice_num`；Car `M=key_ratio*(d/h)` | `linearno_rank`为实际M；`qk_dim`为互斥别名；`rank_mapping`保存原名、原值及公式 | Car默认32→key_ratio1；拒绝不整除head_dim的Car rank；M>0、d%h=0 |
| 四种 Standard variant 及两工业专属拓扑 | `linearno_variant` 与旧 `--model` 完全分开 | 非法跨任务variant拒绝；未来注册名不会被当plain |
| Conv/统一网格位置要求N=H×W | `validate_model(...,N=...)`；H/W独立 | 5×7通过、N36拒绝；Elasticity任意N允许，M>N不裁剪 |
| 优先级显式CLI > profile > family默认 | 各叶字段 `cli_explicit/profile/family_default`；`legacy_default`为协议保留枚举、当前不应用 | 旧slice_num/hidden/mlp/batch仅记 `ignored_legacy_defaults`；空argv没有覆盖字段 |
| paper/release/matched三配置 | `_profile_data.py`，默认paper；8×3独立resolved记录 | [resolved-profiles.json](linearno_audit/l1/resolved-profiles.json)，逐项对照L0审计 |
| checkpoint结构与运行协议分离 | `model_spec`仅class_path/constructor_kwargs；独立profile/data/objective/evaluation/provenance/normalizer/resume/ensemble | JSON往返/错family/错shape/错hash/缺字段/禁止覆盖/重复JSON key/结构冲突及resume协议冲突 |
| 数值normalizer不能eval重拟合 | 命名state、算法、fit split、数据checksum、dtype/shape/base64/hash | float64、bfloat16、标量、空tensor、非本机字节序NumPy可逆；损坏字节/shape拒绝 |
| optimizer/scheduler/RNG完整性 | tagged state保留整数key/tuple；Python、NumPy、Torch CPU/CUDA、显式generator device/state、sampler mapping；可选scaler | 合成真实Transolver训练结果完整编码/还原一致；RNG局部验证不修改全局流 |
| final权重，禁止test挑best | `checkpoint_role`；best仅`selection_split=validation`且metric非空 | test split拒绝；final/epoch不允许伪造best字段 |

三方SHA、paper版本/hash、base_commit/dirty、源码与规范化patch hash、argv、环境及config/schema版本/hash均要求存在。数据checksum在实际checkpoint元数据中不能为未知；本轮单测明确写 synthetic checksum。profile规划阶段仍允许 `data.checksums=None`，不能拿规划记录冒充训练存档。ensemble成员顺序、id、相对路径、hash显式校验。

`compare_metadata` 对 model_spec、结构和任务严格拒绝；runtime变化单独返回；eval协议变化也单列，不能静默采用新指标。resume拒绝training/objective/evaluation/data/normalizer变化。这里没有实现实际权重读写、生产resume或信任外部pickle。

Car备用位置编码在新schema最终自审中按**官方LinearNO**核正为二维 `[0,1]²` 距离替换；不套用旧Transolver Car的三维拼接。默认Car为unified_pos=False不受影响。Air保留append与未使用`linear=True`描述，Car保留未使用`isregular=False`；工业接口无时间输入，显式time_input=True拒绝。

## D. 同权重旧模型与随机流证据

| 旧模型小夹具 | 参数量 | 输出 | 真实选择 |
|---|---:|---|---|
| Darcy | 3589 | [2,35,1] | 实际 `model_dict.get_model` structured2D |
| Elasticity | 1413 | [2,19,1] | 实际factory irregular |
| Airfoil | 3461 | [2,35,1] | 实际factory structured2D |
| Pipe | 3461 | [2,35,1] | 同上 |
| NS | 3733 | [2,35,1] | 同上，fx10 |
| Plasticity | 3648 | [2,35,4] | 同上，T[B,1] |
| Car | 1520 | [19,4] | 原main选择分支AST + `models.Transolver.Model`，tuple(Data,geom) |
| AirfRANS | 2224 | [19,4] | 原main选择分支AST + `models.Transolver.Transolver`，Data |

这些是 **d8/h2/L2/M4、dropout0、eval、合成输入**，不是论文配置参数量。八类分别新进程从原子项目 cwd 启动，保存同份state后先破坏当前权重再strict加载，结果max error=0；Car whole-object、Air model-list仅加载测试即时自行保存的本地可信对象。另与固定文本输出按CPU FP32 atol1e-6/rtol1e-5比较。参数key/shape/count与权重hash严格相等；生成器SHA、Python/Torch版本、输入hash与完整输出在夹具。跨版本初始化/RNG若不等不能自行重录掩盖变化，必须审查后说明环境差异。

原类统一位置有硬编码`.cuda()`。本轮为了无GPU计算的CPU基线，仅在独立测试进程临时将`Tensor.cuda`映射到CPU；未改原模型源码/位置数值。**这不是原代码原生CPU支持或GPU parity证明**。真实PyG Data可用，但不代表radius_graph可用。

随机流测试明确拆分：

- 原Transolver + synthetic MSE，4 epochs×2 batches=8更新；连续8步，对照4步保存→新进程恢复4步；OneCycle及Cosine分别执行。测试专用存档保存optimizer/scheduler/RNG，**不是已有八任务原生resume**。
- 开启真实`PeriodicFields`生成PDF/PNG/数值产物并保存权重，对照全部关闭。调用已有`isolated_evaluation`，附加消耗Python/NumPy/Torch/loader generator来验证恢复。
- 比较所有记录、batch顺序、loss/lr、最终权重、optimizer全部moments、scheduler、RNG均CPU逐位相等。
- Plasticity直接AST提取原`random_collate_fn`：逐样本`torch.randperm(20)`，同时检查目标时间同步排列；不是NumPy。另独立消耗/比较NumPy排列以覆盖其RNG。
- Air直接提取原`train.main`的`random.sample(range(data_sampled.x.size(0)),hparams['subsampling'])`表达式，用真实Data；不是替代完整radius_graph/采样训练协议。
- 另重跑现有 `TrainingArchive`/`EpochObserver` CPU基础测试：3种CDLNO前段模式×两scheduler，真实新进程续跑、存档事务、恢复失败不污染状态等。该基础本身绑定CDLNO，不给旧Transolver伪造config来假称接入。

## E. 实际命令、结果与环境

```bash
# 新L1完整测试（不导入exp/main）
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  linearno.test_profiles linearno.test_schema linearno.test_legacy linearno.test_rng -v

# 可重算夹具，必须指定一个不存在的新文件；不可直接覆盖受审查fixture
python -B tests/linearno/capture_fixtures.py /NEW/ABSOLUTE/transolver_cpu.json

# 当前旧产物/存档CPU测试；实际运行排除了GPU方法
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import unittest, test_training_state
loader=unittest.TestLoader(); suite=unittest.TestSuite()
for name in loader.getTestCaseNames(test_training_state.TrainingStateChecks):
    if name != 'test_gpu_fresh_process_identity_onecycle':
        suite.addTest(test_training_state.TrainingStateChecks(name))
suite.addTests(loader.loadTestsFromName('test_experiment_records'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

旧源码六项AST/字节检查命令与L0相同，详见 [L0实际命令](linearno_audit/l0/legacy-static-results.json)，本轮输出 [old-source-tests.txt](linearno_audit/l1/old-source-tests.txt)。

| 本轮执行 | 结果 |
|---|---|
| 新L1完整16方法 | 16/16，46.241s；8个freshcwd同权重模型、24profile、10parser、两个scheduler续跑及真实绘图；无skip |
| 最终schema自审修正后相关11方法 | 11/11；[config-schema-final.txt](linearno_audit/l1/config-schema-final.txt)；只重跑受影响测试 |
| 既存存档/产物22方法 | 0失败/错误，67.785s；1个Air完整采样epoch子案例skip；其余包括24任务/模式产物、重复eval不改sidecar、Car实际合成epoch、Air原weighted-loss/记录片段 |
| 旧入口完整AST及核心文件 | 6/6，0.514s；包括34个Car/Air文件、21个Air文件字节核对 |
| L1新增Python语法 | Python3.10语法AST检查；不安装或生成pycache |
| 冻结 | L0重点221文件不变；L0既存1161文件不变；L1起点1192文件仅允许STATUS增量，其余1191不变；详情delivery-check |

首轮新schema测试遇到测试比较器`_same`不能比较NumPy数组，改为`np.testing.assert_array_equal`后通过；没有改既存比较器。未发生被放宽容差掩盖的数值失败。最终自审修正均限新文件。

环境：Python3.13.9，Torch2.13.0+cu130，PyG2.3.1，timm1.0.28，einops0.8.2，NumPy2.2.6。RTX5090 Laptop可见，但L1只做CPU模型计算，没有GPU模型/AMP。缺torch_cluster/torch_scatter/pyg-lib；不安装。项目声明的Python3.10/3.11安装范围保持；当前仅源目录测试，不宣称Python3.13包安装或远端Torch2.11/cu128兼容验收。

## F. 官方checkpoint转换清单与未执行范围

[机器清单](linearno_audit/l1/checkpoint-conversion-policy.json) 固定来源及源码hash：

| 来源 | 必须保留/转换策略 |
|---|---|
| Standard bare state_dict | 四variant、actualM、完整kwargs事先明确；不能靠当前旧parser默认推断 |
| Car whole-object | 原`blocks.i.Attn.tempreature_q/k`拼写保留；任何未来改名须一一可逆及全key/shape校验 |
| Air whole-object/list | 未用`blocks.i.Attn.temperature`仍保留；显式成员id/order/path/hash，不丢参数 |
| 所有转换 | 拒绝缺失/未知/重复/shape不符key，不补随机参数；最终strict=True并做数值parity |
| 外部pickle | 本轮未加载；只有用户明确提供并确认来源可信后，后续授权阶段局部、隔离使用weights_only=False；不扩大旧信任边界 |

NOT RUN：LinearNO算子/官方→移植forward、梯度、optimizer parity及模型checkpoint重建（L1未实现）；实际GPU/AMP；Air完整采样图epoch（缺torch_cluster）；八任务真实loader、数据checksum、真实normalizer、mini-run、收敛/精度/时长；远端Python3.10/Torch2.11/cu128；任务级完整resume（当前不存在，后续阶段接线）。既存monitor不存在，标N/A，不能算通过。没有改模型以适应测试。

## G. 自审与后续需要审查的边界

已自审：①全部旧文件/用户dirty保护；②旧model注册与新variant/来源隔离；③同权重严格checkpoint和真实随机源；④工业paper未定字段不伪装可训练默认；⑤数值normalizer/完整RNG/构造器与运行字段分离。结果均见测试/冻结证据。

剩余事项延续L0，不在L1自行决策：工业paper rL2通道聚合与normalized/physical空间（C08）；Car实际fold/sample force路径（C12）；Airfoil评价cadence/Pipe非canonical长度split/Air图构造差异（C25—C27）。下一阶段可继续独立算子，不代表上述任务接入选择已解决。无新增需要改变已确认架构的冲突。

**本L阶段结束，未执行下一阶段。**
