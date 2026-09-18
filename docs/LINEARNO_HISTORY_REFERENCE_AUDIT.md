# LinearNO history extension R0：只读来源与基线审计

日期：2026-09-18。本文只审计当前 checkout、已有纯 LinearNO 和固定参考资料；本阶段没有实现 AttnRes 或 history-conditioned K。

## 状态与边界

R0 状态为 **PARTIAL**。纯 LinearNO 的独立 oracle、官方同权重 parity、八个完整 block、输入/参数梯度、AdamW 一步和严格 checkpoint 往返均有通过证据；当前 CPU 回归为 84 个方法，78 个通过、4 个 GPU 方法按本阶段限制跳过，另有 4 处非数学冻结断言失败。失败来自历史 L1 快照与当前 HEAD 已接受的 README、`path.sh`、旧复现矩阵差异，及相同 `path.sh` 字节断言；没有发现 LinearNO 数学 parity 失败。该结论不升级为 PASS，也不阻止后续研究阶段重新审查这些冻结断言。

本阶段未读取真实数据、未运行 GPU、未启动长训练、未安装依赖、未修改生产代码/模型/factory/配置/测试/launcher，未执行 git 写操作。所有证据和二进制 fixture 在仓库外的 `/home/hwz/CDLNO-artifacts/linearno-history-r0-20260918T055437Z`；仓库内只新增本研究审计文档及 `docs/linearno_history_audit/r0/` 证据索引。

## Git 与用户修改保护

R0 开始时记录：

```text
root   /home/hwz/CDLNO
branch main
HEAD   e1c8552799dd9a7b939ddba1dfabe20e88dfa41a
tree   435c3b3a14112f98e4e6de0fbf1e2b60762bea68
remote origin git@github.com:hxh5159/CDLNO-w.git
status ## main...origin/main
       M .claude/settings.json
       M depth_ablation/README.md
       ?? depth_ablation/schedule.sh
```

起点清单为 1241 tracked、1 untracked、328 ignored。完整 size/SHA-256 分类清单见 [start-manifest.json](linearno_history_audit/r0/start-manifest.json)，起点 diff 见 [start-tracked.diff](linearno_history_audit/r0/start-tracked.diff)。只有根 [AGENTS.md](../AGENTS.md) 适用，未发现嵌套 AGENTS.md。

审计过程中用户的 `depth_ablation` 三项变化被提交到新 HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`；这是外部并发状态变化，R0 没有执行 commit。起点到结束的 Git 变化记录在 [concurrent-git-state.json](linearno_history_audit/r0/concurrent-git-state.json)，起点 manifest 中所有文件内容 hash 仍一致。该并发提交不计为本阶段生产修改。

## 固定来源

| 来源 | 固定版本/树 | 只读证据 |
|---|---|---|
| hxh 当前 checkout | `main`，R0 起点 `e1c8552` | `baseline.json`、manifest |
| thuml/Transolver | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`，tree `22e7386201596e45236907f774478f79d3913da7` | [local-sources.json](linearno_history_audit/r0/local-sources.json)；71 blobs 逐 blob 校验 |
| HiPRL/LinearNO | `3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`，tree `4aa99d7dece8462d932e690a82e23dba78b11b0b` | 同上；61 blobs 逐 blob 校验 |
| LinearNO 论文 | arXiv `2511.06294v3` | [papers.json](linearno_history_audit/r0/papers.json)；HTML SHA-256 `637e2953…470fd` |
| Attention Residuals | git `85e2231`；PDF/arXiv v1 | [remote-source-trees.json](linearno_history_audit/r0/remote-source-trees.json)、PDF/text |
| Kimi K3 | git `3cb39df`；arXiv v2 | 同上；固定树只有 README、许可证、logo、报告 |

LinearNO 固定树没有 LICENSE/NOTICE 文件；这里只记录来源事实和分发风险，未作法律结论。AttnRes 固定树没有可执行模型代码，只有 README、报告 PDF 和图片；Kimi 固定树没有可用于本项目的模型实现。参考树保存在目标仓库之外，没有 vendor、merge、rebase 或共同祖先推断。

## 既有实现调用链

### Standard 六任务

```text
tran_evaluate/linearno/{airfoil,darcy,elasticity,pipe,ns,plasticity}_{train,eval}.sh
  -> tran_evaluate/_common.sh / _static.sh
  -> PDE-Solving-StandardBenchmark/{exp_airfoil,exp_darcy,exp_elas,exp_pipe,exp_ns,exp_plas}.py
  -> cdlno_entry.py -> cdlno.linearno.standard_entry.parse_args
  -> model_dict.get_model
  -> model.LinearNO.Model
  -> Model.forward(x, fx, T=None) -> blocks -> prediction
  -> StandardRun / cdlno.linearno.checkpoint.py -> eval 读取 metadata 后 strict state_dict
```

标准模型类在 [PDE-Solving-StandardBenchmark/model/LinearNO.py](../PDE-Solving-StandardBenchmark/model/LinearNO.py)，原语在 [cdlno/linearno/attention.py](../cdlno/linearno/attention.py)。`plain/temp/conv/conv_temp` 由独立 `linearno_variant` 解析；注册键仍是 `LinearNO_Structured_Mesh_2D` 或 `LinearNO_Irregular_Mesh`，不会把研究开关塞入旧 `--model`。

### AirfRANS

```text
tran_evaluate/linearno/airfrans_{train,eval}.sh
  -> Airfoil-Design-AirfRANS/main.py 或 main_evaluation.py
  -> cdlno_entry.py -> cdlno.linearno.air_entry.parse_args/run_cli
  -> cdlno.linearno.airfrans.AirfRANSLinearNO
  -> Data.x(7)+原始 Data.pos(2) reference distances -> 8 blocks -> [N,4]
  -> AirRun / ensemble metadata + strict member state_dict -> eval
```

Air 使用绝对 `M=32`、死的 `temperature` 键保留但 forward 不使用；采样、mask、指标和原 Data 对象由旧入口保留。

### ShapeNet-Car

```text
tran_evaluate/linearno/car_{train,eval}.sh
  -> Car-Design-ShapeNetCar/main.py 或 main_evaluation.py
  -> models/cdlno_run.py -> cdlno.linearno.car_entry.parse_args/run_cli
  -> cdlno.linearno.shapenet.ShapeNetLinearNO
  -> tuple(Data,None), Data.x(7), 单图校验 -> 8 blocks -> [N,4]
  -> CarRun / strict pair + 保留可信 whole-object/list 协议 -> eval
```

Car 使用 `M=key_ratio*d_h` 和官方拼写 `tempreature_q/k`；多图没有可靠隔离时拒绝。工业入口不会把图边送进新模型，但保留 loader/evaluator 所需图构造。

## 纯 LinearNO 基线证据

原子 attention 生产实现为 `Q=softmax_M(XWq)`、`K=softmax_N(XWk)`、`V=XWv`、`C=K^T V`、`Y=QC`；Q/K/V 小投影无 bias、跨 head 共享，未构造 `N×N`。完整 Standard、Air 和 Car 均执行八个完整 block，最后才做 output norm/head。A0K0 的未来要求是继续使用这些旧类、旧 state_dict 和旧 checkpoint schema。

现有测试生成的 [parity-summary.json](linearno_history_audit/r0/parity-summary.json) 汇总如下：

| 套件/域 | 行数 | dtype | 最大绝对误差 | 容差 |
|---|---:|---|---:|---|
| attention oracle | 12 | float64 | `8.88e-16` | `1e-12/1e-10` |
| attention oracle | 12 | float32 | `7.15e-7` | `1e-6/1e-5` |
| attention 官方同权重 | 24 | float64/32 | `0` | `1e-12/1e-10`, `1e-6/1e-5` |
| Standard 官方逐 block/最终/梯度/一步 | 10 | float64/32 | `1.87e-16`（double） | 同上 |
| AirfRANS 官方完整模型 | 2 | float64/32 | `0` | 同上 |
| ShapeNet 官方完整模型 | 3 | float64/32 | `0` | 同上 |

每项 parity 使用同一权重、输入和 RNG；独立 double oracle、逐参数梯度、optimizer step、strict checkpoint 也在已有测试中执行。八任务新增小型 fixture 在两个独立 CPU 进程中 JSON 完全一致，包含八个 block 输出、最终输出、所有输入/参数梯度、更新后权重、RNG 演化和 state_dict key/shape/count；见 [fixtures.json](linearno_history_audit/r0/fixtures.json) 与 [fixture-replay.json](linearno_history_audit/r0/fixture-replay.json)。这些是 d16/h2/M8 等缩小配置，不是正式参数量、精度或收敛证据。

旧 Transolver 的 model/factory、旧 CLI、旧 state_dict/whole-object 路径通过既有数值与静态回归；L0 冻结比对见 [old-l0-freeze-comparison.json](linearno_history_audit/r0/old-l0-freeze-comparison.json)。历史已批准的集成差异（例如 `cdlno/experiment.py`、Car runner 和研究测试）与 R0 新增变化分开记录。

## 回归命令与结果

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  python -B -m unittest discover tests/linearno -p 'test_*.py' -t tests -v

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -B -m unittest discover -s monitor -p 'test_*.py' -v
```

第一条实际运行 84 个方法：78 passed、4 skipped（CUDA 限制）、4 个失败断言。失败只为：`README.md` 快照 hash、`path.sh` 快照 hash、旧 `docs/LINEARNO_REPRODUCTION_MATRIX.md` 标题前缀，以及静态 launcher 对同一 `path.sh` 的字节 hash；详细 traceback 在 [baseline-tests.json](linearno_history_audit/r0/baseline-tests.json)。第二条 monitor 为 3 passed。没有为消除失败而修改测试。

## 来源台账与研究边界

| 内容 | 来源/性质 | R0 结论 |
|---|---|---|
| LinearNO Q/K 因式注意力、任务变体 | LinearNO 论文与官方固定源码 | 已实现并通过 parity |
| AttnRes 的深度来源 softmax、RMSNorm、零初始化 pseudo-query | Attention Residuals v1；Kimi K3 §2.2 仅作相关材料 | 只借鉴思想；固定树没有生产代码 |
| A：latent-summary AttnRes | 本项目对 LinearNO latent `C_raw` 的专用适配 | **不是原 CDPA，也不是 Kimi 官方 AttnRes**；需要后续阶段冻结实现 |
| K：history-conditioned compression K | 本项目原创设计 | LinearNO、AttnRes、Kimi 固定来源均没有官方实现 |
| `P=QK^T` 层间核相似度监测 | 当前 `monitor/` 的 LinearNO 基础 Q/K 监测 | 只用于诊断；Transolver 的核曲线不是 LinearNO 实验证据 |

现有 monitor 的 [runtime.py](../monitor/runtime.py) 通过 hook 重建基础 Q/K 并计算 Gram 公式；它不会自动理解未来 K 修改后的有效因子。后续必须提供显式 effective-factor 诊断接口，不能把旧曲线冒充 A/K 实验证据。

## 最小安全插入点（仅建议）

1. 新建与 `cdlno/linearno/` 平行的研究模块/`linearno_history` family；A0K0 直接复用旧 Standard/Air/Car 类，不在旧类中加入条件分支。
2. 在 factory 选择完成、真正构造 LinearNO core 之前解析两个正交开关；旧 parser、旧模型 key、旧 checkpoint 路径保持原分支。
3. 在每次 model forward 的 block loop 内建立局部 raw-summary tuple；不得跨样本、跨 batch、跨 NS 时间步缓存。
4. 训练/评估 adapter 只消费明确的 prediction/aux 合同，旧训练器不认识新字段；metadata 先解析 family/profile/constructor，再 strict=True 加载。
5. monitor 与模型通过显式、可关闭的 detached factor接口连接，默认不存大 tensor、不同步 CPU。

上述位置只是 R1 以后审查用的安全边界，R0 没有实现任何一个插入点。

**本 R0 阶段结束，未执行 R1 或后续阶段。**
