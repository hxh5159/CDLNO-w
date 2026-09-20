# LL9R - RB AMP and Regression Repair

**Status: PASS.** This repair completes the bounded LL9R scope only. It does
not validate real data, convergence, accuracy, or paper results. LL10 was not
executed.

## Scope and starting state

The repair started from `main@5b991226c5354af3332b2f7306b370aef0950c79`
with remote `https://github.com/hxh5159/CDLNO-w.git`, preserving the existing
dirty worktree. The source snapshot is
`/home/hwz/CDLNO-artifacts/loop-ll9r-before-j6m5x1a9`; no reset, clean, stash,
checkout, rebase, commit, or push was used. The final freeze is recorded in
[end-freeze.json](loop_linearno_audit/ll9r/end-freeze.json).

## RB AMP repair

`LinearNOLoopCore._rb_receive` is the sole numerical production change. It
uses the core-entry RB anchor's dtype as the explicit canonical dtype and
creates a local tuple for that receiver call:

```python
local_sources = tuple(
    source if source.dtype == anchor.dtype else source.to(dtype=anchor.dtype)
    for source in sources
)
```

The authoritative RB raw partial stays in its autocast dtype and keeps its
autograd edge. The helper does not mutate cached tensors, add buffers or
parameters, detach gradients, alter source order/count, add identity residuals,
or add `1/R`. SR and LB do not call this helper. The shared
`PointDepthAttnRes` remains strict about homogeneous source dtypes.

The pre-repair failing path had FP32 anchors and FP16/BF16 raw partials for
Airfoil, Elasticity, Pipe, AirfRANS, and Car. The local cast makes receiver
inputs homogeneous FP32 while the recorded raw cache remains FP16/BF16. LB's
`Y = H + F(H)/R` and `Delta = Y - H` sources were already homogeneous in the
same AMP fixtures, so LB production code was not changed. The complete dtype
records are in [rb-amp-dtype-summary.json](loop_linearno_audit/ll9r/rb-amp-dtype-summary.json)
and [precision-compatibility.json](loop_linearno_audit/ll9r/precision-compatibility.json).

## Compatibility and isolation

Standard, AirfRANS, and Car entry adapters now import loop support only after
an explicit loop flag or `family=linearno_loop` sidecar selects it. Legacy
parsers keep their original actions, defaults, namespace, and selection. If
the shared loop package is absent, only an explicitly selected loop run fails,
with a clear parser error.

Historical checks now use their recorded full Git SHA rather than moving
`HEAD`. AirfRANS AST comparisons remove only the versioned LinearNO/resume
nodes and reject a mutation to retained training/evaluation calls. The source
freeze excludes runtime bytecode and pytest caches, while retaining exact
checks for actual source files. The three reviewed documentation revisions are
closed by commit-pinned hashes in
[frozen-revisions-v1.json](loop_linearno_audit/ll9r/frozen-revisions-v1.json);
unknown hashes fail closed.

## Validation

- CUDA synthetic matrix: 144/144 passed: FP32 48/48, AMP FP16 48/48, AMP BF16
  48/48. This includes all eight tasks, two presets, and SR/RB/LB. The twenty
  formerly failing RB AMP cases are all finite through forward, backward, and
  AdamW; AMP was not disabled or silently retried in FP32.
- Deterministic pre/post capture: 192 cases. The 172 cases that ran before the
  repair are bitwise equal for initial state, inputs, output, loss, input and
  parameter gradients, optimizer-step state, and CPU/CUDA RNG. The 20 former
  RB failures now pass. SR/LB account for 128 bitwise-equal cases. The separate
  nondeterminism review attributes non-deterministic CUDA SR roundoff to the
  backend; deterministic mode restores exact replay without a tolerance change.
- Parameter and cost audit: all 96 task/preset/mode/rank configurations equal
  their analytical parameter decomposition and the prior LL9 accounting.
  State-dict keys/shapes and parameter counts are unchanged.
- Synthetic native closures: Standard 18/18 and industrial 6/6 completed
  original-main train, interrupted checkpoint, resume, and fresh-process eval.
  Old archive replay adds 9/9 exact Darcy/AirfRANS/Car checks across all modes;
  weights, optimizer/scheduler, RNG, and output hashes match and source archive
  bytes remain unchanged.
- Existing regression inventory: 80 modules, 689 tests, 653 passed, 36 skipped,
  zero failures and errors. The final fresh full pass captured one failure caused by
  pytest rewriting `.pytest_cache/v/cache/lastfailed`; it is retained in
  `final-regression/regression-results-before-isolation-fix.json`. The repaired
  source-boundary test was rerun 3/3, and the final composition is in
  [summary.json](loop_linearno_audit/ll9r/final-regression/summary.json).
- LL9R-specific checks plus existing loop performance and launcher checks: 19/19
  passed. `git diff --check` passed.

The CPU timing/forward matrix uses small synthetic tensors and reports only
median/p90 module behavior. It does not measure real epochs. No real loader,
full training, VTK metric, remote Python 3.10/Torch 2.11/CUDA 12.8 run,
distributed run, compile run, convergence claim, or SOTA claim was made.

## 实际修改与旧失败逐项复核

本阶段相对起点修改的生产文件仅以下六个，另新增一个精确 provenance helper。
不是相对 HEAD 的累计 diff：此前 LL6–LL9 工作树改动原样保留。

| 文件 | 本阶段改动 |
|---|---|
| `cdlno/linearno_loop/core.py` | 新增 RB 专属局部 dtype 边界，两个 RB 调用点改用它 |
| 三项目 `cdlno_entry.py` / Car `models/cdlno_run.py` | 先判定 loop 意图/metadata，再延迟 import |
| `cdlno/linearno_loop/ll7_projection.py` | 先精确剥离 LL9R 修复，再使用已有 LL7 projection |
| `cdlno/linearno_loop/industrial_state.py` | 对同一精确修复作 provenance 兼容投影 |
| 新 `cdlno/linearno_loop/ll9r_projection.py` | 登记六文件的固定替换和次数；陌生改动不会被任意接受 |

旧 archive 的 provenance 保持原版本，是经过精确投影的兼容视图，不能解读为
当前源码逐字节未变化。实际修复源码、起止 hash 和 patch 另保存在本阶段证据。
checkpoint payload/schema、构造 kwargs、参数组及 strict 规则没有放宽。

测试改动包括两项 history fingerprint、旧文档冻结、Air AST projection、
`-I -B` 子进程、源码缓存边界和 LL8 launcher 冻结适配。新增
`tests/historical_git_base.py`、`frozen_revisions.py`、`linearno_air_projection.py`、
`ll9r_projection.py`、`ll9r_test_source_projection.py` 及 `loop_linearno/test_ll9r.py`。
LL8 冻结对六处已授权生产改动使用相同精确反向 projection，原 hash 不刷新；
二进制文件仍直接比较字节。完整名单与关键 diff 见
[source.diff](loop_linearno_audit/ll9r/source.diff)。

| LL9 原失败方法（10 个，15 条断言） | 原因与最终结果 |
|---|---|
| `test_experiment_records.test_entries_early_lifecycle_and_complete_observation_only_ast` | Air 的三个后续插入节点未投影；精确剥离后 PASS |
| `test_kcdno_airfrans.test_defaults_and_exact_frozen_projection` | 同一 Air LinearNO guard；PASS |
| `test_kcdno_delivery.test_legacy_parser_imports_without_shared_package` | 未选 loop 就 import 共享包；lazy import 后 PASS |
| `test_msar_industrial.test_frozen_complete_entries_trainers_yaml_and_core` | Air guard/resume 后续插入；PASS |
| `test_periodic_visualization.test_production_hooks_preserve_complete_AST` | Air main/train 两处后续插入；PASS |
| `linearno.test_history_industrial.test_old_source_fingerprints_task_bodies_and_launchers` | 漂移的 HEAD 基点；恢复历史固定 SHA 后 PASS |
| `linearno.test_history_static.test_old_source_fingerprint_and_frozen_task_bodies` | 同一历史 Git 基点问题；PASS |
| `linearno.test_legacy.test_all_preexisting_files_and_absent_frozen_directories` | README/path/复现矩阵三项已授权变更；逐项 commit/hash/diff 台账后 PASS |
| `linearno.test_static_integration.test_all_new_launcher_actions_profiles_and_old_scripts_static` | path.sh 已授权远端路径注释；PASS |
| `loop_linearno.test_isolation.test_preexisting_bytes_after_exact_LL6_routing_projection_and_classifications` | `.pyc` 不是源码；修正枚举和 `-B` 后 PASS |

本阶段额外发现并保留两类基础设施失败：首次完整回归的 LL8 core/provenance
冻结未识别本阶段修复，以及最终 fresh 回归中 `.pytest_cache/lastfailed` 自动更新。
前者用精确 projection 修正，后者仅排除 pytest runtime cache，未删缓存、未改旧
golden、未降低容差。`final-targeted.log` 另保留一次测试投影误读二进制文件的错误；
`final-targeted-v2.log` 为修正后的 19/19。隔离模块最后 3/3 recheck 与 fresh
689 方法结果合成最终统计，未声称最后一次整套运行从未发生失败。

36 个 skip 明细见 [skip-inventory.json](loop_linearno_audit/ll9r/skip-inventory.json)：
33 个是 CPU 回归显式 `CUDA_VISIBLE_DEVICES=''`，3 个是缺 `torch_cluster`。
没有新增 skip/xfail；本机实际有 GPU，loop GPU 验证由独立 144 项矩阵提供。

## 环境、命令及证据边界

本机 Python 3.13.9、PyTorch 2.13.0+cu130、PyG 2.3.1，Intel Core Ultra 9
290HX Plus，RTX 5090 Laptop GPU（capability 12.0）。未安装或变更依赖。
FP32/AMP 合成检查禁用 TF32、cuDNN benchmark 和 compile；GPU 显存采样前后
同步。CPU 48 组固定单线程、CPU0、warmup 3、测量 12 次，用
`perf_counter_ns` 记录 forward 与 forward/backward/AdamW 的 median/p90。
非独占宿主，负载与全部原始样本留在 [cpu.json](loop_linearno_audit/ll9r/cpu.json)。

96 项为全 profile 参数构造/解析计数；48 项 forward/backward/reload 和 CPU 计时
采用真实空间 N、d8/h2/M8/ref3；144 项 CUDA 使用小空间合成输入。三者的规模
不同，不能据此声称全宽 GPU 或真实 epoch 效率验收。

以下是本阶段执行的命令入口；证据已存在，不应覆盖历史文件再次运行。
实际子进程 argv、环境和结果保存在各 JSON/log 中。

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=tests:.
python -B tools/linearno_loop_performance.py counts --output docs/loop_linearno_audit/ll9r/counts.json
CUDA_VISIBLE_DEVICES='' python -B tools/linearno_loop_performance.py matrix --diagnostics --output docs/loop_linearno_audit/ll9r/matrix.json
CUDA_VISIBLE_DEVICES='' taskset -c 0 python -B tools/linearno_loop_performance.py cpu --warmup 3 --steps 12 --output docs/loop_linearno_audit/ll9r/cpu.json
CUDA_VISIBLE_DEVICES=0 python -B tools/linearno_loop_performance.py cuda --output docs/loop_linearno_audit/ll9r/cuda.json
LL9_STANDARD_PRESETS=p2_c2_r2_s2 python -B docs/loop_linearno_audit/ll9r/run_native_matrix.py
LL9_PRESETS=p1_c3_r2_s1 python -B docs/loop_linearno_audit/ll9r/run_industrial.py
python -B docs/loop_linearno_audit/ll9r/replay_old_archives.py
LL9R_REGRESSION_DIR=docs/loop_linearno_audit/ll9r/final-regression python -B docs/loop_linearno_audit/ll9r/run_regressions.py
CUDA_VISIBLE_DEVICES='' python -B -m unittest loop_linearno.test_ll9r loop_linearno.test_performance_tools loop_linearno.test_launchers.LauncherTests.test_gpu_mapping_shell_syntax_and_no_legacy_edits -v
CUDA_VISIBLE_DEVICES='' python -B -m unittest loop_linearno.test_isolation -v
git diff --check
git status --short --branch --untracked-files=all
```

确定性前后捕获使用 `capture_numeric.py before/after --deterministic`；before 从
快照通过 `LL9R_FROZEN_SOURCE` 导入修改前模型，不能把当前模型当作 before。
前后逐张量档案已保存在快照的 `numeric-deterministic/`，192 项记录包括各 dtype
轨迹；额外 `review_nondeterminism.py` 用未修改 SR 源码重现 backend 舍入。

冻结自审确认共享 AttnRes/body/wrappers、三套纯 LinearNO、其他模型数学、schema、
checkpoint、launcher、monitor、任务科学代码和三项旧文档均与本阶段起点相同。
唯一数值代码改动仅 RB helper。无新增需用户裁定的设计冲突。

## Review points

1. [core.py](../cdlno/linearno_loop/core.py) confines dtype normalization to RB
   receiver boundaries and leaves raw cache/state-dict behavior intact.
2. [test_ll9r.py](../tests/loop_linearno/test_ll9r.py) checks local casts,
   autograd, cache preservation, legacy missing-package behavior, and mutation
   sensitive AST/source projections.
3. [precision-compatibility.json](loop_linearno_audit/ll9r/precision-compatibility.json)
   distinguishes the 172 exact pre-existing runs from the 20 repaired AMP runs.
4. [old-checkpoint-replay.json](loop_linearno_audit/ll9r/old-checkpoint-replay.json)
   records exact strict reload and fresh-process replay without modifying old
   artifacts.
5. [end-freeze.json](loop_linearno_audit/ll9r/end-freeze.json) records all
   start-snapshot differences and the final working-tree classification.

**本 LL9R 阶段结束，未执行下一阶段。**
