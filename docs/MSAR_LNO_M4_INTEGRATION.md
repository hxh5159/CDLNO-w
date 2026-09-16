# MSAR-LNO M4：模型选择、显式训练目标与 checkpoint 基础

2026-09-16。本轮只执行 M4。依据已接受的 M0 实际映射、M1 配置、M2 原语和 M3 core，不开展 M5—M7 任务适配。本轮开始为 `main@9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`，原有23处 tracked 修改及未跟踪阶段文件均保留；[起点](msar_lno_audit/m4/before.json)及[用户原diff](msar_lno_audit/m4/user-before.patch)可核查。

## A. 完成范围与公共接入表

M4 实现真实 core 的选择、参数转发、显式 loss adapter，以及保留三种实际 payload 格式的严格 core checkpoint。**没有八任务 MSAR 生产训练入口；当前 factory 的 MSAR Model 接收已经提升的 E0[B,N,d]，不是原任务 x/fx/时间/PyG wrapper。** 不能直接给现有 `exp_*.py` 加 `--model msar_lno` 宣称能训练；任务参数解析、lift、原loss/normalizer及循环接线仍属于后续阶段。

| 部分 | M4实际路径和完成状态 | 后续依赖 |
|---|---|---|
| 公共family | `registry.core_class('msar_lno') → cdlno.msar_lno.core.MSARLNO`，其他key返回None；懒导入，旧配置import不新增torch依赖 | 工业选择分支后续使用此独立家族 |
| 真实PDE factory | `model_dict.get_model(args) → model.MSAR_LNO.Model`，Model是现有真实core的薄导出 | 四静态wrapper M5；两时间wrapper M6 |
| 显式参数 | `entry.core_model_kwargs`仅msar_lno返回config/training_config/output_dim；旧key返回空dict且不解析新参数 | 生产parser仍未接MSAR；沿M1副本parser/explicit-only协议 |
| 训练aux/目标 | `objective.training_forward/training_objective`按单次forward显式组合；四项日志通过`MSARLoss.log_values()`交给调用者 | 实际训练日志/epoch聚合及loss调用位置 M5—M7 |
| 评估 | core默认Tensor；checkpoint重建后eval；不计算coverage，不改变prediction通道 | 原任务decode/导出/物理指标接线 M5—M7 |
| 保存加载 | `checkpoint.save_core_checkpoint/load_core_checkpoint`；state_dict、可信整对象、可信列表/单成员均实际往返 | 任务字段/normalizer/split协议及完整wrapper metadata M5—M7 |
| 目录 | 复用M1 `new_run_path/reserve_run_directory`，family/profile/coverage标识、显式save_name、exclusive保护 | 任务Run/统一Experiment记录尚未接线 |

| 任务 | 当前生产MSAR train/eval | M4已有的保存协议基础 | 待授权阶段 |
|---|---|---|---|
| Darcy | 未接入 | bare state_dict，`model.pt` | M5 |
| Elasticity | 未接入 | bare state_dict，`model.pt` | M5 |
| Airfoil | 未接入 | bare state_dict，`model.pt` | M5 |
| Pipe | 未接入 | bare state_dict，`model.pt` | M5 |
| NS | 未接入 | bare state_dict；未实现时间loss聚合 | M6 |
| Plasticity | 未接入 | bare state_dict；未实现逐时间点调用 | M6 |
| ShapeNet-Car | 未接入 | 整对象，调用方明确给`model_<epochs>.pth` | M7 |
| AirfRANS | 未接入 | run根列表`msar_lno`；单成员`model` | M7 |

表中的checkpoint通过是**真实MSAR core的协议检查**，不是八任务wrapper或真实PyG训练验收。工业main没有统一factory，M4没有另造全仓库框架，也没有改它们的选择/采样/训练代码。

## B. 文件与 diff

新增生产文件：

- `PDE-Solving-StandardBenchmark/model/MSAR_LNO.py`：真实core薄导出，无占位构造器；pickle仍指向`cdlno.msar_lno.core.MSARLNO`。
- `cdlno/msar_lno/entry.py`：仅新家族配置转发，无旧默认值渗入。
- `cdlno/msar_lno/objective.py`：每次forward独立的aux、loss和日志合同。
- `cdlno/msar_lno/checkpoint.py`：M4 core专用快照/加载，完整结构和接口校验。

修改生产文件只有 `cdlno/msar_lno/registry.py`（增懒加载真实类）及 `PDE-Solving-StandardBenchmark/model_dict.py`（3行独立分支）。M1结构/训练配置与解析规则、M2/M3数学代码和所有旧模型原语未改。

新增 `tests/test_msar_integration.py`、[安全构造示例](msar_lno_audit/m4/construct_example.py)、本报告/证据；增量更新独立STATUS、既有CDLNO STATUS与memory。`tests/test_static_standard.py`增加6行：先断言新增factory分支**整个AST精确等于指定3行**，再移除此分支核对旧factory。没有扩大忽略规则，也没有放宽旧exp的数据/训练AST比较。

[freeze](msar_lno_audit/m4/freeze.json)与[本轮patch](msar_lno_audit/m4/stage.patch)相对于真实M4起点，而不是将此前用户diff计作本轮改动。快照位于 `/home/hwz/CDLNO-artifacts/msar-m4-before-i75l1rvp/source`。没有修改 `AGENTS.md`、依赖、旧保存格式、任务数据和训练/评估脚本。

## C. 公式、参数与调用合同

```python
forward = training_forward(msar_model, e0)       # only msar_lno, model.train()
LPDE = original_task_loss(forward.prediction, target)  # 原任务先完成decode/mask等
loss = training_objective(LPDE, forward)
loss.total.backward()
metrics = loss.log_values()  # pde / coverage_raw / coverage_weighted / total
```

| 公式/行为 | 实际实现与证据 |
|---|---|
| `Lcoverage = mean(raw_l, l=1..4)` | M3 `MSARAuxOutput.coverage_mean`；M4直接消费，不重复除batch/N/层 |
| `Ltotal = LPDE + weight * Lcoverage` | `training_objective`；原relative-L2 loss真实导入后连接、反传和AdamW step通过 |
| off/weight0 | `training_forward`默认不请求aux；Down无A请求；`loss.total is LPDE`，raw/weighted为无图FP32零 |
| four logs | `MSARLoss.log_values`返回四个detach标量tensor；不隐式CPU同步。示例明确输出四值，生产epoch日志后续接线 |
| 线程/重入 | frozen per-call dataclass保存该次prediction/aux/config；无model.last_loss，不切换model.training、不改配置；两个不同目标调用交错后独立loss/梯度通过 |
| eval | 直接`model.eval()(...)`返回Tensor；on/off同权重预测一致。加载器不使用训练aux或改变decode |

诊断默认关闭。若用户显式请求diagnostics，则沿M3的no-grad观察路径；它不参与默认loss。M4没有引入新的辅助损失、参数或数学层。时间任务未来必须按原有loss尺度聚合每次forward的raw coverage；**本模块没有提前决定NS/Plasticity的时间归一或optimizer节奏**。

## D. Checkpoint 边界

每个新快照保留三类信息：

1. `architecture.json`：沿用M1 schema1，显式family=msar_lno、architecture_version、完整resolved四级结构、profile provenance、独立training及runtime。
2. `core.json`：schema1、固定class路径、`input_contract=already-lifted-B-N-d-v1`、output_dim、member数、实际weight dtype、payload文件名。这个明确边界防止未来误把core当成已封装原始输入的task wrapper。
3. 原协议payload：PDE裸state_dict；Car原样整对象；Air列表或单对象。没有将它们全改为dict envelope；加载后也不把整对象静默改成新随机模型。

加载先读并严格检查sidecar与显式架构，再构造/反序列化。缺family、缺结构字段、错误family/heads/M/d/output、缺key、多key、shape错误、整对象隐藏heads变化/参数共享均拒绝；`strict=True`。coverage显式变化仅出现在`resolution.training_differences`，不拒绝同结构纯eval；保存的训练配置不被覆盖。初始化仅用于state模型重建或整对象校验的参考实例；同一权重严格装载，已学习fusion.w不会重置。构造参考时保留调用者RNG。

PDE始终`weights_only=True`；whole/list只在已确认本地可信工业run的调用方显式`trusted_local=True`时走局部`weights_only=False`。沿用已有边界，不改全局torch行为、不加fallback。整对象继续复用已有`kcdno.loading.validate_whole_model`的module/type/行为/固定buffer/严格state校验，不修改该工具。两张sidecar在成功eval与负向检查后逐字节不变。

`save_core_checkpoint`只接受新目录；它是独立完整core快照基础，**不是周期保存策略或任务Run，更不是optimizer/scheduler/RNG续训包**。当前八任务没有接入V1共享resume archive；M4不新建resume。不承诺用裸权重/整对象恢复优化器。后续若接入已有完整archive，必须额外校验训练配置并保留其状态协议。

## E. 实际验证

本机：Python3.13.9 / torch2.13.0+cu130 / CUDA13.0 / RTX5090 Laptop / PyG2.3.1；未安装或更改依赖。远端Python3.10 / torch2.11/cu128未运行。[环境](msar_lno_audit/m4/environment.json)，[汇总](msar_lno_audit/m4/summary.json)，[回归索引](msar_lno_audit/m4/fixture-index.json)。

| 验证 | 实际结果 |
|---|---|
| 最终M4定向测试 | 18/18，8.539s，无失败/错误/skip；[log](msar_lno_audit/m4/integration-tests-final.log) |
| M1/M2/M3既有测试 | 49/49；本轮首次MSAR合并套件66/66中运行，之后新增元数据负向项，最终M4单独重跑18项；[log](msar_lno_audit/m4/msar-tests.log) |
| 旧PDE/时间/CDLNO/K任务入口测试 | 33/33，31.286s，无失败/错误/skip；包含旧factory/CLI/AST/strict保存加载；[log](msar_lno_audit/m4/old-entry-tests.log) |
| M0所有既有同权重夹具 | 41旧K0 +6 pre-M1 K core +24任务wrapper +4其他旧模型 =75/75；所有CPU FP32 MATH、atol=rtol=0；182旧夹具hash未变 |
| 原有GUNet例外 | 独立旧模型审计仍明确not_run：本机缺torch_cluster，未执行radius/nearest图链；不计入75份已保存夹具 |
| M3正式Light/Full core | 两份同权重输入、eval、train-floor及coverage重新精确回放；不是重新seed随机初始化互比 |
| M4真实loss连接 | 安全导入原`utils.testloss.TestLoss(size_average=False)`的relative-L2，合成lifted张量上backward/step通过；没有跑Darcy完整decode/边界/梯度loss或其他任务loop |
| CUDA | 新小core FP32 MATH显式loss/AdamW/checkpoint通过，roundtrip容差1e-6/1e-5；本轮M2/M3套件也实际重跑有限FP16/BF16 AMP检查 |
| 三子项目cwd | 各新进程同权重core strict加载通过；Car whole、Air list明确trusted本地；没有直接import exp/main |
| 真实PyG MSAR wrapper | 未实现/未运行；旧fixture中的真实PyG证据不冒充新家族支持 |
| 真实数据训练/评估/指标/收敛 | 全部未运行；没有下载/制造任务数据 |

共覆盖100个不同测试方法（49旧MSAR基础+18最终M4+33旧入口），不是声称一次跑了100项全仓库套件。首次18项之前的M4首轮16项及合并17项也通过，日志保留；没有生产故障被隐藏。基础checkpoint验证只在例外分支又补了缺对象属性/非法dtype的清晰报错，并添加最终元数据负向检查。

实际命令（仓库根目录，输出重定向至上述log）：

```bash
python -B -m unittest discover -s tests -p 'test_msar*.py' -v
PYTHONPATH=tests:. python -B -m unittest test_static_standard test_temporal_standard test_kcdno_tasks test_kcdno_temporal -v
python -B -m unittest discover -s tests -p test_msar_integration.py -v
python -B docs/kcdno_audit/make_regression_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/msar_lno_audit/m4/k0-replay.json
python -B docs/msar_lno_audit/m1/replay_current.py replay --artifacts /home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/current-fixtures --result docs/msar_lno_audit/m4/kcore-replay.json
python -B docs/msar_lno_audit/m0/wrapper_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers --result docs/msar_lno_audit/m4/wrapper-replay.json
python -B docs/msar_lno_audit/m0/other_model_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/other-models --result docs/msar_lno_audit/m4/other-replay.json
python -B docs/msar_lno_audit/m3/core_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/msar-m3-before-jx4v53vh/post-m3-core --result docs/msar_lno_audit/m4/msar-core-replay.json
```

这些绝对artifacts路径为本机已保存证据；远端回放需迁移原夹具并替换路径，不能重新生成所谓旧权重。

## F. 可运行构造示例与自审

从实际仓库根目录运行，只有合成lifted数据、小N和一次优化步，默认正式Light，所有四级M不缩减：

```bash
python -B docs/msar_lno_audit/m4/construct_example.py
python -B docs/msar_lno_audit/m4/construct_example.py --coverage-mode off --coverage-weight 0
# 可选保存至独立测试区域；自动生成唯一family/profile/coverage目录
python -B docs/msar_lno_audit/m4/construct_example.py --output-root /tmp/msar-m4-check
```

[Light示例实际输出](msar_lno_audit/m4/example-light.json)含同权重保存加载，checkpoint保存在外部M4 artifacts；[off示例](msar_lno_audit/m4/example-off.json)四项日志raw/weighted严格为零。脚本使用真实`model_dict.get_model(...).Model(**core_model_kwargs(...))`，不是虚构统一train.py。不附八任务训练命令，因为对应生产入口尚未接通。

已逐项自审：

- 旧分支隔离：新参数仅进入新family；旧factory新增分支移除后精确等于起点，旧模型数值夹具精确回放。
- loss归属：原loss先计算，新家族单独加权；off对象同一性、四项日志、交错forward梯度均通过；无隐式跨forward缓存。
- 加载真实性：保存完整resolved结构和core接口；strict keys/shapes、whole对象隐含属性、已学习w不重置、可信边界与sidecar不改写均通过。
- 阶段边界：M1/M2/M3数学、所有数据/训练/评估脚本、旧输出记录代码保持；新core通过不等于八任务wrapper通过。
- 证据边界：旧夹具是既有固定合成参考，不证明所有历史真实训练checkpoint兼容；远端环境、MSAR PyG/task resume/真实数据需后续验证。

没有发现需用户裁定的新架构冲突。余项是既定M5静态、M6时间、M7工业接入及后续验收，不在本轮补做。

**本M阶段结束，未执行下一阶段。**
