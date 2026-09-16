# MSAR-LNO M8：八任务综合可执行验收

2026-09-17，仅M8；M7已批准。起点为`main@9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`及全部既有未提交工作。按[M0实际映射](MSAR_LNO_REFERENCE_AUDIT.md)复用[M5](MSAR_LNO_M5_STATIC_TASKS.md)、[M6](MSAR_LNO_M6_TEMPORAL_TASKS.md)、[M7](MSAR_LNO_M7_INDUSTRIAL_TASKS.md)的真实parser/factory、wrapper、原loss与安全测试片段。本轮没有生产代码修改，没有下载数据、安装依赖、真实训练、commit/push或M9工作。

## A. 覆盖矩阵与证据边界

以下“通过”均指对应的静态或**随机合成输入**检查。正式profile未缩小d/M/depth：Light为d96、M=[512,256,128,64]；Full为d192、M=[1024,512,256,128]；heads=[4,4,8,8]、encoder/decoder depths=[3,1,1,1]。完整逐例配置、实测参数量、objective值和耗时保存在[formal-cases.json](msar_lno_audit/m8/formal-cases.json)；[机器可读覆盖表](msar_lno_audit/m8/coverage-matrix.json)明确区分未执行项。

表中checkpoint的“任务协议”来自本轮重跑的M5—M7小结构d8/M=[7,5,3,2]；正式Full的纯state_dict另逐任务实际往返。**没有将小结构任务checkpoint测试写成正式Light整模型checkpoint验收。**

| 任务 | 静态参数流 | 正式Light合成forward（off/floor） | 原loss连接 | backward / optimizer | eval | checkpoint | 真实PyG | GPU | 真实数据 |
|---|---|---|---|---|---|---|---|---|---|
| Darcy | 通过 | B2、7×7、两模式通过 | 原decode、相对L2+0.1导数L2通过 | 两模式通过 | 原decode/相对L2通过 | 原state_dict协议；Full strict通过 | 不适用 | d8原loss步通过 | 未执行 |
| Elasticity | 通过 | B2、N35、两模式通过 | 原decode/相对L2通过 | 两模式通过 | 原指标通过 | 同上 | 不适用 | d8原loss步通过 | 未执行 |
| Airfoil | 通过 | B2、5×7、两模式通过 | 原相对L2通过 | 两模式通过 | 原指标通过 | 同上 | 不适用 | d8原loss步通过 | 未执行 |
| Pipe | 通过 | B2、5×7、两模式通过 | 原坐标normalizer、decode/相对L2通过 | 两模式通过 | 原指标通过 | 同上 | 不适用 | d8原loss步通过 | 未执行 |
| NS | 通过 | B1、N4096、fx10→out1、两模式通过 | 原10步真值回填求和loss通过 | 两模式各10 forward→1次更新通过 | 10步预测回填通过 | 原state_dict协议；Full strict通过 | 不适用 | d8原10步更新通过 | 未执行 |
| Plasticity | 通过 | B1、N3131、fx1/T[B,1]→out4、两模式通过 | 原20个时间点loss通过 | 两模式各20次更新、batch末1 scheduler通过 | 原20时间点通过 | 同上 | 不适用 | d8原20步更新通过 | 未执行 |
| ShapeNet-Car | 通过 | 真Data/Batch tuple、单图N19、两模式通过 | 原全点速度MSE+reg×表面压力MSE通过 | 两模式通过 | 原`train.test`通过 | 整对象任务协议；Full strict通过 | 通过，单图/可变N/多图拒绝 | d8真PyG原loss步通过 | 未执行 |
| AirfRANS | 通过 | 真Data/Batch、单图N19、两模式通过 | 原入口`MSE_weighted`体积+reg×表面通过 | 两模式通过 | 原MSE test通过 | 整成员/列表任务协议；Full strict通过 | 通过，含原sampled ptr/scatter片段 | d8真PyG原loss步通过 | 未执行 |

GPU栏为本机FP32/MATH、缩小d/M的真实模型合成训练步，**不是正式Light/Full GPU矩阵**。新正式profile矩阵使用CPU FP32/MATH、TF32关闭、dropout不存在。每任务保存了实际resolved结构。工业loss直接调用可安全导入的原`train.train/test`；六PDE执行原训练batch与eval语句的AST，只替换合成输入/设备传输，不import顶层读取数据的exp/main。没有另外编写近似loss冒充原链路。

正式Full八任务都完成实际参数构造、单次forward、**明确标注的合成MSE+coverage backward**、eval及同结构严格state_dict保存/加载，输出零容差相等；另检查缺key拒绝。Full没有运行原loss完整时间循环/optimizer矩阵。NS/Plasticity wrapper固定原网格，因此没有为了减小N改生产合同，而是保留真实空间N、减为B1/单次forward；其余任务减小N。

| 任务 | Light实际参数量 | Full实际参数量 | Full测试N / 输出shape | Full M1>N |
|---|---:|---:|---|---|
| Darcy | 1,722,169 | 6,835,825 | 49 / [1,49,1] | 是 |
| Elasticity | 1,710,169 | 6,811,825 | 35 / [1,35,1] | 是 |
| Airfoil | 1,710,169 | 6,811,825 | 35 / [1,35,1] | 是 |
| Pipe | 1,710,169 | 6,811,825 | 35 / [1,35,1] | 是 |
| NS | 1,723,897 | 6,839,281 | 4096 / [1,4096,1] | 否 |
| Plasticity | 1,729,180 | 6,886,708 | 3131 / [1,3131,4] | 否 |
| Car | 1,711,420 | 6,814,324 | 19 / [19,4] | 是 |
| AirfRANS | 1,723,708 | 6,838,900 | 19 / [19,4] | 是 |

本轮还重跑M5的正式N布局检查：Darcy7225、Elasticity972、Airfoil11271、Pipe16641，使用d8，不是满宽训练；正式Full Elasticity N972前向也重跑。Darcy原导数loss要求方网格，原loss步用7×7，独立wrapper继续以5×7验证没有sqrt(N)依赖。Car/Air新正式profile为N19，M7 N11/23变长/排列/单图边界也重跑；没有声称真实完整工业网格训练通过。

## B. 公式、执行计数与针对性边界

| 合同 | 实际实现/测试映射 | 本轮结果 |
|---|---|---|
| 同结构同权重off/floor | `tests/test_msar_acceptance.py::FormalTaskAcceptance.run_light`显式复制同一state；`training_forward`只选aux路径 | 八任务两模式预测最大绝对差2.2351741790771484e-7；atol2e-6/rtol1e-4通过 |
| off或weight0 | `objective.training_objective`返回同一个LPDE对象；M2测试禁止显式Tensor.softmax并确认Down走SDPA；M4测试Down不请求aux | 不计算coverage、不创建额外coverage A/图；eval始终只预测。此处不对SDPA后端内部临时存储作性能推断 |
| floor | `CoverageFloorLoss`：head平均、latent平均、有效source求和、batch均值；core四层均值；`total=PDE+weight*raw` | FP32 raw有限；加权只一次。默认kappa=.2的本次Light随机输入均满足floor，raw=0，并非没有调用coverage |
| 有效coverage梯度 | `AblationClosure.test_boundary_aux_isolation_and_coverage_dependency`；复用M2独立oracle | core四级Down的query/key梯度均连接且有限；kappa1下逐个真实Down输入变化的source，验证source/query/key均有非零梯度。无decoder/Up/AttnRes/head的aux梯度 |
| 时间聚合 | `rollout_objective`与原NS/Plasticity时间片段 | NS：Σ10 LPDE + weight×mean10(mean4 raw)；Plasticity：每时间更新LPDE+weight×mean4 raw。原更新/回填/调度不变 |
| 三个AttnRes | `PairwiseAttnResFusion`；`test_three_fusions_source_axis_token_specific_weights_and_gradients`逐个独立公式核对 | 每个只有w；w0时exact E+U、E/U直接Jacobian均1、w有效梯度；非零w时alpha[B,M,2]沿source softmax、每token不同、打乱U改变结果；无gamma/projection/history |
| 完整执行拓扑 | `test_formal_light_execution_counts_slot_fusion_and_storage`的真实hook | Down4、Up4、PairFusion3、latent blocks12、SA12（512×6/256×2/128×2/64×2）、FFN24；N35处没有SA；无Conv/CDPA/kernel/history/bridge；参数对象/storage独立 |
| aux/diagnostics生命周期 | 新边界测试+M3/M4 interleaved/cross-call测试 | B1/2、N3/11/35、无mask/部分mask、空mask拒绝、kappa0/weight0/off通过；完成一次backward后再次独立forward/backward通过，无前次输入梯度连接或module-owned统计；diagnostics无梯度且只返回小统计 |
| 非法结构与mask | 本轮重跑`test_msar_config`、`test_msar_modules`、`test_msar_core` | 非法长度/宽度/heads/depth/family/目标字段/形状/mask明确拒绝；M1>N不截断；工业多图拒绝 |

[边界数值证据](msar_lno_audit/m8/boundary-cases.json)保留各层raw及query/key梯度最大值。首轮新边界测试错误地要求每层每次都非零；最深层在初始化下slot相同、raw=0，合法零梯度触发该断言。修正仅涉及本轮新测试：先检查所有层图连接/有限，再用每个真实Down的变化source检查有效梯度。**没有改模型、权重初始化、coverage公式/默认或数值容差来通过测试。** 初始失败日志保留。

## C. Checkpoint、原cwd与旧模型回归

M5—M7的真实task parser、factory、train/eval选择、sidecar先读后校验、coverage纯eval覆盖、重复eval文件只读及错误family/shape/key拒绝全部重跑。原PDE、Car、Air三个工作目录启动新Python进程：标准/时间模型strict加载、工业稳定`cdlno.msar_lno.industrial.{CarModel,AirfRANSModel}`整对象/成员列表加载均通过；没有import exp/main来取接口。新MSAR类未侵入旧pickle路径。

保存协议不变：六PDE裸state_dict；Car整模型；Air成员整对象及模型列表。本轮Full纯state_dict往返是独立数值验证，不替代工业生产整对象协议，也不新增resume。这些任务权重/整模型仍不是完整optimizer/scheduler/RNG续训档案。

[M0夹具索引及本轮回放](msar_lno_audit/m8/fixture-index.json)：**75份使用原保存权重和原固定输入的夹具精确通过，atol=rtol=0**，182个原夹具文件SHA256不变。

| 夹具组 | 数量 | 实际范围 |
|---|---:|---|
| K0基准 | 41 | 原Transolver、CDLNO full/no_sa/identity，点/卷积/时间/工业，含原checkpoint和必要梯度检查 |
| M0任务wrapper | 24 | 八任务KCDNO all/off、matched LRSA的同权重输出/类路径/状态schema/协议 |
| pre-M1核心 | 6 | KCDNO all/off、matched LRSA，point/conv |
| M0其他现有模型 | 4 | Air MLP/PointNet/GraphSAGE、标准3D Transolver |

GUNet原夹具因缺torch_cluster未生成，本轮仍明确未执行，未用新随机权重替代。夹具是历史代码上构造的合成参考，不能证明所有外部历史训练checkpoint兼容。旧CLI defaults、原loss、旧factory不接收新kwargs由旧157方法及MSAR107方法中的冻结/选择测试验证。本轮全部既有生产及测试文件字节保持，是相对于已接受M7工作区的增量结论；旧阶段更早的冻结依据另保留。

## D. 命令、环境与实际结果

环境：[environment.json](msar_lno_audit/m8/environment.json)。Python3.13.9、Torch2.13.0+cu130、CUDA13.0、PyG2.3.1、RTX5090 Laptop；torch_cluster缺失。没有安装或变更依赖。这不是远端Python3.10/Torch2.11/CUDA12.8的运行证据。

| 实际执行 | 结果 | 耗时 |
|---|---|---:|
| 新M8套件首轮 | 19方法：18通过；1个上述测试断言失败 | 97.619s |
| 修正后的边界方法定向重跑 | 1/1通过；新19方法无遗留失败（分两次，未冒充单次全绿） | 0.122s |
| 既有M1—M7 MSAR套件 | 107方法，106通过、1跳过：Air完整抽样epoch缺torch_cluster | 126.930s |
| 旧模型/任务/记录套件 | 157方法，0失败/错误；1个Air抽样epoch子用例跳过 | 181.846s |
| 41+24+6+4旧夹具 | 75/75精确；GUNet未执行单列 | 35.16+14.89+3.13+10.08s |

耗时是本机测试进程/审计用时，部分独立进程并行执行，不能相加当总墙钟时间或解读为模型benchmark。M2/M3 GPU FP16/BF16原语/core检查随套件重跑通过；八任务仅缩小结构FP32原loss GPU通过，不等于正式profile或任务AMP全链验收。

在仓库根目录实际执行的主要命令如下，完整日志及结果见[证据目录](msar_lno_audit/m8/)与[summary](msar_lno_audit/m8/summary.json)：

```bash
MSAR_M8_RESULTS=docs/msar_lno_audit/m8/formal-cases.json \
PYTHONPATH=tests:. python -B -m unittest test_msar_acceptance -v \
  > docs/msar_lno_audit/m8/formal-tests-initial.log 2>&1

MSAR_M8_RESULTS=docs/msar_lno_audit/m8/boundary-cases.json \
PYTHONPATH=tests:. python -B -m unittest \
  test_msar_acceptance.AblationClosure.test_boundary_aux_isolation_and_coverage_dependency -v \
  > docs/msar_lno_audit/m8/boundary-correction.log 2>&1

PYTHONPATH=tests:. python -B -m unittest \
  test_msar_config test_msar_modules test_msar_core test_msar_integration \
  test_msar_static test_msar_temporal test_msar_industrial -v \
  > docs/msar_lno_audit/m8/accepted-msar-tests.log 2>&1

PYTHONPATH=tests:. python -B -m unittest \
  test_static_standard test_temporal_standard test_shapenet_car test_airfrans \
  test_front_ablation test_front_task_modes test_front_training \
  test_kcdno_config test_kcdno_tasks test_kcdno_temporal test_kcdno_car \
  test_kcdno_airfrans test_kcdno_matched test_kcdno_delivery \
  test_experiment_records -v > docs/msar_lno_audit/m8/legacy-tests.log 2>&1

python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result docs/msar_lno_audit/m8/k0-replay.json
python -B docs/msar_lno_audit/m0/wrapper_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers \
  --result docs/msar_lno_audit/m8/wrapper-replay.json
python -B docs/msar_lno_audit/m1/replay_current.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/current-fixtures \
  --result docs/msar_lno_audit/m8/old-core-replay.json
python -B docs/msar_lno_audit/m0/other_model_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/other-models \
  --result docs/msar_lno_audit/m8/other-replay.json
python -B docs/msar_lno_audit/m8/finalize_evidence.py
```

四条夹具replay实际外层使用`/usr/bin/time -p`并分别重定向至`k0-replay.log`、`wrappers-replay.log`、`core-replay.log`、`other-replay.log`，工具内部记录各项目新进程的command/cwd。不要覆盖本轮证据以“修正”历史失败；重跑请使用新的输出路径。

远端补充检查（以下**本轮未在远端执行**）：在远端实际checkout根目录运行，不需要本机绝对路径，不会加载真实数据；先保留既有能运行Car的环境。正式profile测试为合成单batch/短时间协议，工业单图仍小N。

```bash
python -B - <<'PY'
import sys, importlib.util, torch
import torch_geometric
print(sys.version, torch.__version__, torch.version.cuda, torch_geometric.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
print('torch_cluster:', importlib.util.find_spec('torch_cluster') is not None)
PY

# 独立新目录保护已有结果。GPU不可用会标记跳过，不是GPU通过。
msar_check_dir=$(mktemp -d "${TMPDIR:-/tmp}/msar-m8-remote.XXXXXX")
MSAR_M8_DEVICE=cuda MSAR_M8_RESULTS="$msar_check_dir/formal-cases.json" \
PYTHONPATH=tests:. python -B -m unittest test_msar_acceptance -v \
  > "$msar_check_dir/acceptance.log" 2>&1

# 安全入口、三项目cwd checkpoint；最后一项有torch_cluster才运行原合成抽样epoch。
PYTHONPATH=tests:. python -B -m unittest -v \
  test_msar_static.MSARStaticTests.test_all_eval_shell_modes_read_saved_structure_and_fresh_cwd_load \
  test_msar_temporal.MSARTemporalTests.test_shell_modes_fresh_cwd_loading_and_readonly_records \
  test_msar_industrial.IndustrialMSAR.test_checkpoint_whole_list_member_readfirst_and_fresh_cwd \
  test_msar_industrial.IndustrialMSAR.test_air_original_sampled_epoch_when_extension_available
```

完整历史冻结套件/75份旧夹具回放需要另行复制原索引指向的**同一份外置快照及权重**；仅复制代码不足，不应现场重新随机初始化来代替旧基准。上面远端定向命令不依赖这些外置夹具。实际数据训练/评估脚本仍见[八任务现有命令](../tran_evaluate/msar_lno/README.md)，本轮没有运行。

## E. 增量文件、冻结与自审

新增`tests/test_msar_acceptance.py`：16个正式profile任务方法及3个边界/拓扑方法，复用既有M5—M7 helper；没有建立另一套训练框架。新增本报告与`docs/msar_lno_audit/m8/finalize_evidence.py`及结果证据；三个既有STATUS/memory文件只追加本轮记录。没有修改已有测试逻辑。

起点837文本文件快照`/home/hwz/CDLNO-artifacts/msar-m8-before-fhesnwse/source`和用户原diff保留，[before](msar_lno_audit/m8/before.json)、[preexisting.patch](msar_lno_audit/m8/preexisting.patch)、[阶段diff](msar_lno_audit/m8/stage.patch)、[freeze](msar_lno_audit/m8/freeze.json)分开记录。834个起点文件字节相同，只有上述3个状态文档变化；生产模型/配置/factory/CLI/数据/采样/loss/时间循环/optimizer/scheduler/指标/输出/依赖全部不改。新测试权重仅临时目录保存，不自动提交二进制或覆盖实验结果。

已自审五项：①正式profile与小结构证据分开，Full没有虚构原loss训练；②四Down coverage梯度、合法零梯度及无decoder辅助项；③三PairFusion公式/slot/source轴和执行计数；④新cwd严格checkpoint/先读配置/只读sidecar；⑤旧同权重、文件hash、CLI/loss与用户diff保持。没有发现本轮引入的生产回归或需要用户裁定的架构冲突。

剩余限制：远端2.11/cu128、正式profile GPU/任务AMP/backend/compile矩阵、Full原loss完整更新矩阵、真实数据读取/训练/收敛/准确率均未执行。Air完整抽样epoch/radius_graph/VTK/forces受原torch_cluster缺项限制；Car实际drag/原固定路径fold0及既有日志缺陷保留。M8结果不保证实际数据集精度、显存或速度，也不扩大checkpoint续训能力。

**本M阶段结束，未执行下一阶段。**
