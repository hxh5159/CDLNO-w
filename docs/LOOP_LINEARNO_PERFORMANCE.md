# LL9 — Looped LinearNO 无数据综合验证与成本工具

本阶段只增加 `tools/linearno_loop_{support,accounting,diagnostics,performance}.py`、独立测试和本报告/证据。模型、schema、八任务训练/评估入口、launcher、旧 monitor 和历史测试保持原样。所有数据均为内存合成数据，未执行真实 loader、真实 epoch、远端实验或 SOTA 比较。

## 当前结论与精度边界

CPU FP32 的 48 个 task×preset×mode 组合，在标准空间点数、缩小宽度的条件下通过前向、反向、AdamW 一步、全新对象严格 state/optimizer 重载、连续前向、调用顺序及输出头次数检查。96 个正式宽度配置的实际参数分项与独立解析式完全一致。正式宽度只构造和统计；不能把它们的解析 MAC 与小宽度延迟混成同一模型的结果。

新增 CUDA 验收发现**已有 RB 模型的部分 AMP 路径不兼容**：Airfoil、Elasticity、Pipe、AirfRANS、Car 的 FP32 placeholder 将 anchor 提升为 FP32，autocast raw branch 为 FP16/BF16，`PointDepthAttnRes._validate` 拒绝混合 dtype 的 sources。两个 preset × 两种 AMP，共20项失败；其余124项通过，包括全部48个CUDA FP32、38个FP16和38个BF16。这些失败不是OOM，也没有回退成FP32再声称AMP通过。源码与LL9起点相同；见 [定位](loop_linearno_audit/ll9/cuda-failure-review.json)。LL9未擅自修改已冻结的模型精度语义。另有完整旧套件触发的ignored `.pyc` 冻结断言失败（下文说明）。本阶段最终状态据此为 **PARTIAL**，不能声称整个精度矩阵全绿。

## A/B. 交付、命令与输出

从当前checkout根执行，路径随checkout工作，适用于远端 `LinearNO-monitor`；以下命令只产生合成审计，不启动数据集训练。`--output` 必须是新文件，以免覆盖既有证据。

```bash
python -B tools/linearno_loop_performance.py counts --output output/ll9/counts.json
CUDA_VISIBLE_DEVICES='' python -B tools/linearno_loop_performance.py matrix \
  --diagnostics --output output/ll9/matrix.json
CUDA_VISIBLE_DEVICES='' python -B tools/linearno_loop_performance.py cpu \
  --warmup 3 --steps 12 --output output/ll9/cpu.json
CUDA_VISIBLE_DEVICES=0 python -B tools/linearno_loop_performance.py cuda \
  --output output/ll9/cuda.json
PYTHONPATH=tests:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -B -m unittest loop_linearno.test_performance_tools -v
```

CUDA命令会完成所有精度组合，把失败逐项写入JSON，并以非零状态退出；没有CUDA则明确写NOT RUN。`counts` 给每任务×2preset×3mode×rank1/2完整resolved config及分项；`matrix`保存实际算子、调用顺序、48组重载误差和可选诊断；`cpu`保留每次原始计时而非只给均值；`cuda`记录精度、finite结果和allocator显存。旧训练CLI和结果目录没有新默认行为。

工具实现映射：

| 文件/符号 | 职责 |
|---|---|
| `tools/linearno_loop_support.py` | 原profile派生配置、确定性的合成输入；不import训练入口 |
| `tools/linearno_loop_accounting.py::analytic` | 独立整数解析式；不构造/读取模型 |
| `measured_parameters / audit / Ledger` | 实际named_parameters分区，ATen矩阵/卷积/收缩计数及source softmax轴 |
| `tools/linearno_loop_diagnostics.py::LoopDiagnostics` | 显式context/hook；默认关闭；不保存计算图、不改旧monitor |
| `tools/linearno_loop_performance.py` | 96计数、48合成矩阵、CPU计时、CUDA smoke及显存CLI |
| `tests/loop_linearno/test_performance_tools.py` | 解析对实际运算、真假平方张量、诊断输出/梯度/RNG隔离、异常清理、custom、工具失败传播 |

## C. 解析式与计算量口径

记隐藏宽度为 d，heads=h，head_dim=d/h，rank=M，FFN ratio=f，卷积系数 k=9(conv/conv_temp)否则1，output线性层数 o=2(conv/conv_temp/Car)否则1。这里 d 避免与空间网格 H 混淆。

单个不含最终head的物理block参数：

- 输入投影 `k*d²+d`；跨head共享Q/K/V `2*(d/h)*M+(d/h)²`；output投影 `o*(d²+d)`。
- 温度参数：Standard temp/conv_temp及Car为2h，Air保留h个dead temperature，其余0。
- 两个LayerNorm共4d；FFN `2*f*d²+(f+1)*d`。
- 唯一末层head `2d+d*out_dim+out_dim`。
- stem包括preprocess、placeholder和可选time_fc，按各任务位置语义计算输入通道；time_fc实际对重复到N点的时间embedding执行，MAC按B*N计。

总物理参数 = stem + P×body + C×body + S×body + head + router。报告分别保存 `stem/prefix/shared_core/suffix_body/head/router`；`suffix_body+head` 即suffix/head分项。Air的dead temperature计入存储参数，但不声称有计算或梯度。

- RB receiver数 `2CR+1`，参数 `2d(2CR+1)`。
- LB receiver数 `R`，参数 `2dR`。
- SR没有router。P1/C3/R2/S1有5个独立block，P2/C2/R2/S2有6个；两者均执行8次。不能称两preset参数相同。

一次block visit的矩阵MAC：

`B*N*[k*d² + o*d² + 2*f*d² + d*(d/h) + 4*d*M]`。

末项四份分别来自 Q projection、K projection、KᵀV、QC；每份均 `B*N*d*M`。共享小Q/K/V权重并不减少每次visit、每个head的实际计算。总执行MAC为 `stem_MAC+(P+CR+S)*body_MAC+head_MAC`；另列每物理block只访问一次的参考数，不将其冒充forward成本。

Dense matrix FLOPs约定 `2×MAC`，**不是完整FLOPs**：不含bias、residual、RMSNorm/LayerNorm、softmax、GELU/SiLU、温度clamp/divide、位置距离、sin/cos、layout/storage。逐ATen算子调用与输出元素数单独留档，不能简单把元素数当全部标量FLOPs。

router并没有额外mm，但并非零计算：对每个S>1来源receiver，score点积与raw value加权和合计 `2*B*N*d*S` 个收缩MAC等价值；S=1的恒等receiver在生产中跳过。RMS square/mul、scale/query逐元素乘、rsqrt、source softmax等另外记录。实际reduction形状计数与解析式对照。M翻倍会同时增加Q/K投影与KᵀV/QC，**不属于compute-matched控制**，更不能把独立参数减少写成FLOPs或真实epoch速度降低。

| Task | base→M×2 | SR参数 P1 / P2 (M×2) | RB新增 P1 / P2 | LB新增 | B1 M×1→M×2矩阵GMAC |
|---|---:|---:|---:|---:|---:|
| airfoil | 64→128 | 1,126,737 / 1,345,249 | 3,328 / 2,304 | 512 | 22.7209→25.6755 |
| darcy | 64→128 | 1,126,993 / 1,345,505 | 3,328 / 2,304 | 512 | 14.5665→16.4605 |
| elasticity | 64→128 | 388,817 / 459,745 | 3,328 / 2,304 | 512 | 0.8128→1.0676 |
| pipe | 64→128 | 1,126,737 / 1,345,249 | 3,328 / 2,304 | 512 | 33.5461→37.9085 |
| ns | 32→64 | 2,192,385 / 2,593,025 | 6,656 / 4,608 | 1,024 | 14.9957→16.0694 |
| plasticity | 64→128 | 1,160,324 / 1,378,820 | 3,328 / 2,304 | 512 | 6.4163→7.2371 |
| airfrans | 32→64 | 2,173,228 / 2,573,876 | 6,656 / 4,608 | 1,024 | 116.5394→124.9280 |
| car | 32→64 | 2,469,460 / 2,935,908 | 6,656 / 4,608 | 1,024 | 133.0368→141.4742 |

上表GMAC不含router收缩与标量运算；两个preset执行深度都为8，同rank的dense MAC相同。工业统计为单个成员，不是Air ensemble总成本。96行逐分区完整数据见 [counts.json](loop_linearno_audit/ll9/counts.json)。

## D. 实测协议

正式参数/MAC：八任务论文profile、seed17、两preset、三mode、rank×1/2，B1，canonical N。ShapeNet/Air单图，Car N32186、Air N32000；Standard使用原网格，Elasticity N972。未运行正式宽度的完整任务训练。

48组综合矩阵和CPU延迟：canonical N，Standard B2、工业B1，**d8/h2/M8/ref3**，输入seed902，模型seed17；使用任务原forward合同和时间输入，但一步loss是明确的合成MSE，不冒充Darcy导数loss、NS十步或Plasticity二十查询的完整训练代价。原任务科学协议闭环单独引用/重跑LL6/LL7。

CPU：单intra/inter-op线程，FP32，`perf_counter_ns`同步计时；3次warmup、12次测量；median与nearest-rank p90；forward为eval/no_grad，训练计时包括zero_grad、forward、合成MSE、backward和AdamW.step，优化器状态已在warmup初始化。diagnostic、hook和profiler在计时期间全部关闭。测量不代表真实epoch，跨任务也不代表同参数或同工作量。

CUDA：本机RTX5090 Laptop，torch2.13+cu130；small N35(Standard)/37(工业)，d8/h2/M8；FP32、autocast FP16+GradScaler、autocast BF16。warmup一步后`synchronize→zero_grad→empty_cache→reset_peak_memory_stats`，再一步和synchronize。报告进程allocator的baseline、peak allocated/reserved与增量，不是nvidia-smi全设备显存、不含其他进程，也不是正式训练峰值。只做synthetic smoke，不测GPU速度或推断AMP精度。初次工具误将第一个参数（某些任务不用的placeholder）不更新当成optimizer跳步；已修正为检查实际有梯度的参数更新，保留初次日志。真正mixed-dtype失败在独立repro中仍存在。

### 实际读数（本机合成工作量）

CPU：Intel Core Ultra 9 290HX Plus，WSL2，affinity=CPU0，torch intra/inter-op=1；启动时系统loadavg=[5.47509765625, 10.40234375, 9.65380859375]。同期仍有旧history回归进程，本机并非独占性能实验室；保留原始样本和写入时load，不以此给真实任务加速结论。下表Darcy为B2/N7225/d8/h2/M8/ref3：

| preset | mode | forward median / p90 ms | forward+backward+AdamW median / p90 ms |
|---|---|---:|---:|
| p1_c3_r2_s1 | sr_1_over_r | 48.680 / 53.870 | 129.969 / 136.080 |
| p1_c3_r2_s1 | rb_attnres | 78.192 / 84.027 | 216.011 / 231.702 |
| p1_c3_r2_s1 | lb_attnres_1_over_r | 53.492 / 61.520 | 153.380 / 168.635 |
| p2_c2_r2_s2 | sr_1_over_r | 53.746 / 58.305 | 144.614 / 152.347 |
| p2_c2_r2_s2 | rb_attnres | 71.219 / 77.153 | 194.168 / 204.752 |
| p2_c2_r2_s2 | lb_attnres_1_over_r | 62.166 / 78.510 | 201.126 / 220.921 |

48组完整计时及576个forward/576个step原始样本见 [cpu.json](loop_linearno_audit/ll9/cpu.json)。这些step不含data loading、任务loss/rollout、scheduler、记录/绘图或checkpoint；不能乘batch数推导真实epoch。

另外实际构造原纯LinearNO八层作参数对照，得到NS 3,377,921、Plasticity 1,799,428等已核验值。96个loop配置为原各任务八层baseline参数的63.23%–78.95%；这是存储参数比较，缺少训练精度证据。详见 [96项对照](loop_linearno_audit/ll9/baseline-comparisons.json)。默认M×2的dense MAC已经高于对应pure8/baseM，router再另加；未命名为compute-matched。

成功CUDA小fixture的绝对peak allocated范围为64.328–64.890 MiB，包含进程中保留的CUDA工作区。逐行baseline/reserved/增量见 [cuda-final-v2.json](loop_linearno_audit/ll9/cuda-final-v2.json)；失败20项没有捏造显存/成功指标。

## E. 可选诊断与结构验证

默认 `LoopDiagnostics(model)` 不挂hook、不分配观测张量、不做CPU同步；只有显式 `enabled=True` 或工具 `matrix --diagnostics` 开启。正常生产forward、训练launchers、旧monitor没有任何新增默认逻辑。启用后的JSON为统计摘要，不保存完整Q/K或完整每点source权重，不保留计算图。

记录logical round、physical core index、operator/MLP、source权重mean/min/max/entropy、source/state/Delta范数、Q沿M与K沿N的归一和误差/熵、每个共享core visit入口/出口RMS、update norm、相邻update cosine、raw branch范数、共享参数grad norm和NaN/Inf。

SR/LB的visit update来自实际点状态exit−entry；RB没有普通block residual，故明确记录两个raw branch之和作为raw partial增量，exit为当前轮partial，入口为operator router输出，不能把这三个量套成SR residual。LB轮级Delta直接读output receiver收到的实际Y−H来源。RB凸混合**不自动继承**1/R residual的稳定性结论；没有根据诊断加入任何gate、缩放或裁剪。

观测保留的临时detached张量仅属于外部context，在每次forward完成/异常时清空；模型不注册缓存或新参数。启用/关闭诊断的forward、参数梯度和RNG逐位相同；连续调用、B/N变化和异常后的下一次forward均单独验证。

矩阵收缩按语义/操作数而不是只看方形尺寸分类：每visit仅 `[Bh,M,N]×[Bh,N,d_h]` 与 `[Bh,N,M]×[Bh,M,d_h]`；合法M=d_h的context明确通过，故不误判M×M self-attention。专门注入N×N点attention的负面fixture确实被拒绝。全部48组合每个core调用R次，branch调用顺序精确，head仅一次，state无round复制键，参数对象不重复注册。

## F. 完整结果、历史失败与冻结

旧 `test_kcdno_delivery` 的 `python -I` 子进程没有 `-B`，且忽略环境 `PYTHONDONTWRITEBYTECODE`，刷新了既有 `cdlno_entry.cpython-313.pyc`；由12580变为12736字节，引发原loop isolation的ignored-cache冻结断言。对应 `.py` 源码与LL9起点字节相同，新cache的code object与该源码直接compile结果相等。这是本轮新触发的缓存产物断言，不能冒充历史失败，也没有改golden、删除或伪造旧cache让它变绿。见 [缓存定位](loop_linearno_audit/ll9/bytecode-freeze-review.json)。

最终汇总由 `docs/loop_linearno_audit/ll9/summary.json` 和 `end-freeze.json` 给出，包含每项历史失败/skip及来源，不能把“无本阶段新增旧回归失败”简写成“全仓全绿”。保留运行最初的错误日志，独立复核环境性超时，不放宽测试容差或改旧golden。

LL6/LL7完整原生闭环证据仍有效，本阶段重跑六Standard×preset B×三mode，以及Air/Car×preset A×三mode：原parser→原科学训练片段→pair→新进程resume/eval。Air两独立成员包含成员内/成员间两种中断，Car使用非零fold；合成PyG与surface/drag接口不等于真实VTK指标。其余preset完整闭环见已批准LL6/LL7 36+12组；LL9两preset48组合的工具矩阵使用synthetic state/optimizer重载，明确不把它称为新的生产resume协议。

### 最终逐项结果

| 范围 | 结果与证据 |
|---|---|
| 正式参数配置 | 96/96分项解析值=实测值；8个真实pure8构造对照；同task/topology/rank的三mode主干hash一致 |
| canonical N、小宽度 | 48/48前向/反向/AdamW/strict state及optimizer重载，误差0；调用顺序与单head验证通过 |
| 原生生产合成闭环重跑 | Standard18+工业6=24/24；105新进程；权重/optimizer/scheduler/RNG/预测逐位相同；[Standard](loop_linearno_audit/ll9/native-matrix.json)、[工业](loop_linearno_audit/ll9/industrial-matrix.json) |
| 新工具测试 | 7/7，7.495s；[日志](loop_linearno_audit/ll9/new-tools-final.log)；后续显存计量修正由完整144 GPU组合重新验证 |
| 完整既有CPU套件 | 80模块、689方法：643通过、10失败方法/15断言、36skip、最终0error；[全结果](loop_linearno_audit/ll9/regression-final.json) |
| 首次环境性超时 | history K新进程60s超时原文保留；整模块原样复跑13/13、25.458s，旧阈值未改；[复核](loop_linearno_audit/ll9/history-k-recheck.log) |
| CPU计时 | 48/48，warmup3/测量12，forward与完整合成step均有median/p90及原始样本 |
| CUDA最终矩阵 | 144项，124通过/20失败；FP32 48/48，FP16与BF16各38/48；失败仅上述五任务RB的AMP |
| 数值兼容 | loop八份报告与LL8逐字段全等；pure/history九份完成报告与LL7全等（R4使用超时后的完整重跑）；[loop](loop_linearno_audit/ll9/loop-numeric-report-comparison.json)、[pure/history](loop_linearno_audit/ll9/prior-numeric-report-comparison.json) |
| 源码/序列化 | 七套pure/history/loop provenance全等LL8；原tracked diff、HEAD/tree/staged未变；[指纹](loop_linearno_audit/ll9/preserved-provenance.json)、[冻结](loop_linearno_audit/ll9/end-freeze.json) |

旧失败来源详见 [分类及源码hash](loop_linearno_audit/ll9/regression-failure-review.json)。其中9个方法为LL9起点就存在的源码状态/旧断言问题，**部分是这次扩大套件后首次记录**，不能冒称LL8已全面确认：

- pure旧README/path.sh/REPRODUCTION_MATRIX hash与旧golden不符（2方法、4断言）。
- history static/industrial旧normalized patch包含旧git-before基准，R8与当前HEAD不同（2方法、2断言；LL7已有准确定位）。
- CDLNO/KCDNO/MSAR/visualization旧AST projection未剥离后来的LinearNO入口与resume语句（4方法、7断言）。
- `test_kcdno_delivery` 模拟没有共享`cdlno`包时，当前Standard入口的loop import触发`ModuleNotFoundError`（1方法、1断言）。这是独立项目脱离共享包运行的既有兼容问题，不是数学差异；本阶段未修。
- 另1方法/1断言是本轮ignored `.pyc` 刷新引起的冻结问题，明确单列为新增产物失败。

36个skip：33项因这次完整旧套件显式`CUDA_VISIBLE_DEVICES=''`运行，3项因本机缺torch_cluster；不把CPU套件的CUDA skip解释为本机无GPU。loop自身GPU另由144项检查覆盖，旧模型GPU完整套件未在LL9重新跑。旧测试、golden、源码、容差和skip规则均未更改。

实际主要命令与环境：

```bash
python -B docs/loop_linearno_audit/ll9/run_regressions.py
LL9_STANDARD_PRESETS=p2_c2_r2_s2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' python -B docs/loop_linearno_audit/ll9/run_native_matrix.py
LL9_PRESETS=p1_c3_r2_s1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' python -B docs/loop_linearno_audit/ll9/run_industrial.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  taskset -c 0 python -B tools/linearno_loop_performance.py cpu \
  --warmup 3 --steps 12 --output docs/loop_linearno_audit/ll9/cpu.json
python -B docs/loop_linearno_audit/ll9/recheck_timeout.py
```

上面是实际已执行命令记录；原路径已有证据，不应原样覆盖。对新的计时使用前文fresh output路径。参数/48矩阵/GPU命令见前文，实际产物位于本报告的LL9证据目录。未更改依赖、未commit/push。

交付前五项自审：独立公式与实际算子对照；共享/调用/平方张量语义；默认关闭及启用诊断的数值/RNG隔离；metadata与原生resume精确性；完整失败归因与性能边界。证据见 [审查记录](loop_linearno_audit/ll9/priority-self-review.json)。仅保留真实未解决项：RB AMP来源dtype合同、ignored-cache断言、上述既有独立入口/历史冻结问题。没有凭“参数少”推出训练效果或真实epoch更快。

NOT RUN：真实数据、真实训练或准确率、SOTA、真实epoch效率、远端Python3.10/torch2.11/cu128、正式全宽任务GPU训练、compile/distributed。LL10未执行。待后续明确处理的是RB的混合dtype合同；本阶段既不偷偷cast，也不关闭该检查绕过问题。

本 LL9 阶段结束，未执行下一阶段。
