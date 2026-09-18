# R9 综合验证、诊断、效率与公平实验协议

## A. 范围和状态

**PASS（本阶段范围）**。依据当前用户连续授权从R8进入R9，RUN_REAL_BATCH=false、RUN_MINIRUN=false、MAX_STEPS=0、RUN_FULL_TRAIN=false，CUDA_VISIBLE_DEVICES=''。没有数据、GPU或真实训练。R9不修改生产模型、协议、测试：独立增加诊断/工具/文档。综合回归发现的一项新导入回归按要求退回R6，返修独立通过后恢复R9；不是在R9隐藏失败。见[返修报告](LINEARNO_HISTORY_R6_CORRECTION.md)。

## B. 文件与依据

新增monitor/history_diagnostics.py、history_run.py、history_bootstrap/sitecustomize.py；tools/linearno_history_support.py、linearno_history_diagnostics_check.py、linearno_history_performance.py、linearno_history_cost_table.py。原monitor与测试未改；研究数学模块未改。R6返修只有Standard parser的导入边界和精确routing源指纹片段。记录和命令见本阶段证据目录与[用法](LINEARNO_HISTORY_DIAGNOSTICS.md)。

## C. 实际验证

- 全部现有测试文件清单中的62模块、578方法实际运行972.732s，日志逐模块留存。另3个native模块使用本轮先前R6–R8刚完成的闭环证据，未因重复执行节省而漏项：R6返修4方法/16闭环、R7 3方法/8闭环、R8 4方法/8闭环+双成员与补充negative1方法。R9扫描纳入模型、数学oracle、参数梯度、optimizer step、checkpoint、旧各模型、输出、visualization/resume、旧monitor。
- 首轮578方法包含一个新增R6导入失败，修复后相关旧delivery和四任务共8/8方法381.743s通过。最终逐项判定537 passed、6个历史失败方法（11个断言）、35 skipped、0 errors。不是“整个仓库全绿”。这些历史失败是README/path/旧matrix hash，以及工业入口AST快照缺少旧纯LinearNO接线；均有pre-R6重放证据，未改golden/strict/容差。完整分类在regression-adjudication.json；未解释的新失败为0。
- 32个八任务×四模式原生合成生产闭环PASS；核心20配置完整forward/backward/optimizer/strict新进程reload/eval在当前扫描再次PASS。160个实际preset构造/参数量PASS。完整按task/mode/depth/device/dtype矩阵见verification-matrix.json；真实loader/GPU/AMP/mini-run/完整3seed全部NOT RUN，不能从L4任务闭环扩张为所有深度都做了真实任务训练。
- 独立诊断48个变体×组合×train/eval检查：forward、梯度、Python/NumPy/Torch RNG精确相同；真正绘图分支也精确；whole-object保存和异常退出恢复通过。tiny float64 dense P仅用于工具自审oracle，生产监测只低秩计算。原生研究checkpoint eval在独立runner中完成，2个snapshot+PNG，metadata未改。初次命令误写seed42路径失败，保留monitor-native.txt；核对实际seed17后monitor-native-verified.txt通过，不把错误路径计作模型失败。
- 八任务×五深度×四组合×三action共480个shell dry-run通过；未执行这些真实训练命令。

关键命令：通用环境前缀`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg`；各已有文件用`python -B -m unittest discover -s <parent> -p <test_file> -v`，精确命令和耗时在regression/summary.json。新工具用[诊断文档](LINEARNO_HISTORY_DIAGNOSTICS.md)列出的命令，性能最终为warmup5/steps20。

## D. 诊断/张量/计算

研究core的显式per-call observer暴露实际Q、修正后K、pre-A raw；A/K观察器记录真实alpha/mask/gates/delta。没有修改forward公式。相同确定性点集上计算tr[(Qi^TQj)(Kj^TKi)]及低秩范数；不创建N×N。P是当前block自身路由，不是含A的Jacobian；旧Transolver图不作为实验结果。

profiler-checks.json覆盖20模式深度：A有L(L-1)/2个明确receiver下的跨深度M×M token-softmax和L-1次来源softmax；K有L-1个M×(lM)历史读取与N×M修正；pure无token自注意力，所有模式无N×N。合法M×d_h context不按数值方阵误判。train/eval矩阵MAC完全相等，history dropout固定.1在cross/scoring后mask，不节省这些FLOPs。

参数来自实际注册参数，A增量`2*d_h^2+(L-1)*(2*d_h^2+2*d_h+1)`，K增量`3*d_h^2+(L-1)*heads`；A/K合并严格相加。所有norm/bias/原温度和Air dead temperature计入；raw cache无参数。task-costs/results.json存160项正式配置各组参数和逐层矩阵MAC；24个六变体×四组合的理论计数与实际operator ledger精确相等。

每层基础Q/K/V投影分别BHN*d_h*M、BHN*d_h*M、BHN*d_h²，压缩/重建各BHN*M*d_h。卷积、input/output projection、两层point FFN、stem/time/final head均按真实注册模块计数。A每receiver有一次query投影，每旧source有K/V/O投影及2×BHM²d_h。K每receiver有共享slot query M*d_h²、历史K/V投影2BH(lM)d_h²、读出2BHM(lM)d_h和当前点修正BHNMd_h。LN/softmax/GELU/center/评分/融合等完整算子调用与元素工作另列；表中FLOPs明确是2×矩阵MAC，不冒称包含全部标量运算的完整FLOP总数。

## E. 有限性能与公平对照

本机Python3.13.9/torch2.13cu130，CPU单线程FP32，Elasticity形状N972/B2但缩小d32/h4/M16，合成MSE/AdamW。最终窗口单独运行，warmup5、measure20、诊断关闭、CPU同步计时。早期与回归并发的试运行保留在performance/，正式表使用performance-final/。没有真实task loss/time loop、GPU显存、AMP或远端验收。

| L | 模式 | 参数 | 矩阵MAC | 推理median/p90 ms | 训练样本/s |
|---|---|---:|---:|---:|---:|
| 4 | A0K0 | 21,121 | 54,058,752 | 3.973/4.584 | 120.00 |
| 4 | A1K0 | 21,684 | 54,427,392 | 4.697/5.603 | 102.72 |
| 4 | A0K1 | 21,325 | 57,342,720 | 6.554/8.154 | 103.27 |
| 4 | A1K1 | 21,888 | 57,711,360 | 5.612/6.830 | 93.28 |
| 5 | A0K0 | 25,801 | 66,500,352 | 4.386/5.757 | 90.51 |
| 5 | A1K0 | 26,509 | 67,106,560 | 6.482/9.485 | 89.70 |
| 5 | A0K1 | 26,009 | 70,977,280 | 6.170/7.345 | 76.01 |
| 5 | A1K1 | 26,717 | 71,583,488 | 8.224/9.138 | 67.33 |
| 6 | A0K0 | 30,481 | 78,941,952 | 5.459/6.460 | 90.69 |
| 6 | A1K0 | 31,334 | 79,843,072 | 8.293/9.117 | 67.19 |
| 6 | A0K1 | 30,693 | 84,660,992 | 8.053/9.312 | 71.93 |
| 6 | A1K1 | 31,546 | 85,562,112 | 10.566/11.300 | 54.20 |
| 7 | A0K0 | 35,161 | 91,383,552 | 6.666/7.464 | 68.98 |
| 7 | A1K0 | 36,159 | 92,636,928 | 9.807/10.659 | 51.68 |
| 7 | A0K1 | 35,377 | 98,393,856 | 9.414/10.603 | 49.71 |
| 7 | A1K1 | 36,375 | 99,647,232 | 11.514/13.822 | 44.81 |
| 8 | A0K0 | 39,841 | 103,825,152 | 8.434/9.703 | 54.97 |
| 8 | A1K0 | 40,984 | 105,488,128 | 12.127/13.502 | 44.28 |
| 8 | A0K1 | 40,061 | 112,175,872 | 10.403/11.527 | 46.67 |
| 8 | A1K1 | 41,204 | 113,838,848 | 16.266/18.396 | 38.47 |

单窗口CPU计时会受频率/调度噪声影响；较低MAC不自动带来较低延迟，增强同深度在本窗口有额外开销。comparisons.json同时列增强L4–7相对pureL8与L8同深度比值，不能把减少层数的效率归因于机制本身。

已预先固定匹配算法：同任务/同深度pure只统一增加global rank，按参数或矩阵MAC的最近整数匹配，固定tie选较小合法rank，width不改；Car rank必须是d_h整数倍。正式八任务240个强对照建议在matched-controls.json，记录无法精确匹配的偏差；禁止看test后调整。性能窗口另实际构造/计时30个pure匹配对照。accuracy–parameter/FLOPs/latency Pareto全标NOT RUN，没有填随机误差或虚假边界。

公平完整实验的paired seeds固定17/29/43；四模式同任务同深度共享公共主干权重，新增feature seed隔离且记录，DataLoader generator独立。数据checksum/split/normalizer、objective/evaluation_spec、batch/epochs/steps、optimizer/scheduler、precision/hardware全部一致；只允许预声明的架构模式/深度/对照rank差异。对每seed报告原指标和配对差值，再报mean±sample-std；final checkpoint，不用test选权重/seed/rank。Air ensemble还匹配member顺序与seed+i。原paper/official协议未被研究名称改写。正式实验顺序先pureL8及同深度四组合，再增强L4–7/浅层pure，再预声明matched controls；完整运行需要另行授权。

## F. 冻结与边界

R0清单复算覆盖tracked/untracked/ignored，42个既存模型目录/根attention/monitor源码全部byte-identical；纯共享attention/三任务模型和六exp另有R8保护清单，研究数学也未变。相对R0仅列明的9项既存源码/测试接线路由变更，及ignored pyc状态差异；旧文档/用户dirty没有当作本任务新改动。R9 freeze包括返回R6的2个文件，并明确归因到返修。未删/清理用户文件。

真实数据读取完整性、真实VTK/force、收敛/精度、实际epoch时长、CUDA/AMP/远端、3seed论文指标与SOTA均未验证。训练峰值显存为NOT RUN，不用CPU进程内存冒充。诊断只支持显式opt-in eager单进程，详细限制见用法。现有6项历史测试失败保留，不将其称为全库通过。

自审：核对有效K而非base K的监测、无图残留、RNG/pickle无扰动、每receiver/source的张量标签、参数/MAC逐模块账、旧parser隔离回归、metadata-before-load、八任务原objective与mask。均有来源与运行证据。R9结束，按用户连续授权进入R10。
