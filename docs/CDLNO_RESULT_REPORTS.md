# 三子项目离线结果整理交付

## 2026-09-16 补充：每个数据集仅展示最后验证损失最小的seed

本节更新当前行为，下文首次交付记录原样保留。用户授权只改变离线报告筛选与汇总；源实验（包含未入选seed）、旧报告、checkpoint、模型和训练协议不改写。

### A. 完成范围

三个既有shell共用的报告器现在先应用显式筛选，再为每个task选择一个run：**该run最后一次记录的验证损失最小**，不取best-ever epoch，不用独立测试结果选种子。曲线、复制场图、评价表和ZIP主体仅包含获选run；全部候选仅保留在`selection.json/CSV`作为审计依据。PDE六任务各自选优，以Darcy/Airfoil/Plasticity/Elasticity/NS/Pipe顺序输出`pde_selected_training`和`pde_selected_evaluation`，各为一张2×3子图的PDF/PNG；HTML优先展示这两张汇总。原入选run的详细单图仍保留。

### B. 文件与代码对应

| 文件/符号 | 改动及理由 |
|---|---|
| `tran_evaluate/show/_report.py:validation_value` | 直接解析已存验证指标/正确工业分项；无训练器、模型或checkpoint加载 |
| `Bundle.selection_record/select_runs` | 按成员最后验证epoch算run值；每task独立选一个；状态、缺损、同值和来源审计 |
| `Bundle.history` | 保留旧JSONL/文本解析，识别旧文本NaN/Inf；选优与曲线使用同一历史快照；损坏历史不参与选优 |
| `Bundle.pde_overview/pde_evaluation_overview` | 六任务各选一个seed的训练/正式评价汇总；缺项留空；不以测试结果二次选优 |
| `main/report_run/make_index` | 先选后绘，manifest v2记录依据，已有目录/ZIP继续拒绝覆盖 |
| `tests/test_show_reports.py` | 保留原8项验证，新增6项覆盖本次选择风险；夹具可同时构造多个seed |
| `tran_evaluate/show/README.md` | 当前规则、三原命令、输出文件、指标/legacy边界 |
| 本报告、CDLNO/KCDNO STATUS、memory、`docs/show_audit/selection/` | 增量交付和证据，保留历史 |

三个shell的路径定位和CLI调用未改；没有新增训练功能、修改已存数据、移动目录或删除文件。

### C. 指标公式及选择边界

- 四个静态PDE：最后`validation_relative_l2`。NS/Plasticity：最后`test_full_loss`，这是原训练循环的held-out监控（上游名为test），不声称新增独立验证集。
- Car：`velocity_mse + saved_weight × pressure_mse`；AirfRANS：`volume_mse + saved_reg/weight × surface_mse`。使用已正确命名的分项；Car外层交换日志与Air旧拼写分支标量不作为目标。原文件中的这些旧值完整保留。
- Air默认398轮且最后验证在390时记录390，不将398轮无验证记录视为0，也不回退历史最小值。若run包含多个成员，先取各成员最后验证，再算术平均选run；不挑选最优成员、不跨seed平均，入选成员曲线分开。
- 有`train_results.json`但状态非completed的run不参选。最后验证NaN/Inf、负值、缺分项/权重/成员、损坏JSONL均不参选，不偷偷回溯更早的好值。旧run无状态文件但有有效验证历史可参选，明确标为完成状态未核实；没有seed时显示unrecorded。
- 完全同值按明确整数seed升序、再按源绝对路径字典序确定一个。不同架构/训练设置/fold/split被同时纳入时发出提示，只称为本次纳入run的最优值；`--model`或重复`--run-dir`可限定可比实验组。
- 独立评价图显示选中run全部已完成评价，失败/部分结果仍在表中保留状态。原场图取该run已存的latest/all，不假装是验证epoch重新推理出的图。

### D. 实际验证

`PYTHONPATH=tests:. python -B -m unittest test_show_reports -v`：**14/14通过**，见[最终日志](show_audit/selection/tests.txt)及[命令/耗时](show_audit/selection/test_run.json)。新增覆盖六PDE×三个seed交叉曲线（历史最优与末次最优不同）、独立测试值反向诱导不影响选择、两个六子图轴/seed核查、只复制选中run、ZIP CRC、所有源文件与旧报告hash不变；Car正确加权分项，Air稀疏验证/成员均值/缺成员；最后NaN、缺分项、损坏JSONL、旧文本NaN、失败/运行中、同值排序、legacy状态与dry-run。

过程中的一次8项回归发现通用历史缓存让同一个Bundle重读更新日志仍得到旧内容；已把缓存限定为本次选优后供绘图使用的快照，原重读语义恢复，最终14项通过。没有修改生产日志协议来迁就测试。

实际使用临时目录中明确标记`SYNTHETIC_RECORD_FIXTURE`的18份输出记录生成完整PDE报告，逐任务获选seed为0/1/2/0/1/2；生成PDF/PNG/图注/CSV/HTML/ZIP并检查两张六子图。训练汇总图例调整到注释下方以避免遮挡下降曲线。见[渲染路径、版本、选取结果](show_audit/selection/render.json)。这些是**保存结果格式夹具**，没有进行模型训练、真实数据读取或重新评价。

本机Python3.13.9、NumPy2.2.6、Matplotlib3.10.6，沿用现有环境。三个shell的模拟远端带空格路径/外部cwd检查继续通过。工具不需要GPU/PyG/Torch；未执行GPU计算、未访问远端真实产物。

### E. 冻结与自审

修改前快照`/tmp/show-selection-before-cdua003g`及[freeze](show_audit/selection/freeze.json)核对既有149个生产/启动文件及3个show shell完全一致；模型、数据、损失、采样、时间循环、训练/评估、checkpoint及用户既有修改均保留。本次可审查[增量patch](show_audit/selection/changes.patch)只涉及报告器/定向测试/用法。

四个重点已自审：最后epoch与历史最小/测试值严格区分；工业损失权重及真实稀疏epoch可追溯；PDE各自单seed与六子图完整对应；源实验和旧报告只读、输出目录独立。证据分别为上述数值夹具、CSV/轴检查、文件hash/ZIP检查及源代码写入路径。

### F. 未验证/限制

未连接远端读取实际训练产物；缺验证历史的旧run无法自动确定最优seed，只记录排除原因。混配置的候选应按所需实验组筛选，不能把默认跨配置选出的结果宣称为受控种子比较。源文件逐文件读取不是跨文件原子快照；硬中断遗留状态不由工具擅自修正。未运行真实训练、模型加载或新的物理指标，未commit/push或安装依赖。

本阶段结束，未执行下一阶段。

---

## 首次交付记录（历史，选择策略已由上节更新）

日期：2026-09-16。用户授权：核对当前仓库训练/测试产物，为Car、AirfRANS、PDE标准任务各提供一个远端可用的报告脚本。只新增离线整理工具/测试/说明；未修改生产模型、训练入口、指标、数据、checkpoint或依赖。

## 实际文件与功能

- `tran_evaluate/show/Car-Design-ShapeNetCar.sh`
- `tran_evaluate/show/Airfoil-Design-AirfRANS.sh`
- `tran_evaluate/show/PDE-Solving-StandardBenchmark.sh`
- 共享 `tran_evaluate/show/_report.py`；没有复制三套解析/绘图实现。
- [使用说明、远端命令及完整输出清单](../tran_evaluate/show/README.md)。

脚本路径定位实际checkout。默认扫描output、runs、该项目metrics/scores；显式run/root可覆盖。各项目写入 `result_visualizations/<UTC时间戳_ID>`，同时ZIP，目录/归档不覆盖。PDE内按六task和run区分；既有每次evaluation和成员/seed/fold均保留。HTML图册、PDF/PNG、英文/LaTeX图注、训练/评估CSV与数值LaTeX表、来源hash和缺项记录一起交付。默认只复制最后一个已有周期epoch的场图，其余曲线/评估记录完整；可选所有场图/原数组。

## 逐项源码反查

| 实际生产位置 | 实际产物/语义 | 脚本处理 |
|---|---|---|
| `cdlno/experiment.py:Experiment.__init__/record_epoch/finish` | config、逐epoch JSONL、train_results，task/model/seed/member/parameters | 读取这些结构化字段，参数量使用实际记录，不构造模型 |
| `Experiment._write_result` | evaluations/*/results.json为每次评价权威记录，eval_results.json是索引 | 优先各次记录，索引仅作后备；全部保留状态，不选择最好结果 |
| 六`exp_*.py`的record_epoch/record_metrics与print | 静态relative-L2、Darcy梯度项、NS/Plasticity step/full，epoch+1 | 指标分开作图；旧PDE文本log有限格式后备；不导入exp |
| `Car-Design-ShapeNetCar/train.py:main` | train_components、validation_*与upstream外层日志 | 正确pressure/velocity分项；已记录weight时由正确分项明确派生真实训练目标，原交换名字的upstream值仅进表 |
| Car`main_evaluation.py` | rho_d、c_d、relative-L2、归一化RMSE、未同步forward time；数组缺几何/系数配对 | 指标分图/表；不把旧计时图示为可靠GPU性能，不重新计算Car阻力/空间场 |
| Air`train.py:main` | 实际weighted loss、surface/volume、四通道；仅真实验证轮次有validation键 | 沿记录epoch画曲线，稀疏验证不补点；旧拼写分支标量不冒充修正目标 |
| Air`main_evaluation.py`/`utils/metrics.py:Results_test` | score均值/成员std；true_coefs[N,2]，mean/std[N,group,2] | 正确group/channel轴的表/柱状/升阻力散点，std不是CI |
| Air`metrics_NACA.py:surface_coefficients` | true[2,N,2]，pred[group,2,N,2] | 原Cp/Cf散点，Cp倒置轴，不跨上/下表面画连线 |
| `cdlno/periodic_visualization.py`/`visualization.py` | member/epoch/case的场图、caption、metadata、可选原数组 | 按原样复制，不重跑模型，不隐式归一化或重设色标 |

继承既有正式模型和训练协议。纸面输出风格沿用上一阶段论文可视化审查：STIX、7–9pt、单栏3.5/双栏7.16in、默认600dpiPNG、PDF TrueType嵌入；不声称复刻某论文未披露的精确字体。新图不做平滑，不跨run/seed/fold计算平均。

## 实际验证与范围

`PYTHONPATH=tests:. python -B -m unittest test_show_reports -v`：8项通过，见[日志](show_audit/tests.txt)。覆盖：三个真实shell从带空格模拟远端目录及外部cwd运行；旧CDLNO_REPO_ROOT不重定向脚本；八task结果夹具；原文件hash不变；源epoch/稀疏验证；重复评价/失败状态；seed筛选；只读预览；旧PDE日志；半行JSONL；array轴；拒绝object pickle；重复输出拒绝；真实PDF/PNG/caption/ZIP及CRC检查。

另使用明确标记的合成**输出记录**生成Car、AirfRANS、Darcy完整报告，实际查看损失曲线、表面MSE、气动力散点和指标柱状图；不是实际训练/精度结果。所有演示输入位于临时目录，没有创建冒充真实数据集的文件。源码及既有记录149个文件逐字节不变，证据见[freeze.json](show_audit/freeze.json)。本次只新增show目录、tests/test_show_reports.py和增量文档状态。

实际环境：Python3.13.9、NumPy/Matplotlib沿用现有安装；工具不导入Torch/PyG/项目训练器。远端Python3.10路径合同通过模拟目录验证，未连接远端、未以远端真实训练产物验收。当前本地output仅有此前检查留下的架构记录，不能用它生成真实训练曲线。无真实训练/重新评价/依赖安装/commit/push。

自审结论：曲线/指标口径、Air数组轴、源结果只读、路径可移植、目录隔离与失败/缺项展示均通过。剩余限制是远端实际产物可能缺历史记录、旧格式不完整、缺几何/系数配对；脚本会记录，不能从checkpoint或图片补造原始训练轨迹。正在写入的run不是跨文件原子快照，源变化提示保留在manifest/WARNINGS。生成的报告应根据保存状态阅读，尤其running状态可能由硬中断遗留。

本阶段结束，未执行下一阶段。
