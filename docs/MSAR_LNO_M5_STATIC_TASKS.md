# MSAR-LNO M5：四个静态任务接入

2026-09-17，仅M5。M4已获用户审查通过；本轮在`main@9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`的现有未提交工作上增量实现，未reset/commit/push/PR。起点见[before](msar_lno_audit/m5/before.json)和[先存diff](msar_lno_audit/m5/preexisting.patch)，707个文本文件快照位于`/home/hwz/CDLNO-artifacts/msar-m5-before-2jli41sc/source`。此前M0—M4、CDLNO/KCDNO、seed、可视化及输出记录工作均保留。

## A. 完成范围与任务覆盖

`msar_lno`现已接入Darcy、Elasticity、Airfoil、Pipe的**实际CLI、factory、模型wrapper、训练loss、原eval、严格checkpoint和输出记录**。默认Light，Full可显式选择。NS、Plasticity、Car和AirfRANS没有接入MSAR；标准时间入口显式请求MSAR会报“not integrated in M5”，不暗中使用不适配core。

| 任务 | 实际入口 / 调用 | 输入提升和输出 | 本轮合成原loss训练步 | eval / checkpoint |
|---|---|---|---|---|
| Darcy | `exp_darcy.py`；`model(x,fx.unsqueeze(-1))` | 原索引网格ref64距离替换xy，拼coeff1，stem65，无placeholder；输出[B,N,1] | floor/off，B2/7×7/d8；decode后field relative-L2 +0.1×原边界处理/中心差分derivative-L2；1 backward/optimizer/scheduler | 原decode/relative-L2片段、同权重strict往返通过 |
| Elasticity | `exp_elas.py`；`model(x,None)` | 原xy2、逐点MLP+placeholder；输出[B,N,1] | floor/off，B2/N35/d8；原target decode和relative-L2；Cosine仍在epoch外层步进 | 原eval片段、同权重strict往返通过 |
| Airfoil | `exp_airfoil.py`；`model(x,None)` | 原物理xy2、逐点MLP+placeholder；输出[B,N,1] Mach | floor/off，B2/5×7/d8；原relative-L2；1 optimizer/OneCycle step | 原eval片段、同权重strict往返通过 |
| Pipe | `exp_pipe.py`；`model(x,None)` | 原入口的normalized xy2、逐点MLP+placeholder；输出[B,N,1] | floor/off，B2/5×7/d8；保留x/y normalizer和target decode；1 optimizer/OneCycle step | 原decode/eval片段、同权重strict往返通过 |
| NS / Plasticity | 未接入 | M6依赖：时间输入与rollout/逐T语义 | 未运行MSAR | 未运行MSAR |
| Car / AirfRANS | 未接入 | M7依赖：真实PyG/单图及工业wrapper | 未运行MSAR | M4只有core协议基础，不能算任务通过 |

表中训练步使用明确的缩小测试结构`d8/M=[7,5,3,2]/heads=[2,2,4,4]`，双depth仍[3,1,1,1]，没有修改正式Light/Full。四任务正式网格N7225/972/11271/16641以小d/M单独完成layout forward。Darcy wrapper另以5×7验证，**原Darcy导数函数按方形resolution计算，所以完整loss检查用7×7；没有为了测试改原公式**。

正式Light Elasticity（d96/M512…）在N35、正式Full Elasticity（d192/M1024…）在真实N972完成构造/前向：M1>N仍为1024，未裁剪。实测含lift/head的参数量分别1,710,169和6,811,825；这些是Elasticity的参数，非全部任务通用数。本轮这两个正式尺寸检查没有反传，训练步证据来自完整拓扑的小模型。

## B. 实际文件与复用

| 文件/符号 | 本轮变化及理由 |
|---|---|
| `cdlno/msar_lno/standard.py::StaticModel`（新） | 四任务共用wrapper；复用已确认`STATIC_TASKS`的字段合同、`_grid`与普通Linear初始化，不继承旧d/h/M；独立逐点lift+现有MSAR core |
| `model/MSAR_Standard.py`（新）、`model_dict.py` | 对已解析的四静态任务导出StaticModel；保留M4无task时的真实lifted-core导出，旧factory分支不变 |
| `cdlno/msar_lno/standard_entry.py`（新）、子项目`msar_entry.py`（新） | 新family参数解析/Run/严格metadata；复用M1解析与metadata、M4训练adapter、既有输出记录 |
| `PDE.../cdlno_entry.py` | 仅选中msar_lno后复制parser并解析新参数；旧parser默认值/kwargs不变 |
| 四个`exp_*.py` | 添加独立启动/构造、训练forward/aux backward和四项日志分支；旧调用保留在else。eval预测语句、data、normalizer、原loss、optimizer/scheduler/clip/epoch/采样/图表语句不改 |
| `cdlno/msar_lno/objective.py::ObjectiveMetrics` | 逐epoch、调用者持有的detach累计；四项使用相同优化步分母，无model.last_loss/跨forward状态 |
| `cdlno/experiment.py` | 最小新增family/run_key/seed记录与显示名；复用原early config/参数量/history/独立eval目录及周期可视化 |
| `configs/msar_lno/{darcy,elasticity,airfoil,pipe}.json`（新） | 完整复制既有训练/坐标预设；结构默认profile=light；objective独立floor/.01/.2/eps1e-6/diagnostics=false，可被显式CLI覆盖 |
| `tran_evaluate/msar_lno/`（新） | 四薄脚本+共享`_static.sh`复用原dispatcher；只设置model/profile/eval/data路径，不将M/d写成显式CLI |
| 测试/文档 | 新`test_msar_static.py`和`msar_entry_projection.py`；既有静态/K/front/输出/可视化冻结检查精确移除新家族分支，再继续原断言；更新报告、覆盖表、STATUS和memory |

M1配置数学、M2原语、M3core、M4core checkpoint、所有旧模型/共享数学、所有依赖保持字节不变。没有重复四套模型数学。新family完全没有Conv1d/2d/3d、旧ConvFFN或N点SA；规则网格H/W仍保留为数据布局/坐标合同。

[本轮diff](msar_lno_audit/m5/stage.patch)相对于M5起点，避免把已有用户修改误归本轮。[freeze](msar_lno_audit/m5/freeze.json)记录完整旧入口AST比较与未变文件hash。

## C. 模型、loss和日志合同

统一wrapper执行：

```text
原坐标/条件约定 → Linear(stem,2d) → GELU → Linear(2d,d)
                  → 仅fx=None任务加原placeholder
                  → M3四级MSARLNO（包含唯一最终LN+Linear head）
```

没有额外wrapper head，没有在构造末尾递归初始化core。静态T非None、错误fx/坐标通道、N!=H×W明确拒绝；Elasticity支持不同N。Darcy的reference来自原索引网格，未重新用任意物理x计算reference。

训练每次调用：`training_forward`显式得到prediction/aux → 执行原任务LPDE → `training_objective(LPDE,forward).total.backward()`。因此

`Ltotal = LPDE + coverage_weight * mean_l(raw_coverage_l)`。

off/weight0不请求Down的A，total为原LPDE同一张量；eval和可视化仍直接调用预测Tensor，不计算coverage。默认diagnostics关闭。训练反传、scheduler与梯度裁剪次数/位置均通过真实batch AST核对。

原train_loss保持原按ntrain归一的统计（Darcy保留field项与独立derivative_regularizer）。新增`objective_pde/coverage_raw/coverage_weighted/objective_total`统一记录**每优化步均值**，日志同时写coverage_mode和reduction文字；PDE项保留原batch求和尺度，没有为了日志重标定训练目标。epoch累计只保存detach标量，不持有计算图。默认floor低于已满足的覆盖时raw=0是合法结果；有限GPU连接测试另用kappa1检查活跃floor路径，不改正式默认值。

## D. 配置、输出与加载

训练优先级为显式CLI > 所选新结构profile / 新任务JSON目标默认 > 新family默认。旧n_layers/n_hidden/slice_num的parser默认仍存在于旧Namespace协议中，但不会进入MSAR结构；实际结构以`msar_architecture`/metadata完整resolved值为准。选择Full不会被旧d64/M32覆盖。显式旧front/rear/CDPA/kernel/层数/dropout等不适用选项仍按M1报错。

正式训练预设保持：epochs500、lr.001、weight_decay1e-5、clip.1；batch Darcy4/Elasticity1/Airfoil4/Pipe8；Elasticity Cosine、其他OneCycle。数据、split、通道、ntrain、点序和normalizer不变。

训练在数据读取前通过既有Experiment创建`output/<task>/msar_lno/<light|full>/coverage_<floor|off>/<UTC+uuid>`，写config/log/status；参数量构造后实测。显式save_name保留独占leaf；显式run路径继续exclusive防覆盖。每epoch记录原指标及新增四项目标；沿原观察工具保留50完成epoch/final可视化。

MSAR静态Run保存裸`model.pt` state_dict、完整`architecture.json`及`task.json` wrapper/输入合同。eval先读取已有结构和训练目标，再校验显式覆盖，并恢复ref/unified_pos/downsample(x/y)/ntrain等正常量。错误结构、wrapper网格或缺必要字段拒绝，不补缺失架构，不丢参数，不使用strict=False。显式coverage变化仅记录为eval请求差异，预测路径不受其影响；训练config、两sidecar、train_results在两次eval后逐字节不变。新进程在原PDE cwd四wrapper同权重严格加载通过。

保存频率沿原`ep %100==0`及final，不新增resume；这些文件没有optimizer/scheduler/RNG状态，不能宣称断点续训。旧整对象/列表路径完全未改，工业任务留M7。原eval图表仍有固定网格尺寸的继承限制；非默认downsample下的全部实际绘图未验收，本阶段没有擅改绘图或数据协议。

## E. 实际命令、结果和限制

本机Python3.13.9/torch2.13.0+cu130/CUDA13.0/RTX5090 Laptop/PyG2.3.1；未安装依赖。[environment](msar_lno_audit/m5/environment.json)、[summary](msar_lno_audit/m5/summary.json)、[逐case记录](msar_lno_audit/m5/task-cases.json)、[fixture索引](msar_lno_audit/m5/fixture-index.json)。

| 检查 | 实际结果 |
|---|---|
| MSAR M1—M5 | 82/82，22.268s，零失败/错误/skip；含15个新M5方法；[log](msar_lno_audit/m5/msar-tests-final.log) |
| 相关旧任务/输出/可视化 | 79项，68.121s，零失败/错误；两处AirfRANS因缺torch_cluster跳过（其中一个为子用例）；[log](msar_lno_audit/m5/old-tests-final.log) |
| 同权重旧模型 | 本轮重新回放K0 41 + K/matched任务wrapper24 =65份，CPU FP32 MATH、atol=rtol=0；其余未触及10份沿用M4证据。182旧fixture文件hash全数复核不变 |
| 已接受MSAR core | 正式Light/Full两份M3固定输入、同一权重、eval和floor输出重新精确回放 |
| 新wrapper训练/eval/checkpoint | 四任务×floor/off八case实际原loss、backward/step/原eval片段/strict往返通过；数据是合成张量 |
| CLI/脚本 | 12个train dry-run、12个已有checkpoint的eval dry-run均由真实parser检查；无Python exp执行/真实数据读取 |
| 新wrapper GPU | 四任务小模型FP32 MATH原loss训练步通过；非正式大模型epoch性能测试；M2/M3既有有限AMP检查也随套件运行 |
| 观察与记录 | early config/实测参数/四项目标/history/两次eval不覆盖通过；MSAR可视化调用以捕获render参数验证eval状态/RNG/权重不变，已有renderer测试保持 |

本轮没有真实数据下载/读取/训练/收敛/准确率或远端torch2.11/cu128验收，没有新MSAR工业PyG集成、完整大任务AMP矩阵或新性能结论。

首轮新测试有两处夹具问题：用不同AST树寻找同一Elasticity循环、误以为原Darcy入口可传独立H/W；分别改为同树定位和独立wrapper的5×7检查。随后fresh-cwd零容差比较因子进程未锁定MATH出现最大约6.98e-10差异，统一backend后零容差通过，没有放宽容差或改模型。旧记录冻结测试初次未识别新分支，补充显式分支投影后完整旧AST相等。初次失败日志保留，最终日志覆盖修正结果，不将失败写成已通过。

执行命令（根目录；完整stdout/stderr保存在相应log）：

```bash
MSAR_M5_RESULTS=docs/msar_lno_audit/m5/task-cases.json python -B -m unittest discover -s tests -p 'test_msar*.py' -v
PYTHONPATH=tests:. python -B -m unittest test_static_standard test_kcdno_tasks test_front_training test_experiment_records test_periodic_visualization -v
python -B docs/kcdno_audit/make_regression_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/msar_lno_audit/m5/k0-replay.json
python -B docs/msar_lno_audit/m0/wrapper_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers --result docs/msar_lno_audit/m5/wrapper-replay.json
python -B docs/msar_lno_audit/m3/core_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/msar-m3-before-jx4v53vh/post-m3-core --result docs/msar_lno_audit/m5/msar-core-replay.json
```

本机artifacts路径用于复核已有同权重证据；远端需迁移夹具并替换路径，不重新随机生成“旧基准”。

## F. 可运行命令与自审

[四任务×Light on/off/Full训练及评估模板](../tran_evaluate/msar_lno/README.md)使用实际脚本、真实选项、用户参数最后覆盖。远端先切换到实际checkout根目录，例如：

```bash
# 仅预览
bash tran_evaluate/msar_lno/darcy.sh train --profile light --coverage-mode floor --gpu 0 --dry-run
# 用户自行运行时：同一run训练成功后评估
RUN="$PWD/output/elasticity/msar_lno/full/coverage_floor/$(date -u +%Y%m%dT%H%M%S%NZ)"
bash tran_evaluate/msar_lno/elasticity.sh train --profile full --gpu 0 --msar-run-dir "$RUN" &&
bash tran_evaluate/msar_lno/elasticity.sh eval --gpu 0 --msar-run-dir "$RUN"
```

这些是交付命令，未在本轮启动真实训练。Airfoil/Pipe及Light off的成对命令见链接中的12格表。

已完成五项自审：四任务字段/坐标/loss及调度顺序；MSAR无卷积/无重复head/无递归重置；off无A和coverage仅新family；read-first/strict/sidecar不覆盖及独立目录；新分支移除后旧入口完整AST与起点相等、旧权重精确回放。未发现需用户裁定的新架构冲突。余项为M6时间任务、M7工业任务及后续综合/成本验证，未借本轮实施。

**本M阶段结束，未执行下一阶段。**
