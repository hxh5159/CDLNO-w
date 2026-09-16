# MSAR-LNO M6：Navier–Stokes / Plasticity 时间任务接入

2026-09-17，仅M6。M5已审查通过；基于`main@9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`的已有未提交工作增量修改。读取根AGENTS、[M0实际接口审计](MSAR_LNO_REFERENCE_AUDIT.md)、[M5报告](MSAR_LNO_M5_STATIC_TASKS.md)、MSAR STATUS、memory及当前两时间入口/旧wrapper/配置/测试/启动器，核对总控和`PLAN_MSAR_LNO/MSAR_LNO_Codex_Staged_Prompts.md`的M6；未借旧阶段状态执行下一阶段。

## A. 完成范围与任务覆盖

`msar_lno`现在可通过真实`exp_ns.py`和`exp_plas.py`选择、训练、评估及严格加载。两任务共用独立`TemporalModel`包装已确认的MSAR core；Light默认，Full显式选择，逐点lift/head，无卷积或跨物理时间状态。原数据读取、split、点序、normalizer、loss、时间循环、optimizer/scheduler、裁剪和指标保留。

| 任务 | 实际wrapper输入/输出 | 模型及原loss合成训练步 | 评估与checkpoint | GPU / 真实数据 |
|---|---|---|---|---|
| Darcy / Elasticity / Airfoil / Pipe | 原M5合同不变 | M5有效证据保留；本轮M5测试重跑通过 | M5测试及新增4份pre-M6同权重回放通过 | 继承M5；真实数据未运行 |
| NS | `model(x[B,4096,2], fx[B,4096,10]) → [B,4096,1]` | floor/off；CPU B2/d8；10次forward、逐帧原relative-L2求和、1次backward/optimizer/scheduler通过 | 10次预测回填、原整段metric、严格同权重往返通过 | 本地CUDA B1/d8完整10步通过；真实数据未运行 |
| Plasticity | `model(x[B,3131,2], fx[B,3131,1], T[B,1]) → [B,3131,4]` | floor/off；CPU B2/d8；20次forward/loss/backward/optimizer、batch末1次scheduler通过 | 原20次独立T调用、拼回[B,N,4,20]、严格同权重往返通过 | 本地CUDA B1/d8完整20时间点通过；真实数据未运行 |
| ShapeNet-Car / AirfRANS | 未接入MSAR | 未执行；留M7 | 未执行新family工业checkpoint | MSAR真实PyG未执行；不能标为支持完成 |

缩小训练结构显式使用`d8/M=[7,5,3,2]/heads=[2,2,4,4]`，encoder/decoder各[3,1,1,1]，保留全部拓扑；不是正式Light训练结果。空间N仍是真实布局4096/3131，未把时间并入N。另在CPU以正式Light/Full、B1、同样真实N完成构造/前向，未执行这些大结构的完整训练反传：

| 任务 | Light实际参数 | Full实际参数 | N / Full首层M |
|---|---:|---:|---|
| NS | 1,723,897 | 6,839,281 | 4096 / 1024 |
| Plasticity | 1,729,180 | 6,886,708 | 3131 / 1024 |

## B. 文件、符号与接入

| 文件/符号 | 本轮原因与范围 |
|---|---|
| `cdlno/msar_lno/temporal.py::TemporalModel`（新增） | 两个时间任务输入提升/时间注入/shape检查/adapter metadata；复用同一个原MSAR core，不复制数学 |
| `model/MSAR_Temporal.py`（新增）、`model_dict.get_model` | 稳定共享类路径`cdlno.msar_lno.temporal.TemporalModel`，仅新增MSAR时间task选择；M4无task的lifted-core及M5静态选择保留 |
| `cdlno/msar_lno/standard_entry.py` | 接受两新任务，严格按任务检查StaticModel/TemporalModel；复用M5解析优先级、Run、metadata和独立目录；日志注明时间聚合尺度 |
| `cdlno/msar_lno/objective.py::rollout_objective`、PDE `msar_entry.py` | NS明确的per-batch tuple聚合；调用M4单forward adapter验证raw coverage，未修改原`training_forward/training_objective` |
| `exp_ns.py`、`exp_plas.py` | 新family早期记录、模型构造、显式aux、backward目标和四项日志；旧训练/eval完整分支保留 |
| `configs/msar_lno/{ns,plasticity}.json`（新增） | 独立Light默认、floor/.01/.2；原epochs500、lr.001、weight_decay1e-5、NS batch2/no clip、Plasticity batch8/clip.1 |
| `tran_evaluate/msar_lno/{ns,plasticity}.sh`（新增）、`_static.sh` | 复用已有远端路径/dispatcher；只加task路由；用户参数最后覆盖，profile不被硬写的旧d/M覆盖 |
| `tests/test_msar_temporal.py`（新增11方法） | 安全真实parser/构造/训练和eval AST、模型数值、coverage均值、time梯度、无泄漏、checkpoint、脚本/日志/正式结构/GPU |
| `tests/test_msar_static.py`、`tests/test_static_standard.py` | M5“时间任务未授权”断言更新为M6已支持；精确新增factory分支白名单；时间冻结从字节相等改为移除显式MSAR分支后的完整AST相等 |
| 文档、STATUS、memory、`docs/msar_lno_audit/m6/` | 增量交付与来源/运行/完整性证据，不改写旧阶段报告 |

本轮13个起点已有文本文件修改、9个新增实现/测试/报告/核查文件（不含生成的log/JSON/patch），其余771个起点文本文件字节相等。起点784文件快照：`/home/hwz/CDLNO-artifacts/msar-m6-before-pqh_ca0e/source`。[before](msar_lno_audit/m6/before.json)、[原有diff](msar_lno_audit/m6/preexisting.patch)、[本轮diff](msar_lno_audit/m6/stage.patch)、[冻结核查](msar_lno_audit/m6/freeze.json)分别保存，未reset/commit/push/PR。

## C. 时间及coverage公式→代码→验证

NS保留索引网格ref8²距离替换xy，再拼10帧，stem74；没有placeholder或time_fc。训练窗口在步t只能包含原输入剩余帧和`yy[..., :t]`，本步之后回填本步真值；eval只回填本步预测。模型每次重新执行4个Down、encoder/decoder和final Up，core没有时间cache。循环次数由原`range(0,T,step)`决定，现有数据协议`T_in=10,T=10,step=1`不变。

令`q`为某一次forward、`B`为该次batch、`j`为有效source，

```text
c_q = (1/4) Σ_l [(1/B) Σ_b Σ_j relu(kappa*mu_bj - p_qlbj)^2 / (mu_bj+eps)]

NS 单次优化更新：
LPDE = Σ_(q=1..Q) TestLoss(pred_q, truth_q)          # 原式，TestLoss对batch求和
C = (1/Q) Σ_(q=1..Q) c_q                           # 实际forward数Q，默认10
Ltotal = LPDE + coverage_weight * C

Plasticity 每个独立时间点q的更新：
LPDE_q = TestLoss(pred_q, truth_q)                   # 原式、batch求和
Ltotal_q = LPDE_q + coverage_weight * c_q
```

原PDE聚合没有改成时间均值；NS只对新增coverage按实际次数平均，**不是**Σ_q(LPDE_q+weight*c_q)，没有额外Q倍辅助项。Plasticity20次更新仍互相独立，不把每次coverage再除20；epoch目标日志才对这20次更新取均值。coverage自身既定batch/层均值保持，不新增batch倍乘。原`train_step_loss`仍除样本数/时间数；四项目标日志统一以optimizer step计数，因此不能把它们误认为同一归一化的指标。

| 合同 | 实际代码位置 | 验证 |
|---|---|---|
| raw每forward新建，NS单更新时间均值 | `core.MSARLNO.forward`（原）、`objective.rollout_objective`（新增）；`exp_ns.main`训练batch创建局部list→tuple聚合后删除 | 10份实际aux均值逐项对照；每份raw的聚合梯度为1/Q；重复同一序列不放大raw；所有Down每步重新执行 |
| Plasticity时间注入 | `TemporalModel.forward`：原sin/cos频率/顺序→两Linear的SiLU time_fc→逐点加；时间不进N | 原公式独立拼装相等，T及time_fc有限有效梯度，T改变会改变输出，错误shape拒绝 |
| Plasticity逐时间点更新 | `exp_plas.main`原循环只替换MSAR forward/backward目标 | 原20次optimizer/1次scheduler，原collate按样本置换T及对应label保持匹配，fx不回填预测 |
| off/weight0 | 原`training_forward`不请求aux；`rollout_objective`返回原LPDE对象 | exact对象同一、无coverage构造；同权重floor/off预测atol2e-6/rtol1e-4 |
| 评估无aux、无未来真值泄漏 | 两个原eval循环未改；预测Tensor | 断言CoverageFloorLoss不被调用，eval标签加37后预测exact不变；逐窗口比对预测回填 |
| 调用隔离 | core局部E/D；wrapper仅参数/坐标buffer | eval A→B→A exact；第二forward输出/coverage对第一forward独立输入梯度为None；模块没有新增cache属性 |
| 诊断显式启用 | 既有`return_aux=True` + `MSARTrainingConfig(diagnostics=True)` | eval不计算coverage；diagnostics为小型no-grad统计，预测exact不变；入口常规eval不自动请求aux |

## D. Checkpoint、命令与实际结果

保持原裸`model.pt` state_dict和`architecture.json`/`task.json`协议，不新增resume或宽松加载。eval先读已存resolved结构和任务输入合同再构造；显式d/M/ref/unified_pos冲突在加载前拒绝；Time_Input错误也在读取weights前拒绝；缺state key严格报错。coverage on/off纯eval差异允许但不写回训练metadata。三次重复eval分别创建结果目录，旧config/sidecar/train_results字节不变。

实际运行目录仍为`output/<task>/msar_lno/<light|full>/coverage_<floor|off>/<timestamp+uuid>`，保留显式save_name/run路径独占保护。早期config记录、实测参数数、train_history/四项目标日志和最终eval结果使用已存在记录器。保存节奏仍是原`ep %100 ==0`及final（零基epoch）；**不是optimizer/scheduler/RNG续训归档**。

[六任务脚本与Light on/off、Full训练后立即评估模板](../tran_evaluate/msar_lno/README.md)已增量更新。实际checkout根目录的无数据预览：

```bash
bash tran_evaluate/msar_lno/ns.sh train --profile light --coverage-mode floor --gpu 0 --dry-run
bash tran_evaluate/msar_lno/ns.sh train --profile light --coverage-mode off --coverage-weight 0 --gpu 0 --dry-run
bash tran_evaluate/msar_lno/plasticity.sh train --profile full --gpu 0 --dry-run
```

两个脚本均能使用所有三种设置。已有NS命令/预测回填循环不删除；核查实际源码只有10→10协议，**没有现成20/40步CLI及对应标签读取**，因此不虚构`--rollout`选项、不新增长时数据/训练实验。两时间wrapper沿既有NS64×64、Plasticity101×31合同，非默认NS downsample不列为新支持项。

实际执行（所有任务测试只运行合成张量和安全AST片段，没有import exp/main）：

```bash
MSAR_M6_RESULTS=docs/msar_lno_audit/m6/task-cases.json \
python -B -m unittest discover -s tests -p 'test_msar*.py' -v

PYTHONPATH=tests:. python -B -m unittest \
  test_temporal_standard test_kcdno_temporal test_static_standard test_kcdno_tasks \
  test_front_training test_experiment_records test_periodic_visualization -v

python -B docs/kcdno_audit/replay_subset.py temporal \
  --result docs/msar_lno_audit/m6/k0-temporal-replay.json

# 在PDE子项目cwd，绝对路径指本次本机审计产物，不是远端训练必需文件。
PYTHONPATH=/home/hwz/CDLNO python -B \
  /home/hwz/CDLNO/docs/msar_lno_audit/m0/wrapper_fixtures.py replay --project standard \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers \
  --result /home/hwz/CDLNO/docs/msar_lno_audit/m6/wrapper-replay.json

python -B docs/msar_lno_audit/m6/finalize_evidence.py
```

- MSAR **93/93通过，38.463s**，包含11个M6新方法；相关旧套件**93项，204.050s，零失败/错误、2处skip记录**（一处为子用例，均因原AirfRANS sampled路径缺torch_cluster）。不是单次全仓库测试。
- [实际任务证据](msar_lno_audit/m6/task-cases.json)：2任务×floor/off、CPU完整原loss/time batch、eval/checkpoint；另2项GPU完整时间batch、4项正式profile前向。
- 6条train + 6条eval脚本预览经真实parser校验；原PDE cwd的新进程重建两wrapper、strict加载同权重，CPU FP32 MATH `atol=rtol=0`。
- 修改前旧fixture：K0时间8份（2任务×CDLNO三模式及Transolver）+K/matched标准wrapper18份，**26份exact回放**；其中14份时间、12份静态。另修改前即时捕获4份已接受M5静态wrapper的config/inputs/weights/output，全部exact回放。旧182个夹具文件hash保持，未复制大二进制到仓库。[夹具索引](msar_lno_audit/m6/fixture-index.json)标明来源与未重跑的有效证据；不声称验证所有历史训练checkpoint。
- 首轮新测试有1个错误：将`--history-mode 2`用于“不适用字段”断言，argparse先因非法choice退出；改用合法但对MSAR不适用的`off`后通过。没有为迁就测试改生产模型或容差，保留[初次日志](msar_lno_audit/m6/initial-tests.log)。
- [环境](msar_lno_audit/m6/environment.json)：Python3.13.9、Torch2.13.0+cu130/CUDA13.0、RTX5090 Laptop、PyG2.3.1。GPU新检查是B1/d8、真实N、FP32 MATH；M2/M3既有有限AMP检查随套件重跑，**不代表M6任务AMP矩阵通过**。

## E. 旧路径与冻结区段

两完整时间入口去掉显式`args.model == 'msar_lno'`分支后AST与pre-M6精确相同，含loader、normalizer、随机collate、所有训练/eval/绘图、step位置及旧调用。factory仅去掉经精确匹配的MSAR temporal分支后与pre-M6一致。旧CDLNO/Transolver/KCDNO/matched模型数学、MSAR M1/M2/M3数学、M5静态wrapper及4个入口、工业入口和依赖均字节不变；没有移除数据loader工作。旧测试仅精确识别本次授权分支，不扩大忽略区域。

新增回归只使用保存的同一份权重、输入和backend进行比较；不把重新初始化输出当回归。所有先存用户修改保留，HEAD未变，无安装/下载/真实训练。

## F. 自审与剩余边界

已按源码及证据自审5点：时间点/通道/回填窗口；NS coverage时间均值和Plasticity独立更新；Light/Full显式优先及无卷积/无缓存；read-first/strict/eval不可覆盖；完整旧分支AST与同权重输出。未发现本阶段尚待修复的实现缺陷。

未验证：真实文件完整读取、真实数据训练/收敛/精度、远端Python3.10/Torch2.11/cu128、正式大profile时间反传与性能、任务AMP/其他backend/compile、MSAR工业PyG。原固定网格/绘图、原任务不具备完整resume、原AirfRANS torch_cluster缺失是已知边界，本轮未擅自修复或新增实验。M7的工业接口仍待单独授权。

**本M阶段结束，未执行下一阶段。**
