# LL7 — AirfRANS / ShapeNet-Car 生产接线

唯一阶段状态：**PASS（两工业任务接线与原生合成闭环完成；历史测试失败如实保留）**。只执行 LL7，未执行 LL8–LL10。真实数据/VTK指标/远端/GPU/长训练均 NOT RUN。

## A. 审计范围与真实调用链

当前仓库 `/home/hwz/CDLNO`，remote `https://github.com/hxh5159/CDLNO-w.git`，branch `main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。读取根 AGENTS、LL0路由审计、LL1配置、LL3–LL5 wrapper/core、LL6报告/状态和两任务四入口、parser、pure/history工业适配、train、数据/normalizer、field/drag/ensemble、checkpoint/provenance及相应测试。起点 [2391文件分类/size/hash](loop_linearno_audit/ll7/start-manifest.json)、status、tracked/staged diff 均保存，未reset/clean/stash/checkout/commit/push。

Air 实际链：`main.py` / `main_evaluation.py` → 原 `cdlno_entry.parse_args` 新loop guard → `cdlno.linearno_loop.industrial_entry` → 原 main 的 `model == LinearNO` 分支 → 原 pure `air_entry.run_cli` 新loop guard → `cdlno.linearno_loop.air_entry.run_cli` → `LoopedAirfRANSModel` + 原 `train.main` 的采样/Adam/OneCycle/MSE/validation/输出 → loop回调存档。评估恢复成员后调用原 `Results_test` 和 `_pressure_rL2_dataset`。

Car 实际链：`main.py` / `main_evaluation.py` → 原 `models.cdlno_run.parse_args` 新loop guard → 独立loop parser → 原 main 的 `cfd_model == LinearNO` 分支 → 原 pure `car_entry.run_cli` 新guard → loop CarRun → 原 `train.main(...linearno_run=run)` → 已验证的独立generator Car训练方法。Loop CarRun直接继承该方法，不复制训练循环；objective及evaluate/load_data/coef_norm直接复用pure函数。

四个 main/eval 文件和六个 Standard exp 完整字节不变；这比剥离新guard后的AST一致更强。完整 [增量diff](loop_linearno_audit/ll7/source.diff)、[精确投影](loop_linearno_audit/ll7/routing-projection.json)、[冻结清单](loop_linearno_audit/ll7/end-freeze.json)。

## B. 修改文件与原因

新增五个生产文件，全部集中于 `cdlno/linearno_loop/`：

- `industrial_entry.py`：两任务纯参数路由、profile字段来源、metadata恢复、独立目录与结构冲突拒绝。
- `industrial_state.py`：member初始化/独立对象、loader generator、数据metadata适配、optimizer恢复、ensemble检查和工业源码fingerprint。
- `air_entry.py`：原生Air回调、每member严格pair、成员内/间resume、原evaluation/visualization接口。
- `car_entry.py`：原Car数据/训练/损失/评估方法的loop存档适配。
- `ll7_projection.py`：只剥离准确匹配的LL7增量，保留旧pure/history/LL6 checkpoint来源hash；变更未知时失败。

已有生产文件七项增量：Air公共parser、Car公共parser各4行guard；pure Air/Car `run_cli`各3行guard；Car task provenance补精确投影；Air `train.py`最终保存点、history Car `train`最终保存点增加 `export_final` 回调，只有loop实现该回调，旧路径原 `torch.save(model,...)` 完全保留；loop provenance固定LL6源清单并将LL7增量精确投影回LL6。旧训练循环、loss与采样没有改写。

新增三个测试支持/worker及 `test_industrial.py`；仅增量调整两个已有测试projection/isolation，使其仍完整检查旧字节，不能豁免文件。更新本研究状态和本报告；没有修改八任务旧模型、loop三种core数学、LL1schema、依赖、launcher、monitor。

## C. 模型、协议与恢复合同

两工业wrapper保持LL3–LL5原实现。只组织U=P+C+S套物理block，core参数跨round共享，每visit重算Q/K/V/C/O，suffix[-1]唯一输出头。stem/placeholder和原位置处理、非持久reference/pos buffer、初始化顺序保持；两preset三mode同公共backbone参数/RNG证据与LL6一致。

| 合同 | AirfRANS | ShapeNet-Car |
|---|---|---|
| forward | Data → [N,4]，单图/可变N | (cfd_data,geom) → [N,4]，单图 |
| 特征 | 原x7 + raw pos参考距离拼接 | 原x7，geom仍为既有未使用输入 |
| rank | profile基础32，默认×2=64 | profile基础32，默认×2=64；M/head_dim必须正整数 |
| 温度 | 每物理block保留dead `temperature`，无forward/梯度用途 | 独立 `tempreature_q` / `tempreature_k`，原clamp[.1,2] |
| 默认objective | 标准化四通道 volume MSE + 1×surface MSE | 标准化全点三速度MSE + 0.5×surface压力MSE |
| 数据/评价 | 原manifest、scarce/full/OOD、20次采样验证、n_test3、原field/coefficient及pressure rL2 | 原九fold、原os.listdir点/样本顺序、GraphDataset几何抽样、原surface/drag真实sample路径 |

这里Car的0.5来自当前已批准profile，不改成Air的1。两任务的训练objective仍为MSE；物理field rL2属于原evaluation_spec，未混为训练损失。原工业OneCycle的 `(len(train)//batch+1)*epochs` 和实际epoch步数差异继续保留并校验，没有顺手“修正”。

Parser只在显式loop train或已有sidecar family=linearno_loop的eval/resume触发。新train必须显式有效LinearNO key、topology及residual；eval/resume可省略model和结构字段，从sidecar恢复。preset/custom、actual rank/multiplier、loop/任何A/K或fair-run flags、旧layers/slice_num、task/fold、profile、M/head_dim/schema冲突在构造和tensor load前拒绝。Air single-graph旧ptr在采样后允许original_N≥N且batch全0；Car必须ptr=[0,N]。非空N、浮点x、x7、pos/device/dtype等仍由原wrapper/张量层校验。

目录沿LL1 ID包含task/profile/P-C-R-S/residual/resolvedM/seed/config hash；已有目录拒绝新train。每个成员独立construct，独立module/parameter/router/optimizer；初始化seed按member明确记录于data runtime，成员0保留public seed，后续member由task/seed/member稳定hash派生，且不含residual模式。无共享core实例或跨forward/成员history。DataLoader generator按train/test固定命名顺序记录，原Python random采样仍使用并恢复原状态。

Loop只读取`weights_only=True`的安全state_dict/metadata pair；不读whole-object猜结构。原工业最终`model` / `model_<epochs>.pth`对loop输出额外裸state_dict，旧模型仍保持whole-object。权威档案是LL6的pair+SHA manifest与latest/final，strict逐键/shape/dtype/finite校验、optimizer命名参数组/shape/router/topology/scheduler进度检查、RNG最后恢复。

Air根sidecar和每member sidecar不可变；epoch archive sampler_state保存member id/历史曲线/所有RNG/generator。成员内resume读取该member最新提交档案；成员间resume恢复已完成前member的最终RNG后再开始下一member；跳序/缺前member/错member/错误manifest路径/hash拒绝。最终 `ensemble.json` 保存member id/order→state_dict/path/hash及family/config hash。eval先检查全部member，再建模严格加载；不使用外部pickle或猜最近运行，不重拟合coef_norm。

## D. 验证、实际命令与结果边界

CPU环境：Python3.13.9、torch2.13+cu130、PyG2.3.1，真实PyG Data/Batch对象；`CUDA_VISIBLE_DEVICES=''`、单线程、Agg、禁pyc。未安装或更改环境。

- 完整loop suite：**83方法，81通过、2CUDA skip、0failure/error，68.670s**。新增5方法覆盖12 wrapper组合、可变N/非法batch/ptr/shape/dtype、default M64、Car整数倍率、custom P0/C2/R3/S1、独立member对象/初始化、strict/router缺键、两任务真实parser、旧路由和原函数复用。额外实际public run_cli guard定向检查通过。七份LL2–LL6 oracle/梯度/optimizer/参数/MAC JSON与LL6完全相同。
- 工业 **12配置**：两任务×两preset×三mode。真实parser→生产adapter→原训练路径→安全pair→新进程resume/eval。Air每配置2成员，连续3epoch，对照member1内部epoch1中断及member0最终epoch边界中断；Car每配置fold3、连续3epoch对照epoch1中断。每组比较全部最终权重、optimizer/scheduler/RNG/generator、批次/随机采样序列、评估输出；最大差0。[逐例证据](loop_linearno_audit/ll7/industrial-matrix.json)。66个worker新进程；小d8/h2/M4、dropout.1，不是全宽任务训练。
- Air内存图N13–15、subsampling11，原random.sample和每次validation20遍实际执行；Car N17/23/8205，包含8201个surface点的图以触发原8192几何抽样。Car fold3保存与恢复；surface压力和三速度通过原drag_pair送入实际param3 sample路径。单独边界测试用假系数backend核验形状/路径，**不声称真实VTK drag数值通过**。
- 保存后两任务12配置的 **120项冲突**（family/residual/topology/M/head_dim/profile/hidden/A-K/fold/task等）在模型或torch.load前拒绝。[冲突证据](loop_linearno_audit/ll7/saved-conflicts.json)。
- LL6六Standard **36组合/144进程**重新执行原生合成连续/中断/恢复/评估，real N、缩小width/rank，权重/完整恢复状态/批次/输出逐位一致。[证据](loop_linearno_audit/ll7/native-matrix.json)。
- 原LL0回归153方法保持147通过、2个已批准历史失败方法/4断言、4CUDA skip、0error；结果和失败文本完全相同。失败仍为README/path.sh/旧复现矩阵历史冻结，不宣称全仓全绿。[回归](loop_linearno_audit/ll7/regression-summary.json)。
- 额外旧工业history suite **5方法：4通过、1历史指纹失败，688.606s**。通过项实际覆盖8个四组合原生闭环、4组合双成员ensemble的两类恢复边界、40正式配置及metadata提前拒绝。失败停在Standard `normalized_patch_sha256` 对比R8快照：当前 `a5359e74…`、R8 `c72a40db…`；此差异已存在于[LL0原审计](loop_linearno_audit/ll0/baseline-provenance.json)。逐任务复算表明Standard/Air/Car的源码hash均等于R8，当前patch hash均等于LL7修改前；只将只读 `git show HEAD:path` 的比较基准从当前 `5b991226` 改为R8 `d5abe014`，三套R8 patch hash全部精确复现，定位为历史Git基准变化。没有修改旧测试、golden或生产hash规则。[原失败日志](loop_linearno_audit/ll7/history-industrial.log)、[原因与完整hash](loop_linearno_audit/ll7/historical-fingerprint-review.json)。原方法被中止后的3个源码投影、6个旧launcher、8个共享函数对象与Car objective AST已另行只读核验通过；这些不能把原失败方法算作通过。
- 在任何生产修改前真实生成的Air/Car×pure/history四组档案，改后新进程resume/eval与各自原连续训练的权重/optimizer/RNG/batch/输出全等。旧whole-object兼容由旧worker实际载入自己生成的本地档案验证。保留原始外部档案 `/home/hwz/CDLNO-artifacts/loop-ll7-before-rfukh5pn`。[重放](loop_linearno_audit/ll7/pre-checkpoint-replay.json)。

原field/epoch曲线/日志/eval格式由原recorders生成，Air工业synthetic worker实际运行采样/训练/曲线绘图；VTK读取与完整Results_test系数计算只在测试边界替换，输出shape、criterion、scarce→full_test、n_test=3和成员数有明确断言。Car field evaluation真实执行，raw VTK force在worker禁用，另做上述drag接口边界检查。

中间失败全部保留：首pilot发现Air恢复缺记录器resume_from字段；第二次发现续训metadata误重写provenance.command；第三次发现JSON读取改变generator命名字典顺序、导致打包hash不同（底层状态/权重/输出相同）。分别补恢复指针、保留训练provenance并独立检查当前source、固定train/test打包顺序。定向测试初稿误写Car单一tempreature键以及把其合法0.5当负例，已按真实q/k两键和非法weight2纠正。没有改温度数学、objective、数值容差或旧golden。原失败日志pilot/pilot2/pilot3/focused保留。

在仓库根执行的主要验证命令：

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tests:$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -B docs/loop_linearno_audit/ll7/capture_legacy.py  # 本轮在生产修改前执行；原档案保留
python -B docs/loop_linearno_audit/ll7/replay_legacy.py
python -B docs/loop_linearno_audit/ll7/run_industrial.py
python -B docs/loop_linearno_audit/ll7/run_native_matrix.py
python -B docs/loop_linearno_audit/ll7/run_existing_regressions.py
LINEARNO_R8_REPORT=docs/loop_linearno_audit/ll7/history-industrial.json \
python -B -m unittest linearno.test_history_industrial -v
python -B docs/loop_linearno_audit/ll7/check_historical_fingerprint.py
python -B docs/loop_linearno_audit/ll7/check_saved_conflicts.py
```

完整loop套件（report必须指向LL7，避免写入旧阶段证据）：

```bash
LOOP_LL2_ATTNRES_REPORT=docs/loop_linearno_audit/ll7/attnres-report.json \
LOOP_LL2_BODY_REPORT=docs/loop_linearno_audit/ll7/body-report.json \
LOOP_LL3_PARITY_REPORT=docs/loop_linearno_audit/ll7/sr-parity.json \
LOOP_LL3_ACCOUNTING_REPORT=docs/loop_linearno_audit/ll7/accounting.json \
LOOP_LL4_REPORT=docs/loop_linearno_audit/ll7/rb-report.json \
LOOP_LL5_REPORT=docs/loop_linearno_audit/ll7/lb-report.json \
LOOP_LL5_MODES_REPORT=docs/loop_linearno_audit/ll7/modes-report.json \
python -B -m unittest discover -s tests/loop_linearno -p 'test_*.py' -v
python -B docs/loop_linearno_audit/ll7/finalize_evidence.py
```

capture/replay和一次性apply_routing/build_workers脚本是本轮实际执行证据，不应在已完成的生产树再次运行apply或覆盖原before档案。

## E. 冻结与自审

七个既存生产文件只有准确可逆的路由/安全保存/provenance增量；纯模型数学、industrial wrapper、三core、数据、loss、metric、采样、fold/ensemble含义不变。LL7变化后，Air pure、Car pure、Air history、Car history及LL6 Standard loop两种来源hash都与修改前精确相等。Loop工业档案则hash全部真实工业新源码及复用科学函数，不能把兼容投影用来隐藏新family代码变化。

最终全量2391既存文件按classification/size/SHA-256逐项检查；允许变动仅七生产文件、两个测试projection/isolation和本研究STATUS，共10项，其余2381保持。新ignored仅本阶段日志，模型权重与合成运行放仓库外。四工业main/eval+六Standard exp字节一致，HEAD/tree/staged diff不变，git diff --check通过。

五项优先自审：wrapper原生位置/温度/初始化；真实训练/损失/指标函数复用；metadata先读与strict结构/optimizer拒绝；成员内/间随机流及独立对象；旧pure/history/LL6档案与完整冻结。通过证据见[交付审查](loop_linearno_audit/ll7/delivery-review.json)，无新设计冲突需要用户裁定。

## F. 未运行与风险边界

真实VTK加载与完整drag/lift/采样统计、真实训练、收敛/精度/SOTA、远端Python3.10/torch2.11+cu128、GPU/AMP/compile、全宽工业训练均未运行。合成有限图不能替代真实指标与性能验收；原采样/OneCycle协议怪癖继续保留。专门八任务loop launcher、公平实验命令总表、逻辑visit monitor属于后续阶段，本轮未执行。新工业检查只做CPU，不把可用torch/PyG当作远端验收。旧回归共3个失败方法（原153中2个，额外history中1个）均有早于本阶段的内容/hash证据；它们保留，不宣称旧套件全通过。

本 LL7 阶段结束，未执行下一阶段。
