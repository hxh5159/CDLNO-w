# LinearNO L5 — NS 与 Plasticity

**状态：PASS。** 本轮只接入两个时间任务，复用已接受的 Standard `model.LinearNO.Model`。六个 Standard 任务现已接入；AirfRANS、ShapeNet 的生产入口留待后续。本轮 `RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`，不读取真实数据、不启动真实训练。

## A. 基线、范围与实际改动

目标为 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`。本阶段开始时已有 L4 dirty/untracked 文件全部保留。1,323 文件分类/大小/hash基线和源码快照见 [baseline](linearno_audit/l5/baseline.json)，快照位于 `/home/hwz/CDLNO-artifacts/linearno-l5-before-uh7r5hxj/source`。没有 commit/push/reset/clean/stash、依赖安装或数据下载。

| 文件 | 本轮变化与边界 |
|---|---|
| `PDE-Solving-StandardBenchmark/exp_ns.py` | 独立 LinearNO 构造、profile评估batch、记录/checksum、完整存档/续跑；10步原训练/评估循环、loss表达式不改 |
| `PDE-Solving-StandardBenchmark/exp_plas.py` | 同上及输入normalizer数值恢复；原collate、meshgrid、时间循环、loss表达式不改 |
| `PDE-Solving-StandardBenchmark/linearno_entry.py` | 更新六任务说明；仍为薄导出 |
| `cdlno/linearno/standard_entry.py` | 支持两题CLI解析；NS stride1，Plasticity完整MAT路径；分开optimizer和scheduler进度，元数据记录时间合同，provenance纳入两实际入口 |
| `cdlno/linearno/profiles.py` | 新增 `standard_temporal_l5` 接入合同，只记录设备与resolved scheduler epochs；保留时间任务batch sum，L4合同及无合同的L1事实配置不改 |
| `tran_evaluate/linearno/_static.sh`、新增4个时间任务train/eval脚本 | 实际入口映射、显式有效factory key、已有路径环境变量；旧Transolver脚本不改 |
| `tests/linearno/temporal_worker.py`、`test_temporal_integration.py` | 原main AST完整时间循环合成执行，固定输入、真实N、过程hook、独立loss核对、原生新进程闭环 |
| `tests/linearno/test_static_integration.py`、`test_legacy.py` | 更新获授权任务清单，仍对旧分支完整AST和其余文件bytes作比较 |
| `tests/test_front_task_modes.py` | 仅将旧CDLNO构造分支查找从main.body改为ast.walk(main)，适配新增最外层family分支；不替换构造器或放宽断言 |
| 本报告、STATUS、REPRODUCTION_MATRIX增量、launcher README和l5证据 | 实际命令、结果、来源差异和未执行边界 |

L2/L3模型/attention、六任务factory、四个静态exp、checkpoint/schema数值序列化、experiment/可视化实现、所有工业生产文件均未在本阶段修改。

## B. 公式与真实配置映射

| 合同 | 实际符号/路径 | 独立证据 |
|---|---|---|
| NS plain，无temperature，d256/L8/h8/M32/ratio2 | profiles→`standard_entry.constructor_kwargs`→`model.LinearNO.Model`，显式 `LinearNO_Structured_Mesh_2D` key不决定variant | 两profile实际完整构造、state中无温度参数、参数量 **3,377,921** |
| NS unified on/ref10、10→1帧 | Model替换位置语义复用L3；实际 `exp_ns.main` 每次 `model(x,fx)` | x[B,4096,2]、fx[B,4096,10]→[B,4096,1]；旧ref8/ratio1默认不覆盖profile |
| NS训练/评估反馈 | 原训练 `fx=cat(fx[...,1:],y)`；eval改用im | 每个真实forward前hook逐元素检查完整窗口，十次调用都检查，无未来标签替换、无跨时间缓存 |
| NS loss/步进 | 原 `TestLoss(size_average=False)` 每样本flatten，无epsilon，batch求和；`sum(t=0..9,L_t)`后一次backward/optimizer/scheduler | 独立vector_norm比值累加与实际backward张量核对；3epoch×2batch：60train forward、6backward/optimizer/scheduler |
| Plasticity conv，无temperature，d128/L8/h8/M64/ratio1/H101/W31/Time_Input=True/out4 | 同一Standard Model，T传入实际time_fc | 两profile完整构造，参数量 **1,799,428**；B2/T[B,1]、变T变输出、time_fc及T梯度有限且非零 |
| Plasticity原点序 | 原field flatten、`np.meshgrid(x,y)`、Conv2d reshape(H101,W31) | 逐点tag和卷积prehook核对，不将meshgrid的31×101布局“修正”为field顺序 |
| Plasticity时间采样 | 实际 `random_collate_fn` 中每样本 `torch.randperm(20)` | 保留原函数AST；固定Torch seed独立生成两个排列，检查T和对应label同步；B2样本排列不同 |
| Plasticity步进 | 原20次各自forward/backward/optimizer、outer batch一次scheduler | 3epoch×2batch：120train forward/backward/optimizer、6scheduler；optimizer内部step张量也确认为120 |

两任务训练objective保持原 **batch sum**。这与用户在L4专门要求的四静态任务batch mean不同，不能把L4的缩放改动套到时间任务。NS train显示mean step与flattened trajectory rL2；eval正式路径报告full rL2。Plasticity train显示mean step，validation/eval显示mean step与full rL2。测试用独立 `torch.linalg.vector_norm` 公式核对被backward的值，没有替换原loss。

正式超参数：两任务500epochs/AdamW/lr0.001/weight_decay1e-6；NS batch2、无clip；Plasticity batch8、clip1；两者eval batch1。OneCycle的epochs取实际args.epochs、steps_per_epoch取outer DataLoader长度，两者循环及scheduler一致；Plasticity不得把scheduler步数乘20。显式新family结构/超参覆盖保留来源记录；正式参数没有为测试缩小。

### 已审查过的来源差异 C23

提示词写“NumPy每batch排列20个时间”，但目标 `exp_plas.py:random_collate_fn`、固定官方 `Standard_PDE_Benchmark/exp_plas.py:random_collate_fn` 均为**逐样本Torch排列**。L0 REPRODUCTION_MATRIX C23、REFERENCE_AUDIT §八任务合同已明确这一事实，且L0已经审查通过。本阶段以“不修改时间任务语义”为边界保留原函数，没有新增NumPy排列。NumPy仍属于完整RNG存档；合成resume前故意扰动NumPy seed，再验证恢复后的完整状态和下一组随机数与连续运行一致。

## C. 原生checkpoint、resume、eval

沿用L4完整metadata/schema与epoch-pair机制，未改 `checkpoint.py`：

- parser先读取manifest/metadata并校验family、task、profile、variant、actual M等，才构造真实Model；错误profile/variant/M在构造前拒绝。
- `strict=True` 加载，缺键/多键/shape错误拒绝；训练normalizer的mean/std数值写入存档，Plasticity eval/resume不重新fit。
- `model_spec/profile_spec/objective_spec/evaluation_spec/data_spec/normalizer_spec/provenance_spec/resume_state`完整记录；真实文件checksum按实际loader路径计算。NS是fno目录，Plasticity是完整MAT路径。
- `resume_state.global_step`明确按optimizer更新计数。NS `epoch × outer_batches`；Plasticity `epoch × outer_batches × 20`。scheduler进度仍为`epoch × outer_batches`，在保存与恢复时分别核验。`data_spec.temporal_progress`记录两者口径。
- optimizer/scheduler加载后，RNG最后恢复。Python/NumPy/TorchCPU/CUDA和已有generator/sampler状态均沿用L4协议；不保存跨真实时间latent。
- 旧保存时机ep%100==0及final不改：正式500epoch在完成1/101/201/301/401/500时提交；只在完整epoch边界resume，未存档的更新会重算。eval默认final，不按test选best。
- eval不重写训练config/architecture；resume追加原记录。和L4一样，源码hash变化的旧实验会明确拒绝resume，不能用新版本冒充完全连续轨迹；纯eval不因源码hash变化拒绝同结构权重。本轮接入后四静态题用当前源码重新完成了闭环。

本轮两题真实main的合成证据范围：只将main的真实数据读取prefix替换为标记的内存张量，去掉`.cuda()`传输、关闭standalone showcase；后续原collate、normalizer、DataLoader、模型构造、optimizer/scheduler、全部20/10步循环、原loss、存档和PeriodicFields均实际执行。没有建立同名假数据文件，没有import原exp触发真实读取。

## D. 实际验收和环境

环境：[environment](linearno_audit/l5/environment.json)。本地Python3.13.9/Torch2.13+cu130/RTX5090 Laptop；两时间任务及四静态闭环均在CPU。既有L2/L3测试中的CUDA/AMP原子parity实际重跑，但不等于本轮时间任务GPU/远端验收。没有安装或变更环境。

| 检查 | 结果 | 边界/证据 |
|---|---|---|
| L5静态/数学/CLI | 4方法通过，1.662s | 正式两profile参数量、T梯度、历史点序、原collate/完整旧AST、8新launcher预览 |
| L5两题原生闭环 | 1方法通过，143.287s | 两任务各4新进程角色；N4096/3131、train4/test2、B2、d8/L2/h2/M4、3epoch；[native](linearno_audit/l5/native.json) |
| 连续3epoch vs 1epoch保存+新进程2epoch | 权重/optimizer/scheduler/Python/NumPy/Torch RNG、batch顺序、Plasticity全部T查询逐位相同 | 没有容差放宽；只验证已提交epoch边界 |
| 新进程eval/产物 | prediction hash一致，正式完整时间指标与最后validation逐值相同；两题PeriodicFields completed | [output-records](linearno_audit/l5/output-records.json)；NS6/Plasticity120真实optimizer step与scheduler6计数一致 |
| 14负向检查 | 通过 | 两题×profile/variant/M/family/缺键/额外键/shape，真实新存档，先metadata后构造 |
| L4四题集成回归 | 7方法通过，189.012s | 四题再次原生train/save/resume/fresh eval、真实空间N、小模型、原loss、16launcher预览；[static-integration](linearno_audit/l5/static-integration.json) |
| L1全部旧Transolver/profile/schema/RNG | 16方法通过 | 八旧模型固定输入/同权重、state、参数量、freshcwd往返；旧默认与随机流不改 |
| L2+L3 | 14+10方法通过 | 独立oracle/固定官方同权重parity、完整模型、梯度/step/strict keys；[L2](linearno_audit/l5/l2-parity.json)、[L3](linearno_audit/l5/l3-parity.json) |
| 旧静态/时间/记录 | 2+10+9方法执行，最终受影响项通过 | 首轮记录测试的6个子案例因旧测试AST定位报错，修正后单方法全部24任务模式重跑通过；不是模型数值失败 |
| 受影响回归补跑 | 10方法通过，约67.5s | 8项旧front task modes、1项全部24任务模式记录/往返、1项全文件冻结；[结果](linearno_audit/l5/affected-regression-results.json) |
| 既有monitor | N/A | L0起`LINEARNO/**`在实际checkout不存在；没有虚构monitor通过 |

初轮原始日志和6个AST定位错误保留在 [regression](linearno_audit/l5/regression.txt)。唯一修正是测试helper查找嵌套CDLNO分支；全部旧分支AST和模型字节另行验证。当前无未解决失败。一个既有AirfRANS完整采样epoch子案例因缺torch_cluster跳过；Car真实PyG合成epoch及Air weighted-loss记录片段通过，不将跳过项宣称通过。

完整真实调用/命令与回放环境：[commands](linearno_audit/l5/commands.json)。推荐在repo root无数据复核：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 \
python -B -m unittest linearno.test_temporal_integration linearno.test_static_integration -v

PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  linearno.test_profiles linearno.test_schema linearno.test_legacy linearno.test_rng \
  linearno.test_attention_parity linearno.test_attention_structure \
  linearno.test_standard_model linearno.test_standard_structure \
  test_static_standard.StaticFrozenChecks test_temporal_standard \
  test_experiment_records test_front_task_modes -v
```

新测试产物应使用新的外部目录，不覆盖本轮接受证据。两题存档在 `/home/hwz/CDLNO-artifacts/linearno-l5-native-yhs6wuks`；四静态回归在 `/home/hwz/CDLNO-artifacts/linearno-l5-static-regression-af4pqkxa`。

## E. 冻结与自审

[delivery-check](linearno_audit/l5/delivery-check.json)逐文件检查L5前snapshot与L0：tracked/untracked/ignored分类、size和SHA256；原Transolver模型/Physics_Attention/Embedding及所有benchmark模型文件、本轮之前的LinearNO数学模块/存档实现字节未变。两时间入口去除新增family guard后与修改前**完整AST一致**。L4四个exp、factory、旧数据/loss/normalizer模块不改；新profile参数只到新family。

L0广义221文件清单累计变化为`cdlno/experiment.py`、`tests/msar_entry_projection.py`（均L4）与本轮`tests/test_front_task_modes.py`。这些授权接线/测试适配不冒称为整个清单字节不变；其余冻结项保持。没有新增ignored/cache、用户文件丢失或旧launcher语义变化。

最终清单发现阶段进行期间新增了未跟踪的 `.claude/settings.json`，本阶段工具没有创建或改写它，来源不作猜测。单独在 [concurrent-files](linearno_audit/l5/concurrent-files.json) 记录分类/大小/hash并保留，未将它冒算为本阶段允许新增源码；冻结脚本检查这个确切内容而非豁免整个目录。

完成五点自审：

1. 正式profile/参数量、plain/conv无温度、NS ref10/ratio2与旧默认隔离。
2. NS每步窗口/10步loss、Plasticity逐样本Torch排列/原点序/T梯度；没有换随机源或合并时间为空间。
3. optimizer20:scheduler1、metadata计数、所有RNG恢复、normalizer无refit、严格checkpoint与新进程完整指标。
4. 六Standard闭环和旧Transolver/时间/实验记录回归、测试定位错误修复有原始日志而非隐藏失败。
5. 冻结源/hash与合成/真实、CPU/GPU、local/remote证据边界。

## F. 未验证项与阶段边界

NOT RUN：真实数据loader一批/数据完整性/checksum、真实训练/mini-run/500epoch、论文收敛和精度、真实epoch时间；两时间任务GPU训练/AMP/compile与远端Python3.10/Torch2.11/cu128的完整存档闭环；外部官方checkpoint转换。实际数据协议保持不表示数据文件已经验收。工业任务仍留待L6/L7；不提前接线。

L0 C23已确认的Torch/NumPy描述差异按源码保留，无新增待裁定架构或协议冲突。**本L阶段结束，未执行下一阶段。**
