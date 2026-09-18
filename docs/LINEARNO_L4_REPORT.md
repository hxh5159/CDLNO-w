# LinearNO L4 — 四个静态 Standard 任务接入

**状态：PASS。** 本阶段只接入 Airfoil、Darcy、Elasticity、Pipe，并在本轮完成各题原生入口的合成 train→checkpoint→新进程 resume→新进程 eval 闭环。NS、Plasticity、AirfRANS、ShapeNet 的生产入口未改；没有执行 L5。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`，没有真实数据读取、下载或完整训练。

## A. 实际入口、配置和文件

目标仍为 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`。修改前 1,278 文件快照保存在 `/home/hwz/CDLNO-artifacts/linearno-l4-before-ti1ycp32/source`；tracked/untracked/ignored 及原有 LinearNO L0–L3 代码均纳入 [baseline](linearno_audit/l4/baseline.json)。没有 reset/clean/stash、改写历史、安装、commit/push。

| 任务 | 实际入口 / launcher 名 | 显式 factory key | 默认 Model 参数 | 训练语义 |
|---|---|---|---|---|
| Airfoil | `exp_airfoil.py` / `airfoil_train.sh`,`airfoil_eval.sh` | `LinearNO_Structured_Mesh_2D` | conv_temp/M64/d128/L8/h8/ratio1/H221/W51/fun0/out1/unified off | 原 channel4、first1000/next200、无normalizer；batch4，OneCycle每batch |
| Darcy | `exp_darcy.py` / `darcy_train.sh`,`darcy_eval.sh` | 同上 | conv_temp/M64/d128/L8/h8/ratio1/H85/W85/fun1/out1/unified off | 原 stride5、smooth1 train/smooth2 test、input/output UnitTransformer、decode后loss；batch4，OneCycle每batch |
| Elasticity | `exp_elas.py` / `elasticity_train.sh`,`elasticity_eval.sh` | `LinearNO_Irregular_Mesh` | temp/M64/d128/L8/h8/ratio1/fun0/out1/unified off | 原972点、xy/sigma permute、first1000/last200、输出normalizer，fx=None；batch1，Cosine每epoch |
| Pipe | `exp_pipe.py` / `pipe_train.sh`,`pipe_eval.sh` | `LinearNO_Structured_Mesh_2D` | conv_temp/M64/d128/L8/h8/ratio1/H129/W129/fun0/out1/unified off | 原channel0、first1200内first1000/last200、原坐标/output normalizer；batch4，OneCycle每batch |

两种 key 返回同一个已接受的 `model.LinearNO.Model`，没有复制数学实现或更改 L2/L3 原语。正式参数量仍为 Darcy1,766,145、Elasticity585,217、Airfoil/Pipe各1,765,889。历史默认 `Transolver_1D/2D` 没有擅自修复/注册，新 launcher 显式传有效 key。旧 key、默认、save_name 和旧 strict=False 分支保持。

| 文件 | 修改说明 |
|---|---|
| `PDE-Solving-StandardBenchmark/model_dict.py` | 仅新增两个独立 LinearNO key，返回真实 Model 模块 |
| `PDE-Solving-StandardBenchmark/cdlno_entry.py` | 仅在选中上述 key 时调用新 parser；非四任务明确拒绝 |
| `PDE-Solving-StandardBenchmark/linearno_entry.py` | 新增薄导出，无入口/data import副作用 |
| 四个 `exp_*.py` | 仅受新 key 条件保护的解析、normalizer恢复、Model kwargs、mean-loss/log权重、训练记录/存档/续跑分支；旧 else 逐段保留 |
| `cdlno/linearno/standard_entry.py` | 新增 `parse_args/constructor_kwargs/model_kwargs/start/verify_data/normalizer/StandardRun`；任务数学和训练循环仍在真实 exp |
| `cdlno/linearno/checkpoint.py` | 新增 metadata-first 校验、strict state、完整epoch pair和RNG恢复；复用原 `_atomic`、`_model_state_check`、`restore_rng`，不改旧存档helper |
| `cdlno/linearno/profiles.py` | 可选 `standard_static_l4` integration_contract；不带此参数的L1 24组事实配置完全保持；新决定逐字段标记 integration_contract |
| `cdlno/experiment.py` | 新family目录/计数适配，resume追加日志且保留config，复用现有记录和PeriodicFields；旧family完整AST投影相同 |
| `tran_evaluate/linearno/` | 新增4×train/eval共8脚本、独立dispatcher与说明；旧81个shell字节不改 |
| `tests/linearno/static_worker.py`,`test_static_integration.py`,`tests/linearno_entry_projection.py` | 实际main AST合成测试、loss/point order/负向测试、精确移除新guard恢复旧完整AST |
| `tests/msar_entry_projection.py`,`tests/linearno/test_legacy.py` | 旧测试识别本轮授权分支；恢复完整旧AST再比较，其余字节检查继续保留，没有放宽数学容差 |
| 本报告、STATUS、REPRODUCTION_MATRIX增量和 `linearno_audit/l4/` | 实际命令、结果、来源差异和冻结证据；旧报告/历史证据保留 |

## B. 明确的来源/协议决定

本阶段最新用户要求是“逐样本 flatten、无epsilon后平均”。因此新family训练使用原 `TestLoss(size_average=True)`：

```text
r_i = ||flatten(pred_i-target_i)||_2 / ||flatten(target_i)||_2
Lbatch = mean_i(r_i), 无 epsilon
epoch metric = sum_batches(Lbatch * actual_batch_size) / sample_count
```

**与发布代码的真实区别不能隐藏**：固定发布代码 `TestLoss(size_average=False)` 在 backward 前按batch求和；其epoch显示除以ntrain得到样本平均。两种算法的梯度相差B倍。本轮按最新明确要求在两个接入profile都采用batch mean，不把这个改动归因于论文证明。`objective_spec.reduction`、`integration_contract` 和 `data_spec.protocol_decisions` 明确记录；原L1 release事实表仍保留batch sum，旧模型训练仍为原batch sum。当前 `official_release` 是发布模型/超参数加本次确认的接入合同，**不宣称逐字复现发布训练程序**。这不是一个被遗漏的隐含配置。

其他已接受的 hxh 边界同样显式记录：Airfoil保持原每epoch验证（发布每10epoch/末epoch）；Pipe保持first1200再切分（发布从整个文件取last200，只有长度1200时等价）。eval/test batch使用所选profile的1；数据值、样本序、decode和指标聚合不变。当前没有数据，Pipe文件总长度和真实split等价性仍未验证。

Darcy公式保持为 `L = mean(rL2(decoded_pred,decoded_y)) + 0.1*(mean(rL2(Dx pred_zero_boundary,Dx y)) + mean(rL2(Dy pred_zero_boundary,Dy y)))`。**field-rL2在裁边前计算**；仅导数分支将预测crop interior再zero-pad，真值不被裁边。`central_diff`原函数及zero-padding完全不改，`dx=1/85`。eval仍为原decode后field rL2，不能误把训练导数边界操作带入eval。

官方 `Standard_PDE_Benchmark/exp_darcy.py:45` 为全局 `epochs=500`，scheduler使用这个全局值，训练循环使用 `range(args.epochs)`。新 `official_release` 对非500 `--epochs`在数据读取前拒绝；默认500一致。paper合同使用当前hxh解析后的epochs同时构建循环和scheduler。`data_spec.scheduler`、config.protocol均记录 `epochs/steps_per_epoch/total_steps/state_at_construction`；每次保存检查实际scheduler.last_epoch与已完成epoch、batch数一致。

## C. 原生存档和resume/eval合同

没有把现有 CDLNO `TrainingArchive` 硬装到不兼容的新配置上；新增 `LinearNO StandardRun` 使用 hxh记录器/路径/可视化与其已验证的atomic/RNG底层工具。真正调用在四个原exp内，合成验证执行的也是这些原生调用，不是另建train.py复制训练循环。

1. parser先选择family/profile，只应用**显式**CLI值；新family不会受旧 n_hidden/layers/slice/batch 默认污染。kwargs只有真实构造器字段，variant/rank不发给旧模型。
2. 新训练在读取数据前保留独立目录并创建现有config/log；数据SHA256流式读取并记录。初次训练按原UnitTransformer拟合；保存命名input/output的mean/std、dtype/shape/hash和train fit来源。eval/resume从metadata恢复数值，`__init__`不调用，不重拟合。
3. `architecture.json`为不可变初始完整metadata。epoch存档含 resolved family/profile/variant/actual M、全部model_spec、独立objective/evaluation/data/normalizer/provenance、epoch/global_step、optimizer、scheduler、Python/NumPy/Torch CPU/CUDA RNG、显式loader generator及sampler合同。原loader没有独立generator，因此映射为空并记录实际RandomSampler/SequentialSampler及epoch边界；全局Torch RNG负责shuffle。
4. `checkpoints/epoch_XXXX.pt`、`weights/epoch_XXXX.pt`、`epoch_XXXX.metadata.json`通过三文件哈希manifest提交后才更新latest/final，保留hxh便利`model.pt`。没有unknown-key丢弃或随机补权重；所有新family权重加载逐键 `strict=True`，先检查key/shape/dtype/finite。
5. eval默认final，先读并核验metadata/profile/spec/数据checksum，再构造Model。错误显式profile/variant/M拒绝；`--checkpoint epoch_XXXX`仅是明确诊断。eval目录唯一，训练config/sidecar字节保持。
6. resume默认latest，拒绝回退、总epochs/结构/protocol/源码不一致；恢复完整optimizer/scheduler，在所有构造/加载之后最后恢复RNG。新训练不允许复用现有目录，已完成final不续加epoch。旧裸权重/whole-object没有新schema，不会被猜成LinearNO或伪装可resume。
7. 保留入口原 `ep % 100 == 0` 与final保存时机，实际完成epoch1/101/201/301/401和500时提交。不是承诺每一步可恢复；保存间隔内的更新在中断后重算。resume追加原日志/曲线，不删除已有文件；若日志超出最后存档epoch，恢复后的重算epoch会追加，不能把旧未提交日志当恢复依据。本轮验证的中断点为已提交epoch1。

checkpoint schema采用 `weights_only=True` 加载本地张量/基本结构，不接外部pickle；原工业可信pickle边界没有扩大。`normalized_patch_sha256`根据相关真实源码AST前后形成，包含未跟踪LinearNO新文件，不能只靠tracked git diff遗漏新源码。模型/数据数值部分没有新增研究机制。

## D. 本轮真实执行证据

环境：Python3.13.9，torch2.13.0+cu130，timm1.0.28，einops0.8.2，PyG2.3.1，RTX5090 Laptop可用。四静态闭环**CPU执行**，`CUDA_VISIBLE_DEVICES=''`；本轮L2/L3既有模型parity重跑使用本地CUDA，不等于四任务GPU训练验收。未安装依赖，远端Python3.10/Torch2.11/cu128未运行。见 [environment](linearno_audit/l4/environment.json)。

| 检查 | 实际结果 | 边界 |
|---|---|---|
| L4新增集成suite | 7方法通过，193.116s | 内含4任务×4新进程角色、8任务/profile解析、16launcher预览、loss/顺序/冻结等子项 |
| 原生四题闭环 | 各3个合成epoch、每epoch2个batch，共6训练更新 | d8/h2/L2/M4显式测试override；真实空间N：Darcy7225、Elasticity972、Airfoil11271、Pipe16641；训练4样本/test2样本 |
| 连续3epoch vs 1epoch保存+新进程2epoch | 四题weights、optimizer、scheduler、RNG、batch顺序逐位一致 | `final-integration.json`，没有容差放宽；验证的是保存边界恢复 |
| 新进程eval | 四题prediction hash与连续/续跑一致；实际original eval指标写入现有eval目录 | 断言normalizer constructor不能调用；config/architecture文件hash不改 |
| 周期可视化/记录 | 四题final可视化status completed，history epoch为[1,2,3] | 复用原PeriodicFields，未改频率/图形逻辑；standalone eval showcase绘图在CPU夹具中明确关闭 |
| 28负向检查 | 全部通过 | 4题×profile/variant/rank/family/缺键/额外键/shape；保持构造器signature的spy证明metadata冲突先于构造 |
| L1旧Transolver/schema/RNG | 全部16方法通过 | 八个原模型同权重/固定输入/checkpoint；旧parser默认；可视化保存/新进程resume随机流 |
| L2 attention / L3完整模型 | 14+10方法通过 | 固定官方parity、独立oracle、严格key/grad/step、原有GPU/AMP原子检查；本轮新数值保存在l2/l3-parity.json |
| 旧静态入口和实验工具 | 2静态方法+9记录方法执行；修正测试投影后失败项已重新通过 | 1个旧Air完整采样epoch子案例因缺torch_cluster跳过；Car真实PyG合成epoch和Air weighted-loss记录片段通过 |
| 最终静态复核 | 6方法通过，15.309s | 四exp/parser/factory/记录器全部AST、原Transolver/core字节、旧launcher静态 |
| launcher/shell | 16新preview通过；原81个tracked shell字节相同且bash -n通过 | 未执行真实launcher训练 |

本轮最终无未解决测试失败。保留初轮原始日志：先后修正的均为新测试夹具问题（遗漏Darcy全局epochs、NumPy RNG比较方式、AST投影意外简化无关常量条件、负向spy改变signature）。对应失败与重跑关系在 [results-summary](linearno_audit/l4/results-summary.json) 明列，没有删除失败日志或把初轮失败伪装成原模型缺陷。

严格的“真实main”证据范围：`static_worker.native_main`只替换原main开始的数据读取prefix为显式内存合成张量、去掉`.cuda()`传输、将standalone showcase设0；之后实际normalizer分支、DataLoader、模型选择、optimizer/scheduler构造、训练/validation/eval循环、保存/续跑/记录均执行原AST。不是将随机张量写成同名数据文件，也没有import exp触发真实数据。四题正式宽度完整训练、真实loader、正式500epoch、真实字段精度均未执行。

实际可重跑命令（repository root，外部固定官方源码须保留；新证据请使用新路径）如下，详见 [commands](linearno_audit/l4/commands.json)：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 \
python -B -m unittest linearno.test_static_integration -v

PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  linearno.test_profiles linearno.test_schema linearno.test_legacy linearno.test_rng \
  linearno.test_attention_parity linearno.test_attention_structure \
  linearno.test_standard_model linearno.test_standard_structure \
  test_static_standard.StaticFrozenChecks test_experiment_records -v
```

正式训练/评估命令已分别写在 [四任务launcher说明](../tran_evaluate/linearno/README.md)。分别设置 `PROFILE=paper_table8_on_release_model` 或 `PROFILE=official_release` 后，可对四个 train脚本传 `--linearno-profile "$PROFILE" --gpu 0 --seed 0 --experiment-dir "$RUN_ROOT/<task>"`；对应eval脚本从同一目录恢复。示例不依赖不存在的CLI参数或旧默认factory，也没有将发布`eval=1`脚本冒充训练。

## E. 冻结、映射和自审

[delivery-check](linearno_audit/l4/delivery-check.json)基于L0和L4的**内容+分类+size**清单核验。L0广义221文件清单中，仅本轮授权的 `cdlno/experiment.py` 和测试投影helper发生新增；原Transolver模型/Physics-Attention/Embedding全部字节相同，原 `LINEARNO/**`/monitor从L0起不存在，仍为N/A。L0原始全清单内的改动仅为本轮明确列出的入口/factory/parser/记录及测试helper；对应新guard去除后旧完整AST相同。L1/L2/L3数学源和旧数据/loss/normalizer/optimizerhelper不改。没有新增ignored产物，真实大文件和所有合成存档保存在目标外，用户原有untracked/ignored未丢失。

已完成五点自审：

1. 默认task→variant/M/H/W和Pipe ratio/batch，旧parser默认不会覆盖新profile；只有四任务获准接线。
2. Darcy decode→field loss→derivative专用边界处理→zero-padding dx=1/85，batch mean的数值/日志聚合与发布差异显式记录。
3. 先metadata后构造、无normalizer重拟合、strict key/shape、final角色、scheduler真实步数与最新commit边界。
4. 原生新进程闭环的全部状态和RNG逐位相同，视觉诊断没有改变这条链；没有只保存权重就声称resume。
5. 源码AST与基线hash保护、测试投影范围、尚未运行环境/真实数据验收的声明边界。

## F. 未执行/后续边界

NOT RUN：真实loader一批、真实数据checksum/完整性和Pipe总长度、真实normalizer fit结果、真实训练/minirun/500epoch、论文收敛/精度、实际epoch时长、四静态任务GPU/AMP/remote resume、外部官方checkpoint转换。当前合成数据输出不是论文准确率。多进程/持久worker/分布式数据加载不属于现有这四入口，本轮未新增或宣称恢复支持。

工业paper objective/force协议既有未决项不影响本轮，仍留待对应阶段；没有自行扩到NS/Plasticity或工业任务。**本L阶段结束，未执行下一阶段。**
