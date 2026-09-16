# MSAR-LNO M7：ShapeNet-Car / AirfRANS

2026-09-17，仅M7。M6已审查通过。基于`main@9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`的实际未提交工作增量实现；[M0接口审计](MSAR_LNO_REFERENCE_AUDIT.md)、[M6状态](MSAR_LNO_M6_TEMPORAL_TASKS.md)和当前工业源码共同确定接入位置。根AGENTS适用；没有其他子目录AGENTS。未执行M8、真实数据训练、依赖安装、commit/push或PR。

## A. 完成范围和八任务覆盖

两个工业任务均可通过原main/main_evaluation选择`msar_lno`。默认Light、显式Full，复用同一已接受MSAR core；原模型和旧入口默认值不变。新增wrapper只逐点lift/head，不使用卷积或图边，但保留loader/训练/评价中的全部图构造、采样、字段及后处理工作。原单图batch1保持，多图输入明确拒绝。

| 任务 | wrapper合同 | 静态/选择/参数流 | 合成原loss训练步与eval | checkpoint | 真实PyG | GPU/真实数据 |
|---|---|---|---|---|---|---|
| Darcy | `x[B,N,2], fx=None → [B,N,1]`，原输入/输出normalizer与decode | M5通过，本轮MSAR套件重跑 | M5 floor/off通过，本轮重跑 | 原严格state_dict，通过 | 不适用 | 既有有限GPU随套件重跑；真实数据未执行 |
| Elasticity | `x[B,N,2], fx=None → [B,N,1]` | 同上 | 同上 | 同上 | 不适用 | 同上 |
| Airfoil | `x[B,N,2], fx=None → [B,N,1]` | 同上 | 同上 | 同上 | 不适用 | 同上 |
| Pipe | `x[B,N,2], fx=None → [B,N,1]` | 同上 | 同上 | 同上 | 不适用 | 同上 |
| NS | `x[B,4096,2], fx[B,4096,10] → [B,4096,1]` | M6通过，本轮重跑 | 原10次真值回填训练/预测回填eval通过，本轮重跑 | 原严格state_dict，通过 | 不适用 | 有限原10步GPU随套件重跑；真实数据未执行 |
| Plasticity | `x[B,3131,2], fx[B,3131,1], T[B,1] → [B,3131,4]` | M6通过，本轮重跑 | 原20次独立时间更新通过，本轮重跑 | 原严格state_dict，通过 | 不适用 | 有限20时间点GPU随套件重跑；真实数据未执行 |
| ShapeNet-Car | `forward((cfd_data,geom_data)) → [N,4]`，velocity3/pressure1 | M7通过 | floor/off原loss+Adam/OneCycle步骤通过；两种模式完整合成epoch、原test通过 | 整对象、负向、重复eval、新cwd通过 | 真Data/Batch，单图、变长N通过 | 本地缩小结构FP32步骤通过；真实数据/drag未执行 |
| AirfRANS | `forward(data) → [N,4]`，vx/vy/p/nut | M7通过 | floor/off原`MSE_weighted`训练+原MSE test通过；完整抽样epoch未执行 | 整成员+模型列表、负向、重复eval、新cwd通过 | 真Data/Batch、原sampled-ptr合同通过 | 本地缩小结构FP32步骤通过；真实数据/完整抽样物理指标未执行 |

“通过”指相应合成模型/源码检查，不表示真实数据读取、完整数据训练、收敛或准确率。AirfRANS完整采样需要本机缺失的`torch_cluster`，未伪造PyG/半径图、未跳过生产radius_graph来制造完整链路通过。工业多图支持未扩展。

## B. 文件、符号与必要改动

| 文件 | 符号/作用 |
|---|---|
| `cdlno/msar_lno/industrial.py`（新增） | `_GraphModel`、`CarModel`、`AirfRANSModel`及`adapter_architecture`：共享逐点lift、固定四通道head、任务字段校验、显式aux，稳定pickle类路径 |
| 两项目`models/MSAR_LNO.py`、`msar_entry.py`（新增） | 最薄的任务Model/Run别名，不复制MSAR数学，不改旧类路径 |
| `cdlno/msar_lno/industrial_entry.py`（新增） | `parse_args/model_kwargs/resolve_hparams`、`CarRun/AirRun`：新family显式解析、完整结构恢复、任务sidecar、原整对象/列表协议严格验证 |
| Car `models/cdlno_run.py` / Air `cdlno_entry.py` | 在原parse前识别明确新模型名，仅该family使用新parser副本；旧模型返回原参数，不收到新kwargs |
| 两项目`main.py/main_evaluation.py` | 显式`msar_lno`分支：早期实验记录、新Model/Run、原保存位置和原评价结果记录；数据主体不改 |
| 两项目`train.py` | 新family独立forward/objective分支；每epoch局部`ObjectiveMetrics`，不改变原返回tuple、原loss表达式、optimizer/scheduler/采样节奏；旧分支保持 |
| 两项目`configs/msar_lno/{car,airfrans}.json`及Air `params.yaml` | Light/floor/.01/.2默认，Car原200/batch1/lr.001；Air新增独立YAML键，仅原6项任务设置，无旧d/M注入 |
| `cdlno/msar_lno/car_preflight.py`、`tran_evaluate/msar_lno/{car,airfrans,_dispatch}.sh` | 复用现有路径设置、环境和原Car预检；用户余参最后覆盖，不自动启动训练 |
| `tests/test_msar_industrial.py`（新增14方法） | 真实PyG、原loss/epoch片段、图隔离、metadata/整对象严格加载、cwd、GPU、profile和脚本验证 |
| 6个既有测试/投影helper | 增加精确新family剥离规则、YAML独立键允许项、实际旧family fragment局部变量；完整旧入口/训练AST另与pre-M7验证，不忽略任意代码 |
| 本报告、STATUS/memory、脚本README和`docs/msar_lno_audit/m7/` | 八任务交付、真实命令及限制；保留此前报告/用户diff |

起点快照807文本文件：`/home/hwz/CDLNO-artifacts/msar-m7-before-bocerv4d/source`。19个已有文件增量修改、15个新增实现/测试/报告/核查文件（不计生成证据）；其余788个起点文件字节不变。[起点](msar_lno_audit/m7/before.json)、[先存diff](msar_lno_audit/m7/preexisting.patch)、[本轮diff](msar_lno_audit/m7/stage.patch)、[冻结记录](msar_lno_audit/m7/freeze.json)分开保存，不把原有未提交修改算成本阶段模型改动。

## C. 字段、公式和参数归属

| 合同 | 代码/公式 | 实际检查 |
|---|---|---|
| Car输入 | 只读`cfd_data.x[N,7] = xyz3+sdf1+normal3`；保留tuple的geom，由原loader构造，按既有基线不额外编码 | 真PyG；y或edge变化预测exact不变；输入不修改 |
| Air输入 | 只读`data.x[N,7] = xy2+Uinf2+sdf1+normal2`，追加从`data.pos[N,2]`到固定8×8参考点的64维距离，stem71 | 独立距离公式exact；原参考范围x[-2,4]/y[-1.5,1.5]与点序保持 |
| lift/core/head | `Linear(stem,2d)→GELU→Linear(2d,d) + placeholder → 原MSARLNO(output_dim=4)`；core返回[B1,N,4]，wrapper只squeeze batch轴 | 没有Conv、额外head或跨forward状态；不递归重新初始化core |
| 单图隔离 | 复用已接受`cdlno.airfrans._single_graph`只读校验；Car额外要求ptr=[0,N]；Air允许采样后all-zero batch与原ptr=[0,original_N] | 真Data/Batch，N11/23；多图拒绝；Air原N19采样10节点的ptr不改写 |
| 节点与mask | surf只用于原PDE loss/边界后处理，不作为coverage源mask；无现成面积测度时图内全部点均匀；后续latent各slot均匀 | 同步点排列后输出等变，atol2e-6/rtol1e-4；off/floor同权重节点顺序一致 |
| aux合同 | 既有`training_forward`调用`return_aux`，wrapper用dataclass replace只替换prediction成[N,4]，保持同一个prediction对象归属；局部变量不写model.last_loss | 原adapter有效检查；off/weight0不请求A；eval纯Tensor、不调用CoverageFloorLoss |
| Car PDE | `LPDE = MSE(out[:,:3],y[:,:3]) + reg*MSE(out[surf,3],y[surf,3])` | 当前原`train.train`与独立同权重原式一步Adam更新exact |
| Air PDE | `LPDE = MSE(out[~surf,:],y[~surf,:]) + reg*MSE(out[surf,:],y[surf,:])`，入口明确`criterion='MSE_weighted'`；eval保持MSE | 同上，非默认函数参数替代实际入口损失；eval原test通过 |
| coverage聚合 | 每次单图forward：`C=(1/4)Σ_l Σ_j relu(kappa*mu_lj-p_lj)^2/(mu_lj+eps)`；`Ltotal=LPDE+coverage_weight*C` | 四Down FP32 raw均值，只加一次；off返回原LPDE对象；epoch日志仅对optimizer steps取平均，无跨图attention/图间辅助loss状态 |
| 采样/scatter | `utils/metrics.Infer_test`原gather、`out[n][idx]=o.cpu()`、累积计数平均及surf vx/vy/nut处理不变 | 原AST安全片段+真实Data执行，压力通道/节点顺序对照exact；完整radius_graph/VTK/force链未执行 |

默认diagnostics关闭；显式开启只返回既有小型no-grad统计，不把整图A写入日志。新参数仅在两个wrapper自己的lift/placeholder和同一个MSAR core，层间/模型成员不共享storage。

正式Light/Full均实际构造并在CPU、B1/N11完成前向（M1>N是合法扩张，未截断M）。记录的是**真实模型参数量**，不代表大N训练性能：

| 任务 | Light参数 | Full参数 | 新增图训练数值结构 |
|---|---:|---:|---|
| Car | 1,711,420 | 6,814,324 | d8、M=[7,5,3,2]、heads=[2,2,4,4]，完整6+6 block拓扑 |
| AirfRANS | 1,723,708 | 6,838,900 | 同上；7+64输入合同不缩减 |

正式Light/Full完整图反传/性能未运行。测试中kappa=1用于使coverage低于floor时有可测信号，实际profile仍为.2。示例loss值仅为固定随机合成数据的数值证据，不是数据集结果。

## D. Checkpoint与可运行命令

保存/加载不改原序列化方式：Car最终`model_<nb_epochs>.pth`整对象；Air每成员`member_NNN/model`整对象、根`msar_lno`模型列表。稳定类路径分别为`cdlno.msar_lno.industrial.CarModel`和`AirfRANSModel`，本地`models.MSAR_LNO`只是别名。没有新resume，**这些整模型文件不含完整optimizer/scheduler/RNG续训状态**。

新run默认`output/<car|airfrans>/msar_lno/<light|full>/coverage_<floor|off>/<UTC+uuid>`，显式`--save_name`及`--msar-run-dir`保留独占防覆盖。data读取前预留路径/初始config；模型构造后记录实测参数；epoch四项目标日志与eval独立目录沿用现有experiment。Air成员相互独立。

eval先解析已保存`architecture.json`的完整resolved结构/版本/family及训练目标，再验证用户显式架构覆盖；`task.json`核对adapter和fold/task/nmodel/训练采样协议，再构造或反序列化。coverage纯eval差异只记录请求，不改训练sidecar。缺family、错family、缺/错key、形状、heads、reference固定buffer、共享参数/共享列表成员均严格拒绝。expected模型仅在隔离CPU RNG环境构造作strict验证，**不替换或重新初始化已加载模型**；沿原本地可信pickle `weights_only=False`边界，没有全局权限设置、自动转换或strict=False。

两个工业脚本的Light on/off和Full训练后立即eval模板、Car固定drag路径限制、Air train/eval路径差异见[脚本README](../tran_evaluate/msar_lno/README.md)。仅预览可执行：

```bash
# 实际checkout根目录，远端不依赖本机路径。
bash tran_evaluate/msar_lno/car.sh train --profile light --gpu 0 --dry-run
bash tran_evaluate/msar_lno/car.sh train --coverage-mode off --coverage-weight 0 --gpu 0 --dry-run
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/msar_lno/airfrans.sh train --profile full --dry-run
```

两个任务各支持上述三种配置。6条train和6条eval预览已通过**真实parser**校验，无Python数据入口执行。以下是本机实际执行的审计命令；全套中的冻结/夹具检查依赖保留的外置源码快照和回归文件，其绝对路径不是训练脚本依赖。

```bash
MSAR_M7_RESULTS=docs/msar_lno_audit/m7/task-cases.json \
PYTHONPATH=tests:. python -B -m unittest discover -s tests -p 'test_msar*.py' -v

PYTHONPATH=tests:. python -B -m unittest \
  test_shapenet_car test_airfrans test_kcdno_car test_kcdno_airfrans \
  test_front_training test_experiment_records test_periodic_visualization -v

python -B docs/kcdno_audit/replay_subset.py car --result docs/msar_lno_audit/m7/k0-car-replay.json
python -B docs/kcdno_audit/replay_subset.py airfrans --result docs/msar_lno_audit/m7/k0-airfrans-replay.json

# 分别在原Car/Air项目cwd，project为car或airfrans：
PYTHONPATH=/home/hwz/CDLNO python -B \
  /home/hwz/CDLNO/docs/msar_lno_audit/m0/wrapper_fixtures.py replay --project car \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers \
  --result /home/hwz/CDLNO/docs/msar_lno_audit/m7/car-wrapper-replay.json

python -B docs/msar_lno_audit/m7/finalize_evidence.py
```

远端若仅有代码checkout而没有本机审计快照，可运行下列合成检查；它们不读取真实数据或外置旧夹具。若远端具备torch_cluster，最后一项会执行原radius_graph的**合成**单epoch，仍不会启动真实数据集训练：

```bash
PYTHONPATH=tests:. python -B -m unittest -v \
  test_msar_industrial.IndustrialMSAR.test_real_graph_inputs_permutation_reference_and_batch_rejection \
  test_msar_industrial.IndustrialMSAR.test_original_weighted_losses_steps_and_eval \
  test_msar_industrial.IndustrialMSAR.test_coverage_off_weightzero_order_and_uniform_measure \
  test_msar_industrial.IndustrialMSAR.test_checkpoint_whole_list_member_readfirst_and_fresh_cwd \
  test_msar_industrial.IndustrialMSAR.test_checkpoint_corrupt_family_shape_behavior_and_shared_list \
  test_msar_industrial.IndustrialMSAR.test_cuda_real_graph_loss_steps \
  test_msar_industrial.IndustrialMSAR.test_air_original_sampled_epoch_when_extension_available
```

## E. 实际验证与冻结证据

- [MSAR全套](msar_lno_audit/m7/msar-tests.log)：**107项，106通过、1跳过，166.280s，无失败/错误**，包括新增14方法。唯一跳过为缺torch_cluster的Air原完整抽样epoch。两工业cwd各floor/off共4次新进程真实本地Model别名导入、实际parser恢复、shared-class整对象加载CPU FP32 MATH exact。
- [实际任务结果](msar_lno_audit/m7/task-cases.json)：两个任务×floor/off原loss梯度/一步Adam权重与独立同权重原式exact；覆盖N11/23、图隔离、coverage路径、严格checkpoint、原MSE test。另Car两个完整合成epoch（floor/off各一次）和Air两个原记录片段，重复eval不改训练config/sidecars/results。Car原可视化回调也实际执行；Air没有假装执行完整抽样可视化。
- 本地GPU两个任务真实PyG小结构、FP32/MATH原损失反传/optimizer/eval通过；[环境](msar_lno_audit/m7/environment.json)为Python3.13.9、Torch2.13.0+cu130、CUDA13.0、PyG2.3.1、RTX5090 Laptop。旧M2/M3 AMP检查随套件重跑，但**不等于工业任务AMP验收**。远端Python3.10/Torch2.11/cu128未实测。
- [旧受影响套件首轮](msar_lno_audit/m7/old-tests.log)：90方法、154.530s；2处skip记录均为既有Air缺torch_cluster。原生产回归通过，另有两处测试适配错误（YAML断言未排除新增独立键；单独exec的epoch记录片段未提供新局部family条件）。修正测试后[对应2项重跑](msar_lno_audit/m7/old-tests-corrections.log)均通过，2.524s。没有改旧生产行为或放宽数值容差；不把两次运行描述成一次全绿的90项运行。
- [首轮新增检查](msar_lno_audit/m7/initial-tests.log)唯一错误为合成Car调用漏传normalizer而传None，原最终日志要求可迭代coef_norm；补上四项真实合同的合成normalizer后全套通过。保留失败日志，不改原trainer来迁就错误测试。
- [夹具索引](msar_lno_audit/m7/fixture-index.json)：pre-K0工业8份（Transolver与CDLNO三模式×两任务）+M0工业6份（KCDNO all/off、matched×两任务），**14/14同一权重/固定输入精确回放，atol=rtol=0**。旧182个夹具文件hash保持。未触及的核心/时间/static旧夹具保留M0/M4—M6有效证据，不声称所有历史训练checkpoint兼容。
- 六个完整工业入口/训练文件经**只移除精确MSAR分支**后，与pre-M7完整AST相等；函数签名只增加可选局部统计参数，旧调用/返回不变。loader、sampling/idx、normalizer、mask、原loss/更新频率、metrics/drag、旧模型实现、共享原语和MSAR M1—M6模型数学均未改。[冻结证据](msar_lno_audit/m7/freeze.json)包含字节与AST哈希。本轮没有触及六PDE入口或任务预设。

## F. 自审重点与剩余边界

已完成5项自审：①真实字段/reference/tuple及节点顺序；②单图隔离和保留图构造；③原weighted loss与每forward四层coverage均值；④先读配置、strict整对象/列表、训练文件只读；⑤完整旧分支AST及同权重旧输出。证据在对应测试及冻结索引中，未发现需要修改确认模型公式的阻断冲突。

仍未验证：真实数据读取/训练/收敛/精度、Car drag和Air完整VTK/force指标、远端目标环境、正式大profile完整图反传/性能、工业AMP/backend/compile矩阵。Air缺torch_cluster是本机原依赖缺项，**不会在生产跳过图构造**。Car原drag硬编码param0/路径、外层日志变量压力/速度交换，Air既有验证日志拼写与条件缺陷均保留，不借新增模型修复冻结语义。本轮不新增resume或后续研究实验。

**本M阶段结束，未执行下一阶段。**
