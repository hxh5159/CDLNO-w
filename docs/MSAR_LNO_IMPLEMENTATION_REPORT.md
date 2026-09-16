# MSAR-LNO 最终实施报告（M9）

2026-09-17。M8已批准，本轮仅M9。基于当前`main@9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`及已接受的未提交工作，完成真实参数/全矩阵成本核对、有限GPU测量、独立规格反查和交付。**没有改模型/数据/训练语义，没有启动真实数据训练或自动commit/push。** M0—M8历史文件全部保留。

## A. 已实现架构与实际接口

独立family为`msar_lno`。实现了四级latent encoder–decoder、learned-query Down、12个完整FFN–SA–FFN latent block、query对齐的Up、三处固定scale=2的两来源AttnRes、逐点lift及LN/Linear head。Light默认、Full显式；所有任务保持同一profile，不按N裁剪M。两个profile的heads均[4,4,8,8]、encoder/decoder depths均[3,1,1,1]、两个独立2d GELU FFN。

```text
原任务字段 → pointwise lift → E0[B,N,d]
  Down1 → Encoder1(3 blocks) → E1[B,M1,d]
  Down2 → Encoder2(1 block)  → E2[B,M2,d]
  Down3 → Encoder3(1 block)  → E3[B,M3,d]
  Down4 → Encoder4(1 block)  → E4[B,M4,d]
  D4 = Decoder4(E4)
  l=3,2,1: U_l=Up(Q=E_l,KV=D_(l+1))
           D_l=Decoder_l(2 * softmax_source(w_l·RMS(E_l/U_l)) · raw[E_l,U_l])
  prediction = Linear(LayerNorm(Up0(Q=E0,KV=D1)))
```

Down不加learned query residual、不设额外Wq；Up是纯读取分支，不隐含E残差。三个w零初始化且每个仅d参数，初始化融合exact E+U。deepest无额外processor/skip；final无E0 skip。无卷积、N点SA、CDPA/kernel/history/Bridge、OT或新物理损失。尺度内仍有12次SA，不能称全模型无attention；query地址对齐也不是共享OT/双向守恒。

| 部件 | 稳定实际路径 / 符号 |
|---|---|
| 配置/profile/family | `cdlno.msar_lno.config`、`profiles`、`registry`；`MSARArchitectureConfig/MSARTrainingConfig` |
| 数学原语 | `cdlno.msar_lno.modules.{LearnedQueryDown,LatentFFNSAFFNBlock,QueryAlignedUpCross,PairwiseAttnResFusion,CoverageFloorLoss}` |
| core | `cdlno.msar_lno.core.MSARLNO`、显式每次调用的`MSARAuxOutput` |
| 静态/时间wrapper | `cdlno.msar_lno.standard.StaticModel`、`temporal.TemporalModel` |
| 工业wrapper | `cdlno.msar_lno.industrial.{CarModel,AirfRANSModel}`，本地`models/MSAR_LNO.py`为薄出口 |
| loss / 记录 | `objective.training_forward/training_objective/rollout_objective/ObjectiveMetrics` |
| 选择/checkpoint | 原`model_dict/main`的独立分支、`standard_entry.StandardRun`、`industrial_entry.{CarRun,AirRun}`、`metadata` |

[总控26条规格矩阵](MSAR_LNO_REQUIREMENTS_MATRIX.md)给出每条需求→源码符号→独立证据→状态。反查依据是当前确认的总控；架构讨论附件内已放弃的specialization、双向JS、skip dropout等候选没有带回模型。M0是用户要求的M1后补审，历史顺序和证据边界没有倒填。

## B. 完整参数量及计算口径

参数来自**实际构造的16个任务/profile模型**，不是MoNo论文数字。[完整inventory](msar_lno_audit/m9/parameter-cost-inventory.json)逐key记录shape/count/requires_grad；分组恰好覆盖所有参数且无共享对象/storage。公共四级部分如下：

| 真实参数分组 | Light d96 / M512,256,128,64 | Full d192 / M1024,512,256,128 |
|---|---:|---:|
| 四Down，含P、K/V/O、QK norm | 203,280 | 812,064 |
| 十二latent blocks，含两FFN、SA和所有norm/bias | 1,339,104 | 5,332,416 |
| 四Up，含Q/K/V/O、QK norm | 148,008 | 590,928 |
| 三PairwiseAttnRes | 288 = 3d | 576 = 3d |
| 上述公共部分合计（不含stem/head） | 1,690,680 | 6,735,984 |
| coverage训练参数 | 0 | 0 |

每层参数手式与实物相符：Down=`Md+3d²+d+2d/h`；latent block=`12d²+10d+2d/h`（两个FFN合计`8d²+6d`，SA投影`4d²+d`，三pre-norm`3d`，QK norm`2d/h`）；Up=`4d²+d+2d/h`。head含LayerNorm两组d参数及Linear权重/bias。placeholder及Plasticity time_fc计入stem，不漏作“无用”参数。

| 任务 | stem/time Light / Full | head Light / Full | 总参数 Light / Full |
|---|---:|---:|---:|
| Darcy | 31,200 / 99,264 | 289 / 577 | 1,722,169 / 6,835,825 |
| Elasticity | 19,200 / 75,264 | 289 / 577 | 1,710,169 / 6,811,825 |
| Airfoil | 19,200 / 75,264 | 289 / 577 | 1,710,169 / 6,811,825 |
| Pipe | 19,200 / 75,264 | 289 / 577 | 1,710,169 / 6,811,825 |
| NS | 32,928 / 102,720 | 289 / 577 | 1,723,897 / 6,839,281 |
| Plasticity | 37,920 / 149,568 | 580 / 1,156 | 1,729,180 / 6,886,708 |
| ShapeNet-Car | 20,160 / 77,184 | 580 / 1,156 | 1,711,420 / 6,814,324 |
| AirfRANS | 32,448 / 101,760 | 580 / 1,156 | 1,723,708 / 6,838,900 |

以下为**一次完整forward的矩阵MAC**：1 MAC=乘加，矩阵FLOPs=2×MAC。设Down源长s、目标长m，Up query长q、KV长k：

- Down：`B(2s+m)d² + 2Bsm d`，包括N规模K/V、M规模O及QK/AV；直接P没有Wq。
- 每个latent block：`12BM d² + 2BM²d`，12中包括4份SA投影及两个FFN的8份投影。
- Up：`2B(q+k)d² + 2Bqk d`。四次Up全部计入，包括final N点Q/O和N×M1读出。
- stem：按真实Linear输入/输出宽度逐个`BN cin cout`，time_fc按B而非BN计算，head另计`BNd cout`。

闭式公式逐组件对照既有工具的**实际Linear/SDPA张量shape hook**完全一致，包含所有上述运算。Pair打分点积、raw值加权、两源softmax、RMS/LayerNorm、GELU、bias、残差、reference距离/time三角函数及拷贝有独立scalar/operation ledger，**没有把2×矩阵MAC称作所有算子的精确总FLOPs**。既有profiler FLOPs仍只标partial，不拿漏掉SDPA的数字替代解析计数。Pair score额外有`2B(M1+M2+M3)d`个dot MAC-equivalent，其norm/融合/scale2另列。

| 任务，原预设B / N | Light GMAC | Full GMAC | floor四层FP32 A总载荷 MiB：Light / Full |
|---|---:|---:|---:|
| Darcy，4 / 7225 | 11.241398 | 58.630406 | 238.781 / 503.562 |
| Elasticity，1 / 972 | 1.144799 | 8.176189 | 10.844 / 28.188 |
| Airfoil，4 / 11271 | 14.976407 | 74.254052 | 365.219 / 756.438 |
| Pipe，8 / 16641 | 40.795230 | 191.836528 | 1066.062 / 2184.125 |
| NS，2 / 4096 | 3.979739 | 22.880453 | 70.500 / 154.000 |
| Plasticity，8 / 3131 | 13.529738 | 82.854270 | 221.688 / 495.375 |
| Car，1 / 32186 | 9.062881 | 39.738216 | 254.703 / 515.906 |
| AirfRANS，1 / 32000 | 9.408922 | 40.336589 | 253.250 / 513.000 |

此表使用正式N/B的**解析成本+实际参数构造**，不表示所有行都运行了满batch训练。NS为单forward，原训练10次forward；Plasticity为空间N3131的一次T条件forward，原batch含20次更新。不能把两者时间次数合入N或由MAC直接推断epoch。

coverage训练使用已有Down的QK/AV，不多算一套attention矩阵乘；增加显式FP32 logits/A、head/latent均值、floor/归一/层均值及对应反传。A载荷为`4BΣ_l h_l M_l Nsrc_l`字节；logits+A合计为该数的2倍。**这是tensor载荷，不是总峰值显存上界**：masked copy、Up/SA激活、梯度、optimizer及后端工作区另占内存。所有原batch大网格是否能训练仍未验证；下面已实测单图N32000，不修改M/kappa掩盖成本。

## C. 有限GPU实测

本机Python3.13.9、Torch2.13.0+cu130、CUDA13.0、PyG2.3.1，RTX5090 Laptop（24,463 MiB、driver591.86）。不是4090或远端Torch2.11/cu128。所有行固定B1、FP32、AMP/TF32/compile关闭、CPU threads1、warmup5/测量20，逐样本CUDA同步。使用同一个既有`measure.benchmark`：forward为eval/no_grad；完整合成step为zero_grad→forward→FP32 MSE（MSAR另加所选coverage）→backward→AdamW.step，lr.001/weight_decay1e-5/foreach=False，optimizer state在warmup初始化。计时含Python launch开销，不含数据、图构造、原任务loss、scheduler/rollout。

同profile的off/floor显式加载同一份初始state，初始hash相同；普通eval结果exact。floor默认kappa=.2、weight=.01未改，即便raw=0也确实计算四份A和coverage图。每行20次计时外另做一次不计时profiler，optimizer每活跃参数成功执行25次预热/测量更新，另有1次诊断更新。allocated峰值含模型/buffer/输入及中间张量；训练另含已初始化optimizer/梯度。它不是nvidia-smi reserved进程显存。

**Elasticity合同，N972，math SDPA**（原Transolver本来是手写attention，不强改为SDPA）：

| 模型 | 参数 | 完整GMAC | forward median / p90 ms | step median / p90 ms | forward / step peak MiB |
|---|---:|---:|---:|---:|---:|
| MSAR Light off | 1,710,169 | 1.144799 | 12.069 / 18.652 | 62.641 / 80.888 | 90.040 / 203.724 |
| MSAR Light floor | 同上 | 同上，coverage另列 | 12.334 / 14.827 | 65.782 / 69.581 | 90.040 / 204.150 |
| MSAR Full off | 6,811,825 | 8.176189 | 16.164 / 28.211 | 66.975 / 101.303 | 135.374 / 538.500 |
| MSAR Full floor | 同上 | 同上，coverage另列 | 12.639 / 14.164 | 66.985 / 72.560 | 135.374 / 538.528 |
| 原Transolver | 976,833 | 1.126924 | 4.321 / 5.133 | 26.992 / 28.985 | 73.439 / 156.843 |
| 已有生产matched LRSA | 3,133,569 | 1.440710 | 12.766 / 14.018 | 64.459 / 69.089 | 82.827 / 213.611 |
| KCDNO all | 2,639,240 | 1.405218 | 14.340 / 16.072 | 70.246 / 73.199 | 81.027 / 210.046 |

三种旧对照保持任务预设d128/h8/M64/L8；matched明确为完整LRSA、前段full，无消融；KCDNO r16/history all。MSAR d96/192及多尺度M与它们不同，所以**不是等宽/等参数公平对照，更不是精度比较**。一次有限运行存在明显p90/主机噪声，尤其Full off/floor正常推理是同一路径，不把两行计时差解释为coverage改变推理。没有由参数或MAC比例声称提速。

**真实AirfRANS wrapper + 真PyG合成单图，N32000**；不运行radius_graph/数据读取/抽样/VTK/物理指标。先保持所有SDPA为math，再单独做Light的efficient SDPA off/floor配对以观察显式A成本：

| backend / 模型 | forward median / p90 ms | step median / p90 ms | forward / step peak MiB |
|---|---:|---:|---:|
| math / Light off | 14.423 / 15.253 | 57.392 / 60.933 | 689.268 / 1563.767 |
| math / Light floor | 13.825 / 14.882 | 62.796 / 64.761 | 687.768 / 1563.509 |
| math / Full off | 23.217 / 25.515 | 73.837 / 75.513 | 1309.875 / 3245.791 |
| math / Full floor | 22.836 / 25.003 | 85.467 / 89.865 | 1309.875 / 3245.927 |
| efficient / Light off | 10.277 / 10.912 | 58.047 / 59.425 | 138.477 / 351.937 |
| efficient / Light floor | 10.180 / 11.535 | 61.876 / 64.031 | 138.477 / 1221.181 |

math后端自身已经使用显式attention存储，off/floor峰值差很小，本例训练中位数分别增加约9.4%/15.8%。efficient配对更直接显示floor成本：Light训练峰值增加约**869.244 MiB（3.47倍）**、step中位数增加约**6.6%**；普通推理同权重exact、同路径。floor的四个Down按已确认设计显式FP32 QK/softmax/AV，其余SA/Up继续使用统一请求的SDPA backend。没有悄悄换精度、M、kappa或公式。两组backend的数值不能混在一起宣称模型加速。

原始13行/所有样本/实际backend/完整step计数/峰值/源码hash分别在[gpu-n972.json](msar_lno_audit/m9/gpu-n972.json)、[gpu-air32000.json](msar_lno_audit/m9/gpu-air32000.json)、[gpu-air32000-efficient.json](msar_lno_audit/m9/gpu-air32000-efficient.json)。内存与训练时间只属于这些合成配置，不等于实际dataset epoch成本。

## D. 验证证据、旧模型与未验证范围

[M8覆盖表](MSAR_LNO_M8_ACCEPTANCE.md)及[逐任务JSON](msar_lno_audit/m8/coverage-matrix.json)仍是八任务验收依据；本轮生产代码字节未变，保留其有效证据，不把工具测量冒充原训练链路：

| 任务 | 原loss / 合成训练与eval | checkpoint / PyG / GPU |
|---|---|---|
| Darcy | Light off/floor；decode后field+0.1 derivative relative-L2，小7×7；5×7 wrapper另测 | task strict state_dict小结构；Full真实结构纯state strict；小结构原loss GPU |
| Elasticity | Light off/floor原relative-L2/decode，N35；Full N972扩张前向 | 同上；M9真实Light/Full N972额外合成MSE GPU计时 |
| Airfoil | Light off/floor原relative-L2，5×7，点序/通道保持 | 同Darcy；真实N11271布局小d另测 |
| Pipe | Light off/floor原坐标/目标normalizer及relative-L2，5×7 | 同Darcy；真实N16641布局小d另测 |
| NS | 正式Light、B1/N4096，10帧输入→out1，10真值回填训练/预测回填eval，1次更新 | task strict state_dict小结构；Full纯state strict；小d原10步GPU；没有新长时实验 |
| Plasticity | 正式Light、B1/N3131/fx1/T→out4，原20时间点逐次更新及一次scheduler | 同NS；小d原20步GPU，时间不并入N |
| Car | 正式Light单图N19，原全点速度MSE+reg×表面压力MSE及原test；完整小结构合成epoch | 原整对象协议/新cwd/真Data与Batch单图/多图拒绝；小d原loss GPU；Full纯state strict |
| AirfRANS | 正式Light单图N19，原MSE_weighted体积+reg×表面，原MSE test、sampled ptr/scatter片段 | 整成员/列表与新cwd/真PyG；小d原loss GPU；Full纯state strict；M9正式N32000额外合成MSE GPU |

八任务均完成静态参数流、正式Light两模式合成原loss forward/backward/optimizer/eval；正式Full均有单次**另行标注的合成MSE+coverage** backward/strict权重往返，不能称Full原loss完整优化矩阵。任务checkpoint保存协议测试采用M5—M7的小结构，Full纯state另测，不混称正式Light大结构工业pickle已经全矩阵验收。Air完整抽样epoch因缺torch_cluster未执行，真实PyG对象通过不等于radius_graph/VTK/force链通过。

M8旧157方法无遗留失败，1个原Air抽样子用例skip；MSAR107方法106通过/1个Air方法skip；75份旧Transolver/CDLNO三模式/KCDNO all/off/matched及其他现有模型**同一权重/固定输入**零容差回放，182夹具文件hash保持。M9再次验证文件hash及生产冻结；不重新随机初始化冒充基准，也不虚称重新跑了75次。GUNet原依赖缺失、Car固定drag路径/fold0和原日志问题仍保留。

本轮性能相关20方法通过（新5+旧工具15，19.059s）。独立数学/core36方法通过（包含double reference/gradcheck、FP32/AMP、trace、mask、梯度、无final skip和state roundtrip）；与新增命令方法一起首轮37项出现1个**新测试helper参数调用错误**，7.038s。修正新测试选用实际各项目parser helper后该方法单独通过（2.055s），完成32个真实train parser和32个eval dry-run；初始日志保留，生产未修复、容差未扩大。另一个新手算方法独立验证Down/Up/AttnRes/coverage，不依赖原oracle。最终57个不同方法均已通过，不把分次结果冒充单次57全绿。

**coverage off的准确含义**：默认diagnostics关闭时，off或weight0返回同一个LPDE Tensor，不调用coverage、不请求/保存供coverage使用的A；floor才有该训练图。模型仍要完成cross-attention与latent SA，math SDPA内部可以自行生成权重。显式开启diagnostics时会按需no-grad重算A并返回小统计，这不属于零开销off默认路径。普通任务eval始终预测Tensor，不因为保存的floor开关计算aux。

仍未验证：真实数据读取完整性、真实训练/收敛、预测精度、完整工业物理系数、实际epoch时长、SOTA、远端Python3.10/Torch2.11/CUDA12.8、所有正式任务batch的大网格训练/任务AMP/compile矩阵。M9的synthetic MSE性能不升级上述结论。

## E. 实际命令、checkpoint及远端使用

[八任务实际训练/评估命令](../tran_evaluate/msar_lno/README.md)已扩展为Light/Full×floor/off四列；使用实际`darcy/elasticity/airfoil/pipe/ns/plasticity/car/airfrans.sh train|eval`，没有虚构统一train.py。模板通过`train && eval`使用同一明确run目录；eval不猜latest，默认Light不覆盖保存的Full。AirfRANS用`CUDA_VISIBLE_DEVICES`，没有`--gpu`；train的`--my_path`指Dataset，eval指父目录；Car沿原fold0/固定drag路径预检。用户余参在现有脚本最后覆盖。本文没有执行这些真实训练命令。

输出位于`output/<task>/msar_lno/<profile>/coverage_<floor|off>/<unique>/`；数据读取前预留目录/配置，构造后实测参数，训练/评估结果独立记录。`architecture.json`保存family/resolved架构/版本/目标，`task.json`保存wrapper/任务协议；eval先读，再核对显式结构、strict加载，coverage差异只作请求不改训练文件。六PDE保存裸`model.pt`，Car保存整模型，Air成员整模型+列表；原局部可信pickle边界不扩大。**它们不是完整optimizer/scheduler/RNG续训档案，本系列没有新增任务resume。** 不提供旧模型到MSAR或Light→Full的自动权重迁移。

M9在仓库根目录实际执行：

```bash
python -B tools/msar_benchmark.py --task elasticity --N 35 \
  --models msar_light_off msar_light_floor --audit-only \
  --output docs/msar_lno_audit/m9/initial-audit.json
python -B tools/msar_benchmark.py --inventory-only \
  --output docs/msar_lno_audit/m9/parameter-cost-inventory.json
python -B tools/msar_benchmark.py --task elasticity --B 1 --N 972 \
  --device cuda:0 --precision fp32 --backend math --warmup 5 --iterations 20 \
  --output docs/msar_lno_audit/m9/gpu-n972.json
python -B tools/msar_benchmark.py --task airfrans --B 1 --N 32000 \
  --models msar_light_off msar_light_floor msar_full_off msar_full_floor \
  --device cuda:0 --precision fp32 --backend math --warmup 5 --iterations 20 \
  --output docs/msar_lno_audit/m9/gpu-air32000.json
python -B tools/msar_benchmark.py --task airfrans --B 1 --N 32000 \
  --models msar_light_off msar_light_floor --device cuda:0 --precision fp32 \
  --backend efficient --warmup 5 --iterations 20 \
  --output docs/msar_lno_audit/m9/gpu-air32000-efficient.json

PYTHONPATH=tests:. python -B -m unittest test_msar_performance test_performance test_kcdno_performance -v
PYTHONPATH=tests:. python -B -m unittest \
  test_msar_performance.MSARPerformance.test_eight_task_four_variants_real_train_parsers_and_eval_previews \
  test_msar_modules test_msar_core -v
PYTHONPATH=tests:. python -B -m unittest \
  test_msar_performance.MSARPerformance.test_eight_task_four_variants_real_train_parsers_and_eval_previews -v
python -B docs/msar_lno_audit/m9/finalize_evidence.py
```

实际stdout/stderr重定向分别留在[证据目录](msar_lno_audit/m9/)对应log文件，计时JSON保留20个原始样本；[summary](msar_lno_audit/m9/summary.json)核对命令/结果。首条性能测试命令执行时新文件有5个方法；随后新增的第6个命令测试单独执行，当前再运行将包含6个。不要覆盖既有证据，工具会拒绝已存在的output。

远端只使用现有能运行Car的环境，不升级/降级torch/PyG。以下是**未在远端执行**的无数据命令；CPU/GPU都不读取真实数据，完整冻结回归若要重放旧夹具，须复制原索引指向的同一份外置权重/快照，不能重新生成替代。

```bash
# 在实际远端checkout根目录；先只读检查版本。
python -B tools/cdlno_environment_preflight.py
msar_m9_dir=$(mktemp -d "${TMPDIR:-/tmp}/msar-m9.XXXXXX")
MSAR_M8_RESULTS="$msar_m9_dir/task-cases.json" PYTHONPATH=tests:. \
  python -B -m unittest test_msar_acceptance -v
PYTHONPATH=tests:. python -B -m unittest test_msar_performance -v
python -B tools/msar_benchmark.py --inventory-only --output "$msar_m9_dir/inventory.json"
python -B tools/msar_benchmark.py --task elasticity --B 1 --N 972 \
  --device cuda:0 --backend math --precision fp32 --warmup 5 --iterations 20 \
  --output "$msar_m9_dir/gpu-small.json"
# 原始N的有限单图，仅两个Light消融；保持默认kappa/M/精度。
python -B tools/msar_benchmark.py --task airfrans --B 1 --N 32000 \
  --models msar_light_off msar_light_floor --device cuda:0 --backend efficient \
  --precision fp32 --warmup 5 --iterations 20 --output "$msar_m9_dir/gpu-air.json"
```

若请求的backend不被远端栈支持，工具记录failed，不静默换backend；CUDA不可用则不运行GPU计时。可单独执行inventory/CPU audit检查功能，不用CPU时长推断某型号GPU。正式profile GPU任务验收可将M8命令加`MSAR_M8_DEVICE=cuda`，仍是合成训练步。

## F. 本轮diff、冻结和交付结论

新增`tools/msar_benchmark.py`和`tools/cdlno_perf/msar.py`，复用既有Case/原baseline加载/Linear与SDPA成本hook/Operations/profiler/同步计时；新增`tests/test_msar_performance.py`（6方法）、最终报告/规格矩阵/M9结果索引。唯一已有代码改动为`tools/cdlno_perf/measure.py::benchmark`新增可选显式`loss_closure`，默认原MSE路径保持；自定义目标暂不支持compile，明确报错。计时闭包包含真实MSAR coverage，不使用model.last_loss。旧默认MSE与off闭包相同权重经过全部测量更新后exact，旧性能套件通过。

README只新增独立MSAR章节，原CDLNO及上游论文文本完整保留；脚本README补Full off示例，实际launcher不改。独立STATUS、旧STATUS和memory追加M9结果。起点865文本文件快照`/home/hwz/CDLNO-artifacts/msar-m9-before-e8vxz5y5/source`、用户原diff与旧夹具均保留；[阶段diff](msar_lno_audit/m9/stage.patch)、[freeze](msar_lno_audit/m9/freeze.json)区分本轮与先前工作。生产模型/共享原语/配置/八任务入口/数据/loss/训练/评估/依赖及已有测试均未改。

五项自审已完成：①总控公式/轴/参数归属与独立oracle/手算；②真实参数及N规模全矩阵MAC；③coverage默认off/floor实际图、同权重推理和后端显存差异；④原任务/旧模型/strict sidecar保存边界；⑤声明与证据范围、命令及旧文本保留。未发现需要修改确认模型或数据协议的缺陷；唯一新增测试错误已经修正，未伪造原始成功记录。

最终回答：**A** 四级架构、Light/Full、无卷积、query对齐Up、两源scale2融合和可关闭coverage均实际实现；**B** 八任务静态/Light两模式合成原loss训练/eval、相应checkpoint、工业真PyG及有限GPU实际通过，Full与未执行项按D节区分；**C** 正常off/weight0真正移除辅助目标及额外coverage A，保留模型attention，显式diagnostics另有开销；**D** 旧75份同权重基准、182夹具hash、完整冻结和旧性能回归提供保留证据；**E** 真实读取完整性、收敛、精度、实际epoch时长及SOTA全部仍未验证。

**本M阶段结束，未执行下一阶段。**
