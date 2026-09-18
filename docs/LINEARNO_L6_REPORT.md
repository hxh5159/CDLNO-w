# LinearNO L6 — AirfRANS 模型验证与接入

## 2026-09-18 — L6 重新审查（取代下文旧验收结论）

**PASS；真实数据/真实训练/VTK全量指标/远端环境 NOT RUN。** 用户再次确认默认采用官方 Transolver/LinearNO 的标准化四通道 `MSE(~surf) + 1*MSE(surf)`。原接入已使用此目标，无需切换 loss。此前提出的物理空间联合 rL2 未采用；论文训练细节仍不冒充已公开。

本次修复实际入口测试揭示的工程遗漏：恢复保存的seed/task、设置原生记录器所需resume checkpoint、严格核对profile→constructor、验证pointer hash、单图batch拒绝、多成员顺序续训（包括后续成员尚未开始）、将曲线历史绑定到epoch存档、恢复非validation epoch所需的val_surf。数据checksum现在包含实际VTU/VTP文件，normalizer绑定聚合checksum；provenance使用实际源码和AST规范化patch。运行目录先于loader预留；默认目录含evaluation hash；显式CLI数据/相对运行路径正确转发。新family的周期场图复用hxh渲染，仅在独立AirFields中按原均匀采样提取点，无需图kernel。没有修改旧共享模型、数据loader或旧训练分支。

`tests/linearno/air_entry_worker.py` 现在执行真实parser AST→run_cli→train.main→原记录器→每成员pair→新进程resume/eval。仅以内存真实PyG替代数据边界，VTK正式指标边界在本闭环中替换为shape/选择验证；不能称真实数据或完整物理指标验收。保留原20次采样validation、每batch Adam/OneCycle。3合成epoch、9训练图、每图13–15点采样11点，d8/L2，正式结构未改。单成员及双成员连续/中断后的weights、全部resume_state hash、训练采样顺序、最终预测逐位一致；10项非法参数/结构/pointer检查及final场图通过。评估未改architecture.json/config.json且恢复saved normalizer，不重拟合。

[实际命令与结果](linearno_audit/l6/review/results.json)：31 passed，157.67s；含此前Air全模型官方parity（本机CPU及可用CUDA）、L1旧八任务Transolver回归及旧Air13项。[冻结核验](linearno_audit/l6/review/freeze.json)：旧Air完整训练AST相等，13个Transolver/PhysicsAttention/Embedding hash一致，309个既有ignored文件一致，既有LINEARNO不存在（N/A）。初次复核中的schema checksum和测试hook序列化失败已修复并重跑；不隐藏历史失败日志。

L6自审通过：目标/评价分离、原生续训与随机流、多成员顺序、元数据严格性、旧模型冻结。既往L6 PASS未覆盖真实run_cli恢复缺口，以本次扩展证据为准。按用户已有连续授权进入L7，未启动真实训练。

## 2026-09-18 最终补充：L6 接入完成

**阶段状态：PASS（资源门控项 NOT RUN）。** 本轮完成 AirfRANS 的 LinearNO 生产选择、训练/评估入口、metadata-first 严格 checkpoint、resume 和 ensemble 成员顺序验证。用户已确定默认 Air 目标为标准化四通道 volume MSE + 1×surface MSE；论文 pressure rL2 作为额外评估指标记录，不改变训练目标。没有读取真实数据、没有启动真实训练，也没有进入 L7。

实际修改了 [air_entry.py](../cdlno/linearno/air_entry.py)、AirfRANS 的 `cdlno_entry.py`/`main.py`/`main_evaluation.py`/`train.py`、两个 LinearNO launcher、Air 入口测试及旧回归投影。旧 Transolver 入口、模型、数据采样、normalizer、指标计算和保存语义保持；LinearNO 分支通过显式 `--model LinearNO` 早退进入独立 adapter。

本轮验证命令：

```bash
python -m pytest -q tests/linearno
# 75 passed, 2 warnings, 205.97s
python -m pytest -q tests/test_airfrans.py
# 13 passed, 1 warning, 8.19s
bash -n tran_evaluate/linearno/airfrans_train.sh tran_evaluate/linearno/airfrans_eval.sh
bash tran_evaluate/linearno/airfrans_eval.sh --dry-run --my_path /tmp/Dataset --experiment-dir /tmp/linearno-air
# dry-run passed; no data access
```

新增 Air 原生合成证据覆盖单成员连续训练/中断 resume/新进程 eval，以及 `nmodel=2` 两成员训练、strict state pair、成员 hash 和 manifest order 往返。Air model 参数量为 **3,358,788**；完整 LinearNO 回归和固定旧 AirfRANS 回归均通过。

明确未运行：真实 manifest/VTK loader、32k 随机采样、400 epoch、真实 force/pressure 指标、远端 Python3.10 + Torch2.11 + CUDA12.8、AMP/compile/性能测试。`torch_cluster` 等缺失依赖未安装。

本 L6 阶段结束，未执行下一阶段。


## 历史记录：AirfRANS C08 已由用户决定

用户选择官方 Transolver/LinearNO 的标准化四通道 volume MSE＋1×surface MSE。独立配置默认已落实，20项配置/metadata/旧Transolver/原Air loss检查通过；详见 [协议决定](LINEARNO_AIRFRANS_OBJECTIVE_DECISION.md)。旧paper训练不再作为当前Air默认，历史profile名必须配合 `airfrans_transolver_mse_v1` 和实际objective_spec解读。训练/eval入口尚未接线，**L6当前为PARTIAL而非已完成**；以下BLOCKED段落保留为本次决定前的历史记录。

**阶段状态：BLOCKED。** 用户本轮已授权依次完成 L6–L10，无需再次申请阶段授权。但 L0 冲突 C08 仍没有训练目标的明确决定；L6 第4项禁止擅自选择。已完成不依赖该决定的完整 AirfRANS 模型及独立验证，尚未完成生产训练/评估接线，不能进入 L7。L0–L5 的既有 PASS 不等于 C08 已解决。

本轮 `RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`。没有真实数据读取、下载、真实训练、依赖变更、commit/push/reset/clean。当前检查中的训练步均使用明确标记的合成张量；本机 CUDA 仅用于合成官方数值对照。

## A. 基线、授权和实际文件

目标 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`，origin 为 `git@github.com:hxh5159/CDLNO-w.git`。开始时保留全部 L0–L5 和用户的 tracked/untracked/ignored 文件，完整基线为 [baseline.json](linearno_audit/l6/baseline.json)，共 1,356 文件；源码快照 `/home/hwz/CDLNO-artifacts/linearno-l6-before-ryvj5v4k/source` 保留。

| 文件 | 本轮工作 |
|---|---|
| `cdlno/linearno/airfrans.py` | 新增 `AirfRANSLinearNO`、独立 point MLP/block、单图接口验证；组合原 L2 attention，无旧模型模式分支 |
| `Airfoil-Design-AirfRANS/models/LinearNO.py` | 新增薄导出 `Model`；完整对象稳定类路径 `cdlno.linearno.airfrans.AirfRANSLinearNO` |
| `tests/linearno/test_airfrans_model.py` | 新增完整模型官方同权重/独立 reference/实际 PyG/原训练函数/本地可信对象往返测试 |
| 本报告、STATUS、REPRODUCTION_MATRIX 增量、`docs/linearno_audit/l6/` | 实际证据、来源差异、回归和冻结核验；旧文档原文保留 |

未改 `main.py`、`main_evaluation.py`、`train.py`、`params.yaml`、数据/metrics/launcher、旧模型、共享 attention、schema/profile/checkpoint helper 或已有测试。新增 `Model` 导出尚未注册到实际 AirfRANS CLI，不能宣称命令已支持训练。

## B. 公式、官方符号和 shape 映射

固定官方来源 `HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` 的 `AirfRANS/models/LinearAttnNeuralOperator.py`；测试在提取类前验证 L0 保存的 SHA-256。没有加载外部 pickle 或复制整个官方树。

| 来源/合同 | 实际实现 | 独立证据 |
|---|---|---|
| data.x `[N,7]`、raw data.pos `[N,2]` | `AirfRANSLinearNO.forward`；维度/dtype/device/单图检查 | 真实 PyG Data，N=1/13/29，输入逐元素不变；多图明确拒绝 |
| x[-2,4]、y[-1.5,1.5]，ref8 | 非持久 `reference` buffer；`get_grid` | 固定边界/中心坐标，独立 x-major/y-minor 嵌套循环与手算欧氏距离；不用 Standard [0,1] 网格 |
| 原7维拼接64距离→71维 | `forward`→`preprocess` | preprocess hook 检查前7维原值/后64维距离；不是替换 x |
| Eq8/9：Q softmax_M，K softmax_N，Y=Q(KᵀV) | 复用已接受的 `LinearNOAttention(variant='airfrans')` | 完整函数式 reference 调独立 attention oracle，不调用被测 Model/block/MLP 的 forward |
| Q/K/V bias=False，跨head共享，absolute M32 | `blocks.*.Attn.to_q/to_k/to_v` | 正式参数 key/shape/value 与官方严格一一对应；无丢键/随机补键 |
| Linear input；Linear+Dropout output | 同一 attention 的 AirfRANS 专属路径 | CPU/CUDA dropout=.2 对照前恢复相同 RNG |
| 官方 dead temperature | 保留 `blocks.i.Attn.temperature` | 全8个均无梯度，改为900不改变预测；不把它加入 forward |
| 8个完整 pre-LN attention/FFN block | `AirfRANSBlock.forward` | 8个block逐层 hook；末block仍执行 attention+FFN，再 LN/head |
| 构造全部模块→一次 apply→最后 placeholder | `AirfRANSLinearNO.__init__/initialize_weights` | 正式配置所有参数值、state key顺序、初始化后 RNG 与官方逐位一致 |
| d256/L8/h8/M32/ratio2/out4 | 真实正式构造 | **3,358,788 参数**，包含8个 dead temperature 和 placeholder |
| 采样后原单图 ptr 保留 | `single_graph` | 单图 Batch 从9点取5点、ptr仍为[0,9]可用；真实两个图 Batch 拒绝 |

模型输出 `[N,4]`，保持 `[vx,vy,p,nut]`。模型不使用 graph edges；本轮没有改 graph 构造或采样。`reference` 不进入 state_dict；设备修复不增加权重键。官方 `linear` 是未使用参数，保留签名，不伪造另一条计算路径。

## C. 实际验证与边界

命令索引见 [commands.json](linearno_audit/l6/commands.json)，环境见 [environment.json](linearno_audit/l6/environment.json)，误差原始数据见 [air-model-final.json](linearno_audit/l6/air-model-final.json)。

新模型最终7个测试方法全部通过，unittest 12.086 秒，无失败/错误/跳过；早期6方法记录另存，不与最终7重复计数。

本轮共 **73个不同方法通过**：新模型7 + 旧模型/原Air训练记录18 + L1–L5补跑48，零失败/错误/跳过；文档增量后另重跑1个已计入的保留检查，也通过。L1–L5九个子进程总wall time577.955秒，其中四静态7方法unittest211.420秒、两个时间任务5方法193.694秒。完整逐模块统计见 [results-summary.json](linearno_audit/l6/results-summary.json)。这些是明确列出的测试集合，不声称运行了仓库中所有其他模型的全部测试。

| 检查 | 实际结果 | 不包含的证明 |
|---|---|---|
| 正式 d256/L8 构造/初始化/key/参数量 | 与固定官方逐位一致，3,358,788 | 未运行正式32k训练 |
| CPU float64 完整拓扑官方对照 | block/final/loss/input&parameter grad/Adam一步/strict往返全部误差0；atol1e-12/rtol1e-10 | 官方get_grid硬编码cuda，仅测试参考端临时将Tensor.cuda置为identity；不声称官方原生CPU支持 |
| CPU float32 完整拓扑官方对照 | 全部误差0；atol1e-6/rtol1e-5 | d16/L8/h4/M6/N17，仅缩小数值用例 |
| CUDA float32 原生官方对照 | 全部误差0；atol1e-5/rtol1e-4；官方源类没有设备适配 | 同样缩小宽度/N；不是远端、AMP、性能或真实数据验收 |
| 独立 double 完整网络 oracle | forward max abs4.163336342344337e-17；所有梯度最大abs2.6645352591003757e-15 | 不借用被测forward计算期望值；阈值未放宽 |
| 真实 PyG、可变N、mask与原loss连接 | 原train函数完整AST，normalized volume4-channel MSE+surface4-channel MSE，Adam/OneCycle各一步；同权重手算目标一致 | 独立函数调用，不等于main原生任务接线 |
| 新进程加载本地可信整模型 | AirfRANS实际cwd，稳定Model导出，输出逐位相同；state_dict另有strict往返 | 非官方外部pickle转换，非nmodel>1 ensemble，非任务resume |
| 旧Transolver回归 | `linearno.test_legacy`4、既有`test_airfrans`13、原weighted-loss记录片段1全部通过 | 不将旧模型检查宣称为新模型完整入口验收 |
| L1–L5回归 | [regression-results.json](linearno_audit/l6/regression-results.json)，逐模块独立进程；六Standard原生合成checkpoint/resume/eval重跑 | 不将六Standard闭环转移为AirfRANS已完成 |

同权重对照报告保留每个输入/参数/层的 max/mean absolute、max/mean relative 和 relative-L2；不只记录 allclose。相对误差分母下限仅用于测试误差展示，不是训练 loss epsilon。

本机 Python3.13.9 / torch2.13.0+cu130 / PyG2.3.1 / timm1.0.28 / einops0.8.2，GPU为 RTX5090 Laptop。缺 torch_cluster、torch_scatter、pyg-lib，不安装；完整 radius-graph 采样训练、真实数据一批、远端 Python3.10/Torch2.11/cu128 均 **NOT RUN**。monitor 自 L0 起不存在，标记 N/A，未伪造 monitor 包。

## D. 唯一阻塞决定：C08

实际来源为 [L0冲突台账C08](LINEARNO_REPRODUCTION_MATRIX.md)、`cdlno/linearno/_profile_data.py` 的 Air paper objective，以及阶段文件 L6 第4项：

> 论文未明确训练rL2究竟聚合哪些物理通道及归一化时机，必须遵循L0冲突台账中经审查的显式决定；若L0漏记则停止并返修L0，不能悄悄沿用normalized MSE。

L0记录了选择及影响，**没有记录用户选定哪个方案**；L1仍明确保存 `UNRESOLVED`。论文Table2压力评价不能自动推导训练只监督压力；官方MSE也不能自动决定新的rL2在归一化前还是后计算。此前“L0已通过”接受了带待定项的审计，不是某项具体科学选择。

AirfRANS paper objective已经明确的部分为 `L_volume + 0.5*L_surface`，分别 `~surf/surf`，按样本平均，无epsilon且分母非零断言；评价的pressure index2 physical-rL2也明确。尚需决定训练的三个字段：

| 候选（均不是声称作者未公开的定义） | 训练通道/空间/reduction | 实际影响 |
|---|---|---|
| 物理空间四通道联合rL2 | 输出/标签逆归一化；每region的 `[vx,vy,p,nut]` 一起flatten做一个范数比 | 保留全部通道监督；不同物理量单位/幅度影响联合范数 |
| 归一化空间逐通道rL2取均值 | 保留release训练空间；region内每通道单独范数比，再mean4 | 改变各物理量相对权重；单通道零分母情况与联合范数不同 |
| 物理空间只监督压力 | 逆归一化后取index2，region内rL2 | 与Table2评价对象一致，但失去velocity/nut直接监督，不能从Table2自行推定 |

这不是再次索取实施权限；问题是选择默认论文训练实际优化的函数。已异步提出上述问题，尚无回复。无论选哪项，`official_release`仍须实现已明确的 normalized 四通道MSE、volume+1×surface，并保留两profile分开的元数据/路径。L7 Car对应C08的normalization/reduction也需有明确协议，不能冒充已解决。

## E. 未完成项及源码事实校正

下列均为 L6 待做，不能因模型测试通过标为已完成：

1. `main/params/main_evaluation` 独立选择和真实CLI；两完整objective/profile、resolved scheduler和记录。
2. 正式32k迭代覆盖/重复点平均、最后10%验证/20次采样，paper压力rL2与原MSE分别输出；完整正交evaluation_spec。
3. 数值coef_norm恢复，metadata-first构造，每成员strict加载，完整optimizer/scheduler/RNG存档与续跑。
4. nmodel>1的成员顺序manifest、原生train→成员checkpoint→resume→新进程eval闭环及负向测试；实际train/eval/resume命令。

另有计划事实差异：提示词第7项称当前hxh已经生成 `ensemble_full.pth` / `ensemble_state_dict.pth`，实际 `train.py` 保存成员 `path/model`，`main.py`保存模型列表；当前树没有上述两个命名的产物协议。`main_evaluation.py`也已经有旧CDLNO/KCDNO/MSAR分支，不能说仍只硬编码Transolver。后续须在新family明确新增并验证ensemble方案，保护现有旧分支；不把不存在的能力当可复用前提。此项是源码校正，不要求用户替实施方案逐文件批准。

## F. 冻结与自审

[verify_freeze.py](linearno_audit/l6/verify_freeze.py) 比较全部L6基线文件的分类/size/SHA-256，并复查L0全manifest和重点freeze；[delivery-check.json](linearno_audit/l6/delivery-check.json)保存结果。L6允许的既存改动仅STATUS和REPRODUCTION_MATRIX增量，旧源码和既有测试字节不变，tracked diff与L6起点相同。L0广义冻结已有的三个L4/L5改动单列为历史，不算本轮改动；既有Transolver核心/attention/embedding不变。没有新增ignored；运行存档在仓库外，旧用户`.claude/settings.json`保留。

首次冻结检查发现旧 `test_airfrans.test_sidecar_read_first_member_paths_and_runtime` 的自动路径案例生成了两个根目录 `output/airfrans/<timestamp>/architecture.json`，因此初检失败。已核对它们均不在L6基线、仅含本轮小配置CDLNO测试sidecar，并保留字节/hash整体移到仓库外 `/home/hwz/CDLNO-artifacts/linearno-l6-old-air-test-output-r141xkk6/`，随后原冻结规则通过。详见 [relocated-test-artifacts.json](linearno_audit/l6/relocated-test-artifacts.json)。没有删除旧用户产物，没有为了通过检查放宽冻结清单，也没有改旧测试或路径实现；“没有新增ignored”指最终仓库状态。

自审已核对五点：官方任务专属位置拼接/顺序、完整初始化及dead参数、独立数学reference、单图与采样ptr、真实保存协议与未完成闭环边界。没有未解释的模型parity失败。真正尚需用户决定的是C08训练定义；生产接线和L7–L10没有完成，不能标PASS。

**本L阶段在C08处暂停，L6尚未完成；未执行L7–L10。** 收到协议决定后，继续现有L6工作并按本轮授权依次验证L7–L10，不需要重新申请阶段授权。
