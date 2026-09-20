# Looped LinearNO：LL0 只读审计与冻结

日期：2026-09-19。**唯一阶段状态：PASS（LL0 审计范围）**。未实现 loop 模型、schema、CLI、checkpoint 或 launcher，未执行 LL1。153 个现有 CPU 测试方法：147 通过、2 个方法失败（4 条已解释的历史冻结断言）、4 个 CUDA 专用方法跳过；不是全绿测试声明。

本阶段依据用户的 LL0 确认及 [LL0 提示词](../PLAN_Looped_LinearNO/Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md#ll0只读审计与冻结)。当前 checkout 是唯一集成真值。旧 A/K 只属于兼容性冻结范围，不进入本次模型设计。

## A. 仓库、实际阅读及覆盖边界

| 项目 | 实际结果 |
|---|---|
| 根目录 / cwd | `/home/hwz/CDLNO` |
| remote | `origin https://github.com/hxh5159/CDLNO-w.git` |
| branch | `main`，起始 status 为 `## main...origin/main` |
| HEAD | `5b991226c5354af3332b2f7306b370aef0950c79` |
| HEAD tree | `a661e0a53d367e09dfe9b5afaabcab29ee8aec63` |
| 初始文件 | tracked 1549、untracked 2、ignored 361；tracked staged/unstaged diff 均为空 |
| 用户原有 untracked | `PLAN_Looped_LinearNO/Looped_LinearNO_Codex_Staged_Prompts_CDLNO-w.md`、`PLAN_Looped_LinearNO/研究主线梳理 (2).md` |
| 远端运行路径 | `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor`；不是据 origin 名猜成 CDLNO-w |
| 远端数据根 | `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data`；本轮未访问 |

[start-manifest.json](loop_linearno_audit/ll0/start-manifest.json) 保存全部 1912 个已有文件的分类、size、SHA-256，包含用户 untracked 与 ignored，不仅是 git tracked。原始 [status](loop_linearno_audit/ll0/start-status.txt)、[unstaged diff](loop_linearno_audit/ll0/start-tracked.diff)、[staged diff](loop_linearno_audit/ll0/start-staged.diff) 单独保留。没有 stash/reset/clean/checkout/rebase/commit/push。

已完整阅读根 AGENTS.md；未发现更深层 AGENTS.md。它要求优先读取的当前输出报告、使用说明及输出审计证据，与纯 LinearNO/history 最新状态报告一起核对。AGENTS 中旧 CDLNO 的阶段状态不能覆盖用户这次 LL0 授权，也不能拿旧 CDLNO 的 batch/epoch 配置代替当前 LinearNO profile。

本轮准备与审计阅读范围：

- 新 staged prompt 全文；讨论文件对 loop 的后半段及最后冻结决定，尤其约 13204–14730 行。早期 A/K、Transolver-loop 候选不构成授权。
- `docs/CDLNO_EXPERIMENT_OUTPUTS_REPORT.md`、`CDLNO_EXPERIMENT_OUTPUTS.md`、当前纯 LinearNO 的 IMPLEMENTATION_STATUS / IMPLEMENTATION_REPORT / REPRODUCTION_MATRIX，以及 history 的状态、最终报告、R9 失败分类与 R10 冻结证据。旧 L0–L10 的报告结论作为历史证据，本轮另外重跑关键测试。
- `PDE-Solving-StandardBenchmark/model/LinearNO.py`、`LinearNO_Attention.py`，`cdlno/linearno/attention.py`、`airfrans.py`、`shapenet.py`；对应的固定官方三套任务模型和现有 parity/oracle 测试。
- 六个 exp 的 parser/factory/model/normalizer/loss/时间循环/存档分支；工业 main/main_evaluation、入口适配、原 train 和评价接口；`cdlno/linearno/{profiles,_profile_data,schema,checkpoint,standard_entry,air_entry,car_entry}.py`。
- history 的 core/context/config/factory/CLI、Standard 与工业 intercept、checkpoint/provenance/fair-run；`cdlno/{experiment,training_state,visualization}.py`、现有周期绘图接入、`monitor/runtime.py` 与核统计/导出逻辑；纯模型及 history launcher。

全仓 [逐文件覆盖清单](loop_linearno_audit/ll0/file-coverage.json) 索引 1551 个 tracked/untracked 文件：1526 个文本、25 个二进制；[Python 符号/import 索引](loop_linearno_audit/ll0/python-symbol-import-index.json) 覆盖 318 个原有 Python 文件，全部可按 Python 3.10 语法解析。另索引 notebook code cell。**全量 hash/AST 索引不等于逐字语义阅读全部历史日志、报告和所有其他模型**；清单明确标为 inventory / AST / binary，关键语义阅读范围如上。没有解压二进制充作当前源码，没有 import 会读取数据的 exp/main。

## B. 当前三条模型线与八任务真实调用链

### 纯 LinearNO 数学与构造

Standard 稳定类为 `model.LinearNO.Model`，文件位于 Standard 项目内；六任务共用此类。工业稳定类分别为 `cdlno.linearno.airfrans.AirfRANSLinearNO` 和 `cdlno.linearno.shapenet.ShapeNetLinearNO`，本地 `models/LinearNO.py` 提供任务导出。不能仅凭三个工程都有 `models` 目录而互换 import。

现有 block 是：

\[
x_1=x+\operatorname{Attn}(\operatorname{ln}_1(x)),\qquad
x_2=x_1+\operatorname{MLP}(\operatorname{ln}_2(x_1)).
\]

仅最后 block 再执行 `mlp2(ln_3(x_2))`。它仍执行完整 attention 与 MLP，并非把第八层替换成 head。真实符号：Standard `LinearNOBlock.forward`、Air `AirfRANSBlock.forward`、Car `ShapeNetBlock.forward`。

`Attn` 原语见 `cdlno/linearno/attention.py:LinearNOAttention.forward`：

\[
Z=\operatorname{splitHeads}(\operatorname{in\_project\_x}(x)),\quad
Q=\operatorname{softmax}_M(ZW_q/\tau_q),\quad
K=\operatorname{softmax}_N(ZW_k/\tau_k),\quad
C=K^TV,\quad O=\operatorname{to\_out}(\operatorname{mergeHeads}(QC)).
\]

`Q,K:[B,h,N,M]`、`V:[B,h,N,d_h]`、`C:[B,h,M,d_h]`。无温度变体省略除法；没有 attention scale、slice normalization、N×N 或 slice×slice attention。每种 q/k/v 小投影跨 head 共享，q 与 k 是不同参数。Conv 只有一套 Conv2d 输入投影，按既有 H/W 顺序 reshape。plain/temp/Air 的输出是 Linear+Dropout，conv/conv_temp/Car 是 Linear–GELU–Linear–Dropout。

Standard temp/conv_temp 的 `temperature_q/k` 初始 .5，clamp [.01,1]；Car 保留官方 `tempreature_q/k` 拼写，clamp [.1,2]；Air 的 `temperature` 是已知不参与 forward 的兼容参数。loop 不能统一改成一种温度。

现有初始化顺序：构造 preprocess、可选 time_fc、全部 blocks，之后一次 whole-model apply 重置 Linear/LN/Conv，最后创建 `(1/H)*rand(H)` placeholder。Linear 为 trunc_normal(.02)、bias0，LN scale1/bias0，Conv 为 kaiming_normal。注意源码中 `initialize_weights` 函数定义的位置不能当成其调用顺序。温度不受 apply 影响。主要 state_dict 前缀为 `preprocess.linear_pre/post`、可选 `time_fc`、`blocks.i.ln_1/Attn/ln_2/mlp`、最后 `blocks.last.ln_3/mlp2`、`placeholder`；位置 buffer 不持久化。完整键/shape/count 已由原有官方 fixture 测试重验，不更新 golden。

### 实际配置，不能读取 legacy fallback 代替 profile

下表为当前正式 paper / official 两 profile 的模型配置；全部基准 L=8、operator heads=8、dropout=0。H 表示 hidden width；网格单独列，避免 H 与 grid H 混淆。[24 个解析结果](loop_linearno_audit/ll0/resolved-profiles.json) 还包含 transolver_matched、字段来源、objective/data/evaluation 和训练预算。

| 任务 / exp | variant | hidden / ratio | base M → loop 默认 M | 接口、点序与 stem | 原参数量 |
|---|---|---|---|---|---|
| Airfoil / exp_airfoil | conv_temp | 128 / 1 | 64 → 128 | x[B,11271,2]，fx=None；221×51；坐标→preprocess+placeholder；out1 | 1,765,889 |
| Darcy / exp_darcy | conv_temp | 128 / 1 | 64 → 128 | x[B,7225,2]+fx[B,7225,1]；85×85；unified off；out1 | 1,766,145 |
| Elasticity / exp_elas | temp | 128 / 1 | 64 → 128 | x[B,972,2]，fx=None；无结构网格限制；out1 | 585,217 |
| Pipe / exp_pipe | conv_temp | 128 / 1 | 64 → 128 | x[B,16641,2]，fx=None；129×129 正方形；保留坐标 normalization 和 flatten 顺序；out1 | 1,765,889 |
| NS / exp_ns | plain | 256 / 2 | 32 → 64 | 64×64；unified on/ref10 距离**替换**x，再接 fx10；out1 | 3,377,921 |
| Plasticity / exp_plas | conv | 128 / 1 | 64 → 128 | 101×31；x2+fx1、T[B,1]、Time_Input=True；out4；不调整历史点序 | 1,799,428 |
| AirfRANS | airfrans | 256 / 2 | 32 → 64 | Data.x[N,7] + raw Data.pos[N,2] 的64个距离；**拼接**；out[N,4] | 3,358,788 |
| ShapeNet-Car | shapenet | 256 / 2 | 32 → 64 | (cfd_data,geom)，x[N,7]；geom 不参与模型；out[N,4]；单图 | 3,852,420 |

Standard 原 `key_ratio` 是绝对 M；Air 原 `slice_num` 是绝对 M；Car 原 `key_ratio=M/d_h` 必须为整数。按当前每个 profile 的 base M 倍增，不能把所有任务设为 M128。Standard forward 合同为 `Model(x,fx,T=None)->[B,N,out_dim]`。Air 与 Car 的 batch/ptr 单图校验不同：Air 保留采样后 original_N 的 ptr 合同，Car 为完整图 N，不能统一放宽。

### 八任务生产链

| 任务 | 命令 → parser → factory/class → checkpoint/eval |
|---|---|
| Airfoil | `tran_evaluate/linearno/airfoil_{train,eval}.sh` → `_static.sh` → `exp_airfoil.py` → local `cdlno_entry.parse_args('airfoil')` → `model_dict.get_model` → `model.LinearNO.Model` → `linearno_entry.StandardRun` → shared `StandardRun` / strict pair；eval 同 exp 的 eval 分支 |
| Darcy | `darcy_{train,eval}.sh` → `_static.sh` → `exp_darcy.py` → 同 parser/factory/Run，task=darcy；metadata 在模型构造前解析，保存数值 input/output normalizer |
| Elasticity | `elasticity_{train,eval}.sh` → `_static.sh` → `exp_elas.py` → 同链，显式 key=`LinearNO_Irregular_Mesh`，fx=None |
| Pipe | `pipe_{train,eval}.sh` → `_static.sh` → `exp_pipe.py` → 同链，保存坐标/output normalizer |
| NS | `ns_{train,eval}.sh` → `_static.sh` → `exp_ns.py` → 同链，十步时间协议在 exp，模型不保存真实时间状态 |
| Plasticity | `plasticity_{train,eval}.sh` → `_static.sh` → `exp_plas.py` → 同链，time_fc 和20次优化在原入口，不进入 factory 推导 |
| AirfRANS | `airfrans_{train,eval}.sh` → `tran_evaluate/_common.sh:cdlno_dispatch` → `main.py/main_evaluation.py` → local `cdlno_entry.parse_args` → `cdlno.linearno.air_entry.parse_args/run_cli` → `AirfRANSLinearNO` → `AirRun`、每成员 strict pair、ensemble manifest → 原采样/field/force evaluation |
| Car | `car_{train,eval}.sh` → `tran_evaluate/_common.sh:cdlno_dispatch` → `main.py/main_evaluation.py` → local `models.cdlno_run.parse_args` → `cdlno.linearno.car_entry.parse_args/run_cli` → `ShapeNetLinearNO` → `CarRun` strict pair → field 与当前 hxh 修正的 drag evaluator |

除了 Elasticity，Standard launcher 显式使用 `LinearNO_Structured_Mesh_2D`，NS plain 也由该已存在 key 路由到公共 Model。key 是架构入口，variant 才决定原语，不能把它们混为一谈。原 exp 的历史默认 Transolver_1D/2D 不能当作有效 LinearNO key。

已有 history intercept：Standard 在 `cdlno_entry.py:37` 的 LinearNO guard 内首先尝试 `linearno_history.standard_entry.intercept`；`model_dict.py:5` 根据 `_linearno_history_config` 返回 research module；local `linearno_entry.StandardRun` 再选择 pure/fair/history Run。工业 pure parser 顶部已有 `linearno_history.industrial.intercept`，run_cli 根据 `_linearno_history_adapter` 分流。省略开关/显式 A0K0 仍返回旧纯类；A1K0/A0K1/A1K1 是既存研究类，不能用它们改造 loop。

### 数据、训练与输出必须保留的事实

| 任务 | 当前数据/normalizer | 当前 loss、时间与评价合同 |
|---|---|---|
| Airfoil | NACA x/y/Q；first1000 / next200；Mach 原通道；无新增 normalization | 逐样本 flatten rL2，无 epsilon，静态任务 batch mean；每 batch 一次 AdamW/OneCycle；原几何场图与测试 rL2 |
| Darcy | 两份 piececonst smooth mat 的 coeff/sol；1000/200；train-fit input/output UnitTransformer | decode 后 rL2 + .1 导数项；仅导数分支裁剪预测边界再补零，central_diff zero-padding，dx=1/85；不能说整体输出边界被置零。official_release scheduler固定500且拒绝非500，paper用 resolved epochs |
| Elasticity | Random_UnitCell 数据；first1000/last200；stress output normalizer | fx=None；decode 后静态 mean rL2；原不规则点图 |
| Pipe | pipe x/y/Q；当前先截1200再1000/200；坐标与output train-fit | fx=None；decode 后静态 mean rL2；129×129 正方形；不能把它称为非方形。超1200文件与发布 last200 差异已是历史审计项，不在此修复 |
| NS | NavierStokes V1e-5 N1200 T20；1000/200；无 normalizer | 输入10→每次1帧；train10次真值回填、batch-sum10步loss后一次 backward/optimizer；test10次预测回填。每 batch scheduler一次，step/full-trajectory rL2分别记录 |
| Plasticity | plas_N987_T20；first900/last80；fx input normalizer | **实际 collate 为逐样本 torch.randperm20**，不是旧提示词的 NumPy permutation；20个 T 各自 backward/optimizer，scheduler每 outer batch一次；Torch/NumPy/Python RNG都保存；不修 meshgrid/reshape点序 |
| AirfRANS | manifest、VTU/VTP；full_train末10%作validation，即720/80，full_test200；saved coef_norm，rawpos不标准化 | 用户确认的 normalized四通道 MSE(~surf)+1×MSE(surf)；Adam/OneCycle；采样、边界恢复、pressure physical rL2 / release MSE / C_L误差与Spearman独立。paper与official为400epochs，transolver_matched398，不能挪用旧CDLNO398 |
| Car | param0…8 raw/preprocessed；default fold0；保留 os.listdir、Python随机抽样及 saved coef_norm | normalized全点速度3 MSE + .5×surface pressure MSE；Adam/OneCycle，200epochs；non-surface速度/表面压力physical rL2与release MSE/RMSE及drag/Spearman分开。保留当前正确surface velocity与实际路径接口，不回退发布param0错误 |

六 Standard paper/official 500epochs；batch分别 Airfoil4、Darcy4、Elasticity1、Pipe4、NS2、Plasticity8；Pipe transolver_matched ratio2/batch8，NS matched ratio1，均不能污染 paper/official。工业 OneCycle 的 `(ntrain//batch+1)*epochs` 总步数怪癖与原 step cadence 不顺手改。

现有记录器 `cdlno.experiment.Experiment` 创建 `config.json`、`train.log`、`train_history.jsonl`、`train_results.json`，重复评估独立 `evaluations/<unique>/eval.log,results.json` 并更新 `eval_results.json` 索引。现有周期 field figures 使用共享可视化；不同任务原指标/文件仍保留。LinearNO strict pair 由 `checkpoints/epoch_XXXX.{pt,metadata.json,json}` 与 `weights/epoch_XXXX.pt`、`latest/final.json` 指针组成；保存 optimizer、scheduler、RNG、generator/sampler、normalizer 数值。eval/resume 先校验 metadata/manifest，再构造并 strict=True 载入；eval 不重拟合 normalizer，不重写训练 sidecar。

## C. 固定外部来源与公式归属

完整路径、size、SHA-256、tree/blob 校验在 [reference-ledger.json](loop_linearno_audit/ll0/reference-ledger.json)。复用仓库外缓存，重新验证固定 blob；没有 vendor 参考树，没有加载下载的 checkpoint。

| 来源 | 固定版本 / tree | 已核对内容及边界 |
|---|---|---|
| HiPRL/LinearNO | `3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` / `4aa99d7dece8462d932e690a82e23dba78b11b0b` | 实际目录为 `Standard_PDE_Benchmark`、`AirfRANS`、`ShapeNetCar`；三套模型/变体、初始化和头；固定树无 LICENSE，不自行推定分发许可 |
| LinearNO paper | arXiv `2511.06294v3`，HTML SHA256 `637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd` | Eq.8–9 的因式算子；Table8/profile细节沿已审查实现；不是 loop 的官方实现 |
| thuml/Transolver | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` / `22e7386201596e45236907f774478f79d3913da7` | 历史来源/协议比较；MIT 根LICENSE及 Air LICENSE 保留；不假定与目标有共同 ancestor |
| Attention Residuals | arXiv `2603.15031v1`；PDF SHA256 `444673994328f7be8aee9d96fb240596b6f254f06ebaa53a2673413a244198c9` | Eq.2–4：RMSNorm key、单query、raw value、source-softmax；Eq.5–6/Figure2：completed block + current partial；§5：zero pseudo-query |
| MoonshotAI/Attention-Residuals | `85e22310fe5ee860b4a023de312d791de8a5a5e6` / `e1a7c3b7d398d37ef0a3f87fa67a917773f0c190` | README伪代码、报告与图；该pin没有完整可运行训练工程，也没有单独LICENSE文件；README概述不能代替公式的逐sublayer时序 |
| Kimi K3 | arXiv `2607.24653v2`；HTML SHA256 `8413ace971c97c8ae5fda01b53585c29710042f109ef23ad4ab2824a5300dcad`；官方 `3cb39dfd32e51c3328e2e4b4af21341247d06c43` / `fc92b6c233e751161a2c532b5753695c5789bccc` | §2.2 Eq.8–10 与final aggregate；报告中的8个12-layer blocks加embedding，不是本模型必须照搬的分块数。固定树为README/LICENSE/报告/assets，无可运行production模型；许可证原文保留在外部缓存 |
| Residual Scaling | arXiv `2606.18524v1`，HTML SHA256 `2dcb643477ec29ae9be1e1d2094354e21885c2e0a3080882b6fe75f072ff82ec` | 本轮下载固定v1；§3共享ReLU随机矩阵简化分析；§4多独立层有条件的 `lambda/(R sqrt(L))`。只支持采用loop-axis1/R的动机，不能推导本模型或RB稳定性保证；不改任务LR |
| SMELT（辅助阅读） | arXiv `2609.01343v2`，HTML SHA256 `bef02da921a4e73259ce95be731479ae7612e399a8883c50d078d1507663a0e4` | 中段循环两次和branch1/R；其MoE/宽度/GQA三预算匹配不是这里H不变、M×2的设计。下载缓存名 `k3_v2.html` 只是本地命名误称，ledger明确标作SMELT；它不是Kimi K3报告 |

来源归属：LinearNO 的 operator/stem/task wrapper 来自已验证移植；点域 AttnRes 聚合形式和 RB 来源拓扑来自 AttnRes/Kimi。把一个 core pass 视为 AttnRes block、共享 LinearNO 参数、两 preset、固定 H 下 M倍增及 LB 的 `Delta=Y-H` 来源，是本项目适配。不能称整个模型等同 Kimi K3，也不能拿 Transolver 核图、旧 latent-summary A/K 实验证明本项目有效。

## D. 冻结公式 → 原符号 → 安全插入点

| 冻结项 | 原模型符号 / 拟插入点（尚未实现） | 审计判断 |
|---|---|---|
| Stem/placeholder/time/position | Standard `Model.forward` 前半、工业 wrapper forward前半 | 保留语义；新wrapper复用子模块，禁止调用完整旧forward后追加loop |
| operator raw branch | `block.Attn(block.ln_1(x))` | LL2只加adapter，复用原Q/K/V/温度/to_out，不改原attention |
| MLP raw branch | `block.mlp(block.ln_2(x))` | LL2分别暴露，不把已含residual的整block再加一次 |
| final output | 仅 `last_layer=True` 的 `ln_3/mlp2` | **suffix>=1足够**：最后物理block在suffix、完成普通body后head一次；core全为non-final，不持有head |
| 参数共享 | P+C+S套物理block，core索引重复调用R次 | 模块身份共享，包括LN/operator/MLP；每次重算Q/K/V/C；不deepcopy、不缓存上一轮结果 |
| SR | core每条branch `x + u/R`；prefix/suffix普通残差 | identity不缩放，R是repeats；与RB/LB互斥 |
| RB | 每sublayer之前 point AR；首项看completed，后续另加raw partial；round结束存raw sum | 不做h+u，不乘1/R；exit再AR([anchor,b1,…,bR])；AR后才进原ln_1/ln_2 |
| LB | 一轮scaled SR得Y；实际相减 `Delta=Y-H` | 后续入口/最终AR([anchor,Delta1,…])；不得用Y作为source、不得再乘1/R或source数 |
| AR | `sum softmax_source(w·RMSNorm_g(V)) * V` | [B,N,H]点域，eps1e-6、w0、g1、无bias/value projection/sqrtH；每逻辑receiver独立，非operator heads分组 |
| 缓存生命周期 | 新core.forward局部tuple/list | 不detach、不buffer、不挂self、不跨forward/真实时间/ensemble成员；异常后下次为空 |

唯一拓扑字段为 `prefix_blocks=P`、`recurrent_core_blocks=C`、`loop_repeats=R`、`suffix_blocks=S`，约束 P>=0、C/R/S>=1；unique=P+C+S、executed=P+CR+S只派生。preset `p1_c3_r2_s1`=(1,3,2,1)有5套/执行8次；`p2_c2_r2_s2`=(2,2,2,2)有6套/执行8次。custom必须四字段齐全且不混preset覆盖。旧n_layers不能成为第二套拓扑真值。

RB有2CR+1个receiver、恰2H(2CR+1)个router参数；LB有R个receiver、2HR参数；SR为0。两preset的RB/LB source visits分别31/5、21/5。第一RB receiver只有anchor，其路由参数没有有效梯度是预期；w=0时norm scale初始梯度为零也是预期。初始均匀softmax给平均值，**不是原普通residual的等价初始化**。RB的零query不能靠乘来源数改成基线。

建议新wrapper构造 unique_depth 的原任务子模块一次，再由新core按索引调用；物理block只注册于一个容器，不把同一对象在每round重复注册成多个state键。原whole-model apply只对unique模块一次，placeholder时序保留；router独立初始化并隔离RNG，使同task/topology/rank/seed的三mode公共主干逐键相同。最终实现方式在LL2–LL5以独立oracle、对象身份、state键与初始化RNG证据验收，不在LL0预称验证。

新family=`linearno_loop`、extension=`loop_linearno_v1`；model_spec只放真实constructor字段，loop_spec保存完整拓扑/residual/rank/share/AR/version，profile/data/objective/evaluation/provenance/normalizer/resume各自分离。eval/resume从metadata恢复结构，显式CLI只作一致性检查；family/拓扑/M/模式/router/state冲突在构造和torch.load权重前拒绝；最后strict=True。旧checkpoint缺loop_spec只能走其旧family，不能猜loop。

未来CLI唯一字段：`--linearno-loop`、`--linearno-loop-topology`、`--linearno-loop-prefix-blocks`、`--linearno-loop-core-blocks`、`--linearno-loop-repeats`、`--linearno-loop-suffix-blocks`、`--linearno-loop-residual-mode`、`--linearno-loop-rank-multiplier`，以及现有 `--linearno-rank` 作actual M覆盖。显式rank与显式multiplier互斥；default multiplier2、control1。上述只是冻结设计，当前parser仍不支持。loop与任何旧A/K/history dropout flags（包括显式关闭值）混用都应在读权重/建模/建目录之前拒绝；没有loop字段的旧命令保持原分支。

## E. 后续最小修改预测与输出兼容风险

| 阶段 | 拟新增 / 必要修改（不是本轮改动） |
|---|---|
| LL1 | 顶层 `linearno_loop/` 纯schema/config、测试/配置文档；不import torch，不接parser |
| LL2 | `cdlno/linearno_loop/attnres.py`、`body.py`及独立oracle/tests；只调用旧block子模块 |
| LL3–LL5 | 新core/topology、Standard/工业wrapper、三mode与strict内部checkpoint；共同class路径与构造字段在LL1固定；不接八任务 |
| LL6 | 新 `cdlno/linearno_loop/standard_entry.py`、checkpoint/provenance；最小guard接local `cdlno_entry.py`、`model_dict.py`、`linearno_entry.py`，按实际exp已有 LinearNO guard复用kwargs/Run；不改data/loss/time区域 |
| LL7 | 新loop工业entry/model wrapper；最小guard接Air `cdlno_entry.py`、Car `models/cdlno_run.py` 及四main/eval分派；复用原train callback/metric/output，不复制工业科学协议 |
| LL6–LL7共同 | `cdlno/experiment.py` 新family的run_key/resume/config/attach_model/图标题分支；精确来源投影兼容；旧逻辑保持 |
| LL8 | `tran_evaluate/linearno_loop/<task>.sh`、独立命令/manifest，复用path.sh与原launcher；不会改旧默认或覆盖旧目录 |
| LL9 | 独立cost/diagnostics与测试；若记录Q/K，按logical round、physical block、sublayer识别；保留旧monitor原定义 |
| LL10 | 最终反向审计/报告/回归；不增加新功能 |

主要风险已自行检查，后续验收须覆盖：

1. **严格resume的来源哈希**：`cdlno.linearno.standard_entry.StandardRun.prepare` 比较 source_sha256；history provenance 还hash实际路由源码。已有 `linearno_history/provenance.py:ROUTING_REPLACEMENTS` 是精确字符串/次数、fail-closed投影。本轮8文件投影全等、两项源hash与R6原值一致。以后任何新guard都可能使旧resume拒绝。建议独立loop路由投影，先精确还原LL0旧源码再交旧pure/history哈希路径；loop provenance另外hash完整新增源码。必须用真正接线前checkpoint回归，不能忽略hash或笼统排除整个文件。具体变化仅在获授权接线阶段实施。
2. **family与输出目录**：`Experiment.__init__` 当前只认 `('linearno','linearno_history')`，不是任意family自动兼容。只加factory会误选run_key/resume分支，图标题映射也缺新family。需显式新增loop分支，训练前保留唯一目录、原JSONL/log/field/eval目录和索引；禁止改pure/history旧路径。未来run建议含task/profile/topology/mode/M/seed与unique suffix，确切格式由LL1固化。
3. **monitor重复调用**：`monitor/runtime.py:_write_snapshot` 按模块name保留第一次，当前纯模型每层只访问一次是正确前提。loop不能沿用这种去重后宣称记录了8次执行；LL9用独立观察器，默认关闭，不动旧核指标公式、不保图、不消耗训练RNG。不会用此发现顺手修改当前monitor。
4. **时间与初始化**：NS每个预测时间步、Plasticity每个查询、Air每个成员都新建局部loop状态；共享参数只初始化/保存一次，router按逻辑位置独立。相同seed不能只比较随机新实例，需公共主干逐键hash和独立DataLoader generator证据。
5. **性能归因**：5/6套物理block都执行8次；M翻倍增加rank相关MAC，且RB/LB新增来源读取。不能先宣称省FLOPs/延迟或compute-matched；三residual也不是严格单因素消融。保留纯8层M/2M、loop M/2M控制。

这些是有明确解决路径的工程边界，没有要求更改冻结数学的冲突；无需在LL0补基线或作额外研究决定。

## F. 实际命令、测试结果和失败解释

核心执行命令（只运行已有测试；本轮runner只负责隔离、日志与报告路径）：

```bash
pwd
git remote -v
git rev-parse HEAD HEAD^{tree}
git status --short --branch --untracked-files=all
git ls-files -z
git ls-files -z --others --exclude-standard
git ls-files -z --others --ignored --exclude-standard
rg --files
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -B docs/loop_linearno_audit/ll0/collect_readonly_evidence.py
PYTHONDONTWRITEBYTECODE=1 \
  python -B docs/loop_linearno_audit/ll0/run_existing_regressions.py
git diff --check
```

[runner](loop_linearno_audit/ll0/run_existing_regressions.py) 内逐项命令及耗时在 [regression-results.json](loop_linearno_audit/ll0/regression-results.json)。每个module用独立进程、最多3并行；实际环境固定 `PYTHONPATH=tests:.`、`CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`、OMP/MKL/OpenBLAS线程1、MPLBACKEND=Agg。所有报告环境变量指向本LL0证据目录。合成checkpoint/图片位于现有测试自行创建的临时目录，不提交大二进制。没有调用真实loader，不启动长训练。

本机 Python3.13.9、torch2.13.0+cu130、PyG2.3.1、NumPy2.2.6、timm1.0.28、einops0.8.2；[环境记录](loop_linearno_audit/ll0/environment.json)。未安装任何依赖，未探测/使用GPU；4项skip表示本轮显式屏蔽CUDA，不表示机器一定没有GPU。用户远端Python3.10/torch2.11/cu128未运行。

| 测试集合 | 方法数 | 结果 / 实际覆盖 |
|---|---:|---|
| 纯LinearNO已有18模块 | 84 | 78通过、2方法失败/4断言、4 CUDA skip；attention+完整三套模型官方parity/oracle、参数/键/梯度/一步optimizer/strict往返、profile/schema/转换器、旧Transolver八任务、RNG与八题原生合成闭环 |
| history schema/core/A/K/integration五模块 | 59 | 59通过；局部history/no-op、独立数学oracle、共享/门分解、20配置、strict/reload/错family、A0K0/RNG等核心覆盖；没有重跑R6–R8全部history生产闭环 |
| monitor已有两测试文件 | 10 | 10通过；低秩核公式、batch/head隔离、hook快照、纯模型导出筛选与不带权重 |
| 合计 | 153 | **147通过 / 2失败方法（4断言）/ 4skip / 0error** |

数值摘要见 [parity-summary.json](loop_linearno_audit/ll0/parity-summary.json)，完整JSON保存每层、输入/每个参数梯度、optimizer等tensor的 max/mean abs 与relative error。汇总中的 max_mean_abs 是各tensor mean_abs 的最大值，不是假称全tensor总体平均。

| 对照 | max abs / 最大逐tensor mean abs | 原容差 atol / rtol |
|---|---|---|
| 六attention变体 对官方 CPU FP32/FP64 | 0 / 0 | FP32 1e-6/1e-5；FP64 1e-12/1e-10 |
| attention 独立oracle FP64 | ≤8.89e-16 / ≤2.92e-16 | 1e-12/1e-10 |
| attention 独立oracle FP32 | ≤7.16e-7 / ≤1.74e-7 | 1e-6/1e-5 |
| Standard整模型 对官方 FP32；FP64 | 0/0；≤1.88e-16/3.71e-17 | 同上 |
| Standard整模型 独立double oracle | ≤1.14e-13 / ≤4.00e-14 | 1e-12/1e-10 |
| Air/Car整模型 对官方 CPU FP32/FP64 | 0 / 0 | 同上 |

relative error在接近零分母处会放大，原测试使用组合atol+rtol判断，未改容差。Air CPU官方full parity使用**测试局部** `.cuda()`替身；Standard原生official unified CUDA项没有伪装成CPU通过，已有独立位置/oracle覆盖与CUDA skip分别保留。官方模型文件在执行前由现有测试检查固定hash。

4条失败均可追溯，差异全文在 [historical-freeze-differences.patch](loop_linearno_audit/ll0/historical-freeze-differences.patch)，与历史R9失败登记一致：

| 失败 | 具体原因 | LL0判断 |
|---|---|---|
| legacy冻结：README | 原L1后追加13行纯LinearNO介绍/命令；hash变化 | 已有文档增量，不是模型变化；本轮前后字节相同 |
| legacy冻结：path.sh | 1行远端目录注释替换及5行说明新增；可执行export逻辑不变 | 用户指定LinearNO-monitor路径的已有说明变化；本轮未编辑 |
| legacy冻结：REPRODUCTION_MATRIX | 标题去L0并插入L10当前状态，旧测试要求旧首段精确前缀 | 文档历史检查过时；未删断言/更新golden |
| static launcher整方法 | 原shell清单中同一个path.sh历史hash不等，停止在该assert | 该方法此前16个新launcher/profile dry-run已执行；不能称整方法通过 |

旧Transolver八个真实工厂/权重/输入/输出/参数键数/strict与旧whole-object测试通过；旧CLI隔离、同seed记录/RNG恢复检查通过。额外重新核验 [100个受保护文件](loop_linearno_audit/ll0/historical-protected-recheck.json) 仍与history R0原hash一致，覆盖旧模型/数据/指标；[8个路由投影](loop_linearno_audit/ll0/routing-projection.json) 全等；[pure来源hash](loop_linearno_audit/ll0/baseline-provenance.json) 两项均与R6原值一致。没有把“全仓其他模型未重跑”写成全模型验收。

NOT RUN：loop实现与全部新模型测试、真实数据/VTK/sampling force完整流程、完整epoch/三seed/精度/SOTA、GPU/AMP性能、远端环境、全库578方法及history全部生产矩阵。本阶段不要求这些作为LL0 PASS前提。

## G. 本轮文件、冻结结论及自审

实际新增仅本报告、`docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md` 与 `docs/loop_linearno_audit/ll0/` 下的清单、源码索引、外部来源hash、已有测试日志/报告和只读审计runner。没有修改生产、配置、parser、factory、checkpoint、训练/评估、launcher、依赖、tests/golden、AGENTS、memory、旧文档。

[结束冻结复算](loop_linearno_audit/ll0/end-freeze.json) 以本轮 start manifest 为基线，逐项检查内容与tracked/untracked/ignored分类，并列出新增文件，不能只用git diff为空判定未改文件。两个用户未跟踪设计文档保持原hash；已有ignored项保留。外部reference下载在 `/home/hwz/CDLNO-artifacts/loop-linearno-ll0-20260919/`，不入目标树。

已自审5点：三套模型最后head隔离；SR/RB/LB缩放和raw来源时序；八任务rank/profile/时间合同；pure/history strict来源hash及输出接线风险；已知失败与新失败、CPU合成与真实实验边界。均有上文代码/命令/证据依据。未发现需用户另选架构才能进入LL1的冲突。下一阶段仍须用户明确授权，本阶段不生成可运行loop配置或训练命令。

**本 LL0 阶段结束，未执行下一阶段。**
