# KCDNO K0：现有集成仓库审计

日期：2026-09-15。**K0 完成；KCDNO 生产实现尚未开始。** 本轮只新增本报告、独立状态文档及审计/夹具文件。旧模型、训练/评估入口、依赖、数据及现有测试逻辑均未修改。

结论：已接受的 CDLNO 完整核心、front 三模式及八任务接入实际存在；可以复用 Down/Up、RMSNorm、两个独立 PlainFFN 和点域模块。KCDNO 必须新建 block/core，并将新历史读取放在 FFN1 与 FFN2 之间，不能调用整个旧 no_sa block 后再补读取。没有发现必须先重写旧 CDLNO 才能实施 KCDNO 的结构冲突。

## A. 依据、材料与实际版本

优先级为本轮用户 KCDNO 总控/K0指令 → [KCDNO v1 规格](../PLAN_KCDNO/KCDNO_Model_Specification_v1.md) → 后续单独点名阶段的执行要求。旧 CDLNO v1.2、front 消融和后来确认的输出管理继续约束旧家族。论文/理论附件不授权增加结构、损失或实验。

本轮实际阅读材料：

| 材料 | 本轮阅读范围/用途 |
|---|---|
| `PLAN_KCDNO/KCDNO_Model_Specification_v1.md` | 全文 §§1–13，实际512行；结构、初始化、核读取、任务表、成本、边界、归属。文件存在且已跟踪 |
| `AGENTS.md`、`memory/current-state.md` | 当前实际文件及历史状态；根目录是仓库唯一适用AGENTS，未修改它 |
| `PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md` | §§0–14，尤其完整front/bridge/rear/readout、八任务协议、初始化和checkpoint；旧占位名称不当作当前路径 |
| `docs/CDLNO_IMPLEMENTATION_STATUS.md`、`CDLNO_IMPLEMENTATION_REPORT.md` | 当前状态、核心/八任务/加载/输出记录及已有局限 |
| `docs/CDLNO_FRONT_ABLATION.md`、`CDLNO_FRONT_ABLATION_A1.md`、`A2.md`、`A3.md`、`A4.md` | 三公式、真实旧full夹具、模式传递、原损失/时间接口、计数和性能范围；历史“后阶段未执行”按日期和新记录解释 |
| `docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md`、`CDLNO_TASK_LAUNCHERS.md`、`tran_evaluate/`实际脚本 | 当前CLI、任务预设与脚本转发 |
| `docs/CDLNO_EXPERIMENT_OUTPUTS_REPORT.md`、`CDLNO_EXPERIMENT_OUTPUTS.md`、`CDLNO_VISUALIZATION_RESUME_V1.md` | 统一输出已接入；V1归档/绘图组件与尚未接入的V2–V5分开 |
| `memory/2026-09-13-exported-conversation-review.md`、`docs/CDLNO_THIRD_PARTY_NOTICES.md` | 设计演变、已放弃分支和许可证归属；未重新逐页阅读151页PDF或重跑理论附件 |
| 当前共享模块、八任务wrapper/CLI/checkpoint、六exp、四工业main及两个train、原模型、JSON/YAML/脚本 | 直接审查与AST/hash快照，不通过import入口取接口 |
| `/home/hwz/LRSA-Operator/src/perceiverforpde/modeling/layers/attn.py` | 实际LRSA block、构造/forward、两个disable开关、Up/FFN2边界 |

v1.1 历史材料仍位于 `PLAN_CDLNO/CDPA_v1_1/`；本轮不以它覆盖v1.2和后续决定。未重新运行其理论/环境检查，未查阅新的网络论文。KCDNO目录仅发现上述规格，**未发现另一个逐阶段K1–K10执行指南**；本报告后面的阶段分工是建议，不冒称用户已批准的独立文件。

版本证据：[baseline.json](kcdno_audit/baseline.json)。

- 工作目录 `/home/hwz/CDLNO`；分支 `main`；HEAD `9f72946e0adbfe27ba0b75b1646fa697ae16d3df`，父提交 `769fa333742f73c132868cf560bce5ec21529362`；与当前本地 `origin/main` 指向相同。未fetch/pull。
- 原Transolver审计基线 `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`；当前origin `git@github.com:thuml/Transolver.git`。
- **K0起点工作树与暂存区均干净，无已有未提交/未跟踪文件。** 此前报告中的“未提交A阶段修改”描述的是旧时间点；现在这些成果已包含在HEAD，不能继续照抄为当前dirty状态。`PLAN_KCDNO`也已跟踪。
- 保存328份tracked文件hash，另存302份文本源码/配置/脚本/说明于 `/home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/source`。不包括外部数据/训练结果。
- 初次续接摘要误称仍有大量未提交修改，已以实际git结果纠正；未用摘要代替源码证据。

## B. 来源、当前模型与新模型边界

| 来源 | 实际版本/许可证 | 当前用途及限制 |
|---|---|---|
| thuml/Transolver | 原基线75e0f67；当前集成9f72946；根MIT，Copyright2024 THUML | 保留原Physics-Attention、输入提升及数据/训练/评价协议 |
| Adversarr/LRSA-Operator | 本机干净checkout `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`，origin为对应官方SSH仓库 | 快照无LICENSE/COPYING/pyproject license声明；不推断MIT。复用本项目按已确认公式实现的原语，不vendor参考训练框架 |
| 7tl7qns7ch/IPOT | 历史记录 `18c177846267505ee9503445a146dfd7dee34c41`；原MIT声明见既有第三方说明 | 本轮未定位可读取的本地IPOT源码，未重新验真其快照；新KCDNO本就不需要IPOT。旧bridge/rear代码可直接审查 |
| AirfRANS数据归属 | 子项目LICENSE为ODbL | 是数据库许可，不能当作全仓代码MIT；不下载/再分发数据 |
| Linear Attention / Attention Residuals | 规格§13中的数学/设计引用 | 结合律核摘要和按来源评分的依据；不是照搬时间RNN/AttnRes训练框架或额外功能授权 |

当前正式类/注册是 `cdlno.core.CDLNO`、`cdlno.cdpa.CDPA`、任务键`CDLNO`。旧计划`cdpa_operator`→`cdlno`、`CDPAOperator`→`CDLNO`、整模型键`CDPA`→`CDLNO`；机制CDPA名称不改。新家族键/metadata应为`kcdno`，正式类建议`KCDNO`，尚不存在。

| 家族 | 实际/目标图 | 历史与边界 |
|---|---|---|
| 原Transolver（已实现） | 点提升→Physics-Attention/MLP残差blocks→head | 原slice/deslice与slice-token SA，不能冒充LRSA Down/Up |
| CDLNO full（已实现） | F完整LRSA→bridge→CDPA按模式→P独立SA+GEGLU→HF feature readout | 默认F2/P6；front FFN1→SA→FFN2 |
| CDLNO no_sa / identity（已实现） | 外围同上 | front分别FFN1→FFN2、T=S；rear SA继续存在；不是全模型无attention |
| `lrsa_matched`（已有性能工具） | L完整full LRSA→LN/head | 位于`tools/cdlno_perf/models.py:LRSAMatched`；显式锁full；**未注册八任务训练factory** |
| KCDNO（待实现） | L独立点域block→LN/head | 两FFN之间读全部先前源摘要；无bridge/persistent rear/CDPA/额外final-up；仍有Down/Up Cross |

不存在已落地的前段CDPA、相邻层CDPA、persistent Kimi AttnRes、后段MLP替代或旧校正层。新history只属于KCDNO，不能修改旧CDPA的无gate公式。当前KCDNO没有生产代码或入口，不标为“已支持训练”。

## C. 规格部件→代码→复用差异

行号对应本轮基线。可复用是代码/语义可复用，不代表跨层共享参数对象。

| 规格部件 | 当前实际文件/类/方法 | 复用判断及差异 |
|---|---|---|
| 标准静态输入提升 | `cdlno/standard.py:39 StaticStandardModel.__init__/forward`，实际类在`model/CDLNO_Irregular_Mesh.py`与`CDLNO_Structured_Mesh_2D.py` | 提升/坐标/placeholder语义可复用；基类需`structured_adapter`，不能直接实例化它；构造当前硬连CDLNO config/core，不能原样当KCDNO wrapper |
| 时间输入提升 | `cdlno/standard.py:149 TemporalStandardModel`，`model/CDLNO_Temporal_Structured_Mesh_2D.py:Model` | NS reference74维；Plasticity time_fc/正余弦embedding；保持单次调用。当前core绑定需要独立新家族接入 |
| 工业输入提升 | Car `models/CDLNO.py:37 Model`；`cdlno/airfrans.py:60 AirfRANSModel` | 7/71维stem和placeholder、图检查可复用；两类当前也绑定CDLNO core，不能只改模型名 |
| point入口norm | `cdlno/modules.py:291 LRSAFrontBlock`中的`point_norm` | RMSNorm(d,eps1e-6)，复用`RMSNorm`类独立实例 |
| Down | `cdlno/modules.py:237 _DownAttention.__init__/forward` | direct learnedQ[M,h,dh]，无Wq/无query residual；K/V无bias，O有bias，独立per-head Q/K RMS，SDPA softmax沿N；可复用，虽是内部下划线符号但实际定义存在 |
| FFN1及norm | `LRSAFrontBlock.latent_norm_1/latent_ffn_1`；`PlainFFN:93` | `U=S+FFN1(RMS1(S))`，两Linear有bias、GELU、ratio2；可独立组合 |
| SA及专属norm | `latent_norm_sa/latent_sa`；`_SelfAttention:207` | KCDNO不注册/不执行；不以Identity替换后保留外层加法，也不留下QKV/O |
| FFN2及norm | `latent_norm_2/latent_ffn_2` | 现no_sa紧接FFN1。KCDNO需在中间加入reader，然后`T=Uhat+FFN2(RMS2(Uhat))` |
| T返回/消费 | `LRSAFrontBlock.forward:339`返回`(u+point_update,t)`；`CDLNO.forward:77`按模式append(t)后rear读 | T在Up RMS前、活图；现front不接受history插入回调。KCDNO用新block，不能改变旧forward返回类型 |
| Up latent norm | `up_latent_norm=RMSNorm(d)` | 位于block外部，`_UpAttention`不做此层latent pre-norm；新组合必须恰好执行一次 |
| Up | `_UpAttention:285`继承`_ProjectedAttention:152` | Q来自同层入口Hn；K/V来自RMSup(T)；QKV无bias/O有bias、独立Q/K RMS、SDPA沿M；参数与Down解耦 |
| 点更新 | `point_ffn_norm` + `PlainFFN` / `ConvFFN:125` | `V=X+Up; Xnext=V+PointModule(RMSout(V))`；Conv为dense3×3/groups1→LN→Linear无bias→GELU→Linear有bias；保留5×7显式H/W和点序 |
| 输出norm/head | `LRSAFeatureReadout:411`的`output_norm/output` | 仅LayerNorm(d,eps1e-6)+Linear可独立复用其配置；**不能复用整个readout**，它还有final-up和点模块，会额外增加往返 |
| 原CDPA | `cdlno/cdpa.py:CDPA.forward/_depth_fusion` | 跨源共享receiver K/V/O和两级softmax；与源层writer摘要/raw value/γ不同，不能替作新reader |
| 原核心 | `cdlno/core.py:CDLNO` | 包含F/P、bridge、rear、final readout；整体不可当作KCDNO核心，不通过F=L放宽旧规则 |
| factory/CLI | PDE `model_dict.get_model`与`cdlno_entry.parse_args/model_kwargs`；Car `models/cdlno_run.py`；Air `cdlno_entry.py` | 新family有限分支可放这里；旧分支不能收到rank/history参数；保留原类路径 |
| config/sidecar | `cdlno/config.py`、`cdlno/checkpoint.py` | 现在严格绑定CDLNO model_name/history_rule/F/P。新配置需独立schema/family和完整字段，不能向旧config混入KCDNO字段 |
| checkpoint | `StaticRun.save/load`、`CarRun.load/_check_model`、`AirRun.load` | 复用read-before-compare/strict/局部信任原则；实际格式见下表，不强制统一 |
| 输出记录 | `cdlno/experiment.py:default_directory/Experiment` | 目录/记录原则可复用，但当前记录model='CDLNO'，不能直接用来伪标KCDNO。需新家族有限参数化/独立helper，不改变旧输出 |
| V1归档 | `cdlno/training_state.py:TrainingArchive/TrainingProtocol`、`training_observer.py` | 尚未任务接入且绑定旧sidecar；不自动升级成KCDNO resume支持 |

### 两个FFN的边界与最小新类方案

当前真实计算严格是：

```text
Hn = point_norm(H)
S = Down(Hn)
full:     A=S+FFN1(N1(S)); B=A+SA(Nsa(A)); T=B+FFN2(N2(B))
no_sa:    A=S+FFN1(N1(S)); T=A+FFN2(N2(A))
identity: T=S
V = H + Up(Hn, up_latent_norm(T))
Hnext = V + PointFFN_or_ConvFFN(point_ffn_norm(V))
```

新类组合建议：`RMSNorm`、`_DownAttention`、两套`PlainFFN/RMSNorm`、新reader/writer、`_UpAttention/RMSNorm`和点模块。新block内部执行FFN1→reader→FFN2，明确获得T；写摘要供后续层使用，Up消费同一T。新core负责局部tuple历史、L个独立block及最终LN/head。不要先构建完整旧CDLNO再替换core（会创建/初始化无用旧模块），不要改旧block以服务新语义。

LRSA参考实际位于`attn.py:503/512/566/605`以及结构版本`696/703/770/825`。`disable_interleaved_blocks=True`同时禁用SA和`channel_mixing_2`；FFN1由另一个`disable_interleaved_channel_mixing`控制。当前no_sa/KCDNO保留FFN2，与参考官方开关不等价。两段forward已直接核对，不能只凭名字移植。

### 新历史公式和参数归属（待实现验收合同）

源s的T `[B,M,d]`：`Ks=phi(RMSk_s(Ts) Wk_s) [B,M,r]`，缓存`(Ks^T Ts [B,r,d], sum_token(Ks) [B,r])`。`Wk_s`属于源层，value为raw Ts，无Wv/Wo。每份摘要只写一次。

接收l：`Ul=S+FFN1(N1(S))`；`Ql=phi(RMSq_l(Ul) Wq_l) [B,M,r]`一次。逐来源`Rls=(Ql matrix_s)/(Ql mass_s+1e-6)`，没有M×M token softmax，也没有current kernel self-read。来源候选包含`Rl0=Ul`；`alpha=softmax_source(w_l·RMSdepth_l(Rls)) [B,M,l]`；`Cl=Σ alpha*RAW Rls`；`Uhat=Ul+gamma_l*(Cl-Ul)`，再FFN2。不预合并来源，不重复加Ul。

`r=16`为总核特征数，与h独立；Wq/Wk无bias、Xavier gain1；w=0、gamma为无约束独立标量且初始0.1、norm scale1。ELU+1在FP32后clamp_min1e-6，sum摘要及eps不改mean。关键子图禁用autocast用FP32，返回U.dtype；double reference独立保留float64。其余ordinary Linear trunc_normal(.02)/bias0，Conv原生初始化、Down query[M,d] orthogonal，不递归覆盖。

第一层无reader/scorer/gate，最后一层无writer；L1或history_mode=off无闲置历史参数。L8/all应7次写、7次Q、28份读取；L次Down/Up、2L次latent FFN、0次latent SA。缓存只在单次forward内、tuple快照、不detach，不跨batch/NS真实时间、不捕获增长list。上述均为后续实现门槛，K0未编写核模块。

理论边界：缓存仅与相同phi/eps的显式正核读取等价，不等于softmax也不无损保留T；单来源匹配rank≤r不约束整网rank。eps使行和略小于1；gamma可越界，训练后不保证凸融合。KCDNO仍每层处理N点，不能继承旧CDLNO少做点域往返的速度优势，不能从局部MAC降低推断epoch提速。

## D. 八任务接口、配置与保存协议

下面的N是正式任务合同；K0缩小测试尺寸另列。原loss工具为`PDE-Solving-StandardBenchmark/utils/testloss.py:TestLoss`及`utils/normalizer.py:UnitTransformer`；K0只静态审查它们，夹具的诊断梯度不冒称原训练损失。

| 任务/入口 | 字段→模型实参；位置/时间和stem | 输出/原loss/冻结项 |
|---|---|---|
| Darcy `exp_darcy.py` | 两mat的`coeff/sol`；421×421按r5取85×85。`x[B,7225,2],fx[B,7225,1]`；wrapper用索引网格reference64替换xy再拼fx，stem65，无placeholder/time_fc | `[B,N,1]`；coeff/sol拟合原normalizer，训练decode预测与y，relative L2 +0.1×原central_diff梯度误差，保留padding/边界；eval相对L2 |
| Elasticity `exp_elas.py` | `elasticity/Meshes/Random_UnitCell_XY_10.npy`与`sigma_10.npy`；原轴转置/972点序。`x[B,972,2],fx=None`；xy/placeholder，stem2/PointFFN | `[B,N,1]` stress；y归一化/训练decode两者，测试decode预测；TestLoss，无网格卷积 |
| Airfoil `exp_airfoil.py` | `NACA_Cylinder_X/Y/Q.npy`（Q[:,4] Mach）；221×51原索引/弯曲坐标。`x[B,N,2],fx=None`，stem2/placeholder/ConvFFN | `[B,N,1]`；原TestLoss，无新增normalizer；不按物理坐标排序 |
| Pipe `exp_pipe.py` | `Pipe_X/Y/Q.npy`，Q[:,0]；129×129，原x/y normalizer。`x[B,N,2],fx=None`，stem2/placeholder/ConvFFN | `[B,N,1]` velocity；训练decode预测/y，测试decode预测；TestLoss，M32 |
| NS `exp_ns.py` | `NavierStokes_V1e-5_N1200_T20/...mat`的`u`，前10帧为fx，后10帧label。`x[B,4096,2],fx[B,4096,10]`；reference64+10=74，无time_fc | 每次`[B,N,1]`；10个TestLoss累加后一次backward/step，训练真值回填，验证/测试预测回填；每次core forward重建history |
| Plasticity `exp_plas.py` | mat `input/output`，标签transpose(-2,-1)形成空间/4/20；fx由input沿第二空间轴repeat，归一化fx；`x[B,3131,2],fx[B,3131,1],T[B,1]`，stem3+原sin/cos/time_fc | 每次`[B,N,4]`；20时刻分别TestLoss/backward/optimizer，scheduler每batch一次；collate同步置换T/label，不把20并入N，无反馈 |
| Car `main.py/main_evaluation.py` | `(cfd_data,geom_data)`；x7=`xyz3,sdf1,normal3`，位置[ N,3 ]/surf留给原图/损失；stem7/placeholder，不用geom encoder、不读y | `[N,4]` velocity3/pressure1；`mean MSE(v所有点)+weight*mean MSE(p表面)`，默认weight.5；fold0…8一次一个fold，原normalization/阻力/相关性不改 |
| AirfRANS `main.py/main_evaluation.py` | x7=`xy2,Uinf2,sdf1,normal2`，pos[N,2]；域[-2,4]×[-1.5,1.5]reference64追加到x7，stem71/placeholder；forward(data) | `[N,4]` vx/vy/p/nut；真实main选择`MSE_weighted`，backward=`loss_vol+weight*loss_surf`，默认weight1；保留抽样/radius_graph/idx scatter平均/surf及边界后处理 |

Car `_single_graph`接受Data或单图Batch的全零batch/[0,N] ptr，拒绝多图。Air `cdlno.airfrans._single_graph`同样拒绝多图，但允许原采样留下ptr原N而x/batch变短（单图全零batch且原N≥当前N）。不能以KCDNO不使用edge为由删除上游图处理，也不能把多图节点当一个场。

实际配置快照：[presets.json](kcdno_audit/presets.json)，八份`<project>/configs/CDLNO/<task>.json`加Air `params.yaml`；模型默认full/F2/L8/entry/chunk0、两FFN ratio2。Air CLI `nb_epochs/batch/lr=None`表示继承YAML，不能误读成没有训练预算。

| 任务 | 当前CDLNO d/h/M | epochs/batch/lr | optimizer/scheduler/clip | KCDNO主profile预期 |
|---|---|---|---|---|
| Darcy |128/8/64|500/4/.001|AdamW，OneCycle每batch，clip.1|同d/h/M，L8/r16/all |
| Elasticity |128/8/64|500/1/.001|AdamW，Cosine每epoch，clip.1|同上 |
| Airfoil |128/4/64|500/4/.001|AdamW，OneCycle每batch，clip.1|同上 |
| Pipe |128/4/32|500/8/.001|AdamW，OneCycle每batch，clip.1|保留M32 |
| NS |256/8/64|500/2/.001|AdamW，OneCycle每batch，**无clip**|同d/h/M；保留10→10 |
| Plasticity |128/8/64|500/8/.001|AdamW，20次更新/每batch1次OneCycle，clip.1|同d/h/M；保留T |
| Car |256/8/64|200/1/.001|Adam，OneCycle每batch，原reg/fold|同d/h/M；单图 |
| AirfRANS |256/8/64|398/1/.001|Adam，OneCycle；YAML原subsampling32000/r.05/max_neighbors64|同d/h/M；单图 |

六标准weight_decay1e-5保持。`kcdno_v1`和规格`transolver_shape_match`都尚未实现；后者仅新家族改Airfoil h8、Pipe h8/M64、NS/Car/Air M32，不能写回旧预设，也不叫等参数对照。

| 任务组 | 当前train/eval选择与参数流 | 当前checkpoint/加载 | 当前运行目录 |
|---|---|---|---|
| 六标准 | root `tran_evaluate/<task>.sh train|eval`或子项目`CDLNO_{Darcy,Elas,Airfoil,Pipe,NS,Plasticity}[_Eval].sh`→`exp_*`的`--model CDLNO`→`cdlno_entry.parse_args`合并显式CLI/JSON→`model_dict.get_model`→3个wrapper→core | `StaticRun.save/load`：`model.pt`纯state_dict，weights_only=True/strict=True；sidecar与adapter先验 | `output/<task>/<UTC timestamp>`；显式`--cdlno-run-dir`仍可定位旧runs；eval为run内`evaluations/<timestamp>` |
| Car | `--cfd_model CDLNO`→`models.cdlno_run.parse_args/model_kwargs`→`models.CDLNO.Model`；train和main_evaluation均有新分支 | 原`train.main`保存`model_<nb_epochs>.pth`完整对象；`CarRun.load`局部trusted weights_only=False，精确类/完整config/front属性/严格state检查 | `output/car/<timestamp>`或显式`--run_dir`；旧Transolver保留`metrics/<model>/<fold>/<epochs>_<weight>` |
| AirfRANS | `--model CDLNO`→`cdlno_entry`的JSON和`params.yaml:CDLNO`→`models.CDLNO.Model`导出共享类；main_evaluation可选择 | `member_000/model`等成员整对象，根`CDLNO`为模型list；`AirRun.load`局部trusted unpickle后校验类/成员数/完整config/state | `output/airfrans/<timestamp>`或显式`--run_dir`；旧分支`<save_path>/<task>/<model>` |

标准实际子项目脚本名以[脚本清单](CDLNO_TASK_LAUNCHERS.md)和源码快照为准，root还提供`train_eval.sh TASK`顺序调用且训练失败不评估；K0没有启动它训练。Air训练`--my_path`指Dataset目录，独立评价同名参数指父目录，保留差异。

最新输出管理在参数解析后、读数据前reserve目录并写`config.json/train.log`，模型构造后记实际参数量；训练`train_history.jsonl/train_results.json`，独立评估`evaluations/.../results.json`和根`eval_results.json`。旧full也使用此最新默认，不再声称默认仍是`runs/CDLNO/.../entry`。KCDNO后续建议`output/<dataset>/<timestamp>_kcdno_<profile>`，兼顾既定dataset层级与新family隔离；用户显式目录继续拒绝覆盖。

### 顶层副作用的实际审查

当前六个exp的数据加载位于受`__main__`保护的`main()`中，**并非六个都在import时直接loadmat**；但是parser、CUDA_VISIBLE_DEVICES、CDLNO实验reserve/log等在顶层，因此仍不可直接import，不能称安全库。四个工业main/main_evaluation确有顶层manifest/load_train等读取。K0只执行AST挑出的ArgumentParser/add_argument语句和安全helper；没有import上述十个入口，也没有调用其`--help`。

### checkpoint兼容边界

旧CDLNO config是`model_name=CDLNO/model_version=cdlno-core-v1`，front mode为架构字段；当前history_rule为`front-t-after-selected-processor-before-up-v1`。sidecar读入完整字段后比较；唯一旧缺字段例外是可辨识pre-A1完整18字段full与旧FFN2 history_rule。旧18槽pickle由`__setstate__`迁移；旧front缺属性仅回退full；未知/残缺配置不补默认。工业反序列化不调用新init，因此保留类路径和对象属性校验十分必要。

新KCDNO不复用旧family、F/P、decoder/history_rule或旧pickle类来猜结构；需记录family/profile/L/d/M/h/r/FFN/norm/phi/eps/history_mode/初始化版本，显式冲突加载前拒绝。执行device/dtype/chunk与架构字段分开。三种旧模式之间也不迁移权重，不用strict=False掩盖错误。root输入/输出日志helper当前写死CDLNO字段，后续必须明确新家族记录路径。

## E. 本轮实际验证与回归夹具

证据入口：[fixture_index.json](kcdno_audit/fixture_index.json)、[capture.json](kcdno_audit/capture.json)、[replay.json](kcdno_audit/replay.json)。二进制全部独立保存在`/home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures`，总17,048,463字节（约16.26MiB），未写入output/runs/数据目录，未提交。

每条保存真实构造kwargs/有效architecture与adapter（适用时）、state_dict键/shape/dtype/hash、**同一份权重**、固定合成输入、eval输出、诊断输入/参数梯度、源类路径和后端。`fixture.pt`为纯Tensor/基础类型，weights_only=True读取；另保存工业原格式对象/list及真实helper创建的sidecar。capture拒绝已存在的夹具目录；之后只replay，不生成新随机权重代替基线。

| 范围 | 实际数量/设备/结果 |
|---|---|
| CDLNO八任务×front三模式 | 24份，CPU FP32；真实wrapper导入，eval输出和诊断梯度捕获/strict加载回放全部通过 |
| 核心front三模式×CDPA off/entry/every | 9份，CPU FP32；L8/F2/M4/d8/h2，保存当前真实权重/输入/输出/梯度，全通过 |
| 原Transolver八任务 | 8份，本机CUDA FP32；**直接导入原模型源码，无.cuda monkeypatch/AST数学改写**，保留位置/时间/PyG合同；全通过 |
| 工业真实PyG | Car/Air各四模型（CDLNO三模式+原Transolver），N11和N19单图Batch、真实x/pos/y/surf/edge；输出/梯度及整对象，Air另有list/member，全通过 |
| 同后端严格回放 | 41/41；same weights/inputs，输出与诊断梯度atol=rtol=0；最大输出绝对误差0，None梯度也按基线比较 |
| 24个CDLNO实际Run加载器 | 标准state_dict，Car整对象，Air list+member；chunk0精确，chunk1按atol1e-5/rtol3e-4；eval省略mode从sidecar恢复、另外两模式显式冲突拒绝、文件字节不改写，全通过 |
| 三个原cwd新进程 | capture与replay分别独立子进程；明确PYTHONPATH=root+当前项目，确认`cdlno.__file__`，原工业stable class加载通过 |
| CLI/配置/模型注册 | 八任务默认、24组三模式train/eval解析，3个项目进程exit0；原parser默认/老注册/JSON/YAML精确留存，全通过 |
| 冻结原语义 | 7项既有静态/AST冻结测试，**7/7通过，1.221s**；本轮没改既有测试 |

任务夹具共用小宽度d8/h2/M4、L8，CDLNO F2/entry；这不改变正式预设。Darcy/Airfoil/Pipe取B2/5×7作模型布局依据，**不是Darcy原方格梯度loss检查**；Elasticity B2/N972、NS B2/64×64/fx10、Plasticity B2/101×31/T[B,1]/out4；工业B1图/N11,19。K0没有新跑Darcy85²/Airfoil221×51/Pipe129²/工业32000等大N布局，它们在A3旧报告中的证据继续作为历史结果。

诊断backward使用`mean(output**2)`，**不是原任务loss/optimizer step或训练**；目的仅是给后续K阶段保留梯度基准。K0未执行原损失链路或NS十步/Plasticity20步训练，已保存对应源码AST以便后续验收。PyG通过只指真实对象接口和加载，不含邻居构图、真实文件或物理指标。

真实pre-A1 full旧夹具仍存在于`/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0`，未改写；A1/A2/A3报告记录过回放。本轮41份是当前接受版本的新基准，不能据此声称所有历史checkpoint兼容，也没有将当前对象删属性伪造成历史证据。

固定执行条件：Python3.13.9/torch2.13.0+cu130，CPU单线程，SDPA=MATH，FP32、dropout0/eval、AMP/compile/TF32关闭、cuDNN benchmark=False/deterministic=True。seed、kwargs和每个环境见分项目结果。跨设备/后端容差预留沿用atol1e-5/rtol3e-4，但K0未进行CPU↔GPU同模型精度对比或新模型AMP检查；零容差是当前同后端回归要求，不是对远端机器的逐位保证。

### 环境与安装

[environment.json](kcdno_audit/environment.json)：本机Python3.13.9、torch2.13.0+cu130/CUDA13.0、RTX5090 Laptop、PyG2.3.1、NumPy2.2.6、SciPy1.16.3、timm1.0.28、einops0.8.2、PyYAML6.0.3；`torch_cluster/pyg-lib`本机缺失。不能把本机旧PyG所需扩展误当作远端新PyG必装项。

`pyproject.toml`的distribution为cdlno0.1.0，包发现`cdlno*`，目标Python>=3.10,<3.12；本机**未安装editable distribution**。三个cwd去掉PYTHONPATH均实际ModuleNotFoundError，设置根PYTHONPATH均成功；root launch helper显式加入PYTHONPATH。现状是源码导入成功，不是editable安装已通过。本轮没有安装、更改Python限制或依赖；远端Python3.10/Torch2.11/cu128及用户已成功Car环境仍为兼容目标，未远端执行。

### 可复查命令

在仓库根目录；脚本自行启动三个原cwd的新进程。第一条捕获命令已执行，**已存在目录不可再次capture覆盖**：

```bash
python -B docs/kcdno_audit/make_regression_fixtures.py capture \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result docs/kcdno_audit/capture.json

# 后续K阶段比较必须用这个replay，同一份旧权重/输入；结果可另选新文件。
python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result /tmp/kcdno-old-model-replay.json

python -B docs/kcdno_audit/audit_static.py \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures
python -B docs/kcdno_audit/audit_static.py --freeze
git diff --check
```

7项既有冻结测试的实际命令见[commands.txt](kcdno_audit/commands.txt)，日志[frozen-tests.txt](kcdno_audit/frozen-tests.txt)。`entry-contracts.json`保留12个entry/train函数AST摘要及关键构造/损失/读取/保存语句；完整AST和302份源码放外部快照。后续应与K0快照比较，再按明确授权仅排除新family选择/元数据语句，不能宽泛忽略整个训练函数。

初版临时夹具脚本曾因直接实例化需structured_adapter的基类、NS传5×7及审计脚本变量名错误失败；均为本轮审计脚本问题，未修改生产代码迁就。最初22份不完整夹具已移到外部`superseded-k0-partial-fixtures`保留，**不计入最终41份验收**。最终使用实际wrapper/真实NS网格且全部回放。见[execution-notes.json](kcdno_audit/execution-notes.json)。

## F. 冲突、旧缺项及后续范围

| 发现/边界 | 来源和影响 | 本轮处理/后续最小建议 |
|---|---|---|
| 用户粘贴AGENTS有A2完成/A3A4未执行的旧段落 | 当前仓库顶端STATUS、A3/A4报告及performance源码已更新 | 以当前用户K0授权和实际代码为准，保留旧文档；不重做A3/A4 |
| KCDNO与CDLNO整体图不同 | 新规格§§1/3/11 vs core中的F/P/bridge/rear/readout | 独立配置/核心，复用原语；不是旧实现错误，不“修复”旧CDLNO |
| 旧CLI裸parser模型默认别名不在registry | `Transolver_1D/2D` vs注册`Transolver_Irregular_Mesh/Structured_Mesh_2D` | 原官方scripts显式传有效键；K0记录KeyError，未顺手改旧默认 |
| Car外层loss变量名互换/固定阻力raw路径和param0 | 原train外层日志和utils.drag_coefficient；内部真实backward正确 | 旧问题单列，KCDNO不能以此改loss/指标；完整评价路径需以后真实条件检查 |
| Air test的`if criterion == 'MSE' or 'MSE_weighted'`恒真 | 旧MAE验证分支问题；当前main实际MSE_weighted | 不修旧分支；后续KCDNO保持真实main加权backward，不能把函数默认MSE当实际协议 |
| V2–V5未完成 | V1公共`TrainingArchive/Observer/visualization`已有，十个入口未调用；现任务仍旧weights/object保存 | 不能称现有model.pt/整对象可完整断点续训；本轮不自动完成任务resume/每50epoch绘图 |
| matched LRSA只在性能工具 | `tools/cdlno_perf/models.py:LRSAMatched`明确不注册factory | KCDNO公平对照若要进入任务需专门阶段授权；不称现有八任务已有matched训练支持 |
| 本机安装/邻居图不全 | Python目标范围外、无editable、无torch_cluster/pyg-lib | 显式源码导入和真实PyG对象可继续；不重装；远端实际图链未验证 |

不存在待用户裁定的实质核心公式冲突。新profile/目录/新类路径的具体实现可在被点名阶段按本报告建议落地，不能据此本轮写生产代码。

建议的K1–K10最小分工（**不是已批准的逐阶段指南，不是执行状态**；用户后续点名内容优先）：

| 阶段 | 建议范围与最小文件区段 | 关键验收/依赖 |
|---|---|---|
| K1 | 独立KCDNO config/metadata/profile/schema；可用`cdlno/kcdno/`子包避免改旧包导入路径 | family=kcdno；L独立、无F/P；r/phi/eps/history_mode及init版本；不重解释旧sidecar |
| K2 | 独立kernel writer/reader + 显式float64数学reference | 同核结合律输出/梯度、来源和batch轴、FP32/autocast、初始化、无Wv/Wo/self-read |
| K3 | 新KCDNO block/core，组合`cdlno.modules`原语，局部tuple缓存 | FFN1/2插入边界、T pre-Up、L1/all/off、7写7Q28读、参数独立/梯度/历史隔离 |
| K4 | 薄matched LRSA-full/noSA控制模型和结构配置（新家族范围） | full保留两FFN+SA；noSA保留两FFN；无bridge/rear/final-up；不偷改旧锁full对照 |
| K5 | 四静态任务新wrapper/有限factory/CLI/JSON/脚本/加载 | 原stem/normalizer/loss/网格/预设不变；独立family/目录、strict |
| K6 | NS/Plasticity新wrapper和有限入口 | NS10→10每次重建cache；PlasticityT/20更新、原scheduler；不新增时间实验 |
| K7 | Car新稳定wrapper/模型选择/评价/本地可信整对象加载 | 单图/可变N/x7/out4/无y或geom读取/原mask/一run一fold |
| K8 | Air新稳定wrapper/YAML新key/main_evaluation/list加载 | stem71/reference域、采样/图/weighted损失/路径双语义、成员/list严格性 |
| K9 | 复用已有性能工具补新family成本与有限CPU/GPU合成比较 | 全N投影/卷积、核摘要/stack/训练图计入；同配置同dtype；不真实训练/扫描 |
| K10 | 需求反查、八任务/新旧checkpoint回归、README/命令/最终交付 | 重放K0同权重夹具和原冻结证据；新/旧/未验分开；不自动commit/push |

可复用公共模块的源码应尽量不变；确需提取任务提升/记录helper时只在已点名接入阶段做小范围等价拆分并重放这些夹具，不改变旧类/参数路径或包装器初始化顺序。K0不预建这些文件。

交付前已自审五点：FFN1/2插入边界及Up norm只一次；新writer/reader参数归属不混入旧CDPA；当前真实git/阶段/输出路径；41份同权重数值和工业pickle真实性；冻结/诊断梯度/真实训练边界。证据分别在上文映射、fixture_index/replay、CLI/entry快照与freeze中。剩余依赖是后续新模型实现、远端目标环境及真实数据实验，不向用户抛出未经自审的架构疑问。

本K阶段结束，未执行下一阶段。
