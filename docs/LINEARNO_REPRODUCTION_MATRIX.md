# LinearNO 复现矩阵：论文 → 固定发布源码 → 目标参数/协议

## L10 当前实现状态

实现 family 为 `linearno`，固定来源为 `thuml/Transolver@75e0f67643806a81cd1d3f6adc88dd8c02416fe7`、`HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`、论文 `2511.06294v3`。L0–L10 当前均为 PASS（资源门控的真实数据/完整训练/远端环境另标 NOT RUN）。

AirfRANS resolved training objective：`airfrans_transolver_mse_v1`，标准化四通道 volume MSE + 1×surface MSE。ShapeNet-Car resolved training objective：`car_transolver_mse_v1`，全点三速度 normalized MSE + 0.5×surface pressure MSE。两项来源均为用户确认的官方代码合同；论文 physical rL2、drag、Spearman 仍在 evaluation_spec 中单独登记，不能互称。

L10 抽查 `28 passed`，全 `tests/linearno` `84 passed`；L9 综合 `118 passed`。完整结论、命令和未验证边界见 [LINEARNO_IMPLEMENTATION_REPORT](LINEARNO_IMPLEMENTATION_REPORT.md)。以下 L0 内容保留为来源审计和冲突台账，不代表当前模型尚未实现。

日期2026-09-17。此文件是**只读审计和后续实现合同建议**，没有实现任何profile、CLI或模型。来源/冻结/已执行检查见 [REFERENCE_AUDIT](LINEARNO_REFERENCE_AUDIT.md)。机器可读 [profile-inventory](linearno_audit/l0/profile-inventory.json) 包含8任务×3profile独立记录、实际官方构造参数、训练/data/objective/evaluation字段、来源和UNRESOLVED项；不是被生产代码读取的配置文件。

下文官方路径相对固定 [LinearNO@3f2b80df](https://github.com/HiPRL/LinearNO/tree/3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269)。`S=`Standard_PDE_Benchmark，`A=`AirfRANS，`C=`ShapeNetCar，`E=`Super-Resoltion-AppendixE。当前目标的实际路径使用完整benchmark名称，不能把S/A/C当目标目录名。

## 1. 决策域与三profile

| 决策域 | 证据 | 本轮处理 |
|---|---|---|
| 方法身份/QK归一化/论文指标 | 论文v3 Eq8/9、AppendixB、Table2/8 | Q沿M/K沿N；默认paper objective/metric；没有支持材料的细节明确UNRESOLVED |
| release forward、共享投影、温度、逐block卷积、初始化、key | 每任务专属固定源码 | 主复现保留发布模型；不把论文欠详细的描述解释成删除卷积/换成per-head权重 |
| paper训练表格 | Table8 | `paper_table8_on_release_model`为默认；不是paper_literal，不承诺尚未执行的Table1/2数值 |
| release训练 | shell明确值 + argparse未覆盖默认 + 实际调用 | `official_release`，train action显式eval0；shell原eval1另作为diagnostic |
| 实用数据/产物/设备/可视化 | 当前hxh实际代码 | 保留旧模型路径；新family必要变体独立记录，不能“移植”成全局回退 |
| Transolver公平超参对照 | 固定thuml及当前原脚本 | 可选`transolver_matched`；只匹配声明字段，不声称等参数、等算量或公平精度 |

`official_release`不是含糊的单开关：模型/训练配置、evaluation_spec各轴和诊断action分开。模型相同但objective或force_input不同必须留下可追踪metadata；profile名不能代替resolved字段。目录拟为 `output/<task>/linearno/<profile>/<variant>_M<M>/<evaluation-hash>/seed<seed>/<unique-id>`，保留显式save_name、禁止覆盖已有run。

## 2. 全公式映射

完整LaTeX提取见 [paper-equations](linearno_audit/l0/paper-equations.json)。下面覆盖全部34编号公式及未编号主要定义；推导不是要求增加runtime算子。

| 论文条款/公式 | 固定源码与符号 | 目标实现/metadata（拟） | 独立证据要求/边界 |
|---|---|---|---|
| Eq1、2 | 原Transolver Physics_Attention：slice logits/值投影 | 原模型冻结，非新LinearNO路径 | 固定thuml模型blob相等 |
| Eq3、4、5 | 原slice聚合、slice SA、deslice | 新LinearNO不实现Eq4，不沿用slice sum-normalization | Eq8/9 oracle不得调用旧Physics-Attention |
| Eq6 | 一般softmax attention与因式分解背景 | 只用右侧因式分解；不把左侧sqrt(d)搬入新forward | 后续tiny显式QKᵀV对照，不生产N×N |
| Eq7 | Physics-Attention重写、Q/K绑约束 | 解释旧模型；新Q/K独立 | 检查不同参数对象 |
| Eq8 | S `LinearNO*.forward`；A/C `LinearNO.forward` | `q=softmax(q_logits,dim=-1)`，`k=softmax(k_logits,dim=-2)`；记录M、heads、variant/temperature | Q[K] `[B,h,N,M]`；各Q行和1、各K列沿N和1；B>1不混样本 |
| Eq9 | 两次einsum：`bhnd,bhnc->bhdc`；`bhnd,bhdc->bhnc` | `C=KᵀV [B,h,M,d_h]`，`Y=QC [B,h,N,d_h]` | 无N×N、无slice×slice SA；M=dh时C数值方阵仍合法 |
| 主文未编号kernel/MC极限，Eq10、11、12、13 | 同一K-softmax加权和的Monte Carlo解释；无单独源码模块 | 不添加eps核归一化、ELU核或KCDNO摘要 | 此处不做定理独立证明；有限N数学oracle在L2 |
| Eq14、15、16、17、18、19、20、21、22、23及附录A未编号概率式 | 收敛论证，不是训练loss | 不实现随机采样新机制/额外目标 | 不把理论表述当实验精度证明 |
| Eq24 | Burgers PDE/GP初始条件，E/prepare.py | **八任务范围外** | 只读，不能生成/下载Burgers数据 |
| AppendixB未编号rL2 | S/utils/testloss.py `TestLoss.rel`；C/main_evaluation.py物理rL2 | 无eps、逐样本指定region/channel后范数比；industrial训练细节见冲突C08 | ratio-of-norms不是normalized MSE；零分母需明确报错/另标偏差 |
| Eq25、26 | A/utils/metrics.py `WallShearStress`、`Compute_coefficients`；C/utils/drag_coefficient.py | 力积分保留各任务已有VTK/法向/壁面/剪切处理 | 不用压力误差替代真实force；C错误velocity输入单列 |
| Eq27、28 | A角度旋转和`2*force/Uinf**2`；C `cal_coefficient` | evaluation_spec.force_metric/input/geometry/units | 原离散积分是公式实现选择，不另造面积权重 |
| Eq29 | A/C scipy.stats.spearmanr | 全test的每成员系数相关；之后member/seed聚合分轴 | 不只用n_test=3，不挑最好seed |
| Eq30 | E `CrossLinearNO.forward`/`CrossLinearNO_block` | **不移植**：八任务输入/输出点相同，无坐标query decoder | Appendix-E key_ratio不能污染Standard |
| Eq31、32、33、34 | AppendixF对LNO learned-token Cross的分析；E/module/model.py其他baseline | **不移植** learned-query Down、latent SA、Up或history | 纯LinearNO是Eq8/9，不是LRSA/MSAR多尺度 |

所有release八任务的完整block都执行 `x=Attn(LN1(x))+x; x=MLP(LN2(x))+x`；**第8层也执行两条残差后才LN3+Linear输出**。LayerNorm未显式eps，PyTorch默认1e-5。这里的MLP是一个两Linear GELU FFN，不是CDLNO的两个latent FFN。

## 3. 七变体、投影、温度、状态与初始化

记 `d_h=d/h`。先输入投影至inner_dim后split head；官方每种变体的 `to_q/to_k/to_v` 都是**一组小Linear跨head共享**，但Q/K不共享彼此参数。参数权重shape是 `to_q/to_k.weight=[M,d_h]`、`to_v.weight=[d_h,d_h]`，没有额外h维；QKV bias=False。input/output Linear与Conv有bias。

| 变体 → 固定类 | input projection / actual M | 温度 | output projection | 位置/适用 |
|---|---|---|---|---|
| plain → S `LinearNO`（行105） | Linear(d,d)；`M=key_ratio`绝对值 | 无 | Linear(d,d)+Dropout | 官方`no_temp`及未知args.model会落fallback；目标要显式校验plain |
| temp → S `LinearNO_temp`（133） | 同上 | `temperature_q/k [1,h,1,1]`初值.5，clamp[.01,1] | Linear+Dropout | Elasticity |
| conv → S `LinearNO_Conv`（10） | 每block denseConv2d(d,d,k3,stride1,pad1,groups1)；绝对M | 无 | Linear→GELU→Linear→Dropout | Plasticity；要求N=H×W、不下采样 |
| conv_temp → S `LinearNO_Conv_temp`（57） | 同conv | .5，clamp[.01,1] | 同conv | Airfoil、Darcy、Pipe |
| AirfRANS → A `LinearNO`（10） | Linear(d,d)；`M=slice_num`绝对值32 | `temperature`初值.5，**forward未使用**；`scale`等亦未使用 | 单Linear+Dropout | raw pos算[-2,4]×[-1.5,1.5] ref8距离64，并**拼接**x7→71 |
| ShapeNet → C `LinearNO`（11） | Linear(d,d)；`M=key_ratio*d_h`；1×32=32 | 拼写`tempreature_q/k`；.5，clamp[.1,2] | 两Linear GELU+Dropout | x7直接stem；official构造space3+fun4只是输入总宽7，forward实际fx=None |
| Appendix E → E `LinearNO/CrossLinearNO` | Linear input；Cross另有in_project_y；乘数M=key_ratio*d_h | 无 | 两Linear GELU+Dropout | 独立坐标/函数投影，末尾Cross；范围外 |

所有生产LinearNO路径都不对N×N token logits做softmax，也不在attention权重上dropout；`self.dropout`可能注册但未被用到，尾部Dropout要保留（主配置p=0）。input projection后softmax在自己的数值dtype执行；不可未经证据把KCDNO强制FP32策略套上而宣称逐字parity。FP32/double/AMP验证分开。

初始化严格按release时序：先preprocess/time_fc/全部blocks默认构造消耗RNG，`initialize_weights→apply`一次；Linear trunc_normal(std=.02)、bias0；LayerNorm weight1/bias0；Standard/C/E的Conv分支Kaiming normal/bias0；**最后**创建`placeholder=(1/d)*rand(d)`。temperature不被apply重设。Standard placeholder即使fx!=None仍注册；Air dead temperature保留，不能以“所有参数必须有梯度”为理由删除。完整key族：

```text
placeholder
preprocess.linear_pre.0.{weight,bias}, preprocess.linear_post.{weight,bias}
time_fc.{0,2}.{weight,bias}                         仅Time_Input
blocks.i.ln_1.{weight,bias}, blocks.i.ln_2.{weight,bias}
blocks.i.Attn.in_project_x.{weight,bias}
blocks.i.Attn.to_{q,k,v}.weight
blocks.i.Attn.to_out.0.{weight,bias}                所有变体
blocks.i.Attn.to_out.2.{weight,bias}                conv/conv_temp/Car/E
blocks.i.Attn.temperature_{q,k}                    Standard温度版
blocks.i.Attn.tempreature_{q,k}                    Car官方拼写
blocks.i.Attn.temperature                         Air未使用参数
blocks.i.mlp.linear_pre.0.{weight,bias}, blocks.i.mlp.linear_post.{weight,bias}
blocks.(L-1).ln_3.{weight,bias}, blocks.(L-1).mlp2.{weight,bias}
```

这是一份源码导出的键模式清单，**不是本轮实例化导出的state_dict/参数量测量**。空MLP.linears没有参数。Standard/C的普通`self.pos`不在state_dict；未来persistent=False buffer设备修复必须保持此集合和位置数值。A/C whole-object路径都可能是同一个字符串 `models.LinearAttnNeuralOperator.LinearAttentionNeuralOperator`，实际类不同；转换必须source_task隔离。

Standard的T先做 `Embedding.timestep_embedding` 的sin/cos（max_period=10000），再Linear→SiLU→Linear注入点特征；仅Time_Input配置注册time_fc。初始化parity需检查框架默认初始化所消耗的RNG以及最终apply的调用顺序，不能仅比较截断正态的分布。构造器对非法d/head/M必须显式拒绝，不能沿用整数除法静默截断维度。

## 4. Table8逐列对应与八任务resolved值

| Table8字段 | 固定源码取值位置 | 目标参数/metadata拟映射 | 真实命令证据 |
|---|---|---|---|
| Loss | S exp实际loss语句；A/C train+入口criterion/reg；AppendixB定义 | `objective_spec`完整类型/region/channels/space/reduction/权重 | A `-w/--weight`、C `--weight`只能改系数，**不能据此冒充切rL2/MSE** |
| Epochs/LR/Optimizer | S argparse+shell+AdamW；A params.yaml/Adam；C argparse/Adam | training epochs/lr/optimizer；不放constructor kwargs | S`--epochs/--lr`；C`--nb_epochs/--lr`；A由YAML400/.001 |
| Scheduler | S对应exp；A/C train.main | scheduler class/全参数/total_steps/实际step节奏 | 不存在统一官方scheduler CLI，不能发明 |
| Batch | S shell `--batch-size`；A YAML；C默认 | training batch及独立eval batch | S release test batch固定1，与train分开 |
| Layer/Heads/Dim | S n_layers/n_heads/n_hidden；A/C硬编码构造 | model_spec.constructor_kwargs | S`--n-layers/--n-heads/--n-hidden`；工业发布CLI并不支持自由结构覆盖 |
| Slices | S key_ratio绝对；A slice_num绝对；C key_ratio×head_dim | 对外独立linearno_rank，metadata保存actual M和raw source mapping | 不用目标旧`--slice_num`偷渡rank |
| 表外ratio/unified/ref/dropout/clip/wd | shell覆盖和未显式默认 | model/runtime/training各自字段及value source | [released-standard-commands](linearno_audit/l0/released-standard-commands.json)同时保存两套值 |

下表每任务分别对应 **paper 与release两个独立记录**。Standard两者Table8有效值相同，不意味着名字可以混用；工业loss不同见后表。所有L=8、h=8、GELU、dropout0，Standard space2，工业single graph；未在本轮构造运行。

| task | variant；d；M；FFN ratio | H×W / fun→out / position | epochs/B/lr/optimizer/scheduler | wd/clip |
|---|---|---|---|---|
| Darcy | conv_temp；128；64；1 | 85²；1→1；unified0/ref8 | 500/4/.001/AdamW/OneCycle | 1e-6/1 |
| Elasticity | temp；128；64；1 | N972；0→1；unified0/ref8（H/W构造默认85不用于reshape） | 500/1/.001/AdamW/Cosine每epoch | 1e-5/1 |
| Airfoil | conv_temp；128；64；1 | 221×51；0→1；unified0/ref8 | 500/4/.001/AdamW/OneCycle | 1e-5/1 |
| Pipe | conv_temp；128；64；1 | 129²；0→1；unified0/ref8 | 500/4/.001/AdamW/OneCycle | 1e-5/1 |
| NS | plain（官方no_temp）；256；32；2 | 64²；10→1；unified1/ref10→stem110 | 500/2/.001/AdamW/OneCycle | 1e-6/None |
| Plasticity | conv；128；64；1 | 101×31；1→4；unified0；Time_Input=True | 500/8/.001/AdamW/OneCycle，20optimizer:1scheduler | 1e-6/1 |
| Car | shapenet；256；32=1×32；2 | N可变；space3+fun4=输入7；out4；unified0；无time | 200/1/.001/Adam/OneCycle，final_div1000 | 0/None |
| AirfRANS | airfrans；256；32；2 | N可变，采样32000；space7+fun0+ref64=71；out4；unified1 | 400/1/.001/Adam/OneCycle，final_div10000 | 0/None |

两industrialAdam默认betas(.9,.999)/eps1e-8；StandardAdamW同默认。OneCycle其余未显式值遵从固定源码运行环境PyTorch默认，需记录pct_start=.3、cos anneal、cycle_momentum=True、base/max momentum .85/.95、div_factor25、three_phase=False和实际库版本。Darcy release scheduler单独固定epochs500，修改训练epochs不能静默得到错误曲线。

### 可选transolver_matched（不是已有lrsa_matched）

以目标保留的原始Transolver shell/构造作为字段来源。保留LinearNO release各任务拓扑，只匹配d/L/h/M、FFN ratio、position/ref、batch/lr/epochs/wd/clip；结果不是“等参数/算量”。必须显式声明objective/evaluation协议，不能用matched覆盖paper默认。

| 任务 | 相对上表需改的匹配项（完整值在profile-inventory） |
|---|---|
| Darcy | unified1/ref8，wd1e-5，clip.1；输入stem65 |
| Airfoil | clip.1 |
| Elasticity | clip.1 |
| Pipe | ratio2，batch8，clip.1；M仍64（不是CDLNO任务默认M32） |
| NS | ratio1，ref8，wd1e-5；保持LinearNO plain而非加入Transolver卷积 |
| Plasticity | clip.1、wd由原parser取得1e-5；time loop不变 |
| Car | 原结构/训练超参相同，任务版数据与correct surface force输入需各自明确 |
| AirfRANS | 398epochs；其他原Transolver YAML字段保留 |

## 5. objective和正交evaluation_spec

定义 `r(yhat,y)=||yhat-y||₂/||y||₂`，无eps。Standard两profile都保留原TestLoss batch sum、decode空间、Darcy导数项和时间步节奏。未知数据的零分母不自动改成eps或忽略样本；如以后发现，须报出并另标经确认的偏差。

| task/profile | training objective | evaluation field metric | 未决项 |
|---|---|---|---|
| 六Standard/paper | Table8的rL2；Darcy另0.1导数 | physical per-sample rL2；NS/Plas step/full另名 | 无数学待定；Pipe实际文件长度/顺序待有数据核查 |
| 六Standard/release | 同上，发布真实step cadence | 同上、发布eval batch1 | train需eval0；具体路径语义适配当前入口 |
| Air/paper | `L_v(rL2)+0.5 L_s(rL2)`，~surf/surf | **physical压力channel2**，volume/surface分别rL2；另报release MSE | 论文未明确训练是否all4通道、逐channel还是合并范数、norm空间，见C08 |
| Air/release | normalized all4 channels：region内points mean再channels mean，volume+1×surface MSE | normalized逐channel volume/surface MSE；不可改名rL2 | 无公式待定；图依赖/真实数据未验证 |
| Car/paper | surrounding(~surf) velocity rL2 +1×surface pressure rL2 | physical ~surf velocity3展平rL2、surf pressure rL2 | 训练velocity通道合并/reduction和norm空间欠明确，见C08 |
| Car/release | **所有点**velocity3 normalized MSE +.5×surface pressure normalized MSE | 发布main_evaluation同时报physical rL2及normalized MSE派生physical RMSE | 原force-input bug不与正常新run默认绑定 |

不要将三个不同层次混为一个字符串：**模型forward来源**是release、**训练objective**可选paper/release、**evaluation_spec**可选论文指标/发布指标及实用force输入。两profile同时保存全部字段。

| evaluation轴 | Standard | AirfRANS | Car |
|---|---|---|---|
| prediction_sampling | full points；NS10次预测回填；Plas20独立T | random32k子集，直至全场坐标覆盖，重复点预测均值，按原idx散射 | 单图所有点、原顺序 |
| field_metric | 每样本physical rL2 | paper：pressure2 physical rL2；release：normalized四通道MSE | 发布已有physical v3/p rL2，同时保留MSE/RMSE；不能只写“release=MSE” |
| field_region/channel | 任务输出全部 | ~surf/surf分离；论文报告压力 | ~surf velocity0:3，surf pressure3；与release train的全点velocity不同 |
| force_metric | 不适用 | 逐sample `abs((Ctrue-Cpred)/Ctrue)`→testmean；每member全test的C_L Spearman（可另报Cd） | release `abs(Cpred-Ctrue)/Ctrue`，Cd、全test Spearman；正Cd时等价绝对相对误差，零/负值不静默处理 |
| force_input | 不适用 | denormalize预测字段、壁面v/nut0、VTK梯度/法向/Length、原角度基变换 | 正常：surface pressure+surface velocity；诊断：release volume velocity+param0；real_sample_path是待新adapter实现的独立改进 |
| split | 明确文件及index范围 | full_test全200；scarce→full_test；aoa/reynolds对应manifest | defaultfold0；其他fold可选，论文不是9fold平均 |
| aggregation | 先sample；NS/Plas分别step/full | sample指标→member mean/std；每seed再汇总，不把成员ensemble预测当同一指标 | samplemean与whole-test correlation；seedmean/std另轴 |
| selection | final epoch/checkpoint | final每member；n_test3只export | final；不能用heldout/test loss挑权重或挑seed |
| normalizer | 保存train-fit数值，eval不重拟合 | 完整coef_norm数组及顺序 | 同左 |

paper说3次训练但未公布seed/mean/std细节；拟预声明本地 `[0,1,2]`，逐次及mean±std都报，并记录std ddof；不是作者seed。现在已有show工具选最佳seed的可视化不适合上述统计，旧工具保持不变，未来新报告独立。

## 6. 冲突台账

每项给来源、处理和影响。`已决定`来自当前用户指令或明确源码；`待审查`不自动实现候选。

| ID | 证据/冲突 | 本轮决定或两个选择与影响 |
|---|---|---|
| C01 | S/scripts六个LinearNO shell全`--eval1`，而文件名像训练 | 已决定：official_release是模型/训练配置，实际train显式0；原argv保留`released_eval_exact`诊断动作，action不进profile。 |
| C02 | S `LinearNO_block.__init__`读取`args.model`挑conv/temp；目标model_dict用同字段注册模型 | 新family独立`linearno_variant`，canonical plain/temp/conv/conv_temp；不让注册名落fallback plain。 |
| C03 | S `key_ratio`是绝对M，C/E是`key_ratio*d_h`，A叫slice_num | 对外actual linearno_rank，metadata保留原名/原值/公式；Car不能把1当M1，E乘数不带进八任务。 |
| C04 | 论文AppendixB“a single convolutional layer as a preprocessing step”；S Conv2d在每个Attn实例 | 已按用户决定保留每block卷积；论文未明确只在整个网络入口，属于描述粒度不足，不能称已证明矛盾/删除后续卷积。若实验单stem只能另experimental profile。 |
| C05 | Eq8用整体Linear描述；源码拆head后共享小Linear | release模型以源码为准，Q/K独立且跨head共享；单独per-head权重会改变参数量/假设和checkpoint。 |
| C06 | Table8/LinearNO vs Transolver：Darcyunified0、NSratio2/ref10、Piperatio1/B4；clip/wd也有差异 | 两主profile用已解析release/Table8；另matched记录原Transolver字段，旧defaults不改，不宣称只换attention的公平对照。 |
| C07 | Table8 Air系数.5/Car1；代码Air1/Car.5且MSE | 保留paper和release两行；必须同时切objective类型、mask/channel/space/reduction，不能仅改`--weight`。 |
| C08 | AppendixB `L_v/L_s`=rL2，但工业**训练**多通道聚合和decode时机没写全 | **待审查**：方案A延用release训练通道，逐通道rL2后平均、normalized训练空间（最近release的局部解释）；方案B按物理场合并范数、physical空间（贴近整体rL2公式）。Air还须明确all4训练还是仅pressure。两者梯度/单位权重/零分母不同。表2压力评价physical rL2明确，不受此待定影响；任何训练选择写本地解释，不宣称作者未公开细节。L6/L7前须确认。 |
| C09 | 论文Table2标题全部relativeL2；A `Results_test criterion=MSE`；C evaluator实际同时rL2/MSE | 两指标明确各算各报，Air不能rename；Car不能误报release只MSE。 |
| C10 | C `fold_id0`默认，load_dataset实际支持param0…8留一组 | defaultfold0；保留可选fold；无证据称作者做9fold平均。 |
| C11 | C/main_evaluation.py行67—81把~surf velocity给`cal_coefficient`；thuml/hxh给surf velocity | 正常新run保留hxh的surface修复；如要逐字released bug，仅明确诊断force_input，可能shape报错/指标无物理意义。不能全局引入bug。 |
| C12 | C/utils/drag_coefficient.py行148附近硬编码root/param0；当前helper也仍固定旧root/param0 | **待审查的新接入边界**：A保持现有限制，只有canonical path/fold0可验force；B新family adapter按真实fold/sample路径调用等价计算，不改旧helper；推荐B供L7范围确认。旧计划声称已经real_sample_path是错的。 |
| C13 | A parser debug1、score0/save_namedefault；Train.sh debug0/score1/命名目录 | official_release训练取有效shell并显式debug0；`released_cli_default_exact`另diagnostic，不能拿2train/1val debug跑数复现论文。 |
| C14 | Car preprocessed默认1；GraphDataset不等于任意fake Data | 检查真实预处理文件存在，保持节点顺序/图构造；无数据不运行，不造同名假数据。 |
| C15 | S裸state_dict；A/C whole object，A另列表；A eval期待单model且路径不同于列表 | 新artifact需显式成员manifest；纯state存档/原格式伴随文件按任务接线，不能照抄Evaluation.sh猜路径；用户提供可信pickle才转换。 |
| C16 | S unified用规则网格距离**替换**x；A从raw pos算距离**拼接**x；Car默认off | task wrapper严格分开；位置设备修复不改值、flatten顺序或state_dict集合。 |
| C17 | 八任务release有未调用set_seed函数，shell/入口无启用seed；**E有seed且会调用** | “官方无seed”只指八任务，不概括整个仓库；本地三seed须标provenance，不编造作者seed。 |
| C18 | Darcy release scheduler500 vs args.epochs；工业floor(n/B)+1；Plas20optimizer/1scheduler | 保留默认实际节奏并记录total_steps；非默认epochs需显式偏差/拒绝，不能悄悄修官方怪癖后称exact。 |
| C19 | Car train返回pressure,velocity但main命名颠倒；Air日志avg_loss是全点MSE非weighted backward | 单列真实objective和原log表达；旧模型日志不全局重写。Air test `if criterion=='MSE' or 'MSE_weighted'`恒真，paper rL2需新family显式分支，不复用该bug。 |
| C20 | Standard placeholder即使fx输入仍存在；Airdeadtemperature；Car拼写 | 均保留或做显式可逆全键映射；不删key、不补随机值、不strict=False。 |
| C21 | S hard.cuda()/非bufferpos；current运行环境不同于release | 允许后续语义保持设备修复，但做同权重位置/完整key/parity；不改旧Transolver设备代码。 |
| C22 | 实际目标bb73/无LINEARNO目录/无evaluate_checkpoint模块；旧提示词66bc/monitor/现成resume | 以实际基线为准；冻结不存在目录；monitor=N/A；L1只能独立协议/RNG基础，L4—L7新增任务resume需授权。 |
| C23 | Plas random_collate_fn用torch.randperm，旧提示词写NumPy；release data_path为目录+plas_N987_T20，target为MAT完整路径 | 保存/恢复Torch RNG；normalizer作用fx不是pos；launcher适配路径，不能改随机源或把时间展平空间。 |
| C24 | S NS期望`args.data_path+'/NavierStokes_...mat'`，shell给父目录；target再拼一层同名目录 | 默认目录层级不同，校验实际文件路径后适配，未见数据不能宣称release shell路径错误。 |
| C25 | S Airfoil行266左右 `ep%10==0 or last`，testbatch1；当前每epoch/testbatch=args.batch_size | **新family协议待接入审查**：A遵循release cadence/batch1，B保留current cadence并记录偏差；推荐A以满足双profile发布协议。改变DataLoader迭代频次会影响Torch RNG，观察回调需隔离。不是“官方无eval循环”。 |
| C26 | S Pipe省去current的`[:1200]`截断；last200可能不同 | canonical长度1200两者相同；有数据先验证len/checksum/index。不能自动换split；非1200时paper默认保留声明的canonical/current，release-exact差异明确报出再决定。 |
| C27 | A train.py行172/213特判LinearNO跳过radius_graph；current旧路径构图；eval Infer_test仍构图 | 不全局删除图。可选新family-only skip以执行release训练，或保留图并记录工程性能差异；两者采样/特征相同但依赖/耗时不同。欠torch_cluster不能据此声称完整eval可用。 |
| C28 | E配置n_block4/n_mode256；utils构造未传n_mode/key_ratio，实际d96/h8/defaultkey4→M48；另1Cross；Table8写8/256 | 范围外来源差异，记录不实施；不可用于八任务rank/default。 |
| C29 | Table3/9把parameter列写“GB”但数字形似million参数 | 后续测numel、dtypebytes、MAC/FLOP分别报；不直接引用标签或用论文表作单元测试常数。 |
| C30 | 当前show脚本按最后验证loss选seed；论文3次且本任务要求全部报告 | 冻结旧show；LinearNO另报所有预声明seed与mean/std，final权重，不沿用最佳seed挑选。 |

## 7. 发布命令、动作与目标计划参数

以下是**源码核对用命令，不在L0执行**。路径按官方工作目录解释，真实训练仍需对应阶段/开关或L10后授权。六条shell完整argv和默认补全保存在JSON，已用AST取得的真实parser解析成功；shell均GPU7，设备属于runtime不属于profile。

| task | 官方cwd | released_eval_exact | 从该argv形成train的必要动作差异 |
|---|---|---|---|
| Darcy | Standard_PDE_Benchmark | `bash scripts/LinearNO_Darcy.sh` | `eval1→0`，保留conv_temp/M64/unified0，数据根到两MAT |
| Elasticity | 同上 | `bash scripts/LinearNO_Elas.sh` | `eval1→0`，保留temp/B1 |
| Airfoil | 同上 | `bash scripts/LinearNO_airfoil.sh` | `eval1→0`，NACA目录 |
| Pipe | 同上 | `bash scripts/LinearNO_Pipe.sh` | `eval1→0`，Pipe目录 |
| NS | 同上 | `bash scripts/LinearNO_NS.sh` | `eval1→0`，MAT父目录 |
| Plasticity | 同上 | `bash scripts/LinearNO_Plas.sh` | `eval1→0`，plas_N987_T20.mat父目录 |
| Car | ShapeNetCar | `bash scripts/Evaluation.sh`，预期metrics whole object | `bash scripts/Train.sh`；显式`--batch_size 1/--nb_epochs 200`会被release parser转float，有运行风险；不要原样复制缺陷 |
| AirfRANS | AirfRANS | `bash scripts/Evaluation.sh`；任务full/aoa/reynolds各自单model路径 | `bash scripts/Train.sh`明确debug0/score1；score是动作，不是objective |

例如官方Standard真实训练argv模板（仅改变action和数据/GPU/save位置；不是当前目标命令）：

```bash
# cwd = 固定官方 LinearNO/Standard_PDE_Benchmark；L0未运行
python exp_darcy.py --model conv_temp --n-hidden 128 --n-heads 8 --n-layers 8 \
  --lr 0.001 --weight_decay 0.000001 --max_grad_norm 1 --batch-size 4 \
  --key_ratio 64 --unified_pos 0 --mlp_ratio 1 --ref 8 --eval 0 \
  --downsample 5 --gpu 0 --save_name darcy_LinearNO_release --data_path /ABS/FNO

# cwd = 固定官方 LinearNO/AirfRANS；保留parser的debug1/score0等默认的诊断动作
python main.py --model LinearAttentionNeuralOperator --my_path /ABS/Dataset
# 上面仅供released_cli_default_exact审计，不用于正式论文训练。
```

不能给现有目标入口发上述`--model conv_temp`：它是注册键，不是variant。目标建议命名为`--model linearno`（Car用`--cfd_model linearno`）、`--linearno-profile paper_table8_on_release_model|official_release|transolver_matched`、`--linearno-variant ...`、`--linearno-rank M`、独立run目录。**这些参数目前不存在，不是可执行命令**；L4—L7需按真实parser接入后替换为验证过的命令。不得复用现有KCDNO的`--profile` choices或slice_num默认静默覆盖新profile。

## 8. 新checkpoint协议与后续验证边界

拟定模型构造与运行协议分开，L1只定义schema，不在旧import路径放假构造器：

| 字段 | 必须存内容 |
|---|---|
| model_spec | `class_path`、**真实构造器接受的**constructor_kwargs；variant、actualM和raw key_ratio映射；space/fun/out、d/L/h/ratio、H/W/ref/grid范围、unified_pos/Time_Input/dropout/activation/projection/kernel/temperature由构造字段或明确derived spec完整恢复，不混task/loss/GPU |
| profile_spec | family=linearno、task、profile、每字段value/source（CLI/profile/family default），config/schema version；三profile绝不靠名字推断缺字段 |
| data_spec/normalizer_spec | 文件checksum、split/index/order/采样、命名input/output或coef_norm数值状态、dtype/shape/hash/fit split/checksum；eval绝不悄悄重拟合；无数据L0为null/NOT RUN |
| objective_spec | rL2/MSE、训练region/channel/space/reduction、权重、eps/零分母约定、NS/Plas时间聚合 |
| evaluation_spec | 本文六个正交轴及normalizer、checkpoint选择、成员/seed聚合；不把force输入bug绑入model ctor |
| provenance_spec | 三方SHA/tree、论文v3+hash、basecommit/dirty、源码/规范化patch hash、完整命令、软件/GPU/precision/backend；remote未知如实空 |
| resume_state | checkpoint_role、selection_split/metric、epoch/global_step、optimizer/scheduler、Python/NumPy/TorchCPU/CUDA RNG、显式generator/sampler状态（存在才有）、AMP状态（启用才有）；final不伪装best |
| ensemble manifest | member id/order→state_dict或伴随对象path/hash；nmodel合法/不漏成员；seed与member维度分开 |

新eval流程：读取spec→全字段校验/显式CLI冲突检查→构造→strict=True加载→新eval目录；不得先写sidecar再校验。旧无family checkpoint仍走原规则，不猜linearno。A/C保留旧可信pickle边界；用户明确提供可信官方文件时，L8按指定source_task、固定外部源码、隔离子进程最小`weights_only=False`导出纯state+metadata，全键一一映射/shape校验/拒绝未知缺失重复，strict=True后同输入逐层对照。未来本地新run可另存严格schema伴随旧任务格式，不能把旧模型保存方式改掉。

发布eval会重新读取训练数据拟合normalizer，bare模型权重没有这些数值状态；当前目标旧任务也保留相应流程。新LinearNO按用户要求保存训练时数值并在eval复用：在相同数据/拟合算法下应数值等价，数据checksum或fit split变化应拒绝，不能悄悄重拟合。此为有记录的工程改进，不宣称发布文件原本就是完整resume档案。

当前可用数学验证资源充足，但本轮没有移植模型。因此 forward/input&parameter gradients/optimizer step/strict roundtrip、B>1隔离、温度clamp、非方形H/W、fx/T/unified和投影初始化RNG测试都属于L1—L8计划，**未执行**；CPU建议atol1e-6/rtol1e-5、CUDAFP32 1e-5/1e-4，float64 oracle单独更严格阈值，任何放宽需数值证据和审查。真实loader/mini-run/4090延迟/论文精度均未验证。

**本L阶段结束，未执行下一阶段。**
