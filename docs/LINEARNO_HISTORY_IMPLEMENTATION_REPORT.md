# LinearNO latent-summary AttnRes / history-conditioned K：R10 最终交付

## A. 结论、版本与来源

**R10：PASS（授权范围内的实现与无数据验收）**。R6–R10按当前用户的连续授权依次完成，每阶段均保留报告/证据；R9发现一个R6导入边界回归，明确返回R6返修并重新通过后才继续。未执行真实数据/GPU/完整训练、commit/push、数据或checkpoint下载，没有覆盖用户已存在的tracked/untracked修改。

实际checkout `/home/hwz/CDLNO`，branch main，HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`，tree `d74a1b07aa405009992879847aca0c48c75c31f0`，实际origin `git@github.com:hxh5159/CDLNO-w.git`；远端运行目录名按用户指定是`LinearNO-monitor`，不能从remote仓库名推断本地目录名。R10起点快照 `/home/hwz/CDLNO-artifacts/linearno-history-r10-before-3sqygpwg`。R0–R9所有旧快照均保留。

来源边界：

| 来源 | 固定版本 | 本实现用途 |
|---|---|---|
| LinearNO paper | arXiv 2511.06294 v3 | Q沿M/K沿N、低秩压缩/重建、原指标 |
| HiPRL/LinearNO | 3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269 | 三套任务发布拓扑、逐block卷积、小投影跨head共享、初始化、温度与键 |
| thuml/Transolver | 75e0f67643806a81cd1d3f6adc88dd8c02416fe7 | benchmark原始参照；现有hxh增强才是当前接线真值 |
| Attention Residuals v1/官方材料 | 85e22310fe5ee860b4a023de312d791de8a5a5e6 | 深度来源选择思想，非本模型代码 |
| Kimi K3 v2 §2.2/官方材料 | 3cb39dfd32e51c3328e2e4b4af21341247d06c43 | Block AttnRes思想边界，未引入Block history |
| 用户冻结研究规格 | R1确认history_conditioned_k_v1 | A/K实际公式、共享规则、raw时序、零门、dropout与公平协议 |

A是本项目的latent-history适配，**不是原CDPA，也不是Kimi官方AttnRes**；K-conditioning没有对应官方实现。Transolver核曲线不是LinearNO实验证据；LinearNO原定理不自动证明这两个扩展。固定官方LinearNO树无LICENSE，保留来源/分发待核对的事实，不作法律许可推断、不整树vendor。R10再次核对外部reference所有记录文件和三篇论文hash，均与R0相同。

## B. 从公式反查实现

以head数h、head宽d_h、点N、每head rank M表示，完整block数量L=4..8。

| 冻结条款 | 实际实现 | 反向验证依据 |
|---|---|---|
| Z来自原LN后Linear/Conv；Q=softmax_M(Lq/tauq)，K=softmax_N(Lk/tauk)，V原投影 | `cdlno/linearno_history/core.py:attention_factors`；原`cdlno/linearno/attention.py`未改 | 六变体oracle/逐block/输入及参数梯度/一步optimizer，官方源码parity |
| shared small Q/K/V无bias；基础无sqrt scale | 每head用同一Linear(d_h,M/d_h) | key/shape/count及独立Q/K对象断言 |
| C_raw=K^T V，当前Q重建 | `attention_factors`、`LinearNOHistoryCore.forward` | 合法context[M,d_h]，无N×N，无同层latent SA |
| 每历史内部独立Cross softmax | `attnres.py:LatentSummaryAttnRes._evaluate`逐raw循环，Q_A K_A^T/sqrt(d_h) | token置换、head/batch隔离、每历史归一、profiler来源标签 |
| A全网络K/V；receiver独立Q/O/norm/w/gamma | A模块共享to_k/to_v与receivers[str(l)] | 参数对象/增量公式；无逐边参数 |
| last-axis RMSNorm eps1e-6/scale-only；w=0 | `SummaryRMSNorm`、`AttnResReceiver` | 独立oracle、分阶段branch VJP |
| 仅真实history+固定zero null参与source softmax | raw_scores→masked→cat(null)→softmax | current从未候选；all-mask alpha_null=1/H=0 |
| C_tilde=C_raw+gamma_l H；gamma无约束标量0 | A `result=current+receiver.gamma*H` | 手算、零门parity；不是C+gamma(F−C) |
| A dropout p=.1 train-only/sample×source；singleton keep、null常在、无放大 | Cross/scorer之后对logit mask；来源>1才rand | RNG/广播/全mask/eval不采样；profiler不减矩阵计算 |
| K只读旧raw bank；base to_k.weight的M行作query | `history_k.py:HistoryConditionedK.forward` | actual base-row hook、旧raw VJP；无learned slot query |
| LN0无参/末维/keepdim/unbiased=False/eps1e-6 | `history_k.py:ln0` | 双精度oracle、退化行和gradcheck |
| Uq/Uk/Uv全网络共享且与A分离；跨总S一次softmax | `HistoryConditionedK.uq/uk/uv`与bank concat | history联合置换、head/batch隔离、A/K参数对象不重叠 |
| G[B,h,M,d_h]，Delta=Z G^T−mean_N | K `uncentered`/`delta` | Delta随点与槽变化；常数slot bias无效反例 |
| per-receiver/head eta=tanh(raw_gate)，raw_gate=0，l0无gate | raw_gates仅1..L−1，构造apply后复核zero | 首步gate梯度；非零门隔离增量对Z/raw/Uq/Uk/Uv VJP |
| 先Lk0+etaDelta，再原tauK/softmax_N | K返回combined raw logits，core随后除tau | 六变体顺序与Q/V不变断言 |
| 单forward内tuple raw；block完成后才append | `context.py:RawHistoryContext`；core局部变量 | 不detach/不serialize/self状态；异常/reentry/变batch/跨真实时间无泄漏 |
| L个完整attention+FFN；末block仍LN/head | core原block路径 | core20逐次计数，L8完整执行 |
| A0K0原类/旧schema/旧loader，不构造历史 | `factory.py:build_model`与checkpoint baseline委托 | omitted/显式false的class/key/value/参数/输出/梯度/optimizer/RNG/checkpoint exact |

唯一联合顺序是：旧raw tuple → 修正当前K → 当前C_raw → A读取旧raw → C_tilde → 当前Q重建 → 原to_out与点残差 → 原point FFN残差 → 末层原LN/head（若末层）→ 当前C_raw入库。两支只共享raw引用；不共享参数、投影或mask。未引入relation scorer、MLP评分、depth bias、Block history、额外点残差、Q/V条件化、两步K更新、多样性loss或latent self-attention。

A0K0默认与显式false均保持原行为。研究模型gate=0时eval数学可精确回归，但A train会采样dropout并消耗RNG；没有声称研究训练零门时也与pure RNG相同。正式p固定.1，内部p0/mask seam不作为实验配置。

## C. 任务差异、协议和调用链

| 任务 | variant / actual M / shape | 冻结的特殊语义 |
|---|---|---|
| Airfoil | conv_temp /64/221×51 | 非方形点序，Mach输出；每block单Conv输入 |
| Darcy | conv_temp /64/85×85，unified off | decode后边界0、0.1导数项、zero-padding差分、dx1/85；发布scheduler500差异显式检查 |
| Elasticity | temp /64/972 | 不规则，fx=None，原normalizer/逐样本rL2 |
| Pipe | conv_temp /64/129×129，ratio1/batch4 | 正方形原flatten/reshape顺序 |
| NS | plain /32/d256/L8/h8/ratio2/ref10 unified | 10输入→1输出；训练10步teacher forcing后一次更新，测试预测回填；每调用清空history |
| Plasticity | conv /64/d128/h8/ratio1/101×31/out4/time | 保留历史field点序/T[B,1]；20查询各自optimizer、每batch一次scheduler；当前实际collate为per-sample torch.randperm，未擅改成NumPy |
| AirfRANS | Linear输入7/M32绝对值/Linear+Dropout输出 | 原位置与reference-distance拼接，dead temperature未用，单图可变N/32k采样、ensemble完整test |
| ShapeNet-Car | tuple/Data输入7/M=key_ratio*d_h=32 | tempreature_q/k原键、clamp[.1,2]、MLP输出、原fold/surface/drag修复 |

Standard temp/conv_temp初值.5、clamp[.01,1]，plain/conv无温度；Standard unified position替换坐标、Air拼接真实pos距离、Car按其独立语义，未统一近似。共享小投影与原Linear/LN/Conv初始化、placeholder顺序均由pure类继承。

用户明确选择的训练目标保持：Air标准化四通道volume MSE +1×surface MSE；Car标准化all-point三速度MSE +.5×surface pressure MSE。没有采用先前未确认的反归一化联合rL2解释。paper/official各自evaluation_spec的field/force/sampling/split/aggregation仍正交记录；同一objective不意味着field/force指标被合并。真实force结果未测。

实际链：

- 六Standard：`tran_evaluate/linearno_history/<task>.sh` → 原pure launcher → 原exp/parser → `cdlno_entry.parse_args`显式LinearNO分支 → research `standard_entry`/`model_dict` → 原Model或真实研究类 → 原task main → strict epoch pair；eval先读metadata再construct。
- Air：平行launcher → 原main/cdlno_entry → pure adapter显式intercept → research `air_entry` → 每member先读model/innovation spec → 原native train/指标/可视化 → 独立member pair+ensemble manifest。成员间不共享history；两类中断位置实测恢复exact。
- Car：平行launcher → 原main/cdlno_run → pure adapter显式intercept → research `car_entry` → 原数据与objective/evaluate helper → strict pair；原fold/surface velocity/drag函数复用。

原六exp、Car训练文件、全部旧模型math、数据/指标/可视化未改。Air原train仅3处受guard保护的DataLoader generator kwargs，旧分支得到空kwargs。R6返修将history import放回显式LinearNO分支，旧Transolver独立parser不依赖研究包。

## D. 配置、checkpoint与公平运行

两个bool为唯一架构真值：`linearno_latent_attnres`、`linearno_history_k_conditioning`；signature只能派生。正式A p=.1，关闭时dropout为null。family=linearno_history仅适用于A1K0/A0K1/A1K1，写innovation_spec schema_version1、architecture_extension=linearno_history_v1、真实class_path/constructor_kwargs/base variant/actual M/温度/raw/A/K语义。

model_spec只含真实构造参数；profile/data/objective/evaluation/provenance/numeric normalizer/resume/ensemble分离。metadata/构造key/shape/dtype/版本先检查，随后strict=True。研究缺spec或错A/K/depth/M/variant/head_dim/schema不猜、不回退Transolver；不使用strict=False/随机补键。baseline→research不当作resume，未增加init_from_baseline训练入口。

完整epoch pair包含模型、optimizer/scheduler、epoch/global step、Python/NumPy/Torch CPU/CUDA RNG、DataLoader generators/sampler、normalizer数值与数据/protocol hash。恢复RNG最后执行。eval不重拟合normalizer、不覆盖训练sidecar。研究源码hash改变时拒绝跨版本resume；R6–R8开发期archive仍保留，但正式实验须冻结本交付版。Air每member顺序/path/hash独立检查。外部不可信pickle不加载。

公平运行是显式runtime开关`--linearno-fair-run 1`，不是第三个创新。新parallel train launcher对四组合统一启用，DataLoader独立generator和公共主干显式复制，feature seed隔离记录。A0K0仍是原类/原checkpoint schema；展示signature和fair初始化信息只在external manifest/目录，不加创新metadata。旧命令省略新参数或只给false/false仍维持旧RNG与loader。

目录标识：`{task}__{profile}__L{L}__A{a}K{k}__seed{seed}`。远端完整八任务train/resume/eval、GPU0/1和seed17/29/43示例见[命令](LINEARNO_HISTORY_COMMANDS.md)，480预览数组见r9/commands.json。没有执行文档里的真实训练命令。

## E. 实际验收与数值证据

阶段报告：[R6](LINEARNO_HISTORY_R6_REPORT.md)、[R6返修](LINEARNO_HISTORY_R6_CORRECTION.md)、[R7](LINEARNO_HISTORY_R7_REPORT.md)、[R8](LINEARNO_HISTORY_R8_REPORT.md)、[R9](LINEARNO_HISTORY_R9_REPORT.md)。原R0–R5文档和纯LinearNO L0–L10文档保留。

| 检查 | 实际结果 |
|---|---|
| R6静态4题×4模式 | 16原生真实空间shape合成闭环、80正式preset；返修后再次exact |
| R7时间2题×4模式 | 8原生合成闭环；NS60train forward→6更新、Plasticity120forward→120更新/6scheduler；mask/RNG/query/batch顺序恢复exact |
| R8工业2题×4模式 | 8真实PyG合成闭环；Air四模式双member、成员内/成员间两恢复边界exact；12配置冲突在torch.load前拒绝 |
| 核心4模式×L4..8 | 20构造/forward/backward/optimizer/完整block计数/严格独立进程reload/eval通过 |
| 正式preset | 八任务×五深度×四模式160参数量/构造与矩阵MAC账通过；24六变体小配置理论/实际MAC精确相等 |
| R9所有旧测试扫描 | 62模块578方法；新增导入回归已返修；最终537通过、6历史失败方法11断言、35资源/授权skip、0error |
| R10关键数学/strict/原模型 | 63方法70.732s：62通过；仅旧legacy方法3个既有hash/prefix断言失败，0新失败/0error |
| R10官方parity与输出RNG | 另23方法36.497s：19通过、4CUDA跳过；CPU官方原语/完整模型、初始化、梯度/step与随机流检查通过 |
| 新诊断 | 48组六变体四模式train/eval输出/梯度/RNG精确相同；whole-object保存、异常恢复、dense小oracle通过；独立native eval输出2图 |
| CPU性能 | 20同深度组合+30预声明匹配pure对照；warmup5/measure20；无N×N，无同层SA；A train/eval矩阵量相同 |
| 真实loader/mini-run | NOT RUN，未授权真实数据 |
| GPU/AMP/远端 | NOT RUN，CPU-only授权；不能继承其他模型旧GPU结果当作本研究验收 |
| 完整3seed/论文精度/SOTA | NOT RUN，不作性能提升或SOTA声明 |

R10详细数值在`linearno_history_audit/r10/{core-parity,a-oracle,k-oracle,integration,numerical-summary}.json`。No-op研究core与pure在六变体FP32/FP64、逐block/最终、输入/全参数梯度及optimizer step均max_abs=0；A0K0 omitted/explicit也exact。独立A输出oracle FP64 max/mean abs=1.110e-16/5.011e-18，FP32=5.960e-8/5.484e-9；K combined logits FP64=4.441e-16/3.073e-17，FP32=2.384e-7/1.705e-8。包含全梯度时最大abs为A FP32 1.788e-7、K FP32 9.537e-7；未放宽继承门限（CPU FP32 atol1e-6/rtol1e-5，double按独立测试更严门限）。完整mean/relative误差和relative-floor见JSON。

隔离branch VJP真实非零：K指定旧raw0/1/2的梯度L1约10.038/12.579/13.604，Z27.077，Uq/Uk/Uv22.340/19.725/25.183；当前C/future无梯度。A先gamma≠0/w=0再gamma/w都非零，前一步norm scale梯度0符合预期，后一步非零。未用总loss沿主干的梯度冒充历史分支证据。

现有失败不是隐藏的模型parity失败：README/path/旧matrix前缀、旧工业AST快照仍缺少在R6之前就有的pure LinearNO接线。R9原始失败日志、pre-R6重放与R6返修证据全部保留，未删测试或改golden。不能将本报告表述为“全库测试全部通过”。

## F. 参数、计算、监测与公平实验顺序

以下是实际注册参数，L8，不借用任何论文其他模型参数量：

| 任务 | A0K0 | A1K0 | A0K1 | A1K1 |
|---|---:|---:|---:|---:|
| airfoil | 1,765,889 | 1,770,216 | 1,766,713 | 1,771,040 |
| darcy | 1,766,145 | 1,770,472 | 1,766,969 | 1,771,296 |
| elasticity | 585,217 | 589,544 | 586,041 | 590,368 |
| pipe | 1,765,889 | 1,770,216 | 1,766,713 | 1,771,040 |
| ns | 3,377,921 | 3,394,760 | 3,381,049 | 3,397,888 |
| plasticity | 1,799,428 | 1,803,755 | 1,800,252 | 1,804,579 |
| airfrans | 3,358,788 | 3,375,627 | 3,361,916 | 3,378,755 |
| car | 3,852,420 | 3,869,259 | 3,855,548 | 3,872,387 |

A新增参数=2d_h²+(L−1)(2d_h²+2d_h+1)，K新增=3d_h²+(L−1)h；含norm/w/gamma/gates。全部160条正式depth/mode账与各Linear/Conv/point FFN/time/stem/head、K^TV/QC、A/K历史运算MAC见r9/task-costs/results.json。FLOPs表为2×矩阵MAC，scalar/reduction/norm/softmax/GELU等另有实际算子清单，不声称2×MAC等于全部标量总FLOPs。NS/Plasticity表是一次模型调用，不是整段训练或epoch。

CPU最终样例：N972/B2、d32/h4/M16、L8，pure 39,841参数、103,825,152矩阵MAC、推理median8.434ms；AK 41,204参数、113,838,848矩阵MAC、median16.266ms。两者p90分别9.703/18.396ms；synthetic AdamW/MSE吞吐分别54.97/38.47样本/s。该窗口显示同深度开销，不能从小模型CPU外推真实GPU/epoch。完整L4–8四模式和matched control的median/p90/原始窗口在performance-final/；GPU峰值显存NOT RUN。

可选监测记录A gamma/real-null权重/source熵/drop率，K eta/修正范数比/K熵，Q/K/P层相似度，raw统计和gradient norm。与旧monitor分离，默认不启用；低秩P不建N×N，只描述当前block路由而非含A的完整Jacobian。已实际查看PNG；初始小模型接近均匀路由，固定0..1色标呈接近1的热图，保留精确JSON，不将初始化图解释为训练后塌缩。用法与运行成本说明见[诊断文档](LINEARNO_HISTORY_DIAGNOSTICS.md)。

完整实验尚未启动。未来先固定源码/数据checksum/硬件/precision，再按任务运行pureL8与四模式同深度、增强L4–7及对应pure、预声明global-rank匹配强对照。paired seeds17/29/43、共享公共主干初值/独立feature seed和data generator、相同objective/metric/batch/epoch/optimizer/scheduler、final checkpoint。逐seed与配对差值、mean±sample-std均报告；不得按test挑选配置/seed/checkpoint。240个八任务参数/MAC匹配对照已按固定nearest合法rank算法列出，Car必须整倍d_h；未训练，匹配偏差明确。accuracy–parameter/FLOPs/latency Pareto均NOT RUN。

## G. 文件保护、自审和剩余边界

实际R6–R10文件及SHA/size/tracked-untracked-ignored分类在`linearno_history_audit/r10/r6-through-r10-changes.json`。新增8个平行launcher、独立task适配/公平运行、测试workers与证据、可选monitor/tools。修改9个既存tracked路由/测试投影文件；既存研究factory增加公共权重显式复制。全部原六exp、三套pure模型、原Transolver/CDLNO/KCDNO/MSAR数学、原数据/指标/可视化源码保持不变。R0 frozen42个模型目录/原monitor核心hash相同；R10扩大到100个旧模型/数据/metric/monitor/六exp源码重新逐字节核对也全部相同（protected-source-recheck.json）。完整旧源码identity、纯source fingerprint与原生闭环另证接线路由语义。

没有新增非ignored的数据/checkpoint/output/图片/clone/wheel混入git变更。ignored pyc是已有/测试产生的开发缓存，逐项登记，不清理用户文件；所有大合成checkpoint和图像在外部CDLNO-artifacts或临时目录。没有stash/reset/clean/checkout/rebase/merge/commit/push。R10只读审计本身不改生产源码/测试/配置；末次freeze给出文件级证明。

五个最值得人工抽查的点（已自审，不是尚未裁定的设计）：

1. `core.py`与`context.py`：当前raw仅在完整block后写入一次，A/K看同一份旧tuple，NS/Plasticity每次forward独立。时序/VJP/异常重入测试已通过。
2. `attnres.py`：history-only+zero null、gamma=0、singleton/all-mask、dropout发生在Cross后，norm scale分阶段梯度。oracle与参数对象检查已通过。
3. `history_k.py`：base to_k行query、全网络独立U、N中心化、per-head tanh0、先加logits再过原tau。常数bias反例/六变体/隔离VJP已通过。
4. task adapter与checkpoint：Air M32/dead temperature/ensemble逐member、Car倍率M/拼写/MSE、metadata-before-construction/strict/load冲突以及旧parser隔离。32生产合成闭环和返修后旧parser已通过。
5. 公平性与解释：新launcher A0K0是显式fair runtime，旧省略flags仍原路径；public/feature/data RNG分开；cpu小配置效率不可推断真实精度。对应manifest/恢复/480命令预览已核对。

仍未验证：远端Python3.10/torch2.11/cu128/PyG栈、新模型GPU/AMP、真实数据文件/split/checksum完整性、VTK force完整评价、真实batch/mini-run、收敛/精度/真实epoch速度、三seed完整论文结果、SOTA。有限合成结果不能替代这些项目。官方无LICENSE这一来源事实和现有6项历史测试漂移保留在交付边界中，没有擅自扩大实现或修改其他研究架构。

**R6–R10完成。最终状态：PASS（上述授权范围）。停止；未启动真实训练，未commit/push，未执行后续阶段。**
