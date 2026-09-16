# 最新：MSAR-LNO M9最终交付完成（2026-09-17）

M8已批准，仅授权M9，本轮已完成成本/有限GPU/独立审查/最终文档，停止。报告`docs/MSAR_LNO_IMPLEMENTATION_REPORT.md`，26条矩阵`docs/MSAR_LNO_REQUIREMENTS_MATRIX.md`，证据`docs/msar_lno_audit/m9/`。模型、任务、数据、旧测试和依赖均无改动；新增性能适配/CLI和6个新测试，只给旧measure.benchmark加可选loss_closure（默认旧MSE不变）。README保留旧文新增MSAR节，八任务训练评估文档补Full off。

真实16个task/profile参数与全矩阵MAC核对完成，覆盖两FFN/全部norm/QKVO/N规模投影；coverage0参数、Pair仅3d。13个GPU行：N972/B1七模型含Light/Full off/floor、Transolver、生产matched全LRSA、KCDNO all；AirN32000/B1四个math MSAR及两个Light efficient。FP32/TF32off/AMPoff/compileoff，warmup5/20 CUDA同步，raw timing/peak/optimizer状态可查。不是等宽/等参数对照或dataset epoch。Light efficient step58.047→61.876ms，峰值351.937→1221.181MiB；normal eval同权重off/floor exact，math本身存权重不能把两backend混同。

性能20方法全过；36独立数学/core过；新增命令方法首次错误helper flags参数，修正测试后过，最终57个不同方法分次通过。32真实train parser、32evaldryrun，未执行真实入口。M8的75同权重/原loss/工业保存证据保留并核对182artifact hash；非M9重放75次。GUNet和完整Air采样torch_cluster限制保留。

快照`/home/hwz/CDLNO-artifacts/msar-m9-before-e8vxz5y5/source`865文本保留；已有6文件变（measure、root/脚本README、3状态memory），859相同。新稳定MSAR类路径/checkpoint协议不改，无新resume。Py3.13.9/Torch2.13cu130/PyG2.3.1/RTX5090Laptop本机证据，目标2.11cu128未验；真实数据完整读取/收敛/准确率/epoch/SOTA/完整工业后处理未执行。无安装/下载/commit/push/真实训练。**本M阶段结束，未执行下一阶段。**

# 历史：MSAR-LNO M8验收完成，等待审查（2026-09-17）

用户批准M7，仅授权M8。本轮不改生产/旧测试；新增`tests/test_msar_acceptance.py`、`docs/MSAR_LNO_M8_ACCEPTANCE.md`、`docs/msar_lno_audit/m8/`，更新独立/旧STATUS。完整矩阵与实际命令见报告，下一阶段M9未执行。

正式Light八任务off/floor原loss合成训练步/eval16例通过；正式Full八任务单forward合成MSE+coverage backward/strict权重零容差往返通过。NS N4096/10步，Plasticity N3131/20更新；其他小N，Car/Air真PyG单图。Full未跑原loss完整优化矩阵。任务checkpoint协议仍按M5—M7小结构重跑，Full纯state另测，不混称正式Light整对象协议验收。

同权重off/floor最大差2.235e-7，容差2e-6/1e-4；默认kappa.2初始raw均0合法。新边界首次过强要求所有层每次非零，最深层均匀slot触发；仅改新测试检查连接/有限，并在四个真实Down变化source上各证非零梯度。无decoder/fusion aux梯度，三w0 Pair exact E+U，执行Down4/Up4/Pair3/block12，连续独立backward不残留。

新19方法首轮18过/1失败97.619s，修正定向1/1过.122s；既有MSAR107项106过/1跳过126.930s；旧157项0fail/error，1 Air抽样子用例跳过181.846s。75旧同权重fixture exact，182外置artifact文件hash相同；GUNet仍缺torch_cluster未执行。三cwd factory/metadata/严格保存协议验证通过，无exp/main import。

本地Py3.13.9/torch2.13cu130/PyG2.3.1/RTX5090Laptop；八任务d8 GPU原loss步及M2/M3有限AMP通过。真实数据/收敛/精度、远端2.11cu128、正式profile GPU/任务AMP、完整工业采样/VTK力系数未执行。旧Car drag固定路径/fold0及日志问题不改，无新resume。

837文本起点`/home/hwz/CDLNO-artifacts/msar-m8-before-fhesnwse/source`保留，834字节不变，仅三状态文档增量，模型/训练/数据/依赖/旧测试全冻结。无commit/push/安装/真实训练。**本M阶段结束，未执行下一阶段。**

# 历史：独立MSAR-LNO M7工业任务完成，等待审查（2026-09-17）

## 2026-09-17：MSAR-LNO M7完成，旧模型保持

仅M7；M6已批准。Car/AirfRANS真实工业入口、独立wrapper/family/Light或Full/coverage显式loss/严格整对象或列表checkpoint已接入。单图batch1、变长N、reference/通道/surf及原加权loss保留；loader图构造/采样/散射/物理指标、旧模型、MSAR数学及PDE入口不改。eval先读架构/任务信息，coverage覆盖只记录请求；旧可信pickle边界不扩大，无新resume。

报告`docs/MSAR_LNO_M7_INDUSTRIAL_TASKS.md`、独立STATUS和八任务命令`tran_evaluate/msar_lno/README.md`、证据`docs/msar_lno_audit/m7/`。MSAR107项106通过/1跳过（Air torch_cluster），166.280s；旧相关90项有两处测试适配问题，修正后定向2项通过，原模型回归无遗留失败，仍2处既有Air skip。14份旧工业同权重exact，182夹具hash保持。真实PyG原loss步骤、Car完整合成epoch、Air记录/scatter片段、同模式checkpoint/新cwd、两工业GPU FP32小结构及正式Light/Full小N前向通过。Air完整采样epoch/VTK指标、真实数据训练/收敛、远端2.11cu128、工业AMP、大profile反传/性能未验证。

保留pre-M7快照`/home/hwz/CDLNO-artifacts/msar-m7-before-bocerv4d/source`和先存用户工作。6个工业完整入口/训练AST剥离精确新分支后与pre-M7相同；788起点文件字节相同。Car原drag固定路径/fold0、原日志压力/速度交换及Air日志拼写缺陷仍保留，不借新模型修复。下一阶段M8未执行，无真实训练/安装/commit/push/PR。**本M阶段结束，未执行下一阶段。**

## 2026-09-17：MSAR-LNO M6完成，等待审查

仅M6；M5已批准。新增`cdlno.msar_lno.temporal.TemporalModel`、NS/Plasticity薄factory/JSON/脚本，接真实两个exp的显式aux/loss/日志；原MSAR core、旧模型/数据数学不变。NS fx10→out1，训练10真值回填/1更新，eval10预测回填；coverage是实际forward×四层均值，仅加一次。Plasticity fx1/T[B,1]→out4，保留20独立optimizer+1 scheduler，每次加入本次coverage；无时间latent/cache、无卷积。默认Light、可Full/off；strict裸state_dict不提供optimizer/RNG resume。

报告`docs/MSAR_LNO_M6_TEMPORAL_TASKS.md`、状态`docs/MSAR_LNO_IMPLEMENTATION_STATUS.md`、命令`tran_evaluate/msar_lno/README.md`、证据`docs/msar_lno_audit/m6/`。最终MSAR93通过、旧93零失败/2skip（Air torch_cluster），新11方法。两任务×floor/off B2真实N小d训练/eval/原loss及checkpoint、两CUDA FP32 MATH完整时间batch、真实Light/Full CPU前向通过；大profile反传/远端2.11cu128/真实数据训练/新工业PyG未执行。26旧+4pre-M6 M5同权重回放exact；182旧artifact hash保留。

保留外部快照`/home/hwz/CDLNO-artifacts/msar-m6-before-pqh_ca0e`（784文本+4新fixture）；两入口去掉MSAR分支完整AST等同pre-M6。起点13已有文件改、771不变；旧用户工作保留。NS实际只有10→10无20/40 CLI，不能虚构长时支持。M7未授权；下一步仅等待用户审查。**本M阶段结束，未执行下一阶段。**

仅M5，见[报告/覆盖表](../docs/MSAR_LNO_M5_STATIC_TASKS.md)、[命令](../tran_evaluate/msar_lno/README.md)及独立MSAR STATUS。
四真实exp现已接msar_lno：原坐标/fx/normalizer/loss/调度保留；共用无卷积wrapper+原M3core，Light/Full不裁M。
M4显式coverage adapter只进新family；off/weight0严格原LPDE，四项目标step均值与原train_loss分开标注。
原state_dict频率、read-first/strict加载/sidecar不覆盖与early输出目录记录；新JSON/薄脚本齐备，无新增resume。
MSAR82项通过，相关旧79项零fail/error、2处Air torch_cluster skip；四任务floor/off原loss合成步/评估/往返通过。
正式Full Elasticity N972/M1024前向通过；四任务小GPU原loss step、新cwd严格加载、24train/eval parser预览通过。
旧65夹具与M3两core本轮零容差回放，其他10旧夹具沿M4有效证据；182旧artifact hash保持。
快照msar-m5-before-2jli41sc/source保留；四旧入口完整AST不变，旧数学/数据/依赖冻结。
没有真实数据/训练或远端验收；NS/Plasticity/Car/AirfRANS MSAR仍未接入。无commit/push/PR/安装。
本M阶段结束，未执行下一阶段。

# 最新：MSAR-LNO M4公共接入基础完成，等待审查（2026-09-16）

仅M4。详见[报告](../docs/MSAR_LNO_M4_INTEGRATION.md)与[独立STATUS](../docs/MSAR_LNO_IMPLEMENTATION_STATUS.md)。
PDE真实factory独立msar_lno→稳定lifted-core类；family专用kwargs、显式aux/loss与四项日志、严格core state/whole/list checkpoint完成。
off/weight0直接原LPDE，无A；eval先读sidecar再校验，coverage覆盖不影响纯eval，旧pickle边界和保存格式不变。
M4最终18项+旧MSAR基础49项+旧入口33项通过；75旧同权重和2份M3 core精确回放，182旧artifact hash保留。
三cwd加载、本机小CUDA FP32 loss/往返和本轮M2/M3有限AMP通过；所有任务脚本/旧模型/math/数据/依赖本轮未改。
这不是八任务接入：任务lift/normalizer/原loss/loop/日志/输出Run仍待M5—M7；无新增task resume或真实数据训练。
保留23处用户先存tracked修改；快照msar-m4-before-i75l1rvp/source。无安装/commit/push/PR。
本M阶段结束，未执行下一阶段。

# 最新：MSAR-LNO M3四级core完成，等待审查（2026-09-16）

只实施M3。读docs/MSAR_LNO_M3_CORE.md及MSAR_LNO_IMPLEMENTATION_STATUS.md。
新cdlno.msar_lno.core.MSARLNO接lifted[B,N,d]、内含LN+Linear head；任务lift后续接，勿重复head。
Down4/Up4、encoder6/decoder6、SA12/FFN24/fusion3；D4直接Decoder4(E4)，3→2→1先Up再fusion再Decoder，
final无E0skip/point residual。默认Tensor；显式aux为MSARAuxOutput，coverage仅四Down raw mean。
off/weight0不请求Down A/coverage图；显式diagnostics按no-grad重算或复用观察A，默认不运行且无持久状态。
16core+20M2+13M1=49测试通过，47旧夹具本轮精确回放，其他28沿用M2；182旧fixture hash不变。
Light/Full正式参数小N前向和新夹具精确回放；out4无lift时参数1691260/6737140。
可信whole/list三个原cwd新进程通过，小core本机CUDA FP32/FP16/BF16 AMP通过；远端/真实训练未验收。
720起点文件仅3状态/记忆增量，M1/M2/共享/旧模型/任务/数据/训练/依赖/已有测试未改。
快照msar-m3-before-jx4v53vh/source与post-m3-core保留。没有八任务MSAR接入，未执行M4。
无安装/commit/push/PR/真实训练。本M阶段结束，未执行下一阶段。

# 最新：MSAR-LNO M2原语完成，等待审查（2026-09-16）

只实施M2，读docs/MSAR_LNO_M2_PRIMITIVES.md和MSAR_LNO_IMPLEMENTATION_STATUS.md。
新增cdlno/msar_lno/modules.py五组件、独立tests/msar_reference.py及test_msar_modules.py。
Down只有训练+return_aux+coverage有效才显式A；off/weight0/eval走SDPA；P直接Q，无Wq。
双FFN-SA三残差完整，Up纯branch，fusion只有w且固定2/raw values/零初始，coverage source求和、
batch/层均值且FP32关闭autocast。可选诊断no-grad显式调用，不持久存A/损失图。
20新+18旧公共+13 M1=51测试通过；75 M0同权重回放精确，182夹具hash不变。
有限本机FP32/FP16/BF16 AMP通过；远端/真实数据/完整模型和任务接入未执行。
起点695文件除3份状态/记忆文档增量外均冻结；快照msar-m2-before-bovueh8k保留。
未改共享源码、M1、factory、任务/数据/训练/评估/依赖/已有测试；无安装/commit/push。
没有core，下一阶段须明确授权。本M阶段结束，未执行下一阶段。

# 最新：MSAR-LNO M0补审完成，M1保持，等待审查（2026-09-16）

用户纠正阶段顺序：本轮仅M0审计，核对已有M1兼容，不修改M1或实施M2。
读docs/MSAR_LNO_REFERENCE_AUDIT.md、MSAR_LNO_IMPLEMENTATION_STATUS.md及
docs/msar_lno_audit/m0/{READING_LEDGER.md,fixture-index.json,freeze.json,regression.log}。
653文件静态清单/合同；75同权重回放（41旧K0+6真正pre-M1+24新wrapper+4其他旧模型）精确。
316tests/576.961s零fail/error，2个Air torch_cluster依赖skip；GUNet图链亦未运行。
M1无已发现阻断冲突，仍仅配置；Down mask/A须新family独立实现，旧block/readout不直接复用，
AttnRes按每尺度只有w且总3d约束使用无新增scale评分RMS，pointwise_mlp不是final点残差。
本轮快照/home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s与pre-M1/K0等产物均须保留。
没有生产/原测试改动或依赖安装，无真实训练/远端验收/commit/push。后续只执行明确点名阶段。
以下M1早于M0的记录原样保留，不能倒填时间。本M阶段结束，未执行下一阶段。

# 最新：MSAR-LNO M1配置基础完成，等待审查（2026-09-16）

仅M1获授权，见docs/MSAR_LNO_IMPLEMENTATION_STATUS.md和docs/MSAR_LNO_M1_CONFIGURATION.md。
新cdlno.msar_lno提供family=msar_lno、Light/Full四级配置、显式CLI解析协议、coverage目标/运行记录、
严格sidecar及family/profile/effective-coverage隔离路径。没有MSAR模型/辅助loss实现，没有八任务接入。
旧生产代码/原测试/数据/依赖不改。32配置测试和41 K0+6修改前KCDNO/matched核心同权重回放全部通过。
快照/home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j与既有K0夹具须保留。
用户已批准M0，但本工作区没有MSAR M0报告/索引；报告明确该证据缺口，不把本轮源码索引冒充完整M0。
无真实训练/下载/安装/commit/push；未执行M2。继续工作需用户明确下一M阶段。

# 最新：离线报告仅展示各task最后验证损失最小的seed（2026-09-16）

读取docs/CDLNO_RESULT_REPORTS.md最新补充和tran_evaluate/show/README.md。show三个脚本命令不变，
共享_report.py先按显式筛选纳入候选，再每个task选单run。last validation不是best-ever；不使用
独立test成绩选seed。PDE六task分别选取，pde_selected_training/evaluation各一张2×3图。
selection.json/CSV保留全部候选理由，其余图表/场图只来自获选run；源实验/其他seed/旧报告不修改。
工业使用正确分项+保存权重；Air按最后真实验证epoch和run内成员均值；缺损/非有限/未完成不参选；
legacy无status但有历史可参选且标未核实。混设置会提示，可用model/run-dir限定比较组。
14/14定向检查通过，真实PDF/PNG/HTML/ZIP渲染查看，149生产/启动+3show shell字节未变。
证据docs/show_audit/selection；快照/tmp/show-selection-before-cdua003g。只处理结果文件，
不改模型/数据/训练/依赖，未访问远端真实产物或训练。本阶段结束，未执行下一阶段。

# 最新：离线训练/测试报告脚本交付（2026-09-16）

只新增tran_evaluate/show的三个子项目shell与共享_report.py、测试/文档。默认实际checkout
output/runs/project metrics+scores，子项目result_visualizations时间戳目录+ZIP。训练loss、
独立eval、Air系数配对/CpCf、已有周期场图、CSV/LaTeX/HTML/来源hash；保持seed/fold/member/
evaluation分开，不平滑不补造。原149个生产/启动文件byte未变，8个定向测试通过，模拟远端
带空格路径和真实shell，已看合成格式示例。无模型导入/GPU依赖/真实训练/远端真实产物验收。
阅读docs/CDLNO_RESULT_REPORTS.md及tran_evaluate/show/README.md。当前任务完成即停止。

# 最新：八任务每50轮场图已接入（2026-09-16）

当前请求只授权可视化，不是resume或架构改动。读docs/CDLNO_PERIODIC_VISUALIZATION.md。
CDLNO/KCDNO/lrsa_matched八任务记录器每50completed epoch+final导出2固定held-out案例，
paper-style PDF/600dpiPNG/NPZ/英文与LaTeX caption，原Transolver维持原路径。
9个entry/train只增加观测调用+Air归一化转发；161保护文件byte同预快照
/home/hwz/CDLNO-artifacts/visualization-before-133lyhbo/source，保留seed等已有修改。
287tests无fail/error，Air缺torch_cluster两项skip、LRSA缺env项随后2tests补跑通过。
最终focused11tests仅Air建图skip；Car原完整合成epochs出图、权重/RNG精确，GPU有限确定性
测试后nextstep精确（最初未固定确定性1.9e-9差有记录）。无真实数据/训练或远端验收。
可视化已集成不等于旧V2–V5全部完成；checkpoint/resume保存逻辑未改。详细证据在
/docs/periodic_visualization_audit，用户当前任务完成后停止，不自动后续阶段。

# 最新KCDNO交付状态（2026-09-16）

远端报unrecognized --seed0：本地旧parser可复现，当前parser已支持。只同步两个新脚本
不够，还要三个Python helper（PDE/cdlno_entry、cdlno/kcdno/entry、cdlno/experiment）。
新增docs/kcdno_audit/seed_suite/seed-support.patch，已对独立旧副本apply-check/apply验证；
上传方法和无数据parser核查在KCDNO_SEED_SUITE.md。远端未访问/未自动修改，无实际训练。

最新额外任务：六标准任务单seed队列已完成。`tran_evaluate/kcdlno/run_seed.sh 0|1|2`
按darcy→airfoil→plasticity→elasticity→ns→pipe，每任务train/eval/带seed即时输出后再下一个。
标准新家族helper可选真实RNG种子、初始化/config/result记录；不传seed保持旧行为。
报告docs/KCDNO_SEED_SUITE.md；8+2定向检查通过，195旧源文件192不变（3seed/记录helper），
模型/六exp/预设/旧脚本未改，无实际训练/新增GPU/远端验收。结果记录/失败停止和60条
安全命令预览通过；默认GPU0，保持原训练协议，无新增resume。之前记录保留如下。

新增 `tran_evaluate/kcdlno/` 八个数据集薄启动脚本，复用现有 `kcdno` 入口并支持
train/eval/train_eval、all/off/lrsa_matched 和 dry-run；最终9语法+120保护性分发检查
通过（112条命令预览），173已有源码/脚本hash不变。帮助解析初次误入Darcy数据入口、
因缺数据而失败，已修复且清理本次失败目录；最终检查禁止Python调用，全部通过。
未运行真实训练。说明/证据见 `tran_evaluate/kcdlno/README.md` 和 verification.json。
实际模型 family 仍为 `kcdno`，没有改动模型或训练协议。

用户连续授权K4–K10已全部完成并自审，现停止。八任务kcdno all/off与lrsa_matched可选择并保留原协议；257最终检查OK/1个Air缺torch_cluster子项skip，补1时间off步测试通过；41旧同权重回放精确、119冻结文件和15完整AST/YAML投影通过。有限两配置本机GPU性能all未显示普遍提速。报告docs/KCDNO_IMPLEMENTATION_REPORT.md、需求矩阵、KCDNO_COMMANDS.md和docs/kcdno_audit/k10。没有真实数据读取验收/训练/精度/远端2.11cu128/精确resume；未commit/push/安装依赖。旧状态历史保留如下。

# Current project state

## Latest: K5 complete, next K6 under continuous K4–K10 authorization

Report docs/KCDNO_K5_TEMPORAL.md.12tests/8oldtimefixturespassed; original NS10step/Plasticity20step AST executed with synthetic inputs, schedules1/batch preserved, no cacheacrosstime. Sixstandardwrapper/profile/CLI/checkpoint done; no productionindustry yet. K5 snapshot /home/hwz/CDLNO-artifacts/k5-before-u6632u10/source. Original test scheduler expectation corrected from0to1, productionloops unchanged.


## Latest: K4 static integration complete; user authorized sequential K4–K10

Read docs/KCDNO_K4_STATIC.md. Current explicit user instruction overrides earlier stop-after-one-stage policy: implement/review each K4..K10 in order, then final stop; no real training/download/commit. K4 fourstatic completed6tests/16old same-weight fixtures/freeze. New wrapper/core-head, entry resolution/factory/task sidecar/scripts; old math/data/loss loops unchanged. K4 snapshot /home/hwz/CDLNO-artifacts/k4-before-lxbxc_wh/source. K5 temporal is next, not yet done.


## Latest: K2 approved; K3 feature core complete (2026-09-15)

Only K3 authorized. New cdlno/kcdno/core.py exports explicit KCDNO/KCDNOBlock. Independent L complete point blocks: point RMS/Down/FFN1 residual/K2Reader/FFN2 residual/one Up latent RMS/Up point residual/point FFN or dense Conv residual. Source T is exact pre-Up FFN2 residual; Writer(T) after point work, first no reader/last no writer/L1-off no history params. Core creates local growing list, passes immutable tuple snapshots, returns only features[B,N,d]; internal new block returns points/optionalsummary. No lift/head/bridge/rear/finalextraUp, no oldSA or norm, no KCDNO task registry yet. Explicit grid tuple for Conv, variableN forpoint. Threehidden fixed2d per K3, other K1-resolved widths rejected atcore construction. No production changes except newcore; no initializer recursion.

Report docs/KCDNO_K3_CORE.md, independentSTATUS/evidence docs/kcdno_audit/k3.57/57tests9.912s(new10,K2 16,K1 13,modules18);9/9K0 oldCDLNOfrontxCDPAcore same-weight/input/output+gradient exact replay. L1/2/4/8/12 all/off,5x7conv,independence/initialization/rawT/Upnorm/tuplecausality/isolatedwritergradient/A-B-A/offgamma0/strictstate passed. Independent primitives + double explicit K2 kernel oracle one/two layers point/conv: outputmax3.652e-7 at2e-6/3e-5,gradientmax1.959e-6 at2e-5/3e-4. LimitedlocalGPUcore2FP32+4AMPcasespassed,MathSDPA TF32off; AMP output tolerance plusfinitegrad,notfullgradientparity. K2 original AMPactualFP32ops/gradientchecks rerunpassed. L8defaultactualDown8Up8latentSA0latentFFN16Point8writer7Q7read28ReaderAPI7num/deneinsum7eachsourcesoftmax7SDPA16. Allparameters featurecore only:d128h8M64r16pointall2605063/off2572800;convall3785735/off3753472. No latency/performance claims.

Startmain222647f with approvedK1/K2dirtywork preserved.422filesnapshot /home/hwz/CDLNO-artifacts/k3-before-bd2pluhq/source;420unchangedonlySTATUS/memoryincremental. No existingmodels/modules/CDPA/K1/K2/config/entry/training/data/script/tests/depsmodified. LocalPython3.13.9torch2.13cu130RTX5090;remote2.11cu128unrun,noinstall. No newtaskloss/PyGindustrial/wrapper/head/checkpointbinding/resume/datareading/realtraining. ExistingK0/K2unaffectedfulltaskevidence retained,notrerunorclaimednew. Current K3 awaitreview; noK4+,V2–V5orotherworkauthorized. End “本K阶段结束，未执行下一阶段。”

## Latest: K1 approved; K2 kernel history modules complete (2026-09-15)

Only K2: cdlno/kcdno/history.py KernelHistoryCache/Writer/Reader, independent tests/kernel_history_reference.py and16targeted tests. Source Wk/RMS makes FP32 sum(K^T rawT),sumK; receiver one Wq/RMS reads stacked per-source summaries, unified tokenwise source softmax, raw fusion and U+gamma(C-U). No Wv/Wo/sqrt/token-softmax/current-kernel-read/mask/cached module state. NewQK biasFalse/Xavier1,norm1,w0,gamma scalar.1. Entire production norm/projection/phi/cache/read/depth/gate region disables autocast/FP32; output U.dtype. Double oracle preserves double, separate cached/explicit QK^T and2gradchecks. Empty exact U, no parameter computation. Shared-old RMSNorm untouched.

Report docs/KCDNO_K2_KERNEL_HISTORY.md; STATUS/evidence docs/kcdno_audit/k2.68/68 tests4.169s(new16,K1config13,oldCDPA/modules39),7frozen0.444s,41K0same-weight oldfixtures replay passed. Double12cases maxgrad2.66e-15; FP32/double24cases1.43e-6 at5e-6/1e-4. CPU BF16 and4GPU AMP FP16/BF16 cases observed actualFP32 math/autocastoff andgradient parity. Extreme CPU inputscales0..1e15/clampedden1.5e-11+eps passed, not arbitrary-range guarantee. Selfreview repaired untested emptyoracle new_ones tuple call and added coverage; no production math fix/FP32 tolerance loosening. Source-owned onecache reused by2independent readers gradientchecked; parameter/storage andcache/input immutability passed.

K2start main222647f alreadydirtyK1,403files snapshot /home/hwz/CDLNO-artifacts/k2-before-98ze33eu/source.Only oldSTATUS/memory updated;401others includingK1config/evidence,oldCDPA/entries/train/data/deps/tests/scripts byte-identical. LocalPython3.13.9torch2.13cu130RTX5090,remote2.11cu128unrun,noinstall. No fullKCDNO/core/wrappers/taskcheckpoint/training/performance yet. K3owns nofirstreader/nolastwriter/L1-offinactiveallocation,7write7Q28read,FFN1-reader-FFN2/preUpT,tuplehistory/activationcheckpoint/timeisolation. Do not implement K3 until authorized; oldV2–V5remain pending.

## Latest: K0 approved; K1 independent configuration complete (2026-09-15)

Only K1 authorized. New cdlno/kcdno config/profiles/options/metadata, no model class or task registration. family=kcdno, standalone L/d/h/M/r/history all/off and FFN/norm/kernel/gate fields; no CDLNO F/P/front/CDPA inheritance. Eight-task kcdno_v1 and transolver_shape_match; explicit argparse SUPPRESS probe > profile > new defaults. Car uses cfd_model, PDE/Air model. Eval reads complete saved resolved config first, compares explicit structure, records runtime differences, never reinitializes or rewrites sidecar; profile/init are provenance. Missing checkpoint family stays legacy. Whole-object/list/state_dict recorded without any new weight loader or resume mechanism.

Report docs/KCDNO_K1_CONFIGURATION.md, independent STATUS and docs/kcdno_audit/k1 evidence.13 new tests +6old config +7frozen =26/26 passed7.201s. K0 same41 fixtures replayed before/after exactly (33old CDLNO CPU,8original Transolver GPU, realPyG old industrial objects/list).136fixture hashes verified,none missing,no recapture.371file source snapshot /home/hwz/CDLNO-artifacts/k1-before-khe93ugp/source; start main222647f clean. Only KCDNO STATUS/memory oldtracked files changed; all oldproduction/entries/data/config/tests/scripts/deps unchanged. Python3.13.9torch2.13+cu130PyG2.3.1; target remote2.11cu128 not run/reinstalled. New config pure imports from threecwd passed; existing package discovery includes subpackage, no editable installation. KCDNO math/model/task/GPU acceptance pending. Future adapters must extend metadata for grid/reference/time/output and strict actual weights. Do not execute K2+ or V2–V5. Preserve all K0/A1 artifacts and completed Darcy ablation launchers.

## Latest interruption: Darcy ablation launchers complete (2026-09-15)

User separately requested scripts in two directories under tran_evaluate/ablation. no_sa and identity each now have train.sh/eval.sh/train_eval.sh, forwarding the selected mode then user arguments to the existing Darcy/sequential helpers. Defaults remain F2/L8/P6, entry CDPA and existing Darcy presets/output timestamp paths. Six syntax checks, ten previews/sixteen actual-parser command checks and two missing-eval-directory rejections passed; no exp import, model/data/training/GPU execution. See tran_evaluate/ablation/README.md and verification.json. Production models/entries/configs/dependencies/old launchers unchanged. Preserve existing K0 audit/fixtures; this interruption does not authorize K1 or advance other stages.

## Latest: experiment output management complete (2026-09-15)

User approved unified output/<dataset>/<timestamp> only. EightCDLNO entries create startup config/log before data; actual parameters/architecture/optimizer/protocol follow real model setup. Train results separate from each unique eval results/index. Model math/presets/data/train/eval computations untouched, old checkpoint formats/strict checks preserved, no V2–V5 resume/viz task integration. Report/commands: docs/CDLNO_EXPERIMENT_OUTPUTS_REPORT.md and CDLNO_EXPERIMENT_OUTPUTS.md. Snapshot312files /home/hwz/CDLNO-artifacts/output-before-kql2cnpe; preserve previous dirty work. Final212tests OK/142.652s,1Air full sampling subcase skipped (no torch_cluster);24mode recording/checkpoint+threecwd+Car exact synthetic epoch weights/RNG and Air original weighted kernel/record fragment passed.86existing fileshash same,12complete entry/train AST observation projection equal. No actual data training/remote target/full sampling metric acceptance. Current timestamp path supersedes older mode-suffix/fold0/full path defaults; explicit old paths remain loadable. Stop this scope, no next phase.

Last reviewed: 2026-09-15

## Current: A3 approved; supplemental A4 complete awaiting review

[A4 report](../docs/CDLNO_FRONT_ABLATION_A4.md), [eight-task three-mode commands](../docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md), [raw results/summary](../docs/front_ablation_audit/a4/). Only performance models.py/costs.py/CLI, test_performance and docs changed. Actual hooks now count SA/FFN sublayers, including0 frontSA for no_sa/identity and0 frontFFN for identity; matched LRSA locks full at config and every block. Full default/old commands remain. DefaultF2L8P6: SA8/6/6,frontFFN4/4/0,rearSA/GEGLU6 each,down/up/structuredConv3 each; CDPAentry2sources/1historySDPA,every27/6unchanged. Source batching changes API calls, not MACs/source semantics.

203/203 full tests passed126.687s;13performance tests14.496s;33CPU cost/chunk rows,10finiteGPU measurements,24actual preset parameter counts,24sequential command previews/48parser checks allpassed. Five original performance models retain real pre-A4 initial-weight hashes,parameters,MACs andoldcounts. GPUElasticityN972B1 andscaledAirfoil17x23B2,commonFP32/mathSDPA/TF32off/compileoff,warmup5/iterations20,synchronizedmedian/p90/peakallocated,initializedAdamW andsyntheticMSE. Notrealepoch or taskloss/temporalrollout timing; params/MAC don't predict speed. RearSA remains; fixedCDPA comparison cannot establishCDPA substitutes deletedbranches. RemotePython3.10torch2.11cu128/newmodeAMP/Flash/compile/fullGPUmatrix andrealdata/convergence/accuracy/speed unverified. AiractualMSE_weighted vsA3function-defaultMSE coverage boundary remains explicit.

128existingproduction/task/script/toolfiles match284file snapshot `/home/hwz/CDLNO-artifacts/front-a4-before-ftvcu781/source`;40shell syntaxpassed. Cumulativepre-A1 AST/hash: CDPAunchanged,allmodule definitions exceptfrontunchanged,coreforward/history andbridge/rear/readout constructionidentical,wrappers onlymode,eightpresets onlymodefull,originalexp/main/train/data/loss/metrics/scripts/dependenciesunchanged. PreserveA1-A3/userchanges andV1. V1complete sharedarchives/viz remains separate; V2–V5 unexecuted,tasklaunchers stillno newresume/fullarchive/periodicviz. No commit/push,downloads,dependencyreinstall,realtraining or nextphase. End “本补充阶段结束，未执行下一阶段”. Allbelowstatuseshistorical.

## Current: visualization/resume plan approved, V1 complete awaiting review

User approved implementation while preserving model structure. Following the standing one-stage rule, only V1 shared foundations ran. [Report](../docs/CDLNO_VISUALIZATION_RESUME_V1.md), [plan](../docs/CDLNO_VISUALIZATION_RESUME_PLAN.md), [evidence](../docs/viz_resume_audit/v1/). V2–V5 task integrations were not run; **existing launchers have no new --resume/periodic-viz/full-training-archive integration yet**. A4/performance work remains unexecuted. New files only: cdlno/training_state.py, training_observer.py, visualization.py plus tests/docs. All128 preexisting production/task/script/tool files unchanged against270-file snapshot `/home/hwz/CDLNO-artifacts/viz-v1-before-04ww1akt/source`; keep prior user/A1/A2/A3 modifications and fixtures.

Complete epoch archive stores strict model/Adam(W)/OneCycle or Cosine state, actual update/scheduler counters, normalizers/protocol/ordered fingerprint, Python/NumPy/CPU/CUDA/named generator RNG, history/extra and execution/code provenance. Paired pure weights resolve a verified full companion; bare legacy state_dict/whole-model objects cannot promise optimizer continuation. Manifest commits both file hashes before latest/final. Failure after manifest but before latest now correctly blocks rollback to older committed state; same-state publication retry allowed. No overwriting architecture sidecar, no strict=False, no object reinitialization. Restore RNG last. Drawing uses owned outputs, explicit H/W or point scatter, shared GT/pred/fixed error scale, PNG/PDF/NPZ/JSON; callback isolation restores RNG/buffers/per-module flags and preserves params/grads. Viz failures are logged and checkpoint still saves; archive failures propagate.

201/201 regression passed122.573s, 0fail/error/skip.17 new tests(14archive+3viz).Six CPU fresh-process continuation comparisons full/no_sa/identity×OneCycle/Cosine are bitwise equal across weights/AdamW/scheduler/loss/lr/sampling/DataLoader shuffle/RNG. One localGPU identity/OneCycle FP32/mathSDPA/TF32off case passed atol1e-6/rtol1e-5. New continuation is smallB2/5×7/d8/h2/M4/L3F1/everyblock core + explicit synthetic MSE, not NS or eight task original-loss resume. Legacy original-loss/PyG suite passed but doesn't establish new task integration. Optional GradScaler state path exists; AMP continuation untested. RemotePython3.10/torch2.11/cu128, actual data/training/accuracy/epoch speed/filesystem power-loss/persistent-worker/distributed recovery unverified. No install/download/realtraining/commit/push. Ending “本阶段结束，未执行下一阶段”. Earlier plan-only entries below are historical; source corrections about actual Air weighted loss and Plasticity fx/time remain applicable.

## Latest request: visualization/resume proposal only, awaiting approval

User asks to read Transolver/++/3/LRSA/LinearNO visualization material and propose periodic field figures/checkpoints before any implementation. [Plan](../docs/CDLNO_VISUALIZATION_RESUME_PLAN.md), [sources](../docs/viz_resume_audit/paper_sources.json). Completed only source/figure review and planning docs. Proposed all8 viz every50 completed epochs, NS/Car/AirfRANS/Airfoil full resume state+weights every100+final, other4 final only. Preserve PDE500/Car200/Air398 and actual CLI epoch overrides. Suggested final figure,2 fixed cases,original final eval formats,complete optimizer/scheduler/RNG/normalizer/ordered data provenance/member state,explicit epoch-boundary resume. New CLI/components/callbacks unimplemented; V1–V5 need individual explicit approval. A4/performance tooling unexecuted. No real data/model test/training/dependency or production edits this planning turn.

Current source corrects historical A3 interpretation: AirfRANS main.py uses criterion='MSE_weighted' (loss_vol+args.weight*loss_surf), despite train() function default MSE. A3 all-mode synthetic MSE function tests are real passes but not proof of the actual entry's all-mode weighted training chain; old separate weighted tests exist. Do not repeat the wrong default assertion; future resume acceptance must execute actual weighted path. Plasticity input field fx is normalized by x_normalizer, original meshgrid pos isn't; torch.randperm collate time ordering must remain. Car os.listdir sample order/get_shape Python randomness and Air random sampling/20 validation samples affect exact epoch restart. Existing weights-only/whole-object checkpoints lack resume states and cannot be retroactively claimed exact resume. No production bug fix authorized by these findings; current scope is plan approval.

## Current: A2 approved; A3 verification complete, pending review

The user's 2026-09-15 A3 is synthetic task/train/eval/checkpoint verification, superseding the older suggested performance-tool A3 allocation. A4 and performance adaptation remain unexecuted. Read [A3 report](../docs/CDLNO_FRONT_ABLATION_A3.md) and [evidence](../docs/front_ablation_audit/a3/). Only tests/docs changed; snapshot253 files `/home/hwz/CDLNO-artifacts/front-a3-before-l9s4pe2c/source`, all128 audited production/task/script/tool files byte-identical. Preserve all prior accepted A1/A2/user changes and genuine pre-A1 artifacts.

Full suite184/184 passed75.836s, zero fail/error/skip. New43 cases:24 CPU task×mode,6 Airfoil/Pipe5×7,12 finite GPU Darcy/NS/Plasticity/Car,1 no-exp-import guard. All42 computational cases run original loss, real backward/optimizer update, eval and strict same-mode checkpoint via current loaders with chunk0→1 and read-before-compare/unchanged sidecar; maximum output error6.146728992462158e-8 at inherited atol1e-5/rtol3e-4. Result-only real_N metadata clarified for temporal GPU (still actual4096/3131), final targeted43/43 passed13.469s. Standard tests extract actual optimizer/scheduler/batch AST omitting only CUDA transfer; no exp/main import or fake dataset. Industrial tests call original safe train/test with realPyG2.3.1, default Car velocity+.5surface-pressure/AirMSE, actual gradients compared to independent mask formula, variableN/multigraph/error/y-leak/mutation checks; Air sampled7-of29 ptr retained. Industrial large32186/32000 is separate layout forward, not full training.

NS B2 actual4096:10 training forwards/truth windows,1 backward/optimizer/scheduler then10 prediction-fed eval calls. Plasticity B2/N3131/label[B,N,4,20]:20 independent updates/1 scheduler, time input gradients and output response;20 eval calls. Per-call bridge/T rebuilt and live front T used in CDPA; no time cache. Original A1/core/CDPA independent reference/mode/off-entry-every/history/chunk36 cases reused and passed. Three originalcwd newprocess allmode loads and true pre-A1 full12 core/front+2Car+2Air fixture outputs/gradients/initialization/objects passed again at exact zero tolerance.

LocalPython3.13.9/torch2.13+cu130/RTX5090 Laptop, new GPUFP32 mathSDPA/TF32off only; remotePython3.10/torch2.11/cu128, remaining4task newmodeGPU, AMP/otherSDPA backend/fullwidth resources, realdata completeness/sampling/allphysicalmetrics/training/convergence/accuracy/epoch efficiency unverified. Original Car outer logging swap and dragfold0/path limits plus Air oldMAE test condition remain inherited, not newmode regressions; no production fix was needed. No dependency edits/download/commit/push/nextphase. End “本补充阶段结束，未执行下一阶段”. Lower statuses are historical.

## Current: A1 approved; supplemental A2 completed pending review

Latest user explicitly assigned A2 only. All eight existing tasks now support front_latent_mode full/no_sa/identity through actual CLI (hyphen and underscore alias), model JSON, wrapper architecture/core and existing checkpoint paths. Train default remains full; eval reads existing complete sidecar before checking an explicit mode or recovering omitted mode. Other custom model/task options still follow the old matching protocol. No default-filling of other missing architecture fields: only complete identified pre-A1 full18 fields may omit mode. Industrial whole objects/lists retain stable class import paths/local trusted torch.load boundaries; complete wrapper/core config and per-front actual mode checked before strict state verification. A1 genuine old missing-attribute objects remain full; no new init/rebuild/migration/resume.

See [A2 report](../docs/CDLNO_FRONT_ABLATION_A2.md), [commands](../docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md), [audit](../docs/front_ablation_audit/a2/). Eight model presets add only mode=full, preserving PipeM32/othersM64 and all training fields; Airparams.yaml unchanged. Root4 shell helpers add implicit ablation directory suffix; oldfull paths/user explicitrun precedence remain. Subproject16 scripts/train_eval unchanged, already forward mode. Actual exp/main/train/data/loss/time/metrics and config/modules/core/CDPA mathematics frozen. Performance tooling still full-only until a separately assigned phase.

Final141/141 (62.591s, no failures/errors/skips), including8 A2 tests/24 task-mode synthetic backward+load cases, realPyG industrial original-loss statements, three originalcwd new-process all-mode loads. 48 real-parser root commands and24 sequential dry-run pairs passed. All genuine pre-A1 12 core/front+2Car+2Air full fixtures replay again exactly at original zero tolerances. Initial test harness constants/backend errors and3 old argument-only fake-run-path checks recorded/corrected; actual checkpoint checks not mocked. New partial-config-object rejection included in final suite. Source snapshot A2: `/home/hwz/CDLNO-artifacts/front-a2-before-cmyap6x3`; retain original A1 fixture directory too. LocalPython3.13.9/torch2.13+cu130/PyG; oldfull GPU regressions passed, A2 eight-task newmode GPU matrix/AMP, remote torch2.11/cu128 and real data/training/convergence/accuracy/epoch efficiency not run. No dependency install/commit/push/next phase. A3/A4 unexecuted; older status below is historical. End “本补充阶段结束，未执行下一阶段”.

## Current: supplemental A1 resumed and completed, pending review

The user explicitly resumed A1 after the separate launcher task. A0 is approved; A1 implementation and targeted core acceptance are complete; **A2–A4 have not been executed**. Read [the A1 report](../docs/CDLNO_FRONT_ABLATION_A1.md) and [evidence](../docs/front_ablation_audit/a1/) first. Only config/modules/core and tests/docs changed. `front_latent_mode` is an architecture field, default full; no_sa retains both independent latent FFNs/norms and removes SA/norm; identity directly returns T=S and registers no latent processor sublayers/norms. Down/Up/point update, bridge/rear/readout and CDPA/history are unchanged. F0 registers no front parameters and all three modes are exactly equal with copied weights. No new task CLI/wrapper/JSON/YAML/launcher/performance integration; existing task launchers still construct full.

Preserve genuine pre-edit artifacts `/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0` (old source, start and resumed-start hashes, 12 core/front +2 Car +2 Air fixtures/objects/list). All replay exactly in current code, old key/order/init and all outputs/gradients, original atol=rtol=0. Car capture's initial models import failure was fixed by explicit old project PYTHONPATH **before production edits**; failure record retained. Old 18-slot frozen config pickle is explicitly restored; missing known old JSON mode means full; only the known full history rule aliases to selected-processor/pre-Up. New config pickle is named/versioned; old front objects lacking the attribute default to full. Wrong modes/weights remain strict, existing sidecar bytes unchanged.

13 new targeted tests cover independent formulas, removed branches/parameters, identity live T, copied-weight zero-output references, 36 small core combinations, F0 and extended L, history isolation and chunks. Final suite133/133 passed (53.780s, zero failures/errors/skips); initial2 inherited frozen-inventory errors were fixed only in tests using original git-tree inventory, with the initial log retained. Three limited new-mode CPU/GPU FP32 parity cases passed on local Python3.13.9/torch2.13+cu130/RTX5090 Laptop. No dependency changes, remote new-mode run, new-mode mixed-precision acceptance, real data/training/accuracy/epoch performance or next phase. Frozen hashes protect all three task trees/wrappers/CDPA/checkpoint/dependencies/shell scripts including the sequential launcher. New-mode performance tooling remains future work. Self-review complete; no known remaining A1 architecture defect. Older pause/A0 entries below are historical and superseded; stop for user review, ending “本补充阶段结束，未执行下一阶段”.

## Latest steering: A1 paused; sequential task launcher delivered

A0 was approved and A1 assigned, then the user explicitly paused A1 to request eight-task train/eval commands and a combined script. Completed this separate launcher task only: [report](../docs/CDLNO_SEQUENTIAL_LAUNCH.md), [usage](../tran_evaluate/README.md). `train_eval.sh TASK` calls existing TASK.sh train then eval only on success, propagates failure status, uses one shared/generated tag, and splits optional --train-args/--eval-args. 32 actual AST-parser commands (8 default +8 override pairs) and shell success/failure/tag/quoting checks passed. No model/data/train/eval/dependency or pre-existing task-script edits, no real training/evaluation. Car fixed drag path/param0/fold0 and Air train Dataset vs eval parent limitations remain.

A1 remains paused/incomplete, no production changes or front_latent_mode CLI/config/module implementation yet. Pre-edit copy: `/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0/source`; manifest in parent/start.json. Capture script: docs/front_ablation_audit/a1/capture_before.py. 12 core/front and2 AirfRANS genuine old object/list fixtures passed exact repeat CPUFP32/mathSDPA eval/dropout0 outputs and gradients (atol/rtol0) on local Python3.13.9/torch2.13+cu130; not remote acceptance. Car capture failed before model import because the script path/PYTHONPATH omitted cwd/models; fix the independent process import path and retain the failed attempt before any implementation when A1 resumes. Do not overwrite completed pre-change fixtures or label them full A1 acceptance. Old status below is historical; await user steering before resuming model work.

## Supplemental A0: source audit complete, no implementation authorized

The user explicitly assigned A0 only. Read [CDLNO_FRONT_ABLATION.md](../docs/CDLNO_FRONT_ABLATION.md) and [source audit evidence](../docs/front_ablation_audit/static-results.json). A0 is complete pending review; A1–A4 remain unassigned/unimplemented. The older pending-supplement wording below is historical. Main/HEAD is `769fa333742f73c132868cf560bce5ec21529362`, with existing remote launcher/document changes protected by the 203-file start manifest.

Current LRSAFrontBlock in cdlno/modules.py exactly follows S=Down(Hn), A=S+FFN1(N1(S)), B=A+SA(Nsa(A)), T=B+FFN2(N2(B)), then original Up/point residual/point FFN. T is live and before up_latent_norm. Core consumes it only after bridge at rear CDPA positions; no front CDPA, neighboring-front CDPA or rear Kimi AttnRes has landed. No front_latent_mode attribute/config/CLI exists yet. Reference LRSA 47b03f8c8c8da30bbcc0737b008dc4548f9cb98e remains clean: disable_interleaved_blocks skips both SA and FFN2, so it cannot implement no_sa. All eight original CDLNO task interfaces exist; all new ablation entry paths remain pending.

Future compatibility must handle old missing JSON mode as full only for known versions; explicitly conflicting modes must fail. Config is an 18-field frozen/slotted dataclass: inserting a field misaligns old pickle state, and appending alone can leave a slot unset. Preserve old field order and implement narrow legacy state restoration. Old whole model loads bypass __init__, so missing block mode needs a validated full fallback without resetting weights. Car returns the loaded models.CDLNO.Model object; Air returns cdlno.airfrans.AirfRANSModel members/list after strict validation. Keep these paths/protocols, sidecar-first checks, and strict=True. The old FFN2-specific history_rule is inaccurate for future identity; A0 proposes a narrowly normalized neutral processor-before-up rule, not globally ignoring versions or adding a second user switch.

Before the first A1 production edit, capture genuine old-source configs/sidecar bytes/state_dict/nonpersistent buffers/inputs/outputs/selected gradients and actual legacy Car/Air member/list pickles in independent baseline processes. A0 preserved hashes only, no numeric fixtures; generating new objects then deleting attributes after implementation does not count as old-pickle evidence. See the report for coverage/manifest/precision requirements. Full module names, creation/RNG order and initialization must regress against that baseline. Future cost hooks must count actual SA (default full8/no_sa6/identity6), keeping down/up/structured Conv3 and CDPA source counts unchanged. matched LRSA remains full.

A0 modified only audit/status/memory documentation: of 203 original files, 200 byte-identical and only AGENTS/STATUS/current-state increased. The source-only audit passed13/13; no model forward/backward, PyG, GPU, remote environment, dataset, or training checks ran. Original65-file comparison found53 identical plus the same12 previously integrated entry/config differences; no new production changes. Current full has no material formula conflict requiring user judgment. Existing Car evaluation path/fold restriction, original logging issue, remote new-model/real-data/accuracy/epoch-time limitations remain. End A stages with “本补充阶段结束，未执行下一阶段” and await explicit next-phase scope.

## Remote launch supplement

The user authorized path configuration and eight-task train/eval scripts, plus a read-only remote format inspector. Complete; read [remote launch report](../docs/CDLNO_REMOTE_LAUNCHERS.md) and [usage](../tran_evaluate/README.md). New HEAD at task start is `769fa333742f73c132868cf560bce5ec21529362` (user commit, main); only path.sh was untracked and empty. Current source/entry/model/config/test/old-launcher bytes are protected. No automatic commit/push occurred.

Remote repo: `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w`; data: `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data`. path.sh derives checkout root from its own location, defaults FNO=data/fno. Elasticity must receive FNO (entry appends elasticity/Meshes), NS FNO (entry appends full dataset dir/MAT), Plasticity the MAT itself; AirfRANS train Dataset vs eval parent. Industrial candidate directories are unverified; inspect_data.sh reports actual headers/manifest/missing raw+cache files and bounded candidate search without torch/task loaders or data mutation. Car fixed drag path/param0 restriction persists; new data_dir alone cannot fix metrics.

Remote compatibility follow-up: the older remote `path.sh` shown by the user exports only `REPO/DATA/FNO_DIR/CAR_DATA_DIR/CAR_SAVE_DIR/AIRFRANS_DATASET/CONDA_ENV`. `inspect_data.sh` and `_common.sh` now fall back to `python`, derive all standard-task paths plus `CDLNO_RUNS_ROOT/CDLNO_RUN_TAG`, and keep the current checkout anchored to the script location. A temporary old-path environment passed `inspect_data.sh --no-search`, all 8 train dry-runs, all 8 eval dry-runs, and shell syntax checks. The local data root is not mounted, so no real remote format/training acceptance was added.

16 default train/eval commands plus8 override cases passed actual AST-only original parser checks in fresh processes, with preset comparisons, correct paths and8 isolated run directories. Missing/empty-dir and3 in-memory NPY header versions passed. No new model/GPU/PyG/real-data tests were run; existing120-model-test result remains historical. Full remote file contents/labels/mesh/normalization, convergence and runtime remain unknown.

Pending architecture supplement: user defined front_latent_mode={full,no_sa,identity}, defaultfull, with exact whole latent residual/norm sublayers skipped and T=S for identity. No named supplemental implementation stage was authorized yet; this remote script task does not authorize it. Do not add front CDPA, neighboring-block CDPA, Kimi rear AttnRes or any excluded structure. Future named A-stage must separately audit/implement/verify and stop. Current scripts use the existing full LRSA path and no unimplemented flags.

## Latest launch preparation

User requested a final logic review and Car/NS training/evaluation commands in `tran_evaluate/`; preparation is complete, actual training remains unauthorized. Read [the launch review](../docs/CDLNO_TRAINING_LAUNCH_REVIEW.md) and [launcher README](../tran_evaluate/README.md). Final120 regressions passed (52.764s; no failures/errors/skips); four shell dry-runs parsed through actual AST-only entry parsers in original working directories passed. Paper v2 pp15–17/B.1–B.3/Table8 reread; extraction in docs/training_launch_audit/paper-excerpts.json.

Keep confirmed CDLNO d256/h8/L8/F2/M64/FFN ratios2 for Car and NS. Transolver M32 and NS ratio1 are model differences, not overwritten defaults. Corrected new NS JSON max_grad_norm0.1 to null and removed clip from both CDLNO NS scripts: original Transolver_NS.sh omits clipping and exp_ns.py defaults None. The current requirement to preserve original training authorizes this default correction; no model architecture or frozen loop was changed. Added one regression testing original clip branch and explicit override. Previous reports overstated matching training defaults based on frozen ASTs; loop equality alone did not verify effective clipping.

Car original cal_coefficient hardcodes /data/PDE_data/mlcfd_data/training_data/param0, while evaluation passes basename only. New full evaluation script checks fold0 plus same canonical raw root and existing artifacts; no automatic link/data changes. Other folds can train but original full drag evaluation is not generally portable across folds/roots. Car train/test return pressure then velocity while outer main assigns opposite names; logged train_loss/val_loss aggregates are misleading, but internal backward uses correct velocity+0.5pressure. Frozen train/metric files unchanged. All71 original tracked files unchanged from this task's start, preserving existing integration changes;9 of173 snapshotted files changed only as documented. Real data IO/VTK/full metrics, remote new CDLNO execution, convergence/accuracy/epoch time remain unverified. No install, data download, training, commit/push, or new stage.

## Phase authorization

- Latest user instruction establishes **one explicitly assigned phase per turn**. Complete that phase and its checks, then stop for review and the user's next explicit instruction. This replaces v1.2 §11's all-at-once execution model.
- **Phases 0–9 are approved. Phase10 final audit and delivery are complete, awaiting review. All eight task interfaces/configurations remain integrated.** Read [the final report](../docs/CDLNO_IMPLEMENTATION_REPORT.md) and [requirements matrix](../docs/CDLNO_REQUIREMENTS_MATRIX.md), then [the phase9 report](../docs/CDLNO_PHASE9_PERFORMANCE.md) and [performance usage](../docs/CDLNO_PERFORMANCE_TOOLS.md), then [the phase8 report](../docs/CDLNO_PHASE8_AIRFRANS.md) and [eight-task inventory](../docs/CDLNO_TASK_LAUNCHERS.md) first. No real training/sampling evaluation or further work is authorized.
- Every assigned phase must update that status file and report A–F (scope; files/reasons/diff; formulas/shapes/code; actual commands/environment/results; frozen-region evidence; issues and 3–5 review points), ending with “本阶段结束，未执行下一阶段”.
- No automatic commit/push, PR, training, reset, or unrelated refactor. Material architecture/protocol conflicts must be reported with sources and a recommendation.

## Authoritative design document

- Authority order: current/subsequent explicit user decisions → later confirmed conversation decisions → [v1.2](../PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md) → v1.1 historical supplement. Theory attachments explain/check the design without authorizing new structures or losses.
- [比较模型架构.pdf](../PLAN_CDLNO/比较模型架构.pdf), the exported ChatGPT conversation, has now been read sequentially in full: **151/151 pages**. The [page-indexed review](2026-09-13-exported-conversation-review.md) reconciles its design evolution with v1.2.
- Historical proposals and abandoned alternatives do not override final decisions. Quoted requests to implement and the full plan do not authorize work beyond an explicitly assigned phase.
- Basic modules are implemented in `cdlno/modules.py`, standalone CDPA in `cdlno/cdpa.py`, and the data-free core in `cdlno/core.py`. Six standard task wrappers and both industrial adapters are connected; later real-data validation remains pending. Original Transolver models and data/training protocols are preserved.

## Repository baseline

- Base repository: Transolver, remote `https://github.com/thuml/Transolver.git`.
- Local baseline branch: `main`, commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`.
- Existing root `AGENTS.md` documents the original Transolver architecture and three experiment families.
- No real datasets were downloaded or used during this review.
- Phase 1 added the shared `cdlno` package, `pyproject.toml`, configuration/sidecar protocol and preflight. Its import checks used PYTHONPATH; editable installation itself was not executed.
- Phase 2 added individual modules and two test files. Phase 3 added standalone fusion, an explicit reference and its tests. Phase 4 added the data-free core and core tests. This was the phase4 boundary; phase5 now connects four static task adapters and limited existing-entry branches. No new training framework exists; phase9 later added isolated synthetic performance tools. Configuration imports stay independent of torch; `CDLNO` is lazily imported from `cdlno.core`.

## Model and mechanism names

- The latest user message explicitly confirms **CDLNO — Cross-Depth Latent Neural Operator / 跨深度潜空间神经算子**, shared package `cdlno`, and core class `CDLNO`. This supersedes the previous status that only the PDF recommendation was available.
- **CDPA — Cross-Depth Physics Attention / 跨深度物理注意力** is the user-confirmed mechanism name (pp.143–144), formerly CDPA-Cross. CDPA-Slice remains excluded.
- Map provisional `cdpa_operator` → `cdlno`, `CDPAOperator` → `CDLNO`, and whole-model registration `CDPA` → `CDLNO` when those files are implemented. Wrappers may export `Model`; preserve CDPA mechanism identifiers and settings. The data-free `CDLNO` core exists in `cdlno.core`; all eight task adapters/registration are connected, with real-data validation pending.

## Reference snapshots reviewed

- LRSA-Operator: `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`; phase-2 same-configuration tests read `/home/hwz/LRSA-Operator`.
- IPOT: `18c177846267505ee9503445a146dfd7dee34c41`; historical /tmp checkout unavailable, re-read fixed GitHub raw files in phase10, URLs/SHA persist in docs/final_audit/ipot-source.json.
- Transolver paper: arXiv `2402.02366`; the base implementation is already documented in root `AGENTS.md`.

## Planned CDLNO architecture

The v1.2 architecture, subject to the authority order above, is:

- Default total depth `L=8`, front complete LRSA-style blocks `F=2`, persistent latent blocks `P=L-F=6`.
- Configurable `F` in `0..L-1`; L may grow; all front/bridge/rear latent counts use one scalar M per run.
- `cdpa_mode`: `entry` (main result), `off`, or `every_block` (ablation).
- Front block: point pre-norm -> learned-query point-to-latent down attention -> latent FFN1 -> latent self-attention -> latent FFN2; save the full latent after FFN2; latent-to-point up attention; point FFN/ConvFFN residual.
- Bridge: learned query residual plus one cross-attention from final front point features; no unused bridge FFN.
- Rear block: pre-LN latent self-attention plus GEGLU FFN, with independently instantiated blocks.
- Final decoder: LRSA-style point-feature-conditioned latent-to-point up attention using `H_F`, followed by point residual FFN/ConvFFN and output projection.
- CDPA Cross aligns each historical latent source independently to the current latent, then uses token-wise depth softmax over raw candidates including identity `R_0=Z`. Depth scorer is zero-initialized, so the initial fusion is the uniform mean of identity and histories.
- Kimi K3 recheck: AttnRes directly supplies learned pseudo-query depth scoring/aggregation (Full AttnRes Eq. 8–9; Block AttnRes Eq. 10). CDLNO deliberately adds separate `Z0→T_s` M-to-M Cross alignment before source softmax; this is an adaptation, not a literal AttnRes copy. Final decoder follows LRSA up direction: `H_F` supplies query and raw point residual, while final latent supplies only K/V.
- Source chunking (`0`, `1`, or positive chunk size) must be mathematically and gradient equivalent; do not concatenate all source tokens into one softmax.
- Different CDPA locations use independent parameters; only sources within the same location share projections/norms. Earlier cross-location K/V caching suggestions are superseded. Keep raw history with gradients, never projected K/V or learned-LN outputs from another location.
- `every_block` rear block j uses identity `Z_(j-1)` and history `[T_1,...,T_F,Z_0,...,Z_(j-2)]`; raw bridge `Z_0` is preserved for later blocks, current is never duplicated, and history never persists across forward calls.
- Default 2+6 has three down/bridge operations, three up/readout operations, eight latent SA operations in full (six in no_sa/identity), and three structured ConvFFNs. Entry aligns two historical sources; every_block aligns 27 across six locations. Source batching changes calls and temporary memory, not these logical counts or MACs.
- Use PyTorch SDPA; no custom CUDA/Triton kernels, xFormers, Lightning, Hydra, or new training framework.

## Frozen scope

The planned implementation must preserve existing data loading, fields, point order, splits, sampling, normalizers, target definitions, losses, metrics, time loops, optimizer and scheduler semantics. Only new model modules, model selection/argument forwarding, isolated output paths, and strict new-checkpoint sidecars are allowed exceptions listed by the plan.

Explicitly out of scope for this version: sparse Darcy, latent-count schedules, long-horizon NS 10->20/40, CDPA-Slice, Gram/correction layers, PDE residual losses, latent convolutions, cross-layer parameter tying, new geometry encoders, joint task training, and automatic hyperparameter sweeps.

The export explicitly abandons Gram correction on p.82 and corrects point-memory retrieval to latent-history reuse on pp.103–104. Preserve the final LRSA post-up ConvFFN and feature-conditioned readout, not the earlier pre-compression convolution or coordinate-only decoder proposals. The future sparse Darcy sketch is recorded on pp.143–144 (no-convolution front, coordinate-query readout, CDPA retained), but remains unimplemented and excluded.

Actual training protocols override approximate early tables: NS uses teacher forcing for training and autoregression for ten evaluation steps; Plasticity makes time-conditioned calls with an optimizer update per time query; Pipe batch size is 8 and AirfRANS epochs are 398. Preserve the existing CUDA 12.8 / torch 2.11 environment; no installation or downgrade was performed.

User-reported remote evidence (2026-09-13): a Python 3.10 environment installed the PyTorch 2.11/cu128 stack plus PyG/pyg-lib using the cu128 wheel source, and the original `Car-Design-ShapeNetCar` training program completed successfully. This is the current compatibility anchor. Python 3.11 remains a reasonable but unverified target. The command sequence pins `torch_geometric==2.4.0` and then runs `pip install -U torch_geometric`; the final PyG version must be recorded from the remote environment rather than inferred from the first command.
The pasted example contains a cu121 index URL with a note to select cu121/cu124/cu126/cu128 according to the driver, so the exact final CUDA runtime still needs to be identified from `torch.version.cuda` if a precise environment record is required.

## Theory interpretation

PDF pp.147–150 and v1.2 §14 support conditional historical-information reuse. The final decoder accesses local `H_F` plus latent information, so the model is not a pure final-latent bottleneck. A frozen front/bridge can hide a target-relevant direction from local `H_F` plus `Z_0`; CDPA can expose it if history retains it, Cross preserves it, fusion does not cancel it, and downstream readout uses it.

Do not claim full-model Jacobian rank ≤M, all PDEs are low rank, lossless fusion, guaranteed accuracy/latency gain, or strict containment of a fully retrained off model. Zero scorer is a uniform mean, not identity. The export summarizes mathematical arguments, but does not contain the separate full-proof manuscripts or verification artifacts it links to.

## Validation status

Historical review completed:

- Read the full v1.2 plan and its headings/requirements.
- Cloned and inspected the LRSA and IPOT source snapshots above.
- Read all 151 PDF pages and visually checked selected key formula pages; reconciled final design, numerical details, scope, performance accounting, and naming with v1.2.
- Updated only `AGENTS.md` and project memory for this supplemental reading; protected-file hashes were checked for unintended changes.
- Confirmed no project code was edited in this review.
- Subsequent standing-constraints turn: recorded phase gating, exact design precedence, confirmed names, and the A–F report requirement; created `docs/CDLNO_IMPLEMENTATION_STATUS.md`. No implementation phase was started.

Phase-2 implementation and actual tests (historical phase record):

- `RMSNorm`, plain FFN, GEGLU, dense ConvFFN, full LRSA front, query-residual IPOT bridge, independent pre-LN persistent block and feature readout exist. `T` is returned live after FFN2, before up. No CDPA or full assembly.
- Front/readout branch norms and per-head Q/K norms are RMSNorm; bridge/rear and explicit output `LN_out` are LayerNorm. All eps=1e-6. ConvFFN has internal affine LayerNorm. Ordinary linear biases zero; Conv2d keeps native Kaiming weight/bias initialization. Special query initialization is never recursively overwritten.
- 18 module tests passed, plus 2 optional LRSA same-configuration tests. In tested FP64 examples, LRSA point and 5x7 structured front outputs, T and all corresponding gradients had max absolute error 0. This is not a reproduction of a paper's task training preset.
- Actual local GPU small-module FP32/FP16/BF16 autocast checks passed (torch2.13+cu130, RTX5090 Laptop). CPU BF16 convolution backward initially failed in oneDNN; test-only MKLDNN disable provides a passing native math path. CPU/GPU FP32 parity initially exceeded 2e-6 with TF32 enabled; disabling TF32 within tests passed at unchanged tolerance, restoring original backend settings afterward.
- No remote torch2.11/cu128 run or dependency installation occurred. Local results are individual module validation, not remote/model training acceptance.

Phase-3 implementation and actual tests:

- `CDPA(dim, heads, source_chunk_size=0)` accepts `Z/history` and optional per-call chunk override / source weights. Q once; each historical source has key length M; group layouts are `[B,k,M,d]` to `[B*k,h,M,d_h]`. Shared LN_q/LN_kv and Q/K/V/O within each instance, independent parameters across instances. O+b is completed before the candidate is admitted; no Cross residual. R0 is raw Z.
- Depth scorer w strictly zero, RMS scale1/no bias, eps1e-6; FP32 `[B,M,S+1]` weights are computed per current token and multiply RAW values. No gate, identity bias or outer residual. The whole depth RMS/scoring/softmax/accumulation subgraph disables autocast, with actual dispatch dtype/state verified. Nonempty FP64 Cross still uses FP32 depth by design; empty history returns the exact original Z without projection or SDPA.
- Histories can have differing floating dtypes; Cross casts computation inputs to Z.dtype without detaching or replacing the original history tensors. Mixed BF16/FP32 history tests cover reference/all-grad parity across chunks and CPU BF16 autocast. This anticipates front BF16 / bridge FP32 residual interaction without assembling a model.
- `tests/cdpa_reference.py` was written and checked before production fusion. It explicitly implements LayerNorm/QK/token softmax/AV/O/depth RMS/source softmax/RAW sum using torch primitives and a parameter dictionary; it never imports the production module, reuses its helpers, calls forward or SDPA.
- Final command `python -B -m unittest discover -s tests -p test_cdpa.py -v`: 21/21 passed, 0 failures/errors/skips, 3.141s. CPU48-case matrix: FP32/FP64 × S1/2/5 × w0/nonzero × chunk0/1/2/>S; all input and parameter gradients compared. Largest absolute gradient error 2.17e-5 passes the combined atol3e-6/rtol3e-5 rule, not an absolute-only3e-6 bound. All permutation/batch/layout/zero-w/live-gradient checks passed.
- Actual local execution: Python3.13.9, torch2.13.0+cu130, CUDA13.0, NVIDIA GeForce RTX5090 Laptop GPU. GPU FP32 MATH/automatic SDPA and direct FP16/BF16 plus autocast passed; TF32 disabled only within GPU tests and restored. CPU extreme input scales0/1e-12/1e-6/1/1e4/1e8, GPU low-precision0/1e-4/1/1e4 passed finite output/all-grad checks. These are bounded synthetic tests, not arbitrary-magnitude or speed guarantees. No profiler kernel attribution or remote torch2.11/cu128 acceptance.
- Same standalone state_dict strictly loads across chunk values. Existing sidecar protocol was tested without modification: runtime chunk/device/dtype/AMP changes accepted; mode/F/L/M changes rejected; existing sidecar bytes preserved. It is not yet connected to model/task checkpoint paths.
- §8.3 phase-3 boundaries resolved by phase 4: item1 F0/entry omits unused CDPA registration; item9 now has core front/bridge/rear/decoder gradient coverage; item11 has core per-layer source ordering, raw Z0 retention, no current duplication and two-forward isolation; item14 has core state_dict cross-chunk coverage but actual model/entry checkpoint integration remains pending. Single-module calls and the assembled core are stateless across forwards.
- Frozen manifest `/tmp/cdlno-phase3-frozen-hashes.json` covers86 existing baseline/plan/phase1/phase2 files, all unchanged. Manifest SHA256: `7511a03d5eb63b7c99306c40096b72435b0f93105b1fb068b6625e36ee5102a9`. New Python files pass3.10 syntax parsing on the local interpreter; root config/sidecar imports still avoid loading torch. No data/entry/dependency changes or training.

Phase-4 implementation and actual tests:

- `cdlno.core.CDLNO` accepts lifted `H_0[B,N,d]` and returns only `[B,N,C_out]`. It creates F complete `LRSAFrontBlock`s, one `IPOTBridge`, P=L−F independent `PersistentLatentBlock`s and one `LRSAFeatureReadout`; no input stem or position/time encoding is introduced.
- `off` registers no CDPA and collects no history. `entry` registers one CDPA only when F>0 and consumes `(T_1,…,T_F)` before rear block 0. `every_block` registers every rear location with nonempty history; at location j it uses current `Z_(j−1)` and `(T_1,…,T_F,Z_0,…,Z_(j−2))`, appending the previous raw identity only after use. History is rebuilt per forward and never detached or stored on the module.
- Phase-4 configuration adds `model_version='cdlno-core-v1'`, fixed `history_rule`, `latent_ffn_ratio`, and optional structured `grid_shape`; P remains derived only from L−F. Core rejects unsupported norm/dropout/history variants rather than silently changing the architecture. `CDLNO` is lazily exported by `cdlno` while configuration/sidecar imports remain torch-independent.
- Final regression: `CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v` passed 60/60 in 12.912s, no failures/errors/skips. Phase 4 contributes 13 core + 6 configuration/loading tests. Core covers 42 L8/F0…6/mode/point-grid cases, extended L12/F2 and L16/F6, L1/F0, F7/F8 without a cap, exact F0 entry/off copied-weight output/gradient/call equality, actual SDPA and projection counters, all raw-history source object IDs, two-forward graph isolation, direct live intermediate gradients, initialization and parameter/storage independence without deduplication. Configuration tests cover invalid values/inputs/chunks, independent front/rear ratio binding, JSON grids, derived-only P, sidecar byte preservation and runtime changes, new-process checkpoint restoration and imports from all three task working directories using PYTHONPATH (no install).
- Core raw-prefix reference and real cross-chunk weight/output/all-gradient comparisons use atol=1e-5, rtol=3e-4. The raw-prefix oracle calls the independent phase-3 CDPA mathematical reference, not CDPA.forward. Initial test-count assertions and a frozen-slots exception-type expectation were corrected; the final report records these failures and earlier coverage gaps. They did not require changes to the core mathematics or validated phase-2/3 modules.
- Call contract verified by actual hooks/spies: down+bridge=F+1, up+readout=F+1, latent SA=F+P=L for full (P for no_sa/identity), structured point ConvFFN=F+1. Default F2/P6 chunk0: off 0 logical histories / 0 history SDPA / 14 total SDPA; entry 2 / 1 / 15; every 27 / 6 / 20. L8/F0/every has P8, 28 sources / 7 history SDPA. Chunk0 reduces API calls, not source computation or necessarily kernel count.
- Actual local environment: Python3.13.9, torch2.13.0+cu130, RTX5090 Laptop GPU; core point/grid × modes × FP32/FP16 AMP/BF16 AMP gives 18 finite forward/backward cases. This is not remote torch2.11/cu128 compatibility, real training, full-core mixed-precision error-bound, or performance evidence. Standard caller-controlled dtype/autocast remains; no unapproved automatic dtype conversion was added.
- Frozen phase-4 baseline `/tmp/cdlno-phase4-baseline/` contains 98-file hashes; 93 non-target files remain byte-identical. Original task directories, model/entry/data/loss/time/optimizer/scheduler/evaluation/dependency files are unchanged. No remote target run, data, training, install, commit or push.

Not completed:

- Real-data validation and remote performance; all eight interfaces and phase9 finite synthetic performance tools are complete.
- All eight interfaces have synthetic loss/checkpoint checks. AirfRANS actual sampling, graph construction and geometry/metric postprocessing remain unrun.
- Remote target CUDA/PyTorch/PyG environment validation (local GPU synthetic checks have run).
- Any real-data training, convergence, accuracy, or end-to-end speed measurement.

## Missing plan attachments

The v1.2 plan mentions `CDPA_Mathematical_Foundations.md`, `CDPA_Theory_Manuscript.tex/.pdf`, `check_theory_identities.py`, and `theory_identity_checks.json`. The user says mathematical attachments have been provided; a renewed filename search of this workspace still did not locate those separate files. Their actual paths remain to be identified; do not claim they were not provided elsewhere or that they were read locally. PDF pp.147–150 provides a theory summary and historical claims of eight NumPy checks and a compiled nine-page manuscript; it does not supply those separate artifacts or their complete proofs. Treat the available prose/formulas as context, not independently verified attachments or checks run here. The v1.1 environment requirements, constraints, and preflight script are present under `PLAN_CDLNO/CDPA_v1_1/`; that is a separate delivery from the theory files. This location issue does not block recording the standing constraints.


## Follow-up self-review and standing workflow (2026-09-13)

- User explicitly requests self-review of each phase's proposed review points before delivery; report only remaining defects/uncertainties or decisions requiring user judgment. Record passing evidence rather than returning an unchecked checklist. Continue to honor one authorized phase at a time.
- Reviewed all four phase-4 points. Raw Z0 append timing/full raw prefixes, copied-weight F0 entry/off equivalence, independent parameter/storage and single initialization, actual source/SDPA counts, HF query/residual and normal sidecar compatibility/no-overwrite behavior passed source review and 15 targeted CPU tests (4.991s). Log: `/tmp/cdlno-phase4-self-review-tests.log`. No GPU or remote rerun; prior 60-test pass remains historical evidence, not proof that all invalid protocol inputs were covered.
- Confirmed one open defect: `CDLNORuntimeConfig.validate` in `cdlno/config.py:124` checks only `<0`. Bool, positive float, NaN and Inf are accepted by runtime.validate/save_sidecar/load_sidecar/validate_sidecar, although CDLNO/CDPA reject them. Reproduced True/1.5/NaN/Inf using temporary sidecars; bytes were not overwritten. Correctly configured model outputs are unaffected, but invalid runtime metadata can be declared valid and fail later at core construction.
- Proposed fix, not executed: require `type(source_chunk_size) is int` and nonnegative inside the torch-independent runtime config; add meaningful validate/from_dict/save/load/compare regression including bad requested runtime and malformed existing sidecar, preserve same-weight valid chunk changes and byte integrity. No model, architecture, data or training changes needed. Earlier user instruction requires reporting the plan before the user decides whether to apply it; do not treat this review as approval of the fix or phase5.
- Review-only edits: AGENTS, STATUS, phase4 report and this memory. Snapshot `/tmp/cdlno-phase4-self-review-6ac3ba44/hashes.json`: 102 files before documentation edits, remaining98 to remain unchanged. No implementation/test/dependency changes, real data or training.


## Phase5: four static standard tasks completed (2026-09-13)

- User approved phase4 and explicitly restricted current work to Darcy, Elasticity, Airfoil and Pipe. Added `cdlno.standard.StaticStandardModel`, two thin standard-project Model wrappers and canonical registry `--model CDLNO`; selected via per-exp task marker. NS/Plasticity/both industrial tasks are not enabled.
- Darcy matches original structured model's fixed index-grid reference distances, replacing xy then concatenating fx1: stem65 at ref8. Do not substitute coordinate-derived distances. Other three tasks use existing coordinates and actual fx=None placeholder path; Pipe coordinates arrive after original UnitTransformer encode. Static wrappers reject time inputs and register no time_fc; future wrappers must register it only with Time_Input=True. Stem is original two-Linear GELU MLP; no parent recursive initialization.
- `cdlno_entry.py` handles only new flags/default resolution, isolated paths and strict checkpoint. Parse explicit options using argparse with suppressed defaults so a deliberate value equal to old defaults remains honored. Four JSON presets: d128, heads8/8/4/4, M64/64/64/32, L8/F2, entry/chunk0, ratios2; epochs500, batch4/1/4/8, lr.001/decay1e-5/clip.1. Eight new scripts give explicit defaults and append user args; original launchers unchanged.
- Run folders are independently reserved, auto names include task/L/F/M/mode plus timestamp/UUID; existing explicit training directory is rejected. Eval requires an existing run and reads sidecar before comparison, then weights_only state_dict with strict=True. Both core config and mandatory metadata.wrapper_architecture (position/ref/stem/placeholder version) are semantic checks. Runtime chunk/device/dtype remains separate. Eval creates a separate results subdirectory and never resaves weights; Pipe resave remains only in its unchanged old branch.
- Closed the prior runtime chunk gap as part of strict checkpoint integration: type must be int, not bool, and nonnegative; test validate/from_dict/save/load/compare across bool/float/NaN/Inf/negative/string, preserving file bytes and refusing invalid new files. No mathematical core change.
- Test command: `CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v`; final73/73 passed, no failures/errors/skips,26.554s. New13 task tests include four tasks × three modes default d/M small-N gradients; real N7225/972/11271/16641 with d8/M4 and fullL8F2; exact baseline reference/stem, original Darcy loss AST + normalizer decode and gradient term, strict/new-process checkpoint, byte-preserved sidecars, isolated run/eval paths, scripts and legacy AST. Twelve GPU wrapper FP32/FP16 AMP/BF16 AMP small cases passed. Local Python3.13.9/torch2.13+cu130/RTX5090 only; no remote target or real training.
- Self-review of five points is complete, no remaining new implementation defect. Preserved original plotting uses hardcoded85×85/221×51/129×129; nondefault-downsample visualization is unverified. This inherited boundary was reported, not silently changed.
- Freeze baseline102 files at `/tmp/cdlno-static-baseline-0a0j551e/hashes.json`, manifest SHA256 f46d355a8906b2b09b17c4f1ffc4d2f7266e8627769f069bd940c57fbc395685. Nine existing targets changed; remaining93 unchanged. Only5 of71 tracked files changed (four exp/model_dict); projection of the limited new branches reconstructs exact original module ASTs. No old data/loss/normalizer/optimizer/scheduler/loop/evaluation semantics or dependencies changed; no fake data files, exp imports, downloads, training, install, commit/push/PR/reset.


## Phase6 NS and Plasticity (2026-09-13)

- User authorized only NS and Plasticity. Added `TemporalStandardModel` and a temporal structured wrapper selected by `model_dict` for `cdlno_task='ns'|'plasticity'`; shared core is rebuilt on every forward and no latent/history crosses calls.
- NS contract is x[B,N,2], fx[B,N,10], T=None, output[B,N,1], fixed 64×64 layout and stem width74 from 64 reference distances +10 history channels. Existing exp loop still performs 10 teacher-forced truth feedback steps in training with one accumulated loss/backward/optimizer step, and prediction feedback in evaluation. No 10→20/40 extension.
- Plasticity contract is x[B,3131,2], fx[B,3131,1], T[B,1], output[B,3131,4]. Original time embedding is a wrapper-only time_fc created only for Time_Input=True; each of 20 times performs its own forward/loss/backward/optimizer update and scheduler rhythm. Labels stay spatial3131 × deformation4 × time20; no feedback.
- Added NS/Plasticity JSON presets and train/eval scripts with explicit d/h/M/L/F/epochs/batch settings (NS256/8/64/batch2; Plasticity128/8/64/batch8). Existing parser defaults and old model kwargs remain unchanged outside CDLNO branch. Strict sidecar + weights_only checkpoint and unique run/eval paths are reused.
- Targeted tests 8/8 and final full regression 81/81 passed (0 failures/errors/skips), local Python3.13.9/torch2.13+cu130/RTX5090. Tests cover B>1 truth/prediction windows, per-call core rebuild, T-dependent output and time gradients, 20 independent Plasticity updates, strict temporal checkpoint and AST loop contracts. No remote target or real trajectory training/evaluation.
- Self-review found no new implementation defect. Frozen original temporal data, labels, loss, optimizer/scheduler, loop and plotting code; only model branch, sidecar/run path and helper imports changed.
- Delivery review corrected only the invalid-task error message in `model_dict.py` to refer to six registered standard tasks; selection behavior is unchanged. The focused temporal suite was rerun with 8/8 passing.


## Phase7 ShapeNet-Car completed (2026-09-14)

- Stable `Car-Design-ShapeNetCar/models/CDLNO.py` consumes `(cfd_data, geom_data)` and returns `[N,4]`. Input is unchanged x7 (xyz3/sdf1/normal3), lifted by the original two-Linear GELU stem plus active placeholder. Output order is velocity3/pressure1. No y/geom/surf/pos/edges read in forward, no time_fc/geometry encoder, no mutation/reordering. Metadata batch/ptr validation rejects multiple graphs; absent metadata and valid all-zero single-graph Batch work.
- Shared core unchanged, default d256/h8/M64/L8/F2/entry/chunk0, configurable L/F/mode/chunk. A JSON preset, explicit train/eval scripts with trailing user args, and limited original main/main_evaluation branches connect the model. One run is one fold; new model CLI rejects batch_size!=1 and invalid fold0..8.
- `models/cdlno_run.CarRun` gives independent automatic run directories or a new explicit --run_dir; eval uses per-invocation results below the existing run. It reads sidecar before loading, compares architecture/wrapper/fold/epochs/weight/cfd_mesh/r, then loads the trusted whole object locally with weights_only=False/map_location. Exact model type/config and strict state keys/shapes are checked, then runtime chunk can change without weight rejection. Original train.py whole-model save expression and frequency remain unchanged. No new resume mechanism.
- Confirmed inherited eval bug: explicit --nb_epochs 200 parsed as float generated model_200.0.pth; one type=int change now matches training model_200.pth. The original Transolver whole-model evaluation also gets the local trusted torch.load compatibility keywords. No global serialization permissions changed.
- Final focused13/13 and complete94/94 tests passed (0 failures/errors/skips; full57.568s, /tmp/cdlno-phase7-final-tests.log). Real PyG2.3.1 Data/Batch and actual original train.train/test on a synthetic batch verified mask loss, all gradients, Adam update and OneCycle step. Variable N incl32186 with reduced d/M, no mutation/y dependence, ptr-only multigraph rejection, same-weight baseline stem, strict sidecar errors and two independent Car-cwd checkpoint processes passed. Additional CUDA FP32 off/entry/every PyG checks passed at N31/d16/M4. Local Python3.13.9/torch2.13+cu130/RTX5090; remote target remains unrun.
- Corrected incomplete draft gaps during self-review: placeholder, ptr, script overrides/default/run path consistency, isolated results and read-sidecar-before-load. Old3/84-test evidence did not establish these contracts; final13/94 supersedes it. One expanded test initially used nonexistent depth_score instead of w and was corrected.
- Full entry ASTs project to original commit after only approved branch/path/compatibility removal. All37 other original Car/AirfRANS files remain byte-identical, including train.py/dataset/original models/scripts. Continuation135-file snapshot at /tmp/cdlno-phase7-continuation-baseline/hashes.json protects prior uncommitted work; phase0–6 code/tests and dependency files untouched.
- No editable install was executed: independent import tests used PYTHONPATH pointing only to repo root and launched from original Car cwd. No real data/VTK/radius_graph/drag evaluation or trajectory training, no remote compatibility run, dependencies, commit/push/PR/reset. Phase8+ remains unauthorized.


## Phase8 AirfRANS completed (2026-09-14)

- No existing Air workspace modifications at phase start; Transolver YAML matches stage0/base398epochs. Only appended CDLNO key copying its six training/data fields; original key values and file prefix preserved. JSON records architecture defaults and initial YAML values; explicit CLI may override epochs/batch/lr, otherwise current YAML wins.
- `cdlno.airfrans.AirfRANSModel`, exported as local models.CDLNO.Model, uses x7+[64 distances from pos2] →71-input MLP+placeholder →shared point core →[N,4] vx/vy/p/nut. Reference domain [-2,4]×[-1.5,1.5], NumPy linspace tofloat32 and x-major flatten match original. No label read, time_fc, geometry encoder, internal sampling or node mutation. Core modules unchanged.
- Air graph contract deliberately handles the original Infer_test's stale single ptr after slicing batch/x/pos: allzero batch[N] and two-entry ptr[0,original_N>=N] accepted. Multiple graph ptr or nonzero batch rejected, even if sampled batch allzero but ptr reports multiple graphs. This follows original code without modifying sampling/scatter/graph construction.
- New cdlno_entry.AirRun supports main and main_evaluation CDLNO selection, defaultTransolver/full evaluation preserved. Stable shared class avoids ambiguous models-package checkpoint references. Unique run root holds sidecar and original complete models list CDLNO; each original train.main save/log goes into member_i/model; results get unique eval path. Sidecar reads/comparisons precede unpickling; validate exact class/config/list length and strict weight schema, with allowed runtimechunk changes. Local trusted torch.load kwargs also preserve baseline list loading on modern PyTorch; no global permission change.
- New scripts explicitly specify d256/h8/M64/L8/F2/entry/chunk0 and398epochs/batch1/lr.001, user args last. Train --my_path=/.../Dataset; eval --my_path=/... parent. New training script score0; no actual score invocation occurred. docs/CDLNO_TASK_LAUNCHERS.md inventories all8JSON configs and16scripts.
- Focused13/13 passed8.123s, complete107/107 passed76.872s (0fail/error/skip), logs /tmp/cdlno-phase8-focused.log and /tmp/cdlno-phase8-final-tests.log. Actual PyG2.3.1 Data/Batch, N9/31 and32000smallwidth, no mutation/y leakage, staleptr/multigraph, original train.train weighted MSE/all gradients/Adam/OneCycle and test function verified. No train.main epoch/sampling loop executed.
- Two fresh Air-cwd processes (PYTHONPATH=root, no installation) execute extracted actual model-selection, whole-model/list save and eval-load branches; both member and list outputs agree acrosschunk0→1, sidecar bytes preserved. Model type/list count/strict keys and mismatch-before-load/isolated directory tests pass. Reference and stem identical at same weights after removing only original hardcodedcuda device call for CPU test. Extra CUDA FP32PyG off/entry/every checks passed N29/d16/M4. Local3.13.9/torch2.13+cu130/RTX5090 only, no remote target.
- Frozen proof: both whole entry ASTs project to original, old YAML keys identical,21other original Air files byte-identical. Stage7 freeze test updated only to hand off the now-authorized3Air files to these checks; Car implementation and all preceding task/core code unchanged. Baseline136files at /tmp/cdlno-phase8-baseline/hashes.json protects prior work.
- Unrun: real datasets, VTK/manifest, per-epoch sampling/20 repeated validation samples, radius_graph real path, Infer_test/Results_test/scatter/averaging/boundary/coefficients, true training/metrics/performance, remotePython3.10/3.11+torch2.11/cu128. Existing Car training evidence does not validate these Air paths. No new defects found in the authorized interface/loading scope; await review, no phase9/10 execution.


## Phase9 completed: synthetic performance only (2026-09-14)

- User approved phase8 and authorized only v1.2 §9 plus thin matched LRSA. Added tools/cdlno_benchmark.py, tools/cdlno_perf/{models,costs,measure}.py and test_performance.py; production core/wrappers/config/checkpoint/task/dependency files untouched. Matched composes L complete LRSA blocks with identical task stem and output LN/head, without bridge/CDPA/final-up or changing P legality.
- Eight task presets come from current launchers/main and new JSON; task and matched comparisons are explicit. Original Transolver uses its actual source/attention/init with only in-memory device/import adaptations. AirfRANS unused mlp_new is retained in total parameter count and missing-grad inventory.
- Independent closed-form MAC checks include all dense N projections/Conv, two latent front FFNs, GEGLU, stem/head, per-source Cross and current Q once. Non-matrix bias/norm/depth/softmax/reductions and stack/clone/cast payloads separately recorded; no false exact total FLOPs or summing storage payloads into peak memory. Default counts3/3/8/3, entryS2/historySDPA1, everyS27/historySDPA6 atchunk0; chunk2every15 calls. No cache/detach/sharing/math change.
- Final full regression118/118 in59.769s; no failures/errors/skips. Log docs/performance/phase9/regression.log. Twenty-seven successful GPU rows across only three configurations: Elasticity matchedN972, Airfoil task221x51/B4, Airfoil matched17x23/B2. Same-weight chunk0/1/2 maxoutputabs1.788139e-7/maxgradabs1.430511e-6. Raw JSON preserves all samples/metadata/profiling/costs/state counts and source hashes.
- Local Python3.13.9/torch2.13+cu130/PyG2.3.1/RTX5090 Laptop/driver591.86. FP32, TF32off, AMPoff, compileoff, backendauto actually efficient, threads1,5warmup/20measurements, synchronized perf_counter. Synthetic single-call MSE/AdamW foreachFalse, fresh optimizer perchunk, initialized states and25successfulupdates verified; untimed diagnostics include one separately recorded extra step.
- Elasticity entry0.627GMAC versus Transolver1.127G, but train34.96ms versus24.30ms; profiler counts169/200activeAdamWtensors and7234/8218ATencalls identify additional small-operation/launch/optimizer work. Airfoil taskentry77.64ms versusTransolver135.96ms, withdifferentheads4/8. These are finite local model-only samples, not epoch estimates or guaranteed speed claims. Entrychunk0/2equivalent organization still has timing noise; no chosen-optimalchunk conclusion.
- During continuation old/tmp logs/sessions were unavailable. Earlier timing numbers were excluded from final acceptance; final checks/results persisted under docs/performance/phase9. New152-filecontinuationbaselineSHA256=288568f43d659168d9acf52e5523931ea01ef3a8191ceb0a4bd7a4b03400699d. Final continuation comparison:149unchanged, onlyAGENTS/STATUS/memorychanged,0unexpected (docs/performance/phase9/freeze-check.json). Initial145-filechecks had shown no existing modifications before3statusdocedits; do not fabricate the lost initial manifest.
- Five priority points self-reviewed against source/formulas/tests: no remaining new architecture/interface defect. Unrun: remotePython3.10/3.11+torch2.11/cu128, realdata/graphs/metrics/timeepochs, fulltaskgrid/sweeps, AMP/TF32on/forcedbackend/compile tool paths. No realtraining, dependencies, commit/push/PR. Phase10 requires explicit authorization.


## Phase10 completed: final audit and delivery (2026-09-14)

- User approved phase9 and explicitly authorized final review only. Reviewed actual modules/core/CDPA/wrappers/run helpers, every existing entry diff and baseline/source licenses. Produced README CDLNO instructions, docs/CDLNO_IMPLEMENTATION_REPORT.md, exhaustive §§0/8/13/14 requirements matrix, third-party notices, eight-task position/time/checkpoint inventory, and persistent final_audit evidence/patches. No production or dependency changes.
- Found evidence gaps, not a production defect: previous NS synthetic test ran3 steps; previous temporal AST test checked strings, not whole modules. Changed only tests/test_temporal_standard.py to full10 training/eval feedback steps with explicit window/one-backward-step checks, distinct Plasticity per-sampleT/time_fc gradients, and complete NS/Plasticity AST projection to phase0. Supersedes earlier reports' overbroad temporal validation claims.
- Initial118/11858.082s; focused9/9 10.114s; final119/11949.871s, no failures/errors/skips. Final command CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v; log docs/final_audit/regression.log. Core21cases×point/grid, expandedL/F, independent parameters/raw-history/gradient/counts, CPU real-N/originalloss, truePyG2.3.1 industrial and threecwd independent-process strict loading all pass. SixGPU test methods actually execute on localRTX5090Laptop, Python3.13.9/torch2.13+cu130. Remote3.10/3.11+torch2.11/cu128 and editableinstall unrun.
- LRSA fixed47b03f8 clean checkout direct same-weightFP64 block reference2 tests gives output/history/grad maxdiff0; no LICENSE/COPYING/projectlicense metadata found. Transolver rootLICENSE is MIT(Copyright2024THUML), IPOT MIT(Copyright2023SeungjunLee) re-fetched from pinned18c1778; original Air ODbL preserved. Do not repeat an erroneous impression that Transolver has no rootlicense. No vendored LRSA/framework/kernel. Missing standalone theory attachments remain unlocated; no proofscript/PDF reproduction claim.
- Baseline71files:58identical+12prior approved integration changes+README original-text-preserving addition. Phase10start162manifestSHA aaeefa892951c977ad1336392964a11eca1562fe22749d0ba15a6f99e03a2085;156unchanged,6test/docchanges,0unexpected. Existing phase9code/rawperformance, core/wrappers/entry/dependencies unchanged this phase. Final freeze and reviewpatches in docs/final_audit; no staging/commit/push/reset.
- Self-reviewed architecture/init, two-levelCDPA/FP32/chunks, history/independence, adapters/checkpoints, freeze/performance口径; no remaining known production defect. Still unverified realdata/VTK/graphs/fullsampling/CFDmetrics, true training/convergence/accuracy, realepoch speed, remotetarget/install, fullGPUdefaulttaskmatrix and independent mathattachments. Phase9limited GPU performance is historical measured evidence, not repeated timing here. No further work authorized until explicit instruction.

2026-09-16 K6完成并自审：Car新family/原整对象协议与真实PyG原loss合成训练、5测试和4旧同权重回放通过；继续用户连续授权K7–K10，未运行真实数据。

2026-09-16 K7完成并自审：八任务KCDNO all/off入口齐备，Air真实PyG/原loss/整对象列表/原cwd加载通过；继续已授权K8–K10。

2026-09-16 K8完成并自审：trainable lrsa_matched/full独立family与八任务共享接入，18测试通过；修复Air新增parser argv=None，继续K9–K10。

2026-09-16 K9完成：15 CPU测试、两有限GPU结构配置通过；all较matched未显示提速，完整成本/热点已报告。继续K10最终审查，无真实数据。
