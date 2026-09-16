# MSAR-LNO 实施状态

## 2026-09-17：独立训练后评估脚本

用户另行授权将终端函数保存为脚本。新增`tran_evaluate/msar_lno/run_msar.sh TASK [light|full] --gpu ID`，复用八个既有任务脚本；默认Light、GPU取`MSAR_GPU`或0、coverage floor/.01/.2。自动按checkout/path.sh生成时间戳目录，训练成功后评估同一run；AirfRANS转为CUDA_VISIBLE_DEVICES。支持`--dry-run`，失败退出码保留，脚本不预创建run。README新增实际用法，原模型/训练/数据/依赖/已有脚本未改。

基于`main@3e88166e0d483824be49cf6ebfc2d2d89c5c53c5`干净工作区实施。`bash -n tran_evaluate/msar_lno/run_msar.sh`通过；八任务×Light/Full×GPU0/1共32对、64次真实脚本dry-run通过，从仓库外和含空格路径检查同run/参数/路径，未调用任务Python或创建输出。临时隔离shell桩检查训练失败不评估、评估失败退出码、两种GPU传递、环境变量优先级、默认值/help/非法参数均通过。没有启动真实训练，无新模型验收或commit/push；旧M9结论不变。**本阶段结束，未执行下一阶段。**

## 2026-09-17：M9 成本、有限性能、独立复查与最终交付完成

M8已审查通过；本轮仅M9。见[最终报告](MSAR_LNO_IMPLEMENTATION_REPORT.md)、[总控26条矩阵](MSAR_LNO_REQUIREMENTS_MATRIX.md)、[八任务四种命令](../tran_evaluate/msar_lno/README.md)及[实际结果](msar_lno_audit/m9/summary.json)。本轮没有模型/训练/数据生产改动，也没有真实数据实验。

- 实际构造八任务×Light/Full，逐key统计全部参数；公共四Down/十二latent block/四Up/三融合的参数分别Light203280/1339104/148008/288、Full812064/5332416/590928/576；stem/time/head逐任务另列，w恰好3d、coverage无参数。完整矩阵MAC包括N尺度投影、所有Q/K/V/O、两FFN、M² SA和final Up/head，与实际Linear/SDPA hook一致；norm/bias/softmax/Pair/coverage等另列，不把2×MAC冒称精确全操作FLOPs。
- 13个本地GPU行全部通过：Elasticity N972/B1的Light off/floor、Full off/floor、原Transolver、生产matched LRSA、KCDNO all；AirfRANS真PyG合成N32000/B1四MSAR math行，再加Light两行efficient SDPA。FP32、AMP/TF32/compile off、warmup5/测量20、CUDA同步median/p90、完整synthetic MSE(+coverage)/AdamW步和峰值allocated显存。对照d/M/参数不同，不称公平速度/精度对照。
- coverage默认.2/.01不改；同profile同初始权重，正常eval off/floor exact。Air N32000 Light efficient：step中位数58.047→61.876ms、峰值351.937→1221.181MiB（+869.244MiB），说明显式A训练成本；math后端原本存权重，峰值差很小，不混淆后端与模型收益。推理不计算coverage；显式diagnostics是另一路no-grad观测开销。
- 新6方法；性能套件初5+旧15=20通过19.059s；独立M2/M3的36方法通过，与新增命令方法首轮37项有一个新helper调用TypeError，修正新测试后该项通过2.055s，初轮日志保留。最终57个不同方法均通过，非单次57运行。32种真实train parser/32种eval预览通过，没有调用真实数据入口。
- M8八任务正式Light原loss合成训练/eval、Full synthetic MSE反传/strict state、工业真PyG及小d GPU/任务保存协议证据有效保留；75旧同权重回放和182旧artifact hash由M8提供，M9再次核对hash/生产冻结，不虚称重跑75次。GUNet/完整Air采样仍受原torch_cluster缺失限制。
- 新工具`tools/msar_benchmark.py`、`tools/cdlno_perf/msar.py`复用既有工具；仅修改既有`measure.benchmark`增加可选显式loss_closure，默认旧MSE/返回不变，custom+compile拒绝。新独立手算例验证Down/Up/AttnRes/coverage；总控反查没有发现确认架构冲突。旧README完整保留并新增MSAR章节，旧阶段文档不改。
- 起点865文本快照`/home/hwz/CDLNO-artifacts/msar-m9-before-e8vxz5y5/source`和用户diff保留；六个已有文件变化（measure、两README、三STATUS/memory），其余859字节不变。所有生产模型/配置/入口/数据/loss/optimizer/时间/评价/依赖及旧测试不改。未安装/下载/真实训练/commit/push/PR。

未验证：真实数据读取完整性/收敛/精度/epoch时长/SOTA、目标Python3.10/Torch2.11/cu128、完整工业抽样/物理指标、所有正式大batch/任务AMP/compile矩阵。本机为Python3.13.9/Torch2.13cu130/PyG2.3.1/RTX5090 Laptop；有限合成计时不升级这些结论。任务checkpoint仍非完整optimizer/scheduler/RNG归档，无新resume。

**本M阶段结束，未执行下一阶段。**

## 2026-09-17：M8 八任务综合验收完成，等待审查

M7已审查通过；本轮仅M8，生产代码及既有测试均未修改。见[M8报告/覆盖表](MSAR_LNO_M8_ACCEPTANCE.md)、[逐任务矩阵](msar_lno_audit/m8/coverage-matrix.json)、[结果](msar_lno_audit/m8/summary.json)、[冻结](msar_lno_audit/m8/freeze.json)。下面M7及更早条目的未运行结论是各阶段当时状态，以本轮明确执行的项目补充。

- 八任务正式Light（d96、M512/256/128/64）各完成off/floor原loss合成forward/backward/optimizer/eval，共16例。四静态任务小N，NS保留N4096/10步，Plasticity保留N3131/20时间更新，Car/Air真PyG单图N19。正式Full（d192、M1024/512/256/128）八任务各完成单次合成MSE+coverage反传、eval、strict state_dict零容差往返；没有声称Full原loss完整训练矩阵通过。
- 同权重off/floor预测最大绝对差2.235e-7（atol2e-6/rtol1e-4）。off/weight0的total就是原LPDE对象且不请求coverage A；floor正确加权。默认kappa.2本次Light初始raw均0；kappa1针对性检查显示四Down图连接/有限梯度、每个真实Down在变化source上有效非零梯度，无decoder/AttnRes aux梯度。最深层初始合法零梯度不当作断图。
- 实际Light hook计数Down4/Up4/PairFusion3/latent block12，SA12/FFN24；三w0融合exact E+U、source轴2/slot对应/梯度通过；无Conv/history/CDPA/kernel/N点SA，参数/storage独立。mask、batch、M1>N、连续两次独立backward/无跨调用aux状态均通过。
- 新19方法首轮18通过/1个测试断言失败（错误要求每层每次非零），修正新测试后对应1项重跑通过，分别耗时97.619s/0.122s；没有生产修复或扩大容差。既有MSAR107方法106通过/1跳过，126.930s；旧157方法0失败/错误、1个Air子用例跳过，181.846s。两处跳过均为原torch_cluster缺失，不冒充执行。
- M0全部可用75份原固定输入/同一权重夹具本轮重放exact（K0 41、K/matched wrapper24、core6、其他4），182旧artifact hash不变。旧GUNet缺torch_cluster仍未执行。三项目cwd新进程factory/选择、strict或原整对象/列表加载、错误架构拒绝、eval不覆写sidecar通过；任务checkpoint协议用M5—M7小结构重跑，Full纯state另测，不新增resume。
- 本机Python3.13.9/Torch2.13+cu130/PyG2.3.1/RTX5090 Laptop。八任务缩小d8的GPU FP32原loss步骤通过；M2/M3有限AMP重跑通过。正式Light/Full任务GPU、远端2.11/cu128、Full原loss优化矩阵、完整工业抽样/物理指标、真实数据读取/训练/收敛均未执行，报告含远端无数据补充命令。
- 837起点文本快照`/home/hwz/CDLNO-artifacts/msar-m8-before-fhesnwse/source`及用户diff保留；834既有文件字节相同，只增量更新三份STATUS/memory。仅新增测试、报告与审计证据；模型/factory/配置/数据/loss/时间循环/optimizer/依赖/旧测试不改。未下载/安装/真实训练/commit/push/PR，未执行M9。

**本M阶段结束，未执行下一阶段。**

## 2026-09-17：M7 ShapeNet-Car / AirfRANS接入完成，等待审查

M6已审查通过；本轮仅M7。八任务现已接入`msar_lno`选择/Light默认/Full显式/coverage floor或off及原保存协议。新增工业wrapper共享同一已接受core，保留Car tuple→velocity3/pressure1、Air Data+reference64→vx/vy/p/nut、单图batch1与可变N。没有卷积，未删除任何图构造/采样，未修改MSAR核心或旧模型数学。

- 工业训练使用既有显式`training_forward/training_objective`，无model.last_loss；Car原全点速度MSE+reg×表面压力MSE，Air入口原体积MSE+reg×表面MSE保持。每个单图forward只加入weight×四层raw均值，目标日志按优化步平均，coverage off/weight0不计算A。surf不成为coverage mask；没有新面积权重或标签权重。
- 工业checkpoint保留Car整对象、Air整成员+列表，稳定类路径`cdlno.msar_lno.industrial.{CarModel,AirfRANSModel}`。先读完整family/架构/任务sidecar，再构造或本地可信unpickle；严格key/shape/行为/reference/成员隔离校验。eval coverage请求不改训练metadata，重复eval不覆盖训练文件；不提供新resume。
- 新14方法；MSAR全套**107项/166.280s，106通过、1因缺torch_cluster跳过，0失败/错误**。真实PyG floor/off原loss/优化/eval、变长节点/拒绝多图、整对象/列表往返、两个原cwd新进程加载通过；Car完整合成epoch两模式通过，Air原训练函数+记录及scatter片段通过，完整采样epoch明确未执行。
- 旧相关90方法首轮有2个测试适配问题（新YAML键及片段局部变量），修正后对应2项定向重跑通过，其余原模型回归通过；原Air缺torch_cluster仍2处skip。14旧工业固定权重夹具exact（8 K0 + 6 M0），182旧夹具文件hash保持。完整日志保留，未把分次结果冒充一次全绿运行。
- 本地GPU两工业小结构FP32/MATH原loss更新通过；正式Light/Full B1/N11前向及实际参数统计通过（Car 1,711,420/6,814,324；Air 1,723,708/6,838,900）。本机Py3.13.9/torch2.13+cu130/PyG2.3.1/RTX5090Laptop；远端2.11/cu128、大profile反传/性能、工业AMP、完整数据读取/训练/收敛/物理系数未验证。
- 起点807文本快照`/home/hwz/CDLNO-artifacts/msar-m7-before-bocerv4d/source`保留；19已有文件增量、15新增实现/测试/交付文件、788起点文件字节相同。6个完整工业main/eval/train AST剥离精确新family分支后与pre-M7相等；loader/metrics/旧模型、PDE入口、公共原语/已接受MSAR数学不改。先存用户diff保持，无安装、真实训练、commit/push/PR。

[八任务覆盖及M7报告](MSAR_LNO_M7_INDUSTRIAL_TASKS.md)；[实际命令](../tran_evaluate/msar_lno/README.md)；[结果](msar_lno_audit/m7/summary.json)、[freeze](msar_lno_audit/m7/freeze.json)、[夹具](msar_lno_audit/m7/fixture-index.json)。自审5项通过。剩余限制为Air torch_cluster/完整后处理、原Car固定drag路径/fold0及原日志缺陷、远端与真实数据验证；未扩大信任或训练协议。

**本M阶段结束，未执行下一阶段。**

## 2026-09-17：M6 NS / Plasticity接入完成，等待审查

M5已审查通过；本轮仅M6。`msar_lno`已接入真实`exp_ns.py`/`exp_plas.py`，新增共享`cdlno.msar_lno.temporal.TemporalModel`及薄factory出口，复用原MSAR四级core/单forward objective/strict Run。六PDE可选Light默认、Full显式、coverage floor/off；Car/AirfRANS未接入MSAR，未执行M7。

- NS保留64×64、fx10、单次out1、ref64+fx stem74；原10帧teacher forcing训练求和loss→1次optimizer/scheduler，eval10帧预测回填。新增`rollout_objective`只对该次更新的实际forward raw coverage取时间均值：原LPDE_sum + weight×mean_q(mean_4levels)。不放大为10份辅助项，不改原PDE尺度。
- Plasticity保留101×31、fx1、T[B,1]、out4；原sin/cos+SiLU time_fc注入，原随机collate匹配T/label、20次逐时间optimizer/1次batch scheduler。每次LPDE加本次coverage，epoch目标日志按optimizer更新数平均。无卷积、placeholder、跨时间E/D/aux缓存；没有把时间并入N。
- eval只用prediction，显式模型级诊断开关验证预测不变；按保存结构重建、task/time合同严格校验、无strict=False，coverage差异不阻止纯eval；原裸model.pt保存频率及独立output目录保留，无新resume。六任务命令见[脚本说明](../tran_evaluate/msar_lno/README.md)。
- [M6报告/覆盖表](MSAR_LNO_M6_TEMPORAL_TASKS.md)、[summary](msar_lno_audit/m6/summary.json)、[冻结](msar_lno_audit/m6/freeze.json)、[夹具](msar_lno_audit/m6/fixture-index.json)。MSAR93/93通过（38.463s，11个M6新方法）；相关旧93项零失败/错误（204.050s），2处原Air torch_cluster skip记录，含1子用例。
- 两任务×floor/off CPU B2/真实N/d8/M7,5,3,2完成原时间循环和原loss/反传/optimizer/eval/checkpoint。CUDA B1/d8 FP32 MATH两完整时间batch通过。正式Light/Full B1真实N CPU前向通过，参数NS 1,723,897/6,839,281；Plasticity 1,729,180/6,886,708；正式大profile反传未执行。
- 6train+6eval真实parser预览、两wrapper原PDE cwd新进程strict load、eval sidecar/config字节不变、time条件梯度、无未来标签泄漏、off原loss对象同一、逐forward新图均通过。26份旧同权重夹具（14时间+12标准静态）及4份新捕获pre-M6已接受M5夹具零容差回放；182旧artifact hash保持。
- 起点784文本快照`/home/hwz/CDLNO-artifacts/msar-m6-before-pqh_ca0e/source`；13已有文件增量修改，其余771字节一致。两完整时间入口去掉显式MSAR分支后AST精确相同；旧模型、M1/M2/M3数学、M5静态wrapper/入口、数据、工业和依赖冻结。保留全部用户工作，HEAD不变，无commit/push/PR/下载/实际训练。

当前NS源码只有已确认10→10，无现成20/40步CLI/标签读取，不虚构长时命令；旧NS命令保留。未验证真实数据/训练/收敛、远端Python3.10/Torch2.11cu128、MSAR工业PyG、任务AMP/compile及大profile完整训练。首轮1个新测试用非法history-mode=2导致argparse提前退出，改合法但不适用的off后通过；生产数学和容差未改。**本M阶段结束，未执行下一阶段。**

## 2026-09-17：M5 四静态任务接入完成，等待审查

M4已审查通过；本轮仅M5。Darcy/Elasticity/Airfoil/Pipe现已在真实exp入口接入msar_lno的CLI、factory、共用逐点wrapper、原LPDE+coverage、评估、严格state_dict和既有输出记录。NS/Plasticity/Car/AirfRANS尚未接入MSAR，未执行M6。

- 新`cdlno.msar_lno.standard.StaticModel`保留Darcy ref64+coeff和其余fx=None/placeholder；core自带唯一输出head，无Conv、额外head或递归初始化。四任务共享代码、参数对象独立；Light默认d96/M512…，Full d192/M1024…，不按N裁剪。
- `standard_entry.StandardRun`沿用裸model.pt及原频率，先读保存的resolved结构/输入合同，再校验显式覆盖；coverage差异允许纯eval，旧训练config/sidecar不改写，无新增resume。自动目录含task/family/profile/coverage；显式save_name/run目录防覆盖。原epoch/batch/lr/AdamW/scheduler/clip/normalizer/数据/指标保持。
- 真实训练分支调用M4显式adapter；原loss语句不改，off/weight0的total就是LPDE且不请求A。新四项日志按optimizer step求均值；原train_loss按原定义保留。JSON objective或CLI均可开关，用户参数最后覆盖；周期可视化复用既有预测路径。
- [M5报告与八任务覆盖表](MSAR_LNO_M5_STATIC_TASKS.md)、[四任务×Light on/off/Full命令](../tran_evaluate/msar_lno/README.md)、[summary](msar_lno_audit/m5/summary.json)、[freeze](msar_lno_audit/m5/freeze.json)。MSAR82/82通过（含15个新M5方法）；相关旧79项零失败/错误，2处Air torch_cluster依赖skip，其中1处为子用例。
- 四任务×floor/off合成原loss完整batch、反传/optimizer/scheduler、原eval片段和strict checkpoint通过。正式N用小d/M独立layout检查；5×7非方形wrapper通过，Darcy完整导数loss沿原方形合同用7×7。正式Full Elasticity N972/M1024前向通过；正式profile反传未运行，不将小模型训练步冒充正式大网格训练。
- 新四任务小GPU FP32 MATH原loss step、新进程原PDE cwd严格加载、12train+12eval真实parser预览、early记录/四项目标/两次eval不覆盖通过。新MSAR可视化dispatch检查eval状态/RNG/权重不变，实际renderer继承原验证。M2/M3有限AMP也重跑；远端2.11/cu128、真实数据读取/训练/收敛、MSAR工业PyG未运行。
- 本轮65份旧模型（K0 41+K/matched wrapper24）同权重CPU FP32 MATH零容差回放，M3 Light/Full两份core精确回放；其余未触及10份沿用M4。182旧artifact hash保持。707文件快照`/home/hwz/CDLNO-artifacts/msar-m5-before-2jli41sc/source`；四完整旧入口移除新增family分支后AST精确相等，旧模型/共享数学、M1/M2/M3数学、数据和依赖冻结，所有先存用户修改保留。

初次失败仅为新测试AST定位、Darcy矩形夹具、子进程backend未统一及旧冻结投影未识别新分支；修正测试后零容差通过，未改模型来迁就测试。实际原eval仍有继承的固定绘图网格限制，未声称所有非默认downsample导出已验收。无下载/真实训练/安装/commit/push/PR。**本M阶段结束，未执行下一阶段。**


## 2026-09-16：M4 模型选择、显式训练目标和checkpoint基础完成，等待审查

仅M4。已接受M3不改数学；真实PDE factory新增独立msar_lno分支，薄Model导出稳定的`cdlno.msar_lno.core.MSARLNO`，输入仍是已提升E0[B,N,d]。**八任务生产MSAR train/eval尚未接入，不可直接给exp/main加新模型名声称训练可用。** 静态/时间/工业wrapper与loss循环分别留M5/M6/M7。

- 新`registry.core_class`懒加载与`entry.core_model_kwargs`仅服务msar_lno；旧key返回None/空kwargs，CLI默认/旧模型数值行为不变。
- 新`objective.training_forward/training_objective`显式per-call aux；原LPDE先计算，floor有效才加weight×四Down raw mean。off/weight0不请求A且total就是原LPDE张量；四项detach日志pde/raw/weighted/total交调用者，无last_loss、配置切换或跨forward缓存。生产epoch日志后续接线。
- 新`checkpoint`保存完整M1 metadata+core接口/output/dtype/member合同，保留裸state_dict/可信整对象/列表/单成员格式；先读结构再验证显式覆盖、严格加载。coverage差异允许纯eval，不覆盖sidecar。新目录exclusive，无新的任务resume；这些core快照不是优化器/RNG续训档案。
- [M4报告与公共/任务接入表](MSAR_LNO_M4_INTEGRATION.md)、[结果摘要](msar_lno_audit/m4/summary.json)、[freeze](msar_lno_audit/m4/freeze.json)、[fixture索引](msar_lno_audit/m4/fixture-index.json)、[安全真实factory构造示例](msar_lno_audit/m4/construct_example.py)。M4最终18/18；既有M1—M3 49/49；旧入口33/33，总100个不同方法均通过；不是单次全仓库suite。M0全部75同权重夹具及M3 Light/Full两份精确回放，182旧artifact hash保留。
- 原relative-L2安全loss连接合成core训练步通过；**未验证真实Darcy完整decode/边界loss或八任务训练链**。三cwd新进程core state/whole/list加载与小CUDA FP32 MATH loss/checkpoint通过；M2/M3套件有限AMP也重跑。PyG新wrapper/真实训练/远端2.11cu128未执行。
- 本轮新生产4文件、旧factory3行+MSAR registry扩展；旧freeze测试只增加精确新分支AST断言后移除，不扩大忽略规则。所有任务数据/train/eval、旧模型/共享数学、M1配置/M2/M3数学和依赖字节不变。23处先存tracked修改保留；678文件起点快照`/home/hwz/CDLNO-artifacts/msar-m4-before-i75l1rvp/source`。

无新架构冲突，无真实训练、安装、commit/push/PR；M5未执行。**本M阶段结束，未执行下一阶段。**


## 2026-09-16：M3 四级 core 完成，等待审查

仅M3：新增稳定路径 `cdlno.msar_lno.core.MSARLNO` 和独立no-grad诊断，输入为已提升E0[B,N,d]，core含最终LayerNorm+Linear输出头；任务lift/坐标/时间及八任务接入仍留后续。没有修改M1/M2、旧共享模块、旧模型、factory、数据/训练/评估/依赖或已有测试。

- 精确四级Light/Full、双depths=[3,1,1,1]；实际Down4、Up4、encoder6、decoder6、latent SA12、latent FFN24、fusion3；无Conv/旧history/CDPA/Bridge。每层实例/storage独立。D4直接Decoder4(E4)，3→2→1严格Up(E,D)→fusion→Decoder；final Up(E0,D1)后直接head，无E0skip/N点SA。
- 默认forward/纯eval为预测Tensor；显式aux返回prediction、四层raw FP32 coverage及均值。off/weight0不请求Down的A或构造coverage图；floor仅Encoder Down收集。显式diagnostics另按no-grad观察，训练A可复用，否则重算观察QK；默认关闭，无CPU统计/跨batch状态，预测与RNG不变。
- [M3报告/shape trace/参数归属](MSAR_LNO_M3_CORE.md)、[摘要](msar_lno_audit/m3/summary.json)、[freeze](msar_lno_audit/m3/freeze.json)、[新夹具索引](msar_lno_audit/m3/fixture-index.json)。16/16新core +20/20 M2 +13/13 M1=49测试，无失败/错误/skip。独立完整公式/梯度、一次合成MSE+coverage AdamW step、所有stage梯度、严格state与可信整对象/列表三个cwd新进程加载通过。
- 真实Light B2/N35和Full B1/N7保留正式M/d/heads，不裁M；构造/前向及新进程同权重eval/floor夹具精确回放。四通道head、无input lift的实测参数分别1,691,260/6,737,140；不是八任务最终参数。Full严格state_dict往返通过。小core本机CUDA MATH FP32/FP16 AMP/BF16 AMP有限训练检查及off/floor预测比较通过。
- 本轮旧K0 41+pre-M1核心6=47/47精确回放，其他28个旧wrapper/模型沿用M2有效证据和未变源码；182个旧夹具文件hash不变。720起点文件除3个状态/记忆文档增量外均未改。快照 `/home/hwz/CDLNO-artifacts/msar-m3-before-jx4v53vh/source`，相邻`post-m3-core`为本轮新模型夹具，不冒充历史训练权重。
- 本机Python3.13.9/torch2.13+cu130/RTX5090 Laptop；远端2.11/cu128、真实数据训练/收敛/精度、完整任务与大配置性能未验证。没有任务生产脚本修改、依赖安装、commit/push/PR或真实训练。五项源码/证据自审通过；未来M4必须接真实family/任务/损失/metadata合同，当前不能用八任务命令训练MSAR。

**本M阶段结束，未执行下一阶段。**

## 2026-09-16：M2 原子组件完成，等待审查

本轮仅M2。新增独立 `cdlno/msar_lno/modules.py`：`LearnedQueryDown`、`LatentFFNSAFFNBlock`、`QueryAlignedUpCross`、`PairwiseAttnResFusion`、`CoverageFloorLoss`及显式no-grad诊断。复用M0确认公共原语，旧共享文件、M1配置、factory、八任务、数据/训练/评估/依赖及已有测试均未改。没有MSAR core或任务接入。

- Down直接P[M,d]作为Q，无Wq或query残差；bool[B,N]有效mask在softmax前处理，允许M>N。只有训练+return_aux+有效coverage才显式构造A；off/weight0/eval走SDPA，无新增aux矩阵或模块缓存。
- Latent严格三残差FFN1→SA→FFN2、独立pre-RMS/2d GELU；Up为receiver-head纯读取，无隐藏残差/FFN。Fusion仅w[d]、零初始化、raw E/U、来源softmax、固定2；初始E+U与直接Jacobian=1实际验证。
- Coverage仅head/latent均值→source求和→batch/层均值，mask内均匀mu或可靠显式measure；强制FP32关闭autocast、无参数，返回raw loss。独立loss到P/K/norm/输入有效梯度，V/O未被误连。CoverageRatio/DiversityRatio只是显式no-grad诊断，无默认熵/其他正则。
- [M2报告及公式映射](MSAR_LNO_M2_PRIMITIVES.md)、[摘要](msar_lno_audit/m2/summary.json)、[源码freeze](msar_lno_audit/m2/freeze.json)。20/20新组件、18/18旧共享原语、13/13 M1配置，共51测试通过；M0同权重75/75精确回放，182个夹具文件hash不变。新reference仅math/torch，double数值gradcheck和生产解析梯度对照均通过。新增测试包含一次合成MSE+coverage AdamW step，不是原任务loss或完整模型训练。
- 本机RTX5090 Laptop、Python3.13.9/torch2.13+cu130：有限FP32/FP16 AMP/BF16 AMP前向反传和aux FP32检查通过。GUNet图链、远端torch2.11/cu128、完整MSAR/真实数据训练未执行；不升级M0已有未验证项。
- 快照 `/home/hwz/CDLNO-artifacts/msar-m2-before-bovueh8k/source`；695起点文件中仅3个状态/记忆文档增量，其余692字节不变。旧历史/夹具/M1保留，无安装、下载、commit/push、PR或真实训练。自审5项通过，未发现需要修改确认公式的阻断冲突。

**本M阶段结束，未执行下一阶段。**

## 2026-09-16：M0 补审完成，已核对先前 M1，等待审查

本次按用户纠正后的授权，**在已有M1之后补做M0**。仅新增审计文档、只读分析脚本和仓库外独立夹具，没有重做M1或实施M2。下方M1原文保留，不能倒填为当时已完成M0。

报告：[MSAR_LNO_REFERENCE_AUDIT.md](MSAR_LNO_REFERENCE_AUDIT.md)；[阅读账本](msar_lno_audit/m0/READING_LEDGER.md)、[回归夹具索引](msar_lno_audit/m0/fixture-index.json)、[源码冻结](msar_lno_audit/m0/freeze.json)、[完整测试日志](msar_lno_audit/m0/regression.log)。

- 审计起点main/`9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`，已有23个tracked修改及未跟踪工作全部保留。653文件清单：628文本、25排除二进制；157 Python、70 shell、67文档。建立八任务数据→wrapper→loss→eval→保存格式/目录表、真实模型清单、原语复用表和M1—M9最小范围。全文索引与语义结论/可执行证据明确区分。
- 真正旧K0 41个、真正pre-M1核心6个、本轮post-M1新wrapper24个和其他现有模型4个，共**75/75同权重固定输入精确回放**。真实PyG单图及Car whole/Air whole+list、三个原cwd新进程加载有实际证据。没有历史训练checkpoint时构造的是临时模型参考，不能证明所有历史权重兼容。
- 完整现有套件实际运行**316 tests / 576.961s，0失败/错误，2个skip记录**；包含13个M1配置检查及旧loss/时间/参数/AMP/输出等回归。Air完整抽样epoch与周期图radius_graph因缺torch_cluster未执行；GUNet真实图链也未运行。70 shell `bash -n`通过。未修改已有测试。
- 与M1无已发现阻断冲突：Light/Full、显式CLI优先、结构/coverage分离、先读再校验、独立路径与实际接入边界匹配。M1仍仅配置/metadata，**没有MSAR模型或八任务训练支持**。后续Down须独立补mask/A；完整旧block/readout不可直接复用；三处AttnRes按w-only/无新增可训练评分scale处理，不能从M1的norm字段推导额外参数；pointwise_mlp不授权final点残差。
- 保留本轮快照 `/home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/source`，新小型二进制只在相邻`wrappers`/`other-models`；先前`msar-m1-before-n7y0eg1j`、K0/A阶段等产物保持。除了三个状态/记忆文档增量，其余起点文件字节不变；M1配置/测试及旧生产均未改。
- 本机Python3.13.9、torch2.13+cu130、PyG2.3.1、RTX5090 Laptop；真实小型GPU/AMP旧回归已执行。远端Python3.10/torch2.11/cu128、真实数据读取/训练/收敛/准确率未验证。无依赖安装、下载、commit/push或PR。

剩余依赖是后续被明确点名的数学/接入阶段及上述环境限制，不需要本轮改变确认架构。原有缺陷单列在审计报告，不借M0修复。**本M阶段结束，未执行下一阶段。**

## 2026-09-16：M1 完成，等待审查

当前授权仅 M1。独立 `family=msar_lno` 的配置、Light/Full profile、显式 CLI 解析协议、训练目标/运行记录、严格 metadata 与独立目录基础已完成。见 [M1 报告](MSAR_LNO_M1_CONFIGURATION.md)、[验证摘要](msar_lno_audit/m1/summary.json)、[夹具索引](msar_lno_audit/m1/fixture-index.json)。

- Light：M=[512,256,128,64]，d96；Full：M=[1024,512,256,128]，d192。heads 均[4,4,8,8]，encoder/decoder depths均[3,1,1,1]，2d/GELU/无卷积。
- 默认 coverage=floor/0.01/kappa0.2，off/0 示例已交付；目标字段与结构分离。eval先读保存结构，再核对显式覆盖；目标/运行差异不构成参数结构冲突，sidecar不改写。
- 新目录协议 `output/<task>/msar_lno/<light|full>/<coverage_floor|coverage_off>/<unique或显式save_name>`。只提供 helper，尚未接生产入口。
- 新13+旧19配置测试32/32通过（16.293s）；现有K0旧Transolver/CDLNO同权重41/41精确回放；修改前捕获的KCDNO all/off与matched LRSA point/conv core6/6精确回放。136个K0文件+6个新夹具hash不变。3cwd配置导入、Python3.10语法、10入口/helper参数声明检查通过。
- 保留快照 `/home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/source` 和其中相邻 `current-fixtures`；不覆盖旧K0/A阶段产物。614个既有文件中612字节不变，仅旧STATUS/memory追加本记录；生产模型/旧配置/入口/训练/数据/依赖/旧测试均未改。
- 用户确认M0通过，但当前工作区没有MSAR M0审计、原STATUS或夹具索引。已如实记录缺失并核查M1所需实际源码；本状态文件从M1开始，不补造M0完成记录。文本/AST inventory不是完整M0语义审计。
- 本机Python3.13.9/torch2.13+cu130/PyG2.3.1/RTX5090Laptop；旧Transolver有限GPU回放通过。远端Python3.10/torch2.11/cu128未执行、未安装依赖。MSAR数学/辅助损失/模型权重/参数量/合成训练/真实数据训练均未执行，不能由配置测试推断。

实际注册仅配置路由：`msar_lno → cdlno.msar_lno`，没有虚假Model/core占位类。原八任务factory/CLI保持原样，当前不能用生产训练命令启动MSAR。后续M2及所有后续阶段未执行；无commit/push/真实训练。

本M阶段结束，未执行下一阶段。
