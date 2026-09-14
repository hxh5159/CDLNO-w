# CDLNO 最终实施与综合审查报告

日期：2026-09-14；阶段10，等待用户审查。基线分支 `main`，commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`，origin `git@github.com:thuml/Transolver.git`。阶段0没有已跟踪代码修改；阶段0–9的未提交成果是本阶段必须保留的工作，不是应恢复的上游差异。

**结论：在已确认设计和无数据验收范围内，CDLNO架构、八任务接口、checkpoint与性能工具按计划实现。最终119项测试实际全部通过，无失败/错误/跳过；本轮没有发现需要修改生产模型或训练代码的缺陷。发现并补齐了时间任务的测试证据缺口。真实数据读取完整性、收敛、精度、真实epoch效率及远端目标环境仍未验证。**

逐项追溯见 [v1.2 §§0/8/13/14矩阵](CDLNO_REQUIREMENTS_MATRIX.md)，每行列要求、实际代码符号、测试证据和状态。矩阵中理论未复验/可选全配置GPU未运行不会被总测试通过数掩盖。

## A. 完成范围

本阶段从生产源码与测试断言反查需求：重新读核心、模块、CDPA、三套wrapper和checkpoint帮助代码，逐个读六exp及两工业训练/评价入口的diff；核对LRSA真实固定checkout，重新只读获取IPOT固定commit的3份源码与LICENSE；核对三方归属、配置、训练脚本与原基线。没有只引用旧报告作验收。

交付包括README的环境/八任务训练评价示例、消融/checkpoint与性能使用说明；需求矩阵；来源许可说明；补充八任务位置/时间/checkpoint清单；本报告；持久日志、快照、冻结清单和可审查diff。更新AGENTS、STATUS和memory。没有修改依赖、下载数据、创建假数据文件、导入会读取数据的exp/main入口、执行真实训练/抽样评价、commit/push/PR/reset。

名称映射：旧包 `cdpa_operator` → **`cdlno`**；旧核心 `CDPAOperator` → **`CDLNO`**；旧整网注册名 `CDPA` → **`CDLNO — Cross-Depth Latent Neural Operator`**。机制仍为 **CDPA — Cross-Depth Physics Attention**，`cdpa_mode` 等字段保持。原入口wrapper可导出 `Model`；Car稳定类为 `models.CDLNO.Model`，Air为 `cdlno.airfrans.AirfRANSModel`（本地模块导出别名），避免整模型加载歧义。

## B. 修改文件、原因和diff

| 本阶段文件 | 修改与理由 |
|---|---|
| [tests/test_temporal_standard.py](../tests/test_temporal_standard.py) | NS原测试只有3次调用，改成每种回填各10次并逐步检查窗口、累计loss后一次backward/step；Plasticity同批样本改为不同T并检查time_fc各参数梯度；新增两时间exp完整AST投影比较。只是补验收，不修改训练实现 |
| [README.md](../README.md) | 前置CDLNO使用说明，保留原README全文及结果/引用；区分远端命令示例与本次实际执行 |
| [CDLNO_REQUIREMENTS_MATRIX.md](CDLNO_REQUIREMENTS_MATRIX.md) | §§0、8、13、14逐项追溯，不遗漏未运行/理论边界 |
| [CDLNO_THIRD_PARTY_NOTICES.md](CDLNO_THIRD_PARTY_NOTICES.md) | 三仓固定版本/文件指纹、许可归属和实际源码差异；保留IPOT MIT全文，不推断LRSA许可 |
| [CDLNO_TASK_LAUNCHERS.md](CDLNO_TASK_LAUNCHERS.md) | 核对8配置/16脚本，补齐位置/时间/FFN/checkpoint表 |
| 本报告、[STATUS](CDLNO_IMPLEMENTATION_STATUS.md)、[AGENTS](../AGENTS.md)、[memory](../memory/current-state.md) | 阶段9已批准；阶段10已完成待审查；纠正过强的历史时间验收表述，记录限制 |
| [final_audit/](final_audit/) | 起点快照、来源与环境JSON、实际测试日志、冻结结果及两份patch |

[phase10-changes.patch](final_audit/phase10-changes.patch) 供审查本阶段已有文件变化和新增交付文档；原内容均对阶段10起点SHA核验后生成，未使用git add/commit。该patch不重复嵌入日志/生成patch本身。

[implementation-vs-baseline.patch](final_audit/implementation-vs-baseline.patch) 汇集相对阶段0基线的12个原代码/YAML有限分支，以及新共享包、wrapper、配置、脚本、测试和工具；可检查整个实现而不只看本轮小diff。README/文档在本阶段patch中。原baseline源码累计12文件 +435/−204，本轮没有再改这些文件。

**本轮修复的是验收缺口**：旧NS测试运行3步，不应说成10步实测；旧时间AST测试仅查关键字符串，不足以证明整入口冻结。现在NS两路完整10步与NS/Plasticity整AST比较实际通过。保留历史日志，最终结果以本报告为准。

## C. 公式、张量和代码对应

| 计算阶段 | 确认公式/顺序 | 实际实现与形状 |
|---|---|---|
| stem | 原任务位置/字段/时间语义提升 | wrappers生成 `H0[B,N,d]`；core不加新位置/时间编码 |
| 完整前段 | down → `z+=FFN1(RMS(z))` → `z+=SA(RMS(z))` → `T=z+FFN2(RMS(z))` → up/点残差/点FFN | [LRSAFrontBlock.forward](../cdlno/modules.py)：H为[B,N,d]，T为[B,M,d]；显式返回(Hnext,T)，T在up前，down无query残差 |
| bridge | `Z0=Q0+Cross(LN(Q0),LN(HF))` | [IPOTBridge](../cdlno/modules.py)：独立learned Q0[M,d]，输出[B,M,d]；不注册encoder_ff |
| 单历史对齐 | `Q=LNq(Z)WQ`，`Ks/Vs=LNkv(Ts)WK/WV`；`Rs=ConcatHeads(softmax_M(QKsᵀ/√dh)Vs)WO+bO` | [CDPA.forward](../cdlno/cdpa.py)：每源M keys；同位置共享norm/K/V/O；Q一次。折叠[B,k,M,d]→[B*k,h,M,dh]，无S*M softmax |
| depth融合 | `R0=Z`；`es[b,m]=wᵀRMSdepth(Rs[b,m])`；`α=softmax_sources(e)`；`Zf=Σs αs RAW(Rs)` | [CDPA._depth_fusion](../cdlno/cdpa.py)：raw[B,M,S+1,d]，α[B,M,S+1]；FP32均方/评分/softmax/累加、autocast关闭，末尾转回Z.dtype；不detach |
| 后段 | `A=Z+SA(LN(Z))`，`Znext=A+GEGLU(LN(A))` | [PersistentLatentBlock](../cdlno/modules.py)：Q/K/V来自同一LN(Z)；P个独立实例，均[B,M,d] |
| 最终读出 | `HD=HF+Up(RMS(HF),RMS(ZP))`；`Hout=HD+PointFFN(RMS(HD))`；`y=Linear(LNout(Hout))` | [LRSAFeatureReadout](../cdlno/modules.py)：HF是query与点残差；输出[B,N,Cout]，工业去掉仅有的batch轴 |
| 规则点FFN | 原索引reshape → dense3×3Conv → 内层LN → Linear无bias → GELU → Linear有bias → flatten | [ConvFFN](../cdlno/modules.py)：groups=1，N=H×W；五结构任务保留原索引，非结构三任务用plain point FFN |

norm/bias/初始化再核对：LRSA前段/读出外层RMS及独立per-head Q/K RMS，eps1e-6、scale1/no bias；bridge/后段/CDPA分支是affine LN，eps1e-6且无QK norm；输出LN、ConvFFN内部LN含bias。Q/K/V投影无bias，O有bias；LRSA down没有Q投影。plain FFN两个Linear有bias、ratio2；后段GEGLU ratio2。普通Linear仅一次trunc_normal(std=.02)/bias0；dense Conv保持原生初始化；down query矩阵orthogonal、bridge query normal(.02)、depth w严格0。wrapper不递归apply重置核心；placeholder只在真实fx=None路径存在，time_fc只在Plasticity时间路径存在。

零w时输出为候选均值，默认两历史是 `(Z+R1+R2)/3`；不是identity，不另加Z/gate/当前来源偏置。depth scale首步梯度可以为0，更新w后路径梯度已验证。历史浮点dtype不同则在Cross前转Z.dtype并保持梯度；FP32 depth没有被外层AMP再次降精度。

### 三种模式、历史时序与计数

`0≤F<L`，P仅由L−F派生；全阶段M一致，F不硬限6。`off`不收集T；`entry`只在第一个后段前读T1…TF，F0完全不建CDPA参数，且同权重与off计算完全一致。`every_block`第j个后段前当前为Z(j−1)，历史为 `[T1…TF,Z0…Z(j−2)]`，raw bridge Z0保留。每次forward局部重建；仅把完整后段输出作为后续历史，不记录融合中间值/SA/FFN增量，不跨真实时间缓存，不detach。不同活跃CDPA位置参数/storage独立，无跨位置投影K/V缓存。

| 默认L8/F2/P6 | down/bridge | up/readout | latent SA | 结构ConvFFN | 逻辑历史来源总数 | chunk0历史SDPA | 全模型SDPA |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 3 | 3 | 8 | 3 | 0 | 0 | 14 |
| entry | 3 | 3 | 8 | 3 | 2 | 1 | 15 |
| every_block | 3 | 3 | 8 | 3 | 27 | 6 | 20 |

以上为真实forward spies验证，不只公式估计。entry逻辑总F，every总PF+P(P−1)/2；chunk1每份来源一次，chunkk为各位置ceil(s/k)之和。chunk减少调用不减少数学来源/MAC。正常core只返回Tensor，不无条件保存注意力矩阵或调试梯度/CPU统计。

### 八任务合同与接入

[完整8配置/16脚本清单](CDLNO_TASK_LAUNCHERS.md) 列出具体路径；[README](../README.md) 包含逐任务训练/评价命令。下表所有任务L8/F2/entry/ratios2，默认M按任务值而非旧parser默认。

| 任务 | N/输出 | stem与条件 | d/h/M；epoch/batch | 关键保持项 |
|---|---|---|---|---|
| Darcy | 85×85；[B,N,1] | reference64替换xy+fx1=65；无时间 | 128/8/64；500/4 | 原normalizer encode/decode、边界和梯度loss |
| Elasticity | 972；[B,N,1] | xy2+active placeholder；point FFN | 128/8/64；500/1 | 原不规则点序/应力目标 |
| Airfoil | 221×51；[B,N,1] | 原xy2+placeholder；ConvFFN | 128/4/64；500/4 | 弯曲坐标和索引顺序，不重新采样/排序 |
| Pipe | 129×129；[B,N,1] | 原exp已归一化xy2+placeholder；ConvFFN | 128/4/32；500/8 | 原坐标normalizer/输出decode |
| NS | 64×64；单次[B,N,1] | reference64+fx10=74，无time_fc | 256/8/64；500/2 | 10→10；训练真值回填、10loss一次更新；测试预测回填 |
| Plasticity | 101×31；单次[B,N,4] | xy2+fx1=3，T[B,1]与原time embedding | 128/8/64；500/8 | 空间/通道/时间标签轴不变；每batch20次独立更新，无反馈 |
| ShapeNet-Car | 单图可变N；[N,4] | x7，placeholder；不读geom/y，无time | 256/8/64；200/1 | velocity3/pressure1；surf及速度/压力loss；一次run一fold |
| AirfRANS | 单图可变N；[N,4] | x7追加reference64=71，域[-2,4]×[-1.5,1.5] | 256/8/64；398/1 | vx/vy/p/nut；抽样、图、scatter平均、mask和边界后处理 |

标准用 `run/model.pt` state_dict/weights_only/strict=True；Car保留 `model_<epochs>.pth` 整模型；Air保留成员整模型及run根模型列表 `CDLNO`。eval先load sidecar、比较架构与wrapper/任务合同，再load权重；错误对象、键或形状拒绝。chunk/device/dtype属运行项不当成架构冲突；工业加载后应用请求chunk。读取过程不更新sidecar，结果目录隔离，已有训练目录拒绝复用，无新resume。标准checkpoint不额外强制学习率/epoch相同；工业按其run合同校验fold/task/nmodel/预算/采样参数，不能笼统声称三者完全一样。

## D. 实际执行与通过/失败/未运行

本次环境：Python **3.13.9**，torch **2.13.0+cu130**，CUDA runtime **13.0**，真实PyG **2.3.1**，numpy2.2.6、scipy1.16.3、einops0.8.2、timm1.0.28、PyYAML6.0.3；GPU **RTX5090 Laptop**、driver591.86、24463MiB、支持BF16。详情见 [environment.json](final_audit/environment.json)。本地没有pyg-lib/torch-cluster，不影响已经运行的无邻居构图合成检查；真实邻居图/读取流程不据此算通过。未安装/更改任何依赖。

目标环境仍为用户已工作的Python3.10/torch2.11/CUDA12.8；Python3.11候选，最终远端PyG版本未查询。当前本地Python不满足pyproject目标版本范围，使用源码PYTHONPATH进行验收；本次没有执行editable安装，不用本地成功替代目标环境验证。

实际命令（在仓库根执行）：

```bash
# 在发现测试覆盖缺口前：118项通过，58.082秒，保留原日志。
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
# 补齐NS完整十步、Plasticity不同T、整时间入口AST后的定向检查：9项通过，10.114秒。
python -B -m unittest discover -s tests -p 'test_temporal_standard.py' -v
# 因测试内容有具体变更，重新完成一次最终套件：119项通过，49.871秒。
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
# 静态检查与基线取证：
git rev-parse HEAD
git status --short
git remote -v
git ls-tree -r --name-only HEAD
git diff --check
git diff --stat
```

原始输出：[最终119项](final_audit/regression.log)、[时间定向9项](final_audit/temporal-focused.log)、[修改测试前118项](final_audit/regression-before-temporal-coverage.log)。此外执行只读Python采集环境/逐文件SHA256与AST语法、Bash语法及文档链接检查，结果见 [delivery-checks.json](final_audit/delivery-checks.json)、[freeze-check.json](final_audit/freeze-check.json)。IPOT通过固定URL的urllib只读获取；URL/字节/SHA保存在来源JSON，无数据下载。

| 测试组 | 数量 | 实际验证 |
|---|---:|---|
| modules | 18 | norm/SDPA/FFN/GEGLU/ConvFFN、完整前段/bridge/rear/readout、bias/init/独立参数、5×7；CPU及GPU |
| 原LRSA同配置参考 | 2 | 原checkout实际类、同权重FP64 output/T/全参数与输入梯度；本轮打印最大差均0 |
| CDPA数学 | 21 | 独立显式reference、两softmax轴、RAW/O+b/恒等R0、置换/batch/chunk、极端值/FP32子图、GPU |
| core | 13 | point/grid各21种L8 F0..6×三模式；扩展12/2、16/6、1/0、8/7、10/8；历史时序/梯度/计数/同权重F0等价 |
| core/config/checkpoint | 6 | 非法值/派生P/往返、sidecar先读、不覆盖、三cwd新进程导入、strict加载 |
| 四静态任务 | 13 | 实际接口和原N减小宽度/latent、Darcy原loss连接、checkpoint及四exp全AST；GPUFP32/FP16AMP/BF16AMP |
| 两时间任务 | 9 | NS每路完整10步、训练一次更新；Plasticity不同T/时间参数梯度/20更新；整时间exp AST、checkpoint |
| ShapeNet-Car | 13 | 真实PyG Data/Batch、可变N/多图拒绝、无y/geom泄漏/无mutation、原mask损失、独立cwd整模型加载/冻结 |
| AirfRANS | 13 | 真实PyG、stale单图ptr/多图拒绝、32000小宽度、原loss、独立cwd成员整模型/模型列表加载、冻结 |
| 性能工具 | 11 | matched LRSA完整L块、全投影/Conv/GEGLU/CDPA成本、来源/chunk、八task factory、合成计时与优化器state |
| 合计 | **119** | **全部通过；0失败/错误/跳过** |

CPU CDPA reference为48个dtype/S/w/chunk子case，比较输出与所有参数/输入梯度，atol3e-6/rtol3e-5；本轮最大绝对梯度差2.17e-5在联合相对/绝对容差内。depth即使输入FP64仍按协议FP32，不能要求FP64最终逐位精度。LRSA FP64输出容差1e-10/1e-8、梯度2e-9/2e-7；GPU CDPA FP32 math/reference/fused为5e-6/1e-4。半精度测试使用各自容差/有限性检查，不把FP16称为FP32精度。

日志的 `ImportError: No module named xformers/liger_kernel` 是外部LRSA可选内核导入的原fallback输出；其真实参考测试随后成功，没有伪造模块或skip。CPU BF16模块测试局部关闭MKLDNN规避既有oneDNN backward限制，GPU精度测试局部禁用TF32并恢复；这些测试设置没有写进生产默认。

**实际GPU范围**：最终套件中6个GPU测试方法实际执行，覆盖基础模块、CDPA FP32与FP16/BF16（含直接半精度/AMP/FP32 depth）、三模式point/grid小核心、四静态wrapper；不等于八任务完整默认规模GPU训练。两工业真PyG接口/原损失本轮为CPU；阶段7/8另有已报告的三模式GPU FP32检查，本轮未追加重复工业GPU扫描。阶段9持久27行合成GPU性能结果保留，本轮只回归工具功能/成本，不重做性能扫描。

**失败项**：本阶段上述套件无失败，无待修复生产缺陷。测试覆盖不足已按B项补齐，不将旧118项成功当作此前已完成完整十步的证据。

**未运行**：远端Python3.10/3.11+torch2.11/cu128上的CDLNO与editable安装；真实文件解析/字段覆盖/manifest/VTK/完整采样图构建；真实10帧轨迹与Plasticity实际标签训练；工业全评价/重复采样/散射平均/边界与系数指标；训练收敛、精度或种子重复实验；真实epoch/数据加载效率；八任务全默认GPU矩阵及大扫描。阶段9AMP/TF32开启/强制backend/compile的整套性能对照未运行。数学附件未定位，历史8项理论检查及PDF编译未复现。

## E. 冻结区域与原有问题

[冻结清单](final_audit/freeze-check.json) 对阶段0 commit全部71个跟踪文件比较：**58字节不变**；另12个是此前已批准的模型接入文件（本阶段字节不变），README仅新增前置说明、原全文完整保留。所有原Transolver模型、Physics-Attention副本、原脚本、dataset、train、normalizer/metrics和requirements保持原字节。六exp、Car两个main、Air两个main的完整AST，经仅消去已授权模型分支/路径别名/兼容项后等于基线；Air原YAML所有key及其前缀保留，CDLNO仅新增key，仍398 epochs。

阶段10起点快照 [phase10-start.json](final_audit/phase10-start.json) 记录162个既有文件，SHA256 `aaeefa892951c977ad1336392964a11eca1562fe22749d0ba15a6f99e03a2085`；最终156不变，仅B表列出的6个既有文件变化，无意外变化/删除。新增文档/证据单列；既有PLAN、阶段报告、性能原始数据及全部生产代码保持。没有恢复、覆盖或丢弃先前未提交修改。

原入口保留的已知边界与本次回归区别：

| 项目 | 来源/处理 | 是否本次新回归 |
|---|---|---|
| Car显式epoch曾由float产生model_200.0.pth路径 | 上游类型问题；阶段7授权最小改int，与原保存model_200.pth一致 | 否，已修复的有限兼容项 |
| 工业整模型在新torch的weights_only默认变化 | 阶段7/8仅对应受信任本地load加weights_only=False/map_location，原保存协议不变 | 否，无全局放开 |
| 原Transolver `.cuda()`及未注册位置tensor | 新wrapper按输入device/buffer保持数值语义；原模型本身未改 | 继承限制；真实旧入口CPU不作承诺 |
| 部分标准eval strict=False；Pipe旧resave | 旧分支保持；新模型strict+sidecar且eval不resave | 继承行为，不顺手重构 |
| 原绘图硬编码85×85/221×51/129×129等 | 冻结；改变下采样后可视化不新承诺 | 继承限制 |
| Air抽样时保留原单图ptr，图构造/散射路径复杂 | wrapper兼容有效单图stale ptr且拒绝多图；原图/抽样代码不动 | 接口合成通过，真实流程未运行 |
| Air原Transolver未使用mlp_new参数 | 原模型及性能参数统计保留，工具列missing-grad | 继承事实，不制造删图/删参收益 |

当前源码与确认设计没有实质冲突。LRSA/IPOT源码差异按v1.2明确处理（完整双FFN、dense卷积、后段独立且QKV同LN、bridge无闲置FFN）；与参考原默认配置不同不等于本实现违反计划。目标环境与本地环境不同是验证边界，没有通过改依赖消除它。

## F. 交付前自审与剩余限制

五项自审均已亲自对照源码和最终证据完成：完整block/norm/bias/init；CDPA两softmax/RAW/FP32/chunk；every历史对象/同权重F0/参数独立；八wrapper提升与checkpoint拒绝/不覆盖；基线冻结与计数/性能口径。通过依据均落在需求矩阵及最终日志中，没有把未审查的清单留给用户。

没有仍待修复的已知生产实现缺陷。仍需明确保留三类不确定性：

1. **实验有效性**：尚无真实数据端到端、收敛/预测精度、真实epoch速度或远端新模型运行结果。合成成功不升级这些状态。
2. **理论材料**：独立数学附件路径仍未定位；本轮只核对可读的v1.2/导出摘要与代码，不声称复核了附件全文或新证明，不因此增加结构或损失。
3. **参考许可**：Transolver根MIT与IPOT MIT已核实，Air原ODbL保留；LRSA快照没有找到明确许可，不声称获得整份源码的MIT授权。本工程未vendor其源码，按确认公式实现，并在外部checkout做同配置参考测试。

效率报告保留阶段9的反例：Elasticity局部GPU样本中entry虽较少MAC，合成训练步仍可能更慢；Airfoil原任务比较头数不同，不能作为严格同配置优势。`lrsa_matched`只用于结构性能比较，不是论文复现实验，未注册新训练模型。没有从forward估算epoch或按理想复杂度宣称稳定加速。

**本阶段结束，未执行下一阶段**。
