# 补充阶段 A3：八任务三模式合成训练、评估与 checkpoint 验证

日期：2026-09-15。A2 已由用户审查通过，本轮仅执行用户新指定的 A3 验证范围。**八任务×三模式的 CPU 合成训练步、原损失连接、评估计算与 checkpoint 均通过；完整回归 184/184 通过。没有修改生产实现，没有运行真实数据训练。A4 未执行。**

## A. 范围与结果

复用既有 unittest、真实入口的 parser/constructor AST、任务 wrapper、checkpoint helper、原 normalizer/TestLoss 和工业 `train.py`。新增43项定向检查，包括24个任务模式格、6个5×7网格格、12个有限GPU格及1项禁止导入实验入口检查。没有另建训练框架、导入 `exp_*.py`/`main.py`、创建假数据文件或调用读取数据的 epoch 主程序。

本轮最新指令将 A3 明确为接口/训练步/checkpoint 验证，替代旧 A0/A2 文档对未来“A3性能工具”的建议分工。性能工具本轮完全冻结，不能把旧建议当作实现授权。

覆盖标记：**S**=静态/参数选择通过；**T**=合成 forward/loss/backward/optimizer step 通过；**L**=原损失连接通过；**E**=原 batch 评估计算通过；**K**=更新权重后保存/加载、冲突拒绝及 sidecar 不覆盖通过。所有格均为随机合成数据，E 不包含真实抽样、文件读取、绘图、气动力或边界后处理。

| 任务 | full（CPU） | no_sa（CPU） | identity（CPU） | 真实 PyG 对象（三模式） | 本轮新模式 GPU（三模式） |
|---|---|---|---|---|---|
| Darcy | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 不适用 | FP32 通过，5×5 |
| Elasticity | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 不适用 | 未执行 |
| Airfoil | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 不适用 | 未执行 |
| Pipe | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 不适用 | 未执行 |
| Navier–Stokes | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 不适用 | FP32 通过，64×64 |
| Plasticity | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 不适用 | FP32 通过，101×31 |
| ShapeNet-Car | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 通过 | FP32/PyG 通过，训练N19 |
| AirfRANS | S/T/L/E/K 通过 | S/T/L/E/K 通过 | S/T/L/E/K 通过 | 通过 | 未执行 |

每项实际尺寸、更新/前向次数、checkpoint误差记录在 [training-cases.json](front_ablation_audit/a3/training-cases.json)。CPU30格（24+6）和GPU12格共42格；另外1项是无实验入口导入检查。通过意味着在下述具体范围内可执行，不代表已验证全宽度任务性能、收敛或准确率。

## B. 修改文件、理由和 diff

实际工作区为main/`769fa333742f73c132868cf560bce5ec21529362`，原Transolver基线为`75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。A3前253份文件的源码与hash保存在 `/home/hwz/CDLNO-artifacts/front-a3-before-l9s4pe2c/source` 和 [start.json](front_ablation_audit/a3/start.json)。起点已有A1/A2和用户启动修改全部保留，不归入本轮diff。

| 文件 | 本轮变化及理由 |
|---|---|
| `tests/test_front_training.py` | 新增定向 unittest，复用A2 parser/build/run工具；执行六标准任务原batch AST、工业原train/test，检查实际更新、时间窗口、活历史、真实PyG、checkpoint和有限GPU |
| `tests/test_front_task_modes.py` | 仅让真实PyG导入成为可选，并在缺失时明确skip相关对象/新进程检查；防止没有PyG时所有标准任务在测试收集阶段就失败。不伪造PyG，不改原数学或加载断言 |
| `docs/CDLNO_FRONT_ABLATION_A3.md`、`docs/front_ablation_audit/a3/` | 本报告、命令/运行日志、实际执行的原batch片段、结果、冻结hash与增量patch |
| `docs/CDLNO_FRONT_ABLATION.md`、`docs/CDLNO_IMPLEMENTATION_STATUS.md`、`AGENTS.md`、`memory/current-state.md` | 增量记录A3当前结果、范围及下一阶段未执行；旧记录保留 |

可审查的增量 [a3-changes.patch](front_ablation_audit/a3/a3-changes.patch) 以A3开始时的真实源码为基准。生产模型、wrapper、配置实现/JSON/YAML、脚本、数据、训练/评估入口、checkpoint实现、依赖、性能工具 **0改动**。没有commit/push/reset。

## C. 计算图、任务合同与原损失对应

前段仍由 `cdlno/modules.py:LRSAFrontBlock.forward` 执行，`S,T:[B,M,d]`：

```text
full:     A=S+FFN1(N1(S)); B=A+SA(Nsa(A)); T=B+FFN2(N2(B))
no_sa:    A=S+FFN1(N1(S)); T=A+FFN2(N2(A))
identity: T=S
```

三模式随后都执行原Up、点残差、点FFN/ConvFFN。删除分支不保留专属norm或可训练参数。`cdlno/core.py:CDLNO.forward` 仍只在单次forward中收集pre-Up T；bridge/rear/CDPA/readout不变。默认L8/F2/P6，entry模式将两份T交给唯一CDPA位置。三种模式的实际T对象都被检查确实交给CDPA，并在原损失反传中收到有限、非零梯度；不要求零scorer下每一个参数首步梯度都非零。

核心的36格小模型（3种front × F0/F2/F6及L12F2 × off/entry/every_block）复用 `test_front_ablation.py`，包含独立公式、消融零化参考、删除参数、F0同权重等价、真实T、历史时序/隔离及chunk0/1/2/99输出/梯度比较；原 `test_core.py`、`test_cdpa.py` 继续核对原21格和扩展深度、CDPA两级softmax与独立来源。没有在所有大任务上扩展完整配置笛卡尔积。

| 任务与真实 wrapper | CPU合成训练尺寸（三模式均相同） | 原输入/输出与损失连接 |
|---|---|---|
| Darcy / `StaticStandardModel` | B1、85×85=7225 | x2、fx1；64维reference替换xy后拼fx，正式stem65。实际UnitTransformer encode/decode，TestLoss相对L2 + 0.1×原central_diff梯度损失；测试标签不encode |
| Elasticity / `StaticStandardModel` | B1、N972 | x2、fx=None、输出1；原y normalizer encode/decode + TestLoss；点域PlainFFN |
| Airfoil / `StaticStandardModel` | B1、221×51=11271 | 弯曲物理x2、不重排，fx=None、输出1；原TestLoss，无新增normalizer；dense ConvFFN |
| Pipe / `StaticStandardModel` | B1、129×129=16641 | 弯曲物理x2按原UnitTransformer归一化，fx=None、输出1；原y decode + TestLoss；dense ConvFFN |
| NS / `TemporalStandardModel` | B2、64×64=4096 | x2、fx `[B,N,10]`，每次输出 `[B,N,1]`；reference+10通道stem74；10个原TestLoss累加，训练真值回填、eval预测回填 |
| Plasticity / `TemporalStandardModel` | B2、101×31=3131 | 原x归一化，fx1、T `[B,1]`、每次输出4；原标签 `[B,N,4,20]`，原TestLoss逐时间点连接；20不并入N |
| Car / `models.CDLNO.Model` | B1图、N19训练；N7/29可变图；N32186单独layout forward | `forward((cfd_data,geom))→[N,4]`，x7、velocity3/pressure1。调用原train/test，loss=MSE(所有节点velocity3)+0.5×MSE(surf pressure) |
| AirfRANS / `AirfRANSModel` | B1图、N23训练；N7/29可变图；N32000单独layout forward | `forward(data)→[N,4]`，x7追加reference64，stem71；vx/vy/p/nut。调用原train/test默认MSE，独立核对surface/volume运算；不把可选weighted loss误当默认 |

所有训练格显式测试配置为d8/h2/M4、L8/F2、entry/chunk0；只降低测试实例宽度，不写回正式配置。正式Pipe仍M32，其余M64；任务d/h、epochs/batch/lr等由既有A2 preset静态回归保护。

另外Airfoil/Pipe各三模式以B2、5×7、N35完成同样原损失训练/eval/加载。原 `test_modules.test_conv_5_by_7_row_major_channel_mixing` 与A1独立前段公式检查确认row-major恢复、通道混合groups=1以及Up后的ConvFFN。没有为了5×7改写Darcy原只接受方格的梯度损失。

原语句执行边界：`training_nodes` 只去掉batch中将已构造张量搬到CUDA的tuple赋值，其余zero_grad、loss、backward、clip、step逐句使用源码AST；Elasticity原scheduler在batch外，合成单batch epoch后执行一次。`evaluation_nodes` 选择原forward/decode/TestLoss或原完整时间循环，排除绘图/文件输出。实际片段存入 [executed-standard-fragments.json](front_ablation_audit/a3/executed-standard-fragments.json)，可逐句与 `exp_darcy/elas/airfoil/pipe/ns/plas.py` 比较。这不是导入或完整运行原实验脚本。

时间语义实测：

- **NS**：每个模式训练10 forward，检查每步窗口等于“原始剩余帧+已出现真值”；1 backward/optimizer step/OneCycleLR step。eval10 forward，每步窗口等于“原始剩余帧+此前预测”，最终输出 `[2,4096,10]`。20次调用各自执行bridge并产生新的T，无真实时间latent缓存。
- **Plasticity**：每个模式训练20 forward、20 backward/optimizer step、1 OneCycleLR step；eval20独立时间条件forward，输出 `[2,3131,4,20]`。B>1不同T会改变输出，T及time_fc梯度有限且非零；fx没有预测反馈。训练与eval的T对象不复用。

静态标准任务用原AdamW，Elasticity原CosineAnnealingLR，其余原OneCycleLR；标准任务构造语句直接取源码。工业任务用原默认Adam/.001与OneCycleLR（Car保留final_div_factor1000），用一个合成batch执行原train函数，实际权重变化、optimizer state创建和scheduler次数均断言通过。合成数据量改变scheduler的steps总数是测试规模差异，不是正式训练参数修改。

工业图检查使用安装的真实PyG `Data/Batch/DataLoader`，没有假模块。单图全零batch合法；多图明确报错；改y或移除y不改eval输出；输入字段值和storage指针未被原地改变。AirfRANS额外按原属性切片方式构造乱序抽样图（29取7，保留原ptr `[0,29]`），与等价普通单图输出完全一致。这只验证抽样后wrapper合同，没有执行原完整抽样/radius_graph/scatter平均/物理边界指标流水线。

## D. 实际命令、环境与结果

本地只读环境：[environment.json](front_ablation_audit/a3/environment.json)。Python3.13.9，torch2.13.0+cu130（CUDA build13.0），真实PyG2.3.1，NVIDIA GeForce RTX5090 Laptop GPU。没有安装/替换依赖。用户远端Python3.10/torch2.11/cu128仍是兼容目标，本轮没有访问或验证该远端环境；两个测试文件的Python3.10语法检查通过不等于该环境动态运行通过。

新检查固定随机种子915、CPU单线程、FP32、PyTorch SDPA math backend，GPU关闭TF32后恢复设置；未开启AMP/compile，没有做性能计时。GPU的12格是实际forward、原损失、反传、更新、eval和加载，不从CPU推断GPU结果；也不声称验证了Flash/其他SDPA backend、新模式AMP或全八任务GPU矩阵。

```bash
# 在仓库根目录；仅运行合成验证
CDLNO_A3_EVIDENCE=docs/front_ablation_audit/a3 \
python -B -m unittest discover -s tests -p test_front_training.py -v

# 本地完整约定套件，包含已有LRSA参考/核心/任务/静态/checkpoint检查
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator \
CDLNO_A3_EVIDENCE=docs/front_ablation_audit/a3 \
python -B -m unittest discover -s tests -p 'test_*.py' -v

# 只读源码冻结核查，输出审查证据
python -B docs/front_ablation_audit/a3/audit_scope.py
```

命令实际执行时stdout/stderr分别保存到 [targeted-initial.log](front_ablation_audit/a3/targeted-initial.log)、[final-regression.log](front_ablation_audit/a3/final-regression.log)、[targeted-final.log](front_ablation_audit/a3/targeted-final.log)。首次43/43通过（14.078s）；完整套件184/184通过（75.836s），0失败/错误/跳过。随后只修正结果JSON中的real_N标签：NS/Plasticity即使选小实例factory，实际仍保持4096/3131；再次运行定向43项通过（13.469s）。没有修改验证容差、删除断言或修生产代码来通过测试。

新checkpoint比较为atol=1e-5、rtol=3e-4（沿用A2）；chunk0保存、chunk1加载比较的最大绝对误差为6.146728992462158e-8。同模式eval先读已有sidecar恢复省略mode；另两种显式模式在torch.load前拒绝；sidecar前后字节一致。CPU工业格式仍Car整对象、Air整对象成员+对象列表；标准任务仍严格state_dict。原A2错误架构/缺权重/残缺旧配置/工业对象实际mode冲突与目录隔离测试也在184项中实际通过。

从三个原项目cwd，新进程实际运行A2 `test_three_original_workdirs_new_process_import_and_all_mode_loads` 的全部24格并通过；覆盖共享包导入、原稳定类路径、标准state_dict及工业整对象/list加载。新增42个更新后checkpoint格均通过，Air还检查单成员路径。

真正修改前full基准没有重新生成；本轮用原A1 verifier重新回放其hash、参数键/初始化、输出/梯度、旧缺属性整对象：

```bash
# 按group切换到三个原项目目录；core/airfrans也在各自原cwd执行
PYTHONPATH=.:/home/hwz/CDLNO python -B \
 /home/hwz/CDLNO/docs/front_ablation_audit/a1/verify_before.py \
 /home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0 car \
 --results /home/hwz/CDLNO/docs/front_ablation_audit/a3/full-car.json
```

实际三次调用对应：`core`在PDE-Solving-StandardBenchmark，`car`在Car-Design-ShapeNetCar，`airfrans`在Airfoil-Design-AirfRANS；结果分别是 [full-core.json](front_ablation_audit/a3/full-core.json) 12份、[full-car.json](front_ablation_audit/a3/full-car.json) 2份、[full-airfrans.json](front_ablation_audit/a3/full-airfrans.json) 2份，均原始atol=rtol=0、输出/梯度最大误差0。包括18-slot旧config、旧front无mode属性、工业对象列表与旧sidecar不改写。

远端可运行相同的有限无数据验证（本轮没有执行以下命令）：

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w
CDLNO_A3_EVIDENCE=/tmp/cdlno-a3-remote-evidence \
python -B -m unittest discover -s tests -p test_front_training.py -v

# 再检查入口/加载兼容；真实旧fixture不在远端时对应一项会明确skip
python -B -m unittest discover -s tests -p test_front_task_modes.py -v
```

若远端缺GPU或PyG，相应测试明确skip；标准CPU格继续运行。旧fixture要通过A2旧对象专项验收需另提供原样保留的 `CDLNO_A1_BASELINE` 路径，不能用新模型伪造旧checkpoint。本地本轮这些条件均具备，没有skip。局部缺依赖时的skip是代码路径说明，未声称本轮另建了“无PyG环境”实测。

## E. 冻结区域及原有问题

[freeze.json](front_ablation_audit/a3/freeze.json) 对比起点253份文件，确认128份生产/脚本/工具相关文件逐字节不变：共享核心/模块/融合/配置/checkpoint、三任务项目、数据/采样/normalizer/loss/时间循环/指标/旧Transolver、JSON/YAML、根及子项目脚本、依赖和性能工具。A1/A2原有修改与用户预先修改的hash同样被保留，不能将总git diff归为A3。

完整套件还实际执行原阶段AST投影和git基线冻结检查，验证原入口除先前授权CDLNO分支外保持原Transolver协议。原阶段的修复与现存问题继续分别记录：

- Car原外层日志汇总将train/test返回的pressure/velocity名称互换；内部backward损失正确。本轮直接调用内部函数检查原损失，未修外层冻结代码。
- Car原完整drag评价的固定raw路径与fold0限制仍在；未用合成指标声称完整物理评价通过。
- AirfRANS原 `test` 的 `if criterion == 'MSE' or 'MSE_weighted'` 对MAE选择有问题；当前默认MSE不受影响。本轮只检验当前默认，未顺手修改该旧分支。
- 原标准入口旧Transolver的若干宽松加载写法仍是旧行为；新增CDLNO路径保持strict检查，未扩大加载权限。

没有发现此次A1/A2模式切换造成的新增模型/训练协议回归；A3无生产缺陷修复。

## F. 自审与尚未验证

交付前已自审四个重点，证据均在实际测试中：

1. **是否只验证backward而遗漏更新**：实际optimizer state已创建、stem权重已改变，所有格记录step次数；工业梯度与独立原mask公式比较。
2. **时间语义与活历史**：原时间循环AST运行，NS窗口逐步值检查、Plasticity20独立更新/T梯度检查；每次新bridge/T、真实CDPA消费对象、训练梯度及train/eval对象隔离通过；核心every时序另由原测试验证。
3. **加载证据是否来自真实旧对象/是否覆盖sidecar**：真实pre-A1的16份fixture零容差重放；新训练后checkpoint通过原加载器；省略/冲突模式、严格权重、三个cwd新进程和字节保护通过。
4. **是否借测试改变数据/点序/正式预设**：128文件冻结hash、原AST回归、非方格/真实N、PyG输入指针/值和多图负向通过。只有测试与文档变化。

剩余限制：远端torch2.11/cu128动态兼容、新模式AMP/其它SDPA backend、其余四任务新模式GPU、正式宽度/批量显存、真实数据完整读取/抽样/全指标、真实轨迹训练、收敛、准确率和真实epoch效率均未验证。本轮测试耗时是测试执行时长，不是训练epoch性能。无真实数据下载、实际数据集训练或自动扫描。性能工具新模式接入继续等待单独授权。

本轮没有剩余已知实现缺陷需要用户裁定。**本补充阶段结束，未执行下一阶段。**
