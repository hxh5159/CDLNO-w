# 八任务周期可视化、权重归档与断点续训修改计划

最新进度（2026-09-15）：用户已批准按计划修改。遵守逐阶段要求，本轮仅完成V1公共完整归档/恢复/RNG隔离/绘图基础，见[V1报告](CDLNO_VISUALIZATION_RESUME_V1.md)。201/201回归通过，模型与任务源码冻结。V2–V5未执行，正式任务仍未接入新的周期保存/绘图/`--resume`；不能把V1基础能力当成八任务功能完成。下文保留原审批计划及当时状态。

日期：2026-09-15。**本轮只读论文与源码、编写计划；以下功能尚未实现。** 当前基线为用户已有的CDLNO+A1/A2/A3工作区，HEAD `769fa333742f73c132868cf560bce5ec21529362`。A4没有执行。本计划是新的训练运维需求，不自动推进任何既有补充阶段。

## 1. 请求解释及拟定范围

按“完成的epoch数”计数，而不是源码中从0开始的`ep`：第50、100、150…轮结束后可视化；指定四任务第100、200、300…轮结束后保存完整续训checkpoint及独立模型权重。所有任务在**实际训练完成**时保存最终checkpoint和权重；若最终轮同时命中周期，只保存同一状态一次。建议同时补一次最终可视化，使398轮等非50倍数的最终模型也有对应图。

| 数据集 | 当前正式总epoch默认 | 可视化 | 完整续训checkpoint + 独立权重 |
|---|---:|---|---|
| Navier–Stokes | 500 | 每50轮，结束补图 | 每100轮 + 实际结束 |
| ShapeNet-Car | 200 | 每50轮，结束补图 | 每100轮 + 实际结束 |
| AirfRANS | 398 | 每50轮，结束补图（含398） | 100/200/300 + 398 |
| Airfoil（PDE benchmark） | 500 | 每50轮，结束补图 | 每100轮 + 实际结束 |
| Darcy | 500 | 每50轮，结束补图 | 仅实际结束 |
| Elasticity | 500 | 每50轮，结束补图 | 仅实际结束 |
| Pipe | 500 | 每50轮，结束补图 | 仅实际结束 |
| Plasticity | 500 | 每50轮，结束补图 | 仅实际结束 |

总epoch以本次运行解析后的参数为准，不强制改成500。`--epochs`/`--nb_epochs`原有覆盖能力保留；周期配置属于运行/记录配置，不加入模型architecture字段。默认对所有CDLNO front模式、CDPA模式与合法F/L一致生效，模型数学与权重结构完全不改。原Transolver分支及其保存/评价行为保留。

按用户给定的保存频率，前四任务可恢复到最近一次100轮归档；**后四任务不产生中途续训点**。没有保存的epoch无法凭可视化图片恢复。本计划不偷偷改成每轮保存，也不新增信号抢救/任意batch中断恢复。

## 2. 已阅读论文与可借鉴部分

本轮获取五篇论文arXiv HTML，阅读可视化相关正文/附录与图注，并实际查看下列选定图片；不是重新完整审查其数学或复现实验。链接固定到实际读取版本，来源/hash及图注在 [paper_sources.json](viz_resume_audit/paper_sources.json)。首次访问GitHub raw超时，随后通过GitHub API读取官方README确定论文地址，arXiv HTML及选定图片读取成功。没有下载数据或参考权重，没有复制论文代码/图像进模型。

| 来源 | 本轮重点 | 拟借鉴到CDLNO的内容 |
|---|---|---|
| [Transolver v2](https://arxiv.org/html/2402.02366v2#S4.F7)，Fig.7、附录D.2 Fig.18 | 场预测及误差；Plasticity/Airfoil/Pipe/NS对照；Car表面与体积区域分开 | 真值/预测/误差并排；保留真实几何；重点展示翼型激波、尾流、弯曲管道、变形和涡旋 |
| [Transolver++ v2](https://arxiv.org/html/2502.02414v2#A3.F16)，Fig.6、附录C.2 Fig.14–16 | 复杂工业几何及Elasticity/Airfoil误差 | 固定视角、统一色标、边界局部放大，不以颜色范围变化伪装进步 |
| [Transolver-3 v2](https://arxiv.org/html/2602.04940v2#S4.F7)，Fig.7、附录C.2 Fig.13/15 | 表面pressure与volume velocity分开；侧视/底视流场切片 | Car固定表面视角和固定体积剖面，保留物理区域区分；不引入其缓存/采样训练/结构修改 |
| [LRSA v1](https://arxiv.org/html/2604.03582v1#S3.F3)，Fig.3 | 规则网格、点云、弯曲网格四任务；每任务共享误差色标 | 一致的真值/预测色标、独立误差色标、单样本relative-L2标注；保留972点点云与非方格 |
| [LinearNO v3](https://arxiv.org/html/2511.06294v3#Sx14.F9)，Fig.4、附录G Fig.9/10 | AirfRANS压力场、各任务误差；另有Q/K权重图 | 压力与速度分通道、局部流场细节；只借鉴输出可视化，不把LinearNO切片权重解释成CDLNO已有切片 |

实际查看了Transolver Fig.18、Transolver++ Fig.16、Transolver-3 Fig.7、LRSA Fig.3、LinearNO Fig.9。LRSA本地参考的 `scripts/airfrans/plot_error.py` 另提供了GT/Pred共享clim、独立误差clim、速度分量/模长及局部裁剪的实现参考；拟自行用现有Matplotlib实现，不整体引入其PyVista/Hydra/训练框架。

论文中另有slice、attention map、秩谱、模型效率和气动力散点图。这些不是本请求必需部分：CDLNO没有原Transolver的slice matrix，本轮不修改forward、不强制物化N×M/M×M矩阵、不增加解释性分支或新的物理指标流水线。后续若要论文级跨模型比较，可使用保存的数值结果离线重绘；不伪造未运行基线。

## 3. 当前源码事实与需要解决的差异

| 当前位置 | 实际情况 | 拟做的有限修改 |
|---|---|---|
| 六个`exp_*.py`的epoch末与 `PDE-Solving-StandardBenchmark/cdlno_entry.py:StaticRun.save/load` | 标准任务周期覆盖同一`model.pt`，内容只有state_dict；现有判定多为`ep % 100 == 0`，对应完成1/101/201…轮；独立eval里已有绘图 | CDLNO分支按完成轮数触发；加独立快照/可视化回调，四个final-only任务取消CDLNO旧周期覆盖；原Transolver分支不变 |
| Car `train.py:main/train/test`、`main.py`、`models/cdlno_run.py` | 训练末存`model_{nb_epochs}.pth`整对象，优化器/调度器未存；几何/CFD输入由原GraphDataset提供 | 可选CDLNO训练观察器、完整续训状态、周期权重；旧最终整对象协议保留 |
| AirfRANS `train.py:main/train/test`、`main.py`、`cdlno_entry.py` | 每成员末存`model`整对象，最后存对象列表；epoch内重采样/构图，验证会重复抽样；未保存续训状态 | 每成员独立周期状态，记录ensemble完成进度；保留原最终对象和列表，保留原采样/构图/验证 |
| 三个Run helper | 新训练目录`exist_ok=False`；eval先读sidecar；当前没有resume | 增加**显式resume分支**，只有合法完整checkpoint可进入已有run；普通新训练继续拒绝已存在目录 |
| `tran_evaluate/train_eval.sh`及任务脚本 | 已有train成功后eval、共享run、train-only/eval-only余参 | 增加续训/指定周期权重的明确说明与必要薄转发，不复制训练器；resume参数只传训练 |

两个必须依据实际源码保留的细节：

1. **AirfRANS正式训练入口使用`criterion='MSE_weighted'`**，见`main.py`调用`train.main`，其内部backward为`loss_vol + args.weight*loss_surf`；完整评价仍使用原MSE指标。A3报告把`train()`函数签名默认MSE称为正式入口默认值，表述不准确。A3验证过默认MSE合成函数链路，旧工业测试另验过weighted分支；这不等于三模式正式入口weighted链路已全面验收。本计划纠正认识，后续三模式续训回归应使用**实际入口weighted**，不改损失。
2. Plasticity `random_collate_fn`会用`torch.randperm`打乱每样本时间顺序，随后每个batch执行20次更新。位置`pos`与fx要按原loader区分：`x_normalizer`作用于输入场fx，不可把名字中的x误当坐标而重新归一化位置。保留原collate、时间T与标签的对应，并把随机时间顺序纳入续训对照。

另外Car的`get_samples`使用`os.listdir`收集顺序，GraphDataset的`get_shape`可能使用Python random；AirfRANS采样也使用Python random。只存torch seed不足以恢复原训练。现有权重文件缺少优化器状态，不可将其包装后宣称精确续训。

## 4. 八任务可视化具体设计

统一每次使用原验证/测试划分中的固定2个案例（不足2个时取实际数量），以实际案例ID/原数据索引记录。只用于训练诊断，不据此更改训练划分、选模型或调整loss。默认50轮首次出图，此后固定案例/网格/视角，最终补图。例图默认PNG（300dpi）+PDF；同时保存小范围案例的NPZ数值和JSON描述，以便离线重绘，无需重跑模型。

基本面板为 **Ground truth / CDLNO prediction / Absolute error**。原任务标量场误差为`abs(pred-target)`；矢量误差明确是逐分量绝对差或`||pred-target||₂`，不把“模长之差”冒充矢量误差。图中relative-L2为同案例同通道/时间范围的全场范数比，标明仅案例指标；真值范数为0时标为不可定义，不使用会在零值处爆炸的逐点相对误差图。

GT/Pred共享由固定真值确定的色标并跨epoch锁定；误差使用从0开始的独立色标，其首次标定上限及后续超范围比例写入JSON，后续不自动每张图重缩放掩盖变化。完整未裁剪值保留NPZ。使用固定坐标轴/视角；局部放大区不替换全域图。只给有确切数据定义的量标物理单位。

| 任务 | 主图与补充图 | 几何/归一化/时间约束 |
|---|---|---|
| Darcy | 解场GT/Pred/误差；系数输入图作为上下文 | 输出先用原y normalizer decode；输入系数展示也用原系数normalizer还原。按实际H/W，不写死85或sqrt(N)；不另算损失或更改边界 |
| Elasticity | 原972点上的应力GT/Pred/误差，孔洞/高应力局部图 | 原物理坐标scatter、等比例；不三角剖分填满孔洞；输出原y decode。点域FFN不变 |
| Airfoil | Mach全域三联图 + 翼型附近/尾流局部图 | 原221×51及当前采样设置的结构索引和弯曲物理坐标；pcolormesh按原网格，不重新排序、规则插值或跨翼型填色 |
| Pipe | 原弯曲管道标量速度三联图 | 原129×129索引；用于显示的物理坐标用原normalizer解码的副本，输入模型张量不改；输出y decode |
| NS | 未来10步中的第1/5/10步三联图；另画10步单案例误差曲线 | 每次从原10帧窗口起步，严格预测回填跑10个独立forward；原64×64、每步1通道。图标注原数据u及实际时间/帧号，不把未核实的标量场擅自叫速度；不扩展10→20/40 |
| Plasticity | 第1/10/20个实际T的变形场/位移模长对比，并保存4通道和时间曲线 | N=101×31，20个时间点各自前向，输出 `[N,4]`。展示变形位置按原输出前两通道；逐点误差在同一真值几何上呈现，不能用各自预测坐标算不对应误差；T与标签原轴序保留 |
| ShapeNet-Car | surface压力的固定3D视角三联图；volume的vx/vy/vz及速度模长固定剖面图 | 全单图先按原7通道正常推理，再仅对显示结果按surf/固定剖面筛选；绝不把显示子集送进模型改变全局场。原 `[velocity3, pressure1]` 顺序，原coef_norm还原；geom路径不删除 |
| AirfRANS | vx/vy/p/nut各通道三联图；压力/速度翼型附近放大 | 原x7、pos、surf和抽样后单图合同；输出用同一coef_norm副本解码。固定案例和可视化采样idx，保存idx/valid mask；散点显示不冒充完整网格解，不跨机翼补值 |

Car体积剖面使用固定几何定义的薄层点集，厚度/轴/点数写入元数据；这是点云剖面，不伪装成重建了原CFD网格。无需额外VTK读取或安装新3D依赖。若以后要求精细原网格表面渲染/流线/Cp/升阻散点，另行指定；当前八任务请求采用现有Matplotlib能稳定无界面输出的范围。

AirfRANS可视化采用独立固定案例的诊断路径：按既有sample/graph规则在副本上构造，复用原图处理，保存其idx；不触碰正式每epoch采样和20次验证采样，不替换原metrics中的scatter/平均/边界处理。周期图片注明“固定采样诊断”，不是完整官方评价结果。

渲染只接收detach后的CPU副本。新增观察器保持/恢复Python、NumPy、CPU/CUDA RNG及各模块training标志，推理在eval/no_grad作用域内进行；所有清理用try/finally。构造可视化loader也不能消耗训练的shuffle随机流。优先复用已有验证输出，必须补推理时用隔离副本；不向模型增加缓存、hooks或注意力返回值。

将原epoch已有train/val日志复制进`metrics.jsonl`，每50轮生成损失曲线；训练损失与验证指标明确分开，AirfRANS原volume/surf加权和原MSE评价分开命名。图与日志不能反向参与optimizer或checkpoint挑选。

## 5. 文件组织与checkpoint内容

所有新文件在原run下；AirfRANS在原成员子目录内重复此结构，根目录额外记录成员进度。

```text
run/
  architecture.json                  # 原架构sidecar，eval/resume不覆盖
  training_protocol.json             # 新的固定训练/数据/恢复协议
  visualization.json                 # 案例/idx/视角/色标/频率，独立于architecture
  metrics.jsonl
  checkpoints/
    epoch_0100.pt                     # 完整续训状态
    epoch_0200.pt
    latest.json                      # 最近完整有效归档的索引+校验摘要
    final.json                       # 最终状态索引，周期与final重合时不重复写
  weights/
    epoch_0100.pt                     # 纯model.state_dict，独立用于评估/迁移工具
    epoch_0200.pt
  visualizations/
    epoch_0050/case_0000/{fields.png,fields.pdf,fields.npz,metadata.json}
    epoch_0100/...
  model.pt 或原工业最终整对象/列表      # 保留现有eval定位与保存协议
```

仅最终保存的任务在最终epoch建立一对归档文件和final/latest索引，不生成前面的100/200文件。AirfRANS最终398归档命名`epoch_0398`；实际100轮结束也正是最终时，两索引指向同一对文件。旧默认eval路径仍指向原最终输出，不偷偷加载某个较早周期权重。

完整checkpoint拟保存：

- 版本、模型名称、架构/adapter和原sidecar摘要；task、front模式、CDPA模式、L/F/M/d/h等严格匹配，不做跨架构迁移。
- `model.state_dict()`、optimizer state（含Adam/AdamW矩、step、param groups）、scheduler完整state及原构造参数/total_steps；现有AMP若存在才保存scaler，不为本功能引入AMP。
- `completed_epochs`、`next_epoch`、实际optimizer `global_step`，实际scheduler调用计数，原训练目标epochs；epoch间续训不保存或恢复未完成的梯度累加。
- Python random、NumPy、torch CPU、所有相关CUDA设备RNG；若DataLoader/Sampler使用独立generator，也保存其状态。当前loader语义不改，后续出现persistent workers等未支持状态需明确拒绝“精确恢复”声明。
- 原normalizer状态或`coef_norm`（mean/std等原值）、实际数据/划分/采样/点序的有序清单/摘要、fold/task、训练字段及graph参数；不把数据集打包进checkpoint。
- 原日志序列/计数及已有验证值，用于续写曲线，不增加早停/最佳模型选择；AirfRANS当前member index、已完成member列表、各成员归档位置。
- 程序版本及脏工作区源码摘要、Python/torch/CUDA/PyG版本、dtype/SDPA/TF32等执行设置。运行字段不冒充架构字段；相同chunk可加载不同执行路径，但跨环境/精度不承诺位级同轨迹。

独立权重文件为严格完整state_dict；checkpoint内虽也有权重，独立导出满足评估和用户检查需要。新结构只使用tensor和基础类型，NumPy RNG/normalizer等转换为安全可读取表示；新checkpoint走局部`weights_only=True`，既有工业可信整对象加载边界原样保留，不放大全局反序列化权限。

每对归档先写同文件系统临时文件，flush/fsync后以原子rename完成，再更新latest/final索引；只有完整成对文件通过摘要校验才发布。保存失败明确报错，旧有效checkpoint保留；可视化失败单独记录并恢复所有状态，不销毁checkpoint或吞掉训练异常。保留全部100轮归档，不自动清理用户结果。

## 6. 断点恢复的具体执行协议

本次拟新增`--resume CHECKPOINT_PATH`，仅用于训练；不传时依然是新run，目录必须不存在。**恢复只支持已完整结束的epoch**：epoch100归档后从101开始，保留原目标500，不再训练“额外500轮”。`--resume`是本计划待实施的新选项，现在不要拿它执行现有脚本。

拟定执行顺序：

1. 先定位checkpoint及所属run，读取原architecture与training_protocol，严格校验格式、文件摘要、任务和用户显式覆盖。未指定架构/训练参数从checkpoint恢复；显式结构、loss、batch、epoch总目标、fold、采样/归一化不一致时拒绝。为避免shell默认值被当成用户覆盖，后续修改薄脚本的resume分支：恢复时不重复注入硬编码训练预设；用户余参仍最后覆盖并被验证。
2. 保持原数据读取/划分/normalizer构造，核对已载入数据的有序身份、点序摘要、normalizer数值与checkpoint。Car原`os.listdir`顺序不可默默重新排序；不同顺序/内容时清晰拒绝，不声称可以无条件继续。可记录每个已加载图/张量的有序内容摘要，无需改loader算法；可视化案例也以原索引和摘要对应。
3. 按保存的架构构造同一模型，再strict加载权重；原optimizer/scheduler按原设置构造，恢复其完整状态。不得只设置last_epoch“猜”回学习率。OneCycleLR的初始总epochs/steps_per_epoch/total_steps必须与原运行一致；延长训练总步数属于另一种实验，不在此次精确续训中悄悄支持。
4. 建好数据/model/optimizer/scheduler及加载已完成的AirfRANS成员后，**在下一次DataLoader迭代、时间打乱或采样之前**恢复所有RNG。模型构造/文件读取消耗的随机数不得落到接续训练里。原GraphDataset/构图/collate行为保留；不缓存或复用跨时间latent。
5. 恢复epoch/step/日志/成员进度，执行原epoch循环的剩余部分。截断或分段标记checkpoint之后的诊断日志，保留原文件备份；不修改旧模型状态、不伪造丢失轮次。若run已有更晚有效checkpoint，默认拒绝回退覆盖，用户要回退另开分支run不在本次默认恢复路径。
6. 每个epoch所有原训练与原验证完成后，取得恢复边界状态；运行隔离的周期可视化，再保存应到期的完整状态及独立权重。渲染不能改变此边界的RNG或model state；恢复前后必须保留原scheduler调用时刻。
7. 原最终保存协议继续生效；`train_eval.sh`在恢复训练真正成功结束后才开始eval。训练失败或恢复拒绝不进入eval。

AirfRANS `nmodel>1`需要显式维护成员进度：完成成员按原最终对象文件保存并记录；恢复当前成员的optimizer/scheduler/RNG，不重新训练已完成成员，也不把部分列表标记成完整ensemble。进入下一个成员时保留正常构造时的随机消耗和原初始化顺序，最终仍按原协议保存完整模型列表。

拟同时增加eval的显式`--checkpoint WEIGHTS_OR_TRAINING_CHECKPOINT`选择，用相应run sidecar构造并strict加载；省略时仍按原最终路径加载。工业任务原整模型/list继续可用；新周期state_dict文件不冒充原整对象，不依靠改变类路径加载。eval不读取/恢复optimizer来更新模型，不覆盖sidecar/checkpoint。

现有只保存权重/整模型的旧run仍能按原协议评估，但缺失optimizer/scheduler/RNG不能保证同轨迹续训；新resume对这类文件清晰拒绝并说明缺少字段。现在已经运行着的旧训练进程不会因为磁盘代码更新就自动拥有完整保存功能，本计划不自动中断/重启用户训练，也不把现存`model.pt`升级成伪造的“完整checkpoint”。

恢复语义保证限定为：同一数据/有序输入、模型、训练协议、运行环境，在受支持epoch边界恢复完整训练状态。CPU确定性路径要求连续与恢复一致；GPU实际比较数值容差并记录非确定性。更换GPU/torch/精度/chunk可支持结构兼容加载，但不承诺位级相同训练轨迹。

## 7. 最小变更位置及冻结边界

以下是拟新增/修改清单，不代表本轮已写实现文件。模块名在实施前检查冲突；复用现有`cdlno/checkpoint.py`架构比较而不复制。

| 文件/区段 | 拟增加的内容 | 不得改变 |
|---|---|---|
| 新 `cdlno/training_state.py` | checkpoint格式、原子成对保存、RNG/optimizer/scheduler恢复、训练协议校验 | 不import任务数据/exp，不成为模型子模块 |
| 新 `cdlno/visualization.py` | Matplotlib无界面绘图、数值导出、固定case/色标与副作用隔离 | 不改模型forward、数据点序，不增加模型参数/attention返回 |
| 新 `cdlno/training_observer.py`（必要时合并入前两文件） | epoch计数/周期策略、日志/可视化调度、final去重 | 不实现第二套训练循环/optimizer逻辑 |
| 三项目Run helper：标准 `cdlno_entry.py`、Car `models/cdlno_run.py`、Air `cdlno_entry.py` | CLI runtime选项、read-before-validate的resume/eval分支、协议/输出路径 | 原架构字段、旧full识别、同模式strict加载和旧Transolver构造 |
| 六标准 `exp_darcy/elas/airfoil/pipe/ns/plas.py` | CDLNO分支初始化观察器；epoch起点恢复；原验证输出的可选采集；epoch末/最终保存回调 | 原batch计算、normalizer/坐标/fx/label、loss/clip/optimizer/scheduler、时间循环与原Transolver分支 |
| Car `main.py`、`train.py:main`，必要的 `main_evaluation.py`新分支 | 传入可选观察器/coef_norm，epoch起点与末回调、周期权重eval选择 | `train/test`原损失、GraphDataset/geom、fold/graph/mask、完整drag流程 |
| Air `main.py`、`train.py:main`，必要的 `main_evaluation.py`新分支 | 观察器/coef_norm、成员恢复进度、周期权重eval选择 | 原`MSE_weighted`、每轮random.sample/radius_graph、20次验证采样、官方metrics/scatter平均/边界处理 |
| `tran_evaluate/_common.sh`、`_standard.sh`、工业/NS薄脚本与`train_eval.sh` | 训练专用resume转发、eval checkpoint选择、避免默认参数干扰恢复；帮助与示例 | 原task工作目录、数据路径、Air train/eval不同my_path语义、用户余参优先 |
| 新定向tests、运行说明与STATUS/memory | 续训等价、副作用、可视化与周期验证；真实结果报告 | 不伪造数据或PyG模块，不import会加载数据的入口，不为测试重构训练器 |

模型及原参数配置严格冻结：`cdlno/core.py`、`modules.py`、`cdpa.py`、模型wrapper的计算/初始化、原Transolver模型、八任务architecture/训练预设JSON、Air `params.yaml`均不因本需求修改。新运行策略放到独立training/visualization metadata，不把它们混入模型结构兼容比较。`requirements`/`pyproject`不变，使用既有Matplotlib/NumPy/PyTorch与工业PyG。

新增钩子对旧模型默认`None`/不执行；不以`isinstance`猴子补丁、全局torch.load修改或改造模型forward来接入。只允许经过批准的epoch边界/运行管理差异出现在AST diff中。

## 8. 分阶段实施建议（每阶段单独批准后再执行）

本轮为 **V0：阅读与计划**。为继续遵守“每轮一个明确阶段”，建议后续拆分如下；不会因为这份完整计划已存在而自动全部实施。

| 阶段 | 范围 | 必须交付的验收 |
|---|---|---|
| V1 | 公共完整checkpoint/恢复协议、RNG与可视化基础组件；不接八任务训练 | 49/50/99/100/200/398/500及短run调度；原子写故障；CPU连续vs新进程恢复；图像/NPZ尺寸/色标/零目标；可视化开关不改RNG/权重/梯度；strict及旧格式拒绝 |
| V2 | 四静态标准任务Darcy/Elasticity/Airfoil/Pipe及薄脚本 | 实际normalizer/原loss合成训练；Airfoil100轮策略、其余final-only；真实N布局+5×7、孔洞/弯曲坐标；原训练batch AST冻结；三front模式同模式加载/恢复 |
| V3 | NS/Plasticity及时间图 | 原10→10真值/预测回填，NS每batch1更新；Plasticity随机时间collate/20更新/1scheduler；新进程续训的窗口/时间顺序/学习率/权重一致；时间图对应真实标签和T |
| V4 | ShapeNet-Car/AirfRANS、工业对象/列表与ensemble进度 | 真实PyG原mask损失，Air实际weighted；get_shape/random.sample/radius_graph及验证抽样顺序；coef_norm、单图、多图拒绝；单成员与多成员恢复；原官方评价处理不改 |
| V5 | 八任务综合回归和交付 | 所有task×front模式周期/权重/恢复/eval表；off/entry/every有限组合；旧Transolver AST/hash；三cwd新进程；本地可用GPU有限检查；完整脚本/README与限制 |

每阶段结束报告范围、文件/原因/diff、公式/接口、实际命令/环境/通过失败未运行、冻结证据和剩余问题；先自审重点，再交用户审查。遇到必须改变模型/数据语义的冲突停止该部分并报告，不自行扩大范围。

## 9. 验收标准与重点风险的具体检查

1. **真正断点续训**：用同一初始权重/随机状态的小合成任务运行K个完整epoch，对照“运行至k保存→退出→新进程加载→运行余下K-k”。比较每步loss、lr、optimizer/scheduler计数/状态、最终权重、随机抽样/时间排列。测试中的短K不会修改正式epochs预设；OneCycle计划在两条路径中必须保持相同总步数。除model之外故意删任一关键状态应拒绝恢复，不允许只比较最终能forward。
2. **不干扰训练**：同样合成输入与初始状态，关闭/开启可视化两条路径比较下一轮采样、梯度、权重、module.training与model.state_dict；还要测试绘图异常的finally恢复。CPU确定性比较严格相等，GPU按实际backend报告容差。即使dropout=0也不能忽略loader和采样RNG。
3. **周期正确**：判断`completed_epoch % interval == 0`，不在epoch1误保存；100处同时可视化+保存；398/最终短run补存；整百final无重复；other四任务无中途checkpoint；每文件记录准确epoch/member/task/模式。
4. **图像忠实**：小型已知场与非方格数值检查通道/坐标映射、GT/Pred共享色标、误差符号与量纲、decode、surf/volume、NS时间和Plasticity变形坐标。NPZ保留未截断原值，图不能通过重排或裁剪范围隐藏错误。无GUI后端生成的文件实际打开检查。
5. **数据/协议不变**：比较实施前源码hash与AST；原normalizer/loss/时间循环/训练采样/边界处理保持。实际保存模型结构及同权重输出完全不变；模式/架构/数据manifest/normalizer/fold/optimizer契约错误时早报错。可允许dataset路径迁移但要实际有序数据摘要一致，不把路径相同误当数据相同。
6. **checkpoint稳健**：保存中途故障、不完整文件/摘要不符、旧权重误作resume、已有run新train、eval覆写、回退覆盖、Air部分成员列表、多模式冲突均负向验证。工业旧可信对象路径保留，新的训练checkpoint不新增任意pickle权限。
7. **成本及限制**：默认只2案例、固定时间帧，渲染用CPU，显存张量及时释放；单独记录绘图/保存耗时和文件大小，不能把带额外可视化的epoch耗时直接用作模型速度结论。没有真实数据时只报告合成/PyG/GPU/静态证据，不宣称真实读取、收敛、物理精度通过。

尚需用户在审批中接受的关键选择：当前计划采用**仅CDLNO、epoch边界完整恢复、每次2个固定案例、最终补图/补存、保留原最终评估文件**。后四任务遵照请求只有最终保存，因此没有中途恢复点。若这些选择符合意图，可以先明确批准V1；后续阶段仍逐阶段审查。

## 10. 本轮已做与未做

已做：读取适用AGENTS/当前状态、检查实际源码和已有修改；核对五篇论文的相关段落/图注及五幅实际图；确认八任务保存/epoch/可视化位置、Air正式weighted损失、Plasticity随机时间与Car/Air采样边界；编写本计划和来源记录。没有导入exp/main、执行模型训练/评价、安装依赖或修改生产代码。

计划前266份文件hash与git状态在 [planning_start.json](viz_resume_audit/planning_start.json)；本轮结束核查只允许计划/来源记录及状态/记忆增量变化。**这里不新增任何“模型验证通过”结果**；A3历史证据保留，Air损失覆盖的表述限度在第3节显式更正。

审批是用户本轮明确要求：“先制定修改计划交由我进行审批，然后由我来决定是否进行修改”。因此当前仅交付方案，不实施其中的新runtime选项或训练入口改动。

**本阶段结束，未执行实现阶段。**
