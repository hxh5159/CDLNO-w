# V1：完整续训归档与独立绘图公共组件

日期：2026-09-15。用户已批准按[计划](CDLNO_VISUALIZATION_RESUME_PLAN.md)修改；按持续有效的逐阶段约束，本轮只执行V1。**V1完成，待审查；八任务训练入口尚未接入新功能，当前脚本还不能使用新的`--resume`。** V2–V5没有执行，A4消融性能工作没有执行。

## A. 完成范围

增加模型外部的完整checkpoint、配对纯权重、严格恢复、epoch事件策略与输出场绘图组件。没有改动CDLNO/Transolver结构、forward、初始化、front模式、CDPA或任何任务的数据/训练/评估代码。

新的公共接口可以恢复模型、Adam/AdamW状态、OneCycleLR/CosineAnnealingLR状态、epoch/更新计数、Python/NumPy/torch CPU/CUDA及显式DataLoader generator随机状态，并核对既有sidecar、模型结构、任务/数据顺序协议及normalizer。实际连续训练与跨进程恢复对照已通过，见D节。

**完整checkpoint才含续训所需状态。** `weights/epoch_XXXX.pt`保持纯state_dict；通过本接口指定该路径时，会查找并校验同run下完整checkpoint及manifest，再恢复整个训练状态。单独拷走纯权重、旧`model.pt`或旧工业整模型对象不能恢复原优化器/调度器进度，会明确拒绝。没有用`strict=False`、猜测学习率或伪造历史状态。

## B. 文件与增量diff

| 文件 | 本轮变更与用途 |
|---|---|
| `cdlno/training_state.py` | 新增`TrainingProtocol`、`TrainingArchive`、RNG保存/恢复/隔离、有序数据摘要、成对原子归档及严格恢复 |
| `cdlno/training_observer.py` | 新增`EpochPolicy`和`EpochObserver.after_epoch`；只调度回调/保存，不拥有训练循环 |
| `cdlno/visualization.py` | 新增显式网格/点云三联图、锁定色标、PNG/PDF/NPZ/JSON输出；不导入模型或数据入口 |
| `tests/test_training_state.py` | 14项针对性检查，含六组CPU新进程对照、一组GPU对照、归档故障与恢复负向项 |
| `tests/test_visualization.py` | 3项网格/图像/色标/误差与保存值检查 |
| 本报告、计划顶部状态、STATUS、AGENTS、memory | 增量记录当前范围、证据与后续边界，保留历史内容 |
| `docs/viz_resume_audit/v1/` | 起点清单、环境、实际日志、绘图演示、冻结核查和增量patch |

HEAD为`769fa333742f73c132868cf560bce5ec21529362`；原Transolver历史基线为`75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。本轮对比的是**包含用户和已接受A1/A2/A3修改的真实工作区**，而非把HEAD误当成干净模型基线。

270份起点文件和hash保存于`/home/hwz/CDLNO-artifacts/viz-v1-before-04ww1akt/source`及[start.json](viz_resume_audit/v1/start.json)。审查本轮请使用[v1-changes.patch](viz_resume_audit/v1/v1-changes.patch)；普通`git diff`还包含原先未提交修改。本轮未commit/push。

## C. 接口、数据形状与执行次序

### C1. 周期策略

按**已完成epoch数**`e`判断，不在源码`ep=0`时误当第100轮：

`visualize = (e % 50 == 0) or (e == total_epochs)`。

NS/Car/AirfRANS/Airfoil：`checkpoint = (e % 100 == 0) or final`；Darcy/Elasticity/Pipe/Plasticity：`checkpoint = final`。实际总轮数仍取正式入口原参数；策略默认保留PDE500、Car200、Air398的含义。100轮同时触发时只存一对文件；398和小于50轮的最终训练也补图/补存。显式`visualize_every=0`可关闭包括最终图在内的全部绘图；运行周期不进入模型architecture。

| 任务 | V1公共策略 | 正式任务入口接入 |
|---|---|---|
| Darcy / Elasticity / Pipe / Plasticity | 每50轮图+最终图；仅最终checkpoint与权重 | 未执行，分别属于V2/V3 |
| Airfoil / NS | 每50轮图+最终图；每100轮归档+最终归档 | 未执行，分别属于V2/V3 |
| ShapeNet-Car / AirfRANS | 每50轮图+最终图；每100轮归档+最终归档 | 未执行，属于V4 |

后四个final-only任务仍没有中途恢复点；图片不能替代checkpoint。

### C2. 完整归档与恢复

`TrainingProtocol`要求入口显式提供task、原总epochs、每epoch真实optimizer/scheduler调用次数、训练字段、split/有序数据摘要和adapter架构。`ordered_fingerprint`只处理调用者已载入的数组/ID，不读取或重排数据。`normalizers`必须显式给出状态字典，无normalizer时传`{}`；恢复时与当前构造结果逐值比较，不能静默改用不同均值/标准差。

归档内容包含：严格模型state_dict、optimizer矩/step/参数组、scheduler完整状态及构造计划、计数、normalizer、全部受支持RNG、调用者的历史日志/额外元数据、架构和sidecar摘要。记录实际Python/torch/CUDA/NumPy/Matplotlib/PyG版本、device/dtype/TF32/SDPA开关、HEAD及共享包/实际wrapper文件hash。后续入口仍需通过协议/extra提供实际数据、collate/采样、入口源码及工业成员进度；V1没有凭空构造这些事实。

文件关系：

```text
existing_run/architecture.json      # 仅读取，不改写
existing_run/training_protocol.json # 首次保存创建，以后逐项比较
existing_run/checkpoints/epoch_0100.pt   # 完整状态，weights_only=True读取
existing_run/weights/epoch_0100.pt       # 纯state_dict
existing_run/checkpoints/epoch_0100.json # 两文件的SHA256，完整配对的提交点
existing_run/checkpoints/latest.json
existing_run/checkpoints/final.json     # 仅实际完成训练时发布
```

每个文件先写同目录临时文件，再flush/fsync、原子rename；**两份pt均完成后才提交manifest，再发布latest/final**。同epoch相同状态可重试，不改写pt；不同状态拒绝覆盖。第二份文件写失败保留上一份有效归档，并清理本次未提交文件。若manifest已完成而latest更新失败，恢复仍识别最新已提交epoch，阻止从较早epoch回退覆盖；可用该明确epoch路径恢复/重试发布索引。

恢复流程：读manifest并校验两文件摘要 → `weights_only=True`读取 → 核对sidecar/协议/normalizer/架构/权重keys/shape/dtype/数值 → 检查optimizer参数名及顺序/矩/step、scheduler计划及计数 → 检查RNG/generator → 拒绝回退 → `strict=True`装载原对象 → 清空已完成epoch边界的梯度 → **最后恢复RNG**。不重新初始化模型，不改变参数对象id，不绕过旧工业可信加载边界。

计数采用：`global_step=e×updates_per_epoch`、`scheduler_steps=e×scheduler_steps_per_epoch`。两者可以不同，未来可准确表达Plasticity的20次更新/每batch1次scheduler；没有因此实现或改写该时间循环。`next_epoch=e`是零基循环起点：完成100轮后执行`range(100,total_epochs)`，即从人类计数第101轮继续。恢复原计划剩余轮次，不把目标500改为再跑500。

### C3. 绘图与训练隔离

`render_fields`输入坐标`[N,2/3]`、真值/预测`[N,C]`、每通道固定色标；规则任务显式给`(H,W)`且`N=H×W`，不猜`sqrt(N)`。规则网格保留物理坐标与原索引，使用pcolormesh；2D/3D点云只scatter，不能生成虚假网格连接。可选mask只筛选显示，NPZ保留全部传入坐标/真值/预测/绝对误差/mask。

标量误差`E=abs(pred-truth)`；案例指标`relative_L2=norm(E)/norm(truth)`，零目标记为不可定义。GT/Pred共享由首份真值建立的色标；误差使用独立从0开始的固定色标，超范围比例写JSON，原数组不截断。图内标明案例指标，PNG300dpi/PDF/NPZ/JSON均保存。向量模长、物理decode、时间曲线、工业表面/剖面、固定案例选择属于后续任务适配，未用通用标量图冒充完整任务可视化。

`isolated_evaluation`保存Python/NumPy/torch/generator RNG、各子模块train/eval标志和buffer，使用eval/no_grad，异常时仍恢复。绘图不碰参数、梯度、optimizer；回调只允许推理/渲染。`EpochObserver`记录绘图异常并继续应到期的归档，保存错误则向外抛出。使用logging报告绘图错误，避免环境将warnings当异常时跳过checkpoint。

人工实际打开了两幅明确标注“SYNTHETIC ONLY”的[5×7弯曲网格图](/home/hwz/CDLNO-artifacts/viz-v1-demo-eo0pz_6e/grid_5x7/fields.png)和[3D点云图](/home/hwz/CDLNO-artifacts/viz-v1-demo-eo0pz_6e/cloud_3d/fields.png)。它们来自解析函数加指定扰动，**不是模型预测或实际数据集实验**。自审修正了初版3D图边缘刻度被裁切的问题，最终导出保留完整边界。两案例渲染合计1.627秒，文件大小/hash见[记录](viz_resume_audit/v1/render-demo.json)；这不是训练epoch性能结果。

## D. 实际命令、环境与结果

现有本机环境为Python3.13.9、torch2.13.0+cu130/CUDA13.0、NumPy2.2.6、Matplotlib3.10.6、RTX5090 Laptop GPU。没有安装/修改依赖。远端Python3.10/torch2.11/cu128仍为兼容目标，本轮未访问远端；Python3.10语法解析通过不等于远端运行通过。

从仓库根目录执行：

```bash
python -B -m unittest discover -s tests -p test_training_state.py -v
python -B -m unittest discover -s tests -p test_visualization.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. python -B docs/viz_resume_audit/v1/render_demo.py
python -B docs/viz_resume_audit/v1/audit_scope.py
```

| 实际检查 | 结果与边界 |
|---|---|
| 首轮定向 | training12/12、绘图3/3通过；没有首轮失败测试 |
| 自审补强后定向 | training14/14通过，47.899s；绘图导出调整后3/3通过，2.000s |
| 完整既有+新增套件 | **201/201通过，122.573s，0失败/错误/跳过**；17项新增，184项既有回归 |
| CPU新进程连续vs恢复 | full/no_sa/identity×两种scheduler，共6组，每组连续4epoch对比2epoch保存后新进程再2epoch；每轮2次optimizer，共8步；权重、AdamW状态、scheduler、逐步loss/lr、Python/NumPy采样、真实DataLoader shuffle与RNG全部逐位相等 |
| 本机GPU新进程连续vs恢复 | identity + OneCycleLR一组，FP32、math SDPA、TF32关；权重及optimizer atol1e-6/rtol1e-5，loss小数点后5位，采样/lr/RNG严格一致，通过 |
| 训练隔离 | 同初始状态开/关真实推理+绘图，最终权重、梯度、optimizer/scheduler、逐步记录及RNG严格相等；异常回调也恢复RNG/模块状态，通过 |
| 严格拒绝/故障 | 不同模式/总epoch/loss/数据摘要/normalizer/参数组/scheduler；缺关键字段/Adam矩；不完整RNG/坏dtype/坏beta/坏manifest/摘要；旧纯权重/外run；覆盖/回退；第二文件失败和latest失败，均通过 |
| 周期/绘图 | 八任务49/50/99/100/200/398/500及短run边界、final不重复、5×7顺序/格式/固定色标/零目标/3D显示mask，通过 |
| 模型结构保护 | 原模块文件hash、参数对象id、state_dict键、sidecar字节均保持；恢复没有重新构造或宽松加载 |
| 原损失/PyG | 完整套件重跑了既有原损失及真实PyG接口；**本轮新的续训对照是小CDLNO核心+显式合成MSE，不是八任务原损失续训链路** |

小核心采用B2、5×7、d8/h2/M4、L3/F1、every_block、输出2通道，仅用于状态恢复检查。`TrainingProtocol.task='ns'`只用于策略选择，loss明确记录为synthetic MSE；该测试没有假装执行NS10→10原时间协议。GPU及两调度器检查也不是增加新的正式任务训练设置。

日志：[完整回归](viz_resume_audit/v1/final-regression.log)、[归档定向](viz_resume_audit/v1/training-final.log)、[绘图定向](viz_resume_audit/v1/visualization-final.log)、[环境](viz_resume_audit/v1/environment.json)。参考LRSA测试打印缺少可选xformers/liger的提示后使用其已有替代路径，实际参考比较通过；本项目未安装或引入这两个依赖。

## E. 冻结区域与自审结果

[冻结审计](viz_resume_audit/v1/freeze.json)对本轮270份起点文件逐一比较，128份既有生产/任务/脚本/工具文件全部字节相同。原模型、共享math/config/core/wrapper/checkpoint helper、exp/main/train/eval、数据/采样/normalizer/loss/时间循环/指标、脚本及依赖没有变化；旧AGENTS/STATUS/计划/memory只追加新状态。

| 自审重点 | 结论与证据 |
|---|---|
| 真续训，而非只加载权重 | 新进程6组CPU+1组GPU；对照optimizer/scheduler/RNG和后续每步记录，已通过 |
| 写入中断与覆盖 | 发现并修复“manifest已提交、latest未更新”时潜在回退；新增故障注入验证，已通过 |
| 绘图干扰训练 | 真实推理/渲染开关和异常路径逐值对照，已通过；修正warning-as-error对保存的干扰 |
| 模型结构与既有协议 | 128份冻结文件hash及201项回归通过；没有改forward/初始化/模式/数据训练代码 |
| 图像可信度 | 点序/原数组/固定色标/零目标检查通过；实际开图修复3D刻度裁切，最终图复查通过 |

## F. 未验证项及下一阶段边界

1. **新归档/周期图/续训CLI尚未连接任何正式任务。** V2负责四静态任务，V3负责NS/Plasticity，V4负责工业及ensemble，V5综合交付。当前不能给出可执行的任务`--resume`命令，也不能宣称八任务已经具备本次新功能。
2. 新续训未验证实际normalizer/data/采样/collate/时间链路、PyG工业原mask损失及对象/list/ensemble恢复。既有PyG回归通过只证明原功能未回归。后续Air必须使用实际入口MSE_weighted，保留计划指出的A3历史覆盖差异。
3. 没有真实数据完整读取、真实训练/断电续训、收敛、精度或真实epoch耗时结果。远端torch2.11/cu128、共享文件系统的真实断电/fsync行为未验证。有限本机GPU对照不能代表所有GPU/AMP/Flash/compile/DDP；没有自动改变这些运行设置。
4. 完整恢复限定在完整epoch边界、同训练计划与数据顺序；不保存未完成batch梯度。API可携带已有torch.amp.GradScaler状态，但本轮未运行AMP恢复实验；正式任务也没有为此引入AMP。worker内部隐藏状态、persistent workers、分布式恢复不在当前保证内。
5. 保存只支持单writer。硬中断可能留下`.writer.lock`或未提交孤立文件，接口清楚拒绝覆盖并要求检查；不会自动删除用户文件。文件缺失/摘要损坏不会跳过校验。保存/恢复产生的CPU快照和磁盘耗时未作真实大模型性能验收。

没有遗留已发现的V1缺陷需要用户裁定；以上为本阶段可验证范围与后续工作边界。**本阶段结束，未执行下一阶段。**
