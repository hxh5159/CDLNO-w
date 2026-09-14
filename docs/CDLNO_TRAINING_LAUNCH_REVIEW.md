# CDLNO Car / NS 训练启动前审查

日期：2026-09-14。用户本轮授权：最终逻辑核查、整理Car/NS训练评价命令到 `tran_evaluate/`；没有授权本机真实训练或后续研究功能。分支 `main`，基线HEAD `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。本轮前已有阶段0–10未提交实现，全部保护。

**结论：已确认的CDLNO模型架构未发现新缺陷，最终120项无数据回归通过。发现并纠正了新模型NS默认裁剪偏离原入口的问题；Car原评价硬编码路径/fold及日志汇总问题已明确记录，并在新评价脚本添加路径/fold检查。不能据此断言真实训练流程已经全部无误。**

## A. 完成范围与设计依据

重新从 `core.py`、`cdpa.py`、模块、Car/NS适配器、实际入口/helper、原Car train/loader/drag evaluator和NS真实时间循环核对，复跑约定完整验收。独立核查新命令进入原parser后的实际参数，而不只阅读旧报告。

本轮实际获取并阅读 [Transolver论文v2](https://arxiv.org/pdf/2402.02366v2) 第15–17页（B.1/B.2/B.3、Table8），文本证据为 [paper-excerpts.json](training_launch_audit/paper-excerpts.json)。NS的64×64/10→10/1000+200、Car的789+100/输入xyz-sdf-normal和200epoch等与原代码核对。论文中的Car/NS M32是原Transolver配置；本轮保留已确认CDLNO的M64、2+6和FFN ratio2，明确区分模型配置与训练协议。

曾发出配置对齐澄清；截至交付未收到另选配置回复。本轮采用当前明确要求“训练评估逻辑与原仓库一致”及v1.2 §4.5“clip按原代码保持”，将NS默认恢复无裁剪；保留既有明确确认的CDLNO架构。不将未回复当作改M/FFN架构的授权。

## B. 文件、理由和diff

| 文件 | 改动与理由 |
|---|---|
| [tran_evaluate/car.sh](../tran_evaluate/car.sh)、[ns.sh](../tran_evaluate/ns.sh) | 填充用户已有空文件，各支持train/eval/help；显式CDLNO配置，参数在末尾可覆盖，不自动连续运行 |
| [tran_evaluate/_common.sh](../tran_evaluate/_common.sh) | 仅转发命令、设置原cwd/根PYTHONPATH、处理含空格/相对数据与run路径、dry-run；不含模型/训练逻辑 |
| [tran_evaluate/check_car_eval.py](../tran_evaluate/check_car_eval.py) | 标准库预检已知原阻力评价固定root/param0限制；从真实entry仅提取parser定义，不导入/运行entry；不改指标，不伪造数据 |
| [tran_evaluate/README.md](../tran_evaluate/README.md) | 四条完整命令、数据目录、原协议、CDLNO与Transolver配置差别、checkpoint和实际限制 |
| [NS JSON](../PDE-Solving-StandardBenchmark/configs/CDLNO/ns.json)及两份CDLNO NS launcher | `max_grad_norm:0.1→null`，两脚本移除显式clip0.1；原Transolver默认/脚本与exp循环保持 |
| [tests/test_temporal_standard.py](../tests/test_temporal_standard.py) | 新增1个针对实际缺陷的回归：旧分支None/ratio1/M32；新分支None/ratio2/M64；执行真实clip AST，默认零次调用、自选0.1一次 |
| AGENTS、STATUS、memory、本报告、[证据目录](training_launch_audit/) | 记录本轮边界、实际发现、验证、冻结证据和可审查patch |

修改的既有文件共9个；相对本轮起点173文件，164字节不变，无删除/意外改动。细目见 [freeze-check.json](training_launch_audit/freeze-check.json)。[launch-changes.patch](training_launch_audit/launch-changes.patch) 包含本轮变化；构建patch前按起点SHA验证修改前文本，不执行git add/commit。

## C. 公式、形状和执行协议核对

| 范围 | 源码与实际合同 |
|---|---|
| 核心 | `cdlno/core.py::CDLNO.forward`：`H0[B,N,256] → F=2完整LRSA(HF,Ti[B,64,256]) → bridge Z0[B,64,256] → entry CDPA → P=6独立后段 → readout(HF,ZP)`。NS每次真实时间调用重新走此图 |
| CDPA | `cdlno/cdpa.py`：每历史独立softmax_M，完整O+b，无额外Cross残差；R0=Z。`alpha=softmax_sources(w·RMSdepth(Rs))`，RAW候选加权；w零初始化、FP32 depth、chunk0/1/分块同权重同数学定义 |
| 历史/初始化 | raw Z0保留，every的历史在当前身份使用后追加；局部list不跨forward；off/F0entry无闲置融合参数。子模块各自初始化一次，wrapper不递归重置 |
| 最终readout | `LRSAFeatureReadout` query及点残差为HF；NS有groups1 dense3×3 ConvFFN；Car point FFN。无latent卷积或coordinate-query decoder |
| Car输入/输出 | `models/CDLNO.py::Model`只读cfd.x7及单图batch/ptr，忽略geom/y；`[N,7]→[N,4]`，velocity3/pressure1，不原地改字段；支持可变N、拒绝多图 |
| Car反向 | 原 `train.py::train`：`total_loss=MSE(out[:,:3],y[:,:3])+0.5*MSE(out[surf,3],y[surf,3])`；每图一次backward/Adam/OneCycle更新；fold0、200epoch/batch1、val_iter10 |
| NS接口 | `TemporalStandardModel`：reference64替换xy再拼fx10，stem74；`x[B,4096,2],fx[B,4096,10]→[B,4096,1]`。无placeholder、无time_fc、无跨物理时间latent缓存 |
| NS时间和loss | 原 `exp_ns.py`：10次真值回填、10个单步TestLoss累加后一次backward/AdamW/OneCycle；每epoch验证和独立eval均预测回填。eval以全10帧relativeL2计量；500epoch/batch2/lr1e-3/weight_decay1e-5/无clip |
| checkpoint | 原保存频率：Car末轮整模型model_200.pth；NS每100epoch及末轮state_dict model.pt。CarRun/StaticRun先读sidecar后比较、strict权重；chunk变化可载入；新训练目录隔离，eval写新结果目录 |

NS原脚本未传 `--mlp_ratio`，因此原exp的ratio1生效；此前CDLNO ratio2是已确认架构设置而非协议回归。NS原脚本也未传 `--max_grad_norm`，原exp为None；此前CDLNO把0.1写入preset属于训练配置偏差，本次纠正。**旧报告“冻结循环”等AST证据仍有效，但不能由此推导当时有效clip默认也一致；本报告修正该过强结论。**

## D. 实际验证和边界

在仓库根实际执行：

```bash
# 修正前完整检查：119项通过，50.796秒。
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
# 修正后的定向时间组：10项通过，10.356秒。
python -B -m unittest discover -s tests -p 'test_temporal_standard.py' -v
# 修正后的完整套件：120项通过，52.764秒；0失败/错误/跳过。
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
# 含空格、相对路径、flag=value、真实parser与负向检查；不运行数据入口。
python -B docs/training_launch_audit/verify_launchers.py
bash -n tran_evaluate/car.sh tran_evaluate/ns.sh tran_evaluate/_common.sh
git diff --check
```

实际日志：[最终120项](training_launch_audit/regression-final.log)、[原119项](training_launch_audit/regression.log)、[定向10项](training_launch_audit/temporal-focused.log)、[脚本检查](training_launch_audit/launcher-checks.json)。启动检查分别对Car/NS的train/eval共4条命令执行dry-run，再在各原cwd新进程只执行AST提取的真实parser定义/helper；检验最终超参数、参数覆盖、cwd/PYTHONPATH、空格/相对路径和dry-run无写入。不import exp/main，不创建假数据文件，不调用实际train/eval。

| 项目 | 本轮状态 |
|---|---|
| 基础模块、LRSA原码参考、CDPA数学/FP32/chunk梯度、21种核心+扩展深度 | 通过（完整套件） |
| 八wrapper合成；NS两路完整10步及更新频率；原损失连接 | 通过 |
| 真实PyG Data/Batch：Car/Air单图/可变N/无标签泄漏/多图拒绝 | 通过；合成对象，不是真实数据读取 |
| 三原cwd新进程导入/整模型或state_dict加载；strict负向/sidecar不覆写/chunk兼容 | 通过（完整套件既有用例） |
| 本地GPU：模块、CDPA、point/grid小核心、静态wrapper精度检查 | 通过，6个GPU测试方法实际执行；本轮未追加完整规模Car/NS训练 |
| NS默认无clip与自选clip，四条脚本真实参数解析，Car错误fold/缺run拒绝 | 通过；含`--fold=1`缩写不绕过guard |
| 最终测试失败/错误/跳过 | 0 / 0 / 0 |
| 真实Car guard正向、VTK/缓存/NS MAT内容完整性、实际读取及完整指标流程 | **未运行** |
| 远端Python3.10/torch2.11/cu128的新CDLNO训练/评价、Python3.11 | **未运行**；用户已有原Car成功证据保留 |
| 收敛、论文精度、实际默认规模显存/epoch耗时、种子方差 | **未验证** |

套件在现有本地Python3.13.9/torch2.13.0+cu130/PyG2.3.1/RTX5090 Laptop执行（版本依据既有环境记录和本轮日志），不作为改变远端依赖的理由。没有安装/替换任何依赖，不强制把本地版本写入启动脚本。远端兼容权威仍是用户已有Python3.10/torch2.11/cu128环境；脚本支持 `CDLNO_PYTHON` 指向该解释器。

## E. 冻结范围与证据

相对本轮起点，所有原71个跟踪文件字节不变，包括此前有限模型接入分支与用户已有改动。本轮没有改核心、CDPA、wrapper、数据文件/loader、exp/main入口、normalizer、loss、optimizer、scheduler、时间循环、指标、依赖或原Transolver模型/脚本。唯一生效训练默认变化是新CDLNO NS的clip纠偏（JSON+新脚本），不是悄悄改变冻结循环。

相对阶段0 HEAD仍是此前已批准的12个有限接入文件及README不同，其余58个原跟踪文件字节相同；本轮无新增旧源码变化。完整套件再次通过6个exp和工业入口的原AST投影、原模型/脚本/数据/指标字节检查。证据见 [start.json](training_launch_audit/start.json) 与 [freeze-check.json](training_launch_audit/freeze-check.json)。

## F. 已完成自审、剩余问题

1. **配置不被旧默认覆盖：已自审。** 四条最终argv进入真实parser，均d256/h8/L8/F2/M64/ratio2/entry；Car200/1，NS500/2/无clip。用户选择架构override后评价仍需匹配sidecar。
2. **历史/两级融合/readout/初始化：已自审。** 源码与独立数学、历史时序、梯度和参数独立用例一致，未发现待修复模型问题。默认entry历史2来源，chunk0一次CDPA历史SDPA；这不等于全模型只有一次SDPA。
3. **原训练与时间路径：已自审。** 原AST仍一致；补测并修复原AST检查看不到的NS参数默认偏差。原Car日志loss名称/汇总颠倒为继承问题，实际反向目标正确，冻结旧train文件未改。
4. **checkpoint/目录/加载：已自审。** 既有严格加载与sidecar规则通过；新脚本只转发并规范路径，支持含空格和不同cwd，dry-run不执行任何入口。
5. **真实Car评价存在限制：仍需用户按实际数据环境处理。** 原阻力函数固定`/data/PDE_data/mlcfd_data/training_data/param0`，新完整eval脚本据此限制fold0及同数据路径。其他fold训练本身合法，但原完整阻力评估不能直接推广到其他fold；若要可移植root/多fold评价，需要以后明确授权最小指标路径修复，不能声称本轮已支持。未读取真实VTK/缓存验证节点对应。

没有因性能或训练准备改变CDPA公式、最终decoder、损失或数据协议。训练指令和具体继承限制均在 [tran_evaluate/README.md](../tran_evaluate/README.md)。本轮没有下载数据、运行真实训练/评价或commit/push。

**本阶段结束，未执行下一阶段。**
