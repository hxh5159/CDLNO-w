# 补充阶段 A2：八任务前段消融入口与checkpoint

日期：2026-09-14。用户已审查通过A1，本轮只执行A2。**八任务的full/no_sa/identity参数、构造及加载链路已接入，最终141/141回归通过；A3性能工具等和A4未执行。** 未运行真实数据或训练。

## A. 完成范围与设计边界

在八任务已有CDLNO分支中增加唯一架构字段 `front_latent_mode`。原命令省略参数时新训练仍为full；eval省略时先从显式已有run的architecture.json恢复，显式值与checkpoint不同时报 `architecture mismatch: front_latent_mode`，不加载权重、不写sidecar。

两工业任务仍按实际协议保存：Car整模型对象；AirfRANS每成员整对象和运行根目录模型列表；六标准任务仍为state_dict。未增加resume、跨模式权重转换、strict=False或全局torch.load权限。模型类路径及核心数学不变。

原阶段八任务都已完成，本轮没有“尚未完成原阶段”而被误报支持的任务。这里的“支持”指可执行参数/合成接口/加载验证，不代表真实训练、收敛或数据完整性验收。

## B. 文件及diff

A2起点仍为main/`769fa333742f73c132868cf560bce5ec21529362`，起点已包含用户接受的A1及早先启动修改，不能用总git diff把它们归入A2。已保存235个实际文件及起点status/hash到 [start.json](front_ablation_audit/a2/start.json)，完整旧源码在 `/home/hwz/CDLNO-artifacts/front-a2-before-cmyap6x3/source`。原Transolver基线仍为 `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。

| 修改区段 | 原因/边界 |
|---|---|
| `cdlno/standard.py` 两个wrapper构造、`cdlno/airfrans.py`架构函数/构造、Car `models/CDLNO.py` 架构函数/构造 | 新增默认full参数，仅写入既有核心配置；forward、stem、位置/时间、初始化顺序及图字段不变 |
| 三项目 `cdlno_entry.py` / Car `models/cdlno_run.py` | 添加CLI选项、JSON默认、显式覆盖判定、eval读模式、model_kwargs转发、模式运行名及工业对象一致性检查 |
| `cdlno/checkpoint.py` | sidecar完整字段检查，只对已识别的旧full缺mode兼容；共用eval mode恢复和工业对象mode检查。不读取模型数据或训练代码 |
| 八份 `configs/CDLNO/*.json` | 每份model仅新增 `"front_latent_mode": "full"`；其它字段逐项保持。Air YAML仅供原训练字段使用，未改 `params.yaml` |
| `tran_evaluate/_common.sh`、`_standard.sh`、`car.sh`、`airfrans.sh` | 新参数的薄目录解析：no_sa/identity默认目录追加模式，最后一个标准拼写的模式参数决定后缀。full原路径不变，显式run余参仍优先 |
| `tests/test_front_task_modes.py` | 八任务×三模式、原构造AST/原工业loss语句、负向加载、旧对象、目录/命令和新进程验证，共8个测试方法 |
| 原 `test_static_standard.py`、`test_shapenet_car.py`、`test_airfrans.py` | 只调整三项“参数捕获、不存在run占位符”的测试，mock新增mode读取边界。实际checkpoint及新A2测试使用真实临时sidecar，不mock加载校验 |
| FRONT_ABLATION、TASK_LAUNCHERS、STATUS、AGENTS、memory及启动README | 增量记录当前支持及未验证边界，保留旧文档；本报告和命令页新增 |

必要shell目录改动由用户本次第7/8项授权完成；没有把A0建议分配的A3当作自动执行授权。性能工具、matched LRSA、计时/MAC统计本轮全部冻结。

可独立审查的A2 patch：[a2-changes.patch](front_ablation_audit/a2/a2-changes.patch)。相对A2起点的完整文件和AST冻结证据：[freeze.json](front_ablation_audit/a2/freeze.json)。未commit/push/reset。

## C. 参数流、数学及兼容

```text
训练：CLI显式mode > 当前任务model JSON的full
评估：已有run/architecture.json → 读取并验证完整架构
                            → 未指定mode：使用保存mode
                            → 显式mode：必须与保存值一致
      → 原model_kwargs → wrapper架构配置 → core → 所有front blocks
```

CLI支持 `--front-latent-mode` 和 `--front_latent_mode`。六标准任务通过argparse探针区分显式参数与parser默认；工业任务以None表示尚未指定。旧Transolver分支在新模型解析之前返回，原构造语句没有新增kwargs。

只自动恢复新增mode；其它自定义结构/任务条件仍按原协议在评估时重复传入并比较，不新增整套自动配置恢复。日志原有 `CDLNO resolved architecture`、sidecar `architecture` 和 `metadata.resolved_arguments` 都包含有效mode。

A1三公式完全不改：

```text
full:     A=S+FFN1(N1(S)); B=A+SA(Nsa(A)); T=B+FFN2(N2(B))
no_sa:    A=S+FFN1(N1(S)); T=A+FFN2(N2(A))
identity: T=S
```

S/T为 `[B,M,d]`；之后完整Up、点残差与点FFN仍执行；核心输入提升后为 `[B,N,d]`。bridge/rear/CDPA/readout及历史定义未改；没有前段CDPA、额外norm、共享层或跨真实时间缓存。

### 旧与新checkpoint

1. sidecar不再为缺失的普通架构字段填默认值。必须具有全部当前架构字段；唯一例外是18个旧字段完整、`model_name=CDLNO`、`model_version=cdlno-core-v1`、旧 `front-t-after-ffn2-before-up-v1`，且仅缺mode，此时在内存认作full。新版中性history_rule却缺mode、未知/残缺旧架构均拒绝。文件不写回。
2. A1已完成旧18槽pickle恢复及旧front对象缺属性时执行full。本轮真正加载原A1前保存的Car对象/Air列表验证这条路径，不从新对象删属性伪造历史兼容证据。
3. 工业Run在原局部可信 `torch.load(..., weights_only=False, map_location=...)` 后检查：wrapper/core均为完整配置对象，配置一致，front数量匹配F，每个front的实际mode匹配。旧front缺属性只可能与full相容。随后保留原expected模型的strict state_dict校验；只校验，不用新模型替代保存的对象。
4. 对象block mode错误、wrapper/core mode错误、配置被换成残缺字典、删权重、no_sa/identity对象缺mode均拒绝。三种模式正常往返、来源chunk改变可载入；不提供跨模式迁移。F0虽然没有front参数，配置mode冲突仍拒绝。
5. 新train目录仍原子创建且不复用；eval仍产生独立结果子目录，不改训练sidecar/checkpoint。显式save_name与run路径保留现有保护。

## D. 任务×模式×train/eval/checkpoint覆盖

每行的三个模式均执行；“通过”均指本地源码/合成证据。

| 任务 | full/no_sa/identity train构造+loss backward | eval未传mode恢复/显式冲突 | 保存/加载 | 外部接口/条件 |
|---|---|---|---|---|
| Darcy | 全通过 | 全通过 | strict state_dict | x2+fx1，reference替换xy，stem65；输出[B,N,1] |
| Elasticity | 全通过 | 全通过 | strict state_dict | x2、fx=None/placeholder，point FFN；[B,N,1] |
| Airfoil | 全通过 | 全通过 | strict state_dict | 原曲线坐标/点序，fx=None，ConvFFN；[B,N,1] |
| Pipe | 全通过 | 全通过 | strict state_dict | 原坐标/点序，fx=None，ConvFFN；[B,N,1]，默认M32 |
| NS | 全通过 | 全通过 | strict state_dict | fx[B,4096,10]，stem74，每次[B,4096,1] |
| Plasticity | 全通过 | 全通过 | strict state_dict | fx[B,3131,1]，T[B,1]，每次[B,3131,4] |
| ShapeNet-Car | 真PyG全通过 | 全通过 | `model_<epochs>.pth`整对象 | `(cfd,geom)`，x[N,7]→[N,4]；v3/p1 |
| AirfRANS | 真PyG全通过 | 全通过 | member整对象+根目录list | x7+64距离，stem71；[N,4] vx/vy/p/nut |

新A2合成测试使用B2（工业原单图B1）、d8/h2/M4/F2/L8；静态小网格/点集，NS保留4096点，Plasticity保留3131点。工业使用11节点真实Data/Batch。标准TestLoss连接原相对L2，工业执行原train.py的loss赋值AST（不import训练器）。Air默认MSE的backward使用原total_loss，保留mask分项。原Darcy normalizer/decode/梯度项、NS十步循环、Plasticity时间循环、工业变N/多图拒绝及标签泄漏检查复用既有full回归；不声称新模式每一项都重复了完整真实轨迹训练。

| 检查 | 实际结果 |
|---|---|
| A2定向检查 | 8/8，12.053s；[targeted-final.txt](front_ablation_audit/a2/targeted-final.txt)。最终完整套件还包含随后追加的partial_config负向项 |
| 最终完整套件 | **141/141，62.591s，0失败/错误/跳过**；[regression-final.txt](front_ablation_audit/a2/regression-final.txt) |
| 真正修改前full基准 | 12 front/core +2 Car +2 Air全部通过原atol=rtol=0，含state键/初始化/输出/全部梯度/旧对象；[core](front_ablation_audit/a2/full-core.json)、[Car](front_ablation_audit/a2/full-car.json)、[Air](front_ablation_audit/a2/full-airfrans.json) |
| 三原工作目录新进程 | 标准六任务及两工业各三mode，import/构造/save/load/同权重输出精确通过；无exp导入。工业保持稳定类路径 |
| root任务脚本 | 八任务×三mode×train/eval共48条真实parser检查通过；eval用临时真实sidecar，训练/数据入口未调用 |
| 顺序脚本 | 24组/48条命令预览同mode同run通过；[commands.json](front_ablation_audit/a2/commands.json)。没有自动真实训练 |
| shell/Python语法 | 修改shell通过bash -n；共享/入口Python3.10 AST检查通过，仅语法检查不是3.10环境运行 |
| GPU | 最终套件中的既有full GPU/AMP回归实际通过；A2新八任务×mode GPU矩阵未运行，不能将旧full升级为新mode GPU验收 |

本地实际Python3.13.9、torch2.13.0+cu130、CUDA13.0、RTX5090 Laptop，已有PyG可用。未安装/更新依赖；用户远端Python3.10/torch2.11/cu128本轮未访问执行。

实际主命令：

```bash
python -B -m unittest discover -s tests -p test_front_task_modes.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B docs/front_ablation_audit/a2/audit_scope.py
git diff --check
```

旧fixture重放在三个原cwd各自执行，PYTHONPATH明确当前project和repo；调用原 `docs/front_ablation_audit/a1/verify_before.py BASE core|car|airfrans --results .../a2/full-*.json`。BASE为原 `/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0`，没有重写其产物或放宽原容差。

### 首次失败及修正记录

- 新定向测试首轮7方法报告6子例错误、2失败：测试AST执行缺T_in/Deformation常量导致两个时间任务及新进程失败；旧对象输出差2.98e-8源于测试未沿用原math SDPA。补齐原常量、固定原后端后零容差通过，未改模型。记录：[targeted-initial.txt](front_ablation_audit/a2/targeted-initial.txt)。
- 首次完整140测试3错误：旧参数捕获测试用不存在的 `/existing-run`、`/tmp/user-chosen-run`、`/tmp/new-or-existing`；新eval读sidecar正确提前拒绝。仅这三项参数测试mock模式解析；其真实加载检查和A2的48条parser命令使用实际sidecar。记录：[regression-initial.txt](front_ablation_audit/a2/regression-initial.txt)。
- 自审又加入“整对象不得用残缺配置字典”防护和负向项；保留中间完整通过日志 `regression-before-final-review.txt`，最终重跑上面的141套件。没有把中间状态当最终验收。

## E. 冻结证据与命令交付

[freeze.json](front_ablation_audit/a2/freeze.json)对照A2实际起点逐文件hash，并核对wrapper forward、stem/位置/时间/初始化相关构造AST；六exp、四工业main/main_evaluation、train.py、数据/normalizer/loss/time loops/optimizer/scheduler/metrics、原Transolver、依赖、Air params.yaml、核心config/modules/core/CDPA均未改。八份JSON删除唯一新增字段后与起点完全相同。原子项目16脚本及顺序脚本保留字节内容；root四helper只改mode目录逻辑。以前已有未提交修改受到保护。

实际结果：235个起点文件中207个未变；28个已有文件变化恰为7个共享/适配/加载文件、8个JSON、4个shell helper、3个参数测试及6个说明/状态文件。12个实际exp/main/train文件逐字节相同；36个未修改shell脚本保持原内容。三个wrapper文件在AST中仅移除新增mode参数/传递后完整还原A2前源码。`git diff --check`及本轮patch反向 `git apply --check --reverse` 均通过，未实际应用或回退patch。

八任务×三mode的训练/评估和顺序运行示例已写入 [A2命令页](CDLNO_FRONT_ABLATION_A2_COMMANDS.md)，均使用实际存在的 `tran_evaluate/<task>.sh` 和 `train_eval.sh`。不提供不存在的统一train.py。命令页保留Pipe M32、其余M64、各d/h/batch/epochs/lr及工业路径区别。

## F. 自审及剩余边界

已自行完成四项复核：

1. **mode传递与eval优先级**：CLI/JSON→wrapper/core→front，24个任务模式合成实例、48条parser命令，省略mode恢复、显式冲突加载前拒绝。
2. **旧/新checkpoint真实性**：真正改前工业对象/list、旧18槽配置和缺mode sidecar；新对象属性/配置/权重矛盾拒绝；无strict=False或写回。
3. **full与冻结合同**：旧full数值/梯度零容差、预设逐字段一致、训练数据与核心源码冻结，原模型不接新kwargs。
4. **目录及可用命令**：自动消融名含mode，full路径保持，显式路径优先且已有run拒绝；同mode顺序命令通过。根shell按完整拼写或下划线别名解析后缀，建议使用本文完整选项名而非argparse缩写；显式run可用于精确定位。

A2范围未发现剩余已知实现缺陷需用户裁定。仍未验证：远端新模型执行、真实数据读取完整性、真实训练/轨迹、收敛/精度/真实epoch效率、新八任务模式GPU/混合精度矩阵。性能工具front消融参数/成本统计仍未接入，不用本轮合成结果宣称提速。Car固定raw阻力路径/param0/fold0限制及Air Dataset/parent语义保持。

**本补充阶段结束，未执行下一阶段。**
