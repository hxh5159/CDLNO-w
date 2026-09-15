# 补充阶段 A1：前段三模式与核心验收

日期：2026-09-14。A0已获用户通过；A1中途暂停用于完成独立启动脚本工作，随后由用户明确恢复。**本阶段完成，待审查；未执行A2–A4。**

结论：共享配置、`LRSAFrontBlock` 和完整 `CDLNO` 核心已支持 `full/no_sa/identity`，默认full。修改前full基准的参数键、初始化、输出、梯度与修改后full逐元素一致；两种消融按整段残差子层删除，不注册闲置参数。最终 **133/133回归通过，0失败/错误/跳过**。八任务wrapper、JSON/CLI、训练/评估入口及启动脚本没有接入新字段，仍构造默认full；不能现在向这些脚本传 `--front-latent-mode`。

## A. 范围与基准

实际HEAD仍为 `769fa333742f73c132868cf560bce5ec21529362`，分支main；原Transolver基线为 `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。原已有远端脚本、顺序启动脚本、A0记录均保留，没有commit/push/reset、依赖安装、真实数据下载或训练。

第一次生产实现修改前，已将实际工作树复制到：

```text
/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0/source
```

同一父目录保留 `start.json`、`resume-start.json` 和 `core/`、`car/`、`airfrans/` 原始tensor/整对象基准。恢复A1时核验了生产源码与该快照一致，已有基准文件SHA256与原manifest一致；没有重造或覆盖旧基准。仓库内持久化 [baseline.json](front_ablation_audit/a1/baseline.json)、三个 `before-*.json`、[捕获脚本](front_ablation_audit/a1/capture_before.py) 和 [回归脚本](front_ablation_audit/a1/verify_before.py)。较大的原始tensor文件放在仓库外，未加入Git；后续A2/A4仍应保留该目录，不能将这些新旧对照样本替换为实现后的构造。

基准配置：CPU FP32、eval、dropout=0、math SDPA、threads=1、确定性算法、TF32关闭；固定seed。每份基准两次输出/梯度完全相同，**修改前已固定 `atol=0, rtol=0`，修改后未放宽**。

| 旧基准组 | 覆盖与留存 | 修改后结果 |
|---|---|---|
| 2个独立front | point/5×7，B2/d8/h2/M4；原state键、权重、构造参数、输入、H_next、T、输出cotangent及全部输入/参数梯度 | full严格加载、初始化顺序、H_next/T及全部梯度零误差 |
| 10个完整core | point/grid下F2/L8的off/entry/every；F0/F3/F4；另含every非零scorer样本 | 原配置、state、buffer、输出/梯度、旧整core对象均通过；受控非零w样本只将w排除在“初始w=0”比较之外，仍比较其加载后数值 |
| 2个旧Car对象 | 真PyG单图N11/17，原 `models.CDLNO.Model`；原18槽配置，front无新属性 | 新工作目录进程直接加载旧对象、full初始化/输出/梯度通过 |
| 2个旧AirfRANS对象及list | 真PyG单图N11/17；原共享类、成员与两成员list | 新工作目录进程直接加载旧成员/list、nonpersistent reference buffer、输出/梯度通过 |

Car最初捕获进程报 `ModuleNotFoundError: models`，原因是脚本所在目录替代了当前工作目录的import搜索路径。恢复后在**生产编辑前**将原Car工作目录加入PYTHONPATH，重新捕获2份成功基准；保留原失败目录及说明。没有用伪PyG模块或修改生产import来修复它。这不是模型数学失败，也没有将失败那次计作通过。

本阶段按用户最新A1范围以完整core为主要基准；工业旧对象额外用于保护共享配置pickle变化。未新增八任务消融训练入口或声称其全部新模式已验收。

## B. 文件与diff

| 文件 | 实际修改及原因 |
|---|---|
| [cdlno/config.py](../cdlno/config.py) | +55/−3：新架构枚举字段/校验、准确的pre-Up历史规则、已知旧full的受限映射、旧18槽pickle恢复及新命名pickle状态；运行配置类未改 |
| [cdlno/modules.py](../cdlno/modules.py) | +30/−12：仅front模式参数、条件注册和三公式forward；旧front整对象缺属性时按full；其余模块数学不变 |
| [cdlno/core.py](../cdlno/core.py) | +5/−4：新字段仅转发给front，绑定准确历史规则；core.forward、bridge/rear/readout构造和CDPA位置/历史循环未改 |
| [tests/test_front_ablation.py](../tests/test_front_ablation.py) | 新增13项针对性测试；含独立显式attention/残差公式、拷贝权重的零分支对照、参数和调用、历史、配置/checkpoint及深度边界 |
| [tests/test_shapenet_car.py](../tests/test_shapenet_car.py)、[tests/test_airfrans.py](../tests/test_airfrans.py) | 只修正原冻结测试的文件清单来源：从当前`git ls-files`改为原commit的`git ls-tree`。没有改wrapper或任务训练测试公式 |
| 本报告、前段审查文档、模块/核心说明、STATUS/AGENTS/memory、启动README状态说明 | 增量记录A1完成、可调用API与仍未接入的边界，保留此前记录 |
| [front_ablation_audit/a1/](front_ablation_audit/a1/) | 起点/基准manifest、重放结果、初次及最终测试日志、GPU记录、冻结清单与可审查patch |

没有修改 `cdlno/checkpoint.py`、`cdpa.py`、共享任务wrapper、三个子项目、八JSON/YAML、16个任务脚本、`tran_evaluate/*.sh`、数据/训练/评价/依赖。配置的现有 `to_dict/from_dict` 已自动进入sidecar比较，无需复制一套加载框架。

本轮patch见 [a1-changes.patch](front_ablation_audit/a1/a1-changes.patch)，将已有生产文件变化与新定向测试列出。冻结及文档增量核对见 [freeze.json](front_ablation_audit/a1/freeze.json)。

## C. 三种计算图及形状

共同输入 H 为 `[B,N,d]`，`Hn=point_norm(H)`；完整Down输出 `S[B,M,d]`，包含原learned query、K/V/O+b与Down Q/K norms，没有query residual。前段实现见 `modules.py:LRSAFrontBlock`（291）、构造（300）及forward（339）。

```text
full:
    A = S + FFN1(N1(S))
    B = A + SA(Nsa(A))
    T = B + FFN2(N2(B))

no_sa:
    A = S + FFN1(N1(S))
    T = A + FFN2(N2(A))

identity:
    T = S

三者共同继续：
    U = H + Up(Hn, up_latent_norm(T))       [B,N,d]
    H_next = U + PointFFN(point_ffn_norm(U))
    return H_next, T                       T:[B,M,d]
```

no_sa不存在 `latent_norm_sa`/`latent_sa` 模块、投影或Q/K norm参数；保留FFN1/FFN2及两个独立pre-norm。identity连FFN1/FFN2及其norm一起不注册，不用 `Identity` 模块替换后再外加残差，没有额外LN/Linear。它返回的T就是同次Down结果的同一个张量对象。

Down/Up与其专属norm和投影保留；Up norm只是Up分支输入处理，不改写历史T。点域FFN的非线性继续执行，规则网格保留普通groups1的3×3 ConvFFN及内部LN。独立参考覆盖5×7非方形网格，实际Conv张量为 `[B,d,5,7]`，保持原行优先点序。

所有更新均为非原地加法；测试在front返回时先保存T快照，再核对CDPA消费、backward与第二次forward后未被修改。历史定义和来源份数不变；无新增前段CDPA或后段AttnRes。

| 默认F2/L8 | full | no_sa | identity |
|---|---:|---:|---:|
| Down/bridge调用 | 3 | 3 | 3 |
| Up/readout调用 | 3 | 3 | 3 |
| latent SA调用 | 8 | 6 | 6 |
| 前段latent FFN调用 | 4 | 4 | 0 |
| 规则点ConvFFN调用 | 3 | 3 | 3 |
| entry逻辑历史份数 | 2 | 2 | 2 |
| every逻辑历史份数 | 27 | 27 | 27 |

上述模块与逻辑来源计数已由新测试实际hook核对。chunk0的entry一次/every六次历史SDPA沿用未改CDPA路径；原full调用spy继续通过，新no_sa/identity验证chunk0/1/2/99同权重输出及梯度。没有把来源批处理解释成减少数学来源或MAC。

每个front参数差额在d8/h2/ratio2的point与grid测试中实际核对：no_sa比full少280，identity少856。与A0公式 `4d²+2d+2d/h`、`12d²+10d+2d/h` 一致；F0不注册任何front参数。全网parameter对象/storage检查不使用默认去重来掩盖共享。

### 配置和兼容协议

`front_latent_mode` 是 `CDLNOArchitectureConfig` 架构字段，默认full；非法字符串/None/bool/数值/容器等由明确ValueError拒绝。配置仍需 `.validate()`，core构造会调用验证。L/F/M、P=L−F、ratio、grid规则未改，F不限制到6。F0三模式相同权重下计算和梯度相同；显式模式仍写入架构，sidecar不因此吞掉差异。

共享API用法（不是任务训练入口）：

```python
from cdlno import CDLNO, CDLNOArchitectureConfig

cfg = CDLNOArchitectureConfig(
    L=8, F=2, M=4, d_model=8, num_heads=2,
    front_latent_mode="no_sa", cdpa_mode="entry",
    structured=True, grid_shape=(5, 7), output_dim=1,
).validate()
core = CDLNO(cfg)
# core接收已经提升的 H0[B,35,8]，返回[B,35,1]；不自行加入位置/时间编码。
```

历史规则的canonical名称变为 `front-t-after-selected-processor-before-up-v1`，准确覆盖三种T位置；没有新增用户需要协调的第二套消融开关。只将已知 `cdlno-core-v1` 的旧FFN2规则/full组合映射到新描述；旧规则与显式no_sa/identity或未知历史规则不被自动接纳。缺mode的已知旧配置识别为full；未知版本缺mode不能推断full。`model_version`仍为`cdlno-core-v1`，核心继续拒绝未知模型版本；配置比较仍保留模型版本等字段，不全局忽略。

旧frozen/slotted配置pickle的18字段顺序显式固定并恢复；新pickle写带版本的命名字段，避免未来插槽错位。旧front对象没有mode属性时仅按其历史full路径执行，不重建或初始化权重。真实旧front/core/Car/Air对象数值已重放通过。直接state_dict必须严格加载；跨full/no_sa/identity缺失/多余key被拒绝，没有 `strict=False` 或正式热启动转换功能。

sidecar保持schema1，读旧文件后在内存规范化，验证不会写回；chunk/device/dtype仍为运行字段，同架构切换允许。旧sidecar字节保持不变、显式模式冲突拒绝、新三模式配置/权重及core整对象往返通过。八任务新模式CLI和工业运行helper对新模式的完整一致性检查属于A2；本轮工业旧对象重放不等于新消融工业训练接入。

## D. 验证命令、实际结果与失败记录

实际环境：本地Python3.13.9、torch2.13.0+cu130、CUDA构建13.0、RTX5090 Laptop。没有调整依赖。这是本地合成证据，用户远端Python3.10/torch2.11/cu128仍未执行此次新模式；Python3.11亦未运行。

实际基准流程是三个独立进程，从各原工作目录调用捕获/重放脚本，PYTHONPATH分别明确指定旧snapshot或当前源根。Car重试额外加入原工作目录以解析`models.CDLNO`。相应manifest记录源路径，防止editable安装将“旧基准”指向新代码。可复查命令例：

```bash
# 以下Car捕获是在生产修改前执行的；产物已存在，不应重新覆盖运行。
cd /home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0/source/Car-Design-ShapeNetCar
PYTHONPATH=/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0/source/Car-Design-ShapeNetCar:/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0/source \
python -B /home/hwz/CDLNO/docs/front_ablation_audit/a1/capture_before.py \
  /home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0 car

# 修改后从当前Car工作目录重放原产物；core/Air各自工作目录另有同类命令。
cd /home/hwz/CDLNO/Car-Design-ShapeNetCar
PYTHONPATH=/home/hwz/CDLNO/Car-Design-ShapeNetCar:/home/hwz/CDLNO \
python -B /home/hwz/CDLNO/docs/front_ablation_audit/a1/verify_before.py \
  /home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0 car \
  --results /home/hwz/CDLNO/docs/front_ablation_audit/a1/full-car.json

cd /home/hwz/CDLNO
python -B -m unittest discover -s tests -p test_front_ablation.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B docs/front_ablation_audit/a1/audit_scope.py
git diff --check
```

| 项目 | 实际结果与证据 |
|---|---|
| 真正修改前full重放 | 12 core/front +2 Car +2 Air，全部旧权重键/初始化/输出/梯度通过零容差；[core](front_ablation_audit/a1/full-core.json)、[Car](front_ablation_audit/a1/full-car.json)、[Air](front_ablation_audit/a1/full-airfrans.json) |
| 首次新定向套件 | 13/13，1.486s；[日志](front_ablation_audit/a1/targeted-initial.txt) |
| 最终综合套件 | **133/133，53.780s，0失败/错误/跳过**；含最终修正后的13新检查及既有120检查，[日志](front_ablation_audit/a1/regression.txt) |
| 原完整core边界 | 既有point/grid各21种F0..6×CDPA模式、L12/F2、L16/F6、L1/F0、F>6/非法配置等测试复用并通过 |
| 新模式核心矩阵 | full/no_sa/identity × (L8/F0、L8/F2、L8/F6、L12/F2) × 三CDPA模式，共36个5×7/B2小模型：形状、参数独立、source数和有限backward通过；另验L1/F0与L8/F0的同权重精确等价 |
| GPU新三模式 | F2/L8/every、5×7/B2、同权重CPU/GPU FP32、mathSDPA/TF32off：全通过；最大输出差4.47e-8、最大参数梯度差5.73e-6，沿用核心GPU容差atol1e-5/rtol3e-4；[结果](front_ablation_audit/a1/gpu.json) |
| PyG与任务/加载旧路径 | 既有full工业PyG、原loss合成步、时间循环/normalizer、三工作目录新进程及checkpoint测试通过；另重放上述真实旧对象。不是八任务新消融集成验收 |
| 冻结审查 | 三个项目、数据/训练/评价、旧模型/脚本、依赖均与A1起点字节相同；现有AST冻结测试通过 |

GPU新模式检查实际通过标准输入Python脚本执行；原检查内容整理为 [check_gpu.py](front_ablation_audit/a1/check_gpu.py)，可用 `PYTHONPATH=. python -B docs/front_ablation_audit/a1/check_gpu.py` 重现。没有进行GPU性能计时、完整矩阵扫描或真实训练。新消融AMP/BF16/FP16未单独验收；既有full精度测试的通过不升级为新模式精度通过。

**首次综合套件有2个错误，已保留并处理：** 133项中两工业冻结检查调用 `git ls-files`，把已提交的新CDLNO文件作为原commit文件读取，`git show`因此找不到 `Airfoil-Design-AirfRANS/cdlno_entry.py`。这两测试在A1开始时就存在且与旧snapshot字节相同，属于旧审查工具对“新增文件后来被提交”的遗漏；不是本次模型回归。最小修复改为从固定原commit枚举文件，所有原文件仍逐字节比较，甚至可检测其后被删除的原文件。初次日志：[regression-initial.txt](front_ablation_audit/a1/regression-initial.txt)，51.380s、errors2；修复后133项全通过，不隐藏或跳过失败项。

参考LRSA测试输出中`xformers`/`liger_kernel`的可选导入提示来自参考checkout，实际同权重SDPA测试继续执行并通过；不是要求为本项目安装这些框架。

### 用户7a–f对应证据

| 要求 | 测试/证据 |
|---|---|
| a 隐式full=显式full、旧full同权重输出 | `test_implicit_explicit_full_same_construction_and_computation`及3组真正旧fixture重放；原full构造顺序严格比较，不要求不同消融同seed初始化一致 |
| b 分支调用数 | `test_exact_sublayer_calls_removed_parameters_and_counts`精确子层顺序/调用及state集合 |
| c identity的T=S且live、Up/点非线性保留 | `test_identity_history_is_live_down_before_up_norm_and_point_nonlinearity`：对象is/值相等、Down query和T梯度、点GELU次数、Conv/LN实际执行 |
| d 固定共同权重的独立公式/零分支等价 | `test_block_formula_outputs_and_gradients_point_and_grid` + `test_zero_output_branches_match_ablations_with_copied_weights`；显式attention参考容差沿用模块atol1e-10/rtol1e-8，梯度2e-9/2e-7；零分支输出精确相等 |
| e 不注册闲置参数、独立、有限backward | 删除参数集合/计数，全网对象和storage独立；36配置backward；零w导致的score-only norm梯度0合法，不判断图 |
| f F0/F2/F6/扩展L，F0同权重等价 | 36核心配置和独立F0同权重输出/梯度检查，既有更宽边界复用 |

## E. 冻结区域与实际未验证项

本次只改共享config/modules/core及测试/说明。三个任务项目、八JSON/params.yaml、原训练/评估循环、loader/采样/点序/normalizer/标签/loss/optimizer/scheduler/metrics、原Transolver以及所有启动shell脚本均与恢复A1时的快照一致。此前启动脚本的既有diff未被撤销或归为本轮修改。没有重新递归初始化wrapper或改保存协议。

实际冻结核查以恢复A1时的216文件SHA256清单为依据：204文件未变，12个已有文件变化恰为3个生产文件、2个测试文件和7个说明/状态文件；无丢失文件或范围外新增文件。三个任务目录分别47/20/29文件、40个shell脚本和6个性能工具文件保持字节相同。`CDLNO.forward`源码及AST、bridge/rear/readout/CDPA构造AST、运行配置类及除front外的19个模块/函数定义AST均未变。[audit_scope.py](front_ablation_audit/a1/audit_scope.py)可重现该核查和源码patch。最初内联核查因Git把中文PDF路径转义为引号字符串而错误判断“新增路径”，改用NUL分隔枚举后通过；没有实际新增或修改该PDF，也没有借此扩大允许变化清单。

仍未运行：八任务新模式wrapper/CLI接入（代码也未实现）、工业新模式全套Run helper接入、原始数据完整性、真实训练/轨迹、收敛/精度、真实epoch速度、远端新模式及新模式混合精度。Car固定阻力路径/fold限制等旧问题保持。性能工具仍只为既有full入口计数；新模式的正式工具参数/成本统计留到指定后续阶段，不用本轮结构计数宣称提速。

## F. 已完成的交付自审

1. **full不漂移**：参数名/路径/创建顺序及权重/输出/梯度通过真正改前基准；不是修改后删属性伪造旧对象。
2. **整子层删除与T位置**：no_sa保留两独立FFN，identity没有latent处理参数；T=S对象和梯度、保留Up norm/ConvFFN有hook及数学对照证据。
3. **历史和独立性**：core.forward/CDPA文件字节未改；新模式history对象/消费时序/快照、两次forward、chunk、source数及全部注册参数独立通过。
4. **配置/checkpoint**：架构与runtime分开；旧JSON/18槽pickle受限恢复、真实旧对象、新模式strict往返、sidecar不覆盖通过。新工业mode入口一致性检查明确仍属A2。
5. **冻结与报告真实性**：首轮旧测试错误及修复如实保留；133最终通过、GPU有限检查、真实数据未运行分别记录，没有新增训练流程或未授权研究机制。

当前A1范围内没有剩余已知架构缺陷需要用户裁定。后续接入/实验仍需用户指定相应阶段，不能根据本报告自动继续。

**本补充阶段结束，未执行下一阶段。**
