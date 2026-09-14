# CDLNO 阶段5：四个静态标准任务

日期：2026-09-13。阶段0–4已获用户审查通过。本阶段只接入Darcy、Elasticity、Airfoil、Pipe；NS、Plasticity、ShapeNet-Car、AirfRANS未接入，等待后续明确授权。依据为用户本阶段指令及v1.2 §§4–6。交付前已自行审查输入提升、冻结区段、配置/加载和初始化，结果见F。

## A. 完成范围与使用方式

四个原exp新增 `--model CDLNO` 分支，模型工厂仍返回带 `Model` 的模块。Elasticity映射到不规则wrapper，其余三个映射到结构2D wrapper；两者共用 `cdlno.standard.StaticStandardModel`，数学计算继续使用已验收的 `cdlno.core.CDLNO`，没有复制核心实现。

新增四份JSON预设、四份训练脚本和四份评估脚本。脚本从任意工作目录定位标准任务目录，并在本次进程设置共享包PYTHONPATH；不安装或更换依赖。所有显式默认参数置于 `"$@"` 前，后附CLI值可覆盖；未指定模型结构值时，新分支从对应JSON采用任务默认，旧parser的3层/旧宽度/M32不会静默覆盖CDLNO。旧模型默认值和构造参数不变。

| 任务 | d/h/M | L/F/P | epochs/batch | 输入提升 | 点FFN |
|---|---|---|---|---|---|
| Darcy | 128/8/64 | 8/2/6 | 500/4 | reference距离64 + coeff1，stem65 | ConvFFN |
| Elasticity | 128/8/64 | 8/2/6 | 500/1 | xy2，经MLP后加placeholder | point FFN |
| Airfoil | 128/4/64 | 8/2/6 | 500/4 | 原xy2，经MLP后加placeholder | ConvFFN |
| Pipe | 128/4/32 | 8/2/6 | 500/8 | 原normalizer处理后的xy2，经MLP后加placeholder | ConvFFN |

共同默认：entry、chunk0、front/rear FFN ratio2、dropout0、lr1e-3、weight_decay1e-5、clip0.1；Darcy unified_pos1/ref8，其他三个unified_pos0。lr仍按原scheduler解释；Elasticity保持CosineAnnealing按epoch step，另三任务保持OneCycle按batch step。Airfoil/Pipe的新heads4和Pipe的M32遵循v1.2已确认任务表；未改原Transolver脚本的heads8/M64。

| 任务 | 新训练脚本 | 新评估脚本 |
|---|---|---|
| Darcy | [CDLNO_Darcy.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Darcy.sh) | [CDLNO_Darcy_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Darcy_Eval.sh) |
| Elasticity | [CDLNO_Elas.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Elas.sh) | [CDLNO_Elas_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Elas_Eval.sh) |
| Airfoil | [CDLNO_Airfoil.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Airfoil.sh) | [CDLNO_Airfoil_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Airfoil_Eval.sh) |
| Pipe | [CDLNO_Pipe.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Pipe.sh) | [CDLNO_Pipe_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Pipe_Eval.sh) |

以下是供远端真实数据可用时执行的示例，本阶段没有执行这些训练/评估命令：

```bash
cd PDE-Solving-StandardBenchmark
bash scripts/CDLNO_Darcy.sh --gpu 0 --data_path /path/to/fno
# 用训练启动时打印的实际目录评估；若训练改过架构，评估必须传入相同结构参数。
bash scripts/CDLNO_Darcy_Eval.sh --gpu 0 --data_path /path/to/fno \
  --cdlno-run-dir /path/to/the/printed/run --cdpa-source-chunk-size 1
```

新训练默认目录：`runs/CDLNO/<task>/<save_name>_<UTC时间>_<随机标识>/`。默认save_name含任务/L/F/M/mode。可用 `--cdlno-run-dir` 指定新的显式目录，但已存在目录会报错，不建立resume机制。同一run按原保存频率更新 `model.pt`，不会覆盖其他run。评估必须明确指定已有run，只读取sidecar/checkpoint，结果写入独立 `eval_<时间>_<随机标识>/`；不会执行Pipe旧分支的resave。

## B. 修改文件与diff范围

| 文件 | 本阶段变化与理由 |
|---|---|
| [cdlno/standard.py](../cdlno/standard.py) | 新增四静态任务输入适配，固定任务fx合同、baseline位置/placeholder语义、局部stem初始化；接入共享核心 |
| [结构wrapper](../PDE-Solving-StandardBenchmark/model/CDLNO_Structured_Mesh_2D.py)、[不规则wrapper](../PDE-Solving-StandardBenchmark/model/CDLNO_Irregular_Mesh.py) | 两个稳定模块路径，各导出Model；不复制数学模块 |
| [model_dict.py](../PDE-Solving-StandardBenchmark/model_dict.py) | 新增合法注册名CDLNO，按四静态任务分派；其他任务不被自动开放 |
| [cdlno_entry.py](../PDE-Solving-StandardBenchmark/cdlno_entry.py) | 新增有限parser/参数转发/输出目录/sidecar与strict checkpoint协议；不读取数据、不实现训练循环 |
| [exp_darcy.py](../PDE-Solving-StandardBenchmark/exp_darcy.py)、[exp_elas.py](../PDE-Solving-StandardBenchmark/exp_elas.py)、[exp_airfoil.py](../PDE-Solving-StandardBenchmark/exp_airfoil.py)、[exp_pipe.py](../PDE-Solving-StandardBenchmark/exp_pipe.py) | 只增加helper import、parser调用、显式新/旧模型构造分支、保存/加载分支和结果路径别名 |
| [四任务预设目录](../PDE-Solving-StandardBenchmark/configs/CDLNO)及上表8个脚本 | 预设负责新模型默认值；新脚本显式指定架构/原训练配置，用户参数置末 |
| [cdlno/config.py](../cdlno/config.py) | 运行配置chunk增加严格非负整数校验，修复上轮已复现的类型缺口，使新入口sidecar与核心一致；无架构字段变化 |
| [tests/test_static_standard.py](../tests/test_static_standard.py) | 13项合成接口、真实N、原损失、初始化、sidecar/新进程、脚本和冻结AST验证 |
| 本报告、[STATUS](CDLNO_IMPLEMENTATION_STATUS.md)、[AGENTS](../AGENTS.md)、[current-state](../memory/current-state.md) | 更新授权、验收与后续边界，记录已完成自审 |

旧tracked文件仅5个发生变更：四exp和model_dict，共239行增加、119行删除。主要行数来自显式保留旧构造和checkpoint代码的else分支；没有重构训练器。其他新文件由本阶段新增，config属于此前未tracked的共享工程，仅修改runtime校验两行。

## C. 输入提升、张量形状与模型合同

共同外部签名 `forward(x,fx,T=None)`，最终返回 `[B,N,1]`，不提前squeeze。实际stem是原MLP的 `Linear(stem_width,2d) → GELU → Linear(2d,d)`，没有额外隐藏层。两层Linear单独按原trunc_normal(std.02)/bias0初始化；wrapper不递归apply，不覆盖核心的queries、CDPA scorer或norm。

- Darcy：`x[B,7225,2]`、`fx[B,7225,1]`。**原结构模型的reference来自固定H×W索引网格**，不是根据exp传入xy重新计算。保留其axis顺序和NumPy linspace→float32转换，得到 `pos[1,N,64]`；替换xy后 `cat(pos,fx)[B,N,65]`。常量pos注册为不进入state_dict的buffer，随模型转device/dtype，并由已校验H/W/ref重新生成。没有placeholder，fx缺失明确报错。
- Elasticity：`x[B,972,2]`、fx=None；`stem(x)+placeholder[1,1,d]`。不reshape成网格，无ConvFFN。
- Airfoil：`x[B,11271,2]`、fx=None，保持221×51索引和弯曲物理坐标；`stem(x)+placeholder`。不排序或重采样。
- Pipe：`x[B,16641,2]`、fx=None，保持129×129索引；原exp先对坐标做UnitTransformer encode，wrapper直接使用该结果，不逆归一化或另建坐标。`stem(x)+placeholder`。

placeholder仅为三个实际fx=None任务注册，初始化为 `torch.rand(d)/d`；Darcy不注册。四静态wrapper没有time_fc，Time_Input=True或传入非None的T明确拒绝；没有提前实现NS/Plasticity。此后有时间条件的wrapper也只在Time_Input=True时注册时间投影。

H0经已经验证的核心：F个完整前段 → bridge → 按mode融合 → P个独立后段 → HF query/点残差readout。T、Z始终 `[B,M,d]`；最终 `[B,N,1]`。规则ConvFFN仅在前段与readout，默认共3次；Elasticity为point FFN。输入提升不改变历史定义或初始化。

## D. 实际验证

命令在仓库根目录执行：

```bash
python -B -m unittest discover -s tests -p test_static_standard.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v > /tmp/cdlno-static-full-tests.log 2>&1
```

本阶段 **13/13通过，0失败/错误/跳过，15.705秒**；最终包含此前阶段的完整回归 **73/73通过，0失败/错误/跳过，26.554秒**。日志分别位于 `/tmp/cdlno-static-tests.log`、`/tmp/cdlno-static-full-tests.log`。

| 实际验证 | 证据与范围 |
|---|---|
| 原入口接口 | AST只提取四exp的parser和实际Model构造表达式（移除最后的.cuda），从未import或执行exp/main；四任务×三模式，默认d128/M任务值、L8/F2小N前反向，全部注册参数及活跃输入梯度非None且有限 |
| 真实N布局 | d8/h2/M4，保持L8/F2：Darcy7225、Elasticity972、Airfoil11271、Pipe16641，输出 `[1,N,1]` 和全部参数/活跃输入反向通过；N为真实接口尺寸，输入完全是合成张量 |
| 原提升数学 | 复制权重到原MLP对照；提取原结构模型get_grid方法、仅去掉强制.cuda用于CPU核对，reference值与顺序逐位相同；HF入口stem结果逐位相同。Airfoil/Pipe用弯曲且未排序的合成坐标验证原顺序 |
| Darcy损失 | 使用原UnitTransformer/TestLoss；AST提取原central_diff和训练中model输出至loss的完整语句，执行decode、裁内部/零padding、原梯度项；rL2与deriv_loss分别对coeff有非零有限梯度，再验证 `loss=rL2+0.1*deriv_loss` 和全模型反向。未写数据文件 |
| 其他损失 | 四任务原TestLoss连接通过；Elasticity/Pipe另验原normalizer/decode；Airfoil保持不decode |
| 初始化/参数 | 三任务有placeholder，Darcy没有；所有任务无time_fc；wrapper/core所有Linear的trunc_normal恰好一次、CDPA w仍严格0；此前核心的独立参数/特殊query测试随全回归通过 |
| 新旧默认/转发 | 新任务预设与任务表一致；显式参数即使等于旧默认也不被覆盖；旧parser所有原字段值相同。旧构造保留在else，未传任何新kwargs |
| strict checkpoint | 四wrapper同权重跨chunk输出通过，故意缺少state key报错；四任务从全新进程加载model.pt及sidecar复现输出，无exp import；atol1e-5/rtol3e-4 |
| sidecar语义 | core的F/L/M/mode/grid不一致拒绝；adapter的ref/version等语义也拒绝，原sidecar和权重字节不变；合法chunk变化允许 |
| 输出隔离 | 自动run目录不重合，显式同目录第二次训练拒绝；评估不保存/覆盖checkpoint，结果目录每次独立；缺少sidecar不创建目录 |
| 运行配置修复 | True/1.5/NaN/Inf/负值/字符串在validate/from_dict/save/load/compare处拒绝；失败不创建或覆盖文件；已有配置的合法runtime变更仍通过 |
| 脚本 | 8个脚本bash -n通过；临时argv捕获器验证cwd、带空格路径、参数覆盖和模型/训练默认值，不执行训练程序，不制造数据文件 |
| GPU | 实际RTX5090 Laptop GPU上四wrapper×FP32/FP16 autocast/BF16 autocast共12组小张量前反向有限；完整回归亦运行此前模块/CDPA/core GPU检查 |

实际环境是现有Python3.13.9、torch2.13.0+cu130、CUDA runtime13.0、RTX5090 Laptop GPU。没有安装依赖。用户远端Python3.10/torch2.11/cu128未执行；Python3.10只做语法解析，不声称远端训练通过。旧model_dict导入timm打印弃用提示，LRSA参考打印缺失可选xformers/liger_kernel，实际对应测试仍通过；没有为这些提示重装环境。

本阶段第一次定向运行有1项测试失败：测试把Darcy原exp的 `H=s,W=s` 误当成5×7，实际小N是5×5；修正参考断言使用实际模型H/W后通过，没有更改Darcy网格。编写有限分支时的一次生成脚本缩进错误在ast.parse前置检查中被捕获，未写入exp文件；修正生成步骤后才落盘。最终无剩余测试失败。

## E. 冻结证据

基线分支main，commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。本阶段开始保存102文件清单和源码副本到 `/tmp/cdlno-static-baseline-0a0j551e/`；清单SHA256为 `f46d355a8906b2b09b17c4f1ffc4d2f7266e8627769f069bd940c57fbc395685`。

允许修改的既有文件为四exp、model_dict、runtime config、AGENTS/STATUS/current-state共9个；其余 **93个基线文件SHA256全部不变**。原tracked71文件中仅5个入口/工厂文件有diff，其他66个不变。NS/Plasticity exp、两个工业目录、原Transolver模型与脚本、normalizer/loss文件、依赖/pyproject、阶段2/3数学模块和阶段4核心均未修改。

冻结AST检查不是只比较若干loss行：对每份exp仅删除新增helper import/局部run变量，把新模型及checkpoint条件还原到原else分支，将结果目录别名还原到原表达式、parse调用还原到原parser.parse_args，得到的**整个模块AST与原commit逐项相同**。model_dict删除新增CDLNO分支后同样与原AST相同。这覆盖读入字段、划分/采样/reshape/点序、normalizer、loss、优化器/scheduler、循环、clip、保存频率及原评价处理。新分支自身只构造模型与运行目录协议，未新增数据操作。

另执行git diff --check、新/修改Python的Python3.10语法AST解析、脚本bash -n、JSON解析和文档链接/空白检查。未下载数据、创建假数据文件、运行真实训练/评价、commit/push、PR或reset。

## F. 已完成自审与剩余边界

1. **输入合同自审通过**：Darcy的reference由原索引网格生成且stem65；其余任务原坐标/placeholder语义保留；Pipe normalizer位于原exp；Time_Input=False无闲置参数。
2. **初始化和共享边界自审通过**：输入stem局部初始化，核心只创建一次，没有递归apply；两个wrapper共享cdlno实现，没有复制数学模块。
3. **默认与冻结区段自审通过**：新model=CDLNO才采用新预设/kwargs；四原exp和工厂可完整AST还原；原模型默认和脚本没有被覆盖。
4. **checkpoint自审通过**：eval先读取已有sidecar再比较，随后strict加载；训练原保存频率不变但run目录隔离。sidecar仍用既有core architecture schema，`metadata.wrapper_architecture`是额外**强制比较的架构语义**，不是可忽略的备注；完整wrapper检查必须走StaticRun，不能只用core compare_architecture就声称适配器兼容。
5. **证据边界自审通过**：当前通过项来自合成输入和实际执行的原loss语句、CPU/GPU前反向与权重恢复；没有把真实N等同于真实数据，没有把新进程构造测试说成跑通训练入口。

本阶段未发现剩余新增实现缺陷。上轮runtime chunk类型缺口已修复并回归。

保留的原有边界：原exp的评估绘图硬编码85×85、221×51或129×129，本阶段按冻结要求未修改；改变下采样后的可视化尚未验收。新脚本允许参数透传，并不扩展原任务所有下采样/绘图组合。远端目标环境、真实数据训练/评估、收敛/精度/速度尚未运行。NS、Plasticity及两个工业任务继续等待后续授权。

**本阶段结束，未执行下一阶段**
