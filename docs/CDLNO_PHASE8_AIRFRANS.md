# CDLNO 阶段8：AirfRANS

完成日期：2026-09-14。阶段7已获用户审查通过，本阶段仅接入 AirfRANS。

## A. 完成范围

完成 AirfRANS wrapper、训练/评价 CDLNO 选择、必要参数转发、YAML 新模型 key、JSON 起始配置、独立脚本/输出目录以及可信整模型/模型列表 checkpoint 和 sidecar。八任务配置与启动清单见 [CDLNO_TASK_LAUNCHERS.md](CDLNO_TASK_LAUNCHERS.md)。

默认 d256/h8/M64/L8/F2/P6、point FFN、entry/chunk0；输入原 x7 后追加 reference64，stem71；保留原 fx=None placeholder，不注册时间投影。训练协议仍是原 Adam、batch1、398 epochs、lr=.001、subsampling32000、r=.05、max_neighbors64，以及原每epoch采样/验证20次采样/OneCycle/加权表面与体积损失。

未启动真实训练、数据抽样或实际评价；完成的是模型接口、原损失合成连接和加载层面的验证。

## B. 文件、原因与 diff

| 文件 | 变更 |
|---|---|
| [cdlno/airfrans.py](../cdlno/airfrans.py) | 只实现任务输入适配和单图检查；所有 block/CDPA/readout 数学继续复用原共享核心 |
| [models/CDLNO.py](../Airfoil-Design-AirfRANS/models/CDLNO.py) | 导出 Model；实际类为稳定 `cdlno.airfrans.AirfRANSModel`，避免和 Car 的本地 models 包混淆 |
| [cdlno_entry.py](../Airfoil-Design-AirfRANS/cdlno_entry.py) | 新模型参数、YAML/CLI解析、run/member/eval路径、sidecar及可信完整对象加载 |
| [main.py](../Airfoil-Design-AirfRANS/main.py) | CDLNO独立构造分支，避免落入GNN encoder/decoder；成员/列表保存路径及可选score输出路径旁路，原训练调用保持 |
| [main_evaluation.py](../Airfoil-Design-AirfRANS/main_evaluation.py) | `--model CDLNO`/`--task`/已有run选择，sidecar先读、模型列表加载；默认仍Transolver/full |
| [params.yaml](../Airfoil-Design-AirfRANS/params.yaml) | 仅追加CDLNO key，逐字段复制当前Transolver训练字段；已有key和值及原文件前缀保持 |
| [airfrans.json](../Airfoil-Design-AirfRANS/configs/CDLNO/airfrans.json) | 模型架构默认与初始训练字段记录；运行训练字段以YAML加显式CLI为准 |
| [CDLNO.sh](../Airfoil-Design-AirfRANS/scripts/CDLNO.sh)、[CDLNO_Evaluation.sh](../Airfoil-Design-AirfRANS/scripts/CDLNO_Evaluation.sh) | 显式d/h/M/L/F/CDPA及398epoch配置，用户参数置最后；注释说明两种my_path含义 |
| [test_airfrans.py](../tests/test_airfrans.py) | 13项新测试，包含真实PyG和两个原工作目录新进程 |
| [test_shapenet_car.py](../tests/test_shapenet_car.py) | 仅更新冻结清单：阶段8授权的Air入口/YAML改由新测试核验，Car模型/接口测试未改变 |
| 报告、八任务清单、STATUS、AGENTS、memory | 更新阶段8实际结果和后续边界 |

三个原 tracked AirfRANS 文件 diff 合计 **+49/−18行**：main.py +23/−11、main_evaluation.py +16/−6、params.yaml +10/−1。YAML原末尾无换行，追加key使末行在Git显示替换，但原内容前缀逐字节未改；解析验证其余key完全等于基线。

入口加载沿用计划授权的局部可信整模型兼容：仅本项目加载处显式 `torch.load(..., weights_only=False, map_location=device)`。原Transolver列表的评价加载也在其对应位置使用该兼容参数；没有全局环境变量、monkeypatch或allowlist放宽。旧训练脚本、原模型/保存格式继续保留。

## C. 张量、reference 与保存协议

```text
forward(data) -> [N,4]
x[N,7] = [xy2,Uinf2,sdf1,normal2]（原normalizer后的x）
pos[N,2] = 原物理坐标
r_ab = (linspace(-2,4,8)[a], linspace(-1.5,1.5,8)[b])
distance[n,8a+b] = sqrt(sum((pos[n]-r_ab)^2))
features = concat(x, distance) : [N,71]
H0 = Linear_2(GELU(Linear_1(features[None]))) + placeholder[None,None,:]
     [1,N,71] -> [1,N,512] -> [1,N,256]
H0 -> 完整前段/bridge/CDPA/后段 -> readout(HF,ZP) -> [1,N,4] -> [N,4]
输出通道：vx, vy, p, nut
```

reference使用原NumPy linspace→float32、x主/y次序，域不改。它从 `data.pos` 计算距离，不用归一化后x的前两列替代pos，也不替换x7。CPU对照仅移除原get_grid内硬编码的.cuda()设备调用；所有数值操作、点序和原stem同权重输出精确相等。

wrapper读取x/pos及batch/ptr，不读y/surf/edges，不在内部抽样或重排。placeholder按原 `(1/d)*rand(d)` 初始化，stem局部初始化；不递归覆盖core的queries/w。

**原评价元数据差异及处理**：`utils/metrics.py:Infer_test`在克隆单图Batch后按idx切片pos/x/y/surf/batch，但未改ptr。因此当前N可小于ptr末尾original_N。这不是多图。Air wrapper在全零 `[N]` batch且ptr只有 `[0,original_N]`、original_N≥N时接受，保持原对象不变；ptr多于两个元素或batch含非零图编号仍拒绝。没有为此修改metrics、重建图、idx scatter或平均逻辑。测试使用真PyG Batch克隆并按原属性赋值顺序构造这种接口；没有运行实际反复抽样评价。

原训练加权损失保持：

```text
surface = MSE(out[surf,:], y[surf,:]).mean()
volume  = MSE(out[~surf,:], y[~surf,:]).mean()
backward(volume + reg * surface)
```

原test/Results_test选择MSE及后续边界处理不改。新模型仍进入原radius_graph路径，不能将不使用edges解释为删除其成本。

保存协议保持完整对象：原train.py保存每个 `model`，原main.py保存 `models` 列表。新run根目录包含architecture.json和列表文件 `CDLNO`；各次训练的完整模型、日志和图分到 `member_000/model`、`member_001/model` 等，避免原重复模型输出相互覆盖。没有改成state_dict保存，也未引入resume。

加载顺序：先读sidecar → 比较core/wrapper架构和task/nmodel/weight/训练采样字段 → 局部加载可信完整对象 → 校验类、配置、列表长度和strict state keys/shapes → 使用请求的运行chunk。单模型和列表均有加载测试；normal main_evaluation使用列表并保持外层models列表布局。不同task/预算/M等在torch.load前报错，已有sidecar不写回。运行chunk变更不拒绝相同权重。

## D. 实际验证

```bash
python -B -m unittest discover -s tests -p test_airfrans.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator \
  python -B -m unittest discover -s tests -p 'test_*.py' -v
```

最终阶段测试 **13/13通过，8.123秒**；全回归 **107/107通过，0失败/错误/跳过，76.872秒**。日志分别为 `/tmp/cdlno-phase8-focused.log` 和 `/tmp/cdlno-phase8-final-tests.log`。本阶段实际测试没有失败项。测试后补强了新进程检查，使其直接执行原入口提取的CDLNO构造/加载和保存表达式，而非仅单独调用wrapper；最终记录是补强后的结果。

环境：Python3.13.9、torch2.13.0+cu130、PyG2.3.1；本地RTX5090 Laptop。没有安装或替换依赖。原LRSA对照导入会打印缺少可选xformers/liger_kernel的消息，实际同配置SDPA数学测试通过；没有安装这些可选框架。

| 实际覆盖 | 已通过证据 |
|---|---|
| 默认模型、三模式、placeholder与初始化 | d256小N前反向；全部活跃参数梯度存在且有限，w保持零；无time_fc/ConvFFN，front query初始化检查 |
| reference和输入提升 | 原get_grid相同linspace/域/点序与原MLP同权重精确对照，验证71维追加 |
| 真PyG接口/可变N | Data和单图Batch；N9/31及N32000减小d/M；字段值与storage不变；改/删y不影响eval输出，节点置换等变 |
| 单图旧ptr、多图负向 | 原属性切片后ptr保留original_N仍可用；双图Batch、ptr-only、全零batch+多图ptr均拒绝 |
| 原损失backward | 只安全导入定义函数的train.py，调用其train/test处理真PyG合成一批；对照加权surface/volume损失、全部参数梯度和一次Adam/OneCycle更新 |
| 原工作目录独立进程 | 生产/加载两个独立进程，从Air cwd导入稳定Model；实际构造分支不落入GNN；原保存表达式保存两个完整模型及列表；实际eval加载分支恢复，输出一致，chunk0→1，sidecar字节不变 |
| 严格sidecar和隔离 | task/M/nmodel/epoch/错误wrapper在加载前拒绝；缺失sidecar不重建；错误类/列表长度/缺失参数拒绝；新run/member/eval路径独立，旧目录拒绝复用 |
| 默认配置/脚本/路径语义 | 原YAML398保留、新key继承、显式CLI覆盖和未来合法YAML预算优先于parser默认；脚本参数捕获与eval模型选择通过 |
| 冻结/语法 | 两个完整入口AST投影等于原模块；21个非目标Air原文件逐字节不变；Python3.10语法解析、bash语法通过 |

另行从Air cwd以 `PYTHONPATH=/home/hwz/CDLNO python -B -` 运行真PyG单图Batch的CUDA FP32三模式probe（N29/d16/h4/M4/L8/F2），使用原surface+volume损失：输出及所有参数梯度有限。该检查不计入107项；不是AirfRANS混合精度、远端CUDA12.8或完整评价验收。

独立进程的PYTHONPATH只指仓库根；本阶段未执行editable安装，也不称作远端安装通过。所有main/parser/保存表达式测试均为AST提取；没有import运行顶层读manifest的main.py或main_evaluation.py。

**未运行**：真实manifest/VTK数据、每epoch实际采样、验证20次采样、radius_graph实数据调用、Infer_test/Results_test实际反复抽样、idx scatter/平均、PyVista边界/系数后处理、真实训练/评价、准确率或速度比较。上述区域通过源码冻结核查，不能把它们写成端到端已运行。远端Python3.10/3.11、torch2.11、CUDA12.8也未验收；用户Car原训练成功不代表AirfRANS已验证。

## E. 冻结与工作区保护

本阶段开始AirfRANS无现有diff，params.yaml的Transolver配置与阶段0审计/原commit一致，为398 epochs，没有需要覆盖的合法工作区变更。Git仍为main、commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`；保留之前阶段的未提交修改。

未修改Air train.py、dataset、utils/metrics.py、metrics_NACA、原模型或旧脚本。每epoch采样、验证20次采样、radius_graph、idx scatter/累计次数平均、masks、指标与边界后处理的完整文件字节均保留。两个原入口只允许投影新parser/构造/load/path旁路及局部可信load兼容后，完整AST与原模块一致。

阶段7测试中“所有Air原文件冻结”的历史断言只收窄到本阶段授权的main/main_evaluation/params三项，它们改由阶段8完整AST/YAML测试接管；没有停用原Car模型或checkpoint测试。Car实现、六标准任务及数学核心未改。

阶段开始清单在 `/tmp/cdlno-phase8-baseline/hashes.json`：136个已有文件中，7个授权目标发生变化，其余 **129个非目标文件哈希全部不变**；新增wrapper/配置/测试/报告另列入交付。清单SHA256为 `f29f5f98937b7cb46b6a0dcb6c3e7fd581017e342c89e2591d04bb93bc5e3cfa`。未下载数据、生成伪造数据集、安装依赖、真实训练、reset、commit/push或PR。

## F. 交付前自审与待验证边界

已自行核对五项：reference追加/域/点序与placeholder；原抽样单图ptr及多图拒绝；398预算继承和CLI覆盖；两个真实入口分支、整模型/list加载、sidecar先读/目录隔离；图构造及评价处理冻结和my_path区别。对应测试和源码证据均通过，本阶段范围内未发现剩余实现缺陷。

后续需要真实数据与远端目标环境补充运行证据；本次不请求进入该范围，也不执行实际抽样评价。八任务现具备接口及启动配置，不能称作八任务已完成训练或论文指标复现。

**本阶段结束，未执行下一阶段**
