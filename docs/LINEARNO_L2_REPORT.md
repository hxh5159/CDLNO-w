# LinearNO L2 — 独立 attention 原语及数学验收

日期：2026-09-17。**阶段状态：PASS。** 已完成本轮 L2；不包含八层完整网络、任务 wrapper、factory/CLI/训练/eval 接线。`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`；本轮 GPU 仅按 L2 第6项执行合成 attention FP32/AMP 数值验收。没有真实数据、下载、依赖安装、外部 checkpoint/pickle 加载或 commit/push。

## A. 依据与工作区

沿用已接受的 [L0实际仓库/三方审计](LINEARNO_REFERENCE_AUDIT.md)、[论文v3与发布源码复现矩阵](LINEARNO_REPRODUCTION_MATRIX.md)、[L1协议/旧模型回归](LINEARNO_L1_REPORT.md)。当前仍是 `/home/hwz/CDLNO`，`main@bb73b3099d3b8ce45bd939156b737453b9ca5454`。先检查适用AGENTS、Git状态和L0/L1内容hash，再捕获L2前快照 `/home/hwz/CDLNO-artifacts/linearno-l2-before-2cz69qfr/source/`，1223个既存文件。用户PLAN、所有既存untracked/ignored及前阶段证据继续保留。

固定官方 `HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`，三份attention文件与Git固定blob逐字节一致。外部参考仓有用户已有untracked分析报告，本轮未修改。参考来源只有外部文件；没有复制整个仓库，也没有把官方实现加入生产代码。固定树没有LICENSE/NOTICE的来源风险记录继续有效；新原语依据公式独立实现，保留出处。

## B. 新文件、真实类与边界

| 文件 | 符号/职责 |
|---|---|
| `cdlno/linearno/attention.py` | `LinearNOAttention`，六变体共有的投影、轴和因式分解；`initialize_release_weights`为外层一次性初始化callback |
| `PDE-Solving-StandardBenchmark/model/LinearNO_Attention.py` | `LinearNO`、`LinearNO_temp`、`LinearNO_Conv`、`LinearNO_Conv_temp`；各自纯attention包装，**无Model类** |
| `Airfoil-Design-AirfRANS/models/LinearNO_Attention.py` | `LinearNO`，absolute `slice_num`，默认M32；不是Data任务wrapper |
| `Car-Design-ShapeNetCar/models/LinearNO_Attention.py` | `LinearNO`，`M=key_ratio*dim_head`、原温度拼写；不是tuple/Data任务wrapper |
| `tests/linearno/attention_reference.py` | 最先写入的独立oracle；不导入生产attention、不调用待测forward |
| `tests/linearno/attention_support.py` | 外部固定源码hash验证，选择完整原类AST而不改写；不import官方exp/main/Embedding/数据，不构造完整网络 |
| `tests/linearno/test_attention_parity.py` | 初始化/RNG、CPU double/FP32 oracle、官方同权重forward/input&all-parameter gradients/AdamW/strict checkpoint、CUDA FP32/AMP |
| `tests/linearno/test_attention_structure.py` | 轴、dense等价、真实执行shape、batch/点序/卷积/clamp/共享参数/非法配置/gradcheck/三个cwd导入 |
| `tests/linearno/fixtures/official_attention_sources.json` | 固定SHA及三文件原始字节SHA-256；默认外部路径可由`LINEARNO_REFERENCE_ROOT`指定 |
| `docs/linearno_audit/l2/` | 基线、环境、逐参数误差、初始化问题与修复证据、回归日志、冻结及命令 |

其他已存在文件仅 `docs/LINEARNO_IMPLEMENTATION_STATUS.md` 增量前置L2记录。旧模型/attention/Embedding、L1配置与schema、parser、factory、exp/train/eval、checkpoint helper、launcher、数据、依赖、已有测试均未改。根`Physics_Attention.py`不用于新实现；不存在的冻结`LINEARNO/**`仍不存在。

新attention默认返回单个 `[B,N,dim]` Tensor。没有residual、FFN、LayerNorm、placeholder、位置/time、完整block或model；这些属于后续L3。没有mask扩展或图batch变更。生产forward不返回Q/K/C，不缓存这些张量，无CPU同步/retain_grad/诊断副作用。测试只使用临时PyTorch hooks/TorchDispatch读取中间数据，测试结束移除。

## C. 公式 → 官方符号 → 代码 → 测试

定义输入投影 `F=Input(X)`，拆为 `[B,h,N,d_h]`；`M`为**每个head的Q/K特征数**，不是全部head拼接rank，也不等于slice self-attention维度。

\[
Q=\mathrm{softmax}_{M}(F W_q/\tau_q),\quad
K=\mathrm{softmax}_{N}(F W_k/\tau_k),\quad V=F W_v,
\qquad C=K^T V\in\mathbb R^{B\times h\times M\times d_h},\quad Y=Q C.
\]

无温度变体省略除法；温度按下表clamp。没有sqrt缩放、slice-mass归一化、attention-weight dropout、slice SA或N×N生产矩阵。

| 变体/官方对应类 | 输入与M | 温度 | output projection | L2同配置参数数目* |
|---|---|---|---|---:|
| Standard plain / `LinearNO` | 单Linear(dim,inner_dim)，`key_ratio=M` | 无 | Linear+Dropout | 392 |
| Standard temp / `LinearNO_temp` | 同上 | `temperature_q/k`=.5，clamp[.01,1] | 同上 | 398 |
| Standard conv / `LinearNO_Conv` | **单套**dense Conv2d(dim,inner_dim,k3,pad1,groups1)，M绝对值 | 无 | Linear→GELU→Linear→Dropout | 1700 |
| Standard conv_temp / `LinearNO_Conv_temp` | 同上 | .5，clamp[.01,1] | 同上 | 1706 |
| AirfRANS / `LinearNO` | 单Linear，`slice_num=M`（默认32） | 原`temperature`保留，forward不用 | Linear+Dropout | 395 |
| ShapeNet / `LinearNO` | 单Linear，`M=key_ratio*d_h` | `tempreature_q/k`保留拼写，.5，clamp[.1,2] | Linear→GELU→Linear→Dropout | 554 |

*计数来自 **dim12/h3/d_h4/M8** 的单attention；不是论文完整模型参数量。原子接口可分别指定input dim与inner_dim；完整任务结构今后仍按L1的hidden/heads构造。

Q/K/V各只有一份小Linear：`to_q/to_k.weight=[M,d_h]`，`to_v.weight=[d_h,d_h]`；跨head共享，同层Q/K彼此独立。Q/K/V均bias=False，Input/O均有bias。非卷积变体保留官方未调用的`self.dropout`，仅末端output Dropout实际执行；即使p=0也注册。Air保留官方未使用scale/softmax属性以便审查，其forward不使用它们。key映射为**逐键恒等**，两方向全覆盖，无丢key、补权重或strict=False。

| 公式/约束 | 实际符号 | 独立证据 |
|---|---|---|
| input单投影、row-major非方形网格 | `LinearNOAttention.in_project_x/forward` | `test_conv_grid_order_projection_and_noncontiguous_input`：5×7，单偏移卷积权重配Python索引手算；wrongN明确ValueError |
| Q沿M、K沿N | `q_logits.softmax(-1)`、`k_logits.softmax(-2)` | 执行期截获两个真实softmax，分别sum=1，形状均[B,h,N,M]；独立exp/sum oracle |
| KᵀV再Q乘 | 两个显式标轴einsum | factorized与tiny `(Q@K.T)@V`；真实bmm只有(6,4,15)@(6,15,4)→(6,4,4)与(6,15,4)@(6,4,4)→(6,15,4) |
| 方context不是slice SA | `C[B,h,M,d_h]`，测试M=d_h=4 | 无N×N logits，两个softmax无[M,M]输入；合法C方阵不会被误报 |
| 梯度/共享 | 独立q/k/v模块 | all活跃参数finite grad；批0loss对批1input grad严格0；逐head同输入同投影；两实例storage独立 |
| 温度clamp | 两套独立温度key | below/on/inside边界前向与温度梯度对oracle；clamp外0梯度正常；Airtemperature改为1234输出不变且grad=None |
| Pointwise点置换等变 | plain/temp/Air/Car | permutation同顺序输出；conv仅检查网格顺序，不宣称任意点置换等变 |
| 不改变输入/权重/状态 | forward只局部变量 | 输入/参数不变，forward前后attrs不增、无buffer/context持有；Dropout hook仅输出[B,N,dim]调用 |

独立oracle使用batch/head循环、显式减max→exp→sum归一、unfold+矩阵乘替代生产Conv2d、erf形式GELU；保留调用者double。oracle比较把测试中的二维投影权重乘8以充分激活归一化梯度；固定官方parity则使用原发布初始化，不应用该测试缩放。它不调用待测forward/投影模块。五个有代表变体的输入及全部活跃参数另通过double有限差分gradcheck，温度在clamp内部；conv无温度的数学分支已在所有变体oracle比较中覆盖。

## D. 初始化时序与首轮发现

官方原子类先使用nn.Linear/Conv2d的默认初始化；完整发布模型构造所有模块后才`.apply(_init_weights)`，最后创建placeholder。因此新原子构造器**不自行调用apply**。独立使用时：

```python
attention = LinearNO(dim=128, heads=8, dim_head=16, key_ratio=64)
attention.apply(initialize_release_weights)
```

未来L3只在完整model边界apply一次，不能对子模块先apply后再递归覆盖。callback：Linear用官方timm `trunc_normal_(std=.02)`/bias0，LayerNorm weight1/bias0，Conv2d Kaiming normal/bias0；温度不覆盖。L2没有placeholder，未声称其全模型时序已经验收。

首轮确实发现并修复初始化parity失败：新实现使用`torch.nn.init.trunc_normal_`，本机Torch2.13内部选用不同采样算法，虽分布相同，权重及RNG消耗与官方timm不一致。改用**仓库已有依赖**`timm.layers.trunc_normal_`，不修改torch/timm版本。最终六类在原子构造结束及外层初始化结束两处state和RNG均逐位相同。证据 [initialization-investigation.json](linearno_audit/l2/initialization-investigation.json)、保留的首轮failure日志和最终测试；没有放宽容差或掩盖失败。

## E. 数值结果、阈值、环境与命令

完整 [parity.json](linearno_audit/l2/parity.json) 有72条记录：6初始化、24 oracle、42官方parity；每个forward、input gradient、每个参数gradient、每个step后parameter及checkpoint output都列max/mean absolute、max/mean relative和relative L2。比较含B1/2、N7/15、d8/12、h2/3、M4/8；结构及freshcwd另覆盖5×7及Air默认M32、Standard默认M/Car乘数默认。

| 比较域 | atol / rtol | 结果/边界 |
|---|---|---|
| CPU float64 oracle/官方 | 1e-12 / 1e-10 | oracle最大abs8.88e-16；官方0 |
| CPU FP32 oracle/官方 | 1e-6 / 1e-5 | oracle最大abs7.15e-7；官方0 |
| CUDA FP32官方 | 1e-5 / 1e-4 | 六类forward/input&parameter grad/AdamW/checkpoint全部max与mean差0 |
| CUDA FP16、BF16 autocast vs同precision官方 | 1e-5 / 1e-4 | 十二case全部max与mean差0；这不是AMP与FP32逐位等价声明 |
| 初始化/构造RNG、strict加载keys | atol=rtol=0 | 六类全部相同；ShapeNet拼写与Airdead参数都保留 |

官方CPU比较每类double/FP32×dropout0/.25；GPU各precision dropout.25。每次配对forward恢复相同CPU/CUDA RNG，并检查forward结束RNG一致。使用同份权重与同份输入，**不是比较两个随机实例**；AdamW为合成一步验收，不是task训练。PyTorch自动autocast规则按官方保留，没有引入KCDNO强制FP32策略。TF32关闭、compile关闭，CUDA卷积deterministic=True、benchmark=False。

以下对oracle表格取每一类多个case/多个tensor中的最大统计值；相对误差=`abs(error)/max(abs(reference),1e-12)`。接近0的梯度会放大相对数值，应结合绝对阈值判断；没有因相对数值大而单独放宽allclose。

| variant | oracle dtype | max abs | max tensor mean abs | max relative | max tensor mean relative |
|---|---|---:|---:|---:|---:|
| plain | float64 | 3.886e-16 | 1.119e-16 | 5.467e-14 | 4.165e-15 |
| temp | float64 | 3.331e-16 | 1.261e-16 | 1.301e-12 | 1.032e-14 |
| conv | float64 | 7.772e-16 | 1.735e-16 | 4.487e-13 | 5.001e-15 |
| conv_temp | float64 | 8.882e-16 | 2.914e-16 | 8.146e-13 | 4.316e-15 |
| airfrans | float64 | 3.886e-16 | 1.119e-16 | 5.467e-14 | 4.165e-15 |
| shapenet | float64 | 4.441e-16 | 1.156e-16 | 5.714e-13 | 9.882e-15 |
| plain | float32 | 4.768e-07 | 1.630e-07 | 1.048e-04 | 5.921e-06 |
| temp | float32 | 4.768e-07 | 9.593e-08 | 1.497e-04 | 8.034e-06 |
| conv | float32 | 4.172e-07 | 1.738e-07 | 1.626e-03 | 5.987e-06 |
| conv_temp | float32 | 5.960e-07 | 1.639e-07 | 4.057e-04 | 1.056e-05 |
| airfrans | float32 | 4.768e-07 | 1.630e-07 | 1.048e-04 | 5.921e-06 |
| shapenet | float32 | 7.153e-07 | 1.440e-07 | 7.645e-04 | 3.419e-05 |

| 检查组 | 最终执行结果 |
|---|---|
| L2新attention测试 | 14/14方法通过，8.980s；无skip；CPU/CUDA/FP16/BF16全部实际运行 |
| L1全部16回归 | 16/16，47.961s；八旧Transolver同权重输出/key/shape/count、strict checkpoint、原parser及随机流不变 |
| L1引用旧基础与源码回归 | 28方法，66.440s，0failure/0error；1个Air完整采样epoch子案例缺torch_cluster跳过；Car原合成epoch/Air原weighted loss/基础resume/完整旧AST通过 |
| 三原子项目cwd新进程 | 3/3导入，新primitive strict state_dict往返，未importexp/main或读取数据 |

环境：Python3.13.9、Torch2.13.0+cu130、timm1.0.28、einops0.8.2；GPU为RTX5090 Laptop。PyG2.3.1可import，缺torch_cluster/scatter/pyg-lib；未改变依赖。远端Python3.10/Torch2.11/cu128尚未运行，不能把本机parity当远端验收。

```bash
# cwd=/home/hwz/CDLNO；本机已实际执行，源码根须是固定审计来源
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 \
LINEARNO_REFERENCE_ROOT=/home/hwz/LinearNO \
LINEARNO_L2_PARITY_REPORT=/tmp/linearno-l2-parity.json \
python -B -m unittest linearno.test_attention_parity linearno.test_attention_structure -v

# L1全套旧Transolver/随机流/schema回归
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  linearno.test_profiles linearno.test_schema linearno.test_legacy linearno.test_rng -v
```

实际持久日志与精确结果在`docs/linearno_audit/l2/`，回归套件的完整运行命令见`commands.json`。远端运行同样命令，只需把`LINEARNO_REFERENCE_ROOT`设为外部固定SHA源码位置；若参考缺失/hash不符，测试明确失败，不会下载或悄悄跳过parity。旧L1小fixture对初始化hash严格；远端差异应先审查来源，不覆盖fixture让它通过。

## F. 冻结核查与边界

最终检查对照L0 manifest（含ignored）及L2起点：L0原1161文件、重点221文件均未变；L2起点1223文件中仅本STATUS增量，其他1222保持原hash。L1模块/测试/夹具一字节未改。新增文件仅新原语/三benchmark新包装/新测试及本阶段文档证据，未产生新增ignored源码或改变原dirty分类。详见`delivery-check.json`。

未执行：完整八层model/block/任务wrapper、训练/评估入口、真实数据/loader、完整模型论文参数量/指标、完整模型官方parity、全任务GPU训练/AMP scaler、真实optimizer协议/epoch耗时、官方whole-object checkpoint转换、远端2.11/cu128。原有monitor不存在，N/A。Air完整图采样的缺依赖是既有环境边界，不是LinearNO attention失败。

## G. 自审

已按源码独立检查五点：①Q/K轴与跨head共享而非per-head权重；②conv只有一套输入投影且H/W顺序正确；③Airdead温度与Car拼写/clamp；④官方初始化RNG及时序；⑤旧模型/入口/用户dirty和已有夹具保护。公式oracle、固定源码和真实运算trace三种依据相互交叉，不仅对shape。最终没有未解释的parity失败。

L0已列工业paper loss定义、force路径和部分任务协议差异继续待对应L6/L7范围审查；L2不涉及这些决策，未自行改架构。没有新增需要用户决定的attention冲突。

**本L阶段结束，未执行下一阶段。**
