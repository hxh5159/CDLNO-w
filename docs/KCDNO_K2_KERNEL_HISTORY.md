# KCDNO K2：独立核历史 Writer/Reader 与数学核对

2026-09-15。K1已由用户审查通过；本轮仅K2。**独立Writer、Reader及数学reference已实现并验证；没有组装KCDNO block/core，没有任务训练接入，没有修改旧CDPA。**

## A. 范围、依据与起点

按本轮精确公式和 `PLAN_KCDNO/KCDNO_Model_Specification_v1.md` §§4–6执行，保持已批准K1固定配置与旧CDLNO行为。实际核对根 `AGENTS.md`、K1报告/STATUS、配置和已有RMSNorm/CDPA及其reference/数值测试。无新的mask需求或图batch合同；全部M个latent有效，不增加padding接口。

分支 `main`，HEAD仍为 `222647fef343e9fe929e65412f5c7449ced5517b`。**起点已有未提交K1成果**：KCDNO STATUS/memory修改、新子包配置、K1报告/证据和测试；与K1当时“起点干净”属于不同时间点。本轮保存并保护这些工作，不commit/reset。403个已有文件（含未跟踪K1文件和已有K1日志）完整快照在 `/home/hwz/CDLNO-artifacts/k2-before-98ze33eu/source`，见 [before.json](kcdno_audit/k2/before.json)。

本轮新增生产模块 `cdlno/kcdno/history.py`，使用现有 `cdlno.modules.RMSNorm`；新Wq/Wk单独Xavier初始化。测试reference首先独立编写，随后实现生产路径。核心层号/历史列表生命周期/哪些层创建模块仍属于K3。

## B. 文件、接口与diff

| 文件 | 实际新增/修改内容 |
|---|---|
| `cdlno/kcdno/history.py` | 新 `KernelHistoryCache`、`KernelHistoryWriter`、`KernelHistoryReader`；唯一新增生产数学文件 |
| `tests/kernel_history_reference.py` | 新dtype保留的逐来源缓存reference及显式QKᵀ reference；不导入生产代码 |
| `tests/test_kernel_history.py` | 新16项针对性模块测试、double gradcheck、算子dtype/autocast/shape观测 |
| `docs/kcdno_audit/k2/` | 起点、实际测试日志、旧模型回放、冻结hash、代码diff及结果摘要 |
| 本报告、`KCDNO_IMPLEMENTATION_STATUS.md`、`memory/current-state.md` | 新报告与增量状态；旧记录保留 |

[代码diff](kcdno_audit/k2/code-changes.patch)、[冻结结果](kcdno_audit/k2/freeze.json)、[完整命令](kcdno_audit/k2/commands.txt)。现有子包 `__init__.py`保持纯配置导入，不在根路径放占位模型。实际导入为：

```python
from cdlno.kcdno.history import KernelHistoryCache, KernelHistoryWriter, KernelHistoryReader

writer = KernelHistoryWriter(dim=128, kernel_rank=16)  # 某一个源层独立拥有
reader = KernelHistoryReader(dim=128, kernel_rank=16)  # 某一个接收层独立拥有
cache = writer(T)               # T: [B,M,128]
Uhat = reader(U, (cache,))       # U/Uhat: [B,M,128]
Uhat, alpha = reader(U, (cache,), return_weights=True)
# alpha: FP32 [B,M,2]，来源0是当前U。
```

默认Reader只返回所需Tensor；权重仅在显式 `return_weights=True` 时返回，不保存到模块。Writer只返回memory/mass，不返回或存储T/K；autograd自然保留计算所需的图，不detach。cache是NamedTuple，禁止替换字段；调用方也必须遵守不原地改写内部Tensor的约定。Reader把来源序列转成tuple快照，不持久保存。

Writer/Reader只需要dim、kernel_rank，没有heads参数，r是单组总rank。Reader只接受`KernelHistoryCache`摘要，不能把T交给它重算K。memory必须为FP32[B,r,d]，mass必须为FP32[B,r]，与U同device；错误shape/dtype/原T输入清楚拒绝。所有源的M一致由未来core配置保证，摘要本身不添加token长度/padding字段。

## C. 公式 → 代码 → 测试

| 要求/公式与张量轴 | 代码位置 | 实际测试及结果 |
|---|---|---|
| K=phi(RMSk(T)Wk)，[B,M,d]→[B,M,r] | `history.py:48 KernelHistoryWriter`，forward第73行 | `test_double_cached_equals_explicit_kernel_and_all_gradients`、`test_production_fp32_against_double_oracle`：前向与输入/Wk/norm梯度通过 |
| memory=KᵀT [B,r,d]，mass=sum_token K [B,r]，value为raw T | Writer.forward第74–76行；`KernelHistoryCache:25` | double缓存/显式12组合通过；小分母数值检查区分sum/mean及raw value；无Wv/Wo参数 |
| Q=phi(RMSq(U)Wq)，每次融合一次 | Reader.forward第125行 | `test_one_query_no_rekey_no_token_matrix_and_single_source_softmax`：Reader实际F.linear调用1次，禁止Writer/SDPA调用后仍通过 |
| 每来源R=Q memory/(Q mass+eps)，输出[B,S,M,d] | 第126–130行：memory stack[B,S,r,d]、mass stack[B,S,r]，`einsum('bmr,bsrd->bsmd')`和`einsum('bmr,bsr->bsm')[...,None]` | B1/B2及S1/3，逐来源reference、单样本拆分与样本间零交叉梯度通过；避免把[B,M,r]@ [B,r]当逐batch向量乘法 |
| 候选[U,R1…RS]，当前不kernel self-read | 第131行，[B,M,S+1,d] | 参数清单/一次Q/不rekey检查；uniform系数及非均匀训练后评分对照通过 |
| score=w·RMSdepth(raw候选)，alpha沿source softmax | 第132–133行，alpha[B,M,S+1] | 实际只1次softmax，诊断shape[2,11,4]；不同样本/当前token权重不同；来源换序时权重对应重排、输出不变 |
| C=Σalpha·RAW候选 | 第134行 | 独立reference用原候选逐来源求和；生产/参考全梯度一致；评分norm不被当作value |
| Uhat=U+gamma(C-U) | 第135–136行，最终转回U.dtype | 初始gamma.1、w0系数通过；gamma=-.5/0/1.5均按同一无约束公式，未变成sigmoid/多余残差 |
| 空history严格U | 第118–121行 | B1/B2×FP16/BF16/FP32/FP64：同一Tensor对象，dU=1，无norm/Linear或参数梯度；空reference也验证 |
| phi=ELU+1、clamp1e-6、deneps1e-6 | `_phi:43`、Reader第130行 | 强负投影-100使ELU+1舍入到0时clamp生效；下文小分母案例通过 |
| Wk/Wq biasFalse、Xavier gain1；norm scale1/no bias；w0、gamma标量.1 | 两构造器第57/89行 | `test_initialization_uniform_depth_raw_values_and_unconstrained_gate`核对实际init调用/参数键/shape/范围；没有普通Linear递归reset覆盖新投影 |
| 不共享参数、不原地改cache/输入 | 两独立类和局部tuple | 参数对象/storage唯一，输入和cache值/版本不变，模块无cache/debug buffers；A→B→A同U输出精确重复 |
| 源writer一次写入可供独立receiver复用 | cache包含有grad_fn的memory/mass，无writer指针 | `test_one_source_write_can_feed_independent_readers_with_live_gradients`：一次write、两个独立reader，联合loss对最早T/Wk梯度与double重算reference一致 |

源码行号对应本K2交付版本。两个独立模型层之间复用摘要，不等于共享writer/reader的参数对象。Reader内部无K/V/O投影、无sqrt(d)/sqrt(r)缩放、无token softmax、无current kernel self-read；M×M正值核矩阵只在测试oracle的`explicit_tokens=True`路径出现。

### 初始权重与梯度解释

S份历史共有S+1个候选。w=0使alpha严格均匀 `1/(S+1)`，gamma=.1后的显式当前系数是`.9+.1/(S+1)`，各历史为`.1/(S+1)`。测试S1/S3逐项验证；每份历史内部的正值核token权重通常不均匀，测试也确认这一点。因此不把“深度均匀”写成“核token均匀”，也不把显式系数当完整Jacobian。

隔离Reader输出构造合成loss，没有point旁路。最早T、其Wk、当前Wq、w和gamma得到有效有限非零梯度；初始w=0时depth norm scale的首步梯度严格为0属于正确结果。一次手动沿w梯度更新后，depth norm梯度变为有限非零。极端clamp饱和时某些投影梯度也可为0，不要求每个参数每步非零。

## D. FP32生产与double oracle的边界

生产Writer将T转为FP32；Reader将U转为FP32。norm、Q/K投影、ELU+1/clamp、摘要乘法/sum、stack后读出、分母加eps/除法、depth均方/norm/评分/softmax/RAW累加及gamma融合全部包在 `torch.autocast(device_type=..., enabled=False)` 中；投影权重、scorer、gamma也显式float，norm按传入FP32计算。返回的摘要和alpha保持FP32，Uhat转回U.dtype，所有cast保持梯度。

这是生产FP32合同：即使模块`.double()`也不是float64核数学实现。低精度参数存储也会升为FP32参与运算；其参数梯度回到存储dtype，不能据此声称低精度存储与FP32存储逐位相同。

reference在 `tests/kernel_history_reference.py`，不导入生产Writer/Reader；norm、phi、按来源循环、cache读出、稳定source归一化与raw融合独立写出，**没有.float()强制转换**。测试先将同一输入/参数以有梯度cast转double，再调用oracle；两个double形式只区别矩阵乘法结合顺序：

```text
cached:    Q(KᵀT) / (Q sum(K) + eps)
explicit:  [(QKᵀ)/(sum_token(QKᵀ) + eps)] T
```

二者不是与softmax attention比较。float64 gradcheck使用两个小图（cached/explicit），所有可微输入、Wq/Wk、三个norm scales、w、gamma均参与；投影值刻意大于.5，远离ELU的0和clamp边界。禁止生产forward/SDPA仍能运行reference，防止自证。

| 数值检查 | 实际设置/容差 | 实际结果 |
|---|---|---|
| double cached vs explicit | B1/B2；(M,r,d)=(1,1,4)/(5,3,8)/(3,7,6)；S1/S3，12组合；atol2e-11/rtol2e-10 | 输出/alpha/全部输入及参数梯度通过，最大梯度绝对差2.66453526e-15 |
| double gradcheck | cached和explicit各一次；eps1e-6、atol1e-5、rtol1e-3 | 2/2通过；不对生产强制FP32函数做double gradcheck |
| 生产FP32 vs explicit double | 上述12组合×初始/非零scorer，共24；atol5e-6/rtol1e-4 | 输出/alpha/全部梯度通过，最大梯度绝对差1.43051147e-6 |
| CPU BF16 autocast | FP32输入和BF16输入各1例 | 实际dispatch关键op全FP32、autocast关闭；输出/梯度通过 |
| 本机GPU FP16/BF16 autocast | 每种AMP精度×FP32/对应低精度输入，共4例 | 实际dispatch与输出/梯度通过；最后CUDA同步；不是计时/性能实验 |
| AMP容差 | FP32输入同5e-6/1e-4；低FP16输入atol/rtol .003；低BF16 .02 | 包含输出/输入梯度的存储量化，不把容差扩大用于纯FP32判定 |
| 混合历史dtype/非连续输入/half参数存储 | U FP16/BF16/FP32/FP64；T交替FP16/FP64；half/BF16模块存储 | cache始终FP32、输出回U.dtype，梯度有限且参考通过；此补充混合测试容差.005或BF16 .03，独立于严格FP32矩阵 |
| 极端有限输入 | CPU缩放0、1e-30、1e-12、1e-6、1、1e4、1e8、1e15；loss先按幅值归一化 | 输出/alpha/memory/mass/全部梯度均无NaN/Inf，mass严格正 |
| clamp/极小denominator | M5/r3/d4，T=2、U=1，投影weight=-100 | mass=5e-6，memory=1e-5，Qmass=1.5e-11；R=2×1.5e-11/(1.5e-11+1e-6)；输出/梯度有限且数值一致 |

浮点限制：测试到1e15幅值缩放不保证任意有限输入/任意训练后参数都不溢出。复用RMSNorm直接计算平方均值，极大幅值可能超出FP32范围；half返回也受half范围限制。未添加nan_to_num、归一化缓存或改变sum/eps来隐藏问题。clamp/eps保证正常有限核特征下的正分母，不是任意数值故障的修复器。

实际环境为Python3.13.9、torch2.13.0+cu130、CUDA build13.0、RTX5090 Laptop，PyG仍使用旧可用环境。测试子进程固定CPU单线程，K2精度核查TF32关/compile关并恢复原设置；生产不暗改全局TF32。目标远端Python3.10/3.11、Torch2.11、CUDA12.8本轮未执行，依赖未安装/替换。

## E. 执行证据与冻结范围

| 项目 | 结果 | 本轮验收范围 |
|---|---|---|
| 新模块单元测试 | 16项全部通过 | 单次writer/reader、reference、梯度、置换/batch/初始化/精度/缓存合同；未组装模型 |
| 最终模块套件 | **68/68通过，4.169s，无失败/错误/跳过** | 新16 + K1配置13 + 旧CDPA/基本模块39；既有测试不改 |
| 原入口/训练冻结检查 | **7/7通过，0.444s** | AST与原模型/数据/采样/损失/时间/指标等冻结检查 |
| K0同权重回放 | **41/41通过** | 33旧CDLNO CPU+8原Transolver GPU，旧工业真实PyG对象及整对象/list加载；不是新KCDNO任务验收 |
| 旧K0夹具完整性 | 136文件hash保持 | 未重建/覆盖旧权重/输入/输出基准 |
| 源码冻结 | 403已有文件中401字节相同 | 仅KCDNO STATUS和memory增量记录；含全部现有生产模型/训练/数据/依赖、旧CDPA、K1配置/证据、旧测试和启动脚本 |
| Python3.10语法 | 3个新Python文件通过 | 仅AST语法，不冒称远端环境执行 |
| 真实数据、新KCDNO任务loss/训练、完整model checkpoint、准确率、收敛、epoch效率 | **未执行** | 没有数据下载、真实训练或K3+接入 |

实际命令（仓库根目录；不import exp/main）：

```bash
PYTHONPATH=tests:. python -B -m unittest test_kernel_history test_kcdno_config test_cdpa test_modules -v

python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result /tmp/k2-old-model-replay.json
```

第一条也可在远端现有环境运行：GPU可用时实际执行FP16/BF16 AMP；无CUDA时新GPU测试明确skip。它不跑数据集。全部冻结测试原命令见[commands.txt](kcdno_audit/k2/commands.txt)；[最终日志](kcdno_audit/k2/final-tests.txt)、[冻结日志](kcdno_audit/k2/frozen-tests.txt)、[旧回放](kcdno_audit/k2/old-model-replay.json)、[汇总](kcdno_audit/k2/summary.json)。

首次15项测试全部通过。交付前自审发现reference空历史的`new_ones`尺寸调用未被该首轮覆盖，已修正为tuple尺寸并添加空oracle验证；另补一次write供两reader的梯度检查。最终16项连同既有套件通过；没有依靠改生产数学或放宽FP32容差解决失败。

## F. 自审与K3边界

已自行核对五点：

1. **参数归属/计算路径**：Writer仅key norm/Wk，Reader仅query norm/Wq/depth norm/w/gamma；实际一次Q，无reader rekey、无额外value/output投影或token softmax。
2. **batch/source轴**：显式einsum保留[B,S]，单样本拆分/交叉梯度及三种置换通过；统一来源softmax，不预合并cache。
3. **精度/reference独立性**：实际算子dtype/autocast观测通过；double两形式与gradcheck独立，生产FP32容差单列；oracle空历史边界已补查。
4. **初始化/梯度/不可变性**：Xavier gain1、biasFalse、w0/gamma.1/norm1、RAW系数通过；最早源梯度可达，零w的norm零梯度正确；cache/输入未原地改、参数对象/storage独立。
5. **旧行为/授权**：既有K1未提交成果保持，旧CDPA及全部入口冻结，41份同权重回放通过；无K3架构或任务训练功能混入。

当前没有待裁定的公式冲突或已发现待修复生产问题。以下属于下一阶段，不能由本次单次融合验收替代：按层创建/省略Writer与Reader、L1/off不注册闲置history参数、L8/all的7写7Q28读、每forward局部tuple历史、FFN1/FFN2插入时机和pre-Up T、activation checkpoint重算期间的历史快照、NS真实时间调用隔离。独立Reader被手动构造后即使空历史仍拥有参数；K3负责不创建不活跃模块。

没有新增mask/padding、图batch方式、窗口/分块/流式softmax、完整模型、训练入口或新resume。本K阶段结束，未执行下一阶段。
