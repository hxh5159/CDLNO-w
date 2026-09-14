# CDLNO 阶段3：独立 CDPA 与数学验收

日期：2026-09-13。阶段0、1、2已获用户审查通过；本阶段只执行用户授权的独立 CDPA 实现及验证。依据为 [v1.2 §3、§4.3–4.4、§8.3](../PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md) 和本轮六条明确约束。阶段4未执行。

## A. 完成范围

实现单次融合模块 `cdlno.cdpa.CDPA`：每历史独立 Cross 对齐、逐 token 来源融合、FP32 depth 子图，以及来源逐个/全部/分块的同权重执行路径。先编写不调用 CDPA 的显式数学 reference，再建立 SDPA 输出和输入/参数梯度对照。

验证使用合成 latent 和独立 CDPA 实例。没有组装 CDLNO、收集前后段历史、创建任务 adapter、接入数据/损失/训练入口或改动依赖。单模块权重往返及现有 sidecar 协议回归不构成整模型 checkpoint 接入。

## B. 修改文件、理由与 diff 摘要

| 文件 | 本阶段变化与理由 |
|---|---|
| [cdlno/cdpa.py](../cdlno/cdpa.py) | 新增唯一的单次 CDPA 实现；复用阶段2 RMSNorm 和初始化/形状辅助函数；只依赖 torch 与标准库 |
| [tests/cdpa_reference.py](../tests/cdpa_reference.py) | 新增独立公式实现，显式 LN、QK、token softmax、AV、O、depth RMS/softmax/RAW 累加；不调用待测模块、其 helpers 或 SDPA |
| [tests/test_cdpa.py](../tests/test_cdpa.py) | 新增21项验收测试，涵盖数学、所有输入/参数梯度、执行布局、精度、模块权重及 sidecar 边界 |
| 本报告 | 新增公式/代码对应、完整 §8.3 覆盖表、实际结果与范围限制 |
| [CDLNO_IMPLEMENTATION_STATUS.md](CDLNO_IMPLEMENTATION_STATUS.md) | 更新阶段0–10状态及阶段3 A–F记录 |
| [AGENTS.md](../AGENTS.md)、[memory/current-state.md](../memory/current-state.md) | 同步阶段3完成待审查、单模块API/精度约定和下一阶段边界 |

阶段1的配置、sidecar、包导出、pyproject/预检脚本，以及阶段2模块和测试均未修改。三个原任务目录没有新增或修改文件。以上4个新增文件和3个文档更新，是相对本阶段开始工作区的变化；这些根级工程文件原本尚未纳入 Git，不能用空的 tracked diff 代替文件级核对。

## C. 公式、形状与代码对应

接口：

```python
from cdlno.cdpa import CDPA

fusion = CDPA(dim=d, heads=h, source_chunk_size=0)
z_next = fusion(z, history)
z_next, alpha = fusion(z, history, source_chunk_size=2, return_weights=True)
```

`z` 和每个历史均为 `[B,M,d]`，`d_h=d/h`。历史是本次调用的 list/tuple；模块在入口取 tuple 快照，不复制历史存储、不 detach、不缓存到成员。所有历史的 B/M/d 和 device 必须与 Z 相同。`source_chunk_size` 是运行参数，构造默认0，可在单次调用中覆盖，不写入 `state_dict`，不改变实例默认值。

| 计算顺序 | 公式与形状 | 实现位置 |
|---|---|---|
| 当前 Q | `Q=LN_q(Z) W_Q`，拆头后 `[B,h,M,d_h]`；每次融合只计算一次 | `forward` 中来源循环之前的 `ln_q/to_q` |
| 第 s 份历史 | `K_s=LN_kv(T_s) W_K`，`V_s=LN_kv(T_s) W_V` | 同一模块的 `ln_kv/to_k/to_v` 对所有来源共享 |
| 历史 token attention | `A_s=softmax_token(Q K_s^T / sqrt(d_h))`，概念形状 `[B,h,M,M]`，最后一维只含这一来源的 M 个 token | PyTorch SDPA，dropout=0、noncausal，无 per-head QK norm |
| 完整历史候选 | `R_s=ConcatHeads(A_s V_s) W_O+b_O`，`[B,M,d]` | `to_out` 后才拆回来源；无 Z 或 T_s residual |
| 当前候选 | `R_0=Z`，不经过 LN、Cross 或 O | `_depth_fusion` 的候选首项 |
| 评分用 RMS | `U_s=R_s / sqrt(mean_d(R_s^2)+1e-6) * gamma_depth` | `depth_norm`，所有来源共享，scale=1、无bias |
| 来源评分 | `e[b,m,s]=sum_d(w*U_s)`，`[B,M,S+1]` | `matmul(keys,w.float())`，w严格零初始化 |
| 来源 softmax | `alpha[b,m,:]=softmax_source(e[b,m,:])` | `softmax(dim=2)`，每个当前 token 一份分布 |
| 输出 | `Z_next[b,m]=sum_s(alpha[b,m,s] * RAW_R_s[b,m])`，`[B,M,d]` | 全部来源对齐完后，只调用一次 `_depth_fusion`；没有外加Z、gate或来源bias |

`LN_q/LN_kv` 是 affine LayerNorm，eps=1e-6、scale1/bias0。Q/K/V无bias，O有bias；四个 Linear 分别一次 `trunc_normal_(std=.02)`、bias0。depth RMSNorm eps=1e-6、可学习scale1/无bias，最后建立严格零w。没有来源专属投影或norm；不同CDPA实例参数存储独立。

因此 w=0 时 `alpha=1/(S+1)`，输出是全部 RAW 候选的均值；S=2是 `(Z+R_1+R_2)/3`。空历史则直接返回同一个Z张量对象，可选alpha为FP32 `[B,M,1]` 的全1；不调用投影、norm或SDPA，也不经过FP32往返。未来核心负责在 F=0/entry 等无历史位置省略模块注册；独立 `CDPA(...)` 实例本身仍有参数。

来源分块布局：一组 k 个历史先形成 `[B,k,M,d]`，折叠为 `[B*k,M,d]`；Q/K/V传给SDPA时为 `[B*k,h,M,d_h]`。Q按相同的 batch优先、source次之顺序扩展，K/V长度始终为M。`chunk=1` 每份来源一次，`0` 一次处理全部S份，正整数k每组最多k份；S=5/k=2实际分为2+2+1。identity不参加Cross分组，所有组完成后统一在S+1轴归一化。stack及Q expand/reshape可能分配临时张量；批处理减少调用数，不减少逻辑来源数或MAC，不声明哪个chunk最快。

depth整个子图显式 `torch.autocast(device_type=Z.device.type, enabled=False)`。候选、depth scale与w转FP32，均方、rsqrt、评分、softmax、加权乘法/求和在FP32执行，最后输出转回Z.dtype；转换保持autograd。即使Cross以FP64执行，depth也按本约定使用FP32，因此FP64测试不是纯FP64全链路参考。

历史可以是不同浮点dtype：例如AMP前段T可能是BF16，而bridge的query residual令Z为FP32。Cross投影前把历史计算输入转换到Z.dtype，原历史存储与梯度连接保留；同dtype时不改变其数值。用混合BF16/FP32历史测试了不同chunk与reference的输出和梯度，以及CPU autocast。这是数值接口处理，不引入模型结构或历史调度。

## D. 实际验证与 §8.3 逐项对照

先写reference并独立执行合成检查，当时 `cdlno/cdpa.py` 尚不存在：验证空历史、两次softmax的归一化、零w均值及显式残差导数1/3，通过。随后新增生产实现与对照测试。reference仅接收参数字典和张量；测试额外把 `CDPA.forward` 与SDPA mock为抛异常，reference仍可正常执行。它不复用生产LN/RMSNorm/融合代码。

最终可重复命令（从仓库根目录执行）：

```bash
python -B -m unittest discover -s tests -p test_cdpa.py -v
```

**实际结果：21/21通过，0失败、0错误、0跳过，用时3.141s。** 首次19项运行通过，补充混合dtype后20项通过，再补充§8.3第14条sidecar协议回归后21项通过。阶段3测试未出现失败；阶段2报告中的早期失败是历史记录，不属于本轮测试。

实际执行环境：Python3.13.9、torch2.13.0+cu130、CUDA runtime13.0、NVIDIA GeForce RTX5090 Laptop GPU。真实GPU可用，执行了FP32、FP16/BF16直接半精度和autocast检查。GPU对照测试临时关闭TF32并在测试后恢复；生产模块不改变backend全局设置。GPU自动SDPA路径和显式MATH路径做了比较，但没有用profiler证明具体命中了哪一种融合kernel。

该环境仅证明上述合成检查实际通过。用户远端的Python3.10、torch2.11/cu128仍是兼容目标，本轮未在该远端执行，也未安装或替换依赖。三个新Python文件另以 `ast.parse(..., feature_version=(3,10))` 通过语法检查；这不等价于Python3.10实际运行验收。

| §8.3 | 状态 | 本阶段证据与后续边界 |
|---:|---|---|
| 1 空历史恒等 | 单次调用通过；模型构造部分待下一阶段 | FP32/FP64/FP16/BF16及各chunk返回原Z，梯度全1；mock证明不调用SDPA/norm/Q。F=0/entry不注册CDPA参数需要核心构造测试 |
| 2 均匀初始化 | 通过 | S=1/2/5及各chunk；O权重置0、bias置非零，验证 `(Z+S*b_O)/(S+1)` 和直接Z导数 `1/(S+1)`，防止漏O、错误残差或归一化values；普通随机O也进入reference矩阵 |
| 3 两个softmax轴 | 通过 | reference显式 `[B,h,M,M]` token权重逐行和1；生产 `[B,M,S+1]` 权重和1且跨当前token可不同；对抗样例证明合并S*M或使用归一化values得到不同输出 |
| 4 reference一致 | 通过 | CPU FP32/FP64 × S=1/2/5 × w零/非零 × chunk=0/1/2/>S，共48例；比较输出、alpha、Z/全部T及全部参数梯度；GPU FP32补充MATH/自动SDPA比较 |
| 5 来源交换不变 | 通过 | S=5非零w下换序、各chunk；输出及梯度不变，alpha历史列同步换序，R0保持首列 |
| 6 历史token置换不变 | 通过 | 各来源内部原T同步token置换，经LN/K/V后等价于K/V同步置换；输出及梯度对照通过 |
| 7 当前token置换等变 | 通过 | 置换Z后输出与alpha同步置换，梯度对照通过 |
| 8 batch独立 | 通过 | B=3整体与逐样本结果、全部梯度一致；仅样本0输出求导时，其他样本Z/T梯度严格0 |
| 9 梯度可达 | 独立融合通过；整网部分待下一阶段 | Z、各T、LN_q/LN_kv、Q/K/V/O、w/depth scale纳入梯度比较。尚未连接前段、bridge、后段及decoder query，不声称这些端到端路径已验收 |
| 10 零w梯度判断 | 通过 | 初始depth scale梯度严格0被正确接受，w梯度非零；按其梯度合成更新一次w后depth scale梯度非零且有限，没有更改初始化或加入训练循环 |
| 11 每层历史时序 | 下一阶段 | 本阶段只证实重复调用不缓存/污染输入或状态；尚无核心来源ID、raw Z0保留、当前不重复、未来/中间状态排除、逐层来源数等测试 |
| 12 精度 | 已测试范围内通过 | CPU零/小/大幅值；实际GPU FP32/FP16/BF16；TorchDispatchMode记录depth均方、rsqrt、评分矩阵运算、softmax及累加的真实输入/输出dtype均为FP32且autocast关闭；退出后外层autocast恢复 |
| 13 执行路径等价 | 通过 | 48例及GPUchunk输出/梯度对照；B>1、S=1/2/5、0/1/2/>S、S5最后非整除组；spy确认Q一次、每份K/V长度M、全部组后depth仅一次；空历史无SDPA |
| 14 配置加载边界 | 独立权重和基础协议通过；入口接入待后续阶段 | 同一state_dict严格加载到chunk0/1/2/8并复现输出；现有sidecar允许chunk/device/dtype/AMP变化，拒绝mode/F/L/M变化，成功与失败后原文件字节不变。未接实际模型/任务的保存或eval入口 |

主要容差与范围：

| 对照 | 实际设置 |
|---|---|
| CPU 48例输出/梯度 | atol=3e-6、rtol=3e-5；观察到最大梯度绝对误差2.17e-5，满足 `atol + rtol*abs(reference)` 的逐元素判据，并非每项绝对误差均小于atol |
| GPU FP32 reference/MATH/自动SDPA | atol=5e-6、rtol=1e-4；不要求逐位一致 |
| GPU FP16/BF16 chunk输出及全部梯度 | 先将输出除以max(1,input scale)，atol=.03、rtol=.04；直接半精度和autocast均检查；这是低精度chunk等价性，不是半精度对任意FP64真值的误差保证 |
| CPU混合历史dtype梯度 | atol=1e-4、rtol=.01，包含BF16梯度转换误差 |
| CPU极端输入尺度 | 0、1e-12、1e-6、1、1e4、1e8；另测w=1e4的饱和softmax；输出与全部梯度有限 |
| GPU低精度极端输入尺度 | 0、1e-4、1、1e4；输出与全部梯度有限 |

极端幅值检查使用有限合成范围及按输入尺度缩放的标量目标；不声称FP32均方可覆盖任意大浮点数、任意输入分布或任意损失缩放。没有裁剪候选、clamp均方或改变公式来扩大范围。

其他已通过项：参数共享/不同实例独立、初始化只一次、非连续输入、B1/M1、输入不原地修改、错误形状/类型/chunk拒绝、新进程仅导入配置/sidecar时不会加载torch。

未运行：远端torch2.11/cu128测试、整网前反向/历史时序、原任务loss连接、实际模型保存/eval入口、真实数据训练、速度/显存/精度收益比较。未导入exp/main/train/dataset等任务入口。阶段2代码未修改，本轮不重复其测试。

## E. 冻结区域及核对证据

分支仍为 `main`，HEAD仍为 `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。阶段开始保存的86项SHA256清单覆盖全部71个tracked基线文件、计划资料、阶段1/2已有核心和测试、pyproject及预检脚本；阶段结束逐项核对 **86/86不变**。

清单为 `/tmp/cdlno-phase3-frozen-hashes.json`，自身SHA256为 `7511a03d5eb63b7c99306c40096b72435b0f93105b1fb068b6625e36ee5102a9`。核对命令：

```bash
python -B - <<'PY'
import hashlib, json
from pathlib import Path
saved = json.loads(Path('/tmp/cdlno-phase3-frozen-hashes.json').read_text())
changed = [name for name, digest in saved.items()
           if not Path(name).is_file()
           or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
assert not changed, changed
print(len(saved), 'frozen files unchanged')
PY
git diff --name-only HEAD
git diff --check
git status --short
```

`git diff --name-only HEAD`为空、`git diff --check`通过；由于新增工程尚未tracked，另对本轮7个交付文件直接检查末尾换行/行尾空白、Markdown相对路径，并对3个Python文件检查3.10语法。原任务模型、数据读取/字段/划分/采样/点序/归一化/标签、loss、时间循环、优化器/调度器、评价和依赖均不变。没有commit/push、PR、训练或安装操作。

## F. 未解决问题与优先审查点

本阶段范围内未发现需要改变已确认数学定义的冲突。计划中F=0省略参数、逐层历史来源与整网梯度属于核心组装；按用户明确阶段边界留待下一阶段。远端运行与实际任务checkpoint接入仍未验证。

建议优先审查以下5点：

1. `R_s` 是否完整经过O+b，R0是否严格原Z；零w是否为均值且没有外加残差。
2. `[B,k,M,d]→[B*k,h,M,d_h]` 的来源/样本布局、Q一次及跨全部来源统一softmax。
3. FP32 depth子图的真实dispatch证据、RAW values，以及零w时depth scale零梯度的判断。
4. 独立reference与全部梯度对照的容差、极端幅值范围和混合历史dtype转换。
5. §8.3第1/9/11/14条的模块/协议通过项与核心/入口待验收项，避免将本轮结果当作完整CDLNO验收。

**本阶段结束，未执行下一阶段**
