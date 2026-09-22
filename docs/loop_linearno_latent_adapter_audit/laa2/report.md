# LAA2：独立 latent / bilateral QK adapter 原语

**状态：PASS，仅原语与合成验证范围。LAA3未执行。**

## A. 授权、来源与边界

当前 `/home/hwz/CDLNO`，branch `main`，HEAD `c02e671506f706910e0a1d58f03c310abf188345`，origin `git@github.com:hxh5159/CDLNO-w.git`。用户只授权LAA2；依据冻结提示词§2.4/2.5/4.1/LAA2、已通过的LAA1合同、实际v2 `LatentContextFFN`实现及当前旧模型源码。

起点：tracked2130、untracked650、ignored1352，4132个已有文件hash留档。保留用户 `check_checkpoints/check_pipe_loop_resume.sh` 原改动。未reset/clean/stash/checkout/rebase/commit/push，没有安装依赖、读取真实数据或启动真实训练。

实现仅在 `cdlno/linearno_loop/v3/` 新增独立原语。没有修改纯LinearNO/v1/v2/三残差、任务入口、schema、metadata、版本分派、模型wrapper、完整loop、checkpoint backend、launcher、记录器或可视化。V3尚不能通过生产命令训练；第二轮调用时机、base logits相加后温度/softmax、KtV插入位置归后续LAA3/LAA4。

## B. 文件与增量

| 新增文件 | 原因 |
|---|---|
| `cdlno/linearno_loop/v3/__init__.py` | 独立模型原语命名空间；不自动导入wrapper/生产入口 |
| `cdlno/linearno_loop/v3/adapter.py` | `BilateralQKLowRankAdapter`，独立Q/K低秩delta |
| `cdlno/linearno_loop/v3/initialization.py` | feature seed、CPU/CUDA设备及dtype合同 |
| `cdlno/linearno_loop/v3/latent.py` | 返回原v2类的隔离seed工厂，未重写forward |
| `tests/loop_linearno_latent_adapter/primitive_oracles.py` | 完全独立的索引/einsum/合并矩阵adapter oracle和显式LN/erf latent oracle |
| `primitive_support.py` | 确定性输入与误差记录，不生成expected数学结果 |
| `test_laa2_oracles.py`、`test_laa2_adapter.py`、`test_laa2_latent.py`、`test_laa2_rng_amp.py` | 手算、前反向/有限差分、隔离、AMP和strict reload测试 |
| 本目录的manifest/log/JSON、`run_checks.py`、报告 | 不覆盖LAA0/LAA1及LL/LF证据 |

既有文件只更新独立 `docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md`，原文保留。完整新增代码/测试diff见 `source-diff.patch`；源码位置与hash见 `source-map.json`。

## C. 公式 → 实现 → 测试

输入 `X[B,h,N,dh]`；每实例独立拥有 `A_q/A_k[r,dh]` 和 `B_q/B_k[M,r]`，四个Parameter，无head维、bias、gate或buffer。

\[
\Delta Q=((X A_q^T)B_q^T)\alpha/r,\qquad
\Delta K=((X A_k^T)B_k^T)\alpha/r.
\]

`adapter.py:28`检查dh/M/r/alpha/feature_seed；`:45`以私有CPU Generator对A做Kaiming uniform（a=sqrt5），B严格置零；`:74`仅执行两端各两次 `F.linear` 和固定alpha/r乘法。返回两个 `[B,h,N,M]` tensor。不同core未来由调用者创建不同实例并分配可复现seed；该原语没有round counter，也没有自行共享core的功能。

`adapter_indexed`按b/h/n/m/r/d逐项求和，`adapter_einsum`用三输入收缩，`adapter_merged_matrix`先求BA再施于X。这三种oracle均不导入生产类、helper或调用forward。手算例使用alpha=3、r=2，对两条输入和Q/K各三输出逐值验证，能区分遗漏alpha/r、两端混用等错误。VJP比较输入和全部四个矩阵；gradcheck以中心有限差分检查全部输入/参数，不只检查loss finite。

latent保持冻结公式：将 `[B,h,M,dh]` 合并为 `[B,M,H]`，H=h×dh，然后

\[
Z'=Z+W_2\operatorname{GELU}(W_1\operatorname{LN}(Z)),\quad\epsilon=10^{-5},
\]

两Linear含bias；无dropout，无M轴混合。`v3/latent.py:12`直接构造v2 `LatentContextFFN`，调用原 `initialize_release_identity()` 后返回同一个原类，未子类化、未包裹、未修改state keys。v2源码 `latent.py:21/27`分别定义初始化/forward，文件hash保持不变。

latent oracle采用逐head拼接、手工均值/方差/LN affine、`0.5*x*(1+erf(x/sqrt2))`，最后按channel切片重建heads；不调用生产LN/Linear/GELU模块或forward。独立标量GELU(1)数值验证oracle自身，再比较非零W2状态下所有输入/参数VJP与有限差分。

| 需求 | 测试与证据 |
|---|---|
| Q/K独立、head共享、batch/N隔离 | adapter手算/三oracle；head逐项/单点计算；只改Q矩阵不改K；只改一个输入位置，其余输出不变 |
| B=0函数恒等与梯度启动 | base logits+delta逐位等于base；首步A和输入delta梯度0，Q/K两B非零；AdamW一步后A/B及输入均有有限非零梯度 |
| latent合并拆分、tokenwise | 非contiguous context、跨head各异数值、M置换和逐token计算；单token修改不影响其他token/batch |
| W2=0函数恒等与梯度启动 | 输出/输入梯度等于identity；首步W2/b2非零，LN/W1/b1零；优化一步后所有参数可学习 |
| 原v2等价 | 返回对象类型完全相同；同seed的全部key/value、输出、输入和参数梯度逐位相等 |
| 无跨forward状态 | 不同B/N/M连续独立反传、state不变、无buffer或激活属性、weakref释放输入图；异常/非有限输入后正常调用通过 |
| strict reload | CPU字节流weights_only state_dict加载；缺key/多gate失败；CUDA优化一步后reload输出逐位相同 |
| RNG隔离 | Python/NumPy/Torch CPU及所有可见CUDA状态前后相同；四消融插入前后公共随机tensor相同；非法构造也不推进RNG |

## D. 初始化、dtype与非法输入

adapter使用局部 `torch.Generator(device='cpu').manual_seed(feature_seed)`，不seed公共generator。latent为精确复用旧初始化，在 `torch.device('cpu') + torch.random.fork_rng(devices=[])` 内仅seed CPU default generator，构造并调用原release初始化，退出恢复CPU RNG；随后转换到目标device/dtype。即使外部default device是CUDA，特性默认仍在CPU初始化，不消耗公共CUDA RNG。

工厂须在未来公共树完成初始化后安装，后续不得再用全树apply覆盖W2/B零值；本阶段没有全树或apply。latent seed保存在LAA1配置中，工厂不向旧类添加额外属性/key。不同seed下latent W1不同；相同seed可重复，而对象和Parameter互不共用。

adapter支持float64/float32/float16/bfloat16，拒绝整型/complex、非dense layout、错误shape、空B/h/N、dh不匹配、参数/输入device不一致。无autocast时dtype须相同；autocast允许低精度与FP32参数按原Linear规则计算，混合double仍明确拒绝。没有将整个forward强转FP32、关闭autocast、detach或缓存。

latent直接保留原v2检查与AMP行为。NaN/Inf的数值输入按原代数传播，测试确认可观测非有限结果，不用nan_to_num掩盖，也不增加每次forward的GPU同步扫描；非法alpha NaN/Inf/非正数在构造时拒绝。正常输入在异常/非有限调用之后仍通过。这是明确的数值合同，不是宣称非有限输入可训练。

## E. 参数量与测量边界

每个adapter参数为 `2r(dh+M)`，每个latent FFN为 `Dz(2H+1)+3H`。通过实例 `sum(p.numel())` 检查：

| 原语 | 配置 | 实际参数量=公式 |
|---|---|---:|
| adapter | dh3/M5/r4 | 64 |
| adapter | dh26/M32/r4 | 464 |
| adapter | dh16/M64/r4 | 640 |
| latent | H6/Dz7 | 109 |
| latent | H104/Dz704 | 147448 |
| latent | H208/Dz776 | 324216 |

这是单原语计数；完整模型按物理core实例数累加仍待后续实现验证。没有N×N或M×M attention；latent不混M，adapter只从dh经r投影到M。不提供时延/显存/epoch速度结论。

## F. 实际命令、环境与结果

环境：Python3.13.9、Torch2.13.0+cu130、NumPy2.2.6、PyG2.3.1、timm1.0.28，Linux WSL2，NVIDIA GeForce RTX5090 Laptop GPU。CUDA可用且支持BF16。OMP/MKL/Torch线程为1，仅在验证进程设置；CUDA原语数值检查暂关TF32，finally恢复。无compile/distributed，无远端torch2.11/cu128验证。

执行前环境：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；unittest命令另用 `PYTHONPATH=tests:.`。原始log保留。

| 命令 | 实际结果 | 时间 |
|---|---|---:|
| `python -B -m unittest -v loop_linearno_latent_adapter.test_laa2_oracles` | 先于生产实现，2通过 | unittest0.517s |
| `python -B -m unittest -v loop_linearno_latent_adapter.test_laa2_adapter loop_linearno_latent_adapter.test_laa2_latent loop_linearno_latent_adapter.test_laa2_rng_amp` | 生产包不存在，20项预期ModuleNotFoundError | unittest1.470s |
| `python -B docs/loop_linearno_latent_adapter_audit/laa2/run_checks.py new` | **22 passed / 0 failed / 0 error / 0 skipped** | 3.317s（含加载） |
| `python -B docs/loop_linearno_latent_adapter_audit/laa2/run_checks.py schema` | **LAA1 30 passed / 0 failed / 0 error / 0 skipped** | 3.192s |
| `python -B docs/loop_linearno_latent_adapter_audit/laa2/run_checks.py regression` | **旧定向72 passed / 0 failed / 0 error / 0 skipped** | 46.256s；unittest45.087s |

`test-first-manifest.json`证明oracle/测试文件及失败日志在生产v3包创建前已存在。所有源码生成的测试expected均来自独立oracle或明确手算值。preimplementation-red是预期缺实现，不是模型数学失败。

旧72项涵盖pure LinearNO attention结构/六variant数值对照、PointDepthAttnRes、v1 SR/RB/LB完整小core、LL9R边界/入口隔离、v2共享operator/round-FFN core及原latent。旧测试、golden、容差未编辑。历史全仓689项未重跑，LAA0的644通过/9历史失败/36跳过仍是旧全量证据，不能把本阶段定向测试写成全仓全绿。

CPU独立对照139条误差记录，float64最大绝对差 `7.105427357601002e-15`，float32最大 `3.814697265625e-06`（含VJP；按atol+rtol联合判定，均通过）。原始各项容差及误差见 `new-final.json`；有限差分使用eps1e-6/atol1e-5/rtol1e-4。

CUDA六个原语/精度组合全部通过，使用小输入B2/h2/N或M7/dh3；非零参数、输入/全部参数VJP、AdamW一步和strict reload均实际执行。与独立FP64 oracle比较：

| 原语/精度 | forward最大绝对差 | VJP最大绝对差 | atol / rtol |
|---|---:|---:|---|
| adapter FP32 | 3.99447e-8 | 6.26012e-9 | 4e-6 / 5e-5 |
| latent FP32 | 5.35221e-7 | 5.63007e-7 | 4e-6 / 5e-5 |
| adapter FP16 AMP | 3.91437e-4 | 3.28171e-5 | .004 / .03 |
| latent FP16 AMP | .00219907 | .000821881 | .004 / .03 |
| adapter BF16 AMP | .00206646 | .000121065 | .04 / .08 |
| latent BF16 AMP | .0163858 | .00790612 | .04 / .08 |

同精度strict reload的输出误差均为0；这不等于跨精度逐位相同。低精度测试保留原语autocast，未使用GradScaler，也不代表完整任务AMP训练。

### 首轮非预期测试失败

`primitives-attempt1.log`：22项中21通过、1失败。重复相同head输入的测试错误地要求不同GEMM行逐位相等，FP64误差仅6.938893903907228e-18。新测试改用此前已定义的FP64容差2e-12/2e-11；单独复测通过，再完整22项通过。未修改生产数学、旧golden或旧容差，原失败日志保留。旧回归产生oneDNN/Intel GPU TF32提示，不是失败，也未更改依赖处理该提示。

## G. 冻结、自审与未执行项

`end-freeze.json`核对本阶段4132起点文件；仅允许独立LAA状态文档追加，其余保持字节不变、无删除。另对LAA0起点3377文件和LAA1的新schema/测试hash核验，用户tracked diff与staged状态保持。所有新增Python内存compile、既有130个shell `bash -n`、`git diff --check`结果见 `static-checks.json`。

| 优先自审点 | 结论 |
|---|---|
| 数学独立性与轴 | 手算+索引+einsum+BA合并矩阵三oracle；LN/erf独立latent；forward/VJP/全参数有限差分通过 |
| 旧latent完全冻结 | 原类直接返回，无新forward/keys/初始化语义；源码hash、同seed值/梯度与旧LF3测试通过 |
| 初始化与公平性 | 局部adapter generator、恢复CPU scope、Python/NumPy/CPU/CUDA状态不变；四消融后续公共随机序列相同 |
| AMP与梯度启动 | 六GPU组合及CPU两精度通过；B/W2先学习、A/W1/LN后启动；无全局FP32强制或detach |
| 阶段与旧模型隔离 | 只有独立原语新增；30配置和72旧定向回归通过；未接完整attention/core/wrapper/入口 |

没有剩余需要用户裁决的数学规格冲突。NOT RUN：LAA3及后续、第二轮时机与base+delta温度顺序的完整attention验收、V3 loop/wrapper/任务接线、真实数据、完整epoch、三seed收敛、SOTA、远端栈、真实时延/显存/epoch效率、完整任务AMP/scaler续训、compile/distributed。没有真实训练或commit/push。

本 LAA2 阶段结束，未执行下一阶段。
