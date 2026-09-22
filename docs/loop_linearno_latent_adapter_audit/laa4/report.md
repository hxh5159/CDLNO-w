# LAA4：任务无关的 V3 完整 block 共享 core

**状态：PASS，仅 core 与无数据合成验证范围。LAA5 未执行。**

## A. 授权、来源和范围

当前 `/home/hwz/CDLNO`，`main@c02e671506f706910e0a1d58f03c310abf188345`，origin `git@github.com:hxh5159/CDLNO-w.git`。本阶段依据当前 v1 `core.py/body.py/attnres.py`、已通过的 LAA1 配置合同、LAA2 原语、LAA3 attention 和冻结提示词 LAA4。当前代码优先于历史报告中的路径/测试数量。

起点 4192 个已有文件（tracked2130/untracked696/ignored1366）建立 hash 清单。保留用户 checkpoint 检查脚本已有修改，以及所有 LL/LF/LAA0–3 代码和证据。没有安装依赖、访问真实数据、训练真实任务、reset/clean/stash/rebase/commit/push。

交付是独立 `LinearNOLoopCoreV3`，输入已经 lift 的 `[B,N,H]`，输出由最后 suffix head 决定。**没有 stem/time/placeholder、任务 wrapper、八任务 parser/factory 接线、任务 checkpoint archive、训练评估或输出记录接线。** 不能据此宣称 V3 生产训练命令可用。

## B. 文件和所有权

| 文件 | 原因 |
|---|---|
| 新增 `cdlno/linearno_loop/v3/core.py` | 唯一新增生产文件：完整 block 注册、显式 round dispatch、继承 RB dtype helper、core 内存 state replay guard |
| 新增 `tests/loop_linearno_latent_adapter/core_oracle.py` | 独立三残差控制、LN/MLP/AttnRes 和逐轮 trace，复用独立 LAA3 attention oracle |
| 新增同目录 `core_support.py` | 明确标注的合成 profile、原 block 工厂、局部 hook trace |
| 新增 `test_laa4_oracle.py/core.py/accounting.py/amp.py` | 手算、数值、共享、初始化、恢复负例、参数/MAC 和有限 CUDA |
| 新增本目录的脚本、日志、JSON、报告 | create-only 证据，不覆盖旧阶段 |
| 更新独立 `docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md` | 追加 LAA4，保留原历史正文 |

`LinearNOLoopCoreV3(config=..., block_factory=...)` 先完整验证 LAA1 配置，再调用工厂恰好 U=P+C+S 次。工厂只供应完整原 Standard/AirfRANS/ShapeNet block，不创建完整任务模型。P/S 使用原 `PhysicalBlock`，C 使用只增加显式 round 参数的 `RecurrentPhysicalBlock`；其中都只注册一次 `block`。

为了保留新建 block 的初始化顺序，构造后只对**该 core 新拥有的 recurrent attention 实例**更换 Python dispatch class 为已通过 LAA3 的 `V3LinearNOAttention`。先验证原 forward/独立参数，再切换；不会更换/重建该 attention 对象、子模块、Parameter 或 storage。原类源码不变，prefix/suffix 类不变。测试捕获工厂返回时的 block/attention/全部参数 id，与构造后逐一比较；构造 RNG、单次 release apply 后 RNG、公共 key/value 和后续随机向量与同结构 v1 逐位相同。

公共树完成原 release 初始化后才调用 `install_features()`。每个 core 位置的 seed 为 `(LAA1.saved_feature_seed + position) % 2**63`，latent/adapter 使用各自 seed；继承 LAA2 的隔离构造。没有第二次全树 apply、round-specific FFN、丢弃后重建的 block，或跨位置参数共享。未安装特性即 forward/load 会明确拒绝。构造包含已完成的 LR release 初始化和真实 placeholder 的完整 wrapper 时序仍属于 LAA5，本阶段只证明 core 边界不额外消耗公共 RNG。

实际 key 示例：

```text
prefix.0.block.{ln_1,Attn,ln_2,mlp}.*
core.0.block.{ln_1,Attn,ln_2,mlp}.*
core.0.block.Attn.latent_processor.{norm,linear1,linear2}.*   # 仅 latent-on
core.0.block.Attn.adapter.{A_q,B_q,A_k,B_k}                  # 仅 adapter-on
suffix.<last>.block.{ln_3,mlp2}.*
rb_receivers.<round>.<sublayer>.* / rb_output.*             # 仅 RB
lb_boundaries.<boundary>.* / lb_output.*                   # 仅 LB
```

core backbone 没有 round 副本，point FFN 和 ln_2 也跨轮共享。RB receiver 按 round/sublayer 独立是冻结设计，不等于 core operator 按轮复制。feature-off 的整个 core tensor state key 集合与同结构 v1 完全一致。

## C. 公式 → 调用 → 证据

`VisitBody.operator` 仅把 `round_index` 显式传给原 block 的 v3 attention，`ln_1` 仍是原对象。`VisitBody.mlp/scaled/native/finalize` 继承原 LL2 body；没有复制 v2 的 FFN 所有权。

| 区段/模式 | 冻结数学及执行 | 验证 |
|---|---|---|
| prefix/suffix | 原 native 两残差；最后 suffix 执行一次 ln_3+mlp2 | 原类、无特性 key/hook、原调用序列、单次 head |
| SR | 每个 core visit：`x=x+(1/R)*Attn(ln1(x),r)`，随后 `x=x+(1/R)*MLP(ln2(x))` | 继承原 scaled body；完整 oracle；观察到的两条实际 branch 后状态严格满足公式 |
| RB | completed 初始只有 anchor；每 sublayer 前 AR；partial 只累加当前轮 raw outputs；轮末 summary；最终 AR | 每个 source/h/weight/raw/partial/summary 与独立 oracle；无 1/R；逐轮张量落盘 |
| LB | 每轮用 SR Phi；`Delta=实际Y-实际H`；boundary/output 读取 anchor 和全部 Delta | 真实 block 入口/出口 hook 与实际 receiver 来源逐位相减；未重算 raw sum 代替 Delta |
| RB AMP | **直接继承原 `LinearNOLoopCore._rb_receive` 同一函数对象** | 来源 tuple 按 anchor dtype 临时转换，原 raw partial dtype/value/autograd 保持 |
| latent/adapter | 同位置 latent 每轮用同对象；adapter 仅 round_index=1；P/S 无特性 | 实际 module id、调用次数、第一轮 loss 的 adapter 梯度 None/latent 梯度非零、非零特性影响 |

支持原 preset A=P1/C3/R2/S1、B=P2/C2/R2/S2，自定义 P0/C2/R3/S1（adapter-off），以及 V3 D12/D20 shorthand。实际验证 matched_v1/efficient_v1 两 profile 的 NS/Car × D12/D20 × 三残差共24个正式宽度 core；其余完整 profile/深度/任务 wrapper 验收不冒充已完成。

原 factory `last_layer=True` 只发生在最后一个 suffix。每个 core attention 访问都重新计算 Q/K/V/KtV/QC；B2/N6 小矩阵 hooks 逐项核对。单次 forward 中所有 completed/partial/deltas 都是局部变量，没有 buffer、detach、跨 batch 历史或模块侧激活属性。连续反传、weakref 释放和异常后的新调用通过。

## D. 独立 oracle、初始化与 strict replay

oracle 不调用生产 core/body/receiver helper 或其 forward。三残差控制在 `loop_equations` 独立展开；point LN 手动均值/方差，point FFN 使用矩阵乘法与 erf-GELU；AttnRes 在 `[B,N,S,H]` 上独立归一化；attention 复用 LAA3 独立 oracle，非生产 attention。

先运行两项手算 oracle（SR输出48、RB输出20、LB输出10；输入3、两条 raw branch 均为2x、R2），再运行未实现的目标方法。`test-first-manifest.json` 证明生产 core 不存在时9个方法报告360个预期缺模块子项。随后才写生产 core。

CPU完整矩阵：6 variant × 3 residual × 2 preset及R3 custom × FP64/FP32 = **108行**。比较输入/全部参数VJP、所有接收来源/权重、h、raw、partial、round summary、LB H/Y/Delta和final。首个RB单来源receiver不参与router参数梯度，Air dead temperature不参与前向，均按旧合同记录，而不是误报缺梯度。

`round-traces.json` 另保存三模式 FP64、P2/C2/R2/S2、非零特性/receiver 的完整小张量。RB partial来自**下一实际receiver的来源**，LB Delta来自实际receiver源；SR/LB的scale取自physical block实际kwargs，逐条验证 `after=h+scale*raw`，并记录round/position/sublayer。

四消融初始化公共 tensors 逐位一致；相同 task/topology/rank 的三 residual 公共 backbone 逐位一致；零初始化四消融输出与 dropout RNG 一致。非零状态下分别去掉adapter增量或latent更新，三模式最终输出均改变；第一轮adapter无调用/梯度，两轮同位置 latent 与全 block 共用参数；第二轮loss能回传第一轮计算图与同一组共享参数。

`load_configured_state_dict(state, saved_config=..., strict=True)` 是 **core 内存重放 guard**，不是任务文件 checkpoint backend。它先验证/比较完整 config，再检查全部 key/shape，最后只用 `strict=True` 应用权重；不允许 strict=False。测试覆盖family/version、task、H、heads、M、P/C/R/S、residual、开关、Dz、adapter rank/alpha冲突，以及同形状SR的R2→R3冲突。缺key/多key/错shape不会部分改写目标参数。标准 `nn.Module.load_state_dict` 本身不提供配置识别；未来完整wrapper仍必须先执行LAA1 metadata-first协议，不能用裸权重猜结构。

## E. 参数、矩阵 MAC 与 router 口径

192组：8 task × 2 preset × 3 residual × 4消融。使用真实三类 native block，小H6/h2/M5/Dz7/r2/a3/B2/N6，分别统计prefix/shared core/suffix/head/latent/adapter/router。参数是唯一实例；矩阵MAC是实际执行调用；`1 MAC=2 FLOPs`，不包含softmax、LN/RMS、GELU、bias、residual、缩放和reshape等非矩阵工作，也不包含backward/optimizer。

- latent参数：`C*[Dz*(2H+1)+3H]`；实际矩阵MAC：`2*B*C*R*M*H*Dz`。
- adapter参数：`C*2*r*(dh+M)`；实际矩阵MAC：`2*B*C*h*N*r*(dh+M)`，只执行第二轮。除nn.Linear/Conv hooks外，单独跟踪adapter内部四次F.linear的实际输入/权重shape，不能把它们漏计。
- RB router参数：`2H*(2CR+1)`；LB：`2HR`。inactive mode没有router参数。
- 其余投影、KtV、QC、point FFN及head的分项整数与LAA1独立解析结果完全一致。

**既有LAA1 router公式的口径差异单列保留：** `router_contraction_macs` 按所有逻辑sources计算，包括RB第一个singleton；生产原语对这个receiver直接返回identity，实际不做dot/weighted sum。因此每个RB forward有 `2*B*N*H` 的恒等短路节省。192行中记录 `router_logical_macs`、`router_executed_macs`、`singleton_identity_saved_macs`，验证 `logical=executed+saved`；SR/LB差额0。没有修改LAA1旧公式/golden/metadata hash，也没有为匹配计数去破坏singleton行为。后续性能或实际执行报告应使用executed字段；不能把LAA1该逻辑计数称为实测router工作量。

正式宽度例子（**仅core，含suffix head/router/特性，不含stem/time/placeholder**）：NS matched_v1 D12/SR H208/h8/M32/Dz680为3,249,585参数；Car matched_v1 D12/SR H208/h8/M32/Dz776为3,758,244参数。24个正式宽度core均与完整解析结果扣除stem/time后精确一致。它们不是完整任务模型的参数量。

合成空间网格的小H/W只在测试中以明确的已保存profile快照注入：先用原纯profile解析H2/W3/dropout，再由V3解析和重放；没有新增生产override或放宽旧schema。正式宽度NS/Car测试直接使用未替换的LAA1正式config，N6只用于不规则/point attention。真实任务数据、网格协议与生产配置不变。

## F. 实际命令、环境、失败和结果

环境：Python3.13.9、Torch2.13.0+cu130、NumPy2.2.6、PyG2.3.1、timm1.0.28；WSL2 Linux、RTX5090 Laptop GPU。以下进程使用 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，最终runner还显式Torch线程1；unittest直接调用使用 `PYTHONPATH=tests:.`。没有安装/降级依赖。

| 实际命令（根目录，python -B） | 结果 | 实际耗时 |
|---|---|---:|
| `-m unittest -v loop_linearno_latent_adapter.test_laa4_oracle` | 生产实现前2通过 | 0.441s |
| `-m unittest -v loop_linearno_latent_adapter.test_laa4_core loop_linearno_latent_adapter.test_laa4_accounting loop_linearno_latent_adapter.test_laa4_amp` | 生产不存在；9方法、360预期缺模块子项 | 0.430s |
| 上述三个模块加 `test_laa4_oracle` | 首轮11方法；96个adapter MAC漏计断言失败，无数学/AMP失败 | 32.873s |
| `test_laa4_accounting` 与 `CoreTests.test_structure_rejects_aliasing_or_mismatch_and_install_is_rng_isolated` | 修正计数观察器后2通过 | 2.896s |
| `CoreTests.test_formal_profiles_d12_d20_actual_widths_and_shapenet_rank`、`test_constructor_rng_matches_v1_and_post_apply_feature_installation`、`test_first_round_feature_gradients_and_nonzero_ablation_effects`、`test_strict_reload_metadata_conflicts_before_weight_application` | 补充4通过 | 4.843s |
| `docs/loop_linearno_latent_adapter_audit/laa4/run_checks.py new` | **14 passed / 0 failed / 0 error / 0 skipped** | 42.490s含加载 |
| 同runner `previous` | **LAA1–LAA3 68 passed / 0 failed / 0 error / 0 skipped** | 23.754s含加载 |
| 同runner `regression` | **旧定向72 passed / 0 failed / 0 error / 0 skipped** | 61.743s含加载 |
| `docs/loop_linearno_latent_adapter_audit/laa4/record_trace.py` | 三残差完整逐轮记录通过 | 0.091s，不含import |

原始失败日志保留。首轮漏计原因是adapter使用F.linear，nn.Linear模块hook观察不到；修正新测试观察器，无生产公式/旧golden/容差修改。构造初始化写法简化为显式nn.Module.__init__，并补强重复参数所有权/非native工厂拒绝，均在最终全套新测试前完成。

CPU误差：FP64 final/all comparisons最大均4.440892098500626e-16；FP32 final最大2.384185791015625e-7，全部中间量/梯度最大4.172325134277344e-7。独立展开预设atol/rtol：FP64 2e-11/2e-9，FP32 5e-6/1e-4。feature-off与v1的CPU输出、全部梯度、AdamW一步及optimizer state、dropout RNG均使用零容差，全部通过。

CUDA **6 variant × 3 residual × FP32/FP16 AMP/BF16 AMP =54行**。每行实际运行nonzero on/on core前向/反向/AdamW/strict replay，并另对feature-off与v1的输出/全部梯度/dropout RNG做零容差对照；reload和off最大误差均0。这不是所有四消融的完整CUDA任务矩阵；四消融core检查在CPU，attention完整消融CUDA矩阵已随LAA3回归重跑。

RB下FP16/BF16 raw branch和authoritative summary保持低精度；receiver来源为anchor的FP32局部视图，数值逐位等于必要cast。图未detach，final loss到第一轮raw output梯度有限且非零。SR/LB亦在54行内通过原v1逐位回归。测试未使用GradScaler，未测完整任务AMP续训或峰值显存/时延。

旧72项包括pure attention官方对照/结构、PointDepthAttnRes、SR/RB/LB、LL9R、v2 core/latent；环境oneDNN/Intel GPU TF32提示非失败。全仓689项未重跑；此前LAA0的644通过/9历史失败/36跳过不改写成全仓全绿。

## G. 冻结、自审和剩余边界

`end-freeze.json`核验LAA4起点4192文件，仅允许独立状态文档追加；无旧文件删除/生产漂移。LAA0起点3377文件全不变，用户原tracked diff、staged和HEAD保持。`static-checks.json`保存新增Python内存compile、130个旧shell语法和git diff检查；源码/公式位置及增量见 `source-map.json`、`source-diff.patch`。

1. **完整block所有权：PASS。** U次工厂调用、block/attention/参数id保持，ln2/point FFN跨轮共享，position间独立，最后suffix唯一head，无round core副本。
2. **残差与图：PASS。** 108行独立oracle、手算、三模式实际张量记录；LB实际相减；RB无1/R且直接继承旧dtype helper；第二轮向第一轮和共享参数反传。
3. **初始化与兼容：PASS（core范围）。** 原分配和release apply RNG、四消融公共key/value、三模式backbone、dropout和feature-off/v1 AdamW严格一致；无旧源码修改。
4. **state与成本：PASS（core范围）。** metadata比较先于权重应用，key/shape预检、strict replay；192组分项参数/MAC匹配；RB singleton的逻辑计数与实际短路差额明确单列。
5. **阶段隔离：PASS。** LAA1–3及旧定向回归通过；只新增core，不接wrapper/八任务/生产checkpoint或训练逻辑。

NOT RUN：LAA5及以后、八任务生产接入/新任务checkpoint archive/完整RNG与DataLoader续训闭环、真实数据/完整epoch/paired seeds/精度/SOTA、真实时延/峰值显存/epoch效率、远端Python3.10/torch2.11/cu128、完整任务AMP/GradScaler、compile/distributed。没有需要改变模型数学的规格冲突。生产接口与真实实验尚不能由本阶段PASS推断已完成。

本 LAA4 阶段结束，未执行下一阶段。
