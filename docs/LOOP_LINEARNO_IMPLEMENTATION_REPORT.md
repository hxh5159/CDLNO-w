# Looped LinearNO：LL10 最终实现审查

本报告针对用户当前 checkout 的 `linearno_loop`，不是公开 main 的重建。LL10 不新增功能，不修改生产模型、parser、factory、训练/评估、配置、launcher、旧测试或容差。真实数据实验与实现验收分开记录。

<!-- FINAL_VERDICT -->
最终状态：**PARTIAL（loop 实现目标与合成验收通过；完整旧回归受既有冻结资料/历史 provenance 断言影响）**。没有发现 LL10 引入的生产源码漂移。实现范围的合成验收可以交付；完整全仓回归不能标为 PASS。
<!-- /FINAL_VERDICT -->

## A. 真值、来源与范围

仓库 `/home/hwz/CDLNO`，remote `https://github.com/hxh5159/CDLNO-w.git`，branch `main`；HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。起点是已有大量 tracked/untracked 修改的工作树；完整状态/diff/hash 保存在 [LL10 起点](loop_linearno_audit/ll10/start-manifest.json)，源文件副本在 `/home/hwz/CDLNO-artifacts/loop-ll10-before-ky0r8z1j/source`。未执行 reset/clean/stash/checkout/rebase/commit/push。

| 固定来源 | 版本 | 本项目采用的范围 |
|---|---|---|
| thuml/Transolver | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` | 原任务协议和原模型对照，不是 loop 数学来源 |
| HiPRL/LinearNO | `3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`、论文 `2511.06294v3` | 三套算子、stem、位置、温度和任务差异的纯 LinearNO 基线 |
| Attention Residuals | `2603.15031v1`、`85e22310fe5ee860b4a023de312d791de8a5a5e6` | 点域 key RMSNorm、pseudo-query、raw value/source-softmax；RB 来源时序 |
| Kimi K3 | `2607.24653v2`、`3cb39dfd32e51c3328e2e4b4af21341247d06c43` | 权重共享/loop 与残差研究背景，不冒称该仓库的 PDE 官方实现 |
| residual scaling | `2606.18524v1` | 1/R 残差研究背景；不是三种模式共同稳定性的证明 |

LL0 来源台账的 **142 个缓存文件重新核对 SHA256 全等**，包含五份论文缓存；见 [复核](loop_linearno_audit/ll10/reference-revalidation.json)。没有下载数据或权重。P/C/R/S 任务适配、rank×1/2 控制、LB 的实际 Delta 来源及三模式公平比较是本项目冻结设计。新点域 AttnRes 不使用旧 A/K latent history；Transolver 核图不能作为本模型的实验结果。

## B. 完整架构与逐式反查

记点状态 `x∈R[B,N,d]`，heads=h，head_dim=d/h，rank=M；d 避免与空间网格 H 混淆。完整路径为：

```text
原任务输入/位置/可选时间处理 → 原stem/placeholder
 → P个独立prefix（各1次）
 → C个共享物理core（按原顺序访问R轮，每次重算Q/K/V/KᵀV/QC）
 → S个独立suffix（各1次）
 → 最后suffix的ln_3/mlp2（仅1次） → 原任务输出
```

| 模块/公式 | 实际实现 | 独立验证 |
|---|---|---|
| 三套原始 forward | `standard.py::LoopedStandardModel` 继承 Standard `Model`；`airfrans.py`/`shapenet.py` 继承各自纯模型 | `test_block_body`、`test_sr_core`、八任务 native workers |
| 输入处理后的 block loop | `construction.py::LoopForwardView.blocks` 返回 `(self.loop,)`，不二次注册 block | 单head、state key、module id、逐visit hook |
| 原分支 `O(x)=Attn(ln_1(x))`、`F(x)=mlp(ln_2(x))` | `body.py::LinearNOBlockBody` | 原block对照及独立 SR/RB/LB oracle |
| 共享P/C/S组织与三枚举分派 | `core.py::LinearNOLoopCore` | `test_sr_core`、`test_rb_core`、`test_lb_core`、`test_modes` |
| 点域AR | `attnres.py::PointDepthAttnRes` | `point_attnres_oracle.py`；FP64/FP32、来源置换、gradcheck、手算 |
| LL9R AMP来源边界 | `core.py::_rb_receive` | `test_ll9r` 显式cast reference、raw graph/对象检查；144 CUDA smoke |

上述文件位于 `cdlno/linearno_loop/`；仅展示这几个文件还不完整，必须同时包括 `PDE-Solving-StandardBenchmark/model/LinearNO.py`、`cdlno/linearno/{attention,airfrans,shapenet}.py`。wrapper 的早期 docstring 仍称 synthetic，实际已经由 LL6/LL7 factory 使用；LL10 没有为修饰文案改生产源码。完整符号/行号在 [source-symbols.json](loop_linearno_audit/ll10/source-symbols.json)。

**LinearNO 算子。** 每次 visit 重新产生 `Z[B,h,N,d_h]`，`Q=softmax_M(logit_Q/tau_Q)`、`K=softmax_N(logit_K/tau_K)`、`V`；然后 `C=KᵀV[B,h,M,d_h]`、`QC[B,h,N,d_h]`、原 output projection。plain/conv没有温度，Standard temp/conv_temp沿用原温度与clamp；Air保留未使用的temperature参数；Car保留`tempreature_*`拼写及clamp。卷积核/reshape/点序与原FFN均未改。没有Transolver式latent self-attention，也没有点对点 N×N attention。

**SR。** prefix/suffix为原native residual。core每块依次：`x'=x+O(x)/R`，`x''=x'+F(x')/R`。identity不缩放；两条分支均缩放。最终head不在循环内。

**AR原语。** 来源`s_j[B,N,d]`沿来源轴stack：

`k_j = norm_scale ⊙ s_j / sqrt(mean_d(s_j²)+1e-6)`；
`e_j = sum_d(query⊙k_j)`；`alpha=softmax_source(e)`；`AR=Σ_j alpha_j s_j`。

query=0、norm_scale=1，每receiver只有2d参数，无bias、value投影、sqrt(d)、来源数乘法、dropout或1/R。RMSNorm只用于key，value保留raw。每个点独立沿来源轴归一；从不沿N做attention。singleton直接返回原tensor，对应query/norm无有效梯度；query=0时所有多来源receiver是均匀平均，并非普通residual恒等初始化。低精度打分/累加用FP32，FP64保持FP64。

**RB。** core入口a为b0；每轮partial从None开始，每个operator/MLP前独立receiver读取完成轮的`[b0..b_(r-1)]`，有partial时再附加它。branch只累加raw输出；不把router输出h、anchor或普通identity加进partial。轮末`b_r=partial`；final独立receiver混合`[b0..b_R]`再进入suffix。全路径无1/R或来源数补偿。receiver数`2CR+1`，query/norm按执行位置独立；operator/MLP参数仍跨round共享。C3/R2来源数为`1,2,2,2,2,2 | 2,3,3,3,3,3 | 3`；C2/R2为`1,2,2,2 | 2,3,3,3 | 3`。

LL9R已批准的AMP处理仅在receiver边界把临时来源tuple转为anchor dtype；authoritative raw partial不变，不detach。共享AR原语仍拒绝任意混合dtype。该修复在LL10未再变动。RB凸混合不自动继承SR的1/R稳定性结论。

**LB。** 轮内执行同样的`Phi_(1/R)`；保存实际入口`H_r`和出口`Y_r`，严格`Delta_r=Y_r-H_r`。中间boundary读取`[a,Delta_1..Delta_r]`；最后output读取`[a,Delta_1..Delta_R]`，不对Delta再次缩放。只有R个receiver，不在轮内每个sublayer前做AR。R2零query时严格为`H2=(a+Delta1)/2`、`final=(a+Delta1+Delta2)/3`；无隐藏补偿。R1仍有output AR，因此不等同SR。

所有历史仅是单次forward局部变量；不注册parameter/buffer、不放self、不序列化、不跨batch/NS物理时间/Plasticity查询/Air成员。外置诊断的临时detached观察副本与权威计算图分离，退出/异常清空。诊断默认关闭，不修改旧monitor定义或增加gate、缩放、裁剪。

## C. 八任务、拓扑、初始化与科学协议

| task | wrapper输入→输出 | d/h | variant | base M→默认M | 任务约束 |
|---|---|---:|---|---:|---|
| Airfoil | `(x, None)`→`[B,11271,1]` |128/8|conv_temp|64→128|221×51，非方形点序保持 |
| Darcy | `(x, fx)`→`[B,7225,1]` |128/8|conv_temp|64→128|85×85，unified off，decode/边界置零/0.1导数项、zero padding、dx=1/85保持 |
| Elasticity | `(x, None)`→`[B,972,1]` |128/8|temp|64→128|不规则点 |
| Pipe | `(x, None)`→`[B,16641,1]` |128/8|conv_temp|64→128|129×129，ratio1/batch4，原flatten/reshape |
| NS | `(x, fx10)`→`[B,4096,1]` |256/8|plain|32→64|ratio2，unified on/ref10，训练10步真值回填/一次optimizer；eval10步预测回填 |
| Plasticity | `(x, fx, T[B,1])`→`[B,3131,4]` |128/8|conv|64→128|101×31，Time_Input；每batch20次forward/backward/optimizer，scheduler一次 |
| AirfRANS | `Data`→`[N,4]` |256/8|airfrans|32→64|单图、可变N、原位置距离拼接、独立ensemble成员 |
| ShapeNet-Car | `(Data,geom)`→`[N,4]` |256/8|shapenet|32→64|单图、原7通道、fold/surface/drag接口；M/head_dim正整数 |

具体profile从纯LinearNO原配置派生，不能用Transolver旧默认覆盖。rank×1仍可用。Standard rL2为逐样本flatten、无epsilon再均值。Plasticity实际仓库collate使用**torch.randperm**排列20次查询（不是历史提示词中描述的NumPy）；resume保存Python/NumPy/Torch及独立generator状态。Air训练为标准化四通道MSE，volume+1×surface；Car为标准化三速度MSE+0.5×表面压力MSE。二者都未改成论文未明确的本地rL2。paper profile预算为Standard500、Air400、Car200；其他profile保持各自原义，Darcy official_release非500拒绝静默错步数。

| preset | P/C/R/S | unique blocks | executed visits | 共享core访问 |
|---|---|---:|---:|---|
| p1_c3_r2_s1 |1/3/2/1|5|8|C1 C2 C3 / C1 C2 C3 |
| p2_c2_r2_s2 |2/2/2/2|6|8|C1 C2 / C1 C2 |
| custom（测试例） |0/2/3/1|3|7|C1 C2 / C1 C2 / C1 C2 |

不把两个preset称为参数相同。构造恰好U=P+C+S个原生block，不先造8层再删。stem/time/所有物理block构造完成后只执行一次原apply(init)，placeholder保持之后的原随机时序；router zeros/ones不消费RNG。映射到同U纯模型的公共state/初始化后RNG精确对照，跨三mode同seed/topology/rank主干键值/RNG相同；原DataLoader使用单独记录的generator。Air每成员有不同且独立的model/core/router/optimizer；不同mode同一成员初值和数据顺序可配对。

## D. 真实调用链与严格恢复

```text
tran_evaluate/linearno_loop/<task>.sh {train|resume|eval}
 → launch.py：path.sh + 原任务launcher argv + GPU映射 + 真parser预检
 → entry.py：安装独立记录器，runpy调用原入口
 Standard exp_*.py → cdlno_entry.parse_args → loop.standard_entry
   → 原model_dict.get_model的显式guard → model_module → build_from_config
   → LoopedStandardModel → 原任务train/eval → LoopStandardRun
 Air main.py/main_evaluation.py → cdlno_entry → loop.industrial_entry
   → loop.air_entry/AirRun → LoopedAirfRANSModel → 原train/metric
 Car main.py/main_evaluation.py → models/cdlno_run → loop.industrial_entry
   → loop.car_entry/CarRun → LoopedShapeNetModel → 原train/metric
 所有loop存档 → loop.checkpoint：独立schema、成对checkpoint/weights/JSON manifest
```

只有显式loop train或saved family=`linearno_loop`进入新adapter；旧命令无loop flags不加载这些adapter。family=`linearno_loop`，extension=`loop_linearno_v1`。eval/resume先读取`architecture.json`，完整校验model constructor、loop_spec、profile/data/objective/evaluation/provenance/normalizer/resume信息，显式参数只作一致性断言。loop/A/K（包括显式false）混用非法；preset/custom、rank/multiplier、family/residual/PCRS/M/d/h等冲突在模型构造/torch.load前拒绝。

校验pair文件hash与immutable metadata后才`torch.load(weights_only=True)`，随后逐键strict加载；无whole-object推断、strict=False或随机补键。权重tensor缺失/router key错配必须在读出tensor后由strict检查拒绝，不能误称未反序列化就知道tensor损坏。optimizer校验参数签名/组/shape/dtype/进度，再恢复AdamW和scheduler；Python/NumPy/Torch CPU/CUDA/loader generator的RNG最后恢复。NS10/Plasticity20调用保留，各次forward历史从空开始。

每epoch先提交不可覆盖的pair/metadata/hash manifest再发布latest/final；已提交较新epoch阻止回滚。共享core只序列化一次，无round2复制键。Air根/成员manifest保持独立和明确顺序；成员内/成员间中断都测试。便捷model.pt或工业model文件不能单独冒充完整resume档案。

兼容证据包含旧类输出/state/RNG回归、旧parser默认、旧路径、whole-object/state_dict原加载链、LL0/LL1冻结及LL6/LL7/LL9R精确源码/AST反投影。反投影仅剥离已批准的具体接线，不忽略任意源码更改；真实源码变异负测会失败。LL10自身相对起点的生产源码字节完全冻结以end-freeze为准。

## E. 参数与计算量

下表为本次实际全宽构造统计，SR存储参数；RB/LB在此基础上加router。完整96项分stem/prefix/shared core/suffix body/head/router及独立解析对照见 [counts.json](loop_linearno_audit/ll10/counts.json)。全部96行与LL9R相同。

| task | P1 M×1 / M×2 | P2 M×1 / M×2 | RB增量P1/P2 | LB增量 |
|---|---:|---:|---:|---:|
| airfoil |1,116,497 / 1,126,737|1,332,961 / 1,345,249|3,328 / 2,304|512|
| darcy |1,116,753 / 1,126,993|1,333,217 / 1,345,505|3,328 / 2,304|512|
| elasticity |378,577 / 388,817|447,457 / 459,745|3,328 / 2,304|512|
| pipe |1,116,497 / 1,126,737|1,332,961 / 1,345,249|3,328 / 2,304|512|
| ns |2,182,145 / 2,192,385|2,580,737 / 2,593,025|6,656 / 4,608|1,024|
| plasticity |1,150,084 / 1,160,324|1,366,532 / 1,378,820|3,328 / 2,304|512|
| airfrans |2,162,988 / 2,173,228|2,561,588 / 2,573,876|6,656 / 4,608|1,024|
| car |2,459,220 / 2,469,460|2,923,620 / 2,935,908|6,656 / 4,608|1,024|

RB=`2d(2CR+1)`，LB=`2dR`。矩阵MAC按**executed=P+CR+S**计，每visit的KᵀV及QC各`B*N*d*M`。卷积系数k=9或1、output投影层数o、FFN ratio=f时，body矩阵MAC为`B*N*(k*d²+o*d²+2f*d²+d*d_h+4dM)`。stem/time与单head另计。router非singleton来源数S时score/raw聚合有`2*B*N*d*S`收缩MAC等价值，RMS/softmax等标量操作另列。

`FLOPs=2*矩阵MAC`不含softmax、RMSNorm/LayerNorm、GELU、bias、residual、clamp/除法、距离、sin/cos、layout等，不是完整FLOPs。参数共享减少存储，不减少R次执行；M翻倍不是compute-matched。工业参数是每个ensemble成员。

LL9已测的CPU48组单线程小宽度计时仍保留在 [cpu.json](loop_linearno_audit/ll9/cpu.json)：warmup3、测量12，median/p90；例如Darcy P1 SR forward48.680/53.870ms，forward+backward+AdamW129.969/136.080ms。**这些是LL9历史计时，LL10未重测延迟**，非独占宿主、B2/N7225/d8/h2/M8，不能外推正式宽度、真实loss、data loader或epoch效率。LL10只重新验证参数/执行量/合成正确性和CUDA smoke/显存。

## F. 最终实际检查、环境、失败/skip

<!-- FINAL_RESULTS -->
最终实际结果如下。各原始日志保留；早先被会话中断的尝试只作为中断记录，不计入通过。

| 检查 | 实际结果 | 用时/环境 |
|---|---|---|
| 96 项全宽配置/参数分项/解析成本 | 96/96，与 LL9R `counts.json` 逐行相等 | 8.594s；CPU、Python3.13.9、torch2.13+cu130 |
| 48 项 canonical-N 小宽度矩阵 | 48/48 forward、backward、AdamW、strict reload、调用/诊断 | 21.588s；CPU、诊断显式开启 |
| Standard 原生六任务 | 36/36 train→checkpoint→新进程 resume/eval | 597.769s；真实任务 forward AST、内存合成数据 |
| AirfRANS/Car 原生工业 | 12/12（Air 两成员与两种中断边界，Car fold/接口） | 239.824s；真实 PyG 对象的内存合成数据 |
| 旧 archive replay | 9/9，strict weights、optimizer/scheduler/RNG、输出与旧 bytes 全等 | 98.889s；新进程 |
| LL9R 修复专项 | 11/11 | 26.754s；CPU |
| CUDA synthetic | 144/144：FP32、AMP FP16、AMP BF16 各48 | 140.136s；RTX5090 Laptop，TF32/compile关闭 |
| Python/shell/static/source pins | 335 Python compile、129 `bash -n`、`git diff --check`、142 固定来源 hash | static 2.251s；reference revalidation PASS |

完整 80 模块、689 方法的并行回归记录为 36 skip；4 个模块返回失败：`linearno.test_car_entry` 首次一个 eval 超时、`linearno.test_legacy` 缺失旧 `docs/` 审计产物、`loop_linearno.test_isolation` 同一历史快照文件/分类不一致、`loop_linearno.test_standard_entry` 既有 legacy provenance hash 不一致。排除这4个已定位的历史/环境模块，剩余76模块为 **638通过、36 skip、0 failure、0 error**。Car 模块随后独立复测 **4/4通过，139.493s**；因此首次 Car 失败是资源/超时现象，不是模型失败。完整原始数据在 [regression-results.json](loop_linearno_audit/ll10/attempt2/regressions/regression-results.json)，命令与环境在 [regressions-command.json](loop_linearno_audit/ll10/attempt7/regressions-command.json)。没有修改这些旧测试、golden、容差或缺失审计资料来制造全绿。

静态冻结分两层记录：[production-freeze.json](loop_linearno_audit/ll10/production-freeze.json) 对 loop/纯模型/任务入口/测试/工具/launcher 的 `.py/.sh/.json/.yaml/.toml` 源集合为 **0 changed/missing/new**；完整 4,740 路径快照仍包含工作树中用户/历史 catalog、实验包和审计输出的增删，详见 [final-freeze.json](loop_linearno_audit/ll10/final-freeze.json)，因此不能把完整快照误读为干净 git 状态。`.claude/settings.json` 的 hash 差异和输出目录中的六个运行 sidecar 是工作树既有/运行时变化，未由 LL10 回滚。
<!-- /FINAL_RESULTS -->

本机Python3.13.9、torch2.13.0+cu130、WSL2、Intel Core Ultra9 290HX Plus；CPU检查显式空CUDA mask、OMP/MKL/OpenBLAS各1线程、Agg、`python -B`。CUDA检查为本机RTX5090 Laptop、FP32/FP16/BF16，TF32/compile关闭；warmup后同步、清allocator cache/reset peak，执行一步再同步，记录baseline/peak allocated/reserved。不把allocator读数当全设备显存或正式全宽训练峰值。

独立oracle不调用生产forward生成expected：FP64继承atol1e-12/rtol1e-10，FP32通常1e-6/1e-5；LB实际相减传播后source权重沿用既有1e-5/1e-4（Delta消减与RMS放大），同source原语仍按原容差。FP32独立数学oracle的optimizer对照用SGD；原生相同kernel展开另测AdamW/dropout/RNG，不能混称逐位独立AdamW oracle。LL10没有放宽任何阈值。逐tensor误差与命令耗时在机器证据中保留。

## G. 命令、输出与交付边界

完整八题、presetB/custom/rank×1、三paired seeds和数据路径见 [命令文档](LOOP_LINEARNO_COMMANDS.md)。以下是**未来远端真实运行示例，LL10未执行**：

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
export CDLNO_REPO_ROOT="$PWD" CDLNO_RUNS_ROOT="$PWD/output"
source ./path.sh
FLAGS=(--linearno-loop 1 --linearno-loop-topology p1_c3_r2_s1
  --linearno-loop-residual-mode lb_attnres_1_over_r
  --linearno-loop-rank-multiplier 2
  --linearno-profile paper_table8_on_release_model --seed 0 --gpu 0)
RUN="$(bash tran_evaluate/linearno_loop/airfoil.sh train "${FLAGS[@]}" --print-run-dir)"
bash tran_evaluate/linearno_loop/airfoil.sh train "${FLAGS[@]}" --experiment-dir "$RUN" --then-eval
# 对未完成run续训；评估只读同一RUN的metadata/weights。
# bash tran_evaluate/linearno_loop/airfoil.sh resume --gpu 0 --experiment-dir "$RUN" --then-eval
# bash tran_evaluate/linearno_loop/airfoil.sh eval --gpu 0 --experiment-dir "$RUN"
```

其他task仅替换脚本；`--dry-run`可预检而不读数据。训练成功才执行eval；不猜最近run，显式运行目录不可覆盖，GPU掩码按命令文档映射。结果位于`output/<task>/linearno_loop/<task/profile/P-C-R-S/residual/M/seed/config-hash/唯一后缀>/`；包括config/log/status、architecture、manifest、checkpoints/weights、训练曲线与visualizations、每次独立evaluations目录。`loop_run_manifest.json`记录公共backbone初始hash、seed、实际参数和首次forward实际call schedule；eval/resume不重写训练记录或重拟合normalizer。用户已有`experiment_config/`及压缩包保持原样，未重生成实验计划。

自审重点已落实到源码和独立证据：①继承输入处理/完整共享计算/单head；②三种公式与来源时序/AMP边界；③公共初始化、placeholder、数据generator；④metadata-first严格恢复及错配拒绝；⑤冻结兼容、失败/skip来源与实验边界。没有以“我实现的”代替证明。

明确 **NOT RUN**：真实loader/VTK指标、真实训练、正式三seed全矩阵、完整epoch、收敛/泛化/精度/SOTA、真实epoch效率、远端Python3.10/torch2.11/cu128与远端GPU。恢复验收是epoch边界；不宣称mid-batch、分布式/persistent-worker、断电文件系统或AMP scaler完整续训验收。CUDA仅小型合成，正式宽度仅参数构造；canonical N矩阵使用缩小宽度。未安装/升级/降级任何依赖。

本 LL10 阶段结束，未执行真实实验
