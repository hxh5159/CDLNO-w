# KCDNO 最终需求反查矩阵（K10）

依据为 `PLAN_KCDNO/KCDNO_Model_Specification_v1.md` 与用户确认的总控、K0–K10提示。后续用户明确授权连续K4–K10优先于旧逐阶段停顿要求。本矩阵依据真实源码与独立检查；数学讨论不授权额外网络、训练实验或物理损失。K0历史表保留，不将当时“待实现”误当当前状态。

证据缩写：**M**=`tests/test_kernel_history.py` + `tests/kernel_history_reference.py`；**C**=`tests/test_kcdno_core.py`；**T4/T5/T6/T7/T8**=对应`test_kcdno_tasks/temporal/car/airfrans/matched.py`；**D**=`tests/test_kcdno_delivery.py`；**P**=`tests/test_kcdno_performance.py`和K9原始JSON；**F**=`docs/kcdno_audit/k10/freeze.json`；**R**=同目录`old-replay.json`。最终执行日志见`final-suite.txt`，先前阶段有效证据继续保留。

| 规格/总控要求 | 实际文件与符号 | 独立验证证据 | 状态及边界 |
|---|---|---|---|
| §1/11 新family独立，旧模型不变 | `cdlno/kcdno/config.py`、`families.py`；三个原项目有限分支 | K1 family负向、F/R、D旧parser隔离 | 已实现；原Transolver/CDLNO键、pickle路径不改 |
| §2 L正整数、d/h/M/r固定等宽 | `KCDNOArchitectureConfig.validate`、`core._check_config` | K1合法/非法，C L1/2/4/8/12 | 已验证；hidden2d固定、无F/P继承 |
| §3.1 PointNorm→Down | `KCDNOBlock.forward`、原`modules._DownAttention` | C手工原语reference/残差、P live计数 | 已验证；learned Q直接参数，无多余Wq/query residual |
| §3.1/3.5 Down/Up QK RMS及缩放/投影 | 原`_DownAttention/_UpAttention/_ProjectedAttention` | C与旧modules/LRSA参考；F原语字节一致 | Down/Up标准SDPA，QKV无bias/O有bias，d_h^-1/2 |
| §3.2 FFN1 pre-RMS残差 | `latent_norm_1/latent_ffn_1` | C手工公式/公共权重 | Linear(d,2d) GELU Linear(2d,d)，bias保留 |
| §3.3 history恰在两个FFN之间 | `KCDNOBlock.forward: u→reader→aligned→FFN2` | C手工两层reference/钩子顺序 | 已验证；未调用完整旧no_sa block再补历史 |
| §3.3 删除整个SA子层 | KCDNOBlock只实例化两FFN与核模块 | C模块/参数枚举；P SA计数0 | 专属SA norm/QKVO全无；Down/Up仍有attention |
| §3.4 T在FFN2后、Up norm前 | `t = aligned + latent_ffn_2(...)`、`writer(t)` | C对象/数值/梯度钩子 | live raw T，不detach，不用norm(T)/点特征代替 |
| §3.5 一次Up latent norm及点残差 | `up(h,up_latent_norm(t));v=x+...` | C独立公式/一次norm计数 | 已验证；最终层自己的Up后直接任务head |
| §3.5 point/ConvFFN及点序 | 原`PlainFFN/ConvFFN`、各wrapper H/W | C 5×7和非法N，T4真实N、F | dense3×3 groups1，内部LN/外部RMS，无latent卷积/重排 |
| §4.1 源层Wk/raw value/sum摘要 | `KernelHistoryWriter.forward` | M double cache vs显式正核QKᵀ的输出/输入/Wk梯度 | 已验证；无Wv/Wo/mean/token softmax |
| §4.2 接收层Wq一次/单组rank | `KernelHistoryReader.forward` | M实际投影计数、M轴/B隔离，C/P | 摘要按来源保留，不重投影历史K，不与Down头数绑定 |
| §4.2 分子/分母逐来源 | `einsum bmr,bsrd→bsmd`和`bmr,bsr→bsm` | M逐来源double oracle、B1/B2/S0/1/3 | 无B广播混样、无生产M×M矩阵 |
| §4.3 当前候选恒等、per-token depth softmax | `candidates/scores/alpha/fused` | M uniform/置换/逐token权重 | [B,M,S+1]；RMS只评分，RAW候选融合 |
| §4.4 unconstrained gamma .1/w0 | Reader.__init__/forward | M系数、w0/首步梯度、C特殊初始化 | U+gamma(C−U)；无sigmoid、正当前bias或双残差 |
| §4.5 来源/轴例子 | K2 Cache[B,r,d]/mass[B,r]，Reader stack[B,S,...] | M不同M/r/B/S及explicit reference | 第4层3份历史+当前，独立来源读后统一融合 |
| §5 因果tuple/一次forward缓存 | `KCDNO.forward`局部list→tuple→append | C先读后写、A→B→A、NS T5十次完整core | 不跨batch/真实时间、不保存future/cache、不detach |
| §5 首末/L1/off边界 | block条件reader/writer注册 | C/P存在性与计数 | 第一无reader，最后无writer；L1/off无历史参数 |
| §5 L8 7写7Q28读 | 实际Writer/Reader钩子与P | K3 core-evidence、K9 JSON counts | 28逻辑来源，7批量Reader API；非28次API |
| §6 phi/eps/FP32 | `_phi`与writer/reader autocast disabled | M FP32/double、真实GPU FP16/BF16算子探针 | ELU+1/clamp1e-6/deneps1e-6；oracle不强转float |
| §6 参数初始化与无共享 | 原公共原语构造、K2 Xavier/w/gamma | C初始化次数/storage、M独立参数 | query正交，Linear截断正态，Conv native；无wrapper递归apply |
| §6 有限梯度/极端幅值 | M隔离Reader loss与C causal链 | M earliest T/Wk有效梯度、极端1e15/小分母 | 首步depth norm零梯度正常；不保证任意有限FP32均不溢出 |
| §7 八任务主profile | `profiles.profile_values`与各JSON/YAML训练字段 | K1两profile×八任务；D真实parser | 结构L8/r16，PipeM32其余64；h等见命令表 |
| §7 shape-match公平尺寸 | `profiles._MATCH`、matched独立配置 | D两family/profile默认值；K8 | 仅L/d/h/M匹配，非等参数/FLOPs/官方LRSA复现 |
| §8 Darcy | StandardModel + exp_darcy有限分支 | T4原decode/梯度loss AST、真实N7225/小网格 | ref64替换xy+fx1，stem65；原数据与loss保留 |
| §8 Elasticity/Airfoil/Pipe | StandardModel + 各exp | T4原loss/normalizer、安全合成 | Elasticity点FFN/N972；另两Conv/原物理xy和网格索引 |
| §8 NS | StandardModel与exp_ns有限分支 | T5直接原十步训练/eval语句、cache对象隔离 | fx10/stem74/output1，训练真值回填/测试预测回填 |
| §8 Plasticity | StandardModel.time_fc与exp_plas | T5原20个时间点更新/一次batch scheduler | N3131/fx1/T[B,1]/output4，不扁平时间；d>=2清晰校验 |
| §8 Car | `models.KCDNO.Model`、CarRun | T6真实PyG/原train.train/test及mask、D | x7/tuple/velocity3,p；单图可变N，全0batch合法，多图拒绝 |
| §8 AirfRANS | `cdlno.kcdno.airfrans.AirfRANSModel`/AirRun | T7真实PyG原MSE_weighted/采样ptr、F | x7+reference64/stem71；原域/输出/抽样/图/scatter流程保持 |
| §9.1/9.2 结合律/置换数学 | 独立double oracle逐来源显式QKᵀ | M前向及梯度/gradcheck | 核归一化等价，不与softmax attention等价 |
| §9.3/9.5/9.6 表达边界 | 规格及交付说明 | 源码/文档反查 | 只声明单来源读取秩≤r；eps下非严格凸组合，不外推整网秩/精度 |
| §9.4 gamma0与off | C all gamma0与off复制同公共权重 | C输出等价，P成本区分 | 数学对照成立；gamma0的all仍有核计算 |
| §10 完整参数/MAC/内存/计时 | 原perf工具+`kernel_costs.py` | P闭式vslive，K9 2配置×4 GPU | 大N全计；MAC非实测，推理cache非训练峰值；无epoch外推 |
| §11 显式CLI > profile > defaults | `options`/`families`与各parse | K1、D32默认/24真实脚本对 | 旧默认不覆盖新尺寸；旧front/rear/CDPA显式参数拒绝 |
| §11 eval先读/严格family | K1 metadata + families + 各Run | T4–T8/D冲突/sidecar字节 | 先重建已有架构再核对；不默认覆盖no-history等已存设置 |
| §11 checkpoint原格式 | StandardRun/CarRun/AirRun | T4/T6/T7/T8工业新cwd进程 | state_dict/整对象/列表，稳定路径；局部可信加载不扩权 |
| §11 整对象非权重属性 | `loading.validate_whole_model` | D篡改heads/eps/reference/激活拒绝 | strict keys之外核对行为；已学习gamma不重置 |
| §11 输出隔离/配置/日志 | `new_run_path`、原Experiment新family分支 | D启动先建目录/config、实际参数、eval只读 | 默认output/task/family/profile/时间结构；显式已有run拒绝 |
| §12 matched LRSA-full/noSA | `MatchedLRSA`；KCDNO history=off | T8六标准+两工业step/checkpoint，SAzero共同权重等价 | full固定完整L层；noSA不另复制第三core |
| §12 旧三mode/CDPA/原Transolver | 原文件不改、K0固定权重夹具 | R 41份精确回放；旧核心/任务/模块最终回归 | 相同权重/输入/输出/梯度，不用独立seed假证明 |
| §13 来源/归属 | 原LICENSE、THIRD_PARTY与K0审计 | F原文件、原语来源核对 | 保留THUML MIT；LRSA参考缺LICENSE不虚构授权，不vendor其框架 |
| 总控 禁止新增机制/训练框架 | 新package全源/最终diff | F、C模块枚举、需求逐行审查 | 无Bridge/AttnRes/Slice/新loss/多图扩展/自定义CUDA/真实时间cache |
| 总控 环境/真实数据边界 | K0/K9环境，最终报告 | 实际CPU/PyG/GPU日志分开 | 无下载/真实训练；远端2.11cu128、完整读数/收敛/精度/epoch未验 |
| 总控 保存既有修改/连续阶段 | K4–K10阶段快照与最终patch | 各before.json/changes.patch、git-status | 不commit/push/reset/改AGENTS；最新连续授权执行至K10后停止 |

“已验证”仅指该行列明的源码、合成或设备范围。没有真实轨迹或完整工业抽样评价证据；不将模型checkpoint当作完整optimizer/RNG断点。
