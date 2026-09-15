# KCDNO K1：独立配置、profile 与元数据协议

2026-09-15。K0已审查通过；本轮仅完成K1。**已有独立配置/元数据基础，尚无KCDNO模型构造器、核读取、任务factory注册或训练接入。** 本报告中的配置恢复是恢复配置对象，不是新模型权重加载验收。

## A. 完成范围、依据与基线

实际读取 `AGENTS.md`、K0审计/STATUS/夹具索引、`PLAN_KCDNO/KCDNO_Model_Specification_v1.md`全文，以及现有配置、sidecar、三个项目的实际parser与安全辅助模块。当前用户K1要求优先；旧CDLNO/A/V约束仍保护旧行为。未将数学说明扩展为额外结构授权。

本轮起点 `main` / `222647fef343e9fe929e65412f5c7449ced5517b`，工作树干净。它晚于K0基线 `9f72946e0adbfe27ba0b75b1646fa697ae16d3df`，包含已完成的Darcy消融薄脚本；这些已有成果全部保留。

修改前保存371份tracked文件SHA256与外部源码快照 `/home/hwz/CDLNO-artifacts/k1-before-khe93ugp/source`。核验K0的41份同权重夹具涉及136个文件，全部存在且hash一致；修改前41/41回放通过，**没有缺失而需要补捕获的旧基准**。没有重新随机初始化输出来替代旧基准，未覆盖K0或pre-A1夹具。

证据：[修改前版本/环境/源码hash](kcdno_audit/k1/before.json)、[回放前](kcdno_audit/k1/replay-before.json)、[回放后](kcdno_audit/k1/replay-after.json)。

## B. 实际文件、符号与diff范围

| 新增文件 | 主要符号及目的 |
|---|---|
| `cdlno/kcdno/__init__.py` | 新配置显式导入；`FAMILY='kcdno'`、显示名`MODEL_NAME='KCDNO'`；没有占位模型构造器 |
| `cdlno/kcdno/config.py` | `KCDNOArchitectureConfig`、`KCDNOInitializationConfig`、`KCDNORuntimeConfig`，独立完整字段校验/序列化 |
| `cdlno/kcdno/profiles.py` | `profile_values/resolve_profile`，八任务×两profile的新家族结构默认 |
| `cdlno/kcdno/options.py` | `explicit_arguments/architecture_overrides/resolve_training`，明确显式参数及优先级，不绑定入口 |
| `cdlno/kcdno/metadata.py` | `KCDNOMetadata`、family路由契约、JSON读写、`resolve_evaluation`及独立目录建议 |
| `tests/test_kcdno_config.py` | 13项针对性配置/metadata/parser测试，无任务入口import |
| `docs/kcdno_audit/k1/` | 前后回放、26项最终检查、16份resolved profile、安装发现/语法、冻结证据和可审查代码diff |
| 本报告 | 字段、解析/加载合同、验收范围与限制 |

仅增量更新独立 `KCDNO_IMPLEMENTATION_STATUS.md` 与 `memory/current-state.md`。现有`cdlno/__init__.py`、CDLNO config/checkpoint/core/模块、任务wrappers/入口/JSON/YAML/脚本、性能工具、依赖、旧测试和K0文件均未改变。`pyproject.toml`继续发现`cdlno*`，因此新子包在现有安装结构内；没有重建项目或重新安装。

[生产配置/协议与测试diff](kcdno_audit/k1/code-changes.patch)，[冻结结果](kcdno_audit/k1/freeze.json)。

## C. 字段、解析、公式与参数归属

### 独立字段与命名

| 类别 | 实际字段/约定 | 加载时的含义 |
|---|---|---|
| 架构/行为 | `family, architecture_version, L,d,h,M,kernel_rank,history_mode` | 逐字段严格比较；无F/P/front_latent_mode/CDPA模式。d统一point/latent/attention宽度，M统一各层；r不要求整除h |
| FFN/点模块 | `ffn1_hidden,ffn2_hidden,point_hidden,latent_activation,point_activation,point_module` | 默认hidden均2d，各hidden记录为实际正整数；plain GELU固定，PointFFN/ConvFFN明确。两个hidden字段不表示共享参数 |
| 归一化 | `norm='rmsnorm', qk_norm='per_head_rmsnorm', output_norm='layernorm', norm_eps=1e-6` | 保留Down/Up QK RMS；ConvFFN内部LN/dense3×3由v1语义固定，不增加depthwise开关 |
| 核与gate行为 | `kernel_phi='elu_plus_one', kernel_clamp=1e-6, kernel_denominator_eps=1e-6, kernel_projection_bias=False, gate_parameterization='unconstrained_scalar'` | 不支持softmax核、sigmoid gate、额外value/output projection开关；attention dropout固定0 |
| 训练起点/复现 | `initialization_version, public_init, kernel_init, gamma_init, scorer_init, norm_scale_init, seed` | 默认kcdno-init-v1 / cdlno-front-v1 / Xavier uniform gain1 / .1 / 0 / 1。只记录来源，不参与已学权重结构比较，不在eval重新初始化 |
| 运行 | `device,dtype,batch_size,sdpa_backend,amp,tf32,compile` | 不作为权重结构冲突。eval返回`runtime_differences`，后续入口须写日志；数值/性能可能改变 |
| 元数据外层 | `schema_version=1,family='kcdno',task,profile,checkpoint_format,architecture,initialization,runtime` | profile是来源标签；完整resolved architecture才是重建依据。外层/内部family都须明确且一致 |

新配置构造时未给hidden会按最终d生成2d；序列化后必须有明确整数值，读旧/新JSON不能用null或缺字段触发补默认。正整数拒绝bool、浮点数、字符串、非有限数及非正数，不静默取整。L≥1、d%h=0；可使用r=5/h=3等独立rank；允许扩展L。除了直接配置正整数hidden，首版不提供新激活或结构候选开关。

初始化记录与实际初始化函数的边界：本轮没有构造任何新权重。`public_init`描述旧front规范（普通Linear trunc_normal .02、bias0、Down query [M,d] orthogonal、Conv native、norm1/bias0），`kernel_init`描述新Wq/Wk Xavier gain1。记录可保留其他初始化版本/gamma起点以便比较来源，**这不是已经实现了另一种初始化实验**；后续K2/K3必须按批准的默认初始化实际模块，加载后不重新应用这些起点。

### 两个profile

| task键 | 主`kcdno_v1` d/h/M | 点模块 | 两latent及point hidden | `transolver_shape_match`相对主配置 |
|---|---|---|---|---|
| darcy | 128/8/64 | conv_ffn | 256/256/256 | 相同 |
| elasticity | 128/8/64 | point_ffn | 256/256/256 | 相同 |
| airfoil | 128/4/64 | conv_ffn | 256/256/256 | h8 |
| pipe | 128/4/32 | conv_ffn | 256/256/256 | h8、M64 |
| ns | 256/8/64 | conv_ffn | 512/512/512 | M32 |
| plasticity | 128/8/64 | conv_ffn | 256/256/256 | 相同 |
| car | 256/8/64 | point_ffn | 512/512/512 | M32 |
| airfrans | 256/8/64 | point_ffn | 512/512/512 | M32 |

均L8/r16/history=all。16份完整值见[profiles.json](kcdno_audit/k1/profiles.json)。这里只记录结构，不复制或更改epochs/batch/lr/fold等训练预设；运行类默认batch1仅为泛用记录默认，**不是八任务训练batch的覆盖值**，任务接入必须记录其原实际batch。

### 显式参数与train/eval优先级

`explicit_arguments(parser, argv)`深拷贝parser，清空`_defaults`并将每个action.default设为`argparse.SUPPRESS`，再用argparse自身解析alias、类型、`--flag=value`和同flag最后覆盖规则。原parser完全不改。测试直接提取真实exp/main的parser声明AST，不执行顶层数据/模型/输出代码。

新家族唯一模型键为`kcdno`，显示名KCDNO。PDE/Air的`model`和Car实际的`cfd_model`映射到同一family；没有额外大写别名或构造器占位。`family_for_model_key()`遇到其他键返回None，表示将来交还旧selector，由旧selector处理合法名/未知名，本轮尚未接线。

训练：**显式CLI > 所选profile > 新家族默认**。`n_layers→L,n_hidden→d,n_heads/n_head→h,slice_num→M,dropout→attention_dropout`，canonical字段只有一套；两个不同alias给冲突值会报错。不能传完整旧Namespace冒充显式参数。旧parser裸默认L3/d64/M32不会覆盖profile。例：Pipe只显式profile=transolver_shape_match即得到h8/M64；再显式slice_num=96才得到M96。改d后未指定hidden按最终d重算2d。

显式front/rear/CDPA参数（含F/P/front_blocks/front_latent_mode/latent_ffn_ratio/cdpa_mode/source chunk等）拒绝，指出KCDNO无前后段；旧`mlp_ratio/ffn_ratio`也不隐式转为两FFN的多个来源。不把Car原几何参数`r`当作kernel_rank。旧任务其余路径/训练参数留给原入口处理。

评估：`load_metadata(path)`先读且校验已有完整metadata → 从已保存architecture重建配置 → 只比较用户显式架构字段。错误family、r、history_mode、activation、task等明确拒绝；缺文件先报FileNotFoundError，不先采用CLI默认或创建文件。eval即使显式传了有效profile，也只记录`requested_profile`，不展开profile默认覆盖已有L/d/M/r；要断言形状须传显式结构参数。旧profile名本身不能取代resolved值，保存的profile始终保留。

`checkpoint_family(payload)`只读外层family。缺family返回None，沿旧规则处理，绝不从内部字段/权重形状猜成KCDNO；强行交给新loader会拒绝。当前旧CDLNO loader仍原样支持其严格字段及pre-A1完整full唯一例外。

本轮**没有新薄训练脚本或八任务CLI选项接入**。后续脚本应只用profile给默认结构，再将用户余参最后传入，不能把主profile全量展开为显式参数后阻碍profile切换。

### 公式与配置对应（仅合同，数学尚未实现）

| 后续规格计算 | K1对应字段/参数归属 |
|---|---|
| Down→U=S+FFN1(N1(S))→history→T=Uhat+FFN2(N2(Uhat))→Up | L个完整block；S/U/T[B,M,d]；独立ffn1/ffn2 hidden；T仍是FFN2后Up norm前 |
| Ks=phi(RMSk(Ts) Wk_s) | `d,kernel_rank,norm_eps,kernel_projection_bias`；Wk形状[d,r]属于源s，PyTorch Linear实际weight将是[r,d] |
| cache=(KsᵀTs,sum_token Ks) | v1固定sum、raw Ts；[B,r,d]与[B,r]；不是mean、不增加Wv/Wo |
| Ql=phi(RMSq(Ul)Wq_l)，Rls=Ql cache.matrix/(Ql cache.mass+eps) | Wq属接收l；Q[B,M,r]，每来源读取[B,M,d]；`kernel_phi/clamp/denominator_eps`明确，FP32为v1生产合同 |
| alpha=softmax_source(wᵀRMSdepth(R))，Uhat=U+gamma(C-U) | 无约束每接收层标量gate；w0、gamma.1、norm1记录训练起点；source权重[B,M,l] |
| L1 / history off | 配置合法；后续不注册闲置history参数，L8/all应7写7Q28读。K1未进行调用计数/数学验收 |

### checkpoint格式与目录基础

metadata分别记录六PDE `state_dict`、Car `whole_model`、Air `model_list`或成员`whole_model`；没有torch.load/save调用、没有扩大可信pickle边界、没有strict=False或旧到新权重转换。现有whole-model类路径不改。

`save_metadata`使用文件`open('x')`拒绝覆盖现有sidecar，eval只读。`new_run_path(root,task,architecture)`仅返回 `output/<task>/kcdno/<UTC时间戳>_L…_d…_h…_M…_r…_<history>` 建议路径，不创建目录，不改变旧`output/<task>/<timestamp>`。后续入口仍必须按原保护规则独占预留目录。

K1 sidecar是**核心配置基础**；尚未记录新wrapper的reference/grid/time/placeholder/output完整合同或绑定真实权重文件。K5–K8接入前必须补充这些任务语义并严格检查；不能凭本轮JSON测试宣称真实任务KCDNO checkpoint已可恢复。V1公共resume基础和V2–V5未接入状态保持，没有新增resume协议。

## D. 实际验证命令与结果

实际环境：Python3.13.9；torch实际运行版本2.13.0+cu130、CUDA build13.0；RTX5090 Laptop；PyG2.3.1。目标仍是用户远端Python3.10/3.11、Torch2.11、CUDA12.8，本轮未远端执行、未安装/重装依赖。

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 新配置/metadata | 13/13通过 | 正/负配置、两profile八任务、显式优先、序列化、错误family/r/history/activation、sidecar不改写、legacy分流 |
| 最终定向套件 | 26/26通过，7.201s | 新13 + 旧core/config6 + 既有冻结7；没有修改旧测试 |
| 修改前/后旧同权重回放 | 各41/41通过 | 同一K0权重/输入、state键/shape/输出/诊断梯度；同backend零容差，chunk变化沿用1e-5/3e-4 |
| 旧CPU合成 | 33份CDLNO通过 | 24八任务×三front模式 + 9front×CDPA核心；不使用新模型 |
| 旧GPU合成 | 8份原Transolver通过 | 本机CUDA FP32/mathSDPA，非新KCDNO GPU验收 |
| 旧真实PyG与checkpoint | Car/Air旧wrapper对象、整模型/列表加载回放通过 | 不含真实文件读取/radius_graph/物理指标；未运行新的KCDNO权重加载 |
| 新配置三原cwd子进程 | 3/3通过 | 显式PYTHONPATH，import/config/metadata读入；确认不导入torch或入口 |
| Python3.10语法/包发现 | 新5模块+测试共6文件通过；find_packages发现cdlno.kcdno | 语法/静态包发现不是实际Python3.10或editable安装验收 |
| 真实训练/数据/新模型/GPU性能 | 未运行 | 本轮无新数学模型；无下载、数据集训练或指标声明 |

首次定向运行有1个测试错误：新测试把Car parser误按`--model`调用，实际选项为`--cfd_model`。已修正测试并补齐新协议对`cfd_model`的family校验，未改Car原入口；最终13/13及26/26通过。保留[初次日志](kcdno_audit/k1/config-tests-initial.txt)，不将失败写成通过。

可复核命令（从仓库根目录；不会import数据入口）：

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_config -v

python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result /tmp/k1-old-model-replay.json

# 仅配置示例，不构造/训练模型；Pipe匹配profile为h8/M64。
python -B -c 'from cdlno.kcdno.profiles import resolve_profile; print(resolve_profile("pipe", "transolver_shape_match").to_dict())'
```

完整最终26项命令见[commands.txt](kcdno_audit/k1/commands.txt)，结果[final-tests.txt](kcdno_audit/k1/final-tests.txt)，包发现[package-check.json](kcdno_audit/k1/package-check.json)。旧夹具只能replay，不能覆盖capture目录。

## E. 冻结范围与旧问题

371个K1起点tracked文件中，最终仅独立KCDNO STATUS与memory增量更新；其余369文件SHA256完全相同。模型数学、旧CDLNO F/P/front模式/CDPA、数据/采样/split/fold/normalizer/loss/时间循环/optimizer/scheduler/指标、旧Transolver模型/脚本、三项目入口/配置、旧checkpoint、依赖/安装声明和K0审计夹具均未改变。

K0已记录的Car外层日志命名/固定阻力路径、Air旧MAE判断、旧裸parser别名、尚未完成任务resume/周期图等问题没有顺手修复，也不归为KCDNO回归。本机未editable安装和缺邻居图扩展不阻止本轮纯配置/旧对象回放，但不能充当远端环境验收。

## F. 自审结果与后续依赖

已完成四点自审：

1. **新旧隔离**：新文件只在子包、测试和证据；旧371文件冻结检查、41份前后回放和稳定pickle路径检查通过。
2. **profile与显式覆盖**：真实parser AST证明旧默认不进入显式字典，Car cfd_model单独核对；16份resolved值与规格表一致，d覆盖后hidden重算，原profile未改。
3. **eval/family/兼容**：先读、完整字段、family双重校验、legacy None分流、错误结构拒绝、runtime差异报告、只读byte检查通过。profile/初始化起点不会覆盖保存结构或已学值。
4. **授权边界**：没有核公式执行、模型构造器、factory注册、训练启动器、新resume或任务wrapper接线。新JSON的检查不被表述为新模型数学/权重验收。

没有需要先改变旧baseline或用户裁定的架构冲突。后续依赖：K2核writer/reader的数学与初始化实测；K3完整block/core；随后按用户授权接入wrapper完整metadata/严格权重加载、输出预留和原任务训练参数。仅本K1完成，不自动推进这些阶段。

本K阶段结束，未执行下一阶段。
