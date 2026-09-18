# LinearNO L0：固定来源与当前仓库只读审计

日期：2026-09-17。范围：用户本轮 L0；`RUN_REAL_BATCH=false`、`RUN_MINIRUN=false`、`MAX_STEPS=0`。

本报告审计的是 **`/home/hwz/CDLNO` 当前实际 checkout**，不是提示词中另一仓库的 `66bc489` 快照。没有实施 LinearNO，没有修改任何既存生产文件、launcher、依赖或测试。配置/公式复现明细见 [REPRODUCTION_MATRIX](LINEARNO_REPRODUCTION_MATRIX.md)，阶段结论见 [STATUS](LINEARNO_IMPLEMENTATION_STATUS.md)。文中的拟建路径、字段和命令模板均为后续阶段建议，不表示现在已有可运行实现。

## 1. 基线、指令、来源

已核对根 `AGENTS.md`、当前 README、三任务 README、现有 CDLNO/KCDNO/MSAR 实施/阶段报告、`memory/current-state.md` 与 LinearNO 提示词。根 AGENTS 的 2026-09-15 状态是历史记录：当前代码已包含 KCDNO、MSAR 和周期可视化。没有找到更深层适用的 AGENTS/CLAUDE。本轮用户 L0 的允许文件范围优先，因此不更新旧模型 STATUS、memory 或 AGENTS。

| 来源 | 实际固定 commit | tree | 最后提交 / 取得方式 |
|---|---|---|---|
| 当前目标 `hxh5159/CDLNO-w` | `bb73b3099d3b8ce45bd939156b737453b9ca5454` | `cfd5a28549cc109e8c390a1865b315e9d8b7fa92` | `006`；2026-09-17 03:44:49 +08；当前 `main` |
| `thuml/Transolver` | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` | `22e7386201596e45236907f774478f79d3913da7` | 2026-02-26；本地已有 Git 对象，按固定树导出到目标外 |
| `HiPRL/LinearNO` | `3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` | `4aa99d7dece8462d932e690a82e23dba78b11b0b` | `UPDATE README`；2026-03-24；`/home/hwz/LinearNO` 固定树导出 |
| 论文 v3 | [arXiv:2511.06294v3](https://arxiv.org/html/2511.06294v3) | HTML SHA-256 `637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd` | 2026-09-17 11:36:49 UTC 取得 HTML，阅读正文及 A—F 附录 |

目标 remote 为 `git@github.com:hxh5159/CDLNO-w.git`；不是声明的 `hxh5159/Transolver`。采用实际 HEAD 为 L0 基线；没有 merge-base、三点 diff、merge/rebase、checkout/reset/clean/stash。固定 Transolver 对象存在不等于需要对两树作祖先关系假定。

起点 910 tracked 文件，tracked/staged diff 均空；已有两份 untracked 文件，完整字节保留：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `PLAN_LinearNO/LinearNO_Codex_Staged_Prompts_hxh_Transolver.md` | 67856 | `0bb6a39477d08df57c164e9852ebbfa34c9d6082ba4de9c5bc1fa0c96d822b6c` |
| `PLAN_LinearNO/比较模型架构 (2).md` | 1030963 | `8b3521423463ab12761cf707b136f7f2d9899be2cdc246e06f9e37b173c7b980` |

完整机器清单见 [manifest](linearno_audit/l0/manifest.json)。249 个原有 ignored 文件也列入基线，不能把缓存中的用户文件当成本任务可清理物。

完整外部快照：`/home/hwz/CDLNO-artifacts/linearno-l0-before-65a2wvms`。其中 `source/` 保存当前文本；`references/transolver/`、`references/linearno/` 是固定树只读参考；二进制记录哈希，不解包或充当源码。两仓都未被 checkout 到其他版本。详细 SHA、remote、状态和来源在 [baseline](linearno_audit/l0/baseline.json)、[sources](linearno_audit/l0/sources.json)。

许可：当前目标/Transolver 根 MIT；AirfRANS 子目录 ODbL 和当前第三方 NOTICE 均保留。**固定 LinearNO 树没有 LICENSE/NOTICE 文件**，不能把公开可读等同于已有任意再分发许可。后续优先依据公式独立实现、保留出处、把固定源作为外部 parity 参考，不整树 vendoring；如需分发较多源码再单独核实许可。此为来源与分发风险记录，不作法律结论。

## 2. 阅读覆盖与证据边界

使用 `rg --files`、`git ls-files`、`git ls-tree -rl <SHA>` 获取全清单，对每个文本读取完整内容。Python 完整 AST 遍历记录全部类/函数/签名、调用、测试断言和顶层副作用；配置/shell 全文及 notebook code cell 独立提取；旧 JSON/patch/log 作为历史证据结构化检查，不能拿旧运行结果当本轮新通过。关键模型、八入口、loss/time/data/checkpoint、两官方差异和论文条款另做语义核对。

| 树 | tracked 总数 | 独立全文/结构读取 | 相同 SHA 分组代表 | 二进制排除 | 可读文本覆盖 |
|---|---:|---:|---:|---:|---:|
| 当前目标 | 910 | 795 | 90 | 25 | 885/885 |
| Transolver 固定树 | 71 | 16 | 43 | 12 | 59/59 |
| LinearNO 固定树 | 61 | 60 | 1 | 0 | 61/61 |

共 255 份唯一 Python AST、138 份配置/shell/notebook 提取记录、84 份唯一文档记录。逐文件 `read/group/excluded/reason`、size、SHA、Git blob 与分组代表见 [reading-inventory](linearno_audit/l0/reading-inventory.json)。分组仅以**全文 SHA-256 完全相等**为依据；例如 CRLF 不同的官方文件不会因“看起来相似”而跳过。另有 [AST 对照](linearno_audit/l0/ast-comparison.json)，AST 一致仅用于判断语义差异，不冒充字节一致。[关键符号行号](linearno_audit/l0/source-anchors.json)给出32个模型/训练/指标/存档符号的固定源码定位。

这里的 100% 是文本及结构阅读覆盖率，不是“所有代码执行通过”或“逐行数学正确性证明”。[source-contracts](linearno_audit/l0/source-contracts.json) 保存符号证据；[configuration-shells](linearno_audit/l0/configuration-shells.json) 保存配置/argv；[document-sections](linearno_audit/l0/document-sections.json) 为文档索引。两份 dataset_stats notebook 的唯一 code cell 均已读取：从 manifest 作数据统计和绘图，不属于模型/训练，不执行。

排除项逐文件在清单：图片、PDF、NPZ 等二进制（包括用户设计附件和历史合成图）只记 size/hash；`.git` 对象内部、`__pycache__`、真实数据/权重、构建产物不当源码阅读。没有解压 zip/wheel、读真实样本、下载 checkpoint。`Super-Resoltion-AppendixE` 的模型、训练/测试、工具、数据生成器和全部五个配置已审阅，但不接入八任务，也不运行 Burgers/分布式训练。

## 3. 当前结构与模型注册

```text
/home/hwz/CDLNO
├── Physics_Attention.py                     根参考，不是八任务运行 import
├── PDE-Solving-StandardBenchmark/
│   ├── exp_{darcy,elas,airfoil,pipe,ns,plas}.py
│   ├── model_dict.py -> 各 module.Model
│   ├── {cdlno,kcdno,msar}_entry.py           薄 family 解析/Run 接口
│   ├── model/                               原模型及新增薄 wrapper
│   └── utils/{normalizer,testloss}.py        原任务数学工具
├── Car-Design-ShapeNetCar/
│   ├── main.py / main_evaluation.py / train.py
│   ├── models/                              tuple(Data,geom) 合同
│   └── dataset/、utils/drag_coefficient.py
├── Airfoil-Design-AirfRANS/
│   ├── main.py / main_evaluation.py / train.py / params.yaml
│   └── models/、dataset/、utils/metrics*.py   Data/随机采样/全场散射
├── cdlno/                                   现有共享包
│   ├── config/modules/core/cdpa/checkpoint
│   ├── kcdno/、msar_lno/                     已有独立 family
│   └── experiment/training_state/periodic_visualization
├── tran_evaluate/                           当前真正 launcher 位置
│   ├── kcdlno/、msar_lno/、ablation/、show/
├── tests/、tools/cdlno_perf/、tools/*benchmark.py
└── docs/、PLAN_*/、memory/                    阶段史与证据
```

**实际不存在** `LINEARNO/`、`train_and_evaluate/`、`CODEX/train_and_evaluate/`、`evaluate/`；因而没有既存 PropagationMonitor 可运行。不得新造这些目录/假测试来满足旧提示词。冻结 manifest 记录其起点不存在，后续新增也必须识别。

| 模型/注册名 | 当前选择与真实类 | 合同/边界 |
|---|---|---|
| 原 Transolver Standard | `model_dict.get_model` 的完整 key：`Transolver_Irregular_Mesh`、`Transolver_Structured_Mesh_2D`、`Transolver_Structured_Mesh_3D` → 对应 `model.*.Model` | `forward(x,fx,T=None)`；旧 parser 的 `Transolver_1D/2D` 是历史无效默认 key，原 shell 显式传正确 key；不借 L0 修复 |
| 原 Car Transolver | `main.py --cfd_model Transolver` → `models.Transolver.Model` | tuple `(cfd_data,geom)`，返回 `[N,4]` |
| 原 Air Transolver及图模型 | `main.py --model`/YAML → `models.Transolver.Transolver`、`MLP/GraphSAGE/PointNet/GUNet`（NN 为内部工具） | `forward(Data)`；保留图模型/图构造 |
| CDLNO | `--model CDLNO` / Car `--cfd_model CDLNO` → `cdlno.standard.{StaticStandardModel,TemporalStandardModel}`、Car `models.CDLNO.Model`、`cdlno.airfrans.AirfRANSModel` | core `cdlno.core.CDLNO`；front `full/no_sa/identity`；CDPA `off/entry/every_block`，独立参数、旧合同不改 |
| KCDNO | key `kcdno` → Standard薄模块、Car `models.KCDNO.Model`、`cdlno.kcdno.airfrans.AirfRANSModel` | core `KCDNO`，`history_mode=all/off`，核摘要不属于纯 LinearNO |
| matched LRSA | key `lrsa_matched` 复用 KCDNO entry/wrapper 接口选择 `cdlno.kcdno.matched.MatchedLRSA` | 独立完整 LRSA block；另有旧性能工具 `tools.cdlno_perf.models.LRSAMatched`，不是本任务的 `transolver_matched` profile |
| MSAR-LNO | key `msar_lno` → `model.MSAR_Standard/Temporal`、`cdlno.msar_lno.industrial.{CarModel,AirfRANSModel}` | core `MSARLNO` Light/Full；显式 `return_aux`/objective adapter，仅其路径有 coverage |
| 纯 LinearNO | **尚未注册/实现** | 拟新增 `linearno` family 和任务本地新模型文件；不复用其他 family 名称 |

配置走 family-specific parser → resolved config → 任务 wrapper → core。`pyproject.toml` 只安装 `cdlno*`，限制 Python `>=3.10,<3.12`；当前本地是源目录导入，不是安装成功的兼容证明。若新增共享协议辅助，可在现有包内隔离命名空间；新 LinearNO 模型仍放三个 benchmark 模型目录，避免重建工程或修改旧包安装合同。

## 4. hxh ↔ 固定 Transolver：字节和语义比较

[tree-to-tree 清单](linearno_audit/l0/transolver-comparison.json)：固定官方 71 文件中 **55 相同 blob、16 改动、0 删除**，当前另有 839 新增 tracked 文件。13 个原有 Python model/Physics-Attention/Embedding 文件全部字节相同（包括 Air 的其他模型；原核心 attention 的温度、slice SA、位置、初始化没被此次新增 family 重写）。[普通 diff](linearno_audit/l0/upstream-to-target.diff) 保存完整差异。

16 个不同文件为 `.gitignore`、README、六个 `exp_*.py`、`model_dict.py`、Car 的 main/main_evaluation/train、Air 的 main/main_evaluation/train/params.yaml。

| 差异类别 | 实际效果 | 旧路径判断 |
|---|---|---|
| 独立 family 分支/kwargs | CDLNO、KCDNO、matched LRSA、MSAR 选择与 aux 仅指定分支 | 不把新 kwargs 传给 Transolver |
| experiment/checkpoint/viz | 新 family 提前 reserve run、配置/状态/JSONL、周期场图、独立 eval 目录 | 原旧模型输出仍有旧 checkpoints/metrics 路径，不能宣称全仓已统一 |
| seed | KCDNO/matched/MSAR 受限显式种子支持 | 原 Standard `--seed` 并非普遍支持；需未来 LinearNO 独立接线 |
| 路径/设备兼容 | 新 launcher 适配远端；Car eval epoch 整数及可信本地 whole-object `map_location/weights_only=False` | 不是模型数学改动；保持局部信任边界 |
| industrial train callback | 记录/可视化回调；MSAR 显式训练目标分支 | 旧 loss、scheduler、sampling 通过完整 AST 投影检查 |
| README/ignore/YAML | 文档增量、output忽略、新模型 YAML key | 旧 YAML 配置/原脚本保留，Air 原 Transolver398epochs |

本轮重跑 6 项现有源码冻结测试，**6/6通过，unittest 0.483秒，进程3.944秒**，覆盖六 Standard 及四工业 train/eval入口、两工业 train、factory 和旧文件字节。测试会剥离**精确的已知新增分支**并比较完整 AST，不是简单删全部 if；投影器本身也在全源码清单中。详见 [原始输出](linearno_audit/l0/legacy-static-tests.txt)、[命令](linearno_audit/l0/legacy-static-results.json)。这支持“原始源码和投影后的旧任务逻辑保留”，不替代 L1 同权重 forward/gradient/RNG 回归。

Car 表面速度修复**已在固定 thuml 75e0f676 内**，当前 hxh 保留；LinearNO release 相比它又传了 volume velocity 给 surface drag helper。不要把这个差异误归因于本轮或认为 hxh 的 drag 路径/fold 已全部修好。

## 5. 八任务真实入口/数据/loss/训练合同

表中 shape 是实际任务合同；N 来自数据，未在本轮读取数据验证。PDE 六题 factory 见上表，模型调用均为 tensor；工业不得展平多图后跨几何 attention。当前工业正式合同是单图 batch1、可变N；旧类并不自动隔离多图，新模型需拒绝不可靠的 batch>1。

| 任务/入口 | 模型调用、输入→输出 | 数据/split/normalizer | 原训练与评价 |
|---|---|---|---|
| Darcy `exp_darcy.py` | `model(x,fx=fx.unsqueeze(-1))`；x `[B,7225,2]`、fx1→out1；85×85 | smooth1/2 MAT coeff/sol；1000/200；stride5；x/y `UnitTransformer` 训练拟合，std+1e-8 | decode预测/标签→逐样本rL2和 +0.1×dx/dy导数rL2；l2先算，**仅导数分支**预测裁内边再pad0，central_diff零padding，dx=1/85；eval物理rL2 |
| Elasticity `exp_elas.py` | `model(x,None)`；`[B,972,2]→[B,972,1]` | Random_UnitCell_XY_10/sigma_10.npy；XY转置(2,0,1)、sigma(1,0)；first1000/last200；只y归一化 | 预测/训练y decode后rL2；eval physical rL2 |
| Airfoil `exp_airfoil.py` | `model(x,None)`；221×51 xy→Mach1 | NACA_Cylinder_X/Y/Q.npy；Q通道4；first1000/next200；无normalizer | 原rL2；当前每epoch测试；独立eval有原固定裁剪场图 |
| Pipe `exp_pipe.py` | `model(x,None)`；129×129 xy→velocity1 | Pipe_X/Y/Q.npy；Q通道0；**先截前1200再first1000/last200**；x/y都归一化 | decode y后rL2；不能把normalized坐标换成原坐标传模型 |
| NS `exp_ns.py` | `model(x,fx)`；64² xy、fx10→out1 | MAT u；first1000/last200；前10帧输入、后10帧目标；无归一化 | train10次真值回填，loss求和→1 backward/optimizer/scheduler；eval10次预测回填；分别 step/full rL2，不跨时间缓存 |
| Plasticity `exp_plas.py` | `model(x,fx,T=input_T)`；101×31 xy、fx1、T `[B,1]`→4 | plas_N987_T20.mat input/output；900/80；输入力沿空间重复并归一化；output原转置/meshgrid点序保持 | 20时间点，collate **逐样本torch.randperm**；20次独立optimizer、每outer batch仅1scheduler；eval原时间顺序，step及full rL2 |
| Car `main.py/train.py/main_evaluation.py` | `model((cfd_data,geom))`；x `[N,7]=xyz+sdf+normal`→`[N,4]=vxyz,p`；geom仍由loader产生 | param0…8 VTK/预处理x,y,pos,surf,edge_index.npy；defaultfold0；coef_norm用训练集拟合；`os.listdir`顺序和Python random几何采样保持 | normalized **全点**velocity3均值MSE +.5×surface pressure MSE；test/train原loss；正式eval物理surface-p/non-surface-v rL2、normalized MSE派生RMSE、drag/Spearman |
| Air `main.py/train.py/main_evaluation.py` | `model(Data)`；x7=xy+inlet2+distance1+normal2，pos2原始，out4=vx,vy,p,nut；unified额外64距离 | Dataset/manifest；full800训练池末10%验证=720/80，200test；vtu/vtp `implicit_distance`等字段；coef_norm训练拟合；每epoch随机32k | normalized四通道volume MSE +1×surface MSE；每10epoch/末轮验证且20次重采样；eval迭代子采样散射到全场、重复点均值、MSE及物理力系数 |

Standard `TestLoss(size_average=False)` 是每个样本展平后无epsilon的比值，**batch求和用于反传**，epoch除样本数；不能照提示词的“平均”改成训练batch mean后仍称同一优化轨迹。Standard全部AdamW，Elasticity Cosine每epoch；其余OneCycle每outer batch。NS无clip默认；Plasticity没有把T并入N。

工业均Adam；Car OneCycle `total_steps=(len(train)//batch+1)*epochs, final_div_factor=1000`、drop_last=True；Air同total_steps公式、默认final_div_factor=10000。这不是简单epochs×实际更新数，必须保存resolved值。Car `train()` 返回pressure再velocity，外层变量反名形成错误日志加权表达，**内部反传目标正确**；后续LinearNO真实目标和legacy日志值应各自命名，不能全局修旧日志。

Air `Results_test(...,n_test=3)` 的3仅用于随机选VTK展示样本；指标遍历完整test manifest。`Infer_test`按坐标集合覆盖、随机32000节点、累计散射/重复点均值，仍有radius_graph；prediction的surface velocity/nut设物理0。`score.json`是normalized四通道MSE及系数误差/Spearman，不能因论文Table2标题写rL2就给MSE改名。Car正式eval确实同时计算物理rL2和MSE/RMSE，不能把两个工业任务概括为“全部只算MSE”。

### import副作用与产物路径

六个 `exp_*` 的真实数据读取在 guarded main，但 **模块顶层parse argv、设置设备及新run记录副作用存在**；四个工业main/main_evaluation直接顶层读manifest/数据或执行流程。均不 import/不直接跑 `--help` 做本轮核查。当前只提取AST解析参数；安全模型、normalizer、loss模块与经检查的train函数才可供后续合成测试。

| 路径 | 保存/选择/再评估 |
|---|---|
| 原Standard | `./checkpoints/<save_name>.pt` bare state_dict；同exp `--eval1`；Darcy/NS/Pipe/Plas旧加载有strict=False，Pipe还有resave副作用，**新LinearNO不得继承** |
| 原Car | `metrics/<cfd_model>/<fold>/<epochs>_<weight>/model_<epochs>.pth` whole object；`main_evaluation.py`；固定drag helper还依赖旧root/param0 |
| 原Air | 每成员whole model + models列表；训练可score，独立eval按旧metrics结构；不是所有成员都天然有唯一目录 |
| 新family当前基础 | Standard各Run `model.pt`+architecture.json；Car whole object；Air `member_NNN/model`和family列表；`cdlno.experiment`记录config/train.log/epochs JSONL/result，eval独立目录/index且不改训练sidecar |
| resume现状 | `cdlno.training_state.TrainingArchive`有独立基础、绑定CDLNO sidecar，**八任务现有entry没有完整resume接线**；仅weights/whole model不足恢复optimizer/scheduler/RNG |
| visualization现状 | `periodic_visualization.PeriodicFields`、`training_state.isolated_evaluation`可参考RNG隔离；`tran_evaluate/show`当前按最后验证loss选单seed，只是用户展示工具，**不可直接用于LinearNO三seed论文汇总** |

## 6. 可复用边界与冻结

| 新LinearNO需要 | 现有真实位置 | 可复用程度/理由 |
|---|---|---|
| 输入字段、split、loss工具、decode、time loop | 八exp/main、dataset、normalizer/TestLoss、工业train/metrics | 原合同复用；profile差异另设LinearNO分支，旧路径不变 |
| attention/block/norm | 原Physics_Attention、`cdlno.modules`、KCDNO/MSAR原语 | **不能直接复用整attention/block**：sliceSA、QKRMSNorm、Down/Up、两个FFN、历史/coverage都非纯LinearNO；仅可参考tensor布局 |
| MLP/位置/timestep | 三任务原Transolver及官方LinearNO专属文件 | 数学细节可参考，但需新模块保持release key/初始化；不能把RMSNorm/eps1e-6搬入release LayerNorm默认eps1e-5 |
| 配置显式CLI识别 | `cdlno.kcdno.options`、`cdlno.msar_lno.options` 的SUPPRESS probe | 可复用解析思想；不能继承旧M/F/P/history等字段；`--linearno-profile/variant/rank`独立最薄接线 |
| output/atomic记录 | `cdlno.experiment.{reserve_directory,write_json,Experiment}` | 需新增明确family支持或薄adapter；现有Experiment识别有限family，不能仅调用start就以为LinearNO路径正确 |
| metadata/strict load | 各family metadata/Run | 参考read-first/no-sidecar-write规则；新spec更完整，不能直接套旧schema假装保存normalizer/RNG |
| resume/RNG | `cdlno.training_state`、EpochObserver | RNG隔离/原子保存可复用；任务resume是**新增接入工作**，须L4—L7授权时落实，不能L1越界 |
| 性能 | `tools/cdlno_perf/{costs,measure}.py` | 可借计时/同步口径，LinearNO新算子需独立成本映射；L9不可临时改模型/测试 |

[freeze-manifest](linearno_audit/l0/freeze-manifest.json) 包含既存保护路径221文件（185 tracked、36 ignored），保存原 `git status --porcelain=v1 --untracked-files=all -- <paths>`、缺失目录、每文件分类/bytes/SHA。更宽的 [manifest](linearno_audit/l0/manifest.json) 保护全部1161既存文件，**包括两个untracked计划和249ignored**。本轮文件哈希对照及新增文件清单另见交付核验 `delivery-check.json`。

后续新增与变化必须对L0清单比较，不能只看git diff；冻结`LINEARNO/**`即使起点为空也不得把新模型放进去。新产物可使用的ignored目录见 [ignored-output-policy](linearno_audit/l0/ignored-output-policy.json)：仅专属 `output/_linearno_audit/` 和 `output/<八任务>/linearno/` 子树；不创建通用checkpoints/metrics来覆盖旧结果。临时只读分析和小夹具优先目标外专属artifact，`python -B`避免修改既存缓存。此清单不授权真实训练或L0生成模型权重。

旧数值证据索引可复用 `docs/kcdno_audit/fixture_index.json`、`docs/msar_lno_audit/m0/fixture-index.json`、M8 index及A1真实pre-change whole-object。L0只核对索引/存在/哈希，不加载pickle、不重生成权重，不把历史exact replay写成本轮通过；[本轮索引](linearno_audit/l0/regression-index.json)保存核查记录，L1先建立本L0源码版本的有效对应关系。

## 7. 环境、已执行命令与限制

[environment](linearno_audit/l0/environment.json)：Python3.13.9；Torch2.13.0+cu130/CUDA build13.0；RTX5090 Laptop GPU×1；PyG2.3.1可import；timm1.0.28/einops0.8.2/NumPy2.2.6/SciPy1.16.3/VTK9.6.2/PyVista0.48.4。torch_cluster、torch_scatter、pyg-lib分发未安装。没有重装/升级依赖，没有GPU模型计算。

官方Transolver Standard requirements含torch1.10.1、einops0.6.1、scipy1.7.3、timm0.9.2；工业列表较宽松并要求torch-cluster。LinearNO environment.yml记录Python3.12、Torch2.4/cu118、PyG依赖、timm1.0.14；还混有cuda-version12.6条目，不能简写为“官方就是用户cu128环境”。远端Python3.10/Torch2.11/cu128/PyG实际版本尚未本轮验证。

| 实际命令/检查 | 本轮结果 |
|---|---|
| `git rev-parse --show-toplevel; git remote -v; git branch --show-current; git rev-parse HEAD HEAD^{tree}; git show -s ...`（独立调用） | 实际基线如§1；未用假定66bc489 |
| `git status --porcelain=v1 --untracked-files=all`、`git diff`、`git diff --cached`、`git ls-files -z --others --ignored --exclude-standard` | 起点2untracked、249ignored、tracked/staged diff空；原始文件留存 |
| `rg --files` / `git ls-tree -rl <固定SHA>` / SHA-256读取 | 三树清单、逐文件分组、冻结manifest完成 |
| `python -B <外部artifact>/audit_scan.py` | 全文/AST清单及普通双树diff，见§2 |
| AST提取原始 `parser.add_argument` 后解析六份release shell argv | 6/6成功，**六份均eval=1**；保存parser default与shell resolved两列，不调用exp |
| 全三树Python AST parse和shell `bash -n` | **384/384成功（279 Python、105 shell）**；见[syntax-checks](linearno_audit/l0/syntax-checks.json)，只是语法，不证明有数据可跑 |
| `PYTHONPATH=tests:. python -B -m unittest <6个既存冻结方法> -v` | **6/6通过**；完整方法名见命令JSON；有timm FutureWarning，无失败/错误 |
| 固定源码→移植模型forward/gradient/optimizer、strict新checkpoint | **NOT RUN：L0未实现模型/协议**；不是已有parity通过 |
| GPU tensor/AMP/PyG合成模型/真实loader/完整训练/精度 | **NOT RUN：阶段未授权且无真实数据**；PyG只import，GPU只设备查询 |

后续可执行CPU数学oracle/合成模型/原loss、local PyG Data及strict状态往返；GPU存在但需对应L阶段授权。没有torch_cluster会阻止当前工业完整radius_graph链，不能制造假PyG包/跳过图构造后声称完整入口通过。远端可先只读执行 `python -c "import torch,torch_geometric; print(torch.__version__,torch.version.cuda,torch_geometric.__version__,torch.cuda.is_available())"`；具体新parity命令在L2—L7落地后给出，本轮不捏造不存在的测试入口。

## 8. L1—L10最小变更预测（非执行授权）

| 阶段 | 预计最小位置 | 与实际代码适配/前置条件 |
|---|---|---|
| L1 | 新独立schema/profile协议模块（可在现有包新命名空间）、新tests/linearno、文档 | 不接现有parser/factory；复用有效小fixture、补当前Transolver/产物RNG基线；不存在的monitor标N/A；Plasticity用Torch随机排列 |
| L2 | benchmark新LinearNO原语/独立oracle与tests | Standard四变体、两工业原语；旧attention文件冻结；以公式重写，外部固定源码parity |
| L3 | `PDE-Solving-StandardBenchmark/model/LinearNO.py`（拟）、tests | 完整Standard模型及严格初始化/逐层parity；不改exp |
| L4 | `model_dict.py`独立分支、cdlno_entry薄分派、新linearno_entry、四静态exp、平行launcher/新spec Run | artifact接入、normalizer序列化、**新family完整resume**不是原生现成能力；保留旧分支；Airfoil cadence/Pipe split差异按冲突台账 |
| L5 | 两时间exp、新LinearNO时间run/launcher/tests | 10次真值回填/预测回填；20次Torch时间排列更新；一并完成新familyresume/eval链 |
| L6 | Air新models文件、新entry/metadata、main/eval/train最小分支、params新key、平行脚本 | 先确认论文工业objective细节；图构造只允许清楚的LinearNO-specific差异；complete-field指标/ensemble/normalizer |
| L7 | Car新models文件、新entry、main/eval/train分支、新family force adapter、脚本 | 先确认paper objective；surface velocity与真实样本路径可分开记录；不全局改旧drag helper |
| L8 | 新外部checkpoint转换工具/逐键映射与集成检查 | 两工业同pickle限定名必须按source_task隔离子进程；不下载/加载未知pickle |
| L9 | 执行已存在新测试/性能入口并产出报告 | 此阶段按计划不写源码，计时工具/shape检查须此前授权阶段准备；资源开关默认false/0 |
| L10 | 独立终审、最终README/报告按该阶段授权 | 无数据精度不可声称复现；不自动训练/commit/push |

需审查的实质事项：工业paper训练rL2定义仍有欠规范字段；当前Car路径并未完全修复；L1/L4中的“已有任务resume”假定不成立；官方Airfoil测试cadence、Pipe边界split及Air新family图构造差异需要明确协议。详情、原文/源码行和两个选择的影响均在复现矩阵冲突台账。未借审计修旧缺陷或选择另一研究架构。

**本L阶段结束，未执行下一阶段。**
