# Standard 六任务实验配置与参数清单

这是当前 checkout 的离线实验设计清单。每个实验一个 `config.json`，没有时间戳；只在 CPU 上构造实际模型、统计参数并校验，不读取数据、不执行 forward/backward、训练或评估。不包含模型权重或虚构实验成绩。

## 先看这些文件

- `architecture_summary.csv`：492 行架构配置（不重复列出 seed），便于 Excel 查看参数量、rank、深度、模式、batch、学习率等。
- `summary.csv`：1,476 行完整实验，包含 seed 和对应 `config.json` 路径。
- `index.json`：同一完整索引及覆盖统计。
- `experiments/`：1,476 个独立的 `config.json`，包含实际参数名/shape/numel、完整构造参数、训练协议、来源、初始化 hash 和参数分项。
- `validation.json`、`catalog_review.json`：参数公式、已有 LL9R 数值、配置覆盖和 CPU 位置参考的核验结果。
- `source_snapshot.json`、`preservation_check.json`：开始时已有 tracked/untracked/ignored 文件的内容摘要及完成后的保持检查。
- `generate_catalog.py`：本次清单生成程序，仅存在于这个新建文件夹。不会编辑仓库原文件，已有清单存在时拒绝重写。无需运行它即可查看所有结果。

## 目录结构

```text
experiments/
  airfoil/                         # 另有 darcy/elasticity/pipe/ns/plasticity
    paper_table8_on_release_model/  # 主比较协议
      base_depth8/                 # 另有 depth_extension/custom_topology_control
        D8/                        # 总执行 block 深度
          M64_x1/                  # 另有 M128_x2；NS 为 M32_x1/M64_x2
            linearno_loop/
              p1_c3_r2_s1/
                sr_1_over_r/       # 另有 rb_attnres/lb_attnres_1_over_r
                  seed0/config.json
                  seed1/config.json
                  seed2/config.json
            linearno/
              independent_L8/native_residual/seed0/config.json
    official_release/              # 8 层补充配置，不混入主协议聚合
    transolver_matched/            # 8 层协议控制，仍是 LinearNO/loop 算子
    transolver_original/           # 真正 Transolver 原生算子和原脚本设置
```

`run_directory` 是未来结果目录的无时间戳预览，没有创建训练运行目录。ZIP/清单目录不是 checkpoint 运行目录，不能直接用于 resume。`production_directory_id`（loop）来自现有目录解析器；正式执行时继续使用原 launcher、路径和 checkpoint 规则。

## 覆盖的实验

| 分组 | 覆盖 | 实验数（含三 seed） |
|---|---|---:|
| 主 profile | 六任务，两个 loop 拓扑×三残差 + 纯 LinearNO；M×1/2；四个静态任务 D8/16/24/32，NS/Plasticity D8 | 756 |
| custom 配置控制 | 六任务，P0/C2/R3/S1，三残差、M×1/2、三 seed | 108 |
| 官方发布 profile 补充 | 六任务 D8，两个拓扑×三残差 + 纯 LinearNO，M×1/2、三 seed | 252 |
| transolver_matched 补充 | 同上，但解析仓库该 profile 的真实设置 | 252 |
| 原生 Transolver 对照 | 四静态任务 D8/16/24/32，NS/Plasticity D8，M×1/2、三初始化 seed | 108 |
| 合计 | 492 种不计 seed 的配置 | **1,476** |

按模型 family：loop 1,188，纯 LinearNO 180，Transolver 108。按任务：Airfoil/Darcy/Elasticity/Pipe 各294，NS/Plasticity 各150。没有加入当前暂不研究的旧 latent A/K 机制，也没有加入本次范围外的工业任务或其他模型 family。

主矩阵及纯 LinearNO 秩控制来自现有 Looped LinearNO 计划与 LL1/LL8 文档。深度扩展来自本次用户要求。新增深度的具体 custom PCRS 以及将 custom 示例配齐两档 rank/三 seed 是本清单明确标注的实验设计，不冒称为原先已有的正式 preset。官方发布与 matched 作为单独的 8 层补充控制，避免将不同 profile 混成一种协议。

## M、物理深度与执行深度

原始 Transolver 的设置按当前 `PDE-Solving-StandardBenchmark/scripts/Transolver_*.sh` 读取：

| 任务 | 原始 M | 双倍 M | 主 LinearNO variant | hidden / heads |
|---|---:|---:|---|---|
| Airfoil | 64 | 128 | conv_temp | 128 / 8 |
| Darcy | 64 | 128 | conv_temp | 128 / 8 |
| Elasticity | 64 | 128 | temp | 128 / 8 |
| Pipe | 64 | 128 | conv_temp | 128 / 8 |
| NS | 32 | 64 | plain | 256 / 8 |
| Plasticity | 64 | 128 | conv | 128 / 8 |

这里 Pipe 的 Transolver 原始 M 是64；另一个 CDLNO preset 中的 M32 不属于本次 LinearNO/Transolver 设置。Standard LinearNO 的 M 是每个 head 的绝对压缩槽数；Transolver 的 M 是 slice 数。二者名称相近，内部算子不同。

独立层基线有 `unique_depth=executed_depth=D`。loop 有 `unique_depth=P+C+S`、`executed_depth=P+C*R+S`，core 参数跨轮共享。四个静态任务的扩展保持 R=2、各拓扑的 P/S 不变，通过增大 C 达到指定执行深度：

| D | 拓扑 A 的 P/C/R/S | A 独立 block | 拓扑 B 的 P/C/R/S | B 独立 block |
|---:|---|---:|---|---:|
| 8 | 1/3/2/1（正式 preset） | 5 | 2/2/2/2（正式 preset） | 6 |
| 16 | 1/7/2/1（custom） | 9 | 2/6/2/2（custom） | 10 |
| 24 | 1/11/2/1（custom） | 13 | 2/10/2/2（custom） | 14 |
| 32 | 1/15/2/1（custom） | 17 | 2/14/2/2（custom） | 18 |

额外 P0/C2/R3/S1 示例有3个物理 block、7次执行，是配置控制，不属于 D8/16/24/32 正式深度系列。未把“保持 C 固定、增加 R”的另一种扩展混进本表。

## 参数与记录口径

沿用 `cdlno/experiment.py` 的 `config.json` 命名与 `parameters.total/trainable`、`architecture`、`hparams`、`resolved_arguments`、`environment` 字段，并将现有 `loop_run_manifest.json` 的预期 schedule、公共主干 hash、loader seed 等信息嵌入同一个实验文件。额外加 `record_kind=offline_planned_experiment`，防止将清单误当已训练的 config/checkpoint。

每个配置都以实际宽度、深度和 rank 构造一次 CPU FP32 模型，逐个 `named_parameters()` 计数。loop 分为 stem、prefix、shared_core、suffix_body、head、router；独立层模型分为 stem、independent_block_bodies、head。共享 core 只计一次，最终 head 只计一次。`parameters.inventory` 给出全部参数的真实键、形状、数量及是否可训练。`state_dict` 记录 key/shape 摘要和初始化值 hash，不保存权重。

RB 新增 `2*hidden*(2*C*R+1)` 参数，LB 新增 `2*hidden*R`，SR 为0。注册为可训练的参数按仓库口径计入，包括单来源 RB receiver 中数学上不影响输出的 query/norm；参数量不等于本次梯度活跃参数量。

全量统计与独立整数公式比较；主 profile 的六任务、两 preset、三模式、两 rank、三 seed 共216条另外与既有 LL9R 正式宽度参数统计核对。每个 task/profile/topology/rank/seed 的三个 mode 校验公共主干初始值 hash 和 loader generator seed 相同。router 的 zero-query/one-scale 初始化不消耗 RNG。

Transolver 的 Darcy/NS 原构造器在无参数位置网格中硬编码 `.cuda()`。清单工具仅以本目录中的临时继承类在 CPU 计算同一 reference distance；原构造器、学习参数、初始化与 state_dict 不变。另用原 `get_grid` AST 只移除两处 `.cuda()` 的独立 CPU 对照，核验网格数值、参数键和值完全一致。该 CPU 处理只用于统计，不是新的生产模型或训练可用性声明。

`compute` 给的是 B=1、任务实际 N 下**一次 forward 的解析矩阵 MAC/FLOPs**，不是计时、不是真实 epoch 成本，且不包括 softmax、RMSNorm/LayerNorm、GELU 等标量操作。router 收缩量单独列出。NS 一个 rollout 有10次模型调用，Plasticity 一个 batch 有20次；不能把单次 forward 成本当整段训练成本。M翻倍增加计算量，参数共享减少参数不等于 FLOPs 按同比例减少。

## 训练协议和比较边界

主 profile 保持当前 LinearNO 设置：Darcy unified position 关闭；NS ratio2/ref10；Pipe ratio1/batch4；Plasticity Time_Input=True/out4。`transolver_matched` 下 Darcy unified 开启、NS ratio1/ref8、Pipe ratio2/batch8，均按现有解析器原样记录。原生 Transolver 的参数则从其真实脚本 + parser 默认解析，不能把其他 family 的默认值代进去。

静态任务 LinearNO/loop 的 rL2 为逐样本 flatten、无 epsilon、batch mean；Transolver 原路径使用 `TestLoss(size_average=False)`，batch sum。Darcy保留0.1导数项及原差分/边界处理。NS保留10步训练真值回填/测试预测回填，Plasticity保留20次更新而 scheduler 每外层 batch 只 step 一次。`hparams.training.resolved_scheduler` 显式列出 scheduler epochs、steps_per_epoch、total_steps；此处所有训练预算为500 epochs，因此 Darcy fixed500 不冲突。只记录预算，未执行训练。

“同 profile”“相同 M”“相同执行深度”不代表两种架构参数量、计算量、loss reduction、所有 normalizer 行为都相同。`transolver_matched` 是当前仓库已有 profile 名称，不能据此宣称已经完成完全相同训练协议下的公平性能比较。三种 residual 比较是完整残差架构比较，不是只改一个缩放系数的单因素消融。

使用本地预声明 seed=0/1/2、final checkpoint，逐 seed 报告并聚合 mean/std，不以 test 选最好 seed/checkpoint。原始 Transolver launcher 没有 seed 参数：本清单的三个 seed 是**已测的初始化 seed 与拟议配对实验标识**，不代表已给旧入口新增 seed/DataLoader 控制。loop 的 generator seed 来自现有实现；本次不运行 loader，也不修改任何入口。

## 未执行项

全部训练、评估、真实数据加载、GPU、checkpoint 写入、收敛/精度和远端环境验收均 NOT RUN。深层配置的成功构造与参数计数不意味着实际 batch 能在特定 GPU 显存中运行。未启动旧回归套件，因为本次仅新增清单与压缩包；保持性通过已有文件逐项内容 hash 与新增路径白名单验证。

远端路径仍以用户提供的 `.../transolver/LinearNO-monitor` 为准，数据根为同项目下 `.../data/fno`。清单文件中无时间戳；ZIP 使用固定条目时间，确保归档本身不会给实验加运行时间标签。
