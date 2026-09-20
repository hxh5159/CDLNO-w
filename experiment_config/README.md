# Looped LinearNO 精简实验配置

这里只收录当前最新的 loop 化 LinearNO：family=`linearno_loop`，类为 `cdlno.linearno_loop.standard.LoopedStandardModel`。一个实验一个 `config.json`，统一 seed=0、profile=`paper_table8_on_release_model`，文件及目录不加时间戳。

## 保留范围

| 数据集 | 残差配置 | 总执行深度 | 秩 M | 拓扑系列 | 文件数 |
|---|---|---|---|---|---:|
| Airfoil | SR、RB、LB 三种 | 8/16/24/32 | 64/128 | P1…S1、P2…S2 | 48 |
| Darcy | SR、RB、LB 三种 | 8/16/24/32 | 64/128 | P1…S1、P2…S2 | 48 |
| Elasticity | SR、RB、LB 三种 | 8/16/24/32 | 64/128 | P1…S1、P2…S2 | 48 |
| Pipe | SR、RB、LB 三种 | 8/16/24/32 | 64/128 | P1…S1、P2…S2 | 48 |
| NS | 仅混合 LB | 8 | 32/64 | 两个正式 preset | 4 |
| Plasticity | 仅混合 LB | 8 | 64/128 | 两个正式 preset | 4 |
| 合计 | 只保留 seed=0 | | | | **200** |

混合残差明确指 `lb_attnres_1_over_r`：轮内 operator/MLP residual 乘1/R，轮间及出口使用点域 AttnRes 混合 anchor 和各轮实际 Delta。它不是旧 A/K 模型。

- SR：`sr_1_over_r`。
- RB：`rb_attnres`。
- LB（混合）：`lb_attnres_1_over_r`。

范围继续限定为 PDE-Solving-StandardBenchmark 六任务。剔除了原生 Transolver、纯 LinearNO、旧 A/K 模型、额外 profile、seed1/2 和 P0/C2/R3/S1 示例。配置数从1,476减到200。保留原秩和双倍秩用于最新 loop 模型的 rank 控制，不代表保留了 Transolver 模型。

## 查阅方式

先看 `summary.csv`，包含各配置参数总量/可训练参数/主干/router参数、物理深度/执行深度、M、hidden、heads、FFN ratio、batch size、lr和对应文件路径。`index.json` 提供同一索引。

```text
experiment_config/
  README.md
  summary.csv
  index.json
  source_manifest.json
  verification.json
  airfoil/
    D8/
      M64/
        p1_c3_r2_s1/
          sr_1_over_r/config.json
          rb_attnres/config.json
          lb_attnres_1_over_r/config.json
  darcy/...
  elasticity/...
  pipe/...
  ns/.../lb_attnres_1_over_r/config.json
  plasticity/.../lb_attnres_1_over_r/config.json
```

每个 config 保留实际构造参数、完整 loop_spec、任务训练协议、全部参数名/shape/numel、实测参数分项、初始化 hash 和预期调用顺序。这些是**参数信息与配置文件，不是训练完成的权重文件**，不能把本目录作为 checkpoint 运行目录用于 resume。

## 深度口径

D 表示实际执行 block 次数；同时记录独立物理 block 数。保持 R=2、P/S不变，增加 C：

| D | 拓扑A P/C/R/S | A独立block | 拓扑B P/C/R/S | B独立block |
|---:|---|---:|---|---:|
| 8 | 1/3/2/1 | 5 | 2/2/2/2 | 6 |
| 16 | 1/7/2/1 | 9 | 2/6/2/2 | 10 |
| 24 | 1/11/2/1 | 13 | 2/10/2/2 | 14 |
| 32 | 1/15/2/1 | 17 | 2/14/2/2 | 18 |

D8使用两个正式preset；D16/24/32是前次清单明确采用的custom深度扩展，不是额外的正式preset。未加入其他custom拓扑控制。

## 数据来源与验证边界

本次从刚完成的实测清单筛选。模型/schema/profile/统计相关源文件与此前CPU实际构造计数时逐项hash一致，因此沿用已测参数值；没有重新训练或声称重新测量性能。每条配置的构造参数、参数清单、初始化state摘要和完整resolved_config均与来源记录保持一致，schema再次校验通过。

只更新新清单自己的目录索引、来源说明和单seed展示策略。resolved_config中的原profile有历史性的三seed政策文字，为维持原schema可验证性而原样保留；本次实际选定的runtime.seed与所有文件均为0，不再宣称多seed均值/标准差。

`compute`仍是已有解析矩阵MAC，未计入所有标量运算，不是实测epoch耗时。M翻倍不代表compute-matched。训练、评估、数据加载、GPU和checkpoint均未执行；没有修改模型、launcher、旧清单或任何既有文件。
