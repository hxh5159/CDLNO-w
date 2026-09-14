# CDLNO 八任务配置与启动清单

更新：2026-09-14，阶段10最终核对。八任务均已具备新模型接口、配置和训练/评估脚本；仅完成合成验证，未完成真实数据训练或实际抽样评价。原 Transolver 配置、模型和脚本保留。

## 配置与入口

统一默认 L8/F2/P6、CDPA entry、chunk0、ratio2；合法范围为0≤F<L，P仅派生。下表是计划确认的初始值，不代表最优配置或已验证性能。N为默认布局；两个工业任务支持单图可变N。

| 任务 | 单次接口 | stem | d/h/M | epochs/batch | 配置 | 训练脚本 | 评估脚本 |
|---|---|---:|---|---|---|---|---|
| Darcy | `[B,7225,2], fx1 → [B,7225,1]` | 65 | 128/8/64 | 500/4 | [darcy.json](../PDE-Solving-StandardBenchmark/configs/CDLNO/darcy.json) | [CDLNO_Darcy.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Darcy.sh) | [CDLNO_Darcy_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Darcy_Eval.sh) |
| Elasticity | `[B,972,2], fx=None → [B,972,1]` | 2 | 128/8/64 | 500/1 | [elasticity.json](../PDE-Solving-StandardBenchmark/configs/CDLNO/elasticity.json) | [CDLNO_Elas.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Elas.sh) | [CDLNO_Elas_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Elas_Eval.sh) |
| Airfoil | `[B,11271,2], fx=None → [B,11271,1]` | 2 | 128/4/64 | 500/4 | [airfoil.json](../PDE-Solving-StandardBenchmark/configs/CDLNO/airfoil.json) | [CDLNO_Airfoil.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Airfoil.sh) | [CDLNO_Airfoil_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Airfoil_Eval.sh) |
| Pipe | `[B,16641,2], fx=None → [B,16641,1]` | 2 | 128/4/32 | 500/8 | [pipe.json](../PDE-Solving-StandardBenchmark/configs/CDLNO/pipe.json) | [CDLNO_Pipe.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Pipe.sh) | [CDLNO_Pipe_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Pipe_Eval.sh) |
| Navier–Stokes | `[B,4096,2], fx10 → [B,4096,1]` | 74 | 256/8/64 | 500/2 | [ns.json](../PDE-Solving-StandardBenchmark/configs/CDLNO/ns.json) | [CDLNO_NS.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_NS.sh) | [CDLNO_NS_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_NS_Eval.sh) |
| Plasticity | `[B,3131,2], fx1, T[B,1] → [B,3131,4]` | 3 | 128/8/64 | 500/8 | [plasticity.json](../PDE-Solving-StandardBenchmark/configs/CDLNO/plasticity.json) | [CDLNO_Plasticity.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Plasticity.sh) | [CDLNO_Plasticity_Eval.sh](../PDE-Solving-StandardBenchmark/scripts/CDLNO_Plasticity_Eval.sh) |
| ShapeNet-Car | `(cfd_data, geom_data), x[N,7] → [N,4]` | 7 | 256/8/64 | 200/1 | [shapenet_car.json](../Car-Design-ShapeNetCar/configs/CDLNO/shapenet_car.json) | [CDLNO.sh](../Car-Design-ShapeNetCar/scripts/CDLNO.sh) | [CDLNO_Evaluation.sh](../Car-Design-ShapeNetCar/scripts/CDLNO_Evaluation.sh) |
| AirfRANS | `data.x[N,7], pos[N,2] → [N,4]` | 71 | 256/8/64 | 398/1 | [airfrans.json](../Airfoil-Design-AirfRANS/configs/CDLNO/airfrans.json) | [CDLNO.sh](../Airfoil-Design-AirfRANS/scripts/CDLNO.sh) | [CDLNO_Evaluation.sh](../Airfoil-Design-AirfRANS/scripts/CDLNO_Evaluation.sh) |

## 工作目录及路径语义

在表中对应子项目工作目录运行 `bash scripts/<脚本名>.sh`。根级共享包 `cdlno` 应已在当前环境可导入；本阶段未安装依赖，独立进程验证使用仓库根 PYTHONPATH，不能将其描述成远端 editable 安装验收。

六标准任务用 `--data_path`；CDLNO 评估需要 `--cdlno-run-dir` 指向已有训练目录。ShapeNet-Car 保留 `--data_dir/--save_dir/--fold_id`，评估用 `--run_dir`；一次 run 只对应一个 fold。

AirfRANS 两个入口的 `--my_path` **含义不同**：

```bash
cd Airfoil-Design-AirfRANS
# 训练参数指向 Dataset 本身：该路径下直接有 manifest.json。
bash scripts/CDLNO.sh --my_path /data/naca/Dataset
# 评价参数指向 Dataset 的父目录；run_dir 使用训练打印的完整新运行目录。
bash scripts/CDLNO_Evaluation.sh --my_path /data/naca --run_dir /path/to/printed/run
```

AirfRANS 可用 `--task full/scarce/reynolds/aoa`；训练和评价要匹配。`--nmodel` 控制原有重复训练次数，成员整模型分别保存到 `member_000/model` 等目录，模型列表保存为 run 根目录的 `CDLNO`。默认398 epochs来自原 params.yaml；新增CDLNO key逐字段继承当前Transolver，原key未动。JSON的initial_training是起点记录，运行时以YAML加显式CLI覆盖为准。新脚本明确写出398；如果用户以后修改YAML预算并用脚本启动，应在末尾明确传入相应预算覆盖脚本值。

所有新脚本将用户参数放最后；改变架构、训练预算、task/fold/nmodel时，评价应重复相应配置。评价先读sidecar再比较，不能覆盖后再校验；改变运行chunk不构成权重结构不兼容。新训练目录不复用，评价写到独立结果目录。AirfRANS新训练脚本默认 `--score 0`；用户明确设置 `--score 1` 时沿用原完整评价流程。本阶段未执行该流程。

## 已验证范围

六标准任务已完成各阶段合成接口/原损失/时间语义与state_dict检查。ShapeNet-Car和AirfRANS已完成真PyG单图接口、可变N、多图拒绝、原损失反向及整模型checkpoint检查；AirfRANS额外覆盖模型列表与保留原ptr的抽样单图接口。

这份清单不宣称真实轨迹、VTK/manifest/图构造/反复抽样/边界后处理已端到端验证，也不授权自动执行后续阶段。实际运行记录见 [STATUS](CDLNO_IMPLEMENTATION_STATUS.md) 及对应阶段报告。


## 最终位置、时间和保存合同

| 任务 | 位置/字段提升与条件 | 单次点域FFN | checkpoint |
|---|---|---|---|
| Darcy | 原索引grid到8×8 reference的64距离替换xy，拼fx1；无placeholder/时间投影 | dense ConvFFN；85×85 | 标准run/model.pt严格state_dict |
| Elasticity | 原xy2，fx=None的active placeholder；无time_fc | point FFN；972点 | 同上 |
| Airfoil | 原弯曲xy2和221×51索引顺序；placeholder；不重排 | dense ConvFFN | 同上 |
| Pipe | 原exp归一化后的xy2和129×129索引；placeholder | dense ConvFFN | 同上 |
| NS | 原64×64索引reference距离64+窗口fx10；无time_fc/placeholder；训练真值、测试预测回填，每次重新进入core | dense ConvFFN | 同上 |
| Plasticity | xy2+fx1，原sincos时间embedding与time_fc，只在Time_Input=True注册；T[B,1]，每batch20个时刻独立更新，无反馈 | dense ConvFFN；101×31 | 同上 |
| ShapeNet-Car | cfd.x7=xyz3/sdf1/normal3；active placeholder；不读geom/y；out velocity3/pressure1 | point FFN；单图可变N | run/model_200.pth（或指定epochs），原整模型协议 |
| AirfRANS | x7追加pos到ref8×8的64距离，domain x[-2,4]/y[-1.5,1.5]；placeholder；out vx/vy/p/nut | point FFN；单图可变N | run/member_000/model等整模型，run/CDLNO原模型列表 |

所有run均配architecture.json；标准使用StaticRun，Car使用CarRun，Air使用AirRun。三原cwd新进程的导入/保存/加载和chunk0→1输出一致已在最终119项套件中通过。环境和完整负向检查见 [最终报告](CDLNO_IMPLEMENTATION_REPORT.md)。工业整模型使用局部trusted weights_only=False，标准weights_only=True；mode/F/L/M及wrapper语义不兼容时拒绝，运行chunk变化允许。Air抽样后保留原单图ptr的合同已覆盖，不等于真实抽样/图构建工作流已执行。
