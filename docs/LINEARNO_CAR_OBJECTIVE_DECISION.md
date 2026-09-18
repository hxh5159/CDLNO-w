# ShapeNet-Car 默认训练目标：用户确认的官方 MSE

2026-09-18 用户明确选择“Car 默认也沿用官方 MSE”。不再等待论文 rL2 的本地补全方案。

默认 `paper_table8_on_release_model`、`official_release`、`transolver_matched` 都采用已发布且与 Transolver 相同的训练定义：标准化空间，**所有点**的三个速度通道 mean MSE + **0.5 × 表面压力** mean MSE。不是 AirfRANS 的四通道两个区域 MSE（Air 的表面系数为1）。体积之外的表面点速度也进入 Car 速度项，非表面压力不进入 Car 压力项。

实现：`cdlno.linearno.profiles.CAR_OBJECTIVE_CONTRACT=car_transolver_mse_v1`，`CarRun.objective`。每项来源在config中记为integration_contract；explicit CLI权重覆盖仍记录cli_explicit。结构、epoch、lr、split、预测/物理指标各轴不因本决定改变。Car修复后的drag输入保持surface velocity和真实sample路径。

论文的 surrounding-region velocity rL2 + 1×surface pressure rL2 描述保留在不可变L0 catalog与显式contract=None/car_l7历史配置中，未补全的计算空间/reduction仍拒绝作为可运行paper loss。当前默认配置名称表示论文Table8结构/训练预算叠加本次明确的官方目标决定，不能称论文训练目标逐字复现。评估仍同时区分physical-rL2与normalized-MSE；Transolver的AirfRANS指标勘误不能推广为Car所有指标也只有MSE。

公平比较需要另对齐训练预算、split、数据、归一化、评价采样/force input、seed与硬件。采用相同MSE只解决训练目标这一项，不证明全部实验条件相同或精度已复现。

测试：`tests/linearno/test_car_objective_decision.py` 给出手算包含surface速度的例子、梯度、三个profile往返、旧metadata语义和显式覆盖；默认profile进入真实Car parser/train→pair→新进程resume/eval合成闭环。正式模型参数仍为3,852,420。真实数据、完整训练和指标数值未执行。
