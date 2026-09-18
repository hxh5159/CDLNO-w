# AirfRANS 训练目标：用户确认的官方 MSE 协议

2026-09-18。用户明确选择沿用 Transolver/LinearNO 官方 AirfRANS 训练目标：**标准化后的四通道 volume MSE + 1 × surface MSE**。这一决定覆盖早期计划中 AirfRANS 默认 paper rL2 的要求；不是把 MSE 宣称为论文 Table 8 的文字公式。

## 决定与来源

| 项目 | 证据与处理 |
|---|---|
| 论文 Table 8 | `L_v + 0.5 L_s`，表注称两项为区域物理场的 rL2。未充分规定训练通道、标准化空间和通道聚合。保留这一事实。 |
| 此前助手建议 | “反归一化后四通道联合 rL2”是助手对未说明部分的补充解释；**不是两份官方代码的训练目标，也不是作者公开的完整实现**。用户没有采纳；不得作为默认或隐藏 fallback。 |
| LinearNO 官方固定源码 | `HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`，`AirfRANS/main.py:14,103` 和 `train.py:44`：`MSE_weighted`、默认 `reg=1`；四通道逐区域求均值。 |
| Transolver 官方固定源码 | `thuml/Transolver@75e0f67643806a81cd1d3f6adc88dd8c02416fe7` 的同任务入口/训练代码执行相同目标。AirfRANS README 明确勘误 field metric 为 MSE，非 rL2。 |
| 当前 Transolver | `Airfoil-Design-AirfRANS/main.py:15,142`、`train.py:51`；上述默认路径不变。 |
| 用户决定 | 新 LinearNO 的 AirfRANS 默认训练直接沿用已明确的 normalized-MSE。无需再决定假设中的 rL2 空间/通道。 |

论文：[Table 8](https://arxiv.org/html/2511.06294v3#Sx9.T8)、[Table 2](https://arxiv.org/html/2511.06294v3#Sx3.T2)。Transolver [官方勘误](https://github.com/thuml/Transolver/blob/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/Airfoil-Design-AirfRANS/README.md)。不根据勘误推断 LinearNO Table 2 原始数字究竟采用了什么未公开协议。

## 精确训练合同

令 `z_hat,z ∈ R[N,4]` 为模型输出与 **已标准化** 标签，通道顺序 `[vx, vy, p, nut]`。`s=surf` 是现有布尔 mask。

```text
volume_per_channel = ((z_hat[~s] - z[~s]) ** 2).mean(dim=0)
surface_per_channel = ((z_hat[ s] - z[ s]) ** 2).mean(dim=0)
L_volume  = volume_per_channel.mean()
L_surface = surface_per_channel.mean()
L_train   = L_volume + 1.0 * L_surface
```

区域先独立按点数归一，再在四通道上平均；**不是**所有点一起取均值后再给表面加权。训练不反归一化，不作相对范数除法。正式 batch=1，体积/表面均应存在；本决定不改变 loader、mask、采样或标签。

## 配置及兼容实现

`cdlno.linearno.profiles.resolve_config('airfrans', profile)` 默认应用版本化合同 `airfrans_transolver_mse_v1`，覆盖 `objective` 各字段；其来源记录为 `integration_contract`。三个既有 profile 的模型、训练预算、评价和数据轴保留原定义；显式字段覆盖仍优先，变更会进入 config hash。

由于默认 profile 的历史名字仍是 `paper_table8_on_release_model`，必须同时保存/展示 `integration_contract` 与完整 `objective_spec`，不能只用 profile 名宣称逐字论文训练复现。`official_release` 的目标本来就是这一 MSE；不再生成两套名称不同但未经说明的 loss。

显式 `contract=None` **仅保留早期来源审计/旧配置的解释**：paper 训练仍有 UNRESOLVED 并被 `require_resolved_objective` 拒绝；不是生产可运行的 rL2 模式。`validate_resolved` 按存档中有没有合同来校验，不把旧 metadata 自动迁移为 MSE。原 `_profile_data.py`、L0 清单和历史证据不改写。

`tests/linearno/test_air_objective_decision.py` 覆盖三个默认配置、字段来源/优先级、其他七任务不变、旧配置往返及拒绝伪造、真实 Air 类构造器的 metadata 往返与 resume 目标冲突。L1 来源清单测试显式传 `contract=None`，继续验证全部24条 L0 事实；不把新决定反写为旧证据。

## 公平比较和阶段边界

这一决定对齐了 LinearNO 与当前 Transolver 的 **训练目标**，并不自动证明整套实验条件一致。当前 Transolver 为398 epochs，paper/release LinearNO为400；`transolver_matched`保留398。评价也仍是独立轴：paper标签为pressure rL2，release为normalized MSE；结果必须同指标、同数据/采样/聚合/预算比较，不能把 MSE 改名 rL2。

本次补充只落实用户的训练目标决定及独立配置/校验；没有生产训练、数据读取或依赖变更。AirfRANS C08 的训练定义已解决，但 L6 生产接线、完整 native checkpoint/resume/eval 仍须完成，不能凭本次配置检查标为 PASS。Car 是独立冲突，本次 AirfRANS 决定不自动套用到 Car。L8–L10仍未执行。

检查命令、结果与全文件冻结核验见 [air-objective-decision](linearno_audit/air-objective-decision/)。
