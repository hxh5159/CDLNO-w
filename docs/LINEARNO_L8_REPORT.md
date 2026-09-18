# LinearNO L8 报告

## 状态

**PASS（资源门控项 NOT RUN）。** 本阶段完成了显式外部 checkpoint 转换边界、八任务
profile/目标汇总和无数据 checkpoint 复核，没有下载未知文件、读取真实数据或启动长训练。

## 转换协议

`cdlno.linearno.converter` 对 Standard 只接受裸 `state_dict`，并要求调用者显式提供
`source_task="standard"`。AirfRANS 与 ShapeNet whole-object 文件可能携带相同的官方限定类名，
因此必须显式给出 `source_task` 和 `trusted=True`；转换器在固定 `/home/hwz/LinearNO` 的对应
任务子目录启动隔离 Python 子进程，只导出 state_dict 与类名/任务 provenance，父进程不导入
两个任务的模块。没有用户明确提供且可信的外部 pickle 时，本阶段只用子进程临时构造的官方
等价对象测试格式，不把它称为外部 checkpoint parity。

归一化流程可逆处理 `module.` 前缀；所有键集合、重复键、shape、任务和目标模型类别都会
先检查，之后由现有入口 `strict=True` 加载。AirfRANS 的未使用 `temperature` 参数和
ShapeNet 官方拼写 `tempreature_q/tempreature_k` 均作为正式 state key 保留。未知/缺失键
不会以 `strict=False` 或随机补权重掩盖。

## 八任务 profile 与命令

三组 profile 的机器字段继续由 `profiles.resolve_config` 和
`linearno_audit/l0/profile-inventory.json` 提供；每个运行目录的 `architecture.json`、
epoch metadata 都保存 family、profile、model constructor、objective、evaluation、data
checksums、normalizer、provenance 和 resume state。默认实验应预声明本地 seed（例如
`0 1 2`）；官方发布脚本没有作者 seed，不能写成官方 seed。

Standard 六题使用 `tran_evaluate/linearno/{airfoil,darcy,elasticity,pipe,ns,plasticity}_{train,eval}.sh`；
AirfRANS 使用 `airfrans_train.sh`/`airfrans_eval.sh`，Car 使用 `car_train.sh`/`car_eval.sh`。
每条 eval 都显式带同一 `--experiment-dir`，不会猜最近运行，也不会覆盖旧 Transolver
目录。完整参数和数据路径模板保留在 `tran_evaluate/linearno/README.md`。

Car 默认训练是用户确认的官方 MSE：三速度通道所有点 normalized-MSE，加表面压力
normalized-MSE 的 `0.5` 倍；论文 physical rL2、drag、Spearman 是独立评价轴。AirfRANS
默认训练是官方 normalized 四通道 volume MSE + 1×surface MSE；论文 rL2 仅作为独立评价
描述。其余六任务的历史 hxh 数据、loss、时间循环、normalizer 和 scheduler 协议保持各自
阶段报告中的实际 resolved 字段。

## 实际检查与边界

- `tests/linearno/test_converter.py`：2 passed；覆盖裸权重、`module.` 前缀、任务冲突、
  本地可信 Air/Car 对象、dead temperature、ShapeNet 拼写键和 strict load。
- L6 AirfRANS 原生两成员 ensemble、连续/中断 resume、新进程 eval 证据继续有效。
- L7 Car 原生 PyG 合成 train/checkpoint/resume/eval 证据继续有效；默认 MSE 决策已写入
  `docs/LINEARNO_CAR_OBJECTIVE_DECISION.md`。
- 六 Standard 的原生合成闭环和旧 Transolver 回归引用 L4/L5/L6/L7 证据，不在本阶段重复
  造一份大权重文件。
- 真实数据、外部用户 checkpoint、完整 epoch、收敛/精度、实际耗时、远端
  Python3.10/Torch2.11/CUDA12.8 均 **NOT RUN**。

**本 L8 阶段结束，未执行下一阶段。**
