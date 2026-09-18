# R7：NS 与 Plasticity

## A. 范围/读取

用户连续授权中的第二阶段。读取并保存原R7提示词；检查两个exp、collate、旧L5真实合成worker、StandardRun与R6研究Run、raw context和A dropout。快照见 `linearno_history_audit/r7/baseline.json`。六个exp继续字节不变，不改四静态任务协议或工业实现。

## B. 决定/依据

R7要求每个物理时间调用清空网络层历史并保持原节奏。模型本来就在forward中创建局部RawHistoryContext，不需要改模型数学。当前真实Plasticity代码是per-sample `torch.randperm(20)`（不是旧聊天中的NumPy）；保留原collate函数本体。为兑现R1独立数据RNG，在显式fair run里通过GeneratorCollate临时切换到已保存DataLoader generator执行原函数，再恢复模型CPU RNG。默认num_workers=0；多worker明确拒绝其完整恢复声明。原命令不使用此wrapper。

## C. 改动与公式映射

`standard_entry.TASKS`增加ns/plasticity；`fair_run.GeneratorCollate`隔离时间排列与A随机mask。新增两个平行launcher、history_temporal_worker/test_history_temporal、命令/证据/报告。R6测试的“尚未接入时间任务”断言改为拒绝未知任务，没有删除已有成功断言。

NS保持plain/d256/L8/h8/M32/ratio2/unified ref10、输入10输出1；10次teacher forcing调用→1 backward/optimizer/scheduler，测试10次prediction回填。Plasticity保持conv/d128/h8/M64/ratio1/H101/W31/time embedding/out4、原meshgrid/reshape顺序、20次forward/backward/optimizer→1 scheduler。raw历史长度每次调用0,1,...,L-1，append只存pre-A raw，无真实时间缓存。

## D. 测试

CPU、无真实数据/GPU。环境与R6相同。命令前缀 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

`python -B -m unittest linearno.test_history_temporal -v`：40真实preset计数、两任务×四组合的真实空间/时间尺寸合成原生闭环，连续3epochs vs1+恢复2、新进程eval。直接采集每次A mask与每层历史长度，对比连续/恢复的mask/batch/time_query/最终权重/optimizer/scheduler/RNG，不用总loss证明历史隔离。四组合公共初始hash和data/query序列比较。mask捕获仅测试观察，不改变生产forward。

`python -B -m unittest linearno.test_temporal_integration -v`：原5项时间协议回归；`python -B -m unittest linearno.test_history_static -v`：R6四题矩阵再次回归。旧时间任务5/5方法已通过，63.932s。新矩阵与静态回归结果待下条验收记录。

## E. 保护

六exp字节不变，A/K/core/context/wrapper模型不变；旧parser默认、基线schema/loader和非fair命令不变。复算所有tracked/untracked/ignored manifest，不只看git diff。

## F. 边界/自审

核对NS窗口、Plasticity原T形状/点序、20/1步数、dropout/data RNG恢复和时间隔离。真实loader/mini-run/完整实验、GPU/AMP、远端环境、收敛精度均NOT RUN。未调整objective或实验profile。

## G. 验收

新R7测试3/3通过，282.733s；40真实preset计数和8原生闭环PASS，所有权重/optimizer/scheduler/RNG/mask/batch/时间查询精确一致。纯NS L8参数3,377,921，Plasticity1,799,428。R6四静态回归4/4通过，346.428s，16闭环再次通过。唯一状态 **PASS**。完成独立自审后按用户授权进入R8。
