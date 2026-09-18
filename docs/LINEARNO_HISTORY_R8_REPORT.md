# R8：AirfRANS 与 ShapeNet-Car

## A. 范围/读取

按连续授权，在R6/R7独立PASS后接入两工业任务。读取原R8提示词、旧industrial parser/Run/run_cli/checkpoint、两个main/train/eval调用链、数据与force helper、ensemble与原worker。快照 `/home/hwz/CDLNO-artifacts/linearno-history-r8-before-0byje7ax`，提示词/环境/hash证据位于 `linearno_history_audit/r8/`。未执行数据下载、真实loader或GPU。

## B. 决定及来源

保留用户已确认的训练协议：Air标准化四通道MSE，volume + 1×surface；Car标准化all-point三速度MSE + .5×surface压力MSE。paper/official各自evaluation_spec仍由原profile定义，没有引入本地rL2训练解释。

工业研究入口从当前pure adapter复制必要parser/Run/run_cli分支，数据/指标/normalizer/preflight/evaluation帮助函数直接复用原函数对象。未从官方整树复制训练框架。pure入口只有显式研究/fair调度；默认和单独A0K0仍走原类/schema/helper。

公平ensemble每个member公共主干以seed+i隔离生成，避免前一成员A dropout改变后一成员初始权重；feature seed按既定任务命名空间记录。各member独立模型/状态/history。两数据generator固定train/test并完整保存恢复，Python随机采样继续沿用原规则和完整RNG存档。

## C. 改动及调用链

新增 `cdlno/linearno_history/{industrial,air_entry,car_entry}.py`；新增两个平行launcher、两worker及industrial测试。`cdlno/linearno/{air_entry,car_entry}.py`仅parser/run_cli入口调度；Air原train三处DataLoader新增受`hasattr(linearno_run,'loader_kwargs')`保护的generator kwargs，旧分支严格传空kwargs。`provenance.py`逐片段记录可逆改动，pure Air源码fingerprint额外应用同一投影。原benchmark parser/default/main/evaluation/model/data/metrics文件不变，Car train不变。

链：平行launcher → 原main/parser → pure adapter显式拦截 → task research adapter → 现有实际research class → 原生task loop/复用objective → 完整研究metadata与strict epoch pair。eval读取每个member/model spec后才构造，未知family/缺spec不fallback Transolver。Car显式保留fold0..8和surface velocity/drag修复函数；Air保留full/scarce/reynolds/aoa和test全覆盖sampling实现。

模型数学未改：Air input7+真实pos2 reference distances、M32绝对值、dead temperature不参与forward；Car input7 tuple/PyG、M=key_ratio*d_h实际32、tempreature_q/k原拼写和[.1,2]。每forward独立raw history，图和ensemble成员间不混合。

## D. 测试

CPU、Python3.13.9/torch2.13cu130/PyG2.3.1，全部 `CUDA_VISIBLE_DEVICES=''`，无依赖安装。通用前缀同R6/R7，工业测试另用`MPLBACKEND=Agg`。

- `python -B -m unittest linearno.test_history_industrial -v`：40真实preset构造/参数量；8个四模式native PyG train→checkpoint→resume→新进程eval；4组合×2member ensemble及两类中断位置；公共初值/data顺序、最终权重/完整resume hash、字段预测精确对比。Air N13–15采样11；Car N17/23/8205，真实GraphDataset与Python几何采样；都是标明synthetic的内存对象。
- `test_production_metadata_conflicts_before_tensor_load`：两工业入口的A/K/rank/depth/head/profile错配，禁止torch.load后仍须提前拒绝。
- `python -B -m unittest linearno.test_air_entry linearno.test_car_entry linearno.test_car_metrics linearno.test_air_objective_decision linearno.test_car_objective_decision -v`：旧路径、原objective、force修复/normalizer/metric逻辑回归。
- preflight两个方法通过3.075s：40构造、source hash、共享帮助函数identity、原Car objective AST和launcher previews。

旧industrial回归17/17通过，200.609s；详见old-industrial.txt。六Standard预检查4/4通过5.809s；新工业8/8单成员闭环已经PASS。双成员矩阵与额外负面检查结果在末尾补记。初轮draft为补充任务身份显式校验而主动终止，未声明验收通过；原输出保留。

## E. 保护证据

旧Standard/Air/Car的source_sha256与normalized_patch_sha256同R8前真实记录精确相同。整个pure/研究模型数学与Standard六exp、数据/指标/可视化实现原样保留。新增Air generator接线可逐字逆转回原train；未改变sampling/loss/optimizer/OneCycle实际步数或日志旧命名。manifest覆盖tracked/untracked/ignored；3个本轮py_compile产生的ignored pyc仅属开发产物，不是源/模型变更，不提交。

## F. 边界与自审

检查metadata先于模型/权重、每个ensemble成员独立重建、原M/温度、严格family、公平成员初始化、完整采样RNG恢复、保留force函数。真实VTK/完整force评估在synthetic worker处明确替换为边界fixture或force=False，不能声称真实力系数已验证。真实loader/mini-run/GPU/远端/完整训练/精度/SOTA均NOT RUN。研究源码hash严格；完成开发后须冻结版本才开展正式实验。

## G. 验收

**PASS**。固定源码完整4/4方法通过，654.323s；额外metadata负面1/1方法（两任务12错配）通过6.477s。8个单成员闭环、4组合双成员ensemble的成员内与成员间两类中断恢复均exact。旧工业17/17通过200.609s，Standard预检4/4通过5.809s。实际日志tests.txt与negatives.txt分别保存，不将补充方法计入前一进程。五项自审均无新缺陷：实际M/温度、MSE目标、metadata先读、成员独立、采样与模型RNG连续。依用户连续授权进入R9。
