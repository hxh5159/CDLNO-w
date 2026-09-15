# K4 静态任务接入（2026-09-15）

本轮用户明确授权按提示词连续完成 K4–K10，每阶段自主自审后继续；此决定优先于旧提示词的阶段停止模板。K0–K3 已完成，不重复实施。K4 自审完成，进入 K5。

## A–C. 范围、代码与合同

四任务 `exp_* → cdlno_entry.parse_args → cdlno.kcdno.entry.resolve_args → model_dict.get_model(kcdno) → model.KCDNO.Model → cdlno.kcdno.standard.StandardModel → KCDNO core → output_norm/Linear` 已接通。原 CDLNO/Transolver 分支不接新 kwargs，K2/K3 与旧公共原语未改。

新增 `cdlno/kcdno/standard.py` 只实现任务输入提升和输出头；grid 复用原 `_grid`。Darcy ref64 替换 xy，再拼 fx1，stem65；其余保留物理坐标+fx=None placeholder。固定 fx 任务不注册 placeholder；四静态任务没有 time_fc。Darcy/Airfoil/Pipe dense ConvFFN；Elasticity point FFN。原曲线网格索引不重排。

新增 `cdlno/kcdno/entry.py`、标准子项目 `kcdno_entry.py`、`model/KCDNO.py`、四 JSON 和 `tran_evaluate/kcdno/{darcy,elasticity,airfoil,pipe}.sh`。主 profile L8/r16、d/h/M=128/8/64、128/8/64、128/4/64、128/4/32；epochs500、batch4/1/4/8，复用当前原 lr/wd/clip/调度。用户显式 CLI > profile > 家族默认，薄脚本不将 profile 尺寸硬写成显式参数。

`architecture.json` 保持 K1 完整 schema，新增 `task.json` 绑定 wrapper、网格、输入语义、协议和实际参数。eval 先读二者，从 saved resolved core 重建，显式冲突拒绝；恢复 ref/unified_pos/downsample/ntrain（影响 normalizer 的训练样本数），旧文件不重写。仍保存 strict `model.pt` state_dict，没有新增 resume 或跨家族转换。

默认目录 `output/<task>/kcdno/<profile>/<timestamp>_L…_d…_h…_M…_r…_<history>_<hash>`；独立路径和排他创建避免覆盖。`cdlno/experiment.py` 仅增加可选新 family/run-key，使现有启动 config/参数量/训练结果/独立评价记录工具能服务新家族，旧默认记录不变。

## D. 实际命令与结果

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_tasks -v
PYTHONPATH=tests:. python -B -m unittest test_static_standard.StaticFrozenChecks.test_legacy_ast_recovered_exactly_from_limited_branches -v
bash tran_evaluate/kcdno/darcy.sh train --dry-run
bash tran_evaluate/kcdno/darcy.sh train --profile transolver_shape_match --history-mode off --dry-run
# 真训练留待用户执行；以下是实际存在的入口，不在本轮运行：
bash tran_evaluate/kcdno/darcy.sh train --gpu 0 --kcdno-run-dir /absolute/new/run
bash tran_evaluate/kcdno/darcy.sh eval --gpu 0 --kcdno-run-dir /absolute/existing/run
# elasticity/airfoil/pipe 同样替换脚本名。
```

新6项通过9.886s：四任务×all/off共8个真实factory构造、合成相对L2/backward/AdamW step/eval/strict保存加载；省略结构参数重建、错误history/r/h/ref拒绝、损坏权重拒绝、sidecar字节不变。原Darcy decode+相对L2+零边界梯度项完整AST和另外三任务原loss/decode语句执行；Pipe原坐标normalizer连接通过。真实N=7225/972/11271/16641的小d8/M3/L2训练布局与5×7数值检查分开执行，没有改变正式配置。标准原cwd新进程加载通过。shell预览实际参数重新送入真实parser通过。

K0四任务旧Transolver+CDLNO三模式共16份原权重/输入/输出/梯度回放通过；未重新随机初始化互比。最初回放脚本从错误cwd启动，报找不到model；修正为原标准子项目cwd后16份通过，原失败日志保留。环境沿用Python3.13.9/Torch2.13+cu130，旧Transolver夹具在本机GPU；新K4任务检查为CPU，不声称新任务GPU或远端2.11/cu128通过。

证据：`docs/kcdno_audit/k4/final-tests.txt`、`old-replay.json`、`old-freeze.txt`、`before.json`、`changes.patch`。本阶段旧冻结AST比较只增加删除明确新家族分支的测试投影，随后仍与原commit完整AST比对；另有新测试直接把四exp投影回修改前K3快照，未放宽loss/loop检查。

## E–F. 冻结、自审和限制

四exp仅新模型构造/记录分支，数据读取、normalizer、loss、optimizer/scheduler、原循环、指标/可视化原文保留。自审输入拼接、placeholder/初始化、profile优先级、sidecar先读后验/不覆盖以及完整AST/同权重回归均通过。无待裁定架构差异。自审补上了eval恢复ntrain，防止重算normalizer时样本数不同。

真实数据完整读取、实际训练/评价、收敛和准确率均未执行。旧Darcy非默认downsample的绘图硬编码、工业固定路径和未接任务resume等旧限制不在K4顺手修复。NS/Plasticity、Car、AirfRANS尚未在本阶段接入，按已授权顺序继续。
