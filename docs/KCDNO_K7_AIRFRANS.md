# K7 AirfRANS（2026-09-16）

八任务KCDNO接口现在均已接入。新稳定类 `cdlno.kcdno.airfrans.AirfRANSModel` 复用旧wrapper的输入校验、reference/placeholder和单图规则，只用新core及最终LN/head；x7追加64距离→stem71，reference域[-2,4]×[-1.5,1.5]，输出[vx,vy,p,nut]。新params.yaml kcdno项复制当前Transolver训练字段398epochs/batch1/lr.001，所有旧key保留。

新增 `air_entry.py`、本地模型/entry再导出、`tran_evaluate/kcdno/airfrans.sh`；main/main_evaluation只增加构造、记录和可信加载有限分支。原每epoch抽样、图构造、val重复抽样、scatter/平均、mask/边界/指标完全保留。训练my_path为Dataset本身，评价为其父目录；脚本明确区分。原train保存member_000/model整对象，main保存模型列表至新家族run/kcdno。eval先读取结构和完整run合同，恢复省略的任务、模型数、epoch等；显式冲突拒绝，旧sidecar不改写。

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_airfrans test_airfrans.AirFrozenChecks -v
PYTHONPATH=. python -B docs/kcdno_audit/replay_subset.py airfrans --result docs/kcdno_audit/k7/old-replay.json
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/kcdno/airfrans.sh train --kcdno-run-dir /absolute/new/run
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/kcdno/airfrans.sh eval --kcdno-run-dir /absolute/existing/run
```

5/5通过6.302s，真实PyG Data/Batch、N11/19及原sampled ptr合同，all/off实际原train.train/test MSE_weighted及Adam/OneCycle训练步通过；无y泄漏/原地修改、多图拒绝、两member整对象和列表严格往返、显式错误拒绝、sidecar不变、原Air cwd新进程输出一致。2个新测试加3个旧冻结检查；21旧文件字节相同，完整入口AST对K7前快照相同。第一次命令误写旧测试类名而失败（两个新测试均通过），已改为实际AirFrozenChecks后通过，失败日志保留。K0旧Air Transolver与CDLNO三模式4份同权重输出/梯度精确回放通过。

自审通过：reference追加/域、stable pickle路径与列表成员检查、train/eval选择及路径、新旧YAML与冻结AST。未实际执行完整scatter物理评价、radius_graph或真实数据读取/训练。新任务为CPU/PyG合成；本机GPU旧基准与K2/K3 GPU证据单列，不宣称新Air全宽训练已测。继续已授权K8。
