# K6 ShapeNet-Car（2026-09-16）

新 `models.KCDNO.Model` 位于 Car 稳定模块，独立 KCDNO core，复用原单图校验；x7→stem→L个block→LN/head→[velocity3,pressure1]，不读取 y/geom，不改变原 tuple、surf、点序或图处理。默认 L8/d256/h8/M64/r16、PointFFN，200epochs/batch1/Adam lr.001/reg.5/fold0；原 train.py 与评价计算保持。

新增 `cdlno/kcdno/industrial_entry.py` 的 CarRun、Car 配置/选择/有限入口分支、`tran_evaluate/kcdno/car.sh`。保留原整模型 `model_<nb_epochs>.pth`，局部可信 weights_only=False，先读 architecture/task sidecar，再严格检查类、core/block配置与state_dict；eval恢复省略的结构和fold/epoch等运行合同，显式冲突拒绝。没有新增resume。脚本读取checkpoint合同后运行原Car拖曳评价前置检查，原固定param0路径与fold0限制仍存在。

实际命令：

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_car test_shapenet_car.FrozenSourceAndScripts -v
PYTHONPATH=. python -B docs/kcdno_audit/replay_subset.py car --result docs/kcdno_audit/k6/old-replay.json
bash tran_evaluate/kcdno/car.sh train --kcdno-run-dir /absolute/new/run --dry-run
bash tran_evaluate/kcdno/car.sh eval --kcdno-run-dir /absolute/existing/run
```

5/5检查通过5.557s：真实PyG Data/Batch N11/19、all/off原train.train/train.test函数及mask损失、Adam/OneCycle一步、反传有限、无y泄漏/无原地修改、多图拒绝；整模型严格往返、架构冲突、sidecar字节不变，以及Car原cwd新进程加载输出相同。全main/eval AST移除新有限分支后与K6前一致，34个旧工业冻结文件字节相同。K0旧Car Transolver和CDLNO三模式4份同权重输出/梯度完全回放，atol=rtol=0。证据 `docs/kcdno_audit/k6/`，含最终日志、旧回放和阶段diff。

自审已核对单图和通道、原mask损失、原200epoch文件名、先验后加载/稳定类路径、原工作目录与新旧隔离。没有待裁定结构差异。新Car为CPU合成/PyG验证，旧Transolver基准使用本机GPU；不代表新Car GPU全宽/远端2.11cu128/真实fold训练或拖曳评价已通过。继续已授权K7。
