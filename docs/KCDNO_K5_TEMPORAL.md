# K5 NS / Plasticity（2026-09-15）

## A–C. 完成范围与映射

扩展现有 `cdlno/kcdno/standard.py` 至六标准任务；`TEMPORAL_TASKS` 和 `_grid` 复用旧适配器合同，没有复制模型数学。NS stem74（reference64替换xy+fx10），64×64，单步输出1，无time_fc/placeholder；Plasticity stem3（xy+fx1），101×31，输出4，只有该任务注册原SiLU time_fc，sin/cos embedding和注入位置保持。每次调用完整K3 core，无跨真实时间缓存。

`exp_ns.py`、`exp_plas.py` 仅新增模型构造/记录有限分支，沿用原H/W变量并由wrapper校验固定网格。parse/profile/strict state_dict和先读sidecar重建复用K4。新增两JSON和 `tran_evaluate/kcdno/ns.sh` / `plasticity.sh`。NS默认L8/d256/h8/M64/r16、500epochs/batch2/原无clip；Plasticity L8/d128/h8/M64/r16、500epochs/batch8/clip.1。原优化器和scheduler定义/执行语句保留。

## D. 命令与结果

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_temporal test_kcdno_tasks test_temporal_standard.StaticTemporalFreeze -v
PYTHONPATH=. python -B docs/kcdno_audit/replay_subset.py temporal --result docs/kcdno_audit/k5/old-replay.json
bash tran_evaluate/kcdno/ns.sh train --gpu 0 --kcdno-run-dir /absolute/new/run
bash tran_evaluate/kcdno/ns.sh eval --gpu 0 --kcdno-run-dir /absolute/existing/run
# Plasticity 同样替换脚本名；可追加 --history-mode off 或 --profile transolver_shape_match。
```

12/12通过10.928s，含4个新时间测试、6个K4静态回归和2个原时间冻结/脚本检查。新时间任务实际B2/N4096或3131/d8/M3/L2合成输入：

- 直接安全提取原NS完整训练batch语句（仅将.cuda设备调用去掉），10步原TestLoss累加、真值回填、一次backward/optimizer.step/scheduler.step；eval原十步循环预测回填。实际窗口逐步核对，10份cache对象互不混用。
- 直接提取原Plasticity训练batch语句：yy[B,3131,4,20]，每时间T[B,1]，20次独立前向/backward/optimizer step，随后scheduler一次；fx不回填。不同T影响输出，T和time_fc梯度有限非零。最初测试误期待整batch scheduler零次，实际源码为一次；修正测试断言，未改生产循环。初始失败日志保留。
- 两任务×all/off配置/输出/strict权重往返4例，eval省略架构字段按saved重建，sidecar字节不变；完整两exp投影与K5前快照、原baseline AST均一致。
- K0旧NS/Plasticity各Transolver+CDLNO三模式共8份原权重输出/诊断梯度回放完全相同。环境仍本机Python3.13.9/Torch2.13+cu130；旧Transolver夹具用GPU，新时间模型本阶段为CPU检查。

证据 `docs/kcdno_audit/k5/{final-tests.txt,old-replay.json,before.json,changes.patch}`。未启动实际训练，上面train/eval命令只作为用户可运行示例。

## E–F. 自审与边界

已核对输入时间轴、缓存寿命、time_fc只按需注册、输出通道与20时间标签轴、原loss/optimizer/scheduler调用数以及新旧加载隔离，无待裁定结构差异。数据、训练和评价循环未改；不改10→10，不增加跨时间潜空间推进，不扁平化时空。NS/Plasticity真实轨迹、原数据完整读取、远端2.11/cu128、新时间任务GPU/性能未执行。K5自审完成，按本轮连续授权进入K6，尚未将工业任务标为本阶段完成。
