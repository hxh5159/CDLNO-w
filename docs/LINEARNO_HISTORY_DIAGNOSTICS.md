# 可选研究诊断与有限性能检查

所有新增诊断位于独立 `monitor/history_*` 与 `tools/linearno_history_*`；模型、训练、指标不依赖这些文件。旧monitor文件未改。默认命令不安装诊断、不开hook、不保存大张量。用独立runner显式启用：

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
source ./path.sh
TASK=airfoil
GPU=0; SEED=17; L=8; A=1; K=1
PROFILE=paper_table8_on_release_model
RUN="$PWD/output/$TASK/linearno_history/${TASK}__${PROFILE}__L${L}__A${A}K${K}__seed${SEED}"
python -B monitor/history_run.py --output "${RUN}_monitor_train" \
  --every 100 --max-snapshots 8 --max-points 256 -- \
  bash "tran_evaluate/linearno_history/$TASK.sh" train \
  --gpu "$GPU" --seed "$SEED" --linearno-profile "$PROFILE" --n-layers "$L" \
  --linearno_latent_attnres "$A" --linearno_history_k_conditioning "$K" \
  --experiment-dir "$RUN"
python -B monitor/history_run.py --output "${RUN}_monitor_eval" \
  --every 1 --max-snapshots 8 --max-points 256 -- \
  bash "tran_evaluate/linearno_history/$TASK.sh" eval --gpu "$GPU" --experiment-dir "$RUN"
```

Darcy/Elasticity/Pipe/NS/Plasticity只改TASK；AirfRANS/Car同时将train的`--n-layers`改成`--linearno-layers`。resume保留RUN并给监测输出一个新的目录：

```bash
python -B monitor/history_run.py --output "${RUN}_monitor_resume_01" -- \
  bash "tran_evaluate/linearno_history/$TASK.sh" resume --gpu "$GPU" --experiment-dir "$RUN"
```

runner拒绝覆盖已有监测目录。相同输出命令再次执行须选择新目录。训练是否完成由原命令退出码判断；原命令成功但未捕获模型或bootstrap失败时runner返回2，详情写入run_summary.json或bootstrap_error.txt。不得同时安装旧pure monitor和新history monitor。

输出是diagnostics.jsonl、gradients.jsonl、call_XXXXXX.png（三幅Q/K/P层相似度图）、runtime_config.json和run_summary.json。采集间隔以整个进程内模型forward计数，不是epoch；NS每物理时间步也是一次forward。图上使用相同的确定性等间隔点集；Q/K因子只保存最多max_points点的detach FP32副本，并在该次采集结束释放。保存到JSON的只有矩阵与统计量，不保存完整Q/K/图。

字段：A的receiver/gamma/各real-source平均权重/null权重/source entropy/sample×source dropout率；K的各head eta、修正范数比、N均值中心化误差、M×S与N×M形状；原路由Q/K熵、归一和误差、Q/K余弦与低秩P余弦；各block raw的shape/mean/std/norm/finite；一次Tensor.backward结束后的累计梯度组范数。autograd.grad不是Tensor.backward，因此后者没有自动梯度记录。第一层没有A/K记录。

P诊断使用tr[(Qi^TQj)(Kj^TKi)]和低秩范数，绝不在监测路径构造N×N。它描述当前block自身QK^T，包含实际K-conditioning，**不代表含A支路的端到端Jacobian**。采样点数小于N时是子核估计；K保留原全点归一权重，不在采样集上重新softmax。Q/K的逐槽相似度还依赖槽的排列，P相似度对联合槽置换更稳健。

诊断有额外计算和同步：pure路径需重算确定性factor；研究core的observe会额外计算用于观察的QC_raw；绘图/写文件也有耗时。不得用开启诊断的延迟冒充模型性能。默认最多8次采集；需要观察较晚训练阶段须调整every与max-snapshots。只支持eager单线程forward；torch.compile、多线程并发、DDP/多worker进程联合聚合未验证。全局拦截只在独立子进程/显式context内生效，退出恢复，不向模型附加hook/closure，whole-object序列化仍可用。

本地无数据自检（CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
 python -B tools/linearno_history_diagnostics_check.py --output /tmp/history-diagnostics-check
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 python -B tools/linearno_history_cost_table.py --output /tmp/history-cost-table
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 python -B tools/linearno_history_performance.py --output /tmp/history-cpu-perf \
 --points 972 --batch 2 --warmup 5 --steps 20
```

性能命令固定Elasticity接口的合成输入、d32/h4/M16、L4–8、四组合、CPU FP32及AdamW/MSE计时。不得当成真实任务rL2、完整NS/Plasticity epoch、GPU显存或真实吞吐。模型矩阵MAC由实际Linear/Conv/bmm算子计数，FLOPs列为2×MAC；norm/softmax/GELU/标量融合等另列完整算子工作清单，未将2×MAC冒称包含所有标量操作的总FLOPs。参数量直接sum(parameter.numel)，计入所有norm/bias/temperature（包括Air dead temperature）。
