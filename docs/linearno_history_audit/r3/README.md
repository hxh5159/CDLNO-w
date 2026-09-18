# R3 evidence

完整说明：[R3 report](../../LINEARNO_HISTORY_R3_REPORT.md)。

- `baseline.json`：实际root/HEAD/tree及开始时1625文件计数；外部artifact保存完整分类hash、status/diff及原始source。
- `environment.json`：实际Python/torch/PyG等版本，CPU执行范围。
- `attnres-initial.txt`、`attnres-final.txt`：第一次9项和最终13项均通过。
- `attnres-results.json`、`numerical-summary.json`：float64/float32全部前向/梯度/step误差、隔离分支VJP、零门模型parity、三工作目录fresh-process加载。
- `regression.txt`、`regression-results.json`：R2+旧回归77方法，72通过/4跳过/1已知失败方法（3子断言），没有新失败。
- `freeze.json`、`research-source.diff`：相对R3开始snapshot的完整分类比较和本轮研究源码增量。

仅CPU合成，无数据、GPU、训练launcher或外部pickle加载。测试自己生成的state-dict全部以`weights_only=True`/`strict=True`恢复。

复跑新测试（从仓库根）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_latent_attnres -v
```

完整旧回归的实际模块列表/命令见报告；不删除或豁免R0已知失败来制造全绿结果。
