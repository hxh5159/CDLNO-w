# R4 evidence

完整A–G报告：[R4 report](../../LINEARNO_HISTORY_R4_REPORT.md)。

- `baseline.json`：R4开始时root/HEAD/tree/1640文件；外部artifact含原source、完整tracked/untracked/ignored manifest、status/diff。
- `environment.json`：实际依赖版本与CPU/无数据范围。
- `history-k-initial.txt`：首轮12方法中1个采集器形状断言失败；原因是只记录bmm而漏掉mm。
- `history-k-final.txt`：修正采集器、补充因果query/M检查后，13/13通过，未改变模型公式或容差。
- `history-k-results.json`、`numerical-summary.json`：独立oracle全部误差、VJP、形状、反例、18模型案例和三个fresh-cwd加载。
- `regression.txt`、`regression-results.json`：90方法85通过/4跳过/1历史失败方法（3子断言），无新增回归。
- `freeze.json`、`research-source.diff`：相对R4开始快照的文件hash、允许差异、原点域更新AST与新增源码。

复跑（仓库根）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_history_k -v
```

可设置 `LINEARNO_R4_REPORT=/tmp/linearno-r4-results.json` 另存数值证据。原回归命令在报告中。未加载外部pickle；所有checkpoint均由测试生成state_dict并用weights_only=True/strict=True恢复。无真实数据/GPU/benchmark训练。
