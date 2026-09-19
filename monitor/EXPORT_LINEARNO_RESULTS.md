# 只导出纯 LinearNO 的相似度监测结果

在远端已激活的 Python 环境中执行（任意工作目录均可）：

```bash
python /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor/monitor/export_linearno_results.py
```

默认扫描脚本所属项目的 `output/`，不依赖 `RUN_ROOT`、`RUN_SPEC`、`CDLNO_REPO_ROOT`
或当前工作目录。只读取已有结果，不执行训练、评估、模型构造或 checkpoint 加载；
直接执行 `.py` 只需要 Python 标准库，不需要导入 Torch、NumPy 或项目模型。

每次创建独立的 `LinearNO-monitor/monitor_exports/pure_linearno_<UTC>_<id>/`：

- `pure_linearno_monitor.tar.gz`：全部原始快照 NPZ、CSV、metadata、PNG/PDF/SVG，
  监测配置/退出记录/错误记录，以及已定位训练目录的 config、训练曲线、结果和评估结果。
- `summary.txt`：可直接粘贴的首尾平均相似度矩阵、非对角统计、快照计数、监测状态及跳过原因。
- `manifest.json`：包含/跳过目录、身份判断依据、源路径、打包文件 size/SHA-256。

完整 `architecture.json` 可能带大型 normalizer 或训练状态，故只导出指定字段到
`architecture_summary.json`，保留模型构造、profile、objective、evaluation 和 provenance；
它不是可加载 checkpoint。模型权重、数据集、优化器/RNG 数组不打包。

## 纯模型筛选规则

只接收旧 `monitor.run` 的 `capture_every_validations` 格式。`history_run` 的
`diagnostics.jsonl`/`max_snapshots` 格式全部排除；即使这个 runner 当时监测的是 A0K0，
也不会将其混入本导出格式。启动命令中的 A/K 开启值、A1K0 专用 launcher，或保存元数据中的
`family=linearno_history`、innovation_spec、architecture_extension、研究类路径/特征标识
都会否决该目录。不会根据“目录名含 linearno”判断其为纯模型。

优先核对命令显式指向的 run 中的 `architecture.json`、`config.json` 和 fair-run manifest。
绝对路径失效时，仅尝试在当前 checkout 下恢复完全相同的 `output/…` 相对后缀，并记录此事；
不猜最近一次实验。没有 sidecar 时，只接受显式纯 LinearNO 模型键或已知纯 launcher 的命令证据，
并在摘要标注证据局限。命令和元数据矛盾、元数据损坏或模型身份未知均跳过。纯模型显式 A0K0
可包含。所有跳过项列入清单；不打包其数值、图片和训练文件。

退出码 0 表示导出/列举完成，不代表训练完成或模型精度达标。纯监测目录即使零快照也会保留
配置/错误诊断，并明确标记无数值结果；没有可确认的纯模型目录则退出 2，不创建导出目录。
建议在监测进程结束后打包；正在写入的末尾文件可能尚未完整，摘要会报告读取问题。

`validation_index` 表示 eval-mode forward 次数，不是 epoch。摘要的“最后快照”指最后保存的
快照，不自动等同 final checkpoint。均值沿被保留的 sample/head 计算；完整逐样本/逐 head
数值仍保存在 NPZ/CSV。单凭高相似度不作塌缩或精度结论。

## 可选参数

```bash
# 先只查看哪些会被收集；不创建文件。
python monitor/export_linearno_results.py --list-only

# 结果实际保存在其他位置，或只打包一个实验组。
python monitor/export_linearno_results.py --root /实际结果目录

# 所有数值仍保留，仅图片缩减为每次监测的首尾两张（含其 PDF/SVG）。
python monitor/export_linearno_results.py --plots endpoints

# 自定义导出位置。
python monitor/export_linearno_results.py --output-dir /实际导出目录
```

未启用 `monitor.run` 的普通训练不会有这些监测结果；本脚本不能事后恢复未采集的训练过程。
它也不会自行用已有权重重新跑评估。

无真实数据验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s monitor -p 'test_export_linearno_results.py' -v
```

测试用人工监测目录验证纯模型/A/K混合目录的筛选、元数据否决、损坏/未知来源、
样本/head均值、归档内容与哈希、只读源文件、不同工作目录、重复导出及零快照提示。
不代表已访问远端的真实结果。

2026-09-19 本地验证：7 项导出测试通过；已有 `test_monitor.py` 的 3 项 CPU
监测测试通过。新增功能前后的 12 个既存 monitor 文件 SHA-256 全部一致。
本次只新增导出脚本、本说明和独立测试，没有修改模型、训练/评估入口或原监测代码。
