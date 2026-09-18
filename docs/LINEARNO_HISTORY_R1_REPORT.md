# LinearNO history 扩展 R1 报告

日期：2026-09-18。阶段状态：**PASS**。

## A. 范围与判定

本阶段只固化 A/K 的数学、配置、metadata、checkpoint 规则、测试矩阵和公平初始化协议。没有实现 A/K 数学模块，没有创建占位模型，没有修改 factory/CLI/launcher/任务 wrapper，没有读取真实数据或运行 GPU/长训练。

`history_conditioned_k_v1` 的三项首版选择已逐项冻结：

- 使用当前基础 `to_k.weight` 的 M 行作为槽 query；
- 全网络共享、跨原始 head 共享且与 A 分离的无 bias `Uq_K/Uk_K/Uv_K`；
- 每个 receiver layer × head 一个 raw gate，使用 `tanh`，严格零初始化。

用户没有提出改变上述选择的决定，因此没有 BLOCKED 条件。

## B. 文件和变更摘要

- `linearno_history/schema.py`、`linearno_history/__init__.py`：独立、无 torch/model import 的 schema validator。它不构造模型，也不接旧 loader。
- `docs/linearno_history_audit/r1/innovation_schema.json`：draft 2020-12 机器 schema。
- `configuration_matrix.json`：A0K0/A1K0/A0K1/A1K1 × L4–L8 的 20 个核心项；八任务 × 四组合的 32 个 wrapper/生产闭环项。
- `negative_cases.json`：非法 bool/dropout、family、错配、非 strict、缺 spec、全 mask、跨 forward、禁用状态、越界深度等负面规格。
- `fair_initialization_protocol.json`：公共 backbone seed、独立 innovation seed、独立 DataLoader generator 和对照规则。
- `tests/linearno/fixtures/history_r1_schema.json`、`tests/linearno/test_history_schema.py`：不构造模型的 valid/legacy fixture 和 10 个 schema/matrix 测试。
- `docs/LINEARNO_HISTORY_IMPLEMENTATION_STATUS.md`、`docs/LINEARNO_HISTORY_RESEARCH_MATRIX.md`：追加 R1 状态和冻结矩阵；R0 内容保留。

受保护的 31 个纯 LinearNO、任务入口、factory 和工业入口文件与 HEAD 内容 hash 全部一致，详见 [source-freeze.json](linearno_history_audit/r1/source-freeze.json)。

## C. 规格到 schema/符号映射

| 规格 | R1 固化位置 |
|---|---|
| 两个 bool 是唯一真值、dropout null/0.1 解析、派生 A/K 签名 | `linearno_history.schema.resolve_feature_config` |
| 研究仅 A1K0/A0K1/A1K1 | `validate_innovation_spec`、JSON `feature_signature` enum |
| 基础 variant、L/H/d_h/M、温度语义 | `base_linearno`；`TEMPERATURE_SEMANTICS` |
| raw 为 pre-AttnRes `C_raw`、forward-local、不可 detach/跨样本时间 | `raw_cache` |
| A per-head Cross、d_m=d_h、history-only source softmax、zero null、gamma、dropout | `innovation_spec.attnres` |
| K row query、all-history token bank、point×slot dot、N-centering、per-head tanh zero gate | `innovation_spec.history_k` |
| metadata-first、strict、逐字段结构比较 | `validate_research_metadata`、`assert_structural_compatibility`、`validate_strict_load_policy` |
| legacy/A0K0 隔离、禁止猜测 research | `validate_legacy_metadata` |
| 运行目录与公平 seed | `run_directory_id`、`derive_fair_seeds` |

研究 class path 只作为未来真实实现的稳定 metadata 字段；当前 `cdlno.linearno_history.core.LinearNOHistoryCore` 不存在，validator 也不会 import 它。因此没有用占位模型使开关“成功”。

## D. 实际命令与结果

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 \
python -B -m unittest linearno.test_history_schema linearno.test_schema linearno.test_profiles -v
# 21 passed

python - <<'PY'
import json
from pathlib import Path
for p in Path('docs/linearno_history_audit/r1').glob('*.json'):
    json.load(open(p))
PY
# all R1 JSON files parsed successfully

git diff --check
# passed
```

测试子进程确认导入 `linearno_history.schema` 不导入 torch；测试没有调用任务 factory、model constructor、launcher 或数据 loader。没有运行真实数据、GPU、长训练或 A/K forward。

## E. 纯基线和旧模型兼容性

R1 不修改 `cdlno/linearno/schema.py`、旧 LinearNO 类、model registry、任务 parser、checkpoint helper 或 launcher。显式 A0K0 和省略字段都解析为 legacy `linearno`；旧 checkpoint 缺 `innovation_spec` 时不会被猜成 research。研究 metadata 要求 `family=linearno_history`、`architecture_extension=linearno_history_v1` 和完整 `innovation_spec`，且只允许 strict=True。

R0 的纯 LinearNO parity、旧 Transolver 回归和已知历史冻结断言结果保持原记录，没有在 R1 重写或放宽它们。

## F. 未运行项与限制

核心 20 项和 32 项任务矩阵全部标记 `NOT_STARTED`；尚未构造 A/K 模型、执行 forward/backward、optimizer、checkpoint round-trip、任务闭环、真实数据或 GPU。公平初始化协议目前是 schema/fixture，尚未在模型中执行。R1 没有改变 official/paper profile 的数据、objective 或 metric 语义。

## G. 自审重点

1. A0K0 不会注册 innovation 参数或写入旧 checkpoint metadata；
2. history K 的三项用户指定参数化已在 validator 中逐项强制检查；
3. schema 校验在未来构造和 strict load 前执行，A/K/L/variant/M/head_dim/class path 不匹配会先失败；
4. 新包位于 `linearno_history/`，没有扩大 Standard resume 的 `cdlno/linearno/*.py` 源码哈希范围；
5. 未把任何 schema 通过结果描述成 A/K 数学实现或任务支持。

**本 R1 阶段结束，未执行 R2 或后续阶段。**
