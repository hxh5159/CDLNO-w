# R1 审计证据

R1 只固化研究合同，不实现 A/K。`innovation_schema.json` 是 JSON Schema draft 2020-12；`configuration_matrix.json` 是 4 个 A/K 组合 × L4–L8 的 20 项核心矩阵以及 8 个任务 × 4 组合的 wrapper/生产闭环矩阵；`negative_cases.json` 固化负面验收；`fair_initialization_protocol.json` 固化公共主干、独立 feature seed 和 DataLoader generator 协议；`source-freeze.json` 记录受保护生产文件与 HEAD hash。

`tests/linearno/test_history_schema.py` 只导入 `linearno_history.schema`，在子进程中确认不导入 torch，也不构造模型、factory 或任务入口。

R1 的研究 class path 仅作为 metadata 合同字段校验；`cdlno.linearno_history.core.LinearNOHistoryCore` 目前不可导入，不能作为占位模型使用。A/K 数学实现和生产接线留给后续授权阶段。
