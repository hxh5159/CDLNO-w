# R6 导入边界返修（由 R9 综合回归发现）

R9 的 `test_kcdno_delivery.DeliveryChecks.test_legacy_parser_imports_without_shared_package` 发现 Standard parser 在旧 Transolver 分支也提前导入 `cdlno.linearno_history.standard_entry`。隔离 Python `-I` 时共享包不可见，旧 parser 因而报 ModuleNotFoundError；这是本次 R6 新增回归，不是历史失败。遵循 R9「发现失败应退回对应阶段」，暂停 R9 验收并返回 R6。

返修前完整 snapshot 为 `/home/hwz/CDLNO-artifacts/linearno-history-r6-correction-before-q_7wkrsw`；证据在 `linearno_history_audit/r6-correction/`。在真实 pre-R6 snapshot 重跑同检查通过，同时 KCDNO Air AST 检查失败，后者确认早已存在。未修改旧 golden 或容差。

只改变 `PDE-Solving-StandardBenchmark/cdlno_entry.py`：研究 intercept 的导入/调用移入现有 LinearNO 注册键分支，旧 Transolver/CDLNO/KCDNO/MSAR parser 不加载研究包；其他 family 携带创新 flag 仍由原 parser 作为未知参数立即拒绝。同步更新 `cdlno/linearno_history/provenance.py` 中精确可逆 routing 片段，原 pure source hash 仍必须一致。未改模型数学、数据、loss、scheduler 或 schema。

旧 parser 三个项目独立导入检查已通过。随后运行完整 `test_kcdno_delivery` 与 `linearno.test_history_static`，涵盖 80 构造、16 原生 train/checkpoint/resume/新进程eval、公平初始化和原 fingerprint。一次定向命令误写了不存在的测试方法名，日志 targeted.txt 保留；这不是模型失败，也未计作通过。固定命令 tests.txt 是验收依据。

本返修会更新研究源码hash，旧开发期研究 archive 不允许跨版本续训；pure archive 的源码身份保持一致。最终实验必须使用完成审查后的固定源码。

**PASS**：固定源码8/8方法通过381.743s。包括4个旧KCDNO delivery方法、80正式构造、16四静态题原生合成连续/中断恢复/新进程eval完整闭环，所有权重、RNG、batch/输出exact；源指纹与六exp不变。无测试或容差修改。返修结束，恢复R9。
