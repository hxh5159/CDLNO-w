# L0 阅读与证据索引

本目录是审计清单，不是生产LinearNO配置包。固定来源及外部副本见sources.json；所有文件清单以Git固定树枚举，非`rg`的忽略后子集。

`reading-inventory.json` 的 `read` 表示完整文本已读取并进行相应全文/AST/配置/历史记录结构审阅；`group`表示完全相同SHA的代表已读取；`excluded`给出二进制路径、大小、哈希和理由。该口径不等于所有函数动态执行或逐行数学证明。Python读取包含完整AST，不仅搜索已猜测的几个模型文件。模型/训练入口/数据loss/初始化/checkpoint和论文冲突另在两份报告给出语义结论。

| 文件 | 内容 |
|---|---|
| baseline.json、manifest.json | 实际HEAD/tree/remote/原始git状态、全部tracked/untracked/ignored基线 |
| freeze-manifest.json | 重点保护路径、起点缺失目录和221文件分类/size/hash；不放过untracked |
| rg-files.txt、reading-inventory.json | rg发现与三树逐文件读取/分组/排除清单 |
| source-contracts.json | 全部唯一Python类/函数/签名、AST hash、调用、测试断言、import和顶层副作用 |
| source-anchors.json | 关键模型、训练、指标与存档符号的固定源码行号 |
| configuration-shells.json | 全文配置/shell/依赖及全部notebook code cells（仅只读） |
| document-sections.json | 既存README/阶段/计划文档的标题、段落计数和审阅范围；不重新发布完整历史文档副本 |
| transolver-comparison.json、upstream-to-target.diff | 当前目标与固定thuml的逐路径blob及普通diff，不用共同祖先假设 |
| transolver-to-linearno.diff、ast-comparison.json | 两官方任务文件差异、映射到目标的类/函数AST比较 |
| cli-inventory.json | 三树全部parser.add_argument表达式及行号；不用import exp/main |
| released-standard-commands.json | 六个shell的完整真实argv、parser默认、解析后值；模型kwargs与动作分开 |
| paper-equations.json | v3公式LaTeX与锚点；嵌套equationgroup原始提取有子式，报告按公式编号归并 |
| profile-inventory.json | 8任务×paper/release/可选matched；非执行配置，UNRESOLVED明确保留 |
| regression-index.json | 既有K0/M0/M8回归索引及文件哈希/存在性；本轮未load/replay权重 |
| syntax-checks.json | 各源码AST/shell语法结果及实际耗时，不代表有数据可跑 |
| legacy-static-results.json、legacy-static-tests.txt | 六项既存源码冻结测试的完整命令、计时和原始日志 |
| ignored-output-policy.json | 唯一允许的新增ignored专属目录；仍受后续阶段/数据/GPU授权约束 |
| environment.json | 当前版本/可用设备只读查询，非远端环境验证 |
| delivery-check.json | 最终已有文件/状态/固定来源哈希检查及本轮新增清单 |

旧阶段 JSON/log/patch 的读取用于判断证据来自哪个阶段和哪份源码，不能将其中的GPU/训练步成功重复计为L0执行。物理数据和pickle没有加载。参考树均在目标外；二进制未解包、未当当前模型实现。
