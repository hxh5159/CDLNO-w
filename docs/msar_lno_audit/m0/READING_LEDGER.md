# M0 阅读账本（2026-09-16，M1之后补审）

本账本对应 inventory.json 的审计起点，不回写旧文件，不把本次阅读倒填成M1之前。
完整路径、SHA256、行数、源码符号在 reading-index.json / inventory.json；每个Python函数的AST hash和相关调用在 source-contracts.json。
全文读取、语法/AST解析属于静态证据，并不证明每一行语义无误。语义结论限于下列审查合同与主报告实际列举内容；运行证据另见回放结果和regression.log。

## 实际审查分组

| 范围 | 阅读与交叉核对重点 | 结论位置 |
|---|---|---|
| 根AGENTS、README、pyproject、忽略规则、Physics_Attention | 当前授权与历史停止记录区分；共享包发现/PYTHONPATH；根attention非运行副本 | 主报告A/B/D |
| 三子项目所有model/models及factory | 原Transolver逐副本差异、CDLNO前段三模式、K all/off/matched、Air其他四模型、3D模型；输入/输出/初始化/稳定类名 | 主报告C/D；75夹具 |
| cdlno公共modules/core/cdpa/config/checkpoint | Down learned Q、bias/norm、双FFN与SA边界、Up branch、Bridge/readout residual、历史与严格兼容 | 主报告D |
| cdlno/kcdno全部模块 | 源writer/receiver reader、FP32生产/double reference、matched完整block、CLI显式覆盖、Run及pickle验证 | 主报告B/D/F |
| cdlno/msar_lno全部M1模块及M1测试 | 仅配置注册；Light/Full、profile优先级、coverage与结构分离、eval先读不写、目录隔离 | 主报告F；13实际测试 |
| 六个exp及其入口helper、四个工业main/eval、两个train | parser及导入副作用、八任务真实loss、eval、优化器/调度、teacher forcing、逐T更新、文件保存 | 主报告B/C/G；源码合同 |
| 三项目dataset及utils所有源码 | 字段/切分/采样/点序/图、normalizer/TestLoss、surf、系数/VTK指标、硬编码路径 | 主报告C/G；无数据导入 |
| 正式JSON/YAML/shell、根path、顺序/seed/ablation启动器 | preset与parser裸默认区别、余参覆盖、同run训练评估、失败停止；没有执行训练 | configs-shells；70 bash -n |
| experiment/training_state/observer/visualization/periodic_visualization | 记录与模型分离；现有任务仅权重保存不等于完整resume；周期图已接，V1 archive未接task resume | 主报告C/G |
| tran_evaluate/show | 只读展示；按每任务最后验证loss选单run，PDE汇总，不删源结果，不按test选seed | 主报告C/G；已有show测试 |
| tools/cdlno_perf和环境preflight | 实际/公式参数成本、NM投影/Conv/最终读出、hook计数、backend/warmup/sync/median/p90/峰值/训练步 | 主报告D/F；不运行新性能实验 |
| tests全部源码/辅助reference/projection | 全部方法/断言索引，动态任务矩阵和AST冻结、原loss/时间/图/参数/AMP/归档测试范围，skip条件 | test-contracts；316运行记录 |
| docs已有审计脚本/历史证据 | 夹具生成/同权重replay、来源hash和当时边界，区分旧结论与当前接入；旧日志只作证据 | 主报告A/E/G；fixture-index |
| PLAN和memory、任务/启动README | 当前公式优先；旧候选不授权；M1先于M0的历史保留；未来M1—M9范围来自实际源码 | 主报告A/F |
| Notebook/其他文本 | 读取文本/code cells，不执行数据统计；LICENSE保留 | inventory；未作运行声明 |

对于同名模板，先比较hash，再逐文件检查差异。没有把六exp、工业wrapper或Physics_Attention副本当作完全相同文件。只有下列全文SHA256相同的组可共用文本阅读，路径和用途仍分别记录。

## 精确重复组

- `docs/front_ablation_audit/a1/full-airfrans.json`、`docs/front_ablation_audit/a2/full-airfrans.json`、`docs/front_ablation_audit/a3/full-airfrans.json`
- `docs/front_ablation_audit/a1/full-car.json`、`docs/front_ablation_audit/a2/full-car.json`、`docs/front_ablation_audit/a3/full-car.json`
- `docs/front_ablation_audit/a1/full-core.json`、`docs/front_ablation_audit/a2/full-core.json`、`docs/front_ablation_audit/a3/full-core.json`
- `docs/kcdno_audit/cli-airfrans.txt`、`docs/kcdno_audit/cli-car.txt`
- `docs/kcdno_audit/k1/replay-after-airfrans.json`、`docs/kcdno_audit/k1/replay-before-airfrans.json`、`docs/kcdno_audit/k10/old-replay-airfrans.json`、`docs/kcdno_audit/k2/old-model-replay-airfrans.json`、`docs/kcdno_audit/replay-airfrans.json`、`docs/msar_lno_audit/m1/old-after-airfrans.json`、`docs/msar_lno_audit/m1/old-before-airfrans.json`
- `docs/kcdno_audit/k1/replay-after-car.json`、`docs/kcdno_audit/k1/replay-before-car.json`、`docs/kcdno_audit/k10/old-replay-car.json`、`docs/kcdno_audit/k2/old-model-replay-car.json`、`docs/kcdno_audit/replay-car.json`、`docs/msar_lno_audit/m1/old-after-car.json`、`docs/msar_lno_audit/m1/old-before-car.json`
- `docs/kcdno_audit/k1/replay-after-standard.json`、`docs/kcdno_audit/k1/replay-before-standard.json`、`docs/kcdno_audit/k10/old-replay-standard.json`、`docs/kcdno_audit/k2/old-model-replay-standard.json`、`docs/kcdno_audit/replay-standard.json`、`docs/msar_lno_audit/m1/old-after-standard.json`、`docs/msar_lno_audit/m1/old-before-standard.json`
- `docs/kcdno_audit/k10/old-replay.txt`、`docs/msar_lno_audit/m1/old-after.txt`、`docs/msar_lno_audit/m1/old-before.txt`
- `docs/periodic_visualization_audit/synthetic_previews/curved_grid/caption.tex`、`docs/periodic_visualization_audit/synthetic_previews/point_cloud_hole/caption.tex`、`docs/periodic_visualization_audit/synthetic_previews/surface_3d/caption.tex`
- `docs/periodic_visualization_audit/synthetic_previews/curved_grid/caption.txt`、`docs/periodic_visualization_audit/synthetic_previews/point_cloud_hole/caption.txt`、`docs/periodic_visualization_audit/synthetic_previews/surface_3d/caption.txt`

## 实际现有文档路径

共67份Markdown。下列路径对应 document-reading.json 中保留的标题及完整段落；本轮新增主报告/账本不计入起点。阶段文档保留时间语境，不以早期“未接入”覆盖当前源码。

- `AGENTS.md`
- `Airfoil-Design-AirfRANS/README.md`
- `Car-Design-ShapeNetCar/README.md`
- `PDE-Solving-StandardBenchmark/README.md`
- `PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md`
- `PLAN_CDLNO/CDPA_v1_1/CDPA_Transolver_Implementation_Plan_v1_1.md`
- `PLAN_CDLNO/CDPA_v1_1/README.md`
- `PLAN_KCDNO/KCDNO_Codex_Staged_Prompts.md`
- `PLAN_KCDNO/KCDNO_Model_Specification_v1.md`
- `PLAN_MSAR_LNO/MRSA_LNO_arch.md`
- `PLAN_MSAR_LNO/MSAR_LNO_Codex_Staged_Prompts.md`
- `README.md`
- `docs/CDLNO_EXPERIMENT_OUTPUTS.md`
- `docs/CDLNO_EXPERIMENT_OUTPUTS_REPORT.md`
- `docs/CDLNO_FRONT_ABLATION.md`
- `docs/CDLNO_FRONT_ABLATION_A1.md`
- `docs/CDLNO_FRONT_ABLATION_A2.md`
- `docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md`
- `docs/CDLNO_FRONT_ABLATION_A3.md`
- `docs/CDLNO_FRONT_ABLATION_A4.md`
- `docs/CDLNO_IMPLEMENTATION_REPORT.md`
- `docs/CDLNO_IMPLEMENTATION_STATUS.md`
- `docs/CDLNO_PERFORMANCE_TOOLS.md`
- `docs/CDLNO_PERIODIC_VISUALIZATION.md`
- `docs/CDLNO_PHASE2_MODULES.md`
- `docs/CDLNO_PHASE3_CDPA.md`
- `docs/CDLNO_PHASE4_CORE.md`
- `docs/CDLNO_PHASE5_STATIC_TASKS.md`
- `docs/CDLNO_PHASE6_TEMPORAL_TASKS.md`
- `docs/CDLNO_PHASE7_SHAPENET_CAR.md`
- `docs/CDLNO_PHASE8_AIRFRANS.md`
- `docs/CDLNO_PHASE9_PERFORMANCE.md`
- `docs/CDLNO_REFERENCE_AUDIT.md`
- `docs/CDLNO_REMOTE_LAUNCHERS.md`
- `docs/CDLNO_REQUIREMENTS_MATRIX.md`
- `docs/CDLNO_RESULT_REPORTS.md`
- `docs/CDLNO_SEQUENTIAL_LAUNCH.md`
- `docs/CDLNO_TASK_LAUNCHERS.md`
- `docs/CDLNO_THIRD_PARTY_NOTICES.md`
- `docs/CDLNO_TRAINING_LAUNCH_REVIEW.md`
- `docs/CDLNO_VISUALIZATION_RESUME_PLAN.md`
- `docs/CDLNO_VISUALIZATION_RESUME_V1.md`
- `docs/KCDNO_COMMANDS.md`
- `docs/KCDNO_IMPLEMENTATION_REPORT.md`
- `docs/KCDNO_IMPLEMENTATION_STATUS.md`
- `docs/KCDNO_K1_CONFIGURATION.md`
- `docs/KCDNO_K2_KERNEL_HISTORY.md`
- `docs/KCDNO_K3_CORE.md`
- `docs/KCDNO_K4_STATIC.md`
- `docs/KCDNO_K5_TEMPORAL.md`
- `docs/KCDNO_K6_CAR.md`
- `docs/KCDNO_K7_AIRFRANS.md`
- `docs/KCDNO_K8_MATCHED.md`
- `docs/KCDNO_K9_PERFORMANCE.md`
- `docs/KCDNO_REFERENCE_AUDIT.md`
- `docs/KCDNO_REQUIREMENTS_MATRIX.md`
- `docs/KCDNO_SEED_SUITE.md`
- `docs/MSAR_LNO_IMPLEMENTATION_STATUS.md`
- `docs/MSAR_LNO_M1_CONFIGURATION.md`
- `memory/2026-09-13-cdpa-plan-and-reference-audit.md`
- `memory/2026-09-13-exported-conversation-review.md`
- `memory/README.md`
- `memory/current-state.md`
- `tran_evaluate/README.md`
- `tran_evaluate/ablation/README.md`
- `tran_evaluate/kcdlno/README.md`
- `tran_evaluate/show/README.md`

## 排除项

不遍历.git内部、缓存、构建产物或外部真实数据/实验目录；外部回归夹具只按既有索引读取。清单内25个二进制仅记录hash，未当作源码：

- `Airfoil-Design-AirfRANS/fig/results.png`：binary (figure/PDF/other); not executable source
- `Airfoil-Design-AirfRANS/fig/task.png`：binary (figure/PDF/other); not executable source
- `Car-Design-ShapeNetCar/fig/car_slice_surf.png`：binary (figure/PDF/other); not executable source
- `Car-Design-ShapeNetCar/fig/case_study.png`：binary (figure/PDF/other); not executable source
- `Car-Design-ShapeNetCar/fig/results.png`：binary (figure/PDF/other); not executable source
- `Car-Design-ShapeNetCar/fig/task.png`：binary (figure/PDF/other); not executable source
- `PDE-Solving-StandardBenchmark/fig/scalibility.png`：binary (figure/PDF/other); not executable source
- `PDE-Solving-StandardBenchmark/fig/showcase.png`：binary (figure/PDF/other); not executable source
- `PDE-Solving-StandardBenchmark/fig/standard_benchmark.png`：binary (figure/PDF/other); not executable source
- `PLAN_CDLNO/比较模型架构.pdf`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/curved_grid/fields.npz`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/curved_grid/fields.pdf`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/curved_grid/fields.png`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/point_cloud_hole/fields.npz`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/point_cloud_hole/fields.pdf`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/point_cloud_hole/fields.png`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/surface_3d/fields.npz`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/surface_3d/fields.pdf`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/surface_3d/fields.png`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/time_curve/curve.npz`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/time_curve/curve.pdf`：binary (figure/PDF/other); not executable source
- `docs/periodic_visualization_audit/synthetic_previews/time_curve/curve.png`：binary (figure/PDF/other); not executable source
- `pic/Transolver.png`：binary (figure/PDF/other); not executable source
- `pic/physical_states.png`：binary (figure/PDF/other); not executable source
- `pic/showcases.png`：binary (figure/PDF/other); not executable source

## 外部参考与证据范围

本地LRSA `/home/hwz/LRSA-Operator` commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e` 的 `src/perceiverforpde/modeling/layers/attn.py`：检查两个forward中`disable_interleaved_blocks`包围SA及第二FFN；不是MSAR可直接使用的消融开关。现有参考测试通过环境变量启用并真实执行，输出/T/梯度精确一致。

未重新OCR旧会话PDF，也未声称本轮重新审阅论文图像；设计依据是当前用户总控、MSAR文本架构和staged prompts。可执行证据来自真实模型/原有测试、真实PyG对象及有限GPU，不含真实数据集、收敛、完整Air radius抽样或远端环境验收。
