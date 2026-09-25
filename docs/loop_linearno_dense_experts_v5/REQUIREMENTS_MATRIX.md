# V5 reverse requirements matrix

Status values refer only to the authorized no-real-data implementation and
validation scope.

| Frozen requirement | Implementation | Test/evidence | Status |
|---|---|---|---|
| Explicit isolated V5 selector/schema | `linearno_loop/v5/contracts.py`, `config.py`, `schema.py`; both version dispatchers | config/hash/roundtrip and pre-tensor conflict tests | PASS |
| P1, P2, custom P/C/R/S and depth shorthand | `linearno_loop/v5/config.py`; `V5LoopCore.visit_schedule` | preset ownership, R1/R2/R3, invalid combinations, eight parser previews | PASS |
| Core non-Q/K operator body/experts shared by physical position | `cdlno/linearno_loop/v5/{operator,core}.py` | module/parameter identity, state-key and recorder schedule checks | PASS |
| Core LN1/LN2 default to independent `(position,visit)` ownership; explicit shared compatibility mode | `V5PhysicalBlock.norms_for_visit`; V5 config/CLI/checkpoint/recording | storage identity, equal-init output/RNG, R2/R3 gradients, exact parameter delta, strict pair tests | PASS |
| Q/K, active temperatures, router independent by visit | `VisitRouting`, `PartialSharedLinearNOOperator` | distinct storage/equal initial values, visit gradients and divergence | PASS |
| Prefix/suffix complete and independent; final head once | `V5LoopCore` construction/forward | owner graph, call schedule, finalization checks | PASS |
| Native six attention variants and Q/K axes | `PartialSharedLinearNOOperator.forward` | exact output/input-gradient parity for plain/temp/conv/conv_temp/AirfRANS/ShapeNet | PASS |
| No explicit N-by-N or M-by-M attention | Q/K shape path in `operator.py` | Q/K hook shape trace and forbidden-attention audit | PASS |
| K complete biased GELU two-layer experts all execute | `DensePointExpert`, `V5PhysicalBlock.forward` | hand oracle, K=1, all-expert call and gradient checks | PASS |
| Point router softmax only on expert axis | `router(u).softmax(dim=-1)` | hand oracle, per-point probabilities and gradient tests | PASS |
| Operator residual 1; core expert residual 1/R only | `V5PhysicalBlock.forward`, `V5LoopCore.forward` | frozen hand oracle and R1/R2/R3 checks | PASS |
| Native whole-model initialization once; Q/Q and K/K copied; routers zero | three V5 wrappers plus `synchronize_visit_initialization` | exact common tensors, storage identity, independent experts, RNG isolation | PASS |
| K/F independent; default F from selected profile | `resolve_config` | all eight default F/M assertions and independent override matrix | PASS |
| Standard six task contracts | V5 Standard wrapper and Standard entry adapters | native task AST workers: loss, temporal call counts, save/resume/eval | PASS, synthetic inputs |
| AirfRANS contract and member isolation | V5 AirfRANS wrapper and `air_entry.py` | native weighted-loss/member checkpoint/eval test | PASS, synthetic PyG |
| ShapeNet-Car tuple/fold/output/drag boundary | V5 ShapeNet wrapper and `car_entry.py` | native graph train/save/resume/eval boundary | PASS, synthetic PyG |
| Metadata-first, immutable, strict checkpoint pairs | V5 schema/checkpoint modules | checksum/path/config/tamper/cross-version/strict reload negatives | PASS |
| Complete optimizer/scheduler/RNG/sampler state | tagged resume metadata and existing run adapters | interrupted Standard continuation and industrial resume tests | PASS, controlled runs |
| Unique run/output and actual owner/call recording | V5 run ID, recorder, existing task recorders | config-hash/run-ID and first-forward manifest checks | PASS |
| train/resume/eval/train_eval launchers for eight tasks | `tran_evaluate/linearno_loop_v5/` | shell syntax, 8/8 parser dry-runs, selected-GPU train_eval test | PASS |
| Parameter and matrix-operation accounting | `tools/linearno_loop_accounting.py`, `linearno_loop_v5_report.py` | 104 analytic/live rows, hook traces, 10 bounded timing rows | PASS |
| Original Transolver selectors, outputs and checkpoints remain compatible | unchanged task-local Transolver modules and guarded entry branches | eight-task fresh-workdir frozen fixture, strict state reload, Car/AirfRANS local object roundtrip | PASS |
| Pure LinearNO and V1-V4 remain compatible | explicit dispatcher branches and projection guards | V4 136 pass/1 skip; V3 isolated 6 pass/1 skip; V2 34 pass; provenance checks | PARTIAL due inherited aggregate-suite failures documented in final report |
| Real-data convergence, accuracy, SOTA and epoch performance | intentionally outside authorization | no evidence generated | NOT RUN |

The detailed raw logs are under `evidence/stage_e/`. A PASS on a synthetic task
row confirms interface and lifecycle behavior; it does not assert benchmark
accuracy or convergence.
