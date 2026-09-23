# Reverse requirements matrix

| Frozen requirement | Config/owner | Forward/source | Evidence | Result/boundary |
|---|---|---|---|---|
| Explicit v4 selector and isolated schema | `linearno_loop/v4/config.py`, `contracts.py` | version dispatch only imports v4 when selected | `test_v4.py`, hardening, checkpoint review | PASS; old routes unchanged |
| Eight independent operators/LN1/LN2 | `v4/core.py` | eight visits in fixed topology | core/math tests; native run manifests | PASS |
| Five RMLP owners and A/B/C route | `v4/resmlp.py`, `core.py` | `[first,A,B,C,A,B,C,last]`, L2/L3 | ownership/hooks/parameter tests | PASS |
| FLARE L=2/3, GELU(tanh), endpoint conditions | `v4/resmlp.py` | four/five Linear modules; no LN/dropout | reference/math tests | PASS |
| Middle `1/sqrt(2)` raw branch scaling | `v4/core.py` | operator and RMLP branch only | independent core oracle | PASS |
| Base attention delegation and six variants | `v4/attention.py` | native projection/layout/output paths | attention tests | PASS; real data NOT RUN |
| Q/K adaptive temperatures | `v4/temperature.py` | shapes `[B,Hd,N,1]`, `[B,Hd,1,M]`, `[B,Hd,N,1]`; bounded multiplier | math/gradient/AMP tests | PASS |
| K over N, Q over M, KtV/QZ | `v4/attention.py` | no dense N×N/M×M | shape and accounting traces | PASS |
| Zero initialization and isolated predictor RNG | `v4/construction.py`, `temperature.py` | install after public initialization | RNG/zero-init/hardening tests | PASS |
| Original task H/head/M/ratio and protocols | profile resolver and wrappers | native six-task loops, Air/Car adapters | stable native matrix | PASS synthetic; real datasets NOT RUN |
| NS ten-step and Plasticity twenty-step behavior | task entry dispatch | original AST loops | standard worker reports | PASS synthetic |
| AirfRANS ensemble independence | `v4/airfrans.py`, entry | member-specific model/optimizer/archive | industrial worker and entry tests | PASS synthetic PyG; radius graph skipped without torch_cluster |
| ShapeNet M independent of d_h | `v4/shapenet.py`, config | actual M passed directly to q/k | ShapeNet construction/attention tests | PASS synthetic |
| Metadata-first strict checkpoint | `v4/checkpoint.py`, `schema.py` | sidecar/hash/config before `torch.load`, strict state | checkpoint review/hardening/native resume | PASS synthetic |
| Output/recording ownership | `v4/recording.py`, old recorder dispatch | v4 schedule recorded separately | recorder tests/native manifests | PASS |
| Cost accounting | `tools/linearno_loop_accounting.py`, v4 cost review | partitions, matrix MAC/FLOPs, non-matrix inventory | `COSTS.md`, costs JSON | PASS for stated matrix口径; not measured real speed |
| Old pure/V1/V2/V3 compatibility | old dispatch/projections untouched | old artifact routes remain strict | pure integration; legacy evidence | PARTIAL: inherited inventory/provenance/float/import failures remain |
| Reproducible run scripts | `tran_evaluate/linearno_loop_v4` | train/resume/eval/dry-run and saved config | launcher suite and `USAGE.md` | PASS parser/synthetic; real remote stack NOT RUN |
| Final scientific claims | documentation only | no claim of accuracy/acceleration | final status | NOT RUN: data, convergence, SOTA, epoch speed, distributed/compile |
