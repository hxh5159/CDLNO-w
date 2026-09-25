# Stage B: V5 core, configuration, and ownership

## A. Authorized scope and status

Status: **PASS** for the data-free Stage B scope.

The new selector is `partial_share_feature_gate_v5`, with
`architecture_extension=partial_share_feature_gate`, `architecture_version=5`,
and `family=linearno_loop`. The default topology is `p2_c2_r2_s2`. The
alternative `p1_c3_r2_s1`, executed-depth shorthand, and explicit custom
`P/C/R/S` are supported without adding values to the v1-v4 preset contracts.

## B. Files and reasons

- `linearno_loop/v5/{contracts,config,schema}.py`: torch-free V5 contracts,
  profile resolution, topology validation, stable hashes, and metadata schema.
- `cdlno/linearno_loop/v5/{operator,core,construction}.py`: partial-shared
  native LinearNO operator, dense point experts, physical owner graph, and
  RNG-isolated config-first construction.
- `cdlno/linearno_loop/v5/{standard,airfrans,shapenet}.py`: wrappers preserving
  the three existing task interface families.
- `linearno_loop/versioning.py` and `cdlno/linearno_loop/versioning.py`:
  explicit V5-only config/runtime dispatch.
- `tests/loop_linearno_v5/{oracle,test_v5_contract,test_v5_integration}.py`:
  independent numerical oracle, ownership, initialization, gradient, and
  config tests.

No pure LinearNO, Transolver, or v1-v4 operator/core forward was edited.

## C. Formula and ownership mapping

For physical position `p` and visit `r`:

```text
Z = X + Operator[p,r](LN1[p](X))
U = LN2[p](Z)
pi = softmax(router[p,r](U), dim=expert)
G = sum_j pi[...,j] * Expert[p,j](U)
core:   X_next = Z + G / R
prefix/suffix: X_next = Z + G
```

The formula is implemented by `V5PhysicalBlock.forward` and
`V5LoopCore.forward` in `cdlno/linearno_loop/v5/core.py`. Every expert is
`Linear(C,F,bias) -> GELU -> Linear(F,C,bias)` and every expert executes.

The operator in `cdlno/linearno_loop/v5/operator.py` preserves native Q/K/V,
temperature, softmax, compression, reconstruction, and complete `to_out`
semantics:

```text
Q = softmax_M(to_q[p,r](features) / active_tau_q[p,r])
K = softmax_N(to_k[p,r](features) / active_tau_k[p,r])
context = K^T V[p]
output = to_out[p](Q context)
```

| Owner | Per physical position | Per logical visit |
|---|---|---|
| shared core | `in_project_x`, `to_v`, full `to_out`, inert Air temperature, LN1, LN2, all experts | no |
| route | no | `to_q`, `to_k`, active Q/K temperatures, `Linear(C,K,bias)` router |
| prefix/suffix | complete independent single-visit block | one visit each |
| final output | only the final suffix owns `ln_3 + mlp2` | executes once |

Initialization is complete-wrapper native initialization once, followed by
value copies Q-to-Q, K-to-K and active-temperature-to-same-temperature across
visits, then zero router weights and biases. Storage remains independent.
Experts are not copied and retain independent native initialization. Model
construction uses `torch.random.fork_rng`, so it does not advance the caller's
CPU RNG.

Schedules:

- `p1_c3_r2_s1`: `P0,A0,B0,C0,A1,B1,C1,S0`; five physical owners, eight visits.
- `p2_c2_r2_s2`: `P0,P1,A0,B0,A1,B1,S0,S1`; six physical owners, eight visits.

## D. Commands and evidence

```bash
PYTHONPATH="$PWD:$PWD/tests" pytest -q \
  tests/loop_linearno_v5/test_v5_contract.py \
  tests/loop_linearno_v5/test_v5_integration.py
```

These tests are included in the final V5 result of `48 passed`. They cover the
frozen hand example (`[[5.75,1.5],[2.2,5.6]]`), K=1, R=1/2/3, both presets,
custom topology, owner IDs, state keys, native operator parity, all-expert
gradients, second-visit Q/K/router gradients, JSON roundtrip, and strict
checkpoint conflicts. Full output is in
`evidence/stage_e/v5_full_pytest_final.log`.

Environment: Python 3.13.9, torch 2.13.0+cu130, PyG 2.3.1.

## E. Frozen-region evidence

- V5 lives in separate namespaces and requires its explicit selector.
- Old selectors and omitted architecture fields retain their existing routes.
- V5 does not use V3 latent FFNs/adapters, V4 ResMLPs/dynamic temperatures,
  AttnRes, sparse experts, or history caches.
- The final compatibility tests restore the historical pure LinearNO source
  and normalized-patch fingerprints exactly.

## F. Limits and priority review points

Real data, convergence, accuracy, and full-epoch training were not run.
Priority source review points are the visit/shared split in `operator.py`, the
`G/R` placement in `core.py`, one-time initialization in the three wrappers,
custom-R visit allocation, and task-profile derivation of `F` and `M`.

This Stage B is complete; Stage C was executed under the user's continuous authorization.
