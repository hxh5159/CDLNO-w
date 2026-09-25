# Stage A reference audit

## Repository and dispatch

The current tree contains pure LinearNO and loop versions v1-v4. The public
V3 selector is `operator_latent_adapter_v3`; V4 is
`resmlp_dual_temp_v4`. No existing selector or extension uses
`partial_share_feature_gate_v5`.

The version-independent configuration dispatcher is
`linearno_loop/versioning.py`; runtime construction/checkpoint dispatch is
`cdlno/linearno_loop/versioning.py`. Standard selection enters through
`cdlno/linearno_loop/standard_entry.py`; AirfRANS and ShapeNet-Car enter
through `cdlno/linearno_loop/industrial_entry.py`. The native experiment
bodies remain responsible for data, losses, optimizer/scheduler, temporal
rollouts, metrics, plots, and task-specific outputs.

## Pure LinearNO facts

`cdlno/linearno/attention.py` computes shared-across-head Q/K/V projections,
K softmax over points, `K^T V`, Q softmax over latent rank, Q readout, and the
variant-specific native output projection. The active temperature parameters
are `temperature_q/k` for temp variants and the compatible
`tempreature_q/k` spelling for ShapeNet. AirfRANS' registered temperature is
inert in forward.

The selected pure profile supplies C, heads, M, and FFN ratio. For the paper
and official profiles, Airfoil/Darcy/Elasticity/Pipe/Plasticity use
C128/M64/F128; NS/AirfRANS/Car use C256/M32/F512. The
`transolver_matched` profile differs for Pipe (F256) and NS (F256), so V5 must
resolve F from the selected profile instead of a hard-coded task table.

## Ownership decision

V1 shares complete recurrent blocks. V2 shares the operator and owns point
FFNs per round. V3 shares complete recurrent blocks and adds latent processing
and a second-visit low-rank adapter. V4 owns eight independent operators,
shares five ResMLP owners, and adds dynamic temperatures. None matches V5.

V5 therefore needs a new owner graph: one non-Q/K operator body, LN1/LN2 and
dense expert bank per physical position; one Q/K/active-temperature/router per
logical visit. Prefix and suffix have one visit and fully independent owners.

## Frozen regions

Pure LinearNO, Transolver, loop v1-v4, task data readers, losses, schedules,
normalization, rollouts, industrial sampling/folds/ensemble behavior, metrics,
visualization math and old checkpoint schemas remain frozen. Shared files may
only gain branches selected by explicit V5 metadata or selector.

## Baseline limitations

This audit did not run real data, a complete epoch, convergence, SOTA,
distributed execution, or the remote Python 3.10/Torch 2.11 stack.
