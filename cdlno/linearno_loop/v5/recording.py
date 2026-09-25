"""V5 ownership and actual-call evidence for launcher-created runs."""
import json
from pathlib import Path

from linearno_loop.v5.config import validate_config
from .checkpoint import measure_parameters


def _visits(config):
    spec = config["loop_spec"]
    visits = [("prefix", index, 0) for index in range(spec["prefix_blocks"])]
    visits += [("core", index, round_index)
               for round_index in range(spec["loop_repeats"])
               for index in range(spec["recurrent_core_blocks"])]
    visits += [("suffix", index, 0) for index in range(spec["suffix_blocks"])]
    return visits


def expected_schedule(config):
    events = []
    visits = _visits(config)
    for logical_index, (group, position, visit) in enumerate(visits):
        owner = f"loop.{group}.{position}"
        events.extend((
            owner + ".ln_1",
            owner + ".shared.in_project_x",
            owner + f".visit.{visit}.to_q",
            owner + f".visit.{visit}.to_k",
            owner + ".shared.to_v",
            owner + ".shared.to_out",
            owner + ".ln_2",
            owner + f".visit.{visit}.router",
        ))
        events.extend(owner + f".experts.{expert}"
                      for expert in range(config["expert_count"]))
        if logical_index == len(visits) - 1:
            events.extend((owner + ".final_norm", owner + ".head"))
    return events


def observe(args, model, member=0):
    from cdlno.training_state import _atomic
    from cdlno.linearno_loop.industrial_state import member_seed
    from tran_evaluate.linearno_loop.recording import state_hash

    config = validate_config(args._linearno_loop_config)
    path = Path(args.linearno_run_dir) / "loop_run_manifest.json"
    schedule = expected_schedule(config)
    spec = config["loop_spec"]
    base = dict(
        schema_version=5,
        architecture=config["architecture"],
        architecture_extension=config["architecture_extension"],
        family="linearno_loop",
        config_hash=config["config_hash"],
        task=config["task"],
        profile=config["profile"],
        topology=config["topology_preset"],
        residual_mode=config["residual_mode"],
        unique_depth=spec["unique_depth"],
        executed_depth=spec["executed_depth"],
        expert_count=config["expert_count"],
        expert_width=config["expert_width"],
        actual_M=config["actual_M"],
        ownership="physical_shared_body_experts_visit_owned_qk_router",
        state_partition=spec["state_partition"],
        expected_call_schedule=schedule,
        fair_comparison=config["fair_comparison"],
    )
    key = f"member_{member:03d}"
    if path.exists():
        saved = json.loads(path.read_text())
        for field, value in base.items():
            if saved.get(field) != value:
                raise ValueError("V5 run manifest mismatch: " + field)
        if key in saved["members"]:
            if not (args.eval or args.resume):
                raise ValueError("V5 initial member already recorded")
            return model
    elif args.eval or args.resume:
        return model
    else:
        saved = {**base, "members": {}}
    if args.eval or not path.parent.is_dir():
        raise ValueError("V5 recorder requires a reserved training run")
    state = model.state_dict()
    measurement = measure_parameters(model, config)
    shared = {name: value for name, value in state.items()
              if ".visits." not in name}
    saved["members"][key] = dict(
        initial_state_sha256=state_hash(state),
        shared_owner_initial_sha256=state_hash(shared),
        parameters=measurement["total"],
        parameter_parts=measurement["groups"],
        physical_block_owners=spec["unique_depth"],
        logical_block_visits=spec["executed_depth"],
        qk_router_owners=spec["executed_depth"],
        expert_owners=spec["unique_depth"] * config["expert_count"],
        expert_calls=spec["executed_depth"] * config["expert_count"],
        initialization_seed=(member_seed(args, member)
                             if config["task"] in ("car", "airfrans") else args.seed),
        dataloader_generator_seeds=config["fair_comparison"]["dataloader_generators"],
        actual_call_schedule=None,
        observation="pending_first_successful_forward",
    )
    _atomic(path, saved, json_file=True)

    events, handles = [], []

    def begin(*unused):
        events.clear()

    def event(name):
        def record(*unused):
            events.append(name)
        return record

    def end(module, inputs, output):
        if output is None:
            events.clear()
            return
        try:
            if events != schedule:
                raise ValueError("V5 actual call schedule differs from ownership contract")
            current = json.loads(path.read_text())
            current["members"][key].update(
                actual_call_schedule=list(events),
                observation="first_forward_verified",
            )
            _atomic(path, current, json_file=True)
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            events.clear()

    handles.append(model.register_forward_pre_hook(begin))
    for group in ("prefix", "core", "suffix"):
        for position, block in enumerate(getattr(model.loop, group)):
            owner = f"loop.{group}.{position}"
            handles.append(block.ln_1.register_forward_pre_hook(event(owner + ".ln_1")))
            attention = block.Attn
            handles.append(attention.in_project_x.register_forward_pre_hook(
                event(owner + ".shared.in_project_x")))
            for visit, route in enumerate(attention.visits):
                handles.append(route.to_q.register_forward_pre_hook(
                    event(owner + f".visit.{visit}.to_q")))
                handles.append(route.to_k.register_forward_pre_hook(
                    event(owner + f".visit.{visit}.to_k")))
                handles.append(route.router.register_forward_pre_hook(
                    event(owner + f".visit.{visit}.router")))
            handles.append(attention.to_v.register_forward_pre_hook(
                event(owner + ".shared.to_v")))
            handles.append(attention.to_out.register_forward_pre_hook(
                event(owner + ".shared.to_out")))
            handles.append(block.ln_2.register_forward_pre_hook(event(owner + ".ln_2")))
            for expert, module in enumerate(block.experts):
                handles.append(module.register_forward_pre_hook(
                    event(owner + f".experts.{expert}")))
            if block.last_layer:
                handles.append(block.ln_3.register_forward_pre_hook(
                    event(owner + ".final_norm")))
                handles.append(block.mlp2.register_forward_pre_hook(
                    event(owner + ".head")))
    handles.append(model.register_forward_hook(end, always_call=True))
    return model
