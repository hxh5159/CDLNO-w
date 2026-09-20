#!/usr/bin/env python3
"""Offline CPU model inventory. Writes only inside this delivery directory.

Run with python -B from this checkout. Existing experiment records are never
overwritten. No datasets, benchmark entry execution, forward, optimizer or GPU.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import warnings

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STANDARD = ROOT / 'PDE-Solving-StandardBenchmark'
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(STANDARD)]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore', category=FutureWarning, module='timm.*')

import torch
from cdlno.linearno.profiles import resolve_config as pure_config, validate_resolved
from cdlno.linearno.standard_entry import constructor_kwargs
from cdlno.linearno_loop.construction import build_from_config
from linearno_loop.config import resolve_config, validate_config, run_directory_id
from tools.linearno_loop_accounting import analytic, measured_parameters
from tran_evaluate.linearno_loop.recording import schedule, state_hash, is_router

TASKS = ('airfoil', 'darcy', 'elasticity', 'pipe', 'ns', 'plasticity')
STATIC = TASKS[:4]
MODES = ('sr_1_over_r', 'rb_attnres', 'lb_attnres_1_over_r')
PRESETS = ('p1_c3_r2_s1', 'p2_c2_r2_s2')
PROFILES = ('paper_table8_on_release_model', 'official_release', 'transolver_matched')
SEEDS = (0, 1, 2)
FILES = dict(airfoil='airfoil', darcy='darcy', elasticity='elas', pipe='pipe', ns='ns', plasticity='plas')
SCRIPTS = dict(airfoil='Airfoil', darcy='Darcy', elasticity='Elas', pipe='Pipe', ns='NS', plasticity='Plas')
NPOINTS = dict(airfoil=221*51, darcy=85*85, elasticity=972, pipe=129*129, ns=64*64, plasticity=101*31)
REMOTE = '/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor'
DATA_ROOT = '/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno'
native = importlib.import_module('PDE-Solving-StandardBenchmark.model.LinearNO')
trans_structured = importlib.import_module('model.Transolver_Structured_Mesh_2D').Model
trans_irregular = importlib.import_module('model.Transolver_Irregular_Mesh').Model


class TransolverCPUPositionReference(trans_structured):
    """Only move deterministic, non-parameter get_grid arithmetic to CPU.

    Transolver's own __init__, learned modules, initialization, state_dict and
    forward are inherited unchanged. This helper is never a training model.
    """
    def get_grid(self, batchsize=1):
        return native._unified_positions(self.H, self.W, self.ref).repeat(batchsize, 1, 1, 1)


def write(relative, value):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def contract(task):
    return 'standard_temporal_l5' if task in ('ns', 'plasticity') else 'standard_static_l4'


def original_arguments(task):
    """Only compile parser declaration nodes, never execute dataset entry code."""
    entry = STANDARD / f'exp_{FILES[task]}.py'
    nodes = []
    for node in ast.parse(entry.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'parser.add_argument':
            nodes.append(node)
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<parser-declarations-only>', 'exec'), scope)
    source = STANDARD / f'scripts/Transolver_{SCRIPTS[task]}.sh'
    argv = shlex.split(source.read_text().replace('\\\n', ' '))[2:]
    return vars(scope['parser'].parse_args(argv)), source.relative_to(ROOT).as_posix()


ORIGINAL = {t: original_arguments(t) for t in TASKS}


def topology(depth, preset):
    if depth == 8:
        return dict(topology_preset=preset)
    p = 1 if preset == PRESETS[0] else 2
    return dict(topology_preset='custom', prefix_blocks=p,
                recurrent_core_blocks=(depth-2*p)//2, loop_repeats=2, suffix_blocks=p)


def parts_for_baseline(model):
    parts = dict(stem=0, independent_block_bodies=0, head=0, router=0)
    for name, param in model.named_parameters():
        key = 'head' if '.ln_3.' in name or '.mlp2.' in name else 'independent_block_bodies' if name.startswith('blocks.') else 'stem'
        parts[key] += param.numel()
    return parts


def transolver_formula(kw, task):
    d, h, m, f, layers = (kw[k] for k in ('n_hidden', 'n_head', 'slice_num', 'mlp_ratio', 'n_layers'))
    dh = d//h
    kernel = 1 if task == 'elasticity' else 9
    channels = kw['fun_dim'] + (kw['ref']**2 if kw['unified_pos'] else kw['space_dim'])
    stem = 2*d*channels + 2*d + 2*d*d + d + d
    if kw['Time_Input']:
        stem += 2*(d*d+d)
    attention = 2*(kernel*d*d+d) + (dh+1)*m + 3*dh*dh + (d*d+d) + h
    body = attention + 4*d + 2*f*d*d + (f+1)*d
    head = 2*d + d*kw['out_dim'] + kw['out_dim']
    n = NPOINTS[task]
    stem_mac = n*(2*d*channels + 2*d*d + (2*d*d if kw['Time_Input'] else 0))
    body_mac = n*(2*kernel*d*d+3*d*m+d*d+2*f*d*d) + 3*h*m*dh*dh + 2*d*m*m
    return dict(parameter_parts=dict(stem=stem, independent_block_bodies=layers*body, head=head, router=0),
                parameters=stem+layers*body+head, executed_matrix_macs=stem_mac+layers*body_mac+n*d*kw['out_dim'],
                token_self_attention=True, matrix_flops_are_total_flops=False)


def baseline_linear_formula(task, profile, depth, rank, seed):
    c = resolve_config(task, profile, options=dict(topology_preset='custom', prefix_blocks=0,
        recurrent_core_blocks=depth-1, loop_repeats=1, suffix_blocks=1,
        residual_mode='sr_1_over_r', linearno_rank=rank), profile_overrides={'runtime.seed': seed})
    a = analytic(c)
    p = a['parameter_parts']
    return dict(parameter_parts=dict(stem=p['stem'], independent_block_bodies=p['shared_core']+p['suffix_body'], head=p['head'], router=0),
                parameters=a['parameters'], executed_matrix_macs=a['executed_matrix_macs'],
                token_self_attention=False, matrix_flops_are_total_flops=False)


def measure(model, expected, *, loop):
    named = dict(model.named_parameters())
    total = sum(p.numel() for p in named.values())
    trainable = sum(p.numel() for p in named.values() if p.requires_grad)
    parts = measured_parameters(model) if loop else parts_for_baseline(model)
    assert total == expected['parameters'] == sum(parts.values())
    assert parts == expected['parameter_parts'], (parts, expected['parameter_parts'])
    state = model.state_dict()
    inventory = [dict(name=k, shape=list(v.shape), numel=v.numel(), dtype=str(v.dtype), requires_grad=v.requires_grad) for k,v in named.items()]
    assert sum(i['numel'] for i in inventory) == total
    return dict(total=total, trainable=trainable, non_trainable=total-trainable,
                backbone=total-parts['router'], router=parts['router'], by_component=parts,
                inventory=inventory, independent_formula_total=expected['parameters'], formula_equals_measured=True), dict(
                key_count=len(state), tensor_numel=sum(p.numel() for p in state.values()),
                key_shape_sha256=digest([(k,list(v.shape),str(v.dtype)) for k,v in state.items()]),
                initial_state_sha256=state_hash(state),
                public_backbone_initial_sha256=state_hash({k:v for k,v in state.items() if not is_router(k)}))


def protocol(task, profile_values, *, trans_args=None):
    values = deepcopy(profile_values)
    train = values['training']
    if trans_args is not None:
        for k, src in (('epochs','epochs'),('batch_size','batch_size'),('lr','lr'),('weight_decay','weight_decay'),('gradient_clip','max_grad_norm')):
            train[k] = trans_args[src]
        train['test_batch_size'] = trans_args['batch_size']
        train['seed_policy'] = 'catalog initialization seeds 0/1/2; original Transolver launcher has no seed flag; no training/dataloader seeding claimed'
        values['objective']['reduction'] = 'sum sample-wise flattened ratios per batch; no epsilon (original TestLoss(size_average=False))'
        values['evaluation']['batch_size'] = trans_args['batch_size']
        values['evaluation']['normalizer'] = 'original entry behavior; do not infer LinearNO metadata-first saved-normalizer loading'
    nt = 900 if task == 'plasticity' else 1000
    batch = train['batch_size']
    steps = math.ceil(nt/batch)
    updates = 20 if task == 'plasticity' else 1
    scheduler = dict(kind=train['scheduler'], epochs=train['epochs'], outer_batches_per_epoch=steps,
                     optimizer_steps_per_outer_batch=updates, optimizer_steps_total=steps*updates*train['epochs'],
                     source='derived from planned split/batch; no loader constructed')
    if train['scheduler'] == 'OneCycleLR':
        scheduler.update(steps_per_epoch=steps, total_steps=steps*train['epochs'], step_cadence='once per outer batch')
    else:
        scheduler.update(T_max=train['epochs'], total_steps=train['epochs'], step_cadence='once per epoch')
    if task == 'darcy':
        scheduler['fixed500_release_caveat'] = 'official_release/original Transolver use fixed500; all records here use epochs500; do not silently change epochs'
    values['training']['resolved_scheduler'] = scheduler
    values['data']['planned_ntrain'] = nt
    values['data']['planned_ntest'] = 80 if task == 'plasticity' else 200
    values['data']['planned_points_per_model_call'] = NPOINTS[task]
    values['data']['remote_data_root'] = DATA_ROOT
    if task == 'plasticity':
        values['data']['current_entry_time_query_order'] = 'random_collate_fn uses torch.randperm(20); model called20 times; optimizer20 times; scheduler once per outer batch'
    if task == 'ns':
        values['data']['current_entry_rollout'] = 'input10 frames; output1 per call; 10 teacher-forced train calls then one backward/step; eval10 prediction-fed calls'
    return {k:values[k] for k in ('training','objective','evaluation','data','runtime')}


def provenance(task):
    return dict(repository_head=SNAPSHOT['head'], remote=SNAPSHOT['remote'], current_dirty_checkout_is_truth=True,
                source_manifest='source_snapshot.json', task_entry=f'PDE-Solving-StandardBenchmark/exp_{FILES[task]}.py',
                records_format='cdlno/experiment.py config.json fields plus loop_run_manifest fields; offline catalogue, not a checkpoint schema',
                output_recording_source='tran_evaluate/linearno_loop/recording.py',
                rank_source=ORIGINAL[task][1])


def store_record(task, profile, family, topology_label, depth, rank_mult, seed, record):
    rank = record['architecture']['resolved_rank']
    mode = record['architecture']['residual_mode']
    group = 'custom_topology_control' if topology_label == 'p0_c2_r3_s1' else 'depth_extension' if depth != 8 else 'base_depth8'
    path = Path('experiments') / task / profile / group / f'D{depth}' / f'M{rank}_x{rank_mult}' / family / topology_label / mode / f'seed{seed}' / 'config.json'
    run_id = f'{task}__{profile}__{family}__{topology_label}__{mode}__M{rank}__seed{seed}'
    record.update(schema_version=1, record_kind='offline_planned_experiment', experiment_id=run_id,
                  task=task, model=record['model_spec']['class_path'], family=family, protocol_profile=profile,
                  seed=seed, run_directory=f"output/planned_standard/{record.get('production_directory_id',run_id)}",
                  parameter_count_status='measured_cpu_model_construction',
                  environment=ENV, provenance=provenance(task),
                  experiment_status=dict(config='RESOLVED', construction='PASS', parameter_count='MEASURED_AND_FORMULA_CHECKED',
                     forward='NOT RUN', backward='NOT RUN', train='NOT RUN', eval='NOT RUN', checkpoint='NOT RUN', dataset='NOT READ', gpu='NOT USED'),
                  architecture_extension='loop_linearno_v1' if family=='linearno_loop' else None,
                  selection_policy='predeclared local seeds0/1/2; final checkpoint; report each and mean/std; no test-best seed/checkpoint',
                  optional_monitor='not enabled or changed by this inventory')
    record['record_sha256'] = digest(record)
    write(path, record)
    a = record['architecture'];p = record['parameters'];tr = record['hparams']['training']
    ROWS.append(dict(task=task, profile=profile, family=family, group=group, topology=topology_label,
       mode=mode, seed=seed, hidden=a['hidden'], heads=a['heads'], head_dim=a['head_dim'], variant=a['variant'],
       base_rank=a['base_rank'], rank_multiplier=rank_mult, actual_M=rank, unique_depth=a['unique_depth'],
       executed_depth=a['executed_depth'], total_parameters=p['total'], trainable_parameters=p['trainable'],
       backbone_parameters=p['backbone'], router_parameters=p['router'], batch_size=tr['batch_size'],
       epochs=tr['epochs'], lr=tr['lr'], mlp_ratio=record['model_spec']['constructor_kwargs']['mlp_ratio'],
       unified_pos=record['model_spec']['constructor_kwargs']['unified_pos'], ref=record['model_spec']['constructor_kwargs']['ref'],
       matrix_macs_B1=record['compute']['matrix_macs_B1'], config=path.as_posix()))


def cost(expected, task):
    return dict(B=1, N=NPOINTS[task], scope='one model forward call; not an NS rollout/Plasticity batch/epoch',
                kind='analytic_matrix_operations_only_not_timed', matrix_macs_B1=expected['executed_matrix_macs'],
                matrix_flops_B1=2*expected['executed_matrix_macs'],
                router_contraction_mac_equivalents=expected.get('router_contraction_mac_equivalents',0),
                excluded='softmax, RMSNorm/LayerNorm, GELU/SiLU, bias, residual arithmetic, distance/sin/cos, memory/layout and other scalar ops',
                rank_doubling_is_compute_matched=False, measured_latency=None)


def generate_loop(task, profile, depth, preset, rank_mult, seed, mode, *, custom=False):
    options = dict(topology_preset='custom',prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1) if custom else topology(depth,preset)
    options.update(residual_mode=mode,rank_multiplier=rank_mult)
    c=resolve_config(task,profile,options=options,profile_overrides={'runtime.seed':seed});validate_config(c)
    initial_rng=torch.get_rng_state().clone();model=build_from_config(c)
    assert torch.equal(initial_rng,torch.get_rng_state()), 'construction leaked CPU RNG'
    expected=analytic(c);params,state=measure(model,expected,loop=True)
    s=c['loop_spec'];actual_blocks=list(model.loop.prefix)+list(model.loop.core)+list(model.loop.suffix)
    assert len(actual_blocks)==s['unique_depth'] and len({id(x) for x in actual_blocks})==len(actual_blocks)
    assert sum(int(x.block.last_layer) for x in actual_blocks)==1
    assert all(not x.block.last_layer for x in model.loop.core)
    assert not any('round2' in k for k in model.state_dict())
    assert params['router']==s['attnres']['router_parameter_count']
    pair=(task,profile,preset if not custom else 'custom',depth,rank_mult,seed)
    common=(state['public_backbone_initial_sha256'],c['fair_comparison']['dataloader_generators'])
    if pair in PAIRS:assert PAIRS[pair]==common, 'residual-mode backbone/loader-seed mismatch'
    else:PAIRS[pair]=common
    if depth==8 and not custom and profile==PROFILES[0]:
        old=HISTORICAL[(task,preset,mode,rank_mult)]
        assert old['measured_parameter_parts']==params['by_component']
        CHECKS['historical_LL9R_count_matches']+=1
    label=f"p{s['prefix_blocks']}_c{s['recurrent_core_blocks']}_r{s['loop_repeats']}_s{s['suffix_blocks']}"
    hparams=protocol(task,c['profile_spec']['values'])
    a=dict(family='linearno_loop',variant=s['variant'],hidden=s['hidden'],heads=s['heads'],head_dim=s['head_dim'],
           unique_depth=s['unique_depth'],executed_depth=s['executed_depth'],base_rank=s['base_rank'],
           rank_multiplier=rank_mult,resolved_rank=s['resolved_rank'],residual_mode=mode,loop_spec=s,
           depth_extension_policy='R=2 with fixed P/S and increased C; local custom design' if depth not in (7,8) else 'named preset' if not custom else 'documented custom configuration control',
           topology_lineage=preset if not custom else 'custom_P0C2R3S1',current_stage_forward_evidence=None)
    expected_schedule=schedule(s)
    assert len([e for e in expected_schedule if e.endswith('.operator')])==s['executed_depth']
    assert len([e for e in expected_schedule if e.endswith('.head')])==1
    manifest=dict(family='linearno_loop',config_hash=c['config_hash'],unique_depth=s['unique_depth'],executed_depth=s['executed_depth'],
       expected_call_schedule=expected_schedule,actual_call_schedule=None,observation='construction_only; forward_NOT_RUN',
       initialization_seed=seed,public_backbone_initial_sha256=state['public_backbone_initial_sha256'],
       dataloader_generator_seeds=c['fair_comparison']['dataloader_generators'],
       router_initialization='query=zeros; norm=ones; does not consume RNG',fair_comparison=c['fair_comparison'])
    store_record(task,profile,'linearno_loop',label,s['executed_depth'],rank_mult,seed,
      dict(model_spec=c['model_spec'],resolved_arguments=c['request'],resolved_config=c,hparams=hparams,architecture=a,
           parameters=params,state_dict=state,loop_run_manifest=manifest,compute=cost(expected,task),
           production_directory_id=run_directory_id(c),construction_method='actual unmodified LoopedStandardModel on CPU'))
    del model


def generate_baseline(task, profile, depth, rank_mult, seed, *, transolver=False):
    base=ORIGINAL[task][0]['slice_num'];rank=base*rank_mult
    c=pure_config(task,profile if not transolver else 'transolver_matched',
        explicit={'model.layers':depth,'model.linearno_rank':rank,'runtime.seed':seed},contract=contract(task))
    validate_resolved(c);kw=constructor_kwargs(c)
    family='transolver' if transolver else 'linearno'
    args=None
    if transolver:
        args=deepcopy(ORIGINAL[task][0]);args.update(n_layers=depth,slice_num=rank)
        for k,src in (('n_hidden','n_hidden'),('n_head','n_heads'),('mlp_ratio','mlp_ratio'),('dropout','dropout'),('ref','ref')):kw[k]=args[src]
        kw['unified_pos']=bool(args['unified_pos']);kw['slice_num']=kw.pop('linearno_rank');kw.pop('linearno_variant')
        if task=='elasticity':kw.pop('H');kw.pop('W')
        cls=trans_irregular if task=='elasticity' else TransolverCPUPositionReference if kw['unified_pos'] else trans_structured
        class_path='model.'+args['model']+'.Model'
        expected=transolver_formula(kw,task)
        method='unmodified Transolver learned modules; get_grid only evaluated on CPU via local subclass' if kw['unified_pos'] else 'actual unmodified Transolver Model on CPU'
    else:
        cls=native.Model;class_path='model.LinearNO.Model'
        expected=baseline_linear_formula(task,profile,depth,rank,seed)
        method='actual unmodified pure LinearNO Model on CPU'
    initial_rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed);model=cls(**kw)
    assert torch.equal(initial_rng,torch.get_rng_state())
    params,state=measure(model,expected,loop=False)
    assert len(model.blocks)==depth and sum(int(b.last_layer) for b in model.blocks)==1
    a=dict(family=family,variant=('Physics_Attention_Irregular_Mesh' if task=='elasticity' else 'Physics_Attention_Structured_Mesh_2D') if transolver else kw['linearno_variant'],
       hidden=kw['n_hidden'],heads=kw['n_head'],head_dim=kw['n_hidden']//kw['n_head'],unique_depth=depth,executed_depth=depth,
       base_rank=base,rank_multiplier=rank_mult,resolved_rank=rank,residual_mode='native_residual',parameter_sharing='no cross-depth weight sharing')
    hparams=protocol(task,c['values'],trans_args=args)
    hparams['runtime']['seed']=seed
    record=dict(model_spec=dict(class_path=class_path,constructor_kwargs=kw),resolved_arguments=args if transolver else c['values'],
       hparams=hparams,architecture=a,parameters=params,state_dict=state,compute=cost(expected,task),construction_method=method,
       initialization_seed=seed,seed_scope='CPU initialization only; original task has no seed CLI' if transolver else 'resolved LinearNO runtime seed',
       comparison_notes='native Transolver reference differs from LinearNO in operator and task defaults; transolver_matched is not automatic proof of identical loss/normalization/seeding' if transolver else 'independent full blocks; same profile as loop, not parameter-matched')
    if not transolver:record['resolved_config']=c
    else:
        record['original_launcher']=ORIGINAL[task][1]
        record['protocol_source']='actual Transolver script+parser AST+entry; LinearNO profile data fields used only as task documentation'
    store_record(task,'transolver_original' if transolver else profile,family,f'independent_L{depth}',depth,rank_mult,seed,record)
    del model


def main():
    if (OUT/'experiments').exists():raise SystemExit('Refusing to overwrite any previous experiment records')
    torch.set_num_threads(1)
    for task in TASKS:
        for profile in PROFILES:
            depths=(8,16,24,32) if task in STATIC and profile==PROFILES[0] else (8,)
            for depth in depths:
                for mult in (1,2):
                    for seed in SEEDS:
                        generate_baseline(task,profile,depth,mult,seed)
                        for preset in PRESETS:
                            for mode in MODES:generate_loop(task,profile,depth,preset,mult,seed,mode)
            if profile==PROFILES[0]:
                for mult in (1,2):
                    for seed in SEEDS:
                        for mode in MODES:generate_loop(task,profile,7,'custom',mult,seed,mode,custom=True)
            print(task,profile,'records_so_far',len(ROWS),flush=True)
        for depth in ((8,16,24,32) if task in STATIC else (8,)):
            for mult in (1,2):
                for seed in SEEDS:generate_baseline(task,'transolver_original',depth,mult,seed,transolver=True)
        print(task,'complete',len(ROWS),flush=True)
    expected_count=1476
    assert len(ROWS)==expected_count,(len(ROWS),expected_count)
    assert len({r['config'] for r in ROWS})==expected_count
    write('index.json',dict(schema_version=1,total_experiments=len(ROWS),paired_seeds=list(SEEDS),
          counts_by_task=dict(Counter(r['task'] for r in ROWS)),counts_by_family=dict(Counter(r['family'] for r in ROWS)),
          counts_by_profile=dict(Counter(r['profile'] for r in ROWS)),experiments=ROWS))
    with (OUT/'summary.csv').open('x',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(ROWS[0]));writer.writeheader();writer.writerows(ROWS)
    # One architectural row (seed-independent counts) for compact comparison.
    with (OUT/'architecture_summary.csv').open('x',newline='',encoding='utf-8-sig') as handle:
        fields=[k for k in ROWS[0] if k not in ('seed','config')]
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in ROWS if r['seed']==0)
    write('validation.json',dict(status='PASS',experiment_files=len(ROWS),measured_model_constructions=len(ROWS),
        formula_equal_measurement=len(ROWS),paired_mode_groups_verified=len(PAIRS),
        historical_LL9R_parameter_checks=CHECKS['historical_LL9R_count_matches'],
        pure_and_transolver_parameter_formulas='independent integer formulas equal actual module counts',
        cpu_rng_preserved_for_all_constructions=True,expected_schedule_checked=True,
        actual_forward_schedule='NOT RUN',training='NOT RUN',real_data='NOT READ',gpu='NOT USED',
        depth_interpretation='executed D8/16/24/32; keep R2 and preset P/S, grow C; extras are local custom designs',
        optional_profile_controls='official_release and transolver_matched at D8 only; custom R3 at paper profile only'))
    print('COMPLETE',len(ROWS),'records',flush=True)


SNAPSHOT=json.loads((OUT/'source_snapshot.json').read_text())
ENV=dict(python=platform.python_version(),torch=str(torch.__version__),construction_device='cpu',
         dtype='torch.float32',torch_threads=1,cuda_used=False,remote_checkout=REMOTE,
         remote_execution_or_training_verified=False)
HISTORICAL={(r['task'],r['preset'],r['mode'],r['rank_multiplier']):r
            for r in json.loads((ROOT/'docs/loop_linearno_audit/ll9r/counts.json').read_text())['rows']}
ROWS=[];PAIRS={};CHECKS=Counter()

if __name__=='__main__':main()
