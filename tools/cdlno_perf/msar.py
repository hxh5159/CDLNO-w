"""MSAR-only cost/accounting adapter for the existing synthetic perf tools.

No task entry imports, model edits, trained-checkpoint loads or data access.
Dense MAC formulas are independently checked against live Linear/SDPA shapes.
Scalar work/explicit coverage storage is reported separately, never omitted
under a claim of exact total FLOPs or confused with allocator peak memory.
"""
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch import nn

from cdlno.msar_lno.config import MSARTrainingConfig
from cdlno.msar_lno.profiles import resolve_profile
from cdlno.msar_lno.standard import StaticModel
from cdlno.msar_lno.temporal import TemporalModel
from cdlno.msar_lno.industrial import CarModel, AirfRANSModel
from cdlno.msar_lno.objective import training_forward, training_objective
from cdlno.msar_lno.modules import (LearnedQueryDown, QueryAlignedUpCross,
    LatentFFNSAFFNBlock, PairwiseAttnResFusion, CoverageFloorLoss)
from tools.cdlno_perf.costs import audit, Operations, storage_key, tensors
from tools.cdlno_perf.models import Case, TASKS

MODELS = ('msar_light_off', 'msar_light_floor', 'msar_full_off', 'msar_full_floor')


def build_msar(case, name, device='cpu', seed=20260917):
    if name not in MODELS:
        raise ValueError('unknown MSAR performance variant')
    case.validate()
    _, profile, mode = name.split('_')
    torch.manual_seed(seed)
    cfg = resolve_profile(profile)  # Never reuse the other families' d/M defaults.
    objective = MSARTrainingConfig(coverage_mode=mode)  # kappa .2 / weight .01 unchanged
    kwargs = dict(config=cfg, training_config=objective)
    if case.task == 'shapenet-car':
        model = CarModel(**kwargs)
    elif case.task == 'airfrans':
        model = AirfRANSModel(**kwargs)
    else:
        cls = TemporalModel if case.task in ('ns','plasticity') else StaticModel
        model = cls(task_name=case.task, H=case.grid[0] if case.grid else None,
                    W=case.grid[1] if case.grid else None, **kwargs)
    return model.to(device)


def component(name):
    if name.startswith(('preprocess.', 'time_fc.')) or name == 'placeholder': return 'stem_time'
    if name.startswith(('core.output.', 'core.output_norm.')): return 'head'
    if name.startswith('core.downs.'): return 'down4'
    if name.startswith(('core.encoders.', 'core.decoders.')): return 'latent12'
    if name.startswith(('core.ups.', 'core.final_up.')): return 'up4'
    if name.startswith('core.fusions.'): return 'pair3'
    raise ValueError('unaccounted MSAR parameter/operator: ' + name)


def parameters(model):
    rows = list(model.named_parameters(remove_duplicate=False))
    assert len(rows) == len({id(p) for _,p in rows})
    assert len(rows) == len({p.untyped_storage().data_ptr() for _,p in rows})
    groups = Counter(); detail = {}
    for name,p in rows:
        groups[component(name)] += p.numel()
        detail[name] = dict(shape=list(p.shape), count=p.numel(), requires_grad=p.requires_grad)
    c = model.config; d = c.d
    expected_down = sum(m*d + 3*d*d + d + 2*(d//h) for m,h in zip(c.num_latents,c.heads))
    expected_latent = sum((e+de)*(12*d*d + 10*d + 2*(d//h))
                          for e,de,h in zip(c.encoder_depths,c.decoder_depths,c.heads))
    expected_up = sum(4*d*d + d + 2*(d//h) for h in (*c.heads[:3],c.heads[0]))
    assert groups['down4'] == expected_down
    assert groups['latent12'] == expected_latent
    assert groups['up4'] == expected_up
    assert groups['pair3'] == 3*d
    assert not list(CoverageFloorLoss().parameters())
    assert sum(groups.values()) == sum(p.numel() for p in model.parameters())
    return dict(total=sum(groups.values()), trainable=sum(p.numel() for _,p in rows if p.requires_grad),
        by_component=dict(groups), key_shapes_counts=detail, coverage_parameters=0,
        formulas_verified=dict(down=expected_down,latent=expected_latent,up=expected_up,pair=3*d),
        independent_objects_and_storage=True)


def matrix_formula(model, b, n):
    """All dense matrices, including both FFNs and N-scale projections.

    Direct learned Down Q has no Wq projection. Pair score dot/raw weighting,
    norms/softmax/GELU/bias/residuals/coordinates are separate scalar work.
    """
    c=model.config;d=c.d;m=c.num_latents
    ledger=[]
    def add(name,projections,attention=0):
        ledger.append(dict(name=name,projection_and_ffn_mac=projections,qk_av_mac=attention,
                           matrix_mac=projections+attention))
    for name,layer in model.named_modules():
        if isinstance(layer,nn.Linear) and name.startswith(('preprocess.','time_fc.')):
            add(name,b*(1 if name.startswith('time_fc.') else n)*layer.in_features*layer.out_features)
    sources=(n,*m[:3])
    for i,(src,dest) in enumerate(zip(sources,m)):
        add(f'core.downs.{i}',b*(2*src+dest)*d*d,2*b*src*dest*d)
    for group,depths in (('encoders',c.encoder_depths),('decoders',c.decoder_depths)):
        for i,(tokens,depth) in enumerate(zip(m,depths)):
            for j in range(depth):
                add(f'core.{group}.{i}.{j}',12*b*tokens*d*d,2*b*tokens*tokens*d)
    for i in range(3):
        add(f'core.ups.{i}',2*b*(m[i]+m[i+1])*d*d,2*b*m[i]*m[i+1]*d)
    add('core.final_up',2*b*(n+m[0])*d*d,2*b*n*m[0]*d)
    add('core.output',b*n*d*model.core.output.out_features)
    groups=Counter()
    for row in ledger:groups[component(row['name']+'.weight')]+=row['matrix_mac']
    return dict(matrix_macs=sum(row['matrix_mac'] for row in ledger),
        matrix_flops_2_per_mac=2*sum(row['matrix_mac'] for row in ledger),
        by_component=dict(groups),by_module=ledger,
        convention='Forward dense matrices only, 1 MAC=2 FLOPs. All other work separately inventoried; NOT all-operation total FLOPs.')


def scalar_and_coverage_ledger(model,b,n):
    c=model.config;d=c.d;m=c.num_latents;sources=(n,*m[:3]);levels=[]
    for i,(src,dest,h) in enumerate(zip(sources,m,c.heads)):
        elems=b*h*dest*src
        levels.append(dict(level=i+1,shape=[b,h,dest,src],fp32_attention_payload_bytes=4*elems,
            head_mean_input_elements=elems,latent_mean_input_elements=b*dest*src,
            source_floor_elements=b*src,source_sum_batch_mean=True))
    r=b*sum(m[:3]);slots=sum((en+de)*mi for mi,en,de in zip(m,c.encoder_depths,c.decoder_depths))
    return dict(coverage_training_only=dict(levels=levels,
            summed_fp32_A_payload_bytes=sum(row['fp32_attention_payload_bytes'] for row in levels),
            summed_logits_plus_A_payload_bytes=2*sum(row['fp32_attention_payload_bytes'] for row in levels),
            note='Payload inventory, NOT peak/total upper bound: masked copies, backward, Up/SA activations and optimizer also allocate. See measured train peak. No inference coverage.',
            reduction='source SUM, batch MEAN, four-level MEAN, weight once; no additional dense QK/AV matrix MAC versus off'),
        pair3=dict(parameter_count=3*d,score_dot_mac_equivalents=2*r*d,
            raw_value_weight_multiplications=2*r*d,source_sum_additions=r*d,factor2_multiplications=r*d,
            parameter_free_RMS_vectors=2*r,vector_width=d,source_softmax_rows=r,source_axis=2),
        latent_ffns=dict(two_per_block=True,gelu_elements=4*b*slots*d,ffn_residual_additions=2*b*slots*d),
        latent_sa_residual_additions=b*slots*d,
        other='All bias additions, affine RMS/LayerNorm calls and executed scalar/reduction/copy ops recorded by live audit; inline Pair RMS covered here. SDPA softmax shape inventory also retained.')


def live_audit(model,args,target,context,b,n):
    """Reuse existing dispatcher/Linear/SDPA cost hooks without changing them."""
    counts=Counter();handles=[]
    types=(LearnedQueryDown,QueryAlignedUpCross,LatentFFNSAFFNBlock,PairwiseAttnResFusion)
    for layer in model.modules():
        if isinstance(layer,types):
            handles.append(layer.register_forward_hook(lambda m,a,o:counts.update([type(m).__name__])))
    try:live=audit(model,args,target,context)
    finally:
        for handle in handles:handle.remove()
    expected=matrix_formula(model,b,n);actual=Counter()
    for name,mac in live['linear_conv_macs_by_module'].items():actual[component(name+'.weight')]+=mac
    for row in live['sdpa']:actual[component(row['path']+'.weight')]+=row['macs']
    assert dict(actual)==expected['by_component'],(actual,expected['by_component'])
    assert live['matrix_macs']==expected['matrix_macs']
    assert counts==Counter(LearnedQueryDown=4,QueryAlignedUpCross=4,LatentFFNSAFFNBlock=12,PairwiseAttnResFusion=3)
    assert live['finite_output'] and not live['parameters']['missing_grad_names'] and not live['parameters']['nonfinite_grad_names']
    # Old count/category labels describe CDLNO; don't mislabel MSAR as its front.
    return dict(parameters=parameters(model),matrix=expected,live_module_mac=live['linear_conv_macs_by_module'],
        live_sdpa=live['sdpa'],verified_live=True,counts=dict(counts),bias_adds=live['bias_adds_by_component'],
        norm_work=live['norm_work'],operations=live['operations'],temporary_materializations=live['temporary_materializations'],
        saved_storage=live['storage'],separate_nonmatrix=scalar_and_coverage_ledger(model,b,n))


def synthetic_loss(model,args,target):
    f=training_forward(model,*args)
    return training_objective((f.prediction.float()-target.float()).square().mean(),f).total


def coverage_path_audit(model,args,target,context):
    """Actual floor/off graph storage and A shapes, outside timed regions."""
    model.train();model.zero_grad(set_to_none=True)
    attention=[];saved={};handles=[];scope=[]
    excluded={storage_key(t) for t in [*model.parameters(),*model.buffers(),*tensors(args)]}
    def pack(t):
        key=storage_key(t)
        if key not in excluded:saved[key]=key[2]
        return t
    def observe(m,a,out):
        if isinstance(out,tuple) and out[1] is not None:
            t=out[1];attention.append(dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size()))
    for down in model.core.downs:handles.append(down.register_forward_hook(observe))
    dispatch=Operations(scope)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t),dispatch,context():
            f=training_forward(model,*args)
            obj=training_objective((f.prediction.float()-target.float()).square().mean(),f)
        obj.total.backward()
        values={key:float(value) for key,value in obj.log_values().items()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert len(attention)==(4 if model.training_config.coverage_enabled else 0)
        return dict(objective=values,config=model.training_config.to_dict(),attention=attention,
            saved_activation_unique_backing_bytes=sum(saved.values()),operations=dict(dispatch.ops),
            note='Unique autograd storage excluding params/buffers/inputs; NOT allocator peak; no graph retained.')
    finally:
        for handle in handles:handle.remove()
        model.zero_grad(set_to_none=True)


def task_inventory():
    """Real models for all task/profile parameter counts; formula cost at real N/B.

    No large-grid forward/backward here. Representative live equality elsewhere.
    """
    rows=[]
    for task in TASKS:
        case=Case.preset(task,'task')
        for profile in ('light','full'):
            model=build_msar(case,f'msar_{profile}_off')
            rows.append(dict(task=task,profile=profile,B=case.B,N=case.N,architecture=model.config.to_dict(),
                adapter=model.adapter_architecture(),parameters=parameters(model),
                forward_matrix=matrix_formula(model,case.B,case.N),separate=scalar_and_coverage_ledger(model,case.B,case.N),
                execution='Actual model construction/parameters; analytical full-N cost, NOT full-N training or timing'))
            del model
    return rows
