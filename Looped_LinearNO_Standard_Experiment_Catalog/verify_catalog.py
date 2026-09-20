#!/usr/bin/env python3
"""Read-only verification, apart from a new report inside this delivery folder."""
import ast
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
scope=runpy.run_path(str(OUT/'generate_catalog.py'),run_name='catalog_verification_import')
torch=scope['torch']
torch.set_num_threads(1)


def main():
    rows=json.loads((OUT/'index.json').read_text())['experiments']
    found=set(p.relative_to(OUT).as_posix() for p in (OUT/'experiments').rglob('config.json'))
    assert found=={r['config'] for r in rows} and len(rows)==1476
    signatures=set();runs=set();pairs={};depths={};schemas=Counter()
    for row in rows:
        c=json.loads((OUT/row['config']).read_text())
        saved=c.pop('record_sha256');assert saved==scope['digest'](c)
        p=c['parameters'];a=c['architecture']
        assert p['total']==sum(v['numel'] for v in p['inventory'])==sum(p['by_component'].values())
        assert p['trainable']==sum(v['numel'] for v in p['inventory'] if v['requires_grad'])
        assert p['total']==row['total_parameters'] and p['router']==row['router_parameters']
        assert a['resolved_rank']==row['actual_M']==a['base_rank']*row['rank_multiplier']
        assert a['base_rank']==scope['ORIGINAL'][row['task']][0]['slice_num']
        assert c['seed']==row['seed'] and c['experiment_status']['train']=='NOT RUN'
        assert c['experiment_id'] not in signatures;signatures.add(c['experiment_id'])
        assert c['run_directory'] not in runs;runs.add(c['run_directory'])
        if c['family']=='linearno_loop':
            scope['validate_config'](c['resolved_config']);schemas['loop']+=1
            s=a['loop_spec'];P,C,R,S=(s[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'))
            assert a['unique_depth']==P+C+S and a['executed_depth']==P+C*R+S
            expected_receivers=2*C*R+1 if a['residual_mode']=='rb_attnres' else R if a['residual_mode']=='lb_attnres_1_over_r' else 0
            assert p['router']==2*a['hidden']*expected_receivers
            schedule=c['loop_run_manifest']['expected_call_schedule']
            assert sum(e.endswith('.operator') for e in schedule)==a['executed_depth']
            assert sum(e.endswith('.head') for e in schedule)==1
            pair=(row['task'],row['profile'],row['topology'],row['actual_M'],row['seed'])
            value=(c['state_dict']['public_backbone_initial_sha256'],c['loop_run_manifest']['dataloader_generator_seeds'])
            if pair in pairs:assert pairs[pair]==value
            else:pairs[pair]=value
            assert c['production_directory_id'] in c['run_directory']
        elif c['family']=='linearno':
            scope['validate_resolved'](c['resolved_config']);schemas['pure']+=1
        else:schemas['transolver_from_native_parser']+=1
        if row['task'] in scope['STATIC'] and row['profile'] in ('paper_table8_on_release_model','transolver_original') and row['group']!='custom_topology_control':
            key=(row['task'],row['family'],row['mode'],row['rank_multiplier'],row['seed'])
            depths.setdefault(key,set()).add(row['executed_depth'])
    assert all(v=={8,16,24,32} for v in depths.values())
    # Each seeded architecture appears for exactly the predeclared three seeds.
    seed_sets={}
    for r in rows:
        key=(r['task'],r['profile'],r['family'],r['topology'],r['mode'],r['actual_M'])
        seed_sets.setdefault(key,set()).add(r['seed'])
    assert len(seed_sets)==492 and all(s=={0,1,2} for s in seed_sets.values())
    with (OUT/'summary.csv').open(encoding='utf-8-sig') as f:
        table=list(csv.DictReader(f))
    assert len(table)==1476 and {r['config'] for r in table}==found

    # Independent CPU reference: exact original get_grid source, only remove
    # device transfers. No global Tensor.cuda patch and no repository edits.
    original=scope['trans_structured']
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.get_grid)))
    removed=[]
    class RemoveCuda(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
                assert not node.args and not node.keywords
                removed.append(ast.unparse(node));return node.func.value
            return node
    tree=ast.fix_missing_locations(RemoveCuda().visit(tree));assert len(removed)==2
    import numpy as np
    namespace=dict(torch=torch,np=np)
    exec(compile(tree,'<original-grid-only-cuda-removed>','exec'),namespace)
    cpu_original=type('OriginalTransolverGridCPUReference',(original,),{'get_grid':namespace['get_grid']})
    comparisons=[]
    for task in ('darcy','ns'):
        row=next(r for r in rows if r['task']==task and r['family']=='transolver' and r['executed_depth']==8 and r['seed']==0 and r['rank_multiplier']==1)
        kw=json.loads((OUT/row['config']).read_text())['model_spec']['constructor_kwargs']
        models=[]
        rng=torch.get_rng_state().clone()
        for cls in (scope['TransolverCPUPositionReference'],cpu_original):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0);models.append(cls(**kw))
        assert torch.equal(rng,torch.get_rng_state())
        a,b=models
        assert scope['state_hash'](a.state_dict())==scope['state_hash'](b.state_dict())
        errors=[]
        for batch in (1,2):
            x,y=a.get_grid(batch),b.get_grid(batch)
            assert torch.equal(x,y)
            errors.append(dict(batch=batch,shape=list(x.shape),max_abs_error=(x-y).abs().max().item()))
        comparisons.append(dict(task=task,position_checks=errors,full_initial_state_identical=True,
                                source_cuda_calls_removed=2,learned_modules_modified=False))
        del models,a,b
    result=dict(status='PASS',records_verified=len(rows),architecture_configurations=len(seed_sets),
        unique_output_paths=len(runs),schema_validations=dict(schemas),paired_mode_groups_verified=len(pairs),
        depth_matrix_verified=True,seed_matrix_verified=True,all_record_hashes_verified=True,
        json_csv_agreement=True,transolver_CPU_reference_comparison=comparisons,
        scope='configuration, construction-derived counts and CPU initialization only; no dataset/forward/training/GPU')
    scope['write']('catalog_review.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
