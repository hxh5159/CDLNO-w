"""Pure V3 study preview. This is not a production launcher or parser."""
from .config import resolve_config, run_directory_id
from .contracts import (TASKS, RESIDUAL_MODES, ABLATIONS, ARCHITECTURE_SELECTOR,
                        ADAPTER_MODES, CLI_CONTRACT, V3SchemaError, integer)
from .costs import analytic_cost

PAIRED_SEEDS=(0,1,2)


def preview(config):
    opts=config['request']['options'];task=config['request']['task']
    argv=['bash',f'tran_evaluate/linearno_loop/{task}.sh','train','--linearno-loop','1',
          '--linearno-profile',config['request']['profile'],'--seed',str(config['profile_spec']['values']['runtime']['seed'])]
    for flag,spec in CLI_CONTRACT.items():
        if spec['field'] in opts:
            value=opts[spec['field']]
            argv += [flag,str(int(value)) if type(value) is bool else str(value)]
    return dict(status='SCHEMA_PREVIEW_NOT_RUNNABLE_LAA1',argv=argv,
                run_id=run_directory_id(config),constructor='NOT IMPLEMENTED',training='NOT RUN')


def iter_configurations(seeds=PAIRED_SEEDS):
    if type(seeds) not in (tuple,list) or not seeds or len(set(seeds))!=len(seeds):
        raise V3SchemaError('paired seeds must be a nonempty distinct list/tuple')
    for seed in seeds:integer(seed,'seed',0)
    for task in TASKS:
        for profile in ('matched_v1','efficient_v1'):
            for depth in (12,20,28,60):
                for residual in RESIDUAL_MODES:
                    for latent,adapter in ABLATIONS:
                        for seed in seeds:
                            c=resolve_config(task,options=dict(architecture=ARCHITECTURE_SELECTOR,cost_profile=profile,
                                executed_depth=depth,residual_mode=residual,latent_enabled=latent,
                                adapter_mode=ADAPTER_MODES[1] if adapter else 'none'),profile_overrides={'runtime.seed':seed})
                            yield dict(config=c,preview=preview(c),cost=analytic_cost(c))


def custom_cases():
    good=dict(architecture=ARCHITECTURE_SELECTOR,cost_profile='custom',hidden_width=200,
              latent_width=96,actual_M=31,heads=8,topology_preset='custom',
              prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1,adapter_mode='none')
    positives=[]
    for residual in RESIDUAL_MODES:
        c=resolve_config('car',options={**good,'residual_mode':residual})
        positives.append(dict(config=c,preview=preview(c),cost=analytic_cost(c)))
    negatives=[]
    for label,changes in [('H_not_divisible',{'hidden_width':201}),('missing_width',{'latent_width':None}),
        ('R3_adapter_on',{'adapter_mode':ADAPTER_MODES[1]}),('C0',{'recurrent_core_blocks':0}),
        ('S0',{'suffix_blocks':0}),('R0',{'loop_repeats':0}),('M_bool',{'actual_M':True}),
        ('alpha_string',{'adapter_alpha':'4'}),('unknown',{'unknown':0})]:
        opts={**good,**changes}
        try:resolve_config('car',options=opts)
        except ValueError as e:negatives.append(dict(label=label,request=opts,status='EXPECTED_REJECTION',message=str(e)))
        else:raise AssertionError('invalid custom accepted: '+label)
    return dict(valid=positives,invalid=negatives)


def main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    # Exclusive output creation. Re-running cannot overwrite a prior matrix.
    count=0
    with (args.output_dir/'configuration-matrix.jsonl').open('x') as stream:
        for row in iter_configurations():
            stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n');count+=1
    custom=custom_cases()
    with (args.output_dir/'custom-cases.json').open('x') as stream:json.dump(custom,stream,indent=2)
    summary=dict(status='PREVIEW_ONLY',count=count,paired_seeds=list(PAIRED_SEEDS),
        axes=dict(tasks=8,profiles=2,depths=4,residuals=3,ablations=4,seeds=3),
        configuration_file='configuration-matrix.jsonl',custom_valid=len(custom['valid']),custom_invalid=len(custom['invalid']),
        no_torch=True,no_models=True,no_model_construction=True,no_training=True,no_production_parser=True,
        aggregation='all paired seeds and mean/std; final checkpoint; never select by test',
        cost_scope='on/on r4 alpha4 SR approximate matrix match only; all others separately recomputed')
    with (args.output_dir/'matrix-summary.json').open('x') as stream:json.dump(summary,stream,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
