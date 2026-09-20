"""Configuration-only study declaration; preview argv is not runnable in LL1."""
from .config import resolve_config, run_directory_id
from .contracts import PRESETS, RESIDUAL_MODES, TOPOLOGY_FIELDS, CLI_CONTRACT
from cdlno.linearno.profiles import TASKS

PAIRED_SEEDS = (0, 1, 2)  # Local predeclared seeds, not the original authors' seeds.


def preview(config):
    s = config['loop_spec']; seed = config['profile_spec']['values']['runtime']['seed']
    options = config['request']['options']
    args = ['bash', f"tran_evaluate/linearno_loop/{s['task']}.sh", 'train',
            '--linearno-loop','1','--linearno-loop-topology',s['topology_preset'],
            '--linearno-loop-residual-mode',s['residual_mode'],
            '--linearno-profile',s['profile'],'--seed',str(seed)]
    for flag, spec in CLI_CONTRACT.items():
        field = spec['field']
        if field in options and field in (*TOPOLOGY_FIELDS, 'rank_multiplier', 'linearno_rank'):
            args.extend([flag,str(options[field])])
    if s['rank_policy']=='profile_multiplier' and 'rank_multiplier' not in options:
        args.extend(['--linearno-loop-rank-multiplier',str(s['rank_multiplier'])])
    args.extend(['--gpu','0','--experiment-dir','output/'+run_directory_id(config)+'__RUN_ID'])
    return dict(status='PLANNED_NOT_RUNNABLE_LL1',argv=args)


def configuration_matrix():
    rows = []
    for group, multiplier in (('main',2),('rank_x1_control',1)):
        for task in TASKS:
            for topology in PRESETS:
                for mode in RESIDUAL_MODES:
                    for seed in PAIRED_SEEDS:
                        c=resolve_config(task, options=dict(topology_preset=topology,residual_mode=mode,
                            rank_multiplier=multiplier),profile_overrides={'runtime.seed':seed})
                        rows.append(dict(group=group,config=c,preview=preview(c),
                            future_acceptance=dict(construction='NOT RUN',forward='NOT RUN',backward='NOT RUN',
                                strict_roundtrip='NOT RUN',native_resume_eval='NOT RUN')))
    custom=[]
    for task in TASKS:
        for mode in RESIDUAL_MODES:
            c=resolve_config(task, options=dict(topology_preset='custom',prefix_blocks=0,
                recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1,residual_mode=mode))
            custom.append(dict(config=c,preview=preview(c),status='CONFIG_ONLY'))
    return dict(schema_version=1,paired_seeds=list(PAIRED_SEEDS),author_seeds=False,
        profile='paper_table8_on_release_model',main_count=144,rank_x1_count=144,
        custom_count=len(custom),runs=rows,custom_controls=custom,
        no_real_data=True,no_model_construction=True,no_launcher_integration=True,
        additional_required_controls=['pure_8_blocks_base_M','pure_8_blocks_2M'],
        aggregation='report each paired local seed and mean/std; final checkpoint; never select by test',
        budget='same task/profile training and evaluation; not FLOP-matched')
