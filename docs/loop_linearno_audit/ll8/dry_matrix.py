"""Real new-shell/old-launcher/public-parser previews. Never imports task data."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import contextlib,hashlib,io,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from tran_evaluate.linearno_loop import launch
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES
from linearno_loop.config import resolve_config,run_directory_id

ENV=dict(os.environ,CDLNO_REPO_ROOT=str(ROOT),CDLNO_PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def base(preset,mode):return ['--linearno-loop','1','--linearno-loop-topology',preset,'--linearno-loop-residual-mode',mode]
def shell(task,action,tokens,label):
    cmd=['bash',str(ROOT/f'tran_evaluate/linearno_loop/{task}.sh'),action,'--plan-json',*map(str,tokens)]
    r=subprocess.run(cmd,cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=45)
    if r.returncode:raise AssertionError(r.stderr+r.stdout)
    plan=json.loads(r.stdout);cfg=plan['config']
    assert plan['task']==task and plan['action']==action and cfg['family']=='linearno_loop'
    if action=='train':assert not Path(plan['run']).exists()
    return dict(label=label,command=cmd,plan=plan,passed=True)

def main():
    archive=[]
    for name in ('native-matrix.json','industrial-matrix.json'):
        archive+=json.loads((OUT.parent/'ll7'/name).read_text())['rows']
    jobs=[]
    # Copy the genuine LL7 epoch1 pair to an isolated transfer snapshot. The
    # original split runs are already complete and correctly reject resume.
    resume_runs={}
    with contextlib.nullcontext(tempfile.mkdtemp(prefix='loop ll8 preview ',dir='/home/hwz/CDLNO-artifacts')) as tmp:
        for task in launch.TASKS:
            for preset in PRESETS:
                for mode in RESIDUAL_MODES:
                    flags=base(preset,mode)+['--linearno-loop-rank-multiplier','2','--seed','17','--gpu','0']
                    cfg=resolve_config(task,options=dict(topology_preset=preset,residual_mode=mode,rank_multiplier=2),profile_overrides={'runtime.seed':17})
                    run=Path(tmp)/task/run_directory_id(cfg)
                    jobs.append((task,'train',flags+['--experiment-dir',str(run)],'formal_train_Mx2'))
                    saved=next(r for r in archive if r['task']==task and r['preset']==preset and r['mode']==mode)
                    existing=next((Path(saved['artifact'])/'split').iterdir())
                    resume=Path(tmp)/'epoch1_transfer'/task/preset/mode/existing.name
                    resume.mkdir(parents=True)
                    shutil.copyfile(existing/'architecture.json',resume/'architecture.json')
                    src=existing/'member_000' if task=='airfrans' else existing
                    dst=resume/'member_000' if task=='airfrans' else resume
                    dst.mkdir(exist_ok=True)
                    shutil.copyfile(src/'architecture.json',dst/'architecture.json')
                    manifest=src/'checkpoints/epoch_0001.json';pair=json.loads(manifest.read_text())
                    for key in ('checkpoint','weights','metadata'):
                        relative=pair[key]['path'];(dst/relative).parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(src/relative,dst/relative)
                    shutil.copyfile(manifest,dst/'checkpoints'/manifest.name)
                    (dst/'checkpoints/latest.json').write_text(json.dumps(dict(manifest=manifest.name,sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
                    resume_runs[(task,preset,mode)]=resume
                    for action,directory in (('eval',existing),('resume',resume)):
                        jobs.append((task,action,['--experiment-dir',str(directory),'--gpu','0'],'actual_LL7_archive_scaled_model'))
            for extra,label in ((['--linearno-loop-rank-multiplier','1'],'rank_x1'),
                (['--linearno-loop-prefix-blocks','0','--linearno-loop-core-blocks','2','--linearno-loop-repeats','3','--linearno-loop-suffix-blocks','1'],'custom'),
                (['--linearno-rank','64' if task in ('car','airfrans') else '128'],'explicit_actual'),
                (['--linearno-profile','official_release'],'official_release'),
                (['--linearno-profile','transolver_matched'],'transolver_matched')):
                jobs.append((task,'train',base('custom' if label=='custom' else 'p1_c3_r2_s1','sr_1_over_r')+extra,label))
        results=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(shell,*job) for job in jobs]
            for f in as_completed(futures):
                results.append(f.result())
                if len(results)%24==0:print('validated previews',len(results),'/',len(jobs),flush=True)
        train_paths=[r['plan']['run'] for r in results if r['plan']['action']=='train']
        assert len(train_paths)==len(set(train_paths))
        assert not any(Path(p).exists() for p in train_paths)
    failures=[]
    import torch
    for task in launch.TASKS:
        flags=base('p1_c3_r2_s1','sr_1_over_r')
        cases=[('missing_topology',['--linearno-loop','1','--linearno-loop-residual-mode','sr_1_over_r']),
               ('missing_mode',['--linearno-loop','1','--linearno-loop-topology','p1_c3_r2_s1']),
               ('loop_false',flags+['--linearno-loop','0']),('bad_mode',flags+['--linearno-loop-residual-mode','both']),
               ('other_family',flags+['--model','Transolver']),('A_flag',flags+['--linearno_latent_attnres','0']),
               ('K_flag',flags+['--linearno_history_k_conditioning','1']),('history_dropout',flags+['--linearno_attnres_history_dropout_p','0']),
               ('preset_custom',flags+['--linearno-loop-prefix-blocks','1']),('rank_conflict',flags+['--linearno-rank','64','--linearno-loop-rank-multiplier','2']),
               ('bad_multiplier',flags+['--linearno-loop-rank-multiplier','3']),('bool_rank',flags+['--linearno-rank','True']),
               ('nan_rank',flags+['--linearno-rank','NaN']),('old_depth',flags+['--n-layers','8']),
               ('incomplete_custom',base('custom','sr_1_over_r')+['--linearno-loop-core-blocks','2']),
               ('wrong_heads',flags+(['--n-hidden','7','--n-heads','2'] if task in launch.ENTRIES else ['--linearno-hidden','7','--linearno-heads','2']))]
        for field in ('prefix','core','repeats','suffix'):
            values=dict(prefix='0',core='2',repeats='3',suffix='1');values[field]='-1' if field=='prefix' else '0'
            cases.append(('invalid_'+field,base('custom','sr_1_over_r')+sum((['--linearno-loop-'+k+('-blocks' if k in ('prefix','core','suffix') else ''),v] for k,v in values.items()),[])))
        for label,tokens in cases:
            with contextlib.redirect_stderr(io.StringIO()),patch('torch.load',side_effect=AssertionError('early tensor load')):
                try:launch.plan(task,'train',tokens,ENV)
                except (SystemExit,ValueError):failures.append(dict(task=task,case=label,rejected_before_load=True))
                else:raise AssertionError((task,label))
        saved=next(r for r in archive if r['task']==task and r['preset']=='p1_c3_r2_s1' and r['mode']=='sr_1_over_r')
        existing=resume_runs[(task,'p1_c3_r2_s1','sr_1_over_r')]
        for action in ('eval','resume'):
            directory=next((Path(saved['artifact'])/'split').iterdir()) if action=='eval' else existing
            for label,extra in (('saved_mode',['--linearno-loop-residual-mode','rb_attnres']),('saved_topology',['--linearno-loop-topology','p2_c2_r2_s2']),
                               ('saved_rank',['--linearno-rank','128']),('saved_profile',['--linearno-profile','official_release']),('saved_seed',['--seed','999'])):
                with contextlib.redirect_stderr(io.StringIO()),patch('torch.load',side_effect=AssertionError('early tensor load')):
                    try:launch.plan(task,action,['--experiment-dir',str(directory),*extra],ENV)
                    except (ValueError,SystemExit):failures.append(dict(task=task,action=action,case=label,rejected_before_load=True))
                    else:raise AssertionError((task,action,label))
    value=dict(previews=len(results),matrix_train=48,matrix_eval=48,matrix_resume=48,extra_train=40,artifact_root=tmp,
        negative_checks=len(failures),new_run_paths_unique=True,created_run_directories=0,rows=results,negative_rows=failures,
        scope='No model/data/tensor loads. Eval/resume inspect actual LL7 small synthetic archives, not imaginary trained full-size models.')
    (OUT/'dry-run-matrix.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','negative_rows')}),flush=True)

if __name__=='__main__':main()
